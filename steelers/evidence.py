"""Append-only pregame evidence: projected/confirmed QBs, injuries and weather."""

import csv
import hashlib
import io
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from .forecast import kickoff_utc
from .features import feature_key
from .model import team_key
from .travel import VENUES
from .weather import summarize

AVAILABILITY_VERSION = "availability-v2"


def availability_complete(record, home, away):
    """Legacy/partial reports cannot establish a complete absence comparison."""
    if not record or record.get("version") != AVAILABILITY_VERSION:
        return False
    reports = record.get("team_reports", {})
    for team in (team_key(home), team_key(away)):
        report = reports.get(team, {})
        status = report.get("status")
        has_players = any(team_key(p["team"]) == team for p in record.get("players", []))
        if report.get("complete") is not True or status not in {"reported", "explicitly_empty"}:
            return False
        if (status == "reported") != has_players:
            return False
    return True


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Evidence timestamps must include a timezone")
    return result.astimezone(timezone.utc)


class EvidenceStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = self.connect()
        try:
            with db:
                db.execute("CREATE TABLE IF NOT EXISTS evidence (fingerprint TEXT PRIMARY KEY, kind TEXT, key TEXT, available TEXT, payload TEXT)")
        finally:
            db.close()

    def connect(self):
        return sqlite3.connect(self.path)

    def save(self, kind, key, available, payload):
        available = timestamp(available).isoformat()
        content = json.dumps(payload, sort_keys=True, allow_nan=False)
        fingerprint = hashlib.sha256(json.dumps([kind, key, available, content]).encode()).hexdigest()
        db = self.connect()
        try:
            with db:
                db.execute("INSERT OR IGNORE INTO evidence VALUES (?,?,?,?,?)", (fingerprint, kind, key, available, content))
        finally:
            db.close()

    def snapshot(self):
        db = self.connect()
        try:
            rows = db.execute("SELECT kind,key,available,payload,fingerprint FROM evidence ORDER BY available,rowid").fetchall()
        finally:
            db.close()
        indexed, digest = {}, hashlib.sha256()
        for kind, key, available, payload, fingerprint in rows:
            indexed.setdefault((kind, key), []).append({**json.loads(payload), "available_at": available})
            digest.update(fingerprint.encode())
        return Evidence(indexed, digest.hexdigest())

    def confirm(self, game, team, player_id, name, source, now=None):
        now = now or datetime.now(timezone.utc)
        kickoff = kickoff_utc(game)
        if game.completed or kickoff is None or now >= kickoff or team not in {game.home, game.away}:
            raise ValueError("A starter can only be confirmed for an upcoming game.")
        if not player_id or not name or not source.strip():
            raise ValueError("Choose a quarterback and describe the confirmation source.")
        self.save("confirmed_qb", f"{game.id}:{team_key(team)}", now.isoformat(),
                  {"id": player_id, "name": name, "source": source.strip()[:500], "status": "User-confirmed"})

    def import_depth(self, content):
        reader = csv.DictReader(io.StringIO(content))
        if not {"dt", "team", "gsis_id", "player_name", "pos_abb", "pos_rank"}.issubset(reader.fieldnames or []):
            raise ValueError("Timestamped depth charts (2025+) are required")
        snapshots = {}
        for row in reader:
            if row["pos_abb"] != "QB" or row["gsis_id"] in {"", "NA"}:
                continue
            when = timestamp(row["dt"]).isoformat()
            rank = int(row["pos_rank"])
            snapshots.setdefault((team_key(row["team"]), when), []).append(
                {"id": row["gsis_id"], "name": row["player_name"], "rank": rank})
        if not snapshots:
            raise ValueError("No timestamped quarterbacks found")
        for (team, when), players in snapshots.items():
            self.save("depth", team, when, {"players": sorted(players, key=lambda r: (r["rank"], r["id"])),
                                            "source": "nflverse / ESPN depth chart", "status": "Projected"})
        return len(snapshots)

    def capture(self, game, bundle, weather=None, now=None):
        now = now or datetime.now(timezone.utc)
        kickoff = kickoff_utc(game)
        if game.completed or kickoff is None or now >= kickoff:
            return
        metadata = next((s for s in bundle.get("sources", []) if s["file"] == f"injuries_{game.season}.csv"), {})
        retrieved = metadata.get("downloaded_at")
        if retrieved and timestamp(retrieved) <= now and now - timestamp(retrieved) <= timedelta(days=7):
            injuries = [{"team": team_key(r["team"]), "id": r.get("gsis_id", ""), "name": r.get("full_name", ""),
                         "position": r.get("position", ""), "status": r.get("report_status", "")}
                        for r in bundle.get("injuries", []) if team_key(r["team"]) in {team_key(game.home), team_key(game.away)}
                        and int(r["week"]) == game.week and r.get("game_type", "REG") == game.kind]
            # Snapshot time is when this app could use the record; provider dates
            # are retained separately and cannot backdate later-acquired evidence.
            reports = {}
            for raw_team in (game.home, game.away):
                team = team_key(raw_team)
                descriptor = bundle.get("injury_reports", {}).get(feature_key(game.season, game.week, game.kind, team), {})
                has_players = any(p["team"] == team for p in injuries)
                status = "reported" if has_players else "unknown"
                # The current player-row feed does not attest team-report
                # completeness. Only an explicit provider descriptor can do so.
                complete = descriptor.get("complete") is True and (
                    descriptor.get("status") == "reported" and has_players or
                    descriptor.get("status") == "explicitly_empty" and not has_players)
                if complete:
                    status = descriptor["status"]
                reports[team] = {"status": status, "complete": complete}
            payload = {"version": AVAILABILITY_VERSION, "players": injuries,
                       "team_reports": reports, "source": metadata}
            previous = self.snapshot().latest("availability", game.id, now)
            if previous is None or {k: v for k, v in previous.items() if k != "available_at"} != payload:
                self.save("availability", game.id, now.isoformat(), payload)
        if weather and weather.get("status") == "forecast" and timestamp(weather["retrieved_at"]) <= now:
            self.save("weather", game.id, weather["retrieved_at"], {**weather, "roof": game.roof, "kind": "issued_forecast"})

    def historical_weather(self, games, refresh=False):
        """Import previous-day forecasts, never observed historical conditions.

        Group requests by stadium and season. The latest valid-hour minus 24h
        is the conservative availability bound for the three-hour window.
        """
        grouped, errors, count = {}, [], 0
        existing = self.snapshot()
        for game in games:
            if game.season < 2024 or not game.completed or game.roof != "outdoors" or game.stadium_id not in VENUES or kickoff_utc(game) is None:
                continue
            if not refresh and existing.latest("weather", game.id, kickoff_utc(game)):
                continue
            grouped.setdefault((game.season, game.stadium_id), []).append(game)
        fields = ("temperature_2m", "wind_speed_10m", "wind_gusts_10m", "precipitation")
        for (season, venue), batch in grouped.items():
            lat, lon = VENUES[venue]
            query = urlencode({"latitude": lat, "longitude": lon,
                               "start_date": min(kickoff_utc(g).date() for g in batch).isoformat(),
                               "end_date": (max(kickoff_utc(g).date() for g in batch) + timedelta(days=1)).isoformat(),
                               "hourly": ",".join(f"{f}_previous_day1" for f in fields),
                               "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "precipitation_unit": "inch", "timezone": "UTC"})
            url = "https://previous-runs-api.open-meteo.com/v1/forecast?" + query
            try:
                with urlopen(url, timeout=30) as response:
                    raw = response.read(3_000_001)
                if len(raw) > 3_000_000:
                    raise ValueError("Weather history too large")
                payload = json.loads(raw)
                normalized = {"hourly": {"time": payload["hourly"]["time"],
                              **{f: payload["hourly"][f"{f}_previous_day1"] for f in fields}}}
                for game in batch:
                    kickoff = kickoff_utc(game)
                    try:
                        values = summarize(normalized, kickoff)
                    except (ValueError, KeyError, IndexError, TypeError):
                        continue
                    available = kickoff.replace(minute=0, second=0, microsecond=0) - timedelta(hours=22)
                    self.save("weather", game.id, available.isoformat(),
                              {**values, "status": "forecast", "roof": "outdoors", "kind": "previous_day1",
                               "source": "Open-Meteo Previous Runs", "source_url": url, "sha256": hashlib.sha256(raw).hexdigest(),
                               "retrieved_at": datetime.now(timezone.utc).isoformat(), "lead_hours": 24})
                    count += 1
            except (OSError, ValueError, KeyError, TypeError) as error:
                errors.append(f"{season}/{venue}: {error}")
        return {"imported": count, "errors": errors}


class Evidence:
    def __init__(self, indexed=None, digest=""):
        self.indexed, self.digest = indexed or {}, digest

    def latest(self, kind, key, cutoff, max_age=None):
        if cutoff is None:
            return None
        rows = self.indexed.get((kind, key), [])
        candidates = [r for r in rows if timestamp(r["available_at"]) < cutoff
                      and (max_age is None or cutoff - timestamp(r["available_at"]) <= max_age)]
        return candidates[-1] if candidates else None

    def quarterback_candidates(self, game, team, fallback, cutoff, estimator=None):
        """Ordered starter possibilities with probabilities summing to one.

        Evidence classes: ``confirmed`` (user confirmation), ``projected``
        (rank-one eligible depth entry) and ``previous`` (last leading passer).
        A later definitive Out/Inactive record invalidates an earlier
        confirmation; the conflict is reported, never silently erased. Depth
        membership rejects a departed previous passer; without membership the
        fallback is labeled unverified. ``estimator(class)`` may return earlier
        observed frequencies; otherwise the top candidate is deterministic and
        the alternatives are exposed with probability zero.
        """
        key = team_key(team)
        confirmed = self.latest("confirmed_qb", f"{game.id}:{key}", cutoff)
        depth = self.latest("depth", key, cutoff, timedelta(days=7))
        availability = self.latest("availability", game.id, cutoff, timedelta(days=7)) or {}
        unavailable = {r["id"] for r in availability.get("players", [])
                       if r["team"] == key and r["status"].lower() in {"out", "inactive"}}
        conflicts = []
        if confirmed and confirmed["id"] in unavailable and timestamp(availability["available_at"]) > timestamp(confirmed["available_at"]):
            conflicts.append({"kind": "confirmation_ruled_out", "id": confirmed["id"], "name": confirmed["name"],
                              "confirmed_at": confirmed["available_at"], "report_at": availability["available_at"]})
            confirmed = None
        listed = [p for p in depth["players"]] if depth else []
        eligible = [p for p in listed if p["id"] not in unavailable]
        membership = {p["id"] for p in listed} if depth else None

        def candidate(player, status, source, when, rank=None):
            return {"id": player["id"], "name": player["name"], "evidence_status": status, "source": source,
                    "evidence_time": when, "rank": rank, "probability": 0.}

        unknown = {"id": "", "name": "Unknown available starter", "evidence_status": "Unknown", "source": "",
                   "evidence_time": None, "rank": None, "probability": 0.}
        if confirmed:
            evidence_class, top = "confirmed", candidate(confirmed, "User-confirmed", confirmed.get("source", ""), confirmed["available_at"])
            alternatives = [candidate(p, "Projected", depth["source"], depth["available_at"], p["rank"]) for p in eligible if p["id"] != confirmed["id"]]
        elif eligible:
            evidence_class = "projected"
            top = candidate(eligible[0], "Projected", depth["source"], depth["available_at"], eligible[0]["rank"])
            alternatives = [candidate(p, "Projected", depth["source"], depth["available_at"], p["rank"]) for p in eligible[1:]]
        else:
            evidence_class, alternatives = "previous", []
            if fallback[0] in unavailable:
                top, unknown = None, {**unknown, "source": "Previous passer ruled out"}
            elif membership is not None and fallback[0] not in membership:
                top, unknown = None, {**unknown, "source": "Previous passer not on the current depth chart; all listed quarterbacks ruled out" if listed else "Previous passer not on the current depth chart"}
            else:
                status = "Previous passer" if membership is not None else "Previous passer (unverified)"
                top = {"id": fallback[0], "name": fallback[1] or "Unknown", "evidence_status": status, "source": "Prior game statistics",
                       "evidence_time": None, "rank": None, "probability": 0.}
        # Depth ranks resolve the learned categories; ties split equally.
        frequencies = estimator(evidence_class) if estimator and top is not None else None
        if top is None:
            candidates = [{**unknown, "probability": 1.}]
        elif frequencies is None:
            candidates = [{**top, "probability": 1.}, *alternatives, unknown]
        else:
            candidates = [{**top, "probability": frequencies["top"]}, *alternatives, unknown]
            groups = {"rank2": [c for c in alternatives if c["rank"] == 2],
                      "rank3": [c for c in alternatives if c["rank"] is not None and c["rank"] >= 3]}
            unresolved = frequencies["unknown"]
            for category, members in groups.items():
                if members:
                    for member in members:
                        member["probability"] += frequencies[category] / len(members)
                else:
                    unresolved += frequencies[category]
            candidates[-1]["probability"] = unresolved
            total = sum(c["probability"] for c in candidates)
            for member in candidates:
                member["probability"] /= total
        return {"class": evidence_class, "candidates": candidates, "conflicts": conflicts,
                "support": None if frequencies is None else frequencies.get("examples")}

    def quarterback(self, game, team, fallback, cutoff):
        """Compatibility view: the leading candidate in the earlier record shape."""
        return top_record(self.quarterback_candidates(game, team, fallback, cutoff))


def top_record(resolved):
    top = resolved["candidates"][0]
    status = {"User-confirmed": "User-confirmed", "Projected": "Projected", "Unknown": "Unknown"}.get(
        top["evidence_status"], "Previous passer")
    record = {"id": top["id"], "name": top["name"], "status": status, "source": top["source"]}
    if top["evidence_time"]:
        record["available_at"] = top["evidence_time"]
    if resolved["conflicts"]:
        record["conflicts"] = resolved["conflicts"]
    return record
