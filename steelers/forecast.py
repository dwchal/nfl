"""Immutable local pre-kickoff snapshots and an eventual forward evaluation."""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .model import metrics
from .provenance import SnapshotStore, canonical, identity, utc

MODELS = ("elo", "matchup", "advanced")
HORIZONS = {"24h": {"hours": 24, "tolerance_minutes": 120}, "1h": {"hours": 1, "tolerance_minutes": 30}}


def kickoff_utc(game):
    if not game.kickoff:
        return None
    try:
        local = datetime.fromisoformat(f"{game.day.isoformat()}T{game.kickoff}")
        return local.replace(tzinfo=ZoneInfo("America/New_York")).astimezone(timezone.utc)
    except ValueError:
        return None


class ForecastArchive:
    def __init__(self, path):
        self.path = Path(path)

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("CREATE TABLE IF NOT EXISTS forecasts (fingerprint TEXT PRIMARY KEY, game_id TEXT, team TEXT, season INTEGER, choice TEXT, created TEXT, kickoff TEXT, probability REAL, payload TEXT)")
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS model_artifacts (artifact_id TEXT PRIMARY KEY, family TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS capture_policies (policy_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS input_manifests (manifest_hash TEXT PRIMARY KEY, payload TEXT NOT NULL, snapshot_directory TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS collection_runs (
                collection_id TEXT PRIMARY KEY, game_id TEXT NOT NULL, season INTEGER NOT NULL,
                created_at TEXT NOT NULL, kickoff_at_capture TEXT NOT NULL, slot TEXT NOT NULL,
                manifest_hash TEXT NOT NULL, artifact_set TEXT NOT NULL, policy_id TEXT,
                FOREIGN KEY(policy_id) REFERENCES capture_policies(policy_id),
                FOREIGN KEY(manifest_hash) REFERENCES input_manifests(manifest_hash));
            CREATE INDEX IF NOT EXISTS collection_game_time ON collection_runs(game_id,created_at);
            CREATE TABLE IF NOT EXISTS forecast_runs (
                collection_id TEXT NOT NULL, game_id TEXT NOT NULL, family TEXT NOT NULL,
                created_at TEXT NOT NULL, kickoff_at_capture TEXT NOT NULL,
                home_probability REAL NOT NULL, artifact_id TEXT NOT NULL, manifest_hash TEXT NOT NULL,
                payload TEXT NOT NULL, PRIMARY KEY(collection_id,family),
                FOREIGN KEY(collection_id) REFERENCES collection_runs(collection_id),
                FOREIGN KEY(artifact_id) REFERENCES model_artifacts(artifact_id));
        """)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def register_artifacts(self, artifacts):
        with self.connect() as db:
            for family, artifact in artifacts.items():
                if identity(artifact["payload"]) != artifact["id"] or artifact["payload"]["family"] != family:
                    raise ValueError("Invalid model artifact")
                db.execute("INSERT OR IGNORE INTO model_artifacts VALUES (?,?,?)", (artifact["id"], family, canonical(artifact["payload"]).decode()))

    def register_manifest(self, identifier, manifest, snapshot_directory):
        if identity(manifest) != identifier or snapshot_directory is None:
            raise ValueError("An immutable, persisted input manifest is required")
        snapshots = SnapshotStore(snapshot_directory)
        if json.loads(snapshots.get(identifier)) != manifest:
            raise ValueError("Input manifest does not match its immutable snapshot")
        for source in manifest["sources"]:
            snapshots.get(source["sha256"])
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO input_manifests VALUES (?,?,?)", (identifier, canonical(manifest).decode(), str(Path(snapshot_directory).resolve())))

    def register_policy(self, payload):
        if set(payload) != {"version", "name", "artifacts", "horizons"} or payload["version"] != "capture-policy-v1" or payload["horizons"] != HORIZONS:
            raise ValueError("A frozen capture-policy-v1 definition with declared horizons is required")
        if not payload["name"] or set(payload["artifacts"]) != set(MODELS):
            raise ValueError("A policy must declare the three model families")
        with self.connect() as db:
            for family, identifiers in payload["artifacts"].items():
                if not isinstance(identifiers, list) or not identifiers or len(set(identifiers)) != len(identifiers):
                    raise ValueError("Policy artifact allowlists must be nonempty and unique")
                for identifier in identifiers:
                    row = db.execute("SELECT family FROM model_artifacts WHERE artifact_id=?", (identifier,)).fetchone()
                    if row != (family,):
                        raise ValueError("A policy references an unknown or mismatched model artifact")
            identifier = identity(payload)
            db.execute("INSERT OR IGNORE INTO capture_policies VALUES (?,?)", (identifier, canonical(payload).decode()))
        return identifier

    def save_collection(self, game, predictions, manifest_hash, created_at, policy_id=None):
        import math
        created_at = utc(created_at)
        kickoff = kickoff_utc(game)
        if game.completed or kickoff is None or not created_at < kickoff <= created_at + timedelta(days=7):
            raise ValueError("Collections require an unplayed game within seven days")
        if set(predictions) != set(MODELS):
            raise ValueError("A paired collection must include all three model choices, with explicit fallbacks")
        artifact_set = {family: p["artifact_id"] for family, p in predictions.items()}
        slot = created_at.replace(minute=created_at.minute // 15 * 15, second=0, microsecond=0).isoformat()
        identifier = identity([game.id, slot, manifest_hash, artifact_set, policy_id])
        with self.connect() as db:
            manifest = db.execute("SELECT payload FROM input_manifests WHERE manifest_hash=?", (manifest_hash,)).fetchone()
            if manifest is None:
                raise ValueError("The input manifest has not been archived")
            policy = None
            if policy_id:
                row = db.execute("SELECT payload FROM capture_policies WHERE policy_id=?", (policy_id,)).fetchone()
                if row is None:
                    raise ValueError("Unknown frozen policy")
                policy = json.loads(row[0])
            cutoffs = {p["as_of_utc"] for p in predictions.values()}
            if len(cutoffs) != 1 or utc(datetime.fromisoformat(next(iter(cutoffs)))) > created_at:
                raise ValueError("Paired predictions must share a cutoff at or before collection completion")
            if datetime.fromisoformat(json.loads(manifest[0])["available_at"]) > datetime.fromisoformat(next(iter(cutoffs))):
                raise ValueError("Inputs were retrieved after the forecast cutoff")
            for family, prediction in predictions.items():
                probability = prediction["home_probability"]
                if not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("A finite home probability is required")
                if prediction["input_manifest_hash"] != manifest_hash:
                    raise ValueError("Paired predictions must share one input manifest and clock")
                registered = db.execute("SELECT family FROM model_artifacts WHERE artifact_id=?", (prediction["artifact_id"],)).fetchone()
                if registered != (family,):
                    raise ValueError("A paired prediction references a missing model artifact")
                if policy and prediction["artifact_id"] not in policy["artifacts"][family]:
                    raise ValueError("Model artifact is not allowed by the frozen policy")
            added = db.execute("INSERT OR IGNORE INTO collection_runs VALUES (?,?,?,?,?,?,?,?,?)",
                               (identifier, game.id, game.season, created_at.isoformat(), kickoff.isoformat(), slot,
                                manifest_hash, canonical(artifact_set).decode(), policy_id)).rowcount
            if added:
                for family, p in predictions.items():
                    db.execute("INSERT INTO forecast_runs VALUES (?,?,?,?,?,?,?,?,?)",
                               (identifier, game.id, family, created_at.isoformat(), kickoff.isoformat(), p["home_probability"],
                                p["artifact_id"], manifest_hash, canonical(p).decode()))
        return {"collection_id": identifier, "inserted": bool(added)}

    def paired_report(self, games, season, policy_id=None):
        from .evaluation import paired_uncertainty
        from .context_research import pick_comparison
        completed = sorted({g.id: g for g in games if g.season == season and g.kind == "REG" and g.completed}.values(), key=lambda g: (g.day, g.kickoff, g.id))
        report = {"label": "Paired prospective forecasts at declared horizons", "season": season, "policy_id": policy_id,
                  "pooling": "Frozen policy allowlist" if policy_id else "Exact artifact set; different artifacts are never pooled",
                  "horizons": {}, "legacy_label": "Legacy dashboard-use forecasts; unknown artifact/horizon"}
        with self.connect() as db:
            if policy_id and db.execute("SELECT 1 FROM capture_policies WHERE policy_id=?", (policy_id,)).fetchone() is None:
                raise ValueError("Unknown frozen policy")
            report["legacy_records"] = db.execute("SELECT count(*) FROM forecasts WHERE season=?", (season,)).fetchone()[0]
            for horizon, settings in HORIZONS.items():
                groups, gaps = {}, []
                for game in completed:
                    kickoff = kickoff_utc(game)
                    if kickoff is None:
                        gaps.append({"id": game.id, "reason": "Unknown kickoff"})
                        continue
                    deadline = kickoff - timedelta(hours=settings["hours"])
                    earliest = deadline - timedelta(minutes=settings["tolerance_minutes"])
                    collections = db.execute("SELECT collection_id,created_at,kickoff_at_capture,artifact_set,policy_id,manifest_hash FROM collection_runs WHERE game_id=? ORDER BY created_at DESC,collection_id", (game.id,)).fetchall()
                    selected, stale = None, False
                    for collection_id, created, captured_kickoff, artifact_set, collection_policy, manifest_hash in collections:
                        when = datetime.fromisoformat(created)
                        if when > deadline or when >= min(kickoff, datetime.fromisoformat(captured_kickoff)) or (policy_id and collection_policy != policy_id):
                            continue
                        if db.execute("SELECT 1 FROM input_manifests WHERE manifest_hash=?", (manifest_hash,)).fetchone() is None:
                            continue
                        rows = db.execute("SELECT f.family,f.home_probability,f.artifact_id,f.payload FROM forecast_runs f JOIN model_artifacts a ON a.artifact_id=f.artifact_id AND a.family=f.family WHERE collection_id=?", (collection_id,)).fetchall()
                        if {r[0] for r in rows} != set(MODELS) or {r[0]: r[2] for r in rows} != json.loads(artifact_set):
                            continue
                        if when < earliest:
                            stale = True
                            break
                        selected = (collection_id, created, artifact_set, rows, manifest_hash)
                        break
                    if selected is None:
                        gaps.append({"id": game.id, "deadline": deadline.isoformat(),
                                     "reason": "Latest paired capture is stale" if stale else "No paired capture at or before the deadline"})
                        continue
                    collection_id, created, artifact_set, rows, manifest_hash = selected
                    group_key = policy_id or identity(json.loads(artifact_set))
                    group = groups.setdefault(group_key, {"group_id": group_key, "artifacts": {m: set() for m in MODELS}, "games": []})
                    for family, _, artifact_id, _ in rows:
                        group["artifacts"][family].add(artifact_id)
                    result = float(game.home_score > game.away_score) if game.home_score != game.away_score else .5
                    group["games"].append({"id": game.id, "season": game.season, "week": game.week, "home": game.home, "away": game.away,
                                           "result": result, "collection_id": collection_id, "created_at": created, "manifest_hash": manifest_hash,
                                           "fallback_reasons": {r[0]: json.loads(r[3]).get("fallback_reasons", []) for r in rows},
                                           "probabilities": {r[0]: r[1] for r in rows}, "artifact_ids": {r[0]: r[2] for r in rows}})
                for group in groups.values():
                    group["artifacts"] = {m: sorted(ids) for m, ids in group["artifacts"].items()}
                    forecasts = {m: [{**r, "probability": r["probabilities"][m]} for r in group["games"]] for m in MODELS}
                    group["models"] = {m: {"metrics": metrics(rows), "by_team": {t: metrics(rows, t) for t in ("PIT", "MIN")},
                                           "fallback_games": sum(bool(r["fallback_reasons"][m]) for r in rows),
                                           "uncertainty": paired_uncertainty(forecasts["elo"], rows),
                                           "winner_changes": pick_comparison(forecasts["elo"], rows) if any(r["result"] != .5 for r in rows) else None}
                                       for m, rows in forecasts.items()}
                report["horizons"][horizon] = {**settings, "eligible_games": len(completed), "paired_games": sum(len(g["games"]) for g in groups.values()),
                                                "gaps": gaps, "groups": list(groups.values())}
        return report

    def save(self, game, team, choice, probability, payload, now=None):
        now = now or datetime.now(timezone.utc)
        kickoff = kickoff_utc(game)
        if game.completed or kickoff is None or now >= kickoff or (kickoff - now).total_seconds() > 7 * 86400:
            return False
        serialized = json.dumps(payload, sort_keys=True, allow_nan=False)
        fingerprint = hashlib.sha256(json.dumps([game.id, team, choice, probability, serialized], sort_keys=True).encode()).hexdigest()
        with self.connect() as connection:
            connection.execute("INSERT OR IGNORE INTO forecasts VALUES (?,?,?,?,?,?,?,?,?)",
                               (fingerprint, game.id, team, game.season, choice, now.isoformat(), kickoff.isoformat(), probability, serialized))
        return True

    def report(self, games, team, season, choice="auto"):
        by_id = {g.id: g for g in games}
        with self.connect() as connection:
            saved = connection.execute("SELECT game_id, created, probability FROM forecasts WHERE team=? AND season=? AND choice=? ORDER BY created", (team, season, choice)).fetchall()
        latest = {}
        for identifier, created, probability in saved:
            game = by_id.get(identifier)
            kickoff = kickoff_utc(game) if game else None
            # Re-check against revised kickoff times; a reschedule cannot turn a
            # post-kickoff forecast into valid evidence for the actual game.
            if kickoff and datetime.fromisoformat(created) < kickoff:
                latest[identifier] = (created, probability)
        evaluated = []
        for identifier, (created, probability) in latest.items():
            game = by_id[identifier]
            if not game.completed:
                continue
            result = float(game.home_score > game.away_score) if game.home_score != game.away_score else .5
            evaluated.append({"home": game.home, "away": game.away, "result": result,
                              "probability": probability if game.home == team else 1 - probability})
        return {"label": "Legacy dashboard-use forecasts; mixed lead times", "snapshots": len(saved), "games": len(latest), "evaluated": metrics(evaluated),
                "latest_saved": saved[-1][1] if saved else None,
                "policy": "Latest saved forecast before kickoff per game and model choice. Scenarios are excluded. Snapshots are saved only while this app is used, within seven days of kickoff."}
