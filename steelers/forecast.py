"""Immutable local pre-kickoff snapshots and an eventual forward evaluation."""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .model import metrics


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
        connection.execute("CREATE TABLE IF NOT EXISTS forecasts (fingerprint TEXT PRIMARY KEY, game_id TEXT, team TEXT, season INTEGER, choice TEXT, created TEXT, kickoff TEXT, probability REAL, payload TEXT)")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

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
        return {"snapshots": len(saved), "games": len(latest), "evaluated": metrics(evaluated),
                "latest_saved": saved[-1][1] if saved else None,
                "policy": "Latest saved forecast before kickoff per game and model choice. Scenarios are excluded. Snapshots are saved only while this app is used, within seven days of kickoff."}
