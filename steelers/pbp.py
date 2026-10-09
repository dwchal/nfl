"""Compact, validated situational play-by-play aggregates (stdlib only)."""

import csv
import gzip
import hashlib
import io
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

from .model import team_key

VERSION = "situations-v1"
LABELS = ("Pass EPA", "Pass success", "Rush EPA", "Rush success", "Sack rate",
          "Early-down EPA", "Third/fourth-and-long success", "Pass reliance")
SCALES = (.15, .1, .15, .1, .04, .15, .15, .15)


def number(row, name):
    value = row.get(name)
    if value in (None, "", "NA", "NaN", "nan"):
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite play statistic: {name}")
    return value


def aggregate(rows, season):
    totals = defaultdict(lambda: [[0., 0.] for _ in LABELS])
    seen, included = set(), 0
    for row in rows:
        identifier = row["game_id"]
        if not identifier.startswith(f"{season}_"):
            raise ValueError("Wrong season in play-by-play file")
        play_id = (identifier, row["play_id"])
        if play_id in seen:
            raise ValueError("Duplicate play ID")
        seen.add(play_id)
        if row.get("season_type") not in {"REG", "POST"}:
            continue
        if any(number(row, k) == 1 for k in ("qb_kneel", "qb_spike", "no_play")):
            continue
        passing = number(row, "pass_attempt") == 1 or number(row, "sack") == 1 or number(row, "qb_scramble") == 1
        rushing = number(row, "rush_attempt") == 1 and not passing
        if not (passing or rushing) or row.get("play_type") == "no_play":
            continue
        epa, down, distance = (number(row, k) for k in ("epa", "down", "ydstogo"))
        remaining, margin = number(row, "game_seconds_remaining"), number(row, "score_differential")
        if epa is None or down not in (1, 2, 3, 4) or distance is None or not row.get("posteam"):
            continue
        # Exclude late blowout situations using the pre-play score, never final score.
        if remaining is not None and margin is not None and 0 <= remaining <= 900 and abs(margin) > 16:
            continue
        target = totals[(identifier, team_key(row["posteam"]))]
        observations = {7: float(passing)}
        if passing:
            observations.update({0: epa, 1: float(epa > 0), 4: float(number(row, "sack") == 1)})
        else:
            observations.update({2: epa, 3: float(epa > 0)})
        if down <= 2:
            observations[5] = epa
        if down >= 3 and distance >= 7:
            observations[6] = float(epa > 0)
        for index, value in observations.items():
            target[index][0] += value
            target[index][1] += 1
        included += 1
    if not totals or included < 1:
        raise ValueError("No valid competitive plays")
    return {"version": VERSION, "season": season, "plays": included,
            "games": {identifier: {team: values for (game, team), values in totals.items() if game == identifier}
                      for identifier in sorted({game for game, _ in totals})}}


def validate(payload, season):
    if payload.get("version") != VERSION or payload.get("season") != season or not payload.get("games"):
        raise ValueError("Invalid play-by-play aggregate")
    for identifier, teams in payload["games"].items():
        if not identifier.startswith(f"{season}_"):
            raise ValueError("Wrong aggregate season")
        for values in teams.values():
            if len(values) != len(LABELS):
                raise ValueError("Wrong aggregate width")
            for pair in values:
                if len(pair) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in pair) or pair[1] < 0:
                    raise ValueError("Invalid aggregate values")
    return payload


class PBPStore:
    def __init__(self, directory, offline=False):
        self.directory, self.offline = Path(directory), offline

    def load(self, season, refresh=False):
        path = self.directory / f"pbp_{season}.json"
        cached = None
        try:
            cached = validate(json.loads(path.read_text()), season)
        except (OSError, ValueError, TypeError, KeyError):
            pass
        if cached and not refresh:
            return cached
        if self.offline:
            return cached
        self.directory.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
        try:
            digest = hashlib.sha256()
            with tempfile.TemporaryFile() as compressed:
                with urlopen(url, timeout=45) as response:
                    size = 0
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > 100_000_000:
                            raise ValueError("Play-by-play download too large")
                        digest.update(chunk)
                        compressed.write(chunk)
                compressed.seek(0)
                with gzip.GzipFile(fileobj=compressed) as unpacked, io.TextIOWrapper(unpacked) as stream:
                    payload = aggregate(csv.DictReader(stream), season)
            payload.update(source_url=url, source_sha256=digest.hexdigest(), retrieved_at=datetime.now(timezone.utc).isoformat())
            validate(payload, season)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, allow_nan=False))
            os.replace(temporary, path)
            return payload
        except (OSError, ValueError, KeyError, EOFError, csv.Error) as error:
            if cached:
                return {**cached, "warning": f"Saved play-by-play used: {error}"}
            return {"games": {}, "season": season, "warning": f"Play-by-play unavailable: {error}"}
