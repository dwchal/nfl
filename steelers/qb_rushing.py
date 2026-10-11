"""Separate, versioned QB-rushing inputs and chronological research features.

Prepare explicitly; the dashboard and scheduled collector never download these
files. Annual roster positions classify players, not historical team membership.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.request import urlopen

from .matchup import FeatureState
from .model import team_key
from .pbp import number

VERSION = "qb-rushing-v1"
LABELS = ("Pass offense", "Pass defense", "Rush offense", "Rush defense", "Rest", "QB passing replacement", "QB rushing replacement")


def aggregate(rows, season, positions):
    games, seen, excluded = {}, set(), 0
    for row in rows:
        identifier = row["game_id"]
        if not identifier.startswith(f"{season}_"):
            raise ValueError("Wrong QB-rushing season")
        key = (identifier, row["play_id"])
        if key in seen:
            raise ValueError("Duplicate rushing play")
        seen.add(key)
        if row.get("season_type") not in {"REG", "POST"} or not row.get("posteam"):
            continue
        team = team_key(row["posteam"])
        entry = games.setdefault(identifier, {}).setdefault(team, {"players": {}, "unknown_runs": 0})
        if any(number(row, k) == 1 for k in ("qb_kneel", "qb_spike", "no_play")) or row.get("play_type") == "no_play":
            continue
        if number(row, "rush_attempt") != 1:
            continue
        player = row.get("rusher_player_id", "")
        if player not in positions:
            entry["unknown_runs"] += 1
            excluded += 1
            continue
        if positions[player] != "QB":
            continue
        epa = number(row, "epa")
        if epa is None:
            entry["unknown_runs"] += 1
            excluded += 1
            continue
        value = entry["players"].setdefault(player, [0., 0])
        value[0] += epa
        value[1] += 1
    if not games:
        raise ValueError("No QB-rushing source games")
    return validate({"version": VERSION, "season": season, "games": games, "excluded_runs": excluded}, season)


def validate(payload, season):
    if payload.get("version") != VERSION or payload.get("season") != season or not payload.get("games"):
        raise ValueError("Invalid QB-rushing cache")
    for identifier, teams in payload["games"].items():
        if not identifier.startswith(f"{season}_"):
            raise ValueError("Wrong QB-rushing game season")
        for entry in teams.values():
            if type(entry.get("unknown_runs")) is not int or entry["unknown_runs"] < 0:
                raise ValueError("Invalid unknown rushing count")
            for player, pair in entry["players"].items():
                if not player or len(pair) != 2 or not all(type(v) in (int, float) and math.isfinite(v) for v in pair) or pair[1] < 0 or pair[1] != int(pair[1]) or pair[1] == 0 and pair[0] != 0:
                    raise ValueError("Invalid QB-rushing aggregate")
    return payload


class RushingState(FeatureState):
    def __init__(self, rushing):
        super().__init__()
        self.rushing = rushing or {}
        self.rush_qbs, self.rush_teams, self.rush_games = {}, {}, {}
        self.rush_league = [0., 0.]

    def features(self, game, home_qb=None, away_qb=None):
        replacement = []
        for raw, selected in ((game.home, home_qb), (game.away, away_qb)):
            team = team_key(raw)
            identifier = selected if selected is not None else self.last_qb.get(team, ("", ""))[0]
            q, t = self.rush_qbs.get(identifier, (0., 0.)), self.rush_teams.get(team, (0., 0.))
            replacement.append(q[0] / (q[1] + 50) - t[0] / (t[1] + 50) if identifier else 0.)
        return [*super().features(game, home_qb, away_qb), (replacement[0] - replacement[1]) / .15]

    def coverage(self, game, home_qb=None, away_qb=None):
        result = super().coverage(game, home_qb, away_qb)
        for team, values in result["teams"].items():
            count = self.rush_teams.get(team, (0., 0.))[1]
            values["qb_rushing"] = {"games": self.rush_games.get(team, 0), "effective_carries": count,
                                     "neutral_prior": count == 0}
        return result

    def offseason(self):
        super().offseason()
        self.rush_qbs = {q: (epa * .8, count * .8) for q, (epa, count) in self.rush_qbs.items()}
        self.rush_teams = {t: (epa * .65, count * .65) for t, (epa, count) in self.rush_teams.items()}

    def observe(self, game, bundle):
        mean = self.rush_league[0] / max(1, self.rush_league[1])
        observations = []
        for raw in (game.home, game.away):
            team = team_key(raw)
            entry = self.rushing.get(game.id, {}).get(team)
            if entry is None or entry["unknown_runs"]:
                continue
            total, count = 0., 0
            for identifier, (epa, carries) in entry["players"].items():
                old = self.rush_qbs.get(identifier, (0., 0.))
                self.rush_qbs[identifier] = (old[0] * .95 + epa - mean * carries, old[1] * .95 + carries)
                total += epa
                count += carries
            old = self.rush_teams.get(team, (0., 0.))
            self.rush_teams[team] = (old[0] * .9 + total - mean * count, old[1] * .9 + count)
            self.rush_games[team] = self.rush_games.get(team, 0) + 1
            observations.append((total, count))
        for total, count in observations:
            self.rush_league[0] += total
            self.rush_league[1] += count
        super().observe(game, bundle)


def prepare(directory, start, end):
    directory.mkdir(parents=True, exist_ok=True)
    for year in range(start, end + 1):
        output = directory / f"qb_rushing_{year}.json"
        if output.exists():
            validate(json.loads(output.read_text()), year)
            print(f"{year}: valid saved aggregate", flush=True)
            continue
        paths = {}
        for name, url in {
            "roster": f"https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{year}.csv",
            "pbp": f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.csv.gz",
        }.items():
            path = directory / f"{name}_{year}.{'csv.gz' if name == 'pbp' else 'csv'}"
            if not path.exists():
                temporary = path.with_suffix(path.suffix + ".tmp")
                try:
                    with urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
                        size = 0
                        while chunk := response.read(1024 * 1024):
                            size += len(chunk)
                            if size > 100_000_000:
                                raise ValueError("Source download too large")
                            stream.write(chunk)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            paths[name] = (path, url)
        with paths["roster"][0].open() as stream:
            positions = {}
            for row in csv.DictReader(stream):
                identifier, position = row.get("gsis_id"), row.get("position")
                if identifier and position:
                    if identifier in positions and positions[identifier] != position:
                        positions[identifier] = "AMBIGUOUS"
                    else:
                        positions[identifier] = position
        # Ambiguous classification is missing, never a verified non-QB.
        positions = {q: p for q, p in positions.items() if p != "AMBIGUOUS"}
        with gzip.open(paths["pbp"][0], "rt") as stream:
            payload = aggregate(csv.DictReader(stream), year, positions)
        payload["sources"] = {name: {"url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                              for name, (path, url) in paths.items()}
        payload["position_policy"] = "Annual roster position classification only; not starter or team-membership evidence. Reconstructed history."
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, output)
        print(f"{year}: {len(payload['games'])} games; {payload['excluded_runs']} unclassified/missing-EPA runs", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(".cache/qb-rushing"))
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()
    if not 1999 <= args.start <= args.end <= 2100:
        parser.error("Require 1999 <= start <= end <= 2100")
    prepare(args.directory, args.start, args.end)


if __name__ == "__main__":
    main()
