"""Offline, chronological ablations of schedule context and granular team stats.

Research only: this command never changes the dashboard's deployed model.
"""

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from itertools import groupby
from pathlib import Path

from .data import parse_games
from .evaluation import paired_uncertainty
from .model import backtest, metrics, select_model, sigmoid, team_key

CONTEXT = ("Home field", "Permanent dome", "Retractable roof", "Outdoor stadium",
           "Early kickoff ET", "Evening kickoff ET", "Thursday", "Monday", "Saturday")
SIGNALS = ("Net passing yards/dropback", "Passing first downs/dropback", "Sacks/dropback",
           "Rushing first downs/carry", "Turnovers/play", "EPA/play")
LABELS = CONTEXT + tuple(f"{side}: {signal}" for side in ("Offense", "Defense") for signal in SIGNALS)
GROUPS = {"Home-field recalibration": (0,), "Stadium type": (0, 1, 2, 3), "Kickoff time": (0, 4, 5),
          "Day of week": (0, 6, 7, 8), "All schedule context": tuple(range(9)),
          "Team efficiency": tuple(range(9, 21)), "Schedule + efficiency": tuple(range(21))}
SCALES = (1.5, .08, .04, .08, .02, .15)
REQUIRED = {"season", "week", "season_type", "team", "attempts", "sacks_suffered",
            "passing_yards", "sack_yards_lost", "passing_first_downs", "carries",
            "rushing_first_downs", "passing_interceptions", "fumbles_lost_total",
            "passing_epa", "rushing_epa"}


def load_stats(directory, years):
    stats, sources = {}, []
    for year in years:
        path = Path(directory) / f"stats_team_week_{year}.csv"
        raw = path.read_bytes()
        reader = csv.DictReader(raw.decode("utf-8-sig").splitlines())
        if not REQUIRED.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing team-stat columns in {path.name}")
        sources.append({"file": path.name, "sha256": hashlib.sha256(raw).hexdigest()})
        for row in reader:
            if int(row["season"]) != year:
                raise ValueError(f"Wrong season in {path.name}")
            key = (year, int(row["week"]), row["season_type"], team_key(row["team"]))
            if key in stats:
                raise ValueError(f"Duplicate team/week in {path.name}: {key}")
            numeric = {}
            for field in REQUIRED - {"season", "week", "season_type", "team"}:
                numeric[field] = float(row[field])
                if not math.isfinite(numeric[field]):
                    raise ValueError(f"Missing/nonfinite {field} in {path.name}")
            db = numeric["attempts"] + numeric["sacks_suffered"]
            carries = numeric["carries"]
            if db <= 0 or carries <= 0:
                raise ValueError(f"Invalid play counts in {path.name}")
            totals = (numeric["passing_yards"] - numeric["sack_yards_lost"],
                      numeric["passing_first_downs"], numeric["sacks_suffered"],
                      numeric["rushing_first_downs"],
                      numeric["passing_interceptions"] + numeric["fumbles_lost_total"],
                      numeric["passing_epa"] + numeric["rushing_epa"])
            stats[key] = tuple(zip(totals, (db, db, db, carries, db + carries, db + carries)))
    return stats, sources


def roof_type(roof):
    # Actual open/closed status may only be recorded after kickoff. Treat both
    # as the same building type, learned from earlier games at that venue.
    return "retractable" if roof in {"open", "closed", "retractable"} else roof


def schedule_features(game, venues):
    home = float(not game.neutral)
    roof = venues.get(game.stadium_id or game.stadium, "")
    try:
        hour = int(game.kickoff.split(":")[0])
    except (ValueError, IndexError):
        hour = None
    weekday = game.day.weekday()
    # These estimate differences in home advantage, not benefits for both teams.
    return [home, home * (roof == "dome"), home * (roof == "retractable"),
            home * (roof == "outdoors"), home * (hour is not None and hour <= 13),
            home * (hour is not None and hour >= 19), home * (weekday == 3),
            home * (weekday == 0), home * (weekday == 5)]


class EfficiencyState:
    def __init__(self):
        self.teams = {}
        self.league = [[0., 0.] for _ in SIGNALS]
        self.venues = {}

    def strength(self, team):
        values = self.teams.get(team, [[0., 0.] for _ in range(12)])
        return [total / (count + (200 if i % 6 >= 4 else 100)) for i, (total, count) in enumerate(values)]

    def features(self, game):
        h, a = self.strength(team_key(game.home)), self.strength(team_key(game.away))
        offense = [(h[i] - a[i]) / SCALES[i] for i in range(6)]
        defense = [(a[i + 6] - h[i + 6]) / SCALES[i] for i in range(6)]
        return schedule_features(game, self.venues) + offense + defense

    def offseason(self):
        self.teams = {team: [[total * .65, count * .65] for total, count in values]
                      for team, values in self.teams.items()}

    def observe(self, game, stats):
        venue = game.stadium_id or game.stadium
        if venue and game.roof:
            self.venues[venue] = roof_type(game.roof)
        kind = "REG" if game.kind == "REG" else "POST"
        home, away = team_key(game.home), team_key(game.away)
        own = stats.get((game.season, game.week, kind, home))
        other = stats.get((game.season, game.week, kind, away))
        if own is None or other is None:
            return False
        strengths = {home: self.strength(home), away: self.strength(away)}
        means = [total / max(1, count) for total, count in self.league]
        for team, opponent, offense, defense in ((home, away, own, other), (away, home, other, own)):
            previous = self.teams.get(team, [[0., 0.] for _ in range(12)])
            observations = []
            for i, (total, count) in enumerate(offense + defense):
                j = i % 6
                opponent_strength = strengths[opponent][j + 6 if i < 6 else j]
                observations.append([.9 * previous[i][0] + total - count * (means[j] + opponent_strength),
                                     .9 * previous[i][1] + count])
            self.teams[team] = observations
        for values in (own, other):
            for i, (total, count) in enumerate(values):
                self.league[i][0] += total
                self.league[i][1] += count
        return True


def feature_history(games, stats):
    state, rows, previous_year = EfficiencyState(), {}, None
    for _, same_day in groupby(sorted(games, key=lambda g: (g.day, g.id)), key=lambda g: g.day):
        batch = list(same_day)
        if previous_year is not None and previous_year != batch[0].season:
            state.offseason()
        previous_year = batch[0].season
        for game in batch:
            # Same-day and own-game statistics cannot inform the forecast.
            rows[game.id] = state.features(game)
        for game in batch:
            if game.completed:
                state.observe(game, stats)
    return rows


def fit(rows, indices, penalty=.1):
    weights = [0.] * len(indices)
    inputs = [(r["offset"], [max(-4, min(4, r["features"][i])) for i in indices], r["result"]) for r in rows]
    for _ in range(200):
        gradient = [0.] * len(indices)
        for offset, features, outcome in inputs:
            error = sigmoid(offset + sum(w * x for w, x in zip(weights, features))) - outcome
            for i, value in enumerate(features):
                gradient[i] += error * value
        steps = [.5 * (g / max(1, len(inputs)) + penalty * w) for g, w in zip(gradient, weights)]
        weights = [w - s for w, s in zip(weights, steps)]
        if max(abs(s) for s in steps) < 1e-7:
            break
    return weights


def pick_comparison(baseline, candidate, samples=2000):
    """Report changed winner picks and a paired season/week bootstrap interval."""
    if not baseline or [r["id"] for r in baseline] != [r["id"] for r in candidate]:
        raise ValueError("Winner comparison requires the same ordered games.")
    blocks, gained, lost = defaultdict(lambda: [0, 0]), 0, 0
    for old, new in zip(baseline, candidate):
        if old["result"] != new["result"]:
            raise ValueError("Winner comparison outcomes must match.")
        if old["result"] == .5:
            continue
        old_correct = (old["probability"] >= .5) == (old["result"] == 1)
        new_correct = (new["probability"] >= .5) == (new["result"] == 1)
        change = int(new_correct) - int(old_correct)
        gained += change == 1
        lost += change == -1
        block = blocks[(old["season"], old["week"])]
        block[0] += 1
        block[1] += change
    values, rng, draws = list(blocks.values()), random.Random(42), []
    if not values or samples < 1:
        raise ValueError("Decisive games and at least one bootstrap draw are required.")
    for _ in range(samples):
        chosen = [rng.choice(values) for _ in values]
        draws.append(sum(b[1] for b in chosen) / sum(b[0] for b in chosen))
    draws.sort()
    count = sum(b[0] for b in values)
    return {"decisive_games": count, "corrected_picks": gained, "new_errors": lost,
            "net_correct_picks": gained - lost, "accuracy_delta": (gained - lost) / count,
            "interval_95": [draws[int(.025 * samples)], draws[min(samples - 1, int(.975 * samples))]],
            "method": "Paired season/week block bootstrap", "samples": samples}


def experiment(games, stats, start=2021, end=2025, history_start=2016):
    features = feature_history([g for g in games if history_start <= g.season <= end], stats)
    forecasts = []
    for year in range(history_start + 1, end + 1):
        config, _ = select_model([g for g in games if g.season < year], year)
        for row in backtest([g for g in games if g.season <= year], [year], config):
            p = max(1e-9, min(1 - 1e-9, row["probability"]))
            forecasts.append({**row, "offset": math.log(p / (1 - p)), "features": features[row["id"]]})
    baseline = [r for r in forecasts if start <= r["season"] <= end]
    results = {}
    for name, indices in GROUPS.items():
        predictions, folds = [], []
        for year in range(start, end + 1):
            training = [r for r in forecasts if max(history_start + 1, year - 6) <= r["season"] < year]
            target = [r for r in baseline if r["season"] == year]
            if len(training) < 500 or len(target) < 200:
                raise ValueError(f"Insufficient complete history for {year}")
            weights = fit(training, indices)
            predicted = [{**r, "probability": sigmoid(r["offset"] + sum(w * max(-4, min(4, r["features"][i]))
                                                                                      for w, i in zip(weights, indices)))} for r in target]
            predictions.extend(predicted)
            folds.append({"season": year, "training_seasons": sorted({r["season"] for r in training}),
                          "weights": dict(zip((LABELS[i] for i in indices), weights)),
                          "baseline": metrics(target), "candidate": metrics(predicted)})
        results[name] = {"metrics": metrics(predictions), "by_season": folds,
                         "by_team": {t: {"baseline": metrics(baseline, t), "candidate": metrics(predictions, t)} for t in ("PIT", "MIN")},
                         "uncertainty": paired_uncertainty(baseline, predictions),
                         "winner_picks": pick_comparison(baseline, predictions)}
    return {"protocol": "Each season refits on up to six completed earlier seasons; each historical Elo offset was itself selected before its season. Fixed L2 penalty 0.1. Team statistics and venue types become available the following day. No production model is selected by this experiment.",
            "limitations": "Retrospective exploration of seven feature groups, not an untouched/live audit. Final schedule revisions and corrected statistics may differ from information available earlier. Kickoff is Eastern time, not team body-clock time. Actual weather and current-game roof position are excluded. No confirmed starting-QB or injury information is used.",
            "baseline": metrics(baseline), "experiments": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = args.data.read_bytes()
    games = parse_games(raw.decode("utf-8-sig"))
    for year in range(2016, 2026):
        season = [g for g in games if g.season == year and g.kind == "REG"]
        if len(season) < 200 or not all(g.completed for g in season):
            parser.error(f"Complete 2016–2025 schedule required; {year} is incomplete")
    stats, sources = load_stats(args.features, range(2016, 2026))
    report = {"schedule_sha256": hashlib.sha256(raw).hexdigest(), "sources": sources, **experiment(games, stats)}
    content = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(content)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
