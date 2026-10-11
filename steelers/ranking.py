"""Opponent-adjusted scoring-margin research; never promotes a live model.

Run with --help. All fits use scores from strictly earlier calendar dates.
"""

import argparse
import hashlib
import json
import math
from itertools import groupby
from pathlib import Path

from .data import parse_games
from .evaluation import paired_uncertainty, auc_uncertainty
from .model import chronological_forecasts, metrics, team_key
from .optimization import sigmoid

VERSION = "margin-ranking-v1"


def validate_config(config):
    if set(config) != {"version", "history_start", "evaluation_seasons", "bootstrap_samples", "candidates"} or config["version"] != VERSION:
        raise ValueError("Unsupported ranking protocol")
    seasons = config["evaluation_seasons"]
    if (not isinstance(seasons, list) or not seasons or any(type(y) is not int for y in seasons)
            or seasons != sorted(set(seasons)) or type(config["history_start"]) is not int
            or not 1999 <= config["history_start"] <= min(seasons) - 5):
        raise ValueError("Ordered unique seasons and at least five warmup years are required")
    if type(config["bootstrap_samples"]) is not int or config["bootstrap_samples"] < 1:
        raise ValueError("Positive integer bootstrap_samples required")
    if not isinstance(config["candidates"], list) or not config["candidates"]:
        raise ValueError("At least one candidate is required")
    seen = {"elo"}
    for candidate in config["candidates"]:
        if set(candidate) != {"id", "half_life_weeks", "ridge", "margin_cap"}:
            raise ValueError("Unknown candidate fields")
        key = candidate["id"]
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("Candidate identifiers must be unique and cannot be elo")
        seen.add(key)
        if any(type(candidate[k]) not in (int, float) or not math.isfinite(candidate[k]) or candidate[k] <= 0
               for k in ("half_life_weeks", "ridge", "margin_cap")):
            raise ValueError("Candidate parameters must be finite and positive")
    return config


def solve(matrix, target):
    """Solve a positive-definite ridge system with Cholesky decomposition."""
    n = len(target)
    lower = [[0.] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            value = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            lower[i][j] = math.sqrt(value) if i == j else value / lower[j][j]
    forward = []
    for i in range(n):
        forward.append((target[i] - sum(lower[i][j] * forward[j] for j in range(i))) / lower[i][i])
    result = [0.] * n
    for i in reversed(range(n)):
        result[i] = (forward[i] - sum(lower[j][i] * result[j] for j in range(i + 1, n))) / lower[i][i]
    return result


def margin_replay(games, candidate, history_start):
    """Recency-weighted ridge ratings with a jointly fitted home advantage.

    Ridge shrinks team strength to zero and home advantage to 2 points.
    Each offseason retains half the accumulated evidence. In-season evidence
    decays with calendar weeks. All games on one date share the same fit.
    """
    teams = sorted({team_key(t) for g in games for t in (g.home, g.away)})
    index, n = {t: i for i, t in enumerate(teams)}, len(teams) + 1
    matrix, target = [[0.] * n for _ in range(n)], [0.] * n
    previous_day, previous_season = None, None
    rows = []
    ordered = sorted((g for g in games if g.season >= history_start), key=lambda g: (g.day, g.id))
    for day, grouped in groupby(ordered, key=lambda g: g.day):
        today = list(grouped)
        season = today[0].season
        if previous_day is not None:
            decay = (.5 ** (season - previous_season) if season != previous_season
                     else .5 ** ((day - previous_day).days / (7 * candidate["half_life_weeks"])))
            matrix = [[v * decay for v in row] for row in matrix]
            target = [v * decay for v in target]
        regularized = [row[:] for row in matrix]
        adjusted = target[:]
        for i in range(n):
            regularized[i][i] += candidate["ridge"]
        adjusted[-1] += 2 * candidate["ridge"]
        weights = solve(regularized, adjusted)
        for game in today:
            home, away = team_key(game.home), team_key(game.away)
            features = [(index[home], 1.), (index[away], -1.)]
            if not game.neutral:
                features.append((n - 1, 1.))
            margin = sum(weights[i] * v for i, v in features)
            if game.completed and game.kind == "REG":
                rows.append({"id": game.id, "season": season, "week": game.week,
                             "home": home, "away": away,
                             "result": float(game.home_score > game.away_score) if game.home_score != game.away_score else .5,
                             "probability": sigmoid(margin / 7.5)})
        # Updates are delayed until every forecast on this date is recorded.
        for game in today:
            if not game.completed:
                continue
            features = [(index[team_key(game.home)], 1.), (index[team_key(game.away)], -1.)]
            if not game.neutral:
                features.append((n - 1, 1.))
            cap = candidate["margin_cap"]
            outcome = max(-cap, min(cap, game.home_score - game.away_score))
            for i, x in features:
                target[i] += x * outcome
                for j, y in features:
                    matrix[i][j] += x * y
        previous_day, previous_season = day, season
    return rows


def select_candidate(predictions, years):
    """Choose by inner AUC, requiring no degradation in either proper score."""
    scores = {key: metrics([r for r in rows if r["season"] in years]) for key, rows in predictions.items()}
    baseline = scores["elo"]
    eligible = [key for key, score in scores.items() if score and score["roc_auc"] is not None
                and score["roc_auc"] > baseline["roc_auc"]
                and score["brier"] < baseline["brier"] and score["log_loss"] < baseline["log_loss"]]
    return (min(eligible, key=lambda k: (-scores[k]["roc_auc"], scores[k]["log_loss"], k))
            if eligible else "elo"), scores


def summarize(baseline, candidate, samples):
    return {"metrics": metrics(candidate),
            "by_season": {str(y): metrics([r for r in candidate if r["season"] == y])
                          for y in sorted({r["season"] for r in candidate})},
            "uncertainty": paired_uncertainty(baseline, candidate, samples),
            "auc_uncertainty": auc_uncertainty(baseline, candidate, samples)}


def run(games, config):
    validate_config(config)
    # Future schedules must not change time decay, roster enumeration or work.
    games = [g for g in games if g.season <= max(config["evaluation_seasons"])]
    seasons = config["evaluation_seasons"]
    years = list(range(min(seasons) - 2, max(seasons) + 1))
    for year in years:
        target = [g for g in games if g.season == year and g.kind == "REG"]
        if len(target) < 200 or not all(g.completed for g in target):
            raise ValueError(f"Season {year} must be complete with at least 200 games")
    forecasts, configs = chronological_forecasts(games, years)
    predictions = {"elo": sorted((r for r in forecasts.values() if r["kind"] == "REG"),
                                  key=lambda r: r["id"])}
    for candidate in config["candidates"]:
        predictions[candidate["id"]] = sorted((r for r in margin_replay(games, candidate, config["history_start"])
                                                if r["season"] in years), key=lambda r: r["id"])
    baseline_ids = [(r["id"], r["result"]) for r in predictions["elo"]]
    if any([(r["id"], r["result"]) for r in rows] != baseline_ids for rows in predictions.values()):
        raise ValueError("Every candidate must score the same games")
    folds, selected = [], []
    for year in seasons:
        key, inner = select_candidate(predictions, [year - 2, year - 1])
        rows = [r for r in predictions[key] if r["season"] == year]
        selected.extend(rows)
        folds.append({"season": year, "inner_seasons": [year - 2, year - 1], "selected": key,
                      "inner_scores": inner, "metrics": metrics(rows)})
    outer = {key: [r for r in rows if r["season"] in seasons] for key, rows in predictions.items()}
    selected.sort(key=lambda r: r["id"])
    baseline, samples = outer["elo"], config["bootstrap_samples"]
    indexed = {key: {r["id"]: r["probability"] for r in rows} for key, rows in outer.items()}
    return {"version": VERSION, "config": config, "elo_configs": configs,
            "protocol": "Prior two seasons select by AUC, gated on better Brier and log loss. Ratings use only earlier dates; offseason retention 0.5, home prior 2 points, logistic margin scale 7.5.",
            "caveat": "Retrospective development experiment; these seasons have been examined before. No untouched prospective claim and no automatic promotion.",
            "folds": folds, "baseline": metrics(baseline),
            "candidates": {key: summarize(baseline, rows, samples) for key, rows in outer.items()},
            "selected_policy": summarize(baseline, selected, samples),
            "games": [{**r, "candidate_probabilities": {k: values[r["id"]] for k, values in indexed.items()},
                       "selected_probability": selected[i]["probability"]} for i, r in enumerate(baseline)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {args.data.resolve(), args.config.resolve()}:
        parser.error("Output must not overwrite inputs")
    raw, settings = args.data.read_bytes(), args.config.read_bytes()
    report = run(parse_games(raw.decode("utf-8-sig")), json.loads(settings))
    report["source_sha256"] = hashlib.sha256(raw).hexdigest()
    report["config_sha256"] = hashlib.sha256(settings).hexdigest()
    report["code_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(Path(__file__).parent.glob("*.py"))}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"baseline": report["baseline"], "selected_policy": report["selected_policy"]}, indent=2))


if __name__ == "__main__":
    main()
