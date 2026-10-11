"""Validate regular-season records and playoff probabilities at fixed checkpoints.

Uses only local schedules. Forecasts reconstruct the prior-selected Elo baseline
on final historical fixtures; actual future scores and postseason fields are
retained exclusively as labels. No production calibration or promotion occurs.
"""

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .data import parse_games
from .forecast import kickoff_utc
from .model import BASE_RATING, game_probability, replay_season, select_model, team_key, update_ratings
from .playoffs import DIVISIONS, NOTE, PlayoffRace

VERSION = "projection-audit-v1"


def validate_config(config):
    required = {"version", "seasons", "weeks", "simulations", "seed", "bootstrap_samples", "model",
                "cutoff_policy", "ties", "postseason", "schedule"}
    if set(config) != required or config["version"] != VERSION or config["model"] != "prior_selected_elo":
        raise ValueError("Unsupported projection audit configuration")
    for key in ("seasons", "weeks"):
        values = config[key]
        if not isinstance(values, list) or not values or values != sorted(set(values)) or any(type(v) is not int for v in values):
            raise ValueError("Seasons/weeks must be sorted unique integer lists")
    if min(config["seasons"]) < 2002 or any(w not in (1, 5, 9, 13) for w in config["weeks"]):
        raise ValueError("Require seasons >= 2002 and checkpoints in weeks 1/5/9/13")
    if any(type(config[k]) is not int or config[k] < 1 for k in ("simulations", "bootstrap_samples")) or config["seed"] != 42:
        raise ValueError("Positive simulation/bootstrap counts and seed 42 are required")
    expected = {
        "cutoff_policy": "00:00 America/New_York on the first scheduled date of the checkpoint week",
        "ties": "Known ties fixed; future ties not simulated",
        "postseason": "Actual postseason field is used for labels only; never for simulation",
        "schedule": "Reconstructed final historical fixtures; not an archived schedule vintage",
    }
    if any(config[k] != v for k, v in expected.items()):
        raise ValueError("Unsupported cutoff, schedule, or outcome policy")
    return config


def snapshot(games, season, week):
    fixtures = sorted((g for g in games if g.season == season and g.kind == "REG"),
                      key=lambda g: (g.day, g.kickoff, g.id))
    starts = [g.day for g in fixtures if g.week == week]
    if not starts:
        raise ValueError("Checkpoint week is absent")
    day = min(starts)
    cutoff = datetime.combine(day, time(), ZoneInfo("America/New_York")).astimezone(timezone.utc)
    masked = [g if (kickoff_utc(g) < cutoff if kickoff_utc(g) else g.day < day)
              else replace(g, home_score=None, away_score=None) for g in fixtures]
    return masked, cutoff


def simulate_league(fixtures, ratings, config, simulations, seed=42):
    """One shared outcome path for all 32 clubs, never 32 independent leagues."""
    if simulations < 1 or not fixtures:
        raise ValueError("A league schedule and positive simulation count are required")
    races = [PlayoffRace(fixtures, fixtures[0].season, t) for t in ("PIT", "MIN")]
    if any(r.reason for r in races):
        raise ValueError(next(r.reason for r in races if r.reason))
    teams = races[0].teams
    current = {t: [0, 0, 0] for t in teams}
    for g in fixtures:
        if g.completed:
            h, a = team_key(g.home), team_key(g.away)
            outcome = 0 if g.home_score > g.away_score else 1 if g.home_score < g.away_score else 2
            current[h][outcome] += 1
            current[a][1 - outcome if outcome != 2 else 2] += 1
    remaining = sorted((g for g in fixtures if not g.completed), key=lambda g: (g.day, g.kickoff, g.id))
    wins, playoffs, division, wild_card = {t: Counter() for t in teams}, Counter(), Counter(), Counter()
    rng, tie_rng = random.Random(seed), random.Random(seed + 1)
    for _ in range(simulations):
        strength = ratings.copy()
        tally = {t: current[t][0] for t in teams}
        outcomes = {}
        for game in remaining:
            h, a = team_key(game.home), team_key(game.away)
            strength.setdefault(h, BASE_RATING)
            strength.setdefault(a, BASE_RATING)
            home_win = rng.random() < game_probability(strength[h], strength[a], game, config)
            outcomes[game.id] = home_win
            tally[h if home_win else a] += 1
            update_ratings(strength, h, a, float(home_win), game.neutral, config)
        for t in teams:
            wins[t][tally[t]] += 1
        for race in races:
            champions, wildcards, _ = race.qualifiers(outcomes, tie_rng)
            for i in champions | wildcards:
                playoffs[race.teams[i]] += 1
            for i in champions:
                division[race.teams[i]] += 1
            for i in wildcards:
                wild_card[race.teams[i]] += 1

    def quantile(counts, fraction):
        cumulative = 0
        for w, n in sorted(counts.items()):
            cumulative += n
            if cumulative >= fraction * simulations:
                return w

    return {t: {"expected_wins": sum(w*n for w, n in wins[t].items()) / simulations,
                "low": quantile(wins[t], .1), "high": quantile(wins[t], .9),
                "distribution": [{"wins": w, "probability": n / simulations} for w, n in sorted(wins[t].items())],
                "known_record": dict(zip(("wins", "losses", "ties"), current[t])),
                "playoff_probability": playoffs[t] / simulations,
                "division_probability": division[t] / simulations, "wild_card_probability": wild_card[t] / simulations}
            for t in teams}


def score(forecast, actual_wins, qualified, total_games):
    cdf, rps = 0., 0.
    mass = {r["wins"]: r["probability"] for r in forecast["distribution"]}
    for wins in range(total_games + 1):
        cdf += mass.get(wins, 0.)
        rps += (cdf - float(actual_wins <= wins)) ** 2
    p = forecast["playoff_probability"]
    clipped = min(1 - 1e-12, max(1e-12, p))
    return {"win_absolute_error": abs(forecast["expected_wins"] - actual_wins), "ranked_probability_score": rps,
            "interval_covered": float(forecast["low"] <= actual_wins <= forecast["high"]),
            "interval_width": forecast["high"] - forecast["low"], "playoff_brier": (p - qualified) ** 2,
            "playoff_log_loss": -(qualified * math.log(clipped) + (1-qualified) * math.log1p(-clipped))}


def summarize(rows):
    if not rows:
        return None
    summary = {key: sum(r["scores"][key] for r in rows) / len(rows) for key in rows[0]["scores"]}
    bins = []
    for i in range(10):
        selected = [r for r in rows if min(9, int(r["playoff_probability"] * 10)) == i]
        bins.append({"lower": i/10, "upper": (i+1)/10, "count": len(selected),
                     "mean_probability": sum(r["playoff_probability"] for r in selected) / len(selected) if selected else None,
                     "qualified_fraction": sum(r["qualified"] for r in selected) / len(selected) if selected else None})
    return {"forecasts": len(rows), **summary, "playoff_reliability": bins}


def season_uncertainty(rows, samples):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["season"]].append(row)
    means = [summarize(block) for _, block in sorted(grouped.items())]
    if not means:
        return None
    keys = list(rows[0]["scores"])
    draws = {key: [] for key in keys}
    rng = random.Random(42)
    for _ in range(samples):
        chosen = rng.choices(means, k=len(means))
        for key in keys:
            draws[key].append(sum(m[key] for m in chosen) / len(chosen))
    def interval(values):
        values.sort()
        return [values[int(.025 * (len(values)-1))], values[int(.975 * (len(values)-1))]]
    return {"season_blocks": len(means), "samples": samples, "seed": 42,
            "note": "Whole seasons resampled; teams and checkpoints within a season remain together. Few season blocks; Monte Carlo error is not included.",
            "intervals": {k: interval(v) for k, v in draws.items()}}


def run(games, config, manifest=None, progress=None):
    config = validate_config(config)
    rows, skipped, configs = [], [], {}
    expected = {t for members in DIVISIONS.values() for t in members}
    for year in config["seasons"]:
        final = [g for g in games if g.season == year and g.kind == "REG"]
        field = {team_key(t) for g in games if g.season == year and g.kind in {"WC", "DIV", "CON", "SB"}
                 for t in (g.home, g.away)}
        slots = 7 if year >= 2020 else 6
        field_valid = all(len(field & {t for name, members in DIVISIONS.items() if name.startswith(c) for t in members}) == slots
                          for c in ("AFC", "NFC")) and field <= expected
        if not final or not all(g.completed for g in final) or not field_valid or PlayoffRace(final, year, "PIT").reason:
            skipped.append({"season": year, "reason": "Need complete regular-season results/schedule and an actual full postseason field."})
            continue
        prior = [g for g in games if g.season < year]
        elo, _ = select_model(prior, year)
        configs[str(year)] = asdict(elo)
        actual = Counter(team_key(g.home if g.home_score > g.away_score else g.away)
                         for g in final if g.home_score != g.away_score)
        totals = Counter(team_key(t) for g in final for t in (g.home, g.away))
        for week in config["weeks"]:
            fixtures, cutoff = snapshot(games, year, week)
            ratings = replay_season(prior + fixtures, year, elo)[0]
            forecasts = simulate_league(fixtures, ratings, elo, config["simulations"], config["seed"])
            for team, forecast in forecasts.items():
                rows.append({"season": year, "checkpoint_week": week, "cutoff": cutoff.isoformat(), "team": team,
                             **forecast, "actual_wins": actual[team], "qualified": team in field,
                             "scores": score(forecast, actual[team], team in field, totals[team])})
            if progress:
                progress(year, week)
    return {"version": VERSION, "config": config, "inputs": manifest or {}, "elo_configs": configs,
            "note": "Retrospective development audit on reconstructed final schedules. Elo only; QB/advanced corrections are not evaluated here. " + NOTE,
            "summary": summarize(rows), "uncertainty": season_uncertainty(rows, config["bootstrap_samples"]),
            "by_checkpoint": {str(w): summarize([r for r in rows if r["checkpoint_week"] == w]) for w in config["weeks"]},
            "by_season": {str(y): summarize([r for r in rows if r["season"] == y]) for y in config["seasons"]},
            "by_team": {t: summarize([r for r in rows if r["team"] == t]) for t in ("PIT", "MIN")},
            "skipped": skipped, "forecasts": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {args.data.resolve(), args.config.resolve()}:
        parser.error("Output cannot replace the schedule or configuration")
    raw, config_bytes = args.data.read_bytes(), args.config.read_bytes()
    manifest = {"schedule_sha256": hashlib.sha256(raw).hexdigest(), "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "code": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}}
    report = run(parse_games(raw.decode("utf-8-sig")), json.loads(config_bytes), manifest,
                 lambda y, w: print(f"Finished {y}, week {w}", flush=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Scored {len(report['forecasts'])} team/checkpoint forecasts; report: {args.output}")


if __name__ == "__main__":
    main()
