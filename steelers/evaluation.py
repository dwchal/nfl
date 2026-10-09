"""Reproducible season-by-season audit of the complete model selection policy.

Run: python3 -m steelers.evaluation --data .cache/games.csv --start 2018 --end 2025
Each season's parameters and deployment decision use only preceding seasons.
"""

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from .data import parse_games
from .model import ModelConfig, backtest, metrics, select_model


def paired_uncertainty(baseline, candidate, samples=2000):
    """Paired season/week block bootstrap; negative loss deltas favor candidate."""
    if samples < 1:
        raise ValueError("At least one bootstrap sample is required.")
    if not baseline or [p["id"] for p in baseline] != [p["id"] for p in candidate]:
        raise ValueError("Comparison requires the same nonempty ordered set of games.")
    blocks = defaultdict(lambda: [0, 0., 0.])

    def loss(row):
        p = max(1e-9, min(1 - 1e-9, row["probability"]))
        y = row["result"]
        return -y * math.log(p) - (1 - y) * math.log1p(-p)

    for old, new in zip(baseline, candidate):
        if old["result"] != new["result"]:
            raise ValueError("Comparison outcomes must match.")
        block = blocks[(old["season"], old["week"])]
        block[0] += 1
        block[1] += (new["probability"] - new["result"]) ** 2 - (old["probability"] - old["result"]) ** 2
        block[2] += loss(new) - loss(old)
    values, rng = list(blocks.values()), random.Random(42)
    draws = [[], []]
    for _ in range(samples):
        chosen = [rng.choice(values) for _ in values]
        count = sum(b[0] for b in chosen)
        for i in range(2):
            draws[i].append(sum(b[i + 1] for b in chosen) / count)
    result = {"method": "Paired season/week block bootstrap", "samples": samples,
              "blocks": len(values), "direction": "Negative candidate-minus-baseline deltas favor candidate."}
    for i, name in enumerate(("brier", "log_loss")):
        ordered = sorted(draws[i])
        result[name] = {"delta": sum(b[i + 1] for b in values) / len(baseline),
                        "interval_95": [ordered[int(.025 * samples)], ordered[min(samples - 1, int(.975 * samples))]]}
    return result


def walk_forward(games, seasons):
    incumbent_rows, active_rows, folds, skipped = [], [], [], []
    for season in seasons:
        target = [g for g in games if g.season == season and g.kind == "REG"]
        if len(target) < 200 or not all(g.completed for g in target):
            skipped.append({"season": season, "reason": "A complete regular season with at least 200 games is required."})
            continue
        # Explicitly isolate model selection from the audited season and beyond.
        config, report = select_model([g for g in games if g.season < season], season)
        if report["status"] != "evaluated":
            skipped.append({"season": season, "reason": report["reason"]})
            continue
        previous = ModelConfig(**report["calibration"]["incumbent"])
        history = [g for g in games if g.season <= season]
        old, new = backtest(history, [season], previous), backtest(history, [season], config)
        incumbent_rows.extend(old)
        active_rows.extend(new)
        folds.append({"season": season, "active": asdict(config), "incumbent": asdict(previous),
                      "calibration_promoted": report["calibration"]["promoted"],
                      "selection_seasons": report["tuning_seasons"], "gate_seasons": report["test_seasons"],
                      "baseline": metrics(old), "candidate": metrics(new)})
    return {"protocol": "Each audited season is excluded from all fitting and deployment gates. Completed earlier games within that season update ratings.",
            "caveat": "Retrospective development audit, not archived live forecasts. Historical provider revisions and research selection can affect results. Bootstrap intervals do not account for model-development selection.",
            "folds": folds, "skipped": skipped, "baseline": metrics(incumbent_rows), "candidate": metrics(active_rows),
            "uncertainty": paired_uncertainty(incumbent_rows, active_rows) if active_rows else None,
            "by_team": {team: {"baseline": metrics(incumbent_rows, team), "candidate": metrics(active_rows, team)}
                        for team in ("PIT", "MIN")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--start", type=int, default=2018)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("--start must not exceed --end")
    raw = args.data.read_bytes()
    report = {"source_sha256": hashlib.sha256(raw).hexdigest(),
              **walk_forward(parse_games(raw.decode("utf-8-sig")), range(args.start, args.end + 1))}
    content = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(content)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
