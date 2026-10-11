"""Re-score saved chronological forecasts and a separate historical odds reference.

No fitting or production-model changes. The odds snapshot has no issuance
timestamps, so it cannot establish performance at a 24h/1h forecast deadline.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from .model import metrics, team_key
from .ranking import summarize


def market_probability(home, away):
    """Normalize the two American-moneyline implied probabilities to sum to 1."""
    def implied(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or abs(value) < 100:
            return None
        return 100 / (value + 100) if value > 0 else -value / (100 - value)
    h, a = implied(home), implied(away)
    return h / (h + a) if h is not None and a is not None else None


def audit(saved, ranking, schedule, samples=1000):
    games = sorted(saved["games"], key=lambda r: r["id"])
    margin = sorted(ranking["games"], key=lambda r: r["id"])
    if (len({r["id"] for r in games}) != len(games)
            or [(r["id"], r["result"]) for r in games] != [(r["id"], r["result"]) for r in margin]):
        raise ValueError("Reports must contain the same unique games and outcomes")
    baseline = [{**r, "probability": r["incumbent_probability"]} for r in games]
    if any(abs(r["probability"] - m["probability"]) > 1e-12 for r, m in zip(baseline, margin)):
        raise ValueError("Baseline forecasts differ between reports")
    predictions = {"elo": baseline}
    for key in games[0]["candidate_probabilities"]:
        if key != "elo":
            predictions[key] = [{**r, "probability": r["candidate_probabilities"][key]} for r in games]
    predictions["existing-selected"] = [{**r, "probability": r["selected_probability"]} for r in games]
    predictions["margin-selected"] = [{**r, "probability": r["selected_probability"]} for r in margin]
    market, available, fallback = [], [], []
    for row in baseline:
        game = schedule[row["id"]]
        h, a = float(game["home_score"]), float(game["away_score"])
        result = .5 if h == a else float(h > a)
        if (result != row["result"] or team_key(game["home_team"]) != row["home"]
                or team_key(game["away_team"]) != row["away"]):
            raise ValueError("Schedule and report teams/outcomes disagree")
        probability = market_probability(game.get("home_moneyline"), game.get("away_moneyline"))
        (available if probability is not None else fallback).append(row["id"])
        market.append({**row, "probability": probability if probability is not None else row["probability"]})
    # Intervals for the predeclared comparisons; all other existing candidates
    # retain descriptive metrics rather than becoming newly selected winners.
    detailed = {"elo", "full-0.1", "full-0.03", "existing-selected", "margin-selected"}
    comparisons = {key: summarize(baseline, rows, samples) if key in detailed else {"metrics": metrics(rows)}
                   for key, rows in predictions.items()}
    return {"target_roc_auc": .8, "auc_policy": "One row per game; home-win orientation; ties excluded; tied scores half credit; undefined with one class.",
            "caveat": "Previously explored retrospective development seasons. Confidence intervals exclude model-search uncertainty. No candidate is promoted by this audit.",
            "comparisons": comparisons,
            "market_reference": {**summarize(baseline, market, samples), "available_games": len(available),
                                 "elo_fallback_game_ids": fallback,
                                 "scope": "Historical schedule odds with no issuance timestamps. Separate reference only, not eligible for live deployment or fixed-horizon claims.",
                                 "source": "https://nflreadr.nflverse.com/articles/dictionary_schedules.html"},
            "market_predictions": [{"id": r["id"], "probability": r["probability"]} for r in market]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {"predictions": args.predictions, "ranking": args.ranking, "schedule": args.data}
    if args.output.resolve() in {p.resolve() for p in paths.values()}:
        parser.error("Output cannot overwrite inputs")
    raw = {name: path.read_bytes() for name, path in paths.items()}
    schedule = {r["game_id"]: r for r in csv.DictReader(io.StringIO(raw["schedule"].decode("utf-8-sig")))}
    report = audit(json.loads(raw["predictions"]), json.loads(raw["ranking"]), schedule)
    report["inputs"] = {name: {"path": str(paths[name]), "sha256": hashlib.sha256(content).hexdigest()}
                        for name, content in raw.items()}
    report["code_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(Path(__file__).parent.glob("*.py"))}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"AUC audit written to {args.output}")


if __name__ == "__main__":
    main()
