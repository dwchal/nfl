"""Margin-aware Elo and chronological evaluation, without third-party packages.

Settings are selected on three older seasons and reported on three later seasons.
Neither set includes the season being viewed. Every game is predicted before its
result updates the ratings; postseason games can update strength but evaluation
uses regular-season games only.
"""

import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import product

BASE_RATING = 1500.0
HOME_ADVANTAGE = 55.0
K_FACTOR = 20.0
ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}


@dataclass(frozen=True)
class ModelConfig:
    name: str = "Original Elo"
    k_factor: float = K_FACTOR
    home_advantage: float = HOME_ADVANTAGE
    carryover: float = 2 / 3
    use_margin: bool = False


BASELINE = ModelConfig()


def team_key(team):
    return ALIASES.get(team, team)


def home_probability(home_rating, away_rating, neutral=False, config=BASELINE):
    advantage = 0 if neutral else config.home_advantage
    return 1 / (1 + 10 ** ((away_rating - home_rating - advantage) / 400))


def margin_multiplier(margin, winner_difference):
    if margin == 0:
        return 1.0
    # Logarithmic weighting limits the effect of blowouts. The correction reduces
    # rating drift when a strong favorite beats a weak opponent by a large margin.
    return math.log1p(abs(margin)) * 2.2 / max(1.0, 2.2 + winner_difference * .001)


def update_ratings(ratings, home, away, result, neutral=False, config=BASELINE, margin=None):
    expected = home_probability(ratings[home], ratings[away], neutral, config)
    multiplier = 1.0
    if config.use_margin and margin is not None:
        difference = ratings[home] - ratings[away] + (0 if neutral else config.home_advantage)
        winner_difference = difference if result > .5 else -difference
        multiplier = margin_multiplier(margin, winner_difference)
    change = config.k_factor * multiplier * (result - expected)
    ratings[home] += change
    ratings[away] -= change


def replay_season(games, season, config=BASELINE, team=None):
    ratings, history, predictions = {}, [], []
    current_season = None
    for game in sorted(games, key=lambda g: (g.day, g.kickoff, g.id)):
        if not season - 3 <= game.season <= season:
            continue
        if game.season != current_season:
            if current_season is not None:
                ratings = {t: BASE_RATING + (r - BASE_RATING) * config.carryover
                           for t, r in ratings.items()}
            current_season = game.season
        home, away = team_key(game.home), team_key(game.away)
        ratings.setdefault(home, BASE_RATING)
        ratings.setdefault(away, BASE_RATING)
        if not game.completed:
            continue
        result = 1.0 if game.home_score > game.away_score else 0.0 if game.home_score < game.away_score else .5
        # Forecast first, then observe the score. No game's score predicts itself.
        probability = home_probability(ratings[home], ratings[away], game.neutral, config)
        if game.season == season:
            predictions.append({"id": game.id, "season": season, "kind": game.kind,
                                "home": home, "away": away, "probability": probability,
                                "result": result})
        update_ratings(ratings, home, away, result, game.neutral, config,
                       game.home_score - game.away_score)
        if game.season == season and team in {home, away}:
            history.append({"week": game.week, "date": game.day.isoformat(),
                            "kind": game.kind, "rating": round(ratings[team], 1)})
    return ratings, history, predictions


def backtest(games, seasons, config=BASELINE):
    forecasts = []
    for season in seasons:
        _, _, predictions = replay_season(games, season, config)
        forecasts.extend(p for p in predictions if p["kind"] == "REG")
    return forecasts


def metrics(forecasts, team=None):
    rows = [p for p in forecasts if team is None or team in {p["home"], p["away"]}]
    if not rows:
        return None
    brier = sum((p["probability"] - p["result"]) ** 2 for p in rows) / len(rows)
    log_loss = 0.0
    for row in rows:
        probability = max(1e-9, min(1 - 1e-9, row["probability"]))
        log_loss -= row["result"] * math.log(probability) + (1 - row["result"]) * math.log1p(-probability)
    decisive = [p for p in rows if p["result"] != .5]
    correct = sum((p["probability"] >= .5) == (p["result"] == 1) for p in decisive)
    return {"games": len(rows), "brier": brier, "log_loss": log_loss / len(rows),
            "accuracy": correct / len(decisive) if decisive else None,
            "ties": len(rows) - len(decisive)}


def reliability_bins(forecasts):
    bins = []
    for index in range(5):
        low, high = index / 5, (index + 1) / 5
        rows = [p for p in forecasts if low <= p["probability"] < high or
                (index == 4 and p["probability"] == 1)]
        bins.append({"low": low, "high": high, "games": len(rows),
                     "predicted": sum(p["probability"] for p in rows) / len(rows) if rows else None,
                     "actual": sum(p["result"] for p in rows) / len(rows) if rows else None})
    return bins


def select_model(games, season):
    # Current-season scores never affect settings or the historical evaluation.
    prior_games = tuple(g for g in games if g.season < season)
    return _select_model(prior_games, season)


@lru_cache(maxsize=8)
def _select_model(games, season):
    available = sorted({g.season for g in games if g.completed and g.kind == "REG"})
    completed = [year for year in available if all(g.completed for g in games if g.season == year and g.kind == "REG")]
    if len(completed) < 6:
        return BASELINE, {"status": "insufficient_history", "active": asdict(BASELINE),
                          "reason": "Six completed prior seasons are needed to tune and evaluate the model."}
    tuning_years, test_years = completed[-6:-3], completed[-3:]
    if any(sum(g.completed and g.kind == "REG" and g.season in years for g in games) < 500
           for years in (tuning_years, test_years)):
        return BASELINE, {"status": "insufficient_history", "active": asdict(BASELINE),
                          "reason": "At least 500 regular-season results are needed in each three-season tuning and evaluation window."}
    # A fixed, small search space is declared before evaluating the test seasons.
    candidates = [BASELINE] + [ModelConfig("Margin-aware Elo", k, home, carry, True)
                              for k, home, carry in product((10., 20., 30.), (35., 55., 75.), (.5, 2 / 3, .8))]
    tuning_games = tuple(g for g in games if g.season <= max(tuning_years))
    scored = [(metrics(backtest(tuning_games, tuning_years, config)), config) for config in candidates]
    tuning_score, challenger = min(scored, key=lambda pair: (pair[0]["brier"], pair[0]["log_loss"]))
    baseline_predictions = backtest(games, test_years, BASELINE)
    challenger_predictions = backtest(games, test_years, challenger)
    baseline_score = metrics(baseline_predictions)
    challenger_score = metrics(challenger_predictions)
    # Deploy the one preselected challenger only if both probability scores improve.
    promoted = (challenger != BASELINE and challenger_score["brier"] < baseline_score["brier"]
                and challenger_score["log_loss"] < baseline_score["log_loss"])
    active = challenger if promoted else BASELINE
    active_predictions = challenger_predictions if promoted else baseline_predictions
    report = {
        "status": "evaluated", "active": asdict(active), "challenger": asdict(challenger),
        "promoted": promoted, "tuning_seasons": tuning_years, "test_seasons": test_years,
        "candidates": len(candidates), "tuning": tuning_score,
        "baseline": baseline_score, "challenger_metrics": challenger_score,
        "active_metrics": challenger_score if promoted else baseline_score,
        "brier_improvement_pct": 100 * (baseline_score["brier"] - challenger_score["brier"]) / baseline_score["brier"],
        "by_season": [{"season": year,
                       "baseline": metrics([p for p in baseline_predictions if p["season"] == year]),
                       "challenger": metrics([p for p in challenger_predictions if p["season"] == year]),
                       "active": metrics([p for p in active_predictions if p["season"] == year])} for year in test_years],
        "by_team": {team: {"baseline": metrics(baseline_predictions, team),
                           "challenger": metrics(challenger_predictions, team),
                           "active": metrics(active_predictions, team)} for team in ("PIT", "MIN")},
        "reliability": reliability_bins(active_predictions),
        "reason": "Margin-aware settings improved both Brier score and log loss on later seasons."
                  if promoted else "The tested challenger did not improve both probability scores; original Elo remains active.",
    }
    return active, report
