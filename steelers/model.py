"""Margin-aware Elo and chronological evaluation, without third-party packages.

Settings are selected on three older seasons and reported on three later seasons.
Neither set includes the season being viewed. Every game is predicted before its
result updates the ratings; postseason games can update strength but evaluation
uses regular-season games only.
"""

import math
from dataclasses import asdict, dataclass, replace
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
    probability_scale: float = 1.0
    rest_coefficient: float = 0.0


BASELINE = ModelConfig()


def team_key(team):
    return ALIASES.get(team, team)


def rating_probability(home_rating, away_rating, neutral=False, config=BASELINE):
    advantage = 0 if neutral else config.home_advantage
    return 1 / (1 + 10 ** ((away_rating - home_rating - advantage) / 400))


def rest_difference(game):
    """Rest in weeks, capped at two weeks; unknown rest contributes nothing."""
    if game.home_rest is None or game.away_rest is None:
        return 0.0
    return (max(0, min(14, game.home_rest)) - max(0, min(14, game.away_rest))) / 7


def sigmoid(value):
    return 1 / (1 + math.exp(-max(-35, min(35, value))))


def calibrated_probability(probability, config, rest=0.0):
    if config.probability_scale == 1 and config.rest_coefficient == 0:
        return probability
    probability = max(1e-12, min(1 - 1e-12, probability))
    odds = math.log(probability / (1 - probability))
    return sigmoid(config.probability_scale * odds + config.rest_coefficient * rest)


def home_probability(home_rating, away_rating, neutral=False, config=BASELINE, rest=0.0):
    return calibrated_probability(rating_probability(home_rating, away_rating, neutral, config), config, rest)


def game_probability(home_rating, away_rating, game, config=BASELINE):
    return home_probability(home_rating, away_rating, game.neutral, config, rest_difference(game))


def fit_calibration(forecasts, config):
    """Fit confidence/rest on older forecasts, shrinking toward unchanged Elo.

    The fixed L2 penalty is shared with the matchup model. Calibration changes
    forecast probabilities only; the underlying team-strength updates stay Elo.
    """
    weights = [0.0, 0.0]
    inputs = []
    for row in forecasts:
        p = max(1e-12, min(1 - 1e-12, row["probability"]))
        inputs.append((math.log(p / (1 - p)), row["rest"], row["result"]))
    for _ in range(300):
        gradient = [0.0, 0.0]
        for odds, rest, result in inputs:
            error = sigmoid((1 + weights[0]) * odds + weights[1] * rest) - result
            gradient[0] += error * odds
            gradient[1] += error * rest
        steps = [.5 * (g / max(1, len(inputs)) + .03 * w) for g, w in zip(gradient, weights)]
        # Keep confidence monotonic and limit extrapolation on unusual histories.
        weights = [max(-.5, min(.5, weights[0] - steps[0])),
                   max(-1., min(1., weights[1] - steps[1]))]
        if max(abs(s) for s in steps) < 1e-7:
            break
    return replace(config, name=f"Calibrated {config.name[0].lower()}{config.name[1:]}",
                   probability_scale=1 + weights[0], rest_coefficient=weights[1])


def margin_multiplier(margin, winner_difference):
    if margin == 0:
        return 1.0
    # Logarithmic weighting limits the effect of blowouts. The correction reduces
    # rating drift when a strong favorite beats a weak opponent by a large margin.
    return math.log1p(abs(margin)) * 2.2 / max(1.0, 2.2 + winner_difference * .001)


def update_ratings(ratings, home, away, result, neutral=False, config=BASELINE, margin=None):
    # Rest and confidence are a forecast layer, not additional team strength.
    expected = rating_probability(ratings[home], ratings[away], neutral, config)
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
        probability = game_probability(ratings[home], ratings[away], game, config)
        if game.season == season:
            predictions.append({"id": game.id, "season": season, "kind": game.kind,
                                "home": home, "away": away, "probability": probability,
                                "result": result, "week": game.week, "rest": rest_difference(game)})
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
    # Fit only on the older window. Later seasons gate deployment against the
    # existing default, never just against the weaker original Elo baseline.
    calibrated = fit_calibration(backtest(tuning_games, tuning_years, challenger), challenger)
    calibrated_predictions = [{**p, "probability": calibrated_probability(p["probability"], calibrated, p["rest"])}
                              for p in challenger_predictions]
    calibrated_score = metrics(calibrated_predictions)
    incumbent_score = metrics(active_predictions)
    calibration_promoted = (calibrated_score["brier"] < incumbent_score["brier"]
                            and calibrated_score["log_loss"] < incumbent_score["log_loss"])
    calibration = {
        "status": "evaluated", "version": "confidence-rest-v1", "candidate": asdict(calibrated),
        "training_seasons": tuning_years, "penalty": .03, "promoted": calibration_promoted,
        "incumbent": asdict(active), "baseline": incumbent_score, "challenger_metrics": calibrated_score,
        "brier_improvement_pct": 100 * (incumbent_score["brier"] - calibrated_score["brier"]) / incumbent_score["brier"],
        "by_team": {team: {"baseline": metrics(active_predictions, team),
                           "challenger": metrics(calibrated_predictions, team)} for team in ("PIT", "MIN")},
        "by_season": [{"season": year,
                       "baseline": metrics([p for p in active_predictions if p["season"] == year]),
                       "challenger": metrics([p for p in calibrated_predictions if p["season"] == year])} for year in test_years],
        "reason": "Confidence and rest calibration improved both probability scores against the previous default."
                  if calibration_promoted else "Calibration did not improve both probability scores; the previous default is retained.",
    }
    if calibration_promoted:
        active, active_predictions = calibrated, calibrated_predictions
    report = {
        "status": "evaluated", "active": asdict(active), "challenger": asdict(challenger),
        "promoted": promoted, "tuning_seasons": tuning_years, "test_seasons": test_years,
        "candidates": len(candidates), "tuning": tuning_score,
        "baseline": baseline_score, "challenger_metrics": challenger_score,
        "active_metrics": metrics(active_predictions), "calibration": calibration,
        "brier_improvement_pct": 100 * (baseline_score["brier"] - challenger_score["brier"]) / baseline_score["brier"],
        "by_season": [{"season": year,
                       "baseline": metrics([p for p in baseline_predictions if p["season"] == year]),
                       "challenger": metrics([p for p in challenger_predictions if p["season"] == year]),
                       "active": metrics([p for p in active_predictions if p["season"] == year])} for year in test_years],
        "by_team": {team: {"baseline": metrics(baseline_predictions, team),
                           "challenger": metrics(challenger_predictions, team),
                           "active": metrics(active_predictions, team)} for team in ("PIT", "MIN")},
        "reliability": reliability_bins(active_predictions),
        "reason": ("Margin-aware settings improved both Brier score and log loss on later seasons. "
                   if promoted else "The margin-aware challenger did not improve both probability scores. ") + calibration["reason"],
    }
    return active, report
