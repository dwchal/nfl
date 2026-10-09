import math
import unittest
from dataclasses import replace
from datetime import date, timedelta
from unittest.mock import patch

from steelers.analysis import build_dashboard, project_season
from steelers.data import Game
from steelers.model import (BASELINE, ModelConfig, backtest, home_probability,
                           calibrated_probability, fit_calibration, game_probability, rest_difference,
                           margin_multiplier, metrics, reliability_bins,
                           replay_season, select_model, update_ratings)

MARGIN_MODEL = ModelConfig("Margin-aware Elo", 20, 35, 2 / 3, True)


def game(identifier, year, day, home_score, away_score, neutral=False, kind="REG"):
    return Game(identifier, year, kind, day, date(year, 1, 1) + timedelta(days=day),
                "13:00", "PIT", "BAL", home_score, away_score, neutral, "Test stadium")


class ModelTests(unittest.TestCase):
    def test_calibration_preserves_strength_updates_and_uses_known_rest(self):
        config = replace(MARGIN_MODEL, probability_scale=.8, rest_coefficient=.2)
        first = replace(game("a", 2026, 1, 21, 14), home_rest=14, away_rest=7)
        later = game("b", 2026, 8, 0, 30)
        old_ratings, _, old = replay_season([first, later], 2026, MARGIN_MODEL)
        ratings, _, new = replay_season([first, later], 2026, config)
        self.assertEqual(old_ratings, ratings)
        self.assertAlmostEqual(new[0]["probability"], game_probability(1500, 1500, first, config))
        self.assertNotEqual(new[0]["probability"], old[0]["probability"])
        altered = replay_season([first, replace(later, home_score=100)], 2026, config)[2]
        self.assertEqual([r["probability"] for r in new], [r["probability"] for r in altered])
        self.assertEqual(rest_difference(replace(first, home_rest=None)), 0)
        self.assertEqual(rest_difference(replace(first, home_rest=100)), 1)
        swapped = replace(first, home_rest=7, away_rest=14, neutral=True)
        self.assertAlmostEqual(game_probability(1500, 1500, swapped, config),
                               1 - game_probability(1500, 1500, replace(first, neutral=True), config))

    def test_calibration_shrinks_overconfidence_and_learns_rest(self):
        rows = [{"probability": .9, "rest": 0., "result": float(i % 4 != 0)} for i in range(100)]
        fitted = fit_calibration(rows, MARGIN_MODEL)
        self.assertLess(fitted.probability_scale, 1)
        self.assertLess(calibrated_probability(.9, fitted), .9)
        self.assertGreaterEqual(fitted.probability_scale, .5)
        rows = [{"probability": .5, "rest": 1 if i % 2 else -1, "result": float(i % 2)} for i in range(100)]
        fitted = fit_calibration(rows, MARGIN_MODEL)
        self.assertGreater(fitted.rest_coefficient, 0)
        self.assertEqual(calibrated_probability(.5, fitted), .5)
        self.assertEqual(fit_calibration([], MARGIN_MODEL).probability_scale, 1)

    def test_upcoming_dashboard_and_simulation_apply_calibrated_rest(self):
        config = replace(MARGIN_MODEL, probability_scale=.8, rest_coefficient=.8)
        future = replace(game("next", 2026, 1, None, None, neutral=True), home_rest=14, away_rest=7)
        expected = game_probability(1500, 1500, future, config)
        with patch("steelers.analysis.select_model", return_value=(config, {})):
            dashboard = build_dashboard([future], 2026, simulations=100)
        self.assertAlmostEqual(dashboard["next_game"]["win_probability"], expected, places=4)
        projection = project_season([future], {"PIT": 1500., "BAL": 1500.}, 0, 0, 0,
                                    simulations=4000, config=config)
        self.assertAlmostEqual(projection["expected_wins"], expected, delta=.05)

    def test_blowouts_have_more_weight_and_favorites_have_correction(self):
        narrow, blowout = {"PIT": 1500., "BAL": 1500.}, {"PIT": 1500., "BAL": 1500.}
        update_ratings(narrow, "PIT", "BAL", 1, config=MARGIN_MODEL, margin=1)
        update_ratings(blowout, "PIT", "BAL", 1, config=MARGIN_MODEL, margin=28)
        self.assertGreater(blowout["PIT"], narrow["PIT"])
        self.assertAlmostEqual(sum(blowout.values()), 3000)
        self.assertGreater(margin_multiplier(14, -400), margin_multiplier(14, 400))
        self.assertEqual(margin_multiplier(0, 200), 1)

    def test_home_neutral_and_tie_behavior_use_selected_settings(self):
        self.assertLess(home_probability(1500, 1500, config=MARGIN_MODEL),
                        home_probability(1500, 1500, config=BASELINE))
        self.assertEqual(home_probability(1500, 1500, True, MARGIN_MODEL), .5)
        ratings = {"PIT": 1500., "BAL": 1500.}
        update_ratings(ratings, "PIT", "BAL", .5, True, MARGIN_MODEL, 0)
        self.assertEqual(ratings, {"PIT": 1500., "BAL": 1500.})

    def test_own_score_never_changes_own_prediction(self):
        first = game("a", 2026, 1, 21, 20)
        second = game("b", 2026, 8, 10, 14)
        ratings, _, predictions = replay_season([first, second], 2026, MARGIN_MODEL)
        changed = replace(second, home_score=40, away_score=0)
        changed_ratings, _, changed_predictions = replay_season([first, changed], 2026, MARGIN_MODEL)
        self.assertEqual([p["probability"] for p in predictions],
                         [p["probability"] for p in changed_predictions])
        self.assertNotEqual(ratings, changed_ratings)
        # A past score can affect a later forecast.
        _, _, after_blowout = replay_season([replace(first, home_score=42, away_score=0), second], 2026, MARGIN_MODEL)
        self.assertGreater(after_blowout[1]["probability"], predictions[1]["probability"])

    def test_replay_sorts_games_and_excludes_future_seasons(self):
        first, second = game("a", 2026, 1, 21, 14), game("b", 2026, 8, 10, 14)
        expected = replay_season([first, second], 2026, MARGIN_MODEL)
        actual = replay_season([game("future", 2027, 1, 100, 0), second, first], 2026, MARGIN_MODEL)
        self.assertEqual(actual, expected)

    def test_regular_season_evaluation_excludes_playoffs(self):
        games = [game("a", 2025, 1, 21, 14), game("b", 2025, 8, 0, 14, kind="WC")]
        self.assertEqual([p["id"] for p in backtest(games, [2025])], ["a"])

    def test_probability_scores_and_ties(self):
        forecasts = [{"probability": .8, "result": 1, "home": "PIT", "away": "BAL"},
                     {"probability": .2, "result": 0, "home": "MIN", "away": "GB"},
                     {"probability": .7, "result": .5, "home": "PIT", "away": "BAL"}]
        score = metrics(forecasts)
        self.assertAlmostEqual(score["brier"], .04)
        self.assertEqual(score["accuracy"], 1)
        self.assertEqual(score["ties"], 1)
        self.assertAlmostEqual(metrics(forecasts[:1])["log_loss"], -math.log(.8))
        self.assertEqual(metrics(forecasts, "PIT")["games"], 2)
        self.assertIsNone(metrics(forecasts, "CLE"))
        self.assertEqual(sum(b["games"] for b in reliability_bins(forecasts)), 3)

    def test_missing_history_falls_back_and_completed_games_get_pregame_estimates(self):
        games = [game("a", 2026, 1, 21, 14), game("b", 2026, 8, None, None)]
        config, report = select_model(games, 2026)
        self.assertEqual(config, BASELINE)
        self.assertEqual(report["status"], "insufficient_history")
        dashboard = build_dashboard(games, 2026, simulations=50)
        self.assertGreater(dashboard["schedule"][0]["pregame_probability"], .5)
        self.assertIsNone(dashboard["schedule"][1]["pregame_probability"])
        self.assertEqual(dashboard["model"]["name"], "Original Elo")


class ModelSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Enough synthetic results to exercise selection without a network or data download.
        cls.history = [game(f"{year}_{day}", year, day, 24 if day % 3 else 10, 17)
                       for year in range(2018, 2026) for day in range(180)]

    def test_current_and_future_scores_never_tune_model(self):
        unchanged = select_model(self.history + [game("current", 2026, 1, 21, 14)], 2026)
        changed = select_model(self.history + [game("current", 2026, 1, 0, 90), game("future", 2027, 1, 100, 0)], 2026)
        self.assertEqual(unchanged, changed)
        report = unchanged[1]
        self.assertEqual(report["tuning_seasons"], [2020, 2021, 2022])
        self.assertEqual(report["test_seasons"], [2023, 2024, 2025])
        self.assertEqual(report["baseline"]["games"], 540)

    def test_later_test_results_do_not_choose_candidate_parameters(self):
        _, original = select_model(self.history, 2026)
        changed = [replace(g, home_score=0, away_score=35) if g.season >= 2023 else g for g in self.history]
        _, altered = select_model(changed, 2026)
        self.assertEqual(original["challenger"], altered["challenger"])
        self.assertEqual(original["tuning"], altered["tuning"])
        self.assertEqual(original["calibration"]["candidate"], altered["calibration"]["candidate"])
        self.assertNotEqual(original["baseline"], altered["baseline"])

    def test_tuning_receives_no_test_season_results(self):
        # Use a distinct target to bypass cached selection from other tests.
        shifted = [replace(g, season=g.season - 1,
                           day=date(g.season - 1, 1, 1) + (g.day - date(g.season, 1, 1)))
                   for g in self.history]
        with patch("steelers.model.backtest", wraps=backtest) as evaluate:
            select_model(shifted, 2025)
        for call in evaluate.call_args_list:
            games, years = call.args[:2]
            if years == [2019, 2020, 2021]:
                self.assertTrue(all(g.season <= 2021 for g in games))
        self.assertEqual(sum(call.args[1] == [2019, 2020, 2021] for call in evaluate.call_args_list), 29)

    def test_default_requires_both_scores_to_improve_and_baseline_is_selectable(self):
        active, report = select_model(self.history, 2026)
        if report["promoted"]:
            self.assertLess(report["challenger_metrics"]["brier"], report["baseline"]["brier"])
            self.assertLess(report["challenger_metrics"]["log_loss"], report["baseline"]["log_loss"])
        elif not report["calibration"]["promoted"]:
            self.assertEqual(active, BASELINE)
        calibration = report["calibration"]
        if calibration["promoted"]:
            self.assertLess(calibration["challenger_metrics"]["brier"], calibration["baseline"]["brier"])
            self.assertLess(calibration["challenger_metrics"]["log_loss"], calibration["baseline"]["log_loss"])
            self.assertEqual(active.probability_scale, calibration["candidate"]["probability_scale"])
        current = game("current", 2026, 1, 21, 14)
        dashboard = build_dashboard(self.history + [current], 2026, simulations=50, model="baseline")
        self.assertEqual(dashboard["model"]["name"], "Original Elo")
        self.assertEqual(dashboard["model"]["choice"], "baseline")
        with self.assertRaises(ValueError):
            build_dashboard([current], 2026, model="unknown")


if __name__ == "__main__":
    unittest.main()
