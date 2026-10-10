import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from steelers.context_research import (EfficiencyState, GROUPS, LABELS, REQUIRED, experiment, feature_history,
                                       fit, load_stats, pick_comparison, roof_type, schedule_features)
from steelers.model import BASELINE, metrics, sigmoid
from test_matchup import fixture


def statistics(games):
    return {(g.season, g.week, "REG", team): ((210., 35.), (12., 35.), (2., 35.),
                                           (6., 25.), (1., 60.), (10., 60.))
            for g in games for team in (g.home, g.away)}


class ContextResearchTests(unittest.TestCase):
    def test_experiment_maps_full_width_weights_to_selected_features(self):
        features = [0.1 * (i + 1) for i in range(len(LABELS))]
        rows = [{"id": str(i), "season": year, "week": 1, "home": "PIT", "away": "MIN",
                 "probability": .5, "result": float(i % 2)}
                for year, count in ((2017, 500), (2018, 200)) for i in range(count)]
        inputs = {r["id"]: features for r in rows}
        def fitting(training, indices, penalty):
            weights = [.01 * (i + 1) if i in indices else 0. for i in range(len(LABELS))]
            return weights, {"converged": True}
        with patch("steelers.context_research.feature_history", return_value=inputs), \
             patch("steelers.context_research.select_model", return_value=(BASELINE, {})), \
             patch("steelers.context_research.backtest", side_effect=lambda games, years, config: [r for r in rows if r["season"] in years]), \
             patch("steelers.context_research.fit_logistic_offset", side_effect=fitting), \
             patch("steelers.context_research.paired_uncertainty"), \
             patch("steelers.context_research.pick_comparison"):
            report = experiment([], {}, start=2018, end=2018)
        for name, indices in GROUPS.items():
            with self.subTest(group=name):
                probability = sigmoid(sum(.01 * (i + 1) * features[i] for i in indices))
                expected = metrics([{**r, "probability": probability} for r in rows if r["season"] == 2018])
                actual = report["experiments"][name]
                self.assertEqual(actual["metrics"], expected)
                self.assertEqual(actual["by_season"][0]["weights"], {LABELS[i]: .01 * (i + 1) for i in indices})

    def test_winner_comparison_excludes_ties_and_reports_new_errors(self):
        old = [{"id": str(i), "season": 2025, "week": 1, "probability": .4, "result": r}
               for i, r in enumerate((1., 1., 0., .5))]
        new = [{**r, "probability": .6} for r in old]
        result = pick_comparison(old, new, samples=100)
        self.assertEqual(result["corrected_picks"], 2)
        self.assertEqual(result["new_errors"], 1)
        self.assertEqual(result["decisive_games"], 3)
        self.assertEqual(result["interval_95"], [1 / 3, 1 / 3])
        with self.assertRaises(ValueError):
            pick_comparison(old, list(reversed(new)))

    def test_roof_features_use_previous_venue_type_not_current_roof_decision(self):
        first = replace(fixture("a"), roof="open", stadium_id="X")
        later = replace(fixture("b", week=2, day=8), roof="closed", stadium_id="X")
        rows = feature_history([first, later], statistics([first, later]))
        self.assertEqual(rows[first.id][1:4], [0., 0., 0.])
        self.assertEqual(rows[later.id][1:4], [0., 1., 0.])
        changed = feature_history([first, replace(later, roof="dome")], statistics([first, later]))
        self.assertEqual(rows[later.id], changed[later.id])
        self.assertEqual(roof_type("open"), roof_type("closed"))
        self.assertEqual(schedule_features(replace(later, neutral=True), {"X": "dome"}), [0.] * 9)

    def test_own_same_day_and_future_stats_do_not_enter_prediction(self):
        first = fixture("a")
        same_day = fixture("b", week=2)
        later = fixture("c", week=3, day=8)
        games = [first, same_day, later]
        stats = statistics(games)
        rows = feature_history(games, stats)
        self.assertEqual(rows[first.id], rows[same_day.id])
        stats[(first.season, first.week, "REG", "PIT")] = ((900., 35.),) * 6
        changed = feature_history(games, stats)
        self.assertEqual(rows[first.id], changed[first.id])
        self.assertEqual(rows[same_day.id], changed[same_day.id])
        self.assertNotEqual(rows[later.id], changed[later.id])
        stats[(later.season, later.week, "REG", "PIT")] = ((999., 35.),) * 6
        self.assertEqual(changed, feature_history(games, stats))

    def test_kickoff_missing_and_eastern_time_buckets(self):
        game = fixture()
        self.assertEqual(schedule_features(replace(game, kickoff=""), {})[4:6], [0., 0.])
        self.assertEqual(schedule_features(replace(game, kickoff="13:00"), {})[4:6], [1., 0.])
        self.assertEqual(schedule_features(replace(game, kickoff="20:15"), {})[4:6], [0., 1.])

    def test_opponent_adjustment_shrinks_across_offseason(self):
        game = fixture()
        state = EfficiencyState()
        self.assertFalse(state.observe(game, {}))
        state.observe(game, statistics([game]))
        before = state.strength("PIT")
        state.offseason()
        self.assertTrue(all(abs(a) < abs(b) for a, b in zip(state.strength("PIT"), before)))

    def test_fitting_positive_signal_and_regularization(self):
        rows = [{"offset": 0., "features": [1. if i % 2 else -1.], "result": float(i % 2)} for i in range(100)]
        weak, strong = fit(rows, (0,), .03)[0], fit(rows, (0,), 1.)[0]
        self.assertGreater(weak, strong)
        self.assertGreater(strong, 0)

    def test_team_feed_rejects_missing_values_and_normalizes_aliases(self):
        row = dict.fromkeys(REQUIRED, "1")
        row.update(season="2025", week="1", season_type="REG", team="LAR")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats_team_week_2025.csv"
            def write():
                with path.open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=sorted(REQUIRED))
                    writer.writeheader()
                    writer.writerow(row)
            write()
            stats, sources = load_stats(directory, [2025])
            self.assertIn((2025, 1, "REG", "LA"), stats)
            self.assertEqual(len(sources[0]["sha256"]), 64)
            row["passing_epa"] = "nan"
            write()
            with self.assertRaises(ValueError):
                load_stats(directory, [2025])


if __name__ == "__main__":
    unittest.main()
