import unittest
from dataclasses import replace
from unittest.mock import patch

from steelers.evaluation import paired_uncertainty, walk_forward
from steelers.model import BASELINE
from test_model import game


class EvaluationTests(unittest.TestCase):
    def test_paired_bootstrap_uses_same_games_and_week_blocks(self):
        old = [{"id": str(i), "season": 2025, "week": i // 2, "result": 1., "probability": .5}
               for i in range(20)]
        new = [{**r, "probability": .75} for r in old]
        report = paired_uncertainty(old, new, samples=100)
        self.assertEqual(report["blocks"], 10)
        self.assertAlmostEqual(report["brier"]["delta"], -.1875)
        self.assertEqual(report["brier"]["interval_95"], [-.1875, -.1875])
        self.assertEqual(report, paired_uncertainty(old, new, samples=100))
        with self.assertRaises(ValueError):
            paired_uncertainty(old, list(reversed(new)))
        with self.assertRaises(ValueError):
            paired_uncertainty([], [])

    def test_audit_hides_target_season_from_selection_and_skips_incomplete(self):
        history = [game(f"{year}-{i}", year, i, 24, 14) for year in (2024, 2025, 2026) for i in range(200)]
        history[-1] = replace(history[-1], home_score=None, away_score=None)
        with patch("steelers.evaluation.select_model", return_value=(BASELINE, {"status": "insufficient_history", "reason": "fixture"})) as select:
            report = walk_forward(history, [2025, 2026])
        self.assertEqual(select.call_count, 1)
        passed, target = select.call_args.args
        self.assertEqual(target, 2025)
        self.assertTrue(all(g.season < target for g in passed))
        self.assertEqual(len(report["skipped"]), 2)
        self.assertIsNone(report["candidate"])


if __name__ == "__main__":
    unittest.main()
