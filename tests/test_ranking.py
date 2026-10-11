import unittest
from dataclasses import replace
from datetime import timedelta

from steelers.evaluation import auc_uncertainty
from steelers.auc_audit import audit, market_probability
from steelers.model import metrics, roc_auc
from steelers.ranking import margin_replay, select_candidate, solve, validate_config
from test_model import game


class RankingTests(unittest.TestCase):
    def rows(self, probabilities, results):
        return [{"id": str(i), "home": "PIT", "away": "BAL", "season": 2024,
                 "week": i // 2, "probability": p, "result": y}
                for i, (p, y) in enumerate(zip(probabilities, results))]

    def test_auc_known_pairwise_values_and_ties(self):
        rows = self.rows([.1, .4, .35, .8], [0, 0, 1, 1])
        self.assertEqual(roc_auc(rows), .75)
        self.assertEqual(metrics(rows)["roc_auc"], .75)
        self.assertEqual(roc_auc(self.rows([.5] * 4, [0, 1, 0, 1])), .5)
        self.assertEqual(roc_auc(self.rows([.1, .9], [0, 1])), 1)
        self.assertEqual(roc_auc(self.rows([.9, .1], [0, 1])), 0)
        self.assertEqual(roc_auc(rows + self.rows([.99], [.5])), .75)
        self.assertIsNone(roc_auc([]))
        self.assertIsNone(roc_auc(self.rows([.2, .8], [1, 1])))
        self.assertIsNone(roc_auc(self.rows([.2, .8], [.5, .5])))

    def test_auc_matches_brute_force_and_monotone_rescaling(self):
        import random
        rng = random.Random(42)
        rows = self.rows([rng.choice([.1, .3, .7, .9]) for _ in range(100)],
                         [rng.choice([0, 1]) for _ in range(100)])
        pairs = [(a["probability"], b["probability"]) for a in rows if a["result"] == 1
                 for b in rows if b["result"] == 0]
        brute = sum(1 if a > b else .5 if a == b else 0 for a, b in pairs) / len(pairs)
        self.assertEqual(roc_auc(rows), brute)
        self.assertEqual(roc_auc(rows), roc_auc([{**r, "probability": r["probability"] ** 2} for r in rows]))

    def test_paired_auc_intervals_and_undefined_resamples(self):
        old = self.rows([.5] * 20, [0, 1] * 10)
        new = [{**r, "probability": .9 if r["result"] else .1} for r in old]
        result = auc_uncertainty(old, new, 30)
        self.assertEqual(result["interval_95"], [1, 1])
        self.assertEqual(result["delta_interval_95"], [.5, .5])
        self.assertEqual(result, auc_uncertainty(old, new, 30))
        with self.assertRaises(ValueError):
            auc_uncertainty(old, list(reversed(new)))
        single = self.rows([.5] * 2, [1, 1])
        self.assertIsNone(auc_uncertainty(single, single, 10)["interval_95"])

    def test_linear_solver(self):
        self.assertEqual(solve([[4., 2.], [2., 5.]], [8., 12.]), [1., 2.])

    def test_margin_replay_excludes_own_same_day_and_future_scores(self):
        a, b, c = [game(str(i), 2024, i * 7 + 1, 30, 14) for i in range(3)]
        b = replace(b, day=a.day, home="KC", away="PIT")
        candidate = {"half_life_weeks": 8, "ridge": 4, "margin_cap": 21}
        original = margin_replay([a, b, c], candidate, 2020)
        changed = margin_replay([replace(a, home_score=0), b, replace(c, home_score=0)], candidate, 2020)
        self.assertEqual([r["probability"] for r in original[:2]], [r["probability"] for r in changed[:2]])
        self.assertNotEqual(original[2]["probability"], changed[2]["probability"])
        self.assertEqual(original, margin_replay([c, b, a], candidate, 2020))
        future = replace(c, id="future", day=c.day + timedelta(days=7), home_score=100)
        self.assertEqual(original, margin_replay([a, b, c, future], candidate, 2020)[:3])
        own = margin_replay([a, b, replace(c, home_score=0)], candidate, 2020)
        self.assertEqual(original[2]["probability"], own[2]["probability"])

    def test_selection_ignores_outer_results_and_keeps_probability_gate(self):
        old = self.rows([.5] * 4, [0, 0, 1, 1])
        new = self.rows([.2, .3, .7, .8], [0, 0, 1, 1])
        predictions = {"elo": old, "new": new}
        self.assertEqual(select_candidate(predictions, [2024])[0], "new")
        predictions["new"] += [{**r, "season": 2025, "result": 1 - r["result"]} for r in new]
        self.assertEqual(select_candidate(predictions, [2024])[0], "new")
        # Better ordering but worse probability error cannot pass the gate.
        predictions["new"] = self.rows([.8, .85, .9, .95], [0, 0, 1, 1])
        self.assertEqual(select_candidate(predictions, [2024])[0], "elo")

    def test_market_conversion_and_invalid_odds(self):
        self.assertEqual(market_probability(-110, -110), .5)
        self.assertAlmostEqual(market_probability(-200, 200), 2 / 3)
        self.assertAlmostEqual(market_probability(200, -200), 1 / 3)
        for value in (None, "", "NA", "nan", "inf", 0, 99, -99):
            self.assertIsNone(market_probability(value, -110))

    def test_audit_keeps_missing_odds_games_and_rejects_misalignment(self):
        rows = self.rows([.4, .6], [0, 1])
        saved = {"games": [{**r, "incumbent_probability": r["probability"],
                            "selected_probability": r["probability"],
                            "candidate_probabilities": {"elo": r["probability"]}} for r in rows]}
        ranking = {"games": [{**r, "selected_probability": r["probability"]} for r in rows]}
        schedule = {r["id"]: {"home_team": "PIT", "away_team": "BAL",
                              "home_score": 21 if r["result"] else 14,
                              "away_score": 14 if r["result"] else 21,
                              "home_moneyline": -110 if r["result"] else "NA", "away_moneyline": -110}
                    for r in rows}
        result = audit(saved, ranking, schedule, samples=5)
        self.assertEqual(result["market_reference"]["available_games"], 1)
        self.assertEqual(result["market_reference"]["elo_fallback_game_ids"], ["0"])
        self.assertEqual(result["market_reference"]["metrics"]["games"], 2)
        ranking["games"][0]["probability"] = .3
        with self.assertRaises(ValueError):
            audit(saved, ranking, schedule, samples=5)

    def test_protocol_rejects_invalid_or_ambiguous_candidates(self):
        config = {"version": "margin-ranking-v1", "history_start": 2010,
                  "evaluation_seasons": [2021], "bootstrap_samples": 5,
                  "candidates": [{"id": "margin", "ridge": 4, "half_life_weeks": 8, "margin_cap": 21}]}
        self.assertEqual(validate_config(config), config)
        for key, value in (("ridge", 0), ("half_life_weeks", float("nan")), ("id", "elo")):
            with self.assertRaises(ValueError):
                validate_config({**config, "candidates": [{**config["candidates"][0], key: value}]})
        with self.assertRaises(ValueError):
            validate_config({**config, "candidates": config["candidates"] * 2})


if __name__ == "__main__":
    unittest.main()
