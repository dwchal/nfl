import copy
import json
import random
import unittest
from datetime import timedelta
from pathlib import Path

from steelers.evidence import Evidence
from steelers.experiments import Adapter, ReplayRows, run, validate_config
from steelers.forecast import kickoff_utc
from steelers.matchup import replay
from steelers.qb_rushing import RushingState, aggregate, validate
from test_advanced import play
from test_experiments import history, PROTOCOL
from test_matchup import CONFIG, bundle_for, fixture


class StarterEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.game = fixture()
        self.cutoff = kickoff_utc(self.game)
        self.early = (self.cutoff - timedelta(days=2)).isoformat()
        self.late = (self.cutoff - timedelta(hours=1)).isoformat()
        self.records = {
            ("confirmed_qb", self.game.id + ":PIT"): [{"available_at": self.early, "id": "old", "name": "Old", "source": "Team announcement"}],
            ("depth", "PIT"): [{"available_at": self.early, "source": "depth", "players": [
                {"id": "old", "name": "Old", "rank": 1}, {"id": "backup", "name": "Backup", "rank": 2}]}],
            ("availability", self.game.id): [{"available_at": self.late, "players": [{"team": "PIT", "id": "old", "status": "Out"}]}]}

    def test_later_out_invalidates_confirmation_without_changing_earlier_forecast(self):
        evidence = Evidence(self.records)
        self.assertEqual(evidence.quarterback(self.game, "PIT", ("old", "Old"), self.cutoff - timedelta(hours=2))["id"], "old")
        chosen = evidence.quarterback(self.game, "PIT", ("old", "Old"), self.cutoff)
        self.assertEqual(chosen["id"], "backup")
        self.assertEqual(len(chosen["conflicts"]), 1)

    def test_later_explicit_confirmation_can_reverse_out_but_sourceless_cannot(self):
        newer = (self.cutoff - timedelta(minutes=30)).isoformat()
        confirmations = self.records[("confirmed_qb", self.game.id + ":PIT")]
        confirmations.append({"available_at": newer, "id": "old", "name": "Old"})
        self.assertEqual(Evidence(self.records).quarterback(self.game, "PIT", ("old", "Old"), self.cutoff)["id"], "backup")
        confirmations[-1]["source"] = "Team explicitly clears QB to play"
        self.assertEqual(Evidence(self.records).quarterback(self.game, "PIT", ("old", "Old"), self.cutoff)["id"], "old")

    def test_partial_report_omission_cannot_clear_out_and_future_out_cannot_leak(self):
        self.records[("availability", self.game.id)].append({"available_at": (self.cutoff - timedelta(minutes=30)).isoformat(), "players": []})
        self.assertEqual(Evidence(self.records).quarterback(self.game, "PIT", ("old", "Old"), self.cutoff)["id"], "backup")
        self.records[("availability", self.game.id)][0]["available_at"] = (self.cutoff + timedelta(seconds=1)).isoformat()
        self.assertEqual(Evidence(self.records).quarterback(self.game, "PIT", ("old", "Old"), self.cutoff)["id"], "old")

    def test_departed_previous_passer_is_not_restored_when_depth_is_ruled_out(self):
        self.records.pop(("confirmed_qb", self.game.id + ":PIT"))
        self.records[("availability", self.game.id)][0]["players"].append({"team": "PIT", "id": "backup", "status": "Inactive"})
        chosen = Evidence(self.records).quarterback(self.game, "PIT", ("traded", "Traded QB"), self.cutoff)
        self.assertEqual(chosen["id"], "")
        self.assertEqual(chosen["status"], "Unknown")
        no_depth = Evidence().quarterback(self.game, "PIT", ("prior-season", "Prior QB"), self.cutoff)
        self.assertEqual(no_depth["membership_status"], "Unverified")


class RushingResearchTests(unittest.TestCase):
    def test_scrambles_designed_runs_filters_and_unknown_positions(self):
        rows = [play(str(i), pass_attempt="0", rush_attempt="1", rusher_player_id="qb", **values)
                for i, values in enumerate([{"qb_scramble": "1"}, {}, {"qb_kneel": "1"}, {"qb_spike": "1"}, {"no_play": "1"}])]
        rows.extend([play("rb", pass_attempt="0", rush_attempt="1", rusher_player_id="rb"),
                     play("unknown", pass_attempt="0", rush_attempt="1", rusher_player_id="unknown")])
        result = aggregate(rows, 2025, {"qb": "QB", "rb": "RB"})
        entry = result["games"]["2025_01_PIT_MIN"]["PIT"]
        self.assertEqual(entry["players"], {"qb": [1., 2]})
        self.assertEqual(entry["unknown_runs"], 1)
        with self.assertRaises(ValueError):
            aggregate([rows[0], rows[0]], 2025, {"qb": "QB"})
        with self.assertRaises(ValueError):
            validate(result, 2024)

    def test_own_future_and_same_day_rushing_never_predict_earlier_games(self):
        games = [fixture("a"), fixture("b", 2, 1), fixture("c", 3, 8), fixture("d", 4, 15)]
        data = {g.id: {t: {"players": {t + "-qb": [float(i), 2]}, "unknown_runs": 0}
                       for t in (g.home, g.away)} for i, g in enumerate(games)}
        before = replay(games, 2026, CONFIG, bundle_for(games), RushingState(data))[0]
        changed = copy.deepcopy(data)
        changed['c']['PIT']['players']['PIT-qb'] = [999., 2]
        after = replay(games, 2026, CONFIG, bundle_for(games), RushingState(changed))[0]
        self.assertEqual([r['features'] for r in before[:3]], [r['features'] for r in after[:3]])
        self.assertNotEqual(before[3]['features'], after[3]['features'])
        self.assertEqual(before[1]['input_coverage']['teams']['PIT']['qb_rushing']['games'], 0)

    def test_rushing_training_and_prediction_fallbacks_are_explicit(self):
        games = history(2023)
        protocol = {**PROTOCOL, 'bootstrap_samples': 2}
        replay_rows = ReplayRows(games, bundle_for(games), {})
        adapter = Adapter({'id':'rush','family':'qb_rushing','penalty':.03}, replay_rows, protocol)
        fit = adapter.fit(list(range(2016, 2023)))
        self.assertIsNotNone(fit.report['fallback_reason'])
        self.assertEqual(len(fit.weights), 7)
        for row, incumbent in zip(adapter.predict(2023, fit), replay_rows.rows('elo', 2023)):
            self.assertEqual(row['probability'], incumbent['probability'])
            self.assertIsNotNone(row['fallback_reason'])

    def test_feature_masks_keep_unselected_coefficients_zero(self):
        games = history(2023)
        protocol = {**PROTOCOL, 'bootstrap_samples': 2}
        candidate = {'id':'small','family':'matchup','penalty':.3,'indices':[0,1,5]}
        adapter = Adapter(candidate, ReplayRows(games, bundle_for(games), {}), protocol)
        fit = adapter.fit(list(range(2016, 2023)))
        self.assertTrue(fit.report['optimizer']['converged'])
        self.assertEqual(fit.report['enabled_indices'], [0,1,5])
        self.assertTrue(all(fit.weights[i] == 0 for i in (2,3,4)))
        for indices in ([0,0], [6], [], [True]):
            with self.assertRaises(ValueError):
                validate_config({**protocol,'candidates':[{'id':'elo','family':'elo'},{**candidate,'indices':indices}]})


if __name__ == '__main__':
    unittest.main()
