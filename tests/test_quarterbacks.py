"""Item 5: starter candidates, probability mixtures and QB rushing."""

import unittest
from dataclasses import replace
from datetime import timedelta

from steelers.advanced import (LABELS, MIN_STARTER_EXAMPLES, RUSHING, AdvancedModel, AdvancedState,
                               enabled_indices)
from steelers.evidence import Evidence
from steelers.features import QBWeek
from steelers.forecast import kickoff_utc
from steelers.matchup import corrected, corrected_mixture, next_context, replay
from steelers.pbp import aggregate, validate
from test_advanced import play
from test_matchup import CONFIG, bundle_for, fixture


def when(game, hours):
    return (kickoff_utc(game) - timedelta(hours=hours)).isoformat()


def depth(game, hours, *players):
    return {"available_at": when(game, hours), "source": "depth", "status": "Projected",
            "players": [{"id": i, "name": n, "rank": r} for r, (i, n) in enumerate(players, start=1)]}


def availability(game, hours, *out):
    return {"available_at": when(game, hours),
            "players": [{"team": "PIT", "id": i, "position": "QB", "status": "Out"} for i in out]}


def confirmation(game, hours, identifier):
    return {"available_at": when(game, hours), "id": identifier, "name": identifier.title(), "status": "User-confirmed", "source": "beat writer"}


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.game = fixture()
        self.cutoff = kickoff_utc(self.game)

    def resolve(self, records, fallback=("old", "Old"), estimator=None):
        return Evidence(records).quarterback_candidates(self.game, "PIT", fallback, self.cutoff, estimator)

    def test_later_out_report_invalidates_earlier_confirmation_and_is_explained(self):
        records = {("confirmed_qb", f"{self.game.id}:PIT"): [confirmation(self.game, 48, "starter")],
                   ("depth", "PIT"): [depth(self.game, 72, ("starter", "Starter"), ("backup", "Backup"))],
                   ("availability", self.game.id): [availability(self.game, 1, "starter")]}
        resolved = self.resolve(records)
        self.assertEqual(resolved["class"], "projected")
        self.assertEqual(resolved["candidates"][0]["id"], "backup")
        self.assertEqual(resolved["conflicts"][0]["kind"], "confirmation_ruled_out")
        self.assertEqual(Evidence(records).quarterback(self.game, "PIT", ("old", "Old"), self.cutoff)["conflicts"], resolved["conflicts"])
        # An earlier Out report cannot override a later confirmation.
        records[("availability", self.game.id)] = [availability(self.game, 72, "starter")]
        resolved = self.resolve(records)
        self.assertEqual(resolved["class"], "confirmed")
        self.assertEqual(resolved["candidates"][0]["id"], "starter")
        self.assertEqual(resolved["conflicts"], [])

    def test_departed_previous_passer_and_all_candidates_ruled_out(self):
        records = {("depth", "PIT"): [depth(self.game, 72, ("starter", "Starter"), ("backup", "Backup"))],
                   ("availability", self.game.id): [availability(self.game, 1, "starter", "backup")]}
        resolved = self.resolve(records, fallback=("traded", "Traded"))
        self.assertEqual(resolved["candidates"], [{**resolved["candidates"][0], "id": "", "probability": 1.}])
        self.assertIn("not on the current depth chart", resolved["candidates"][0]["source"])
        resolved = self.resolve(records, fallback=("starter", "Starter"))
        self.assertEqual(resolved["candidates"][0]["evidence_status"], "Unknown")
        self.assertEqual(resolved["candidates"][0]["source"], "Previous passer ruled out")
        resolved = self.resolve(records, fallback=("veteran", "Veteran"))
        self.assertEqual(resolved["candidates"][0]["evidence_status"], "Unknown")
        # Without any membership evidence the fallback is kept but labeled unverified.
        resolved = self.resolve({}, fallback=("old", "Old"))
        self.assertEqual(resolved["class"], "previous")
        self.assertEqual(resolved["candidates"][0]["evidence_status"], "Previous passer (unverified)")
        self.assertEqual(resolved["candidates"][0]["probability"], 1.)

    def test_deterministic_baseline_exposes_alternatives_with_zero_probability(self):
        records = {("depth", "PIT"): [depth(self.game, 72, ("starter", "Starter"), ("backup", "Backup"), ("third", "Third"))]}
        resolved = self.resolve(records)
        self.assertEqual([c["probability"] for c in resolved["candidates"]], [1., 0., 0., 0.])
        self.assertEqual([c["id"] for c in resolved["candidates"]], ["starter", "backup", "third", ""])
        self.assertIsNone(resolved["support"])

    def test_learned_probabilities_follow_depth_ranks_sum_to_one_and_exclude_ruled_out(self):
        def estimator(evidence_class):
            self.assertEqual(evidence_class, "projected")
            return {"top": .8, "rank2": .1, "rank3": .06, "unknown": .04, "examples": 250}
        records = {("depth", "PIT"): [depth(self.game, 72, ("starter", "Starter"), ("backup", "Backup"), ("third", "Third"), ("fourth", "Fourth"))]}
        resolved = self.resolve(records, estimator=estimator)
        probabilities = {c["id"]: c["probability"] for c in resolved["candidates"]}
        self.assertAlmostEqual(sum(probabilities.values()), 1.)
        self.assertAlmostEqual(probabilities["starter"], .8)
        self.assertAlmostEqual(probabilities["backup"], .1)
        self.assertAlmostEqual(probabilities["third"], .03)
        self.assertAlmostEqual(probabilities["fourth"], .03)
        self.assertAlmostEqual(probabilities[""], .04)
        self.assertEqual(resolved["support"], 250)
        # A ruled-out backup sends the rank-two mass to unknown rather than inventing a starter.
        records[("availability", self.game.id)] = [availability(self.game, 1, "backup")]
        resolved = self.resolve(records, estimator=estimator)
        probabilities = {c["id"]: c["probability"] for c in resolved["candidates"]}
        self.assertNotIn("backup", probabilities)
        self.assertAlmostEqual(probabilities[""], .14)
        self.assertAlmostEqual(sum(probabilities.values()), 1.)


class MixtureTests(unittest.TestCase):
    def test_mixture_averages_probabilities_not_logits(self):
        weights = (1.,) + (0.,) * (len(LABELS) - 1)
        low = [corrected(.5, [v] + [0.] * (len(LABELS) - 1), weights) for v in (-4, 4)]
        self.assertAlmostEqual(low[0], 1 - low[1])
        scenarios = [{"weight": .5, "features": [-4.] + [0.] * (len(LABELS) - 1)},
                     {"weight": .5, "features": [4.] + [0.] * (len(LABELS) - 1)}]
        self.assertAlmostEqual(corrected_mixture(.5, None, scenarios, weights), .5)
        self.assertEqual(corrected_mixture(.3, [0.] * len(LABELS), [], weights), corrected(.3, [0.] * len(LABELS), weights))
        with self.assertRaises(ValueError):
            corrected_mixture(.5, None, [{"weight": 0., "features": [0.] * len(LABELS)}], weights)

    def test_examples_come_only_from_earlier_games_with_candidate_lists(self):
        games = [replace(fixture(f"g{i}", week=i + 1, day=1 + i), day=fixture(day=1).day + timedelta(days=i)) for i in range(3)]
        bundle = bundle_for(games)
        records = {("depth", "PIT"): [depth(g, 72, ("PIT-qb", "Starter"), ("backup", "Backup")) for g in games],
                   ("depth", "MIN"): [depth(g, 72, ("other", "Other"), ("MIN-qb", "Actual")) for g in games]}
        state = AdvancedState(evidence=Evidence(records), now=kickoff_utc(games[-1]) + timedelta(days=30))
        rows, state, _ = replay(games, 2026, CONFIG, bundle, state)
        projected = state.starter_examples["projected"]
        self.assertEqual(projected["examples"], 6)
        self.assertEqual((projected["top"], projected["rank2"]), (3, 3))
        self.assertEqual(state.starter_examples["previous"]["examples"], 0)
        self.assertEqual(rows[0]["input_coverage"]["starter_examples"]["projected"]["examples"], 0)
        self.assertEqual(rows[2]["input_coverage"]["starter_examples"]["projected"]["examples"], 4)
        # Below the support threshold the forecast stays deterministic.
        self.assertEqual([r["scenarios"] for r in rows], [[], [], []])

    def test_supported_mixture_weights_sum_to_one_and_what_if_collapses_a_side(self):
        game = fixture()
        records = {("depth", "PIT"): [depth(game, 72, ("PIT-qb", "Starter"), ("backup", "Backup"))],
                   ("depth", "MIN"): [depth(game, 72, ("MIN-qb", "Starter"), ("reserve", "Reserve"))]}
        state = AdvancedState(evidence=Evidence(records), now=kickoff_utc(game))
        state.starter_examples["projected"].update(examples=MIN_STARTER_EXAMPLES, top=90, rank2=8, rank3=1, unknown=1)
        scenarios = state.scenarios(game)
        self.assertEqual(len(scenarios), 9)
        self.assertAlmostEqual(sum(s["weight"] for s in scenarios), 1.)
        self.assertEqual(scenarios[0]["features"], state.features(game))
        self.assertEqual({s["home_qb"] for s in scenarios}, {"PIT-qb", "backup", ""})
        collapsed = state.scenarios(game, home_qb="backup")
        self.assertEqual({s["home_qb"] for s in collapsed}, {"backup"})
        self.assertEqual(len(collapsed), 3)
        weights = (0.,) * 5 + (1.,) + (0.,) * (len(LABELS) - 6)
        state.qbs["backup"] = (30., 100.)  # a backup with passing history changes the QB feature
        model = AdvancedModel(weights, state, {})
        self.assertEqual(model.probability(game, .6), corrected(.6, state.features(game), weights))
        mixed = AdvancedModel(weights, state, {}, mixture=True)
        self.assertNotEqual(mixed.probability(game, .6), model.probability(game, .6))
        self.assertEqual(mixed.probability(game, .6, home_qb="PIT-qb", away_qb="MIN-qb"), model.probability(game, .6))
        context = next_context(model, game, .6, "PIT", {**bundle_for([game]), "rosters": [
            {"team": "PIT", "position": "QB", "gsis_id": "backup", "full_name": "Backup"}]}, selected_qb="backup")
        self.assertEqual(context["assumption_source"], {"PIT": "projected", "MIN": "projected"})
        self.assertEqual({s["team_qb"] for s in context["starter_scenarios"]}, {"backup"})
        self.assertAlmostEqual(sum(s["weight"] for s in context["starter_scenarios"]), 1.)
        self.assertEqual(context["qb_candidates"]["PIT"]["candidates"][0]["id"], "PIT-qb")


class RushingTests(unittest.TestCase):
    def test_aggregate_separates_qb_rushing_and_excludes_kneels_and_unverified_rushers(self):
        rows = [play("1", passer_player_id="qb"),
                play("2", pass_attempt="0", rush_attempt="1", qb_scramble="1", rusher_player_id="qb", epa="1"),
                play("3", pass_attempt="0", rush_attempt="1", rusher_player_id="qb", epa="2", play_type="run"),
                play("4", pass_attempt="0", rush_attempt="1", rusher_player_id="rb", epa="5", play_type="run"),
                play("5", pass_attempt="0", rush_attempt="1", rusher_player_id="qb", qb_kneel="1", epa="-1"),
                play("6", pass_attempt="0", rush_attempt="1", rusher_player_id="NA", epa="3", play_type="run")]
        payload = validate(aggregate(rows, 2025), 2025)
        self.assertEqual(payload["qb_rushing"], {"2025_01_PIT_MIN": {"qb": {"team": "PIT", "epa": 3., "carries": 2.}}})
        # Team arrays keep scrambles as dropbacks and count every valid rush.
        team = payload["games"]["2025_01_PIT_MIN"]["PIT"]
        self.assertEqual(team[7][1], 5)
        self.assertEqual(team[2], [10., 3.])
        with self.assertRaises(ValueError):
            validate({**payload, "qb_rushing": {"2025_01_PIT_MIN": {"qb": {"team": "NYJ", "epa": 1., "carries": 1.}}}}, 2025)
        with self.assertRaises(ValueError):
            validate({k: v for k, v in payload.items() if k != "qb_rushing"}, 2025)
        legacy = {**payload, "version": "situations-v1"}
        legacy.pop("qb_rushing")
        self.assertEqual(validate(legacy, 2025)["version"], "situations-v1")

    def test_rushing_feature_is_a_verified_change_relative_to_the_team_and_needs_support(self):
        games = [replace(fixture(f"g{i}", week=i + 1, day=1 + i), day=fixture(day=1).day + timedelta(days=i)) for i in range(3)]
        bundle = bundle_for(games)
        bundle["player"][(2026, 3, "REG", "PIT")] = [QBWeek("runner", "Runner", 1., 35.)]
        rushing = {games[0].id: {"PIT-qb": {"team": "PIT", "epa": -2., "carries": 4.}, "MIN-qb": {"team": "MIN", "epa": 0., "carries": 4.},
                                 "unknown-rb": {"team": "PIT", "epa": 9., "carries": 9.}},
                   games[1].id: {"PIT-qb": {"team": "PIT", "epa": -2., "carries": 4.}, "MIN-qb": {"team": "MIN", "epa": 0., "carries": 4.}},
                   games[2].id: {"runner": {"team": "PIT", "epa": 6., "carries": 6.}}}
        state = AdvancedState(qb_rushing=rushing)
        rows, state, _ = replay(games, 2026, CONFIG, bundle, state)
        self.assertEqual(rows[0]["features"][RUSHING], 0.)
        self.assertTrue(rows[0]["input_coverage"]["missing"]["qb_rushing"])
        self.assertFalse(rows[2]["input_coverage"]["missing"]["qb_rushing"])
        self.assertNotIn("unknown-rb", state.rushers)
        self.assertEqual(rows[2]["features"][RUSHING], 0.)
        # The new runner differs from the team's recent QB rushing; the change is relative.
        next_game = replace(fixture("g3", week=4, day=20), day=fixture(day=20).day)
        self.assertGreater(state.features(next_game)[RUSHING], 0.)
        self.assertLess(state.features(next_game, home_qb="PIT-qb")[RUSHING], 0.)
        self.assertEqual(state.features(next_game, home_qb="")[RUSHING], 0.)
        self.assertEqual(state.coverage(next_game)["teams"]["PIT"]["quarterback"]["rushing_carries"], 6.)
        indices, support = enabled_indices(rows, state.contexts)
        self.assertFalse(support["qb_rushing_enabled"])
        self.assertEqual(support["qb_rushing_games"], 2)
        self.assertNotIn(RUSHING, indices)
        # Continuous: coverage and nonzero variation suffice, no zero observations needed.
        covered = [{**r, "features": r["features"][:RUSHING] + [.3], "input_coverage": {"missing": {"qb_rushing": False}}} for r in rows] * 40
        indices, support = enabled_indices(covered, {r["id"]: state.contexts[r["id"]] for r in rows})
        self.assertTrue(support["qb_rushing_enabled"])
        self.assertIn(RUSHING, indices)
        self.assertEqual(support["variation"]["Quarterback rushing change"]["zero"], 0)


if __name__ == "__main__":
    unittest.main()
