import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from steelers.advanced import FEATURE_GROUPS, LABELS as ADVANCED_LABELS, group_indices
from steelers.evidence import Evidence, EvidenceStore
from steelers.experiments import ReplayRows, encode_report, load_inputs, run, validate_config
from steelers.features import QBWeek, TeamWeek
from steelers.matchup import evaluate
from steelers.model import BASELINE, chronological_forecasts
from test_matchup import bundle_for, fixture


PROTOCOL = json.loads((Path(__file__).resolve().parents[1] / "docs/experiments/chronological.json").read_text())


def history(end=2024):
    games = []
    for year in range(2016, end + 1):
        for index in range(200):
            game = fixture(f"{year}_{index:03}", week=index + 1, year=year,
                           home_score=10 if index % 3 == 0 else 24,
                           away_score=24 if index % 3 == 0 else 10)
            games.append(replace(game, day=date(year, 1, 1) + timedelta(days=index)))
    return games


class ChronologicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.games = history()
        cls.bundle = bundle_for(cls.games)
        cls.data = {"plays": {g.id: {team: [[2., 35.] for _ in range(8)] for team in (g.home, g.away)} for g in cls.games}}
        cls.config = {**PROTOCOL, "bootstrap_samples": 5}
        cls.report = run(cls.games, cls.bundle, cls.data, 2023, 2024, cls.config)

    def test_entire_selection_path_ignores_outer_and_future_outcomes_and_statistics(self):
        changed = [replace(g, home_score=g.away_score, away_score=g.home_score)
                   if g.season == 2024 and g.week >= 101 else g for g in self.games]
        bundle, data = copy.deepcopy(self.bundle), copy.deepcopy(self.data)
        for key in bundle["team"]:
            if key[0] == 2024 and key[1] >= 101:
                bundle["team"][key] = TeamWeek(999., 35., -999., 25.)
                bundle["player"][key] = [QBWeek("replacement", "Replacement", 999., 35.)]
        for game in changed:
            if game.season == 2024 and game.week >= 101:
                data["plays"][game.id] = {team: [[999., 35.] for _ in range(8)] for team in (game.home, game.away)}
        altered = run(changed, bundle, data, 2023, 2024, self.config)
        for old, new in zip(self.report["folds"], altered["folds"]):
            for key in ("selected_config", "ranked_candidate", "selection_reason", "inner", "elo_config", "fits"):
                self.assertEqual(old[key], new[key], (old["season"], key))
        self.assertEqual(self.report["fits"], altered["fits"])
        # The modified game's own stats and score cannot alter its forecast.
        for old, new in zip(self.report["games"][:301], altered["games"][:301]):
            self.assertEqual(old["candidate_probabilities"], new["candidate_probabilities"], old["id"])
            self.assertEqual(old["coverage"], new["coverage"])
        self.assertNotEqual(self.report["folds"][-1]["metrics"], altered["folds"][-1]["metrics"])

    def test_every_inner_and_outer_fit_excludes_its_own_year(self):
        for fold in self.report["folds"]:
            self.assertEqual(fold["inner_seasons"], [fold["season"] - 2, fold["season"] - 1])
            for inner in fold["inner"].values():
                for validation in inner["folds"]:
                    fit = self.report["fits"][validation["fit_id"]]
                    self.assertTrue(all(y < validation["season"] for y in fit["training_seasons"]))
                    self.assertLessEqual(len(fit["training_seasons"]), 6)
                    if fit["candidate"]["family"] != "elo":
                        self.assertGreaterEqual(len(fit["training_seasons"]), 3)
            for fit_id in fold["fits"].values():
                self.assertTrue(all(y < fold["season"] for y in self.report["fits"][fit_id]["training_seasons"]))

    def test_selector_receives_strictly_earlier_games_and_reports_exact_config(self):
        def selector(games, year):
            self.assertTrue(all(g.season < year for g in games))
            return replace(BASELINE, home_advantage=year - 2000), {}
        forecasts, configs = chronological_forecasts(self.games, [2023, 2024], selector)
        self.assertEqual(configs["2023"]["home_advantage"], 23)
        self.assertEqual(forecasts["2023_000"]["elo_config"], configs["2023"])

    def test_every_candidate_scores_identical_ordered_game_ids(self):
        identifiers = [r["id"] for r in self.report["games"]]
        self.assertEqual(identifiers, [g.id for g in self.games if g.season >= 2023])
        for row in self.report["games"]:
            self.assertEqual(set(row["candidate_probabilities"]), {"elo", "matchup", "advanced"})
            self.assertEqual(row["incumbent_probability"], row["candidate_probabilities"]["elo"])
            self.assertEqual(row["selected_probability"], row["candidate_probabilities"][row["selected_config"]["id"]])
            self.assertIsNotNone(row["cutoff"])
        for summary in self.report["candidates"].values():
            self.assertEqual(summary["metrics"]["games"], len(identifiers))
        self.assertEqual(json.loads(encode_report(self.report)), self.report)

    def test_app_training_offsets_do_not_depend_on_target_elo_settings(self):
        original, _ = evaluate(self.games, 2025, BASELINE, self.bundle)
        altered, _ = evaluate(self.games, 2025, replace(BASELINE, home_advantage=100., k_factor=40.), self.bundle)
        self.assertEqual(original.weights, altered.weights)
        self.assertEqual(original.report["elo_configs"], altered.report["elo_configs"])
        self.assertEqual(original.report["baseline"], altered.report["baseline"])

    def test_app_failed_correction_fit_is_unavailable(self):
        with patch("steelers.matchup.fit_diagnostic", return_value=((99.,) * 6, {"converged": False, "status": "max_iter"})):
            model, probabilities = evaluate(self.games, 2025, BASELINE, self.bundle)
        self.assertEqual(model.report["status"], "unavailable")
        self.assertFalse(model.report["promoted"])
        self.assertFalse(any(model.weights))
        self.assertFalse(probabilities)

    def test_missing_optional_inputs_score_explicit_elo_fallbacks(self):
        report = run(self.games, {"team": {}, "player": {}}, {}, 2024, 2024, self.config)
        self.assertEqual(len(report["games"]), 200)
        self.assertEqual(report["folds"][0]["selected_config"]["id"], "elo")
        for row in report["games"]:
            for candidate in ("matchup", "advanced"):
                self.assertIsNotNone(row["fallback_reasons"][candidate])
                self.assertEqual(row["candidate_probabilities"][candidate], row["incumbent_probability"])

    def test_failed_fit_scores_fallback_instead_of_failed_coefficients(self):
        def failed(rows, indices, penalty):
            return (99.,) * len(rows[0]["features"]), {"converged": False, "status": "max_iter"}
        with patch("steelers.matchup.fit_diagnostic", side_effect=failed):
            report = run(self.games, self.bundle, self.data, 2023, 2023, self.config)
        for row in report["games"]:
            self.assertEqual(row["incumbent_probability"], row["candidate_probabilities"]["matchup"])
            self.assertIn("converge", row["fallback_reasons"]["matchup"])
        self.assertTrue(all(not any(fit["weights"]) for fit in report["fits"].values()))

    def test_extending_audit_preserves_earlier_predictions_and_selection(self):
        short = run(self.games, self.bundle, self.data, 2023, 2023, self.config)
        self.assertEqual(short["games"], self.report["games"][:200])
        self.assertEqual(short["folds"][0], self.report["folds"][0])

    def test_gate_uses_inner_metrics_only(self):
        for fold in self.report["folds"]:
            scores = {key: value["metrics"] for key, value in fold["inner"].items()}
            order = {c["id"]: index for index, c in enumerate(self.config["candidates"])}
            ranked = min(scores, key=lambda key: (scores[key]["brier"], scores[key]["log_loss"], order[key]))
            self.assertEqual(ranked, fold["ranked_candidate"])
            qualifies = scores[ranked]["brier"] < scores["elo"]["brier"] and scores[ranked]["log_loss"] < scores["elo"]["log_loss"]
            self.assertEqual(fold["selected_config"]["id"], ranked if qualifies else "elo")

    def test_partial_outer_season_is_excluded_explicitly(self):
        games = [replace(g, home_score=None, away_score=None) if g.id == "2024_199" else g for g in self.games]
        report = run(games, self.bundle, self.data, 2024, 2024, self.config)
        self.assertEqual(report["games"], [])
        self.assertEqual(report["skipped"][0]["season"], 2024)
        self.assertIsNone(report["selected_policy"])

    def test_ties_and_insufficient_history_are_serializable_and_deterministic(self):
        games = [replace(g, home_score=10, away_score=10) for g in self.games if g.season == 2016]
        config = {**self.config, "evaluation_seasons": [2016], "candidates": [{"id": "elo", "family": "elo"}]}
        first = run(games, self.bundle, {}, 2016, 2016, config)
        second = run(list(reversed(games)), self.bundle, {}, 2016, 2016, config)
        self.assertEqual(json.dumps(first, sort_keys=True, allow_nan=False), json.dumps(second, sort_keys=True, allow_nan=False))
        self.assertIsNone(first["selected_policy"]["winner_changes"])
        self.assertIn("Insufficient", first["folds"][0]["selection_reason"])


class ReducedModelTests(unittest.TestCase):
    """chronological-v2: declared feature groups and an inner-fold Elo blend."""

    @classmethod
    def setUpClass(cls):
        cls.games = ChronologicalTests.games if hasattr(ChronologicalTests, "games") else history()
        cls.bundle = bundle_for(cls.games)
        cls.data = {"plays": {g.id: {team: [[2., 35.] for _ in range(8)] for team in (g.home, g.away)} for g in cls.games}}
        cls.config = {**PROTOCOL, "version": "chronological-v2", "bootstrap_samples": 5,
                      "blend": {"alphas": [0, 0.5, 1]},
                      "candidates": [{"id": "elo", "family": "elo"},
                                     {"id": "travel", "family": "advanced", "features": "travel", "penalty": 0.1},
                                     {"id": "full", "family": "advanced", "features": "full", "penalty": 0.1}]}
        cls.report = run(cls.games, cls.bundle, cls.data, 2023, 2024, cls.config)

    def test_registry_groups_resolve_to_unique_labels_within_width(self):
        for name in FEATURE_GROUPS:
            indices = group_indices(name)
            self.assertEqual(len(set(indices)), len(indices))
            self.assertTrue(all(0 <= i < len(ADVANCED_LABELS) for i in indices))
        self.assertEqual(group_indices("full"), tuple(range(len(ADVANCED_LABELS))))
        self.assertNotIn(ADVANCED_LABELS.index("Rest advantage"), group_indices("qb_weekly_travel"))
        with self.assertRaises(ValueError):
            group_indices("unknown")
        with patch.dict(FEATURE_GROUPS, {"broken": ("Travel distance", "Not a feature")}):
            with self.assertRaises(ValueError):
                group_indices("broken")

    def test_full_group_reproduces_the_v1_advanced_candidate(self):
        v1 = {**PROTOCOL, "bootstrap_samples": 5, "candidates": [{"id": "elo", "family": "elo"},
                                                                 {"id": "advanced", "family": "advanced", "penalty": 0.1}]}
        legacy = run(self.games, self.bundle, self.data, 2023, 2024, v1)
        self.assertEqual([r["candidate_probabilities"]["advanced"] for r in legacy["games"]],
                         [r["candidate_probabilities"]["full"] for r in self.report["games"]])
        self.assertEqual(legacy["version"], "chronological-v1")
        self.assertNotIn("blended_policy", legacy)
        self.assertTrue(all("blend" not in fold and "blended_probability" not in game
                            for fold in legacy["folds"] for game in legacy["games"]))

    def test_declared_group_only_removes_coefficients(self):
        travel = {i for i, label in enumerate(ADVANCED_LABELS) if label in FEATURE_GROUPS["travel"]}
        for fit in self.report["fits"].values():
            if fit["candidate"]["id"] == "travel":
                self.assertTrue(all(w == 0 for i, w in enumerate(fit["weights"]) if i not in travel))
                self.assertEqual(set(fit["features"]), set(FEATURE_GROUPS["travel"]))
                self.assertEqual(set(fit["clipping"]), set(FEATURE_GROUPS["travel"]))
                self.assertEqual(fit["training_rows"], 200 * len(fit["training_seasons"]))
                self.assertLessEqual(fit["active_coefficients"], len(travel))

    def test_blend_endpoints_are_exact_and_chosen_on_inner_folds_only(self):
        for fold in self.report["folds"]:
            blend = fold["blend"]
            self.assertIn(blend["alpha"], self.config["blend"]["alphas"])
            if fold["selected_config"]["id"] == "elo":
                self.assertEqual(blend["alpha"], 0)
                self.assertIsNone(blend["inner"])
            else:
                self.assertEqual([b["alpha"] for b in blend["inner"]], self.config["blend"]["alphas"])
                elo, chosen = fold["inner"]["elo"]["metrics"], fold["inner"][fold["selected_config"]["id"]]["metrics"]
                self.assertEqual(blend["inner"][0]["metrics"], elo)
                self.assertEqual(blend["inner"][-1]["metrics"], chosen)
                if blend["alpha"]:
                    scores = {b["alpha"]: b["metrics"] for b in blend["inner"]}
                    self.assertLess(scores[blend["alpha"]]["brier"], scores[0]["brier"])
                    self.assertLess(scores[blend["alpha"]]["log_loss"], scores[0]["log_loss"])
        for game in self.report["games"]:
            alpha, chosen = game["blend_alpha"], game["candidate_probabilities"][game["selected_config"]["id"]]
            expected = {0: game["incumbent_probability"], 1: chosen}.get(alpha, (1 - alpha) * game["incumbent_probability"] + alpha * chosen)
            self.assertEqual(game["blended_probability"], expected)
        self.assertEqual(self.report["blended_policy"]["metrics"]["games"], len(self.report["games"]))
        self.assertEqual(set(self.report["games"][0]["coverage"]), {"elo", "advanced"})
        # Outer outcomes cannot move the blend or the selection.
        flipped = [replace(g, home_score=g.away_score, away_score=g.home_score) if g.season == 2024 else g for g in self.games]
        altered = run(flipped, self.bundle, self.data, 2024, 2024, self.config)
        self.assertEqual(altered["folds"][0]["blend"]["alpha"], self.report["folds"][-1]["blend"]["alpha"])
        self.assertEqual(altered["folds"][0]["blend"]["inner"], self.report["folds"][-1]["blend"]["inner"])
        self.assertEqual(altered["folds"][0]["selected_config"], self.report["folds"][-1]["selected_config"])

    def test_v2_protocol_validation(self):
        validate_config(self.config)
        rejected = [{"version": "chronological-v3"},
                    {"blend": {"alphas": [0.5, 1]}}, {"blend": {"alphas": [0, 1, 0.5]}}, {"blend": {"alphas": [0, 2]}},
                    {"blend": {"alphas": [0]}}, {"blend": {"alphas": [0, 1], "extra": 1}},
                    {"candidates": self.config["candidates"] + [{"id": "x", "family": "advanced", "features": "unknown", "penalty": 0.1}]},
                    {"candidates": self.config["candidates"] + [{"id": "y", "family": "matchup", "features": "travel", "penalty": 0.1}]}]
        for override in rejected:
            with self.assertRaises(ValueError, msg=override):
                validate_config({**self.config, **override})
        with self.assertRaises(ValueError):
            validate_config({**PROTOCOL, "blend": {"alphas": [0, 1]}})
        with self.assertRaises(ValueError):
            validate_config({**PROTOCOL, "candidates": PROTOCOL["candidates"] + [
                {"id": "grouped", "family": "advanced", "features": "travel", "penalty": 0.1}]})


class LocalInputsTests(unittest.TestCase):
    SCHEDULE = "game_id,season,game_type,week,gameday,home_team,away_team,home_score,away_score\n2023_01_PIT_MIN,2023,REG,1,2023-09-01,PIT,MIN,24,10\n"

    def test_reconstructed_cutoff_uses_kickoff_independently_of_wall_clock(self):
        game = fixture("future-completed-fixture", year=2070)
        evidence = Evidence({("confirmed_qb", game.id + ":PIT"): [{"available_at": "2070-08-31T12:00:00+00:00",
                               "id": "confirmed", "name": "Confirmed QB", "status": "User-confirmed"}]})
        rows = ReplayRows([game], bundle_for([game]), {"evidence": evidence}).rows("advanced", 2070)
        self.assertEqual(rows[0]["input_coverage"]["teams"]["PIT"]["quarterback"]["id"], "confirmed")

    def test_inputs_are_local_read_only_and_missing_files_are_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schedule = root / "games.csv"
            schedule.write_text(self.SCHEDULE)
            advanced = root / "advanced"
            store = EvidenceStore(advanced / "evidence.sqlite3")
            store.save("confirmed_qb", "2023_01_PIT_MIN:PIT", "2023-08-31T12:00:00Z", {"id": "qb"})
            database = store.path.read_bytes()
            with patch("steelers.features.urlopen", side_effect=AssertionError("network")), patch("steelers.pbp.urlopen", side_effect=AssertionError("network")):
                games, bundle, data, manifest = load_inputs(schedule, root / "weekly", advanced, 2023, 2023)
            self.assertEqual(store.path.read_bytes(), database)
            self.assertEqual(len(games), 1)
            self.assertFalse(bundle["team"])
            self.assertEqual(data["evidence"].indexed[("confirmed_qb", "2023_01_PIT_MIN:PIT")][0]["id"], "qb")
            self.assertEqual(manifest["sources"][0]["sha256"], hashlib.sha256(schedule.read_bytes()).hexdigest())
            self.assertTrue(any(source["status"] == "missing" for source in manifest["sources"]))
            self.assertFalse((root / "weekly").exists())

    def test_invalid_optional_data_is_hashed_and_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "games.csv").write_text(self.SCHEDULE)
            (root / "stats_team_week_2023.csv").write_text("invalid")
            (root / "pbp_2023.json").write_text("{}")
            _, bundle, data, manifest = load_inputs(root / "games.csv", root, root, 2023, 2023)
            self.assertFalse(bundle["team"])
            self.assertFalse(data["plays"])
            invalid = [source for source in manifest["sources"] if source["status"] == "invalid"]
            self.assertEqual(len(invalid), 2)
            self.assertTrue(all(source["sha256"] for source in invalid))
            self.assertFalse((root / "evidence.sqlite3").exists())

    def test_invalid_optional_evidence_is_reported_without_repairing_database(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "games.csv").write_text(self.SCHEDULE)
            (root / "evidence.sqlite3").write_text("invalid database")
            _, _, data, manifest = load_inputs(root / "games.csv", root, root, 2023, 2023)
            self.assertEqual(data["evidence"].indexed, {})
            self.assertEqual(manifest["sources"][-1]["status"], "invalid")
            self.assertEqual((root / "evidence.sqlite3").read_text(), "invalid database")

    def test_protocol_rejects_unimplemented_or_nonfinite_settings(self):
        for override in ({"seed": 7}, {"inner_seasons": 1}, {"unrecognized": True}):
            with self.assertRaises(ValueError):
                validate_config({**PROTOCOL, **override})
        config = copy.deepcopy(PROTOCOL)
        config["candidates"][1]["penalty"] = float("nan")
        with self.assertRaises(ValueError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
