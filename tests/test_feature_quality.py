import csv
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from steelers.features import FeatureStore, feature_key, number, parse_feature_csv
from steelers.coverage import sufficient_history, training_coverage
from steelers.matchup import replay
from test_matchup import CONFIG, bundle_for, fixture


TEAM = {"season": 2026, "team": "PIT", "week": 1, "season_type": "REG",
        "passing_epa": 12, "attempts": 30, "sacks_suffered": 2,
        "rushing_epa": 4, "carries": 20}
QB = {"season": 2026, "team": "PIT", "week": 1, "season_type": "REG",
      "passing_epa": 12, "attempts": 30, "sacks_suffered": 2,
      "player_id": "qb", "player_display_name": "Quarterback", "position": "QB"}


def csv_rows(*rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


class FeatureQualityTests(unittest.TestCase):
    def test_training_coverage_checks_each_source_and_season(self):
        rows = [{"season": 2025, "kind": "REG", "covered": i < 190, "pbp_covered": True} for i in range(200)]
        coverage = training_coverage(rows, [2025], require_pbp=True)
        self.assertTrue(sufficient_history(coverage, require_pbp=True))
        self.assertEqual(coverage["2025"]["pregame_weekly"], 0)
        rows[189]["covered"] = False
        self.assertFalse(sufficient_history(training_coverage(rows, [2025], True), True))
        rows[189]["covered"] = True
        for row in rows[189:]:
            row["pbp_covered"] = False
        coverage = training_coverage(rows, [2025], True)
        self.assertTrue(sufficient_history(coverage))
        self.assertFalse(sufficient_history(coverage, True))
        self.assertFalse(sufficient_history(training_coverage(rows[:199], [2025])))
        self.assertFalse(sufficient_history(training_coverage(rows, [2024, 2025])))

    def test_missing_numbers_are_distinct_from_numeric_zero(self):
        for missing in ("", " NA ", "NaN", "null", None):
            with self.subTest(missing=missing):
                with self.assertRaises(ValueError):
                    number({"x": missing}, "x")
                self.assertIsNone(number({"x": missing}, "x", allow_missing=True))
        self.assertEqual(number({"x": "0"}, "x"), 0.)
        for invalid in ("inf", "-inf", "invalid"):
            with self.assertRaises(ValueError):
                number({"x": invalid}, "x", allow_missing=True)

    def test_missing_epa_requires_explicit_zero_opportunities(self):
        for kind, row in (("team", TEAM), ("player", QB)):
            with self.subTest(kind=kind):
                with self.assertRaises(ValueError):
                    parse_feature_csv(csv_rows({**row, "passing_epa": ""}), kind, 2026)
                parsed = parse_feature_csv(csv_rows({**row, "attempts": 0, "sacks_suffered": 0,
                                                     "passing_epa": ""}), kind, 2026)
                value = parsed[feature_key(2026, 1, "REG", "PIT")]
                self.assertEqual(value.passing_epa if kind == "team" else value[0].epa, 0.)
        with self.assertRaises(ValueError):
            parse_feature_csv(csv_rows({**TEAM, "rushing_epa": ""}), "team", 2026)
        self.assertEqual(parse_feature_csv(csv_rows({**TEAM, "carries": 0, "rushing_epa": ""}), "team", 2026)
                         [feature_key(2026, 1, "REG", "PIT")].rushing_epa, 0.)

    def test_missing_negative_or_fractional_counts_are_rejected(self):
        for field in ("attempts", "sacks_suffered", "carries"):
            for invalid in ("", "NA", -1, 1.5):
                with self.subTest(field=field, value=invalid), self.assertRaises(ValueError):
                    parse_feature_csv(csv_rows({**TEAM, field: invalid}), "team", 2026)

    def test_aliases_and_duplicates_use_canonical_team_keys(self):
        for kind, row in (("team", TEAM), ("player", QB)):
            parsed = parse_feature_csv(csv_rows({**row, "team": "LAR"}), kind, 2026)
            self.assertIn(feature_key(2026, 1, "REG", "LA"), parsed)
            with self.assertRaises(ValueError):
                parse_feature_csv(csv_rows({**row, "team": "LAR"}, {**row, "team": "LA"}), kind, 2026)
        parsed = parse_feature_csv(csv_rows(QB, {**QB, "player_id": "backup"}), "player", 2026)
        self.assertEqual(len(parsed[feature_key(2026, 1, "REG", "PIT")]), 2)

    def test_missing_epa_refresh_preserves_valid_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats_team_week_2026.csv"
            valid = csv_rows(TEAM)
            path.write_text(valid)
            with patch("steelers.features.urlopen") as download:
                download.return_value.__enter__.return_value.read.return_value = csv_rows({**TEAM, "passing_epa": ""}).encode()
                _, _, parsed, _, metadata = FeatureStore(directory)._file("team", 2026, True)
            self.assertEqual(path.read_text(), valid)
            self.assertEqual(parsed[feature_key(2026, 1, "REG", "PIT")].passing_epa, 12.)
            self.assertIn("saved", metadata["warning"])

    def test_replay_tracks_only_earlier_eligible_observations(self):
        games = [fixture("first"), fixture("same-day", week=2), fixture("later", week=3, day=8)]
        rows, state, _ = replay(games, 2026, CONFIG, bundle_for(games))
        for row in rows[:2]:
            team = row["input_coverage"]["teams"]["PIT"]
            self.assertTrue(team["weekly"]["neutral_prior"])
            self.assertIsNone(team["weekly"]["last_game"])
        later = rows[2]["input_coverage"]["teams"]["PIT"]
        self.assertEqual(later["weekly"]["games"], 2)
        self.assertEqual(later["weekly"]["last_game"], "same-day")
        self.assertEqual(later["quarterback"]["last_game"], "same-day")
        self.assertGreater(later["weekly"]["effective_plays"]["pass_offense"], 0)
        before = state.coverage(games[-1])["teams"]["PIT"]
        state.offseason()
        after = state.coverage(games[-1])["teams"]["PIT"]
        self.assertLess(after["weekly"]["effective_plays"]["pass_offense"],
                        before["weekly"]["effective_plays"]["pass_offense"])
        self.assertLess(after["quarterback"]["dropbacks"], before["quarterback"]["dropbacks"])
        changed = bundle_for(games)
        changed["team"][feature_key(2026, 3, "REG", "PIT")] = replace(changed["team"][feature_key(2026, 3, "REG", "PIT")], passing_epa=999.)
        self.assertEqual(rows, replay(games, 2026, CONFIG, changed)[0])

    def test_observe_and_coverage_accept_schedule_aliases(self):
        games = [replace(fixture("first"), home="LAR"), replace(fixture("later", week=2, day=8), home="LAR")]
        bundle = bundle_for([replace(g, home="LA") for g in games])
        rows, state, _ = replay(games, 2026, CONFIG, bundle)
        self.assertTrue(all(r["covered"] for r in rows))
        self.assertIn("LA", state.teams)
        self.assertEqual(rows[1]["input_coverage"]["teams"]["LA"]["weekly"]["last_game"], "first")


if __name__ == "__main__":
    unittest.main()
