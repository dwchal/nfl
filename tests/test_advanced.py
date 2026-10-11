import csv
import io
import json
import tempfile
import threading
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app import make_handler
from steelers.analysis import build_dashboard
from steelers.advanced import (AdvancedModel, AdvancedState, AdvancedStore, LABELS,
                              WEATHER_START, enabled_indices, evaluate_advanced)
from steelers.data import ScheduleStore
from steelers.evidence import AVAILABILITY_VERSION, Evidence, EvidenceStore, availability_complete
from steelers.features import QBWeek, feature_key
from steelers.forecast import kickoff_utc
from steelers.matchup import corrected, explain, fit, replay
from steelers.model import BASELINE
from steelers.pbp import PBPStore, aggregate, validate
from steelers.travel import base, context, distance
from test_matchup import CONFIG, bundle_for, fixture


def play(identifier="1", **overrides):
    return {"game_id": "2025_01_PIT_MIN", "play_id": identifier, "season_type": "REG", "posteam": "PIT",
            "pass_attempt": "1", "rush_attempt": "0", "sack": "0", "qb_scramble": "0",
            "qb_kneel": "0", "qb_spike": "0", "play_type": "pass", "epa": ".5",
            "down": "1", "ydstogo": "10", "game_seconds_remaining": "1800", "score_differential": "0", **overrides}


class PlayByPlayTests(unittest.TestCase):
    def test_filters_situations_and_counts_sacks_and_scrambles_as_dropbacks(self):
        rows = [play("1"), play("2", sack="1", pass_attempt="0", epa="-1"),
                play("3", qb_scramble="1", rush_attempt="1", pass_attempt="0"),
                play("4", qb_kneel="1"), play("5", qb_spike="1"),
                play("6", game_seconds_remaining="300", score_differential="21"),
                play("7", play_type="no_play"),
                play("8", pass_attempt="0", rush_attempt="1", down="3", ydstogo="8")]
        result = aggregate(rows, 2025)
        values = result["games"]["2025_01_PIT_MIN"]["PIT"]
        self.assertEqual(result["plays"], 4)
        self.assertEqual(values[0], [0., 3.])
        self.assertEqual(values[4], [1., 3.])
        self.assertEqual(values[6], [1., 1.])
        self.assertEqual(values[7], [3., 4.])

    def test_invalid_download_preserves_valid_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pbp_2025.json"
            payload = aggregate([play()], 2025)
            path.write_text(json.dumps(payload))
            with patch("steelers.pbp.urlopen", side_effect=OSError("offline")):
                self.assertIn("warning", PBPStore(directory).load(2025, refresh=True))
            self.assertEqual(json.loads(path.read_text()), payload)
            self.assertEqual(PBPStore(directory, True).load(2025), payload)
        with self.assertRaises(ValueError):
            aggregate([play(), play()], 2025)
        with self.assertRaises(ValueError):
            validate(payload, 2026)


class EvidenceTests(unittest.TestCase):
    def test_capture_distinguishes_reported_unknown_and_explicitly_empty(self):
        game = fixture(home_score=None, away_score=None)
        now = kickoff_utc(game) - timedelta(hours=2)
        bundle = bundle_for([])
        bundle["sources"] = [{"file": "injuries_2026.csv", "downloaded_at": (now - timedelta(hours=1)).isoformat(),
                              "sha256": "source-hash"}]
        bundle["injuries"] = [{"team": "PIT", "week": "1", "game_type": "REG", "gsis_id": "out",
                               "position": "QB", "report_status": "Out"}]
        with tempfile.TemporaryDirectory() as directory:
            store = EvidenceStore(Path(directory) / "evidence.sqlite3")
            store.capture(game, bundle, now=now)
            record = store.snapshot().latest("availability", game.id, now + timedelta(seconds=1))
            self.assertEqual(record["team_reports"]["PIT"], {"status": "reported", "complete": False})
            self.assertEqual(record["team_reports"]["MIN"], {"status": "unknown", "complete": False})
            self.assertFalse(availability_complete(record, "PIT", "MIN"))
            self.assertEqual(record["source"]["sha256"], "source-hash")
            bundle["injury_reports"] = {
                feature_key(2026, 1, "REG", "PIT"): {"status": "reported", "complete": True},
                feature_key(2026, 1, "REG", "MIN"): {"status": "explicitly_empty", "complete": True}}
            store.capture(game, bundle, now=now + timedelta(minutes=1))
            record = store.snapshot().latest("availability", game.id, now + timedelta(minutes=2))
            self.assertTrue(availability_complete(record, "PIT", "MIN"))
            self.assertEqual(record["team_reports"]["MIN"]["status"], "explicitly_empty")
            self.assertEqual(len(store.snapshot().indexed[("availability", game.id)]), 2)

    def test_confirmations_are_pregame_immutable_and_override_projections(self):
        game = fixture(home_score=None, away_score=None)
        before = kickoff_utc(game) - timedelta(hours=2)
        with tempfile.TemporaryDirectory() as directory:
            store = EvidenceStore(Path(directory) / "evidence.sqlite3")
            store.save("depth", "PIT", (before - timedelta(hours=1)).isoformat(),
                       {"players": [{"id": "starter", "name": "Projected", "rank": 1}], "source": "depth"})
            self.assertEqual(store.snapshot().quarterback(game, "PIT", ("old", "Old"), before)["status"], "Projected")
            store.confirm(game, "PIT", "backup", "Confirmed QB", "Team announcement", before)
            evidence = store.snapshot()
            self.assertEqual(evidence.quarterback(game, "PIT", ("old", "Old"), before)["id"], "starter")
            self.assertEqual(evidence.quarterback(game, "PIT", ("old", "Old"), before + timedelta(seconds=1))["id"], "backup")
            with self.assertRaises(ValueError):
                store.confirm(game, "PIT", "other", "Other", "source", kickoff_utc(game))

    def test_out_qb_is_excluded_and_future_or_stale_depth_is_ignored(self):
        game = fixture()
        cutoff = kickoff_utc(game)
        records = {
            ("depth", "PIT"): [{"available_at": (cutoff - timedelta(hours=1)).isoformat(), "source": "depth",
                                  "players": [{"id": "out", "name": "Out", "rank": 1}, {"id": "backup", "name": "Backup", "rank": 2}]}],
            ("availability", game.id): [{"available_at": (cutoff - timedelta(minutes=30)).isoformat(), "players": [{"team": "PIT", "id": "out", "status": "Out"}]}]}
        evidence = Evidence(records)
        self.assertEqual(evidence.quarterback(game, "PIT", ("out", "Out"), cutoff)["id"], "backup")
        self.assertEqual(evidence.quarterback(game, "PIT", ("old", "Old"), cutoff - timedelta(days=1))["status"], "Previous passer")
        self.assertEqual(evidence.quarterback(game, "PIT", ("old", "Old"), cutoff + timedelta(days=8))["status"], "Previous passer")

    def test_capture_does_not_backdate_injuries_or_duplicate_same_inputs(self):
        game = fixture(home_score=None, away_score=None)
        now = kickoff_utc(game) - timedelta(hours=2)
        bundle = bundle_for([])
        bundle["sources"] = [{"file": "injuries_2026.csv", "downloaded_at": (now - timedelta(hours=1)).isoformat()}]
        with tempfile.TemporaryDirectory() as directory:
            store = EvidenceStore(Path(directory) / "evidence.sqlite3")
            store.capture(game, bundle, now=now)
            store.capture(game, bundle, now=now + timedelta(minutes=1))
            self.assertEqual(len(store.snapshot().indexed[("availability", game.id)]), 1)
            self.assertIsNone(store.snapshot().latest("availability", game.id, now))
            self.assertIsNotNone(store.snapshot().latest("availability", game.id, now + timedelta(seconds=1)))

    def test_previous_run_weather_uses_lead_time_and_never_observations(self):
        game = fixture(year=2025)
        kickoff = kickoff_utc(game)
        payload = {"hourly": {"time": [(kickoff + timedelta(hours=i)).strftime("%Y-%m-%dT%H:00") for i in range(3)],
                             "temperature_2m_previous_day1": [70, 69, 68], "wind_speed_10m_previous_day1": [15, 16, 17],
                             "wind_gusts_10m_previous_day1": [20, 21, 22], "precipitation_previous_day1": [.1, .2, .3]}}
        with tempfile.TemporaryDirectory() as directory, patch("steelers.evidence.urlopen") as download:
            download.return_value.__enter__.return_value.read.return_value = json.dumps(payload).encode()
            store = EvidenceStore(Path(directory) / "evidence.sqlite3")
            result = store.historical_weather([game])
            self.assertEqual(result["imported"], 1)
            weather = store.snapshot().latest("weather", game.id, kickoff)
            self.assertEqual(weather["kind"], "previous_day1")
            self.assertEqual(weather["wind_mph"], 17)
            self.assertEqual(weather["lead_hours"], 24)
            self.assertIsNone(store.snapshot().latest("weather", game.id, kickoff - timedelta(days=1)))
            self.assertIn("previous_day1", download.call_args.args[0])


class AdvancedFeaturesTests(unittest.TestCase):
    def test_pbp_presence_does_not_replace_weekly_or_qb_coverage(self):
        games = [replace(fixture(f"{year}-{i}", week=i + 1, year=year),
                         day=fixture(year=year).day + timedelta(days=i))
                 for year in range(2020, 2026) for i in range(200)]
        plays = {g.id: {team: [[0., 30.]] * 8 for team in (g.home, g.away)} for g in games}
        for missing in ("team", "player"):
            bundle = bundle_for(games)
            bundle[missing] = {}
            with self.subTest(missing=missing), \
                 patch("steelers.advanced.select_model", return_value=(BASELINE, {})), \
                 patch("steelers.advanced.fit_diagnostic") as fit_model:
                model, probabilities = evaluate_advanced(games, 2026, BASELINE, bundle, {"plays": plays})
                self.assertEqual(model.report["status"], "unavailable")
                self.assertEqual(model.report["coverage"]["2025"]["pbp"], 200)
                self.assertEqual(model.report["coverage"]["2025"]["covered"], 0)
                self.assertFalse(any(model.weights))
                self.assertEqual(probabilities, {})
                fit_model.assert_not_called()

    def test_pregame_pbp_coverage_describes_observed_history(self):
        games = [fixture("a"), fixture("b", week=2), fixture("c", week=3, day=8)]
        plays = {g.id: {t: [[10., 30.]] * 8 for t in (g.home, g.away)} for g in games}
        rows, _, _ = replay(games, 2026, CONFIG, bundle_for(games), AdvancedState(plays))
        self.assertTrue(all(r["pbp_covered"] for r in rows))
        self.assertEqual(rows[0]["input_coverage"]["teams"]["PIT"]["pbp"]["games"], 0)
        self.assertEqual(rows[1]["input_coverage"]["teams"]["PIT"]["pbp"]["games"], 0)
        later = rows[2]["input_coverage"]["teams"]["PIT"]["pbp"]
        self.assertEqual(later["games"], 2)
        self.assertEqual(later["last_game"], "b")
        self.assertGreater(later["effective_plays"][0], 0)

    def test_unknown_or_legacy_availability_cannot_enable_or_apply_effects(self):
        game = fixture()
        cutoff = kickoff_utc(game)
        legacy = {"available_at": (cutoff - timedelta(hours=1)).isoformat(),
                  "players": [{"team": "PIT", "id": "out", "position": "WR", "status": "Out"}]}
        state = AdvancedState(evidence=Evidence({("availability", game.id): [legacy]}), now=cutoff)
        self.assertEqual(state.features(game)[-4:], [0.] * 4)
        self.assertTrue(state.coverage(game)["missing"]["availability"])
        rows = [{"id": str(i), "features": [1.] * len(LABELS)} for i in range(100)]
        contexts = {r["id"]: {"weather": None, "availability": {"players": []},
                              "availability_complete": False} for r in rows}
        indices, support = enabled_indices(rows, contexts)
        self.assertEqual(support["availability_games"], 0)
        self.assertFalse(support["availability_enabled"])
        self.assertTrue(all(i < len(LABELS) - 4 for i in indices))
        complete = {**legacy, "version": AVAILABILITY_VERSION, "team_reports": {
            "PIT": {"status": "reported", "complete": True},
            "MIN": {"status": "explicitly_empty", "complete": True}}}
        state.evidence = Evidence({("availability", game.id): [complete]})
        self.assertNotEqual(state.features(game)[-4:], [0.] * 4)
        self.assertFalse(state.coverage(game)["missing"]["availability"])

    def test_availability_effect_requires_complete_varied_training_evidence(self):
        rows = [{"id": str(i), "features": [0.] * (len(LABELS) - 4) + [float(i < 50), 0., 1., 0.]}
                for i in range(100)]
        contexts = {r["id"]: {"weather": None, "availability_complete": True} for r in rows}
        indices, support = enabled_indices(rows, contexts)
        self.assertTrue(support["availability_enabled"])
        self.assertIn(len(LABELS) - 4, indices)
        self.assertNotIn(len(LABELS) - 2, indices)
        for row in rows[:50]:
            contexts[row["id"]]["availability_complete"] = False
        indices, support = enabled_indices(rows, contexts)
        self.assertEqual(support["availability_games"], 50)
        self.assertFalse(support["availability_enabled"])

    def test_failed_final_fit_disables_advanced_forecasts_in_dashboard(self):
        games = [replace(fixture(f"{year}-{i}", week=i + 1, year=year),
                         day=fixture(year=year).day + timedelta(days=i))
                 for year in range(2020, 2026) for i in range(200)]
        games.append(fixture("target", year=2026, home_score=None, away_score=None))
        plays = {g.id: {team: [[0., 30.]] * 8 for team in (g.home, g.away)} for g in games}
        def fitting(rows, indices, penalty):
            final = len(rows) == 1200
            return (.1,) * len(LABELS), {"status": "max_iterations" if final else "converged",
                                        "converged": not final}
        with patch("steelers.advanced.select_model", return_value=(BASELINE, {})), \
             patch("steelers.advanced.fit_diagnostic", side_effect=fitting):
            model, probabilities = evaluate_advanced(games, 2026, BASELINE, bundle_for(games), {"plays": plays})
        self.assertEqual(model.report["status"], "unavailable")
        self.assertIn("did not converge", model.report["reason"])
        self.assertFalse(model.report["qualifies"])
        self.assertFalse(any(model.weights))
        self.assertEqual(probabilities, {})
        self.assertIsNotNone(model.report["challenger_metrics"])
        with patch("steelers.analysis.select_model", return_value=(BASELINE, {})):
            dashboard = build_dashboard(games, 2026, model="advanced", matchup=model,
                                        matchup_pregame=probabilities, simulations=20)
        self.assertFalse(dashboard["model"]["matchup_active"])
        self.assertEqual(dashboard["model"]["name"], BASELINE.name)

    def test_annual_fitting_excludes_target_results_and_emits_paired_evaluation(self):
        games = [replace(fixture(f"{year}-{i}", week=i + 1, year=year),
                         day=fixture(year=year).day + timedelta(days=i))
                 for year in range(2020, 2026) for i in range(200)]
        games.append(fixture("target", year=2026))
        plays = {g.id: {team: [[0., 30.]] * 8 for team in (g.home, g.away)} for g in games}
        windows = []
        def fit_without_tuning(rows, indices, penalty):
            windows.append(max(r["season"] for r in rows))
            return (0.,) * len(LABELS), {"status": "converged", "converged": True, "iterations": 0,
                                          "objective": 0.0, "gradient_norm": 0.0}
        with patch("steelers.advanced.select_model", return_value=(BASELINE, {})), patch("steelers.advanced.fit_diagnostic", side_effect=fit_without_tuning):
            model, probabilities = evaluate_advanced(games, 2026, BASELINE, bundle_for(games), {"plays": plays})
        self.assertEqual(windows, [2022] * 4 + [2023] * 4 + [2024] * 4 + [2025])
        self.assertEqual(model.report["status"], "evaluated")
        self.assertEqual(model.report["uncertainty"]["brier"]["delta"], 0)
        self.assertIn("target", probabilities)

    def test_travel_time_zones_neutral_venues_and_relocations(self):
        game = replace(fixture(), away="SEA", kickoff="13:00", stadium_id="PIT00")
        details = context(game)
        self.assertEqual(details["teams"]["SEA"]["body_clock"], "10:00")
        self.assertEqual(details["teams"]["PIT"]["miles"], 0)
        self.assertGreater(details["features"][1], 0)
        neutral = context(replace(game, neutral=True, stadium_id="LON00"))
        self.assertGreater(neutral["teams"]["PIT"]["miles"], 3000)
        self.assertNotEqual(base("LV", 2019), base("LV", 2020))
        self.assertEqual(base("ARI", 2026)[1], "America/Phoenix")
        self.assertAlmostEqual(distance((0, 0), (0, 0)), 0)
        self.assertIsNone(context(replace(game, stadium_id="unknown"))["teams"]["SEA"]["miles"])

    def test_weather_is_style_interaction_and_excludes_enclosed_or_unknown_roof(self):
        game = fixture()
        cutoff = kickoff_utc(game)
        weather = {"available_at": (cutoff - timedelta(hours=2)).isoformat(), "roof": "outdoors",
                   "wind_mph": 25, "precipitation_inches": .25, "temperature_f": 10}
        state = AdvancedState(evidence=Evidence({("weather", game.id): [weather]}), now=cutoff)
        state.situations.styles = {"PIT": (80, 100), "MIN": (30, 100)}
        self.assertEqual(state.features(game)[WEATHER_START:WEATHER_START + 3], [0.] * 3)
        state.venue_roofs[game.stadium_id] = "outdoors"
        self.assertTrue(all(x > 0 for x in state.features(game)[WEATHER_START:WEATHER_START + 3]))
        state.venue_roofs[game.stadium_id] = "dome"
        self.assertEqual(state.features(game)[WEATHER_START:WEATHER_START + 3], [0.] * 3)
        state.venue_roofs[game.stadium_id] = "outdoors"
        state.now = cutoff - timedelta(days=3)
        weather["available_at"] = (state.now - timedelta(hours=1)).isoformat()
        self.assertEqual(state.features(game)[WEATHER_START:WEATHER_START + 3], [0.] * 3)

    def test_pbp_features_are_isolated_from_own_same_day_and_future_stats(self):
        games = [fixture("a"), fixture("b", week=2, day=1), fixture("c", week=3, day=8)]
        plays = {g.id: {t: [[10., 30.]] * 8 for t in (g.home, g.away)} for g in games}
        first = replay(games, 2026, CONFIG, bundle_for(games), AdvancedState(plays))[0]
        plays["a"]["PIT"] = [[900., 30.]] * 8
        changed = replay(games, 2026, CONFIG, bundle_for(games), AdvancedState(plays))[0]
        self.assertEqual(first[0]["features"], changed[0]["features"])
        self.assertEqual(first[1]["features"], changed[1]["features"])
        self.assertNotEqual(first[2]["features"], changed[2]["features"])
        plays["c"]["PIT"] = [[999., 30.]] * 8
        again = replay(games, 2026, CONFIG, bundle_for(games), AdvancedState(plays))[0]
        self.assertEqual(changed, again)

    def test_optional_groups_need_historical_support_and_contributions_sum(self):
        rows = [{"id": str(i), "features": [0.] * WEATHER_START + [float(i < 50), 0., 1.] + [0.] * 4} for i in range(100)]
        contexts = {r["id"]: {"weather": {}, "availability": None} for r in rows}
        indices, support = enabled_indices(rows, contexts)
        self.assertTrue(support["weather_enabled"])
        self.assertFalse(support["availability_enabled"])
        self.assertIn(WEATHER_START, indices)
        self.assertNotIn(WEATHER_START + 1, indices)
        self.assertNotIn(WEATHER_START + 2, indices)
        self.assertEqual(support["variation"][LABELS[WEATHER_START]]["nonzero"], 50)
        self.assertFalse(enabled_indices(rows[:99], contexts)[1]["weather_enabled"])
        self.assertNotIn(len(LABELS) - 1, indices)
        model = AdvancedModel((.01,) * len(LABELS), AdvancedState(), {"status": "evaluated"})
        self.assertAlmostEqual(.6 + sum(c["change"] for c in explain(model, fixture(), .6, "PIT")), model.probability(fixture(), .6))
        training = [{"offset": 0., "features": [0.] * (len(LABELS) - 1) + [1.], "result": 1.} for _ in range(20)]
        weights = fit(training, (len(LABELS) - 1,))
        self.assertGreater(weights[-1], 0)
        self.assertGreater(corrected(.5, training[0]["features"], weights), .5)


class AdvancedServerTests(unittest.TestCase):
    def test_confirmed_lineup_changes_advanced_forecast_and_rejects_cross_origin(self):
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.csv"
            path.write_text("game_id,season,game_type,week,gameday,gametime,home_team,away_team,home_score,away_score\n"
                            "prior,2026,REG,1,2026-09-01,13:00,PIT,MIN,24,10\n"
                            f"next,2026,REG,5,{(now + timedelta(days=1)).date()},13:00,PIT,MIN,,\n")
            bundle = bundle_for([fixture("prior")])
            bundle["player"][(2026, 1, "REG", "PIT")] = [QBWeek("starter", "Starter", 100, 35), QBWeek("backup", "Backup", -100, 10)]
            bundle["rosters"] = [{"team": "PIT", "position": "QB", "gsis_id": p, "full_name": p} for p in ("starter", "backup")]
            from types import SimpleNamespace
            features = SimpleNamespace(load=lambda *args: bundle)
            advanced = AdvancedStore(Path(directory) / "advanced", True)
            def evaluation(games, season, config, bundle, data, *, as_of_utc=None):
                state = AdvancedState(evidence=data["evidence"], now=as_of_utc)
                state.qbs = {"starter": (100, 200), "backup": (-100, 200)}
                state.last_qb["PIT"] = ("starter", "Starter")
                weights = [0.] * len(LABELS)
                weights[5] = .3
                return AdvancedModel(tuple(weights), state, {"status": "evaluated", "promoted": False}), {}
            with patch("app.evaluate_advanced", side_effect=evaluation):
                server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(ScheduleStore(path, source_file=path), features, advanced_store=advanced))
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                base_url = f"http://127.0.0.1:{server.server_port}"
                def fetch():
                    with urlopen(base_url + "/api/dashboard?season=2026&model=advanced") as response:
                        return json.load(response)
                def confirm(origin):
                    request = Request(base_url + "/api/lineup", data=json.dumps({"game_id": "next", "team": "PIT", "player_id": "backup", "source": "Team announcement"}).encode(),
                                      headers={"Content-Type": "application/json", "Origin": origin}, method="POST")
                    with urlopen(request) as response:
                        return json.load(response)
                try:
                    original = fetch()
                    with self.assertRaises(HTTPError) as caught:
                        confirm("https://untrusted.example")
                    self.assertEqual(caught.exception.code, 403)
                    caught.exception.close()
                    self.assertTrue(confirm(base_url)["saved"])
                    changed = fetch()
                    self.assertEqual(changed["matchup"]["assumed_qbs"]["PIT"]["status"], "User-confirmed")
                    self.assertLess(changed["next_game"]["win_probability"], original["next_game"]["win_probability"])
                    self.assertAlmostEqual(changed["next_game"]["win_probability"], changed["matchup"]["default_probability"], places=4)
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join()


if __name__ == "__main__":
    unittest.main()
