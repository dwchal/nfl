import json
import math
import tempfile
import threading
import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen
from unittest.mock import patch
from app import make_handler

from steelers.analysis import build_dashboard, project_season
from steelers.data import Game, ScheduleStore
from steelers.features import FeatureStore, QBWeek, TeamWeek, parse_feature_csv
from steelers.forecast import ForecastArchive, kickoff_utc
from steelers.matchup import (FeatureState, MatchupModel, corrected, evaluate,
                              explain, next_context, replay)
from steelers.model import ModelConfig, replay_season
from steelers.weather import WeatherStore, summarize

CONFIG = ModelConfig("Margin-aware Elo", 20, 35, 2 / 3, True)


def fixture(identifier="g", week=1, day=1, home_score=24, away_score=10, year=2026):
    return Game(identifier, year, "REG", week, date(year, 9, day), "13:00", "PIT", "MIN",
                home_score, away_score, False, "Acrisure Stadium", 10, 7, "outdoors", "PIT00")


def bundle_for(games):
    bundle = {"team": {}, "player": {}, "rosters": [], "injuries": [], "sources": [], "digest": "fixture"}
    for game in games:
        for team, epa in ((game.home, 10.), (game.away, -5.)):
            key = (game.season, game.week, "REG", team)
            bundle["team"][key] = TeamWeek(epa, 35., 2., 25.)
            bundle["player"][key] = [QBWeek(team + "-qb", team + " Quarterback", epa, 35.)]
    return bundle


class MatchupTests(unittest.TestCase):
    def test_own_and_future_stats_do_not_predict_earlier_games(self):
        games = [fixture("a"), fixture("b", 2, 8), fixture("c", 3, 15)]
        bundle = bundle_for(games)
        original = replay(games, 2026, CONFIG, bundle)[0]
        bundle["team"][(2026, 2, "REG", "PIT")] = TeamWeek(900, 35, 900, 25)
        bundle["player"][(2026, 2, "REG", "PIT")] = [QBWeek("new", "New QB", 900, 35)]
        changed = replay(games, 2026, CONFIG, bundle)[0]
        self.assertEqual(original[0]["features"], changed[0]["features"])
        self.assertEqual(original[1]["features"], changed[1]["features"])
        self.assertNotEqual(original[2]["features"], changed[2]["features"])

    def test_same_day_stats_are_not_available_and_starter_is_past_passer(self):
        games = [fixture("a"), fixture("b", 2, 1), fixture("c", 3, 8)]
        rows, state, _ = replay(games, 2026, CONFIG, bundle_for(games))
        self.assertEqual(rows[0]["features"], rows[1]["features"])
        self.assertNotEqual(rows[1]["features"], rows[2]["features"])
        self.assertEqual(state.last_qb["PIT"][0], "PIT-qb")

    def test_elo_offsets_exactly_match_existing_model(self):
        games = [fixture("a"), fixture("b", 2, 8), fixture("c", 3, 15)]
        rows = replay(games, 2026, CONFIG, bundle_for(games))[0]
        expected = replay_season(games, 2026, CONFIG)[2]
        self.assertEqual([r["probability"] for r in rows], [r["probability"] for r in expected])

    def test_missing_rest_is_neutral_and_offseason_shrinks_state(self):
        state = FeatureState()
        game = fixture()
        self.assertEqual(state.features(replace(game, home_rest=None))[4], 0)
        self.assertAlmostEqual(state.features(game)[4], 3/7)
        state.observe(game, bundle_for([game]))
        before = state.strength("PIT")[0]
        state.offseason()
        self.assertLess(abs(state.strength("PIT")[0]), abs(before))

    def test_qb_scenarios_validate_rosters_and_explanation_sums(self):
        game = fixture(home_score=None, away_score=None)
        state = FeatureState()
        state.qbs = {"starter": (100, 200), "backup": (-100, 200)}
        state.last_qb["PIT"] = ("starter", "Starter")
        model = MatchupModel((.2, .1, .1, .1, .1, .3), state, {"status": "evaluated", "promoted": False})
        bundle = bundle_for([])
        bundle["rosters"] = [{"team": "PIT", "position": "QB", "gsis_id": "starter", "full_name": "Starter"},
                             {"team": "PIT", "position": "QB", "gsis_id": "backup", "full_name": "Backup"}]
        scenario = next_context(model, game, .6, "PIT", bundle, "backup")
        self.assertLess(scenario["probability"], scenario["default_probability"])
        with self.assertRaises(ValueError):
            next_context(model, game, .6, "PIT", bundle, "not-on-roster")
        self.assertAlmostEqual(.6 + sum(c["change"] for c in explain(model, game, .6, "PIT")), model.probability(game, .6))
        self.assertAlmostEqual(.4 + sum(c["change"] for c in explain(model, game, .6, "MIN")), 1-model.probability(game, .6))

    def test_missing_coverage_falls_back_without_fabricated_coefficients(self):
        games = [fixture()]
        model, predictions = evaluate(games, 2026, CONFIG, bundle_for(games))
        self.assertEqual(model.weights, (0.,)*6)
        self.assertEqual(model.report["status"], "unavailable")
        self.assertEqual(predictions, {})

    def test_model_choice_and_simulations_apply_same_correction(self):
        games = [fixture("a"), fixture("b", 2, 8, None, None)]
        state = FeatureState()
        model = MatchupModel((0, 0, 0, 0, 2, 0), state, {"status": "evaluated", "promoted": False})
        original = build_dashboard(games, 2026, simulations=30, matchup=model)
        advanced = build_dashboard(games, 2026, simulations=30, model="matchup", matchup=model, matchup_pregame={})
        self.assertFalse(original["model"]["matchup_active"])
        self.assertTrue(advanced["model"]["matchup_active"])
        self.assertGreater(advanced["next_game"]["win_probability"], original["next_game"]["win_probability"])
        self.assertGreater(advanced["projection"]["expected_wins"], original["projection"]["expected_wins"])
        self.assertEqual(corrected(.6, [100], [0]), .6)

    def test_training_weights_ignore_later_and_target_scores_and_stats(self):
        # Six full-size synthetic seasons; tests train real regression weights.
        games = []
        for year in range(2020, 2027):
            for index in range(205):
                g = fixture(f"{year}-{index}", index+1, 1, 24 if index%3 else 10, 17, year)
                games.append(replace(g, day=date(year, 1, 1)+timedelta(days=index)))
        bundle = bundle_for(games)
        original, _ = evaluate(games, 2026, CONFIG, bundle)
        changed = [replace(g, home_score=0, away_score=50) if g.season>=2023 else g for g in games]
        for key in bundle["team"]:
            if key[0]>=2023:
                bundle["team"][key] = TeamWeek(999, 35, 999, 25)
        altered, _ = evaluate(changed, 2026, CONFIG, bundle)
        self.assertEqual(original.weights, altered.weights)
        self.assertEqual(original.report["ablations"], altered.report["ablations"])
        self.assertNotEqual(original.report["challenger_metrics"], altered.report["challenger_metrics"])


class FeatureCacheTests(unittest.TestCase):
    CSV = "season,team,week,season_type,passing_epa,attempts,sacks_suffered,rushing_epa,carries\n2026,PIT,1,REG,12,30,2,4,20\n"

    def test_bad_refresh_preserves_valid_offline_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"stats_team_week_2026.csv"
            path.write_text(self.CSV)
            with patch("steelers.features.urlopen") as download:
                download.return_value.__enter__.return_value.read.return_value = b"broken"
                _, _, parsed, _, metadata = FeatureStore(directory)._file("team", 2026, True)
            self.assertEqual(parsed[(2026,1,"REG","PIT")].dropbacks, 32)
            self.assertIn("saved", metadata["warning"])
            self.assertEqual(path.read_text(), self.CSV)
            offline = FeatureStore(directory, True)._file("team", 2026, False)
            self.assertEqual(offline[2], parsed)

    def test_invalid_season_duplicates_and_nonfinite_stats_are_rejected(self):
        for content in (self.CSV.replace("2026,PIT", "2025,PIT"), self.CSV + self.CSV.splitlines()[1]+"\n", self.CSV.replace(",12,", ",inf,")):
            with self.assertRaises(ValueError):
                parse_feature_csv(content, "team", 2026)


class ArchiveTests(unittest.TestCase):
    def test_only_pre_kickoff_snapshots_are_saved_and_latest_is_scored(self):
        game = fixture(home_score=None, away_score=None)
        now = datetime(2026,9,1,12,tzinfo=timezone.utc)
        self.assertEqual(kickoff_utc(game).hour, 17)
        with tempfile.TemporaryDirectory() as directory:
            archive = ForecastArchive(Path(directory)/"forecasts.sqlite3")
            self.assertTrue(archive.save(game, "MIN", "auto", .2, {"inputs": 1}, now))
            archive.save(game, "MIN", "auto", .2, {"inputs": 1}, now)
            archive.save(game, "MIN", "auto", .4, {"inputs": 2}, now+timedelta(hours=1))
            self.assertFalse(archive.save(game, "MIN", "auto", .9, {}, kickoff_utc(game)))
            self.assertFalse(archive.save(replace(game,kickoff=""), "MIN", "auto", .9, {}, now))
            self.assertFalse(archive.save(game, "MIN", "auto", .9, {}, now-timedelta(days=8)))
            report = archive.report([replace(game,home_score=24,away_score=10)], "MIN", 2026)
            self.assertEqual(report["snapshots"], 2)
            self.assertEqual(report["evaluated"]["games"], 1)
            self.assertAlmostEqual(report["evaluated"]["brier"], .16)
            self.assertAlmostEqual(report["evaluated"]["log_loss"], -math.log(.6))
            # Revised kickoff earlier than the last save disqualifies that save.
            revised = archive.report([replace(game,kickoff="08:30",home_score=24,away_score=10)], "MIN", 2026)
            self.assertAlmostEqual(revised["evaluated"]["brier"], .04)


class WeatherTests(unittest.TestCase):
    def test_three_hour_window_not_actual_historical_weather(self):
        kickoff = datetime(2026,9,1,17,25,tzinfo=timezone.utc)
        payload = {"hourly": {"time": ["2026-09-01T17:00", "2026-09-01T18:00", "2026-09-01T19:00", "2026-09-01T20:00"],
                              "temperature_2m": [70,69,68,67], "wind_speed_10m": [10,11,12,99],
                              "wind_gusts_10m": [15,16,17,99], "precipitation": [.1,.2,.3,99]}}
        self.assertEqual(summarize(payload,kickoff), {"temperature_f":70,"wind_mph":12,"gust_mph":17,"precipitation_inches":.6})

    def test_indoor_and_past_games_do_not_request_weather(self):
        game = fixture(home_score=None, away_score=None)
        with tempfile.TemporaryDirectory() as directory, patch("steelers.weather.urlopen") as download:
            store = WeatherStore(directory)
            self.assertEqual(store.load(replace(game,roof="dome"))["status"], "indoors")
            self.assertEqual(store.load(game,now=datetime(2026,9,2,tzinfo=timezone.utc))["status"], "unavailable")
            download.assert_not_called()

    def test_download_is_saved_as_issued_and_reused_offline(self):
        now = datetime(2026,9,1,12,tzinfo=timezone.utc)
        game = fixture(home_score=None,away_score=None)
        payload = {"hourly": {"time": ["2026-09-01T17:00", "2026-09-01T18:00", "2026-09-01T19:00"],
                              "temperature_2m": [70,69,68], "wind_speed_10m": [10,11,12],
                              "wind_gusts_10m": [15,16,17], "precipitation": [.1,.2,.3]}}
        with tempfile.TemporaryDirectory() as directory:
            with patch("steelers.weather.urlopen") as download:
                download.return_value.__enter__.return_value.read.return_value = json.dumps(payload).encode()
                loaded = WeatherStore(directory).load(game,now=now)
            self.assertEqual(loaded["status"], "forecast")
            self.assertEqual(loaded["warning"], "")
            self.assertEqual(loaded["retrieved_at"], now.isoformat())
            with patch("steelers.weather.urlopen") as download:
                saved = WeatherStore(directory,True).load(game,now=now+timedelta(hours=1))
                download.assert_not_called()
            self.assertEqual(saved["retrieved_at"], loaded["retrieved_at"])
            self.assertEqual(saved["gust_mph"], 17)


class ScenarioServerTests(unittest.TestCase):
    def test_scenarios_are_isolated_from_schedule_cache_and_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"games.csv"
            future_day = (datetime.now(timezone.utc)+timedelta(days=1)).date().isoformat()
            path.write_text("game_id,season,game_type,week,gameday,gametime,home_team,away_team,home_score,away_score,home_rest,away_rest\n"
                            f"next,2026,REG,1,{future_day},13:00,PIT,MIN,,,10,7\n")
            bundle = bundle_for([])
            bundle["rosters"] = [{"team":"PIT","position":"QB","gsis_id":"starter","full_name":"Starter"},
                                 {"team":"PIT","position":"QB","gsis_id":"backup","full_name":"Backup"}]
            state = FeatureState()
            state.qbs = {"starter":(100,200),"backup":(-100,200)}
            state.last_qb["PIT"] = ("starter","Starter")
            model = MatchupModel((0,0,0,0,.1,.3),state,{"status":"evaluated","promoted":False})
            archive = ForecastArchive(Path(directory)/"forecasts.sqlite3")
            features = SimpleNamespace(load=lambda *args:bundle)
            with patch("app.evaluate",return_value=(model,{})) as evaluation:
                server = ThreadingHTTPServer(("127.0.0.1",0),make_handler(ScheduleStore(path,source_file=path),features,archive))
                thread = threading.Thread(target=server.serve_forever,daemon=True)
                thread.start()
                base = f"http://127.0.0.1:{server.server_port}/api/dashboard?season=2026"
                def fetch(query=""):
                    with urlopen(base+query) as response:
                        return json.load(response)
                try:
                    original = fetch()
                    scenario = fetch("&qb=backup")
                    self.assertLess(scenario["matchup"]["probability"],original["matchup"]["probability"])
                    self.assertEqual(scenario["schedule"],original["schedule"])
                    self.assertEqual(scenario["forward_evaluation"]["snapshots"],1)
                    self.assertEqual(fetch()["matchup"]["selected_qb"],"")
                    with self.assertRaises(HTTPError) as error:
                        fetch("&qb=invalid")
                    self.assertEqual(error.exception.code,400)
                    error.exception.close()
                    selected = fetch("&model=matchup")
                    self.assertTrue(selected["model"]["matchup_active"])
                    self.assertAlmostEqual(selected["next_game"]["win_probability"],selected["matchup"]["default_probability"],places=4)
                    self.assertEqual(evaluation.call_count,1)
                    with archive.connect() as connection:
                        payload = json.loads(connection.execute("SELECT payload FROM forecasts LIMIT 1").fetchone()[0])
                    self.assertIn("sha256",payload["schedule"])
                    self.assertIn("features",payload["matchup"])
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join()


if __name__ == "__main__":
    unittest.main()
