import json
import tempfile
import threading
import unittest
from datetime import date
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app import make_handler
from steelers.analysis import (build_dashboard, default_season, elo_ratings,
                               home_probability, project_season, record_table, update_ratings)
from steelers.data import DataUnavailable, Game, ScheduleStore, parse_games

CSV = """game_id,season,game_type,week,gameday,home_team,away_team,home_score,away_score,location
a,2026,REG,1,2026-09-10,PIT,BAL,21,14,Home
b,2026,REG,2,2026-09-17,CIN,PIT,20,20,Home
c,2026,REG,3,2026-09-24,PIT,CLE,,,Home
d,2026,PRE,0,2026-08-01,PIT,BAL,0,14,Home
e,2026,WC,19,2027-01-17,PIT,BAL,10,24,Home
"""
VIKINGS_ROWS = """min_a,2026,REG,1,2026-09-03,MIN,GB,24,10,Home
min_b,2026,REG,2,2026-09-10,DET,MIN,14,7,Home
min_c,2026,REG,3,2026-09-24,CHI,MIN,,,Home
min_d,2026,WC,19,2027-01-17,MIN,GB,21,14,Home
"""


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.games = parse_games(CSV)

    def test_scores_ties_and_preseason(self):
        self.assertEqual(len(self.games), 4)
        rows = record_table(self.games[:3], {"PIT": 1500})
        pit = next(r for r in rows if r["team"] == "PIT")
        self.assertEqual((pit["wins"], pit["losses"], pit["ties"]), (1, 0, 1))
        self.assertEqual((pit["pf"], pit["pa"], pit["differential"]), (41, 34, 7))
        self.assertEqual(pit["win_pct"], .75)
        cle = next(r for r in rows if r["team"] == "CLE")
        self.assertIsNone(cle["win_pct"])

    def test_home_advantage_neutral_and_rating_conservation(self):
        self.assertGreater(home_probability(1500, 1500), .5)
        self.assertEqual(home_probability(1500, 1500, True), .5)
        ratings = {"PIT": 1500., "BAL": 1500.}
        update_ratings(ratings, "PIT", "BAL", .5, True)
        self.assertEqual(ratings, {"PIT": 1500., "BAL": 1500.})
        update_ratings(ratings, "PIT", "BAL", 1)
        self.assertAlmostEqual(sum(ratings.values()), 3000)
        self.assertGreater(ratings["PIT"], ratings["BAL"])

    def test_year_rollover_and_missing_calendar_season(self):
        historic = [Game("old", 2025, "REG", 1, date(2025, 9, 1), "", "PIT", "BAL", 10, 0, False, "")]
        self.assertEqual(default_season(historic + self.games, date(2026, 1, 3)), 2025)
        self.assertEqual(default_season(historic + self.games, date(2026, 10, 7)), 2026)
        self.assertEqual(default_season(historic, date(2026, 10, 7)), 2025)

    def test_carryover_regresses_toward_mean(self):
        prior = Game("old", 2025, "REG", 1, date(2025, 9, 1), "", "PIT", "BAL", 10, 0, False, "")
        future = Game("new", 2026, "REG", 1, date(2026, 9, 1), "", "PIT", "BAL", None, None, False, "")
        old, _ = elo_ratings([prior], 2025)
        new, _ = elo_ratings([prior, future], 2026)
        self.assertAlmostEqual(new["PIT"], 1500 + (old["PIT"] - 1500) * 2 / 3)

    def test_regular_view_excludes_playoffs_from_stats_and_ratings(self):
        regular = build_dashboard(self.games, 2026, simulations=200)
        all_games = build_dashboard(self.games, 2026, True, simulations=200)
        self.assertEqual(regular["team_stats"]["losses"], 0)
        self.assertEqual(all_games["team_stats"]["losses"], 1)
        self.assertGreater(regular["team_stats"]["rating"], all_games["team_stats"]["rating"])
        self.assertEqual(regular["projection"], all_games["projection"])
        self.assertEqual(regular["next_game"]["opponent"], "CLE")
        self.assertIsNone(regular["schedule"][0]["win_probability"])

    def test_season_complete_and_invalid_selection(self):
        finished = [g for g in self.games if g.completed]
        dashboard = build_dashboard(finished, 2026, simulations=100)
        self.assertIsNone(dashboard["next_game"])
        self.assertEqual(dashboard["projection"]["distribution"], [
            {"wins": 1, "losses": 0, "ties": 1, "probability": 1.0}])
        with self.assertRaises(ValueError):
            build_dashboard(self.games, 1900)

    def test_simulation_bounds_probability_mass_and_reproducibility(self):
        remaining = [g for g in self.games if not g.completed]
        ratings = {"PIT": 1500., "CLE": 1500.}
        projection = project_season(remaining, ratings, 1, 0, 1, 2000)
        self.assertEqual(projection, project_season(remaining, ratings, 1, 0, 1, 2000))
        self.assertEqual(ratings, {"PIT": 1500., "CLE": 1500.})
        self.assertEqual(projection["remaining"], 1)
        self.assertGreater(projection["expected_wins"], 1.5)
        self.assertEqual(projection["low"], 1)
        self.assertEqual(projection["high"], 2)
        self.assertAlmostEqual(sum(r["probability"] for r in projection["distribution"]), 1, places=3)
        self.assertTrue(all(r["wins"] + r["losses"] + r["ties"] == 3 for r in projection["distribution"]))

    def test_validation_and_chronological_order(self):
        with self.assertRaises(ValueError):
            parse_games("not,a,schedule\n1,2,3")
        with self.assertRaises(ValueError):
            parse_games(CSV.replace("c,2026", "a,2026"))
        with self.assertRaises(ValueError):
            parse_games(CSV.replace("21,14", "-1,14"))
        reversed_csv = "\n".join([CSV.splitlines()[0], *reversed(CSV.splitlines()[1:])])
        self.assertEqual([g.id for g in parse_games(reversed_csv)], ["a", "b", "c", "e"])

    def test_vikings_use_their_own_record_schedule_and_division(self):
        games = parse_games(CSV + VIKINGS_ROWS)
        vikings = build_dashboard(games, 2026, simulations=200, team="MIN")
        self.assertEqual(vikings["team"]["name"], "Minnesota Vikings")
        self.assertEqual(vikings["team"]["division"], "NFC North")
        self.assertEqual({r["team"] for r in vikings["division"]}, {"MIN", "GB", "CHI", "DET"})
        stats = vikings["team_stats"]
        self.assertEqual((stats["wins"], stats["losses"], stats["ties"]), (1, 1, 0))
        self.assertEqual((stats["pf"], stats["pa"], stats["differential"]), (31, 24, 7))
        self.assertEqual(vikings["form"], ["W", "L"])
        self.assertEqual(len(vikings["elo_history"]), 2)
        self.assertEqual([g["opponent"] for g in vikings["schedule"]], ["GB", "DET", "CHI"])
        self.assertEqual(vikings["next_game"]["venue"], "Away")
        ratings, _ = elo_ratings([g for g in games if g.kind == "REG"], 2026, "MIN")
        self.assertAlmostEqual(vikings["next_game"]["win_probability"],
                               1 - home_probability(ratings["CHI"], ratings["MIN"]), places=4)
        self.assertEqual(vikings["projection"]["remaining"], 1)
        self.assertEqual(vikings["projection"]["total_games"], 3)
        self.assertTrue(all(r["ties"] == 0 for r in vikings["projection"]["distribution"]))
        playoffs = build_dashboard(games, 2026, True, 200, team="MIN")
        self.assertEqual(playoffs["team_stats"]["wins"], 2)
        self.assertEqual(playoffs["projection"], vikings["projection"])

    def test_team_selection_validation_and_default(self):
        self.assertEqual(build_dashboard(self.games, simulations=100)["team"]["code"], "PIT")
        with self.assertRaisesRegex(ValueError, "Choose"):
            build_dashboard(self.games, team="DAL")
        with self.assertRaisesRegex(ValueError, "No Vikings games"):
            build_dashboard(self.games, team="MIN")

    def test_projection_counts_selected_team_only(self):
        remaining = parse_games(CSV + VIKINGS_ROWS)
        remaining = [g for g in remaining if not g.completed]
        # Strong Vikings win on the road in every deterministic sample. Other games don't add wins.
        result = project_season(remaining, {"MIN": 3000, "CHI": 0}, 1, 1, 0, 100, team="MIN")
        self.assertEqual(result["expected_wins"], 2)
        self.assertEqual(result["distribution"], [{"wins": 2, "losses": 1, "ties": 0, "probability": 1.0}])


class CacheTests(unittest.TestCase):
    def test_offline_and_failed_refresh_preserve_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "games.csv"
            cache.write_text(CSV)
            games, metadata = ScheduleStore(cache, offline=True).load()
            self.assertEqual(len(games), 4)
            self.assertIn("Offline", metadata["warning"])
            with patch("steelers.data.urlopen", side_effect=URLError("offline")):
                games, metadata = ScheduleStore(cache).load(refresh=True)
            self.assertIn("Download unavailable", metadata["warning"])
            self.assertEqual(cache.read_text(), CSV)

    def test_no_cache_and_corrupt_cache_have_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "games.csv"
            with self.assertRaisesRegex(DataUnavailable, "Connect to the internet"):
                ScheduleStore(cache, offline=True).load()
            cache.write_text("invalid")
            with self.assertRaises(DataUnavailable):
                ScheduleStore(cache, offline=True).load()

    def test_bad_download_does_not_replace_valid_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "games.csv"
            cache.write_text(CSV)
            with patch("steelers.data.urlopen") as download:
                download.return_value.__enter__.return_value.read.return_value = b"broken"
                _, metadata = ScheduleStore(cache).load(refresh=True)
            self.assertTrue(metadata["warning"])
            self.assertEqual(cache.read_text(), CSV)


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name) / "games.csv"
        cls.path.write_text(CSV + VIKINGS_ROWS)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(ScheduleStore(cls.path, source_file=cls.path)))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.directory.cleanup()

    def test_dashboard_endpoint_and_input_validation(self):
        with urlopen(self.base + "/api/dashboard?season=2026") as response:
            body = json.load(response)
            self.assertEqual(body["team_stats"]["ties"], 1)
            self.assertIn("downloaded_at", body["data"])
        for query in ("season=banana", "season=1900", "phase=unknown", "team=DAL"):
            with self.assertRaises(HTTPError) as error:
                urlopen(self.base + "/api/dashboard?" + query)
            self.assertEqual(error.exception.code, 400)
            error.exception.close()

    def test_switching_team_does_not_reuse_the_other_teams_dashboard(self):
        def dashboard(team):
            with urlopen(self.base + f"/api/dashboard?season=2026&team={team}") as response:
                return json.load(response)
        steelers = dashboard("PIT")
        vikings = dashboard("MIN")
        self.assertEqual(vikings["team_stats"]["team"], "MIN")
        self.assertEqual(vikings["team_stats"]["ties"], 0)
        self.assertEqual(vikings["next_game"]["opponent"], "CHI")
        self.assertEqual(dashboard("PIT"), steelers)
        with urlopen(Request(self.base + "/api/refresh?season=2026&team=MIN", method="POST")) as response:
            self.assertEqual(json.load(response)["team_stats"]["team"], "MIN")

    def test_assets_are_restricted_to_public_files(self):
        with urlopen(self.base) as response:
            self.assertEqual(response.status, 200)
            self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])
        for path in ("/.cache/games.csv", "/app.py", "/../README.md"):
            with self.assertRaises(HTTPError) as error:
                urlopen(self.base + path)
            self.assertEqual(error.exception.code, 404)
            error.exception.close()

    def test_refresh_and_cross_origin_rejection(self):
        request = Request(self.base + "/api/refresh?season=2026", method="POST")
        with urlopen(request) as response:
            self.assertEqual(json.load(response)["season"], 2026)
        request.add_header("Origin", "https://example.com")
        with self.assertRaises(HTTPError) as error:
            urlopen(request)
        self.assertEqual(error.exception.code, 403)
        error.exception.close()


if __name__ == "__main__":
    unittest.main()
