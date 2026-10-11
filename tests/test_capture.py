import copy
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from steelers.advanced import AdvancedModel, LABELS as ADVANCED_LABELS
from steelers.capture import collect, main
from steelers.evidence import Evidence
from steelers.forecast import ForecastArchive, HORIZONS, MODELS, kickoff_utc
from steelers.matchup import FeatureState, MatchupModel, replay
from steelers.model import BASELINE
from steelers.prediction import artifact, decode_inputs, freeze_inputs, predict_game
from steelers.provenance import SnapshotStore, identity
from test_matchup import bundle_for, fixture


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = ForecastArchive(self.root / "forecasts.sqlite3")
        self.snapshots = SnapshotStore(self.root / "snapshots")
        self.past = fixture("prior", day=1)
        self.game = fixture("next", week=2, day=8, home_score=None, away_score=None)
        self.clock = kickoff_utc(self.game) - timedelta(hours=26)
        self.games = [self.past, self.game]
        self.bundle = bundle_for(self.games)
        self.data = {"plays": {self.past.id: {team: [[2., 35.] for _ in range(8)] for team in ("PIT", "MIN")}}}
        self.inputs = self.freeze(self.games, self.bundle, self.data, self.clock)
        self.archive.register_manifest(self.inputs.manifest_hash, self.inputs.manifest, self.inputs.snapshot_directory)
        report = {"status": "evaluated", "tuning_seasons": [2020, 2021, 2022], "test_seasons": [2023, 2024, 2025]}
        matchup = MatchupModel((.1, -.2, .1, 0., .13, .2), FeatureState(), report)
        advanced = AdvancedModel(tuple([.1] * len(ADVANCED_LABELS)), FeatureState(), report)
        revision = {"git_commit": "fixture", "source_sha256": self.inputs.manifest["code_sha256"]}
        self.artifacts = {family: artifact(family, BASELINE, model, revision=revision, season=2026)
                          for family, model in (("elo", None), ("matchup", matchup), ("advanced", advanced))}
        self.archive.register_artifacts(self.artifacts)

    def freeze(self, games, bundle, data, clock):
        return freeze_inputs(self.snapshots, games, bundle, data, clock, input_type="fixture_reconstructed")

    def predictions(self, when, game=None, artifacts=None):
        game, artifacts = game or self.game, artifacts or self.artifacts
        return {family: {"home_probability": .6123456789012345, "artifact_id": a["id"],
                         "input_manifest_hash": self.inputs.manifest_hash, "as_of_utc": when.isoformat(),
                         "fallback_reasons": []} for family, a in artifacts.items()}

    def save(self, when, game=None, artifacts=None, policy=None):
        return self.archive.save_collection(game or self.game, self.predictions(when, game, artifacts), self.inputs.manifest_hash, when, policy)

    def report(self, games=None, policy=None):
        games = games or [replace(self.game, home_score=24, away_score=10)]
        return self.archive.paired_report(games, 2026, policy)

    def test_collect_pairs_models_on_one_clock_and_preserves_full_precision(self):
        result = collect(self.inputs, self.artifacts, self.archive, 2026, self.clock, completed_at=self.clock)
        self.assertEqual(result["new_collections"], 1)
        with self.archive.connect() as db:
            rows = db.execute("SELECT created_at,manifest_hash,home_probability,payload FROM forecast_runs").fetchall()
        self.assertEqual(len(rows), 3)
        self.assertEqual({r[0] for r in rows}, {self.clock.isoformat()})
        self.assertEqual({r[1] for r in rows}, {self.inputs.manifest_hash})
        self.assertTrue(any(r[2] != round(r[2], 4) for r in rows))
        for row in rows:
            self.assertEqual(row[2], json.loads(row[3])["home_probability"])

    def test_exact_float_survives_sqlite_storage(self):
        self.save(self.clock)
        with self.archive.connect() as db:
            value = db.execute("SELECT home_probability FROM forecast_runs LIMIT 1").fetchone()[0]
        self.assertEqual(value, .6123456789012345)

    def test_duplicate_retry_is_ignored_but_later_slot_is_preserved(self):
        first = self.save(self.clock)
        retry = self.save(self.clock + timedelta(minutes=1))
        later = self.save(self.clock + timedelta(minutes=16))
        self.assertTrue(first["inserted"])
        self.assertFalse(retry["inserted"])
        self.assertTrue(later["inserted"])
        self.assertEqual(first["collection_id"], retry["collection_id"])
        with self.archive.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM forecast_runs").fetchone()[0], 6)
            self.assertEqual(db.execute("SELECT min(created_at) FROM collection_runs").fetchone()[0], self.clock.isoformat())

    def test_incomplete_or_missing_artifact_collection_is_atomic(self):
        missing = self.predictions(self.clock)
        del missing["advanced"]
        with self.assertRaises(ValueError):
            self.archive.save_collection(self.game, missing, self.inputs.manifest_hash, self.clock)
        missing = self.predictions(self.clock)
        missing["advanced"]["artifact_id"] = "missing"
        with self.assertRaises(ValueError):
            self.archive.save_collection(self.game, missing, self.inputs.manifest_hash, self.clock)
        with self.archive.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM collection_runs").fetchone()[0], 0)

    def test_mismatched_manifest_or_clock_is_rejected(self):
        for key, value in (("input_manifest_hash", "wrong"), ("as_of_utc", (self.clock + timedelta(hours=1)).isoformat())):
            predictions = self.predictions(self.clock)
            predictions["advanced"][key] = value
            with self.assertRaises(ValueError):
                self.archive.save_collection(self.game, predictions, self.inputs.manifest_hash, self.clock)

    def test_actual_completion_controls_horizon_deadline(self):
        cutoff = kickoff_utc(self.game) - timedelta(hours=24)
        collect(self.inputs, self.artifacts, self.archive, 2026, cutoff,
                completed_at=cutoff + timedelta(microseconds=1))
        self.assertEqual(self.report()["horizons"]["24h"]["paired_games"], 0)

    def test_deadline_boundaries_and_latest_paired_run(self):
        for horizon, settings in HORIZONS.items():
            deadline = kickoff_utc(self.game) - timedelta(hours=settings["hours"])
            earliest = deadline - timedelta(minutes=settings["tolerance_minutes"])
            self.save(earliest)
            final = self.save(deadline)
            report = self.report()["horizons"][horizon]
            self.assertEqual(report["paired_games"], 1)
            self.assertEqual(report["groups"][0]["games"][0]["collection_id"], final["collection_id"])
            self.assertEqual(report["groups"][0]["models"]["elo"]["metrics"]["games"], 1)

    def test_stale_capture_and_capture_after_deadline_are_gaps(self):
        self.save(self.clock)
        revised_game = replace(self.game, kickoff="13:01", home_score=24, away_score=10)
        report = self.report([revised_game])["horizons"]["24h"]
        self.assertEqual(report["paired_games"], 0)
        self.assertEqual(report["gaps"][0]["reason"], "Latest paired capture is stale")

    def test_later_forecast_is_never_substituted_for_missed_deadline(self):
        self.save(kickoff_utc(self.game) - timedelta(hours=24) + timedelta(microseconds=1))
        report = self.report()["horizons"]["24h"]
        self.assertEqual(report["paired_games"], 0)
        self.assertEqual(report["gaps"][0]["reason"], "No paired capture at or before the deadline")

    def test_unarchived_input_manifest_is_rejected(self):
        with self.assertRaises(ValueError):
            self.archive.save_collection(self.game, self.predictions(self.clock), "unknown", self.clock)

    def test_revised_kickoff_rechecks_deadline_and_staleness(self):
        self.save(kickoff_utc(self.game) - timedelta(hours=24))
        earlier = replace(self.game, kickoff="11:00", home_score=24, away_score=10)
        later = replace(self.game, kickoff="16:00", home_score=24, away_score=10)
        for game in (earlier, later):
            self.assertEqual(self.report([game])["horizons"]["24h"]["paired_games"], 0)

    def test_unknown_kickoff_and_postkickoff_capture_are_rejected(self):
        for game, when in ((replace(self.game, kickoff=""), self.clock), (self.game, kickoff_utc(self.game))):
            with self.assertRaises(ValueError):
                self.save(when, game)

    def test_missing_artifact_in_existing_database_cannot_score(self):
        self.save(kickoff_utc(self.game) - timedelta(hours=24))
        with sqlite3.connect(self.archive.path) as db:
            db.execute("DELETE FROM model_artifacts WHERE family='advanced'")
        self.assertEqual(self.report()["horizons"]["24h"]["paired_games"], 0)

    def test_cutoff_blocks_own_future_stats_results_lineups_and_weather(self):
        original = {family: predict_game(self.game, a, self.inputs, self.clock) for family, a in self.artifacts.items()}
        later_game = fixture("later", week=3, day=9)
        changed = [self.past, replace(self.game, home_score=99, away_score=0), later_game]
        bundle = bundle_for(changed)
        for key in bundle["team"]:
            if key[1] >= 2:
                bundle["team"][key] = replace(bundle["team"][key], passing_epa=999.)
        data = copy.deepcopy(self.data)
        data["plays"][self.game.id] = {"PIT": [[999., 1.] for _ in range(8)]}
        data["evidence"] = Evidence({("confirmed_qb", self.game.id + ":PIT"): [
            {"available_at": (self.clock + timedelta(seconds=1)).isoformat(), "id": "future", "name": "Future", "status": "User-confirmed"}],
            ("weather", self.game.id): [{"available_at": (self.clock + timedelta(seconds=1)).isoformat(), "roof": "outdoors", "wind_mph": 100, "temperature_f": 0, "precipitation_inches": 10}]})
        altered = self.freeze(changed, bundle, data, self.clock)
        for family, a in self.artifacts.items():
            forecast = predict_game(self.game, a, altered, self.clock)
            self.assertEqual(original[family]["home_probability"], forecast["home_probability"])
            self.assertEqual(original[family]["feature_digest"], forecast["feature_digest"])

    def test_later_retrieved_corrections_cannot_be_backdated_and_old_snapshot_survives(self):
        original = predict_game(self.game, self.artifacts["matchup"], self.inputs, self.clock)
        changed = [replace(self.past, home_score=0, away_score=99), self.game]
        later = self.freeze(changed, bundle_for(changed), self.data, self.clock + timedelta(hours=1))
        with self.assertRaises(ValueError):
            predict_game(self.game, self.artifacts["matchup"], later, self.clock)
        payload, manifest = self.snapshots.load(self.inputs.manifest_hash)
        reloaded = decode_inputs(payload, self.inputs.manifest_hash, manifest)
        self.assertEqual(original, predict_game(self.game, self.artifacts["matchup"], reloaded, self.clock))

    def test_pending_flush_does_not_observe_todays_statistics(self):
        cutoff = kickoff_utc(self.past) + timedelta(hours=4)
        rows, state, _ = replay([self.past], 2026, BASELINE, self.bundle, as_of_utc=cutoff)
        self.assertEqual(len(rows), 1)
        self.assertFalse(state.observations)
        _, next_day, _ = replay([self.past], 2026, BASELINE, self.bundle, as_of_utc=cutoff + timedelta(days=1))
        self.assertEqual(next_day.observations["PIT"]["games"], 1)

    def test_aware_cutoff_and_exact_artifact_are_required(self):
        with self.assertRaises(ValueError):
            predict_game(self.game, self.artifacts["elo"], self.inputs, self.clock.replace(tzinfo=None))
        changed = copy.deepcopy(self.artifacts["elo"])
        changed["payload"]["elo_config"]["k_factor"] = 99
        with self.assertRaises(ValueError):
            predict_game(self.game, changed, self.inputs, self.clock)
        changed = copy.deepcopy(self.artifacts["elo"])
        changed["payload"]["code_revision"]["source_sha256"] = "different-source"
        changed["id"] = identity(changed["payload"])
        with self.assertRaises(ValueError):
            predict_game(self.game, changed, self.inputs, self.clock)

    def test_snapshot_retains_first_observed_time_and_source_metadata(self):
        first = self.snapshots.observe("provider", b"original", self.clock, {"schema": "test", "source_url": "https://example.test", "provider_timestamp": "provider-supplied"})
        retry = self.snapshots.observe("provider", b"original", self.clock + timedelta(days=1))
        revision = self.snapshots.observe("provider", b"corrected", self.clock + timedelta(days=1))
        self.assertEqual(first["first_retrieved_at"], retry["first_retrieved_at"])
        self.assertNotEqual(first["sha256"], revision["sha256"])
        self.assertEqual(self.snapshots.get(first["sha256"]), b"original")
        self.assertEqual(first["provider_timestamp"], "provider-supplied")
        spoof = self.snapshots.observe("provider", b"original", self.clock, {"first_retrieved_at": "1900", "sha256": "wrong"})
        self.assertEqual(spoof["first_retrieved_at"], first["first_retrieved_at"])
        self.assertEqual(spoof["sha256"], first["sha256"])

    def test_unavailable_model_gets_an_identifiable_fallback(self):
        unavailable = artifact("advanced", BASELINE, revision={"git_commit": "fixture", "source_sha256": self.inputs.manifest["code_sha256"]}, season=2026)
        forecast = predict_game(self.game, unavailable, self.inputs, self.clock)
        elo = predict_game(self.game, self.artifacts["elo"], self.inputs, self.clock)
        self.assertEqual(forecast["home_probability"], elo["home_probability"])
        self.assertEqual(forecast["artifact_id"], unavailable["id"])
        self.assertTrue(forecast["fallback_reasons"])

    def test_artifacts_are_separate_unless_frozen_policy_explicitly_pools_them(self):
        second_game = replace(self.game, id="other", home="BUF", away="NYJ")
        modified = copy.deepcopy(self.artifacts)
        modified["matchup"]["payload"]["weights"][0] += .1
        modified["matchup"]["id"] = identity(modified["matchup"]["payload"])
        self.archive.register_artifacts(modified)
        deadline = kickoff_utc(self.game) - timedelta(hours=24)
        self.save(deadline)
        self.save(deadline, second_game, modified)
        completed = [replace(g, home_score=24, away_score=10) for g in (self.game, second_game)]
        report = self.report(completed + completed)["horizons"]["24h"]
        self.assertEqual(report["paired_games"], 2)
        self.assertEqual(len(report["groups"]), 2)
        payload = {"version": "capture-policy-v1", "name": "frozen-fixture", "horizons": HORIZONS,
                   "artifacts": {family: sorted({self.artifacts[family]["id"], modified[family]["id"]}) for family in MODELS}}
        policy = self.archive.register_policy(payload)
        self.save(deadline, policy=policy)
        self.save(deadline, second_game, modified, policy)
        pooled = self.report(completed, policy)["horizons"]["24h"]
        self.assertEqual(pooled["paired_games"], 2)
        self.assertEqual(len(pooled["groups"]), 1)
        self.assertEqual(len(pooled["groups"][0]["artifacts"]["matchup"]), 2)

    def test_legacy_rows_survive_schema_upgrade(self):
        legacy = self.root / "legacy.sqlite3"
        with sqlite3.connect(legacy) as db:
            db.execute("CREATE TABLE forecasts (fingerprint TEXT PRIMARY KEY, game_id TEXT, team TEXT, season INTEGER, choice TEXT, created TEXT, kickoff TEXT, probability REAL, payload TEXT)")
            db.execute("INSERT INTO forecasts VALUES (?,?,?,?,?,?,?,?,?)", ("legacy", "next", "PIT", 2026, "auto", self.clock.isoformat(), kickoff_utc(self.game).isoformat(), .6, "{}"))
            before = db.execute("SELECT * FROM forecasts").fetchall()
        with ForecastArchive(legacy).connect() as db:
            self.assertEqual(db.execute("SELECT * FROM forecasts").fetchall(), before)
            self.assertEqual(db.execute("SELECT count(*) FROM forecast_runs").fetchone()[0], 0)
        self.assertEqual(ForecastArchive(legacy).paired_report([], 2026)["legacy_records"], 1)


class FixtureCLITests(unittest.TestCase):
    def test_fixture_requires_separate_archive_before_any_file_is_opened(self):
        with patch("sys.argv", ["capture", "--once", "--season", "2026", "--data", "fixture.csv"]), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 2)

    def test_fixture_capture_is_isolated_and_marks_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schedule, archive, real = root / "fixture.csv", root / "fixture.sqlite3", root / "real.sqlite3"
            future = datetime.now(timezone.utc) + timedelta(days=2)
            schedule.write_text("game_id,season,game_type,week,gameday,gametime,home_team,away_team,home_score,away_score\n"
                                f"fixture,{future.year},REG,1,{future.date()},13:00,PIT,MIN,,\n")
            stdout = io.StringIO()
            argv = ["capture", "--once", "--offline", "--season", str(future.year), "--data", str(schedule), "--archive", str(archive)]
            with patch("sys.argv", argv), patch("steelers.capture.DEFAULT_ARCHIVE", real), redirect_stdout(stdout):
                main()
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["new_collections"], 1)
            self.assertEqual(result["input_type"], "fixture_reconstructed")
            self.assertFalse(real.exists())
            self.assertTrue((root / "fixture-snapshots").exists())
            with sqlite3.connect(archive) as db:
                self.assertEqual(db.execute("SELECT count(*) FROM forecast_runs").fetchone()[0], 3)
                self.assertTrue(all(json.loads(row[0])["fallback_reasons"] for row in db.execute("SELECT payload FROM forecast_runs WHERE family!='elo'")))


if __name__ == "__main__":
    unittest.main()
