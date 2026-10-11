"""Collect paired league-wide forecasts; optionally grade fixed forecast horizons.

python3 -m steelers.capture --once --season 2026 --offline
python3 -m steelers.capture --report --season 2026 --offline
"""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .advanced import AdvancedStore, evaluate_advanced
from .data import DataUnavailable, ScheduleStore
from .features import FeatureStore
from .forecast import ForecastArchive, HORIZONS, MODELS, kickoff_utc
from .matchup import evaluate
from .model import select_model
from .prediction import artifact, code_revision, freeze_inputs, local_sources, predict_game
from .provenance import SnapshotStore, utc
from .weather import WeatherStore

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / ".cache" / "forecasts.sqlite3"


def model_artifacts(inputs, season, as_of_utc):
    games = inputs.games_at(as_of_utc)
    config, selection = select_model([g for g in games if g.season < season], season)
    matchup, _ = evaluate(games, season, config, inputs.bundle, as_of_utc=as_of_utc)
    advanced, _ = evaluate_advanced(games, season, config, inputs.bundle, inputs.advanced, as_of_utc=as_of_utc)
    revision = code_revision()
    return {family: artifact(family, config, model, selection, revision, season) for family, model in
            (("elo", None), ("matchup", matchup), ("advanced", advanced))}


def collect(inputs, artifacts, archive, season, as_of_utc, policy_id=None, completed_at=None):
    """Calculate all choices together, then stamp actual collection completion.

    created_at is after calculation, so slow fitting cannot manufacture a
    forecast before a horizon deadline. completed_at is for controlled fixtures.
    """
    cutoff = utc(as_of_utc)
    archive.register_manifest(inputs.manifest_hash, inputs.manifest, inputs.snapshot_directory)
    archive.register_artifacts(artifacts)
    pending = []
    for game in inputs.games_at(cutoff):
        kickoff = kickoff_utc(game)
        if game.season == season and not game.completed and kickoff and cutoff < kickoff <= cutoff + timedelta(days=7):
            predictions = {family: predict_game(game, artifacts[family], inputs, cutoff) for family in MODELS}
            pending.append((game, predictions))
    created = utc(completed_at) if completed_at is not None else datetime.now(timezone.utc)
    if created < cutoff:
        raise ValueError("Collection completion cannot precede its input cutoff")
    captured, skipped = [], []
    for game, predictions in pending:
        if kickoff_utc(game) <= created:
            skipped.append({"id": game.id, "reason": "Kickoff passed while calculating; no capture written"})
            continue
        result = archive.save_collection(game, predictions, inputs.manifest_hash, created, policy_id)
        captured.append({"id": game.id, **result})
    return {"season": season, "input_cutoff": cutoff.isoformat(), "created_at": created.isoformat(),
            "manifest_hash": inputs.manifest_hash, "input_type": inputs.manifest["input_type"],
            "artifacts": {family: value["id"] for family, value in artifacts.items()},
            "fallbacks": {family: value["payload"]["fallback_reason"] for family, value in artifacts.items()},
            "policy_id": policy_id, "captured": captured, "skipped": skipped,
            "new_collections": sum(row["inserted"] for row in captured), "archive": str(archive.path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--once", action="store_true")
    action.add_argument("--report", action="store_true")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--data", type=Path, help="Fixture schedule; requires a separate --archive")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--snapshots", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--advanced", type=Path)
    parser.add_argument("--policy", type=Path, help="Frozen artifact allowlist for explicitly pooled comparisons")
    parser.add_argument("--freeze-policy", type=Path, help="Write a new frozen policy for these artifacts before capture")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    archive_path = args.archive or DEFAULT_ARCHIVE
    if args.data and (args.archive is None or archive_path.resolve() == DEFAULT_ARCHIVE.resolve()):
        parser.error("Fixture --data requires --archive pointing to a separate database")
    if args.data and args.snapshots and args.snapshots.resolve().is_relative_to((ROOT / ".cache" / "snapshots").resolve()):
        parser.error("Fixture snapshots must use a separate directory")
    if args.freeze_policy and (args.report or args.policy):
        parser.error("--freeze-policy is only valid with --once and without --policy")
    if args.freeze_policy and args.freeze_policy.exists():
        parser.error("The frozen policy already exists; use --policy to reuse it")
    protected = [archive_path, args.data, args.policy, args.freeze_policy, ROOT / ".cache" / "games.csv"]
    directories = [args.snapshots, args.features, args.advanced, ROOT / ".cache" / "snapshots",
                   ROOT / ".cache" / "features", ROOT / ".cache" / "advanced", ROOT / ".cache" / "weather",
                   archive_path.parent / (archive_path.stem + "-snapshots") if args.data else None]
    if args.output and (args.output.resolve() in {p.resolve() for p in protected if p} or
                        any(args.output.resolve().is_relative_to(p.resolve()) for p in directories if p)):
        parser.error("Output cannot replace an input snapshot, policy, or archive")
    try:
        schedule_store = ScheduleStore(ROOT / ".cache" / "games.csv", args.offline, args.data)
        games, metadata = schedule_store.load()
        if args.season not in {g.season for g in games}:
            raise ValueError("That season is not available in the schedule")
        archive = ForecastArchive(archive_path)
        if args.report:
            policy_id = archive.register_policy(json.loads(args.policy.read_text())) if args.policy else None
            result = archive.paired_report(games, args.season, policy_id)
        else:
            features = FeatureStore(args.features or ROOT / ".cache" / "features", args.offline or bool(args.data)) if not args.data or args.features else None
            advanced = AdvancedStore(args.advanced or ROOT / ".cache" / "advanced", args.offline or bool(args.data)) if not args.data or args.advanced else None
            bundle = features.load(args.season) if features else {"team": {}, "player": {}}
            now = datetime.now(timezone.utc)
            if advanced and not args.data:
                weather = WeatherStore(ROOT / ".cache" / "weather", args.offline)
                for game in games:
                    kickoff = kickoff_utc(game)
                    if game.season == args.season and not game.completed and kickoff and now < kickoff <= now + timedelta(days=7):
                        forecast = weather.load(game)
                        advanced.evidence.capture(game, bundle, forecast)
            data = advanced.load(args.season) if advanced else {}
            directory = args.snapshots or (archive_path.parent / (archive_path.stem + "-snapshots") if args.data else ROOT / ".cache" / "snapshots")
            snapshots = SnapshotStore(directory)
            sources = local_sources(schedule_store, features, advanced, bundle, args.season, metadata=metadata, data=data)
            inputs = freeze_inputs(snapshots, games, bundle, data, datetime.now(timezone.utc), sources, metadata,
                                   "fixture_reconstructed" if args.data else "prospective_local_snapshot")
            fitting_cutoff = datetime.now(timezone.utc)
            artifacts = model_artifacts(inputs, args.season, fitting_cutoff)
            archive.register_artifacts(artifacts)
            policy_id = None
            if args.freeze_policy:
                payload = {"version": "capture-policy-v1", "name": args.freeze_policy.stem,
                           "horizons": HORIZONS, "artifacts": {family: [a["id"]] for family, a in artifacts.items()}}
                policy_id = archive.register_policy(payload)
                args.freeze_policy.parent.mkdir(parents=True, exist_ok=True)
                with args.freeze_policy.open("x") as file:
                    file.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            elif args.policy:
                policy_id = archive.register_policy(json.loads(args.policy.read_text()))
            result = collect(inputs, artifacts, archive, args.season, datetime.now(timezone.utc), policy_id)
        content = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content)
        else:
            print(content, end="")
    except (ValueError, OSError, DataUnavailable) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
