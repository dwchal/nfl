"""Shared full-precision forecasts from pinned inputs and explicit UTC cutoffs."""

import json
import subprocess
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from pathlib import Path

from .advanced import AdvancedState, LABELS as ADVANCED_LABELS, VERSION as ADVANCED_VERSION
from .data import Game
from .evidence import Evidence, timestamp
from .features import QBWeek, TeamWeek
from .forecast import kickoff_utc
from .matchup import LABELS, VERSION as MATCHUP_VERSION, corrected, replay
from .model import BASE_RATING, ModelConfig, game_probability, replay_season, team_key
from .pbp import validate as validate_pbp
from .provenance import canonical, identity, sha256, utc


_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_CODE = {str(p.relative_to(_ROOT)): p.read_bytes() for p in [_ROOT / "app.py", *sorted((_ROOT / "steelers").glob("*.py"))]}


def _revision_at_startup():
    code = identity({name: sha256(raw) for name, raw in _SOURCE_CODE.items()})
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    return {"git_commit": commit, "source_sha256": code}


_REVISION = _revision_at_startup()


def code_revision():
    """Revision pinned when this process starts, even if files later change."""
    return dict(_REVISION)


def artifact(family, config, model=None, selection=None, revision=None, season=None):
    report = model.report if model else selection or {}
    labels = ADVANCED_LABELS if family == "advanced" else LABELS if family == "matchup" else ()
    available = family == "elo" or (model is not None and report.get("status") == "evaluated")
    payload = {"family": family, "version": {"elo": "elo-v1", "matchup": MATCHUP_VERSION, "advanced": ADVANCED_VERSION}[family],
               "feature_names": list(labels), "weights": list(model.weights) if model else [0.] * len(labels),
               "elo_config": asdict(config), "training_seasons": report.get("tuning_seasons", []),
               "fitting_options": {"penalty": .1 if family == "advanced" else .03 if family == "matchup" else None,
                                   "protocol": "annual-six-season" if family == "advanced" else "fixed-three-season" if family == "matchup" else "prior-six-season-selection",
                                   "comparison_seasons": report.get("test_seasons", []), "optimizer": report.get("optimizer"),
                                   "support": report.get("support"), "automatic_promotion": report.get("promoted", False)},
               "available": available, "fallback_reason": None if available else report.get("reason", "Model inputs unavailable; using Elo."),
               "forecast_season": season,
               "code_revision": revision or code_revision()}
    return {"id": identity(payload), "payload": payload}


def encode_inputs(games, bundle=None, data=None, metadata=None):
    bundle, data = bundle or {}, data or {}
    return {"code_sha256": code_revision()["source_sha256"], "games": [{**asdict(g), "day": g.day.isoformat()} for g in games],
            "bundle": {**{k: v for k, v in bundle.items() if k not in {"team", "player"}},
                       "team": [[list(key), asdict(value)] for key, value in sorted(bundle.get("team", {}).items())],
                       "player": [[list(key), [asdict(q) for q in values]] for key, values in sorted(bundle.get("player", {}).items())]},
            "advanced": {"plays": data.get("plays", {}), "sources": data.get("sources", []),
                         "evidence": [[list(key), rows] for key, rows in sorted(data.get("evidence", Evidence()).indexed.items())]},
            "metadata": metadata or {}}


@dataclass
class PredictionInputs:
    games: list
    bundle: dict
    advanced: dict
    manifest_hash: str
    manifest: dict
    snapshot_directory: str | None = None
    cache: dict = field(default_factory=dict)

    def games_at(self, as_of_utc):
        cutoff = utc(as_of_utc)
        if timestamp(self.manifest["available_at"]) > cutoff:
            raise ValueError("Inputs were first observed after the requested prediction time")
        # A future completed row cannot update ratings or feature state.
        return [replace(g, home_score=None, away_score=None) if g.completed and
                (kickoff_utc(g) is None or kickoff_utc(g) >= cutoff) else g for g in self.games]

    def runtime(self, model_artifact, as_of_utc, season=None):
        cutoff = utc(as_of_utc)
        season = season or model_artifact["payload"].get("forecast_season") or max(g.season for g in self.games)
        key = (model_artifact["id"], season, cutoff.isoformat())
        if key not in self.cache:
            games = self.games_at(cutoff)
            payload = model_artifact["payload"]
            config = ModelConfig(**payload["elo_config"])
            state = (AdvancedState(self.advanced.get("plays"), self.advanced.get("evidence"), now=cutoff)
                     if payload["family"] == "advanced" else None)
            _, state, _ = replay(games, season, config, self.bundle, state, as_of_utc=cutoff)
            ratings = replay_season(games, season, config)[0]
            self.cache[key] = (state, ratings)
        return self.cache[key]


def decode_inputs(payload, manifest_hash, manifest):
    games = [Game(**{**row, "day": date.fromisoformat(row["day"])}) for row in payload["games"]]
    bundle = {**payload["bundle"],
              "team": {tuple(key): TeamWeek(**value) for key, value in payload["bundle"]["team"]},
              "player": {tuple(key): [QBWeek(**q) for q in values] for key, values in payload["bundle"]["player"]}}
    data = {**payload["advanced"], "evidence": Evidence({tuple(key): rows for key, rows in payload["advanced"]["evidence"]})}
    bundle.setdefault("rosters", [])
    bundle.setdefault("injuries", [])
    return PredictionInputs(games, bundle, data, manifest_hash, manifest)


def freeze_inputs(store, games, bundle, data, as_of_utc, sources=(), metadata=None, input_type="prospective_local_snapshot"):
    payload = encode_inputs(games, bundle, data, metadata)
    if store:
        sources = [*sources, *(("code/" + name, raw, {"schema": "python-source"}) for name, raw in _SOURCE_CODE.items())]
        identifier, manifest = store.freeze(payload, sources, as_of_utc, input_type)
        # Decode the stored representation, so later in-memory mutations cannot
        # alter a forecast pinned to this manifest.
        stored, _ = store.load(identifier)
        inputs = decode_inputs(stored, identifier, manifest)
        inputs.snapshot_directory = str(store.directory.resolve())
        return inputs
    manifest = {"input_type": "unarchived_fixture", "available_at": utc(as_of_utc).isoformat(), "normalized_sha256": identity(payload)}
    return decode_inputs(json.loads(canonical(payload)), identity(manifest), manifest)


def predict_game(game, model_artifact, inputs, as_of_utc):
    cutoff = utc(as_of_utc)
    kickoff = kickoff_utc(game)
    if kickoff is None or cutoff >= kickoff:
        raise ValueError("Forecasts require a known future kickoff")
    if identity(model_artifact["payload"]) != model_artifact["id"]:
        raise ValueError("Model artifact does not match its identifier")
    if inputs.manifest.get("code_sha256") and model_artifact["payload"]["code_revision"].get("source_sha256") != inputs.manifest["code_sha256"]:
        raise ValueError("Model artifact and input manifest use different source code")
    observed = next((g for g in inputs.games_at(cutoff) if g.id == game.id), None)
    if observed is None or observed.completed:
        raise ValueError("Game is missing or already completed in the pinned snapshot")
    if observed != game:
        raise ValueError("Game does not match the pinned input snapshot")
    payload = model_artifact["payload"]
    state, ratings = inputs.runtime(model_artifact, cutoff, game.season)
    config = ModelConfig(**payload["elo_config"])
    probability = game_probability(ratings.get(team_key(game.home), BASE_RATING), ratings.get(team_key(game.away), BASE_RATING), game, config)
    base, features, coverage, evidence, reasons = probability, [], {}, {}, []
    if payload["family"] != "elo":
        features = state.features(game)
        coverage = state.coverage(game)
        if payload["family"] == "advanced":
            evidence = state.contexts[game.id]
        if not payload["available"]:
            reasons.append(payload["fallback_reason"])
        else:
            probability = corrected(base, features, payload["weights"])
    feature_digest = identity({"features": features, "coverage": coverage, "evidence": evidence, "elo_probability": base})
    return {"home_probability": probability, "elo_home_probability": base, "artifact_id": model_artifact["id"],
            "input_manifest_hash": inputs.manifest_hash, "feature_digest": feature_digest,
            "as_of_utc": cutoff.isoformat(), "evidence": evidence, "input_coverage": coverage,
            "fallback_reasons": reasons, "input_type": inputs.manifest["input_type"]}


def local_sources(schedule_store, feature_store, advanced_store, bundle, season, *, metadata=None, data=None):
    """Pin original cached bytes in addition to normalized, replayable inputs."""
    paths = [("schedule", schedule_store.source_file or schedule_store.cache,
              {"source_url": None if schedule_store.source_file else "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv", "schema": "nflverse-games",
               "expected_sha256": (metadata or {}).get("sha256"), "provider_timestamp": None})]
    directory = getattr(feature_store, "directory", None)
    if directory:
        paths.extend(("weekly/" + source["file"], directory / source["file"],
                      {"schema": source.get("schema"), "source_url": "https://github.com/nflverse/nflverse-data/releases",
                       "expected_sha256": source.get("sha256")}) for source in bundle.get("sources", []) if source.get("sha256"))
    directory = getattr(advanced_store, "directory", None)
    if directory:
        paths.extend((f"advanced/pbp_{year}.json", directory / f"pbp_{year}.json", {"schema": "situations-v1"}) for year in range(season - 8, season + 1))
    sources = []
    for name, path, metadata in paths:
        if path.exists():
            raw = path.read_bytes()
            expected = metadata.get("expected_sha256")
            if expected and sha256(raw) != expected:
                raise ValueError("Cached source changed during capture; retry collection")
            if name.startswith("advanced/pbp_"):
                year = int(path.stem.split("_")[-1])
                try:
                    payload = validate_pbp(json.loads(raw), year)
                except (ValueError, TypeError, KeyError):
                    if data and any(s.get("season") == year for s in data.get("sources", [])):
                        raise ValueError("PBP source changed during capture; retry collection")
                    sources.append((name, raw, {**metadata, "status": "invalid_ignored"}))
                    continue
                if data is not None and payload["games"] != {key: value for key, value in data.get("plays", {}).items() if key.startswith(f"{year}_")}:
                    raise ValueError("PBP source changed during capture; retry collection")
                metadata = {**metadata, "source_url": payload.get("source_url"), "provider_timestamp": None,
                            "retrieved_at_during_preparation": payload.get("retrieved_at"), "provider_file_sha256": payload.get("source_sha256")}
            sources.append((name, raw, metadata))
    return sources
