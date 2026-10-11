"""Local-only, nested chronological comparisons of Elo, QB and advanced models.

Run with --help for paths. This reconstructs retrospective kickoff forecasts;
it does not download inputs, write evidence, or promote production models.
"""

import argparse
import hashlib
import json
import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .advanced import AdvancedState, FEATURE_GROUPS, LABELS as ADVANCED_LABELS, enabled_indices, group_indices
from .context_research import pick_comparison
from .coverage import MIN_COVERAGE, MIN_GAMES, sufficient_history, training_coverage
from .data import parse_games
from .evaluation import paired_uncertainty
from .evidence import Evidence, timestamp
from .features import VERSION as WEEKLY_VERSION, parse_feature_csv
from .forecast import kickoff_utc
from .matchup import LABELS as MATCHUP_LABELS, corrected_mixture, fit_correction, replay
from .model import ModelConfig, chronological_forecasts, metrics, reliability_bins
from .optimization import CLIP_BOUND
from .pbp import validate as validate_pbp

# v1: Elo, QB matchup and full advanced candidates. v2 adds declared advanced
# feature groups and an optional Elo probability blend chosen on inner folds.
VERSIONS = ("chronological-v1", "chronological-v2")
VERSION = VERSIONS[-1]
CUTOFF_POLICY = "Kickoff; weekly/PBP observations from earlier calendar days only."


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_config(config):
    """Reject unsupported protocol changes rather than silently ignoring them."""
    required = {"version", "history_start", "evaluation_seasons", "cutoff_policy", "inner_seasons", "min_training_seasons",
                "max_training_seasons", "minimum_games", "minimum_coverage",
                "bootstrap_samples", "seed", "gate", "candidates"}
    version = config.get("version")
    if version not in VERSIONS:
        raise ValueError("Unsupported chronological protocol version")
    extended = version != VERSIONS[0]
    optional = {"blend"} if extended else set()
    if not required <= set(config) <= required | optional:
        raise ValueError("Experiment config must contain exactly: " + ", ".join(sorted(required))
                         + (" and optionally: " + ", ".join(sorted(optional)) if optional else ""))
    fixed = {"cutoff_policy": "kickoff_previous_day", "inner_seasons": 2, "min_training_seasons": 3,
             "max_training_seasons": 6, "minimum_games": MIN_GAMES,
             "minimum_coverage": MIN_COVERAGE, "seed": 42, "gate": "both_probability_scores"}
    if any(config[key] != value for key, value in fixed.items()):
        raise ValueError("Unsupported chronological protocol settings")
    if "blend" in config:
        blend = config["blend"]
        alphas = blend.get("alphas") if isinstance(blend, dict) else None
        if (set(blend) != {"alphas"} or not isinstance(alphas, list) or len(alphas) < 2
                or any(type(a) not in (int, float) or not 0 <= a <= 1 for a in alphas)
                or alphas != sorted(set(alphas)) or alphas[0] != 0):
            raise ValueError("blend.alphas must be distinct ascending values in [0, 1] starting at 0")
    if type(config["history_start"]) is not int or config["history_start"] < 1999:
        raise ValueError("history_start must be a season from 1999 onward")
    years = config["evaluation_seasons"]
    if not isinstance(years, list) or not years or any(type(y) is not int or y < config["history_start"] for y in years) or years != sorted(set(years)):
        raise ValueError("evaluation_seasons must be sorted unique seasons from history_start onward")
    if type(config["bootstrap_samples"]) is not int or config["bootstrap_samples"] < 1:
        raise ValueError("bootstrap_samples must be positive")
    candidates = config["candidates"]
    if not isinstance(candidates, list) or not candidates or candidates[0] != {"id": "elo", "family": "elo"}:
        raise ValueError("The first candidate must be the Elo incumbent")
    seen = set()
    for candidate in candidates:
        family, identifier = candidate.get("family"), candidate.get("id")
        if family not in {"elo", "matchup", "advanced"} or not isinstance(identifier, str) or not identifier:
            raise ValueError("Invalid candidate family or identifier")
        if identifier in seen or (family == "elo" and identifier != "elo"):
            raise ValueError("Candidate identifiers must be unique; Elo is listed once")
        seen.add(identifier)
        keys = {"id", "family"} if family == "elo" else {"id", "family", "penalty"}
        if family == "advanced" and extended:
            keys |= {"features", "starters"}
        if not keys - {"features", "starters"} <= set(candidate) <= keys:
            raise ValueError("Invalid candidate settings")
        if candidate.get("features", "full") not in FEATURE_GROUPS:
            raise ValueError("Unknown advanced feature group: " + str(candidate.get("features")))
        if candidate.get("starters", "deterministic") not in {"deterministic", "mixture"}:
            raise ValueError("starters must be deterministic or mixture")
        if family != "elo" and (type(candidate["penalty"]) not in (int, float)
                                or not math.isfinite(candidate["penalty"]) or candidate["penalty"] <= 0):
            raise ValueError("Correction penalties must be finite and positive")
    return config


def load_inputs(schedule, features, advanced, history_start, end):
    """Read immutable local snapshots, including SQLite through a read-only URI.

    Optional absent/invalid sources are recorded and left absent. The adapters
    then score Elo fallbacks on the same games instead of deleting observations.
    """
    sources = []

    def read(path, name):
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            sources.append({"file": name, "sha256": None, "status": "missing"})
            return None
        sources.append({"file": name, "sha256": hashlib.sha256(raw).hexdigest(), "status": "loaded"})
        return raw

    raw = read(schedule, "schedule")
    if raw is None:
        raise ValueError("A local schedule is required")
    games = parse_games(raw.decode("utf-8-sig"))
    bundle = {"team": {}, "player": {}, "schema": WEEKLY_VERSION}
    plays, rushing, evidence = {}, {}, Evidence()
    for year in range(max(1999, history_start - 8), end + 1):
        for kind, prefix in (("team", "stats_team"), ("player", "stats_player")):
            filename = f"{prefix}_week_{year}.csv"
            content = read(features / filename, "weekly/" + filename)
            if content is not None:
                try:
                    bundle[kind].update(parse_feature_csv(content.decode("utf-8-sig"), kind, year))
                except (ValueError, UnicodeError) as error:
                    sources[-1].update(status="invalid", reason=str(error))
        if advanced is not None:
            filename = f"pbp_{year}.json"
            content = read(advanced / filename, "advanced/" + filename)
            if content is not None:
                try:
                    payload = validate_pbp(json.loads(content), year)
                    plays.update(payload["games"])
                    rushing.update(payload.get("qb_rushing", {}))
                except (ValueError, TypeError, KeyError) as error:
                    sources[-1].update(status="invalid", reason=str(error))
    if advanced is not None:
        path = advanced / "evidence.sqlite3"
        if path.exists():
            # Hash logical rows, including WAL-visible records, rather than the
            # SQLite container bytes or modification times.
            try:
                with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
                    records = db.execute("SELECT kind,key,available,payload,fingerprint FROM evidence ORDER BY available,rowid").fetchall()
                indexed = {}
                for kind, key, available, payload, _ in records:
                    timestamp(available)
                    indexed.setdefault((kind, key), []).append({**json.loads(payload), "available_at": available})
                evidence = Evidence(indexed, digest(records))
                sources.append({"file": "advanced/evidence.sqlite3", "sha256": evidence.digest, "status": "loaded",
                                "hash_policy": "Logical ordered evidence rows, including committed WAL records"})
            except (sqlite3.DatabaseError, ValueError, TypeError) as error:
                sources.append({"file": "advanced/evidence.sqlite3", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "status": "invalid", "reason": str(error)})
        else:
            sources.append({"file": "advanced/evidence.sqlite3", "sha256": None, "status": "missing"})
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}
    manifest = {"sources": sources, "code": code, "weekly_schema": WEEKLY_VERSION}
    return games, bundle, {"plays": plays, "qb_rushing": rushing, "evidence": evidence}, manifest


class ReplayRows:
    """Stable per-season features: extending the audit never changes warmup.

    Each year has its own eight-year feature warmup and three-year Elo warmup.
    Its Elo selection receives strictly earlier games. Source data from later
    games may be present in memory but never enters that game's pregame state.
    """

    def __init__(self, games, bundle, data):
        self.games, self.bundle, self.data = games, bundle, data
        self.configs, self.cache, self.contexts = {}, {}, {}

    def rows(self, family, year):
        key = (family, year)
        if key not in self.cache:
            if ("elo", year) not in self.cache:
                forecasts, configs = chronological_forecasts(self.games, [year])
                self.configs.update(configs)
                self.cache[("elo", year)] = [r for r in forecasts.values() if r["kind"] == "REG"]
            if family != "elo":
                # Reconstruct at kickoff independently of the wall clock used
                # by the live dashboard's min(kickoff, now) cutoff.
                state = (AdvancedState(self.data.get("plays"), self.data.get("evidence"),
                                       now=datetime.max.replace(tzinfo=timezone.utc),
                                       qb_rushing=self.data.get("qb_rushing")) if family == "advanced" else None)
                rows, state, _ = replay([g for g in self.games if g.season <= year], year,
                                        ModelConfig(**self.configs[str(year)]), self.bundle, state)
                self.cache[key] = [{**r, "elo_config": self.configs[str(year)]} for r in rows
                                   if r["season"] == year and r["kind"] == "REG"]
                if family == "advanced":
                    self.contexts.update({r["id"]: state.contexts[r["id"]] for r in self.cache[key]})
        return self.cache[key]


@dataclass
class Fitted:
    weights: tuple
    report: dict


class Adapter:
    """Fit specified previous seasons without running another evaluation.

    v1 advanced candidates mean the passing-only label set with deterministic
    starters, so earlier reports keep their meaning; v2 declares both.
    """

    def __init__(self, candidate, replay_rows, protocol):
        self.candidate, self.replay, self.protocol = candidate, replay_rows, protocol
        self.family = candidate["family"]
        legacy = protocol["version"] == VERSIONS[0]
        self.group = candidate.get("features", "full_passing" if legacy else "full")
        self.mixture = candidate.get("starters", "deterministic") == "mixture"
        self.cache = {}

    def fit(self, previous_years):
        key = tuple(previous_years)
        if key in self.cache:
            return self.cache[key]
        report = {"candidate": self.candidate, "training_seasons": [], "excluded_seasons": [],
                  "coverage": {}, "optimizer": None, "support": None, "fallback_reason": None}
        weights = ()
        if self.family != "elo":
            width = len(ADVANCED_LABELS) if self.family == "advanced" else 6
            weights = (0.,) * width
            eligible = []
            for year in previous_years:
                coverage = training_coverage(self.replay.rows(self.family, year), [year], self.family == "advanced")
                report["coverage"].update(coverage)
                if sufficient_history(coverage, self.family == "advanced"):
                    eligible.append(year)
                else:
                    report["excluded_seasons"].append(year)
            years = eligible[-self.protocol["max_training_seasons"]:]
            report["training_seasons"] = years
            if len(years) < self.protocol["min_training_seasons"]:
                report["fallback_reason"] = "Fewer than three complete earlier seasons satisfy source coverage."
            else:
                train = [r for year in years for r in self.replay.rows(self.family, year)]
                indices = tuple(range(width))
                labels = MATCHUP_LABELS
                if self.family == "advanced":
                    labels = ADVANCED_LABELS
                    indices, report["support"] = enabled_indices(train, self.replay.contexts)
                    # Support gates still apply; a declared group can only remove coefficients.
                    declared = set(group_indices(self.group))
                    indices = tuple(i for i in indices if i in declared)
                if any(len(r["features"]) != len(labels) for r in train):
                    raise ValueError("Training rows do not match the feature registry width")
                weights, report["optimizer"] = fit_correction(train, indices, self.candidate["penalty"])
                if not report["optimizer"]["converged"]:
                    report["fallback_reason"] = "Correction optimizer did not converge."
                report["features"] = [labels[i] for i in indices]
                report["training_rows"] = len(train)
                report["clipping"] = {labels[i]: sum(abs(r["features"][i]) > CLIP_BOUND for r in train) / len(train)
                                      for i in indices} if train else {}
        report["active_coefficients"] = sum(w != 0 for w in weights)
        report["weights"] = list(weights)
        report["fit_id"] = digest(report)
        fitted = Fitted(weights, report)
        self.cache[key] = fitted
        return fitted

    def predict(self, year, fitted):
        predictions = []
        for row in self.replay.rows(self.family, year):
            reason = fitted.report["fallback_reason"]
            if self.family != "elo" and reason is None:
                teams = list(row["input_coverage"]["teams"].values())
                if not all(t["weekly"]["games"] > 0 for t in teams):
                    reason = "No earlier weekly observations for one or both teams."
                elif self.family == "advanced" and not all(t["pbp"]["games"] > 0 for t in teams):
                    reason = "No earlier PBP observations for one or both teams."
            probability = row["probability"]
            if self.family != "elo" and reason is None:
                probability = corrected_mixture(probability, row["features"], row.get("scenarios") if self.mixture else None, fitted.weights)
            predictions.append({**row, "probability": probability, "fallback_reason": reason,
                                "fit_id": fitted.report["fit_id"]})
        return predictions


def comparison(baseline, candidate, samples):
    if not baseline:
        return None
    availability = {}
    if any("input_coverage" in row for row in candidate):
        for signal in ("weekly", "quarterback", "pbp", "weather", "availability", "qb_rushing"):
            available, missing = [], []
            for row in candidate:
                coverage = row.get("input_coverage", {})
                if signal in {"weather", "availability", "qb_rushing"}:
                    if signal not in coverage.get("missing", {}):
                        continue
                    present = not coverage["missing"][signal]
                else:
                    teams = list(coverage.get("teams", {}).values())
                    if len(teams) != 2 or signal not in teams[0]:
                        continue
                    field = "dropbacks" if signal == "quarterback" else "games"
                    present = all(t[signal][field] > 0 for t in teams)
                (available if present else missing).append(row)
            availability[signal] = {"available": metrics(available), "missing": metrics(missing)}
    return {"metrics": metrics(candidate), "reliability": reliability_bins(candidate),
            "by_input_availability": availability,
            "by_team": {t: metrics(candidate, t) for t in ("PIT", "MIN")},
            "uncertainty": paired_uncertainty(baseline, candidate, samples),
            "winner_changes": pick_comparison(baseline, candidate, samples) if any(r["result"] != .5 for r in baseline) else None,
            "by_input_coverage": {status: metrics([r for r in candidate if bool(r.get("fallback_reason")) == missing])
                                  for status, missing in (("model_applied", False), ("elo_fallback", True))},
            "fallback_games": sum(r.get("fallback_reason") is not None for r in candidate)}


def run(games, bundle, data, start, end, config, manifest=None):
    config = validate_config(config)
    if start > end or any(year not in config["evaluation_seasons"] for year in range(start, end + 1)):
        raise ValueError("Evaluation range must be declared in evaluation_seasons and have start <= end")
    replay_rows = ReplayRows(games, bundle, data)
    adapters = [Adapter(c, replay_rows, config) for c in config["candidates"]]
    seasons = sorted({g.season for g in games if config["history_start"] <= g.season <= end})
    complete = [year for year in seasons if len(target := [g for g in games if g.season == year and g.kind == "REG"]) >= MIN_GAMES
                and all(g.completed for g in target)]
    manifest = manifest or {}
    input_hashes = {"manifest": digest(manifest), "experiment": digest(config)}
    alphas = config.get("blend", {}).get("alphas")
    protocol_extension = ("" if alphas is None else
                          " Advanced candidates may declare a feature group; support gates still apply. After selection, a blend"
                          " p = (1 - alpha) * p_elo + alpha * p_selected is chosen on the same inner predictions, preferring smaller"
                          " alpha and requiring both scores to beat alpha 0, then frozen for the outer season.")
    report = {"version": config["version"], "runner": VERSION, "config": config, "evaluation_seasons": list(range(start, end + 1)),
              "input_hashes": input_hashes, "inputs": manifest,
              "input_type": "Retrospective reconstructed kickoff forecasts, not archived live forecasts.",
              "caveat": "Previously explored development years; not an untouched test. Provider revisions and model-development selection remain. Bootstrap intervals do not account for development selection.",
              "protocol": "For each outer season, select on the last two earlier complete seasons. Each candidate fits up to six earlier complete seasons meeting its source coverage, minimum three. Rank pooled inner Brier, then log loss, then declared candidate order; require both scores to beat Elo. Refit before the outer season. All candidates score identical games, including fallbacks. No production promotion." + protocol_extension,
              "cutoff_policy": CUTOFF_POLICY, "folds": [], "skipped": [], "fits": {}, "games": []}
    pooled = {a.candidate["id"]: [] for a in adapters}
    selected_rows, blended_rows = [], []
    game_index = {g.id: g for g in games}
    families = {}
    for adapter in adapters:
        families.setdefault(adapter.family, adapter.candidate["id"])

    def blended(elo_rows, rows, alpha):
        # alpha 0 reproduces Elo exactly and alpha 1 the correction exactly.
        return [{**r, "probability": (1 - alpha) * e["probability"] + alpha * r["probability"]}
                for e, r in zip(elo_rows, rows)]

    def choose_alpha(elo_rows, rows):
        """Pick the blend on earlier inner predictions only; smaller alpha wins ties."""
        scores = {alpha: metrics(blended(elo_rows, rows, alpha)) for alpha in alphas}
        best = min(alphas, key=lambda alpha: (scores[alpha]["brier"], scores[alpha]["log_loss"], alpha))
        if best and not (scores[best]["brier"] < scores[0]["brier"] and scores[best]["log_loss"] < scores[0]["log_loss"]):
            best = 0
        return best, [{"alpha": alpha, "metrics": scores[alpha]} for alpha in alphas]

    def predict(adapter, year):
        previous = [y for y in complete if y < year]
        fitted = adapter.fit(previous)
        report["fits"][fitted.report["fit_id"]] = fitted.report
        predictions = adapter.predict(year, fitted)
        if [r["id"] for r in predictions] != [r["id"] for r in replay_rows.rows("elo", year)]:
            raise ValueError("Adapters must forecast identical ordered games in inner and outer folds")
        return predictions, fitted.report

    for year in range(start, end + 1):
        if year not in complete:
            report["skipped"].append({"season": year, "reason": "Not a complete regular season with at least 200 listed games; excluded from headline metrics."})
            continue
        inner_years = [y for y in complete if y < year][-config["inner_seasons"]:]
        inner, selection_scores, inner_predictions = {}, {}, {}
        for adapter in adapters:
            identifier = adapter.candidate["id"]
            inner_rows, folds = [], []
            for validation in inner_years:
                predictions, fit = predict(adapter, validation)
                inner_rows.extend(predictions)
                folds.append({"season": validation, "training_seasons": fit["training_seasons"],
                              "fit_id": fit["fit_id"], "metrics": metrics(predictions),
                              "fallback_games": sum(r["fallback_reason"] is not None for r in predictions)})
            inner[identifier] = {"folds": folds, "metrics": metrics(inner_rows)}
            selection_scores[identifier] = metrics(inner_rows)
            inner_predictions[identifier] = inner_rows
        selected = 0
        reason = "Insufficient earlier complete inner validation seasons; use Elo."
        ranked = "elo"
        if len(inner_years) == config["inner_seasons"]:
            best = min(range(len(adapters)), key=lambda i: (selection_scores[adapters[i].candidate["id"]]["brier"],
                                                           selection_scores[adapters[i].candidate["id"]]["log_loss"], i))
            ranked = adapters[best].candidate["id"]
            incumbent, challenger = selection_scores["elo"], selection_scores[ranked]
            if best and challenger["brier"] < incumbent["brier"] and challenger["log_loss"] < incumbent["log_loss"]:
                selected, reason = best, "Selected on earlier inner seasons; both probability scores improve."
            else:
                reason = "No candidate beats Elo on both pooled inner probability scores; use Elo."
        outer, fits = {}, {}
        for adapter in adapters:
            identifier = adapter.candidate["id"]
            outer[identifier], fits[identifier] = predict(adapter, year)
            pooled[identifier].extend(outer[identifier])
        chosen = adapters[selected].candidate
        selected_rows.extend(outer[chosen["id"]])
        fold = {"season": year, "elo_config": replay_rows.configs[str(year)],
                "inner_seasons": inner_years, "inner": inner, "ranked_candidate": ranked,
                "selected_config": chosen, "selection_reason": reason,
                "fits": {key: fit["fit_id"] for key, fit in fits.items()},
                "metrics": {key: metrics(rows) for key, rows in outer.items()},
                "selected_metrics": metrics(outer[chosen["id"]])}
        blended_outer = None
        if alphas is not None:
            alpha, inner_blends = 0, None
            if selected:
                alpha, inner_blends = choose_alpha(inner_predictions["elo"], inner_predictions[chosen["id"]])
                blend_reason = ("Blend chosen on the same inner seasons; alpha 0 retained unless a larger alpha improves both scores."
                                if alpha else "No blend improves both inner probability scores over Elo; alpha 0.")
            else:
                blend_reason = "Elo selected; nothing to blend."
            blended_outer = blended(outer["elo"], outer[chosen["id"]], alpha)
            blended_rows.extend(blended_outer)
            fold["blend"] = {"alpha": alpha, "reason": blend_reason, "inner": inner_blends, "metrics": metrics(blended_outer)}
        report["folds"].append(fold)
        for index, incumbent in enumerate(outer["elo"]):
            identifier = incumbent["id"]
            cutoff = kickoff_utc(game_index[identifier])
            candidates = {key: rows[index] for key, rows in outer.items()}
            row = {**{key: incumbent[key] for key in ("id", "season", "week", "home", "away", "result")},
                   "cutoff": cutoff.isoformat() if cutoff else None, "cutoff_policy": CUTOFF_POLICY,
                   "incumbent_probability": incumbent["probability"],
                   "candidate_probabilities": {key: row["probability"] for key, row in candidates.items()},
                   "selected_probability": candidates[chosen["id"]]["probability"], "selected_config": chosen,
                   "elo_config": replay_rows.configs[str(year)],
                   "fit_ids": {key: fit["fit_id"] for key, fit in fits.items()},
                   "fallback_reasons": {key: row["fallback_reason"] for key, row in candidates.items()},
                   # v2 keys coverage by family: candidates in a family share one replay row.
                   "coverage": ({key: row.get("input_coverage", {}) for key, row in candidates.items()}
                                if config["version"] == VERSIONS[0] else
                                {family: candidates[key].get("input_coverage", {}) for family, key in families.items()}),
                   "input_hashes": input_hashes}
            if blended_outer is not None:
                row.update(blend_alpha=fold["blend"]["alpha"], blended_probability=blended_outer[index]["probability"])
            report["games"].append(row)
    baseline = pooled["elo"]
    report["elo_configs"] = replay_rows.configs
    report["baseline"] = metrics(baseline)
    report["candidates"] = {key: comparison(baseline, rows, config["bootstrap_samples"]) for key, rows in pooled.items()}
    report["selected_policy"] = comparison(baseline, selected_rows, config["bootstrap_samples"])
    if alphas is not None:
        report["blended_policy"] = comparison(baseline, blended_rows, config["bootstrap_samples"])
    return report


def encode_report(report):
    """Pretty summary followed by one JSON line per game, in a single document."""
    summary = {key: value for key, value in report.items() if key != "games"}
    header = json.dumps(summary, indent=2, allow_nan=False)
    rows = ",\n".join("    " + json.dumps(row, separators=(",", ":"), allow_nan=False) for row in report["games"])
    return header[:-2] + ',\n  "games": [\n' + rows + '\n  ]\n}\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--advanced", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = validate_config(json.loads(args.config.read_text()))
    if args.start > args.end or any(year not in config["evaluation_seasons"] for year in range(args.start, args.end + 1)):
        parser.error("Require --start <= --end and seasons declared in the experiment config")
    if args.output.resolve() in {args.data.resolve(), args.config.resolve()} or any(
            args.output.resolve().is_relative_to(p.resolve()) for p in (args.features, args.advanced) if p is not None):
        parser.error("Output must not replace an input snapshot")
    games, bundle, data, manifest = load_inputs(args.data, args.features, args.advanced, config["history_start"], args.end)
    manifest["experiment_file_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    report = run(games, bundle, data, args.start, args.end, config, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encode_report(report))
    print(f"Scored {len(report['games'])} regular-season games; report: {args.output}")


if __name__ == "__main__":
    main()
