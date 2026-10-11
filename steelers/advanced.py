"""Pregame QB/availability, situational PBP, weather-style and travel model."""

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import pbp, travel
from .evidence import Evidence, EvidenceStore, availability_complete, timestamp, top_record
from .coverage import sufficient_history, training_coverage
from .evaluation import paired_uncertainty
from .features import feature_key
from .forecast import kickoff_utc
from .matchup import (FeatureState, MatchupModel, LABELS as BASE_LABELS,
                      corrected, corrected_mixture, fit_correction, fit_diagnostic, replay)
from .model import historical_offsets, metrics, select_model, team_key
from .pbp import PBPStore

VERSION = "advanced-v6"
LABELS = BASE_LABELS + tuple(f"Situational {side}: {label}" for side in ("offense", "defense") for label in pbp.LABELS) + travel.LABELS + (
    "Wind × passing reliance", "Rain × passing reliance", "Cold × passing reliance",
    "Unavailable receivers", "Unavailable offensive line", "Unavailable defenders", "Questionable players",
    "Quarterback rushing change")
WEATHER_START = 6 + 2 * len(pbp.LABELS) + len(travel.LABELS)
AVAILABILITY_START = WEATHER_START + 3
RUSHING = AVAILABILITY_START + 4
assert len(LABELS) == RUSHING + 1 and len(set(LABELS)) == len(LABELS), "Advanced label layout changed"
# Experiment settings for QB rushing, not established optimum values.
RUSHING_PRIOR_CARRIES = 50.
RUSHING_SCALE = .15
STARTER_CLASSES = ("confirmed", "projected", "previous")
STARTER_CATEGORIES = ("rank2", "rank3", "unknown")
MIN_STARTER_EXAMPLES = 100

# Versioned name-to-index registry. Experiments declare feature groups by label
# so an old weight vector is never reinterpreted under a changed layout.
REGISTRY = {label: index for index, label in enumerate(LABELS)}
SITUATIONAL_LABELS = tuple(LABELS[6:6 + 2 * len(pbp.LABELS)])
FEATURE_GROUPS = {
    "full": LABELS,
    "full_passing": LABELS[:RUSHING],
    # Reduced groups omit the extra rest coefficient: calibrated Elo supplies it.
    "qb_weekly_travel": ("Passing offense", "Passing defense", "Rushing offense", "Rushing defense",
                         "Quarterback change") + travel.LABELS,
    "qb_situational_travel": ("Quarterback change",) + SITUATIONAL_LABELS + travel.LABELS,
    "travel": travel.LABELS,
}


def group_indices(name):
    """Resolve a declared feature group to registry indices; unknown labels fail loudly."""
    if name not in FEATURE_GROUPS:
        raise ValueError(f"Unknown feature group: {name}")
    labels = FEATURE_GROUPS[name]
    if len(set(labels)) != len(labels) or any(label not in REGISTRY for label in labels):
        raise ValueError(f"Feature group {name} does not match the advanced label registry")
    return tuple(REGISTRY[label] for label in labels)


class Situations:
    def __init__(self):
        self.teams, self.styles = {}, {}
        self.league = [[0., 0.] for _ in pbp.LABELS]
        self.observations = {}

    def strength(self, team):
        values = self.teams.get(team, [[0., 0.] for _ in range(16)])
        return [total / (count + 100) for total, count in values]

    def style(self, team):
        passes, plays = self.styles.get(team, (0., 0.))
        return (passes + 55) / (plays + 100)

    def observe(self, game, records):
        home, away = team_key(game.home), team_key(game.away)
        own, other = records.get(home), records.get(away)
        if own is None or other is None:
            return
        previous = {home: self.strength(home), away: self.strength(away)}
        means = [total / max(1, count) for total, count in self.league]
        for team, opponent, offense, defense in ((home, away, own, other), (away, home, other, own)):
            old = self.teams.get(team, [[0., 0.] for _ in range(16)])
            values = []
            for i, (total, count) in enumerate(offense + defense):
                j = i % 8
                adjustment = previous[opponent][j + 8 if i < 8 else j]
                values.append([.9 * old[i][0] + total - count * (means[j] + adjustment), .9 * old[i][1] + count])
            self.teams[team] = values
            FeatureState.record_observation(self.observations, team, game)
            passes, plays = self.styles.get(team, (0., 0.))
            self.styles[team] = (passes * .9 + offense[7][0], plays * .9 + offense[7][1])
        for values in (own, other):
            for i, (total, count) in enumerate(values):
                self.league[i][0] += total
                self.league[i][1] += count

    def offseason(self):
        self.teams = {t: [[a * .65, b * .65] for a, b in v] for t, v in self.teams.items()}
        self.styles = {t: (a * .65, b * .65) for t, (a, b) in self.styles.items()}


class AdvancedState(FeatureState):
    def __init__(self, plays=None, evidence=None, now=None, qb_rushing=None):
        super().__init__()
        self.plays, self.evidence = plays or {}, evidence or Evidence()
        self.qb_rushing_plays = qb_rushing or {}
        self.now = now or datetime.now(timezone.utc)
        self.situations, self.away_streaks, self.contexts = Situations(), {}, {}
        self.venue_roofs = {}
        # Decayed QB rushing EPA above the league QB mean, by player and team.
        self.rushers, self.team_rushing, self.league_rushing = {}, {}, [0., 0.]
        self.rushing_observations = {}
        # Labeled starter examples from earlier games: did the pregame top
        # candidate start, and if not, where did the actual starter rank?
        self.starter_examples = {c: {"examples": 0, "top": 0, **{k: 0 for k in STARTER_CATEGORIES}} for c in STARTER_CLASSES}
        self.starter_examples_excluded = 0

    def cutoff(self, game):
        kickoff = kickoff_utc(game)
        return min(kickoff, self.now) if kickoff else None

    def estimator(self, evidence_class):
        """Add-one smoothed category frequencies, or None below the support threshold."""
        example = self.starter_examples[evidence_class]
        if example["examples"] < MIN_STARTER_EXAMPLES:
            return None
        top = (example["top"] + 1) / (example["examples"] + 2)
        misses = example["examples"] - example["top"]
        frequencies = {c: (example[c] + 1) / (misses + len(STARTER_CATEGORIES)) * (1 - top) for c in STARTER_CATEGORIES}
        return {"top": top, **frequencies, "examples": example["examples"]}

    def context(self, game):
        cutoff = self.cutoff(game)
        resolved = {team: self.evidence.quarterback_candidates(game, team, self.last_qb.get(team_key(team), ("", "")), cutoff, self.estimator)
                    for team in (game.home, game.away)}
        qbs = {team: top_record(value) for team, value in resolved.items()}
        weather = self.evidence.latest("weather", game.id, cutoff, timedelta(hours=48))
        kickoff = kickoff_utc(game)
        if weather and kickoff and kickoff - timestamp(weather["available_at"]) > timedelta(hours=48):
            weather = None
        # Historical roof-open/closed decisions cannot establish pregame exposure.
        known_roof = self.venue_roofs.get(game.stadium_id or game.stadium)
        weather_valid = weather is not None and all(
            isinstance(weather.get(k), (int, float)) and math.isfinite(weather[k])
            for k in ("wind_mph", "precipitation_inches", "temperature_f"))
        if weather is not None and (not weather_valid or weather.get("roof") != "outdoors" or known_roof != "outdoors"):
            weather = None
        availability = self.evidence.latest("availability", game.id, cutoff, timedelta(days=7))
        return {"quarterbacks": qbs, "quarterback_candidates": resolved, "weather": weather, "availability": availability,
                "availability_complete": availability_complete(availability, game.home, game.away),
                "travel": travel.context(game, self.away_streaks),
                "pbp_available": all(team_key(t) in self.situations.teams for t in (game.home, game.away))}

    def rushing_quality(self, identifier):
        epa, carries = self.rushers.get(identifier, (0., 0.))
        return epa / (carries + RUSHING_PRIOR_CARRIES), carries

    def rushing_change(self, team, identifier):
        """QB rushing value relative to the team's recent QB rushing; verified QBs only."""
        if not identifier or identifier not in self.qbs:
            return 0.
        epa, carries = self.team_rushing.get(team, (0., 0.))
        return self.rushing_quality(identifier)[0] - epa / (carries + RUSHING_PRIOR_CARRIES)

    def coverage(self, game, home_qb=None, away_qb=None):
        context = self.contexts.get(game.id) or self.context(game)
        qbs = context["quarterbacks"]
        coverage = super().coverage(game, qbs[game.home]["id"] if home_qb is None else home_qb,
                                    qbs[game.away]["id"] if away_qb is None else away_qb)
        for team, values in coverage["teams"].items():
            observed = self.situations.observations.get(team, {"games": 0, "last_game": None, "last_date": None})
            samples = self.situations.teams.get(team, [[0., 0.] for _ in range(16)])
            values["pbp"] = {**observed, "effective_plays": [v[1] for v in samples],
                             "neutral_prior": observed["games"] == 0,
                             "qb_rushing_games": self.rushing_observations.get(team, 0)}
            values["quarterback"]["rushing_carries"] = self.rushing_quality(values["quarterback"]["id"])[1]
            values["quarterback"]["evidence_class"] = context["quarterback_candidates"][
                game.home if team == team_key(game.home) else game.away]["class"]
        coverage["missing"] = {"weather": context["weather"] is None,
                               "availability": not context["availability_complete"],
                               "qb_rushing": not all(v["pbp"]["qb_rushing_games"] > 0 for v in coverage["teams"].values())}
        coverage["starter_examples"] = {**{c: dict(v) for c, v in self.starter_examples.items()},
                                        "excluded": self.starter_examples_excluded}
        return coverage

    def scenarios(self, game, home_qb=None, away_qb=None):
        """Weighted starter combinations; empty when the assumption is deterministic.

        Independent starter assignments are assumed. A what-if selection
        collapses that side to one candidate; an unknown candidate passes the
        explicit empty identifier so no unsupported QB change is applied.
        """
        context = self.contexts.get(game.id) or self.context(game)
        sides = []
        for team, selected in ((game.home, home_qb), (game.away, away_qb)):
            if selected is not None:
                sides.append([(selected, 1.)])
            else:
                sides.append([(c["id"], c["probability"]) for c in context["quarterback_candidates"][team]["candidates"] if c["probability"] > 0])
        if len(sides[0]) == 1 and len(sides[1]) == 1:
            return []
        return [{"home_qb": h, "away_qb": a, "weight": wh * wa, "features": self.features(game, h, a)}
                for h, wh in sides[0] for a, wa in sides[1]]

    def features(self, game, home_qb=None, away_qb=None):
        context = self.context(game)
        self.contexts[game.id] = context
        qbs = context["quarterbacks"]
        base = super().features(game, qbs[game.home]["id"] if home_qb is None else home_qb,
                                qbs[game.away]["id"] if away_qb is None else away_qb)
        home, away = team_key(game.home), team_key(game.away)
        h, a = self.situations.strength(home), self.situations.strength(away)
        situations = [(h[i] - a[i]) / pbp.SCALES[i] for i in range(8)] + [(a[i + 8] - h[i + 8]) / pbp.SCALES[i] for i in range(8)]
        weather, climate = context["weather"], [0.] * 3
        if weather:
            reliance = (self.situations.style(home) - self.situations.style(away)) / .15
            climate = [max(0, weather["wind_mph"] - 10) / 15 * reliance,
                       weather["precipitation_inches"] / .25 * reliance,
                       max(0, 40 - weather["temperature_f"]) / 30 * reliance]
        missing = {home: [0.] * 4, away: [0.] * 4}
        players = context["availability"].get("players", []) if context["availability_complete"] else []
        for player in players:
            if player["team"] not in missing:
                continue
            status, position = player["status"].lower(), player["position"]
            if status in {"out", "inactive"}:
                if position in {"WR", "TE", "RB", "FB"}:
                    missing[player["team"]][0] += 1 / 3
                elif position in {"T", "G", "C", "OT", "OG", "OL"}:
                    missing[player["team"]][1] += 1 / 3
                elif position in {"DE", "DT", "NT", "DL", "LB", "OLB", "ILB", "CB", "S", "FS", "SS", "DB"}:
                    missing[player["team"]][2] += 1 / 3
            elif status in {"questionable", "doubtful"}:
                missing[player["team"]][3] += 1 / 6
        availability = [a - h for a, h in zip(missing[away], missing[home])]
        hq = home_qb if home_qb is not None else qbs[game.home]["id"]
        aq = away_qb if away_qb is not None else qbs[game.away]["id"]
        rushing = (self.rushing_change(home, hq) - self.rushing_change(away, aq)) / RUSHING_SCALE
        return base + situations + context["travel"]["features"] + climate + availability + [rushing]

    def observe_starters(self, game, bundle):
        context = self.contexts.get(game.id)
        if context is None:
            return
        for raw_team in (game.home, game.away):
            quarterbacks = [q for q in bundle["player"].get(feature_key(game.season, game.week, game.kind, raw_team), []) if q.dropbacks > 0]
            resolved = context["quarterback_candidates"][raw_team]
            top = resolved["candidates"][0]
            # Only timestamped depth/confirmation evidence yields a candidate
            # list; previous-passer assumptions and unknown starters are excluded.
            if not quarterbacks or not top["id"] or resolved["class"] not in {"confirmed", "projected"}:
                self.starter_examples_excluded += 1
                continue
            starter = max(quarterbacks, key=lambda q: q.dropbacks)
            example = self.starter_examples[resolved["class"]]
            example["examples"] += 1
            if starter.id == top["id"]:
                example["top"] += 1
            else:
                rank = next((c["rank"] for c in resolved["candidates"] if c["id"] == starter.id), None)
                example["rank2" if rank == 2 else "rank3" if rank is not None and rank >= 3 else "unknown"] += 1

    def observe_rushing(self, game):
        rushers = self.qb_rushing_plays.get(game.id)
        if not rushers:
            return
        mean = self.league_rushing[0] / max(1., self.league_rushing[1])
        teams = set()
        for identifier, entry in rushers.items():
            if identifier not in self.qbs:
                continue
            above = entry["epa"] - mean * entry["carries"]
            epa, carries = self.rushers.get(identifier, (0., 0.))
            self.rushers[identifier] = (epa * .95 + above, carries * .95 + entry["carries"])
            # Same decay as the player so an unchanged starter is exactly no change.
            team_epa, team_carries = self.team_rushing.get(entry["team"], (0., 0.))
            self.team_rushing[entry["team"]] = (team_epa * .95 + above, team_carries * .95 + entry["carries"])
            self.league_rushing[0] += entry["epa"]
            self.league_rushing[1] += entry["carries"]
            teams.add(entry["team"])
        for team in (team_key(game.home), team_key(game.away)):
            if team in teams:
                self.rushing_observations[team] = self.rushing_observations.get(team, 0) + 1

    def observe(self, game, bundle):
        self.observe_starters(game, bundle)
        super().observe(game, bundle)
        self.situations.observe(game, self.plays.get(game.id, {}))
        self.observe_rushing(game)
        for team in (game.home, game.away):
            key = team_key(team)
            self.away_streaks[key] = 0 if team == game.home and not game.neutral else self.away_streaks.get(key, 0) + 1
        if game.roof:
            roof = "retractable" if game.roof in {"open", "closed", "retractable"} else game.roof
            self.venue_roofs[game.stadium_id or game.stadium] = roof

    def offseason(self):
        super().offseason()
        self.situations.offseason()
        self.away_streaks = {}
        self.rushers = {q: (epa * .8, carries * .8) for q, (epa, carries) in self.rushers.items()}
        self.team_rushing = {t: (epa * .8, carries * .8) for t, (epa, carries) in self.team_rushing.items()}


@dataclass
class AdvancedModel(MatchupModel):
    labels: tuple = LABELS
    # Starter mixtures and QB rushing did not pass their separate chronological
    # evaluations, so the application keeps deterministic starters by default.
    mixture: bool = False

    def probability(self, game, elo_probability, home_qb=None, away_qb=None):
        # Average probabilities over supported starter combinations, not logits.
        scenarios = self.state.scenarios(game, home_qb, away_qb) if self.mixture else None
        return corrected_mixture(elo_probability, self.state.features(game, home_qb, away_qb), scenarios, self.weights)


def enabled_indices(rows, contexts):
    # Optional signals need enough timestamped historical coverage to estimate
    # coefficients; absence of a snapshot is never evidence of healthy players.
    weather_rows = [r for r in rows if contexts[r["id"]]["weather"] is not None]
    availability_rows = [r for r in rows if contexts[r["id"]].get("availability_complete", False)]
    rushing_rows = [r for r in rows if not r.get("input_coverage", {}).get("missing", {}).get("qb_rushing", True)]
    weather, availability = len(weather_rows), len(availability_rows)
    indices = list(range(WEATHER_START))
    variation = {}
    for eligible, group in ((weather_rows, range(WEATHER_START, AVAILABILITY_START)),
                            (availability_rows, range(AVAILABILITY_START, RUSHING)),
                            (rushing_rows, range(RUSHING, len(LABELS)))):
        for i in group:
            nonzero = sum(abs(r["features"][i]) > 1e-12 for r in eligible)
            zeros = len(eligible) - nonzero
            # Weather/availability effects are indicator-like and need both
            # states; QB rushing change is continuous and needs coverage only.
            enabled = len(eligible) >= 100 and nonzero >= 20 and (zeros >= 20 or i >= RUSHING)
            variation[LABELS[i]] = {"games": len(eligible), "nonzero": nonzero, "zero": zeros, "enabled": enabled}
            if enabled:
                indices.append(i)
    return tuple(indices), {"weather_games": weather, "availability_games": availability, "qb_rushing_games": len(rushing_rows),
                            "weather_enabled": any(i in indices for i in range(WEATHER_START, AVAILABILITY_START)),
                            "availability_enabled": any(i in indices for i in range(AVAILABILITY_START, RUSHING)),
                            "qb_rushing_enabled": RUSHING in indices,
                            "variation": variation, "minimum_games": 100, "minimum_zero_and_nonzero": 20}


def evaluate_advanced(games, season, config, bundle, data, *, as_of_utc=None):
    state = AdvancedState(data.get("plays"), data.get("evidence"), now=as_of_utc, qb_rushing=data.get("qb_rushing"))
    rows, state, _ = replay(games, season, config, bundle, state, as_of_utc=as_of_utc)
    years = list(range(season - 6, season))
    regular = [r for r in rows if r["kind"] == "REG" and r["season"] in years]
    # Every training/evaluation Elo offset is itself selected using older years.
    elo_configs = historical_offsets(regular, games, years, selector=select_model)
    coverage = training_coverage(regular, years, require_pbp=True)
    report = {"version": VERSION, "status": "unavailable", "promoted": False, "coverage": coverage,
              "reason": "Prepare six prior seasons with at least 200 regular games and 95% weekly team/QB and play-by-play coverage.", "sources": data.get("sources", []),
              "evidence_digest": data.get("evidence", Evidence()).digest, "optimizer": None,
              "elo_configs": elo_configs}
    if not sufficient_history(coverage, require_pbp=True):
        return AdvancedModel((0.,) * len(LABELS), state, report), {}
    predictions, baseline, folds = [], [], []
    groups = {"QB + weekly efficiency": tuple(range(6)),
              "+ situational play-by-play": tuple(range(22)),
              "+ travel and body-clock": tuple(range(WEATHER_START))}
    ablations = {name: [] for name in groups}
    ablation_optimizers = {name: [] for name in groups}

    def converged_weights(rows, indices):
        return fit_correction(rows, indices, penalty=.1, fitter=fit_diagnostic)

    def deployed(train):
        # The application fits the passing-only label set; QB rushing is
        # experiment-only until it passes the chronological gate.
        indices, support = enabled_indices(train, state.contexts)
        return tuple(i for i in indices if i < RUSHING), support

    for year in years[3:]:
        train = [r for r in regular if r["season"] < year]
        target = [r for r in regular if r["season"] == year]
        indices, support = deployed(train)
        weights, optimizer = converged_weights(train, indices)
        predicted = [{**r, "probability": corrected(r["probability"], r["features"], weights)} for r in target]
        predictions.extend(predicted)
        baseline.extend(target)
        for name, group in groups.items():
            subset_weights, subset_optimizer = converged_weights(train, group)
            ablation_optimizers[name].append(subset_optimizer)
            ablations[name].extend({**r, "probability": corrected(r["probability"], r["features"], subset_weights)} for r in target)
        folds.append({"season": year, "baseline": metrics(target), "challenger": metrics(predicted),
                      "support": support, "optimizer": optimizer})
    indices, support = deployed(regular)
    weights, optimizer = converged_weights(regular, indices)
    b, c = metrics(baseline), metrics(predictions)
    qualifies = (optimizer["converged"] and c["brier"] < b["brier"] and c["log_loss"] < b["log_loss"]
                 and c["accuracy"] > b["accuracy"])
    reason = ("The advanced correction fit did not converge; forecasts use Elo without advanced corrections."
              if not optimizer["converged"] else
              "Annual chronological evaluation; explicit experimental selection only. Weather/availability effects need 100 complete earlier snapshots and at least 20 zero and 20 nonzero observations per feature. Unknown reports contribute no availability effect. Confirmed lineups require timestamped evidence.")
    report.update(status="evaluated" if optimizer["converged"] else "unavailable",
                  name="Advanced matchup", baseline=b, challenger_metrics=c,
                  tuning_seasons=years, test_seasons=years[3:], by_season=folds, support=support,
                  weights=dict(zip(LABELS, weights)), qualifies=qualifies, optimizer=optimizer,
                  by_team={t: {"baseline": metrics(baseline, t), "challenger": metrics(predictions, t)} for t in ("PIT", "MIN")},
                  ablations=[{"name": name, "metrics": metrics(values),
                              "optimizer": ablation_optimizers[name][-1]}
                              for name, values in ablations.items()] + [{"name": "+ forecast weather / supported availability", "metrics": c, "optimizer": None}],
                  uncertainty=paired_uncertainty(baseline, predictions),
                  brier_improvement_pct=100 * (b["brier"] - c["brier"]) / b["brier"],
                  reason=reason)
    probabilities = ({r["id"]: corrected(r["probability"], r["features"], weights) for r in rows if r["season"] == season}
                     if optimizer["converged"] else {})
    return AdvancedModel(weights, state, report), probabilities


class AdvancedStore:
    def __init__(self, directory, offline=False):
        self.directory, self.offline = Path(directory), offline
        self.evidence = EvidenceStore(self.directory / "evidence.sqlite3")

    def load(self, season, refresh=False):
        plays, rushing, sources, digest = {}, {}, [], hashlib.sha256()
        store = PBPStore(self.directory, offline=True)
        for year in range(season - 8, season + 1):
            # Bulk preparation is explicit; dashboard refresh only updates the
            # active season after a cache has been prepared.
            if refresh and year == season and not self.offline and (self.directory / f"pbp_{year}.json").exists():
                payload = PBPStore(self.directory).load(year, refresh=True)
            else:
                payload = store.load(year)
            if payload:
                plays.update(payload["games"])
                rushing.update(payload.get("qb_rushing", {}))
                metadata = {k: v for k, v in payload.items() if k not in {"games", "qb_rushing"}}
                sources.append(metadata)
                digest.update(json.dumps(metadata, sort_keys=True).encode())
        if refresh and not self.offline and season >= 2025:
            from .prepare import depth_chart
            depth_chart(self.evidence, self.directory, season, refresh=True)
        evidence = self.evidence.snapshot()
        digest.update(evidence.digest.encode())
        return {"plays": plays, "qb_rushing": rushing, "evidence": evidence, "sources": sources, "digest": digest.hexdigest()}
