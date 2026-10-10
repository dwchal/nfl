"""Pregame QB/availability, situational PBP, weather-style and travel model."""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import pbp, travel
from .evidence import Evidence, EvidenceStore, timestamp
from .evaluation import paired_uncertainty
from .forecast import kickoff_utc
from .matchup import (FeatureState, MatchupModel, LABELS as BASE_LABELS,
                      corrected, fit_diagnostic, logit, replay)
from .model import backtest, metrics, select_model, team_key
from .pbp import PBPStore

VERSION = "advanced-v2"
LABELS = BASE_LABELS + tuple(f"Situational {side}: {label}" for side in ("offense", "defense") for label in pbp.LABELS) + travel.LABELS + (
    "Wind × passing reliance", "Rain × passing reliance", "Cold × passing reliance",
    "Unavailable receivers", "Unavailable offensive line", "Unavailable defenders", "Questionable players")
WEATHER_START = 6 + 2 * len(pbp.LABELS) + len(travel.LABELS)
AVAILABILITY_START = WEATHER_START + 3


class Situations:
    def __init__(self):
        self.teams, self.styles = {}, {}
        self.league = [[0., 0.] for _ in pbp.LABELS]

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
    def __init__(self, plays=None, evidence=None, now=None):
        super().__init__()
        self.plays, self.evidence = plays or {}, evidence or Evidence()
        self.now = now or datetime.now(timezone.utc)
        self.situations, self.away_streaks, self.contexts = Situations(), {}, {}
        self.venue_roofs = {}

    def cutoff(self, game):
        kickoff = kickoff_utc(game)
        return min(kickoff, self.now) if kickoff else None

    def context(self, game):
        cutoff = self.cutoff(game)
        qbs = {team: self.evidence.quarterback(game, team, self.last_qb.get(team_key(team), ("", "")), cutoff)
               for team in (game.home, game.away)}
        weather = self.evidence.latest("weather", game.id, cutoff, timedelta(hours=48))
        kickoff = kickoff_utc(game)
        if weather and kickoff and kickoff - timestamp(weather["available_at"]) > timedelta(hours=48):
            weather = None
        # Historical roof-open/closed decisions cannot establish pregame exposure.
        known_roof = self.venue_roofs.get(game.stadium_id or game.stadium)
        if weather and (weather.get("roof") != "outdoors" or known_roof != "outdoors"):
            weather = None
        availability = self.evidence.latest("availability", game.id, cutoff, timedelta(days=7))
        return {"quarterbacks": qbs, "weather": weather, "availability": availability,
                "travel": travel.context(game, self.away_streaks),
                "pbp_available": all(team_key(t) in self.situations.teams for t in (game.home, game.away))}

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
        for player in (context["availability"] or {}).get("players", []):
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
        return base + situations + context["travel"]["features"] + climate + availability

    def observe(self, game, bundle):
        super().observe(game, bundle)
        self.situations.observe(game, self.plays.get(game.id, {}))
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


@dataclass
class AdvancedModel(MatchupModel):
    labels: tuple = LABELS


def enabled_indices(rows, contexts):
    # Optional signals need enough timestamped historical coverage to estimate
    # coefficients; absence of a snapshot is never evidence of healthy players.
    weather = sum(contexts[r["id"]]["weather"] is not None for r in rows)
    availability = sum(contexts[r["id"]]["availability"] is not None for r in rows)
    indices = list(range(WEATHER_START))
    if weather >= 100:
        indices.extend(range(WEATHER_START, AVAILABILITY_START))
    if availability >= 100:
        indices.extend(range(AVAILABILITY_START, len(LABELS)))
    return tuple(indices), {"weather_games": weather, "availability_games": availability,
                            "weather_enabled": weather >= 100, "availability_enabled": availability >= 100}


def evaluate_advanced(games, season, config, bundle, data):
    state = AdvancedState(data.get("plays"), data.get("evidence"))
    rows, state, _ = replay(games, season, config, bundle, state)
    years = list(range(season - 6, season))
    regular = [r for r in rows if r["kind"] == "REG" and r["season"] in years]
    # Every training/evaluation Elo offset is itself selected using older years.
    for year in years:
        previous_config, _ = select_model(games, year)
        previous = {p["id"]: p["probability"] for p in backtest(games, [year], previous_config)}
        for row in regular:
            if row["season"] == year:
                row.update(probability=previous[row["id"]], offset=logit(previous[row["id"]]))
    coverage = {str(year): {"games": sum(r["season"] == year for r in regular),
                           "pbp": sum(r["season"] == year and all(team_key(t) in data.get("plays", {}).get(r["id"], {}) for t in (r["home"], r["away"])) for r in regular)} for year in years}
    report = {"version": VERSION, "status": "unavailable", "promoted": False, "coverage": coverage,
              "reason": "Prepare six prior seasons with at least 95% play-by-play coverage.", "sources": data.get("sources", []),
              "evidence_digest": data.get("evidence", Evidence()).digest, "optimizer": None}
    if any(c["games"] < 200 or c["pbp"] / c["games"] < .95 for c in coverage.values()):
        return AdvancedModel((0.,) * len(LABELS), state, report), {}
    predictions, baseline, folds = [], [], []
    groups = {"QB + weekly efficiency": tuple(range(6)),
              "+ situational play-by-play": tuple(range(22)),
              "+ travel and body-clock": tuple(range(WEATHER_START))}
    ablations = {name: [] for name in groups}
    ablation_optimizers = {name: [] for name in groups}

    def converged_weights(rows, indices):
        weights, optimizer = fit_diagnostic(rows, indices, penalty=.1)
        if not optimizer["converged"]:
            weights = (0.,) * len(weights)
        return weights, optimizer

    for year in years[3:]:
        train = [r for r in regular if r["season"] < year]
        target = [r for r in regular if r["season"] == year]
        indices, support = enabled_indices(train, state.contexts)
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
    indices, support = enabled_indices(regular, state.contexts)
    weights, optimizer = converged_weights(regular, indices)
    b, c = metrics(baseline), metrics(predictions)
    qualifies = (optimizer["converged"] and c["brier"] < b["brier"] and c["log_loss"] < b["log_loss"]
                 and c["accuracy"] > b["accuracy"])
    reason = ("The advanced correction fit did not converge; forecasts use Elo without advanced corrections."
              if not optimizer["converged"] else
              "Annual chronological evaluation; explicit experimental selection only. Availability/weather coefficients stay zero until at least 100 earlier snapshots are available. Confirmed lineups require timestamped evidence.")
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
        plays, sources, digest = {}, [], hashlib.sha256()
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
                metadata = {k: v for k, v in payload.items() if k != "games"}
                sources.append(metadata)
                digest.update(json.dumps(metadata, sort_keys=True).encode())
        if refresh and not self.offline and season >= 2025:
            from .prepare import depth_chart
            depth_chart(self.evidence, self.directory, season, refresh=True)
        evidence = self.evidence.snapshot()
        digest.update(evidence.digest.encode())
        return {"plays": plays, "evidence": evidence, "sources": sources, "digest": digest.hexdigest()}
