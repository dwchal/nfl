"""Chronological EPA/QB/rest corrections to Elo, using only past games.

Weekly stats are observed after a game (with a one-day publication lag). Starter
proxies come from the previous game, never the schedule's postgame QB columns.
Corrections are regularized logistic regression, implemented in the stdlib.
"""

import math
from dataclasses import dataclass

from .model import BASE_RATING, home_probability, metrics, replay_season, team_key, update_ratings

VERSION = "matchup-v1"
LABELS = ("Passing offense", "Passing defense", "Rushing offense", "Rushing defense", "Rest advantage", "Quarterback change")
GROUPS = {"Rest": (4,), "Efficiency + rest": (0, 1, 2, 3, 4), "Efficiency + rest + QB": tuple(range(6))}


def sigmoid(value):
    return 1 / (1 + math.exp(-max(-20, min(20, value))))


def logit(probability):
    return math.log(probability / (1 - probability))


class FeatureState:
    def __init__(self):
        self.teams = {}
        self.qbs = {}
        self.last_qb = {}
        self.league = [0., 0., 0., 0.]

    def means(self):
        return self.league[0] / max(1, self.league[1]), self.league[2] / max(1, self.league[3])

    def strength(self, team):
        row = self.teams.get(team, [0.] * 8)
        return [row[i] / (row[i + 1] + (200 if i in (0, 4) else 120)) for i in (0, 2, 4, 6)]

    def qb_quality(self, identifier):
        epa, plays = self.qbs.get(identifier, (0., 0.))
        return epa / (plays + 200), plays

    def features(self, game, home_qb=None, away_qb=None):
        home, away = team_key(game.home), team_key(game.away)
        h, a = self.strength(home), self.strength(away)
        hq = home_qb if home_qb is not None else self.last_qb.get(home, ("", ""))[0]
        aq = away_qb if away_qb is not None else self.last_qb.get(away, ("", ""))[0]
        rest = 0 if game.home_rest is None or game.away_rest is None else (min(14, game.home_rest) - min(14, game.away_rest)) / 7
        # QB skill already appears in team offense; estimate only the change
        # relative to that offense, rather than adding a second full QB rating.
        h_change = self.qb_quality(hq)[0] - h[0] if hq else 0
        a_change = self.qb_quality(aq)[0] - a[0] if aq else 0
        return [(h[0] - a[0]) / .15, (a[2] - h[2]) / .15,
                (h[1] - a[1]) / .10, (a[3] - h[3]) / .10, rest,
                (h_change - a_change) / .15]

    def offseason(self):
        self.teams = {t: [v * .65 for v in values] for t, values in self.teams.items()}
        self.qbs = {q: (epa * .8, count * .8) for q, (epa, count) in self.qbs.items()}

    def observe(self, game, bundle):
        kind = "REG" if game.kind == "REG" else "POST"
        keys = [(game.season, game.week, kind, t) for t in (game.home, game.away)]
        rows = [bundle["team"].get(key) for key in keys]
        if any(row is None for row in rows):
            return
        mean_pass, mean_rush = self.means()
        previous = {team_key(t): self.strength(team_key(t)) for t in (game.home, game.away)}
        for index, raw_team in enumerate((game.home, game.away)):
            team, opponent = team_key(raw_team), team_key(game.away if index == 0 else game.home)
            own, other, opp = rows[index], rows[1 - index], previous[opponent]
            values = self.teams.setdefault(team, [0.] * 8)
            observations = (own.passing_epa - own.dropbacks * (mean_pass + opp[2]), own.dropbacks,
                            own.rushing_epa - own.carries * (mean_rush + opp[3]), own.carries,
                            other.passing_epa - other.dropbacks * (mean_pass + opp[0]), other.dropbacks,
                            other.rushing_epa - other.carries * (mean_rush + opp[1]), other.carries)
            self.teams[team] = [v * .9 + new for v, new in zip(values, observations)]
            quarterbacks = bundle["player"].get(keys[index], [])
            for qb in quarterbacks:
                epa, plays = self.qbs.get(qb.id, (0., 0.))
                self.qbs[qb.id] = (epa * .95 + qb.epa - mean_pass * qb.dropbacks, plays * .95 + qb.dropbacks)
            candidates = [q for q in quarterbacks if q.dropbacks > 0]
            if candidates:
                starter = max(candidates, key=lambda q: q.dropbacks)
                self.last_qb[team] = (starter.id, starter.name)
        for row in rows:
            self.league = [old + new for old, new in zip(self.league, (row.passing_epa, row.dropbacks, row.rushing_epa, row.carries))]


def replay(games, season, config, bundle):
    """Return input rows, current state, and Elo; no target's own stats enter it."""
    state, ratings, rows, pending = FeatureState(), {}, [], []
    elo_predictions = {p["id"]: p["probability"] for year in range(season - 8, season + 1)
                       for p in replay_season(games, year, config)[2]}
    year = None
    for game in sorted(games, key=lambda g: (g.day, g.kickoff, g.id)):
        if not season - 8 <= game.season <= season:
            continue
        if year != game.season:
            if year is not None:
                for finished in pending:
                    state.observe(finished, bundle)
                pending = []
                state.offseason()
                ratings = {t: BASE_RATING + (r - BASE_RATING) * config.carryover for t, r in ratings.items()}
            year = game.season
        # Stats from same-day games cannot inform another forecast that day.
        ready = [g for g in pending if (game.day - g.day).days >= 1]
        pending = [g for g in pending if (game.day - g.day).days < 1]
        for finished in ready:
            state.observe(finished, bundle)
        home, away = team_key(game.home), team_key(game.away)
        ratings.setdefault(home, BASE_RATING)
        ratings.setdefault(away, BASE_RATING)
        if not game.completed:
            continue
        probability = elo_predictions[game.id]
        key = (game.season, game.week, "REG" if game.kind == "REG" else "POST")
        result = float(game.home_score > game.away_score) if game.home_score != game.away_score else .5
        rows.append({"id": game.id, "season": game.season, "kind": game.kind, "home": home, "away": away,
                     "probability": probability, "offset": logit(probability), "features": state.features(game),
                     "result": result, "covered": all((*key, t) in bundle["team"] and (*key, t) in bundle["player"] for t in (game.home, game.away))})
        update_ratings(ratings, home, away, result, game.neutral, config, game.home_score - game.away_score)
        pending.append(game)
    # Future forecasts use all completed games' available statistics.
    for game in pending:
        state.observe(game, bundle)
    return rows, state, ratings


def fit(rows, indices, penalty=.03):
    weights = [0.] * 6
    for _ in range(300):
        gradient = [0.] * 6
        for row in rows:
            x = [max(-4, min(4, v)) for v in row["features"]]
            error = sigmoid(row["offset"] + sum(weights[i] * x[i] for i in indices)) - row["result"]
            for i in indices:
                gradient[i] += error * x[i]
        change = 0.
        for i in indices:
            step = .5 * (gradient[i] / max(1, len(rows)) + penalty * weights[i])
            weights[i] -= step
            change = max(change, abs(step))
        if change < 1e-7:
            break
    return tuple(weights)


def corrected(probability, features, weights):
    return sigmoid(logit(probability) + sum(w * max(-4, min(4, v)) for w, v in zip(weights, features)))


@dataclass
class MatchupModel:
    weights: tuple
    state: FeatureState
    report: dict

    def probability(self, game, elo_probability, home_qb=None, away_qb=None):
        return corrected(elo_probability, self.state.features(game, home_qb, away_qb), self.weights)


def evaluate(games, season, config, bundle):
    rows, state, _ = replay(games, season, config, bundle)
    years = list(range(season - 6, season))
    tuning, tests = years[:3], years[3:]
    regular = [r for r in rows if r["kind"] == "REG"]
    train = [r for r in regular if r["season"] in tuning]
    test = [r for r in regular if r["season"] in tests]
    coverage = {str(year): {"games": len([r for r in regular if r["season"] == year]),
                           "covered": sum(r["covered"] for r in regular if r["season"] == year)} for year in years}
    report = {"status": "unavailable", "version": VERSION, "coverage": coverage,
              "tuning_seasons": tuning, "test_seasons": tests,
              "reason": "Six prior seasons with at least 95% team and QB statistics coverage are required.", "promoted": False}
    if any(c["games"] < 200 or c["covered"] / c["games"] < .95 for c in coverage.values()):
        return MatchupModel((0.,) * 6, state, report), {}
    early = [r for r in train if r["season"] < tuning[-1]]
    validation = [r for r in train if r["season"] == tuning[-1]]

    def forecasts(inputs, weights):
        return [{**r, "probability": corrected(r["probability"], r["features"], weights)} for r in inputs]

    candidates = []
    for name, indices in GROUPS.items():
        weights = fit(early, indices)
        candidates.append((metrics(forecasts(validation, weights)), name, indices))
    # Fixed full challenger. Smaller feature families are diagnostics, rather
    # than repeated searches against the later comparison seasons.
    name = "Efficiency + rest + QB"
    indices = GROUPS[name]
    weights = fit(train, indices)
    predictions = forecasts(test, weights)
    baseline, challenger = metrics(test), metrics(predictions)
    promoted = challenger["brier"] < baseline["brier"] and challenger["log_loss"] < baseline["log_loss"]
    report.update(status="evaluated", name=name, weights=dict(zip(LABELS, weights)), baseline=baseline,
                  challenger_metrics=challenger, promoted=promoted,
                  brier_improvement_pct=100 * (baseline["brier"] - challenger["brier"]) / baseline["brier"],
                  by_team={team: {"baseline": metrics(test, team), "challenger": metrics(predictions, team)} for team in ("PIT", "MIN")},
                  by_season=[{"season": year, "baseline": metrics([r for r in test if r["season"] == year]),
                              "challenger": metrics([r for r in predictions if r["season"] == year])} for year in tests],
                  ablations=[{"name": group, "validation": score} for score, group, _ in candidates],
                  reason="Matchup corrections improved both probability scores and are enabled by default." if promoted else "Matchup corrections did not improve both scores. Elo remains the default; the matchup model is available for comparison.")
    # Completed-game estimates use the frozen pre-target coefficients, then the
    # feature state recorded before each game. Never current-season fitting.
    probabilities = {r["id"]: corrected(r["probability"], r["features"], weights) for r in rows if r["season"] == season}
    return MatchupModel(weights, state, report), probabilities


def explain(model, game, elo_probability, team):
    sign = 1 if game.home == team else -1
    features = model.state.features(game)
    running = elo_probability
    contributions = []
    for label, feature, weight in zip(LABELS, features, model.weights):
        updated = corrected(running, [feature], [weight])
        delta = sign * (updated - running)
        contributions.append({"label": label, "change": delta})
        running = updated
    return contributions


def next_context(model, game, elo_probability, team, bundle, selected_qb="", opponent_qb=""):
    def quarterbacks(code):
        rows = [r for r in bundle["rosters"] if r["team"] == code and r["position"] == "QB"]
        unique = {r["gsis_id"]: r for r in rows if r["gsis_id"]}
        return [{"id": identifier, "name": row["full_name"], "status": row.get("status", ""),
                 "prior_dropbacks": round(model.state.qb_quality(identifier)[1]),
                 "efficiency": round(model.state.qb_quality(identifier)[0], 3)} for identifier, row in sorted(unique.items(), key=lambda item: item[1]["full_name"])]
    opponent = game.away if game.home == team else game.home
    options, opposing = quarterbacks(team), quarterbacks(opponent)
    for selected, choices in ((selected_qb, options), (opponent_qb, opposing)):
        if selected and selected not in {q["id"] for q in choices}:
            raise ValueError("Choose a quarterback listed on this team's roster.")
    home_qb, away_qb = (selected_qb, opponent_qb) if game.home == team else (opponent_qb, selected_qb)
    probability = model.probability(game, elo_probability, home_qb or None, away_qb or None)
    regular_probability = model.probability(game, elo_probability)
    convert = lambda p: p if game.home == team else 1 - p
    injuries = [r for r in bundle["injuries"] if r["team"] in {team, opponent}
                and int(r["week"]) == game.week and r.get("game_type", "REG") == game.kind]
    return {"game_id": game.id, "team_qbs": options, "opponent_qbs": opposing,
            "features": dict(zip(LABELS, model.state.features(game))), "elo_home_probability": elo_probability,
            "selected_qb": selected_qb, "opponent_qb": opponent_qb,
            "probability": convert(probability), "default_probability": convert(regular_probability),
            "elo_probability": convert(elo_probability), "contributions": explain(model, game, elo_probability, team),
            "assumed_qbs": {code: {"id": model.state.last_qb.get(team_key(code), ("", ""))[0],
                                  "name": model.state.last_qb.get(team_key(code), ("", "Unknown"))[1]} for code in (team, opponent)},
            "injuries": [{"team": r["team"], "name": r["full_name"], "position": r["position"],
                          "status": r["report_status"] or r["practice_status"],
                          "injury": r.get("report_primary_injury") or r.get("practice_primary_injury", "")} for r in injuries],
            "rest": {"home": game.home_rest, "away": game.away_rest},
            "note": "QB assumptions use the previous game's leading passer, not a confirmed lineup. New or lightly used QBs are shrunk toward league average. Injury reports are context; no unvalidated injury penalties are added."}
