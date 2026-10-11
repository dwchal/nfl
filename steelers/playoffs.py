"""Playoff estimates from simulated regular-season results (2002 onward).

Apply result-based NFL tiebreakers through strength of schedule. Remaining
ties are randomized because simulations do not predict points or touchdowns.
"""

from collections import Counter

from .model import team_key

DIVISIONS = {
    "AFC East": ("BUF", "MIA", "NE", "NYJ"),
    "AFC North": ("BAL", "CIN", "CLE", "PIT"),
    "AFC South": ("HOU", "IND", "JAX", "TEN"),
    "AFC West": ("DEN", "KC", "LAC", "LV"),
    "NFC East": ("DAL", "NYG", "PHI", "WAS"),
    "NFC North": ("CHI", "DET", "GB", "MIN"),
    "NFC South": ("ATL", "CAR", "NO", "TB"),
    "NFC West": ("ARI", "LA", "SEA", "SF"),
}
NOTE = ("Uses result-based NFL tiebreakers through strength of schedule. "
        "Ties still unresolved are randomized; points and touchdowns are not simulated. "
        "Future ties are not simulated. These are model estimates, not official clinching odds.")


class PlayoffRace:
    def __init__(self, games, season, team, postseason=()):
        self.team = team_key(team)
        self.wild_card_slots = 3 if season >= 2020 else 2
        self.reason = None
        self.counts = Counter()
        self.fallback_samples = 0
        if season < 2002:
            self.reason = "Playoff estimates support the division alignment used since 2002."
            return
        expected = {t for division in DIVISIONS.values() for t in division}
        totals = Counter(team_key(t) for g in games for t in (g.home, g.away))
        size = 17 if season >= 2021 else 16
        # The canceled 2022 BUF–CIN game left these clubs with 16 games.
        lengths = {t: size - (season == 2022 and t in {"BUF", "CIN"}) for t in expected}
        if set(totals) != expected or any(totals[t] != lengths[t] for t in expected):
            self.reason = "A complete league regular-season schedule is needed to estimate playoff chances."
            return
        self.teams = sorted(expected)
        self.index = {t: i for i, t in enumerate(self.teams)}
        self.target = self.index[self.team]
        conference = next(name[:3] for name, members in DIVISIONS.items() if self.team in members)
        self.divisions = [[self.index[t] for t in members] for name, members in DIVISIONS.items()
                          if name.startswith(conference)]
        self.conference = {t for members in self.divisions for t in members}
        self.meetings = [[0] * 32 for _ in self.teams]
        self.base_points = [[0] * 32 for _ in self.teams]
        self.ties = [[0] * 32 for _ in self.teams]
        self.remaining = []
        for game in games:
            h, a = self.index[team_key(game.home)], self.index[team_key(game.away)]
            self.meetings[h][a] += 1
            self.meetings[a][h] += 1
            if game.completed:
                points = 2 if game.home_score > game.away_score else 0 if game.home_score < game.away_score else 1
                self.base_points[h][a] += points
                self.base_points[a][h] += 2 - points
                if points == 1:
                    self.ties[h][a] += 1
                    self.ties[a][h] += 1
            else:
                self.remaining.append((game.id, h, a))
        self.played = [sum(row) for row in self.meetings]
        self.opponents = [{j for j, n in enumerate(row) if n} for row in self.meetings]
        self.observed = None
        if not self.remaining:
            field = {team_key(t) for g in postseason for t in (g.home, g.away)
                     if g.kind in {"WC", "DIV", "CON", "SB"}}
            conference_fields = [{t for name, members in DIVISIONS.items() if name.startswith(c) for t in members}
                                 for c in ("AFC", "NFC")]
            if field <= expected and all(len(field & members) == 4 + self.wild_card_slots
                                         for members in conference_fields):
                self.observed = self.team in field

    def sample(self, outcomes, rng):
        if self.reason or self.observed is not None:
            return
        points = [row.copy() for row in self.base_points]
        for identifier, h, a in self.remaining:
            result = 2 * outcomes[identifier]
            points[h][a] += result
            points[a][h] += 2 - result
        standings = _Standings(self, points, rng)
        ranks = []
        for division in self.divisions:
            pending, ranked = list(division), []
            while pending:
                winner = standings.choose(pending, division)
                ranked.append(winner)
                pending.remove(winner)
            ranks.append(ranked)
        champions = {ranked[0] for ranked in ranks}
        wildcards = set()
        pending = [ranked[1:] for ranked in ranks]
        for _ in range(self.wild_card_slots):
            candidates = [ranked[0] for ranked in pending if ranked]
            winner = standings.choose(candidates)
            wildcards.add(winner)
            for ranked in pending:
                if ranked and ranked[0] == winner:
                    ranked.pop(0)
                    break
        if self.target in champions:
            self.counts["division"] += 1
        elif self.target in wildcards:
            self.counts["wild_card"] += 1
        else:
            self.counts["miss"] += 1
        self.fallback_samples += standings.random_tie

    def report(self, simulations):
        if self.reason:
            return {"status": "unavailable", "reason": self.reason, "probability": None}
        if self.observed is not None:
            return {"status": "observed", "probability": float(self.observed),
                    "division_probability": None, "wild_card_probability": None,
                    "miss_probability": float(not self.observed), "simulations": 0,
                    "note": "Regular season complete. Qualification is taken from the actual postseason schedule."}
        return {"status": "estimated", "probability": (self.counts["division"] + self.counts["wild_card"]) / simulations,
                "division_probability": self.counts["division"] / simulations,
                "wild_card_probability": self.counts["wild_card"] / simulations,
                "miss_probability": self.counts["miss"] / simulations,
                "simulations": simulations, "wild_card_slots": self.wild_card_slots,
                "random_tiebreak_simulations": self.fallback_samples, "note": NOTE}


class _Standings:
    def __init__(self, race, points, rng):
        self.race, self.points, self.rng = race, points, rng
        self.totals = [sum(row) for row in points]
        self.random_tie = False

    def percentage(self, team, opponents):
        games = sum(self.race.meetings[team][o] for o in opponents)
        return sum(self.points[team][o] for o in opponents) / (2 * games) if games else None

    def strength(self, team, victory=False):
        weights = [(o, (self.points[team][o] - self.race.ties[team][o]) // 2
                    if victory else self.race.meetings[team][o]) for o in self.race.opponents[team]]
        games = sum(n * self.race.played[o] for o, n in weights)
        return sum(n * self.totals[o] for o, n in weights) / (2 * games) if games else 0

    def head_to_head(self, tied, division):
        if division is not None or len(tied) == 2:
            return {t: self.percentage(t, set(tied) - {t}) for t in tied}
        # Across divisions, multi-club head-to-head applies only to a sweep.
        winners = [t for t in tied if all(self.race.meetings[t][o] and
                   self.points[t][o] == 2 * self.race.meetings[t][o] for o in tied if o != t)]
        if winners:
            return {t: int(t in winners) for t in tied}
        losers = [t for t in tied if all(self.race.meetings[t][o] and self.points[t][o] == 0
                                       for o in tied if o != t)]
        return {t: int(t not in losers) for t in tied}

    def choose(self, candidates, division=None):
        best = max(self.totals[t] / self.race.played[t] for t in candidates)
        tied = [t for t in candidates if self.totals[t] / self.race.played[t] == best]
        if len(tied) == 1:
            return tied[0]
        common = set.intersection(*(self.race.opponents[t] for t in tied)) - set(tied)
        steps = [lambda: self.head_to_head(tied, division)]
        division_record = lambda: {t: self.percentage(t, division) for t in tied}
        conference_record = lambda: {t: self.percentage(t, self.race.conference) for t in tied}
        common_record = lambda: {t: self.percentage(t, common) for t in tied}
        if division is not None:
            steps.extend([division_record, common_record, conference_record])
        else:
            steps.append(conference_record)
            if all(sum(self.race.meetings[t][o] for o in common) >= 4 for t in tied):
                steps.append(common_record)
        steps.extend([lambda: {t: self.strength(t, True) for t in tied},
                      lambda: {t: self.strength(t) for t in tied}])
        for step in steps:
            values = step()
            if any(v is None for v in values.values()):
                continue
            high = max(values.values())
            kept = [t for t in tied if values[t] == high]
            if len(kept) == 1:
                return kept[0]
            if len(kept) < len(tied):
                # NFL restarts the appropriate procedure when a group shrinks.
                return self.choose(kept, division)
        self.random_tie = True
        return self.rng.choice(tied)
