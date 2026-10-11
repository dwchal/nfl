import random
import unittest
from dataclasses import replace
from datetime import date, timedelta
from types import SimpleNamespace

from steelers.analysis import build_dashboard, project_season
from steelers.data import Game
from steelers.playoffs import DIVISIONS, PlayoffRace, _Standings


def league(season=2026, completed=False):
    """A full league fixture; circle scheduling gives each club 16/17 games."""
    teams = sorted(t for members in DIVISIONS.values() for t in members)
    games = []
    for week in range(1, 18 if season >= 2021 else 17):
        for i in range(16):
            home, away = teams[i], teams[-i - 1]
            games.append(Game(f"{week}-{i}", season, "REG", week,
                              date(season, 9, 1) + timedelta(days=7 * week), "13:00",
                              home, away, 20 if completed else None, 10 if completed else None, False, ""))
        teams = [teams[0], teams[-1], *teams[1:-1]]
    return games


def standings(points, meetings, rng=None):
    count = len(points)
    race = SimpleNamespace(meetings=meetings, ties=[[0] * count for _ in range(count)],
                           played=[sum(row) for row in meetings],
                           opponents=[{o for o, n in enumerate(row) if n} for row in meetings],
                           conference=set(range(count)))
    return _Standings(race, points, rng or random.Random(43))


class PlayoffTests(unittest.TestCase):
    def test_full_simulation_records_and_playoff_paths_are_consistent(self):
        games = league()
        for team in ("PIT", "MIN"):
            ratings = {t: 0. for members in DIVISIONS.values() for t in members}
            ratings[team] = 6000.
            result = project_season(games, ratings, 0, 0, 0, 60, team=team,
                                    season_games=games, season=2026)
            self.assertEqual(result["predicted_record"], {"wins": 17, "losses": 0, "ties": 0, "probability": 1.})
            self.assertEqual(result["expected_losses"], 0.)
            self.assertEqual(result["playoffs"]["probability"], 1.)
            self.assertEqual(result["playoffs"]["division_probability"], 1.)
            self.assertEqual(result["playoffs"]["wild_card_probability"], 0.)
            self.assertEqual(ratings[team], 6000.)
            self.assertEqual(result, project_season(games, ratings, 0, 0, 0, 60, team=team,
                                                   season_games=games, season=2026))
            ratings[team] = -6000.
            loser = project_season(games, ratings, 0, 0, 0, 60, team=team,
                                   season_games=games, season=2026)
            self.assertEqual(loser["predicted_record"]["losses"], 17)
            self.assertEqual(loser["playoffs"]["probability"], 0.)
            self.assertEqual(loser["playoffs"]["miss_probability"], 1.)

    def test_playoff_format_changes_and_no_duplicate_qualifiers(self):
        for season, wildcards in ((2019, 4), (2020, 6), (2026, 6)):
            games = league(season)
            outcomes = {g.id: bool(i % 3) for i, g in enumerate(games)}
            reports = {}
            for team in sorted(t for members in DIVISIONS.values() for t in members):
                race = PlayoffRace(games, season, team)
                race.sample(outcomes, random.Random(43))
                reports[team] = race.report(1)
            self.assertEqual(sum(r["division_probability"] for r in reports.values()), 8)
            self.assertEqual(sum(r["wild_card_probability"] for r in reports.values()), wildcards)
            self.assertEqual(sum(r["probability"] for r in reports.values()), 8 + wildcards)
            for r in reports.values():
                self.assertEqual(r["division_probability"] + r["wild_card_probability"], r["probability"])
                self.assertEqual(r["probability"] + r["miss_probability"], 1.)

    def test_missing_schedule_and_old_alignment_do_not_invent_odds(self):
        games = league()
        for fixture, season in ((games[:-1], 2026), (league(2001), 2001)):
            result = PlayoffRace(fixture, season, "PIT").report(100)
            self.assertEqual(result["status"], "unavailable")
            self.assertIsNone(result["probability"])
        renamed = [replace(g, home="LAR" if g.home == "LA" else g.home,
                           away="LAR" if g.away == "LA" else g.away) for g in games]
        self.assertIsNone(PlayoffRace(renamed, 2026, "MIN").reason)

    def test_finished_season_uses_actual_postseason_field(self):
        games = league(completed=True)
        # Include PIT and exclude MIN in the observed 14-team field.
        field = ["PIT", "BAL", "CIN", "BUF", "MIA", "KC", "LAC",
                 "GB", "CHI", "DET", "PHI", "DAL", "SEA", "SF"]
        postseason = [replace(games[i], id=f"post-{i}", kind="WC", home=field[2*i], away=field[2*i+1])
                      for i in range(7)]
        for team, expected in (("PIT", 1.), ("MIN", 0.)):
            race = PlayoffRace(games, 2026, team, postseason)
            race.sample({}, random.Random(43))
            result = race.report(100)
            self.assertEqual(result["status"], "observed")
            self.assertEqual(result["probability"], expected)
            self.assertEqual(result["simulations"], 0)

    def test_postseason_results_do_not_change_season_projections(self):
        games = league(completed=True)
        future = next(i for i, g in enumerate(games) if "PIT" in (g.home, g.away))
        games[future] = replace(games[future], home_score=None, away_score=None)
        postseason = replace(games[0], id="post", kind="WC", home="PIT", away="BAL", home_score=0, away_score=50)
        regular = build_dashboard(games + [postseason], 2026, simulations=30, model="baseline")
        all_games = build_dashboard(games + [postseason], 2026, True, simulations=30, model="baseline")
        self.assertEqual(regular["projection"], all_games["projection"])
        self.assertEqual(regular["projection"]["playoffs"]["status"], "estimated")

    def test_existing_ties_are_preserved_in_predicted_record(self):
        games = league(completed=True)
        index = next(i for i, g in enumerate(games) if "PIT" in (g.home, g.away))
        games[index] = replace(games[index], home_score=10, away_score=10)
        d = build_dashboard(games, 2026, simulations=10, model="baseline")
        p = d["projection"]
        self.assertEqual(p["predicted_record"]["ties"], 1)
        self.assertEqual(p["predicted_record"]["wins"] + p["predicted_record"]["losses"] + p["ties"], 17)
        self.assertEqual(p["predicted_record"]["probability"], 1.)

    def test_two_team_head_to_head_precedes_strength(self):
        # Clubs 0 and 1 have identical overall records; 0 won head-to-head.
        s = standings([[0, 2, 0, 0], [0, 0, 2, 0], [2, 0, 0, 2], [0, 0, 0, 0]],
                      [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]])
        self.assertEqual(s.choose([0, 1]), 0)
        self.assertEqual(s.choose([0, 1], [0, 1, 2, 3]), 0)

    def test_multi_team_sweep_and_randomized_unresolved_ties(self):
        s = standings([[0, 2, 2], [0, 0, 2], [0, 0, 0]], [[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        self.assertEqual(s.head_to_head([0, 1, 2], None), {0: 1, 1: 0, 2: 0})
        # A rock-paper-scissors tie must not be broken alphabetically/by rating.
        points = [[0, 2, 0], [0, 0, 2], [2, 0, 0]]
        meetings = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        rng = random.Random(43)
        winners = set()
        for _ in range(30):
            s = standings(points, meetings, rng)
            winners.add(s.choose([0, 1, 2], [0, 1, 2]))
            self.assertTrue(s.random_tie)
        self.assertEqual(winners, {0, 1, 2})

    def test_division_and_conference_records_break_equal_overall_records(self):
        # No head-to-head. Both clubs are 1–1, but 0 won the division game.
        s = standings([[0, 0, 2, 0], [0, 0, 0, 2], [0, 2, 0, 0], [2, 0, 0, 0]],
                      [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]])
        self.assertEqual(s.choose([0, 1], [0, 1, 2]), 0)
        # The cross-conference victory must not improve the conference record.
        s.race.conference = {0, 1, 2}
        self.assertEqual(s.choose([0, 1]), 0)

    def test_strength_uses_opponents_combined_records_and_repeated_meetings(self):
        s = standings([[0, 4, 0], [0, 0, 2], [2, 0, 0]],
                      [[0, 2, 1], [2, 0, 1], [1, 1, 0]])
        self.assertEqual(s.strength(0, True), 1 / 3)
        # Opponent 1 counts twice, opponent 2 once (not mean of percentages).
        self.assertEqual(s.strength(0), 6 / 16)


if __name__ == "__main__":
    unittest.main()
