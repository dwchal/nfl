import copy
import unittest
from dataclasses import replace
from pathlib import Path
import json
from unittest.mock import patch

from steelers.analysis import project_season
from steelers.model import BASELINE, replay_season
from steelers.projection_evaluation import (run, score, simulate_league, snapshot,
                                           season_uncertainty, validate_config)
from test_playoffs import league


CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'docs/experiments/projections.json').read_text())


class ProjectionAuditTests(unittest.TestCase):
    def test_snapshot_hides_own_week_and_later_results(self):
        games = league(2023, completed=True)
        fixtures, cutoff = snapshot(games, 2023, 5)
        self.assertEqual(sum(g.completed for g in fixtures), 4 * 16)
        self.assertTrue(all(not g.completed for g in fixtures if g.week >= 5))
        changed = [replace(g, home_score=0, away_score=100) if g.week >= 5 else g for g in games]
        self.assertEqual(snapshot(changed, 2023, 5), (fixtures, cutoff))

    def test_shared_league_paths_match_dashboard_win_distributions(self):
        fixtures, _ = snapshot(league(2023, completed=True), 2023, 5)
        ratings = replay_season(fixtures, 2023, BASELINE)[0]
        original = ratings.copy()
        result = simulate_league(fixtures, ratings, BASELINE, 100)
        self.assertEqual(result, simulate_league(list(reversed(fixtures)), ratings, BASELINE, 100))
        self.assertEqual(ratings, original)
        self.assertAlmostEqual(sum(r['playoff_probability'] for r in result.values()), 14.)
        self.assertAlmostEqual(sum(r['division_probability'] for r in result.values()), 8.)
        for team in ('PIT','MIN'):
            known = result[team]['known_record']
            dashboard = project_season([g for g in fixtures if not g.completed], ratings,
                                       known['wins'], known['losses'], known['ties'], 100,
                                       team=team, config=BASELINE)
            self.assertEqual(result[team]['distribution'], [{k:r[k] for k in ('wins','probability')}
                                                           for r in dashboard['distribution']])
            self.assertAlmostEqual(sum(r['probability'] for r in result[team]['distribution']), 1.)

    def test_zero_remaining_games_and_known_ties_are_fixed(self):
        games = league(2023, completed=True)
        games[0] = replace(games[0], home_score=10, away_score=10)
        result = simulate_league(games, {}, BASELINE, 5)
        for forecast in result.values():
            self.assertEqual(len(forecast['distribution']), 1)
            self.assertEqual(forecast['distribution'][0]['wins'], forecast['known_record']['wins'])
            self.assertEqual(forecast['distribution'][0]['probability'], 1.)
        with self.assertRaises(ValueError):
            simulate_league(games, {}, BASELINE, 0)

    def test_metrics_have_known_values_and_bootstrap_clusters_whole_seasons(self):
        forecast = {'distribution':[{'wins':1,'probability':.5},{'wins':2,'probability':.5}],
                    'expected_wins':1.5,'low':1,'high':2,'playoff_probability':.75}
        scores = score(forecast, 2, True, 2)
        self.assertEqual(scores['ranked_probability_score'], .25)
        self.assertEqual(scores['win_absolute_error'], .5)
        self.assertEqual(scores['playoff_brier'], .0625)
        rows = [{'season':year,'scores':scores,'playoff_probability':.75,'qualified':True}
                for year in (2022,2023) for _ in range(4)]
        uncertainty = season_uncertainty(rows, 20)
        self.assertEqual(uncertainty['season_blocks'], 2)
        self.assertEqual(uncertainty['intervals']['ranked_probability_score'], [.25,.25])

    def test_future_labels_do_not_change_forecasts_or_selected_config(self):
        games = league(2023, completed=True)
        field = ['PIT','BAL','CIN','BUF','MIA','KC','LAC','GB','CHI','DET','PHI','DAL','SEA','SF']
        postseason = [replace(games[i], id=f'post-{i}', kind='WC', home=field[2*i], away=field[2*i+1]) for i in range(7)]
        config = {**CONFIG,'seasons':[2023],'weeks':[5],'simulations':30,'bootstrap_samples':5}
        def selector(prior, year):
            self.assertTrue(all(g.season < year for g in prior))
            return BASELINE, {}
        with patch('steelers.projection_evaluation.select_model',side_effect=selector):
            first = run(games + postseason, config)
            changed = [replace(g, home_score=0,away_score=100) if g.week >= 5 else g for g in games]
            second = run(changed + postseason, config)
        self.assertEqual(first['elo_configs'], second['elo_configs'])
        for a,b in zip(first['forecasts'],second['forecasts']):
            for key in ('distribution','playoff_probability','known_record','low','high','expected_wins'):
                self.assertEqual(a[key],b[key])
        self.assertNotEqual(first['summary'], second['summary'])
        self.assertEqual(len(first['forecasts']), 32)
        self.assertEqual(first['uncertainty']['season_blocks'], 1)

    def test_incomplete_seasons_are_explicitly_skipped_and_policies_validated(self):
        config = {**CONFIG,'seasons':[2023],'simulations':2,'bootstrap_samples':2}
        report = run(league(2023), config)
        self.assertEqual(report['forecasts'], [])
        self.assertEqual(report['skipped'][0]['season'], 2023)
        self.assertIsNone(report['summary'])
        with self.assertRaises(ValueError):
            validate_config({**config,'cutoff_policy':'Use all final scores'})


if __name__ == '__main__':
    unittest.main()
