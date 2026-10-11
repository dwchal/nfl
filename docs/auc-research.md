# AUC investigation

The requested target is **ROC AUC ≥ 0.80**. This investigation did not reach
it and does not change the production model. AUC is now included in shared
evaluation metrics, dashboard comparisons, and saved-forecast evaluation.

On the same 1,359 regular-season games from 2021–2025 (1,355 decisive games):

| Model / policy | ROC AUC | 95% AUC interval | Brier ↓ | Log loss ↓ |
| --- | ---: | --- | ---: | ---: |
| Chronologically selected Elo | 0.6848 | 0.6566–0.7105 | 0.22412 | 0.64062 |
| Existing full feature model, penalty 0.1 | 0.6902 | 0.6622–0.7176 | 0.22206 | 0.63668 |
| Existing full feature model, penalty 0.03 | 0.6907 | 0.6618–0.7187 | 0.22188 | 0.63631 |
| Existing inner-selected correction policy | 0.6852 | 0.6565–0.7127 | 0.22332 | 0.63939 |
| New inner-selected scoring-margin policy | 0.6821 | 0.6545–0.7085 | 0.22464 | 0.64162 |
| Historical moneyline reference | 0.7226 | 0.6948–0.7488 | 0.21125 | 0.61140 |

Intervals use 1,000 paired season/week block bootstrap draws, seed 42.
They quantify sampling variation conditional on these predictions, and do not
account for model-development selection. These seasons were already explored
in earlier work; these are retrospective development results, not a new
untouched test. The full feature model's AUC improvement over Elo has a paired
95% interval of −0.0026 to +0.0148 at penalty 0.03, which includes zero.

The [AUC audit](auc-evaluation.json) records all existing candidate AUCs,
selected comparisons and intervals, per-season results, historical odds
probabilities, and source/code hashes. Older reports are preserved unchanged.

## What AUC means here

Each decisive game contributes one home-win probability and outcome. AUC is
the fraction of home-win/home-loss pairs ranked correctly, with half credit
for equal probabilities. Game ties are excluded; a sample containing only
one outcome class returns `null`, shown as a dash. Team subsets retain this
home-win orientation. AUC 0.80 does **not** mean 80% correct winner picks.

Uniform monotonic rescaling of probabilities cannot improve their ranking;
confidence calibration alone cannot close this gap. The distinction between
ranking and calibration is also explained in the
[scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html).
Brier and log loss remain necessary to assess the probabilities themselves.

## New model experiment

[Frozen candidate configuration](experiments/ranking.json),
[complete predictions and results](ranking-evaluation.json), and
[implementation](../steelers/ranking.py).

This tests recency-weighted, opponent-adjusted scoring margins as an alternative
to Elo. A ridge regression jointly estimates each team's point strength and a
home advantage, with team priors of zero and home prior of two points. The
12 declared candidates cross half-lives of 8/16/32 calendar weeks, ridge
strengths of 4/12, and absolute margin caps of 21/42. Offseasons retain half
the accumulated evidence. A fixed logistic margin scale of 7.5 converts the
predicted margin into a home-win probability. No new dependencies are needed.

All games on one calendar date are predicted before any scores from that date
update the fit. Earlier postseason games can inform ratings; only regular
season games are evaluated. History begins in 2010. Each evaluation season
chooses its candidate using the preceding two seasons' AUC, and requires both
Brier and log loss to improve over Elo on those earlier seasons. Otherwise it
uses Elo. The choice is frozen for the evaluation year; ratings continue to
learn from completed earlier dates.

The policy selected Elo in 2021, 2022 and 2025; the 16-week/4-ridge/21-cap
candidate in 2023; and the 16-week/4-ridge/42-cap candidate in 2024. It improved
2023 but lost more in 2024, worsening the pooled AUC and probability scores.
It is not promoted. The best fixed candidate reaches only 0.6861 AUC, and
choosing it after seeing the results would not be a valid prospective policy.

## Historical odds reference and next steps

The cached schedule contains both teams' American moneylines for all 1,359
games. Convert each to implied probability, then divide the home probability
by their sum to remove the proportional overround. Invalid or missing odds
use the same game's Elo forecast and are explicitly counted; this snapshot
requires zero fallbacks. The fields are documented by
[nflverse](https://nflreadr.nflverse.com/articles/dictionary_schedules.html).

These historical odds have no issuance timestamps in this snapshot. They
cannot establish what was known 24 hours or one hour before kickoff, and the
audit does not assume a documented closing time. They are a separate reference,
not a production input or proof of a deployable improvement. Their AUC is
also not a theoretical ceiling on what a model could achieve.

The evidence points toward improving information rather than continuing to
adjust confidence or scoring-margin settings. If market inputs are desired,
the next experiment should capture timestamped moneylines at the existing
24h/1h collection deadlines and compare independent, market-only and blended
forecasts on identical games. For independent predictions, collect complete,
timestamped starter and player-availability evidence before testing its
incremental value. Freeze choices before the next prospective evaluation;
do not keep selecting winners on 2021–2025 until an attractive AUC appears.

## Reproduce

```sh
python3 -m steelers.ranking --data .cache/games.csv \
  --config docs/experiments/ranking.json --output docs/ranking-evaluation.json
python3 -m steelers.auc_audit --predictions docs/reduced-evaluation.json \
  --ranking docs/ranking-evaluation.json --data .cache/games.csv \
  --output docs/auc-evaluation.json
python3 -m unittest discover -s tests -v
node --check static/app.js
```

The audit re-scores stored forecasts without refitting them. It rejects
mismatched games, outcomes, teams or baseline probabilities. Reports include
input and Python-source hashes; use the same snapshots to reproduce the exact
numbers. No network downloads, archive writes, or production promotion occur.
