# Quarterbacks, season projections, and smaller models

Implemented 2026-10-11: four follow-ups requested after the record/playoff
dashboard additions. All runtime code remains Python 3.10+ standard library.
The production starter-evidence correction is enabled; research candidates do
not change the default forecast model or apply a new playoff calibration.

## 1. Quarterback evidence correctness

`Evidence.quarterback` now reconciles eligible confirmations, timestamped team
depth charts, and per-player availability reports before choosing a starter.
A later Out/Inactive report invalidates an older confirmation. The returned
replacement includes the conflict and both timestamps, visible in the advanced
dashboard's pregame-evidence section. An omitted player in a partial report
cannot silently clear an earlier Out. A later sourced status reversal or a
newer sourced confirmation can reverse the exclusion. Records at or after the
forecast cutoff remain ineligible.

Depth candidates are sorted by rank. When every eligible depth candidate is
ruled out, the result is an explicitly unknown starter; it cannot restore an
old passer who is absent from that chart. Without eligible membership evidence,
the previous-passer fallback is marked unverified, including across offseasons.
Today's roster is never used to assert historical membership. Evidence remains
append-only. The integrated advanced model version is `advanced-v7`.

This integrates with the candidate lists, learned starter mixtures, conditional
forecasts and advanced rushing feature from the concurrently published
[starter-uncertainty work](quarterback-model.md). Those interfaces are preserved.
Mixtures still require enough earlier timestamped labels and remain experimental;
the application retains deterministic starters and passing-only advanced features.
The evidence correction also handles partial-report omissions and sourced reversals.

## 2. QB rushing experiment

This seven-feature weekly matchup experiment is separate from the preserved
35-feature advanced experiment in [quarterback-model.md](quarterback-model.md).
It classifies rushers using annual roster positions and uses its own cache and
state; the advanced experiment uses `situations-v2` and verified weekly QB
positions. Their feature definitions and baselines differ, so their scores
are separate comparisons. Neither is promoted.

`steelers/qb_rushing.py` prepares a separate `qb-rushing-v1` cache. It preserves
the original downloaded PBP/annual-roster files locally and records their URLs
and SHA-256 hashes. Annual roster positions classify player roles only; they
do not establish a starter or team membership at a historical prediction time.
Source definitions: [nflverse PBP dictionary](https://nflreadr.nflverse.com/articles/dictionary_pbp.html).

The aggregate counts identified QB scrambles and designed rushes, excludes
kneels, spikes and no-plays, and retains EPA and carries separately. Unknown
IDs/ambiguous positions/missing EPA are recorded as unknown, not neutral runs.
A team-game with unclassified rushing observations cannot establish complete
coverage. All 2016–2025 regular-season games in the prepared snapshots have
classified coverage. Passing/dropback definitions in the existing PBP cache
remain unchanged.

The research state observes only earlier eligible game statistics. Rushing
quality uses an earlier league QB-rushing mean, a fixed 50-carry neutral prior,
0.95 decay per observed QB-rushing performance and 0.8 QB offseason decay. The new predictor is
the assumed QB's rushing quality minus the team's recent QB-rushing quality,
differenced across the matchup. Team rushing history uses 0.9 observation decay
and 0.65 offseason decay. These settings were specified before scoring; they
were not optimized against the outer outcomes.

The new seven-feature family uses the existing chronological adapter. Training
requires at least three earlier eligible seasons with 95% weekly and classified
rushing coverage. Missing earlier team history produces an explicit Elo
fallback on that game; the game stays in the scores. New QBs use the neutral
prior. Fitting, candidate selection, and penalties use earlier seasons only.

Reproduce:

```sh
python3 -m steelers.qb_rushing --start 2016 --end 2025
python3 -m steelers.experiments --data .cache/games.csv \
  --features .cache/features --advanced .cache/advanced --rushing .cache/qb-rushing \
  --start 2021 --end 2025 --config docs/experiments/qb-rushing.json \
  --output docs/qb-rushing-evaluation.json
```

The preparation command may download inputs; the experiment itself is local-only
and read-only with respect to inputs/evidence/forecast archives. Neither command
is invoked by the dashboard or its scheduled collector.

### Results: identical 1,359 games, 2021–2025

| Candidate | Brier | Log loss | Decisive accuracy |
| --- | ---: | ---: | ---: |
| Prior-selected Elo | 0.224121 | 0.640620 | 62.66% |
| Passing-only QB matchup | 0.222987 | 0.638071 | 62.58% |
| Passing plus QB rushing | 0.222928 | 0.638019 | 62.95% |

Against passing-only, rushing corrects 13 picks and introduces 8 new errors:
net **5** additional correct picks. Its Brier difference is **−0.000059**, with
paired week-block 95% interval **[−0.000681, +0.000561]**; log-loss difference
is **−0.000053**, interval **[−0.001373, +0.001305]**. The gain is small and
uncertain. There are zero outer-game fallbacks. Keep the candidate experimental.
The nested policy selects Elo, rushing, rushing, passing-only, passing-only for
2021–2025; its Brier is 0.223471. This policy is research, not a deployment change.
Rushing worsens passing-only Brier in 2023 and 2024. It also worsens both target
team subsets: PIT 0.268729 → 0.269905, MIN 0.220322 → 0.221977 (85 games each).
These adverse results strengthen the decision to retain it only as an experiment.

[Full report](qb-rushing-evaluation.json), including annual/team metrics,
coefficients, source hashes, fallback reasons and direct paired comparisons.

## 3. Validate season records and playoff probabilities

`steelers/projection_evaluation.py` reconstructs forecasts before weeks 1, 5, 9
and 13 of 2018–2025. Every cutoff is midnight Eastern at the start of the first
scheduled day of that week. All later regular-season scores are masked. Elo
settings are selected from strictly earlier seasons; warmup and state updates
see only eligible completed games. Future scores and the actual postseason
field are used only as labels after forecasting.

Each checkpoint simulates one shared league 5,000 times. All 32 teams share
each outcome path and both conferences use the existing approximate playoff
tiebreaker logic. This avoids treating each team's opponents as independent
leagues. It preserves the dashboard's sorted-game, seeded win distribution.
The dashboard also now sorts simulation fixtures internally for reproducibility.

The audit records every win distribution, expected wins, middle 80% range,
known record, playoff probability, division/wild-card split and actual outcome.
It reports win MAE, discrete ranked probability score, interval coverage/width,
playoff Brier/log loss and reliability bins. Summaries include each checkpoint,
each season and PIT/MIN. Uncertainty resamples whole seasons, retaining all
teams and checkpoints together, with 2,000 draws and seed 42.

```sh
python3 -m steelers.projection_evaluation --data .cache/games.csv \
  --config docs/experiments/projections.json --output docs/projection-evaluation.json
```

### Results: 1,024 team/checkpoint forecasts

| Checkpoint | Mean absolute win error | Ranked probability score | 80% range coverage | Playoff Brier |
| --- | ---: | ---: | ---: | ---: |
| Before week 1 | 2.322 | 1.623 | 76.6% | 0.2215 |
| Before week 5 | 1.873 | 1.286 | 82.4% | 0.1618 |
| Before week 9 | 1.493 | 1.033 | 80.9% | 0.1372 |
| Before week 13 | 1.051 | 0.719 | 86.3% | 0.1092 |
| All checkpoints | 1.685 | 1.165 | 81.5% | 0.1574 |

The pooled range-coverage interval is **[76.8%, 86.5%]**. Pooled playoff
log loss is 0.4695; Brier interval **[0.1394, 0.1767]**. Reliability is uneven:
the 10–20% bin averages 14.9% predicted qualification but 27.9% actual, while
the 90–100% bin averages 96.7% predicted and 91.7% actual. These are diagnostics,
not sufficient evidence for fitting a new calibration on these same outcomes.

Limits: eight season blocks, reconstructed final schedules, previously explored
years, no future ties, no simulated points/touchdowns, and unresolved standings
ties randomized. The audit validates **prior-selected Elo projections only**;
QB/advanced season corrections have not been audited by this new runner.
The 2022 canceled BUF–CIN game is retained as the actual shortened schedule.
No seasons were skipped in the local run. Simulation count is 5,000 here versus
10,000 in the dashboard; Monte Carlo error is not included in bootstrap intervals.
No margin-sampling or latent-strength variant is adopted by this validation.

[Full report](projection-evaluation.json), including all distributions and labels.

## 4. Smaller-model comparison

The shared runner now accepts explicit, validated feature indices and preserves
full-width coefficient vectors with excluded coefficients exactly zero. Advanced
feature masks intersect the existing evidence-support gates. It reports ordered
feature labels, enabled indices, penalties and fit diagnostics. Optional declared
pairwise comparisons use the same existing paired uncertainty implementation.

The frozen grid tests three reduced groups at L2 penalties 0.1 and 0.3:
passing + QB (3 features), weekly efficiency + QB without extra residual rest
(5 features), and weekly/QB + travel (11 features). Existing six-feature matchup
(0.03), full advanced (0.1), and Elo remain reference candidates. Candidate and
penalty selection uses the same two earlier inner validation seasons; outer
games do not select the candidate. Every choice scores identical games.

```sh
python3 -m steelers.experiments --data .cache/games.csv \
  --features .cache/features --advanced .cache/advanced \
  --start 2021 --end 2025 --config docs/experiments/smaller-models.json \
  --output docs/smaller-models-evaluation.json
```

### Results: identical 1,359 games, 2021–2025

| Candidate | Brier | Log loss | Decisive accuracy |
| --- | ---: | ---: | ---: |
| Passing + QB, L2 0.1 | 0.223311 | 0.638728 | 62.51% |
| Weekly + QB without extra rest, L2 0.1 | 0.223175 | 0.638586 | 62.80% |
| Weekly/QB + travel, L2 0.1 | 0.222306 | 0.637224 | 63.69% |
| Existing six-feature matchup | 0.222987 | 0.638071 | 62.58% |
| Full advanced | 0.222061 | 0.636680 | 64.65% |

Every reduced group has worse pooled probability scores than its corresponding
full reference; penalties 0.3 perform worse than 0.1 in these pooled diagnostics.
The 11-feature travel model's Brier difference versus full advanced is
**+0.000246**, interval **[−0.000598, +0.001066]**. Its picks lose a net 13
correct games. There is no supported replacement for the full model here.
The nested policy's Brier is 0.223126 and accuracy 63.54%, with zero outer
fallbacks. This selected policy also remains experimental.
The 11-feature model improves Pittsburgh Brier relative to full advanced
(0.268889 → 0.266994) but worsens Minnesota (0.221364 → 0.222442).
Every tested correction still worsens Elo Brier in 2025. Team/season slices
are diagnostics and were not used to retune the grid after evaluation.

[Full report](smaller-models-evaluation.json) includes every grid candidate,
inner/outer selection, annual/team scores and direct paired comparisons.
This implements a specific reduced-feature/penalty comparison from roadmap
item 6; decay/prior grids and other feature transformations remain separate work.

## Verification and next work

Regression tests cover later-Out conflicts and explicit reversals, partial
report omissions, unknown starters and membership; QB-rushing play filters,
coverage fallback and own/future/same-day isolation; feature masks; checkpoint
score masking, prior-only Elo selection, all-team shared paths, exact seeded
win-distribution agreement, known ties, zero remaining games, score formulas,
season-block resampling and incomplete-season exclusion. All **183 tests pass**;
JavaScript syntax and Git whitespace checks pass. The three final reports have
source hashes verified against the final Python files.

Keep Elo as the current default. Continue prospective paired collection, measure
coverage at the declared horizons, and freeze an evaluation period before any
promotion. A later projection experiment can test margin-aware path updates or
persistent team-strength uncertainty using the new audit as the baseline.
