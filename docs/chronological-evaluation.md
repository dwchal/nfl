# Shared chronological model evaluation

Implemented as item 3 of the [model improvement guide](model-improvement-plan.md).
Protocol `chronological-v2` ([reduced models and blend](reduced-model.md)) extends
this runner; v1 configurations still run and reproduce this report's forecasts.
The [experiment configuration](experiments/chronological.json) declares the
candidate order, penalties, seasons, coverage thresholds, cutoff, selection
gate, and bootstrap settings. The [full report](chronological-evaluation.json)
contains every scored game, fitted coefficients, optimizer diagnostics, and
source/code hashes. Historical snapshots are preserved.

## Reproduce from local snapshots

Run from the repository root with the existing schedule, weekly statistics,
and prepared advanced inputs:

```sh
python3 -m steelers.experiments --data .cache/games.csv \
  --features .cache/features --advanced .cache/advanced \
  --start 2021 --end 2025 --config docs/experiments/chronological.json \
  --output docs/chronological-evaluation.json
```

The command uses only Python's standard library. It does not download inputs,
capture forecasts, modify evidence, or change the application's model choice.
SQLite evidence is read through `mode=ro`; a missing database is not created.
Omit `--advanced` to audit with no PBP/evidence sources. Missing or invalid
optional inputs are recorded in the manifest and result in explicit fallbacks.
Output cannot replace the schedule, experiment file, or files inside input
directories. A changed raw snapshot or implementation changes the hashes.

The checked-in report was produced from the local snapshots with schedule SHA-256
`a06ca5f608332ba4936d5f13f8868d92bf423b9d8b88b3c2dba293162575cfac`.
The canonical experiment SHA-256 is
`25f69d837d25136ed0124e230a94811c558673238f815d74dce396ce4ddede02`.
File bytes and canonical configuration have separate hashes in the report.
The summary is indented; each game occupies one JSON line to make comparisons
manageable. The entire artifact is a single valid JSON document.

## Frozen selection protocol

1. A complete season has at least 200 listed regular-season games, all completed.
   Partial seasons are listed under `skipped` and excluded from headline scores.
   Completeness assumes the supplied schedule contains the full league slate.
   Evaluation seasons are declared in the configuration; the CLI may run a subset.
2. For every season, select Elo settings with `select_model` receiving only
   games from strictly earlier seasons. Preserve its existing margin,
   calibration, and deployment gates. Replay that year's games with those fixed
   settings and a three-year Elo warmup. Save the exact configuration.
3. Build weekly/QB and advanced features with an eight-year warmup specific to
   each forecast year. Statistics are observed only on later calendar days;
   postseason games can update state, but only regular games are scored.
   Extending the evaluation range cannot shorten an earlier year's warmup.
   Timestamped QB/weather/availability evidence must precede its game cutoff.
   These are reconstructed kickoff forecasts, not archived live forecasts.
4. For outer year `Y`, reserve the last two earlier complete seasons as inner
   validation years. For each candidate and inner year `V`, fit the latest up
   to six complete, source-eligible seasons strictly before `V`, with at least
   three required. History starts in 2016. Each fit reports selected/excluded
   years, coverage, weights, and optimizer status; there is no fitting on `V`.
5. Source eligibility requires at least 95% team/QB dataset coverage per season;
   advanced also requires 95% PBP coverage. Candidates can have different
   available training years: weekly files start in 2016 and PBP in 2018.
   Inadequate history or optimizer failure yields Elo forecasts for the fold.
   All inner and outer comparisons enforce identical ordered game IDs.
6. Rank candidates by pooled inner Brier error, then log loss, then the declared
   simpler-first order: Elo, six-feature matchup (penalty 0.03), 34-feature
   advanced (penalty 0.1). The best candidate must improve **both** probability
   scores against Elo. Otherwise select Elo. Fallback forecasts remain included
   in this comparison. Outer outcomes never make this decision.
7. Refit each candidate on eligible seasons strictly before `Y`; freeze the
   coefficients throughout `Y`. Score both the individual candidates and the
   policy chosen from the inner folds. Earlier completed games within `Y` may
   update Elo ratings and feature histories, as they would in use.
8. Missing pregame weekly/PBP histories trigger explicit per-game Elo fallback.
   A new QB can use a neutral shrunk prior. Unknown weather/availability has a
   neutral contribution, with missing metadata; optional coefficients still
   need the existing 100 eligible / 20 zero / 20 nonzero training observations.
   These support gates use only training rows, never the scored season.
9. Report pooled and annual probability scores, decisive-game accuracy, PIT/MIN
   subsets, reliability bins, available/missing-input subsets, winner changes,
   and paired season/week bootstrap intervals (2,000 draws, seed 42). Ties are
   outcomes of 0.5 for probability scores and excluded from pick accuracy.

Per-game rows include the UTC cutoff, outcome, incumbent and all candidate
probabilities, selected configuration, exact Elo settings, fit references,
pregame sample metadata, fallback reasons, and experiment/input hashes. `fits`
contains the weights and diagnostics referenced by each game and fold.
`elo_configs` also records earlier training-year configurations.

## Results: 2021–2025

All choices score the same **1,359 games**, including four ties. There are no
outer-season model fallbacks in these snapshots. The advanced model falls back
to Elo in its 2019 and 2020 inner folds because it has fewer than three earlier
PBP seasons; those forecasts remain in candidate selection.

| Choice | Brier | Log loss | Decisive accuracy | Net correct picks vs Elo |
| --- | ---: | ---: | ---: | ---: |
| Chronologically selected Elo | 0.224121 | 0.640620 | 62.66% | 0 |
| QB matchup | 0.222987 | 0.638071 | 62.58% | -1 |
| Advanced | 0.222061 | 0.636680 | 64.65% | +27 |
| Inner-selected policy | 0.223150 | 0.638683 | 63.76% | +15 |

The inner-selected choices are Elo (2021), advanced (2022 and 2023), matchup
(2024), and advanced (2025). In 2025 both correction models worsen Brier and
log loss against Elo; that year's outcomes cannot retroactively change selection.

Advanced Brier delta is -0.002060, with a paired 95% interval of
[-0.004517, +0.000420]. The selected policy's delta is -0.000971,
with interval [-0.003138, +0.001241]. Both include no probability-score
improvement. Advanced's decisive accuracy delta is +1.99 percentage points,
with interval [+0.52, +3.50]; the selected policy's interval still includes zero.
These intervals do not account for prior model-development selection.

Pittsburgh Brier worsens from 0.264540 (Elo) to 0.268889 (advanced); Minnesota
improves from 0.223600 to 0.221364. There are 85 games per team. There are 357
eligible weather records and no complete availability records; three advanced
rows have no usable history for the projected QB. Input subset metrics describe
these samples and do not establish the causal value of a feature.

These years have already been explored during development. This is a
retrospective audit, not an untouched holdout or a production promotion. Weather
publication timing, provider revisions, and projected starters retain the
limitations described in [advanced-model.md](advanced-model.md).

## Application compatibility and verification

The existing `evaluate` and `evaluate_advanced` entry points and report fields
remain available. They share the pure correction fitting helper and historical
Elo-offset helper. QB training/comparison offsets now use each historical
season's prior-only settings instead of the requested target year's settings.
Both models report exact historical configurations and use version `v4`.
Failed QB fits now report `unavailable`, matching advanced fallback behavior.

The application still uses its existing QB development gate and explicit
experimental advanced selector. It does not adopt the runner's inner-selected
policy automatically. The [updated 2026 QB report](chronological-matchup-2026.json)
retains the application's older three-fit/three-comparison protocol with the
offset bug fixed: comparison Brier 0.222224 → 0.222369 and log loss
0.635855 → 0.636009. The challenger fails the gate, so Elo remains the default.
Use the common audit above for cross-model comparisons.

`tests/test_experiments.py` checks the entire selection path under outer-score,
weekly/QB-statistic and PBP mutations, earlier forecast invariance, fit-season
boundaries, exact Elo selection inputs, candidate game alignment, failed and
missing-input fallbacks, range extension, ties, deterministic JSON, and read-only
local input loading. The next priority is item 4: capture comparable prospective
forecasts at explicit times instead of depending on dashboard visits.
