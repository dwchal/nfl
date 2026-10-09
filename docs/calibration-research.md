# Confidence and rest calibration research

This change improves probability calibration slightly; it does not demonstrate a
material increase in winner-pick accuracy. It retains the standard-library-only
runtime and uses the same cached nflverse schedule as the earlier evaluations.
The input SHA-256 is recorded in both new JSON snapshots.

## Retained model

The existing 28-candidate Elo search and its promotion gate remain intact. A
second candidate calibrates the Elo model chosen by the older tuning window:

```text
log_odds = log(p_elo / (1 - p_elo))
rest = (min(home_rest, 14) - min(away_rest, 14)) / 7
p_calibrated = sigmoid(scale * log_odds + rest_coefficient * rest)
```

Negative rest values are clipped to zero; missing rest on either side makes the
rest difference zero. Neutral sites receive no Elo home advantage, but known
rest differences still apply. The underlying Elo updates are unchanged, so
rest is not stored as persistent team strength. Season simulations use the same
calibrated probabilities and retain the previous win/loss-only rating updates.

Two coefficients minimize mean logistic loss plus an L2 penalty of 0.03,
shrinking scale toward 1 and the rest coefficient toward 0. Scale is bounded to
0.5–1.5; rest coefficient to −1–1. These fixed bounds limit extrapolation on
unusual histories. No current-season results fit either parameter.

For the 2026 model, fitting uses 2020–2022; deployment compares with the previous
default on 2023–2025. Both Brier and log loss must improve. This comparison is a
deployment gate, so it is not an untouched audit. A failed gate retains the
previous default. The independent season audit below repeats the entire gate
using only seasons preceding each audited season.

## Results

| Comparison | Games | Previous Brier | New Brier | Previous log loss | New log loss |
|---|---:|---:|---:|---:|---:|
| 2026 deployment comparison: 2023–2025 | 816 | 0.221939 | 0.221810 | 0.635599 | 0.634927 |
| Complete policy, walk forward: 2018–2025 | 2,127 | 0.222120 | 0.221934 | 0.637376 | 0.636771 |

The 2026 fit gives scale 0.91333 and rest coefficient 0.10807. Confidence becomes
slightly more conservative. The 816-game comparison improves Pittsburgh's
Brier error and worsens Minnesota's; the longer policy audit slightly improves
both teams' Brier error. Winner accuracy in the longer audit is unchanged at
63.38% overall and falls for Pittsburgh. Each report includes annual and team
subsets to make these differences visible.

The walk-forward audit freezes fitting and deployment before each target season.
It compares the previous default policy with the new policy, including years
when calibration fails its prior-season gate and predictions remain identical.
Calibration activates in 2019, 2021, and 2025 among the eight audited seasons.
Incomplete target seasons are excluded. This audit covers the schedule model;
it does not evaluate the optional QB matchup model or simulated season ranges.

A paired bootstrap resamples season/week blocks, keeping the two forecasts for
each game together. Across 2,000 draws with seed 42, the 95% interval for
candidate-minus-previous Brier error is approximately −0.000453 to +0.000054.
The log-loss interval also includes zero. These estimates do not establish a
statistically clear gain. They also do not account for model-development
selection, longer-term dependence, or later provider revisions.

## Alternatives examined during this work

All parameter selection below used earlier seasons. However, several methods
were compared against the already-examined 2023–2025 period during development.
That makes this research exploratory; the live forecast archive remains the
appropriate source of genuinely prospective evidence.

| Alternative | Selection / fitting | Outcome |
|---|---|---|
| Linear score-residual ratings | 27 candidates on 2020–2022; update proportional to capped margin minus expected margin | Worse tuning and later Brier than existing Elo; excluded |
| Matchup feature-family and penalty selection | Fit 2020–2021, select on 2022; rest / efficiency / full QB groups, penalties 0.03, 0.1, 0.3 | Selected efficiency + rest, penalty 0.03; later Brier 0.222269, worse than Elo |
| Annual matchup refitting | Existing full model, penalty 0.03; expand training from 2020 before each later season | Later Brier 0.222713, worse than Elo |
| Finer Elo parameter grid | 125 candidates on 2020–2022 | Selected home advantage 15; later Brier 0.222108, worse than incumbent |
| Longer Elo tuning window | Original 28 candidates on 2013–2022 | Selected the same incumbent settings |
| Average top Elo candidates | Tune ensemble size 1, 3, 5, 10, 28 on 2020–2022 | Selected size 1; unchanged model |
| Elo / score-residual probability blend | Blend weights 0, 0.25, 0.5, 0.75, 1 on 2020–2022 | Selected score-residual weight zero; unchanged model |
| Rest-only probability correction | Fit on 2020–2022 with penalty 0.03 | Later Brier 0.221855; small gain, but worse for both team subsets |
| Confidence-only correction | Fit on 2020–2022 with penalty 0.03 | Later Brier 0.221902; small gain |
| Joint confidence + rest | Fit on 2020–2022 with penalty 0.03 | Retained candidate; later Brier 0.221810; broader walk-forward audit above |

The selected calibration's two coefficients are learned from the older window;
they are not fitted to the later comparison. The decision to retain this model
family was informed by exploratory comparisons. No failed alternative is added
to the runtime's candidate grid. The existing QB scenario model remains
experimental because it still fails the probability-score gate when trained
against the calibrated Elo offsets.

## Reproduce

Use the same schedule and weekly-statistics cache hashes to reproduce the stored
numbers exactly. `--offline` prevents replacement of those inputs. New provider
snapshots may produce different results.

```sh
python3 app.py --backtest --season 2026 --offline > docs/calibrated-model-evaluation-2026.json
python3 -m steelers.evaluation --data .cache/games.csv --start 2018 --end 2025 --output docs/calibration-walk-forward.json
python3 -m unittest discover -s tests -v
node --check static/app.js
```

The audit can use any local nflverse-format schedule; it makes no network calls,
does not load optional player feeds, and does not write to the forecast archive.
The older JSON snapshots remain as records of the previous implementation.
