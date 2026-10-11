# Reduced correction models and Elo blend

Implemented as item 6 of the [model improvement guide](model-improvement-plan.md).
The [experiment configuration](experiments/reduced.json) uses protocol
`chronological-v2`: the [shared chronological protocol](chronological-evaluation.md)
plus declared advanced feature groups and an optional Elo probability blend.
The [full report](reduced-evaluation.json) contains every scored game, fitted
coefficients, clipping rates, training-row counts, optimizer diagnostics,
per-year blend choices and source/code hashes.

**Outcome: no reduced model, penalty or blend is promoted.** Fewer features did
not generalize better than the full model, the penalty mattered more than the
feature group, and the inner-fold blend never improved on plain selection. The
application default remains Elo; the advanced selector is unchanged.

## Reproduce from local snapshots

```sh
python3 -m steelers.experiments --data .cache/games.csv \
  --features .cache/features --advanced .cache/advanced \
  --start 2021 --end 2025 --config docs/experiments/reduced.json \
  --output docs/reduced-evaluation.json
```

The checked-in report was produced from local snapshots with schedule SHA-256
`88077a76db885d3078b2718f5cb104dac24f4dfb9c201f974b3d31a368e1f951`. That
schedule is newer than the one behind the v1 report; rerunning the v1
configuration under the v2 runner reproduces every 2021–2025 forecast of the
[checked-in v1 report](chronological-evaluation.json) exactly, so the newer
snapshot only changed 2026 rows. The experiment file SHA-256 is
`e84f447120eb520d283bde53e0b29eff692e4ef18b592b985ad26ed58b2dc915`; its
canonical configuration digest is
`f857e501b9f72692682e86ca1c9751f3152d765d16ef605364d529e8dc9f2ea2`. Two
independent runs produce byte-identical output. Output takes about 100 seconds.

## Protocol additions (`chronological-v2`)

Everything in the v1 frozen protocol still applies: strictly earlier Elo
selection, eight-year feature warmup, two inner validation seasons, three to
six training seasons, 95% source coverage, pooled inner Brier then log loss
then declared order, both probability scores must beat Elo, refit before each
outer season, identical game sets. v1 configurations still validate and run,
producing identical per-game output.

1. **Feature registry.** `steelers/advanced.py` names every advanced feature in
   `REGISTRY` and declares groups by label: `full` (34 labels, still subject to
   the weather/availability support gates), `qb_weekly_travel` (weekly
   pass/rush offense and defense, QB change, five travel features),
   `qb_situational_travel` (QB change, 16 situational features, travel), and
   `travel`. The reduced groups omit the extra rest coefficient because
   calibrated Elo already supplies it. A declared group can only remove
   coefficients; it cannot enable an unsupported optional feature. Unknown
   labels, duplicate labels, and width mismatches fail loudly.
2. **Candidate grid.** Each advanced candidate names a group and an L2 penalty
   from {0.03, 0.1, 0.3}. Candidates are declared simplest-first and Elo stays
   first, so ties resolve toward fewer coefficients. Thirteen candidates score
   the same 1,359 games.
3. **Blend.** After the usual selection, the runner evaluates
   `p = (1 − α)·p_elo + α·p_selected` for α in {0, 0.25, 0.5, 0.75, 1} on the
   same inner-fold predictions. α = 0 reproduces Elo exactly and α = 1 the
   correction exactly. The smallest α wins ties, and a nonzero α must beat
   α = 0 on both pooled inner scores. The chosen α is frozen for the outer
   season and recorded on every game row. Nothing is fitted on in-sample
   probabilities or outer outcomes.
4. **Diagnostics.** Each fit reports its active labels, training rows,
   per-feature clipping rate at the ±4 bound, active coefficient count and
   optimizer status. Coverage on game rows is keyed by family in v2 because
   candidates within a family share one replay row.

## Results: 2021–2025, 1,359 games

| Candidate | Coefficients | Brier | Log loss | Decisive accuracy | Brier delta vs Elo [95% interval] |
| --- | ---: | ---: | ---: | ---: | --- |
| Elo (chronologically selected) | 0 | 0.224121 | 0.640620 | 62.66% | — |
| travel, penalty 0.3 | 5 | 0.223300 | 0.639200 | 63.54% | −0.000820 [−0.001872, +0.000144] |
| travel, 0.1 | 5 | 0.223156 | 0.639038 | 63.76% | −0.000965 [−0.002524, +0.000448] |
| travel, 0.03 | 5 | 0.223122 | 0.639040 | 63.76% | −0.000999 [−0.002913, +0.000715] |
| QB + weekly + travel, 0.3 | 10 | 0.222811 | 0.638177 | 63.54% | −0.001309 [−0.002503, −0.000119] |
| QB + weekly + travel, 0.1 | 10 | 0.222283 | 0.637201 | 63.84% | −0.001837 [−0.003804, +0.000186] |
| QB + weekly + travel, 0.03 | 10 | 0.222000 | 0.636640 | 64.87% | −0.002121 [−0.004897, +0.000614] |
| QB + situational + travel, 0.3 | 22 | 0.222595 | 0.637698 | 63.69% | −0.001526 [−0.002924, −0.000139] |
| QB + situational + travel, 0.1 | 22 | 0.222139 | 0.636846 | 64.80% | −0.001982 [−0.004208, +0.000262] |
| QB + situational + travel, 0.03 | 22 | 0.221897 | 0.636324 | 64.35% | −0.002223 [−0.005223, +0.000838] |
| full, 0.3 | 27 | 0.222474 | 0.637448 | 63.76% | −0.001647 [−0.003226, −0.000031] |
| full, 0.1 (v1 advanced) | 27 | 0.222061 | 0.636680 | 64.65% | −0.002060 [−0.004517, +0.000420] |
| full, 0.03 | 27 | 0.221880 | 0.636306 | 64.43% | −0.002240 [−0.005480, +0.000987] |
| Inner-selected policy | — | 0.223320 | 0.639385 | 64.58% | −0.000800 [−0.003606, +0.001999] |
| Inner-selected policy, blended | — | 0.223603 | 0.639922 | 64.65% | −0.000517 [−0.003176, +0.002116] |

Weather and availability coefficients stay disabled by their support gates in
every fit, so `full` has 27 active coefficients. All 36 outer and inner fits
converge; there are no fallback games.

Per-year Brier for selected candidates:

| Year | Elo | travel 0.1 | QB+weekly+travel 0.03 | QB+situational+travel 0.03 | full 0.3 | full 0.1 | full 0.03 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021 | 0.23098 | 0.22709 | 0.22441 | 0.22532 | 0.22790 | 0.22653 | 0.22533 |
| 2022 | 0.22295 | 0.22429 | 0.22313 | 0.22268 | 0.22160 | 0.22170 | 0.22229 |
| 2023 | 0.23374 | 0.23273 | 0.23238 | 0.23184 | 0.23282 | 0.23242 | 0.23168 |
| 2024 | 0.21057 | 0.20874 | 0.20216 | 0.20309 | 0.20570 | 0.20343 | 0.20196 |
| 2025 | 0.22236 | 0.22294 | 0.22792 | 0.22657 | 0.22434 | 0.22621 | 0.22815 |

The inner-selected policy chose Elo in 2021, QB + weekly + travel at 0.03 in
2022 and 2023, and full at 0.03 in 2024 and 2025. The blend chose α = 1 in every
selected year except 2024 (α = 0.75). The policy's 2024 gain (−0.0086) and 2025
loss (+0.0058) are each larger than the five-year pooled effect of any
candidate.

## What this says

- **Fewer inputs did not help.** Every reduced group scores worse than the full
  model at the same penalty, although the 22-feature situational group is
  within 0.00002 of full at penalty 0.03. The overlap between weekly and
  situational efficiency is not the source of the 2025 loss.
- **Penalty dominates group.** At each group, 0.03 has the best pooled Brier
  and 0.3 the worst; the differences across penalties are as large as the
  differences across groups. The more heavily shrunk models have smaller but
  more consistent gains: at penalty 0.3, three candidates' Brier intervals
  exclude zero, while no 0.03 or 0.1 interval does. Less shrinkage buys
  average accuracy at the cost of year-to-year variance.
- **The effect flips sign by season.** All correction models improve on Elo
  in 2021, 2023 and 2024 and worsen it in 2025 (and travel-only in 2022). The
  2025 weekly/PBP snapshots have the same row counts and no missing EPA as
  2024, so this is not a coverage problem: corrections make forecasts more
  confident (mean |p − 0.5| 0.161 versus 0.141 for Elo) and 2025 punished
  confidence. Two inner seasons cannot anticipate that.
- **Blending does not hedge this.** The inner folds say the full correction is
  best, so α = 1 is chosen and the blend inherits the same 2025 loss; the one
  α = 0.75 year (2024) gave up part of a large gain. A blend chosen on earlier
  predictions cannot protect against a regime change in the scored year.
- **Pittsburgh remains worse under the full model** (0.26889 at penalty 0.1
  versus 0.26454 for Elo) and better under travel-only (0.26330); Minnesota is
  the reverse. With 85 games per team these are not reliable signals.

These seasons were explored during development and this grid was chosen with
knowledge of the earlier reports, so the intervals do not account for that
selection. The live [paired forecast archive](forecast-collection.md) remains
the source of prospective evidence. Per the shared gate, the simpler incumbent
is retained.

## Not done here

Item 6 step 5 (recency decay and count-prior grids) requires threading a frozen
`FeatureConfig` through `FeatureState` and `Situations` and replaying features
from scratch per configuration; it is deferred to a separate change so that
this report measures groups, penalties and the blend alone. No blend setting is
archived with live forecasts because none is deployed.
