# Starter uncertainty and quarterback rushing

Implemented as item 5 of the [model improvement guide](model-improvement-plan.md).
The [experiment configuration](experiments/quarterbacks.json) scores four
advanced variants against Elo under the
[shared chronological protocol](chronological-evaluation.md) (`chronological-v2`):
passing-only features with deterministic starters (the previous `advanced-v5`
model), the same features with a starter **mixture**, and both again with the
new **QB rushing** feature. The [full report](quarterback-evaluation.json)
contains every scored game, the per-game starter candidate lists, fitted
coefficients and source/code hashes.

**Outcome: neither the mixture nor the rushing feature changes accuracy, so
neither is deployed.** The evidence-ordering fixes, the candidate lists and the
conditional forecasts are live because they are correctness and transparency
improvements; the application's advanced model keeps deterministic starters and
the passing-only label set. The default remains Elo.

## Reproduce from local snapshots

```sh
python3 -m steelers.prepare --start 2018 --end 2026 --refresh   # situations-v2 aggregates
python3 -m steelers.experiments --data .cache/games.csv \
  --features .cache/features --advanced .cache/advanced \
  --start 2021 --end 2025 --config docs/experiments/quarterbacks.json \
  --output docs/quarterback-evaluation.json
```

Preparation re-downloads nflverse play-by-play so that `pbp_<season>.json`
carries the `qb_rushing` map; `situations-v1` files remain valid but give the
rushing feature no support. Two independent runs produce byte-identical output.
The checked-in report used schedule SHA-256
`88077a76db885d3078b2718f5cb104dac24f4dfb9c201f974b3d31a368e1f951`; the
experiment file's SHA-256 is
`3506f3030b1ccbf279c313abab25cc0b01f7ffa6ef8a8041fb2a33cf7f9ff75a` and its
canonical configuration digest
`0c76c7f04ff6bbee309989f7ed72d89fdc158c8de4f8dfd20a230dc3737ca6e7`. Weekly,
play-by-play and evidence hashes are inside the report. All 29 fits converge.

## What changed

1. **Evidence ordering.** `Evidence.quarterback_candidates` loads the eligible
   confirmation, depth and availability records. An Out/Inactive report
   issued after a confirmation invalidates it and the conflict is reported with
   both timestamps; an earlier report cannot. A previous passer missing from
   the current depth chart is rejected; without any depth evidence the fallback
   is kept but labeled unverified. All candidates ruled out gives an unknown
   starter with the reason. `quarterback()` is now a compatibility view of the
   top candidate.
2. **Candidate lists.** Each team's starter is a list
   `[{id, name, probability, evidence_status, evidence_time, source, rank}]`
   with an explicit unknown entry. Evidence classes are `confirmed`, `projected`
   and `previous`.
3. **Labeled examples.** As games are observed, `AdvancedState` records whether
   the pregame top candidate was the actual leading passer and, if not, the
   depth rank of the actual starter (rank two, rank three or lower, unknown).
   Only games with a timestamped candidate list contribute; previous-passer
   assumptions and unknown starters are counted as excluded. The leading
   passer is a labeled proxy outcome only and never enters a feature.
4. **Estimator.** With at least 100 earlier examples in a class, the top
   candidate's probability is `(top_correct + 1) / (examples + 2)` and the
   remaining mass is split across the rank categories with add-one smoothing,
   mapped to the eligible candidates (ties split equally, unresolved mass to
   unknown) and renormalized. Below the threshold the top candidate has
   probability one and the alternatives are exposed with probability zero, so
   forecasts are identical to the deterministic assumption.
5. **Mixture.** `corrected_mixture` averages corrected win probabilities over
   home × away candidate combinations (independent assignments). Equal weights
   on forecasts of 0.2 and 0.8 give 0.5. An unknown candidate passes the empty
   identifier so no QB change applies. A what-if selection collapses that side.
6. **QB rushing aggregate.** `situations-v2` adds `qb_rushing[game][player] =
   {team, epa, carries}` for identified scrambles and designed runs by a player
   who also threw a pass in that game, excluding kneels, spikes and no-plays.
   Team arrays and the dropback classification are unchanged. Feature-time use
   requires the rusher to appear as a QB in weekly statistics.
7. **Rushing feature.** `Quarterback rushing change` is the assumed starter's
   decayed rushing EPA above the league QB mean (50-carry prior, per-game decay
   0.95, offseason 0.8) minus the team's recent QB rushing contribution under
   the same decay, divided by 0.15. An unchanged starter contributes exactly
   zero. The coefficient needs 100 earlier training games with rushing history
   for both teams and 20 nonzero observations. The feature is continuous
   (never exactly zero on real data), so the indicator-style zero-observation
   rule used for weather and availability does not apply to it.
8. **Outputs.** `next_context` and the capture service report `qb_candidates`,
   `starter_scenarios` (conditional forecasts with weights) and
   `assumption_source`. The scenario spread is not a confidence interval.

## Results: 2021–2025, 1,359 games

| Candidate | Brier | Log loss | Decisive accuracy | Brier delta vs Elo [95% interval] |
| --- | ---: | ---: | ---: | --- |
| Elo (chronologically selected) | 0.224121 | 0.640620 | 62.66% | — |
| Passing, deterministic (= `advanced-v5`) | 0.222061 | 0.636680 | 64.65% | −0.002060 [−0.004517, +0.000420] |
| Passing, mixture | 0.222071 | 0.636705 | 64.65% | −0.002049 [−0.004518, +0.000429] |
| Passing + rushing, deterministic | 0.222079 | 0.636718 | 64.50% | −0.002042 [−0.004523, +0.000409] |
| Passing + rushing, mixture | 0.222085 | 0.636735 | 64.50% | −0.002036 [−0.004527, +0.000413] |
| Inner-selected policy | 0.223013 | | | [−0.003415, +0.001092] |

The deterministic passing candidate reproduces the earlier advanced model
exactly, so the refactoring changed no historical forecast.

**Mixture.** Timestamped depth charts begin in August 2025, so labeled examples
exist only from 2025 and the estimator activates in week 4 of 2025. It changes
210 forecasts by 0.0010 on average and worsens their Brier from 0.23408 to
0.23415. By the end of 2025 the projected class holds 516 examples: the
rank-one depth entry started 465 times (90%), rank two 48 times and rank three
or lower 3 times. Because the projection is usually right and the QB-change
coefficient is small, there is little to hedge. Earlier seasons have no
candidate lists and are unaffected.

**Rushing.** The coefficient is enabled in every fit (745–1,599 eligible
training games) and is small and positive after 2021: −0.0025, 0.0278, 0.0489,
0.0405 and 0.0182 by outer season. Annual Brier moves by at most 0.0003 in
either direction; the pooled effect is +0.000018 against passing-only.
Pittsburgh improves slightly (0.26889 → 0.26853) and Minnesota worsens
(0.22136 → 0.22167); both are 85-game subsets.

Neither change beats its passing-only, deterministic comparison on both pooled
probability scores, so the shared gate retains the simpler model. The inner
selection picks the rushing variant in 2023 and 2024 and the passing variant
elsewhere, with no systematic gain. These seasons were explored during
development; intervals do not account for that selection.

## Not done here

Starter assignments are treated as independent across teams. The estimator
pools all teams and seasons within an evidence class; a per-team or
injury-aware model would need far more labeled seasons than one. The live
[paired forecast archive](forecast-collection.md) now records candidate lists
and conditional forecasts, which is the evidence a future promotion would use.
