# Model review and implementation guide

Reviewed on 2026-10-10 at commit
[`ffc9db23a509accfa72f34b59293dfcf90fe2498`](https://github.com/dwchal/nfl/tree/ffc9db23a509accfa72f34b59293dfcf90fe2498).
This document proposes future implementation work. The review commit changes
documentation only. Items 1–4 and 6 have been implemented as recorded below in the
[optimizer record](#item-1-implementation-record) and
[input-quality record](#item-2-implementation-record) and
[chronological evaluation record](#item-3-implementation-record) and
[prospective collection record](#item-4-implementation-record) and
[reduced-model record](#item-6-implementation-record); items 5 and 7–9 remain
proposed, along with item 6's deferred recency experiment.

The highest priorities are reliable optimization, honest missing-data handling,
and consistent chronological evaluation. After those foundations, test starter
uncertainty, smaller/shrunk feature models, player-weighted availability, and
better season simulations. Treat an accuracy improvement as a hypothesis until
it survives evaluation on the same games as the incumbent.

## What the repository already does

- `steelers/model.py` implements margin-aware Elo, confidence/rest calibration,
  and prior-season selection gates.
- `steelers/matchup.py` adds six regularized corrections using weekly efficiency,
  rest, and a quarterback passing-efficiency difference.
- `steelers/advanced.py` expands this to 34 features with situational plays,
  travel, issued weather, and timestamped availability. Its historical Elo
  offsets are selected separately for each season.
- `steelers/forecast.py` saves real pre-kickoff forecasts while the dashboard is
  used. `steelers/analysis.py` simulates remaining regular-season wins.
- Existing tests already check many own-game/future-game isolation properties,
  scenario isolation, offline fallback, and API behavior. Extend these tests.

The committed [advanced evaluation](advanced-evaluation-2026.json), under its
`matchup` field, reports these results. These are existing recorded results,
not a new backtest performed for this review:

| Comparison | Games | Elo Brier | Advanced Brier | Elo accuracy | Advanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2023–2025 combined | 816 | 0.222224 | 0.221240 | 63.56% | 66.38% |
| 2025 alone | 272 | 0.222360 | 0.226557 | 63.47% | 64.94% |

The combined 95% interval for the paired Brier difference is
`[-0.005136, +0.003099]`, including zero. The travel ablation explains most of
the recorded winner-pick improvement. Weather slightly worsens the probability
scores relative to travel alone; availability has zero historical support.
The [calibration audit](calibration-walk-forward.json) also has an uncertainty
interval including zero. This supports cautious experiments rather than adding
many more predictors at once.

## Findings and work order

Priorities: **P0** affects correctness or whether experiments can be trusted;
**P1** is a plausible improvement requiring measurement; **P2** is a larger
experiment after the earlier work.

| ID | Priority | Finding or opportunity | Main code | Depends on |
| --- | --- | --- | --- | --- |
| 1 | P0 | Fixed optimizer step can return a fit worse than zero weights | `matchup.fit`, `context_research.fit` | None |
| 2 | P0 | Missing values become zeros; advanced coverage omits weekly/QB checks | `features.number`, `evaluate_advanced`, `enabled_indices` | None |
| 3 | P0 | QB evaluation reuses target-season Elo settings; evaluation policies differ | `matchup.evaluate`, `evaluation.walk_forward` | 1, 2 |
| 4 | P1 | Forecast capture depends on dashboard use and compares varying lead times | `ForecastArchive`, `Handler.dashboard`, `AdvancedState.cutoff` | 2, 3 |
| 5 | P1 | One assumed QB; old confirmation wins over later unavailability | `Evidence.quarterback`, `FeatureState.qb_quality` | 1–4 |
| 6 | P1 | Overlapping features and fixed decay/penalties may overfit | `Situations`, `FeatureState`, `matchup.fit` | 1–3 |
| 7 | P2 | Availability counts treat players equally and need better evidence coverage | `AdvancedState.features`, `EvidenceStore.capture` | 2–4 |
| 8 | P2 | Weather training and prediction need consistent horizons and complete windows | `weather.summarize`, `historical_weather` | 2–4 |
| 9 | P2 | Simulated Elo updates omit margins and season ranges lack historical validation | `analysis.project_season` | 3, 4; 5/6 if enabled |

Implement one numbered item per change. Within an item, complete the listed
steps in order. Keep Python 3.10+ and the standard-library runtime. Preserve
existing JSON fields and append new metadata unless an explicit migration is
described. Each model/feature change needs a new version and fresh evaluation;
keep the existing JSON snapshots as historical records.

## Shared evaluation contract

Use this contract for items 3–9:

1. Before fitting, write an experiment configuration containing the candidate
   grid, feature versions, forecast cutoff policy, seed, evaluation seasons,
   and promotion rule. Include the configuration hash in the report.
2. For outer evaluation season `Y`, use only seasons `< Y` to select features,
   penalties, blend weights, and deployment decisions. Within `Y`, completed
   earlier games may update state once their data are eligible at the forecast
   cutoff. Freeze fitted coefficients throughout that season.
3. Use the same ordered game IDs and outcomes for candidate and incumbent.
   When optional inputs are missing, score the documented fallback on that
   game and report its use. Never improve a score by silently dropping hard
   games. Exclude incomplete outer seasons from complete-season headline
   results; label partial prospective results separately.
4. Primary metrics are Brier error and log loss. Also report decisive-game
   accuracy, ties, reliability bins, by-season results, PIT/MIN subsets, and
   available/missing-input subsets. Current probability metrics treat a tie
   as `0.5`; preserve that expected-result convention and label it explicitly.
5. Reuse `evaluation.paired_uncertainty` for paired season/week bootstrap
   intervals. Reuse `context_research.pick_comparison` for corrected picks and
   new errors, guarding an all-ties sample. Team subsets are diagnostics, not
   separate opportunities to tune settings.
6. Experimental eligibility requires lower pooled Brier **and** log loss
   against the incumbent. A winner-accuracy gain alone is insufficient.
   Report adverse seasons even when pooled scores improve. Retrospective
   research remains experimental: 2023–2025 has already been inspected.
7. For an eventual automatic promotion, freeze the candidate before collecting
   a prospective comparison and declare its end date/sample requirement in
   advance. A suggested conservative gate is one full eligible regular season,
   improvements in both scores, and a paired Brier interval wholly below zero.
   If the gate fails or evidence is insufficient, keep the current default.
   Treat this gate as a proposed policy, not evidence that any candidate passes.

## 1. Make the logistic optimizer stable

**Evidence.** [`matchup.fit`](../steelers/matchup.py) takes a fixed gradient
step of `0.5` for at most 300 iterations and does not inspect objective change
or report convergence. Thirty-four correlated features can make that step too
large. A synthetic probe using ten identical `[4.0] * 34` feature vectors,
six wins and four losses, returns approximately `0.999999998` with penalty
`0.1`. Its prediction log loss is approximately `8.0`; zero weights give
`0.693147`. This is a reproduced optimizer defect, not evidence that the stored
real-data fit encountered the same failure.

**Goal.** Every returned fit is finite, has a non-increasing penalized objective,
and communicates whether optimization converged.

1. Create `steelers/optimization.py`. Implement stable `softplus(z)` as
   `max(z, 0) + log1p(exp(-abs(z)))` and a sign-aware sigmoid. For row logits
   `z = offset + dot(weights, clipped_features)`, minimize
   `mean(softplus(z) - result*z) + penalty/2 * sum(weight**2)`.
   Clip features to `[-4, 4]` consistently with existing inference; do not clip
   logits when computing the objective or gradient.
2. Add `fit_logistic_offset(rows, indices, penalty, max_iter=2000, tol=1e-7)`
   returning weights plus `{converged, iterations, objective, gradient_norm}`.
   Validate finite inputs, positive penalty, common feature width, valid indices,
   and outcomes in `[0, 1]`. Empty training data must return zero weights and
   an explicit insufficient-data status.
3. Start weights at zero. Compute the gradient including `penalty * weight`.
   Try step `0.5`, repeatedly halve it until the objective is at most
   `old_objective - 1e-4 * step * sum(gradient**2)`. Stop with success when
   the maximum absolute gradient is below `tol`. If the step falls below
   `1e-12` or the iteration limit is reached, return the best finite weights
   and `converged=False`; never label that result an accepted fit.
4. Keep unselected coefficients exactly zero. Add an internal diagnostic fit
   entry point in `matchup.py`; retain the public `fit(...) -> tuple` wrapper
   for existing callers/tests. Evaluation paths must consume the diagnostic
   result, record it, and use Elo fallback when a fit has failed.
5. Adapt `context_research.fit` to the shared solver, preserving its current
   selected-coefficient return shape. Its implementation currently differs
   from the full-width matchup return shape. Change callers deliberately.
6. Add `tests/test_optimization.py`: reproduce the correlated-feature failure,
   verify objective no worse than zero initialization, verify the mixed-label
   example has a non-extreme probability, compare analytical gradients with
   central finite differences, and exercise empty/constant columns and invalid
   inputs. Also test coefficient placement for nonconsecutive indices.
7. Bump matchup and advanced versions when their fitting algorithm changes.
   Run the full suite. With saved research inputs, rerun the existing reports
   into new files and record how coefficients and scores changed.

**Done when:** the synthetic failure is fixed; failed fits cannot silently
become forecasts; tests pass; changes in real-data results are documented if
the data are available. Do not tune optimizer settings on held-out outcomes.

## 2. Distinguish unknown inputs from measured zero

**Evidence.** [`features.number`](../steelers/features.py) maps blank, `NA`,
`nan`, and `None` to zero. `parse_feature_csv` can therefore accept a missing
EPA as a measured zero. Its weekly keys use raw team abbreviations, unlike
`context_research.load_stats`, which normalizes aliases. Advanced evaluation
checks current-game PBP presence but does not enforce the weekly/QB coverage
check used by `matchup.evaluate`. Optional availability support counts any
snapshot, including an empty list whose report completeness is unknown.

**Goal.** Invalid or unavailable observations cannot masquerade as healthy
players, average efficiency, or sufficient training coverage.

1. Replace implicit missing-to-zero parsing with an explicit numeric parser
   accepting `allow_missing=False`. Normalize missing tokens, require finite
   values, and distinguish missing values from an actual numeric zero.
   For a positive play count, missing EPA invalidates that observation.
   A documented zero-opportunity statistic may have a neutral total only when
   its opportunity count is explicitly zero. Missing opportunity counts are
   never zero. Preserve `FeatureStore._file`'s valid-cache fallback on rejection.
2. Add a single `feature_key(season, week, kind, team)` helper using `team_key`.
   Use it in parsing, `FeatureState.observe`, replay coverage checks, and any
   new QB/snap loaders. Detect duplicates after normalization; preserve raw
   schedule IDs and display codes. Check duplicate player IDs within a team
   week as well as duplicate team weeks.
3. Track which observations actually entered each pregame state: previous
   eligible game/date, counts, and effective play samples for both teams/QBs.
   Attach this coverage metadata to replay rows. A target game's own PBP
   presence is a dataset-quality check, not proof of usable pregame history.
4. Share training-coverage logic across QB and advanced evaluation. Retain the
   existing minimum 200 regular games per prior season and 95% weekly team/QB
   coverage; require PBP coverage for advanced features as well. A complete
   lack of weekly data must return an unavailable model with a reason.
   Report separate pregame-state coverage; allow explicitly labeled neutral
   priors for season-opening/new-team history.
5. In `EvidenceStore.capture`, add per-team report status:
   `reported`, `explicitly_empty`, or `unknown`. File freshness alone and no
   matching rows cannot establish an empty injury report. Without provider
   evidence of completeness, assign `unknown`. Save source hash/timestamp.
6. In `enabled_indices`, count only eligible, complete game evidence toward
   the 100-game optional-family threshold. Also require variation: at least
   20 nonzero and 20 zero eligible observations for a proposed scalar effect;
   log these counts and freeze this proposed rule before evaluation. Unknown
   observations contribute a neutral effect with an explicit missing flag in
   coverage metadata, not a healthy observation. Disable unsupported
   coefficients explicitly. Learning coefficients for missing flags would be
   a separate experiment; this correctness fix need not add predictors.
7. Extend `tests/test_matchup.py` and `tests/test_advanced.py` with missing EPA,
   zero-opportunity, aliases, duplicate QBs, PBP-present/weekly-absent, and 100
   unknown injury snapshots. Verify invalid refresh preserves the prior cache
   and unsupported families stay disabled.

**Done when:** coverage describes information available before prediction,
unknown inputs produce a visible fallback, and blank data cannot enable a
feature family. Bump feature/cache schemas so older aggregates cannot silently
satisfy new coverage rules.

## 3. Evaluate every model with the same chronological policy

**Evidence.** [`matchup.evaluate`](../steelers/matchup.py) replays all seasons
with the `config` selected for the requested target season. For a 2026 report,
that config's deployment gate has already used 2023–2025 outcomes. Frozen QB
coefficients do not remove that upstream selection from the reported 2023–2025
comparison. [`evaluate_advanced`](../steelers/advanced.py) already replaces
historical offsets with per-year selections; `evaluation.walk_forward` currently
audits only Elo calibration. Existing notes acknowledge reused development data.

**Goal.** A shared experiment runner evaluates the complete deployed policy,
including selection, coverage fallbacks, and probability calibration.

1. Create `steelers/experiments.py` with a local-files-only CLI. Required inputs:
   schedule path, weekly-feature directory, optional advanced directory,
   first/last complete evaluation season, experiment JSON, output path.
   Proposed invocation **after implementing the command**:

   ```sh
   python3 -m steelers.experiments --data .cache/games.csv \
     --features .cache/features --advanced .cache/advanced \
     --start 2021 --end 2025 --config docs/experiments/chronological.json \
     --output docs/chronological-evaluation.json
   ```

2. Extract a helper that builds each historical season's Elo offsets using
   `select_model([g for g in games if g.season < year], year)`, then replays
   that season. Use those offsets in both QB and advanced training rows.
   Include the exact chosen Elo config beside each season's rows.
3. Define adapters for `elo`, `matchup`, and `advanced`: build eligible replay
   rows, fit on specified earlier years, and predict specified rows. Separate
   fitting from reporting; do not recursively run the whole report just to
   obtain weights. Preserve existing application entry points as wrappers.
4. For each outer year `Y`, reserve the last two earlier complete seasons as
   inner validation folds. For each candidate and inner year `V`, train on up
   to six complete years `< V` with at least three available. Select by pooled
   inner Brier, log loss as tiebreaker, then declared simpler-model order.
   Insufficient history means an explicit incumbent fallback.
5. Refit the selected configuration on up to six seasons `< Y`, freeze it,
   and predict `Y` chronologically. If evaluating an automatic gate, make that
   gate's decision solely from the inner comparisons. The outer year's result
   may be reported but cannot decide which forecast gets scored for that year.
6. Write per-game rows as well as summaries: game ID, season/week, home/away,
   cutoff policy, outcome, incumbent/candidate probabilities, selected config,
   coverage/fallback reason, and input hashes. Sort deterministically. Use
   `json.dumps(..., allow_nan=False)` and identify retrospective reconstructed
   inputs separately from actual archived inputs.
7. Add `tests/test_experiments.py`. Change outer-season scores/statistics and
   verify selected parameters and earlier forecasts remain identical. Change
   the outer season to ensure its metrics can change. Verify an inner fold
   never receives its own outcomes, every candidate scores identical game IDs,
   and unavailable optional features still produce fallback predictions.
8. Link new reports from the README with the complete selection protocol.
   Never overwrite the old snapshots or label already-explored years as an
   untouched test set.

**Done when:** one command reproduces comparable per-game predictions for all
model choices from fixed local inputs, and mutation tests cover the entire
selection path rather than only a fit receiving an already-frozen config.

## 4. Collect comparable forecasts at explicit prediction times

**Evidence.** [`Handler.dashboard`](../app.py) archives the selected team's
next game and selected model when a page is requested.
[`ForecastArchive.report`](../steelers/forecast.py) scores the latest record
before kickoff, pooling all lead times and implementations of a model choice.
[`AdvancedState.cutoff`](../steelers/advanced.py) uses `min(kickoff, now)`;
historical evaluation therefore approximates a kickoff forecast, although a
live user may predict days earlier. `matchup.replay` also flushes all pending
completed-game statistics into its returned state without a caller cutoff.

**Goal.** Produce repeatable, league-wide prospective comparisons and state
whose inputs were actually available at prediction time.

1. Extract a shared prediction service into `steelers/prediction.py`:
   `predict_game(game, model_artifact, inputs, as_of_utc)`. Return an unrounded
   home probability, input/feature digest, model artifact ID, evidence used,
   and fallback reasons. Call it from both the HTTP path and the collector.
   Keep rounding at the UI boundary; archive full precision.
2. Pass a required timezone-aware `as_of_utc` through state construction,
   evidence lookup, and live prediction. Replay may observe only completed
   games with eligible result/statistic timestamps before this cutoff. Apply
   the rule to the final pending-statistics flush as well as the main loop.
   For reconstructed history, keep the documented next-day publication
   approximation and label it. For live inputs, record when this application
   first retrieved each result/statistics snapshot; file mtime alone is not a
   durable provenance record.
3. Store immutable source snapshots addressed by SHA-256 with a manifest of
   retrieval times, provider timestamps when supplied, source URLs, schema
   versions, and file hashes. Preserve the manifest used by each forecast;
   future provider corrections must append a new snapshot. Keep bulk data
   under ignored `.cache/`, with small manifests/results suitable for Git.
4. Add a new archive table `forecast_runs` rather than rewriting the old table.
   Store `game_id`, `created_at`, `kickoff_at_capture`, home probability,
   `artifact_id`, input-manifest hash, and payload. Define `artifact_id` as a
   deterministic hash of model family/version, ordered feature names, weights,
   Elo config, training seasons, fitting options, and code revision. Store the
   artifact once in a separate table. Migrate old records only as legacy records
   with unknown artifact/horizon; retain the original table.
5. Implement `python3 -m steelers.capture --once --season 2026` to consider all
   upcoming league games within seven days and all configured available model
   choices. Capture inputs once, then produce paired predictions with the same
   clock and manifest. If an advanced model is unavailable, record its fallback
   explicitly. Accept an `--offline` mode. Fixture `--data` must never write to
   the real archive; require a separate output database for fixture runs.
6. Add reports for declared `24h` and `1h` forecast horizons. For horizon `H`,
   use the latest *paired collection run* at or before `kickoff - H`, no older
   than two hours before the 24h cutoff or 30 minutes before the 1h cutoff.
   Report missing captures rather than substituting a later forecast. Group
   by artifact ID and horizon; require an explicitly frozen policy ID to pool
   artifacts. Recheck revised kickoff times and deduplicate PIT/MIN views of
   the same game. Retain the legacy dashboard-use report under its old label.
7. Document an example user-invoked collector and an optional 15-minute local
   scheduling setup. The implementation itself need not install a background
   task. Sleep/offline periods must appear as capture gaps, not fabricated
   historical forecasts. Deduplicate retries by 15-minute UTC collection slot,
   manifest, artifact, and game. Preserve successful collections in later slots
   even when inputs are unchanged, so the 24h and 1h reports can both have a
   real collection record.
8. Add `tests/test_capture.py`: paired model snapshots, full-precision storage,
   deadline boundaries, revised kickoff, stale capture, missing artifact,
   duplicate requests, and fixture isolation. A cutoff mutation test must show
   that later results, statistics, lineups, and weather cannot change an earlier
   forecast. A migrated database must retain every old row.

**Done when:** both models are evaluated on identical games, collection times,
and horizons; every new forecast can identify its exact model and inputs.
Historical fixed files remain reconstructed history even when hashed.

## 5. Model quarterback uncertainty and rushing contribution

**Evidence.** [`Evidence.quarterback`](../steelers/evidence.py) returns a
confirmation before consulting availability, so a later Out report cannot
invalidate it. This was reproduced with a two-day-old confirmation and an
Out report one hour before kickoff. `last_qb` persists across offseasons and
is not checked against time-eligible roster/depth membership.
[`QBWeek`](../steelers/features.py) and `qb_quality` represent passing only.
Every matchup currently uses one QB assumption.

**Goal.** Make contradictory/stale starter evidence explicit and average over
supported starter possibilities instead of treating a projection as certainty.

1. First fix evidence ordering without adding a learned model. Load all
   eligible confirmation, depth, and availability records. A later definitive
   Out/Inactive record invalidates an earlier confirmation. Return the eligible
   depth alternative or an unknown starter with the conflict explained.
   A later reversal must have a new timestamp and explicit source; do not
   silently erase earlier evidence.
2. Add `quarterback_candidates(...)` returning
   `[{id, name, probability, evidence_status, evidence_time, source}]`.
   Keep `quarterback(...)` as a compatibility view of the top candidate.
   Use time-eligible depth/roster evidence to reject a departed previous passer.
   Historical membership cannot use today's season roster. If membership is
   unavailable, label the previous-passer fallback as unverified.
3. Begin with the existing deterministic assumption as the baseline. To learn
   candidate probabilities, create examples from timestamped depth/availability
   records at the declared horizon. Use the subsequently observed leading
   passer as a clearly labeled proxy outcome only, never as an input. Exclude
   games without an eligible candidate list and report that coverage.
4. For an initial small-data estimator, group by evidence class (confirmed,
   rank-one projection, previous passer). With at least 100 earlier labeled
   examples in a class, estimate top-candidate frequency as
   `(top_correct + 1) / (examples + 2)`. Allocate remaining mass to depth-rank
   categories (rank two, rank three or lower, and unknown) using their earlier
   observed frequencies with add-one smoothing. Map categories to the eligible
   candidates; split tied ranks equally and send unresolved mass to unknown.
   Normalize and exclude ruled-out players. Below the threshold, retain the
   deterministic baseline and expose scenarios; do not invent a 70/30 injury
   probability.
5. For home candidates `h` and away candidates `a`, compute
   `sum(q_home[h] * q_away[a] * p_win_given(h, a))`. Average probabilities,
   not logits or QB ratings. This first version assumes independent starter
   assignments; record that assumption. An unknown candidate contributes no
   unsupported QB change: pass the explicit empty-string QB identifier, since
   `None` currently means use the previous-passer default. A user what-if
   selection collapses that side to one candidate and remains excluded from
   the default archive.
6. Add an optional player rushing aggregate to `pbp.py` in a new cache version.
   Track identified QB scrambles and designed runs, excluding kneels, spikes,
   and no-plays. Use `rusher_player_id` joined to verified QB positions;
   ambiguous IDs/positions are missing. Store a separate `qb_rushing` mapping
   by game ID and player ID with EPA sum and carry count, rather than changing
   the meaning of the existing team arrays.
   Keep passing and rushing totals/counts separate, record denominators, and
   preserve the existing team dropback classification. Require historical
   coverage before enabling the extra QB feature.
7. In `FeatureState`, maintain decayed rushing EPA relative to an earlier
   league QB-rushing mean. Start with a declared 50-carry prior and the existing
   QB offseason decay `0.8`; these are experiment settings, not established
   optimum values. Add a rushing replacement feature centered against the
   team's recent QB rushing contribution so the model learns a *change* in QB
   value. Fit passing-only versus passing-plus-rushing with item 3's protocol.
8. Extend matchup output with candidate probabilities, conditional forecasts,
   and assumption source. Scenario spread is not a confidence interval.
   Test later-Out conflicts, trades/offseason membership, all candidates ruled
   out, unknown QBs, weights summing to one, deterministic equivalence, and
   mixture arithmetic (equal weights on `0.2` and `0.8` must produce `0.5`).

**Done when:** confirmed/projected/unknown assumptions remain distinguishable;
later evidence can correct old assumptions; probability mixtures and rushing
features pass chronological evaluation separately before being combined.

## 6. Prefer a smaller, better-regularized correction model

**Evidence.** Weekly passing/rushing efficiency overlaps with 16 situational
offense/defense features. Rest already appears in calibrated Elo and in the
matchup corrections; learning a residual rest effect can be valid but needs an
ablation. Team decay `0.9`, QB decay `0.95`, offseason factors, play-count
priors, and advanced L2 `0.1` are fixed. The 2025 advanced probability losses
and the existing ablations make shrinkage a useful experiment. The repository
has already tried Elo ensembles and larger grids; see
[calibration research](calibration-research.md) before repeating those trials.

**Goal.** Find whether fewer inputs, different recency, and a conservative Elo
blend generalize better than the fixed full advanced model.

1. Create a frozen `FeatureConfig` containing named decay and prior values.
   Thread it through `FeatureState`, `Situations`, replay, and inference.
   Defaults must reproduce current features before tuning. Cache keys and
   artifact IDs must include the configuration.
2. Replace hardcoded feature-index ranges in new experiment code with a
   versioned name-to-index registry. Add assertions that label count, vector
   width, and weight count agree. Preserve the current label order for old
   artifacts; never reinterpret an old weight vector under new labels.
3. Declare four groups: current full advanced; QB+weekly efficiency+travel;
   QB+situational efficiency+travel; and travel only. For the two reduced
   efficiency groups, omit the extra rest coefficient because calibrated Elo
   already supplies it. Keep full advanced as the unchanged comparison.
4. In item 3's inner folds, compare these four groups with L2 penalties
   `{0.03, 0.1, 0.3}`. Select by pooled Brier, then log loss, then fewer active
   coefficients. Refit the chosen setting before predicting the outer season.
   Include a zero-correction/Elo candidate so adding features is optional.
5. Run recency as a separate preregistered experiment: with the selected
   group/penalty chosen inside each outer fold, compare team/PBP per-game
   decay `{0.85, 0.90, 0.95}` and count-prior multiplier `{0.5, 1.0, 2.0}`
   using only inner folds. Hold QB/offseason decay fixed in this experiment.
   Replay each configuration from scratch; changing a divisor after features
   were generated does not test a different decay process.
6. Generate chronological inner validation predictions for the selected
   correction policy and Elo. Test the probability blend
   `p = (1 - alpha) * p_elo + alpha * p_correction` for
   `alpha in {0, 0.25, 0.5, 0.75, 1}`. Choose using only those earlier
   predictions and prefer smaller `alpha` on ties. Refit on earlier data,
   freeze `alpha`, then predict the outer season. Never fit a blend on
   in-sample fitted probabilities or the outer outcomes.
7. Report features selected, clipping rates, effective sample sizes,
   optimizer status, weights, and blend by year. Show the full fixed model,
   reduced model, recency variant, and blend as separate comparisons so the
   source of any change is visible. Count every tried configuration in the
   experiment manifest.
8. Add tests for default feature equivalence, changed decay affecting only
   subsequent games, registry mismatches, `alpha=0` exact Elo equivalence,
   `alpha=1` exact correction equivalence, and all selection staying earlier
   than the scored year. Archive the blend setting with every forecast.

**Done when:** the complete selection/blend policy has comparable outer-fold
results. Retain the simpler incumbent when improvements fail the shared gate.
Do not promote a particular group's lucky outer-season performance.

## 7. Weight player absences by expected participation

**Evidence.** [`AdvancedState.features`](../steelers/advanced.py) adds `1/3`
for each unavailable receiver, lineman, or defender, and `1/6` for a
questionable/doubtful player. A reserve counts as much as a starter. The
recorded advanced fit has zero eligible historical availability games, so
collecting suitable evidence is a prerequisite to learning these effects.

**Goal.** Learn whether the amount of missing expected participation predicts
performance beyond the team strength already represented in Elo/efficiency.

1. Add `steelers/availability.py` and a validated cached snap-count loader.
   [nflverse's snap-count loader](https://nflreadr.nflverse.com/reference/load_snap_counts.html)
   documents game-level PFR data and CSV support. Preserve its source player
   IDs and join to injury GSIS IDs through an explicit, versioned ID crosswalk;
   never fuzzy-match names. Report unmatched-player coverage.
2. Store per-player offensive/defensive snap shares, team, game ID, and first
   known availability time. Only earlier eligible games may establish expected
   participation. Missing snaps or a failed ID join are unknown, not zero use.
3. Estimate expected share as a weighted mean of up to four previous eligible
   team games, with newest-to-oldest weights `1, 0.8, 0.64, 0.512`. Start with
   a conservative shrinkage prior of two game-equivalents at the earlier
   same-position mean. Carry a player's identity across teams but require
   time-eligible evidence of membership/role; expose unknown roles explicitly.
4. For definitive Out/Inactive players, sum expected shares by receivers,
   offensive line, and defense. Exclude QBs from these groups so item 5 owns
   QB changes. For each group, subtract its team's earlier mean missing-share
   level: persistent absences already influence observed team performance.
   Use away-minus-home differences, matching the current availability sign.
   Record the reference history and sample sizes.
5. Keep Questionable and Doubtful as separate weighted-share predictors in
   the first version. Let regularized regression learn their association;
   do not assign an invented probability of being absent. Optionally learn
   participation probabilities later, using earlier timestamped reports and
   subsequent participation as labels only.
6. Start with L2 `0.3` for this experiment and retain item 2's coverage/support
   rules. Compare count-based, share-weighted, and disabled-availability
   versions inside the common evaluator. Unknown reports must trigger the
   same declared neutral/fallback behavior across candidates. Until sufficient
   evidence exists, ship collection and display with coefficients fixed at zero.
7. Explain contributions as missing expected participation, with evidence age
   and unmatched counts. Avoid causal language: correlated player absences
   do not establish individual player value.
8. Test starter-versus-reserve weights, absent ID mappings, midseason moves,
   duplicate injuries, no QB double counting, unknown-versus-empty reports,
   and future snap mutations. Require identical historical features before
   the modified game's data become eligible.

**Done when:** the data pipeline can distinguish a high-participation absence
from a reserve absence; the learned adjustment stays disabled without enough
timestamped support; any claimed gain survives the same outer comparisons.

## 8. Align weather features with the forecast horizon

**Evidence.** [`weather.summarize`](../steelers/weather.py) accepts any
nonempty set of hours in its three-hour window, so one available hour can be
treated as a full game window. Historical weather in
[`EvidenceStore.historical_weather`](../steelers/evidence.py) combines
`previous_day1` hourly values and conservatively timestamps the window near
kickoff minus 22 hours. That combined record cannot support a 24-hour-before-
kickoff prediction. The current 100-game gate checks counts, not diversity of
weather exposure or forecast horizon.

**Goal.** Evaluate weather only when historical and live forecasts use
comparable information, and avoid fitting effects to a few unusual games.

1. Make `summarize` require exactly the three distinct expected UTC hourly
   timestamps beginning at the floored kickoff hour. Reject missing,
   duplicated, nonfinite, or misaligned values and inconsistent array lengths.
   Preserve temperature, wind, gust, and precipitation units. Failed downloads
   must keep a previously valid cache, as they do now.
2. Store forecast model/source, issue or availability time, retrieval time,
   valid hours, and horizon in the evidence payload. Keep roof exposure from
   evidence available at the cutoff; unknown/retractable exposure remains
   unsupported unless an eligible roof decision was actually captured.
3. For retrospective fixed-horizon work, request a forecast whose full window
   was available before that horizon. Do not relabel `previous_day1`'s
   multi-hour window as available at kickoff minus 24 hours. A `previous_day2`
   window can serve as an explicitly older approximation; prefer a documented
   specific run when available. Open-Meteo distinguishes
   [fixed lead-time values from complete model runs](https://open-meteo.com/en/docs/previous-runs-api).
   If publication availability cannot be established, mark the record as
   reconstructed and keep it out of the prospective archive.
4. Fit the existing three weather/style interactions first, without adding
   more columns. Report eligible outdoor games by season, source/horizon, wind
   exposure, rain exposure, and cold exposure. Apply item 2's variation checks
   separately to each interaction, not just to the weather family total.
5. Compare weather-on versus weather-off on the identical complete set of
   outer games, plus the eligible outdoor subset. Indoor/unknown exposure must
   yield exactly the weather-off probability. Group-slice metrics should not
   become new hyperparameter selection opportunities.
6. Only after the existing interactions pass the gate, register a separate
   gust-versus-sustained-wind experiment using the already-collected gust
   value. Select the alternative within inner folds; the current poor weather
   ablation is not a reason to search many thresholds on 2023–2025.
7. Add weather tests for a one-hour response, missing middle hour, duplicate
   hour, DST conversion, a forecast arriving just after cutoff, a 24h cutoff
   rejecting the current previous-day window, and indoor equivalence.

**Done when:** incomplete forecasts cannot be treated as complete exposure,
and the weather comparison states exactly what forecast was knowable at the
time. Keep weather contextual if its probability scores do not improve.

## 9. Validate and improve the season-win distribution

**Evidence.** [`project_season`](../steelers/analysis.py) samples wins/losses
and calls `update_ratings` without a margin, even for a margin-aware fit.
`build_dashboard` computes corrections once for all remaining games; future
team efficiency, QB uncertainty, and schedule-derived away streak changes
are not simulated. Existing tests cover bounds and reproducibility, but the
reported middle 80% range has no historical coverage evaluation.

**Goal.** Measure the accuracy of the full season-win distribution and test
more realistic state evolution while preserving reproducibility.

1. Create `steelers/projection_evaluation.py`. For each historical season,
   build snapshots at fixed predeclared dates before weeks 1, 5, 9, and 13.
   Reconstruct only state available at that cutoff; hide all later scores and
   statistics during forecasting. Retain future scheduled fixtures, labeling
   final historical schedules as reconstructed unless an actual schedule
   snapshot exists. Fit the model using prior seasons under item 3.
2. Evaluate all 32 teams, with separate PIT/MIN diagnostics. Save expected
   wins, the full distribution, 10th/90th percentiles, and actual final wins.
   Report mean absolute error, ranked probability score
   `sum_k (forecast_CDF(k) - indicator(actual_wins <= k))**2`, interval
   coverage and width, and results by checkpoint/season. Resample whole
   seasons for uncertainty so four snapshots of one team are not treated as
   independent evidence. Warn about the small number of season blocks.
3. Preserve the current simulation as the baseline. Add an experimental
   margin sampler trained only on earlier games with chronological pregame
   probabilities. For decisive historical games, store absolute margin and
   the eventual winner's pregame probability. Use three fixed probability
   bands: `[0, .35)`, `[.35, .65)`, `[.65, 1]`. Each band needs 100 games;
   otherwise use the pooled earlier-season distribution. Exclude ties from
   margin magnitudes and report this support/fallback.
4. Simulate the winner using the existing game probability, then sample an
   absolute margin from the matching winner-probability band. Pass the signed
   home-minus-away margin into `update_ratings`. This preserves the initial
   winner probability while testing realistic magnitudes of subsequent Elo
   changes. Do not generate a separate score model whose implied winner
   probability contradicts the win model.
5. Sort remaining games internally by date/kickoff/ID. Clone mutable feature
   state per simulation. Advance deterministic schedule context such as away
   streaks for each simulated fixture. Continue to label future efficiency
   and injuries as held fixed until a separately validated evolution model
   exists. Do not feed actual future PBP into a simulated path.
6. Test latent team-strength uncertainty in a separate variant: draw one
   zero-mean normal Elo offset per team at the start of each path and carry it
   through that path. Choose `sigma` from `{0, 25, 50}` Elo points using only
   earlier inner projection checkpoints and ranked probability score, with
   smaller sigma breaking ties. This is an empirical sensitivity model,
   not a fitted Bayesian posterior. Include `sigma=0` as the incumbent.
7. If item 5's QB mixtures are available, sample a starter per fixture from
   its supported distribution and use that conditional probability. Never
   both sample a starter and apply the already-averaged QB adjustment.
   Future roster transitions require their own model; leave them fixed and
   labeled. Archive simulation settings, seed, and model/input artifact IDs.
8. Keep the existing tie convention for this first experiment: recorded ties
   are fixed and future ties are not simulated. State that limitation in the
   projection report. A later three-outcome extension must model
   `P(home win)`, `P(tie)`, `P(away win)` jointly and satisfy
   `expected_result = P(home win) + 0.5*P(tie)`; do not bolt a tie rate onto
   an unchanged binary probability.
9. Add tests for same-seed reproducibility, normalized mass, actual known
   record bounds, zero remaining games, margin-dependent rating updates,
   per-path state isolation, and future-result mutation. Validate that
   disabling every new option reproduces the old seeded distribution.
   Compare variants with the same seed and game order.

**Done when:** a reproducible historical report quantifies distribution
quality. Adopt a simulation change only if earlier selection and outer tests
support its ranked probability score and interval behavior; a wider interval
alone is not an improvement. Evaluate 1,000 paths during development and
10,000 for final reports, and record runtime.

## Commands and handoff checklist

Run these existing checks from the repository root after each implementation:

```sh
python3 -m unittest discover -s tests -v
node --check static/app.js
git diff --check
```

New tests should use small deterministic fixtures and temporary caches/SQLite
databases. HTTP tests need permission to bind localhost. They should not need
live downloads. Research commands require saved data; a fresh clone does not
contain `.cache/` and cannot reproduce the committed historical numbers alone.

When the necessary caches already exist, these **existing** commands provide
comparison reports without downloading new inputs:

```sh
python3 app.py --offline --backtest --season 2026 \
  > docs/review-candidate-elo-matchup.json
python3 app.py --offline --backtest --season 2026 --model advanced \
  > docs/review-candidate-advanced.json
python3 -m steelers.evaluation --data .cache/games.csv --start 2018 --end 2025 \
  --output docs/review-candidate-calibration-audit.json
```

These commands preserve the current evaluation semantics until item 3 is
implemented. Do not call their output the new common-policy audit prematurely.
Check report status/coverage before quoting any metric. Compare source hashes;
changing data at the same time as the model prevents a clean attribution.

Each implementation handoff should include:

- The single item implemented, changed functions, and version/schema changes.
- Tests run and any missing-data or convergence fallbacks exercised.
- The frozen experiment configuration, input manifest, and new report path.
- Paired metrics, annual/team diagnostics, and uncertainty; failed experiments
  belong in the report too.
- Whether the runtime remains experimental or meets the declared gate.

Suggested instruction to an implementation model:

> Implement item N in `docs/model-improvement-plan.md`. Complete its numbered
> steps and named tests, satisfying its dependencies first. Preserve the
> standard-library runtime and existing API fields. Use the shared evaluation
> contract. If required data are absent, complete the code and fixture tests,
> report exactly which dataset/time coverage is missing, and keep unsupported
> effects disabled. Do not fabricate a backtest, change promotion rules after
> seeing results, or implement unrelated items in the same commit.

## Review verification and limits

At the reviewed commit, **69 tests pass** with Python 3.14.7, and
`node --check static/app.js` passes. The initial sandbox run could not bind
localhost; rerunning with local socket permission passed the HTTP tests.
The review inspected Python model/data/evaluation modules, app integration,
tests, CI, and committed research reports. The retained R project is legacy
and is not the current prediction implementation.

Small local probes reproduced optimizer instability, blank-to-zero parsing,
and confirmation precedence over a newer Out report. They did not use external
data or modify production code. To reproduce the optimizer probe against the
reviewed commit:

```sh
python3 - <<'PY'
import math
from steelers.advanced import LABELS
from steelers.matchup import corrected, fit

rows = [
    {"offset": 0.0, "features": [4.0] * len(LABELS), "result": float(i < 6)}
    for i in range(10)
]
weights = fit(rows, tuple(range(len(LABELS))), penalty=0.1)
probability = corrected(0.5, rows[0]["features"], weights)
loss = -0.6 * math.log(probability) - 0.4 * math.log1p(-probability)
print({"probability": probability, "log_loss": loss,
       "zero_weight_log_loss": math.log(2)})
PY
```

No historical source caches were downloaded for this review, and the full
historical fits were not rerun. Reported historical metrics were read directly
from the committed JSON. No proposed feature, threshold, or simulation setting
in this guide has a demonstrated new predictive gain.

Provider contracts checked on the review date:

- [nflverse data schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html):
  statistics can receive later corrections; retrieval time and eligible
  observation time must be preserved separately.
- [Depth-chart dictionary](https://nflreadr.nflverse.com/articles/dictionary_depth_charts.html):
  the 2025+ feed supplies timestamped updates and player/depth identifiers.
  A depth rank is evidence for a projection, not proof of a confirmed starter.
- [Snap-count interface](https://nflreadr.nflverse.com/reference/load_snap_counts.html):
  game-level participation can support a player-share experiment.
- [Open-Meteo Previous Runs](https://open-meteo.com/en/docs/previous-runs-api):
  fixed lead-time hourly values differ from an individual model run. The
  conservative availability time of the whole game window matters.

## Item 1 implementation record

Implemented 2026-10-10 against the review commit, one item per the plan.

**Changed.** New `steelers/optimization.py` implements stable `softplus`/`sigmoid`,
the shared `objective`/`gradient`, and `fit_logistic_offset` (zero start,
backtracking from step 0.5 with the sufficient-decrease condition, `max_iter=2000`,
`tol=1e-7`, full-width weights with unselected coefficients exactly zero, explicit
`insufficient_data`/`max_iterations`/`step_too_small` statuses).
`matchup.fit_diagnostic` exposes it with the matchup width convention and keeps the
public `fit(...) -> tuple` wrapper; `matchup.evaluate` and `advanced.evaluate_advanced`
consume the diagnostic, record it in the report (`optimizer` fields, additive), and
fall back to zero weights (Elo) whenever a fit does not converge, which also blocks
promotion. `context_research.fit` preserves its selected-coefficient return shape over
the shared solver, and `experiment` records per-fold optimizer status with the same
zero-correction fallback. Versions bumped: `matchup-v1` → `matchup-v2`,
`advanced-v1` → `advanced-v2` (`static/app.js` version check updated to match).
New `tests/test_optimization.py` (12 tests) covers the correlated-feature failure,
non-increasing objective, non-extreme mixed-label probability, analytical-versus-
central finite-difference gradients, constant/empty columns, invalid inputs, and
nonconsecutive index placement. Full suite: 69 → 81 tests, all passing on Python
3.14.7; `node --check static/app.js` passes.

**Synthetic probe.** The review's ten-identical-`[4.0]*34`-row, six-win/four-loss,
penalty-0.1 probe now converges in 6 iterations to probability **0.5999** (log loss
0.673, objective 0.673 ≤ 0.693 at zero weights) instead of the old 0.999999998
(log loss ≈ 8.0).

**Real-data A/B.** The fresh clone had no `.cache/`, so nflverse inputs were
downloaded for the rerun (games.csv; team/player weekly stats 2016–2026;
play-by-play, depth charts, and previous-run weather history 2018–2026 via
`steelers.prepare`). The current provider files differ from the committed snapshots
(games.csv SHA now `a06ca5f6…` vs `d52363f2…`), so committed-file diffs mix data
drift with the optimizer change. The controlled comparison therefore ran the
review-commit code and the new code against the *identical* local cache in separate
worktrees:

| Model (same `.cache`) | Weights | Challenger Brier | Challenger log loss | Verdict | Optimizer |
| --- | --- | --- | --- | --- | --- |
| matchup (6 features) | max Δ 1.1e-4 | Δ 1.7e-7 | Δ 3.3e-7 | unchanged: not promoted, Elo default | converged, 460 iters |
| advanced, weather off | max Δ < 1e-6 | Δ 1.4e-9 | Δ 1.9e-10 | unchanged: qualifies | all fits converged (188–201 iters) |
| advanced, weather on (357 games) | max Δ 7.7e-7 | Δ 2.2e-9 | Δ 4.0e-9 | unchanged: qualifies | all fits converged |

The weather-on rerun reproduces the committed
[advanced evaluation](advanced-evaluation-2026.json) challenger Brier to seven
decimal places (0.2212402) and sample weights to five, despite the newer provider
file. Review of the context-research rerun found a full-width versus selected-
coefficient indexing error in the experiment adapter. The earlier attribution
of changed subgroup results to optimizer convergence was incorrect. After fixing
the adapter and regenerating the report on the same inputs, kickoff Brier is
0.223550, day-of-week Brier is 0.223603, and team-efficiency Brier is 0.223224.
The other four groups are unchanged; the combined model remains at 0.222625.
These are corrected research diagnostics, not a newly validated model gain.
The Elo calibration path
(`model.fit_calibration`) is unchanged, so the regenerated
[calibration audit](review-candidate-calibration-audit.json) reproduces the existing
walk-forward audit on the current schedule file.

**Regenerated reports (new files; committed snapshots untouched).**
[review-candidate-elo-matchup.json](review-candidate-elo-matchup.json),
[review-candidate-advanced.json](review-candidate-advanced.json),
[review-candidate-calibration-audit.json](review-candidate-calibration-audit.json),
[review-candidate-context.json](review-candidate-context.json).

**Limits.** No fit in any regenerated report failed to converge, so the Elo
fallback path was not exercised on real data. The follow-up regression test below
covers the advanced final-fit fallback through the dashboard. All
reported results remain retrospective. Optimizer settings were not tuned on
held-out outcomes. This record predates items 2–9, whose data-quality fixes
(unknown ≠ zero, explicit forecast cutoffs) may change these fits when they land.

**Follow-up review fixes.** The context experiment now extracts selected
coefficients from the solver's full-width vector before prediction and reporting.
An integration test checks every feature group, including nonconsecutive indices.
If the advanced model's final fit fails, it now returns an unavailable model,
zero corrections, no advanced replay probabilities, and an explicit Elo fallback
message. A regression test verifies that the dashboard disables advanced
predictions while preserving the historical comparison diagnostics.

## Item 2 implementation record

Implemented 2026-10-10 after the two item 1 review fixes. Model versions are now
`matchup-v3` and `advanced-v3`; the weekly parsing/digest schema is `weekly-v2`
and new availability payloads use `availability-v2`.

**Numeric data and identities.** Blank/NA numeric statistics now remain missing
or reject the file. Missing EPA is accepted as a neutral total only when the
associated opportunity count is explicitly zero. Missing, negative, fractional,
or nonfinite counts are rejected. All 22 locally cached weekly team/player
files (2016–2026) pass the stricter parser, including 283 legitimate missing
QB passing-EPA entries with zero dropbacks. A failed refresh preserves the last
valid snapshot. Weekly keys are normalized with a shared `feature_key`; duplicate
teams after alias normalization and duplicate player IDs within a team week are
rejected. Source metadata and the bundle digest include the parsing schema so
raw caches are revalidated under the new rules without requiring redownloads.

**Training versus pregame coverage.** New `steelers/coverage.py` gives QB and
advanced models the same per-season 200-game/95% weekly team/QB coverage gate.
Advanced additionally requires 95% PBP coverage. Dataset presence and usable
pregame history are reported separately. Replay rows record each team's previous
eligible observation, observed game count, decayed opportunity counts, assumed
QB history, and advanced PBP history. Metadata is copied before the game is
observed; same-day/own-game/future-data mutation tests check that isolation.
The matchup lab exposes missing-history priors and the date of earlier team data.

**Optional evidence.** Availability snapshots preserve a per-team status
(`reported`, `explicitly_empty`, or `unknown`), a completeness flag, and source
metadata. A fresh file or an empty player list cannot establish a healthy team.
The current row-based injury feed does not attest complete team reports; future
providers can supply explicit per-team descriptors through the internal
`injury_reports` mapping keyed by `feature_key`. Until then, listed absences can
still exclude a projected QB, but no non-QB availability effect is applied.
Legacy availability payloads remain in SQLite and cannot enable new coefficients.

Each optional coefficient needs 100 eligible complete game records, with at least
20 zero and 20 nonzero feature observations. Unknown records are excluded from
those counts and contribute a neutral effect with explicit missing metadata.
No learned missingness predictors were added. The thresholds were specified in
this guide before the rerun; the reports include the counts and settings.

**Evaluation.** New reports preserve the prior optimizer snapshots:
[input-quality-matchup-2026.json](input-quality-matchup-2026.json) and
[input-quality-advanced-2026.json](input-quality-advanced-2026.json).
Both use the same locally cached raw sources as the prior reports, with schedule
SHA-256 `a06ca5f608332ba4936d5f13f8868d92bf423b9d8b88b3c2dba293162575cfac`.
These snapshots use the earlier retrospective evaluation policies. Item 3 below
records the later common chronological evaluation.

| Candidate on 2023–2025 | Prior Brier | New Brier | Prior log loss | New log loss |
| --- | ---: | ---: | ---: | ---: |
| QB matchup | 0.222054864 | 0.222066885 | 0.635423576 | 0.635446683 |
| Advanced | 0.221240188 | 0.221243138 | 0.634400623 | 0.634410002 |

These small increases are reported, not treated as predictive gains. Canonical
weekly keys change earlier feature history for relocated teams. In the 2025
advanced fold, the new variation gate also disables rain; final 2026 training
has sufficient variation for all three weather coefficients. Earlier seasons
have 252/256 covered weekly games in 2020 and 268/272 in 2021, both above 95%.
Usable pregame QB history is separately missing in three 2025 advanced rows.
Final training has 357 eligible weather games and zero complete availability
games. All reported fits converge. The QB candidate still fails promotion; the
advanced model remains an explicit experimental option, with no default change.

**Validation.** 96 tests pass on Python 3.14.7, including the two review
regressions, stricter numeric/identity parsing, valid-cache preservation,
95% coverage boundaries, PBP-present/weekly-absent fallback, pregame sample
metadata, unknown/legacy reports, and per-feature support/variation gates.
JavaScript syntax and Git whitespace checks pass. No network downloads or
forecast-archive writes were needed for these reruns.

## Item 3 implementation record

`steelers/experiments.py` now implements the common local-only experiment runner.
Its [configuration](experiments/chronological.json) fixes seasons, candidate order,
penalties, cutoffs, coverage rules, inner folds, fit windows, gate and seed.
The [protocol and reproduction instructions](chronological-evaluation.md)
describe the complete selection path. The [report](chronological-evaluation.json)
includes comparable per-game predictions, fitted parameters, coverage/fallback
metadata, and hashes for raw inputs, logical evidence rows, configuration, and
Python implementation files. JSON rejects non-finite values and output is
deterministic for fixed snapshots and code.

Each historical Elo configuration is selected with strictly prior-year games.
QB and advanced application entry points now share that helper and the pure
correction fitting helper. Their version is `v4`; failed QB fits now explicitly
disable the correction model. Existing report fields and earlier snapshots are
preserved. The [updated QB application report](chronological-matchup-2026.json)
still fails its development gate; the new runner does not change deployment.

For each outer year the runner selects on the last two earlier complete seasons,
fits at most six earlier complete source-eligible seasons (minimum three), and
requires both pooled inner probability scores to improve. Candidate predictions
always include fallback games. Weekly history starts in 2016 and PBP in 2018;
source-eligible training years are recorded separately for each model. Feature
warmup is fixed per forecast year, so extending a run does not alter earlier
inputs. Optional support gates only inspect training observations.

On 1,359 games from 2021–2025, Brier/log loss are 0.224121/0.640620 for Elo,
0.222987/0.638071 for QB, 0.222061/0.636680 for advanced, and
0.223150/0.638683 for the inner-selected policy. All fits converge. Advanced
corrects 69 winner picks and introduces 42 errors; the selected policy has a net
15 additional correct picks. Both advanced and selected-policy probability-score
intervals include zero. Pittsburgh probability scores worsen, and both
correction models worsen 2025 probability scores. No complete availability
records exist. These are already-explored development years, not an untouched
holdout. Full year/team/input subsets and paired intervals are in the report.

New tests mutate outer scores and weekly/QB/PBP data through the entire selection
path, verify frozen settings and earlier predictions, check inner/outer fit-year
boundaries and candidate game alignment, and exercise missing/invalid sources,
failed optimizers, ties, partial seasons, stable range extension, deterministic
JSON, and read-only SQLite loading. Application regression tests verify that
target Elo settings cannot alter historical QB training and that failed fits
remain unavailable.

**Validation.** All 113 tests pass on Python 3.14.7, including the HTTP integration
tests. JavaScript syntax and Git whitespace checks pass. Two independent local
CLI runs produce byte-identical 1,359-game reports; their implementation hashes
match the committed Python sources. No downloads or forecast captures are needed.

## Item 4 implementation record

`python3 -m steelers.capture --once --season 2026` now collects all upcoming
league games within seven days, pairing Elo, QB matchup and advanced from one
input manifest and cutoff. `--offline` uses local snapshots. The shared
`steelers/prediction.py` service is also used by the HTTP path. It returns
unrounded probabilities, model/input/feature identities, evidence and explicit
fallbacks; the UI alone rounds estimates.

Required aware cutoffs bound live state and evidence. The pending-statistics
flush now respects the same next-calendar-day publication rule as replay, and
future completed rows cannot enter current ratings/features. Model versions are
`matchup-v5` and `advanced-v5`. Source bytes and normalized inputs are immutable
SHA-256 blobs with durable first-observed timestamps. Stable historical inputs
are split by season to avoid copying all history for each weather update.
The process's actual Python sources are pinned alongside inputs, with Git and
source-code hashes in each model artifact. Historical training data remain
explicitly reconstructed history.

New archive tables store atomic paired collections, full-precision forecasts,
registered artifacts/manifests, and optional frozen policies. All old dashboard
rows remain in their original table under a legacy label. Collection completion
is stamped after calculation; a slow run cannot pretend it finished before a
deadline. Retries deduplicate within a 15-minute UTC slot; later slots remain
available even when inputs do not change. Fixture schedules require a separate
database and are labeled reconstructed, with no live evidence capture.

`--report` grades the latest paired run at/before the 24h or 1h deadline,
within the declared two-hour or 30-minute tolerance. It rechecks revised
kickoffs, deduplicates team views, and reports gaps instead of substituting late
forecasts. Different artifacts stay separate unless an explicitly frozen policy
allowlist was registered before collection. Reports include paired probability
scores, team subsets, fallbacks, uncertainty and changed picks.

See [the collector guide](forecast-collection.md) for commands, optional
15-minute cron setup, frozen policies, local storage and fixture isolation.
No scheduler is installed automatically. An offline verification run saved
45 predictions for 15 upcoming games; all 65 already-completed games correctly
have missing fixed-horizon captures. Earlier snapshots and historical results
were preserved. Tests cover cutoff mutations, full precision, atomicity,
deadlines, revisions, stale/late gaps, missing artifacts/manifests, retries,
policy pooling, fixture isolation and old-row retention.

**Validation.** All 136 tests pass on Python 3.14.7, including the updated HTTP
integration tests. Git whitespace and JavaScript syntax checks pass. The
[collection verification](forecast-collection-verification.json) records a real
offline capture and its horizon coverage. All 45 probabilities and feature
digests reproduce exactly from the saved immutable inputs and artifacts.

The next planned priority is item 5: model quarterback uncertainty and rushing
contribution.

## Item 6 implementation record

`steelers/advanced.py` now carries a label-based feature registry and four
declared groups (`full`, `qb_weekly_travel`, `qb_situational_travel`,
`travel`); the reduced groups omit the rest coefficient that calibrated Elo
already supplies. `steelers/experiments.py` accepts protocol
`chronological-v2`: advanced candidates may name a group, support gates still
apply, and an optional blend `p = (1 − α)·p_elo + α·p_selected` is chosen on the
inner-fold predictions only, preferring smaller α and requiring both scores to
beat α = 0. α = 0 is exactly Elo and α = 1 exactly the correction. Fits report
active labels, training rows, clipping rates at the ±4 bound, active
coefficient counts and optimizer status. v1 configurations still validate and
reproduce the [v1 report](chronological-evaluation.json) game for game.

The [experiment](experiments/reduced.json) compares the four groups at L2
penalties 0.03, 0.1 and 0.3 with Elo on the same 1,359 games from 2021–2025;
[results](reduced-model.md) and the [full report](reduced-evaluation.json)
are checked in. No reduced model beats the full model at the same penalty; the
best pooled candidate is full at 0.03 (Brier 0.221880 versus 0.224121), the
22-feature situational group is within 0.00002 of it, and the penalty matters
more than the group. Heavier shrinkage (0.3) gives smaller gains whose
intervals exclude zero; lighter shrinkage gives larger but less stable gains.
The inner-selected policy scores 0.223320 and the blended policy 0.223603,
because every correction model worsens 2025 and the blend chooses α = 1 in all
selected years but 2024. The 2025 snapshots have full coverage; corrections
are simply more confident in a season that punished confidence. Nothing is
promoted; the application default and the experimental selector are unchanged.
Step 5 (recency and count-prior grids) is deferred to its own change.

New tests check registry resolution and width, that an unknown or mismatched
group fails, that the `full` group reproduces the v1 advanced candidate exactly,
that a declared group only removes coefficients, that blend endpoints are exact
and chosen on inner predictions alone, that outer outcomes cannot move the
blend or the selection, and the v2 configuration validation rules.

**Validation.** All 151 tests pass on Python 3.14.2. Two independent local CLI
runs produce byte-identical reports; the v1 configuration under the v2 runner
reproduces every checked-in 2021–2025 forecast. No downloads or forecast
captures are needed.

The next planned priority is item 5: model quarterback uncertainty and rushing
contribution.
