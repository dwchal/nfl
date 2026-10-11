# Advanced matchup model

Choose **Advanced matchup · experimental** in the dashboard to use the four additions below. Processing uses Python's standard library; downloaded data and pregame evidence remain local.

For a common comparison of Elo, QB matchup and advanced, see the
[nested chronological audit](chronological-evaluation.md). It uses identical
game sets and prior-season candidate selection. Earlier snapshots below retain
their original evaluation protocols.

## Prepare and run

For 2026, download eight warmup/history seasons plus the current season:

```sh
python3 app.py --check
python3 -m steelers.prepare --start 2018 --end 2026 --weather-history
python3 app.py
```

Preparation can take several minutes and reports counts/warnings for each source. The dashboard loads compact aggregates afterward. **Refresh data** updates the prepared current season and depth charts; rerun preparation with `--refresh` to update older aggregates. `--offline` uses saved data. Each prior training season needs at least 200 regular-season games and 95% weekly team/QB and play-by-play coverage. Without sufficient history, or if the final fit fails to converge, the advanced selector falls back to Elo and explains the reason.

## 1. Starting quarterbacks and availability

The model distinguishes **User-confirmed**, **Projected**, and **Previous passer**. Timestamped depth charts provide projections, not confirmations. An Out/Inactive QB in a saved pregame injury report is excluded from the projection. Missing/stale evidence falls back to the previous passer, or unknown when that passer is ruled out. Lightly used quarterbacks shrink toward league-average passing efficiency.

To record a confirmed starter, select the QB in the matchup controls, enter the source, and use the confirmation button. The app records the submission time and source in an append-only local archive. It rejects completed games and post-kickoff submissions. Merely selecting a what-if QB does not confirm a starter or alter archived forecasts. This is a user-supplied confirmation, not an automated verification of the source.

Availability features count unavailable skill-position players, offensive linemen, defenders, and questionable/doubtful players. Coefficients stay zero until at least 100 games in earlier training seasons have timestamped, complete availability reports for both teams. Each coefficient also needs 20 zero and 20 nonzero eligible observations. The current historical training set has **zero** complete games: non-QB injury effects are implemented but not yet fitted. A missing report or empty player list does not establish that a team is healthy. The current row-based feed does not certify completeness, so its records remain partial/unknown. Historical report files are never backdated into pregame evidence. QB exclusion can still operate on a current report.

Opening/refreshing the advanced dashboard captures available injury reports for upcoming league games within seven days and weather for the selected next game. The [paired collector](forecast-collection.md) can capture all upcoming league games independently, with optional user-managed scheduling; the application does not install a background task. Depth and injury evidence expire after seven days. nflverse's [data schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html) and [depth-chart dictionary](https://nflreadr.nflverse.com/articles/dictionary_depth_charts.html) describe the timestamped depth-chart feed used from 2025 onward.

## 2. Situational play-by-play efficiency

The model adds opponent-adjusted offensive and defensive pass/rush EPA and success rates, sack rate, early-down EPA, third/fourth-and-long success, and passing reliance. Sacks and scrambles count as dropbacks. Kneels, spikes, no-plays, preseason plays, and plays starting in the final 15 minutes with a score margin above 16 are excluded. The blowout filter uses the pre-play score.

Recent games receive more weight, sparse samples shrink toward average, and offseason history decays. A game's own statistics and same-day statistics cannot enter its prediction. This one-day availability approximation does not reproduce the exact publication time of every play; later provider corrections remain a retrospective-data limitation.

## 3. Forecast weather and playing style

Wind, precipitation, and cold interact with the difference in the teams' earlier passing reliance. The model requires an issued forecast within 48 hours of kickoff and a venue previously known to be outdoors. Enclosed, retractable, and unknown roof exposure are excluded from these coefficients. The interface may display a longer-range forecast even when it is not eligible for the model.

Historical weather uses Open-Meteo's [Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api), specifically forecasts from 24 hours before each valid hour, rather than observed game weather. The three-hour window receives a conservative availability time based on its last constituent forecast. Provider retrieval time is retained separately. Current forecasts are saved when retrieved.

Weather coefficients require 100 earlier training games with eligible forecasts, plus at least 20 zero and 20 nonzero observations for each interaction. Preparation saved 423 historical forecasts; 357 qualify for the final 2026 training window. Availability, roof, and season filters explain the difference. Weather has **not** demonstrated an additional accuracy gain in this evaluation.

## 4. Travel and body-clock kickoff

Features cover distance from each team's home base, kickoff before noon or after 9 p.m. in that team's time zone, short-rest travel, and consecutive away games. Time conversion respects daylight saving and historical franchise relocations. Neutral games calculate travel for both teams. Missing venue coordinates do not produce invented distances.

These are schedule-based approximations. Actual flight itineraries, arrival dates, and acclimatization are unknown. Home advantage and rest remain in the underlying model; travel adds information about the particular trip.

## Historical evaluation

The original snapshot below predates stricter input validation. The current
[input-quality evaluation](input-quality-advanced-2026.json) uses `advanced-v3`
with weekly/QB coverage checks, explicit pregame history, and per-feature
variation requirements. Its 816-game Brier is 0.221243 and log loss is 0.634410;
the default remains unchanged. See the
[implementation record](model-improvement-plan.md#item-2-implementation-record)
for the comparison and data limitations.

The [complete 2026 snapshot](advanced-evaluation-2026.json) compares identical games using annual chronological fitting. Each evaluated year's Elo settings and advanced coefficients use earlier seasons only. Corrections train on 2020–2022 for 2023, expand through 2023 for 2024, and expand through 2024 for 2025. Current 2026 coefficients train on 2020–2025. The fixed L2 penalty is 0.1. Each prior season must contain at least 200 regular-season games and 95% play-by-play coverage.

| Metric, 2023–2025 | Chronological Elo | Advanced |
| --- | ---: | ---: |
| Winner accuracy | 63.56% | **66.38%** |
| Correct decisive-game picks | 518 / 815 | **541 / 815** |
| Brier error (lower is better) | 0.222224 | **0.221240** |
| Log loss (lower is better) | 0.635855 | **0.634401** |

There are 816 games, including one tie excluded from winner accuracy. The advanced model gets **23 additional winners right**. This baseline differs from the earlier fixed-2026-settings evaluation because Elo itself is reselected before each evaluated year.

| Added features, independently fitted | Accuracy | Brier error |
| --- | ---: | ---: |
| QB + weekly efficiency | 63.56% | 0.222102 |
| + situational play-by-play | 63.44% | 0.222019 |
| + travel and body-clock | **66.38%** | **0.221240** |
| + supported weather/availability | **66.38%** | 0.221240 |

These diagnostics do not select the final model. Travel accounts for most of the observed accuracy gain. Weather changes Brier only slightly, in the wrong direction versus the travel model; non-QB availability coefficients remain zero.

The result is promising but uncertain. The week-block bootstrap 95% interval for the Brier difference is **−0.00514 to +0.00310**, including zero. In 2025 alone, winner accuracy improves but Brier worsens from 0.22236 to 0.22656. Pittsburgh accuracy improves from 50.98% to 56.86%; Minnesota from 56.86% to 58.82%, with only 51 games per team. These seasons were already explored during development, and the intervals do not account for that exploration. The model remains an explicit experimental choice, pending forward evidence.

The forecast archive saves genuine pre-kickoff default assumptions and their evidence. Scenarios are excluded. Season simulations hold current advanced corrections fixed while simulated outcomes update Elo; they do not simulate changing injuries, weather, itineraries, or future efficiency.

Reproduce the snapshot and checks:

```sh
python3 app.py --offline --backtest --season 2026 --model advanced > docs/advanced-evaluation-2026.json
python3 app.py --offline --check --season 2026 --model advanced --team PIT
python3 app.py --offline --check --season 2026 --model advanced --team MIN
python3 -m unittest discover -s tests -v
```

The advanced report is under `matchup` in the JSON; outer fields describe the underlying Elo selection. The snapshot records schedule, weekly-statistics and play-by-play provenance plus an evidence digest. Reproduction requires the saved inputs; provider revisions and newly captured evidence can change results.
