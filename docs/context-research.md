# Does more granular context improve winner predictions?

The first controlled comparison is promising but modest. Adding schedule
context and team efficiency raises accuracy from 62.66% to 63.76% on the same
2021–2025 games: 15 additional correct picks among 1,355 decisive games.
Four ties are excluded from pick accuracy and included in probability metrics.
These are different years from the earlier 2018–2025 calibration audit, so its
63.38% baseline should not be used for this comparison.

## Feature comparisons

| Model | Winner accuracy | Brier error | Log loss |
|---|---:|---:|---:|
| Current schedule-model policy | 62.66% | 0.224121 | 0.640620 |
| Refit home advantage only | 62.95% | 0.223569 | 0.639822 |
| Home advantage + stadium type | 63.32% | 0.223514 | 0.639722 |
| Home advantage + kickoff time | 63.17% | 0.223550 | 0.639833 |
| Home advantage + day of week | 62.80% | 0.223603 | 0.639904 |
| All schedule context | 62.95% | 0.223513 | 0.639773 |
| Recent team efficiency | 63.25% | 0.223224 | 0.638789 |
| Schedule context + team efficiency | **63.76%** | **0.222625** | **0.637951** |

Home advantage is already included in the current model. The extra home-field
control shows that some apparent context benefit comes from updating that
advantage. Day of week does not outperform that control in this sample. More
features do not automatically increase accuracy; all schedule features together
do not beat stadium type alone on winner picks.

Team efficiency includes recent net passing yards per dropback, passing first
down rate, sack rate, rushing first down rate, turnovers per play, and EPA per
play. Offense and defense are tracked separately, adjusted using opponents'
earlier strength, decayed by 0.9 each game, and shrunk toward league average.
Offseason accumulated totals/counts decay by 0.65. Fixed play-count priors are
100 for pass/rush signals and 200 for turnover/EPA signals. Input differences
use fixed scales and are clipped to ±4 before fitting. None of these settings
was searched against the evaluated seasons.

The combined model makes seven more correct picks in 2023, six in 2024, and two
in 2025, with no net change in 2021 or 2022. Pittsburgh accuracy is unchanged;
Minnesota rises from 60.00% to 62.35%. Both teams' probability errors worsen
slightly despite the league-wide improvement. The bootstrap interval for the
league-wide Brier change includes zero. Full pick-change counts, accuracy
intervals, yearly coefficients, input hashes, and team metrics are in
[the evaluation snapshot](context-evaluation-2021-2025.json).

This remains an **experimental comparison**, without a change to the dashboard's
default model. Seven related feature groups were examined, and historical data
were already used in previous research. Neither the model comparison nor its
bootstrap intervals establish an independently confirmed improvement.

## Timing and evaluation controls

- Team statistics from 2016 warm up the feature state; regression training begins
  in 2017. Each evaluated season from 2021–2025 fits on up to six preceding
  seasons, then freezes its coefficients for that season.
- Every historical Elo probability was selected using seasons preceding its
  own season. The experiment adds regularized logistic corrections, with a
  fixed L2 penalty of 0.1 and no tuning against the evaluation years.
- Each game's features use only earlier days' statistics. Results and statistics
  of that game, other games that day, and later games cannot predict it.
- Stadium type comes from a previous game at that venue. Actual open/closed
  retractable-roof decisions are combined into one building type; unknown venues
  are unknown, rather than filled from a later game.
- Early kickoff means at or before 13:59 Eastern; evening means at or after
  19:00 Eastern. Kickoff is not converted to the team's home time zone, so this
  does not yet test travel/body-clock effects. Thursday, Monday, and Saturday
  each have an indicator. Neutral games have no home-context effects.
- No observed game weather, confirmed starting lineup, postgame schedule QB
  fields, or betting lines enter these models. Provider corrections and final
  schedule revisions can still make a retrospective dataset differ from what
  was available earlier.

## Where to investigate next

These are priorities for further experiments, not claims of validated gains:

1. **Confirmed starting quarterbacks and availability.** The current QB model
   assumes the previous game's leading passer, which misses actual lineup
   changes. Historical lineup information needs pre-kickoff timestamps.
2. **Situation-specific play-by-play.** Separate passing/rushing success and
   efficiency, account for down/distance, remove kneel-downs, and test excluding
   late low-competition situations. Weekly aggregates cannot express all of this.
3. **Forecast weather interacting with team style.** Test issued wind,
   precipitation, and temperature forecasts together with passing reliance and
   roof type. Historical observed weather is not a substitute for an issued
   forecast when evaluating a model used days before kickoff.
4. **Travel and body-clock time.** Combine stadium locations and team time zones
   with kickoff, short rest, and consecutive road trips. Eastern kickoff alone
   does not capture this hypothesis.

Available fields are documented in the official nflverse
[schedule dictionary](https://nflreadr.nflverse.com/articles/dictionary_schedules.html),
[team-stat dictionary](https://nflreadr.nflverse.com/articles/dictionary_team_stats.html),
and [play-by-play loader](https://nflreadr.nflverse.com/reference/load_pbp.html).
In particular, the weekly rushing totals include scrambles and kneel-downs, and
passing EPA uses QB EPA; those definitions matter when interpreting aggregates.

## Reproduce

```sh
python3 -m steelers.context_research --data .cache/games.csv --features .cache/features --output docs/context-evaluation-2021-2025.json
python3 -m unittest discover -s tests -v
```

The command requires complete 2016–2025 schedules and weekly team files, uses
only local data and the Python standard library, and makes no changes to saved
forecasts or dashboard selection. Seven regression tests cover data validation,
team aliases, same-day/future isolation, roof history, kickoff buckets,
regularization, and paired winner-pick comparisons.
