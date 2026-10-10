# Steelers & Vikings Season Dashboard

A dashboard for the **Pittsburgh Steelers and Minnesota Vikings** that runs locally on your Mac with **Python 3.10 or newer**. No R, virtual environment, package installation, API key, or account is required. The browser interface is served by Python’s standard library.

## Run on this Mac

Double-click **Start Steelers.command** in Finder. It opens the dashboard in your default browser. Keep the Terminal window open while using the app; press **Control-C** there to stop it.

Or use Terminal:

```sh
cd ~/Developer/nfl
python3 app.py
```

The usual address is **http://127.0.0.1:8765**. If that port is occupied, the app chooses another port and prints/opens its address. It listens only on this computer. You need an internet connection on the first launch to download data. Later launches can use saved data when offline.

On a fresh clone, if Finder does not allow launching the command file, run `python3 app.py` from the repository folder instead. If Python is missing, install it from [python.org](https://www.python.org/downloads/macos/).

## What it shows

- A team selector for Steelers or Vikings, with team colors and your choice remembered between visits. Pittsburgh is the default on a first visit.
- Selected team’s record, winning percentage, point differential, recent form, and Elo rank.
- Next unplayed matchup, venue, date, Eastern kickoff time, and an estimated win probability.
- Regular-season win projection from 10,000 simulations, with the middle 80% range and outcome distribution.
- Cumulative point differential, AFC North or NFC North comparison, and a filterable team schedule.
- Searchable NFL rankings with sortable columns and your selected team highlighted.
- Season selection, optional postseason results, refresh controls, download timestamp, and offline status.
- A backtested forecast model, original-model comparison, league-wide and team-specific evaluation, and replayed pre-game estimates for completed games.

The default season follows the NFL calendar (January–February belong to the preceding year), with a fallback to the latest season available in the feed. Seasons are no longer hardcoded to 2024. Preseason games are excluded. The regular-season view excludes playoff results from records and current-season ratings; “Regular + playoffs” includes them. The outlook always predicts the **regular-season record**. A finished season shows its final record rather than inventing another matchup.

## Data and model

Schedules and scores come directly from [nflverse’s games.csv](https://github.com/nflverse/nfldata/blob/master/data/games.csv), the schedule feed used by the original R project. Downloads are validated and cached in `.cache/games.csv`. Data is refreshed when requested after an hour, or immediately using **Refresh data**. An unsuccessful download preserves the last valid snapshot and shows a warning. Provider update delays can affect scores and kickoff times; this is not a live play-by-play feed. Data remains subject to the provider’s terms.

The **backtested default** adds scoring-margin weighting to Elo, followed by a gated confidence and rest calibration. Large wins have more influence, with logarithmic scaling and a correction for strong favorites beating weaker teams. The weighting is inspired by [FiveThirtyEight’s published NFL Elo approach](https://github.com/fivethirtyeight/nfl-elo-game/blob/master/forecast.py); this is our own implementation and evaluation, not FiveThirtyEight’s complete model.

For each season being viewed, the app uses the six most recent **completed prior seasons**. Three older seasons select settings; three later seasons compare the chosen candidate with the original model. Neither set includes the selected season. A fixed grid of 27 score-aware models tests update factors of 10/20/30, home advantages of 35/55/75, and offseason carryover of 0.5/two-thirds/0.8. Original Elo is also a tuning candidate. Selection minimizes Brier error, with log loss breaking ties. The chosen challenger becomes the default only if it improves **both** Brier error and log loss in the later comparison. That comparison is used as a deployment gate, so it is not a completely untouched final audit. Each three-season window must contain at least 500 regular-season games; otherwise the app uses original Elo.

A second step learns two probability corrections from the same older three seasons: a confidence multiplier on Elo log odds and a rest-difference coefficient. L2 regularization (0.03) shrinks them toward unchanged Elo. Confidence is bounded to 0.5–1.5; the rest coefficient to −1–1. Rest is measured in weeks, capped at 14 days per team; missing rest contributes zero. This layer leaves team-strength updates unchanged and applies consistently to replays, upcoming games, QB-model offsets, and simulations. It becomes the default only if **both** Brier error and log loss improve against the previous selected default on the later comparison seasons. Coefficients remain frozen during that comparison and the selected season. Original Elo remains available.

All teams use the same selected settings. Ratings start at 1,500 and warm up using up to three preceding seasons. During evaluation, a forecast is calculated **before** that game’s result updates the ratings. Playoff games can update team strength; evaluation metrics cover regular-season games. Ties count as half a win for Elo, winning percentage, and probability-error metrics, and are excluded from winner-pick accuracy. Historical settings never use current-season or future-season scores. Completed-game probabilities in the schedule are retrospective pre-game replays, not archived published forecasts.

The **Original Elo** option preserves the previous model: update factor 20, 55-point home advantage, two-thirds offseason carryover, and no margin weighting. Neutral venues receive zero home advantage in both models. The forecast model selector changes upcoming probabilities, ratings, and season simulations; your choice is remembered between visits.

The [original 2026 evaluation snapshot](docs/model-evaluation-2026.json) predates confidence/rest calibration. It tunes on 2020–2022 and evaluates 816 games from 2023–2025. It selected margin-aware Elo with update factor 20, home advantage 35, and two-thirds carryover. Brier error dropped from **0.2280 to 0.2219 (2.64%)**, log loss from **0.6479 to 0.6356**, and decisive-game pick accuracy rose from **62.1% to 64.4%**. Vikings-only Brier error improved (0.2359 → 0.2262); Steelers-only error worsened (0.2546 → 0.2666), with 51 games per team. The interface displays these team-specific results rather than assuming a league-wide improvement applies equally to Pittsburgh and Minnesota. The snapshot includes the source-file SHA-256; future provider corrections may change reproduced numbers slightly.

The [calibrated 2026 snapshot](docs/calibrated-model-evaluation-2026.json) uses the same data and selects a confidence multiplier of **0.9133** and rest coefficient of **0.1081**. Against the previous default on 2023–2025, Brier improves **0.22194 → 0.22181**, log loss **0.63560 → 0.63493**, and pick accuracy **64.42% → 64.54%**. Pittsburgh Brier improves (0.26662 → 0.26443), while Minnesota worsens (0.22615 → 0.22773). These are small development gains, not evidence of a large increase in predictive power.

A separate [walk-forward audit](docs/calibration-walk-forward.json) reruns the complete selection and deployment policy before each season from 2018–2025. Across **2,127 games**, Brier improves **0.22212 → 0.22193** and log loss **0.63738 → 0.63677**; pick accuracy stays **63.38%**. Both team subsets have slightly lower Brier error, though Pittsburgh pick accuracy falls. A paired week-block bootstrap gives a 95% Brier-difference interval of **−0.00045 to +0.00005**, which includes zero. This remains retrospective research, not a live audit, and those intervals do not account for choosing a method after exploring alternatives. [Research notes](docs/calibration-research.md) record those alternatives and reproduction commands.

For the regular-season outlook, each simulation plays all remaining league games and updates ratings along the simulated path. Future margins are unknown, so simulated updates use wins and losses alone. Results use a fixed random seed for reproducibility. Future ties are not simulated. The model does **not** predict playoff qualification or implement official standings tiebreakers. The experimental matchup model adds player passing history, team efficiency, and rest. In the original QB-aware model, injury reports and weather are displayed and archived as context. The advanced model adds timestamped availability and weather features with training-coverage requirements. Betting markets are not used. Historical evaluation does not guarantee future accuracy; the displayed season range describes simulated outcomes, not validated statistical confidence. Historical summaries and remaining-game estimates use the final available data for the selected season.

## Advanced matchup model

Current models distinguish missing statistics from measured zero, normalize
historical team aliases, and report the earlier history available to each
forecast. Advanced training requires weekly team/QB coverage as well as
play-by-play coverage. Optional weather and injury effects require sufficient
complete evidence and variation; incomplete injury reports apply no non-QB
adjustment. See the [input-quality implementation and evaluation](docs/model-improvement-plan.md#item-2-implementation-record).

The **Advanced matchup · experimental** selector implements timestamped QB projections/user confirmations and availability capture, situational play-by-play efficiency, forecast-weather interactions with passing style, and travel/body-clock kickoff effects. Prepare its data once:

```sh
python3 -m steelers.prepare --start 2018 --end 2026 --weather-history
```

On 816 games from 2023–2025, annual chronological fitting improves winner accuracy **63.56% → 66.38%**, or **23 additional correct picks**, with slightly lower Brier error and log loss. Most of the accuracy gain comes from travel/body-clock features. Statistical uncertainty still includes no improvement, and 2025 probability scores worsen; the model remains experimental. Weather adds no demonstrated accuracy gain yet, and non-QB injury coefficients wait for enough timestamped historical evidence. See [setup, methods, limitations, and results](docs/advanced-model.md).

## Granular context research

An offline [context experiment](docs/context-research.md) tests stadium type, kickoff time, day of week, and recent opponent-adjusted team efficiency. On the same 2021–2025 games, the combined candidate improves winner accuracy from **62.66% to 63.76%** (15 additional correct picks), with lower league-wide Brier error and log loss. Team-specific results are mixed, and this exploratory candidate is **not deployed**. The research command and complete results are linked in the notes.

## QB-aware matchup lab

The **Matchup lab** adds quarterback what-if controls for both teams, passing/rushing offense and defense contributions, game-week injury reports, kickoff weather, and a local archive of actual pre-kickoff forecasts. The **QB-aware matchup · experimental** selector applies the new model to upcoming games, completed-game replays, and the season outlook. **Backtested default** keeps the existing Elo model unless the new candidate improves both Brier error and log loss.

Weekly team/player statistics, rosters, and injuries come from [nflverse releases](https://github.com/nflverse/nflverse-data/releases). Validated downloads are saved in `.cache/features/`; failed downloads preserve valid snapshots. The first launch downloads nine seasons of compact weekly CSV data and may take longer. Extra data are optional: missing history falls back to Elo. `--offline` uses saved statistics and weather. `--data` uses only the supplied schedule and disables external features and archiving, so fixture data cannot be mistaken for genuine forecasts.

The challenger uses regularized logistic regression with Elo as its probability offset. It learns six corrections: passing offense/defense, rushing offense/defense, rest differential, and expected QB passing-efficiency change relative to the team's offense. Team efficiencies are EPA (expected points added) per dropback/carry, adjusted online for the opponent's earlier offensive or defensive strength. Recent games receive more weight; small samples and offseason history shrink toward league average. QB passing statistics follow the individual across teams. This version does not include rushing QB value, protection/receiver-specific injury adjustments, travel, coaching, or drive-based score simulations.

For a selected season, coefficients train on the first three of the six preceding seasons and stay frozen throughout the later comparison and selected season. A fixed full-feature challenger uses L2 penalty 0.03. Smaller rest-only and efficiency-plus-rest models train on the first two older seasons and validate on the third as feature diagnostics; those checks do not choose the final challenger. Each historical forecast sees only earlier games' statistics, with a one-day publication-lag approximation; same-day statistics are excluded. Schedule QB starter columns are never used, because they can be populated after the game. Historical weekly statistics can contain later provider corrections, so this remains a retrospective development comparison, not an untouched live audit. At least 200 games per prior season and 95% team/QB statistics coverage are required. Coefficients are shared across all teams; Steelers/Vikings results are also shown separately.

The [original 2026 matchup evaluation](docs/matchup-evaluation-2026.json), before confidence/rest calibration, compares 816 regular-season games from 2023–2025 after training corrections on 2020–2022. Winner-pick accuracy rose **64.4% → 65.3%**, but Brier error worsened **0.2219 → 0.2223**, and log loss worsened **0.6356 → 0.6363**. Both team subsets also had worse Brier error (Pittsburgh 0.2666 → 0.2747; Minnesota 0.2262 → 0.2295). Consequently the challenger is **experimental and not the default**. The later calibrated snapshot also failed the two-score gate. Historical QB training/comparison offsets now use each year's strictly prior Elo settings; the [updated application report](docs/chronological-matchup-2026.json) still fails that gate. These snapshots include input hashes and feature coverage. These seasons were already examined during earlier development, which makes the forward archive especially important.

QB controls offer players listed as quarterbacks in the selected season's roster feed. The default assumption is the previous game's leading passer, **not a confirmed starter**; roster changes can invalidate it. New/lightly used quarterbacks shrink toward league average rather than receiving an invented backup penalty. What-if selections affect the scenario estimate only; they do not alter the schedule, outlook, or saved forecasts. Contributions add from Elo to the default matchup estimate in a fixed order and describe model arithmetic rather than causal effects. The experimental season outlook holds current efficiency/QB corrections fixed while simulated wins/losses update Elo; it does not simulate future injuries, QB changes, or future efficiency statistics.

Weather comes from [Open-Meteo](https://open-meteo.com/) using stadium coordinates derived from [greerreNFL's stadium dataset](https://github.com/greerreNFL/Stadiums). Outdoors/open-roof games within 15 days show forecast temperature and the maximum wind/gusts and total precipitation across the three-hour game window. Enclosed roofs exclude outdoor weather. Unknown stadiums/roof status or unavailable forecasts show an explanation. The forecast and retrieval timestamp are saved, never replaced by observed historical weather for evaluation. Weather provides context only in the original QB-aware model. The advanced model applies learned weather/style interactions when its stricter exposure, lead-time, and training-coverage requirements are met.

While the dashboard is used, eligible next-game forecasts within seven days of kickoff are saved locally to `.cache/forecasts.sqlite3`, with model settings, input-file hashes, assumptions, and weather. Duplicate inputs do not create another record; changed forecasts append a record. Completed games, missing kickoff times, and post-kickoff forecasts are excluded. US Eastern kickoff times are converted using daylight-saving-aware time zones. The interface evaluates the latest saved pre-kickoff forecast per game and model choice after results arrive, and rechecks revised kickoff times. Scenarios never enter this accuracy record. **No background scheduler runs**: open the dashboard or refresh it before games to capture forecasts. Refresh after games to evaluate them. The SQLite archive stays on this computer and is excluded from Git.

## Other launch options

```sh
python3 app.py --no-browser             # Print the address without opening it
python3 app.py --port 9000              # Choose a port
python3 app.py --offline                # Use saved data only
python3 app.py --data /path/to/games.csv # Use a local nflverse-format file
python3 app.py --check                  # Download/load data and validate calculations
python3 app.py --check --team MIN       # Validate Vikings calculations
python3 app.py --check --model baseline # Compare with original Elo
python3 app.py --check --model matchup  # Check experimental matchup model
python3 app.py --check --model advanced # Check advanced model after preparation
python3 app.py --backtest --season 2026 # Print model settings and evaluation as JSON
python3 -m steelers.evaluation --data .cache/games.csv --start 2018 --end 2025 # Audit the selection policy
python3 -m unittest discover -s tests -v
```

## Common chronological evaluation

The [shared evaluation protocol and results](docs/chronological-evaluation.md)
compare Elo, QB matchup, and advanced on the same 1,359 games from 2021–2025.
For each outer season, candidates fit only earlier seasons and selection uses
the last two earlier complete seasons. The runner records per-game forecasts,
exact settings, coverage fallbacks, and source hashes in the
[full report](docs/chronological-evaluation.json).

```sh
python3 -m steelers.experiments --data .cache/games.csv \
  --features .cache/features --advanced .cache/advanced \
  --start 2021 --end 2025 --config docs/experiments/chronological.json \
  --output docs/chronological-evaluation.json
```

This command reads local snapshots without downloads or evidence/archive writes.
Advanced Brier is 0.222061 versus Elo's 0.224121, but its probability-score
uncertainty interval includes no improvement. Previously explored years remain
a retrospective development audit. The runner does not promote a production
model. The application also now uses prior-only Elo offsets for historical QB
training; its [updated 2026 development report](docs/chronological-matchup-2026.json)
still fails the two-score gate, so Elo remains the default. Older snapshots are
preserved and use their original protocols.

Only Python’s standard library is used. The frontend is plain HTML/CSS/JavaScript with local SVG charts, without a build step, CDN, or tracking scripts. Python 3.10+ also works on Windows and Linux via `python app.py` or `python3 app.py`.

## Repository layout

For a code review and nine prioritized improvement plans, see the
[model improvement implementation guide](docs/model-improvement-plan.md).
It includes specific code findings, ordered implementation steps, test cases,
data requirements, and evaluation gates for each proposal.

`app.py` serves the local interface; `steelers/data.py` downloads and validates schedules; `steelers/model.py` tunes/calibrates Elo; `steelers/evaluation.py` audits Elo calibration; `steelers/experiments.py` runs shared nested chronological model comparisons; `steelers/features.py` loads weekly statistics; `steelers/matchup.py` fits/evaluates the QB challenger; `steelers/advanced.py` evaluates the advanced model; `steelers/pbp.py`, `steelers/evidence.py`, and `steelers/travel.py` prepare its inputs; `steelers/weather.py` loads weather forecasts; `steelers/forecast.py` archives pre-kickoff predictions; `steelers/analysis.py` calculates statistics and projections; `static/` contains the interface; `tests/` verifies calculations, future-result isolation, offline fallback, and HTTP endpoints. The original R/Shiny/Quarto project is retained under `legacy/r/` for reference and is not needed to run the Python app.
