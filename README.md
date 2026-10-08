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

The **backtested default** adds scoring-margin weighting to Elo. Large wins have more influence, with logarithmic scaling and a correction for strong favorites beating weaker teams. The weighting is inspired by [FiveThirtyEight’s published NFL Elo approach](https://github.com/fivethirtyeight/nfl-elo-game/blob/master/forecast.py); this is our own implementation and evaluation, not FiveThirtyEight’s complete model.

For each season being viewed, the app uses the six most recent **completed prior seasons**. Three older seasons select settings; three later seasons compare the chosen candidate with the original model. Neither set includes the selected season. A fixed grid of 27 score-aware models tests update factors of 10/20/30, home advantages of 35/55/75, and offseason carryover of 0.5/two-thirds/0.8. Original Elo is also a tuning candidate. Selection minimizes Brier error, with log loss breaking ties. The chosen challenger becomes the default only if it improves **both** Brier error and log loss in the later comparison. That comparison is used as a deployment gate, so it is not a completely untouched final audit. Each three-season window must contain at least 500 regular-season games; otherwise the app uses original Elo.

All teams use the same selected settings. Ratings start at 1,500 and warm up using up to three preceding seasons. During evaluation, a forecast is calculated **before** that game’s result updates the ratings. Playoff games can update team strength; evaluation metrics cover regular-season games. Ties count as half a win for Elo, winning percentage, and probability-error metrics, and are excluded from winner-pick accuracy. Historical settings never use current-season or future-season scores. Completed-game probabilities in the schedule are retrospective pre-game replays, not archived published forecasts.

The **Original Elo** option preserves the previous model: update factor 20, 55-point home advantage, two-thirds offseason carryover, and no margin weighting. Neutral venues receive zero home advantage in both models. The forecast model selector changes upcoming probabilities, ratings, and season simulations; your choice is remembered between visits.

The [2026 evaluation snapshot](docs/model-evaluation-2026.json) tunes on 2020–2022 and evaluates 816 games from 2023–2025. It selected margin-aware Elo with update factor 20, home advantage 35, and two-thirds carryover. Brier error dropped from **0.2280 to 0.2219 (2.64%)**, log loss from **0.6479 to 0.6356**, and decisive-game pick accuracy rose from **62.1% to 64.4%**. Vikings-only Brier error improved (0.2359 → 0.2262); Steelers-only error worsened (0.2546 → 0.2666), with 51 games per team. The interface displays these team-specific results rather than assuming a league-wide improvement applies equally to Pittsburgh and Minnesota. The snapshot includes the source-file SHA-256; future provider corrections may change reproduced numbers slightly.

For the regular-season outlook, each simulation plays all remaining league games and updates ratings along the simulated path. Future margins are unknown, so simulated updates use wins and losses alone. Results use a fixed random seed for reproducibility. Future ties are not simulated. The model does **not** predict playoff qualification or implement official standings tiebreakers. The experimental matchup model adds player passing history, team efficiency, and rest. Injury reports and weather are displayed and archived as context; they do not receive unvalidated probability penalties. Betting markets are not used. Historical evaluation does not guarantee future accuracy; the displayed season range describes simulated outcomes, not validated statistical confidence. Historical summaries and remaining-game estimates use the final available data for the selected season.

## QB-aware matchup lab

The **Matchup lab** adds quarterback what-if controls for both teams, passing/rushing offense and defense contributions, game-week injury reports, kickoff weather, and a local archive of actual pre-kickoff forecasts. The **QB-aware matchup · experimental** selector applies the new model to upcoming games, completed-game replays, and the season outlook. **Backtested default** keeps the existing Elo model unless the new candidate improves both Brier error and log loss.

Weekly team/player statistics, rosters, and injuries come from [nflverse releases](https://github.com/nflverse/nflverse-data/releases). Validated downloads are saved in `.cache/features/`; failed downloads preserve valid snapshots. The first launch downloads nine seasons of compact weekly CSV data and may take longer. Extra data are optional: missing history falls back to Elo. `--offline` uses saved statistics and weather. `--data` uses only the supplied schedule and disables external features and archiving, so fixture data cannot be mistaken for genuine forecasts.

The challenger uses regularized logistic regression with Elo as its probability offset. It learns six corrections: passing offense/defense, rushing offense/defense, rest differential, and expected QB passing-efficiency change relative to the team's offense. Team efficiencies are EPA (expected points added) per dropback/carry, adjusted online for the opponent's earlier offensive or defensive strength. Recent games receive more weight; small samples and offseason history shrink toward league average. QB passing statistics follow the individual across teams. This version does not include rushing QB value, protection/receiver-specific injury adjustments, travel, coaching, or drive-based score simulations.

For a selected season, coefficients train on the first three of the six preceding seasons and stay frozen throughout the later comparison and selected season. A fixed full-feature challenger uses L2 penalty 0.03. Smaller rest-only and efficiency-plus-rest models train on the first two older seasons and validate on the third as feature diagnostics; those checks do not choose the final challenger. Each historical forecast sees only earlier games' statistics, with a one-day publication-lag approximation; same-day statistics are excluded. Schedule QB starter columns are never used, because they can be populated after the game. Historical weekly statistics can contain later provider corrections, so this remains a retrospective development comparison, not an untouched live audit. At least 200 games per prior season and 95% team/QB statistics coverage are required. Coefficients are shared across all teams; Steelers/Vikings results are also shown separately.

The [2026 matchup evaluation](docs/matchup-evaluation-2026.json) compares 816 regular-season games from 2023–2025 after training corrections on 2020–2022. Winner-pick accuracy rose **64.4% → 65.3%**, but Brier error worsened **0.2219 → 0.2223**, and log loss worsened **0.6356 → 0.6363**. Both team subsets also had worse Brier error (Pittsburgh 0.2666 → 0.2747; Minnesota 0.2262 → 0.2295). Consequently the challenger is **experimental and not the default**. The snapshot includes input hashes and feature coverage. These seasons were already examined during earlier development, which makes the forward archive especially important.

QB controls offer players listed as quarterbacks in the selected season's roster feed. The default assumption is the previous game's leading passer, **not a confirmed starter**; roster changes can invalidate it. New/lightly used quarterbacks shrink toward league average rather than receiving an invented backup penalty. What-if selections affect the scenario estimate only; they do not alter the schedule, outlook, or saved forecasts. Contributions add from Elo to the default matchup estimate in a fixed order and describe model arithmetic rather than causal effects. The experimental season outlook holds current efficiency/QB corrections fixed while simulated wins/losses update Elo; it does not simulate future injuries, QB changes, or future efficiency statistics.

Weather comes from [Open-Meteo](https://open-meteo.com/) using stadium coordinates derived from [greerreNFL's stadium dataset](https://github.com/greerreNFL/Stadiums). Outdoors/open-roof games within 15 days show forecast temperature and the maximum wind/gusts and total precipitation across the three-hour game window. Enclosed roofs exclude outdoor weather. Unknown stadiums/roof status or unavailable forecasts show an explanation. The forecast and retrieval timestamp are saved, never replaced by observed historical weather for evaluation. Weather currently provides context only; archived forecasts can support a later, properly tested weather correction.

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
python3 app.py --backtest --season 2026 # Print model settings and evaluation as JSON
python3 -m unittest discover -s tests -v
```

Only Python’s standard library is used. The frontend is plain HTML/CSS/JavaScript with local SVG charts, without a build step, CDN, or tracking scripts. Python 3.10+ also works on Windows and Linux via `python app.py` or `python3 app.py`.

## Repository layout

`app.py` serves the local interface; `steelers/data.py` downloads and validates schedules; `steelers/model.py` tunes Elo; `steelers/features.py` loads weekly statistics; `steelers/matchup.py` fits/evaluates the challenger; `steelers/weather.py` loads weather forecasts; `steelers/forecast.py` archives pre-kickoff predictions; `steelers/analysis.py` calculates statistics and projections; `static/` contains the interface; `tests/` verifies calculations, future-result isolation, offline fallback, and HTTP endpoints. The original R/Shiny/Quarto project is retained under `legacy/r/` for reference and is not needed to run the Python app.
