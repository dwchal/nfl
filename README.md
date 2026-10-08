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

The default season follows the NFL calendar (January–February belong to the preceding year), with a fallback to the latest season available in the feed. Seasons are no longer hardcoded to 2024. Preseason games are excluded. The regular-season view excludes playoff results from records and current-season ratings; “Regular + playoffs” includes them. The outlook always predicts the **regular-season record**. A finished season shows its final record rather than inventing another matchup.

## Data and model

Schedules and scores come directly from [nflverse’s games.csv](https://github.com/nflverse/nfldata/blob/master/data/games.csv), the schedule feed used by the original R project. Downloads are validated and cached in `.cache/games.csv`. Data is refreshed when requested after an hour, or immediately using **Refresh data**. An unsuccessful download preserves the last valid snapshot and shows a warning. Provider update delays can affect scores and kickoff times; this is not a live play-by-play feed. Data remains subject to the provider’s terms.

Elo uses an initial rating of 1,500, update factor 20, and a 55-point home-field adjustment (zero at neutral venues). It warms up using up to three earlier seasons, regressing each rating one-third toward 1,500 at the season boundary. Games update ratings in chronological order. Ties count as half a win in records’ winning percentages and Elo results.

For the regular-season outlook, each simulation plays all remaining league games and updates ratings along the simulated path. Results use a fixed random seed for reproducibility. Future ties are not simulated. The model does **not** predict playoff qualification, implement official standings tiebreakers, or incorporate injuries, player statistics, roster changes, rest, weather, margins of victory, or betting markets. The probabilities are **uncalibrated Elo estimates**; the displayed range describes simulated outcomes, not validated statistical confidence. Historical views use the final available season data and are not backtests.

## Other launch options

```sh
python3 app.py --no-browser             # Print the address without opening it
python3 app.py --port 9000              # Choose a port
python3 app.py --offline                # Use saved data only
python3 app.py --data /path/to/games.csv # Use a local nflverse-format file
python3 app.py --check                  # Download/load data and validate calculations
python3 app.py --check --team MIN       # Validate Vikings calculations
python3 -m unittest discover -s tests -v
```

Only Python’s standard library is used. The frontend is plain HTML/CSS/JavaScript with local SVG charts, without a build step, CDN, or tracking scripts. Python 3.10+ also works on Windows and Linux via `python app.py` or `python3 app.py`.

## Repository layout

`app.py` serves the local interface; `steelers/data.py` downloads and validates schedules; `steelers/analysis.py` calculates statistics and projections; `static/` contains the interface; `tests/` verifies calculations, offline fallback, and HTTP endpoints. The original R/Shiny/Quarto project is retained under `legacy/r/` for reference and is not needed to run the Python app.
