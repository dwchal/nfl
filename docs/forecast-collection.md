# Paired forecast collection

Step 4 adds a collector that runs independently of the dashboard. It forecasts
every unplayed league game within seven days using Elo, QB matchup, and advanced
from the same pinned inputs and UTC cutoff. Missing model inputs produce an
explicit Elo fallback for that choice. Neither experimental model is promoted.

## Run once

From the repository root:

```sh
python3 -m steelers.capture --once --season 2026
```

The default reads/refreshes the normal schedule and optional input caches, and
collects current injury/weather evidence for eligible upcoming games. To use
only saved data:

```sh
python3 -m steelers.capture --once --season 2026 --offline
```

Forecasts remain local in `.cache/forecasts.sqlite3`. Immutable snapshots and
their first-observed index are under `.cache/snapshots/`. Both are ignored by
Git. `--archive` and `--snapshots` can select other locations. The JSON result
reports captured game IDs, the shared input cutoff, actual completion time,
manifest/artifact IDs, explicit model fallbacks, and inserted/retried counts.
`--output /path/to/summary.json` saves this small result.

Inputs are loaded once, pinned, and fitted before prediction. Actual completion
time is recorded after all predictions are calculated. A calculation that runs
past kickoff is skipped, and slow fitting cannot manufacture a capture before
a horizon deadline. There is no historical `--as-of` collection option.

Retries deduplicate by game, 15-minute UTC slot, manifest, artifact set, and
optional policy. Changed inputs can create another record within a slot.
Successful later slots are retained even when inputs are unchanged, allowing
separate real records near the 24-hour and 1-hour deadlines.

## Optional schedule: every 15 minutes

The implementation does not install a background task. On this Mac, an optional
user-managed cron entry can run the command independently of dashboard visits.
First run it manually above and confirm the output. Then use `crontab -e` to add:

```cron
*/15 * * * * cd /Users/dougchallener/github/nfl && /opt/homebrew/bin/python3 -m steelers.capture --once --season 2026 >> /Users/dougchallener/github/nfl/.cache/capture.log 2>&1
```

Use `command -v python3` to check the absolute interpreter path on another
computer. Keep the season argument current. The example enables normal refresh;
add `--offline` for a collector that only uses caches. Inspect `.cache/capture.log`
for source/fit errors. Removing this entry stops scheduled collection.

The machine must be awake, and online refresh requires connectivity. Missed
executions are capture gaps; the command does not invent forecasts for sleeping
or offline periods. A scheduled command can continue to run when the dashboard
is closed.

## Evaluate the declared horizons

After results arrive:

```sh
python3 -m steelers.capture --report --season 2026 --offline \
  --output .cache/paired-report-2026.json
```

Remove `--offline` to refresh the schedule/results first. The report uses
completed regular-season games and separately grades:

| Horizon | Deadline | Earliest allowed capture |
| --- | --- | --- |
| `24h` | Kickoff minus 24 hours | Two hours before that deadline |
| `1h` | Kickoff minus one hour | 30 minutes before that deadline |

Both endpoints are inclusive. Select the latest valid **paired collection**
at or before the deadline; every model uses the same game, collection time,
and manifest. A later forecast is never substituted. Revised kickoff times are
checked again, and repeated team views count only once. Missing/incomplete
captures, unknown kickoffs, stale records, and missing artifacts/manifests do
not get scored as valid pairs. Reports list capture gaps explicitly.

By default, the report separates each exact artifact set within each horizon.
Every model summary includes Brier error, log loss, decisive-game accuracy,
PIT/MIN subsets, fallback counts, paired week-block uncertainty, and changed
winner picks. Ties contribute outcome 0.5 to probability scores and are excluded
from winner accuracy. Small groups need more captures before conclusions are
useful. Model availability and input provenance can be inspected through the
registered artifacts/manifests and the per-game forecast payloads.

### Explicitly frozen pooling policy

The default never pools different artifacts. To declare a policy for a
prospective run, create a new allowlist before collection:

```sh
python3 -m steelers.capture --once --season 2026 \
  --freeze-policy .cache/policies/season-2026.json
```

This writes a new file containing the three artifact IDs and declared horizons,
registers its deterministic policy ID, then labels the captures with that ID.
Reuse it for subsequent collection and reporting:

```sh
python3 -m steelers.capture --once --season 2026 --policy .cache/policies/season-2026.json
python3 -m steelers.capture --report --season 2026 --offline \
  --policy .cache/policies/season-2026.json --output .cache/policy-report-2026.json
```

A policy may explicitly allow multiple already-registered artifact IDs per
family. It must be frozen before the captures being pooled. Changing the file
creates a different policy ID and cannot relabel earlier captures. Unknown
artifacts or artifacts outside its allowlist cause collection to fail; an
implementation update, new fit, or changed fallback artifact requires a new
deliberate policy. Run without `--policy` to retain separate artifact groups.

## What is preserved

- `steelers/prediction.py` provides `predict_game(game, artifact, inputs,
  as_of_utc)` to both the collector and HTTP path. It returns full-precision
  home probability, input/feature hashes, artifact ID, evidence, and fallbacks.
  Only the UI rounds estimates. The legacy archive now receives full precision.
- `steelers/provenance.py` saves immutable SHA-256-addressed original source
  bytes and normalized replayable inputs. Normalized games, weekly statistics,
  and PBP are split by season, so a weather update reuses stable historical
  chunks. First retrieval times are durably indexed rather than inferred from
  mtime. Provider timestamps and source URLs are preserved when available;
  unavailable publication times are not invented.
- The actual Python source files and their hash are also pinned. The code
  revision is fixed when the process starts. Restart a long-running dashboard
  after updating its code. Artifacts include Git revision, source hash, family,
  version, ordered labels, coefficients, Elo settings, training/comparison
  seasons, fitting options, support gates, and fallback status.
- New `collection_runs`, `forecast_runs`, `model_artifacts`, `input_manifests`,
  and `capture_policies` tables preserve paired runs. Every collection is atomic;
  its manifest and artifacts must already be registered. A 15-minute retry
  cannot overwrite an earlier capture. Source corrections create new snapshots.
- The original `forecasts` table and every existing row remain intact. Its
  dashboard-use evaluation remains explicitly labeled as legacy and mixes lead
  times; those rows are not migrated into fixed-horizon results.

Prediction times must be timezone-aware. Future completed rows cannot update
ratings/state. Weekly and PBP observations use the documented next-calendar-day
publication approximation, including the final pending-statistics flush, using
the schedule's Eastern calendar. Evidence lookup is bounded by prediction time.
Later-retrieved corrections cannot be used with an earlier cutoff. Historical
training inputs remain reconstructed history even when their files are hashed;
the prospective archive records when this application actually observed inputs.

## Fixture isolation

`--data` is a fixture run and requires a separate archive:

```sh
python3 -m steelers.capture --once --season 2026 --offline \
  --data /path/to/fixture.csv --archive /tmp/nfl-fixture.sqlite3
```

The real archive is rejected for fixtures. A separate snapshot directory is
derived beside the fixture database. Optional fixture `--features`/`--advanced`
inputs use saved files only, and no live weather/injury evidence is captured.
All such records are labeled `fixture_reconstructed`. Output files cannot
replace the archive, policy, or source snapshots. The dashboard's existing
`app.py --data` behavior still disables external features and archiving.

## Verification

The new tests cover shared clocks/manifests, full-precision storage, atomic
pairs, artifact and code identities, retrieval-time isolation, pending-flush
cutoffs, later results/statistics/lineups/weather mutations, retries and later
slots, both deadline boundaries, late/stale captures, revised kickoff, duplicate
team views, missing artifacts/manifests, frozen pooling, and old-row retention.
Fixture CLI tests exercise a real isolated SQLite archive.

All 136 tests pass. The [verification record](forecast-collection-verification.json)
documents a local run whose 45 probabilities and feature digests reproduce
exactly from immutable inputs and registered artifacts.

An offline run during implementation captured all three choices for 15 upcoming
2026 games. The report correctly showed no fixed-horizon pairs for 65 games that
had already finished before this collector existed. Historical accuracy was not
fabricated. The next priority is item 5: QB uncertainty and rushing contribution.
