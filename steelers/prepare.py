"""Prepare compact advanced-model data without adding runtime dependencies."""

import argparse
import json
import os
from pathlib import Path
from urllib.request import urlopen

from .data import parse_games
from .evidence import EvidenceStore
from .pbp import PBPStore


def depth_chart(store, directory, year, offline=False, refresh=False):
    path = Path(directory) / f"depth_{year}.csv"
    if path.exists() and not refresh:
        return store.import_depth(path.read_text())
    if offline:
        return 0
    url = f"https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{year}.csv"
    try:
        with urlopen(url, timeout=45) as response:
            raw = response.read(100_000_001)
        if len(raw) > 100_000_000:
            raise ValueError("Depth chart file exceeds limit")
        count = store.import_depth(raw.decode("utf-8-sig"))
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(raw)
        os.replace(temporary, path)
        return count
    except (OSError, ValueError) as error:
        if path.exists():
            return store.import_depth(path.read_text())
        return {"warning": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(".cache/games.csv"))
    parser.add_argument("--directory", type=Path, default=Path(".cache/advanced"))
    parser.add_argument("--start", type=int, default=2018)
    parser.add_argument("--end", type=int, default=2026)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--weather-history", action="store_true")
    args = parser.parse_args()
    if not 1999 <= args.start <= args.end <= 2100:
        parser.error("Choose an ordered season range starting in 1999 or later")
    store = PBPStore(args.directory, args.offline)
    evidence = EvidenceStore(args.directory / "evidence.sqlite3")
    for year in range(args.start, args.end + 1):
        payload = store.load(year, args.refresh)
        print(json.dumps({"season": year, "games": len((payload or {}).get("games", {})),
                          "plays": (payload or {}).get("plays"), "warning": (payload or {}).get("warning")}), flush=True)
    for year in range(max(2025, args.start), args.end + 1):
        print(json.dumps({"depth_season": year, "snapshots": depth_chart(evidence, args.directory, year, args.offline, args.refresh)}), flush=True)
    if args.weather_history and not args.offline:
        games = [g for g in parse_games(args.data.read_text()) if args.start <= g.season <= args.end]
        print(json.dumps({"weather": evidence.historical_weather(games, args.refresh)}), flush=True)


if __name__ == "__main__":
    main()
