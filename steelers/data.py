"""Validated nflverse schedules with atomic caching and offline fallback."""

import csv
import io
import os
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

SOURCE = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
CACHE_SECONDS = 3600
MISSING = {"", "NA", "nan"}


@dataclass(frozen=True)
class Game:
    id: str
    season: int
    kind: str
    week: int
    day: date
    kickoff: str
    home: str
    away: str
    home_score: int | None
    away_score: int | None
    neutral: bool
    stadium: str

    @property
    def completed(self):
        return self.home_score is not None and self.away_score is not None


def parse_games(content):
    reader = csv.DictReader(io.StringIO(content))
    required = {"game_id", "season", "game_type", "week", "gameday",
                "home_team", "away_team", "home_score", "away_score"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("The schedule download is missing required columns.")
    games = []
    seen = set()
    for row in reader:
        # Preseason games do not contribute to records or team strength.
        if row["game_type"] not in {"REG", "WC", "DIV", "CON", "SB"}:
            continue
        try:
            score = lambda value: None if value in MISSING else int(value)
            game = Game(
                row["game_id"], int(row["season"]), row["game_type"],
                int(row["week"]), date.fromisoformat(row["gameday"]),
                row.get("gametime", ""), row["home_team"], row["away_team"],
                score(row["home_score"]), score(row["away_score"]),
                row.get("location", "").lower() == "neutral", row.get("stadium", ""),
            )
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid schedule row: {row['game_id']}") from error
        if not game.id or not game.home or not game.away or game.home == game.away:
            raise ValueError("Invalid team or game identifier in schedule.")
        if game.id in seen:
            raise ValueError(f"Duplicate game identifier: {game.id}")
        if any(s is not None and s < 0 for s in (game.home_score, game.away_score)):
            raise ValueError(f"Invalid score: {game.id}")
        seen.add(game.id)
        games.append(game)
    if not games:
        raise ValueError("The schedule contains no regular-season or playoff games.")
    return sorted(games, key=lambda g: (g.day, g.kickoff, g.id))


class DataUnavailable(Exception):
    pass


class ScheduleStore:
    def __init__(self, cache, offline=False, source_file=None):
        self.cache = Path(cache)
        self.offline = offline
        self.source_file = Path(source_file) if source_file else None
        self.lock = threading.Lock()
        self.last_attempt = 0

    def load(self, refresh=False):
        with self.lock:
            return self._load(refresh)

    def _load(self, refresh):
        if self.source_file:
            try:
                games = parse_games(self.source_file.read_text(encoding="utf-8-sig"))
                return games, self._metadata(self.source_file, "Local data file")
            except (OSError, ValueError) as error:
                raise DataUnavailable(f"Cannot read local schedule: {error}") from error

        cached = None
        if self.cache.exists():
            try:
                cached = parse_games(self.cache.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                pass
        age = time.time() - self.cache.stat().st_mtime if cached else float("inf")
        # Back off briefly after failed network requests, even for a fresh launch.
        if cached and not refresh and (age < CACHE_SECONDS or time.time() - self.last_attempt < 60):
            warning = "Offline mode: using saved data." if self.offline else ""
            if age >= CACHE_SECONDS:
                warning = "Using saved data. Refresh to try downloading the latest results."
            return cached, self._metadata(self.cache, "Saved nflverse data", warning)
        if not self.offline:
            self.last_attempt = time.time()
            try:
                request = Request(SOURCE, headers={"User-Agent": "SteelersDashboard/1.0"})
                with urlopen(request, timeout=20) as response:
                    raw = response.read(12_000_001)
                if len(raw) > 12_000_000:
                    raise ValueError("Schedule download exceeds the size limit.")
                content = raw.decode("utf-8-sig")
                games = parse_games(content)
                self.cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.cache.with_suffix(".tmp")
                temporary.write_text(content, encoding="utf-8")
                os.replace(temporary, self.cache)
                return games, self._metadata(self.cache, "nflverse")
            except (OSError, URLError, ValueError, UnicodeError):
                if cached:
                    return cached, self._metadata(
                        self.cache, "Saved nflverse data",
                        "Download unavailable. Showing saved data; results may be out of date.",
                    )
        if cached:
            return cached, self._metadata(self.cache, "Saved nflverse data", "Offline mode: using saved data.")
        raise DataUnavailable(
            "No saved schedule is available. Connect to the internet and press Retry, "
            "or start the app with --data /path/to/games.csv."
        )

    @staticmethod
    def _metadata(path, source, warning=""):
        return {"source": source, "source_url": SOURCE, "warning": warning,
                "downloaded_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}
