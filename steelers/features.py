"""Small weekly nflverse datasets, validated before replacing offline snapshots."""

import csv
import hashlib
import io
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"


def number(row, key):
    value = row.get(key, "")
    if value in {"", "NA", "nan", None}:
        return 0.0
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite player/team statistic")
    return result


@dataclass(frozen=True)
class TeamWeek:
    passing_epa: float
    dropbacks: float
    rushing_epa: float
    carries: float


@dataclass(frozen=True)
class QBWeek:
    id: str
    name: str
    epa: float
    dropbacks: float


def parse_feature_csv(content, kind, year):
    reader = csv.DictReader(io.StringIO(content))
    required = {"season", "team"}
    required |= {"week", "season_type", "passing_epa", "attempts", "sacks_suffered"} if kind in {"team", "player"} else {"gsis_id", "position", "full_name"}
    if kind == "team":
        required |= {"rushing_epa", "carries"}
    if kind == "player":
        required |= {"player_id", "player_display_name", "position"}
    if kind == "injuries":
        required |= {"week", "report_status", "practice_status"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"Missing {kind} columns")
    result = {} if kind in {"team", "player"} else []
    for row in reader:
        if int(row["season"]) != year:
            raise ValueError("Wrong season in feature download")
        if kind in {"team", "player"}:
            key = (year, int(row["week"]), row["season_type"], row["team"])
            if row["season_type"] not in {"REG", "POST"}:
                continue
            if kind == "team":
                if key in result:
                    raise ValueError("Duplicate weekly team statistics")
                result[key] = TeamWeek(number(row, "passing_epa"), number(row, "attempts") + number(row, "sacks_suffered"),
                                       number(row, "rushing_epa"), number(row, "carries"))
                if result[key].dropbacks < 0 or result[key].carries < 0:
                    raise ValueError("Negative play count")
            elif row["position"] == "QB":
                qb = QBWeek(row["player_id"], row["player_display_name"],
                            number(row, "passing_epa"), number(row, "attempts") + number(row, "sacks_suffered"))
                if qb.dropbacks < 0 or not qb.id:
                    raise ValueError("Invalid quarterback statistics")
                result.setdefault(key, []).append(qb)
        else:
            result.append(row)
    if not result:
        raise ValueError(f"Empty {kind} download")
    return result


class FeatureStore:
    def __init__(self, directory, offline=False):
        self.directory = Path(directory)
        self.offline = offline
        self.last_attempts = {}

    def _file(self, kind, year, refresh):
        names = {"team": ("stats_team", f"stats_team_week_{year}.csv"),
                 "player": ("stats_player", f"stats_player_week_{year}.csv"),
                 "rosters": ("rosters", f"roster_{year}.csv"),
                 "injuries": ("injuries", f"injuries_{year}.csv")}
        release, filename = names[kind]
        path = self.directory / filename
        cached = None
        warning = ""
        if path.exists():
            try:
                raw = path.read_bytes()
                cached = (parse_feature_csv(raw.decode("utf-8-sig"), kind, year), raw)
            except (OSError, ValueError, UnicodeError):
                warning = "Invalid saved data"
        current_year = datetime.now(timezone.utc).year
        ttl = 3600 if year >= current_year - 1 else 30 * 86400
        age = time.time() - path.stat().st_mtime if cached else float("inf")
        backed_off = time.time() - self.last_attempts.get(filename, 0) < 60
        if not self.offline and (refresh or (age >= ttl and not backed_off)):
            self.last_attempts[filename] = time.time()
            try:
                request = Request(f"{RELEASES}/{release}/{filename}", headers={"User-Agent": "NFLDashboard/2.0"})
                with urlopen(request, timeout=15) as response:
                    raw = response.read(15_000_001)
                if len(raw) > 15_000_000:
                    raise ValueError("Feature file exceeds size limit")
                parsed = parse_feature_csv(raw.decode("utf-8-sig"), kind, year)
                self.directory.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(raw)
                os.replace(temporary, path)
                cached = parsed, raw
            except (OSError, ValueError, UnicodeError):
                warning = "Download unavailable; saved data used" if cached else "Download unavailable"
        if self.offline:
            warning = "Offline snapshot" if cached else "No saved data"
        if not cached:
            return kind, year, None, None, {"file": filename, "warning": warning or "No data"}
        return kind, year, cached[0], cached[1], {
            "file": filename, "warning": warning,
            "downloaded_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "sha256": hashlib.sha256(cached[1]).hexdigest()}

    def load(self, season, refresh=False):
        # Two warmup years plus the six-year development comparison, and target.
        tasks = [(kind, year) for year in range(max(1999, season - 8), season + 1) for kind in ("team", "player")]
        tasks += [(kind, season) for kind in ("rosters", "injuries")]
        bundle = {"team": {}, "player": {}, "rosters": [], "injuries": [], "sources": []}
        digest = hashlib.sha256()
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = pool.map(lambda task: self._file(*task, refresh), tasks)
            for kind, year, parsed, raw, metadata in results:
                bundle["sources"].append(metadata)
                if raw is not None:
                    digest.update(f"{kind}:{year}:".encode())
                    digest.update(raw)
                    if isinstance(parsed, dict):
                        bundle[kind].update(parsed)
                    else:
                        bundle[kind] = parsed
        bundle["digest"] = digest.hexdigest()
        bundle["available"] = bool(bundle["team"] and bundle["player"])
        return bundle
