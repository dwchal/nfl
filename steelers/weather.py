"""As-issued kickoff weather snapshots; context until independently validated."""

import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from .forecast import kickoff_utc

VENUES = Path(__file__).resolve().parent.parent / "data" / "venues.json"


def summarize(payload, kickoff):
    hourly = payload["hourly"]
    hours = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in hourly["time"]]
    start = kickoff.replace(minute=0, second=0, microsecond=0)
    indices = [i for i, hour in enumerate(hours) if start <= hour < start + timedelta(hours=3)]
    if not indices:
        raise ValueError("Kickoff outside forecast horizon")
    def values(key):
        result = [hourly[key][i] for i in indices]
        if any(v is None or not isinstance(v, (float, int)) or not math.isfinite(v) for v in result):
            raise ValueError("Missing hourly weather values")
        return result
    return {"temperature_f": round(values("temperature_2m")[0]),
            "wind_mph": round(max(values("wind_speed_10m"))),
            "gust_mph": round(max(values("wind_gusts_10m"))),
            "precipitation_inches": round(sum(values("precipitation")), 2)}


class WeatherStore:
    def __init__(self, directory, offline=False):
        self.directory = Path(directory)
        self.offline = offline
        self.last_attempts = {}

    def load(self, game, refresh=False, now=None):
        now = now or datetime.now(timezone.utc)
        result = {"status": "unavailable", "source": "Open-Meteo", "source_url": "https://open-meteo.com/",
                  "note": "Weather is saved as forecast context; it does not yet change win probabilities."}
        if game.roof.lower() in {"dome", "closed"}:
            return {**result, "status": "indoors", "reason": "Enclosed roof: outdoor weather is excluded."}
        kickoff = kickoff_utc(game)
        if kickoff is None or kickoff <= now or kickoff > now + timedelta(days=15):
            return {**result, "reason": "A kickoff forecast is available only for upcoming games within 15 days."}
        if game.roof.lower() not in {"outdoors", "open"}:
            return {**result, "reason": "Roof status is unknown; weather effects are not assumed."}
        venues = json.loads(VENUES.read_text())
        venue = venues.get(game.stadium_id)
        if venue is None:
            return {**result, "reason": "Stadium coordinates are unavailable."}
        path = self.directory / f"{game.id}.json"
        cached = None
        warning = ""
        if path.exists():
            try:
                cached = json.loads(path.read_text())
                summarize(cached, kickoff)
                datetime.fromisoformat(cached["retrieved_at"])
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                cached = None
        fresh = cached and time.time() - path.stat().st_mtime < 3600
        backed_off = time.time() - self.last_attempts.get(game.id, 0) < 60
        if not self.offline and (refresh or (not fresh and not backed_off)):
            self.last_attempts[game.id] = time.time()
            query = urlencode({"latitude": venue[0], "longitude": venue[1], "forecast_days": 16,
                               "hourly": "temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation",
                               "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
                               "precipitation_unit": "inch", "timezone": "UTC"})
            try:
                with urlopen(f"https://api.open-meteo.com/v1/forecast?{query}", timeout=8) as response:
                    raw = response.read(300_001)
                if len(raw) > 300_000:
                    raise ValueError("Weather response too large")
                downloaded = json.loads(raw)
                summarize(downloaded, kickoff)
                downloaded["retrieved_at"] = now.isoformat()
                self.directory.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(downloaded, allow_nan=False))
                os.replace(temporary, path)
                cached = downloaded
                fresh = True
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                warning = "Weather download unavailable."
        if not cached:
            return {**result, "reason": warning or "No saved forecast; connect to the internet to load weather."}
        return {**result, "status": "forecast", **summarize(cached, kickoff), "retrieved_at": cached["retrieved_at"],
                "warning": "Using saved weather forecast." if self.offline or warning or not fresh else "",
                "latitude": venue[0], "longitude": venue[1]}
