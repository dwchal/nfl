"""Scheduled travel and team-local kickoff features; no guessed itineraries."""

import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

from .forecast import kickoff_utc
from .model import team_key

VENUES = json.loads((Path(__file__).resolve().parent.parent / "data/venues.json").read_text())
HOME_VENUES = dict(zip(
    "ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LA LAC LV MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS".split(),
    "PHO00 ATL97 BAL00 BUF00 CAR00 CHI98 CIN00 CLE00 DAL00 DEN00 DET00 GNB00 HOU00 IND00 JAX00 KAN00 LAX01 LAX01 VEG00 MIA00 MIN01 BOS00 NOR00 NYC01 NYC01 PHI00 PIT00 SEA00 SFO01 TAM00 NAS00 WAS00".split()))
ZONES = {**dict.fromkeys("ATL BAL BUF CAR CIN CLE DET IND JAX MIA NE NYG NYJ PHI PIT TB WAS".split(), "America/New_York"),
         **dict.fromkeys("CHI DAL GB HOU KC MIN NO TEN".split(), "America/Chicago"),
         **dict.fromkeys("LA LAC LV SEA SF".split(), "America/Los_Angeles"),
         "DEN": "America/Denver", "ARI": "America/Phoenix"}
LABELS = ("Travel distance", "Early body-clock kickoff", "Late body-clock kickoff",
          "Short-rest travel", "Consecutive away games")


def base(team, season):
    key = team_key(team)
    if key == "LV" and season < 2020:
        return (37.7516, -122.2005), "America/Los_Angeles"
    if key == "LAC" and season < 2017:
        return (32.7831, -117.1196), "America/Los_Angeles"
    if key == "LA" and season < 2016:
        return (38.6329, -90.1885), "America/Chicago"
    return VENUES.get(HOME_VENUES.get(key)), ZONES.get(key)


def distance(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    value = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * math.asin(min(1, math.sqrt(value)))


def context(game, away_streaks=None):
    kickoff, venue = kickoff_utc(game), VENUES.get(game.stadium_id)
    streaks, teams = away_streaks or {}, {}
    for team, rest in ((game.home, game.home_rest), (game.away, game.away_rest)):
        origin, zone = base(team, game.season)
        local = kickoff.astimezone(ZoneInfo(zone)) if kickoff and zone else None
        miles = distance(origin, venue) if origin and venue else None
        # A true home game has no travel even when the historical stadium moved.
        home = team == game.home and not game.neutral
        if home:
            miles = 0.
        hour = local.hour + local.minute / 60 if local else None
        teams[team] = {"miles": round(miles) if miles is not None else None,
                       "body_clock": local.strftime("%H:%M") if local else None, "time_zone": zone,
                       "early": max(0, 12 - hour) / 3 if hour is not None else 0.,
                       "late": max(0, hour - 21) / 3 if hour is not None else 0.,
                       "short_travel": miles / 1000 if miles is not None and rest is not None and rest < 6 else 0.,
                       "away_streak": 0 if home else streaks.get(team_key(team), 0) + 1}
    h, a = teams[game.home], teams[game.away]
    features = [(a["miles"] - h["miles"]) / 1000 if a["miles"] is not None and h["miles"] is not None else 0.,
                a["early"] - h["early"], a["late"] - h["late"], a["short_travel"] - h["short_travel"],
                (a["away_streak"] - h["away_streak"]) / 3]
    return {"teams": teams, "features": features,
            "note": "Distance from team home base, not actual itinerary. Team-local kickoff uses daylight-saving-aware time zones; travel acclimatization is unknown."}
