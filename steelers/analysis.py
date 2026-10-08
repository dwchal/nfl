"""Records, chronological Elo ratings, and transparent season projections."""

import random
from collections import Counter
from datetime import date

BASE_RATING = 1500.0
HOME_ADVANTAGE = 55.0
K_FACTOR = 20.0
SIMULATIONS = 10000
AFC_NORTH = {"PIT", "BAL", "CIN", "CLE"}
NFC_NORTH = {"MIN", "GB", "CHI", "DET"}
SUPPORTED_TEAMS = {
    "PIT": {"code": "PIT", "name": "Pittsburgh Steelers", "short_name": "Steelers",
            "city": "Pittsburgh", "division": "AFC North", "tagline": "Black & gold.",
            "members": AFC_NORTH},
    "MIN": {"code": "MIN", "name": "Minnesota Vikings", "short_name": "Vikings",
            "city": "Minnesota", "division": "NFC North", "tagline": "Purple & gold.",
            "members": NFC_NORTH},
}
TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAR": "Los Angeles Rams",
    "LAC": "Los Angeles Chargers", "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints",
    "NYG": "New York Giants", "NYJ": "New York Jets", "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}
# Historical franchise abbreviations share rating history with today's team.
ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}


def team_key(team):
    return ALIASES.get(team, team)


def home_probability(home_rating, away_rating, neutral=False):
    advantage = 0 if neutral else HOME_ADVANTAGE
    return 1 / (1 + 10 ** ((away_rating - home_rating - advantage) / 400))


def update_ratings(ratings, home, away, result, neutral=False):
    expected = home_probability(ratings[home], ratings[away], neutral)
    change = K_FACTOR * (result - expected)
    ratings[home] += change
    ratings[away] -= change


def elo_ratings(games, season, team="PIT"):
    ratings = {}
    current_season = None
    history = []
    for game in games:
        if not season - 3 <= game.season <= season:
            continue
        if game.season != current_season:
            if current_season is not None:
                ratings = {team: BASE_RATING + (rating - BASE_RATING) * 2 / 3
                           for team, rating in ratings.items()}
            current_season = game.season
        home, away = team_key(game.home), team_key(game.away)
        ratings.setdefault(home, BASE_RATING)
        ratings.setdefault(away, BASE_RATING)
        if not game.completed:
            continue
        result = 1 if game.home_score > game.away_score else 0 if game.home_score < game.away_score else .5
        update_ratings(ratings, home, away, result, game.neutral)
        if game.season == season and team in {home, away}:
            history.append({"week": game.week, "date": game.day.isoformat(),
                            "kind": game.kind, "rating": round(ratings[team], 1)})
    return ratings, history


def record_table(games, ratings):
    teams = {}
    for game in games:
        for team in (game.home, game.away):
            teams.setdefault(team, {"team": team, "name": TEAM_NAMES.get(team, team),
                                    "wins": 0, "losses": 0, "ties": 0, "pf": 0, "pa": 0})
        if not game.completed:
            continue
        for team, scored, allowed in ((game.home, game.home_score, game.away_score),
                                     (game.away, game.away_score, game.home_score)):
            row = teams[team]
            row["pf"] += scored
            row["pa"] += allowed
            row["wins" if scored > allowed else "losses" if scored < allowed else "ties"] += 1
    for row in teams.values():
        played = row["wins"] + row["losses"] + row["ties"]
        row.update(games=played, differential=row["pf"] - row["pa"],
                   win_pct=(row["wins"] + .5 * row["ties"]) / played if played else None,
                   rating=round(ratings.get(team_key(row["team"]), BASE_RATING), 1))
    ranked = sorted(teams.values(), key=lambda r: (-r["rating"], r["team"]))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    return ranked


def project_season(remaining, ratings, wins, losses, ties, simulations=SIMULATIONS, team="PIT"):
    """Simulate all remaining REG games, updating ratings along each sample path."""
    rng = random.Random(42)
    counts = Counter()
    for _ in range(simulations):
        sample_ratings = ratings.copy()
        sample_wins = wins
        for game in remaining:
            home, away = team_key(game.home), team_key(game.away)
            sample_ratings.setdefault(home, BASE_RATING)
            sample_ratings.setdefault(away, BASE_RATING)
            home_win = rng.random() < home_probability(sample_ratings[home], sample_ratings[away], game.neutral)
            if (home == team and home_win) or (away == team and not home_win):
                sample_wins += 1
            update_ratings(sample_ratings, home, away, float(home_win), game.neutral)
        counts[sample_wins] += 1
    outcomes = sorted(counts)

    def quantile(fraction):
        cumulative = 0
        for outcome in outcomes:
            cumulative += counts[outcome]
            if cumulative >= simulations * fraction:
                return outcome
        return outcomes[-1]

    team_remaining = sum(team in {g.home, g.away} for g in remaining)
    total_games = wins + losses + ties + team_remaining
    return {"expected_wins": round(sum(w * n for w, n in counts.items()) / simulations, 1),
            "low": quantile(.1), "high": quantile(.9), "simulations": simulations,
            "remaining": team_remaining, "total_games": total_games,
            "distribution": [{"wins": w, "losses": total_games - ties - w, "ties": ties,
                              "probability": round(counts[w] / simulations, 4)} for w in outcomes]}


def default_season(games, today=None):
    today = today or date.today()
    # January–February belong to the preceding NFL season; March starts offseason.
    calendar_season = today.year - (today.month < 3)
    seasons = sorted({g.season for g in games})
    available = [s for s in seasons if s <= calendar_season]
    return max(available) if available else min(seasons)


def build_dashboard(games, season=None, include_playoffs=False, simulations=SIMULATIONS, team="PIT"):
    if team not in SUPPORTED_TEAMS:
        raise ValueError("Choose the Pittsburgh Steelers (PIT) or Minnesota Vikings (MIN).")
    profile = SUPPORTED_TEAMS[team]
    team_games = [g for g in games if team in {g.home, g.away}]
    if not team_games:
        raise ValueError(f"No {profile['short_name']} games are available in this schedule.")
    seasons = sorted({g.season for g in team_games}, reverse=True)
    season = default_season(team_games) if season is None else season
    if season not in seasons:
        raise ValueError("That season is not available in the schedule.")
    all_season = [g for g in games if g.season == season]
    selected = [g for g in all_season if include_playoffs or g.kind == "REG"]
    # Regular-season views exclude postseason results from ratings and next game.
    rating_games = [g for g in games if g.season != season or include_playoffs or g.kind == "REG"]
    ratings, history = elo_ratings(rating_games, season, team)
    rankings = record_table(selected, ratings)
    team_stats = next((r for r in rankings if r["team"] == team), None)
    if team_stats is None:
        raise ValueError(f"No {profile['short_name']} games are available in this season view.")
    schedule = []
    cumulative = 0
    for game in selected:
        if team not in {game.home, game.away}:
            continue
        at_home = game.home == team
        opponent = game.away if at_home else game.home
        scored, allowed = (game.home_score, game.away_score) if at_home else (game.away_score, game.home_score)
        result = None
        if game.completed:
            result = "W" if scored > allowed else "L" if scored < allowed else "T"
            cumulative += scored - allowed
        p_home = home_probability(ratings.get(team_key(game.home), BASE_RATING),
                                  ratings.get(team_key(game.away), BASE_RATING), game.neutral)
        schedule.append({"id": game.id, "week": game.week, "kind": game.kind,
                         "date": game.day.isoformat(), "kickoff": game.kickoff,
                         "opponent": opponent, "opponent_name": TEAM_NAMES.get(opponent, opponent),
                         "venue": "Neutral" if game.neutral else "Home" if at_home else "Away",
                         "stadium": game.stadium, "result": result, "scored": scored, "allowed": allowed,
                         "differential": scored - allowed if game.completed else None,
                         "cumulative": cumulative if game.completed else None,
                         "win_probability": None if game.completed else round(p_home if at_home else 1 - p_home, 4)})
    upcoming = [g for g in schedule if g["result"] is None]
    next_game = upcoming[0] if upcoming else None
    reg_games = [g for g in all_season if g.kind == "REG"]
    regular = next(r for r in record_table(reg_games, ratings) if r["team"] == team)
    # Projection always uses ratings as of the regular season, even in playoff view.
    reg_ratings, _ = elo_ratings([g for g in games if g.season != season or g.kind == "REG"], season, team)
    projection = project_season([g for g in reg_games if not g.completed], reg_ratings,
                                regular["wins"], regular["losses"], regular["ties"], simulations, team)
    division = sorted([r for r in rankings if r["team"] in profile["members"]],
                      key=lambda r: (-(r["win_pct"] or 0), -r["differential"], r["team"]))
    completed = [g for g in schedule if g["result"]]
    return {"season": season, "seasons": seasons, "include_playoffs": include_playoffs,
            "team": {key: value for key, value in profile.items() if key != "members"},
            "team_stats": team_stats, "rankings": rankings, "division": division,
            "schedule": schedule, "next_game": next_game, "projection": projection,
            "elo_history": history, "form": [g["result"] for g in completed[-5:]],
            "latest_result_date": max((g.day.isoformat() for g in selected if g.completed), default=None),
            "model": {"home_advantage": HOME_ADVANTAGE, "k_factor": K_FACTOR,
                      "initial_rating": BASE_RATING, "warmup_seasons": 3}}
