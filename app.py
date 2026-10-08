#!/usr/bin/env python3
"""Launch the local Steelers and Vikings dashboard. No third-party packages required."""

import argparse
import json
import mimetypes
import threading
import time
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from steelers.analysis import SUPPORTED_TEAMS, build_dashboard, default_season
from steelers.data import DataUnavailable, ScheduleStore
from steelers.model import BASE_RATING, home_probability, replay_season, select_model, team_key
from steelers.features import FeatureStore
from steelers.forecast import ForecastArchive
from steelers.matchup import VERSION, evaluate, next_context
from steelers.weather import WeatherStore

ROOT = Path(__file__).resolve().parent


def make_handler(store, feature_store=None, archive=None, weather_store=None):
    dashboards = OrderedDict()
    feature_cache, matchup_cache = {}, {}
    feature_checked = {}
    analysis_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type="application/json; charset=utf-8"):
            if isinstance(body, dict):
                body = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/dashboard":
                return self.dashboard(parse_qs(parsed.query))
            # Explicit routes avoid exposing cache files, repository contents, or traversal.
            assets = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css",
                      "/favicon.svg": "favicon.svg", "/favicon-vikings.svg": "favicon-vikings.svg"}
            if parsed.path not in assets:
                return self.send(404, {"error": "Not found"})
            path = ROOT / "static" / assets[parsed.path]
            return self.send(200, path.read_bytes(), mimetypes.guess_type(path)[0] or "application/octet-stream")

        def do_POST(self):
            if urlparse(self.path).path != "/api/refresh":
                return self.send(404, {"error": "Not found"})
            # A remote website must not be able to trigger downloads against this server.
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                return self.send(403, {"error": "Refresh must be requested from this dashboard."})
            return self.dashboard(parse_qs(urlparse(self.path).query), refresh=True)

        def dashboard(self, query, refresh=False):
            try:
                season = int(query["season"][0]) if "season" in query else None
                phase = query.get("phase", ["regular"])[0]
                team = query.get("team", ["PIT"])[0]
                model = query.get("model", ["auto"])[0]
                if team not in SUPPORTED_TEAMS:
                    raise ValueError("Choose the Pittsburgh Steelers (PIT) or Minnesota Vikings (MIN).")
                if phase not in {"regular", "all"}:
                    raise ValueError("Choose regular season or regular season + playoffs.")
                if model not in {"auto", "baseline", "elo", "matchup"}:
                    raise ValueError("Choose the backtested default or original Elo model.")
                games, metadata = store.load(refresh)
                with analysis_lock:
                    target = season if season is not None else default_season(games)
                    if target not in {g.season for g in games}:
                        raise ValueError("That season is not available in the schedule.")
                    matchup, pregame, bundle = None, None, None
                    if feature_store:
                        if target not in feature_cache or refresh or time.time() - feature_checked.get(target, 0) > 3600:
                            feature_cache[target] = feature_store.load(target, refresh)
                            feature_checked[target] = time.time()
                            if len(feature_cache) > 4:
                                feature_cache.pop(next(iter(feature_cache)))
                        bundle = feature_cache[target]
                        matchup_key = (metadata["downloaded_at"], target, phase, bundle["digest"])
                        if matchup_key not in matchup_cache:
                            config, _ = select_model(games, target)
                            rating_games = [g for g in games if g.season != target or phase == "all" or g.kind == "REG"]
                            matchup_cache[matchup_key] = evaluate(rating_games, target, config, bundle)
                            if len(matchup_cache) > 8:
                                matchup_cache.pop(next(iter(matchup_cache)))
                        matchup, pregame = matchup_cache[matchup_key]
                    key = (metadata["downloaded_at"], target, phase, team, model, bundle["digest"] if bundle else "")
                    if key not in dashboards:
                        dashboards[key] = build_dashboard(games, target, phase == "all", team=team, model=model,
                                                          matchup=matchup, matchup_pregame=pregame)
                        if len(dashboards) > 8:
                            dashboards.popitem(last=False)
                    result = dict(dashboards[key])
                    if bundle:
                        result["features"] = {"sources": bundle["sources"], "digest": bundle["digest"]}
                    next_game = next((g for g in games if result["next_game"] and g.id == result["next_game"]["id"]), None)
                    if next_game and weather_store:
                        result["weather"] = weather_store.load(next_game, refresh)
                    if next_game and matchup and matchup.report["status"] == "evaluated":
                        config, _ = select_model(games, target)
                        rating_games = [g for g in games if g.season != target or phase == "all" or g.kind == "REG"]
                        ratings = replay_season(rating_games, target, config)[0]
                        p_home = home_probability(ratings.get(team_key(next_game.home), BASE_RATING), ratings.get(team_key(next_game.away), BASE_RATING), next_game.neutral, config)
                        result["matchup"] = next_context(matchup, next_game, p_home, team, bundle,
                                                         query.get("qb", [""])[0], query.get("opponent_qb", [""])[0])
                    if archive:
                        if next_game and not query.get("qb") and not query.get("opponent_qb"):
                            archive.save(next_game, team, model, result["next_game"]["win_probability"],
                                         {"version": VERSION, "model": result["model"], "schedule": metadata,
                                          "feature_sources": bundle["sources"] if bundle else [],
                                          "weather": result.get("weather"),
                                          "matchup": result.get("matchup"), "game": result["next_game"]})
                        result["forward_evaluation"] = archive.report(games, team, target, model)
                result["data"] = metadata
                return self.send(200, result)
            except DataUnavailable as error:
                return self.send(503, {"error": str(error)})
            except ValueError as error:
                return self.send(400, {"error": str(error)})
            except Exception as error:
                print(f"Dashboard error: {error}", flush=True)
                return self.send(500, {"error": "The dashboard could not be calculated. Check the Terminal for details."})

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Steelers & Vikings dashboard · Python standard library only")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Use only saved data")
    parser.add_argument("--data", type=Path, help="Use a local nflverse games.csv")
    parser.add_argument("--check", action="store_true", help="Load data, calculate dashboard, and exit")
    parser.add_argument("--team", choices=SUPPORTED_TEAMS, default="PIT", help="Team for --check (default: PIT)")
    parser.add_argument("--backtest", action="store_true", help="Print historical model evaluation as JSON and exit")
    parser.add_argument("--season", type=int, help="Season for --check or --backtest; settings use only prior seasons")
    parser.add_argument("--model", choices=("auto", "baseline", "elo", "matchup"), default="auto", help="Model for --check")
    args = parser.parse_args()
    store = ScheduleStore(ROOT / ".cache" / "games.csv", args.offline, args.data)
    feature_store = FeatureStore(ROOT / ".cache" / "features", args.offline) if not args.data else None
    archive = ForecastArchive(ROOT / ".cache" / "forecasts.sqlite3") if not args.data else None
    weather_store = WeatherStore(ROOT / ".cache" / "weather", args.offline) if not args.data else None
    if args.check or args.backtest:
        try:
            games, metadata = store.load()
            season = args.season if args.season is not None else default_season(games)
            if season not in {g.season for g in games}:
                raise ValueError("That season is not available in the schedule.")
            config, report = select_model(games, season)
            matchup, pregame = None, None
            if feature_store:
                bundle = feature_store.load(season)
                matchup, pregame = evaluate(games, season, config, bundle)
            if args.backtest:
                print(json.dumps({"season": season, "data": metadata, **report,
                                  "matchup": matchup.report if matchup else None,
                                  "feature_sources": bundle["sources"] if feature_store else []}, indent=2, allow_nan=False))
                return
            dashboard = build_dashboard(games, args.season, team=args.team, model=args.model, matchup=matchup, matchup_pregame=pregame)
        except (DataUnavailable, ValueError) as error:
            parser.exit(1, f"{error}\n")
        s = dashboard["team_stats"]
        print(f"Season {dashboard['season']} · {dashboard['team']['short_name']} {s['wins']}-{s['losses']}-{s['ties']} · Elo {s['rating']} · {len(games):,} games loaded")
        print(f"Data: {metadata['source']} · downloaded {metadata['downloaded_at']}")
        print(f"Model: {dashboard['model']['name']}")
        if metadata["warning"]:
            print(metadata["warning"])
        return
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store, feature_store, archive, weather_store))
    except OSError:
        # Avoid interfering with another app already using the default port.
        if args.port != 8765:
            raise
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store, feature_store, archive, weather_store))
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"\nSteelers & Vikings Dashboard is running at {url}\nKeep this window open. Press Control-C to stop.\n", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
