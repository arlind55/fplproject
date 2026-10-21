#!/usr/bin/env python3
"""Pull every Premier League season from Understat (2014/15 onwards).

Understat's pages load their data from JSON endpoints that the site's own
JavaScript calls with an ``X-Requested-With: XMLHttpRequest`` header:

    /getLeagueData/EPL/<season>   -> {"dates": [...], "teams": {...}, "players": [...]}
    /getMatchData/<match_id>      -> {"shots": {"h": [...], "a": [...]}, "rosters": {...}, "tmpl": {...}}

``<season>`` is the starting year, so 2014 means 2014/15.

Raw responses are cached under ``data/raw`` so a rerun only fetches what is
missing, and the script then flattens everything into CSVs under ``data/``:

    matches.csv        one row per fixture: goals, xG, Understat's win/draw/loss forecast
    team_matches.csv   one row per team per match: xG, npxG, PPDA, deep passes, xPts
    players.csv        one row per player per season: goals, xG, assists, xA, xGChain...
    shots.csv          one row per shot (only with --shots; about 4,500 requests)
    appearances.csv    one row per player per match: started, minutes, xG, npxG, xA... (with --shots)

Standard library only. Usage:

    python scrape_understat.py                      # all seasons, no shots
    python scrape_understat.py --shots              # also every shot (slow, ~1 req/s)
    python scrape_understat.py --seasons 2023 2024  # just these seasons
"""

import argparse
import csv
import gzip
import datetime as dt
import http.cookiejar
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://understat.com"
LEAGUE = "EPL"
FIRST_SEASON = 2014
HEADERS = {
    "User-Agent": "Mozilla/5.0 (personal research; xG study)",
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}


class Client:
    def __init__(self, raw_dir, delay):
        self.raw_dir = raw_dir
        self.delay = delay
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )
        self.primed = False
        self.last = 0.0

    def _get(self, url, headers):
        wait = self.delay - (time.monotonic() - self.last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers=headers)
                with self.opener.open(req, timeout=30) as res:
                    self.last = time.monotonic()
                    body = res.read()
                    # Understat gzips responses even when not asked to.
                    if body[:2] == b"\x1f\x8b":
                        body = gzip.decompress(body)
                    return body
            except urllib.error.HTTPError as err:
                if err.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise
            except urllib.error.URLError:
                if attempt == 3:
                    raise
            time.sleep(2 ** (attempt + 1))

    def json(self, path, cache_name, refresh=False):
        cache = self.raw_dir / cache_name
        if cache.exists() and not refresh:
            return json.loads(cache.read_text())
        if not self.primed:
            # The endpoints expect the session cookie the homepage sets.
            self._get(BASE + "/", {"User-Agent": HEADERS["User-Agent"]})
            self.primed = True
        body = self._get(BASE + path, {**HEADERS, "Referer": BASE + "/league/" + LEAGUE})
        data = json.loads(body)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(body)
        return data


def current_season():
    today = dt.date.today()
    return today.year if today.month >= 8 else today.year - 1


def season_label(season):
    return f"{season}/{str(season + 1)[-2:]}"


def num(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    return int(f) if f.is_integer() and "." not in str(value) else f


def flatten_matches(season, dates):
    rows = []
    for m in dates:
        played = bool(m.get("isResult"))
        forecast = m.get("forecast") or {}
        rows.append({
            "season": season_label(season),
            "match_id": num(m["id"]),
            "datetime": m.get("datetime"),
            "played": int(played),
            "home_id": num(m["h"]["id"]),
            "home": m["h"]["title"],
            "away_id": num(m["a"]["id"]),
            "away": m["a"]["title"],
            "home_goals": num(m["goals"]["h"]) if played else None,
            "away_goals": num(m["goals"]["a"]) if played else None,
            "home_xg": num(m["xG"]["h"]) if played else None,
            "away_xg": num(m["xG"]["a"]) if played else None,
            "forecast_home_win": num(forecast.get("w")),
            "forecast_draw": num(forecast.get("d")),
            "forecast_away_win": num(forecast.get("l")),
        })
    return rows


def flatten_team_matches(season, teams):
    rows = []
    for team in teams.values():
        for i, h in enumerate(team.get("history", []), start=1):
            ppda = h.get("ppda") or {}
            ppda_allowed = h.get("ppda_allowed") or {}
            rows.append({
                "season": season_label(season),
                "team_id": num(team["id"]),
                "team": team["title"],
                "match_no": i,
                "date": h.get("date"),
                "venue": h.get("h_a"),
                "goals_for": num(h.get("scored")),
                "goals_against": num(h.get("missed")),
                "xg": num(h.get("xG")),
                "xga": num(h.get("xGA")),
                "npxg": num(h.get("npxG")),
                "npxga": num(h.get("npxGA")),
                "npxgd": num(h.get("npxGD")),
                "ppda_att": num(ppda.get("att")),
                "ppda_def": num(ppda.get("def")),
                "ppda_allowed_att": num(ppda_allowed.get("att")),
                "ppda_allowed_def": num(ppda_allowed.get("def")),
                "deep": num(h.get("deep")),
                "deep_allowed": num(h.get("deep_allowed")),
                "result": h.get("result"),
                "pts": num(h.get("pts")),
                "xpts": num(h.get("xpts")),
            })
    return rows


PLAYER_FIELDS = [
    "id", "player_name", "team_title", "position", "games", "time", "goals", "npg",
    "xG", "npxG", "assists", "xA", "shots", "key_passes", "xGChain", "xGBuildup",
    "yellow_cards", "red_cards",
]


def flatten_players(season, players):
    rows = []
    for p in players:
        row = {"season": season_label(season)}
        row.update({k: num(p.get(k)) for k in PLAYER_FIELDS})
        rows.append(row)
    return rows


SHOT_FIELDS = [
    "id", "match_id", "date", "minute", "h_a", "h_team", "a_team", "player_id", "player",
    "player_assisted", "X", "Y", "xG", "result", "situation", "shotType", "lastAction",
    "h_goals", "a_goals",
]


def flatten_shots(season, shots):
    rows = []
    for side in ("h", "a"):
        for s in shots.get(side, []):
            row = {"season": season_label(season)}
            row.update({k: num(s.get(k)) for k in SHOT_FIELDS})
            rows.append(row)
    return rows


def flatten_appearances(season, match, md):
    """Per player per match, from the match page's line-ups. Starters have a real position;
    substitutes are listed as "Sub" and their `time` is the minutes they actually played."""
    pens = {}
    for side in ("h", "a"):
        for s in md.get("shots", {}).get(side, []):
            if s.get("situation") == "Penalty":
                pens[s["player_id"]] = pens.get(s["player_id"], 0.0) + float(s["xG"])
    rows = []
    for side, team_key in (("h", "home"), ("a", "away")):
        for r in md.get("rosters", {}).get(side, {}).values():
            xg = float(r.get("xG") or 0)
            rows.append({
                "season": season_label(season),
                "match_id": match["match_id"],
                "date": match["datetime"],
                "team_id": match[f"{team_key}_id"],
                "team": match[team_key],
                "h_a": side,
                "player_id": num(r["player_id"]),
                "player": r["player"],
                "position": r["position"],
                "started": int(r["position"] != "Sub"),
                "minutes": num(r.get("time")),
                "goals": num(r.get("goals")),
                "own_goals": num(r.get("own_goals")),
                "shots": num(r.get("shots")),
                "xG": xg,
                "npxG": max(xg - pens.get(r["player_id"], 0.0), 0.0),
                "assists": num(r.get("assists")),
                "xA": num(r.get("xA")),
                "key_passes": num(r.get("key_passes")),
                "xGChain": num(r.get("xGChain")),
                "xGBuildup": num(r.get("xGBuildup")),
                "yellow_card": num(r.get("yellow_card")),
                "red_card": num(r.get("red_card")),
            })
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows):>7} rows  {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", type=int, nargs="*", help="starting years, e.g. 2014 2015 (default: all)")
    parser.add_argument("--shots", action="store_true", help="also fetch every match's shots")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between requests (default 1)")
    args = parser.parse_args()

    seasons = args.seasons or list(range(FIRST_SEASON, current_season() + 1))
    latest = current_season()
    client = Client(args.out / "raw", args.delay)

    matches, team_matches, players, shots, appearances = [], [], [], [], []
    for season in seasons:
        # The season in progress changes week to week, so always refetch it.
        data = client.json(f"/getLeagueData/{LEAGUE}/{season}",
                           f"league_{LEAGUE}_{season}.json", refresh=season == latest)
        season_matches = flatten_matches(season, data["dates"])
        matches += season_matches
        team_matches += flatten_team_matches(season, data["teams"])
        players += flatten_players(season, data["players"])
        print(f"{season_label(season)}: {sum(m['played'] for m in season_matches)} played matches")

        if args.shots:
            played = [m for m in season_matches if m["played"]]
            for i, m in enumerate(played, start=1):
                try:
                    md = client.json(f"/getMatchData/{m['match_id']}", f"match/{m['match_id']}.json")
                except urllib.error.HTTPError as err:
                    print(f"  match {m['match_id']}: HTTP {err.code}, skipped", file=sys.stderr)
                    continue
                shots += flatten_shots(season, md.get("shots", {}))
                appearances += flatten_appearances(season, m, md)
                if i % 50 == 0:
                    print(f"  shots: {i}/{len(played)} matches")

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "matches.csv", matches)
    write_csv(args.out / "team_matches.csv", team_matches)
    write_csv(args.out / "players.csv", players)
    if args.shots:
        write_csv(args.out / "shots.csv", shots)
        write_csv(args.out / "appearances.csv", appearances)


if __name__ == "__main__":
    main()
