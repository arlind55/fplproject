#!/usr/bin/env python3
"""Career finishing for every FPL player from Understat -> data/finishing.csv.

For each player this season, from every Premier League match Understat has (2014/15 onwards):

    career_npg, career_npxg, career_mins    non-penalty goals, npxG and minutes before this season
    career_npg90, career_npxg90             the same per 90
    season_npg, season_npxg, season_mins    this season so far, from FPL's own data (Opta xG, penalties out)
    season_npg90, season_npxg90
    npg90_change                            season_npg90 - career_npg90
    finishing                               damped goals per xG before this season (1.0 = average)

Understat's xG has drifted high over the years (league npG per npxG ~1.0 in 2014-18, ~0.84 now), so every
shot's xG is first rescaled so each season's league total matches its goals. `finishing` is then
(npg + K) / (npxg + K) with K = 150 npxG of average finishing, about 1 / (true spread between finishers)^2:
real differences are small (about ±7.5% of xG) next to the noise.

Players are matched by club and name, the same way scripts/penalties.py matches penalty takers.
Understat only has the Premier League here, so a player new to the league has no career rows.
The projection model does not use this table: research/understat/finishing_backtest.md found it adds nothing
to goal forecasts once npxG is in. It is here to read, e.g. to tell a proven poor finisher from a cold streak.

Usage (from repo root):
    python3 scripts/finishing.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from fplvalue.load import load, DATA  # noqa: E402
from fplvalue.project import penalties, strip_pens  # noqa: E402
from penalties import TEAM, norm as _norm  # noqa: E402

UNDERSTAT = ROOT / "research" / "understat" / "data"
OUT = DATA / "finishing.csv"
K = 150.0
PARTICLES = {"de", "da", "do", "dos", "das", "di", "del", "van", "von", "der", "den", "el", "al", "la", "le", "ben", "bin"}


def norm(s):
    # NFKD has no decomposition for ø or ı, so "Ødegaard" would lose its first letter
    return _norm(str(s).replace("ø", "o").replace("Ø", "O").replace("ı", "i"))


def season_label(start_year):
    return f"{start_year}/{str(start_year + 1)[2:]}"


def understat_player_matches():
    """One row per player per match: minutes, non-penalty goals and season-rescaled npxG."""
    sh = pd.read_csv(UNDERSTAT / "shots.csv", usecols=["season", "match_id", "player_id", "xG", "result", "situation"])
    sh = sh[(sh["situation"] != "Penalty") & (sh["result"] != "OwnGoal")]
    sh["goal"] = (sh["result"] == "Goal").astype(int)
    cal = sh.groupby("season")["goal"].sum() / sh.groupby("season")["xG"].sum()
    sh["npxg"] = sh["xG"] * sh["season"].map(cal)
    per = sh.groupby(["match_id", "player_id"]).agg(npg=("goal", "sum"), npxg=("npxg", "sum"))
    app = pd.read_csv(UNDERSTAT / "appearances.csv",
                      usecols=["season", "match_id", "team", "player_id", "player", "minutes"])
    app = app[app["minutes"] > 0].join(per, on=["match_id", "player_id"])
    app[["npg", "npxg"]] = app[["npg", "npxg"]].fillna(0)
    return app


def match_players(snap, app):
    """FPL player id -> Understat player id, by club this season (or last) and name."""
    season = season_label(snap.season_start.year)
    prev = season_label(snap.season_start.year - 1)
    teams = snap.teams.set_index("id")["name"]
    rev = {v: k for k, v in TEAM.items()}
    us = app[app["season"].isin([season, prev])].drop_duplicates(["player_id", "team"])[["player_id", "player", "team"]]
    us_names = {pid: norm(n) for pid, n in us.drop_duplicates("player_id")[["player_id", "player"]].itertuples(index=False)}
    out, fallback = {}, {}
    for p in snap.players.itertuples():
        club = rev.get(teams.get(p.team), teams.get(p.team))
        first, last = norm(p.first_name), norm(f"{p.second_name} {p.web_name}") - PARTICLES
        def best(cands):
            # surname matches count double, so "Rayan Cherki" beats "Rayan Ait Nouri"
            # and a first name alone is never enough ("Harvey Davies" is not "Harvey Elliott")
            scored = sorted(((2 * len(us_names[c] & last) + len(us_names[c] & first), c)
                             for c in cands if us_names[c] & last), reverse=True)
            if scored and scored[0][0] > 0 and (len(scored) == 1 or scored[1][0] < scored[0][0]):
                return scored[0][1]
        hit = best(us.loc[us["team"] == club, "player_id"].unique())
        if hit is None:   # moved club since: a unique full-name match anywhere (or one-name players like Sávio)
            full, web = norm(f"{p.first_name} {p.second_name}"), norm(p.web_name)
            cands = [c for c, n in us_names.items() if (full and full <= n) or n == web]
            if len(cands) == 1:
                fallback[p.id] = cands[0]
        if hit is not None:
            out[p.id] = hit
    taken = set(out.values())
    for pid, uid in fallback.items():
        if uid not in taken and list(fallback.values()).count(uid) == 1:
            out[pid] = uid
    return out


def per90(g, m):
    return (g / m * 90).where(m > 0).round(3)


def build(snap=None, app=None):
    snap = snap or load()
    app = app if app is not None else understat_player_matches()
    season = season_label(snap.season_start.year)
    ids = match_players(snap, app)
    before = app[app["season"] < season].groupby("player_id")[["npg", "npxg", "minutes"]].sum()
    us_name = app.drop_duplicates("player_id").set_index("player_id")["player"]
    # this season from FPL, which is refreshed every run (the committed Understat file is not)
    h = strip_pens(snap.history, season)
    pz = penalties()
    pg = pz[(pz["season"] == season) & pz["scored"].astype(bool)].groupby(["fixture", "element"]).size()
    h["npg"] = h["goals_scored"] - h.join(pg.rename("pg"), on=["fixture", "player_id"])["pg"].fillna(0)
    now = h.groupby("player_id").agg(npg=("npg", "sum"), npxg=("expected_goals", "sum"), minutes=("minutes", "sum"))
    p = snap.players.set_index("id")
    rows = []
    for pid in p.index:
        uid = ids.get(pid)
        b = before.loc[uid] if uid in before.index else pd.Series({"npg": 0, "npxg": 0.0, "minutes": 0})
        n = now.loc[pid] if pid in now.index else pd.Series({"npg": 0, "npxg": 0.0, "minutes": 0})
        rows.append({"id": pid, "code": p.at[pid, "code"], "web_name": p.at[pid, "web_name"],
                     "understat_id": uid, "understat_name": us_name.get(uid),
                     "career_npg": int(b["npg"]), "career_npxg": round(b["npxg"], 2), "career_mins": int(b["minutes"]),
                     "season_npg": int(n["npg"]), "season_npxg": round(n["npxg"], 2), "season_mins": int(n["minutes"])})
    df = pd.DataFrame(rows)
    df["career_npg90"] = per90(df["career_npg"], df["career_mins"])
    df["career_npxg90"] = per90(df["career_npxg"], df["career_mins"])
    df["season_npg90"] = per90(df["season_npg"], df["season_mins"])
    df["season_npxg90"] = per90(df["season_npxg"], df["season_mins"])
    df["npg90_change"] = (df["season_npg90"] - df["career_npg90"]).round(3)
    df["finishing"] = ((df["career_npg"] + K) / (df["career_npxg"] + K)).round(3)
    return df


if __name__ == "__main__":
    df = build()
    df.to_csv(OUT, index=False)
    print("wrote", OUT, len(df), "players,", int(df["understat_id"].notna().sum()), "matched to Understat")
