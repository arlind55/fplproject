#!/usr/bin/env python3
"""Match Understat's penalty shots to FPL players and fixtures -> data/penalties.csv.

FPL's API has no penalty flag on goals or xG, so the projection model takes penalties from Understat's
shot data (research/understat/data/shots.csv, situation == "Penalty"). Each penalty is matched to an FPL
fixture by date and team, then to the FPL player on that team who played in it, by name.

Usage (from repo root):
    python3 scripts/penalties.py            # from the committed shots.csv
    python3 scripts/penalties.py --fetch    # first re-fetch this season's shots from Understat (~1 request per match)
The weekly refresh (scripts/update_fpl_data.py) runs refresh(), i.e. --fetch, before writing projections.
"""
import sys, unicodedata, re, subprocess, tempfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fplvalue.load import load, last_season, DATA  # noqa: E402

SHOTS = ROOT / "research" / "understat" / "data" / "shots.csv"
OUT = DATA / "penalties.csv"

# Understat team name -> FPL team name, where they differ
TEAM = {"Coventry": "Coventry City", "Hull": "Hull City", "Ipswich": "Ipswich Town", "Manchester City": "Man City",
        "Manchester United": "Man Utd", "Newcastle United": "Newcastle", "Nottingham Forest": "Nott'm Forest",
        "Tottenham": "Spurs", "Wolverhampton Wanderers": "Wolves", "Leicester": "Leicester City"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return set(re.findall(r"[a-z]+", s))


def match_season(snap, shots):
    season = f"{snap.season_start.year}/{str(snap.season_start.year + 1)[2:]}"
    pens = shots[(shots["season"] == season) & (shots["situation"] == "Penalty")].copy()
    if not len(pens):
        return pd.DataFrame(), [], season
    teams = snap.teams.set_index("name")
    fx = snap.fixtures.dropna(subset=["kickoff_time"]).copy()
    fx["day"] = pd.to_datetime(fx["kickoff_time"], utc=True).dt.date
    h = snap.history[snap.history["minutes"] > 0]
    pl = snap.players.set_index("id")
    rows, misses = [], []
    for s in pens.itertuples():
        team_name = TEAM.get(s.h_team if s.h_a == "h" else s.a_team, s.h_team if s.h_a == "h" else s.a_team)
        if team_name not in teams.index:
            misses.append((season, s.date, s.player, "team")); continue
        tid = teams.loc[team_name, "id"]
        day = pd.Timestamp(s.date).date()
        f = fx[((fx["team_h"] == tid) | (fx["team_a"] == tid)) & (abs(pd.to_datetime(fx["day"]) - pd.Timestamp(day)) <= pd.Timedelta(days=1))]
        if len(f) != 1:
            misses.append((season, s.date, s.player, "fixture")); continue
        fid = int(f["id"].iloc[0])
        cand = h[(h["fixture"] == fid) & (h["team"] == tid)]["player_id"].unique()
        target = norm(s.player)
        def score(pid):
            r = pl.loc[pid]
            names = norm(f"{r.get('first_name', '')} {r.get('second_name', '')} {r.get('web_name', '')}")
            return len(target & names)
        scored = sorted(((score(p), p) for p in cand), reverse=True)
        if not scored or scored[0][0] == 0 or (len(scored) > 1 and scored[1][0] == scored[0][0]):
            misses.append((season, s.date, s.player, "player")); continue
        pid = scored[0][1]
        rows.append({"season": season, "kickoff_time": s.date, "fixture": fid, "team_code": int(teams.loc[team_name, "code"]),
                     "element": int(pid), "player_code": int(pl.loc[pid, "code"]), "web_name": pl.loc[pid, "web_name"],
                     "understat_name": s.player, "scored": s.result == "Goal", "xg": round(float(s.xG), 4)})
    return pd.DataFrame(rows), misses, season


def fetch_current() -> pd.DataFrame:
    """This season's shots, fetched fresh from Understat into a temp folder (the committed file may be stale)."""
    scraper = SHOTS.parents[1] / "scrape_understat.py"
    sys.path.insert(0, str(scraper.parent))
    import scrape_understat
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(scraper), "--seasons", str(scrape_understat.current_season()), "--shots",
                        "--out", tmp], check=True)
        return pd.read_csv(Path(tmp) / "shots.csv")


def refresh(fetch: bool = True):
    cols = ["season", "date", "h_a", "h_team", "a_team", "player", "xG", "result", "situation"]
    shots = pd.read_csv(SHOTS, usecols=cols)
    if fetch:
        new = fetch_current()[cols]
        shots = pd.concat([shots[~shots["season"].isin(set(new["season"]))], new], ignore_index=True)
    cur = load(); last = last_season(cur)
    out, all_miss = [], []
    for snap in [s for s in (last, cur) if s is not None]:
        df, miss, season = match_season(snap, shots)
        print(f"{season}: {len(df)} penalties matched, {len(miss)} unmatched")
        out.append(df); all_miss += miss
    for m in all_miss:
        print("  unmatched:", *m)
    res = pd.concat(out, ignore_index=True)
    res.to_csv(OUT, index=False)
    print("wrote", OUT, len(res), "rows")


if __name__ == "__main__":
    refresh(fetch="--fetch" in sys.argv)
