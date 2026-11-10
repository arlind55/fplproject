#!/usr/bin/env python3
"""Numbers and charts for the season round-up post (first N gameweeks, this season vs last).
Usage (from repo root): python3 blog/roundup.py [GW]    (defaults to the latest finished GW)

Sources
- This season: data/current/player_history.csv (FPL per-player per-match, incl. FPL xG and DefCon).
- Last two seasons: vaastav/Fantasy-Premier-League merged_gw.csv (same FPL fields), downloaded to
  research/vaastav/cache/ on first run.
- Shot situations (set pieces, penalties, distance): research/understat/data/shots.csv, compared over
  the same number of matches from the start of each season (Understat lags FPL by a week or two).
Prints every number used in the post and writes docs/blog/img/<slug>.png (hero) plus three inline charts."""
import sys, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "current"; US = ROOT / "research" / "understat" / "data"
CACHE = ROOT / "research" / "vaastav" / "cache"; CACHE.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "docs" / "blog" / "img"; OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 240)

BG, LINE, INK, INK2, INK3, ACCENT = "#131521", "#2f3449", "#eef0f7", "#b4b9cc", "#7c8299", "#8b97ff"
OK, ERR, WARN = "#5fc48f", "#e57a86", "#e0a24a"
POSC = {"GK": "#e0b23a", "DEF": "#6fa3e6", "MID": "#5fc48f", "FWD": "#f08a5c"}

def font(name):
    for f in fm.findSystemFonts():
        if name.lower() in Path(f).name.lower(): return fm.FontProperties(fname=f)
    return fm.FontProperties()
DISP = font("DejaVuSansCondensed-Bold"); BODY = font("LiberationSans-Regular"); MONO = font("DejaVuSansMono")
def canvas(): return plt.figure(figsize=(12, 6.3), dpi=100, facecolor=BG)
def frame(ax, grid="x"):
    ax.set_facecolor(BG)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(LINE)
    ax.tick_params(colors=INK3, labelsize=10)
    for lab in ax.get_xticklabels() + ax.get_yticklabels(): lab.set_fontproperties(MONO)
    if grid: ax.grid(axis=grid, color=LINE, linewidth=.6)
    ax.set_axisbelow(True)
def header(fig, title, sub):
    fig.text(.045, .93, title, fontproperties=DISP, fontsize=26, color=INK, va="top")
    fig.text(.045, .855, sub, fontproperties=BODY, fontsize=12, color=INK2, va="top")
    fig.text(.955, .04, "fplvalue.co", fontproperties=DISP, fontsize=14, color=ACCENT, ha="right")
def save(fig, name):
    fig.savefig(OUT / f"{name}.png", facecolor=BG, dpi=100); plt.close(fig); print("wrote", name)

events = pd.read_csv(DATA / "events.csv")
GW = int(sys.argv[1]) if len(sys.argv) > 1 else int(events.query("finished == True").id.max())
SLUG = "first-ten-gameweeks-review"   # fixed slug: the post keeps its URL when the numbers are refreshed
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# ── load ────────────────────────────────────────────────────────────────────────
def current():
    h = pd.read_csv(DATA / "player_history.csv"); p = pd.read_csv(DATA / "players.csv").set_index("id")
    fx = pd.read_csv(DATA / "fixtures.csv").set_index("id"); tn = pd.read_csv(DATA / "teams.csv").set_index("id")
    h = h[h["round"] <= GW].copy()
    h["pos"] = h.element.map(p.element_type).map(POS); h["name"] = h.element.map(p.web_name)
    h["team"] = np.where(h.was_home, h.fixture.map(fx.team_h), h.fixture.map(fx.team_a))
    h["team"] = h.team.map(tn.short_name); h["opp"] = h.opponent_team.map(tn.short_name)
    h["GW"] = h["round"]; h["season"] = "2026/27"
    return h

def vaastav(season):
    out = {}
    for f, path in (("mgw", "gws/merged_gw.csv"), ("teams", "teams.csv")):
        dst = CACHE / f"{season}_{f}.csv"
        if not dst.exists():
            urllib.request.urlretrieve(f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/{path}", dst)
        out[f] = pd.read_csv(dst)
    d = out["mgw"]; d = d[d.GW <= GW].copy(); tn = out["teams"].set_index("id").short_name
    d["pos"] = d.position.replace({"GKP": "GK", "AM": "MID"}); d["opp"] = d.opponent_team.map(tn)
    d["team"] = d.team.map(out["teams"].set_index("name").short_name)
    d["season"] = season.replace("-", "/"); return d

S = {"2026/27": current(), "2025/26": vaastav("2025-26"), "2024/25": vaastav("2024-25")}
THIS, LAST = "2026/27", "2025/26"
R = {}  # every number quoted in the post

# ── 1. how the game has changed (FPL data, same GWs) ────────────────────────────
for s, h in S.items():
    m = h.groupby("fixture").agg(hs=("team_h_score", "first"), as_=("team_a_score", "first"))
    n = len(m); pl = h[h.minutes > 0]
    r = dict(matches=n, goals_pg=(m.hs + m.as_).sum() / n, xg_pg=h.expected_goals.sum() / n,
             home_win=(m.hs > m.as_).mean(), draw=(m.hs == m.as_).mean(), away_win=(m.hs < m.as_).mean(),
             nil_nil=((m.hs == 0) & (m.as_ == 0)).mean(), cs_rate=((m.hs == 0).sum() + (m.as_ == 0).sum()) / (2 * n),
             yellows_pg=pl.yellow_cards.sum() / n, reds=int(pl.red_cards.sum()), own_goals=int(pl.own_goals.sum()))
    if "defensive_contribution" in pl:
        thr = np.where(pl.pos == "DEF", 10, 12)
        dcp = np.where((pl.pos != "GK") & (pl.defensive_contribution >= thr), 2, 0)
        r["defcon_pts"] = int(dcp.sum()); r["defcon_share"] = dcp.sum() / pl.total_points.sum()
    st = pl[pl.starts == 1]
    for pos in POS.values():
        x = st[st.pos == pos]
        r[f"{pos}_pps"] = x.total_points.mean(); r[f"{pos}_haul"] = (x.total_points >= 10).mean(); r[f"{pos}_blank"] = (x.total_points <= 2).mean()
    tot = pl.groupby("pos").total_points.sum(); r.update({f"{k}_share": v / tot.sum() for k, v in tot.items()})
    R[s] = r
game = pd.DataFrame(R); print(f"\n== Game-wide, GW1-{GW} ==\n", game.round(3).to_string())

# Understat: shot situations over the first K matches of each season
sh = pd.read_csv(US / "shots.csv"); um = pd.read_csv(US / "matches.csv"); um = um[um.played == 1].copy(); um["dt"] = pd.to_datetime(um.datetime)
K = int((um.season == THIS).sum()); us_rows = {}
for s in ["2021/22", "2022/23", "2023/24", "2024/25", "2025/26", "2026/27"]:
    ids = um[um.season == s].sort_values("dt").head(K)
    x = sh[sh.match_id.isin(ids.match_id)]; g = x[x.result == "Goal"]; sp = ["SetPiece", "FromCorner", "DirectFreekick"]
    us_rows[s] = dict(shots_pg=len(x) / K, setpiece_goal_share=g.situation.isin(sp).mean(), corner_goals_pg=(g.situation == "FromCorner").sum() / K,
                      pens_pg=(x.situation == "Penalty").sum() / K, header_goal_share=(g.shotType == "Head").mean(),
                      outside_box_shot_share=(x.X < .83).mean(), outside_box_goal_share=(g.X < .83).mean(),
                      home_win=(ids.home_goals > ids.away_goals).mean(), draw=(ids.home_goals == ids.away_goals).mean())
us = pd.DataFrame(us_rows); print(f"\n== Understat, first {K} matches of each season (through {um[um.season == THIS].dt.max():%d %b}) ==\n", us.round(3).to_string())
full = um.groupby("season").apply(lambda d: pd.Series(dict(home_win=(d.home_goals > d.away_goals).mean(), draw=(d.home_goals == d.away_goals).mean())))
print("\nHome win / draw by full season:\n", full.round(3).T.to_string())

# ── 2. teams: table vs underlying, this season vs last ──────────────────────────
def table(h):
    g = h.groupby(["fixture", "team", "opp", "was_home"]).agg(xg=("expected_goals", "sum"), hs=("team_h_score", "first"), as_=("team_a_score", "first")).reset_index()
    g["gf"] = np.where(g.was_home, g.hs, g.as_); g["ga"] = np.where(g.was_home, g.as_, g.hs)
    g = g.merge(g[["fixture", "team", "xg"]].rename(columns={"team": "opp", "xg": "xga"}), on=["fixture", "opp"])
    g["pts"] = np.select([g.gf > g.ga, g.gf == g.ga], [3, 1], 0)
    t = g.groupby("team").agg(P=("pts", "size"), pts=("pts", "sum"), gf=("gf", "sum"), ga=("ga", "sum"), xg=("xg", "sum"), xga=("xga", "sum"), cs=("ga", lambda v: (v == 0).sum()))
    t["gd"] = t.gf - t.ga; t["xgd_pg"] = (t.xg - t.xga) / t.P
    return t.sort_values(["pts", "gd", "gf"], ascending=False)
T = table(S[THIS]); L_same = table(S[LAST])
lf_raw = pd.read_csv(CACHE / "2025-26_mgw.csv"); lf_raw["team"] = lf_raw.team.map(pd.read_csv(CACHE / "2025-26_teams.csv").set_index("name").short_name)
lf_raw["opp"] = lf_raw.opponent_team.map(pd.read_csv(CACHE / "2025-26_teams.csv").set_index("id").short_name); L_full = table(lf_raw)
T["pos"] = range(1, len(T) + 1); T["xgd_rank"] = T.xgd_pg.rank(ascending=False).astype(int)
T["pts_last_same"] = L_same.pts; T["xgd_last_full"] = L_full.xgd_pg; T["xgd_change"] = T.xgd_pg - T.xgd_last_full
T["finishing"] = T.gf - T.xg; T["keeping"] = T.xga - T.ga
L_full["pos"] = range(1, 21); T["pos_last"] = L_full.pos
print(f"\n== Table after GW{GW} with last season ==\n", T.round(2).to_string())
print("\nLast season same stage:\n", L_same[["pts", "gd", "xgd_pg"]].round(2).T.to_string())

# ── 3. players and prices ───────────────────────────────────────────────────────
def players(h):
    g = h.groupby(["element", "name", "pos"]).agg(team=("team", "last"), pts=("total_points", "sum"), mins=("minutes", "sum"), price=("value", "first"),
                                                  g=("goals_scored", "sum"), a=("assists", "sum"), xg=("expected_goals", "sum"), xa=("expected_assists", "sum"),
                                                  bonus=("bonus", "sum"), sel=("selected", "last")).reset_index()
    g["price"] /= 10; g["ppm"] = g.pts / g.price
    g["band"] = pd.cut(g.price, [0, 5, 6.5, 8.5, 10.5, 20], labels=["£5.0m or less", "£5.1–6.5m", "£6.6–8.5m", "£8.6–10.5m", "Over £10.5m"])
    return g
P = {s: players(h) for s, h in S.items()}
pm = pd.read_csv(DATA / "players.csv").set_index("id"); P[THIS]["own"] = P[THIS].element.map(pm.selected_by_percent); P[THIS]["now"] = P[THIS].element.map(pm.now_cost)
bands = {}
for s, g in P.items():
    reg = g[g.mins >= 45 * GW]
    bands[s] = reg.groupby("band", observed=True).ppm.median()
    top = g.nlargest(20, "pts"); R[s]["top20_def_gk"] = int(top.pos.isin(["DEF", "GK"]).sum()); R[s]["top20_le55"] = int((top.price <= 5.5).sum())
    R[s]["n_over_10"] = int((g.price >= 10).sum())
    print(f"\n{s} top 12:", top.head(12)[["name", "pos", "price", "pts"]].values.tolist(), "| DEF/GK in top 20:", R[s]["top20_def_gk"], "| ≤£5.5m in top 20:", R[s]["top20_le55"])
bands = pd.DataFrame(bands); print("\nMedian points per £m by starting price (regulars, half the minutes or more):\n", bands.round(2).to_string())
c = P[THIS]
print("\nPriciest at the start:\n", c.nlargest(12, "price")[["name", "team", "pos", "price", "now", "pts", "mins", "g", "a", "xg", "xa", "own"]].round(1).to_string())
print("\nMost owned now:\n", c.nlargest(15, "own")[["name", "team", "pos", "price", "now", "pts", "mins", "g", "a", "xg", "xa", "own"]].round(1).to_string())
c["gmx"] = c.g - c.xg
print("\nFinishing above xG:\n", c.nlargest(8, "gmx")[["name", "team", "g", "xg", "own"]].round(1).to_string())
print("\nFinishing below xG:\n", c.nsmallest(8, "gmx")[["name", "team", "g", "xg", "own"]].round(1).to_string())
c["xgi"] = c.xg + c.xa
print("\nMost xGI:\n", c.nlargest(10, "xgi")[["name", "team", "pos", "price", "pts", "g", "a", "xg", "xa", "own"]].round(1).to_string())

# captaincy: Haaland vs hindsight best
h = S[THIS]; wk = h.groupby(["GW", "name"]).total_points.sum().reset_index()
cap = wk.loc[wk.groupby("GW").total_points.idxmax()].set_index("GW")
cap["haaland"] = wk[wk.name == "Haaland"].set_index("GW").total_points
print("\nCaptaincy, Haaland vs best:\n", cap.to_string())
hl = S[LAST]; hl = hl[hl.name.str.contains("Haaland")].groupby("GW").total_points.sum(); print("Haaland last season same GWs:", hl.tolist(), hl.sum())
ev = events[events.id <= GW][["id", "average_entry_score", "highest_score"]]; print("\nAverage / highest manager score:\n", ev.T.to_string())

# model track record
import json
acc = json.load(open(DATA / "accuracy.json")); print("\nTrack record:", acc["summary"])

# ── charts ──────────────────────────────────────────────────────────────────────
# hero: each team's xG difference per match last season (hollow) vs this season (filled)
t = T.sort_values("xgd_pg"); fig = canvas(); ax = fig.add_axes([.1, .1, .84, .68]); frame(ax, grid="x")
for i, (tm, r) in enumerate(t.iterrows()):
    last = r.xgd_last_full; now = r.xgd_pg
    col = OK if (pd.notna(last) and now - last >= .3) else ERR if (pd.notna(last) and now - last <= -.3) else INK2
    if pd.notna(last):
        ax.plot([last, now], [i, i], color=col, linewidth=2, alpha=.7, zorder=2)
        ax.scatter(last, i, s=46, facecolor=BG, edgecolor=INK3, linewidth=1.4, zorder=3)
    ax.scatter(now, i, s=70, color=col, zorder=4)
    ax.text(-1.32, i, f"{tm}", va="center", ha="left", color=INK, fontproperties=MONO, fontsize=10)
    ax.text(-1.07, i, f"{int(r.pos)}{'th' if 10 <= r.pos % 100 <= 20 else {1:'st',2:'nd',3:'rd'}.get(r.pos % 10,'th')}", va="center", ha="left", color=INK3, fontproperties=MONO, fontsize=9)
ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.axvline(0, color=INK3, linewidth=.8, zorder=1)
ax.set_xlim(-1.35, 1.25); ax.set_ylim(-.8, len(t) - .2)
ax.set_xlabel("xG difference per match (FPL xG).  Hollow: last season, full 38.  Filled: this season so far.  Left: team and table position.", fontproperties=BODY, color=INK3, fontsize=10)
header(fig, f"Ten weeks in: who's really changed", "Green: underlying numbers up 0.3+ goals a game on last season. Red: down 0.3+. Promoted sides have no hollow dot.")
save(fig, SLUG)

# game-wide changes: small multiples across seasons (Understat, first K matches)
seasons = list(us.columns); lab = [s[2:4] + "/" + s[-2:] for s in seasons]
panels = [("home_win", "Home wins", "{:.0%}"), ("draw", "Draws", "{:.0%}"), ("setpiece_goal_share", "Goals from set pieces", "{:.0%}"),
          ("pens_pg", "Penalties per match", "{:.2f}"), ("shots_pg", "Shots per match", "{:.1f}"), ("outside_box_shot_share", "Shots from outside the box", "{:.0%}")]
fig = canvas()
for k, (key, ttl, fmt) in enumerate(panels):
    ax = fig.add_axes([.06 + (k % 3) * .315, .47 - (k // 3) * .38, .26, .25]); frame(ax, grid=None)
    v = us.loc[key].values; cols = [ACCENT if s == THIS else LINE for s in seasons]
    ax.bar(range(len(v)), v, color=cols, width=.7)
    for j, val in enumerate(v): ax.text(j, val, fmt.format(val), ha="center", va="bottom", color=INK if seasons[j] == THIS else INK3, fontproperties=MONO, fontsize=8.5)
    ax.set_xticks(range(len(v))); ax.set_xticklabels(lab, fontproperties=MONO, fontsize=8.5, color=INK3); ax.set_yticks([])
    ax.spines["left"].set_visible(False); ax.set_ylim(0, max(v) * 1.25)
    ax.set_title(ttl, fontproperties=BODY, fontsize=12, color=INK2, loc="left")
header(fig, "How the game has changed", f"The first {K} matches of each of the last six seasons (Understat). This season in blue.")
save(fig, "roundup-game")

# value by price band
fig = canvas(); ax = fig.add_axes([.07, .14, .88, .6]); frame(ax, grid="y")
x = np.arange(len(bands)); w = .26
for j, (s, col) in enumerate((("2024/25", "#4a5070"), (LAST, INK3), (THIS, ACCENT))):
    ax.bar(x + (j - 1) * w, bands[s].values, width=w, color=col, label=s)
    for xi, val in zip(x, bands[s].values): ax.text(xi + (j - 1) * w, val + .05, f"{val:.1f}", ha="center", color=INK if s == THIS else INK3, fontproperties=MONO, fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(bands.index, fontproperties=BODY, fontsize=11, color=INK2)
ax.set_ylabel("Median points per £m", fontproperties=BODY, color=INK3)
leg = ax.legend(frameon=False, loc="upper right", prop=BODY, labelcolor=INK2)
header(fig, "Where the points per pound are", f"Median points per £m of starting price after GW{GW}, regular starters only. The cheap end has paid best for two seasons running.")
save(fig, "roundup-value")

json.dump({k: {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv) for kk, vv in v.items()} for k, v in R.items()},
          open(ROOT / "research" / "vaastav" / "cache" / f"roundup_gw{GW}.json", "w"), indent=1, default=float)
