#!/usr/bin/env python3
"""Numbers and chart images for the GW9 feature posts (João Pedro replacements, team form).
Usage (from repo root): python3 blog/feature_charts.py
Prints the tables used in the posts and writes docs/blog/img/gw9-joao-pedro-replacements.png and
gw9-team-form.png. Same palette as blog/charts.py.

Team form follows the form-signals backtest (Understat 2015/16-2026/27): a long-run, opponent- and
venue-adjusted npxG rating (EWMA, half-life 15 matches, carried across seasons while a team stays up)
predicts the next six matches; last-five results and short xG trends add nothing on top. Shot share and
field tilt (deep-completion share) at the same half-life add a small, consistent extra signal."""
import importlib, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# palette and helpers as in charts.py (importing it would regenerate that script's charts)
BG, LINE, INK, INK2, INK3, ACCENT = "#131521", "#2f3449", "#eef0f7", "#b4b9cc", "#7c8299", "#8b97ff"
OK, ERR = "#5fc48f", "#e57a86"
FWD_C = "#f08a5c"
OUT = ROOT / "docs" / "blog" / "img"; OUT.mkdir(parents=True, exist_ok=True)

def font(name):
    for f in fm.findSystemFonts():
        if name.lower() in Path(f).name.lower(): return fm.FontProperties(fname=f)
    return fm.FontProperties()
DISP = font("DejaVuSansCondensed-Bold"); BODY = font("LiberationSans-Regular"); MONO = font("DejaVuSansMono")

def canvas(): return plt.figure(figsize=(12, 6.3), dpi=100, facecolor=BG)

def frame(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(LINE)
    ax.tick_params(colors=INK3, labelsize=10)
    for lab in ax.get_xticklabels() + ax.get_yticklabels(): lab.set_fontproperties(MONO)
    ax.grid(axis="x", color=LINE, linewidth=.6); ax.set_axisbelow(True)

def header(fig, title, sub):
    fig.text(.045, .93, title, fontproperties=DISP, fontsize=26, color=INK, va="top")
    fig.text(.045, .855, sub, fontproperties=BODY, fontsize=12, color=INK2, va="top")
    fig.text(.955, .04, "fplvalue.co", fontproperties=DISP, fontsize=14, color=ACCENT, ha="right")

def save(fig, name):
    fig.savefig(OUT / f"{name}.png", facecolor=BG, dpi=100); plt.close(fig); print("wrote", name)

import fplvalue as fv  # noqa: E402

DATA = ROOT / "data" / "current"; US = ROOT / "research" / "understat" / "data"
pd.set_option("display.width", 220)

# ── 1. João Pedro replacements ──────────────────────────────────────────────────
P = importlib.import_module("fplvalue.project")
proj = pd.read_csv(DATA / "projections.csv")
snap = fv.load(); last = fv.last_season(snap)
st = P.state_from_snapshot(snap, last)
JP = 165
st.players.loc[st.players.id == JP, ["status", "chance", "news"]] = ["a", 100.0, ""]
st.overrides = pd.DataFrame([dict(id=JP, web_name="João Pedro", p_start=0.95, gw_from=np.nan, gw_to=np.nan, note="", source="", origin="manual", updated="")])
fit = P.project(st, P.Params()).set_index("id")
gws = [c for c in proj.columns if c.startswith("gw_")]
print("João Pedro if fit:", fit.loc[JP, gws].round(2).to_dict(), "next 5:", round(fit.loc[JP, "next_n"], 1))

fwd = proj[(proj.pos == "FWD") & (proj.status == "a")].sort_values("next_n", ascending=False)
print(fwd[["web_name", "team_short_name", "price", "selected_by_percent", "p_start", "xg90", "xa90", "penalties_order", "next_n"] + gws].head(10).round(2).to_string())

top = fwd[fwd.id != 411].head(7)
fig = canvas(); ax = fig.add_axes([.22, .10, .72, .66]); frame(ax)
names = list(top.web_name) + ["João Pedro"]; teams = list(top.team_short_name) + ["CHE"]; prices = list(top.price) + [proj.set_index("id").loc[JP, "price"]]
vals = list(top.next_n) + [proj.set_index("id").loc[JP, "next_n"]]
y = np.arange(len(names))[::-1]
ax.barh(y, vals, color=[FWD_C] * len(top) + [ERR], height=.6, zorder=2)
ax.scatter([fit.loc[JP, "next_n"]], [y[-1]], marker="|", s=420, color=INK, linewidths=2.2, zorder=3, )
for yi, n, tm, pr, v in zip(y, names, teams, prices, vals):
    ax.text(-.02, yi, f"{n}  ", ha="right", va="center", color=INK, fontproperties=BODY, fontsize=12, transform=ax.get_yaxis_transform())
    ax.text(-.02, yi - .32, f"{tm} · £{pr:.1f}m", ha="right", va="center", color=INK3, fontproperties=MONO, fontsize=8.5, transform=ax.get_yaxis_transform())
    ax.text(v + .3, yi, f"{v:.1f}", va="center", color=INK, fontproperties=MONO, fontsize=10.5)
ax.set_yticks([]); ax.set_xlim(0, max(vals + [fit.loc[JP, "next_n"]]) * 1.22)
ax.set_xlabel("Projected points, GW9–13", fontproperties=BODY, color=INK3, fontsize=10)
jf = fit.loc[JP, "next_n"]; ax.text(jf + .3, y[-1], f"{jf:.1f} if fit", va="center", color=INK2, fontproperties=MONO, fontsize=10.5)
header(fig, "Replacing João Pedro", "The model's best forwards for the next five gameweeks, Haaland aside. Red: João Pedro with his injury flag.")
save(fig, "gw9-joao-pedro-replacements")

# ── 2. Team form ────────────────────────────────────────────────────────────────
US_TO_FPL = {"Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "Brentford": "BRE", "Brighton": "BHA", "Chelsea": "CHE",
             "Coventry": "COV", "Crystal Palace": "CRY", "Everton": "EVE", "Fulham": "FUL", "Hull": "HUL", "Ipswich": "IPS", "Leeds": "LEE",
             "Liverpool": "LIV", "Manchester City": "MCI", "Manchester United": "MUN", "Newcastle United": "NEW",
             "Nottingham Forest": "NFO", "Sunderland": "SUN", "Tottenham": "TOT"}
m = pd.read_csv(US / "matches.csv"); m = m[m.played == 1]
d = pd.read_csv(US / "team_matches.csv", parse_dates=["date"])
key = []
for side, opp in (("home", "away"), ("away", "home")):
    k = m[["season", "match_id", "datetime", f"{side}_id", f"{opp}_id"]].copy(); k.columns = ["season", "match_id", "date", "team_id", "opp_id"]; k["venue"] = side[0]; key.append(k)
key = pd.concat(key); key["date"] = pd.to_datetime(key.date)
d = d.merge(key, on=["season", "team_id", "date", "venue"])
s = pd.read_csv(US / "shots.csv", usecols=["match_id", "h_a", "result", "situation"])
s = s[(s.result != "OwnGoal") & (s.situation != "Penalty")]
shots = s.groupby(["match_id", "h_a"]).size().unstack(fill_value=0)
d["shots_f"] = [shots.loc[i, v] if i in shots.index else np.nan for i, v in zip(d.match_id, d.venue)]
d["shots_a"] = [shots.loc[i, "a" if v == "h" else "h"] if i in shots.index else np.nan for i, v in zip(d.match_id, d.venue)]
d = d.sort_values(["team_id", "date"]).reset_index(drop=True)

seasons = sorted(d.season.unique()); nxt = dict(zip(seasons, seasons[1:])); spell = {}
for tid, ss in d.groupby("team_id").season.apply(lambda v: sorted(set(v))).items():
    k = 0
    for i, sn in enumerate(ss):
        k += bool(i and nxt.get(ss[i - 1]) != sn); spell[(tid, sn)] = k
d["spell"] = [f"{t}-{spell[(t, sn)]}" for t, sn in zip(d.team_id, d.season)]
for c in ("npxg", "npxga"):
    d[c + "_v"] = d[c] - d.venue.map(d.groupby("venue")[c].mean()) + d[c].mean()
prior = lambda x, hl: x.groupby(d.spell).transform(lambda v: v.shift().ewm(halflife=hl, min_periods=1).mean())
upto = lambda x, hl: x.groupby(d.spell).transform(lambda v: v.ewm(halflife=hl, min_periods=1).mean())
opp = pd.DataFrame({"match_id": d.match_id, "opp_id": d.team_id, "opp_for": prior(d.npxg_v, 15), "opp_against": prior(d.npxga_v, 15)})
d = d.merge(opp, on=["match_id", "opp_id"], how="left").sort_values(["team_id", "date"]).reset_index(drop=True)
lg = d.npxg.mean()
d["adj_for"] = d.npxg_v - (d.opp_against.fillna(lg) - lg); d["adj_against"] = d.npxga_v - (d.opp_for.fillna(lg) - lg)
d["rating"] = upto(d.adj_for - d.adj_against, 15)
d["shot_share"] = upto(d.shots_f / (d.shots_f + d.shots_a), 15)
d["tilt"] = upto(d.deep / (d.deep + d.deep_allowed).replace(0, np.nan), 15)

rows = []
for sp, g in d.groupby("spell"):
    cur = g[g.season == seasons[-1]]
    if cur.empty: continue
    before = g[g.season != seasons[-1]]
    rows.append(dict(team=US_TO_FPL[cur.team.iloc[-1]], played=len(cur), rating=cur.rating.iloc[-1],
                     start=before.rating.iloc[-1] if len(before) else np.nan, shot_share=cur.shot_share.iloc[-1], tilt=cur.tilt.iloc[-1],
                     pts=cur.pts.sum(), xpts=cur.xpts.sum(), npxgd_pm=(cur.npxg - cur.npxga).mean()))
tf = pd.DataFrame(rows).set_index("team"); tf["change"] = tf.rating - tf.start

# results (all finished league games, including any Understat hasn't caught up with)
fx = pd.read_csv(DATA / "fixtures.csv"); fx = fx[fx.finished == True]; tn = pd.read_csv(DATA / "teams.csv").set_index("id").short_name
res = []
for r in fx.itertuples():
    for tm, gf, ga in ((r.team_h, r.team_h_score, r.team_a_score), (r.team_a, r.team_a_score, r.team_h_score)):
        res.append((tn[tm], r.event, gf, ga, 3 if gf > ga else int(gf == ga)))
res = pd.DataFrame(res, columns=["team", "gw", "gf", "ga", "p"]).sort_values(["team", "gw"])
g5 = res.groupby("team").tail(5).groupby("team")
tf["last5_pts"] = g5.p.sum(); tf["last5"] = g5.p.apply(lambda v: "".join({3: "W", 1: "D", 0: "L"}[x] for x in v))
tf["table_pts"] = res.groupby("team").p.sum()
tf["rank_rating"] = tf.rating.rank(ascending=False).astype(int); tf["rank_last5"] = tf.last5_pts.rank(ascending=False, method="min").astype(int)
print(f"\nUnderstat through {d.date.max():%d %b}; results through GW{int(res.gw.max())}")
print(tf.sort_values("rating", ascending=False).round(2).to_string())

UP, DOWN = ["BOU", "BHA", "NFO"], ["CRY", "NEW", "AVL"]
fig = canvas(); ax = fig.add_axes([.08, .12, .86, .64]); frame(ax); ax.grid(axis="y", color=LINE, linewidth=.6)
ax.axhline(0, color=INK3, linewidth=.8, zorder=1); ax.axvline(tf.last5_pts.median(), color=INK3, linewidth=.8, linestyle=":", zorder=1)
for tm, r in tf.iterrows():
    c = OK if tm in UP else ERR if tm in DOWN else INK3
    ax.scatter(r.last5_pts, r.rating, s=90 if c != INK3 else 36, color=c, zorder=3)
    dy = {"BRE": .035, "LEE": -.035, "TOT": .035, "FUL": -.035}.get(tm, 0)
    ax.text(r.last5_pts + .22, r.rating + dy, tm, va="center", color=c if c != INK3 else INK2, fontproperties=MONO, fontsize=10 if c != INK3 else 8.5, zorder=4)
ax.set_xlabel("League points from the last five matches", fontproperties=BODY, color=INK3, fontsize=10)
ax.set_ylabel("Underlying rating (adj. npxG difference per match)", fontproperties=BODY, color=INK3, fontsize=10)
ax.set_xlim(-.5, 16); lo, hi = tf.rating.min(), tf.rating.max(); ax.set_ylim(lo - .15, hi + .15)
ax.text(.01, .97, "Better than their results", transform=ax.transAxes, color=OK, fontproperties=BODY, fontsize=10, va="top")
ax.text(.99, .03, "Results ahead of the performance", transform=ax.transAxes, color=ERR, fontproperties=BODY, fontsize=10, ha="right")
header(fig, "Form vs the numbers that predict it", "Recent results (x) say little about the next six games. The long-run xG rating (y) says much more. Green: improving. Red: declining.")
save(fig, "gw9-team-form")
