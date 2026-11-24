#!/usr/bin/env python3
"""Numbers and hero charts for the post-gameweek team-form and transfer-targets posts.
Usage (from repo root): python3 blog/review.py [GW]   (GW = the gameweek just played; defaults to meta current_event)

Team form uses FPL match xG (summed from player_history.csv), opponent-adjusted by subtracting each
opponent's season-average xG conceded / created relative to the league average. "Earlier" is every match
before the last five. "Upturn" is the fixture model's expected points for the next six (Poisson on
fixture_model.csv xG) minus points from the last six: the form-shifts study found that gap is what
predicts visible turnarounds (AUC 0.83), while recent results on their own add nothing.

Targets use projections.csv (next five gameweeks, injury flags applied).
Writes docs/blog/img/gw<GW+1>-team-form.png and gw<GW+1>-targets.png and prints the tables used in the posts."""
import json, sys
from math import exp, factorial
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data" / "current"
OUT = ROOT / "docs" / "blog" / "img"; OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
BG, LINE, INK, INK2, INK3, ACCENT = "#131521", "#2f3449", "#eef0f7", "#b4b9cc", "#7c8299", "#8b97ff"
OK, ERR = "#5fc48f", "#e57a86"
POSC = {"GKP": "#e0b23a", "DEF": "#6fa3e6", "MID": "#5fc48f", "FWD": "#f08a5c"}
def font(name):
    for f in fm.findSystemFonts():
        if name.lower() in Path(f).name.lower(): return fm.FontProperties(fname=f)
    return fm.FontProperties()
DISP = font("DejaVuSansCondensed-Bold"); BODY = font("LiberationSans-Regular"); MONO = font("DejaVuSansMono")
def canvas(): return plt.figure(figsize=(12, 6.3), dpi=100, facecolor=BG)
def header(fig, title, sub):
    fig.text(.045, .93, title, fontproperties=DISP, fontsize=26, color=INK, va="top")
    fig.text(.045, .855, sub, fontproperties=BODY, fontsize=12, color=INK2, va="top")
    fig.text(.955, .04, "fplvalue.co", fontproperties=DISP, fontsize=14, color=ACCENT, ha="right")
def save(fig, name):
    fig.savefig(OUT / f"{name}.png", facecolor=BG, dpi=100); plt.close(fig); print("wrote", name)

meta = json.load(open(DATA / "meta.json"))
GW = int(sys.argv[1]) if len(sys.argv) > 1 else int(meta["current_event"]); NXT = GW + 1
tn = pd.read_csv(DATA / "teams.csv").set_index("id").short_name

# ── Team match xG from player history ───────────────────────────────────────────
fx = pd.read_csv(DATA / "fixtures.csv"); fx = fx[(fx.finished == True) & (fx.event <= GW)]
h = pd.read_csv(DATA / "player_history.csv").merge(fx[["id", "team_h", "team_a"]], left_on="fixture", right_on="id")
h["team"] = np.where(h.was_home, h.team_h, h.team_a)
tx = h.groupby(["fixture", "team"]).expected_goals.sum()
rows = []
for r in fx.itertuples():
    xh, xa = tx.get((r.id, r.team_h), 0.0), tx.get((r.id, r.team_a), 0.0)
    for tm, op, gf, ga, xf, xg_a in ((r.team_h, r.team_a, r.team_h_score, r.team_a_score, xh, xa), (r.team_a, r.team_h, r.team_a_score, r.team_h_score, xa, xh)):
        rows.append(dict(team=tn[tm], opp=tn[op], gw=int(r.event), gf=int(gf), ga=int(ga), xgf=xf, xga=xg_a, pts=3 if gf > ga else int(gf == ga)))
m = pd.DataFrame(rows).sort_values(["team", "gw"]); lg = m.xgf.mean()
avg = m.groupby("team").agg(of=("xgf", "mean"), oa=("xga", "mean"))
m["adj_f"] = m.xgf - (m.opp.map(avg.oa) - lg); m["adj_a"] = m.xga - (m.opp.map(avg.of) - lg); m["adj"] = m.adj_f - m.adj_a
g = m.groupby("team"); early = lambda col: g.apply(lambda x: x.head(len(x) - 5)[col].mean()); late = lambda col: g.apply(lambda x: x.tail(5)[col].mean())
s = pd.DataFrame({"P": g.size(), "pts": g.pts.sum(), "gd": g.gf.sum() - g.ga.sum(), "rating": g.adj.mean(),
                  "early": early("adj"), "late": late("adj"), "late_f": late("adj_f"), "late_a": late("adj_a"), "early_f": early("adj_f"), "early_a": early("adj_a"),
                  "pts_early": g.apply(lambda x: x.head(len(x) - 5).pts.sum()), "pts_l5": g.apply(lambda x: x.tail(5).pts.sum()),
                  "pts_l6": g.apply(lambda x: x.tail(6).pts.sum()),
                  "form": g.apply(lambda x: "".join(x.tail(5).pts.map({3: "W", 1: "D", 0: "L"})))})
s["change"] = s.late - s.early
s["table"] = (s.pts + s.gd / 100).rank(ascending=False, method="first").astype(int); s["rank_rating"] = s.rating.rank(ascending=False).astype(int)

def pmf(lam): return np.array([exp(-lam) * lam ** k / factorial(k) for k in range(10)])
def exp_pts(a, b):
    M = np.outer(pmf(a), pmf(b)); return 3 * np.tril(M, -1).sum() + np.trace(M)
fm_ = pd.read_csv(DATA / "fixture_model.csv"); fm_ = fm_[(fm_.event >= NXT) & (fm_.event < NXT + 6)]
E = {}
for r in fm_.itertuples():
    E.setdefault(tn[r.team_h], []).append(exp_pts(r.xg_home, r.xg_away)); E.setdefault(tn[r.team_a], []).append(exp_pts(r.xg_away, r.xg_home))
s["exp_next6"] = pd.Series({k: sum(v) for k, v in E.items()}); s["upturn"] = s.exp_next6 - s.pts_l6
print(f"== Team form after GW{GW} (earlier = GW1–{GW-5}, last five = GW{GW-4}–{GW}) ==\n", s.sort_values("change", ascending=False).round(2).to_string())

order = s.sort_values("late").index
up, down = list(s.change.nlargest(3).index), list(s.change.nsmallest(3).index)
fig = canvas(); ax = fig.add_axes([.10, .09, .84, .70]); ax.set_facecolor(BG)
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(LINE); ax.tick_params(colors=INK3, labelsize=9.5)
for lab in ax.get_xticklabels(): lab.set_fontproperties(MONO)
ax.grid(axis="x", color=LINE, linewidth=.6); ax.set_axisbelow(True); ax.axvline(0, color=INK3, linewidth=.8)
for i, tm in enumerate(order):
    a, b = s.loc[tm, "early"], s.loc[tm, "late"]; c = OK if tm in up else ERR if tm in down else INK3
    ax.plot([a, b], [i, i], color=c, linewidth=2.6 if c != INK3 else 1.4, solid_capstyle="round", zorder=2)
    ax.scatter([a], [i], s=26, color=BG, edgecolors=c, linewidths=1.4, zorder=3); ax.scatter([b], [i], s=46 if c != INK3 else 26, color=c, zorder=4)
    ax.text(-.01, i, tm, transform=ax.get_yaxis_transform(), ha="right", va="center", color=INK if c != INK3 else INK2, fontproperties=MONO, fontsize=9.5)
ax.set_yticks([]); ax.set_ylim(-.7, len(order) - .3)
ax.set_xlabel(f"Opponent-adjusted xG difference per match.  Hollow: GW1–{GW-5}.  Solid: GW{GW-4}–{GW}.", fontproperties=BODY, color=INK3, fontsize=10)
header(fig, f"Who has changed since GW{GW-5}", f"Every team's underlying numbers, earlier games vs the last five. Green: biggest improvements. Red: biggest falls.")
save(fig, f"gw{NXT}-team-form")

# ── Player form: xGI per 90 over the last four vs earlier ───────────────────────
POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
pl = pd.read_csv(DATA / "players.csv"); pl["pos"] = pl.element_type.map(POS)
proj = pd.read_csv(DATA / "projections.csv"); gws = [c for c in proj.columns if c.startswith("gw_")]
ph = pd.read_csv(DATA / "player_history.csv"); ph["late"] = ph["round"] > GW - 4
a = ph.groupby(["player_id", "late"]).agg(m=("minutes", "sum"), xgi=("expected_goal_involvements", "sum"), pts=("total_points", "sum"), st=("starts", "sum")).unstack()
pf = pd.DataFrame({"m_e": a.m[False], "m_l": a.m[True], "xgi90_e": a.xgi[False] / a.m[False] * 90, "xgi90_l": a.xgi[True] / a.m[True] * 90,
                   "pts_l": a.pts[True], "st_e": a.st[False], "st_l": a.st[True]}).reset_index()
pf = pf.merge(pl[["id", "web_name", "team_short_name", "pos", "now_cost", "selected_by_percent"]], left_on="player_id", right_on="id").merge(proj[["id", "next_n"]], on="id", how="left")
pf["d"] = pf.xgi90_l - pf.xgi90_e
q = pf[(pf.m_e >= 300) & (pf.m_l >= 270) & pf.pos.isin(["MID", "FWD"])]
cols = ["web_name", "team_short_name", "pos", "now_cost", "selected_by_percent", "xgi90_e", "xgi90_l", "d", "pts_l", "next_n"]
print(f"\n== xGI/90 risers, GW{GW-3}–{GW} vs earlier ==\n", q.nlargest(12, "d")[cols].round(2).to_string())
print("\n== xGI/90 fallers ==\n", q[q.xgi90_e >= .3].nsmallest(10, "d")[cols].round(2).to_string())
print("\n== New starters ==\n", pf[(pf.st_l >= 3) & (pf.st_e <= 2)].nlargest(10, "next_n")[cols + ["st_e", "st_l"]].round(2).to_string())

# ── Transfer targets: next five gameweeks ───────────────────────────────────────
pc = ["web_name", "team_short_name", "pos", "price", "selected_by_percent", "status", "next_opp", "next_n"] + gws
proj["per_m"] = proj.next_n / proj.price
for pos in ["GKP", "DEF", "MID", "FWD"]:
    x = proj[proj.pos == pos]
    print(f"\n{pos} top:\n", x.nlargest(8, "next_n")[pc].round(1).to_string())
    print(f"{pos} under 10% owned:\n", x[x.selected_by_percent < 10].nlargest(5, "next_n")[pc].round(1).to_string())
proj["rank_pos"] = proj.groupby("pos").next_n.rank(ascending=False)
print("\nMost owned, positional rank:\n", proj.nlargest(25, "selected_by_percent")[pc + ["rank_pos"]].round(1).to_string())
print(f"\nGW{NXT} captain:\n", proj.nlargest(8, f"gw_{NXT}")[["web_name", "team_short_name", "next_opp", f"gw_{NXT}", "selected_by_percent"]].round(2).to_string())

TARGETS = ["Haaland", "B.Fernandes", "Gabriel", "Groß", "Tavernier", "Gibbs-White", "Calvert-Lewin", "E.Le Fée", "Barry", "Hill", "Justin", "Kinsky"]
t = proj[proj.web_name.isin(TARGETS)].sort_values("next_n").reset_index(drop=True)
fig = canvas(); ax = fig.add_axes([.22, .09, .72, .70]); ax.set_facecolor(BG)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"): ax.spines[sp].set_color(LINE)
ax.tick_params(colors=INK3, labelsize=10); ax.grid(axis="x", color=LINE, linewidth=.6); ax.set_axisbelow(True)
for lab in ax.get_xticklabels(): lab.set_fontproperties(MONO)
y = np.arange(len(t))
ax.barh(y, t.next_n, color=[POSC[p] for p in t.pos], height=.62, zorder=2)
for yi, r in t.iterrows():
    ax.text(-.02, yi + .08, f"{r.web_name}  ", ha="right", va="center", color=INK, fontproperties=BODY, fontsize=11.5, transform=ax.get_yaxis_transform())
    ax.text(-.02, yi - .3, f"{r.team_short_name} · £{r.price:.1f}m · {r.selected_by_percent:.0f}%  ", ha="right", va="center", color=INK3, fontproperties=MONO, fontsize=8, transform=ax.get_yaxis_transform())
    ax.text(r.next_n + .3, yi, f"{r.next_n:.1f}", va="center", color=INK, fontproperties=MONO, fontsize=10)
ax.set_yticks([]); ax.set_xlim(0, t.next_n.max() * 1.12); ax.set_ylim(-.6, len(t) - .4)
ax.set_xlabel(f"Projected points, GW{NXT}–{NXT + len(gws) - 1}", fontproperties=BODY, color=INK3, fontsize=10)
for i, (pos, c) in enumerate(POSC.items()):
    fig.text(.70 + i * .065, .80, pos, color=c, fontproperties=MONO, fontsize=10)
header(fig, f"Who to target for GW{NXT}–{NXT + len(gws) - 1}", "The model's best buys at each price, with price and ownership. Bars coloured by position.")
save(fig, f"gw{NXT}-targets")
