#!/usr/bin/env python3
"""Generate the chart image for each blog post from data/current.
Usage (from repo root): python3 blog/charts.py
Writes docs/blog/img/<post-slug>.png at 1200x630 (doubles as the post's social card)."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "current"
OUT = ROOT / "docs" / "blog" / "img"; OUT.mkdir(parents=True, exist_ok=True)

# ── Site palette (dark theme) ──────────────────────────────────────────────────
BG, SURFACE, LINE = "#131521", "#1b1e2c", "#2f3449"
INK, INK2, INK3, ACCENT = "#eef0f7", "#b4b9cc", "#7c8299", "#8b97ff"
OK, ERR, WARN = "#5fc48f", "#e57a86", "#e0a24a"
POSC = {"GKP": "#e0b23a", "DEF": "#6fa3e6", "MID": "#5fc48f", "FWD": "#f08a5c"}

def font(name):
    for f in fm.findSystemFonts():
        if name.lower() in Path(f).name.lower(): return fm.FontProperties(fname=f)
    return fm.FontProperties()
DISP = font("DejaVuSansCondensed-Bold"); BODY = font("LiberationSans-Regular"); MONO = font("DejaVuSansMono")
if not fm.findSystemFonts(): BODY = DISP

def canvas():
    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=BG)
    return fig

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

# ── Data ───────────────────────────────────────────────────────────────────────
POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}; G = {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}; CS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
p = pd.read_csv(DATA / "players.csv"); h = pd.read_csv(DATA / "player_history.csv"); t = pd.read_csv(DATA / "teams.csv").set_index("id")
p["pos"] = p.element_type.map(POS)
h = h.merge(p[["id", "pos"]], left_on="player_id", right_on="id", suffixes=("", "_p"))
def xp(r):
    g = G[r.pos]; cs = CS[r.pos]; gk = r.pos in ("GKP", "DEF")
    act = r.goals_scored * g + r.assists * 3 + r.clean_sheets * cs - (r.goals_conceded // 2 if gk else 0)
    exp = r.expected_goals * g + r.expected_assists * 3 + (np.exp(-r.expected_goals_conceded) * cs if r.minutes >= 60 else 0) - (r.expected_goals_conceded / 2 if gk else 0)
    return r.total_points - act + exp
h["xpts"] = h.apply(xp, axis=1)
agg = h.groupby("player_id").agg(games=("minutes", lambda s: (s > 0).sum()), mins=("minutes", "sum"), pts=("total_points", "sum"), xpts=("xpts", "sum")).reset_index()
d = p[["id", "web_name", "team_short_name", "pos", "now_cost", "selected_by_percent"]].merge(agg, left_on="id", right_on="player_id")
d["ppg"] = d.pts / d.games; d["vapm"] = (d.ppg - 2) / d.now_cost; d["xvapm"] = (d.xpts / d.games - 2) / d.now_cost; d["delta"] = d.pts - d.xpts

# ── 1. Value report: VAPM top 8, actual vs expected ─────────────────────────────
top = d[d.mins >= 90].sort_values("vapm", ascending=False).head(8).iloc[::-1]
fig = canvas(); ax = fig.add_axes([.20, .10, .74, .66]); frame(ax)
y = np.arange(len(top))
ax.barh(y, top.vapm, color=[POSC[x] for x in top.pos], height=.6, zorder=2)
ax.scatter(top.xvapm, y, marker="|", s=420, color=INK, linewidths=2.2, zorder=3, label="VAPM on expected points")
for yi, (n, tm, pr, v, xv) in enumerate(zip(top.web_name, top.team_short_name, top.now_cost, top.vapm, top.xvapm)):
    ax.text(-.02, yi, f"{n}  ", ha="right", va="center", color=INK, fontproperties=BODY, fontsize=12, transform=ax.get_yaxis_transform())
    ax.text(-.02, yi - .32, f"{tm} · £{pr:.1f}m", ha="right", va="center", color=INK3, fontproperties=MONO, fontsize=8.5, transform=ax.get_yaxis_transform())
    ax.text(max(v, xv) + .04, yi, f"{v:.2f}", va="center", color=INK, fontproperties=MONO, fontsize=10.5)
ax.set_yticks([]); ax.set_xlim(0, top.vapm.max() * 1.18)
ax.set_xlabel("VAPM = (points per game − 2) ÷ price", fontproperties=BODY, color=INK3, fontsize=10)
leg = ax.legend(loc="lower right", frameon=False, prop=BODY, labelcolor=INK2, fontsize=10)
header(fig, "Top VAPM after two gameweeks", "Bars: actual VAPM (90+ minutes). Tick: where VAPM would sit on expected points — the gap is finishing and clean-sheet luck.")
save(fig, "gw3-value-report")

# ── 2. Template check: points vs xPts for ≥25% owned ────────────────────────────
tpl = d[d.selected_by_percent >= 25].sort_values("selected_by_percent", ascending=False)
fig = canvas(); ax = fig.add_axes([.06, .16, .90, .60]); frame(ax); ax.grid(axis="y", color=LINE, linewidth=.6); ax.grid(axis="x", visible=False)
x = np.arange(len(tpl)); w = .38
ax.bar(x - w / 2, tpl.pts, w, color=ACCENT, label="Points", zorder=2)
ax.bar(x + w / 2, tpl.xpts, w, color=WARN, label="Expected points (xPts)", zorder=2)
for xi, (n, tm, own, dl) in enumerate(zip(tpl.web_name, tpl.team_short_name, tpl.selected_by_percent, tpl.delta)):
    ax.text(xi, -1.2, n, ha="center", va="top", color=INK, fontproperties=BODY, fontsize=10, rotation=0)
    ax.text(xi, -3.6, f"{own:.0f}% owned", ha="center", va="top", color=INK3, fontproperties=MONO, fontsize=8)
    ax.text(xi, max(tpl.pts.iloc[xi], tpl.xpts.iloc[xi]) + .5, f"{dl:+.1f}", ha="center", color=OK if dl > 0 else ERR, fontproperties=MONO, fontsize=9.5)
ax.set_xticks([]); ax.set_ylim(0, max(tpl.pts.max(), tpl.xpts.max()) * 1.15)
ax.legend(loc="upper right", frameon=False, prop=BODY, labelcolor=INK2, fontsize=10)
header(fig, "The template vs its expected points", "Every player owned by 25%+ of managers. Green/red: points minus xPts. Arsenal players have one match in this dataset.")
save(fig, "gw3-template-check")

# ── 3. Fixture swing: slope chart GW3–5 → GW6–10 ─────────────────────────────────
f = pd.read_csv(DATA / "fixtures.csv", usecols=["event", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"]).dropna(subset=["event"])
rows = []
for r in f.itertuples():
    rows.append((r.team_h, int(r.event), r.team_h_difficulty)); rows.append((r.team_a, int(r.event), r.team_a_difficulty))
x = pd.DataFrame(rows, columns=["team", "gw", "d"])
pre = x[x.gw.between(3, 5)].groupby("team").d.mean(); post = x[x.gw.between(6, 10)].groupby("team").d.mean()
sw = pd.DataFrame({"pre": pre, "post": post}); sw["swing"] = sw.post - sw.pre; sw["name"] = [t.loc[i, "short_name"] for i in sw.index]
fig = canvas(); ax = fig.add_axes([.20, .10, .60, .66]); frame(ax); ax.grid(False)
better = sw.sort_values("swing").head(3).index; worse = sw.sort_values("swing").tail(3).index
hl = list(better) + list(worse)
for i, r in sw.iterrows():
    c = OK if i in better else ERR if i in worse else LINE; lw = 3 if c != LINE else 1.2; z = 3 if c != LINE else 1
    ax.plot([0, 1], [r.pre, r.post], color=c, linewidth=lw, zorder=z, solid_capstyle="round")
    ax.scatter([0, 1], [r.pre, r.post], color=c, s=28 if c != LINE else 10, zorder=z + 1)
# merged labels: one label per distinct value at each end, listing the highlighted teams at that value
for col, xpos, ha, fmtr in (("pre", -.04, "right", lambda v, names: f"{'/'.join(names)}  {v:.2f}"), ("post", 1.04, "left", lambda v, names: f"{v:.2f}  {'/'.join(names)}")):
    grp = sw.loc[hl].groupby(sw.loc[hl, col].round(2))
    for v, g in grp:
        cols = [OK if i in better else ERR for i in g.index]
        ax.text(xpos, v, fmtr(v, list(g["name"])), ha=ha, va="center", color=cols[0] if len(set(cols)) == 1 else INK, fontproperties=MONO, fontsize=10.5)
ax.set_xlim(-.05, 1.05); ax.set_ylim(4.15, 2.15)
ax.set_xticks([0, 1]); ax.set_xticklabels(["GW3–5", "GW6–10"])
for lab in ax.get_xticklabels(): lab.set_fontproperties(DISP); lab.set_fontsize(13); lab.set_color(INK2)
ax.set_yticks([]); ax.spines["bottom"].set_visible(False); ax.spines["left"].set_visible(False); ax.tick_params(length=0)
fig.text(.045, .40, "Average FPL difficulty\n(lower = easier)", fontproperties=BODY, color=INK3, fontsize=10, va="center")
for v in (2.5, 3.0, 3.5, 4.0):
    ax.axhline(v, color=LINE, linewidth=.5, zorder=0); ax.text(.5, v - .02, f"{v:.1f}", ha="center", va="bottom", color=INK3, fontproperties=MONO, fontsize=8.5)
header(fig, "Whose fixtures turn after the October break", "Average difficulty of the three matches before the break vs the five after. Green: biggest improvement. Red: biggest deterioration.")
save(fig, "gw6-fixture-swing")
