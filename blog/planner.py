#!/usr/bin/env python3
"""Numbers and hero chart for the international-break planner post.
Usage (from repo root): python3 blog/planner.py [FIRST_GW] [N_GWS]   (defaults: next GW, 6 weeks)

Fixture swing = how much easier (or harder) a team's next N fixtures are than the average game it has
played so far: model xG for minus its season xG per game, plus its season xGA per game minus model xGA.
Difficulty colours are the model's 1-5 ratings from data/current/fixture_model.csv (the planner's Rating view).
Writes docs/blog/img/gw<FIRST>-break-planner.png and prints the tables used in the post."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data" / "current"
OUT = ROOT / "docs" / "blog" / "img"; OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 240)
BG, LINE, INK, INK2, INK3, ACCENT = "#131521", "#2f3449", "#eef0f7", "#b4b9cc", "#7c8299", "#8b97ff"
DC = {1: ("#1f7a4d", "#ffffff"), 2: ("#3fa66e", "#0f2a1c"), 3: ("#c9ccd6", "#2b2f3d"), 4: ("#d95f6a", "#ffffff"), 5: ("#8f1f33", "#ffffff")}
def font(name):
    for f in fm.findSystemFonts():
        if name.lower() in Path(f).name.lower(): return fm.FontProperties(fname=f)
    return fm.FontProperties()
DISP = font("DejaVuSansCondensed-Bold"); BODY = font("LiberationSans-Regular"); MONO = font("DejaVuSansMono")

meta = json.load(open(DATA / "meta.json"))
FIRST = int(sys.argv[1]) if len(sys.argv) > 1 else int(meta["next_event"])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6
LAST = FIRST + N - 1

tn = pd.read_csv(DATA / "teams.csv").set_index("id").short_name
fm_ = pd.read_csv(DATA / "fixture_model.csv"); fm_ = fm_[(fm_.event >= FIRST) & (fm_.event <= LAST)]
rows = []
for r in fm_.itertuples():
    rows.append((tn[r.team_h], r.event, "H", tn[r.team_a], r.xg_home, r.xg_away, r.cs_prob_home, r.model_diff_home))
    rows.append((tn[r.team_a], r.event, "A", tn[r.team_h], r.xg_away, r.xg_home, r.cs_prob_away, r.model_diff_away))
d = pd.DataFrame(rows, columns=["team", "gw", "v", "opp", "xgf", "xga", "cs", "diff"])
tr = pd.read_csv(DATA / "team_ratings.csv").set_index("short_name")
s = d.groupby("team").agg(n=("gw", "size"), xgf=("xgf", "mean"), xga=("xga", "mean"), cs=("cs", "sum"), diff=("diff", "mean"))
s["att_boost"] = s.xgf - tr.xg_for / tr.games; s["def_boost"] = tr.xg_against / tr.games - s.xga; s["swing"] = s.att_boost + s.def_boost
s = s.sort_values(["diff", "swing"], ascending=[True, False])
print(f"== Fixture swing GW{FIRST}-{LAST} ==\n", s.round(2).to_string())
for tm in s.index:
    cells = "".join(f'<div class="fix d{int(r.diff)}">{r.opp}<small>GW{r.gw} {r.v}</small></div>' for r in d[d.team == tm].sort_values("gw").itertuples())
    print(f'{tm}: <div class="fixrow">{cells}</div>')

# hero: fixture grid sorted by swing
fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=BG)
ax = fig.add_axes([.02, .07, .96, .72]); ax.set_facecolor(BG); ax.axis("off")
teams = list(s.index); cw, ch = 1.0, 1.0
ax.set_xlim(-.9, N + 2.3); ax.set_ylim(len(teams) + .2, -1.1)
for j, gw in enumerate(range(FIRST, LAST + 1)):
    ax.text(j + .5, -.45, f"GW{gw}", ha="center", va="center", color=INK3, fontproperties=MONO, fontsize=9.5)
ax.text(N + .55, -.45, "xG/g", ha="center", va="center", color=INK3, fontproperties=MONO, fontsize=9.5)
ax.text(N + 1.55, -.45, "CS", ha="center", va="center", color=INK3, fontproperties=MONO, fontsize=9.5)
for i, tm in enumerate(teams):
    ax.text(-.25, i + .5, tm, ha="right", va="center", color=INK, fontproperties=MONO, fontsize=10)
    for r in d[d.team == tm].itertuples():
        j = r.gw - FIRST; bg, ink = DC[int(r.diff)]
        ax.add_patch(plt.Rectangle((j + .04, i + .06), cw - .08, ch - .12, color=bg, linewidth=0))
        ax.text(j + .5, i + .52, f"{r.opp} {r.v}", ha="center", va="center", color=ink, fontproperties=MONO, fontsize=8.6)
    ax.text(N + .55, i + .5, f"{s.loc[tm, 'xgf']:.2f}", ha="center", va="center", color=INK2, fontproperties=MONO, fontsize=9.5)
    ax.text(N + 1.55, i + .5, f"{s.loc[tm, 'cs']:.1f}", ha="center", va="center", color=INK2, fontproperties=MONO, fontsize=9.5)
fig.text(.045, .95, f"GW{FIRST}–{LAST}: who has the fixtures", fontproperties=DISP, fontsize=26, color=INK, va="top")
fig.text(.045, .875, "Model difficulty, easiest run at the top. xG/g: model goals per game. CS: expected clean sheets over the six games.",
         fontproperties=BODY, fontsize=11.5, color=INK2, va="top")
fig.text(.955, .03, "fplvalue.co", fontproperties=DISP, fontsize=14, color=ACCENT, ha="right")
fig.savefig(OUT / f"gw{FIRST}-break-planner.png", facecolor=BG, dpi=100); plt.close(fig); print("wrote", f"gw{FIRST}-break-planner")

# players: best over the projection window, by position, plus budget options and big-owned laggards
p = pd.read_csv(DATA / "projections.csv"); gws = [c for c in p.columns if c.startswith("gw_")]
cols = ["web_name", "team_short_name", "price", "selected_by_percent", "status", "next_n"] + gws
for pos in ["GKP", "DEF", "MID", "FWD"]:
    x = p[p.pos == pos]
    print(f"\n{pos} top:\n", x.nlargest(8, "next_n")[cols].round(1).to_string())
    print(f"{pos} budget:\n", x[x.price <= {"GKP": 4.6, "DEF": 5.1, "MID": 6.0, "FWD": 6.0}[pos]].nlargest(5, "next_n")[cols].round(1).to_string())
p["rank_pos"] = p.groupby("pos").next_n.rank(ascending=False)
print("\nMost owned, with positional rank:\n", p.nlargest(25, "selected_by_percent")[cols + ["rank_pos"]].round(1).to_string())
print(f"\nGW{FIRST} captain:\n", p.nlargest(8, f"gw_{FIRST}")[["web_name", "team_short_name", "next_opp", f"gw_{FIRST}"]].to_string())
pl = pd.read_csv(DATA / "players.csv")
print("\nFlags:\n", pl[pl.status != "a"].nlargest(15, "selected_by_percent")[["web_name", "status", "chance_of_playing_next_round", "news", "selected_by_percent"]].to_string())
