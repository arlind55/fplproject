#!/usr/bin/env python3
"""Numbers, chart and post for the wildcard piece (best £100m squads over the next six gameweeks).
Usage (from repo root): python3 blog/wildcard.py      then   python3 blog/build.py blog/shell.css docs

Everything comes from data/current/projections_long.csv (the published per-gameweek projections) and
fplvalue.optimise (the same integer programme as the site's optimiser). Squads maximise the XI's discounted
GW11–16 projection (0.9 per week, as the site does) plus 0.15 × bench, under £100.0m at current prices.
"Total" in the post = the XI's undiscounted GW11–16 projection + the best captain each week (the XI's top
projection that gameweek). Prints every number used and writes blog/posts/<SLUG>.html and docs/blog/img/<SLUG>.png."""
import datetime, html, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from fplvalue.optimise import optimise

DATA = ROOT / "data" / "current"; OUT = ROOT / "docs" / "blog" / "img"; POSTS = ROOT / "blog" / "posts"
pd.set_option("display.width", 240)
esc = html.escape

events = pd.read_csv(DATA / "events.csv").set_index("id")
meta = json.loads((DATA / "meta.json").read_text())
FIRST = int(meta["next_event"]); H = 6
GWS = list(range(FIRST, FIRST + H)); C = [f"gw_{g}" for g in GWS]
SLUG = f"gw{FIRST}-wildcard"; LABEL = f"GW{GWS[0]}–{GWS[-1]}"
BUDGET = 100.0

L = pd.read_csv(DATA / "projections_long.csv")
L["v"] = sum(L[c] * 0.9 ** j for j, c in enumerate(C))
L["next_gw"] = L[C[0]]
L["tot"] = L[C].sum(axis=1)

def total(sq):
    xi = sq[sq.is_starter]
    return float(xi.tot.sum() + xi[C].max().sum())

def solve(pool=L, budget=BUDGET, value="v", **kw):
    sq, info = optimise(pool, budget, value_col=value, **kw)
    sq = L.set_index("id").loc[sq.id].reset_index().assign(is_starter=sq.is_starter.values, is_captain=sq.is_captain.values)
    return sq, total(sq)

def xi_own(sq): return float(sq[sq.is_starter].selected_by_percent.mean())

# ── squads ──────────────────────────────────────────────────────────────────────
A, tA = solve()
HAAL = int(L.loc[L.web_name == "Haaland", "id"].iloc[0])
B, tB = solve(exclude=[HAAL])
D_PEN = 0.04   # points of discounted value given up per 1% ownership: enough to change ~a third of the squad
P = L.assign(v=L.v - D_PEN * L.selected_by_percent)
D, tD = solve(pool=P)
T_ = L.assign(v=L.selected_by_percent); Tsq, _ = optimise(T_, BUDGET, value_col="v")   # most-owned legal squad
T, tT = solve(pool=L[L.id.isin(Tsq.id)])                                                 # its best XI by projection

for n, sq, t in [("model", A, tA), ("no Haaland", B, tB), ("differential", D, tD), ("template", T, tT)]:
    print(f"\n{n}: £{sq.price.sum():.1f}m  total {t:.1f}  XI own {xi_own(sq):.1f}%")
    print(sq[["web_name", "team_short_name", "pos", "price", "selected_by_percent", "status", "p_start", "tot", "is_starter"]].round(2).to_string(index=False))

# which picks matter: drop each starter, re-solve, points lost
loo = []
for r in A[A.is_starter].itertuples():
    sq, t = solve(exclude=[r.id])
    ins = [x for x in sq.web_name if x not in set(A.web_name)]
    loo.append((r, tA - t, ins))
loo.sort(key=lambda x: -x[1])
print("\nleave-one-out:"); [print(f"  {r.web_name:14s} {d:5.1f}  in: {ins}") for r, d, ins in loo]

# horizons: who survives from 3 to 10 weeks
hz = {}
for h in (3, 5, 6, 8, 10):
    cc = [f"gw_{g}" for g in range(FIRST, FIRST + h)]
    pool = L.assign(v=sum(L[c] * 0.9 ** j for j, c in enumerate(cc)))
    sq, _ = optimise(pool, BUDGET, value_col="v"); hz[h] = set(sq.id)
core = set.intersection(*hz.values()); print("\ncore over all horizons:", sorted(L.set_index("id").loc[list(core)].web_name))

# budgets
bud = {}
for b in (98.0, 102.0):
    sq, t = solve(budget=b); bud[b] = (sq, t)
    print(f"budget {b}: total {t:.1f}  in {sorted(set(sq.web_name) - set(A.web_name))}  out {sorted(set(A.web_name) - set(sq.web_name))}")

caps = A[A.is_starter].set_index("web_name")[C].idxmax().value_counts()
print("captains:", dict(A[A.is_starter].set_index("web_name")[C].idxmax(axis=0).items()))

# ── fixtures (model's blended 1–5 difficulty) ───────────────────────────────────
fmod = pd.read_csv(DATA / "fixture_model.csv"); teams = pd.read_csv(DATA / "teams.csv").set_index("id")
short = teams.short_name; tid = {v: k for k, v in short.items()}
runs = {}
for r in fmod[fmod.event.between(GWS[0], GWS[-1])].sort_values(["event", "kickoff_time"]).itertuples():
    runs.setdefault(r.team_h, []).append((int(r.event), short[r.team_a], "H", int(r.blend_diff_home)))
    runs.setdefault(r.team_a, []).append((int(r.event), short[r.team_h], "A", int(r.blend_diff_away)))
def strip(ts):
    cells = "".join(f'<span class="fix d{d}" title="GW{gw} {o} ({ha}) · difficulty {d}">{o}<small>{ha}</small></span>' for gw, o, ha, d in runs.get(tid[ts], []))
    return f'<span class="fixrow mini">{cells}</span>'

# ── HTML pieces ─────────────────────────────────────────────────────────────────
ORDER = ["GKP", "DEF", "MID", "FWD"]
def own(x): return f"{x:.1f}%" if x < 10 else f"{x:.0f}%"
def chip(r):
    return f'<div class="chip"><b>{esc(r.web_name)}</b><span>{r.team_short_name} · £{r.price:.1f}m</span><em>{r.tot:.1f}</em></div>'
def pitch(sq, title):
    xi, bench = sq[sq.is_starter], sq[~sq.is_starter]
    lines = "".join(f'<div class="line">{"".join(chip(r) for r in xi[xi.pos == p].sort_values("tot", ascending=False).itertuples())}</div>' for p in ORDER)
    shape = "-".join(str((xi.pos == p).sum()) for p in ORDER[1:])
    return (f'<h4 class="sq">{title} <small>{shape} · £{sq.price.sum():.1f}m · {total(sq):.0f} pts</small></h4>'
            f'<div class="pitch">{lines}</div>'
            f'<div class="benchrow"><span class="lbl">Bench</span>{"".join(chip(r) for r in bench.sort_values(["pos", "tot"], key=lambda c: c.map(ORDER.index) if c.name == "pos" else -c).itertuples())}</div>')
def table(sq):
    sq = sq.assign(o=sq.pos.map(ORDER.index)).sort_values(["is_starter", "o", "tot"], ascending=[False, True, False])
    rows = []
    for r in sq.itertuples():
        b = "" if r.is_starter else ' <span class="tag">bench</span>'
        rows.append(f'<tr><td><strong>{esc(r.web_name)}</strong>{b}<span class="sub">{r.pos} · {r.team_short_name} · £{r.price:.1f}m · {own(r.selected_by_percent)}</span></td>'
                    f'<td class="n"><strong>{r.tot:.1f}</strong></td><td>{strip(r.team_short_name)}</td></tr>')
    return (f'<div class="tw"><table class="t tgt wc"><thead><tr><th>Player</th><th class="n">{LABEL}</th><th>Fixtures (model difficulty)</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')

def names(sq, ids_not_in):
    return [r for r in sq.itertuples() if r.web_name not in set(ids_not_in.web_name)]

dl = datetime.datetime.fromisoformat(events.loc[FIRST, "deadline_time"].replace("Z", "+00:00"))
dl_txt = f"{dl:%A} {dl.day} {dl:%B} at {dl:%H:%M}"
gen = json.loads((DATA / "projection_meta.json").read_text())["generated_at"]
gen_dt = datetime.datetime.fromisoformat(gen.replace("Z", "+00:00"))
today = datetime.date.today().isoformat()
r = lambda x: f"{x:.0f}"

h = A.set_index("web_name"); b_in = names(B, A); b_out = names(A, B); d_in = names(D, A); d_out = names(A, D)
loo_rows = "".join(
    f'<tr{" class=hl" if i < 3 else ""}><td><strong>{esc(rr.web_name)}</strong><span class="sub">{rr.pos} · {rr.team_short_name} · £{rr.price:.1f}m</span></td>'
    f'<td class="n{" down" if d >= 3 else ""}">{max(d, 0):.1f}</td><td>{esc(", ".join(ins)) or "—"}</td></tr>'
    for i, (rr, d, ins) in enumerate(loo))
cheap = [rr.web_name for rr, d, _ in loo if d < 1]
t_extra = [x for x in T.web_name if x not in set(A.web_name)]
cap_weeks = A[A.is_starter].set_index("web_name")[C].idxmax(axis=0)
haal_caps = int((cap_weeks == "Haaland").sum())
other_cap = [f"{n} in GW{c[3:]}" for c, n in cap_weeks.items() if n != "Haaland"]
b02_in = sorted(set(bud[102.0][0].web_name) - set(A.web_name)); b02_out = sorted(set(A.web_name) - set(bud[102.0][0].web_name))
b98_in = sorted(set(bud[98.0][0].web_name) - set(A.web_name)); b98_out = sorted(set(A.web_name) - set(bud[98.0][0].web_name))
jp = T[T.web_name == "João Pedro"]

body = f"""<div class="article">
<article class="prose">
<div class="kicker"><span>Wildcard</span><time datetime="{today}">{datetime.date.today():%-d %B %Y} · planning for {LABEL}</time></div>
<h1>The best GW{FIRST} wildcard teams</h1>
<p class="lede">The international break is the most popular time of the season to wildcard. There are eleven days until the GW{FIRST} deadline ({dl_txt} UK time), injuries have had time to settle, and the fixture list is clear through to Christmas. We asked the site's projection model, refreshed with GW10's results, for the best £100m squad over the next six gameweeks ({LABEL}). Then we built two alternatives: one without Haaland, and one that avoids the template.</p>

<h3>The model's wildcard</h3>
<p>This is the best £{BUDGET:.0f}m squad that the model can find for {LABEL}, picked by the same optimiser as the one on the <a href="../projections.html">projections page</a>. The numbers on each shirt are the player's projected points over the six weeks. The squad's total is <strong>{tA:.0f} points</strong>, counting the best captain each week.</p>
{pitch(A, "Squad A: the model's pick")}
<p>The shape is the one the model has liked all season. There's one premium, <strong>Haaland</strong>, and the rest of the money goes into players around £5–8m who start every week and have good fixtures. <strong>Gabriel</strong> (£8.1m) is the most expensive defender in the game, and the model still thinks he's worth it. At {h.loc["Gabriel", "tot"]:.1f} points he projects higher than every midfielder. Arsenal and Brighton provide three players each, and Bournemouth and Leeds two.</p>
<p>The bench is cheap but useful. <strong>Silva</strong> ({h.loc["Silva", "tot"]:.1f}) and <strong>Calvert-Lewin</strong> ({h.loc["Calvert-Lewin", "tot"]:.1f}) would both start in most squads, which matters when a starter gets injured or rotated. Haaland is the captain in {haal_caps} of the six weeks{(", with " + " and ".join(other_cap) + (" the exceptions" if len(other_cap) > 1 else " the exception")) if other_cap else ""}.</p>
{table(A)}

<h3>Which picks matter</h3>
<p>A wildcard squad from an optimiser looks precise, but a lot of these picks are close calls. To see which ones aren't, we took each starter out in turn, let the optimiser rebuild the best squad without him, and measured how many points it lost over the six weeks.</p>
<div class="tw"><table class="t wc"><thead><tr><th>Without…</th><th class="n">Points lost</th><th>What comes in</th></tr></thead><tbody>
{loo_rows}
</tbody></table></div>
<p>Only two players are worth more than a couple of points over six weeks. Losing <strong>{esc(loo[0][0].web_name)}</strong> costs {loo[0][1]:.0f} points, most of it captaincy, and <strong>{esc(loo[1][0].web_name)}</strong> costs {loo[1][1]:.0f}, because nobody else at £6.3m projects close to him. <strong>{esc(loo[2][0].web_name)}</strong> and <strong>{esc(loo[3][0].web_name)}</strong> are next at about two points each. Everyone else has a replacement within about a point. For {", ".join(f"<strong>{esc(n)}</strong>" for n in cheap[:-1])} and <strong>{esc(cheap[-1])}</strong> the gap is under a point. That's about a sixth of a point a week, well inside the model's margin of error. So if you'd rather have a different keeper or a different budget defender, go ahead. The core you want is Haaland and Groß, then Tavernier and Gabriel.</p>

<h3>Squad B: without Haaland</h3>
<p>Haaland has 68 points this season, against 85 at the same stage last year, and plenty of managers are wondering about life without him. The model's answer is that it costs less than you might think. Its best squad without him totals <strong>{tB:.0f}</strong>, {tA - tB:.0f} points behind Squad A over six weeks, or about {(tA - tB) / H:.1f} a week. Most of that is the captaincy.</p>
{pitch(B, "Squad B: no Haaland")}
<p>His £15.8m goes on <strong>{esc(b_in[0].web_name)}</strong> (£{b_in[0].price:.1f}m) and <strong>{esc(b_in[1].web_name)}</strong> (£{b_in[1].price:.1f}m){", with " + " and ".join(f"<strong>{esc(x.web_name)}</strong>" for x in b_in[2:]) + " also coming in" if len(b_in) > 2 else ""}. Fernandes becomes the main captain. The model has him at {B.set_index("web_name").loc["B.Fernandes", "tot"]:.1f} points, the highest of any midfielder, because he takes United's penalties and the model expects him to start (he missed GW8 completely). If you go this way, you are taking a real risk. Haaland is in {own(h.loc["Haaland", "selected_by_percent"])} of squads, so every big Haaland week costs you rank. On these numbers, though, it's a reasonable risk, not a reckless one.</p>

<h3>Squad C: the differential</h3>
<p>If you're chasing rank, you want points from players few others own. For this squad we told the optimiser to treat each percentage point of ownership as a small cost. Its starting XI averages <strong>{xi_own(D):.0f}%</strong> ownership, against {xi_own(A):.0f}% for Squad A, and it projects <strong>{tD:.0f}</strong>, only {tA - tD:.0f} points less.</p>
{pitch(D, "Squad C: differential")}
<p>It keeps the picks that matter. Haaland, Groß, Tavernier and Gabriel are too good to give up for a few points of ownership. The changes are in the places where Squad A's picks were close calls. Leeds keeper <strong>Trafford</strong> ({own(D.set_index("web_name").loc["Trafford", "selected_by_percent"])} owned) starts instead of Raya ({own(h.loc["Raya", "selected_by_percent"])}). Bournemouth's <strong>Hill</strong> ({own(D.set_index("web_name").loc["Hill", "selected_by_percent"])}) and <strong>Silva</strong> ({own(h.loc["Silva", "selected_by_percent"])}) start in defence instead of De Cuyper, who is in nearly half of all squads and projects only about a point more. <strong>Isak</strong> comes in up front for the money saved, with Ødegaard and Calvert-Lewin dropping out. Because so many of the picks in Squad A were close calls, the differential squad costs very little.</p>

<h3>What the template is missing</h3>
<p>For comparison, we built the most-owned legal squad under £100m and let the model pick its best XI. It projects <strong>{tT:.0f}</strong>, which is {tA - tT:.0f} points behind Squad A over six weeks. Two of its bench spots are wasted on players who won't play: {"" if jp.empty else f"<strong>João Pedro</strong> (still injured, {jp.tot.iloc[0]:.1f} projected points) and "}<strong>Hughes</strong> (a 4% chance of starting). <strong>Cherki</strong> is only a 68% starter, and <strong>Rogers</strong>, the most-owned midfielder at 54%, projects below Le Fée, who costs £2.5m less. If your current squad looks like the template, those are the places to start.</p>

<h3>How robust is it?</h3>
<p>We re-ran the optimiser over five horizons, from the next three gameweeks to the next ten. <strong>{len(core)} of the 15 players</strong> are picked every time: {", ".join(esc(n) for n in sorted(L.set_index("id").loc[list(core)].web_name))}. So it doesn't matter much whether you plan for a short run or through to the new year.</p>
<p>The budget doesn't change much either. With £2m more ({", ".join(b02_in)} in, {", ".join(b02_out)} out) the squad gains just {bud[102.0][1] - tA:.0f} point{"" if round(bud[102.0][1] - tA) == 1 else "s"}. With £2m less ({", ".join(b98_in)} in, {", ".join(b98_out)} out) it loses {tA - bud[98.0][1]:.0f}. If your team value is £98–102m, you can build almost the same squad.</p>

<div class="callout"><b>The call.</b> If you wildcard, start from Haaland and Groß, add Tavernier and Gabriel, then fill the squad with £5–7m players who start every week and have good fixtures. Squad A's are from Arsenal, Brighton, Bournemouth, Spurs, Sunderland and Everton. Squad A is the highest-scoring option. Squad C gives up about {tA - tD:.0f} points for a much less common team. Without Haaland (Squad B) the cost is about {(tA - tB) / H:.0f} points a week. Don't confirm your wildcard until after the international games, because injuries on international duty are the main thing that could change these picks.</div>

<p>For fixture runs and the best transfers if you're not wildcarding, see the <a href="./gw11-break-planner.html">GW11 break planner</a>.</p>

<p class="note">Method: projections from the site's <a href="../metrics.html#model">model</a>, refreshed with GW10's results on {gen_dt.day} November at {gen_dt:%H:%M} UTC. Squads are picked by the site's optimiser: 2 GK, 5 DEF, 5 MID, 3 FWD, at most three per club, £{BUDGET:.1f}m at current prices. It maximises the starting XI's projection over {LABEL} (later weeks count slightly less) plus a small weight on the bench. Totals are the XI's projected points plus the best captain each week. The differential squad subtracts {D_PEN} points per 1% ownership from each player's six-week value. Prices and ownership as of the refresh; prices will move before the deadline. A player's projected points over six weeks typically miss the real total by several points either way, so treat gaps of a point or two as a tie.</p>
<div class="next"><a href="./first-ten-gameweeks-review.html">Read the first ten gameweeks review →</a><a href="./">All posts</a></div>
</article>
</div>
"""
META = {"seo_title": f"Best FPL GW{FIRST} wildcard team: three model-picked squads",
        "title": f"The best GW{FIRST} wildcard teams",
        "description": f"Three £100m wildcard squads for {LABEL} from the fplvalue model: the highest-scoring team, one without Haaland and a differential. Plus which picks matter, and which are close calls.",
        "date": today, "tag": "Wildcard", "kicker": f"Wildcard · GW{FIRST}"}
(POSTS / f"{SLUG}.html").write_text(f"<!--META {json.dumps(META, ensure_ascii=False)} -->\n{body}")
print("wrote", POSTS / f"{SLUG}.html")

# ── hero chart: six-week totals of the four squads ──────────────────────────────
BG, LINE, INK, INK2, INK3, ACCENT = "#131521", "#2f3449", "#eef0f7", "#b4b9cc", "#7c8299", "#8b97ff"
MUTED = "#4a5172"
def font(name):
    for f in fm.findSystemFonts():
        if name.lower() in Path(f).name.lower(): return fm.FontProperties(fname=f)
    return fm.FontProperties()
DISP = font("DejaVuSansCondensed-Bold"); BODY = font("LiberationSans-Regular"); MONO = font("DejaVuSansMono")
fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=BG)
fig.text(.045, .93, f"The best GW{FIRST} wildcard teams", fontproperties=DISP, fontsize=26, color=INK, va="top")
fig.text(.045, .855, f"Projected points {LABEL} for four £100m squads (starting XI + best captain each week)", fontproperties=BODY, fontsize=12, color=INK2, va="top")
fig.text(.955, .04, "fplvalue.co", fontproperties=DISP, fontsize=14, color=ACCENT, ha="right")
ax = fig.add_axes([.30, .14, .60, .62]); ax.set_facecolor(BG)
rows = [("A · Model's pick", tA, f"XI {xi_own(A):.0f}% owned"), ("C · Differential", tD, f"XI {xi_own(D):.0f}% owned"),
        ("B · No Haaland", tB, f"XI {xi_own(B):.0f}% owned"), ("Most-owned template", tT, f"XI {xi_own(T):.0f}% owned")]
y = np.arange(len(rows))[::-1]; lo = 280
for yi, (lab, v, sub) in zip(y, rows):
    ax.barh(yi, v - lo, left=lo, height=.56, color=ACCENT if lab.startswith("A") else MUTED, edgecolor=BG, linewidth=2)
    ax.text(v + 1.5, yi, f"{v:.0f}", va="center", fontproperties=MONO, fontsize=15, color=INK)
    fig.text(.045, .14 + .62 * (yi + .5) / len(rows) + .012, lab, fontproperties=DISP, fontsize=16, color=INK, va="center")
    fig.text(.045, .14 + .62 * (yi + .5) / len(rows) - .035, sub, fontproperties=BODY, fontsize=12.5, color=INK2, va="center")
ax.set_ylim(-.5, len(rows) - .5); ax.set_xlim(lo, max(tA, tB, tD, tT) + 12); ax.set_yticks([])
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(LINE); ax.tick_params(colors=INK3, labelsize=10); ax.grid(axis="x", color=LINE, linewidth=.6); ax.set_axisbelow(True)
for lab in ax.get_xticklabels(): lab.set_fontproperties(MONO)
fig.savefig(OUT / f"{SLUG}.png", facecolor=BG, dpi=100); plt.close(fig); print("wrote", OUT / f"{SLUG}.png")
