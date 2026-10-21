#!/usr/bin/env python3
"""Build the weekly Deadline Brief (gameweek preview) from data/current + a short notes file.
Usage (from repo root): python3 blog/preview.py [GW]      (GW defaults to the next gameweek in meta.json)
Reads   blog/notes/gw<GW>.md                      the editorial prose and this week's calls
Writes  blog/posts/gw<GW>-preview.html            the post fragment (then run build.py as usual)
        docs/blog/img/gw<GW>-preview.png          hero chart + social card
Every table, chart and number in the post comes from the data; only the paragraphs come from the notes.
The calls are stored in the post's META so next week's brief can grade them automatically."""
import sys, re, json, html, datetime, pathlib
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "current"
POSTS = ROOT / "blog" / "posts"; NOTES = ROOT / "blog" / "notes"; IMG = ROOT / "docs" / "blog" / "img"
UK = ZoneInfo("Europe/London")
POS_ORDER = ["GKP", "DEF", "MID", "FWD"]
esc = html.escape

GW = int(sys.argv[1]) if len(sys.argv) > 1 else json.loads((DATA / "meta.json").read_text())["next_event"]
SLUG = f"gw{GW}-preview"

# ── Notes: "## section" blocks; meta/calls are "key: value" lines, everything else is HTML ──
def read_notes(path):
    out, key = {}, None
    for line in path.read_text().splitlines():
        m = re.match(r"^##\s+(\S+)\s*$", line)
        if m: key = m.group(1); out[key] = []; continue
        if key: out[key].append(line)
    return {k: "\n".join(v).strip() for k, v in out.items()}
notes = read_notes(NOTES / f"gw{GW}.md")
def kv(block): return dict(re.match(r"\s*([^:]+?)\s*:\s*(.*)", l).groups() for l in block.splitlines() if ":" in l)
meta_n = kv(notes["meta"])
def prose(key):
    """Paragraphs separated by blank lines; a paragraph that opens with a block tag (<div>, <ul>, <table>…) is left alone."""
    txt = notes.get(key, "")
    block = re.compile(r"\s*<(p|div|ul|ol|table|figure|h\d|blockquote)[\s>]")
    return "\n".join(p if block.match(p) else f"<p>{p}</p>" for p in re.split(r"\n\s*\n", txt) if p.strip())

# ── Data ───────────────────────────────────────────────────────────────────────
teams = pd.read_csv(DATA / "teams.csv").set_index("id")
short = teams.short_name
proj = pd.read_csv(DATA / "projections.csv")
caps = pd.read_csv(DATA / "captains.csv")
squad = pd.read_csv(DATA / "model_squad.csv")
players = pd.read_csv(DATA / "players.csv")
events = pd.read_csv(DATA / "events.csv").set_index("id")
fm = pd.read_csv(DATA / "fixture_model.csv")
acc = json.loads((DATA / "accuracy.json").read_text())
hist = pd.read_csv(DATA / "player_history.csv")
pm = json.loads((DATA / "projection_meta.json").read_text())
H = [c for c in proj.columns if re.fullmatch(r"gw_\d+", c)]          # horizon columns, e.g. gw_8..gw_12
H_LABEL = f"GW{H[0][3:]}–{H[-1][3:]}"
proj = proj.merge(players[["id", "transfers_in_event", "transfers_out_event", "news", "team"]], on="id", how="left")
by_name = proj.set_index("web_name")
deadline = datetime.datetime.fromisoformat(events.loc[GW, "deadline_time"].replace("Z", "+00:00")).astimezone(UK)
today = datetime.date.today().isoformat()

def player(name):
    """web_name as in projections.csv; add the team when two players share it, e.g. "Thomas (COV)"."""
    m = re.fullmatch(r"(.+?)\s*\((\w{3})\)", name)
    web, team = m.groups() if m else (name, None)
    r = proj[(proj.web_name == web) & ((proj.team_short_name == team) if team else True)]
    if len(r) != 1: sys.exit(f"notes: {name!r} matches {len(r)} players (use web_name, plus the team in brackets if shared)")
    return r.iloc[0]

# per-team run of fixtures over the projection horizon, with the model's blended 1–5 difficulty
runs = {}
for r in fm[fm.event.between(int(H[0][3:]), int(H[-1][3:]))].sort_values(["event", "kickoff_time"]).itertuples():
    runs.setdefault(r.team_h, []).append((int(r.event), short[r.team_a], "H", int(r.blend_diff_home)))
    runs.setdefault(r.team_a, []).append((int(r.event), short[r.team_h], "A", int(r.blend_diff_away)))
team_id = {v: k for k, v in short.items()}
def strip(team_short, small=True):
    cells = "".join(f'<span class="fix d{d}" title="GW{gw} {o} ({ha}) · difficulty {d}">{o}<small>{ha}</small></span>' for gw, o, ha, d in runs.get(team_id[team_short], []))
    return f'<span class="fixrow{" mini" if small else ""}">{cells}</span>'

def pos(p): return f'<span class="pos {p}">{p}</span>'
def pct(x): return f"{x * 100:.0f}%"
def own(x): return f"{x:.1f}%" if x < 10 else f"{x:.0f}%"

# ── 1. The brief: this week's calls ─────────────────────────────────────────────
calls = []
for line in notes["calls"].splitlines():
    m = re.match(r"\s*([^:]+?)\s*:\s*([^|]+?)\s*\|\s*(.*)", line)
    if not m: continue
    label, name, why = m.groups(); r = player(name)
    calls.append({"label": label, "id": int(r.id), "name": r.web_name, "why": why, "team": r.team_short_name, "pos": r.pos, "price": float(r.price), "gw": float(r[H[0]]), "n": float(r.next_n), "own": float(r.selected_by_percent)})
brief = '<div class="brief">' + "".join(f"""
<div class="call">
  <div class="lbl">{esc(c['label'])}</div>
  <div class="who">{esc(c['name'])} <small>{c['team']} · £{c['price']:.1f}m · {own(c['own'])} owned</small></div>
  <div class="nums"><span><b>{c['gw']:.1f}</b>GW{GW} xPts</span><span><b>{c['n']:.1f}</b>{H_LABEL}</span></div>
  <p>{c['why']}</p>
</div>""" for c in calls) + "\n</div>"

# ── 2. Scorecard: grade the previous brief's calls ──────────────────────────────
prev = POSTS / f"gw{GW - 1}-preview.html"
scorecard = ""
if prev.exists():
    pmeta = json.loads(re.match(r"<!--META\s*(\{.*?\})\s*-->", prev.read_text(), re.S).group(1))
    rows = []
    for c in pmeta.get("calls", []):
        g = hist[(hist.player_id == c["id"]) & (hist["round"] == GW - 1)]
        pts = int(g.total_points.sum()) if len(g) else None
        mult = 2 if c["label"].lower().startswith("captain") else 1
        exp = c["gw"] * mult
        if pts is None: verdict, cls = "Didn't play", "down"
        else:
            got = pts * mult
            verdict, cls = (("Hit", "up") if got >= exp + 1 else ("Miss", "down") if got <= exp - 2 else ("Par", ""))
        got_txt = "–" if pts is None else (f"{pts}×2 = <strong>{pts * 2}</strong>" if mult == 2 else f"<strong>{pts}</strong>")
        rows.append(f'<tr><td>{esc(c["label"])}</td><td>{esc(c["name"])} <span class="dim">{c["team"]}</span></td><td class="n">{exp:.1f}</td><td class="n">{got_txt}</td><td class="{cls}">{verdict}</td></tr>')
    if rows:
        scorecard = f"""<h3>Last week's calls <small>GW{GW - 1} scorecard</small></h3>
<div class="tw"><table class="t"><thead><tr><th>Call</th><th>Player</th><th class="n">Expected</th><th class="n">Scored</th><th>Verdict</th></tr></thead><tbody>
{''.join(rows)}
</tbody></table></div>
{prose('scorecard')}"""

# ── 3. Captaincy ───────────────────────────────────────────────────────────────
cap = caps.head(8).copy()
pool = caps.head(15)
most_owned = pool.loc[pool.selected_by_percent.idxmax()]
if most_owned.id not in set(cap.id): cap = pd.concat([cap.head(7), most_owned.to_frame().T])
cap_rows = []
top_mean = float(cap.sim_mean.max())
for r in cap.itertuples():
    w_mean = r.sim_mean / cap.p90.max() * 100; w_p90 = r.p90 / cap.p90.max() * 100
    flag = f' <span class="flag" title="{esc(str(by_name.loc[r.web_name].news) if r.web_name in by_name.index else "")}">{pct(r.avail_next)} fit</span>' if r.avail_next < 1 else ""
    cap_rows.append(f"""<tr{' class="hl"' if r.sim_mean == top_mean else ''}><td>{pos(r.pos)} <strong>{esc(r.web_name)}</strong>{flag}<span class="sub">{r.team_short_name} · {esc(r.next_opp)}</span></td>
<td class="bar"><span class="range" style="width:{w_p90:.1f}%"></span><span class="mean" style="width:{w_mean:.1f}%"></span></td>
<td class="n"><strong>{r.sim_mean:.1f}</strong></td><td class="n">{r.p90:.0f}</td><td class="n">{pct(r.haul_prob)}</td><td class="n">{own(r.selected_by_percent)}</td></tr>""")
captaincy = f"""<h3>Captaincy <small>10,000 simulated GW{GW}s</small></h3>
<div class="tw"><table class="t cap"><thead><tr><th>Player</th><th class="bar"><span class="key"><i class="mean"></i>expected <i class="range"></i>1-in-10 ceiling</span></th><th class="n">xPts</th><th class="n">Ceiling</th><th class="n">10+ pts</th><th class="n">Owned</th></tr></thead><tbody>
{''.join(cap_rows)}
</tbody></table></div>
{prose('captain')}"""

# ── 4. The fixtures: model xG and clean-sheet odds for every match ─────────────
games = fm[fm.event == GW].sort_values(["kickoff_time", "id"])
cards = []
for r in games.itertuples():
    k = datetime.datetime.fromisoformat(r.kickoff_time.replace("Z", "+00:00")).astimezone(UK)
    tot = r.xg_home + r.xg_away; wh = r.xg_home / tot * 100
    cards.append(f"""<div class="match">
  <div class="ko">{k:%a %H:%M}</div>
  <div class="side"><b>{short[r.team_h]}</b><span>{r.xg_home:.2f}</span></div>
  <div class="split"><i style="width:{wh:.1f}%"></i></div>
  <div class="side"><b>{short[r.team_a]}</b><span>{r.xg_away:.2f}</span></div>
  <div class="cs"><span>CS {pct(r.cs_prob_home)}</span><span>CS {pct(r.cs_prob_away)}</span></div>
</div>""")
fixtures = f"""<h3>The fixtures <small>model xG · clean-sheet odds</small></h3>
<div class="matches">{''.join(cards)}</div>
{prose('fixtures')}"""

# ── 5. The model XI on a pitch ─────────────────────────────────────────────────
xi = squad[squad.is_starter]; bench = squad[~squad.is_starter]
def chip(r):
    c = '<i class="c">C</i>' if r.is_captain else ""
    return f'<div class="chip">{c}<b>{esc(r.web_name)}</b><span>{esc(r.next_opp)}</span><em>{r.next_gw * (2 if r.is_captain else 1):.1f}</em></div>'
lines = "".join(f'<div class="line">{"".join(chip(r) for r in xi[xi.pos == p].itertuples())}</div>' for p in POS_ORDER)
shape = "-".join(str((xi.pos == p).sum()) for p in POS_ORDER[1:])
xi_pts = float((xi.next_gw * np.where(xi.is_captain, 2, 1)).sum())
pitch = f"""<h3>The model's XI <small>{shape} · £{squad.price.sum():.1f}m</small></h3>
<div class="pitch">{lines}</div>
<div class="benchrow"><span class="lbl">Bench</span>{''.join(chip(r) for r in bench.itertuples())}</div>
<p class="note">Best 15 under £{pm['squad']['budget']:.1f}m for the next five gameweeks, picked by the optimiser on the <a href="../projections.html">projections page</a>. Expected GW{GW} total with the armband: <strong>{xi_pts:.1f}</strong>.</p>
{prose('xi')}"""

# ── 6. Transfer targets by position ────────────────────────────────────────────
ok = proj[(proj.avail_next >= .75) & (proj.p_start >= .7)].copy(); ok["per_m"] = ok.next_n / ok.price
tgt_rows = []
for p in POS_ORDER:
    g = ok[ok.pos == p]
    picks = list(g.nlargest(3, "next_n").id)
    value = g[g.next_n >= g.next_n.nlargest(20).min()].nlargest(4, "per_m")
    picks += [i for i in value.id if i not in picks][:2]
    tgt_rows.append(f'<tr class="grp"><td colspan="5">{pos(p)}</td></tr>')
    for r in g.set_index("id").loc[picks].reset_index().itertuples():
        tag = "value" if r.id not in list(g.nlargest(3, "next_n").id) else ("differential" if r.selected_by_percent < 10 else "")
        tag = f' <span class="tag">{tag}</span>' if tag else ""
        tgt_rows.append(f'<tr><td><strong>{esc(r.web_name)}</strong>{tag}<span class="sub">{r.team_short_name} · £{r.price:.1f}m · {own(r.selected_by_percent)}</span></td><td class="n"><strong>{r.next_n:.1f}</strong></td><td class="n">{r.per_m:.2f}</td><td class="n">{getattr(r, H[0]):.1f}</td><td>{strip(r.team_short_name)}</td></tr>')
targets = f"""<h3>Transfer targets <small>{H_LABEL} projection</small></h3>
<div class="tw"><table class="t tgt"><thead><tr><th>Player</th><th class="n">{H_LABEL}</th><th class="n">Per £m</th><th class="n">GW{GW}</th><th>Next five (model difficulty)</th></tr></thead><tbody>
{''.join(tgt_rows)}
</tbody></table></div>
{prose('targets')}"""

# ── 7. Crowd vs model: this week's transfer market ─────────────────────────────
ok_all = proj.copy(); ok_all["pos_rank"] = ok_all.groupby("pos").next_n.rank(ascending=False, method="min")
def verdict(r, buying):
    if r.avail_next == 0: return ("Agree: injured", "up") if not buying else ("Ruled out", "down")
    if buying: return ("Agree", "up") if r.pos_rank <= 10 else ("Model is cooler", "down") if r.pos_rank > 20 else ("Fair", "")
    return ("Model disagrees", "down") if r.pos_rank <= 10 else ("Agree", "up") if r.pos_rank > 20 else ("Fair", "")
def market(df, col, buying):
    out = []
    for r in df.nlargest(5, col).itertuples():
        v, cls = verdict(r, buying)
        out.append(f'<tr><td><strong>{esc(r.web_name)}</strong><span class="sub">{r.team_short_name} · £{r.price:.1f}m</span></td><td class="n">{getattr(r, col) / 1000:.0f}k</td><td class="n">#{int(r.pos_rank)} {r.pos}</td><td class="{cls}">{v}</td></tr>')
    return "".join(out)
crowd = f"""<h3>Crowd vs model <small>this week's transfers</small></h3>
<div class="pair">
<div class="tw"><table class="t"><thead><tr><th>Most bought</th><th class="n">In</th><th class="n">Model rank</th><th>Verdict</th></tr></thead><tbody>{market(ok_all, 'transfers_in_event', True)}</tbody></table></div>
<div class="tw"><table class="t"><thead><tr><th>Most sold</th><th class="n">Out</th><th class="n">Model rank</th><th>Verdict</th></tr></thead><tbody>{market(ok_all, 'transfers_out_event', False)}</tbody></table></div>
</div>
<p class="note">Model rank = position among players at the same position on the {H_LABEL} projection. Transfers are this gameweek's totals so far, from the FPL API.</p>
{prose('crowd')}"""

# ── 8. Fitness flags ───────────────────────────────────────────────────────────
fl = proj[(proj.avail_next < 1) & (proj.selected_by_percent >= 5)].sort_values("selected_by_percent", ascending=False)
flags = ""
if len(fl):
    flags = f"""<h3>Fitness flags <small>5%+ owned</small></h3>
<ul class="flags">{''.join(f'<li class="{"out" if r.avail_next == 0 else "doubt"}"><b>{esc(r.web_name)}</b> <span class="dim">{r.team_short_name} · {own(r.selected_by_percent)} owned</span><span class="st">{"Out" if r.avail_next == 0 else pct(r.avail_next)}</span><small>{esc(str(r.news))}</small></li>' for r in fl.itertuples())}</ul>
{prose('flags')}"""

# ── 9. Luck check: league table vs Understat expected points ───────────────────
luck = ""
us = ROOT / "research" / "understat" / "data" / "team_matches.csv"
if us.exists():
    m = pd.read_csv(us); season = m.season.max(); m = m[m.season == season]
    t = m.groupby("team").agg(gp=("pts", "size"), pts=("pts", "sum"), xpts=("xpts", "sum")).reset_index()
    t["d"] = t.pts - t.xpts; t = t.sort_values("d", ascending=False)
    mx = t.d.abs().max()
    bars = "".join(f'<div class="lk"><span class="tm">{esc(r.team)}</span><span class="axis"><i class="{"pos" if r.d > 0 else "neg"}" style="width:{abs(r.d) / mx * 50:.1f}%"></i></span><span class="v {"up" if r.d > 0 else "down"}">{r.d:+.1f}</span><span class="dim">{r.pts} pts · {r.xpts:.1f} xPts</span></div>' for r in t.itertuples())
    luck = f"""<h3>Luck check <small>points minus Understat xPts, {season}</small></h3>
<div class="luck">{bars}</div>
{prose('luck')}"""

# ── 10. Track record ───────────────────────────────────────────────────────────
s = acc["summary"]; wk = acc["weeks"]
tiles = f"""<div class="tiles">
<div><span>Model error</span><b>{s['model_mae']:.2f}</b><small>avg points per player</small></div>
<div><span>FPL's own error</span><b>{s['fpl_mae']:.2f}</b><small>same players, same weeks</small></div>
<div><span>Weeks won</span><b>{s['model_wins']}/{s['head_to_head_weeks']}</b><small>lower error than FPL</small></div>
<div><span>Captain points</span><b>{s['captain_points']}</b><small>{s['captain_points'] / s['weeks']:.1f} a week before doubling</small></div>
</div>"""
wrows = "".join(f'<tr><td>GW{w["gw"]}</td><td class="n">{w["model"]["mae"]:.2f}</td><td class="n">{w["fpl"]["mae"]:.2f}</td><td class="n">{w["model"]["spearman"]:.2f}</td><td>{esc(w["model"]["captain"]["name"])} <span class="dim">{w["model"]["captain"]["projected"]:.1f} → </span><strong>{w["model"]["captain"]["actual"]}</strong></td><td>{esc(w["fpl"]["captain"]["name"])} <span class="dim">→ </span><strong>{w["fpl"]["captain"]["actual"]}</strong></td></tr>' for w in wk)
record = f"""<h3>Track record <small>live since GW{wk[0]['gw']}</small></h3>
{tiles}
<div class="tw"><table class="t"><thead><tr><th>GW</th><th class="n">Model error</th><th class="n">FPL error</th><th class="n">Rank corr.</th><th>Model captain</th><th>FPL's captain</th></tr></thead><tbody>{wrows}</tbody></table></div>
{prose('record')}"""

# ── Assemble ───────────────────────────────────────────────────────────────────
headline = meta_n["headline"]
post_meta = {"title": meta_n["title"], "description": meta_n["description"], "date": meta_n.get("date", today), "tag": "Deadline brief",
             "kicker": f"Deadline brief · GW{GW}", "calls": [{k: c[k] for k in ("label", "id", "name", "team", "gw")} for c in calls]}
body = f"""<!--META {json.dumps(post_meta, ensure_ascii=False)} -->
<div class="article">
<article class="prose brief-post">
<div class="kicker"><span>Deadline brief</span><time datetime="{post_meta['date']}">GW{GW} · deadline {deadline:%a %-d %b, %H:%M} UK</time></div>
<h2>{headline}</h2>
<p class="lede">{notes['lede']}</p>
<h3>The brief <small>three calls for GW{GW}</small></h3>
{brief}
{scorecard}
{captaincy}
{fixtures}
{pitch}
{targets}
{crowd}
{flags}
{luck}
{record}
<div class="callout"><b>Before the deadline.</b> {notes['verdict']}</div>
<p class="note">Method: projections are the site's own model (xG, xA, clean-sheet and minutes rates, shrunk toward last season and price; details on the <a href="../metrics.html">metrics page</a>), run on data pulled {pd.Timestamp(pm['generated_at']).tz_convert(UK):%a %-d %b %H:%M}. Captaincy uses 10,000 simulations of GW{GW}; the ceiling is the 90th percentile. Fixture difficulty is the model's 1–5 blend of xG ratings and FPL's own. Luck check uses <a href="https://understat.com">Understat</a> expected points. Check team news before the deadline: the model only knows what FPL's availability flags said at the time.</p>
<div class="next"></div>
</article>
</div>
"""
(POSTS / f"{SLUG}.html").write_text(body)
print("wrote", POSTS / f"{SLUG}.html")

# ── Hero chart / social card ───────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fmgr
BG, LINE, INK, INK2, INK3, ACCENT = "#131521", "#2f3449", "#eef0f7", "#b4b9cc", "#7c8299", "#8b97ff"
POSC = {"GKP": "#e0b23a", "DEF": "#6fa3e6", "MID": "#5fc48f", "FWD": "#f08a5c"}
def font(name):
    for f in fmgr.findSystemFonts():
        if name.lower() in pathlib.Path(f).name.lower(): return fmgr.FontProperties(fname=f)
    return fmgr.FontProperties()
DISP = font("DejaVuSansCondensed-Bold"); BODY = font("LiberationSans-Regular"); MONO = font("DejaVuSansMono")
c = cap.iloc[::-1].reset_index(drop=True)
fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=BG)
ax = fig.add_axes([.20, .12, .70, .64]); ax.set_facecolor(BG)
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(LINE); ax.tick_params(colors=INK3, labelsize=10, left=False)
ax.grid(axis="x", color=LINE, linewidth=.6); ax.set_axisbelow(True)
y = np.arange(len(c))
ax.barh(y, c.p90, height=.62, color=LINE, zorder=1, label="1-in-10 ceiling")
ax.barh(y, c.sim_mean, height=.62, color=[POSC[p] for p in c.pos], zorder=2, label="Expected points")
for yi, r in c.iterrows():
    ax.text(-.02, yi + .1, r.web_name, ha="right", va="center", color=INK, fontproperties=BODY, fontsize=13, transform=ax.get_yaxis_transform())
    ax.text(-.02, yi - .28, f"{r.team_short_name} v {r.next_opp} · {own(r.selected_by_percent)} owned", ha="right", va="center", color=INK3, fontproperties=MONO, fontsize=8.5, transform=ax.get_yaxis_transform())
    ax.text(r.sim_mean - .15, yi, f"{r.sim_mean:.1f}", ha="right", va="center", color=BG, fontproperties=MONO, fontsize=11)
    ax.text(r.p90 + .2, yi, f"{pct(r.haul_prob)} chance of 10+" + ("  ·  " + pct(r.avail_next) + " fit" if r.avail_next < 1 else ""), va="center", color=INK2, fontproperties=MONO, fontsize=9.5)
ax.set_yticks([]); ax.set_xlim(0, c.p90.max() * 1.42)
for lab in ax.get_xticklabels(): lab.set_fontproperties(MONO)
ax.set_xlabel(f"GW{GW} points (simulated)", fontproperties=BODY, color=INK3, fontsize=10)
fig.text(.045, .93, f"GW{GW} captaincy: who the model backs", fontproperties=DISP, fontsize=26, color=INK, va="top")
fig.text(.045, .855, "Coloured bar: expected points. Grey: the 1-in-10 ceiling. From 10,000 simulated gameweeks.", fontproperties=BODY, fontsize=12, color=INK2, va="top")
fig.text(.955, .04, "fplvalue.co", fontproperties=DISP, fontsize=14, color=ACCENT, ha="right")
IMG.mkdir(parents=True, exist_ok=True)
fig.savefig(IMG / f"{SLUG}.png", facecolor=BG, dpi=100); plt.close(fig)
print("wrote", IMG / f"{SLUG}.png")
