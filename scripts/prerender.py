#!/usr/bin/env python3
"""Bake a plain-HTML snapshot of the current data into the data pages, so search engines (and anyone
without JavaScript) see real numbers instead of "Loading…". Also refreshes <lastmod> for those pages
in docs/sitemap.xml.

Run from the repo root after the data refresh:  python3 scripts/prerender.py
Each page gets a <section class="snap"> between <!-- snapshot:start --> and <!-- snapshot:end -->,
inserted before <footer> the first time and replaced in place after that. The live page above it is
unchanged; the snapshot is a short "at a glance" summary that links into the tools."""
import csv, datetime, html, json, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "current"
DOCS = ROOT / "docs"
SITE = "https://fplvalue.co"
START, END = "<!-- snapshot:start -->", "<!-- snapshot:end -->"

CSS = """<style>
.snap{margin:28px 0 8px;padding:20px 22px;border:1px solid var(--line);border-radius:12px;background:var(--surface)}
.snap h2{font-family:"Barlow Condensed","Arial Narrow",sans-serif;font-weight:700;font-size:26px;line-height:1.1;margin:0 0 4px}
.snap h3{font-family:"Barlow Condensed","Arial Narrow",sans-serif;font-weight:600;font-size:19px;margin:0 0 8px}
.snap>p{margin:0 0 16px;color:var(--ink-2);font-size:14px;line-height:1.5}
.snap .cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:18px 28px}
.snap ol{margin:0;padding-left:20px;font-size:14px;line-height:1.6}
.snap ol span{color:var(--ink-3)}
.snap .tw{overflow-x:auto}
.snap table{border-collapse:collapse;width:100%;font-size:13.5px}
.snap th,.snap td{text-align:left;padding:6px 10px 6px 0;border-bottom:1px solid var(--line);white-space:nowrap}
.snap th{font-weight:600;color:var(--ink-3);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.snap td.n,.snap th.n{text-align:right;font-variant-numeric:tabular-nums}
.snap td.w{white-space:normal}
.snap a{color:var(--accent)}
</style>"""


def rows(name):
    with open(DATA / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(v, default=0.0):
    try: return float(v)
    except (TypeError, ValueError): return default


e = html.escape
meta = json.loads((DATA / "meta.json").read_text())
updated = (meta.get("flags_updated_at") or meta.get("updated_at") or "")[:10] or datetime.date.today().isoformat()
nice_date = datetime.date.fromisoformat(updated).strftime("%-d %B %Y")
pmeta = json.loads((DATA / "projection_meta.json").read_text())
gws = pmeta.get("gws") or []
gw = gws[0] if gws else meta.get("next_event")
span = f"GW{gws[0]}–{gws[-1]}" if len(gws) > 1 else f"GW{gw}"

projections = rows("projections.csv")
captains = sorted(rows("captains.csv"), key=lambda r: -num(r["sim_mean"]))[:5]
runs = sorted(rows("team_fixture_difficulty.csv"), key=lambda r: (num(r["avg_difficulty_next_5"], 9), r["team_name"]))
# value = projected points over the window per £m, among players expected to start
likely = [p for p in projections if num(p["p_start"]) >= 0.6 and num(p["avail_next"]) >= 0.75 and num(p["price"]) > 0]
value = sorted(likely, key=lambda p: -num(p["next_n"]) / num(p["price"]))[:5]
top = sorted(projections, key=lambda p: -num(p["next_n"]))[:25]


def who(p): return f'{e(p["web_name"])} <span>({e(p["team_short_name"])})</span>'


def cap_list():
    return "<ol>" + "".join(f'<li>{who(c)} v {e(c["next_opp"])}: {num(c["sim_mean"]):.1f} pts</li>' for c in captains) + "</ol>"


def value_list():
    return "<ol>" + "".join(f'<li>{who(p)} £{num(p["price"]):.1f}m: {num(p["next_n"]):.1f} pts, {num(p["next_n"]) / num(p["price"]):.2f} per £m</li>' for p in value) + "</ol>"


def runs_list(n=5):
    return "<ol>" + "".join(f'<li>{e(r["team_name"])} <span>(avg difficulty {num(r["avg_difficulty_next_5"]):.1f})</span></li>' for r in runs[:n]) + "</ol>"


def section(title, intro, inner):
    return f'{START}\n<section class="snap" aria-labelledby="snapTitle">\n{CSS}\n<h2 id="snapTitle">{title}</h2>\n<p>{intro}</p>\n{inner}\n</section>\n{END}'


snap_index = section(
    f"GW{gw} at a glance",
    f"The headline numbers from the FPL Value model, updated {nice_date}. Full detail is in the "
    f'<a href="./projections.html">projections</a> and the <a href="./planner.html">fixture planner</a>.',
    f'<div class="cols"><div><h3>Captain shortlist, GW{gw}</h3>{cap_list()}</div>'
    f'<div><h3>Best value, {span}</h3>{value_list()}</div>'
    f'<div><h3>Easiest fixture runs</h3>{runs_list()}</div></div>')

proj_rows = "".join(
    f'<tr><td>{e(p["web_name"])}</td><td>{e(p["team_short_name"])}</td><td>{e(p["pos"])}</td>'
    f'<td class="n">£{num(p["price"]):.1f}m</td><td>{e(p["next_opp"])}</td><td class="n">{num(p["next_gw"]):.1f}</td><td class="n">{num(p["next_n"]):.1f}</td></tr>'
    for p in top)
snap_proj = section(
    f"Top 25 FPL players by expected points, {span}",
    f"Snapshot of the model's projections, updated {nice_date}. The table above has every player, with filters.",
    f'<div class="cols"><div><h3>Captain picks, GW{gw}</h3>{cap_list()}</div><div><h3>Best value, {span}</h3>{value_list()}</div></div>'
    f'<div class="tw" style="margin-top:18px"><table><thead><tr><th>Player</th><th>Team</th><th>Pos</th><th class="n">Price</th>'
    f'<th>GW{gw}</th><th class="n">GW{gw} pts</th><th class="n">{span} pts</th></tr></thead><tbody>{proj_rows}</tbody></table></div>')

run_rows = "".join(
    f'<tr><td>{i}</td><td>{e(r["team_name"])}</td><td class="n">{num(r["avg_difficulty_next_5"]):.1f}</td><td class="w">{e(r["opponents_next_5"]).replace(" | ", ", ")}</td></tr>'
    for i, r in enumerate(runs, 1))
snap_planner = section(
    "Easiest FPL fixture runs, next five gameweeks",
    f"All 20 Premier League teams ranked by average official FPL difficulty over their next five fixtures, updated {nice_date}. "
    "Use the planner above to change the window, venue or rating.",
    f'<div class="tw"><table><thead><tr><th>#</th><th>Team</th><th class="n">Avg difficulty</th><th>Next five opponents</th></tr></thead><tbody>{run_rows}</tbody></table></div>')

pages = {"index.html": snap_index, "projections.html": snap_proj, "planner.html": snap_planner}
for name, snap in pages.items():
    f = DOCS / name; s = f.read_text()
    if START in s:
        s = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: snap, s, count=1, flags=re.S)
    else:
        assert s.count("\n<footer>") == 1, f"{name}: expected one top-level <footer>"
        s = s.replace("\n<footer>", "\n" + snap + "\n\n<footer>", 1)
    f.write_text(s)

# sitemap: the data pages change with every refresh
sm = DOCS / "sitemap.xml"; s = sm.read_text()
for path in ("/", "/planner.html", "/projections.html", "/compare.html", "/team.html", "/accuracy.html"):
    loc = f"<loc>{SITE}{path}</loc>"
    s = re.sub(re.escape(loc) + r"(<lastmod>[^<]*</lastmod>)?", f"{loc}<lastmod>{updated}</lastmod>", s, count=1)
sm.write_text(s)
print(f"snapshot for GW{gw} ({updated}) written to {', '.join(pages)}; sitemap lastmod updated")
