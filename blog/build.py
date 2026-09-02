#!/usr/bin/env python3
"""Assemble Value Blog pages from the shared shell CSS + per-post body fragments.
Usage (from repo root): python3 blog/build.py blog/shell.css docs
Bodies live in blog/posts/*.html with a JSON front-matter comment on line 1.
Charts: run blog/charts.py first; a post whose slug has docs/blog/img/<slug>.png gets it as
its hero figure and social card automatically."""
import sys, re, pathlib, json, datetime, html

shell_css = pathlib.Path(sys.argv[1]).read_text()
out = pathlib.Path(sys.argv[2]); blog_out = out / "blog"; blog_out.mkdir(parents=True, exist_ok=True)
here = pathlib.Path(__file__).parent
ARTICLE_CSS = (here / "article.css").read_text()
THEME_JS = (here / "theme.js").read_text()
SITE = "https://fplvalue.co"

def nice(d): return datetime.date.fromisoformat(d).strftime("%-d %b %Y")

def tabs(active):
    cur = ' aria-current="page"'
    items = [("Dashboard", "../index.html"), ("Fixture planner", "../planner.html"), ("Metrics", "../metrics.html"), ("Value Blog", "./index.html")]
    return "".join(f'<a href="{h}"{cur if n == active else ""}>{n}</a>' for n, h in items)

def slugify(text):
    s = re.sub(r"<small>.*?</small>", "", text, flags=re.S)          # drop the mono annotation
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s[:60] or "section"

def add_ids(body):
    """Give every h3 without an id one derived from its text (unique within the post)."""
    seen = {}
    def rep(m):
        attrs, inner = m.group(1), m.group(2)
        if "id=" in attrs: return m.group(0)
        base = slugify(inner); n = seen.get(base, 0); seen[base] = n + 1
        return f'<h3 id="{base if n == 0 else f"{base}-{n+1}"}"{attrs}>{inner}</h3>'
    return re.sub(r"<h3([^>]*)>(.*?)</h3>", rep, body, flags=re.S)

def sections(body):
    """(id, text) for every h3 in the post body — text without the <small> annotation."""
    out = []
    for m in re.finditer(r'<h3 id="([^"]+)"[^>]*>(.*?)</h3>', body, re.S):
        text = re.sub(r"<small>.*?</small>", "", m.group(2), flags=re.S)
        out.append((m.group(1), re.sub(r"<[^>]+>", "", text).strip()))
    return out

def sidebar(posts, current=None, current_sections=()):
    """Date-grouped post list. The group containing the current post is open and its
    sections are listed under it; other groups are collapsed <details>."""
    by_date = {}
    for p in posts: by_date.setdefault(p["date"], []).append(p)
    parts = ['<nav class="side" aria-label="Posts"><div class="label">Value Blog</div>']
    for date in sorted(by_date, reverse=True):
        group = by_date[date]; is_open = any(p["path"] == current for p in group)
        parts.append(f'<details{" open" if is_open else ""}><summary><time datetime="{date}">{nice(date)}</time><span class="n">{len(group)}</span></summary>')
        for p in group:
            here_ = p["path"] == current
            cur_cls = " cur" if here_ else ""; cur_attr = ' aria-current="page"' if here_ else ""
            parts.append(f'<a class="p{cur_cls}" href=".{p["path"][5:]}"{cur_attr}><span class="tag">{p["tag"]}</span>{html.escape(p["title"])}</a>')
            if here_ and current_sections:
                parts.append('<div class="secs">' + "".join(f'<a href="#{i}">{html.escape(t)}</a>' for i, t in current_sections) + "</div>")
        parts.append("</details>")
    parts.append('<a class="all" href="./index.html">All posts →</a></nav>')
    return "".join(parts)

def page(meta, body, posts, active="Value Blog"):
    url = SITE + meta["path"]
    is_post = bool(meta.get("date"))
    img = meta.get("image") or f"{SITE}/og-image.png"
    date_meta = f'<meta property="article:published_time" content="{meta["date"]}">' if is_post else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(meta['title'])} — FPL Value</title>
<meta name="description" content="{html.escape(meta['description'])}">
<link rel="canonical" href="{url}">
<meta property="og:type" content="{'article' if is_post else 'website'}">
<meta property="og:site_name" content="FPL Value">
<meta property="og:title" content="{html.escape(meta['title'])}">
<meta property="og:description" content="{html.escape(meta['description'])}">
<meta property="og:url" content="{url}">
<meta property="og:image" content="{img}">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
{date_meta}
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{html.escape(meta['title'])}">
<meta name="twitter:description" content="{html.escape(meta['description'])}">
<meta name="twitter:image" content="{img}">
<meta name="theme-color" content="#131521" media="(prefers-color-scheme: dark)">
<meta name="theme-color" content="#f3f4f8" media="(prefers-color-scheme: light)">
<link rel="icon" type="image/png" sizes="32x32" href="../favicon-32.png">
<link rel="apple-touch-icon" href="../apple-touch-icon.png">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap">
<style>
{shell_css}
{ARTICLE_CSS}
</style>
</head>
<body>
<div class="wrap">
<header class="topbar">
  <div class="brand"><div><h1><a href="../index.html" style="color:inherit;text-decoration:none">FPL Value</a></h1><small>Premier League 2026/27</small></div></div>
  <nav class="tabs" aria-label="Pages">{tabs(active)}</nav>
  <div class="right">
    <span class="status"><i style="background:var(--accent)"></i><span>{html.escape(meta.get('kicker', 'Value Blog'))}</span></span>
    <button class="theme" id="themeBtn" type="button" aria-label="Toggle light / dark"></button>
  </div>
</header>
{body}
<footer>
  <span>FPL Value · independent and unaffiliated with the Premier League or Fantasy Premier League. Data: official FPL API via <a href="https://github.com/arlind55/fplproject">fplproject</a>.</span>
  <span><a href="../index.html">Dashboard</a> · <a href="../planner.html">Fixture planner</a> · <a href="../metrics.html">Metrics</a></span>
</footer>
</div>
<script>
{THEME_JS}
</script>
</body>
</html>
"""

# ── Collect posts ──────────────────────────────────────────────────────────────
posts = []
for f in sorted((here / "posts").glob("*.html")):
    raw = f.read_text()
    m = re.match(r"<!--META\s*(\{.*?\})\s*-->", raw, re.S)
    meta = json.loads(m.group(1)); meta["path"] = f"/blog/{f.name}"; meta["slug"] = f.stem
    meta["body"] = raw[m.end():]
    if (blog_out / "img" / f"{f.stem}.png").exists():
        meta["image"] = f"{SITE}/blog/img/{f.stem}.png"
    posts.append(meta)
posts.sort(key=lambda p: p["date"], reverse=True)

# ── Render posts ───────────────────────────────────────────────────────────────
for i, p in enumerate(posts):
    body = add_ids(p["body"])
    # hero figure straight after the lede, if a chart exists
    if p.get("image"):
        fig = f'<figure class="hero"><img src="./img/{p["slug"]}.png" alt="{html.escape(p["title"])} — chart" width="1200" height="630" loading="eager"></figure>'
        body = re.sub(r"(</p>)", r"\1" + fig, body, count=1) if 'class="lede"' in body else fig + body
    # prev / next
    newer = posts[i - 1] if i > 0 else None; older = posts[i + 1] if i + 1 < len(posts) else None
    nav = '<div class="next">' + (f'<a href=".{older["path"][5:]}">← {html.escape(older["title"])}</a>' if older else "<span></span>") + (f'<a href=".{newer["path"][5:]}">{html.escape(newer["title"])} →</a>' if newer else '<a href="./index.html">All posts</a>') + "</div>"
    body = re.sub(r'<div class="next">.*?</div>', nav, body, flags=re.S) if '<div class="next">' in body else body.replace("</article>", nav + "</article>")
    body = body.replace('<div class="article">', '<div class="article with-side">' + sidebar(posts, p["path"], sections(body)), 1)
    (blog_out / f"{p['slug']}.html").write_text(page(p, body, posts))

# ── Index ──────────────────────────────────────────────────────────────────────
cards = "".join(f"""
  <a class="post" href=".{p['path'][5:]}">
    {f'<img src="./img/{p["slug"]}.png" alt="" loading="lazy">' if p.get('image') else ''}
    <div>
      <div class="post-meta"><span class="tag">{p['tag']}</span><time datetime="{p['date']}">{nice(p['date'])}</time></div>
      <h3>{html.escape(p['title'])}</h3>
      <p>{html.escape(p['description'])}</p>
    </div>
  </a>""" for p in posts)
index_body = f"""
<div class="article with-side">{sidebar(posts)}
<article class="prose wide">
<h2>Value Blog</h2>
<p class="lede">Short, data-led pieces built from the numbers on the dashboard and planner: a weekly value report, template checks and fixture swings ahead of the breaks.</p>
<div class="posts">{cards}
</div>
</article>
</div>"""
(blog_out / "index.html").write_text(page({"title": "Value Blog", "description": "FPL analysis from FPL Value: weekly value reports (VAPM, points per million, xPts under-performers), fixture swings and template checks.", "path": "/blog/", "kicker": f"{len(posts)} posts"}, index_body, posts))

# ── Sitemap: rewrite the blog section in docs/sitemap.xml ──────────────────────
sm = out / "sitemap.xml"
if sm.exists():
    s = sm.read_text()
    s = re.sub(r"\s*<url><loc>https://fplvalue\.co/blog/[^<]*</loc>.*?</url>", "", s, flags=re.S)
    block = f'\n  <url><loc>{SITE}/blog/</loc><changefreq>weekly</changefreq><priority>0.8</priority></url>' + "".join(
        f'\n  <url><loc>{SITE}{p["path"]}</loc><lastmod>{p["date"]}</lastmod><changefreq>monthly</changefreq><priority>0.6</priority></url>' for p in posts)
    sm.write_text(s.replace("\n</urlset>", block + "\n</urlset>"))
print(f"built {len(posts)} posts + index; sitemap updated")
