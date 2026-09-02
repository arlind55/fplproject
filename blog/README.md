# Value Blog

Posts are HTML fragments in `posts/` with a JSON front-matter comment on line 1:

    <!--META {"title":"…","description":"…","date":"YYYY-MM-DD","tag":"Value report","kicker":"Value report · after GW7"} -->

Write the body with `<h3>` section headings (ids are added automatically and become the sidebar links).

## Weekly routine (from the repo root)

    python3 blog/charts.py                   # regenerates docs/blog/img/<slug>.png for the current posts
    python3 blog/build.py blog/shell.css docs   # builds docs/blog/*.html, the index, and updates docs/sitemap.xml
    git add docs/ blog/ && git commit -m "Value Blog: GWxx" && git push

A post whose slug matches an image in `docs/blog/img/` gets it as the hero figure and social card.
To add a chart for a new post, add a block to `charts.py` that saves `<slug>.png`.
`shell.css` is the shared header/theme CSS copied from `docs/planner.html` — re-copy it if the shell changes.
