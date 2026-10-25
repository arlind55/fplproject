# Value Blog

Posts are HTML fragments in `posts/` with a JSON front-matter comment on line 1:

    <!--META {"title":"…","description":"…","date":"YYYY-MM-DD","tag":"Value report","kicker":"Value report · after GW7"} -->

Add `"seo_title"` to set the browser/Google title and social cards separately from the on-page title. Lead with
what people search for, e.g. `"FPL GW9 value report: best value players and xPts under-performers"`, and keep it
under about 65 characters. Deadline briefs get one automatically from `preview.py` (override with `seo_title:` in the notes meta).

Write the body with `<h3>` section headings (ids are added automatically and become the sidebar links).

## Weekly routine (from the repo root)

    python3 blog/charts.py [GW]              # writes docs/blog/img/gw<GW>-value-report.png and -template-check.png (GW defaults to latest finished)
    python3 blog/build.py blog/shell.css docs   # builds docs/blog/*.html, the index, and updates docs/sitemap.xml
    git add docs/ blog/ && git commit -m "Value Blog: GWxx" && git push

A post whose slug matches an image in `docs/blog/img/` gets it as the hero figure and social card.
To add a chart for a new post, add a block to `charts.py` that saves `<slug>.png`.
`shell.css` is the shared header/theme CSS copied from `docs/planner.html` — re-copy it if the shell changes.

## Deadline brief (the weekly gameweek preview)

One post per gameweek, built by `blog/preview.py` from `data/current` plus a short notes file. Every table, chart and
number comes from the data; the notes only hold the paragraphs and this week's three calls.

    cp blog/notes/gw8.md blog/notes/gw<GW>.md   # edit: meta, lede, calls, then one short block per section
    python3 blog/preview.py [GW]                 # writes blog/posts/gw<GW>-preview.html + docs/blog/img/gw<GW>-preview.png
    python3 blog/build.py blog/shell.css docs

Sections, in order: the brief (three call cards), last week's scorecard, captaincy (10k simulations), the fixtures
(model xG + clean-sheet odds), the model XI on a pitch, transfer targets by position, crowd vs model (this week's
transfers against the model's rank), fitness flags, luck check (Understat xPts vs the table), track record, verdict.

Calls are lines of the form `Label: web_name | one or two sentences` (add the team when a name is shared, e.g.
`Thomas (COV)`). A label starting with "Captain" is graded at double points. The calls are saved in the post's META,
so the next brief opens with a scorecard of them automatically. Notes paragraphs are separated by blank lines and
may contain inline HTML; a section with no notes block just shows its data.

## Team model
`scripts/team_model.py` builds `team_ratings.csv` (xG attack/defence by venue, shrunk toward average with a 6-game prior)
and `fixture_model.csv` (expected goals + 1–5 model difficulties per fixture). Both run inside `update_fpl_data.py`
and feed the planner's Rating toggle. Method is documented on `docs/metrics.html#model`.
