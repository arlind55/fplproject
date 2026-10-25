# Premier League Data Repo Blueprint v2

This version extends the starter repo with:
- per-player `element-summary` pulls
- derived app-ready tables
- weekly snapshots
- a schema reference for frontend use

## Outputs

### Core current files
- `data/current/bootstrap_static.json`
- `data/current/players.csv`
- `data/current/teams.csv`
- `data/current/events.csv`
- `data/current/fixtures.csv`

### Player detail files
- `data/current/player_summaries/{player_id}.json`
- `data/current/player_history.csv`
- `data/current/player_future_fixtures.csv`

### Derived files
- `data/current/player_prices.csv`
- `data/current/player_form.csv`
- `data/current/next_fixtures.csv`
- `data/current/team_fixture_difficulty.csv`

### Snapshots
Every run also stores dated copies under `data/history/YYYY-MM-DD/`.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/update_fpl_data.py
```

## App usage

Recommended app tables:
- `player_form.csv` for rankings and cards
- `next_fixtures.csv` for fixture views
- `team_fixture_difficulty.csv` for schedule analysis
- `player_history.csv` for trend charts

## Site snapshot (SEO)

`scripts/prerender.py` writes a plain-HTML "at a glance" summary of the current data into `docs/index.html`,
`docs/projections.html` and `docs/planner.html` (between `<!-- snapshot:start -->` / `<!-- snapshot:end -->`) and bumps
`<lastmod>` in `docs/sitemap.xml`, so search engines see real numbers without running JavaScript. Both data workflows
run it after refreshing the data.
