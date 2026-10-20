# Understat EPL xG history

Standalone research kit, separate from the FPL value site.

- `scrape_understat.py` pulls every Premier League season Understat has (2014/15 to now) from the JSON endpoints the site itself uses (`/getLeagueData/EPL/<year>` and `/getMatchData/<id>`). Standard library only. Raw responses are cached in `data/raw/`, so reruns only fetch what's new (the current season is always refetched).
- `analyse_xg.py` (pandas + numpy) writes `data/analysis.md`, comparing xG against points and goal difference as predictors: first half of a season vs the second half, next-match prediction from rolling form trained on older seasons and tested on newer ones, and how repeatable each measure is.

```
python scrape_understat.py            # ~12 requests, a few seconds
python scrape_understat.py --shots    # + every shot, ~4,500 requests, ~75 min at 1 req/s
python analyse_xg.py
```

Outputs: `matches.csv`, `team_matches.csv` (xG, npxG, PPDA, deep passes, xPts per team per match), `players.csv` (per player per season), `shots.csv` (with `--shots`).

Notes
- Understat has no official API. These endpoints are what the site's own pages call, and the community libraries (soccerdata, understatapi) use them too. They can change without notice. Keep requests slow (default 1 s apart) and use the data for personal research.
- Understat's `forecast_*` columns are computed from the shots in that same match, so they are not a pre-match prediction.
- Rolling form needs 8 prior league games, so a promoted team's first 8 matches are left out of the next-match test.
