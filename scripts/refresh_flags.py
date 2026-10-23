"""Quick between-weeks refresh: latest injury flags, news and prices from FPL, then re-run the projections.

The weekly update_fpl_data.py refresh runs on Tuesday, but most flag changes come out of the Thursday/Friday
press conferences. This only re-downloads bootstrap-static (one request), patches those columns into
data/current/players.csv and rewrites the projection outputs, so it takes well under a minute.
It only touches data/current: the dated history snapshots (and the track record built on them) stay weekly.

    python scripts/refresh_flags.py           # skips once the next deadline has passed
    python scripts/refresh_flags.py --force   # run anyway
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys

import pandas as pd
import requests
from utils import write_json

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
CURRENT_DIR = BASE_DIR / 'data' / 'current'
BOOTSTRAP_URL = 'https://fantasy.premierleague.com/api/bootstrap-static/'
HEADERS = {'User-Agent': 'fplproject-data-pipeline/2.0'}

# Columns that move between weekly refreshes without a match being played.
FLAG_COLS = ['status', 'chance_of_playing_next_round', 'chance_of_playing_this_round', 'news', 'news_added',
             'now_cost', 'cost_change_event', 'selected_by_percent', 'ep_next', 'ep_this']
# Only these trigger a rewrite (ownership drifts all day and would mean a commit every run).
WATCH = ['status', 'chance_of_playing_next_round', 'news', 'now_cost']


def main(force: bool = False) -> None:
    events = pd.read_csv(CURRENT_DIR / 'events.csv')
    nxt = events.loc[events['is_next'].astype(str).str.lower() == 'true']
    if len(nxt) and not force:
        deadline = pd.to_datetime(nxt['deadline_time'].iloc[0], utc=True)
        if deadline < pd.Timestamp.now(tz='UTC'):
            # The gameweek has started: the projections would still be for it until the weekly refresh
            # brings in its results, so leave everything as it is.
            print(f'GW{int(nxt["id"].iloc[0])} deadline has passed, nothing to do until the weekly refresh')
            return

    response = requests.get(BOOTSTRAP_URL, headers=HEADERS, timeout=30)
    response.raise_for_status()
    bootstrap = response.json()
    fresh = pd.DataFrame(bootstrap['elements']).set_index('id')
    fresh['now_cost'] = fresh['now_cost'] / 10

    players = pd.read_csv(CURRENT_DIR / 'players.csv')
    raw = pd.read_csv(CURRENT_DIR / 'players.csv', dtype=str, keep_default_na=False)   # written back as-is, so the diff
    before = players.set_index('id')[WATCH].copy()                                         # only shows real changes
    idx = players['id']
    for c in FLAG_COLS:
        if c not in fresh.columns or c not in players.columns:
            continue
        new = idx.map(fresh[c])
        if players[c].dtype.kind in 'if':
            new = pd.to_numeric(new, errors='coerce')
        old = players[c]
        diff = idx.isin(fresh.index) & ~((new == old) | (new.isna() & old.isna()))
        players.loc[diff, c] = new[diff]
        is_int = players[c].dtype.kind == 'i'
        raw.loc[diff, c] = new[diff].map(lambda v: '' if pd.isna(v) else str(int(v)) if is_int
                                         else str(float(v)) if isinstance(v, (int, float)) else str(v))

    after = players.set_index('id')[WATCH]
    changed = (before.fillna('').astype(str) != after.fillna('').astype(str)).any(axis=1)
    names = players.set_index('id')['web_name']
    print(f'{int(changed.sum())} players with changed flags, news or price')
    if not changed.any():
        return
    raw.to_csv(CURRENT_DIR / 'players.csv', index=False)
    write_json(CURRENT_DIR / 'bootstrap_static.json', bootstrap)
    for pid in changed[changed].index:
        print(f'  {names[pid]}: {before.at[pid, "status"]} -> {after.at[pid, "status"]}, '
              f'chance {after.at[pid, "chance_of_playing_next_round"]}, £{after.at[pid, "now_cost"]}m, {after.at[pid, "news"]!r}')

    meta_path = CURRENT_DIR / 'meta.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta['updated_at'] = meta['flags_updated_at'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    write_json(meta_path, meta)

    # The backtest only changes when a gameweek is played, so keep last week's result instead of re-running it.
    from fplvalue.publish import write as write_projections
    pm_path = CURRENT_DIR / 'projection_meta.json'
    old = json.loads(pm_path.read_text()) if pm_path.exists() else {}
    write_projections([CURRENT_DIR], with_backtest=False)
    if 'backtest' in old:
        pm = json.loads(pm_path.read_text())
        pm['backtest'] = old['backtest']
        pm_path.write_text(json.dumps(pm, indent=2))
    print('projections written')


if __name__ == '__main__':
    main(force='--force' in sys.argv)
