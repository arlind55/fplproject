"""Turn vaastav/Fantasy-Premier-League season folders into end-of-season snapshots fplvalue.load() can read.

vaastav (https://github.com/vaastav/Fantasy-Premier-League) keeps every season's FPL data: players_raw.csv
(bootstrap-static elements at season end), teams.csv, fixtures.csv and gws/merged_gw.csv (one row per
player per fixture, i.e. element-summary history, plus FPL's own expected points `xP`). FPL only started
publishing expected goals in 2022/23, so that is the first season the projection model can replay.

    python build_snapshots.py      # downloads into data/raw/<season>/ on the first run, writes data/snapshots/<season>/

Each snapshot has players.csv, teams.csv, fixtures.csv, events.csv (deadlines rebuilt as first kick-off minus
90 minutes) and player_history.csv. FPL's xP is kept in player_history.csv as `xP` for the backtest.
"""
from __future__ import annotations

import os
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'data', 'raw')
OUT = os.path.join(HERE, 'data', 'snapshots')
URL = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{}/{}'
SEASONS = ['2022-23', '2023-24', '2024-25', '2025-26', '2026-27']
FILES = {'players_raw.csv': 'players_raw.csv', 'teams.csv': 'teams.csv', 'fixtures.csv': 'fixtures.csv',
         'merged_gw.csv': 'gws/merged_gw.csv'}


def fetch(season: str, name: str) -> str:
    path = os.path.join(RAW, season, name)
    if not os.path.exists(path) or season == SEASONS[-1]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        req = urllib.request.Request(URL.format(season, FILES[name]), headers={'User-Agent': 'fplvalue-research'})
        with urllib.request.urlopen(req, timeout=120) as r, open(path, 'wb') as f:
            f.write(r.read())
        time.sleep(0.5)
    return path


def build(season: str) -> str:
    players = pd.read_csv(fetch(season, 'players_raw.csv'))
    teams = pd.read_csv(fetch(season, 'teams.csv'))
    fixtures = pd.read_csv(fetch(season, 'fixtures.csv'))
    h = pd.read_csv(fetch(season, 'merged_gw.csv'), encoding='utf-8', on_bad_lines='skip')

    h = h.rename(columns={'element': 'player_id'})
    h['element'] = h['player_id']
    for c in ('starts', 'defensive_contribution', 'expected_goals', 'expected_assists',
              'expected_goal_involvements', 'expected_goals_conceded'):
        if c not in h.columns:
            h[c] = 0
    h = h.drop(columns=['GW', 'name', 'position', 'team'], errors='ignore')
    h = h.sort_values(['player_id', 'kickoff_time']).drop_duplicates(['player_id', 'fixture'])

    f = fixtures.copy()
    f['kickoff_time'] = pd.to_datetime(f['kickoff_time'], utc=True)
    ev = f.dropna(subset=['event']).groupby('event')['kickoff_time'].min()
    events = pd.DataFrame({'id': ev.index.astype(int),
                           'deadline_time': (ev - pd.Timedelta(minutes=90)).dt.strftime('%Y-%m-%dT%H:%M:%SZ').values})
    last_played = int(h['round'].max())
    events['finished'] = events['id'] <= last_played
    events['is_next'] = events['id'] == last_played + 1
    f['kickoff_time'] = f['kickoff_time'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    out = os.path.join(OUT, season)
    os.makedirs(out, exist_ok=True)
    players.to_csv(os.path.join(out, 'players.csv'), index=False)
    teams.to_csv(os.path.join(out, 'teams.csv'), index=False)
    f.to_csv(os.path.join(out, 'fixtures.csv'), index=False)
    events.to_csv(os.path.join(out, 'events.csv'), index=False)
    h.to_csv(os.path.join(out, 'player_history.csv'), index=False)
    print(f'{season}: {len(players)} players, {len(h)} history rows, GW1-{last_played}', flush=True)
    return out


if __name__ == '__main__':
    for s in SEASONS:
        build(s)
