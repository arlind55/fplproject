"""Live track record: score the projections the site actually published, week by week.

Unlike backtest.py (which rewinds the season and re-runs today's model), this only uses the
projections.csv each weekly snapshot kept, so nothing here can be tuned after the fact.

    python -m fplvalue.accuracy              # print the table
    data/current/accuracy.json               # written by publish.py for docs/accuracy.html

For each finished gameweek g, the forecast is the gw_<g> column of the latest snapshot dated before
g's deadline. FPL's ep_next from that same snapshot is the comparison, when it was for gameweek g.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .load import Snapshot, load, snapshot_dirs


def _forecast_for(gw: int, deadline: pd.Timestamp) -> tuple[str, pd.DataFrame] | None:
    best = None
    for p in snapshot_dirs():
        if pd.Timestamp(p.name, tz='UTC') >= deadline.normalize():
            break
        f = p / 'projections.csv'
        if f.exists() and f'gw_{gw}' in pd.read_csv(f, nrows=0).columns:
            best = p
    if best is None:
        return None
    proj = pd.read_csv(best / 'projections.csv')
    out = proj[['id', 'web_name', 'team_short_name', 'pos', 'price']].copy()
    out['model'] = proj[f'gw_{gw}']
    meta = json.loads((best / 'meta.json').read_text()) if (best / 'meta.json').exists() else {}
    if meta.get('next_event') == gw:
        players = pd.read_csv(best / 'players.csv', usecols=['id', 'ep_next'])
        out['fpl'] = out['id'].map(players.set_index('id')['ep_next'])
    else:
        out['fpl'] = np.nan
    return best.name, out


def _metrics(df: pd.DataFrame, col: str) -> dict:
    d = df.dropna(subset=[col])
    if not len(d):
        return {}
    err = d[col] - d['actual']
    top = d.nlargest(20, col)
    cap = d.nlargest(1, col).iloc[0]
    return {'mae': float(err.abs().mean()), 'bias': float(err.mean()),
            'spearman': float(d[col].corr(d['actual'], method='spearman')),
            'top20_actual': float(top['actual'].mean()),
            'captain': {'name': cap['web_name'], 'team': cap['team_short_name'],
                        'projected': float(cap[col]), 'actual': int(cap['actual'])}}


def run(cur: Snapshot | None = None) -> dict:
    cur = cur or load()
    actual = cur.history.groupby(['round', 'player_id'])['total_points'].sum() if len(cur.history) else pd.Series(dtype=float)
    weeks = []
    for gw in range(1, cur.last_finished_gw + 1):
        got = _forecast_for(gw, cur.deadline(gw))
        if got is None:
            continue
        snap_name, df = got
        pts = actual.loc[gw] if gw in actual.index.get_level_values(0) else pd.Series(dtype=float)
        df['actual'] = df['id'].map(pts).fillna(0)
        # score players who were expected to feature or did: ignores the long tail of 0-vs-0 bench fodder
        df = df[(df['model'] >= 1) | (df['actual'] != 0)]
        weeks.append({'gw': gw, 'snapshot': snap_name, 'n': int(len(df)),
                      'model': _metrics(df, 'model'), 'fpl': _metrics(df, 'fpl')})
    both = [w for w in weeks if w['fpl']]
    summary = {}
    if weeks:
        summary['weeks'] = len(weeks)
        summary['model_mae'] = float(np.mean([w['model']['mae'] for w in weeks]))
        summary['captain_points'] = int(sum(w['model']['captain']['actual'] for w in weeks))
    if both:
        summary['head_to_head_weeks'] = len(both)
        summary['model_wins'] = int(sum(w['model']['mae'] < w['fpl']['mae'] for w in both))
        summary['fpl_mae'] = float(np.mean([w['fpl']['mae'] for w in both]))
        summary['model_mae_same_weeks'] = float(np.mean([w['model']['mae'] for w in both]))
    return {'summary': summary, 'weeks': weeks}


def _round(o):
    if isinstance(o, float):
        return round(o, 3)
    if isinstance(o, dict):
        return {k: _round(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_round(v) for v in o]
    return o


def to_json(res: dict) -> dict:
    return _round(res)


if __name__ == '__main__':
    r = run()
    for w in r['weeks']:
        m, f = w['model'], w['fpl']
        fpl = f"FPL MAE {f['mae']:.2f}" if f else 'FPL MAE  -  '
        print(f"GW{w['gw']:>2} ({w['snapshot']}, n={w['n']})  model MAE {m['mae']:.2f}  {fpl}  "
              f"captain {m['captain']['name']} {m['captain']['projected']:.1f} -> {m['captain']['actual']}")
    print(json.dumps(to_json(r)['summary'], indent=2))
