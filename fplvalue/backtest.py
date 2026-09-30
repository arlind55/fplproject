"""Walk-forward backtest: rewind a season to before gameweek k, project GW k, compare to what happened.

    python -m fplvalue.backtest                  # 2025/26 (GW5-38) and this season so far
    python -m fplvalue.backtest --json out.json  # also write the summary for the site

Baselines, all scaled by the number of fixtures the player's team has that week:
    points/match   season points ÷ team matches so far (0-minute matches count)
    form           mean points over the last 4 team matches
    FPL ep_next    FPL's own projection, where a snapshot from before that deadline exists
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

from .load import DATA, Snapshot, load, last_season, snapshot_before
from .project import Params, project, state_as_of

warnings.simplefilter('ignore', category=FutureWarning)


def season_rows(final: Snapshot, gws, prm: Params | None = None, last: Snapshot | None = None,
                use_snapshots: bool = True) -> pd.DataFrame:
    prm = prm or Params()
    start = final.season_start
    rows = []
    for gw in gws:
        dl = final.deadline(gw)
        avail = None
        if use_snapshots:
            p = snapshot_before(dl, after=start - pd.Timedelta(days=30))
            # only trust a snapshot taken within the week before the deadline
            if p is not None and (dl.normalize() - pd.Timestamp(p.name, tz='UTC')).days <= 7:
                avail = load(p)
        st = state_as_of(final, gw, avail=avail, last=last)
        pr = project(st, replace(prm, horizon=1), gws=[gw]).set_index('id')
        h = st.history
        nfix = st.fixtures[st.fixtures['event'] == gw]
        nfix = pd.concat([nfix['team_h'], nfix['team_a']]).value_counts()
        team_n = st.players.set_index('id')['team'].map(nfix).fillna(0)
        g = h.sort_values('kickoff_time').groupby('player_id')
        ppm = g['total_points'].sum() / g.size()
        form = g.tail(4).groupby('player_id')['total_points'].mean()
        reg = g.tail(4).groupby('player_id')['starts'].mean() >= 0.5
        act = final.history[final.history['round'] == gw].groupby('player_id').agg(
            actual=('total_points', 'sum'), mins=('minutes', 'sum'))
        df = pd.DataFrame({
            'gw': gw, 'model': pr['next_gw'],
            'ppm': ppm.reindex(pr.index).fillna(0) * team_n,
            'form': form.reindex(pr.index).fillna(0) * team_n,
            'ep_next': pr['ep_next'] if avail is not None else np.nan,
            'regular': reg.reindex(pr.index).fillna(False),
            'price': pr['price'], 'pos': pr['pos'], 'web_name': pr['web_name'],
        }).join(act).fillna({'actual': 0, 'mins': 0})
        rows.append(df.reset_index())
    return pd.concat(rows, ignore_index=True)


def metrics(df: pd.DataFrame, cols=('model', 'ppm', 'form', 'ep_next'), top: int = 20) -> pd.DataFrame:
    out = {}
    for c in cols:
        d = df[df[c].notna()]
        if not len(d):
            continue
        err = d[c] - d['actual']
        # top-N: average actual points of each week's N highest-projected players
        topn = d.sort_values(c, ascending=False).groupby('gw').head(top)['actual'].mean()
        sp = d.groupby('gw').apply(lambda g: g[c].corr(g['actual'], method='spearman'), include_groups=False).mean()
        out[c] = {'weeks': int(d['gw'].nunique()), 'mae': err.abs().mean(), 'rmse': np.sqrt((err ** 2).mean()),
                  'bias': err.mean(), 'spearman': sp, f'top{top}_actual': topn}
    return pd.DataFrame(out).T


def compare(df: pd.DataFrame) -> dict:
    """Metrics on all players and on regular starters, plus model vs ep_next on the weeks both exist."""
    res = {'all': metrics(df), 'regulars': metrics(df[df['regular']])}
    both = df[df['ep_next'].notna()]
    if len(both):
        res['vs_fpl'] = metrics(both[both['regular']], cols=('model', 'ep_next', 'form'))
    return res


def calibration(df: pd.DataFrame, col: str = 'model') -> pd.DataFrame:
    b = pd.cut(df[col], [-1, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 7, 20])
    return df.groupby(b, observed=True).agg(n=('actual', 'size'), projected=(col, 'mean'), actual=('actual', 'mean'))


def run(prm: Params | None = None, verbose: bool = True) -> dict:
    cur = load()
    last = last_season(cur)
    out = {}
    if last is not None:
        rows = season_rows(last, range(5, 39), prm)
        out['last_season'] = {'label': f"{last.season_start.year}/{str(last.season_start.year + 1)[2:]}",
                              'rows': rows, **compare(rows)}
    fin = cur.last_finished_gw
    if fin >= 2:
        rows = season_rows(cur, range(2, fin + 1), prm, last=last)
        out['this_season'] = {'label': f"{cur.season_start.year}/{str(cur.season_start.year + 1)[2:]}",
                              'rows': rows, **compare(rows)}
    if verbose:
        pd.set_option('display.width', 200)
        for k, v in out.items():
            print(f"\n=== {v['label']} ({k}) ===")
            for part in ('all', 'regulars', 'vs_fpl'):
                if part in v:
                    print(f'-- {part}'); print(v[part].round(3).to_string())
            print('-- calibration (all players)'); print(calibration(v['rows']).round(2).to_string())
    return out


def summary_json(res: dict) -> dict:
    """Compact version for the site."""
    js = {}
    for k, v in res.items():
        js[k] = {'label': v['label']}
        for part in ('regulars', 'vs_fpl'):
            if part in v:
                js[k][part] = {m: {kk: (round(float(vv), 3) if pd.notna(vv) else None) for kk, vv in row.items()}
                               for m, row in v[part].iterrows()}
    return js


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', help='write a summary JSON here')
    a = ap.parse_args()
    r = run()
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(summary_json(r), f, indent=2)
        print('wrote', a.json)
