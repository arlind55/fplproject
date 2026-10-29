"""Replay the projection model over every season FPL has published xG for (2022/23 on).

Same walk-forward as fplvalue/backtest.py (rewind to before each gameweek, project it, compare with what
happened), on snapshots rebuilt from vaastav's archive by build_snapshots.py. Each season gets the previous
season as its prior, except 2022/23 (FPL had no xG in 2021/22, so it starts cold like a brand-new game).
No injury flags exist for past weeks, so the model assumes everyone is available.

Baselines, as in fplvalue/backtest.py: points/match, form (last 4), and FPL's own expected points `xP` as
archived by vaastav. xP is taken around the deadline, so it has seen the injury news the model has not.

    python build_snapshots.py && python season_backtest.py

Writes data/season_backtest.md.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from fplvalue import backtest as bt  # noqa: E402
from fplvalue.load import load  # noqa: E402

warnings.simplefilter('ignore')
SNAP = os.path.join(HERE, 'data', 'snapshots')
SEASONS = ['2022-23', '2023-24', '2024-25', '2025-26', '2026-27']
FIRST_GW = 5          # like fplvalue/backtest.py: the first weeks have too little this-season data to judge
CAPTAINS = 1


def season(label: str, prev: str | None) -> pd.DataFrame:
    snap = load(os.path.join(SNAP, label))
    last = load(os.path.join(SNAP, prev)) if prev else None
    played = int(snap.history['round'].max())
    first = 2 if label == SEASONS[-1] else FIRST_GW
    gws = [g for g in range(first, played + 1) if g in set(snap.events['id'])]   # 2022/23 GW7 was called off
    rows = bt.season_rows(snap, gws, last=last, use_snapshots=False)
    # vaastav's xP in round r is FPL's ep_next scraped after round r, i.e. FPL's projection for round r + 1
    # (it correlates 0.65 with round r's own points and 0.33 with round r + 1's), so shift it forward a week
    xp = snap.history.groupby(['player_id', 'round'])['xP'].first().rename('fpl_xp').reset_index()
    # some rounds were archived with every xP the same (0), i.e. not captured: leave those out
    flat = xp.groupby('round')['fpl_xp'].transform('std') < 0.01
    xp.loc[flat, 'fpl_xp'] = np.nan
    xp['round'] += 1
    rows = rows.join(xp.set_index(['player_id', 'round'])['fpl_xp'], on=['id', 'gw'])
    rows['season'] = label
    return rows


def metrics(d: pd.DataFrame, cols=('model', 'fpl_xp', 'ppm', 'form')) -> pd.DataFrame:
    out = {}
    for c in cols:
        q = d[d[c].notna()]
        err = q[c] - q['actual']
        top = q.sort_values(c, ascending=False).groupby(['season', 'gw'])
        sp = q.groupby(['season', 'gw']).apply(lambda g: g[c].corr(g['actual'], method='spearman'),
                                                include_groups=False).mean()
        out[c] = {'player-weeks': len(q), 'MAE': err.abs().mean(), 'RMSE': np.sqrt((err ** 2).mean()),
                  'bias': err.mean(), 'Spearman': sp, 'top-20 pts': top.head(20)['actual'].mean(),
                  'captain pts': top.head(CAPTAINS)['actual'].mean()}
    return pd.DataFrame(out).T


def boot_mse_gap(d: pd.DataFrame, a: str = 'model', b: str = 'fpl_xp', n: int = 2000, seed: int = 0):
    """MSE of a minus MSE of b, with a 90% bootstrap interval over gameweeks."""
    g = d.assign(da=(d[a] - d['actual']) ** 2, db=(d[b] - d['actual']) ** 2).groupby(['season', 'gw'])[['da', 'db']].agg(['sum', 'size'])
    sa, sb, cnt = g[('da', 'sum')].values, g[('db', 'sum')].values, g[('da', 'size')].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sa), (n, len(sa)))
    gaps = (sa[idx].sum(1) - sb[idx].sum(1)) / cnt[idx].sum(1)
    point = (sa.sum() - sb.sum()) / cnt.sum()
    return point, np.percentile(gaps, 5), np.percentile(gaps, 95)


def main():
    parts = []
    for i, s in enumerate(SEASONS):
        prev = SEASONS[i - 1] if i else None
        parts.append(season(s, prev))
        print('done', s, flush=True)
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(os.path.join(HERE, 'data', 'season_backtest_rows.csv.gz'), index=False)

    rep = ['# Projection model over four and a bit seasons', '',
           'Walk-forward replay of `fplvalue` on vaastav\'s FPL archive (see season_backtest.py). Regular starters',
           '(started at least 2 of their last 4 team matches), each gameweek from GW5 (GW2 this season).',
           '`fpl_xp` is FPL\'s own projection (ep_next) for that gameweek; it has seen injury news, the model here has not.', '']
    reg = df[df['regular'] & df['fpl_xp'].notna()]
    rep += ['FPL projections are missing for some weeks (most of 2025/26), so model and FPL are compared on the',
            f'{reg.groupby(["season", "gw"]).ngroups} gameweeks where both exist.', '']
    rep += ['## All seasons together, regular starters', '', metrics(reg).to_markdown(floatfmt='.3f'), '']
    p, lo, hi = boot_mse_gap(reg)
    rep += [f'Model MSE minus FPL xP MSE: **{p:+.3f}** (90% interval {lo:+.3f} to {hi:+.3f}; negative = model better).', '']

    rows = []
    for s, g in reg.groupby('season'):
        m = metrics(g)
        p, lo, hi = boot_mse_gap(g)
        rows.append({'season': s, 'weeks': g['gw'].nunique(),
                     'model RMSE': m.loc['model', 'RMSE'], 'FPL xP RMSE': m.loc['fpl_xp', 'RMSE'],
                     'MSE gap (90% CI)': f'{p:+.2f} ({lo:+.2f}, {hi:+.2f})',
                     'model Spearman': m.loc['model', 'Spearman'], 'FPL xP Spearman': m.loc['fpl_xp', 'Spearman'],
                     'model top-20': m.loc['model', 'top-20 pts'], 'FPL xP top-20': m.loc['fpl_xp', 'top-20 pts'],
                     'model captain': m.loc['model', 'captain pts'], 'FPL xP captain': m.loc['fpl_xp', 'captain pts'],
                     'model bias': m.loc['model', 'bias']})
    rep += ['## By season, regular starters', '', pd.DataFrame(rows).to_markdown(index=False, floatfmt='.2f'), '']

    rows = []
    for pos, g in reg.groupby('pos'):
        m = metrics(g)
        rows.append({'pos': pos, 'player-weeks': len(g), 'model RMSE': m.loc['model', 'RMSE'],
                     'FPL xP RMSE': m.loc['fpl_xp', 'RMSE'], 'model bias': m.loc['model', 'bias'],
                     'FPL xP bias': m.loc['fpl_xp', 'bias'], 'model Spearman': m.loc['model', 'Spearman'],
                     'FPL xP Spearman': m.loc['fpl_xp', 'Spearman']})
    rep += ['## By position, regular starters, all seasons', '', pd.DataFrame(rows).to_markdown(index=False, floatfmt='.3f'), '']

    reg = reg.assign(phase=pd.cut(reg['gw'], [0, 10, 20, 30, 38], labels=['GW2-10', 'GW11-20', 'GW21-30', 'GW31-38']))
    rows = []
    for ph, g in reg.groupby('phase', observed=True):
        m = metrics(g)
        rows.append({'phase': ph, 'player-weeks': len(g), 'model RMSE': m.loc['model', 'RMSE'],
                     'FPL xP RMSE': m.loc['fpl_xp', 'RMSE'], 'model Spearman': m.loc['model', 'Spearman'],
                     'FPL xP Spearman': m.loc['fpl_xp', 'Spearman']})
    rep += ['## By point in the season, regular starters', '', pd.DataFrame(rows).to_markdown(index=False, floatfmt='.3f'), '']

    rep += ['## Calibration, all players with a fixture, all seasons', '',
            bt.calibration(df).to_markdown(floatfmt='.2f'), '']
    with open(os.path.join(HERE, 'data', 'season_backtest.md'), 'w') as f:
        f.write('\n'.join(rep))
    print('\n'.join(rep))


if __name__ == '__main__':
    main()
