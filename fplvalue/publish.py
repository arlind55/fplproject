"""Write the projection outputs the site reads. Called at the end of scripts/update_fpl_data.py,
or by hand:  python -m fplvalue.publish

    data/current/projections.csv      one row per player: gw_<n> columns, next-GW breakdown, inputs
    data/current/captains.csv         top 40 by next-GW projection, with simulated ceiling / haul odds
    data/current/model_squad.csv      best £100m squad for the next five gameweeks
    data/current/projections_long.csv one row per player: gw_<n> columns for the next ten gameweeks (the My team page)
    data/current/projection_meta.json gameweeks, parameters, squad totals, backtest summary
    data/current/accuracy.json        live track record of past published projections (see accuracy.py)
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import accuracy, backtest
from .load import DATA, last_season, load
from .optimise import optimise
from .project import COMPONENTS, Params, project, state_from_snapshot
from .simulate import captain_table

warnings.simplefilter('ignore', category=FutureWarning)

LONG_HORIZON = 10
LONG_COLS = ['id', 'web_name', 'team', 'team_short_name', 'pos', 'price', 'selected_by_percent', 'status', 'avail_next', 'p_start']
PROJ_COLS = ['id', 'web_name', 'team_short_name', 'pos', 'price', 'selected_by_percent', 'status', 'next_opp',
             'avail_next', 'p_start', 'next_xmins', 'xg90', 'xa90', 'penalties_order', 'pens_taken', 'pen90', 'next_gw', 'next_n']


def write(out_dirs: list[Path] | None = None, with_backtest: bool = True, prm: Params | None = None) -> dict:
    out_dirs = out_dirs or [DATA / 'current']
    snap = load(out_dirs[0])
    last = last_season(snap)
    prm = prm or Params()
    st = state_from_snapshot(snap, last)
    proj, x = project(st, prm, detail=True)
    gws = proj.attrs['gws']
    gw_cols = [f'gw_{g}' for g in gws]
    comp_cols = [f'next_{c}' for c in COMPONENTS]
    P = proj[[c for c in PROJ_COLS if c in proj.columns] + gw_cols + comp_cols].copy()
    P = P[(P['next_n'] > 0.05) | (P['status'] == 'a')]
    P = P.round({c: 2 for c in P.columns if P[c].dtype.kind == 'f'}).round({'price': 1, 'selected_by_percent': 1})

    # Ten-gameweek projections for the My team page: same model, longer window, kept in its own file so
    # the five-week projections.csv (and the pages and track record built on it) are unchanged.
    lp = project(st, replace(prm, horizon=max(prm.horizon, LONG_HORIZON)))
    lgw = [f'gw_{g}' for g in lp.attrs['gws']]
    L = lp[[c for c in LONG_COLS if c in lp.columns] + lgw].copy()
    L['next_n'] = L[lgw].sum(1)
    L['news'] = L['id'].map(st.players.set_index('id')['news']).fillna('')     # every player, so any squad can be shown
    L = L.round({c: 2 for c in L.columns if L[c].dtype.kind == 'f'}).round({'price': 1, 'selected_by_percent': 1})

    cap = captain_table(proj, x, prm, top=40)
    cap = cap.round(3)

    sq, info = optimise(proj, 100.0)
    squad = sq[['id', 'web_name', 'team_short_name', 'pos', 'price', 'next_opp', 'next_gw', 'next_n', 'is_starter', 'is_captain']].round(2)

    meta = {'generated_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'gws': gws,
            'first_gw': gws[0], 'last_season_prior': last.path.name if last is not None else None,
            'params': asdict(prm), 'squad': info, 'long_gws': lp.attrs['gws']}
    if with_backtest:
        try:
            meta['backtest'] = backtest.summary_json(backtest.run(prm, verbose=False))
        except Exception as e:      # never let the backtest block the data refresh
            meta['backtest_error'] = str(e)
    try:
        acc = accuracy.to_json(accuracy.run(snap))
    except Exception as e:          # the track record is a nice-to-have, never block the refresh
        acc = {'error': str(e)}
    acc['generated_at'] = meta['generated_at']
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        P.to_csv(d / 'projections.csv', index=False)
        L.to_csv(d / 'projections_long.csv', index=False)
        cap.to_csv(d / 'captains.csv', index=False)
        squad.to_csv(d / 'model_squad.csv', index=False)
        (d / 'projection_meta.json').write_text(json.dumps(meta, indent=2))
        (d / 'accuracy.json').write_text(json.dumps(acc, indent=2))
    return meta


if __name__ == '__main__':
    m = write()
    print(json.dumps({k: v for k, v in m.items() if k != 'backtest'}, indent=2))
    print('backtest:', json.dumps(m.get('backtest', {}), indent=1)[:1500])
