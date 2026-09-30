"""Write the projection outputs the site reads. Called at the end of scripts/update_fpl_data.py,
or by hand:  python -m fplvalue.publish

    data/current/projections.csv      one row per player: gw_<n> columns, next-GW breakdown, inputs
    data/current/captains.csv         top 40 by next-GW projection, with simulated ceiling / haul odds
    data/current/model_squad.csv      best £100m squad for the next five gameweeks
    data/current/projection_meta.json gameweeks, parameters, squad totals, backtest summary
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import backtest
from .load import DATA, last_season, load
from .optimise import optimise
from .project import COMPONENTS, Params, project, state_from_snapshot
from .simulate import captain_table

warnings.simplefilter('ignore', category=FutureWarning)

PROJ_COLS = ['id', 'web_name', 'team_short_name', 'pos', 'price', 'selected_by_percent', 'status', 'next_opp',
             'avail_next', 'p_start', 'next_xmins', 'xg90', 'xa90', 'next_gw', 'next_n']


def write(out_dirs: list[Path] | None = None, with_backtest: bool = True, prm: Params | None = None) -> dict:
    out_dirs = out_dirs or [DATA / 'current']
    snap = load(out_dirs[0])
    last = last_season(snap)
    prm = prm or Params()
    proj, x = project(state_from_snapshot(snap, last), prm, detail=True)
    gws = proj.attrs['gws']
    gw_cols = [f'gw_{g}' for g in gws]
    comp_cols = [f'next_{c}' for c in COMPONENTS]
    P = proj[[c for c in PROJ_COLS if c in proj.columns] + gw_cols + comp_cols].copy()
    P = P[(P['next_n'] > 0.05) | (P['status'] == 'a')]
    P = P.round({c: 2 for c in P.columns if P[c].dtype.kind == 'f'}).round({'price': 1, 'selected_by_percent': 1})

    cap = captain_table(proj, x, prm, top=40)
    cap = cap.round(3)

    sq, info = optimise(proj, 100.0)
    squad = sq[['id', 'web_name', 'team_short_name', 'pos', 'price', 'next_opp', 'next_gw', 'next_n', 'is_starter', 'is_captain']].round(2)

    meta = {'generated_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'gws': gws,
            'first_gw': gws[0], 'last_season_prior': last.path.name if last is not None else None,
            'params': asdict(prm), 'squad': info}
    if with_backtest:
        try:
            meta['backtest'] = backtest.summary_json(backtest.run(prm, verbose=False))
        except Exception as e:      # never let the backtest block the data refresh
            meta['backtest_error'] = str(e)
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        P.to_csv(d / 'projections.csv', index=False)
        cap.to_csv(d / 'captains.csv', index=False)
        squad.to_csv(d / 'model_squad.csv', index=False)
        (d / 'projection_meta.json').write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == '__main__':
    m = write()
    print(json.dumps({k: v for k, v in m.items() if k != 'backtest'}, indent=2))
    print('backtest:', json.dumps(m.get('backtest', {}), indent=1)[:1500])
