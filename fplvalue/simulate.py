"""Monte Carlo of next-gameweek points, for captaincy: the mean is the projection, but a captain pick
also cares about the upside (haul probability) and the downside (blank probability)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .project import CS_PTS, GOAL, Params


def simulate(x: pd.DataFrame, ids, prm: Params | None = None, n: int = 20000, seed: int = 7) -> pd.DataFrame:
    """x: the long per-fixture table from project(..., detail=True). Simulates the first gameweek in it
    for the given player ids. Returns mean, median, p90, P(>=10 pts) i.e. a captain 'haul', P(<=2)."""
    prm = prm or Params()
    gw = x['event'].min()
    d = x[(x['event'] == gw) & x['id'].isin(ids)].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    m = len(d)
    if not m:
        return pd.DataFrame()
    col = lambda c: d[c].to_numpy(float)[None, :]
    ps, pb, q60, ms, mb, av = (col(c) for c in ('p_start', 'p_sub', 'q60', 'm_start', 'm_sub', 'avail'))
    m60 = np.clip((ms - (1 - q60) * 45) / np.maximum(q60, 0.05), 60, 90)

    u = rng.random((n, m))
    start = u < av * ps
    sub = (~start) & (u < av * (ps + pb))
    full = start & (rng.random((n, m)) < q60)
    mins = np.where(full, m60, np.where(start, 45.0, np.where(sub, mb, 0.0)))
    played = mins > 0

    pos = d['pos'].to_numpy()
    gpts = np.array([GOAL[p] for p in pos])[None, :]
    cspts = np.array([CS_PTS[p] for p in pos])[None, :]
    gkdef = np.isin(pos, ['GKP', 'DEF'])[None, :]
    gk = (pos == 'GKP')[None, :]

    goals = rng.poisson(col('xg90') * prm.goal_cal * col('mult_att') * mins / 90)
    assists = rng.poisson(col('xa90') * prm.assist_cal * col('mult_att') * mins / 90)
    conc = rng.poisson(col('xga') * prm.cs_cal * mins / 90)
    saves = rng.poisson(col('saves90') * col('mult_def') * mins / 90)
    defcon = full & (rng.random((n, m)) < col('hit_rate'))
    cs_p = np.exp(-col('xga') * prm.cs_cal)
    nudge = np.where(gkdef, 0.6 + 0.4 * cs_p / max(cs_p.mean(), 1e-6), 0.6 + 0.4 * col('mult_att'))

    pts = (played * 1 + full * 1 + goals * gpts + assists * 3
           + (full & (conc == 0)) * cspts
           - np.where(gkdef & played, conc // 2, 0)
           + np.where(gk, saves // 3, 0)
           + defcon * 2
           + col('bonus90') * mins / 90 * nudge
           - col('yc90') * mins / 90)
    tot = pd.DataFrame(pts.T, index=d['id']).groupby(level=0).sum().T.to_numpy()   # sum double gameweeks
    ids_out = sorted(d['id'].unique())
    return pd.DataFrame({
        'id': ids_out,
        'sim_mean': tot.mean(0),
        'sim_median': np.median(tot, 0),
        'p90': np.percentile(tot, 90, axis=0),
        'haul_prob': (tot >= 10).mean(0),
        'blank_prob': (tot <= 2).mean(0),
    })


def captain_table(proj: pd.DataFrame, x: pd.DataFrame, prm: Params | None = None, top: int = 40) -> pd.DataFrame:
    """Captain candidates: the top `top` players by next-GW projection, with simulated upside."""
    cand = proj.sort_values('next_gw', ascending=False).head(top)
    sim = simulate(x, cand['id'], prm)
    keep = ['id', 'web_name', 'team_short_name', 'pos', 'price', 'selected_by_percent', 'next_opp', 'avail_next', 'next_gw']
    out = cand[[c for c in keep if c in cand.columns]].merge(sim, on='id', how='left')
    return out.sort_values('next_gw', ascending=False).reset_index(drop=True)
