"""Squad optimiser (integer programme, solved with scipy's HiGHS).

Best squad from scratch:
    python -m fplvalue.optimise                       # £100.0m, next 5 GWs
Best transfers for your team (reads your public picks from the FPL API):
    python -m fplvalue.optimise --team-id 1234567 --ft 1
    python -m fplvalue.optimise --team-id 1234567 --ft 2 --max-transfers 3

Objective: starting XI × (discounted next-N projection) + 0.15 × bench + captain × next-GW projection
           − 4 points per transfer beyond the free ones.
Constraints: 2 GK / 5 DEF / 5 MID / 3 FWD, max 3 per club, budget, XI of 1 GK, ≥3 DEF, ≥2 MID, ≥1 FWD.
Note: selling prices aren't public, so squads are valued at current prices; your in-game bank may differ slightly.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

SQUAD = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
XI_MIN = {'GKP': 1, 'DEF': 3, 'MID': 2, 'FWD': 1}
XI_MAX = {'GKP': 1, 'DEF': 5, 'MID': 5, 'FWD': 3}
BENCH_W = 0.15


def optimise(proj: pd.DataFrame, budget: float = 100.0, value_col: str = 'next_n_discounted',
             current: list[int] | None = None, free_transfers: int = 1, hit: float = 4.0,
             max_transfers: int | None = None, exclude: list[int] | None = None, include: list[int] | None = None):
    """Returns (squad DataFrame with is_starter / is_captain flags, info dict)."""
    p = proj.copy()
    p = p[~p['status'].isin(['u', 'n'])] if 'status' in p.columns else p
    if exclude:
        p = p[~p['id'].isin(exclude)]
    if current:
        keep = proj[proj['id'].isin(current)]
        p = pd.concat([p, keep]).drop_duplicates('id')
    # prune: each position's top players by value and by value per £ is plenty of candidates
    parts = []
    for pos, g in p.groupby('pos'):
        g = g.assign(vpm=g[value_col] / g['price'])
        idx = set(g.nlargest(45, value_col).index) | set(g.nlargest(30, 'vpm').index) | set(g.nsmallest(6, 'price').index)
        parts.append(g.loc[sorted(idx)])
    p = pd.concat(parts)
    if current:
        p = pd.concat([p, proj[proj['id'].isin(current)]]).drop_duplicates('id')
    if include:
        p = pd.concat([p, proj[proj['id'].isin(include)]]).drop_duplicates('id')
    p = p.reset_index(drop=True)
    n = len(p)
    v = p[value_col].to_numpy(float)
    v1 = p['next_gw'].to_numpy(float)
    price = p['price'].to_numpy(float)
    cur = p['id'].isin(current or []).to_numpy(float)

    # variables: x (squad) | y (starter) | c (captain) | e (extra transfers, integer)
    N = 3 * n + 1
    obj = np.zeros(N)
    obj[:n] = -BENCH_W * v
    obj[n:2 * n] = -(1 - BENCH_W) * v
    obj[2 * n:3 * n] = -v1
    obj[3 * n] = hit
    A, lo, hi = [], [], []

    def row(coefs, l, h):
        r = np.zeros(N)
        for sl, val in coefs:
            r[sl] = val
        A.append(r); lo.append(l); hi.append(h)

    X, Y, C = slice(0, n), slice(n, 2 * n), slice(2 * n, 3 * n)
    row([(X, 1)], 15, 15)
    row([(Y, 1)], 11, 11)
    row([(C, 1)], 1, 1)
    for pos, k in SQUAD.items():
        m = (p['pos'] == pos).to_numpy(float)
        row([(X, m)], k, k)
        row([(Y, m)], XI_MIN[pos], XI_MAX[pos])
    for t in p['team'].unique():
        row([(X, (p['team'] == t).to_numpy(float))], 0, 3)
    for i in range(n):          # y <= x, c <= y
        r = np.zeros(N); r[n + i] = 1; r[i] = -1; A.append(r); lo.append(-np.inf); hi.append(0)
        r = np.zeros(N); r[2 * n + i] = 1; r[n + i] = -1; A.append(r); lo.append(-np.inf); hi.append(0)
    row([(X, price)], 0, budget + 1e-6)
    if current:
        # transfers = 15 − |kept| ; e >= transfers − free
        r = np.zeros(N); r[:n] = -cur; r[3 * n] = -1
        A.append(r); lo.append(-np.inf); hi.append(free_transfers - 15)
        if max_transfers is not None:
            row([(X, cur)], 15 - max_transfers, 15)
    if include:
        row([(X, p['id'].isin(include).to_numpy(float))], len(include), len(include))

    integrality = np.ones(N)
    ub = np.ones(N); ub[3 * n] = 15
    res = milp(obj, constraints=LinearConstraint(np.array(A), lo, hi), integrality=integrality,
               bounds=Bounds(np.zeros(N), ub), options={'time_limit': 60})
    if res.x is None:
        raise RuntimeError(f'optimiser failed: {res.message}')
    s = np.round(res.x).astype(int)
    sq = p[s[:n] == 1].copy()
    sq['is_starter'] = s[n:2 * n][s[:n] == 1] == 1
    sq['is_captain'] = s[2 * n:3 * n][s[:n] == 1] == 1
    order = {'GKP': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
    sq = sq.sort_values(['is_starter', 'pos', value_col], key=lambda c: c.map(order) if c.name == 'pos' else c,
                        ascending=[False, True, False])
    info = {'cost': round(float(sq['price'].sum()), 1), 'budget': budget,
            'xi_value': round(float(sq.loc[sq['is_starter'], value_col].sum()), 1),
            'xi_next_gw': round(float(sq.loc[sq['is_starter'], 'next_gw'].sum() + sq.loc[sq['is_captain'], 'next_gw'].sum()), 1),
            'extra_transfers': int(s[3 * n])}
    if current:
        info['out'] = sorted(set(current) - set(sq['id']))
        info['in'] = sorted(set(sq['id']) - set(current))
    return sq, info


def fetch_team(team_id: int, gw: int):
    import requests
    h = {'User-Agent': 'fplvalue/1.0'}
    picks = requests.get(f'https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/', headers=h, timeout=30).json()
    ids = [x['element'] for x in picks['picks']]
    bank = picks.get('entry_history', {}).get('bank', 0) / 10
    return ids, bank


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--team-id', type=int)
    ap.add_argument('--ft', type=int, default=1, help='free transfers available')
    ap.add_argument('--max-transfers', type=int)
    ap.add_argument('--budget', type=float, default=100.0)
    ap.add_argument('--horizon', type=int, default=5)
    a = ap.parse_args()

    import fplvalue as fv
    snap = fv.load()
    proj = fv.project_snapshot(snap, fv.last_season(snap), fv.Params(horizon=a.horizon))
    cols = ['web_name', 'team_short_name', 'pos', 'price', 'next_opp', 'next_gw', 'next_n', 'is_starter', 'is_captain']
    pd.set_option('display.width', 160)
    if a.team_id:
        ids, bank = fetch_team(a.team_id, snap.last_finished_gw)
        budget = round(bank + proj[proj['id'].isin(ids)]['price'].sum(), 1)
        name = proj.set_index('id')['web_name']
        for k in range(0, (a.max_transfers if a.max_transfers is not None else a.ft + 1) + 1):
            sq, info = optimise(proj, budget, current=ids, free_transfers=a.ft, max_transfers=k)
            gain = info['xi_value']
            print(f"\n{k} transfer(s): XI value {gain} (hits: {info['extra_transfers']})  "
                  f"OUT {[name[i] for i in info['out']]}  IN {[name[i] for i in info['in']]}")
        print(sq[cols].round(2).to_string(index=False))
    else:
        sq, info = optimise(proj, a.budget)
        print(sq[cols].round(2).to_string(index=False)); print(info)


if __name__ == '__main__':
    main()
