"""Retune the projection model's settings on three full seasons instead of one.

Coordinate descent over fplvalue.project.Params (plus the team model's half-life and prior), scored by the
walk-forward backtest (season_backtest.py's replay) on 2023/24, 2024/25 and 2025/26, each with the previous
season as its prior. 2022/23 is left out of tuning because it has no prior season (a cold start the live
site never has). A change is kept only if it lowers the pooled squared error AND helps in at least two of the
three seasons, so one season's quirks cannot pull a setting around.

Objective: mean squared error of projected v actual FPL points over every player-week (GW5 on), the same
as research/feature_ablation.py. Seasons run in parallel.

Held out: 2026/27 so far, replayed from our own snapshots (with injury flags) via fplvalue.backtest.

    python build_snapshots.py && python tune_params.py      # ~30-60 minutes on 4 cores

Writes data/tune_params.md.
"""
from __future__ import annotations

import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from fplvalue import backtest as bt, team_model  # noqa: E402
from fplvalue.load import load  # noqa: E402
from fplvalue.project import Params  # noqa: E402

warnings.simplefilter('ignore')
SNAP = os.path.join(HERE, 'data', 'snapshots')
TUNE = [('2023-24', '2022-23'), ('2024-25', '2023-24'), ('2025-26', '2024-25')]
COLD = ('2022-23', None)
FIRST_GW = 5

# candidate values per setting; goal_cal is tuned as two scale factors (defensive / attacking positions)
GRID = {
    'k_min': [0.25, 0.5, 1.0, 2.0],
    'decay': [0.5, 0.6, 0.7, 0.8, 0.9],
    'n_recent': [4, 6, 8, 10],
    'lam': [0.25, 0.5, 0.75, 1.0],
    'k_xg': [2.0, 4.0, 6.0, 9.0, 14.0],
    'k_xa': [2.0, 4.0, 6.0, 9.0, 14.0],
    'k_saves': [3.0, 6.0, 10.0, 16.0],
    'k_bonus': [4.0, 8.0, 12.0, 20.0],
    'k_cards': [5.0, 10.0, 20.0, 40.0],
    'cs_cal': [0.9, 0.95, 1.0, 1.05, 1.1],
    'assist_cal': [1.2, 1.3, 1.38, 1.45, 1.55],
    'goal_cal_def': [0.75, 0.85, 0.95],
    'goal_cal_att': [0.95, 1.0, 1.04, 1.1],
    'HALF_LIFE': [10, 15, 20, 30],
    'PRIOR_GAMES': [1, 2, 4],
}


def make(cfg: dict) -> Params:
    p = Params()
    kw = {k: v for k, v in cfg.items() if k in Params.__dataclass_fields__}
    gc = dict(p.goal_cal)
    if 'goal_cal_def' in cfg:
        gc.update(GKP=cfg['goal_cal_def'], DEF=cfg['goal_cal_def'])
    if 'goal_cal_att' in cfg:
        f = cfg['goal_cal_att'] / 1.04          # keep the current MID/FWD gap
        gc.update(MID=round(1.03 * f, 3), FWD=round(1.04 * f, 3))
    return replace(p, goal_cal=gc, **kw)


def _apply_team(cfg: dict):
    team_model.HALF_LIFE = cfg.get('HALF_LIFE', 15)
    team_model.PRIOR_GAMES = cfg.get('PRIOR_GAMES', 2)
    team_model.team_ratings.__defaults__ = (None, team_model.HALF_LIFE, None)   # half_life is bound at def time


_CACHE: dict = {}


def _snap(label):
    if label not in _CACHE:
        _CACHE[label] = load(os.path.join(SNAP, label)) if label else None
    return _CACHE[label]


def season_sse(args) -> tuple[float, int]:
    (label, prev), cfg = args
    _apply_team(cfg)
    snap = _snap(label)
    gws = [g for g in range(FIRST_GW, int(snap.history['round'].max()) + 1) if g in set(snap.events['id'])]
    r = bt.season_rows(snap, gws, make(cfg), last=_snap(prev), use_snapshots=False)
    return float(((r['model'] - r['actual']) ** 2).sum()), len(r)


def evaluate(pool, cfg: dict, seasons=TUNE) -> np.ndarray:
    """Per-season MSE."""
    out = list(pool.map(season_sse, [(s, cfg) for s in seasons]))
    return np.array([sse / n for sse, n in out])


def holdout(cfg: dict) -> dict:
    _apply_team(cfg)
    cur = load()
    last = bt.last_season(cur)
    rows = bt.season_rows(cur, range(2, cur.last_finished_gw + 1), make(cfg), last=last)
    m = bt.metrics(rows)
    reg = bt.metrics(rows[rows['regular']])
    return {'MSE': float(((rows['model'] - rows['actual']) ** 2).mean()), 'RMSE regulars': reg.loc['model', 'rmse'],
            'Spearman regulars': reg.loc['model', 'spearman'], 'top-20 pts': m.loc['model', 'top20_actual']}


def main():
    cfg = {'k_min': 0.5, 'decay': 0.7, 'n_recent': 6, 'lam': 0.5, 'k_xg': 4.0, 'k_xa': 4.0, 'k_saves': 6.0,
           'k_bonus': 8.0, 'k_cards': 10.0, 'cs_cal': 1.0, 'assist_cal': 1.38, 'goal_cal_def': 0.85,
           'goal_cal_att': 1.04, 'HALF_LIFE': 15, 'PRIOR_GAMES': 2}
    start = dict(cfg)
    log = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        base = evaluate(pool, cfg)
        print('baseline', base.round(4), base.mean().round(4), flush=True)
        for rnd in (1, 2):
            changed = False
            for k, vals in GRID.items():
                best_k, best_v = base, cfg[k]
                for v in vals:
                    if v == cfg[k]:
                        continue
                    trial = evaluate(pool, {**cfg, k: v})
                    better = trial.mean() < best_k.mean() and (trial < base).sum() >= 2
                    log.append({'round': rnd, 'setting': k, 'value': v, 'from': cfg[k],
                                'pooled MSE change %': 100 * (trial.mean() / base.mean() - 1),
                                **{f'{s} %': 100 * (t / b - 1) for (s, _), t, b in zip(TUNE, trial, base)}})
                    print(rnd, k, v, (100 * (trial / base - 1)).round(3), flush=True)
                    if better:
                        best_k, best_v = trial, v
                if best_v != cfg[k]:
                    print(f'  -> {k}: {cfg[k]} -> {best_v}', flush=True)
                    cfg[k], base, changed = best_v, best_k, True
            if not changed:
                break
        start_mse = evaluate(pool, start)
        cold = [evaluate(pool, c, seasons=[COLD])[0] for c in (start, cfg)]

    ho = pd.DataFrame({'current': holdout(start), 'tuned': holdout(cfg)}).T
    changes = pd.DataFrame([{'setting': k, 'current': start[k], 'tuned': cfg[k]} for k in cfg if cfg[k] != start[k]])
    per = pd.DataFrame({'season': [s for s, _ in TUNE], 'current MSE': start_mse, 'tuned MSE': base,
                        'change %': 100 * (base / start_mse - 1)})
    rep = ['# Retuning the projection settings on three seasons', '',
           'See tune_params.py. Tuned on 2023/24, 2024/25 and 2025/26 (GW5 on, each with last season as prior);',
           'a change had to lower the pooled error and help in at least two of the three seasons.', '',
           '## Changes', '', changes.to_markdown(index=False) if len(changes) else 'None.', '',
           '## Tuning seasons', '', per.to_markdown(index=False, floatfmt='.3f'), '',
           f'2022/23 (cold start, not tuned on): MSE {cold[0]:.3f} -> {cold[1]:.3f} '
           f'({100 * (cold[1] / cold[0] - 1):+.2f}%).', '',
           '## Held out: 2026/27 so far (our own snapshots, with injury flags)', '',
           ho.to_markdown(floatfmt='.3f'), '',
           '## Every trial', '', pd.DataFrame(log).to_markdown(index=False, floatfmt='.3f'), '']
    with open(os.path.join(HERE, 'data', 'tune_params.md'), 'w') as f:
        f.write('\n'.join(rep))
    print('\n'.join(rep[:40]))


if __name__ == '__main__':
    main()
