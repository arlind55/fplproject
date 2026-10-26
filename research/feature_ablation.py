"""Which model inputs earn their place? Ablation backtest of the player-points projection.

Each run removes or neutralises one input, then replays the same walk-forward backtest as
fplvalue/backtest.py (rewind to before each deadline, project that gameweek, score against what
happened). The drop in accuracy against the full model is that input's contribution.

    python research/feature_ablation.py              # writes research/feature_ablation.md

Seasons: 2025/26 GW5-38 (no prior season in the data, so the last-season input is idle there) and
2026/27 GW2 to the last finished gameweek (last season carried in at half weight, as live).
FPL flags only exist where a snapshot from the week before the deadline was kept, so that input is
scored on those weeks only. Bookmaker odds and news overrides are live-only (no history of either is
stored), so they cannot be replayed here.
"""
from __future__ import annotations

import contextlib
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fplvalue.project  # noqa: E402,F401
pj = sys.modules["fplvalue.project"]   # the package re-exports project(), which shadows the module
from fplvalue.backtest import season_rows  # noqa: E402
from fplvalue.load import load, last_season, snapshot_before  # noqa: E402
from fplvalue.project import Params  # noqa: E402

OUT = Path(__file__).with_suffix('.md')
BIG = 1e6       # shrinkage weight that leaves only the prior
RNG = np.random.default_rng(0)


# ── neutralisations: each is a context manager that patches the model for the duration of a run ────
@contextlib.contextmanager
def patch(obj, name, fn):
    old = getattr(obj, name)
    setattr(obj, name, fn(old))
    try:
        yield
    finally:
        setattr(obj, name, old)


def no_pens(_):
    """Penalty model off: no team penalty rate, so the pens component is zero (xG stays non-penalty)."""
    return lambda state, prm: pd.Series(0.0, index=state.teams['id'].values)


def keep_pen_xg(_):
    """The older approach: total xG (penalties left in) and no separate penalty-taker model."""
    return lambda h, season: h.assign(pens=0.0, pen_xg=0.0)


def flat_prior(_):
    """Price-based prior replaced by a flat per-position average (slope 0)."""
    def fit(tab, y, w, min_w):
        out = {}
        for pos in ('GKP', 'DEF', 'MID', 'FWD'):
            g = tab[(tab['pos'] == pos) & (tab[w] >= min_w) & tab[y].notna()]
            g = g if len(g) else tab[(tab['pos'] == pos) & tab[y].notna()]
            out[pos] = (float(np.average(g[y], weights=g[w])) if len(g) and g[w].sum() > 0 else 0.0, 0.0)
        return out
    return fit


def fixtures_neutral(orig, keep_team=True):
    """Opponent and home/away ignored: every fixture is played against an average side at a neutral venue.
    keep_team=False also drops the team's own attack/defence rating (every team = league average)."""
    def tf(state, gws):
        t, avg = orig(state, gws)
        t = t.copy()
        if keep_team:
            t['xgf'] = t['xgf'] / t['mult_att']
            t['xga'] = t['xga'] / t['mult_def']
        else:
            t['xgf'] = t['xga'] = avg
        t['mult_att'] = t['mult_def'] = 1.0
        return t, avg
    return tf


# name, label, Params changes, patches, run with last season, flags blanked
ABLATIONS = [
    ('full', 'Full model', {}, [], True, False),
    ('xg', 'Open-play xG/90 (price prior only)', {'k_xg': BIG}, [], True, False),
    ('xa', 'xA/90 (price prior only)', {'k_xa': BIG}, [], True, False),
    ('xgxa', 'xG and xA both (price prior only)', {'k_xg': BIG, 'k_xa': BIG}, [], True, False),
    ('price', 'Price-based prior (flat positional average)', {}, [('_fit_linear', flat_prior)], True, False),
    ('lam', 'Last season at half weight (λ = 0)', {'lam': 0.0}, [], True, False),
    ('nolast', 'Last season entirely (rates, minutes, team model)', {}, [], False, False),
    ('pens', 'Penalty-taker model (no pens points)', {}, [('team_pen_rate', no_pens)], True, False),
    ('penxg', 'npxG + pens model → total xG', {}, [('team_pen_rate', no_pens), ('strip_pens', keep_pen_xg)], True, False),
    ('mins', 'Recent starts / minutes (prior only)', {'k_min': BIG}, [], True, False),
    ('fix', 'Fixture difficulty (opponent and venue ignored)', {}, [('team_fixtures', fixtures_neutral)], True, False),
    ('team', 'Team model entirely (every team average)', {},
     [('team_fixtures', lambda o: fixtures_neutral(o, keep_team=False))], True, False),
    ('flags', 'FPL flags and news (everyone available)', {}, [], True, True),
]


def blank_flags(orig):
    def st(snap, gw, avail=None, last=None):
        s = orig(snap, gw, avail=avail, last=last)
        s.players['status'], s.players['chance'], s.players['news'] = 'a', np.nan, ''
        return s
    return st


def run_one(snap, gws, last, prm_kw, patches, flags_off):
    """Backtest rows for one ablation, plus the attacking (goals, pens, assists) and clean-sheet parts of each
    projection next to what those parts actually scored."""
    import fplvalue.backtest as bt
    parts = []

    def capture(orig):
        def proj(state, prm=None, gws=None, detail=False):
            out = orig(state, prm, gws=gws, detail=detail)
            parts.append(out[['id']].assign(gw=gws[0], att_model=out['next_goals'] + out['next_pens'] + out['next_assists'],
                                            cs_model=out['next_cs']))
            return out
        return proj

    with contextlib.ExitStack() as es:
        for name, fn in patches:
            es.enter_context(patch(pj, name, fn))
        if flags_off:
            es.enter_context(patch(bt, 'state_as_of', blank_flags))
        es.enter_context(patch(bt, 'project', capture))
        r = season_rows(snap, gws, replace(Params(), **prm_kw), last=last)
    h = snap.history[snap.history['round'].isin(gws)]
    pos = h['player_id'].map(snap.players.set_index('id')['pos'])
    act = pd.DataFrame({'gw': h['round'], 'id': h['player_id'],
                        'att_actual': h['goals_scored'] * pos.map(pj.GOAL) + 3 * h['assists'],
                        'cs_actual': h['clean_sheets'] * pos.map(pj.CS_PTS)}).groupby(['gw', 'id']).sum()
    r = r.merge(pd.concat(parts), on=['gw', 'id'], how='left').join(act, on=['gw', 'id'])
    r[['att_actual', 'cs_actual']] = r[['att_actual', 'cs_actual']].fillna(0)
    return r.sort_values(['gw', 'id']).reset_index(drop=True)


def flag_weeks(snap, gws):
    """Gameweeks with a snapshot from the 7 days before the deadline (same rule as backtest.season_rows)."""
    out = []
    for gw in gws:
        dl = snap.deadline(gw)
        p = snapshot_before(dl, after=snap.season_start - pd.Timedelta(days=30))
        if p is not None and (dl.normalize() - pd.Timestamp(p.name, tz='UTC')).days <= 7:
            out.append(gw)
    return out


def score(d: pd.DataFrame, col: str) -> dict:
    err = d[col] - d['actual']
    top = d.sort_values(col, ascending=False).groupby('gw').head(20)['actual'].mean()
    cap = d.sort_values(col, ascending=False).groupby('gw').head(1)['actual'].mean()
    sp = d.groupby('gw').apply(lambda g: g[col].corr(g['actual'], method='spearman'), include_groups=False).mean()
    return {'rmse': float(np.sqrt((err ** 2).mean())), 'mae': float(err.abs().mean()), 'spearman': float(sp),
            'top20': float(top), 'captain': float(cap)}


def delta_ci(base: pd.DataFrame, abl: pd.DataFrame, pred='model', act='actual', n=2000) -> tuple[float, float, float]:
    """Change in MSE (ablated − full), with a 90% bootstrap interval over gameweeks."""
    assert (base[['gw', 'id']].values == abl[['gw', 'id']].values).all()
    se = pd.DataFrame({'gw': base['gw'], 'b': (base[pred] - base[act]) ** 2,
                       'a': (abl[pred].values - base[act].values) ** 2})
    w = se.groupby('gw')[['a', 'b']].sum()
    cnt = se.groupby('gw').size()
    d = (w['a'] - w['b']).values
    c = cnt.values
    pt = d.sum() / c.sum()
    idx = RNG.integers(0, len(d), (n, len(d)))
    bs = d[idx].sum(1) / c[idx].sum(1)
    return pt, float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))


def main():
    cur = load()
    last = last_season(cur)
    seasons = [(f"{last.season_start.year}/{str(last.season_start.year + 1)[2:]}", last, range(5, 39), None),
               (f"{cur.season_start.year}/{str(cur.season_start.year + 1)[2:]}", cur, range(2, cur.last_finished_gw + 1), last)]
    rows = {}
    for label, snap, gws, prev in seasons:
        for key, name, kw, patches, use_last, flags_off in ABLATIONS:
            if key in ('nolast', 'lam') and prev is None:      # no prior season loaded: nothing to remove
                continue
            if key == 'flags':
                g = flag_weeks(snap, gws)
                if not g:
                    continue
            else:
                g = gws
            print(label, key, file=sys.stderr)
            rows[(label, key)] = run_one(snap, list(g), prev if use_last else None, kw, patches, flags_off)
            if key == 'flags':      # matching full-model rows on the same weeks
                rows[(label, 'full@flags')] = rows[(label, 'full')][rows[(label, 'full')]['gw'].isin(g)].reset_index(drop=True)

    lines = ['# Feature ablation: which inputs matter for player points', '',
             'Generated by `research/feature_ablation.py`. Each row removes one input and replays the walk-forward',
             'backtest (same as `fplvalue/backtest.py`). ΔMSE % is the change in mean squared error against the full',
             'model: above zero means the model got worse without the input, so the input helps. Brackets are a 90%',
             'bootstrap interval over gameweeks. Scored on every player whose team had a fixture.', '',
             '- **Points**: total projected points vs FPL points.',
             '- **Attack**: projected goals + penalties + assists points vs points actually scored from goals and assists.',
             '- **Clean sheets**: projected vs actual clean-sheet points.', '']
    names = {k: n for k, n, *_ in ABLATIONS}
    summary = []

    def pct_ci(ref, d, pred, act):
        mse = float(((ref[pred] - ref[act]) ** 2).mean())
        pt, lo, hi = delta_ci(ref, d, pred, act)
        star = ' *' if lo > 0 or hi < 0 else ''
        return 100 * pt / mse, f'{100 * pt / mse:+.1f}% ({100 * lo / mse:+.1f}, {100 * hi / mse:+.1f}){star}', star

    for label, snap, gws, prev in seasons:
        base = rows[(label, 'full')]
        lines += [f'## {label} (GW{min(gws)}-{max(gws)}, {base["gw"].nunique()} weeks, {len(base):,} player-weeks)', '',
                  '| Input removed | Points ΔMSE % | Attack ΔMSE % | Clean sheets ΔMSE % | RMSE | MAE | Spearman | Top-20 pts | Captain pts |',
                  '|---|---|---|---|---|---|---|---|---|']
        for key, *_ in ABLATIONS:
            if (label, key) not in rows:
                continue
            ref = rows[(label, 'full@flags')] if key == 'flags' else base
            d = rows[(label, key)]
            s = score(d, 'model')
            if key == 'full':
                cells = ['—', '—', '—']
            else:
                p_pt, p_txt, p_star = pct_ci(ref, d, 'model', 'actual')
                a_pt, a_txt, a_star = pct_ci(ref, d, 'att_model', 'att_actual')
                _, c_txt, _ = pct_ci(ref, d, 'cs_model', 'cs_actual')
                cells = [p_txt, a_txt, c_txt]
                summary.append((label, key, p_pt, p_star, a_pt, a_star))
            nm = names[key] + (f' ({d["gw"].nunique()} weeks with flags)' if key == 'flags' else '')
            lines.append(f'| {nm} | ' + ' | '.join(cells) + f' | {s["rmse"]:.3f} | {s["mae"]:.3f} | {s["spearman"]:.3f} | '
                         f'{s["top20"]:.2f} | {s["captain"]:.2f} |')
        if (label, 'flags') in rows:
            f = score(rows[(label, 'full@flags')], 'model')
            lines.append(f'| *Full model, flag weeks only* | — | — | — | {f["rmse"]:.3f} | {f["mae"]:.3f} | '
                         f'{f["spearman"]:.3f} | {f["top20"]:.2f} | {f["captain"]:.2f} |')
        lines.append('')

    lines += ['## Ranking', '', 'Points and attack ΔMSE % by season, sorted by the average points effect.', '',
              '| Input | ' + ' | '.join(f'Points {l} | Attack {l}' for l, *_ in seasons) + ' |',
              '|---|' + '---|---|' * len(seasons)]
    tab = {}
    for label, key, p, ps, a_, as_ in summary:
        tab.setdefault(key, {})[label] = f'{p:+.1f}%{ps} | {a_:+.1f}%{as_}'
    order = sorted(tab, key=lambda k: -np.mean([x[2] for x in summary if x[1] == k]))
    for k in order:
        lines.append(f'| {names[k]} | ' + ' | '.join(tab[k].get(l, 'n/a | n/a') for l, *_ in seasons) + ' |')
    lines += ['', '\\* 90% interval excludes zero.', '']
    OUT.write_text('\n'.join(lines))
    print(OUT.read_text())


if __name__ == '__main__':
    main()
