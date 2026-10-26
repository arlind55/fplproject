"""Do shot-level signals beyond npxG predict a player's next goals and assists?

For every outfield appearance, the player's previous 10 appearances (within ~13 months) give per-90
rates of: npxG, shots, shots on target, shots in the box, xG per shot, actual non-penalty goals,
headers' share, key passes, xA, xGChain, xGBuildup, and a "rising" signal (last 3 vs last 10).
The target is what the player does in his next 6 appearances: non-penalty goals and assists.

Poisson regression on log rates (exposure = next-6 minutes), trained on 2014/15-2020/21 and scored on
2021/22-2026/27. Each signal is scored two ways:
    alone            deviance explained vs a constant rate (how much it predicts by itself)
    on top of npxG   extra deviance explained once npxG/90 (or xA/90 for assists) is already in,
                     i.e. what the site's model would gain by adding it
Intervals are a 90% bootstrap over players.

FPL's own "threat" score (built from touches and shots in dangerous areas, the nearest thing we have
to touches in the box) is checked the same way on FPL 2025/26 + 2026/27 data, against FPL xG.

    python shot_signals_backtest.py     # needs shots.csv and appearances.csv (scrape_understat.py --shots)

Writes data/shot_signals_backtest.md.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

warnings.simplefilter('ignore')
DATA = Path(__file__).with_name('data')
ROOT = Path(__file__).resolve().parents[2]
N_PAST, N_NEXT, MIN_PAST, MIN_NEXT = 10, 6, 450, 180
TRAIN = [f'{y}/{str(y + 1)[2:]}' for y in range(2014, 2021)]
RNG = np.random.default_rng(0)


def per_match() -> pd.DataFrame:
    a = pd.read_csv(DATA / 'appearances.csv', parse_dates=['date'])
    s = pd.read_csv(DATA / 'shots.csv')
    s = s[~s['situation'].eq('Penalty') & ~s['result'].eq('OwnGoal')]
    s['on_target'] = s['result'].isin(['Goal', 'SavedShot'])
    # penalty area: 16.5 m of a 105 m pitch, 40.3 m of a 68 m width
    s['in_box'] = (s['X'] >= 0.843) & s['Y'].between(0.204, 0.796)
    s['header'] = s['shotType'].eq('Head')
    s['np_goal'] = s['result'].eq('Goal')
    agg = s.groupby(['match_id', 'player_id']).agg(
        np_shots=('xG', 'size'), sot=('on_target', 'sum'), box=('in_box', 'sum'), heads=('header', 'sum'),
        npg=('np_goal', 'sum'), npxg_s=('xG', 'sum'))
    a = a.join(agg, on=['match_id', 'player_id'])
    a[agg.columns] = a[agg.columns].fillna(0)
    a = a[(a['minutes'] > 0) & a['position'].ne('GK')]
    # Understat's position for the player's starts; GKs out
    pos = a[a['position'].ne('Sub')].groupby('player_id')['position'].agg(lambda p: p.mode().iloc[0])
    a = a[a['player_id'].map(pos).fillna('GK').ne('GK')]
    a['pos'] = a['player_id'].map(pos).str[0].map({'D': 'DEF', 'M': 'MID', 'A': 'MID', 'F': 'FWD'}).fillna('MID')
    return a.sort_values(['player_id', 'date']).reset_index(drop=True)


def windows(a: pd.DataFrame) -> pd.DataFrame:
    cols = ['minutes', 'npxG', 'np_shots', 'sot', 'box', 'heads', 'npg', 'key_passes', 'xA', 'assists',
            'xGChain', 'xGBuildup']
    g = a.groupby('player_id')
    past = {c: g[c].transform(lambda x: x.shift(1).rolling(N_PAST, min_periods=1).sum()) for c in cols}
    last3 = {c: g[c].transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum()) for c in ('minutes', 'npxG', 'np_shots')}
    nxt = {c: g[c].transform(lambda x: x[::-1].rolling(N_NEXT, min_periods=1).sum()[::-1]) for c in ('minutes', 'npg', 'assists')}
    span = a['date'] - g['date'].shift(N_PAST)
    d = pd.DataFrame({'season': a['season'], 'player_id': a['player_id'], 'pos': a['pos'],
                      'span_ok': span.isna() | (span <= pd.Timedelta(days=400))})
    m90 = past['minutes'] / 90
    d['past_min'] = past['minutes']
    for c in ('npxG', 'np_shots', 'sot', 'box', 'npg', 'key_passes', 'xA', 'xGChain', 'xGBuildup'):
        d[c + '90'] = past[c] / m90
    d['xg_per_shot'] = past['npxG'] / past['np_shots'].replace(0, np.nan)
    d['sot_share'] = past['sot'] / past['np_shots'].replace(0, np.nan)
    d['box_share'] = past['box'] / past['np_shots'].replace(0, np.nan)
    d['head_share'] = past['heads'] / past['np_shots'].replace(0, np.nan)
    d['finishing'] = (past['npg'] + 1) / (past['npxG'] + 1)          # goals over xG, damped
    d['xg_rise'] = (last3['npxG'] / (last3['minutes'] / 90)) / d['npxG90'].clip(lower=0.02)
    d['shot_rise'] = (last3['np_shots'] / (last3['minutes'] / 90)) / d['np_shots90'].clip(lower=0.1)
    d['next_min'], d['next_npg'], d['next_ast'] = nxt['minutes'], nxt['npg'], nxt['assists']
    # next window starts at this appearance: features use only earlier ones
    d = d[(d['past_min'] >= MIN_PAST) & (d['next_min'] >= MIN_NEXT) & d['span_ok']]
    return d.replace([np.inf, -np.inf], np.nan)


def design(d: pd.DataFrame, feats: list[str]) -> np.ndarray:
    cols = []
    for f in feats:
        v = d[f].astype(float)
        if f.endswith('90'):
            v = np.log(v + 0.02)
        elif f in ('xg_rise', 'shot_rise', 'finishing'):
            v = np.log(v.clip(0.05, 20))
        cols.append(v.fillna(v.median()).to_numpy())
    return np.column_stack(cols) if cols else np.zeros((len(d), 0))


def fit_dev(tr, te, feats, target):
    """Per-row Poisson deviance on the test set for a model on feats (rate = target / next 90s)."""
    e_tr, e_te = tr['next_min'] / 90, te['next_min'] / 90
    Xtr, Xte = design(tr, feats), design(te, feats)
    if Xtr.shape[1] == 0:
        mu = np.full(len(te), tr[target].sum() / e_tr.sum()) * e_te
    else:
        m = PoissonRegressor(alpha=1e-6, max_iter=1000).fit(Xtr, tr[target] / e_tr, sample_weight=e_tr)
        mu = m.predict(Xte) * e_te
    y = te[target].to_numpy()
    mu = np.asarray(mu)
    return 2 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu))


def gain(te, dev_base, dev_new, n=500):
    """% of baseline deviance removed, with a 90% player-bootstrap interval."""
    df = pd.DataFrame({'p': te['player_id'].values, 'b': dev_base, 'n': dev_new}).groupby('p').sum()
    b, nn = df['b'].to_numpy(), df['n'].to_numpy()
    pt = 100 * (b.sum() - nn.sum()) / b.sum()
    idx = RNG.integers(0, len(b), (n, len(b)))
    bs = 100 * (b[idx].sum(1) - nn[idx].sum(1)) / b[idx].sum(1)
    lo, hi = np.quantile(bs, [0.05, 0.95])
    return pt, lo, hi


def fmt(g):
    pt, lo, hi = g
    return f'{pt:+.2f}% ({lo:+.2f}, {hi:+.2f})' + (' *' if lo > 0 or hi < 0 else '')


LABELS = {'npxG90': 'npxG / 90', 'np_shots90': 'Shots / 90 (non-pen)', 'sot90': 'Shots on target / 90',
          'box90': 'Shots in the box / 90', 'xg_per_shot': 'xG per shot', 'sot_share': 'Share of shots on target',
          'box_share': 'Share of shots from the box', 'head_share': 'Share of shots that are headers',
          'npg90': 'Non-penalty goals / 90', 'finishing': 'Goals minus xG (finishing, damped)',
          'xg_rise': 'Rising xG (last 3 vs last 10)', 'shot_rise': 'Rising shots (last 3 vs last 10)',
          'key_passes90': 'Key passes / 90', 'xA90': 'xA / 90', 'xGChain90': 'xGChain / 90', 'xGBuildup90': 'xGBuildup / 90'}


def understat_part(lines):
    d = windows(per_match())
    tr, te = d[d['season'].isin(TRAIN)], d[~d['season'].isin(TRAIN)]
    lines += [f'Training rows {len(tr):,} (2014/15-2020/21), test rows {len(te):,} (2021/22-2026/27), '
              f'{te["player_id"].nunique():,} test players. A row is one appearance with 450+ minutes in the '
              f'previous {N_PAST} and 180+ in the next {N_NEXT}.', '']
    for target, base, title, feats in (
            ('next_npg', 'npxG90', 'Next-6 non-penalty goals',
             ['npxG90', 'np_shots90', 'sot90', 'box90', 'xg_per_shot', 'sot_share', 'box_share', 'head_share',
              'npg90', 'finishing', 'xg_rise', 'shot_rise', 'xGChain90']),
            ('next_ast', 'xA90', 'Next-6 assists',
             ['xA90', 'key_passes90', 'xGChain90', 'xGBuildup90', 'np_shots90', 'npxG90'])):
        d0 = fit_dev(tr, te, [], target)
        db = fit_dev(tr, te, [base], target)
        lines += [f'## {title}', '', f'Baseline already in the model: **{LABELS[base]}**, which alone explains '
                  f'{fmt(gain(te, d0, db))} of deviance.', '',
                  f'| Signal | Alone | On top of {LABELS[base]} |', '|---|---|---|']
        rows = []
        for f in feats:
            if f == base:
                continue
            alone = gain(te, d0, fit_dev(tr, te, [f], target))
            extra = gain(te, db, fit_dev(tr, te, [base, f], target))
            rows.append((extra[0], f'| {LABELS[f]} | {fmt(alone)} | {fmt(extra)} |'))
        lines += [r for _, r in sorted(rows, reverse=True)]
        lines.append('')
    # all shot signals together on top of npxG
    allf = ['npxG90', 'np_shots90', 'sot90', 'box90', 'xg_per_shot', 'npg90', 'xg_rise', 'shot_rise', 'xGChain90']
    g = gain(te, fit_dev(tr, te, ['npxG90'], 'next_npg'), fit_dev(tr, te, allf, 'next_npg'))
    lines += [f'All goal signals together on top of npxG/90: {fmt(g)}.', '']
    # what does a rise look like in raw terms
    te2 = te.assign(rise=pd.qcut(te['xg_rise'].rank(method='first'), 5, labels=['lowest', '2', '3', '4', 'highest']))
    t = te2.groupby('rise', observed=True).apply(lambda x: pd.Series({
        'past npxG/90': x['npxG90'].mean(), 'next npG/90': x['next_npg'].sum() / (x['next_min'].sum() / 90)}),
        include_groups=False)
    t['next / past'] = t['next npG/90'] / t['past npxG/90']
    lines += ['**Recent xG rise, in plain numbers** (test seasons, players split into fifths by last-3 vs last-10 npxG):', '',
              '| Last-3 vs last-10 | Past npxG/90 (last 10) | Next-6 goals/90 | Ratio |', '|---|---|---|---|']
    for k, r in t.iterrows():
        lines.append(f'| {k} | {r["past npxG/90"]:.3f} | {r["next npG/90"]:.3f} | {r["next / past"]:.2f} |')
    lines.append('')


def fpl_part(lines):
    """FPL threat / ICT vs FPL xG, on FPL's own seasons (2025/26 final + this season)."""
    sys.path.insert(0, str(ROOT))
    from fplvalue.load import load, last_season
    cur = load(); last = last_season(cur)
    frames = []
    for snap in (last, cur):
        h = snap.history.copy()
        pos = h['player_id'].map(snap.players.set_index('id')['pos'])
        h = h[pos.isin(['DEF', 'MID', 'FWD']) & (h['minutes'] > 0)].sort_values(['player_id', 'kickoff_time'])
        for c in ('threat', 'creativity', 'expected_goals', 'expected_assists', 'goals_scored', 'assists', 'minutes',
                  'bps', 'total_points'):
            h[c] = pd.to_numeric(h[c], errors='coerce').fillna(0)
        g = h.groupby('player_id')
        out = pd.DataFrame({'player_id': h['player_id'].astype(str) + snap.path.name, 'season': snap.path.name})
        pm = g['minutes'].transform(lambda x: x.shift(1).rolling(N_PAST, min_periods=1).sum())
        for c, n in (('expected_goals', 'fxg90'), ('threat', 'threat90'), ('expected_assists', 'fxa90'),
                     ('creativity', 'creat90')):
            out[n] = g[c].transform(lambda x: x.shift(1).rolling(N_PAST, min_periods=1).sum()) / (pm / 90)
        for c, n in (('minutes', 'next_min'), ('goals_scored', 'next_npg'), ('assists', 'next_ast')):
            out[n] = g[c].transform(lambda x: x[::-1].rolling(N_NEXT, min_periods=1).sum()[::-1])
        out['past_min'] = pm
        frames.append(out[(pm >= MIN_PAST) & (out['next_min'] >= MIN_NEXT)])
    d = pd.concat(frames).replace([np.inf, -np.inf], np.nan)
    # too little data for a season split: 5-fold by player
    players = d['player_id'].unique()
    fold = pd.Series(RNG.integers(0, 5, len(players)), index=players)
    d['fold'] = d['player_id'].map(fold)

    def cv(feats, target):
        dev = np.zeros(len(d))
        for k in range(5):
            m = (d['fold'] == k).to_numpy()
            dev[m] = fit_dev(d[~m], d[m], feats, target)
        return dev
    lines += ['## FPL threat (touches and shots in dangerous areas) vs FPL xG', '',
              f'FPL 2025/26 + 2026/27 so far, {len(d):,} rows, 5-fold cross-validation by player (goals include penalties here).', '',
              '| Target | Signal | Alone | On top of FPL xG (or xA) |', '|---|---|---|---|']
    for target, base, feats in (('next_npg', 'fxg90', ['threat90']), ('next_ast', 'fxa90', ['creat90'])):
        d0, db = cv([], target), cv([base], target)
        name = {'fxg90': 'FPL xG / 90', 'fxa90': 'FPL xA / 90', 'threat90': 'Threat / 90', 'creat90': 'Creativity / 90'}
        lines.append(f'| {target[5:]} | {name[base]} | {fmt(gain(d, d0, db))} | — |')
        for f in feats:
            lines.append(f'| {target[5:]} | {name[f]} | {fmt(gain(d, d0, cv([f], target)))} | '
                         f'{fmt(gain(d, db, cv([base, f], target)))} |')
    lines.append('')


def main():
    lines = ['# Shot signals beyond npxG: do they predict the next goals and assists?', '',
             'Generated by `shot_signals_backtest.py`. "Alone" is the share of Poisson deviance a signal explains '
             'against a flat rate; "on top of" is the extra share once the model\'s existing rate is in. Brackets: '
             '90% bootstrap over players; * = interval excludes zero.', '']
    understat_part(lines)
    fpl_part(lines)
    (DATA / 'shot_signals_backtest.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
