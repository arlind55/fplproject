"""Do bookmaker odds improve the fixture model's team goals and clean sheets?

Odds come from football-data.co.uk (free CSVs, one per season: E0.csv). For every match we take the
market-average 1X2 and over/under 2.5 prices, remove the margin, and find the home and away goal rates
(independent Poisson) that reproduce P(over 2.5) and P(home win) - P(away win). That gives

    odds xG for each side, and P(clean sheet) = exp(-opponent's rate)

The team model is the same opponent-adjusted, recency-weighted attack/defence fit that
fplvalue/team_model.py uses (half-life 15 matches, prior 2 matches), built from Understat xG with
fixture_backtest.py's walk-forward, predicting the next week only (odds are only priced a week or so out).
The blend is geometric:

    log rate = w x log(odds rate) + (1 - w) x log(model rate)

w is chosen on the tuning seasons and reported on the test seasons. Scored on goals (Poisson deviance)
and clean sheets (log loss), lower is better. Pre-closing odds (football-data collects them on Friday
afternoon for weekend games, Tuesday for midweek ones) are the main case, since the site refreshes before
prices close; closing odds are shown as an upper bound.

    python odds_backtest.py     # downloads ~10 small CSVs into data/raw/odds/ on the first run

Writes data/odds_backtest.md.
"""
from __future__ import annotations

import io
import os
import time
import urllib.request

import numpy as np
import pandas as pd

import fixture_backtest as fb

DATA = 'data'
RAW = f'{DATA}/raw/odds'
SEASONS = [f'{y % 100:02d}{(y + 1) % 100:02d}' for y in range(2016, 2026)]   # 1617 .. 2526
URL = 'https://www.football-data.co.uk/mmz4281/{}/E0.csv'
MODEL = {'window': 'decay', 'n': 15, 'k': 2.0, 'adj': True}
WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 2)

# football-data.co.uk name -> Understat name
NAMES = {'Man United': 'Manchester United', 'Man City': 'Manchester City', 'Newcastle': 'Newcastle United',
         "Nott'm Forest": 'Nottingham Forest', 'West Brom': 'West Bromwich Albion', 'Wolves': 'Wolverhampton Wanderers',
         'Sheffield United': 'Sheffield United', 'QPR': 'Queens Park Rangers'}

# (label, 1X2 columns, over/under 2.5 columns), first set present wins. Avg* from 2019/20, BbAv* before.
PRE = [(('AvgH', 'AvgD', 'AvgA'), ('Avg>2.5', 'Avg<2.5')), (('BbAvH', 'BbAvD', 'BbAvA'), ('BbAv>2.5', 'BbAv<2.5'))]
CLOSE = [(('AvgCH', 'AvgCD', 'AvgCA'), ('AvgC>2.5', 'AvgC<2.5'))]


def fetch(season: str) -> pd.DataFrame:
    os.makedirs(RAW, exist_ok=True)
    path = f'{RAW}/E0_{season}.csv'
    if not os.path.exists(path) or season == SEASONS[-1]:
        try:
            req = urllib.request.Request(URL.format(season), headers={'User-Agent': 'fplvalue-research'})
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
        except OSError:
            if not os.path.exists(path):
                raise
            print(f'{season}: download failed, using cached {path}')
        else:
            with open(path, 'wb') as f:
                f.write(body)
            time.sleep(1)
    with open(path, 'rb') as f:
        return pd.read_csv(io.BytesIO(f.read()), encoding='latin-1', on_bad_lines='skip')


def _pick(df: pd.DataFrame, sets) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index, columns=['oh', 'od', 'oa', 'oo', 'ou'], dtype=float)
    for x12, ou in sets:
        cols = list(x12) + list(ou)
        if all(c in df.columns for c in cols):
            vals = df[cols].apply(pd.to_numeric, errors='coerce')
            vals.columns = out.columns
            out = out.fillna(vals)
    return out


def odds_table() -> pd.DataFrame:
    parts = []
    for s in SEASONS:
        d = fetch(s).dropna(subset=['HomeTeam', 'AwayTeam'])
        d['date'] = pd.to_datetime(d['Date'], dayfirst=True, errors='coerce').dt.normalize()
        d['home'] = d['HomeTeam'].replace(NAMES)
        d['away'] = d['AwayTeam'].replace(NAMES)
        pre, close = _pick(d, PRE), _pick(d, CLOSE).add_prefix('c')
        parts.append(pd.concat([d[['date', 'home', 'away']], pre, close], axis=1))
    return pd.concat(parts, ignore_index=True)


def _pois(lam, kmax=12):
    k = np.arange(kmax + 1)
    lf = np.cumsum(np.log(np.maximum(k, 1)))
    return np.exp(k[None, :] * np.log(lam[:, None]) - lam[:, None] - lf[None, :])


def implied_rates(oh, od, oa, oo, ou):
    """Margin-free probabilities -> (home rate, away rate) under independent Poisson."""
    p = 1 / np.vstack([oh, od, oa]).T
    p = p / p.sum(1, keepdims=True)
    q = 1 / np.vstack([oo, ou]).T
    p_under = q[:, 1] / q.sum(1)
    # total goals T: P(Poisson(T) <= 2) = p_under, by bisection
    lo, hi = np.full(len(oh), 0.3), np.full(len(oh), 7.0)
    for _ in range(50):
        m = (lo + hi) / 2
        cdf = np.exp(-m) * (1 + m + m * m / 2)
        lo, hi = np.where(cdf > p_under, m, lo), np.where(cdf > p_under, hi, m)
    T = (lo + hi) / 2
    # home share s of T: P(H) - P(A) matches the market, by bisection
    target = p[:, 0] - p[:, 2]
    lo, hi = np.full(len(oh), 0.02), np.full(len(oh), 0.98)
    tri = np.tril(np.ones((13, 13)), -1)       # [i, j] = 1 when home goals i > away goals j
    for _ in range(40):
        s = (lo + hi) / 2
        ph, pa = _pois(T * s), _pois(T * (1 - s))
        joint = ph[:, :, None] * pa[:, None, :]
        diff = (joint * tri).sum((1, 2)) - (joint * tri.T).sum((1, 2))
        lo, hi = np.where(diff < target, s, lo), np.where(diff < target, hi, s)
    s = (lo + hi) / 2
    return T * s, T * (1 - s)


def main():
    t = fb.load()
    m = pd.read_csv(f'{DATA}/matches.csv', parse_dates=['datetime'])
    m = m[m['played'] == 1]
    m['date'] = m['datetime'].dt.normalize()
    o = odds_table()
    j = m.merge(o, on=['date', 'home', 'away'], how='left')
    # a handful of kick-offs straddle midnight in UK time; retry those a day either side
    for shift in (-1, 1):
        miss = j['oh'].isna()
        alt = m.loc[miss.values].assign(date=lambda x: x['date'] + pd.Timedelta(days=shift))
        alt = alt.merge(o, on=['date', 'home', 'away'], how='left')
        j.loc[miss.values, o.columns.drop(['date', 'home', 'away'])] = alt[o.columns.drop(['date', 'home', 'away'])].values
    seasons = [f'20{s[:2]}/{s[2:]}' for s in SEASONS]
    inscope = j['season'].isin(seasons)
    unmatched = j[inscope & j['oh'].isna()]
    print(f'matched odds for {(inscope & j["oh"].notna()).sum()} of {inscope.sum()} matches; '
          f'{len(unmatched)} unmatched', flush=True)

    for pre, cols in (('', ['oh', 'od', 'oa', 'oo', 'ou']), ('c', ['coh', 'cod', 'coa', 'coo', 'cou'])):
        ok = j[cols].notna().all(1) & (j[cols] > 1).all(1)
        lh, la = np.full(len(j), np.nan), np.full(len(j), np.nan)
        if ok.any():
            lh[ok.values], la[ok.values] = implied_rates(*[j.loc[ok, c].values for c in cols])
        j[pre + 'lam_h'], j[pre + 'lam_a'] = lh, la

    # per team-match odds rates
    h = j[['match_id', 'home_id', 'lam_h', 'lam_a', 'clam_h', 'clam_a']].rename(
        columns={'home_id': 'team_id', 'lam_h': 'o_for', 'lam_a': 'o_ag', 'clam_h': 'c_for', 'clam_a': 'c_ag'})
    a = j[['match_id', 'away_id', 'lam_a', 'lam_h', 'clam_a', 'clam_h']].rename(
        columns={'away_id': 'team_id', 'lam_a': 'o_for', 'lam_h': 'o_ag', 'clam_a': 'c_for', 'clam_h': 'c_ag'})
    od = pd.concat([h.assign(home=1), a.assign(home=0)])
    od = od.merge(m[['match_id', 'datetime']], on='match_id')

    fb.HORIZONS = (1,)
    p = fb.run(t, MODEL)
    p = p.merge(t[['team_id', 'season', 'match_no', 'date']], left_on=['team_id', 'season', 'n_in_season'],
                right_on=['team_id', 'season', 'match_no'], how='left')
    p = p.merge(od, left_on=['team_id', 'date'], right_on=['team_id', 'datetime'], how='left')
    p = p.dropna(subset=['lam', 'lam_ag', 'o_for', 'o_ag'])
    print('scored team-matches', len(p), flush=True)

    def sc(q, f, g):
        pcs = np.clip(np.exp(-g), 1e-4, 1 - 1e-4)
        return (fb.poisson_dev(q['gf'].values, f).mean(),
                -(q['cs'] * np.log(pcs) + (1 - q['cs']) * np.log(1 - pcs)).mean())

    def blend(q, w, src='o'):
        f = np.exp(w * np.log(q[src + '_for']) + (1 - w) * np.log(q['lam']))
        g = np.exp(w * np.log(q[src + '_ag']) + (1 - w) * np.log(q['lam_ag']))
        return f.values, g.values

    rows = []
    for src, lab in (('o', 'pre-closing'), ('c', 'closing')):
        for w in WEIGHTS:
            for seasons_, part in ((fb.TUNE, 'tune'), (fb.TEST, 'test')):
                q = p[p['season'].isin(seasons_)]
                if src == 'c':
                    q = q.dropna(subset=['c_for', 'c_ag'])
                if not len(q):
                    continue
                gd, cs = sc(q, *blend(q, w, src))
                rows.append({'odds': lab, 'w': w, 'part': part, 'n': len(q), 'goals dev': gd, 'CS log loss': cs})
    r = pd.DataFrame(rows)
    base = r[r['w'] == 0].set_index(['odds', 'part'])
    r['goals dev vs model %'] = 100 * (r['goals dev'] / r.set_index(['odds', 'part']).index.map(base['goals dev']).values - 1)
    r['CS vs model %'] = 100 * (r['CS log loss'] / r.set_index(['odds', 'part']).index.map(base['CS log loss']).values - 1)

    tune = r[(r['part'] == 'tune') & (r['odds'] == 'pre-closing')]
    w_goals = float(tune.loc[tune['goals dev'].idxmin(), 'w'])
    w_cs = float(tune.loc[tune['CS log loss'].idxmin(), 'w'])

    # early season, where the model has least to go on
    early = []
    for w in (0.0, w_goals, 1.0):
        q = p[p['season'].isin(fb.TEST) & (p['n_in_season'] <= 8)]
        gd, cs = sc(q, *blend(q, w))
        early.append({'w': w, 'n': len(q), 'goals dev': gd, 'CS log loss': cs})

    cols = ['odds', 'w', 'n', 'goals dev', 'goals dev vs model %', 'CS log loss', 'CS vs model %']
    rep = ['# Betting odds v the fixture model', '',
           'One week ahead, Understat goals, football-data.co.uk market-average odds. w = 0 is the team model alone,',
           'w = 1 is odds alone. Lower is better.', '',
           f'Matched odds for {(inscope & j["oh"].notna()).sum()} of {inscope.sum()} matches; '
           f'{len(p)} team-matches scored.', '',
           f'Best weight on tuning seasons (pre-closing): goals **w = {w_goals}**, clean sheets **w = {w_cs}**.', '']
    for part, lab in (('tune', 'Tuning seasons ' + ', '.join(fb.TUNE)), ('test', 'Test seasons ' + ', '.join(fb.TEST))):
        rep += [f'## {lab}', '', r[r['part'] == part][cols].to_markdown(index=False, floatfmt='.4f'), '']
    rep += ['## Test seasons, first 8 matches of a season only (pre-closing)', '',
            pd.DataFrame(early).to_markdown(index=False, floatfmt='.4f'), '']
    if len(unmatched):
        rep += ['## Unmatched matches', '', unmatched[['season', 'datetime', 'home', 'away']].head(30).to_markdown(index=False), '']
    with open(f'{DATA}/odds_backtest.md', 'w') as f:
        f.write('\n'.join(rep))
    print('\n'.join(rep))


if __name__ == '__main__':
    main()
