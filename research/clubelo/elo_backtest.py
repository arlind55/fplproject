"""Does an Elo rating improve the fixture model's team goals and clean sheets?

Two Elo sources:
  clubelo   Club Elo (clubelo.com) rating history, saved to data/elo_history.csv from a machine that can
            reach api.clubelo.com (it blocks cloud and GitHub Actions IPs). Used if the file exists.
  home      our own Elo, built from openfootball Premier League + Championship results (CC0), so
            promoted sides carry a rating earned in the Championship. Standard football Elo with a
            goal-difference multiplier and a home edge; K and the home edge are tuned on the tuning seasons.

Elo is one number per team, so it is turned into goal rates with a Poisson fit on the tuning seasons:

    log E[goals for] = a + b x (Elo team - Elo opponent) + c x home

FPL's own fixture difficulty (FDR, vaastav/Fantasy-Premier-League fixtures, 2018/19 on) is fitted the same
way as a baseline: log E[goals] = a + b x FDR faced + c x FDR the opponent faces + d x home.

Everything uses fixture_backtest.py's walk-forward: Elo is read at the weekly cutoff, so a match 5 weeks out
uses the Elo from 5 weeks before. The blend with the fixture model (the site's decay-15 opponent-adjusted
xG fit) is geometric, w = 0 model alone, w = 1 Elo alone; w is chosen on tuning seasons.

    python elo_backtest.py        # downloads ~40 small JSON/CSV files into data/raw/ on the first run

Writes data/elo_backtest.md.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'understat'))
import fixture_backtest as fb  # noqa: E402

DATA = os.path.join(HERE, 'data')
RAW = os.path.join(DATA, 'raw')
UNDERSTAT = os.path.join(HERE, '..', 'understat', 'data')
fb.DATA = UNDERSTAT
OF_URL = 'https://raw.githubusercontent.com/openfootball/football.json/master/{}/{}.json'
VA_URL = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{}'
MODEL = {'window': 'decay', 'n': 15, 'k': 2.0, 'adj': True}
WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 2)

# openfootball / Club Elo / vaastav name -> Understat name (after stripping FC/AFC)
NAMES = {
    'Brighton & Hove Albion': 'Brighton', 'Cardiff City': 'Cardiff', 'Coventry City': 'Coventry',
    'Huddersfield Town': 'Huddersfield', 'Hull City': 'Hull', 'Ipswich Town': 'Ipswich', 'Leeds United': 'Leeds',
    'Leicester City': 'Leicester', 'Luton Town': 'Luton', 'Norwich City': 'Norwich', 'Stoke City': 'Stoke',
    'Swansea City': 'Swansea', 'Tottenham Hotspur': 'Tottenham', 'West Ham United': 'West Ham',
    # Club Elo
    'Man City': 'Manchester City', 'Man United': 'Manchester United', 'Newcastle': 'Newcastle United',
    'Forest': 'Nottingham Forest', 'West Brom': 'West Bromwich Albion', 'Wolves': 'Wolverhampton Wanderers',
    'Sheffield United': 'Sheffield United', 'QPR': 'Queens Park Rangers',
    # vaastav short forms
    'Man Utd': 'Manchester United', "Nott'm Forest": 'Nottingham Forest', 'Spurs': 'Tottenham',
    'Sheffield Utd': 'Sheffield United',
}


def norm(name: str) -> str:
    n = name.strip()
    for suf in (' AFC', ' FC'):
        if n.endswith(suf):
            n = n[: -len(suf)]
    if n.startswith('AFC '):
        n = n[4:]
    return NAMES.get(n, n)


def fetch(url: str, path: str, refresh: bool = False) -> str | None:
    if os.path.exists(path) and not refresh:
        with open(path) as f:
            return f.read()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'fplvalue-research'})
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read().decode()
    except OSError:
        return None
    with open(path, 'w') as f:
        f.write(body)
    time.sleep(0.3)
    return body


# ---------------------------------------------------------------- results and home-made Elo

def results() -> pd.DataFrame:
    rows = []
    for y in range(2010, 2027):
        s = f'{y}-{(y + 1) % 100:02d}'
        for lg in ('en.1', 'en.2'):
            body = fetch(OF_URL.format(s, lg), f'{RAW}/openfootball/{s}_{lg}.json', refresh=(y == 2026))
            if body is None:
                continue
            for m in json.loads(body)['matches']:
                sc = m.get('score') or {}
                ft = sc[:2] if isinstance(sc, list) else sc.get('ft')
                if not ft:
                    continue
                rows.append({'date': pd.Timestamp(m['date']), 'league': lg, 'home': norm(m['team1']),
                             'away': norm(m['team2']), 'hg': ft[0], 'ag': ft[1]})
    return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)


def home_elo(res: pd.DataFrame, k: float, hfa: float, start: float = 1500.0, ch_start: float = 1350.0) -> pd.DataFrame:
    """Rating after each match day, per team. Championship sides start lower."""
    r: dict[str, float] = {}
    out = []
    for row in res.itertuples(index=False):
        rh = r.get(row.home, start if row.league == 'en.1' else ch_start)
        ra = r.get(row.away, start if row.league == 'en.1' else ch_start)
        exp_h = 1 / (1 + 10 ** (-(rh + hfa - ra) / 400))
        gd = abs(row.hg - row.ag)
        mult = 1 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
        act = 1.0 if row.hg > row.ag else 0.5 if row.hg == row.ag else 0.0
        d = k * mult * (act - exp_h)
        r[row.home], r[row.away] = rh + d, ra - d
        out += [(row.date, row.home, rh + d), (row.date, row.away, ra - d)]
    e = pd.DataFrame(out, columns=['date', 'team', 'elo'])
    return e.groupby(['team', 'date'], as_index=False).last()


def club_elo() -> pd.DataFrame | None:
    path = os.path.join(DATA, 'elo_history.csv')
    if not os.path.exists(path):
        return None
    c = pd.read_csv(path, parse_dates=['From', 'To'])
    c['team'] = c['Club'].map(norm)
    # rating valid From..To; store as "rating from this date on"
    return c.rename(columns={'From': 'date', 'Elo': 'elo'})[['team', 'date', 'elo']].sort_values(['team', 'date'])


def elo_at(e: pd.DataFrame, team: pd.Series, when: pd.Series) -> np.ndarray:
    """Latest rating strictly before `when` for each (team, when)."""
    q = pd.DataFrame({'team': team.values, 'when': when.values - pd.Timedelta(seconds=1), 'i': np.arange(len(team))})
    q = q.sort_values('when')
    e = e.sort_values('date')
    m = pd.merge_asof(q, e.rename(columns={'date': 'when'}), on='when', by='team', direction='backward')
    return m.sort_values('i')['elo'].values


# ---------------------------------------------------------------- FPL FDR

def fdr() -> pd.DataFrame:
    master = fetch(VA_URL.format('master_team_list.csv'), f'{RAW}/vaastav/master_team_list.csv')
    mt = pd.read_csv(pd.io.common.StringIO(master))
    rows = []
    for y in range(2018, 2027):
        s = f'{y}-{(y + 1) % 100:02d}'
        body = fetch(VA_URL.format(f'{s}/fixtures.csv'), f'{RAW}/vaastav/fixtures_{s}.csv', refresh=(y == 2026))
        if body is None:
            continue
        f = pd.read_csv(pd.io.common.StringIO(body))
        names = mt[mt['season'] == s].set_index('team')['team_name']
        if names.empty:
            tb = fetch(VA_URL.format(f'{s}/teams.csv'), f'{RAW}/vaastav/teams_{s}.csv')
            names = pd.read_csv(pd.io.common.StringIO(tb)).set_index('id')['name']
        f['home'] = f['team_h'].map(names).map(norm)
        f['away'] = f['team_a'].map(names).map(norm)
        f['date'] = pd.to_datetime(f['kickoff_time'], utc=True).dt.tz_convert(None).dt.normalize()
        rows.append(f[['date', 'home', 'away', 'team_h_difficulty', 'team_a_difficulty']])
    f = pd.concat(rows, ignore_index=True)
    h = f.rename(columns={'home': 'team', 'away': 'opp', 'team_h_difficulty': 'fdr_faced', 'team_a_difficulty': 'fdr_opp'})
    a = f.rename(columns={'away': 'team', 'home': 'opp', 'team_a_difficulty': 'fdr_faced', 'team_h_difficulty': 'fdr_opp'})
    return pd.concat([h, a])[['date', 'team', 'opp', 'fdr_faced', 'fdr_opp']]


# ---------------------------------------------------------------- scoring

def poisson_fit(X: np.ndarray, y: np.ndarray, iters: int = 30) -> np.ndarray:
    X = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X.shape[1])
    b[0] = np.log(y.mean())
    for _ in range(iters):
        mu = np.exp(X @ b)
        b = b + np.linalg.solve(X.T @ (X * mu[:, None]), X.T @ (y - mu))
    return b


def poisson_pred(b: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.exp(np.column_stack([np.ones(len(X)), X]) @ b)


def scores(q: pd.DataFrame, f: np.ndarray, g: np.ndarray) -> dict:
    pcs = np.clip(np.exp(-g), 1e-4, 1 - 1e-4)
    return {'n': len(q), 'xG MSE': float(((q['xg'].values - f) ** 2).mean()),
            'goals dev': float(fb.poisson_dev(q['gf'].values, f).mean()),
            'CS log loss': float(-(q['cs'] * np.log(pcs) + (1 - q['cs']) * np.log(1 - pcs)).mean())}


def main():
    t = fb.load()
    names = pd.read_csv(f'{UNDERSTAT}/team_matches.csv')[['team_id', 'team']].drop_duplicates().set_index('team_id')['team']
    origin = t['date'].min().normalize() - pd.Timedelta(days=t['date'].min().weekday())

    fb.HORIZONS = (1, 5, 10)
    p = fb.run(t, MODEL)
    p = p.merge(t[['team_id', 'season', 'match_no', 'date', 'opp_id', 'home']],
                left_on=['team_id', 'season', 'n_in_season'], right_on=['team_id', 'season', 'match_no'], how='left')
    p['team'] = p['team_id'].map(names)
    p['opp'] = p['opp_id'].map(names)
    p['cutoff'] = origin + pd.to_timedelta(7 * (p['week'] - p['h'] + 1), unit='D')
    p['day'] = p['date'].dt.normalize()
    p = p.dropna(subset=['lam', 'lam_ag']).reset_index(drop=True)

    # promoted = not in the Understat Premier League the season before
    seasons = sorted(t['season'].unique())
    in_pl = t.groupby('season')['team_id'].apply(set)
    prev = {s: in_pl[seasons[i - 1]] for i, s in enumerate(seasons) if i}
    p['promoted'] = [tid not in prev.get(s, {tid}) for tid, s in zip(p['team_id'], p['season'])]

    tune = p['season'].isin(fb.TUNE)
    res = results()
    print('results', len(res), res['date'].min().date(), res['date'].max().date(), flush=True)
    unk = sorted(set(p['team']) - set(res['home']))
    print('Understat teams missing from openfootball:', unk, flush=True)

    sources = {}
    # tune home-made Elo K and home edge on the tuning seasons (1 week ahead, goals deviance of the Elo-only fit)
    best = None
    for k in (5, 10, 20, 30):
        for hfa in (40, 70, 100, 130):
            e = home_elo(res, k, hfa)
            d = elo_at(e, p['team'], p['cutoff']) - elo_at(e, p['opp'], p['cutoff'])
            ok = tune & (p['h'] == 1) & ~np.isnan(d)
            X = np.column_stack([d, p['home']])
            b = poisson_fit(X[ok], p.loc[ok, 'gf'].values)
            dev = fb.poisson_dev(p.loc[ok, 'gf'].values, poisson_pred(b, X[ok])).mean()
            if best is None or dev < best[0]:
                best = (dev, k, hfa, e)
    _, k_best, hfa_best, e_home = best
    sources[f'home-made Elo (K={k_best}, home edge {hfa_best})'] = e_home
    ce = club_elo()
    if ce is not None:
        sources['Club Elo'] = ce

    rows, early, promo, coefs = [], [], [], []

    def add(label, f, g, w):
        for seas, part in ((fb.TUNE, 'tune'), (fb.TEST, 'test')):
            for h in fb.HORIZONS:
                m = p['season'].isin(seas) & (p['h'] == h) & ~np.isnan(f) & ~np.isnan(g)
                rows.append({'source': label, 'w': w, 'part': part, 'h': h, **scores(p[m], f[m], g[m])})
        m = p['season'].isin(fb.TEST) & (p['h'] == 1) & (p['n_in_season'] <= 8) & ~np.isnan(f)
        early.append({'source': label, 'w': w, **scores(p[m], f[m], g[m])})
        m = m & p['promoted']
        promo.append({'source': label, 'w': w, **scores(p[m], f[m], g[m])})

    def opp_rates(f):
        # rate the opponent is expected to score in the same match = the team's goals against
        key = pd.Series(f, index=pd.MultiIndex.from_arrays([p['h'], p['day'], p['team']]))
        return key.reindex(pd.MultiIndex.from_arrays([p['h'], p['day'], p['opp']])).values

    model_f, model_g = p['lam'].values, p['lam_ag'].values
    add('fixture model', model_f, model_g, 0.0)

    for label, e in sources.items():
        d = elo_at(e, p['team'], p['cutoff']) - elo_at(e, p['opp'], p['cutoff'])
        X = np.column_stack([d, p['home']])
        ok = tune & (p['h'] == 1) & ~np.isnan(d)
        b = poisson_fit(X[ok], p.loc[ok, 'gf'].values)
        coefs.append({'source': label, 'intercept': b[0], 'per 100 Elo': 100 * b[1], 'home': b[2]})
        ef = np.where(np.isnan(d), np.nan, poisson_pred(b, np.nan_to_num(X)))
        eg = opp_rates(ef)
        for w in WEIGHTS:
            f = np.exp(w * np.log(ef) + (1 - w) * np.log(model_f))
            g = np.exp(w * np.log(eg) + (1 - w) * np.log(model_g))
            add(label, f, g, w)

    # FPL FDR baseline (2018/19 on; tuned on 2018/19 and 2019/20)
    fd = fdr()
    q = p[['day', 'team', 'opp']].merge(fd.rename(columns={'date': 'day'}), on=['day', 'team', 'opp'], how='left')
    for shift in (-1, 1):
        miss = q['fdr_faced'].isna().values
        alt = p.loc[miss, ['day', 'team', 'opp']].assign(day=lambda x: x['day'] + pd.Timedelta(days=shift))
        alt = alt.merge(fd.rename(columns={'date': 'day'}), on=['day', 'team', 'opp'], how='left')
        q.loc[miss, ['fdr_faced', 'fdr_opp']] = alt[['fdr_faced', 'fdr_opp']].values
    have = q['fdr_faced'].notna().values
    print('FDR matched', have.sum(), 'of', len(q), flush=True)
    X = np.column_stack([q['fdr_faced'].fillna(3), q['fdr_opp'].fillna(3), p['home']]).astype(float)
    ok = have & p['season'].isin(['2018/19', '2019/20']).values & (p['h'] == 1).values
    b = poisson_fit(X[ok], p.loc[ok, 'gf'].values)
    coefs.append({'source': 'FPL FDR', 'intercept': b[0], 'per 100 Elo': np.nan, 'home': b[3],
                  'per FDR step faced': b[1], 'per FDR step opp faces': b[2]})
    ff = np.where(have, poisson_pred(b, X), np.nan)
    fg = opp_rates(ff)
    add('FPL FDR', ff, fg, 1.0)

    r = pd.DataFrame(rows)

    def rel(df, keys):
        base = df[df['source'] == 'fixture model'].set_index(keys)
        out = df.copy()
        for c in ('xG MSE', 'goals dev', 'CS log loss'):
            out[c + ' vs model %'] = 100 * (out[c] / out.set_index(keys).index.map(base[c]).values - 1)
        return out

    r = rel(r, ['part', 'h'])
    best_w = {}
    for label in sources:
        tt = r[(r['source'] == label) & (r['part'] == 'tune') & (r['h'] == 1)]
        best_w[label] = float(tt.loc[tt['goals dev'].idxmin(), 'w'])

    cols = ['source', 'w', 'n', 'xG MSE vs model %', 'goals dev vs model %', 'CS log loss vs model %']
    rep = ['# Elo v the fixture model', '',
           'Walk-forward like fixture_backtest.py. "fixture model" is the site\'s decay-15 opponent-adjusted xG fit.',
           'w = 0 is the fixture model alone, w = 1 is the Elo fit alone. Negative % = better than the fixture model.', '',
           '## Elo-to-goals fits (tuning seasons, 1 week ahead)', '',
           pd.DataFrame(coefs).to_markdown(index=False, floatfmt='.3f'), '',
           'Best Elo weight on tuning seasons (goals, 1 week ahead): ' +
           ', '.join(f'{k}: **w = {v}**' for k, v in best_w.items()), '']
    for h in fb.HORIZONS:
        sel = r[(r['part'] == 'test') & (r['h'] == h) &
                ((r['source'].isin(['fixture model', 'FPL FDR'])) |
                 r.apply(lambda x: x['w'] in (best_w.get(x['source'], -1), 1.0), axis=1))]
        rep += [f'## Test seasons ({", ".join(fb.TEST)}), {h} week{"s" if h > 1 else ""} ahead', '',
                sel[cols].to_markdown(index=False, floatfmt='.2f'), '']
    for lab, data in (('first 8 matches of a season', early), ('promoted teams, first 8 matches', promo)):
        e = pd.DataFrame(data)
        ref = e[e['source'] == 'fixture model'].iloc[0]
        for c in ('xG MSE', 'goals dev', 'CS log loss'):
            e[c + ' vs model %'] = 100 * (e[c] / ref[c] - 1)
        e = e[(e['source'].isin(['fixture model', 'FPL FDR'])) |
              e.apply(lambda x: x['w'] in (best_w.get(x['source'], -1), 0.5, 1.0), axis=1)]
        rep += [f'## Test seasons, 1 week ahead, {lab}', '', e[cols].to_markdown(index=False, floatfmt='.2f'), '']
    for label, w in best_w.items():
        ps = []
        d = elo_at(sources[label], p['team'], p['cutoff']) - elo_at(sources[label], p['opp'], p['cutoff'])
        X = np.column_stack([d, p['home']])
        ok = tune & (p['h'] == 1) & ~np.isnan(d)
        b = poisson_fit(X[ok], p.loc[ok, 'gf'].values)
        ef = poisson_pred(b, np.nan_to_num(X))
        eg = opp_rates(ef)
        f = np.exp(w * np.log(ef) + (1 - w) * np.log(model_f))
        g = np.exp(w * np.log(eg) + (1 - w) * np.log(model_g))
        for s_ in fb.TEST:
            m = (p['season'] == s_) & (p['h'] == 1) & ~np.isnan(g)
            a, c = scores(p[m], model_f[m], model_g[m]), scores(p[m], f[m], g[m])
            ps.append({'season': s_, 'n': a['n'], 'goals dev vs model %': 100 * (c['goals dev'] / a['goals dev'] - 1),
                       'CS log loss vs model %': 100 * (c['CS log loss'] / a['CS log loss'] - 1)})
        rep += [f'## {label}, w = {w}, by test season (1 week ahead)', '',
                pd.DataFrame(ps).to_markdown(index=False, floatfmt='.2f'), '']
    rep += ['## Full weight grid, tuning seasons, 1 week ahead', '',
            r[(r['part'] == 'tune') & (r['h'] == 1)][cols].to_markdown(index=False, floatfmt='.2f'), '']
    with open(os.path.join(DATA, 'elo_backtest.md'), 'w') as f:
        f.write('\n'.join(rep))
    print('\n'.join(rep))


if __name__ == '__main__':
    main()
