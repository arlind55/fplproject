"""Bookmaker odds for the next round of fixtures, as expected goals for each side.

football-data.co.uk's fixtures.csv carries market-average 1X2 and over/under 2.5 prices for the coming
round (collected on Friday afternoon for weekend games, Tuesday for midweek ones). We remove the margin and
find the home and away goal rates (independent Poisson) that reproduce P(over 2.5) and P(home win) − P(away win).

fixture_model() blends these with the team model as log rate = W x log(odds) + (1 − W) x log(model).
research/understat/odds_backtest.py picked W = 0.8 on 2016/17-2019/20; on 2020/21-2025/26 it cut goals
deviance 2.2% and clean-sheet log loss 1.1% against the team model alone. Fixtures without odds (anything
beyond the next round) stay on the team model.

    fetch(fixtures_df, teams_df, old)   # -> one row per FPL fixture with odds, written to odds.csv
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests

URL = 'https://www.football-data.co.uk/fixtures.csv'
HEADERS = {'User-Agent': 'fplproject-data-pipeline/2.0'}
W = 0.8
COLS = ['fixture', 'odds_xg_home', 'odds_xg_away', 'odds_fetched_at']

# football-data.co.uk name -> FPL name, where they differ
NAMES = {'Man United': 'Man Utd', 'Tottenham': 'Spurs', 'Coventry': 'Coventry City', 'Hull': 'Hull City',
         'Ipswich': 'Ipswich Town', 'Sheffield United': 'Sheffield Utd', "Nottingham Forest": "Nott'm Forest",
         'Leeds United': 'Leeds', 'Luton Town': 'Luton', 'Leicester City': 'Leicester'}

# (1X2 columns, over/under 2.5 columns), first complete set wins
PRICES = [(('AvgH', 'AvgD', 'AvgA'), ('Avg>2.5', 'Avg<2.5')),
          (('B365H', 'B365D', 'B365A'), ('B365>2.5', 'B365<2.5'))]


def _pois(lam, kmax=12):
    k = np.arange(kmax + 1)
    lf = np.cumsum(np.log(np.maximum(k, 1)))
    return np.exp(k[None, :] * np.log(lam[:, None]) - lam[:, None] - lf[None, :])


def implied_rates(oh, od, oa, oo, ou):
    """Margin-free probabilities -> (home rate, away rate) under independent Poisson."""
    oh, od, oa, oo, ou = (np.asarray(v, dtype=float) for v in (oh, od, oa, oo, ou))
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


def parse(body: bytes) -> pd.DataFrame:
    """fixtures.csv -> Premier League rows: date, home, away (FPL names), odds rates."""
    d = pd.read_csv(io.BytesIO(body), encoding='utf-8-sig', on_bad_lines='skip')
    d = d[d.get('Div', pd.Series(dtype=str)).eq('E0')].dropna(subset=['HomeTeam', 'AwayTeam'])
    out = pd.DataFrame(index=d.index, columns=['oh', 'od', 'oa', 'oo', 'ou'], dtype=float)
    for x12, ou in PRICES:
        cols = list(x12) + list(ou)
        if all(c in d.columns for c in cols):
            vals = d[cols].apply(pd.to_numeric, errors='coerce')
            vals.columns = out.columns
            out = out.fillna(vals)
    ok = (out > 1).all(axis=1)
    d, out = d[ok], out[ok]
    if not len(d):
        return pd.DataFrame(columns=['date', 'home', 'away', 'odds_xg_home', 'odds_xg_away'])
    h, a = implied_rates(*(out[c].values for c in out.columns))
    return pd.DataFrame({'date': pd.to_datetime(d['Date'], dayfirst=True, errors='coerce').dt.normalize().values,
                         'home': d['HomeTeam'].replace(NAMES).values, 'away': d['AwayTeam'].replace(NAMES).values,
                         'odds_xg_home': h.round(3), 'odds_xg_away': a.round(3)})


def match(o: pd.DataFrame, fixtures_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    """Odds rows -> FPL fixture ids, by home and away club and a kick-off within a day (UK date) of the odds date."""
    name = teams_df.set_index('id')['name']
    f = fixtures_df[~fixtures_df['finished'].astype(str).str.lower().eq('true')].dropna(subset=['kickoff_time'])
    f = pd.DataFrame({'fixture': f['id'], 'home': f['team_h'].map(name), 'away': f['team_a'].map(name),
                      'ko': pd.to_datetime(f['kickoff_time'], utc=True).dt.tz_convert('Europe/London')
                              .dt.tz_localize(None).dt.normalize()})
    j = o.merge(f, on=['home', 'away'], how='left')
    j = j[(j['ko'] - j['date']).abs() <= pd.Timedelta(days=1)]
    missing = o.merge(j[['home', 'away']], how='left', indicator=True).query('_merge == "left_only"')
    for _, r in missing.iterrows():
        print(f'odds: no FPL fixture for {r["home"]} v {r["away"]} on {r["date"]:%Y-%m-%d}')
    return j[['fixture', 'odds_xg_home', 'odds_xg_away']].drop_duplicates('fixture')


def fetch(fixtures_df: pd.DataFrame, teams_df: pd.DataFrame, old: pd.DataFrame | None = None) -> pd.DataFrame:
    """Download the next round's odds and merge them over `old` (newest price per fixture wins), so a
    snapshot keeps the odds its projections used after the games drop out of fixtures.csv."""
    r = requests.get(URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    new = match(parse(r.content), fixtures_df, teams_df)
    new['odds_fetched_at'] = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
    if old is not None and len(old):
        new = pd.concat([old[~old['fixture'].isin(new['fixture'])], new], ignore_index=True)
    return new[COLS].sort_values('fixture').reset_index(drop=True)


def blend(model, odds, w: float = W):
    """Geometric blend of model and odds rates; the model alone where odds are missing."""
    model, odds = np.asarray(model, dtype=float), np.asarray(odds, dtype=float)
    mixed = np.exp(w * np.log(np.where(odds > 0, odds, 1)) + (1 - w) * np.log(model))
    return np.where(np.isfinite(odds) & (odds > 0), mixed, model)
