"""How should team ratings for the fixture model be built?

Walk-forward test on Understat team matches. At each weekly cutoff, team attack/defence ratings are
built from matches before the cutoff only, and used to predict every match 1, 5 and 10 weeks later:

    expected xG for team T v opponent O = league avg x att(T) x [def(O) x home edge]^alpha

Scored on the team's actual xG in the match (MSE), actual goals (Poisson deviance) and clean sheets
(log loss with P(CS) = exp(-expected xG against)).

Variants
  v1         what fplvalue/team_model.py does: season to date, home and away ratings kept separate,
             prior worth 6 matches = last season's rating pulled 40% toward average (promoted teams:
             average of the relegated sides)
  season     season to date, one rating per team + a league home edge, same prior
  last N     a team's last N league matches, across seasons
  decay H    every earlier match, weight halves every H matches
  +adj       ratings adjusted for opponent strength (each xG divided by what the opponent usually
             allows), solved iteratively, i.e. a Poisson attack/defence model fitted to xG

Writes data/fixture_backtest.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DATA = 'data'
TUNE = ['2016/17', '2017/18', '2018/19', '2019/20']
TEST = ['2020/21', '2021/22', '2022/23', '2023/24', '2024/25', '2025/26']
HORIZONS = (1, 5, 10)


def load() -> pd.DataFrame:
    t = pd.read_csv(f'{DATA}/team_matches.csv', parse_dates=['date'])
    m = pd.read_csv(f'{DATA}/matches.csv', parse_dates=['datetime'])
    m = m[m['played'] == 1]
    h = m[['datetime', 'home_id', 'away_id']].rename(columns={'datetime': 'date', 'home_id': 'team_id', 'away_id': 'opp_id'})
    a = m[['datetime', 'away_id', 'home_id']].rename(columns={'datetime': 'date', 'away_id': 'team_id', 'home_id': 'opp_id'})
    t = t.merge(pd.concat([h, a]), on=['date', 'team_id'], how='inner')
    t['home'] = (t['venue'] == 'h').astype(int)
    t['season_no'] = t['season'].str[:4].astype(int)
    origin = t['date'].min().normalize() - pd.Timedelta(days=t['date'].min().weekday())
    t['week'] = ((t['date'] - origin).dt.days // 7).astype(int)
    t = t.sort_values(['team_id', 'date']).reset_index(drop=True)
    t['idx'] = t.groupby('team_id').cumcount()
    t['cs'] = (t['goals_against'] == 0).astype(int)
    return t


def season_prior(t: pd.DataFrame, season_no: int, avg: float, shrink: float = 0.6) -> dict:
    """Last season's raw ratings pulled toward 1; promoted teams get the relegated sides' average."""
    last = t[t['season_no'] == season_no - 1]
    cur_teams = set(t.loc[t['season_no'] == season_no, 'team_id'])
    if not len(last):
        return {}
    la = last['xg'].mean()
    r = last.groupby('team_id').agg(att=('xg', 'mean'), dfn=('xga', 'mean')) / la
    gone = [i for i in r.index if i not in cur_teams]
    fill = r.loc[gone].mean() if gone else pd.Series({'att': 0.85, 'dfn': 1.15})
    out = {}
    for tid in cur_teams:
        v = r.loc[tid] if tid in r.index else fill
        out[tid] = (1 + shrink * (v['att'] - 1), 1 + shrink * (v['dfn'] - 1))
    return out


def ratings_at(t: pd.DataFrame, week: int, season_no: int, cfg: dict, prior: dict):
    """Ratings from matches before `week`. Returns (att, dfn, att_home, att_away, dfn_home, dfn_away, avg, home_edge)."""
    past = t[t['week'] < week]
    if cfg['window'] == 'season':
        past = past[past['season_no'] == season_no]
        w = np.ones(len(past))
    else:
        last_idx = past.groupby('team_id')['idx'].transform('max')
        ago = (last_idx - past['idx']).values
        if cfg['window'] == 'last':
            w = (ago < cfg['n']).astype(float)
        else:
            w = 0.5 ** (ago / cfg['n'])
            w[ago > 6 * cfg['n']] = 0
    keep = w > 0
    past, w = past[keep], w[keep]
    # league level from the last 380 team-matches, so it is defined at a season start too
    recent = t[t['week'] < week].tail(760)
    avg = recent['xg'].mean()
    he = recent.loc[recent['home'] == 1, 'xg'].mean() / avg if len(recent) else 1.1
    teams = sorted(set(t.loc[t['season_no'] == season_no, 'team_id']))
    k = cfg.get('k', 6.0)
    pa = np.array([prior.get(i, (1.0, 1.0))[0] for i in teams])
    pdf = np.array([prior.get(i, (1.0, 1.0))[1] for i in teams])
    if cfg['window'] != 'season':
        # rolling windows carry history across seasons; promoted teams have none, so they get the prior
        pa_use = np.array([prior.get(i, (0.85, 1.15))[0] if not (past['team_id'] == i).any() else 1.0 for i in teams])
        pd_use = np.array([prior.get(i, (0.85, 1.15))[1] if not (past['team_id'] == i).any() else 1.0 for i in teams])
    else:
        pa_use, pd_use = pa, pdf
    pos = {tid: j for j, tid in enumerate(teams)}
    ti = past['team_id'].map(pos)
    oi = past['opp_id'].map(pos)
    ok = ti.notna().values
    ti = ti.fillna(0).astype(int).values
    oi_ok = oi.notna().values
    oi = oi.fillna(0).astype(int).values
    xg, xga, home = past['xg'].values, past['xga'].values, past['home'].values
    nT = len(teams)
    hf = np.where(home == 1, he, 2 - he)
    att, dfn = np.ones(nT), np.ones(nT)
    iters = 12 if cfg.get('adj') else 1
    for _ in range(iters):
        if cfg.get('adj'):
            # expected exposure: what an average team would get against this opponent at this venue
            e_for = np.where(oi_ok, dfn[oi], 1.0) * hf
            e_against = np.where(oi_ok, att[oi], 1.0) * (2 - hf)
        else:
            e_for = np.ones(len(past))
            e_against = np.ones(len(past))
        sw = w * ok
        num_a = np.bincount(ti, sw * xg, nT) + k * avg * pa_use
        den_a = np.bincount(ti, sw * e_for * avg, nT) + k * avg
        num_d = np.bincount(ti, sw * xga, nT) + k * avg * pd_use
        den_d = np.bincount(ti, sw * e_against * avg, nT) + k * avg
        att, dfn = num_a / den_a, num_d / den_d
    out = {'teams': pos, 'att': att, 'dfn': dfn, 'avg': avg, 'he': he}
    if cfg.get('venue_split'):
        for v, name in ((1, 'home'), (0, 'away')):
            sel = (home == v)
            na = np.bincount(ti[sel], (w * ok)[sel] * xg[sel], nT) + k * avg * pa_use
            nd = np.bincount(ti[sel], (w * ok)[sel] * xga[sel], nT) + k * avg * pd_use
            dn = np.bincount(ti[sel], (w * ok)[sel], nT) * avg + k * avg
            out['att_' + name], out['dfn_' + name] = na / dn, nd / dn
    return out


def predict(r: dict, team: np.ndarray, opp: np.ndarray, home: np.ndarray, cfg: dict, alpha: float):
    p = r['teams']
    ti = np.array([p.get(i, -1) for i in team])
    oi = np.array([p.get(i, -1) for i in opp])
    good = (ti >= 0) & (oi >= 0)
    ti, oi = np.where(good, ti, 0), np.where(good, oi, 0)
    if cfg.get('venue_split'):
        # v1: home team uses its home attack and the visitor's away defence, and vice versa
        att = np.where(home == 1, r['att_home'][ti], r['att_away'][ti])
        mult = np.where(home == 1, r['dfn_away'][oi], r['dfn_home'][oi])
    else:
        att = r['att'][ti]
        mult = r['dfn'][oi] * np.where(home == 1, r['he'], 2 - r['he'])
    lam = r['avg'] * att * mult ** alpha
    return np.where(good, lam, np.nan)


def poisson_dev(y, mu):
    mu = np.clip(mu, 1e-6, None)
    return 2 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu))


def run(t: pd.DataFrame, cfg: dict, alphas=(1.0,)) -> pd.DataFrame:
    """Predictions for every match 1/5/10 weeks after each weekly cutoff, for each alpha."""
    weeks = np.sort(t['week'].unique())
    by_week = {w: g for w, g in t.groupby('week')}
    season_of_week = t.groupby('week')['season_no'].min()
    rows = []
    prior_cache = {}
    for c in weeks:
        s = int(season_of_week[c])
        if s < 2015:
            continue
        if s not in prior_cache:
            prior_cache[s] = season_prior(t, s, t['xg'].mean())
        r = ratings_at(t, c, s, cfg, prior_cache[s])
        for h in HORIZONS:
            tw = c + h - 1
            if tw not in by_week:
                continue
            g = by_week[tw]
            if int(g['season_no'].min()) != s:
                continue        # do not predict across a summer
            for a in alphas:
                lam_for = predict(r, g['team_id'].values, g['opp_id'].values, g['home'].values, cfg, a)
                lam_ag = predict(r, g['opp_id'].values, g['team_id'].values, 1 - g['home'].values, cfg, a)
                rows.append(pd.DataFrame({'season': g['season'].values, 'h': h, 'alpha': a, 'team_id': g['team_id'].values,
                                          'week': tw, 'n_in_season': g['match_no'].values,
                                          'xg': g['xg'].values, 'gf': g['goals_for'].values, 'cs': g['cs'].values,
                                          'lam': lam_for, 'lam_ag': lam_ag}))
    return pd.concat(rows, ignore_index=True)


def score(p: pd.DataFrame) -> pd.Series:
    p = p.dropna(subset=['lam', 'lam_ag'])
    pcs = np.clip(np.exp(-p['lam_ag']), 1e-4, 1 - 1e-4)
    return pd.Series({'n': len(p),
                      'xG MSE': ((p['xg'] - p['lam']) ** 2).mean(),
                      'goals dev': poisson_dev(p['gf'].values, p['lam'].values).mean(),
                      'CS log loss': -(p['cs'] * np.log(pcs) + (1 - p['cs']) * np.log(1 - pcs)).mean()})


def main():
    t = load()
    configs = {
        'v1 (season, home/away split)': {'window': 'season', 'venue_split': True},
        'season to date': {'window': 'season'},
        'season to date +adj': {'window': 'season', 'adj': True},
    }
    for n in (6, 10, 19, 38):
        configs[f'last {n}'] = {'window': 'last', 'n': n, 'k': 2.0}
        configs[f'last {n} +adj'] = {'window': 'last', 'n': n, 'k': 2.0, 'adj': True}
    for hl in (5, 10, 20, 40):
        configs[f'decay {hl}'] = {'window': 'decay', 'n': hl, 'k': 2.0}
        configs[f'decay {hl} +adj'] = {'window': 'decay', 'n': hl, 'k': 2.0, 'adj': True}

    res = {}
    for name, cfg in configs.items():
        res[name] = run(t, cfg)
        print('done', name, flush=True)

    # league average only and "no opponent effect" baselines
    base = res['season to date'].copy()
    base['lam'] = base.groupby('week')['lam'].transform('mean')
    base['lam_ag'] = base['lam']
    res['league average only'] = base

    report = ['# Fixture model: how to build team ratings', '',
              'Walk-forward on Understat, 2016/17 to 2025/26. At each weekly cutoff, ratings use only earlier',
              'matches and predict every match 1, 5 and 10 weeks later (within the same season). Lower is better',
              'for every column. `xG MSE` scores expected xG against the xG the team actually produced;',
              '`goals dev` is Poisson deviance against goals scored; `CS log loss` scores P(clean sheet) = e^(-xG against).',
              'Tuning seasons ' + ', '.join(TUNE) + '; test seasons ' + ', '.join(TEST) + '.', '']

    def table(seasons, h, extra=None):
        rows = []
        for name, p in res.items():
            q = p[(p['h'] == h) & p['season'].isin(seasons)]
            if extra is not None:
                q = q[extra(q)]
            s = score(q)
            s['model'] = name
            rows.append(s)
        d = pd.DataFrame(rows)
        ref = d.loc[d['model'] == 'v1 (season, home/away split)'].iloc[0]
        for c in ('xG MSE', 'goals dev', 'CS log loss'):
            d[c + ' vs v1 %'] = 100 * (d[c] / ref[c] - 1)
        d = d.sort_values('xG MSE')
        return d[['model', 'n', 'xG MSE', 'xG MSE vs v1 %', 'goals dev vs v1 %', 'CS log loss vs v1 %']]

    for seasons, lab in ((TUNE, 'tuning seasons'), (TEST, 'test seasons')):
        for h in HORIZONS:
            report += [f'## {lab}, {h} week{"s" if h > 1 else ""} ahead', '',
                       table(seasons, h).to_markdown(index=False, floatfmt='.3f'), '']
    report += ['## Test seasons, 1 week ahead, first 8 matches of a season only', '',
               table(TEST, 1, lambda q: q['n_in_season'] <= 8).to_markdown(index=False, floatfmt='.3f'), '',
               '## Test seasons, 1 week ahead, matches 20 onwards', '',
               table(TEST, 1, lambda q: q['n_in_season'] >= 20).to_markdown(index=False, floatfmt='.3f'), '']

    # how strongly to apply the fixture effect: refit alpha for v1 and the best tuning-season model
    tune1 = {n: score(p[(p['h'] == 1) & p['season'].isin(TUNE)])['xG MSE'] for n, p in res.items() if n != 'league average only'}
    best = min(tune1, key=tune1.get)
    rows = []
    for name in ('v1 (season, home/away split)', best):
        p = run(t, configs[name], alphas=(0.0, 0.5, 0.75, 1.0, 1.25, 1.5))
        for a, g in p[(p['h'] == 1)].groupby('alpha'):
            st, ss = score(g[g['season'].isin(TUNE)]), score(g[g['season'].isin(TEST)])
            rows.append({'model': name, 'alpha': a, 'tune xG MSE': st['xG MSE'], 'test xG MSE': ss['xG MSE'],
                         'test goals dev': ss['goals dev']})
    report += [f'## Fixture strength: expected xG = avg x att x (opponent defence x home)^alpha', '',
               'alpha = 0 ignores the opponent; 1 is what v1 uses. 1 week ahead.', '',
               pd.DataFrame(rows).to_markdown(index=False, floatfmt='.4f'), '',
               f'Best on tuning seasons (1 week ahead, xG MSE): **{best}**.', '']

    with open(f'{DATA}/fixture_backtest.md', 'w') as f:
        f.write('\n'.join(report))
    print('\n'.join(report))


if __name__ == '__main__':
    main()
