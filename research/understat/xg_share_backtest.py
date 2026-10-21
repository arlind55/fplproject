"""Does "team xG x player share of team xG" predict player xG better than a player's own per-90 rate?

Walk-forward on Understat appearances. For every appearance we predict the player's npxG (and xA)
in that match from information strictly before kick-off, multiplied by the minutes they actually
played, so only the rate is being judged (minutes are a separate model in fplvalue).

Every model gets the same team-level fixture expectation T (team attack x opponent defence x home
edge, from npxG before the match). They differ only in how the player's slice of it is estimated:

  rate   v1 style: player npxG/90 (shrunk to a position prior) x T / team's own average npxG
  share  player npxG / team npxG while he was on the pitch (shrunk to a position prior share) x T

and in how much history they use:

  S+L    this season + last season at half weight (what v1 does)
  multi  every earlier PL season, exponentially decayed by date

Writes data/xg_share_backtest.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DATA = 'data'
TUNE = ['2016/17', '2017/18', '2018/19']
TEST = ['2019/20', '2020/21', '2021/22', '2022/23', '2023/24', '2024/25', '2025/26']


def pos_group(code: str) -> str:
    if code == 'GK':
        return 'GK'
    if code.startswith('D') and not code.startswith('DM'):
        return 'D'
    if code.startswith('FW'):
        return 'F'
    return 'M'


def load() -> pd.DataFrame:
    a = pd.read_csv(f'{DATA}/appearances.csv', parse_dates=['date'])
    s = pd.read_csv(f'{DATA}/shots.csv')
    m = pd.read_csv(f'{DATA}/matches.csv')
    a = a[a['minutes'] > 0].copy()

    # primary position: most common starting position over a career
    st = a[a['position'] != 'Sub']
    modal = st.groupby('player_id')['position'].agg(lambda x: x.value_counts().index[0])
    a['pos'] = a['player_id'].map(modal).fillna('MC').map(pos_group)

    # team npxG while the player was on the pitch, from shot minutes.
    # starters: [0, minutes]; subs: [90 - minutes, end]. Stoppage time is folded into 90.
    s = s[s['situation'] != 'Penalty'].copy()
    s['minute'] = s['minute'].clip(upper=90)
    s['side'] = s['h_a']
    shot_groups = {k: (g['minute'].values, g['xG'].values) for k, g in s.groupby(['match_id', 'side'])}
    lo = np.where(a['started'] == 1, 0, 90 - a['minutes'])
    hi = np.where(a['started'] == 1, a['minutes'], 91)
    on = np.zeros(len(a))
    tot = np.zeros(len(a))
    for i, (mid, side) in enumerate(zip(a['match_id'].values, a['h_a'].values)):
        g = shot_groups.get((mid, side))
        if g is None:
            continue
        mins, xg = g
        tot[i] = xg.sum()
        on[i] = xg[(mins >= lo[i]) & (mins <= hi[i])].sum()
    a['team_np_on'] = on
    a['team_np'] = tot

    # opponent and match context
    m = m[m['played'] == 1]
    home = m.set_index('match_id')
    a['opp_id'] = np.where(a['h_a'] == 'h', a['match_id'].map(home['away_id']), a['match_id'].map(home['home_id']))
    a = a[a['match_id'].isin(home.index)]
    a['season_no'] = a['season'].str[:4].astype(int)
    return a.sort_values(['date', 'match_id']).reset_index(drop=True)


def team_expectation(a: pd.DataFrame, k: float = 6.0, lam: float = 0.5) -> pd.Series:
    """Expected team npxG per match, from npxG for/against before kick-off (this season + lam x last)."""
    tm = a.drop_duplicates(['match_id', 'team_id'])[['match_id', 'date', 'season_no', 'team_id', 'opp_id', 'h_a', 'team_np']]
    opp = tm[['match_id', 'team_id', 'team_np']].rename(columns={'team_id': 'opp_id', 'team_np': 'opp_np'})
    tm = tm.merge(opp, on=['match_id', 'opp_id']).sort_values('date')
    avg = tm['team_np'].mean()
    home_edge = tm.loc[tm['h_a'] == 'h', 'team_np'].mean() / avg

    # per team-season running sums before each match
    tm['n'] = tm.groupby(['team_id', 'season_no']).cumcount()
    for c in ('team_np', 'opp_np'):
        tm[c + '_cum'] = tm.groupby(['team_id', 'season_no'])[c].cumsum() - tm[c]
    last = tm.groupby(['team_id', 'season_no']).agg(ln=('team_np', 'size'), lf=('team_np', 'sum'), la=('opp_np', 'sum')).reset_index()
    last['season_no'] += 1
    tm = tm.merge(last, on=['team_id', 'season_no'], how='left').fillna({'ln': 0, 'lf': 0, 'la': 0})
    # promoted / new teams: prior a bit below average
    w = tm['n'] + lam * tm['ln'] + k
    tm['att'] = (tm['team_np_cum'] + lam * tm['lf'] + k * avg * np.where(tm['ln'] == 0, 0.85, 1)) / w / avg
    tm['dfn'] = (tm['opp_np_cum'] + lam * tm['la'] + k * avg * np.where(tm['ln'] == 0, 1.15, 1)) / w / avg
    d = tm.set_index(['match_id', 'team_id'])[['att', 'dfn']]
    tm = tm.join(d.rename(columns={'att': 'opp_att', 'dfn': 'opp_def'}), on=['match_id', 'opp_id'])
    tm['T'] = avg * tm['att'] * tm['opp_def'] * np.where(tm['h_a'] == 'h', home_edge, 2 - home_edge)
    tm['team_avg'] = avg * tm['att']
    return tm.set_index(['match_id', 'team_id'])[['T', 'team_avg']]


def player_history(a: pd.DataFrame, y: str, half_life_days: float) -> pd.DataFrame:
    """For each appearance, sums over the player's earlier appearances (any team) of
    y, minutes/90 and team on-pitch npxG: this season, last season, and exponentially decayed."""
    out = np.zeros((len(a), 9))
    rate = np.log(2) / half_life_days
    order = a.sort_values(['player_id', 'date']).index.values
    pid = a['player_id'].values
    seas = a['season_no'].values
    t = (a['date'] - a['date'].min()).dt.days.values.astype(float)
    yy = a[y].values
    nn = a['minutes'].values / 90
    tt = a['team_np_on'].values
    prev = None
    for i in order:
        if pid[i] != prev:
            prev = pid[i]
            cur_s, cur = None, np.zeros(3)
            by_season = {}
            dec, last_t = np.zeros(3), t[i]
        if seas[i] != cur_s:
            if cur_s is not None:
                by_season[cur_s] = cur
            cur_s, cur = seas[i], np.zeros(3)
        dec = dec * np.exp(-rate * (t[i] - last_t))
        last_t = t[i]
        out[i, 0:3] = cur
        out[i, 3:6] = by_season.get(seas[i] - 1, np.zeros(3))
        out[i, 6:9] = dec
        v = np.array([yy[i], nn[i], tt[i]])
        cur = cur + v
        dec = dec + v
    cols = [f'{p}_{q}' for p in ('s', 'l', 'd') for q in ('y', 'n', 't')]
    return pd.DataFrame(out, columns=cols, index=a.index)


def predict(a: pd.DataFrame, h: pd.DataFrame, y: str, k: float, lam: float, prior_rate: pd.Series,
            prior_share: pd.Series) -> dict:
    pr = a['pos'].map(prior_rate).values
    ps = a['pos'].map(prior_share).values
    n90 = a['minutes'].values / 90
    T, tavg = a['T'].values, a['team_avg'].values
    sl_y = h['s_y'] + lam * h['l_y']
    sl_n = h['s_n'] + lam * h['l_n']
    sl_t = h['s_t'] + lam * h['l_t']
    # share prior strength is in units of "team npxG seen on the pitch"; 1 team-90 is about 1.35
    ks = k * 1.35
    return {
        'rate S+L': ((sl_y + k * pr) / (sl_n + k)).values * T / tavg * n90,
        'rate multi': ((h['d_y'] + k * pr) / (h['d_n'] + k)).values * T / tavg * n90,
        'share S+L': ((sl_y + ks * ps) / (sl_t + ks)).values * T * n90,
        'share multi': ((h['d_y'] + ks * ps) / (h['d_t'] + ks)).values * T * n90,
    }


def poisson_dev(y, mu):
    mu = np.clip(mu, 1e-6, None)
    return 2 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu))


def evaluate(df: pd.DataFrame, preds: dict, y: str, ref: str = 'rate S+L') -> pd.DataFrame:
    rows = []
    yv = df[y].values
    base = (yv - preds[ref]) ** 2
    for name, p in preds.items():
        se = (yv - p) ** 2
        d = se - base
        # cluster the interval by player so one long career doesn't count as many independent rows
        g = pd.Series(d).groupby(df['player_id'].values).sum()
        ci = 1.96 * g.std(ddof=1) * np.sqrt(len(g)) / len(d)
        rows.append({'model': name, 'MSE x1000': 1000 * se.mean(),
                     'vs ref %': 100 * d.mean() / base.mean(), '95% CI ±': 100 * ci / base.mean(),
                     'bias (pred/actual)': p.sum() / yv.sum()})
    return pd.DataFrame(rows)


def main():
    a = load()
    a = a.join(team_expectation(a), on=['match_id', 'team_id'])
    a = a[a['pos'] != 'GK'].reset_index(drop=True)

    report = ['# Team xG x player share vs player per-90 rate', '',
              'Understat appearances, outfield players. Each appearance\'s npxG (or xA) is predicted from',
              'data before kick-off, times the minutes actually played. Every model uses the same team',
              'fixture expectation. Reference is `rate S+L`, which is how v1 works.',
              '"vs ref %" is the change in mean squared error; negative is better. CI is clustered by player.', '']

    for y in ('npxG', 'xA'):
        sub_train = a[a['season'].isin(['2014/15', '2015/16'])]
        prior_rate = sub_train.groupby('pos').apply(lambda g: g[y].sum() / (g['minutes'].sum() / 90))
        prior_share = sub_train.groupby('pos').apply(lambda g: g[y].sum() / g['team_np_on'].sum())

        # tune half-life and k on the tuning seasons, by the multi-season share model and rate model each
        best = {}
        tune = a['season'].isin(TUNE)
        grid = []
        for hl in (180, 365, 730, 1460):
            h = player_history(a, y, hl)
            for k in (2, 4, 8, 16):
                p = predict(a, h, y, k, 0.5, prior_rate, prior_share)
                for name, v in p.items():
                    grid.append((name, hl, k, ((a[y].values - v)[tune.values] ** 2).mean()))
        g = pd.DataFrame(grid, columns=['model', 'hl', 'k', 'mse'])
        for name, gg in g.groupby('model'):
            r = gg.sort_values('mse').iloc[0]
            best[name] = (int(r['hl']), float(r['k']))

        preds = {}
        for name, (hl, k) in best.items():
            h = player_history(a, y, hl)
            preds[name] = predict(a, h, y, k, 0.5, prior_rate, prior_share)[name]
        test = a['season'].isin(TEST).values
        df = a[test].reset_index(drop=True)
        P = {n: v[test] for n, v in preds.items()}
        P['blend 50/50 multi'] = 0.5 * P['rate multi'] + 0.5 * P['share multi']

        report += [f'## Target: {y}', '', 'Tuned on ' + ', '.join(TUNE) + ' (half-life days, k): ' +
                   '; '.join(f'{n} {v}' for n, v in best.items()), '']

        # context flags
        last_season_team = a.groupby(['player_id', 'season_no'])['team_id'].agg(lambda x: x.mode()[0])
        lst = df.set_index(['player_id', 'season_no']).index.map(
            lambda ix: last_season_team.get((ix[0], ix[1] - 1), np.nan))
        df['new_team'] = (np.asarray(lst, dtype=float) != df['team_id'].values) & ~np.isnan(np.asarray(lst, dtype=float))
        team_mno = a[test].drop_duplicates(['match_id', 'team_id']).copy()
        team_mno['mno'] = team_mno.groupby(['team_id', 'season']).cumcount()
        df = df.merge(team_mno[['match_id', 'team_id', 'mno']], on=['match_id', 'team_id'])
        # team attack change vs last season (actual npxG per match), to find teams whose level moved
        ts = a.drop_duplicates(['match_id', 'team_id']).groupby(['team_id', 'season_no'])['team_np'].mean()
        chg = ts / ts.groupby(level=0).shift()
        df['att_change'] = df.set_index(['team_id', 'season_no']).index.map(chg).values

        subsets = [
            ('All test appearances', np.ones(len(df), bool)),
            ('Players at a different club from last season', df['new_team'].values),
            ('First 6 team matches of a season', df['mno'].values < 6),
            ('Team npxG/match moved 20%+ from last season', (np.abs(df['att_change'].values - 1) >= 0.2)),
            ('Forwards', df['pos'].values == 'F'),
            ('Midfielders', df['pos'].values == 'M'),
            ('Defenders', df['pos'].values == 'D'),
        ]
        for title, mask in subsets:
            mask = np.asarray(mask, bool)
            r = evaluate(df[mask].reset_index(drop=True), {n: v[mask] for n, v in P.items()}, y)
            report += [f'### {title} ({mask.sum():,} appearances)', '', r.to_markdown(index=False, floatfmt='.3f'), '']

        if y == 'npxG':
            # sanity: Salah's share of Liverpool on-pitch npxG by season
            sal = a[a['player'].str.contains('Mohamed Salah')]
            tab = sal.groupby('season').agg(mins=('minutes', 'sum'), npxG=('npxG', 'sum'), team_on=('team_np_on', 'sum'))
            tab['share'] = tab['npxG'] / tab['team_on']
            tab['npxG/90'] = tab['npxG'] / tab['mins'] * 90
            report += ['### Salah: share of Liverpool npxG while on the pitch', '',
                       tab.reset_index().to_markdown(index=False, floatfmt='.3f'), '']
            # season-to-season stability of share vs per-90 rate, same team vs moved
            ps = a.groupby(['player_id', 'season_no', 'team_id']).agg(
                mins=('minutes', 'sum'), y=('npxG', 'sum'), t=('team_np_on', 'sum')).reset_index()
            ps = ps[ps['mins'] >= 1350]
            ps['rate'] = ps['y'] / ps['mins'] * 90
            ps['share'] = ps['y'] / ps['t']
            nxt = ps.copy()
            nxt['season_no'] -= 1
            pair = ps.merge(nxt, on=['player_id', 'season_no'], suffixes=('', '_n'))
            rows = []
            for lab, pm in (('same club', pair['team_id'] == pair['team_id_n']),
                            ('changed club', pair['team_id'] != pair['team_id_n'])):
                pp = pair[pm]
                rows.append({'pairs': lab, 'n': len(pp),
                             'r(per-90 rate)': pp['rate'].corr(pp['rate_n']),
                             'r(share)': pp['share'].corr(pp['share_n'])})
            report += ['### Year-to-year repeatability (players with 1350+ minutes both seasons)', '',
                       pd.DataFrame(rows).to_markdown(index=False, floatfmt='.3f'), '']

    with open(f'{DATA}/xg_share_backtest.md', 'w') as f:
        f.write('\n'.join(report))
    print('\n'.join(report))


if __name__ == '__main__':
    main()
