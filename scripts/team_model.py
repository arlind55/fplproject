"""xG-based team strength ratings and a model fixture difficulty.

Built from player_history (per-player xG / xGC per match). For every team-match we take
  xG for      = sum of that team's players' expected_goals
  xG against  = the goalkeeper's expected_goals_conceded (== opposition xG)
Ratings are multiplicative and shrunk toward the league average with a prior worth
PRIOR_GAMES matches, so early-season numbers stay sane:
  attack  = (sum xG for  + K*avg) / ((n + K) * avg)      1.00 = league average
  defence = (sum xG against + K*avg) / ((n + K) * avg)   lower is better
Home and away versions use only matches at that venue (with the same prior).

Expected goals in a fixture:  home team xG = avg * att_home(H) * def_away(A)
                              away team xG = avg * att_away(A) * def_home(H)
Model difficulty (1 easiest … 5 hardest), from the perspective of each team:
  attack difficulty  – how hard it is to score: low expected xG for  -> high difficulty
  defence difficulty – how hard to keep a clean sheet: high expected xG against -> high difficulty
Bands follow FPL's shape: 10% / 20% / 40% / 20% / 10% across all remaining fixture-teams.
"""
import numpy as np
import pandas as pd

PRIOR_GAMES = 6
BANDS = (0.10, 0.30, 0.70, 0.90)   # cumulative quantile cut points for ratings 1..5


def team_matches(history_df: pd.DataFrame, players_df: pd.DataFrame, fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per played fixture with xG for/against and goals for/against."""
    h = history_df.merge(players_df[['id', 'element_type']], left_on='player_id', right_on='id', suffixes=('', '_p'))
    fx = fixtures_df[['id', 'team_h', 'team_a']].rename(columns={'id': 'fixture'})
    h = h.merge(fx, on='fixture', how='inner')
    # Team from the fixture, not the player's current club (players who moved carry old matches)
    h['team'] = np.where(h['was_home'], h['team_h'], h['team_a'])
    gk = (h[(h['element_type'] == 1) & (h['minutes'] > 0)]
          .groupby(['fixture', 'team'])['expected_goals_conceded'].sum().rename('xga'))
    tm = (h.groupby(['fixture', 'team', 'round', 'was_home'])
            .agg(xg=('expected_goals', 'sum'), hs=('team_h_score', 'first'), as_=('team_a_score', 'first'))
            .reset_index().join(gk, on=['fixture', 'team']))
    tm['gf'] = np.where(tm['was_home'], tm['hs'], tm['as_'])
    tm['ga'] = np.where(tm['was_home'], tm['as_'], tm['hs'])
    tm = tm[tm['gf'].notna() & tm['xga'].notna()].drop(columns=['hs', 'as_'])
    tm['round'] = tm['round'].astype(int)
    return tm


def _rate(g: pd.DataFrame, avg: float, k: int) -> pd.Series:
    n = len(g)
    return pd.Series({
        'att': (g['xg'].sum() + k * avg) / ((n + k) * avg),
        'def': (g['xga'].sum() + k * avg) / ((n + k) * avg),
        'n': n,
    })


def team_ratings(tm: pd.DataFrame, teams_df: pd.DataFrame, k: int = PRIOR_GAMES) -> pd.DataFrame:
    avg = float(tm['xg'].mean()) if len(tm) else 1.3
    base = tm.groupby('team')[['xg', 'xga']].apply(lambda g: _rate(g, avg, k))
    home = tm[tm['was_home']].groupby('team')[['xg', 'xga']].apply(lambda g: _rate(g, avg, k))[['att', 'def']].add_suffix('_home')
    away = tm[~tm['was_home']].groupby('team')[['xg', 'xga']].apply(lambda g: _rate(g, avg, k))[['att', 'def']].add_suffix('_away')
    tot = tm.groupby('team')[['xg', 'xga', 'gf', 'ga']].sum()
    out = teams_df[['id', 'name', 'short_name']].rename(columns={'id': 'team_id'}).set_index('team_id')
    out = out.join(base).join(home).join(away).join(tot)
    for c in ('att', 'def', 'att_home', 'def_home', 'att_away', 'def_away'):
        out[c] = out[c].fillna(1.0)
    out['n'] = out['n'].fillna(0).astype(int)
    for c in ('xg', 'xga', 'gf', 'ga'):
        out[c] = out[c].fillna(0)
    out['finishing'] = out['gf'] - out['xg']        # goals scored above chance quality (+ = clinical / lucky)
    out['leak'] = out['ga'] - out['xga']            # goals conceded above chance quality (+ = leaky / unlucky)
    out['league_avg_xg'] = avg
    out['prior_games'] = k
    return (out.reset_index()
               .rename(columns={'n': 'games', 'xg': 'xg_for', 'xga': 'xg_against', 'gf': 'goals_for', 'ga': 'goals_against'})
               .round(3))


def _band(series: pd.Series, harder_when_high: bool) -> pd.Series:
    """Map a numeric series to 1..5 by quantile bands. harder_when_high: high value -> 5."""
    if series.notna().sum() < 5:
        return pd.Series(3, index=series.index)
    q = series.quantile(BANDS).values
    r = np.searchsorted(q, series.values, side='right') + 1      # 1..5 ascending in value
    r = pd.Series(r, index=series.index)
    return r if harder_when_high else 6 - r


def fixture_model(fixtures_df: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Per fixture: expected xG for each side and 1-5 model difficulties from each side's view."""
    r = ratings.set_index('team_id')
    avg = float(ratings['league_avg_xg'].iloc[0]) if len(ratings) else 1.3
    f = fixtures_df[['id', 'event', 'kickoff_time', 'finished', 'team_h', 'team_a', 'team_h_difficulty', 'team_a_difficulty']].copy()
    f = f[f['team_h'].isin(r.index) & f['team_a'].isin(r.index)]
    f['xg_home'] = avg * r.loc[f['team_h'], 'att_home'].values * r.loc[f['team_a'], 'def_away'].values
    f['xg_away'] = avg * r.loc[f['team_a'], 'att_away'].values * r.loc[f['team_h'], 'def_home'].values
    f['cs_prob_home'] = np.exp(-f['xg_away'])       # P(home team clean sheet)
    f['cs_prob_away'] = np.exp(-f['xg_home'])
    # Bands computed across unplayed fixtures only (both sides pooled), so the scale reflects the remaining calendar
    pool = f[~f['finished'].astype(str).str.lower().eq('true')]
    att_pool = pd.concat([pool['xg_home'], pool['xg_away']])
    def_pool = pd.concat([pool['xg_away'], pool['xg_home']])
    aq = att_pool.quantile(BANDS).values; dq = def_pool.quantile(BANDS).values
    def band_att(v): return 6 - (np.searchsorted(aq, v, side='right') + 1)   # more xG for -> easier
    def band_def(v): return np.searchsorted(dq, v, side='right') + 1          # more xG against -> harder
    f['att_diff_home'] = band_att(f['xg_home'].values); f['att_diff_away'] = band_att(f['xg_away'].values)
    f['def_diff_home'] = band_def(f['xg_away'].values); f['def_diff_away'] = band_def(f['xg_home'].values)
    f['model_diff_home'] = np.rint((f['att_diff_home'] + f['def_diff_home']) / 2).astype(int)
    f['model_diff_away'] = np.rint((f['att_diff_away'] + f['def_diff_away']) / 2).astype(int)
    f['blend_diff_home'] = np.rint((f['team_h_difficulty'].fillna(3) + f['model_diff_home']) / 2).astype(int)
    f['blend_diff_away'] = np.rint((f['team_a_difficulty'].fillna(3) + f['model_diff_away']) / 2).astype(int)
    for c in ('xg_home', 'xg_away', 'cs_prob_home', 'cs_prob_away'):
        f[c] = f[c].round(3)
    return f.sort_values(['event', 'kickoff_time', 'id'], na_position='last').reset_index(drop=True)


def build(history_df, players_df, fixtures_df, teams_df):
    tm = team_matches(history_df, players_df, fixtures_df)
    ratings = team_ratings(tm, teams_df)
    model = fixture_model(fixtures_df, ratings)
    return ratings, model
