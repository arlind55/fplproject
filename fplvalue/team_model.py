"""xG-based team strength ratings and a model fixture difficulty.

Built from player_history (per-player xG / xGC per match). For every team-match we take
  xG for      = sum of that team's players' expected_goals
  xG against  = the goalkeeper's expected_goals_conceded (== opposition xG)

Ratings are a multiplicative attack/defence model fitted to xG, adjusted for opponents:
  expected xG for team T against O = avg × attack(T) × defence(O) × home edge
Each match is judged against what that opponent usually allows (or creates), so a team that has
just played the top three isn't marked down for it. Matches are weighted by recency, halving every
HALF_LIFE matches and carrying on across seasons (last season's matches come in through `prev`), and
the ratings are shrunk toward average with a prior worth PRIOR_GAMES matches. Promoted teams, with
no last-season matches, are shrunk toward the average of the sides they replaced.
  attack 1.00 = league average; defence lower is better.

The Understat backtest in research/understat/fixture_backtest.py chose this set-up: against
season-to-date raw ratings split by venue (the previous version) it cut the error in predicted
team xG by about 5%, and short form windows (last 6 or 10 matches) were clearly worse than both.

Expected goals in a fixture:  home team xG = avg × att(H) × def(A) × home_edge
                              away team xG = avg × att(A) × def(H) × (2 − home_edge)
Model difficulty (1 easiest … 5 hardest), from the perspective of each team:
  attack difficulty  – how hard it is to score: low expected xG for  -> high difficulty
  defence difficulty – how hard to keep a clean sheet: high expected xG against -> high difficulty
Bands follow FPL's shape: 10% / 20% / 40% / 20% / 10% across all remaining fixture-teams.
"""
import numpy as np
import pandas as pd

from . import odds as odds_mod

HALF_LIFE = 15       # matches; a match this many games back counts half
PRIOR_GAMES = 2
PRIOR_GAMES_NO_HISTORY = 6   # when there is no last season to carry over (early backtest weeks)
BANDS = (0.10, 0.30, 0.70, 0.90)   # cumulative quantile cut points for ratings 1..5


def team_matches(history_df: pd.DataFrame, players_df: pd.DataFrame, fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per played fixture with xG for/against and goals for/against."""
    h = history_df.copy()
    # The FPL API returns xG fields as strings ("0.52"); coerce everything we aggregate to numbers.
    for c in ('expected_goals', 'expected_goals_conceded', 'minutes', 'team_h_score', 'team_a_score', 'round', 'player_id', 'fixture'):
        if c in h.columns:
            h[c] = pd.to_numeric(h[c], errors='coerce')
    h['was_home'] = h['was_home'].astype(str).str.lower().isin(('true', '1'))
    h = h.merge(players_df[['id', 'element_type']], left_on='player_id', right_on='id', suffixes=('', '_p'))
    fx = fixtures_df[['id', 'team_h', 'team_a']].rename(columns={'id': 'fixture'})
    h = h.merge(fx, on='fixture', how='inner')
    # Team from the fixture, not the player's current club (players who moved carry old matches)
    h['team'] = np.where(h['was_home'], h['team_h'], h['team_a'])
    gk = (h[(h['element_type'] == 1) & (h['minutes'] > 0)]
          .groupby(['fixture', 'team'])['expected_goals_conceded'].sum().rename('xga'))
    tm = (h.groupby(['fixture', 'team', 'round', 'was_home'])
            .agg(xg=('expected_goals', 'sum'), hs=('team_h_score', 'first'), as_=('team_a_score', 'first'))
            .reset_index().join(gk, on=['fixture', 'team']))
    other = fx.set_index('fixture')
    tm['opp'] = np.where(tm['was_home'], tm['fixture'].map(other['team_a']), tm['fixture'].map(other['team_h']))
    tm['gf'] = np.where(tm['was_home'], tm['hs'], tm['as_'])
    tm['ga'] = np.where(tm['was_home'], tm['as_'], tm['hs'])
    tm = tm[tm['gf'].notna() & tm['xga'].notna()].drop(columns=['hs', 'as_'])
    tm['round'] = tm['round'].astype(int)
    return tm


def _fit(rows: pd.DataFrame, current: list, half_life: float, k: float, iters: int = 15):
    """Opponent-adjusted attack/defence from team-match rows keyed by club code.
    rows: team, opp, order (later = more recent), was_home, xg, xga. Returns (att, def, avg, home_edge) dicts."""
    rows = rows.sort_values('order')
    ago = rows.groupby('team').cumcount(ascending=False).values
    w = 0.5 ** (ago / half_life)
    avg = float(rows['xg'].tail(380).mean())
    home_edge = float(rows.loc[rows['was_home'], 'xg'].mean() / rows['xg'].mean())
    clubs = sorted(set(rows['team']) | set(rows['opp']) | set(current))
    ix = {c: i for i, c in enumerate(clubs)}
    t = rows['team'].map(ix).values
    o = rows['opp'].map(ix).values
    hf = np.where(rows['was_home'].values, home_edge, 2 - home_edge)
    xg, xga = rows['xg'].values, rows['xga'].values
    n = len(clubs)
    # promoted clubs (no matches before this season) are shrunk toward the clubs they replaced
    new = np.array([c in set(current) and c not in set(rows.loc[rows['season'] == 'prev', 'team']) for c in clubs])
    gone = np.array([c not in set(current) for c in clubs])
    has_prev = (rows['season'] == 'prev').any()
    att, dfn = np.ones(n), np.ones(n)
    for _ in range(iters):
        pa = np.where(new & has_prev & gone.any(), att[gone].mean() if gone.any() else 1.0, 1.0)
        pd_ = np.where(new & has_prev & gone.any(), dfn[gone].mean() if gone.any() else 1.0, 1.0)
        att = ((np.bincount(t, w * xg, n) + k * avg * pa) /
               (np.bincount(t, w * dfn[o] * hf * avg, n) + k * avg))
        dfn = ((np.bincount(t, w * xga, n) + k * avg * pd_) /
               (np.bincount(t, w * att[o] * (2 - hf) * avg, n) + k * avg))
    cur = np.array([ix[c] for c in current])
    # express ratings relative to this season's clubs, so 1.00 is the average Premier League side
    ma, md = att[cur].mean(), dfn[cur].mean()
    return ({c: att[ix[c]] / ma for c in current}, {c: dfn[ix[c]] / md for c in current}, avg * ma * md, home_edge)


def team_ratings(tm: pd.DataFrame, teams_df: pd.DataFrame, prev: pd.DataFrame = None,
                 half_life: float = HALF_LIFE, k: float = None) -> pd.DataFrame:
    """tm: this season's team_matches(). prev: last season's team_matches() with `team` and `opp` already
    mapped to club codes (see fplvalue.project.last_season_matches), or None."""
    code = teams_df.set_index('id')['code']
    cur = tm.assign(team=tm['team'].map(code), opp=tm['opp'].map(code), order=1000 + tm['round'], season='cur')
    parts = [cur]
    if prev is not None and len(prev):
        parts.insert(0, prev.assign(order=prev['round'], season='prev'))
    rows = pd.concat(parts, ignore_index=True)[['team', 'opp', 'order', 'season', 'was_home', 'xg', 'xga']]
    rows = rows.dropna(subset=['team', 'opp', 'xg', 'xga'])
    rows['was_home'] = rows['was_home'].astype(bool)
    if k is None:
        k = PRIOR_GAMES if prev is not None and len(prev) else PRIOR_GAMES_NO_HISTORY
    out = teams_df[['id', 'name', 'short_name', 'code']].rename(columns={'id': 'team_id'}).set_index('team_id')
    if len(rows):
        att, dfn, avg, he = _fit(rows, list(out['code']), half_life, k)
        out['att'] = out['code'].map(att)
        out['def'] = out['code'].map(dfn)
    else:
        out['att'], out['def'], avg, he = 1.0, 1.0, 1.35, 1.1
    tot = tm.groupby('team')[['xg', 'xga', 'gf', 'ga']].sum()
    out = out.join(tm.groupby('team').size().rename('games')).join(tot)
    out['games'] = out['games'].fillna(0).astype(int)
    for c in ('xg', 'xga', 'gf', 'ga'):
        out[c] = out[c].fillna(0)
    out['finishing'] = out['gf'] - out['xg']        # goals scored above chance quality (+ = clinical / lucky)
    out['leak'] = out['ga'] - out['xga']            # goals conceded above chance quality (+ = leaky / unlucky)
    out['league_avg_xg'] = avg
    out['home_edge'] = he
    out['half_life'] = half_life
    out['prior_games'] = k
    return (out.drop(columns='code').reset_index()
               .rename(columns={'xg': 'xg_for', 'xga': 'xg_against', 'gf': 'goals_for', 'ga': 'goals_against'})
               .round(3))


def _band(series: pd.Series, harder_when_high: bool) -> pd.Series:
    """Map a numeric series to 1..5 by quantile bands. harder_when_high: high value -> 5."""
    if series.notna().sum() < 5:
        return pd.Series(3, index=series.index)
    q = series.quantile(BANDS).values
    r = np.searchsorted(q, series.values, side='right') + 1      # 1..5 ascending in value
    r = pd.Series(r, index=series.index)
    return r if harder_when_high else 6 - r


def fixture_model(fixtures_df: pd.DataFrame, ratings: pd.DataFrame, odds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per fixture: expected xG for each side and 1-5 model difficulties from each side's view.
    Unplayed fixtures with bookmaker odds (odds.py: fixture, odds_xg_home, odds_xg_away) blend them in at
    odds.W; xg_*_model keeps the team model alone."""
    r = ratings.set_index('team_id')
    avg = float(ratings['league_avg_xg'].iloc[0]) if len(ratings) else 1.3
    f = fixtures_df[['id', 'event', 'kickoff_time', 'finished', 'team_h', 'team_a', 'team_h_difficulty', 'team_a_difficulty']].copy()
    for c in ('team_h_difficulty', 'team_a_difficulty', 'event'):
        f[c] = pd.to_numeric(f[c], errors='coerce')
    f = f[f['team_h'].isin(r.index) & f['team_a'].isin(r.index)]
    he = float(ratings['home_edge'].iloc[0]) if len(ratings) else 1.1
    f['xg_home'] = avg * r.loc[f['team_h'], 'att'].values * r.loc[f['team_a'], 'def'].values * he
    f['xg_away'] = avg * r.loc[f['team_a'], 'att'].values * r.loc[f['team_h'], 'def'].values * (2 - he)
    f['xg_home_model'], f['xg_away_model'] = f['xg_home'], f['xg_away']
    f['odds'] = False
    if odds is not None and len(odds):
        o = odds.set_index('fixture')
        live = f['id'].isin(o.index) & ~f['finished'].astype(str).str.lower().eq('true')
        ids = f.loc[live, 'id']
        f.loc[live, 'xg_home'] = odds_mod.blend(f.loc[live, 'xg_home'], ids.map(o['odds_xg_home']))
        f.loc[live, 'xg_away'] = odds_mod.blend(f.loc[live, 'xg_away'], ids.map(o['odds_xg_away']))
        f.loc[live, 'odds'] = True
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
    for c in ('xg_home', 'xg_away', 'xg_home_model', 'xg_away_model', 'cs_prob_home', 'cs_prob_away'):
        f[c] = f[c].round(3)
    return f.sort_values(['event', 'kickoff_time', 'id'], na_position='last').reset_index(drop=True)


def build(history_df, players_df, fixtures_df, teams_df, prev=None, odds=None):
    tm = team_matches(history_df, players_df, fixtures_df)
    ratings = team_ratings(tm, teams_df, prev=prev)
    model = fixture_model(fixtures_df, ratings, odds)
    return ratings, model
