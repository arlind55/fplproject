"""Transparent expected-points projection model.

For every player and every fixture in the projection window:

    expected points = availability × Σ components

    minutes      p_start, p_sub, minutes-when-started, share of starts reaching 60'   (recency-weighted, shrunk)
    appearance   P(plays) + P(60+)
    goals        npxG/90 × fixture attack multiplier × E[minutes]/90 × goal points      (open play: penalties removed)
    pens         team penalties/match × fixture attack multiplier × P(first available taker) × 0.80 × goal points
    assists      xA/90  × fixture attack multiplier × E[minutes]/90 × 3
    clean sheet  P(60+) × e^(−fixture xG against) × CS points
    conceded     −E[floor(goals conceded while on pitch / 2)]                          (GK, DEF)
    saves        E[floor(saves / 3)]                                                    (GK)
    DefCon       P(60+) × P(hits the DefCon threshold | 60+) × 2                         (DEF, MID, FWD)
    bonus        bonus/90 × E[minutes]/90, nudged by fixture
    cards        −yellows/90 × E[minutes]/90

Every per-90 rate is a shrinkage estimate
    rate = (this season + λ·last season + K·price prior) / (this-season 90s + λ·last-season 90s + K)
where the price prior is a per-position linear fit of the rate on price: FPL prices are a good
early-season prior for how much a player will be involved.

Fixture multipliers come from the xG team model (team_model.py): the team's expected xG in that
fixture divided by its average, so a player's rate goes up against weak defences and down against
strong ones.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from math import lgamma

import numpy as np
import pandas as pd

from . import overrides, team_model
from .load import DATA, Snapshot

GOAL = {'GKP': 10, 'DEF': 6, 'MID': 5, 'FWD': 4}
CS_PTS = {'GKP': 4, 'DEF': 4, 'MID': 1, 'FWD': 0}
DEFCON_T = {'GKP': 999, 'DEF': 10, 'MID': 12, 'FWD': 12}
COMPONENTS = ['app', 'goals', 'pens', 'assists', 'cs', 'conceded', 'saves', 'defcon', 'bonus', 'cards']


@dataclass
class Params:
    # shrinkage strength, in 90-minute units (or 60+ games for DefCon, starts for minutes)
    k_xg: float = 4.0
    k_xa: float = 4.0
    k_saves: float = 6.0
    k_bonus: float = 8.0
    k_cards: float = 10.0
    k_defcon: float = 4.0
    k_min: float = 0.5            # minutes-model prior weight, in (recency-weighted) matches
    lam: float = 0.5              # weight of a last-season 90 relative to a this-season 90
    decay: float = 0.7            # recency weight per match for the minutes model
    n_recent: int = 6             # matches looked at for the minutes model
    goal_cal: float = 0.93        # FPL non-penalty goals per Opta npxG (2025/26)
    pen_conv: float = 0.80        # penalties scored per penalty taken (Understat, 1,175 PL penalties since 2014/15)
    k_pens: float = 20.0          # team penalty-rate prior weight, in matches
    assist_cal: float = 1.38      # FPL assists per Opta xA (FPL counts rebounds, pens won, deflections)
    cs_cal: float = 1.0           # multiplier on xG against inside e^(−xGA)
    horizon: int = 5


# ── helpers ─────────────────────────────────────────────────────────────────────────────────────
def _poisson_pmf(mu: np.ndarray, kmax: int = 40) -> np.ndarray:
    mu = np.clip(np.asarray(mu, float), 1e-9, None)
    k = np.arange(kmax + 1)
    lg = np.array([lgamma(i + 1) for i in k])
    return np.exp(-mu[:, None] + k[None, :] * np.log(mu[:, None]) - lg[None, :])


def floor_mean(mu, d: int) -> np.ndarray:
    """E[floor(N / d)] for N ~ Poisson(mu)."""
    mu = np.asarray(mu, float)
    if mu.size == 0:
        return mu
    pmf = _poisson_pmf(mu)
    return (pmf * (np.arange(pmf.shape[1]) // d)[None, :]).sum(1)


def _fit_linear(tab: pd.DataFrame, y: str, w: str, min_w: float) -> dict:
    """Per position: y ≈ a + b·price, weighted by w, on rows with w >= min_w. Returns {pos: (a, b)}."""
    out = {}
    for pos in ('GKP', 'DEF', 'MID', 'FWD'):
        g = tab[(tab['pos'] == pos) & (tab[w] >= min_w) & tab[y].notna()]
        if len(g) < 3:
            out[pos] = (float(tab.loc[tab['pos'] == pos, y].mean() if (tab['pos'] == pos).any() else 0.0), 0.0)
            continue
        if len(g) < 12 or g['price'].nunique() < 3:
            out[pos] = (float(np.average(g[y], weights=g[w])), 0.0)
            continue
        sw = np.sqrt(g[w].values)
        X = np.c_[np.ones(len(g)), g['price'].values]
        a, b = np.linalg.lstsq(X * sw[:, None], g[y].values * sw, rcond=None)[0]
        out[pos] = (float(a), float(b))
    return out


def _apply_linear(coef: dict, pos: pd.Series, price: pd.Series, lo=0.0, hi=None) -> pd.Series:
    a = pos.map(lambda p: coef.get(p, (0, 0))[0]).astype(float)
    b = pos.map(lambda p: coef.get(p, (0, 0))[1]).astype(float)
    return (a + b * price).clip(lower=lo, upper=hi)


# ── the model state: what we know at the moment of projecting ───────────────────────────────────
@dataclass
class State:
    players: pd.DataFrame          # id, code, web_name, team, pos, price, status, chance, news, (+ display cols)
    history: pd.DataFrame          # per-player per-fixture rows from finished gameweeks
    fixtures: pd.DataFrame         # whole season, 'finished' consistent with first_gw
    teams: pd.DataFrame
    events: pd.DataFrame
    first_gw: int
    prior_history: pd.DataFrame | None = None     # last season, with 'code' and 'pos'
    prior_players: pd.DataFrame | None = None     # last season end: code, pos, price
    team_prev: pd.DataFrame | None = None         # last season's team matches, clubs as codes
    overrides: pd.DataFrame | None = None         # news-driven start probabilities (overrides.py), live only


def last_season_matches(last: Snapshot | None) -> pd.DataFrame | None:
    """Last season's team matches with clubs as FPL codes (ids change between seasons), carried into
    this season's team ratings with a recency weight."""
    if last is None or not len(last.history):
        return None
    tm = team_model.team_matches(last.history, last.players[['id', 'element_type']], last.fixtures)
    code = last.teams.set_index('id')['code']
    return tm.assign(team=tm['team'].map(code), opp=tm['opp'].map(code))


def _players_frame(snap: Snapshot) -> pd.DataFrame:
    p = snap.players.copy()
    p['price'] = p['now_cost']
    p['chance'] = p['chance_of_playing_next_round']
    t = snap.teams.set_index('id')['short_name']
    p['team_short_name'] = p['team'].map(t)
    keep = ['id', 'code', 'web_name', 'team', 'team_short_name', 'pos', 'price', 'status', 'chance', 'news',
            'selected_by_percent', 'ep_next', 'penalties_order']
    return p[[c for c in keep if c in p.columns]]


def _prior_tables(last: Snapshot | None):
    if last is None or not len(last.history):
        return None, None
    lp = last.players[['id', 'code', 'pos', 'now_cost']].rename(columns={'now_cost': 'price'})
    lh = last.history.merge(lp[['id', 'code', 'pos']], left_on='player_id', right_on='id', how='inner')
    return lh, lp


def state_from_snapshot(snap: Snapshot, last: Snapshot | None = None) -> State:
    """Live state: everything played so far, projecting from the next gameweek still open for picks
    (so a mid-gameweek refresh does not project the week already under way)."""
    played = snap.last_finished_gw     # a gameweek under way has partial rows (and 0-minute ones for unplayed games)
    lh, lp = _prior_tables(last)
    return State(_players_frame(snap), snap.history[snap.history['round'] <= played].copy(), snap.fixtures,
                 snap.teams, snap.events, snap.next_open_gw, lh, lp, last_season_matches(last))


def state_as_of(snap: Snapshot, gw: int, avail: Snapshot | None = None, last: Snapshot | None = None) -> State:
    """Backtest state: rewind a finished season to just before gameweek `gw`.
    Price and club come from each player's latest history row; availability from `avail` (a snapshot taken
    before that deadline) when given, otherwise everyone is assumed available."""
    h = snap.history[snap.history['round'] < gw].copy()
    f = snap.fixtures.copy()
    f['finished'] = f['event'] < gw
    last_row = h.sort_values(['round', 'kickoff_time']).groupby('player_id').tail(1).set_index('player_id')
    p = snap.players[['id', 'code', 'web_name', 'pos', 'selected_by_percent']].copy()
    p = p[p['id'].isin(last_row.index)]
    p['team'] = p['id'].map(last_row['team']).astype(int)
    p['price'] = p['id'].map(last_row['value']) / 10
    p['team_short_name'] = p['team'].map(snap.teams.set_index('id')['short_name'])
    p['status'], p['chance'], p['news'], p['ep_next'] = 'a', np.nan, '', np.nan
    p['penalties_order'] = p['id'].map(snap.players.set_index('id')['penalties_order']) if 'penalties_order' in snap.players else np.nan
    if avail is not None:
        a = avail.players.set_index('id')
        for c_src, c_dst in (('status', 'status'), ('chance_of_playing_next_round', 'chance'), ('news', 'news'),
                             ('ep_next', 'ep_next'), ('selected_by_percent', 'selected_by_percent'),
                             ('penalties_order', 'penalties_order')):
            if c_src in a.columns:
                p[c_dst] = p['id'].map(a[c_src]).where(p['id'].isin(a.index), p[c_dst])
        p['status'] = p['status'].fillna('a')
        p['news'] = p['news'].fillna('')
    lh, lp = _prior_tables(last)
    return State(p.reset_index(drop=True), h, f, snap.teams, snap.events, gw, lh, lp,
                 last_season_matches(last))


# ── component models ────────────────────────────────────────────────────────────────────────────
# ── penalties ───────────────────────────────────────────────────────────────────────────────────
_PENS: pd.DataFrame | None = None


def penalties() -> pd.DataFrame:
    """Every penalty taken, matched to FPL fixture + player (data/penalties.csv, from scripts/penalties.py)."""
    global _PENS
    if _PENS is None:
        f = DATA / 'penalties.csv'
        _PENS = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=['season', 'fixture', 'element', 'team_code', 'xg'])
    return _PENS


def season_label(events: pd.DataFrame) -> str:
    y = pd.to_datetime(events['deadline_time'], utc=True).min().year
    return f"{y}/{str(y + 1)[2:]}"


def strip_pens(h: pd.DataFrame, season: str) -> pd.DataFrame:
    """History with each penalty's xG taken out of expected_goals (so it becomes npxG) and a pens column."""
    pz = penalties()
    pz = pz[pz['season'] == season].groupby(['fixture', 'element']).agg(pens=('xg', 'size'), pen_xg=('xg', 'sum'))
    h = h.join(pz, on=['fixture', 'player_id'])
    h[['pens', 'pen_xg']] = h[['pens', 'pen_xg']].fillna(0)
    h['expected_goals'] = (h['expected_goals'] - h['pen_xg']).clip(lower=0)
    return h


def team_pen_rate(state: 'State', prm: 'Params') -> pd.Series:
    """Penalties per match by team (index: team id), this season + λ·last season, shrunk to the league rate."""
    pz = penalties(); season = season_label(state.events)
    y = int(season[:4]); last = f"{y - 1}/{str(y)[2:]}"
    code = state.teams.set_index('id')['code']
    played = state.history.groupby('team')['fixture'].nunique().reindex(code.index).fillna(0)
    cur = pz[(pz['season'] == season) & pz['fixture'].isin(set(state.history['fixture']))]
    cur_n = cur.groupby('team_code').size().reindex(code.values).fillna(0).to_numpy()
    prev = pz[pz['season'] == last]
    prev_n = prev.groupby('team_code').size()
    in_last = code.isin(prev_n.index) if len(prev) else code.map(lambda c: False)
    avg = len(prev) / (20 * 38) if len(prev) else 0.12
    lam = prm.lam
    rate = (cur_n + lam * code.map(prev_n).fillna(0).to_numpy() + prm.k_pens * avg) / (
        played.to_numpy() + lam * 38 * in_last.to_numpy() + prm.k_pens)
    return pd.Series(rate, index=code.index)


def _totals(h: pd.DataFrame, pos_of: pd.Series) -> pd.DataFrame:
    """Per-player season totals the rate models need. pos_of: player_id -> pos."""
    if not len(h):
        return pd.DataFrame(columns=['mins', 'xg', 'xa', 'saves', 'bonus', 'yc', 'n60', 'hits'])
    h = h.assign(pos=h['player_id'].map(pos_of))
    h['thr'] = h['pos'].map(DEFCON_T).fillna(999)
    h['p60'] = h['minutes'] >= 60
    h['hit'] = h['p60'] & (h['defensive_contribution'] >= h['thr'])
    return h.groupby('player_id').agg(mins=('minutes', 'sum'), xg=('expected_goals', 'sum'), xa=('expected_assists', 'sum'),
                                      saves=('saves', 'sum'), bonus=('bonus', 'sum'), yc=('yellow_cards', 'sum'),
                                      n60=('p60', 'sum'), hits=('hit', 'sum'))


def rates(state: State, prm: Params) -> pd.DataFrame:
    """Shrunk per-90 rates (and DefCon hit rate per 60+ match) for every player in state.players."""
    P = state.players.set_index('id')
    season = season_label(state.events)
    cur = _totals(strip_pens(state.history, season), P['pos']).reindex(P.index).fillna(0)
    tab = cur.join(P[['pos', 'price']])
    if state.prior_history is not None:
        y = int(season[:4])
        lh = strip_pens(state.prior_history, f"{y - 1}/{str(y)[2:]}")
        ls = _totals(lh, lh.drop_duplicates('player_id').set_index('player_id')['pos'])
        ls = ls.join(lh.drop_duplicates('player_id').set_index('player_id')['code'])
        ls = ls.groupby('code').sum(numeric_only=True)
        prev = P[['code']].join(ls, on='code').drop(columns='code').fillna(0)
        # price prior fitted on last season (full sample), prices from season end
        fit_tab = ls.join(state.prior_players.drop_duplicates('code').set_index('code')[['pos', 'price']], how='inner')
        min_fit = 900
    else:
        prev = pd.DataFrame(0.0, index=P.index, columns=cur.columns)
        fit_tab = tab
        min_fit = max(90.0, cur['mins'].max() * 0.4)

    ft = fit_tab.copy()
    for s in ('xg', 'xa', 'saves', 'bonus', 'yc'):
        ft[s + '90'] = ft[s] / (ft['mins'] / 90).replace(0, np.nan)
    ft['hit_rate'] = ft['hits'] / ft['n60'].replace(0, np.nan)
    out = pd.DataFrame(index=P.index)
    lam = prm.lam
    for s, k in (('xg', prm.k_xg), ('xa', prm.k_xa), ('saves', prm.k_saves), ('bonus', prm.k_bonus), ('yc', prm.k_cards)):
        coef = _fit_linear(ft, s + '90', 'mins', min_fit)
        pp = _apply_linear(coef, P['pos'], P['price'])
        out[s + '90'] = (cur[s] + lam * prev[s] + k * pp) / (cur['mins'] / 90 + lam * prev['mins'] / 90 + k)
    pos_hit = ft.groupby('pos').apply(lambda g: g['hits'].sum() / max(g['n60'].sum(), 1), include_groups=False)
    pp = P['pos'].map(pos_hit).fillna(0)
    out['hit_rate'] = (cur['hits'] + lam * prev['hits'] + prm.k_defcon * pp) / (cur['n60'] + lam * prev['n60'] + prm.k_defcon)
    out.loc[P['pos'] == 'GKP', 'hit_rate'] = 0.0
    out.loc[P['pos'] != 'GKP', 'saves90'] = 0.0
    out['mins_season'] = cur['mins']
    return out


def minutes(state: State, prm: Params) -> pd.DataFrame:
    """p_start, p_sub, minutes when starting / subbing, and share of starts reaching 60."""
    P = state.players.set_index('id')
    h = state.history.sort_values(['player_id', 'kickoff_time'])
    h = h[h['player_id'].isin(P.index)]
    # league-wide shapes
    st = h[h['starts'] == 1]
    sb = h[(h['starts'] == 0) & (h['minutes'] > 0)]
    base_ms = st['minutes'].mean() if len(st) else 84.0
    base_q60 = (st['minutes'] >= 60).mean() if len(st) else 0.9
    base_mb = sb['minutes'].mean() if len(sb) else 20.0

    # price prior for starting share, fitted on player-level start share
    if state.prior_history is not None:
        lh = state.prior_history
        g = lh.groupby('code').agg(starts=('starts', 'sum'), n=('starts', 'size'))
        g = g.join(state.prior_players.drop_duplicates('code').set_index('code')[['pos', 'price']], how='inner')
        # last season's final 10 matches: recent role is what carries over
        lh10 = lh.sort_values('kickoff_time').groupby('code').tail(10).groupby('code')
        ls_share = lh10['starts'].mean()
        ls_sub = lh10.apply(lambda x: ((x['starts'] == 0) & (x['minutes'] > 0)).mean(), include_groups=False)
    else:
        g = h.groupby('player_id').agg(starts=('starts', 'sum'), n=('starts', 'size')).join(P[['pos', 'price']])
        ls_share = ls_sub = None
    g['share'] = g['starts'] / g['n']
    coef = _fit_linear(g, 'share', 'n', 3)
    prior_start = _apply_linear(coef, P['pos'], P['price'], lo=0.02, hi=0.95)
    prior_sub = pd.Series(0.12, index=P.index)
    if ls_share is not None:
        last_share = P['code'].map(ls_share)
        prior_start = (0.7 * last_share + 0.3 * prior_start).where(last_share.notna(), prior_start)
        prior_sub = P['code'].map(ls_sub).fillna(0.12)

    rec = h.groupby('player_id').tail(prm.n_recent).copy()
    rec['age'] = rec.groupby('player_id').cumcount(ascending=False)
    rec['w'] = prm.decay ** rec['age']
    rec['sub'] = ((rec['starts'] == 0) & (rec['minutes'] > 0)).astype(float)
    rec['ws'], rec['wb'] = rec['w'] * rec['starts'], rec['w'] * rec['sub']
    agg = rec.groupby('player_id')[['w', 'ws', 'wb']].sum().reindex(P.index).fillna(0)
    k = prm.k_min
    out = pd.DataFrame(index=P.index)
    out['p_start'] = (agg['ws'] + k * prior_start) / (agg['w'] + k)
    out['p_sub'] = (agg['wb'] + k * prior_sub) / (agg['w'] + k)
    tot = (out['p_start'] + out['p_sub']).clip(lower=1)
    out['p_start'] /= tot
    out['p_sub'] /= tot

    stp = st.groupby('player_id')['minutes'].agg(['sum', 'size']).reindex(P.index).fillna(0)
    q = st.assign(o=st['minutes'] >= 60).groupby('player_id')['o'].sum().reindex(P.index).fillna(0)
    sbp = sb.groupby('player_id')['minutes'].agg(['sum', 'size']).reindex(P.index).fillna(0)
    out['m_start'] = (stp['sum'] + 3 * base_ms) / (stp['size'] + 3)
    out['q60'] = (q + 3 * base_q60) / (stp['size'] + 3)
    out['m_sub'] = (sbp['sum'] + 3 * base_mb) / (sbp['size'] + 3)
    return out


_BACK = re.compile(r'(?:expected back|until|back on|return(?:s)? on)\s+(\d{1,2})\s+([A-Za-z]{3})', re.I)
_MONTHS = {m: i for i, m in enumerate(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], 1)}


def availability(state: State, gws: list[int]) -> pd.DataFrame:
    """P(available) per player per gameweek, from FPL's status / chance of playing / news."""
    P = state.players.set_index('id')
    season_year = pd.to_datetime(state.events['deadline_time'], utc=True).min().year
    dl = {gw: pd.to_datetime(state.events.loc[state.events['id'] == gw, 'deadline_time'].iloc[0], utc=True) for gw in gws}
    status = P['status'].fillna('a')
    c0 = (P['chance'] / 100).where(P['chance'].notna(), np.where(status == 'a', 1.0, np.where(status == 'd', 0.75, 0.0)))
    c0 = pd.Series(c0, index=P.index).astype(float)

    def back_date(news):
        m = _BACK.search(news or '')
        if not m:
            return pd.NaT
        mon = _MONTHS.get(m.group(2).lower()[:3])
        if not mon:
            return pd.NaT
        yr = season_year if mon >= 7 else season_year + 1
        try:
            return pd.Timestamp(year=yr, month=mon, day=int(m.group(1)), tz='UTC')
        except ValueError:
            return pd.NaT
    back = pd.to_datetime(P['news'].map(back_date), utc=True)
    out = pd.DataFrame(index=P.index)
    for j, gw in enumerate(gws):
        if j == 0:
            a = c0.copy()
        else:
            rec = np.where(status.isin(['i', 's']), 1 - 0.7 ** j, 1 - 0.5 ** j)
            a = c0 + (1 - c0) * rec
        known = back.notna()
        if j == 0:      # FPL's own chance of playing next round wins for the next gameweek
            known &= P['chance'].isna()
        a = a.where(~known, np.where(back <= dl[gw] + pd.Timedelta(days=1), np.maximum(a, 0.8), 0.0))
        a = a.where(~status.isin(['u', 'n']), 0.0)
        out[gw] = a
    return out


def team_fixtures(state: State, gws: list[int]) -> tuple[pd.DataFrame, float]:
    """One row per team per fixture in gws: xG for / against and multipliers vs the team's average."""
    pl = state.players[['id', 'pos']].copy()
    pl['element_type'] = pl['pos'].map({'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4})
    if len(state.history):
        ratings, model = team_model.build(state.history, pl, state.fixtures, state.teams, prev=state.team_prev)
    else:   # first gameweek of a season: last season's matches only
        tm = pd.DataFrame(columns=['fixture', 'team', 'opp', 'round', 'was_home', 'xg', 'xga', 'gf', 'ga'])
        tm = tm.astype({'was_home': bool, 'round': int, 'xg': float, 'xga': float, 'gf': float, 'ga': float})
        ratings = team_model.team_ratings(tm, state.teams, prev=state.team_prev)
        model = team_model.fixture_model(state.fixtures, ratings)
    avg = float(ratings['league_avg_xg'].iloc[0])
    r = ratings.set_index('team_id')
    m = model[model['event'].isin(gws)]
    home = pd.DataFrame({'team': m['team_h'], 'opp': m['team_a'], 'event': m['event'], 'fixture': m['id'],
                         'home': True, 'xgf': m['xg_home'], 'xga': m['xg_away']})
    away = pd.DataFrame({'team': m['team_a'], 'opp': m['team_h'], 'event': m['event'], 'fixture': m['id'],
                         'home': False, 'xgf': m['xg_away'], 'xga': m['xg_home']})
    tf = pd.concat([home, away], ignore_index=True)
    tf['mult_att'] = tf['xgf'] / (avg * tf['team'].map(r['att']))
    tf['mult_def'] = tf['xga'] / (avg * tf['team'].map(r['def']))
    return tf, avg


# ── the projection ──────────────────────────────────────────────────────────────────────────────
def project(state: State, prm: Params | None = None, gws: list[int] | None = None, detail: bool = False):
    """Expected points per player per gameweek.
    Returns a wide DataFrame (one row per player) with gw_<n> columns, next-GW component breakdown and
    the inputs; with detail=True also the long per-fixture table."""
    prm = prm or Params()
    gws = gws or [g for g in range(state.first_gw, state.first_gw + prm.horizon) if g <= int(state.events['id'].max())]
    P = state.players.set_index('id')
    R = rates(state, prm)
    M = minutes(state, prm)
    A = availability(state, gws)
    tf, avg = team_fixtures(state, gws)
    cs_mean = float(np.exp(-tf['xga'] * prm.cs_cal).mean()) if len(tf) else 0.3

    base = P[['team', 'pos']].join(R).join(M).reset_index()
    x = base.merge(tf, on='team', how='inner')
    x['avail'] = [A.at[i, e] for i, e in zip(x['id'], x['event'])]
    if state.overrides is not None and len(state.overrides):
        # team news: replace the recent-starts estimate; a dropped player mostly moves to the bench
        x = x.merge(overrides.per_gw(state.overrides, gws), on=['id', 'event'], how='left')
        new = x['p_start_adj'].fillna(x['p_start'])
        benched = (x['p_start'] - new).clip(lower=0)
        x['p_sub'] = np.minimum(x['p_sub'] + 0.6 * benched, 1 - new)
        x['p_start'] = new
    pos = x['pos']
    ps, pb, q60, ms, mb = x['p_start'], x['p_sub'], x['q60'], x['m_start'], x['m_sub']
    emin = ps * ms + pb * mb
    p60 = ps * q60
    cs_p = np.exp(-x['xga'] * prm.cs_cal)
    gk_def = pos.isin(['GKP', 'DEF'])

    x['xmins'] = emin
    x['app'] = (ps + pb) + p60
    x['goals'] = x['xg90'] * prm.goal_cal * x['mult_att'] * emin / 90 * pos.map(GOAL)
    # penalties: the team's rate × attack multiplier, taken by the first listed taker who is on the pitch.
    # on = share of the match a player is expected to be on (incl. availability); pen90 is per 90 on the pitch.
    x['on'] = (emin / 90 * x['avail']).clip(0, 1)
    x['pen_order'] = x['id'].map(P['penalties_order']) if 'penalties_order' in P else np.nan
    x['team_pens'] = x['team'].map(team_pen_rate(state, prm)).fillna(0.12)
    tk = x[x['pen_order'].notna()].sort_values(['fixture', 'team', 'pen_order'])
    ahead_off = tk.groupby(['fixture', 'team'])['on'].transform(lambda o: (1 - o).cumprod().shift(fill_value=1.0))
    x['pen90'] = (ahead_off.reindex(x.index).fillna(0) * x['team_pens'] * x['mult_att'] * prm.pen_conv)
    x['pens'] = x['pen90'] * emin / 90 * pos.map(GOAL)
    x['assists'] = x['xa90'] * prm.assist_cal * x['mult_att'] * emin / 90 * 3
    x['cs'] = p60 * cs_p * pos.map(CS_PTS)
    x['conceded'] = np.where(gk_def, -(ps * floor_mean(x['xga'] * ms / 90, 2) + pb * floor_mean(x['xga'] * mb / 90, 2)), 0.0)
    x['saves'] = np.where(pos == 'GKP', ps * floor_mean(x['saves90'] * x['mult_def'] * ms / 90, 3), 0.0)
    x['defcon'] = 2 * p60 * x['hit_rate']
    nudge = np.where(gk_def, 0.6 + 0.4 * cs_p / cs_mean, 0.6 + 0.4 * x['mult_att'])
    x['bonus'] = x['bonus90'] * emin / 90 * nudge
    x['cards'] = -x['yc90'] * emin / 90
    for c in COMPONENTS + ['xmins']:
        x[c] = x[c] * x['avail']
    x['xp'] = x[COMPONENTS].sum(1)

    wide = x.pivot_table(index='id', columns='event', values='xp', aggfunc='sum').reindex(columns=gws)
    wide = wide.reindex(P.index).fillna(0.0)
    wide.columns = [f'gw_{g}' for g in gws]
    nxt = x[x['event'] == gws[0]].groupby('id')[COMPONENTS + ['xmins']].sum().reindex(P.index).fillna(0)
    nxt.columns = [f'next_{c}' for c in nxt.columns]
    fx = (x[x['event'] == gws[0]].assign(lbl=lambda d: d['opp'].map(state.teams.set_index('id')['short_name'])
                                          + np.where(d['home'], ' (H)', ' (A)'))
          .groupby('id')['lbl'].agg(' + '.join).reindex(P.index).fillna('—'))

    out = P.drop(columns=[c for c in ('news',) if c in P.columns]).copy()
    out['next_opp'] = fx
    out['avail_next'] = A[gws[0]]
    out = out.join(M).join(R).join(wide).join(nxt)
    if 'adj_note' in x:
        nx = x[x['event'] == gws[0]].groupby('id')[['p_start', 'p_sub']].first()
        hit = nx.index[x[x['event'] == gws[0]].groupby('id')['adj_note'].first().notna()]
        out.loc[hit, ['p_start', 'p_sub']] = nx.loc[hit, ['p_start', 'p_sub']]
        out['adj_note'] = x.groupby('id')['adj_note'].first().reindex(out.index)
    out['pen90'] = x[x['event'] == gws[0]].groupby('id')['pen90'].max().reindex(P.index).fillna(0)
    out['pens_taken'] = strip_pens(state.history, season_label(state.events)).groupby('player_id')['pens'].sum().reindex(P.index).fillna(0)
    out['next_gw'] = out[f'gw_{gws[0]}']
    out['next_n'] = out[[f'gw_{g}' for g in gws]].sum(1)
    out['next_n_discounted'] = sum(out[f'gw_{g}'] * 0.9 ** j for j, g in enumerate(gws))
    out.attrs.update(gws=gws, first_gw=gws[0], league_avg_xg=avg)
    out = out.reset_index().sort_values('next_n', ascending=False)
    return (out, x) if detail else out


def project_snapshot(snap: Snapshot, last: Snapshot | None = None, prm: Params | None = None, **kw):
    return project(state_from_snapshot(snap, last), prm, **kw)
