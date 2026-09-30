"""Loading the pipeline's CSV snapshots into typed DataFrames.

A *snapshot* is any folder the pipeline writes: data/current or data/history/<YYYY-MM-DD>.
Everything the projection model needs is in one:
  players.csv, teams.csv, fixtures.csv, events.csv, player_history.csv, meta.json (optional)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
POS = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}

_HIST_NUM = ('minutes', 'total_points', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded', 'saves', 'bonus',
             'bps', 'yellow_cards', 'red_cards', 'starts', 'defensive_contribution', 'expected_goals',
             'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded', 'value', 'round',
             'fixture', 'player_id', 'element', 'team_h_score', 'team_a_score', 'opponent_team', 'own_goals',
             'penalties_saved', 'penalties_missed')


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(('true', '1'))


@dataclass
class Snapshot:
    path: Path
    players: pd.DataFrame
    teams: pd.DataFrame
    fixtures: pd.DataFrame
    events: pd.DataFrame
    history: pd.DataFrame
    meta: dict = field(default_factory=dict)

    @property
    def season_start(self) -> pd.Timestamp:
        return pd.to_datetime(self.events['deadline_time'], utc=True).min()

    @property
    def last_finished_gw(self) -> int:
        fin = self.events[_bool(self.events['finished'])]
        return int(fin['id'].max()) if len(fin) else 0

    def deadline(self, gw: int) -> pd.Timestamp:
        row = self.events[self.events['id'] == gw]
        return pd.to_datetime(row['deadline_time'].iloc[0], utc=True)


def load(path: str | Path | None = None) -> Snapshot:
    """Load a snapshot folder (default data/current)."""
    path = Path(path) if path else DATA / 'current'
    players = pd.read_csv(path / 'players.csv')
    teams = pd.read_csv(path / 'teams.csv')
    fixtures = pd.read_csv(path / 'fixtures.csv')
    events = pd.read_csv(path / 'events.csv')
    hp = path / 'player_history.csv'
    try:
        history = pd.read_csv(hp) if hp.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        history = pd.DataFrame()
    meta = json.loads((path / 'meta.json').read_text()) if (path / 'meta.json').exists() else {}

    players['pos'] = players['element_type'].map(POS)
    for c in ('now_cost', 'selected_by_percent', 'chance_of_playing_next_round', 'ep_next', 'form', 'points_per_game'):
        if c in players.columns:
            players[c] = pd.to_numeric(players[c], errors='coerce')
    if players['now_cost'].max() > 30:           # very old snapshots stored tenths of a million
        players['now_cost'] = players['now_cost'] / 10
    players['news'] = players.get('news', pd.Series('', index=players.index)).fillna('')

    for c in ('event', 'team_h', 'team_a', 'team_h_score', 'team_a_score', 'team_h_difficulty', 'team_a_difficulty'):
        if c in fixtures.columns:
            fixtures[c] = pd.to_numeric(fixtures[c], errors='coerce')
    fixtures['finished'] = _bool(fixtures['finished'])

    if len(history):
        for c in _HIST_NUM:
            if c in history.columns:
                history[c] = pd.to_numeric(history[c], errors='coerce')
        history['was_home'] = _bool(history['was_home'])
        if 'player_id' not in history.columns:
            history['player_id'] = history['element']
        history = history.merge(fixtures[['id', 'team_h', 'team_a']].rename(columns={'id': 'fixture'}), on='fixture', how='left')
        history['team'] = np.where(history['was_home'], history['team_h'], history['team_a'])
        history = history.drop(columns=['team_h', 'team_a'])
    return Snapshot(path, players, teams, fixtures, events, history, meta)


def snapshot_dirs(root: Path | None = None) -> list[Path]:
    root = root or DATA / 'history'
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name[:4].isdigit())


def last_season(current: Snapshot, root: Path | None = None) -> Snapshot | None:
    """Most recent snapshot from before this season started that has a complete 38-round history."""
    start = current.season_start.date()
    for p in reversed(snapshot_dirs(root)):
        if pd.Timestamp(p.name).date() >= start or not (p / 'player_history.csv').exists():
            continue
        try:
            rounds = pd.read_csv(p / 'player_history.csv', usecols=['round'])['round']
        except (pd.errors.EmptyDataError, ValueError):
            continue
        if rounds.max() >= 38:
            return load(p)
    return None


def snapshot_before(when: pd.Timestamp, root: Path | None = None, after: pd.Timestamp | None = None) -> Path | None:
    """Latest snapshot folder dated strictly before `when` (and optionally after `after`)."""
    best = None
    for p in snapshot_dirs(root):
        d = pd.Timestamp(p.name, tz='UTC')
        if d < when.normalize() and (after is None or d >= after.normalize()):
            best = p
    return best
