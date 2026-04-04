from pathlib import Path
import time
import requests
import pandas as pd
from utils import ensure_dir, write_json, today_str

BASE_DIR = Path(__file__).resolve().parents[1]
CURRENT_DIR = BASE_DIR / 'data' / 'current'
HISTORY_DIR = BASE_DIR / 'data' / 'history' / today_str()
PLAYER_SUMMARIES_DIR = CURRENT_DIR / 'player_summaries'
HISTORY_PLAYER_SUMMARIES_DIR = HISTORY_DIR / 'player_summaries'

BOOTSTRAP_URL = 'https://fantasy.premierleague.com/api/bootstrap-static/'
FIXTURES_URL = 'https://fantasy.premierleague.com/api/fixtures/'
ELEMENT_SUMMARY_URL = 'https://fantasy.premierleague.com/api/element-summary/{player_id}/'
HEADERS = {'User-Agent': 'fplproject-data-pipeline/2.0'}


def fetch_json(url: str):
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def save_csv(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(CURRENT_DIR / filename, index=False)
    df.to_csv(HISTORY_DIR / filename, index=False)


def normalise_players(players_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    team_cols = teams_df[['id', 'name', 'short_name']].rename(columns={'id': 'team', 'name': 'team_name', 'short_name': 'team_short_name'})
    out = players_df.merge(team_cols, on='team', how='left').copy()
    if 'now_cost' in out.columns:
        out['now_cost'] = out['now_cost'] / 10
    if 'selected_by_percent' in out.columns:
        out['selected_by_percent'] = pd.to_numeric(out['selected_by_percent'], errors='coerce')
    if 'form' in out.columns:
        out['form'] = pd.to_numeric(out['form'], errors='coerce')
    return out


def build_player_prices(players_df: pd.DataFrame) -> pd.DataFrame:
    cols = ['id', 'web_name', 'team_name', 'team_short_name', 'element_type', 'now_cost', 'cost_change_event', 'cost_change_start']
    existing = [c for c in cols if c in players_df.columns]
    return players_df[existing].sort_values(['team_name', 'web_name']).reset_index(drop=True)


def build_player_form(players_df: pd.DataFrame) -> pd.DataFrame:
    cols = ['id', 'web_name', 'team_name', 'team_short_name', 'element_type', 'now_cost', 'form', 'points_per_game', 'total_points', 'selected_by_percent', 'minutes', 'goals_scored', 'assists', 'clean_sheets', 'bonus', 'status']
    existing = [c for c in cols if c in players_df.columns]
    out = players_df[existing].copy()
    sort_cols = [c for c in ['form', 'total_points', 'points_per_game'] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=False)
    return out.reset_index(drop=True)


def build_next_fixtures(fixtures_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    team_map = teams_df[['id', 'name', 'short_name']].copy()
    home = team_map.rename(columns={'id': 'team_h', 'name': 'team_h_name', 'short_name': 'team_h_short'})
    away = team_map.rename(columns={'id': 'team_a', 'name': 'team_a_name', 'short_name': 'team_a_short'})
    out = fixtures_df.merge(home, on='team_h', how='left').merge(away, on='team_a', how='left')
    if 'finished' in out.columns:
        out = out[out['finished'] != True].copy()
    cols = ['id', 'event', 'kickoff_time', 'team_h', 'team_h_name', 'team_h_short', 'team_h_difficulty', 'team_a', 'team_a_name', 'team_a_short', 'team_a_difficulty']
    existing = [c for c in cols if c in out.columns]
    sort_cols = [c for c in ['event', 'kickoff_time', 'id'] if c in out.columns]
    return out[existing].sort_values(sort_cols).reset_index(drop=True)


def build_team_fixture_difficulty(next_fixtures_df: pd.DataFrame, teams_df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    rows = []
    teams = teams_df[['id', 'name', 'short_name']].to_dict('records')

    for team in teams:
        home = next_fixtures_df[next_fixtures_df['team_h'] == team['id']].copy()
        home = home[['event', 'kickoff_time', 'team_h_difficulty', 'team_a_name']]
        home = home.rename(columns={
            'team_h_difficulty': 'difficulty',
            'team_a_name': 'opponent'
        })
        home['is_home'] = True

        away = next_fixtures_df[next_fixtures_df['team_a'] == team['id']].copy()
        away = away[['event', 'kickoff_time', 'team_a_difficulty', 'team_h_name']]
        away = away.rename(columns={
            'team_a_difficulty': 'difficulty',
            'team_h_name': 'opponent'
        })
        away['is_home'] = False

        combined = pd.concat([home, away], ignore_index=True)
        combined = combined.sort_values(['event', 'kickoff_time']).head(horizon)

        if len(combined):
            opponent_list = (
                combined['opponent'].fillna('Unknown').astype(str) +
                combined['is_home'].map({True: ' (H)', False: ' (A)'})
            ).tolist()
            avg_difficulty = round(combined['difficulty'].astype(float).mean(), 3)
            min_difficulty = combined['difficulty'].min()
            max_difficulty = combined['difficulty'].max()
        else:
            opponent_list = []
            avg_difficulty = None
            min_difficulty = None
            max_difficulty = None

        rows.append({
            'team_id': team['id'],
            'team_name': team['name'],
            'team_short_name': team['short_name'],
            'fixtures_in_horizon': len(combined),
            'avg_difficulty_next_5': avg_difficulty,
            'min_difficulty_next_5': min_difficulty,
            'max_difficulty_next_5': max_difficulty,
            'opponents_next_5': ' | '.join(opponent_list)
        })

    return pd.DataFrame(rows).sort_values(['avg_difficulty_next_5', 'team_name'], na_position='last').reset_index(drop=True)


def fetch_player_summaries(player_ids):
    history_rows = []
    fixture_rows = []
    for idx, player_id in enumerate(player_ids, start=1):
        payload = fetch_json(ELEMENT_SUMMARY_URL.format(player_id=player_id))
        write_json(PLAYER_SUMMARIES_DIR / f'{player_id}.json', payload)
        write_json(HISTORY_PLAYER_SUMMARIES_DIR / f'{player_id}.json', payload)
        for row in payload.get('history', []):
            row = dict(row)
            row['player_id'] = player_id
            history_rows.append(row)
        for row in payload.get('fixtures', []):
            row = dict(row)
            row['player_id'] = player_id
            fixture_rows.append(row)
        if idx % 50 == 0:
            time.sleep(1)
    return pd.DataFrame(history_rows), pd.DataFrame(fixture_rows)


def main() -> None:
    ensure_dir(CURRENT_DIR)
    ensure_dir(HISTORY_DIR)
    ensure_dir(PLAYER_SUMMARIES_DIR)
    ensure_dir(HISTORY_PLAYER_SUMMARIES_DIR)

    bootstrap = fetch_json(BOOTSTRAP_URL)
    fixtures = fetch_json(FIXTURES_URL)

    write_json(CURRENT_DIR / 'bootstrap_static.json', bootstrap)
    write_json(HISTORY_DIR / 'bootstrap_static.json', bootstrap)

    raw_players_df = pd.DataFrame(bootstrap['elements'])
    teams_df = pd.DataFrame(bootstrap['teams'])
    events_df = pd.DataFrame(bootstrap['events'])
    fixtures_df = pd.DataFrame(fixtures)
    players_df = normalise_players(raw_players_df, teams_df)

    save_csv(players_df, 'players.csv')
    save_csv(teams_df, 'teams.csv')
    save_csv(events_df, 'events.csv')
    save_csv(fixtures_df, 'fixtures.csv')

    save_csv(build_player_prices(players_df), 'player_prices.csv')
    save_csv(build_player_form(players_df), 'player_form.csv')

    next_fixtures_df = build_next_fixtures(fixtures_df, teams_df)
    save_csv(next_fixtures_df, 'next_fixtures.csv')
    save_csv(build_team_fixture_difficulty(next_fixtures_df, teams_df), 'team_fixture_difficulty.csv')

    player_history_df, player_future_fixtures_df = fetch_player_summaries(players_df['id'].tolist())
    save_csv(player_history_df, 'player_history.csv')
    save_csv(player_future_fixtures_df, 'player_future_fixtures.csv')

    print('FPL v2 data refresh complete')


if __name__ == '__main__':
    main()
