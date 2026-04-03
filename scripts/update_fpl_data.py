from pathlib import Path
import requests
import pandas as pd
from utils import ensure_dir, write_json, today_str

BASE_DIR = Path(__file__).resolve().parents[1]
CURRENT_DIR = BASE_DIR / 'data' / 'current'
HISTORY_DIR = BASE_DIR / 'data' / 'history' / today_str()

BOOTSTRAP_URL = 'https://fantasy.premierleague.com/api/bootstrap-static/'
FIXTURES_URL = 'https://fantasy.premierleague.com/api/fixtures/'

HEADERS = {
    'User-Agent': 'pl-data-repo-blueprint/1.0'
}


def fetch_json(url: str):
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def save_csv(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(CURRENT_DIR / filename, index=False)
    df.to_csv(HISTORY_DIR / filename, index=False)


def main() -> None:
    ensure_dir(CURRENT_DIR)
    ensure_dir(HISTORY_DIR)

    bootstrap = fetch_json(BOOTSTRAP_URL)
    fixtures = fetch_json(FIXTURES_URL)

    write_json(CURRENT_DIR / 'bootstrap_static.json', bootstrap)
    write_json(HISTORY_DIR / 'bootstrap_static.json', bootstrap)

    players_df = pd.DataFrame(bootstrap['elements'])
    teams_df = pd.DataFrame(bootstrap['teams'])
    events_df = pd.DataFrame(bootstrap['events'])
    fixtures_df = pd.DataFrame(fixtures)

    save_csv(players_df, 'players.csv')
    save_csv(teams_df, 'teams.csv')
    save_csv(events_df, 'events.csv')
    save_csv(fixtures_df, 'fixtures.csv')

    print('FPL data refresh complete')


if __name__ == '__main__':
    main()
