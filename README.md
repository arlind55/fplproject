# Premier League Data Repo Blueprint

This starter repository is designed to power a web app with regularly refreshed Premier League / FPL data.

## What this repo does

- Pulls data from the public Fantasy Premier League API
- Stores cleaned files for your app to read from GitHub raw URLs
- Keeps both latest files and dated historical snapshots
- Runs on a weekly schedule using GitHub Actions

## Suggested repo structure

```text
.
├── .github/
│   └── workflows/
│       └── update-data.yml
├── data/
│   ├── current/
│   │   ├── bootstrap_static.json
│   │   ├── players.csv
│   │   ├── teams.csv
│   │   ├── events.csv
│   │   └── fixtures.csv
│   ├── history/
│   │   └── 2026-04-03/
│   │       ├── bootstrap_static.json
│   │       ├── players.csv
│   │       ├── teams.csv
│   │       ├── events.csv
│   │       └── fixtures.csv
│   └── reference/
├── scripts/
│   ├── update_fpl_data.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## FPL endpoints used

- `https://fantasy.premierleague.com/api/bootstrap-static/`
- `https://fantasy.premierleague.com/api/fixtures/`
- Optional later: `https://fantasy.premierleague.com/api/element-summary/{player_id}/`

## Setup

### 1. Create the repo

Create a new public GitHub repo, for example:
- `pl-data`
- `premier-league-data`
- `fpl-data-pipeline`

### 2. Add files

Copy this blueprint into the repo.

### 3. Local install

```bash
pip install -r requirements.txt
python scripts/update_fpl_data.py
```

### 4. Enable Actions

Push to GitHub, then enable GitHub Actions for the repository.

## Output files

### Latest files for app

Your web app should read from `data/current/` because those files are overwritten with the latest refresh.

Example raw URL pattern:

```text
https://raw.githubusercontent.com/<your-user>/<your-repo>/main/data/current/players.csv
```

### Historical snapshots

Each run also writes a dated snapshot into `data/history/YYYY-MM-DD/`.

## Recommended frontend usage

Point the web app at:
- `players.csv` for player master data
- `teams.csv` for team mapping
- `events.csv` for gameweeks and deadlines
- `fixtures.csv` for match-level schedule data

## Next upgrades

- Add `element-summary` pulls for per-player history
- Add data validation tests
- Add derived files such as form tables, fixture difficulty, and weekly rankings
- Add a small schema doc for each CSV
