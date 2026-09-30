# fplvalue

Python client for the data in `data/`, plus the projection model behind fplvalue.co/projections.html.

```bash
pip install -r requirements.txt
python -m fplvalue.publish                            # rebuild projections.csv, captains.csv, model_squad.csv
python -m fplvalue.backtest                           # walk-forward backtest vs FPL ep_next, form, points/match
python -m fplvalue.optimise                           # best £100m squad for the next 5 GWs
python -m fplvalue.optimise --team-id 1234567 --ft 1  # transfer suggestions for your team
```

```python
import fplvalue as fv
snap = fv.load()                         # data/current (or any data/history/<date>)
last = fv.last_season(snap)              # prior season's final snapshot
proj = fv.project_snapshot(snap, last)   # one row per player: gw_<n>, next_gw, next_n, next_<component>
```

| Module | What it does |
|---|---|
| `load.py` | Reads a snapshot folder into typed DataFrames; finds last season and the snapshot before a deadline |
| `team_model.py` | xG team attack/defence ratings and per-fixture expected goals (also used by the planner) |
| `project.py` | Minutes, per-90 rates, availability, fixture multipliers → expected points per component. `Params` holds every setting |
| `simulate.py` | Monte Carlo of the next gameweek for the captain ranking |
| `optimise.py` | Integer-programme squad / transfer optimiser (scipy HiGHS) |
| `backtest.py` | Rewinds a season to before each deadline and scores the projections |
| `publish.py` | Writes the site's CSV/JSON outputs; called from `scripts/update_fpl_data.py` every week |

`notebooks/projections.ipynb` walks through all of it, including how to change a setting and re-run the backtest.
The weekly history snapshots keep each week's `projections.csv`, so live accuracy can be checked as the season goes.
