# Schema Notes

## players.csv
Flattened player master table from `bootstrap-static`, enriched with team names and cost converted from tenths.

## player_prices.csv
Lean pricing table for app cards and price change tracking.

## player_form.csv
App-ready ranking table with form, points per game, ownership, points, and key counting stats.

## next_fixtures.csv
Upcoming fixtures joined to team names and FPL difficulty fields.

## team_fixture_difficulty.csv
One row per team summarising average difficulty across the next five fixtures using official FPL difficulty values.

## player_history.csv
One row per player per past gameweek from `element-summary/{player_id}` history.

## player_future_fixtures.csv
One row per player per upcoming fixture from `element-summary/{player_id}` fixtures.

## finishing.csv (data/)
One row per current FPL player: Premier League career non-penalty goals, npxG and minutes before this season from Understat (xG rescaled per season to league goals), the same this season from FPL, per-90 rates for both, `npg90_change` (this season minus career goals per 90) and `finishing`, a damped career goals-per-xG ratio (1.0 = average). Players new to the Premier League have no career rows. Written by `scripts/finishing.py`; not used by the projection model.
