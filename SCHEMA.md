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
