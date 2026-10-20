# Does xG predict Premier League results?

Data: Understat, 2014/15 to 2026/27, 4630 played matches.

## 1. First half of the season predicting the second half

240 team-seasons. Each row uses a team's average over matches 1-19 to predict its
points per game over matches 20-38. Higher r means a better predictor.

| first-half measure | r with 2nd-half pts/game | R² | repeatability (1st vs 2nd half of itself) |
|---|---|---|---|
| Points per game | 0.696 | 0.485 | 0.696 |
| Understat xPts per game | 0.756 | 0.571 | 0.808 |
| Goal difference per game | 0.715 | 0.511 | 0.745 |
| xG difference per game | 0.759 | 0.576 | 0.815 |
| Non-penalty xG difference per game | 0.764 | 0.584 | 0.811 |

## 2. Predicting the next match from recent form

Rolling average of each team's last 8 league games (before kick-off), fed into a
home/draw/away model. Trained on seasons before 2021/22
(2494 matches), scored on 2021/22 onwards (1931 matches).
Lower log loss and RPS are better.

| model (rolling form of both teams) | log loss | RPS | accuracy |
|---|---|---|---|
| Home/draw/away base rates only | 1.0683 | 0.2317 | 0.4417 |
| Points form | 1.0204 | 0.2145 | 0.5028 |
| Goal difference form | 1.0122 | 0.2115 | 0.5142 |
| xG difference form | 0.9940 | 0.2054 | 0.5318 |
| Non-penalty xG difference form | 0.9961 | 0.2061 | 0.5287 |
| xG for + xG against form | 0.9964 | 0.2059 | 0.5334 |
| Goal difference + xG difference | 0.9956 | 0.2059 | 0.5334 |

## 3. Within a match

- Correlation between a match's xG difference and its goal difference: 0.683
- In matches with a winner, the side with more xG won 79.3% of the time

Note: Understat's own `forecast_*` columns in matches.csv are computed from the shots in that
match, after the fact, so they describe the match rather than predict it. They are left out of
the models above.
