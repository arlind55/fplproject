# Projection model over four and a bit seasons

Walk-forward replay of `fplvalue` on vaastav's FPL archive (see season_backtest.py). Regular starters
(started at least 2 of their last 4 team matches), each gameweek from GW5 (GW2 this season).
`fpl_xp` is FPL's own projection (ep_next) for that gameweek; it has seen injury news, the model here has not.

FPL projections are missing for some weeks (most of 2025/26), so model and FPL are compared on the
96 gameweeks where both exist.

## All seasons together, regular starters

|        |   player-weeks |   MAE |   RMSE |   bias |   Spearman |   top-20 pts |   captain pts |
|:-------|---------------:|------:|-------:|-------:|-----------:|-------------:|--------------:|
| model  |      22954.000 | 2.093 |  2.938 |  0.064 |      0.409 |        5.059 |         7.167 |
| fpl_xp |      22954.000 | 2.457 |  3.459 |  0.248 |      0.281 |        4.022 |         5.594 |
| ppm    |      22954.000 | 2.167 |  3.088 | -0.011 |      0.327 |        4.623 |         7.990 |
| form   |      22954.000 | 2.327 |  3.254 |  0.284 |      0.316 |        4.381 |         6.406 |

Model MSE minus FPL xP MSE: **-3.329** (90% interval -3.782 to -2.918; negative = model better).

## By season, regular starters

| season   |   weeks |   model RMSE |   FPL xP RMSE | MSE gap (90% CI)     |   model Spearman |   FPL xP Spearman |   model top-20 |   FPL xP top-20 |   model captain |   FPL xP captain |   model bias |
|:---------|--------:|-------------:|--------------:|:---------------------|-----------------:|------------------:|---------------:|----------------:|----------------:|-----------------:|-------------:|
| 2022-23  |      21 |         3.02 |          3.70 | -4.58 (-5.79, -3.42) |             0.39 |              0.27 |           5.08 |            4.14 |            6.24 |             2.86 |        -0.03 |
| 2023-24  |      33 |         2.95 |          3.50 | -3.53 (-4.30, -2.80) |             0.44 |              0.29 |           5.12 |            3.91 |            6.24 |             5.45 |         0.05 |
| 2024-25  |      31 |         2.86 |          3.30 | -2.73 (-3.39, -2.20) |             0.41 |              0.28 |           5.05 |            3.89 |            8.58 |             7.52 |         0.14 |
| 2025-26  |       7 |         2.91 |          3.31 | -2.50 (-3.00, -1.98) |             0.34 |              0.28 |           4.58 |            4.19 |            7.14 |             5.86 |         0.05 |
| 2026-27  |       4 |         3.09 |          3.37 | -1.81 (-2.07, -1.52) |             0.33 |              0.28 |           5.31 |            4.92 |            8.75 |             8.75 |         0.06 |

## By position, regular starters, all seasons

| pos   |   player-weeks |   model RMSE |   FPL xP RMSE |   model bias |   FPL xP bias |   model Spearman |   FPL xP Spearman |
|:------|---------------:|-------------:|--------------:|-------------:|--------------:|-----------------:|------------------:|
| DEF   |           8534 |        2.826 |         3.343 |        0.223 |         0.209 |            0.386 |             0.220 |
| FWD   |           2287 |        3.423 |         4.045 |       -0.087 |         0.334 |            0.373 |             0.267 |
| GKP   |           1968 |        2.748 |         3.270 |        0.176 |         0.214 |            0.349 |             0.176 |
| MID   |          10165 |        2.948 |         3.447 |       -0.057 |         0.268 |            0.411 |             0.298 |

## By point in the season, regular starters

| phase   |   player-weeks |   model RMSE |   FPL xP RMSE |   model Spearman |   FPL xP Spearman |
|:--------|---------------:|-------------:|--------------:|-----------------:|------------------:|
| GW2-10  |           5129 |        2.893 |         3.304 |            0.379 |             0.283 |
| GW11-20 |           5515 |        2.925 |         3.315 |            0.375 |             0.276 |
| GW21-30 |           7156 |        2.865 |         3.511 |            0.441 |             0.281 |
| GW31-38 |           5154 |        3.093 |         3.679 |            0.430 |             0.286 |

## Calibration, all players with a fixture, all seasons

| model       |        n |   projected |   actual |
|:------------|---------:|------------:|---------:|
| (-1.0, 0.5] | 55510.00 |        0.12 |     0.14 |
| (0.5, 1.5]  | 18267.00 |        0.93 |     1.31 |
| (1.5, 2.5]  | 13276.00 |        2.00 |     2.06 |
| (2.5, 3.5]  | 11218.00 |        2.96 |     2.98 |
| (3.5, 4.5]  |  4600.00 |        3.91 |     3.83 |
| (4.5, 5.5]  |  1197.00 |        4.89 |     4.60 |
| (5.5, 7.0]  |   547.00 |        6.11 |     6.29 |
| (7.0, 20.0] |   294.00 |        8.15 |     7.61 |
