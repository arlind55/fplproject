# Elo v the fixture model

Walk-forward like fixture_backtest.py. "fixture model" is the site's decay-15 opponent-adjusted xG fit.
w = 0 is the fixture model alone, w = 1 is the Elo fit alone. Negative % = better than the fixture model.

## Elo-to-goals fits (tuning seasons, 1 week ahead)

| source                              |   intercept |   per 100 Elo |   home |   per FDR step faced |   per FDR step opp faces |
|:------------------------------------|------------:|--------------:|-------:|---------------------:|-------------------------:|
| home-made Elo (K=10, home edge 100) |       0.125 |         0.197 |  0.255 |              nan     |                  nan     |
| FPL FDR                             |       0.013 |       nan     |  0.090 |               -0.232 |                    0.300 |

Best Elo weight on tuning seasons (goals, 1 week ahead): home-made Elo (K=10, home edge 100): **w = 0.4**

## Test seasons (2020/21, 2021/22, 2022/23, 2023/24, 2024/25, 2025/26), 1 week ahead

| source                              |    w |    n |   xG MSE vs model % |   goals dev vs model % |   CS log loss vs model % |
|:------------------------------------|-----:|-----:|--------------------:|-----------------------:|-------------------------:|
| fixture model                       | 0.00 | 4560 |                0.00 |                   0.00 |                     0.00 |
| home-made Elo (K=10, home edge 100) | 0.40 | 4560 |                0.96 |                  -0.36 |                    -0.07 |
| home-made Elo (K=10, home edge 100) | 1.00 | 4560 |                8.81 |                   1.92 |                     1.14 |
| FPL FDR                             | 1.00 | 4560 |                7.53 |                  -0.41 |                    -0.20 |

## Test seasons (2020/21, 2021/22, 2022/23, 2023/24, 2024/25, 2025/26), 5 weeks ahead

| source                              |    w |    n |   xG MSE vs model % |   goals dev vs model % |   CS log loss vs model % |
|:------------------------------------|-----:|-----:|--------------------:|-----------------------:|-------------------------:|
| fixture model                       | 0.00 | 3492 |                0.00 |                   0.00 |                     0.00 |
| home-made Elo (K=10, home edge 100) | 0.40 | 3492 |                0.75 |                  -0.49 |                     0.08 |
| home-made Elo (K=10, home edge 100) | 1.00 | 3492 |                8.39 |                   1.72 |                     1.54 |
| FPL FDR                             | 1.00 | 3492 |                5.82 |                  -0.96 |                     0.00 |

## Test seasons (2020/21, 2021/22, 2022/23, 2023/24, 2024/25, 2025/26), 10 weeks ahead

| source                              |    w |    n |   xG MSE vs model % |   goals dev vs model % |   CS log loss vs model % |
|:------------------------------------|-----:|-----:|--------------------:|-----------------------:|-------------------------:|
| fixture model                       | 0.00 | 3178 |                0.00 |                   0.00 |                     0.00 |
| home-made Elo (K=10, home edge 100) | 0.40 | 3178 |                0.40 |                  -0.55 |                    -0.02 |
| home-made Elo (K=10, home edge 100) | 1.00 | 3178 |                7.68 |                   1.50 |                     1.31 |
| FPL FDR                             | 1.00 | 3178 |                3.72 |                  -2.02 |                    -0.76 |

## Test seasons, 1 week ahead, first 8 matches of a season

| source                              |    w |   n |   xG MSE vs model % |   goals dev vs model % |   CS log loss vs model % |
|:------------------------------------|-----:|----:|--------------------:|-----------------------:|-------------------------:|
| fixture model                       | 0.00 | 960 |                0.00 |                   0.00 |                     0.00 |
| home-made Elo (K=10, home edge 100) | 0.40 | 960 |               -0.31 |                  -0.38 |                     0.05 |
| home-made Elo (K=10, home edge 100) | 0.50 | 960 |                0.25 |                  -0.22 |                     0.19 |
| home-made Elo (K=10, home edge 100) | 1.00 | 960 |                6.26 |                   2.11 |                     1.63 |
| FPL FDR                             | 1.00 | 960 |                8.26 |                   0.80 |                     0.49 |

## Test seasons, 1 week ahead, promoted teams, first 8 matches

| source                              |    w |   n |   xG MSE vs model % |   goals dev vs model % |   CS log loss vs model % |
|:------------------------------------|-----:|----:|--------------------:|-----------------------:|-------------------------:|
| fixture model                       | 0.00 | 144 |                0.00 |                   0.00 |                     0.00 |
| home-made Elo (K=10, home edge 100) | 0.40 | 144 |               -0.43 |                  -1.42 |                     0.33 |
| home-made Elo (K=10, home edge 100) | 0.50 | 144 |                0.12 |                  -1.59 |                     0.53 |
| home-made Elo (K=10, home edge 100) | 1.00 | 144 |                6.66 |                  -1.34 |                     2.27 |
| FPL FDR                             | 1.00 | 144 |                8.73 |                  -2.20 |                    -1.70 |

## home-made Elo (K=10, home edge 100), w = 0.4, by test season (1 week ahead)

| season   |   n |   goals dev vs model % |   CS log loss vs model % |
|:---------|----:|-----------------------:|-------------------------:|
| 2020/21  | 760 |                  -1.10 |                    -0.37 |
| 2021/22  | 760 |                  -0.54 |                    -0.37 |
| 2022/23  | 760 |                   0.42 |                    -0.17 |
| 2023/24  | 760 |                   0.30 |                     0.83 |
| 2024/25  | 760 |                  -0.00 |                     0.29 |
| 2025/26  | 760 |                  -1.16 |                    -0.46 |

## Full weight grid, tuning seasons, 1 week ahead

| source                              |    w |    n |   xG MSE vs model % |   goals dev vs model % |   CS log loss vs model % |
|:------------------------------------|-----:|-----:|--------------------:|-----------------------:|-------------------------:|
| fixture model                       | 0.00 | 3040 |                0.00 |                   0.00 |                     0.00 |
| home-made Elo (K=10, home edge 100) | 0.00 | 3040 |                0.00 |                   0.00 |                     0.00 |
| home-made Elo (K=10, home edge 100) | 0.10 | 3040 |               -0.70 |                  -0.33 |                    -0.15 |
| home-made Elo (K=10, home edge 100) | 0.20 | 3040 |               -1.17 |                  -0.58 |                    -0.26 |
| home-made Elo (K=10, home edge 100) | 0.30 | 3040 |               -1.40 |                  -0.73 |                    -0.34 |
| home-made Elo (K=10, home edge 100) | 0.40 | 3040 |               -1.38 |                  -0.80 |                    -0.37 |
| home-made Elo (K=10, home edge 100) | 0.50 | 3040 |               -1.09 |                  -0.78 |                    -0.37 |
| home-made Elo (K=10, home edge 100) | 0.60 | 3040 |               -0.52 |                  -0.66 |                    -0.32 |
| home-made Elo (K=10, home edge 100) | 0.70 | 3040 |                0.35 |                  -0.45 |                    -0.24 |
| home-made Elo (K=10, home edge 100) | 0.80 | 3040 |                1.54 |                  -0.14 |                    -0.12 |
| home-made Elo (K=10, home edge 100) | 0.90 | 3040 |                3.08 |                   0.26 |                     0.05 |
| home-made Elo (K=10, home edge 100) | 1.00 | 3040 |                5.00 |                   0.76 |                     0.25 |
| FPL FDR                             | 1.00 | 1520 |               13.11 |                  -2.23 |                    -0.59 |
