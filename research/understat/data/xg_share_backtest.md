# Team xG x player share vs player per-90 rate

Understat appearances, outfield players. Each appearance's npxG (or xA) is predicted from
data before kick-off, times the minutes actually played. Every model uses the same team
fixture expectation. Reference is `rate S+L`, which is how v1 works.
"vs ref %" is the change in mean squared error; negative is better. CI is clustered by player.

## Target: npxG

Tuned on 2016/17, 2017/18, 2018/19 (half-life days, k): rate S+L (180, 8.0); rate multi (730, 8.0); share S+L (180, 8.0); share multi (730, 8.0)

### All test appearances (71,904 appearances)

| model             |   MSE x1000 |   vs ref % |   95% CI ± |   bias (pred/actual) |
|:------------------|------------:|-----------:|-----------:|---------------------:|
| rate S+L          |      37.190 |      0.000 |      0.000 |                0.997 |
| rate multi        |      37.112 |     -0.210 |      0.203 |                1.000 |
| share S+L         |      37.249 |      0.158 |      0.193 |                1.020 |
| share multi       |      37.140 |     -0.135 |      0.254 |                1.021 |
| blend 50/50 multi |      37.041 |     -0.401 |      0.210 |                1.010 |

### Players at a different club from last season (5,774 appearances)

| model             |   MSE x1000 |   vs ref % |   95% CI ± |   bias (pred/actual) |
|:------------------|------------:|-----------:|-----------:|---------------------:|
| rate S+L          |      42.084 |      0.000 |      0.000 |                0.991 |
| rate multi        |      42.033 |     -0.122 |      1.073 |                1.006 |
| share S+L         |      42.613 |      1.255 |      1.029 |                1.021 |
| share multi       |      42.495 |      0.975 |      1.083 |                1.030 |
| blend 50/50 multi |      42.038 |     -0.111 |      0.881 |                1.018 |

### First 6 team matches of a season (11,362 appearances)

| model             |   MSE x1000 |   vs ref % |   95% CI ± |   bias (pred/actual) |
|:------------------|------------:|-----------:|-----------:|---------------------:|
| rate S+L          |      37.629 |      0.000 |      0.000 |                0.987 |
| rate multi        |      37.638 |      0.025 |      0.739 |                0.999 |
| share S+L         |      37.524 |     -0.279 |      0.982 |                1.026 |
| share multi       |      37.357 |     -0.722 |      1.130 |                1.024 |
| blend 50/50 multi |      37.391 |     -0.633 |      0.880 |                1.012 |

### Team npxG/match moved 20%+ from last season (20,951 appearances)

| model             |   MSE x1000 |   vs ref % |   95% CI ± |   bias (pred/actual) |
|:------------------|------------:|-----------:|-----------:|---------------------:|
| rate S+L          |      38.921 |      0.000 |      0.000 |                0.957 |
| rate multi        |      38.986 |      0.168 |      0.368 |                0.958 |
| share S+L         |      39.027 |      0.274 |      0.367 |                0.973 |
| share multi       |      38.887 |     -0.087 |      0.455 |                0.974 |
| blend 50/50 multi |      38.845 |     -0.196 |      0.358 |                0.966 |

### Forwards (13,413 appearances)

| model             |   MSE x1000 |   vs ref % |   95% CI ± |   bias (pred/actual) |
|:------------------|------------:|-----------:|-----------:|---------------------:|
| rate S+L          |      99.741 |      0.000 |      0.000 |                1.002 |
| rate multi        |      99.485 |     -0.256 |      0.340 |                1.001 |
| share S+L         |      99.789 |      0.048 |      0.353 |                1.018 |
| share multi       |      99.359 |     -0.382 |      0.424 |                1.013 |
| blend 50/50 multi |      99.110 |     -0.632 |      0.347 |                1.007 |

### Midfielders (33,981 appearances)

| model             |   MSE x1000 |   vs ref % |   95% CI ± |   bias (pred/actual) |
|:------------------|------------:|-----------:|-----------:|---------------------:|
| rate S+L          |      28.940 |      0.000 |      0.000 |                1.008 |
| rate multi        |      28.913 |     -0.096 |      0.271 |                1.015 |
| share S+L         |      29.039 |      0.341 |      0.205 |                1.037 |
| share multi       |      29.007 |      0.232 |      0.346 |                1.039 |
| blend 50/50 multi |      28.910 |     -0.104 |      0.277 |                1.027 |

### Defenders (24,510 appearances)

| model             |   MSE x1000 |   vs ref % |   95% CI ± |   bias (pred/actual) |
|:------------------|------------:|-----------:|-----------:|---------------------:|
| rate S+L          |      14.398 |      0.000 |      0.000 |                0.950 |
| rate multi        |      14.347 |     -0.352 |      0.349 |                0.955 |
| share S+L         |      14.407 |      0.061 |      0.135 |                0.981 |
| share multi       |      14.367 |     -0.216 |      0.425 |                0.990 |
| blend 50/50 multi |      14.348 |     -0.347 |      0.378 |                0.973 |

### Salah: share of Liverpool npxG while on the pitch

| season   |   mins |   npxG |   team_on |   share |   npxG/90 |
|:---------|-------:|-------:|----------:|--------:|----------:|
| 2014/15  |     23 |  0.502 |     1.482 |   0.339 |     1.966 |
| 2017/18  |   2954 | 23.614 |    68.107 |   0.347 |     0.719 |
| 2018/19  |   3274 | 19.509 |    70.667 |   0.276 |     0.536 |
| 2019/20  |   2904 | 18.380 |    64.074 |   0.287 |     0.570 |
| 2020/21  |   3085 | 15.684 |    60.010 |   0.261 |     0.458 |
| 2021/22  |   2757 | 19.797 |    76.330 |   0.259 |     0.646 |
| 2022/23  |   3307 | 20.295 |    77.113 |   0.263 |     0.552 |
| 2023/24  |   2556 | 16.613 |    69.811 |   0.238 |     0.585 |
| 2024/25  |   3392 | 20.856 |    87.347 |   0.239 |     0.553 |
| 2025/26  |   2163 |  8.077 |    44.108 |   0.183 |     0.336 |

### Year-to-year repeatability (players with 1350+ minutes both seasons)

| pairs        |    n |   r(per-90 rate) |   r(share) |
|:-------------|-----:|-----------------:|-----------:|
| same club    | 1389 |            0.885 |      0.892 |
| changed club |  132 |            0.818 |      0.832 |

## Target: xA

Tuned on 2016/17, 2017/18, 2018/19 (half-life days, k): rate S+L (180, 4.0); rate multi (1460, 8.0); share S+L (180, 4.0); share multi (1460, 8.0)

### All test appearances (71,904 appearances)

| model             |   MSE x1000 |   vs ref % |   95% CI ± |   bias (pred/actual) |
|:------------------|------------:|-----------:|-----------:|---------------------:|
| rate S+L          |      26.707 |      0.000 |      0.000 |                1.021 |
| rate multi        |      26.471 |     -0.885 |      0.267 |                1.005 |
| share S+L         |      26.678 |     -0.110 |      0.126 |                1.029 |
| share multi       |      26.512 |     -0.733 |      0.281 |                1.032 |
| blend 50/50 multi |      26.447 |     -0.976 |      0.257 |                1.018 |

### Players at a different club from last season (5,774 appearances)

| model             |   MSE x1000 |   vs ref % |   95% CI ± |   bias (pred/actual) |
|:------------------|------------:|-----------:|-----------:|---------------------:|
| rate S+L          |      27.908 |      0.000 |      0.000 |                1.024 |
| rate multi        |      27.587 |     -1.150 |      0.724 |                1.011 |
| share S+L         |      27.887 |     -0.074 |      0.888 |                1.040 |
| share multi       |      27.838 |     -0.251 |      1.350 |                1.041 |
| blend 50/50 multi |      27.590 |     -1.139 |      0.867 |                1.026 |

### First 6 team matches of a season (11,362 appearances)

| model             |   MSE x1000 |   vs ref % |   95% CI ± |   bias (pred/actual) |
|:------------------|------------:|-----------:|-----------:|---------------------:|
| rate S+L          |      26.928 |      0.000 |      0.000 |                1.001 |
| rate multi        |      26.577 |     -1.304 |      0.559 |                0.988 |
| share S+L         |      26.932 |      0.014 |      0.349 |                1.021 |
| share multi       |      26.620 |     -1.146 |      0.651 |                1.019 |
| blend 50/50 multi |      26.549 |     -1.409 |      0.569 |                1.003 |

### Team npxG/match moved 20%+ from last season (20,951 appearances)

| model             |   MSE x1000 |   vs ref % |   95% CI ± |   bias (pred/actual) |
|:------------------|------------:|-----------:|-----------:|---------------------:|
| rate S+L          |      27.359 |      0.000 |      0.000 |                0.976 |
| rate multi        |      27.190 |     -0.620 |      0.380 |                0.956 |
| share S+L         |      27.362 |      0.009 |      0.269 |                0.977 |
| share multi       |      27.218 |     -0.517 |      0.446 |                0.979 |
| blend 50/50 multi |      27.160 |     -0.728 |      0.388 |                0.968 |

### Forwards (13,413 appearances)

| model             |   MSE x1000 |   vs ref % |   95% CI ± |   bias (pred/actual) |
|:------------------|------------:|-----------:|-----------:|---------------------:|
| rate S+L          |      34.204 |      0.000 |      0.000 |                1.032 |
| rate multi        |      33.805 |     -1.169 |      0.605 |                1.010 |
| share S+L         |      34.097 |     -0.313 |      0.217 |                1.026 |
| share multi       |      33.849 |     -1.040 |      0.538 |                1.027 |
| blend 50/50 multi |      33.772 |     -1.265 |      0.550 |                1.018 |

### Midfielders (33,981 appearances)

| model             |   MSE x1000 |   vs ref % |   95% CI ± |   bias (pred/actual) |
|:------------------|------------:|-----------:|-----------:|---------------------:|
| rate S+L          |      30.971 |      0.000 |      0.000 |                1.024 |
| rate multi        |      30.686 |     -0.921 |      0.372 |                1.003 |
| share S+L         |      30.945 |     -0.083 |      0.184 |                1.035 |
| share multi       |      30.744 |     -0.733 |      0.406 |                1.030 |
| blend 50/50 multi |      30.654 |     -1.022 |      0.359 |                1.017 |

### Defenders (24,510 appearances)

| model             |   MSE x1000 |   vs ref % |   95% CI ± |   bias (pred/actual) |
|:------------------|------------:|-----------:|-----------:|---------------------:|
| rate S+L          |      16.694 |      0.000 |      0.000 |                1.001 |
| rate multi        |      16.615 |     -0.475 |      0.422 |                1.003 |
| share S+L         |      16.703 |      0.051 |      0.261 |                1.017 |
| share multi       |      16.629 |     -0.387 |      0.523 |                1.039 |
| blend 50/50 multi |      16.605 |     -0.531 |      0.457 |                1.021 |
