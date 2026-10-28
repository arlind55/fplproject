# Goal calibration and player finishing

October 2026. Three questions: is the model's goal term biased now that xG providers run hot, does a player's career finishing record improve goal forecasts, and do xG underperformers "bounce back"?

## Understat xG has drifted high

League non-penalty goals per Understat npxG, all shots:

| Seasons | Goals per xG |
|---|---|
| 2014/15 to 2017/18 | 0.96 to 1.04 |
| 2018/19 to 2023/24 | 0.90 to 0.93 |
| 2024/25 | 0.86 |
| 2025/26 | 0.83 |
| 2026/27 (to Oct 19) | 0.84 |

Shots have moved closer to goal (xG per shot 0.094 to 0.119) without the matching rise in conversion (9.0% to 9.9%), more shots are blocked (28.8% to 30.9%), and headers and set pieces are the most overrated (0.71 goals per xG since 2024/25). A location-based model can't see defenders, so the likely cause is deeper, more crowded defending that the model was never refitted for. Any Understat goals-minus-xG work should rescale each season's xG to its goals first, as `xg_bounce_back.py` and `scripts/finishing.py` do.

The live model uses FPL's (Opta) xG, which runs less hot: 0.90 non-penalty goals per npxG this season.

## The model's goal calibration

Walk-forward (rewind to before each gameweek, project it, compare non-penalty goals), with the old flat `goal_cal = 0.93`:

| Position | 2025/26 actual / projected | 2026/27 actual / projected |
|---|---|---|
| DEF | 0.94 | 0.84 |
| MID | 1.11 | 1.13 |
| FWD | 1.13 | 1.07 |

The flat factor was over-correcting midfielders and forwards and under-correcting defenders (whose xG is mostly set-piece headers). Per-position factors fitted on 2025/26 only (DEF 0.87, MID 1.03, FWD 1.05) cut 2026/27 goal deviance by 0.23%, against 0.06% for a flat 1.0. Pooled over both seasons they are DEF 0.85, MID 1.03, FWD 1.04, which is what `Params.goal_cal` now holds.

Points backtest (regular starters), old flat 0.93 vs new per-position:

| Season | MAE | RMSE | Bias | Spearman | Top-20 actual |
|---|---|---|---|---|---|
| 2025/26 old | 2.151 | 2.940 | +0.020 | 0.365 | 4.71 |
| 2025/26 new | 2.160 | 2.940 | +0.044 | 0.367 | 4.68 |
| 2026/27 old | 2.218 | 2.998 | -0.079 | 0.357 | 4.88 |
| 2026/27 new | 2.226 | 2.997 | -0.054 | 0.356 | 4.90 |

A small change at the points level: bias moves toward zero this season, RMSE is flat, MAE gets slightly worse (MAE rewards under-projecting a skewed target) and rankings are unchanged within noise. The case for it is the goal forecasts themselves, above.

## Career finishing as a goal multiplier

Multiplying each player's projected goals by their damped career finishing before the season, (npG + K) / (npxG + K) from season-rescaled Understat data, never helped:

| K (npxG of average finishing) | 2025/26 deviance change | 2026/27 deviance change |
|---|---|---|
| 25 | -0.11% (-0.29 to +0.08) | +0.12% (-0.29 to +0.48) |
| 150 | -0.02% (-0.07 to +0.03) | +0.06% (-0.05 to +0.17) |
| 600 | -0.01% (-0.02 to +0.01) | +0.02% (-0.01 to +0.06) |

90% intervals from resampling players. Finishing skill is real but small: the true spread between attackers is about ±7.5% of xG against ±19% of noise for a typical career sample (split-half correlation 0.34 for players with 15+ npxG). So `data/current/finishing.csv` is shown in the screener for reading, not fed into projections.

## Do underperformers bounce back?

`xg_bounce_back.py`, 784 Understat attackers since 2014/15. Cold runs (8 appearances, 2+ non-penalty goals under season-rescaled npxG) happen as often as pure luck predicts (216 vs 214 simulated). In the next 10 appearances they convert at 1.02x xG, and 6% fully "pay back" the deficit, the same as luck. They don't overperform to catch up, they return to their xG rate. Their goal rate still rises from 0.14 to 0.40 per 90. The only thing that separates bad luck from a poor finisher is the career record: cold runs by poor finishers (damped ratio below 0.93) kept converting at 0.81x.
