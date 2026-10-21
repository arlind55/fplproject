# Player-based team strength vs team xG form

Each team is rated for each match as the sum of its starting XI's non-penalty xG and xA
per 90, from every earlier Premier League appearance (decayed, half-life 730 days,
shrunk towards a new-player prior with weight k = 3 matches). Defence uses the team's
rolling xG against over its last 8 games. Trained on 2015/16 to
2020/21 (2194 matches; 2014/15 is left out while player histories build up),
tested on 2021/22 onwards. Every model is scored on the same matches.

The last column is the average per-match change in log loss against team xG form.
Negative means better; if the interval crosses zero, the difference could be noise.

### All test matches (1931 matches)

| model | log loss | RPS | accuracy | log loss vs team xG form (95% CI) |
|---|---|---|---|---|
| Home/draw/away base rates only | 1.0682 | 0.2317 | 0.4417 | +0.0724 ± 0.0165 |
| Team xG form (xG for + against) | 0.9959 | 0.2057 | 0.5313 | (reference) |
| Player XI attack + team xG against | 0.9848 | 0.2011 | 0.5360 | -0.0111 ± 0.0109 |
| Both: player XI + team xG form | 0.9824 | 0.2007 | 0.5381 | -0.0135 ± 0.0101 |

### Either side made 4+ changes to its XI (697 matches)

| model | log loss | RPS | accuracy | log loss vs team xG form (95% CI) |
|---|---|---|---|---|
| Home/draw/away base rates only | 1.0639 | 0.2304 | 0.4534 | +0.1003 ± 0.0280 |
| Team xG form (xG for + against) | 0.9636 | 0.1949 | 0.5681 | (reference) |
| Player XI attack + team xG against | 0.9646 | 0.1944 | 0.5466 | +0.0010 ± 0.0185 |
| Both: player XI + team xG form | 0.9621 | 0.1939 | 0.5538 | -0.0015 ± 0.0172 |

### Both sides made 0-1 changes (245 matches)

| model | log loss | RPS | accuracy | log loss vs team xG form (95% CI) |
|---|---|---|---|---|
| Home/draw/away base rates only | 1.0795 | 0.2335 | 0.4163 | +0.0572 ± 0.0417 |
| Team xG form (xG for + against) | 1.0223 | 0.2130 | 0.5020 | (reference) |
| Player XI attack + team xG against | 1.0059 | 0.2059 | 0.5184 | -0.0164 ± 0.0284 |
| Both: player XI + team xG form | 1.0037 | 0.2055 | 0.5224 | -0.0186 ± 0.0266 |

### Opening 6 matches of a season (334 matches)

| model | log loss | RPS | accuracy | log loss vs team xG form (95% CI) |
|---|---|---|---|---|
| Home/draw/away base rates only | 1.0811 | 0.2297 | 0.4251 | +0.0756 ± 0.0393 |
| Team xG form (xG for + against) | 1.0055 | 0.2041 | 0.5060 | (reference) |
| Player XI attack + team xG against | 0.9816 | 0.1944 | 0.5240 | -0.0239 ± 0.0258 |
| Both: player XI + team xG form | 0.9781 | 0.1938 | 0.5210 | -0.0274 ± 0.0237 |

## Tuning (validation seasons 2019/20 to 2020/21, player model)

| half-life (days) | k (90s of prior) | validation log loss |
|---|---|---|
| 180 | 3 | 1.0244 |
| 180 | 6 | 1.0235 |
| 180 | 12 | 1.0228 |
| 365 | 3 | 1.0226 |
| 365 | 6 | 1.0226 |
| 365 | 12 | 1.0229 |
| 730 | 3 | 1.0216 |
| 730 | 6 | 1.0222 |
| 730 | 12 | 1.0235 |

## New-player prior (per 90, players with under 900 previous PL minutes)

| position group | npxG | xA |
|---|---|---|
| D | 0.037 | 0.047 |
| F | 0.347 | 0.107 |
| GK | 0.000 | 0.002 |
| M | 0.125 | 0.118 |

Uses the actual starting XI, which is known about an hour before kick-off. For FPL,
feed predicted line-ups instead.
