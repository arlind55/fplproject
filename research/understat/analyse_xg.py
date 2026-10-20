#!/usr/bin/env python3
"""First look at whether xG predicts Premier League results better than goals.

Reads the CSVs written by scrape_understat.py and runs three checks:

1. Split season: does a team's form over matches 1-19 predict its points per
   game over matches 20-38? Compares points, goal difference, xG difference
   and non-penalty xG difference as the predictor.
2. Next match: using only matches already played, build each team's rolling
   average over its last N league games (crossing season boundaries), then fit
   a small home/draw/away model on older seasons and score it on newer ones.
3. Repeatability: how well does each measure in the first half of a season
   correlate with the same measure in the second half? A predictor that is
   mostly noise will not repeat.

Needs pandas and numpy. Usage:

    python analyse_xg.py                 # reads ./data, writes ./data/analysis.md
    python analyse_xg.py --window 6 --test-from 2021
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load(data_dir):
    matches = pd.read_csv(data_dir / "matches.csv", parse_dates=["datetime"])
    team = pd.read_csv(data_dir / "team_matches.csv", parse_dates=["date"])
    matches = matches[matches.played == 1].sort_values("datetime").reset_index(drop=True)
    team = team.sort_values(["team_id", "date"]).reset_index(drop=True)
    team["gd"] = team.goals_for - team.goals_against
    team["xgd"] = team.xg - team.xga
    team["season_start"] = team.season.str[:4].astype(int)
    return matches, team


def split_season(team):
    team = team.copy()
    team["half"] = np.where(team.match_no <= 19, "first", "second")
    agg = (team.groupby(["season", "team", "half"])[["pts", "gd", "xgd", "npxgd", "xpts"]]
           .mean().unstack("half"))
    agg = agg.dropna()
    target = agg[("pts", "second")]
    rows = []
    for col, label in [("pts", "Points per game"), ("xpts", "Understat xPts per game"),
                       ("gd", "Goal difference per game"), ("xgd", "xG difference per game"),
                       ("npxgd", "Non-penalty xG difference per game")]:
        x = agg[(col, "first")]
        r = np.corrcoef(x, target)[0, 1]
        repeat = np.corrcoef(x, agg[(col, "second")])[0, 1]
        rows.append({"first-half measure": label, "r with 2nd-half pts/game": r,
                     "R²": r * r, "repeatability (1st vs 2nd half of itself)": repeat})
    return pd.DataFrame(rows), len(agg)


def rolling_form(team, window):
    """Per team, the average of the previous `window` games, excluding the current one."""
    cols = ["pts", "gd", "xgd", "npxgd", "xg", "xga"]
    rolled = (team.groupby("team_id")[cols]
              .transform(lambda s: s.shift(1).rolling(window, min_periods=window).mean()))
    out = team[["team_id", "date"]].copy()
    out[[f"r_{c}" for c in cols]] = rolled
    return out


def attach_form(matches, form):
    # Understat stamps a team's history entry with the fixture's kick-off time.
    m = matches.copy()
    for side, id_col in (("h", "home_id"), ("a", "away_id")):
        f = form.rename(columns={c: f"{side}_{c}" for c in form.columns if c.startswith("r_")})
        m = m.merge(f.rename(columns={"date": "datetime"}), left_on=[id_col, "datetime"],
                    right_on=["team_id", "datetime"], how="left").drop(columns="team_id")
    if len(m) != len(matches):
        raise ValueError("team history and fixtures did not line up one-to-one")
    m["outcome"] = np.select([m.home_goals > m.away_goals, m.home_goals == m.away_goals], [0, 1], 2)
    m["season_start"] = m.season.str[:4].astype(int)
    return m


def fit_softmax(X, y, l2=1e-3, iters=3000, lr=0.5):
    """Multinomial logistic regression by plain gradient descent (X already standardised)."""
    X = np.hstack([np.ones((len(X), 1)), X])
    W = np.zeros((X.shape[1], 3))
    Y = np.eye(3)[y]
    for _ in range(iters):
        Z = X @ W
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        grad = X.T @ (P - Y) / len(X) + l2 * np.vstack([np.zeros((1, 3)), W[1:]])
        W -= lr * grad
    return W


def predict(W, X):
    X = np.hstack([np.ones((len(X), 1)), X])
    Z = X @ W
    Z -= Z.max(axis=1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(axis=1, keepdims=True)


def score(P, y):
    Y = np.eye(3)[y]
    logloss = -np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)))
    # Ranked probability score respects that a draw sits between home and away win.
    rps = np.mean(np.sum((np.cumsum(P, axis=1) - np.cumsum(Y, axis=1))[:, :2] ** 2, axis=1) / 2)
    acc = np.mean(P.argmax(axis=1) == y)
    return logloss, rps, acc


def next_match(matches, team, window, test_from):
    m = attach_form(matches, rolling_form(team, window))
    feature_sets = {
        "Points form": ["pts"],
        "Goal difference form": ["gd"],
        "xG difference form": ["xgd"],
        "Non-penalty xG difference form": ["npxgd"],
        "xG for + xG against form": ["xg", "xga"],
        "Goal difference + xG difference": ["gd", "xgd"],
    }
    needed = [f"{s}_r_{c}" for s in "ha" for c in ["pts", "gd", "xgd", "npxgd", "xg", "xga"]]
    m = m.dropna(subset=needed)
    train, test = m[m.season_start < test_from], m[m.season_start >= test_from]

    rows = []
    base = np.bincount(train.outcome, minlength=3) / len(train)
    rows.append(("Home/draw/away base rates only", *score(np.tile(base, (len(test), 1)), test.outcome.values)))
    for label, cols in feature_sets.items():
        feats = [f"h_r_{c}" for c in cols] + [f"a_r_{c}" for c in cols]
        mu, sd = train[feats].mean(), train[feats].std()
        W = fit_softmax(((train[feats] - mu) / sd).values, train.outcome.values)
        P = predict(W, ((test[feats] - mu) / sd).values)
        rows.append((label, *score(P, test.outcome.values)))
    df = pd.DataFrame(rows, columns=["model (rolling form of both teams)", "log loss", "RPS", "accuracy"])
    return df, len(train), len(test)


def per_match(matches):
    gd = matches.home_goals - matches.away_goals
    xgd = matches.home_xg - matches.away_xg
    return {
        "matches": len(matches),
        "r(match xG diff, match goal diff)": np.corrcoef(xgd, gd)[0, 1],
        "share of matches won by the side with more xG": np.mean(np.sign(gd[gd != 0]) == np.sign(xgd[gd != 0])),
    }


def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--window", type=int, default=8, help="games in the rolling form window (default 8)")
    parser.add_argument("--test-from", type=int, default=2021, help="first season held out for testing (default 2021)")
    args = parser.parse_args()

    matches, team = load(args.data)
    seasons = sorted(matches.season.unique())
    split, n_teams = split_season(team)
    nm, n_train, n_test = next_match(matches, team, args.window, args.test_from)
    pm = per_match(matches)

    report = f"""# Does xG predict Premier League results?

Data: Understat, {seasons[0]} to {seasons[-1]}, {pm['matches']} played matches.

## 1. First half of the season predicting the second half

{n_teams} team-seasons. Each row uses a team's average over matches 1-19 to predict its
points per game over matches 20-38. Higher r means a better predictor.

{md_table(split)}

## 2. Predicting the next match from recent form

Rolling average of each team's last {args.window} league games (before kick-off), fed into a
home/draw/away model. Trained on seasons before {args.test_from}/{str(args.test_from + 1)[-2:]}
({n_train} matches), scored on {args.test_from}/{str(args.test_from + 1)[-2:]} onwards ({n_test} matches).
Lower log loss and RPS are better.

{md_table(nm, "{:.4f}")}

## 3. Within a match

- Correlation between a match's xG difference and its goal difference: {pm['r(match xG diff, match goal diff)']:.3f}
- In matches with a winner, the side with more xG won {pm['share of matches won by the side with more xG']:.1%} of the time

Note: Understat's own `forecast_*` columns in matches.csv are computed from the shots in that
match, after the fact, so they describe the match rather than predict it. They are left out of
the models above.
"""
    out = args.data / "analysis.md"
    out.write_text(report)
    print(report)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
