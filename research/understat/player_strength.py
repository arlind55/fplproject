#!/usr/bin/env python3
"""Does a team strength built from its players beat team xG form?

Team xG form (analyse_xg.py) only knows what the club did in its last few games. It
cannot see that the striker is injured, that a new signing starts, or that the
manager rotated six players. This script rates each player from his own past
Understat numbers and rates a team, for one match, as the sum of its starting XI.

Player ratings, all computed from matches before kick-off:

- Every past appearance counts, decayed with a half-life in days, across seasons and clubs.
- Per 90 rate = (decayed stat + k * prior) / (decayed minutes / 90 + k), so a player with
  little history is pulled towards a prior. The prior is what players with under 900
  previous minutes actually produced in the training seasons, by position group, which
  keeps newly promoted and newly signed players from being rated as average PL regulars.
- Attack: non-penalty xG per 90 and xA per 90, summed over the XI.

Defence stays team-based (rolling xG against), since Understat has no useful player-level
defensive numbers.

The same home/draw/away model as analyse_xg.py is fit on older seasons and scored on
newer ones, on exactly the same matches for every model. The half-life and k are picked
on a validation split inside the training seasons only.

Note: this uses the actual starting XI, which is public about an hour before kick-off.
For FPL deadlines you would feed it predicted line-ups instead.

Needs data/appearances.csv (scrape_understat.py --shots). Usage:

    python player_strength.py                  # writes data/player_strength.md
    python player_strength.py --test-from 2021
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyse_xg import fit_softmax, load, md_table, predict, rolling_form, score

POS_GROUP = {
    "GK": "GK",
    "DC": "D", "DL": "D", "DR": "D", "DMR": "D", "DML": "D",
    "DMC": "M", "MC": "M", "ML": "M", "MR": "M", "AMC": "M", "AML": "M", "AMR": "M",
    "FW": "F",
}
STATS = ["npxG", "xA"]
NEW_PLAYER_MINUTES = 900


def load_appearances(data_dir):
    app = pd.read_csv(data_dir / "appearances.csv", parse_dates=["date"])
    app = app.sort_values(["date", "match_id", "team_id", "started"],
                          ascending=[True, True, True, False]).reset_index(drop=True)
    # Substitutes are listed as "Sub"; give them the position they usually start in.
    app["pos"] = app.position.map(POS_GROUP)
    usual = app.dropna(subset=["pos"]).groupby("player_id").pos.agg(lambda s: s.mode().iat[0])
    app["pos"] = app.pos.fillna(app.player_id.map(usual)).fillna("M")
    app["season_start"] = app.season.str[:4].astype(int)
    return app


def decayed_history(app, half_life):
    """For each appearance, the player's decayed minutes and stats from earlier matches only."""
    lam = np.log(2) / half_life
    days = (app.date - app.date.min()).dt.total_seconds().values / 86400
    mins = app.minutes.values.astype(float)
    stats = app[STATS].values.astype(float)
    prior_min = np.zeros(len(app))
    prior_stats = np.zeros((len(app), len(STATS)))
    raw_min = np.zeros(len(app))
    for _, idx in app.groupby("player_id").indices.items():
        m = 0.0
        st = np.zeros(len(STATS))
        last = None
        total = 0.0
        for i in idx:
            if last is not None:
                f = np.exp(-lam * (days[i] - last))
                m, st = m * f, st * f
            prior_min[i], prior_stats[i], raw_min[i] = m, st, total
            m += mins[i]
            st = st + stats[i]
            total += mins[i]
            last = days[i]
    out = pd.DataFrame(prior_stats, columns=[f"d_{c}" for c in STATS], index=app.index)
    out["d_minutes"] = prior_min
    out["prior_minutes"] = raw_min
    return out


def new_player_prior(app, hist, train_mask):
    """Per-90 output by position group of players with little previous PL history."""
    sel = train_mask & (hist.prior_minutes < NEW_PLAYER_MINUTES)
    g = app[sel].groupby("pos")
    return pd.DataFrame({c: g[c].sum() / (g.minutes.sum() / 90) for c in STATS})


def xi_strength(app, hist, prior, k):
    """Sum of starters' shrunk per-90 ratings, one row per team per match."""
    starters = app.started == 1
    p = prior.reindex(app.pos).values
    rate = (hist[[f"d_{c}" for c in STATS]].values + k * p) / (hist.d_minutes.values[:, None] / 90 + k)
    df = pd.DataFrame(rate, columns=[f"xi_{c}" for c in STATS], index=app.index)
    df[["match_id", "team_id"]] = app[["match_id", "team_id"]]
    return df[starters].groupby(["match_id", "team_id"]).sum().reset_index()


def lineup_changes(app):
    """Starters who did not start the team's previous league match."""
    xi = (app[app.started == 1].groupby(["team_id", "date", "match_id"]).player_id
          .apply(frozenset).reset_index().sort_values(["team_id", "date"]))
    prev = xi.groupby("team_id").player_id.shift(1)
    xi["changes"] = [len(a - b) if isinstance(b, frozenset) else np.nan for a, b in zip(xi.player_id, prev)]
    return xi[["match_id", "team_id", "changes"]]


def build_matches(matches, team, app, hist, prior, k, window):
    m = matches.copy()
    m["season_start"] = m.season.str[:4].astype(int)
    m["outcome"] = np.select([m.home_goals > m.away_goals, m.home_goals == m.away_goals], [0, 1], 2)
    form = rolling_form(team, window)
    xi = xi_strength(app, hist, prior, k)
    ch = lineup_changes(app)
    for side, id_col in (("h", "home_id"), ("a", "away_id")):
        f = form.rename(columns={c: f"{side}_{c}" for c in form.columns if c.startswith("r_")})
        m = m.merge(f.rename(columns={"date": "datetime"}), left_on=[id_col, "datetime"],
                    right_on=["team_id", "datetime"], how="left").drop(columns="team_id")
        x = xi.rename(columns={c: f"{side}_{c}" for c in xi.columns if c.startswith("xi_")})
        m = m.merge(x, left_on=["match_id", id_col], right_on=["match_id", "team_id"],
                    how="left").drop(columns="team_id")
        c = ch.rename(columns={"changes": f"{side}_changes"})
        m = m.merge(c, left_on=["match_id", id_col], right_on=["match_id", "team_id"],
                    how="left").drop(columns="team_id")
    if len(m) != len(matches):
        raise ValueError("fixtures did not line up one-to-one")
    return m


MODELS = {
    "Team xG form (xG for + against)": ["r_xg", "r_xga"],
    "Player XI attack + team xG against": ["xi_npxG", "xi_xA", "r_xga"],
    "Both: player XI + team xG form": ["xi_npxG", "xi_xA", "r_xg", "r_xga"],
}


def fit_and_score(train, test, cols):
    feats = [f"h_{c}" for c in cols] + [f"a_{c}" for c in cols]
    mu, sd = train[feats].mean(), train[feats].std()
    W = fit_softmax(((train[feats] - mu) / sd).values, train.outcome.values)
    return predict(W, ((test[feats] - mu) / sd).values)


def per_match_logloss(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--window", type=int, default=8, help="team form window, as in analyse_xg.py")
    parser.add_argument("--test-from", type=int, default=2021)
    parser.add_argument("--burn-in", type=int, default=1, help="seasons left out while player histories build up")
    args = parser.parse_args()

    matches, team = load(args.data)
    app = load_appearances(args.data)
    first = app.season_start.min()
    train_from = first + args.burn_in
    val_from = args.test_from - 2
    needed = [f"{s}_{c}" for s in "ha" for c in ["r_xg", "r_xga"]]

    # Pick half-life and k on the last two training seasons, fit on the ones before.
    tuning = []
    for half_life in (180, 365, 730):
        hist = decayed_history(app, half_life)
        prior = new_player_prior(app, hist, app.season_start < val_from)
        for k in (3, 6, 12):
            m = build_matches(matches, team, app, hist, prior, k, args.window).dropna(subset=needed)
            tr = m[(m.season_start >= train_from) & (m.season_start < val_from)]
            va = m[(m.season_start >= val_from) & (m.season_start < args.test_from)]
            cols = MODELS["Player XI attack + team xG against"]
            ll = per_match_logloss(fit_and_score(tr, va, cols), va.outcome.values).mean()
            tuning.append((half_life, k, ll))
    half_life, k, _ = min(tuning, key=lambda t: t[2])

    hist = decayed_history(app, half_life)
    prior = new_player_prior(app, hist, app.season_start < args.test_from)
    m = build_matches(matches, team, app, hist, prior, k, args.window).dropna(subset=needed)
    train = m[(m.season_start >= train_from) & (m.season_start < args.test_from)]
    test = m[m.season_start >= args.test_from].reset_index(drop=True)
    y = test.outcome.values

    preds = {label: fit_and_score(train, test, cols) for label, cols in MODELS.items()}
    base = np.bincount(train.outcome, minlength=3) / len(train)
    preds = {"Home/draw/away base rates only": np.tile(base, (len(test), 1)), **preds}

    subsets = {
        "All test matches": np.ones(len(test), bool),
        "Either side made 4+ changes to its XI": (test[["h_changes", "a_changes"]].max(axis=1) >= 4).values,
        "Both sides made 0-1 changes": (test[["h_changes", "a_changes"]].max(axis=1) <= 1).values,
        "Opening 6 matches of a season": (pd.to_datetime(test.datetime).dt.month.isin([8, 9])).values,
    }

    ref = per_match_logloss(preds["Team xG form (xG for + against)"], y)
    sections = []
    for name, mask in subsets.items():
        rows = []
        for label, P in preds.items():
            ll, rps, acc = score(P[mask], y[mask])
            d = per_match_logloss(P, y)[mask] - ref[mask]
            se = d.std(ddof=1) / np.sqrt(mask.sum())
            vs = "(reference)" if label.startswith("Team xG") else f"{d.mean():+.4f} ± {1.96 * se:.4f}"
            rows.append((label, ll, rps, acc, vs))
        df = pd.DataFrame(rows, columns=["model", "log loss", "RPS", "accuracy",
                                         "log loss vs team xG form (95% CI)"])
        sections.append(f"### {name} ({mask.sum()} matches)\n\n{md_table(df, '{:.4f}')}")

    tune = pd.DataFrame(tuning, columns=["half-life (days)", "k (90s of prior)", "validation log loss"])

    report = f"""# Player-based team strength vs team xG form

Each team is rated for each match as the sum of its starting XI's non-penalty xG and xA
per 90, from every earlier Premier League appearance (decayed, half-life {half_life} days,
shrunk towards a new-player prior with weight k = {k} matches). Defence uses the team's
rolling xG against over its last {args.window} games. Trained on {train_from}/{str(train_from + 1)[-2:]} to
{args.test_from - 1}/{str(args.test_from)[-2:]} ({len(train)} matches; {first}/{str(first + 1)[-2:]} is left out while player histories build up),
tested on {args.test_from}/{str(args.test_from + 1)[-2:]} onwards. Every model is scored on the same matches.

The last column is the average per-match change in log loss against team xG form.
Negative means better; if the interval crosses zero, the difference could be noise.

{(chr(10) * 2).join(sections)}

## Tuning (validation seasons {val_from}/{str(val_from + 1)[-2:]} to {args.test_from - 1}/{str(args.test_from)[-2:]}, player model)

{md_table(tune.astype({'half-life (days)': str, 'k (90s of prior)': str}), '{:.4f}')}

## New-player prior (per 90, players with under {NEW_PLAYER_MINUTES} previous PL minutes)

{md_table(prior.reset_index().rename(columns={'pos': 'position group'}), '{:.3f}')}

Uses the actual starting XI, which is known about an hour before kick-off. For FPL,
feed predicted line-ups instead.
"""
    out = args.data / "player_strength.md"
    out.write_text(report)
    print(report)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
