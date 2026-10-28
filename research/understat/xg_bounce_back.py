#!/usr/bin/env python3
"""Do attackers who underperform their xG then 'pay it back' by overperforming?

Uses the Understat cache (shots.csv + appearances.csv, EPL 2014/15 onwards).
Every number is non-penalty: goals minus penalty goals vs npxG.

    python xg_bounce_back.py <data_dir> <out_dir>

Writes results.json (all tables) and candidates.csv to out_dir.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
ATTACK = {"FW", "FWL", "FWR", "AMC", "AML", "AMR", "ML", "MR"}
GROUP = {"FW": "Striker", "FWL": "Striker", "FWR": "Striker", "AMC": "No. 10",
         "AML": "Winger", "AMR": "Winger", "ML": "Winger", "MR": "Winger"}
PAST, FUT = 8, 10          # the user's example: cold for 8, then look at the next 10
COLD = -2.0                # npG - npxG over the past window
MIN_XG = 2.0               # and they were getting chances (npxG over the window)
N_SIM = 300


def load(data_dir):
    app = pd.read_csv(data_dir / "appearances.csv", parse_dates=["date"])
    app = app[app.minutes > 0].drop(columns=["shots", "xG", "npxG"])
    sh = pd.read_csv(data_dir / "shots.csv")
    sh = sh[(sh.situation != "Penalty") & (sh.result != "OwnGoal")].copy()
    sh["goal"] = (sh.result == "Goal").astype(int)
    # Understat's xG runs hot in recent seasons (league npG/npxG fell from ~1.0 to ~0.84),
    # so rescale every shot so each season's league total matches its goals. Without this
    # everyone looks like an underperformer in 2024/25 onwards.
    cal = sh.groupby("season").goal.sum() / sh.groupby("season").xG.sum()
    sh["xG_raw"] = sh.xG
    sh["xG"] = sh.xG * sh.season.map(cal)
    sh["on_t"] = sh.result.isin(["Goal", "SavedShot"]).astype(int)
    sh["post"] = (sh.result == "ShotOnPost").astype(int)
    sh["head"] = (sh.shotType == "Head").astype(int)
    sh["big"] = (sh.xG >= 0.3).astype(int)
    sh["box"] = ((sh.X >= 0.83) & sh.Y.between(0.21, 0.79)).astype(int)
    sh["far"] = (sh.X < 0.83).astype(int)
    sh["setp"] = sh.situation.isin(["FromCorner", "SetPiece", "DirectFreekick"]).astype(int)
    sh["cross"] = (sh.lastAction == "Cross").astype(int)
    agg = sh.groupby(["match_id", "player_id"]).agg(
        npg=("goal", "sum"), npxg=("xG", "sum"), shots=("goal", "size"), on_t=("on_t", "sum"),
        post=("post", "sum"), head=("head", "sum"), big=("big", "sum"), box=("box", "sum"),
        far=("far", "sum"), setp=("setp", "sum"), cross=("cross", "sum")).reset_index()
    df = app.merge(agg, on=["match_id", "player_id"], how="left")
    cols = ["npg", "npxg", "shots", "on_t", "post", "head", "big", "box", "far", "setp", "cross"]
    df[cols] = df[cols].fillna(0)

    # Role: most common starting position for the player in that season.
    starts = df[df.position != "Sub"]
    role = starts.groupby(["player_id", "season"]).position.agg(lambda s: s.mode().iat[0])
    df = df.join(role.rename("role"), on=["player_id", "season"])
    df = df[df.role.isin(ATTACK)].copy()
    df["group"] = df.role.map(GROUP)
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
    df["i"] = df.groupby("player_id").cumcount()

    # Keep shots that belong to kept appearances, with the row they map to (for simulation).
    key = df.reset_index()[["index", "match_id", "player_id"]]
    sh = sh.merge(key, on=["match_id", "player_id"])
    return df, sh, cal


def rolling(df, col, n, forward=False):
    """Sum of the last n appearances (inclusive), or of the next n (exclusive) if forward.
    NaN where the player's history doesn't reach that far."""
    v = df[col].to_numpy(dtype=float)
    i = df.groupby("player_id").cumcount().to_numpy() if "i" not in df else df["i"].to_numpy()
    size = df.groupby("player_id")["player_id"].transform("size").to_numpy()
    c = np.concatenate([[0.0], np.cumsum(v)])
    idx = np.arange(len(v))
    out = np.full(len(v), np.nan)
    if not forward:
        ok = i >= n - 1
        out[ok] = c[idx[ok] + 1] - c[idx[ok] + 1 - n]
    else:
        ok = i + n < size
        out[ok] = c[idx[ok] + 1 + n] - c[idx[ok] + 1]
    return pd.Series(out, index=df.index)


def build_windows(df, npg, past=PAST, fut=FUT):
    d = df[["player_id", "i", "npxg", "minutes"]].copy()
    d["npg"] = npg
    d["diff"] = d.npg - d.npxg
    out = pd.DataFrame(index=df.index)
    out["p_diff"] = rolling(d, "diff", past)
    out["p_xg"] = rolling(d, "npxg", past)
    out["f_diff"] = rolling(d, "diff", fut, forward=True)
    out["f_xg"] = rolling(d, "npxg", fut, forward=True)
    out["f_g"] = rolling(d, "npg", fut, forward=True)
    out["f_min"] = rolling(d, "minutes", fut, forward=True)
    return out


def pick_events(df, w, cold=COLD, min_xg=MIN_XG, gap=FUT):
    """First appearance where the past window turns cold, then no new event until the
    future window has passed, so events never overlap."""
    flag = (w.p_diff <= cold) & (w.p_xg >= min_xg) & w.f_diff.notna()
    idx = []
    for pid, rows in df[flag].groupby("player_id").i:
        last = -10**9
        for r, i in rows.items():
            if i - last > gap:
                idx.append(r)
                last = i
    return pd.Index(idx)


def simulate_npg(df, sh, n):
    """n simulated goal columns per appearance: each shot scores with prob = its xG."""
    rows = sh["index"].to_numpy()
    p = sh.xG.to_numpy()
    sims = np.zeros((n, len(df)))
    for k in range(n):
        g = (RNG.random(len(p)) < p).astype(float)
        sims[k] = np.bincount(rows, weights=g, minlength=len(df))
    return sims


def boot_ci(values, groups, stat=np.mean, n=1000):
    v = pd.Series(values).reset_index(drop=True)
    g = pd.Series(groups).reset_index(drop=True)
    by = {k: v[ix].to_numpy() for k, ix in g.groupby(g).groups.items()}
    keys = list(by)
    res = []
    for _ in range(n):
        pick = RNG.choice(len(keys), len(keys))
        res.append(stat(np.concatenate([by[keys[j]] for j in pick])))
    return float(np.percentile(res, 5)), float(np.percentile(res, 95))


def time_to_level(df, npg, ev, horizon=38):
    """Appearances after the cold window until cumulative npG - npxG since the window
    started is back to >= 0 (the 'level up' the user describes). NaN = not within horizon."""
    diff = (npg - df.npxg.to_numpy())
    pid = df.player_id.to_numpy()
    out = []
    for r in ev:
        start = r - PAST + 1
        cum = diff[start:r + 1].sum()
        k_found = np.nan
        for k in range(1, horizon + 1):
            j = r + k
            if j >= len(df) or pid[j] != pid[r]:
                break
            cum += diff[j]
            if cum >= 0:
                k_found = k
                break
        out.append(k_found)
    return np.array(out)


def main():
    data_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    df, sh, cal = load(data_dir)
    res = {"season_calibration": cal.round(3).to_dict(),"rows": len(df), "players": int(df.player_id.nunique()),
           "seasons": f"{df.season.min()} to {df.season.max()}", "last_date": str(df.date.max())}
    print(res)

    real = df.npg.to_numpy()
    sims = simulate_npg(df, sh, N_SIM)
    w = build_windows(df, real)

    # ---- 1. Event study at several definitions, real vs pure-luck simulation
    defs = [(5, -1.5), (8, -2.0), (8, -3.0), (10, -2.5), (15, -3.0)]
    tables = []
    sim_cache = {}
    for past, cold in defs:
        wd = build_windows(df, real, past=past)
        ev = pick_events(df, wd, cold=cold, min_xg=cold * -1)
        e = wd.loc[ev]
        n_sim_ev, fpos_sim, payback_sim = [], [], []
        for k in range(40):
            ws = build_windows(df, sims[k], past=past)
            evs = pick_events(df, ws, cold=cold, min_xg=cold * -1)
            n_sim_ev.append(len(evs))
            es = ws.loc[evs]
            fpos_sim.append((es.f_diff > 0).mean())
            payback_sim.append((es.f_diff >= -es.p_diff).mean())
        # Under pure luck, given the real future shots, how often does the next window come out positive?
        fut_sim = np.stack([build_windows(df, sims[k], past=past).loc[ev, "f_diff"].to_numpy()
                            for k in range(40)])
        row = {
            "past": past, "cold": cold, "events": len(e), "players": int(df.loc[ev].player_id.nunique()),
            "events_if_pure_luck": float(np.mean(n_sim_ev)),
            "past_diff": float(e.p_diff.mean()), "past_xg": float(e.p_xg.mean()),
            "fut_diff": float(e.f_diff.mean()),
            "fut_diff_ci": boot_ci(e.f_diff, df.loc[ev].player_id),
            "fut_ratio_g_per_xg": float(e.f_g.sum() / e.f_xg.sum()),
            "share_fut_over": float((e.f_diff > 0).mean()),
            "share_fut_over_luck": float(fut_sim.__gt__(0).mean()),
            "share_full_payback": float((e.f_diff >= -e.p_diff).mean()),
            "share_full_payback_luck": float(np.mean(payback_sim)),
        }
        tables.append(row)
        print(row)
        if (past, cold) == (PAST, COLD):
            sim_cache["ev"] = ev
    res["event_study"] = tables

    ev = sim_cache["ev"]
    e = w.loc[ev].join(df[["player_id", "player", "season", "group", "i", "date"]])

    # Baseline: every 8-game window with the same chance volume (npxG >= 2), any finishing.
    base = w[(w.p_xg >= MIN_XG) & w.f_diff.notna()]
    res["baseline_all_windows"] = {
        "fut_diff": float(base.f_diff.mean()), "share_fut_over": float((base.f_diff > 0).mean()),
        "fut_ratio": float(base.f_g.sum() / base.f_xg.sum())}
    hot = base[base.p_diff >= 2]
    res["hot_mirror"] = {"n": len(hot), "fut_diff": float(hot.f_diff.mean()),
                         "fut_ratio": float(hot.f_g.sum() / hot.f_xg.sum())}

    # Does the chance volume hold up after a cold run? (npxG per 90 before vs after)
    pre_min = rolling(df, "minutes", PAST)
    e["p_min"] = pre_min.loc[ev]
    res["volume_after_cold"] = {
        "pre_xg90": float(e.p_xg.sum() / e.p_min.sum() * 90),
        "post_xg90": float(e.f_xg.sum() / e.f_min.sum() * 90),
        "pre_min_per_app": float(e.p_min.mean() / PAST), "post_min_per_app": float(e.f_min.mean() / FUT),
        "pre_g90": float((e.p_xg + e.p_diff).sum() / e.p_min.sum() * 90),
        "post_g90": float(e.f_g.sum() / e.f_min.sum() * 90)}
    b_min = pre_min.loc[base.index]
    res["volume_all"] = {"pre_xg90": float(base.p_xg.sum() / b_min.sum() * 90),
                         "post_xg90": float(base.f_xg.sum() / base.f_min.sum() * 90)}

    # ---- 2. Time to get level (cumulative npG - npxG back to >= 0)
    ttl = time_to_level(df, real, ev)
    # Pure luck: the real cold window, then the real future shots with simulated outcomes.
    ttl_sim = []
    for k in range(40):
        mixed = sims[k].copy()
        for r in ev:
            mixed[r - PAST + 1:r + 1] = real[r - PAST + 1:r + 1]
        ttl_sim.append(time_to_level(df, mixed, ev))
    ttl_sim = np.concatenate(ttl_sim)
    def ttl_summary(t):
        return {f"by_{h}": float(np.mean(t <= h)) for h in (5, 10, 15, 20, 38)} | {
            "median_if_levelled": float(np.nanmedian(t))}
    res["time_to_level"] = {"real": ttl_summary(ttl), "pure_luck": ttl_summary(ttl_sim)}

    # ---- 3. Bad finishers vs bad luck: prior career finishing before the cold window
    cum = df.groupby("player_id")
    df["c_g"] = cum.npg.cumsum() - df.npg
    df["c_xg"] = cum.npxg.cumsum() - df.npxg
    df["c_sh"] = cum.shots.cumsum() - df.shots
    start = ev - PAST + 1
    pr = pd.DataFrame({"g": df.loc[start, "c_g"].to_numpy(), "xg": df.loc[start, "c_xg"].to_numpy(),
                       "sh": df.loc[start, "c_sh"].to_numpy()}, index=ev)
    # Shrink toward 1.0: ratio with 25 npxG of prior weight (finishing is mostly noise).
    pr["fin"] = (pr.g + 25) / (pr.xg + 25)
    e = e.join(pr)
    e["fin_band"] = pd.cut(e.fin, [0, 0.93, 1.03, 9],
                           labels=["Poor finisher record", "About average", "Good finisher record"])
    e.loc[e.xg < 5, "fin_band"] = np.nan
    fb = e.dropna(subset=["fin_band"]).groupby("fin_band", observed=True).apply(
        lambda x: pd.Series({"n": len(x), "fut_diff": x.f_diff.mean(),
                             "fut_ratio": x.f_g.sum() / x.f_xg.sum(),
                             "fut_g90": x.f_g.sum() / x.f_min.sum() * 90}))
    res["by_prior_finishing"] = fb.reset_index().to_dict("records")

    # How much real finishing skill is there? split-half reliability of G/xG over careers.
    car = df.groupby("player_id").agg(g=("npg", "sum"), xg=("npxg", "sum"), sh=("shots", "sum"))
    car = car[car.xg >= 15]
    car["ratio"] = car.g / car.xg
    noise_var = (1 / car.xg).mean()  # Poisson-ish variance of G/xG
    res["finishing_skill"] = {"players": len(car), "sd_observed": float(car.ratio.std()),
                              "sd_noise_est": float(np.sqrt(noise_var)),
                              "sd_true_est": float(np.sqrt(max(car.ratio.var() - noise_var, 0)))}
    half = df[df.player_id.isin(car.index)].copy()
    half["h"] = half.i % 2
    hh = half.groupby(["player_id", "h"]).agg(g=("npg", "sum"), xg=("npxg", "sum")).unstack()
    r1 = hh["g"][0] - hh["xg"][0]
    r2 = hh["g"][1] - hh["xg"][1]
    res["finishing_skill"]["split_half_corr_g_minus_xg"] = float(np.corrcoef(r1, r2)[0, 1])

    # ---- 4. Pre-switch signals within cold cases
    feat = {}
    for c in ["shots", "on_t", "post", "head", "big", "box", "far", "setp", "cross", "minutes"]:
        feat[c] = rolling(df, c, PAST).loc[ev]
    f = pd.DataFrame(feat)
    e = e.join(f)
    e["shots90"] = e.shots / e.minutes * 90
    e["xg90"] = e.p_xg / e.minutes * 90
    e["xg_per_shot"] = e.p_xg / e.shots
    e["on_target_share"] = e.on_t / e.shots
    e["big_share"] = e.big / e.shots
    e["head_share"] = e["head"] / e.shots
    e["far_share"] = e.far / e.shots
    e["setp_share"] = e.setp / e.shots
    e["post_hits"] = e.post
    e["f_g90"] = e.f_g / e.f_min * 90
    e["f_xg90"] = e.f_xg / e.f_min * 90
    sig = []
    for c in ["xg90", "shots90", "xg_per_shot", "big_share", "on_target_share", "head_share",
              "far_share", "setp_share", "post_hits", "fin"]:
        x = e[[c, "f_diff", "f_g90", "f_xg90", "f_min", "player_id"]].dropna()
        x = x[x.f_min >= 450]
        q = pd.qcut(x[c].rank(method="first"), 3, labels=["low", "mid", "high"])
        tab = x.groupby(q, observed=True).apply(lambda y: pd.Series({
            "n": len(y), "lo": y[c].min(), "hi": y[c].max(),
            "fut_diff_per10": y.f_diff.mean(),
            "fut_g90": y.f_g90.mean(), "fut_xg90": y.f_xg90.mean()}))
        hi = x[q == "high"]; lo = x[q == "low"]
        gap = hi.f_diff.mean() - lo.f_diff.mean()
        # proper bootstrap of the hi-lo gap over players
        pids = x.player_id.unique()
        gaps = []
        for _ in range(500):
            pick = RNG.choice(pids, len(pids))
            cnt = pd.Series(pick).value_counts()
            wgt = x.player_id.map(cnt).fillna(0)
            h = (q == "high"); l = (q == "low")
            gaps.append(np.average(x.f_diff[h], weights=wgt[h] + 1e-9) -
                        np.average(x.f_diff[l], weights=wgt[l] + 1e-9))
        sig.append({"signal": c, "tiers": tab.reset_index().to_dict("records"),
                    "gap_fut_diff": float(gap),
                    "gap_ci": (float(np.percentile(gaps, 5)), float(np.percentile(gaps, 95))),
                    "gap_fut_g90": float(hi.f_g90.mean() - lo.f_g90.mean())})
    res["signals"] = sig
    pos = e[e.f_min >= 450].groupby("group").apply(lambda y: pd.Series({
        "n": len(y), "fut_diff": y.f_diff.mean(), "fut_ratio": y.f_g.sum() / y.f_xg.sum(),
        "share_over": (y.f_diff > 0).mean()}))
    res["by_position"] = pos.reset_index().to_dict("records")

    # ---- 5. The practical rule: cold + still high volume + not a proven poor finisher.
    e["rule"] = (e.xg90 >= 0.35) & (e.minutes >= 450) & (e.fin.fillna(1) >= 0.93)
    for name, sub in [("rule", e[e.rule & (e.f_min >= 450)]),
                      ("cold_not_rule", e[~e.rule & (e.f_min >= 450)])]:
        res[name] = {"n": len(sub), "players": int(sub.player_id.nunique()),
                     "past_g90": float(((sub.p_xg + sub.p_diff).sum() / sub.minutes.sum()) * 90),
                     "past_xg90": float(sub.p_xg.sum() / sub.minutes.sum() * 90),
                     "fut_g90": float(sub.f_g.sum() / sub.f_min.sum() * 90),
                     "fut_xg90": float(sub.f_xg.sum() / sub.f_min.sum() * 90),
                     "fut_diff": float(sub.f_diff.mean()),
                     "share_3plus_goals_next10": float((sub.f_g >= 3).mean())}
    # Same-volume players who were NOT cold, for comparison.
    allw = w.join(rolling(df, "minutes", PAST).rename("p_min"))
    allw = allw[(allw.p_min >= 450) & (allw.f_min >= 450) & allw.f_diff.notna()]
    allw["xg90"] = allw.p_xg / allw.p_min * 90
    peers = allw[(allw.xg90 >= 0.35) & (allw.p_diff.abs() < 1)]
    res["rule_peers_level"] = {"n": len(peers), "past_xg90": float(peers.p_xg.sum() / peers.p_min.sum() * 90),
                               "fut_g90": float(peers.f_g.sum() / peers.f_min.sum() * 90)}

    e.drop(columns=["date"]).to_csv(out_dir / "cold_events.csv", index=False)
    famous = e.sort_values("p_diff").head(25)[["player", "season", "group", "p_xg", "p_diff", "f_g", "f_xg", "f_diff"]]
    res["deepest_cold_runs"] = famous.to_dict("records")

    # ---- 6. Current candidates: last 8 appearances as of the latest data.
    last = df.groupby("player_id").tail(1)
    cur_season = df.season.max()
    last = last[last.season == cur_season]
    cand = []
    for r in last.index:
        pid = df.at[r, "player_id"]
        lo = r - PAST + 1
        if lo < 0 or df.at[lo, "player_id"] != pid:
            continue
        win = df.loc[lo:r]
        mins = win.minutes.sum()
        xg = win.npxg.sum(); g = win.npg.sum(); s = win.shots.sum()
        prior_g = df.at[lo, "c_g"]; prior_xg = df.at[lo, "c_xg"]
        fin = (prior_g + 25) / (prior_xg + 25)
        season_w = df[(df.player_id == pid) & (df.season == cur_season)]
        cand.append({"player": df.at[r, "player"], "team": df.at[r, "team"], "role": df.at[r, "group"],
                     "apps": len(win), "minutes": int(mins), "npg": int(g), "npxg": round(xg, 2),
                     "diff": round(g - xg, 2), "xg90": round(xg / mins * 90, 2) if mins else 0,
                     "shots90": round(s / mins * 90, 2) if mins else 0,
                     "xg_per_shot": round(xg / s, 3) if s else 0,
                     "big": int(win.big.sum()), "on_target": int(win.on_t.sum()), "posts": int(win.post.sum()),
                     "career_fin": round(fin, 2), "career_npxg": round(prior_xg, 1),
                     "season_npg": int(season_w.npg.sum()), "season_npxg": round(season_w.npxg.sum(), 2),
                     "first_window_date": str(df.at[lo, "date"].date()),
                     "last_app": str(df.at[r, "date"].date())})
    cand = pd.DataFrame(cand)
    cand["fits_rule"] = (cand["diff"] <= COLD) & (cand.xg90 >= 0.35) & (cand.minutes >= 450) & (cand.career_fin >= 0.93)
    cand["watch"] = (cand["diff"] <= -1.0) & (cand.xg90 >= 0.30) & (cand.minutes >= 400)
    cand = cand.sort_values("diff")
    cand.to_csv(out_dir / "candidates.csv", index=False)
    res["candidates"] = cand[cand.watch].to_dict("records")

    (out_dir / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("signals", "deepest_cold_runs", "candidates")},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
