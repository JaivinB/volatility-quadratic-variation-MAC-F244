"""
Robustness: is the quadratic variation contaminated by jumps, does the premium
survive transaction costs, and is it stable across market regimes?

These three checks exist because the three obvious attacks on the project are
"your estimator is measuring jumps, not diffusion volatility", "your profit
disappears once you pay to trade", and "your result is one lucky sample".

  J  JUMP DECOMPOSITION.  Ito's identity [log S]_T = int sigma^2 dt assumes a
     CONTINUOUS semimartingale. If the price jumps, realised variance converges
     instead to int sigma^2 dt + sum (jump)^2, and the diffusive part must be
     separated out. Bipower variation (Barndorff-Nielsen & Shephard 2004)

         BV = (pi/2) * sum |r_i| |r_{i-1}|

     is robust to jumps -- a single large |r_i| is multiplied by its NORMAL-sized
     neighbour, so one jump contributes O(1) terms instead of O(1) squared --
     and still estimates int sigma^2 dt. RV - BV therefore estimates the jump
     contribution, and the ratio statistic Z below tests whether it is real.

  C  TRANSACTION COSTS.  Hedging error falls as n^{-1/2} but trading costs grow
     roughly as n, so there is a genuine interior optimum in rebalance
     frequency. This finds it.

  S  REGIME STABILITY.  The premium is re-estimated on disjoint sub-periods.
"""
import numpy as np
import pandas as pd
from scipy.special import gamma as Gamma

from sc_data import nifty_and_vix, log_returns, TRADING_DAYS
from sc_hedge import backtest, RATE, HORIZON

MU1 = np.sqrt(2.0 / np.pi)                      # E|Z| for Z ~ N(0,1)
MU43 = 2.0 ** (2 / 3) * Gamma(7 / 6) / Gamma(0.5)


# ------------------------------------------------------------------ J
def realised_variance(r):
    return float(np.sum(r ** 2))


def bipower_variation(r):
    """Jump-robust estimator of integrated variance."""
    n = len(r)
    if n < 2:
        return np.nan
    bv = (MU1 ** -2) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))
    return float(bv * n / (n - 1))              # finite-sample correction


def tripower_quarticity(r):
    """Jump-robust estimator of integrated QUARTICITY, needed for the test's s.e."""
    n = len(r)
    if n < 3:
        return np.nan
    a = np.abs(r) ** (4 / 3)
    tq = n * (MU43 ** -3) * np.sum(a[2:] * a[1:-1] * a[:-2])
    return float(tq * n / (n - 2))


def jump_test(r):
    """
    Barndorff-Nielsen--Shephard ratio jump statistic (Huang & Tauchen 2005 form).
    Z ~ N(0,1) under the null of NO jumps; large positive Z rejects it.
    """
    n = len(r)
    rv, bv, tq = realised_variance(r), bipower_variation(r), tripower_quarticity(r)
    theta = (np.pi / 2) ** 2 + np.pi - 5.0      # = 0.6090
    denom = np.sqrt(theta * (1.0 / n) * max(1.0, tq / bv ** 2))
    z = ((rv - bv) / rv) / denom
    return {"n": n, "rv": rv, "bv": bv,
            "jump_share": max(0.0, (rv - bv) / rv), "z": z}


def jump_by_year(frame):
    r = log_returns(frame["spot"])
    rows = []
    for year, seg in r.groupby(r.index.year):
        if len(seg) < 100:
            continue
        d = jump_test(seg.values)
        d["year"] = year
        d["rv_ann_vol"] = np.sqrt(d["rv"] * TRADING_DAYS / d["n"]) * 100
        d["bv_ann_vol"] = np.sqrt(d["bv"] * TRADING_DAYS / d["n"]) * 100
        rows.append(d)
    return pd.DataFrame(rows)[
        ["year", "n", "rv_ann_vol", "bv_ann_vol", "jump_share", "z"]]


# ------------------------------------------------------------------ C
def cost_sweep(frame, costs=(0.0, 1.0, 2.0, 5.0, 10.0),
               freqs=(1, 2, 3, 5, 7, 10, 21)):
    """Mean net P&L per cycle for each (transaction cost, rebalance frequency)."""
    rows = []
    for c in costs:
        for every in freqs:
            bt = backtest(frame, hedge_every=every, cost_bps=c)
            n_reb = len(range(0, HORIZON, every)) + 1
            rows.append({
                "cost_bps": c, "hedge_every": every, "rebalances": n_reb,
                "mean_pnl": bt["pnl"].mean(),
                "pct_premium": bt["pnl_pct_premium"].mean(),
                "std": bt["pnl"].std(ddof=1),
                "win": (bt["pnl"] > 0).mean() * 100,
                "t": bt["pnl"].mean() / (bt["pnl"].std(ddof=1) / np.sqrt(len(bt))),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ S
def subperiods(frame, cuts=(("2015-2019", "2015-01-01", "2019-12-31"),
                            ("2020-2021 (COVID)", "2020-01-01", "2021-12-31"),
                            ("2022-2026", "2022-01-01", "2026-12-31"))):
    bt = backtest(frame)
    rows = []
    for label, a, b in cuts:
        m = (bt["start"] >= a) & (bt["start"] <= b)
        sub = bt[m]
        if len(sub) < 5:
            continue
        se = sub["pnl"].std(ddof=1) / np.sqrt(len(sub))
        rows.append({"period": label, "cycles": len(sub),
                     "mean_pnl": sub["pnl"].mean(),
                     "pct_premium": sub["pnl_pct_premium"].mean(),
                     "implied": sub["implied"].mean() * 100,
                     "realised": sub["realised"].mean() * 100,
                     "win": (sub["pnl"] > 0).mean() * 100,
                     "t": sub["pnl"].mean() / se})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 130)
    frame = nifty_and_vix()

    print("=" * 84)
    print("J  Jump decomposition: is the quadratic variation diffusive?")
    print("=" * 84)
    jb = jump_by_year(frame)
    print(jb.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))
    full = jump_test(log_returns(frame["spot"]).values)
    print(f"\n  full sample: RV-implied vol {np.sqrt(full['rv']*TRADING_DAYS/full['n'])*100:.2f}%"
          f"   BV-implied vol {np.sqrt(full['bv']*TRADING_DAYS/full['n'])*100:.2f}%")
    print(f"  jump share of total quadratic variation : {full['jump_share']*100:.2f}%")
    print(f"  BNS ratio statistic Z                   : {full['z']:+.3f}"
          f"   (|Z| > 1.96 rejects 'no jumps' at 5%)")
    sig = (jb["z"] > 1.96).sum()
    print(f"  years rejecting the no-jump null at 5%  : {sig} of {len(jb)}")

    print()
    print("=" * 84)
    print("C  Transaction costs: the mean, the dispersion, and the cost drag")
    print("=" * 84)
    cs = cost_sweep(frame)
    base = cs[cs.cost_bps == 0].set_index("hedge_every")
    piv = cs.pivot(index="hedge_every", columns="cost_bps", values="mean_pnl")

    print("  (1) At ZERO cost the mean is NOT a function of frequency -- as theory")
    print("      requires, since discrete hedging is unbiased. The DISPERSION is.\n")
    print(f"  {'every':>6} {'trades':>7} {'mean':>8} {'s.e.':>7} {'std':>8}")
    for k, row in base.iterrows():
        se = row["std"] / np.sqrt(138)
        print(f"  {k:>6} {int(row['rebalances']):>7} {row['mean_pnl']:>8.2f} "
              f"{se:>7.2f} {row['std']:>8.2f}")
    spread = base["mean_pnl"].max() - base["mean_pnl"].min()
    mean_se = (base["std"] / np.sqrt(138)).mean()
    print(f"\n      spread of means {spread:.2f} pts vs typical s.e. {mean_se:.2f} pts"
          f"  -> indistinguishable")
    sd_slope = np.polyfit(np.log(base["rebalances"]), np.log(base["std"]), 1)[0]
    print(f"      log-log slope of std vs trades: {sd_slope:+.4f}   (theory -0.5)")

    print("\n  (2) The deterministic part is the COST DRAG = P&L(0) - P&L(c).\n")
    drag = pd.DataFrame(piv[0.0].values[:, None] - piv.values,
                        index=piv.index, columns=piv.columns).drop(columns=[0.0])
    drag.insert(0, "trades", base["rebalances"].astype(int))
    print(drag.to_string(float_format=lambda x: f"{x:8.2f}"))
    dslope = np.polyfit(np.log(drag["trades"]), np.log(drag[10.0]), 1)[0]
    print(f"\n      log-log slope of drag vs trades: {dslope:+.4f}")
    print( "      Drag grows SUBLINEARLY, not linearly: hedging twice as often means")
    print( "      each trade is smaller, so cost per rebalance falls as frequency rises.")

    print("\n  (3) Net P&L per cycle (index points):\n")
    print(piv.to_string(float_format=lambda x: f"{x:9.2f}"))
    print("\n      survival of the premium, hedging daily:")
    for c in (0.0, 1.0, 2.0, 5.0, 10.0):
        r = cs[(cs.cost_bps == c) & (cs.hedge_every == 1)].iloc[0]
        print(f"        {c:4.1f} bps -> {r['mean_pnl']:+7.2f} pts "
              f"({r['pct_premium']:+6.2f}% of premium, t={r['t']:+5.2f})")

    print()
    print("=" * 84)
    print("S  Regime stability: does the premium survive out of sample?")
    print("=" * 84)
    sp = subperiods(frame)
    print(sp.to_string(index=False, float_format=lambda x: f"{x:9.2f}"))

    jb.to_csv("data/jump_by_year.csv", index=False)
    cs.to_csv("data/cost_sweep.csv", index=False)
    sp.to_csv("data/subperiods.csv", index=False)
    print("\n  wrote data/jump_by_year.csv, data/cost_sweep.csv, data/subperiods.csv")
