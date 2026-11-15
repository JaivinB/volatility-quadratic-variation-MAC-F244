"""
Figures for the MAC F244 project. Reads the CSVs written by sc_vol.py and
sc_hedge.py so the plots and the report quote the same numbers.

Each figure is written twice: PDF for LaTeX, PNG for quick viewing.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sc_data import nifty_and_vix, log_returns, TRADING_DAYS
from sc_vol import realised_variance, vol_diagnostics, implied_vs_realised, ols

FIG = Path(__file__).resolve().parent / "figures"
DATA = Path(__file__).resolve().parent / "data"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8,
    "figure.constrained_layout.use": True,
})
INK, ACC, WARN = "#1f3b57", "#c2622d", "#7a2e3e"


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=170)
    plt.close(fig)
    print(f"  figures/{name}.pdf")


# ------------------------------------------------------------------ 1
def fig_data_overview(frame):
    fig, ax = plt.subplots(2, 1, figsize=(6.6, 3.4), sharex=True)
    ax[0].plot(frame.index, frame["spot"], lw=0.8, color=INK)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("NIFTY 50 (log)")
    ax[0].set_title("Eleven years of NIFTY 50 and its volatility "
                    "(Yahoo Finance, 2,903 trading days)")

    vol = np.sqrt(realised_variance(frame["spot"])) * 100
    ax[1].plot(frame.index, frame["iv"] * 100, lw=0.8, color=ACC,
               label="India VIX (implied, 30-day)")
    ax[1].plot(vol.index, vol.values, lw=0.8, color=INK, alpha=0.75,
               label="realised, 21-day quadratic variation")
    ax[1].set_ylabel("volatility (% p.a.)")
    ax[1].set_xlabel("date")
    ax[1].legend(loc="upper right")
    save(fig, "data_overview")


# ------------------------------------------------------------------ 2
def fig_rv_convergence():
    c = pd.read_csv(DATA / "rv_convergence.csv")
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7))

    ax[0].loglog(c["n"], c["sd_theory"], "-", color=ACC, lw=1.4,
                 label=r"theory  $\sqrt{2T\!\int\!\sigma^4dt\,/\,n}$")
    ax[0].loglog(c["n"], c["sd_measured"], "o", color=INK, ms=4.5,
                 label="measured, 4000 paths")
    ax[0].set_xlabel("sampling intervals $n$")
    ax[0].set_ylabel(r"s.d. of $RV_n-IV$")
    ax[0].set_title("Realised variance converges at the predicted rate")
    ax[0].legend(loc="lower left")

    ax[1].axhline(0, color="0.6", lw=0.8)
    ax[1].semilogx(c["n"], c["bias_pct_iv"], "o-", color=INK, ms=4, lw=1)
    ax[1].set_xlabel("sampling intervals $n$")
    ax[1].set_ylabel("bias (% of $IV$)")
    ax[1].set_ylim(-2, 2)
    ax[1].set_title("and is unbiased throughout")
    save(fig, "rv_convergence")


# ------------------------------------------------------------------ 3
def fig_vol_clustering(frame):
    d = vol_diagnostics(frame)
    r = log_returns(frame["spot"])
    lags = np.array(d["lags"])

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7))
    w = 0.28
    idx = np.arange(len(lags))
    ax[0].bar(idx - w, d["acf_returns"], w, color="0.65", label="$r_t$")
    ax[0].bar(idx, d["acf_abs"], w, color=INK, label="$|r_t|$")
    ax[0].bar(idx + w, d["acf_sq"], w, color=ACC, label="$r_t^2$")
    ci = 1.96 / np.sqrt(len(r))
    ax[0].axhline(ci, color=WARN, ls=":", lw=0.9)
    ax[0].axhline(-ci, color=WARN, ls=":", lw=0.9)
    ax[0].axhline(0, color="0.5", lw=0.7)
    ax[0].set_xticks(idx)
    ax[0].set_xticklabels(lags)
    ax[0].set_xlabel("lag (trading days)")
    ax[0].set_ylabel("autocorrelation")
    ax[0].set_title("Returns are near-unpredictable; their size is not")
    ax[0].legend(loc="upper right", ncol=3)

    vol = np.sqrt(realised_variance(frame["spot"]))
    dvol = vol.diff().dropna()
    common = r.index.intersection(dvol.index)
    ax[1].scatter(r.loc[common] * 100, dvol.loc[common] * 100, s=2.5,
                  color=INK, alpha=0.3, edgecolors="none")
    xs = np.linspace(r.loc[common].min(), r.loc[common].max(), 50)
    b = np.polyfit(r.loc[common], dvol.loc[common], 1)
    ax[1].plot(xs * 100, np.polyval(b, xs) * 100, color=ACC, lw=1.5)
    ax[1].axhline(0, color="0.6", lw=0.7)
    ax[1].axvline(0, color="0.6", lw=0.7)
    ax[1].set_xlabel("daily log return (%)")
    ax[1].set_ylabel(r"change in realised vol (pp)")
    ax[1].set_title(f"Leverage effect: corr $=$ {d['leverage_corr']:+.3f}")
    save(fig, "vol_clustering")


# ------------------------------------------------------------------ 4
def fig_implied_vs_realised(frame):
    pairs = pd.read_csv(DATA / "implied_vs_realised.csv", parse_dates=["start"])
    X = np.column_stack([np.ones(len(pairs)), pairs["implied"].values])
    fit = ols(pairs["realised"].values, X, ["a", "b"])
    a, b = fit["beta"]

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.8))
    lo, hi = 0.05, 0.60
    ax[0].plot([lo * 100, hi * 100], [lo * 100, hi * 100], "--",
               color="0.55", lw=1,
               label="unbiased forecast ($a{=}0$, $b{=}1$)")
    xs = np.linspace(lo, hi, 50)
    ax[0].plot(xs * 100, (a + b * xs) * 100, color=ACC, lw=1.6,
               label=f"fitted  $b={b:.2f}$ (s.e. {fit['se'][1]:.2f})")
    ax[0].scatter(pairs["implied"] * 100, pairs["realised"] * 100, s=14,
                  color=INK, alpha=0.65, edgecolors="none")
    ax[0].set_xlabel("India VIX at inception (%)")
    ax[0].set_ylabel("realised vol, next 21 days (%)")
    ax[0].set_title("Implied overstates what follows")
    ax[0].legend(loc="upper left")
    ax[0].set_xlim(8, 42)
    ax[0].set_ylim(0, 62)

    g = pairs["gap_vol_pts"]
    ax[1].bar(pairs["start"], g, width=22,
              color=[INK if v > 0 else WARN for v in g])
    ax[1].axhline(g.mean(), color=ACC, lw=1.4,
                  label=f"mean {g.mean():+.2f} pts ($t={g.mean()/(g.std(ddof=1)/np.sqrt(len(g))):+.2f}$)")
    ax[1].axhline(0, color="0.5", lw=0.8)
    ax[1].set_ylabel("implied $-$ realised (vol pts)")
    ax[1].set_xlabel("cycle start")
    ax[1].set_title(f"Premium positive in {(g>0).mean()*100:.0f}% of 138 months")
    ax[1].legend(loc="lower left")
    save(fig, "implied_vs_realised")


# ------------------------------------------------------------------ 5
def fig_hedge(frame):
    conv = pd.read_csv(DATA / "convergence.csv")
    bt = pd.read_csv(DATA / "backtest_cycles.csv", parse_dates=["start"])

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7))
    ax[0].loglog(conv["rebalances"], conv["std"], "o", color=INK, ms=4.5,
                 label="measured")
    k = conv["std"].iloc[0] * conv["rebalances"].iloc[0] ** 0.5
    ax[0].loglog(conv["rebalances"], k * conv["rebalances"] ** -0.5, "-",
                 color=ACC, lw=1.4, label=r"$n^{-1/2}$ reference")
    ax[0].set_xlabel("rebalances per month $n$")
    ax[0].set_ylabel("s.d. of hedge error (pts)")
    ax[0].set_title("Discrete hedging error falls as $n^{-1/2}$")
    ax[0].legend(loc="lower left")

    cum = bt["pnl"].cumsum()
    ax[1].plot(bt["start"], cum, color=INK, lw=1.3)
    ax[1].fill_between(bt["start"], 0, cum, color=INK, alpha=0.12)
    ax[1].axhline(0, color="0.5", lw=0.8)
    ax[1].set_ylabel("cumulative P&L (index pts)")
    ax[1].set_xlabel("cycle start")
    ax[1].set_title(f"Cumulative P&L: {cum.iloc[-1]:,.0f} pts, 138 months")
    save(fig, "hedge")


# ------------------------------------------------------------------ 6
def fig_ito_vs_actual(frame):
    from sc_hedge import backtest, ito_prediction
    bt = pd.read_csv(DATA / "backtest_cycles.csv")
    pred = ito_prediction(frame)
    act = bt["pnl"].values
    corr = np.corrcoef(pred, act)[0, 1]

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    lim = 1.05 * max(np.abs(pred).max(), np.abs(act).max())
    ax.plot([-lim, lim], [-lim, lim], "--", color="0.55", lw=1)
    ax.scatter(pred, act, s=16, color=INK, alpha=0.65, edgecolors="none")
    ax.set_xlabel("Itô prediction (pts)")
    ax.set_ylabel("realised P&L (pts)")
    ax.set_title(f"corr $=$ {corr:+.3f}")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    save(fig, "ito_vs_actual")
    return corr


# ------------------------------------------------------------------ 7
def fig_robustness():
    """Jumps, transaction costs and regime stability, in one strip."""
    jb = pd.read_csv(DATA / "jump_by_year.csv")
    cs = pd.read_csv(DATA / "cost_sweep.csv")
    sp = pd.read_csv(DATA / "subperiods.csv")

    fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.15))

    # (a) realised vs bipower volatility by year
    x = np.arange(len(jb))
    ax[0].bar(x - 0.21, jb["rv_ann_vol"], 0.42, color=INK, label="$\\sqrt{RV}$ (all)")
    ax[0].bar(x + 0.21, jb["bv_ann_vol"], 0.42, color=ACC, label="$\\sqrt{BV}$ (diffusive)")
    ax[0].set_xticks(x[::2])
    ax[0].set_xticklabels([str(y)[-2:] for y in jb["year"][::2]])
    ax[0].set_xlabel("year")
    ax[0].set_ylabel("volatility (% p.a.)")
    ax[0].set_title("Jumps are a small share of QV")
    ax[0].legend(loc="upper left", fontsize=6.5)

    # (b) net P&L against transaction cost, by rebalance frequency
    for every, col, lab in ((1, INK, "daily"), (7, ACC, "weekly"),
                            (21, "0.55", "twice only")):
        g = cs[cs.hedge_every == every].sort_values("cost_bps")
        ax[1].plot(g["cost_bps"], g["mean_pnl"], "o-", color=col, ms=3.5,
                   lw=1.2, label=lab)
    ax[1].axhline(0, color=WARN, ls=":", lw=1)
    ax[1].set_xlabel("round-trip cost (bps)")
    ax[1].set_ylabel("net P&L / cycle (pts)")
    ax[1].set_title("The premium survives costs")
    ax[1].legend(loc="lower left", fontsize=6.5)

    # (c) sub-period stability
    y = np.arange(len(sp))
    se = sp["mean_pnl"] / sp["t"]
    ax[2].barh(y, sp["mean_pnl"], xerr=se, color=INK, alpha=0.85,
               error_kw={"lw": 1.1, "capsize": 2.5, "ecolor": WARN})
    ax[2].axvline(0, color="0.5", lw=0.8)
    ax[2].set_yticks(y)
    ax[2].set_yticklabels([p.replace(" (COVID)", "\nCOVID") for p in sp["period"]],
                          fontsize=6.5)
    ax[2].set_xlabel("mean P&L / cycle (pts)")
    ax[2].set_title("Stable across regimes")
    save(fig, "robustness")


if __name__ == "__main__":
    frame = nifty_and_vix()
    print("writing figures")
    fig_data_overview(frame)
    fig_rv_convergence()
    fig_vol_clustering(frame)
    fig_implied_vs_realised(frame)
    fig_hedge(frame)
    c = fig_ito_vs_actual(frame)
    fig_robustness()
    print(f"  (Itô vs actual corr = {c:+.4f})")
