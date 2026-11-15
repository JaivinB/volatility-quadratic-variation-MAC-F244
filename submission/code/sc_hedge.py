"""
Discrete delta hedging: the convergence law, and a backtest on NIFTY 50.

A short call hedged with Delta = V_S shares is riskless only in the continuous
limit. Rebalancing n times over [0, T] leaves a residual. Two questions:

  (R1) How fast does that residual vanish as n grows?  Boyle-Emanuel (1980)
       predicts std(error) proportional to n^{-1/2}.  Measured on simulated GBM,
       where sigma is known exactly, so the only error is discretisation.

  (R2) On real NIFTY data, hedging at the *implied* vol quoted by India VIX
       while the index realises something else, what is the P&L?  Ito gives

           P&L  =  1/2 * integral of e^{r(T-t)} S^2 Gamma (sigma_i^2 - sigma_r^2) dt

       so a short option earns when implied exceeds realised. This is the
       variance risk premium, and it is what the backtest measures.
"""
import numpy as np
import pandas as pd

from sc_data import nifty_and_vix, log_returns, TRADING_DAYS
from sc_model import call_price, call_delta, call_gamma

RATE = 0.065          # flat annual; ~91-day T-bill over the sample. See report.
HORIZON = 21          # trading days per cycle, i.e. one listed month
RNG_SEED = 2026


def hedge_once(path, strike, rate, price_vol, hedge_vol, ttm_grid, cost_bps=0.0):
    """
    Self-financing hedge of ONE short call along one price path.

    path      : spot on each rebalance date, path[0] at inception, path[-1] expiry
    price_vol : vol used to set the premium received at inception
    hedge_vol : vol used to compute each delta (per step, scalar or array)
    ttm_grid  : time to maturity on each date, ttm_grid[-1] == 0
    cost_bps  : proportional transaction cost, in basis points of the notional
                traded at each rebalance. 0.0 reproduces the frictionless case
                exactly, so every earlier result is unchanged.

    Returns terminal cash, which is the hedge error (0 for a perfect hedge).
    """
    path = np.asarray(path, float)
    ttm = np.asarray(ttm_grid, float)
    hv = np.broadcast_to(np.asarray(hedge_vol, float), path.shape)
    fee = cost_bps * 1e-4

    premium = float(call_price(path[0], strike, rate, price_vol, ttm[0]))
    delta = float(call_delta(path[0], strike, rate, hv[0], ttm[0]))
    cash = premium - delta * path[0]
    cash -= fee * abs(delta) * path[0]                   # cost of the initial hedge

    for i in range(1, len(path)):
        dt = ttm[i - 1] - ttm[i]
        cash *= np.exp(rate * dt)
        if ttm[i] > 0:
            new = float(call_delta(path[i], strike, rate, hv[i], ttm[i]))
            cash -= (new - delta) * path[i]
            cash -= fee * abs(new - delta) * path[i]     # cost of rebalancing
            delta = new

    cash += delta * path[-1]                             # liquidate the hedge
    cash -= fee * abs(delta) * path[-1]                  # cost of liquidating
    cash -= max(path[-1] - strike, 0.0)                  # settle the short call
    return cash


# ---------------------------------------------------------------- R1
def convergence_study(n_paths=20000, vol=0.16, ttm=21 / TRADING_DAYS,
                      steps=(1, 2, 3, 5, 7, 10, 21, 42, 84), spot=25000.0):
    """std of the hedge error against the number of rebalances, on GBM."""
    rng = np.random.default_rng(RNG_SEED)
    strike = spot
    rows = []
    for n in steps:
        dt = ttm / n
        z = rng.standard_normal((n_paths, n))
        incr = (RATE - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z
        paths = spot * np.exp(np.cumsum(incr, axis=1))
        paths = np.column_stack([np.full(n_paths, spot), paths])
        grid = ttm - np.arange(n + 1) * dt
        grid[-1] = 0.0
        err = np.array([hedge_once(p, strike, RATE, vol, vol, grid) for p in paths])
        prem = float(call_price(spot, strike, RATE, vol, ttm))
        rows.append({"rebalances": n, "mean": err.mean(), "std": err.std(ddof=1),
                     "std_pct_premium": 100 * err.std(ddof=1) / prem})
    out = pd.DataFrame(rows)
    slope, intercept = np.polyfit(np.log(out["rebalances"]), np.log(out["std"]), 1)
    return out, slope, prem


# ---------------------------------------------------------------- R2
def backtest(frame, horizon=HORIZON, hedge_every=1, rate=RATE, cost_bps=0.0):
    """
    Non-overlapping cycles. At each inception sell a 1-month ATM call priced at
    that day's India VIX, then delta-hedge at the same vol until expiry.
    """
    spot, iv = frame["spot"].values, frame["iv"].values
    dates = frame.index
    rows = []
    for a in range(0, len(frame) - horizon, horizon):
        b = a + horizon
        sl = slice(a, b + 1)
        path_full, ttm_full = spot[sl], (horizon - np.arange(horizon + 1)) / TRADING_DAYS
        keep = sorted(set(list(range(0, horizon, hedge_every)) + [horizon]))
        path, ttm = path_full[keep], ttm_full[keep]
        ttm[-1] = 0.0
        strike = round(path_full[0] / 50) * 50          # nearest listed strike
        sigma_i = iv[a]
        pnl = hedge_once(path, strike, rate, sigma_i, sigma_i, ttm, cost_bps)
        rr = np.diff(np.log(path_full))
        sigma_r = rr.std(ddof=1) * np.sqrt(TRADING_DAYS)
        prem = float(call_price(path_full[0], strike, rate, sigma_i, ttm_full[0]))
        rows.append({"start": dates[a], "expiry": dates[b], "spot0": path_full[0],
                     "strike": strike, "implied": sigma_i, "realised": sigma_r,
                     "premium": prem, "pnl": pnl, "pnl_pct_premium": 100 * pnl / prem})
    return pd.DataFrame(rows)


def ito_prediction(frame, horizon=HORIZON, rate=RATE):
    """
    The Ito integral 1/2 * sum e^{r(T-t)} S^2 Gamma (sigma_i^2 - sigma_r^2) dt,
    evaluated on the realised path. Compared against the backtest P&L.
    """
    spot, iv = frame["spot"].values, frame["iv"].values
    out = []
    for a in range(0, len(frame) - horizon, horizon):
        b = a + horizon
        path = spot[a:b + 1]
        strike = round(path[0] / 50) * 50
        sigma_i = iv[a]
        dt = 1 / TRADING_DAYS
        total = 0.0
        for i in range(horizon):
            ttm = (horizon - i) / TRADING_DAYS
            g = float(call_gamma(path[i], strike, rate, sigma_i, ttm))
            inst_var = (np.log(path[i + 1] / path[i]) ** 2) / dt
            total += 0.5 * np.exp(rate * ttm) * path[i] ** 2 * g * (sigma_i ** 2 - inst_var) * dt
        out.append(total)
    return np.array(out)


if __name__ == "__main__":
    pd.set_option("display.width", 100)
    frame = nifty_and_vix()

    print("=" * 74)
    print("R1  Discrete hedge error vs rebalance count (simulated GBM, sigma known)")
    print("=" * 74)
    conv, slope, prem = convergence_study()
    print(conv.to_string(index=False, float_format=lambda x: f"{x:10.4f}"))
    print(f"\n  one-month ATM premium          : {prem:,.2f} index points")
    print(f"  log-log slope of std vs n      : {slope:+.4f}   (Boyle-Emanuel: -0.5)")

    print()
    print("=" * 74)
    print("R2  Backtest: short 1M ATM NIFTY call hedged at India VIX, 2015-2026")
    print("=" * 74)
    for every, label in ((1, "daily"), (3, "every 3 days"), (7, "weekly")):
        bt = backtest(frame, hedge_every=every)
        wins = (bt["pnl"] > 0).mean() * 100
        print(f"  {label:<13} cycles={len(bt):3d}  mean P&L={bt['pnl'].mean():+8.2f} pts "
              f"({bt['pnl_pct_premium'].mean():+6.2f}% of premium)  "
              f"std={bt['pnl'].std():7.2f}  win={wins:5.1f}%")

    bt = backtest(frame, hedge_every=1)
    gap = (bt["implied"] - bt["realised"]) * 100
    print(f"\n  mean implied  : {bt['implied'].mean()*100:5.2f}%")
    print(f"  mean realised : {bt['realised'].mean()*100:5.2f}%")
    print(f"  mean gap      : {gap.mean():+5.2f} vol points   (implied above realised "
          f"in {(gap>0).mean()*100:.1f}% of cycles)")

    pred = ito_prediction(frame)
    corr = np.corrcoef(pred, bt["pnl"].values)[0, 1]
    print(f"\n  Ito prediction vs realised P&L : corr = {corr:+.4f}")
    print(f"    mean predicted {pred.mean():+8.2f} pts   mean actual {bt['pnl'].mean():+8.2f} pts")

    bt.to_csv("data/backtest_cycles.csv", index=False)
    conv.to_csv("data/convergence.csv", index=False)
    print("\n  wrote data/backtest_cycles.csv, data/convergence.csv")
