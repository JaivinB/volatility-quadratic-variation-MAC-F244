"""
Volatility as an Ito object: quadratic variation, realised variance, and the
implied-realised gap on the NIFTY 50.

The course fact is [W]_T = T. Carried through Ito's formula, it says that for
dS = mu S dt + sigma(t) S dW the quadratic variation of log S is

        [log S]_T  =  integral_0^T sigma(t)^2 dt        ("integrated variance")

so the realised variance  RV_n = sum (delta log S)^2  is not a statistic bolted
onto the model: it is the sample version of an Ito object. That identity is what
makes volatility measurable at all, and it is the spine of this project.

  V1  Does RV_n actually converge to integrated variance, at the rate theory
      predicts?  Tested on simulated paths with sigma(t) KNOWN and time-varying,
      against the closed-form asymptotic standard error
            sd(RV_n - IV) = sqrt( (2T/n) * integral sigma^4 dt ).
  V2  What does NIFTY volatility look like once measured this way: clustering,
      the leverage effect, vol-of-vol.  I.e. how GBM's constant sigma fails.
  V3  Is India VIX an unbiased forecast of the realised volatility that follows?
      Mincer-Zarnowitz regression on non-overlapping months.
  V4  The variance risk premium in volatility-point and variance terms.
"""
import numpy as np
import pandas as pd

from sc_data import nifty_and_vix, log_returns, TRADING_DAYS

RNG_SEED = 2026
HORIZON = 21


# ------------------------------------------------------------------ V1
def sigma_path(t, base=0.15, amp=0.10, period=1.0):
    """A deterministic, time-varying volatility. Known, so IV is computable."""
    return base + amp * np.sin(2 * np.pi * t / period)


def integrated_moments(T=1.0, power=2, nquad=200_000):
    """integral_0^T sigma(t)^power dt by fine quadrature (trapezoid)."""
    t = np.linspace(0.0, T, nquad + 1)
    return float(np.trapezoid(sigma_path(t) ** power, t))


def rv_convergence(T=1.0, mu=0.08, s0=25_000.0, n_paths=4000,
                   steps=(21, 42, 84, 168, 336, 672, 1344, 2688)):
    """
    Realised variance against the number of sampling intervals.

    Reports the measured bias and spread of RV_n - IV, and the spread theory
    predicts, so the check is of a constant and not merely of a rate.
    """
    rng = np.random.default_rng(RNG_SEED)
    iv = integrated_moments(T, 2)
    quartic = integrated_moments(T, 4)
    rows = []
    for n in steps:
        dt = T / n
        grid = np.arange(n) * dt                      # left endpoints
        sig = sigma_path(grid + 0.5 * dt)             # midpoint, 2nd-order
        z = rng.standard_normal((n_paths, n))
        dlog = (mu - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * z
        rv = (dlog**2).sum(axis=1)
        err = rv - iv
        rows.append({
            "n": n,
            "rv_mean": rv.mean(),
            "bias": err.mean(),
            "bias_pct_iv": 100 * err.mean() / iv,
            "sd_measured": err.std(ddof=1),
            "sd_theory": np.sqrt(2 * T * quartic / n),
        })
    out = pd.DataFrame(rows)
    out["sd_ratio"] = out["sd_measured"] / out["sd_theory"]
    slope = np.polyfit(np.log(out["n"]), np.log(out["sd_measured"]), 1)[0]
    return out, slope, iv, quartic


# ------------------------------------------------------------------ V2
def realised_variance(spot, window=HORIZON, annualise=True):
    """Rolling realised variance from squared log returns (the QV estimator)."""
    r2 = log_returns(spot) ** 2
    rv = r2.rolling(window).sum()
    if annualise:
        rv = rv * (TRADING_DAYS / window)
    return rv.dropna()


def autocorr(x, lags):
    x = np.asarray(x, float)
    x = x - x.mean()
    denom = (x * x).sum()
    return np.array([float((x[:-k] * x[k:]).sum() / denom) for k in lags])


def vol_diagnostics(frame, lags=(1, 2, 3, 5, 10, 21, 42, 63)):
    """Clustering, the leverage effect, and vol-of-vol on the real index."""
    r = log_returns(frame["spot"])
    lags = list(lags)
    rv = realised_variance(frame["spot"])
    vol = np.sqrt(rv)

    # leverage: today's return against the change in realised vol around it
    dvol = vol.diff().dropna()
    common = r.index.intersection(dvol.index)
    leverage = float(np.corrcoef(r.loc[common], dvol.loc[common])[0, 1])

    lv = np.log(vol.dropna())
    return {
        "acf_returns": autocorr(r.values, lags),
        "acf_abs": autocorr(np.abs(r.values), lags),
        "acf_sq": autocorr((r.values - r.values.mean()) ** 2, lags),
        "lags": lags,
        "leverage_corr": leverage,
        "vol_mean": float(vol.mean()),
        "vol_sd": float(vol.std(ddof=1)),
        "vol_min": float(vol.min()),
        "vol_max": float(vol.max()),
        "vol_of_vol": float(lv.diff().std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "ljung_box_sq": ljung_box((r.values - r.values.mean()) ** 2, 10),
        "ljung_box_ret": ljung_box(r.values, 10),
    }


def ljung_box(x, m):
    """Ljung-Box Q on the first m autocorrelations; chi2(m) under no structure."""
    n = len(x)
    rho = autocorr(x, range(1, m + 1))
    q = n * (n + 2) * np.sum(rho**2 / (n - np.arange(1, m + 1)))
    return float(q)


# ------------------------------------------------------------------ V3
def ols(y, X, names):
    """OLS with classical standard errors. X must already carry a constant."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    s2 = float(resid @ resid) / (n - k)
    se = np.sqrt(np.diag(s2 * xtx_inv))
    tss = float(((y - y.mean()) ** 2).sum())
    return {
        "names": names, "beta": beta, "se": se, "t": beta / se,
        "r2": 1 - float(resid @ resid) / tss, "n": n,
        "sigma": np.sqrt(s2),
    }


def implied_vs_realised(frame, horizon=HORIZON):
    """
    Non-overlapping months. Pair the India VIX quoted on day t with the realised
    volatility over the NEXT `horizon` trading days, then ask whether implied is
    an unbiased forecast:   RV_{t+h} = a + b * IV_t + e,   unbiased => a=0, b=1.
    """
    spot, iv = frame["spot"].values, frame["iv"].values
    dates = frame.index
    rows = []
    for a in range(0, len(frame) - horizon, horizon):
        b = a + horizon
        seg = np.diff(np.log(spot[a:b + 1]))
        rv_ann = np.sqrt((seg**2).sum() * TRADING_DAYS / horizon)
        rows.append({"start": dates[a], "expiry": dates[b],
                     "implied": iv[a], "realised": rv_ann,
                     "gap_vol_pts": 100 * (iv[a] - rv_ann),
                     "gap_var": iv[a] ** 2 - rv_ann ** 2})
    out = pd.DataFrame(rows)
    X = np.column_stack([np.ones(len(out)), out["implied"].values])
    fit = ols(out["realised"].values, X, ["intercept", "implied"])
    return out, fit


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    frame = nifty_and_vix()

    print("=" * 78)
    print("V1  Realised variance -> integrated variance  (sigma(t) known)")
    print("=" * 78)
    conv, slope, iv, quartic = rv_convergence()
    print(f"  integral sigma^2 dt = {iv:.6f}      integral sigma^4 dt = {quartic:.6f}")
    print(conv.to_string(index=False, float_format=lambda x: f"{x:12.6f}"))
    print(f"\n  log-log slope of sd(RV_n - IV) vs n : {slope:+.4f}   (theory -0.5)")
    print(f"  measured / theoretical sd           : "
          f"{conv['sd_ratio'].min():.3f} to {conv['sd_ratio'].max():.3f}")

    print()
    print("=" * 78)
    print("V2  NIFTY volatility, measured as quadratic variation")
    print("=" * 78)
    d = vol_diagnostics(frame)
    print(f"  21-day realised vol : mean {d['vol_mean']*100:5.2f}%   "
          f"sd {d['vol_sd']*100:5.2f}%   range {d['vol_min']*100:.2f}%"
          f"-{d['vol_max']*100:.2f}%")
    print(f"  vol of vol          : {d['vol_of_vol']*100:.1f}% per year (log scale)")
    print(f"  leverage corr(r, d vol) : {d['leverage_corr']:+.4f}")
    print(f"  Ljung-Box Q(10) returns  : {d['ljung_box_ret']:8.1f}")
    print(f"  Ljung-Box Q(10) squared  : {d['ljung_box_sq']:8.1f}   (chi2(10) 1% = 23.2)")
    print("\n  lag      acf(r)   acf(|r|)   acf(r^2)")
    for i, L in enumerate(d["lags"]):
        print(f"  {L:3d}   {d['acf_returns'][i]:+8.4f}   "
              f"{d['acf_abs'][i]:+8.4f}   {d['acf_sq'][i]:+8.4f}")

    print()
    print("=" * 78)
    print("V3  Is India VIX an unbiased forecast of the realised vol that follows?")
    print("=" * 78)
    pairs, fit = implied_vs_realised(frame)
    print(f"  non-overlapping months : {fit['n']}")
    for nm, b, se, t in zip(fit["names"], fit["beta"], fit["se"], fit["t"]):
        print(f"    {nm:<10} {b:+9.4f}   se {se:.4f}   t {t:+7.2f}")
    b1, se1 = fit["beta"][1], fit["se"][1]
    print(f"  R^2 = {fit['r2']:.4f}")
    print(f"  test b = 1 :  t = {(b1 - 1)/se1:+.2f}")
    print(f"  test a = 0 :  t = {fit['t'][0]:+.2f}")

    print()
    print("=" * 78)
    print("V4  The variance risk premium")
    print("=" * 78)
    g = pairs["gap_vol_pts"]
    print(f"  mean implied           : {pairs['implied'].mean()*100:5.2f}%")
    print(f"  mean realised (next 1M): {pairs['realised'].mean()*100:5.2f}%")
    print(f"  mean gap               : {g.mean():+5.2f} vol points")
    print(f"  implied > realised in  : {(g > 0).mean()*100:5.1f}% of months")
    t_gap = g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))
    print(f"  t-statistic on the gap : {t_gap:+.3f}")
    print(f"  mean variance gap      : {pairs['gap_var'].mean():+.6f}")

    conv.to_csv("data/rv_convergence.csv", index=False)
    pairs.to_csv("data/implied_vs_realised.csv", index=False)
    print("\n  wrote data/rv_convergence.csv, data/implied_vs_realised.csv")
