"""
Black-Scholes under geometric Brownian motion, and the Ito identity the hedge
backtest rests on.

Model.  dS_t = mu S_t dt + sigma S_t dW_t.

Ito's lemma applied to V(t, S_t) gives

    dV = (V_t + mu S V_S + 1/2 sigma^2 S^2 V_SS) dt + sigma S V_S dW.

Holding Delta = V_S shares against a short option removes the dW term, so the
hedged portfolio is locally riskless and must earn r. That is the Black-Scholes
PDE,

    V_t + 1/2 sigma^2 S^2 V_SS + r S V_S - r V = 0,

and `check_pde` verifies the closed-form call satisfies it numerically.
"""
import numpy as np
from scipy.stats import norm

SQRT_EPS = 1.4901161193847656e-08


def _d1_d2(spot, strike, rate, vol, ttm):
    spot, strike, vol, ttm = map(np.asarray, (spot, strike, vol, ttm))
    v = vol * np.sqrt(ttm)
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * ttm) / v
    return d1, d1 - v


def call_price(spot, strike, rate, vol, ttm):
    """Black-Scholes European call. At ttm == 0 returns the payoff."""
    spot, strike = np.asarray(spot, float), np.asarray(strike, float)
    ttm = np.asarray(ttm, float)
    out = np.maximum(spot - strike, 0.0) * np.ones_like(spot + ttm)
    live = ttm > 0
    if np.any(live):
        d1, d2 = _d1_d2(spot, strike, rate, vol, np.where(live, ttm, 1.0))
        bs = spot * norm.cdf(d1) - strike * np.exp(-rate * ttm) * norm.cdf(d2)
        out = np.where(live, bs, out)
    return out


def call_delta(spot, strike, rate, vol, ttm):
    """dV/dS. At expiry it is the step function 1{S > K}."""
    ttm = np.asarray(ttm, float)
    out = (np.asarray(spot, float) > np.asarray(strike, float)).astype(float)
    live = ttm > 0
    if np.any(live):
        d1, _ = _d1_d2(spot, strike, rate, vol, np.where(live, ttm, 1.0))
        out = np.where(live, norm.cdf(d1), out)
    return out


def call_gamma(spot, strike, rate, vol, ttm):
    """d2V/dS2. Zero at expiry away from the strike."""
    spot, ttm = np.asarray(spot, float), np.asarray(ttm, float)
    out = np.zeros_like(spot + ttm)
    live = ttm > 0
    if np.any(live):
        d1, _ = _d1_d2(spot, strike, rate, vol, np.where(live, ttm, 1.0))
        g = norm.pdf(d1) / (spot * np.asarray(vol) * np.sqrt(np.where(live, ttm, 1.0)))
        out = np.where(live, g, out)
    return out


def check_pde(spot=25000.0, strike=25000.0, rate=0.065, vol=0.16, ttm=0.25, h=None):
    """
    Residual of the Black-Scholes PDE at one point, by central differences.

    V is written as a function of time-to-maturity tau, so V_t = -dV/dtau.
    Returns (residual, scale) where scale = r*V is the natural size of the
    terms being cancelled, so residual/scale is a relative error.
    """
    hs = h or spot * 1e-4
    ht = h or 1e-5
    v = lambda s, t: float(call_price(s, strike, rate, vol, t))

    v_tau = (v(spot, ttm + ht) - v(spot, ttm - ht)) / (2 * ht)
    v_t = -v_tau
    v_s = (v(spot, ttm + 0) * 0 + v(spot + hs, ttm) - v(spot - hs, ttm)) / (2 * hs)
    v_ss = (v(spot + hs, ttm) - 2 * v(spot, ttm) + v(spot - hs, ttm)) / hs**2

    residual = v_t + 0.5 * vol**2 * spot**2 * v_ss + rate * spot * v_s - rate * v(spot, ttm)
    return residual, rate * v(spot, ttm)


def check_greeks(spot=25000.0, strike=25000.0, rate=0.065, vol=0.16, ttm=0.25):
    """Closed-form delta and gamma against central differences of the price."""
    hs = spot * 1e-4
    v = lambda s: float(call_price(s, strike, rate, vol, ttm))
    fd_delta = (v(spot + hs) - v(spot - hs)) / (2 * hs)
    fd_gamma = (v(spot + hs) - 2 * v(spot) + v(spot - hs)) / hs**2
    cf_delta = float(call_delta(spot, strike, rate, vol, ttm))
    cf_gamma = float(call_gamma(spot, strike, rate, vol, ttm))
    return (cf_delta, fd_delta), (cf_gamma, fd_gamma)


if __name__ == "__main__":
    res, scale = check_pde()
    (cd, fd), (cg, fg) = check_greeks()
    print("Black-Scholes consistency checks at S=K=25000, r=6.5%, sigma=16%, T=0.25")
    print(f"  PDE residual        : {res:+.3e}   (r*V = {scale:.3f}, relative {abs(res)/scale:.2e})")
    print(f"  delta closed/finite : {cd:.10f} / {fd:.10f}   diff {abs(cd-fd):.2e}")
    print(f"  gamma closed/finite : {cg:.3e} / {fg:.3e}   rel  {abs(cg-fg)/cg:.2e}")
    # put-call parity is an independent route to the same price
    from scipy.stats import norm as _n
    S, K, r, s, T = 25000.0, 25000.0, 0.065, 0.16, 0.25
    c = float(call_price(S, K, r, s, T))
    d1, d2 = _d1_d2(S, K, r, s, T)
    p = K * np.exp(-r * T) * _n.cdf(-d2) - S * _n.cdf(-d1)
    print(f"  put-call parity     : C-P = {c - float(p):.6f}  vs  S-Ke^-rT = {S - K*np.exp(-r*T):.6f}")
