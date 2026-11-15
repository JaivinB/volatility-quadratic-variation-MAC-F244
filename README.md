# Volatility as a quadratic variation

Measuring it, forecasting it, and pricing it on the NIFTY 50.

**MAC F244 — Stochastic Calculus and Applications to Finance**
BITS Pilani, K. K. Birla Goa Campus · Instructor: Prof. Mayank Goel · Submitted 15 November 2026

---

## What this project is

Itô's lemma applied to `f(S) = log S` turns an unobservable model parameter into an
observable property of a price path:

```
d(log S) = (μ − ½σ²) dt + σ dW      ⟹      [log S]_T = ∫₀ᵀ σ²_t dt
```

The quadratic variation of the log price *is* the integrated variance. Its sample
version — realised variance, `RV_n = Σ (Δ log S)²` — is therefore an estimator of
something real, not a calibration knob. The project takes that identity and pushes it
through three stages on eleven years of index data:

1. **Measure.** Verify that `RV_n → IV` at the rate and with the constant Itô predicts,
   then measure it on the NIFTY 50 and characterise what the measurements look like
   (clustering, leverage, fat tails, jumps).
2. **Forecast.** Test whether India VIX — a model-free implied volatility — is an
   unbiased forecast of the realised volatility that follows (Mincer–Zarnowitz), and
   measure the gap it leaves.
3. **Price.** Show that the gap is harvestable, and that the P&L of harvesting it is the
   one Itô says it should be:

```
Π_T = ½ ∫₀ᵀ e^{r(T−t)} S²_t Γ_t (σ²_implied − σ²_t) dt
```

The last step is the project's main check. The Itô integral and the hedging P&L are
computed by two independent routes — quadrature along the path versus a self-financing
accounting simulation — and compared. Deriving one from the other would make the
agreement circular.

## Headline results

| | |
|---|---|
| Data | 2,903 trading days, 2 Jan 2015 → 12 Nov 2026, zero NaNs |
| `RV_n` convergence slope | **−0.4986** (Itô predicts −0.5); sd/theory ratio 0.991–1.021 over a 128× range of `n` |
| Ljung–Box Q(10) | 1752.1 on squared returns vs 59.3 on returns (crit. 23.2) — clustering, not predictability |
| Leverage correlation | −0.1996 |
| Mincer–Zarnowitz | a = 0.0310 (0.0163), b = 0.6637 (0.0909), R² = 0.282, t(b=1) = −3.70 |
| Variance risk premium | +2.50 vol points, t = 4.042, positive in 76.8% of 138 months |
| **Itô prediction vs realised P&L** | **corr +0.9493** (+39.21 predicted vs +36.06 actual) |
| Backtest | +36.06 pts/cycle = +11.31% of premium, 74.6% win rate, t = 4.80, p = 4.1e−06 |
| Jump test (BNS) | jump share 5.60%, Z = +1.617 — null of no jumps **not** rejected |
| Transaction costs | +32.4 @ 1 bp (t = 4.30), +28.8 @ 2 bps, +17.8 @ 5 bps, break-even ≈ 10 bps |
| Sub-periods | 2015–19 +24.0 (t = 3.22) · COVID +56.9 (t = 2.91) · 2022–26 +40.2 (t = 2.74) |

Every number above is printed by the scripts in this repository and none is hand-copied
between stages.

## Documents

| File | |
|---|---|
| `SC_Report.pdf` | The 4-page submitted report (hard limit per the spec). |
| `SC_Slides.pdf` | 16-frame presentation for the 15-20 minute viva. |
| `SC_Explained.pdf` | 22-page companion: where the idea came from, the theory with its class-hours provenance, exactly which numerical methods were and were not used, real-world uses, a per-figure walkthrough, the viva defence, and an honest account of what is weak. |

## Code

| File | |
|---|---|
| `sc_data.py` | Pulls `^NSEI` and `^INDIAVIX` from Yahoo Finance, caches to `data/`, and refuses to proceed on a failed sanity check (row count, date monotonicity, duplicates, non-positive prices, high < low, implied vol outside [0.02, 1.50], any calendar hole over 15 days). |
| `sc_model.py` | Black–Scholes price, delta, gamma — plus `check_pde()` and `check_greeks()`, which verify the closed forms against the PDE residual and against central finite differences. |
| `sc_vol.py` | Realised variance, the convergence experiment (exact log-solution of GBM, so zero discretisation error), autocorrelations, Ljung–Box, hand-written OLS, implied-vs-realised. |
| `sc_hedge.py` | The self-financing hedging simulation and the historical backtest, with optional proportional transaction costs in basis points. |
| `sc_robust.py` | Bipower variation, tripower quarticity, the BNS/Huang–Tauchen jump test, the cost sweep, and the sub-period split. |
| `sc_figures.py` | All six figures used by the documents, read back from the CSVs rather than recomputed. |

Each file runs standalone and prints its own results.

## Reproducing everything

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python sc_data.py       # fetch + validate (cached CSVs are committed, so this is optional)
python sc_model.py      # closed-form verification
python sc_vol.py        # convergence, diagnostics, implied vs realised
python sc_hedge.py      # hedging simulation + backtest
python sc_robust.py     # jumps, costs, sub-periods
python sc_figures.py    # figures

latexmk -pdf SC_Report.tex
latexmk -pdf SC_Explained.tex
```

Monte Carlo uses a fixed seed, so the convergence and simulation numbers are
bit-reproducible. `data/` holds the cached market data, so the whole pipeline reruns
offline.

## A note on the model's age

Black–Scholes is from 1973 and the index data is not lognormal — which is the point
rather than an oversight. The P&L identity above is derived from self-financing
accounting plus Itô's lemma and the Black–Scholes PDE; it does not require the
underlying to actually be a geometric Brownian motion, because `σ_t` enters it as the
*realised* volatility of whatever path occurred. That is why the identity holds at
corr 0.95 on real data with jumps, clustering and fat tails. `SC_Explained.pdf` works
through this in full.
