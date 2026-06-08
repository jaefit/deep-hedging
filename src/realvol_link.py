"""Wire the volatility forecast into the hedger, then stress the forecast.

Reads `data/vol_input.json` (written by the volatility-forecasting project) and
runs the hedge at a realistic SPY-like vol instead of the toy 0.2. Then the key
experiment — the VOL-MISMATCH STRESS:

    The desk prices and hedges at its forecast sigma_h (HAR's view). The market
    then realizes a DIFFERENT sigma_real. We sweep sigma_real around sigma_h and
    ask which hedger's tail risk degrades more gracefully when the forecast is
    wrong. This is exactly the seam between the two projects: forecast vol there,
    pay for forecast error here.

Both hedgers receive the SAME premium = BS(sigma_h) and both "believe" sigma_h;
only the realized path vol changes.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.baselines.bs import bs_delta_hedge_pnl, bs_price
from src.env.market import cvar, hedging_pnl, simulate_gbm, summary
from src.training.train import policy_holdings, train_deep_hedger

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
MET = ROOT / "results" / "metrics"
VOL_INPUT = ROOT / "data" / "vol_input.json"

S0, K, T, r, N_STEPS = 100.0, 100.0, 1.0, 0.0, 50
COST = 0.01            # representative friction where deep hedging wins when matched
N_TEST = 50000
RATIOS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]  # sigma_real / sigma_h


def main():
    vin = json.loads(VOL_INPUT.read_text())
    sigma_h = float(vin["sigma_forecast"])  # the desk's forecast = pricing/hedging vol
    print(f"Loaded vol input ({vin['ticker']} as of {vin['asof']}): "
          f"sigma_forecast={sigma_h:.4f}, sigma_mean={vin['sigma_mean']:.4f}")

    # premium and BS hedge both use sigma_h; deep hedger is trained on sigma_h dynamics.
    premium = float(bs_price(S0, K, T, r, sigma_h, call=True))
    print(f"\n[A] Hedging at realistic SPY vol sigma_h={sigma_h:.4f} (premium={premium:.4f}); "
          f"training deep hedger (cost={COST}) ...")
    net, _ = train_deep_hedger(
        S0=S0, K=K, T=T, r=r, sigma=sigma_h, n_steps=N_STEPS,
        cost_rate=COST, epochs=400, batch=8192, seed=0,
    )

    rng = np.random.default_rng(123)
    rows = []
    for ratio in RATIOS:
        sigma_real = sigma_h * ratio
        paths = simulate_gbm(S0, r, sigma_real, T, N_STEPS, N_TEST, rng)  # market truth
        bs_pnl, _, _ = bs_delta_hedge_pnl(
            paths, K, T, r, sigma_h, cost_rate=COST, premium=premium  # hedge with sigma_h
        )
        dh_pnl = hedging_pnl(paths, policy_holdings(net, paths, K, T), K, premium, COST)
        row = dict(ratio=ratio, sigma_real=sigma_real,
                   bs=summary(bs_pnl), deep=summary(dh_pnl))
        rows.append(row)
        tag = "MATCHED" if ratio == 1.0 else ""
        print(f"  sigma_real/sigma_h={ratio:>4}  BS CVaR={cvar(bs_pnl):7.3f}  "
              f"Deep CVaR={cvar(dh_pnl):7.3f}  {tag}")

    _plot(rows, sigma_h)
    (MET / "vol_mismatch.json").write_text(json.dumps(rows, indent=2))
    print(f"\nSaved -> {MET}/vol_mismatch.json, {FIG}/vol_mismatch.png")
    return rows


def _plot(rows, sigma_h):
    x = [r_["ratio"] for r_ in rows]
    bs = [r_["bs"]["cvar95"] for r_ in rows]
    dp = [r_["deep"]["cvar95"] for r_ in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, bs, "o-", color="tab:blue", label="BS delta")
    ax.plot(x, dp, "s-", color="tab:red", label="Deep hedge")
    ax.axvline(1.0, color="gray", ls="--", lw=1, label="forecast = realized")
    ax.set_title(f"Vol-mismatch stress (hedge at sigma_h={sigma_h:.3f}, cost={COST*100:.0f}%)\n"
                 "tail risk when realized vol deviates from the forecast")
    ax.set_xlabel("realized vol / forecast vol")
    ax.set_ylabel("CVaR95 (lower = better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "vol_mismatch.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
