"""Does the deep hedger's edge grow when the market breaks the BS assumptions?

Three markets at fixed cost=1%: GBM (BS is correct), Merton jump-diffusion (gap
risk a continuous delta cannot hedge), Heston stochastic vol (latent vol the
constant-vol delta mis-specifies). Both hedgers get the SAME MC-fair premium, so
the comparison isolates hedging quality from pricing. The BS delta uses its
(mis-specified) constant vol; the deep hedger learns directly from each market's
paths.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.baselines.bs import bs_delta_hedge_pnl
from src.env.market import (cvar, hedging_pnl, mc_premium, simulate_gbm,
                            simulate_heston, simulate_merton, summary)
from src.training.train import policy_holdings, train_deep_hedger

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
MET = ROOT / "results" / "metrics"

S0, K, T, r, N_STEPS, COST = 100.0, 100.0, 1.0, 0.0, 50, 0.01
SIG = 0.2  # diffusion / assumed constant vol (Heston theta=0.04 -> sqrt=0.2 too)


def markets():
    return {
        "GBM": lambda n, g: simulate_gbm(S0, r, SIG, T, N_STEPS, n, g),
        "Merton-jump": lambda n, g: simulate_merton(S0, r, SIG, T, N_STEPS, n, g,
                                                    lam=1.0, jump_mean=-0.10, jump_std=0.15),
        "Heston-stochvol": lambda n, g: simulate_heston(S0, r, T, N_STEPS, n, g,
                                                        v0=0.04, kappa=2.0, theta=0.04,
                                                        xi=0.4, rho=-0.7),
    }


def main():
    rows = []
    for name, sampler in markets().items():
        print(f"\n=== {name} ===")
        prem = mc_premium(sampler(200000, np.random.default_rng(7)), K)  # MC fair value
        print(f"  MC fair premium = {prem:.4f}  (BS-GBM ref ~7.97)")
        net, _ = train_deep_hedger(
            S0=S0, K=K, T=T, r=r, sigma=SIG, n_steps=N_STEPS, cost_rate=COST,
            epochs=400, batch=8192, seed=0, path_sampler=sampler, premium=prem,
        )
        test = sampler(50000, np.random.default_rng(123))
        bs_pnl, _, _ = bs_delta_hedge_pnl(test, K, T, r, SIG, cost_rate=COST, premium=prem)
        dh_pnl = hedging_pnl(test, policy_holdings(net, test, K, T), K, prem, COST)
        bs_c, dh_c = cvar(bs_pnl), cvar(dh_pnl)
        rows.append(dict(market=name, premium=prem, bs=summary(bs_pnl), deep=summary(dh_pnl),
                         bs_cvar=bs_c, deep_cvar=dh_c, improve_pct=100 * (bs_c - dh_c) / bs_c))
        print(f"  BS delta  CVaR={bs_c:.3f}   Deep CVaR={dh_c:.3f}   "
              f"deep improves CVaR by {100*(bs_c-dh_c)/bs_c:.1f}%")

    _plot(rows)
    (MET / "markets.json").write_text(json.dumps(rows, indent=2))
    print(f"\nSaved -> {MET}/markets.json, {FIG}/markets_cvar.png")
    return rows


def _plot(rows):
    names = [r_["market"] for r_ in rows]
    bs = [r_["bs_cvar"] for r_ in rows]
    dp = [r_["deep_cvar"] for r_ in rows]
    x = np.arange(len(names)); w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, bs, w, label="BS delta", color="tab:blue")
    ax.bar(x + w / 2, dp, w, label="Deep hedge", color="tab:red")
    for i, r_ in enumerate(rows):
        ax.text(i, max(bs[i], dp[i]) + 0.1, f"-{r_['improve_pct']:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_title("Hedging tail risk by market (cost=1%) — deep edge grows as BS assumptions break")
    ax.set_ylabel("CVaR95 (lower = better)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "markets_cvar.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
