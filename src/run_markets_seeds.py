"""Multi-seed version of the market experiment, so the cross-market ordering is
stated with seed bands rather than a single lucky draw. BS delta is deterministic
per market; the deep hedger is retrained under 3 seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.baselines.bs import bs_delta_hedge_pnl
from src.env.market import cvar, hedging_pnl, mc_premium
from src.run_markets import K, N_STEPS, S0, SIG, T, COST, markets, r
from src.training.train import policy_holdings, train_deep_hedger

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
MET = ROOT / "results" / "metrics"
SEEDS = [0, 1, 2]


def main():
    rows = []
    for name, sampler in markets().items():
        prem = mc_premium(sampler(200000, np.random.default_rng(7)), K)
        test = sampler(50000, np.random.default_rng(123))
        bs_pnl, _, _ = bs_delta_hedge_pnl(test, K, T, r, SIG, cost_rate=COST, premium=prem)
        bs_c = cvar(bs_pnl)
        deep_c, impr = [], []
        for sd in SEEDS:
            net, _ = train_deep_hedger(
                S0=S0, K=K, T=T, r=r, sigma=SIG, n_steps=N_STEPS, cost_rate=COST,
                epochs=400, batch=8192, seed=sd, path_sampler=sampler, premium=prem,
            )
            c = cvar(hedging_pnl(test, policy_holdings(net, test, K, T), K, prem, COST))
            deep_c.append(c)
            impr.append(100 * (bs_c - c) / bs_c)
        impr = np.array(impr)
        row = dict(market=name, premium=prem, bs_cvar=bs_c,
                   deep_cvar_mean=float(np.mean(deep_c)), deep_cvar_std=float(np.std(deep_c)),
                   improve_mean=float(impr.mean()), improve_std=float(impr.std()),
                   improve_min=float(impr.min()), improve_max=float(impr.max()))
        rows.append(row)
        print(f"{name:16s} BS={bs_c:6.3f}  Deep={np.mean(deep_c):6.3f}  "
              f"improve={impr.mean():.1f}%+/-{impr.std():.1f}% [{impr.min():.1f},{impr.max():.1f}]")

    _plot(rows)
    (MET / "markets_seeds.json").write_text(json.dumps(rows, indent=2))
    print(f"\nSaved -> {MET}/markets_seeds.json, {FIG}/markets_cvar_seeds.png")
    return rows


def _plot(rows):
    names = [r_["market"] for r_ in rows]
    im = [r_["improve_mean"] for r_ in rows]
    er = [r_["improve_std"] for r_ in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, im, yerr=er, capsize=5, color=["tab:gray", "tab:red", "tab:orange"])
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_title(f"Deep-hedge CVaR improvement over BS delta by market "
                 f"(cost={COST*100:.0f}%, {len(SEEDS)} seeds)")
    ax.set_ylabel("CVaR improvement vs BS delta (%)")
    fig.tight_layout(); fig.savefig(FIG / "markets_cvar_seeds.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
