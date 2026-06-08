"""Multi-seed robustness of the cost sweep.

BS delta is deterministic (no training), so it is evaluated once per cost. The
deep hedger is retrained under several seeds; we report its CVaR mean +/- std so
the reader can see whether the high-cost win is a seed artifact or a real effect.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.baselines.bs import bs_delta_hedge_pnl, bs_price
from src.env.market import cvar, hedging_pnl, simulate_gbm
from src.training.train import policy_holdings, train_deep_hedger

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
MET = ROOT / "results" / "metrics"

S0, K, T, r, SIG, N_STEPS = 100.0, 100.0, 1.0, 0.0, 0.2, 50
COSTS = [0.0, 0.001, 0.005, 0.01, 0.02]
SEEDS = [0, 1, 2]
EPOCHS, BATCH = 400, 8192


def main():
    premium = float(bs_price(S0, K, T, r, SIG, call=True))
    rng = np.random.default_rng(123)
    test_paths = simulate_gbm(S0, r, SIG, T, N_STEPS, 50000, rng)

    rows = []
    for c in COSTS:
        bs_pnl, _, _ = bs_delta_hedge_pnl(test_paths, K, T, r, SIG, cost_rate=c, premium=premium)
        bs_cvar = cvar(bs_pnl)
        deep_cvars = []
        for sd in SEEDS:
            net, _ = train_deep_hedger(
                S0=S0, K=K, T=T, r=r, sigma=SIG, n_steps=N_STEPS,
                cost_rate=c, epochs=EPOCHS, batch=BATCH, seed=sd,
            )
            dh_pnl = hedging_pnl(test_paths, policy_holdings(net, test_paths, K, T), K, premium, c)
            deep_cvars.append(cvar(dh_pnl))
        dc = np.array(deep_cvars)
        row = dict(cost=c, bs_cvar=bs_cvar, deep_mean=float(dc.mean()),
                   deep_std=float(dc.std()), deep_min=float(dc.min()),
                   deep_max=float(dc.max()), deep_all=deep_cvars,
                   deep_wins_all=bool((dc < bs_cvar).all()))
        rows.append(row)
        print(f"cost={c:.3f}  BS={bs_cvar:.3f}  Deep={dc.mean():.3f}+/-{dc.std():.3f} "
              f"[{dc.min():.3f},{dc.max():.3f}]  deep_wins_all_seeds={row['deep_wins_all']}")

    _plot(rows)
    (MET / "sweep_seeds.json").write_text(json.dumps(rows, indent=2))
    print(f"\nSaved -> {MET}/sweep_seeds.json, {FIG}/cvar_vs_cost_seeds.png")
    return rows


def _plot(rows):
    costs = [r_["cost"] * 100 for r_ in rows]
    bs = [r_["bs_cvar"] for r_ in rows]
    dm = [r_["deep_mean"] for r_ in rows]
    ds = [r_["deep_std"] for r_ in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(costs, bs, "o-", color="tab:blue", label="BS delta (deterministic)")
    ax.errorbar(costs, dm, yerr=ds, fmt="s-", color="tab:red", capsize=4,
                label=f"Deep hedge (mean+/-std, {len(SEEDS)} seeds)")
    ax.set_title("Tail risk vs cost — multi-seed robustness")
    ax.set_xlabel("transaction cost (% of notional)")
    ax.set_ylabel("CVaR95 (lower = better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "cvar_vs_cost_seeds.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
