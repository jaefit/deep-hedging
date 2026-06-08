"""Transaction-cost sweep: does cost-aware deep hedging beat naive BS delta?

For each cost level we train a deep hedger that SEES that cost in its objective,
then score both it and the (cost-blind) BS delta hedge on a common set of test
paths. Expectation: at zero cost BS delta is optimal and the NN only matches it;
as cost rises the NN learns to trade less / smarter and wins on CVaR.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.baselines.bs import bs_delta_hedge_pnl, bs_price
from src.env.market import hedging_pnl, simulate_gbm, summary
from src.training.train import policy_holdings, train_deep_hedger

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
MET = ROOT / "results" / "metrics"
FIG.mkdir(parents=True, exist_ok=True)
MET.mkdir(parents=True, exist_ok=True)

S0, K, T, r, SIG, N_STEPS = 100.0, 100.0, 1.0, 0.0, 0.2, 50
COSTS = [0.0, 0.001, 0.005, 0.01, 0.02]
EPOCHS, BATCH = 400, 8192  # validated config: top-k ES converges, deep beats BS at high cost


def main():
    premium = float(bs_price(S0, K, T, r, SIG, call=True))
    rng = np.random.default_rng(123)
    test_paths = simulate_gbm(S0, r, SIG, T, N_STEPS, 50000, rng)  # shared test set

    rows = []
    pnl_store = {}  # for the high-cost histogram
    for c in COSTS:
        print(f"\n=== cost = {c} ===")
        net, _ = train_deep_hedger(
            S0=S0, K=K, T=T, r=r, sigma=SIG, n_steps=N_STEPS,
            cost_rate=c, epochs=EPOCHS, batch=BATCH, seed=0,
        )
        bs_pnl, _, _ = bs_delta_hedge_pnl(test_paths, K, T, r, SIG, cost_rate=c, premium=premium)
        dh_hold = policy_holdings(net, test_paths, K, T)
        dh_pnl = hedging_pnl(test_paths, dh_hold, K, premium, cost_rate=c)
        for name, pnl in [("BS_delta", bs_pnl), ("DeepHedge", dh_pnl)]:
            s = summary(pnl)
            s.update(cost=c, model=name)
            rows.append(s)
            print(f"  {name:10s} mean={s['mean']:.3f} std={s['std']:.3f} CVaR95={s['cvar95']:.3f}")
        pnl_store[c] = (bs_pnl, dh_pnl)

    _plot_cvar(rows)
    _plot_hist(pnl_store, cost=0.02)
    (MET / "sweep_metrics.json").write_text(json.dumps(rows, indent=2))
    _write_csv(rows)
    print(f"\nSaved -> {MET}, {FIG}")
    return rows


def _write_csv(rows):
    import csv

    keys = ["cost", "model", "mean", "std", "cvar95", "q05"]
    with open(MET / "sweep_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r_ in rows:
            w.writerow({k: r_[k] for k in keys})


def _plot_cvar(rows):
    costs = sorted({r_["cost"] for r_ in rows})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, color in [("BS_delta", "tab:blue"), ("DeepHedge", "tab:red")]:
        cv = [next(r_["cvar95"] for r_ in rows if r_["cost"] == c and r_["model"] == name) for c in costs]
        mn = [next(r_["mean"] for r_ in rows if r_["cost"] == c and r_["model"] == name) for c in costs]
        ax[0].plot([c * 100 for c in costs], cv, "o-", color=color, label=name)
        ax[1].plot([c * 100 for c in costs], mn, "o-", color=color, label=name)
    ax[0].set_title("Tail risk: CVaR95 of P&L vs transaction cost")
    ax[0].set_xlabel("cost (% of notional)"); ax[0].set_ylabel("CVaR95 (lower=better)"); ax[0].legend()
    ax[1].set_title("Mean P&L vs transaction cost")
    ax[1].set_xlabel("cost (% of notional)"); ax[1].set_ylabel("mean P&L"); ax[1].legend()
    fig.tight_layout(); fig.savefig(FIG / "cvar_vs_cost.png", dpi=120); plt.close(fig)


def _plot_hist(pnl_store, cost):
    bs_pnl, dh_pnl = pnl_store[cost]
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(min(bs_pnl.min(), dh_pnl.min()), max(bs_pnl.max(), dh_pnl.max()), 80)
    ax.hist(bs_pnl, bins=bins, alpha=0.5, label="BS delta", color="tab:blue", density=True)
    ax.hist(dh_pnl, bins=bins, alpha=0.5, label="Deep Hedge", color="tab:red", density=True)
    ax.set_title(f"Hedged-book P&L distribution at cost={cost*100:.1f}% "
                 "(deep hedge shifts the left tail right)")
    ax.set_xlabel("P&L"); ax.set_ylabel("density"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / f"pnl_hist_cost{int(cost*1000)}.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
