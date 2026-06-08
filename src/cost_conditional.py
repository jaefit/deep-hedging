"""One cost-conditional policy vs many cost-specialized policies.

Instead of training a separate hedger per transaction-cost level, train a single
network that takes the cost as an extra input feature. Each training step
evaluates the policy on every cost in the grid (one mini-group per cost), computes
the Expected-Shortfall loss per group, and minimizes their average — so the shared
net must hedge well across the whole cost range. We then check it matches the
per-cost specialists from the multi-seed sweep.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.baselines.bs import bs_price
from src.env.market import cvar, hedging_pnl, simulate_gbm, summary
from src.models.policy import HedgePolicy
from src.training.train import policy_holdings  # not used for CC eval, kept for parity

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
MET = ROOT / "results" / "metrics"

S0, K, T, r, SIG, N_STEPS = 100.0, 100.0, 1.0, 0.0, 0.2, 50
COSTS = [0.0, 0.001, 0.005, 0.01, 0.02]
COST_SCALE = 0.02  # normalize cost feature to ~[0,1]


def _rollout_cc(net, S, K, T, premium, cost):
    """Differentiable P&L with the COST passed as a policy feature (scalar cost)."""
    B, n1 = S.shape
    n_steps = n1 - 1
    dt = T / n_steps
    cfeat = torch.full((B,), cost / COST_SCALE)
    h_prev = torch.zeros(B)
    trading = torch.zeros(B)
    cost_acc = torch.zeros(B)
    for t in range(n_steps):
        tau = (T - t * dt) / T
        feats = torch.stack(
            [torch.log(S[:, t] / K), torch.full((B,), tau), h_prev, cfeat], dim=1
        )
        h = net(feats)
        trading = trading + h * (S[:, t + 1] - S[:, t])
        cost_acc = cost_acc + cost * S[:, t] * torch.abs(h - h_prev)
        h_prev = h
    cost_acc = cost_acc + cost * S[:, n_steps] * torch.abs(h_prev)
    payoff = torch.clamp(S[:, -1] - K, min=0.0)
    return premium + trading - cost_acc - payoff


def train_cost_conditional(epochs=400, batch_g=2048, lr=1e-3, alpha=0.95, seed=0):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    premium = float(bs_price(S0, K, T, r, SIG, call=True))
    net = HedgePolicy(n_features=4)  # +1 for the cost feature
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    k = max(1, int(np.ceil((1.0 - alpha) * batch_g)))
    net.train()
    for ep in range(epochs):
        opt.zero_grad()
        total = 0.0
        for c in COSTS:
            paths = simulate_gbm(S0, r, SIG, T, N_STEPS, batch_g, rng)
            S = torch.from_numpy(paths.astype(np.float32))
            pnl = _rollout_cc(net, S, K, T, premium, c)
            es = torch.topk(-pnl, k).values.mean()
            total = total + es
        loss = total / len(COSTS)  # average ES across the cost grid
        loss.backward()
        opt.step()
        if (ep + 1) % 50 == 0:
            print(f"  ep{ep + 1:4d}  avg-ES={loss.item():.4f}")
    return net, premium


def holdings_cc(net, paths, K, T, cost):
    net.eval()
    B, n1 = paths.shape
    n_steps = n1 - 1
    dt = T / n_steps
    S = torch.from_numpy(paths.astype(np.float32))
    cfeat = torch.full((B,), cost / COST_SCALE)
    h_prev = torch.zeros(B)
    out = np.empty((B, n_steps), dtype=np.float32)
    with torch.no_grad():
        for t in range(n_steps):
            tau = (T - t * dt) / T
            feats = torch.stack(
                [torch.log(S[:, t] / K), torch.full((B,), tau), h_prev, cfeat], dim=1
            )
            h = net(feats)
            out[:, t] = h.numpy()
            h_prev = h
    return out


def main():
    print("training single cost-conditional policy across all costs ...")
    net, premium = train_cost_conditional()

    # specialized baselines (deep, mean CVaR over seeds) from the multi-seed sweep
    spec = {}
    sp = MET / "sweep_seeds.json"
    if sp.exists():
        for row in json.loads(sp.read_text()):
            spec[row["cost"]] = row["deep_mean"]

    rng = np.random.default_rng(123)
    test = simulate_gbm(S0, r, SIG, T, N_STEPS, 50000, rng)
    rows = []
    for c in COSTS:
        pnl = hedging_pnl(test, holdings_cc(net, test, K, T, c), K, premium, c)
        cc_cvar = cvar(pnl)
        rows.append(dict(cost=c, cc_cvar=cc_cvar, specialized_cvar=spec.get(c)))
        s = f"  cost={c:.3f}  cost-conditional CVaR={cc_cvar:.3f}"
        if spec.get(c) is not None:
            s += f"   specialized={spec[c]:.3f}   gap={100*(cc_cvar-spec[c])/spec[c]:+.1f}%"
        print(s)

    _plot(rows)
    (MET / "cost_conditional.json").write_text(json.dumps(rows, indent=2))
    print(f"\nSaved -> {MET}/cost_conditional.json, {FIG}/cost_conditional.png")
    return rows


def _plot(rows):
    x = [r_["cost"] * 100 for r_ in rows]
    cc = [r_["cc_cvar"] for r_ in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, cc, "s-", color="tab:green", label="single cost-conditional net")
    if all(r_["specialized_cvar"] is not None for r_ in rows):
        sp = [r_["specialized_cvar"] for r_ in rows]
        ax.plot(x, sp, "o--", color="tab:red", label="per-cost specialized nets")
    ax.set_title("One cost-conditional policy vs per-cost specialists (GBM)")
    ax.set_xlabel("transaction cost (% of notional)")
    ax.set_ylabel("CVaR95 (lower = better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "cost_conditional.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
