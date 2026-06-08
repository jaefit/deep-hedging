"""Train the neural hedger end-to-end to minimize CVaR of the hedged P&L.

Differentiable rollout: rolling the policy forward through GBM paths and the
transaction-cost accounting yields a P&L that is differentiable in the policy
parameters, so we backprop the risk objective straight through the hedge.

Risk objective — empirical Expected Shortfall via top-k (convex, differentiable,
parameter-free): mean of the worst k = ceil((1-a)*B) losses in the batch. This
isolates the tail from step one. (An earlier Rockafellar-Uryasev formulation with
a free VaR scalar `w` under-converged: when `w` lags the true VaR the objective
silently degenerates toward pure mean-P&L maximization, so the policy learned to
just trade less — better mean, fatter tail. Top-k removes that failure mode.)
"""
from __future__ import annotations

import numpy as np
import torch

from src.env.market import simulate_gbm
from src.models.policy import HedgePolicy


def _rollout_pnl(net, S, K, T, premium, cost_rate):
    """Differentiable hedged-book P&L over torch price paths S: (B, n_steps+1)."""
    B, n1 = S.shape
    n_steps = n1 - 1
    dt = T / n_steps
    h_prev = torch.zeros(B)
    trading = torch.zeros(B)
    cost = torch.zeros(B)
    for t in range(n_steps):
        tau = (T - t * dt) / T
        feats = torch.stack(
            [torch.log(S[:, t] / K), torch.full((B,), tau), h_prev], dim=1
        )
        h = net(feats)
        trading = trading + h * (S[:, t + 1] - S[:, t])
        cost = cost + cost_rate * S[:, t] * torch.abs(h - h_prev)
        h_prev = h
    cost = cost + cost_rate * S[:, n_steps] * torch.abs(h_prev)  # final unwind
    payoff = torch.clamp(S[:, -1] - K, min=0.0)
    return premium + trading - cost - payoff


def train_deep_hedger(
    S0=100.0, K=100.0, T=1.0, r=0.0, sigma=0.2, n_steps=50,
    cost_rate=0.0, alpha=0.95, epochs=400, batch=8192, lr=1e-3, seed=0,
):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    from src.baselines.bs import bs_price

    premium = float(bs_price(S0, K, T, r, sigma, call=True))
    net = HedgePolicy()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    k = max(1, int(np.ceil((1.0 - alpha) * batch)))  # tail size for empirical ES

    net.train()
    for ep in range(epochs):
        paths = simulate_gbm(S0, r, sigma, T, n_steps, batch, rng)  # risk-neutral drift=r
        S = torch.from_numpy(paths.astype(np.float32))
        pnl = _rollout_pnl(net, S, K, T, premium, cost_rate)
        loss_var = -pnl  # loss = negative P&L
        es = torch.topk(loss_var, k).values.mean()  # mean of worst k losses = CVaR
        opt.zero_grad()
        es.backward()
        opt.step()
        if (ep + 1) % 50 == 0:
            print(f"  ep{ep + 1:4d}  ES(CVaR)={es.item():.4f}  "
                  f"meanPnL={pnl.mean().item():.4f}  stdPnL={pnl.std().item():.4f}")
    return net, premium


def policy_holdings(net, paths, K, T):
    """Evaluate the trained policy on numpy paths -> holdings (n_paths, n_steps)."""
    net.eval()
    B, n1 = paths.shape
    n_steps = n1 - 1
    dt = T / n_steps
    S = torch.from_numpy(paths.astype(np.float32))
    h_prev = torch.zeros(B)
    holdings = np.empty((B, n_steps), dtype=np.float32)
    with torch.no_grad():
        for t in range(n_steps):
            tau = (T - t * dt) / T
            feats = torch.stack(
                [torch.log(S[:, t] / K), torch.full((B,), tau), h_prev], dim=1
            )
            h = net(feats)
            holdings[:, t] = h.numpy()
            h_prev = h
    return holdings
