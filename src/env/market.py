"""Market environment: GBM paths, option payoff, transaction-cost P&L accounting.

We SELL one European call and hedge it. P&L of the hedged book over a path:

    PnL = premium + sum_t h_t (S_{t+1}-S_t) - costs - payoff(S_T)

where h_t is the share holding over interval [t, t+1). Costs are proportional to
traded notional, charged on every rebalance including the initial buy and the
full unwind at maturity. A perfect continuous cost-free hedge gives PnL -> 0;
real (discrete, costly) hedging leaves a P&L distribution whose tail risk we
minimize. Both the BS-delta baseline and the neural policy are scored through
this same function, so comparisons are apples-to-apples.
"""
from __future__ import annotations

import numpy as np


def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, rng):
    """Return GBM price paths, shape (n_paths, n_steps+1)."""
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    incr = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    logpath = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(incr, axis=1)], axis=1)
    return S0 * np.exp(logpath)


def call_payoff(S_T, K):
    return np.maximum(S_T - K, 0.0)


def hedging_pnl(paths, holdings, K, premium, cost_rate):
    """P&L of the hedged short-call book, per path.

    paths    : (n_paths, n_steps+1) prices
    holdings : (n_paths, n_steps)   shares held over [t, t+1), t=0..n_steps-1
    premium  : scalar option premium received at t0 (e.g. BS price)
    cost_rate: proportional transaction cost (fraction of traded notional)
    Returns  : (n_paths,) P&L.
    """
    S = paths
    n_steps = holdings.shape[1]

    # self-financing trading gains: sum_t h_t (S_{t+1}-S_t)
    trading = np.sum(holdings * (S[:, 1:] - S[:, :-1]), axis=1)

    # transaction costs on every rebalance: h_{-1}=0, unwind to h_T=0 at maturity
    h_prev = np.zeros((S.shape[0],))
    cost = np.zeros((S.shape[0],))
    for t in range(n_steps):
        cost += cost_rate * S[:, t] * np.abs(holdings[:, t] - h_prev)
        h_prev = holdings[:, t]
    cost += cost_rate * S[:, n_steps] * np.abs(h_prev)  # final liquidation

    payoff = call_payoff(S[:, -1], K)
    return premium + trading - cost - payoff


def cvar(pnl, alpha=0.95):
    """CVaR of LOSS at level alpha (mean of the worst (1-alpha) P&L outcomes).

    Returned as a positive number = expected shortfall. Lower is better.
    """
    losses = -np.asarray(pnl)
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    return float(tail.mean()) if tail.size else float(var)


def summary(pnl, alpha=0.95):
    pnl = np.asarray(pnl)
    return {
        "mean": float(pnl.mean()),
        "std": float(pnl.std()),
        "cvar95": cvar(pnl, alpha),
        "q05": float(np.quantile(pnl, 0.05)),
    }
