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


def simulate_merton(S0, r, sigma, T, n_steps, n_paths, rng,
                    lam=1.0, jump_mean=-0.10, jump_std=0.15):
    """Merton jump-diffusion under the risk-neutral measure, (n_paths, n_steps+1).

    Diffusion + compound-Poisson jumps J=exp(Y), Y~N(jump_mean, jump_std^2).
    Drift carries the jump compensator -lam*kappa so E[S_t]=S0*e^{rt} (risk-neutral),
    which makes the MC premium an unbiased fair value. Jumps are the gap risk a
    continuous BS delta hedge structurally cannot remove.
    """
    dt = T / n_steps
    kappa = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0
    drift = (r - 0.5 * sigma**2 - lam * kappa) * dt
    z = rng.standard_normal((n_paths, n_steps))
    n_jumps = rng.poisson(lam * dt, (n_paths, n_steps))
    # sum of n_jumps iid Normal(jump_mean, jump_std^2) = Normal(n*m, n*s^2)
    jump = n_jumps * jump_mean + np.sqrt(n_jumps) * jump_std * rng.standard_normal((n_paths, n_steps))
    incr = drift + sigma * np.sqrt(dt) * z + jump
    logpath = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(incr, axis=1)], axis=1)
    return S0 * np.exp(logpath)


def simulate_heston(S0, r, T, n_steps, n_paths, rng,
                    v0=0.04, kappa=2.0, theta=0.04, xi=0.4, rho=-0.7):
    """Heston stochastic-volatility paths under risk-neutral measure (full-trunc Euler).

    dS = r S dt + sqrt(v) S dW1 ;  dv = kappa(theta - v) dt + xi sqrt(v) dW2 ;
    corr(dW1, dW2) = rho. Default v0=theta=0.04 -> ~20% vol. The hedger sees only
    S (vol is latent), so a constant-vol BS delta is mis-specified here.
    """
    dt = T / n_steps
    S = np.empty((n_paths, n_steps + 1)); S[:, 0] = S0
    v = np.full(n_paths, v0)
    for t in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n_paths)
        vp = np.maximum(v, 0.0)
        S[:, t + 1] = S[:, t] * np.exp((r - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1)
        v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * z2
    return S


def call_payoff(S_T, K):
    return np.maximum(S_T - K, 0.0)


def mc_premium(paths, K):
    """Risk-neutral fair premium = E[payoff] on the (r-drift) path measure (r=0 here)."""
    return float(call_payoff(paths[:, -1], K).mean())


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
