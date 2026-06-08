"""Black-Scholes analytics + discrete delta-hedge baseline."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from src.env.market import hedging_pnl


def bs_price(S, K, T, r, sigma, call=True):
    S = np.asarray(S, dtype=float)
    T = np.maximum(T, 1e-12)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, call=True):
    S = np.asarray(S, dtype=float)
    T = np.maximum(T, 1e-12)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if call else norm.cdf(d1) - 1.0


def bs_delta_holdings(paths, K, T, r, sigma):
    """BS delta at each rebalance time, shape (n_paths, n_steps).

    Holding over [t, t+1) = delta computed at time t with the remaining maturity.
    """
    n_paths, n_steps1 = paths.shape
    n_steps = n_steps1 - 1
    dt = T / n_steps
    holdings = np.empty((n_paths, n_steps))
    for t in range(n_steps):
        tau = T - t * dt  # remaining maturity
        holdings[:, t] = bs_delta(paths[:, t], K, tau, r, sigma, call=True)
    return holdings


def bs_delta_hedge_pnl(paths, K, T, r, sigma, cost_rate, premium=None):
    """Run the discrete BS delta hedge through the shared P&L accounting."""
    if premium is None:
        premium = float(bs_price(paths[0, 0], K, T, r, sigma, call=True))
    holdings = bs_delta_holdings(paths, K, T, r, sigma)
    pnl = hedging_pnl(paths, holdings, K, premium, cost_rate)
    return pnl, holdings, premium
