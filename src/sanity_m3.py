"""M3 sanity: with zero cost, the trained policy should approximate BS delta
and match its CVaR. If deep hedging can't recover BS delta when costs are off,
nothing downstream is trustworthy.
"""
import numpy as np

from src.baselines.bs import bs_delta_hedge_pnl, bs_delta_holdings
from src.env.market import hedging_pnl, simulate_gbm, summary
from src.training.train import policy_holdings, train_deep_hedger

S0, K, T, r, sig, n_steps = 100.0, 100.0, 1.0, 0.0, 0.2, 50

print("training deep hedger at cost=0 ...")
net, premium = train_deep_hedger(
    S0=S0, K=K, T=T, r=r, sigma=sig, n_steps=n_steps,
    cost_rate=0.0, epochs=300, batch=4096, seed=0,
)

rng = np.random.default_rng(123)  # fresh test paths
paths = simulate_gbm(S0, r, sig, T, n_steps, 40000, rng)

dh_pnl, bs_hold, _ = bs_delta_hedge_pnl(paths, K, T, r, sig, cost_rate=0.0, premium=premium)
pol_hold = policy_holdings(net, paths, K, T)
pol_pnl = hedging_pnl(paths, pol_hold, K, premium, cost_rate=0.0)

mad = float(np.mean(np.abs(pol_hold - bs_hold)))
print(f"\nmean|policy_delta - BS_delta| = {mad:.4f}  (small => recovered BS delta)")
print("BS delta  :", summary(dh_pnl))
print("DeepHedge :", summary(pol_pnl))
assert mad < 0.06, f"policy did not recover BS delta (MAD={mad:.3f})"
print("\nPASS: deep hedger recovers BS delta at zero cost")
