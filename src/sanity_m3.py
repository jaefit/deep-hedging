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
    cost_rate=0.0, epochs=400, batch=8192, seed=0,  # validated config
)

rng = np.random.default_rng(123)  # fresh test paths
paths = simulate_gbm(S0, r, sig, T, n_steps, 40000, rng)

dh_pnl, bs_hold, _ = bs_delta_hedge_pnl(paths, K, T, r, sig, cost_rate=0.0, premium=premium)
pol_hold = policy_holdings(net, paths, K, T)
pol_pnl = hedging_pnl(paths, pol_hold, K, premium, cost_rate=0.0)

mad = float(np.mean(np.abs(pol_hold - bs_hold)))
bs_s, dh_s = summary(dh_pnl), summary(pol_pnl)
print(f"\nmean|policy_delta - BS_delta| = {mad:.4f}  (informational; ES does not"
      " constrain the body of the hedge, only the tail)")
print("BS delta  :", bs_s)
print("DeepHedge :", dh_s)

# The objective is tail risk (CVaR), so the right zero-cost sanity bar is
# tail-recovery: with no cost, BS delta is optimal and the deep hedger should
# reach comparable CVaR (it cannot beat BS here, only approach it).
ratio = dh_s["cvar95"] / bs_s["cvar95"]
print(f"\nzero-cost CVaR ratio Deep/BS = {ratio:.3f}")
assert ratio < 1.30, f"deep hedger did not recover BS-level tail risk (ratio={ratio:.3f})"
assert mad < 0.10, f"policy delta far from BS delta (MAD={mad:.3f})"
print("PASS: deep hedger recovers BS-level hedging at zero cost")
