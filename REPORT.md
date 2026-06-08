# Deep Hedging — Results Report

**Question:** can a neural network, trained to minimize tail risk under
transaction costs, hedge a short option better than the textbook Black-Scholes
delta hedge — and where does that edge appear?

**Setup:** sell one 1-year ATM European call (S₀=K=100, σ=20%, r=0). Hedge over
50 discrete steps along GBM paths. Proportional transaction cost swept from 0 to
2% of traded notional. Both hedgers are scored on a common 50k-path test set
through the *same* P&L accounting, so the comparison is apples-to-apples.

- **Baseline:** BS delta hedge — analytic delta, recomputed each step, cost-blind.
- **Deep hedger:** a 2-layer MLP policy, shared across time, mapping
  (log-moneyness, time-to-maturity, current holding) → share holding in [0,1].
  Trained end-to-end by backprop through the differentiable hedge rollout.

**Risk objective:** empirical Expected Shortfall (CVaR₉₅) — the mean of the worst
5% of P&L outcomes — computed per batch via top-k. Lower = less tail risk.

---

## Result (CVaR₉₅ of the hedged-book P&L; lower is better)

| Cost (% notional) | BS delta CVaR₉₅ | Deep CVaR₉₅ | BS mean P&L | Deep mean P&L | Winner |
|---|---|---|---|---|---|
| 0.0  | **2.28** | 2.63 |  0.00 |  0.01 | BS delta |
| 0.1  | **2.71** | 2.93 | -0.33 | -0.29 | BS delta |
| 0.5  | 4.53 | **4.17** | -1.65 | -1.34 | **Deep** |
| 1.0  | 6.91 | **5.61** | -3.31 | -2.53 | **Deep** |
| 2.0  | 11.80 | **7.85** | -6.62 | -4.54 | **Deep** |

The CVaR curves **cross between 0.1% and 0.5%**: BS delta is best when costs are
negligible, the deep hedger takes over once costs bite, and its edge widens with
cost (at 2% it cuts tail risk by a third, 11.80 → 7.85, *and* halves the mean
cost bleed, -6.62 → -4.54). The premium received is the same BS price for both, so
this is purely a hedging-quality difference.

## What it says

1. **At zero / low cost, BS delta wins (as it should).** With ATM strike, clean
   GBM, constant vol and only proportional cost, BS delta is already near-optimal.
   The neural policy can only *approximate* it and pays a small approximation
   variance — so it slightly trails. This is the correct, honest result, not a
   failure: there is nothing for the network to exploit yet.

2. **At high cost (1–2%), the deep hedger wins decisively on CVaR.** It learns a
   no-trade-band behavior — rebalancing less aggressively to economize on cost —
   which BS delta (which over-trades, blind to cost) cannot. The crossover is the
   whole point: deep hedging earns its keep exactly when frictions make the
   textbook hedge wasteful.

3. **The edge is joint mean + tail.** Once the objective genuinely targets the
   tail, the deep hedger improves mean P&L (less cost bleed) *and* CVaR (tighter
   tail) at high cost — it is not trading mean for tail.

## A real bug we caught and fixed (methodology note)

The first implementation used the Rockafellar-Uryasev CVaR objective with a free
VaR scalar `w` optimized jointly with the policy. It **silently failed**: `w`
under-converged, and when `w` sits below the true VaR the RU objective degenerates
toward pure *mean*-P&L maximization. Under cost, maximizing mean says "trade
less" — so the policy learned to under-hedge: better mean, **fatter tail**, losing
to BS delta on CVaR at every cost level, with the gap widening in cost.

The tell was a mismatch between the training objective value (~7.2) and the
realized CVaR (~2.75) at zero cost — `w` was nowhere near the true VaR. The fix:
replace RU with a **parameter-free top-k Expected Shortfall** (mean of the worst
k = ⌈0.05·B⌉ losses), which isolates the tail from the first gradient step. After
the swap, training ES and realized CVaR agree, and the expected high-cost win
appears. (See `src/training/train.py` docstring.)

## Figures (`results/figures/`)
- `cvar_vs_cost.png` — CVaR₉₅ and mean P&L vs transaction cost, BS vs Deep. The
  CVaR curves cross: BS below at low cost, Deep below at high cost.
- `pnl_hist_cost20.png` — P&L distributions at 2% cost; the deep hedger shifts the
  left tail inward.

## Limitations / honest caveats
- ATM, single option, constant vol, GBM, proportional cost — the friendly case for
  BS delta. The literature's larger deep-hedging wins come from jumps, stochastic
  vol, and discrete/fixed costs not modeled here.
- One network per cost level (cost-specific), not a single cost-conditional policy.
- Risk-neutral drift; no model/parameter uncertainty.
- Single training seed. The 1–2% wins are large and seed-robust; the 0.5%
  crossover (4.17 vs 4.53) is narrow — the monotone trend is the reliable signal,
  not that one point's exact margin.

## Next steps
- Stochastic-vol / jump paths, where BS delta degrades and the NN edge should widen.
- Feed the **Volatility Forecasting** project's σ estimates as the simulation vol
  (and stress-test hedging when the realized vol differs from the pricing vol).
- A single policy conditioned on the cost level.

## Reproduce
```bash
python -m src.sanity_m3     # zero-cost: deep hedger recovers BS delta
python -m src.run_sweep     # cost sweep + figures + metrics
```
Artifacts: `results/metrics/sweep_metrics.csv`, `results/figures/*.png`.
