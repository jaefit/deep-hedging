# Two Finance + ML Projects, Explained Simply

*For a 2nd-year undergrad. No finance background needed. Easy English.*

This walks through two small projects about **volatility** (how much prices jump
around) and **hedging** (protecting yourself from those jumps). Both use machine
learning (ML). The goal is not "ML is cool" — it is to learn **when ML actually
helps and when it does not.**

---

## Part 0: The 5 ideas you need first

1. **Return** — how much a price changed, in percent. If a stock goes 100 → 102,
   the return is +2%.
2. **Volatility (σ, "sigma")** — how *wild* the returns are. Calm market = low
   volatility. Panic/crash = high volatility. It is basically the standard
   deviation of returns. We usually quote it "annualized" (e.g. 20% per year).
3. **Option (a call)** — a contract: the buyer gets the right to buy a stock at a
   fixed price later. If you **sell** a call, you have a risk: if the stock shoots
   up, you owe money. You got paid a fee (the **premium**) for taking that risk.
4. **Hedging** — buying/selling the stock to cancel out your option risk, so you
   don't care which way the market moves. A perfect hedge = you keep the premium
   and take no risk.
5. **Delta** — how many shares to hold to hedge. The classic **Black-Scholes (BS)**
   formula tells you the delta. "Delta hedging" = follow that formula.

Why a trader cares: banks **sell** options to clients (the "sales" desk) and then
must **hedge** them (the "trading" desk). Doing this well = profit. Doing it badly
= big losses. Both projects live in this world.

---

## Part 1: Volatility Forecasting

### The question
Can ML predict tomorrow's volatility better than old-school statistics?

### Why it matters
Volatility is the main input to option prices and to risk limits. If you can
forecast it, you price and manage risk better.

### What we did (the build)
- **Data:** 15 years of daily prices for SPY, QQQ, AAPL (free, from Yahoo).
- **Target:** future "realized volatility" — we estimate each day's volatility
  from the high/low/open/close prices (a standard trick called Garman-Klass),
  then try to predict the *next* day's value.
- **The golden rule — no cheating (no "look-ahead"):** to predict day *t+1* we
  only use information available up to day *t*. We test with **walk-forward**:
  train on the past, predict the future, slide forward, repeat. If you ever let
  future data leak into training, your results look amazing and are fake.
- **Models we compared:**
  - *Naive:* "tomorrow = today" (a baseline to beat).
  - *Classic stats:* EWMA, **GARCH**, and **HAR** (a simple formula that uses
    yesterday's, last week's, and last month's volatility).
  - *ML:* **XGBoost** (decision trees) and an **LSTM** (a neural network for
    sequences).
- **Scoreboard:** we measure error with RMSE and **QLIKE** (a standard volatility
  error score), plus **direction accuracy** (did we at least get "vol goes up vs
  down" right?).

### The result
| What we measured | Winner |
|---|---|
| Size of the forecast (how big is vol) | **HAR** — the simple classic formula |
| Direction (is vol rising or falling) | **LSTM** — the neural network |

The fancy ML models did **not** beat the simple HAR formula at predicting the
*level* of volatility. HAR is famously hard to beat. But ML **did** win at
predicting the *direction* of change.

### What it means
> ML did not replace the classic model. It added a *different* skill (direction).
> If your job is to bet on volatility going up or down, ML helps. If you just need
> the number, the simple formula is enough.

A nice sanity check: XGBoost, when we looked inside it, mostly used "last week's
volatility" — exactly what HAR uses by hand. So the ML model re-discovered the
classic idea. That is *why* HAR is so hard to beat.

---

## Part 2: Deep Hedging

### The question
Can a neural network hedge an option better than the textbook Black-Scholes delta
— especially when the real world is messier than the textbook?

### Why it matters
The BS delta formula assumes a perfect world: you can trade continuously, for
free, and prices move smoothly. The real world has **trading costs** and **sudden
jumps**. So the textbook hedge is never quite right. Can a network *learn* a
better hedge directly from data?

### What we did (the build)
- **Simulate a market:** generate thousands of fake-but-realistic price paths
  (starting with "GBM", the smooth textbook model).
- **Sell one call option** and hedge it step by step over its life (50 steps).
- **Two hedgers compete:**
  - *Baseline:* the BS delta formula (ignores trading cost).
  - *Deep hedger:* a small neural network that, at each step, looks at the current
    price, time left, and current position, and decides how many shares to hold.
- **How the network learns:** we let it hedge over many simulated paths, measure
  the **tail risk** of its profit/loss, and adjust the network to make that tail
  less bad. We do this with backpropagation (the same training idea as any neural
  net).
- **Tail risk = CVaR** ("the average of your worst 5% days"). Lower is better. We
  care about the tail because that is what blows up trading desks.

### A real bug we hit (and why it's a good story)
Our first "risk" formula had a subtle flaw: it secretly drifted into maximizing
*average* profit instead of controlling the *tail*. So the network learned to just
**trade less** — which looks good on average but leaves a fat, dangerous tail. We
caught it because the training score (≈7) didn't match the real measured risk
(≈2.7). We switched to a simpler, direct tail formula and it worked. **Lesson:
always check that your model is optimizing the thing you actually care about.**

### The results

**(a) Trading costs.** We swept the cost from 0% to 2%:
- At ~0% cost, the **BS formula wins** (it's already optimal; the network can only
  copy it). This is the honest, correct answer — not a failure.
- Once cost is ≥ 0.5%, the **deep hedger wins** and the gap grows. At 2% cost it
  cuts tail risk by about a third. It learned a "don't trade unless you really
  need to" style that the textbook formula can't.

**(b) When the textbook world breaks.** We tested harder markets:
- *Jumps* (prices can suddenly gap): deep hedger's advantage **clearly grows**
  (~27% better vs ~21% in the smooth world). A continuous formula simply cannot
  react to a jump; a learned policy handles it better. (Checked over 3 random
  seeds — it's real, not luck.)
- *Stochastic (moving) volatility:* the edge was only **slightly** bigger, and
  within random noise — because we used a mild setup close to the textbook world.

**(c) One model instead of many.** Instead of training a separate network per cost
level, we trained **one** network that takes the cost as an input. It matched all
the specialized networks, and even handled cost levels it never saw. Practical win.

### What it means
> A learned hedge earns its keep exactly where the textbook breaks: **high trading
> costs and sudden jumps.** In the clean, cheap, textbook world, the simple formula
> is already optimal and ML adds nothing.

---

## Part 3: How the two projects connect

Project 1 *predicts* volatility. Project 2 *uses* a volatility number to hedge.
So we wired them together: we took the volatility forecast from Project 1 (for
SPY) and fed it into the hedger in Project 2.

Then we asked the realistic question: **what if the forecast is wrong?** The desk
hedges using its forecast, but the market does something else. The deep hedger
kept its advantage across this whole range — but, honestly, it did *not* degrade
more gracefully than the textbook; it just started lower and stayed lower.

---

## Part 4: The one big lesson

> **ML is not a magic upgrade. It is a tool for the exact spots where the classic
> assumptions break.**

- Volatility level → classic HAR is enough. ML only adds *direction*.
- Hedging in a clean, cheap world → textbook BS is enough.
- Hedging with big costs or jumps → the learned hedge clearly wins.

And just as important — **how we stayed honest:**
1. **No look-ahead** (train only on the past).
2. **Check the model optimizes the real goal** (we caught a bug doing this).
3. **Don't trust one lucky run** — repeat with several random seeds and report the
   *range*, not a single number. (We caught ourselves over-claiming three times
   this way.)

That honesty — knowing *when your method wins, when it doesn't, and how you proved
it* — is the actual skill. Not "I used a neural network."

---

## Mini-glossary
- **σ / volatility:** how wildly prices move.
- **Option / call:** right to buy later at a set price; the seller takes risk for a fee.
- **Hedging:** trading the stock to cancel option risk.
- **Delta / Black-Scholes:** the textbook formula for how much to hedge.
- **GBM:** the smooth textbook model of price movement.
- **Jump model (Merton) / stochastic vol (Heston):** more realistic, messier markets.
- **HAR / GARCH / EWMA:** classic volatility formulas.
- **XGBoost / LSTM:** machine-learning models.
- **CVaR (tail risk):** average of your worst-case outcomes; lower is safer.
- **Walk-forward / look-ahead:** the right way to test (only past data) vs the
  cheating way (peeking at the future).
