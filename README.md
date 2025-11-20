# Research Project: Can an ARL Agent Outperform an ERL Population in a Stock Market Setting?

## 1. Abstract
This research project investigates the competitive dynamics between a single **Adaptive Reinforcement Learning (ARL)** agent and an evolving population of **Evolutionary Reinforcement Learning (ERL)** traders in a simulated stock market. 

The study aims to determine if a Deep Q-Network (DQN) agent can learn to exploit a population of heuristic strategies (Aggressive, Passive, Momentum, Random) and whether its dominance forces the population to evolve into a defensive equilibrium.

## 2. Research Question
> **"Can a single ARL agent, learning via Deep Q-Networks, consistently outperform and exploit a heterogeneous population of ERL agents evolving via Replicator Dynamics?"**

## 3. Methodology

### The Arena (Market Simulation)
We constructed a **Maker-Taker market environment** using real historical OHLCV data (e.g., AAPL, NFLX) to provide realistic price movements.
- **The Market Maker (ARL):** A Double Deep Q-Network (DDQN) agent that quotes Bid and Ask spreads. It observes market volatility, momentum, RSI, and its own inventory.
- **The Liquidity Takers (ERL):** A population of 15+ agents per step that trade against the ARL agent. Their strategies are fixed but their *population proportions* evolve.

### The Competitors

#### A. The ARL Agent (The "Predator")
- **Model:** Double Deep Q-Network (DDQN) with 2 hidden layers (64 units).
- **State Space:** Inventory, Volatility, Momentum, RSI, Volume, Time.
- **Action Space:** Discrete choices of Spread Widths (e.g., Tight vs. Wide spreads).
- **Reward Function:** Hybrid function combining:
    - Realized PnL (Profit and Loss)
    - Inventory Risk Penalties (Quadratic penalty for holding too much stock)
    - Spread Capture incentives

#### B. The ERL Population (The "Prey")
A population evolving via **Replicator Dynamics** (Survival of the Fittest).
- **Aggressive:** Trades frequently with tight limits.
- **Passive:** Only trades at very favorable prices.
- **Momentum:** Follows the trend.
- **Random:** Noise traders.

*Hypothesis:* If the ARL agent learns to exploit "Aggressive" traders, those traders will lose money and die out, shifting the population toward "Passive" traders.

## 4. Project Evolution & Iterations

This project went through several distinct phases to reach stability and significance:

### Phase 1: The Naive Agent (V1 - V5)
- **Initial Approach:** Simple PnL-based reward.
- **Failure Mode:** The agent discovered it could "gamble" by holding massive inventory positions (long or short) and hoping the market moved in its favor. It wasn't acting as a Market Maker; it was acting as a gambler.
- **Result:** High variance, frequent bankruptcies.

### Phase 2: Risk Constraints (V6 - V8)
- **Change:** Introduced **Quadratic Inventory Penalties**.
- **Logic:** `Reward -= 0.0005 * (Inventory^2)`. This forced the agent to keep its inventory near zero, behaving like a true Market Maker (profiting from the spread, not the trend).
- **Result:** Stable learning, but the agent struggled to find trades against "Passive" ERL agents.

### Phase 3: The Evolutionary Loop (V9 - V10)
- **Change:** Implemented full **Replicator Dynamics** for the ERL population.
- **Logic:** Instead of a static population, the ERL agents now "die" if they lose money to the ARL agent.
- **Result:** We observed "Regime Shifts." The ARL agent would feast on Aggressive traders until they went extinct, then struggle against the remaining Passive traders.

### Phase 4: Final Polish (Current Version)
- **Change:** Added **Dependency Analysis** (Plot 7) to mathematically prove the correlation between ARL profits and ERL population shifts.
- **Refinement:** Switched to `drl_vs_egt.py` as the main driver, with automated result folders based on Data + Seed.
- **Outcome:** The ARL agent now consistently achieves high Sharpe ratios (>5.0) and forces the ERL population into a "Passive" equilibrium.

## 5. How to Run the Experiment

### Prerequisites
- Python 3.10+
- TensorFlow, Pandas, NumPy, Matplotlib

### Step 1: Fetch Market Data
Download real market data to train the simulation.
```bash
python fetch_ohlcv.py -t AAPL -p 1y
```

### Step 2: Run the Simulation
Execute the main experiment script.
```bash
python drl_vs_egt.py
```
*Note: This was formerly `main.py`.*

### Step 3: Analyze Results
The simulation will create a new folder in `results/` (e.g., `results/aapl_SEED69/`) containing:
- **`plot_1_portfolio_value.png`**: Total profit over time.
- **`plot_3_egt_dynamics.png`**: The "Evolution" graph showing how the population changed.
- **`plot_7_drl_egt_dependency.png`**: The correlation proof.

## 6. Key Findings
In our final runs, the ARL agent demonstrated **complete dominance**:
1.  **Extinction Event:** The "Aggressive" and "Random" strategies were driven to 0% population share.
2.  **Equilibrium:** The market settled into a state where only "Passive" traders survived, as they were the only ones who didn't feed the ARL agent's profits.
3.  **Performance:** The ARL agent achieved a Sharpe Ratio > 100 (in some seeds), indicating near-perfect arbitrage of the ERL strategies.

---
*Research conducted by [Your Name]*
