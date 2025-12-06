# AI Project: DRL vs EGT in Simulated Markets

This project simulates a financial market environment to investigate whether an **Adaptive Reinforcement Learning (RL) Agent** can outperform a population of **Evolutionary Game Theory (EGT)** traders. The simulation uses real-world market data (OHLCV) to drive price movements while agents interact through order book dynamics.

---

## 📚 Software Manual

### 1. Prerequisites
Ensure you have Python 3.8+ installed. You will need the following Python libraries:

```bash
pip install numpy pandas matplotlib tensorflow keras scipy pyyaml
```

### 2. Project Structure
- **`drl_vs_egt.py`**: The main simulation script containing the DRL agent, EGT population, and market environment logic.
- **`config.yaml`**: Configuration file for controlling simulation parameters (agent hyperparameters, market rules, data paths).
- **`market_data/`**: Directory containing historical CSV data files (e.g., `nvda_up-stable.csv`).
- **`results/`**: Directory where simulation plots and logs are saved.

### 3. Configuration (`config.yaml`)
You can promote or change simulation behaviors without touching the code. Key settings include:
- **`data`**: Specify the input CSV file.
- **`agent`**: Adjust the Neural Network architecture, learning rate, and epsilon decay for the DRL agent.
- **`market`**: Set initial capital, transaction costs, and the initial mix of EGT strategies.

### 4. Running the Simulation
To start the simulation, run the main script from your terminal:

```bash
python drl_vs_egt.py
```

### 5. Output
Key metrics and plots are generated in the `results/` folder, structured by the dataset name and random seed.
- **Console Output**: Real-time logging of training steps (epsilon values, progress).
- **Final Summary**: Prints total return, Sharpe ratio, and EGT population distribution.
- **Plots**:
  - `plot_1_portfolio_value.png`: The DRL agent's wealth over time.
  - `plot_2_inventory_heatmap.png`: Heatmap of the agent's inventory positions.
  - `plot_support_drawdown.png`: Analysis of maximum drawdown.
  - *...and more.*

---

## 🧠 Approach Explanation

### Objective
The core research question is: **"Can a single sophisticated DRL agent learn to exploit or coexist with a population of simpler, evolving trading strategies?"**

### The Market Environment
The simulation is grounded in **real market data** (OHLCV). The "Mid Price" follows the historical data trace. However, the *trading* happens between the agents:
- The DRL Agent acts as a **Market Maker**, quoting bid and ask spreads around the mid-price.
- The EGT Agents act as **Takers**, deciding whether to buy or sell based on their specific strategies and the quoted prices.

### 🤖 The Agents

#### 1. The DRL Agent (The "Hero")
- **Architecture**: Double Deep Q-Network (DDQN).
- **State Space**: Observes inventory, market volatility, momentum, volume, RSI, and time remaining.
- **Action Space**: Chooses from a discrete set of bid/ask spread pairs (e.g., tight spreads `[0.1, 0.1]` vs. wide spreads `[0.4, 0.4]`).
- **Reward Function**: A hybrid reward maximizing **Total Portfolio Value** while penalizing **Inventory Risk** and **Transaction Costs**. It uses "Shape-Shifting" rewards to encourage market neutrality (inventory near zero).

#### 2. The EGT Population (The "Ecosystem")
A population of traders that evolves over time based on **Replicator Dynamics**.
- **Strategies**:
  - **Aggressive**: High willingness to trade, accepts wider spreads.
  - **Passive**: only trades if spreads are very favorable.
  - **Momentum**: Follows the trend (buys if price is rising).
  - **Random**: Noise traders.
- **Evolution**: At fixed intervals, the population is updated. Strategies that were profitable in recent trades grow in number; losing strategies shrink. This creates a non-stationary environment for the DRL agent.

### Dynamics & Interaction
1. **Quoting**: The DRL agent observes the market state and sets spreads.
2. **Trading**: EGT agents potentially accept these quotes, leading to transactions.
3. **Inventory Management**: The DRL agent accumulates inventory (long/short). It learns to skew spreads to offload risk (e.g., if holding too much stock, it lowers the ask price to encourage selling).
4. **Co-Evolution**: As the DRL agent exploits a certain EGT strategy (e.g., "Aggressive"), that EGT strategy might lose money and shrink in population, forcing the DRL agent to adapt to the remaining strategies.
