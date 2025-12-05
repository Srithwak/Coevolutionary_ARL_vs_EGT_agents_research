# v12.py
#
# Simulation environment for:
# "Can an ARL Agent Outperform EGT Populations in a Simulated Market?"
#
# Key features:
# - DRL (ARL) agent vs evolving EGT population
# - Real OHLCV midprice stream
# - Agent quotes bid/ask spreads and accumulates inventory + cash
# - Full portfolio accounting with starting capital
# - Hybrid reward: absolute PnL + normalized return
# - Performance analytics: Sharpe, drawdown, skew, kurtosis
# - Cleaned plot set (1..7) + combined summary figure
#
# Notes:
# - Plot 1: Portfolio Value
# - Plot 2: Inventory Distribution Heatmap   (replaces old simple inventory line plot)
# - Plot 3: EGT Population Dynamics
# - Plot 4: Spread Width + Epsilon
# - Plot 5: Rolling Reward
# - Plot 6: Policy Distribution
# - Plot 7: DRL–EGT Dependency
#
# Extra supporting plots (not numbered, still saved):
# - drawdown curve
# - return distribution
#
# Requirements:
# - numpy, pandas, matplotlib
# - tensorflow / keras
# - scipy (for gaussian_filter, spline, skew/kurtosis)
#
# ---------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import random
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import yaml
import argparse

# Suppress TensorFlow warnings
tf.get_logger().setLevel("ERROR")

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Global config placeholder - will be loaded in main
CONFIG = {}


class MarketData:
    """
    Loads, processes, and provides real OHLCV market data.
    """

    def __init__(self, ohlcv_filename, config):
        self.filename = ohlcv_filename
        self.config = config
        
        volatility_window = self.config["features"]["volatility_window"]
        momentum_window = self.config["features"]["momentum_window"]
        rsi_window = self.config["features"]["rsi_window"]
        self.filename = ohlcv_filename
        if not os.path.exists(self.filename):
            print(f"Error: Data file not found at '{self.filename}'")
            print(
                "Please provide a valid CSV file with 'High', 'Low', 'Volume' columns."
            )
            raise FileNotFoundError(f"Missing data file: {self.filename}")

        print(f"Loading data from '{self.filename}'...")
        self.data = pd.read_csv(self.filename)

        print("Validating and converting data types...")
        required_cols = ["High", "Low", "Volume"]

        for col in required_cols:
            if col not in self.data.columns:
                print(f"Error: CSV must contain '{col}' column.")
                raise ValueError(f"Missing required column: {col}")
            self.data[col] = pd.to_numeric(self.data[col], errors="coerce")

        original_len = len(self.data)
        self.data.dropna(subset=required_cols, inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        new_len = len(self.data)

        if new_len < original_len:
            print(f"Dropped {original_len - new_len} rows containing non-numeric data.")

        # mid_price ~ quoted "fair" price
        self.data["mid_price"] = (self.data["High"] + self.data["Low"]) / 2.0
        self.max_steps = len(self.data)

        if self.max_steps < volatility_window:
            print(
                f"Warning: Data length ({self.max_steps}) is shorter than feature window ({volatility_window})."
            )

        print("Calculating technical features (volatility, momentum, RSI, etc.)...")
        self.data["returns"] = self.data["mid_price"].pct_change()
        self.data["volatility"] = (
            self.data["returns"].rolling(window=volatility_window).std().fillna(0)
        )
        self.data["price_change"] = self.data["mid_price"].diff()
        self.data["momentum"] = (
            self.data["price_change"].rolling(window=momentum_window).mean().fillna(0)
        )
        rolling_vol_mean = (
            self.data["Volume"].rolling(window=volatility_window).mean().fillna(1)
        )
        self.data["norm_volume"] = (
            self.data["Volume"] / (rolling_vol_mean + 1e-6)
        ).fillna(0)

        delta = self.data["mid_price"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
        rs = gain / (loss + 1e-6)
        self.data["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

        print(
            f"Successfully loaded and processed '{self.filename}' with {self.max_steps} steps."
        )

    def get_market_state(self, step):
        if step >= self.max_steps:
            return None, None, None, None, None
        price = self.data.loc[step, "mid_price"]
        volatility = self.data.loc[step, "volatility"]
        momentum = self.data.loc[step, "momentum"]
        norm_volume = self.data.loc[step, "norm_volume"]
        rsi = self.data.loc[step, "rsi"]
        return price, volatility, momentum, norm_volume, rsi


class DQNAgent:
    """
    Double Deep Q-Network (DDQN) agent = the ARL learner.
    Chooses spreads to quote and accumulates cash/inventory.
    """

    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.config = config
        
        # Load actions from config
        self.actions = [tuple(a) for a in self.config["agent"]["actions"]]
        self.action_size = action_size

        # portfolio state
        self.inventory = 0
        self.cash = 0.0  # will be overwritten by Market with INITIAL_CAPITAL

        agent_params = self.config["agent"]
        self.gamma = agent_params["gamma"]
        self.epsilon = agent_params["epsilon_start"]
        self.epsilon_decay = agent_params["epsilon_decay"]
        self.epsilon_min = agent_params["epsilon_min"]
        self.learning_rate = agent_params["learning_rate"]
        self.batch_size = agent_params["batch_size"]
        self.replay_buffer = deque(maxlen=agent_params["replay_buffer_maxlen"])
        
        self.nn_architecture = agent_params["nn_architecture"]

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(
            Dense(
                self.nn_architecture[0]["units"],
                input_dim=self.state_size,
                activation=self.nn_architecture[0]["activation"],
            )
        )
        for layer in self.nn_architecture[1:]:
            model.add(Dense(layer["units"], activation=layer["activation"]))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def format_state(
        self, inventory, volatility, momentum, norm_volume, rsi, step, max_steps
    ):
        # Normalize inventory with tanh squashing so +/-5 units ~ significant
        norm_inventory = np.tanh(inventory / 5.0)

        norm_time = step / max_steps
        norm_rsi = (rsi / 50.0) - 1.0
        norm_vol = np.tanh(norm_volume - 1.0)

        return np.array(
            [[norm_inventory, volatility, momentum, norm_vol, norm_rsi, norm_time]]
        )

    def remember(self, state, action_idx, reward, next_state, done):
        self.replay_buffer.append((state, action_idx, reward, next_state, done))

    def choose_action(self, state):
        # epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        minibatch = random.sample(self.replay_buffer, self.batch_size)

        states = np.squeeze(np.array([item[0] for item in minibatch]))
        actions = np.array([item[1] for item in minibatch])
        rewards = np.array([item[2] for item in minibatch])
        next_states = np.squeeze(np.array([item[3] for item in minibatch]))
        dones = np.array([item[4] for item in minibatch])

        # Double DQN target calc
        target_q_next = self.target_model.predict(next_states, verbose=0)
        best_actions_next = np.argmax(
            self.model.predict(next_states, verbose=0), axis=1
        )
        target_q_values = target_q_next[range(self.batch_size), best_actions_next]
        targets = rewards + self.gamma * target_q_values * (1 - dones)

        current_q = self.model.predict(states, verbose=0)
        for i in range(self.batch_size):
            current_q[i, actions[i]] = targets[i]

        self.model.fit(states, current_q, epochs=1, verbose=0)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class Market:
    """
    Market environment:
    - One DRL agent (the ARL learner)
    - Population of EGT strategies that evolves via replicator-like dynamics
    - Real price path via provided OHLCV
    """

    def __init__(self, config):
        self.config = config
        
        data_path = os.path.join(self.config["data"]["folder"], self.config["data"]["file"])
        self.market_data = MarketData(ohlcv_filename=data_path, config=self.config)
        self.step = 0

        self.state_size = 6
        self.action_size = len(self.config["agent"]["actions"])
        
        self.drl_agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            config=self.config
        )

        # --- NEW: Assign starting capital to agent
        self.INITIAL_CAPITAL = self.config["market"]["initial_capital"]
        self.drl_agent.cash = self.INITIAL_CAPITAL

        # target update cadence
        self.UPDATE_TARGET_EVERY = self.config["market"]["update_target_every"]

        # EGT population setup
        self.egt_strategies = self.config["market"]["egt_strategies"]
        self.egt_proportions = np.array(self.config["market"]["egt_proportions"])
        self.egt_total_payoffs = np.zeros(len(self.egt_strategies))
        self.egt_total_trades = np.zeros(len(self.egt_strategies))
        self.N_EGT_AGENTS_PER_STEP = self.config["market"]["n_egt_agents_per_step"]
        self.EVOLVE_EVERY = self.config["market"]["evolve_every"]

        # cost & behavior knobs
        self.transaction_cost_per_trade = self.config["market"]["transaction_cost_per_trade"]

        # logging for analysis and plotting
        self.history = {
            "step": [],
            "mid_price": [],
            # keep "drl_profit" for backwards compat, but it's just total portfolio value:
            "drl_profit": [],
            "portfolio_value": [],
            "drl_inventory": [],
            "chosen_spread_width": [],
            "epsilon": [],
            "reward": [],
            "egt_prop_aggressive": [],
            "egt_prop_passive": [],
            "egt_prop_random": [],
            "egt_prop_momentum": [],
        }

        # track last portfolio value for reward calc
        self.last_drl_value = self.INITIAL_CAPITAL

    def get_egt_action(self, strategy_name, mid_price, momentum):
        """
        Sample what an EGT agent is trying to do:
        - Returns ("buy"/"sell", price they'll cross at)
        """
        action_type = random.choice(["buy", "sell"])

        if strategy_name == "aggressive":
            price_offset = random.uniform(0.02, 0.08)
        elif strategy_name == "passive":
            price_offset = random.uniform(0.05, 0.15)
        elif strategy_name == "momentum":
            if momentum > 0:
                action_type = "buy"
            elif momentum < 0:
                action_type = "sell"
            price_offset = random.uniform(0.04, 0.12)
        else:  # "random"
            price_offset = random.uniform(0.01, 0.5)

        if action_type == "buy":
            return ("buy", mid_price + price_offset)
        else:
            return ("sell", mid_price - price_offset)

    def evolve_population(self):
        """
        Replicator-like update:
        Strategies with higher profit-per-trade gain population share.
        Small mutation keeps diversity.
        """
        fitness = self.egt_total_payoffs / (self.egt_total_trades + 1e-6)
        positive_fitness = fitness - np.min(fitness) + 1
        avg_fitness = np.dot(self.egt_proportions, positive_fitness)
        if avg_fitness == 0:
            return

        # proportional growth
        self.egt_proportions = self.egt_proportions * (positive_fitness / avg_fitness)

        # tiny uniform mutation
        mutation = 0.001
        self.egt_proportions = self.egt_proportions * (1 - mutation) + (
            mutation / len(self.egt_strategies)
        )
        self.egt_proportions /= np.sum(self.egt_proportions)

        # reset accumulators per epoch
        self.egt_total_payoffs.fill(0)
        self.egt_total_trades.fill(0)

    def run_step(self):
        # pull market state at current step
        mid_price, volatility, momentum, norm_volume, rsi = (
            self.market_data.get_market_state(self.step)
        )
        if mid_price is None:
            return False  # ran out of data

        # --- 1. Agent picks spreads ---
        state = self.drl_agent.format_state(
            self.drl_agent.inventory,
            volatility,
            momentum,
            norm_volume,
            rsi,
            self.step,
            self.market_data.max_steps,
        )
        action_idx = self.drl_agent.choose_action(state)
        bid_spread, ask_spread = self.drl_agent.actions[action_idx]

        # --- 2. Inventory-aware spread adjustment ---
        inv_bias = np.clip(abs(self.drl_agent.inventory) / 5.0, 1.0, 3.0)
        if self.drl_agent.inventory > 0:
            # holding long -> discourage buying more, encourage selling
            bid_spread *= inv_bias
            ask_spread /= inv_bias
        elif self.drl_agent.inventory < 0:
            # holding short -> discourage selling more, encourage buying back
            bid_spread /= inv_bias
            ask_spread *= inv_bias

        drl_bid_price = mid_price - bid_spread
        drl_ask_price = mid_price + ask_spread

        # --- 3. Simulate trades with sampled EGT agents ---
        step_payoffs = np.zeros(len(self.egt_strategies))
        step_trades = np.zeros(len(self.egt_strategies))

        drl_trades_this_step = 0
        trade_profit_total = 0.0  # spread capture metric

        sampled_agents = np.random.choice(
            self.egt_strategies,
            size=self.N_EGT_AGENTS_PER_STEP,
            p=self.egt_proportions,
        )

        for strategy_name in sampled_agents:
            egt_action, egt_price = self.get_egt_action(
                strategy_name, mid_price, momentum
            )
            strat_idx = self.egt_strategies.index(strategy_name)

            # EGT sells to us at/below our bid -> we BUY one unit
            if egt_action == "sell" and egt_price <= drl_bid_price:
                self.drl_agent.inventory += 1
                self.drl_agent.cash -= drl_bid_price

                # EGT "profit" is what they sold for
                step_payoffs[strat_idx] += drl_bid_price
                step_trades[strat_idx] += 1
                drl_trades_this_step += 1

                # capture = we bought below mid
                trade_profit_total += mid_price - drl_bid_price

            # EGT buys from us at/above our ask -> we SELL one unit
            elif egt_action == "buy" and egt_price >= drl_ask_price:
                self.drl_agent.inventory -= 1
                self.drl_agent.cash += drl_ask_price

                # EGT "profit" negative because they paid above mid
                step_payoffs[strat_idx] -= drl_ask_price
                step_trades[strat_idx] += 1
                drl_trades_this_step += 1

                # capture = we sold above mid
                trade_profit_total += drl_ask_price - mid_price

        # update running totals for replicator dynamics
        self.egt_total_payoffs += step_payoffs
        self.egt_total_trades += step_trades

        # --- 4. Compute portfolio value at this step ---
        inv = self.drl_agent.inventory
        portfolio_value = self.drl_agent.cash + (inv * mid_price)

        # --- 5. HYBRID reward signal ---
        abs_change = portfolio_value - self.last_drl_value
        rel_change = abs_change / max(self.last_drl_value, 1e-8)

        reward = 0.5 * abs_change + 0.5 * (rel_change * 1000.0)

        # Risk shaping / behavior incentives:

        # Penalize absolute inventory and squared inventory (risk)
        reward -= 0.002 * abs(inv)
        reward -= 0.0005 * (inv**2)

        # Transaction costs
        reward -= drl_trades_this_step * self.transaction_cost_per_trade

        # Bonus for reducing inventory magnitude (mean-reversion to flat)
        prev_inv = (
            self.history["drl_inventory"][-1] if self.history["drl_inventory"] else 0
        )
        if abs(inv) < abs(prev_inv):
            reward += 0.003 * (abs(prev_inv) - abs(inv))

        # Encourage staying near 0 inventory (market maker neutrality)
        reward += 0.002 * (1 - min(abs(inv) / 10.0, 1.0))

        # Reward for spread capture (profitable fills)
        reward += 0.002 * trade_profit_total

        # Soft penalty to discourage persistent directional drift
        reward -= 0.0015 * inv

        # update baseline for next step
        self.last_drl_value = portfolio_value

        # --- 6. Build next state & train DRL ---
        next_data_tuple = self.market_data.get_market_state(self.step + 1)
        next_price, next_vol, next_mom, next_norm_vol, next_rsi = next_data_tuple
        done = next_price is None

        next_state = (
            state
            if done
            else self.drl_agent.format_state(
                inv,
                next_vol,
                next_mom,
                next_norm_vol,
                next_rsi,
                self.step + 1,
                self.market_data.max_steps,
            )
        )

        self.drl_agent.remember(state, action_idx, reward, next_state, done)
        self.drl_agent.replay()

        # --- 7. Population evolution & target net sync ---
        if self.step % self.EVOLVE_EVERY == 0:
            self.evolve_population()
        if self.step % self.UPDATE_TARGET_EVERY == 0:
            self.drl_agent.update_target_model()

        # --- 8. Log for analysis / plotting ---
        self.history["step"].append(self.step)
        self.history["mid_price"].append(mid_price)

        self.history["drl_profit"].append(portfolio_value)
        self.history["portfolio_value"].append(portfolio_value)

        self.history["drl_inventory"].append(inv)
        self.history["chosen_spread_width"].append(bid_spread + ask_spread)
        self.history["epsilon"].append(self.drl_agent.epsilon)
        self.history["reward"].append(reward)

        for i, strat in enumerate(self.egt_strategies):
            self.history[f"egt_prop_{strat}"].append(self.egt_proportions[i])

        self.step += 1
        return True

    def run_simulation(self):
        print("Starting simulation: 1 DRL agent vs Evolving EGT population.")
        print(f"Total data steps: {self.market_data.max_steps}")
        while self.run_step():
            if self.step % 100 == 0:
                print(
                    f"Step {self.step}/{self.market_data.max_steps}... "
                    f"Epsilon: {self.drl_agent.epsilon:.2f}"
                )

        print("Simulation complete.")
        self.print_summary()
        self.plot_results()

    def print_summary(self):
        """
        Print final performance summary:
        - Final portfolio value / total return
        - Sharpe-like ratio (per step)
        - Max drawdown
        - Mean/std/skew/kurtosis of per-step returns
        """
        if not self.history["portfolio_value"]:
            print("Simulation ended before any history was recorded.")
            return

        from scipy.stats import skew, kurtosis  # for stats on returns

        pv = np.array(self.history["portfolio_value"], dtype=float)
        final_portfolio = pv[-1]

        # per-step returns
        if len(pv) > 1:
            step_returns = np.diff(pv) / pv[:-1]
        else:
            step_returns = np.array([0.0])

        mean_ret = float(np.mean(step_returns)) if len(step_returns) > 0 else 0.0
        std_ret = float(np.std(step_returns)) if len(step_returns) > 0 else 0.0
        skew_ret = float(skew(step_returns)) if len(step_returns) > 1 else 0.0
        kurt_ret = float(kurtosis(step_returns)) if len(step_returns) > 1 else 0.0

        # Sharpe-like metric (signal/noise per simulation step)
        sharpe = (mean_ret / (std_ret + 1e-8)) if len(step_returns) > 1 else 0.0

        # drawdown: how bad did it get vs running peak
        rolling_max = np.maximum.accumulate(pv)
        drawdowns = pv / rolling_max - 1.0
        max_drawdown = float(np.min(drawdowns))

        total_return_pct = (final_portfolio / self.INITIAL_CAPITAL - 1.0) * 100.0

        steps_per_year = 13000  # change this for your data interval
        sharpe_annualized = sharpe * np.sqrt(steps_per_year)

        print("\n" + "=" * 30)
        print("=== FINAL RESULTS ===")
        print(f"Starting Capital: ${self.INITIAL_CAPITAL:,.2f}")
        print(f"Final Portfolio Value: ${final_portfolio:,.2f}")
        print(f"Total Return: {total_return_pct:.2f}%")

        print(f"Sharpe (per-step signal/noise): {sharpe:.3f}")
        print(f"Sharpe (annualized): {sharpe_annualized:.3f}")

        print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
        print("--- Step Return Distribution ---")
        print(f"Mean Return per Step: {mean_ret:.6f}")
        print(f"Std Dev of Return per Step: {std_ret:.6f}")
        print(f"Skewness of Returns: {skew_ret:.6f}")
        print(f"Kurtosis of Returns: {kurt_ret:.6f}")

        print("\nFinal EGT Population Distribution:")
        for i, strat in enumerate(self.egt_strategies):
            print(f"  {strat.capitalize()}: {self.egt_proportions[i]:.3f}")
        print("=" * 30 + "\n")

    def plot_results(self):
        """
        Save all paper-quality plots:
        1. Portfolio value
        2. Inventory distribution heatmap (replaces simple inventory line plot)
        3. EGT population proportions over time
        4. Spread width vs epsilon
        5. Rolling reward (learning stability)
        6. Learned action policy distribution
        7. DRL profit vs EGT population dependency
        Plus:
           - Drawdown curve (supporting)
           - Step return histogram (supporting)
        Also:
           - Combined summary multi-panel figure (plot_0_combined_summary.png)
        """

        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        from scipy.ndimage import gaussian_filter, gaussian_filter1d
        from scipy.interpolate import make_interp_spline

        STD_FIGSIZE = (8, 6)
        STD_DPI = 300

        df_history = pd.DataFrame(self.history)
        if df_history.empty:
            print("History is empty, skipping plotting.")
            return

        print("Saving plots...")

        # Create results subfolder
        # Remove extension from filename for cleaner folder name
        data_name = self.config["data"]["file"].replace(".csv", "")
        folder_name = f"{data_name}_SEED{SEED}"
        save_dir = os.path.join(self.config["data"]["results_folder"], folder_name)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created results directory: {save_dir}")
        else:
            print(f"Saving to existing directory: {save_dir}")

        # -------------------------------------------------
        # Plot 1: Portfolio Value Over Time
        # -------------------------------------------------
        fig1, ax1 = plt.subplots(figsize=STD_FIGSIZE)
        ax1.plot(
            df_history["step"],
            df_history["portfolio_value"],
            label="Portfolio Value ($)",
            color="darkblue",
        )
        ax1.set_title("DRL Agent Portfolio Value Over Time")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        plt.tight_layout()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "plot_1_portfolio_value.png"), dpi=STD_DPI)
        plt.close(fig1)

        # -------------------------------------------------
        # Plot 0: Market Data (Mid Price)
        # -------------------------------------------------
        fig0, ax0 = plt.subplots(figsize=STD_FIGSIZE)
        ax0.plot(
            df_history["step"],
            df_history["mid_price"],
            label="Market Mid Price ($)",
            color="black",
            alpha=0.7,
        )
        ax0.set_title(f"Market Data: {data_name}")
        ax0.set_xlabel("Time Step")
        ax0.set_ylabel("Price ($)")
        ax0.grid(True, alpha=0.3)
        ax0.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "plot_0_market_data.png"), dpi=STD_DPI)
        plt.close(fig0)

        # -------------------------------------------------
        # Plot 2: Inventory Distribution Heatmap (formerly plot 8)
        # This replaces the old simple inventory line plot.
        # -------------------------------------------------
        inventory_series = np.array(df_history["drl_inventory"], dtype=float)
        steps_arr = np.array(df_history["step"], dtype=float)

        # normalize time to [0,1] so x-axis is unit-scaled
        if len(steps_arr) > 0:
            time_normalized = (
                steps_arr / steps_arr.max() if steps_arr.max() != 0 else steps_arr
            )
        else:
            time_normalized = steps_arr

        bins_x = 256
        bins_y = 256
        heatmap, xedges, yedges = np.histogram2d(
            time_normalized, inventory_series, bins=[bins_x, bins_y], density=True
        )
        heatmap_smooth = gaussian_filter(heatmap, sigma=3)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        im = ax2.imshow(
            heatmap_smooth.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
            interpolation="bicubic",
        )

        # overlay raw inventory (faint)
        ax2.plot(
            time_normalized,
            inventory_series,
            color="white",
            alpha=0.25,
            linewidth=0.6,
            label="Inventory",
        )

        # smoothed average inventory line
        if len(inventory_series) > 3:
            smoothed_inventory = gaussian_filter1d(
                inventory_series, sigma=len(inventory_series) / 40
            )
            x_smooth = np.linspace(time_normalized.min(), time_normalized.max(), 1000)
            spline = make_interp_spline(time_normalized, smoothed_inventory, k=3)
            y_smooth = spline(x_smooth)

            ax2.plot(
                x_smooth,
                y_smooth,
                color="white",
                linewidth=2.0,
                alpha=0.95,
                label="Smoothed Avg",
            )

        ax2.set_title("Inventory Distribution Over Time (Heatmap)")
        ax2.set_xlabel("Time [-]")
        ax2.set_ylabel("Inventory [units]")
        ax2.legend(loc="upper left", fontsize=8, facecolor="black", framealpha=0.3)

        cbar = fig2.colorbar(im, ax=ax2)
        cbar.set_label("Relative Frequency Density")

        plt.tight_layout()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "plot_2_inventory_heatmap.png"), dpi=STD_DPI)
        plt.close(fig2)

        # -------------------------------------------------
        # SUPPORTING: Drawdown Curve
        # -------------------------------------------------
        pv = df_history["portfolio_value"].to_numpy(dtype=float)
        rolling_max = np.maximum.accumulate(pv)
        drawdowns = pv / rolling_max - 1.0

        fig_dd, ax_dd = plt.subplots(figsize=STD_FIGSIZE)
        ax_dd.plot(df_history["step"], drawdowns * 100.0, color="firebrick")
        ax_dd.set_title("Drawdown (%) Over Time")
        ax_dd.set_xlabel("Time Step")
        ax_dd.set_ylabel("Drawdown (%)")
        ax_dd.axhline(y=0, color="black", linestyle="--", linewidth=1)
        ax_dd.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "plot_support_drawdown.png"), dpi=STD_DPI)
        plt.close(fig_dd)

        # -------------------------------------------------
        # SUPPORTING: Distribution of Step Returns
        # -------------------------------------------------
        if len(pv) > 1:
            step_returns = np.diff(pv) / pv[:-1]
        else:
            step_returns = np.array([0.0])

        fig_ret, ax_ret = plt.subplots(figsize=STD_FIGSIZE)
        ax_ret.hist(step_returns, bins=30, alpha=0.7, edgecolor="black")
        ax_ret.set_title("Distribution of Step Returns")
        ax_ret.set_xlabel("Return per Step")
        ax_ret.set_ylabel("Frequency")
        plt.tight_layout()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "plot_support_return_dist.png"), dpi=STD_DPI)
        plt.close(fig_ret)

        # -------------------------------------------------
        # Plot 3: EGT Population Dynamics
        # -------------------------------------------------
        fig3, ax3 = plt.subplots(figsize=STD_FIGSIZE)
        egt_labels = [s.capitalize() for s in self.egt_strategies]
        egt_data = [df_history[f"egt_prop_{s}"] for s in self.egt_strategies]
        ax3.stackplot(df_history["step"], egt_data, labels=egt_labels)
        ax3.set_title("EGT Population Evolution (Replicator Dynamics)")
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("Proportion of Population")
        ax3.set_ylim(0, 1)
        ax3.legend(loc="upper left")
        plt.tight_layout()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "plot_3_egt_dynamics.png"), dpi=STD_DPI)
        plt.close(fig3)

        # -------------------------------------------------
        # Plot 4: Spread Width + Epsilon
        # -------------------------------------------------
        fig4, ax4 = plt.subplots(figsize=STD_FIGSIZE)
        ax4.plot(
            df_history["step"],
            df_history["chosen_spread_width"],
            label="Chosen Spread Width",
            color="red",
            alpha=0.6,
        )
        ax4.set_title("DRL Agent's Quoted Spread Width")
        ax4.set_xlabel("Time Step")
        ax4.set_ylabel("Spread Width ($)")
        ax4.grid(True, alpha=0.3)

        ax4_twin = ax4.twinx()
        ax4_twin.plot(
            df_history["step"],
            df_history["epsilon"],
            label="Epsilon",
            color="grey",
            linestyle=":",
        )
        ax4_twin.set_ylabel("Epsilon")

        lines, labels = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines + lines2, labels + labels2, loc="upper right")

        plt.tight_layout()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "plot_4_drl_spread.png"), dpi=STD_DPI)
        plt.close(fig4)

        # -------------------------------------------------
        # Plot 5: Rolling Average Reward
        # -------------------------------------------------
        df_history["rolling_avg_reward"] = (
            df_history["reward"].rolling(window=100, min_periods=1).mean()
        )

        fig5, ax5 = plt.subplots(figsize=STD_FIGSIZE)
        ax5.plot(
            df_history["step"],
            df_history["rolling_avg_reward"],
            label="Rolling Avg. Reward (100 steps)",
            color="purple",
        )
        ax5.axhline(y=0, color="black", linestyle="--", linewidth=1)
        ax5.set_title("DRL Agent's Progressive Learning (Rolling Reward)")
        halfway_idx = len(df_history["step"]) // 2
        exploiting_spreads = df_history["chosen_spread_width"].iloc[halfway_idx:]

        # -------------------------------------------------
        # Plot 6: Learned Policy Distribution
        # -------------------------------------------------
        fig6, ax6 = plt.subplots(figsize=STD_FIGSIZE)

        if not exploiting_spreads.empty:
            spread_widths_rounded = exploiting_spreads.round(2)
            spread_counts = spread_widths_rounded.value_counts().sort_index()
            spread_dist = spread_counts / len(spread_widths_rounded)

            # log color normalization to highlight preference peaks
            global_min, global_max = 0.001, 0.4
            norm = mcolors.LogNorm(vmin=global_min, vmax=global_max)
            colors = [cm.viridis(norm(v)) for v in spread_dist]

            bars = spread_dist.plot(kind="bar", ax=ax6, color=colors)
            ax6.set_title(f"DRL Learned Policy (Last {len(exploiting_spreads)} Steps)")
            ax6.set_xlabel("Chosen Spread Width ($)")
            ax6.set_ylabel("Proportion of Actions")
            ax6.set_ylim(0, 1)
            ax6.set_xticklabels(spread_dist.index, rotation=45)
            ax6.grid(axis="y", linestyle="--", alpha=0.5)

            # label bars with %
            for bar, val in zip(bars.patches, spread_dist):
                ax6.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.01,
                    f"{val * 100:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
            sm.set_array([])
            cbar = fig6.colorbar(sm, ax=ax6)
            ticks = [0.001, 0.01, 0.05, 0.1, 0.2, 0.4]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{t*100:.1f}%" for t in ticks])
            cbar.set_label("Relative Frequency (log scale)", rotation=270, labelpad=15)
        else:
            ax6.set_title("DRL Learned Policy (No data)")

        plt.tight_layout()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "plot_6_policy_dist.png"), dpi=STD_DPI)
        plt.close(fig6)

        # -------------------------------------------------
        # Plot 7: DRL–EGT Dependency (rolling correlation)
        # -------------------------------------------------
        fig7, ax7 = plt.subplots(figsize=STD_FIGSIZE)

        df_history["egt_avg"] = df_history[
            [f"egt_prop_{s}" for s in self.egt_strategies]
        ].mean(axis=1)

        df_history["drl_profit_change"] = df_history["drl_profit"].diff().fillna(0)
        df_history["egt_change"] = df_history["egt_avg"].diff().fillna(0)

        rolling_corr = (
            df_history["drl_profit_change"]
            .rolling(window=100, min_periods=10)
            .corr(df_history["egt_change"])
        )

        ax7.plot(
            df_history["step"],
            rolling_corr,
            color="darkorange",
            label="Rolling Corr (DRL Profit vs EGT Mix)",
        )
        ax7.axhline(y=0, color="black", linestyle="--", linewidth=1)
        ax7.set_title("Dependency of DRL Agent on EGT Population")
        ax7.set_xlabel("Time Step")
        ax7.set_ylabel("Rolling Correlation")
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "plot_7_drl_egt_dependency.png"), dpi=STD_DPI)
        plt.close(fig7)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DRL vs EGT Simulation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    CONFIG = load_config(args.config)
    
    SEED = CONFIG["seed"]
    print(f"Setting global random seed to: {SEED}")
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print(f"Running simulation with config: {args.config}")
    try:
        market_sim = Market(config=CONFIG)
        market_sim.run_simulation()
    except FileNotFoundError as e:
        print("\n--- ERROR ---")
        print(f"File not found: {e}")
        print(
            "Please download a real OHLCV data file and update the DATA_FILE variable."
        )
        print("It must contain 'High', 'Low', 'Close', 'Open', and 'Volume' columns.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()

#
