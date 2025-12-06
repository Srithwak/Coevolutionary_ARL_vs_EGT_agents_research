# egt_only.py
#
# Simulation environment for:
# "EGT Population Dynamics Baseline (No AI Agent)"
#
# Key features:
# - Random Agent (Baseline) vs evolving EGT population
# - Real OHLCV midprice stream
# - Agent quotes random bid/ask spreads from config
# - Full portfolio accounting
# - No Helper Reward / No Training
# - Performance analytics
#
# ---------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import random
import os
import yaml
import argparse

# Global config placeholder - will be loaded in main
CONFIG = {}


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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


class RandomAgent:
    """
    Baseline agent that picks spreads randomly.
    Replaces DQNAgent.
    """

    def __init__(self, action_size, config):
        self.config = config
        
        # Load actions from config
        self.actions = [tuple(a) for a in self.config["agent"]["actions"]]
        self.action_size = action_size

        # portfolio state
        self.inventory = 0
        self.cash = 0.0  # will be overwritten by Market with INITIAL_CAPITAL

    def choose_action(self):
        # Purely random choice
        return random.choice(range(self.action_size))


class Market:
    """
    Market environment:
    - One Random agent (Baseline)
    - Population of EGT strategies that evolves via replicator-like dynamics
    - Real price path via provided OHLCV
    """

    def __init__(self, config):
        self.config = config
        
        data_path = os.path.join(self.config["data"]["folder"], self.config["data"]["file"])
        self.market_data = MarketData(ohlcv_filename=data_path, config=self.config)
        self.step = 0

        self.action_size = len(self.config["agent"]["actions"])
        
        # Swapped DQNAgent for RandomAgent
        self.drl_agent = RandomAgent(
            action_size=self.action_size,
            config=self.config
        )

        # --- Assign starting capital to agent
        self.INITIAL_CAPITAL = self.config["market"]["initial_capital"]
        self.drl_agent.cash = self.INITIAL_CAPITAL

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
            "portfolio_value": [],
            "drl_inventory": [],
            "chosen_spread_width": [],
            "egt_prop_aggressive": [],
            "egt_prop_passive": [],
            "egt_prop_random": [],
            "egt_prop_momentum": [],
        }

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

        # --- 1. Agent picks spreads (RANDOMLY) ---
        action_idx = self.drl_agent.choose_action()
        bid_spread, ask_spread = self.drl_agent.actions[action_idx]

        # --- 2. Inventory-aware spread adjustment ---
        # (Still keeping this logic? Yes, even random agents might be forced to adjust by their inventory 
        # constraints if we want a fair comparison of "strategy" vs "learning".
        # However, a purely random agent might not even check inventory. 
        # But to be a functional market maker, you arguably need self-preservation.
        # I will KEEP the inventory bias so the only difference is the CHOICE of base spread.)
        inv_bias = np.clip(abs(self.drl_agent.inventory) / 5.0, 1.0, 3.0)
        if self.drl_agent.inventory > 0:
            bid_spread *= inv_bias
            ask_spread /= inv_bias
        elif self.drl_agent.inventory < 0:
            bid_spread /= inv_bias
            ask_spread *= inv_bias

        drl_bid_price = mid_price - bid_spread
        drl_ask_price = mid_price + ask_spread

        # --- 3. Simulate trades with sampled EGT agents ---
        step_payoffs = np.zeros(len(self.egt_strategies))
        step_trades = np.zeros(len(self.egt_strategies))

        # drl_trades_this_step = 0
        # trade_profit_total = 0.0 

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
                # drl_trades_this_step += 1
                # trade_profit_total += mid_price - drl_bid_price

            # EGT buys from us at/above our ask -> we SELL one unit
            elif egt_action == "buy" and egt_price >= drl_ask_price:
                self.drl_agent.inventory -= 1
                self.drl_agent.cash += drl_ask_price

                # EGT "profit" negative because they paid above mid
                step_payoffs[strat_idx] -= drl_ask_price
                step_trades[strat_idx] += 1
                # drl_trades_this_step += 1
                # trade_profit_total += drl_ask_price - mid_price

        # update running totals for replicator dynamics
        self.egt_total_payoffs += step_payoffs
        self.egt_total_trades += step_trades

        # --- 4. Compute portfolio value at this step ---
        inv = self.drl_agent.inventory
        portfolio_value = self.drl_agent.cash + (inv * mid_price)

        # --- 5. Step Logic & Evolution ---
        # No training step here.

        if self.step % self.EVOLVE_EVERY == 0:
            self.evolve_population()
        
        # --- 6. Log for analysis / plotting ---
        self.history["step"].append(self.step)
        self.history["mid_price"].append(mid_price)
        self.history["portfolio_value"].append(portfolio_value)
        self.history["drl_inventory"].append(inv)
        self.history["chosen_spread_width"].append(bid_spread + ask_spread)
        
        for i, strat in enumerate(self.egt_strategies):
            self.history[f"egt_prop_{strat}"].append(self.egt_proportions[i])

        self.step += 1
        return True

    def run_simulation(self):
        print("Starting simulation: 1 RANDOM Baseline Agent vs Evolving EGT population.")
        print(f"Total data steps: {self.market_data.max_steps}")
        while self.run_step():
            if self.step % 100 == 0:
                # No Epsilon to print
                print(f"Step {self.step}/{self.market_data.max_steps}...")

        print("Simulation complete.")
        self.print_summary()
        self.plot_results()

    def print_summary(self):
        """
        Print final performance summary
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
        print("=== FINAL BASELINE RESULTS ===")
        print(f"Starting Capital: ${self.INITIAL_CAPITAL:,.2f}")
        print(f"Final Portfolio Value: ${final_portfolio:,.2f}")
        print(f"Total Return: {total_return_pct:.2f}%")
        print(f"Sharpe (per-step): {sharpe:.3f}")
        print(f"Sharpe (annualized): {sharpe_annualized:.3f}")
        print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
        print("=" * 30 + "\n")

    def plot_results(self):
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        from scipy.ndimage import gaussian_filter, gaussian_filter1d
        from scipy.interpolate import make_interp_spline

        STD_FIGSIZE = (8, 6)
        STD_DPI = 200

        df_history = pd.DataFrame(self.history)
        if df_history.empty:
            print("History is empty, skipping plotting.")
            return

        print("Saving plots...")

        # Create results subfolder
        # SAME folder as drl_vs_egt.py (removed _EGT_ONLY suffix)
        data_name = self.config["data"]["file"].replace(".csv", "")
        folder_name = f"{data_name}_SEED{SEED}"
        save_dir = os.path.join(self.config["data"]["results_folder"], folder_name)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created results directory: {save_dir}")
        else:
            print(f"Saving to existing directory: {save_dir}")

        # -------------------------------------------------
        # Plot 8: EGT Population Dynamics (Baseline)
        # -------------------------------------------------
        # We perform ONLY this plot to avoid overwriting plots 1-7 from the Main DRL run.
        
        fig8, ax8 = plt.subplots(figsize=STD_FIGSIZE)
        egt_labels = [s.capitalize() for s in self.egt_strategies]
        egt_data = [df_history[f"egt_prop_{s}"] for s in self.egt_strategies]
        ax8.stackplot(df_history["step"], egt_data, labels=egt_labels)
        ax8.set_title("EGT Dynamics vs Random Baseline (Plot 8)")
        ax8.set_xlabel("Time Step")
        ax8.set_ylabel("Proportion of Population")
        ax8.set_ylim(0, 1)
        ax8.legend(loc="upper left")
        plt.tight_layout()
        plt.tight_layout() # double layout as seen in original
        
        # Save as plot_8...
        save_path = os.path.join(save_dir, "plot_8_egt_baseline_dynamics.png")
        plt.savefig(save_path, dpi=STD_DPI)
        print(f"Saved {save_path}")
        plt.close(fig8)



# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EGT Only Baseline Simulation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config")
    args = parser.parse_args()

    CONFIG = load_config(args.config)
    
    SEED = CONFIG["seed"]
    print(f"Setting global random seed to: {SEED}")
    random.seed(SEED)
    np.random.seed(SEED)
    # No TF seed needed

    print(f"Running BASELINE simulation with config: {args.config}")
    try:
        market_sim = Market(config=CONFIG)
        market_sim.run_simulation()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()