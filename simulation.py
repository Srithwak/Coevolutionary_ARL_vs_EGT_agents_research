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

# Suppress TensorFlow warnings
tf.get_logger().setLevel("ERROR")


class MarketData:
    """
    Loads and provides market data.
    If 'market_data.csv' is not found, it generates it.
    """

    def __init__(
        self,
        filename="market_data.csv",
        volatility_window=20,
        momentum_window=5,
        steps=1000,
    ):
        self.filename = filename
        if not os.path.exists(self.filename):
            print(f"'{self.filename}' not found. Generating new data...")
            self._generate_market_data(steps)

        self.data = pd.read_csv(self.filename)
        self.max_steps = len(self.data)

        # Pre-calculate features for speed
        self.data["returns"] = self.data["mid_price"].pct_change()
        self.data["volatility"] = (
            self.data["returns"].rolling(window=volatility_window).std().fillna(0)
        )

        self.data["price_change"] = self.data["mid_price"].diff()
        self.data["momentum"] = (
            self.data["price_change"].rolling(window=momentum_window).mean().fillna(0)
        )

    def _generate_market_data(self, steps, start_price=100.0, mu=0.0001, sigma=0.01):
        """Generates market price data using Geometric Brownian Motion (GBM)."""
        prices = [start_price]
        for _ in range(steps - 1):
            dt = 1
            Z = np.random.standard_normal()
            new_price = prices[-1] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            )
            prices.append(new_price)
        df = pd.DataFrame(prices, columns=["mid_price"])
        df.to_csv(self.filename, index=False)
        print(f"Successfully created '{self.filename}'.")

    def get_market_state(self, step):
        """Returns the market mid-price and pre-calculated features for a given step."""
        if step >= self.max_steps:
            return None, None, None

        price = self.data.loc[step, "mid_price"]
        volatility = self.data.loc[step, "volatility"]
        momentum = self.data.loc[step, "momentum"]

        return price, volatility, momentum


class DQNAgent:
    """
    The Deep Q-Network (DRL) agent that acts as the market maker.
    It learns to quote the optimal spread.
    """

    def __init__(self, state_size, action_size, nn_architecture, drl_params):
        self.state_size = state_size
        self.actions = [0.1, 0.2, 0.4]  # [Narrow, Medium, Wide] spreads
        self.action_size = action_size

        # Agent's internal state
        self.inventory = 0
        self.cash = 0

        # Default DRL parameters
        default_params = {
            "gamma": 0.95,
            "epsilon": 1.0,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
            "learning_rate": 0.001,
            "replay_buffer_maxlen": 2000,
            "batch_size": 32,
        }
        # Override defaults with any provided params
        self.drl_params = default_params
        self.drl_params.update(drl_params or {})

        # Unpack parameters for easy access
        self.replay_buffer = deque(maxlen=self.drl_params["replay_buffer_maxlen"])
        self.gamma = self.drl_params["gamma"]
        self.epsilon = self.drl_params["epsilon"]
        self.epsilon_decay = self.drl_params["epsilon_decay"]
        self.epsilon_min = self.drl_params["epsilon_min"]
        self.learning_rate = self.drl_params["learning_rate"]
        self.batch_size = self.drl_params["batch_size"]

        self.nn_architecture = nn_architecture

        # Build the Q-Network and the Target Network
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """Builds the deep neural network based on the provided architecture."""
        model = Sequential()
        # Add the first layer, specifying the input dimension
        model.add(
            Dense(
                self.nn_architecture[0]["units"],
                input_dim=self.state_size,
                activation=self.nn_architecture[0]["activation"],
            )
        )
        # Add subsequent hidden layers
        for layer in self.nn_architecture[1:]:
            model.add(Dense(layer["units"], activation=layer["activation"]))
        # Output layer (Q-values for each action)
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Copies weights from the main model to the target model."""
        self.target_model.set_weights(self.model.get_weights())

    def format_state(self, inventory, volatility, momentum, step, max_steps):
        """Formats the agent's and market's state into a vector for the NN."""
        # Normalize inventory to keep it in a reasonable range
        norm_inventory = np.tanh(inventory / 10.0)
        # Normalize time to be between 0 and 1
        norm_time = step / max_steps

        return np.array([[norm_inventory, volatility, momentum, norm_time]])

    def remember(self, state, action_idx, reward, next_state, done):
        """Stores an experience (s, a, r, s', done) in the replay buffer."""
        self.replay_buffer.append((state, action_idx, reward, next_state, done))

    def choose_action(self, state):
        """Chooses an action (a spread) using an epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))  # Explore

        act_values = self.model.predict(state, verbose=0)  # Exploit
        return np.argmax(act_values[0])

    def replay(self):
        """Trains the neural network using a batch of stored experiences."""
        if len(self.replay_buffer) < self.batch_size:
            return

        minibatch = random.sample(self.replay_buffer, self.batch_size)

        # Vectorized operations for speed
        states = np.squeeze(np.array([item[0] for item in minibatch]))
        actions = np.array([item[1] for item in minibatch])
        rewards = np.array([item[2] for item in minibatch])
        next_states = np.squeeze(np.array([item[3] for item in minibatch]))
        dones = np.array([item[4] for item in minibatch])

        # Get Q-values for the next states from the *target* model
        target_q_next = self.target_model.predict(next_states, verbose=0)

        # Bellman equation: Target = R + gamma * max(Q_target(s', a'))
        targets = rewards + self.gamma * np.amax(target_q_next, axis=1) * (1 - dones)

        # Get current Q-values from the *main* model
        current_q = self.model.predict(states, verbose=0)

        # Update the Q-values for only the actions that were taken
        for i in range(self.batch_size):
            current_q[i, actions[i]] = targets[i]

        # Train the main model
        self.model.fit(states, current_q, epochs=1, verbose=0)

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class Market:
    """
    The main simulation environment.
    Manages the DRL agent, the EGT population, the market data, and the simulation loop.
    """

    def __init__(self, nn_architecture, drl_params):
        self.market_data = MarketData()
        self.step = 0

        # --- DRL Agent Setup ---
        self.state_size = 4  # [inventory, volatility, momentum, normalized_time]
        self.action_size = 3  # [narrow, medium, wide]
        self.drl_agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            nn_architecture=nn_architecture,
            drl_params=drl_params,
        )
        self.UPDATE_TARGET_EVERY = 20  # Steps to update target network

        # --- EGT Population Setup ---
        self.egt_strategies = ["aggressive", "passive", "random", "momentum"]
        # This is the core of EGT: the *proportion* of each strategy
        self.egt_proportions = np.array([0.25, 0.25, 0.25, 0.25])
        self.egt_total_payoffs = np.zeros(len(self.egt_strategies))
        self.egt_total_trades = np.zeros(len(self.egt_strategies))
        self.N_EGT_AGENTS_PER_STEP = 15  # How many EGT agents trade per step
        self.EVOLVE_EVERY = 50  # How often the EGT population evolves

        # --- Simulation Parameters ---
        self.transaction_cost_per_trade = 0.01
        self.inventory_penalty_factor = 0.01

        # --- History Tracking ---
        self.history = {
            "step": [],
            "mid_price": [],
            "drl_profit": [],
            "drl_inventory": [],
            "chosen_spread": [],
            "epsilon": [],
            "reward": [],
            "egt_prop_aggressive": [],
            "egt_prop_passive": [],
            "egt_prop_random": [],
            "egt_prop_momentum": [],
        }
        self.last_drl_value = 0  # For calculating Mark-to-Market PnL

    def get_egt_action(self, strategy_name, mid_price, momentum):
        """Defines the behavior of the non-adaptive EGT traders."""
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
        else:  # 'random'
            price_offset = random.uniform(0.01, 0.5)

        return (
            (action_type, mid_price + price_offset)
            if action_type == "buy"
            else (action_type, mid_price - price_offset)
        )

    def evolve_population(self):
        """
        Updates the EGT population proportions using the Replicator Equation.
        This is the core of the Evolutionary Game Theory.
        """
        # Fitness = average payoff per trade
        fitness = self.egt_total_payoffs / (self.egt_total_trades + 1e-6)

        # Replicator dynamics requires non-negative fitness. Shift all values.
        positive_fitness = fitness - np.min(fitness) + 1
        avg_fitness = np.dot(self.egt_proportions, positive_fitness)

        if avg_fitness == 0:
            return  # No trades, no evolution

        # The Replicator Equation: p_i' = p_i * (f_i / avg_f)
        self.egt_proportions = self.egt_proportions * (positive_fitness / avg_fitness)

        # Add mutation to prevent extinction
        mutation = 0.001
        self.egt_proportions = self.egt_proportions * (1 - mutation) + (
            mutation / len(self.egt_strategies)
        )

        # Re-normalize to ensure proportions sum to 1
        self.egt_proportions /= np.sum(self.egt_proportions)

        # Reset payoffs for the next evolution cycle
        self.egt_total_payoffs.fill(0)
        self.egt_total_trades.fill(0)

    def run_step(self):
        """Execute one time step of the market simulation."""

        # 1. GET MARKET DATA
        mid_price, volatility, momentum = self.market_data.get_market_state(self.step)
        if mid_price is None:
            return False  # Simulation over

        # 2. DRL AGENT'S TURN (Set Quotes)
        state = self.drl_agent.format_state(
            self.drl_agent.inventory,
            volatility,
            momentum,
            self.step,
            self.market_data.max_steps,
        )
        action_idx = self.drl_agent.choose_action(state)
        chosen_spread = self.drl_agent.actions[action_idx]
        drl_bid_price = mid_price - chosen_spread
        drl_ask_price = mid_price + chosen_spread

        # 3. EGT POPULATION'S TURN (Trade against DRL)
        step_payoffs = np.zeros(len(self.egt_strategies))
        step_trades = np.zeros(len(self.egt_strategies))
        drl_trades_this_step = 0

        # Sample N agents from the population based on current proportions
        sampled_agents = np.random.choice(
            self.egt_strategies, size=self.N_EGT_AGENTS_PER_STEP, p=self.egt_proportions
        )

        for strategy_name in sampled_agents:
            egt_action, egt_price = self.get_egt_action(
                strategy_name, mid_price, momentum
            )
            strat_idx = self.egt_strategies.index(strategy_name)

            if egt_action == "sell" and egt_price <= drl_bid_price:
                # EGT agent SELLS, DRL agent BUYS
                self.drl_agent.inventory += 1
                self.drl_agent.cash -= drl_bid_price
                step_payoffs[strat_idx] += drl_bid_price  # EGT profit
                step_trades[strat_idx] += 1
                drl_trades_this_step += 1
            elif egt_action == "buy" and egt_price >= drl_ask_price:
                # EGT agent BUYS, DRL agent SELLS
                self.drl_agent.inventory -= 1
                self.drl_agent.cash += drl_ask_price
                step_payoffs[strat_idx] -= drl_ask_price  # EGT "profit" (cost)
                step_trades[strat_idx] += 1
                drl_trades_this_step += 1

        self.egt_total_payoffs += step_payoffs
        self.egt_total_trades += step_trades

        # 4. DRL AGENT LEARNS (Calculate Reward & Replay)

        # Reward is change in Mark-to-Market (MtM) portfolio value
        current_portfolio_value = self.drl_agent.cash + (
            self.drl_agent.inventory * mid_price
        )
        reward = current_portfolio_value - self.last_drl_value

        # Apply costs/penalties
        reward -= drl_trades_this_step * self.transaction_cost_per_trade
        reward -= abs(self.drl_agent.inventory) * self.inventory_penalty_factor

        # Get next state
        next_price, next_vol, next_mom = self.market_data.get_market_state(
            self.step + 1
        )
        done = next_price is None

        if done:
            next_state = state  # Use current state as placeholder
        else:
            next_state = self.drl_agent.format_state(
                self.drl_agent.inventory,
                next_vol,
                next_mom,
                self.step + 1,
                self.market_data.max_steps,
            )

        # Store this experience
        self.drl_agent.remember(state, action_idx, reward, next_state, done)

        # Train the DRL agent
        self.drl_agent.replay()

        # Update value for next step's reward calculation
        self.last_drl_value = current_portfolio_value

        # 5. EVOLVE EGT POPULATION (if it's time)
        if self.step % self.EVOLVE_EVERY == 0:
            self.evolve_population()

        # 6. UPDATE DRL TARGET NETWORK (if it's time)
        if self.step % self.UPDATE_TARGET_EVERY == 0:
            self.drl_agent.update_target_model()

        # 7. RECORD HISTORY
        self.history["step"].append(self.step)
        self.history["mid_price"].append(mid_price)
        self.history["drl_profit"].append(current_portfolio_value)
        self.history["drl_inventory"].append(self.drl_agent.inventory)
        self.history["chosen_spread"].append(chosen_spread)
        self.history["epsilon"].append(self.drl_agent.epsilon)
        self.history["reward"].append(reward)
        for i, strat in enumerate(self.egt_strategies):
            self.history[f"egt_prop_{strat}"].append(self.egt_proportions[i])

        self.step += 1
        return True

    def run_simulation(self):
        """Run the full simulation."""
        print(f"Starting simulation: 1 DRL agent vs Evolving EGT population.")
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
        """Print final summary statistics."""
        final_drl_profit = self.history["drl_profit"][-1]
        print("\n" + "=" * 30)
        print("=== FINAL RESULTS ===")
        print(f"DRL Agent Final Profit: ${final_drl_profit:,.2f}")
        print("\nFinal EGT Population Distribution:")
        for i, strat in enumerate(self.egt_strategies):
            print(f"  {strat.capitalize()}: {self.egt_proportions[i]:.3f}")
        print("=" * 30 + "\n")

    # --- ENTIRELY MODIFIED FUNCTION ---
    # This function now saves 6 separate files AND shows 1 combined window.
    def plot_results(self):
        """
        Visualize simulation results by saving 6 separate, consistent-sized plot files
        AND showing 1 combined window with all plots.
        """

        # Standard size and DPI for all plots (for consistency)
        STD_FIGSIZE = (8, 6)
        STD_DPI = 300

        # Create a DataFrame from history for easy rolling calculations
        df_history = pd.DataFrame(self.history)

        # ---
        # --- PASS 1: SAVE SEPARATE PLOTS ---
        # ---
        print("PASS 1: Saving individual plot files...")

        # --- PLOT 1: DRL Profit ---
        fig1, ax1 = plt.subplots(figsize=STD_FIGSIZE)
        ax1.plot(
            self.history["step"],
            self.history["drl_profit"],
            label="DRL Profit",
            color="blue",
        )
        ax1.set_title("DRL Agent Cumulative Profit Over Time")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Total Profit ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plot_1_drl_profit.png", dpi=STD_DPI)
        plt.close(fig1)  # Close the figure to free memory

        # --- PLOT 2: DRL Agent Inventory ---
        fig2, ax2 = plt.subplots(figsize=STD_FIGSIZE)
        ax2.plot(
            self.history["step"],
            self.history["drl_inventory"],
            label="DRL Inventory",
            color="green",
        )
        ax2.axhline(
            y=0, color="black", linestyle="--", linewidth=1, label="Neutral (0)"
        )
        ax2.set_title("DRL Agent Inventory Over Time")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Inventory (Units)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plot_2_drl_inventory.png", dpi=STD_DPI)
        plt.close(fig2)

        # --- PLOT 3: EGT Population Dynamics ---
        fig3, ax3 = plt.subplots(figsize=STD_FIGSIZE)
        egt_labels = [s.capitalize() for s in self.egt_strategies]
        egt_data = [self.history[f"egt_prop_{s}"] for s in self.egt_strategies]
        ax3.stackplot(self.history["step"], egt_data, labels=egt_labels)
        ax3.set_title("EGT Population Evolution (Replicator Dynamics)")
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("Proportion of Population")
        ax3.set_ylim(0, 1)
        ax3.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig("plot_3_egt_dynamics.png", dpi=STD_DPI)
        plt.close(fig3)

        # --- PLOT 4: DRL's Chosen Spread vs Epsilon ---
        fig4, ax4 = plt.subplots(figsize=STD_FIGSIZE)
        ax4.plot(
            self.history["step"],
            self.history["chosen_spread"],
            label="Chosen Spread",
            color="red",
            alpha=0.6,
        )
        ax4.set_title("DRL Agent's Chosen Spread")
        ax4.set_xlabel("Time Step")
        ax4.set_ylabel("Spread ($)")
        ax4.grid(True, alpha=0.3)
        ax4_twin = ax4.twinx()  # Twin axis for epsilon
        ax4_twin.plot(
            self.history["step"],
            self.history["epsilon"],
            label="Epsilon",
            color="grey",
            linestyle=":",
        )
        ax4_twin.set_ylabel("Epsilon")
        lines, labels = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines + lines2, labels + labels2, loc="upper right")
        plt.tight_layout()
        plt.savefig("plot_4_drl_spread.png", dpi=STD_DPI)
        plt.close(fig4)

        # --- PLOT 5: Rolling Average Reward ---
        fig5, ax5 = plt.subplots(figsize=STD_FIGSIZE)
        df_history["rolling_avg_reward"] = (
            df_history["reward"].rolling(window=100, min_periods=1).mean()
        )
        ax5.plot(
            self.history["step"],
            df_history["rolling_avg_reward"],
            label="Rolling Avg. Reward (100 steps)",
            color="purple",
        )
        ax5.axhline(y=0, color="black", linestyle="--", linewidth=1)
        ax5.set_title("DRL Agent's Progressive Learning (Rolling Reward)")
        ax5.set_xlabel("Time Step")
        ax5.set_ylabel("Average Reward")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plot_5_rolling_reward.png", dpi=STD_DPI)
        plt.close(fig5)

        # --- PLOT 6: DRL Learned Policy Distribution ---
        fig6, ax6 = plt.subplots(figsize=STD_FIGSIZE)
        halfway_idx = len(self.history["step"]) // 2
        exploiting_spreads = self.history["chosen_spread"][halfway_idx:]

        if exploiting_spreads:
            spread_counts = pd.Series(exploiting_spreads).value_counts().sort_index()
            spread_dist = spread_counts / len(exploiting_spreads)
            spread_dist.plot(kind="bar", ax=ax6, color="teal")
            ax6.set_title(f"DRL Learned Policy (Last {len(exploiting_spreads)} Steps)")
        else:
            ax6.set_title("DRL Learned Policy (No data)")
        ax6.set_xlabel("Chosen Spread ($)")
        ax6.set_ylabel("Proportion of Actions")
        ax6.set_ylim(0, 1)
        ax6.tick_params(axis="x", rotation=0)
        ax6.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("plot_6_policy_dist.png", dpi=STD_DPI)
        plt.close(fig6)

        print(f"Successfully saved 6 plots (e.g., 'plot_1_drl_profit.png').")

        # ---
        # --- PASS 2: CREATE COMBINED POP-UP WINDOW ---
        # ---
        print("PASS 2: Generating combined plot window...")
        fig_display, axes = plt.subplots(3, 2, figsize=(16, 15))
        fig_display.suptitle("DRL Agent vs EGT Population Simulation", fontsize=16)

        # --- Plot 1 (in grid) ---
        ax_1_0 = axes[0, 0]
        ax_1_0.plot(
            self.history["step"],
            self.history["drl_profit"],
            label="DRL Profit",
            color="blue",
        )
        ax_1_0.set_title("DRL Agent Cumulative Profit Over Time")
        ax_1_0.set_xlabel("Time Step")
        ax_1_0.set_ylabel("Total Profit ($)")
        ax_1_0.legend()
        ax_1_0.grid(True, alpha=0.3)

        # --- Plot 2 (in grid) ---
        ax_0_1 = axes[0, 1]
        ax_0_1.plot(
            self.history["step"],
            self.history["drl_inventory"],
            label="DRL Inventory",
            color="green",
        )
        ax_0_1.axhline(
            y=0, color="black", linestyle="--", linewidth=1, label="Neutral (0)"
        )
        ax_0_1.set_title("DRL Agent Inventory Over Time")
        ax_0_1.set_xlabel("Time Step")
        ax_0_1.set_ylabel("Inventory (Units)")
        ax_0_1.legend()
        ax_0_1.grid(True, alpha=0.3)

        # --- Plot 3 (in grid) ---
        ax_1_0 = axes[1, 0]
        ax_1_0.stackplot(self.history["step"], egt_data, labels=egt_labels)
        ax_1_0.set_title("EGT Population Evolution (Replicator Dynamics)")
        ax_1_0.set_xlabel("Time Step")
        ax_1_0.set_ylabel("Proportion of Population")
        ax_1_0.set_ylim(0, 1)
        ax_1_0.legend(loc="upper left")

        # --- Plot 4 (in grid) ---
        ax_1_1 = axes[1, 1]
        ax_1_1.plot(
            self.history["step"],
            self.history["chosen_spread"],
            label="Chosen Spread",
            color="red",
            alpha=0.6,
        )
        ax_1_1.set_title("DRL Agent's Chosen Spread")
        ax_1_1.set_xlabel("Time Step")
        ax_1_1.set_ylabel("Spread ($)")
        ax_1_1.grid(True, alpha=0.3)
        ax_1_1_twin = ax_1_1.twinx()
        ax_1_1_twin.plot(
            self.history["step"],
            self.history["epsilon"],
            label="Epsilon",
            color="grey",
            linestyle=":",
        )
        ax_1_1_twin.set_ylabel("Epsilon")
        lines, labels = ax_1_1.get_legend_handles_labels()
        lines2, labels2 = ax_1_1_twin.get_legend_handles_labels()
        ax_1_1.legend(lines + lines2, labels + labels2, loc="upper right")

        # --- Plot 5 (in grid) ---
        ax_2_0 = axes[2, 0]
        ax_2_0.plot(
            self.history["step"],
            df_history["rolling_avg_reward"],
            label="Rolling Avg. Reward (100 steps)",
            color="purple",
        )
        ax_2_0.axhline(y=0, color="black", linestyle="--", linewidth=1)
        ax_2_0.set_title("DRL Agent's Progressive Learning (Rolling Reward)")
        ax_2_0.set_xlabel("Time Step")
        ax_2_0.set_ylabel("Average Reward")
        ax_2_0.legend()
        ax_2_0.grid(True, alpha=0.3)

        # --- Plot 6 (in grid) ---
        ax_2_1 = axes[2, 1]
        if exploiting_spreads:
            spread_counts = pd.Series(exploiting_spreads).value_counts().sort_index()
            spread_dist = spread_counts / len(exploiting_spreads)
            spread_dist.plot(kind="bar", ax=ax_2_1, color="teal")
            ax_2_1.set_title(
                f"DRL Learned Policy (Last {len(exploiting_spreads)} Steps)"
            )
        else:
            ax_2_1.set_title("DRL Learned Policy (No data)")
        ax_2_1.set_xlabel("Chosen Spread ($)")
        ax_2_1.set_ylabel("Proportion of Actions")
        ax_2_1.set_ylim(0, 1)
        ax_2_1.tick_params(axis="x", rotation=0)
        ax_2_1.grid(axis="y", linestyle="--", alpha=0.5)

        # --- Show the combined window ---
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        print("Displaying combined plot window...")
        plt.show()


if __name__ == "__main__":
    # Define the hyperparameters for the DRL agent
    # These are good defaults, but for a real paper, you would tune them
    drl_params = {
        "gamma": 0.95,
        "epsilon_decay": 0.995,
        "learning_rate": 0.001,
        "replay_buffer_maxlen": 2000,
        "batch_size": 32,
        "epsilon_min": 0.01,
    }

    # Define the Neural Network structure
    nn_architecture = [
        {"units": 32, "activation": "relu"},
        {"units": 32, "activation": "relu"},
    ]

    # Run the simulation
    print("Running simulation with default configuration...")
    market_sim = Market(nn_architecture=nn_architecture, drl_params=drl_params)
    market_sim.run_simulation()
