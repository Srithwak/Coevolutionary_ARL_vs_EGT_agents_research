import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import random
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

class MarketData:
    """
    Loads and provides market data from the CSV file.
    (This class is unchanged from your original)
    """
    def __init__(self, filename="market_data.csv", volatility_window=20):
        if not os.path.exists(filename):
            print(f"Error: '{filename}' not found.")
            print("Please run 'generate_data.py' first to create the market data file.")
            exit()
            
        self.data = pd.read_csv(filename)
        self.max_steps = len(self.data)
        self.volatility_window = volatility_window
        
        # Pre-calculate volatility
        self.data['returns'] = self.data['mid_price'].pct_change()
        self.data['volatility'] = self.data['returns'].rolling(window=volatility_window).std().fillna(0)
        
    def get_price(self, step):
        if step >= self.max_steps:
            return None
        return self.data.loc[step, 'mid_price']
        
    def get_volatility(self, step):
        if step >= self.max_steps:
            return None
        return self.data.loc[step, 'volatility']

class DQNAgent:
    """
    Deep Q-Network (DRL) Agent, as described in the paper.
    This agent learns to quote the optimal spread using a neural network.
    """
    def __init__(self, agent_id, state_size, action_size):
        self.id = agent_id
        
        # Agent's state (continuous, not discrete)
        self.inventory = 0
        self.cash = 0
        self.state_size = state_size # e.g., [inventory, volatility]
        
        # Actions are the *spreads* it can quote
        self.actions = [0.1, 0.2, 0.4] # Narrow, Medium, Wide
        self.action_size = action_size
        
        # DQN parameters
        self.replay_buffer = deque(maxlen=2000) # Experience Replay
        self.gamma = 0.95    # Discount factor
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.batch_size = 32
        
        # Build two models: one for acting, one for targets
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """Builds the deep neural network (as described in the paper)."""
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear')) # Outputs Q-values
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Copies weights from the main model to the target model."""
        self.target_model.set_weights(self.model.get_weights())

    def get_state(self, volatility):
        """
        Returns the continuous state vector for the neural network.
        This matches the "high-dimensional state" concept.
        """
        # We normalize inventory to keep it in a reasonable range for the NN
        norm_inventory = np.tanh(self.inventory / 10.0) # Scale inventory
        return np.array([[norm_inventory, volatility]])

    def remember(self, state, action_idx, reward, next_state, done):
        """Stores an experience in the replay buffer."""
        self.replay_buffer.append((state, action_idx, reward, next_state, done))

    def choose_action(self, state):
        """Chooses an action (a spread) using epsilon-greedy."""
        if random.random() < self.epsilon:
            # Explore
            return random.choice(range(self.action_size))
        
        # Exploit: Ask the model for the best action
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        """Trains the neural network using a batch of stored experiences."""
        if len(self.replay_buffer) < self.batch_size:
            return
            
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        
        states = np.squeeze(np.array([item[0] for item in minibatch]))
        actions = np.array([item[1] for item in minibatch])
        rewards = np.array([item[2] for item in minibatch])
        next_states = np.squeeze(np.array([item[3] for item in minibatch]))
        dones = np.array([item[4] for item in minibatch])

        # Get Q-values for the next states from the *target* model
        target_q_next = self.target_model.predict(next_states, verbose=0)
        
        # Calculate the target Q-value: R + gamma * max(Q_target(s', a'))
        targets = rewards + self.gamma * np.amax(target_q_next, axis=1) * (1 - dones)
        
        # Get current Q-values from the *main* model
        current_q = self.model.predict(states, verbose=0)
        
        # Update the Q-values for the actions that were actually taken
        for i in range(self.batch_size):
            current_q[i, actions[i]] = targets[i]
            
        # Train the main model
        self.model.fit(states, current_q, epochs=1, verbose=0)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class Market:
    """
    Simulated market environment.
    This class now manages the DRL agent AND the EGT population dynamics.
    """
    def __init__(self):
        self.market_data = MarketData()
        self.step = 0
        
        # --- DRL Agent Setup ---
        self.state_size = 2 # [inventory, volatility]
        self.action_size = 3 # [narrow, medium, wide]
        self.drl_agent = DQNAgent(agent_id=0, 
                                  state_size=self.state_size, 
                                  action_size=self.action_size)
        self.UPDATE_TARGET_EVERY = 20 # Steps to update target network
        
        # --- EGT Population Setup ---
        self.egt_strategies = ['aggressive', 'passive', 'random']
        
        # This is the core of EGT: the *proportion* of each strategy
        self.egt_proportions = np.array([1/3, 1/3, 1/3])
        
        # Store cumulative payoffs to calculate fitness
        self.egt_total_payoffs = np.zeros(len(self.egt_strategies))
        self.egt_total_trades = np.zeros(len(self.egt_strategies))
        
        # How many EGT agents trade per step
        self.N_EGT_AGENTS = 15 
        
        # How often the EGT population evolves (replicator dynamics)
        self.EVOLVE_EVERY = 50 
        
        # --- History Tracking ---
        self.history = {
            'step': [],
            'mid_price': [],
            'drl_profit': [],
            'drl_inventory': [],
            'chosen_spread': [],
            'epsilon': [],
            'egt_prop_aggressive': [],
            'egt_prop_passive': [],
            'egt_prop_random': [],
        }
        
        self.last_drl_value = 0

    def get_egt_action(self, strategy_name, mid_price):
        """
        Gets a single action from an EGT agent based on its strategy.
        (Logic from the old EGTAgent class)
        """
        action_type = random.choice(['buy', 'sell'])
        
        if strategy_name == 'aggressive':
            price_offset = random.uniform(0.01, 0.05)
        elif strategy_name == 'passive':
            price_offset = random.uniform(0.3, 0.6)
        else: # 'random'
            price_offset = random.uniform(0.01, 0.5)
            
        if action_type == 'buy':
            return 'buy', mid_price + price_offset
        else: # 'sell'
            return 'sell', mid_price - price_offset

    def evolve_population(self):
        """
        Applies the Replicator Dynamic to evolve the EGT population.
        This is the core EGT mechanism described in the paper.
        """
        # Fitness = average payoff per trade
        fitness = self.egt_total_payoffs / (self.egt_total_trades + 1e-6)
        
        # Replicator dynamics requires non-negative fitness.
        # We shift all fitness values to be positive.
        min_fitness = np.min(fitness)
        positive_fitness = fitness - min_fitness + 1 
        
        # Calculate average fitness of the whole population
        avg_fitness = np.dot(self.egt_proportions, positive_fitness)
        
        if avg_fitness == 0:
            return # No trades, no evolution

        # The Replicator Equation: p_i' = p_i * (f_i / avg_f)
        self.egt_proportions = self.egt_proportions * (positive_fitness / avg_fitness)
        
        # Re-normalize to ensure proportions sum to 1
        self.egt_proportions /= np.sum(self.egt_proportions)

        # Reset payoffs for the next evolution cycle
        self.egt_total_payoffs.fill(0)
        self.egt_total_trades.fill(0)

    def run_step(self):
        """Execute one time step of the market."""
        mid_price = self.market_data.get_price(self.step)
        volatility = self.market_data.get_volatility(self.step)
        
        if mid_price is None:
            return False # Simulation over
            
        # 1. DRL AGENT'S TURN (Set Quotes)
        state = self.drl_agent.get_state(volatility)
        action_idx = self.drl_agent.choose_action(state)
        chosen_spread = self.drl_agent.actions[action_idx]
        
        drl_bid_price = mid_price - chosen_spread
        drl_ask_price = mid_price + chosen_spread
        
        # 2. EGT AGENTS' TURN (Trade against DRL)
        
        # We sample N agents from the population based on current proportions
        step_payoffs = np.zeros(len(self.egt_strategies))
        step_trades = np.zeros(len(self.egt_strategies))
        
        sampled_agents = np.random.choice(
            self.egt_strategies, 
            size=self.N_EGT_AGENTS, 
            p=self.egt_proportions
        )
        
        for strategy_name in sampled_agents:
            egt_action, egt_price = self.get_egt_action(strategy_name, mid_price)
            strat_idx = self.egt_strategies.index(strategy_name)
            
            if egt_action == 'sell' and egt_price <= drl_bid_price:
                # EGT agent SELLS, DRL agent BUYS
                self.drl_agent.inventory += 1
                self.drl_agent.cash -= drl_bid_price
                step_payoffs[strat_idx] += drl_bid_price # EGT profit
                step_trades[strat_idx] += 1
                
            elif egt_action == 'buy' and egt_price >= drl_ask_price:
                # EGT agent BUYS, DRL agent SELLS
                self.drl_agent.inventory -= 1
                self.drl_agent.cash += drl_ask_price
                step_payoffs[strat_idx] -= drl_ask_price # EGT cash goes down
                step_trades[strat_idx] += 1
        
        # Add this step's results to the EGT totals
        self.egt_total_payoffs += step_payoffs
        self.egt_total_trades += step_trades
        
        # 3. DRL AGENT LEARNS (Calculate Reward & Replay)
        
        # Reward is change in portfolio value (Mark-to-Market)
        current_portfolio_value = self.drl_agent.cash + (self.drl_agent.inventory * mid_price)
        reward = current_portfolio_value - self.last_drl_value
        
        # Inventory penalty (as before)
        inventory_penalty = (self.drl_agent.inventory ** 2) * 0.01 
        reward -= inventory_penalty
        
        # Get next state
        next_mid_price = self.market_data.get_price(self.step + 1)
        next_volatility = self.market_data.get_volatility(self.step + 1)
        
        done = (next_mid_price is None)
        if done:
            next_state = self.drl_agent.get_state(volatility) # Use current as placeholder
        else:
            next_state = self.drl_agent.get_state(next_volatility)
            
        # Store this experience in the replay buffer
        self.drl_agent.remember(state, action_idx, reward, next_state, done)
        
        # Update value for next step's reward calculation
        self.last_drl_value = current_portfolio_value
        
        # Train the DRL agent from its replay buffer
        self.drl_agent.replay()
        
        # 4. EVOLVE EGT POPULATION (if it's time)
        if self.step % self.EVOLVE_EVERY == 0:
            self.evolve_population()

        # 5. UPDATE DRL TARGET NETWORK (if it's time)
        if self.step % self.UPDATE_TARGET_EVERY == 0:
            self.drl_agent.update_target_model()
            
        # 6. RECORD HISTORY
        self.history['step'].append(self.step)
        self.history['mid_price'].append(mid_price)
        self.history['drl_profit'].append(current_portfolio_value)
        self.history['drl_inventory'].append(self.drl_agent.inventory)
        self.history['chosen_spread'].append(chosen_spread)
        self.history['epsilon'].append(self.drl_agent.epsilon)
        self.history['egt_prop_aggressive'].append(self.egt_proportions[0])
        self.history['egt_prop_passive'].append(self.egt_proportions[1])
        self.history['egt_prop_random'].append(self.egt_proportions[2])
        
        self.step += 1
        return True

    def run_simulation(self):
        """Run the full simulation."""
        print(f"Starting simulation: 1 DRL agent vs Evolving EGT population.")
        print(f"Data steps: {self.market_data.max_steps}")
        
        while self.run_step():
            if self.step % 100 == 0:
                print(f"Step {self.step}/{self.market_data.max_steps}...")
                print(f"  EGT Proportions: "
                      f"Agg: {self.egt_proportions[0]:.2f}, "
                      f"Pas: {self.egt_proportions[1]:.2f}, "
                      f"Ran: {self.egt_proportions[2]:.2f}")
        
        print("Simulation complete.")
        self.print_summary()
        self.plot_results()

    def print_summary(self):
        """Print final summary statistics."""
        final_drl_profit = self.history['drl_profit'][-1]
        
        print("\n" + "="*30)
        print("=== FINAL RESULTS ===")
        print(f"DRL Agent Final Profit: ${final_drl_profit:,.2f}")
        
        print("\nFinal EGT Population Distribution:")
        print(f"  Aggressive: {self.egt_proportions[0]:.3f}")
        print(f"  Passive:    {self.egt_proportions[1]:.3f}")
        print(f"  Random:     {self.egt_proportions[2]:.3f}")
        print("="*30 + "\n")

    def plot_results(self):
        """Visualize simulation results with 4 plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('DRL Agent vs EGT Population Simulation', fontsize=16)
        
        # Plot 1: DRL Profit
        ax1 = axes[0, 0]
        ax1.plot(self.history['step'], self.history['drl_profit'], 
                     label='DRL Agent Profit', color='blue', linewidth=2)
        ax1.set_title('DRL Agent Cumulative Profit Over Time')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Total Profit ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: DRL Agent Inventory
        ax2 = axes[0, 1]
        ax2.plot(self.history['step'], self.history['drl_inventory'], 
                     label='DRL Inventory', color='green')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Neutral (0)')
        ax2.set_title('DRL Agent Inventory Over Time')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Inventory (Units)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: EGT Population Dynamics (THIS IS THE NEW, KEY PLOT)
        ax3 = axes[1, 0]
        ax3.stackplot(self.history['step'],
                      self.history['egt_prop_aggressive'],
                      self.history['egt_prop_passive'],
                      self.history['egt_prop_random'],
                      labels=['Aggressive', 'Passive', 'Random'],
                      colors=['#FF5733', '#33FF57', '#3357FF'])
        ax3.set_title('EGT Population Evolution (Replicator Dynamics)')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Proportion of Population')
        ax3.set_ylim(0, 1)
        ax3.legend(loc='upper left')
        
        # Plot 4: DRL's Chosen Spread vs Epsilon
        ax4 = axes[1, 1]
        ax4.plot(self.history['step'], self.history['chosen_spread'], 
                     label='Chosen Spread', color='red', alpha=0.6)
        ax4.set_title('DRL Agent\'s Chosen Spread')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Spread ($)')
        ax4.grid(True, alpha=0.3)
        
        ax4_twin = ax4.twinx()
        ax4_twin.plot(self.history['step'], self.history['epsilon'], 
                          label='Epsilon (Exploration)', color='grey', linestyle=':', alpha=0.8)
        ax4_twin.set_ylabel('Epsilon')
        
        lines, labels = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines + lines2, labels + labels2, loc='upper right')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('drl_vs_egt_results.png', dpi=300)
        print("Plots saved as 'drl_vs_egt_results.png'")
        plt.show()

if __name__ == "__main__":
    # Make sure 'market_data.csv' exists before running.
    market = Market()
    market.run_simulation()