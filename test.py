import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd # Added pandas for data handling in plotting

# --- 1. Simulation Configuration ---

# Payoff matrix for the Prisoner's Dilemma (Market Simulation)
# (My_Payoff, Opponent_Payoff)
PAYOFF_MATRIX = {
    'C': {  # My Action: Cooperate
        'C': (3, 3),  # Opponent: Cooperate (Mutual Cooperation)
        'D': (0, 5)   # Opponent: Defect (Sucker Payoff)
    },
    'D': {  # My Action: Defect
        'C': (5, 0),  # Opponent: Cooperate (Temptation to Defect)
        'D': (1, 1)   # Opponent: Defect (Mutual Defection)
    }
}
ACTIONS = ['C', 'D']
STRATEGIES = ['Always_Cooperate', 'Always_Defect', 'Tit_for_Tat', 'Random']

# --- 2. Fixed-Strategy EGT Agent Class ---

class EGTAgent:
    """
    Represents an agent with a fixed strategy from Evolutionary Game Theory.
    The strategy is set at initialization and does not change.
    """
    def __init__(self, strategy):
        if strategy not in STRATEGIES:
            raise ValueError(f"Invalid strategy: {strategy}")
        self.strategy = strategy
        # Note: last_opponent_action is now managed by the simulation memory

    def choose_action(self, opponent_last_action):
        """
        Chooses an action based on the fixed strategy.
        'opponent_last_action' is needed for Tit-for-Tat.
        """
        if self.strategy == 'Always_Cooperate':
            return 'C'
        elif self.strategy == 'Always_Defect':
            return 'D'
        elif self.strategy == 'Tit_for_Tat':
            # Cooperate on the first move, then copy opponent's last move
            if opponent_last_action is None:
                return 'C'
            else:
                return opponent_last_action
        elif self.strategy == 'Random':
            return random.choice(ACTIONS)

    def __repr__(self):
        return f"EGTAgent(strategy='{self.strategy}')"

# --- 3. Adaptive Reinforcement Learning (ARL) Agent Class ---

class ARLAgent:
    """
    Represents the self-learning "mutant" agent using Q-learning.
    It learns to play against different opponent types.
    """
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.9999, min_exploration_rate=0.01):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = min_exploration_rate
        
        # The Q-table.
        # State: Opponent's last action ('Start', 'C', 'D')
        # Action: My next action ('C', 'D')
        # Value: Expected future reward
        # We use defaultdict for convenience, so new states are automatically initialized with 0.0 values.
        self.q_table = defaultdict(lambda: defaultdict(float))

    def get_state(self, opponent_last_action):
        """
        Determines the current state based on the opponent's last move.
        """
        if opponent_last_action is None:
            return 'Start'
        return opponent_last_action

    def choose_action(self, state):
        """
        Chooses an action using an epsilon-greedy strategy.
        """
        # Exploration: Choose a random action
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        
        # Exploitation: Choose the best-known action for the current state
        else:
            # If all Q-values for this state are the same (e.g., all 0),
            # pick randomly to break ties.
            q_values = self.q_table[state]
            if not q_values or all(v == list(q_values.values())[0] for v in q_values.values()):
                return random.choice(self.actions)
            
            # Otherwise, pick the action with the highest Q-value
            return max(q_values, key=q_values.get)

    def update_q_table(self, state, action, reward, next_state):
        """
        Updates the Q-table using the Q-learning update rule.
        """
        # Get the Q-value for the current state-action pair
        old_value = self.q_table[state][action]
        
        # Get the maximum Q-value for the next state
        # If next_state is new, its Q-values will be 0.0 by default
        next_max = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
        
        # Q-learning formula:
        # Q(s,a) = (1-α) * Q(s,a) + α * (r + γ * max_a' Q(s',a'))
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        
        self.q_table[state][action] = new_value

    def decay_exploration(self):
        """
        Decays the exploration rate (epsilon) over time.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- 4. Main Simulation ---

def run_simulation(num_generations=100, 
                   rounds_per_generation=1000, 
                   num_egt_agents=100, 
                   mutation_rate=0.2): # <-- CHANGED: Increased mutation rate
    
    # --- Initialization ---
    
    # 1. Create the ARL "mutant" agent
    arl_agent = ARLAgent(actions=ACTIONS)
    
    # 2. Create the initial EGT population
    population = []
    for i in range(num_egt_agents):
        population.append(EGTAgent(strategy=STRATEGIES[i % len(STRATEGIES)]))
    random.shuffle(population)
    
    # 3. History trackers (for plotting)
    arl_payoff_history = []
    egt_payoff_history = []
    population_composition_history = []
    
    print(f"Starting simulation...")
    print(f"ARL Agent (Mutant) vs. Evolving Population of {num_egt_agents} EGT Agents.")
    print(f"{num_generations} generations, {rounds_per_generation} rounds per generation.")
    print(f"Mutation Rate: {mutation_rate}")
    print("-" * 30)

    # --- Generation Loop (Outer Loop) ---
    for gen in range(1, num_generations + 1):
        
        # 1. Track population composition for this generation
        comp_counts = defaultdict(int)
        for agent in population:
            comp_counts[agent.strategy] += 1
        population_composition_history.append(comp_counts)
        
        # 2. Reset scores and memories for the new generation
        egt_payoffs_this_generation = np.zeros(num_egt_agents)
        arl_payoff_this_generation = 0
        
        # Memories for Tit-for-Tat and ARL state
        arl_memory = [None] * num_egt_agents
        egt_memory = [None] * num_egt_agents
        
        total_egt_payoff = 0 # EGT total payoff in this gen

        # --- Interaction Loop (Inner Loop) ---
        for _ in range(rounds_per_generation):
            
            # 1. Pick a random EGT agent to play against
            opponent_index = random.randint(0, num_egt_agents - 1)
            opponent = population[opponent_index]
            
            # 2. Get the current state for each agent
            arl_state = arl_agent.get_state(egt_memory[opponent_index])
            egt_state = arl_memory[opponent_index] # For Tit-for-Tat

            # 3. Agents choose their actions
            arl_action = arl_agent.choose_action(arl_state)
            egt_action = opponent.choose_action(egt_state)
            
            # 4. Get payoffs from the matrix
            arl_payoff, egt_payoff = PAYOFF_MATRIX[arl_action][egt_action]
            
            # 5. Update scores for this generation
            arl_payoff_this_generation += arl_payoff
            egt_payoffs_this_generation[opponent_index] += egt_payoff
            total_egt_payoff += egt_payoff
            
            # 6. ARL agent learns
            next_arl_state = arl_agent.get_state(egt_action)
            arl_agent.update_q_table(arl_state, arl_action, arl_payoff, next_arl_state)
            
            # 7. Update memories for this specific interaction
            arl_memory[opponent_index] = arl_action
            egt_memory[opponent_index] = egt_action
            
            # 8. Decay exploration rate
            arl_agent.decay_exploration()
        
        # --- End of Generation: Evolution Step ---

        # 1. Record average payoffs for this generation
        arl_avg = arl_payoff_this_generation / rounds_per_generation
        egt_avg = total_egt_payoff / rounds_per_generation
        arl_payoff_history.append(arl_avg)
        egt_payoff_history.append(egt_avg)

        # 2. Create the next generation using tournament selection
        new_population = []
        for _ in range(num_egt_agents):
            
            # Check for mutation
            if random.random() < mutation_rate:
                new_population.append(EGTAgent(strategy=random.choice(STRATEGIES)))
            
            # Tournament Selection
            else:
                # Pick two random "parents"
                p1_idx = random.randint(0, num_egt_agents - 1)
                p2_idx = random.randint(0, num_egt_agents - 1)
                
                # Get their fitness (payoff)
                p1_fitness = egt_payoffs_this_generation[p1_idx]
                p2_fitness = egt_payoffs_this_generation[p2_idx]
                
                # The winner (with higher fitness) reproduces
                if p1_fitness > p2_fitness:
                    winner_strategy = population[p1_idx].strategy
                else:
                    winner_strategy = population[p2_idx].strategy
                    
                new_population.append(EGTAgent(strategy=winner_strategy))
        
        # The new generation replaces the old one
        population = new_population

        if gen % (num_generations // 10) == 0 or gen == 1:
            print(f"Generation {gen}/{num_generations}...")
            print(f"  ARL Epsilon: {arl_agent.epsilon:.4f}")
            print(f"  ARL Avg Payoff (Gen): {arl_avg:.3f}")
            print(f"  EGT Avg Payoff (Gen): {egt_avg:.3f}")
            print(f"  Population: {dict(comp_counts)}")

    # --- 5. Final Results ---
    
    print("-" * 30)
    print("Simulation Complete.")
    print(f"Final ARL Agent Avg Payoff: {np.mean(arl_payoff_history[-10:]):.4f} (Avg. of last 10 gens)")
    print(f"Final EGT Population Avg Payoff: {np.mean(egt_payoff_history[-10:]):.4f} (Avg. of last 10 gens)")
    
    # Print the learned Q-table
    print("\n--- ARL Agent's Learned Q-Table ---")
    print("(State: Opponent's Last Move)")
    print(f"{'State':<7} | {'Action C':<10} | {'Action D':<10}")
    print("-" * 30)
    for state in ['Start', 'C', 'D']:
        c_val = arl_agent.q_table[state]['C']
        d_val = arl_agent.q_table[state]['D']
        print(f"{state:<7} | {c_val:<10.4f} | {d_val:<10.4f}")
        
    # --- 6. Plotting ---
    
    # Plot 1: Payoff Comparison
    plt.figure(figsize=(12, 7))
    plt.plot(arl_payoff_history, label="ARL 'Mutant' Agent Avg. Payoff", color='blue', linewidth=2)
    plt.plot(egt_payoff_history, label="EGT Population Avg. Payoff", color='red', linestyle='--', linewidth=2)
    plt.title("ARL 'Mutant' vs. Evolving EGT Population (Payoff per Generation)")
    plt.xlabel("Generation")
    plt.ylabel("Average Payoff per Generation")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    plot1_filename = "arl_vs_egt_payoff.png"
    plt.savefig(plot1_filename)
    print(f"\nPayoff plot saved as '{plot1_filename}'")
    plt.show()

    # Plot 2: Population Dynamics
    pop_df = pd.DataFrame(population_composition_history).fillna(0)
    # Ensure all strategies are columns, even if one dies out
    for strategy in STRATEGIES:
        if strategy not in pop_df.columns:
            pop_df[strategy] = 0
    pop_df = pop_df[STRATEGIES] # Re-order columns for consistency
    
    plt.figure(figsize=(12, 7))
    # Create stacked area plot
    plt.stackplot(pop_df.index, 
                  [pop_df[strategy] for strategy in STRATEGIES], 
                  labels=STRATEGIES,
                  alpha=0.8)
    
    plt.title("EGT Population Dynamics Over Generations")
    plt.xlabel("Generation")
    plt.ylabel(f"Number of Agents (out of {num_egt_agents})")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()

    plot2_filename = "arl_vs_egt_population.png"
    plt.savefig(plot2_filename)
    print(f"Population plot saved as '{plot2_filename}'")
    plt.show()


# --- Run the main function ---
if __name__ == "__main__":
    run_simulation(num_generations=100, 
                   rounds_per_generation=1000, 
                   num_egt_agents=100, 
                   mutation_rate=0.2) # <-- CHANGED: Increased mutation rate

