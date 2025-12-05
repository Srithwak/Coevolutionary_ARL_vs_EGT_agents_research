# Presentation Script: ARL Agent vs. EGT Population
**Total Estimated Time:** ~15 Minutes

---

## Part 1: Context & Setup (Minutes 0:00 - 3:30)

### Slide 1: Title & Research Motivation
**(Time: 0:00 - 1:15)**

"Good morning/afternoon everyone. My research asks a fundamental question about the future of AI in finance: **'Can an Adaptive Reinforcement Learning (ARL) Agent Outperform Evolutionary Game Theory (EGT) Populations in a Simulated Market?'**

Let me explain why this matters. Financial markets are notoriously adversarial and non-stationary. However, most AI benchmarks today use static datasets—historical prices that don't react to the agent's decisions. It's like learning to play chess against a recording of a past game; you can't really lose, but you also don't learn how to handle a live opponent.

My research question challenges this paradigm: **Can a Deep Reinforcement Learning agent learn a stable, profitable policy when its opponents—the market population—are constantly evolving specifically to defeat it?**

The core concept here is that we are treating the market not just as a time-series prediction problem, but as a complex stochastic environment designed to stress-test AI reasoning and adaptation."

### Slide 2: Experimental Setup & Data
**(Time: 1:15 - 2:30)**

"To test this, I built a simulated order-book market where agents interact via 'crossing quotes'. This isn't just a replay of history; it's an interactive arena.

For our data source, I used real OHLCV data. This provides the baseline price, volatility, and momentum that grounds our simulation in the real world.

The agent itself observes a state space of 6 key features:
1.  **Volatility:** How wild are the price swings?
2.  **RSI (Relative Strength Index):** Is the asset overbought or oversold?
3.  **Momentum:** What's the trend strength?
4.  **Normalized Volume:** Is there liquidity?
5.  **Inventory:** How much stock am I currently holding?
6.  **Time Remaining:** How long until the session ends?

The agent's goal is to maximize a **Hybrid Reward Function**. It's not just about raw profit. It sums realized Profit, incentives for capturing the Spread, and crucial penalties for holding too much Risk (inventory)."

### Slide 3: Tools & Libraries (Implementation Stack)
**(Time: 2:30 - 3:30)**

"Briefly, let's look at the implementation stack. The entire simulation was built in **Python**.

For the brain of the agent, I used **TensorFlow and Keras** to build a Double Deep Q-Network (DDQN). This involves a sequential model with dense layers, optimized using Adam.

For the heavy lifting of data and math, I relied on **NumPy and Pandas** for high-performance matrix operations and handling the OHLCV structures. **SciPy** was used for calculating statistical moments like skew and kurtosis to give the agent a deeper sense of market shape.

Finally, all the visualizations you'll see—the performance tracking and heatmaps—were generated using **Matplotlib**."

---

## Part 2: Methodology (Minutes 3:30 - 7:30)

### Slide 4: The Agent Architecture (Neural Networks)
**(Time: 3:30 - 4:30)**

Now, let's dive into the methodology.

First, the Agent Architecture. We utilize a **Multilayer Perceptron (MLP)**, which is a standard Feed-Forward Neural Network. Its job is to approximate Q-values—essentially predicting the expected future reward of taking a specific action in a specific state.

The architecture is straightforward but effective:
-   **Input Layer:** 6 neurons receiving our market features.
-   **Hidden Layers:** 64 units each, using **ReLU activation**.
-   **Output Layer:** 5 discrete actions representing different spread widths (from very tight to very wide).

Why did I choose **ReLU** (Rectified Linear Units)? In financial data, volatility dynamics are non-linear and can be extreme. ReLU handles this much better than sigmoid functions and helps us avoid the vanishing gradient problem during training."

### Slide 5: The Learning Mechanism (Backpropagation)
**(Time: 4:30 - 5:30)**

How does the agent actually learn? This brings us to **Gradient Descent and Backpropagation**.

The objective is to minimize the **Loss**, which is the squared error between what the agent *thought* would happen (predicted Q-value) and what *actually* happened (target Q-value derived from the Bellman equation).

After every interaction in the market, the agent calculates this error. We then **back-propagate** this error from the output layer back to the input layer to adjust the synaptic weights ($W_{j,i}$).

To perform this optimization, I used the **Adam optimizer**. It's a variant of Stochastic Gradient Descent that adapts the learning rate, helping us find the global minimum in the error surface more efficiently than standard SGD."

### Slide 6: The Adversary (Evolutionary Game Theory)
**(Time: 5:30 - 6:30)**

But an agent needs an opponent. In my simulation, the 'Adversary' is not a single entity. It is a population of heuristic traders governed by **Evolutionary Game Theory (EGT)**.

This population consists of four strategies:
1.  **Aggressive:** Always trying to trade.
2.  **Passive:** Waiting for perfect prices.
3.  **Momentum:** Chasing the trend.
4.  **Random:** Noise traders.

We use **Replicator Dynamics**, which functions exactly like a Genetic Algorithm.
-   **Fitness Function:** This is simply rolling profitability.
-   **Selection:** Strategies that make money reproduce and increase their share of the population. Strategies that lose money die out.
-   **Mutation:** I introduced a 2% mutation rate to add random noise, preventing the population from getting stuck in a local optimum."

### Slide 7: The Interaction Loop
**(Time: 6:30 - 7:30)**

Here is how the interaction loop works in practice:

1.  **Sensing:** The MLP observes the market state (Volatility, RSI, Inventory).
2.  **Action:** The Agent posts a Bid/Ask Spread. It might choose a 'Tight' spread to capture volume or a 'Wide' spread to stay safe.
3.  **Reaction:** The EGT population evaluates these prices. If the quotes cross their internal thresholds, trades execute.
4.  **Feedback:**
    *   The Agent receives a **Reward** (Profit minus Risk).
    *   The EGT Agents receive **Fitness** (their Profit).
5.  **Update:** Finally, the Agent updates its weights via Backpropagation, and the Population updates its distribution via Evolution.

This creates a continuous cycle of action, reaction, and adaptation."

---

## Part 3: Results (Minutes 7:30 - 11:30)

### Slide 8: Result 1 - Learning & Profitability
**(Time: 7:30 - 8:30)**

Let's look at the results. First, **Learning and Profitability**.

If you look at the 'Portfolio Value' plot, you can see the ARL agent achieved a total return of about **1.5%**. That might sound small, but in a high-frequency market making context over a short period, it's significant.

More importantly, look at the 'Rolling Reward' plot. Initially, it's noisy—the agent is exploring, making mistakes, and losing money. But then, you see a clear upward trend where it converges to a stable, positive-reward policy.

**Simple Explanation:** The AI learned how to make money. It started knowing nothing, learned from its mistakes via gradient descent, and ended up profitable."

### Slide 9: Result 2 - The Arms Race (Co-Evolution)
**(Time: 8:30 - 9:30)**

This is my favorite result: **The Arms Race**.

The 'EGT Dynamics' chart shows the population share over time.
-   **Start:** We begin with a mixed population.
-   **Middle:** You see the 'Aggressive' traders (often red or orange) trying to exploit the agent. But the agent counters this by widening its spreads, effectively refusing to trade on their terms.
-   **End:** The Aggressive traders go bankrupt and die out. The 'Passive' traders (the green area) take over **>99%** of the market.

**Simple Explanation:** The AI 'starved' the aggressive traders. By refusing to give them good prices, it shaped its own environment, leaving only the passive traders who were harmless."

### Slide 10: Result 3 - Learned Policy Behavior
**(Time: 9:30 - 10:30)**

What does this learned policy actually look like?

The 'Spread Width' plot shows distinct spikes (the red lines). These perfectly match periods of high volatility in the market. The agent learned that when the market is crazy, you quote wide to stay safe.

The 'Policy Distribution' chart shows a preference for tight spreads (actions 0.2 and 0.4) to capture volume during calm periods, but it retains those wide spreads for safety.

**Simple Explanation:** The AI learned **'Context-Dependent Tactics'**. It acts like a human market maker: tight spreads when it's calm, wide spreads when it's scared."

### Slide 11: Result 4 - Safety & Green AI
**(Time: 10:30 - 11:30)**

Finally, let's talk about **Safety and Efficiency**.

The 'Inventory Heatmap' shows a bright yellow center around zero. This means the agent learned to keep its inventory flat. The variance was extremely low (< 0.02).
The 'Max Drawdown' was also very small, indicating no catastrophic losses.

**Simple Explanation:** This is **Green AI**. The agent learned to be efficient. It didn't hoard assets, which is risky and capital-intensive. It maximized reward while minimizing resource usage, aligning with principles of safe and efficient AI deployment."

---

## Part 4: Discussion & Conclusion (Minutes 11:30 - 15:00)

### Slide 12: Novelty & Community Contribution
**(Time: 11:30 - 12:45)**

"Moving to the discussion. What is the **Novelty** here?
Most financial RL papers assume a static market. We demonstrated **Co-Evolution**—where the AI and the Market change each other simultaneously. This is a much harder, more realistic problem.

**Contribution:** I am providing this simulation as an open-source 'playground' for the community. It serves as a benchmark for measuring AI Robustness against adaptive adversaries. It's a tool for others to study multi-agent interactions without needing a supercomputer."

### Slide 13: Bias & Future Work
**(Time: 12:45 - 13:45)**

"Of course, no research is perfect. We must address **Bias**.
-   **Data Bias:** The model might overfit to the specific volatility patterns of the NVIDIA dataset I used. It might not work as well on a stable stock like Coca-Cola without retraining.
-   **Architecture Bias:** A simple 2-layer MLP is good, but it may limit expressiveness compared to deeper architectures like LSTMs or Transformers.

**Future Work:**
1.  **Deep Learning:** I'd like to implement Recurrent Neural Networks (RNNs) to capture time-series dependencies better.
2.  **MARL:** I want to replace the heuristic EGT agents with *other* RL agents (Multi-Agent Reinforcement Learning) to create a truly 'smart' enemy."

### Slide 14: Conclusion
**(Time: 13:45 - 15:00)**

"To conclude: We successfully built a simulation where a Neural Network agent fought against an evolving Genetic Algorithm population.

The **Key Takeaway** is that the agent didn't just memorize prices. It learned **Safe AI behaviors** like inventory management and **Strategic Adaptation** to counter aggressive traders.

**Final Thought:** This proves that Reinforcement Learning agents can survive and stabilize in non-stationary, adversarial environments. This is a crucial step for deploying AI in the real world, where things are never static and opponents are always adapting.

Thank you very much for your time. I'm happy to take any questions."
