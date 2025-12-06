## Slide 1

Hi we're Rithwak and Praneeth, and this is our research project.

## Slide 2

**Our detailed research question is:**
> How does an Adversarial Reinforcement Learning agent perform and adapt against an evolving population of heuristic traders governed by Evolutionary Game Theory?

Most reinforcement learning benchmarks today operate in static environments. The environment might be difficult, but it doesn’t fight back or evolve. So we made an environment where the population affects the environment and evolves over time.

We used the stock market as a playground to test: **strategic reasoning**, **risk management**, and **multi agent interactions**.

Our goal isn't just to make a profitable agent, but to understand how it learns and adapts to an evolving market.

## Slide 3

**Our experimental setup is as follows:**
The agents trade in an order book market, with real OHLCV data. The trades happen when the quotes from agents cross each other.
Because there is randomness in the environment, we included a seed feature for reproducibility.
The agents goal is to maximize their portfolio value.
We used Python, TensorFlow/Keras, NumPy & Pandas, SciPy, and Matplotlib.

## Slide 4: Previous Iterations

We tried using a q table first, treating volatility as high and low, and rounding inventory to the nearest integer. This didn't work well, as it couldn't grasp the complexity of the market.

Next we tried using a deep Q network, but it didn't work well either, as the agent just kept hoarding stocks hoping for a price increase.

## Slide 5: Agent Architecture

The agent is a feed forward deep neural network. The multi layer perceptron is configured in a YAML file for easy architecture changes.
It observes the **volatility**, **momentum**, **normalized volume**, **RSI**, **inventory**, and **time remaining** in the simulation.
The output of the network is a bid/ask spread combination to quote into the market.
We made the model simple to capture interatction and adaptation.
We use **ReLU** instead of sigmoid because it handles non-linear patterns more simply and effectively.

## Slide 6: Learning Mechanism

It uses an **epsilon greedy policy** to explore the action space, where it first randomly explores the action space, but over time it learns and exploits the best actions.

After each step, the agent measures its error and adjusts weights by sending that error backward through the network. It uses the **adam optimzer** with gradient descent to update the weights and reduce the model's error.

The end result is that the agent starts clueless, but as it explores, it improves to make a profit.

## Slide 7: Adversary

The adversary is a population of **heuristic traders**. They are categorized into 4 types: **aggressive**, **passive**, **random**, and **momentum**. Each type has different trading strategies.

Every 50 steps, we use a fitness function to evaluate the population and select the best agents to reproduce and mutate. There is also a 0.1% mutation rate to ensure no stagnation and random noise exists.

## Slide 8: Interaction Loop

1. First the agent quotes a bid/ask spread into the market.
2. EGT agents send buy/sell orders to the market.
3. If a quote crosses, a trade occurs.
4. Profit, inventory, and risk are updated.
5. Population evolves once in 50 steps
6. The agent learns from the results and backpropagates.

## Slide 9: Results - Profitability

In this example the agent had over **0.7% profit**.
If we split the market data into trends, we can see that the agent struggled during times of new trends at first, but it adapts to make profitable trades.
The agent played it very safe, only making small trades only when it was sure of a profit.
We tested this with multiple seeds and market data, and most of the time, the agent made a profit.

## Slide 10: Results - Causality

The bottom right image shows the EGT population evolving over time with the ARL agent. The bottom left image shows the EGT population evolving with a random algorithm.

As you can see, the ARL agent **transformed the market**. When a population didnt work to its far, it skewed the market to change the popluation size. This is even more evident, as in between changes of population proportoins, the random population takes over, but that's only because the ARL agent can't affect the random population properly. The dependency plot shows that the AI actively changes the population, and when it is satisifed, there is a blank line because there is no change in population to measure.

## Slide 11: Results - Strategy

The top image shows the chosen spread width, and how many times the agent chose it.
The graph shows us that its learned policy favors a narrow band of spreads over all other options.
The bottom image models the spread width as a function of time.
The gray line shows how often the agent acts to learn/explore. At first, its closer to 1, but as time passes it gets to 0, showing that its not acting randomly, but on what it has learnt.
When compared to the market data, we can see that when the agent wants to trade, there are small spikes, and there are big spikes when there is a crash in the market, showing that the agent is actively trying to stop itself from being exploited.

## Slide 12: Results - Methodology

The histogram shows that the agent isn't lucky, it consistently made small wins which added up over time.
The graph shows the agent dipping and then returning to 0 (the flat line). This demonstrates resilience—when the agent made a mistake or the market turned against it, it adapted its policy to recover the loss rather than "tilting" and losing more.
The bright center line proves the agent learned to keep its inventory near zero, validating that it understands **risk management**. It makes money by trading flow, not by taking dangerous bets on the direction of the stock.

## Slide 13: Novelty

There have been previous research projects with:
* RL agent vs fixed environment or
* GA evolving strategies with no RL or
* multiple RL agents but no evolutionary dynamics or
* EGT populations but no deep learning agent.

We were not able to find any research that combined all of these elements.
Unlike static back testing tools, this framework features an **evolving trader population**, providing a dynamic environment for validating predictive models.

## Slide 14: Bias and Future Work

The model might overfit to the specific volatility patterns of the provided dataset.
A simple 2-layer MLP may limit expressiveness compared to deeper architectures.
Our Possible Future Work is to
* Implement **Recurrent Neural Networks (RNNs)** to capture time-series dependencies better.
* Replace heuristic EGT agents with other RL agents (**Multi-Agent Reinforcement Learning**) for a "smarter" enemy.

## Slide 15: Conclusion

We successfully built a simulation where a **Neural Network agent** fought and exploited against an evolving **Genetic Algorithm population**.
The agent didn't just memorize prices; it learned Safe AI behaviors (inventory management) and Strategic Adaptation (countering aggressive traders).
This proves that RL agents can survive and stabilize in non-stationary, adversarial environments, and even control the environment to accomplish its goal.
