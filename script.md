## Slide 1

Hi everyone this is our research project. We’ll be presenting our project to showcase our work.

## Slide 2

Our detailed research question is:

> "How does an Adversarial Reinforcement Learning agent perform and adapt against an evolving population of heuristic traders governed by Evolutionary Game Theory?"

This is important because most AI tests today are static. This is unlike the real environment, where trades you make affect the environment. It's a game played against other people.

If you find a winning strategy, other traders will notice. They will copy you, or they will find a way to counter you. Your edge disappears.

We wanted to build a simulation that captures this reality. We created a world where the background traders aren't just random noise—they are active players who push back. If our AI starts winning, they start changing.

We used the stock market to test three big things:
- Can the AI outsmart a crowd?
- Can it play the game without going broke? and
- How does it handle a market that is constantly shifting under its feet?

Our goal isn't just to make a bot that gets rich. Our goal is to understand how an AI learns to survive when the rules of the game are constantly changing.

## Slide 3

Here is how we set up the experiment:

We built a real stock market backtesting simulator.

We used a system similar to that of an auction. On one side, you have the **Bid**—that's the buyers saying, “I'll pay this much.” On the other side, you have the **Ask**—that's the sellers saying, “I want at least this much.” 

The gap in the middle? That's the **Spread**.

Now, for a trade to actually happen, someone has to guess that the price will jump to more than what they will buy it for, resulting in a profit. They have to cross that gap and accept the other person's price.

In our simulation, our AI has to decide: Does it want to sit back and wait for a bargain? Or does it want to cross the spread and make a deal happen right now?

Because there are thousands of little agents running around, there's a lot of randomness. So we made sure to use a “seed” feature. This basically lets us replay the same simulation, to prove our results aren't just luck.

The goal for every agent is simple: End the day with more money than you started with.

## Slide 4: Previous Iterations

First, we tried a simple approach called a **Q-table**. It’s just a big table that stores the expected reward for each possible state in each possible action. Although this works sometimes, the market is too complex for this approach to work. The table gets too big, too fast, resulting in it not working.

Next, we tried a standard Neural Network called a **DQN**. It was smarter, but it developed a bad habit. It became a hoarder. It would buy stock and just sit on it forever, praying the price would go up. It wasn't trading; it was gambling. We needed something better.

## Slide 5: Agent Architecture

So, we built our final agent. We designed it to be flexible, using a YAML file for configuration, so we can swap out its “brain” structure easily to test which method is simplest and would work best.

The AI agent sees 6 inputs:
- How wild the price swings are
- Is the price trending up or down
- Is the price trend going to reverse
- How many agents are trading
- How much stock do I own
- How much time remains

The output is special. It doesn't just shout buy or sell. Instead, it chooses a **price range**. We chose this method of simulation because there is more variability in a price range over just buy or sell.

It's acting like a store owner, setting its prices on the shelf.

We used the **ReLU** activation function, along with the **Adam optimizer** and **gradient descent** for backpropagation, to ensure efficient and stable learning.

## Slide 6: Learning Mechanism

We use something called an **Epsilon-Greedy Policy** for its learning policy.

At first, it's just flailing around randomly. It makes noise, buys high, sells low. This is the **Exploration** phase. The agent is exploring the environment and learning its surroundings.

But over time, it starts to figure out that when it there are patterns, and that certain quotes and spreads do different things. So it stops guessing and starts using what it knows works. That's **Exploitation**.

Every time it makes a trade—win or lose—it looks at the result and adjusts its weights. It's basically trial and error, repeated millions of times, until the errors get smaller and the profits get bigger.

In short, it starts off knowing nothing, but it quickly learns its surroundings and gives educated quotes.

## Slide 7: Adversary

The adversary is a population. It's a swarm of simple algorithms. We have four species:

1. **Aggressive bots**: want to trade, so they play riskily and accept even when the price is unreasonable.
2. **Passive bots**: Unlike aggressive bots, passive bots play slowly and wait for a good trade to make money.
3. **Momentum bots**: use a moving average algorithm to calculate and play smart.
4. **Random bots**: are just noise, they dont have meaning to their trades.

We use **Evolution**. Every 50 steps, we look at which agents are the most fit, and change population proportions to match this. There is also a 0.1% mutation rate to ensure no stagnation and random noise.

So if our AI figures out how to trick the Aggressive bots, those Aggressive bots will go bankrupt and disappear. They might be replaced by smarter Passive bots. The AI never gets comfortable because the population is always mutating.

## Slide 8: Interaction Loop

Here is what a single moment in the simulation looks like:

1. **Quote**: Our AI looks at the market and says, “I'll buy at 99 and sell at 101.”
2. **Action**: The thousands of other bots put in their orders.
3. **Match**: The engine checks the books. If someone clicks with our AI's price, boom, a trade happens.
4. **Scoreboard**: Money changes hands. Risk goes up or down.
5. **Evolution**: If it's time, the weak bots are culled, and new ones are born.
6. **Learn**: Our AI thinks, “Did that go well?” and updates its brain for the next round.

## Slide 9: Let’s talk profitability…

As we can see, the agent achieved approximately 0.7% in total return. This might seem like a small number but most of our data sets are only a month in length, which would translate to around 8.4% in annualized returns which outperforms the S&P 500.

Now, if we look at the bottom chart, we can clearly see the exploration and exploitation stages of the agent. The plateaus are where it is experimenting with different strategies, while the surges are where it exploits the learned strategy.

## Slide 10: Causality…

This is actually the most important slide in our presentation because it shows exactly **HOW** our agent is affecting the EGT population.

The bottom 2 charts show the composition of the EGT population as they are evolving. The left side is a control group without an ARL agent while the right side is **WITH** an ARL agent.

On the left, we can see that aggressive and momentum traders, represented by the blue and red, dominated the market as they evolved. On the right however, we see the inverse of the left chart. Our agent actively changed the market, meaning it found the strong bots (which were the aggressive and momentum bots), and exploited them until they went broke, which forced the entire population to shift towards arguably worse strategies in order to benefit the ARL agent.

The agent’s influence was sometimes so strong that it pushed the market dynamics past their normal operating range just to keep the game going. As these resets happen, it may look like the random population is dominating, but that’s not true. It only appears that way because the agent can’t directly control the random population because they don’t react to quotes the same way as the others.

The top right graph represents the dependency of the ARL agent on the EGT population. There are gaps in it because the ARL agent isn’t actively trying to change anything.

## Slide 11: Now, what was the agent’s strategy?

The top chart shows the frequency of the spreads the agent chose. The tall peaks are the spreads it preferred to execute its learned strategy properly. These chosen spreads weren’t just chosen by random, every plateau we looked at in our earlier profitability slide contributed to the agent choosing ideal spreads, which is why we see multiple areas with high frequency.

The bottom chart shows its behavior over time. The gray line is the exploration metric, or epsilon. At first, it starts high when the agent is initially in its exploration phase, and as time goes down, epsilon goes down as well which means it is exploring less and narrowing down some good strategies. When epsilon nearly hits 0, that’s when the agent exploits its learned strategy to make profit.

Now take a look at the red spikes, these are moments when the market crashed, and the agent widened its spreads in an attempt to protect itself.

## Slide 12: Let’s talk about the agent’s methodology…

Now, let’s look at the chart on the top. This chart represents returns per step, and we can clearly see how this is more skewed towards the positive. We essentially make big profits by consistently making smaller profits, and more often than not, the agent made winning trades.

The bottom graph shows the held inventory of the agent. The translucent white line is the true inventory at each time step while the solid white line is the rolling average of the inventory. The agent learned to minimize its net position at near 0 at any given time to avoid direction exposure. What this means is that when a stock crashes, its capital is preserved because it isn’t holding much of that stock to incur a great loss.

## Slide 13: Novelty

Now, how are we different from other research out there? We did some research ourselves and found a few main points of interest across all the research papers.

- Some people put AI in a static market,
- Some used evolution, but no deep learning,
- Some used multiple agents, but with no meaningful interactions between them,

We are the first to combine all three (deep learning, evolution, and a dynamic market). Unlike backtesting, where you just replay the history and test on a static market, our simulation mimics the real world, where if you start winning, the market will adapt and try to work against you. Of course, in real life, this effect is almost non-existent, but we multiplied it so we can observe the behavior of the agents.

## Slide 14: Possible biases and future work

First, we need to make a distinction between overfitting and optimization. We tried our best to diversify our data sets, but ultimately, it was only a few data sets. They were mostly chosen to show different market conditions, like a bullish market, a bearish market, and a somewhat neutral market. So, we can’t know for sure if our hyperparameters are optimizing the bot or letting it overfit to our limited range of data sets.

Second, our neural network is pretty simple, it is essentially a small brain. For our next iteration, we went to give it memory using an RNN, which is a recurrent neural network. This would allow it to remember patterns from longer in the past.

We also want to upgrade the enemy. Instead of the agent going against an EGT population, imagine if it competed with another agent like itself. I think that would be pretty interesting to see.

Lastly, the most important upgrade would be an implementation of a limit order book. This would allow the program to set prices through actual buy and sell orders stacked at different levels. Without a LOB, the model can’t simulate true liquidity, slippage, or how trades actually affect the price.

## Slide 15: In conclusion…

To wrap up, we proved that a neural network can survive in an adversarial environment. It didn’t just memorize prices, but rather, it learned complex human-like behaviors like risk control, inventory management, and adaptive strategies completely from scratch.
