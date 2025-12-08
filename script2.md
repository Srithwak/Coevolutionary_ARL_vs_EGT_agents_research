## Slide 1

Hi everyone, we're Rithwak and Praneeth, and this is our research project. We're excited to walk you through our work on creating adaptive artificial intelligence for financial markets.

## Slide 2

**Our detailed research question is:**
> How does an Adversarial Reinforcement Learning agent perform and adapt against an evolving population of heuristic traders governed by Evolutionary Game Theory?

Let's break down why this is important. Most AI tests today are like putting a robot in a maze. The maze might be tricky, but the walls don't move. They don't try to trap you. It's a **static environment**.
But the financial market isn't a static maze. It's a game played against other people.
If you find a winning strategy, other traders will notice. They will copy you, or they will find a way to counter you. Your edge disappears.
We wanted to build a simulation that captures this reality. We created a world where the background traders aren't just random noise—they are active players who push back. If our AI starts winning, they start changing.
We used the stock market to test three big things:
One, **Strategy**: Can the AI outsmart a crowd?
Two, **Risk**: Can it play the game without going broke?
Three, **Interaction**: How does it handle a market that is constantly shifting under its feet?
So, our goal isn't just to make a bot that gets rich. There are thousands of those. Our goal is to understand *how* an AI learns to survive when the rules of the game are constantly changing.

## Slide 3

**Here is how we set up the experiment:**
We built a real stock market engine called a **Limit Order Book**.
If you haven't seen one before, think of it like an auction. On one side, you have the **Bid**—that's the buyers saying "I'll pay this much." On the other side, you have the **Ask**—that's the sellers saying "I want at least this much."
The gap in the middle? That's the **Spread**.
Now, for a trade to actually happen, someone has to be aggressive. They have to cross that gap and accept the other person's price.
In our simulation, our AI has to decide: Does it want to sit back and wait for a bargain? Or does it want to cross the spread and make a deal happen right now?
Because there are thousands of little agents running around, there's a lot of randomness. So we made sure to use a "seed" feature. This basically lets us rewind time and replay the exact same simulation twice, to prove our results aren't just luck.
The goal for every agent is simple: End the day with more money than you started with.

## Slide 4: Previous Iterations

Science is mostly about trying things that don't work.
First, we tried a simple approach called a **Q-table**. Imagine a giant spreadsheet where every possible market situation has a row. The problem is, the market has infinite possibilities. The spreadsheet gets too big, too fast. It crashed and burned.
Next, we tried a standard Neural Network called a **DQN**. It was smarter, but it developed a bad habit. It became a **hoarder**. It would buy stock and just sit on it forever, praying the price would go up. It wasn't trading; it was gambling. We needed something better.

## Slide 5: Agent Architecture

So, we built our final agent. We designed it to be flexible, so we can swap out its "brain" structure easily.
What does the agent actually see? It looks at **Volatility**—basically, how wild the price swings are. It looks at **Momentum**—is the price trending up or down? And it looks at **Volume**—how busy is the market right now?
But most importantly, it looks at itself. It checks its own **Inventory**: "Do I own too much stock right now?" And it checks the **Time**: "Is the day almost over?"
The output is special. It doesn't just shout "BUY!" or "SELL!". Instead, it chooses a **price range**. It says, "I'm willing to buy low, down here, and sell high, up there." usage.
It's acting like a store owner, setting its prices on the shelf.
We used a math function called **ReLU** for its brain neurons. It's just a simple, efficient way for the AI to handle the messy, chaotic data of the stock market.

## Slide 6: Learning Mechanism

How does it learn? We use something called an **Epsilon-Greedy Policy**.
Think of it like a toddler trying to walk. At first, it's just flailing around randomly. It pushes buttons, makes noise, buys high, sells low. This is the **Exploration** phase.
But over time, it starts to figure it out. "Hey, when I do this, I make money." So it stops guessing and starts using what it knows works. That's **Exploitation**.
We use an **Optimizer** to update its brain. Every time it makes a trade—win or lose—it looks at the result and adjusts its internal wiring. It's basically trial and error, repeated millions of times, until the errors get smaller and the profits get bigger.
In short: It starts as a chaotic mess, and evolves into a disciplined sniper.

## Slide 7: Adversary

Now for the 'Enemy'. This is the coolest part. The enemy isn't one big boss; it's a swarm of simple bots. We have four species:
1. **Aggressive**: The impatient ones. They want to trade NOW.
2. **Passive**: The patient ones. They wait for a deal.
3. **Momentum**: The trend-chasers.
4. **Random**: The wildcards.
Here is the key: We use **Evolution**. Every 50 steps, we look at who made money. The bots that lost money? They die. Gone. The bots that made money? They reproduce.
So if our AI figures out how to trick the Aggressive bots, those Aggressive bots will go bankrupt and disappear. They might be replaced by smarter Passive bots. The AI never gets comfortable because the population is always mutating.

## Slide 8: Interaction Loop

So, here is what a single moment in the simulation looks like:
1. **Quote**: Our AI looks at the market and says, "I'll buy at 99 and sell at 101."
2. **Action**: The thousands of other bots put in their orders.
3. **Match**: The engine checks the books. If someone clicks with our AI's price, **Boom**, a trade happens.
4. **Scoreboard**: Money changes hands. Risk goes up or down.
5. **Evolution**: If it's time, the weak bots are culled and new ones are born.
6. **Learn**: Our AI thinks, "Did that go well?" and updates its brain for the next round.

<br>
<br>
<br>
<br>
<br>

# Slide 9: Results - Profitability

Let's look at the numbers.
In this run, the agent made over **0.7% profit**. That might sound small, but in high-frequency trading, that is a solid number.

If you look at the chart, you can actually see the learning happen. There are clear plateaus and steep rises.

The flat sections are where the agent is experimenting with strategies, and the sharp rise is where it exploits the learned strategy to make a profit. We ran this simulation many times, and it consistently found a way to win.

# Slide 10: Results - Causality

This is actually the most important slide in our presentation.

The 2 bottom charts show the composition of the population as they are evolving. The left side is just a control group without an ARL agent.

On the left, we can see that agressive and momentum traders dominated. But on the right? we can see the inverse of the left chart. Our AI actively **changed the market**, meaning It found the weak bots (the Aggressive ones), and exploited them until they went broke, forcing the entire population to shift towards arguably worse strategies like passive and random trades.

The agent's influence was sometimes so strong that it pushed the market dynamics past their normal operating range just to keep the game going. As these resets happen, it looks like the random population is dominating, but its not. This effect appears because the agen't cant directly control that population.

The top right graph is dependency of the ARL agent on the egt population. There are gaps in it because the ARL agent isnt actively trying to change anything.

# Slide 11: Results - Strategy

The top chart shows the frequency of the spreads the agent chose. The tall peaks are the ideal spreads for its learned strategy. It didn't just pick random numbers. It found specific spreads that benefitted it the most

The bottom chart shows its behavior over time. The gray line is the "Learning Rate." It starts high—trying different strategies and drops to zero as it becomes an better at trading.

Now take a look at the red spikes. Those are moments when the market crashed. In response, the AI kind of panicked in a smart way. It widened its spreads massively to protect itself.

## Slide 12: Results - Methodology

now lets look at these charts
The chart on top shows us returns per step, and we can clearly see how this is more skewed towards the positive. We are making a big profit by consistently making smaller profits.

The bottom graphs shows the inventory of the agent. The white line is the rolling average while the translucent white line is the true inventory at each time step. The agent learned to minimize its net position at near 0 to avoid directional exposure. this basically means that when the stock crashes, its capital is preserved because it isn't holding much of that stock.

## Slide 13: Novelty

Now, how are we different from other research out there.
We did some searching and found a few main points of interest in all the research out there.
Some people put AI in a static market.
Some people used evolution, but no deep learning.
Some people used multiple agents, but they didn't really interact with each other in any meaningful way.

We are the first to combine all three: **Deep Learning**, **Evolution**, and a dynamic market.
Unlike backtesting, where you just replay history, our simulation mimics the real world: If you start winning, the market will adapt to try and work against you.

## Slide 14: Bias and Future Work

First, we need to make a distinction between overfitting and optimization. We tried our best to diversify our data sets but ultimately it was only a few data sets. We can't know for sure if our hyperparameters are optimizing the bot or letting it overfit to our limited range of data sets.

Second, our neural network is pretty simple. It's a small brain.
For our next iteration, we want to give it a "Memory" using something called an **RNN**, which would let it remember patterns from longer in the past.

We also want to upgrade the enemy. Instead of fighting simple evolutionary bots, imagine if we pitted it against another ARL agent. I think that would be interesting to see.

## Slide 15: Conclusion

To wrap up:
We proved that a Neural Network can survive in an adversarial environment.

It didn't just memorize prices. It learned complex human behaviors—like risk control, inventory management, and adaptive strategies completely from scratch.

Thank you for listening.
