---
title: "What actually is reinforcement learning?"
date: 2025-01-21
layout: article
series: "rl-tutorials"
series-part: 1
series-total: 2
---

# Tutorial 1: What actually is reinforcement learning?

Before attempting to do reinforcement learning, we of course must define the problem reinforcement learning aims to solve. This is the sole goal of this first tutorial. The setting we define here will be common (modulo the occasional generalisation/assumption, which we will of course state when used) among all future tutorials. 

This tutorial, while providing rigorous definitions, will be of a far less formal flavour than later tutorials; the aim being to sufficiently engage less mathematically mature readers enough to persevere with the later tutorials. We will spend quite a bit of time intuiting as to why these definitions are the norm; why we choose to work within this framework in the first place. After all, a well chosen basis is vital in producing theoretical work of any value. 

We will cover 

(I) - The discrete Markov decision process formulation of reinforcement learning

(II) - The partially observable Markov decision process generalisation

## Discrete Markov Decision Processes

We begin with the setting for reinforcement learning. The natural way to think of our setting is a game. Games have actions players can take, and states the game can be in. The game starts in some (not necessarily deterministic) initial state, and given a state-action pair, there will be some mechanism for computing (not necessarily deterministically) the resulting state after taking this action in the current state. You may say "but what about the states before? I thought actions had consequences?" Well, you're right, they certainly do. However...

It is reasonably logical to assume games always have the Markov property[^The future given present is independent of the past, or [more mathematically](https://en.wikipedia.org/wiki/Markov_property#Definition)]. At first, this seems restrictive. You may say "but 20 minutes ago I chose to put a point in dexterity, doesn't that matter?" Well, yes, it certainly does matter, but we choose to incorporate that information into the world state. As such, the potentially long history of consequences of our actions will always be encoded into our current world state; the world state determines the full picture of our game at a given time. Given this, the Markov property becomes obvious. Why would the outcome of an action depend on anything but the current state of the world? 

The not so easily convinced reader may be crying out "but this space of possible world states will be huge! How could we ever expect to work with such an object?". This is a reasonable question; in fact it was my initial qualm when encountering the Markov decision process formalisation. The reason, primarily, is as follows: our $(\text{state}, \text{action}) \mapsto \text{state}$ mapping would instead need to be of the form $$(\text{state}_1, \text{state}_2, ..., \text{state}_n, \text{action}) \mapsto \text{state}$$ Notably, the number of arguments grows linearly in $n$, and hence the domain of such a function grows exponentially in the size of the state space. This is a disaster! Further having different domains for transitions at time $n$ is a nightmare for the modeller. We sacrifice clarity in exchange for, at best, equal performance[^If, at each time step, there is no common information across states then we will have our exponentially growing product $\prod_{t=1}^n \mathcal{S}_i$ of equal cardinality to $\mathcal{S}$. Otherwise, we can store this more efficiently]! Additionally, Markov processes are *nice*. They have a rich theory we can draw from and additional structure we can utilise. 

Now you, the reader, are (hopefully) convinced of the Markovian approach to modelling our game, we can begin with the formalisation. Let $\mathcal{S}$ be the set of possible world-states and $\mathcal{A}$ be the set of possible actions. Note that $\mathcal{A}$ is also time-invariant. We take this to be the set of all possible actions **at any time**. While actions may be impossible at certain points (for example, one cannot jump while laying in prone position[^Unless doing the worm counts]), this need not be directly told to our learning agent. Naively, one may think of RL as: given rules, find the optimal way to play. While this can certainly be an application, it is not the only application. Sometimes we may not know all the rules a priori (consider, for example, an RL agent learning to socialise with human counterparts), so it is overly restrictive to, given a state, restrict the agent's action space to any subset of $\mathcal{A}$. Further, as we just argued, our $(\text{state}, \text{action}) \mapsto \text{state}$ map would have domain dependent on state, which vastly increases modelling difficulty. 

Speaking of this $(\text{state}, \text{action}) \mapsto \text{state}$ map, let's define it. First of all, need this map be deterministic? No, certainly, not. Our RL agent could never, for example, learn poker in such a setting[^Our next card is, of course, random]. We instead reason with a *transition distribution*, a conditional distribution of the form $\text{state} \mid \text{state}, \, \text{action}$. We also need to reason about the initial state of the world. Again, if we want our agent to able to learn poker, this will need to be stochastic[^Dealer gives us two cards face down at random], so we sample the initial world state from a distribution on $\mathcal{S}$. Formally, we define the *initial state distribution* and *transition distribution* with probability measures

$$
\mathbb{T}_0(s_0): \mathcal{S} \to [0, 1] \quad \text{and} \quad \mathbb{T}(s_{t+1} \mid s_t, a_t): \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0, 1]
$$

Now we've described, in abstract, the dynamics of the world our agent seeks to reason within, we must ask: "how will our agent learn to reason?" The whole idea of the reinforcement learning is as follows: we **reinforce** "good" behaviour by "rewarding" our agent. Given a state, $s \in \mathcal{S}$ and a chosen action $a \in \mathcal{A}$, our agent will receive some pre-determined reward $r$. This gives rise to a *reward function* $R(a \mid s): \mathcal{A} \times \mathcal{S} \to \mathbb{R}$ . Note two things: 

1 - Our reward must be a real number. This gives us access to the rich field structure of $\mathbb{R}$.

2 - This real number can be negative: we can **penalise** bad behaviour 

This second note is particularly important. We have provided a viable solution to the "invalid actions" qualm. We can simply penalise invalid actions, so that our agent will learn (hopefully!) to take only valid actions. 

Once again, the (ideally singular) voice in your head should be shouting something along the lines of "but actions have consequences!" And again, they certainly do. What we have defined is the *immediate reward*. That said, future consequences are often not as pressing as immediate consequences. When taking an action, we often apply some **discount** to future rewards for the action relative to the immediate reward. Over-discounting future rewards is why we can find ourselves binging mindless media until four in the morning, and under-discounting future rewards is why you never asked that person for their number. 

The simplest way to discount future rewards is with a **discount rate** $\gamma \in [0, 1)$. In this setting, the *cumulative reward* for a sequence of actions, which was previously $\sum_{t=n}^\infty R(a_t \mid s_t)$ for state-action pairs $(s_t, a_t)\_{t=n}^\infty$ from time $t=n$ onwards, becomes $\sum_{t=n}^\infty \gamma^t R(a_t \mid s_t)$. This, first of all, captures the case where future rewards are indeed not as important as present rewards (e.g. trading in the stock-market via the time-value of money), but it does something even more important: **it makes the cumulative reward convergent for any bounded reward function $R$**[^The proof is an exercise in analysis for toddlers]

This latter fact is of great importance. Consider the following game: at each time $t$ I give you $1$ pound, and allow you the choice to pay 1 pound to bet on a (fair) coinflip. If the coin lands heads, I, the banker, will give you 10 pounds. If it lands tails, you receive nothing (and I, the banker, keep your 1 pound). It is obvious that the optimal strategy at each move is to always bet, for which the (non-discounted) cumulative return over $t > 0$ is infinite. However, if we chose to not bet, our cumulative return will also be infinite. How can we posit as to which strategy is better if our cumulative returns are identical. If we fix $\gamma \in [0, 1)$ and consider the discounted returns, we see that the optimal-strategy has (expected) cumulative return $\sum_{t = 0}^\infty \gamma^t \cdot (1 + 9/2) = \frac{11/2}{1 - \gamma}$, whereas our "do-nothing" strategy has (expected) cumulative return $\sum_{t=0}^\infty \gamma^t = \frac{1}{1 - \gamma}$, so the optimal-strategy indeed yields greater (expected discounted) cumulative returns. Problem solved! 

Finally, we need to decide on a horizon of moves that are relevant. Does our game go on indefinitely, or does it stop after a predetermined number of moves? To do so, we choose $N \in \mathbb{N} \cup \{+\infty\}$ (i.e. $N$ is a possibly infinite positive integer) and call this our *time-horizon*, so that our agent need only consider the tuple $(s_0, a_0, r_0, \dots, s_{N-1}, a_{N-1}, r_{N-1}, s_N)$ where $s_0 \sim \mathbb{T}_0$ , $s_{t+1} \mid s_t, a_t \sim \mathbb{T}$ for $t = 0, 1, \dots, N-1$ and $r_t := R(a_t \mid s_t)$ . We will call this tuple, $\tau$, an *episode*. 

All together, we've described both the world an agent learns in, and how it learns within this world. Everything is encapsulated in the tuple $(\mathcal{S}, \mathcal{A}, \mathbb{T}, \mathbb{T}_0, R, \gamma, N)$, which we call a Markov decision process. Lets conclude with a concise formal definition.

**Definition 1 - Discrete Markov Decision Process**

A *discrete Markov decision process* is a tuple $(\mathcal{S}, \mathcal{A}, \mathbb{T}, \mathbb{T}_0, R, \gamma, N)$ where $\mathcal{S}, \mathcal{A}$ are sets representing the state/action spaces respectively, $\mathbb{T}_0(s_0): \mathcal{S} \to [0,1], \; \mathbb{T}(s_{t+1} \mid s_t, a_t): \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0, 1]$ are probability measures for the initial state and transition distributions respectively, $R(a_t \mid s_t): \mathcal{S} \times \mathcal{A} \to \mathbb{R}$  is a map giving the reward for taking action $a$ in state $s$ , $\gamma \in [0,1)$ denotes the discount rate of future rewards and $N \in \mathbb{N} \cup \{+\infty\}$ denotes the time-horizon of the process.

**End of Definition** 

If $N < +\infty$ we will call our Markov decision process *finite*. Intuitively, finite Markov decision processes model games with a deterministic window of moves to consider, or at least the fact we've modelled our agent to only consider a deterministic number. note that in this case we may additionally permit $\gamma = 1$, as cumulative rewards will surely be finite.

## The Objective of Reinforcement Learning

Let $(\mathcal{S}, \mathcal{A}, \mathbb{T}, \mathbb{T}_0, R, \gamma, N)$ be a discrete Markov decision process. Recall that we called a tuple of the form $\tau = (s_0, a_0, r_0, \dots, s_{N-1}, a_{N-1}, r_{N-1}, s_N)$ where $r_t := R(a_t \mid s_t)$, $s_t \mid s_{t-1}, a_{t-1} \sim \mathbb{T}$ for $t = 1, \dots, N$ and $s_0 \sim \mathbb{T}_0$ an *episode*. These can be thought of as a playthrough of our game. We'd like to know about how these are distributed. To answer this, we first need to decide how actions are chosen.

We call a *policy* a probability measure $\pi(a \mid s): \mathcal{S} \times \mathcal{A} \to [0,1]$. This is simply a, well, policy, for choosing actions for certain states. We are often uncertain in our actions, or want to randomly vary our actions (e.g. game-theoretically optimal poker), hence we (among other, learning related, reasons) utilise stochastic policies. The goal of reinforcement learning is of course to learn a "good" policy. Given a policy $\pi$, we can now specify the *episodic distribution* by
$$
\mathbb{T}^\pi(\tau) = \mathbb{T}_0(s_0)\prod_{t=1}^N \pi(a_t \mid s_t)\mathbb{T}(s_t \mid s_{t-1}, a_{t-1})
$$
for each episode $\tau$. Note of course that the episodic distribution depends on the chosen policy. It is not too hard to show that this indeed is a probability measure. 

Earlier, we reasoned that to choose a good strategy amounts to choosing a strategy with the greatest *expected discounted cumulative reward*. We define this quantity, which is a function of the chosen policy $\pi$, by 
$$
J(\pi) = \mathbb{E}_{\tau \sim \mathbb{T}^\pi}\left[\sum_{t = 0}^{N-1}\gamma^t r_t\right]
$$
The objective of RL, formally, is tnus to find $\pi^\star = \arg\min_{\pi} J(\pi)$.

## Partially Observable Discrete Markov Decision Processes

Often, it may be the case where the underlying game's dynamics are well described by a Markov decision process, but our agent cannot fully observe the underlying state. A natural example of this is a robot learning to navigate a maze. The world is the full maze, but our robot can only see a, say, 90 degree window in-front of it at any given moment. Gradually, as it tries and tries to navigate this maze, it will discover more of the world, but it will only have a partial picture of the world at any given moment in time. 

So, our world is still described by the tuple $(\mathcal{S}, \mathcal{A}, \mathbb{T}, \mathbb{T}_0)$, where we use $\mathbb{T}, \mathbb{T}_0$ to denote our transition/initial state measures, but our agent can't directly observe $s \in \mathcal{S}$ any longer. Instead, it will observe $o \in \mathcal{O}$, an element of a specified observation space. In this case of our robot, this is the set of possible images the robot can perceive (e.g. $H \times W \times 3$ for images with $H \times W$ pixel dimension and RGB colouring). Our episodes will now be of the form $$\tau := (o_0, a_0, \dots, o_{N-1}, a_{N-1}, o_N) \in \mathcal{T} =: (\mathcal{O} \times \mathcal{A})^N \times \mathcal{O}$$ You may wonder "what about rewards?" Great question!

In this case of fully observable Markov decision processes, our agent was able to compute $R(a \mid s)$ to update its beliefs. This is no longer the case. We could naively just compute $R(a \mid o)$ and use this to update our agent's beliefs, but this is incredibly problematic: it is, by it's very definition, myopic. Consider the following simple two-player game: 10 cards are laid out on a table, labelled 1 through 10, face down. Each player picks up a card, and **individually** observes the label. Then, each player is given the choice to bet (with the first bettor alternating each round) 1 pound, or to check (i.e. do nothing and give the opponent this same setting). Suppose we pick up 10, 9, 8, and get called on bets 3 times in a row. The next hand, we pick up a 1, check, and our opponent decides to bet. It is clear that, no matter what our opponent picked up, we are in the lead. However, if we were to define a reward for the observation (receive bet, see 1, call) we'd be foolish to make it anything but negative. Absurd! These myopic agents could never learn this trivial card game. 

Clearly, when we compute a reward, it will depend on all the actions and observations made thus far. This is, fundamentally, non-Markovian behaviour. We're in the horror state where our domain now varies on time; we need to consider both the present and the past. The natural way to deal with this modelling complexity is as follows: compute rewards at the end of each episode. This leads us to define our reward by $R(\tau): \mathcal{T} \to \mathbb{R}$. This will, especially in the case of large, $N$, surely slow down learning. Rather than our agent observing $N$ rewards at the end of each game, they only have one to work with. This is one of the reasons that learning agents utilising POMDPs are [notoriously difficult to implement](https://web.mit.edu/jnt/www/Papers/J016-87-mdp-complexity.pdf). 

So now we have our world's dynamics, the notion of an observation, an episode, and rewards defined. What remains? Well, analogously we will define $N \in \mathbb{N} \cup \{+\infty\}$ to be the time-horizon of our game. More pressingly, how are our observations distributed? Well, what we observe depends on our current state, so it is natural to consider a distribution of the form $\text{observation} \; \mid \; \text{state}$. Actually, this is fine. We need not consider prior observations, as only the current state determines what we observe. Hence, we have Markovian behaviour. Thank goodness! Formally, we will define the *observation distribution* by the probability measure $$\mathbb{O}(o \mid s): \mathcal{O} \times \mathcal{S} \to [0,1]$$ This is what we observe given a hidden state. In the case of our robots, this is stochastic, but it may not be the case for other games (e.g. poker). 

Note we no longer have a discount rate to consider. This is due to the fact that, by definition, there is nothing we can discount. We consider the episodic-reward, not the action-reward. This can be formulated to use a form of action reward, but this approach is both less intuitive and less general (though, on occasion, slightly more mathematically convenient). 

All together, we have the tuple $(\mathcal{S}, \mathcal{A}, \mathcal{O}, \mathbb{T}, \mathbb{T}_0, \mathbb{O}, R, N)$. This 9-tuple defines the *partially observable Markov decision process*. We summarise the situation with a concise formal definition. 

**Definition: Partially Observable Markov Decision Process**

A partially observable Markov decision process is a tuple $(\mathcal{S}, \mathcal{A}, \mathcal{O}, \mathbb{T}, \mathbb{T}_0, \mathbb{O}, R, N)$ where: $\mathcal{S}, \mathcal{A}, \mathcal{O}$ are sets, defining the state/action/observation spaces respectively; $\mathbb{T}(s^\prime \mid a, s): \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0,1]$ and $\mathbb{T}(s)_0: \mathcal{S} \to [0,1]$ are probability measures, defining the underlying system mechanics; $\mathbb{O}(o \mid s): \mathcal{O} \times \mathcal{S} \to [0,1]$ is a probability measure, defining the observation distribution from a hidden world state; $R(\tau): (\mathcal{O} \times \mathcal{A})^N \times \mathcal{O} \to \mathbb{R}$ is map defining the reward our agent receives after an episode $\tau = (o_0, a_0, \dots, o_{N-1}, a_{N-1}, o_N)$ and $N$ defines the time-horizon our agent considers. 

**End of Definition** 

As in last section, we'd like to know how episodes are distributed. To do this, we first need to define the notion of a policy in this setting. Unlike last time, we can't just specify a probability distribution on $\mathcal{S} \times \mathcal{A}$, we'll need to be a tad more creative. Like rewards, our chosen action will need to depend on both present and past. Let $\Delta(\mathcal{A})$ be the $\mathcal{A}$-simplex, namely the space of probability measures on $\mathcal{A}$. After each new observation, we will take a policy in $\Delta(\mathcal{A})$, so it makes sense to take an episodic approach, as we did in rewards, with policies. Formally, we define our policy $\pi$ by a family of $N$ sub-policies $\{\pi_t: \mathcal{T}_t \to \Delta(\mathcal{A})\}_{t=1}^N$ where $\mathcal{T}_t := (\mathcal{O} \times \mathcal{A})^{t-1} \times \mathcal{O}$ denotes the space of $t$*-trajectories* $(o_0, a_0, \dots, o_\{t-2}, a_{t-2}, o_{t-1})$.  

Let $\tau = (o_0, a_0, \dots, o_{N-1}, a_{N-1}, o_N)$ be an episode, and for $1 \leq t \leq N$ let $\tau_t = (o_0, a_0, \dots, o_{t-2}, a_{t-2}, o_{t-1})$ be the corresponding $t$-trajectories. Note in the special case of the $1$-trajectory, we have the 1-tuple $(o_0)$. Let $\pi = \{\pi_t: \mathcal{T}_t \to \Delta(\mathcal{A})\}_{t=1}^N$ be a policy. We denote the *episodic distribution* for the policy $\pi$ by $\mathbb{T}^\pi$. This can be explicitly, computed, but we leave that as an exercise for the motivated reader. The goal of an RL agent in the POMDP setting it thus to maximise the *expected episodic reward* $\mathbb{E}_{\tau \sim \mathbb{T}^\pi} \left[R(\tau)\right]$. 

## Concluding Remarks

And with that, we should hopefully have a good grasp on what reinforcement learning is. Next tutorial, we'll begin doing some reinforcement learning. 

I'll remark that the definition of a POMDP is somewhat contentious. I've seen several different formulations used in the literature (e.g. [here](https://arxiv.org/pdf/2204.08967), for which our definition is a trivial generalisation, and [here](https://www.annualreviews.org/docserver/fulltext/control/5/1/annurev-control-042920-092451.pdf?expires=1761400032&id=id&accname=guest&checksum=9770B090655CB1F76D7FF83C598E3C94), which is nothing like it), but this version is certainly the most general I've come across, while also being reasonably practical. In fact, I actually ended up updating the Wikipedia article for POMDPs, as the definition previously held there was nonsensical. I only plan to use these in one tutorial, so I was hesitant on even including it, but I decided it worthwhile for two reasons: one, showcasing the generalisation process, and two, I may write future tutorials that use it.   






