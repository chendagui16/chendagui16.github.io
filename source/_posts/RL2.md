---
title: RL2 Markov decision Processes
mathjax: true
date: 2017-03-30 23:18:44
categories: reinforcement learning
tags: [reinforcement learning, machine learning]
---
reference:
    [UCL Course on RL](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html)
# lecture 2 Markov decision Processes
MDP formally describe an environment for RL, where the environment is fully observable
> **Markov Property**
> The future is independent of the past given the present
> $$ P(S\_{t+1} | S\_{t} ) = P(S\_{t+1} | S\_1,S\_2,\dots,S\_t) $$
> The state is sufficient statistic of the future

## Markov Process
A Markov Process is a memoryless random process
> **Definition**
> A Markov Process (Markov Chain) is a tuple (S,P)
> * S is a (finite) set of states
> * P is a state transition probability matrix.

## Markov Reward Process
A Markov reward process is a Markov chain with variable
> **Definition**
> A Markov Reward Process is a tuple (S,P,R,y)
> * S is a finite set of states
> * P is a state transition probability matrix
> * R is a reward function: $ R\_s = E[R\_{t+1}|S\_t=s] $
> * y is a discount factor

Return
> **Return**
> The return $G\_t$ is the total discounted reward from time-step t.
> $$ G\_t = \sum\_{k=0}^{\infty} y^k R\_{t+k+1}$$

Why discount?
1. Mathematically convenient to discount reward
1. Avoids infinite returns in cyclic Markov processed
1. Uncertainty about the future may not be fully represented
1. If the reward is financial, immediate rewards may earn more interest than delayed rewards
1. Animal/human behaviour shows preference for immediate reward
1. It is sometimes possible to use undiscounted Markov reward processes

## Value Function
The value function gives the long-term value of state s
> **Definition**
> The state value function $v(s)$ of an MRP is the expected return starting from state s
> $$ v(s) = E[G\_t | S\_t = s]$$

## Bellman Equation for MRPs
The value function can be decomposed into two patrs
* immediate reward $R\_{t+1}$
* discounted value of successor state $y v(S\_{t+1})$

The Bellman equation can be expressed concisely using matrices,
$$ v = R + y P v $$
where $v$ is a column vector with one entry per states

**Solving Bellman Equation**
* linear equation and can be solved directly
* Computational complexity is $O(n^3)$ for $n$ states
* Direct solution only possible for small MRPs
* Many iterative methods for large MRPs
> * Dynamic Programming
> * Monte-Carlo evaluation
> * Temporal-Difference learning

## Markov Decision Process

### Policy 
> **Definition**
> A policy $\pi$ is a distribution over actions given states,
> $$ \pi(a|s) = P[A\_t=a|S\_t=s]$$
MDP policies depend on the current state (not the history)

> Given an MDP M =(S,A,P,R,y) and a Policy $\pi$
> * The state sequence $S\_1,S\_2,\dots$ is a Markov process $ (S,P^{\pi}) $
> * The state and reward sequence is a Markov reward process $ (S,P^{\pi},R^{\pi},y) $

### Value function
** State Value function **

> The state value function $v\_{\pi}(s)$ of an MDP is the expected return starting from state $s$, and then following policy $\pi$
> $$ v\_{\pi} (s)  = E\_{\pi} [G\_t |S\_t=s] $$

** Action value function **

> The action value function $q\_{\pi}(s,a)$ is expected return starting from state $s$, taking action $a$, and then following policy $\pi$
> $$ q\_{s,a} = E\_{\pi} [G\_t |S\_t=s,A\_t=a]$$

### Bellman Expectation Equation
The state-value function/action-value function can be decomposed into immediate reward plus discounted value of successor state.
** For $V^{\pi}$ **
$$ V\_{\pi} (s) = \sum\_{a \in A} \pi (a|s) q\_{\pi} (s,a) $$

** For $Q^{\pi}$ **
$$ q\_{\pi} (s,a) = R\_{s}^{a} + y \sum\_{s' \in S} P\_{ss'}^{a} v\_{\pi} (s') $$

** For $V^{\pi}$ (2) **
$$ V\_{\pi} (s) = \sum\_{a \in A} \pi (a|s) R\_{s}^{a} + y \sum\_{s' \in S} P\_{ss'}^{a} v\_{\pi} (s') $$

** For $Q^{\pi}$ (2)**
$$ q\_{\pi} (s,a) = R\_{s}^{a} + y \sum\_{a \in A} \pi (a'|s') q\_{\pi} (s',a') $$

> Matrix form
> $$ v\_{\pi} = R^{\pi} + yP^{\pi} v\_{\pi}$$

### Optimal value function
> **Definition**
> The optimal state-value function is the maximum value function over all policies
> The optimal action-value function is the maximum action-value function over all policies

* The optimal value function specifies the best possible performance in the MDP.
* An MDP is "solved" when we know the optimal value fn

### Optimal Policy
Define a partial ordering over policies
$$ \pi \ge \pi', v\_{\pi} (s) \ge v\_{\pi '} (s) $$

> ** Theorem **
> For any Markov Decision Process
> * There exists an optimal policy that is better than or equal to all other policies
> * All optimal policies achieve the optimal value function
> * All optimal policies achieve the optimal action-value function

An optimal policy can be found by maximising over optimal q function
* There is always a deterministic optimal policy for any MDP
* If we know optimal q function, we immediately have the optimal policy

### Bellman Optimality Equation
** For $v\_*$ **

$$v\_\* (s) = \max\_{a} q\_\* (s,a)  $$

** For $Q^*$ **

$$ q\_\* (s,a) = R\_s^a + y \sum\_{s' \in S} P\_{ss'}^{a} v\_\* (s') $$

** For $V^*$ **

$$ v\_\*(s) = \max\_{a} R\_s^a + y \sum\_{s' \in S} P\_{ss'}^a v\_\* (s') $$

** For $Q^*$ **

$$ q\_\* (s,a) = R\_s^a + y \sum\_{s' \in S} P\_{ss'}^a \max\_{a'} q\_\* (s',a') $$

### Solving the Bellman Optimality Equation
* Bellman Optimality Equation is non-linear
* No closed form solution (in general)
* Many iterative solution methods
    * value iteration
    * policy iteration
    * q-learning
    * Sarsa

## Extensions to MDPs

### Infinite and continuous MDPs
* Countably infinite state and/or action spaces
* Continuous state and/or action spaces
* Continuous time
    * Requires partial differential equations
    * Hamilton-Jacob-Bellman (HJB) equation
    * Limiting case of Bellman equation as time-step --> 0

### Partially observable MDPs
POMDPs (Partially Observable MDPs)

A Partially Observable Markov Decision Process is an MDP with hidden states. It is a hidden Markov model with actions.
> **Definition**
> A POMDP is a tuple (S,A,O,P,R,Z,y)
> * S is a finite set of states
> * A is a finite set of actions
> * O is a finite set of observations
> * P is a state transition probability matrix
> * R is a reward function
> * Z is an observable function
> * y is a discount factor

** Belief States **
* A history $H\_t$ is a sequence of actions, observations and rewards
* $$ H\_t = A\_0, O\_1, R\_1, \dots, A\_{t-1}, O\_t, R\_t $$
* A belief state $b(h)$ is a probability distribution over states conditioned on the history h
* $$ b(h) = (P[S\_t = s^1 |H\_t = h], \dots, P[S\_t=s^n |H\_t =h]) $$

** Reductions of POMDPs **
* The history $H\_t$ satisfies the Markov property
* The belief state $b(H\_t)$ satisfies the Markov property
### Undiscounted, average reward MDPs
** Ergodic Markov Process **
An ergodic Markov process is 
* Recurrent: each state is visited an infinite number of times
* Aperiodic: each state is visited without any systematic period

> ** Theorem **
> An ergodic Markov process has a limiting stationary distribution $d^{\pi} (s)$ with the property
> $$ d^{\pi} (s) = \sum\_{s' \in S} d^{\pi} (s') P\_{s's} $$

** Ergodic MDP **
An MDP is ergodic if the Markov chain induced by any policy is ergodic.

For any policy $\pi$, an ergodic MDP has an average reward per time-step that is independent of start state

** Average Reward Value Function **
The value function of an undiscounted, ergodic MDP can be expressed in terms of average reward.
