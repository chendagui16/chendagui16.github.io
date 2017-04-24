---
title: RL5 Model-Free Control
mathjax: true
date: 2017-04-10 08:25:59
categories: reinforcement learning
tags: [reinforcement learning, machine learning]
---
reference:
    [UCL Course on RL](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html)
# lecture 5: Model-Free Control
## Introduction
Optimise the value function of an unknown MDP

_Model-free control_ can solve these problems, either:
1. MDP model is unknown, but experience can be sampled
1. MDP model is known, but is too big to use, except by samples

**On and Off-Policy Learning**
* On-policy learning
    * Learn on the job
    * Learn about policy $\pi$ from experience sampled from $\pi$
* Off-policy learning
    * Look over someone's shoulder
    * Learn about policy $\pi$ from experience sampled from $\mu$

## On-Policy Monte-Carlo Control
### Generalised Policy Iteration
**Generalised Policy Iteration(Refresher)**
_Policy evaluation_: Estimate $v\_\pi$, Iterative policy evaluation
_Policy improvement_: Generate $\pi' \ge \pi$, Greedy policy improvement

**Generalised Policy Iteration With Monte-Carlo Evaluation**
_Policy evaluation_: Monte-Carlo policy evaluation, $V=v\_\pi$
_Policy improvement_: Greedy policy improvement

**Model-Free Policy Iteration Using Action-Value Function**
* Greedy policy improvement over $V(s)$ requires model of MDP
$$ \pi'(s) = \arg\max\_{a \in A} R\_s^a +P\_{ss'}^a V(s')$$
* Greedy policy improvement over $Q(s,a)$ is model-free
$$ \pi'(s) = \arg\max\_{a \in A} Q(s,a) $$

**Generalised Policy Iteration with Action-Value Function**
_Policy evaluation_: Monte-Carlo policy evaluation, $Q=q\_\pi$
_Policy improvement_: Greedy policy improvement

### $\epsilon$-Greedy Exploration
* Simplest idea for ensuring continual exploration
* All $m$ actions are tried with non-zero probability
* With probability $1-\epsilon$ choose the greedy action
* With probability $\epsilon$ choose an action at random
$$ \pi(a|s) = \begin{cases} \epsilon/m + 1 -\epsilon \quad &\text{if } a^* = \arg\max\_{a\in A} Q(s,a) \\\\ \epsilon/m &\text{otherwise}\end{cases}$$

> **Theorem**
> For any $\epsilon$-greedy policy $\pi$, the $\epsilon$-greedy policy $\pi'$ with respect to $q\_\pi$ is an improvement, $v\_{\pi'} (s) \ge v\_\pi (s) $

**Monte-Carlo Policy Iteration**
* _Policy evaluation_ Monte-Carlo policy evaluation, $Q=q\_\pi$
* _Policy improvement_ $\epsilon$-greedy policy improvement 

**Monte-Carlo Control**
Every episode
* _Policy evaluation_ Monte-Carlo policy evaluation $Q\approx q\_\pi$
* _Policy improvement_ $\epsilon$-greedy policy improvement

### GLIE
> **Definition**
> Greedy in the Limit with Infinite Exploration(GLIE)
> * All state-action pairs are explored infinitely many times
> $$ \lim\_{k\rightarrow \infty} N\_k (s,a) =\infty $$
> * The policy converges on a greedy policy
> $$ \lim\_{k \rightarrow \infty} \pi\_k (a|s) = 1(a = \arg\max\_{a' \in A} Q\_k (s,a') )$$

**GLIE Monte-Carlo Control**
* Sample $k$th episode using $\pi$: $(S\_1,A\_1,R\_2,\dots,S\_T) \sim \pi$
* For each state $S\_t$ and action $A\_t$ in the episode
$$ N(S\_t,A\_t) \leftarrow N(S\_t,A\_t) +1 $$
$$ Q(S\_t,A\_t) \leftarrow Q(S\_t,A\_t) + \frac{1}{N(S\_t,A\_t)}(G\_t - Q(S\_t,A\_t)) $$
* Improve policy based on new action-value function
$$ \epsilon \leftarrow 1/k\qquad \pi \leftarrow \epsilon-\text{greedy}(Q) $$
> **Theorem**
> GLIE Monte-Carlo control converges to the optimal action-value function, $Q(s,a) \rightarrow q\_* (s,a)$

## On-policy Temporal-Difference Learning
**MC vs. TD Control**
* TD learning has several advantages over Monte-Carlo(MC)
    * Lower variance
    * Online
    * Incomplete sequences
* Natural idea: use TD instead of MC in our control loop
    * Apply TD to $Q(S,A)$
    * Use $\epsilon$-greedy policy improvement
    * Update every time-step

### Sarsa($\lambda$)
Updating Action-Value Functions with Sarsa
$$ Q(S,A) \leftarrow Q(S,A) + \alpha(R+\gamma Q(S',A') -Q(S,A)) $$

On-Policy Control With Sarsa
Every time-step:
Policy evaluation Sarsa, $Q\approx q\_{\pi} $
Policy improvement $\epsilon$-greedy policy improvement

**Sarsa Algorithm for On-Policy Control**
![Sarsa Algorithm for On-Policy Control](http://i1.piimg.com/567571/ded877ca6da675a4.png)

**Convergence of Sarsa**
> **Theorem**
> Sarsa converges to the optimal action-value function, $Q(s,a) \rightarrow q\_* (s,a)$, under the following conditions:
> * GLIE sequence of policies $\pi\_t (a|s)$
> * Robbins-Monro sequence of step-sizes $\alpha\_t$
> $$ \sum\_{t=1}^{\infty} \alpha\_t = \infty \qquad \sum\_{t=1}^{\infty} \alpha\_t^2 < \infty $$

**n-Step Sarsa**
* Define the n-step Q-return 
$$ q\_t^{(n)} = R\_{t+1} + \gamma R\_{t+2} + \dots + \gamma^{n-1} R\_{t+n} + \gamma^n Q(S\_{t+n}) $$
* n-step Sarsa updates $Q(s,a)$ towards the n-step Q-return 
$$ Q(S\_t,A\_t) \leftarrow Q(S\_t,A\_t) + \alpha (q\_t^{(n)} - Q(S\_t,A\_t)) $$

**Forward View Sarsa($\lambda$)**
* The $q^\lambda$ return combines all n-step Q-return $q\_t^{(n)}$
* Using weight $(1-\lambda)\lambda^{n-1}$
$$ q\_t^{\lambda} = (1-\lambda) \sum\_{n=1}^{\infty} \lambda^{n-1} q\_t^{(n)} $$
* Forward-view Sarsa($\lambda$) 
$$ Q(S\_t,A\_t) \leftarrow Q(S\_t,A\_t) + \alpha (q\_t^\lambda - Q(S\_t,A\_t)) $$

**Backward View Sarsa($\lambda$)**
* Just like TD($\lambda$), we use eligibility traces in an online algorithm
* But Sarsa($\lambda$) has one eligibility trace for each state-action pair
$$ E\_0 (s,a) = 0 \qquad E\_t(s,a) = \gamma \lambda E\_{t-1} (s,a) + 1(S\_t = s,A\_t =a) $$
* $Q(s,a)$ is updated for every state $s$ and action $a$
* In proportion to TD-error $\delta\_t$ and eligibility trace $E\_t(s,a)$
$$ \delta\_t = R\_{t+1} + \gamma Q(S\_{t+1},A\_{t+1}) -Q(S\_t,A\_t)$$
$$ Q(s,a) \leftarrow Q(s,a) + \alpha \delta\_t E\_t(s,a) $$

**Sarsa($\lambda$) Algorithm**
![Sarsa($\lambda$) Algorithm](http://i4.buimg.com/567571/aabc3342a37e0958.png)

## Off-Policy Learning
* Evaluate target policy $\pi(a|s)$ to compute $v\_\pi (s)$ or $q\_\pi (s,a)$
* While following behaviour policy $\mu(a|s)$
$$ (S\_1,A\_1,R\_2,\dots,S\_T) \sim \mu $$
* Why is this important?
    * Learn from observing humans or other agents
    * Re-use experience generated from old policies $\pi\_1,\pi\_2,\dots,\pi\_{t-1}$
    * Learn about optimal policy while following exploratory policy
    * Learn about multiple policies while following one policy

### Importance Sampling
Estimate the expectation of a different distribution
$$ E\_{X\sim P} [f(X)] = \sum P(X) f(X) = \sum Q(X) \frac{P(X)}{Q(X)} f(X) = E\_{X\sim Q} \left[ \frac{P(X)}{Q(X)} f(X)\right] $$

**Importance Sampling for Off-Policy Monte-Carlo**
* Use returns generated from $\mu$ to evaluate $\pi$
* Weight return $G\_t$ according to similarity between policies
* Multiply importance sampling corrections along whole episode
$$ G\_t^{\pi/\mu} = \frac{\pi(A\_t|S\_t) \pi(A\_{t+1}|S\_{t+1}) \dots \pi(A\_T|S\_T)}{\mu(A\_t|S\_t) \mu(A\_{t+1}|S\_{t+1}) \dots \mu(A\_t|S\_t)}G\_t $$
* Update value towards corrected return 
$$ V(S\_t) \leftarrow V(S\_t) + \alpha (G\_t^{\pi/\mu} - V(S\_t)) $$
* Cannot use if $\mu$ is zero when $\pi$ is non-zero
* Importance sampling can dramatically increase variance

On the other hand, we can get lower variance using TD as follow:
* Use TD target generated from $\mu$ to evaluate $\pi$
* Weight TD target $R+\gamma V(S')$ by importance sampling 
* Only need a single importance sampling correction 
$$ V(S\_t) \leftarrow V(S\_t) + \alpha \left( \frac{\pi(A\_t|S\_t)}{\mu(A\_t|S\_t)}(R\_{t+1} + \gamma V(S\_{t+1})) -V(S\_t) \right)$$
* Much lower variance than Monte-Carlo importance sampling
* Policies only need to be similar over a single step

### Q-learning
* We now consider off-policy learning of action-values $Q(s,a)$
* No importance sampling is required
* Next action is chosen using behaviour policy $A\_{t+1} \sim \mu (\cdot |S\_t) $
* But we consider alternative successor action $A' \sim \pi(\cdot| S\_t)$
* And update $Q(S\_t,A\_t)$ towards value of alternative action 
$$ Q(S\_t,A\_t) \leftarrow Q(S\_t,A\_t) + \alpha (R\_{t+1} + \gamma Q(S\_{t+1},A') - Q(S\_t,A\_t)) $$

### Off-policy Control with Q-learning
* We now allow both behaviour and target policies to improve
* The target policy $\pi$ is greedy w.r.t $Q(s,a)$
$$ \pi (S\_{t+1}) = \arg\max\_{a'} Q(S\_{t+1},a') $$
* The behaviour policy $\mu$ is e.g. $\epsilon$ w.r.t $Q(s,a)$
* The Q-learning target then simplifies
$$ R\_{t+1} + \gamma Q(S\_{t+1},A') = R\_{t+1} + \gamma Q(S\_{t+1},\arg\max\_{a'} Q(S\_{t+1},a') ) = R\_{t+1}+ \max\_{a'} \gamma Q(S\_{t+1},a') $$

>**Theorem**
> Q-learning control converges to the optimal action-value function $Q(s,a) \rightarrow q\_* (s,a) $

### Q-learning Algorithm for Off-Policy Control
![Q-learning Algorithm for Off-Policy Control](http://i1.piimg.com/567571/820b3bf364cbe260.png)

## Summary
### Relationship between DP and TD
![Realtionship between DP and TD](http://i1.piimg.com/567571/f3670d774cd7c68e.png)

![Relationship between DP and TD(2)](http://i1.piimg.com/567571/44bcc7082678b5f5.png)
where $x \leftarrow^{\alpha} y \iff x \leftarrow x + \alpha (y-x) $
