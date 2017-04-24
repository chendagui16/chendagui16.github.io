---
title: RL4 Model-Free Prediction
mathjax: true
date: 2017-04-06 19:22:03
categories: reinforcement learning
tags: [reinforcement learning, machine learning]
---
reference:
    [UCL Course on RL](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html)
# Lecture 4 Model-Free Prediction
Estimate the value function of an unknown MDP

## Monte-Carlo Reinforcement learning
* MC methods learn directly from episodes of experience
* MC is model-free: no knowledge of MDP transition/rewards
* MC learns from complete episodes: no bootstrapping
* MC uses the simplest possible idea: value = mean return
* Caveat: can only apply MC to episodic MDPs
    * All episodes must terminate

### MC Policy Evaluation
* **Goal**: learn $v\_{\pi}$ from episodes of experience under policy $\pi$, $S\_1,A\_1,R\_2,\dots,S\_k \sim \pi $
* Recall that the return is the total discounted reward: $G\_t = R\_{t+1} + \gamma R\_{t+2} + \dots + \gamma^{T-1} R\_{T} $
* Recall that the value function is the expected return: $v\_{\pi} (s) = E\_{\pi} [G\_t|S\_t =s]$
* MC policy evaluation uses empirical mean return instead of expected return

### First-Visit MC Policy Evaluation
* To evaluate state $s$
* The **first** time-step $t$ that state $s$ is visited in an episode
* Increment counter $N(s) \longleftarrow N(s) +1 $
* Increment total return $S(s) \longleftarrow S(s) +G\_t$
* Value is estimated by mean return $V(s) = S(s)/N(s) $
* By law of large numbers, $V(s) \rightarrow v\_{\pi} (s) $ as $N(s) \rightarrow \infty$

### Every-Visit MC Policy Evaluation
* To evaluate state $s$
* **Every** time-step $t$ that state $s$ is visited in an episode
* Increment counter $N(s) \longleftarrow N(s) +1 $
* Increment total return $S(s) \longleftarrow S(s) +G\_t$
* Value is estimated by mean return $V(s) = S(s)/N(s) $
* By law of large numbers, $V(s) \rightarrow v\_{\pi} (s) $ as $N(s) \rightarrow \infty$

### Incremental MC
**Incremental Mean**
The mean $\mu_1,\mu_2,\dots$ of a sequence $x\_1,x\_2,\dots$ can be computed incrementally,
$$\mu\_k = \mu\_{k-1} + \frac{1}{k} (x\_k - \mu\_{k-1}) $$

**Incremental MC updates**
* Update $V(s)$ incrementally after episode $S\_1,A\_1,R\_1,\dots,S\_T$
* For each state $S\_t$ with return $G\_t$.
$$ N(S\_t) \leftarrow N(S\_t) + 1 $$
$$ V(S\_t) \leftarrow V(S\_t) + \frac{1}{N(S\_t)} (G\_t - V(S\_t)) $$ 
* In non-stationary problems, it can be useful to track a running mean, i.e. forget old episodes
$$ V(S\_t) \leftarrow V(S\_t) + \alpha (G-V(S\_t)) $$

$\alpha$ is the update rate.

## Temporal-Difference Learning
* TD method learn directly from episodes of experience
* TD is *model-free*: no knowledge of MDP transitions/rewards
* TD learns from *incomplete* episodes, by bootstrapping
* TD updates a guess towards a guess

### TD algorithm
* **Goal**: learn $v\_{\pi}$ online from experience under policy $\pi$
* Incremental every-visit MC
    * Update value $V(S\_t)$ toward actual return $G\_t$
    $$ V(S\_t) \leftarrow V(S\_t) + \alpha (G\_t - V(S\_t)) $$
* Simplest temporal-difference learning algorithm: TD(0)
    * Update value $V(S\_t)$ toward estimated return $R\_{t+1} + \gamma V(S\_{t+1})$
    $$ V(S\_t) \leftarrow V(S\_t) + \alpha (R\_{t+1} + \gamma V(S\_{t+1}) -V(S\_t)) $$
    * $R\_{t+1}+\gamma V(S\_{t+1})$ is called the TD target
    * $\delta\_t = R\_{t+1} + \gamma V(S\_{t+1}) -V(S\_t)$ is called the TD error

### Bias/Variance Trade-off
* Return $G\_t = R\_{t+1} + \gamma R\_{t+2} + \dots + \gamma^{T-1} R\_T$ is unbiased estimate of $v\_{\pi} (S\_t)$
* True TD target $R\_{t+1} + \gamma v\_{\pi} (S\_{t+1}) $ is unbiased estimate of $v\_{\pi}(S\_t)$
* TD target $R\_{t+1} + \gamma V(S\_{t+1})$ is biased estimate of $v\_{\pi} (S\_t)$
* TD target is much lower variance than the return
    * Return depends on many random actions, transitions, rewards
    * TD target depends on one random action, transition, reward

### Batch MC and TD
* MC and TD converge: $V(s) \rightarrow v\_{\pi} (s)$ as experience $\rightarrow \infty$
* But what about batch solution for finite experience?
    * e.g. Repeatedly sample episode $k\in [1,K] $
    * Apply MC or TD(0) to episode $k$

### Certainty Equivalence
* MC converges to solution with minimum mean-squared error
    * Best fit to the observed returns
    $$ \sum\_{k=1}^{K} \sum\_{t=1}^{T\_k} (G\_t^k -V(s\_t^k))^2 $$
    
* TD(0) converges to solution of max likelihood Markov model
    * Solution to the MDP $(S,A,\hat{P},\hat{R},\gamma)$ that best fits the data
    $$ \hat{P}\_{s,s'}^a = \frac{1}{N(s,a)} \sum\_{k=1}^{K} \sum\_{t=1}^{T\_k} 1(s\_t^k,a\_t^k,s\_{t+1}^k = s,a,s') $$
    $$ \hat{R}\_s^a = \frac{1}{N(s,a)} \sum\_{k=1}^{K} \sum\_{t=1}^{T\_k} 1(s\_t^k,a\_t^k = s,a) r\_t^k $$ 
### Advantages and Disadvantages of MC vs. TD
* TD can learn before knowing the final outcome
    * TD can learn online after every step
    * MC must wait until end of episode before return is known
* TD can learn without the final outcome
    * TD can learn from incomplete sequences
    * MC can only learn from complete sequences
    * TD works in continuing (non-terminating) environments
    * MC only works for episodic (terminating) environments
* MC has high variance, zero bias
    * Good convergence properties
    * (even with function approximation)
    * Not every sensitive to initial value
    * Very simple to understand and use
* TD has low variance, some bias
    * Usually more efficient than MC
    * TD(0) converges to $v\_{\pi} (s)$
    * (but not always with function approximation)
    * More sensitive to initial value
* TD exploits Markov property
    * Usually more efficient in Markov environments
* MC does not exploit Markov property
    * Usually more efficient in non-Markov environments

### Unified View
Monte-Carlo Backup
![MC Backup](http://i2.muimg.com/567571/5c127fc9c91b8549.png)
Temporal-Difference Backup
![TD Backup](http://i4.buimg.com/567571/7ff4a8d6a853c8e6.png)
Dynamic Programming Backup
![DP Backup](http://i1.piimg.com/567571/2a0f5cc516550dc4.png)
Unified View of Reinforcement Learning
![Unified View of RL](http://i1.piimg.com/567571/6122d65e7fd835a5.png)

### Bootstrapping and Sampling
* **Bootstrapping**: update involves an estimate
    * MC dose not bootstrap
    * DP/TD bootstraps
* **Sampling**: update samples an expectation
    * MC/TD samples
    * DP does not sample

## TD($\lambda$) 
### n-Step Return
* Consider the following n-step returns for $n=1,2,\dots$
    * n=1 (TD) $G\_t^{(1)}=R\_{t+1} + \gamma V(S\_{t+1})$
    * n=2 $G\_t^{(2)} = R\_{t+1} + \gamma R\_{t+2} + \gamma^2 V(S\_{t+2})$
    * n=$\infty$, $G\_t^{(\infty)} = R\_{t+1} + \gamma R\_{t+2} + \dots + \gamma^{T-1} R\_T $
* Define the n-step return
$$ G\_t^{(n)} = R\_{t+1} + \gamma R\_{t+2} + \dots + \gamma^{n-1} R\_{t+n} + \gamma^{n} V(S\_{t+n})$$
* n-step temporal-difference learning
$$ V(S\_t) \leftarrow V(S\_t) + \alpha (G\_{(n)} - V(S\_t)) $$

### Averaging n-Step Returns
* We can average n-step returns over different n
* e.g average the 2-step and 4-step returns $\frac{1}{2} G^{(2)} + \frac{1}{2} G^{(4)}$
* Combines information from two different time-steps
* Can we efficiently combine information from all time-steps

### $\lambda$ return
* The $\lambda-$return $G\_t^{\lambda}$ combines all n-step return $G\_t^{(n)}$
* Using weight $(1-\lambda)\lambda^{n-1}$
$$ G\_t^{\lambda} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G\_t^{(n)} $$
* Forward-view TD($\lambda$)
$$ V(S\_t) \leftarrow V(S\_t) + \alpha (G\_t^{\lambda} -V(S\_t)) $$

### Forward-view TD($\lambda$)
$$ G\_{t}^{\lambda} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G\_t^{(n)} $$
* Update value function towards the $\lambda$-return
* Forward-view looks into the future to compute $G\_t^{\lambda}$
* Like MC, can only be computed from complete episodes

### Backward-view TD($\lambda$)
* Forward view provides theory
* Backward view provides mechanism
* Update online, every step, from incomplete sequences

### Relationship Between Forward and Backward TD
**TD($\lambda$) and TD(0)**
* when $\lambda=0$, only current state is updated
$$ E\_t (s) = 1(S\_t =s) \qquad V(s) \leftarrow V(s) + \alpha \delta\_t E\_t(s)$$
* This is exactly equivalent to TD(0) update
$$ V(S\_t) \leftarrow V(S\_t) + \alpha \delta\_t $$

**TD($\lambda$) and MC**
* When $\lambda=1$, credit is deferred until end of episode
* Consider episodic environments with offline updates
* Over the course of an episode, total update for TD(1) is the same as total update for MC
> **Theorem**
> The sum of offline updates is identical for forward-view and backward-view TD($\lambda$)
> $$ \sum\_{t=1}^{T} \alpha \delta\_t E\_t(s) = \sum\_{t=1}^T \alpha (G\_t^{\lambda} - V(S\_t) ) 1(S\_t =s) $$

### Forward and Backward Equivalence
**MC and TD(1)**
* Consider an episode where $s$ is visited once at time-step $k$.
* TD(1) eligibility trace discounts time since visit,
$$ E\_t(s) = \gamma E\_{t-1} (s) + 1(S\_t = s) = \begin{cases} 0 & \text{if}~ t<k \\\\  \gamma^{t-k} & \text{if} ~ t\ge k \end{cases}$$
* TD(1) updates accumulate error online
$$ \sum\_{t=1}^{T-1} \alpha \delta\_t E\_t(s) = \alpha \sum\_{t=k}^{T-1} \gamma^{t-k} \delta\_t = \alpha (G\_k - V(S\_k)) $$
* By end of episode it accumulates total error
$$ \delta\_k + \gamma \delta\_{k+1} + \gamma^2 \delta\_{k+2} + \dots + \gamma^{T-1-k} \delta\_{T-1} $$

**TD(\lambda) and TD(1)**
* TD(1) is roughly equivalent to every-visit Monte-Carlo
* Error is accumulated online, step-by-step
* If value function is only updated offline at end of episode
* Then total update is exactly the same as MC

**Forward and Backwards TD($\lambda$)**
* Consider an episode where $s$ is visited once at time-step $k$
* TD($\lambda$) eligibility trace discounts time since visit
$$ E\_t(s) = \gamma\lambda E\_{t-1} (s) + 1(S\_t = s) = \begin{cases} 0 & \text{if}~ t<k \\\\  (\gamma\lambda)^{t-k} & \text{if} ~ t\ge k \end{cases}$$
* Backward TD($\lambda$) updates accumulate error online
$$ \sum\_{t=1}^{T} \alpha \delta\_t E\_t(s) = \alpha \sum\_{t=k}^T (\gamma\lambda)^{t-k} \delta\_t = \alpha(G\_k^\lambda - V(S\_k)) $$
* By end of episode it accumulates total error for $\lambda$-return
* For multiple visits to $s$, $E\_t(s)$ accumulates many errors

**Offline Equivalence of Forward and Backward TD**
Offline updates
* Updates are accumulated within episode
* but applied in batch at the end of episode

Online updates
* TD($\lambda$) updates are applied online at each step within episode
* Forward and backward-view TD($\lambda$) are slightly different
* **New**: Exact online TD($\lambda$) achieves perfect equivalence
* By using a slightly differently form of eligibility trace

### Summary of Forward and Backward TD($\lambda$)
![Summary of Forward and Backward TD($\lambda$)](http://i4.buimg.com/567571/718d23886e0d9e53.png)
