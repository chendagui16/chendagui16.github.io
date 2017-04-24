---
title: RL3 Planning by Dynamic Programming
mathjax: true
date: 2017-04-05 20:11:50
categories: reinforcement learning
tags: [reinforcement learning, machine learning]
---
reference:
    [UCL Course on RL](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html)
# lecture 3 Planning by Dynamic Programming

## introduction

### Definition 
> Dynamic: sequential or temporal component to the problem
> Programming: optimising a "program", i.e. a policy

* A method for solving complex problems
* By breaking them down into sub-problems
    * Solve the sub-problems
    * Combine solutions to sub-problems
    
### Requirements 

Dynamic Programming is a very general solution method for problems which have two properties:
1. Optimal substructure
    * Principle of optimality applies
    * Optimal solution can be decomposed into sub-problems
1. Overlapping sub-problems
    * Sub-problems recur many times
    * Solution can be cached and reused

Markov decision processes satisfy both properties
* Bellman equation gives recursive decomposition
* Value function stores and reuses solutions

### Planning by DP
* DP assumes full knowledge of the MDP
* It is used for planning in an MDP
* For prediction
    * Input: MDP (S,A,P,R,y) and policy $\pi$
    * or: MRP $(S,P^{\pi},R^{\pi},y)$
    * Output: value function $v\_{\pi}$
* Or for control:
    * Input: MDP (S,A,P,R,y)
    * Output: optimal value function $v\_*$
    * and: optimal policy $\pi\_*$

## Policy Evaluation
### Iterative Policy Evaluation
* Problem: evaluate a given policy $\pi$
* Solution: iterative application of Bellman expectation backup
* Using synchronous backups
    * At each iteration $k+1$
    * For all states $s\in S$
    * Update $v\_{k+1} (s)$ from $v\_k(s')$
    * where $s'$ is a successor state of $s$
$$ v^{k+1} = R^{\pi} + y P^{\pi} v^k $$

## Policy Iteration
### Policy Improvement
* Given a policy $\pi$
    * Evaluate the policy $\pi$
    * Improve the policy by acting greedily with respect to $v\_{\pi}$
* In general, need more iterations of improvement/evaluation
* But this process of policy iteration always converges to $\pi^*$

** STEP **
1. Consider a deterministic policy, $a=\pi (s) $
1. We can improve the policy by acting greedily 
1. This improves the value from any state $s$ over one step
1. It therefore improves the value function, $v\_{\pi'} (s) \ge v\_{\pi} (s) $
1. If improvements stop, $q\_{\pi} (s,\pi'(s)) = v\_{\pi} (s)$
1. Then the Bellman optimality equation has been satisfied $v\_{\pi} (s) = \max\_{a\in A} q\_{\pi} (s,a)$
1. Therefore $v\_{\pi} (s) = v\_* (s) $ for all $s\in S$, so $\pi$ is an optimal policy

### Extension to Policy Iteration
* Does policy evaluation need to converge to $v\_{\pi}$
* Or should we introduce a stopping condition
* Or simply stop after k iterations of iterative policy evaluation

## Value Iteration
### Principle of Optimality
Any optimal policy can be subdivided into two components
* An optimal first action $A\_*$
* Followed by an optimal policy from successor state $S'$

> ** Theorem **
> A policy $\pi (a|s)$ achieves the optimal value from state $s$, $v\_{\pi} (s) = v\_* (s)$, if and only if 
> * For any state $s'$ reachable from s
> * $\pi$ achieves the optimal value from state $s'$

### Deterministic Value Iteration
* If we know the solution to sub-problems $v\_*(s')$
* Then solution $v\_*(s)$ can be found by one-step lookahead
* The idea of value iteration is to apply these updates iteratively
* Intuition: start with final rewards and work backwards
* Still works with loopy, stochastic MDPs

### Value Iteration
* **Problem**: find optimal policy $\pi$
* **Solution**: iterative application of Bellman optimality backup
* Unlike policy iteration, there is no explicit policy
* Intermediate value functions may not correspond to any policy

### Summary of DP algorithms
| Problem | Bellman Equation | Algorithm |
|:--------| :---------------:| :---------|
| Prediction | Bellman Expectation Equation | Iterative Policy Evaluation |
| Control | Bellman Expectation Equation + Greedy Policy Improvement | Policy Iteration |
| Control | Bellman Optimality Equation | Value Iteration |

* Algorithms are based on state-value function $v\_{\pi} (s) $ or $v\_* (s) $
* Complexity $O(mn^2)$ per iteration, for $m$ action and $n$ states
* Could also apply to action-value function $q\_{\pi} (s,a)$ or $q\_* (s,a)$
* Complexity $O(m^2 n^2)$ per iteration

## Extensions to DP

### Asynchronous DP
* Asynchronous DP backs up states individually, in any order
* For each selected state, apply the appropriate backup
* Can significantly reduce computation
* Guaranteed to converge if all states continue to be selected

**Three simple ideas**
* In-place dynamic programming
> Synchronous value iteration stores two copies of value function for all $s$ in $S$
> In-place value iteration only stores one copy of value function for all $s$

* Prioritised sweeping
> Use magnitude of Bellman error to guide state selection
> Backup the state with the largest remaining Bellman error
> Update Bellman error of affected states after each backup
> Require knowledge of reverse dynamics
> Can be implemented efficiently by maintaining a priority queue

* Real-time dynamic programming 
> Idea: only states that are relevant to agent
> Use agent's experience to guide the selection of states
> After each time-step $S\_t,A\_t,R\_{t+1}$
> Backup the state $S\_t$

### Full-width and sample backups
**Full-width Backups**
* DP uses full-width backups
* For each backup (sync or async)
    * Every successor state and action is considered
    * Using knowledge of the MDP transitions and reward function
* DP is effective for medium-sized problems (millions of states)
* For large problems DP suffers Bellman's *curse of dimensionality*
* Even one backup can be too expensive

**Sample Backups**
* Instead of reward function R and transition dynamics P
* Advantages
    * Model-free: no advance knowledge of MDP required
    * Breaks the curse of dimensionality through sampling
    * Cost of backup is constant, independent of $n=|S|$

### Approximate Dynamic Programming
* Approximate the value function
* Using a function approximator $\hat{v} (s,w)$
* Apply dynamic programming to $\hat{v} (\cdot,w)$

## Contraction Mapping
Contraction Mapping resolves that convergence problem such as converge or not, uniqueness, and converge speed

### Value function Space
* Consider the vector space $V$ over value functions
* There are |S| dimensions
* Each points in this space fully specifies a value function
* Bellman backup brings values functions closer
* And therefore the backups must converge on a unique solution

### Bellman Expectation Backup is a Contraction
When use the $\infty$ norm as the distance metric, we have
* Define the Bellman expectation backup operator $T^{\pi}$
$$ T^{\pi} (v) = R^{\pi} + yP^{\pi} v $$
* This operator is a y-contraction, it makes value functions closer by at least y
![Contraction mapping](http://i1.piimg.com/567571/ed3f6444e4b26860.png)

> **Theorem** (Contraction Mapping Theorem)
> For any metric space $V$ that is complete under an operator $T(v)$, where $T$ is a y-contraction
> * $T$ converges to a unique fixed point
> * At a linear convergence rate of y

**Convergence of Iter. Policy Evaluation and Policy Iteration**
* The Bellman expectation operator $T^{\pi}$ has a unique fixed point
* $v\_{\pi}$ is a fixed point of $T^{\pi}$ (by Bellman expectation equation)
* By contraction mapping theorem 
* Iterative policy evaluation converges on $v\_{\pi}$
* Policy iteration converges on $v\_\*$

### Bellman Optimality Backup is a Contraction
* Define the Bellman optimality backup operator $T^\*$
$$ T^\* (v) = \max_{a\in A} R^a + y P^a v$$
* This operator is a y-contraction, it makes value function closer by at least $y$
$$ ||T^\* (u) - T^\* (v) ||\_{\infty} \le y||u-v||\_{\infty} $$

**Convergence of Value Iteration**
* The bellman optimality operator $T^\*$ has a unique fixed point
* $v\_\*$ is a fixed point of $T^\*$ (by Bellman optimality equation)
* By contraction mapping theorem
* Value iteration converges on $v\_\*$
