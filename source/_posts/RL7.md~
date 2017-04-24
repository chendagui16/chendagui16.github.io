---
title: RL7 Policy Gradient
mathjax: true
date: 2017-04-18 19:27:51
categories: reinforcement learning
tags: [reinforcement learning, machine learning]
---
reference:
    [UCL Course on RL](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html)
# lecture 7 Policy Gradient
## Introduction
**Policy-Based Reinforcement Learning**
* In the last lecture we approximated the value or action-value function using parameter $\theta$
* A policy was generated directly from the value function
* In this lecture, we will directly parametrise the **policy**
$$ \pi\_{\theta} (s,a) = P(a|s,\theta) $$
* We will focus again on model-free reinforcement learning

**Value-Based and Policy-Based RL**
* Value Based
    * Learnt Value Function
    * Implicit policy (e.g. $\epsilon$-greedy)
* Policy Based
    * No Value Function
    * Learnt Policy 
* Actor-Critic
    * Learnt Value Function
    * Learnt Policy

**Advantages of Policy-Based RL**
Advantages:
* Better convergence properties
* Effective in high-dimensional or continuous action spaces
* Can learn stochastic policies
Disadvantages:
* Typically converge to a local rather than global optimum
* Evaluating a policy is typically inefficient and high variance

### Policy Search
**Policy Objective Functions**
* Goal: given policy $\pi\_{\theta} (s,a) $ with parameters $\theta$, find best $\theta$.
* But how do we measure the quality of a policy $\pi\_theta$?
* In episodic environments we can use the start value.
$$ J\_1 (\theta) = V^{\pi\_\theta} (s\_1) = E\_{\pi\_\theta} (v\_1) $$
* In continuing environments we can use the average value
$$ J\_{avV} (\theta) = \sum\_{s} d^{\pi\_\theta} (s) V^{\pi\_\theta} (s) $$
* Or the average reward per time-step
$$ J\_{avR} (\theta) = \sum\_{s} d^{\pi\_\theta} (s) \sum\_{a} \pi\_\theta (s,a) R\_s^a $$
* where $d^{\pi\_\theta} (s)$ is **stationary distribution** of Markov chain for $\pi\_\theta$

### Policy Optimisation
* Policy based reinforcement learning is an **optimisation** problem
* Find $\theta$ that maximises $J(\theta)$ 
* Some approaches do not use gradient 
    * Hill climbing 
    * Simplex/amoeba/Nelder Mead
    * Genetic algorithms
* Greater efficiency often possible using gradient
    * Gradient descent
    * Conjugate gradient
    * Quasi-Newton
* We focus on gradient descent, many extensions possible
* And on methods that exploit sequential structure

## Finite Difference Policy Gradient
**Policy Gradient**
* Let $J(\theta)$ be any policy objective function
* Policy gradient algorithms search for a local maximum in $J(\theta)$ by ascending the gradient of the policy, w.r.t parameters $\theta$
$$ \Delta \theta = \alpha \nabla\_\theta J(\theta) $$
* Where $\Delta\_\theta J(\theta)$ is the **policy gradient**, and $\alpha$ is a step-size parameter

**Computing Gradients By Finite Differences**
* To evaluate policy gradient of $\pi\_\theta (s,a)$
* For each dimension $k \in [1,n]$
    * Estimate $k$th partial derivative of objective function w.r.t $\theta$
    * By perturbing $\theta$ by small amount $\epsilon$ in $k$th dimension
* Uses $n$ evaluations to compute policy gradient in $n$ dimensions
* Simple, noisy, inefficient - but sometimes effective
* Works for arbitrary policies, even if policy is not differentiable

## Monte-Carlo Policy Gradient
### Likelihood Ratios
**Score Function**
* We now compute the policy gradient analytically
* Assume policy $\pi\_\theta$ is differentiable whenever it is non-zero
* and we know the gradient $\nabla\_\theta \pi\_\theta (s,a) $
* **Likelihood ratios** exploit the following identity
$$ \nabla\_\theta \pi\_\theta (s,a) = \pi\_\theta (s,a) \frac{\nabla\_\theta \pi\_\theta (s,a)}{\pi\_\theta (s,a)} = \pi\_{\theta} (s,a) \nabla\_\theta \log \pi\_\theta (s,a) $$
* The **score function** is $\nabla\_\theta \log \pi\_{\theta} (s,a)$

**Softmax Policy**
* We will use a softmax policy as a running example
* Weight actions using linear combination of features $\phi(s,a)^T \theta$
* Probability of action is proportional to exponential weight
$$ \pi\_\theta (s,a) \propto e^{\phi(s,a)^T \theta} $$
* The score function is 
$$ \nabla\_\theta \log \pi\_\theta (s,a) = \phi(s,a) - E\_{\pi\_\theta} (\phi(s,\cdot)) $$

**Gaussian Policy**
* In continuous action spaces, a Gaussian policy is natural
* Mean is a linear combination of state feature $\mu(s) = \phi(s)^T \theta$
* Variance may be fixed $\sigma^2$, or can also parametrised
* Policy is Gaussian, $a \sim N(\mu(s),\sigma^2)$
* The score function is
$$ \nabla\_\theta \log \pi\_{\theta} (s,a) = \frac{(a-\mu(s))\phi(s)}{\sigma^2} $$

### Policy Gradient Theorem
**One-Step MDPs**
* Consider a simple class of **one-step** MDPs
    * Starting in state $s\sim d(s)$
    * Terminating after one time-step with reward $r=R\_{s,a}$
* Use likelihood ratios to compute the policy gradient
$$ J(\theta) = E\_{\pi\_\theta} (r) = \sum\_{s \in S} d(s) \sum\_{a \in A} \pi\_\theta (s,a) R\_{s,a} \\\\ \nabla\_\theta J(\theta) = \sum\_{s \in S} d(s) \sum\_{a \in A} \pi\_\theta (s,a) \nabla\_\theta \log \pi\_\theta (s,a) R\_{s,a} = E\_{\pi\_\theta} (\nabla\_\theta \log \pi\_\theta (s,a) r)  $$

**Policy Gradient Theorem**
* The policy gradient theorem generalised the likelihood ratio approach to multi-step MDPs
* Replaces instantaneous reward $r$ with long-term value $Q^\pi (s,a)$
* Policy gradient theorem applies to start state objective, average reward and average value objective, average reward and average value objective
>**Theorem**
> For any differentiable policy $\pi\_\theta(s,a)$, for any of the policy objective functions $J=J\_1,J\_{avR}$ or $\frac{1}{1-\gamma} J\_{avV}$, the policy gradient is 
> $$ \nabla\_\theta J(\theta) = E\_{\pi\_\theta} \left[ \nabla\_\theta \log \pi\_\theta (s,a) Q^{\pi\_\theta} (s,a)\right] $$

**Monte-Carlo Policy Gradient(REINFORCE)**
* Update parameters by stochastic gradient ascent
* Using policy gradient theorem 
* Using return $v\_t$ as an unbiased sample of $Q^{\pi\_\theta} (s\_t,a\_t)$
$$ \Delta \theta\_t = \alpha \nabla\_\theta \log \pi\_\theta (s\_t,a\_t)v\_t $$
![PG algorithm](http://i1.piimg.com/567571/54c0e94c6b0f2936.png)

## Actor-Critic Policy Gradient
### Introduction to AC
**Reducing Variance Using a Critic**
* Monte-Carlo policy gradient still has high variance
* We use a **critic** to estimate the action-value function, $ Q\_w (s,a) \approx Q^{\pi\_\theta} (s,a)$
* Actor-critic algorithms maintain two sets of parameters
    * **Critic** Updates action-value function parameters $w$
    * **Actor** Updates policy parameters $\theta$, in direction suggested by critic
* Actor-critic algorithms follow an approximate policy gradient
$$ \nabla\_\theta J(\theta) \approx E\_{\pi\_\theta} (\nabla\_\theta \log \pi\_\theta (s,a) Q\_w (s,a) ) \\\\ \Delta \theta = \alpha \nabla\_\theta \log \pi\_\theta (s,a) Q\_w (s,a) $$

**Estimating the Action-Value Function**
* The critic is solving a familiar problem: **policy evaluation**
* How good is policy $\pi\_theta$ for current parameters $\theta$?
* This problem was explored in previous two lectures, e.g.
    * Monte-Carlo policy evaluation
    * Temporal-Difference learning
    * TD($\lambda$)
* Could also use e.g least-squares policy evaluation

**Action-value Actor-Critic**
* Simple actor-critic algorithm based on action-value critic
* Using linear value fn approx $Q\_w (s,a) = \phi(s,a)^T w $
    * **Critic** Updates $w$ by linear TD(0)
    * **Actor** Updates $\theta$ by policy gradient 
![Simple actor-critic algorithm](http://i1.piimg.com/567571/bcf6e47f246e7bf9.png)

### Compatible Function Approximation
**Bias in Actor-Critic Algorithms**
* Approximating the policy gradient introduces bias
* A biased policy gradient may not find the right solution
    * e.g if $Q\_w(s,a)$ uses aliased features, can we solve gridworld example?
* Luckily, if we choose value function approximation carefully
* Then we can avoid introducing any bias
* i.e We can still follow the exact policy gradient 

**Compatible Function Approximation**
>**Theorem(Compatible Function Approximation Theorem)**
> If the following two conditions are satisfied
> 1. Value function approximator is compatible to the policy
> $$ \nabla\_w Q\_w(s,a) = \nabla\_\theta \log \pi\_\theta (s,a) $$
> 1. Value function parameters $w$ minimise the mean-squared error
> $$ \epsilon = E\_{\pi\_\theta} \left[ (Q^{\pi\_\theta}(s,a) -Q\_w(s,a))^2\right] $$
> Then the policy gradient is exact
> $$ \nabla\_\theta J(\theta) = E\_{\pi\_\theta} \left[ \nabla\_\theta \log \pi\_\theta (s,a) Q\_w (s,a) \right] $$

**Proof of Compatible Function Approximation Theorem**
If $w$ is chosen to minimise mean-squared error, gradient of $\epsilon$ w.r.t $w$ must be zero
![Proof](http://i2.muimg.com/567571/ef9684519bf5952d.png)
So $Q\_w (s,a)$ can be substituted directly into the policy gradient
$$ \nabla\_\theta J(\theta) = E\_{\pi\_\theta} \left[ \nabla\_\theta \log \pi\_\theta (s,a) Q\_w (s,a) \right] $$

### Advantage Function Critic
**Reducing Variance Using a Baseline**
* We subtract a baseline function $B(s)$ from the policy gradient
* This can reduce variance, without changing expectation
$$ E\_{\pi\_\theta} \left[ \nabla\_\theta \log \pi\_\theta (s,a) B(s) \right] = \sum\_{s \in S} d^{\pi\_\theta} (s) \sum\_s \nabla\_\theta \pi\_\theta (s,a) B(s) \\\\ = \sum\_{s \in S} d^{\pi\_\theta} B(s) \nabla\_\theta \sum\_{a \in A} \pi\_\theta (s,a) = 0 $$
* A good baseline is the state value function $B(s) = V^{\pi\_\theta} (s)$
* So we can rewrite the policy gradient using the advantage function $A^{\pi\_\theta} (s,a)$
$$ A^{\pi\_\theta} (s,a) = Q^{\pi\_\theta} (s,a) - V^{\pi\_\theta} (s) \\\\ \nabla\_\theta J(\theta) = E\_{\pi\_\theta} \left[ \nabla\_\theta \log \pi\_\theta (s,a) A^{\pi\_\theta} (s,a) \right] $$

**Estimating the Advantage Function**
* The advantage function can significantly reduce variance of policy gradient
* So the critic should really estimate the advantage function
* For example, by estimating both $V^{\pi\_\theta} (s)$ and $Q^{\pi\_\theta} (s,a)$
* Using two function approximating and two parameter vectors
$$ V\_v (s) \approx V^{\pi\_\theta} (s) \\\\ Q\_w (s,a) \approx Q^{\pi\_\theta} (s,a) \\\\ A(s,a) = Q\_w(s,a) - V\_v (s) $$
* And updating both value functions by e.g TD learning 
* For the true value function $V^{\pi\_\theta} (s)$, the TD error $\delta^{\pi\_\theta}$
$$ \delta^{\pi\_\theta} = r + \gamma V^{\pi\_\theta} (s') - V^{\pi\_\theta} (s)$$
* is an unbiased estimate of the advantage function
$$ E\_{\pi\_\theta} \left[ \delta^{\pi\_\theta} | s,a\right]=  E\_{\pi\_\theta} \left[r+\gamma V^{\pi\_\theta} (s') | s,a \right] - V^{\pi\_\theta} (s) \\\\ = Q^{\pi\_\theta} (s,a) -V^{\pi\_\theta} (s) = A^{\pi\_\theta} (s,a) $$ 
* So we can use the TD error to compute the policy gradient 
$$ \nabla\_\theta J(\theta) = E\_{\pi\_\theta} \left[ \nabla\_\theta \log \pi\_\theta (s,a) \delta^{\pi\_\theta} \right] $$
* In practice we can use an approximate TD error
$$ \delta\_v = r + \gamma V\_v (s') - V\_v (s) $$
* This approach only requires one set of critic parameters $v$

### Eligibility Traces
**Critics at Different Time-Scales**
* Critic can estimate value function $V\_\theta(s)$ from many targets at different time-scales 
    * For MC, the target is the return $v\_t$
    $$ \Delta \theta = \alpha (v\_t - V\_\theta(s)) \phi(s) $$
    * For TD(0), the target is the TD target $r+ \gamma V(s')$
    $$ \Delta \theta = \alpha (r + \gamma V(s')  - V\_\theta(s)) \phi(s) $$
    * For forward-view TD($\lambda$), the target is the $\lambda$-return $v\_t^\lambda$
    $$ \Delta \theta =\alpha (v\_t^\lambda - V\_\theta (s)) \phi(s) $$
    * For backward-view TD($\lambda$), we use eligibility traces
    $$ \delta\_t = r\_{t+1} + \gamma V(s\_{t+1}) - V(s\_t) \\\\ e\_t = \gamma \lambda e\_{t-1} + \phi(s\_t) \\\\ \Delta \theta = \alpha \delta\_t e\_t $$

**Policy Gradient with Eligibility Traces**
* Just like forward-view TD($\lambda$), we can mix over time-scales
$$ \Delta \theta = \alpha (v\_t^\lambda - V\_v (s\_t) ) \nabla\_\theta \log \pi\_\theta (s\_t,a\_t) $$
* where $v\_t^\lambda -V\_v(s\_t)$ is a biased estimate of advantage fn
* Like backward-view TD($\lambda$), we can also use eligibility traces
    * By equivalence with TD($\lambda$), substituting $\phi(s) = \nabla\_\theta \log \pi\_\theta (s,a)$
    $$ \delta\_t = r\_{t+1} + \gamma V(s\_{t+1}) - V(s\_t) \\\\ e\_{t+1} = \lambda e\_{t} + \nabla\_\theta \log \pi\_\theta (s,a)  \\\\ \Delta \theta = \alpha \delta\_t e\_t $$
* This update can be applied online, to incomplete sequences

### Natural Policy Gradient 
**Alternative Policy Gradient Direction**
* Gradient ascent algorithm can follow any ascent direction
* A good ascent direction can significantly speed convergence
* Also, a policy can often be reparametrised without changing action probabilities
* For example, increasing score of all actions in a softmax policy
* The vanilla gradient is sensitive to these reparametrisations

**Natural Policy Gradient**
* The **natural policy gradient** is parametrisation independent
* It finds ascent direction that is closet to vanilla gradient, when changing policy by a small, fixed amount
$$ \nabla\_\theta^{nat} \pi\_\theta (s,a) = G\_\theta^{-1} \nabla\_\theta \pi\_\theta (s,a) $$
* where $G\_\theta$ is the Fisher information matrix
$$ G\_\theta = E\_\theta \left[ \nabla\_\theta \log \pi\_\theta (s,a) \nabla\_\theta \log \pi\_\theta (s,a)^T \right] $$

**Natural Actor-Critic**
* Using compatible function approximation
$$ \nabla\_w A\_w (s,a) = \nabla\_\theta \log \pi\_\theta (s,a) $$
* So the natural policy gradient simplifies,
$$ \nabla\_\theta J(\theta) = E\_{\theta\_\pi} \left[ \nabla\_\theta \log \pi\_\theta (s,a) A^{\pi\_\theta} (s,a) \right] \\\\ = E\_{\pi\_\theta} \left[ \nabla\_\theta \log \pi\_\theta (s,a) \nabla\_\theta \log \pi\_\theta (s,a)^T w \right] = G\_\theta w $$
so we can get $ \nabla\_\theta^{nat} J(\theta) = w $
* update actor parameters in direction of critic parameters

## Summary of Policy Gradient Algorithms
* The **policy gradient** has many equivalent forms
    ![PG equivalent forms](http://i2.muimg.com/567571/45365e91fb7af247.png)
* Each leads a stochastic gradient ascent algorithm
* Critic uses **policy evaluation** (e.g MC or TD learning) to estimate $Q^\pi (s,a), A^\pi (s,a)$ or $V^\pi (s)$
