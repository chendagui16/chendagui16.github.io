---
title: RL6 Value Function Approximation
mathjax: true
date: 2017-04-12 20:05:28
categories: reinforcement learning
tags: [reinforcement learning, machine learning]
---
reference:
    [UCL Course on RL](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html)
# lecture 6 Value Function Approximation
## Introduction
**Large-Scale Reinforcement Learning**
Reinforcement learning can be used to solve large problems

**Value Function Approximation**
Value Function by a lookup table
* Every state $s$ has an entry Q(s,a) 
* Or every state-action pair $s,a$ has an entry $Q(s,a)$
Problem with larger MDPs
* There are many states and/or actions to store in memory
* It is too slow to learn the value of each state individually
Solution for large MDPs
* Estimate value functions with function approximation $ \hat{v} (s,w) \approx v\_{\pi}(s) $ or $ \hat{q} (s,a,w) \approx q\_\pi (s,a) $
* Generalise from seen states to unseen states
* Update parameters w using MC or TD learning

### Types of value function approximation
![types of value function approximation](http://i2.muimg.com/567571/7a57308447c99963.png)
We consider differentiable function approximators 
1. Linear combinations of features
1. Neural network
1. Decision tree
1. Nearest neighbour
1. Fourier/wavelet bases
1. $\dots$
Furthermore, we require a training method that is suitable for non-stationary, non-iid data

## Incremental Methods
### Stochastic Gradient Descent
* Goal: find parameters vector $w$ minimising mean-squared error between approximate value fn $\hat{v}(s,w)$ and true value fn $v\_{\pi} (s) $
$$ J(w) = E\_\pi [(v\_\pi (S) -\hat{v} (S,w) )^2] $$
* Gradient descent finds a local minimum 
$$ \Delta w = -\frac{1}{2} \alpha \nabla\_w J(w)= \alpha E\_\pi \left[ (v\_\pi(S) -\hat{v}(S,w))\nabla\_w \hat{v} (S,w) \right] $$
* Stochastic gradient descent samples the gradient
$$ \Delta w = \alpha (v\_\pi (S) - \hat{v} (S,w)) \nabla\_w \hat{v} (S,w) $$
* Expected update is equal to full gradient update

### Linear Function Approximation
**Linear Value Function Approximation**
* Represent value function by a linear combination of features
$$ \hat{S,w} = x(S)^T w = \sum\_{j=1}^{n} x\_j (S) w\_j $$
* Objective function is quadratic in parameters $w$
$$ J(w) = E\_\pi \left[ (v\_\pi(S) - x(S)^T w)^2\right] $$
* Stochastic gradient descent converges on global optimum
* Update rule is particularly simple
_Updata = step-size $\times$ prediction error $\times$ feature value_

**Table Lookup Features**
* Table lookup is special case of linear value function approximation
* Using table lookup features
$$ x^{table} (S) = \begin{bmatrix} 1(S=s\_1) \\\\ \vdots \\\\ 1(S=s\_n) \end{bmatrix} $$
* Parameter vector $w$ gives value of each individual state
$$ \hat{v} (S,w) = \begin{bmatrix} 1(S=s\_1) \\\\ \vdots \\\\ 1(S=s\_n) \end{bmatrix} \cdot \begin{bmatrix} w\_1 \\\\ \vdots \\\\ w\_n \end{bmatrix} $$

### Incremental Prediction Algorithms
* Have assumed true value function $v\_{\pi} (s)$ given by supervisor
* But in RL there is no supervisor, only rewards
* In practice, we substitute a target for $v\_{\pi} (s)$
    * For MC, the target is the return $G\_t$
    $$ \Delta w = \alpha (G\_t - \hat{v} (S\_t,w)) \nabla\_{w} \hat{v} (S\_t,w) $$
    * For TD(0), the target is the TD target $R\_{t+1} + \gamma \hat{v} (S\_{t+1},w) $
    $$ \Delta w = \alpha (R\_{t+1} + \gamma \hat{v} (S\_{t+1},w) - \hat{v} (S\_t,w)) \nabla\_{w} \hat{v} (S\_t,w) $$
    * For TD($\lambda$), the target is the $\lambda$-return $G\_t^\lambda$
    $$ \Delta w = \alpha (G\_t^\lambda - \hat{v} (S\_t,w)) \nabla\_{w} \hat{v} (S\_t,w) $$
    
**MC with Value Function Approximation**
* Return $G\_t$ is an unbiased, noisy sample of true value $v\_{pi} (S\_t)$
* Can therefore apply supervised learning to "training data":
$$ (S\_1, G\_1), (S\_2, G\_2), \dots, (S\_T, G\_T) $$
* For example, using linear MC policy evaluation 
$$ \Delta w = \alpha (G\_t - \hat{v} (S\_t,w)) \Delta\_w \hat{v} (S\_t,w) = \alpha(G\_t - \hat{v} (S\_t,w))x(S\_t) $$
* Monte-Carlo evaluation converges to a local optimum 
* Even when using non-linear value function approximation

**TD Learning with Value Function Approximation**
* The TD-target $R\_{t+1} + \gamma \hat{v} (S\_{t+1},w)$ is a biased sample of true value $v\_\pi (S\_t)$
* Can still apply supervised learning to "training data":
$$ (S\_1, R\_2 + \gamma \hat{v} (S\_2,w) ), (S\_2, R\_3 + \gamma \hat{v} (S\_3,w)), \dots, (R\_{T-1},R\_T) $$
* For example, using linear TD(0)
$$ \Delta w = \alpha (R+\gamma \hat{v} (S',w) - \hat{v} (S,w)) \Delta\_w \hat{v} (S,w) = \alpha \delta x(S) $$
* Linear TD(0) converges (close) to global optimum

### Incremental Control Algorithm
**Control with Value function Approximation**
_Policy evaluation_: Approximate policy evaluation, $\hat{q}(\cdot,\cdot,w) \approx q\_\pi$
_Policy improvement_: $\epsilon$-greedy policy improvement

**Action-Value Function Approximation**
* Approximate the action-value function $\hat{q}(S,A,w) \approx q\_{\pi} (S,A)$
* Minimise mean-squared error between approximate action-value fn $\hat{q}(S,A,w)$ and true action-value fn $q\_\pi (S,A)$
$$ J(w) = E\_\pi \left[(q\_\pi (S,A) - \hat{q} (S,A,w))^2 \right] $$
* Use stochastic gradient descent to find a local minimum 
$$ -\frac{1}{2} \nabla\_w J(w) = ( q\_{\pi} (S,A) -\hat{q} (S,A,w)) \nabla\_w \hat{q} (S,A,w) \\\\ \Delta w = \alpha (q\_{\pi} (S,A) -\hat{q} (S,A,w))\nabla\_w \hat{q} (S,A,w) $$

**Linear Action-Value Funtion Approximation**
* Represent state and action by a feature vector
$$ X(S,A) = \begin{pmatrix} x\_1 (S,A) \\\\ \vdots \\\\ x\_n (S,A) \end{pmatrix} $$
* Represent action-value fn by linear combination of features
$$ \hat{q} (S,A,w) = x(S,A)^T w = \sum\_{j=1}^n x\_j (S,A)w\_j $$
* Stochastic gradient descent update 
$$ \nabla\_w \hat{q} (S,A,w) = x(S,A) \\\\ \Delta w =\alpha (q\_\pi (S,A) - \hat{q} (S,A,w)) x(S,A) $$

**Incremental Control Algorithms**
* Like prediction, we must substitute a target for $q\_\pi (S,A)$
    * For MC, the target is the return $G\_t$
    $$ \Delta w = \alpha (G\_t - \hat{q} (S\_t,A\_t,w)) \nabla\_w \hat{q} (S\_t,A\_t,w) $$
    * For TD(0), the target is the TD target $R\_{t+1} + \gamma Q(S\_{t+1},A\_{t+1}) $
    $$ \Delta w = \alpha (R\_{t+1}+\gamma\hat{q} (S\_{t+1},A\_{t+1},w) - \hat{q} (S\_t,A\_t,w)) \nabla\_w \hat{q} (S\_t,A\_t,w) $$
    * For forward-view TD($\lambda$), target is the action-value $\lambda$-return
    $$ \Delta w = \alpha (q\_t^\lambda - \hat{q} (S\_t,A\_t,w)) \nabla\_w \hat{q} (S\_t,A\_t,w) $$
    * For backward-view TD($\lambda$), equivalent update is
    $$ \delta\_t = R\_{t+1}+\gamma\hat{q} (S\_{t+1},A\_{t+1},w) - \hat{q} (S\_t,A\_t,w) \\\\ E\_t =\gamma \lambda E\_{t-1} + \Delta\_w \hat{q} (S\_t,A\_t,w) \\\\ \Delta w = \alpha \delta\_t E\_t $$

### Convergence
**Convergence of Prediction Algorithms**
![Convergence of Prediction Algorhms](http://i1.piimg.com/567571/cadbe9e888675272.png)
**Gradient Temporal-Difference Learning**
* TD does not follow the gradient of any objective function
* This is why TD can diverge when off-policy or using non-linear function approximation
* _Gradient TD_ follows true gradient of projected Bellman error
![Gradient TD](http://i4.buimg.com/567571/38730a39ee8d3a93.png)

**Convergence of Control Algorithms**
![Convergence of Control Algorithms](http://i2.muimg.com/567571/16ffcde59dce7eca.png)

## Batch Methods
**Batch Reinforcement Learning**
* Gradient descent is simple and appealing
* But it is not sample efficient
* Batch methods seek to find the best fitting value function
* Given the agent's experience ('training data")

### Least Squares Prediction
* Given value function approximation $\hat{v}(s,w) \approx v\_\pi (s) $
* And experience D consisting of (_state_,_value_) pairs
$$ D = ((s\_1,v\_1^\pi),(s\_2,v\_2^\pi),\dots,(s\_T,v\_T^\pi)) $$
* Which parameters $w$ given the best fitting value fn $\hat{v} (s,w)$?
* **Least squares** algorithms find parameters vector $w$ minimising sum-squared error between $\hat{v} (s\_t,w)$ and target values $v\_t^\pi$
$$ LS(w) = \sum\_{t=1}^{T} (v\_t^T - \hat{v} (s\_t,w))^2 = E\_D \left[ (v^\pi - \hat{v} (s,w))^2\right] $$

**Stochastic Gradient Descent with Experience Replay**
Repeat:
1. Sample state, value from experience $(s,v^\pi)\sim D$
1. Apply stochastic gradient descent update $\Delta w = \alpha (v^\pi -\hat{v} (s,w)) \nabla\_w \hat{v} (s,w) $
Converges to least squares solution
$$ w^\pi = \arg\min\_w LS(w) $$

**Experience Replay in Deep Q-Network(DQN)**
DQN uses **experience replay** and **fixed Q-targets**
* Take action $a\_t$ according to $\epsilon$-greedy policy 
* Store transition $(s\_t,a\_t,r\_{t+1},s\_{t+1})$ in replay memory $D$
* Sample random mini-batch of transitions $(s,a,r,s')$ from $D$
* Compute Q-learning targets w.r.t old, fixed parameters $w^-$
* Optimise MSE between Q-network and Q-learning targets
$$ L\_i (w\_i) = E\_{s,a,r,s' \sim D\_i} \left[ \left( r + \gamma \max\_{a'} Q(s',a';w\_i^-) - Q(s,a;w\_i) \right)^2 \right] $$
* Using variant of stochastic gradient descent

**Linear Least Squares Prediction**
* Experience replay finds least squares solution
* But it may take many iterations
* Using linear value function approximation $\hat{v} (s,w) = x(s)^T w $
* We can solve the least squares solution directly
    * At minimum of $LS(w)$, the expected update must be zero, $E\_D (\Delta w) = 0$
    $$ \alpha \sum\_{t=1}^{T} x(s\_t) (v\_t^\pi -x(s\_t)^T w) = 0 \\\\  w =\left( \sum\_{t=1}^T x(s\_t)x(s\_t)^T \right)^{-1} \sum\_{t=1}^T x(s\_t) v\_t^\pi $$
    * For $N$ features, direct solution time is $O(n^3)$
    * Incremental solution time is $O(n^2)$ using Shermann-Morrison

**Linear Least Squares Prediction Algorithms**
* We don't know true values $v\_t^\pi$
* In practice, our "training data" must use noisy or biased sample of $v\_t^\pi$
    * **LSMC** Least Squares MC uses return $v\_t^\pi \approx G\_t$
    * **LSTD** Least Squares TD uses TD target $v\_t^\pi \approx R\_{t+1} + \gamma \hat{v}(S\_{t_1},w) $
    * **LSTD($\lambda$)** Least Squares TD($\lambda$) use $\lambda$-return $v\_t^\pi \approx G\_t^\lambda$
* In each case solve directly for fixed point of MC/TD/TD($\lambda$)
 ![Direct solution for LS](http://i4.buimg.com/567571/887027f627e53e74.png)

**Convergence of Linear Least Squares Prediction Algorithms**
![Convergence of LS prediction algorithms](http://i1.piimg.com/567571/389f21e4a037e7b3.png)

### Least Squares Control
**Least Squares Policy Iteration**
_Policy evaluation_ Policy evaluation by least squares Q-learning
_Policy improvement_ Greedy policy improvement

**Least Squares Action-Value Function Approximation**
* Approximate action-value function $q\_\pi (s,a)$
* using linear combination of features $x(s,a)$
$$ \hat{q} (s,a,w) = x(s,a)^T w \approx q\_\pi (s,a) $$
* Minimise least squares error between $\hat{q} (s,a,w)$ and $q\_\pi (s,a)$
* form experience generated using policy $\pi$
* consisting of $<(state,action),value\>$ pairs
$$ D = \\{ < (s\_1,a\_1),v\_1^\pi \>, <(s\_2,a\_2),v\_2^\pi\>,\dots,<(s\_T,a\_T),v\_T^\pi\> \\} $$

**Least Squares Control**
* For policy evaluation, we want to efficiently use all experience
* For control, we also want to improve the policy
* This experience is generated from many policies
* So to evaluate $q\_pi (S,A)$ we must learn off-policy
* We use the same idea as Q-learning:
    * Use experience generated by old policy, $ S\_t,A\_t,R\_{t+1},S\_{t+1} \sim \pi\_{old} $
    * Consider alternative successor action $ A' = \pi\_{new} (S\_{t+1}) $
    * Update $\hat{q} (S\_t,A\_t,w) $ towards value of alternative action $R\_{t+1} + \gamma \hat{q} (S\_{t+1},A',w)$

**Least Squares Q-Learning**
* Consider the following linear Q-learning update
$$ \delta = R\_{t+1} + \gamma \hat{q} (S\_{t+1},\pi(S\_{t+1}),w) - \hat{q} (S\_t,A\_t,w) \\\\ \Delta w =\alpha \delta x(S\_t,A\_t) $$
* LSTDQ alorithm: solve for total update = zero
$$ 0 = \sum\_{t=1}^T \alpha (R\_{t+1} +\gamma \hat{q} (S\_{t+1},\pi(S\_{t+1}),w) -\hat{q} (S\_t,A\_t,w)) x(S\_t,A\_t) \\\\ w = \left( \sum\_{t=1}^T x(S\_t,A\_t) (x(S\_t,A\_t)-\gamma x(S\_{t+1},\pi(S\_{t+1})))^T \right)^{-1} \sum\_{t=1}^T x(S\_t,A\_t) R\_{t+1} $$

**Least Squares Policy Iteration Algorithm**
* The following pseudocode uses LSTDQ for policy evaluation
* It repeatedly re-evaluates experience $D$ with difference policy 
![LSPI algorithm](http://i4.buimg.com/567571/12115cfd47cc51c7.png)

**Convergence of Control Algorithms**
![Convergence of Control Algorithms](http://i4.buimg.com/567571/5103a9f8f57b6a60.png)
$(\checkmark)$ = chatters around near-optimal value function
