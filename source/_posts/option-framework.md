---
title: option-framework
mathjax: true
date: 2017-08-31 11:02:07
categories: reinforcement learning
tags: [reinforcement learning, option]
---
# Option framework in reinforcement learning
参考自[Learning Options in Reinforcement Learning](http://www.cs.cmu.edu/~mstoll/pubs/stolle2002learning.pdf)
option 是在强化学习中的特殊的结构，它被设计在一些顶层任务中加入一些小的子目标的任务。同时一个option可以看成是一个time-extended的action.
## 强化学习
这里只定义一些符号，具体有关强化学习的细节略去
强化学习基于MDP，其中包括这样一些部分，状态集合S, 动作集合A, reward集合R. 给定一个状态$s\in S$和一个动作$a \in A$, 我们会得到一个immediate reward $r\_s^a$, 以及一个从状态$s$到状态$s'$的转移概率$P\_{ss'}^a$. policy被定义为在每一个状态下执行各个动作的概率$\pi:S\times A \rightarrow [0, 1]$
## Option framework
一个option (每一个agent可能包含多个option)包括三个部分：一，初始状态集$I \subset S$, 初始状态集包含了所有能够启动option的状态，也就是说当agent进入这样的状态时，agent的就可以被一个option控制。二，停止条件$\beta \in [0,1]$, 表示agent停止被option控制的概率。 三，内部policy，$\pi': S\times A \rightarrow [0, 1]$, 这是某个option中特定的policy。一个option可以用这样一个三元组来表示$o=<I, \pi', \beta\>$
> option的工作流程是，当agent运行到初始状态集$I$所包含的的状态时，agent会被option所控制，开始调用option内部的policy,$\pi'$，执行相应的动作，跳转到下一个状态，此时判断停止条件$\beta$, 若停止，则退出option，否则继续执行$\pi'$并重复上面的步骤。

原始的强化学习也能被视为option的特例，我们把每一个动作都看成一个option，所有执行该动作的状态看成该option的初始状态，$I\in \{ s: a\in A\_s\}$, 然后内部的policy就是以1的概率执行动作$a$, 停止条件为$\beta=1$, 意味着这个option每次只执行一步。

然后我们可以把原始的强化学习中的action-value (Q) function拓展为option-value function，我们定义$Q^\mu (s, o)$, 表示在状态$s\in I$时采用策略$\mu$执行option o所获得的期望收益为
$$ Q^\mu (s, o)  = E \left[ r\_{t+1} + \gamma r\_{t+2} + \gamma^2 r\_{t+3} + \cdots  \mid E(o\mu, s, t)\right] $$
这里$o\mu$表示执行option o, 直到它停止，然后使用策略$\mu$, $E(o \mu, s, t)$表示在时间$t$和状态$s$处执行$o\mu$。

所以我们可以把option当成一种特殊的动作，这种动作具有可以用来表示一连串的基本动作，而且有着其对应的子目标。它在形式上类似于semi-MDP中的动作，所以我们能够运用semi-MDP的方法来解决带option的agent的寻优问题。当在状态$s$中执行option，直到停止时，agent切换到了$s'$, 中间经历了$k$个time step， 同时中间的累计收益用$r$表示，那么更新算法为
$$ Q(s, o) \leftarrow Q(s, o) + \alpha \left[r + \gamma^k \max_{a \in O} Q(s',a) - Q(s,o)\right] $$

## Find optimal option
对于一个option来说, 如果其初始状态和终止条件均已知了，那么其内部策略可以用标准的强化学习算法去优化，因此option优化的关键是求解$I$和$\beta$.

在文章[learning Options in Reinforcement Learning](http://www.cs.cmu.edu/~mstoll/pubs/stolle2002learning.pdf)中，认为如果某一个状态频繁地在trajectories中出现，那么代表这个状态很重要，那么就将这些状态作为option的target。在这篇文章中的算法如下：
```c
1. 随机选择一定数量的开始状态S和目标状态T, 这些状态可以用来产生针对agent的随机任务
2. 对于每一对<S, T>, 执行
	(a). 使用Q学习学习一定数量的episodes，学出一个从S到T的policy
	(b). 执行学好的policy数个episodes， 统计每个状态s出现的次数n(s)
3. 重复以下过程，直到期望的option的数量得以满足
	(a). 选择出现最多的状态，作为option的target state，T'
	(b). 统计经过T'的路径上的所有状态的出现的数量
	(c). 统计上一步中所有状态出现数量的平均值
	(d). 选择大于平均值的状态作为初始状态I的一部分
	(e). 使用一些插值的方法对初始状态进行插值，获得完整的初始状态
4. 对于每一个option，学习其内部policy，直到到达对应的T'
```

## option结构的自学习
参考文章[The Option-Critic Architecture](http://rll.berkeley.edu/deeprlworkshop/papers/BaconPrecup2015.pdf)
在这篇文章中，提出了这样的观点：**过去的find optimal option的方法都是寻找子目标然后学习如何相应的inter-policy，这样的方法很难将内部option的策略和统一的策略统一起来，同时子目标的解决有时也很低效，甚至和原问题一样难**，因此这篇文章提出了一个能够同时学习policy, inter-policy, terminations的方法。并做了理论推导。这样的方法有以下的好处：
* 速度快，不用单独学习大量的子任务
* end-to-end，option的设计不需要人为的设计
* 通用性好，更加适合transfer learning

### 预定义的符号
MDP的状态集合$S$，动作集$A$，转移概率$P: S\times A\rightarrow [0, 1]$，reward function $r: S \times A \rightarrow [0, 1]$. 策略$\pi: S\times A \rightarrow [0,1]$，以及expected return, $V\_{\pi}(s) = E\_{\pi} [\sum\_{t=0}^{\infty} \gamma^t r\_{t+1}|s\_0 = s]$，和action-value function $Q\_{\pi}(s,a)= E\_{\pi}[\sum\_{t=0}^{\infty}\gamma^t r\_{t+1} | s\_0 =s ,a\_0=a]$
### Policy gradient methods
将策略进行参数化，用$\pi\_\theta$来表示，那么从$s\_0$处的期望收益表示为$\rho(\theta, s\_0)=E\_{\pi} [\sum\_{t=0}^{\infty} \gamma^t r\_{t+1}|s\_0 = s]$，根据policy gradient therem求出梯度
	$$\frac{\partial \rho(\theta, s\_0)}{\partial\theta} = \sum\_s \mu\_{\pi\_\theta}(s | s\_0) \sum\_a \frac{\partial \pi\_\theta (a|s)}{\partial \theta}Q\_{\pi\_\theta}(s,a)$$
在这里$\mu\_{\pi\_\theta} (s|s\_0) = \sum\_{t=0}^{\infty} \gamma^t P(s\_t = s|s\_0)$表示在以$s\_0$为初始状态下的trajectories上的所有状态的discouted weighting
### option framework
这里我们用$\omega \in \Omega$表示一个option，可以表达成三元组$(I\_\omega, \pi\_\omega, \beta\_\omega)$

### 学习option
这篇文章学习option的思想如下
> 在任何阶段，网络都会充分利用experience，并同时更新value function, policy over options, inter-policy,和termination functions. 网络采用一种叫做call-and-return的option执行模型，agent通过policy over option $\pi\_\Omega$来选择option $\omega$,然后执行intra-policy $\pi\_\omega$直到termination,$\beta\_\omega$.

我们用$\pi\_{\omega, \theta}$, 表示intra-policy，并用$\theta$进行参数化，同时用$\beta\_{\omega, \\vartheta}$表示停止条件，并用$\vartheta$进行参数化。

< 
此时网络的收益函数变成了
$$ \rho(\Omega, \theta, \vartheta, s\_0, \omega\_0) = E\_{\Omega, \theta, \omega} \left[ \sum\_{t=0}^{\infty} \gamma^t r\_{t+1} | s\_0, \omega\_0\right] $$

这里我们需要同时求得$\rho$相对$\theta, \vartheta$的梯度。

我们首先定义函数$U:\Omega \times S \rightarrow R$, 把这个函数叫做option-value function opon arrival。 这个函数表示在状态$s'$下执行option$\omega$所获获得的收益
$$ U(\omega, s') =(1-\beta\_{\omega, \vartheta} (s')) Q\_{\Omega} (s', \omega) + \beta\_{\omega, \vartheta} (s') V\_{\Omega}(s')$$
这个函数由两个部分组成，分为两种情况，第一项表示如果没有跳出这个$option$的情况下，第二项表示如果已经停止执行option $\omega$之后。

我们可以把$(s,\omega)$的视为一个拓展的状态，那么在$(s, \omega)$处执行动作$a$的收益为
$$ Q\_U (s, \omega, a) = r(s, a) + \gamma \sum\_{s'} P(s' |s,a) U(\omega, s')$$
这里$P$表示状态转移概率，那么考虑所有的动作，option-value function如下
$$ Q\_{\Omega} (s,\omega) = \sum\_a \pi\_{\omega, \theta} (a|s) Q\_U (s, \omega, a) $$

如果将$(s,\omega)$看成一个拓展的状态, 那么从$(s\_t, \omega\_t)$跳转到$(s\_{t+1}, \omega\_{t+1}$的概率为
$$P(s\_{t+1}, \omega\_{t+1} | s\_t, \omega\_t) = \sum\_a \pi\_{\omega\_t, \theta} (a | s\_t) P(s\_{t+1}| s\_t, a) \left( (1-\beta\_{\omega\_t, \vartheta}) \mathbf{1}\_{\omega\_t = \omega\_{t+1}} + \beta\_{\omega\_t, \vartheta} (s\_{t+1})\pi\_{\Omega} (\omega\_{t+1} | s\_{t+1})\right) $$

这个公式运用了概率公式的链式法则，其中包括三项，第一项是在$s\_t$下执行动作$a$的概率，第二项是在$s\_t$下执行$a$后能跳转到$s\_{t+1}$的概率，第三项比较复杂，表示在$s\_{t+1}$后选择option $\omega\_{t+1}$的概率。由于转移概率与动作无关，所以对动作求和。第三项包括两个部分，第一个部分表示在跳转的过程中，并没有停止option，但是这种情况必须满足$\omega\_t = \omega\_{t+1}$，另外一种情况是，中间发生了option的切换，所以这一项的概率等于停止的概率乘以重新选择到option $\omega\_{t+1}$的概率。

根据求导的法则，我们可以得到
$$ \frac{\partial Q\_{\Omega} (s,\omega)}{\partial \theta} = \left( \sum\_{a} \frac{\partial \pi\_{\omega, \theta} (a|s)}{\partial \theta} Q\_U(s,\omega, a)\right)+\sum\_a \pi\_{\omega,\theta} (a|s) \sum\_{s'} \gamma P(s'|s,a)\frac{\partial U(\omega, s')}{\partial \theta}$$

根据论文中的两个定理，具体参考论文，可以求解出相应的$\rho$相对$\theta, \vartheta$的梯度。
_根据定理一，求出梯度如下_
$$ \frac{\partial Q\_{\Omega} (s,\omega)}{\partial \theta} = \sum\_{s,\omega} \mu\_{\Omega} (s,\omega|s\_0, \omega\_0) \sum\_a \frac{\partial \pi\_{\omega,\theta} (a|s)}{\partial \theta} Q\_U (s,\omega, a)$$
这里$\mu\_{\Omega} (s,\omega|s\_0, \omega\_0)$表示从$(s\_0, \omega\_0)$开始的trajectories上的state-option pairs上的加权权重。$\mu\_{\Omega}(s,\omega|s\_0, \omega\_0) = \sum\_{t=0}^{\infty} \gamma^t P(s\_t=s,\omega\_t =\omega|s\_0, \omega\_0)$

_根据定理二，求出梯度如下_
$$ \frac{\partial Q\_{\Omega} (s,\omega)}{\partial \vartheta} = -\sum\_{s',\omega} \mu\_{\Omega} (s',\omega|s\_1, \omega\_0) \frac{\partial \beta\_{\omega,\vartheta} (s')}{\partial \vartheta} A\_\Omega (s',\omega)$$
这里$\mu\_{\Omega} (s',\omega|s\_1, \omega\_0)$表示从$(s\_1, \omega\_0)$开始的trajectories上的state-option的加权权重。$\mu\_{\Omega}(s,\omega|s\_1, \omega\_0) = \sum\_{t=0}^{\infty} \gamma^t P(s\_{t+1}=s,\omega\_t =\omega|s\_1, \omega\_0)$

改论文使用的网络结构的如下图
![option-network](http://odchpimz0.bkt.clouddn.com/20170903150444684214022.png)
使用的更新算法如下
![algorithm](http://odchpimz0.bkt.clouddn.com/20170903150444715321956.png)
