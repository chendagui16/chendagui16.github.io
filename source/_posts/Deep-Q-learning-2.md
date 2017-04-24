---
title: Deep-Q-learning-2
mathjax: true
date: 2016-09-09 09:13:52
categories: code analyze
tags: [reinforcement learning, Q-learning, deepmind]
---
## Deep Q Learning Code Analyze (2)

分析的源码来自于deepmind在Natrue上发表的论文[Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/abs/nature14236.html)所附的源码。[源码下载](sites.google.com/a/deepmind.com/dqn)

### 文件结构   [续上](https://chendagui16.github.io/2016/09/06/Deep-Q-learning)
---
#### NeuralQLearner.lua
该文件定义了一个dqn.NerualQLearner的类，该类主要制定了深度Q学习的学习规则。同样地，这里对该类的成员函数一一进行解读。
```lua
local nql = torch.class('dqn.NeuralQLearner')
```

##### nql:__init(args)
类对象的初始化。由于初始化的对象很多，这里就不一一介绍，主要介绍几个难以理解的成员变量，其他成员变量可参考源文件。
```lua
    self.verbose    = args.verbose  --verbose，数字越大，训练时所输出的信息越多
    self.best       = args.best  --布尔型变量，if true，那么载入模型时载入best_model
    
    self.discount       = args.discount or 0.99 --在求future discount reward时的衰减因子.
    self.update_freq    = args.update_freq or 1 --更新频率，每执行多少次action才进行一次学习，即每两次更新中所执行的action次数
    self.n_replay       = args.n_replay or 1 --每次更新时重复学习的次数
    self.learn_start    = args.learn_start or 0  --学习开始时的步数

    self.rescale_r      = args.rescale_r  --布尔型变量，if true，则将reward除以self.r_max。
    self.clip_delta     = args.clip_delta --如果定义之后，会将输出层的残差限定在[-self.clip_delta,self.clip_delta]之间
    self.target_q       = args.target_q   --整数，每隔target_q步将update网络的参数copy到target网络
    self.bestq          = 0  --记录网络的最大的q值
    
    self.ncols          = args.ncols or 1  -- 颜色通道数量
    self.preproc        = args.preproc  -- 预处理网络名
    
    self.network    = args.network or self:createNetwork() --如果参数给了网络，就载入网络，否则就调用self:createNetwork()函数新建一个网络
   
   --参考TransitionTable.lua文件，初始化transiitons类 
    local transition_args = { 
        stateDim = self.state_dim, numActions = self.n_actions,
        histLen = self.hist_len, gpu = self.gpu,
        maxSize = self.replay_memory, histType = self.histType,
        histSpacing = self.histSpacing, nonTermProb = self.nonTermProb,
        bufferSize = self.bufferSize
    }
    self.transitions = dqn.TransitionTable(transition_args)
    
    self.numSteps = 0 -- 执行的步数.
    self.v_avg = 0 -- Validation上的平均q值.
    self.tderr_avg = 0 -- target和destination之间的平均误差.
    
    self.w, self.dw = self.network:getParameters() --得到网络的参数w和dw
    self.dw:zero()
    self.deltas = self.dw:clone():fill(0)
    --设立中间变量，用来求梯度
    self.tmp= self.dw:clone():fill(0)
    self.g  = self.dw:clone():fill(0)
    self.g2 = self.dw:clone():fill(0)  
```
另外，在初始化的函数中，还调用了lua语言内置的pcall函数来载入网络和预处理网络。例如如下代码：
```lua
    local msg, err = pcall(require, self.network)
    if not msg then
        local err_msg, exp = pcall(torch.load, self.network)
        if not err_msg then
            error("Could not find network file ")
        end
        if self.best and exp.best_model then
            self.network = exp.best_model
        else
            self.network = exp.model
        end
    else
        print('Creating Agent Network from ' .. self.network)
        self.network = err
        self.network = self:network()
    end
```
> pcall函数是lua的内置处理函数,一般的使用方法是msg,err=pcall(func,param)，通过调用func(param)函数，如果调用成功，则msg返回true，err返回func(param)的返回值，如果出现错误和异常，则msg返回nil，err返回错误的信息。该函数在lua相当于try，out的作用。

通过使用pcall调用载入函数，可以事先对self.network和self.preproc进行初始化。

##### nql:reset(state)
重置类对象，主要是载入state.best_network和state.model。然后将self.dw归零，将执行步数self.numSteps置零。

##### nql:preprocess(rawstate)
将原始状态进行预处理
```lua
function nql:preprocess(rawstate)
    if self.preproc then
        return self.preproc:forward(rawstate:float())
                    :clone():reshape(self.state_dim)
    end
    return rawstate
end
```

##### nql:getQUpdate(args)
该函数主要以一个transition: < s,a,r,s',term>作为输入，然后通过计算获得Q值，以及targets，残差等。
首先该函数会载入args的参数，包括args.s,args.a,args.r,args.s2,args.term。然后按照下面的程序计算：
```lua
-- delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)
term = term:clone():float():mul(-1):add(1)  -- (1-term)

q2_max = target_q_net:forward(s2):float():max(2) --max_a Q(s_2,a)
-- 计算 q2 = (1-terminal) * gamma * max_a Q(s2, a)
q2 = q2_max:clone():mul(self.discount):cmul(term)
delta = r:clone():float()

if self.rescale_r then  --如果self.rescale_r定义了之后，就将reward除以self.r_max
   delta:div(self.r_max)
end
delta:add(q2)  --r + (1-terminal) * gamma * max_a Q(s2, a)
local q_all = self.network:forward(s):float()  --q_all矩阵用来存储q值，q_all[i][j]表示batchsize中的第i个输入的第j个action对应的Q值
q = torch.FloatTensor(q_all:size(1))
for i=1,q_all:size(1) do  --q向量，这里q=Q(s,a)
    q[i] = q_all[i][a[i]]
end
delta:add(-1, q)  --delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)
```
这里得到了delta=*r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)*,注意到，如果定义了self.clip_delta，那么将残差进行**限幅**操作，将幅度不在[-self.clip\_delta,self.clip\_delta]的delta值强行clip。
同时，函数定义了targets矩阵，其中target是一个二维矩阵，第一维表示batch_size，第二维表示actions。这里，我们将delta的值赋给target对应的action位置，其他action处，target=0。
```lua
    local targets = torch.zeros(self.minibatch_size, self.n_actions):float()
    for i=1,math.min(self.minibatch_size,a:size(1)) do
        targets[i][a[i]] = delta[i]
    end
```
最后函数返回targets,delta以及q2_max的值。

##### nql:qLearnMinibatch()
这个函数的主要目的是执行一个minibatch的Q-learning的update，其中采用的更新权重的方法是PMSProp，这里**w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw**
```lua
    local s, a, r, s2, term = self.transitions:sample(self.minibatch_size) --利用transition类的sample函数，得到新的transition
    local targets, delta, q2_max = self:getQUpdate{s=s, a=a, r=r, s2=s2,term=term, update_qmax=true} --更新Q
    
    self.dw:zero()
    self.network:backward(s, targets) --反向传播，更新dw
    self.dw:add(-self.wc, self.w) --加入一阶正则化约束

    -- 更新学习率
    local t = math.max(0, self.numSteps - self.learn_start)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt +self.lr_end
    self.lr = math.max(self.lr, self.lr_end)

    -- 利用PMSProp求得梯度，这里加入了一阶的梯度momentum和二阶的梯度momentum
    self.g:mul(0.95):add(0.05, self.dw)
    self.tmp:cmul(self.dw, self.dw)
    self.g2:mul(0.95):add(0.05, self.tmp)
    self.tmp:cmul(self.g, self.g)
    self.tmp:mul(-1)
    self.tmp:add(self.g2)
    self.tmp:add(0.01)
    self.tmp:sqrt()
    
    self.deltas:mul(0):addcdiv(self.lr, self.dw, self.tmp) --lr*tmp/dw
    self.w:add(self.deltas)
```
> a=addcdiv(b,c,d)表示a=a+b*d/c

##### nql:sample_validation_data()
利用transition类的sample函数，采样self.valid_size个样本，并将数据存储到self.valid\_(s,a,r,s2,term)中。

##### nql:compute_validation_statistics()
计算得到validation上的平均Q_max值，和平均误差（误差指target和destination之间的差）
```lua
function nql:compute_validation_statistics()
    local targets, delta, q2_max = self:getQUpdate{s=self.valid_s,
        a=self.valid_a, r=self.valid_r, s2=self.valid_s2, term=self.valid_term}

    self.v_avg = self.q_max * q2_max:mean()
    self.tderr_avg = delta:clone():abs():mean()
end
```
##### nql:eGreedy()
该函数主要的目的按照greed expolation的方式去选择一个action
```lua
function nql:eGreedy(state, testing_ep)
    self.ep = testing_ep or (self.ep_end + math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt - math.max(0, self.numSteps - self.learn_start))/self.ep_endt)) --更新self.ep的值
    if torch.uniform() < self.ep then
        return torch.random(1, self.n_actions) --以ep的概率值随机选择action的值
    else
        return self:greedy(state) --以1-ep的概率值选择最大Q值的action，具体实现参考nql:greedy(state)函数
    end
end
```

##### nql:greedy(state)
这个函数的目的就是用来根据最大Q值选择一个action的值，注意到，如果有几个action的Q值均为最大，那么随机选择一个action执行。
```lua
    local q = self.network:forward(state):float():squeeze()
    local maxq = q[1]  --max q
    local besta = {1}  --best action
    for a = 2, self.n_actions do
        if q[a] > maxq then
            besta = { a }
            maxq = q[a]
        elseif q[a] == maxq then
            besta[#besta+1] = a
        end
    end
    self.bestq = maxq
    local r = torch.random(1, #besta)
    self.lastAction = besta[r] --存储到self.lastAction
    return besta[r]
```

##### nql:perceive(reward,rawstate,terminal,testing,testing_ep)
这个函数会与transition类之间进行交互，然后更新Q值，选择action，并进行参数的优化。
首先，将rawstate进行预处理，并定义当前状态
```lua
    local state = self:preprocess(rawstate):float()
    local curState
```
然后根据self.max_reward,self.min_reward和self.rescale_r将reward进行限幅。
```lua
    if self.max_reward then
        reward = math.min(reward, self.max_reward)
    end
    if self.min_reward then
        reward = math.max(reward, self.min_reward)
    end
    if self.rescale_r then
        self.r_max = math.max(self.r_max, reward)
    end
```
调用transition类，将state（这里的state只包含一帧图像）加入recent存储区，然后从transition中采样得到新的state，此时的state是由多帧构成的。接下来将新的transition<s,a,r,t\>存储到存储区内。
```lua
    self.transitions:add_recent_state(state, terminal)
    local currentFullState = self.transitions:get_recent()
    if self.lastState and not testing then  --testing标志位表示是否进行测试模式
        self.transitions:add(self.lastState, self.lastAction, reward,
                             self.lastTerminal, priority)
    end
```
```lua
    if self.numSteps == self.learn_start+1 and not testing then --如果训练才刚刚开始，那么先采样验证集的数据
        self:sample_validation_data()
    end
    curState= self.transitions:get_recent() --得到当前专题太
    curState = curState:resize(1, unpack(self.input_dims))
```
利用eGreedy算法得到新的action
```lua
    local actionIndex = 1
    if not terminal then
        actionIndex = self:eGreedy(curState, testing_ep)
    end
    self.transitions:add_recent_action(actionIndex)
```
进行Q-learning更行权重，这里更新每隔self.update_freq步才进行一次权重的的更新，也就是说每两次更新之间执行self.update_freq次action，然后每次更新会重复连续学习self.n_replay次
```lua
    if self.numSteps > self.learn_start and not testing and
        self.numSteps % self.update_freq == 0 then
        for i = 1, self.n_replay do
            self:qLearnMinibatch()
        end
    end
```
更新学习步数
```lua
    if not testing then
        self.numSteps = self.numSteps + 1
    end
```
学习完之后，此时的状态和action都发生的改变，我们需要将last的状态和action进行一个更新。
```lua
    self.lastState = state:clone()
    self.lastAction = actionIndex
    self.lastTerminal = terminal
```
每隔self.target_q个步骤，将参数copy到target网络。
```lua
    if self.target_q and self.numSteps % self.target_q == 1 then
        self.target_network = self.network:clone()
    end
```
最后返回要执行的actionIndex值
```lua
    if not terminal then
        return actionIndex
    else
        return 0
    end
```

##### nql:createNetwork()
创建一个三个线性层的网络，这是一个三层的多层感知器
```lua
function nql:createNetwork()
    local n_hid = 128
    local mlp = nn.Sequential()
    mlp:add(nn.Reshape(self.hist_len*self.ncols*self.state_dim))
    mlp:add(nn.Linear(self.hist_len*self.ncols*self.state_dim, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, self.n_actions))
    return mlp
end
```
##### nql:_loadNet()
载入网络，返回self.network

##### nql:init(arg)
手动初始化

##### nql:report()
调用nnutil.lua中的get\_weight\_norms,get\_grad\_norms函数，输出network的信息。

#### train_agent.lua
这是训练的主程序，这里对其进行解析。
##### 初始化
调用setup.lua进行初始化，得到game\_env,game\_actions,agent,opt。
```lua
local opt = cmd:parse(arg)
local game_env, game_actions, agent, opt = setup(opt)
```
然后初始化参数列表
```lua
local learn_start = agent.learn_start  --学习开始时的步数
local start_time = sys.clock() --开始的时间
local reward_counts = {}  --记录每次测试时的不等于0的reward数量
local episode_counts = {} --记录每次测试时的episode数量
local time_history = {} --记录每次测试的时间
local v_history = {} --记录每次测试时的平均Q值
local qmax_history = {} --记录每次测试时的qmax
local td_history = {} --记录每次测试时的误差值
local reward_history = {} --记录每次测试时的总的reward
local step = 0  --训练时的步数
time_history[1] = 0
--测试时使用的参数
local total_reward --总的reward
local nrewards --不为0的reward数量
local nepisodes --总的episodes
local episode_reward --中间变量，用来存储一个episode中的reward总量

local screen, reward, terminal = game_env:getState() --从environment中获得rawstate(screen)，reward，terminal。
```
##### 训练
调用nql:perceive()函数进行训练，得到执行的action_index。
```lua
    step = step + 1  --步数更新
    local action_index = agent:perceive(reward, screen, terminal)
```
如果游戏已经结束，那么重新进入下一个游戏
```lua
    if not terminal then
        screen, reward, terminal = game_env:step(game_actions[action_index], true)  --游戏没有结束，那么玩游戏，并得到下一个screen,reward,terminal
    else
        if opt.random_starts > 0 then --opt.random_start表示是否重新开始
            screen, reward, terminal = game_env:nextRandomGame()
        else
            screen, reward, terminal = game_env:newGame()
        end
    end
```
每隔opt.prog_freq步就输出网络的信息
```lua
    if step % opt.prog_freq == 0 then
        assert(step==agent.numSteps, 'trainer step: ' .. step ..' & agent.numSteps: ' .. agent.numSteps)
        print("Steps: ", step)
        agent:report()  --调用nql:report()函数
        collectgarbage()
    end
```
> 在lua语言中，不会自动处理垃圾，需要调用collectgarbage()手动处理。

##### 训练（在特定的步数上进行验证）
每隔opt.eval_freq步就进行验证。首先进行初始化。
```lua
        screen, reward, terminal = game_env:newGame()
        total_reward = 0 --总的reward
        nrewards = 0  --不为0的reward的数量
        nepisodes = 0 --一共执行的episodes的数量
        episode_reward = 0 --中间变量，保存每个episode的reward的总量
```
然后调用nql:perceive()进行验证，注意到这里的testing参数为true，ep固定为0.05。
```lua
        local eval_time = sys.clock() --记录测试开始时的时间
        for estep=1,opt.eval_steps do  --opt.eval_step表示测试时的步数
            local action_index = agent:perceive(reward, screen, terminal, true, 0.05) --这里testing=true，testing_ep=0.05

            -- 在测试模式下进行游戏
            screen, reward, terminal = game_env:step(game_actions[action_index])
            if estep%1000 == 0 then collectgarbage() end
            episode_reward = episode_reward + reward
            if reward ~= 0 then
               nrewards = nrewards + 1
            end
            if terminal then
                total_reward = total_reward + episode_reward
                episode_reward = 0
                nepisodes = nepisodes + 1
                screen, reward, terminal = game_env:nextRandomGame()
            end
        end
```
计算时间
```lua
        eval_time = sys.clock() - eval_time --计算得到刚才测试所消耗的时间
        start_time = start_time + eval_time --更新时间
```
获得统计数据，注意到由于每次测试都有可能执行了不同的eposide，我们这里计算每个eposide的平均值。
```lua
        agent:compute_validation_statistics() --调用nql:compute_validation_statistics函数来计算得到平均Q值self.v_avg，和平均误差self.tferr_avg
        local ind = #reward_history+1 --迭代器，指向reward_history的下标
        total_reward = total_reward/math.max(1, nepisodes) --得到每个eposide的平均reward
        if #reward_history == 0 or total_reward > torch.Tensor(reward_history):max() then --如果reward_history中没有数据，或者测试时产生的新的total_reward比之前产生的都要好，那么更新agent.best_network
            agent.best_network = agent.network:clone()
        end

        if agent.v_avg then --记录统计数据
            v_history[ind] = agent.v_avg
            td_history[ind] = agent.tderr_avg
            qmax_history[ind] = agent.q_max
        end
        print("V", v_history[ind], "TD error", td_history[ind], "Qmax", qmax_history[ind])
    --记录total_reward，nrewards，nepisodes，以及运行时间
        reward_history[ind] = total_reward  
        reward_counts[ind] = nrewards
        episode_counts[ind] = nepisodes
        time_history[ind+1] = sys.clock() - start_time --记录本次测试结束的时间
        local time_dif = time_history[ind+1] - time_history[ind] --计算两次测试之间的时间差
        local training_rate = opt.actrep*opt.eval_freq/time_dif  --计算训练速率，指单位时间内训练的次数，opt.actrep表示重复执行actions的次数，opt.eval_freq表示验证的频率
```
输出信息
```lua
print(string.format('\nSteps: %d (frames: %d), reward: %.2f, epsilon: %.2f, lr: %G, ' .. 'training time: %ds, training rate: %dfps, testing time: %ds, ' .. 'testing rate: %dfps,  num. ep.: %d,  num. rewards: %d', step, step*opt.actrep, total_reward, agent.ep, agent.lr, time_dif, training_rate, eval_time, opt.actrep*opt.eval_steps/eval_time, nepisodes, nrewards))
```
##### 训练（在特定的步数上进行保存）
每隔opt.save_freq步或者训练完之后，将网络进行保存。对于保存的程序，这里就不进行分析了。

### [上一页](https://chendagui16.github.io/2016/09/06/Deep-Q-learning)