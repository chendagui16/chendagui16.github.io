---
title: Deep_Q_learning
mathjax: true
date: 2016-09-06 09:07:49
categories: code analyze
tags: [reinforcement learning, Q-learning, deepmind]
---
## Deep Q Learning Code Analyze (1)

分析的源码来自于deepmind在Natrue上发表的论文[Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/abs/nature14236.html)所附的源码。[源码下载](sites.google.com/a/deepmind.com/dqn)

### 文件结构 
---
代码采用torch框架进行组织，编写的语言均为lua语言，其中包括convnet.lua, convnet_atari3.lua, initenv.lua, net\_downsample\_2x\_full\_y.lua, NeuralQLearner.lua, nnutils.lua, Rectifier.lua, Scale.lua, train\_agent.lua, TransitionTable.lua。

训练的主程序是从train\_agent.lua(具体的train\_agent.lua的解析见[这里](https://chendagui16.github.io/2016/09/09/Deep-Q-learning-2/))开始。训练时的参数表如下：
```lua
cmd:option('-framework', '', 'name of training framework')
cmd:option('-env', '', 'name of environment to use')
cmd:option('-game_path', '', 'path to environment file (ROM)')
cmd:option('-env_params', '', 'string of environment parameters')
cmd:option('-pool_frms', '',
           'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', false,
           'saves the agent network in a separate file')
cmd:option('-prog_freq', 5*10^3, 'frequency of progress output')
cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 10^4, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')

cmd:option('-steps', 10^5, 'number of training steps to perform')
cmd:option('-eval_steps', 10^5, 'number of evaluation steps')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')
```
训练的开始会调用initenv.lua初始化game_env, game_actions, agent, opt。


#### initenv.lua

initenv文件是在训练的初始阶段，用来初始化gamEnv，gameActions，agent，以及opt参数。提供了torchSetup函数和Setup函数，在这里torchSetup函数用来初始化一些与torch相关的参数，包括gpu参数，计算线程，以及tensorType等。

而Setup参数用来调用torchSetup函数，并对gameEnv，gameActions，agent进行了初始化操作。

gameEnv表示游戏的环境，通过调用getState()方法可以得到screen, reward和terminal参数。screen表示屏幕状态，这是DQN中的输入，terminal是布尔型变量，表示是否游戏结束。
```lua
local screen, reward, terminal = game_env:getState()
```
#### nnutils.lua

nnutils文件主要提供了一些辅助函数。
该文件首先提供了recursive\_map的函数，该函数接受module, field, func作为输入，返回一个字符串，其中module表示训练的模型，field指模型中的某类参数名，比如field='weight’时，module[field]表示模型中的权重。该函数会返回字符串，包含了模型的类型名，对module[field]的统计数据（统计的方法视func而定）。
```lua
function recursive_map(module, field, func)
```

由于模型中包含了子模型，因此recusive\_map函数会递归调用子模型，因此会形成模型的树状表示。
```lua
    if module.modules then
        str = str .. "["
        for i, submodule in ipairs(module.modules) do
            local submodule_str = recursive_map(submodule, field, func)
            str = str .. submodule_str
            if i < #module.modules and string.len(submodule_str) > 0 then
                str = str .. " "
            end
        end
        str = str .. "]"
    end
```

在nnuils的文件中，定义了abs\_mean()和abs\_max()的函数，表示平均值和最大值。另外也定义了get\_weight\_norms()和get\_grad\_norms()的函数，这两个函数会调用recursive\_map函数，分别对权重和梯度值求均值和最大值。
```lua
function get_weight_norms(module)
    return "Weight norms:\n" .. recursive_map(module, "weight", abs_mean) ..
            "\nWeight max:\n" .. recursive_map(module, "weight", abs_max)
end

function get_grad_norms(module)
    return "Weight grad norms:\n" ..
        recursive_map(module, "gradWeight", abs_mean) ..
        "\nWeight grad max:\n" .. recursive_map(module, "gradWeight", abs_max)
end
```

#### Scale.lua

scale.lua文件定义了训练时的scale层（此时的torch并没有内置scale的层），并定义了forward和updateOutput方法，实际上这两个方法都是相同的功能。
```lua
function scale:updateOutput(input)
    return self:forward(input)
end
```
在scale:forward(x)函数中，x表示输入的图像，该函数会调用image.rgb2y(x)将输入的图像变成灰度图，然后将它按照初始化的宽高进行放缩。

#### Rectifier.lua

同样地，Rectifier.lua文件定义了训练时的ReLU函数层，这里对前向传播和反向传播都进行了定义。
```lua
function Rectifier:updateOutput(input)
    return self.output:resizeAs(input):copy(input):abs():add(input):div(2)
end

function Rectifier:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(self.output)
    return self.gradInput:sign(self.output):cmul(gradOutput) 
end
```
> 这里self.output.resizeAs(input)的意思就是将output，resize成和input同样的size。cmul()表示矩阵对应元素相乘。

#### convnet.lua
convnet.lua文件的目的是建立CNN结构，该文件仅仅包含一个函数：create\_network。输入层的定义由初始化时的input\_dims给出。注意到，在函数里对GPU和CPU的卷积层的实现方式有所区分。
卷积层的数量由初始化时的arg.n\_units的长度给出（arg.n\_units的每个元素的数值表示每一层的输出的feature map个数），如下所示，这里arg.nl()表示非线性层的意思。
```lua
    for i=1,(#args.n_units-1) do ---第二个卷积层到最后一个卷积层
        net:add(convLayer(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1]))
        net:add(args.nl())
    end
```
在卷积的最后一层通过人为构造0的输入的方式，进行前向传播，并对输出层进行nElement()的方法可以求得卷积最后一层的神经元数量。
```lua
nel = net:forward(torch.zeros(1,unpack(args.input_dims))):nElement()
net:add(nn.Reshape(nel)) 
```
然后加入多个线性层，同样的，线性层的数量由arg.n\_hid的长度给出（arg.n\_hid的每个元素的数值表示每个线性层输出的神经元数量）
```lua
    for i=1,(#args.n_hid-1) do  --第二哥线性层到最后一个线性层
        last_layer_size = args.n_hid[i+1]
        net:add(nn.Linear(args.n_hid[i], last_layer_size))
        net:add(args.nl())
    end
```
最后加入一个线性层，其输出神经元的额数量等于actions的数量
```lua
net:add(nn.Linear(last_layer_size, args.n_actions))
```

#### convnet\_atari3.lua
这个文件主要是调用convnet.lua文件，并设置了一些对应的参数。
```lua
return function(args)
    args.n_units        = {32, 64, 64} --三个卷积层，输出的feature map的数量是32,64,64
    args.filter_size    = {8, 4, 3} --每个卷积层的卷积核大小
    args.filter_stride  = {4, 2, 1} --每个卷积层的步长
    args.n_hid          = {512}   --线性层的输出神经元数量
    args.nl             = nn.Rectifier  --非线性类型

    return create_network(args)
end
```

#### net\_downsample\_2x\_full\_y.lua
这个文件会在构建网络时，在输入层增加一个Scale层，此时设置的长和宽均为84，Scale层会将输入的图像先变成灰度图，然后放缩成84x84的大小。

#### TransitionTable.lua
该文件主要创造了一个dqn.TransitionTable类，每个transition表示<s,a,r,s'\>，其中s表示state，a表示actions，r表示rewards，s'表示在s状态下执行a，得到的下一个状态s'。这个类用来存储一定数量的transitions，充当replay memory的角色。在CNN训练时，从这个replay memory中进行sample，sample出来的样本作为了网络的输入。
```lua
local trans = torch.class('dqn.TransitionTable')
```

对于dqn.TransitionTable类，该文件中设计了不少的方法，这里进行一一的解读。
##### trans:__init(args)
首先通过读args直接进行对象的初始化，这里包含的参数如下,在这里hist表示history的意思，每一个history中存储的帧图像合并才构成一个状态（**这样做的原因是因为单独的某一帧的图像无法得到运动物体的速度信息等**）：
```lua
    self.stateDim = args.stateDim    --state的维度
    self.numActions = args.numActions   --Actions的数量
    self.histLen = args.histLen    --History的长度
    self.maxSize = args.maxSize or 1024^2    --最大存储空间大小
    self.bufferSize = args.bufferSize or 1024   --缓存区的大小
    self.histType = args.histType or "linear"  --采样History时使用类型，包括'linear','exp2','exp1.25'
    self.histSpacing = args.histSpacing or 1 --History的间隔，如果histType的类型是’linear‘，表示每个histIndices之间相差histSpacing
    self.zeroFrames = args.zeroFrames or 1  --若该参数为0，则表示每一个history中可以包含不同episode的帧图像
    self.nonTermProb = args.nonTermProb or 1
    self.nonEventProb = args.nonEventProb or 1
    self.gpu = args.gpu
    self.numEntries = 0  --存储transition的数量
    self.insertIndex = 0
    self.histIndices = {} --表示采样时的history下标
```

然后函数会针对不同的self.histType来设定不同的self.histIndices，同时，self.recentMemSize表示存储时的history的跨度，也就是histIndices[histLen]的值。
> 在self.histLen=5的情况下，如果self.histType="linear"，且self.histSpacing=2时，那么self.histIndices={2,4,6,8,10}，self.recentMemSize=10。如果self.histType="exp2"，那么self.histIndices={1,2,4,8,16}，self.recentMemSize=16。

接下来对self.s，self.a，self.r，self.t进行初始化设置。
```lua
    self.s = torch.ByteTensor(self.maxSize, self.stateDim):fill(0) --state，这里的state仅仅指一帧图像
    self.a = torch.LongTensor(self.maxSize):fill(0)  --actions
    self.r = torch.zeros(self.maxSize)  --reward
    self.t = torch.ByteTensor(self.maxSize):fill(0)  --terminal
    self.action_encodings = torch.eye(self.numActions)
```
然后初始化了recent存储区，用来存储最近recentMemSize个帧的图像，也就是说在采样时这里只能采样一个状态，这可以用来建立最新的状态。
```lua
    self.recent_s = {}
    self.recent_a = {}
    self.recent_t = {}
```
另外初始化时也定义了buffer区，在训练时的transition即来自buffer区。
```lua
    local s_size = self.stateDim*histLen  --s_size表示将histLen个帧图像连接在一起构成的新的状态的大小
    self.buf_a      = torch.LongTensor(self.bufferSize):fill(0)
    self.buf_r      = torch.zeros(self.bufferSize)
    self.buf_term   = torch.ByteTensor(self.bufferSize):fill(0)
    self.buf_s      = torch.ByteTensor(self.bufferSize, s_size):fill(0)
    self.buf_s2     = torch.ByteTensor(self.bufferSize, s_size):fill(0) --s2表示s'，即在s下执行a得到的新的s
```
> buffer区的state是由几个frame连接得到的，而self.s仅仅指一帧。

##### trans:reset()
重置transition memory
```lua
function trans:reset()
    self.numEntries = 0
    self.insertIndex = 0
end
```
##### trans:size()
返回self.numEntries

##### trans.empty()
将self.numEntries置0

##### trans.concatFrames(index,use_recent)
该函数负责将histLen个Frames的图像连接在一起，组成一个状态。至于Frames的选取方法，由self.histIndices的值来决定。
use_recent是一个bool型的变量，这个变量决定是否使用recent table
```lua
    if use_recent then
        s, t = self.recent_s, self.recent_t
    else
        s, t = self.s, self.t
    end
```
函数新建了一个局部变量fullstate，用来存储histLen个Frames的数据。函数的输入变量index表示在s中采样的Frames的初始下标。
这个函数会在index与index+self.histIndice[histLen]-1之间的Frames，按照index+self.histIndice的方式进行采样，然而，如果在这些帧图像之间出现了terminal状态，也就是说游戏重新开始了一遍，这里会将出现terminal状态前的采样帧进行**归零**处理。也就是说最后得到的fullstate只包含最新的episode（每次从游戏开始到结束称为一个episode）。最终得到的一个fullstate称为一个状态。
```lua
--初始化fullstate，大小是histLen个s的大小
    local fullstate = s[1].new()
    fullstate:resize(self.histLen, unpack(s[1]:size():totable()))
--将除了最新的episode外的帧图像归零
    local zero_out = false  --归零标志位
    local episode_start = self.histLen   --最新的episode开始的帧在fullstate中的下标

    for i=self.histLen-1,1,-1 do  --反向搜索，一旦搜索到terminal，就对前面的采样进行操作
        if not zero_out then
            for j=index+self.histIndices[i]-1,index+self.histIndices[i+1]-2 do
                if t[j] == 1 then   --t表示terminal，如果在两个采样的帧之间出现了terminal，代表这两个采样属于不同的episode，因此将之前的采样全部归零。
                    zero_out = true
                    break
                end
            end
        end

        if zero_out then   --一旦zero_out变为true之后，会一直保持为ture的状态
            fullstate[i]:zero()
        else
            episode_start = i
        end
    end

    if self.zeroFrames == 0 then --self.zeroFrames参数，一旦等于0，则阻止归零的操作。
        episode_start = 1
    end

    for i=episode_start,self.histLen do
        fullstate[i]:copy(s[index+self.histIndices[i]-1])  --将最新的episode中的帧copy到fullstate中
    end

    return fullstate
```

##### trans:concatActions(index,use_recent)
该函数的作用类似于trans:concatFrames，唯一的区别是它作用的对象是actions。

##### trans:get(index)
调用self:concatFrames(index)得到s和s2，我们取s中的最后一帧的action和reward作为整个state的action和reward，terminal取整个state后的第一帧的t值。
```lua
function trans:get(index)
    local s = self:concatFrames(index)
    local s2 = self:concatFrames(index+1)
    local ar_index = index+self.recentMemSize-1  --训练状态的最后一帧的下标

    return s, self.a[ar_index], self.r[ar_index], s2, self.t[ar_index+1]
end
```

##### trans:sample_one()
在（2，self.numEntries-self.recentMemSize）之间进行均匀采样得到一个index，从2开始的原因是保证有一个previous action，index的最大值是self.numEntries-self.rencentMemSize，这样设置是因为训练的状态的最后一帧的下标与第一帧的下标之间相差recentMemSize。
同时如果self.nonTermProb和self.nonEventProb不等于1的情况下，采样的状态会被随机抛弃。
```lua
function trans:sample_one()
    assert(self.numEntries > 1)
    local index
    local valid = false
    while not valid do
        index = torch.random(2, self.numEntries-self.recentMemSize) --均匀随机采样一个index
        if self.t[index+self.recentMemSize-1] == 0 then
            valid = true
        end
        if self.nonTermProb < 1 and self.t[index+self.recentMemSize] == 0 and torch.uniform() > self.nonTermProb then
        --以（1-self.nonTermProb）的概率抛弃所采样的非terminal状态
            valid = false
        end
        if self.nonEventProb < 1 and self.t[index+self.recentMemSize] == 0 and self.r[index+self.recentMemSize-1] == 0 and torch.uniform() > self.nonEventProb then 
         --以（1-nonEventProb）的概率随机抛弃所采样的非terminal和无reward状态
            valid = false
        end
    end

    return self:get(index)
end
```

##### trans:fill_buffer()
这个函数通过调用trans:sample_one()的函数来进行采样，然后将这些随机采样的样本加入到buffer区。执行这个函数会刷新buffer区的数据。
注意到这里必须保证原存储区的样本个数大于buffer区。
```lua
assert(self.numEntries >= self.bufferSize)
self.buf_ind = 1
```
然后进行采样，注意到该函数调用后会初始化一个类成员变量self.buf\_ind，这个变量表示在buffer中训练时的下标指示器。每次调用该函数就会使这个变量置为1，即表示现在的buffer区的数据还没有被训练。
```lua
    for buf_ind=1,self.bufferSize do
        local s, a, r, s2, term = self:sample_one(1)
        self.buf_s[buf_ind]:copy(s)
        self.buf_a[buf_ind] = a
        self.buf_r[buf_ind] = r
        self.buf_s2[buf_ind]:copy(s2)
        self.buf_term[buf_ind] = term
    end
```

##### trans:sample(batch_size)
在buffer区得到batch\_size个tansition，注意到如果buffer区中所剩下的数据少于batch\_size时会重新更新buffer区。
```lua
function trans:sample(batch_size)
    local batch_size = batch_size or 1
    assert(batch_size < self.bufferSize)

    if not self.buf_ind or self.buf_ind + batch_size - 1 > self.bufferSize then 
        self:fill_buffer() --如果buffer区未更新过，或者剩余的数据量少于batch_size时，重新装填buffer区
    end

    local index = self.buf_ind

    self.buf_ind = self.buf_ind+batch_size -- 更新self.buf_ind的值
    local range = {{index, index+batch_size-1}}

    local buf_s, buf_s2, buf_a, buf_r, buf_term = self.buf_s, self.buf_s2,self.buf_a, self.buf_r, self.buf_term
    return buf_s[range], buf_a[range], buf_r[range], buf_s2[range], buf_term[range]
end
```

##### trans:add(s,a,r,term)
该文件会将一组新的s，a，r，term（terminal）写进存储区，每写进一个数据self.numEntries会加1，直到self.maxSize为止。
```lua
    if self.numEntries < self.maxSize then
        self.numEntries = self.numEntries + 1
    end
```
这里用self.inserIndex来控制写入的下标，当存储区写满后，又从头开始写入。
```lua
    self.insertIndex = self.insertIndex + 1
    -- 如果写满了，则重头开始
    if self.insertIndex > self.maxSize then
        self.insertIndex = 1
    end
```
写入存储区
```lua
    self.s[self.insertIndex] = s:clone():float():mul(255)
    self.a[self.insertIndex] = a
    self.r[self.insertIndex] = r
    if term then
        self.t[self.insertIndex] = 1
    else
        self.t[self.insertIndex] = 0
    end
```

##### trans:add_recent_state(s,term),trans:add_recent_action(a)
这两个函数分别将s，term和a加入recent存储区，注意到由于recent存储区只存储一个状态，因此函数里面有维持recent存储区的大小等于self.recentMemSize的操作。

##### trans:get_recent()
 从recent存储区取一个状态
```lua
function trans:get_recent()
    return self:concatFrames(1, true):float():div(255)
end
```
##### trans:write(file)
将trans类的参数序列化写入文件

##### trans:read(file)
执行反序列化，从文件中读取参数

### [下一页](https://chendagui16.github.io/2016/09/09/Deep-Q-learning-2/)