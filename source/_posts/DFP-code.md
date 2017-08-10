---
title: DFP_code
mathjax: true
date: 2017-08-07 21:10:59
categories: reinforecement learning
tags: [reinforcement learning, code]
---

# Code Structure

* common
	* util.py
	* tf_ops.py
	* defaults.py
* Simulator
  * doom_simulator.py
  * multi_doom_simulator.py
* Agent 
	* agent.py
	* future_predictor_agent_basic.py
	* future_predictor_agent_advantage.py
	* future_predictor_agent_advantage_nonorm.py
* target
	* future_target_maker.py
* experience
	* multi_experience_memory.py
	* multi_experience.py

# Common

## util.py

### make_objective_indices_and_coeffs(temporal_coeffs, meas_coeffs):
网络使用两个系数作为目标的系数作为最终决断的系数，一个是temporal_coeffs, 即时间尺度上的系数，分别对应不同时间段对目标值的贡献，meas_coeffs表示不同的measure对目标值的贡献。

### make_array(shape=(1,), dtype=np.float32, shared=False, fill_val=None):
生成array，如果有array共享的话，从共享的区域中生成array

### merge_two_dicts(x, y)
合并两个dict，用于defaults.py 中的参数和不同模型的参数合并

### StackedBarPlot
用于画不同的预测值的直方图，并可以动态显示

## tf_ops

### msra_stddev(x, k_h, k_w)
参数初始化方差

### mse_ignore_nans(preds, targs. **kwargs)
忽略Nan 求MSE

### conv2d(input_, output_dim, k_h, k_w, d_h, d_w, msra_coeff, name)
构建卷积层的api

### lrelu(x, leak, name)
构建lrelu层的api

### linear(input_, output_size, name, msra_coeff)
线性函数api

### conv_encoder(data, params, name, msra_coeff)
根据conv_params的参数构建连续的卷积层

### fc_net(data, param, name, last_linear, return_layers, msra_coeff)
根据fc_params的参数构建连续的全连接层；根据return_layers的性质，确定最后一层的fc是否要接relu

### flatten(data)
构建flatten层的api

## defaults.py
default.py主要提供不同的默认参数dict，其中主要分为4类， target_maker, simulator, experiment(experiment, train_experiment, test_policy), agent。 给不同的文件提供不同的默认参数配置， 在具体使用的时候，该参数会被不同文件中的参数给覆盖。

### target_maker_args
这里的min_num_targs, 表示目标target的最小数量， 对于那些target数量不足的样本，会用nan替换

### simulator_args
frame\_skip表示不使用连续的帧作为状态，使用间隔的帧作为状态。

### experience_args
这里分为三个文件experience_args, train_experience_args, test_policy_experience_args

### agent_args
存储所有关于网络的参数

# Simulator

## DoomSimulator
DoomSimulator类

### __init__(self, args)
初始化程序配置，设置doom的各项参数，具体参考[doom_doc](https://github.com/mwydmuch/ViZDoom/tree/master/doc)
使用analyze_controls(self, config_file), 解析获得available_controls, continuous_controls, discrete_control，注意到这些都是index. 比如continuous_controls = [1,2,3], discrete_control = [4,5,6]

### analyze_controls(self, config_file)
利用正则表达式解析available_controls, continuous_controls, discrete_control.

### init_game, close_game, get_random_action, is_new_episode, next_map, new_episode
函数作用如名所示，注意doom的具体动作可以使用多个动作的组合，比如可控的动作是[1, 2]， 那么实际可能的动作为[True, True] or [True, False] or [False, True] or [False, False]

### step(self, action)
执行一步动作，返回img，measure, reward, terminal. 这里action是一个bool vector, 图像均为黑白(彩色会报错), img指doomSimulator返回的图像，meas指config文件中的指定的variable
，这里特指{AMMO2, HEALTH, USER2}(这里的user2指kill_count, 在wad文件里面使用). rwrd特指doomSimulator返回的reward，这个是doom内部自带的reward.

## MultiDoomSimulator
多个doomSimulator的类，可以用来同时执行一个action的列表，同时返回imgs, meass, rwrds, terms（均为list）


# Agent

## agent.py

### __init__(self, sess, args):
初始化各项参数
同时调用prepare_controls_and_actions(), 初始化所有的control和action

### prepare_controls_and_actions(self)
注意到这里controls表示action的index，action用bool vector表示。其中controls分为两类，一类是由网络生成的discrete_controls_to_net, 一类是外部赋予的discrete_controls_manual. 同时net_discrete_actions是一个numpy array，表示list of vector(除掉了冲突的action), 这些vector可以用来直接执行动作。
另外，初始化了action_to_index的dict，用来查询不同的action的index, 初始化了onehot_dizhscrete_actions用一个one-hot向量表达不同的action，这里可以用作网络的输入。

### preprocess_actiosn(self, acts)
这里的acts表示batch_size个动作，每个动作用一个bool vector(包含了net_action和manunal_action)表示，该函数的作用是把这些acts转化成对应的one-hot-vector输入到网络中去.

### postprocess_actions(self, acts_net, act_manual)
将acts_net和act_manual组合成一个正式的输出的动作，该函数的输入的acts_net是一个[batch_size]的输入，每个值均为action的index，act_manual是一个bool_vector。将act_nets, act_manual组合成一个正式的action，可以用做simulator的输入.

### random_actions(self, num_samples)
获得随机的action

### make_net, make_losses，act_nets, act_manual
这几个函数都没有执行，在装饰器文件如future_predictor_agent_basic等中执行

### act(self, state_imgs, state_meas, objective_coeffs)
调用act_net得到net_action，调用act_manual得到munual_action，使用postprocess_actions函数将其组合成一个具体的动作

### build_model(self)
该函数主要构建tf的模型，具体的模型参考论文
* 需要输入的参数包括input_images, input_measurement, input_targets, input_actions, input_objective_coeffs
* 如果提供了预处理的函数，则会根据预处理函数对输入进行预处理，预处理的方式参考论文
	* 将img的值放缩在[-1, 1]
	* 将mesurements的值放缩在[-1, 1]
	* 将三个target分别除以7.5, 30, 1
* 调用make_net和make_loss构建网络
* 构建训练节点，学习率衰减节点，summary节点

### Actor
一个子类，用作实际操作action的一个接口
**__init__(self, agent, objective_coeffs, random_prob, random_objective_coeffs)**
初始化objective_coeffs, 如果使用随机的初始化策略，则调用reset_objective_coeffs(self, indices), 使用均匀分布来初始化，否则直接对objective_coeffs赋值
**reset_objective_coeffs(self, indices)**
使用均匀分布初始化objective_coeffs
**act(self, state_imgs, state_meas)**
$\epsilon$-greedy算法，以$\epsilon$的概率执行随机的动作，以1-$\epsilon$的概率调用agent.act()函数执行动作
**act_with_multi_memory(slef, multi_memory)**
执行和act一样的功能，但是针对mutil_memory，调用这个函数执行多个动作要比使用上面的函数高校，因为这样只会在需要的时候才会读取state。

### get_actor(self, objective_coeffs, random_prob, random_objective_coeffs)
返回actor的子类

### train_one_batch(self, experience)
* 调用experience.get_random_batch函数得到state_imgs, state_meas, rwrds, terms, acts, targs, objs
* forward网络一次
* 在特定的步数打印错误，存储summary，或存储histogram
* 步数+1

### train(self, simulator, experience, num_steps, test_police_experience)
* 如果可以的话，载入checkpoint
* 初始化writer, 和actor接口
* 调用experience.add_n_steps_with_actor填充训练的memory
* 调用train_one_batch共num_steps次，一共训练num_steps个batch
* 每隔固定的步数保存checkpoint或者测试test_policy，同时每隔一定的步数重新填充memory，注意到这里会衰减$\epsilon$

### test_policy(self, simulator, experience, objective_coeffs, num_steps, random_prob, write_summary, write_prdiction)
测试一次policy, 注意到这里会调用experience.compute_avg_meas_and_rwrd去计算得到这一次的policy的平均reward和平均measure。
另外，为了保证每次测试不改变head_offset, 会暂时保存该值，并在测试完之后恢复

### save, load, set_init_step
功能如其，set_init_step可以接着之前的step继续训练

## future_predictor_agent_*.py
这里一共包含3个文件(future_predictor_agent_basic, future_predictor_agent_advantage, future_predictor_agent_advantage_nonorm), 都是作为agent类的子类，重写了agent类中的make_net, make_losses, act_nets的成员函数，用于对比实验

### make_net(self, input_images, input_measurement, input_actions, input_objectives, reuse)
* 根据卷积和全联接的参数调用tf_ops中的接口来搭建网络
	* future_predictor_agent_basic中没有使用分支结构
	* future_predictor_agent_advantage中是论文中的标准结构
	* future_predictor_agent_advantage_nonorm是论文中的标准结构但是去除了normalization的环节
* 这里input_images, input_measurement, input_actions均为网络的输入，而input_actions(bool of vector)用来指出哪个所有的预测中与当前动作相关的预测。

### make_losses(self, pred_relevant, targets_preprocessed, objective_indices, objective_coeffs)
构建loss的计算节点，并同时去除nan，构建summry节点

### act_net(self, state_imgs, state_meas, objective_coeffs)
* 更新prediction
* 按照objective_coeffs求出使收益最大的动作。

# target

## future_target_maker.py
target_maker类，负责生成网络的学习目标

### __init__(self, args)
构造函数，注意几点
* min_num_targs: target的最小数量，对于某些靠近end of episode的样本，其可能的target数量少于min_num_targs，所以这些样本无效，会用nan替代
* 根据不同的gamma值，对不同的future step对reward进行指数衰减。在具体的实验中，由于并没有用到simulator中的reward值，因此gamma值为空
* 每个时刻需要预测的target的维度包括两个部分: 1, 不同gamma衰减下的reward；2，不同的measure。 所以num_targets = len(self.meas_to_predict) + self.num_reward_targets。注意num_targets的长度必须与agent中的objective_coeffs_meas的长度相同，因为objective_coeffs_meas就是用来衡量不同的target对最终结果的加权系数
* 总的target的维度为self.num_targets * len(self.future_steps). 注意self.future_steps的长度必须与agent中的objective_coeffs_temporal的长度相同
* 根据min_num_targs，确定min_future_frames，将来用来确定该帧与之后的min_future_frames处的帧是否处于同一个episode来判断当前帧是否有效.

### make_targets(self, indices, meas, rwrds, n_episode, meas_mean, meas_std)
生成对应indices处的targets
该函数主要是在multi_experience_memory处调用，所以提供的meas，rwards都是一个大的memory, indices负责指示想要生成batch的样本所在memory处的位置.
* capacity指memory的大小
* target分为两个部分，一个是measurement, 一个是reward
* 对于measurement，如果有meas_mean或者meas_std的话，会进行normalization的处理，不在一个epsisode的样本的measurement会用最近的measurement进行替换。
* 对于reward，target是对一个时间窗口(self.future_steps)内的rwrd进行指数衰减下的加权求和。

# experience

## multi_experience_memory.py
MultiExperienceMemory类

### __init__(self, args, multi_simulator, target_maker)
初始化空的memory，这里包含两个假设：1，观测都是连续的，在每一个episode内都有一个停止的状态；2，每一个episode都比memory的长度要短。
调用reset函数，初始化各项私有变量

### reset(self)
* self._curr_indices设置成不同memory的indices
* self._episode_counts用来设置不同memory的eposide值，用来区分不同frame在memory中的位置

### add(self, imgs, meass, rwrds, terms, acts, objs, preds)
* 更新images, measurements, rewards, terminals, actions，n_episode, objectives, predictions
* 在term处的measurements使用上一个状态measurements，同时对于term处，对measurement进行一定的衰减
* 为了防止出现两个term状态，在出现第二个term时，将第二个term状态的measurements置零
* 更新episode_counts, curr_indices和terminals

### add_step(self, multi_simulator, acts, objs, preds)
调用multi_simulator执行一次acts, 并将其结果加入memory

### add_n_steps_with_actor(self, multi_simulator, num_steps, actor, verbose, write_prediction, write_logs, global_step)
* log中会写入具体的episode, step, time, accu_reward, prev_meas, avg_meas
* 如果有prediction的话，会写入prediction
* 调用add_steps共num_steps次，使用actor类的接口加入memory

### get_states(self, indices)
根据indices从memory中选择state_imgs, state_meas。注意到这里的的state_imgs包含history_step个历史，且每个历史之间相差不同的history_len
而state_meas仅仅包含当前indices处的measurements

### get_current_state(self)
获得当前最近的观测值

### get_last_indices(self)
返回最近的indices

### get_target(self, indices)
使用target_maker.make_targets类返回indices处的targets.
这里使用了一个hack，对不同的memory处使用了一12345678 * self._n\_head的数值，用来区分不同的head处的episode值

### has_valid_history(self, index)
判断index处是否有足够的history进行训练

### has_valid_target(self, index)
判断index处是否有足够的target进行训练

### is_valid_target(self, index)
判断index处的状态是否有效，即是否有足够的history和target

### get_observations(self, indices)
得到indices处的所有信息包括
state_imgs, state_meas, rwrds, terms, acts, targs, objs

### get_random_batch(self, batch_size)
从memory中随机采用batch_size个有效的样本

### compute_avg_meas_and_rwrd(self, start_idx, end_idx)
统计从start_idx，end_idx中所有的episode中的平均measurements和rewards

### show(self, start_idx, end_idx, display, write_imgs, write_video, preprocess_targets, show_predictions, net_discrete_actions)
show的接口，提供display, write_imgs, write_vedio, show_prediction的选项

## multi_experiement.py

### __init__(self, target_maker_args, simulator_args, train_experience_args, test_policy_experience_args, agent_args, experiment_args)
将默认的参数和给定的参数进行合并，得到具体的参数.
部分参数在此处求解，比如各种shape的参数

### run(self, mode)
* show模式，将训练的head_offset和测试的head_offset错开。测试policy，并show_memory中的结果
* train模式，训练网络, 更新参数.
