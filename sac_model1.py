# sac_model1.py - SAC (Soft Actor-Critic) 算法实现
# 该文件实现了软演员-评论家算法，用于连续动作空间的强化学习
# 主要包含策略网络(Actor)、价值网络(Critic)和SAC算法主体

import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 神经网络函数
import numpy as np  # 数值计算
import copy  # 深拷贝
from tensorboardX import SummaryWriter  # TensorBoard可视化
from torch.distributions import Normal  # 正态分布
from buffer import ReplayBuffer  # 经验回放缓冲区

# 设备选择：优先使用GPU，否则使用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from config import parser  # 导入配置解析器
args = parser.parse_args()  # 解析命令行参数

# Actor 类：策略网络（Policy Network）
# 作用：学习确定性策略，使用高斯分布建模动作的概率分布
class Actor(nn.Module):
    # 初始化方法：构建策略网络结构
    # 参数：
    #   state_dim: 状态维度
    #   action_dim: 动作维度
    #   hidden_width: 隐藏层宽度
    #   max_action: 动作最大值（用于缩放输出）
    def __init__(self, state_dim, action_dim, hidden_width, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action  # 动作最大值

        # 特征提取网络：多层全连接网络
        self.layer = nn.Sequential(
            nn.Linear(state_dim, 128),  # 输入层到第一个隐藏层
            nn.ReLU(True),  # ReLU激活函数
            nn.Linear(128, 256),  # 第一个隐藏层到第二个隐藏层
            nn.ReLU(True),
            nn.Linear(256, 256),  # 第二个隐藏层到第三个隐藏层
            nn.ReLU(True),
            nn.Linear(256, 256),  # 第三个隐藏层到第四个隐藏层
            nn.ReLU(True),
            nn.Linear(256, 256),  # 第四个隐藏层到第五个隐藏层
            nn.ReLU(True),
            nn.Linear(256, 128),  # 第五个隐藏层到输出前的隐藏层
            nn.ReLU(True),
        )

        # 输出层：均值和对数标准差
        self.mean_layer = nn.Linear(128, action_dim)  # 动作均值输出层
        self.log_std_layer = nn.Linear(128, action_dim)  # 动作对数标准差输出层

    # 前向传播方法：根据状态生成动作
    # 参数：
    #   x: 输入状态
    #   deterministic: 是否使用确定性策略（评估时使用）
    #   with_logprob: 是否返回对数概率
    # 返回：动作和对数概率
    def forward(self, x, deterministic=False, with_logprob=True):
        x = self.layer(x)  # 特征提取

        mean = self.mean_layer(x)  # 计算动作均值
        log_std = self.log_std_layer(x)  # 计算对数标准差
        log_std = torch.clamp(log_std, -20, 2)  # 限制对数标准差范围，避免数值不稳定
        std = torch.exp(log_std)  # 计算标准差

        # 创建正态分布
        dist = Normal(mean, std)

        if deterministic:  # 确定性策略：直接使用均值
            a = mean
        else:
            a = dist.rsample()  # 重参数化技巧：从分布中采样

        if with_logprob:  # 计算对数概率（用于SAC的熵项）
            log_pi = dist.log_prob(a).sum(dim=1, keepdim=True)  # 对数概率
            # 熵修正项：用于tanh压缩的熵校正（来自OpenAI Spinning Up，更稳定）
            log_pi -= (2 * (np.log(2) - a - F.softplus(-2 * a))).sum(dim=1, keepdim=True)
        else:
            log_pi = None

        # 使用tanh将无界高斯分布压缩到有界动作区间 [-max_action, max_action]
        a = self.max_action * torch.tanh(a)

        return a, log_pi


# Network 类：基础网络结构
# 作用：通用的多层感知机网络，用于构建价值函数
class Network(nn.Module):
    # 初始化方法：构建基础网络
    # 参数：
    #   state_dim: 输入维度（状态或状态+动作）
    def __init__(self, state_dim):
        super(Network, self).__init__()
        # 多层全连接网络结构
        self.layer = nn.Sequential(
            nn.Linear(state_dim, 128),  # 输入层
            nn.ReLU(True),
            nn.Linear(128, 256),  # 第一个隐藏层
            nn.ReLU(True),
            nn.Linear(256, 256),  # 第二个隐藏层
            nn.ReLU(True),
            nn.Linear(256, 256),  # 第三个隐藏层
            nn.ReLU(True),
            nn.Linear(256, 256),  # 第四个隐藏层
            nn.ReLU(True),
            nn.Linear(256, 128),  # 第五个隐藏层
            nn.ReLU(True),
            nn.Linear(128, 1),  # 输出层（标量值）
        )

    # 前向传播：计算网络输出
    def forward(self, x):
        x = self.layer(x)
        return x


# Critic 类：评论家网络（Q网络）
# 作用：估计状态-动作对的价值函数 Q(s,a)
class Critic(nn.Module):
    # 初始化方法：构建双Q网络结构
    # 参数：
    #   state_dim: 状态维度
    #   action_dim: 动作维度
    #   hidden_width: 隐藏层宽度（未使用）
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()

        # 双Q网络：使用两个独立的网络来减少过估计
        self.network1 = Network(state_dim + action_dim)  # Q1网络
        self.network2 = Network(state_dim + action_dim)  # Q2网络

    # 前向传播：计算两个Q值的估计
    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)  # 将状态和动作拼接作为输入
        return self.network1(x), self.network2(x)  # 返回两个Q值估计


# SAC 类：Soft Actor-Critic 算法主体
# 作用：实现完整的SAC算法，包括训练和推理
class SAC(object):
    # 初始化方法：设置算法参数和网络
    # 参数：
    #   state_dim: 状态维度
    #   action_dim: 动作维度
    #   max_action: 动作最大值
    def __init__(self, state_dim, action_dim, max_action=1):
        self.max_action = max_action
        self.hidden_width = 256  # 隐藏层神经元数量
        self.batch_size = 256  # 批次大小
        self.GAMMA = 0.99  # 折扣因子
        self.TAU = 0.005  # 软更新系数
        self.lr = 3e-4  # 学习率

        # 初始化经验回放缓冲区
        self.replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim,
                                          max_capacity=args.capacity)

        # 自适应温度系数alpha
        self.adaptive_alpha = True  # 是否自动学习温度参数
        if self.adaptive_alpha:
            # 目标熵：-dim(A)，用于平衡探索和利用
            self.target_entropy = -action_dim
            # 学习log_alpha而不是alpha，确保alpha > 0
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()  # 温度参数
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)  # alpha优化器
        else:
            self.alpha = 0.2  # 固定温度参数

        # 初始化网络
        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action).to(device)  # 策略网络
        self.critic = Critic(state_dim, action_dim, self.hidden_width).to(device)  # Q网络
        self.critic_target = copy.deepcopy(self.critic).to(device)  # 目标Q网络

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)  # 策略网络优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)  # Q网络优化器

    # choose_action 方法：根据当前状态选择动作
    # 参数：
    #   state: 当前状态
    #   deterministic: 是否使用确定性策略
    # 返回：选择的动作
    def choose_action(self, state, deterministic=False):
        # 将状态转换为张量并移到设备
        state = torch.FloatTensor(state).to(device)
        # 从策略网络采样动作（不需要计算对数概率）
        a, _ = self.actor(state, deterministic, False)
        # 返回CPU上的numpy数组
        return a.detach().cpu().numpy()

    # update 方法：执行一次训练更新
    def update(self):
        # 从缓冲区采样一个批次的数据
        batch_s, batch_a, batch_r, batch_s_, batch_dw = self.replay_buffer.sample_batch(self.batch_size)

        # 计算目标Q值（使用目标网络）
        with torch.no_grad():
            batch_a_, log_pi_ = self.actor(batch_s_)  # 从当前策略采样下一动作
            # 计算目标Q值
            target_Q1, target_Q2 = self.critic_target(batch_s_, batch_a_)
            # SAC目标Q值：r + γ * (min(Q1,Q2) - α * logπ)
            target_Q = batch_r + batch_dw * (torch.min(target_Q1, target_Q2) - self.alpha * log_pi_)

        # 计算当前Q值
        current_Q1, current_Q2 = self.critic(batch_s, batch_a)
        # Q网络损失：均方误差
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # 更新Q网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 冻结Q网络参数，避免浪费计算
        for params in self.critic.parameters():
            params.requires_grad = False

        # 计算策略网络损失：α * logπ - Q
        a, log_pi = self.actor(batch_s)  # 从策略采样动作
        Q1, Q2 = self.critic(batch_s, a)  # 计算Q值
        Q = torch.min(Q1, Q2)  # 使用最小Q值
        actor_loss = (self.alpha * log_pi - Q).mean()  # SAC策略损失

        # 更新策略网络
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 解冻Q网络参数
        for params in self.critic.parameters():
            params.requires_grad = True

        # 更新温度参数alpha
        if self.adaptive_alpha:
            # alpha损失：-α * (logπ + H_target)
            alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()  # 更新alpha

        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

    # save 方法：保存模型参数
    # 参数：
    #   ep: 训练轮数
    def save(self, ep):
        torch.save(self.actor.state_dict(), './SAC_model/actor_' + str(ep) + '.pth')
        torch.save(self.critic.state_dict(), './SAC_model/critic_' + str(ep) + '.pth')
        torch.save(self.critic_target.state_dict(), './SAC_model/critic_target_' + str(ep) + '.pth')


    # load 方法：加载模型参数
    # 参数：
    #   ep: 训练轮数
    def load(self, ep):
        self.actor.load_state_dict(torch.load('./SAC_model/actor_' + str(ep) + '.pth'))
        self.critic.load_state_dict(torch.load('./SAC_model/critic_' + str(ep) + '.pth'))
        self.critic_target.load_state_dict(torch.load('./SAC_model/critic_target_' + str(ep) + '.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


# evaluate_policy 函数：评估策略性能
# 作用：在测试环境中评估训练好的策略，返回平均奖励
# 参数：
#   env: 测试环境
#   agent: SAC智能体
# 返回：平均奖励
def evaluate_policy(env, agent):
    times = 1  # 执行一次评估
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()  # 重置环境
        done = False
        episode_reward = 0
        while not done:
            # 使用确定性策略选择动作（评估时不探索）
            a = agent.choose_action(s, deterministic=True)
            s_, r, done, _ = env.step(a)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return int(evaluate_reward / times)


# reward_adapter 函数：奖励适配器
# 作用：对不同环境的奖励进行标准化处理
# 参数：
#   r: 原始奖励
#   env_index: 环境索引
# 返回：适配后的奖励
def reward_adapter(r, env_index):
    if env_index == 0:  # Pendulum-v1
        r = (r + 8) / 8  # 将奖励归一化到[0,1]区间
    elif env_index == 1:  # BipedalWalker-v3
        if r <= -100:
            r = -1  # 将失败奖励设为-1
    return r


# 主程序：SAC算法训练入口
if __name__ == '__main__':
    from env import *  # 导入环境模块
    env_name = ['UAV-ISAC']  # 环境名称列表
    env_index = 0  # 选择的环境索引

    # 初始化UAV参数
    init_energy = 30.  # 初始能量
    #uav_init_state = np.array([500., 2600., 100., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., init_energy])  # UAV初始状态

    #----初始化--------------------1
    uav_init_state = np.array([500., 2600., 100., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., init_energy, 0., 0.])
    #-----------------------------1

    uav_target_position = np.array([0., 2600., 100.])  # UAV目标位置
    target_position_1 = np.array([500., 3000., 0.])  # 目标位置

    target_velocity = 0.  # 目标速度
    target_model = 0  # 目标模型
    end_time = 2000  # 结束时间（400s，时间间隔0.2s）
    time_slot = 1  # 时间间隔
    dt = 1  # 时间步长
    # state_dim = 14 + 3 + 3 + 2 + 1  # 状态维度：UAV状态 + 目标位置 + 终点 + 目标速度 + 感知总量
    # action_dim = 4  # 动作维度

    #-----------状态维度----------1
    state_dim = 16 + 3 + 3 + 2 + 2 + 2 + 1  # 29
    action_dim = 5
    #-------------------------------1



    # 创建训练和测试环境
    env = Environment(uav_init_state, target_position_1, uav_target_position, init_energy, time_slot)  # 训练环境
    env_test = copy.deepcopy(env)  # 测试环境（深拷贝）

    # 设置随机种子
    seed = 10
    number = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    max_action = 1  # 动作最大值
    max_episode_steps = 100  # 每个episode的最大步数
    print("env={}".format(env_name[env_index]))
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_action={}".format(max_action))
    print("max_episode_steps={}".format(max_episode_steps))

    # 初始化SAC智能体
    agent = SAC(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(state_dim, action_dim)  # 经验回放缓冲区

    # 创建tensorboard记录器
    writer = SummaryWriter(log_dir='.results/sac/SAC_env_{}_number_{}_seed_{}'.format(env_name[env_index], number, seed))

    # 训练参数
    max_train_steps = 1e5  # 最大训练步数
    random_steps = 1e3  # 初始随机探索步数
    evaluate_freq = 1e3  # 评估频率
    evaluate_num = 0  # 评估次数
    evaluate_rewards = []  # 评估奖励记录
    total_steps = 0  # 总训练步数

    # 主训练循环
    while total_steps < max_train_steps:
        s = env.reset()  # 重置环境
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            # 初始随机探索阶段
            if total_steps < random_steps:
                a = (np.random.normal(0, 2, size=action_dim)).clip(-1, 1)  # 随机动作
            else:
                a = agent.choose_action(s)  # 使用策略选择动作
            s_, r, done, _ = env.step(a)  # 执行动作
            r = reward_adapter(r, env_index)  # 奖励适配

            # 判断是否为终止状态（死亡/胜利 vs 达到最大步数）
            if done and episode_steps != max_episode_steps:
                dw = True  # 死亡或胜利，没有下一状态
            else:
                dw = False  # 达到最大步数，有下一状态

            replay_buffer.store(s, a, r, s_, dw)  # 存储经验
            s = s_

            # 训练阶段
            if total_steps >= random_steps:
                agent.update()  # 更新智能体（注意：这里应该调用agent.update()而不是agent.update(replay_buffer)，因为replay_buffer已经在__init__中设置）

            # 定期评估策略
            if (total_steps + 1) % evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(env_test, agent)  # 评估策略
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{}".format(evaluate_num, evaluate_reward))
                print('uav_position:', env_test.uav.position, 'energy:', env_test.uav.energy, 'distance_end_point:',
                      env_test.uav.distance_end_point, 'time:', env_test.time_index,
                      'sensing:', env_test.uav.sensing_total, 'reward:', evaluate_reward)
                writer.add_scalar('step_rewards_{}'.format(env_name[env_index]), evaluate_reward, global_step=total_steps)
                # 定期保存奖励数据
                if evaluate_num % 10 == 0:
                    np.save('./results/sac/SAC_env_{}_number_{}_seed_{}.npy'.format(env_name[env_index], number, seed), np.array(evaluate_rewards))

            total_steps += 1
