# sac_train_qc.py - SAC算法训练脚本
# 该文件实现了Soft Actor-Critic (SAC) 算法的训练过程，用于训练无人机在复杂环境中执行目标跟踪和到达任务
# 包含环境交互、经验收集、策略学习和性能评估

import sys  # 系统相关功能
from itertools import count  # 迭代计数器

import numpy as np  # 数值计算库

from env import *  # 导入环境相关类
from sac_model1 import *  # 导入SAC模型
import os  # 操作系统接口
import csv  # CSV文件处理
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 解决OpenMP库冲突问题

from tqdm import tqdm # 进度条库


# evaluate_policy 函数：评估训练好的策略性能
# 参数：
#   env: 测试环境
#   agent: SAC智能体
#   iter_num: 评估轮数
# 返回：平均episode奖励、平均步奖励、平均episode成本、平均步成本
def evaluate_policy(env, agent, iter_num):
    times = 1  # 执行一次评估（可以设置为多次取平均）
    step_num = 0  # 总步数
    evaluate_reward = 0  # 累计奖励
    evaluate_cost = 0  # 累计成本
    env.rewards_step['num'] = iter_num  # 设置环境评估轮数

    for _ in range(times):
        s = env.reset()  # 重置环境
        done = False  # episode是否结束
        episode_reward = 0  # 单episode奖励
        episode_cost = 0  # 单episode成本

        while not done:
            a = agent.choose_action(s, deterministic=True)  # 使用确定性策略（评估时不加噪声）
            s_, r, done, _ = env.step(a)  # 执行动作
            episode_reward += r[0]  # 累加奖励
            episode_cost += r[1]  # 累加成本
            s = s_  # 更新状态
            step_num += 1  # 步数计数

        evaluate_reward += episode_reward  # 累加episode奖励
        evaluate_cost += episode_cost  # 累加episode成本

    # 返回平均值
    return (evaluate_reward / times), (evaluate_reward / (times * step_num)), (evaluate_cost / times), (evaluate_cost / (times * step_num))


# 训练配置
# continue_train = True  # 是否继续训练（从检查点恢复）
continue_train = False  # 从头开始训练
result_path = './'  # 结果保存路径
buffer_path = './'  # 缓冲区保存路径
csv_file_path = './'  # CSV文件路径

# 环境参数
init_energy = 30.  # 初始能量
# uav_init_state = np.array([500., 2600., 100., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., init_energy])
# # 无人机初始状态：[x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, dot_phi, energy]

#---------- uav_init_state 从 14 维扩展为 16 维 (追加 phi=0, dot_phi=0)----1
uav_init_state = np.array([500., 2600., 100., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., init_energy, 0., 0.])

#--------------------1


uav_target_position = np.array([0., 2600., 100.])  # 无人机目标终点位置
# target_position_0 = np.array([250., 3000., 0.])  # 备用目标位置
target_position_1 = np.array([500., 3000., 0.])  # 目标初始位置

target_velocity = 0.  # 目标速度
target_model = 0  # 目标模型
end_time = 2000  # 结束时间（400s，时间槽=0.2s）
time_slot = 1  # 时间槽长度
dt = 1  # 时间步长

# SAC算法超参数
gamma = train_config['gamma']  # 折扣因子
batch_size = train_config['batch_size']  # 批次大小
exploration_noise = train_config['exploration_noise']  # 探索噪声

# 创建环境
env = Environment(uav_init_state, target_position_1, uav_target_position, init_energy, time_slot)  # 训练环境
env_test = copy.deepcopy(env)  # 测试环境（深拷贝，避免干扰训练）

# 状态和动作维度
# state_dim = 14 + 3 + 3 + 2 + 1  # 状态维度：无人机状态(14) + 终点相对位置(3) + 目标相对位置(3) + 目标速度(2) + 感知总量(1)
# action_dim = 4  # 动作维度：4个控制输入
#----状态维度：无人机+天线(16) + 终点相对(3) + 目标相对(3) + 目标速度(2) + 天线自身(2) + 夹角特征(2) + 感知总量(1)----1
state_dim = 14 + 3 + 3 + 2 + 2 + 2 + 1  # 等于 27
action_dim = 5  # 增加1维天线控制
#-----------------------------------1



# 创建SAC智能体
agent = SAC(state_dim, action_dim)

# 如果继续训练，从检查点加载模型和缓冲区
# agent.load('best')  # 可选择加载最佳模型
if continue_train:
    agent.load('last')  # 加载最新模型
    agent.replay_buffer.save_or_load_history(buffer_path, False)  # 加载缓冲区历史
    # 加载训练历史数据
    rewards_test_list = np.load(result_path+"rewards_test_list.npy").tolist()
    rewards_done_list = np.load(result_path+"rewards_done_list.npy").tolist()
    rewards_step_list = np.load(result_path+"rewards_step_list.npy").tolist()

    distance_list = np.load(result_path+"distance_list.npy").tolist()
    energy_list = np.load(result_path+"energy_list.npy").tolist()
    sensing_list = np.load(result_path+"sensing_list.npy").tolist()
    time_list = np.load(result_path+"time_list.npy").tolist()
else:
    # 初始化训练历史列表
    rewards_test_list = []
    rewards_done_list = []
    rewards_step_list = []

    distance_list = []
    energy_list = []
    sensing_list = []
    time_list = []

not_start_train = True  # 标记是否开始训练
costs_test_list = []  # 测试成本列表
costs_step_list = []  # 步成本列表


# 训练参数
max_train_steps = train_config['max_train_steps']  # 最大训练步数
evaluate_freq = 500  # 评估频率（每500步评估一次）
best_reward = -10000.  # 最佳episode奖励
step_best_reward = -10000.  # 最佳步奖励
arrive_flag_test = 0  # 到达标志测试
total_steps = 0  # 总训练步数
evaluate_num = 0  # 评估次数

# 经验收集缓冲区
state_list = []  # 状态列表
action_list = []  # 动作列表
reward_list = []  # 奖励列表
mask_list = []  # 掩码列表（用于折扣因子）


# <--- [新增 3] 初始化 tqdm 进度条
pbar = tqdm(total=max_train_steps, initial=total_steps, desc="C-SAC Training", unit="step")


# 主训练循环
while total_steps < max_train_steps:
    ep_r_test_mean = 0.  # 测试episode奖励均值
    ep_r_test_done = 0.  # 测试完成episode奖励
    step_r_test = 0.  # 测试步奖励
    distance_mean = 0.  # 距离均值
    energy_mean = 0.  # 能量均值
    sensing_mean = 0.  # 感知均值
    time_mean = 0.  # 时间均值

    state = env.reset_random_init()  # 随机重置环境初始状态

    done = 0  # episode结束标志
    while not done:
        # 前1000步使用随机动作以提高探索
        if total_steps < 1000:
            action = (np.random.normal(0, 2, size=action_dim)).clip(-1, 1)  # 随机动作，限制在[-1,1]
        else:
            action = agent.choose_action(state)  # 使用策略选择动作

        total_steps += 1  # 总步数计数

        pbar.update(1)  # <--- [新增 3] 每次走一步，进度条推进一步

        state_, reward, done, arrive_flag = env.step(action, dt)  # 执行动作，获取下一状态、奖励、结束标志

        # 计算折扣掩码
        mask = [0.0 if done else gamma]  # 如果episode结束，掩码为0，否则为gamma

        # 收集经验数据
        state_list.append(state)  # 添加当前状态
        action_list.append(action)  # 添加动作
        if reward[1] > 0:  # 如果有成本（惩罚）
            reward_list.append([-reward[1]])  # 使用负成本作为奖励
        else:
            reward_list.append([reward[0]])  # 使用正奖励
        mask_list.append(mask)  # 添加掩码

        state = state_  # 更新状态

        # 每500步更新一次缓冲区
        if (total_steps + 1) % 500 == 0:
            # 将收集的经验添加到重放缓冲区
            agent.replay_buffer.update_buffer(torch.FloatTensor(np.array(state_list)).to(device),
                                              torch.FloatTensor(np.array(action_list)).to(device),
                                              torch.as_tensor(np.array(reward_list)).to(device),
                                              torch.FloatTensor(np.array(mask_list)).to(device))
            # 清空临时缓冲区
            state_list = []
            action_list = []
            reward_list = []
            mask_list = []

            # 如果缓冲区容量足够，开始训练
            if agent.replay_buffer.cur_capacity >= batch_size:
                if not_start_train:
                    not_start_train = False
                    print('----------------------------------------------------')
                    print('start training!')  # 开始训练标志
                    print('----------------------------------------------------')

                # 执行30次更新
                for _ in range(30):
                    agent.update()

                # 每5000步保存临时模型
                if (total_steps + 1) % 5000 == 0:
                    agent.save('tmp')

        # 定期评估策略性能
        if (total_steps + 1) % evaluate_freq == 0:
            evaluate_num += 1  # 评估次数计数
            # 执行评估
            ep_r_test, step_r_test, ep_c_test, step_c_test = evaluate_policy(env_test, agent, evaluate_num)

            # 保存评估结果
            rewards_test_list.append(ep_r_test)
            np.save(result_path + 'rewards_test_list.npy', rewards_test_list)
            rewards_step_list.append(step_r_test)
            np.save(result_path + 'rewards_step_list.npy', rewards_step_list)
            costs_test_list.append(ep_c_test)
            np.save(result_path + 'costs_test_list.npy', costs_test_list)
            costs_step_list.append(step_c_test)
            np.save(result_path + 'costs_step_list.npy', costs_step_list)

            # # 打印评估结果
            # print(
            #     'train_test: {:d} uav_position: ({:.2f}, {:.2f},  {:.2f}) energy: {:.2f} distance_end_point: {:.2f} '.format(
            #         evaluate_num,
            #         env_test.uav.position[0], env_test.uav.position[1], env_test.uav.position[2],  # 无人机位置
            #         env_test.uav.energy,  # 能量
            #         env_test.uav.distance_end_point  # 到终点距离
            #     ) + ' time: {:.2f} sensing: {:.2f} reward: {:.2f} cost: {:.2f}'.format(
            #         env_test.time_index,  # 时间
            #         env_test.uav.sensing_total,  # 感知总量
            #         ep_r_test,  # episode奖励
            #         ep_c_test  # episode成本
            #     ))

            # 保存最佳模型
            if ep_r_test > best_reward:
                agent.save('best')
                best_reward = ep_r_test

            if step_r_test > step_best_reward:
                agent.save('step_best')
                step_best_reward = step_r_test

             # <--- [新增 3] 将最新的奖励和惩罚实时挂在进度条尾巴上
            pbar.set_postfix({
                'Eval': evaluate_num,
                'Reward': f"{ep_r_test:.1f}",
                'Cost': f"{ep_c_test:.1f}",
                'Best': f"{best_reward:.1f}"
            })

# 训练结束，保存最终模型和缓冲区
pbar.close()  # <--- [新增 3] 结束时关闭进度条

# 训练结束，保存最终模型和缓冲区
agent.save('last')  # 保存最新模型
agent.replay_buffer.save_or_load_history(buffer_path, True)  # 保存缓冲区历史
