import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import copy
import os

# 导入你原有的环境和模型
from env import Environment, Target
from sac_model1 import SAC
from config import train_config

# 设置全局绘图字体格式 (类似 IEEE 论文风格)
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

def plot_convergence(result_path='./'):
    """
    1. 画出 C-SAC 算法的 Reward 与 Cost 收敛曲线
    """
    try:
        rewards = np.load(result_path + 'rewards_test_list.npy')
        costs = np.load(result_path + 'costs_test_list.npy')
    except FileNotFoundError:
        print("未找到 rewards_test_list.npy 或 costs_test_list.npy，请确保训练已经保存了数据。")
        return

    # 计算滑动平均以平滑曲线 (Window size = 10)
    def moving_average(a, n=10):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    smooth_rewards = moving_average(rewards, n=10)
    smooth_costs = moving_average(costs, n=10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reward 曲线
    ax1.plot(rewards, alpha=0.3, color='blue', label='Raw Reward')
    ax1.plot(np.arange(9, len(rewards)), smooth_rewards, color='blue', linewidth=2, label='Smoothed Reward')
    ax1.set_xlabel('Evaluation Episodes')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title('C-SAC Reward Convergence')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Cost 曲线
    ax2.plot(costs, alpha=0.3, color='red', label='Raw Cost')
    ax2.plot(np.arange(9, len(costs)), smooth_costs, color='red', linewidth=2, label='Smoothed Cost')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='Zero Cost Threshold')
    ax2.set_xlabel('Evaluation Episodes')
    ax2.set_ylabel('Cumulative Cost (Penalty)')
    ax2.set_title('C-SAC Cost Convergence')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('Convergence_Curves_1.png', dpi=300)
    plt.show()

def evaluate_and_plot_best_model():
    """
    2. 加载最好模型并画出 3D轨迹、速率、感知MI、角度解耦等核心曲线
    """
    print("加载最佳模型并开始采集物理遥测数据...")
    
    # 1. 环境与智能体初始化
    init_energy = 30.
    uav_init_state = np.array([500., 2600., 100., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., init_energy, 0., 0.])
    uav_target_position = np.array([0., 2600., 100.])
    target_position_1 = np.array([500., 3000., 0.])
    
    # 请确保这里的维度与 sac_train_qc.py 修改后保持完全一致！
    state_dim = 14 + 3 + 3 + 2 + 2 + 2 + 1  # 27维
    action_dim = 5  
    
    env = Environment(uav_init_state, target_position_1, uav_target_position, init_energy, time_slot=1)
    agent = SAC(state_dim, action_dim)
    
    # 加载最佳模型
    try:
        agent.load('best')
    except Exception as e:
        print(f"加载模型失败: {e}。请确保 SAC_model 文件夹下有 actor_best.pth")
        return

    # 2. 数据采集列表
    logs = {
        'time': [], 'uav_x': [], 'uav_y': [], 'uav_z':[],
        'tar_x': [], 'tar_y': [], 'tar_z': [],
        'yaw':[], 'phi': [], 'theta_RA': [], 'theta_tar': [], 'theta_bs':[],
        'rate_c': [], 'rate_s': [], 'rho': [], 'cum_MI':[]
    }

    state = env.reset()
    done = False
    cum_mi = 0.
    
    # 物理参数 (用于重新计算绘图数据)
    bs_pos = np.array([0., 0., 30.])
    
    while not done:
        # 使用确定性策略进行测试评估
        action = agent.choose_action(state, deterministic=True)
        state_next, reward, done, info = env.step(action, dt=1.0)
        
        # 提取无人机物理状态
        pos = env.uav.position
        tar_pos = env.target.position
        uav_state = env.uav.state
        
        # 提取姿态和天线角度
        q0, q1, q2, q3 = uav_state[9:13]
        phi = uav_state[14]
        
        # 计算偏航角(Yaw)和天线绝对朝向
        yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
        theta_RA = yaw + phi
        
        # 计算目标和基站方位角
        theta_tar = np.arctan2(tar_pos[1] - pos[1], tar_pos[0] - pos[0])
        theta_bs = np.arctan2(bs_pos[1] - pos[1], bs_pos[0] - pos[0])
        
        # 计算瞬时信道增益 (MRT原则下的余弦衰减)
        M = 8
        gain_c = M * max(0, np.cos(theta_bs - theta_RA))**2
        gain_s = M * max(0, np.cos(theta_tar - theta_RA))**2
        
        # 近似计算当前速率与 rho (为绘图展示，采用简化公式，lambda参数适当缩放)
        dist_c = np.linalg.norm(pos - bs_pos)
        dist_s = np.linalg.norm(pos - tar_pos)
        rate_c = np.log2(1 + gain_c * 1e8 / (dist_c**2 + 1))
        rate_s = np.log2(1 + gain_s * 1e9 / (dist_s**4 + 1))
        rho = max(0, min(1, (rate_c - 2) / (rate_c + rate_s + 1e-6) if rate_c > 2 else 0))
        
        # 累积 MI
        cum_mi += rho * rate_s * 1.0 # dt = 1.0

        # 记录数据
        logs['time'].append(env.time_index)
        logs['uav_x'].append(pos[0]); logs['uav_y'].append(pos[1]); logs['uav_z'].append(pos[2])
        logs['tar_x'].append(tar_pos[0]); logs['tar_y'].append(tar_pos[1]); logs['tar_z'].append(0)
        logs['yaw'].append(np.degrees(yaw)); logs['phi'].append(np.degrees(phi))
        logs['theta_RA'].append(np.degrees(theta_RA))
        logs['theta_tar'].append(np.degrees(theta_tar)); logs['theta_bs'].append(np.degrees(theta_bs))
        logs['rate_c'].append(rate_c); logs['rate_s'].append(rate_s); logs['rho'].append(rho)
        logs['cum_MI'].append(cum_mi)
        
        state = state_next

    print(f"数据采集完毕！存活时间: {logs['time'][-1]} 秒")

    # ==================== 开始绘制 4 张核心图表 ====================

    # 图 1: 3D 飞行轨迹与波束快照图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(logs['uav_x'], logs['uav_y'], logs['uav_z'], label='UAV Trajectory', color='b', linewidth=2)
    ax.plot(logs['tar_x'], logs['tar_y'], logs['tar_z'], label='Target Trajectory', color='r', linestyle='--')
    ax.scatter(0, 0, 30, color='g', marker='^', s=100, label='Base Station (BS)')
    ax.scatter(logs['uav_x'][0], logs['uav_y'][0], logs['uav_z'][0], color='k', marker='o', label='Start')
    ax.scatter(0, 2600, 100, color='m', marker='*', s=150, label='End Point')
    
    # 画出波束方向快照 (每隔一定时间画一个箭头)
    step_size = max(1, len(logs['time']) // 10)
    for i in range(0, len(logs['time']), step_size):
        length = 150 # 波束箭头可视化长度
        dx = length * np.cos(np.radians(logs['theta_RA'][i]))
        dy = length * np.sin(np.radians(logs['theta_RA'][i]))
        ax.quiver(logs['uav_x'][i], logs['uav_y'][i], logs['uav_z'][i], dx, dy, 0, color='orange', alpha=0.6, arrow_length_ratio=0.2)
    
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title('3D UAV Trajectory with RA-ULA Beam Snapshots')
    ax.legend()
    plt.savefig('3D_Trajectory_Beam_1.png', dpi=300)
    plt.show()

    # 图 2: 实时速率与时间分配占比曲线 (双 Y 轴)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(logs['time'], logs['rate_c'], color='blue', label='Communication Rate ($R_c$)')
    ax1.plot(logs['time'], logs['rate_s'], color='green', label='Sensing Rate ($R_s$)')
    ax1.set_xlabel('Flight Time (s)')
    ax1.set_ylabel('Data Rate (Mbits/s)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2 = ax1.twinx()
    ax2.plot(logs['time'], logs['rho'], color='red', linestyle='--', label='Sensing Time Ratio ($\\rho$)')
    ax2.set_ylabel('Sensing Time Ratio $\\rho$', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    plt.title('Real-time ISAC Rate & Time Allocation Ratio')
    plt.savefig('Realtime_Rate_Rho_1.png', dpi=300)
    plt.show()

    # 图 3: 累计感知互信息量 (Cumulative MI)
    plt.figure(figsize=(10, 5))
    plt.plot(logs['time'], logs['cum_MI'], color='purple', linewidth=2, label='Cumulative MI (RA-ULA)')
    plt.fill_between(logs['time'], logs['cum_MI'], color='purple', alpha=0.1)
    plt.xlabel('Flight Time (s)')
    plt.ylabel('Cumulative Sensing MI (Mbits)')
    plt.title('Cumulative Sensing Mutual Information Over Time')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('Cumulative_MI_1.png', dpi=300)
    plt.show()

    # 图 4: 角度解耦曲线 (机体偏航角 vs 天线相对转角 vs 目标方位)
    plt.figure(figsize=(10, 6))
    plt.plot(logs['time'], logs['yaw'], label='UAV Yaw ($\\theta_{body}$)', color='blue', linestyle='-.')
    plt.plot(logs['time'], logs['phi'], label='Antenna Relative Angle ($\\phi$)', color='green')
    plt.plot(logs['time'], logs['theta_RA'], label='Absolute Beam Direction ($\\theta_{RA}$)', color='red', linewidth=2)
    plt.plot(logs['time'], logs['theta_tar'], label='Target Direction ($\\theta_{tar}$)', color='black', linestyle=':')
    
    # 标出天线机械限位 (假设设定为 +-90度)
    plt.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='Mechanical Limit $\pm\phi_{max}$')
    plt.axhline(y=-90, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Flight Time (s)')
    plt.ylabel('Angle (Degrees)')
    plt.title('Mechanical Decoupling: UAV Attitude vs. Antenna Rotation')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('Angle_Decoupling_1.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    print("====== 开始生成分析图表 ======")
    # 1. 绘制收敛图
    plot_convergence()
    
    # 2. 加载最佳模型生成物理曲线
    evaluate_and_plot_best_model()
    print("====== 图表生成完毕，已保存为图片文件 ======")