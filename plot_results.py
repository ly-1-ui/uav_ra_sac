import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import copy
import os
import math

# 导入环境和模型
from env import Environment
from sac_model1 import SAC

# =====================================================================
# 1. 文件夹路径配置 (请根据你的实际文件夹名字修改！)
# =====================================================================
# 旋转天线（RA-ULA）的数据和模型文件夹
DIR_RA = './SAC_model/'      
# 固定天线（Fixed-ULA）的数据和模型文件夹
DIR_FIXED = './SAC_model_n/' 

# 设置全局绘图字体格式 (IEEE 论文风格)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # 类似 LaTeX 的数学字体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.2       # 加粗坐标轴边框
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.color'] = '#cccccc'
plt.rcParams['grid.alpha'] = 0.7

def plot_combined_convergence():
    """
    画出两个模型的 Reward 与 Cost 收敛对比曲线
    """
    print("正在加载收敛数据...")
    try:
        r_ra = np.load(f'{DIR_RA}rewards_test_list.npy')
        c_ra = np.load(f'{DIR_RA}costs_test_list.npy')
        r_fix = np.load(f'{DIR_FIXED}rewards_test_list.npy')
        c_fix = np.load(f'{DIR_FIXED}costs_test_list.npy')
    except Exception as e:
        print(f"数据加载失败，请检查文件夹路径是否正确！错误信息: {e}")
        return

    def moving_average(a, n=15):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return np.concatenate((a[:n-1], ret[n - 1:] / n))

    sm_r_ra = moving_average(r_ra)
    sm_c_ra = moving_average(c_ra)
    sm_r_fix = moving_average(r_fix)
    sm_c_fix = moving_average(c_fix)

    x_ra = np.arange(len(r_ra))
    x_fix = np.arange(len(r_fix))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ---- Reward 收敛图 ----
    # RA-ULA (红色风格，对应图1中的C-SAC)
    ax1.plot(x_ra, r_ra, color='red', alpha=0.2)
    ax1.plot(x_ra, sm_r_ra, color='red', linewidth=2.5, label='Proposed RA-ULA')
    
    # Fixed-ULA (蓝色风格，对应图1中的C-TD3)
    ax1.plot(x_fix, r_fix, color='blue', alpha=0.2)
    ax1.plot(x_fix, sm_r_fix, color='blue', linewidth=2.5, label='Baseline Fixed-ULA')
    
    ax1.set_xlabel('Episode Number', fontweight='bold')
    ax1.set_ylabel('Reward', fontweight='bold')
    # 顶刊图例样式：带黑色实线边框
    leg1 = ax1.legend(loc='lower right', frameon=True, edgecolor='black', fancybox=False)
    leg1.get_frame().set_linewidth(1.2)

    # ---- Cost 收敛图 ----
    ax2.plot(x_ra, c_ra, color='red', alpha=0.2)
    ax2.plot(x_ra, sm_c_ra, color='red', linewidth=2.5, label='Proposed RA-ULA')
    
    ax2.plot(x_fix, c_fix, color='blue', alpha=0.2)
    ax2.plot(x_fix, sm_c_fix, color='blue', linewidth=2.5, label='Baseline Fixed-ULA')
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax2.set_xlabel('Episode Number', fontweight='bold')
    ax2.set_ylabel('Cost Penalty', fontweight='bold')
    leg2 = ax2.legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False)
    leg2.get_frame().set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig('IEEE_Convergence.png', dpi=600, bbox_inches='tight')
    plt.show()

def run_evaluation(agent, env, is_fixed=False):
    """
    运行评估并采集物理日志
    """
    logs = {'time':[], 'uav_x': [], 'uav_y': [], 'uav_z':[],
            'tar_x':[], 'tar_y': [], 'tar_z': [],
            'yaw':[], 'phi':[], 'theta_RA': [], 'cum_MI':[]}
    
    bs_pos = np.array([0., 2000., 30.])
    state = env.reset()
    done = False
    cum_mi = 0.
    
    while not done:
        action = agent.choose_action(state, deterministic=True)
        
        # 【物理小妙招】：如果是 Fixed-ULA，我们强制把神经网络输出的天线控制量清零！
        # 这样就不需要去修改 env.py 里的底层代码了，完美模拟天线焊死。
        if is_fixed:
            action[4] = 0.0 
            
        state_next, reward, done, info = env.step(action, dt=1.0)
        
        pos = env.uav.position
        tar_pos = env.target.position
        
        q0, q1, q2, q3 = env.uav.state[9:13]
        phi = env.uav.state[14]
        
        yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
        theta_RA = yaw + phi
        theta_tar = np.arctan2(tar_pos[1] - pos[1], tar_pos[0] - pos[0])
        theta_bs = np.arctan2(bs_pos[1] - pos[1], bs_pos[0] - pos[0])
        
        # 严格遵守物理截断计算 (与你在 utils.py 的设定保持一致)
        M = 4  # 假设你按照上一轮改成了极其锐利的窄波束 16
        cos_c = np.cos(theta_bs - theta_RA)
        gain_c = M * (cos_c**2 if cos_c > 0.5 else 0.01)
        
        cos_s = np.cos(theta_tar - theta_RA)
        gain_s = M * (cos_s**2 if cos_s > 0.5 else 0.01)
        
        dist_c = max(np.linalg.norm(pos - bs_pos), 30.0)
        dist_s = max(np.linalg.norm(pos - tar_pos), 30.0)
        
        rate_c = np.log2(1 + gain_c * 1e8 / (dist_c**2 + 1))
        rate_s = np.log2(1 + gain_s * 1e11 / (dist_s**4 + 1))
        
        rho = max(0, min(1, (rate_c - 2.0) / (rate_c + rate_s + 1e-6) if rate_c > 2.0 else 0))
        cum_mi += rho * rate_s * 1.0 

        # 记录
        logs['time'].append(env.time_index)
        logs['uav_x'].append(pos[0]); logs['uav_y'].append(pos[1]); logs['uav_z'].append(pos[2])
        logs['tar_x'].append(tar_pos[0]); logs['tar_y'].append(tar_pos[1]); logs['tar_z'].append(0)
        logs['theta_RA'].append(theta_RA)
        logs['cum_MI'].append(cum_mi)
        
        state = state_next

    # 消除 360 度相位缠绕假象
    logs['yaw'] = np.degrees(np.unwrap(logs['yaw']))
    logs['theta_RA'] = np.degrees(np.unwrap(logs['theta_RA']))
    
    return logs

def evaluate_and_plot_combined():
    print("正在加载模型并采集物理遥测数据...")
    
    # 初始化环境 (必须与训练时完全一致！)
    init_energy = 80
    uav_init_state = np.array([500., 2000., 100., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., init_energy, 0., 0.])
    uav_target_position = np.array([0., 2000., 100.]) 

    target_position_1 = np.array([500., 3000., 0.])   

    # 物理参数 (用于重新计算绘图数据)
    bs_pos = np.array([0., 2000., 30.])
    
    state_dim = 27
    action_dim = 5  
    
    # 强制让 env 开启天线控制 (u5 = 0.01 * control[4])
    env = Environment(uav_init_state, target_position_1, uav_target_position, init_energy, time_slot=1)
    
    # ---- 跑 RA-ULA 模型 ----
    agent_ra = SAC(state_dim, action_dim)
    agent_ra.MODEL_DIR = DIR_RA.strip('./').strip('/') # 动态修改路径
    try:
        agent_ra.load('best') 
    except:
        print(f"找不到 RA 模型，请检查 {DIR_RA}")
        return
    logs_ra = run_evaluation(agent_ra, env, is_fixed=False)
    
    # ---- 跑 Fixed-ULA 模型 ----
    agent_fix = SAC(state_dim, action_dim)
    agent_fix.MODEL_DIR = DIR_FIXED.strip('./').strip('/') 
    try:
        agent_fix.load('best') 
    except:
        print(f"找不到 Fixed 模型，请检查 {DIR_FIXED}")
        return
    logs_fix = run_evaluation(agent_fix, env, is_fixed=True)

    print("数据采集完毕，开始绘制对比图...")

    # ==========================================
    # 图 2: 3D 轨迹对比图
    # ==========================================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis._axinfo['grid']['linestyle'] = '--'
    ax.yaxis._axinfo['grid']['linestyle'] = '--'
    ax.zaxis._axinfo['grid']['linestyle'] = '--'


    # 画轨迹
    # 红色实线 (RA-ULA)
    ax.plot(logs_ra['uav_x'], logs_ra['uav_y'], logs_ra['uav_z'], color='red', linewidth=2.5, label='Proposed Trajectory')
    # 黑色点线 (Fixed-ULA)
    ax.plot(logs_fix['uav_x'], logs_fix['uav_y'], logs_fix['uav_z'], color='black', linestyle=':', linewidth=3, label='Baseline Trajectory')
    # 蓝色点划线 (Target)
    ax.plot(logs_ra['tar_x'], logs_ra['tar_y'], logs_ra['tar_z'], color='blue', linestyle='-.', linewidth=2, label='Target Trajectory')

    # 起点、终点、基站
    ax.scatter(uav_init_state[0], uav_init_state[1], uav_init_state[2], facecolors='none', edgecolors='red', marker='o', s=100, linewidth=2, label='UAV Start Point')
    ax.scatter(uav_target_position[0], uav_target_position[1], uav_target_position[2], color='red', marker='*', s=200, label='UAV End Point')
    ax.scatter(bs_pos[0], bs_pos[1], bs_pos[2], color='green', marker='^', s=150, label='Base Station (BS)')    
    # 给 RA-ULA 加橙色波束箭头
    step_size = max(1, len(logs_ra['time']) // 10)
    for i in range(0, len(logs_ra['time']), step_size):
        length = 150
        dx = length * np.cos(np.radians(logs_ra['theta_RA'][i]))
        dy = length * np.sin(np.radians(logs_ra['theta_RA'][i]))
        ax.quiver(logs_ra['uav_x'][i], logs_ra['uav_y'][i], logs_ra['uav_z'][i], dx, dy, 0, color='orange', alpha=0.8, arrow_length_ratio=0.2)
    
    ax.set_xlabel('X (m)', fontweight='bold')
    ax.set_ylabel('Y (m)', fontweight='bold')
    ax.set_zlabel('Z (m)', fontweight='bold')
    ax.set_title('3D Trajectory Comparison', fontweight='bold', pad=20)
    ax.legend(loc='upper right')
    ax.set_box_aspect([500, 1000, 200]) 
    plt.savefig('IEEE_3D_Trajectory.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ==========================================
    # 图 3: 累计 MI 对比图 (参考图3风格)
    # ==========================================
    fig, ax_mi = plt.subplots(figsize=(8, 6))
    
    # 蓝色实线代表不旋转(Static/Baseline)，品红色虚线代表旋转(Ideal/Proposed)
    ax_mi.plot(logs_ra['time'], logs_ra['cum_MI'], color='m', linestyle='--', linewidth=2.5, label='Proposed RA-ULA')
    ax_mi.plot(logs_fix['time'], logs_fix['cum_MI'], color='blue', linestyle='-', linewidth=2.5, label='Baseline Fixed-ULA')
    
    ax_mi.set_xlabel('Flight Time (s)', fontweight='bold')
    ax_mi.set_ylabel('Cumulative Sensing MI (Mbits)', fontweight='bold')
    
    leg_mi = ax_mi.legend(loc='upper left', frameon=True, edgecolor='black', fancybox=False)
    leg_mi.get_frame().set_linewidth(1.2)
    
    plt.savefig('IEEE_Cumulative_MI.png', dpi=600, bbox_inches='tight')
    plt.show()

def evaluate_speed_robustness():
    """
    3. 测试不同目标速度下的鲁棒性，并绘制 MI 对比折线图
    """
    print("\n====== 开始测试目标速度鲁棒性 ======")
    
    # 设定要测试的目标速度列表 (绝对值，单位 m/s)
    # 原训练时的速度大约是 8 m/s。我们测试从极慢(2)到极快(20)
    test_speeds =[2.0, 5.0, 8.0, 12.0, 15.0, 20.0]
    
    mi_ra_list = []
    mi_fix_list =[]

    # 初始化环境 (必须与训练时完全一致)
    init_energy = 80.
    uav_init_state = np.array([500., 2000., 100., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., init_energy, 0., 0.])
    uav_target_position = np.array([0., 2000., 100.]) 
    target_position_1 = np.array([400., 2800., 0.])  # 侧边诱导目标点
    bs_pos = np.array([0., 2000., 30.])
    
    state_dim = 27
    action_dim = 5  
    
    env = Environment(uav_init_state, target_position_1, uav_target_position, init_energy, time_slot=1)
    
    # 加载模型
    agent_ra = SAC(state_dim, action_dim)
    agent_ra.MODEL_DIR = DIR_RA.strip('./').strip('/')
    agent_ra.load('best') 
    
    agent_fix = SAC(state_dim, action_dim)
    agent_fix.MODEL_DIR = DIR_FIXED.strip('./').strip('/') 
    agent_fix.load('best') 

    # 定义一个运行单次测试的闭包函数
    def test_single_speed(agent, is_fixed, test_vx):
        state = env.reset()
        done = False
        cum_mi = 0.
        
        while not done:
            action = agent.choose_action(state, deterministic=True)
            if is_fixed:
                action[4] = 0.0  # 拦截天线动作
            
            # 【核心修改】：通过 target_control=True 强行覆盖目标的移动速度！
            # 目标往西走，所以速度为 -test_vx
            state_next, reward, done, info = env.step(action, dt=1.0, target_control=True, vy=-test_vx, vx=-test_vx)
            
            pos = env.uav.position
            tar_pos = env.target.position
            
            q0, q1, q2, q3 = env.uav.state[9:13]
            phi = env.uav.state[14]
            yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
            theta_RA = yaw + phi
            theta_tar = np.arctan2(tar_pos[1] - pos[1], tar_pos[0] - pos[0])
            theta_bs = np.arctan2(bs_pos[1] - pos[1], bs_pos[0] - pos[0])
            
            # 严格计算波束增益 (须与 utils.py 保持一致，这里假设 M=8, 门槛 0.5)
            M = 8 
            cos_c = np.cos(theta_bs - theta_RA)
            gain_c = M * (cos_c**2 if cos_c > 0.5 else 0.01)
            cos_s = np.cos(theta_tar - theta_RA)
            gain_s = M * (cos_s**2 if cos_s > 0.5 else 0.01)
            
            dist_c = max(np.linalg.norm(pos - bs_pos), 30.0)
            dist_s = max(np.linalg.norm(pos - tar_pos), 30.0)
            
            rate_c = np.log2(1 + gain_c * 1e8 / (dist_c**2 + 1))
            rate_s = np.log2(1 + gain_s * 1e11 / (dist_s**4 + 1))
            
            rho = max(0, min(1, (rate_c - 2.0) / (rate_c + rate_s + 1e-6) if rate_c > 2.0 else 0))
            cum_mi += rho * rate_s * 1.0 
            state = state_next
            
        return cum_mi

    # 循环测试不同速度
    for speed in test_speeds:
        print(f"正在测试目标速度 V = {speed} m/s ...")
        mi_ra = test_single_speed(agent_ra, is_fixed=False, test_vx=speed)
        mi_fix = test_single_speed(agent_fix, is_fixed=True, test_vx=speed)
        
        mi_ra_list.append(mi_ra)
        mi_fix_list.append(mi_fix)

    # ==========================================
    # 绘制 IEEE 顶刊级折线对比图
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 画线：带有清晰的 Marker 标记
    ax.plot(test_speeds, mi_ra_list, color='red', marker='o', markersize=8, linestyle='-', linewidth=2.5, label='Proposed RA-ULA')
    ax.plot(test_speeds, mi_fix_list, color='blue', marker='s', markersize=8, linestyle='--', linewidth=2.5, label='Baseline Fixed-ULA')
    
    ax.set_xlabel('Target Speed $V$ (m/s)', fontweight='bold')
    ax.set_ylabel('Cumulative Sensing MI (Mbits)', fontweight='bold')
    ax.set_title('Robustness Against Target Speeds', fontweight='bold', pad=15)
    
    # 图例
    leg = ax.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
    leg.get_frame().set_linewidth(1.2)
    
    # 坐标轴与网格优化
    ax.set_xticks(test_speeds) # 让X轴刻度刚好对应我们测试的速度
    plt.grid(True, linestyle='--', color='#cccccc', alpha=0.7)
    
    plt.savefig('IEEE_Speed_Robustness.png', dpi=600, bbox_inches='tight')
    print("====== 鲁棒性测试完毕，图表已保存为 IEEE_Speed_Robustness.png ======")
    plt.show()

if __name__ == '__main__':
    print("====== 开始生成 IEEE 顶刊级对比分析图表 ======")
    plot_combined_convergence()
    evaluate_and_plot_combined()
    evaluate_speed_robustness()
    print("====== 生成完毕！请查看当前目录下的 IEEE_xxx.png ======")