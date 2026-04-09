from scipy.integrate import odeint
from math import dist, log2, pi, sin, sqrt
import numpy as np

# global C1, C2
# motor parameter
C0 = 0.036
C1 = 0.0075
C2 = 8.5938e-6
C3 = 8.8949e-7
C4 = 5.1287e-10

# UAV parameter
mass = 3
L = 0.3
gravity = 9.8
Ixy = 4.29e-2
Iz = 7.703e-2
Im = 8.02e-4 / Ixy
Ct = 4.848e-5
Ct1 = 4.848e-5 / mass
Ct2 = Ct * L / Ixy
Ct3 = Ct2
Cm = 8.891e-7 / Iz

Cdxy = 0.11 / mass
Cdz = 0.2 / mass
Cdmxy = 0.016 / Ixy
Cdmz = 0.1 / Iz
k1 = (Ixy - Iz) / Ixy
k2 = (Iz - Ixy) / Ixy

#---------- 新增 RA 物理参数 -----------1
M=4             # ULA阵元数量
Ja = 0.05      # 天线阵列转动惯量
da = 0.01      # 天线转动阻尼系数
eta_a = 0.8    # 旋转电机效率
#-----------------------------------1

Gt = 10 ** 1.7  # 雷达天线增益17dBi
Gc = 10 ** 2.0  # 基站接收天线增益20dBi
fc = 28e9  # 载波频率(28GHz)
c = 3e8  # 光速
lambda0 = c / fc  # 载波波长(m)
B = 1e7  # 带宽10 MHz
k0 = 1.38e-23  # 玻尔兹曼常数 J/K
T0 = 290  # 绝对温度273+17（摄氏度）=290K
# Fn = 10  # 接收机噪声系数
coss = 2  # 雷达目标反射截面积
# Ls = 1.5  # 总损耗

lambda1 = Gt * Gc * lambda0 ** 2 / ((4 * pi) ** 2 * k0 * T0 * B)
lambda2 = Gt * Gt * lambda0 ** 2 * coss / ((4 * pi) ** 3 * k0 * T0 * B)
# lambda0 = 91039451901.5000
# lambda1 = 7261891670.98391


def uav_power(control) -> object:
    """
    :rtype: object
    """
    power = 0
    for i in range(4):
        power += C0 + C1 * control[i] + C2 * control[i] ** 2 + C3 * control[i] ** 3 + C4 * control[i] ** 4
    return (power + 10) * 0.001  # communication power 1 w, camera power 9 w


def u_saturation(control):
    # control_max = [26, 0.55, 0.55, 0.02]
    # control_min = [6, -0.55, -0.55, -0.02]

    #以下是初始版本
    # control_max = [26, 0.7, 0.7, 0.05]
    # control_min = [6, -0.7, -0.7, -0.05]
    #   # 控制量饱和
    # for i in range(4):
    #     if control[i] > control_max[i]:
    #         control[i] = control_max[i]
    #     if control[i] < control_min[i]:
    #         control[i] = control_min[i]

    #----------新增动作维度第五维------1
    control_max =[26, 0.7, 0.7, 0.05, 0.01]  # tau_max = 0.01
    control_min =[6, -0.7, -0.7, -0.05, -0.01] 
    for i in range(5):
        if control[i] > control_max[i]:
            control[i] = control_max[i]
        if control[i] < control_min[i]:
            control[i] = control_min[i]
    #----------------------------------1



  
    return control


def uavfun(x, t, u, target_position, bs_position):
    state = x
    state_num = len(state)
    position = x[0:3]  # 不包括3
    target_dist = dist(position, target_position)
    bs_dist = dist(position, bs_position)
    

    # 控制量饱和处理
    control = u_saturation(u)

    u1 = control[0] / Ct1
    u2 = control[1] / Ct2
    u3 = control[2] / Ct3
    u4 = control[3] / Cm

    #--------新增天线力矩----1
    tau_a = control[4] # [NEW] 获取天线力矩
    #----------1

    omega1 = 0.5 * (u1 + u4 - 2 * u3) ** 0.5
    omega2 = 0.5 * (u1 - u4 + 2 * u2) ** 0.5
    omega3 = 0.5 * (u1 + u4 + 2 * u3) ** 0.5
    omega4 = 0.5 * (u1 - u4 - 2 * u2) ** 0.5
    omega = omega1 - omega2 + omega3 - omega4
    omega_uav = [omega1, omega2, omega3, omega4]

    state_dot = [0] * state_num
    omega_x = state[6]
    omega_y = state[7]
    omega_z = state[8]
    q0 = state[9]
    q1 = state[10]
    q2 = state[11]
    q3 = state[12]

    #------天线状态-----1
    phi_RA, dot_phi_RA = state[14], state[15] # [NEW] 提取天线状态
    #-------------------1

    state_dot[0] = state[3]
    state_dot[1] = state[4]
    state_dot[2] = state[5]
    state_dot[3] = (control[0] * 2 * (q1 * q3 + q0 * q2) - Cdxy * abs(state[3]) * state[3])
    state_dot[4] = (control[0] * 2 * (q2 * q3 - q0 * q1) - Cdxy * abs(state[4]) * state[4])
    state_dot[5] = (control[0] * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2) - Cdz * abs(state[5]) * state[5]) - gravity
    state_dot[6] = (control[1] + k1 * omega_y * omega_z - Im * omega * omega_y - Cdmxy * abs(state[6]) * state[6])
    state_dot[7] = (control[2] + k2 * omega_x * omega_z + Im * omega * omega_x - Cdmxy * abs(state[7]) * state[7])
    state_dot[8] = (control[3] - Cdmz * abs(state[8]) * state[8])
    state_dot[9] = -0.5 * (omega_x * q1 + omega_y * q2 + omega_z * q3)
    state_dot[10] = 0.5 * (omega_x * q0 + omega_z * q2 - omega_y * q3)
    state_dot[11] = 0.5 * (omega_y * q0 - omega_z * q1 + omega_x * q3)
    state_dot[12] = 0.5 * (omega_z * q0 + omega_y * q1 - omega_x * q2)
    # state_dot[13] = - uav_power(omega_uav)
    # # com_rate = 10 * log2(1 + lambda0 / bs_dist ** 2)
    # rate1 = log2(1 + lambda1 / bs_dist ** 2)
    # rate2 = log2(1 + lambda2 / target_dist ** 4)
    # if rate1 > 8:
    #     pho = (rate1-8)/(rate1+rate2)
    # else:
    #     pho = 0
    # state_dot[14] = 10 * (1-pho) * rate1
    # # sen_rate = 10 * log2(1 + lambda2*pt / target_dist ** 4)
    # state_dot[15] = 10 * pho * rate2
    # # state_dot[14] = 10*sen_rate - 10 * (min(state[2] - 60, 0) ** 2 + min(state[2] + 0.2*state[5] - 60, 0) ** 2 +
    # #                                      min(com_rate - sen_rate - 80, 0) ** 2)
    # state_dot[16] = (min(state[2] - 60, 0) * 10)
    # # 新增：将pho作为状态向量的最后一个分量（假设原状态维度为17，新增后为18）
    # state_dot[17] = pho  # 将pho赋值给新增的状态导数

    #---------新增----1
    # 2. [NEW] 天线阵列机械动力学
    state_dot[14] = dot_phi_RA
    state_dot[15] = (1 / Ja) * tau_a - da * dot_phi_RA

    # 3.[NEW] 计算波束绝对角度与 MRT 阵列增益
    # 计算无人机机体偏航角 (Yaw)
    theta_body = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
    theta_RA_abs = theta_body + phi_RA  # 天线阵列在世界坐标系下的绝对朝向

    # 计算目标和基站相对于无人机的方位角
    theta_c = np.arctan2(bs_position[1] - position[1], bs_position[0] - position[0])
    theta_s = np.arctan2(target_position[1] - position[1], target_position[0] - position[0])

    # # 结合 MRT 原则：波束对准产生 M 倍增益，外加阵元方向图的余弦衰减
    # gain_c = M * max(0, np.cos(theta_c - theta_RA_abs))**2
    # gain_s = M * max(0, np.cos(theta_s - theta_RA_abs))**2

    # # 4. [MODIFIED] 修改 ISAC 速率公式（融入阵列增益）
    # rate1 = log2(1 + (gain_c * lambda1) / bs_dist ** 2)
    # rate2 = log2(1 + (gain_s * lambda2) / target_dist ** 4)
    
    # # 自适应求解时间分配因子 rho (pho)
    # if rate1 > 8:
    #     pho = (rate1 - 8) / (rate1 + rate2)
    # else:
    #     pho = 0

    # #------平滑梯度版波束增益-------------2
    # # 将 max(0, cos) 改为 (cos + 1)/2，这样即使在背面，RL也能感知到微弱的梯度指引它转过来！
    # gain_c = M * ((np.cos(theta_c - theta_RA_abs) + 1.0) / 2.0)**2
    # gain_s = M * ((np.cos(theta_s - theta_RA_abs) + 1.0) / 2.0)**2

    # === 修改为 (降低阈值) ===
    # rate1 = log2(1 + (gain_c * lambda1) / bs_dist ** 2)
    # rate2 = log2(1 + (gain_s * lambda2) / target_dist ** 4)
    
    # # 将原本死板的 8.0 阈值降低到 2.0（或更低），让 rho 能够大于 0
    # if rate1 > 0.5:
    #     pho = (rate1 - 0.5) / (rate1 + rate2)
    # else:
    #     pho = 0
    # #--------------------------------2

    # # ✅ 修改后（硬截断，侧向真正为零）
    # gain_c = M * max(0, np.cos(theta_c - theta_RA_abs))**2
    # gain_s = M * max(0, np.cos(theta_s - theta_RA_abs))**2

    # 要求夹角必须小于 60度 (cos(60)=0.5) 才有信号，否则切断！
    cos_c = np.cos(theta_c - theta_RA_abs)
    gain_c = M * (cos_c**2 if cos_c > 0.5 else 0.01)

    cos_s = np.cos(theta_s - theta_RA_abs)
    gain_s = M * (cos_s**2 if cos_s > 0.5 else 0.01)

    rate1 = log2(1 + (gain_c * lambda1) / bs_dist ** 2)
    rate2 = log2(1 + (gain_s * lambda2) / target_dist ** 4)
    if rate1 > 2.0:
        pho = (rate1 - 2.0) / (rate1 + rate2)
    else:
        pho = 0




    # 5.[MODIFIED] 能耗计算包含无人机推进与天线机械旋转能耗
    P_uav = uav_power(omega_uav)
    P_RA = (tau_a ** 2) / eta_a * 0.001 # 转换为 kW
    state_dot[13] = - (P_uav + P_RA) 

    # 6. 累加器变量
    state_dot[16] = 10 * (1 - pho) * rate1          # 通信速率积分
    state_dot[17] = 10 * pho * rate2                # 感知速率积分
    state_dot[18] = (min(state[2] - 60, 0) * 10)    # 高度惩罚积分
    state_dot[19] = pho                             # 时间因子积分

    #-----------------1




    return state_dot


def uavint(state, target_position, control, dt):
    # bs_position = [0, 0, 30]
    bs_position = np.array([0., 2000., 30.])

    t = np.linspace(0, dt, 10)
    # y = odeint(uavfun, np.append(state, [0, 0, 0, 0]), t, args=(control, target_position, bs_position))
    
    #-----------state 扩展到 16 维，附带 4 个累加器 = 20 维---1
    y = odeint(uavfun, np.append(state,[0, 0, 0, 0]), t, args=(control, target_position, bs_position))
    #----------------1
    
    
    tmp = y[-1]
    return tmp[0:-4], tmp[-1], tmp[-2], tmp[-3], tmp[-4]

