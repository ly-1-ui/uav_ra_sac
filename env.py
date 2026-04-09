# env.py - 强化学习环境文件
# 该文件定义了无人机跟踪目标的仿真环境，包括目标(Target)、无人机(Uav)和环境(Environment)类
# 用于 SAC (Soft Actor-Critic) 算法的训练和测试
#！！！更改有/无旋转：Class UAV def step处修改第五个控制量（tau_a 力矩）

from utils import *  # 导入工具函数
from config import *  # 导入配置文件

import random  # 随机数生成
import copy  # 深拷贝，用于对象复制


# Target 类：表示运动的目标对象
# 作用：模拟一个在二维平面上运动的目标，具有位置、速度和运动轨迹
# class Target(object):
#     # 初始化方法：设置目标的初始状态
#     # 参数：
#     #   position: 初始位置 [x, y]
#     def __init__(self, position):
#         self.position = copy.deepcopy(position)  # 当前位置 [x, y]，深拷贝避免引用问题
#         self.vx = -10  # x方向速度，初始向左移动
#         self.vy = 0  # y方向速度，初始无垂直运动
#         self.vx_min, self.vx_max = -20, 0  # x速度范围：-20到0（只能向左或不动）
#         self.vy_min, self.vy_max = -10, 10  # y速度范围：-10到10
#         self.r = 10  # 轨迹振幅参数
#         # 初始化y位置为正弦波轨迹：y = 3000 + r * sin(0.02 * x)
#         self.position[1] = 2700 + self.r * sin(0.02 * self.position[0])
#         self.init_position = self.position  # 保存初始位置用于重置

#         # 计算速度范围的尺度，用于归一化
#         self.vx_scale = self.vx_max - self.vx_min
#         self.vy_scale = self.vy_max - self.vy_min




#     # step 方法：更新目标位置（随机运动）
#     # 参数：
#     #   dt: 时间步长，默认1.0
#     def step(self, dt=1.):
#         # # 更新x速度：添加高斯噪声，标准差为2
#         # self.vx = np.random.normal(self.vx, 2)
#         # # 限制x速度：不能向右移动（vx > 0），最小为vx_min
#         # if self.vx > 0:
#         #     self.vx = 0
#         # elif self.vx < self.vx_min:
#         #     self.vx = self.vx_min  
#         # # 更新x位置
#         # self.position[0] += self.vx * dt

#         # last_y = self.position[1]  # 保存上一时刻的y位置
#         # # 计算轨迹边界：上边界和下边界
#         # ru = 2410 + self.r * sin(0.02 * self.position[0])  # 上边界
#         # rl = 2390 + self.r * sin(0.02 * self.position[0])  # 下边界
#         # # 更新y位置：基于正弦轨迹 + 高斯噪声
#         # self.position[1] = 2400 + self.r * sin(0.02 * self.position[0]) + np.random.normal(0, 0.5)*dt
#         # # 限制y位置在边界内
#         # if self.position[1] > ru:
#         #     self.position[1] = ru
#         # elif self.position[1] < rl:
#         #     self.position[1] = rl

#         # # 计算y方向速度
#         # self.vy = (self.position[1] - last_y) / dt

#         self.vx = -8
#         self.vy = 0
        
#         self.position[0] += self.vx * dt
#         self.position[1] += self.vy * dt

#     # step_velocity 方法：使用指定速度更新目标位置
#     # 参数：
#     #   vx, vy: 指定的x和y方向速度
#     #   dt: 时间步长，默认1.0
#     def step_velocity(self, vx, vy, dt=1.):
#         self.vx = vx
#         # 限制x速度
#         if self.vx > 0:
#             self.vx = 0
#         elif self.vx < self.vx_min:
#             self.vx = self.vy_min  # 同样是笔误
#         # 更新x位置
#         self.position[0] += self.vx * dt

#         last_y = self.position[1]
#         # 计算轨迹边界
#         ru = 3010 + self.r * sin(0.02 * self.position[0])
#         rl = 2990 + self.r * sin(0.02 * self.position[0])
#         # 更新y位置
#         self.position[1] += vy*dt
#         # 限制边界
#         if self.position[1] > ru:
#             self.position[1] = ru
#         elif self.position[1] < rl:
#             self.position[1] = rl

#         # 计算y速度
#         self.vy = (self.position[1] - last_y) / dt

#     # reset 方法：重置目标到初始状态
#     def reset(self):
#         self.position = copy.deepcopy(self.init_position)
#         self.vx = -8
#         self.vy = 0

#     # reset_init_position 方法：重置初始位置和速度
#     # 参数：
#     #   init_position: 新的初始位置
#     #   velocity: 新的初始速度
#     def reset_init_position(self, init_position, velocity):
#         self.init_position = init_position
#         self.position = copy.deepcopy(init_position)
#         self.vx = velocity

# Target 类：表示运动的目标对象
# 作用：模拟一个在二维平面上运动的目标，当前设定为匀速直线运动
class Target(object):
    def __init__(self, position):
        # 1. 忠实记录外部传入的坐标，绝不强制覆写
        self.position = copy.deepcopy(position)  
        self.init_position = copy.deepcopy(position)  
        
        # 2. 统一初始化直线运动的速度
        self.vx = -10.0  
        self.vy = 0.0  
        
        # 3. 保留归一化需要的物理极限（供神经网络观测空间使用）
        self.vx_min, self.vx_max = -20.0, 0.0  
        self.vy_min, self.vy_max = -10.0, 10.0  
        
        self.vx_scale = self.vx_max - self.vx_min
        self.vy_scale = self.vy_max - self.vy_min

    # step 方法：按设定速度自然运动
    def step(self, dt=1.):
        # 极简纯粹的运动方程
        self.position[0] += self.vx * dt
        self.position[1] += self.vy * dt

    # step_velocity 方法：外界强行改变速度时的更新逻辑
    def step_velocity(self, vx, vy, dt=1.):
        self.vx = vx
        self.vy = vy
        
        # 仅对横向速度作合理的物理限制
        if self.vx > 0:
            self.vx = 0.0
        elif self.vx < self.vx_min:
            self.vx = self.vx_min  
            
        self.position[0] += self.vx * dt
        self.position[1] += self.vy * dt

    # reset 方法：回合结束时重置状态
    def reset(self):
        self.position = copy.deepcopy(self.init_position)
        self.vx = -8.0
        self.vy = 0.0

    # reset_init_position 方法：运行中途彻底改变初始锚点
    def reset_init_position(self, init_position, velocity):
        self.init_position = copy.deepcopy(init_position)
        self.position = copy.deepcopy(init_position)
        self.vx = velocity


# Uav 类：表示无人机智能体
# 作用：模拟无人机的运动、能量消耗、感知和通信能力
class Uav(object):
    # 初始化方法：设置无人机的初始状态和参数
    # 参数：
    #   init_state: 初始状态向量（14维）
    #   time_slot: 时间槽长度
    #   init_energy: 初始能量
    #   end_point: 目标终点位置
    def __init__(self, init_state, time_slot, init_energy, end_point):
        self.time_slot = time_slot  # 时间槽长度
        self.state = copy.deepcopy(init_state)  # 当前状态向量（14维）
        self.init_state = init_state  # 初始状态备份
        self.position = self.state[0:3]  # 当前位置 [x, y, z]
        self.energy = self.state[13]  # 当前能量
        self.flying_time = 0  # 飞行时间累计

        # 位置边界限制
        self.x_min, self.x_max = -200, 500  # x坐标范围
        #--------------扩大y方向边界限制--------2
        # self.y_min, self.y_max = 2500, 3000  # y坐标范围
        self.y_min, self.y_max = 1500, 3100  # y坐标范围
        #-----------------------------------2

        self.z_min, self.z_max = 60, 200  # z坐标范围（高度）
        self.velocity_max = 25  # 最大速度

        self.energy_total = init_energy  # 总能量
        self.end_point = end_point  # 目标终点
        self.distance_end_point = dist(self.position, self.end_point)  # 到终点的距离

        # 计算位置范围的尺度，用于状态归一化
        self.x_scale = self.x_max - self.x_min
        self.y_scale = self.y_max - self.y_min
        self.z_scale = self.z_max - self.z_min

        # 性能指标初始化
        self.com_rate = 0  # 通信速率
        self.sen_rate = 0  # 感知速率
        self.pho = 0  # 功率消耗
        self.sen_com_rate = 0  # 感知+通信速率
        self.sensing_total = 0  # 总感知量
        self.communication_total = 0  # 总通信量

        #-----------修改视频传输速率------------2
        self.video_rate = 2  # 视频传输速率
        #-----------------------------------2

        #--------------添加天线阵列-----1
        self.phi_max = np.pi   # 相对机头最大偏转角 90 度
        self.dot_phi_max = 5.0    # 最大角速度 2 rad/s
        #--------------------------1

        print('---------------------uav built!---------------------------')

    # step 方法：根据控制输入更新无人机状态
    # 参数：
    #   control: 控制输入向量 [4维]
    #   target_positions: 目标位置列表
    #   dt: 时间步长
    # 返回：奖励值
    def step(self, control, target_positions, dt):
        # 将控制输入转换为实际控制量
        u1 = 16 + 10 * control[0]  # 推力或升力
        u2 = 0.7 * control[1]  # 滚转控制
        u3 = 0.7 * control[2]  # 俯仰控制
        u4 = 0.05 * control[3]  # 偏航控制
        # u = [u1, u2, u3, u4]  # 控制向量


        #---------加入第五个控制量----1
        u5 = 0.01 * control[4]     # tau_a 力矩
        # u5=0
        u = [u1, u2, u3, u4,u5]  # 控制向量
        #-----------------------1

        

        # 调用外部函数更新无人机状态
        self.state, pho, reward, sen_rate, com_rate = uavint(self.state, target_positions, u, dt)

        # 更新性能指标
        self.com_rate = com_rate  # 通信速率
        self.sen_rate = sen_rate  # 感知速率
        self.pho = pho  # 功率消耗
        self.sen_com_rate = sen_rate + self.video_rate * dt  # 感知+视频通信速率

        # 根据通信能力更新总感知和通信量
        if com_rate >= self.sen_com_rate:
            self.sensing_total += sen_rate  # 增加感知总量
            self.communication_total += self.sen_com_rate  # 增加通信总量

        # 更新位置和能量状态
        self.position = self.state[0:3]  # 更新位置
        self.energy = self.state[13]  # 更新能量
        self.distance_end_point = dist(self.position, self.end_point)  # 更新到终点距离
        self.flying_time += dt  # 累计飞行时间

        return reward

    # reset 方法：重置无人机到初始状态
    def reset(self):  # reset init_position and target_position.
        self.state = copy.deepcopy(self.init_state)  # 重置状态
        self.energy = copy.deepcopy(self.energy_total)  # 重置能量
        self.position = self.state[0:3]  # 重置位置
        self.flying_time = 0  # 重置飞行时间
        self.distance_end_point = dist(self.position, self.end_point)  # 重置距离
        self.sensing_total = 0  # 重置感知总量
        self.communication_total = 0  # 重置通信总量

    # reset_random_init 方法：随机重置初始状态
    def reset_random_init(self):
        self.state = copy.deepcopy(self.init_state)
        self.energy = random.random() * copy.deepcopy(self.energy_total)  # 随机能量
        # 随机位置
        self.state[0] = self.x_min + random.random() * self.x_scale  # 随机x
        self.state[1] = self.y_min + random.random() * self.y_scale  # 随机y
        self.state[2] = self.z_min + random.random() * self.z_scale  # 随机z
        self.position = self.state[0:3]
        self.flying_time = 0
        self.distance_end_point = dist(self.position, self.end_point)
        self.sensing_total = 0
        self.communication_total = 0

    # reset_random_location 方法：如果位置超出边界则随机重置
    def reset_random_location(self):
        # 检查x边界
        if self.state[0] < map_scale['x_min'] or self.state[0] > map_scale['x_max']:
            self.state[0] = self.x_min + random.random() * self.x_scale

        # 检查y边界
        if self.state[1] < map_scale['y_min'] or self.state[1] > map_scale['y_max']:
            self.state[1] = self.y_min + random.random() * self.y_scale

        # 检查z边界
        if self.state[2] < map_scale['z_min'] or self.state[2] > map_scale['z_max']:
            self.state[2] = self.z_min + random.random() * self.z_scale

        self.position = self.state[0:3]
        self.distance_end_point = dist(self.position, self.end_point)


# Environment 类：强化学习环境
# 作用：管理无人机和目标的交互，计算奖励和状态观测，实现环境接口
class Environment(object):
    # 初始化方法：创建环境实例
    # 参数：
    #   uav_init_state: 无人机初始状态
    #   target_position: 目标初始位置
    #   uav_end_point: 无人机目标终点
    #   init_energy: 初始能量
    #   time_slot: 时间槽长度
    def __init__(self, uav_init_state, target_position, uav_end_point, init_energy, time_slot):
        self.time_slot = time_slot  # time_slot: time_slot_length
        self.time_index = 0  # 当前时间索引
        self.target = Target(target_position)  # 创建目标对象
        self.uav = Uav(uav_init_state, time_slot, init_energy, uav_end_point)  # 创建无人机对象
        self.rewards_step = {'num': 0, 'r_height': 0, 'r_arrive': 0, 'r_bound': 0, 'r_rate': 0, 'r_energy': 0, 'r_all': 0}  # 存储过程奖励

        self.observation_dim = 14 + 3 + 3 + 2 + 1  # 观测维度：无人机状态(14) + 终点相对位置(3) + 目标相对位置(3) + 目标速度(2) + 感知总量(1)
        self.action_dim = 4  # 动作维度：4个控制输入

        print('----------------Environment Built!--------------------')

    # step 方法：执行一个时间步的环境交互
    # 参数：
    #   control: 控制动作 [4维]
    #   dt: 时间步长，默认1.0
    #   target_control: 是否控制目标运动，默认False
    #   vx, vy: 目标速度控制（当target_control=True时使用）
    # 返回：
    #   state_: 下一状态观测
    #   [reward, cost]: 奖励和成本
    #   done: 是否结束
    #   info: 额外信息
    def step(self, control, dt=1., target_control=False, vx=0., vy=0):
        """
        :param control:
        :param dt:
        :return:
        """
        self.time_index += dt  # 更新时间索引
        last_dis = self.uav.distance_end_point  # 上一步到终点的距离
        last_dis_target = dist(self.uav.position, self.target.position)  # 上一步到目标的距离
        last_energy = self.uav.energy  # 上一步能量

        reward = 0.  # 总奖励
        cost = 0.  # 总成本
        height_con = 0  # 高度约束违反标志
        commun_con = 0  # 通信约束违反标志

        # 更新目标位置
        if target_control:
            self.target.step_velocity(vx, vy, dt)  # 使用指定速度
        else:
            self.target.step(dt)  # 随机运动

        # 执行无人机动作并获取奖励
        cost_tmp = -self.uav.step(control, self.target.position, dt)  # 无人机步进（返回的是负奖励，所以取反）
        self.rewards_step['r_height'] = cost_tmp  # 高度相关奖励
        cost += cost_tmp  # 累加成本

        # 计算当前位置到终点和目标的距离
        dis = dist(self.uav.position, self.uav.end_point)  # 到终点距离
        dis_target = dist(self.uav.position, self.target.position)  # 到目标距离

        # # 判断是否到达终点（距离小于20）
        # if dis <= 20.:
        #     arrive_flag = 1  # 已到达
        # else:
        #     arrive_flag = 0  # 未到达

        # # 根据能量和到达状态计算奖励
        # if self.uav.energy <= 0:
        #     self.uav.energy = 0.  # 能量清零
        #     done = 1  #  episode结束
        #     reward_tmp = 0 
        #     dis_cost = dis * 5  # 坠毁重罚
        #     arrive_flag = 0

        # # 2. 提前到达终点，立刻通关
        # elif dis <= 20.0:
        #     arrive_flag = 1
        #     done = 1  # 到达立刻结束回合
        #     dis_cost = 0
            
        #     # 巨额通关奖励 + 剩余电量折算奖金+ 累计感知奖励
        #     # 剩余能量越多，奖励越大
        #     energy_bonus = self.uav.energy * 20.0  
        #     reward_tmp = 1000.0 + energy_bonus + self.uav.sensing_total*5
            
        # # 3. 还在飞行途中
        # else:
        #     arrive_flag = 0
        #     done = 0
        #     reward_tmp = 0
        #     dis_cost = 0

        # cost += dis_cost  # 累加距离成本
        # self.rewards_step['r_arrive'] = reward_tmp  # 到达奖励
        # self.rewards_step['c_arrive'] = dis  # 到达成本
        # reward += reward_tmp  # 累加奖励

        # # 检查高度约束
        # if self.uav.position[2] < 60:
        #     height_con = 1  # 高度过低

        # # 检查边界约束
        # if self.uav.position[0] > 600 or self.uav.position[0] < -300 or \
        #         self.uav.position[1] > 3100 or self.uav.position[1] < 2500 or \
        #         self.uav.position[2] > 200 or self.uav.position[2] < 30:
        #     done = 1  # 超出边界，结束
        #     #--------修改越界惩罚------4
        #     reward_tmp = -1000  # 禁止撞墙
        #     cost += 5*dis  # 禁止坠毁
        # else:
        #     reward_tmp = dt  # 保持在边界内给予时间奖励
        # self.rewards_step['r_bound'] = reward_tmp  # 边界奖励
        # reward += reward_tmp  # 累加奖励

        # # 检查通信约束
        # if self.uav.video_rate*dt > self.uav.com_rate:
        #     cost_tmp = (self.uav.video_rate * dt - self.uav.com_rate)  # 通信不足的成本
        #     reward_tmp = 0.  # 无奖励
        #     commun_con = 1  # 通信约束违反
        # else:
        #     cost_tmp = 0.  # 无成本
        #     reward_tmp = self.uav.sen_rate  # 感知速率作为奖励
        # self.rewards_step['r_rate'] = reward_tmp  # 速率奖励
        # self.rewards_step['c_rate'] = cost_tmp  # 速率成本
        # reward += reward_tmp  # 累加奖励
        # cost += cost_tmp  # 累加成本

        # #--------检查天线旋转机械约束----1
        # phi_current = self.uav.state[14]
        # if abs(phi_current) > self.uav.phi_max:
        #     cost_tmp = abs(phi_current) - self.uav.phi_max
        #     reward_tmp = 0.
        # else:
        #     cost_tmp = 0.
        #     reward_tmp = dt * 0.1 # 安全范围内给个极小奖励
            
        # self.rewards_step['r_phi'] = reward_tmp
        # self.rewards_step['c_phi'] = cost_tmp
        # reward += reward_tmp
        # cost += cost_tmp
        # #----------------------------1

        # # 能量相关奖励
        # if self.uav.energy < 15:
        #     reward_tmp = max(last_dis - dis, 0)  # 能量低时，接近终点给予奖励
        # else:
        #     reward_tmp = max(last_dis_target - dis_target, 0)  # 能量充足时，接近目标给予奖励
        # self.rewards_step['r_energy'] = reward_tmp  # 能量奖励
        # reward += reward_tmp  # 累加奖励

        # # 记录总奖励和成本
        # self.rewards_step['r_all'] = reward
        # self.rewards_step['c_all'] = cost

        # self.uav.distance_end_point = dis  # 更新无人机到终点距离

        # state_ = self.observation_state()  # 获取观测状态

        # info = {'cost': cost, 'arrive_flag': arrive_flag,
        #         'height_con': height_con, 'commun_con': commun_con}  # 返回额外信息
        # return state_, [reward, cost], done, info

        # =====================================================================
        # 1. 终局判定与通关奖励 (Terminal Conditions & Rewards)
        # =====================================================================
        if self.uav.energy <= 0:
            # 没电坠毁
            self.uav.energy = 0.  
            done = 1  
            reward_tmp = 0 
            dis_cost = dis * 5.0  # 距离终点越远，坠毁惩罚越大
            arrive_flag = 0

        elif dis <= 20.0:
            # 成功抵达终点！
            arrive_flag = 1
            done = 1  # 抵达即立刻结束！
            dis_cost = 0
            
            # 【重塑价值观】：大幅削弱剩余电量奖金，暴增感知累计奖金！
            # 让它明白：当个满电的快递员不值钱，带回海量的感知数据才是王者。
            energy_bonus = self.uav.energy * 2.0         # 剩余电量奖金（倍率降到2）
            sensing_bonus = self.uav.sensing_total * 30.0 # 感知总数据奖金（倍率飙到20）
            reward_tmp = 1000.0 + energy_bonus +sensing_bonus
            
        else:
            # 还在安全飞行中
            arrive_flag = 0
            done = 0
            reward_tmp = 0
            dis_cost = 0

        cost += dis_cost  
        self.rewards_step['r_arrive'] = reward_tmp  
        self.rewards_step['c_arrive'] = dis_cost  
        reward += reward_tmp  

        # =====================================================================
        # 2. 物理与安全约束惩罚 (Physical Constraints & Penalties)
        # =====================================================================
        # 检查高度
        if self.uav.position[2] < 60:
            height_con = 1  

        # 检查越界 (如果越界，直接坠毁重罚)
        if self.uav.position[0] > 600 or self.uav.position[0] < -300 or \
                self.uav.position[1] > 3100 or self.uav.position[1] < 1500 or \
                self.uav.position[2] > 200 or self.uav.position[2] < 30:
            done = 1  
            reward_tmp = -1000.0  # 撞墙直接给暴击惩罚
            cost += dis * 5.0     
        else:
            reward_tmp = dt  # 存活一秒给一点基础奖励
        self.rewards_step['r_bound'] = reward_tmp  
        reward += reward_tmp  

        # 检查通信底线 (门槛2.0与utils.py的rho门槛对齐)
        if 2.0 * dt > self.uav.com_rate:
            cost_tmp = (2.0 * dt - self.uav.com_rate) * 10.0  # 通信断连惩罚
            reward_tmp = 0.
            commun_con = 1
        else:
            cost_tmp = 0.
            # 【核心修改】感知奖励必须乘以实时波束对准质量，没对准=没奖励
            q0, q1, q2, q3 = self.uav.state[9:13]
            theta_body = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))
            theta_RA = theta_body + self.uav.state[14]
            pos = self.uav.position
            theta_s_now = np.arctan2(
                self.target.position[1] - pos[1],
                self.target.position[0] - pos[0])
            align_quality = max(0, np.cos(theta_s_now - theta_RA))**2  # 硬截断对准质量[0,1]
            reward_tmp = self.uav.sen_rate * 2.0 * align_quality  # 对准才有感知奖励
            reward_tmp += align_quality * 3.0 * dt                # 额外对准鼓励
        self.rewards_step['r_rate'] = reward_tmp
        self.rewards_step['c_rate'] = cost_tmp
        reward += reward_tmp
        cost += cost_tmp  

        # 检查天线机械限位 (超限受罚)
        phi_current = self.uav.state[14]
        if abs(phi_current) > self.uav.phi_max:
            cost_tmp = (abs(phi_current) - self.uav.phi_max) * 5.0
            reward_tmp = 0.
        else:
            cost_tmp = 0.
            reward_tmp = dt * 0.1 
            
        self.rewards_step['r_phi'] = reward_tmp
        self.rewards_step['c_phi'] = cost_tmp
        reward += reward_tmp
        cost += cost_tmp

        # =====================================================================
        # 3. 【核心新增】两阶段行为引导 (Two-Phase Tracking & Returning)
        # =====================================================================
        # 假设初始总能量为 40 左右。设定电量告急阈值为 15.0
        # Phase 1: 电量充足时（>15），忽略终点，死死咬住目标！
        if self.uav.energy > 15.0:
            # (last_dis_target - dis_target) 为正代表正在靠近目标
            # 乘以 3.0 的强烈系数，逼迫它向目标飞去
            reward_guide = (last_dis_target - dis_target) * 3.0
            
        # Phase 2: 电量告急时（<=15），放弃目标，全速向终点返航！
        else:
            # (last_dis - dis) 为正代表正在靠近终点
            # 乘以 5.0 的极强系数，保命要紧，逼迫它赶紧回家
            reward_guide = (last_dis - dis) * 5.0
            
        self.rewards_step['r_energy'] = reward_guide  
        reward += reward_guide  

        # =====================================================================
        # 4. 【核心新增】机身平滑飞行惩罚 (Smoothness Penalty)
        # =====================================================================
        # 提取机身的滚转(p)、俯仰(q)、偏航(r)角速度
        p, q, r_yaw = self.uav.state[6], self.uav.state[7], self.uav.state[8]
        angular_velocity_penalty = abs(p) + abs(q) + abs(r_yaw)
        
        # 施加平滑惩罚：机身扭动越剧烈，扣分越狠！
        # 这将完美凸显 RA-ULA (只需转天线，机身平稳) 相比 Fixed-ULA (机身疯狂扭动) 的降维打击优势！
        r_smooth = -4 * angular_velocity_penalty
        reward += r_smooth

        self.rewards_step['r_all'] = reward
        self.rewards_step['c_all'] = cost
        self.uav.distance_end_point = dis  

        state_ = self.observation_state()  
        info = {'cost': cost, 'arrive_flag': arrive_flag, 'height_con': height_con, 'commun_con': commun_con}  
        return state_,[reward, cost], done, info

    # reset 方法：重置环境到初始状态
    def reset(self):
        self.time_index = 0  # 重置时间
        self.uav.reset()  # 重置无人机
        self.target.reset()  # 重置目标

        state_ = self.observation_state()  # 获取初始观测
        return state_

    # reset_random_location 方法：随机重置无人机位置
    def reset_random_location(self):
        self.uav.reset_random_location()  # 随机重置无人机位置

    # reset_random_init 方法：完全随机重置环境
    def reset_random_init(self):
        self.time_index = 0  # 重置时间
        self.uav.reset_random_init()  # 随机重置无人机
        self.target.reset()  # 重置目标
        state_ = self.observation_state()  # 获取观测
        return state_

    # observation_state 方法：生成状态观测向量
    # 返回：归一化的观测向量，包含无人机状态、相对位置、目标信息等
    def observation_state(self):
        observation = []  # 初始化观测列表

        # 无人机位置（归一化）
        observation = np.append(observation, (self.uav.position[0] - self.uav.x_min) / self.uav.x_scale)  # x位置
        observation = np.append(observation, (self.uav.position[1] - self.uav.y_min) / self.uav.y_scale)  # y位置
        observation = np.append(observation, (self.uav.position[2] - self.uav.z_min) / self.uav.z_scale)  # z位置（高度）

        # 无人机速度（归一化）
        observation = np.append(observation, self.uav.state[3:6] / self.uav.velocity_max)  # vx, vy, vz

        # 无人机其他状态（角度、角速度等）
        observation = np.append(observation, self.uav.state[6:13])  # phi, theta, psi, p, q, r, dot_phi等

        # 无人机能量（归一化）
        observation = np.append(observation, self.uav.energy / self.uav.energy_total)

        # 相对终点位置（归一化）
        observation = np.append(observation,
                                (self.uav.end_point[0] - self.uav.position[0]) / self.uav.x_scale)  # 相对x
        observation = np.append(observation,
                                (self.uav.end_point[1] - self.uav.position[1]) / self.uav.y_scale)  # 相对y
        observation = np.append(observation,
                                (self.uav.end_point[2] - self.uav.position[2]) / self.uav.z_scale)  # 相对z

        # 相对目标位置（归一化）
        observation = np.append(observation,
                                (self.target.position[0] - self.uav.position[0]) / self.uav.x_scale)  # 相对x
        observation = np.append(observation,
                                (self.target.position[1] - self.uav.position[1]) / self.uav.y_scale)  # 相对y
        observation = np.append(observation,
                                (self.target.position[2] - self.uav.position[2]) / self.uav.z_scale)  # 相对z

        # 目标速度（归一化）
        observation = np.append(observation,
                                (self.target.vx - self.target.vx_min) / self.target.vx_scale)  # vx
        observation = np.append(observation,
                                (self.target.vy - self.target.vy_min) / self.target.vy_scale)  # vy

        #-----天线部分----1
        # [NEW] 1. 天线机械状态归一化
        observation = np.append(observation, self.uav.state[14] / self.uav.phi_max)
        observation = np.append(observation, self.uav.state[15] / self.uav.dot_phi_max)

        # [NEW] 2. 提供机体与目标的角度关系特征，帮助网络快速掌握波束对准诀窍
        pos = self.uav.position
        q0, q1, q2, q3 = self.uav.state[9:13]
        theta_body = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))
        theta_RA = theta_body + self.uav.state[14]
        
        theta_c = np.arctan2(self.uav.end_point[1] - pos[1], self.uav.end_point[0] - pos[0])
        theta_s = np.arctan2(self.target.position[1] - pos[1], self.target.position[0] - pos[0])
        
        # 将相对波束对准误差（余弦值）输入网络，1为完全对准，-1为背对
        observation = np.append(observation, np.cos(theta_c - theta_RA))
        observation = np.append(observation, np.cos(theta_s - theta_RA))
        #-----------------1

        # 感知总量（归一化）
        observation = np.append(observation, (self.uav.sensing_total / 1000.))

        return observation

