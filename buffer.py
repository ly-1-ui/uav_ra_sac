import os
import numpy as np
import numpy.random as rd
import torch
from torch import Tensor

# ReplayBuffer 类：用于离线策略强化学习算法的经验回放缓冲区
# 作用：存储和采样经验数据（状态、动作、奖励、下一状态、掩码），支持优先经验回放（PER）
class ReplayBuffer:  # for off-policy
    # 初始化方法：设置缓冲区的容量、维度和设备
    # 参数：
    #   max_capacity: 缓冲区最大容量
    #   state_dim: 状态维度（整数或元组）
    #   action_dim: 动作维度
    #   gpu_id: GPU ID，-1表示使用CPU
    #   if_use_per: 是否使用优先经验回放
    def __init__(self, max_capacity: int, state_dim: int, action_dim: int, gpu_id=0, if_use_per=False):
        self.prev_p = 0  # previous pointer：上一个指针，用于轨迹拼接
        self.next_p = 0  # next pointer：下一个指针，指向下一个存储位置
        self.if_full = False  # 缓冲区是否已满
        self.cur_capacity = 0  # current capacity：当前容量
        self.max_capacity = max_capacity  # 最大容量
        self.add_capacity = 0  # update in self.update_buffer：本次更新的数据量

        # 设置设备：优先使用GPU，否则使用CPU
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        # 初始化缓冲区张量：动作、奖励、掩码、状态
        self.buf_action = torch.empty((max_capacity, action_dim), dtype=torch.float32, device=self.device)
        self.buf_reward = torch.empty((max_capacity, 1), dtype=torch.float32, device=self.device)
        self.buf_mask = torch.empty((max_capacity, 1), dtype=torch.float32, device=self.device)

        # 状态缓冲区：根据state_dim决定形状
        buf_state_size = (max_capacity, state_dim) if isinstance(state_dim, int) else (max_capacity, *state_dim)
        self.buf_state = torch.empty(buf_state_size, dtype=torch.float32, device=self.device)

        # 优先经验回放相关
        self.if_use_per = if_use_per
        if if_use_per:
            self.per_tree = BinarySearchTree(max_capacity)  # 二叉搜索树用于PER
            self.sample_batch = self.sample_batch_per  # 切换采样方法为PER版本

    # update_buffer 方法：向缓冲区添加新的经验数据
    # 作用：将轨迹数据（状态、动作、奖励、掩码）存储到缓冲区中，支持循环覆盖
    # 参数：
    #   states: 状态张量
    #   actions: 动作张量
    #   rewards: 奖励张量
    #   masks: 掩码张量（用于终止状态）
    def update_buffer(self, states, actions, rewards, masks):
        # traj_items = [map(list, zip(*traj_list))]
        #
        # states, rewards, masks, actions = [torch.cat(item, dim=0) for item in traj_items]

        self.add_capacity = rewards.shape[0]  # 本次添加的数据量
        p = self.next_p + self.add_capacity  # 计算更新后的指针位置

        # 如果使用PER，更新优先级树
        if self.if_use_per:
            self.per_tree.update_ids(data_ids=np.arange(self.next_p, p) % self.max_capacity, prob=rewards.cpu().detach().numpy().flatten())

        # 处理缓冲区溢出：如果新数据超过剩余空间，先填充剩余空间，再从头开始覆盖
        if p > self.max_capacity:
            self.buf_state[self.next_p:self.max_capacity] = states[:self.max_capacity - self.next_p]
            self.buf_reward[self.next_p:self.max_capacity] = rewards[:self.max_capacity - self.next_p]
            self.buf_mask[self.next_p:self.max_capacity] = masks[:self.max_capacity - self.next_p]
            self.buf_action[self.next_p:self.max_capacity] = actions[:self.max_capacity - self.next_p]
            self.if_full = True  # 标记缓冲区已满

            p = p - self.max_capacity  # 计算剩余数据的位置
            self.buf_state[0:p] = states[-p:]
            self.buf_reward[0:p] = rewards[-p:]
            self.buf_mask[0:p] = masks[-p:]
            self.buf_action[0:p] = actions[-p:]
        else:
            # 直接存储数据
            self.buf_state[self.next_p:p] = states
            self.buf_reward[self.next_p:p] = rewards
            self.buf_mask[self.next_p:p] = masks
            self.buf_action[self.next_p:p] = actions

        self.next_p = p  # 更新下一个指针
        self.cur_capacity = self.max_capacity if self.if_full else self.next_p  # 更新当前容量

        # steps = rewards.shape[0]
        # r_exp = rewards.mean().item()
        # return steps, r_exp

    # sample_batch 方法：从缓冲区中随机采样一批经验数据（不使用PER）
    # 作用：用于训练时随机选择经验，支持最新的经验优先采样
    # 参数：
    #   batch_size: 采样批次大小
    # 返回：(状态, 动作, 奖励, 下一状态, 掩码)
    def sample_batch(self, batch_size: int) -> (Tensor, Tensor, Tensor, Tensor):
        indices = torch.randint(self.cur_capacity - 1, size=(batch_size,), device=self.device)  # 随机采样索引

        '''replace indices using the latest sample'''  # 用最新的样本替换部分索引
        i1 = self.next_p
        i0 = self.next_p - self.add_capacity
        num_new_indices = 1  # 替换的数量
        new_indices = torch.randint(i0, i1, size=(num_new_indices,)) % (self.max_capacity - 1)
        indices[0:num_new_indices] = new_indices

        return (
            self.buf_state[indices],  # 当前状态
            self.buf_action[indices],  # 动作
            self.buf_reward[indices],  # 奖励
            self.buf_state[indices + 1],  # 下一状态
            self.buf_mask[indices],  # 掩码
        )

    # sample_batch_per 方法：使用优先经验回放（PER）采样一批经验数据
    # 作用：根据优先级采样经验，更高优先级的经验被采样概率更大
    # 参数：
    #   batch_size: 采样批次大小
    # 返回：(状态, 动作, 奖励, 下一状态, 掩码)
    def sample_batch_per(self, batch_size: int) -> (Tensor, Tensor, Tensor, Tensor, Tensor):
        beg = -self.max_capacity  # 开始索引
        end = (self.cur_capacity - self.max_capacity) if (self.cur_capacity < self.max_capacity) else None  # 结束索引

        indices, is_weights = self.per_tree.get_indices_is_weights(batch_size, beg, end)  # 获取索引和重要性权重

        return (
            self.buf_state[indices],  # 当前状态
            self.buf_action[indices],  # 动作
            self.buf_reward[indices],  # 奖励
            self.buf_state[indices + 1],  # 下一状态
            self.buf_mask[indices]  # 掩码
            # torch.as_tensor(is_weights, dtype=torch.float32, device=self.device)  # important sampling weights
        )

    # td_error_update 方法：更新时间差分误差，用于PER优先级更新
    # 作用：根据TD误差调整经验的优先级
    # 参数：
    #   td_error: 时间差分误差张量
    def td_error_update(self, td_error: Tensor):
        self.per_tree.td_error_update(td_error)

    # save_or_load_history 方法：保存或加载缓冲区历史数据
    # 作用：持久化存储缓冲区内容，或从文件加载
    # 参数：
    #   cwd: 当前工作目录
    #   if_save: True表示保存，False表示加载
    def save_or_load_history(self, cwd: str, if_save: bool):
        obj_names = (
            (self.buf_reward, "reward"),  # 奖励缓冲区
            (self.buf_mask, "mask"),  # 掩码缓冲区
            (self.buf_action, "action"),  # 动作缓冲区
            (self.buf_state, "state"),  # 状态缓冲区
        )

        if if_save:
            print(f"| {self.__class__.__name__}: Saving in cwd {cwd}")
            for obj, name in obj_names:
                if self.cur_capacity == self.next_p:
                    buf_tensor = obj[:self.cur_capacity]  # 连续存储
                else:
                    buf_tensor = torch.vstack((obj[self.next_p:self.cur_capacity], obj[0:self.next_p]))  # 环形缓冲区拼接

                torch.save(buf_tensor, f"{cwd}/replay_buffer_{name}.pt")  # 保存为PyTorch张量文件

            print(f"| {self.__class__.__name__}: Saved in cwd {cwd}")

        elif os.path.isfile(f"{cwd}/replay_buffer_state.pt"):
            print(f"| {self.__class__.__name__}: Loading from cwd {cwd}")
            buf_capacity = 0
            for obj, name in obj_names:
                buf_tensor = torch.load(f"{cwd}/replay_buffer_{name}.pt")  # 加载张量
                buf_capacity = buf_tensor.shape[0]

                obj[:buf_capacity] = buf_tensor  # 恢复到缓冲区
            self.cur_capacity = buf_capacity

            print(f"| {self.__class__.__name__}: Loaded from cwd {cwd}")

    # concatenate_state 方法：拼接并返回自上次调用以来新增的状态
    # 作用：用于轨迹拼接，获取从prev_p到next_p之间的状态序列
    # 返回：状态张量
    def concatenate_state(self) -> Tensor:
        if self.prev_p <= self.next_p:
            buf_state = self.buf_state[self.prev_p:self.next_p]  # 连续段
        else:
            buf_state = torch.vstack((self.buf_state[self.prev_p:], self.buf_state[:self.next_p],))  # 环形拼接
        self.prev_p = self.next_p  # 更新prev_p
        return buf_state

    # concatenate_buffer 方法：拼接并返回自上次调用以来新增的完整缓冲区数据
    # 作用：获取从prev_p到next_p之间的完整经验数据（状态、动作、奖励、掩码）
    # 返回：(状态, 动作, 奖励, 掩码)
    def concatenate_buffer(self) -> (Tensor, Tensor, Tensor, Tensor):
        if self.prev_p <= self.next_p:
            buf_state = self.buf_state[self.prev_p:self.next_p]
            buf_action = self.buf_action[self.prev_p:self.next_p]
            buf_reward = self.buf_reward[self.prev_p:self.next_p]
            buf_mask = self.buf_mask[self.prev_p:self.next_p]
        else:
            buf_state = torch.vstack((self.buf_state[self.prev_p:], self.buf_state[:self.next_p],))
            buf_action = torch.vstack((self.buf_action[self.prev_p:], self.buf_action[:self.next_p],))
            buf_reward = torch.vstack((self.buf_reward[self.prev_p:], self.buf_reward[:self.next_p],))
            buf_mask = torch.vstack((self.buf_mask[self.prev_p:], self.buf_mask[:self.next_p],))
        self.prev_p = self.next_p  # 更新prev_p
        return buf_state, buf_action, buf_reward, buf_mask


# BinarySearchTree 类：用于优先经验回放（PER）的二叉搜索树
# 作用：维护经验的优先级，实现高效的优先级采样
# 参考：https://github.com/kaixindelele/DRLib/tree/main/algos/pytorch/td3_sp
# 参考：https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
class BinarySearchTree:
    """Binary Search Tree for PER
    Contributor: Github GyChou, Github mississippiu
    Reference: https://github.com/kaixindelele/DRLib/tree/main/algos/pytorch/td3_sp
    Reference: https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
    """
    # 初始化方法：设置树的容量和PER参数
    # 参数：
    #   memo_len: 记忆长度（缓冲区大小）
    def __init__(self, memo_len):
        self.memo_len = memo_len  # replay buffer len：回放缓冲区长度
        self.prob_ary = np.zeros((memo_len - 1) + memo_len)  # parent_nodes_num + leaf_nodes_num：概率数组（父节点+叶节点）
        self.max_capacity = len(self.prob_ary)  # 最大容量
        self.cur_capacity = self.memo_len - 1  # pointer：当前指针
        self.indices = None  # 采样索引
        self.depth = int(np.log2(self.max_capacity))  # 树深度

        # PER参数：Prioritized Experience Replay
        # alpha, beta = 0.7, 0.5 for rank-based variant
        # alpha, beta = 0.6, 0.4 for proportional variant
        self.per_alpha = 0.6  # alpha = (Uniform:0, Greedy:1)：优先级指数
        self.per_beta = 0.4  # beta = (PER:0, NotPER:1)：重要性采样指数

    # update_id 方法：更新单个数据的优先级
    # 作用：修改指定数据的优先级，并向上传播更新父节点
    # 参数：
    #   data_id: 数据ID
    #   prob: 优先级概率（默认10为最大概率）
    def update_id(self, data_id, prob=10):  # 10 is max_prob
        tree_id = data_id + self.memo_len - 1  # 计算树中的索引
        if self.cur_capacity == tree_id:
            self.cur_capacity += 1

        delta = prob - self.prob_ary[tree_id]  # 计算变化量
        self.prob_ary[tree_id] = prob  # 更新叶节点

        while tree_id != 0:  # propagate the change through tree：向上传播变化
            tree_id = (tree_id - 1) // 2  # faster than the recursive loop：计算父节点索引
            self.prob_ary[tree_id] += delta  # 更新父节点

    # update_ids 方法：批量更新多个数据的优先级
    # 作用：高效地更新多个叶节点的优先级，并向上传播
    # 参数：
    #   data_ids: 数据ID数组
    #   prob: 优先级概率（默认10）
    def update_ids(self, data_ids, prob=10):  # 10 is max_prob
        ids = data_ids + self.memo_len - 1  # 计算树索引
        self.cur_capacity += (ids >= self.cur_capacity).sum()  # 更新当前容量

        upper_step = self.depth - 1  # 上层步骤数
        self.prob_ary[ids] = prob  # here, ids means the indices of given children (maybe the right ones or left ones)：设置叶节点概率
        p_ids = (ids - 1) // 2  # 父节点索引

        while upper_step:  # propagate the change through tree：向上传播
            ids = p_ids * 2 + 1  # in this while loop, ids means the indices of the left children：左子节点索引
            self.prob_ary[p_ids] = self.prob_ary[ids] + self.prob_ary[ids + 1]  # 更新父节点为左右子节点之和
            p_ids = (p_ids - 1) // 2  # 继续向上
            upper_step -= 1

        self.prob_ary[0] = self.prob_ary[1] + self.prob_ary[2]  # because we take depth-1 upper steps, ps_tree[0] need to be updated alone：单独更新根节点

    # get_leaf_id 方法：根据值v找到对应的叶节点索引
    # 作用：通过二分搜索找到累积概率对应的叶节点，用于优先级采样
    # 参数：
    #   v: 随机值（0到总概率之间）
    # 返回：叶节点索引
    def get_leaf_id(self, v):
        """Tree structure and array storage:
        Tree index:
              0       -> storing priority sum
            |  |
          1     2
         | |   | |
        3  4  5  6    -> storing priority for transitions
        Array type for storing: [0, 1, 2, 3, 4, 5, 6]
        """
        parent_idx = 0  # 从根节点开始
        while True:
            l_idx = 2 * parent_idx + 1  # the leaf's left node：左子节点
            r_idx = l_idx + 1  # the leaf's right node：右子节点
            if l_idx >= (len(self.prob_ary)):  # reach bottom, end search：到达底部，结束搜索
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node：向下搜索，总是选择更高优先级的节点
                if v <= self.prob_ary[l_idx]:
                    parent_idx = l_idx  # 选择左子树
                else:
                    v -= self.prob_ary[l_idx]  # 减去左子树概率
                    parent_idx = r_idx  # 选择右子树
        return min(leaf_idx, self.cur_capacity - 2)  # leaf_idx：返回叶节点索引，确保不超过当前容量

    # get_indices_is_weights 方法：获取采样索引和重要性权重
    # 作用：根据优先级进行采样，返回索引和重要性采样权重
    # 参数：
    #   batch_size: 批次大小
    #   start: 开始索引
    #   end: 结束索引
    # 返回：(索引数组, 重要性权重数组)
    def get_indices_is_weights(self, batch_size, start, end):
        self.per_beta = min(1., self.per_beta + 0.001)  # 逐渐增加beta到1

        # get random values for searching indices with proportional prioritization：生成随机值用于比例优先级采样
        values = (rd.rand(batch_size) + np.arange(batch_size)) * (self.prob_ary[0] / batch_size)

        # get proportional prioritization：获取比例优先级
        leaf_ids = np.array([self.get_leaf_id(v) for v in values])  # 获取叶节点ID
        self.indices = leaf_ids - (self.memo_len - 1)  # 转换为缓冲区索引

        prob_ary = self.prob_ary[leaf_ids] / self.prob_ary[start:end].min()  # 归一化概率
        is_weights = np.power(prob_ary, -self.per_beta)  # important sampling weights：重要性采样权重
        return self.indices, is_weights

    # td_error_update 方法：根据时间差分误差更新优先级
    # 作用：使用TD误差计算新的优先级，并更新树
    # 参数：
    #   td_error: 时间差分误差张量 (q-q).detach_().abs()
    def td_error_update(self, td_error):  # td_error = (q-q).detach_().abs()
        prob = td_error.squeeze().clamp(1e-6, 10).pow(self.per_alpha)  # 计算优先级：TD误差^alpha，限制在1e-6到10之间
        prob = prob.cpu().numpy()  # 转换为numpy数组
        self.update_ids(self.indices, prob)  # 更新优先级
