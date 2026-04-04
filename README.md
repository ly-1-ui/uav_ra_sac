## 项目简介
本项目提出了一种基于受限软演员-评论家算法（Constrained SAC, C-SAC）的无人机通感算控一体化（ISAC）联合控制框架。

项目的目标在于：为真实的 6自由度（6-DoF）刚体无人机引入了可旋转天线阵列（Rotatable Antenna, RA-ULA）。通过强化学习，无人机需实现“飞行轨迹”与“电磁波束指向”的解耦，打破了传统固定天线无人机在“能量消耗”与“感知性能”之间的物理悖论。

### 安装
```bash
# 克隆仓库
git clone https://github.com/ly-1-ui/uav_ra_sac.git
cd uav_ra_sac

# 创建并激活环境
conda env create -f environment.yml
conda activate rl
```
### 运行训练
```bash
python sac_train_qc.py  
```

### 绘图
```bash
python plot_training_results.py
```