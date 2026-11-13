# iLQR 与 TVLQR

::: tip 一句话总结
五次多项式只保证“平滑”，并不考虑机械臂的动力学。iLQR 在完整的非线性动力学上直接优化力矩序列，在给定时长内找到最省力的运动；再用沿最优轨迹线性化得到的时变 LQR（TVLQR）增益做闭环执行。在同样 1 s 内完成同样的点到点运动，它比五次多项式 + 计算力矩少用约 15–20 % 的 $\int\lVert\tau\rVert^2\,dt$。
:::

## 问题

$$
\begin{aligned}
\min_{u_0,\dots,u_{T-1}}\ \ & \sum_{t=0}^{T-1}\Big(\tfrac{w_u}{2}\lVert u_t\rVert^2 + \tfrac{w_v}{2}\lVert\dot q_t\rVert^2 + \tfrac{w_{\ell}}{2}\big\lVert\max(\lvert\dot q_t\rvert - 0.8\,\dot q_{\max}, 0)\big\rVert^2\Big)\Delta \\
& + \tfrac{w_g}{2}\lVert q_T - q_g\rVert^2 + \tfrac{w_{vg}}{2}\lVert\dot q_T\rVert^2 \\
\text{s.t.}\ \ & x_{t+1} = f(x_t, u_t)\quad(\text{完整刚体动力学，半隐式欧拉，}\Delta = 10\ \text{ms}) \\
& \lvert u_t\rvert \le \tau_{\max}
\end{aligned}
$$

- 终端代价把机械臂“钉”在目标上并要求停止；
- **只惩罚力矩是不够的**：手腕关节惯量小，转动它们几乎不花力矩，没有速度项时“最优解”会让手腕以数倍于限值的速度甩动。速度的小权重 $w_v$ 和超限惩罚 $w_\ell$ 让解保持在关节速度限值的 80 % 以内。

## 算法

iLQR（Li & Todorov 2004）在当前轨迹附近反复做二阶近似：

1. **线性化**：$\delta x_{t+1} = A_t\delta x_t + B_t\delta u_t$，$A_t, B_t$ 由自动微分（`torch.func.jacfwd`）在整条轨迹上批量求得；
2. **反向递推**（Riccati）：从终端代价开始，求每一步的值函数二次近似和最优修正 $\delta u_t = k_t + K_t\delta x_t$；
3. **前向展开 + 线搜索**：$u_t \leftarrow u_t + \alpha k_t + K_t(x_t^{\text{new}} - x_t)$，在完整非线性模型上展开，尝试 $\alpha \in \{1, 0.5, 0.25, 0.1, 0.03\}$，取代价下降的；
4. **力矩约束**：展开时裁剪到 $\pm\tau_{\max}$；反向递推中，已饱和且梯度指向外侧的分量被视为固定（Tassa 等 2014 的 box-DDP 简化版）；
5. Levenberg–Marquardt 式的正则化 $\mu$：成功时减小，失败时增大。

收敛后，在最优轨迹上再做一次 LQR 反向递推，得到用于跟踪的时变增益 $K_t$（<Src path="rczoo/control/optimal.py" symbol="ilqr" />）。

## 执行：两种方式

- **iLQR + TVLQR**：$\tau = u_t^\star + K_t(x - x_t^\star)$，前馈加时变反馈；
- **iLQR + 计算力矩**：把最优轨迹当作参考（加速度分段常数、精确积分得到连续的 $q, \dot q, \ddot q$），交给普通的计算力矩控制器。

两者结果几乎相同。后者说明“最优”主要体现在**轨迹的形状**上：iLQR 学会了借助重力、让大关节少出力、让动作在时间上更均匀地分布，任何好的跟踪控制器都能执行它。

## 基准结果

16 对起止位形（每个关节最多转 0.7 rad），时长都是 1 s，再保持 0.5 s。

<BenchTable
  track="ilqr"
  :rows="['quintic_ctc', 'ilqr_ctc', 'ilqr_tvlqr']"
  row-header="方案"
  :columns="[
    { key: 'tau_sq_motion', label: '运动段 ∫‖τ‖²dt', unit: 'N²m²s', best: 'min' },
    { key: 'peak_tau_rel', label: '峰值力矩', unit: '% 额定', scale: 100, digits: 0 },
    { key: 'energy_J', label: '机械能 ∫|τ·q̇|dt', unit: 'J' },
    { key: 'final_err_deg', label: '终点误差', unit: '°', best: 'min' },
  ]"
/>

<BenchChart
  track="ilqr"
  field="tau_rel"
  reduce="maxabs"
  :controllers="['quintic_ctc', 'ilqr_ctc']"
  y-label="max |τ| / τ_max"
/>

几点观察：

- 节省的是**力矩平方积分**（与电机铜损成正比）；机械功 $\int\lvert\tau\dot q\rvert$ 反而更大，因为 iLQR 的速度曲线更“饱满”、峰值力矩更高——优化什么就得到什么；
- iLQR 在 GPU 上批量求解 16 个问题、60 次迭代约需 2 分钟，适合离线规划；在线使用需要更少的迭代、热启动和更快的动力学导数（解析导数而不是自动微分）。

## 参考文献

- W. Li, E. Todorov, "Iterative linear quadratic regulator design for nonlinear biological movement systems," *ICINCO*, 2004.
- Y. Tassa, N. Mansard, E. Todorov, "Control-limited differential dynamic programming," *ICRA*, 2014.
- R. Tedrake, *Underactuated Robotics*, ch. "Trajectory optimization" and "Linear quadratic regulators", 2023.
