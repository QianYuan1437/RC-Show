# 模型预测控制 MPC

::: tip 一句话总结
MPC 每隔一段时间向前“看”一个时域，在模型预测的未来状态上解一个带约束的最优控制问题，只执行第一段，然后重新规划。与只看当前误差的反馈控制相比，它能**预瞄**参考轨迹、提前减速，并把力矩、速度、位置限制作为硬约束。本站的实现是反馈线性化之上的线性 MPC：把机械臂变成双积分器后，每次规划就是一个凸 QP。
:::

## 反馈线性化 + 线性 MPC

计算力矩 $\tau = M(q)u + h(q,\dot q)$ 把每个关节变成 $\ddot q = u$。在这个线性模型上，用步长 $\Delta = 20$ ms、时域 $N = 20$（0.4 s）规划加速度序列 $U = (u_0,\dots,u_{N-1})$：

$$
\begin{aligned}
\min_U\ \ & \sum_{k=1}^{N} w_q\lVert q_k - q^*_k\rVert^2 + w_v\lVert\dot q_k - \dot q^*_k\rVert^2 + w_u\lVert u_{k-1}\rVert^2 \\
\text{s.t.}\ \ & q_k, \dot q_k\ \text{由双积分器从当前状态递推（对 } U \text{ 线性）} \\
& \lvert\dot q_k\rvert \le \dot q_{\max},\quad q_{\min} \le q_k \le q_{\max} \\
& \tau_{\min} \le M_0 u_k + h_0 \le \tau_{\max}
\end{aligned}
$$

- 状态被“消元”（condensed），决策变量只有 $N n$ 个加速度，预测矩阵 $q_k = q_0 + k\Delta\dot q_0 + \sum_j S_{kj}u_j$ 预先算好；
- **预瞄**：$q^*_k$ 是参考轨迹在未来 $t + k\Delta$ 时刻的值（`ref.extra['preview']`），所以 MPC 能在拐点之前提前动作；
- 力矩约束用当前状态的 $M_0, h_0$ 冻结近似，并留 10 % 余量；
- 每 20 ms 重新规划一次，用上次的解热启动；两次规划之间，用计算力矩跟踪规划出的状态轨迹（<Src path="rczoo/control/optimal.py" symbol="JointMPC" />）。

## 与反馈控制的区别

| | 计算力矩 | MPC |
| --- | --- | --- |
| 使用的信息 | 当前误差 | 当前状态 + 未来参考 |
| 约束 | 事后裁剪 | 规划中显式满足 |
| 计算量 | 一次逆动力学 | 每次规划一个 $Nn$ 维 QP（Panda 140 维） |
| 参考不可行时 | 力矩饱和，误差失控 | 自动放慢，保持可行 |

在基准的随机轨迹上（参考本身是可行的），MPC 的跟踪精度与计算力矩相当或略好（Panda 最大 TCP 误差约 0.5 mm vs 0.7 mm），说明预瞄带来了一些收益。负载未知时它与计算力矩一样下垂，因为它依赖同一个模型——MPC 并不自动带来鲁棒性。

## 非线性 MPC 与实践

- 冻结 $M_0, h_0$ 的线性化在 0.4 s 时域内对一般运动足够准；大范围快速运动需要沿预测轨迹重新线性化（实时迭代 RTI），或者直接用非线性 MPC（如 acados、Crocoddyl）；
- 在 1 kHz 的力矩环之上以 50–100 Hz 运行 MPC，是足式机器人和机械臂上常见的分层结构；
- MPC 的计算在主机上调度（是否到了重新规划的时刻），因此本站不能把它捕获进 CUDA graph，基准只用 64 个世界，每步约 10 ms。

## 基准结果

<BenchTable
  track="tracking"
  :conditions="{ nominal: '标称模型', payload: '1.5 kg 未知负载', friction: '关节干摩擦 2 %' }"
  :rows="['ctc', 'pd_ff', 'mpc', 'qp']"
  :columns="[
    { key: 'B', label: '世界数', digits: 0 },
    { key: 'tcp_max_mm', label: '最大 TCP 误差', unit: 'mm', best: 'min' },
    { key: 'tcp_rms_mm', label: 'RMS', unit: 'mm', best: 'min' },
    { key: 'final_mm', label: '结束误差', unit: 'mm', best: 'min' },
    { key: 'energy_J', label: '机械能', unit: 'J', best: 'min' },
  ]"
/>

## 参考文献

- J. B. Rawlings, D. Q. Mayne, M. Diehl, *Model Predictive Control: Theory, Computation, and Design*, 2nd ed., 2017.
- M. Diehl, H. G. Bock, J. P. Schlöder, "A real-time iteration scheme for nonlinear optimization in optimal feedback control," *SIAM J. Control Optim.*, 2005.
