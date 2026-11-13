# QP 任务空间控制

::: tip 一句话总结
把“我想要的末端加速度”写成二次型代价，把电机力矩、关节速度和位置限位写成线性约束，每个控制周期解一个小型二次规划（QP）。约束不起作用时它就是操作空间控制；约束起作用时，它给出满足所有限制的“最好的”加速度，而不是先算力矩再裁剪。这是人形机器人全身控制（whole-body control）的基本形式。
:::

## 问题形式

以关节加速度 $\ddot q$ 为决策变量（<Src path="rczoo/control/optimal.py" symbol="QPTaskSpace" />）：

$$
\begin{aligned}
\min_{\ddot q}\ \ & \tfrac12\big\lVert J\ddot q + \dot J\dot q - \ddot x^\star\big\rVert^2 + \tfrac{w_n}{2}\big\lVert\ddot q - \ddot q_{\text{null}}\big\rVert^2 \\
\text{s.t.}\ \ & \tau_{\min} \le M\ddot q + h \le \tau_{\max} \\
& \ddot q_{\min}(q,\dot q) \le \ddot q \le \ddot q_{\max}(q,\dot q)
\end{aligned}
$$

- $\ddot x^\star = \ddot x_d + K_p e + K_d(\dot x_d - \dot x)$ 是 OSC 的期望末端加速度；
- 第二项是零空间姿态任务，权重 $w_n = 10^{-2}$ 很小（软优先级），对 6 自由度的 SR3-C 还起正则化作用；
- 加速度的盒约束由**速度阻尼器**构造：保证在 $t_v = 50$ ms 内能把速度拉回限值以内、在 $t_p = 150$ ms 内不越过位置限位；
- 解出 $\ddot q^\star$ 后，$\tau = M\ddot q^\star + h$。

## 批量 ADMM 求解器

每个控制周期、每个世界都要解一个 7 变量、14 约束的 QP。本站在 GPU 上批量运行 OSQP 的 ADMM 迭代（<Src path="rczoo/qp.py" symbol="solve_qp" />）：

$$
\begin{aligned}
x^{k+1} &= (P + \sigma I + \rho A^\top A)^{-1}\big(\sigma x^k - c + A^\top(\rho z^k - y^k)\big) \\
z^{k+1} &= \Pi_{[l,u]}\big(\alpha A x^{k+1} + (1-\alpha)z^k + y^k/\rho\big) \\
y^{k+1} &= y^k + \rho\big(\alpha A x^{k+1} + (1-\alpha)z^k - z^{k+1}\big)
\end{aligned}
$$

- 矩阵 $K = P + \sigma I + \rho A^\top A$ 在一次求解中不变，求逆一次，之后每次迭代只有矩阵-向量乘法；
- 固定 30 次迭代，用上一周期的解热启动，耗时可预测；
- 全程没有主机同步，可以和控制器其它部分一起被捕获进 CUDA graph。测试中与 SciPy SLSQP 的解对照（<Src path="tests/test_qp.py" />）。

## 什么时候它比 OSC 好

基准里的随机轨迹几乎不触碰约束，QP 与 OSC 的误差几乎相同（QP 稍大一点，来自固定迭代次数的不精确解）。它的价值在约束起作用时：

- OSC 算出超限的力矩后被电机**逐关节裁剪**，裁剪后的力矩在末端产生的是一个方向错误的加速度；
- QP 在力矩可行集内找最接近期望加速度的解，末端方向保持正确，只是变慢。

同理，关节限位和速度限位在 OSC 里只能靠额外的排斥势场，在 QP 里则是硬约束。更复杂的全身控制会加入接触力变量、摩擦锥和多个优先级（层级 QP）。

## 基准结果

<BenchTable
  track="tracking"
  :conditions="{ nominal: '标称模型', payload: '1.5 kg 未知负载', friction: '关节干摩擦 2 %' }"
  :rows="['osc', 'qp', 'mpc']"
  :columns="[
    { key: 'tcp_max_mm', label: '最大 TCP 误差', unit: 'mm', best: 'min' },
    { key: 'final_mm', label: '结束误差', unit: 'mm', best: 'min' },
    { key: 'saturation', label: '饱和比例', unit: '%', scale: 100, digits: 2, best: 'min' },
    { key: 'us_per_step', label: '每步耗时（256 世界）', unit: 'µs', digits: 0, best: 'min' },
  ]"
/>

## 参考文献

- B. Stellato et al., "OSQP: An operator splitting solver for quadratic programs," *Math. Prog. Comp.*, 2020.
- A. Escande, N. Mansard, P.-B. Wieber, "Hierarchical quadratic programming," *IJRR*, 2014.
- L. Sentis, O. Khatib, "Synthesis of whole-body behaviors through hierarchical control of behavioral primitives," *IJHR*, 2005.
