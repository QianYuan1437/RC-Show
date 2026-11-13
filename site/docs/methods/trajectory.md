# 轨迹生成

::: tip 一句话总结
控制器只能跟踪“跟得上”的参考。轨迹生成把稀疏的目标（终点、路径点）变成每个控制周期的位置、速度、加速度参考，同时让速度、加速度、加加速度不超限：从闭式的五次多项式和梯形 / 双 S 速度曲线，到在力矩约束下时间最优的 TOPP-RA，再到目标随时可变的在线轨迹生成。
:::

## 为什么需要平滑的参考

如果直接把目标位置作为参考（阶跃），PD 控制器在第一个周期就会输出 $K_p\Delta q$ 的巨大力矩，马上饱和；参考的速度、加速度不连续，也会激发机械臂和减速器的弹性振动。因此参考至少要满足：

- **位置和速度连续**（加速度有界）：梯形速度曲线；
- **加速度连续**（加加速度有界）：双 S 曲线、五次多项式。加加速度有界的轨迹对柔性关节和减速器更友好，工业控制器的 MoveJ / MoveL 通常使用这一类曲线。

## 点到点的单轴曲线

### 五次多项式

给定起止位置、速度、加速度（6 个边界条件）和时长 $T$，五次多项式唯一确定。静止到静止时，记 $s = t/T$、$h = q_1 - q_0$：

$$
q(t) = q_0 + h\,(10 s^3 - 15 s^4 + 6 s^5),\qquad
\dot q_{\max} = \frac{15 h}{8T},\qquad \ddot q_{\max} = \frac{10 h}{\sqrt3\,T^2}.
$$

优点是解析、各阶导数光滑；缺点是峰值速度只在中点达到，在给定速度、加速度上限时比梯形曲线慢。基准中所有关节轨迹都是五次多项式（<Src path="rczoo/traj.py" symbol="quintic" />）。

### 梯形速度曲线

“加速–匀速–减速”，加速度在 $\pm a_{\max}$ 与 0 之间跳变。距离 $h < v_{\max}^2/a_{\max}$ 时到不了 $v_{\max}$，退化成三角形；否则

$$
T = \frac{h}{v_{\max}} + \frac{v_{\max}}{a_{\max}} .
$$

它是给定 $v_{\max}, a_{\max}$ 时最快的曲线，但加速度阶跃、加加速度是冲激。

### 双 S（七段）曲线

在梯形的基础上把加速度的阶跃换成斜率为 $\pm j_{\max}$ 的斜坡，共七段：加加速、匀加速、减加速、匀速、加减速、匀减速、减减速（Biagiotti & Melchiorri 2008）。要分情况讨论能否达到 $v_{\max}$ 和 $a_{\max}$。本站的实现（<Src path="rczoo/traj.py" symbol="DoubleS" />）在批量上用 `torch.where` 处理所有分支。

<TrajectoryDemo />

拖动滑块可以看到：

- $j_{\max}$ 很大时双 S 退化成梯形；
- 距离很短时三种曲线都达不到 $v_{\max}$；
- 在同样的速度、加速度上限下，五次多项式总是最慢的，因为它的加速度只在一瞬间达到峰值。

### 多轴同步

每个关节单独规划时各自的时长不同，末端走的就不是“一条直线”在关节空间里的像。常见做法是取最慢关节的时长 $T^\star$，把其它关节的曲线在时间上**拉伸**到 $T^\star$（等比例缩放速度上限，<Src path="rczoo/traj.py" symbol="sync" />），这样所有关节同时出发、同时到达，关节空间里是一条直线（MoveJ / MoveAbsJ 的行为）。

## 笛卡尔路径

- **MoveL**：位置走直线，姿态用 SLERP（在旋转向量上线性插值），两者共用一个标量双 S 曲线 $s(t)\in[0,1]$，限值按路径长度和转角折算（<Src path="rczoo/traj.py" symbol="MoveL" />）。
- **MoveC**：三点确定圆弧，按弧长参数化（<Src path="rczoo/traj.py" symbol="MoveC" />）。
- **过渡（blending）**：相邻两段在交接前 $T_o$ 秒就开始叠加位移，拐角被圆滑过渡，速度、加速度保持连续（<Src path="rczoo/traj.py" symbol="blend_positions" />）。对应工业控制器的“转弯区”参数。

笛卡尔路径需要逆运动学把 TCP 参考变成关节参考，或者直接交给[任务空间控制器](./osc)。

## 时间最优参数化：TOPP-RA

给定一条几何路径 $q(s)$，问题是找出最快的速度曲线 $\dot s(t)$，使所有关节速度、加速度、**力矩**都不超限。令 $x = \dot s^2$、$u = \ddot s$，则

$$
\dot q = q'\,\dot s,\qquad \ddot q = q' u + q'' x,\qquad
\tau = m(s)\,u + c(s)\,x + g(s),
$$

其中 $m = M q'$，$c = M q'' + C(q,q')q'$。所有约束都是 $(u, x)$ 的**线性**不等式，且相邻网格点满足 $x_{i+1} = x_i + 2 u_i \Delta s$。TOPP-RA（Pham & Pham 2018）的两遍算法：

1. **反向**：从终点 $x_N = 0$ 出发，逐点计算“能控集” $K_i$，即从该点出发、在之后都满足约束并能停在终点的 $x_i$ 区间；每一步是一个二维线性规划。
2. **正向**：从起点出发，每一步在 $K_{i+1}$ 允许的范围内取最大的 $u_i$（贪心）。

本站用二维 LP 的顶点枚举实现，并在测试中检查约束满足与“速度曲线贴着最大速度曲线（MVC）”（<Src path="rczoo/traj.py" symbol="toppra" />）。

## 在线轨迹生成（OTG）

当目标会随时改变时（遥操作、视觉伺服、被拖动的目标点），离线规划不再适用。在线轨迹生成每个周期从**当前状态**出发重新规划到目标的时间最优曲线，只执行第一步（Kröger 的 Reflexxes 库是典型代表）。本站实现了加速度受限的单轴版本（Reflexxes Type II 不含相位同步）：

$$
d_{\text{stop}} = \frac{v\lvert v\rvert}{2 a_{\max}},\qquad
\ddot q = \begin{cases}
-\operatorname{sign}(d)\,\dfrac{v^2}{2\lvert d\rvert} & \text{需要刹车时}\\[4pt]
\pm a_{\max}\ \text{或}\ 0 & \text{否则（加速 / 匀速）}
\end{cases}
$$

[控制演练场](/playground/) 用它把被拖动的目标位姿平滑成 TCP 参考（每个平移轴一个，姿态沿误差旋转轴一个），所以无论目标怎么跳，控制器看到的参考速度都连续且不超过 0.4 m/s、加速度不超过 2 m/s²（<Src path="site/docs/.vitepress/theme/lib/traj.ts" symbol="otgStep" />）。

## 参考文献

- L. Biagiotti, C. Melchiorri, *Trajectory Planning for Automatic Machines and Robots*, Springer, 2008.
- H. Pham, Q.-C. Pham, "A new approach to time-optimal path parameterization based on reachability analysis," *IEEE T-RO*, 2018.
- T. Kröger, *On-Line Trajectory Generation in Robotic Systems*, Springer, 2010.
