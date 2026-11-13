# 导纳与力/位控制

::: tip 一句话总结
要以确定的力压住表面（打磨、擦拭、装配），只靠阻抗的“弹簧压缩量”是不够的：力的大小取决于环境刚度和参考有多深，而这两者都不准确。显式力控制用末端六维力传感器闭环：**导纳控制**把测得的力变成位置修正，**力/位混合控制**在法向控力、切向控位，**力/位并联控制**让力环和位置环在同一方向上“协商”。
:::

## 接触模型与实验设置

仿真中 TCP 处有一个半径 12 mm 的探针球和一个六维力传感器，接触面是一块可移动的平板。接触是 MuJoCo 的软约束，表现为非线性弹簧；在 10 N 附近，三种设置的等效刚度约为：

| 表面 | MuJoCo `solref` 时间常数 | 10 N 附近的刚度 |
| --- | --- | --- |
| 软 | 0.1 s | ≈ 1.4 kN/m |
| 中 | 0.05 s | ≈ 3.7 kN/m |
| 硬 | 0.02 s | ≈ 21 kN/m |

任务：从表面上方 5 cm 下降，1 s 时接触，0.3 s 内把期望力斜坡升到 10 N，然后保持压力沿切向滑动 15 cm（2 s），再保持 1 s。

::: info 为什么接触面无摩擦
带摩擦时，位置控制的探针在表面上滑动会出现“粘–滑”：MuJoCo 软接触的摩擦约束在粘住和滑动之间切换，法向力随之跳变 ±5 N，所有控制器的力误差都被这个与控制无关的现象淹没。因此基准使用无摩擦接触（`condim = 1`），只比较法向力控制本身。
:::

## 阻抗控制：用“深度”换力

[笛卡尔阻抗](./impedance) 没有力反馈，接触力完全由弹簧压缩量决定。设计者通常假设表面是刚性的，把参考放到表面下 $\delta = f_d / K_z$（2000 N/m 时 5 mm）。但真实表面会变形：探针压进表面 $d$，弹簧只压缩 $\delta - d$，力为两个弹簧串联的结果

$$
f = \frac{K_z k_{\text{env}}}{K_z + k_{\text{env}}}\,\delta < f_d .
$$

表面越软，力越小：基准中软表面上只有约 3 N（期望 10 N），硬表面上约 9 N。

## 导纳控制

导纳控制测量外力 $F$，让一个虚拟的质量-弹簧-阻尼系统产生位置修正 $x_c$：

$$
M_a\ddot x_c + D_a\dot x_c + K_a x_c = F ,
$$

然后把 $x_d + x_c$ 交给一个刚性的内环位置控制器（本站为 $\omega_n = 40$ rad/s 的 OSC）。它和阻抗控制是对偶的：阻抗是“运动进、力出”，导纳是“力进、运动出”。

- 本体刚性、摩擦大、减速比大的**工业臂**很难做好力矩控制，但位置环很准，配一个末端力传感器就能做导纳控制；
- 导纳的虚拟刚度可以很低（本站 $K_a = 300$ N/m），所以对硬表面也柔顺；但它与刚性环境接触时，高增益的内环加上力传感器的延迟容易激发振荡，峰值力较大（硬表面上约 25 N）；
- 力的精度取决于参考深度 $f_d/K_a$ 是否准确：软表面上同样会力不足。

实现见 <Src path="rczoo/control/task.py" symbol="Admittance" />（力传感器读数先经过 0.5 N 死区）。

## 力/位混合控制

Raibert & Craig（1981）的思路：任务空间被约束分成正交的两部分，沿表面法向控制力，沿切向和姿态控制运动。以 OSC 形式写出（Khatib 1987）：

$$
F = \Lambda\,\Omega\big(\ddot x_d + K_p e + K_d\dot e - \dot J\dot q\big) + n\,f_{\text{cmd}},\qquad \Omega = I - n n^\top ,
$$

$$
f_{\text{cmd}} = -\Big(f_d + K_{pf}(f_d - f) + K_{if}\!\int (f_d - f)\,dt\Big) - K_{vf}\,(n^\top\dot x) .
$$

- $\Omega$ 是选择矩阵，把法向从运动控制中去掉；
- 法向力指令 = 前馈 $f_d$ + 力误差的 PI + 法向速度阻尼（$K_{vf}$ 抑制接触瞬间的冲击和振荡）；
- **导引运动（guarded move）**：未接触时法向也是位置控制，测得的力超过 1 N 后锁存为接触状态，再切换到力控制，积分也只在接触后启用。否则力环会从空中就“全力推”，以很大的速度撞上表面。

实现见 <Src path="rczoo/control/task.py" symbol="HybridForceMotion" />。

## 力/位并联控制

Chiaverini & Sciavicco（1993）：不划分方向，而是在阻抗控制器的参考上叠加一个由力误差 PI 驱动的偏移：

$$
x_r = x_d - n\left(k_{pf}(f_d - f) + k_{if}\!\int (f_d - f)\,dt\right).
$$

在不受约束的方向上力误差不变，位置环胜出；在受约束的方向上，积分会一直把参考往表面里推，直到力误差为零——**力环“优先”于位置环**。它不需要事先知道约束方向，对表面倾斜、未知形状更宽容（<Src path="rczoo/control/task.py" symbol="ParallelForcePosition" />）。

## 基准结果

<BenchTable
  track="contact"
  :conditions="{ soft: '软表面', medium: '中等', stiff: '硬表面' }"
  :rows="['cart_impedance', 'admittance', 'hybrid', 'parallel']"
  :columns="[
    { key: 'mean_N', label: '平均力', unit: 'N' },
    { key: 'rms_err_N', label: '力 RMS 误差', unit: 'N', best: 'min' },
    { key: 'peak_N', label: '峰值力', unit: 'N' },
    { key: 'xy_err_mm', label: '切向位置误差', unit: 'mm', best: 'min' },
    { key: 'penetration_mm', label: '压入深度', unit: 'mm' },
  ]"
/>

<BenchChart
  track="contact"
  :conditions="{ soft: '软表面', medium: '中等', stiff: '硬表面' }"
  field="fz"
  :controllers="['cart_impedance', 'admittance', 'hybrid', 'parallel']"
  ref-field="f_ref"
  ref-label="期望力"
  y-label="接触力 f_z (N)"
  unit=" N"
/>

力误差在期望力达到 10 N 之后统计（包括滑动和保持阶段），256 个起始位形。混合控制在三种表面上的 RMS 力误差都在 0.15 N 以内，切向位置误差约 0.04 mm；并联控制力精度略低但没有接触冲击（峰值等于期望力）；导纳控制在硬表面上峰值力最大。

## 动手试试

在 [控制演练场](/playground/) 选择“接触力控”场景，拖动目标让探针沿平面滑动，调节期望接触力；对比笛卡尔阻抗（力不足）与混合控制。

## 参考文献

- M. H. Raibert, J. J. Craig, "Hybrid position/force control of manipulators," *J. Dyn. Sys., Meas., Control*, 1981.
- S. Chiaverini, L. Sciavicco, "The parallel approach to force/position control of robotic manipulators," *IEEE T-RA*, 1993.
- L. Villani, J. De Schutter, "Force control," in *Springer Handbook of Robotics*, 2016.
