# 阻抗控制

::: tip 一句话总结
阻抗控制不追求“到达位置”，而是规定“被推开多少、用多大的力回来”：让机械臂在关节或末端表现成一个可调的弹簧-阻尼器。它只需要关节力矩控制，不需要末端力传感器，接触、碰撞、被人推时都是安全的。这是协作机器人的默认工作方式，代价是精度随刚度下降。
:::

## 目标阻抗

Hogan（1985）把机械臂与环境的交互看成两个系统通过“端口”交换功率：环境是导纳（受力产生运动），机械臂就应该是阻抗（受运动产生力）。目标是让末端偏差 $\tilde x = x - x_d$ 与外力 $F_{\text{ext}}$ 满足

$$
\Lambda_d \ddot{\tilde x} + D_d \dot{\tilde x} + K_d \tilde x = F_{\text{ext}} .
$$

- **刚度** $K_d$：静态时被推开 $\tilde x = K_d^{-1}F_{\text{ext}}$；
- **阻尼** $D_d$：决定被推后回到参考的快慢和超调；
- **惯量** $\Lambda_d$：改变惯量需要测量外力（$\Lambda_d \ne \Lambda(q)$ 时 $F_{\text{ext}}$ 出现在控制律里），所以常用的是**不改变惯量**的版本。

## 笛卡尔阻抗（不改变惯量）

取 $\Lambda_d = \Lambda(q)$，控制律不含 $F_{\text{ext}}$（Ott 2008）：

$$
\boxed{\ \tau = J^\top\!\left(K e + D(\dot x_d - \dot x) + \Lambda\ddot x_d\right) + C(q,\dot q)\dot q + g(q) + N^\top\tau_{\text{null}}\ }
$$

- $e = [p_d - p;\ \operatorname{Log}(R_d R^\top)]$，平移刚度 2000 N/m、转动刚度 150 N·m/rad（<Src path="rczoo/control/gains.py" />）；
- $\Lambda\ddot x_d$ 是可选的加速度前馈，跟踪运动时显著减小误差；
- 冗余机械臂加一个零空间姿态力矩 $N^\top\tau_{\text{null}}$，见 [OSC](./osc)。

### 阻尼设计

固定的 $D$ 在不同位形下阻尼比不同，因为 $\Lambda(q)$ 变化很大。理想的“双对角化”设计（Albu-Schäffer 等 2003）同时对角化 $\Lambda$ 与 $K$，让每个模态都临界阻尼。本站用它的对角近似：

$$
D_i = 2\zeta\sqrt{K_i\,\Lambda_{ii}(q)},\qquad \zeta = 1 ,
$$

每个周期随位形更新（<Src path="rczoo/control/task.py" symbol="CartesianImpedance" />）。

### 被动性

不改变惯量的阻抗控制器以 $V = \tfrac12\dot q^\top M\dot q + \tfrac12 e^\top K e$ 为存储函数（定点情形），满足 $\dot V \le \dot x^\top F_{\text{ext}}$：**机器人与环境组成的系统不会凭空产生能量**，与任何被动环境（墙、人手）接触都稳定。这是阻抗控制在接触任务中受欢迎的根本原因。

## 关节阻抗

把弹簧放在关节空间：

$$
\tau = K_q (q_d - q) + D_q(\dot q_d - \dot q) + M\ddot q_d + C(q,\dot q)\dot q + g(q).
$$

它与 PD+ 的结构几乎一样，只是增益低得多（本站取 Franka 增益的四分之一），使机械臂“软”下来。Franka 的所有位置类接口默认由内部的关节阻抗控制器执行。

## 精度与柔顺的折中

被推时的偏差、负载造成的下垂都与刚度成反比：

| | 笛卡尔阻抗 | 关节阻抗 |
| --- | --- | --- |
| 标称跟踪最大误差（Panda） | ~5 mm | ~2.3 mm |
| 1.5 kg 负载结束误差 | ~7 mm | ~47 mm |
| 30 N 推力下最大偏差 | ~10 mm | ~64 mm |

关节阻抗的刚度在末端表现为 $(J K_q^{-1} J^\top)^{-1}$，伸展的手臂上小关节刚度的“杠杆”效应会让末端很软，所以负载造成的误差大得多。

## 基准：抗扰（推一下）

机械臂保持静止，TCP 受到 30 N、0.2 s 的水平推力（随机方向）。

<BenchTable
  track="push"
  :rows="['pid', 'pd_grav', 'joint_impedance', 'cart_impedance', 'ctc', 'osc', 'qp', 'smc']"
  :columns="[
    { key: 'max_dev_mm', label: '最大偏差', unit: 'mm', best: 'min' },
    { key: 'stiffness_N_per_mm', label: '等效刚度', unit: 'N/mm', best: 'max' },
    { key: 'settle_s', label: '恢复时间（1 mm）', unit: 's', best: 'min' },
    { key: 'final_mm', label: '2 s 时误差', unit: 'mm', best: 'min' },
  ]"
/>

<BenchChart
  track="push"
  field="dev_mm"
  :controllers="['joint_impedance', 'cart_impedance', 'pd_grav', 'ctc', 'smc']"
  y-label="TCP 偏差 (mm)"
  unit=" mm"
  :worlds="3"
/>

“恢复时间”是推力结束后 TCP 最后一次超出参考 1 mm 的时刻；PID 在 1.5 s 内一直没有回到 1 mm 以内（积分项被推力“充电”后要慢慢放掉）。阻抗控制器偏差大但平稳，滑模控制几乎纹丝不动——但在接触任务里，这种“硬”恰恰是危险的。

## 动手试试

在 [控制演练场](/playground/) 选择“笛卡尔阻抗”，点“推一下”；再切到“接触力控”场景，看它压在平面上的力为什么达不到设定值。

## 参考文献

- N. Hogan, "Impedance control: An approach to manipulation, Parts I–III," *J. Dyn. Sys., Meas., Control*, 1985.
- C. Ott, *Cartesian Impedance Control of Redundant and Flexible-Joint Robots*, Springer, 2008.
- A. Albu-Schäffer, C. Ott, U. Frese, G. Hirzinger, "Cartesian impedance control of redundant robots: Recent results with the DLR-light-weight-arms," *ICRA*, 2003.
