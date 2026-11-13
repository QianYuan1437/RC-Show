# PID 与 PD + 重力补偿

::: tip 一句话总结
不依赖任何模型的独立关节 PID 是伺服驱动器的经典做法，但对多关节机械臂它只能靠积分项“慢慢”抵消重力，响应慢而且容易超调。只加一项模型——重力补偿 $g(q)$——就得到位置模式的标准实现 PD + 重力补偿：定点控制全局稳定，代价是快速运动时滞后、负载不准时下垂。
:::

## 独立关节 PID

$$
\tau = K_p e + K_i \!\int_0^t e\,ds + K_d \dot e,\qquad e = q_d - q .
$$

每个关节各自为政，别的关节的运动、重力、科氏力都被当作扰动，由积分项和高增益去对抗。对大减速比的工业臂这很有效（关节侧的惯量和耦合被减速比平方削弱）；对直驱或小减速比、力矩控制的协作臂，重力矩相对电机能力很大，积分项要承担几十 N·m 的力矩：

- 积分需要时间累积，起动和换向时有明显误差；
- 积分饱和（windup）会造成超调，必须限幅（本站把积分力矩限制在额定力矩的一半，<Src path="rczoo/control/joint.py" symbol="JointPID" />）；
- 提高 $K_i$ 会降低相位裕度。

基准中 Panda 用 PID 跟踪五次多项式轨迹时最大 TCP 误差约 45 mm，结束 0.5 s 后仍有十几毫米没有被积分消除——这是所有控制器里最差的。

## PD + 重力补偿

$$
\boxed{\ \tau = K_p (q_d - q) + K_d (\dot q_d - \dot q) + g(q)\ }
$$

### 稳定性（定点）

取 $\dot q_d = 0$，候选李雅普诺夫函数为动能加“弹簧势能”：

$$
V = \tfrac12 \dot q^\top M(q)\dot q + \tfrac12 e^\top K_p e .
$$

沿闭环轨迹求导，利用 $\dot M - 2C$ 的反对称性：

$$
\dot V = \dot q^\top\!\left(\tau - C\dot q - g\right) + \tfrac12\dot q^\top\dot M\dot q - \dot q^\top K_p e
= -\dot q^\top K_d \dot q \le 0 .
$$

再由 LaSalle 不变集原理，$\dot q \equiv 0$ 时 $\ddot q = M^{-1}K_p e = 0 \Rightarrow e = 0$，所以平衡点全局渐近稳定（Takegaki & Arimoto 1981）。**它对任意正定的 $K_p, K_d$ 都成立，不需要惯性参数**——这是它被广泛用作位置模式的原因。

### 两个误差来源

**跟踪滞后**。跟踪运动轨迹时，惯性力 $M\ddot q_d$ 和科氏力 $C\dot q_d$ 没有前馈，必须由误差“撑起来”：$e \approx K_p^{-1}(M\ddot q_d + C\dot q_d)$。增益越低、运动越快，误差越大。

**模型误差造成下垂**。如果末端有一个模型里没有的负载，重力项少补了 $\Delta g(q)$，稳态时

$$
e_{\infty} = K_p^{-1}\,\Delta g(q_d) .
$$

Panda 带 1.5 kg 未知负载时，这对应约 12 mm 的 TCP 误差，并且**一直存在**。消除它要么加积分（PID + 重力补偿，或速度模式的内环 PI），要么让控制器估计负载（[自适应控制](./robust-adaptive)）。

### 增益

本站的关节增益取自 Franka 示例控制器（$K_p = 600$ N·m/rad、$K_d = 50$ N·m·s/rad 用于大关节，腕部依次减小），SR3-C 按额定力矩比例缩放（<Src path="rczoo/control/gains.py" />）。

## 基准结果

<BenchTable
  track="tracking"
  :conditions="{ nominal: '标称模型', payload: '1.5 kg 未知负载', friction: '关节干摩擦 2 %' }"
  :rows="['pid', 'pd_grav', 'velocity', 'pd_ff', 'ctc']"
  :columns="[
    { key: 'tcp_max_mm', label: '最大 TCP 误差', unit: 'mm', best: 'min' },
    { key: 'final_mm', label: '结束误差', unit: 'mm', best: 'min' },
    { key: 'q_rms_deg', label: '关节 RMS 误差', unit: '°', best: 'min' },
    { key: 'energy_J', label: '机械能', unit: 'J', best: 'min' },
  ]"
/>

表中为 256 条随机轨迹的中位数（关节 RMS 误差与能量为均值）；“结束误差”是轨迹结束、保持 0.5 s 后的误差。切到“1.5 kg 未知负载”可以看到 PD + 重力补偿和 PD+ 都停在约 12 mm（同样的 $K_p^{-1}\Delta g$），计算力矩约 9 mm；速度模式的内环积分在保持阶段就把它消除了，而 PID 的积分为了稳定性不能取大，0.5 s 后仍有约 19 mm。

<BenchChart
  track="tracking"
  :conditions="{ nominal: '标称模型', payload: '1.5 kg 未知负载' }"
  field="e_tcp_mm"
  :controllers="['pid', 'pd_grav', 'velocity', 'ctc']"
  y-label="TCP 误差 (mm)"
  unit=" mm"
  log-y
/>

## 参考文献

- M. Takegaki, S. Arimoto, "A new feedback method for dynamic control of manipulators," *J. Dyn. Sys., Meas., Control*, 1981.
- B. Siciliano, L. Sciavicco, L. Villani, G. Oriolo, *Robotics: Modelling, Planning and Control*, Springer, 2009, ch. 8.
