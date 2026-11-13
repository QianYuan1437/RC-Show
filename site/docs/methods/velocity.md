# 速度模式与分解速率

::: tip 一句话总结
速度模式把一个带积分的关节速度环放在最内层，外面套一个比例位置环：$\dot q_{\text{cmd}} = \dot q_d + K_x e$，$\tau = K_v(\dot q_{\text{cmd}}-\dot q) + K_i\!\int(\dot q_{\text{cmd}}-\dot q) + g$。速度环的积分自动吸收负载和摩擦，所以它对模型误差很宽容。把末端速度指令经过阻尼伪逆映射到关节，就得到笛卡尔速度模式——分解速率控制。
:::

## 关节速度模式

### 串级结构

工业伺服驱动器几乎都是“位置 P → 速度 PI → 电流 PI”的串级结构：内环带宽高（速度环约 100 Hz 量级），外环带宽低，每一层只需要把下一层当成一个快速的执行器。本站在 1 kHz 力矩层面实现同样的结构（<Src path="rczoo/control/joint.py" symbol="JointVelocityMode" />）：

$$
\dot q_{\text{cmd}} = \operatorname{clip}\!\left(\dot q_d + K_x (q_d - q),\ \pm \dot q_{\max}\right),
$$

$$
\tau = K_v (\dot q_{\text{cmd}} - \dot q) + K_i \!\int (\dot q_{\text{cmd}} - \dot q)\,dt + g(q).
$$

- 速度指令被限制在关节最大速度以内，这是速度模式天然的安全特性；
- 积分作用在**速度误差**上：在稳态 $\dot q = \dot q_{\text{cmd}} = 0$，积分值恰好等于所有未被补偿的静态力矩（负载重力、库仑摩擦），而位置误差 $e = 0$；
- 本质上它等价于一个 PID 位置控制器（$\int \dot e = e$），但积分作用在内环，相位更好，抗饱和也更简单。

### 与 PD + 重力补偿的对比

基准中速度模式的标称跟踪误差（Panda 最大约 1.2 mm）明显优于 PD + 重力补偿（约 6 mm）：内环增益 $K_v$ 等效于很大的阻尼和刚度。负载为 1.5 kg 时它的最大误差升到约 4 mm，但结束误差只有 0.16 mm，而 PD + 重力补偿停在约 12 mm。

## 笛卡尔速度模式：分解速率控制

给定 TCP 参考 $(x_d, \dot x_d)$，末端速度指令为

$$
\dot x_{\text{cmd}} = \dot x_d + K_x e,\qquad e = [p_d - p;\ \operatorname{Log}(R_d R^\top)],
$$

关节速度由雅可比的阻尼伪逆给出（Whitney 1969；阻尼最小二乘，Nakamura & Hanafusa 1986）：

$$
\dot q_{\text{cmd}} = J^\top (J J^\top + \lambda^2 I)^{-1}\dot x_{\text{cmd}} + (I - J^{+}J)\,k_0 (q_{\text{null}} - q).
$$

第二项只对冗余的 Panda 存在：在不影响末端的零空间里把关节拉向一个舒适的姿态，避免肘部漂移。之后 $\dot q_{\text{cmd}}$ 交给上面的关节速度环执行（<Src path="rczoo/control/task.py" symbol="ResolvedRate" />）。

阻尼 $\lambda$ 是奇异鲁棒性与精度之间的折中：远离奇异时几乎不起作用；接近奇异时把病态方向上的关节速度压到有界，代价是该方向的末端误差。SR3-C 在腕部奇异附近尤其需要它。

### 与操作空间控制的区别

分解速率是**运动学**的：它认为关节速度指令能被立即实现，动力学交给内环积分去“硬扛”。[操作空间控制](./osc) 则在力矩层面考虑任务空间的惯量 $\Lambda$，可以精确解耦末端动力学。前者实现简单、鲁棒，适合速度接口的机器人；后者精度更高，但依赖准确的模型。

## 基准结果

<BenchTable
  track="tracking"
  :conditions="{ nominal: '标称模型', payload: '1.5 kg 未知负载', friction: '关节干摩擦 2 %' }"
  :rows="['pd_grav', 'velocity', 'resolved_rate', 'osc']"
  :columns="[
    { key: 'tcp_max_mm', label: '最大 TCP 误差', unit: 'mm', best: 'min' },
    { key: 'final_mm', label: '结束误差', unit: 'mm', best: 'min' },
    { key: 'saturation', label: '饱和比例', unit: '%', scale: 100, digits: 1 },
  ]"
/>

## 参考文献

- D. E. Whitney, "Resolved motion rate control of manipulators and human prostheses," *IEEE Trans. Man-Machine Systems*, 1969.
- Y. Nakamura, H. Hanafusa, "Inverse kinematic solutions with singularity robustness for robot manipulator control," *J. Dyn. Sys., Meas., Control*, 1986.
