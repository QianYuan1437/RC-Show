# 计算力矩与 PD+

::: tip 一句话总结
如果模型准确，用 $\tau = M(q)v + h(q,\dot q)$ 就能把耦合、非线性的机械臂变成 $n$ 个独立的二重积分器 $\ddot q = v$，再用线性控制让每个关节的误差以同样的带宽收敛——这就是计算力矩（反馈线性化）。PD+ 只在参考轨迹上做模型前馈，反馈仍是固定的 PD，对模型误差更宽容。两者都把跟踪误差降到 1 mm 以下，但都对未建模负载没有办法。
:::

## 计算力矩控制

动力学 $M\ddot q + h = \tau$ 中，$M$ 总是可逆的。令

$$
\boxed{\ \tau = M(q)\left(\ddot q_d + K_p e + K_d \dot e\right) + h(q,\dot q)\ },\qquad e = q_d - q,
$$

代入得 $M(\ddot q_d - \ddot q + K_p e + K_d\dot e) = 0$，即

$$
\ddot e + K_d \dot e + K_p e = 0 .
$$

每个关节的误差都是一个独立的二阶线性系统。取 $K_p = \omega_n^2$、$K_d = 2\zeta\omega_n$，所有关节都有相同的固有频率和阻尼比。本站取 $\omega_n = 25$ rad/s、$\zeta = 1$（临界阻尼），与机器人无关（<Src path="rczoo/control/joint.py" symbol="ComputedTorque" />）。

### 对模型误差的敏感性

真实对象是 $\hat M\ddot q + \hat h = \tau$（帽子表示真实值），代入控制律后

$$
\ddot e + K_d\dot e + K_p e = \hat M^{-1}\big[(\hat M - M)\,(\ddot q_d + K_p e + K_d\dot e) + (\hat h - h)\big] =: \eta .
$$

模型误差变成了一个依赖状态的扰动 $\eta$。未知负载使 $\hat h - h \approx \Delta g$，稳态误差为

$$
e_\infty = K_p^{-1} M^{-1}\Delta g = \omega_n^{-2} M^{-1}\Delta g .
$$

与 PD + 重力补偿的 $K_p^{-1}\Delta g$ 相比，增益“被 $M$ 加权”了，但量级相近：基准中 1.5 kg 负载下计算力矩停在约 9 mm，PD + 重力补偿约 12 mm。加一个积分项（`ctc` 的 `ki` 参数）或者用[鲁棒、自适应](./robust-adaptive)的方法才能消除它。

## PD+（Paden & Panja 1988）

$$
\boxed{\ \tau = M(q)\ddot q_d + C(q,\dot q)\dot q_d + g(q) + K_p e + K_d \dot e\ }
$$

与计算力矩的区别在于：模型只用于沿**参考轨迹**的前馈，PD 反馈不乘 $M$。以 $V = \tfrac12\dot e^\top M\dot e + \tfrac12 e^\top K_p e$ 为李雅普诺夫函数，利用 $\dot M - 2C$ 的反对称性可得 $\dot V = -\dot e^\top K_d\dot e$，因此跟踪误差全局渐近收敛。注意这里的 $C$ **必须**是 Christoffel 形式，且是 $C(q,\dot q)\dot q_d$ 而不是 $C(q_d,\dot q_d)\dot q_d$。本站用修改版 RNEA 一次算出 $M\ddot q_d + C(q,\dot q)\dot q_d + g$（见 [动力学基础](/guide/dynamics#christoffel)）。

### 两者怎么选

| | 计算力矩 | PD+ |
| --- | --- | --- |
| 闭环误差动力学 | 解耦、线性、与位形无关 | 耦合，依赖 $M(q)$ |
| 增益设计 | 一个 $\omega_n$ 适用于所有关节 | 每个关节单独调（大关节高、小关节低） |
| 对模型误差 | 反馈也被 $M$ 放大 / 缩小 | 反馈不依赖模型 |
| 稳定性证明 | 需要精确模型 | 被动性，模型误差只影响性能 |

实践中两者精度相近（基准中 Panda 最大 TCP 误差分别约 0.7 mm 与 0.6 mm）。PD+ 的被动性结构是后续[滑模、自适应控制](./robust-adaptive)的出发点。

## 基准结果

<BenchTable
  track="tracking"
  :conditions="{ nominal: '标称模型', payload: '1.5 kg 未知负载', friction: '关节干摩擦 2 %' }"
  :rows="['pd_grav', 'pd_ff', 'ctc', 'smc', 'adaptive']"
  :columns="[
    { key: 'tcp_max_mm', label: '最大 TCP 误差', unit: 'mm', best: 'min' },
    { key: 'tcp_rms_mm', label: 'RMS', unit: 'mm', best: 'min' },
    { key: 'final_mm', label: '结束误差', unit: 'mm', best: 'min' },
    { key: 'us_per_step', label: '每步耗时（256 世界）', unit: 'µs', digits: 0 },
  ]"
/>

<BenchChart
  track="tracking"
  :conditions="{ nominal: '标称模型', payload: '1.5 kg 未知负载' }"
  field="tau_rel"
  reduce="maxabs"
  :controllers="['pd_grav', 'pd_ff', 'ctc']"
  y-label="max |τ| / τ_max"
/>

上图是同一条轨迹上三种控制器的力矩（各关节相对额定力矩的最大值）：计算力矩与 PD+ 在起动时就输出了加速所需的力矩，PD + 重力补偿要等误差积累起来才“使劲”。

## 参考文献

- B. Paden, R. Panja, "Globally asymptotically stable 'PD+' controller for robot manipulators," *Int. J. Control*, 1988.
- R. M. Murray, Z. Li, S. S. Sastry, *A Mathematical Introduction to Robotic Manipulation*, CRC Press, 1994, ch. 4.
