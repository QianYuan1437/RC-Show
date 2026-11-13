# 操作空间控制 OSC

::: tip 一句话总结
操作空间控制（Khatib 1987）是“任务空间里的计算力矩”：用任务空间惯量 $\Lambda(q)$ 把末端动力学解耦、线性化，让 TCP 误差在每个方向上都按 $\ddot e + K_d\dot e + K_p e = 0$ 收敛；冗余自由度通过动力学一致的零空间投影去完成次要任务，而不干扰末端。
:::

## 任务空间动力学

由 $\dot x = J\dot q$、$\ddot x = J\ddot q + \dot J\dot q$，把关节动力学投影到末端：

$$
\Lambda(q)\,\ddot x + \mu(q,\dot q) + p(q) = F,\qquad \tau = J^\top F,
$$

$$
\Lambda = (J M^{-1} J^\top)^{-1},\qquad
\mu + p = \bar J^\top h - \Lambda\dot J\dot q,\qquad
\bar J = M^{-1}J^\top\Lambda .
$$

$\Lambda$ 是末端在各个方向上“感受到”的惯量，$\bar J$ 是动力学一致的广义逆。

## 控制律

$$
\boxed{\ \tau = J^\top\Lambda\big(\ddot x_d + K_p e + K_d(\dot x_d - \dot x) - \dot J\dot q\big) + h(q,\dot q) + N^\top\tau_{\text{null}}\ }
$$

在模型准确时，代入任务空间动力学得 $\ddot e + K_d\dot e + K_p e = 0$（6 个独立方向）。这里把 $h$ 直接加在关节空间，而不是用 $J^\top(\mu + p)$，两者对末端等价，前者还补偿了零空间里的重力和科氏力（<Src path="rczoo/control/task.py" symbol="OperationalSpace" />）。本站取 $\omega_n = 25$ rad/s。

## 冗余与零空间

Panda 有 7 个关节、末端任务 6 维，剩下一维“自运动”（肘部绕肩–腕连线转动）。如果不管它，肘部会在运动中漂移，最终撞上关节限位。零空间力矩

$$
N^\top\tau_{\text{null}},\qquad N^\top = I - J^\top\bar J^\top = I - J^\top\Lambda J M^{-1},\qquad
\tau_{\text{null}} = K_n(q_{\text{null}} - q) - D_n\dot q,
$$

满足 $J M^{-1}N^\top\tau_{\text{null}} = 0$：它产生的关节加速度在末端看来恰好为零。注意必须用**动力学一致**的投影；用运动学的 $I - J^\top (J^{+})^\top$ 会让零空间力矩“泄漏”到末端。SR3-C 是 6 自由度，$N^\top = 0$，这一项自然消失。

## 奇异

接近奇异时 $J M^{-1}J^\top$ 的最小特征值趋于零，$\Lambda$ 的对应方向趋于无穷：控制器要在一个“推不动”的方向上产生加速度，力矩急剧增大。本站在求逆前加 $10^{-3}I$ 正则化，相当于给 $\Lambda$ 设上限。SR3-C 的腕部奇异（关节 5 接近 0）在基准中会导致关节空间漂移，所以基准只用 TCP 误差评价任务空间控制器。

## 与其它方法的关系

- **笛卡尔阻抗**是 OSC 去掉 $\Lambda$ 对反馈项的加权（反馈项直接是力），换来被动性；
- **分解速率**是 OSC 的运动学版本；
- **QP 任务空间控制**在 OSC 的基础上加了力矩和限位约束，约束不起作用时两者完全相同；
- **力/位混合控制**在 OSC 的某些方向上把位置控制换成力控制。

## 基准结果

<BenchTable
  track="tracking"
  :conditions="{ nominal: '标称模型', payload: '1.5 kg 未知负载', friction: '关节干摩擦 2 %' }"
  :rows="['resolved_rate', 'cart_impedance', 'osc', 'qp', 'ctc']"
  :columns="[
    { key: 'tcp_max_mm', label: '最大 TCP 误差', unit: 'mm', best: 'min' },
    { key: 'tcp_rms_mm', label: 'RMS', unit: 'mm', best: 'min' },
    { key: 'final_mm', label: '结束误差', unit: 'mm', best: 'min' },
    { key: 'saturation', label: '饱和比例', unit: '%', scale: 100, digits: 1 },
  ]"
/>

## 参考文献

- O. Khatib, "A unified approach for motion and force control of robot manipulators: The operational space formulation," *IEEE J. Robot. Autom.*, 1987.
- J. Nakanishi, R. Cory, M. Mistry, J. Peters, S. Schaal, "Operational space control: A theoretical and empirical comparison," *IJRR*, 2008.
