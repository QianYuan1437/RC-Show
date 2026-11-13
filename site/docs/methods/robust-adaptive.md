# 滑模与自适应控制

::: tip 一句话总结
模型总有误差：负载未知、摩擦变化。**滑模控制**承认模型不准，用一个足够大的切换项把误差“压”到滑模面上，换来对任意有界扰动的鲁棒性，代价是需要知道扰动的上界并处理抖振。**自适应控制**（Slotine–Li）利用动力学对惯性参数线性的性质，在线估计负载参数，靠李雅普诺夫函数保证误差收敛——它不只“扛住”了误差，而是把误差辨识出来、补偿掉。
:::

## 共同的出发点：参考速度与滑模变量

两者都基于 Slotine & Li 的参考速度：

$$
\dot q_r = \dot q_d + \Lambda e,\qquad \ddot q_r = \ddot q_d + \Lambda\dot e,\qquad
s = \dot q - \dot q_r = -(\dot e + \Lambda e).
$$

$s = 0$ 是一个“滑模面”：一旦 $s$ 被保持为零，误差就满足 $\dot e = -\Lambda e$，以速率 $\Lambda$ 指数衰减（本站 $\Lambda = 20$ s⁻¹）。于是问题变成**一阶**的：只要让 $s \to 0$。

## 滑模控制

$$
\tau = \hat M\ddot q_r + \hat C(q,\dot q)\dot q_r + \hat g - K_d s - \eta\,\operatorname{sat}(s/\phi).
$$

取 $V = \tfrac12 s^\top M s$，利用 $\dot M - 2C$ 反对称：

$$
\dot V = s^\top\big(\tilde M\ddot q_r + \tilde C\dot q_r + \tilde g + d\big) - s^\top K_d s - \eta\, s^\top\operatorname{sat}(s/\phi),
$$

其中 $\tilde{(\cdot)}$ 是模型误差、$d$ 是其它扰动。只要切换增益 $\eta$ 大于模型误差与扰动的上界，在边界层 $|s_i| > \phi$ 之外 $\dot V < 0$，$s$ 在有限时间内进入边界层。

- **理想的切换项** $\eta\operatorname{sign}(s)$ 在离散控制下会高频抖振（chattering），激发未建模的高频动态；
- **边界层** $\operatorname{sat}(s/\phi)$ 在 $|s|<\phi$ 内变成高增益的线性反馈，抖振消失，误差被限制在 $\phi/\Lambda$ 量级。

本站取 $\eta = 12\,\%\ \tau_{\max}$、$\phi = 0.05$ rad/s（<Src path="rczoo/control/joint.py" symbol="SlidingMode" />）。1.5 kg 未知负载下，滑模的结束误差约 1.2 mm，而计算力矩约 9 mm——残差正是边界层内的有限增益造成的。

## Slotine–Li 自适应控制

把负载对动力学的贡献写成对 10 个惯性参数线性的形式（[动力学基础](/guide/dynamics)）：

$$
M_\pi\ddot q_r + C_\pi\dot q_r + g_\pi = Y(q,\dot q,\dot q_r,\ddot q_r)\,\pi,\qquad
\pi = [m,\ m c_x,\ m c_y,\ m c_z,\ I_{xx}, I_{yy}, I_{zz}, I_{xy}, I_{xz}, I_{yz}] .
$$

控制律与参数更新律：

$$
\tau = M_0\ddot q_r + C_0\dot q_r + g_0 + Y\hat\pi - K_d s,\qquad
\dot{\hat\pi} = -\Gamma\,Y^\top s .
$$

取 $V = \tfrac12 s^\top M s + \tfrac12\tilde\pi^\top\Gamma^{-1}\tilde\pi$，代入后交叉项恰好抵消：

$$
\dot V = -s^\top K_d s \le 0 .
$$

于是 $s \to 0$（Barbalat 引理），跟踪误差收敛。注意：

- **参数不一定收敛到真值**，只有在激励充分（persistently exciting）时才收敛；但控制目标——误差收敛——不需要这个条件；
- $Y$ 必须用 Christoffel 形式的 $C$，否则交叉项不能抵消。本站对 10 个参数各做一次 Slotine–Li 递推得到 $Y$（<Src path="rczoo/dynamics.py" symbol="Dynamics.payload_regressor" />），并在测试中与 MuJoCo 里真实加上负载的模型核对；
- 质量估计被投影到 $\hat m \ge 0$。

实现见 <Src path="rczoo/control/joint.py" symbol="AdaptivePayload" />。在 1.5 kg 未知负载下，自适应控制的结束误差约 0.07 mm，接近无负载时的水平。

## 基准结果

<BenchTable
  track="tracking"
  :conditions="{ payload: '1.5 kg 未知负载', nominal: '标称模型', friction: '关节干摩擦 2 %' }"
  :rows="['pd_grav', 'ctc', 'pid', 'velocity', 'smc', 'adaptive']"
  :columns="[
    { key: 'tcp_max_mm', label: '最大 TCP 误差', unit: 'mm', best: 'min' },
    { key: 'final_mm', label: '结束误差', unit: 'mm', best: 'min' },
    { key: 'tcp_max_p90_mm', label: '最大误差 P90', unit: 'mm', best: 'min' },
    { key: 'energy_J', label: '机械能', unit: 'J', best: 'min' },
  ]"
/>

<BenchChart
  track="tracking"
  :conditions="{ payload: '1.5 kg 未知负载', friction: '关节干摩擦 2 %' }"
  field="e_tcp_mm"
  :controllers="['ctc', 'smc', 'adaptive', 'velocity']"
  y-label="TCP 误差 (mm)"
  unit=" mm"
  log-y
/>

自适应控制在运动开始的几百毫秒里误差和计算力矩一样大（参数还没学到），然后迅速下降；滑模一开始就压住了误差，但始终留着边界层带来的残差。

## 参考文献

- J.-J. E. Slotine, W. Li, *Applied Nonlinear Control*, Prentice Hall, 1991, ch. 7, 9.
- J.-J. E. Slotine, W. Li, "On the adaptive control of robot manipulators," *IJRR*, 1987.
- V. I. Utkin, "Variable structure systems with sliding modes," *IEEE TAC*, 1977.
