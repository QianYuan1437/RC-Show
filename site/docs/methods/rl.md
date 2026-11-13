# 强化学习

::: tip 一句话总结
强化学习不推导控制律，而是在仿真里试错，学到一个从观测到动作的策略。本站在 MuJoCo Warp 上用 4096 个并行世界训练 PPO 策略，让它在**未知负载和关节摩擦**下把末端送到目标位姿；部署方式和真实机器人上的做法一致：策略以 50 Hz 输出关节目标增量，1 kHz 的关节阻抗环负责执行。它不需要动力学模型，但精度和可解释性都不如基于模型的方法。
:::

## 动作空间：为什么不直接输出力矩

在力矩控制的机械臂上部署学习策略，几乎都**不让策略直接输出 1 kHz 的力矩**：

- 策略网络的推理和通信很难稳定地跑到 1 kHz；
- 力矩空间的探索噪声会让机械臂剧烈抖动，仿真到现实（sim-to-real）的差距也最大；
- 低层的 PD / 阻抗环本身就提供了稳定性和柔顺性。

本站的策略以 50 Hz 输出关节目标的**增量**，由 1 kHz 的关节 PD + 名义重力补偿执行（即关节阻抗模式）：

$$
q_{\text{cmd}} \leftarrow \operatorname{clip}\big(q_{\text{cmd}} + s\odot a\big),\quad a \in [-1,1]^n,\qquad
\tau = \operatorname{clip}\big(K_p(q_{\text{cmd}} - q) - K_d\dot q + g_0(q)\big).
$$

增量形式的目标是一个积分器：要托住一个未知的负载，策略只需让 $q_{\text{cmd}}$ 停在 $q$ “上方”一点，而观测中的 $q_{\text{cmd}} - q$（= 负载力矩 / $K_p$）恰好告诉它负载有多重。步长 $s$ 取关节速度限值的 80 %（<Src path="rczoo/rl/reach_env.py" />）。

在 MuJoCo Warp 中，内环用位置执行器实现（增益 $K_p$、偏置 $-K_p q - K_d\dot q$），名义重力补偿用 MuJoCo 的执行器重力补偿（只补偿连杆、不补偿负载），关节力矩范围限制两者之和。

## 观测与奖励

观测（$4n + 9$ 维）：

$$
o = \big[q,\ 0.1\dot q,\ 10(q_{\text{cmd}} - q),\ p,\ 5(p_g - p),\ \operatorname{Log}(R_g R^\top),\ a_{\text{prev}}\big].
$$

奖励（每 20 ms）：

$$
r = -4\lVert e_p\rVert - \lVert e_R\rVert + e^{-\lVert e_p\rVert/0.05} + 2e^{-\lVert e_p\rVert/0.005} + e^{-\lVert e_R\rVert/0.1} + 2e^{-\lVert e_R\rVert/0.02} - 0.01\lVert\Delta a\rVert^2 - 0.002\lVert\dot q\rVert^2 - 5\,(\text{限位惩罚}).
$$

线性项在远处提供梯度，尖锐的指数项奖励精度。早期版本只有指数项，策略学会了把位置做准、却放弃了姿态（回合末姿态误差一直停在约 30°）；加入线性的姿态项后姿态误差降到 1° 以内。

## 训练

- **环境**：起点在关节限位的中间 80 % 内均匀采样，目标是起点附近 ±1 rad 范围内另一个位形的末端位姿（TCP 高于桌面、远离底座）；每回合 3 s（150 步）。
- **域随机化**：每回合随机负载 0–2 kg（在 TCP 下 5 cm 处）和关节干摩擦 0–3 % 额定力矩。
- **PPO**：4096 个并行世界，每次迭代 32 步，网络 256-256-128（ELU），自适应学习率（KL 目标 0.01），1500 次迭代约 2 亿步转移（<Src path="rczoo/rl/ppo.py" />，<Src path="rczoo/rl/train.py" />）。
- 一个策略周期（20 个 MuJoCo 物理步 + 运动学）被捕获成 Warp CUDA graph，RTX 3090 上训练约 35 分钟。

<TrainingCurves />

## 结果

赛道“学习控制”：同样的起止位形，基于模型的控制器沿 1.5 s 的五次多项式跟踪到目标位形，PPO 策略只知道目标位姿（自己决定怎么走），都给 3 s 时间。

<BenchTable
  track="learning"
  :conditions="{ '0kg': '无负载', '1kg': '1 kg 未知负载', '2kg': '2 kg 未知负载', friction: '关节干摩擦 2 %' }"
  :rows="['pd_grav', 'ctc', 'smc', 'adaptive', 'rl']"
  :columns="[
    { key: 'final_mm', label: '终点位置误差', unit: 'mm', best: 'min' },
    { key: 'final_p90_mm', label: 'P90', unit: 'mm', best: 'min' },
    { key: 'final_rot_deg', label: '姿态误差', unit: '°', best: 'min' },
    { key: 'success', label: '成功率（5 mm / 2°）', unit: '%', scale: 100, digits: 0, best: 'max' },
    { key: 'reach_s', label: '到达时间', unit: 's', best: 'min' },
  ]"
/>

- 策略**不依赖负载模型**：在 0–2 kg 之间性能几乎不变（中位误差 Panda 约 3 mm、成功率 73–78 %；SR3-C 约 1.3–2.2 mm、成功率 87–91 %），而 PD + 重力补偿和计算力矩随负载线性变差（2 kg 时 12–22 mm，几乎全部失败）。在 SR3-C 上 2 kg 负载时它甚至比滑模更准。
- 策略自己决定速度：SR3-C 的策略约 0.9 s 就进入 5 mm，比 1.5 s 的五次多项式参考快。
- 但它的精度只有毫米级，远不如自适应控制（0.01–0.03 mm），而且有 10–25 % 的回合没有进入 5 mm（Panda 的 P90 约 10 mm）。奖励是稀疏、带噪的信号，要把误差从 1 mm 推到 0.1 mm，需要比从 10 cm 推到 1 cm 多得多的样本。
- 策略的行为没有稳定性证明，换一个没见过的负载范围或摩擦就可能失效。

## 实践建议

- **残差强化学习**：让策略只学基于模型控制器的修正量（$\tau = \tau_{\text{model}} + \tau_{\text{RL}}$），兼得模型的精度和学习的适应性；
- **学习到的是高层**：把 RL 用在模型难以描述的部分（接触丰富的装配、柔性物体），低层仍用阻抗控制保证安全；
- **sim-to-real**：域随机化（质量、摩擦、延迟、传感器噪声）、动作平滑惩罚、低层阻抗控制都是让策略能部署到真机的关键。

在 [控制演练场](/playground/) 中选择“强化学习策略（PPO）”，策略通过 onnxruntime-web 在浏览器里推理。拖动目标并加上未建模负载，看它怎么调整。

## 参考文献

- J. Schulman et al., "Proximal policy optimization algorithms," arXiv:1707.06347, 2017.
- N. Rudin et al., "Learning to walk in minutes using massively parallel deep reinforcement learning," *CoRL*, 2022.
- T. Johannink et al., "Residual reinforcement learning for robot control," *ICRA*, 2019.
- J. Tobin et al., "Domain randomization for transferring deep neural networks from simulation to the real world," *IROS*, 2017.
