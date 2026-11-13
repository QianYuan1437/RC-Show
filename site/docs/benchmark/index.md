---
pageClass: wide-page
aside: false
---

# 基准测试结果

本页把所有控制器放到七条“赛道”上，在两台机械臂上用 MuJoCo Warp 批量仿真比较。表格上方可以切换机械臂和实验条件，最好的值以强调色标出。所有数字和曲线都在浏览器里从基准脚本写出的 JSON 渲染（<Src path="rczoo/bench/suite.py" />），重新运行脚本后会自动更新。

<nav class="bench-toc">
  <a href="#setup">实验设置</a>
  <a href="#tracking">A · 轨迹跟踪</a>
  <a href="#push">B · 抗扰</a>
  <a href="#contact">C · 接触力控</a>
  <a href="#guiding">D · 手引导</a>
  <a href="#collision">E · 碰撞检测</a>
  <a href="#ilqr">F · 能量最优</a>
  <a href="#learning">G · 学习控制</a>
  <a href="#conclusions">结论</a>
</nav>

## 实验设置 {#setup}

- **仿真**：MuJoCo Warp 4.0，物理步长与控制周期 1 ms，单块 RTX 3090；每条赛道 256 个世界（MPC 64 个，iLQR 16 个），每个世界一个随机任务，固定随机种子。
- **被控对象与标称模型**：控制器只使用 MJCF 中的连杆参数。实验条件改变的是被仿真的对象：
  - “负载”是 TCP 下方 5 cm 处一个 8 cm 的实心立方体，模型中没有；
  - “摩擦”是每个关节额定力矩 2 % 的库仑摩擦；
  - 电机的粘性阻尼和转子惯量模型中有。
- **力矩**：所有控制器的输出都被裁剪到额定力矩（Panda 87/12 N·m，SR3-C 66–16 N·m）。
- **统计**：除特别说明外，表中是 256 个世界的**中位数**。“最大 TCP 误差”是每个世界整条轨迹上的最大值再取中位数，P90 是其 90 % 分位数。
- **增益**：所有控制器使用同一套默认增益（<Src path="rczoo/control/gains.py" />），没有针对单项指标调参。

## A · 轨迹跟踪 {#tracking}

随机起止位形之间的五次多项式关节轨迹，时长 1.5 s（每个关节最多转 0.9 rad），然后保持 0.5 s。任务空间控制器跟踪的是同一条运动的正运动学。

<BenchTable
  track="tracking"
  :conditions="{ nominal: '标称模型', payload: '1.5 kg 未知负载', friction: '关节干摩擦 2 %' }"
  :columns="[
    { key: 'tcp_max_mm', label: '最大 TCP 误差', unit: 'mm', best: 'min' },
    { key: 'tcp_max_p90_mm', label: 'P90', unit: 'mm', best: 'min' },
    { key: 'tcp_rms_mm', label: 'RMS', unit: 'mm', best: 'min' },
    { key: 'final_mm', label: '结束误差', unit: 'mm', best: 'min' },
    { key: 'saturation', label: '饱和', unit: '%', scale: 100, digits: 1, best: 'min' },
    { key: 'energy_J', label: '机械能', unit: 'J', best: 'min' },
    { key: 'us_per_step', label: '每步耗时', unit: 'µs', digits: 0 },
  ]"
/>

<BenchChart
  track="tracking"
  :conditions="{ nominal: '标称模型', payload: '1.5 kg 未知负载', friction: '关节干摩擦 2 %' }"
  field="e_tcp_mm"
  :controllers="['pd_grav', 'velocity', 'ctc', 'smc', 'adaptive', 'osc', 'cart_impedance']"
  y-label="TCP 误差 (mm)"
  unit=" mm"
  log-y
/>

“每步耗时”是 256 个世界一起推进一个控制周期（控制器 + 物理）的墙钟时间，不是单个控制器的延迟。

## B · 抗扰 {#push}

保持静止，TCP 受到 30 N、0.2 s 的水平推力（随机方向）。

<BenchTable
  track="push"
  :columns="[
    { key: 'max_dev_mm', label: '最大偏差', unit: 'mm', best: 'min' },
    { key: 'stiffness_N_per_mm', label: '等效刚度', unit: 'N/mm', best: 'max' },
    { key: 'settle_s', label: '恢复时间（1 mm）', unit: 's', best: 'min' },
    { key: 'final_mm', label: '2 s 时误差', unit: 'mm', best: 'min' },
  ]"
/>

## C · 接触力控 {#contact}

从表面上方 5 cm 下降，以 10 N 压住并沿切向滑动 15 cm。软 / 中 / 硬三种表面在 10 N 附近的等效刚度约为 1.4 / 3.7 / 21 kN/m。力误差从期望力斜坡升到 10 N 之后开始统计。

<BenchTable
  track="contact"
  :conditions="{ soft: '软表面', medium: '中等', stiff: '硬表面' }"
  :columns="[
    { key: 'mean_N', label: '平均力', unit: 'N' },
    { key: 'rms_err_N', label: '力 RMS 误差', unit: 'N', best: 'min' },
    { key: 'rms_err_slide_N', label: '滑动段 RMS', unit: 'N', best: 'min' },
    { key: 'peak_N', label: '峰值力', unit: 'N' },
    { key: 'xy_err_mm', label: '切向误差', unit: 'mm', best: 'min' },
    { key: 'lost_contact', label: '脱离接触', unit: '%', scale: 100, digits: 1, best: 'min' },
  ]"
/>

<BenchChart
  track="contact"
  :conditions="{ soft: '软表面', medium: '中等', stiff: '硬表面' }"
  field="fz"
  ref-field="f_ref"
  ref-label="期望力"
  y-label="接触力 (N)"
  unit=" N"
/>

## D · 手引导 {#guiding}

关节带 2 % 额定力矩的干摩擦；“手”通过 600 N/m 的弹簧在 2 s 内把 TCP 水平拉动 30 cm。

<BenchTable
  track="guiding"
  :row-labels="{ joint_impedance: '关节阻抗（未切换到示教）', gravity: '纯重力补偿', drag_nofc: '拖动示教，无摩擦补偿', drag: '拖动示教，80 % 摩擦补偿' }"
  :columns="[
    { key: 'mean_force_N', label: '平均拉力', unit: 'N', best: 'min' },
    { key: 'peak_force_N', label: '峰值拉力', unit: 'N', best: 'min' },
    { key: 'final_lag_mm', label: '结束时滞后', unit: 'mm', best: 'min' },
    { key: 'drift_mm', label: '松手后漂移', unit: 'mm', best: 'min' },
  ]"
/>

## E · 碰撞检测 {#collision}

计算力矩控制器跟踪关节轨迹，动量观测器（$K_O = 50$ s⁻¹）在后台运行。偶数号世界在随机时刻受到 40 N、0.15 s 的冲击，奇数号世界不受冲击。在冲击之前就报警也算误报。

<BenchTable
  track="collision"
  :conditions="{ nominal: '标称负载', payload: '1 kg 负载（未输入模型）', payload_id: '1 kg 负载（已输入模型）' }"
  row-header="阈值（额定力矩的比例）"
  :row-labels="{ '0.03': '3 %', '0.05': '5 %', '0.1': '10 %', '0.2': '20 %' }"
  :columns="[
    { key: 'detection_rate', label: '检测率', unit: '%', scale: 100, digits: 0, best: 'max' },
    { key: 'false_alarm_rate', label: '误报率', unit: '%', scale: 100, digits: 0, best: 'min' },
    { key: 'delay_ms', label: '检测延迟', unit: 'ms', digits: 1, best: 'min' },
  ]"
/>

## F · 能量最优 {#ilqr}

同样的起止位形、同样 1 s：五次多项式 + 计算力矩，对比 iLQR 优化的轨迹（用计算力矩或 TVLQR 执行）。

<BenchTable
  track="ilqr"
  :rows="['quintic_ctc', 'ilqr_ctc', 'ilqr_tvlqr']"
  row-header="方案"
  :columns="[
    { key: 'tau_sq_motion', label: '运动段 ∫‖τ‖²dt', unit: 'N²m²s', best: 'min' },
    { key: 'peak_tau_rel', label: '峰值力矩', unit: '% 额定', scale: 100, digits: 0 },
    { key: 'energy_J', label: '机械能', unit: 'J' },
    { key: 'final_err_deg', label: '终点误差', unit: '°', best: 'min' },
  ]"
/>

## G · 学习控制 {#learning}

到达目标位姿：基于模型的控制器沿 1.5 s 五次多项式到目标位形，PPO 策略只知道目标位姿。成功指 3 s 时位置误差 < 5 mm 且姿态误差 < 2°。

<BenchTable
  track="learning"
  :conditions="{ '0kg': '无负载', '1kg': '1 kg 未知负载', '2kg': '2 kg 未知负载', friction: '关节干摩擦 2 %' }"
  :columns="[
    { key: 'final_mm', label: '终点位置误差', unit: 'mm', best: 'min' },
    { key: 'final_p90_mm', label: 'P90', unit: 'mm', best: 'min' },
    { key: 'final_rot_deg', label: '姿态误差', unit: '°', best: 'min' },
    { key: 'success', label: '成功率', unit: '%', scale: 100, digits: 0, best: 'max' },
    { key: 'reach_s', label: '到达时间（5 mm）', unit: 's', best: 'min' },
    { key: 'energy_J', label: '机械能', unit: 'J', best: 'min' },
  ]"
/>

## 结论 {#conclusions}

1. **前馈决定动态精度，积分 / 鲁棒 / 自适应决定稳态精度。** 有模型前馈的控制器（PD+、计算力矩、OSC、MPC）把动态跟踪误差降到 1 mm 左右；但只要模型里少了负载，它们都会稳态下垂，只有带积分（PID、速度模式）、滑模和自适应的控制器能处理。
2. **“硬”和“软”各有用处。** 滑模和计算力矩被推时几乎不动，阻抗控制让开几厘米；与环境接触时，后者才安全。
3. **力控需要力反馈。** 阻抗控制的接触力取决于环境刚度；混合力/位控制在三种表面上都把力误差控制在 0.15 N 内；导纳控制在硬表面上有较大的冲击。
4. **模型准确是交互功能的前提。** 碰撞检测和拖动示教都直接用模型力矩；负载没有输入模型，碰撞检测几乎全部误报。
5. **优化与学习各有代价。** iLQR 省力但需要离线计算；MPC 能预瞄、处理约束，但精度受同一模型限制；PPO 策略不需要模型、对负载变化稳健，但精度只有毫米级、训练要上亿步。

## 注意事项

- 仿真中的关节力矩被精确施加：没有电机动态、通信延迟、传感器噪声和关节柔性。这些因素在真机上会降低所有高增益控制器的可用带宽。
- 接触面无摩擦（见 [力控](/methods/force)），探针接触是 MuJoCo 的软约束。
- 每个控制器使用一套通用增益。精心调参能改变某些排名，但不会改变上面的结论（例如任何纯 PD 增益都不能消除负载造成的稳态误差）。

<style>
.bench-toc {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 8px;
  margin: 16px 0 8px;
}
.bench-toc a {
  padding: 3px 12px;
  border-radius: var(--ik-radius-pill);
  background: var(--vp-c-default-soft);
  color: var(--vp-c-text-1);
  font-size: 13px;
  text-decoration: none;
}
.bench-toc a:hover {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
}
</style>
