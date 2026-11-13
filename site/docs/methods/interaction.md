# 拖动示教与碰撞检测

::: tip 一句话总结
协作机器人的两项标志性功能都只用到关节力矩和动力学模型：**拖动示教**让机械臂在重力和摩擦都被补偿后“失重”，人轻轻一推就能动；**碰撞检测**用广义动量观测器从关节力矩和速度中估计外力矩，不需要加速度、不需要皮肤传感器，几毫秒内就能发现碰撞。两者成败都取决于模型——尤其是负载有没有被正确输入。
:::

## 拖动示教

$$
\tau = g(q) + \hat f_c \tanh(\dot q / v_s) - D\dot q + \tau_{\text{walls}} .
$$

- **重力补偿** $g(q)$ 让机械臂在任意姿态都能停住；
- **摩擦补偿** $\hat f_c\tanh(\dot q/v_s)$：库仑摩擦是操作者感觉“沉”的主要原因。用平滑的 $\tanh$ 代替 $\operatorname{sign}$，避免在零速附近来回切换引起抖动；补偿量通常取辨识值的 70–90 %，补偿过头会让机械臂“自己跑”；
- **小阻尼** $D\dot q$ 防止松手后漂移；
- **虚拟墙**：关节限位前 5° 开始加弹簧力矩，也可以在笛卡尔空间加一个工作盒（<Src path="rczoo/control/interaction.py" symbol="DragTeaching" />）。

### 基准：手引导

关节带 2 % 额定力矩的干摩擦。一只“手”通过 600 N/m 的弹簧-阻尼连接到 TCP，在 2 s 内把 TCP 沿水平方向拉 30 cm。指标是操作者要用的力。

<BenchTable
  track="guiding"
  :rows="['joint_impedance', 'gravity', 'drag_nofc', 'drag']"
  :row-labels="{ joint_impedance: '关节阻抗（未切换到示教）', gravity: '纯重力补偿', drag_nofc: '拖动示教，无摩擦补偿', drag: '拖动示教，80 % 摩擦补偿' }"
  :columns="[
    { key: 'mean_force_N', label: '平均拉力', unit: 'N', best: 'min' },
    { key: 'peak_force_N', label: '峰值拉力', unit: 'N', best: 'min' },
    { key: 'drift_mm', label: '松手后漂移', unit: 'mm', best: 'min' },
  ]"
/>

<BenchChart
  track="guiding"
  field="force_N"
  :controllers="['joint_impedance', 'gravity', 'drag_nofc', 'drag']"
  y-label="手的拉力 (N)"
  unit=" N"
  :worlds="3"
/>

## 碰撞检测：广义动量观测器

### 为什么用动量

最直接的外力矩估计是 $\hat\tau_{\text{ext}} = M\ddot q + h - \tau$，但需要关节加速度，数值微分噪声很大。De Luca 等（2003）利用广义动量 $p = M(q)\dot q$ 的动力学

$$
\dot p = \tau + C^\top(q,\dot q)\dot q - g(q) - \tau_f + \tau_{\text{ext}}
$$

（这里用到了 $\dot M = C + C^\top$），定义残差

$$
r(t) = K_O\left(p(t) - p(0) - \int_0^t \big(\tau + C^\top\dot q - g - \tau_f + r\big)\,ds\right).
$$

于是 $\dot r = K_O(\tau_{\text{ext}} - r)$：**$r$ 是外力矩经过一阶低通（带宽 $K_O$）的估计**，只用到 $q,\dot q,\tau$，没有加速度、没有 $M^{-1}$（<Src path="rczoo/control/interaction.py" symbol="MomentumObserver" />）。

### 检测与反应

当某个关节 $|r_i|$ 超过阈值即判定碰撞。$r$ 的分布还能告诉我们碰撞发生在哪个连杆上（只有碰撞点之前的关节有响应），$J^{+\top}r$ 可估计末端接触力。检测后的反应策略（Haddadin 等 2017）：

- **停止**：切换到重力补偿 + 阻尼，机械臂“软”下来；
- **反射**：沿 $r$ 的方向施加力矩（$\tau = g + k_r r - D\dot q$），主动远离碰撞（本站默认）。

### 基准：检测率、误报与延迟

关节轨迹运动中，一半的世界在随机时刻、随机方向受到 40 N、0.15 s 的冲击；另一半不受冲击，用来统计误报。观测器带宽 $K_O = 50$ s⁻¹，阈值为额定力矩的百分比。

<BenchTable
  track="collision"
  :conditions="{ nominal: '标称负载', payload: '1 kg 负载（未输入模型）', payload_id: '1 kg 负载（已输入模型）' }"
  row-header="阈值（额定力矩的比例）"
  :row-labels="{ '0.03': '3 %', '0.05': '5 %', '0.1': '10 %', '0.2': '20 %' }"
  :columns="[
    { key: 'detection_rate', label: '检测率', unit: '%', scale: 100, digits: 0, best: 'max' },
    { key: 'false_alarm_rate', label: '误报率', unit: '%', scale: 100, digits: 0, best: 'min' },
    { key: 'delay_ms', label: '检测延迟（中位数）', unit: 'ms', digits: 1, best: 'min' },
  ]"
/>

<BenchChart
  track="collision"
  :conditions="{ nominal: '标称负载', payload: '1 kg 负载（未输入模型）' }"
  :fields="{ r_rel_max: 'max |r_i| / 阈值（5 %）', push: '冲击作用中', hit: '已判定碰撞' }"
  y-label="观测器残差"
  :worlds="2"
/>

- 标称情况下，3–5 % 的阈值能在约 8 ms 内检测到全部冲击，没有误报；阈值越高越迟钝，20 % 时漏掉了一部分斜向的轻微冲击。
- **负载没有输入模型**时，负载的重力和惯性力全被当成了外力：残差一开始就超过阈值，几乎所有轨迹都误报。这就是 Franka、珞石等机器人都要求用户在安装工具后设置负载质量、质心和惯量的原因。
- 把同样的负载输入模型（或像[自适应控制](./robust-adaptive)那样在线辨识）后，误报消失、检测率恢复。

## 动手试试

[控制演练场](/playground/) 的“手引导”场景里，拖动目标就是用手拉 TCP：对比“拖动示教”与“关节阻抗”需要的拉力，再把关节干摩擦调高、把摩擦补偿调到 0。在“轨迹跟踪”场景勾选“碰撞检测与反应”，点“推一下”；再把未建模负载拉到 1 kg 以上，看误报。

## 参考文献

- A. De Luca, R. Mattone, "Actuator failure detection and isolation using generalized momenta," *ICRA*, 2003.
- A. De Luca, A. Albu-Schäffer, S. Haddadin, G. Hirzinger, "Collision detection and safe reaction with the DLR-III lightweight manipulator arm," *IROS*, 2006.
- S. Haddadin, A. De Luca, A. Albu-Schäffer, "Robot collisions: A survey on detection, isolation, and identification," *IEEE T-RO*, 2017.
