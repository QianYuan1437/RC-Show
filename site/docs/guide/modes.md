# 控制模式总览

::: tip 一句话总结
“位置模式”“速度模式”“力矩模式”“阻抗模式”说的是**用户在哪一层回路下指令**；每种模式下，控制器内部仍然要把指令变成关节力矩。本页把这几层回路拆开，给出每种模式对应的控制律，并对照 Franka FCI 与珞石 xCore 的接口。
:::

## 控制回路的分层结构

一台力矩控制的协作臂，从“我想让末端去那里”到电机电流，大致经过下面几层（频率为典型值）：

| 层 | 频率 | 输入 → 输出 | 本站对应 |
| --- | --- | --- | --- |
| 任务 / 运动规划 | 事件驱动 | 目标位姿、路径点 → 轨迹 | [轨迹生成](/methods/trajectory)、MPC、iLQR |
| 轨迹插补 | 1 kHz | 轨迹 → 每周期的参考 $q_d,\dot q_d,\ddot q_d$ 或 $x_d,\dot x_d,\ddot x_d$ | 在线轨迹生成、TOPP-RA |
| **控制律** | 1 kHz | 参考 + 测量 → 期望关节力矩 $\tau$ | 本站所有控制器 |
| 关节力矩环 | 数 kHz | $\tau$ + 关节力矩传感器 → 电流指令 | 仿真中视为理想 |
| 电流环（驱动器） | 10–40 kHz | 电流指令 → PWM | 仿真中视为理想 |

所谓“模式”，就是用户把指令接到哪一层：

- **位置模式**：用户给 $q_d$（或 $x_d$），控制器内部用位置闭环算出 $\tau$；
- **速度模式**：用户给 $\dot q_d$（或末端速度），控制器内部用速度闭环算出 $\tau$；
- **力矩模式**：用户直接给 $\tau$（通常机器人还会叠加它自己的重力 / 摩擦补偿）；
- **阻抗模式**：用户给 $q_d$（或 $x_d$）和刚度、阻尼，控制器让机械臂表现得像围绕参考的弹簧-阻尼器。

本站在仿真里把最下面两层视为理想（关节力矩被精确施加），从 1 kHz 的控制律这一层开始实现。真实的力矩控制机械臂开放的也是这一层：Franka 的力矩接口和 xCore 的实时力矩模式都要求用户每 1 ms 给出关节力矩。

## 位置模式

工业机器人传统的位置模式是**伺服驱动器内的串级 PID**：位置环 P → 速度环 PI → 电流环 PI，每个关节独立，减速比大、负载相对电机惯量小，各关节之间的耦合可以当作扰动。

力矩控制的协作臂在 1 kHz 层面实现位置模式，最基本的控制律是 **PD + 重力补偿**（Takegaki & Arimoto 1981）：

$$
\tau = K_p (q_d - q) + K_d (\dot q_d - \dot q) + g(q).
$$

对定点控制它是全局渐近稳定的，而且不需要惯性参数。不足有两点：

- 没有惯性和科氏力的前馈，快速运动时会滞后；
- 重力项来自模型，负载没有输入模型时会产生 $K_p^{-1}\Delta g$ 的稳态误差（“下垂”）。

前者由 [PD+ / 计算力矩](/methods/computed-torque) 解决，后者由积分项、[滑模或自适应控制](/methods/robust-adaptive) 解决。详见 [PID 与 PD + 重力补偿](/methods/joint-pd)。

## 速度模式

速度模式下，用户指令是关节速度 $\dot q_{\text{cmd}}$。控制器内部有一个**速度闭环**，通常是带重力前馈的 PI：

$$
\tau = K_v (\dot q_{\text{cmd}} - \dot q) + K_i \!\int (\dot q_{\text{cmd}} - \dot q)\,dt + g(q).
$$

速度环的积分项会自动补上未建模的重力和摩擦，所以速度模式对负载变化不敏感，这一点在基准中看得很清楚。用户侧常用一个比例位置外环 $\dot q_{\text{cmd}} = \dot q_d + K_x (q_d - q)$ 把它变回位置跟踪。

**笛卡尔速度模式**把末端速度指令经过（阻尼）伪逆映射成关节速度，即分解速率控制（Whitney 1969）：$\dot q_{\text{cmd}} = J^{+}(\dot x_d + K_x e)$。详见 [速度模式与分解速率](/methods/velocity)。

## 力矩模式

力矩模式把 1 kHz 控制律完全交给用户：每个周期读 $q,\dot q$（以及关节力矩、末端力），发出 $\tau$。本站的 [计算力矩](/methods/computed-torque)、[操作空间控制](/methods/osc)、[力控](/methods/force)、[MPC](/methods/mpc) 等都只能在这一层实现。

实际系统通常会**替用户加上重力补偿**：Franka 的力矩接口中，用户给出的力矩会叠加机器人自己算出的重力和摩擦补偿；在本站的仿真中，控制器需要自己算出完整的 $\tau$（包括 $g(q)$），这样每个控制律都完整可见。

## 阻抗模式

阻抗控制（Hogan 1985）不直接控制位置，而是规定**位置偏差与外力之间的关系**：

$$
M_d \ddot{\tilde x} + D_d \dot{\tilde x} + K_d \tilde x = F_{\text{ext}}.
$$

在力矩控制的机械臂上，最常用的实现不改变惯量（$M_d = \Lambda(q)$），只需要关节力矩传感器，不需要末端力传感器：

$$
\tau = J^\top\!\left(K (x_d - x) + D (\dot x_d - \dot x)\right) + C(q,\dot q)\dot q + g(q).
$$

关节阻抗是同样的思路，只是弹簧在关节空间。阻抗模式是协作机器人的“默认”模式：Franka 的所有运动接口（关节位置、笛卡尔位姿……）其实都在机器人内部由关节或笛卡尔阻抗控制器执行。它的代价是精度：刚度有限，快速运动和负载都会带来误差。详见 [阻抗控制](/methods/impedance)。

**导纳模式**是阻抗的“对偶”：测量外力，经过虚拟质量-弹簧-阻尼器算出位置修正，再交给一个刚性的位置环执行。它适合本身很“硬”的工业机械臂加末端六维力传感器，详见 [导纳与力/位控制](/methods/force)。

## 接口对照

| 用户接口 | Franka FCI（libfranka） | 珞石 xCore SDK | 本站实现 |
| --- | --- | --- | --- |
| 关节位置（实时） | `JointPositions` 运动生成器 + 内部关节 / 笛卡尔阻抗控制器 | `RtControllerMode::jointPosition` | `pd_grav`、`pid`、`ctc` |
| 关节速度（实时） | `JointVelocities` | — | `velocity` |
| 笛卡尔位姿（实时） | `CartesianPose` | `RtControllerMode::cartesianPosition` | `osc`、`cart_impedance`、`qp` |
| 笛卡尔速度（实时） | `CartesianVelocities` | — | `resolved_rate` |
| 关节阻抗 | `ControllerMode::kJointImpedance`（默认） | `RtControllerMode::jointImpedance` | `joint_impedance` |
| 笛卡尔阻抗 | `ControllerMode::kCartesianImpedance` | `RtControllerMode::cartesianImpedance` | `cart_impedance` |
| 力矩（实时） | `Torques` 回调（自动叠加重力、摩擦补偿） | `RtControllerMode::torque` | 全部控制器的最终输出 |
| 非实时运动指令 | — | MoveAbsJ / MoveJ / MoveL / MoveC | [轨迹生成](/methods/trajectory) |
| 拖动示教 | 引导模式（Guiding mode，示教按钮） | 拖动模式（关节 / 笛卡尔空间） | `drag` |
| 碰撞检测 | 碰撞阈值（`setCollisionBehavior`） | 碰撞检测灵敏度设置 | 动量观测器 |

表中的接口名用于对照，具体参数和行为以各厂商文档为准。本站重在说明这些接口**背后**通常是什么控制律，不模拟具体产品的实现细节。

## 动手试试

[控制演练场](/playground/) 里可以在同一个场景中切换上述所有模式：在“轨迹跟踪”场景里拖动目标，对比 PD + 重力补偿（位置模式）、速度模式、关节阻抗与计算力矩；把“未建模负载”拉到 2 kg，看哪一个会下垂。
