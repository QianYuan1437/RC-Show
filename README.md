# RC Zoo：机械臂控制方法全景

在两台力矩控制协作臂（7 自由度 Franka Panda、6 自由度珞石 xMate SR3-C）上，用同一套动力学模型、同一组任务，实现、评测并可视化机械臂的常见控制模式（位置 / 速度 / 力矩 / 阻抗）背后的控制律，以及力控、人机交互、鲁棒与自适应、QP、MPC、iLQR 和强化学习等先进控制方法。基准测试在 [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) 上批量运行；配套的 GitHub Pages 站点逐一讲解每种方法，并用官方 MuJoCo WebAssembly 在浏览器里以 1 kHz 实时运行控制器。

**在线站点：<https://qianyuan1437.github.io/RC-Show/>**（[基准测试结果](https://qianyuan1437.github.io/RC-Show/benchmark/) · [控制演练场](https://qianyuan1437.github.io/RC-Show/playground/)）

| 类别 | 方法 | 实现 |
| --- | --- | --- |
| 关节空间 | 独立关节 PID、PD + 重力补偿（位置模式）、速度模式（外环 P + 内环 PI）、PD+ 前馈、计算力矩 | `rczoo/control/joint.py` |
| 柔顺 / 任务空间 | 关节阻抗、笛卡尔阻抗、分解速率（笛卡尔速度模式）、操作空间控制 OSC（动力学一致零空间） | `rczoo/control/task.py` |
| 力控 | 导纳、力/位混合（导引运动）、力/位并联 | `rczoo/control/task.py` |
| 人机交互 | 拖动示教（重力 + 摩擦补偿、虚拟墙）、广义动量观测器碰撞检测与反应 | `rczoo/control/interaction.py` |
| 鲁棒 / 自适应 | 滑模（边界层）、Slotine–Li 负载自适应（10 参数回归矩阵） | `rczoo/control/joint.py` |
| 优化 | QP 任务空间控制（批量 ADMM）、关节 MPC（反馈线性化 + 预瞄）、iLQR + TVLQR | `rczoo/control/optimal.py`、`rczoo/qp.py` |
| 学习 | PPO 到达策略（关节阻抗内环、负载与摩擦域随机化，MuJoCo Warp 4096 并行世界） | `rczoo/rl/`、`rczoo/control/learned.py` |
| 轨迹 | 五次多项式、梯形、双 S、多轴同步、MoveL / MoveC、过渡、TOPP-RA、在线轨迹生成 | `rczoo/traj.py` |

## 快速开始

```bash
uv sync --extra dev            # Python 3.12, torch 2.9 (cu128), mujoco + mujoco-warp 4.0
uv run pytest                  # 动力学（与 MuJoCo 逐项对照）、轨迹、QP、被控对象

# 基准（RTX 3090 上约 40 分钟）
uv run python -m rczoo.bench.suite --out site/docs/public/data

# 强化学习（每台机械臂约 35 分钟）与学习赛道
uv run python -m rczoo.rl.train --robot panda
uv run python -m rczoo.rl.train --robot sr3
uv run python -m rczoo.bench.suite --out site/docs/public/data --tracks learning

# 网站资源
uv run python -m rczoo.export.web            # GLB 网格、robot.json、MuJoCo WASM 用的 plant.xml、对照夹具
uv run python -m rczoo.export.onnx_export    # 策略 ONNX 与训练曲线

# 网站
cd site && npm ci
npm test && npm run typecheck   # TS 动力学 / 控制器 / 轨迹与 Python 对照，浏览器闭环测试
npm run dev
```

## 目录

```
rczoo/
  robot.py, dynamics.py    机器人参数；批量 RNEA、M、C（Christoffel）、g、J、负载回归矩阵
  model.py, sim.py         被控对象（负载 / 摩擦 / 接触面）与 MuJoCo Warp、CPU 仿真封装
  traj.py, qp.py           轨迹生成、TOPP-RA、在线轨迹生成；批量 ADMM QP
  control/                 控制器（统一接口：tau = ctrl(state, ref)）
  tasks.py, runner.py      基准任务；闭环运行器（每个控制周期捕获为 CUDA graph）
  bench/suite.py           七条基准赛道
  rl/                      PPO、到达任务环境
  export/                  网站资源导出
site/                      VitePress 站点（zh-CN），控制器的 TypeScript 移植在 docs/.vitepress/theme/lib/
tests/                     Python 单元测试
```

## 机器人模型

- Franka Panda：来自 [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)（Apache-2.0），改为力矩电机并加入探针与六维力传感器。
- 珞石 xMate SR3-C：根据官方 URDF 的惯性参数与网格建模，末端加 60 mm 探针。

## 许可

MIT（机器人网格与模型遵循其各自的许可，见 `rczoo/assets/*/LICENSE`）。
