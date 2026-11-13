---
layout: home

hero:
  name: RC Zoo
  text: 机械臂控制方法全景
  image:
    src: /images/hero.png
    alt: Franka Panda 沿关节轨迹运动（半透明为途经的位姿，蓝线为末端路径），最终以探针压在平板上（橙色箭头为接触力）
  tagline: 位置、速度、力矩、阻抗——各种控制“模式”背后到底在算什么？在 Franka Panda 与珞石 xMate SR3-C 上统一实现从 PID 到 MPC、iLQR 与强化学习的控制器，批量基准测试，并在浏览器里实时运行
  actions:
    - theme: brand
      text: 开始阅读
      link: /guide/intro
    - theme: alt
      text: 控制演练场
      link: /playground/
    - theme: alt
      text: 基准测试
      link: /benchmark/

features:
  - title: 控制模式拆解
    details: 示教器上的位置模式、速度模式、力矩模式、阻抗模式，分别对应哪条控制律、哪一层回路——对照 Franka FCI 与珞石 xCore 的接口。
    link: /guide/modes
  - title: 关节空间与基于模型的控制
    details: PID、PD + 重力补偿、PD+ 前馈、计算力矩，以及它们对模型误差（未知负载、摩擦）的敏感程度。
    link: /methods/computed-torque
  - title: 柔顺、任务空间与力控
    details: 关节 / 笛卡尔阻抗、操作空间控制、导纳、力/位混合与并联控制，在刚度不同的接触面上比较力跟踪精度。
    link: /methods/force
  - title: 人机交互
    details: 拖动示教（重力 + 摩擦补偿）、基于广义动量观测器的碰撞检测与反应。
    link: /methods/interaction
  - title: 先进控制
    details: 滑模、Slotine–Li 自适应、QP 任务空间控制、MPC、iLQR，以及在 MuJoCo Warp 上大规模并行训练的 PPO 策略。
    link: /methods/robust-adaptive
  - title: 浏览器内实时仿真
    details: MuJoCo WASM 以 1 kHz 运行物理与控制器，可以拖动目标、加负载、推一把、压在平面上，实时看误差、力矩与接触力。
    link: /playground/
---
