# 复现指南

::: tip 一句话总结
Python 部分（动力学、控制器、基准、强化学习）用 `uv` 管理环境，需要一块 CUDA GPU；网站部分是 Node 22 + VitePress。所有站上的数据和模型都由下面的命令重新生成。
:::

## 环境

- Linux，NVIDIA GPU（本站结果在 RTX 3090 上得到），CUDA 12.x 驱动；
- Python 3.12，[uv](https://docs.astral.sh/uv/)；
- Node.js ≥ 22（网站）。

```bash
git clone https://github.com/QianYuan1437/RC-Show.git
cd RC-Show
uv sync --extra dev          # torch 2.9 (cu128)、mujoco 4.0、mujoco-warp、warp 等
uv run pytest                # 动力学、轨迹、QP 的单元测试（与 MuJoCo / SciPy 对照）
```

## 基准测试

```bash
uv run python -m rczoo.bench.suite --out site/docs/public/data            # 六条赛道，两台机械臂，每条 256 个世界
uv run python -m rczoo.bench.suite --out site/docs/public/data --tracks learning   # 学习控制赛道（需要先训练策略）
```

- `--robots panda,sr3`、`--tracks tracking,push,...`、`--B 256` 可选；结果写入 `benchmark.json`（指标）和 `examples.json`（示例时间序列），已有的其它赛道结果会保留。
- 全部赛道在 RTX 3090 上约 40 分钟。大部分时间花在每个 1 kHz 控制周期的批量动力学计算上；运行器把一个控制周期捕获成 CUDA graph（<Src path="rczoo/runner.py" symbol="run" />），比逐个启动内核快约 20 倍。
- MPC 每 20 ms 在主机上决定是否重新规划，不能捕获成 CUDA graph，所以只用 64 个世界。

## 强化学习

```bash
uv run python -m rczoo.rl.train --robot panda --iters 1500    # 4096 个并行世界，约 35 分钟
uv run python -m rczoo.rl.train --robot sr3   --iters 1500
uv run python -m rczoo.export.onnx_export                       # 导出 reach.onnx 与训练曲线
```

训练日志在 `runs/reach_<robot>/log.jsonl`，模型在 `runs/reach_<robot>/actor.pt`。

## 网站资源

```bash
uv run python -m rczoo.export.web     # 每台机械臂的网格 GLB、robot.json、plant.xml 与测试夹具
cd site
npm ci
npm test                              # TypeScript 移植与 Python 结果的对照测试 + 浏览器闭环测试
npm run dev                           # 本地预览
npm run build                         # 构建静态站点（输出 docs/.vitepress/dist）
```

`fixtures.json` 包含 Python 计算的动力学量、控制器力矩和轨迹值。网站的单元测试用它们检查 TypeScript 移植：动力学、7 个控制器、4 种轨迹曲线。测试还在 Node 里用 MuJoCo WASM 跑演练场的闭环：每个控制器都要能到达目标，力控制器要能压出 10 N 的接触力。

## 部署

仓库的 `.github/workflows/pages.yml` 在推送到 `main` 时构建并发布到 GitHub Pages（仓库设置 → Pages → Source 选择 “GitHub Actions”）。构建时 `VITEPRESS_BASE` 被设为 `/<仓库名>/`。

## 常见问题

- **Warp 第一次运行很慢**：MuJoCo Warp 第一次运行会编译内核（缓存在 `~/.cache/warp`），之后就快了。
- **显存**：256 个世界的基准约占 3 GB，4096 个世界的强化学习约占 6 GB。
- **结果与站上略有不同**：基准使用固定随机种子，但 GPU 浮点运算的顺序可能不同，统计量应只有很小的差异。
