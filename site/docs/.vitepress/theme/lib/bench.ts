/** Benchmark data written by `python -m rczoo.bench.suite` (fetched once, cached). */

export type Json = Record<string, unknown>

const cache = new Map<string, Promise<Json>>()

export function fetchJson(url: string): Promise<Json> {
  let p = cache.get(url)
  if (!p) {
    p = fetch(url).then((r) => {
      if (!r.ok) throw new Error(`无法加载 ${url}（HTTP ${r.status}）`)
      return r.json() as Promise<Json>
    })
    p.catch(() => cache.delete(url))
    cache.set(url, p)
  }
  return p
}

/** Display names of the controllers (shared by tables, charts and the playground). */
export const CTRL_LABELS: Record<string, string> = {
  pid: '独立关节 PID',
  pd_grav: 'PD + 重力补偿',
  velocity: '速度模式',
  joint_impedance: '关节阻抗',
  pd_ff: 'PD+ 前馈',
  ctc: '计算力矩',
  smc: '滑模',
  adaptive: '自适应（Slotine–Li）',
  resolved_rate: '笛卡尔速度（分解速率）',
  cart_impedance: '笛卡尔阻抗',
  osc: '操作空间 OSC',
  qp: 'QP 任务空间',
  mpc: '关节 MPC',
  admittance: '导纳',
  hybrid: '力/位混合',
  parallel: '力/位并联',
  gravity: '纯重力补偿',
  drag: '拖动示教（含摩擦补偿）',
  drag_nofc: '拖动示教（无摩擦补偿）',
  rl: '强化学习（PPO）',
  quintic_ctc: '五次多项式 + 计算力矩',
  ilqr_ctc: 'iLQR 轨迹 + 计算力矩',
  ilqr_tvlqr: 'iLQR 轨迹 + TVLQR',
}

export const label = (id: string) => CTRL_LABELS[id] ?? id

/** Categorical palette ([light, dark]) assigned by position. */
export const PALETTE: [string, string][] = [
  ['#2a78d6', '#5fa0f0'],
  ['#eb6834', '#f08a5d'],
  ['#1a9e74', '#3cc49a'],
  ['#7a5af8', '#a48bfa'],
  ['#d4a017', '#e8c14a'],
  ['#c2408a', '#e06bb0'],
  ['#5b6b7f', '#8a99ab'],
  ['#0e9fb5', '#4cc3d6'],
]

/** Walk a dotted path into a nested object ("tracking.nominal.ctc.tcp_max_mm"). */
export function at(obj: unknown, path: string): unknown {
  return path.split('.').reduce<unknown>((o, k) => (o && typeof o === 'object' ? (o as Json)[k] : undefined), obj)
}
