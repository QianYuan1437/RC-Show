<script setup lang="ts">
/**
 * Interactive control playground: MuJoCo WASM plant at 1 kHz in the browser, the TypeScript
 * ports of the controllers, a draggable target, payload / friction / contact surface / pushes /
 * hand guiding, and live plots. Engine: lib/playground.ts.
 */
import { computed, onBeforeUnmount, ref, shallowRef, watch } from 'vue'
import { withBase } from 'vitepress'
import type * as THREE from 'three'
import LineChart, { type ChartSeries } from './LineChart.vue'
import RobotPicker from './RobotPicker.vue'
import RobotViewer from './RobotViewer.vue'
import UiSelect from './UiSelect.vue'
import type { Pose } from '../lib/kinematics'
import type { Mode, Playground, PlaygroundSpec, ReachMeta, Sample } from '../lib/playground'
import type { GizmoMode, RobotScene } from '../lib/robotScene'
import type { RobotId } from '../lib/robots'

const props = withDefaults(defineProps<{ initialController?: string; initialMode?: Mode; height?: string }>(), {
  initialController: 'osc',
  initialMode: 'track',
  height: '480px',
})

interface CtrlOption {
  id: string
  label: string
  modes: Mode[]
}
const GROUPS: { label: string; items: CtrlOption[] }[] = [
  {
    label: '关节空间（位置 / 速度模式）',
    items: [
      { id: 'pid', label: '独立关节 PID', modes: ['track'] },
      { id: 'pd_grav', label: 'PD + 重力补偿（位置模式）', modes: ['track'] },
      { id: 'velocity', label: '速度模式（外环 P + 内环 PI）', modes: ['track'] },
      { id: 'joint_impedance', label: '关节阻抗', modes: ['track', 'guide'] },
    ],
  },
  {
    label: '基于模型（力矩模式）',
    items: [
      { id: 'pd_ff', label: 'PD+ 前馈', modes: ['track'] },
      { id: 'ctc', label: '计算力矩', modes: ['track'] },
      { id: 'smc', label: '滑模控制', modes: ['track'] },
    ],
  },
  {
    label: '任务空间',
    items: [
      { id: 'resolved_rate', label: '笛卡尔速度模式（分解速率）', modes: ['track'] },
      { id: 'cart_impedance', label: '笛卡尔阻抗', modes: ['track', 'contact', 'guide'] },
      { id: 'osc', label: '操作空间控制 OSC', modes: ['track', 'contact'] },
    ],
  },
  {
    label: '力控',
    items: [
      { id: 'admittance', label: '导纳控制', modes: ['contact'] },
      { id: 'hybrid', label: '力/位混合控制', modes: ['contact'] },
      { id: 'parallel', label: '力/位并联控制', modes: ['contact'] },
    ],
  },
  {
    label: '交互与学习',
    items: [
      { id: 'gravity', label: '纯重力补偿（浮动）', modes: ['track', 'guide'] },
      { id: 'drag', label: '拖动示教（重力 + 摩擦补偿）', modes: ['guide'] },
      { id: 'rl', label: '强化学习策略（PPO）', modes: ['track'] },
    ],
  },
]
const ALL = GROUPS.flatMap((g) => g.items)
const MODES: { id: Mode; label: string; hint: string }[] = [
  { id: 'track', label: '轨迹跟踪', hint: '拖动目标，参考经在线轨迹生成器（限速、限加速度）平滑后交给控制器' },
  { id: 'contact', label: '接触力控', hint: '探针下方 5 cm 出现一块平面；目标被投影到平面上，力控制器以期望力压住并跟随' },
  { id: 'guide', label: '手引导', hint: '目标标记代表“人手”，通过 600 N/m 的弹簧拉动 TCP；看手需要用多大的力' },
]

const robot = ref<RobotId>('panda')
const mode = ref<Mode>(props.initialMode)
const ctrlId = ref(props.initialController)
const payload = ref(0)
const friction = ref(0)
const fricComp = ref(0.8)
const fDesired = ref(10)
const reaction = ref(false)
const running = ref(true)
const gizmoMode = ref<GizmoMode>('translate')
const status = ref<'loading' | 'ready' | 'error'>('loading')
const message = ref('')
const policyState = ref<'idle' | 'loading' | 'ready' | 'missing' | 'error'>('idle')
const live = ref<Sample | null>(null)
const reacted = ref(false)

const options = computed(() =>
  GROUPS.map((g) => ({ ...g, items: g.items.filter((c) => c.modes.includes(mode.value)) })).filter((g) => g.items.length),
)
const ctrlOptions = computed(() =>
  options.value.map((g) => ({
    label: g.label,
    items: g.items.map((c) => {
      const rl = c.id === 'rl'
      const hint = !rl ? undefined : policyState.value === 'missing' ? '模型未导出' : policyState.value === 'error' ? '加载失败' : policyState.value !== 'ready' ? '加载中' : undefined
      return { value: c.id, label: c.label, hint, disabled: rl && policyState.value !== 'ready' }
    }),
  })),
)
const modeHint = computed(() => MODES.find((m) => m.id === mode.value)!.hint)

const pg = shallowRef<Playground>()
let scene: RobotScene | null = null
let dragging = false
let raf = 0
let lastWall = 0
let busy = false
let session: unknown = null
let runOnnx: ((s: unknown, feeds: Record<string, { data: Float32Array; dims: number[] }>) => Promise<Record<string, { data: Float32Array }>>) | null = null
const extras: { surface?: THREE.Mesh; arrow?: THREE.ArrowHelper; three?: typeof THREE } = {}

// rolling plot buffers (50 Hz, last 6 s)
const WINDOW = 300
const buf = ref({ t: [] as number[], eT: [] as number[], eR: [] as number[], tau: [] as number[], f: [] as number[], fRef: [] as number[], r: [] as number[] })
let sampleCountdown = 0

async function onReady(s: RobotScene) {
  scene = s
  status.value = 'loading'
  try {
    const [{ loadMujoco }, { Playground }, three] = await Promise.all([
      import('../lib/mujoco'),
      import('../lib/playground'),
      import('three'),
    ])
    extras.three = three
    const base = withBase(`/models/${robot.value}/`)
    const [mj, spec, xml] = await Promise.all([
      loadMujoco(),
      fetch(`${base}robot.json`).then((r) => r.json() as Promise<PlaygroundSpec>),
      fetch(`${base}plant.xml`).then((r) => r.text()),
    ])
    pg.value?.dispose()
    const p = new Playground(mj, spec, xml)
    pg.value = p
    loadPolicy(p, base)
    applySettings()
    p.setController(ctrlId.value === 'rl' && policyState.value !== 'ready' ? 'osc' : ctrlId.value)
    p.setMode(mode.value)
    s.setTarget(p.target)
    s.onTargetChange((pose: Pose, drag: boolean) => {
      dragging = drag
      p.setTarget(pose)
    })
    addExtras()
    status.value = 'ready'
    lastWall = performance.now()
    cancelAnimationFrame(raf)
    raf = requestAnimationFrame(loop)
  } catch (e) {
    console.error(e)
    message.value = e instanceof Error ? e.message : String(e)
    status.value = 'error'
  }
}

async function loadPolicy(p: Playground, base: string) {
  policyState.value = 'loading'
  try {
    const metaRes = await fetch(`${base}reach.json`)
    if (!metaRes.ok) throw new Error('missing')
    const meta = (await metaRes.json()) as ReachMeta
    const onnx = await import('../lib/onnx')
    session = await onnx.loadModel(`${base}reach.onnx`)
    runOnnx = onnx.run as typeof runOnnx
    p.rlMeta = meta
    policyState.value = 'ready'
    if (ctrlId.value === 'rl') p.setController('rl')
  } catch (e) {
    policyState.value = e instanceof Error && e.message === 'missing' ? 'missing' : 'error'
  }
}

function addExtras() {
  const T = extras.three
  if (!scene || !T) return
  const [sx, sy] = [0.25, 0.25]
  const surface = new T.Mesh(
    new T.BoxGeometry(2 * sx, 2 * sy, 0.02),
    new T.MeshStandardMaterial({ color: 0x8c9db5, transparent: true, opacity: 0.55, roughness: 0.6 }),
  )
  surface.visible = false
  scene.scene.add(surface)
  const arrow = new T.ArrowHelper(new T.Vector3(0, 0, 1), new T.Vector3(), 0.1, 0xeb6834, 0.03, 0.018)
  arrow.visible = false
  scene.scene.add(arrow)
  extras.surface = surface
  extras.arrow = arrow
}

function applySettings() {
  const p = pg.value
  if (!p) return
  p.setPayload(payload.value)
  p.setFriction(friction.value / 100)
  p.fDesired = fDesired.value
  p.collisionReaction = reaction.value
  p.setFrictionCompensation((fricComp.value * friction.value) / 100)
}

async function loop(now: number) {
  raf = requestAnimationFrame(loop)
  const p = pg.value
  if (!p || !scene || busy) return
  const due = Math.min(Math.round((now - lastWall) / 1), 40) // 1 kHz, at most 40 ms per frame
  lastWall = now
  if (!running.value) return
  busy = true
  try {
    for (let k = 0; k < due; k++) {
      if (p.policyDue() && session && runOnnx) {
        const obs = p.policyObservation()
        const out = await runOnnx(session, { obs: { data: obs, dims: [1, obs.length] } })
        if (pg.value !== p) return
        p.applyPolicyAction(out.action.data)
      }
      p.step()
      if (--sampleCountdown <= 0) {
        sampleCountdown = 20
        record(p.sample())
      }
    }
  } finally {
    busy = false
  }
  scene.setQ(p.q)
  if (!dragging) scene.setTarget(p.target)
  updateExtras(p)
  live.value = p.sample()
  reacted.value = p.reacted
}

function record(s: Sample) {
  const b = buf.value
  b.t.push(s.t)
  b.eT.push(s.eTarget)
  b.eR.push(s.eRef)
  b.tau.push(100 * s.tauRel)
  b.f.push(mode.value === 'guide' ? s.fHand : s.fz)
  b.fRef.push(mode.value === 'contact' ? fDesired.value : NaN)
  b.r.push(s.rRel)
  if (b.t.length > WINDOW) for (const k of Object.keys(b) as (keyof typeof b)[]) b[k].splice(0, b[k].length - WINDOW)
  buf.value = { ...b }
}

function updateExtras(p: Playground) {
  const { surface, arrow, three: T } = extras
  if (!surface || !arrow || !T) return
  surface.visible = p.mode === 'contact'
  if (surface.visible) {
    const mp = p.data.mocap_pos as Float64Array
    surface.position.set(mp[0], mp[1], mp[2])
  }
  const tcp = p.tcp()
  const f = p.mode === 'guide' ? p.handForce : [p.lastFt[0], p.lastFt[1], p.lastFt[2]]
  const mag = Math.hypot(f[0], f[1], f[2])
  arrow.visible = mag > 0.5
  if (arrow.visible) {
    // contact force drawn where it acts (the probe), pointing as the environment pushes
    arrow.position.set(tcp.p[0], tcp.p[1], tcp.p[2])
    arrow.setDirection(new T.Vector3(f[0] / mag, f[1] / mag, f[2] / mag))
    arrow.setLength(Math.min(0.02 + 0.006 * mag, 0.4), 0.03, 0.018)
  }
}

function resetArm() {
  const p = pg.value
  if (!p) return
  p.reset(p.limits.home)
  scene?.clearTrail()
  clearPlots()
}

function clearPlots() {
  buf.value = { t: [], eT: [], eR: [], tau: [], f: [], fRef: [], r: [] }
}

watch(mode, (m) => {
  if (!ALL.find((c) => c.id === ctrlId.value)!.modes.includes(m)) {
    ctrlId.value = m === 'contact' ? 'hybrid' : m === 'guide' ? 'drag' : 'osc'
  }
  const p = pg.value
  if (!p) return
  p.setController(ctrlId.value)
  p.setMode(m)
  applySettings()
  scene?.setTarget(p.target)
  clearPlots()
})
watch(ctrlId, (id) => {
  const p = pg.value
  if (!p) return
  p.setController(id === 'rl' && policyState.value !== 'ready' ? 'osc' : id)
  if (p.mode === 'contact') p.setTarget(p.target)
  applySettings()
})
watch([payload, friction, fricComp, fDesired, reaction], () => {
  applySettings()
  const p = pg.value
  if (p && p.mode === 'contact') p.setTarget(p.target)
})
watch(robot, () => {
  status.value = 'loading'
  pg.value?.dispose()
  pg.value = undefined
  scene = null
  clearPlots()
})

onBeforeUnmount(() => {
  cancelAnimationFrame(raf)
  pg.value?.dispose()
})

const fmt = (x: number | undefined, d = 1) => (x === undefined || !Number.isFinite(x) ? '—' : x.toFixed(d))
const colors = {
  a: ['#2a78d6', '#5fa0f0'] as [string, string],
  b: ['#eb6834', '#f08a5d'] as [string, string],
  c: ['#7a5af8', '#a48bfa'] as [string, string],
  d: ['#8a8f98', '#9aa0a6'] as [string, string],
}
const errSeries = computed<ChartSeries[]>(() => [
  { id: 'et', label: '到目标', color: colors.a, x: buf.value.t, y: buf.value.eT },
  { id: 'er', label: '到参考', color: colors.b, x: buf.value.t, y: buf.value.eR },
])
const forceSeries = computed<ChartSeries[]>(() => {
  const b = buf.value
  if (mode.value === 'track')
    return [
      { id: 'tau', label: '最大力矩占比', color: colors.c, x: b.t, y: b.tau },
      { id: 'r', label: '观测器 |r|/阈值 ×100', color: colors.d, x: b.t, y: b.r.map((v) => 100 * v) },
    ]
  if (mode.value === 'contact')
    return [
      { id: 'f', label: '接触力 f_z', color: colors.b, x: b.t, y: b.f },
      { id: 'fd', label: '期望力', color: colors.d, x: b.t, y: b.fRef, dash: '4 3' },
    ]
  return [{ id: 'f', label: '手的拉力', color: colors.b, x: b.t, y: b.f }]
})
const errMax = computed(() => {
  let m = 1
  for (const v of buf.value.eT) if (v > m) m = v
  for (const v of buf.value.eR) if (v > m) m = v
  return 1.1 * m
})
const forceLabel = computed(() => (mode.value === 'track' ? '%' : 'N'))
const xRange = computed(() => {
  const t = buf.value.t
  const t1 = t.length ? t[t.length - 1] : 6
  return { min: Math.max(0, t1 - 6), max: Math.max(6, t1) }
})
</script>

<template>
  <div class="cp">
    <div class="cp__top">
      <div class="cp__view" :style="{ minHeight: height }">
        <RobotViewer :key="robot" :robot="robot" height="100%" gizmo trail :gizmo-mode="gizmoMode" @ready="onReady" />
        <div class="cp__legend">
          <span>坐标轴：目标（{{ mode === 'guide' ? '人手' : '可拖动' }}）</span>
          <span v-if="mode !== 'track'"><i style="background: #eb6834" />{{ mode === 'guide' ? '手的拉力' : '接触力' }}</span>
        </div>
        <div v-if="status === 'loading'" class="cp__overlay">正在加载 MuJoCo WASM 与模型…</div>
        <div v-if="live" class="cp__clock">t = {{ live.t.toFixed(1) }} s · 1 kHz</div>
        <div v-if="reacted" class="cp__alarm">检测到碰撞 → 已切换为重力补偿 + 阻尼</div>
      </div>
      <aside class="cp__panel">
        <RobotPicker v-model="robot" />
        <div v-if="message" class="notice notice--error"><b>错误</b><span>{{ message }}</span></div>
        <div class="field">
          <span>场景</span>
          <div class="seg" role="radiogroup" aria-label="场景">
            <button v-for="m in MODES" :key="m.id" type="button" role="radio" :aria-checked="mode === m.id" :class="{ on: mode === m.id }" @click="mode = m.id">
              {{ m.label }}
            </button>
          </div>
        </div>
        <p class="hint">{{ modeHint }}</p>
        <label class="field">
          <span>控制器</span>
          <UiSelect v-model="ctrlId" :options="ctrlOptions" aria-label="控制器" />
        </label>
        <label class="slider">
          <span>未建模负载</span>
          <input v-model.number="payload" type="range" min="0" max="3" step="0.1" />
          <em>{{ payload.toFixed(1) }} kg</em>
        </label>
        <label class="slider">
          <span>关节干摩擦</span>
          <input v-model.number="friction" type="range" min="0" max="5" step="0.5" />
          <em>{{ friction.toFixed(1) }} %</em>
        </label>
        <label v-if="mode === 'contact'" class="slider">
          <span>期望接触力</span>
          <input v-model.number="fDesired" type="range" min="2" max="30" step="1" />
          <em>{{ fDesired }} N</em>
        </label>
        <label v-if="ctrlId === 'drag'" class="slider">
          <span>摩擦补偿</span>
          <input v-model.number="fricComp" type="range" min="0" max="1.2" step="0.1" />
          <em>{{ Math.round(fricComp * 100) }} %</em>
        </label>
        <label v-if="mode === 'track'" class="toggle"><input v-model="reaction" type="checkbox" /> 碰撞检测与反应（动量观测器，阈值 6 % 额定力矩）</label>
        <section class="actions">
          <button class="btn primary" :disabled="status !== 'ready'" @click="pg?.push()">推一下（30 N, 0.2 s）</button>
          <button class="btn" :disabled="status !== 'ready'" @click="resetArm">复位</button>
          <button class="btn" :disabled="status !== 'ready'" @click="running = !running">{{ running ? '暂停' : '继续' }}</button>
          <button class="btn" @click="gizmoMode = gizmoMode === 'translate' ? 'rotate' : 'translate'">拖动：{{ gizmoMode === 'translate' ? '平移' : '旋转' }}</button>
        </section>
        <section class="readout">
          <div><span>到目标</span><b>{{ fmt(live?.eTarget, 2) }}</b><em>mm</em></div>
          <div><span>到参考</span><b>{{ fmt(live?.eRef, 2) }}</b><em>mm</em></div>
          <div><span>力矩占比</span><b>{{ fmt(live ? 100 * live.tauRel : undefined, 0) }}</b><em>%</em></div>
          <div v-if="mode === 'contact'"><span>接触力</span><b>{{ fmt(live?.fz, 2) }}</b><em>N</em></div>
          <div v-else-if="mode === 'guide'"><span>手的拉力</span><b>{{ fmt(live?.fHand, 1) }}</b><em>N</em></div>
          <div v-else><span>观测器</span><b>{{ fmt(live?.rRel, 2) }}</b><em>× 阈值</em></div>
        </section>
      </aside>
    </div>
    <div class="cp__charts">
      <LineChart :series="errSeries" x-label="t (s)" y-label="TCP 误差 (mm)" unit=" mm" :height="200" :x-min="xRange.min" :x-max="xRange.max" :y-min="0" :y-max="errMax" :x-format="(x: number) => x.toFixed(1)" />
      <LineChart :series="forceSeries" x-label="t (s)" :y-label="mode === 'track' ? '%' : '力 (N)'" :unit="` ${forceLabel}`" :height="200" :x-min="xRange.min" :x-max="xRange.max" :x-format="(x: number) => x.toFixed(1)" />
    </div>
  </div>
</template>

<style scoped>
.cp {
  container-type: inline-size;
  margin: 16px 0;
  font-size: 13px;
}
.cp__top {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 330px;
  gap: 16px;
}
.cp__view {
  position: relative;
  border-radius: var(--ik-radius-lg);
  overflow: hidden;
}
.cp__view > :first-child {
  position: absolute;
  inset: 0;
}
.cp__legend {
  position: absolute;
  left: 10px;
  top: 8px;
  display: flex;
  flex-wrap: wrap;
  gap: 2px 12px;
  font-size: 12px;
  color: var(--vp-c-text-2);
  pointer-events: none;
}
.cp__legend i {
  display: inline-block;
  width: 10px;
  height: 10px;
  margin-right: 4px;
  border-radius: var(--ik-radius-pill);
  vertical-align: -1px;
}
.cp__overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--vp-c-text-2);
  pointer-events: none;
}
.cp__clock,
.cp__alarm {
  position: absolute;
  left: 10px;
  bottom: 10px;
  padding: 2px 8px;
  border-radius: var(--ik-radius-sm);
  background: var(--vp-c-bg);
  font-size: 12px;
  font-variant-numeric: tabular-nums;
  color: var(--vp-c-text-2);
}
.cp__alarm {
  left: auto;
  right: 10px;
  color: var(--vp-c-danger-1);
  font-weight: 500;
}
.cp__panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.cp__charts {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-top: 12px;
}
.notice {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 10px 12px;
  border-radius: var(--ik-radius-md);
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
  font-size: 12.5px;
}
.notice--error b {
  color: var(--vp-c-danger-1);
}
.field {
  display: grid;
  grid-template-columns: 76px 1fr;
  align-items: center;
  gap: 8px;
  color: var(--vp-c-text-2);
}
.field .seg {
  display: flex;
}
.field .seg button {
  flex: 1;
  padding: 3px 6px;
}
.hint {
  margin: -4px 0 0 84px;
  font-size: 12px;
  line-height: 1.5;
  color: var(--vp-c-text-3);
}
.slider {
  display: grid;
  grid-template-columns: 76px 1fr 52px;
  align-items: center;
  gap: 8px;
  color: var(--vp-c-text-2);
}
.slider em {
  font-style: normal;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.toggle {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  color: var(--vp-c-text-2);
  line-height: 1.5;
}
.toggle input {
  flex: none;
  margin: 3px 0 0;
}
.actions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
}
.btn {
  padding: 5px 8px;
}
.readout {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4px 10px;
  padding: 8px 12px;
  border-radius: var(--ik-radius-md);
  background: var(--vp-c-bg-soft);
}
.readout div {
  display: flex;
  align-items: baseline;
  gap: 4px;
}
.readout span {
  font-size: 11px;
  color: var(--vp-c-text-3);
  min-width: 52px;
}
.readout b {
  font-size: 15px;
  font-variant-numeric: tabular-nums;
}
.readout em {
  font-style: normal;
  font-size: 11.5px;
  color: var(--vp-c-text-2);
}
@container (max-width: 900px) {
  .cp__top {
    grid-template-columns: 1fr;
  }
}
@container (max-width: 600px) {
  .cp__charts {
    grid-template-columns: 1fr;
  }
  .cp__view {
    min-height: 420px !important;
  }
}
</style>
