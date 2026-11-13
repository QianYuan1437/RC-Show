<script setup lang="ts">
/**
 * Small, dependency-free SVG line chart shared by the learned / RL demos:
 * linear or log y axis, crosshair tooltip listing every series, vertical markers (e.g. the RLIK
 * hand-over), shaded x regions and horizontal reference lines (tolerances).
 */
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useData } from 'vitepress'

export interface ChartSeries {
  id: string
  label: string
  /** [light, dark] colours (categorical palette). */
  color: [string, string]
  y: (number | null)[]
  /** Optional x values (default: index). */
  x?: number[]
  dash?: string
  width?: number
}

const props = withDefaults(
  defineProps<{
    series: ChartSeries[]
    logY?: boolean
    xLabel?: string
    yLabel?: string
    /** Unit appended to tooltip values. */
    unit?: string
    height?: number
    markers?: { x: number; label: string }[]
    regions?: { x0: number; x1: number; label?: string }[]
    hlines?: { y: number; label?: string }[]
    /** Highlighted x (e.g. playback position). */
    cursor?: number | null
    xMin?: number
    xMax?: number
    yMin?: number
    yMax?: number
    emptyText?: string
    /** Format of x values in ticks / tooltip. */
    xFormat?: (x: number) => string
    legend?: boolean
    /** Label of the x value in the tooltip (default: xLabel). */
    xTipLabel?: string
  }>(),
  {
    logY: false,
    xLabel: '',
    yLabel: '',
    unit: '',
    height: 220,
    markers: () => [],
    regions: () => [],
    hlines: () => [],
    cursor: null,
    emptyText: '暂无数据',
    legend: true,
  },
)

const { isDark } = useData()
// The SVG is drawn at its real pixel width (text stays at its nominal size).
const root = ref<HTMLElement>()
const width = ref(640)
const W = computed(() => width.value)
let ro: ResizeObserver | null = null
onMounted(() => {
  if (!root.value) return
  const update = () => {
    const w = root.value?.clientWidth ?? 0
    if (w > 0) width.value = Math.max(260, Math.round(w))
  }
  update()
  ro = new ResizeObserver(update)
  ro.observe(root.value)
})
onBeforeUnmount(() => ro?.disconnect())
const H = computed(() => props.height)
const M = { l: 52, r: 14, t: 12, b: 34 }
const FLOOR = 1e-6

const xsOf = (s: ChartSeries) => s.x ?? s.y.map((_, i) => i)
const finite = (v: number | null | undefined): v is number => v !== null && v !== undefined && Number.isFinite(v)
const yVal = (v: number) => (props.logY ? Math.max(v, FLOOR) : v)

const xDomain = computed<[number, number]>(() => {
  let lo = Infinity
  let hi = -Infinity
  for (const s of props.series) {
    const xs = xsOf(s)
    s.y.forEach((v, i) => {
      if (!finite(v)) return
      lo = Math.min(lo, xs[i])
      hi = Math.max(hi, xs[i])
    })
  }
  lo = props.xMin ?? (Number.isFinite(lo) ? lo : 0)
  hi = props.xMax ?? (Number.isFinite(hi) ? hi : 1)
  return [lo, hi > lo ? hi : lo + 1]
})

const yDomain = computed<[number, number]>(() => {
  const vals: number[] = []
  for (const s of props.series) for (const v of s.y) if (finite(v)) vals.push(yVal(v))
  for (const h of props.hlines) vals.push(yVal(h.y))
  if (props.logY) {
    const lo = props.yMin ?? 10 ** Math.floor(Math.log10(Math.min(...vals, 1)))
    const hi = props.yMax ?? 10 ** Math.ceil(Math.log10(Math.max(...vals, lo * 10)))
    return [Math.log10(lo), Math.log10(hi) > Math.log10(lo) ? Math.log10(hi) : Math.log10(lo) + 1]
  }
  let lo = props.yMin ?? Math.min(0, ...vals)
  let hi = props.yMax ?? Math.max(...vals, lo)
  if (!Number.isFinite(lo)) lo = 0
  if (!Number.isFinite(hi) || hi - lo < 1e-9 * Math.max(1, Math.abs(lo))) hi = lo + 1
  if (props.yMax === undefined) hi += (hi - lo) * 0.05
  return [lo, hi]
})

const px = (x: number) => {
  const [a, b] = xDomain.value
  return M.l + ((x - a) / (b - a)) * (W.value - M.l - M.r)
}
const py = (v: number) => {
  const [a, b] = yDomain.value
  const t = props.logY ? Math.log10(Math.max(v, FLOOR)) : v
  return M.t + (1 - (t - a) / (b - a)) * (H.value - M.t - M.b)
}

const colorOf = (s: ChartSeries) => (isDark.value ? s.color[1] : s.color[0])

const paths = computed(() =>
  props.series.map((s) => {
    const xs = xsOf(s)
    let d = ''
    let pen = false
    s.y.forEach((v, i) => {
      if (!finite(v)) {
        pen = false
        return
      }
      const yy = Math.min(Math.max(py(v), M.t - 2), H.value - M.b + 2)
      d += `${pen ? 'L' : 'M'}${px(xs[i]).toFixed(1)},${yy.toFixed(1)}`
      pen = true
    })
    return { id: s.id, d, color: colorOf(s), dash: s.dash, width: s.width ?? 2 }
  }),
)

function niceStep(span: number, count: number): number {
  const raw = span / count
  const mag = 10 ** Math.floor(Math.log10(raw))
  return ([1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag)
}

const fmtNum = (v: number) => {
  const a = Math.abs(v)
  if (a === 0) return '0'
  if (a >= 1e6) return `${(v / 1e6).toFixed(a >= 1e7 ? 0 : 1)}M`
  if (a >= 1e4) return `${(v / 1e3).toFixed(0)}k`
  if (a >= 100) return v.toFixed(0)
  if (a >= 1) return v.toFixed(a >= 10 ? 1 : 2).replace(/\.?0+$/, '')
  if (a >= 0.01) return v.toFixed(3).replace(/\.?0+$/, '')
  return v.toExponential(1)
}
const fmtX = (x: number) => (props.xFormat ? props.xFormat(x) : fmtNum(x))

const yTicks = computed(() => {
  const [a, b] = yDomain.value
  if (props.logY) {
    const step = Math.max(1, Math.ceil((b - a) / 6))
    const out: { v: number; label: string }[] = []
    for (let e = Math.ceil(a); e <= b; e += step) {
      const v = 10 ** e
      out.push({ v, label: e >= -2 && e <= 3 ? fmtNum(v) : `1e${e}` })
    }
    return out
  }
  const step = niceStep(b - a, 4)
  const out: { v: number; label: string }[] = []
  for (let v = Math.ceil(a / step) * step; v <= b + 1e-12; v += step) out.push({ v, label: fmtNum(v) })
  return out
})

const xTicks = computed(() => {
  const [a, b] = xDomain.value
  const step = niceStep(b - a, 5)
  const out: number[] = []
  for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(v)
  return out
})

// ----------------------------------------------------------------------------- hover

const hoverX = ref<number | null>(null)
const activeX = computed(() => hoverX.value ?? props.cursor)

function onMove(ev: PointerEvent) {
  const svg = ev.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const sx = ((ev.clientX - rect.left) / rect.width) * W.value
  const [a, b] = xDomain.value
  const x = a + ((sx - M.l) / (W.value - M.l - M.r)) * (b - a)
  // Snap to the nearest data x of the longest series.
  const ref_ = props.series.reduce<ChartSeries | null>((m, s) => (!m || s.y.length > m.y.length ? s : m), null)
  if (!ref_) return
  const xs = xsOf(ref_)
  let best = xs[0]
  for (const v of xs) if (Math.abs(v - x) < Math.abs(best - x)) best = v
  hoverX.value = Math.min(Math.max(best, a), b)
}

const fmtVal = (v: number) => {
  const a = Math.abs(v)
  if (a === 0) return '0'
  if (a >= 1000) return v.toFixed(0)
  if (a >= 1) return v.toFixed(a >= 100 ? 1 : 2)
  if (a >= 0.01) return v.toFixed(3)
  return v.toExponential(2)
}

const tooltip = computed(() => {
  const x = activeX.value
  if (x === null || x === undefined || !props.series.length) return null
  const rows = props.series
    .map((s) => {
      const xs = xsOf(s)
      let k = -1
      let d = Infinity
      xs.forEach((v, i) => {
        if (finite(s.y[i]) && Math.abs(v - x) < d) {
          d = Math.abs(v - x)
          k = i
        }
      })
      if (k < 0) return null
      const v = s.y[k] as number
      return { id: s.id, label: s.label, color: colorOf(s), v, y: py(yVal(v)), x: px(xs[k]) }
    })
    .filter((r): r is NonNullable<typeof r> => r !== null)
  if (!rows.length) return null
  const sx = px(x)
  const left = sx > W.value * 0.62
  return { sx, rows, left, tx: left ? sx - 10 : sx + 10 }
})

const hasData = computed(() => props.series.some((s) => s.y.some(finite)))
</script>

<template>
  <figure ref="root" class="line-chart">
    <div v-if="legend && series.length > 1" class="line-chart__legend">
      <span v-for="s in series" :key="s.id">
        <i :style="{ borderTopColor: colorOf(s), borderTopStyle: s.dash ? 'dashed' : 'solid' }" />{{ s.label }}
      </span>
    </div>
    <svg
      :viewBox="`0 0 ${W} ${H}`"
      role="img"
      :aria-label="`${yLabel}${xLabel ? ' vs ' + xLabel : ''}`"
      @pointermove="onMove"
      @pointerleave="hoverX = null"
    >
      <g class="regions">
        <g v-for="(r, i) in regions" :key="'r' + i">
          <rect
            :x="px(Math.max(r.x0, xDomain[0]))"
            :y="M.t"
            :width="Math.max(0, px(Math.min(r.x1, xDomain[1])) - px(Math.max(r.x0, xDomain[0])))"
            :height="H - M.t - M.b"
          />
          <text v-if="r.label" :x="px(Math.max(r.x0, xDomain[0])) + 4" :y="M.t + 11">{{ r.label }}</text>
        </g>
      </g>
      <g class="grid">
        <line v-for="t in yTicks" :key="t.v" :x1="M.l" :x2="W - M.r" :y1="py(t.v)" :y2="py(t.v)" />
      </g>
      <g class="axis">
        <text v-for="t in yTicks" :key="'l' + t.v" :x="M.l - 6" :y="py(t.v) + 3.5" text-anchor="end">{{ t.label }}</text>
        <text v-for="x in xTicks" :key="'x' + x" :x="px(x)" :y="H - M.b + 14" text-anchor="middle">{{ fmtX(x) }}</text>
        <text :x="(M.l + W - M.r) / 2" :y="H - 4" text-anchor="middle">{{ xLabel }}</text>
        <text :x="12" :y="(M.t + H - M.b) / 2" text-anchor="middle" :transform="`rotate(-90 12 ${(M.t + H - M.b) / 2})`">
          {{ yLabel }}
        </text>
        <line :x1="M.l" :x2="W - M.r" :y1="H - M.b" :y2="H - M.b" class="baseline" />
      </g>
      <g class="hlines">
        <g v-for="(h, i) in hlines" :key="'h' + i">
          <line :x1="M.l" :x2="W - M.r" :y1="py(h.y)" :y2="py(h.y)" />
          <text v-if="h.label" :x="W - M.r - 2" :y="py(h.y) - 4" text-anchor="end">{{ h.label }}</text>
        </g>
      </g>
      <template v-if="hasData">
        <path
          v-for="p in paths"
          :key="p.id"
          :d="p.d"
          class="line"
          :stroke="p.color"
          :stroke-width="p.width"
          :stroke-dasharray="p.dash"
        />
      </template>
      <text v-else :x="(M.l + W - M.r) / 2" :y="H / 2" text-anchor="middle" class="empty">{{ emptyText }}</text>
      <g class="markers">
        <g v-for="(m, i) in markers" :key="'m' + i">
          <line :x1="px(m.x)" :x2="px(m.x)" :y1="M.t" :y2="H - M.b" />
          <text
            :x="px(m.x) > W * 0.7 ? px(m.x) - 4 : px(m.x) + 4"
            :y="H - M.b - 6"
            :text-anchor="px(m.x) > W * 0.7 ? 'end' : 'start'"
          >
            {{ m.label }}
          </text>
        </g>
      </g>
      <g v-if="tooltip" class="cursor">
        <line :x1="tooltip.sx" :x2="tooltip.sx" :y1="M.t" :y2="H - M.b" />
        <circle v-for="r in tooltip.rows" :key="'c' + r.id" :cx="r.x" :cy="r.y" r="4" :fill="r.color" class="dot" />
        <g class="tip" :transform="`translate(${tooltip.tx}, ${M.t + 4})`">
          <text :text-anchor="tooltip.left ? 'end' : 'start'">
            <tspan x="0" dy="10" class="tip__x">{{ (xTipLabel ?? xLabel) ? `${xTipLabel ?? xLabel} ` : '' }}{{ fmtX(activeX as number) }}</tspan>
            <tspan v-for="r in tooltip.rows" :key="'t' + r.id" x="0" dy="14">
              <tspan class="tip__v">{{ fmtVal(r.v) }}{{ unit ? ' ' + unit : '' }}</tspan>
              <tspan class="tip__l">{{ `\u2002${r.label}` }}</tspan>
            </tspan>
          </text>
        </g>
      </g>
    </svg>
  </figure>
</template>

<style scoped>
.line-chart {
  margin: 0;
  font-size: 12px;
  color: var(--vp-c-text-2);
}
.line-chart__legend {
  display: flex;
  flex-wrap: wrap;
  gap: 2px 14px;
  margin-bottom: 2px;
}
.line-chart__legend i {
  display: inline-block;
  width: 16px;
  height: 0;
  margin-right: 5px;
  vertical-align: middle;
  border-top-width: 2px;
}
svg {
  display: block;
  width: 100%;
  height: auto;
  touch-action: none;
}
.grid line {
  stroke: var(--vp-c-divider);
  stroke-width: 1;
}
.axis text,
.empty {
  fill: var(--vp-c-text-3);
  font-size: 11px;
}
.empty {
  font-size: 13px;
}
.axis .baseline {
  stroke: var(--vp-c-text-3);
}
.regions rect {
  fill: var(--vp-c-default-soft);
  opacity: 0.6;
}
.regions text {
  fill: var(--vp-c-text-3);
  font-size: 11px;
}
.hlines line {
  stroke: var(--vp-c-text-3);
  stroke-dasharray: 4 3;
  opacity: 0.8;
}
.hlines text {
  fill: var(--vp-c-text-3);
  font-size: 10.5px;
}
.markers line {
  stroke: var(--vp-c-text-2);
  stroke-dasharray: 2 3;
}
.markers text {
  fill: var(--vp-c-text-2);
  font-size: 11px;
}
.line {
  fill: none;
  stroke-linejoin: round;
  stroke-linecap: round;
}
.cursor line {
  stroke: var(--vp-c-text-3);
}
.dot {
  stroke: var(--vp-c-bg);
  stroke-width: 2;
}
.tip text {
  font-size: 11.5px;
  paint-order: stroke;
  stroke: var(--vp-c-bg);
  stroke-width: 3.5px;
  stroke-linejoin: round;
}
.tip__x {
  fill: var(--vp-c-text-2);
}
.tip__v {
  fill: var(--vp-c-text-1);
  font-weight: 600;
}
.tip__l {
  fill: var(--vp-c-text-2);
}
</style>
