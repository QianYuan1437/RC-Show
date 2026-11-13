<script setup lang="ts">
/**
 * Time series of example worlds from `examples.json` (written by the benchmark).
 *
 * Either one curve per controller (`field` of every controller block) or, with `fields`, one curve
 * per field of the condition block itself (collision example). Vector-valued samples (per joint)
 * are reduced with `reduce`.
 */
import { computed, onMounted, ref } from 'vue'
import { withBase } from 'vitepress'
import LineChart, { type ChartSeries } from './LineChart.vue'
import { type Json, PALETTE, at, fetchJson, label } from '../lib/bench'
import { ROBOTS, type RobotId } from '../lib/robots'

const props = withDefaults(
  defineProps<{
    track: string
    conditions?: Record<string, string>
    field?: string
    controllers?: string[]
    fields?: Record<string, string>
    /** Reference curve (dashed), a field of the first controller block. */
    refField?: string
    refLabel?: string
    reduce?: 'maxabs' | 'none'
    yLabel?: string
    unit?: string
    logY?: boolean
    height?: number
    yMin?: number
    yMax?: number
    tMax?: number
    /** Number of example worlds offered by the selector. */
    worlds?: number
  }>(),
  { reduce: 'none', yLabel: '', unit: '', logY: false, height: 240, worlds: 1 },
)

const data = ref<Json | null>(null)
const error = ref('')
const robot = ref<RobotId>('panda')
const cond = ref(props.conditions ? Object.keys(props.conditions)[0] : '')
const world = ref(0)

onMounted(async () => {
  try {
    data.value = await fetchJson(withBase('/data/examples.json'))
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  }
})

const block = computed(() => {
  if (!data.value) return null
  const path = [robot.value, props.track, cond.value].filter(Boolean).join('.')
  return (at(data.value, path) as Json | undefined) ?? null
})

function curve(src: unknown): (number | null)[] {
  const rows = src as unknown[] | undefined
  const w = rows?.[Math.min(world.value, rows.length - 1)] as (number | number[] | null)[] | undefined
  if (!w) return []
  return w.map((v) => {
    if (v === null) return null
    if (Array.isArray(v)) return props.reduce === 'maxabs' ? Math.max(...v.map(Math.abs)) : v[0]
    return v
  })
}

const series = computed<ChartSeries[]>(() => {
  const b = block.value
  if (!b) return []
  const out: ChartSeries[] = []
  const tMax = props.tMax ?? Infinity
  const clipT = (t: number[], y: (number | null)[]) => {
    const k = t.findIndex((x) => x > tMax)
    return k < 0 ? { x: t, y } : { x: t.slice(0, k), y: y.slice(0, k) }
  }
  if (props.fields) {
    const t = b.t as number[]
    Object.entries(props.fields).forEach(([f, l], i) => {
      out.push({ id: f, label: l, color: PALETTE[i % PALETTE.length], ...clipT(t, curve(b[f])) })
    })
    return out
  }
  const ids = (props.controllers ?? Object.keys(b)).filter((c) => b[c])
  ids.forEach((c, i) => {
    const cb = b[c] as Json
    out.push({ id: c, label: label(c), color: PALETTE[i % PALETTE.length], ...clipT(cb.t as number[], curve(cb[props.field!])) })
  })
  if (props.refField && ids.length) {
    const cb = b[ids[0]] as Json
    out.push({ id: 'ref', label: props.refLabel ?? '参考', color: ['#8a8f98', '#9aa0a6'], dash: '5 4', ...clipT(cb.t as number[], curve(cb[props.refField])) })
  }
  return out
})
</script>

<template>
  <div class="bc">
    <div class="bc__bar">
      <div class="seg" role="radiogroup" aria-label="机械臂">
        <button v-for="r in ROBOTS" :key="r.id" type="button" role="radio" :aria-checked="robot === r.id" :class="{ on: robot === r.id }" @click="robot = r.id">{{ r.label }}</button>
      </div>
      <div v-if="conditions" class="seg" role="radiogroup" aria-label="条件">
        <button v-for="(l, k) in conditions" :key="k" type="button" role="radio" :aria-checked="cond === k" :class="{ on: cond === k }" @click="cond = String(k)">{{ l }}</button>
      </div>
      <div v-if="worlds > 1" class="seg" role="radiogroup" aria-label="示例">
        <button v-for="w in worlds" :key="w" type="button" role="radio" :aria-checked="world === w - 1" :class="{ on: world === w - 1 }" @click="world = w - 1">示例 {{ w }}</button>
      </div>
    </div>
    <p v-if="error" class="bc__msg">{{ error }}</p>
    <LineChart v-else :series="series" :log-y="logY" x-label="t (s)" :y-label="yLabel" :unit="unit" :height="height" :y-min="yMin" :y-max="yMax" :x-format="(x: number) => x.toFixed(2)" :empty-text="data ? '暂无数据' : '加载中…'" />
  </div>
</template>

<style scoped>
.bc {
  margin: 16px 0;
  font-size: 13px;
}
.bc__bar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 12px;
  margin-bottom: 8px;
}
.bc__bar .seg button {
  padding: 3px 10px;
}
.bc__msg {
  color: var(--vp-c-text-3);
}
</style>
