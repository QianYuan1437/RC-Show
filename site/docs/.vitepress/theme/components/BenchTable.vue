<script setup lang="ts">
/**
 * Table of benchmark metrics: rows = controllers (or other keys), columns = metrics, for one
 * robot (selector) and one condition. The best value of each column is highlighted.
 *
 *   <BenchTable track="tracking" :conditions="{ nominal: '标称', payload: '1.5 kg 负载' }"
 *     :columns="[{ key: 'tcp_max_mm', label: '最大误差', unit: 'mm', best: 'min' }]" />
 */
import { computed, onMounted, ref } from 'vue'
import { withBase } from 'vitepress'
import { type Json, at, fetchJson, label } from '../lib/bench'
import { ROBOTS, type RobotId } from '../lib/robots'

interface Column {
  key: string
  label: string
  unit?: string
  digits?: number
  best?: 'min' | 'max' | 'zero'
  /** Multiply before display (e.g. 100 for %). */
  scale?: number
}

const props = defineProps<{
  track: string
  columns: Column[]
  /** Condition keys -> labels (omit if the track has no conditions). */
  conditions?: Record<string, string>
  /** Row keys (default: all in the data, in data order). */
  rows?: string[]
  /** Row labels (default: controller names). */
  rowLabels?: Record<string, string>
  rowHeader?: string
}>()

const data = ref<Json | null>(null)
const error = ref('')
const robot = ref<RobotId>('panda')
const cond = ref(props.conditions ? Object.keys(props.conditions)[0] : '')

onMounted(async () => {
  try {
    data.value = await fetchJson(withBase('/data/benchmark.json'))
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  }
})

const block = computed(() => {
  if (!data.value) return null
  const path = [robot.value, props.track, cond.value].filter(Boolean).join('.')
  return (at(data.value, path) as Record<string, Json> | undefined) ?? null
})
const rowKeys = computed(() => (props.rows ?? Object.keys(block.value ?? {})).filter((k) => block.value?.[k]))
const values = computed(() =>
  rowKeys.value.map((r) => props.columns.map((c) => {
    const v = block.value?.[r]?.[c.key]
    return typeof v === 'number' ? v * (c.scale ?? 1) : null
  })),
)
const bestRow = computed(() =>
  props.columns.map((c, j) => {
    if (!c.best) return -1
    let best = -1
    let bv = 0
    values.value.forEach((row, i) => {
      const v = row[j]
      if (v === null) return
      const score = c.best === 'max' ? -v : c.best === 'zero' ? Math.abs(v) : v
      if (best < 0 || score < bv) {
        best = i
        bv = score
      }
    })
    return best
  }),
)
function fmt(v: number | null, c: Column) {
  if (v === null) return '—'
  if (c.digits !== undefined) return v.toFixed(c.digits)
  const a = Math.abs(v)
  return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : a >= 1 ? v.toFixed(2) : a >= 0.01 ? v.toFixed(3) : v.toPrecision(2)
}
</script>

<template>
  <div class="bt">
    <div class="bt__bar">
      <div class="seg" role="radiogroup" aria-label="机械臂">
        <button v-for="r in ROBOTS" :key="r.id" type="button" role="radio" :aria-checked="robot === r.id" :class="{ on: robot === r.id }" @click="robot = r.id">{{ r.label }}</button>
      </div>
      <div v-if="conditions" class="seg" role="radiogroup" aria-label="条件">
        <button v-for="(l, k) in conditions" :key="k" type="button" role="radio" :aria-checked="cond === k" :class="{ on: cond === k }" @click="cond = String(k)">{{ l }}</button>
      </div>
    </div>
    <p v-if="error" class="bt__msg">{{ error }}</p>
    <p v-else-if="!data" class="bt__msg">加载中…</p>
    <div v-else class="bt__scroll">
      <table>
        <thead>
          <tr>
            <th>{{ rowHeader ?? '控制器' }}</th>
            <th v-for="c in columns" :key="c.key" class="num">{{ c.label }}<small v-if="c.unit"> ({{ c.unit }})</small></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(r, i) in rowKeys" :key="r">
            <td>{{ rowLabels?.[r] ?? label(r) }}</td>
            <td v-for="(c, j) in columns" :key="c.key" class="num" :class="{ best: bestRow[j] === i }">{{ fmt(values[i][j], c) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.bt {
  margin: 16px 0;
  font-size: 13px;
}
.bt__bar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 12px;
  margin-bottom: 10px;
}
.bt__bar .seg button {
  padding: 3px 10px;
}
.bt__msg {
  color: var(--vp-c-text-3);
}
.bt__scroll {
  overflow-x: auto;
  border: 1px solid var(--vp-c-divider);
  border-radius: var(--ik-radius-md);
}
table {
  display: table;
  width: 100%;
  margin: 0;
  border-collapse: collapse;
}
th,
td {
  padding: 6px 12px;
  border: none;
  border-bottom: 1px solid var(--vp-c-divider);
  white-space: nowrap;
}
tbody tr:last-child td {
  border-bottom: none;
}
th {
  background: var(--vp-c-bg-soft);
  font-weight: 500;
  color: var(--vp-c-text-2);
  text-align: left;
}
th small {
  font-weight: 400;
  color: var(--vp-c-text-3);
}
.num {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
td.best {
  font-weight: 600;
  color: var(--vp-c-brand-1);
}
</style>
