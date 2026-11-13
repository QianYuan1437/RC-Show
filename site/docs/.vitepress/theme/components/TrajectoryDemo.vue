<script setup lang="ts">
/** Compare rest-to-rest profiles (quintic / trapezoid / double S) for one axis: p, v, a, j. */
import { computed, ref } from 'vue'
import LineChart, { type ChartSeries } from './LineChart.vue'
import { PALETTE } from '../lib/bench'
import { type Profile1D, doubleS, quintic, trapezoid } from '../lib/traj'

const dist = ref(1.0)
const vmax = ref(1.0)
const amax = ref(2.0)
const jmax = ref(10)

const profiles = computed(() => {
  const tr = trapezoid(0, dist.value, vmax.value, amax.value)
  const ds = doubleS(0, dist.value, vmax.value, amax.value, jmax.value)
  // quintic with the same peak velocity limit: T = 15 h / (8 vmax), and acceleration limit
  // a_peak = 10 h / (sqrt(3) T^2)  ->  the slower of the two
  const T = Math.max((15 * dist.value) / (8 * vmax.value), Math.sqrt((10 * dist.value) / (Math.sqrt(3) * amax.value)))
  const q = quintic(0, dist.value, T)
  return [
    { id: 'quintic', label: '五次多项式', prof: q },
    { id: 'trap', label: '梯形速度', prof: tr },
    { id: 'ds', label: '双 S（加加速度受限）', prof: ds },
  ] as { id: string; label: string; prof: Profile1D }[]
})
const tEnd = computed(() => Math.max(...profiles.value.map((p) => p.prof.T)) * 1.05 + 0.05)

function series(k: 'p' | 'v' | 'a' | 'j'): ChartSeries[] {
  const N = 400
  const ts = Array.from({ length: N + 1 }, (_, i) => (i / N) * tEnd.value)
  return profiles.value.map((p, i) => ({
    id: p.id,
    label: p.label,
    color: PALETTE[i],
    x: ts,
    // the trapezoid's jerk is a train of Dirac impulses: leave it out of the jerk plot
    y: ts.map((t) => (k === 'j' && p.id === 'trap' ? null : p.prof.at(t)[k])),
  }))
}
const plots = computed(() => [
  { k: 'p' as const, y: '位置', unit: ' rad' },
  { k: 'v' as const, y: '速度', unit: ' rad/s' },
  { k: 'a' as const, y: '加速度', unit: ' rad/s²' },
  { k: 'j' as const, y: '加加速度', unit: ' rad/s³' },
])
</script>

<template>
  <div class="td">
    <div class="td__controls">
      <label><span>距离 h</span><input v-model.number="dist" type="range" min="0.05" max="2" step="0.05" /><em>{{ dist.toFixed(2) }} rad</em></label>
      <label><span>v<sub>max</sub></span><input v-model.number="vmax" type="range" min="0.2" max="3" step="0.1" /><em>{{ vmax.toFixed(1) }} rad/s</em></label>
      <label><span>a<sub>max</sub></span><input v-model.number="amax" type="range" min="0.5" max="10" step="0.5" /><em>{{ amax.toFixed(1) }} rad/s²</em></label>
      <label><span>j<sub>max</sub></span><input v-model.number="jmax" type="range" min="2" max="100" step="2" /><em>{{ jmax }} rad/s³</em></label>
    </div>
    <p class="td__dur">
      时长：<span v-for="p in profiles" :key="p.id">{{ p.label }} <b>{{ p.prof.T.toFixed(3) }} s</b></span>
    </p>
    <div class="td__grid">
      <LineChart v-for="pl in plots" :key="pl.k" :series="series(pl.k)" x-label="t (s)" :y-label="pl.y" :unit="pl.unit" :height="180" :legend="pl.k === 'p'" :x-format="(x: number) => x.toFixed(2)" />
    </div>
  </div>
</template>

<style scoped>
.td {
  container-type: inline-size;
  margin: 16px 0;
  font-size: 13px;
}
.td__controls {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 8px 20px;
}
.td__controls label {
  display: grid;
  grid-template-columns: 56px 1fr 78px;
  align-items: center;
  gap: 8px;
  color: var(--vp-c-text-2);
}
.td__controls em {
  font-style: normal;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.td__dur {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 18px;
  color: var(--vp-c-text-2);
}
.td__dur b {
  font-variant-numeric: tabular-nums;
  color: var(--vp-c-text-1);
}
.td__grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px 16px;
}
@container (max-width: 640px) {
  .td__grid {
    grid-template-columns: 1fr;
  }
}
</style>
