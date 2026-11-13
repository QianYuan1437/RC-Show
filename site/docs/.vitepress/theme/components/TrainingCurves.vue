<script setup lang="ts">
/** PPO training curves of the reaching policy (data/training/reach_<robot>.json). */
import { computed, onMounted, ref, watch } from 'vue'
import { withBase } from 'vitepress'
import LineChart, { type ChartSeries } from './LineChart.vue'
import { PALETTE, fetchJson } from '../lib/bench'
import { ROBOTS, type RobotId } from '../lib/robots'

interface Log {
  iter: number[]
  transitions: number[]
  reward: number[]
  final_pos_mm: number[]
  final_rot_deg: number[]
  time_min: number[]
}

const robot = ref<RobotId>('panda')
const log = ref<Log | null>(null)
const error = ref('')
async function load() {
  log.value = null
  error.value = ''
  try {
    log.value = (await fetchJson(withBase(`/data/training/reach_${robot.value}.json`))) as unknown as Log
  } catch {
    error.value = '该机械臂的训练日志尚未导出'
  }
}
onMounted(load)
watch(robot, load)

const x = computed(() => (log.value?.transitions ?? []).map((t) => t / 1e6))
const errSeries = computed<ChartSeries[]>(() =>
  log.value
    ? [
        { id: 'p', label: '回合末位置误差 (mm)', color: PALETTE[0], x: x.value, y: log.value.final_pos_mm },
        { id: 'r', label: '回合末姿态误差 (°)', color: PALETTE[1], x: x.value, y: log.value.final_rot_deg },
      ]
    : [],
)
const rewSeries = computed<ChartSeries[]>(() =>
  log.value ? [{ id: 'rew', label: '平均每步奖励', color: PALETTE[2], x: x.value, y: log.value.reward }] : [],
)
const minutes = computed(() => (log.value ? log.value.time_min[log.value.time_min.length - 1] : 0))
</script>

<template>
  <div class="tc">
    <div class="tc__bar">
      <div class="seg" role="radiogroup" aria-label="机械臂">
        <button v-for="r in ROBOTS" :key="r.id" type="button" role="radio" :aria-checked="robot === r.id" :class="{ on: robot === r.id }" @click="robot = r.id">{{ r.label }}</button>
      </div>
      <span v-if="log" class="tc__meta">{{ (x[x.length - 1] ?? 0).toFixed(0) }} M 步转移 · RTX 3090 上 {{ minutes.toFixed(0) }} 分钟</span>
    </div>
    <p v-if="error" class="tc__msg">{{ error }}</p>
    <div v-else class="tc__grid">
      <LineChart :series="errSeries" log-y x-label="环境步数 (百万)" y-label="误差" :height="220" :x-format="(v: number) => v.toFixed(0)" />
      <LineChart :series="rewSeries" x-label="环境步数 (百万)" y-label="奖励" :height="220" :x-format="(v: number) => v.toFixed(0)" />
    </div>
  </div>
</template>

<style scoped>
.tc {
  container-type: inline-size;
  margin: 16px 0;
  font-size: 13px;
}
.tc__bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px 14px;
  margin-bottom: 8px;
}
.tc__bar .seg button {
  padding: 3px 10px;
}
.tc__meta,
.tc__msg {
  color: var(--vp-c-text-3);
}
.tc__grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
@container (max-width: 640px) {
  .tc__grid {
    grid-template-columns: 1fr;
  }
}
</style>
