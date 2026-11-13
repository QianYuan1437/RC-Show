<script setup lang="ts">
/** Segmented control choosing the robot of an interactive scene. */
import { computed } from 'vue'
import { ROBOTS, type RobotId, robotOption } from '../lib/robots'

const model = defineModel<RobotId>({ required: true })
defineProps<{ disabled?: boolean }>()
const current = computed(() => robotOption(model.value))
</script>

<template>
  <div class="robot-picker">
    <div class="robot-picker__row">
      <span class="robot-picker__label">机械臂</span>
      <div class="seg" role="radiogroup" aria-label="机械臂">
        <button
          v-for="r in ROBOTS"
          :key="r.id"
          type="button"
          role="radio"
          :aria-checked="model === r.id"
          :class="{ on: model === r.id }"
          :disabled="disabled"
          @click="model = r.id"
        >
          {{ r.label }}
        </button>
      </div>
    </div>
    <p class="robot-picker__desc">{{ current.description }}</p>
  </div>
</template>

<style scoped>
.robot-picker {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.robot-picker__row {
  display: grid;
  grid-template-columns: 76px 1fr;
  align-items: center;
  gap: 8px;
}
.robot-picker__label {
  color: var(--vp-c-text-2);
  font-size: 0.8125rem;
}
.robot-picker__desc {
  margin: 0 0 0 84px;
  font-size: 0.75rem;
  line-height: 1.5;
  color: var(--vp-c-text-3);
  text-wrap: pretty;
}
.seg {
  display: flex;
}
.seg button {
  flex: 1;
  padding: 3px 8px;
}
</style>
