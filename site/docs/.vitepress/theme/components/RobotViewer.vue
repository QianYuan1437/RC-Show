<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue'
import { useData, withBase } from 'vitepress'
import type { Pose } from '../lib/kinematics'
import type { GizmoMode, RobotScene } from '../lib/robotScene'

const props = withDefaults(
  defineProps<{
    robot?: string
    q?: number[]
    height?: string
    gizmo?: boolean
    gizmoMode?: GizmoMode
    trail?: boolean
  }>(),
  { robot: 'panda', height: '460px', gizmo: false, gizmoMode: 'translate', trail: false },
)

const emit = defineEmits<{
  ready: [scene: RobotScene]
  target: [pose: Pose, dragging: boolean]
}>()

const { isDark } = useData()
const host = ref<HTMLDivElement>()
const scene = shallowRef<RobotScene>()
const status = ref<'loading' | 'ready' | 'error'>('loading')
const error = ref('')
let unmounted = false

onMounted(async () => {
  try {
    const { RobotScene } = await import('../lib/robotScene')
    if (!host.value) return
    const s = await RobotScene.create(host.value, {
      assetsUrl: withBase(`/models/${props.robot}/`),
      gizmo: props.gizmo,
      trail: props.trail,
      dark: isDark.value,
    })
    if (unmounted) {
      s.dispose()
      return
    }
    s.setGizmoMode(props.gizmoMode)
    if (props.q) s.setQ(props.q)
    s.onTargetChange((pose, dragging) => emit('target', pose, dragging))
    scene.value = s
    status.value = 'ready'
    emit('ready', s)
  } catch (e) {
    console.error(e)
    error.value = e instanceof Error ? e.message : String(e)
    status.value = 'error'
  }
})

onBeforeUnmount(() => {
  unmounted = true
  scene.value?.dispose()
})

watch(isDark, (d) => scene.value?.setDark(d))
watch(
  () => props.q,
  (q) => q && scene.value?.setQ(q),
)
watch(
  () => props.gizmo,
  (on) => scene.value?.setGizmoEnabled(on),
)
watch(
  () => props.gizmoMode,
  (m) => scene.value?.setGizmoMode(m),
)
watch(
  () => props.trail,
  (on) => scene.value?.setTrailEnabled(on),
)

defineExpose({ scene })
</script>

<template>
  <div class="robot-viewer" :style="{ height }">
    <div ref="host" class="robot-viewer__canvas" />
    <div v-if="status === 'loading'" class="robot-viewer__overlay">正在加载机器人模型…</div>
    <div v-else-if="status === 'error'" class="robot-viewer__overlay robot-viewer__overlay--error">
      加载失败：{{ error }}
    </div>
    <button
      v-if="status === 'ready'"
      class="robot-viewer__reset"
      type="button"
      title="重置视角"
      @click="scene?.resetCamera()"
    >
      重置视角
    </button>
    <slot />
  </div>
</template>

<style scoped>
.robot-viewer {
  position: relative;
  width: 100%;
  border-radius: var(--ik-radius-lg);
  overflow: hidden;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
}
.robot-viewer__canvas {
  position: absolute;
  inset: 0;
}
.robot-viewer__overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--vp-c-text-2);
  font-size: 14px;
  pointer-events: none;
}
.robot-viewer__overlay--error {
  color: var(--vp-c-danger-1);
  padding: 1rem;
  text-align: center;
}
.robot-viewer__reset {
  position: absolute;
  right: 10px;
  bottom: 10px;
  font-size: 12px;
  padding: 3px 10px;
  border-radius: var(--ik-radius-sm);
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  opacity: 0.85;
}
.robot-viewer__reset:hover {
  opacity: 1;
  color: var(--vp-c-text-1);
}
</style>
