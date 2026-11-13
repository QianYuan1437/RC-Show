<script setup lang="ts" generic="T extends string = string">
/**
 * Dropdown of the design system (ui.css): a rounded trigger with the `.btn` skin and a rounded,
 * shadowed list that follows the light / dark theme. Native <select> pop-ups are drawn by the OS
 * (square, unthemed), hence this component; it keeps the native semantics through ARIA
 * (combobox + listbox) and keyboard support (arrows, Home / End, Enter / Space, Esc, Tab, typing).
 *
 *   <UiSelect v-model="id" :options="[{ value: 'a', label: 'A' }, { label: '组', items: [...] }]" />
 */
import { computed, nextTick, onBeforeUnmount, ref, useId, watch } from 'vue'

export interface UiOption<V extends string = string> {
  value: V
  label: string
  disabled?: boolean
  /** Small secondary text after the label. */
  hint?: string
}
export interface UiOptionGroup<V extends string = string> {
  label: string
  items: UiOption<V>[]
}

const props = withDefaults(
  defineProps<{
    options: (UiOption<T> | UiOptionGroup<T>)[]
    disabled?: boolean
    placeholder?: string
    /** Colour of a small marker in the trigger (e.g. the curve colour of a pane). */
    accent?: string
    ariaLabel?: string
  }>(),
  { disabled: false, placeholder: '请选择', accent: undefined, ariaLabel: undefined },
)
const model = defineModel<T>({ required: true })
const emit = defineEmits<{ change: [value: T] }>()

const isGroup = (o: UiOption<T> | UiOptionGroup<T>): o is UiOptionGroup<T> => 'items' in o
const groups = computed<UiOptionGroup<T>[]>(() => {
  const out: UiOptionGroup<T>[] = []
  for (const o of props.options) {
    if (isGroup(o)) out.push(o)
    else if (out.length && out[out.length - 1].label === '') out[out.length - 1].items.push(o)
    else out.push({ label: '', items: [o] })
  }
  return out
})
const flat = computed(() => groups.value.flatMap((g) => g.items))
const current = computed(() => flat.value.find((o) => o.value === model.value))

const uid = useId()
const open = ref(false)
const active = ref(-1)
const trigger = ref<HTMLButtonElement>()
const list = ref<HTMLUListElement>()
const pos = ref({ left: 0, top: 0, width: 0, maxHeight: 320, above: false })
const optionId = (i: number) => `${uid}-opt-${i}`

function place() {
  const r = trigger.value?.getBoundingClientRect()
  if (!r) return
  const gap = 6
  const below = window.innerHeight - r.bottom - gap - 8
  const aboveSpace = r.top - gap - 8
  const want = Math.min(320, (list.value?.scrollHeight ?? 320) + 2)
  const above = below < Math.min(want, 200) && aboveSpace > below
  const maxHeight = Math.max(120, Math.min(320, above ? aboveSpace : below))
  pos.value = {
    left: Math.max(8, Math.min(r.left, window.innerWidth - Math.max(r.width, 180) - 8)),
    top: above ? r.top - gap : r.bottom + gap,
    width: r.width,
    maxHeight,
    above,
  }
}

function firstEnabled(from: number, step: 1 | -1): number {
  const n = flat.value.length
  for (let k = 0, i = from; k < n; k++, i += step) {
    const j = ((i % n) + n) % n
    if (!flat.value[j].disabled) return j
  }
  return -1
}

async function show() {
  if (props.disabled || open.value) return
  open.value = true
  const i = flat.value.findIndex((o) => o.value === model.value)
  active.value = i >= 0 && !flat.value[i].disabled ? i : firstEnabled(0, 1)
  await nextTick()
  place()
  scrollActive()
  window.addEventListener('scroll', place, true)
  window.addEventListener('resize', place)
  document.addEventListener('pointerdown', onOutside, true)
}

function hide(refocus = false) {
  if (!open.value) return
  open.value = false
  window.removeEventListener('scroll', place, true)
  window.removeEventListener('resize', place)
  document.removeEventListener('pointerdown', onOutside, true)
  if (refocus) trigger.value?.focus()
}

function choose(i: number) {
  const o = flat.value[i]
  if (!o || o.disabled) return
  if (o.value !== model.value) {
    model.value = o.value
    emit('change', o.value)
  }
  hide(true)
}

function onOutside(e: PointerEvent) {
  const t = e.target as Node
  if (trigger.value?.contains(t) || list.value?.contains(t)) return
  hide()
}

function scrollActive() {
  nextTick(() => document.getElementById(optionId(active.value))?.scrollIntoView({ block: 'nearest' }))
}

let typed = ''
let typedAt = 0
function onKey(e: KeyboardEvent) {
  if (props.disabled) return
  const k = e.key
  if (!open.value) {
    if (k === 'ArrowDown' || k === 'ArrowUp' || k === 'Enter' || k === ' ') {
      e.preventDefault()
      show()
    }
    return
  }
  if (k === 'Escape') {
    e.preventDefault()
    hide(true)
  } else if (k === 'Tab') {
    hide()
  } else if (k === 'ArrowDown' || k === 'ArrowUp') {
    e.preventDefault()
    const step = k === 'ArrowDown' ? 1 : -1
    active.value = firstEnabled(active.value + step, step)
    scrollActive()
  } else if (k === 'Home' || k === 'End') {
    e.preventDefault()
    active.value = k === 'Home' ? firstEnabled(0, 1) : firstEnabled(flat.value.length - 1, -1)
    scrollActive()
  } else if (k === 'Enter' || k === ' ') {
    e.preventDefault()
    choose(active.value)
  } else if (k.length === 1) {
    // type-ahead on the label
    const now = Date.now()
    typed = now - typedAt > 700 ? k : typed + k
    typedAt = now
    const i = flat.value.findIndex((o) => !o.disabled && o.label.toLowerCase().startsWith(typed.toLowerCase()))
    if (i >= 0) {
      active.value = i
      scrollActive()
    }
  }
}

watch(
  () => props.disabled,
  (d) => d && hide(),
)
onBeforeUnmount(() => hide())

/** Index of an option in the flattened list (for ids / active state inside the groups). */
const indexOf = (o: UiOption<T>) => flat.value.indexOf(o)
</script>

<template>
  <div class="ui-select" :class="{ 'is-open': open, 'is-disabled': disabled }">
    <button
      ref="trigger"
      type="button"
      class="ui-select__trigger"
      role="combobox"
      aria-haspopup="listbox"
      :aria-expanded="open"
      :aria-controls="`${uid}-list`"
      :aria-activedescendant="open && active >= 0 ? optionId(active) : undefined"
      :aria-label="ariaLabel"
      :disabled="disabled"
      @click="open ? hide() : show()"
      @keydown="onKey"
    >
      <i v-if="accent" class="ui-select__accent" :style="{ background: accent }" />
      <span class="ui-select__value" :class="{ 'is-placeholder': !current }">{{ current?.label ?? placeholder }}</span>
      <svg class="ui-select__chevron" viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
        <path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" />
      </svg>
    </button>
    <Teleport to="body">
      <Transition name="ui-select-pop">
        <ul
          v-if="open"
          :id="`${uid}-list`"
          ref="list"
          class="ui-select__list"
          :class="{ above: pos.above }"
          role="listbox"
          :aria-label="ariaLabel"
          :style="{
            left: `${pos.left}px`,
            top: `${pos.top}px`,
            minWidth: `${pos.width}px`,
            maxHeight: `${pos.maxHeight}px`,
          }"
          @mousedown.prevent
        >
          <template v-for="(g, gi) in groups" :key="gi">
            <li v-if="g.label" class="ui-select__group" role="presentation">{{ g.label }}</li>
            <li
              v-for="o in g.items"
              :id="optionId(indexOf(o))"
              :key="o.value"
              :data-value="o.value"
              class="ui-select__option"
              :class="{ 'is-active': indexOf(o) === active, 'is-selected': o.value === model, 'is-disabled': o.disabled }"
              role="option"
              :aria-selected="o.value === model"
              :aria-disabled="o.disabled || undefined"
              @pointermove="!o.disabled && (active = indexOf(o))"
              @click="choose(indexOf(o))"
            >
              <span class="ui-select__label">{{ o.label }}<small v-if="o.hint">{{ o.hint }}</small></span>
              <svg v-if="o.value === model" class="ui-select__check" viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
                <path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" />
              </svg>
            </li>
          </template>
        </ul>
      </Transition>
    </Teleport>
  </div>
</template>

<style scoped>
.ui-select {
  position: relative;
  min-width: 0;
}
.ui-select__trigger {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  min-height: 32px;
  padding: 5px 10px 5px 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: var(--ik-radius-md);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font: inherit;
  font-size: 13px;
  line-height: 1.4;
  text-align: left;
  cursor: pointer;
  transition:
    border-color 0.15s var(--ik-ease),
    box-shadow 0.15s var(--ik-ease),
    background-color 0.15s var(--ik-ease);
}
.ui-select__trigger:hover:not(:disabled),
.is-open .ui-select__trigger {
  border-color: var(--vp-c-brand-1);
}
.is-open .ui-select__trigger {
  box-shadow: 0 0 0 3px var(--vp-c-brand-soft);
}
.ui-select__trigger:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 2px;
}
.ui-select__trigger:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.ui-select__accent {
  flex: none;
  width: 10px;
  height: 10px;
  border-radius: var(--ik-radius-pill);
}
.ui-select__value {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ui-select__value.is-placeholder {
  color: var(--vp-c-text-3);
}
.ui-select__chevron {
  flex: none;
  color: var(--vp-c-text-2);
  transition: transform 0.18s var(--ik-ease);
}
.is-open .ui-select__chevron {
  transform: rotate(180deg);
  color: var(--vp-c-brand-1);
}

.ui-select__list {
  position: fixed;
  z-index: 100;
  box-sizing: border-box;
  max-width: min(420px, calc(100vw - 16px));
  margin: 0;
  padding: 4px;
  overflow-y: auto;
  overscroll-behavior: contain;
  list-style: none;
  border: 1px solid var(--vp-c-divider);
  border-radius: var(--ik-radius-md);
  background: var(--vp-c-bg-elv);
  box-shadow: var(--ik-shadow-2);
  font-size: 13px;
  line-height: 1.4;
  color: var(--vp-c-text-1);
  scrollbar-width: thin;
  scrollbar-color: var(--ik-scrollbar-thumb) transparent;
}
.ui-select__list.above {
  transform: translateY(-100%);
}
.ui-select__group {
  padding: 8px 10px 4px;
  font-size: 11.5px;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: var(--vp-c-text-3);
}
.ui-select__group:not(:first-child) {
  margin-top: 4px;
  border-top: 1px solid var(--vp-c-divider);
}
.ui-select__option {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  border-radius: var(--ik-radius-sm);
  cursor: pointer;
  user-select: none;
  transition: background-color 0.1s var(--ik-ease);
}
.ui-select__option.is-active {
  background: var(--vp-c-default-soft);
}
.ui-select__option.is-selected {
  color: var(--vp-c-brand-1);
  font-weight: 500;
}
.ui-select__option.is-selected.is-active,
.ui-select__option.is-selected {
  background: var(--vp-c-brand-soft);
}
.ui-select__option.is-disabled {
  opacity: 0.45;
  cursor: not-allowed;
}
.ui-select__label {
  flex: 1;
  min-width: 0;
}
.ui-select__label small {
  margin-left: 6px;
  font-size: 11.5px;
  font-weight: 400;
  color: var(--vp-c-text-3);
}
.ui-select__check {
  flex: none;
}

.ui-select-pop-enter-active,
.ui-select-pop-leave-active {
  transition:
    opacity 0.12s var(--ik-ease),
    scale 0.12s var(--ik-ease);
}
.ui-select-pop-enter-from,
.ui-select-pop-leave-to {
  opacity: 0;
  scale: 0.97;
}
@media (prefers-reduced-motion: reduce) {
  .ui-select-pop-enter-active,
  .ui-select-pop-leave-active,
  .ui-select__chevron {
    transition: none;
  }
}
</style>
