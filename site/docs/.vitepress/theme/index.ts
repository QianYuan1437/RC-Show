import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import { type Component, defineComponent, h, onMounted, shallowRef } from 'vue'
import './custom.css'
import './ui.css'
import PageActions from './components/PageActions.vue'
import Src from './components/Src.vue'

/**
 * Register a browser-only component: it is code-split, never rendered during SSR, and shows
 * a fixed-height placeholder until it has loaded (avoids layout shift).
 */
function clientOnly(name: string, loader: () => Promise<{ default: Component }>, minHeight: string) {
  return defineComponent({
    name: `ClientOnly${name}`,
    inheritAttrs: false,
    setup(_, { attrs, slots }) {
      const comp = shallowRef<Component>()
      onMounted(async () => {
        comp.value = (await loader()).default
      })
      return () =>
        comp.value
          ? h(comp.value, attrs, slots)
          : h('div', { class: 'client-only-placeholder', style: { minHeight } }, '加载中…')
    },
  })
}

export default {
  extends: DefaultTheme,
  // Download / copy-as-Markdown / source actions above every documentation page.
  Layout: () => h(DefaultTheme.Layout, null, { 'doc-before': () => h(PageActions) }),
  enhanceApp({ app }) {
    app.component('Src', Src)
    const lazy: [string, () => Promise<{ default: Component }>, string][] = [
      ['ControlPlayground', () => import('./components/ControlPlayground.vue'), '760px'],
      ['RobotViewer', () => import('./components/RobotViewer.vue'), '460px'],
      ['TrajectoryDemo', () => import('./components/TrajectoryDemo.vue'), '520px'],
      ['BenchTable', () => import('./components/BenchTable.vue'), '320px'],
      ['BenchChart', () => import('./components/BenchChart.vue'), '300px'],
      ['TrainingCurves', () => import('./components/TrainingCurves.vue'), '280px'],
    ]
    for (const [name, loader, minH] of lazy) app.component(name, clientOnly(name, loader, minH))
  },
} satisfies Theme
