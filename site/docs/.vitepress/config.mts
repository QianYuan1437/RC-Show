import { fileURLToPath } from 'node:url'
import { type DefaultTheme, defineConfigWithTheme } from 'vitepress'
import { type ExportOptions, mdExportPlugin, writeMarkdownExports } from './mdExport'

const base = process.env.VITEPRESS_BASE || '/'

/** GitHub repository of the project: change it here once, every source link (<Src>) follows. */
export const REPO_URL = 'https://github.com/QianYuan1437/RC-Show'
/** Branch used for source links. */
export const REPO_BRANCH = 'main'
/** Published site (absolute links in the Markdown downloads). */
export const SITE_URL = 'https://qianyuan1437.github.io/RC-Show/'

const SRC_DIR = fileURLToPath(new URL('..', import.meta.url))
const MD_EXPORT: ExportOptions = {
  siteUrl: SITE_URL,
  repoUrl: REPO_URL,
  repoBranch: REPO_BRANCH,
  docsPathInRepo: 'site/docs',
}

export interface ThemeConfig extends DefaultTheme.Config {
  repoUrl: string
  repoBranch: string
  /** Path of the docs directory inside the repository. */
  docsPathInRepo: string
}

export default defineConfigWithTheme<ThemeConfig>({
  base,
  lang: 'zh-CN',
  title: 'RC Zoo',
  description: '机械臂控制方法全景：位置 / 速度 / 力矩 / 阻抗等控制模式背后的算法，力控、交互、MPC、iLQR 与强化学习，在 Franka Panda 与珞石 SR3-C 上统一实现、基准测试与浏览器内交互。',
  cleanUrls: true,
  lastUpdated: true,
  head: [['link', { rel: 'icon', type: 'image/svg+xml', href: `${base}favicon.svg` }]],
  markdown: {
    math: true,
  },
  themeConfig: {
    repoUrl: REPO_URL,
    repoBranch: REPO_BRANCH,
    docsPathInRepo: MD_EXPORT.docsPathInRepo,
    socialLinks: [{ icon: 'github', link: REPO_URL }],
    logo: '/favicon.svg',
    nav: [
      { text: '指南', link: '/guide/intro', activeMatch: '/guide/' },
      { text: '方法', link: '/methods/', activeMatch: '/methods/' },
      { text: '交互演示', link: '/playground/', activeMatch: '/playground/' },
      { text: '基准测试', link: '/benchmark/', activeMatch: '/benchmark/' },
    ],
    sidebar: [
      {
        text: '指南',
        items: [
          { text: '项目介绍', link: '/guide/intro' },
          { text: '控制模式总览', link: '/guide/modes' },
          { text: '动力学基础', link: '/guide/dynamics' },
          { text: '复现指南', link: '/guide/reproduce' },
        ],
      },
      {
        text: '轨迹与参考',
        items: [
          { text: '方法总览', link: '/methods/' },
          { text: '轨迹生成', link: '/methods/trajectory' },
        ],
      },
      {
        text: '关节空间控制',
        items: [
          { text: 'PID 与 PD + 重力补偿', link: '/methods/joint-pd' },
          { text: '速度模式与分解速率', link: '/methods/velocity' },
          { text: '计算力矩与 PD+', link: '/methods/computed-torque' },
        ],
      },
      {
        text: '柔顺与任务空间',
        items: [
          { text: '阻抗控制', link: '/methods/impedance' },
          { text: '操作空间控制 OSC', link: '/methods/osc' },
        ],
      },
      {
        text: '力控与交互',
        items: [
          { text: '导纳与力/位控制', link: '/methods/force' },
          { text: '拖动示教与碰撞检测', link: '/methods/interaction' },
        ],
      },
      {
        text: '先进控制',
        items: [
          { text: '滑模与自适应控制', link: '/methods/robust-adaptive' },
          { text: 'QP 任务空间控制', link: '/methods/qp' },
          { text: '模型预测控制 MPC', link: '/methods/mpc' },
          { text: 'iLQR 与 TVLQR', link: '/methods/ilqr' },
          { text: '强化学习', link: '/methods/rl' },
        ],
      },
      {
        text: '交互演示',
        items: [{ text: '控制演练场', link: '/playground/' }],
      },
      {
        text: '基准测试',
        items: [{ text: '基准测试结果', link: '/benchmark/' }],
      },
    ],
    outline: { level: [2, 3], label: '本页目录' },
    docFooter: { prev: '上一页', next: '下一页' },
    lastUpdated: { text: '最后更新' },
    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '菜单',
    darkModeSwitchLabel: '外观',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索', buttonAriaLabel: '搜索' },
          modal: {
            noResultsText: '没有找到相关结果',
            resetButtonTitle: '清除',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' },
          },
        },
      },
    },
    footer: {
      message: '基于 VitePress 构建 · 每篇介绍页均可下载 Markdown 原文',
    },
  },
  async buildEnd(siteConfig) {
    const n = await writeMarkdownExports(siteConfig.srcDir, siteConfig.outDir, MD_EXPORT)
    console.log(`  Markdown exports: ${n} pages -> ${siteConfig.outDir}/md/`)
  },
  vite: {
    plugins: [mdExportPlugin(SRC_DIR, base, MD_EXPORT)],
    optimizeDeps: {
      exclude: ['onnxruntime-web', '@mujoco/mujoco'],
    },
    build: {
      chunkSizeWarningLimit: 1500,
    },
    worker: { format: 'es' },
  },
})
