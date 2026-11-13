// Headless smoke test of a running site (dev or preview server): loads every page, reports
// console errors and failed requests, exercises the control playground and writes screenshots.
//
//   VITEPRESS_BASE=/RC-Show/ npm run build && node scripts/serve-dist.mjs docs/.vitepress/dist /RC-Show/ 4173 &
//   node scripts/check-site.mjs http://localhost:4173/RC-Show/
import { mkdirSync } from 'node:fs'
import puppeteer from 'puppeteer-core'

const root = (process.argv[2] ?? 'http://localhost:4173/').replace(/\/?$/, '/')
const outDir = process.argv[3] ?? '.screenshots'
const chrome = process.env.CHROME_PATH ?? '/usr/bin/google-chrome'
mkdirSync(outDir, { recursive: true })

const browser = await puppeteer.launch({
  executablePath: chrome,
  headless: true,
  args: ['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--window-size=1440,1000'],
  defaultViewport: { width: 1440, height: 1000 },
})

const problems = []
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

async function open(path) {
  const page = await browser.newPage()
  page.on('console', (m) => {
    if (m.type() === 'error' || m.type() === 'warning') problems.push(`[${path}] console.${m.type()}: ${m.text()}`)
  })
  page.on('pageerror', (e) => problems.push(`[${path}] pageerror: ${e.message}`))
  page.on('response', (r) => {
    if (r.status() >= 400) problems.push(`[${path}] HTTP ${r.status()} ${r.url()}`)
  })
  page.on('requestfailed', (r) => problems.push(`[${path}] request failed: ${r.url()} ${r.failure()?.errorText}`))
  await page.goto(root + path, { waitUntil: 'networkidle0', timeout: 120000 })
  return page
}

const clickText = (page, text) =>
  page.evaluate((t) => {
    const b = [...document.querySelectorAll('button')].find((x) => x.textContent.trim().startsWith(t))
    if (!b) throw new Error(`button not found: ${t}`)
    b.click()
  }, text)

const waitText = (page, text, timeout = 60000) =>
  page.waitForFunction((t) => document.body.innerText.includes(t), { timeout }, text)

const readout = (page) =>
  page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout div')].map((d) => [d.querySelector('span')?.textContent, Number(d.querySelector('b')?.textContent)])))

// Every documentation page: renders, no errors, formulas typeset, components mounted
const PAGES = [
  '', 'guide/intro', 'guide/modes', 'guide/dynamics', 'guide/reproduce',
  'methods/', 'methods/trajectory', 'methods/joint-pd', 'methods/velocity', 'methods/computed-torque',
  'methods/impedance', 'methods/osc', 'methods/force', 'methods/interaction', 'methods/robust-adaptive',
  'methods/qp', 'methods/mpc', 'methods/ilqr', 'methods/rl', 'benchmark/',
]
for (const path of PAGES) {
  const page = await open(path)
  await sleep(1500)
  const info = await page.evaluate(() => ({
    placeholders: document.querySelectorAll('.client-only-placeholder').length,
    tables: document.querySelectorAll('.bt table tbody tr').length,
    charts: document.querySelectorAll('svg path').length,
    mathErrors: document.querySelectorAll('mjx-merror').length,
  }))
  if (info.placeholders) problems.push(`[${path}] ${info.placeholders} component(s) still loading`)
  if (info.mathErrors) problems.push(`[${path}] ${info.mathErrors} MathJax error(s)`)
  console.log(`${path || 'home'}: ${info.tables} table rows, ${info.charts} svg paths`)
  await page.screenshot({ path: `${outDir}/${(path || 'home').replace(/\/$/, '').replace(/\//g, '-') || 'index'}.png`, fullPage: true })
  await page.close()
}

// Control playground
{
  const page = await open('playground/')
  await waitText(page, 't = ', 90000)
  await sleep(3000)
  const r0 = await readout(page)
  console.log('playground track:', r0)
  if (!(r0['到目标'] < 1)) problems.push(`[playground] TCP not at rest at start: ${JSON.stringify(r0)}`)
  await page.screenshot({ path: `${outDir}/playground-track.png` })
  await clickText(page, '推一下')
  await sleep(300)
  const r1 = await readout(page)
  console.log('after push:', r1)
  await sleep(2500)
  for (const [mode, ctrl] of [['接触力控', 'hybrid'], ['手引导', 'drag']]) {
    await clickText(page, mode)
    await sleep(500)
    await page.select('.cp select', ctrl)
    await sleep(4000)
    const r = await readout(page)
    console.log(`${mode} (${ctrl}):`, r)
    if (mode === '接触力控' && !(Math.abs(r['接触力'] - 10) < 1)) problems.push(`[playground] contact force ${r['接触力']} N`)
    await page.screenshot({ path: `${outDir}/playground-${ctrl}.png` })
  }
  // learned policy
  await clickText(page, '轨迹跟踪')
  await page.select('.cp select', 'rl')
  await sleep(4000)
  const rl = await readout(page)
  console.log('rl:', rl)
  if (!(rl['到目标'] < 10)) problems.push(`[playground] rl policy does not hold the target: ${JSON.stringify(rl)}`)
  // robot switch
  await clickText(page, '珞石 xMate SR3-C')
  await waitText(page, 't = ', 90000)
  await sleep(3000)
  const sr = await readout(page)
  console.log('sr3:', sr)
  await page.screenshot({ path: `${outDir}/playground-sr3.png` })
  await page.close()
}

// Dark mode + narrow viewport
{
  const page = await browser.newPage()
  await page.evaluateOnNewDocument(() => localStorage.setItem('vitepress-theme-appearance', 'dark'))
  await page.setViewport({ width: 420, height: 900 })
  page.on('pageerror', (e) => problems.push(`[dark mobile] pageerror: ${e.message}`))
  await page.goto(root + 'playground/', { waitUntil: 'networkidle0', timeout: 120000 })
  await waitText(page, 't = ', 90000)
  await sleep(2000)
  await page.screenshot({ path: `${outDir}/playground-dark-mobile.png`, fullPage: true })
  await page.close()
}

await browser.close()
if (problems.length) {
  console.log(`\n${problems.length} problem(s):`)
  for (const p of problems) console.log('  ' + p)
  process.exit(1)
}
console.log('\nno problems')
