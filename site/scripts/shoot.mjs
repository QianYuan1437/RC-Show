// Screenshot arbitrary pages of a running site (full page or one element), report console errors.
//
//   node scripts/shoot.mjs http://localhost:5173/ out-dir guide/problem methods/dls[@.selector] ...
//   DARK=1 node scripts/shoot.mjs ...      (dark appearance)
//   SCROLL=1 node scripts/shoot.mjs ...    (viewport shots while scrolling instead of one full-page image)
import { mkdirSync } from 'node:fs'
import puppeteer from 'puppeteer-core'

const [rootArg, outDir = '.screenshots', ...paths] = process.argv.slice(2)
const root = (rootArg ?? 'http://localhost:5173/').replace(/\/?$/, '/')
mkdirSync(outDir, { recursive: true })
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME_PATH ?? '/usr/bin/google-chrome',
  headless: true,
  args: ['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
  defaultViewport: { width: Number(process.env.WIDTH ?? 1440), height: Number(process.env.HEIGHT ?? 1000) },
})
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const problems = []
for (const spec of paths) {
  const [path, selector] = spec.split('@')
  const page = await browser.newPage()
  if (process.env.DARK) await page.evaluateOnNewDocument(() => localStorage.setItem('vitepress-theme-appearance', 'dark'))
  page.on('console', (m) => {
    if (m.type() === 'error') problems.push(`[${path}] console.error: ${m.text()}`)
  })
  page.on('pageerror', (e) => problems.push(`[${path}] pageerror: ${e.message}`))
  page.on('response', (r) => {
    if (r.status() >= 400) problems.push(`[${path}] HTTP ${r.status()} ${r.url()}`)
  })
  await page.goto(root + path, { waitUntil: 'networkidle0', timeout: 120000 })
  await page
    .waitForFunction(() => !document.querySelector('.robot-viewer__overlay, .client-only-placeholder'), {
      timeout: 60000,
    })
    .catch(() => problems.push(`[${path}] demos did not finish loading`))
  await sleep(Number(process.env.WAIT ?? 1500))
  const name = (spec.replace(/[^\w.-]+/g, '_') || 'home').replace(/^_+|_+$/g, '')
  if (selector) {
    const els = await page.$$(selector)
    for (let i = 0; i < els.length; i++) {
      await els[i].scrollIntoView()
      await sleep(600)
      await els[i].screenshot({ path: `${outDir}/${name}-${i}.png` })
    }
    if (!els.length) problems.push(`[${path}] selector not found: ${selector}`)
  } else if (process.env.SCROLL) {
    // Viewport-sized shots while scrolling (full-page captures of tall WebGL pages can tile).
    const total = await page.evaluate(() => document.documentElement.scrollHeight)
    const vh = page.viewport().height
    for (let y = 0, i = 0; y < total; y += vh - 60, i++) {
      await page.evaluate((yy) => window.scrollTo(0, yy), y)
      await sleep(700)
      await page.screenshot({ path: `${outDir}/${name}-s${String(i).padStart(2, '0')}.png` })
    }
  } else {
    await page.screenshot({ path: `${outDir}/${name}.png`, fullPage: true })
  }
  await page.close()
}
await browser.close()
console.log(problems.length ? `${problems.length} problem(s):\n${problems.join('\n')}` : 'no console errors')
