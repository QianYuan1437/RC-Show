// Render the home-page hero image (docs/public/images/hero.png) with headless Chrome.
//
//   node scripts/make-hero.mjs
//
// Starts a throw-away Vite server on scripts/hero/ (sharing docs/public for the robot assets),
// renders the Panda moving along a trajectory into contact with a plate (transparent background).
import { mkdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import puppeteer from 'puppeteer-core'
import { createServer } from 'vite'

const site = fileURLToPath(new URL('..', import.meta.url))
const out = `${site}docs/public/images/hero.png`
mkdirSync(`${site}docs/public/images`, { recursive: true })

const server = await createServer({
  root: `${site}scripts/hero`,
  publicDir: `${site}docs/public`,
  logLevel: 'error',
  server: { port: 5287, strictPort: false, fs: { allow: [site] } },
  optimizeDeps: { exclude: ['onnxruntime-web', '@mujoco/mujoco'] },
})
await server.listen()
const url = server.resolvedUrls.local[0]
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME_PATH ?? '/usr/bin/google-chrome',
  headless: true,
  args: ['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
  defaultViewport: { width: 700, height: 640, deviceScaleFactor: 2 },
})
try {
  const page = await browser.newPage()
  page.on('pageerror', (e) => console.error('pageerror:', e.message))
  page.on('console', (m) => m.type() === 'error' && console.error('console.error:', m.text()))
  await page.goto(url, { waitUntil: 'networkidle0', timeout: 120000 })
  await page.waitForFunction(() => window.heroReady === true, { timeout: 120000 })
  console.log('hero:', await page.evaluate(() => window.heroInfo))
  const el = await page.$('#view')
  await el.screenshot({ path: out, omitBackground: true })
  console.log('wrote', out)
} finally {
  await browser.close()
  await server.close()
}
