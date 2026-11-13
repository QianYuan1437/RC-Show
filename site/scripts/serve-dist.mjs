// Minimal static server for a VitePress build in an arbitrary directory (`vitepress preview`
// always serves docs/.vitepress/dist). Handles cleanUrls (x -> x.html, x/ -> x/index.html).
//
//   node scripts/serve-dist.mjs /tmp/dist /ik-zoo/ 4199
import { createServer } from 'node:http'
import { readFile, stat } from 'node:fs/promises'
import { extname, join } from 'node:path'

const [dir, base = '/', port = '4199'] = process.argv.slice(2)
const types = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.woff2': 'font/woff2',
  '.glb': 'model/gltf-binary',
  '.xml': 'application/xml',
}

createServer(async (req, res) => {
  let p = decodeURIComponent(new URL(req.url ?? '/', 'http://x').pathname)
  if (!p.startsWith(base)) {
    res.writeHead(404)
    return res.end()
  }
  p = p.slice(base.length)
  for (const c of [p, `${p}.html`, join(p, 'index.html')]) {
    try {
      const f = join(dir, c)
      if (!(await stat(f)).isFile()) continue
      res.writeHead(200, { 'content-type': types[extname(f)] ?? 'application/octet-stream' })
      return res.end(await readFile(f))
    } catch {
      // try the next candidate
    }
  }
  res.writeHead(404, { 'content-type': types['.html'] })
  res.end(await readFile(join(dir, '404.html')).catch(() => ''))
}).listen(Number(port), () => console.log(`serving ${dir} at http://localhost:${port}${base}`))
