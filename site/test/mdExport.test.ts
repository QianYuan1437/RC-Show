import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { exportMarkdown, exportable, routeOf } from '../docs/.vitepress/mdExport'

const OPTS = {
  siteUrl: 'https://example.org/RC-Show/',
  repoUrl: 'https://github.com/o/r',
  repoBranch: 'main',
  docsPathInRepo: 'site/docs',
}
const DOCS = fileURLToPath(new URL('../docs/', import.meta.url))

function pages(dir = DOCS): string[] {
  return readdirSync(dir).flatMap((n) => {
    const p = join(dir, n)
    if (n.startsWith('.') || n === 'public') return []
    if (statSync(p).isDirectory()) return pages(p)
    return n.endsWith('.md') ? [relative(DOCS, p)] : []
  })
}

describe('Markdown export', () => {
  it('maps pages to routes', () => {
    expect(routeOf('methods/dls.md')).toBe('methods/dls')
    expect(routeOf('methods/index.md')).toBe('methods/')
    expect(routeOf('index.md')).toBe('')
  })

  it('rewrites components, Src links, containers and relative links', () => {
    const src = [
      '---',
      'aside: false',
      '---',
      '# T',
      '',
      '::: tip 一句话',
      '见 [OSC](./osc) 与 <Src path="rczoo/robot.py" symbol="load_robot" />。',
      ':::',
      '',
      '```ts',
      '::: not a container',
      '```',
      '',
      '<ControlPlayground initial-controller="osc" />',
      '<ConvergenceCompare',
      '  :ids="[1]"',
      '/>',
      '$$ x^2 $$',
      '<style>',
      '.a { color: red }',
      '</style>',
    ].join('\n')
    const md = exportMarkdown(src, 'methods/impedance.md', OPTS)
    expect(md).not.toContain('aside: false')
    expect(md).toContain('> **一句话**')
    expect(md).toContain('> 见 [OSC](https://example.org/RC-Show/methods/osc)')
    expect(md).toContain('[`rczoo/robot.py · load_robot`](https://github.com/o/r/blob/main/rczoo/robot.py)')
    expect(md).toContain('::: not a container')
    expect(md).toContain('［交互内容：ControlPlayground］')
    expect(md).toContain('［交互内容：ConvergenceCompare］')
    expect(md).toContain('<https://example.org/RC-Show/methods/impedance>')
    expect(md).toContain('$$ x^2 $$')
    expect(md).not.toMatch(/<style|color: red|<[A-Z]/)
  })

  it.each(pages().filter((p) => exportable(readFileSync(join(DOCS, p), 'utf8'))))(
    '%s exports to plain Markdown',
    (rel) => {
      const md = exportMarkdown(readFileSync(join(DOCS, rel), 'utf8'), rel, OPTS)
      expect(md).not.toMatch(/<[A-Z][A-Za-z]*[\s/>]/) // no Vue components left
      expect(md).not.toMatch(/^:::/m)
      expect(md).not.toMatch(/<(style|script)\b/)
      expect(md).toMatch(/^# /m)
    },
  )
})
