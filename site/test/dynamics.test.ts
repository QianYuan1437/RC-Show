import { describe, expect, it } from 'vitest'
import { NominalModel } from '../docs/.vitepress/theme/lib/dynamics'
import { loadMujocoNode, maxAbsDiff, readJson, readText } from './helpers'

interface Fixtures {
  q: number[][]
  qd: number[][]
  M: number[][][]
  bias: number[][]
  g: number[][]
  p: number[][]
  R: number[][][]
  J: number[][][]
  jdot_qd: number[][]
}

for (const robot of ['panda', 'sr3']) {
  describe(`nominal model (MuJoCo WASM) vs rczoo.dynamics: ${robot}`, async () => {
    const mj = await loadMujocoNode()
    const fx = readJson<Fixtures>('fixtures.json', robot)
    const n = fx.q[0].length
    const nom = new NominalModel(mj, mj.MjModel.from_xml_string(readText('plant.xml', robot)), n)
    fx.q.forEach((q, k) => {
      it(`state ${k}`, () => {
        const d = nom.compute(q, fx.qd[k])
        expect(maxAbsDiff(d.M, fx.M[k].flat())).toBeLessThan(2e-5)
        expect(maxAbsDiff(d.g, fx.g[k])).toBeLessThan(2e-4)
        expect(maxAbsDiff(d.bias, fx.bias[k])).toBeLessThan(2e-4)
        expect(maxAbsDiff(d.p, fx.p[k])).toBeLessThan(1e-6)
        expect(maxAbsDiff(d.R, fx.R[k].flat())).toBeLessThan(1e-6)
        expect(maxAbsDiff(d.J, fx.J[k].flat())).toBeLessThan(1e-6)
        expect(maxAbsDiff(d.jdotqd, fx.jdot_qd[k])).toBeLessThan(1e-6)
      })
    })
  })
}
