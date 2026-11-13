import { describe, expect, it } from 'vitest'
import { CONTROLLERS, type RobotLimits } from '../docs/.vitepress/theme/lib/controllers'
import { NominalModel } from '../docs/.vitepress/theme/lib/dynamics'
import type { Mat3, Vec3 } from '../docs/.vitepress/theme/lib/kinematics'
import { loadMujocoNode, maxAbsDiff, readJson, readText } from './helpers'

interface Fixtures {
  q: number[][]
  qd: number[][]
  q_ref: number[][]
  qd_ref: number[][]
  qdd_ref: number[][]
  ref: { p: number[][]; R: number[][][]; v: number[][]; a: number[][] }
  tau: Record<string, number[][]>
}

for (const robot of ['panda', 'sr3']) {
  describe(`controller ports vs rczoo.control: ${robot}`, async () => {
    const mj = await loadMujocoNode()
    const fx = readJson<Fixtures>('fixtures.json', robot)
    const spec = readJson<RobotLimits & { ndof: number }>('robot.json', robot)
    const limits: RobotLimits = { ...spec, n: spec.ndof }
    const nom = new NominalModel(mj, mj.MjModel.from_xml_string(readText('plant.xml', robot)), limits.n)
    for (const [name, taus] of Object.entries(fx.tau)) {
      it(name, () => {
        taus.forEach((tauPy, k) => {
          const c = CONTROLLERS[name](limits)
          c.reset()
          const d = nom.compute(fx.q[k], fx.qd[k])
          const tau = c.step(
            { t: 0, q: Float64Array.from(fx.q[k]), qd: Float64Array.from(fx.qd[k]), ft: new Float64Array(6) },
            {
              q: fx.q_ref[k], qd: fx.qd_ref[k], qdd: fx.qdd_ref[k],
              p: fx.ref.p[k] as Vec3, R: fx.ref.R[k].flat() as Mat3, v: fx.ref.v[k], a: fx.ref.a[k],
              qNull: spec.home,
            },
            d,
            nom,
          )
          const scale = Math.max(1, ...tauPy.map(Math.abs))
          expect(maxAbsDiff(tau, tauPy) / scale).toBeLessThan(1e-4)
        })
      })
    }
  })
}
