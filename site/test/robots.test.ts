import { describe, expect, it } from 'vitest'
import { ROBOTS } from '../docs/.vitepress/theme/lib/robots'
import { loadRobot, readJson } from './helpers'

describe.each(ROBOTS.map((r) => r.id))('%s', (robot) => {
  const kin = loadRobot(robot)

  it('TCP forward kinematics of robot.json matches rczoo.dynamics to 1e-9', () => {
    const fx = readJson<{ q: number[][]; p: number[][]; R: number[][][] }>('fixtures.json', robot)
    fx.q.forEach((q, k) => {
      const { p, R } = kin.fk(q)
      p.forEach((v, i) => expect(Math.abs(v - fx.p[k][i])).toBeLessThan(1e-9))
      fx.R[k].flat().forEach((v, i) => expect(Math.abs(v - R[i])).toBeLessThan(1e-9))
    })
  })

  it('exports the gains and limits the controllers need', () => {
    const spec = readJson<Record<string, unknown>>('robot.json', robot)
    for (const key of ['torque_limit', 'vel_limit', 'gains', 'home', 'lower', 'upper']) expect(spec).toHaveProperty(key)
  })
})
