import { describe, expect, it } from 'vitest'
import { Playground, type PlaygroundSpec } from '../docs/.vitepress/theme/lib/playground'
import { loadMujocoNode, readJson, readText } from './helpers'

const TRACKERS = ['pid', 'pd_grav', 'pd_ff', 'ctc', 'smc', 'joint_impedance', 'velocity', 'resolved_rate', 'cart_impedance', 'osc']

async function make(robot: string) {
  const mj = await loadMujocoNode()
  return new Playground(mj, readJson<PlaygroundSpec>('robot.json', robot), readText('plant.xml', robot))
}

function run(pg: Playground, seconds: number) {
  const steps = Math.round(seconds / 1e-3)
  for (let k = 0; k < steps; k++) pg.step()
  return pg.sample()
}

describe.each(['panda', 'sr3'])('playground engine: %s', (robot) => {
  it.each(TRACKERS)('%s moves the TCP 10 cm to a new target and holds it', async (id) => {
    const pg = await make(robot)
    pg.setController(id)
    const t = pg.tcp()
    pg.setTarget({ p: [t.p[0] - 0.05, t.p[1] + 0.08, t.p[2] + 0.03], R: t.R })
    const s = run(pg, 2.5)
    expect(Number.isFinite(s.eTarget)).toBe(true)
    // PID holds against gravity only through its integrator: allow it more
    expect(s.eTarget).toBeLessThan(id === 'pid' ? 20 : id === 'cart_impedance' || id === 'joint_impedance' ? 6 : 2)
    pg.dispose()
  })

  it.each(['hybrid', 'parallel', 'admittance'])('%s presses on the surface with 10 N', async (id) => {
    const pg = await make(robot)
    pg.setController(id)
    pg.setMode('contact')
    run(pg, 2.5)
    let f = 0
    for (let k = 0; k < 500; k++) {
      pg.step()
      f += pg.sample().fz / 500
    }
    expect(Math.abs(f - 10)).toBeLessThan(id === 'admittance' ? 2 : 0.5)
    pg.dispose()
  })

  it('drag teaching follows a hand pulling the TCP', async () => {
    const pg = await make(robot)
    pg.setController('drag')
    pg.setMode('guide')
    const t = pg.tcp()
    pg.setTarget({ p: [t.p[0] + 0.1, t.p[1], t.p[2]], R: t.R })
    const s = run(pg, 3)
    expect(s.eTarget).toBeLessThan(25) // ends where the hand is, up to the friction-limited lag
  })

  it('the momentum observer stops the arm after a push', async () => {
    const pg = await make(robot)
    pg.setController('ctc')
    pg.collisionReaction = true
    run(pg, 0.3)
    expect(pg.reacted).toBe(false)
    pg.push([1, 0, 0], 40)
    run(pg, 0.1)
    expect(pg.reacted).toBe(true)
  })
})
