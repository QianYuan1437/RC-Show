import { describe, expect, it } from 'vitest'
import { doubleS, otgStep, quintic, trapezoid } from '../docs/.vitepress/theme/lib/traj'
import { maxAbsDiff, readJson } from './helpers'

interface TrajFx {
  traj: {
    t: number[]
    quintic: number[][]
    trapezoid: number[][]
    trapezoid_short: number[][]
    double_s: { T: number; p: number[]; v: number[]; a: number[] }
    double_s_short: { T: number; p: number[]; v: number[]; a: number[] }
  }
}

describe('trajectory profiles vs rczoo.traj', () => {
  const fx = readJson<TrajFx>('fixtures.json').traj
  const sample = (prof: ReturnType<typeof quintic>, k: 'p' | 'v' | 'a') => fx.t.map((t) => prof.at(t)[k])
  it('quintic', () => {
    const q = quintic(0, 1, 2)
    ;(['p', 'v', 'a'] as const).forEach((k, i) => expect(maxAbsDiff(sample(q, k), fx.quintic[i])).toBeLessThan(1e-12))
  })
  it('trapezoid (with and without cruise)', () => {
    for (const [prof, ref] of [
      [trapezoid(0, 1, 1, 2), fx.trapezoid],
      [trapezoid(0, 0.2, 1, 2), fx.trapezoid_short],
    ] as const) {
      expect(maxAbsDiff(sample(prof, 'p'), ref[0])).toBeLessThan(1e-12)
      expect(maxAbsDiff(sample(prof, 'v'), ref[1])).toBeLessThan(1e-12)
    }
  })
  it('double S (full and short move)', () => {
    for (const [prof, ref] of [
      [doubleS(0, 1, 1, 2, 10), fx.double_s],
      [doubleS(0, 0.05, 1, 2, 10), fx.double_s_short],
    ] as const) {
      expect(prof.T).toBeCloseTo(ref.T, 12)
      ;(['p', 'v', 'a'] as const).forEach((k) => expect(maxAbsDiff(sample(prof, k), ref[k])).toBeLessThan(1e-9))
    }
  })
  it('OTG reaches a jumping target without exceeding the limits', () => {
    let p = 0
    let v = 0
    let vMax = 0
    let aMax = 0
    for (let k = 0; k < 4000; k++) {
      const target = k < 300 ? 0.5 : -0.2 // the target jumps while the axis is moving
      const s = otgStep(p, v, target, 1, 4, 1e-3)
      vMax = Math.max(vMax, Math.abs(s.v))
      aMax = Math.max(aMax, Math.abs(s.a))
      p = s.p
      v = s.v
    }
    expect(p).toBeCloseTo(-0.2, 9)
    expect(v).toBe(0)
    expect(vMax).toBeLessThanOrEqual(1 + 1e-9)
    expect(aMax).toBeLessThanOrEqual(4 + 1e-6)
  })
})
