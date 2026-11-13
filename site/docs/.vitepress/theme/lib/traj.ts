/**
 * Scalar ports of `rczoo.traj` (rest-to-rest profiles and the online trajectory generator).
 * `test/traj.test.ts` compares them with values computed in Python (fixtures.json).
 */

export interface Sample {
  p: number
  v: number
  a: number
  j: number
}

export interface Profile1D {
  /** Duration (s). */
  T: number
  at(t: number): Sample
}

/** Quintic polynomial, rest to rest, from q0 to q1 in time T. */
export function quintic(q0: number, q1: number, T: number): Profile1D {
  const h = q1 - q0
  return {
    T,
    at(t) {
      const s = Math.min(Math.max(t, 0), T) / T
      return {
        p: q0 + h * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5),
        v: (h / T) * (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4),
        a: (h / T ** 2) * (60 * s - 180 * s ** 2 + 120 * s ** 3),
        j: t >= T ? 0 : (h / T ** 3) * (60 - 360 * s + 360 * s ** 2),
      }
    },
  }
}

/** Trapezoidal velocity (bang-coast-bang acceleration); triangular if vmax is not reached. */
export function trapezoid(q0: number, q1: number, vmax: number, amax: number): Profile1D {
  const sg = q1 >= q0 ? 1 : -1
  const h = Math.abs(q1 - q0)
  const tri = h < (vmax * vmax) / amax
  const v = tri ? Math.sqrt(h * amax) : vmax
  const Ta = v / amax
  const T = h < 1e-12 ? 0 : tri ? 2 * Ta : h / v + v / amax
  return {
    T,
    at(t0) {
      if (T === 0) return { p: q1, v: 0, a: 0, j: 0 }
      const t = Math.min(Math.max(t0, 0), T)
      let p: number, vel: number, acc: number
      if (t < Ta) [p, vel, acc] = [0.5 * amax * t * t, amax * t, amax]
      else if (t > T - Ta) [p, vel, acc] = [h - 0.5 * amax * (T - t) ** 2, amax * (T - t), -amax]
      else [p, vel, acc] = [v * (t - Ta / 2), v, 0]
      if (t0 >= T) acc = 0
      return { p: q0 + sg * p, v: sg * vel, a: sg * acc, j: 0 }
    },
  }
}

/** Jerk-limited seven-segment ("double S") profile, rest to rest (Biagiotti & Melchiorri 2008). */
export function doubleS(q0: number, q1: number, vmax: number, amax: number, jmax: number): Profile1D {
  const sg = q1 >= q0 ? 1 : -1
  const H = Math.abs(q1 - q0)
  const J = jmax
  let Tj: number, Ta: number, Tv: number
  {
    const reachA = vmax * J >= amax * amax
    Tj = reachA ? amax / J : Math.sqrt(vmax / J)
    Ta = reachA ? Tj + vmax / amax : 2 * Tj
    Tv = H / vmax - Ta
    if (Tv <= 0) {
      const reachA2 = H >= (2 * amax ** 3) / (J * J)
      Tj = reachA2 ? amax / J : Math.cbrt(H / (2 * J))
      Ta = reachA2 ? Tj / 2 + Math.sqrt((Tj / 2) ** 2 + H / amax) : 2 * Tj
      Tv = 0
    }
  }
  const alim = J * Tj
  const vlim = (Ta - Tj) * alim
  const T = H < 1e-12 ? 0 : 2 * Ta + Tv
  const accPhase = (tau: number): Sample => {
    if (tau < Tj) return { p: (J * tau ** 3) / 6, v: (J * tau * tau) / 2, a: J * tau, j: J }
    if (tau > Ta - Tj) {
      const s = Ta - tau
      return { p: (vlim * Ta) / 2 - vlim * s + (J * s ** 3) / 6, v: vlim - (J * s * s) / 2, a: J * s, j: -J }
    }
    return { p: (alim * (3 * tau * tau - 3 * Tj * tau + Tj * Tj)) / 6, v: alim * (tau - Tj / 2), a: alim, j: 0 }
  }
  return {
    T,
    at(t0) {
      if (T === 0) return { p: q1, v: 0, a: 0, j: 0 }
      const t = Math.min(Math.max(t0, 0), T)
      let s: Sample
      if (t < Ta) s = accPhase(t)
      else if (t < Ta + Tv) s = { p: (vlim * Ta) / 2 + vlim * (t - Ta), v: vlim, a: 0, j: 0 }
      else {
        const d = accPhase(T - t)
        s = { p: H - d.p, v: d.v, a: -d.a, j: d.j }
      }
      return { p: q0 + sg * s.p, v: sg * s.v, a: sg * s.a, j: sg * s.j }
    },
  }
}

/**
 * One cycle of the acceleration-limited time-optimal online trajectory generator per axis
 * (`rczoo.traj.otg_step`): bang-coast-bang towards (target, 0), re-planned every cycle, so the
 * target may jump at any time (a dragged gizmo) and the output stays continuous in velocity.
 */
export function otgStep(p: number, v: number, target: number, vmax: number, amax: number, dt: number) {
  const d = target - p
  const dStop = (v * Math.abs(v)) / (2 * amax)
  const s = d - dStop >= 0 ? 1 : -1
  const vt = s * v
  const remaining = Math.max(d * s, 0)
  const needBrake = vt > 0 && (vt * vt) / (2 * amax) >= remaining - 1e-12
  let a: number
  if (needBrake) a = -s * Math.min(amax, (vt * vt) / (2 * Math.max(remaining, 1e-12)))
  else a = vt < vmax ? s * amax : vt > vmax ? -s * amax : 0
  let vNew = v + a * dt
  if (!needBrake && s * vNew > vmax && vt <= vmax) vNew = s * vmax
  let pNew = p + 0.5 * (v + vNew) * dt
  if (Math.abs(d) < 1e-6 && Math.abs(v) < amax * dt) {
    pNew = target
    vNew = 0
  }
  return { p: pNew, v: vNew, a: (vNew - v) / dt }
}
