/**
 * The controllers' *nominal* model in the browser: a second MuJoCo instance of the plant XML
 * with the payload removed, used only for model queries (never stepped). It provides the same
 * quantities as `rczoo.dynamics.Dynamics` (checked against `fixtures.json` in the tests):
 *
 *   M(q)                 joint-space inertia (with rotor armature)          mj_fullM
 *   h(q, qd) = C qd + g  bias torque                                        mj_rne, flg_acc = 0
 *   g(q)                 gravity torque                                      mj_rne with qd = 0
 *   C(q, qd) qd_r        Christoffel Coriolis term along another velocity   polarisation of h
 *   p, R, J              TCP pose and world Jacobian [v; w]                 mj_jacSite
 *   Jdot qd              TCP bias acceleration                              mj_jacDot
 */

import type { DoubleBuffer, MjData, MjModel } from '@mujoco/mujoco'
import type { Mat3, Vec3 } from './kinematics'
import type { Mujoco } from './mujoco'

export interface DynState {
  n: number
  q: Float64Array
  qd: Float64Array
  /** n x n, row-major. */
  M: Float64Array
  bias: Float64Array
  g: Float64Array
  p: Vec3
  R: Mat3
  /** 6 x n, row-major, rows [vx vy vz wx wy wz]. */
  J: Float64Array
  jdotqd: Float64Array
}

export class NominalModel {
  readonly n: number
  private readonly mj: Mujoco
  private readonly model: MjModel
  private readonly data: MjData
  private readonly site: number
  private readonly bufM: DoubleBuffer
  private readonly bufTau: DoubleBuffer
  private readonly jacp: DoubleBuffer
  private readonly jacr: DoubleBuffer

  /** `model` must be a separate compile of the plant XML (it is modified: payload massless). */
  constructor(mj: Mujoco, model: MjModel, n: number, payloadBody = 'payload') {
    this.mj = mj
    this.model = model
    this.n = n
    const pid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, payloadBody)
    if (pid >= 0) {
      ;(model.body_mass as Float64Array)[pid] = 1e-9
      ;(model.body_inertia as Float64Array).fill(1e-12, 3 * pid, 3 * pid + 3)
    }
    this.data = new mj.MjData(model)
    this.site = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE.value, 'tcp')
    const nv = model.nv
    this.bufM = new mj.DoubleBuffer(nv * nv)
    this.bufTau = new mj.DoubleBuffer(nv)
    this.jacp = new mj.DoubleBuffer(3 * nv)
    this.jacr = new mj.DoubleBuffer(3 * nv)
  }

  private setState(q: ArrayLike<number>, qd: ArrayLike<number> | null): void {
    const qpos = this.data.qpos as Float64Array
    const qvel = this.data.qvel as Float64Array
    for (let i = 0; i < this.n; i++) {
      qpos[i] = q[i]
      qvel[i] = qd ? qd[i] : 0
    }
  }

  /** Bias with the current positions and velocity v: C(q, v) v + g(q). */
  private rne(v: ArrayLike<number> | null): Float64Array {
    const { mj, model, data, n } = this
    const qvel = data.qvel as Float64Array
    for (let i = 0; i < n; i++) qvel[i] = v ? v[i] : 0
    mj.mj_comVel(model, data)
    mj.mj_rne(model, data, 0, this.bufTau)
    return Float64Array.from((this.bufTau.GetView() as Float64Array).subarray(0, n))
  }

  /** All model quantities at (q, qd). */
  compute(q: ArrayLike<number>, qd: ArrayLike<number>): DynState {
    const { mj, model, data, n } = this
    const nv = model.nv
    this.setState(q, qd)
    mj.mj_kinematics(model, data)
    mj.mj_comPos(model, data)
    mj.mj_crb(model, data)
    mj.mj_fullM(model, data, this.bufM)
    const Mf = this.bufM.GetView() as Float64Array
    const M = new Float64Array(n * n)
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) M[i * n + j] = Mf[i * nv + j]
    const g = this.rne(null)
    const bias = this.rne(qd)
    // TCP pose and Jacobian (positions only depend on q)
    const xp = data.site_xpos as Float64Array
    const xm = data.site_xmat as Float64Array
    const s = this.site
    const p: Vec3 = [xp[3 * s], xp[3 * s + 1], xp[3 * s + 2]]
    const R: Mat3 = Array.from(xm.subarray(9 * s, 9 * s + 9))
    mj.mj_jacSite(model, data, this.jacp, this.jacr, s)
    const J = new Float64Array(6 * n)
    const jp = this.jacp.GetView() as Float64Array
    const jr = this.jacr.GetView() as Float64Array
    for (let r = 0; r < 3; r++)
      for (let j = 0; j < n; j++) {
        J[r * n + j] = jp[r * nv + j]
        J[(r + 3) * n + j] = jr[r * nv + j]
      }
    // Jdot qd (cdof_dot of the current velocity is set by the last rne() call)
    mj.mj_jacDot(model, data, this.jacp, this.jacr, p, model.site_bodyid[s])
    const jdotqd = new Float64Array(6)
    const jdp = this.jacp.GetView() as Float64Array
    const jdr = this.jacr.GetView() as Float64Array
    for (let r = 0; r < 3; r++)
      for (let j = 0; j < n; j++) {
        jdotqd[r] += jdp[r * nv + j] * qd[j]
        jdotqd[r + 3] += jdr[r * nv + j] * qd[j]
      }
    return { n, q: Float64Array.from(q), qd: Float64Array.from(qd), M, bias, g, p, R, J, jdotqd }
  }

  /**
   * C(q, qd) qd_r with the Christoffel-consistent C (as `Dynamics.slotine_li`). C(q, a) b is a
   * symmetric bilinear form in (a, b), so by polarisation
   * C(q, qd) qd_r = 1/4 [c(qd + qd_r) - c(qd - qd_r)],  c(v) = h(q, v) - g(q).
   * Call after `compute` (uses the same q).
   */
  coriolis(qd: ArrayLike<number>, qdr: ArrayLike<number>): Float64Array {
    const n = this.n
    const a = new Float64Array(n)
    const b = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      a[i] = qd[i] + qdr[i]
      b[i] = qd[i] - qdr[i]
    }
    const ha = this.rne(a)
    const hb = this.rne(b)
    const out = new Float64Array(n)
    for (let i = 0; i < n; i++) out[i] = 0.25 * (ha[i] - hb[i]) // g cancels
    return out
  }

  dispose(): void {
    this.bufM.delete()
    this.bufTau.delete()
    this.jacp.delete()
    this.jacr.delete()
    this.data.delete()
    this.model.delete()
  }
}
