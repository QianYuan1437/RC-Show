/**
 * TypeScript ports of the `rczoo.control` controllers (same gains and formulas, one world).
 * Each controller maps the measured state, a reference and the nominal model at the current
 * state (`DynState`, see dynamics.ts) to joint torques; the plant clips them to the motor limits.
 * `test/controllers.test.ts` checks them against torques computed in Python (fixtures.json).
 */

import type { DynState, NominalModel } from './dynamics'
import { type Mat3, type Vec3, matMul3, so3Exp, so3Log, transpose3 } from './kinematics'

export interface Gains {
  kp: number[]
  kd: number[]
  ki: number[]
  imp_k: number[]
  imp_d: number[]
  kv: number[]
  kvi: number[]
  wn_joint: number
  wn_task: number
  cart_k: number[]
  cart_k_soft: number[]
  null_k: number
  null_d: number
}

export interface RobotLimits {
  n: number
  lower: number[]
  upper: number[]
  home: number[]
  vel_limit: number[]
  torque_limit: number[]
  gains: Gains
}

export interface CtrlState {
  t: number
  q: Float64Array
  qd: Float64Array
  /** Wrench the environment applies to the robot at the TCP (world frame), from the F/T sensor. */
  ft: Float64Array
}

export interface CtrlRef {
  q?: ArrayLike<number>
  qd?: ArrayLike<number>
  qdd?: ArrayLike<number>
  p?: Vec3
  R?: Mat3
  /** TCP twist [v; w] and acceleration (world). */
  v?: ArrayLike<number>
  a?: ArrayLike<number>
  /** Desired wrench the robot applies to the environment (world). */
  f?: ArrayLike<number>
  qNull?: ArrayLike<number>
}

export interface Controller {
  readonly id: string
  reset(): void
  step(s: CtrlState, ref: CtrlRef, d: DynState, nom: NominalModel): Float64Array
}

export const DT = 1e-3

// ----------------------------------------------------------------------------- small linalg

const zeros = (n: number) => new Float64Array(n)

function matVec(A: ArrayLike<number>, x: ArrayLike<number>, m: number, n: number): Float64Array {
  const y = zeros(m)
  for (let i = 0; i < m; i++) {
    let s = 0
    for (let j = 0; j < n; j++) s += A[i * n + j] * x[j]
    y[i] = s
  }
  return y
}

function matTVec(A: ArrayLike<number>, x: ArrayLike<number>, m: number, n: number): Float64Array {
  const y = zeros(n)
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) y[j] += A[i * n + j] * x[i]
  return y
}

function matMul(A: ArrayLike<number>, B: ArrayLike<number>, m: number, k: number, n: number): Float64Array {
  const C = zeros(m * n)
  for (let i = 0; i < m; i++)
    for (let l = 0; l < k; l++) {
      const a = A[i * k + l]
      if (a === 0) continue
      for (let j = 0; j < n; j++) C[i * n + j] += a * B[l * n + j]
    }
  return C
}

/** Inverse of a square matrix (Gauss-Jordan with partial pivoting). */
export function inverse(A: ArrayLike<number>, n: number): Float64Array {
  const a = Float64Array.from(A)
  const inv = zeros(n * n)
  for (let i = 0; i < n; i++) inv[i * n + i] = 1
  for (let c = 0; c < n; c++) {
    let piv = c
    for (let r = c + 1; r < n; r++) if (Math.abs(a[r * n + c]) > Math.abs(a[piv * n + c])) piv = r
    if (piv !== c)
      for (let j = 0; j < n; j++) {
        ;[a[c * n + j], a[piv * n + j]] = [a[piv * n + j], a[c * n + j]]
        ;[inv[c * n + j], inv[piv * n + j]] = [inv[piv * n + j], inv[c * n + j]]
      }
    const d = a[c * n + c] || 1e-300
    for (let j = 0; j < n; j++) {
      a[c * n + j] /= d
      inv[c * n + j] /= d
    }
    for (let r = 0; r < n; r++) {
      if (r === c) continue
      const f = a[r * n + c]
      if (f === 0) continue
      for (let j = 0; j < n; j++) {
        a[r * n + j] -= f * a[c * n + j]
        inv[r * n + j] -= f * inv[c * n + j]
      }
    }
  }
  return inv
}

const clip = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x))

/** e = [p* - p; Log(R* R^T)] (world frame). */
export function poseErr(pRef: Vec3, RRef: Mat3, p: Vec3, R: Mat3): Float64Array {
  const w = so3Log(matMul3(RRef, transpose3(R)))
  return Float64Array.of(pRef[0] - p[0], pRef[1] - p[1], pRef[2] - p[2], w[0], w[1], w[2])
}

/** Lambda = (J M^-1 J^T + 1e-3 I)^-1 and M^-1. */
function taskInertia(d: DynState): { Minv: Float64Array; Lam: Float64Array } {
  const n = d.n
  const Minv = inverse(d.M, n)
  const JMi = matMul(d.J, Minv, 6, n, n)
  const A = zeros(36)
  for (let i = 0; i < 6; i++)
    for (let j = 0; j < 6; j++) {
      let s = 0
      for (let k = 0; k < n; k++) s += JMi[i * n + k] * d.J[j * n + k]
      A[i * 6 + j] = s + (i === j ? 1e-3 : 0)
    }
  return { Minv, Lam: inverse(A, 6) }
}

/** Posture torque through N^T = I - J^T Lambda J M^-1 (redundant arms only). */
function nullTorque(r: RobotLimits, s: CtrlState, ref: CtrlRef, d: DynState, Minv: Float64Array, Lam: Float64Array) {
  const n = r.n
  const out = zeros(n)
  if (n <= 6) return out
  const qn = ref.qNull ?? r.home
  const tau0 = zeros(n)
  for (let i = 0; i < n; i++) tau0[i] = r.gains.null_k * (qn[i] - s.q[i]) - r.gains.null_d * s.qd[i]
  // N^T tau0 = tau0 - J^T Lam J M^-1 tau0
  const x = matVec(Minv, tau0, n, n)
  const y = matVec(d.J, x, 6, n)
  const z = matVec(Lam, y, 6, 6)
  const w = matTVec(d.J, z, 6, n)
  for (let i = 0; i < n; i++) out[i] = tau0[i] - w[i]
  return out
}

function twist(d: DynState, qd: ArrayLike<number>): Float64Array {
  return matVec(d.J, qd, 6, d.n)
}

const refQ = (ref: CtrlRef, n: number) => ({
  q: ref.q ?? zeros(n),
  qd: ref.qd ?? zeros(n),
  qdd: ref.qdd ?? zeros(n),
})

const ZERO6 = zeros(6)

// ----------------------------------------------------------------------------- joint space

abstract class Base implements Controller {
  abstract readonly id: string
  constructor(protected readonly r: RobotLimits) {}
  reset(): void {}
  abstract step(s: CtrlState, ref: CtrlRef, d: DynState, nom: NominalModel): Float64Array
}

export class GravityOnly extends Base {
  readonly id = 'gravity'
  step(_s: CtrlState, _r: CtrlRef, d: DynState) {
    return Float64Array.from(d.g)
  }
}

export class JointPID extends Base {
  readonly id = 'pid'
  private integ = zeros(this.r.n)
  reset() {
    this.integ = zeros(this.r.n)
  }
  step(s: CtrlState, ref: CtrlRef) {
    const { n, gains, torque_limit } = this.r
    const { q, qd } = refQ(ref, n)
    const tau = zeros(n)
    for (let i = 0; i < n; i++) {
      const e = q[i] - s.q[i]
      const lim = 0.5 * torque_limit[i]
      this.integ[i] = clip(this.integ[i] + gains.ki[i] * e * DT, -lim, lim)
      tau[i] = gains.kp[i] * e + gains.kd[i] * (qd[i] - s.qd[i]) + this.integ[i]
    }
    return tau
  }
}

export class PDGravity extends Base {
  readonly id = 'pd_grav'
  step(s: CtrlState, ref: CtrlRef, d: DynState) {
    const { n, gains } = this.r
    const { q, qd } = refQ(ref, n)
    const tau = zeros(n)
    for (let i = 0; i < n; i++) tau[i] = gains.kp[i] * (q[i] - s.q[i]) + gains.kd[i] * (qd[i] - s.qd[i]) + d.g[i]
    return tau
  }
}

/** M qdd_r + C(q, qd) qd_r + g (Christoffel C), as `Dynamics.slotine_li`. */
function slotineLi(d: DynState, nom: NominalModel, qdr: ArrayLike<number>, qddr: ArrayLike<number>) {
  const n = d.n
  const Ma = matVec(d.M, qddr, n, n)
  const C = nom.coriolis(d.qd, qdr)
  const out = zeros(n)
  for (let i = 0; i < n; i++) out[i] = Ma[i] + C[i] + d.g[i]
  return out
}

export class PDFeedforward extends Base {
  readonly id = 'pd_ff'
  step(s: CtrlState, ref: CtrlRef, d: DynState, nom: NominalModel) {
    const { n, gains } = this.r
    const { q, qd, qdd } = refQ(ref, n)
    const tau = slotineLi(d, nom, qd, qdd)
    for (let i = 0; i < n; i++) tau[i] += gains.kp[i] * (q[i] - s.q[i]) + gains.kd[i] * (qd[i] - s.qd[i])
    return tau
  }
}

export class ComputedTorque extends Base {
  readonly id = 'ctc'
  step(s: CtrlState, ref: CtrlRef, d: DynState) {
    const { n, gains } = this.r
    const wn = gains.wn_joint
    const { q, qd, qdd } = refQ(ref, n)
    const v = zeros(n)
    for (let i = 0; i < n; i++) v[i] = qdd[i] + wn * wn * (q[i] - s.q[i]) + 2 * wn * (qd[i] - s.qd[i])
    const tau = matVec(d.M, v, n, n)
    for (let i = 0; i < n; i++) tau[i] += d.bias[i]
    return tau
  }
}

export class SlidingMode extends Base {
  readonly id = 'smc'
  lam = 20
  phi = 0.05
  etaFrac = 0.12
  step(s: CtrlState, ref: CtrlRef, d: DynState, nom: NominalModel) {
    const { n, gains, torque_limit } = this.r
    const { q, qd, qdd } = refQ(ref, n)
    const qdr = zeros(n)
    const qddr = zeros(n)
    const sv = zeros(n)
    for (let i = 0; i < n; i++) {
      qdr[i] = qd[i] + this.lam * (q[i] - s.q[i])
      qddr[i] = qdd[i] + this.lam * (qd[i] - s.qd[i])
      sv[i] = s.qd[i] - qdr[i]
    }
    const tau = slotineLi(d, nom, qdr, qddr)
    for (let i = 0; i < n; i++)
      tau[i] += -2 * gains.kd[i] * sv[i] - this.etaFrac * torque_limit[i] * clip(sv[i] / this.phi, -1, 1)
    return tau
  }
}

export class JointImpedance extends Base {
  readonly id = 'joint_impedance'
  step(s: CtrlState, ref: CtrlRef, d: DynState) {
    const { n, gains } = this.r
    const { q, qd, qdd } = refQ(ref, n)
    const tau = matVec(d.M, qdd, n, n)
    for (let i = 0; i < n; i++)
      tau[i] += gains.imp_k[i] * (q[i] - s.q[i]) + gains.imp_d[i] * (qd[i] - s.qd[i]) + d.bias[i]
    return tau
  }
}

export class JointVelocityMode extends Base {
  readonly id: string = 'velocity'
  kx = 20
  protected integ = zeros(this.r.n)
  reset() {
    this.integ = zeros(this.r.n)
  }
  velocityCommand(s: CtrlState, ref: CtrlRef, _d: DynState): Float64Array {
    const { n, vel_limit } = this.r
    const { q, qd } = refQ(ref, n)
    const v = zeros(n)
    for (let i = 0; i < n; i++) v[i] = clip(qd[i] + this.kx * (q[i] - s.q[i]), -vel_limit[i], vel_limit[i])
    return v
  }
  step(s: CtrlState, ref: CtrlRef, d: DynState) {
    const { n, gains, torque_limit } = this.r
    const cmd = this.velocityCommand(s, ref, d)
    const tau = zeros(n)
    for (let i = 0; i < n; i++) {
      const ev = cmd[i] - s.qd[i]
      const lim = 0.5 * torque_limit[i]
      this.integ[i] = clip(this.integ[i] + gains.kvi[i] * ev * DT, -lim, lim)
      tau[i] = gains.kv[i] * ev + this.integ[i] + d.g[i]
    }
    return tau
  }
}

/** Damped pseudo-inverse J^T (J J^T + lam^2 I)^-1 (n x 6). */
function dampedPinv(J: ArrayLike<number>, n: number, lam: number): Float64Array {
  const A = zeros(36)
  for (let i = 0; i < 6; i++)
    for (let j = 0; j < 6; j++) {
      let s = 0
      for (let k = 0; k < n; k++) s += J[i * n + k] * J[j * n + k]
      A[i * 6 + j] = s + (i === j ? lam * lam : 0)
    }
  const Ai = inverse(A, 6)
  const P = zeros(n * 6)
  for (let k = 0; k < n; k++)
    for (let j = 0; j < 6; j++) {
      let s = 0
      for (let i = 0; i < 6; i++) s += J[i * n + k] * Ai[i * 6 + j]
      P[k * 6 + j] = s
    }
  return P
}

// ----------------------------------------------------------------------------- task space

export class ResolvedRate extends JointVelocityMode {
  readonly id = 'resolved_rate'
  kNull = 2
  lam = 0.02
  velocityCommand(s: CtrlState, ref: CtrlRef, d: DynState): Float64Array {
    const { n, vel_limit, home } = this.r
    const e = poseErr(ref.p!, ref.R!, d.p, d.R)
    const vr = ref.v ?? ZERO6
    const x = zeros(6)
    for (let i = 0; i < 6; i++) x[i] = vr[i] + this.kx * e[i]
    const Jp = dampedPinv(d.J, n, this.lam)
    const qd = matVec(Jp, x, n, 6)
    if (n > 6) {
      const qn = ref.qNull ?? home
      const z = zeros(n)
      for (let i = 0; i < n; i++) z[i] = this.kNull * (qn[i] - s.q[i])
      const JpJ = matMul(Jp, d.J, n, 6, n)
      const corr = matVec(JpJ, z, n, n)
      for (let i = 0; i < n; i++) qd[i] += z[i] - corr[i]
    }
    for (let i = 0; i < n; i++) qd[i] = clip(qd[i], -vel_limit[i], vel_limit[i])
    return qd
  }
}

export class CartesianImpedance extends Base {
  readonly id: string = 'cart_impedance'
  K: number[]
  zeta = 1
  constructor(r: RobotLimits, K?: number[]) {
    super(r)
    this.K = K ?? r.gains.cart_k
  }
  step(s: CtrlState, ref: CtrlRef, d: DynState) {
    const n = this.r.n
    const { Minv, Lam } = taskInertia(d)
    const e = poseErr(ref.p!, ref.R!, d.p, d.R)
    const xd = twist(d, s.qd)
    const v = ref.v ?? ZERO6
    const a = ref.a ?? ZERO6
    const La = matVec(Lam, a, 6, 6)
    const F = zeros(6)
    for (let i = 0; i < 6; i++) {
      const D = 2 * this.zeta * Math.sqrt(this.K[i] * Math.max(Lam[i * 6 + i], 1e-6))
      F[i] = this.K[i] * e[i] + D * (v[i] - xd[i]) + La[i]
    }
    const tau = matTVec(d.J, F, 6, n)
    const tn = nullTorque(this.r, s, ref, d, Minv, Lam)
    for (let i = 0; i < n; i++) tau[i] += d.bias[i] + tn[i]
    return tau
  }
}

export class OperationalSpace extends Base {
  readonly id: string = 'osc'
  wn: number
  constructor(r: RobotLimits, wn?: number) {
    super(r)
    this.wn = wn ?? r.gains.wn_task
  }
  /** Desired TCP acceleration of the decoupled loop (before Lambda). */
  protected accel(s: CtrlState, ref: CtrlRef, d: DynState): Float64Array {
    const e = poseErr(ref.p!, ref.R!, d.p, d.R)
    const xd = twist(d, s.qd)
    const v = ref.v ?? ZERO6
    const a = ref.a ?? ZERO6
    const kp = this.wn * this.wn
    const kd = 2 * this.wn
    const acc = zeros(6)
    for (let i = 0; i < 6; i++) acc[i] = a[i] + kp * e[i] + kd * (v[i] - xd[i]) - d.jdotqd[i]
    return acc
  }
  step(s: CtrlState, ref: CtrlRef, d: DynState) {
    const n = this.r.n
    const { Minv, Lam } = taskInertia(d)
    const F = matVec(Lam, this.accel(s, ref, d), 6, 6)
    const tau = matTVec(d.J, F, 6, n)
    const tn = nullTorque(this.r, s, ref, d, Minv, Lam)
    for (let i = 0; i < n; i++) tau[i] += d.bias[i] + tn[i]
    return tau
  }
}

// ----------------------------------------------------------------------------- force control

export class Admittance extends Base {
  readonly id = 'admittance'
  Ma = [4, 4, 4, 0.15, 0.15, 0.15]
  Ka: number[]
  Da: number[]
  deadband = 0.5
  inner: OperationalSpace
  x = zeros(6)
  xd = zeros(6)
  /** Reference actually tracked by the inner loop (for display). */
  last?: { p: Vec3; R: Mat3 }
  constructor(r: RobotLimits) {
    super(r)
    this.Ka = r.gains.cart_k_soft
    this.Da = this.Ma.map((m, i) => 2 * Math.sqrt(m * this.Ka[i]))
    this.inner = new OperationalSpace(r, 40)
  }
  reset() {
    this.x = zeros(6)
    this.xd = zeros(6)
  }
  step(s: CtrlState, ref: CtrlRef, d: DynState, nom: NominalModel) {
    const xdd = zeros(6)
    for (let i = 0; i < 6; i++) {
      const f = Math.sign(s.ft[i]) * Math.max(Math.abs(s.ft[i]) - this.deadband, 0)
      xdd[i] = (f - this.Da[i] * this.xd[i] - this.Ka[i] * this.x[i]) / this.Ma[i]
    }
    for (let i = 0; i < 6; i++) this.xd[i] += xdd[i] * DT
    for (let i = 0; i < 6; i++) this.x[i] += this.xd[i] * DT
    const p = ref.p!
    const v = ref.v ?? ZERO6
    const a = ref.a ?? ZERO6
    const cref: CtrlRef = {
      ...ref,
      p: [p[0] + this.x[0], p[1] + this.x[1], p[2] + this.x[2]],
      R: matMul3(so3Exp([this.x[3], this.x[4], this.x[5]]), ref.R!),
      v: v.length ? Array.from({ length: 6 }, (_, i) => v[i] + this.xd[i]) : v,
      a: Array.from({ length: 6 }, (_, i) => a[i] + xdd[i]),
    }
    this.last = { p: cref.p!, R: cref.R! }
    return this.inner.step(s, cref, d)
  }
}

export class HybridForceMotion extends OperationalSpace {
  readonly id = 'hybrid'
  normal: Vec3 = [0, 0, 1]
  kpf = 0.3
  kif = 6
  kvf = 40
  integ = 0
  contact = 0
  reset() {
    this.integ = 0
    this.contact = 0
  }
  step(s: CtrlState, ref: CtrlRef, d: DynState) {
    const n = this.r.n
    const nn = this.normal
    const { Minv, Lam } = taskInertia(d)
    const fMeas = s.ft[0] * nn[0] + s.ft[1] * nn[1] + s.ft[2] * nn[2]
    if (fMeas > 1) this.contact = 1
    const c = this.contact
    const acc = this.accel(s, ref, d)
    // Omega acc: remove the normal component once in contact
    const an = acc[0] * nn[0] + acc[1] * nn[1] + acc[2] * nn[2]
    for (let i = 0; i < 3; i++) acc[i] -= c * an * nn[i]
    const F = matVec(Lam, acc, 6, 6)
    const f = ref.f ?? ZERO6
    const fd = -(f[0] * nn[0] + f[1] * nn[1] + f[2] * nn[2])
    const err = fd - fMeas
    this.integ = clip((this.integ + err * DT) * c, -20, 20)
    const xd = twist(d, s.qd)
    const vn = xd[0] * nn[0] + xd[1] * nn[1] + xd[2] * nn[2]
    const fCmd = -(fd + this.kpf * err + this.kif * this.integ) - this.kvf * vn
    for (let i = 0; i < 3; i++) F[i] += c * fCmd * nn[i]
    const tau = matTVec(d.J, F, 6, n)
    const tn = nullTorque(this.r, s, ref, d, Minv, Lam)
    for (let i = 0; i < n; i++) tau[i] += d.bias[i] + tn[i]
    return tau
  }
}

export class ParallelForcePosition extends CartesianImpedance {
  readonly id = 'parallel'
  normal: Vec3 = [0, 0, 1]
  kpf = 0.0005
  kif = 0.02
  integ = 0
  constructor(r: RobotLimits) {
    super(r, [1500, 1500, 1500, 150, 150, 150])
  }
  reset() {
    this.integ = 0
  }
  step(s: CtrlState, ref: CtrlRef, d: DynState) {
    const nn = this.normal
    const fMeas = s.ft[0] * nn[0] + s.ft[1] * nn[1] + s.ft[2] * nn[2]
    const f = ref.f ?? ZERO6
    const fd = -(f[0] * nn[0] + f[1] * nn[1] + f[2] * nn[2])
    const err = fd - fMeas
    this.integ = clip(this.integ + err * DT, -50, 50)
    const shift = -(this.kpf * err + this.kif * this.integ)
    const p = ref.p!
    return super.step(s, { ...ref, p: [p[0] + shift * nn[0], p[1] + shift * nn[1], p[2] + shift * nn[2]] }, d)
  }
}

// ----------------------------------------------------------------------------- interaction

export class DragTeaching extends Base {
  readonly id = 'drag'
  /** Coulomb friction compensation per joint (N m). */
  friction: number[]
  damping = 0.8
  vSmooth = 0.02
  wallK = 200
  constructor(r: RobotLimits, friction?: number[]) {
    super(r)
    this.friction = friction ?? new Array(r.n).fill(0)
  }
  step(s: CtrlState, _ref: CtrlRef, d: DynState) {
    const { n, lower, upper } = this.r
    const tau = zeros(n)
    const m = 0.087
    for (let i = 0; i < n; i++) {
      tau[i] = d.g[i] + this.friction[i] * Math.tanh(s.qd[i] / this.vSmooth) - this.damping * s.qd[i]
      tau[i] += this.wallK * (Math.max(lower[i] + m - s.q[i], 0) - Math.max(s.q[i] - upper[i] + m, 0))
    }
    return tau
  }
}

/**
 * Generalised-momentum observer r ~ tau_ext (De Luca & Mattone 2003):
 * r = K_O (M qd - p0 - int (tau + C^T qd - g - D qd + r) dt). C^T qd = Mdot qd - C qd, with
 * Mdot from a finite difference of M along qd.
 */
export class MomentumObserver {
  r: Float64Array
  private integ: Float64Array
  private p0: Float64Array | null = null
  constructor(
    private readonly n: number,
    private readonly damping: number[],
    public k = 50,
  ) {
    this.r = zeros(n)
    this.integ = zeros(n)
  }
  reset() {
    this.r = zeros(this.n)
    this.integ = zeros(this.n)
    this.p0 = null
  }
  update(d: DynState, nom: NominalModel, tau: ArrayLike<number>): Float64Array {
    const n = this.n
    const p = matVec(d.M, d.qd, n, n)
    if (!this.p0) {
      this.p0 = p
      return this.r
    }
    const eps = 1e-6
    const qp = Float64Array.from(d.q, (x, i) => x + eps * d.qd[i])
    const M2 = nom.compute(qp, d.qd).M
    const Mdq = zeros(n)
    for (let i = 0; i < n; i++) {
      let sacc = 0
      for (let j = 0; j < n; j++) sacc += ((M2[i * n + j] - d.M[i * n + j]) / eps) * d.qd[j]
      Mdq[i] = sacc
    }
    // C qd = bias - g at the current state (recompute: nom.compute above changed its internal state)
    const back = nom.compute(d.q, d.qd)
    for (let i = 0; i < n; i++) {
      const Cqd = back.bias[i] - back.g[i]
      const CTqd = Mdq[i] - Cqd
      this.integ[i] += (tau[i] + CTqd - d.g[i] - this.damping[i] * d.qd[i] + this.r[i]) * DT
      this.r[i] = this.k * (p[i] - this.p0[i] - this.integ[i])
    }
    return this.r
  }
}

// ----------------------------------------------------------------------------- registry

export const CONTROLLERS: Record<string, (r: RobotLimits) => Controller> = {
  gravity: (r) => new GravityOnly(r),
  pid: (r) => new JointPID(r),
  pd_grav: (r) => new PDGravity(r),
  pd_ff: (r) => new PDFeedforward(r),
  ctc: (r) => new ComputedTorque(r),
  smc: (r) => new SlidingMode(r),
  joint_impedance: (r) => new JointImpedance(r),
  velocity: (r) => new JointVelocityMode(r),
  resolved_rate: (r) => new ResolvedRate(r),
  cart_impedance: (r) => new CartesianImpedance(r),
  osc: (r) => new OperationalSpace(r),
  admittance: (r) => new Admittance(r),
  hybrid: (r) => new HybridForceMotion(r),
  parallel: (r) => new ParallelForcePosition(r),
  drag: (r) => new DragTeaching(r),
}
