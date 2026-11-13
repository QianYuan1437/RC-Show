/**
 * The control playground's engine (no DOM): a MuJoCo WASM plant at 1 kHz, the nominal model,
 * a reference generator and the ported controllers, plus the scripted "world" (payload, joint
 * friction, contact surface, pushes, a hand pulling the TCP). The Vue component only renders.
 *
 * Reference generation: the target pose (gizmo) is filtered by the online trajectory generator
 * (acceleration-limited, per axis) into a smooth TCP reference (p, R, v, a); a joint reference
 * (q, qd, qdd) follows it by damped-least-squares differential IK on the nominal kinematics, so
 * joint-space and task-space controllers track the same motion.
 */

import type { MjData, MjModel } from '@mujoco/mujoco'
import {
  CONTROLLERS,
  type Controller,
  type CtrlRef,
  type CtrlState,
  DT,
  DragTeaching,
  type Gains,
  MomentumObserver,
  type RobotLimits,
  inverse,
} from './controllers'
import { type DynState, NominalModel } from './dynamics'
import { Kinematics, type Mat3, type Pose, type RobotSpec, type Vec3, matMul3, so3Exp, so3Log, transpose3 } from './kinematics'
import type { Mujoco } from './mujoco'
import { otgStep } from './traj'

export interface PlaygroundSpec extends RobotSpec {
  label: string
  vel_limit: number[]
  torque_limit: number[]
  damping: number[]
  gains: Gains
  plant: { probe_radius: number; surface_size: number[] }
}

export interface ReachMeta {
  obs_dim: number
  act_dim: number
  policy_dt: number
  target_step: number[]
  kp: number[]
  kd: number[]
}

export type Mode = 'track' | 'contact' | 'guide'

export interface Sample {
  t: number
  /** |p_tcp - p_target| and |p_tcp - p_ref| (mm). */
  eTarget: number
  eRef: number
  /** Max |tau_i| / tau_max,i. */
  tauRel: number
  /** Contact force along +z (N, environment on robot). */
  fz: number
  /** |hand force| (N), guiding mode. */
  fHand: number
  /** max_i |r_i| / threshold_i of the momentum observer. */
  rRel: number
}

/** Cartesian reference limits (OTG). */
const V_LIN = 0.4
const A_LIN = 2.0
const V_ANG = 1.2
const A_ANG = 6.0
const K_HAND = 600
const D_HAND = 40
const IK_GAIN = 20

export class Playground {
  readonly n: number
  readonly kin: Kinematics
  readonly limits: RobotLimits
  readonly nom: NominalModel
  readonly model: MjModel
  readonly data: MjData
  t = 0
  mode: Mode = 'track'
  controllerId = 'osc'
  ctrl: Controller
  /** Desired contact force (N) in contact mode. */
  fDesired = 10
  /** Observer threshold as a fraction of the torque limit; 0 disables the reaction. */
  collisionReaction = false
  obsThreshold = 0.06
  reacted = false
  /** Target pose (gizmo). */
  target: Pose
  /** Smoothed TCP reference. */
  ref: { p: Vec3; R: Mat3; v: number[]; a: number[] }
  /** Joint reference. */
  qRef: Float64Array
  qdRef: Float64Array
  qddRef: Float64Array
  surfaceZ = -1
  lastTau: Float64Array<ArrayBufferLike>
  lastFt: Float64Array<ArrayBufferLike> = new Float64Array(6)
  handForce: Vec3 = [0, 0, 0]
  dyn: DynState | null = null
  readonly observer: MomentumObserver
  private pushUntil = -1
  private pushForce: Vec3 = [0, 0, 0]
  private readonly mj: Mujoco
  private readonly site: number
  private readonly ftSite: number
  private readonly tcpBody: number
  private readonly payloadBody: number
  private readonly adrF: number
  private readonly adrT: number
  private readonly surfaceMocap: number
  // learned reaching policy (driven from outside: the policy runs asynchronously)
  rl: { meta: ReachMeta; qCmd: Float64Array; aPrev: Float32Array; k: number } | null = null

  constructor(mj: Mujoco, spec: PlaygroundSpec, plantXml: string) {
    this.mj = mj
    this.n = spec.ndof
    this.kin = new Kinematics(spec)
    this.limits = { ...spec, n: spec.ndof }
    this.model = mj.MjModel.from_xml_string(plantXml)
    this.data = new mj.MjData(this.model)
    this.nom = new NominalModel(mj, mj.MjModel.from_xml_string(plantXml), this.n)
    const id = (type: number, name: string) => mj.mj_name2id(this.model, type, name)
    this.site = id(mj.mjtObj.mjOBJ_SITE.value, 'tcp')
    this.ftSite = id(mj.mjtObj.mjOBJ_SITE.value, 'ft')
    this.tcpBody = this.model.site_bodyid[this.site]
    this.payloadBody = id(mj.mjtObj.mjOBJ_BODY.value, 'payload')
    this.adrF = this.model.sensor_adr[id(mj.mjtObj.mjOBJ_SENSOR.value, 'ft_force')]
    this.adrT = this.model.sensor_adr[id(mj.mjtObj.mjOBJ_SENSOR.value, 'ft_torque')]
    this.surfaceMocap = this.model.body_mocapid[id(mj.mjtObj.mjOBJ_BODY.value, 'surface')]
    this.observer = new MomentumObserver(this.n, spec.damping)
    this.ctrl = CONTROLLERS[this.controllerId](this.limits)
    this.lastTau = new Float64Array(this.n)
    this.qRef = new Float64Array(this.n)
    this.qdRef = new Float64Array(this.n)
    this.qddRef = new Float64Array(this.n)
    const home = this.kin.fk(spec.home)
    this.target = home
    this.ref = { p: home.p, R: home.R, v: [0, 0, 0, 0, 0, 0], a: [0, 0, 0, 0, 0, 0] }
    this.reset(spec.home)
  }

  // ------------------------------------------------------------------- state access

  get q(): Float64Array {
    return (this.data.qpos as Float64Array).slice(0, this.n)
  }

  get qd(): Float64Array {
    return (this.data.qvel as Float64Array).slice(0, this.n)
  }

  tcp(): Pose {
    const xp = this.data.site_xpos as Float64Array
    const xm = this.data.site_xmat as Float64Array
    const s = this.site
    return { p: [xp[3 * s], xp[3 * s + 1], xp[3 * s + 2]], R: Array.from(xm.subarray(9 * s, 9 * s + 9)) }
  }

  /** Wrench of the environment on the robot at the probe (world), from the F/T sensor. */
  ft(): Float64Array {
    const sd = this.data.sensordata as Float64Array
    const R = Array.from((this.data.site_xmat as Float64Array).subarray(9 * this.ftSite, 9 * this.ftSite + 9))
    const out = new Float64Array(6)
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) {
        out[i] -= R[3 * i + j] * sd[this.adrF + j]
        out[3 + i] -= R[3 * i + j] * sd[this.adrT + j]
      }
    return out
  }

  // ------------------------------------------------------------------- world settings

  setPayload(kg: number): void {
    if (this.payloadBody < 0) return
    const m = Math.max(kg, 1e-6)
    ;(this.model.body_mass as Float64Array)[this.payloadBody] = m
    const I = (m * 0.08 * 0.08) / 6
    ;(this.model.body_inertia as Float64Array).fill(I, 3 * this.payloadBody, 3 * this.payloadBody + 3)
  }

  /** Joint dry friction as a fraction of the torque limit. */
  setFriction(frac: number): void {
    const fl = this.model.dof_frictionloss as Float64Array
    for (let i = 0; i < this.n; i++) fl[i] = frac * this.limits.torque_limit[i]
  }

  /** Friction compensation of the drag-teaching controller (fraction of the torque limit). */
  setFrictionCompensation(frac: number): void {
    if (this.ctrl instanceof DragTeaching) this.ctrl.friction = this.limits.torque_limit.map((t) => frac * t)
  }

  setController(id: string): void {
    this.controllerId = id
    if (id === 'rl') {
      this.ctrl = CONTROLLERS.gravity(this.limits)
      this.startPolicy()
    } else {
      this.rl = null
      this.ctrl = CONTROLLERS[id](this.limits)
    }
    this.ctrl.reset()
    this.reacted = false
    this.contactT0 = -1
    this.syncReference()
    if (this.mode === 'contact') this.target.p[2] = this.contactHeight()
  }

  setMode(mode: Mode): void {
    this.mode = mode
    this.contactT0 = -1
    const tcp = this.tcp()
    if (mode === 'contact') {
      // surface 5 cm below the probe, centred under it
      this.surfaceZ = tcp.p[2] - this.probeRadius() - 0.05
      const mp = this.data.mocap_pos as Float64Array
      mp.set([tcp.p[0], tcp.p[1], this.surfaceZ - 0.01], 3 * this.surfaceMocap)
      this.target = { p: [tcp.p[0], tcp.p[1], this.contactHeight()], R: tcp.R }
    } else {
      this.surfaceZ = -1
      ;(this.data.mocap_pos as Float64Array).set([0.6, 0, -1], 3 * this.surfaceMocap)
      this.target = tcp
    }
    this.ctrl.reset()
    this.syncReference()
  }

  /** Height of the TCP reference in contact mode (impedance-type controllers press by deflection). */
  contactHeight(): number {
    const g = this.limits.gains
    const lower =
      this.controllerId === 'cart_impedance'
        ? this.fDesired / g.cart_k[2]
        : this.controllerId === 'admittance'
          ? this.fDesired / g.cart_k_soft[2]
          : this.controllerId === 'hybrid'
            ? 0.003
            : 0
    return this.surfaceZ + this.probeRadius() - lower
  }

  probeRadius(): number {
    return 0.012
  }

  setTarget(pose: Pose): void {
    this.target = { p: [...pose.p] as Vec3, R: [...pose.R] }
    if (this.mode === 'contact') this.target.p[2] = this.contactHeight()
  }

  /** 30 N horizontal pulse for 0.2 s at the TCP body. */
  push(direction?: Vec3, force = 30): void {
    const a = Math.random() * 2 * Math.PI
    const d = direction ?? [Math.cos(a), Math.sin(a), 0]
    this.pushForce = [force * d[0], force * d[1], force * d[2]]
    this.pushUntil = this.t + 0.2
  }

  reset(q: ArrayLike<number>): void {
    const { mj, model, data } = this
    mj.mj_resetData(model, data)
    ;(data.qpos as Float64Array).set(Array.from(q).slice(0, this.n))
    mj.mj_forward(model, data)
    this.t = 0
    this.qRef.set(Array.from(q))
    this.qdRef.fill(0)
    this.qddRef.fill(0)
    this.ctrl.reset()
    this.observer.reset()
    this.reacted = false
    this.lastTau.fill(0)
    this.pushUntil = -1
    if (this.mode === 'contact') this.setMode('contact')
    this.target = this.tcp()
    this.syncReference()
    if (this.rl) this.startPolicy()
  }

  /** Restart the reference at the current state (after switching controller / mode). */
  private syncReference(): void {
    const tcp = this.tcp()
    this.ref = { p: tcp.p, R: tcp.R, v: [0, 0, 0, 0, 0, 0], a: [0, 0, 0, 0, 0, 0] }
    this.qRef.set(this.q)
    this.qdRef.fill(0)
    this.qddRef.fill(0)
    this.observer.reset()
  }

  // ------------------------------------------------------------------- learned policy

  private startPolicy(): void {
    if (!this.rlMeta) return
    this.rl = { meta: this.rlMeta, qCmd: this.q.slice(), aPrev: new Float32Array(this.n), k: 0 }
  }

  rlMeta: ReachMeta | null = null

  /** True when the policy has to act before the next physics step. */
  policyDue(): boolean {
    return !!this.rl && this.controllerId === 'rl' && this.rl.k % Math.round(this.rl.meta.policy_dt / DT) === 0
  }

  /** Observation of `rczoo.rl.common.observation` (float32). */
  policyObservation(): Float32Array {
    const rl = this.rl!
    const d = this.nom.compute(this.q, this.qd)
    const q = this.q
    const qd = this.qd
    const e = so3Log(matMul3(this.target.R, transpose3(d.R)))
    const obs: number[] = [
      ...q,
      ...Array.from(qd, (v) => 0.1 * v),
      ...Array.from(q, (v, i) => 10 * (rl.qCmd[i] - v)),
      ...d.p,
      ...[0, 1, 2].map((i) => 5 * (this.target.p[i] - d.p[i])),
      ...e,
      ...rl.aPrev,
    ]
    return Float32Array.from(obs)
  }

  applyPolicyAction(a: ArrayLike<number>): void {
    const rl = this.rl!
    for (let i = 0; i < this.n; i++) {
      const ai = Math.min(1, Math.max(-1, a[i]))
      rl.qCmd[i] = Math.min(this.limits.upper[i], Math.max(this.limits.lower[i], rl.qCmd[i] + rl.meta.target_step[i] * ai))
      rl.aPrev[i] = ai
    }
  }

  // ------------------------------------------------------------------- reference

  private updateReference(): void {
    const r = this.ref
    // translation: per-axis OTG
    const p: Vec3 = [0, 0, 0]
    for (let i = 0; i < 3; i++) {
      const s = otgStep(r.p[i], r.v[i], this.target.p[i], V_LIN, A_LIN, DT)
      p[i] = s.p
      r.v[i] = s.v
      r.a[i] = s.a
    }
    r.p = p
    // rotation: OTG of the remaining angle about the current error axis
    const e = so3Log(matMul3(this.target.R, transpose3(r.R)))
    const ang = Math.hypot(e[0], e[1], e[2])
    const w0 = [r.v[3], r.v[4], r.v[5]]
    let w = [0, 0, 0]
    if (ang > 1e-9) {
      const ax = e.map((x) => x / ang)
      const wPar = w0[0] * ax[0] + w0[1] * ax[1] + w0[2] * ax[2]
      const s = otgStep(0, wPar, ang, V_ANG, A_ANG, DT)
      w = ax.map((x) => x * s.v)
    }
    for (let i = 0; i < 3; i++) {
      r.a[3 + i] = (w[i] - w0[i]) / DT
      r.v[3 + i] = w[i]
    }
    r.R = matMul3(so3Exp(w.map((x) => x * DT)), r.R)
    // joint reference: DLS differential IK on the nominal kinematics
    const { J, ee } = this.kin.jacobian(this.qRef)
    const ev = so3Log(matMul3(r.R, transpose3(ee.R)))
    const x = [0, 1, 2].map((i) => r.v[i] + IK_GAIN * (r.p[i] - ee.p[i])).concat([0, 1, 2].map((i) => r.v[3 + i] + IK_GAIN * ev[i]))
    const n = this.n
    const A = new Float64Array(36)
    for (let i = 0; i < 6; i++)
      for (let j = 0; j < 6; j++) {
        let s = 0
        for (let k = 0; k < n; k++) s += J[i * n + k] * J[j * n + k]
        A[i * 6 + j] = s + (i === j ? 1e-4 : 0)
      }
    const y = inverse(A, 6)
    const z = new Float64Array(6)
    for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) z[i] += y[i * 6 + j] * x[j]
    const qd = new Float64Array(n)
    for (let k = 0; k < n; k++) for (let i = 0; i < 6; i++) qd[k] += J[i * n + k] * z[i]
    if (n > 6) {
      // posture: 2 (home - q) projected into the null space of J
      const h = Array.from(this.qRef, (v, i) => 2 * (this.limits.home[i] - v))
      const Jh = new Float64Array(6)
      for (let i = 0; i < 6; i++) for (let k = 0; k < n; k++) Jh[i] += J[i * n + k] * h[k]
      const yh = new Float64Array(6)
      for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) yh[i] += y[i * 6 + j] * Jh[j]
      for (let k = 0; k < n; k++) {
        let s = 0
        for (let i = 0; i < 6; i++) s += J[i * n + k] * yh[i]
        qd[k] += h[k] - s
      }
    }
    for (let k = 0; k < n; k++) {
      const v = Math.min(this.limits.vel_limit[k], Math.max(-this.limits.vel_limit[k], qd[k]))
      this.qddRef[k] = 0.9 * this.qddRef[k] + 0.1 * ((v - this.qdRef[k]) / DT) // light low-pass
      this.qdRef[k] = v
      this.qRef[k] = Math.min(this.limits.upper[k], Math.max(this.limits.lower[k], this.qRef[k] + v * DT))
    }
  }

  // ------------------------------------------------------------------- one control cycle

  step(): void {
    const { mj, model, data, n } = this
    const q = this.q
    const qd = this.qd
    const ft = this.ft()
    this.lastFt = ft
    this.updateReference()
    const d = this.nom.compute(q, qd)
    this.dyn = d
    const state: CtrlState = { t: this.t, q, qd, ft }
    const ref: CtrlRef = {
      q: this.qRef,
      qd: this.qdRef,
      qdd: this.qddRef,
      p: this.ref.p,
      R: this.ref.R,
      v: this.ref.v,
      a: this.ref.a,
      f: this.mode === 'contact' ? [0, 0, -this.fDesired * Math.min(1, this.contactTime() / 0.3), 0, 0, 0] : undefined,
      qNull: this.limits.home,
    }
    let tau: Float64Array
    if (this.reacted) {
      // reflex: float with gravity compensation and damping
      tau = Float64Array.from(d.g, (g, i) => g - 5 * qd[i])
    } else if (this.controllerId === 'rl' && this.rl) {
      const rl = this.rl
      const { kp, kd } = rl.meta
      tau = Float64Array.from(q, (v, i) => kp[i] * (rl.qCmd[i] - v) - kd[i] * qd[i] + d.g[i])
      rl.k++
    } else {
      tau = this.ctrl.step(state, ref, d, this.nom)
    }
    for (let i = 0; i < n; i++) {
      const L = this.limits.torque_limit[i]
      tau[i] = Math.min(L, Math.max(-L, tau[i]))
    }
    // momentum observer (uses the torque applied in the previous cycle)
    const r = this.observer.update(d, this.nom, this.lastTau)
    if (this.collisionReaction && !this.reacted && this.t > 0.05) {
      for (let i = 0; i < n; i++) if (Math.abs(r[i]) > this.obsThreshold * this.limits.torque_limit[i]) this.reacted = true
    }
    this.lastTau = tau
    // external world: hand spring (guiding mode) and pushes on the TCP body
    const xf = data.xfrc_applied as Float64Array
    xf.fill(0, 6 * this.tcpBody, 6 * this.tcpBody + 6)
    const tcp = this.tcp()
    if (this.mode === 'guide') {
      const v = [0, 1, 2].map((i) => {
        let s = 0
        for (let k = 0; k < n; k++) s += d.J[i * n + k] * qd[k]
        return s
      })
      this.handForce = [0, 1, 2].map((i) => K_HAND * (this.target.p[i] - tcp.p[i]) - D_HAND * v[i]) as Vec3
      for (let i = 0; i < 3; i++) xf[6 * this.tcpBody + i] += this.handForce[i]
    } else this.handForce = [0, 0, 0]
    if (this.t < this.pushUntil) for (let i = 0; i < 3; i++) xf[6 * this.tcpBody + i] += this.pushForce[i]
    ;(data.ctrl as Float64Array).set(tau)
    mj.mj_step(model, data)
    this.t += model.opt.timestep
  }

  private contactT0 = -1
  private contactTime(): number {
    if (this.lastFt[2] > 1 && this.contactT0 < 0) this.contactT0 = this.t
    return this.contactT0 < 0 ? 0 : this.t - this.contactT0
  }

  sample(): Sample {
    const tcp = this.tcp()
    const dist = (a: ArrayLike<number>, b: ArrayLike<number>) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) * 1e3
    let tauRel = 0
    let rRel = 0
    for (let i = 0; i < this.n; i++) {
      tauRel = Math.max(tauRel, Math.abs(this.lastTau[i]) / this.limits.torque_limit[i])
      rRel = Math.max(rRel, Math.abs(this.observer.r[i]) / (this.obsThreshold * this.limits.torque_limit[i]))
    }
    return {
      t: this.t,
      eTarget: dist(tcp.p, this.target.p),
      eRef: dist(tcp.p, this.ref.p),
      tauRel,
      fz: this.lastFt[2],
      fHand: Math.hypot(...this.handForce),
      rRel,
    }
  }

  dispose(): void {
    this.nom.dispose()
    this.data.delete()
    this.model.delete()
  }
}
