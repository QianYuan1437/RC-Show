/**
 * Forward kinematics, geometric Jacobian and SO(3) utilities for a serial revolute chain.
 *
 * Conventions (identical to the Python package `ikzoo.kinematics`):
 *   - quaternions are wxyz, rotation matrices are row-major `Mat3` (length 9);
 *   - a pose is (p, R) expressed in the robot base (= world) frame, Z up;
 *   - the 6-D pose error is e = [p* - p ; Log(R* R^T)], so e_new ~= e - J dq with the
 *     world-frame geometric Jacobian J = [Jv; Jw] (6 x n, row-major).
 */

export type Vec3 = [number, number, number]
/** Row-major 3x3 matrix: R[3 * row + col]. */
export type Mat3 = number[]
export type Quat = [number, number, number, number]

export interface Pose {
  p: Vec3
  R: Mat3
}

export interface RobotLink {
  index: number
  file: string
  joint: number
}

/** Contents of `models/<robot>/robot.json`. */
export interface RobotSpec {
  name: string
  ndof: number
  joint_names: string[]
  lower: number[]
  upper: number[]
  home: number[]
  base_pos: number[]
  base_quat: number[]
  body_pos: number[][]
  body_quat: number[][]
  joint_axis: number[][]
  tcp_pos: number[]
  tcp_quat: number[]
  links: RobotLink[]
  convention?: string
}

// ----------------------------------------------------------------------------- vectors

export const dot = (a: ArrayLike<number>, b: ArrayLike<number>): number =>
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

export const cross = (a: ArrayLike<number>, b: ArrayLike<number>): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
]

export const add = (a: ArrayLike<number>, b: ArrayLike<number>): Vec3 => [
  a[0] + b[0],
  a[1] + b[1],
  a[2] + b[2],
]

export const sub = (a: ArrayLike<number>, b: ArrayLike<number>): Vec3 => [
  a[0] - b[0],
  a[1] - b[1],
  a[2] - b[2],
]

export const scale = (a: ArrayLike<number>, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s]

export const norm3 = (a: ArrayLike<number>): number => Math.hypot(a[0], a[1], a[2])

export function norm(a: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * a[i]
  return Math.sqrt(s)
}

// ----------------------------------------------------------------------------- matrices

export const eye3 = (): Mat3 => [1, 0, 0, 0, 1, 0, 0, 0, 1]

export function matMul3(A: Mat3, B: Mat3): Mat3 {
  const C = new Array<number>(9)
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      C[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j]
    }
  }
  return C
}

export const matVec3 = (A: Mat3, v: ArrayLike<number>): Vec3 => [
  A[0] * v[0] + A[1] * v[1] + A[2] * v[2],
  A[3] * v[0] + A[4] * v[1] + A[5] * v[2],
  A[6] * v[0] + A[7] * v[1] + A[8] * v[2],
]

export const transpose3 = (A: Mat3): Mat3 => [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]]

/** Column `j` of a 3x3 matrix. */
export const col3 = (A: Mat3, j: number): Vec3 => [A[j], A[3 + j], A[6 + j]]

/** Nested [[r0], [r1], [r2]] -> row-major Mat3. */
export const matFromRows = (rows: number[][]): Mat3 => [...rows[0], ...rows[1], ...rows[2]]

export const matToRows = (R: Mat3): number[][] => [R.slice(0, 3), R.slice(3, 6), R.slice(6, 9)]

export function skew(v: ArrayLike<number>): Mat3 {
  return [0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0]
}

// ----------------------------------------------------------------------------- SO(3)

/** Exponential map: rotation vector -> rotation matrix (Rodrigues). */
export function so3Exp(w: ArrayLike<number>): Mat3 {
  const th = norm3(w)
  const th2 = th * th
  let a: number
  let b: number
  if (th < 1e-8) {
    a = 1 - th2 / 6
    b = 0.5 - th2 / 24
  } else {
    a = Math.sin(th) / th
    b = (1 - Math.cos(th)) / th2
  }
  const K = skew(w)
  const K2 = matMul3(K, K)
  const R = eye3()
  for (let i = 0; i < 9; i++) R[i] += a * K[i] + b * K2[i]
  return R
}

/** Logarithm map: rotation matrix -> rotation vector, robust near 0 and near pi. */
export function so3Log(R: Mat3): Vec3 {
  const tr = R[0] + R[4] + R[8]
  const c = Math.min(1, Math.max(-1, (tr - 1) / 2))
  const w: Vec3 = [0.5 * (R[7] - R[5]), 0.5 * (R[2] - R[6]), 0.5 * (R[3] - R[1])]
  const s = norm3(w)
  const th = Math.atan2(s, c)
  if (c < -0.999) {
    // R + R^T = 2 cos(t) I + 2 (1 - cos(t)) a a^T: take the axis from the symmetric part,
    // using the column with the largest diagonal entry; the sign comes from the skew part.
    const d = 2 * (1 - c)
    const B = new Array<number>(9)
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        B[3 * i + j] = (R[3 * i + j] + R[3 * j + i] - (i === j ? 2 * c : 0)) / d
      }
    }
    let k = 0
    for (let i = 1; i < 3; i++) if (B[4 * i] > B[4 * k]) k = i
    const axis = col3(B, k)
    const n = Math.max(norm3(axis), 1e-12)
    let sgn = Math.sign(dot(axis, w))
    if (sgn === 0) sgn = 1
    return scale(axis, (sgn * th) / n)
  }
  return scale(w, s > 1e-9 ? th / s : 1 + (th * th) / 6)
}

/** Geodesic angle between two rotations (radians). */
export function rotAngle(Ra: Mat3, Rb: Mat3): number {
  const R = matMul3(Ra, transpose3(Rb))
  const w = [R[7] - R[5], R[2] - R[6], R[3] - R[1]]
  return Math.atan2(0.5 * norm3(w), Math.min(1, Math.max(-1, (R[0] + R[4] + R[8] - 1) / 2)))
}

/** wxyz quaternion -> rotation matrix. */
export function quatToMat(q: ArrayLike<number>): Mat3 {
  const n = Math.hypot(q[0], q[1], q[2], q[3])
  const w = q[0] / n
  const x = q[1] / n
  const y = q[2] / n
  const z = q[3] / n
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ]
}

/** Rotation matrix -> wxyz quaternion with w >= 0 (Shepperd's method). */
export function matToQuat(m: Mat3): Quat {
  const tr = m[0] + m[4] + m[8]
  let q: Quat
  if (tr > m[0] && tr > m[4] && tr > m[8]) {
    const s = 2 * Math.sqrt(1 + tr)
    q = [0.25 * s, (m[7] - m[5]) / s, (m[2] - m[6]) / s, (m[3] - m[1]) / s]
  } else if (m[0] > m[4] && m[0] > m[8]) {
    const s = 2 * Math.sqrt(1 + m[0] - m[4] - m[8])
    q = [(m[7] - m[5]) / s, 0.25 * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s]
  } else if (m[4] > m[8]) {
    const s = 2 * Math.sqrt(1 - m[0] + m[4] - m[8])
    q = [(m[2] - m[6]) / s, (m[1] + m[3]) / s, 0.25 * s, (m[5] + m[7]) / s]
  } else {
    const s = 2 * Math.sqrt(1 - m[0] - m[4] + m[8])
    q = [(m[3] - m[1]) / s, (m[2] + m[6]) / s, (m[5] + m[7]) / s, 0.25 * s]
  }
  const sgn = q[0] < 0 ? -1 : 1
  const n = Math.hypot(...q)
  return q.map((v) => (sgn * v) / n) as Quat
}

/** Hamilton product of two wxyz quaternions. */
export function quatMul(a: ArrayLike<number>, b: ArrayLike<number>): Quat {
  return [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ]
}

/** Continuous 6-D rotation representation: the first two columns of R. */
export const rot6d = (R: Mat3): number[] => [R[0], R[3], R[6], R[1], R[4], R[7]]

/** Inverse of `rot6d` via Gram-Schmidt. */
export function rot6dToMat(r: ArrayLike<number>): Mat3 {
  const a: Vec3 = [r[0], r[1], r[2]]
  const b0: Vec3 = [r[3], r[4], r[5]]
  const x = scale(a, 1 / norm3(a))
  const bp = sub(b0, scale(x, dot(x, b0)))
  const y = scale(bp, 1 / norm3(bp))
  const z = cross(x, y)
  return [x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]]
}

/** 6-D world-frame pose error [p* - p ; Log(R* R^T)]. */
export function poseError(p: Vec3, R: Mat3, pT: Vec3, RT: Mat3): number[] {
  const w = so3Log(matMul3(RT, transpose3(R)))
  return [pT[0] - p[0], pT[1] - p[1], pT[2] - p[2], w[0], w[1], w[2]]
}

/** Position (m) and rotation (rad) error norms of a pose w.r.t. a target. */
export function poseErrorNorms(pose: Pose, target: Pose): { pos: number; rot: number } {
  const e = poseError(pose.p, pose.R, target.p, target.R)
  return { pos: Math.hypot(e[0], e[1], e[2]), rot: Math.hypot(e[3], e[4], e[5]) }
}

// ----------------------------------------------------------------------------- random

export type Rng = () => number

/** Small seedable PRNG (mulberry32), uniform in [0, 1). */
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// ----------------------------------------------------------------------------- FK

export interface Frames {
  /** World position of each joint (n). */
  jointPos: Vec3[]
  /** World direction of each joint axis (n). */
  jointAxis: Vec3[]
  /** Frames of links 0..n (link 0 is the base). */
  links: Pose[]
  /** Tool centre point. */
  ee: Pose
}

export interface JacobianResult extends Frames {
  /** 6 x n row-major geometric Jacobian, rows [v; w]. */
  J: Float64Array
}

export class Kinematics {
  readonly spec: RobotSpec
  readonly n: number
  readonly lower: number[]
  readonly upper: number[]
  readonly home: number[]
  private readonly bodyP: Vec3[]
  private readonly bodyR: Mat3[]
  private readonly axis: Vec3[]
  private readonly baseP: Vec3
  private readonly baseR: Mat3
  private readonly eeP: Vec3
  private readonly eeR: Mat3

  constructor(spec: RobotSpec) {
    this.spec = spec
    this.n = spec.ndof
    this.lower = spec.lower.slice()
    this.upper = spec.upper.slice()
    this.home = spec.home.slice()
    this.bodyP = spec.body_pos.map((v) => [v[0], v[1], v[2]])
    this.bodyR = spec.body_quat.map(quatToMat)
    this.axis = spec.joint_axis.map((a) => {
      const n = norm3(a)
      return [a[0] / n, a[1] / n, a[2] / n]
    })
    this.baseP = [spec.base_pos[0], spec.base_pos[1], spec.base_pos[2]]
    this.baseR = quatToMat(spec.base_quat)
    this.eeP = [spec.tcp_pos[0], spec.tcp_pos[1], spec.tcp_pos[2]]
    this.eeR = quatToMat(spec.tcp_quat)
  }

  frames(q: ArrayLike<number>): Frames {
    let p: Vec3 = [...this.baseP]
    let R: Mat3 = this.baseR.slice()
    const jointPos: Vec3[] = []
    const jointAxis: Vec3[] = []
    const links: Pose[] = [{ p, R }]
    for (let i = 0; i < this.n; i++) {
      p = add(p, matVec3(R, this.bodyP[i]))
      R = matMul3(R, this.bodyR[i])
      jointPos.push(p)
      jointAxis.push(matVec3(R, this.axis[i]))
      R = matMul3(R, so3Exp(scale(this.axis[i], q[i])))
      links.push({ p, R })
    }
    const ee: Pose = { p: add(p, matVec3(R, this.eeP)), R: matMul3(R, this.eeR) }
    return { jointPos, jointAxis, links, ee }
  }

  fk(q: ArrayLike<number>): Pose {
    return this.frames(q).ee
  }

  jacobian(q: ArrayLike<number>): JacobianResult {
    const f = this.frames(q)
    const n = this.n
    const J = new Float64Array(6 * n)
    for (let i = 0; i < n; i++) {
      const a = f.jointAxis[i]
      const v = cross(a, sub(f.ee.p, f.jointPos[i]))
      for (let r = 0; r < 3; r++) {
        J[r * n + i] = v[r]
        J[(r + 3) * n + i] = a[r]
      }
    }
    return { ...f, J }
  }

  clamp(q: ArrayLike<number>): number[] {
    const out = new Array<number>(this.n)
    for (let i = 0; i < this.n; i++) out[i] = Math.min(this.upper[i], Math.max(this.lower[i], q[i]))
    return out
  }

  /** Uniform sample in the joint box. */
  sampleQ(rng: Rng = Math.random): number[] {
    return this.lower.map((lo, i) => lo + rng() * (this.upper[i] - lo))
  }

  /** A reachable target: FK of a uniformly sampled configuration. */
  randomTarget(rng: Rng = Math.random): { q: number[]; pose: Pose } {
    const q = this.sampleQ(rng)
    return { q, pose: this.fk(q) }
  }
}
