/**
 * Minimal wrapper around the official MuJoCo WASM bindings (`@mujoco/mujoco`, single-threaded
 * build). Browser only; the module (~11 MB of wasm) is loaded lazily and cached.
 */

import type { MainModule, MjData, MjModel } from '@mujoco/mujoco'
import mujocoWasmUrl from '@mujoco/mujoco/mujoco.wasm?url'
import type { Mat3, Vec3 } from './kinematics'

export type Mujoco = MainModule

let mjPromise: Promise<Mujoco> | null = null

export function loadMujoco(): Promise<Mujoco> {
  if (!mjPromise) {
    mjPromise = import('@mujoco/mujoco').then(({ default: factory }) =>
      factory({ locateFile: (path: string) => (path.endsWith('.wasm') ? mujocoWasmUrl : path) }),
    )
    mjPromise.catch(() => (mjPromise = null))
  }
  return mjPromise
}

export class MujocoSim {
  readonly mj: Mujoco
  readonly model: MjModel
  readonly data: MjData
  readonly nq: number
  readonly nu: number
  readonly timestep: number
  private disposed = false

  constructor(mj: Mujoco, model: MjModel) {
    this.mj = mj
    this.model = model
    this.data = new mj.MjData(model)
    this.nq = model.nq
    this.nu = model.nu
    this.timestep = model.opt.timestep
  }

  get time(): number {
    return this.data.time
  }

  /** Reset to qpos (default: keyframe 0 if present, else the model's qpos0) with zero velocity. */
  reset(qpos?: ArrayLike<number>): void {
    const { mj, model, data } = this
    if (model.nkey > 0) mj.mj_resetDataKeyframe(model, data, 0)
    else mj.mj_resetData(model, data)
    if (qpos) {
      const view = data.qpos as Float64Array
      for (let i = 0; i < Math.min(qpos.length, this.nq); i++) view[i] = qpos[i]
      ;(data.qvel as Float64Array).fill(0)
    }
    mj.mj_forward(model, data)
  }

  setCtrl(ctrl: ArrayLike<number>): void {
    const view = this.data.ctrl as Float64Array
    for (let i = 0; i < Math.min(ctrl.length, this.nu); i++) view[i] = ctrl[i]
  }

  step(n = 1): void {
    for (let i = 0; i < n; i++) this.mj.mj_step(this.model, this.data)
  }

  get qpos(): number[] {
    return Array.from(this.data.qpos as Float64Array)
  }

  get qvel(): number[] {
    return Array.from(this.data.qvel as Float64Array)
  }

  siteId(name: string): number {
    return this.mj.mj_name2id(this.model, this.mj.mjtObj.mjOBJ_SITE.value, name)
  }

  /** World pose of a site (row-major rotation). */
  sitePose(name = 'tcp'): { p: Vec3; R: Mat3 } {
    const id = this.siteId(name)
    if (id < 0) throw new Error(`MuJoCo 模型中没有 site "${name}"`)
    const xp = this.data.site_xpos as Float64Array
    const xm = this.data.site_xmat as Float64Array
    return {
      p: [xp[3 * id], xp[3 * id + 1], xp[3 * id + 2]],
      R: Array.from(xm.subarray(9 * id, 9 * id + 9)),
    }
  }

  dispose(): void {
    if (this.disposed) return
    this.disposed = true
    this.data.delete()
    this.model.delete()
  }
}

/** Fetch an MJCF file (URL already passed through withBase) and compile it. */
export async function loadModel(xmlUrl: string): Promise<MujocoSim> {
  const [mj, xml] = await Promise.all([
    loadMujoco(),
    fetch(xmlUrl).then((r) => {
      if (!r.ok) throw new Error(`无法加载 ${xmlUrl}（HTTP ${r.status}）`)
      return r.text()
    }),
  ])
  return new MujocoSim(mj, mj.MjModel.from_xml_string(xml))
}
