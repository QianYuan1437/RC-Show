import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { Kinematics, type RobotSpec } from '../docs/.vitepress/theme/lib/kinematics'

export const modelDir = (robot: string): string =>
  fileURLToPath(new URL(`../docs/public/models/${robot}/`, import.meta.url))

export const PANDA_DIR = modelDir('panda')

export function readJson<T>(name: string, robot = 'panda'): T {
  return JSON.parse(readFileSync(modelDir(robot) + name, 'utf8')) as T
}

export const loadRobot = (robot: string): Kinematics => new Kinematics(readJson<RobotSpec>('robot.json', robot))

export const loadPanda = (): Kinematics => loadRobot('panda')

/** MuJoCo WASM in Node (the module finds mujoco.wasm next to mujoco.js). */
let mjPromise: Promise<import('../docs/.vitepress/theme/lib/mujoco').Mujoco> | null = null
export function loadMujocoNode() {
  if (!mjPromise) mjPromise = import('@mujoco/mujoco').then(({ default: factory }) => factory())
  return mjPromise
}

export const readText = (name: string, robot = 'panda'): string => readFileSync(modelDir(robot) + name, 'utf8')

export function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]))
  return m
}
