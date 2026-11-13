/**
 * Off-line render of the home-page image: the Panda moving along a joint trajectory (earlier poses
 * as ghosts, the TCP path as a tube) and pressing its probe on a plate (contact force arrow).
 * Driven by ../make-hero.mjs.
 */
import * as THREE from 'three'
import { RobotScene } from '../../docs/.vitepress/theme/lib/robotScene'
import { quintic } from '../../docs/.vitepress/theme/lib/traj'

const Q0 = [-1.25, 0.05, 0.0, -1.55, 0.0, 1.65, 0.0]
const Q1 = [0.45, 0.38, 0.0, -2.05, 0.0, 2.43, 1.2]
const GHOSTS: [number, string][] = [
  [0.0, '#3987e5'],
  [0.5, '#1baf7a'],
]

declare global {
  interface Window {
    heroReady?: boolean
    heroInfo?: unknown
  }
}

async function main() {
  const host = document.getElementById('view')!
  const s = await RobotScene.create(host, { assetsUrl: '/models/panda/', transparent: true })
  const kin = s.kin
  const qAt = (u: number) => Q0.map((a, i) => quintic(a, Q1[i], 1).at(u).p)
  s.setQ(Q1)
  for (const [u, c] of GHOSTS) s.addGhost(qAt(u), c, 0.28)
  // TCP path
  const pts = Array.from({ length: 80 }, (_, k) => new THREE.Vector3(...kin.fk(qAt(k / 79)).p))
  const tube = new THREE.Mesh(
    new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 160, 0.005, 12),
    new THREE.MeshStandardMaterial({ color: 0x2a78d6, roughness: 0.35 }),
  )
  s.scene.add(tube)
  // plate under the probe and the contact force
  const end = kin.fk(Q1).p
  const plate = new THREE.Mesh(
    new THREE.BoxGeometry(0.34, 0.34, 0.02),
    new THREE.MeshStandardMaterial({ color: 0x8c9db5, roughness: 0.55, transparent: true, opacity: 0.85 }),
  )
  plate.position.set(end[0], end[1], end[2] - 0.012 - 0.01)
  plate.receiveShadow = true
  s.scene.add(plate)
  const arrow = new THREE.ArrowHelper(
    new THREE.Vector3(0, 0, 1),
    new THREE.Vector3(end[0] + 0.07, end[1] - 0.07, end[2] - 0.012),
    0.17,
    0xeb6834,
    0.045,
    0.026,
  )
  s.scene.add(arrow)
  s.camera.fov = 32
  s.camera.position.set(2.35, -1.25, 1.45)
  s.orbit.target.set(0.18, -0.12, 0.42)
  s.camera.updateProjectionMatrix()
  s.orbit.update()
  window.heroInfo = { end }
  setTimeout(() => (window.heroReady = true), 800)
}

void main()
