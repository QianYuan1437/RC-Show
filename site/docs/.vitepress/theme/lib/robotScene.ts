/**
 * three.js scene for a serial robot described by `robot.json` + per-link GLB meshes.
 * Browser only: import it lazily (e.g. from `onMounted`) so SSR never touches WebGL.
 */

import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { TransformControls } from 'three/addons/controls/TransformControls.js'
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import { toCreasedNormals } from 'three/addons/utils/BufferGeometryUtils.js'
import { Kinematics, type Mat3, type Pose, type RobotSpec, type Vec3 } from './kinematics'

export interface RobotAssets {
  spec: RobotSpec
  kin: Kinematics
  /** One merged geometry per link, in the link frame. */
  geometries: THREE.BufferGeometry[]
}

const assetCache = new Map<string, Promise<RobotAssets>>()

/** Load (once per URL) `robot.json` and the link meshes from a directory URL ending in '/'. */
export function loadRobotAssets(dirUrl: string): Promise<RobotAssets> {
  let p = assetCache.get(dirUrl)
  if (!p) {
    p = fetchRobotAssets(dirUrl)
    p.catch(() => assetCache.delete(dirUrl))
    assetCache.set(dirUrl, p)
  }
  return p
}

async function fetchRobotAssets(dirUrl: string): Promise<RobotAssets> {
  const res = await fetch(`${dirUrl}robot.json`)
  if (!res.ok) throw new Error(`无法加载 ${dirUrl}robot.json（HTTP ${res.status}）`)
  const spec = (await res.json()) as RobotSpec
  const loader = new GLTFLoader()
  const geometries = await Promise.all(
    [...spec.links].sort((a, b) => a.index - b.index).map(async (link) => {
      const gltf = await loader.loadAsync(dirUrl + link.file)
      return mergeScene(gltf.scene)
    }),
  )
  return { spec, kin: new Kinematics(spec), geometries }
}

/** Bake every mesh of a glTF scene into one geometry with creased vertex normals. */
function mergeScene(root: THREE.Object3D): THREE.BufferGeometry {
  root.updateMatrixWorld(true)
  const parts: THREE.BufferGeometry[] = []
  root.traverse((o) => {
    const mesh = o as THREE.Mesh
    if (!mesh.isMesh) return
    const g = mesh.geometry.index ? mesh.geometry.toNonIndexed() : mesh.geometry.clone()
    g.applyMatrix4(mesh.matrixWorld)
    g.setAttribute('color', linearVertexColors(g))
    parts.push(g)
  })
  const merged = parts.length === 1 ? parts[0] : concatGeometries(parts)
  // toCreasedNormals hashes positions at 1e-2 resolution; work in millimetres.
  merged.scale(1000, 1000, 1000)
  const out = toCreasedNormals(merged, THREE.MathUtils.degToRad(35))
  out.scale(1e-3, 1e-3, 1e-3)
  out.computeBoundingSphere()
  return out
}

/**
 * The exported vertex colours are MuJoCo material rgba values, i.e. *display* (sRGB) colours,
 * but glTF / three.js treat COLOR_0 as linear. Used as-is, "black" (0.25) renders as a mid
 * grey and the cool off-white (0.90, 0.92, 0.93) is pushed towards a pale cyan / mint once
 * lighting and tone mapping act on it. Convert to linear RGB (alpha dropped: all opaque).
 */
function linearVertexColors(g: THREE.BufferGeometry): THREE.BufferAttribute {
  const src = g.getAttribute('color')
  const n = g.getAttribute('position').count
  const out = new Float32Array(n * 3)
  const c = new THREE.Color()
  for (let i = 0; i < n; i++) {
    if (src) c.setRGB(src.getX(i), src.getY(i), src.getZ(i), THREE.SRGBColorSpace)
    else c.setRGB(0.85, 0.85, 0.85, THREE.SRGBColorSpace)
    out[3 * i] = c.r
    out[3 * i + 1] = c.g
    out[3 * i + 2] = c.b
  }
  return new THREE.BufferAttribute(out, 3)
}

function concatGeometries(parts: THREE.BufferGeometry[]): THREE.BufferGeometry {
  const out = new THREE.BufferGeometry()
  for (const name of ['position', 'color']) {
    const itemSize = name === 'position' ? 3 : parts[0].getAttribute(name).itemSize
    const total = parts.reduce((s, g) => s + g.getAttribute(name).count, 0)
    const arr = new Float32Array(total * itemSize)
    let off = 0
    for (const g of parts) {
      const a = g.getAttribute(name)
      for (let i = 0; i < a.count; i++) {
        for (let k = 0; k < itemSize; k++) arr[off++] = a.getComponent(i, k)
      }
    }
    out.setAttribute(name, new THREE.BufferAttribute(arr, itemSize))
  }
  return out
}

// ----------------------------------------------------------------------------- helpers

export function poseToMatrix(p: ArrayLike<number>, R: Mat3, m = new THREE.Matrix4()): THREE.Matrix4 {
  return m.set(R[0], R[1], R[2], p[0], R[3], R[4], R[5], p[1], R[6], R[7], R[8], p[2], 0, 0, 0, 1)
}

export function matrixToPose(m: THREE.Matrix4): Pose {
  const e = m.elements // column-major
  const R: Mat3 = [e[0], e[4], e[8], e[1], e[5], e[9], e[2], e[6], e[10]]
  // Remove any scale so R stays orthonormal.
  for (let j = 0; j < 3; j++) {
    const n = Math.hypot(R[j], R[3 + j], R[6 + j]) || 1
    R[j] /= n
    R[3 + j] /= n
    R[6 + j] /= n
  }
  return { p: [e[12], e[13], e[14]], R }
}

/** RGB axes triad made of thin cylinders (visible at any line-width limit). */
export function makeTriad(length = 0.1, radius = 0.004, opacity = 1): THREE.Group {
  const g = new THREE.Group()
  const colors = [0xe5484d, 0x30a46c, 0x3e63dd]
  const geo = new THREE.CylinderGeometry(radius, radius, length, 10)
  geo.translate(0, length / 2, 0)
  const tip = new THREE.ConeGeometry(radius * 2.2, radius * 5, 12)
  tip.translate(0, length + radius * 2.5, 0)
  colors.forEach((c, k) => {
    const mat = new THREE.MeshBasicMaterial({ color: c, transparent: opacity < 1, opacity, depthTest: true })
    const axis = new THREE.Group()
    axis.add(new THREE.Mesh(geo, mat), new THREE.Mesh(tip, mat))
    if (k === 0) axis.rotation.z = -Math.PI / 2
    if (k === 2) axis.rotation.x = Math.PI / 2
    g.add(axis)
  })
  return g
}

// ----------------------------------------------------------------------------- scene

export type GizmoMode = 'translate' | 'rotate'
export type TargetListener = (pose: Pose, dragging: boolean) => void

export interface RobotSceneOptions {
  /** Directory URL containing robot.json and the link GLBs (already passed through withBase). */
  assetsUrl: string
  gizmo?: boolean
  trail?: boolean
  dark?: boolean
  shadows?: boolean
  /** Transparent background without the floor grid (offline renders such as the home-page image). */
  transparent?: boolean
}

interface LinkSet {
  group: THREE.Group
  meshes: THREE.Mesh[]
}

const TRAIL_MAX = 4000
const _m = new THREE.Matrix4()

export class RobotScene {
  readonly kin: Kinematics
  readonly spec: RobotSpec
  readonly renderer: THREE.WebGLRenderer
  readonly scene = new THREE.Scene()
  readonly camera: THREE.PerspectiveCamera
  readonly orbit: OrbitControls

  private readonly container: HTMLElement
  private readonly assets: RobotAssets
  private readonly robot: LinkSet
  private readonly ghosts: LinkSet[] = []
  private readonly tcpTriad = makeTriad(0.09, 0.0035)
  private readonly target = new THREE.Group()
  private readonly gizmo: TransformControls
  private readonly gizmoHelper: THREE.Object3D
  private readonly grid: THREE.Group = new THREE.Group()
  private readonly trail: THREE.Line
  private trailCount = 0
  private trailEnabled: boolean
  private readonly listeners = new Set<TargetListener>()
  private readonly resizeObserver: ResizeObserver
  private readonly visibilityObserver: IntersectionObserver
  private visible = true
  private qCur: number[]
  private disposed = false
  private readonly transparent: boolean
  private readonly materials: THREE.Material[] = []

  static async create(container: HTMLElement, opts: RobotSceneOptions): Promise<RobotScene> {
    const assets = await loadRobotAssets(opts.assetsUrl)
    return new RobotScene(container, assets, opts)
  }

  private constructor(container: HTMLElement, assets: RobotAssets, opts: RobotSceneOptions) {
    this.container = container
    this.assets = assets
    this.kin = assets.kin
    this.spec = assets.spec
    this.trailEnabled = opts.trail ?? false
    this.transparent = opts.transparent ?? false

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: this.transparent,
      preserveDrawingBuffer: this.transparent,
    })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.outputColorSpace = THREE.SRGBColorSpace
    // Khronos PBR Neutral keeps whites white and hues unshifted (ACES desaturates and tints
    // bright off-whites, which made the arm look greenish-grey).
    renderer.toneMapping = THREE.NeutralToneMapping
    renderer.toneMappingExposure = 1.0
    renderer.shadowMap.enabled = opts.shadows ?? true
    renderer.shadowMap.type = THREE.PCFShadowMap
    renderer.domElement.style.display = 'block'
    renderer.domElement.style.width = '100%'
    renderer.domElement.style.height = '100%'
    renderer.domElement.style.touchAction = 'none'
    container.appendChild(renderer.domElement)
    this.renderer = renderer

    const pmrem = new THREE.PMREMGenerator(renderer)
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture
    this.scene.environmentIntensity = 0.45
    pmrem.dispose()

    const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 50)
    camera.up.set(0, 0, 1)
    this.camera = camera
    this.orbit = new OrbitControls(camera, renderer.domElement)
    this.orbit.enableDamping = true
    this.orbit.dampingFactor = 0.12
    this.orbit.maxDistance = 8
    this.orbit.minDistance = 0.3
    this.resetCamera()

    this.addLights()
    this.addGround()

    this.robot = this.makeLinkSet((i) => {
      // Painted plastic: dielectric (metalness 0), fairly glossy.
      const m = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.38, metalness: 0 })
      this.materials.push(m)
      return m
    }, true)
    this.scene.add(this.robot.group)
    this.tcpTriad.matrixAutoUpdate = false
    this.scene.add(this.tcpTriad)

    // Target marker + gizmo.
    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(0.014, 24, 16),
      new THREE.MeshStandardMaterial({ color: 0xf59e0b, roughness: 0.3, transparent: true, opacity: 0.85 }),
    )
    this.target.add(sphere, makeTriad(0.12, 0.004, 0.75))
    this.target.visible = opts.gizmo ?? false
    this.scene.add(this.target)
    this.gizmo = new TransformControls(camera, renderer.domElement)
    this.gizmo.setSize(0.85)
    this.gizmo.setSpace('local')
    this.gizmo.attach(this.target)
    this.gizmo.enabled = opts.gizmo ?? false
    this.gizmoHelper = this.gizmo.getHelper()
    this.gizmoHelper.visible = opts.gizmo ?? false
    this.scene.add(this.gizmoHelper)
    this.gizmo.addEventListener('dragging-changed', (ev) => {
      const dragging = Boolean((ev as unknown as { value: boolean }).value)
      this.orbit.enabled = !dragging
      if (!dragging) this.emitTarget(false)
    })
    this.gizmo.addEventListener('objectChange', () => this.emitTarget(this.gizmo.dragging))

    // TCP trail.
    const trailGeo = new THREE.BufferGeometry()
    trailGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(TRAIL_MAX * 3), 3))
    trailGeo.setDrawRange(0, 0)
    this.trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: 0xf97316 }))
    this.trail.frustumCulled = false
    this.scene.add(this.trail)

    this.setDark(opts.dark ?? false)
    this.qCur = this.kin.home.slice()
    this.setQ(this.qCur)
    this.setTarget(this.kin.fk(this.qCur))

    this.resizeObserver = new ResizeObserver(() => this.resize())
    this.resizeObserver.observe(container)
    this.visibilityObserver = new IntersectionObserver((entries) => {
      this.visible = entries.some((e) => e.isIntersecting)
    })
    this.visibilityObserver.observe(container)
    this.resize()
    renderer.setAnimationLoop(() => this.frame())
  }

  // --------------------------------------------------------------------------- public API

  get q(): number[] {
    return this.qCur.slice()
  }

  /** Set the joint configuration of the main robot (and extend the TCP trail if enabled). */
  setQ(q: ArrayLike<number>): void {
    this.qCur = Array.from(q)
    const f = this.kin.frames(this.qCur)
    this.placeLinks(this.robot, f.links)
    poseToMatrix(f.ee.p, f.ee.R, this.tcpTriad.matrix)
    this.tcpTriad.matrixWorldNeedsUpdate = true
    if (this.trailEnabled) this.pushTrail(f.ee.p)
  }

  /** Add a semi-transparent copy of the robot at configuration q. */
  addGhost(q: ArrayLike<number>, color: THREE.ColorRepresentation = 0x4f8cff, opacity = 0.3): THREE.Group {
    // Two passes per ghost: a depth-only pre-pass, then the tinted colour pass with depth test
    // "less or equal". Only the ghost's front-most surface is shaded, so a ghost reads as one
    // clean translucent silhouette instead of a stack of overlapping inner faces.
    // Both passes are pushed slightly back (polygon offset) so that wherever a ghost coincides
    // with the real arm (shared links of near-identical solutions) the opaque arm wins instead
    // of being tinted by the ghost colour.
    const offset = { polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 8 }
    const depthMat = new THREE.MeshBasicMaterial({ colorWrite: false, ...offset })
    const mat = new THREE.MeshStandardMaterial({
      color,
      transparent: true,
      opacity,
      depthWrite: false,
      depthFunc: THREE.LessEqualDepth,
      roughness: 0.55,
      metalness: 0,
      ...offset,
    })
    this.materials.push(mat, depthMat)
    const order = 1 + 2 * this.ghosts.length
    const pre = this.makeLinkSet(() => depthMat, false)
    pre.meshes.forEach((m) => (m.renderOrder = order))
    const ghost = this.makeLinkSet(() => mat, false)
    ghost.meshes.forEach((m) => (m.renderOrder = order + 1))
    const links = this.kin.frames(q).links
    this.placeLinks(pre, links)
    this.placeLinks(ghost, links)
    // The base link is identical for every configuration: drawing it would only tint the real base.
    pre.meshes[0].visible = false
    ghost.meshes[0].visible = false
    ghost.group.add(pre.group)
    ghost.meshes.push(...pre.meshes)
    this.ghosts.push(ghost)
    this.scene.add(ghost.group)
    return ghost.group
  }

  clearGhosts(): void {
    for (const g of this.ghosts) {
      this.scene.remove(g.group)
      const mats = new Set(g.meshes.map((m) => m.material as THREE.Material))
      for (const mat of mats) {
        mat.dispose()
        const k = this.materials.indexOf(mat)
        if (k >= 0) this.materials.splice(k, 1)
      }
    }
    this.ghosts.length = 0
  }

  /** Move the target marker (does not emit a change event). */
  setTarget(pose: Pose): void {
    poseToMatrix(pose.p, pose.R, _m).decompose(this.target.position, this.target.quaternion, this.target.scale)
    this.target.updateMatrixWorld()
  }

  getTarget(): Pose {
    this.target.updateMatrix()
    return matrixToPose(this.target.matrix)
  }

  /** Show/hide the target marker and its gizmo. */
  setGizmoEnabled(on: boolean): void {
    this.target.visible = on
    this.gizmo.enabled = on
    this.gizmoHelper.visible = on
  }

  setGizmoMode(mode: GizmoMode): void {
    this.gizmo.setMode(mode)
  }

  /** Subscribe to target changes; returns an unsubscribe function. */
  onTargetChange(cb: TargetListener): () => void {
    this.listeners.add(cb)
    return () => this.listeners.delete(cb)
  }

  setTrailEnabled(on: boolean): void {
    this.trailEnabled = on
    this.trail.visible = on
    if (!on) this.clearTrail()
  }

  clearTrail(): void {
    this.trailCount = 0
    this.trail.geometry.setDrawRange(0, 0)
  }

  setDark(dark: boolean): void {
    this.scene.background = this.transparent ? null : new THREE.Color(dark ? 0x1b1b1f : 0xf3f4f6)
    this.grid.visible = !this.transparent
    this.grid.children.forEach((c, i) => {
      const mat = (c as THREE.LineSegments).material as THREE.LineBasicMaterial
      mat.color.set(dark ? (i === 0 ? 0x3a3a44 : 0x2a2a31) : i === 0 ? 0xc4c8cf : 0xe0e3e8)
    })
  }

  resetCamera(): void {
    this.camera.position.set(1.75, -1.55, 1.05)
    this.orbit.target.set(0.28, 0, 0.36)
    this.orbit.update()
  }

  dispose(): void {
    if (this.disposed) return
    this.disposed = true
    this.renderer.setAnimationLoop(null)
    this.resizeObserver.disconnect()
    this.visibilityObserver.disconnect()
    this.listeners.clear()
    this.gizmo.detach()
    this.gizmo.dispose()
    this.orbit.dispose()
    this.materials.forEach((m) => m.dispose())
    this.scene.environment?.dispose()
    this.trail.geometry.dispose()
    this.renderer.dispose()
    this.renderer.forceContextLoss()
    this.renderer.domElement.remove()
  }

  // --------------------------------------------------------------------------- internals

  private makeLinkSet(material: (i: number) => THREE.Material, shadows: boolean): LinkSet {
    const group = new THREE.Group()
    const meshes = this.assets.geometries.map((geo, i) => {
      const mesh = new THREE.Mesh(geo, material(i))
      mesh.matrixAutoUpdate = false
      mesh.castShadow = shadows
      mesh.receiveShadow = shadows
      group.add(mesh)
      return mesh
    })
    return { group, meshes }
  }

  private placeLinks(set: LinkSet, links: Pose[]): void {
    set.meshes.forEach((mesh, i) => {
      const f = links[i]
      if (!f) return
      poseToMatrix(f.p, f.R, mesh.matrix)
      mesh.matrixWorldNeedsUpdate = true
    })
  }

  private addLights(): void {
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8d8d8d, 1.4))
    const sun = new THREE.DirectionalLight(0xffffff, 2.2)
    sun.position.set(1.5, -1.0, 2.5)
    sun.castShadow = true
    sun.shadow.mapSize.set(2048, 2048)
    const c = sun.shadow.camera
    c.left = -1.2
    c.right = 1.2
    c.top = 1.2
    c.bottom = -1.2
    c.near = 0.5
    c.far = 6
    sun.shadow.bias = -0.0004
    sun.shadow.normalBias = 0.01
    sun.shadow.radius = 4
    this.scene.add(sun)
    const fill = new THREE.DirectionalLight(0xffffff, 0.5)
    fill.position.set(-1.5, 1.5, 1.0)
    this.scene.add(fill)
  }

  private addGround(): void {
    const major = new THREE.GridHelper(4, 8, 0xffffff, 0xffffff)
    const minor = new THREE.GridHelper(4, 40, 0xffffff, 0xffffff)
    for (const g of [major, minor]) {
      g.rotation.x = Math.PI / 2
      const mat = g.material as THREE.LineBasicMaterial
      mat.transparent = true
      mat.opacity = 0.9
      mat.depthWrite = false
      this.grid.add(g)
    }
    minor.position.z = -0.0005
    this.scene.add(this.grid)
    const shadowPlane = new THREE.Mesh(
      new THREE.PlaneGeometry(4, 4),
      new THREE.ShadowMaterial({ opacity: 0.18 }),
    )
    shadowPlane.position.z = 0.0005
    shadowPlane.receiveShadow = true
    this.scene.add(shadowPlane)
    this.scene.add(makeTriad(0.15, 0.003, 0.6))
  }

  private pushTrail(p: Vec3): void {
    const attr = this.trail.geometry.getAttribute('position') as THREE.BufferAttribute
    const arr = attr.array as Float32Array
    if (this.trailCount >= TRAIL_MAX) {
      arr.copyWithin(0, 3)
      this.trailCount = TRAIL_MAX - 1
    }
    arr.set(p, this.trailCount * 3)
    this.trailCount++
    attr.needsUpdate = true
    this.trail.geometry.setDrawRange(0, this.trailCount)
  }

  private emitTarget(dragging: boolean): void {
    const pose = this.getTarget()
    this.listeners.forEach((cb) => cb(pose, dragging))
  }

  private resize(): void {
    const w = this.container.clientWidth
    const h = this.container.clientHeight
    if (w === 0 || h === 0) return
    this.renderer.setSize(w, h, false)
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
  }

  private frame(): void {
    if (!this.visible) return
    this.orbit.update()
    this.renderer.render(this.scene, this.camera)
  }
}
