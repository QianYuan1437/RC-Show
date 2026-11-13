/**
 * Thin wrapper around onnxruntime-web (WASM backend, single-threaded so it works on static
 * hosts without cross-origin isolation). Browser only; the runtime is loaded lazily.
 */

import type * as Ort from 'onnxruntime-web'
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url'

export type OrtModule = typeof Ort
export type Session = Ort.InferenceSession

export interface TensorLike {
  data: Float32Array | number[]
  dims: number[]
}

export interface TensorOut {
  data: Float32Array
  dims: number[]
}

/** Raised when an .onnx file is missing (typically: the model has not been exported yet). */
export class ModelNotFoundError extends Error {
  readonly url: string
  constructor(url: string) {
    super(`模型尚未导出：${url}`)
    this.name = 'ModelNotFoundError'
    this.url = url
  }
}

let ortPromise: Promise<OrtModule> | null = null

export function getOrt(): Promise<OrtModule> {
  if (!ortPromise) {
    ortPromise = import('onnxruntime-web/wasm').then((mod) => {
      const ort = ((mod as { default?: OrtModule }).default ?? mod) as OrtModule
      ort.env.wasm.wasmPaths = { wasm: ortWasmUrl }
      ort.env.wasm.numThreads = 1
      ort.env.wasm.proxy = false
      return ort
    })
    ortPromise.catch(() => (ortPromise = null))
  }
  return ortPromise
}

const sessionCache = new Map<string, Promise<Session>>()

/**
 * Fetch and initialise an ONNX model (cached per URL). `url` must already include the site
 * base, e.g. `withBase('/models/onnx/mlp.onnx')`.
 */
export function loadModel(url: string): Promise<Session> {
  let p = sessionCache.get(url)
  if (!p) {
    p = (async () => {
      const res = await fetch(url)
      if (res.status === 404) throw new ModelNotFoundError(url)
      if (!res.ok) throw new Error(`加载模型失败（HTTP ${res.status}）：${url}`)
      const bytes = new Uint8Array(await res.arrayBuffer())
      // Dev servers may answer unknown paths with an HTML page instead of a 404.
      if (bytes.length === 0 || bytes[0] === 0x3c) throw new ModelNotFoundError(url)
      const ort = await getOrt()
      return ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })
    })()
    p.catch(() => sessionCache.delete(url))
    sessionCache.set(url, p)
  }
  return p
}

/** Run a session on float32 feeds; returns plain {data, dims} outputs. */
export async function run(session: Session, feeds: Record<string, TensorLike>): Promise<Record<string, TensorOut>> {
  const ort = await getOrt()
  const inputs: Record<string, Ort.Tensor> = {}
  for (const [name, t] of Object.entries(feeds)) {
    const data = t.data instanceof Float32Array ? t.data : Float32Array.from(t.data)
    inputs[name] = new ort.Tensor('float32', data, t.dims)
  }
  const out = await session.run(inputs)
  const res: Record<string, TensorOut> = {}
  for (const [name, t] of Object.entries(out)) {
    res[name] = { data: t.data as Float32Array, dims: [...t.dims] }
    t.dispose()
  }
  return res
}

export interface OnnxManifestEntry {
  file: string
  inputs?: Record<string, unknown>
  outputs?: Record<string, unknown>
  [key: string]: unknown
}

/** Read `models/onnx/manifest.json`; resolves to null if it does not exist yet. */
export async function loadManifest(url: string): Promise<Record<string, OnnxManifestEntry> | null> {
  const res = await fetch(url)
  if (!res.ok) return null
  try {
    return (await res.json()) as Record<string, OnnxManifestEntry>
  } catch {
    return null
  }
}
