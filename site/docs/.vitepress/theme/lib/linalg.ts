/** Small dense linear algebra on row-major Float64Arrays (sizes up to ~10). */

export type Mat = Float64Array

/** C = A B with A (m x k), B (k x n). */
export function matmul(A: Mat, B: Mat, m: number, k: number, n: number): Mat {
  const C = new Float64Array(m * n)
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const a = A[i * k + p]
      if (a === 0) continue
      for (let j = 0; j < n; j++) C[i * n + j] += a * B[p * n + j]
    }
  }
  return C
}

export function transpose(A: Mat, m: number, n: number): Mat {
  const T = new Float64Array(m * n)
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) T[j * m + i] = A[i * n + j]
  return T
}

/** y = A x with A (m x n). */
export function matvec(A: Mat, x: ArrayLike<number>, m: number, n: number): Float64Array {
  const y = new Float64Array(m)
  for (let i = 0; i < m; i++) {
    let s = 0
    for (let j = 0; j < n; j++) s += A[i * n + j] * x[j]
    y[i] = s
  }
  return y
}

/** y = A^T x with A (m x n). */
export function matTvec(A: Mat, x: ArrayLike<number>, m: number, n: number): Float64Array {
  const y = new Float64Array(n)
  for (let i = 0; i < m; i++) {
    const xi = x[i]
    for (let j = 0; j < n; j++) y[j] += A[i * n + j] * xi
  }
  return y
}

/** A A^T + lambda I for A (m x n). */
export function gramRows(A: Mat, m: number, n: number, lambda = 0): Mat {
  const G = new Float64Array(m * m)
  for (let i = 0; i < m; i++) {
    for (let j = i; j < m; j++) {
      let s = 0
      for (let k = 0; k < n; k++) s += A[i * n + k] * A[j * n + k]
      G[i * m + j] = G[j * m + i] = s
    }
    G[i * m + i] += lambda
  }
  return G
}

/** A^T A + lambda I for A (m x n). */
export function gramCols(A: Mat, m: number, n: number, lambda = 0): Mat {
  const G = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let s = 0
      for (let k = 0; k < m; k++) s += A[k * n + i] * A[k * n + j]
      G[i * n + j] = G[j * n + i] = s
    }
    G[i * n + i] += lambda
  }
  return G
}

/** Solve A x = b for symmetric positive definite A (n x n) by Cholesky. Returns null if not SPD. */
export function solveSPD(A: Mat, b: ArrayLike<number>, n: number): Float64Array | null {
  const L = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i * n + j]
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k]
      if (i === j) {
        if (!(s > 0)) return null
        L[i * n + i] = Math.sqrt(s)
      } else {
        L[i * n + j] = s / L[j * n + j]
      }
    }
  }
  const y = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let s = b[i]
    for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k]
    y[i] = s / L[i * n + i]
  }
  const x = new Float64Array(n)
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i]
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k]
    x[i] = s / L[i * n + i]
  }
  return x
}

/** Solve A x = b for a general square A by Gaussian elimination with partial pivoting. */
export function solve(A: Mat, b: ArrayLike<number>, n: number): Float64Array {
  const M = Float64Array.from(A)
  const x = Float64Array.from(b as ArrayLike<number>)
  for (let c = 0; c < n; c++) {
    let piv = c
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r * n + c]) > Math.abs(M[piv * n + c])) piv = r
    if (piv !== c) {
      for (let k = 0; k < n; k++) {
        const t = M[c * n + k]
        M[c * n + k] = M[piv * n + k]
        M[piv * n + k] = t
      }
      const t = x[c]
      x[c] = x[piv]
      x[piv] = t
    }
    const d = M[c * n + c] || 1e-300
    for (let r = c + 1; r < n; r++) {
      const f = M[r * n + c] / d
      if (f === 0) continue
      for (let k = c; k < n; k++) M[r * n + k] -= f * M[c * n + k]
      x[r] -= f * x[c]
    }
  }
  for (let r = n - 1; r >= 0; r--) {
    let s = x[r]
    for (let k = r + 1; k < n; k++) s -= M[r * n + k] * x[k]
    x[r] = s / (M[r * n + r] || 1e-300)
  }
  return x
}

/** Solve an SPD system, falling back to pivoted elimination when Cholesky fails. */
export function solveSym(A: Mat, b: ArrayLike<number>, n: number): Float64Array {
  return solveSPD(A, b, n) ?? solve(A, b, n)
}

export interface SVD {
  /** m x r, orthonormal columns. */
  U: Mat
  /** r singular values, descending. */
  S: Float64Array
  /** n x r, orthonormal columns (A = U diag(S) V^T). */
  V: Mat
  r: number
}

/**
 * Thin SVD of A (m x n) by one-sided Jacobi rotations, r = min(m, n).
 * Accurate to machine precision for the small matrices used here.
 */
export function svd(A: Mat, m: number, n: number): SVD {
  // Orthogonalise the columns of B (rows x cols, rows >= cols); B V = W, W_k = sigma_k u_k.
  const wide = n > m
  const rows = wide ? n : m
  const cols = wide ? m : n
  const B = wide ? transpose(A, m, n) : Float64Array.from(A)
  const V = new Float64Array(cols * cols)
  for (let i = 0; i < cols; i++) V[i * cols + i] = 1
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0
    for (let p = 0; p < cols - 1; p++) {
      for (let q = p + 1; q < cols; q++) {
        let alpha = 0
        let beta = 0
        let gamma = 0
        for (let i = 0; i < rows; i++) {
          const bp = B[i * cols + p]
          const bq = B[i * cols + q]
          alpha += bp * bp
          beta += bq * bq
          gamma += bp * bq
        }
        if (gamma === 0) continue
        off = Math.max(off, Math.abs(gamma) / Math.sqrt(alpha * beta))
        const zeta = (beta - alpha) / (2 * gamma)
        const t = Math.sign(zeta || 1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta))
        const c = 1 / Math.sqrt(1 + t * t)
        const s = c * t
        for (let i = 0; i < rows; i++) {
          const bp = B[i * cols + p]
          const bq = B[i * cols + q]
          B[i * cols + p] = c * bp - s * bq
          B[i * cols + q] = s * bp + c * bq
        }
        for (let i = 0; i < cols; i++) {
          const vp = V[i * cols + p]
          const vq = V[i * cols + q]
          V[i * cols + p] = c * vp - s * vq
          V[i * cols + q] = s * vp + c * vq
        }
      }
    }
    if (off < 1e-15) break
  }
  const sig = new Float64Array(cols)
  for (let k = 0; k < cols; k++) {
    let s = 0
    for (let i = 0; i < rows; i++) s += B[i * cols + k] ** 2
    sig[k] = Math.sqrt(s)
  }
  const order = Array.from({ length: cols }, (_, i) => i).sort((a, b) => sig[b] - sig[a])
  const S = new Float64Array(cols)
  const Uw = new Float64Array(rows * cols) // left vectors of B
  const Vw = new Float64Array(cols * cols) // right vectors of B
  order.forEach((k, j) => {
    S[j] = sig[k]
    for (let i = 0; i < rows; i++) Uw[i * cols + j] = sig[k] > 0 ? B[i * cols + k] / sig[k] : 0
    for (let i = 0; i < cols; i++) Vw[i * cols + j] = V[i * cols + k]
  })
  // For the wide case A^T = Uw S Vw^T, i.e. A = Vw S Uw^T.
  return wide ? { U: Vw, S, V: Uw, r: cols } : { U: Uw, S, V: Vw, r: cols }
}

/** Moore-Penrose pseudo-inverse applied to a vector: x = A^+ b. */
export function pinvSolve(A: Mat, b: ArrayLike<number>, m: number, n: number, rtol?: number): Float64Array {
  const { U, S, V, r } = svd(A, m, n)
  const tol = (rtol ?? Math.max(m, n) * Number.EPSILON) * S[0]
  const x = new Float64Array(n)
  for (let k = 0; k < r; k++) {
    if (S[k] <= tol) continue
    let c = 0
    for (let i = 0; i < m; i++) c += U[i * r + k] * b[i]
    c /= S[k]
    for (let j = 0; j < n; j++) x[j] += V[j * r + k] * c
  }
  return x
}
