use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_matmul::{
    Fp16MatmulExecutor, Fp16MatmulPool, Fp16MatmulPoolPreparedWeights, Fp16PrepackedWeights,
    MatmulError, MatmulPoolError, PoolPreparedWeightStats, PrepackedWeightStats,
};
use rocknpu_tensor::{Matrix, MatrixShape, TensorError, TensorLayout};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulPrecision {
    Fp16Fast,
    Fp32Accurate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionTarget {
    Auto,
    Cpu,
    NpuSingle,
    NpuPool,
}

/// Logical operation: C[M,N] = A[M,K] x B[N,K]^T.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulSpec {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub precision: MatmulPrecision,
    pub target: ExecutionTarget,
}

impl MatmulSpec {
    pub const fn new(
        m: usize,
        k: usize,
        n: usize,
        precision: MatmulPrecision,
        target: ExecutionTarget,
    ) -> Self {
        Self {
            m,
            k,
            n,
            precision,
            target,
        }
    }
    pub const fn lhs_shape(self) -> MatrixShape {
        MatrixShape::new(self.m, self.k)
    }
    pub const fn rhs_shape(self) -> MatrixShape {
        MatrixShape::new(self.n, self.k)
    }
    pub const fn output_shape(self) -> MatrixShape {
        MatrixShape::new(self.m, self.n)
    }
    pub const fn npu_shape_supported(self) -> bool {
        self.m > 0
            && self.k > 0
            && self.n > 0
            && self.m % 4 == 0
            && self.k % 32 == 0
            && self.n % 16 == 0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MatmulOutput {
    F16(Matrix<f16>),
    F32(Matrix<f32>),
}
impl MatmulOutput {
    pub fn shape(&self) -> MatrixShape {
        match self {
            Self::F16(v) => v.shape(),
            Self::F32(v) => v.shape(),
        }
    }
}

#[derive(Debug)]
pub enum OpError {
    Tensor(TensorError),
    InvalidShape(&'static str),
    BackendUnavailable(ExecutionTarget),
    UnsupportedPolicy(&'static str),
    Npu(MatmulError),
    Pool(MatmulPoolError),
}
impl fmt::Display for OpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tensor(e) => e.fmt(f),
            Self::InvalidShape(s) => write!(f, "invalid MatMul contract: {s}"),
            Self::BackendUnavailable(t) => {
                write!(f, "requested MatMul backend is unavailable: {t:?}")
            }
            Self::UnsupportedPolicy(s) => write!(f, "unsupported MatMul policy: {s}"),
            Self::Npu(e) => e.fmt(f),
            Self::Pool(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for OpError {}
impl From<TensorError> for OpError {
    fn from(v: TensorError) -> Self {
        Self::Tensor(v)
    }
}
impl From<MatmulError> for OpError {
    fn from(v: MatmulError) -> Self {
        Self::Npu(v)
    }
}
impl From<MatmulPoolError> for OpError {
    fn from(v: MatmulPoolError) -> Self {
        Self::Pool(v)
    }
}

fn validate_contract(spec: MatmulSpec, a: &Matrix<f16>, b: &Matrix<f16>) -> Result<(), OpError> {
    if spec.m == 0 || spec.k == 0 || spec.n == 0 {
        return Err(OpError::InvalidShape("M/K/N must be non-zero"));
    }
    if a.layout() != TensorLayout::RowMajor || b.layout() != TensorLayout::RowMajor {
        return Err(OpError::InvalidShape(
            "only row-major host tensors are supported",
        ));
    }
    if a.shape() != spec.lhs_shape() {
        return Err(OpError::InvalidShape("A must have shape [M,K]"));
    }
    if b.shape() != spec.rhs_shape() {
        return Err(OpError::InvalidShape(
            "B must have shape [N,K] and is logically transposed",
        ));
    }
    Ok(())
}

fn cpu_fp16(spec: MatmulSpec, a: &[f16], b: &[f16]) -> Vec<f16> {
    let mut out = vec![f16::ZERO; spec.m * spec.n];
    for row in 0..spec.m {
        for col in 0..spec.n {
            let mut acc = 0.0f32;
            for kk in 0..spec.k {
                acc += a[row * spec.k + kk].to_f32() * b[col * spec.k + kk].to_f32();
            }
            out[row * spec.n + col] = f16::from_f32(acc);
        }
    }
    out
}

fn cpu_fp32(spec: MatmulSpec, a: &[f16], b: &[f16]) -> Vec<f32> {
    let mut out = vec![0.0f32; spec.m * spec.n];
    for row in 0..spec.m {
        for col in 0..spec.n {
            let mut acc = 0.0f64;
            for kk in 0..spec.k {
                acc +=
                    (a[row * spec.k + kk].to_f32() as f64) * (b[col * spec.k + kk].to_f32() as f64);
            }
            out[row * spec.n + col] = acc as f32;
        }
    }
    out
}

pub fn execute_cpu(
    spec: MatmulSpec,
    a: &Matrix<f16>,
    b: &Matrix<f16>,
) -> Result<MatmulOutput, OpError> {
    validate_contract(spec, a, b)?;
    match spec.precision {
        MatmulPrecision::Fp16Fast => Ok(MatmulOutput::F16(Matrix::from_vec(
            spec.m,
            spec.n,
            cpu_fp16(spec, a.values(), b.values()),
        )?)),
        MatmulPrecision::Fp32Accurate => Ok(MatmulOutput::F32(Matrix::from_vec(
            spec.m,
            spec.n,
            cpu_fp32(spec, a.values(), b.values()),
        )?)),
    }
}

pub struct PreparedFp16Matmul<'d> {
    spec: MatmulSpec,
    weights: Fp16PrepackedWeights<'d>,
    compatible_m: bool,
}
impl PreparedFp16Matmul<'_> {
    pub const fn spec(&self) -> MatmulSpec {
        self.spec
    }
    pub fn weight_stats(&self) -> PrepackedWeightStats {
        self.weights.stats()
    }
    pub const fn is_m_compatible(&self) -> bool {
        self.compatible_m
    }
    pub const fn k(&self) -> usize {
        self.spec.k
    }
    pub const fn n(&self) -> usize {
        self.spec.n
    }
}

pub struct SingleNpuBackend<'d> {
    executor: Fp16MatmulExecutor<'d>,
}
impl<'d> SingleNpuBackend<'d> {
    pub fn new(device: &'d RocketDevice) -> Result<Self, OpError> {
        Ok(Self {
            executor: Fp16MatmulExecutor::new(device)?,
        })
    }
    pub fn prepare_fp16(
        &self,
        spec: MatmulSpec,
        b: &Matrix<f16>,
    ) -> Result<PreparedFp16Matmul<'d>, OpError> {
        if spec.precision != MatmulPrecision::Fp16Fast {
            return Err(OpError::UnsupportedPolicy(
                "prepared weights currently support FP16-fast only",
            ));
        }
        if spec.target != ExecutionTarget::NpuSingle {
            return Err(OpError::UnsupportedPolicy(
                "prepared weights currently require NpuSingle target",
            ));
        }
        if !spec.npu_shape_supported() {
            return Err(OpError::UnsupportedPolicy(
                "prepared NPU weights require M%4==0, K%32==0, N%16==0",
            ));
        }
        if b.layout() != TensorLayout::RowMajor || b.shape() != spec.rhs_shape() {
            return Err(OpError::InvalidShape(
                "prepared B must have row-major shape [N,K]",
            ));
        }
        let weights = self
            .executor
            .prepack_weights(b.values(), spec.m, spec.k, spec.n)?;
        Ok(PreparedFp16Matmul {
            spec,
            weights,
            compatible_m: false,
        })
    }

    /// Prepare B[N,K] once with N/K tile geometry that is stable across M.
    /// K and N must already satisfy the NPU alignment contract; M is supplied
    /// later by each execution call.
    pub fn prepare_fp16_compatible_m(
        &self,
        b: &Matrix<f16>,
    ) -> Result<PreparedFp16Matmul<'d>, OpError> {
        if b.layout() != TensorLayout::RowMajor {
            return Err(OpError::InvalidShape(
                "M-compatible prepared B must be row-major",
            ));
        }
        let n = b.rows();
        let k = b.cols();
        if k == 0 || n == 0 || k % 32 != 0 || n % 16 != 0 {
            return Err(OpError::UnsupportedPolicy(
                "M-compatible prepared weights require K%32==0 and N%16==0",
            ));
        }
        let weights = self
            .executor
            .prepack_weights_compatible_m(b.values(), k, n)?;
        Ok(PreparedFp16Matmul {
            spec: MatmulSpec::new(
                4,
                k,
                n,
                MatmulPrecision::Fp16Fast,
                ExecutionTarget::NpuSingle,
            ),
            weights,
            compatible_m: true,
        })
    }

    pub fn execute_prepared_fp16_compatible_m(
        &mut self,
        prepared: &PreparedFp16Matmul<'d>,
        a: &Matrix<f16>,
    ) -> Result<Matrix<f16>, OpError> {
        if !prepared.compatible_m {
            return Err(OpError::UnsupportedPolicy(
                "prepared weights are fixed-M; use execute_prepared_fp16",
            ));
        }
        if a.layout() != TensorLayout::RowMajor
            || a.cols() != prepared.spec.k
            || a.rows() == 0
            || a.rows() % 4 != 0
        {
            return Err(OpError::InvalidShape(
                "M-compatible prepared A requires row-major [M,K], M%4==0",
            ));
        }
        let out = self.executor.execute_prepacked_compatible_m(
            a.values(),
            a.rows(),
            &prepared.weights,
        )?;
        Ok(Matrix::from_vec(a.rows(), prepared.spec.n, out.values)?)
    }

    pub fn execute_prepared_fp16(
        &mut self,
        prepared: &PreparedFp16Matmul<'d>,
        a: &Matrix<f16>,
    ) -> Result<Matrix<f16>, OpError> {
        if prepared.compatible_m {
            return Err(OpError::UnsupportedPolicy(
                "prepared weights are M-compatible; use execute_prepared_fp16_compatible_m",
            ));
        }
        let spec = prepared.spec;
        if a.layout() != TensorLayout::RowMajor || a.shape() != spec.lhs_shape() {
            return Err(OpError::InvalidShape(
                "prepared A must have row-major shape [M,K]",
            ));
        }
        let out = self
            .executor
            .execute_prepacked(a.values(), &prepared.weights)?;
        Ok(Matrix::from_vec(spec.m, spec.n, out.values)?)
    }

    pub fn execute(
        &mut self,
        spec: MatmulSpec,
        a: &Matrix<f16>,
        b: &Matrix<f16>,
    ) -> Result<MatmulOutput, OpError> {
        validate_contract(spec, a, b)?;
        if !spec.npu_shape_supported() {
            return Err(OpError::UnsupportedPolicy(
                "NPU requires M%4==0, K%32==0, N%16==0",
            ));
        }
        match spec.precision {
            MatmulPrecision::Fp16Fast => {
                let out = self
                    .executor
                    .execute(a.values(), b.values(), spec.m, spec.k, spec.n)?;
                Ok(MatmulOutput::F16(Matrix::from_vec(
                    spec.m, spec.n, out.values,
                )?))
            }
            MatmulPrecision::Fp32Accurate => {
                let out =
                    self.executor
                        .execute_f32(a.values(), b.values(), spec.m, spec.k, spec.n)?;
                Ok(MatmulOutput::F32(Matrix::from_vec(
                    spec.m, spec.n, out.values,
                )?))
            }
        }
    }
    pub fn executor_mut(&mut self) -> &mut Fp16MatmulExecutor<'d> {
        &mut self.executor
    }
}

pub struct PreparedPoolFp16Matmul {
    weights: Fp16MatmulPoolPreparedWeights,
}
impl PreparedPoolFp16Matmul {
    pub const fn k(&self) -> usize {
        self.weights.k()
    }
    pub const fn n(&self) -> usize {
        self.weights.n()
    }
    pub const fn weight_stats(&self) -> PoolPreparedWeightStats {
        self.weights.stats()
    }
}

pub struct PoolNpuBackend {
    pool: Fp16MatmulPool,
}
impl PoolNpuBackend {
    pub fn new(workers: usize) -> Result<Self, OpError> {
        Ok(Self {
            pool: Fp16MatmulPool::new(workers)?,
        })
    }
    pub fn workers(&self) -> usize {
        self.pool.workers()
    }
    pub fn prepare_fp16_compatible_m(
        &mut self,
        b: &Matrix<f16>,
    ) -> Result<PreparedPoolFp16Matmul, OpError> {
        if b.layout() != TensorLayout::RowMajor {
            return Err(OpError::InvalidShape("prepared pool B must be row-major"));
        }
        let n = b.rows();
        let k = b.cols();
        if k == 0 || n == 0 || k % 32 != 0 || n % 16 != 0 {
            return Err(OpError::UnsupportedPolicy(
                "prepared pool weights require K%32==0 and N%16==0",
            ));
        }
        let bv: Arc<[f16]> = Arc::from(b.values());
        let weights = self.pool.prepare_weights_compatible_m(bv, k, n)?;
        Ok(PreparedPoolFp16Matmul { weights })
    }

    pub fn execute_prepared_fp16_compatible_m(
        &mut self,
        prepared: &PreparedPoolFp16Matmul,
        a: &Matrix<f16>,
    ) -> Result<Matrix<f16>, OpError> {
        if a.layout() != TensorLayout::RowMajor
            || a.cols() != prepared.k()
            || a.rows() == 0
            || a.rows() % 4 != 0
        {
            return Err(OpError::InvalidShape(
                "prepared pool A requires row-major [M,K], M%4==0",
            ));
        }
        let av: Arc<[f16]> = Arc::from(a.values());
        let out = self
            .pool
            .execute_prepared(av, a.rows(), &prepared.weights)?;
        Ok(Matrix::from_vec(a.rows(), prepared.n(), out.values)?)
    }

    pub fn release_prepared_fp16(
        &mut self,
        prepared: &PreparedPoolFp16Matmul,
    ) -> Result<(), OpError> {
        self.pool.release_prepared(&prepared.weights)?;
        Ok(())
    }

    pub fn effective_workers_for_n(&self, n: usize, requested: usize) -> Result<usize, OpError> {
        Ok(self.pool.effective_workers_for_n(n, requested)?)
    }

    pub fn execute_with_workers(
        &mut self,
        spec: MatmulSpec,
        a: &Matrix<f16>,
        b: &Matrix<f16>,
        workers: usize,
    ) -> Result<MatmulOutput, OpError> {
        validate_contract(spec, a, b)?;
        if !spec.npu_shape_supported() {
            return Err(OpError::UnsupportedPolicy(
                "NPU pool requires M%4==0, K%32==0, N%16==0",
            ));
        }
        let av: Arc<[f16]> = Arc::from(a.values());
        let bv: Arc<[f16]> = Arc::from(b.values());
        match spec.precision {
            MatmulPrecision::Fp16Fast => {
                let out = self
                    .pool
                    .execute_with_workers(av, bv, spec.m, spec.k, spec.n, workers)?;
                Ok(MatmulOutput::F16(Matrix::from_vec(
                    spec.m, spec.n, out.values,
                )?))
            }
            MatmulPrecision::Fp32Accurate => {
                let out = self
                    .pool
                    .execute_f32_with_workers(av, bv, spec.m, spec.k, spec.n, workers)?;
                Ok(MatmulOutput::F32(Matrix::from_vec(
                    spec.m, spec.n, out.values,
                )?))
            }
        }
    }

    pub fn execute(
        &mut self,
        spec: MatmulSpec,
        a: &Matrix<f16>,
        b: &Matrix<f16>,
    ) -> Result<MatmulOutput, OpError> {
        self.execute_with_workers(spec, a, b, self.pool.workers())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct AutoTuneKey {
    m: usize,
    k: usize,
    n: usize,
    precision: MatmulPrecision,
}

/// Adaptive NPU policy for pure MatMul. The first aligned occurrence of a
/// shape/precision warms direct-single and every distinct pool worker count,
/// measures three interleaved samples per candidate, caches the fastest backend,
/// and reuses that decision until the cache is cleared. Unaligned Auto remains
/// CPU fallback. Candidate `1` means direct `SingleNpuBackend`; values >1 mean
/// that many requested pool workers (deduplicated by effective N slices).
pub struct AutoTunedNpuBackend<'d> {
    single: SingleNpuBackend<'d>,
    pool: PoolNpuBackend,
    cache: HashMap<AutoTuneKey, usize>,
}

impl<'d> AutoTunedNpuBackend<'d> {
    pub fn new(device: &'d RocketDevice, max_workers: usize) -> Result<Self, OpError> {
        Ok(Self {
            single: SingleNpuBackend::new(device)?,
            pool: PoolNpuBackend::new(max_workers)?,
            cache: HashMap::new(),
        })
    }

    pub fn cached_workers(&self, spec: MatmulSpec) -> Option<usize> {
        self.cache
            .get(&AutoTuneKey {
                m: spec.m,
                k: spec.k,
                n: spec.n,
                precision: spec.precision,
            })
            .copied()
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    pub fn cached_shapes(&self) -> usize {
        self.cache.len()
    }

    fn run_candidate(
        &mut self,
        workers: usize,
        spec: MatmulSpec,
        a: &Matrix<f16>,
        b: &Matrix<f16>,
    ) -> Result<MatmulOutput, OpError> {
        if workers == 1 {
            let mut s = spec;
            s.target = ExecutionTarget::NpuSingle;
            self.single.execute(s, a, b)
        } else {
            let mut s = spec;
            s.target = ExecutionTarget::NpuPool;
            self.pool.execute_with_workers(s, a, b, workers)
        }
    }

    fn candidates(&self, n: usize) -> Result<Vec<usize>, OpError> {
        let mut out = vec![1usize];
        let mut last_effective = 1usize;
        for requested in 2..=self.pool.workers() {
            let effective = self.pool.effective_workers_for_n(n, requested)?;
            if effective > last_effective {
                out.push(requested);
                last_effective = effective;
            }
        }
        Ok(out)
    }

    fn tune(
        &mut self,
        spec: MatmulSpec,
        a: &Matrix<f16>,
        b: &Matrix<f16>,
    ) -> Result<usize, OpError> {
        let candidates = self.candidates(spec.n)?;
        if candidates.len() == 1 {
            return Ok(1);
        }

        // Warm each candidate once so one-time scratch allocation/growth is not timed.
        for &workers in &candidates {
            let _ = self.run_candidate(workers, spec, a, b)?;
        }

        let mut samples: Vec<Vec<u128>> =
            candidates.iter().map(|_| Vec::with_capacity(3)).collect();
        for round in 0..3 {
            if round % 2 == 0 {
                for (i, &workers) in candidates.iter().enumerate() {
                    let start = Instant::now();
                    let _ = self.run_candidate(workers, spec, a, b)?;
                    samples[i].push(start.elapsed().as_nanos());
                }
            } else {
                for (i, &workers) in candidates.iter().enumerate().rev() {
                    let start = Instant::now();
                    let _ = self.run_candidate(workers, spec, a, b)?;
                    samples[i].push(start.elapsed().as_nanos());
                }
            }
        }

        let mut best_index = 0usize;
        let mut best_median = u128::MAX;
        for (i, values) in samples.iter_mut().enumerate() {
            values.sort_unstable();
            let median = values[values.len() / 2];
            if median < best_median {
                best_median = median;
                best_index = i;
            }
        }
        Ok(candidates[best_index])
    }

    pub fn execute(
        &mut self,
        spec: MatmulSpec,
        a: &Matrix<f16>,
        b: &Matrix<f16>,
    ) -> Result<MatmulOutput, OpError> {
        validate_contract(spec, a, b)?;
        match spec.target {
            ExecutionTarget::Cpu => execute_cpu(spec, a, b),
            ExecutionTarget::NpuSingle => self.single.execute(spec, a, b),
            ExecutionTarget::NpuPool => self.pool.execute(spec, a, b),
            ExecutionTarget::Auto => {
                if !spec.npu_shape_supported() {
                    return execute_cpu(spec, a, b);
                }
                let key = AutoTuneKey {
                    m: spec.m,
                    k: spec.k,
                    n: spec.n,
                    precision: spec.precision,
                };
                let workers = match self.cache.get(&key).copied() {
                    Some(v) => v,
                    None => {
                        let v = self.tune(spec, a, b)?;
                        self.cache.insert(key, v);
                        v
                    }
                };
                self.run_candidate(workers, spec, a, b)
            }
        }
    }
}

/// Correctness-first dispatcher. `Auto` uses the supplied single-NPU backend only
/// for a currently validated NPU shape; otherwise it falls back to CPU. It never pads.
pub fn execute_auto(
    spec: MatmulSpec,
    a: &Matrix<f16>,
    b: &Matrix<f16>,
    mut single: Option<&mut SingleNpuBackend<'_>>,
    mut pool: Option<&mut PoolNpuBackend>,
) -> Result<MatmulOutput, OpError> {
    validate_contract(spec, a, b)?;
    match spec.target {
        ExecutionTarget::Cpu => execute_cpu(spec, a, b),
        ExecutionTarget::NpuSingle => single
            .as_deref_mut()
            .ok_or(OpError::BackendUnavailable(ExecutionTarget::NpuSingle))?
            .execute(spec, a, b),
        ExecutionTarget::NpuPool => pool
            .as_deref_mut()
            .ok_or(OpError::BackendUnavailable(ExecutionTarget::NpuPool))?
            .execute(spec, a, b),
        ExecutionTarget::Auto => {
            if spec.npu_shape_supported() {
                if let Some(backend) = single.as_deref_mut() {
                    return backend.execute(spec, a, b);
                }
            }
            execute_cpu(spec, a, b)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn data(m: usize, k: usize, n: usize) -> (Matrix<f16>, Matrix<f16>) {
        let a = (0..m * k)
            .map(|i| f16::from_f32(((i * 7 + 3) % 5) as f32 - 2.0))
            .collect();
        let b = (0..n * k)
            .map(|i| f16::from_f32(((i * 3 + 1) % 5) as f32 - 2.0))
            .collect();
        (
            Matrix::from_vec(m, k, a).unwrap(),
            Matrix::from_vec(n, k, b).unwrap(),
        )
    }
    #[test]
    fn cpu_contract_fp16_and_fp32_have_declared_output_shape() {
        let (a, b) = data(3, 7, 5);
        for precision in [MatmulPrecision::Fp16Fast, MatmulPrecision::Fp32Accurate] {
            let out = execute_cpu(
                MatmulSpec::new(3, 7, 5, precision, ExecutionTarget::Cpu),
                &a,
                &b,
            )
            .unwrap();
            assert_eq!(out.shape(), MatrixShape::new(3, 5));
        }
    }
    #[test]
    fn contract_rejects_wrong_rhs_shape() {
        let (a, _) = data(4, 32, 16);
        let bad = Matrix::from_vec(32, 16, vec![f16::ZERO; 32 * 16]).unwrap();
        let e = execute_cpu(
            MatmulSpec::new(4, 32, 16, MatmulPrecision::Fp16Fast, ExecutionTarget::Cpu),
            &a,
            &bad,
        )
        .unwrap_err();
        assert!(matches!(e, OpError::InvalidShape(_)));
    }
    #[test]
    fn auto_falls_back_to_cpu_for_unaligned_shape() {
        let (a, b) = data(3, 7, 5);
        let out = execute_auto(
            MatmulSpec::new(3, 7, 5, MatmulPrecision::Fp16Fast, ExecutionTarget::Auto),
            &a,
            &b,
            None,
            None,
        )
        .unwrap();
        assert!(matches!(out, MatmulOutput::F16(_)));
    }
    #[test]
    fn prepared_spec_requires_aligned_single_fp16_contract() {
        let spec = MatmulSpec::new(
            4,
            32,
            16,
            MatmulPrecision::Fp16Fast,
            ExecutionTarget::NpuSingle,
        );
        assert!(spec.npu_shape_supported());
        assert_eq!(spec.lhs_shape(), MatrixShape::new(4, 32));
        assert_eq!(spec.rhs_shape(), MatrixShape::new(16, 32));
    }

    #[test]
    fn explicit_npu_without_backend_is_an_error() {
        let (a, b) = data(4, 32, 16);
        let e = execute_auto(
            MatmulSpec::new(
                4,
                32,
                16,
                MatmulPrecision::Fp16Fast,
                ExecutionTarget::NpuSingle,
            ),
            &a,
            &b,
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(
            e,
            OpError::BackendUnavailable(ExecutionTarget::NpuSingle)
        ));
    }
}
