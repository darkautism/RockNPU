use super::{Node, OnnxError, TinyOnnxModel, TraceTensor};
use half::f16;
use rocknpu_ops::{
    ExecutionTarget, MatmulPrecision, MatmulSpec, PreparedFp16Matmul, SingleNpuBackend,
};
use rocknpu_tensor::{Matrix, MatrixShape};
use std::collections::HashMap;
use std::time::Instant;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PreparedModelStats {
    pub dense_nodes: usize,
    pub padded_dense_nodes: usize,
    pub m_compatible_dense_nodes: usize,
    pub resident_weight_bytes: usize,
    pub resident_weight_tiles: usize,
    pub weight_pack_ns: u128,
    pub prepare_total_ns: u128,
}

struct PreparedDense<'d> {
    fixed_m: Option<usize>,
    logical_k: usize,
    logical_n: usize,
    padded_k: usize,
    padded_n: usize,
    matmul: PreparedFp16Matmul<'d>,
}

impl PreparedDense<'_> {
    fn is_padded_for_m(&self, logical_m: usize) -> bool {
        logical_m != align_up_infallible(logical_m, 4)
            || self.logical_k != self.padded_k
            || self.logical_n != self.padded_n
    }
}

enum PreparedNode<'d> {
    MatMul {
        lhs: String,
        output: String,
        dense: PreparedDense<'d>,
    },
    Gemm {
        lhs: String,
        output: String,
        bias: Vec<f16>,
        dense: PreparedDense<'d>,
    },
    Add {
        input: String,
        output: String,
        bias: Vec<f16>,
    },
    Relu {
        input: String,
        output: String,
    },
}

pub struct PreparedNpuModel<'d> {
    input_name: String,
    output_name: String,
    input_shape_hint: MatrixShape,
    dynamic_batch: bool,
    nodes: Vec<PreparedNode<'d>>,
    stats: PreparedModelStats,
}

impl PreparedNpuModel<'_> {
    /// Shape used while preparing graph metadata. For dynamic-batch sessions the
    /// row count is only a hint; `run_fp16` accepts any positive row count.
    pub const fn input_shape(&self) -> MatrixShape {
        self.input_shape_hint
    }

    pub const fn is_dynamic_batch(&self) -> bool {
        self.dynamic_batch
    }

    pub const fn stats(&self) -> PreparedModelStats {
        self.stats
    }
}

impl<'d> PreparedNpuModel<'d> {
    pub fn run_fp16(
        &self,
        input: &Matrix<f16>,
        backend: &mut SingleNpuBackend<'d>,
    ) -> Result<(Matrix<f16>, super::RunStats), OnnxError> {
        let (out, stats, _) = self.run_impl(input, backend, false)?;
        Ok((out, stats))
    }

    pub fn run_fp16_traced(
        &self,
        input: &Matrix<f16>,
        backend: &mut SingleNpuBackend<'d>,
    ) -> Result<(Matrix<f16>, super::RunStats, Vec<TraceTensor>), OnnxError> {
        self.run_impl(input, backend, true)
    }

    fn validate_input(&self, input: &Matrix<f16>) -> Result<(), OnnxError> {
        if self.dynamic_batch {
            if input.rows() == 0 || input.cols() != self.input_shape_hint.cols {
                return Err(OnnxError::InvalidModel(format!(
                    "dynamic prepared model input must be Mx{} with M>0, got {}x{}",
                    self.input_shape_hint.cols,
                    input.rows(),
                    input.cols()
                )));
            }
        } else if input.shape() != self.input_shape_hint {
            return Err(OnnxError::InvalidModel(format!(
                "prepared model input shape must be {}x{}, got {}x{}",
                self.input_shape_hint.rows,
                self.input_shape_hint.cols,
                input.rows(),
                input.cols()
            )));
        }
        Ok(())
    }

    fn run_impl(
        &self,
        input: &Matrix<f16>,
        backend: &mut SingleNpuBackend<'d>,
        trace_enabled: bool,
    ) -> Result<(Matrix<f16>, super::RunStats, Vec<TraceTensor>), OnnxError> {
        self.validate_input(input)?;
        let mut values: HashMap<String, Matrix<f16>> = HashMap::new();
        values.insert(self.input_name.clone(), input.clone());
        let mut stats = super::RunStats::default();
        let mut trace = Vec::new();

        for node in &self.nodes {
            match node {
                PreparedNode::MatMul { lhs, output, dense } => {
                    let a = values
                        .get(lhs)
                        .ok_or_else(|| OnnxError::InvalidModel(format!("missing lhs {lhs}")))?;
                    let m = execute_prepared_dense(a, dense, backend)?;
                    if trace_enabled {
                        trace.push(trace_tensor("MatMul", output, &m));
                    }
                    let logical_m = a.rows();
                    values.insert(output.clone(), m);
                    stats.matmul_nodes += 1;
                    stats.npu_dense_nodes += 1;
                    if dense.is_padded_for_m(logical_m) {
                        stats.padded_npu_dense_nodes += 1;
                    }
                }
                PreparedNode::Gemm {
                    lhs,
                    output,
                    bias,
                    dense,
                } => {
                    let a = values
                        .get(lhs)
                        .ok_or_else(|| OnnxError::InvalidModel(format!("missing lhs {lhs}")))?;
                    let logical_m = a.rows();
                    let dense_out = execute_prepared_dense(a, dense, backend)?;
                    if bias.len() != dense_out.cols() {
                        return Err(OnnxError::InvalidModel("prepared Gemm bias shape".into()));
                    }
                    let mut v = dense_out.values().to_vec();
                    for r in 0..dense_out.rows() {
                        for c in 0..dense_out.cols() {
                            let i = r * dense_out.cols() + c;
                            v[i] = f16::from_f32(v[i].to_f32() + bias[c].to_f32());
                        }
                    }
                    let m = Matrix::from_vec(dense_out.rows(), dense_out.cols(), v)?;
                    if trace_enabled {
                        trace.push(trace_tensor("Gemm", output, &m));
                    }
                    values.insert(output.clone(), m);
                    stats.gemm_nodes += 1;
                    stats.npu_dense_nodes += 1;
                    if dense.is_padded_for_m(logical_m) {
                        stats.padded_npu_dense_nodes += 1;
                    }
                }
                PreparedNode::Add {
                    input,
                    output,
                    bias,
                } => {
                    let a = values.get(input).ok_or_else(|| {
                        OnnxError::InvalidModel(format!("missing Add input {input}"))
                    })?;
                    if bias.len() != a.cols() {
                        return Err(OnnxError::InvalidModel("prepared Add bias shape".into()));
                    }
                    let mut v = a.values().to_vec();
                    for r in 0..a.rows() {
                        for c in 0..a.cols() {
                            let i = r * a.cols() + c;
                            v[i] = f16::from_f32(v[i].to_f32() + bias[c].to_f32());
                        }
                    }
                    let m = Matrix::from_vec(a.rows(), a.cols(), v)?;
                    if trace_enabled {
                        trace.push(trace_tensor("Add", output, &m));
                    }
                    values.insert(output.clone(), m);
                    stats.add_nodes += 1;
                }
                PreparedNode::Relu { input, output } => {
                    let a = values.get(input).ok_or_else(|| {
                        OnnxError::InvalidModel(format!("missing Relu input {input}"))
                    })?;
                    let v = a
                        .values()
                        .iter()
                        .map(|x| if x.to_f32() < 0.0 { f16::ZERO } else { *x })
                        .collect();
                    let m = Matrix::from_vec(a.rows(), a.cols(), v)?;
                    if trace_enabled {
                        trace.push(trace_tensor("Relu", output, &m));
                    }
                    values.insert(output.clone(), m);
                    stats.relu_nodes += 1;
                }
            }
        }

        let out = values
            .remove(&self.output_name)
            .ok_or_else(|| OnnxError::InvalidModel("prepared graph output not produced".into()))?;
        Ok((out, stats, trace))
    }
}

impl TinyOnnxModel {
    /// Prepare an optimized fixed-batch session. Weight tiling may depend on M,
    /// so this path keeps the best currently validated exact planner geometry.
    pub fn prepare_npu<'d>(
        &self,
        input_shape: MatrixShape,
        backend: &SingleNpuBackend<'d>,
    ) -> Result<PreparedNpuModel<'d>, OnnxError> {
        self.prepare_npu_impl(input_shape, backend, false)
    }

    /// Prepare a session whose resident weights are reusable across batch/M.
    /// `input_shape.rows` is only a preparation/statistics hint; execution may use
    /// any positive M with the same input feature width.
    pub fn prepare_npu_dynamic_batch<'d>(
        &self,
        input_shape: MatrixShape,
        backend: &SingleNpuBackend<'d>,
    ) -> Result<PreparedNpuModel<'d>, OnnxError> {
        self.prepare_npu_impl(input_shape, backend, true)
    }

    fn prepare_npu_impl<'d>(
        &self,
        input_shape: MatrixShape,
        backend: &SingleNpuBackend<'d>,
        dynamic_batch: bool,
    ) -> Result<PreparedNpuModel<'d>, OnnxError> {
        if input_shape.rows == 0 || input_shape.cols == 0 {
            return Err(OnnxError::InvalidModel(
                "prepared input shape must be nonzero".into(),
            ));
        }
        let start = Instant::now();
        let mut shapes: HashMap<String, MatrixShape> = HashMap::new();
        shapes.insert(self.input_name.clone(), input_shape);
        let mut prepared_nodes = Vec::with_capacity(self.nodes.len());
        let mut stats = PreparedModelStats::default();

        for node in &self.nodes {
            match node {
                Node::MatMul { lhs, rhs, output } => {
                    let a_shape = *shapes.get(lhs).ok_or_else(|| {
                        OnnxError::InvalidModel(format!("missing lhs shape {lhs}"))
                    })?;
                    let w = self.initializers.get(rhs).ok_or_else(|| {
                        OnnxError::Unsupported(format!("MatMul rhs {rhs} must be initializer"))
                    })?;
                    if w.dims.len() != 2 || w.dims[0] != a_shape.cols {
                        return Err(OnnxError::InvalidModel(
                            "prepared MatMul weight shape".into(),
                        ));
                    }
                    let k = w.dims[0];
                    let n = w.dims[1];
                    let mut bt = vec![f16::ZERO; n * k];
                    for kk in 0..k {
                        for nn in 0..n {
                            bt[nn * k + kk] = w.values[kk * n + nn];
                        }
                    }
                    let b = Matrix::from_vec(n, k, bt)?;
                    let dense = if dynamic_batch {
                        prepare_dense_dynamic(&b, backend)?
                    } else {
                        prepare_dense_fixed(a_shape.rows, &b, backend)?
                    };
                    account_dense(&mut stats, &dense, a_shape.rows);
                    shapes.insert(output.clone(), MatrixShape::new(a_shape.rows, n));
                    prepared_nodes.push(PreparedNode::MatMul {
                        lhs: lhs.clone(),
                        output: output.clone(),
                        dense,
                    });
                }
                Node::Gemm {
                    lhs,
                    rhs,
                    bias,
                    output,
                } => {
                    let a_shape = *shapes.get(lhs).ok_or_else(|| {
                        OnnxError::InvalidModel(format!("missing lhs shape {lhs}"))
                    })?;
                    let w = self.initializers.get(rhs).ok_or_else(|| {
                        OnnxError::Unsupported(format!("Gemm rhs {rhs} must be initializer"))
                    })?;
                    let b = self.initializers.get(bias).ok_or_else(|| {
                        OnnxError::Unsupported("Gemm bias must be initializer".into())
                    })?;
                    if w.dims.len() != 2 || w.dims[1] != a_shape.cols {
                        return Err(OnnxError::InvalidModel("prepared Gemm weight shape".into()));
                    }
                    let n = w.dims[0];
                    let k = w.dims[1];
                    if b.dims.len() != 1 || b.dims[0] != n {
                        return Err(OnnxError::InvalidModel("prepared Gemm bias shape".into()));
                    }
                    let weights = Matrix::from_vec(n, k, w.values.clone())?;
                    let dense = if dynamic_batch {
                        prepare_dense_dynamic(&weights, backend)?
                    } else {
                        prepare_dense_fixed(a_shape.rows, &weights, backend)?
                    };
                    account_dense(&mut stats, &dense, a_shape.rows);
                    shapes.insert(output.clone(), MatrixShape::new(a_shape.rows, n));
                    prepared_nodes.push(PreparedNode::Gemm {
                        lhs: lhs.clone(),
                        output: output.clone(),
                        bias: b.values.clone(),
                        dense,
                    });
                }
                Node::Add { lhs, rhs, output } => {
                    let (input_name, bias_name) =
                        if shapes.contains_key(lhs) && self.initializers.contains_key(rhs) {
                            (lhs, rhs)
                        } else if shapes.contains_key(rhs) && self.initializers.contains_key(lhs) {
                            (rhs, lhs)
                        } else {
                            return Err(OnnxError::Unsupported(
                                "prepared Add supports matrix+bias initializer".into(),
                            ));
                        };
                    let shape = *shapes.get(input_name).unwrap();
                    let bias = self.initializers.get(bias_name).unwrap();
                    if bias.dims.len() != 1 || bias.dims[0] != shape.cols {
                        return Err(OnnxError::InvalidModel("prepared Add bias shape".into()));
                    }
                    shapes.insert(output.clone(), shape);
                    prepared_nodes.push(PreparedNode::Add {
                        input: input_name.clone(),
                        output: output.clone(),
                        bias: bias.values.clone(),
                    });
                }
                Node::Relu { input, output } => {
                    let shape = *shapes.get(input).ok_or_else(|| {
                        OnnxError::InvalidModel(format!("missing Relu shape {input}"))
                    })?;
                    shapes.insert(output.clone(), shape);
                    prepared_nodes.push(PreparedNode::Relu {
                        input: input.clone(),
                        output: output.clone(),
                    });
                }
            }
        }

        if !shapes.contains_key(&self.output_name) {
            return Err(OnnxError::InvalidModel(
                "prepared graph output shape missing".into(),
            ));
        }
        stats.prepare_total_ns = start.elapsed().as_nanos();
        Ok(PreparedNpuModel {
            input_name: self.input_name.clone(),
            output_name: self.output_name.clone(),
            input_shape_hint: input_shape,
            dynamic_batch,
            nodes: prepared_nodes,
            stats,
        })
    }
}

fn padded_weights(b: &Matrix<f16>) -> Result<(Matrix<f16>, usize, usize), OnnxError> {
    let k = b.cols();
    let n = b.rows();
    let kp = align_up(k, 32)?;
    let np = align_up(n, 16)?;
    let mut bv = vec![f16::ZERO; np * kp];
    for r in 0..n {
        bv[r * kp..r * kp + k].copy_from_slice(&b.values()[r * k..(r + 1) * k]);
    }
    Ok((Matrix::from_vec(np, kp, bv)?, kp, np))
}

fn prepare_dense_fixed<'d>(
    m: usize,
    b: &Matrix<f16>,
    backend: &SingleNpuBackend<'d>,
) -> Result<PreparedDense<'d>, OnnxError> {
    let logical_k = b.cols();
    let logical_n = b.rows();
    let mp = align_up(m, 4)?;
    let (bp, padded_k, padded_n) = padded_weights(b)?;
    let spec = MatmulSpec::new(
        mp,
        padded_k,
        padded_n,
        MatmulPrecision::Fp16Fast,
        ExecutionTarget::NpuSingle,
    );
    let matmul = backend.prepare_fp16(spec, &bp)?;
    Ok(PreparedDense {
        fixed_m: Some(m),
        logical_k,
        logical_n,
        padded_k,
        padded_n,
        matmul,
    })
}

fn prepare_dense_dynamic<'d>(
    b: &Matrix<f16>,
    backend: &SingleNpuBackend<'d>,
) -> Result<PreparedDense<'d>, OnnxError> {
    let logical_k = b.cols();
    let logical_n = b.rows();
    let (bp, padded_k, padded_n) = padded_weights(b)?;
    let matmul = backend.prepare_fp16_compatible_m(&bp)?;
    Ok(PreparedDense {
        fixed_m: None,
        logical_k,
        logical_n,
        padded_k,
        padded_n,
        matmul,
    })
}

fn execute_prepared_dense<'d>(
    a: &Matrix<f16>,
    dense: &PreparedDense<'d>,
    backend: &mut SingleNpuBackend<'d>,
) -> Result<Matrix<f16>, OnnxError> {
    if a.cols() != dense.logical_k {
        return Err(OnnxError::InvalidModel(
            "prepared dense activation K mismatch".into(),
        ));
    }
    if let Some(fixed_m) = dense.fixed_m {
        if a.rows() != fixed_m {
            return Err(OnnxError::InvalidModel(
                "fixed prepared dense activation M mismatch".into(),
            ));
        }
    }
    let logical_m = a.rows();
    let padded_m = align_up(logical_m, 4)?;
    let mut av = vec![f16::ZERO; padded_m * dense.padded_k];
    for r in 0..logical_m {
        av[r * dense.padded_k..r * dense.padded_k + dense.logical_k]
            .copy_from_slice(&a.values()[r * dense.logical_k..(r + 1) * dense.logical_k]);
    }
    let ap = Matrix::from_vec(padded_m, dense.padded_k, av)?;
    let padded = if dense.fixed_m.is_some() {
        backend.execute_prepared_fp16(&dense.matmul, &ap)?
    } else {
        backend.execute_prepared_fp16_compatible_m(&dense.matmul, &ap)?
    };
    let mut cropped = vec![f16::ZERO; logical_m * dense.logical_n];
    for r in 0..logical_m {
        cropped[r * dense.logical_n..(r + 1) * dense.logical_n].copy_from_slice(
            &padded.values()[r * dense.padded_n..r * dense.padded_n + dense.logical_n],
        );
    }
    Ok(Matrix::from_vec(logical_m, dense.logical_n, cropped)?)
}

fn account_dense(stats: &mut PreparedModelStats, dense: &PreparedDense<'_>, m_hint: usize) {
    let ws = dense.matmul.weight_stats();
    stats.dense_nodes += 1;
    if dense.is_padded_for_m(m_hint) {
        stats.padded_dense_nodes += 1;
    }
    if dense.matmul.is_m_compatible() {
        stats.m_compatible_dense_nodes += 1;
    }
    stats.resident_weight_bytes += ws.resident_bytes;
    stats.resident_weight_tiles += ws.unique_tiles;
    stats.weight_pack_ns += ws.pack_ns;
}

fn trace_tensor(op: &'static str, name: &str, m: &Matrix<f16>) -> TraceTensor {
    TraceTensor {
        op,
        name: name.to_string(),
        rows: m.rows(),
        cols: m.cols(),
        values: m.values().to_vec(),
    }
}

fn align_up(value: usize, alignment: usize) -> Result<usize, OnnxError> {
    let add = alignment
        .checked_sub(1)
        .ok_or_else(|| OnnxError::InvalidModel("zero alignment".into()))?;
    value
        .checked_add(add)
        .map(|v| (v / alignment) * alignment)
        .ok_or_else(|| OnnxError::InvalidModel("alignment overflow".into()))
}

fn align_up_infallible(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}
