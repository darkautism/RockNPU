use half::f16;
use onnx_protobuf::{AttributeValue, Message, ModelProto, TensorProto, tensor_proto};
use rocknpu_ops::{
    ExecutionTarget, MatmulOutput, MatmulPrecision, MatmulSpec, OpError, SingleNpuBackend,
    execute_auto,
};
use rocknpu_tensor::{Matrix, TensorError};
use std::collections::{HashMap, HashSet};
use std::fmt;

mod cnn;
mod prepared;
mod prepared_pool;
pub use cnn::{
    CnnOnnxModel, CnnPreparedConvState, CnnPreparedConvStats, CnnPreparedDenseStats, CnnRunStats,
    CnnTimingStats, TensorTrace,
};
pub use prepared::{PreparedModelStats, PreparedNpuModel};
pub use prepared_pool::{PreparedPoolModelStats, PreparedPoolNpuModel};
pub use rocknpu_ir::F16Tensor as TensorF16;

#[derive(Debug)]
pub enum OnnxError {
    Protobuf(String),
    MissingGraph,
    MissingInput,
    MissingOutput,
    InvalidModel(String),
    Unsupported(String),
    Tensor(TensorError),
    Op(OpError),
    Conv(rocknpu_conv::ConvError),
    Ir(rocknpu_ir::IrError),
}
impl fmt::Display for OnnxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Protobuf(s) => write!(f, "ONNX protobuf error: {s}"),
            Self::MissingGraph => write!(f, "ONNX model has no graph"),
            Self::MissingInput => write!(f, "ONNX graph has no non-initializer input"),
            Self::MissingOutput => write!(f, "ONNX graph has no output"),
            Self::InvalidModel(s) => write!(f, "invalid ONNX model: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported ONNX feature: {s}"),
            Self::Tensor(e) => e.fmt(f),
            Self::Op(e) => e.fmt(f),
            Self::Conv(e) => e.fmt(f),
            Self::Ir(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for OnnxError {}
impl From<TensorError> for OnnxError {
    fn from(v: TensorError) -> Self {
        Self::Tensor(v)
    }
}
impl From<OpError> for OnnxError {
    fn from(v: OpError) -> Self {
        Self::Op(v)
    }
}
impl From<rocknpu_conv::ConvError> for OnnxError {
    fn from(v: rocknpu_conv::ConvError) -> Self {
        Self::Conv(v)
    }
}
impl From<rocknpu_ir::IrError> for OnnxError {
    fn from(v: rocknpu_ir::IrError) -> Self {
        Self::Ir(v)
    }
}

/// Import the currently supported ONNX subset into RockNPU's frontend-neutral IR.
/// The returned graph owns all constants and does not borrow the source model bytes.
pub fn import_graph(bytes: &[u8]) -> Result<rocknpu_ir::Graph, OnnxError> {
    Ok(CnnOnnxModel::from_bytes(bytes)?.into_graph())
}
#[derive(Debug, Clone)]
pub(crate) struct ConstTensor {
    pub(crate) dims: Vec<usize>,
    pub(crate) values: Vec<f16>,
}
#[derive(Debug, Clone)]
pub(crate) enum Node {
    MatMul {
        lhs: String,
        rhs: String,
        output: String,
    },
    Gemm {
        lhs: String,
        rhs: String,
        bias: String,
        output: String,
    },
    Add {
        lhs: String,
        rhs: String,
        output: String,
    },
    Relu {
        input: String,
        output: String,
    },
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RunStats {
    pub matmul_nodes: usize,
    pub gemm_nodes: usize,
    pub add_nodes: usize,
    pub relu_nodes: usize,
    pub npu_dense_nodes: usize,
    pub padded_npu_dense_nodes: usize,
}
#[derive(Debug, Clone, PartialEq)]
pub struct TraceTensor {
    pub op: &'static str,
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f16>,
}
#[derive(Debug, Clone)]
pub struct TinyOnnxModel {
    pub(crate) input_name: String,
    pub(crate) output_name: String,
    pub(crate) nodes: Vec<Node>,
    pub(crate) initializers: HashMap<String, ConstTensor>,
}
impl TinyOnnxModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, OnnxError> {
        let model =
            ModelProto::parse_from_bytes(bytes).map_err(|e| OnnxError::Protobuf(e.to_string()))?;
        let graph = model.graph.as_ref().ok_or(OnnxError::MissingGraph)?;
        let mut initializers = HashMap::new();
        for t in &graph.initializer {
            if t.name.is_empty() {
                return Err(OnnxError::InvalidModel("initializer without name".into()));
            }
            initializers.insert(t.name.clone(), parse_initializer(t)?);
        }
        let names: HashSet<&str> = initializers.keys().map(String::as_str).collect();
        let input_name = graph
            .input
            .iter()
            .map(|v| v.name.as_str())
            .find(|n| !names.contains(*n))
            .ok_or(OnnxError::MissingInput)?
            .to_string();
        let output_name = graph
            .output
            .first()
            .ok_or(OnnxError::MissingOutput)?
            .name
            .clone();
        if output_name.is_empty() {
            return Err(OnnxError::MissingOutput);
        }
        let mut nodes = Vec::with_capacity(graph.node.len());
        for n in &graph.node {
            if !n.domain.is_empty() && n.domain != "ai.onnx" {
                return Err(OnnxError::Unsupported(format!("node domain {}", n.domain)));
            }
            match n.op_type.as_str() {
                "MatMul" => {
                    if n.input.len() != 2 || n.output.len() != 1 {
                        return Err(OnnxError::InvalidModel("MatMul arity".into()));
                    }
                    nodes.push(Node::MatMul {
                        lhs: n.input[0].clone(),
                        rhs: n.input[1].clone(),
                        output: n.output[0].clone(),
                    });
                }
                "Gemm" => {
                    if n.input.len() != 3 || n.output.len() != 1 {
                        return Err(OnnxError::InvalidModel("Gemm arity".into()));
                    }
                    let mut alpha = 1.0f32;
                    let mut beta = 1.0f32;
                    let mut trans_a = 0i64;
                    let mut trans_b = 0i64;
                    for a in &n.attribute {
                        match (a.name.as_str(), a.as_value()) {
                            ("alpha", AttributeValue::Float32(v)) => alpha = v,
                            ("beta", AttributeValue::Float32(v)) => beta = v,
                            ("transA", AttributeValue::Integer64(v)) => trans_a = v,
                            ("transB", AttributeValue::Integer64(v)) => trans_b = v,
                            (name, _) => {
                                return Err(OnnxError::Unsupported(format!(
                                    "Gemm attribute {name}"
                                )));
                            }
                        }
                    }
                    if alpha != 1.0 || beta != 1.0 || trans_a != 0 || trans_b != 1 {
                        return Err(OnnxError::Unsupported(format!(
                            "Gemm requires alpha=1 beta=1 transA=0 transB=1; got alpha={alpha} beta={beta} transA={trans_a} transB={trans_b}"
                        )));
                    }
                    nodes.push(Node::Gemm {
                        lhs: n.input[0].clone(),
                        rhs: n.input[1].clone(),
                        bias: n.input[2].clone(),
                        output: n.output[0].clone(),
                    });
                }
                "Add" => {
                    if n.input.len() != 2 || n.output.len() != 1 {
                        return Err(OnnxError::InvalidModel("Add arity".into()));
                    }
                    nodes.push(Node::Add {
                        lhs: n.input[0].clone(),
                        rhs: n.input[1].clone(),
                        output: n.output[0].clone(),
                    });
                }
                "Relu" => {
                    if n.input.len() != 1 || n.output.len() != 1 {
                        return Err(OnnxError::InvalidModel("Relu arity".into()));
                    }
                    nodes.push(Node::Relu {
                        input: n.input[0].clone(),
                        output: n.output[0].clone(),
                    });
                }
                other => return Err(OnnxError::Unsupported(format!("operator {other}"))),
            }
        }
        Ok(Self {
            input_name,
            output_name,
            nodes,
            initializers,
        })
    }
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    pub fn run_fp16(
        &self,
        input: &Matrix<f16>,
        target: ExecutionTarget,
        single: Option<&mut SingleNpuBackend<'_>>,
    ) -> Result<(Matrix<f16>, RunStats), OnnxError> {
        let (out, stats, _) = self.run_fp16_impl(input, target, single, false)?;
        Ok((out, stats))
    }
    pub fn run_fp16_traced(
        &self,
        input: &Matrix<f16>,
        target: ExecutionTarget,
        single: Option<&mut SingleNpuBackend<'_>>,
    ) -> Result<(Matrix<f16>, RunStats, Vec<TraceTensor>), OnnxError> {
        self.run_fp16_impl(input, target, single, true)
    }
    fn run_fp16_impl(
        &self,
        input: &Matrix<f16>,
        target: ExecutionTarget,
        mut single: Option<&mut SingleNpuBackend<'_>>,
        trace_enabled: bool,
    ) -> Result<(Matrix<f16>, RunStats, Vec<TraceTensor>), OnnxError> {
        let mut values: HashMap<String, Matrix<f16>> = HashMap::new();
        values.insert(self.input_name.clone(), input.clone());
        let mut stats = RunStats::default();
        let mut trace = Vec::new();
        for node in &self.nodes {
            match node {
                Node::MatMul { lhs, rhs, output } => {
                    let a = values
                        .get(lhs)
                        .ok_or_else(|| OnnxError::InvalidModel(format!("missing lhs {lhs}")))?
                        .clone();
                    let w = self.initializers.get(rhs).ok_or_else(|| {
                        OnnxError::Unsupported(format!("MatMul rhs {rhs} must be initializer"))
                    })?;
                    if w.dims.len() != 2 {
                        return Err(OnnxError::InvalidModel("MatMul rhs rank".into()));
                    }
                    let k = w.dims[0];
                    let n = w.dims[1];
                    if a.cols() != k {
                        return Err(OnnxError::InvalidModel("MatMul K mismatch".into()));
                    }
                    let mut bt = vec![f16::ZERO; n * k];
                    for kk in 0..k {
                        for nn in 0..n {
                            bt[nn * k + kk] = w.values[kk * n + nn];
                        }
                    }
                    let b = Matrix::from_vec(n, k, bt)?;
                    let (m, padded) = execute_dense_fp16(&a, &b, target, single.as_deref_mut())?;
                    if target == ExecutionTarget::NpuSingle {
                        stats.npu_dense_nodes += 1;
                        if padded {
                            stats.padded_npu_dense_nodes += 1;
                        }
                    }
                    if trace_enabled {
                        trace.push(TraceTensor {
                            op: "MatMul",
                            name: output.clone(),
                            rows: m.rows(),
                            cols: m.cols(),
                            values: m.values().to_vec(),
                        });
                    }
                    values.insert(output.clone(), m);
                    stats.matmul_nodes += 1;
                }
                Node::Gemm {
                    lhs,
                    rhs,
                    bias,
                    output,
                } => {
                    let a = values
                        .get(lhs)
                        .ok_or_else(|| OnnxError::InvalidModel(format!("missing lhs {lhs}")))?
                        .clone();
                    let w = self.initializers.get(rhs).ok_or_else(|| {
                        OnnxError::Unsupported(format!("Gemm rhs {rhs} must be initializer"))
                    })?;
                    let bias = self.initializers.get(bias).ok_or_else(|| {
                        OnnxError::Unsupported("Gemm bias must be initializer".into())
                    })?;
                    if w.dims.len() != 2 {
                        return Err(OnnxError::InvalidModel("Gemm rhs rank".into()));
                    }
                    let n = w.dims[0];
                    let k = w.dims[1];
                    if a.cols() != k {
                        return Err(OnnxError::InvalidModel("Gemm K mismatch".into()));
                    }
                    if bias.dims.len() != 1 || bias.dims[0] != n {
                        return Err(OnnxError::InvalidModel("Gemm bias shape".into()));
                    }
                    let b = Matrix::from_vec(n, k, w.values.clone())?;
                    let (dense, padded) =
                        execute_dense_fp16(&a, &b, target, single.as_deref_mut())?;
                    let mut v = dense.values().to_vec();
                    for r in 0..dense.rows() {
                        for c in 0..dense.cols() {
                            let i = r * dense.cols() + c;
                            v[i] = f16::from_f32(v[i].to_f32() + bias.values[c].to_f32());
                        }
                    }
                    let m = Matrix::from_vec(dense.rows(), dense.cols(), v)?;
                    if trace_enabled {
                        trace.push(TraceTensor {
                            op: "Gemm",
                            name: output.clone(),
                            rows: m.rows(),
                            cols: m.cols(),
                            values: m.values().to_vec(),
                        });
                    }
                    values.insert(output.clone(), m);
                    stats.gemm_nodes += 1;
                    if target == ExecutionTarget::NpuSingle {
                        stats.npu_dense_nodes += 1;
                        if padded {
                            stats.padded_npu_dense_nodes += 1;
                        }
                    }
                }
                Node::Add { lhs, rhs, output } => {
                    let (mn, bn) =
                        if values.contains_key(lhs) && self.initializers.contains_key(rhs) {
                            (lhs, rhs)
                        } else if values.contains_key(rhs) && self.initializers.contains_key(lhs) {
                            (rhs, lhs)
                        } else {
                            return Err(OnnxError::Unsupported(
                                "Add supports matrix+bias initializer".into(),
                            ));
                        };
                    let a = values.get(mn).unwrap();
                    let bias = self.initializers.get(bn).unwrap();
                    if bias.dims.len() != 1 || bias.dims[0] != a.cols() {
                        return Err(OnnxError::InvalidModel("bias shape".into()));
                    }
                    let mut v = a.values().to_vec();
                    for r in 0..a.rows() {
                        for c in 0..a.cols() {
                            let i = r * a.cols() + c;
                            v[i] = f16::from_f32(v[i].to_f32() + bias.values[c].to_f32());
                        }
                    }
                    let m = Matrix::from_vec(a.rows(), a.cols(), v)?;
                    if trace_enabled {
                        trace.push(TraceTensor {
                            op: "Add",
                            name: output.clone(),
                            rows: m.rows(),
                            cols: m.cols(),
                            values: m.values().to_vec(),
                        });
                    }
                    values.insert(output.clone(), m);
                    stats.add_nodes += 1;
                }
                Node::Relu { input, output } => {
                    let a = values
                        .get(input)
                        .ok_or_else(|| OnnxError::InvalidModel("missing Relu input".into()))?;
                    let v = a
                        .values()
                        .iter()
                        .map(|x| if x.to_f32() < 0.0 { f16::ZERO } else { *x })
                        .collect();
                    let m = Matrix::from_vec(a.rows(), a.cols(), v)?;
                    if trace_enabled {
                        trace.push(TraceTensor {
                            op: "Relu",
                            name: output.clone(),
                            rows: m.rows(),
                            cols: m.cols(),
                            values: m.values().to_vec(),
                        });
                    }
                    values.insert(output.clone(), m);
                    stats.relu_nodes += 1;
                }
            }
        }
        let out = values
            .remove(&self.output_name)
            .ok_or_else(|| OnnxError::InvalidModel("graph output not produced".into()))?;
        Ok((out, stats, trace))
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

/// Execute the project-native dense contract B[N,K]^T. Explicit NPU execution
/// transparently pads M/K/N to the currently validated hardware alignment and
/// crops the result back to the ONNX-visible shape. Zero padding preserves the
/// mathematical product exactly before FP16 rounding.
pub(crate) fn execute_dense_fp16(
    a: &Matrix<f16>,
    b: &Matrix<f16>,
    target: ExecutionTarget,
    single: Option<&mut SingleNpuBackend<'_>>,
) -> Result<(Matrix<f16>, bool), OnnxError> {
    let m = a.rows();
    let k = a.cols();
    let n = b.rows();
    if b.cols() != k {
        return Err(OnnxError::InvalidModel("dense K mismatch".into()));
    }

    let direct_spec = MatmulSpec::new(m, k, n, MatmulPrecision::Fp16Fast, target);
    if target != ExecutionTarget::NpuSingle || direct_spec.npu_shape_supported() {
        let out = execute_auto(direct_spec, a, b, single, None)?;
        return match out {
            MatmulOutput::F16(v) => Ok((v, false)),
            MatmulOutput::F32(_) => Err(OnnxError::InvalidModel("unexpected FP32".into())),
        };
    }

    let mp = align_up(m, 4)?;
    let kp = align_up(k, 32)?;
    let np = align_up(n, 16)?;
    let mut av = vec![f16::ZERO; mp * kp];
    for r in 0..m {
        av[r * kp..r * kp + k].copy_from_slice(&a.values()[r * k..(r + 1) * k]);
    }
    let mut bv = vec![f16::ZERO; np * kp];
    for r in 0..n {
        bv[r * kp..r * kp + k].copy_from_slice(&b.values()[r * k..(r + 1) * k]);
    }
    let ap = Matrix::from_vec(mp, kp, av)?;
    let bp = Matrix::from_vec(np, kp, bv)?;
    let spec = MatmulSpec::new(
        mp,
        kp,
        np,
        MatmulPrecision::Fp16Fast,
        ExecutionTarget::NpuSingle,
    );
    let out = execute_auto(spec, &ap, &bp, single, None)?;
    let padded = match out {
        MatmulOutput::F16(v) => v,
        MatmulOutput::F32(_) => return Err(OnnxError::InvalidModel("unexpected FP32".into())),
    };
    let mut cropped = vec![f16::ZERO; m * n];
    for r in 0..m {
        cropped[r * n..(r + 1) * n].copy_from_slice(&padded.values()[r * np..r * np + n]);
    }
    Ok((Matrix::from_vec(m, n, cropped)?, true))
}

fn parse_initializer(t: &TensorProto) -> Result<ConstTensor, OnnxError> {
    let mut dims = Vec::new();
    for &d in &t.dims {
        if d <= 0 {
            return Err(OnnxError::InvalidModel(
                "nonpositive initializer dim".into(),
            ));
        }
        dims.push(d as usize);
    }
    let expected = dims
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or_else(|| OnnxError::InvalidModel("initializer overflow".into()))?;

    let values: Vec<f16> = if t.data_type == tensor_proto::DataType::FLOAT as i32 {
        let fs: Vec<f32> = if !t.float_data.is_empty() {
            t.float_data.clone()
        } else if !t.raw_data.is_empty() {
            if t.raw_data.len() % 4 != 0 {
                return Err(OnnxError::InvalidModel("bad FLOAT raw_data".into()));
            }
            t.raw_data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        } else {
            Vec::new()
        };
        fs.into_iter().map(f16::from_f32).collect()
    } else if t.data_type == tensor_proto::DataType::FLOAT16 as i32 {
        if t.raw_data.len() != expected * 2 {
            return Err(OnnxError::InvalidModel("bad FLOAT16 raw_data".into()));
        }
        t.raw_data
            .chunks_exact(2)
            .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])))
            .collect()
    } else {
        return Err(OnnxError::Unsupported(format!(
            "initializer dtype {}",
            t.data_type
        )));
    };

    if values.len() != expected {
        return Err(OnnxError::InvalidModel(format!(
            "initializer value count {} != {expected}",
            values.len()
        )));
    }
    Ok(ConstTensor { dims, values })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_garbage() {
        assert!(TinyOnnxModel::from_bytes(&[0xff, 0xff]).is_err());
    }
}
