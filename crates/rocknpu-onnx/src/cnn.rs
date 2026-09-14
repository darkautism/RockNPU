use crate::{OnnxError, execute_dense_fp16};
use half::f16;
use onnx_protobuf::{AttributeValue, Message, ModelProto, TensorProto, tensor_proto};
use rocknpu_conv::{Conv2dSpec, Fp16Conv2dExecutor, Fp16PreparedConvWeights};
use rocknpu_ops::{
    ExecutionTarget, MatmulPrecision, MatmulSpec, PreparedFp16Matmul, SingleNpuBackend,
};
use rocknpu_tensor::Matrix;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

#[derive(Debug, Clone, PartialEq)]
pub struct TensorF16 {
    dims: Vec<usize>,
    values: Vec<f16>,
}
impl TensorF16 {
    pub fn from_vec(dims: Vec<usize>, values: Vec<f16>) -> Result<Self, OnnxError> {
        if dims.is_empty() || dims.iter().any(|&d| d == 0) {
            return Err(OnnxError::InvalidModel(
                "tensor dims must be nonzero".into(),
            ));
        }
        let expected = checked_elements(&dims)?;
        if expected != values.len() {
            return Err(OnnxError::InvalidModel(format!(
                "tensor value count {} != {expected}",
                values.len()
            )));
        }
        Ok(Self { dims, values })
    }
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
    pub fn values(&self) -> &[f16] {
        &self.values
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorTrace {
    pub op: &'static str,
    pub name: String,
    pub dims: Vec<usize>,
    pub values: Vec<f16>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CnnTimingStats {
    pub total_ns: u128,
    pub conv_ns: u128,
    pub maxpool_ns: u128,
    pub reshape_ns: u128,
    pub matmul_ns: u128,
    pub gemm_ns: u128,
    pub add_ns: u128,
    pub relu_ns: u128,
}
impl CnnTimingStats {
    pub const fn node_ns(self) -> u128 {
        self.conv_ns
            + self.maxpool_ns
            + self.reshape_ns
            + self.matmul_ns
            + self.gemm_ns
            + self.add_ns
            + self.relu_ns
    }
    pub const fn graph_overhead_ns(self) -> u128 {
        self.total_ns.saturating_sub(self.node_ns())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CnnPreparedConvStats {
    pub tensors: usize,
    pub resident_bytes: usize,
    pub pack_ns: u128,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CnnPreparedDenseStats {
    pub tensors: usize,
    pub unique_tiles: usize,
    pub resident_bytes: usize,
    pub pack_ns: u128,
}

pub struct CnnPreparedConvState<'d> {
    weights: HashMap<String, Fp16PreparedConvWeights<'d>>,
    dense_weights: HashMap<String, PreparedFp16Matmul<'d>>,
}
impl<'d> CnnPreparedConvState<'d> {
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
            dense_weights: HashMap::new(),
        }
    }
    pub fn stats(&self) -> CnnPreparedConvStats {
        let mut out = CnnPreparedConvStats {
            tensors: self.weights.len(),
            ..Default::default()
        };
        for w in self.weights.values() {
            let s = w.stats();
            out.resident_bytes += s.resident_bytes;
            out.pack_ns += s.pack_ns;
        }
        out
    }
    pub fn dense_stats(&self) -> CnnPreparedDenseStats {
        let mut out = CnnPreparedDenseStats {
            tensors: self.dense_weights.len(),
            ..Default::default()
        };
        for w in self.dense_weights.values() {
            let s = w.weight_stats();
            out.unique_tiles += s.unique_tiles;
            out.resident_bytes += s.resident_bytes;
            out.pack_ns += s.pack_ns;
        }
        out
    }
}
impl Default for CnnPreparedConvState<'_> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CnnRunStats {
    pub conv_nodes: usize,
    pub maxpool_nodes: usize,
    pub reshape_nodes: usize,
    pub matmul_nodes: usize,
    pub gemm_nodes: usize,
    pub add_nodes: usize,
    pub relu_nodes: usize,
    pub npu_dense_nodes: usize,
    pub npu_conv_nodes: usize,
    pub padded_npu_dense_nodes: usize,
}

#[derive(Debug, Clone)]
enum ConstTensor {
    F16(TensorF16),
    I64 { _dims: Vec<usize>, values: Vec<i64> },
}
impl ConstTensor {
    fn f16(&self) -> Result<&TensorF16, OnnxError> {
        match self {
            Self::F16(v) => Ok(v),
            _ => Err(OnnxError::InvalidModel("expected FLOAT initializer".into())),
        }
    }
    fn i64_values(&self) -> Result<&[i64], OnnxError> {
        match self {
            Self::I64 { values, .. } => Ok(values),
            _ => Err(OnnxError::InvalidModel("expected INT64 initializer".into())),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum AutoPad {
    NotSet,
    SameUpper,
}

#[derive(Debug, Clone)]
enum Node {
    Conv {
        input: String,
        weight: String,
        bias: Option<String>,
        output: String,
        kernel: [usize; 2],
        strides: [usize; 2],
        pads: [usize; 4],
        auto_pad: AutoPad,
    },
    MaxPool {
        input: String,
        output: String,
        kernel: [usize; 2],
        strides: [usize; 2],
        pads: [usize; 4],
        auto_pad: AutoPad,
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
    Reshape {
        input: String,
        shape: String,
        output: String,
        allowzero: bool,
    },
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
}

#[derive(Debug, Clone)]
pub struct CnnOnnxModel {
    input_name: String,
    output_name: String,
    nodes: Vec<Node>,
    initializers: HashMap<String, ConstTensor>,
}

impl CnnOnnxModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, OnnxError> {
        let model =
            ModelProto::parse_from_bytes(bytes).map_err(|e| OnnxError::Protobuf(e.to_string()))?;
        let graph = model.graph.as_ref().ok_or(OnnxError::MissingGraph)?;
        let mut initializers = HashMap::new();
        for t in &graph.initializer {
            if t.name.is_empty() {
                return Err(OnnxError::InvalidModel("initializer without name".into()));
            }
            initializers.insert(t.name.clone(), parse_const(t)?);
        }
        let initializer_names: HashSet<&str> = initializers.keys().map(String::as_str).collect();
        let input_name = graph
            .input
            .iter()
            .map(|v| v.name.as_str())
            .find(|n| !initializer_names.contains(*n))
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
                "Conv" => {
                    if !(n.input.len() == 2 || n.input.len() == 3) || n.output.len() != 1 {
                        return Err(OnnxError::InvalidModel("Conv arity".into()));
                    }
                    let conv_weight = initializers
                        .get(&n.input[1])
                        .ok_or_else(|| {
                            OnnxError::InvalidModel(format!("missing Conv weight {}", n.input[1]))
                        })?
                        .f16()?;
                    let default_kernel = if conv_weight.dims.len() == 4 {
                        Some([conv_weight.dims[2], conv_weight.dims[3]])
                    } else {
                        None
                    };
                    let attrs = spatial_attrs(
                        &n.attribute,
                        [1, 1],
                        AutoPad::NotSet,
                        "Conv",
                        default_kernel,
                    )?;
                    let group = int_attr(&n.attribute, "group", 1)?;
                    let dil = ints_attr(&n.attribute, "dilations", vec![1, 1])?;
                    if group != 1 || dil != [1, 1] {
                        return Err(OnnxError::Unsupported(
                            "CNN fallback Conv requires group=1 dilations=[1,1]".into(),
                        ));
                    }
                    nodes.push(Node::Conv {
                        input: n.input[0].clone(),
                        weight: n.input[1].clone(),
                        bias: n.input.get(2).cloned(),
                        output: n.output[0].clone(),
                        kernel: attrs.0,
                        strides: attrs.1,
                        pads: attrs.2,
                        auto_pad: attrs.3,
                    });
                }
                "MaxPool" => {
                    require_arity(n.input.len(), 1, n.output.len(), 1, "MaxPool")?;
                    let attrs =
                        spatial_attrs(&n.attribute, [1, 1], AutoPad::NotSet, "MaxPool", None)?;
                    let ceil_mode = int_attr(&n.attribute, "ceil_mode", 0)?;
                    let storage_order = int_attr(&n.attribute, "storage_order", 0)?;
                    let dil = ints_attr(&n.attribute, "dilations", vec![1, 1])?;
                    if ceil_mode != 0 || storage_order != 0 || dil != [1, 1] {
                        return Err(OnnxError::Unsupported(
                            "CNN MaxPool requires ceil_mode=0 storage_order=0 dilations=[1,1]"
                                .into(),
                        ));
                    }
                    nodes.push(Node::MaxPool {
                        input: n.input[0].clone(),
                        output: n.output[0].clone(),
                        kernel: attrs.0,
                        strides: attrs.1,
                        pads: attrs.2,
                        auto_pad: attrs.3,
                    });
                }
                "Add" => {
                    require_arity(n.input.len(), 2, n.output.len(), 1, "Add")?;
                    nodes.push(Node::Add {
                        lhs: n.input[0].clone(),
                        rhs: n.input[1].clone(),
                        output: n.output[0].clone(),
                    });
                }
                "Relu" => {
                    require_arity(n.input.len(), 1, n.output.len(), 1, "Relu")?;
                    nodes.push(Node::Relu {
                        input: n.input[0].clone(),
                        output: n.output[0].clone(),
                    });
                }
                "Reshape" => {
                    require_arity(n.input.len(), 2, n.output.len(), 1, "Reshape")?;
                    let allowzero = int_attr(&n.attribute, "allowzero", 0)?;
                    if allowzero != 0 && allowzero != 1 {
                        return Err(OnnxError::Unsupported(
                            "Reshape allowzero must be 0 or 1".into(),
                        ));
                    }
                    nodes.push(Node::Reshape {
                        input: n.input[0].clone(),
                        shape: n.input[1].clone(),
                        output: n.output[0].clone(),
                        allowzero: allowzero == 1,
                    });
                }
                "MatMul" => {
                    require_arity(n.input.len(), 2, n.output.len(), 1, "MatMul")?;
                    nodes.push(Node::MatMul {
                        lhs: n.input[0].clone(),
                        rhs: n.input[1].clone(),
                        output: n.output[0].clone(),
                    });
                }
                "Gemm" => {
                    require_arity(n.input.len(), 3, n.output.len(), 1, "Gemm")?;
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
                        return Err(OnnxError::Unsupported(
                            "CNN Gemm currently requires alpha=beta=1 transA=0 transB=1".into(),
                        ));
                    }
                    nodes.push(Node::Gemm {
                        lhs: n.input[0].clone(),
                        rhs: n.input[1].clone(),
                        bias: n.input[2].clone(),
                        output: n.output[0].clone(),
                    });
                }
                other => return Err(OnnxError::Unsupported(format!("CNN operator {other}"))),
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

    pub fn run_fp16<'d>(
        &self,
        input: &TensorF16,
        target: ExecutionTarget,
        single: Option<&mut SingleNpuBackend<'d>>,
    ) -> Result<(TensorF16, CnnRunStats), OnnxError> {
        let (out, stats, _) = self.run_impl(input, target, single, None, None, None, false)?;
        Ok((out, stats))
    }
    pub fn run_fp16_traced<'d>(
        &self,
        input: &TensorF16,
        target: ExecutionTarget,
        single: Option<&mut SingleNpuBackend<'d>>,
    ) -> Result<(TensorF16, CnnRunStats, Vec<TensorTrace>), OnnxError> {
        self.run_impl(input, target, single, None, None, None, true)
    }

    pub fn run_fp16_with_conv<'d>(
        &self,
        input: &TensorF16,
        target: ExecutionTarget,
        single: Option<&mut SingleNpuBackend<'d>>,
        conv: Option<&mut Fp16Conv2dExecutor<'d>>,
    ) -> Result<(TensorF16, CnnRunStats), OnnxError> {
        let (out, stats, _) = self.run_impl(input, target, single, conv, None, None, false)?;
        Ok((out, stats))
    }
    pub fn run_fp16_traced_with_conv<'d>(
        &self,
        input: &TensorF16,
        target: ExecutionTarget,
        single: Option<&mut SingleNpuBackend<'d>>,
        conv: Option<&mut Fp16Conv2dExecutor<'d>>,
    ) -> Result<(TensorF16, CnnRunStats, Vec<TensorTrace>), OnnxError> {
        self.run_impl(input, target, single, conv, None, None, true)
    }

    pub fn run_fp16_with_prepared_conv<'d>(
        &self,
        input: &TensorF16,
        target: ExecutionTarget,
        single: Option<&mut SingleNpuBackend<'d>>,
        conv: Option<&mut Fp16Conv2dExecutor<'d>>,
        prepared: Option<&mut CnnPreparedConvState<'d>>,
    ) -> Result<(TensorF16, CnnRunStats), OnnxError> {
        let (out, stats, _) = self.run_impl(input, target, single, conv, prepared, None, false)?;
        Ok((out, stats))
    }
    pub fn run_fp16_traced_with_prepared_conv<'d>(
        &self,
        input: &TensorF16,
        target: ExecutionTarget,
        single: Option<&mut SingleNpuBackend<'d>>,
        conv: Option<&mut Fp16Conv2dExecutor<'d>>,
        prepared: Option<&mut CnnPreparedConvState<'d>>,
    ) -> Result<(TensorF16, CnnRunStats, Vec<TensorTrace>), OnnxError> {
        self.run_impl(input, target, single, conv, prepared, None, true)
    }

    pub fn run_fp16_profiled_with_prepared_conv<'d>(
        &self,
        input: &TensorF16,
        target: ExecutionTarget,
        single: Option<&mut SingleNpuBackend<'d>>,
        conv: Option<&mut Fp16Conv2dExecutor<'d>>,
        prepared: Option<&mut CnnPreparedConvState<'d>>,
    ) -> Result<(TensorF16, CnnRunStats, CnnTimingStats), OnnxError> {
        let mut timing = CnnTimingStats::default();
        let (out, stats, _) = self.run_impl(
            input,
            target,
            single,
            conv,
            prepared,
            Some(&mut timing),
            false,
        )?;
        Ok((out, stats, timing))
    }

    fn run_impl<'d>(
        &self,
        input: &TensorF16,
        target: ExecutionTarget,
        mut single: Option<&mut SingleNpuBackend<'d>>,
        mut conv: Option<&mut Fp16Conv2dExecutor<'d>>,
        mut prepared_conv: Option<&mut CnnPreparedConvState<'d>>,
        mut timing: Option<&mut CnnTimingStats>,
        trace_enabled: bool,
    ) -> Result<(TensorF16, CnnRunStats, Vec<TensorTrace>), OnnxError> {
        let mut values: HashMap<String, TensorF16> = HashMap::new();
        values.insert(self.input_name.clone(), input.clone());
        let mut stats = CnnRunStats::default();
        let mut trace = Vec::new();
        let graph_started = timing.as_ref().map(|_| Instant::now());
        for node in &self.nodes {
            let node_started = timing.as_ref().map(|_| Instant::now());
            let (op, name, result) = match node {
                Node::Conv {
                    input,
                    weight,
                    bias,
                    output,
                    kernel,
                    strides,
                    pads,
                    auto_pad,
                } => {
                    let x = values.get(input).ok_or_else(|| {
                        OnnxError::InvalidModel(format!("missing Conv input {input}"))
                    })?;
                    let w = self
                        .initializers
                        .get(weight)
                        .ok_or_else(|| {
                            OnnxError::InvalidModel(format!("missing Conv weight {weight}"))
                        })?
                        .f16()?;
                    let y = if target == ExecutionTarget::NpuSingle
                        && conv.is_some()
                        && strides[0] >= 1
                        && strides[0] <= 2
                        && strides[1] >= 1
                        && strides[1] <= 2
                        && x.dims.len() == 4
                        && x.dims[0] == 1
                        && w.dims.len() == 4
                    {
                        let pad = match auto_pad {
                            AutoPad::SameUpper if kernel[0] % 2 == 1 && kernel[1] % 2 == 1 => {
                                Some([(kernel[0] - 1) / 2, (kernel[1] - 1) / 2])
                            }
                            AutoPad::NotSet if pads[0] == pads[2] && pads[1] == pads[3] => {
                                Some([pads[0], pads[1]])
                            }
                            _ => None,
                        };
                        if let Some([pt, pl]) = pad {
                            let spec = Conv2dSpec {
                                ic: x.dims[1],
                                ih: x.dims[2],
                                iw: x.dims[3],
                                oc: w.dims[0],
                                kh: w.dims[2],
                                kw: w.dims[3],
                                pad_top: pt,
                                pad_left: pl,
                                stride_y: strides[0],
                                stride_x: strides[1],
                            };
                            if w.dims[1] != spec.ic || kernel != &[spec.kh, spec.kw] {
                                return Err(OnnxError::InvalidModel(
                                    "NPU Conv weight/kernel mismatch".into(),
                                ));
                            }
                            let conv_exec = conv.as_deref_mut().unwrap();
                            let outv = if let Some(state) = prepared_conv.as_deref_mut() {
                                if !state.weights.contains_key(weight) {
                                    let resident = conv_exec.prepare_weights(&w.values, spec)?;
                                    state.weights.insert(weight.clone(), resident);
                                }
                                let resident = state.weights.get(weight).ok_or_else(|| {
                                    OnnxError::InvalidModel(
                                        "prepared Conv weight insertion failed".into(),
                                    )
                                })?;
                                if resident.spec() != spec {
                                    return Err(OnnxError::InvalidModel(
                                        "prepared Conv weight spec changed across runs".into(),
                                    ));
                                }
                                conv_exec.execute_prepared(&x.values, resident)?
                            } else {
                                conv_exec.execute(&x.values, &w.values, spec)?
                            };
                            stats.npu_conv_nodes += 1;
                            TensorF16::from_vec(
                                vec![1, spec.oc, spec.output_h(), spec.output_w()],
                                outv,
                            )?
                        } else {
                            conv2d(x, w, *kernel, *strides, *pads, *auto_pad)?
                        }
                    } else {
                        conv2d(x, w, *kernel, *strides, *pads, *auto_pad)?
                    };
                    let y = if let Some(bias_name) = bias {
                        let b = self
                            .initializers
                            .get(bias_name)
                            .ok_or_else(|| {
                                OnnxError::InvalidModel(format!("missing Conv bias {bias_name}"))
                            })?
                            .f16()?;
                        add_conv_bias(&y, b)?
                    } else {
                        y
                    };
                    stats.conv_nodes += 1;
                    ("Conv", output, y)
                }
                Node::MaxPool {
                    input,
                    output,
                    kernel,
                    strides,
                    pads,
                    auto_pad,
                } => {
                    let x = values.get(input).ok_or_else(|| {
                        OnnxError::InvalidModel(format!("missing MaxPool input {input}"))
                    })?;
                    let y = maxpool2d(x, *kernel, *strides, *pads, *auto_pad)?;
                    stats.maxpool_nodes += 1;
                    ("MaxPool", output, y)
                }
                Node::Add { lhs, rhs, output } => {
                    let l = resolve_f16(lhs, &values, &self.initializers)?;
                    let r = resolve_f16(rhs, &values, &self.initializers)?;
                    let y = add_broadcast(l, r)?;
                    stats.add_nodes += 1;
                    ("Add", output, y)
                }
                Node::Relu { input, output } => {
                    let x = values.get(input).ok_or_else(|| {
                        OnnxError::InvalidModel(format!("missing Relu input {input}"))
                    })?;
                    let y = TensorF16::from_vec(
                        x.dims.clone(),
                        x.values
                            .iter()
                            .map(|&v| if v.to_f32() < 0.0 { f16::ZERO } else { v })
                            .collect(),
                    )?;
                    stats.relu_nodes += 1;
                    ("Relu", output, y)
                }
                Node::Reshape {
                    input,
                    shape,
                    output,
                    allowzero,
                } => {
                    let x = resolve_f16(input, &values, &self.initializers)?;
                    let shape = self
                        .initializers
                        .get(shape)
                        .ok_or_else(|| {
                            OnnxError::InvalidModel("Reshape shape must be initializer".into())
                        })?
                        .i64_values()?;
                    let dims = resolve_reshape(&x.dims, shape, *allowzero)?;
                    let y = TensorF16::from_vec(dims, x.values.clone())?;
                    stats.reshape_nodes += 1;
                    ("Reshape", output, y)
                }
                Node::MatMul { lhs, rhs, output } => {
                    let a = values.get(lhs).ok_or_else(|| {
                        OnnxError::InvalidModel(format!("missing MatMul lhs {lhs}"))
                    })?;
                    let w = resolve_f16(rhs, &values, &self.initializers)?;
                    if a.dims.len() != 2 || w.dims.len() != 2 || a.dims[1] != w.dims[0] {
                        return Err(OnnxError::InvalidModel(
                            "CNN MatMul expects [M,K] x [K,N]".into(),
                        ));
                    }
                    let (m, k, n) = (a.dims[0], a.dims[1], w.dims[1]);
                    let am = Matrix::from_vec(m, k, a.values.clone())?;
                    let use_prepared_dense = target == ExecutionTarget::NpuSingle
                        && prepared_conv.is_some()
                        && single.is_some()
                        && self.is_static_f16_value(rhs);
                    let (outm, padded) = if use_prepared_dense {
                        execute_dense_prepared_fp16(
                            &am,
                            w,
                            rhs,
                            single.as_deref_mut().unwrap(),
                            prepared_conv.as_deref_mut().unwrap(),
                        )?
                    } else {
                        let mut bt = vec![f16::ZERO; n * k];
                        for kk in 0..k {
                            for nn in 0..n {
                                bt[nn * k + kk] = w.values[kk * n + nn];
                            }
                        }
                        let bm = Matrix::from_vec(n, k, bt)?;
                        execute_dense_fp16(&am, &bm, target, single.as_deref_mut())?
                    };
                    if target == ExecutionTarget::NpuSingle {
                        stats.npu_dense_nodes += 1;
                        if padded {
                            stats.padded_npu_dense_nodes += 1;
                        }
                    }
                    let y = TensorF16::from_vec(vec![m, n], outm.values().to_vec())?;
                    stats.matmul_nodes += 1;
                    ("MatMul", output, y)
                }
                Node::Gemm {
                    lhs,
                    rhs,
                    bias,
                    output,
                } => {
                    let a = values.get(lhs).ok_or_else(|| {
                        OnnxError::InvalidModel(format!("missing Gemm lhs {lhs}"))
                    })?;
                    let w = self
                        .initializers
                        .get(rhs)
                        .ok_or_else(|| {
                            OnnxError::InvalidModel(format!("missing Gemm weight {rhs}"))
                        })?
                        .f16()?;
                    let b = self
                        .initializers
                        .get(bias)
                        .ok_or_else(|| {
                            OnnxError::InvalidModel(format!("missing Gemm bias {bias}"))
                        })?
                        .f16()?;
                    if a.dims.len() != 2
                        || w.dims.len() != 2
                        || b.dims.len() != 1
                        || a.dims[1] != w.dims[1]
                        || b.dims[0] != w.dims[0]
                    {
                        return Err(OnnxError::InvalidModel(
                            "CNN Gemm expects A[M,K], B[N,K] with transB=1, C[N]".into(),
                        ));
                    }
                    let (m, k, n) = (a.dims[0], a.dims[1], w.dims[0]);
                    let am = Matrix::from_vec(m, k, a.values.clone())?;
                    let bm = Matrix::from_vec(n, k, w.values.clone())?;
                    let (outm, padded) =
                        execute_dense_fp16(&am, &bm, target, single.as_deref_mut())?;
                    let mut vals = outm.values().to_vec();
                    for r in 0..m {
                        for c in 0..n {
                            vals[r * n + c] =
                                f16::from_f32(vals[r * n + c].to_f32() + b.values[c].to_f32());
                        }
                    }
                    if target == ExecutionTarget::NpuSingle {
                        stats.npu_dense_nodes += 1;
                        if padded {
                            stats.padded_npu_dense_nodes += 1;
                        }
                    }
                    stats.gemm_nodes += 1;
                    ("Gemm", output, TensorF16::from_vec(vec![m, n], vals)?)
                }
            };
            if let (Some(t), Some(started)) = (timing.as_deref_mut(), node_started) {
                let ns = started.elapsed().as_nanos();
                match op {
                    "Conv" => t.conv_ns += ns,
                    "MaxPool" => t.maxpool_ns += ns,
                    "Reshape" => t.reshape_ns += ns,
                    "MatMul" => t.matmul_ns += ns,
                    "Gemm" => t.gemm_ns += ns,
                    "Add" => t.add_ns += ns,
                    "Relu" => t.relu_ns += ns,
                    _ => {}
                }
            }
            if trace_enabled {
                trace.push(TensorTrace {
                    op,
                    name: name.clone(),
                    dims: result.dims.clone(),
                    values: result.values.clone(),
                });
            }
            values.insert(name.clone(), result);
        }
        let out = values
            .remove(&self.output_name)
            .ok_or_else(|| OnnxError::InvalidModel("CNN graph output not produced".into()))?;
        if let (Some(t), Some(started)) = (timing.as_deref_mut(), graph_started) {
            t.total_ns = started.elapsed().as_nanos();
        }
        Ok((out, stats, trace))
    }

    fn is_static_f16_value(&self, name: &str) -> bool {
        if matches!(self.initializers.get(name), Some(ConstTensor::F16(_))) {
            return true;
        }
        self.nodes.iter().any(|node| match node {
            Node::Reshape {
                input,
                shape,
                output,
                ..
            } if output == name => {
                matches!(self.initializers.get(input), Some(ConstTensor::F16(_)))
                    && matches!(self.initializers.get(shape), Some(ConstTensor::I64 { .. }))
            }
            _ => false,
        })
    }
}

fn align_dense_up(value: usize, alignment: usize) -> Result<usize, OnnxError> {
    value
        .checked_add(alignment - 1)
        .map(|v| (v / alignment) * alignment)
        .ok_or_else(|| OnnxError::InvalidModel("dense alignment overflow".into()))
}

fn execute_dense_prepared_fp16<'d>(
    a: &Matrix<f16>,
    w_kn: &TensorF16,
    key: &str,
    backend: &mut SingleNpuBackend<'d>,
    state: &mut CnnPreparedConvState<'d>,
) -> Result<(Matrix<f16>, bool), OnnxError> {
    if w_kn.dims.len() != 2 || a.cols() != w_kn.dims[0] {
        return Err(OnnxError::InvalidModel(
            "prepared dense shape mismatch".into(),
        ));
    }
    let m = a.rows();
    let k = a.cols();
    let n = w_kn.dims[1];
    let mp = align_dense_up(m, 4)?;
    let kp = align_dense_up(k, 32)?;
    let np = align_dense_up(n, 16)?;
    let spec = MatmulSpec::new(
        mp,
        kp,
        np,
        MatmulPrecision::Fp16Fast,
        ExecutionTarget::NpuSingle,
    );
    if !state.dense_weights.contains_key(key) {
        let mut bv = vec![f16::ZERO; np * kp];
        for kk in 0..k {
            for nn in 0..n {
                bv[nn * kp + kk] = w_kn.values[kk * n + nn];
            }
        }
        let bp = Matrix::from_vec(np, kp, bv)?;
        let prepared = backend.prepare_fp16(spec, &bp)?;
        state.dense_weights.insert(key.to_string(), prepared);
    }
    let prepared = state
        .dense_weights
        .get(key)
        .ok_or_else(|| OnnxError::InvalidModel("prepared dense insertion failed".into()))?;
    if prepared.spec() != spec {
        return Err(OnnxError::InvalidModel(
            "prepared dense shape changed across runs".into(),
        ));
    }
    let mut av = vec![f16::ZERO; mp * kp];
    for r in 0..m {
        av[r * kp..r * kp + k].copy_from_slice(&a.values()[r * k..(r + 1) * k]);
    }
    let ap = Matrix::from_vec(mp, kp, av)?;
    let padded_out = backend.execute_prepared_fp16(prepared, &ap)?;
    let mut cropped = vec![f16::ZERO; m * n];
    for r in 0..m {
        cropped[r * n..(r + 1) * n].copy_from_slice(&padded_out.values()[r * np..r * np + n]);
    }
    Ok((
        Matrix::from_vec(m, n, cropped)?,
        mp != m || kp != k || np != n,
    ))
}

fn require_arity(
    inputs: usize,
    want_i: usize,
    outputs: usize,
    want_o: usize,
    op: &str,
) -> Result<(), OnnxError> {
    if inputs != want_i || outputs != want_o {
        Err(OnnxError::InvalidModel(format!("{op} arity")))
    } else {
        Ok(())
    }
}

fn int_attr(
    attrs: &[onnx_protobuf::AttributeProto],
    name: &str,
    default: i64,
) -> Result<i64, OnnxError> {
    for a in attrs {
        if a.name == name {
            return match a.as_value() {
                AttributeValue::Integer64(v) => Ok(v),
                _ => Err(OnnxError::Unsupported(format!("{name} attribute type"))),
            };
        }
    }
    Ok(default)
}
fn ints_attr(
    attrs: &[onnx_protobuf::AttributeProto],
    name: &str,
    default: Vec<i64>,
) -> Result<Vec<i64>, OnnxError> {
    for a in attrs {
        if a.name == name {
            return match a.as_value() {
                AttributeValue::Integer64s(v) => Ok(v.to_vec()),
                _ => Err(OnnxError::Unsupported(format!("{name} attribute type"))),
            };
        }
    }
    Ok(default)
}
fn string_attr(
    attrs: &[onnx_protobuf::AttributeProto],
    name: &str,
    default: &[u8],
) -> Result<Vec<u8>, OnnxError> {
    for a in attrs {
        if a.name == name {
            return match a.as_value() {
                AttributeValue::String(v) => Ok(v.to_vec()),
                _ => Err(OnnxError::Unsupported(format!("{name} attribute type"))),
            };
        }
    }
    Ok(default.to_vec())
}
fn spatial_attrs(
    attrs: &[onnx_protobuf::AttributeProto],
    default_stride: [usize; 2],
    default_pad: AutoPad,
    op: &str,
    default_kernel: Option<[usize; 2]>,
) -> Result<([usize; 2], [usize; 2], [usize; 4], AutoPad), OnnxError> {
    let kernel_attr = ints_attr(attrs, "kernel_shape", Vec::new())?;
    let kernel = if kernel_attr.is_empty() {
        default_kernel
            .ok_or_else(|| OnnxError::Unsupported(format!("{op} requires 2D kernel_shape")))?
    } else if kernel_attr.len() == 2 && kernel_attr.iter().all(|&x| x > 0) {
        [kernel_attr[0] as usize, kernel_attr[1] as usize]
    } else {
        return Err(OnnxError::Unsupported(format!(
            "{op} requires 2D kernel_shape"
        )));
    };
    let strides = ints_attr(
        attrs,
        "strides",
        default_stride.iter().map(|&x| x as i64).collect(),
    )?;
    let pads = ints_attr(attrs, "pads", vec![0, 0, 0, 0])?;
    if strides.len() != 2
        || pads.len() != 4
        || strides.iter().any(|&x| x <= 0)
        || pads.iter().any(|&x| x < 0)
    {
        return Err(OnnxError::Unsupported(format!("{op} strides/pads")));
    }
    let auto = string_attr(
        attrs,
        "auto_pad",
        match default_pad {
            AutoPad::NotSet => b"NOTSET",
            AutoPad::SameUpper => b"SAME_UPPER",
        },
    )?;
    let auto_pad = match auto.as_slice() {
        b"NOTSET" => AutoPad::NotSet,
        b"SAME_UPPER" => AutoPad::SameUpper,
        _ => {
            return Err(OnnxError::Unsupported(format!(
                "{op} auto_pad {:?}",
                String::from_utf8_lossy(&auto)
            )));
        }
    };
    Ok((
        kernel,
        [strides[0] as usize, strides[1] as usize],
        [
            pads[0] as usize,
            pads[1] as usize,
            pads[2] as usize,
            pads[3] as usize,
        ],
        auto_pad,
    ))
}

fn checked_elements(dims: &[usize]) -> Result<usize, OnnxError> {
    dims.iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or_else(|| OnnxError::InvalidModel("tensor element count overflow".into()))
}

fn parse_const(t: &TensorProto) -> Result<ConstTensor, OnnxError> {
    let dims: Vec<usize> = t
        .dims
        .iter()
        .map(|&d| {
            if d > 0 {
                Ok(d as usize)
            } else {
                Err(OnnxError::InvalidModel(
                    "nonpositive initializer dim".into(),
                ))
            }
        })
        .collect::<Result<_, _>>()?;
    let expected = checked_elements(&dims)?;
    if t.data_type == tensor_proto::DataType::FLOAT as i32 {
        let fs: Vec<f32> = if !t.float_data.is_empty() {
            t.float_data.clone()
        } else if !t.raw_data.is_empty() {
            if t.raw_data.len() != expected * 4 {
                return Err(OnnxError::InvalidModel("bad FLOAT raw_data".into()));
            }
            t.raw_data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        } else {
            Vec::new()
        };
        if fs.len() != expected {
            return Err(OnnxError::InvalidModel("FLOAT initializer count".into()));
        }
        Ok(ConstTensor::F16(TensorF16::from_vec(
            dims,
            fs.into_iter().map(f16::from_f32).collect(),
        )?))
    } else if t.data_type == tensor_proto::DataType::FLOAT16 as i32 {
        if t.raw_data.len() != expected * 2 {
            return Err(OnnxError::InvalidModel("bad FLOAT16 raw_data".into()));
        }
        let v = t
            .raw_data
            .chunks_exact(2)
            .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])))
            .collect();
        Ok(ConstTensor::F16(TensorF16::from_vec(dims, v)?))
    } else if t.data_type == tensor_proto::DataType::INT64 as i32 {
        let v: Vec<i64> = if !t.int64_data.is_empty() {
            t.int64_data.clone()
        } else if !t.raw_data.is_empty() {
            if t.raw_data.len() != expected * 8 {
                return Err(OnnxError::InvalidModel("bad INT64 raw_data".into()));
            }
            t.raw_data
                .chunks_exact(8)
                .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
                .collect()
        } else {
            Vec::new()
        };
        if v.len() != expected {
            return Err(OnnxError::InvalidModel("INT64 initializer count".into()));
        }
        Ok(ConstTensor::I64 {
            _dims: dims,
            values: v,
        })
    } else {
        Err(OnnxError::Unsupported(format!(
            "CNN initializer dtype {}",
            t.data_type
        )))
    }
}

fn resolve_f16<'a>(
    name: &str,
    values: &'a HashMap<String, TensorF16>,
    initializers: &'a HashMap<String, ConstTensor>,
) -> Result<&'a TensorF16, OnnxError> {
    if let Some(v) = values.get(name) {
        return Ok(v);
    }
    initializers
        .get(name)
        .ok_or_else(|| OnnxError::InvalidModel(format!("missing value {name}")))?
        .f16()
}

fn resolve_reshape(
    input_dims: &[usize],
    shape: &[i64],
    allowzero: bool,
) -> Result<Vec<usize>, OnnxError> {
    if shape.is_empty() {
        return Err(OnnxError::InvalidModel("empty Reshape shape".into()));
    }
    let input_elems = checked_elements(input_dims)?;
    let mut out = Vec::with_capacity(shape.len());
    let mut infer = None;
    let mut known = 1usize;
    for (i, &d) in shape.iter().enumerate() {
        if d == -1 {
            if infer.is_some() {
                return Err(OnnxError::Unsupported("multiple -1 Reshape dims".into()));
            }
            infer = Some(i);
            out.push(1);
        } else if d == 0 {
            if allowzero {
                return Err(OnnxError::Unsupported(
                    "Reshape allowzero=1 with an actual zero target dim is not supported".into(),
                ));
            }
            if i >= input_dims.len() {
                return Err(OnnxError::InvalidModel("Reshape 0 dim out of range".into()));
            }
            out.push(input_dims[i]);
            known = known
                .checked_mul(input_dims[i])
                .ok_or_else(|| OnnxError::InvalidModel("Reshape overflow".into()))?;
        } else if d > 0 {
            out.push(d as usize);
            known = known
                .checked_mul(d as usize)
                .ok_or_else(|| OnnxError::InvalidModel("Reshape overflow".into()))?;
        } else {
            return Err(OnnxError::Unsupported(
                "negative Reshape dim other than -1".into(),
            ));
        }
    }
    if let Some(i) = infer {
        if known == 0 || input_elems % known != 0 {
            return Err(OnnxError::InvalidModel("Reshape inference mismatch".into()));
        }
        out[i] = input_elems / known;
    }
    if checked_elements(&out)? != input_elems {
        return Err(OnnxError::InvalidModel("Reshape element mismatch".into()));
    }
    Ok(out)
}

fn spatial_geometry(
    input: usize,
    kernel: usize,
    stride: usize,
    pad_before: usize,
    pad_after: usize,
    auto: AutoPad,
) -> (usize, usize, usize) {
    match auto {
        AutoPad::SameUpper => {
            let out = input.div_ceil(stride);
            let needed = ((out.saturating_sub(1)) * stride + kernel).saturating_sub(input);
            let before = needed / 2;
            (out, before, needed - before)
        }
        AutoPad::NotSet => {
            let out = (input + pad_before + pad_after - kernel) / stride + 1;
            (out, pad_before, pad_after)
        }
    }
}

fn conv2d(
    x: &TensorF16,
    w: &TensorF16,
    kernel: [usize; 2],
    strides: [usize; 2],
    pads: [usize; 4],
    auto: AutoPad,
) -> Result<TensorF16, OnnxError> {
    if x.dims.len() != 4 || w.dims.len() != 4 {
        return Err(OnnxError::InvalidModel("Conv expects NCHW/OIHW".into()));
    }
    let (batch, cin, h, ww) = (x.dims[0], x.dims[1], x.dims[2], x.dims[3]);
    let (cout, wcin, kh, kw) = (w.dims[0], w.dims[1], w.dims[2], w.dims[3]);
    if cin != wcin || [kh, kw] != kernel {
        return Err(OnnxError::InvalidModel(
            "Conv channel/kernel mismatch".into(),
        ));
    }
    let (oh, pt, _pb) = spatial_geometry(h, kh, strides[0], pads[0], pads[2], auto);
    let (ow, pl, _pr) = spatial_geometry(ww, kw, strides[1], pads[1], pads[3], auto);
    let mut out = vec![f16::ZERO; batch * cout * oh * ow];
    for n in 0..batch {
        for oc in 0..cout {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = 0.0f32;
                    for ic in 0..cin {
                        for ky in 0..kh {
                            let iy = oy * strides[0] + ky;
                            if iy < pt {
                                continue;
                            }
                            let iy = iy - pt;
                            if iy >= h {
                                continue;
                            }
                            for kx in 0..kw {
                                let ix = ox * strides[1] + kx;
                                if ix < pl {
                                    continue;
                                }
                                let ix = ix - pl;
                                if ix >= ww {
                                    continue;
                                }
                                let xi = ((n * cin + ic) * h + iy) * ww + ix;
                                let wi = ((oc * cin + ic) * kh + ky) * kw + kx;
                                acc += x.values[xi].to_f32() * w.values[wi].to_f32();
                            }
                        }
                    }
                    out[((n * cout + oc) * oh + oy) * ow + ox] = f16::from_f32(acc);
                }
            }
        }
    }
    TensorF16::from_vec(vec![batch, cout, oh, ow], out)
}

fn add_conv_bias(x: &TensorF16, bias: &TensorF16) -> Result<TensorF16, OnnxError> {
    if x.dims.len() != 4 || bias.dims.len() != 1 || bias.dims[0] != x.dims[1] {
        return Err(OnnxError::InvalidModel(
            "Conv bias expects rank-1 tensor matching output channels".into(),
        ));
    }
    let (batch, channels, h, w) = (x.dims[0], x.dims[1], x.dims[2], x.dims[3]);
    let mut out = x.values.clone();
    for n in 0..batch {
        for c in 0..channels {
            let b = bias.values[c].to_f32();
            let base = (n * channels + c) * h * w;
            for v in &mut out[base..base + h * w] {
                *v = f16::from_f32(v.to_f32() + b);
            }
        }
    }
    TensorF16::from_vec(x.dims.clone(), out)
}

fn maxpool2d(
    x: &TensorF16,
    kernel: [usize; 2],
    strides: [usize; 2],
    pads: [usize; 4],
    auto: AutoPad,
) -> Result<TensorF16, OnnxError> {
    if x.dims.len() != 4 {
        return Err(OnnxError::InvalidModel("MaxPool expects NCHW".into()));
    }
    let (batch, c, h, w) = (x.dims[0], x.dims[1], x.dims[2], x.dims[3]);

    // Dominant inference path for MNIST-like CNNs: non-overlapping 2x2 pooling.
    // Every sampled coordinate is in range, so avoid the generic per-element
    // padding/stride/kernel checks. Keep the exact existing `>` comparison
    // behavior rather than using f32::max, including its NaN semantics.
    if kernel == [2, 2]
        && strides == [2, 2]
        && pads == [0, 0, 0, 0]
        && matches!(auto, AutoPad::NotSet)
        && h >= 2
        && w >= 2
    {
        let oh = h / 2;
        let ow = w / 2;
        let mut out = Vec::with_capacity(batch * c * oh * ow);
        for n in 0..batch {
            for ch in 0..c {
                let plane = (n * c + ch) * h * w;
                for oy in 0..oh {
                    let row = plane + (oy * 2) * w;
                    for ox in 0..ow {
                        let base = row + ox * 2;
                        let mut best = f16::NEG_INFINITY;
                        for idx in [base, base + 1, base + w, base + w + 1] {
                            let v = x.values[idx];
                            if v.to_f32() > best.to_f32() {
                                best = v;
                            }
                        }
                        out.push(best);
                    }
                }
            }
        }
        return TensorF16::from_vec(vec![batch, c, oh, ow], out);
    }

    let (oh, pt, _) = spatial_geometry(h, kernel[0], strides[0], pads[0], pads[2], auto);
    let (ow, pl, _) = spatial_geometry(w, kernel[1], strides[1], pads[1], pads[3], auto);
    let mut out = vec![f16::NEG_INFINITY; batch * c * oh * ow];
    for n in 0..batch {
        for ch in 0..c {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut best = f16::NEG_INFINITY;
                    for ky in 0..kernel[0] {
                        let iy = oy * strides[0] + ky;
                        if iy < pt {
                            continue;
                        }
                        let iy = iy - pt;
                        if iy >= h {
                            continue;
                        }
                        for kx in 0..kernel[1] {
                            let ix = ox * strides[1] + kx;
                            if ix < pl {
                                continue;
                            }
                            let ix = ix - pl;
                            if ix >= w {
                                continue;
                            }
                            let v = x.values[((n * c + ch) * h + iy) * w + ix];
                            if v.to_f32() > best.to_f32() {
                                best = v;
                            }
                        }
                    }
                    out[((n * c + ch) * oh + oy) * ow + ox] = best;
                }
            }
        }
    }
    TensorF16::from_vec(vec![batch, c, oh, ow], out)
}

fn add_broadcast(a: &TensorF16, b: &TensorF16) -> Result<TensorF16, OnnxError> {
    // Hot exact-shape path (e.g. the final [1,N] bias Add). Preserve the
    // established FP16 -> FP32 add -> FP16 rounding contract exactly.
    if a.dims == b.dims {
        let out = a
            .values
            .iter()
            .zip(&b.values)
            .map(|(&x, &y)| f16::from_f32(x.to_f32() + y.to_f32()))
            .collect();
        return TensorF16::from_vec(a.dims.clone(), out);
    }

    // NCHW activation + per-channel bias is the dominant CNN Add shape. ONNX
    // right-aligned broadcasting represents the bias as [C,1,1] or [1,C,1,1].
    if let Some(out) = add_nchw_channel_bias(a, b)? {
        return Ok(out);
    }
    if let Some(out) = add_nchw_channel_bias(b, a)? {
        return Ok(out);
    }

    // Fully generic right-aligned NumPy/ONNX broadcast fallback.
    let rank = a.dims.len().max(b.dims.len());
    let mut ad = vec![1usize; rank - a.dims.len()];
    ad.extend_from_slice(&a.dims);
    let mut bd = vec![1usize; rank - b.dims.len()];
    bd.extend_from_slice(&b.dims);
    let mut od = Vec::with_capacity(rank);
    for i in 0..rank {
        if ad[i] != bd[i] && ad[i] != 1 && bd[i] != 1 {
            return Err(OnnxError::InvalidModel("Add broadcast mismatch".into()));
        }
        od.push(ad[i].max(bd[i]));
    }
    let elems = checked_elements(&od)?;
    let astr = strides(&ad);
    let bstr = strides(&bd);
    let ostr = strides(&od);
    let mut out = Vec::with_capacity(elems);
    for flat in 0..elems {
        let mut rem = flat;
        let mut ai = 0usize;
        let mut bi = 0usize;
        for d in 0..rank {
            let coord = rem / ostr[d];
            rem %= ostr[d];
            if ad[d] != 1 {
                ai += coord * astr[d];
            }
            if bd[d] != 1 {
                bi += coord * bstr[d];
            }
        }
        out.push(f16::from_f32(a.values[ai].to_f32() + b.values[bi].to_f32()));
    }
    TensorF16::from_vec(od, out)
}

fn add_nchw_channel_bias(
    activation: &TensorF16,
    bias: &TensorF16,
) -> Result<Option<TensorF16>, OnnxError> {
    if activation.dims.len() != 4 {
        return Ok(None);
    }
    let n = activation.dims[0];
    let c = activation.dims[1];
    let h = activation.dims[2];
    let w = activation.dims[3];
    let bias_matches = bias.dims.as_slice() == [c, 1, 1] || bias.dims.as_slice() == [1, c, 1, 1];
    if !bias_matches || bias.values.len() != c {
        return Ok(None);
    }
    let spatial = h.checked_mul(w).ok_or(OnnxError::InvalidModel(
        "NCHW Add spatial size overflow".into(),
    ))?;
    let elems = n
        .checked_mul(c)
        .and_then(|v| v.checked_mul(spatial))
        .ok_or(OnnxError::InvalidModel("NCHW Add size overflow".into()))?;
    if activation.values.len() != elems {
        return Err(OnnxError::InvalidModel("NCHW Add activation size".into()));
    }
    let mut out = Vec::with_capacity(elems);
    for bn in 0..n {
        for ch in 0..c {
            let bv = bias.values[ch].to_f32();
            let base = (bn * c + ch) * spatial;
            out.extend(
                activation.values[base..base + spatial]
                    .iter()
                    .map(|&v| f16::from_f32(v.to_f32() + bv)),
            );
        }
    }
    Ok(Some(TensorF16::from_vec(activation.dims.clone(), out)?))
}
fn strides(dims: &[usize]) -> Vec<usize> {
    let mut s = vec![1; dims.len()];
    if dims.len() > 1 {
        for i in (0..dims.len() - 1).rev() {
            s[i] = s[i + 1] * dims[i + 1];
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn broadcast_bias_nchw() {
        let a = TensorF16::from_vec(
            vec![1, 2, 2, 2],
            (0..8).map(|x| f16::from_f32(x as f32)).collect(),
        )
        .unwrap();
        let b = TensorF16::from_vec(
            vec![2, 1, 1],
            vec![f16::from_f32(10.0), f16::from_f32(20.0)],
        )
        .unwrap();
        let o = add_broadcast(&a, &b).unwrap();
        assert_eq!(
            o.values.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
            vec![10., 11., 12., 13., 24., 25., 26., 27.]
        );
    }
    #[test]
    fn broadcast_bias_nchw_reversed_is_identical() {
        let a = TensorF16::from_vec(
            vec![1, 2, 2, 2],
            (0..8).map(|x| f16::from_f32(x as f32)).collect(),
        )
        .unwrap();
        let b = TensorF16::from_vec(
            vec![2, 1, 1],
            vec![f16::from_f32(10.0), f16::from_f32(20.0)],
        )
        .unwrap();
        assert_eq!(
            add_broadcast(&a, &b).unwrap(),
            add_broadcast(&b, &a).unwrap()
        );
    }
    #[test]
    fn add_same_shape_fast_path() {
        let a = TensorF16::from_vec(
            vec![1, 3],
            vec![f16::from_f32(1.0), f16::from_f32(-2.0), f16::from_f32(0.5)],
        )
        .unwrap();
        let b = TensorF16::from_vec(
            vec![1, 3],
            vec![f16::from_f32(3.0), f16::from_f32(4.0), f16::from_f32(0.25)],
        )
        .unwrap();
        let o = add_broadcast(&a, &b).unwrap();
        assert_eq!(
            o.values.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
            vec![4.0, 2.0, 0.75]
        );
    }

    #[test]
    fn pool2_stride2_odd_floor_geometry() {
        let a = TensorF16::from_vec(
            vec![1, 1, 3, 5],
            (0..15).map(|x| f16::from_f32(x as f32)).collect(),
        )
        .unwrap();
        let o = maxpool2d(&a, [2, 2], [2, 2], [0, 0, 0, 0], AutoPad::NotSet).unwrap();
        assert_eq!(o.dims, vec![1, 1, 1, 2]);
        assert_eq!(
            o.values.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
            vec![6., 8.]
        );
    }

    #[test]
    fn pool2() {
        let a = TensorF16::from_vec(
            vec![1, 1, 4, 4],
            (0..16).map(|x| f16::from_f32(x as f32)).collect(),
        )
        .unwrap();
        let o = maxpool2d(&a, [2, 2], [2, 2], [0, 0, 0, 0], AutoPad::NotSet).unwrap();
        assert_eq!(o.dims, vec![1, 1, 2, 2]);
        assert_eq!(
            o.values.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
            vec![5., 7., 13., 15.]
        );
    }
}
