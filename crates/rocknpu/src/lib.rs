use half::f16;
use onnx_protobuf::{
    Message, ModelProto, ValueInfoProto, tensor_proto, tensor_shape_proto, type_proto,
};
use rocket_runtime::RocketDevice;
use rocknpu_conv::Fp16Conv2dExecutor;
use rocknpu_onnx::{CnnOnnxModel, CnnRunStats, CnnTimingStats, TensorF16};
use rocknpu_ops::{ExecutionTarget, SingleNpuBackend};
use std::collections::HashSet;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::Instant;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    InvalidModel(String),
    InvalidInput(String),
    Runtime(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => e.fmt(f),
            Self::InvalidModel(s) => write!(f, "invalid model: {s}"),
            Self::InvalidInput(s) => write!(f, "invalid input: {s}"),
            Self::Runtime(s) => write!(f, "RockNPU runtime error: {s}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    values: Vec<f32>,
}

impl Tensor {
    pub fn from_f32(shape: Vec<usize>, values: Vec<f32>) -> Result<Self, Error> {
        if shape.is_empty() || shape.contains(&0) {
            return Err(Error::InvalidInput(
                "tensor shape must contain nonzero dimensions".into(),
            ));
        }
        let expected = shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));
        if expected != Some(values.len()) {
            return Err(Error::InvalidInput(format!(
                "tensor value count {} does not match shape {:?}",
                values.len(),
                shape
            )));
        }
        Ok(Self { shape, values })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub const fn dtype(&self) -> DType {
        DType::F32
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    name: String,
    dtype: DType,
    shape: Vec<Option<usize>>,
}

impl TensorInfo {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &[Option<usize>] {
        &self.shape
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionTarget {
    Npu,
    Cpu,
}

#[derive(Debug, Clone)]
pub struct SessionOptions {
    target: SessionTarget,
    device_path: PathBuf,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            target: SessionTarget::Npu,
            device_path: PathBuf::from("/dev/accel/accel0"),
        }
    }
}

impl SessionOptions {
    pub fn cpu() -> Self {
        Self {
            target: SessionTarget::Cpu,
            ..Self::default()
        }
    }

    pub fn npu() -> Self {
        Self::default()
    }

    pub fn with_device_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.device_path = path.into();
        self
    }

    pub const fn target(&self) -> SessionTarget {
        self.target
    }

    pub fn device_path(&self) -> &Path {
        &self.device_path
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrepareStats {
    pub prepare_ns: u128,
    pub conv_weight_tensors: usize,
    pub dense_weight_tensors: usize,
    pub resident_weight_bytes: usize,
    pub weight_pack_ns: u128,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExecutionStats {
    pub total_ns: u128,
    pub conv_ns: u128,
    pub maxpool_ns: u128,
    pub reshape_ns: u128,
    pub matmul_ns: u128,
    pub gemm_ns: u128,
    pub add_ns: u128,
    pub relu_ns: u128,
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

impl ExecutionStats {
    fn from_cnn(run: CnnRunStats, timing: CnnTimingStats) -> Self {
        Self {
            total_ns: timing.total_ns,
            conv_ns: timing.conv_ns,
            maxpool_ns: timing.maxpool_ns,
            reshape_ns: timing.reshape_ns,
            matmul_ns: timing.matmul_ns,
            gemm_ns: timing.gemm_ns,
            add_ns: timing.add_ns,
            relu_ns: timing.relu_ns,
            conv_nodes: run.conv_nodes,
            maxpool_nodes: run.maxpool_nodes,
            reshape_nodes: run.reshape_nodes,
            matmul_nodes: run.matmul_nodes,
            gemm_nodes: run.gemm_nodes,
            add_nodes: run.add_nodes,
            relu_nodes: run.relu_nodes,
            npu_dense_nodes: run.npu_dense_nodes,
            npu_conv_nodes: run.npu_conv_nodes,
            padded_npu_dense_nodes: run.padded_npu_dense_nodes,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunOutput {
    name: String,
    tensor: Tensor,
    stats: ExecutionStats,
}

impl RunOutput {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    pub const fn stats(&self) -> ExecutionStats {
        self.stats
    }

    pub fn into_tensor(self) -> Tensor {
        self.tensor
    }
}

struct ModelMetadata {
    input: TensorInfo,
    output: TensorInfo,
}

enum Command {
    Run {
        input: Tensor,
        reply: mpsc::Sender<Result<(Tensor, ExecutionStats), String>>,
    },
    Shutdown,
}

pub struct Session {
    input: TensorInfo,
    output: TensorInfo,
    prepare_stats: PrepareStats,
    tx: mpsc::Sender<Command>,
    worker: Option<JoinHandle<()>>,
}

impl Session {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        Self::load_with_options(path, SessionOptions::default())
    }

    pub fn load_with_options(
        path: impl AsRef<Path>,
        options: SessionOptions,
    ) -> Result<Self, Error> {
        let bytes = fs::read(path)?;
        Self::from_bytes_with_options(bytes, options)
    }

    fn from_bytes_with_options(bytes: Vec<u8>, options: SessionOptions) -> Result<Self, Error> {
        let metadata = parse_metadata(&bytes)?;
        let prepare_dims = concrete_prepare_dims(&metadata.input)?;
        let (tx, rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        let target = options.target;
        let device_path = options.device_path.clone();
        let worker = thread::Builder::new()
            .name("rocknpu-session".into())
            .spawn(move || match target {
                SessionTarget::Npu => npu_worker(bytes, prepare_dims, device_path, ready_tx, rx),
                SessionTarget::Cpu => cpu_worker(bytes, ready_tx, rx),
            })
            .map_err(Error::Io)?;

        let prepare_stats = match ready_rx.recv() {
            Ok(Ok(stats)) => stats,
            Ok(Err(e)) => {
                let _ = worker.join();
                return Err(Error::Runtime(e));
            }
            Err(_) => {
                let _ = worker.join();
                return Err(Error::Runtime(
                    "session worker exited during initialization".into(),
                ));
            }
        };

        Ok(Self {
            input: metadata.input,
            output: metadata.output,
            prepare_stats,
            tx,
            worker: Some(worker),
        })
    }

    pub fn input(&self) -> &TensorInfo {
        &self.input
    }

    pub fn output(&self) -> &TensorInfo {
        &self.output
    }

    /// Session creation is eager: this returns the already-completed preparation
    /// summary and does not reparse or repack the model.
    pub const fn prepare(&self) -> PrepareStats {
        self.prepare_stats
    }

    pub const fn prepare_stats(&self) -> PrepareStats {
        self.prepare_stats
    }

    pub fn run(&self, input: Tensor) -> Result<RunOutput, Error> {
        self.run_named(self.input.name(), input)
    }

    pub fn run_named(&self, name: &str, input: Tensor) -> Result<RunOutput, Error> {
        if name != self.input.name {
            return Err(Error::InvalidInput(format!(
                "model input is {:?}, got {name:?}",
                self.input.name
            )));
        }
        validate_shape(&self.input, input.shape())?;
        let (reply_tx, reply_rx) = mpsc::channel();
        self.tx
            .send(Command::Run {
                input,
                reply: reply_tx,
            })
            .map_err(|_| Error::Runtime("session worker is not running".into()))?;
        let (tensor, stats) = reply_rx
            .recv()
            .map_err(|_| Error::Runtime("session worker exited during run".into()))?
            .map_err(Error::Runtime)?;
        Ok(RunOutput {
            name: self.output.name.clone(),
            tensor,
            stats,
        })
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.tx.send(Command::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn npu_worker(
    bytes: Vec<u8>,
    prepare_dims: Vec<usize>,
    device_path: PathBuf,
    ready: mpsc::Sender<Result<PrepareStats, String>>,
    rx: mpsc::Receiver<Command>,
) {
    let model = match CnnOnnxModel::from_bytes(&bytes) {
        Ok(v) => v,
        Err(e) => {
            let _ = ready.send(Err(e.to_string()));
            return;
        }
    };
    let device = match RocketDevice::open_path(device_path.to_string_lossy().as_ref()) {
        Ok(v) => v,
        Err(e) => {
            let _ = ready.send(Err(format!("open {}: {e}", device_path.display())));
            return;
        }
    };
    let mut dense = match SingleNpuBackend::new(&device) {
        Ok(v) => v,
        Err(e) => {
            let _ = ready.send(Err(e.to_string()));
            return;
        }
    };
    let mut conv = match Fp16Conv2dExecutor::new(&device) {
        Ok(v) => v,
        Err(e) => {
            let _ = ready.send(Err(e.to_string()));
            return;
        }
    };
    let prepare_started = Instant::now();
    let mut prepared = match model.prepare_npu_weights(&prepare_dims, &dense, &conv) {
        Ok(v) => v,
        Err(e) => {
            let _ = ready.send(Err(e.to_string()));
            return;
        }
    };
    let conv_stats = prepared.stats();
    let dense_stats = prepared.dense_stats();
    let stats = PrepareStats {
        prepare_ns: prepare_started.elapsed().as_nanos(),
        conv_weight_tensors: conv_stats.tensors,
        dense_weight_tensors: dense_stats.tensors,
        resident_weight_bytes: conv_stats.resident_bytes + dense_stats.resident_bytes,
        weight_pack_ns: conv_stats.pack_ns + dense_stats.pack_ns,
    };
    if ready.send(Ok(stats)).is_err() {
        return;
    }

    while let Ok(command) = rx.recv() {
        match command {
            Command::Shutdown => break,
            Command::Run { input, reply } => {
                let result = (|| {
                    let input = TensorF16::from_vec(
                        input.shape.clone(),
                        input.values.iter().copied().map(f16::from_f32).collect(),
                    )
                    .map_err(|e| e.to_string())?;
                    let (output, run, timing) = model
                        .run_fp16_profiled_with_prepared_conv(
                            &input,
                            ExecutionTarget::NpuSingle,
                            Some(&mut dense),
                            Some(&mut conv),
                            Some(&mut prepared),
                        )
                        .map_err(|e| e.to_string())?;
                    let tensor = Tensor::from_f32(
                        output.dims().to_vec(),
                        output.values().iter().map(|v| v.to_f32()).collect(),
                    )
                    .map_err(|e| e.to_string())?;
                    Ok((tensor, ExecutionStats::from_cnn(run, timing)))
                })();
                let _ = reply.send(result);
            }
        }
    }
}

fn cpu_worker(
    bytes: Vec<u8>,
    ready: mpsc::Sender<Result<PrepareStats, String>>,
    rx: mpsc::Receiver<Command>,
) {
    let model = match CnnOnnxModel::from_bytes(&bytes) {
        Ok(v) => v,
        Err(e) => {
            let _ = ready.send(Err(e.to_string()));
            return;
        }
    };
    if ready.send(Ok(PrepareStats::default())).is_err() {
        return;
    }
    while let Ok(command) = rx.recv() {
        match command {
            Command::Shutdown => break,
            Command::Run { input, reply } => {
                let result = (|| {
                    let input = TensorF16::from_vec(
                        input.shape.clone(),
                        input.values.iter().copied().map(f16::from_f32).collect(),
                    )
                    .map_err(|e| e.to_string())?;
                    let started = Instant::now();
                    let (output, run) = model
                        .run_fp16(&input, ExecutionTarget::Cpu, None)
                        .map_err(|e| e.to_string())?;
                    let timing = CnnTimingStats {
                        total_ns: started.elapsed().as_nanos(),
                        ..Default::default()
                    };
                    let tensor = Tensor::from_f32(
                        output.dims().to_vec(),
                        output.values().iter().map(|v| v.to_f32()).collect(),
                    )
                    .map_err(|e| e.to_string())?;
                    Ok((tensor, ExecutionStats::from_cnn(run, timing)))
                })();
                let _ = reply.send(result);
            }
        }
    }
}

fn parse_metadata(bytes: &[u8]) -> Result<ModelMetadata, Error> {
    let model = ModelProto::parse_from_bytes(bytes)
        .map_err(|e| Error::InvalidModel(format!("ONNX protobuf: {e}")))?;
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| Error::InvalidModel("ONNX model has no graph".into()))?;
    let initializer_names: HashSet<&str> =
        graph.initializer.iter().map(|v| v.name.as_str()).collect();
    let inputs: Vec<_> = graph
        .input
        .iter()
        .filter(|v| !initializer_names.contains(v.name.as_str()))
        .collect();
    if inputs.len() != 1 {
        return Err(Error::InvalidModel(format!(
            "Session currently requires exactly one runtime input, got {}",
            inputs.len()
        )));
    }
    if graph.output.len() != 1 {
        return Err(Error::InvalidModel(format!(
            "Session currently requires exactly one output, got {}",
            graph.output.len()
        )));
    }
    Ok(ModelMetadata {
        input: parse_tensor_info(inputs[0])?,
        output: parse_tensor_info(&graph.output[0])?,
    })
}

fn parse_tensor_info(value: &ValueInfoProto) -> Result<TensorInfo, Error> {
    if value.name.is_empty() {
        return Err(Error::InvalidModel("tensor value has no name".into()));
    }
    let ty = value
        .type_
        .as_ref()
        .ok_or_else(|| Error::InvalidModel(format!("{} has no type", value.name)))?;
    let tensor = match ty.value.as_ref() {
        Some(type_proto::Value::TensorType(v)) => v,
        _ => {
            return Err(Error::InvalidModel(format!(
                "{} is not a tensor value",
                value.name
            )));
        }
    };
    if tensor.elem_type != tensor_proto::DataType::FLOAT as i32 {
        return Err(Error::InvalidModel(format!(
            "{} currently requires ONNX FLOAT input/output metadata, dtype={}",
            value.name, tensor.elem_type
        )));
    }
    let shape = tensor
        .shape
        .as_ref()
        .ok_or_else(|| Error::InvalidModel(format!("{} has no shape", value.name)))?;
    if shape.dim.is_empty() {
        return Err(Error::InvalidModel(format!(
            "{} scalar tensors are not supported yet",
            value.name
        )));
    }
    let mut dims = Vec::with_capacity(shape.dim.len());
    for dim in &shape.dim {
        match dim.value.as_ref() {
            Some(tensor_shape_proto::dimension::Value::DimValue(v)) if *v > 0 => {
                dims.push(Some(*v as usize));
            }
            Some(tensor_shape_proto::dimension::Value::DimValue(v)) => {
                return Err(Error::InvalidModel(format!(
                    "{} has nonpositive dimension {v}",
                    value.name
                )));
            }
            Some(tensor_shape_proto::dimension::Value::DimParam(_)) | None => dims.push(None),
            Some(_) => {
                return Err(Error::InvalidModel(format!(
                    "{} uses an unsupported dimension metadata variant",
                    value.name
                )));
            }
        }
    }
    Ok(TensorInfo {
        name: value.name.clone(),
        dtype: DType::F32,
        shape: dims,
    })
}

fn concrete_prepare_dims(info: &TensorInfo) -> Result<Vec<usize>, Error> {
    let mut dims = Vec::with_capacity(info.shape.len());
    for (index, dim) in info.shape.iter().copied().enumerate() {
        match dim {
            Some(v) => dims.push(v),
            None if index == 0 => dims.push(1),
            None => {
                return Err(Error::InvalidModel(format!(
                    "{} has dynamic non-batch dimension {}; load-time NPU preparation requires static feature/spatial dimensions",
                    info.name, index
                )));
            }
        }
    }
    Ok(dims)
}

fn validate_shape(info: &TensorInfo, got: &[usize]) -> Result<(), Error> {
    if got.len() != info.shape.len() {
        return Err(Error::InvalidInput(format!(
            "{} expects rank {}, got shape {:?}",
            info.name,
            info.shape.len(),
            got
        )));
    }
    for (index, (&actual, expected)) in got.iter().zip(&info.shape).enumerate() {
        if actual == 0 {
            return Err(Error::InvalidInput(format!(
                "{} dimension {index} must be nonzero",
                info.name
            )));
        }
        if let Some(expected) = expected
            && actual != *expected
        {
            return Err(Error::InvalidInput(format!(
                "{} dimension {index} must be {expected}, got {actual}",
                info.name
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_protobuf::{
        GraphProto, NodeProto, OperatorSetIdProto, TensorProto, TensorShapeProto, TypeProto,
        tensor_shape_proto,
    };
    use protobuf::MessageField;

    fn value_info(name: &str, dims: &[i64]) -> ValueInfoProto {
        let mut shape = TensorShapeProto::new();
        for &dim in dims {
            let mut d = tensor_shape_proto::Dimension::new();
            d.set_dim_value(dim);
            shape.dim.push(d);
        }
        let mut tensor = type_proto::Tensor::new();
        tensor.elem_type = tensor_proto::DataType::FLOAT as i32;
        tensor.shape = MessageField::some(shape);
        let mut ty = TypeProto::new();
        ty.set_tensor_type(tensor);
        let mut value = ValueInfoProto::new();
        value.name = name.into();
        value.type_ = MessageField::some(ty);
        value
    }

    fn initializer(name: &str, dims: &[i64], values: Vec<f32>) -> TensorProto {
        let mut tensor = TensorProto::new();
        tensor.name = name.into();
        tensor.dims = dims.to_vec();
        tensor.data_type = tensor_proto::DataType::FLOAT as i32;
        tensor.float_data = values;
        tensor
    }

    fn node(op: &str, inputs: &[&str], output: &str) -> NodeProto {
        let mut node = NodeProto::new();
        node.op_type = op.into();
        node.input = inputs.iter().map(|v| v.to_string()).collect();
        node.output = vec![output.into()];
        node
    }

    fn fixture() -> Vec<u8> {
        let mut graph = GraphProto::new();
        graph.input.push(value_info("input", &[4, 32]));
        graph.output.push(value_info("output", &[4, 16]));
        graph.initializer.push(initializer(
            "weight",
            &[32, 16],
            (0..512).map(|i| ((i % 5) as f32 - 2.0) * 0.125).collect(),
        ));
        graph
            .initializer
            .push(initializer("bias", &[16], vec![0.25; 16]));
        graph.node.push(node("MatMul", &["input", "weight"], "mm"));
        graph.node.push(node("Add", &["mm", "bias"], "output"));
        let mut opset = OperatorSetIdProto::new();
        opset.version = 13;
        let mut model = ModelProto::new();
        model.ir_version = 9;
        model.opset_import.push(opset);
        model.graph = MessageField::some(graph);
        model.write_to_bytes().unwrap()
    }

    #[test]
    fn tensor_rejects_bad_shape() {
        assert!(Tensor::from_f32(vec![2, 2], vec![0.0; 3]).is_err());
        assert!(Tensor::from_f32(vec![2, 0], Vec::new()).is_err());
    }

    #[test]
    fn cpu_session_loads_once_and_runs_named_tensor() {
        let session = Session::from_bytes_with_options(fixture(), SessionOptions::cpu()).unwrap();
        assert_eq!(session.input().name(), "input");
        assert_eq!(session.input().shape(), &[Some(4), Some(32)]);
        assert_eq!(session.output().name(), "output");
        let input = Tensor::from_f32(
            vec![4, 32],
            (0..128).map(|i| (i % 7) as f32 * 0.25).collect(),
        )
        .unwrap();
        let out = session.run(input).unwrap();
        assert_eq!(out.name(), "output");
        assert_eq!(out.tensor().shape(), &[4, 16]);
        assert_eq!(out.stats().matmul_nodes, 1);
        assert_eq!(out.stats().add_nodes, 1);
        assert_eq!(out.stats().npu_dense_nodes, 0);
    }

    #[test]
    fn session_validates_static_input_shape_before_dispatch() {
        let session = Session::from_bytes_with_options(fixture(), SessionOptions::cpu()).unwrap();
        let input = Tensor::from_f32(vec![1, 32], vec![0.0; 32]).unwrap();
        assert!(matches!(session.run(input), Err(Error::InvalidInput(_))));
    }
}
