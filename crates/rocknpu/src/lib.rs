use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_conv::Fp16Conv2dExecutor;
pub use rocknpu_ir::{
    AutoPad, ConstantTensor, DType, F16Tensor, Graph, Node, TensorSpec as TensorInfo,
};
use rocknpu_onnx::{CnnOnnxModel, CnnRunStats, CnnTimingStats, TensorF16, import_graph};
use rocknpu_ops::{ExecutionTarget, SingleNpuBackend};
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

    pub fn load_npy(path: impl AsRef<Path>) -> Result<Self, Error> {
        let bytes = fs::read(path)?;
        Self::from_npy_bytes(&bytes)
    }

    pub fn save_npy(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        fs::write(path, self.to_npy_bytes()?)?;
        Ok(())
    }

    pub fn from_npy_bytes(bytes: &[u8]) -> Result<Self, Error> {
        const MAGIC: &[u8; 6] = b"\x93NUMPY";
        if bytes.len() < 10 || &bytes[..6] != MAGIC {
            return Err(Error::InvalidInput("invalid NumPy .npy magic".into()));
        }
        let major = bytes[6];
        let (header_start, header_len) = match major {
            1 => {
                let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
                (10usize, len)
            }
            2 | 3 => {
                if bytes.len() < 12 {
                    return Err(Error::InvalidInput("truncated NumPy .npy header".into()));
                }
                let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
                (12usize, len)
            }
            _ => {
                return Err(Error::InvalidInput(format!(
                    "unsupported NumPy .npy version {major}.{}",
                    bytes[7]
                )));
            }
        };
        let header_end = header_start
            .checked_add(header_len)
            .ok_or_else(|| Error::InvalidInput("NumPy .npy header length overflow".into()))?;
        if header_end > bytes.len() {
            return Err(Error::InvalidInput("truncated NumPy .npy header".into()));
        }
        let header = std::str::from_utf8(&bytes[header_start..header_end])
            .map_err(|_| Error::InvalidInput("NumPy .npy header is not UTF-8/ASCII".into()))?;
        let descr = npy_string_field(header, "descr")?;
        if npy_bool_field(header, "fortran_order")? {
            return Err(Error::InvalidInput(
                "Fortran-order NumPy arrays are not supported; use C-order".into(),
            ));
        }
        let shape = npy_shape_field(header)?;
        let little_endian = match descr.as_str() {
            "<f4" => true,
            ">f4" => false,
            "=f4" | "f4" if cfg!(target_endian = "little") => true,
            "=f4" | "f4" => false,
            _ => {
                return Err(Error::InvalidInput(format!(
                    "unsupported NumPy dtype {descr:?}; expected float32"
                )));
            }
        };
        let elements = shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));
        let data_len = elements
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| Error::InvalidInput("NumPy shape byte length overflow".into()))?;
        let data = &bytes[header_end..];
        if data.len() != data_len {
            return Err(Error::InvalidInput(format!(
                "NumPy payload length {} does not match shape {:?} ({data_len} bytes)",
                data.len(),
                shape
            )));
        }
        let values = data
            .chunks_exact(4)
            .map(|chunk| {
                let raw = [chunk[0], chunk[1], chunk[2], chunk[3]];
                if little_endian {
                    f32::from_le_bytes(raw)
                } else {
                    f32::from_be_bytes(raw)
                }
            })
            .collect();
        Self::from_f32(shape, values)
    }

    pub fn to_npy_bytes(&self) -> Result<Vec<u8>, Error> {
        let shape = if self.shape.len() == 1 {
            format!("({},)", self.shape[0])
        } else {
            format!(
                "({})",
                self.shape
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape}, }}");
        let (major, preamble, header) = encode_npy_header(&dict)?;
        let mut out = Vec::with_capacity(
            preamble + header.len() + self.values.len().saturating_mul(std::mem::size_of::<f32>()),
        );
        out.extend_from_slice(b"\x93NUMPY");
        out.extend_from_slice(&[major, 0]);
        match major {
            1 => out.extend_from_slice(&(header.len() as u16).to_le_bytes()),
            2 => out.extend_from_slice(&(header.len() as u32).to_le_bytes()),
            _ => unreachable!(),
        }
        out.extend_from_slice(header.as_bytes());
        for &value in &self.values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        Ok(out)
    }
}

fn npy_field_tail<'a>(header: &'a str, key: &str) -> Result<&'a str, Error> {
    let single = format!("'{key}'");
    let double = format!("\"{key}\"");
    let key_pos = header
        .find(&single)
        .or_else(|| header.find(&double))
        .ok_or_else(|| Error::InvalidInput(format!("NumPy header missing {key:?}")))?;
    let after_key = &header[key_pos + key.len() + 2..];
    let colon = after_key
        .find(':')
        .ok_or_else(|| Error::InvalidInput(format!("NumPy header malformed {key:?}")))?;
    Ok(after_key[colon + 1..].trim_start())
}

fn npy_string_field(header: &str, key: &str) -> Result<String, Error> {
    let tail = npy_field_tail(header, key)?;
    let quote = tail
        .as_bytes()
        .first()
        .copied()
        .filter(|q| *q == b'\'' || *q == b'"')
        .ok_or_else(|| Error::InvalidInput(format!("NumPy header {key:?} is not a string")))?;
    let rest = &tail[1..];
    let end = rest
        .find(quote as char)
        .ok_or_else(|| Error::InvalidInput(format!("NumPy header unterminated {key:?}")))?;
    Ok(rest[..end].to_string())
}

fn npy_bool_field(header: &str, key: &str) -> Result<bool, Error> {
    let tail = npy_field_tail(header, key)?;
    if tail.starts_with("False") {
        Ok(false)
    } else if tail.starts_with("True") {
        Ok(true)
    } else {
        Err(Error::InvalidInput(format!(
            "NumPy header {key:?} is not a boolean"
        )))
    }
}

fn npy_shape_field(header: &str) -> Result<Vec<usize>, Error> {
    let tail = npy_field_tail(header, "shape")?;
    let start = tail
        .find('(')
        .ok_or_else(|| Error::InvalidInput("NumPy header shape is not a tuple".into()))?;
    let end = tail[start + 1..]
        .find(')')
        .map(|v| v + start + 1)
        .ok_or_else(|| Error::InvalidInput("NumPy header shape tuple is unterminated".into()))?;
    let body = &tail[start + 1..end];
    let mut dims = Vec::new();
    for raw in body.split(',') {
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let dim = raw
            .parse::<usize>()
            .map_err(|_| Error::InvalidInput(format!("invalid NumPy shape dimension {raw:?}")))?;
        if dim == 0 {
            return Err(Error::InvalidInput(
                "zero-sized NumPy dimensions are not supported yet".into(),
            ));
        }
        dims.push(dim);
    }
    if dims.is_empty() {
        return Err(Error::InvalidInput(
            "scalar NumPy arrays are not supported yet".into(),
        ));
    }
    Ok(dims)
}

fn encode_npy_header(dict: &str) -> Result<(u8, usize, String), Error> {
    fn padded(dict: &str, preamble: usize) -> String {
        let unpadded = dict.len() + 1;
        let padding = (64 - ((preamble + unpadded) % 64)) % 64;
        let mut header = String::with_capacity(unpadded + padding);
        header.push_str(dict);
        header.extend(std::iter::repeat_n(' ', padding));
        header.push('\n');
        header
    }

    let v1 = padded(dict, 10);
    if u16::try_from(v1.len()).is_ok() {
        return Ok((1, 10, v1));
    }
    let v2 = padded(dict, 12);
    u32::try_from(v2.len()).map_err(|_| Error::InvalidInput("NumPy header is too large".into()))?;
    Ok((2, 12, v2))
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

enum Command {
    Run {
        input: Tensor,
        reply: mpsc::Sender<Result<(Tensor, ExecutionStats), String>>,
    },
    Shutdown,
}

#[derive(Debug, Clone)]
pub struct Executable {
    model: CnnOnnxModel,
    input: TensorInfo,
    output: TensorInfo,
    prepare_dims: Vec<usize>,
}

impl Executable {
    pub fn compile(graph: Graph) -> Result<Self, Error> {
        let input = graph.input().clone();
        let output = graph.output().clone();
        if input.dtype() != DType::F32 || output.dtype() != DType::F32 {
            return Err(Error::InvalidModel(
                "Executable currently requires F32 graph input/output".into(),
            ));
        }
        let prepare_dims = concrete_prepare_dims(&input)?;
        Ok(Self {
            model: CnnOnnxModel::from_graph(graph),
            input,
            output,
            prepare_dims,
        })
    }

    pub fn input(&self) -> &TensorInfo {
        &self.input
    }

    pub fn output(&self) -> &TensorInfo {
        &self.output
    }

    pub fn graph(&self) -> &Graph {
        self.model.graph()
    }
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
        let graph = import_graph(&bytes).map_err(|error| Error::InvalidModel(error.to_string()))?;
        Self::from_graph_with_options(graph, options)
    }

    pub fn from_graph(graph: Graph) -> Result<Self, Error> {
        Self::from_graph_with_options(graph, SessionOptions::default())
    }

    pub fn from_graph_with_options(graph: Graph, options: SessionOptions) -> Result<Self, Error> {
        Self::from_executable_with_options(Executable::compile(graph)?, options)
    }

    pub fn from_executable(executable: Executable) -> Result<Self, Error> {
        Self::from_executable_with_options(executable, SessionOptions::default())
    }

    pub fn from_executable_with_options(
        executable: Executable,
        options: SessionOptions,
    ) -> Result<Self, Error> {
        let Executable {
            model,
            input,
            output,
            prepare_dims,
        } = executable;
        let (tx, rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        let target = options.target;
        let device_path = options.device_path.clone();
        let worker = thread::Builder::new()
            .name("rocknpu-session".into())
            .spawn(move || match target {
                SessionTarget::Npu => npu_worker(model, prepare_dims, device_path, ready_tx, rx),
                SessionTarget::Cpu => cpu_worker(model, ready_tx, rx),
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
            input,
            output,
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
        if name != self.input.name() {
            return Err(Error::InvalidInput(format!(
                "model input is {:?}, got {name:?}",
                self.input.name()
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
            name: self.output.name().to_string(),
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
    model: CnnOnnxModel,
    prepare_dims: Vec<usize>,
    device_path: PathBuf,
    ready: mpsc::Sender<Result<PrepareStats, String>>,
    rx: mpsc::Receiver<Command>,
) {
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
    model: CnnOnnxModel,
    ready: mpsc::Sender<Result<PrepareStats, String>>,
    rx: mpsc::Receiver<Command>,
) {
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

fn concrete_prepare_dims(info: &TensorInfo) -> Result<Vec<usize>, Error> {
    let mut dims = Vec::with_capacity(info.shape().len());
    for (index, dim) in info.shape().iter().copied().enumerate() {
        match dim {
            Some(v) => dims.push(v),
            None if index == 0 => dims.push(1),
            None => {
                return Err(Error::InvalidModel(format!(
                    "{} has dynamic non-batch dimension {}; load-time NPU preparation requires static feature/spatial dimensions",
                    info.name(),
                    index
                )));
            }
        }
    }
    Ok(dims)
}

fn validate_shape(info: &TensorInfo, got: &[usize]) -> Result<(), Error> {
    if got.len() != info.shape().len() {
        return Err(Error::InvalidInput(format!(
            "{} expects rank {}, got shape {:?}",
            info.name(),
            info.shape().len(),
            got
        )));
    }
    for (index, (&actual, expected)) in got.iter().zip(info.shape()).enumerate() {
        if actual == 0 {
            return Err(Error::InvalidInput(format!(
                "{} dimension {index} must be nonzero",
                info.name()
            )));
        }
        if let Some(expected) = expected
            && actual != *expected
        {
            return Err(Error::InvalidInput(format!(
                "{} dimension {index} must be {expected}, got {actual}",
                info.name()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_protobuf::{
        GraphProto, Message, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
        TensorShapeProto, TypeProto, ValueInfoProto, tensor_proto, tensor_shape_proto, type_proto,
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
    fn npy_round_trip_preserves_tensor() {
        let tensor = Tensor::from_f32(vec![2, 3], vec![0.0, 1.25, -2.5, 3.0, 4.5, -6.75]).unwrap();
        let bytes = tensor.to_npy_bytes().unwrap();
        assert_eq!(&bytes[..6], b"\x93NUMPY");
        assert_eq!(
            (10 + u16::from_le_bytes([bytes[8], bytes[9]]) as usize) % 64,
            0
        );
        assert_eq!(Tensor::from_npy_bytes(&bytes).unwrap(), tensor);
    }

    #[test]
    fn npy_rejects_fortran_order_and_non_f32() {
        let tensor = Tensor::from_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut fortran = tensor.to_npy_bytes().unwrap();
        let false_pos = fortran
            .windows(5)
            .position(|window| window == b"False")
            .unwrap();
        fortran[false_pos..false_pos + 5].copy_from_slice(b"True ");
        assert!(matches!(
            Tensor::from_npy_bytes(&fortran),
            Err(Error::InvalidInput(message)) if message.contains("Fortran-order")
        ));

        let mut f64_header = tensor.to_npy_bytes().unwrap();
        let dtype_pos = f64_header
            .windows(3)
            .position(|window| window == b"<f4")
            .unwrap();
        f64_header[dtype_pos..dtype_pos + 3].copy_from_slice(b"<f8");
        assert!(matches!(
            Tensor::from_npy_bytes(&f64_header),
            Err(Error::InvalidInput(message)) if message.contains("float32")
        ));
    }

    #[test]
    fn npy_reads_big_endian_f32() {
        let tensor = Tensor::from_f32(vec![2], vec![1.5, -9.25]).unwrap();
        let mut bytes = tensor.to_npy_bytes().unwrap();
        let dtype_pos = bytes
            .windows(3)
            .position(|window| window == b"<f4")
            .unwrap();
        bytes[dtype_pos..dtype_pos + 3].copy_from_slice(b">f4");
        let header_end = 10 + u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        for chunk in bytes[header_end..].chunks_exact_mut(4) {
            chunk.reverse();
        }
        assert_eq!(Tensor::from_npy_bytes(&bytes).unwrap(), tensor);
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
    fn frontend_graph_runs_without_model_bytes() {
        let bytes = fixture();
        let graph = import_graph(&bytes).unwrap();
        drop(bytes);
        let executable = Executable::compile(graph).unwrap();
        assert_eq!(executable.input().name(), "input");
        assert_eq!(executable.output().name(), "output");
        assert_eq!(executable.graph().nodes().len(), 2);
        let session =
            Session::from_executable_with_options(executable, SessionOptions::cpu()).unwrap();
        assert_eq!(session.input().name(), "input");
        assert_eq!(session.output().name(), "output");
        let input = Tensor::from_f32(
            vec![4, 32],
            (0..128).map(|i| (i % 7) as f32 * 0.25).collect(),
        )
        .unwrap();
        let out = session.run(input).unwrap();
        assert_eq!(out.tensor().shape(), &[4, 16]);
        assert_eq!(out.stats().matmul_nodes, 1);
        assert_eq!(out.stats().add_nodes, 1);
    }

    #[test]
    fn session_validates_static_input_shape_before_dispatch() {
        let session = Session::from_bytes_with_options(fixture(), SessionOptions::cpu()).unwrap();
        let input = Tensor::from_f32(vec![1, 32], vec![0.0; 32]).unwrap();
        assert!(matches!(session.run(input), Err(Error::InvalidInput(_))));
    }
}
