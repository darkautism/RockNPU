use half::f16;
use onnx_protobuf::{
    GraphProto, Message, ModelProto, NodeProto, OperatorSetIdProto, TensorProto, TensorShapeProto,
    TypeProto, ValueInfoProto, tensor_proto, tensor_shape_proto, type_proto,
};
use protobuf::MessageField;
use rocket_runtime::RocketDevice;
use rocknpu_onnx::TinyOnnxModel;
use rocknpu_ops::{ExecutionTarget, SingleNpuBackend};
use rocknpu_tensor::Matrix;
use std::{fs, path::PathBuf};
fn artifacts() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts")
}
fn vi(name: &str, dims: &[i64]) -> ValueInfoProto {
    let mut sh = TensorShapeProto::new();
    for &d in dims {
        let mut x = tensor_shape_proto::Dimension::new();
        x.set_dim_value(d);
        sh.dim.push(x);
    }
    let mut tt = type_proto::Tensor::new();
    tt.elem_type = tensor_proto::DataType::FLOAT as i32;
    tt.shape = MessageField::some(sh);
    let mut ty = TypeProto::new();
    ty.set_tensor_type(tt);
    let mut v = ValueInfoProto::new();
    v.name = name.into();
    v.type_ = MessageField::some(ty);
    v
}
fn init(name: &str, dims: &[i64], vals: Vec<f32>) -> TensorProto {
    let mut t = TensorProto::new();
    t.name = name.into();
    t.dims = dims.to_vec();
    t.data_type = tensor_proto::DataType::FLOAT as i32;
    t.float_data = vals;
    t
}
fn node(op: &str, ins: &[&str], out: &str) -> NodeProto {
    let mut n = NodeProto::new();
    n.op_type = op.into();
    n.input = ins.iter().map(|s| s.to_string()).collect();
    n.output = vec![out.into()];
    n
}
fn fixture() -> Vec<u8> {
    let w1 = (0..1024).map(|i| ((i * 7 + 3) % 5) as f32 - 2.0).collect();
    let b1 = (0..32).map(|i| (i % 3) as f32 - 1.0).collect();
    let w2 = (0..512).map(|i| ((i * 5 + 1) % 5) as f32 - 2.0).collect();
    let b2 = (0..16).map(|i| (i % 3) as f32 - 1.0).collect();
    let mut g = GraphProto::new();
    g.name = "rocknpu_tiny_mlp".into();
    g.input.push(vi("input", &[4, 32]));
    g.output.push(vi("output", &[4, 16]));
    g.initializer.push(init("w1", &[32, 32], w1));
    g.initializer.push(init("b1", &[32], b1));
    g.initializer.push(init("w2", &[32, 16], w2));
    g.initializer.push(init("b2", &[16], b2));
    g.node.push(node("MatMul", &["input", "w1"], "mm1"));
    g.node.push(node("Add", &["mm1", "b1"], "biased1"));
    g.node.push(node("Relu", &["biased1"], "hidden"));
    g.node.push(node("MatMul", &["hidden", "w2"], "mm2"));
    g.node.push(node("Add", &["mm2", "b2"], "output"));
    let mut os = OperatorSetIdProto::new();
    os.version = 13;
    let mut m = ModelProto::new();
    m.ir_version = 9;
    m.producer_name = "rocknpu-smoke".into();
    m.opset_import.push(os);
    m.graph = MessageField::some(g);
    m.write_to_bytes().unwrap()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(artifacts())?;
    let p = artifacts().join("tiny-mlp.onnx");
    fs::write(&p, fixture())?;
    let bytes = fs::read(&p)?;
    let model = TinyOnnxModel::from_bytes(&bytes)?;
    let input = Matrix::from_vec(
        4,
        32,
        (0..128)
            .map(|i| f16::from_f32(((i * 11 + 2) % 5) as f32 - 2.0))
            .collect(),
    )?;
    let (cpu, cs) = model.run_fp16(&input, ExecutionTarget::Cpu, None)?;
    let dev = RocketDevice::open()?;
    let mut npu = SingleNpuBackend::new(&dev)?;
    let (hw, hs, trace) =
        model.run_fp16_traced(&input, ExecutionTarget::NpuSingle, Some(&mut npu))?;
    let mm = cpu
        .values()
        .iter()
        .zip(hw.values())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    if mm != 0 {
        return Err(format!("CPU/NPU mismatch={mm}").into());
    }
    if cs.matmul_nodes != 2 || hs.matmul_nodes != 2 || hs.add_nodes != 2 || hs.relu_nodes != 1 {
        return Err("unexpected node stats".into());
    }
    let cb: Vec<u8> = cpu
        .values()
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect();
    let hb: Vec<u8> = hw
        .values()
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect();
    fs::write(artifacts().join("tiny-mlp-cpu-f16.bin"), cb)?;
    fs::write(artifacts().join("tiny-mlp-npu-f16.bin"), hb)?;
    let input_bytes: Vec<u8> = input
        .values()
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect();
    fs::write(artifacts().join("tiny-mlp-input-f16.bin"), input_bytes)?;
    let mut trace_text = String::new();
    for t in &trace {
        let bits = t
            .values
            .iter()
            .map(|v| format!("{:04x}", v.to_bits()))
            .collect::<Vec<_>>()
            .join(",");
        trace_text.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\n",
            t.op, t.name, t.rows, t.cols, bits
        ));
    }
    fs::write(artifacts().join("tiny-mlp-npu-trace.tsv"), trace_text)?;
    let summary = format!(
        "tiny ONNX MLP PASS file_bytes={} nodes={} input=4x32 hidden=4x32 output=4x16 matmul_npu=2 add_cpu=2 relu_cpu=1 mismatches=0 first8={:?}",
        bytes.len(),
        model.node_count(),
        &hw.values()[..8]
    );
    println!("{summary}");
    fs::write(artifacts().join("tiny-mlp-run.txt"), summary + "\n")?;
    Ok(())
}
