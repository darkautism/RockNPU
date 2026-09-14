use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_onnx::{TinyOnnxModel, TraceTensor};
use rocknpu_ops::{ExecutionTarget, SingleNpuBackend};
use rocknpu_tensor::Matrix;
use std::{fs, path::PathBuf};

const M: usize = 50;
const K: usize = 784;
const N: usize = 10;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read_f32(path: &PathBuf) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{} has non-f32 byte length", path.display()).into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    for i in 1..row.len() {
        if row[i] > row[best] {
            best = i;
        }
    }
    best
}

fn save_trace(trace: &[TraceTensor]) -> Result<(), Box<dyn std::error::Error>> {
    let art = root().join("artifacts");
    let mut meta = String::new();
    let mut raw = Vec::new();
    let mut offset = 0usize;
    for t in trace {
        let len = t.values.len();
        meta.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\n",
            t.op, t.name, t.rows, t.cols, offset, len
        ));
        for v in &t.values {
            raw.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        offset += len;
    }
    fs::write(art.join("real-mnist-npu-trace.tsv"), meta)?;
    fs::write(art.join("real-mnist-npu-trace-f16.bin"), raw)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let art = root().join("artifacts");
    let model_bytes = fs::read(art.join("external/pico-cnn-mnist-mlp.onnx"))?;
    let model = TinyOnnxModel::from_bytes(&model_bytes)?;

    let input_f32 = read_f32(&art.join("external/mnist-first50-f32.bin"))?;
    if input_f32.len() != M * K {
        return Err(format!("input count {} != {}", input_f32.len(), M * K).into());
    }
    let input = Matrix::from_vec(M, K, input_f32.iter().copied().map(f16::from_f32).collect())?;

    let reference = read_f32(&art.join("external/mnist-first50-onnx-ref-f32.bin"))?;
    if reference.len() != M * N {
        return Err(format!("reference count {} != {}", reference.len(), M * N).into());
    }
    let labels = fs::read(art.join("external/mnist-first50-labels-u8.bin"))?;
    if labels.len() != M {
        return Err(format!("label count {} != {M}", labels.len()).into());
    }

    let (cpu, cpu_stats) = model.run_fp16(&input, ExecutionTarget::Cpu, None)?;
    let dev = RocketDevice::open()?;
    let mut npu_backend = SingleNpuBackend::new(&dev)?;
    let (npu, npu_stats, trace) =
        model.run_fp16_traced(&input, ExecutionTarget::NpuSingle, Some(&mut npu_backend))?;

    if cpu.rows() != M || cpu.cols() != N || npu.rows() != M || npu.cols() != N {
        return Err("unexpected real-model output shape".into());
    }
    if npu_stats.gemm_nodes != 4
        || npu_stats.relu_nodes != 3
        || npu_stats.npu_dense_nodes != 4
        || npu_stats.padded_npu_dense_nodes != 4
    {
        return Err(format!("unexpected NPU stats: {npu_stats:?}").into());
    }
    if cpu_stats.gemm_nodes != 4 || cpu_stats.relu_nodes != 3 {
        return Err(format!("unexpected CPU stats: {cpu_stats:?}").into());
    }

    let cpu32: Vec<f32> = cpu.values().iter().map(|x| x.to_f32()).collect();
    let npu32: Vec<f32> = npu.values().iter().map(|x| x.to_f32()).collect();
    let mut cpu_npu_max = 0.0f32;
    let mut ref_npu_max = 0.0f32;
    let mut ref_npu_sum = 0.0f64;
    for i in 0..M * N {
        cpu_npu_max = cpu_npu_max.max((cpu32[i] - npu32[i]).abs());
        let e = (reference[i] - npu32[i]).abs();
        ref_npu_max = ref_npu_max.max(e);
        ref_npu_sum += e as f64;
    }

    let ref_pred: Vec<usize> = reference.chunks_exact(N).map(argmax).collect();
    let cpu_pred: Vec<usize> = cpu32.chunks_exact(N).map(argmax).collect();
    let npu_pred: Vec<usize> = npu32.chunks_exact(N).map(argmax).collect();
    let ref_npu_pred_match = ref_pred
        .iter()
        .zip(&npu_pred)
        .filter(|(a, b)| a == b)
        .count();
    let cpu_npu_pred_match = cpu_pred
        .iter()
        .zip(&npu_pred)
        .filter(|(a, b)| a == b)
        .count();
    let ref_correct = ref_pred
        .iter()
        .zip(&labels)
        .filter(|(p, y)| **p == **y as usize)
        .count();
    let npu_correct = npu_pred
        .iter()
        .zip(&labels)
        .filter(|(p, y)| **p == **y as usize)
        .count();

    let mut raw = Vec::with_capacity(M * N * 2);
    for v in npu.values() {
        raw.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    fs::write(art.join("real-mnist-npu-f16.bin"), raw)?;
    save_trace(&trace)?;

    println!(
        "real MNIST MLP RUN model_nodes={} gemm_npu={} padded_npu={} relu_cpu={} ref_pred_match={}/{} cpu_npu_pred_match={}/{} ref_accuracy={}/{} npu_accuracy={}/{} cpu_npu_max_abs={} ref_npu_max_abs={} ref_npu_mean_abs={:.6}",
        model.node_count(),
        npu_stats.npu_dense_nodes,
        npu_stats.padded_npu_dense_nodes,
        npu_stats.relu_nodes,
        ref_npu_pred_match,
        M,
        cpu_npu_pred_match,
        M,
        ref_correct,
        M,
        npu_correct,
        M,
        cpu_npu_max,
        ref_npu_max,
        ref_npu_sum / (M * N) as f64,
    );
    println!("labels={labels:?}");
    println!("reference_pred={ref_pred:?}");
    println!("npu_pred={npu_pred:?}");
    println!("reference_first_logits={:?}", &reference[..N]);
    println!("npu_first_logits={:?}", &npu32[..N]);

    if ref_npu_pred_match != M {
        return Err(format!(
            "NPU top-1 differs from ONNX reference on {} samples",
            M - ref_npu_pred_match
        )
        .into());
    }
    Ok(())
}
