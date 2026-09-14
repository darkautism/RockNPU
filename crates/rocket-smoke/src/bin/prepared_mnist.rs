use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_onnx::{TinyOnnxModel, TraceTensor};
use rocknpu_ops::{ExecutionTarget, SingleNpuBackend};
use rocknpu_tensor::{Matrix, MatrixShape};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

const M: usize = 50;
const K: usize = 784;
const N: usize = 10;
const REPS: usize = 9;

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

fn median(mut values: Vec<u128>) -> u128 {
    values.sort_unstable();
    values[values.len() / 2]
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

fn bits(v: &Matrix<f16>) -> Vec<u16> {
    v.values().iter().map(|x| x.to_bits()).collect()
}

fn save_trace(trace: &[TraceTensor], art: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
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
    fs::write(art.join("prepared-mnist-npu-trace.tsv"), meta)?;
    fs::write(art.join("prepared-mnist-npu-trace-f16.bin"), raw)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let art = root().join("artifacts");
    let model_bytes = fs::read(art.join("external/pico-cnn-mnist-mlp.onnx"))?;
    let model = TinyOnnxModel::from_bytes(&model_bytes)?;
    let input_f32 = read_f32(&art.join("external/mnist-first50-f32.bin"))?;
    let reference = read_f32(&art.join("external/mnist-first50-onnx-ref-f32.bin"))?;
    let labels = fs::read(art.join("external/mnist-first50-labels-u8.bin"))?;
    if input_f32.len() != M * K || reference.len() != M * N || labels.len() != M {
        return Err("prepared MNIST artifact shape mismatch".into());
    }
    let input = Matrix::from_vec(M, K, input_f32.iter().copied().map(f16::from_f32).collect())?;

    let dev = RocketDevice::open()?;
    let mut backend = SingleNpuBackend::new(&dev)?;

    // Warm the old streaming path first so every model shape has reusable scratch.
    let (stream_warm, stream_stats) =
        model.run_fp16(&input, ExecutionTarget::NpuSingle, Some(&mut backend))?;
    if stream_stats.npu_dense_nodes != 4 || stream_stats.padded_npu_dense_nodes != 4 {
        return Err(format!("unexpected streaming stats: {stream_stats:?}").into());
    }
    let stream_bits = bits(&stream_warm);
    let mut streaming_ns = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        let (out, _) = model.run_fp16(&input, ExecutionTarget::NpuSingle, Some(&mut backend))?;
        streaming_ns.push(t.elapsed().as_nanos());
        if bits(&out) != stream_bits {
            return Err("streaming repeated output changed".into());
        }
    }

    // Prepare static weights once. The session must remain usable after the parsed
    // model and its original initializer storage are dropped.
    let prepare_start = Instant::now();
    let session = model.prepare_npu(MatrixShape::new(M, K), &backend)?;
    let prepare_wall_ns = prepare_start.elapsed().as_nanos();
    let prepared_stats = session.stats();
    if prepared_stats.dense_nodes != 4 || prepared_stats.padded_dense_nodes != 4 {
        return Err(format!("unexpected prepared stats: {prepared_stats:?}").into());
    }
    drop(model);

    let (prepared_warm, prepared_run_stats, prepared_trace) =
        session.run_fp16_traced(&input, &mut backend)?;
    save_trace(&prepared_trace, &art)?;
    if prepared_run_stats.npu_dense_nodes != 4 || prepared_run_stats.padded_npu_dense_nodes != 4 {
        return Err(format!("unexpected prepared run stats: {prepared_run_stats:?}").into());
    }
    let prepared_bits = bits(&prepared_warm);
    if prepared_bits != stream_bits {
        let mismatches = prepared_bits
            .iter()
            .zip(&stream_bits)
            .filter(|(a, b)| a != b)
            .count();
        return Err(format!("prepared vs streaming mismatch={mismatches}").into());
    }

    let scratch_before = backend.executor_mut().scratch_stats();
    let mut prepared_ns = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        let (out, _) = session.run_fp16(&input, &mut backend)?;
        prepared_ns.push(t.elapsed().as_nanos());
        if bits(&out) != prepared_bits {
            return Err("prepared repeated output changed".into());
        }
    }
    let scratch_after = backend.executor_mut().scratch_stats();
    if scratch_after.bo_allocations != scratch_before.bo_allocations {
        return Err(format!(
            "prepared repeated inference allocated scratch: {} -> {}",
            scratch_before.bo_allocations, scratch_after.bo_allocations
        )
        .into());
    }

    let out32: Vec<f32> = prepared_warm.values().iter().map(|x| x.to_f32()).collect();
    let ref_pred: Vec<usize> = reference.chunks_exact(N).map(argmax).collect();
    let out_pred: Vec<usize> = out32.chunks_exact(N).map(argmax).collect();
    let pred_match = ref_pred
        .iter()
        .zip(&out_pred)
        .filter(|(a, b)| a == b)
        .count();
    let accuracy = out_pred
        .iter()
        .zip(&labels)
        .filter(|(p, y)| **p == **y as usize)
        .count();
    if pred_match != M {
        return Err(format!(
            "prepared top-1 differs from reference on {} samples",
            M - pred_match
        )
        .into());
    }

    let stream_med = median(streaming_ns);
    let prepared_med = median(prepared_ns);
    let summary = format!(
        "prepared MNIST PASS dense=4 padded=4 resident_MB={:.3} resident_tiles={} prepare_wall_ms={:.3} weight_pack_ms={:.3} streaming_median_ms={:.3} prepared_median_ms={:.3} speedup={:.2}x scratch_allocs={} scratch_grows={} top1_ref={}/{} accuracy={}/{} bit_identical_streaming=true model_dropped_before_prepared_runs=true",
        prepared_stats.resident_weight_bytes as f64 / 1e6,
        prepared_stats.resident_weight_tiles,
        prepare_wall_ns as f64 / 1e6,
        prepared_stats.weight_pack_ns as f64 / 1e6,
        stream_med as f64 / 1e6,
        prepared_med as f64 / 1e6,
        stream_med as f64 / prepared_med.max(1) as f64,
        scratch_after.bo_allocations,
        scratch_after.bo_grows,
        pred_match,
        M,
        accuracy,
        M,
    );
    println!("{summary}");
    fs::write(art.join("prepared-mnist-run.txt"), summary + "\n")?;
    Ok(())
}
