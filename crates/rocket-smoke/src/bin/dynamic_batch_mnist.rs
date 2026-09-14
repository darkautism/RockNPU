use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_onnx::{TinyOnnxModel, TraceTensor};
use rocknpu_ops::SingleNpuBackend;
use rocknpu_tensor::{Matrix, MatrixShape};
use std::fs;
use std::path::PathBuf;

const K: usize = 784;
const N: usize = 10;
const MAX_M: usize = 50;
const BATCHES: [usize; 4] = [1, 4, 16, 50];

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

fn input_matrix(all: &[f32], m: usize) -> Result<Matrix<f16>, Box<dyn std::error::Error>> {
    Ok(Matrix::from_vec(
        m,
        K,
        all[..m * K].iter().copied().map(f16::from_f32).collect(),
    )?)
}

fn save_trace(prefix: &str, trace: &[TraceTensor]) -> Result<(), Box<dyn std::error::Error>> {
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
    fs::write(art.join(format!("{prefix}-trace.tsv")), meta)?;
    fs::write(art.join(format!("{prefix}-trace-f16.bin")), raw)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let art = root().join("artifacts");
    let model_bytes = fs::read(art.join("external/pico-cnn-mnist-mlp.onnx"))?;
    let model = TinyOnnxModel::from_bytes(&model_bytes)?;
    let input_f32 = read_f32(&art.join("external/mnist-first50-f32.bin"))?;
    let reference = read_f32(&art.join("external/mnist-first50-onnx-ref-f32.bin"))?;
    if input_f32.len() != MAX_M * K || reference.len() != MAX_M * N {
        return Err("unexpected MNIST artifact size".into());
    }

    let dev = RocketDevice::open()?;
    let mut backend = SingleNpuBackend::new(&dev)?;
    let prepared = model.prepare_npu_dynamic_batch(MatrixShape::new(MAX_M, K), &backend)?;
    let ps = prepared.stats();
    if !prepared.is_dynamic_batch() || ps.dense_nodes != 4 || ps.m_compatible_dense_nodes != 4 {
        return Err(format!("unexpected dynamic prepared stats: {ps:?}").into());
    }
    drop(model);

    // Warm the largest batch first so every reusable scratch slot has already
    // reached the maximum capacity needed by the subsequent smaller batches.
    let warm_input = input_matrix(&input_f32, MAX_M)?;
    let (warm, warm_stats) = prepared.run_fp16(&warm_input, &mut backend)?;
    if warm_stats.npu_dense_nodes != 4 || warm.rows() != MAX_M || warm.cols() != N {
        return Err("dynamic warmup did not execute four dense NPU nodes".into());
    }
    let scratch_before = backend.executor_mut().scratch_stats();

    let mut lines = Vec::new();
    for m in BATCHES {
        let input = input_matrix(&input_f32, m)?;
        let (out, stats) = prepared.run_fp16(&input, &mut backend)?;
        if out.rows() != m || out.cols() != N || stats.npu_dense_nodes != 4 {
            return Err(format!("batch {m}: unexpected output/stats {stats:?}").into());
        }
        let got: Vec<f32> = out.values().iter().map(|v| v.to_f32()).collect();
        let ref_slice = &reference[..m * N];
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        let mut matches = 0usize;
        for r in 0..m {
            let gp = argmax(&got[r * N..(r + 1) * N]);
            let rp = argmax(&ref_slice[r * N..(r + 1) * N]);
            if gp == rp {
                matches += 1;
            }
            for c in 0..N {
                let e = (got[r * N + c] - ref_slice[r * N + c]).abs();
                max_abs = max_abs.max(e);
                sum_abs += e as f64;
            }
        }
        if matches != m {
            return Err(format!("batch {m}: top-1 reference match {matches}/{m}").into());
        }
        lines.push(format!(
            "batch={m} top1_ref={matches}/{m} max_abs={max_abs:.8} mean_abs={:.8} padded_dense={}",
            sum_abs / (m * N) as f64,
            stats.padded_npu_dense_nodes,
        ));
    }

    let scratch_after = backend.executor_mut().scratch_stats();
    if scratch_after.bo_allocations != scratch_before.bo_allocations
        || scratch_after.bo_grows != scratch_before.bo_grows
    {
        return Err(format!(
            "dynamic batches changed scratch after max-M warmup: {scratch_before:?} -> {scratch_after:?}"
        )
        .into());
    }

    let (trace_out, _, trace) = prepared.run_fp16_traced(&warm_input, &mut backend)?;
    if trace_out
        .values()
        .iter()
        .zip(warm.values())
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err("dynamic traced batch50 differs from warm output".into());
    }
    save_trace("dynamic-mnist-npu", &trace)?;

    let summary = format!(
        "dynamic MNIST PASS batches={:?} dense=4 m_compatible=4 resident_MB={:.3} resident_tiles={} prepare_ms={:.3} pack_ms={:.3} scratch_allocs={} scratch_grows={} model_dropped=true\n{}",
        BATCHES,
        ps.resident_weight_bytes as f64 / 1e6,
        ps.resident_weight_tiles,
        ps.prepare_total_ns as f64 / 1e6,
        ps.weight_pack_ns as f64 / 1e6,
        scratch_after.bo_allocations,
        scratch_after.bo_grows,
        lines.join("\n"),
    );
    println!("{summary}");
    fs::write(art.join("dynamic-mnist-run.txt"), summary + "\n")?;
    Ok(())
}
