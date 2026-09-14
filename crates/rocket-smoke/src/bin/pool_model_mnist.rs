use half::f16;
use rocknpu_onnx::{TinyOnnxModel, TraceTensor};
use rocknpu_ops::PoolNpuBackend;
use rocknpu_tensor::{Matrix, MatrixShape};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

const K: usize = 784;
const N: usize = 10;
const MAX_M: usize = 50;
const BATCHES: [usize; 4] = [1, 4, 16, 50];
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

fn median(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
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

    let mut backend = PoolNpuBackend::new(3)?;
    let prepared = model.prepare_npu_pool(MatrixShape::new(MAX_M, K), &mut backend)?;
    let ps = prepared.stats();
    if ps.dense_nodes != 4 || ps.resident_worker_copies < 4 {
        return Err(format!("unexpected prepared pool model stats: {ps:?}").into());
    }
    drop(model);

    let input50 = input_matrix(&input_f32, MAX_M)?;
    let (warm, warm_stats) = prepared.run_fp16(&input50, &mut backend)?;
    if warm.rows() != MAX_M || warm.cols() != N || warm_stats.npu_dense_nodes != 4 {
        return Err("pool model warmup failed".into());
    }

    let mut lines = Vec::new();
    for m in BATCHES {
        let input = input_matrix(&input_f32, m)?;
        let (out, stats) = prepared.run_fp16(&input, &mut backend)?;
        let got: Vec<f32> = out.values().iter().map(|v| v.to_f32()).collect();
        let mut matches = 0usize;
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        for r in 0..m {
            let gp = argmax(&got[r * N..(r + 1) * N]);
            let rp = argmax(&reference[r * N..(r + 1) * N]);
            if gp == rp {
                matches += 1;
            }
            for c in 0..N {
                let e = (got[r * N + c] - reference[r * N + c]).abs();
                max_abs = max_abs.max(e);
                sum_abs += e as f64;
            }
        }
        if matches != m || stats.npu_dense_nodes != 4 {
            return Err(format!("batch {m}: top1={matches}/{m}, stats={stats:?}").into());
        }
        lines.push(format!(
            "batch={m} top1_ref={matches}/{m} max_abs={max_abs:.8} mean_abs={:.8} padded_dense={}",
            sum_abs / (m * N) as f64,
            stats.padded_npu_dense_nodes,
        ));
    }

    let mut times = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let start = Instant::now();
        let (out, _) = prepared.run_fp16(&input50, &mut backend)?;
        times.push(start.elapsed().as_nanos());
        if out
            .values()
            .iter()
            .zip(warm.values())
            .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err("pool prepared batch50 output changed across repeats".into());
        }
    }
    let med = median(times);

    let (traced, _, trace) = prepared.run_fp16_traced(&input50, &mut backend)?;
    if traced
        .values()
        .iter()
        .zip(warm.values())
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err("pool prepared traced output changed".into());
    }
    save_trace("pool-mnist-npu", &trace)?;
    prepared.release(&mut backend)?;

    let summary = format!(
        "pool model MNIST PASS workers={} dense=4 resident_copies={} resident_MB={:.3} resident_tiles={} prepare_ms={:.3} pack_sum_ms={:.3} batch50_median_ms={:.3} model_dropped=true release=true\n{}",
        backend.workers(),
        ps.resident_worker_copies,
        ps.resident_weight_bytes as f64 / 1e6,
        ps.resident_weight_tiles,
        ps.prepare_total_ns as f64 / 1e6,
        ps.weight_pack_ns_sum as f64 / 1e6,
        med as f64 / 1e6,
        lines.join("\n"),
    );
    println!("{summary}");
    fs::write(art.join("pool-mnist-run.txt"), summary + "\n")?;
    Ok(())
}
