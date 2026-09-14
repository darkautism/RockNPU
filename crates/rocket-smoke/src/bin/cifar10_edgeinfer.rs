use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_conv::Fp16Conv2dExecutor;
use rocknpu_onnx::{CnnOnnxModel, TensorF16, TensorTrace};
use rocknpu_ops::{ExecutionTarget, SingleNpuBackend};
use std::{error::Error, fs};

fn read_f32(path: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let b = fs::read(path)?;
    if b.len() % 4 != 0 {
        return Err(format!("bad f32 bytes {path}").into());
    }
    Ok(b.chunks_exact(4)
        .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect())
}
fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|x| x.0)
        .unwrap()
}
fn save_trace(prefix: &str, trace: &[TensorTrace]) -> Result<(), Box<dyn Error>> {
    let mut meta = String::from("index\top\tname\tdims\toffset\tcount\n");
    let mut raw = Vec::new();
    let mut off = 0usize;
    for (i, t) in trace.iter().enumerate() {
        meta.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\n",
            i,
            t.op,
            t.name,
            t.dims
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x"),
            off,
            t.values.len()
        ));
        for v in &t.values {
            raw.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        off += t.values.len();
    }
    fs::write(format!("artifacts/{prefix}-trace.tsv"), meta)?;
    fs::write(format!("artifacts/{prefix}-trace-f16.bin"), raw)?;
    Ok(())
}
fn compare_final(
    name: &str,
    out: &TensorF16,
    reference: &[f32],
) -> Result<(f32, f32, usize), Box<dyn Error>> {
    if out.dims() != [1, 10] || reference.len() != 10 {
        return Err(format!("{name} final shape mismatch {:?}", out.dims()).into());
    }
    let got: Vec<f32> = out.values().iter().map(|x| x.to_f32()).collect();
    let mut max = 0.0f32;
    let mut sum = 0.0f64;
    for (&a, &b) in got.iter().zip(reference) {
        let e = (a - b).abs();
        max = max.max(e);
        sum += e as f64;
    }
    Ok((max, (sum / 10.0) as f32, argmax(&got)))
}
fn main() -> Result<(), Box<dyn Error>> {
    let model = CnnOnnxModel::from_bytes(&fs::read("artifacts/cifar10-edgeinfer.onnx")?)?;
    if model.node_count() != 13 {
        return Err(format!("expected 13 nodes got {}", model.node_count()).into());
    }
    let input = read_f32("artifacts/cifar10-edgeinfer-input-f32.bin")?;
    let reference = read_f32("artifacts/cifar10-edgeinfer-ref-f32.bin")?;
    if input.len() != 3 * 32 * 32 {
        return Err("CIFAR input length mismatch".into());
    }
    let x = TensorF16::from_vec(
        vec![1, 3, 32, 32],
        input.into_iter().map(f16::from_f32).collect(),
    )?;

    let (cpu, cpu_stats, cpu_trace) = model.run_fp16_traced(&x, ExecutionTarget::Cpu, None)?;
    let (cpu_max, cpu_mean, cpu_pred) = compare_final("cpu", &cpu, &reference)?;
    save_trace("cifar10-edgeinfer-cpu", &cpu_trace)?;
    if cpu_pred != 8 {
        return Err(format!("CPU path predicted {cpu_pred}, expected ship=8").into());
    }

    let dev = RocketDevice::open()?;
    let mut dense = SingleNpuBackend::new(&dev)?;
    let mut conv = Fp16Conv2dExecutor::new(&dev)?;
    let (npu, npu_stats, npu_trace) = model.run_fp16_traced_with_conv(
        &x,
        ExecutionTarget::NpuSingle,
        Some(&mut dense),
        Some(&mut conv),
    )?;
    let (npu_max, npu_mean, npu_pred) = compare_final("npu", &npu, &reference)?;
    save_trace("cifar10-edgeinfer-npu", &npu_trace)?;
    if npu_pred != 8 {
        return Err(format!("NPU path predicted {npu_pred}, expected ship=8").into());
    }
    if cpu_stats.conv_nodes != 3
        || cpu_stats.maxpool_nodes != 3
        || cpu_stats.reshape_nodes != 1
        || cpu_stats.gemm_nodes != 2
        || cpu_stats.relu_nodes != 4
    {
        return Err(format!("unexpected CPU graph stats {cpu_stats:?}").into());
    }
    if npu_stats.conv_nodes != 3
        || npu_stats.npu_conv_nodes != 3
        || npu_stats.gemm_nodes != 2
        || npu_stats.npu_dense_nodes != 2
        || npu_stats.padded_npu_dense_nodes != 2
        || npu_stats.maxpool_nodes != 3
        || npu_stats.reshape_nodes != 1
        || npu_stats.relu_nodes != 4
    {
        return Err(format!("unexpected NPU graph stats {npu_stats:?}").into());
    }
    let bitdiff = cpu
        .values()
        .iter()
        .zip(npu.values())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    println!(
        "CIFAR10 EDGE-INFER PASS nodes=13 true=8 cpu_pred={cpu_pred} npu_pred={npu_pred} conv_npu={} gemm_npu={} padded_dense={} cpu_max_abs={cpu_max:.6} cpu_mean_abs={cpu_mean:.6} npu_max_abs={npu_max:.6} npu_mean_abs={npu_mean:.6} cpu_npu_final_bitdiff={bitdiff}/10 conv_scratch={:?}",
        npu_stats.npu_conv_nodes,
        npu_stats.npu_dense_nodes,
        npu_stats.padded_npu_dense_nodes,
        conv.scratch_stats()
    );
    Ok(())
}
