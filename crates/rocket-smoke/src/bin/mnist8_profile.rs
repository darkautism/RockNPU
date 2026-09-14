use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_conv::Fp16Conv2dExecutor;
use rocknpu_onnx::{CnnOnnxModel, CnnPreparedConvState, TensorF16};
use rocknpu_ops::{ExecutionTarget, SingleNpuBackend};
use std::{error::Error, fs};
fn read_f32(path: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let b = fs::read(path)?;
    Ok(b.chunks_exact(4)
        .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect())
}
fn med(v: &mut [u128]) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}
fn main() -> Result<(), Box<dyn Error>> {
    let model = CnnOnnxModel::from_bytes(&fs::read("artifacts/mnist-8.onnx")?)?;
    let raw = read_f32("artifacts/mnist8-input100-f32.bin")?;
    let x = TensorF16::from_vec(
        vec![1, 1, 28, 28],
        raw[..784].iter().copied().map(f16::from_f32).collect(),
    )?;
    let dev = RocketDevice::open()?;
    let mut dense = SingleNpuBackend::new(&dev)?;
    let mut conv = Fp16Conv2dExecutor::new(&dev)?;
    let mut prepared = CnnPreparedConvState::new();
    for _ in 0..8 {
        let _ = model.run_fp16_with_prepared_conv(
            &x,
            ExecutionTarget::NpuSingle,
            Some(&mut dense),
            Some(&mut conv),
            Some(&mut prepared),
        )?;
    }
    let mut total = Vec::new();
    let mut cv = Vec::new();
    let mut pool = Vec::new();
    let mut reshape = Vec::new();
    let mut mm = Vec::new();
    let mut add = Vec::new();
    let mut relu = Vec::new();
    let mut overhead = Vec::new();
    for _ in 0..101 {
        let (_, _, t) = model.run_fp16_profiled_with_prepared_conv(
            &x,
            ExecutionTarget::NpuSingle,
            Some(&mut dense),
            Some(&mut conv),
            Some(&mut prepared),
        )?;
        total.push(t.total_ns);
        cv.push(t.conv_ns);
        pool.push(t.maxpool_ns);
        reshape.push(t.reshape_ns);
        mm.push(t.matmul_ns);
        add.push(t.add_ns);
        relu.push(t.relu_ns);
        overhead.push(t.graph_overhead_ns());
    }
    let vals = [
        ("conv", med(&mut cv)),
        ("maxpool", med(&mut pool)),
        ("reshape", med(&mut reshape)),
        ("matmul", med(&mut mm)),
        ("add", med(&mut add)),
        ("relu", med(&mut relu)),
        ("graph_overhead", med(&mut overhead)),
    ];
    let tm = med(&mut total);
    println!(
        "mnist8 prepared profile total_ms={:.4} prepared_conv={:?} prepared_dense={:?} scratch={:?}",
        tm as f64 / 1e6,
        prepared.stats(),
        prepared.dense_stats(),
        conv.scratch_stats()
    );
    for (n, v) in vals {
        println!(
            "{n:14} ms={:.4} pct={:.1}%",
            v as f64 / 1e6,
            100.0 * v as f64 / tm as f64
        );
    }
    Ok(())
}
