use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_conv::Fp16Conv2dExecutor;
use rocknpu_onnx::{CnnOnnxModel, CnnPreparedConvState, TensorF16};
use rocknpu_ops::{ExecutionTarget, SingleNpuBackend};
use std::{error::Error, fs, time::Instant};
fn read_f32(path: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let b = fs::read(path)?;
    Ok(b.chunks_exact(4)
        .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect())
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
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
    let mut conv_stream = Fp16Conv2dExecutor::new(&dev)?;
    let mut conv_prepared = Fp16Conv2dExecutor::new(&dev)?;
    let mut prepared = CnnPreparedConvState::new();
    for _ in 0..4 {
        let _ = model.run_fp16(&x, ExecutionTarget::NpuSingle, Some(&mut dense))?;
        let _ = model.run_fp16_with_conv(
            &x,
            ExecutionTarget::NpuSingle,
            Some(&mut dense),
            Some(&mut conv_stream),
        )?;
        let _ = model.run_fp16_with_prepared_conv(
            &x,
            ExecutionTarget::NpuSingle,
            Some(&mut dense),
            Some(&mut conv_prepared),
            Some(&mut prepared),
        )?;
    }
    let ps = prepared.stats();
    if ps.tensors != 2 || ps.resident_bytes != 51_200 {
        return Err(format!("unexpected prepared Conv state {ps:?}").into());
    }
    let ds = prepared.dense_stats();
    if ds.tensors != 1 || ds.resident_bytes == 0 {
        return Err(format!("unexpected prepared dense state {ds:?}").into());
    }
    if conv_prepared.scratch_stats().weight_bytes != 0 {
        return Err(format!(
            "prepared Conv executor allocated weight scratch: {:?}",
            conv_prepared.scratch_stats()
        )
        .into());
    }
    let mut cpu = Vec::new();
    let mut stream = Vec::new();
    let mut prep = Vec::new();
    for round in 0..33 {
        let order = match round % 3 {
            0 => [0, 1, 2],
            1 => [1, 2, 0],
            _ => [2, 0, 1],
        };
        for kind in order {
            let t = Instant::now();
            match kind {
                0 => {
                    let _ = model.run_fp16(&x, ExecutionTarget::NpuSingle, Some(&mut dense))?;
                    cpu.push(t.elapsed().as_secs_f64() * 1e3);
                }
                1 => {
                    let _ = model.run_fp16_with_conv(
                        &x,
                        ExecutionTarget::NpuSingle,
                        Some(&mut dense),
                        Some(&mut conv_stream),
                    )?;
                    stream.push(t.elapsed().as_secs_f64() * 1e3);
                }
                _ => {
                    let _ = model.run_fp16_with_prepared_conv(
                        &x,
                        ExecutionTarget::NpuSingle,
                        Some(&mut dense),
                        Some(&mut conv_prepared),
                        Some(&mut prepared),
                    )?;
                    prep.push(t.elapsed().as_secs_f64() * 1e3);
                }
            }
        }
    }
    let a = median(cpu);
    let b = median(stream);
    let c = median(prep);
    // Batch repeated full inferences so the ~tens-of-microseconds prepared-weight
    // effect is not dominated by per-sample scheduler/timer noise.
    let mut stream_block = Vec::new();
    let mut prep_block = Vec::new();
    const BLOCK_REPS: usize = 40;
    for block in 0..11 {
        if block % 2 == 0 {
            let t = Instant::now();
            for _ in 0..BLOCK_REPS {
                let _ = model.run_fp16_with_conv(
                    &x,
                    ExecutionTarget::NpuSingle,
                    Some(&mut dense),
                    Some(&mut conv_stream),
                )?;
            }
            stream_block.push(t.elapsed().as_secs_f64() * 1e3 / BLOCK_REPS as f64);

            let t = Instant::now();
            for _ in 0..BLOCK_REPS {
                let _ = model.run_fp16_with_prepared_conv(
                    &x,
                    ExecutionTarget::NpuSingle,
                    Some(&mut dense),
                    Some(&mut conv_prepared),
                    Some(&mut prepared),
                )?;
            }
            prep_block.push(t.elapsed().as_secs_f64() * 1e3 / BLOCK_REPS as f64);
        } else {
            let t = Instant::now();
            for _ in 0..BLOCK_REPS {
                let _ = model.run_fp16_with_prepared_conv(
                    &x,
                    ExecutionTarget::NpuSingle,
                    Some(&mut dense),
                    Some(&mut conv_prepared),
                    Some(&mut prepared),
                )?;
            }
            prep_block.push(t.elapsed().as_secs_f64() * 1e3 / BLOCK_REPS as f64);

            let t = Instant::now();
            for _ in 0..BLOCK_REPS {
                let _ = model.run_fp16_with_conv(
                    &x,
                    ExecutionTarget::NpuSingle,
                    Some(&mut dense),
                    Some(&mut conv_stream),
                )?;
            }
            stream_block.push(t.elapsed().as_secs_f64() * 1e3 / BLOCK_REPS as f64);
        }
    }
    let sb = median(stream_block);
    let pb = median(prep_block);
    println!(
        "mnist8 block bench streaming_npu_conv_ms={sb:.4} prepared_npu_conv_ms={pb:.4} stream_to_prepared={:.3}x block_reps={BLOCK_REPS}",
        sb / pb
    );
    println!(
        "mnist8 model bench cpu_conv_ms={a:.3} streaming_npu_conv_ms={b:.3} prepared_npu_conv_ms={c:.3} cpu_to_stream={:.2}x cpu_to_prepared={:.2}x stream_to_prepared={:.2}x prepared_state={:?} prepared_dense={:?} prepared_scratch={:?}",
        a / b,
        a / c,
        b / c,
        ps,
        ds,
        conv_prepared.scratch_stats()
    );
    Ok(())
}
