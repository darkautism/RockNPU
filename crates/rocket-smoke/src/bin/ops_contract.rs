use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_ops::{
    ExecutionTarget, MatmulOutput, MatmulPrecision, MatmulSpec, PoolNpuBackend, SingleNpuBackend,
    execute_auto, execute_cpu,
};
use rocknpu_tensor::Matrix;
use std::error::Error;

fn data(m: usize, k: usize, n: usize) -> (Matrix<f16>, Matrix<f16>) {
    let a = (0..m * k)
        .map(|i| f16::from_f32(((i * 7 + 3) % 5) as f32 - 2.0))
        .collect();
    let b = (0..n * k)
        .map(|i| f16::from_f32(((i * 3 + 1) % 5) as f32 - 2.0))
        .collect();
    (
        Matrix::from_vec(m, k, a).unwrap(),
        Matrix::from_vec(n, k, b).unwrap(),
    )
}

fn fractional_data(m: usize, k: usize, n: usize, mut seed: u64) -> (Matrix<f16>, Matrix<f16>) {
    fn next(seed: &mut u64) -> u32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 32) as u32
    }
    let a = (0..m * k)
        .map(|_| f16::from_f32((next(&mut seed) % 2001) as f32 / 1000.0 - 1.0))
        .collect();
    let b = (0..n * k)
        .map(|_| f16::from_f32((next(&mut seed) % 2001) as f32 / 1000.0 - 1.0))
        .collect();
    (
        Matrix::from_vec(m, k, a).unwrap(),
        Matrix::from_vec(n, k, b).unwrap(),
    )
}

fn normalized_rms(reference: &[f32], got: &[f32]) -> (f32, f64) {
    let mut max_abs = 0.0f32;
    let mut err2 = 0.0f64;
    let mut ref2 = 0.0f64;
    for (&r, &g) in reference.iter().zip(got) {
        let e = (r - g).abs();
        max_abs = max_abs.max(e);
        err2 += (e as f64) * (e as f64);
        ref2 += (r as f64) * (r as f64);
    }
    let nrms = if ref2 == 0.0 {
        err2.sqrt()
    } else {
        (err2 / ref2).sqrt()
    };
    (max_abs, nrms)
}

fn expect_f16(out: MatmulOutput) -> Matrix<f16> {
    match out {
        MatmulOutput::F16(v) => v,
        _ => panic!("expected f16"),
    }
}
fn expect_f32(out: MatmulOutput) -> Matrix<f32> {
    match out {
        MatmulOutput::F32(v) => v,
        _ => panic!("expected f32"),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let dev = RocketDevice::open()?;
    let mut single = SingleNpuBackend::new(&dev)?;

    let (a, b) = data(64, 256, 64);
    let spec = MatmulSpec::new(
        64,
        256,
        64,
        MatmulPrecision::Fp16Fast,
        ExecutionTarget::NpuSingle,
    );
    let cpu = expect_f16(execute_cpu(spec, &a, &b)?);
    let npu = expect_f16(single.execute(spec, &a, &b)?);
    if cpu
        .values()
        .iter()
        .zip(npu.values())
        .any(|(x, y)| x.to_bits() != y.to_bits())
    {
        return Err("FP16 op-contract CPU/NPU mismatch".into());
    }
    println!("ops single-fp16 PASS shape=64x256x64");

    let spec32 = MatmulSpec::new(
        64,
        256,
        64,
        MatmulPrecision::Fp32Accurate,
        ExecutionTarget::NpuSingle,
    );
    let cpu32 = expect_f32(execute_cpu(spec32, &a, &b)?);
    let npu32 = expect_f32(single.execute(spec32, &a, &b)?);
    let mut max = 0.0f32;
    for (x, y) in cpu32.values().iter().zip(npu32.values()) {
        max = max.max((x - y).abs());
    }
    if max != 0.0 {
        return Err(format!("FP32 op-contract integer-vector mismatch max={max}").into());
    }
    println!("ops single-fp32 PASS shape=64x256x64 max_abs_error=0");

    let (ap, bp) = data(64, 256, 96);
    let mut pool = PoolNpuBackend::new(3)?;
    let pspec = MatmulSpec::new(
        64,
        256,
        96,
        MatmulPrecision::Fp16Fast,
        ExecutionTarget::NpuPool,
    );
    let pcpu = expect_f16(execute_cpu(pspec, &ap, &bp)?);
    let pnpu = expect_f16(pool.execute(pspec, &ap, &bp)?);
    if pcpu
        .values()
        .iter()
        .zip(pnpu.values())
        .any(|(x, y)| x.to_bits() != y.to_bits())
    {
        return Err("pool op-contract CPU/NPU mismatch".into());
    }
    println!(
        "ops pool-fp16 PASS shape=64x256x96 workers={}",
        pool.workers()
    );

    let (ap32, bp32) = fractional_data(64, 4096, 96, 0x0f32_2026_0914_5eed);
    let pspec32 = MatmulSpec::new(
        64,
        4096,
        96,
        MatmulPrecision::Fp32Accurate,
        ExecutionTarget::NpuPool,
    );
    let pcpu32 = expect_f32(execute_cpu(pspec32, &ap32, &bp32)?);
    let pnpu32 = expect_f32(pool.execute(pspec32, &ap32, &bp32)?);
    let (pool32_max, pool32_nrms) = normalized_rms(pcpu32.values(), pnpu32.values());
    if !pool32_nrms.is_finite() || pool32_nrms > 1.0e-5 {
        return Err(
            format!("FP32 pool op-contract normalized RMS too large: {pool32_nrms:e}").into(),
        );
    }
    println!(
        "ops pool-fp32 PASS shape=64x4096x96 workers={} max_abs={:.8} nrms={:.3e}",
        pool.workers(),
        pool32_max,
        pool32_nrms
    );

    let (au, bu) = data(3, 7, 5);
    let auto = MatmulSpec::new(3, 7, 5, MatmulPrecision::Fp16Fast, ExecutionTarget::Auto);
    let expected = expect_f16(execute_cpu(auto, &au, &bu)?);
    let got = expect_f16(execute_auto(
        auto,
        &au,
        &bu,
        Some(&mut single),
        Some(&mut pool),
    )?);
    if expected != got {
        return Err("Auto CPU fallback mismatch".into());
    }
    println!("ops auto-cpu-fallback PASS shape=3x7x5");

    println!("PASS: project-owned tensor/op MatMul contract hardware gate");
    Ok(())
}
