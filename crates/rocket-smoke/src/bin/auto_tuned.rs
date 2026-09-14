use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_ops::{
    AutoTunedNpuBackend, ExecutionTarget, MatmulOutput, MatmulPrecision, MatmulSpec, execute_cpu,
};
use rocknpu_tensor::Matrix;
use std::{error::Error, time::Instant};

fn data(m: usize, k: usize, n: usize, mut seed: u64) -> (Matrix<f16>, Matrix<f16>) {
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

fn f16_values(out: MatmulOutput) -> Matrix<f16> {
    match out {
        MatmulOutput::F16(v) => v,
        _ => panic!("expected f16"),
    }
}
fn f32_values(out: MatmulOutput) -> Matrix<f32> {
    match out {
        MatmulOutput::F32(v) => v,
        _ => panic!("expected f32"),
    }
}
fn f16_nrms(reference: &[f16], got: &[f16]) -> (f32, f64) {
    let r: Vec<f32> = reference.iter().map(|v| v.to_f32()).collect();
    let g: Vec<f32> = got.iter().map(|v| v.to_f32()).collect();
    nrms(&r, &g)
}

fn nrms(reference: &[f32], got: &[f32]) -> (f32, f64) {
    let mut max_abs = 0.0f32;
    let mut e2 = 0.0f64;
    let mut r2 = 0.0f64;
    for (&r, &g) in reference.iter().zip(got) {
        let e = (r - g).abs();
        max_abs = max_abs.max(e);
        e2 += (e as f64) * (e as f64);
        r2 += (r as f64) * (r as f64);
    }
    (max_abs, (e2 / r2.max(f64::MIN_POSITIVE)).sqrt())
}

fn gate_fp16(
    auto: &mut AutoTunedNpuBackend<'_>,
    tag: &str,
    m: usize,
    k: usize,
    n: usize,
    seed: u64,
) -> Result<(), Box<dyn Error>> {
    let (a, b) = data(m, k, n, seed);
    let spec = MatmulSpec::new(m, k, n, MatmulPrecision::Fp16Fast, ExecutionTarget::Auto);
    let cpu = f16_values(execute_cpu(spec, &a, &b)?);
    let t0 = Instant::now();
    let first = f16_values(auto.execute(spec, &a, &b)?);
    let first_ns = t0.elapsed().as_nanos();
    let workers = auto
        .cached_workers(spec)
        .ok_or("Auto policy was not cached")?;
    let cache_before = auto.cached_shapes();
    let t1 = Instant::now();
    let second = f16_values(auto.execute(spec, &a, &b)?);
    let cached_ns = t1.elapsed().as_nanos();
    if auto.cached_shapes() != cache_before || auto.cached_workers(spec) != Some(workers) {
        return Err("cached Auto policy changed on second identical shape".into());
    }
    if first != second {
        return Err(format!("{tag}: cached FP16 execution differs from first result").into());
    }
    let (max_abs, norm) = f16_nrms(cpu.values(), first.values());
    if !norm.is_finite() || norm > 5.0e-3 {
        return Err(format!("{tag}: Auto FP16 normalized RMS too large: {norm:e}").into());
    }
    println!(
        "auto {tag} FP16 PASS M{m} K{k} N{n} selected_workers={workers} first_ms={:.3} cached_ms={:.3} max_abs={max_abs:.6} nrms={norm:.3e}",
        first_ns as f64 / 1e6,
        cached_ns as f64 / 1e6,
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let dev = RocketDevice::open()?;
    let mut auto = AutoTunedNpuBackend::new(&dev, 3)?;

    gate_fp16(&mut auto, "small", 64, 256, 96, 1)?;
    gate_fp16(&mut auto, "wide", 256, 384, 768, 2)?;

    let (a32, b32) = data(64, 4096, 192, 3);
    let spec32 = MatmulSpec::new(
        64,
        4096,
        192,
        MatmulPrecision::Fp32Accurate,
        ExecutionTarget::Auto,
    );
    let cpu32 = f32_values(execute_cpu(spec32, &a32, &b32)?);
    let t0 = Instant::now();
    let first32 = f32_values(auto.execute(spec32, &a32, &b32)?);
    let first32_ns = t0.elapsed().as_nanos();
    let workers32 = auto
        .cached_workers(spec32)
        .ok_or("FP32 Auto policy was not cached")?;
    let t1 = Instant::now();
    let second32 = f32_values(auto.execute(spec32, &a32, &b32)?);
    let cached32_ns = t1.elapsed().as_nanos();
    if first32 != second32 {
        return Err("cached FP32 Auto result changed".into());
    }
    let (max_abs, norm) = nrms(cpu32.values(), second32.values());
    if !norm.is_finite() || norm > 1.0e-5 {
        return Err(format!("Auto FP32 normalized RMS too large: {norm:e}").into());
    }
    println!(
        "auto deep FP32 PASS M64 K4096 N192 selected_workers={workers32} first_ms={:.3} cached_ms={:.3} max_abs={max_abs:.8} nrms={norm:.3e}",
        first32_ns as f64 / 1e6,
        cached32_ns as f64 / 1e6,
    );

    let cache_before = auto.cached_shapes();
    let (au, bu) = data(3, 7, 5, 4);
    let unaligned = MatmulSpec::new(3, 7, 5, MatmulPrecision::Fp16Fast, ExecutionTarget::Auto);
    let expected = f16_values(execute_cpu(unaligned, &au, &bu)?);
    let got = f16_values(auto.execute(unaligned, &au, &bu)?);
    if got != expected || auto.cached_shapes() != cache_before {
        return Err("unaligned Auto fallback/cache contract failed".into());
    }
    println!("auto unaligned CPU fallback PASS cache_unchanged=true");
    println!(
        "PASS: adaptive benchmark-driven Auto NPU policy gate cached_shapes={}",
        auto.cached_shapes()
    );
    Ok(())
}
