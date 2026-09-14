use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_matmul::{Fp16MatmulExecutor, Fp16MatmulPool, cpu_reference_fp32};
use std::{error::Error, sync::Arc};

fn integer_data(m: usize, k: usize, n: usize) -> (Vec<f16>, Vec<f16>) {
    let a = (0..m * k)
        .map(|i| f16::from_f32(((i * 7 + 3) % 5) as f32 - 2.0))
        .collect();
    let b = (0..n * k)
        .map(|i| f16::from_f32(((i * 3 + 1) % 5) as f32 - 2.0))
        .collect();
    (a, b)
}

fn fractional_data(m: usize, k: usize, n: usize, mut s: u64) -> (Vec<f16>, Vec<f16>) {
    fn next(s: &mut u64) -> u32 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 32) as u32
    }
    let mut a = Vec::with_capacity(m * k);
    let mut b = Vec::with_capacity(n * k);
    for _ in 0..m * k {
        let x = (next(&mut s) % 2001) as f32 / 1000.0 - 1.0;
        a.push(f16::from_f32(x));
    }
    for _ in 0..n * k {
        let x = (next(&mut s) % 2001) as f32 / 1000.0 - 1.0;
        b.push(f16::from_f32(x));
    }
    (a, b)
}

fn errors(reference: &[f32], got: &[f32]) -> (f32, f64) {
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

fn main() -> Result<(), Box<dyn Error>> {
    let mut pool = Fp16MatmulPool::new(3)?;

    // Exact small-integer operation contract.
    let (a0, b0) = integer_data(64, 256, 96);
    let ref0 = cpu_reference_fp32(&a0, &b0, 64, 256, 96)?;
    let out0 = pool.execute_f32(Arc::from(a0), Arc::from(b0), 64, 256, 96)?;
    if ref0
        .iter()
        .zip(&out0.values)
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err("FP32 pool integer gate is not bit-exact".into());
    }
    println!(
        "pool-fp32 integer PASS M64 K256 N96 workers={} jobs={} exact=true",
        out0.stats.workers_used, out0.stats.jobs_submitted
    );

    // Deep-K fractional case: validates FP32 NPU partials + host-f64 K accumulation.
    const M: usize = 64;
    const K: usize = 4096;
    const N: usize = 192;
    let (a, b) = fractional_data(M, K, N, 0x5eed_f32a_cc01_2026);
    let reference = cpu_reference_fp32(&a, &b, M, K, N)?;

    let dev = RocketDevice::open()?;
    let mut single = Fp16MatmulExecutor::new(&dev)?;
    let single_out = single.execute_f32(&a, &b, M, K, N)?;
    let (single_max, single_nrms) = errors(&reference, &single_out.values);

    let warm = pool.execute_f32(Arc::from(a.clone()), Arc::from(b.clone()), M, K, N)?;
    let scratch_warm: Vec<(usize, usize)> = warm
        .stats
        .worker_scratch
        .iter()
        .map(|s| (s.bo_allocations, s.bo_grows))
        .collect();
    let repeat = pool.execute_f32(Arc::from(a), Arc::from(b), M, K, N)?;
    let scratch_repeat: Vec<(usize, usize)> = repeat
        .stats
        .worker_scratch
        .iter()
        .map(|s| (s.bo_allocations, s.bo_grows))
        .collect();
    if scratch_repeat != scratch_warm {
        return Err(format!(
            "FP32 pool scratch changed after warmup: {scratch_warm:?} -> {scratch_repeat:?}"
        )
        .into());
    }
    if warm
        .values
        .iter()
        .zip(&repeat.values)
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err("FP32 pool repeated output changed".into());
    }
    let (pool_max, pool_nrms) = errors(&reference, &repeat.values);
    let (single_pool_max, _) = errors(&single_out.values, &repeat.values);
    if !pool_nrms.is_finite() || pool_nrms > 1.0e-5 {
        return Err(format!("FP32 pool normalized RMS error too large: {pool_nrms:e}").into());
    }
    println!(
        "pool-fp32 deep PASS M{M} K{K} N{N} workers={} jobs={} wall_ms={:.3} max_abs={:.8} nrms={:.3e} single_max_abs={:.8} single_nrms={:.3e} single_pool_max_abs={:.8} scratch={scratch_repeat:?}",
        repeat.stats.workers_used,
        repeat.stats.jobs_submitted,
        repeat.stats.wall_ns as f64 / 1e6,
        pool_max,
        pool_nrms,
        single_max,
        single_nrms,
        single_pool_max,
    );
    println!("PASS: persistent FP32Accurate pool hardware gate");
    Ok(())
}
