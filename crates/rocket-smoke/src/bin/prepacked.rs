use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_matmul::{Fp16MatmulExecutor, cpu_reference_executor_semantics};
use std::error::Error;

fn data(m: usize, k: usize, n: usize, seed: u64) -> (Vec<f16>, Vec<f16>) {
    fn next(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }
    let mut s = seed;
    let a = (0..m * k)
        .map(|_| f16::from_f32((next(&mut s) % 7) as f32 - 3.0))
        .collect();
    let b = (0..n * k)
        .map(|_| f16::from_f32((next(&mut s) % 7) as f32 - 3.0))
        .collect();
    (a, b)
}

fn mismatches(a: &[f16], b: &[f16]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

fn run_case(
    ex: &mut Fp16MatmulExecutor<'_>,
    label: &str,
    m: usize,
    k: usize,
    n: usize,
    seed: u64,
) -> Result<String, Box<dyn Error>> {
    let (a, b) = data(m, k, n, seed);
    let oracle = cpu_reference_executor_semantics(&a, &b, m, k, n)?;
    let weights = ex.prepack_weights(&b, m, k, n)?;
    let wstats = weights.stats();
    let first = ex.execute_prepacked(&a, &weights)?;
    let scratch1 = ex.scratch_stats();
    let second = ex.execute_prepacked(&a, &weights)?;
    let scratch2 = ex.scratch_stats();
    let mm1 = mismatches(&first.values, &oracle);
    let mm2 = mismatches(&second.values, &oracle);
    if mm1 != 0 || mm2 != 0 {
        return Err(format!("{label}: resident result mismatch first={mm1} second={mm2}").into());
    }
    if scratch2.bo_allocations != scratch1.bo_allocations {
        return Err(format!("{label}: repeated resident execute allocated new scratch BOs").into());
    }
    Ok(format!(
        "{label} PASS M{m} K{k} N{n} jobs={} unique_weight_tiles={} resident_bytes={} prepack_ms={:.3} first_ms={:.3} second_ms={:.3} pack_phase_second_ms={:.3} scratch_allocs={}",
        second.stats.jobs_submitted,
        wstats.unique_tiles,
        wstats.resident_bytes,
        wstats.pack_ns as f64 / 1e6,
        first.stats.timing.total_ns as f64 / 1e6,
        second.stats.timing.total_ns as f64 / 1e6,
        second.stats.timing.pack_ns as f64 / 1e6,
        scratch2.bo_allocations,
    ))
}

fn main() -> Result<(), Box<dyn Error>> {
    let dev = RocketDevice::open()?;
    let mut ex = Fp16MatmulExecutor::new(&dev)?;
    let cases = [
        ("single", 256, 512, 128, 0x3588),
        ("mnk-ragged", 300, 512, 272, 0x5eed),
        ("deep-k-ew", 64, 4096, 128, 0xc0ffee),
        ("tiny-m-host", 4, 4096, 64, 0x2499),
    ];
    for (label, m, k, n, seed) in cases {
        println!("{}", run_case(&mut ex, label, m, k, n, seed)?);
    }
    println!(
        "PASS: resident/prepacked FP16 weight hardware gate; cases={}",
        cases.len()
    );
    Ok(())
}
