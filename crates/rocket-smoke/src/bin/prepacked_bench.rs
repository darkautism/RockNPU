use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_matmul::Fp16MatmulExecutor;
use std::error::Error;

const REPS: usize = 5;

fn data(m: usize, k: usize, n: usize, seed: u64) -> (Vec<f16>, Vec<f16>) {
    fn next(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }
    let mut s = seed;
    let a = (0..m * k)
        .map(|_| f16::from_f32((next(&mut s) % 17) as f32 / 8.0 - 1.0))
        .collect();
    let b = (0..n * k)
        .map(|_| f16::from_f32((next(&mut s) % 17) as f32 / 8.0 - 1.0))
        .collect();
    (a, b)
}

fn median(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
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
    let stream_warm = ex.execute(&a, &b, m, k, n)?;
    let weights = ex.prepack_weights(&b, m, k, n)?;
    let resident_warm = ex.execute_prepacked(&a, &weights)?;
    if stream_warm
        .values
        .iter()
        .zip(&resident_warm.values)
        .any(|(x, y)| x.to_bits() != y.to_bits())
    {
        return Err(format!("{label}: resident warmup differs from streaming").into());
    }

    let scratch_before = ex.scratch_stats().bo_allocations;
    let mut stream_total = Vec::with_capacity(REPS);
    let mut stream_pack = Vec::with_capacity(REPS);
    let mut resident_total = Vec::with_capacity(REPS);
    let mut resident_pack = Vec::with_capacity(REPS);
    let mut baseline = None;
    for _ in 0..REPS {
        let out = ex.execute(&a, &b, m, k, n)?;
        if baseline.is_none() {
            baseline = Some(out.values.clone());
        }
        stream_total.push(out.stats.timing.total_ns);
        stream_pack.push(out.stats.timing.pack_ns);
    }
    let baseline = baseline.unwrap();
    for _ in 0..REPS {
        let out = ex.execute_prepacked(&a, &weights)?;
        if out
            .values
            .iter()
            .zip(&baseline)
            .any(|(x, y)| x.to_bits() != y.to_bits())
        {
            return Err(format!("{label}: resident repeated output differs").into());
        }
        resident_total.push(out.stats.timing.total_ns);
        resident_pack.push(out.stats.timing.pack_ns);
    }
    let scratch_after = ex.scratch_stats().bo_allocations;
    if scratch_after != scratch_before {
        return Err(format!("{label}: benchmark allocated scratch after warmup").into());
    }

    let st_med = median(stream_total.clone());
    let rt_med = median(resident_total.clone());
    let st_best = *stream_total.iter().min().unwrap();
    let rt_best = *resident_total.iter().min().unwrap();
    let sp_med = median(stream_pack);
    let rp_med = median(resident_pack);
    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    let g_stream = flops / st_med as f64;
    let g_resident = flops / rt_med as f64;
    Ok(format!(
        "{label} M{m} K{k} N{n} stream_median_ms={:.3} resident_median_ms={:.3} speedup={:.2}x stream_GFLOPs={:.2} resident_GFLOPs={:.2} stream_best_ms={:.3} resident_best_ms={:.3} stream_pack_ms={:.3} resident_pack_ms={:.3} pack_reduction={:.2}x resident_MB={:.3} one_time_prepack_ms={:.3} unique_tiles={}",
        st_med as f64 / 1e6,
        rt_med as f64 / 1e6,
        st_med as f64 / rt_med as f64,
        g_stream,
        g_resident,
        st_best as f64 / 1e6,
        rt_best as f64 / 1e6,
        sp_med as f64 / 1e6,
        rp_med as f64 / 1e6,
        sp_med as f64 / rp_med.max(1) as f64,
        weights.stats().resident_bytes as f64 / 1e6,
        weights.stats().pack_ns as f64 / 1e6,
        weights.stats().unique_tiles,
    ))
}

fn main() -> Result<(), Box<dyn Error>> {
    let dev = RocketDevice::open()?;
    let mut ex = Fp16MatmulExecutor::new(&dev)?;
    let cases = [
        ("single-ish", 256, 512, 128, 0x3588),
        ("deep-k", 64, 4096, 128, 0x2499),
        ("prefill-ish", 64, 4096, 512, 0x5eed),
        ("wide-tiled", 256, 1024, 256, 0xc0ffee),
    ];
    for (label, m, k, n, seed) in cases {
        println!("{}", run_case(&mut ex, label, m, k, n, seed)?);
    }
    println!(
        "PASS: streaming vs resident/prepacked FP16 benchmark; cases={}",
        cases.len()
    );
    Ok(())
}
