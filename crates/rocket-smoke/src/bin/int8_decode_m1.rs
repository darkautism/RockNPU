use rocket_runtime::RocketDevice;
use rocknpu_matmul::Int8DecodeExecutor;

const DEFAULT_K: usize = 2048;
const DEFAULT_N: usize = 2048;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let k = args.next().map_or(Ok(DEFAULT_K), |v| v.parse::<usize>())?;
    let n = args.next().map_or(Ok(DEFAULT_N), |v| v.parse::<usize>())?;
    if args.next().is_some() {
        return Err("usage: int8_decode_m1 [K] [N]".into());
    }

    let device = RocketDevice::open()?;
    let executor = Int8DecodeExecutor::new(&device)?;

    let a: Vec<i8> = (0..k).map(|i| ((i * 7 + 3) % 9) as i8 - 4).collect();
    // Public/core convention is B[N,K], matching GGML/FP16 RockNPU weights.
    let b: Vec<i8> = (0..n * k).map(|i| ((i * 5 + 1) % 11) as i8 - 5).collect();
    let mut reference = vec![0i32; n];
    for col in 0..n {
        let mut sum = 0i32;
        for kk in 0..k {
            sum += i32::from(a[kk]) * i32::from(b[col * k + kk]);
        }
        reference[col] = sum;
    }

    // Deliberately judge the first execution: production GGML decode cannot rely
    // on an invisible warm-up submission.
    let result = executor.execute(&a, &b, k, n)?;
    let mut mismatches = Vec::new();
    for (col, (&expected, &actual)) in reference.iter().zip(&result.values).enumerate() {
        if actual != expected && mismatches.len() < 10 {
            mismatches.push((col, expected, actual));
        }
    }
    if !mismatches.is_empty() {
        return Err(format!("INT8 M=1 K={k} N={n} mismatch examples: {mismatches:?}").into());
    }

    println!(
        "INT8 DECODE PASS M=1 K={k} N={n} outputs={n} tasks={} slices={} pack_us={:.1} submit_wait_us={:.1} host_accum_us={:.1}",
        result.stats.npu_tasks,
        result.stats.k_slices,
        result.stats.pack_ns as f64 / 1e3,
        result.stats.submit_wait_ns as f64 / 1e3,
        result.stats.host_accum_ns as f64 / 1e3,
    );
    Ok(())
}
