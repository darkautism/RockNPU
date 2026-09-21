use rocket_runtime::RocketDevice;
use rocknpu_matmul::{Int8DecodeExecutor, Int8DecodeOutput};

const DEFAULT_K: usize = 2048;
const DEFAULT_N: usize = 2048;

fn verify(
    label: &str,
    result: &Int8DecodeOutput,
    reference: &[i32],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut mismatches = Vec::new();
    for (col, (&expected, &actual)) in reference.iter().zip(&result.values).enumerate() {
        if actual != expected && mismatches.len() < 10 {
            mismatches.push((col, expected, actual));
        }
    }
    if !mismatches.is_empty() {
        return Err(format!("{label} mismatch examples: {mismatches:?}").into());
    }
    Ok(())
}

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

    let prepared = if k > 4096 {
        executor.prepare_weights_m1_fullk(&b, k, n)?
    } else {
        executor.prepare_weights(&b, k, n)?
    };
    let prepare = prepared.stats();

    // Judge both the first use and a second reuse of the exact same resident
    // BO. Production decode must not rely on a hidden warm-up/repack cycle.
    let first = executor.execute_prepared(&a, &prepared)?;
    verify("first prepared execution", &first, &reference)?;
    let reused = executor.execute_prepared(&a, &prepared)?;
    verify("reused prepared execution", &reused, &reference)?;

    println!(
        "INT8 PREPARED DECODE PASS M=1 K={k} N={n} outputs={n} resident_mb={:.2} prepare_us={:.1} first_pack_us={:.1} first_npu_us={:.1} reuse_pack_us={:.1} reuse_npu_us={:.1}",
        prepare.resident_bytes as f64 / (1024.0 * 1024.0),
        prepare.pack_ns as f64 / 1e3,
        first.stats.pack_ns as f64 / 1e3,
        first.stats.submit_wait_ns as f64 / 1e3,
        reused.stats.pack_ns as f64 / 1e3,
        reused.stats.submit_wait_ns as f64 / 1e3,
    );
    Ok(())
}
