use rocket_runtime::RocketDevice;
use rocknpu_matmul::Int4DecodeExecutor;
use std::time::Instant;

const K: usize = 2048;
const N: usize = 5632;
const GROUP: usize = 512;

fn cpu_grouped_reference(a: &[i8], b_nk: &[i8]) -> Vec<i16> {
    let groups = K / GROUP;
    let mut out = vec![0i16; groups * N];
    for group in 0..groups {
        let k0 = group * GROUP;
        for n in 0..N {
            let mut sum = 0i32;
            for kk in 0..GROUP {
                sum += i32::from(a[k0 + kk]) * i32::from(b_nk[n * K + k0 + kk]);
            }
            out[group * N + n] = i16::try_from(sum).expect("test values cannot saturate int16");
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<i8> = (0..K).map(|i| ((i * 7 + 3) % 5) as i8 - 2).collect();
    let b: Vec<i8> = (0..N * K).map(|i| ((i * 5 + 1) % 5) as i8 - 2).collect();
    let expected = cpu_grouped_reference(&a, &b);

    let device = RocketDevice::open()?;
    let executor = Int4DecodeExecutor::new(&device);
    let prepare_start = Instant::now();
    let prepared = executor.prepare_grouped_weights(&b, K, N, GROUP)?;
    let prepare_ms = prepare_start.elapsed().as_secs_f64() * 1.0e3;

    for run in 0..2 {
        let start = Instant::now();
        let result = executor.execute_grouped_prepared(&a, &prepared)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1.0e3;
        let mut wrong = 0usize;
        let mut examples = Vec::new();
        for (index, (&got, &want)) in result.values.iter().zip(&expected).enumerate() {
            if got != want {
                wrong += 1;
                if examples.len() < 10 {
                    examples.push((index, want, got));
                }
            }
        }
        println!(
            "W4A4 GROUPED run={} K={} N={} G={} groups={} elapsed_ms={:.3} saturated={} wrong={}",
            run + 1,
            K,
            N,
            GROUP,
            prepared.groups(),
            elapsed_ms,
            result.stats.saturated_outputs,
            wrong,
        );
        if wrong != 0 {
            return Err(format!("grouped W4A4 mismatch examples: {examples:?}").into());
        }
    }

    println!(
        "W4A4 GROUPED PASS K={} N={} G={} resident_mb={:.2} prepare_ms={:.3}",
        K,
        N,
        GROUP,
        prepared.stats().resident_bytes as f64 / (1024.0 * 1024.0),
        prepare_ms,
    );
    Ok(())
}
