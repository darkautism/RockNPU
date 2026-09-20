use rocket_runtime::RocketDevice;
use rocknpu_matmul::Int8DecodeExecutor;
use std::time::Instant;

const M: usize = 16;
const K: usize = 5632;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let n = args.next().map_or(Ok(64usize), |s| s.parse::<usize>())?;
    let timing = matches!(args.next().as_deref(), Some("timing"));
    if args.next().is_some() || !n.is_multiple_of(32) {
        return Err("usage: int8_mtile_ksplit [N multiple of 32] [timing]".into());
    }

    let device = RocketDevice::open()?;
    let executor = Int8DecodeExecutor::new(&device)?;

    let mut a = vec![0i8; M * K];
    let mut b = vec![0i8; n * K];
    if timing {
        a.fill(1);
        b.fill(1);
    } else {
        for row in 0..M {
            for kk in 0..K {
                a[row * K + kk] = ((row * 7 + kk * 3 + 5) % 13) as i8 - 6;
            }
        }
        for col in 0..n {
            for kk in 0..K {
                b[col * K + kk] = ((col * 5 + kk * 11 + 3) % 17) as i8 - 8;
            }
        }
    }

    let prepared = executor.prepare_weights(&b, K, n)?;
    let mut samples_us = Vec::new();
    let mut latest = None;
    for _ in 0..if timing { 5 } else { 1 } {
        let start = Instant::now();
        let result = executor.execute_prepared_m16(&a, &prepared)?;
        samples_us.push(start.elapsed().as_secs_f64() * 1e6);
        latest = Some(result);
    }
    let result = latest.unwrap();

    if timing {
        let expected = i32::try_from(K)?;
        for (index, &got) in result.values.iter().enumerate() {
            if got != expected {
                return Err(format!("timing oracle mismatch index={index}: expected={expected} got={got}").into());
            }
        }
        println!(
            "INT8 MTILE KSPLIT TIMING PASS M={M} K={K} N={n} slices={} samples_us={samples_us:?} submit_wait_us={:.3} host_accum_us={:.3}",
            result.stats.k_slices,
            result.stats.submit_wait_ns as f64 / 1e3,
            result.stats.host_accum_ns as f64 / 1e3,
        );
        return Ok(());
    }

    for row in 0..M {
        let baseline = executor.execute_prepared(&a[row * K..(row + 1) * K], &prepared)?;
        for col in 0..n {
            let index = row * n + col;
            let expected = baseline.values[col];
            let got = result.values[index];
            if got != expected {
                return Err(format!(
                    "row={row} col={col}: M16={got} M1={expected}"
                ).into());
            }
        }
    }

    println!(
        "INT8 MTILE KSPLIT DIFFERENTIAL PASS M={M} K={K} N={n} slices={} m16_us={:.3} submit_wait_us={:.3} host_accum_us={:.3}",
        result.stats.k_slices,
        samples_us[0],
        result.stats.submit_wait_ns as f64 / 1e3,
        result.stats.host_accum_ns as f64 / 1e3,
    );
    Ok(())
}
