use half::f16;
use rocknpu_matmul::{Fp16MatmulPool, cpu_reference_fp32};
use std::{error::Error, sync::Arc};

fn main() -> Result<(), Box<dyn Error>> {
    let mut pool = Fp16MatmulPool::new(3)?;
    for (m, k, n) in [
        (4, 256, 96),
        (64, 4096, 192),
        (512, 2048, 2048),
        (512, 5632, 2048),
    ] {
        let weights: Arc<[f16]> = (0..k * n)
            .map(|i| f16::from_f32(((i * 17 + i / k * 7) % 31) as f32 / 32.0 - 0.5))
            .collect();
        let prepared = pool.prepare_weights_f32(Arc::clone(&weights), m, k, n)?;
        assert!(
            pool.execute_prepared_f32(Arc::from(vec![f16::ZERO; 4 * k]), 4, &prepared)
                .is_err()
                || m == 4
        );
        for variant in 0..3 {
            let a: Arc<[f16]> = (0..m * k)
                .map(|i| {
                    let v = ((i * 13 + i / k * 3) % 17) as f32 / 16.0 - 0.5;
                    f16::from_f32(match variant {
                        0 => v,
                        1 => -v,
                        _ => 0.0,
                    })
                })
                .collect();
            let ordinary = pool.execute_f32(Arc::clone(&a), Arc::clone(&weights), m, k, n)?;
            let cached = pool.execute_prepared_f32(Arc::clone(&a), m, &prepared)?;
            if ordinary
                .values
                .iter()
                .zip(&cached.values)
                .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                return Err(
                    format!("cached output differs M{m} K{k} N{n} variant={variant}").into(),
                );
            }
            if m <= 64 {
                let reference = cpu_reference_fp32(&a, &weights, m, k, n)?;
                if reference.iter().zip(&cached.values).any(|(a, b)| a != b) {
                    return Err("dyadic inputs must match independent CPU oracle exactly".into());
                }
            }
            println!(
                "PREFILL PASS M={m} K={k} N={n} variant={variant} ordinary_ms={:.3} cached_ms={:.3} resident_bytes={} jobs={}",
                ordinary.stats.wall_ns as f64 / 1e6,
                cached.stats.wall_ns as f64 / 1e6,
                prepared.stats().resident_bytes,
                cached.stats.jobs_submitted
            );
        }
        pool.release_prepared(&prepared)?;
        assert!(
            pool.execute_prepared_f32(Arc::from(vec![f16::ZERO; m * k]), m, &prepared)
                .is_err()
        );
    }
    println!("PASS: resident FP32 prefill, changed inputs, shape rejection and release");
    Ok(())
}
