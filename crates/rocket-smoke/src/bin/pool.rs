use half::f16;
use rocknpu_matmul::{Fp16MatmulPool, cpu_reference_fp16};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

const M: usize = 256;
const K: usize = 384;
const N: usize = 768;

fn artifacts() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts")
}

fn make_data() -> (Arc<[f16]>, Arc<[f16]>) {
    let a: Vec<f16> = (0..M * K)
        .map(|i| f16::from_f32(((i * 7 + 3) % 5) as f32 - 2.0))
        .collect();
    let b: Vec<f16> = (0..N * K)
        .map(|i| f16::from_f32(((i * 3 + 1) % 5) as f32 - 2.0))
        .collect();
    (
        Arc::from(a.into_boxed_slice()),
        Arc::from(b.into_boxed_slice()),
    )
}

fn mismatch_count(a: &[f16], b: &[f16]) -> usize {
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

fn ms(ns: u128) -> f64 {
    ns as f64 / 1_000_000.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(artifacts())?;
    let (a, b) = make_data();
    let reference = cpu_reference_fp16(&a, &b, M, K, N)?;

    let mut pool3 = Fp16MatmulPool::new(3)?;
    let first = pool3.execute(Arc::clone(&a), Arc::clone(&b), M, K, N)?;
    let mm = mismatch_count(&first.values, &reference);
    if mm != 0 {
        return Err(format!("3-worker pool first-run mismatches={mm}").into());
    }
    if first.stats.workers_used != 3 {
        return Err(format!("expected 3 workers, got {}", first.stats.workers_used).into());
    }
    let allocs_before: Vec<usize> = first
        .stats
        .worker_scratch
        .iter()
        .map(|s| s.bo_allocations)
        .collect();

    let second = pool3.execute(Arc::clone(&a), Arc::clone(&b), M, K, N)?;
    let mm2 = mismatch_count(&second.values, &reference);
    if mm2 != 0 {
        return Err(format!("3-worker pool repeated-run mismatches={mm2}").into());
    }
    let allocs_after: Vec<usize> = second
        .stats
        .worker_scratch
        .iter()
        .map(|s| s.bo_allocations)
        .collect();
    if allocs_after != allocs_before {
        return Err(format!(
            "persistent pool allocated scratch after warmup: before={allocs_before:?} after={allocs_after:?}"
        )
        .into());
    }

    let mut pool1 = Fp16MatmulPool::new(1)?;
    let warm1 = pool1.execute(Arc::clone(&a), Arc::clone(&b), M, K, N)?;
    if mismatch_count(&warm1.values, &reference) != 0 {
        return Err("1-worker pool warm mismatch".into());
    }
    let serial = pool1.execute(Arc::clone(&a), Arc::clone(&b), M, K, N)?;
    if mismatch_count(&serial.values, &reference) != 0 {
        return Err("1-worker pool repeated mismatch".into());
    }

    let speedup = serial.stats.wall_ns as f64 / second.stats.wall_ns as f64;
    let worker_ms: Vec<f64> = second
        .stats
        .worker_timings
        .iter()
        .map(|t| ms(t.total_ns))
        .collect();
    let line = format!(
        "persistent-pool M{M} K{K} N{N} exact=PASS workers3={} jobs3={} wall3_ms={:.3} wall1_ms={:.3} observed_speedup={speedup:.2}x worker_total_ms={worker_ms:?} scratch_allocs={allocs_after:?} scratch_reuse=PASS",
        second.stats.workers_used,
        second.stats.jobs_submitted,
        ms(second.stats.wall_ns),
        ms(serial.stats.wall_ns),
    );
    println!("{line}");
    fs::write(artifacts().join("pool-run.txt"), &line)?;
    println!("PASS: persistent 3-worker Fp16MatmulPool hardware gate");
    Ok(())
}
