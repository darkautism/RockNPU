use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_matmul::{Fp16MatmulExecutor, cpu_reference_fp16};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

const M: usize = 256;
const K: usize = 384;
const N: usize = 256;
const REPS: usize = 40;

fn artifacts() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts")
}

fn data() -> (Arc<Vec<f16>>, Arc<Vec<f16>>, Arc<Vec<f16>>) {
    let a: Vec<f16> = (0..M * K)
        .map(|i| f16::from_f32(((i * 7 + 3) % 5) as f32 - 2.0))
        .collect();
    let b: Vec<f16> = (0..N * K)
        .map(|i| f16::from_f32(((i * 3 + 1) % 5) as f32 - 2.0))
        .collect();
    let reference = cpu_reference_fp16(&a, &b, M, K, N).unwrap();
    (Arc::new(a), Arc::new(b), Arc::new(reference))
}

fn run_workers(workers: usize) -> Result<(f64, f64, Vec<f64>), Box<dyn std::error::Error>> {
    let (a, b, reference) = data();
    let barrier = Arc::new(Barrier::new(workers));
    let mut handles = Vec::new();
    for worker in 0..workers {
        let a = Arc::clone(&a);
        let b = Arc::clone(&b);
        let reference = Arc::clone(&reference);
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || -> Result<f64, String> {
            let dev = RocketDevice::open().map_err(|e| format!("worker {worker}: open: {e}"))?;
            let mut ex = Fp16MatmulExecutor::new(&dev)
                .map_err(|e| format!("worker {worker}: executor: {e}"))?;
            let weights = ex
                .prepack_weights(&b, M, K, N)
                .map_err(|e| format!("worker {worker}: prepack: {e}"))?;
            let warm = ex
                .execute_prepacked(&a, &weights)
                .map_err(|e| format!("worker {worker}: warm: {e}"))?;
            if warm.values != *reference {
                return Err(format!("worker {worker}: warm correctness mismatch"));
            }
            let allocs = ex.scratch_stats().bo_allocations;
            barrier.wait();
            let t0 = Instant::now();
            for rep in 0..REPS {
                let out = ex
                    .execute_prepacked(&a, &weights)
                    .map_err(|e| format!("worker {worker} rep {rep}: {e}"))?;
                if rep == REPS - 1 && out.values != *reference {
                    return Err(format!("worker {worker}: final correctness mismatch"));
                }
            }
            let elapsed = t0.elapsed().as_secs_f64();
            if ex.scratch_stats().bo_allocations != allocs {
                return Err(format!("worker {worker}: scratch allocated after warmup"));
            }
            Ok(elapsed)
        }));
    }
    let mut elapsed = Vec::new();
    for h in handles {
        match h.join() {
            Ok(Ok(v)) => elapsed.push(v),
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => return Err("resident multicore worker panicked".into()),
        }
    }
    let wall = elapsed.iter().copied().fold(0.0f64, f64::max);
    let ops_per_call = 2.0 * M as f64 * K as f64 * N as f64;
    let aggregate_gflops = ops_per_call * workers as f64 * REPS as f64 / wall / 1e9;
    Ok((wall, aggregate_gflops, elapsed))
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(artifacts())?;
    let _ = run_workers(3)?;
    let order = [3usize, 1, 2, 2, 1, 3, 1, 2, 3];
    let mut samples = [Vec::<f64>::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut lines = Vec::new();
    for workers in order {
        let (wall, gflops, elapsed) = run_workers(workers)?;
        samples[workers].push(gflops);
        let line = format!(
            "sample resident workers={workers} wall_ms={:.3} aggregate_GFLOPs={:.2} worker_seconds={elapsed:?}",
            wall * 1000.0,
            gflops
        );
        println!("{line}");
        lines.push(line);
    }
    let g1 = median(samples[1].clone());
    let g2 = median(samples[2].clone());
    let g3 = median(samples[3].clone());
    let summary = format!(
        "resident median shape=M{M} K{K} N{N} reps_per_worker={REPS} GFLOPs[w1,w2,w3]=[{g1:.2},{g2:.2},{g3:.2}] scaling[w2,w3]=[{:.2}x,{:.2}x]",
        g2 / g1,
        g3 / g1
    );
    println!("{summary}");
    lines.push(summary.clone());
    if g2 / g1 < 1.5 || g3 / g1 < 2.2 {
        return Err(
            format!("resident multicore scaling below conservative gate: {summary}").into(),
        );
    }
    println!("PASS: preconditioned resident-weight multicore RK3588 scaling probe; {summary}");
    fs::write(
        artifacts().join("multicore-prepacked.txt"),
        lines.join("\n") + "\n",
    )?;
    Ok(())
}
