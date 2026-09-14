use half::f16;
use rocknpu_matmul::Fp16MatmulPool;
use std::{error::Error, sync::Arc};

const ROUNDS: usize = 6;
const ORDER: [usize; 6] = [1, 2, 3, 3, 2, 1];

fn data(m: usize, k: usize, n: usize, mut s: u64) -> (Arc<[f16]>, Arc<[f16]>) {
    fn next(s: &mut u64) -> u32 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 32) as u32
    }
    let a: Vec<f16> = (0..m * k)
        .map(|_| f16::from_f32((next(&mut s) % 2001) as f32 / 1000.0 - 1.0))
        .collect();
    let b: Vec<f16> = (0..n * k)
        .map(|_| f16::from_f32((next(&mut s) % 2001) as f32 / 1000.0 - 1.0))
        .collect();
    (Arc::from(a), Arc::from(b))
}

fn median(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn bench(
    pool: &mut Fp16MatmulPool,
    tag: &str,
    m: usize,
    k: usize,
    n: usize,
    seed: u64,
) -> Result<(), Box<dyn Error>> {
    let (a, b) = data(m, k, n, seed);
    // Warm every policy once so allocation/growth cost is not part of the medians.
    for workers in 1..=3 {
        let _ = pool.execute_with_workers(Arc::clone(&a), Arc::clone(&b), m, k, n, workers)?;
    }
    let mut samples: [Vec<u128>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut effective: [usize; 3] = [0; 3];
    for _ in 0..ROUNDS {
        for workers in ORDER {
            let out =
                pool.execute_with_workers(Arc::clone(&a), Arc::clone(&b), m, k, n, workers)?;
            samples[workers - 1].push(out.stats.wall_ns);
            effective[workers - 1] = out.stats.workers_used;
        }
    }
    let med = [
        median(samples[0].clone()),
        median(samples[1].clone()),
        median(samples[2].clone()),
    ];
    let mut best = 0usize;
    for i in 1..3 {
        if med[i] < med[best] {
            best = i;
        }
    }
    let ops = 2.0 * m as f64 * k as f64 * n as f64;
    println!(
        "{tag} M{m} K{k} N{n} effective={effective:?} med_ms=[{:.3},{:.3},{:.3}] gflops=[{:.2},{:.2},{:.2}] best_request={} best_effective={} speedup_vs_1={:.2}x",
        med[0] as f64 / 1e6,
        med[1] as f64 / 1e6,
        med[2] as f64 / 1e6,
        ops / med[0] as f64,
        ops / med[1] as f64,
        ops / med[2] as f64,
        best + 1,
        effective[best],
        med[0] as f64 / med[best] as f64,
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut pool = Fp16MatmulPool::new(3)?;
    // Actual padded shapes used by the dynamic MNIST model session.
    bench(&mut pool, "mnist-l1", 52, 800, 800, 1)?;
    bench(&mut pool, "mnist-l2", 52, 800, 800, 2)?;
    bench(&mut pool, "mnist-l3", 52, 800, 400, 3)?;
    bench(&mut pool, "mnist-l4", 52, 416, 16, 4)?;
    // Boundary cases where fan-out overhead can dominate.
    bench(&mut pool, "tiny-a", 4, 32, 48, 40)?;
    bench(&mut pool, "tiny-b", 4, 256, 48, 41)?;
    bench(&mut pool, "tiny-c", 4, 1024, 48, 42)?;
    bench(&mut pool, "small-a", 16, 256, 48, 43)?;
    bench(&mut pool, "small-b", 16, 256, 96, 44)?;
    bench(&mut pool, "small-c", 16, 1024, 96, 45)?;
    bench(&mut pool, "n32", 52, 416, 32, 46)?;
    bench(&mut pool, "n48", 52, 416, 48, 47)?;
    bench(&mut pool, "n64", 52, 416, 64, 48)?;
    bench(&mut pool, "mid48", 64, 256, 48, 49)?;
    bench(&mut pool, "mid96", 64, 256, 96, 50)?;
    // Previously proven scaling / deeper-K representative shapes.
    bench(&mut pool, "wide", 256, 384, 768, 5)?;
    bench(&mut pool, "deep128", 64, 4096, 128, 6)?;
    bench(&mut pool, "deep192", 64, 4096, 192, 61)?;
    bench(&mut pool, "deep256", 64, 4096, 256, 62)?;
    bench(&mut pool, "deep320", 64, 4096, 320, 63)?;
    bench(&mut pool, "mod128", 64, 1024, 128, 64)?;
    bench(&mut pool, "mod192", 64, 1024, 192, 65)?;
    bench(&mut pool, "mod256", 64, 1024, 256, 66)?;
    bench(&mut pool, "mod320", 64, 1024, 320, 67)?;
    bench(&mut pool, "deep512", 64, 4096, 512, 7)?;
    println!("PASS: interleaved worker-count policy probe");
    Ok(())
}
