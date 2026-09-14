use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_matmul::{Fp16MatmulExecutor, Fp16MatmulPool};
use std::{error::Error, sync::Arc};

fn data(m: usize, k: usize, n: usize, mut seed: u64) -> (Vec<f16>, Vec<f16>) {
    fn next(seed: &mut u64) -> u32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 32) as u32
    }
    let a = (0..m * k)
        .map(|_| f16::from_f32((next(&mut seed) % 257) as f32 / 128.0 - 1.0))
        .collect();
    let b = (0..n * k)
        .map(|_| f16::from_f32((next(&mut seed) % 257) as f32 / 128.0 - 1.0))
        .collect();
    (a, b)
}

fn main() -> Result<(), Box<dyn Error>> {
    let dev = RocketDevice::open()?;
    let mut ex = Fp16MatmulExecutor::new(&dev)?;

    let shapes = [
        (4usize, 256usize, 64usize),
        (64, 512, 128),
        (64, 1024, 256),
        (128, 512, 96),
    ];
    let mut datasets = Vec::new();
    for (i, &(m, k, n)) in shapes.iter().enumerate() {
        let (a, b) = data(m, k, n, 100 + i as u64);
        let _ = ex.execute(&a, &b, m, k, n)?;
        datasets.push((a, b));
    }
    let scratch_warm = ex.scratch_stats();
    for cycle in 0..128usize {
        let i = cycle % shapes.len();
        let (m, k, n) = shapes[i];
        let (a, b) = &datasets[i];
        let _ = ex.execute(a, b, m, k, n)?;
    }
    let scratch_after = ex.scratch_stats();
    if scratch_after != scratch_warm {
        return Err(format!(
            "shape-change scratch mutated after warmup: {scratch_warm:?} -> {scratch_after:?}"
        )
        .into());
    }
    println!(
        "lifecycle scratch PASS cycles=128 allocs={} grows={} capacities=[{},{},{},{},{}]",
        scratch_after.bo_allocations,
        scratch_after.bo_grows,
        scratch_after.regcmd_capacity,
        scratch_after.input_capacity,
        scratch_after.weights_capacity,
        scratch_after.output0_capacity,
        scratch_after.output1_capacity,
    );

    let resident_shapes = [(256usize, 64usize), (512, 128), (1024, 256), (4096, 128)];
    let mut min_dma = u64::MAX;
    let mut max_dma_end = 0u64;
    let mut unique = std::collections::BTreeSet::new();
    for cycle in 0..128usize {
        let (k, n) = resident_shapes[cycle % resident_shapes.len()];
        let (_, b) = data(4, k, n, 1000 + cycle as u64);
        let w = ex.prepack_weights_compatible_m(&b, k, n)?;
        let dma = w.dma_address();
        let bytes = w.stats().resident_bytes as u64;
        let end = dma
            .checked_add(bytes.saturating_sub(1))
            .ok_or("DMA range overflow")?;
        if end > u32::MAX as u64 {
            return Err(format!("resident DMA escaped 32-bit range: {dma:#x}..={end:#x}").into());
        }
        min_dma = min_dma.min(dma);
        max_dma_end = max_dma_end.max(end);
        unique.insert(dma);
        drop(w);
    }
    println!(
        "lifecycle resident churn PASS cycles=128 unique_dma={} dma_span={:#x}..={:#x}",
        unique.len(),
        min_dma,
        max_dma_end
    );

    const K: usize = 1024;
    const N: usize = 512;
    const COPIES: usize = 96;
    let (_, big_b) = data(4, K, N, 0x10a4_0512);
    let mut residents = Vec::with_capacity(COPIES);
    let mut pressure_min = u64::MAX;
    let mut pressure_max = 0u64;
    let mut pressure_bytes = 0usize;
    for _ in 0..COPIES {
        let w = ex.prepack_weights_compatible_m(&big_b, K, N)?;
        let dma = w.dma_address();
        let bytes = w.stats().resident_bytes;
        let end = dma + bytes as u64 - 1;
        if end > u32::MAX as u64 {
            return Err(format!("pressure resident above 4GiB: {dma:#x}..={end:#x}").into());
        }
        pressure_min = pressure_min.min(dma);
        pressure_max = pressure_max.max(end);
        pressure_bytes += bytes;
        residents.push(w);
    }
    println!(
        "lifecycle low4g pressure allocated copies={COPIES} resident_MB={:.1} dma_span={:#x}..={:#x}",
        pressure_bytes as f64 / 1e6,
        pressure_min,
        pressure_max
    );
    drop(residents);
    let w = ex.prepack_weights_compatible_m(&big_b, K, N)?;
    let (a, _) = data(64, K, N, 0xfeed_0512);
    let out = ex.execute_prepacked_compatible_m(&a, 64, &w)?;
    if out.values.len() != 64 * N {
        return Err("post-pressure resident execution wrong length".into());
    }
    println!(
        "lifecycle low4g pressure PASS post_release_dma={:#x} post_release_MB={:.3}",
        w.dma_address(),
        w.stats().resident_bytes as f64 / 1e6
    );
    drop(w);

    let mut pool = Fp16MatmulPool::new(3)?;
    let pool_shapes = [(256usize, 96usize), (512, 192), (1024, 384)];
    let mut released_probe = None;
    for cycle in 0..48usize {
        let (k, n) = pool_shapes[cycle % pool_shapes.len()];
        let (_, b) = data(4, k, n, 5000 + cycle as u64);
        let prepared = pool.prepare_weights_compatible_m(Arc::from(b), k, n)?;
        if cycle == 0 {
            let (a, _) = data(16, k, n, 6000);
            let _ = pool.execute_prepared(Arc::from(a.clone()), 16, &prepared)?;
            pool.release_prepared(&prepared)?;
            released_probe = Some((prepared, Arc::<[f16]>::from(a)));
            continue;
        }
        pool.release_prepared(&prepared)?;
    }
    if let Some((released, a)) = released_probe {
        if pool.execute_prepared(a, 16, &released).is_ok() {
            return Err("released pool resident handle remained executable".into());
        }
    }
    println!("lifecycle pool release PASS cycles=48 use_after_release_rejected=true");
    println!("PASS: bounded resident/scratch/low4g lifecycle stress");
    Ok(())
}
