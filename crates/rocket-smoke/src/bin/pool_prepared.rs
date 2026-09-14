use half::f16;
use rocknpu_matmul::{Fp16MatmulPool, cpu_reference_executor_semantics};
use std::error::Error;
use std::sync::Arc;

const K: usize = 384;
const N: usize = 768;
const REPS: usize = 9;

fn data(m: usize) -> Arc<[f16]> {
    let mut s = 0x3588_1234_5678_9abcu64 ^ m as u64;
    Arc::from(
        (0..m * K)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                f16::from_f32((s % 17) as f32 / 8.0 - 1.0)
            })
            .collect::<Vec<_>>(),
    )
}

fn weights() -> Arc<[f16]> {
    let mut s = 0x8bad_f00d_cafe_3588u64;
    Arc::from(
        (0..N * K)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                f16::from_f32((s % 17) as f32 / 8.0 - 1.0)
            })
            .collect::<Vec<_>>(),
    )
}

fn median(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn exact(got: &[f16], expected: &[f16], tag: &str) -> Result<(), Box<dyn Error>> {
    let mismatches = got
        .iter()
        .zip(expected)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    if mismatches != 0 {
        return Err(format!("{tag}: mismatches={mismatches}").into());
    }
    Ok(())
}

fn scratch_signature(out: &rocknpu_matmul::Fp16MatmulPoolOutput) -> Vec<(usize, usize)> {
    out.stats
        .worker_scratch
        .iter()
        .map(|s| (s.bo_allocations, s.bo_grows))
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let b = weights();
    let mut pool = Fp16MatmulPool::new(3)?;
    let prepared = pool.prepare_weights_compatible_m(Arc::clone(&b), K, N)?;
    let ps = prepared.stats();
    if ps.workers != 3 {
        return Err(format!("expected 3 prepared workers, got {}", ps.workers).into());
    }

    // Warm largest M so all worker scratch reaches the maximum size used below.
    let a256 = data(256);
    let warm = pool.execute_prepared(Arc::clone(&a256), 256, &prepared)?;
    let expected256 = cpu_reference_executor_semantics(&a256, &b, 256, K, N)?;
    exact(&warm.values, &expected256, "warm M256")?;
    let scratch_before = scratch_signature(&warm);

    let mut case_lines = Vec::new();
    for m in [64usize, 128, 256] {
        let a = if m == 256 { Arc::clone(&a256) } else { data(m) };
        let out = pool.execute_prepared(Arc::clone(&a), m, &prepared)?;
        let expected = cpu_reference_executor_semantics(&a, &b, m, K, N)?;
        exact(&out.values, &expected, &format!("prepared M{m}"))?;
        case_lines.push(format!(
            "M{m} exact=true workers={} jobs={} wall_ms={:.3}",
            out.stats.workers_used,
            out.stats.jobs_submitted,
            out.stats.wall_ns as f64 / 1e6,
        ));
    }

    let after = pool.execute_prepared(Arc::clone(&a256), 256, &prepared)?;
    exact(&after.values, &expected256, "repeat M256")?;
    let scratch_after = scratch_signature(&after);
    if scratch_after != scratch_before {
        return Err(format!(
            "worker scratch changed after max-M warmup: {scratch_before:?} -> {scratch_after:?}"
        )
        .into());
    }

    let mut stream = Vec::with_capacity(REPS);
    let mut resident = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let out = pool.execute(Arc::clone(&a256), Arc::clone(&b), 256, K, N)?;
        exact(&out.values, &expected256, "stream benchmark")?;
        stream.push(out.stats.wall_ns);
    }
    for _ in 0..REPS {
        let out = pool.execute_prepared(Arc::clone(&a256), 256, &prepared)?;
        exact(&out.values, &expected256, "resident benchmark")?;
        resident.push(out.stats.wall_ns);
    }
    let st = median(stream);
    let rt = median(resident);

    pool.release_prepared(&prepared)?;

    println!(
        "prepared pool PASS workers={} resident_MB={:.3} tiles={} prepare_wall_ms={:.3} pack_sum_ms={:.3} pack_max_ms={:.3} stream_median_ms={:.3} prepared_median_ms={:.3} speedup={:.2}x scratch={:?}",
        ps.workers,
        ps.resident_bytes as f64 / 1e6,
        ps.unique_tiles,
        ps.prepare_wall_ns as f64 / 1e6,
        ps.pack_ns_sum as f64 / 1e6,
        ps.pack_ns_max as f64 / 1e6,
        st as f64 / 1e6,
        rt as f64 / 1e6,
        st as f64 / rt as f64,
        scratch_after,
    );
    for line in case_lines {
        println!("{line}");
    }
    println!("release=PASS");
    Ok(())
}
