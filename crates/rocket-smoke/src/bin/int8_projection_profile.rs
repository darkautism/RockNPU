use rocknpu_matmul::{Int8DecodePool, Int8DecodeSplit};
use std::sync::Arc;
use std::time::Instant;

const WARMUPS: usize = 4;
const REPS: usize = 21;

type DynError = Box<dyn std::error::Error>;

fn make_data(k: usize, n: usize) -> (Arc<[i8]>, Arc<[i8]>, Vec<i32>) {
    let a: Vec<i8> = (0..k).map(|i| ((i * 7 + 3) % 9) as i8 - 4).collect();
    let b: Vec<i8> = (0..n * k).map(|i| ((i * 5 + 1) % 11) as i8 - 5).collect();
    let mut reference = vec![0i32; n];
    for col in 0..n {
        let row = &b[col * k..(col + 1) * k];
        reference[col] = a
            .iter()
            .zip(row)
            .map(|(&x, &y)| i32::from(x) * i32::from(y))
            .sum();
    }
    (Arc::from(a), Arc::from(b), reference)
}

fn verify(actual: &[i32], expected: &[i32]) -> Result<(), DynError> {
    if actual == expected {
        return Ok(());
    }
    let bad: Vec<_> = actual
        .iter()
        .zip(expected)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .take(8)
        .map(|(i, (&a, &b))| (i, a, b))
        .collect();
    Err(format!("mismatch {bad:?}").into())
}

fn percentile(v: &[u128], num: usize, den: usize) -> f64 {
    let i = ((v.len() - 1) * num) / den;
    v[i] as f64 / 1e6
}

fn bench(k: usize, n: usize, split: Int8DecodeSplit, workers: usize) -> Result<(), DynError> {
    let (a, b, reference) = make_data(k, n);
    let mut pool = Int8DecodePool::new(3)?;
    let prep0 = Instant::now();
    let prepared = pool.prepare_weights_with_split(Arc::clone(&b), k, n, workers, split)?;
    let prep_ms = prep0.elapsed().as_secs_f64() * 1e3;
    for _ in 0..WARMUPS {
        let out = pool.execute_prepared(Arc::clone(&a), &prepared)?;
        verify(&out.values, &reference)?;
    }
    let mut samples = Vec::with_capacity(REPS);
    let mut last = None;
    for _ in 0..REPS {
        let t0 = Instant::now();
        let out = pool.execute_prepared(Arc::clone(&a), &prepared)?;
        let ns = t0.elapsed().as_nanos();
        verify(&out.values, &reference)?;
        samples.push(ns);
        last = Some(out.stats);
    }
    samples.sort_unstable();
    let med_ms = percentile(&samples, 1, 2);
    let p10_ms = percentile(&samples, 1, 10);
    let p90_ms = percentile(&samples, 9, 10);
    let gbps = (k as f64 * n as f64) / (med_ms * 1e6);
    let st = last.unwrap();
    let phases = st.worker_stats.iter().map(|s| format!(
        "alloc={:.1} input={:.1} partial={:.1} regcmd={:.1} submit={:.1} wait={:.1} fini={:.1} accum={:.1} total={:.1}us",
        s.alloc_ns as f64 / 1e3,
        s.input_stage_ns as f64 / 1e3,
        s.partial_stage_ns as f64 / 1e3,
        s.regcmd_stage_ns as f64 / 1e3,
        s.submit_ns as f64 / 1e3,
        s.wait_ns as f64 / 1e3,
        s.output_fini_ns as f64 / 1e3,
        s.host_accum_ns as f64 / 1e3,
        s.total_ns as f64 / 1e3,
    )).collect::<Vec<_>>();
    println!(
        "PROFILE PASS K={k} N={n} split={split:?} req_workers={workers} used={} tasks={} prep_ms={prep_ms:.3} median_ms={med_ms:.3} p10_ms={p10_ms:.3} p90_ms={p90_ms:.3} weight_GBps={gbps:.2} worker_total_ms={:?} phases={phases:?}",
        st.workers_used,
        st.npu_tasks,
        st.worker_total_ns
            .iter()
            .map(|&x| x as f64 / 1e6)
            .collect::<Vec<_>>()
    );
    Ok(())
}

fn main() -> Result<(), DynError> {
    if std::env::var_os("ROCKNPU_W8_DIRECT_SUBMIT").is_none() {
        return Err("set ROCKNPU_W8_DIRECT_SUBMIT=1".into());
    }
    for &(k, n) in &[
        (2048usize, 2560usize),
        (2048, 2048),
        (2048, 11264),
        (5632, 2048),
    ] {
        let first_n_workers = if n > 8192 { 2 } else { 1 };
        for workers in first_n_workers..=3 {
            bench(k, n, Int8DecodeSplit::N, workers)?;
        }
        if k > 4096 {
            for workers in 2..=3 {
                bench(k, n, Int8DecodeSplit::K, workers)?;
            }
        }
    }
    Ok(())
}
