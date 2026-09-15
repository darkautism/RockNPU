use rocket_runtime::RocketDevice;
use rocknpu_matmul::{Int8DecodeExecutor, Int8DecodeOutput, Int8PreparedWeightStats};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Instant;

const DEFAULT_K: usize = 2048;
const DEFAULT_N: usize = 5632;
const DEFAULT_WORKERS: usize = 3;
const WARMUPS: usize = 3;
const REPS: usize = 9;

type DynError = Box<dyn std::error::Error>;
type SpawnedWorkers = (
    Vec<Worker>,
    mpsc::Receiver<WorkerResult>,
    Vec<Int8PreparedWeightStats>,
);
type MulticoreRun = (Vec<i32>, f64, Vec<Int8DecodeOutput>);

enum WorkerCommand {
    Run(Arc<[i8]>),
    Stop,
}

struct WorkerResult {
    worker: usize,
    n0: usize,
    nsub: usize,
    output: Result<Int8DecodeOutput, String>,
}

struct Worker {
    tx: mpsc::Sender<WorkerCommand>,
    handle: thread::JoinHandle<()>,
}

fn split_n(n: usize, workers: usize) -> Result<Vec<(usize, usize)>, Box<dyn std::error::Error>> {
    if n == 0 || !n.is_multiple_of(32) || workers == 0 {
        return Err("N must be non-zero/32-aligned and worker count non-zero".into());
    }
    let blocks = n / 32;
    if blocks < workers {
        return Err("N has fewer 32-channel blocks than workers".into());
    }
    let base = blocks / workers;
    let extra = blocks % workers;
    let mut out = Vec::with_capacity(workers);
    let mut n0 = 0usize;
    for worker in 0..workers {
        let worker_blocks = base + usize::from(worker < extra);
        let nsub = worker_blocks * 32;
        out.push((n0, nsub));
        n0 += nsub;
    }
    if n0 != n {
        return Err("N split did not cover the full output".into());
    }
    Ok(out)
}

fn make_data(k: usize, n: usize) -> (Arc<[i8]>, Arc<[i8]>, Vec<i32>) {
    let a: Vec<i8> = (0..k).map(|i| ((i * 7 + 3) % 9) as i8 - 4).collect();
    let b: Vec<i8> = (0..n * k).map(|i| ((i * 5 + 1) % 11) as i8 - 5).collect();
    let mut reference = vec![0i32; n];
    for col in 0..n {
        let mut sum = 0i32;
        for kk in 0..k {
            sum += i32::from(a[kk]) * i32::from(b[col * k + kk]);
        }
        reference[col] = sum;
    }
    (Arc::from(a), Arc::from(b), reference)
}

fn verify(label: &str, actual: &[i32], expected: &[i32]) -> Result<(), Box<dyn std::error::Error>> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label}: output length {} != {}",
            actual.len(),
            expected.len()
        )
        .into());
    }
    let mut mismatches = Vec::new();
    for (index, (&want, &got)) in expected.iter().zip(actual).enumerate() {
        if want != got && mismatches.len() < 10 {
            mismatches.push((index, want, got));
        }
    }
    if !mismatches.is_empty() {
        return Err(format!("{label}: mismatch examples: {mismatches:?}").into());
    }
    Ok(())
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite timing"));
    values[values.len() / 2]
}

fn spawn_workers(
    b: Arc<[i8]>,
    k: usize,
    slices: &[(usize, usize)],
) -> Result<SpawnedWorkers, DynError> {
    let (result_tx, result_rx) = mpsc::channel::<WorkerResult>();
    let (init_tx, init_rx) = mpsc::channel::<Result<(usize, Int8PreparedWeightStats), String>>();
    let mut workers = Vec::with_capacity(slices.len());

    for (worker, &(n0, nsub)) in slices.iter().enumerate() {
        let (tx, rx) = mpsc::channel::<WorkerCommand>();
        let result_tx = result_tx.clone();
        let init_tx = init_tx.clone();
        let b = Arc::clone(&b);
        let handle = thread::spawn(move || {
            let device = match RocketDevice::open() {
                Ok(device) => device,
                Err(err) => {
                    let _ = init_tx.send(Err(format!("worker {worker}: open: {err}")));
                    return;
                }
            };
            let executor = match Int8DecodeExecutor::new(&device) {
                Ok(executor) => executor,
                Err(err) => {
                    let _ = init_tx.send(Err(format!("worker {worker}: executor: {err}")));
                    return;
                }
            };
            let begin = match n0.checked_mul(k) {
                Some(begin) => begin,
                None => {
                    let _ = init_tx.send(Err(format!("worker {worker}: weight offset overflow")));
                    return;
                }
            };
            let end = match n0.checked_add(nsub).and_then(|v| v.checked_mul(k)) {
                Some(end) => end,
                None => {
                    let _ = init_tx.send(Err(format!("worker {worker}: weight end overflow")));
                    return;
                }
            };
            let weights = match executor.prepare_weights(&b[begin..end], k, nsub) {
                Ok(weights) => weights,
                Err(err) => {
                    let _ = init_tx.send(Err(format!("worker {worker}: prepare: {err}")));
                    return;
                }
            };
            if init_tx.send(Ok((worker, weights.stats()))).is_err() {
                return;
            }

            while let Ok(command) = rx.recv() {
                match command {
                    WorkerCommand::Run(a) => {
                        let output = executor
                            .execute_prepared(&a, &weights)
                            .map_err(|e| format!("worker {worker}: execute: {e}"));
                        if result_tx
                            .send(WorkerResult {
                                worker,
                                n0,
                                nsub,
                                output,
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                    WorkerCommand::Stop => break,
                }
            }
        });
        workers.push(Worker { tx, handle });
    }
    drop(init_tx);
    drop(result_tx);

    let mut stats = vec![Int8PreparedWeightStats::default(); slices.len()];
    for _ in 0..slices.len() {
        match init_rx.recv()? {
            Ok((worker, worker_stats)) if worker < stats.len() => stats[worker] = worker_stats,
            Ok((worker, _)) => return Err(format!("invalid worker init index {worker}").into()),
            Err(err) => return Err(err.into()),
        }
    }
    Ok((workers, result_rx, stats))
}

fn run_multicore(
    workers: &[Worker],
    result_rx: &mpsc::Receiver<WorkerResult>,
    a: Arc<[i8]>,
    n: usize,
) -> Result<MulticoreRun, DynError> {
    let start = Instant::now();
    for worker in workers {
        worker.tx.send(WorkerCommand::Run(Arc::clone(&a)))?;
    }

    let mut values = vec![0i32; n];
    let mut worker_outputs = Vec::with_capacity(workers.len());
    for _ in 0..workers.len() {
        let result = result_rx.recv()?;
        if result.worker >= workers.len() || result.n0 + result.nsub > n {
            return Err("invalid worker result metadata".into());
        }
        let output = result.output.map_err(|e| -> DynError { e.into() })?;
        if output.values.len() != result.nsub {
            return Err(format!("worker {} returned wrong output length", result.worker).into());
        }
        values[result.n0..result.n0 + result.nsub].copy_from_slice(&output.values);
        worker_outputs.push(output);
    }
    Ok((
        values,
        start.elapsed().as_secs_f64() * 1.0e3,
        worker_outputs,
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let k = args.next().map_or(Ok(DEFAULT_K), |v| v.parse::<usize>())?;
    let n = args.next().map_or(Ok(DEFAULT_N), |v| v.parse::<usize>())?;
    let worker_count = args
        .next()
        .map_or(Ok(DEFAULT_WORKERS), |v| v.parse::<usize>())?;
    if args.next().is_some() || !(1..=3).contains(&worker_count) {
        return Err("usage: int8_decode_multicore [K] [N] [workers=1..3]".into());
    }

    let slices = split_n(n, worker_count)?;
    let (a, b, reference) = make_data(k, n);

    let single_device = RocketDevice::open()?;
    let single_executor = Int8DecodeExecutor::new(&single_device)?;
    let single_weights = single_executor.prepare_weights(&b, k, n)?;

    let (mut workers, result_rx, multicore_prepare) = spawn_workers(Arc::clone(&b), k, &slices)?;

    for _ in 0..WARMUPS {
        let single = single_executor.execute_prepared(&a, &single_weights)?;
        verify("single warmup", &single.values, &reference)?;
        let (multi, _, _) = run_multicore(&workers, &result_rx, Arc::clone(&a), n)?;
        verify("multicore warmup", &multi, &reference)?;
    }

    let mut single_samples = Vec::with_capacity(REPS);
    let mut single_pack_samples = Vec::with_capacity(REPS);
    let mut single_submit_samples = Vec::with_capacity(REPS);
    let mut single_accum_samples = Vec::with_capacity(REPS);
    let mut single_other_samples = Vec::with_capacity(REPS);
    let mut multi_samples = Vec::with_capacity(REPS);
    let mut last_worker_stats = Vec::new();
    for rep in 0..REPS {
        if rep % 2 == 0 {
            let start = Instant::now();
            let output = single_executor.execute_prepared(&a, &single_weights)?;
            let wall_ms = start.elapsed().as_secs_f64() * 1.0e3;
            verify("single sample", &output.values, &reference)?;
            single_samples.push(wall_ms);
            single_pack_samples.push(output.stats.pack_ns as f64 / 1.0e6);
            single_submit_samples.push(output.stats.submit_wait_ns as f64 / 1.0e6);
            single_accum_samples.push(output.stats.host_accum_ns as f64 / 1.0e6);
            let accounted = output
                .stats
                .pack_ns
                .saturating_add(output.stats.submit_wait_ns)
                .saturating_add(output.stats.host_accum_ns);
            single_other_samples
                .push(output.stats.total_ns.saturating_sub(accounted) as f64 / 1.0e6);

            let (output, wall_ms, worker_outputs) =
                run_multicore(&workers, &result_rx, Arc::clone(&a), n)?;
            verify("multicore sample", &output, &reference)?;
            multi_samples.push(wall_ms);
            last_worker_stats = worker_outputs.iter().map(|output| output.stats).collect();
        } else {
            let (output, wall_ms, worker_outputs) =
                run_multicore(&workers, &result_rx, Arc::clone(&a), n)?;
            verify("multicore sample", &output, &reference)?;
            multi_samples.push(wall_ms);
            last_worker_stats = worker_outputs.iter().map(|output| output.stats).collect();

            let start = Instant::now();
            let output = single_executor.execute_prepared(&a, &single_weights)?;
            let wall_ms = start.elapsed().as_secs_f64() * 1.0e3;
            verify("single sample", &output.values, &reference)?;
            single_samples.push(wall_ms);
            single_pack_samples.push(output.stats.pack_ns as f64 / 1.0e6);
            single_submit_samples.push(output.stats.submit_wait_ns as f64 / 1.0e6);
            single_accum_samples.push(output.stats.host_accum_ns as f64 / 1.0e6);
            let accounted = output
                .stats
                .pack_ns
                .saturating_add(output.stats.submit_wait_ns)
                .saturating_add(output.stats.host_accum_ns);
            single_other_samples
                .push(output.stats.total_ns.saturating_sub(accounted) as f64 / 1.0e6);
        }
    }

    for worker in &workers {
        let _ = worker.tx.send(WorkerCommand::Stop);
    }
    for worker in workers.drain(..) {
        worker
            .handle
            .join()
            .map_err(|_| "multicore worker panicked")?;
    }

    let single_median_ms = median(single_samples);
    let single_pack_median_ms = median(single_pack_samples);
    let single_submit_median_ms = median(single_submit_samples);
    let single_accum_median_ms = median(single_accum_samples);
    let single_other_median_ms = median(single_other_samples);
    let multi_median_ms = median(multi_samples);
    let speedup = single_median_ms / multi_median_ms;
    let resident_bytes: usize = multicore_prepare
        .iter()
        .map(|stats| stats.resident_bytes)
        .sum();
    let prepare_wall_equivalent_ms = multicore_prepare
        .iter()
        .map(|stats| stats.pack_ns as f64 / 1.0e6)
        .fold(0.0f64, f64::max);

    let last_worker_breakdown_ms: Vec<_> = last_worker_stats
        .iter()
        .map(|stats| {
            let accounted = stats
                .pack_ns
                .saturating_add(stats.submit_wait_ns)
                .saturating_add(stats.host_accum_ns);
            (
                stats.pack_ns as f64 / 1.0e6,
                stats.submit_ns as f64 / 1.0e6,
                stats.wait_ns as f64 / 1.0e6,
                stats.submit_wait_ns as f64 / 1.0e6,
                stats.host_accum_ns as f64 / 1.0e6,
                stats.total_ns.saturating_sub(accounted) as f64 / 1.0e6,
                stats.total_ns as f64 / 1.0e6,
            )
        })
        .collect();
    println!(
        "INT8 MULTICORE DECODE PASS M=1 K={k} N={n} slices={slices:?} single_median_ms={single_median_ms:.3} single_breakdown_ms=[pack:{single_pack_median_ms:.3},submit:{single_submit_median_ms:.3},accum:{single_accum_median_ms:.3},other:{single_other_median_ms:.3}] multicore_median_ms={multi_median_ms:.3} speedup={speedup:.2}x resident_mb={:.2} max_worker_prepare_ms={prepare_wall_equivalent_ms:.3} last_worker_breakdown_ms={last_worker_breakdown_ms:?}",
        resident_bytes as f64 / (1024.0 * 1024.0),
    );

    if speedup <= 1.0 {
        return Err(format!(
            "{worker_count}-fd N-split did not beat single-fd resident decode: {speedup:.2}x"
        )
        .into());
    }
    Ok(())
}
