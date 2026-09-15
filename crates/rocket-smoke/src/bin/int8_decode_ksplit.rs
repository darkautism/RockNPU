use rocket_runtime::RocketDevice;
use rocknpu_matmul::{
    Int8DecodeExecutor, Int8DecodeOutput, Int8DecodePool, Int8DecodeSplit, Int8PreparedWeightStats,
};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Instant;

const K: usize = 5632;
const N: usize = 2048;
const K_SLICES: &[(usize, usize)] = &[(0, 2048), (2048, 2048), (4096, 1536)];
const WARMUPS: usize = 3;
const REPS: usize = 9;

type DynError = Box<dyn std::error::Error>;
type SpawnedWorkers = (
    Vec<Worker>,
    mpsc::Receiver<WorkerResult>,
    Vec<Int8PreparedWeightStats>,
);

enum WorkerCommand {
    Run(Arc<[i8]>),
    Stop,
}

struct WorkerResult {
    worker: usize,
    output: Result<Int8DecodeOutput, String>,
}

struct Worker {
    tx: mpsc::Sender<WorkerCommand>,
    handle: thread::JoinHandle<()>,
}

fn make_data() -> (Arc<[i8]>, Arc<[i8]>, Vec<i32>) {
    let a: Vec<i8> = (0..K).map(|i| ((i * 7 + 3) % 9) as i8 - 4).collect();
    let b: Vec<i8> = (0..N * K).map(|i| ((i * 5 + 1) % 11) as i8 - 5).collect();
    let mut reference = vec![0i32; N];
    for col in 0..N {
        let mut sum = 0i32;
        for kk in 0..K {
            sum += i32::from(a[kk]) * i32::from(b[col * K + kk]);
        }
        reference[col] = sum;
    }
    (Arc::from(a), Arc::from(b), reference)
}

fn extract_weight_slice(weights: &[i8], k0: usize, ksub: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(N * ksub);
    for row in weights.chunks_exact(K) {
        out.extend_from_slice(&row[k0..k0 + ksub]);
    }
    out
}

fn verify(label: &str, actual: &[i32], expected: &[i32]) -> Result<(), DynError> {
    if actual.len() != expected.len() {
        return Err(format!("{label}: output length mismatch").into());
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

fn spawn_workers(weights: Arc<[i8]>) -> Result<SpawnedWorkers, DynError> {
    let (result_tx, result_rx) = mpsc::channel::<WorkerResult>();
    let (init_tx, init_rx) = mpsc::channel::<Result<(usize, Int8PreparedWeightStats), String>>();
    let mut workers = Vec::with_capacity(K_SLICES.len());

    for (worker, &(k0, ksub)) in K_SLICES.iter().enumerate() {
        let (tx, rx) = mpsc::channel::<WorkerCommand>();
        let result_tx = result_tx.clone();
        let init_tx = init_tx.clone();
        let weights = Arc::clone(&weights);
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
            let sliced = extract_weight_slice(&weights, k0, ksub);
            let prepared = match executor.prepare_weights(&sliced, ksub, N) {
                Ok(prepared) => prepared,
                Err(err) => {
                    let _ = init_tx.send(Err(format!("worker {worker}: prepare: {err}")));
                    return;
                }
            };
            if init_tx.send(Ok((worker, prepared.stats()))).is_err() {
                return;
            }

            while let Ok(command) = rx.recv() {
                match command {
                    WorkerCommand::Run(activation) => {
                        let output = executor
                            .execute_prepared(&activation[k0..k0 + ksub], &prepared)
                            .map_err(|err| format!("worker {worker}: execute: {err}"));
                        if result_tx.send(WorkerResult { worker, output }).is_err() {
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

    let mut stats = vec![Int8PreparedWeightStats::default(); K_SLICES.len()];
    for _ in 0..K_SLICES.len() {
        match init_rx.recv()? {
            Ok((worker, worker_stats)) if worker < stats.len() => stats[worker] = worker_stats,
            Ok((worker, _)) => return Err(format!("invalid worker init index {worker}").into()),
            Err(err) => return Err(err.into()),
        }
    }
    Ok((workers, result_rx, stats))
}

fn run_ksplit(
    workers: &[Worker],
    result_rx: &mpsc::Receiver<WorkerResult>,
    activation: Arc<[i8]>,
) -> Result<(Vec<i32>, f64, Vec<rocknpu_matmul::Int8DecodeStats>), DynError> {
    let start = Instant::now();
    for worker in workers {
        worker
            .tx
            .send(WorkerCommand::Run(Arc::clone(&activation)))?;
    }
    let mut values = vec![0i32; N];
    let mut stats = Vec::with_capacity(workers.len());
    for _ in 0..workers.len() {
        let result = result_rx.recv()?;
        if result.worker >= workers.len() {
            return Err("invalid worker result index".into());
        }
        let output = result.output.map_err(|err| -> DynError { err.into() })?;
        if output.values.len() != N {
            return Err("worker output length mismatch".into());
        }
        for (sum, value) in values.iter_mut().zip(output.values) {
            *sum = sum
                .checked_add(value)
                .ok_or("host int32 accumulation overflow")?;
        }
        stats.push(output.stats);
    }
    Ok((values, start.elapsed().as_secs_f64() * 1.0e3, stats))
}

fn main() -> Result<(), DynError> {
    let (activation, weights, reference) = make_data();
    let baseline_device = RocketDevice::open()?;
    let baseline_executor = Int8DecodeExecutor::new(&baseline_device)?;
    let baseline_weights = baseline_executor.prepare_weights(&weights, K, N)?;
    let (mut workers, result_rx, prepared_stats) = spawn_workers(Arc::clone(&weights))?;
    let mut pool = Int8DecodePool::new(3)?;
    let pool_weights =
        pool.prepare_weights_with_split(Arc::clone(&weights), K, N, 3, Int8DecodeSplit::K)?;

    for _ in 0..WARMUPS {
        let baseline = baseline_executor.execute_prepared(&activation, &baseline_weights)?;
        verify("baseline warmup", &baseline.values, &reference)?;
        let (output, _, _) = run_ksplit(&workers, &result_rx, Arc::clone(&activation))?;
        verify("K-split warmup", &output, &reference)?;
        let pool_output = pool.execute_prepared(Arc::clone(&activation), &pool_weights)?;
        verify("pool K-split warmup", &pool_output.values, &reference)?;
    }

    let mut baseline_samples = Vec::with_capacity(REPS);
    let mut ksplit_samples = Vec::with_capacity(REPS);
    let mut pool_samples = Vec::with_capacity(REPS);
    let mut last_stats = Vec::new();
    for rep in 0..REPS {
        let pool_start = Instant::now();
        let pool_output = pool.execute_prepared(Arc::clone(&activation), &pool_weights)?;
        let pool_wall_ms = pool_start.elapsed().as_secs_f64() * 1.0e3;
        verify("pool K-split sample", &pool_output.values, &reference)?;
        pool_samples.push(pool_wall_ms);
        if rep % 2 == 0 {
            let start = Instant::now();
            let output = baseline_executor.execute_prepared(&activation, &baseline_weights)?;
            verify("baseline sample", &output.values, &reference)?;
            baseline_samples.push(start.elapsed().as_secs_f64() * 1.0e3);

            let (output, wall_ms, stats) =
                run_ksplit(&workers, &result_rx, Arc::clone(&activation))?;
            verify("K-split sample", &output, &reference)?;
            ksplit_samples.push(wall_ms);
            last_stats = stats;
        } else {
            let (output, wall_ms, stats) =
                run_ksplit(&workers, &result_rx, Arc::clone(&activation))?;
            verify("K-split sample", &output, &reference)?;
            ksplit_samples.push(wall_ms);
            last_stats = stats;

            let start = Instant::now();
            let output = baseline_executor.execute_prepared(&activation, &baseline_weights)?;
            verify("baseline sample", &output.values, &reference)?;
            baseline_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        }
    }

    for worker in &workers {
        let _ = worker.tx.send(WorkerCommand::Stop);
    }
    for worker in workers.drain(..) {
        worker
            .handle
            .join()
            .map_err(|_| "K-split worker panicked")?;
    }

    let baseline_ms = median(baseline_samples);
    let ksplit_ms = median(ksplit_samples);
    let pool_ms = median(pool_samples);
    let speedup = baseline_ms / ksplit_ms;
    let pool_speedup = baseline_ms / pool_ms;
    let resident_bytes: usize = prepared_stats
        .iter()
        .map(|stats| stats.resident_bytes)
        .sum();
    let worker_breakdown: Vec<_> = last_stats
        .iter()
        .map(|stats| {
            (
                stats.pack_ns as f64 / 1.0e6,
                stats.submit_wait_ns as f64 / 1.0e6,
                stats.host_accum_ns as f64 / 1.0e6,
                stats.total_ns as f64 / 1.0e6,
            )
        })
        .collect();

    println!(
        "INT8 K-SPLIT DECODE PASS M=1 K={K} N={N} k_slices={K_SLICES:?} baseline_median_ms={baseline_ms:.3} prototype_ksplit_median_ms={ksplit_ms:.3} prototype_speedup={speedup:.2}x pool_ksplit_median_ms={pool_ms:.3} pool_speedup={pool_speedup:.2}x resident_mb={:.2} last_worker_breakdown_ms={worker_breakdown:?}",
        resident_bytes as f64 / (1024.0 * 1024.0),
    );
    Ok(())
}
