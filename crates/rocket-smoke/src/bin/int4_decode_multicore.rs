use rocket_runtime::RocketDevice;
use rocknpu_matmul::{Int4DecodeExecutor, Int4DecodeOutput};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Instant;

const K: usize = 2048;
const N: usize = 5632;
const WARMUPS: usize = 3;
const REPS: usize = 9;

type DynError = Box<dyn std::error::Error>;

enum Command {
    Run(Arc<[i8]>),
    Stop,
}

struct WorkerResult {
    worker: usize,
    n0: usize,
    output: Result<Int4DecodeOutput, String>,
}

struct Worker {
    tx: mpsc::Sender<Command>,
    handle: thread::JoinHandle<()>,
}

fn split_n(n: usize, workers: usize) -> Result<Vec<(usize, usize)>, DynError> {
    if n == 0 || !n.is_multiple_of(64) || workers == 0 {
        return Err("N must be non-zero/64-aligned and workers non-zero".into());
    }
    let blocks = n / 64;
    if blocks < workers {
        return Err("not enough 64-channel blocks".into());
    }
    let base = blocks / workers;
    let extra = blocks % workers;
    let mut out = Vec::with_capacity(workers);
    let mut n0 = 0usize;
    for worker in 0..workers {
        let nsub = (base + usize::from(worker < extra)) * 64;
        out.push((n0, nsub));
        n0 += nsub;
    }
    Ok(out)
}

fn make_data() -> (Arc<[i8]>, Arc<[i8]>, Vec<i16>) {
    let a: Vec<i8> = (0..K).map(|i| ((i * 7 + 3) % 5) as i8 - 2).collect();
    let b: Vec<i8> = (0..N * K).map(|i| ((i * 5 + 1) % 5) as i8 - 2).collect();
    let mut reference = vec![0i16; N];
    for n in 0..N {
        let mut sum = 0i32;
        for k in 0..K {
            sum += i32::from(a[k]) * i32::from(b[n * K + k]);
        }
        reference[n] = i16::try_from(sum).expect("test data fits int16");
    }
    (Arc::from(a), Arc::from(b), reference)
}

fn verify(label: &str, actual: &[i16], expected: &[i16]) -> Result<(), DynError> {
    if actual == expected {
        return Ok(());
    }
    let examples = expected
        .iter()
        .zip(actual)
        .enumerate()
        .filter(|(_, (want, got))| want != got)
        .take(8)
        .map(|(i, (&want, &got))| (i, want, got))
        .collect::<Vec<_>>();
    Err(format!("{label}: mismatch examples {examples:?}").into())
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).expect("finite timing"));
    samples[samples.len() / 2]
}

fn spawn_workers(
    b: Arc<[i8]>,
    slices: &[(usize, usize)],
) -> Result<(Vec<Worker>, mpsc::Receiver<WorkerResult>), DynError> {
    let (result_tx, result_rx) = mpsc::channel();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<usize, String>>();
    let mut workers = Vec::with_capacity(slices.len());
    for (worker, &(n0, nsub)) in slices.iter().enumerate() {
        let (tx, rx) = mpsc::channel();
        let result_tx = result_tx.clone();
        let ready_tx = ready_tx.clone();
        let b = Arc::clone(&b);
        let handle = thread::spawn(move || {
            let init = (|| -> Result<_, String> {
                let device = RocketDevice::open().map_err(|e| e.to_string())?;
                let executor = Int4DecodeExecutor::new(&device);
                let begin = n0.checked_mul(K).ok_or("weight offset overflow")?;
                let end = (n0 + nsub).checked_mul(K).ok_or("weight end overflow")?;
                let prepared = executor
                    .prepare_weights(&b[begin..end], K, nsub)
                    .map_err(|e| e.to_string())?;
                Ok((device, prepared))
            })();
            let (device, prepared) = match init {
                Ok(pair) => pair,
                Err(err) => {
                    let _ = ready_tx.send(Err(format!("worker {worker}: {err}")));
                    return;
                }
            };
            let executor = Int4DecodeExecutor::new(&device);
            if ready_tx.send(Ok(worker)).is_err() {
                return;
            }
            while let Ok(command) = rx.recv() {
                match command {
                    Command::Run(a) => {
                        let output = executor
                            .execute_prepared(&a, &prepared)
                            .map_err(|e| format!("worker {worker}: {e}"));
                        if result_tx.send(WorkerResult { worker, n0, output }).is_err() {
                            break;
                        }
                    }
                    Command::Stop => break,
                }
            }
        });
        workers.push(Worker { tx, handle });
    }
    drop(result_tx);
    drop(ready_tx);
    for _ in 0..workers.len() {
        ready_rx.recv()?.map_err(|e| -> DynError { e.into() })?;
    }
    Ok((workers, result_rx))
}

fn run_multi(
    workers: &[Worker],
    rx: &mpsc::Receiver<WorkerResult>,
    a: Arc<[i8]>,
    slices: &[(usize, usize)],
) -> Result<(Vec<i16>, f64), DynError> {
    let start = Instant::now();
    for worker in workers {
        worker.tx.send(Command::Run(Arc::clone(&a)))?;
    }
    let mut values = vec![0i16; N];
    for _ in workers {
        let result = rx.recv()?;
        if result.worker >= workers.len() {
            return Err("invalid worker id".into());
        }
        let nsub = slices[result.worker].1;
        let output = result.output.map_err(|e| -> DynError { e.into() })?;
        if output.values.len() != nsub {
            return Err("worker output length mismatch".into());
        }
        values[result.n0..result.n0 + nsub].copy_from_slice(&output.values);
    }
    Ok((values, start.elapsed().as_secs_f64() * 1.0e3))
}

fn main() -> Result<(), DynError> {
    let workers_requested = std::env::args()
        .nth(1)
        .map_or(Ok(3usize), |v| v.parse::<usize>())?;
    if !(1..=3).contains(&workers_requested) {
        return Err("usage: int4_decode_multicore [workers=1..3]".into());
    }
    let slices = split_n(N, workers_requested)?;
    let (a, b, reference) = make_data();

    let device = RocketDevice::open()?;
    let single_executor = Int4DecodeExecutor::new(&device);
    let single_prepared = single_executor.prepare_weights(&b, K, N)?;
    let (mut workers, rx) = spawn_workers(Arc::clone(&b), &slices)?;

    for _ in 0..WARMUPS {
        let single = single_executor.execute_prepared(&a, &single_prepared)?;
        verify("single warmup", &single.values, &reference)?;
        let (multi, _) = run_multi(&workers, &rx, Arc::clone(&a), &slices)?;
        verify("multi warmup", &multi, &reference)?;
    }

    let mut single_samples = Vec::with_capacity(REPS);
    let mut multi_samples = Vec::with_capacity(REPS);
    for rep in 0..REPS {
        let run_single = || -> Result<f64, DynError> {
            let start = Instant::now();
            let output = single_executor.execute_prepared(&a, &single_prepared)?;
            verify("single", &output.values, &reference)?;
            Ok(start.elapsed().as_secs_f64() * 1.0e3)
        };
        if rep.is_multiple_of(2) {
            single_samples.push(run_single()?);
            let (output, ms) = run_multi(&workers, &rx, Arc::clone(&a), &slices)?;
            verify("multi", &output, &reference)?;
            multi_samples.push(ms);
        } else {
            let (output, ms) = run_multi(&workers, &rx, Arc::clone(&a), &slices)?;
            verify("multi", &output, &reference)?;
            multi_samples.push(ms);
            single_samples.push(run_single()?);
        }
    }

    for worker in &workers {
        let _ = worker.tx.send(Command::Stop);
    }
    for worker in workers.drain(..) {
        worker.handle.join().map_err(|_| "worker panicked")?;
    }

    let single_ms = median(single_samples);
    let multi_ms = median(multi_samples);
    println!(
        "W4A4 MULTICORE PASS M=1 K={K} N={N} slices={slices:?} single_ms={single_ms:.3} multicore_ms={multi_ms:.3} speedup={:.2}x",
        single_ms / multi_ms
    );
    Ok(())
}
