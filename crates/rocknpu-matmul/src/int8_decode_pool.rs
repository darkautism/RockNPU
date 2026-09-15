// SPDX-License-Identifier: MIT

use crate::{Int8DecodeExecutor, Int8DecodeOutput, Int8PreparedWeightStats, Int8PreparedWeights};
use rocket_runtime::RocketDevice;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};
use std::time::Instant;

#[derive(Debug)]
pub enum Int8DecodePoolError {
    InvalidWorkerCount(usize),
    InvalidInput(&'static str),
    Worker(String),
    ChannelClosed,
}

impl fmt::Display for Int8DecodePoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWorkerCount(workers) => {
                write!(
                    f,
                    "RK3588 INT8 decode pool requires 1..=3 workers, got {workers}"
                )
            }
            Self::InvalidInput(message) => write!(f, "invalid INT8 decode pool input: {message}"),
            Self::Worker(message) => write!(f, "INT8 decode pool worker failed: {message}"),
            Self::ChannelClosed => write!(f, "INT8 decode pool worker channel closed"),
        }
    }
}

impl std::error::Error for Int8DecodePoolError {}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Int8DecodePoolPreparedStats {
    pub workers: usize,
    pub resident_bytes: usize,
    pub k_slices: usize,
    pub pack_ns_sum: u128,
    pub pack_ns_max: u128,
    pub prepare_wall_ns: u128,
}

pub struct Int8DecodePoolPreparedWeights {
    weight_id: u64,
    k: usize,
    n: usize,
    slices: Vec<(usize, usize)>,
    stats: Int8DecodePoolPreparedStats,
}

impl Int8DecodePoolPreparedWeights {
    pub const fn k(&self) -> usize {
        self.k
    }

    pub const fn n(&self) -> usize {
        self.n
    }

    pub const fn workers(&self) -> usize {
        self.stats.workers
    }

    pub const fn stats(&self) -> Int8DecodePoolPreparedStats {
        self.stats
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Int8DecodePoolStats {
    pub workers_used: usize,
    pub npu_tasks: usize,
    pub wall_ns: u128,
    pub worker_total_ns: Vec<u128>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Int8DecodePoolOutput {
    pub values: Vec<i32>,
    pub stats: Int8DecodePoolStats,
}

enum WorkerCommand {
    Prepare {
        request_id: u64,
        weight_id: u64,
        weights: Arc<[i8]>,
        k: usize,
        n0: usize,
        nsub: usize,
    },
    RunPrepared {
        request_id: u64,
        weight_id: u64,
        activation: Arc<[i8]>,
        k: usize,
        n0: usize,
        nsub: usize,
    },
    Release {
        request_id: u64,
        weight_id: u64,
        n0: usize,
        nsub: usize,
    },
    Stop,
}

enum WorkerResponse {
    Prepared(Result<Int8PreparedWeightStats, String>),
    Ran(Result<Int8DecodeOutput, String>),
    Released,
}

struct WorkerResult {
    request_id: u64,
    worker: usize,
    n0: usize,
    nsub: usize,
    response: WorkerResponse,
}

pub struct Int8DecodePool {
    senders: Vec<mpsc::Sender<WorkerCommand>>,
    result_rx: mpsc::Receiver<WorkerResult>,
    handles: Vec<JoinHandle<()>>,
    next_request_id: u64,
    next_weight_id: u64,
}

impl Int8DecodePool {
    pub fn new(workers: usize) -> Result<Self, Int8DecodePoolError> {
        if !(1..=3).contains(&workers) {
            return Err(Int8DecodePoolError::InvalidWorkerCount(workers));
        }
        let (result_tx, result_rx) = mpsc::channel::<WorkerResult>();
        let (init_tx, init_rx) = mpsc::channel::<Result<usize, String>>();
        let mut senders = Vec::with_capacity(workers);
        let mut handles = Vec::with_capacity(workers);

        for worker in 0..workers {
            let (command_tx, command_rx) = mpsc::channel::<WorkerCommand>();
            senders.push(command_tx);
            let result_tx = result_tx.clone();
            let init_tx = init_tx.clone();
            handles.push(thread::spawn(move || {
                let device = match RocketDevice::open() {
                    Ok(device) => device,
                    Err(err) => {
                        let _ = init_tx.send(Err(format!("worker {worker} open: {err}")));
                        return;
                    }
                };
                let executor = match Int8DecodeExecutor::new(&device) {
                    Ok(executor) => executor,
                    Err(err) => {
                        let _ = init_tx.send(Err(format!("worker {worker} executor: {err}")));
                        return;
                    }
                };
                let mut resident: HashMap<u64, Int8PreparedWeights> = HashMap::new();
                if init_tx.send(Ok(worker)).is_err() {
                    return;
                }

                while let Ok(command) = command_rx.recv() {
                    match command {
                        WorkerCommand::Stop => break,
                        WorkerCommand::Prepare {
                            request_id,
                            weight_id,
                            weights,
                            k,
                            n0,
                            nsub,
                        } => {
                            let begin = n0.saturating_mul(k);
                            let end = n0.saturating_add(nsub).saturating_mul(k);
                            let prepared = if resident.contains_key(&weight_id) {
                                Err(format!("worker {worker}: duplicate resident weight id"))
                            } else if end > weights.len() {
                                Err(format!("worker {worker}: weight slice out of range"))
                            } else {
                                match executor.prepare_weights(&weights[begin..end], k, nsub) {
                                    Ok(prepared) => {
                                        let stats = prepared.stats();
                                        resident.insert(weight_id, prepared);
                                        Ok(stats)
                                    }
                                    Err(err) => Err(format!("worker {worker}: {err}")),
                                }
                            };
                            if result_tx
                                .send(WorkerResult {
                                    request_id,
                                    worker,
                                    n0,
                                    nsub,
                                    response: WorkerResponse::Prepared(prepared),
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        WorkerCommand::RunPrepared {
                            request_id,
                            weight_id,
                            activation,
                            k,
                            n0,
                            nsub,
                        } => {
                            let output = match resident.get(&weight_id) {
                                Some(prepared) if prepared.k() == k && prepared.n() == nsub => {
                                    executor
                                        .execute_prepared(&activation, prepared)
                                        .map_err(|err| format!("worker {worker}: {err}"))
                                }
                                Some(_) => Err(format!("worker {worker}: resident shape mismatch")),
                                None => {
                                    Err(format!("worker {worker}: resident weight id not found"))
                                }
                            };
                            if result_tx
                                .send(WorkerResult {
                                    request_id,
                                    worker,
                                    n0,
                                    nsub,
                                    response: WorkerResponse::Ran(output),
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        WorkerCommand::Release {
                            request_id,
                            weight_id,
                            n0,
                            nsub,
                        } => {
                            resident.remove(&weight_id);
                            if result_tx
                                .send(WorkerResult {
                                    request_id,
                                    worker,
                                    n0,
                                    nsub,
                                    response: WorkerResponse::Released,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                    }
                }
            }));
        }
        drop(init_tx);
        drop(result_tx);

        for _ in 0..workers {
            match init_rx.recv() {
                Ok(Ok(_)) => {}
                Ok(Err(error)) => {
                    for sender in &senders {
                        let _ = sender.send(WorkerCommand::Stop);
                    }
                    for handle in handles.drain(..) {
                        let _ = handle.join();
                    }
                    return Err(Int8DecodePoolError::Worker(error));
                }
                Err(_) => return Err(Int8DecodePoolError::ChannelClosed),
            }
        }

        Ok(Self {
            senders,
            result_rx,
            handles,
            next_request_id: 1,
            next_weight_id: 1,
        })
    }

    pub fn workers(&self) -> usize {
        self.senders.len()
    }

    pub fn effective_workers_for_n(
        &self,
        n: usize,
        requested: usize,
    ) -> Result<usize, Int8DecodePoolError> {
        if requested == 0 || requested > self.senders.len() {
            return Err(Int8DecodePoolError::InvalidWorkerCount(requested));
        }
        Ok(split_n(n, requested)?.len())
    }

    pub fn prepare_weights(
        &mut self,
        weights: Arc<[i8]>,
        k: usize,
        n: usize,
        workers: usize,
    ) -> Result<Int8DecodePoolPreparedWeights, Int8DecodePoolError> {
        if weights.len() != n.saturating_mul(k) {
            return Err(Int8DecodePoolError::InvalidInput(
                "weights must contain exactly N*K elements",
            ));
        }
        if workers == 0 || workers > self.senders.len() {
            return Err(Int8DecodePoolError::InvalidWorkerCount(workers));
        }
        let slices = split_n(n, workers)?;
        let request_id = self.allocate_request_id();
        let weight_id = self.next_weight_id;
        self.next_weight_id = self.next_weight_id.wrapping_add(1);
        let start = Instant::now();

        for (worker, &(n0, nsub)) in slices.iter().enumerate() {
            self.senders[worker]
                .send(WorkerCommand::Prepare {
                    request_id,
                    weight_id,
                    weights: Arc::clone(&weights),
                    k,
                    n0,
                    nsub,
                })
                .map_err(|_| Int8DecodePoolError::ChannelClosed)?;
        }

        let mut stats = Int8DecodePoolPreparedStats {
            workers: slices.len(),
            ..Int8DecodePoolPreparedStats::default()
        };
        let mut first_error = None::<String>;
        for _ in 0..slices.len() {
            let result = self.recv_result(request_id, slices.len())?;
            match result.response {
                WorkerResponse::Prepared(Ok(worker_stats)) => {
                    stats.resident_bytes = stats
                        .resident_bytes
                        .saturating_add(worker_stats.resident_bytes);
                    stats.k_slices = stats.k_slices.saturating_add(worker_stats.k_slices);
                    stats.pack_ns_sum = stats.pack_ns_sum.saturating_add(worker_stats.pack_ns);
                    stats.pack_ns_max = stats.pack_ns_max.max(worker_stats.pack_ns);
                }
                WorkerResponse::Prepared(Err(error)) => {
                    first_error.get_or_insert(error);
                }
                _ => {
                    first_error.get_or_insert_with(|| {
                        "unexpected worker response while preparing INT8 weights".to_string()
                    });
                }
            }
        }
        stats.prepare_wall_ns = start.elapsed().as_nanos();
        if let Some(error) = first_error {
            let _ = self.release_weight_id(weight_id, &slices);
            return Err(Int8DecodePoolError::Worker(error));
        }

        Ok(Int8DecodePoolPreparedWeights {
            weight_id,
            k,
            n,
            slices,
            stats,
        })
    }

    pub fn execute_prepared(
        &mut self,
        activation: Arc<[i8]>,
        weights: &Int8DecodePoolPreparedWeights,
    ) -> Result<Int8DecodePoolOutput, Int8DecodePoolError> {
        if activation.len() != weights.k {
            return Err(Int8DecodePoolError::InvalidInput(
                "activation length must equal prepared K",
            ));
        }
        let request_id = self.allocate_request_id();
        let start = Instant::now();
        for (worker, &(n0, nsub)) in weights.slices.iter().enumerate() {
            self.senders[worker]
                .send(WorkerCommand::RunPrepared {
                    request_id,
                    weight_id: weights.weight_id,
                    activation: Arc::clone(&activation),
                    k: weights.k,
                    n0,
                    nsub,
                })
                .map_err(|_| Int8DecodePoolError::ChannelClosed)?;
        }

        let mut values = vec![0i32; weights.n];
        let mut npu_tasks = 0usize;
        let mut worker_total_ns = vec![0u128; weights.slices.len()];
        for _ in 0..weights.slices.len() {
            let result = self.recv_result(request_id, weights.slices.len())?;
            let output = match result.response {
                WorkerResponse::Ran(Ok(output)) => output,
                WorkerResponse::Ran(Err(error)) => return Err(Int8DecodePoolError::Worker(error)),
                _ => {
                    return Err(Int8DecodePoolError::Worker(
                        "unexpected worker response while executing INT8 weights".to_string(),
                    ));
                }
            };
            if output.values.len() != result.nsub
                || result.n0.saturating_add(result.nsub) > weights.n
            {
                return Err(Int8DecodePoolError::Worker(format!(
                    "worker {} returned invalid output geometry",
                    result.worker
                )));
            }
            values[result.n0..result.n0 + result.nsub].copy_from_slice(&output.values);
            npu_tasks = npu_tasks.saturating_add(output.stats.npu_tasks);
            worker_total_ns[result.worker] = output.stats.total_ns;
        }

        Ok(Int8DecodePoolOutput {
            values,
            stats: Int8DecodePoolStats {
                workers_used: weights.slices.len(),
                npu_tasks,
                wall_ns: start.elapsed().as_nanos(),
                worker_total_ns,
            },
        })
    }

    pub fn release_prepared(
        &mut self,
        weights: &Int8DecodePoolPreparedWeights,
    ) -> Result<(), Int8DecodePoolError> {
        self.release_weight_id(weights.weight_id, &weights.slices)
    }

    fn allocate_request_id(&mut self) -> u64 {
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.wrapping_add(1);
        request_id
    }

    fn recv_result(
        &self,
        request_id: u64,
        workers: usize,
    ) -> Result<WorkerResult, Int8DecodePoolError> {
        let result = self
            .result_rx
            .recv()
            .map_err(|_| Int8DecodePoolError::ChannelClosed)?;
        if result.request_id != request_id || result.worker >= workers {
            return Err(Int8DecodePoolError::Worker(
                "unexpected worker response metadata".to_string(),
            ));
        }
        Ok(result)
    }

    fn release_weight_id(
        &mut self,
        weight_id: u64,
        slices: &[(usize, usize)],
    ) -> Result<(), Int8DecodePoolError> {
        let request_id = self.allocate_request_id();
        for (worker, &(n0, nsub)) in slices.iter().enumerate() {
            self.senders[worker]
                .send(WorkerCommand::Release {
                    request_id,
                    weight_id,
                    n0,
                    nsub,
                })
                .map_err(|_| Int8DecodePoolError::ChannelClosed)?;
        }
        for _ in 0..slices.len() {
            let result = self.recv_result(request_id, slices.len())?;
            if !matches!(result.response, WorkerResponse::Released) {
                return Err(Int8DecodePoolError::Worker(
                    "unexpected worker response while releasing INT8 weights".to_string(),
                ));
            }
        }
        Ok(())
    }
}

impl Drop for Int8DecodePool {
    fn drop(&mut self) {
        for sender in &self.senders {
            let _ = sender.send(WorkerCommand::Stop);
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

fn split_n(n: usize, workers: usize) -> Result<Vec<(usize, usize)>, Int8DecodePoolError> {
    if n == 0 || !n.is_multiple_of(32) {
        return Err(Int8DecodePoolError::InvalidInput(
            "N must be non-zero and 32-aligned",
        ));
    }
    if !(1..=3).contains(&workers) {
        return Err(Int8DecodePoolError::InvalidWorkerCount(workers));
    }
    let blocks = n / 32;
    let active = workers.min(blocks);
    let base = blocks / active;
    let extra = blocks % active;
    let mut slices = Vec::with_capacity(active);
    let mut n0 = 0usize;
    for worker in 0..active {
        let worker_blocks = base + usize::from(worker < extra);
        let nsub = worker_blocks * 32;
        slices.push((n0, nsub));
        n0 += nsub;
    }
    Ok(slices)
}

#[cfg(test)]
mod tests {
    use super::split_n;

    #[test]
    fn three_way_tinyllama_gate_split_is_aligned_and_complete() {
        let slices = split_n(5632, 3).unwrap();
        assert_eq!(slices, vec![(0, 1888), (1888, 1888), (3776, 1856)]);
        assert_eq!(slices.iter().map(|(_, n)| n).sum::<usize>(), 5632);
        assert!(slices.iter().all(|(n0, n)| n0 % 32 == 0 && n % 32 == 0));
    }

    #[test]
    fn requested_workers_are_capped_by_channel_blocks() {
        assert_eq!(split_n(32, 3).unwrap(), vec![(0, 32)]);
        assert_eq!(split_n(64, 3).unwrap(), vec![(0, 32), (32, 32)]);
    }
}
