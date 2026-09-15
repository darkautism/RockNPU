// SPDX-License-Identifier: MIT

use crate::{Int4DecodeExecutor, Int4DecodeOutput, Int4PreparedWeightStats, Int4PreparedWeights};
use rocket_runtime::RocketDevice;
use std::collections::{HashMap, hash_map::Entry};
use std::fmt;
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};
use std::time::Instant;

#[derive(Debug)]
pub enum Int4DecodePoolError {
    InvalidWorkerCount(usize),
    InvalidInput(&'static str),
    Worker(String),
    ChannelClosed,
}

impl fmt::Display for Int4DecodePoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWorkerCount(workers) => {
                write!(
                    f,
                    "RK3588 W4A4 decode pool requires 1..=3 workers, got {workers}"
                )
            }
            Self::InvalidInput(message) => write!(f, "invalid W4A4 decode pool input: {message}"),
            Self::Worker(message) => write!(f, "W4A4 decode pool worker failed: {message}"),
            Self::ChannelClosed => write!(f, "W4A4 decode pool worker channel closed"),
        }
    }
}

impl std::error::Error for Int4DecodePoolError {}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Int4DecodePoolPreparedStats {
    pub workers: usize,
    pub resident_bytes: usize,
    pub pack_ns_sum: u128,
    pub pack_ns_max: u128,
    pub prepare_wall_ns: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WorkerSlice {
    n0: usize,
    nsub: usize,
}

pub struct Int4DecodePoolPreparedWeights {
    weight_id: u64,
    k: usize,
    n: usize,
    slices: Vec<WorkerSlice>,
    stats: Int4DecodePoolPreparedStats,
}

impl Int4DecodePoolPreparedWeights {
    pub const fn k(&self) -> usize {
        self.k
    }

    pub const fn n(&self) -> usize {
        self.n
    }

    pub const fn workers(&self) -> usize {
        self.stats.workers
    }

    pub const fn stats(&self) -> Int4DecodePoolPreparedStats {
        self.stats
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Int4DecodePoolStats {
    pub workers_used: usize,
    pub npu_tasks: usize,
    pub wall_ns: u128,
    pub saturated_outputs: usize,
    pub worker_total_ns: Vec<u128>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Int4DecodePoolOutput {
    pub values: Vec<i16>,
    pub stats: Int4DecodePoolStats,
}

enum WorkerCommand {
    Prepare {
        request_id: u64,
        weight_id: u64,
        weights: Arc<[i8]>,
        full_k: usize,
        slice: WorkerSlice,
    },
    RunPrepared {
        request_id: u64,
        weight_id: u64,
        activation: Arc<[i8]>,
        slice: WorkerSlice,
    },
    Release {
        request_id: u64,
        weight_id: u64,
        slice: WorkerSlice,
    },
    Stop,
}

enum WorkerResponse {
    Prepared(Result<Int4PreparedWeightStats, String>),
    Ran(Result<Int4DecodeOutput, String>),
    Released,
}

struct WorkerResult {
    request_id: u64,
    worker: usize,
    slice: WorkerSlice,
    response: WorkerResponse,
}

pub struct Int4DecodePool {
    senders: Vec<mpsc::Sender<WorkerCommand>>,
    result_rx: mpsc::Receiver<WorkerResult>,
    handles: Vec<JoinHandle<()>>,
    next_request_id: u64,
    next_weight_id: u64,
}

impl Int4DecodePool {
    pub fn new(workers: usize) -> Result<Self, Int4DecodePoolError> {
        if !(1..=3).contains(&workers) {
            return Err(Int4DecodePoolError::InvalidWorkerCount(workers));
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
                let executor = Int4DecodeExecutor::new(&device);
                let mut resident: HashMap<u64, Int4PreparedWeights> = HashMap::new();
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
                            full_k,
                            slice,
                        } => {
                            let prepared = match resident.entry(weight_id) {
                                Entry::Occupied(_) => {
                                    Err(format!("worker {worker}: duplicate resident weight id"))
                                }
                                Entry::Vacant(entry) => {
                                    let begin = slice.n0.saturating_mul(full_k);
                                    let end =
                                        slice.n0.saturating_add(slice.nsub).saturating_mul(full_k);
                                    if end > weights.len() || begin > end {
                                        Err(format!("worker {worker}: weight slice out of range"))
                                    } else {
                                        executor
                                            .prepare_weights(
                                                &weights[begin..end],
                                                full_k,
                                                slice.nsub,
                                            )
                                            .map(|prepared| {
                                                let stats = prepared.stats();
                                                entry.insert(prepared);
                                                stats
                                            })
                                            .map_err(|err| format!("worker {worker}: {err}"))
                                    }
                                }
                            };
                            if result_tx
                                .send(WorkerResult {
                                    request_id,
                                    worker,
                                    slice,
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
                            slice,
                        } => {
                            let output = match resident.get(&weight_id) {
                                Some(prepared)
                                    if prepared.k() == activation.len()
                                        && prepared.n() == slice.nsub =>
                                {
                                    executor
                                        .execute_prepared(&activation, prepared)
                                        .map_err(|err| format!("worker {worker}: {err}"))
                                }
                                Some(_) => Err(format!(
                                    "worker {worker}: resident/activation shape mismatch"
                                )),
                                None => {
                                    Err(format!("worker {worker}: resident weight id not found"))
                                }
                            };
                            if result_tx
                                .send(WorkerResult {
                                    request_id,
                                    worker,
                                    slice,
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
                            slice,
                        } => {
                            resident.remove(&weight_id);
                            if result_tx
                                .send(WorkerResult {
                                    request_id,
                                    worker,
                                    slice,
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
                    return Err(Int4DecodePoolError::Worker(error));
                }
                Err(_) => return Err(Int4DecodePoolError::ChannelClosed),
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
    ) -> Result<usize, Int4DecodePoolError> {
        self.validate_requested_workers(requested)?;
        Ok(split_n(n, requested)?.len())
    }

    pub fn prepare_weights(
        &mut self,
        weights: Arc<[i8]>,
        k: usize,
        n: usize,
        workers: usize,
    ) -> Result<Int4DecodePoolPreparedWeights, Int4DecodePoolError> {
        if weights.len() != n.saturating_mul(k) {
            return Err(Int4DecodePoolError::InvalidInput(
                "weights must contain exactly N*K elements",
            ));
        }
        self.validate_requested_workers(workers)?;
        let slices = split_n(n, workers)?;
        let request_id = self.allocate_request_id();
        let weight_id = self.next_weight_id;
        self.next_weight_id = self.next_weight_id.wrapping_add(1);
        let start = Instant::now();

        for (worker, &slice) in slices.iter().enumerate() {
            self.senders[worker]
                .send(WorkerCommand::Prepare {
                    request_id,
                    weight_id,
                    weights: Arc::clone(&weights),
                    full_k: k,
                    slice,
                })
                .map_err(|_| Int4DecodePoolError::ChannelClosed)?;
        }

        let mut stats = Int4DecodePoolPreparedStats {
            workers: slices.len(),
            ..Int4DecodePoolPreparedStats::default()
        };
        let mut first_error = None::<String>;
        for _ in 0..slices.len() {
            let result = self.recv_result(request_id, slices.len())?;
            match result.response {
                WorkerResponse::Prepared(Ok(worker_stats)) => {
                    stats.resident_bytes = stats
                        .resident_bytes
                        .saturating_add(worker_stats.resident_bytes);
                    stats.pack_ns_sum = stats.pack_ns_sum.saturating_add(worker_stats.pack_ns);
                    stats.pack_ns_max = stats.pack_ns_max.max(worker_stats.pack_ns);
                }
                WorkerResponse::Prepared(Err(error)) => {
                    first_error.get_or_insert(error);
                }
                _ => {
                    first_error.get_or_insert_with(|| {
                        "unexpected worker response while preparing W4A4 weights".to_string()
                    });
                }
            }
        }
        stats.prepare_wall_ns = start.elapsed().as_nanos();
        if let Some(error) = first_error {
            let _ = self.release_weight_id(weight_id, &slices);
            return Err(Int4DecodePoolError::Worker(error));
        }

        Ok(Int4DecodePoolPreparedWeights {
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
        weights: &Int4DecodePoolPreparedWeights,
    ) -> Result<Int4DecodePoolOutput, Int4DecodePoolError> {
        if activation.len() != weights.k {
            return Err(Int4DecodePoolError::InvalidInput(
                "activation length must equal prepared K",
            ));
        }
        let request_id = self.allocate_request_id();
        let start = Instant::now();
        for (worker, &slice) in weights.slices.iter().enumerate() {
            self.senders[worker]
                .send(WorkerCommand::RunPrepared {
                    request_id,
                    weight_id: weights.weight_id,
                    activation: Arc::clone(&activation),
                    slice,
                })
                .map_err(|_| Int4DecodePoolError::ChannelClosed)?;
        }

        let mut values = vec![0i16; weights.n];
        let mut saturated_outputs = 0usize;
        let mut worker_total_ns = vec![0u128; weights.slices.len()];
        let mut first_error = None::<String>;
        for _ in 0..weights.slices.len() {
            let result = self.recv_result(request_id, weights.slices.len())?;
            match result.response {
                WorkerResponse::Ran(Ok(output)) => {
                    if result.worker >= worker_total_ns.len()
                        || output.values.len() != result.slice.nsub
                        || result.slice.n0.saturating_add(result.slice.nsub) > values.len()
                    {
                        first_error.get_or_insert_with(|| {
                            "worker returned inconsistent W4A4 output metadata".to_string()
                        });
                        continue;
                    }
                    values[result.slice.n0..result.slice.n0 + result.slice.nsub]
                        .copy_from_slice(&output.values);
                    saturated_outputs =
                        saturated_outputs.saturating_add(output.stats.saturated_outputs);
                    worker_total_ns[result.worker] = output.stats.total_ns;
                }
                WorkerResponse::Ran(Err(error)) => {
                    first_error.get_or_insert(error);
                }
                _ => {
                    first_error.get_or_insert_with(|| {
                        "unexpected worker response while executing W4A4 weights".to_string()
                    });
                }
            }
        }
        if let Some(error) = first_error {
            return Err(Int4DecodePoolError::Worker(error));
        }

        Ok(Int4DecodePoolOutput {
            values,
            stats: Int4DecodePoolStats {
                workers_used: weights.slices.len(),
                npu_tasks: weights.slices.len(),
                wall_ns: start.elapsed().as_nanos(),
                saturated_outputs,
                worker_total_ns,
            },
        })
    }

    pub fn release_prepared(
        &mut self,
        weights: &Int4DecodePoolPreparedWeights,
    ) -> Result<(), Int4DecodePoolError> {
        self.release_weight_id(weights.weight_id, &weights.slices)
    }

    fn release_weight_id(
        &mut self,
        weight_id: u64,
        slices: &[WorkerSlice],
    ) -> Result<(), Int4DecodePoolError> {
        let request_id = self.allocate_request_id();
        for (worker, &slice) in slices.iter().enumerate() {
            self.senders[worker]
                .send(WorkerCommand::Release {
                    request_id,
                    weight_id,
                    slice,
                })
                .map_err(|_| Int4DecodePoolError::ChannelClosed)?;
        }
        for _ in 0..slices.len() {
            let result = self.recv_result(request_id, slices.len())?;
            if !matches!(result.response, WorkerResponse::Released) {
                return Err(Int4DecodePoolError::Worker(
                    "unexpected worker response while releasing W4A4 weights".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn allocate_request_id(&mut self) -> u64 {
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.wrapping_add(1);
        request_id
    }

    fn validate_requested_workers(&self, requested: usize) -> Result<(), Int4DecodePoolError> {
        if requested == 0 || requested > self.senders.len() {
            return Err(Int4DecodePoolError::InvalidWorkerCount(requested));
        }
        Ok(())
    }

    fn recv_result(
        &self,
        request_id: u64,
        worker_limit: usize,
    ) -> Result<WorkerResult, Int4DecodePoolError> {
        let result = self
            .result_rx
            .recv()
            .map_err(|_| Int4DecodePoolError::ChannelClosed)?;
        if result.request_id != request_id || result.worker >= worker_limit {
            return Err(Int4DecodePoolError::Worker(
                "out-of-order W4A4 worker response".to_string(),
            ));
        }
        Ok(result)
    }
}

impl Drop for Int4DecodePool {
    fn drop(&mut self) {
        for sender in &self.senders {
            let _ = sender.send(WorkerCommand::Stop);
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

fn split_n(n: usize, workers: usize) -> Result<Vec<WorkerSlice>, Int4DecodePoolError> {
    if n == 0 || !n.is_multiple_of(64) {
        return Err(Int4DecodePoolError::InvalidInput(
            "N must be non-zero and a multiple of 64",
        ));
    }
    if workers == 0 {
        return Err(Int4DecodePoolError::InvalidWorkerCount(workers));
    }
    let blocks = n / 64;
    if blocks < workers {
        return Err(Int4DecodePoolError::InvalidInput(
            "N has fewer 64-channel blocks than workers",
        ));
    }
    let base = blocks / workers;
    let extra = blocks % workers;
    let mut slices = Vec::with_capacity(workers);
    let mut n0 = 0usize;
    for worker in 0..workers {
        let nsub = (base + usize::from(worker < extra)) * 64;
        slices.push(WorkerSlice { n0, nsub });
        n0 += nsub;
    }
    if n0 != n {
        return Err(Int4DecodePoolError::InvalidInput(
            "N split did not cover the complete output",
        ));
    }
    Ok(slices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_way_tinyllama_ffn_split_is_aligned_and_complete() {
        let slices = split_n(5632, 3).expect("valid split");
        assert_eq!(slices.len(), 3);
        assert_eq!(slices[0], WorkerSlice { n0: 0, nsub: 1920 });
        assert_eq!(
            slices[1],
            WorkerSlice {
                n0: 1920,
                nsub: 1856
            }
        );
        assert_eq!(
            slices[2],
            WorkerSlice {
                n0: 3776,
                nsub: 1856
            }
        );
        assert!(slices.iter().all(|slice| slice.nsub.is_multiple_of(64)));
    }

    #[test]
    fn requested_workers_are_capped_by_channel_blocks() {
        assert!(split_n(64, 1).is_ok());
        assert!(split_n(64, 2).is_err());
    }
}
