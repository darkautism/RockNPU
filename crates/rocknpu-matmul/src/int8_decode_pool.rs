// SPDX-License-Identifier: MIT

use crate::int8_decode::{
    Int8MtileBatchOutput, Int8MtileBatchScratch, Int8MtileDirectScratch, Int8MtileScratch,
    Int8OwnedScratch,
};
use crate::{
    Int8DecodeExecutor, Int8DecodeOutput, Int8DecodeStats, Int8PreparedWeightStats,
    Int8PreparedWeights,
};
use rocket_runtime::{RocketDevice, RocketOwnedBuffer};
use std::collections::HashMap;
use std::env;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Int8DecodeSplit {
    N,
    K,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Int8DecodePoolPreparedStats {
    pub workers: usize,
    pub resident_bytes: usize,
    pub k_slices: usize,
    pub pack_ns_sum: u128,
    pub pack_ns_max: u128,
    pub prepare_wall_ns: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WorkerSlice {
    k0: usize,
    ksub: usize,
    n0: usize,
    nsub: usize,
}

pub struct Int8DecodePoolPreparedWeights {
    weight_id: u64,
    k: usize,
    n: usize,
    split: Int8DecodeSplit,
    slices: Vec<WorkerSlice>,
    stats: Int8DecodePoolPreparedStats,
    direct_prepared: Option<Vec<Int8PreparedWeights>>,
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

    pub const fn split(&self) -> Int8DecodeSplit {
        self.split
    }

    /// Whether these weights live on the direct-submit workers.
    pub fn is_direct(&self) -> bool {
        self.direct_prepared.is_some()
    }

    pub const fn stats(&self) -> Int8DecodePoolPreparedStats {
        self.stats
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Int8DecodePoolStats {
    pub split: Int8DecodeSplit,
    pub workers_used: usize,
    pub npu_tasks: usize,
    pub wall_ns: u128,
    pub worker_total_ns: Vec<u128>,
    pub worker_stats: Vec<Int8DecodeStats>,
}

/// Host/NPU phase split of one direct M-tile call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Int8MtileDirectTimings {
    pub stage_submit_ns: u128,
    pub wait_ns: u128,
    pub consume_ns: u128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Int8DecodePoolOutput {
    pub values: Vec<i32>,
    pub stats: Int8DecodePoolStats,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Int8DecodePoolMtileBatchOutput {
    pub values: Vec<Vec<i32>>,
    pub stats: Int8DecodePoolStats,
}

enum WorkerCommand {
    Prepare {
        request_id: u64,
        weight_id: u64,
        weights: Arc<[i8]>,
        full_k: usize,
        slice: WorkerSlice,
        m1_fullk: bool,
    },
    RunPrepared {
        request_id: u64,
        weight_id: u64,
        activation: Arc<[i8]>,
        slice: WorkerSlice,
    },
    RunPreparedMtile {
        request_id: u64,
        weight_id: u64,
        m: usize,
        activation: Arc<[i8]>,
        slice: WorkerSlice,
    },
    RunPreparedMtileBatch {
        request_id: u64,
        m: usize,
        items: Vec<(u64, Arc<[i8]>, WorkerSlice)>,
    },
    Release {
        request_id: u64,
        weight_id: u64,
        slice: WorkerSlice,
    },
    Stop,
}

enum WorkerResponse {
    Prepared(Result<Int8PreparedWeightStats, String>),
    Ran(Result<Int8DecodeOutput, String>),
    RanMtileBatch(Result<Int8MtileBatchOutput, String>),
    Released,
}

struct WorkerResult {
    request_id: u64,
    worker: usize,
    slice: WorkerSlice,
    response: WorkerResponse,
}

struct DirectWorker {
    device: RocketDevice,
    _guard: RocketOwnedBuffer,
    scratch: HashMap<(usize, usize), Option<Int8OwnedScratch>>,
    mtile_scratch: HashMap<(usize, usize), Option<Int8MtileDirectScratch>>,
}

pub struct Int8DecodePool {
    senders: Vec<mpsc::Sender<WorkerCommand>>,
    result_rx: mpsc::Receiver<WorkerResult>,
    handles: Vec<JoinHandle<()>>,
    direct_workers: Option<Vec<DirectWorker>>,
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
                let mut mtile_scratch: HashMap<(usize, usize), Option<Int8MtileScratch>> =
                    HashMap::new();
                let mut mtile_batch_scratch:
                    HashMap<(usize, usize), Option<Int8MtileBatchScratch>> = HashMap::new();
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
                            m1_fullk,
                        } => {
                            let prepared = if resident.contains_key(&weight_id) {
                                Err(format!("worker {worker}: duplicate resident weight id"))
                            } else {
                                prepare_worker_weights(
                                    &executor,
                                    &weights,
                                    full_k,
                                    slice,
                                    m1_fullk,
                                )
                                    .map(|prepared| {
                                        let stats = prepared.stats();
                                        resident.insert(weight_id, prepared);
                                        stats
                                    })
                                    .map_err(|err| format!("worker {worker}: {err}"))
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
                                    if prepared.k() == slice.ksub && prepared.n() == slice.nsub =>
                                {
                                    let end = slice.k0.saturating_add(slice.ksub);
                                    if end > activation.len() {
                                        Err(format!(
                                            "worker {worker}: activation slice out of range"
                                        ))
                                    } else {
                                        executor
                                            .execute_prepared(&activation[slice.k0..end], prepared)
                                            .map_err(|err| format!("worker {worker}: {err}"))
                                    }
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
                                    slice,
                                    response: WorkerResponse::Ran(output),
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        WorkerCommand::RunPreparedMtile {
                            request_id,
                            weight_id,
                            m,
                            activation,
                            slice,
                        } => {
                            let output = match resident.get(&weight_id) {
                                Some(prepared)
                                    if matches!(m, 4 | 8 | 12 | 16 | 32 | 48 | 64 | 128)
                                        && prepared.k() == slice.ksub
                                        && prepared.n() == slice.nsub
                                        && m != 0
                                        && activation.len() % m == 0 =>
                                {
                                    let full_k = activation.len() / m;
                                    let end_k = slice.k0.saturating_add(slice.ksub);
                                    if end_k > full_k {
                                        Err(format!(
                                            "worker {worker}: M-tile activation slice out of range"
                                        ))
                                    } else {
                                        let scratch_slot = mtile_scratch
                                            .entry((slice.ksub, slice.nsub))
                                            .or_insert(None);
                                        if slice.k0 == 0 && slice.ksub == full_k {
                                            executor
                                                .execute_prepared_mtile_persistent(
                                                    m,
                                                    &activation,
                                                    prepared,
                                                    scratch_slot,
                                                )
                                                .map_err(|err| format!("worker {worker}: {err}"))
                                        } else {
                                            let mut sliced =
                                                Vec::with_capacity(m.saturating_mul(slice.ksub));
                                            for row in 0..m {
                                                let begin = row
                                                    .saturating_mul(full_k)
                                                    .saturating_add(slice.k0);
                                                let end = begin.saturating_add(slice.ksub);
                                                sliced.extend_from_slice(&activation[begin..end]);
                                            }
                                            executor
                                                .execute_prepared_mtile_persistent(
                                                    m,
                                                    &sliced,
                                                    prepared,
                                                    scratch_slot,
                                                )
                                                .map_err(|err| format!("worker {worker}: {err}"))
                                        }
                                    }
                                }
                                Some(_) => {
                                    Err(format!("worker {worker}: resident M-tile shape mismatch"))
                                }
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
                        WorkerCommand::RunPreparedMtileBatch {
                            request_id,
                            m,
                            items,
                        } => {
                            let slice = items.first().map(|item| item.2).unwrap_or(WorkerSlice {
                                k0: 0,
                                ksub: 0,
                                n0: 0,
                                nsub: 0,
                            });
                            let output = if items.is_empty() {
                                Err(format!("worker {worker}: empty M-tile batch"))
                            } else {
                                let mut activation_refs = Vec::with_capacity(items.len());
                                let mut prepared_refs = Vec::with_capacity(items.len());
                                let mut error = None;
                                for (weight_id, activation, item_slice) in &items {
                                    match resident.get(weight_id) {
                                        Some(prepared)
                                            if item_slice.k0 == 0
                                                && prepared.k() == item_slice.ksub
                                                && prepared.n() == item_slice.nsub
                                                && activation.len() == m.saturating_mul(item_slice.ksub) =>
                                        {
                                            activation_refs.push(activation.as_ref());
                                            prepared_refs.push(prepared);
                                        }
                                        Some(_) => {
                                            error = Some(format!(
                                                "worker {worker}: resident M-tile batch shape mismatch"
                                            ));
                                            break;
                                        }
                                        None => {
                                            error = Some(format!(
                                                "worker {worker}: resident M-tile batch weight id not found"
                                            ));
                                            break;
                                        }
                                    }
                                }
                                if let Some(error) = error {
                                    Err(error)
                                } else {
                                    let scratch_slot = mtile_batch_scratch
                                        .entry((slice.ksub, slice.nsub))
                                        .or_insert(None);
                                    executor
                                        .execute_prepared_mtile_batch_persistent(
                                            m,
                                            &activation_refs,
                                            &prepared_refs,
                                            scratch_slot,
                                        )
                                        .map_err(|err| format!("worker {worker}: {err}"))
                                }
                            };
                            if result_tx
                                .send(WorkerResult {
                                    request_id,
                                    worker,
                                    slice,
                                    response: WorkerResponse::RanMtileBatch(output),
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
                    return Err(Int8DecodePoolError::Worker(error));
                }
                Err(_) => return Err(Int8DecodePoolError::ChannelClosed),
            }
        }

        let direct_workers = if env::var_os("ROCKNPU_W8_DIRECT_SUBMIT").is_some() {
            let mut direct = Vec::with_capacity(workers);
            for worker in 0..workers {
                let device = RocketDevice::open().map_err(|err| {
                    Int8DecodePoolError::Worker(format!("direct worker {worker} open: {err}"))
                })?;
                let guard = device.alloc_owned_buffer(4096).map_err(|err| {
                    Int8DecodePoolError::Worker(format!("direct worker {worker} guard: {err}"))
                })?;
                direct.push(DirectWorker {
                    device,
                    _guard: guard,
                    scratch: HashMap::new(),
                    mtile_scratch: HashMap::new(),
                });
            }
            Some(direct)
        } else {
            None
        };

        Ok(Self {
            senders,
            result_rx,
            handles,
            direct_workers,
            next_request_id: 1,
            next_weight_id: 1,
        })
    }

    /// Whether direct (calling-thread) submission is enabled.
    pub fn direct_enabled(&self) -> bool {
        self.direct_workers.is_some()
    }

    pub fn workers(&self) -> usize {
        self.direct_workers
            .as_ref()
            .map_or(self.senders.len(), Vec::len)
    }

    pub fn effective_workers_for_n(
        &self,
        n: usize,
        requested: usize,
    ) -> Result<usize, Int8DecodePoolError> {
        self.validate_requested_workers(requested)?;
        Ok(split_n(n, requested)?.len())
    }

    pub fn effective_workers_for_k(
        &self,
        k: usize,
        requested: usize,
    ) -> Result<usize, Int8DecodePoolError> {
        self.validate_requested_workers(requested)?;
        Ok(split_k(k, requested)?.len())
    }

    pub fn prepare_weights(
        &mut self,
        weights: Arc<[i8]>,
        k: usize,
        n: usize,
        workers: usize,
    ) -> Result<Int8DecodePoolPreparedWeights, Int8DecodePoolError> {
        self.prepare_weights_with_split(weights, k, n, workers, Int8DecodeSplit::N)
    }

    pub fn prepare_weights_with_split(
        &mut self,
        weights: Arc<[i8]>,
        k: usize,
        n: usize,
        workers: usize,
        split: Int8DecodeSplit,
    ) -> Result<Int8DecodePoolPreparedWeights, Int8DecodePoolError> {
        self.prepare_weights_with_split_mode(weights, k, n, workers, split, false, true)
    }

    /// Prepare resident weights for the threaded M-tile executor even when
    /// M=1 direct submission is enabled on this pool.
    pub fn prepare_weights_mtile_with_split(
        &mut self,
        weights: Arc<[i8]>,
        k: usize,
        n: usize,
        workers: usize,
        split: Int8DecodeSplit,
    ) -> Result<Int8DecodePoolPreparedWeights, Int8DecodePoolError> {
        self.prepare_weights_with_split_mode(weights, k, n, workers, split, false, false)
    }

    /// Prepare resident M-tile weights on the direct-submit workers (when
    /// enabled) so `execute_prepared_mtile_direct` can drive all cores from
    /// the calling thread. Falls back to threaded workers otherwise.
    pub fn prepare_weights_mtile_direct_with_split(
        &mut self,
        weights: Arc<[i8]>,
        k: usize,
        n: usize,
        workers: usize,
        split: Int8DecodeSplit,
    ) -> Result<Int8DecodePoolPreparedWeights, Int8DecodePoolError> {
        self.prepare_weights_with_split_mode(weights, k, n, workers, split, false, true)
    }

    /// Execute A[M,K] against direct-prepared M-tile weights: stage and submit
    /// every worker's slice, then wait and write the int32 [M,N] result into
    /// `out` (N-split slices are placed, K-split partials are summed).
    pub fn execute_prepared_mtile_direct(
        &mut self,
        m: usize,
        activation: &[i8],
        weights: &Int8DecodePoolPreparedWeights,
        out: &mut [i32],
    ) -> Result<Int8MtileDirectTimings, Int8DecodePoolError> {
        let started = Instant::now();
        let direct_workers = self
            .direct_workers
            .as_mut()
            .ok_or_else(|| Int8DecodePoolError::Worker("direct workers missing".to_string()))?;
        let prepared = weights.direct_prepared.as_ref().ok_or_else(|| {
            Int8DecodePoolError::Worker("direct prepared weights missing".to_string())
        })?;
        if prepared.len() != weights.slices.len()
            || prepared.len() > direct_workers.len()
            || activation.len() != m.saturating_mul(weights.k)
            || out.len() != m.saturating_mul(weights.n)
        {
            return Err(Int8DecodePoolError::InvalidInput(
                "direct M-tile geometry mismatch",
            ));
        }
        let mut pendings = Vec::with_capacity(prepared.len());
        for (worker, (&slice, prepared_worker)) in
            weights.slices.iter().zip(prepared.iter()).enumerate()
        {
            let state = &mut direct_workers[worker];
            let key = (prepared_worker.k(), prepared_worker.n());
            let slot = state.mtile_scratch.entry(key).or_insert(None);
            let executor = Int8DecodeExecutor::from_externally_guarded_device(&state.device);
            let pending = executor
                .begin_mtile_direct(m, activation, weights.k, slice.k0, prepared_worker, slot)
                .map_err(|err| {
                    Int8DecodePoolError::Worker(format!("direct M-tile worker {worker} begin: {err}"))
                })?;
            pendings.push((worker, slice, key, pending));
        }
        let mut timings = Int8MtileDirectTimings {
            stage_submit_ns: started.elapsed().as_nanos(),
            ..Int8MtileDirectTimings::default()
        };
        let n = weights.n;
        for (index, (worker, slice, key, pending)) in pendings.into_iter().enumerate() {
            let state = &mut direct_workers[worker];
            let slot = state.mtile_scratch.get_mut(&key).ok_or_else(|| {
                Int8DecodePoolError::Worker("direct M-tile scratch missing".to_string())
            })?;
            let executor = Int8DecodeExecutor::from_externally_guarded_device(&state.device);
            let split = weights.split;
            let mut consume = |values: &[i32]| match split {
                Int8DecodeSplit::N => {
                    for row in 0..m {
                        out[row * n + slice.n0..row * n + slice.n0 + slice.nsub]
                            .copy_from_slice(&values[row * slice.nsub..(row + 1) * slice.nsub]);
                    }
                }
                Int8DecodeSplit::K => {
                    if index == 0 {
                        out.copy_from_slice(&values[..m * n]);
                    } else {
                        for (dst, &src) in out.iter_mut().zip(&values[..m * n]) {
                            *dst = dst.wrapping_add(src);
                        }
                    }
                }
            };
            let (wait_ns, consume_ns) = executor
                .finish_mtile_direct(pending, slot, &mut consume)
                .map_err(|err| {
                    Int8DecodePoolError::Worker(format!(
                        "direct M-tile worker {worker} finish: {err}"
                    ))
                })?;
            timings.wait_ns += wait_ns;
            timings.consume_ns += consume_ns;
        }
        Ok(timings)
    }

    pub fn prepare_weights_m1_with_split(
        &mut self,
        weights: Arc<[i8]>,
        k: usize,
        n: usize,
        workers: usize,
        split: Int8DecodeSplit,
    ) -> Result<Int8DecodePoolPreparedWeights, Int8DecodePoolError> {
        self.prepare_weights_with_split_mode(weights, k, n, workers, split, true, true)
    }

    fn prepare_weights_with_split_mode(
        &mut self,
        weights: Arc<[i8]>,
        k: usize,
        n: usize,
        workers: usize,
        split: Int8DecodeSplit,
        m1_fullk: bool,
        allow_direct: bool,
    ) -> Result<Int8DecodePoolPreparedWeights, Int8DecodePoolError> {
        if weights.len() != n.saturating_mul(k) {
            return Err(Int8DecodePoolError::InvalidInput(
                "weights must contain exactly N*K elements",
            ));
        }
        self.validate_requested_workers(workers)?;
        let slices = worker_slices(k, n, workers, split)?;

        if let Some(direct_workers) = self.direct_workers.as_ref().filter(|_| allow_direct) {
            let start = Instant::now();
            let mut direct_prepared = Vec::with_capacity(slices.len());
            let mut stats = Int8DecodePoolPreparedStats {
                workers: slices.len(),
                ..Int8DecodePoolPreparedStats::default()
            };
            for (worker, &slice) in slices.iter().enumerate() {
                let executor = Int8DecodeExecutor::from_externally_guarded_device(
                    &direct_workers[worker].device,
                );
                let prepared =
                    prepare_worker_weights(&executor, &weights, k, slice, m1_fullk).map_err(|err| {
                        Int8DecodePoolError::Worker(format!("direct worker {worker}: {err}"))
                    })?;
                let worker_stats = prepared.stats();
                stats.resident_bytes = stats
                    .resident_bytes
                    .saturating_add(worker_stats.resident_bytes);
                stats.k_slices = stats.k_slices.saturating_add(worker_stats.k_slices);
                stats.pack_ns_sum = stats.pack_ns_sum.saturating_add(worker_stats.pack_ns);
                stats.pack_ns_max = stats.pack_ns_max.max(worker_stats.pack_ns);
                direct_prepared.push(prepared);
            }
            stats.prepare_wall_ns = start.elapsed().as_nanos();
            return Ok(Int8DecodePoolPreparedWeights {
                weight_id: 0,
                k,
                n,
                split,
                slices,
                stats,
                direct_prepared: Some(direct_prepared),
            });
        }

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
                    m1_fullk,
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
            split,
            slices,
            stats,
            direct_prepared: None,
        })
    }

    pub fn execute_prepared_mtile(
        &mut self,
        m: usize,
        activation: Arc<[i8]>,
        weights: &Int8DecodePoolPreparedWeights,
    ) -> Result<Int8DecodePoolOutput, Int8DecodePoolError> {
        if !matches!(m, 4 | 8 | 12 | 16 | 32 | 48 | 64 | 128) {
            return Err(Int8DecodePoolError::InvalidInput(
                "M-tile pool path requires M in {4,8,12,16,32,48,64,128}",
            ));
        }
        if activation.len() != m.saturating_mul(weights.k) {
            return Err(Int8DecodePoolError::InvalidInput(
                "M-tile activation length must equal M*K",
            ));
        }
        if weights.direct_prepared.is_some() {
            return Err(Int8DecodePoolError::InvalidInput(
                "M-tile pool path does not yet support ROCKNPU_W8_DIRECT_SUBMIT",
            ));
        }

        let request_id = self.allocate_request_id();
        let start = Instant::now();
        for (worker, &slice) in weights.slices.iter().enumerate() {
            self.senders[worker]
                .send(WorkerCommand::RunPreparedMtile {
                    request_id,
                    weight_id: weights.weight_id,
                    m,
                    activation: Arc::clone(&activation),
                    slice,
                })
                .map_err(|_| Int8DecodePoolError::ChannelClosed)?;
        }

        let total_values = m
            .checked_mul(weights.n)
            .ok_or(Int8DecodePoolError::InvalidInput(
                "M-tile output size overflow",
            ))?;
        let mut values = vec![0i32; total_values];
        let mut npu_tasks = 0usize;
        let mut worker_total_ns = vec![0u128; weights.slices.len()];
        let mut worker_stats = vec![Int8DecodeStats::default(); weights.slices.len()];
        for _ in 0..weights.slices.len() {
            let result = self.recv_result(request_id, weights.slices.len())?;
            let output = match result.response {
                WorkerResponse::Ran(Ok(output)) => output,
                WorkerResponse::Ran(Err(error)) => return Err(Int8DecodePoolError::Worker(error)),
                _ => {
                    return Err(Int8DecodePoolError::Worker(
                        "unexpected worker response while executing M-tile INT8 weights"
                            .to_string(),
                    ));
                }
            };
            match weights.split {
                Int8DecodeSplit::N => {
                    let expected_worker_values = m.saturating_mul(result.slice.nsub);
                    if output.values.len() != expected_worker_values
                        || result.slice.n0.saturating_add(result.slice.nsub) > weights.n
                    {
                        return Err(Int8DecodePoolError::Worker(format!(
                            "worker {} returned invalid N-split M-tile output geometry",
                            result.worker
                        )));
                    }
                    for row in 0..m {
                        let src_start = row * result.slice.nsub;
                        let dst_start = row * weights.n + result.slice.n0;
                        values[dst_start..dst_start + result.slice.nsub].copy_from_slice(
                            &output.values[src_start..src_start + result.slice.nsub],
                        );
                    }
                }
                Int8DecodeSplit::K => {
                    if output.values.len() != total_values
                        || result.slice.k0.saturating_add(result.slice.ksub) > weights.k
                    {
                        return Err(Int8DecodePoolError::Worker(format!(
                            "worker {} returned invalid K-split M-tile output geometry",
                            result.worker
                        )));
                    }
                    for (sum, partial) in values.iter_mut().zip(output.values.iter().copied()) {
                        *sum = sum.checked_add(partial).ok_or_else(|| {
                            Int8DecodePoolError::Worker(
                                "K-split M-tile int32 accumulation overflow".to_string(),
                            )
                        })?;
                    }
                }
            }
            npu_tasks = npu_tasks.saturating_add(output.stats.npu_tasks);
            worker_total_ns[result.worker] = output.stats.total_ns;
            worker_stats[result.worker] = output.stats;
        }

        Ok(Int8DecodePoolOutput {
            values,
            stats: Int8DecodePoolStats {
                split: weights.split,
                workers_used: weights.slices.len(),
                npu_tasks,
                wall_ns: start.elapsed().as_nanos(),
                worker_total_ns,
                worker_stats,
            },
        })
    }

    pub fn execute_prepared_mtile_batch(
        &mut self,
        m: usize,
        activations: &[Arc<[i8]>],
        weights: &[&Int8DecodePoolPreparedWeights],
    ) -> Result<Int8DecodePoolMtileBatchOutput, Int8DecodePoolError> {
        if !matches!(m, 4 | 8 | 12 | 16 | 32 | 48 | 64 | 128) {
            return Err(Int8DecodePoolError::InvalidInput(
                "M-tile batch pool path requires M in {4,8,12,16,32,48,64,128}",
            ));
        }
        if activations.is_empty() || activations.len() != weights.len() || activations.len() > 64 {
            return Err(Int8DecodePoolError::InvalidInput(
                "M-tile batch pool path requires 1..=64 matching entries",
            ));
        }
        let first = weights[0];
        if first.split != Int8DecodeSplit::N || first.direct_prepared.is_some() {
            return Err(Int8DecodePoolError::InvalidInput(
                "M-tile batch pool path requires resident N-split weights",
            ));
        }
        let workers = first.slices.len();
        for (activation, prepared) in activations.iter().zip(weights.iter().copied()) {
            if prepared.split != Int8DecodeSplit::N
                || prepared.k != first.k
                || prepared.n != first.n
                || prepared.slices != first.slices
                || prepared.direct_prepared.is_some()
                || activation.len() != m.saturating_mul(first.k)
            {
                return Err(Int8DecodePoolError::InvalidInput(
                    "M-tile batch pool entries must share N-split geometry",
                ));
            }
        }

        let request_id = self.allocate_request_id();
        let start = Instant::now();
        for worker in 0..workers {
            let mut items = Vec::with_capacity(weights.len());
            for (activation, prepared) in activations.iter().zip(weights.iter().copied()) {
                items.push((
                    prepared.weight_id,
                    Arc::clone(activation),
                    prepared.slices[worker],
                ));
            }
            self.senders[worker]
                .send(WorkerCommand::RunPreparedMtileBatch {
                    request_id,
                    m,
                    items,
                })
                .map_err(|_| Int8DecodePoolError::ChannelClosed)?;
        }

        let mut values = (0..weights.len())
            .map(|_| vec![0i32; m.saturating_mul(first.n)])
            .collect::<Vec<_>>();
        let mut npu_tasks = 0usize;
        let mut worker_total_ns = vec![0u128; workers];
        let mut worker_stats = vec![Int8DecodeStats::default(); workers];
        for _ in 0..workers {
            let result = self.recv_result(request_id, workers)?;
            let output = match result.response {
                WorkerResponse::RanMtileBatch(Ok(output)) => output,
                WorkerResponse::RanMtileBatch(Err(error)) => {
                    return Err(Int8DecodePoolError::Worker(error));
                }
                _ => {
                    return Err(Int8DecodePoolError::Worker(
                        "unexpected worker response while executing M-tile batch".to_string(),
                    ));
                }
            };
            if output.values.len() != weights.len()
                || result.slice.n0.saturating_add(result.slice.nsub) > first.n
            {
                return Err(Int8DecodePoolError::Worker(format!(
                    "worker {} returned invalid M-tile batch geometry",
                    result.worker
                )));
            }
            for (group_index, partial) in output.values.iter().enumerate() {
                if partial.len() != m.saturating_mul(result.slice.nsub) {
                    return Err(Int8DecodePoolError::Worker(format!(
                        "worker {} returned invalid M-tile batch group size",
                        result.worker
                    )));
                }
                for row in 0..m {
                    let src = row * result.slice.nsub;
                    let dst = row * first.n + result.slice.n0;
                    values[group_index][dst..dst + result.slice.nsub]
                        .copy_from_slice(&partial[src..src + result.slice.nsub]);
                }
            }
            npu_tasks = npu_tasks.saturating_add(output.stats.npu_tasks);
            worker_total_ns[result.worker] = output.stats.total_ns;
            worker_stats[result.worker] = output.stats;
        }

        Ok(Int8DecodePoolMtileBatchOutput {
            values,
            stats: Int8DecodePoolStats {
                split: Int8DecodeSplit::N,
                workers_used: workers,
                npu_tasks,
                wall_ns: start.elapsed().as_nanos(),
                worker_total_ns,
                worker_stats,
            },
        })
    }

    pub fn execute_prepared_m16(
        &mut self,
        activation: Arc<[i8]>,
        weights: &Int8DecodePoolPreparedWeights,
    ) -> Result<Int8DecodePoolOutput, Int8DecodePoolError> {
        self.execute_prepared_mtile(16, activation, weights)
    }

    pub fn execute_prepared(
        &mut self,
        activation: Arc<[i8]>,
        weights: &Int8DecodePoolPreparedWeights,
    ) -> Result<Int8DecodePoolOutput, Int8DecodePoolError> {
        self.execute_prepared_overlap(activation, weights, None)
    }

    /// Like `execute_prepared`, but runs `overlap` on the calling thread while
    /// the NPU work is in flight (between submission and completion wait) on
    /// the direct-submit paths. Other paths run it before executing. It runs
    /// exactly once when the call succeeds far enough to submit.
    pub fn execute_prepared_overlap(
        &mut self,
        activation: Arc<[i8]>,
        weights: &Int8DecodePoolPreparedWeights,
        overlap: Option<&mut dyn FnMut()>,
    ) -> Result<Int8DecodePoolOutput, Int8DecodePoolError> {
        if activation.len() != weights.k {
            return Err(Int8DecodePoolError::InvalidInput(
                "activation length must equal prepared K",
            ));
        }
        if self.direct_workers.is_some() {
            if env::var_os("ROCKNPU_EXPERIMENT_DIRECT_SCRATCH").is_some() {
                return self.execute_prepared_direct_scratch(&activation, weights, overlap);
            }
            return self.execute_prepared_direct(&activation, weights, overlap);
        }
        if let Some(overlap) = overlap {
            overlap();
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
                .map_err(|_| Int8DecodePoolError::ChannelClosed)?;
        }

        let mut values = vec![0i32; weights.n];
        let mut npu_tasks = 0usize;
        let mut worker_total_ns = vec![0u128; weights.slices.len()];
        let mut worker_stats = vec![Int8DecodeStats::default(); weights.slices.len()];
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
            if output.values.len() != result.slice.nsub
                || result.slice.n0.saturating_add(result.slice.nsub) > weights.n
            {
                return Err(Int8DecodePoolError::Worker(format!(
                    "worker {} returned invalid output geometry",
                    result.worker
                )));
            }
            match weights.split {
                Int8DecodeSplit::N => {
                    values[result.slice.n0..result.slice.n0 + result.slice.nsub]
                        .copy_from_slice(&output.values);
                }
                Int8DecodeSplit::K => {
                    for (sum, partial) in values.iter_mut().zip(output.values) {
                        *sum = sum.checked_add(partial).ok_or_else(|| {
                            Int8DecodePoolError::Worker(
                                "K-split host int32 accumulation overflow".to_string(),
                            )
                        })?;
                    }
                }
            }
            npu_tasks = npu_tasks.saturating_add(output.stats.npu_tasks);
            worker_total_ns[result.worker] = output.stats.total_ns;
            worker_stats[result.worker] = output.stats;
        }

        Ok(Int8DecodePoolOutput {
            values,
            stats: Int8DecodePoolStats {
                split: weights.split,
                workers_used: weights.slices.len(),
                npu_tasks,
                wall_ns: start.elapsed().as_nanos(),
                worker_total_ns,
                worker_stats,
            },
        })
    }

    fn execute_prepared_direct(
        &self,
        activation: &[i8],
        weights: &Int8DecodePoolPreparedWeights,
        overlap: Option<&mut dyn FnMut()>,
    ) -> Result<Int8DecodePoolOutput, Int8DecodePoolError> {
        let direct_workers = self
            .direct_workers
            .as_ref()
            .ok_or_else(|| Int8DecodePoolError::Worker("direct workers missing".to_string()))?;
        let prepared = weights.direct_prepared.as_ref().ok_or_else(|| {
            Int8DecodePoolError::Worker("direct prepared weights missing".to_string())
        })?;
        if prepared.len() != weights.slices.len() || prepared.len() > direct_workers.len() {
            return Err(Int8DecodePoolError::Worker(
                "direct prepared worker geometry mismatch".to_string(),
            ));
        }

        let start = Instant::now();
        let mut pendings = Vec::with_capacity(prepared.len());
        for (worker, (&slice, prepared_worker)) in
            weights.slices.iter().zip(prepared.iter()).enumerate()
        {
            let end = slice.k0.saturating_add(slice.ksub);
            if end > activation.len() {
                return Err(Int8DecodePoolError::InvalidInput(
                    "activation slice out of range",
                ));
            }
            let executor =
                Int8DecodeExecutor::from_externally_guarded_device(&direct_workers[worker].device);
            let pending = executor
                .begin_execute_prepared(&activation[slice.k0..end], prepared_worker)
                .map_err(|err| {
                    Int8DecodePoolError::Worker(format!("direct worker {worker} begin: {err}"))
                })?;
            pendings.push((worker, slice, pending));
        }
        if let Some(overlap) = overlap {
            overlap();
        }

        let mut values = vec![0i32; weights.n];
        let mut npu_tasks = 0usize;
        let mut worker_total_ns = vec![0u128; prepared.len()];
        let mut worker_stats = vec![Int8DecodeStats::default(); prepared.len()];
        for (worker, slice, pending) in pendings {
            let executor =
                Int8DecodeExecutor::from_externally_guarded_device(&direct_workers[worker].device);
            let output = executor.finish_execute_prepared(pending).map_err(|err| {
                Int8DecodePoolError::Worker(format!("direct worker {worker} finish: {err}"))
            })?;
            if output.values.len() != slice.nsub || slice.n0.saturating_add(slice.nsub) > weights.n
            {
                return Err(Int8DecodePoolError::Worker(format!(
                    "direct worker {worker} returned invalid output geometry"
                )));
            }
            match weights.split {
                Int8DecodeSplit::N => {
                    values[slice.n0..slice.n0 + slice.nsub].copy_from_slice(&output.values);
                }
                Int8DecodeSplit::K => {
                    for (sum, partial) in values.iter_mut().zip(output.values) {
                        *sum = sum.checked_add(partial).ok_or_else(|| {
                            Int8DecodePoolError::Worker(
                                "direct K-split host int32 accumulation overflow".to_string(),
                            )
                        })?;
                    }
                }
            }
            npu_tasks = npu_tasks.saturating_add(output.stats.npu_tasks);
            worker_total_ns[worker] = output.stats.total_ns;
            worker_stats[worker] = output.stats;
        }

        Ok(Int8DecodePoolOutput {
            values,
            stats: Int8DecodePoolStats {
                split: weights.split,
                workers_used: weights.slices.len(),
                npu_tasks,
                wall_ns: start.elapsed().as_nanos(),
                worker_total_ns,
                worker_stats,
            },
        })
    }

    fn execute_prepared_direct_scratch(
        &mut self,
        activation: &[i8],
        weights: &Int8DecodePoolPreparedWeights,
        overlap: Option<&mut dyn FnMut()>,
    ) -> Result<Int8DecodePoolOutput, Int8DecodePoolError> {
        let direct_workers = self
            .direct_workers
            .as_mut()
            .ok_or_else(|| Int8DecodePoolError::Worker("direct workers missing".to_string()))?;
        let prepared = weights.direct_prepared.as_ref().ok_or_else(|| {
            Int8DecodePoolError::Worker("direct prepared weights missing".to_string())
        })?;
        if prepared.len() != weights.slices.len() || prepared.len() > direct_workers.len() {
            return Err(Int8DecodePoolError::Worker(
                "direct prepared worker geometry mismatch".to_string(),
            ));
        }

        let start = Instant::now();
        let mut pendings = Vec::with_capacity(prepared.len());
        for (worker, (&slice, prepared_worker)) in
            weights.slices.iter().zip(prepared.iter()).enumerate()
        {
            let end = slice.k0.saturating_add(slice.ksub);
            if end > activation.len() {
                return Err(Int8DecodePoolError::InvalidInput(
                    "activation slice out of range",
                ));
            }
            let worker_state = &mut direct_workers[worker];
            let key = (prepared_worker.k(), prepared_worker.n());
            let scratch_slot = worker_state.scratch.entry(key).or_insert(None);
            let executor = Int8DecodeExecutor::from_externally_guarded_device(&worker_state.device);
            let pending = executor
                .begin_execute_prepared_owned(
                    &activation[slice.k0..end],
                    prepared_worker,
                    scratch_slot,
                )
                .map_err(|err| {
                    Int8DecodePoolError::Worker(format!(
                        "direct scratch worker {worker} begin: {err}"
                    ))
                })?;
            pendings.push((worker, slice, key, pending));
        }
        if let Some(overlap) = overlap {
            overlap();
        }

        let mut values = vec![0i32; weights.n];
        let mut npu_tasks = 0usize;
        let mut worker_total_ns = vec![0u128; prepared.len()];
        let mut worker_stats = vec![Int8DecodeStats::default(); prepared.len()];
        for (worker, slice, key, pending) in pendings {
            let worker_state = &mut direct_workers[worker];
            let scratch_slot = worker_state.scratch.get_mut(&key).ok_or_else(|| {
                Int8DecodePoolError::Worker(format!(
                    "direct scratch worker {worker} missing shape cache"
                ))
            })?;
            if slice.n0.saturating_add(slice.nsub) > weights.n {
                return Err(Int8DecodePoolError::Worker(format!(
                    "direct scratch worker {worker} returned invalid output geometry"
                )));
            }
            let executor = Int8DecodeExecutor::from_externally_guarded_device(&worker_state.device);
            let stats = match weights.split {
                Int8DecodeSplit::N => executor.finish_execute_prepared_owned_into(
                    pending,
                    scratch_slot,
                    &mut values[slice.n0..slice.n0 + slice.nsub],
                ),
                Int8DecodeSplit::K => {
                    executor.finish_execute_prepared_owned_into(pending, scratch_slot, &mut values)
                }
            }
            .map_err(|err| {
                Int8DecodePoolError::Worker(format!("direct scratch worker {worker} finish: {err}"))
            })?;
            npu_tasks = npu_tasks.saturating_add(stats.npu_tasks);
            worker_total_ns[worker] = stats.total_ns;
            worker_stats[worker] = stats;
        }

        Ok(Int8DecodePoolOutput {
            values,
            stats: Int8DecodePoolStats {
                split: weights.split,
                workers_used: weights.slices.len(),
                npu_tasks,
                wall_ns: start.elapsed().as_nanos(),
                worker_total_ns,
                worker_stats,
            },
        })
    }

    pub fn release_prepared(
        &mut self,
        weights: &Int8DecodePoolPreparedWeights,
    ) -> Result<(), Int8DecodePoolError> {
        if weights.direct_prepared.is_some() {
            return Ok(());
        }
        self.release_weight_id(weights.weight_id, &weights.slices)
    }

    fn validate_requested_workers(&self, workers: usize) -> Result<(), Int8DecodePoolError> {
        if workers == 0 || workers > self.workers() {
            return Err(Int8DecodePoolError::InvalidWorkerCount(workers));
        }
        Ok(())
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
        slices: &[WorkerSlice],
    ) -> Result<(), Int8DecodePoolError> {
        let request_id = self.allocate_request_id();
        for (worker, &slice) in slices.iter().enumerate() {
            self.senders[worker]
                .send(WorkerCommand::Release {
                    request_id,
                    weight_id,
                    slice,
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

fn prepare_worker_weights(
    executor: &Int8DecodeExecutor<'_>,
    weights: &[i8],
    full_k: usize,
    slice: WorkerSlice,
    m1_fullk: bool,
) -> Result<Int8PreparedWeights, crate::Int8DecodeError> {
    if slice.k0 == 0 && slice.ksub == full_k {
        let begin = slice.n0.saturating_mul(full_k);
        let end = slice.n0.saturating_add(slice.nsub).saturating_mul(full_k);
        if end > weights.len() {
            return Err(crate::Int8DecodeError::InvalidInput(
                "weight slice out of range",
            ));
        }
        return if m1_fullk {
            executor.prepare_weights_m1_fullk(&weights[begin..end], slice.ksub, slice.nsub)
        } else {
            executor.prepare_weights(&weights[begin..end], slice.ksub, slice.nsub)
        };
    }

    let mut sliced = Vec::with_capacity(slice.nsub.saturating_mul(slice.ksub));
    for row in slice.n0..slice.n0.saturating_add(slice.nsub) {
        let row_start = row.saturating_mul(full_k);
        let begin = row_start.saturating_add(slice.k0);
        let end = begin.saturating_add(slice.ksub);
        if end > weights.len() {
            return Err(crate::Int8DecodeError::InvalidInput(
                "weight slice out of range",
            ));
        }
        sliced.extend_from_slice(&weights[begin..end]);
    }
    executor.prepare_weights(&sliced, slice.ksub, slice.nsub)
}

fn worker_slices(
    k: usize,
    n: usize,
    workers: usize,
    split: Int8DecodeSplit,
) -> Result<Vec<WorkerSlice>, Int8DecodePoolError> {
    match split {
        Int8DecodeSplit::N => Ok(split_n(n, workers)?
            .into_iter()
            .map(|(n0, nsub)| WorkerSlice {
                k0: 0,
                ksub: k,
                n0,
                nsub,
            })
            .collect()),
        Int8DecodeSplit::K => Ok(split_k(k, workers)?
            .into_iter()
            .map(|(k0, ksub)| WorkerSlice {
                k0,
                ksub,
                n0: 0,
                nsub: n,
            })
            .collect()),
    }
}

fn split_n(n: usize, workers: usize) -> Result<Vec<(usize, usize)>, Int8DecodePoolError> {
    if n == 0 || !n.is_multiple_of(32) {
        return Err(Int8DecodePoolError::InvalidInput(
            "N must be non-zero and 32-aligned",
        ));
    }
    split_blocks(n / 32, 32, workers)
}

fn split_k(k: usize, workers: usize) -> Result<Vec<(usize, usize)>, Int8DecodePoolError> {
    if k == 0 || !k.is_multiple_of(512) {
        return Err(Int8DecodePoolError::InvalidInput(
            "K must be non-zero and 512-aligned",
        ));
    }
    split_blocks(k / 512, 512, workers)
}

fn split_blocks(
    blocks: usize,
    block_width: usize,
    workers: usize,
) -> Result<Vec<(usize, usize)>, Int8DecodePoolError> {
    if !(1..=3).contains(&workers) {
        return Err(Int8DecodePoolError::InvalidWorkerCount(workers));
    }
    let active = workers.min(blocks);
    let base = blocks / active;
    let extra = blocks % active;
    let mut slices = Vec::with_capacity(active);
    let mut start = 0usize;
    for worker in 0..active {
        let worker_blocks = base + usize::from(worker < extra);
        let width = worker_blocks * block_width;
        slices.push((start, width));
        start += width;
    }
    Ok(slices)
}

#[cfg(test)]
mod tests {
    use super::{split_k, split_n};

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

    #[test]
    fn three_way_wide_k_split_is_aligned_and_complete() {
        let slices = split_k(5632, 3).unwrap();
        assert_eq!(slices, vec![(0, 2048), (2048, 2048), (4096, 1536)]);
        assert_eq!(slices.iter().map(|(_, k)| k).sum::<usize>(), 5632);
        assert!(slices.iter().all(|(k0, k)| k0 % 512 == 0 && k % 512 == 0));
    }
}
