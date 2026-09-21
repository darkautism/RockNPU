use half::f16;
use rocket_runtime::{RocketBuffer, RocketDevice, Task};
#[cfg(test)]
use rocknpu_regcmd::weight_fp16;
use rocknpu_regcmd::{
    EncodeError, Fp16MatmulDesc, Fp16MatmulPlan, Fp16MatmulTile, encode_fp16_matmul,
    encode_fp16_matmul_accumulate, encode_fp16_matmul_fp32_output, feature_data, plan_fp16_matmul,
    plan_fp16_matmul_compatible_m,
};
use std::collections::HashMap;
use std::fmt;
use std::io;
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};
use std::time::Instant;

mod int4_decode;
mod int4_decode_pool;
mod int8_decode;
mod int8_decode_pool;
mod prepacked;
pub use int4_decode::{
    Int4DecodeError, Int4DecodeExecutor, Int4DecodeOutput, Int4DecodeStats,
    Int4GroupedPreparedWeights, Int4PreparedWeightStats, Int4PreparedWeights,
};
pub use int4_decode_pool::{
    Int4DecodePool, Int4DecodePoolError, Int4DecodePoolOutput, Int4DecodePoolPreparedStats,
    Int4DecodePoolPreparedWeights, Int4DecodePoolStats,
};
pub use int8_decode::{
    Int8DecodeError, Int8DecodeExecutor, Int8DecodeOutput, Int8DecodeStats,
    Int8MtileScratch, Int8PreparedWeightStats, Int8PreparedWeights,
};
pub use int8_decode_pool::{
    Int8DecodePool, Int8DecodePoolError, Int8DecodePoolOutput, Int8DecodePoolPreparedStats,
    Int8DecodePoolPreparedWeights, Int8DecodePoolStats, Int8DecodeSplit,
};
pub use prepacked::{Fp16PrepackedWeights, PrepackedWeightStats};

const WAIT_NS: i64 = 2_000_000_000;
const REGCMD_BYTES: usize = 4096;

#[derive(Debug)]
pub enum MatmulError {
    InvalidInput(&'static str),
    Encode(EncodeError),
    Io(io::Error),
    AddressAbove32Bit(u64),
    Internal(&'static str),
}

impl fmt::Display for MatmulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid fp16 MatMul input: {msg}"),
            Self::Encode(err) => err.fmt(f),
            Self::Io(err) => err.fmt(f),
            Self::AddressAbove32Bit(v) => {
                write!(
                    f,
                    "Rocket BO IOVA 0x{v:x} exceeds the 32-bit RK3588 regcmd field"
                )
            }
            Self::Internal(msg) => write!(f, "fp16 MatMul executor invariant failed: {msg}"),
        }
    }
}
impl std::error::Error for MatmulError {}
impl From<EncodeError> for MatmulError {
    fn from(value: EncodeError) -> Self {
        Self::Encode(value)
    }
}
impl From<io::Error> for MatmulError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KAccumulation {
    None,
    NpuFp16PingPong,
    HostFp32TinyM,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExecutionTiming {
    pub plan_ns: u128,
    pub scratch_ns: u128,
    pub pack_ns: u128,
    pub encode_ns: u128,
    pub regcmd_write_ns: u128,
    pub submit_ns: u128,
    pub wait_ns: u128,
    pub gather_ns: u128,
    pub total_ns: u128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionStats {
    pub plan: Fp16MatmulPlan,
    pub jobs_submitted: usize,
    pub output_tile_groups: usize,
    pub npu_kacc_groups: usize,
    pub host_kacc_groups: usize,
    pub timing: ExecutionTiming,
}

#[derive(Debug, Clone)]
pub struct Fp16MatmulOutput {
    pub values: Vec<f16>,
    pub stats: ExecutionStats,
}

#[derive(Debug, Clone)]
pub struct Fp32MatmulOutput {
    pub values: Vec<f32>,
    pub stats: ExecutionStats,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScratchStats {
    /// Number of CREATE_BO allocations performed for reusable scratch slots.
    pub bo_allocations: usize,
    /// Number of those allocations that replaced an existing too-small slot.
    pub bo_grows: usize,
    pub regcmd_capacity: usize,
    pub input_capacity: usize,
    pub weights_capacity: usize,
    pub output0_capacity: usize,
    pub output1_capacity: usize,
}

#[derive(Default)]
struct ExecutorScratch<'a> {
    regcmd: Option<RocketBuffer<'a>>,
    input: Option<RocketBuffer<'a>>,
    weights: Option<RocketBuffer<'a>>,
    output0: Option<RocketBuffer<'a>>,
    output1: Option<RocketBuffer<'a>>,
    bo_allocations: usize,
    bo_grows: usize,
}

pub struct Fp16MatmulExecutor<'a> {
    device: &'a RocketDevice,
    // Keep IOVA zero occupied for this executor lifetime. The RK3588 PC base is a
    // 32-bit register and all validated executable regcmd buffers are non-zero.
    _guard: RocketBuffer<'a>,
    scratch: ExecutorScratch<'a>,
}

impl<'a> Fp16MatmulExecutor<'a> {
    pub fn new(device: &'a RocketDevice) -> Result<Self, MatmulError> {
        let guard = device.alloc_buffer(4096)?;
        Ok(Self {
            device,
            _guard: guard,
            scratch: ExecutorScratch::default(),
        })
    }

    pub fn scratch_stats(&self) -> ScratchStats {
        ScratchStats {
            bo_allocations: self.scratch.bo_allocations,
            bo_grows: self.scratch.bo_grows,
            regcmd_capacity: self.scratch.regcmd.as_ref().map_or(0, RocketBuffer::len),
            input_capacity: self.scratch.input.as_ref().map_or(0, RocketBuffer::len),
            weights_capacity: self.scratch.weights.as_ref().map_or(0, RocketBuffer::len),
            output0_capacity: self.scratch.output0.as_ref().map_or(0, RocketBuffer::len),
            output1_capacity: self.scratch.output1.as_ref().map_or(0, RocketBuffer::len),
        }
    }

    /// Execute C[M,N] = A[M,K] x B[N,K]^T. A/B/C use row-major host layout.
    ///
    /// NPU EW accumulation is used for K-split output tiles with M>=12. Tiny-M
    /// tiles use exact host-f32 accumulation of the NPU-produced fp16 partials
    /// because the RK3588 EW surface-stride floor below M=12 is not yet accepted
    /// as a production contract.
    pub fn execute(
        &mut self,
        a: &[f16],
        b: &[f16],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Fp16MatmulOutput, MatmulError> {
        let total_start = Instant::now();
        validate_inputs(a, b, m, k, n)?;
        let plan_start = Instant::now();
        let plan = plan_fp16_matmul(m, k, n)?;
        let mut timing = ExecutionTiming {
            plan_ns: plan_start.elapsed().as_nanos(),
            ..ExecutionTiming::default()
        };
        let k_tiles = plan.k_tiles();
        if k_tiles == 0 || plan.tiles.len() % k_tiles != 0 {
            return Err(MatmulError::Internal(
                "planner tile grouping is not rectangular",
            ));
        }

        let mut values = vec![f16::ZERO; m * n];
        let mut jobs = 0usize;
        let mut groups = 0usize;
        let mut npu_kacc_groups = 0usize;
        let mut host_kacc_groups = 0usize;

        for group in plan.tiles.chunks(k_tiles) {
            groups += 1;
            validate_group(group, k)?;
            let mode = accumulation_mode(group);
            match mode {
                KAccumulation::None => {
                    self.execute_single_tile(a, b, m, k, n, group[0], &mut values, &mut timing)?;
                    jobs += 1;
                }
                KAccumulation::NpuFp16PingPong => {
                    self.execute_npu_kacc_group(a, b, m, k, n, group, &mut values, &mut timing)?;
                    jobs += group.len();
                    npu_kacc_groups += 1;
                }
                KAccumulation::HostFp32TinyM => {
                    self.execute_host_kacc_group(a, b, m, k, n, group, &mut values, &mut timing)?;
                    jobs += group.len();
                    host_kacc_groups += 1;
                }
            }
        }

        timing.total_ns = total_start.elapsed().as_nanos();
        Ok(Fp16MatmulOutput {
            values,
            stats: ExecutionStats {
                plan,
                jobs_submitted: jobs,
                output_tile_groups: groups,
                npu_kacc_groups,
                host_kacc_groups,
                timing,
            },
        })
    }

    /// High-accuracy fp16-input MatMul. Every NPU K partial is emitted as
    /// fp32 (C2=4), then K partials are accumulated on the host in f64 and
    /// narrowed once to f32. This intentionally avoids EW accumulation entirely.
    pub fn execute_f32(
        &mut self,
        a: &[f16],
        b: &[f16],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Fp32MatmulOutput, MatmulError> {
        let total_start = Instant::now();
        validate_inputs(a, b, m, k, n)?;
        let plan_start = Instant::now();
        let plan = plan_fp16_matmul(m, k, n)?;
        let mut timing = ExecutionTiming {
            plan_ns: plan_start.elapsed().as_nanos(),
            ..ExecutionTiming::default()
        };
        let k_tiles = plan.k_tiles();
        if k_tiles == 0 || plan.tiles.len() % k_tiles != 0 {
            return Err(MatmulError::Internal(
                "planner tile grouping is not rectangular",
            ));
        }
        let mut values = vec![0.0f32; m * n];
        let mut jobs = 0usize;
        let mut groups = 0usize;
        let mut host_kacc_groups = 0usize;

        for group in plan.tiles.chunks(k_tiles) {
            groups += 1;
            validate_group(group, k)?;
            let first = group[0];
            let mut acc = vec![0.0f64; first.m * first.n];
            for &tile in group {
                let phase = Instant::now();
                self.ensure_scratch(tile, 4, false)?;
                timing.scratch_ns += phase.elapsed().as_nanos();
                let ExecutorScratch {
                    regcmd: Some(regcmd),
                    input: Some(input),
                    weights: Some(weights),
                    output0: Some(output),
                    ..
                } = &mut self.scratch
                else {
                    return Err(MatmulError::Internal("fp32 scratch allocation invariant"));
                };
                let phase = Instant::now();
                pack_input(input, a, k, tile)?;
                pack_weights(weights, b, k, tile)?;
                prepare_output(output)?;
                timing.pack_ns += phase.elapsed().as_nanos();
                let phase = Instant::now();
                let ops = encode_fp16_matmul_fp32_output(Fp16MatmulDesc::new(
                    tile.m,
                    tile.k,
                    tile.n,
                    input.dma_address(),
                    weights.dma_address(),
                    output.dma_address(),
                ))?;
                timing.encode_ns += phase.elapsed().as_nanos();
                let phase = Instant::now();
                write_regcmd(regcmd, &ops)?;
                timing.regcmd_write_ns += phase.elapsed().as_nanos();
                let phase = Instant::now();
                submit_plain(self.device, regcmd, input, weights, output, ops.len())?;
                timing.submit_ns += phase.elapsed().as_nanos();
                let phase = Instant::now();
                output.prep_relative(WAIT_NS)?;
                timing.wait_ns += phase.elapsed().as_nanos();
                let phase = Instant::now();
                for tm in 0..tile.m {
                    for tn in 0..tile.n {
                        let native = feature_data(tile.n, tile.m, 1, 4, tn + 1, tm + 1, 1);
                        acc[tm * tile.n + tn] += get_f32(output.as_slice(), native) as f64;
                    }
                }
                timing.gather_ns += phase.elapsed().as_nanos();
                output.fini()?;
                jobs += 1;
            }
            if group.len() > 1 {
                host_kacc_groups += 1;
            }
            for tm in 0..first.m {
                for tn in 0..first.n {
                    values[(first.m0 + tm) * n + first.n0 + tn] = acc[tm * first.n + tn] as f32;
                }
            }
        }

        timing.total_ns = total_start.elapsed().as_nanos();
        Ok(Fp32MatmulOutput {
            values,
            stats: ExecutionStats {
                plan,
                jobs_submitted: jobs,
                output_tile_groups: groups,
                npu_kacc_groups: 0,
                host_kacc_groups,
                timing,
            },
        })
    }

    fn execute_single_tile(
        &mut self,
        a: &[f16],
        b: &[f16],
        _m: usize,
        k_total: usize,
        n_total: usize,
        tile: Fp16MatmulTile,
        dst: &mut [f16],
        timing: &mut ExecutionTiming,
    ) -> Result<(), MatmulError> {
        let phase = Instant::now();
        self.ensure_scratch(tile, 2, false)?;
        timing.scratch_ns += phase.elapsed().as_nanos();
        let ExecutorScratch {
            regcmd: Some(regcmd),
            input: Some(input),
            weights: Some(weights),
            output0: Some(output),
            ..
        } = &mut self.scratch
        else {
            return Err(MatmulError::Internal(
                "single-tile scratch allocation invariant",
            ));
        };
        let phase = Instant::now();
        pack_input(input, a, k_total, tile)?;
        pack_weights(weights, b, k_total, tile)?;
        prepare_output(output)?;
        timing.pack_ns += phase.elapsed().as_nanos();
        let phase = Instant::now();
        let ops = encode_fp16_matmul(Fp16MatmulDesc::new(
            tile.m,
            tile.k,
            tile.n,
            input.dma_address(),
            weights.dma_address(),
            output.dma_address(),
        ))?;
        timing.encode_ns += phase.elapsed().as_nanos();
        let phase = Instant::now();
        write_regcmd(regcmd, &ops)?;
        timing.regcmd_write_ns += phase.elapsed().as_nanos();
        let phase = Instant::now();
        submit_plain(self.device, regcmd, input, weights, output, ops.len())?;
        timing.submit_ns += phase.elapsed().as_nanos();
        let phase = Instant::now();
        output.prep_relative(WAIT_NS)?;
        timing.wait_ns += phase.elapsed().as_nanos();
        let phase = Instant::now();
        gather_tile(output.as_slice(), tile, n_total, dst);
        timing.gather_ns += phase.elapsed().as_nanos();
        output.fini()?;
        Ok(())
    }

    fn execute_npu_kacc_group(
        &mut self,
        a: &[f16],
        b: &[f16],
        _m: usize,
        k_total: usize,
        n_total: usize,
        group: &[Fp16MatmulTile],
        dst: &mut [f16],
        timing: &mut ExecutionTiming,
    ) -> Result<(), MatmulError> {
        let first = group[0];
        if first.m < 12 {
            return Err(MatmulError::Internal(
                "tiny-M group routed to NPU EW accumulation",
            ));
        }
        let phase = Instant::now();
        self.ensure_scratch(first, 2, true)?;
        timing.scratch_ns += phase.elapsed().as_nanos();
        {
            let ExecutorScratch {
                output0: Some(ping),
                output1: Some(pong),
                ..
            } = &mut self.scratch
            else {
                return Err(MatmulError::Internal(
                    "KACC output scratch allocation invariant",
                ));
            };
            let phase = Instant::now();
            prepare_output(ping)?;
            prepare_output(pong)?;
            timing.pack_ns += phase.elapsed().as_nanos();
        }

        for (ki, &tile) in group.iter().enumerate() {
            let phase = Instant::now();
            self.ensure_scratch(tile, 2, true)?;
            timing.scratch_ns += phase.elapsed().as_nanos();
            let ExecutorScratch {
                regcmd: Some(regcmd),
                input: Some(input),
                weights: Some(weights),
                output0: Some(ping),
                output1: Some(pong),
                ..
            } = &mut self.scratch
            else {
                return Err(MatmulError::Internal("KACC scratch allocation invariant"));
            };
            let phase = Instant::now();
            pack_input(input, a, k_total, tile)?;
            pack_weights(weights, b, k_total, tile)?;
            timing.pack_ns += phase.elapsed().as_nanos();

            let (out, add) = if ki % 2 == 0 {
                (ping, pong)
            } else {
                (pong, ping)
            };
            // PREP/FINI hands the destination back to the device before WDMA.
            out.prep_relative(0)?;
            out.fini()?;
            let desc = Fp16MatmulDesc::new(
                tile.m,
                tile.k,
                tile.n,
                input.dma_address(),
                weights.dma_address(),
                out.dma_address(),
            );
            let phase = Instant::now();
            let ops = if ki == 0 {
                encode_fp16_matmul(desc)?
            } else {
                encode_fp16_matmul_accumulate(desc, add.dma_address())?
            };
            timing.encode_ns += phase.elapsed().as_nanos();
            let phase = Instant::now();
            write_regcmd(regcmd, &ops)?;
            timing.regcmd_write_ns += phase.elapsed().as_nanos();
            let phase = Instant::now();
            if ki == 0 {
                submit_plain(self.device, regcmd, input, weights, out, ops.len())?;
            } else {
                submit_accumulate(self.device, regcmd, input, weights, add, out, ops.len())?;
            }
            timing.submit_ns += phase.elapsed().as_nanos();
            let phase = Instant::now();
            out.prep_relative(WAIT_NS)?;
            timing.wait_ns += phase.elapsed().as_nanos();
            out.fini()?;
        }

        let final_out = if (group.len() - 1) % 2 == 0 {
            self.scratch.output0.as_ref()
        } else {
            self.scratch.output1.as_ref()
        }
        .ok_or(MatmulError::Internal("KACC final output scratch missing"))?;
        final_out.prep_relative(0)?;
        let phase = Instant::now();
        gather_tile(final_out.as_slice(), group[0], n_total, dst);
        timing.gather_ns += phase.elapsed().as_nanos();
        final_out.fini()?;
        Ok(())
    }

    fn execute_host_kacc_group(
        &mut self,
        a: &[f16],
        b: &[f16],
        _m: usize,
        k_total: usize,
        n_total: usize,
        group: &[Fp16MatmulTile],
        dst: &mut [f16],
        timing: &mut ExecutionTiming,
    ) -> Result<(), MatmulError> {
        let first = group[0];
        let mut acc = vec![0.0f32; first.m * first.n];
        for &tile in group {
            let phase = Instant::now();
            self.ensure_scratch(tile, 2, false)?;
            timing.scratch_ns += phase.elapsed().as_nanos();
            let ExecutorScratch {
                regcmd: Some(regcmd),
                input: Some(input),
                weights: Some(weights),
                output0: Some(output),
                ..
            } = &mut self.scratch
            else {
                return Err(MatmulError::Internal(
                    "host-KACC scratch allocation invariant",
                ));
            };
            let phase = Instant::now();
            pack_input(input, a, k_total, tile)?;
            pack_weights(weights, b, k_total, tile)?;
            prepare_output(output)?;
            timing.pack_ns += phase.elapsed().as_nanos();
            let phase = Instant::now();
            let ops = encode_fp16_matmul(Fp16MatmulDesc::new(
                tile.m,
                tile.k,
                tile.n,
                input.dma_address(),
                weights.dma_address(),
                output.dma_address(),
            ))?;
            timing.encode_ns += phase.elapsed().as_nanos();
            let phase = Instant::now();
            write_regcmd(regcmd, &ops)?;
            timing.regcmd_write_ns += phase.elapsed().as_nanos();
            let phase = Instant::now();
            submit_plain(self.device, regcmd, input, weights, output, ops.len())?;
            timing.submit_ns += phase.elapsed().as_nanos();
            let phase = Instant::now();
            output.prep_relative(WAIT_NS)?;
            timing.wait_ns += phase.elapsed().as_nanos();
            let phase = Instant::now();
            for tm in 0..tile.m {
                for tn in 0..tile.n {
                    let native = feature_data(tile.n, tile.m, 1, 8, tn + 1, tm + 1, 1);
                    acc[tm * tile.n + tn] += get_f16(output.as_slice(), native).to_f32();
                }
            }
            timing.gather_ns += phase.elapsed().as_nanos();
            output.fini()?;
        }
        let phase = Instant::now();
        for tm in 0..first.m {
            for tn in 0..first.n {
                dst[(first.m0 + tm) * n_total + first.n0 + tn] =
                    f16::from_f32(acc[tm * first.n + tn]);
            }
        }
        timing.gather_ns += phase.elapsed().as_nanos();
        Ok(())
    }

    fn ensure_scratch(
        &mut self,
        tile: Fp16MatmulTile,
        output_elem_bytes: usize,
        need_second_output: bool,
    ) -> Result<(), MatmulError> {
        let input_bytes = tile
            .m
            .checked_mul(tile.k)
            .and_then(|v| v.checked_mul(2))
            .ok_or(MatmulError::Internal("input scratch size overflow"))?;
        let weight_bytes = tile
            .n
            .checked_mul(tile.k)
            .and_then(|v| v.checked_mul(2))
            .ok_or(MatmulError::Internal("weight scratch size overflow"))?;
        let output_bytes = tile
            .m
            .checked_mul(tile.n)
            .and_then(|v| v.checked_mul(output_elem_bytes))
            .ok_or(MatmulError::Internal("output scratch size overflow"))?;
        ensure_slot(
            self.device,
            &mut self.scratch.regcmd,
            REGCMD_BYTES,
            &mut self.scratch.bo_allocations,
            &mut self.scratch.bo_grows,
        )?;
        ensure_slot(
            self.device,
            &mut self.scratch.input,
            input_bytes,
            &mut self.scratch.bo_allocations,
            &mut self.scratch.bo_grows,
        )?;
        ensure_slot(
            self.device,
            &mut self.scratch.weights,
            weight_bytes,
            &mut self.scratch.bo_allocations,
            &mut self.scratch.bo_grows,
        )?;
        ensure_slot(
            self.device,
            &mut self.scratch.output0,
            output_bytes,
            &mut self.scratch.bo_allocations,
            &mut self.scratch.bo_grows,
        )?;
        if need_second_output {
            ensure_slot(
                self.device,
                &mut self.scratch.output1,
                output_bytes,
                &mut self.scratch.bo_allocations,
                &mut self.scratch.bo_grows,
            )?;
        }
        Ok(())
    }
}

fn ensure_slot<'a>(
    device: &'a RocketDevice,
    slot: &mut Option<RocketBuffer<'a>>,
    required: usize,
    allocations: &mut usize,
    grows: &mut usize,
) -> Result<(), MatmulError> {
    if slot.as_ref().is_some_and(|bo| bo.len() >= required) {
        return Ok(());
    }
    let replacing = slot.is_some();
    let bo = device.alloc_buffer(required)?;
    if bo.dma_address() >> 32 != 0 {
        return Err(MatmulError::AddressAbove32Bit(bo.dma_address()));
    }
    *slot = Some(bo);
    *allocations += 1;
    if replacing {
        *grows += 1;
    }
    Ok(())
}

#[derive(Debug)]
pub enum MatmulPoolError {
    InvalidWorkerCount(usize),
    InvalidInput(&'static str),
    Worker(String),
    ChannelClosed,
}

impl fmt::Display for MatmulPoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWorkerCount(n) => {
                write!(f, "RK3588 MatMul pool requires 1..=3 workers, got {n}")
            }
            Self::InvalidInput(msg) => write!(f, "invalid multicore MatMul input: {msg}"),
            Self::Worker(msg) => write!(f, "multicore MatMul worker failed: {msg}"),
            Self::ChannelClosed => write!(f, "multicore MatMul worker channel closed"),
        }
    }
}
impl std::error::Error for MatmulPoolError {}

#[derive(Debug, Clone)]
pub struct MulticoreExecutionStats {
    pub workers_used: usize,
    pub jobs_submitted: usize,
    pub wall_ns: u128,
    pub worker_timings: Vec<ExecutionTiming>,
    pub worker_scratch: Vec<ScratchStats>,
}

#[derive(Debug, Clone)]
pub struct Fp16MatmulPoolOutput {
    pub values: Vec<f16>,
    pub stats: MulticoreExecutionStats,
}

#[derive(Debug, Clone)]
pub struct Fp32MatmulPoolOutput {
    pub values: Vec<f32>,
    pub stats: MulticoreExecutionStats,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoolPreparedWeightStats {
    pub workers: usize,
    pub resident_bytes: usize,
    pub unique_tiles: usize,
    pub worker_total_ns_max: u128,
    pub plan_ns_max: u128,
    pub layout_ns_max: u128,
    pub alloc_mmap_ns_max: u128,
    pub prep_ns_max: u128,
    pub zero_ns_max: u128,
    pub tile_pack_ns_max: u128,
    pub fini_ns_max: u128,
    pub pack_ns_sum: u128,
    pub pack_ns_max: u128,
    pub prepare_wall_ns: u128,
}

pub struct Fp16MatmulPoolPreparedWeights {
    weight_id: u64,
    exact_m: Option<usize>,
    k: usize,
    n: usize,
    slices: Vec<(usize, usize)>,
    stats: PoolPreparedWeightStats,
}

impl Fp16MatmulPoolPreparedWeights {
    pub const fn k(&self) -> usize {
        self.k
    }
    pub const fn n(&self) -> usize {
        self.n
    }
    pub const fn stats(&self) -> PoolPreparedWeightStats {
        self.stats
    }
}

#[derive(Clone)]
enum PoolPreparedWeightSource {
    Slice(Arc<[f16]>),
    Vec(Arc<Vec<f16>>),
}

impl PoolPreparedWeightSource {
    fn as_slice(&self) -> &[f16] {
        match self {
            Self::Slice(values) => values,
            Self::Vec(values) => values.as_slice(),
        }
    }

    fn len(&self) -> usize {
        self.as_slice().len()
    }
}

enum PoolCommand {
    Run {
        request_id: u64,
        a: Arc<[f16]>,
        b: Arc<[f16]>,
        m: usize,
        k: usize,
        n0: usize,
        nsub: usize,
    },
    RunF32 {
        request_id: u64,
        a: Arc<[f16]>,
        b: Arc<[f16]>,
        m: usize,
        k: usize,
        n0: usize,
        nsub: usize,
    },
    Prepare {
        exact_m: Option<usize>,
        request_id: u64,
        weight_id: u64,
        b: PoolPreparedWeightSource,
        k: usize,
        n0: usize,
        nsub: usize,
    },
    RunPrepared {
        request_id: u64,
        weight_id: u64,
        a: Arc<[f16]>,
        m: usize,
        k: usize,
        n0: usize,
        nsub: usize,
    },
    RunPreparedF32 {
        request_id: u64,
        weight_id: u64,
        a: Arc<[f16]>,
        m: usize,
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

enum PoolWorkerResponse {
    Ran {
        output: Result<Fp16MatmulOutput, String>,
        scratch: ScratchStats,
    },
    RanF32 {
        output: Result<Fp32MatmulOutput, String>,
        scratch: ScratchStats,
    },
    Prepared(Result<PrepackedWeightStats, String>),
    Released,
}

struct PoolWorkerResult {
    request_id: u64,
    worker: usize,
    n0: usize,
    nsub: usize,
    response: PoolWorkerResponse,
}

pub struct Fp16MatmulPool {
    senders: Vec<mpsc::Sender<PoolCommand>>,
    result_rx: mpsc::Receiver<PoolWorkerResult>,
    handles: Vec<JoinHandle<()>>,
    next_request_id: u64,
    next_weight_id: u64,
}

impl Fp16MatmulPool {
    pub fn new(workers: usize) -> Result<Self, MatmulPoolError> {
        if !(1..=3).contains(&workers) {
            return Err(MatmulPoolError::InvalidWorkerCount(workers));
        }
        let (result_tx, result_rx) = mpsc::channel::<PoolWorkerResult>();
        let (init_tx, init_rx) = mpsc::channel::<Result<usize, String>>();
        let mut senders = Vec::with_capacity(workers);
        let mut handles = Vec::with_capacity(workers);
        for worker in 0..workers {
            let (cmd_tx, cmd_rx) = mpsc::channel::<PoolCommand>();
            senders.push(cmd_tx);
            let result_tx = result_tx.clone();
            let init_tx = init_tx.clone();
            handles.push(thread::spawn(move || {
                let device = match RocketDevice::open() {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = init_tx.send(Err(format!("worker {worker} open: {e}")));
                        return;
                    }
                };
                let mut executor = match Fp16MatmulExecutor::new(&device) {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = init_tx.send(Err(format!("worker {worker} executor: {e}")));
                        return;
                    }
                };
                let mut resident: HashMap<u64, Fp16PrepackedWeights<'_>> = HashMap::new();
                if init_tx.send(Ok(worker)).is_err() {
                    return;
                }
                while let Ok(command) = cmd_rx.recv() {
                    match command {
                        PoolCommand::Stop => break,
                        PoolCommand::Run {
                            request_id,
                            a,
                            b,
                            m,
                            k,
                            n0,
                            nsub,
                        } => {
                            let begin = n0.saturating_mul(k);
                            let end = n0.saturating_add(nsub).saturating_mul(k);
                            let output = if end <= b.len() {
                                executor
                                    .execute(&a, &b[begin..end], m, k, nsub)
                                    .map_err(|e| format!("worker {worker}: {e}"))
                            } else {
                                Err(format!("worker {worker}: B slice out of range"))
                            };
                            let scratch = executor.scratch_stats();
                            if result_tx
                                .send(PoolWorkerResult {
                                    request_id,
                                    worker,
                                    n0,
                                    nsub,
                                    response: PoolWorkerResponse::Ran { output, scratch },
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        PoolCommand::RunF32 {
                            request_id,
                            a,
                            b,
                            m,
                            k,
                            n0,
                            nsub,
                        } => {
                            let begin = n0.saturating_mul(k);
                            let end = n0.saturating_add(nsub).saturating_mul(k);
                            let output = if end <= b.len() {
                                executor
                                    .execute_f32(&a, &b[begin..end], m, k, nsub)
                                    .map_err(|e| format!("worker {worker}: {e}"))
                            } else {
                                Err(format!("worker {worker}: B slice out of range"))
                            };
                            let scratch = executor.scratch_stats();
                            if result_tx
                                .send(PoolWorkerResult {
                                    request_id,
                                    worker,
                                    n0,
                                    nsub,
                                    response: PoolWorkerResponse::RanF32 { output, scratch },
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        PoolCommand::Prepare {
                            exact_m,
                            request_id,
                            weight_id,
                            b,
                            k,
                            n0,
                            nsub,
                        } => {
                            let begin = n0.saturating_mul(k);
                            let end = n0.saturating_add(nsub).saturating_mul(k);
                            let prepared = if resident.contains_key(&weight_id) {
                                Err(format!("worker {worker}: duplicate resident weight id"))
                            } else if end > b.len() {
                                Err(format!("worker {worker}: B slice out of range"))
                            } else {
                                let b = b.as_slice();
                                let packed = match exact_m {
                                    Some(m) => executor.prepack_weights(&b[begin..end], m, k, nsub),
                                    None => executor.prepack_weights_compatible_m(
                                        &b[begin..end],
                                        k,
                                        nsub,
                                    ),
                                };
                                match packed {
                                    Ok(weights) => {
                                        let stats = weights.stats();
                                        resident.insert(weight_id, weights);
                                        Ok(stats)
                                    }
                                    Err(e) => Err(format!("worker {worker}: {e}")),
                                }
                            };
                            if result_tx
                                .send(PoolWorkerResult {
                                    request_id,
                                    worker,
                                    n0,
                                    nsub,
                                    response: PoolWorkerResponse::Prepared(prepared),
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        PoolCommand::RunPrepared {
                            request_id,
                            weight_id,
                            a,
                            m,
                            k,
                            n0,
                            nsub,
                        } => {
                            let output = match resident.get(&weight_id) {
                                Some(weights) if weights.k() == k && weights.n() == nsub => {
                                    executor
                                        .execute_prepacked_compatible_m(&a, m, weights)
                                        .map_err(|e| format!("worker {worker}: {e}"))
                                }
                                Some(_) => {
                                    Err(format!("worker {worker}: resident weight shape mismatch"))
                                }
                                None => {
                                    Err(format!("worker {worker}: resident weight id not found"))
                                }
                            };
                            let scratch = executor.scratch_stats();
                            if result_tx
                                .send(PoolWorkerResult {
                                    request_id,
                                    worker,
                                    n0,
                                    nsub,
                                    response: PoolWorkerResponse::Ran { output, scratch },
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        PoolCommand::RunPreparedF32 {
                            request_id,
                            weight_id,
                            a,
                            m,
                            k,
                            n0,
                            nsub,
                        } => {
                            let output = match resident.get(&weight_id) {
                                Some(weights)
                                    if weights.k() == k
                                        && weights.n() == nsub
                                        && weights.m() == m =>
                                {
                                    executor
                                        .execute_prepacked_f32(&a, weights)
                                        .map_err(|e| format!("worker {worker}: {e}"))
                                }
                                Some(_) => {
                                    Err(format!("worker {worker}: resident weight shape mismatch"))
                                }
                                None => {
                                    Err(format!("worker {worker}: resident weight id not found"))
                                }
                            };
                            let scratch = executor.scratch_stats();
                            if result_tx
                                .send(PoolWorkerResult {
                                    request_id,
                                    worker,
                                    n0,
                                    nsub,
                                    response: PoolWorkerResponse::RanF32 { output, scratch },
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        PoolCommand::Release {
                            request_id,
                            weight_id,
                            n0,
                            nsub,
                        } => {
                            resident.remove(&weight_id);
                            if result_tx
                                .send(PoolWorkerResult {
                                    request_id,
                                    worker,
                                    n0,
                                    nsub,
                                    response: PoolWorkerResponse::Released,
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
                Ok(Err(e)) => {
                    for tx in &senders {
                        let _ = tx.send(PoolCommand::Stop);
                    }
                    for h in handles.drain(..) {
                        let _ = h.join();
                    }
                    return Err(MatmulPoolError::Worker(e));
                }
                Err(_) => return Err(MatmulPoolError::ChannelClosed),
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
    ) -> Result<usize, MatmulPoolError> {
        if requested == 0 || requested > self.senders.len() {
            return Err(MatmulPoolError::InvalidWorkerCount(requested));
        }
        if n == 0 || n % 16 != 0 {
            return Err(MatmulPoolError::InvalidInput(
                "N must be non-zero and 16-aligned",
            ));
        }
        Ok(split_n_slices(n, requested).len())
    }

    fn allocate_request_id(&mut self) -> u64 {
        let id = self.next_request_id;
        self.next_request_id = self.next_request_id.wrapping_add(1);
        id
    }

    fn recv_result(
        &self,
        request_id: u64,
        workers: usize,
    ) -> Result<PoolWorkerResult, MatmulPoolError> {
        let result = self
            .result_rx
            .recv()
            .map_err(|_| MatmulPoolError::ChannelClosed)?;
        if result.request_id != request_id || result.worker >= workers {
            return Err(MatmulPoolError::Worker(
                "unexpected worker response".to_string(),
            ));
        }
        Ok(result)
    }

    pub fn execute(
        &mut self,
        a: Arc<[f16]>,
        b: Arc<[f16]>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Fp16MatmulPoolOutput, MatmulPoolError> {
        self.execute_with_workers(a, b, m, k, n, self.senders.len())
    }

    pub fn execute_with_workers(
        &mut self,
        a: Arc<[f16]>,
        b: Arc<[f16]>,
        m: usize,
        k: usize,
        n: usize,
        workers: usize,
    ) -> Result<Fp16MatmulPoolOutput, MatmulPoolError> {
        validate_inputs(&a, &b, m, k, n)
            .map_err(|_| MatmulPoolError::InvalidInput("A/B lengths or shape"))?;
        if workers == 0 || workers > self.senders.len() {
            return Err(MatmulPoolError::InvalidWorkerCount(workers));
        }
        if m == 0 || k == 0 || n == 0 || m % 4 != 0 || k % 32 != 0 || n % 16 != 0 {
            return Err(MatmulPoolError::InvalidInput(
                "requires M%4==0, K%32==0, N%16==0",
            ));
        }
        let slices = split_n_slices(n, workers);
        let request_id = self.allocate_request_id();
        let start = Instant::now();
        for (worker, &(n0, nsub)) in slices.iter().enumerate() {
            self.senders[worker]
                .send(PoolCommand::Run {
                    request_id,
                    a: Arc::clone(&a),
                    b: Arc::clone(&b),
                    m,
                    k,
                    n0,
                    nsub,
                })
                .map_err(|_| MatmulPoolError::ChannelClosed)?;
        }
        self.gather_run(request_id, &slices, m, n, start)
    }

    /// FP16-input / FP32-output multicore MatMul. Each worker owns a separate
    /// Rocket fd/entity and computes one aligned N slice using `execute_f32`;
    /// row-major FP32 slices are gathered without narrowing.
    pub fn execute_f32(
        &mut self,
        a: Arc<[f16]>,
        b: Arc<[f16]>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Fp32MatmulPoolOutput, MatmulPoolError> {
        self.execute_f32_with_workers(a, b, m, k, n, self.senders.len())
    }

    pub fn execute_f32_with_workers(
        &mut self,
        a: Arc<[f16]>,
        b: Arc<[f16]>,
        m: usize,
        k: usize,
        n: usize,
        workers: usize,
    ) -> Result<Fp32MatmulPoolOutput, MatmulPoolError> {
        validate_inputs(&a, &b, m, k, n)
            .map_err(|_| MatmulPoolError::InvalidInput("A/B lengths or shape"))?;
        if workers == 0 || workers > self.senders.len() {
            return Err(MatmulPoolError::InvalidWorkerCount(workers));
        }
        if m == 0 || k == 0 || n == 0 || m % 4 != 0 || k % 32 != 0 || n % 16 != 0 {
            return Err(MatmulPoolError::InvalidInput(
                "FP32 pool requires M%4==0, K%32==0, N%16==0",
            ));
        }
        let slices = split_n_slices(n, workers);
        let request_id = self.allocate_request_id();
        let start = Instant::now();
        for (worker, &(n0, nsub)) in slices.iter().enumerate() {
            self.senders[worker]
                .send(PoolCommand::RunF32 {
                    request_id,
                    a: Arc::clone(&a),
                    b: Arc::clone(&b),
                    m,
                    k,
                    n0,
                    nsub,
                })
                .map_err(|_| MatmulPoolError::ChannelClosed)?;
        }
        self.gather_run_f32(request_id, &slices, m, n, start)
    }

    /// Prepack one B[N,K] into each active worker's own Rocket fd/IOMMU domain.
    /// The returned handle stores only an ID/shape; host B is not needed by
    /// subsequent `execute_prepared` calls.
    pub fn prepare_weights_compatible_m(
        &mut self,
        b: Arc<[f16]>,
        k: usize,
        n: usize,
    ) -> Result<Fp16MatmulPoolPreparedWeights, MatmulPoolError> {
        self.prepare_weights_impl(PoolPreparedWeightSource::Slice(b), None, k, n)
    }

    /// Prepare resident weights for the exact FP32 execution plan at M.
    pub fn prepare_weights_f32(
        &mut self,
        b: Arc<[f16]>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Fp16MatmulPoolPreparedWeights, MatmulPoolError> {
        if m == 0 || m % 4 != 0 {
            return Err(MatmulPoolError::InvalidInput(
                "FP32 prepared M must be a positive multiple of four",
            ));
        }
        self.prepare_weights_impl(PoolPreparedWeightSource::Slice(b), Some(m), k, n)
    }

    /// Prepare resident exact-M FP32 weights from an existing Vec allocation.
    /// `Arc::new(Vec)` moves only the Vec header, so workers share the original
    /// f16 allocation without the bulk copy required by Arc<[f16]>.
    pub fn prepare_weights_f32_vec(
        &mut self,
        b: Arc<Vec<f16>>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Fp16MatmulPoolPreparedWeights, MatmulPoolError> {
        if m == 0 || m % 4 != 0 {
            return Err(MatmulPoolError::InvalidInput(
                "FP32 prepared M must be a positive multiple of four",
            ));
        }
        self.prepare_weights_impl(PoolPreparedWeightSource::Vec(b), Some(m), k, n)
    }

    fn prepare_weights_impl(
        &mut self,
        b: PoolPreparedWeightSource,
        exact_m: Option<usize>,
        k: usize,
        n: usize,
    ) -> Result<Fp16MatmulPoolPreparedWeights, MatmulPoolError> {
        if k == 0 || n == 0 || k % 32 != 0 || n % 16 != 0 || b.len() != n.saturating_mul(k) {
            return Err(MatmulPoolError::InvalidInput(
                "prepared pool B requires exact N*K length, K%32==0, N%16==0",
            ));
        }
        let slices = split_n_slices(n, self.senders.len());
        let request_id = self.allocate_request_id();
        let weight_id = self.next_weight_id;
        self.next_weight_id = self.next_weight_id.wrapping_add(1);
        let start = Instant::now();
        for (worker, &(n0, nsub)) in slices.iter().enumerate() {
            self.senders[worker]
                .send(PoolCommand::Prepare {
                    exact_m,
                    request_id,
                    weight_id,
                    b: b.clone(),
                    k,
                    n0,
                    nsub,
                })
                .map_err(|_| MatmulPoolError::ChannelClosed)?;
        }

        let mut stats = PoolPreparedWeightStats {
            workers: slices.len(),
            ..PoolPreparedWeightStats::default()
        };
        let mut first_error = None::<String>;
        for _ in 0..slices.len() {
            let result = self.recv_result(request_id, slices.len())?;
            match result.response {
                PoolWorkerResponse::Prepared(Ok(ws)) => {
                    stats.resident_bytes += ws.resident_bytes;
                    stats.unique_tiles += ws.unique_tiles;
                    stats.worker_total_ns_max = stats.worker_total_ns_max.max(ws.total_ns);
                    stats.plan_ns_max = stats.plan_ns_max.max(ws.plan_ns);
                    stats.layout_ns_max = stats.layout_ns_max.max(ws.layout_ns);
                    stats.alloc_mmap_ns_max = stats.alloc_mmap_ns_max.max(ws.alloc_mmap_ns);
                    stats.prep_ns_max = stats.prep_ns_max.max(ws.prep_ns);
                    stats.zero_ns_max = stats.zero_ns_max.max(ws.zero_ns);
                    stats.tile_pack_ns_max = stats.tile_pack_ns_max.max(ws.tile_pack_ns);
                    stats.fini_ns_max = stats.fini_ns_max.max(ws.fini_ns);
                    stats.pack_ns_sum += ws.pack_ns;
                    stats.pack_ns_max = stats.pack_ns_max.max(ws.pack_ns);
                }
                PoolWorkerResponse::Prepared(Err(e)) => {
                    first_error.get_or_insert(e);
                }
                _ => {
                    first_error.get_or_insert_with(|| {
                        "unexpected response while preparing resident weights".to_string()
                    });
                }
            }
        }
        stats.prepare_wall_ns = start.elapsed().as_nanos();
        if let Some(e) = first_error {
            let _ = self.release_weight_id(weight_id, &slices);
            return Err(MatmulPoolError::Worker(e));
        }
        Ok(Fp16MatmulPoolPreparedWeights {
            exact_m,
            weight_id,
            k,
            n,
            slices,
            stats,
        })
    }

    pub fn execute_prepared(
        &mut self,
        a: Arc<[f16]>,
        m: usize,
        weights: &Fp16MatmulPoolPreparedWeights,
    ) -> Result<Fp16MatmulPoolOutput, MatmulPoolError> {
        if m == 0 || m % 4 != 0 || a.len() != m.saturating_mul(weights.k) {
            return Err(MatmulPoolError::InvalidInput(
                "prepared pool A requires exact M*K length and M%4==0",
            ));
        }
        let request_id = self.allocate_request_id();
        let start = Instant::now();
        for (worker, &(n0, nsub)) in weights.slices.iter().enumerate() {
            self.senders[worker]
                .send(PoolCommand::RunPrepared {
                    request_id,
                    weight_id: weights.weight_id,
                    a: Arc::clone(&a),
                    m,
                    k: weights.k,
                    n0,
                    nsub,
                })
                .map_err(|_| MatmulPoolError::ChannelClosed)?;
        }
        self.gather_run(request_id, &weights.slices, m, weights.n, start)
    }

    pub fn execute_prepared_f32(
        &mut self,
        a: Arc<[f16]>,
        m: usize,
        weights: &Fp16MatmulPoolPreparedWeights,
    ) -> Result<Fp32MatmulPoolOutput, MatmulPoolError> {
        if weights.exact_m != Some(m)
            || m == 0
            || m % 4 != 0
            || a.len() != m.saturating_mul(weights.k)
        {
            return Err(MatmulPoolError::InvalidInput(
                "prepared pool A requires exact M*K length and M%4==0",
            ));
        }
        let request_id = self.allocate_request_id();
        let start = Instant::now();
        for (worker, &(n0, nsub)) in weights.slices.iter().enumerate() {
            self.senders[worker]
                .send(PoolCommand::RunPreparedF32 {
                    request_id,
                    weight_id: weights.weight_id,
                    a: Arc::clone(&a),
                    m,
                    k: weights.k,
                    n0,
                    nsub,
                })
                .map_err(|_| MatmulPoolError::ChannelClosed)?;
        }
        self.gather_run_f32(request_id, &weights.slices, m, weights.n, start)
    }

    pub fn release_prepared(
        &mut self,
        weights: &Fp16MatmulPoolPreparedWeights,
    ) -> Result<(), MatmulPoolError> {
        self.release_weight_id(weights.weight_id, &weights.slices)
    }

    fn release_weight_id(
        &mut self,
        weight_id: u64,
        slices: &[(usize, usize)],
    ) -> Result<(), MatmulPoolError> {
        let request_id = self.allocate_request_id();
        for (worker, &(n0, nsub)) in slices.iter().enumerate() {
            self.senders[worker]
                .send(PoolCommand::Release {
                    request_id,
                    weight_id,
                    n0,
                    nsub,
                })
                .map_err(|_| MatmulPoolError::ChannelClosed)?;
        }
        for _ in 0..slices.len() {
            let result = self.recv_result(request_id, slices.len())?;
            if !matches!(result.response, PoolWorkerResponse::Released) {
                return Err(MatmulPoolError::Worker(
                    "unexpected response while releasing resident weights".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn gather_run(
        &self,
        request_id: u64,
        slices: &[(usize, usize)],
        m: usize,
        n: usize,
        start: Instant,
    ) -> Result<Fp16MatmulPoolOutput, MatmulPoolError> {
        let mut values = vec![f16::ZERO; m * n];
        let mut jobs = 0usize;
        let mut worker_timings = vec![ExecutionTiming::default(); slices.len()];
        let mut worker_scratch = vec![ScratchStats::default(); slices.len()];
        let mut first_error = None;
        for _ in 0..slices.len() {
            let result = self.recv_result(request_id, slices.len())?;
            let (output, scratch) = match result.response {
                PoolWorkerResponse::Ran { output, scratch } => match output {
                    Ok(output) => (output, scratch),
                    Err(error) => {
                        first_error.get_or_insert(MatmulPoolError::Worker(error));
                        continue;
                    }
                },
                _ => {
                    first_error.get_or_insert(MatmulPoolError::Worker(
                        "unexpected response while gathering run".to_string(),
                    ));
                    continue;
                }
            };
            if output.values.len() != m * result.nsub {
                first_error.get_or_insert(MatmulPoolError::Worker(format!(
                    "worker {} returned wrong output length",
                    result.worker
                )));
                continue;
            }
            for row in 0..m {
                let src = &output.values[row * result.nsub..(row + 1) * result.nsub];
                let dst0 = row * n + result.n0;
                values[dst0..dst0 + result.nsub].copy_from_slice(src);
            }
            jobs += output.stats.jobs_submitted;
            worker_timings[result.worker] = output.stats.timing;
            worker_scratch[result.worker] = scratch;
        }
        // Drain every submitted worker response before returning an error so
        // the next request cannot consume leftovers from this request.
        if let Some(error) = first_error {
            return Err(error);
        }
        Ok(Fp16MatmulPoolOutput {
            values,
            stats: MulticoreExecutionStats {
                workers_used: slices.len(),
                jobs_submitted: jobs,
                wall_ns: start.elapsed().as_nanos(),
                worker_timings,
                worker_scratch,
            },
        })
    }

    fn gather_run_f32(
        &self,
        request_id: u64,
        slices: &[(usize, usize)],
        m: usize,
        n: usize,
        start: Instant,
    ) -> Result<Fp32MatmulPoolOutput, MatmulPoolError> {
        let mut values = vec![0.0f32; m * n];
        let mut jobs = 0usize;
        let mut worker_timings = vec![ExecutionTiming::default(); slices.len()];
        let mut worker_scratch = vec![ScratchStats::default(); slices.len()];
        let mut first_error = None;
        for _ in 0..slices.len() {
            let result = self.recv_result(request_id, slices.len())?;
            let (output, scratch) = match result.response {
                PoolWorkerResponse::RanF32 { output, scratch } => match output {
                    Ok(output) => (output, scratch),
                    Err(error) => {
                        first_error.get_or_insert(MatmulPoolError::Worker(error));
                        continue;
                    }
                },
                _ => {
                    first_error.get_or_insert(MatmulPoolError::Worker(
                        "unexpected response while gathering FP32 run".to_string(),
                    ));
                    continue;
                }
            };
            if output.values.len() != m * result.nsub {
                first_error.get_or_insert(MatmulPoolError::Worker(format!(
                    "worker {} returned wrong FP32 output length",
                    result.worker
                )));
                continue;
            }
            for row in 0..m {
                let src = &output.values[row * result.nsub..(row + 1) * result.nsub];
                let dst0 = row * n + result.n0;
                values[dst0..dst0 + result.nsub].copy_from_slice(src);
            }
            jobs += output.stats.jobs_submitted;
            worker_timings[result.worker] = output.stats.timing;
            worker_scratch[result.worker] = scratch;
        }
        // Drain every submitted worker response before returning an error so
        // the next request cannot consume leftovers from this request.
        if let Some(error) = first_error {
            return Err(error);
        }
        Ok(Fp32MatmulPoolOutput {
            values,
            stats: MulticoreExecutionStats {
                workers_used: slices.len(),
                jobs_submitted: jobs,
                wall_ns: start.elapsed().as_nanos(),
                worker_timings,
                worker_scratch,
            },
        })
    }
}

impl Drop for Fp16MatmulPool {
    fn drop(&mut self) {
        for tx in &self.senders {
            let _ = tx.send(PoolCommand::Stop);
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

fn split_n_slices(n: usize, max_workers: usize) -> Vec<(usize, usize)> {
    let active = max_workers.min(n / 16).max(1);
    let step = n.div_ceil(active).div_ceil(16) * 16;
    let mut out = Vec::with_capacity(active);
    for worker in 0..active {
        let n0 = worker * step;
        if n0 >= n {
            break;
        }
        let n1 = (n0 + step).min(n);
        out.push((n0, n1 - n0));
    }
    out
}

pub fn cpu_reference_fp16(
    a: &[f16],
    b: &[f16],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f16>, MatmulError> {
    validate_inputs(a, b, m, k, n)?;
    let mut out = vec![f16::ZERO; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[row * k + kk].to_f32() * b[col * k + kk].to_f32();
            }
            out[row * n + col] = f16::from_f32(acc);
        }
    }
    Ok(out)
}

/// Exact fp16-input dot product accumulated in f64 and returned as f32.
/// This is the accuracy reference for `execute_f32`.
pub fn cpu_reference_fp32(
    a: &[f16],
    b: &[f16],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, MatmulError> {
    validate_inputs(a, b, m, k, n)?;
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += (a[row * k + kk].to_f32() as f64) * (b[col * k + kk].to_f32() as f64);
            }
            out[row * n + col] = acc as f32;
        }
    }
    Ok(out)
}

/// CPU oracle matching the executor's current precision contract exactly.
/// NPU-EW groups narrow every K partial and every running sum to fp16; tiny-M
/// fallback groups accumulate the NPU-equivalent fp16 partials in host f32 and
/// narrow only once at the end.
pub fn cpu_reference_executor_semantics(
    a: &[f16],
    b: &[f16],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f16>, MatmulError> {
    validate_inputs(a, b, m, k, n)?;
    let plan = plan_fp16_matmul(m, k, n)?;
    let kt = plan.k_tiles();
    let mut out = vec![f16::ZERO; m * n];
    for group in plan.tiles.chunks(kt) {
        validate_group(group, k)?;
        let first = group[0];
        match accumulation_mode(group) {
            KAccumulation::None => {
                cpu_partial_store(a, b, k, n, first, &mut out);
            }
            KAccumulation::NpuFp16PingPong => {
                for (gi, &tile) in group.iter().enumerate() {
                    for tm in 0..tile.m {
                        for tn in 0..tile.n {
                            let part = cpu_partial(a, b, k, tile, tm, tn);
                            let idx = (tile.m0 + tm) * n + tile.n0 + tn;
                            out[idx] = if gi == 0 {
                                part
                            } else {
                                f16::from_f32(out[idx].to_f32() + part.to_f32())
                            };
                        }
                    }
                }
            }
            KAccumulation::HostFp32TinyM => {
                let mut acc = vec![0.0f32; first.m * first.n];
                for &tile in group {
                    for tm in 0..tile.m {
                        for tn in 0..tile.n {
                            acc[tm * tile.n + tn] += cpu_partial(a, b, k, tile, tm, tn).to_f32();
                        }
                    }
                }
                for tm in 0..first.m {
                    for tn in 0..first.n {
                        out[(first.m0 + tm) * n + first.n0 + tn] =
                            f16::from_f32(acc[tm * first.n + tn]);
                    }
                }
            }
        }
    }
    Ok(out)
}

fn validate_inputs(a: &[f16], b: &[f16], m: usize, k: usize, n: usize) -> Result<(), MatmulError> {
    if a.len()
        != m.checked_mul(k)
            .ok_or(MatmulError::InvalidInput("A size overflow"))?
    {
        return Err(MatmulError::InvalidInput(
            "A must contain exactly M*K elements",
        ));
    }
    if b.len()
        != n.checked_mul(k)
            .ok_or(MatmulError::InvalidInput("B size overflow"))?
    {
        return Err(MatmulError::InvalidInput(
            "B must contain exactly N*K elements",
        ));
    }
    Ok(())
}

fn validate_group(group: &[Fp16MatmulTile], k_total: usize) -> Result<(), MatmulError> {
    let Some(first) = group.first() else {
        return Err(MatmulError::Internal("empty output-tile group"));
    };
    let mut next_k = 0usize;
    for tile in group {
        if tile.m0 != first.m0 || tile.n0 != first.n0 || tile.m != first.m || tile.n != first.n {
            return Err(MatmulError::Internal(
                "K tiles do not share output geometry",
            ));
        }
        if tile.k0 != next_k {
            return Err(MatmulError::Internal("K tiles are not contiguous"));
        }
        next_k = next_k
            .checked_add(tile.k)
            .ok_or(MatmulError::Internal("K tile sum overflow"))?;
    }
    if next_k != k_total {
        return Err(MatmulError::Internal(
            "K tiles do not cover the full contraction dimension",
        ));
    }
    Ok(())
}

fn accumulation_mode(group: &[Fp16MatmulTile]) -> KAccumulation {
    if group.len() == 1 {
        KAccumulation::None
    } else if group[0].m >= 12 {
        KAccumulation::NpuFp16PingPong
    } else {
        KAccumulation::HostFp32TinyM
    }
}

#[inline]
fn copy_f16_block(dst: &mut [u8], dst_elem: usize, src: &[f16]) {
    let byte0 = dst_elem * 2;
    let byte_len = src.len() * 2;
    #[cfg(target_endian = "little")]
    {
        let bytes: &[u8] = bytemuck::cast_slice(src);
        dst[byte0..byte0 + byte_len].copy_from_slice(bytes);
    }
    #[cfg(target_endian = "big")]
    {
        for (i, &v) in src.iter().enumerate() {
            let bits = v.to_bits().to_le_bytes();
            let p = byte0 + i * 2;
            dst[p..p + 2].copy_from_slice(&bits);
        }
    }
}

fn pack_input_bytes(dst: &mut [u8], a: &[f16], k_total: usize, tile: Fp16MatmulTile) {
    dst.fill(0);
    // Native feature cube is [K/8][M][8]. K is at least 32-aligned for all
    // planner tiles, so every C2=8 block is complete and contiguous in dst.
    for kb in 0..tile.k / 8 {
        let tile_k0 = tile.k0 + kb * 8;
        let dst_plane = kb * tile.m * 8;
        for tm in 0..tile.m {
            let src_base = (tile.m0 + tm) * k_total + tile_k0;
            let dst_base = dst_plane + tm * 8;
            copy_f16_block(dst, dst_base, &a[src_base..src_base + 8]);
        }
    }
}

fn pack_weight_bytes(dst: &mut [u8], b: &[f16], k_total: usize, tile: Fp16MatmulTile) {
    dst.fill(0);
    // Native weight layout is [N/16][K/32][16][32] in the generator's
    // addressing convention. For each N-lane, the 32 K values are contiguous.
    for ng in 0..tile.n / 16 {
        let tile_n0 = tile.n0 + ng * 16;
        let n_group_base = ng * 16 * tile.k;
        for kg in 0..tile.k / 32 {
            let tile_k0 = tile.k0 + kg * 32;
            let k_group_base = kg * 32 * 16;
            for nl in 0..16 {
                let src_base = (tile_n0 + nl) * k_total + tile_k0;
                let dst_base = n_group_base + k_group_base + nl * 32;
                copy_f16_block(dst, dst_base, &b[src_base..src_base + 32]);
            }
        }
    }
}

fn pack_input(
    bo: &mut RocketBuffer<'_>,
    a: &[f16],
    k_total: usize,
    tile: Fp16MatmulTile,
) -> Result<(), MatmulError> {
    bo.prep_relative(0)?;
    pack_input_bytes(bo.as_mut_slice(), a, k_total, tile);
    bo.fini()?;
    Ok(())
}

fn pack_output_tile(
    bo: &mut RocketBuffer<'_>,
    values: &[f16],
    n_total: usize,
    tile: Fp16MatmulTile,
) -> Result<(), MatmulError> {
    bo.prep_relative(0)?;
    let dst = bo.as_mut_slice();
    dst.fill(0);
    for ng in 0..tile.n / 8 {
        let tile_n0 = tile.n0 + ng * 8;
        let dst_plane = ng * tile.m * 8;
        for tm in 0..tile.m {
            let src_base = (tile.m0 + tm) * n_total + tile_n0;
            let dst_base = dst_plane + tm * 8;
            copy_f16_block(dst, dst_base, &values[src_base..src_base + 8]);
        }
    }
    bo.fini()?;
    Ok(())
}

fn pack_weights(
    bo: &mut RocketBuffer<'_>,
    b: &[f16],
    k_total: usize,
    tile: Fp16MatmulTile,
) -> Result<(), MatmulError> {
    bo.prep_relative(0)?;
    pack_weight_bytes(bo.as_mut_slice(), b, k_total, tile);
    bo.fini()?;
    Ok(())
}

fn prepare_output(bo: &mut RocketBuffer<'_>) -> Result<(), MatmulError> {
    bo.prep_relative(0)?;
    bo.as_mut_slice().fill(0);
    bo.fini()?;
    Ok(())
}

fn write_regcmd(bo: &mut RocketBuffer<'_>, ops: &[u64]) -> Result<(), MatmulError> {
    if ops.len() * 8 > bo.len() {
        return Err(MatmulError::Internal("regcmd BO too small"));
    }
    bo.prep_relative(0)?;
    bo.as_mut_slice().fill(0);
    for (chunk, word) in bo.as_mut_slice().chunks_exact_mut(8).zip(ops.iter()) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    bo.fini()?;
    Ok(())
}

fn submit_plain(
    device: &RocketDevice,
    regcmd: &RocketBuffer<'_>,
    input: &RocketBuffer<'_>,
    weights: &RocketBuffer<'_>,
    output: &RocketBuffer<'_>,
    count: usize,
) -> Result<(), MatmulError> {
    let task = Task {
        regcmd: u32::try_from(regcmd.dma_address())
            .map_err(|_| MatmulError::AddressAbove32Bit(regcmd.dma_address()))?,
        regcmd_count: count as u32,
    };
    device.submit(
        &[task],
        &[input.handle(), weights.handle(), regcmd.handle()],
        &[output.handle()],
    )?;
    Ok(())
}

fn submit_accumulate(
    device: &RocketDevice,
    regcmd: &RocketBuffer<'_>,
    input: &RocketBuffer<'_>,
    weights: &RocketBuffer<'_>,
    add: &RocketBuffer<'_>,
    output: &RocketBuffer<'_>,
    count: usize,
) -> Result<(), MatmulError> {
    let task = Task {
        regcmd: u32::try_from(regcmd.dma_address())
            .map_err(|_| MatmulError::AddressAbove32Bit(regcmd.dma_address()))?,
        regcmd_count: count as u32,
    };
    device.submit(
        &[task],
        &[
            input.handle(),
            weights.handle(),
            regcmd.handle(),
            add.handle(),
        ],
        &[output.handle()],
    )?;
    Ok(())
}

fn gather_tile(src: &[u8], tile: Fp16MatmulTile, n_total: usize, dst: &mut [f16]) {
    for tm in 0..tile.m {
        for tn in 0..tile.n {
            let native = feature_data(tile.n, tile.m, 1, 8, tn + 1, tm + 1, 1);
            dst[(tile.m0 + tm) * n_total + tile.n0 + tn] = get_f16(src, native);
        }
    }
}

fn cpu_partial(
    a: &[f16],
    b: &[f16],
    k_total: usize,
    tile: Fp16MatmulTile,
    tm: usize,
    tn: usize,
) -> f16 {
    let mut acc = 0.0f32;
    for tk in 0..tile.k {
        acc += a[(tile.m0 + tm) * k_total + tile.k0 + tk].to_f32()
            * b[(tile.n0 + tn) * k_total + tile.k0 + tk].to_f32();
    }
    f16::from_f32(acc)
}

fn cpu_partial_store(
    a: &[f16],
    b: &[f16],
    k_total: usize,
    n_total: usize,
    tile: Fp16MatmulTile,
    dst: &mut [f16],
) {
    for tm in 0..tile.m {
        for tn in 0..tile.n {
            dst[(tile.m0 + tm) * n_total + tile.n0 + tn] = cpu_partial(a, b, k_total, tile, tm, tn);
        }
    }
}

fn get_f16(src: &[u8], index: usize) -> f16 {
    let p = index * 2;
    f16::from_bits(u16::from_le_bytes([src[p], src[p + 1]]))
}

fn get_f32(src: &[u8], index: usize) -> f32 {
    let p = index * 4;
    f32::from_le_bytes([src[p], src[p + 1], src[p + 2], src[p + 3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(m: usize, k: usize, n: usize) -> (Vec<f16>, Vec<f16>) {
        let a = (0..m * k)
            .map(|i| f16::from_f32(((i * 7 + 3) % 5) as f32 - 2.0))
            .collect();
        let b = (0..n * k)
            .map(|i| f16::from_f32(((i * 3 + 1) % 5) as f32 - 2.0))
            .collect();
        (a, b)
    }

    #[test]
    fn cpu_executor_oracle_matches_math_when_no_k_split() {
        let (a, b) = data(64, 256, 64);
        assert_eq!(
            cpu_reference_executor_semantics(&a, &b, 64, 256, 64).unwrap(),
            cpu_reference_fp16(&a, &b, 64, 256, 64).unwrap()
        );
    }

    #[test]
    fn cpu_executor_oracle_handles_m_n_and_k_tiles() {
        let (a, b) = data(300, 512, 272);
        let got = cpu_reference_executor_semantics(&a, &b, 300, 512, 272).unwrap();
        assert_eq!(got.len(), 300 * 272);
        let p = plan_fp16_matmul(300, 512, 272).unwrap();
        assert!(p.m_tiles() > 1 && p.n_tiles() > 1 && p.k_tiles() > 1);
    }

    #[test]
    fn tiny_m_k_split_uses_host_accumulation_contract() {
        let p = plan_fp16_matmul(4, 4096, 64).unwrap();
        assert!(p.k_tiles() > 1);
        assert_eq!(
            accumulation_mode(&p.tiles[..p.k_tiles()]),
            KAccumulation::HostFp32TinyM
        );
    }

    #[test]
    fn block_input_packing_matches_feature_data_reference() {
        let m = 44usize;
        let k_total = 640usize;
        let tile = Fp16MatmulTile {
            m0: 8,
            n0: 0,
            k0: 128,
            m,
            n: 16,
            k: 384,
        };
        let a: Vec<f16> = (0..64 * k_total)
            .map(|i| f16::from_bits((i as u16).wrapping_mul(73).wrapping_add(11)))
            .collect();
        let mut got = vec![0u8; tile.m * tile.k * 2];
        pack_input_bytes(&mut got, &a, k_total, tile);
        for tm in 0..tile.m {
            for tk in 0..tile.k {
                let native = feature_data(tile.k, tile.m, 1, 8, tk + 1, tm + 1, 1);
                assert_eq!(
                    get_f16(&got, native).to_bits(),
                    a[(tile.m0 + tm) * k_total + tile.k0 + tk].to_bits()
                );
            }
        }
    }

    #[test]
    fn block_weight_packing_matches_weight_fp16_reference() {
        let n = 272usize;
        let k_total = 640usize;
        let tile = Fp16MatmulTile {
            m0: 0,
            n0: 16,
            k0: 128,
            m: 12,
            n: 256,
            k: 384,
        };
        let b: Vec<f16> = (0..(n + 16) * k_total)
            .map(|i| f16::from_bits((i as u16).wrapping_mul(29).wrapping_add(7)))
            .collect();
        let mut got = vec![0u8; tile.n * tile.k * 2];
        pack_weight_bytes(&mut got, &b, k_total, tile);
        for tn in 0..tile.n {
            for tk in 0..tile.k {
                let native = weight_fp16(tile.k, tn + 1, tk + 1);
                assert_eq!(
                    get_f16(&got, native).to_bits(),
                    b[(tile.n0 + tn) * k_total + tile.k0 + tk].to_bits()
                );
            }
        }
    }

    #[test]
    fn multicore_n_split_is_aligned_and_complete() {
        assert_eq!(
            split_n_slices(768, 3),
            vec![(0, 256), (256, 256), (512, 256)]
        );
        assert_eq!(
            split_n_slices(512, 3),
            vec![(0, 176), (176, 176), (352, 160)]
        );
        assert_eq!(split_n_slices(32, 3), vec![(0, 16), (16, 16)]);
        for (n, workers) in [(16, 3), (272, 3), (512, 2), (816, 3)] {
            let slices = split_n_slices(n, workers);
            assert_eq!(slices.iter().map(|x| x.1).sum::<usize>(), n);
            assert!(slices.iter().all(|(n0, ns)| n0 % 16 == 0 && ns % 16 == 0));
        }
    }

    #[test]
    fn input_lengths_are_checked() {
        let b = vec![f16::ZERO; 16 * 32];
        assert!(matches!(
            cpu_reference_fp16(&[], &b, 4, 32, 16),
            Err(MatmulError::InvalidInput(_))
        ));
    }

    #[test]
    fn fp32_cpu_reference_uses_f64_accumulation_contract() {
        let (a, b) = data(12, 96, 32);
        let out = cpu_reference_fp32(&a, &b, 12, 96, 32).unwrap();
        assert_eq!(out.len(), 12 * 32);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
