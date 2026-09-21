// SPDX-License-Identifier: MIT

use rocket_runtime::{RocketBuffer, RocketDevice, RocketOwnedBuffer, Task};
use rocknpu_regcmd::{
    INT8_REGCMD_COUNT, Int8DecodeDesc, Int8EncodeError, encode_int8_decode_m1, encode_int8_mtile,
};
use std::fmt;
use std::io;
use std::os::fd::RawFd;
use std::time::Instant;

const WAIT_NS: i64 = 2_000_000_000;
const REGCMD_BYTES: usize = 4096;
const WIDE_K_SLICE: usize = 1024;
const SINGLE_SUBMIT_K_MAX: usize = 4096;
const N_MAX: usize = 8192;

#[derive(Debug)]
pub enum Int8DecodeError {
    InvalidInput(&'static str),
    Encode(Int8EncodeError),
    Io(io::Error),
    AddressAbove32Bit(u64),
    SizeOverflow,
}

impl fmt::Display for Int8DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid int8 M=1 MatMul input: {msg}"),
            Self::Encode(err) => err.fmt(f),
            Self::Io(err) => err.fmt(f),
            Self::AddressAbove32Bit(addr) => write!(
                f,
                "Rocket BO IOVA 0x{addr:x} exceeds the 32-bit RK3588 regcmd field"
            ),
            Self::SizeOverflow => write!(f, "int8 M=1 MatMul size arithmetic overflow"),
        }
    }
}
impl std::error::Error for Int8DecodeError {}
impl From<Int8EncodeError> for Int8DecodeError {
    fn from(value: Int8EncodeError) -> Self {
        Self::Encode(value)
    }
}
impl From<io::Error> for Int8DecodeError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Int8PreparedWeightStats {
    pub k_slices: usize,
    pub resident_bytes: usize,
    pub pack_ns: u128,
}

pub struct Int8PreparedWeights {
    device_fd: RawFd,
    k: usize,
    n: usize,
    slices: usize,
    bo: RocketOwnedBuffer,
    offsets: Vec<usize>,
    stats: Int8PreparedWeightStats,
}

impl Int8PreparedWeights {
    pub fn k(&self) -> usize {
        self.k
    }
    pub fn n(&self) -> usize {
        self.n
    }
    pub fn stats(&self) -> Int8PreparedWeightStats {
        self.stats
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Int8DecodeStats {
    pub k_slices: usize,
    pub npu_tasks: usize,
    /// Per-call input/regcmd/output staging. One-shot `execute` also includes
    /// the one-time resident weight preparation so old benchmark semantics stay
    /// comparable; `execute_prepared` excludes weight preparation.
    pub pack_ns: u128,
    pub alloc_ns: u128,
    pub input_stage_ns: u128,
    pub partial_stage_ns: u128,
    pub regcmd_stage_ns: u128,
    pub output_fini_ns: u128,
    pub submit_ns: u128,
    pub wait_ns: u128,
    pub submit_wait_ns: u128,
    pub host_accum_ns: u128,
    pub total_ns: u128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Int8DecodeOutput {
    pub values: Vec<i32>,
    pub stats: Int8DecodeStats,
}

pub struct Int8PendingDecode<'a> {
    _regcmd: RocketBuffer<'a>,
    _input: RocketBuffer<'a>,
    partials: RocketBuffer<'a>,
    slices: usize,
    n: usize,
    npu_tasks: usize,
    pack_ns: u128,
    alloc_ns: u128,
    input_stage_ns: u128,
    partial_stage_ns: u128,
    regcmd_stage_ns: u128,
    submit_ns: u128,
    submit_wait_start: Instant,
    total_start: Instant,
}

pub(crate) struct Int8OwnedScratch {
    regcmd: RocketOwnedBuffer,
    input: RocketOwnedBuffer,
    partials: RocketOwnedBuffer,
}

pub struct Int8MtileScratch {
    regcmd: RocketOwnedBuffer,
    input: RocketOwnedBuffer,
    output: RocketOwnedBuffer,
}

pub(crate) struct Int8OwnedPending {
    slices: usize,
    n: usize,
    npu_tasks: usize,
    pack_ns: u128,
    alloc_ns: u128,
    input_stage_ns: u128,
    partial_stage_ns: u128,
    regcmd_stage_ns: u128,
    submit_ns: u128,
    submit_wait_start: Instant,
    total_start: Instant,
}

/// W8A8 M=1 executor for RK3588 decode projections.
///
/// Input `a_k` is row-major A[1,K]. Public weights are B[N,K]. Static weights
/// can be transformed once with `prepare_weights` into a Rocket-resident native
/// 32x32 layout and reused by `execute_prepared` for every decode token.
///
/// K<=4096 uses one full-K submit. Wider K is split into 1024-wide slices
/// (with a possible 512-wide tail), each producing int32 partials which are
/// accumulated exactly on the host.
pub struct Int8DecodeExecutor<'a> {
    device: &'a RocketDevice,
    _guard: Option<RocketBuffer<'a>>,
}

impl<'a> Int8DecodeExecutor<'a> {
    pub fn new(device: &'a RocketDevice) -> Result<Self, Int8DecodeError> {
        let guard = device.alloc_buffer(4096)?;
        check_dma32_range(guard.dma_address(), guard.len())?;
        Ok(Self {
            device,
            _guard: Some(guard),
        })
    }

    pub fn from_externally_guarded_device(device: &'a RocketDevice) -> Self {
        Self {
            device,
            _guard: None,
        }
    }
    /// Execute a hardware-validated research W8A8 M-tile against one full-K resident
    /// weight slice while reusing context-owned Rocket BOs across calls.
    pub fn execute_prepared_mtile_persistent(
        &self,
        m: usize,
        a_mk: &[i8],
        weights: &Int8PreparedWeights,
        scratch_slot: &mut Option<Int8MtileScratch>,
    ) -> Result<Int8DecodeOutput, Int8DecodeError> {
        if !matches!(m, 16 | 32 | 48 | 64) {
            return Err(Int8DecodeError::InvalidInput(
                "persistent M-tile path requires M in {16,32,48,64}",
            ));
        }
        let total_start = Instant::now();
        if self.device.fd() != weights.device_fd {
            return Err(Int8DecodeError::InvalidInput(
                "prepared weights belong to a different Rocket device",
            ));
        }
        if weights.slices != 1 || weights.k > SINGLE_SUBMIT_K_MAX {
            return Err(Int8DecodeError::InvalidInput(
                "persistent M-tile path requires one full-K prepared weight slice",
            ));
        }
        let expected_a = m
            .checked_mul(weights.k)
            .ok_or(Int8DecodeError::SizeOverflow)?;
        if a_mk.len() != expected_a {
            return Err(Int8DecodeError::InvalidInput("A length must equal M*K"));
        }
        let output_values = m
            .checked_mul(weights.n)
            .ok_or(Int8DecodeError::SizeOverflow)?;
        let output_bytes = output_values
            .checked_mul(4)
            .ok_or(Int8DecodeError::SizeOverflow)?;

        let alloc_start = Instant::now();
        let needs_grow = scratch_slot.as_ref().is_none_or(|scratch| {
            scratch.regcmd.len() < REGCMD_BYTES
                || scratch.input.len() < expected_a
                || scratch.output.len() < output_bytes
        });
        if needs_grow {
            let old = scratch_slot.as_ref();
            let regcmd_capacity = old.map_or(REGCMD_BYTES, |s| s.regcmd.len().max(REGCMD_BYTES));
            let input_capacity = old.map_or(expected_a, |s| s.input.len().max(expected_a));
            let output_capacity = old.map_or(output_bytes, |s| s.output.len().max(output_bytes));
            let scratch = Int8MtileScratch {
                regcmd: self.device.alloc_owned_buffer(regcmd_capacity)?,
                input: self.device.alloc_owned_buffer(input_capacity)?,
                output: self.device.alloc_owned_buffer(output_capacity)?,
            };
            for (addr, len) in [
                (scratch.regcmd.dma_address(), scratch.regcmd.len()),
                (scratch.input.dma_address(), scratch.input.len()),
                (scratch.output.dma_address(), scratch.output.len()),
            ] {
                check_dma32_range(addr, len)?;
            }
            *scratch_slot = Some(scratch);
        }
        let alloc_ns = alloc_start.elapsed().as_nanos();
        let scratch = scratch_slot.as_mut().ok_or(Int8DecodeError::InvalidInput(
            "persistent M-tile scratch missing",
        ))?;

        let input_stage_start = Instant::now();
        scratch.input.prep_relative(0)?;
        for (dst, src) in scratch.input.as_mut_slice()[..expected_a]
            .iter_mut()
            .zip(a_mk.iter().copied())
        {
            *dst = src as u8;
        }
        scratch.input.fini()?;
        let input_stage_ns = input_stage_start.elapsed().as_nanos();

        let regcmd_stage_start = Instant::now();
        let ops = encode_int8_mtile(
            m,
            Int8DecodeDesc::new(
                weights.k,
                weights.n,
                scratch.input.dma_address(),
                weights.bo.dma_address(),
                scratch.output.dma_address(),
            ),
        )?;
        scratch.regcmd.prep_relative(0)?;
        scratch.regcmd.as_mut_slice()[..REGCMD_BYTES].fill(0);
        write_regcmd_bytes(scratch.regcmd.as_mut_slice(), 0, &ops)?;
        scratch.regcmd.fini()?;
        let regcmd_stage_ns = regcmd_stage_start.elapsed().as_nanos();

        let reg_addr = scratch.regcmd.dma_address();
        let task = Task {
            regcmd: u32::try_from(reg_addr)
                .map_err(|_| Int8DecodeError::AddressAbove32Bit(reg_addr))?,
            regcmd_count: u32::try_from(ops.len()).map_err(|_| Int8DecodeError::SizeOverflow)?,
        };
        let submit_wait_start = Instant::now();
        let submit_start = Instant::now();
        self.device.submit(
            &[task],
            &[
                scratch.input.handle(),
                weights.bo.handle(),
                scratch.regcmd.handle(),
            ],
            &[scratch.output.handle()],
        )?;
        let submit_ns = submit_start.elapsed().as_nanos();
        let wait_start = Instant::now();
        scratch.output.prep_relative(WAIT_NS)?;
        let wait_ns = wait_start.elapsed().as_nanos();
        let submit_wait_ns = submit_wait_start.elapsed().as_nanos();

        let mut values = Vec::with_capacity(output_values);
        for i in 0..output_values {
            values.push(read_i32(scratch.output.as_slice(), i));
        }
        let output_fini_start = Instant::now();
        scratch.output.fini()?;
        let output_fini_ns = output_fini_start.elapsed().as_nanos();

        Ok(Int8DecodeOutput {
            values,
            stats: Int8DecodeStats {
                k_slices: 1,
                npu_tasks: 1,
                pack_ns: input_stage_ns + regcmd_stage_ns,
                alloc_ns,
                input_stage_ns,
                partial_stage_ns: 0,
                regcmd_stage_ns,
                output_fini_ns,
                submit_ns,
                wait_ns,
                submit_wait_ns,
                host_accum_ns: 0,
                total_ns: total_start.elapsed().as_nanos(),
            },
        })
    }

    pub fn execute_prepared_m16_persistent(
        &self,
        a_mk: &[i8],
        weights: &Int8PreparedWeights,
        scratch_slot: &mut Option<Int8MtileScratch>,
    ) -> Result<Int8DecodeOutput, Int8DecodeError> {
        self.execute_prepared_mtile_persistent(16, a_mk, weights, scratch_slot)
    }

    /// Original single-slice M=16 fast path. Keep this separate from the later K-split
    /// experiment so K<=4096 model routing does not pay partial-buffer/host-accumulation overhead.
    pub fn execute_prepared_m16_single(
        &self,
        a_mk: &[i8],
        weights: &Int8PreparedWeights,
    ) -> Result<Int8DecodeOutput, Int8DecodeError> {
        const M: usize = 16;
        let total_start = Instant::now();
        if self.device.fd() != weights.device_fd {
            return Err(Int8DecodeError::InvalidInput(
                "prepared weights belong to a different Rocket device",
            ));
        }
        if weights.slices != 1 || weights.k > SINGLE_SUBMIT_K_MAX {
            return Err(Int8DecodeError::InvalidInput(
                "single-slice M=16 path requires one full-K prepared weight slice",
            ));
        }
        let expected_a = M
            .checked_mul(weights.k)
            .ok_or(Int8DecodeError::SizeOverflow)?;
        if a_mk.len() != expected_a {
            return Err(Int8DecodeError::InvalidInput("A length must equal 16*K"));
        }
        let output_values = M
            .checked_mul(weights.n)
            .ok_or(Int8DecodeError::SizeOverflow)?;
        let output_bytes = output_values
            .checked_mul(4)
            .ok_or(Int8DecodeError::SizeOverflow)?;

        let alloc_start = Instant::now();
        let mut regcmd = self.device.alloc_buffer(REGCMD_BYTES)?;
        let mut input = self.device.alloc_buffer(expected_a)?;
        let output = self.device.alloc_buffer(output_bytes)?;
        for bo in [&regcmd, &input, &output] {
            check_dma32_range(bo.dma_address(), bo.len())?;
        }
        let alloc_ns = alloc_start.elapsed().as_nanos();

        let input_stage_start = Instant::now();
        input.prep_relative(0)?;
        for (dst, src) in input.as_mut_slice().iter_mut().zip(a_mk.iter().copied()) {
            *dst = src as u8;
        }
        input.fini()?;
        let input_stage_ns = input_stage_start.elapsed().as_nanos();

        let regcmd_stage_start = Instant::now();
        let ops = encode_int8_mtile(
            M,
            Int8DecodeDesc::new(
                weights.k,
                weights.n,
                input.dma_address(),
                weights.bo.dma_address(),
                output.dma_address(),
            ),
        )?;
        regcmd.prep_relative(0)?;
        regcmd.as_mut_slice().fill(0);
        write_regcmd_at(&mut regcmd, 0, &ops)?;
        regcmd.fini()?;
        let regcmd_stage_ns = regcmd_stage_start.elapsed().as_nanos();

        let reg_addr = regcmd.dma_address();
        let task = Task {
            regcmd: u32::try_from(reg_addr)
                .map_err(|_| Int8DecodeError::AddressAbove32Bit(reg_addr))?,
            regcmd_count: u32::try_from(ops.len()).map_err(|_| Int8DecodeError::SizeOverflow)?,
        };
        let submit_wait_start = Instant::now();
        let submit_start = Instant::now();
        self.device.submit(
            &[task],
            &[input.handle(), weights.bo.handle(), regcmd.handle()],
            &[output.handle()],
        )?;
        let submit_ns = submit_start.elapsed().as_nanos();
        let wait_start = Instant::now();
        output.prep_relative(WAIT_NS)?;
        let wait_ns = wait_start.elapsed().as_nanos();
        let submit_wait_ns = submit_wait_start.elapsed().as_nanos();

        let mut values = Vec::with_capacity(output_values);
        for i in 0..output_values {
            values.push(read_i32(output.as_slice(), i));
        }
        let output_fini_start = Instant::now();
        output.fini()?;
        let output_fini_ns = output_fini_start.elapsed().as_nanos();

        Ok(Int8DecodeOutput {
            values,
            stats: Int8DecodeStats {
                k_slices: 1,
                npu_tasks: 1,
                pack_ns: input_stage_ns + regcmd_stage_ns,
                alloc_ns,
                input_stage_ns,
                partial_stage_ns: 0,
                regcmd_stage_ns,
                output_fini_ns,
                submit_ns,
                wait_ns,
                submit_wait_ns,
                host_accum_ns: 0,
                total_ns: total_start.elapsed().as_nanos(),
            },
        })
    }

    /// Execute the hardware-proven research M=16 W8A8 tile against resident weights.
    /// K<=4096 is one full-K task. Wider K reuses the production 1024/512 K-slice
    /// packing, submits one M=16 task per slice, and accumulates int32 partials exactly.
    pub fn execute_prepared_m16(
        &self,
        a_mk: &[i8],
        weights: &Int8PreparedWeights,
    ) -> Result<Int8DecodeOutput, Int8DecodeError> {
        const M: usize = 16;
        let total_start = Instant::now();
        if self.device.fd() != weights.device_fd {
            return Err(Int8DecodeError::InvalidInput(
                "prepared weights belong to a different Rocket device",
            ));
        }
        if weights.offsets.len() != weights.slices || weights.slices == 0 {
            return Err(Int8DecodeError::InvalidInput(
                "prepared weight slice metadata mismatch",
            ));
        }
        let expected_a = M
            .checked_mul(weights.k)
            .ok_or(Int8DecodeError::SizeOverflow)?;
        if a_mk.len() != expected_a {
            return Err(Int8DecodeError::InvalidInput("A length must equal 16*K"));
        }

        let slices = weights.slices;
        let output_values = M
            .checked_mul(weights.n)
            .ok_or(Int8DecodeError::SizeOverflow)?;
        let partial_values = slices
            .checked_mul(output_values)
            .ok_or(Int8DecodeError::SizeOverflow)?;
        let partial_bytes = partial_values
            .checked_mul(4)
            .ok_or(Int8DecodeError::SizeOverflow)?;
        let regcmd_bytes = slices
            .checked_mul(REGCMD_BYTES)
            .ok_or(Int8DecodeError::SizeOverflow)?;

        let alloc_start = Instant::now();
        let mut regcmd = self.device.alloc_buffer(regcmd_bytes)?;
        let mut input = self.device.alloc_buffer(expected_a)?;
        let output = self.device.alloc_buffer(partial_bytes)?;
        for bo in [&regcmd, &input, &output] {
            check_dma32_range(bo.dma_address(), bo.len())?;
        }
        let alloc_ns = alloc_start.elapsed().as_nanos();

        let input_stage_start = Instant::now();
        input.prep_relative(0)?;
        let mut input_offsets = Vec::with_capacity(slices);
        let mut input_offset = 0usize;
        for slice in 0..slices {
            input_offsets.push(input_offset);
            let k0 = slice_k0(slices, slice);
            let kp = slice_kp(weights.k, slices, slice);
            for row in 0..M {
                let src_start = row
                    .checked_mul(weights.k)
                    .and_then(|v| v.checked_add(k0))
                    .ok_or(Int8DecodeError::SizeOverflow)?;
                let src_end = src_start
                    .checked_add(kp)
                    .ok_or(Int8DecodeError::SizeOverflow)?;
                let dst_start = input_offset
                    .checked_add(row.checked_mul(kp).ok_or(Int8DecodeError::SizeOverflow)?)
                    .ok_or(Int8DecodeError::SizeOverflow)?;
                let dst_end = dst_start
                    .checked_add(kp)
                    .ok_or(Int8DecodeError::SizeOverflow)?;
                for (dst, &src) in input.as_mut_slice()[dst_start..dst_end]
                    .iter_mut()
                    .zip(&a_mk[src_start..src_end])
                {
                    *dst = src as u8;
                }
            }
            input_offset = input_offset
                .checked_add(M.checked_mul(kp).ok_or(Int8DecodeError::SizeOverflow)?)
                .ok_or(Int8DecodeError::SizeOverflow)?;
        }
        input.fini()?;
        let input_stage_ns = input_stage_start.elapsed().as_nanos();

        let regcmd_stage_start = Instant::now();
        regcmd.prep_relative(0)?;
        regcmd.as_mut_slice().fill(0);
        let mut tasks = Vec::with_capacity(slices);
        for slice in 0..slices {
            let kp = slice_kp(weights.k, slices, slice);
            let weight_dma = weights
                .bo
                .dma_address()
                .checked_add(
                    u64::try_from(weights.offsets[slice])
                        .map_err(|_| Int8DecodeError::SizeOverflow)?,
                )
                .ok_or(Int8DecodeError::SizeOverflow)?;
            let input_dma = input
                .dma_address()
                .checked_add(
                    u64::try_from(input_offsets[slice])
                        .map_err(|_| Int8DecodeError::SizeOverflow)?,
                )
                .ok_or(Int8DecodeError::SizeOverflow)?;
            let partial_offset = slice
                .checked_mul(output_values)
                .and_then(|v| v.checked_mul(4))
                .ok_or(Int8DecodeError::SizeOverflow)?;
            let output_dma = output
                .dma_address()
                .checked_add(
                    u64::try_from(partial_offset).map_err(|_| Int8DecodeError::SizeOverflow)?,
                )
                .ok_or(Int8DecodeError::SizeOverflow)?;
            let ops = encode_int8_mtile(
                M,
                Int8DecodeDesc::new(kp, weights.n, input_dma, weight_dma, output_dma),
            )?;
            let reg_offset = slice
                .checked_mul(REGCMD_BYTES)
                .ok_or(Int8DecodeError::SizeOverflow)?;
            write_regcmd_at(&mut regcmd, reg_offset, &ops)?;
            let reg_addr = regcmd
                .dma_address()
                .checked_add(u64::try_from(reg_offset).map_err(|_| Int8DecodeError::SizeOverflow)?)
                .ok_or(Int8DecodeError::SizeOverflow)?;
            tasks.push(Task {
                regcmd: u32::try_from(reg_addr)
                    .map_err(|_| Int8DecodeError::AddressAbove32Bit(reg_addr))?,
                regcmd_count: u32::try_from(ops.len())
                    .map_err(|_| Int8DecodeError::SizeOverflow)?,
            });
        }
        regcmd.fini()?;
        let regcmd_stage_ns = regcmd_stage_start.elapsed().as_nanos();

        let submit_wait_start = Instant::now();
        let submit_start = Instant::now();
        self.device.submit(
            &tasks,
            &[input.handle(), weights.bo.handle(), regcmd.handle()],
            &[output.handle()],
        )?;
        let submit_ns = submit_start.elapsed().as_nanos();
        let wait_start = Instant::now();
        output.prep_relative(WAIT_NS)?;
        let wait_ns = wait_start.elapsed().as_nanos();
        let submit_wait_ns = submit_wait_start.elapsed().as_nanos();

        let accum_start = Instant::now();
        let mut values = vec![0i32; output_values];
        for slice in 0..slices {
            let base = slice
                .checked_mul(output_values)
                .ok_or(Int8DecodeError::SizeOverflow)?;
            for (index, sum) in values.iter_mut().enumerate() {
                *sum = sum
                    .checked_add(read_i32(output.as_slice(), base + index))
                    .ok_or(Int8DecodeError::InvalidInput("int32 accumulation overflow"))?;
            }
        }
        let host_accum_ns = accum_start.elapsed().as_nanos();
        let output_fini_start = Instant::now();
        output.fini()?;
        let output_fini_ns = output_fini_start.elapsed().as_nanos();

        Ok(Int8DecodeOutput {
            values,
            stats: Int8DecodeStats {
                k_slices: slices,
                npu_tasks: tasks.len(),
                pack_ns: input_stage_ns + regcmd_stage_ns,
                alloc_ns,
                input_stage_ns,
                partial_stage_ns: 0,
                regcmd_stage_ns,
                output_fini_ns,
                submit_ns,
                wait_ns,
                submit_wait_ns,
                host_accum_ns,
                total_ns: total_start.elapsed().as_nanos(),
            },
        })
    }

    /// Compatibility one-shot path. Prefer `prepare_weights` +
    /// `execute_prepared` when B is static across calls.
    pub fn execute(
        &self,
        a_k: &[i8],
        b_nk: &[i8],
        k: usize,
        n: usize,
    ) -> Result<Int8DecodeOutput, Int8DecodeError> {
        let total_start = Instant::now();
        validate_inputs(a_k, b_nk, k, n)?;
        let prepared = self.prepare_weights(b_nk, k, n)?;
        let prepare_ns = prepared.stats.pack_ns;
        let mut output = self.execute_prepared(a_k, &prepared)?;
        output.stats.pack_ns = output
            .stats
            .pack_ns
            .checked_add(prepare_ns)
            .ok_or(Int8DecodeError::SizeOverflow)?;
        output.stats.total_ns = total_start.elapsed().as_nanos();
        Ok(output)
    }

    /// Convert static B[N,K] INT8 weights into the native RK3588 32x32 layout
    /// once and keep them resident in a long-lived Rocket BO.
    pub fn prepare_weights(
        &self,
        b_nk: &[i8],
        k: usize,
        n: usize,
    ) -> Result<Int8PreparedWeights, Int8DecodeError> {
        validate_weights(b_nk, k, n)?;
        let slices = slice_count(k);
        let weight_bytes = k.checked_mul(n).ok_or(Int8DecodeError::SizeOverflow)?;
        let mut bo = self.device.alloc_owned_buffer(weight_bytes)?;
        check_dma32_range(bo.dma_address(), bo.len())?;

        let pack_start = Instant::now();
        bo.prep_relative(0)?;
        let mut offsets = Vec::with_capacity(slices);
        let mut weight_offset = 0usize;
        let b_bytes: &[u8] = bytemuck::cast_slice(b_nk);
        {
            let packed = bo.as_mut_slice();
            for slice in 0..slices {
                offsets.push(weight_offset);
                let k0 = slice_k0(slices, slice);
                let kp = slice_kp(k, slices, slice);
                let kt = kp / 32;
                for nt in 0..n / 32 {
                    for kb in 0..kt {
                        for nl in 0..32 {
                            let col = nt * 32 + nl;
                            let src = col * k + k0 + kb * 32;
                            let dst = weight_offset + nt * kt * 32 * 32 + kb * 32 * 32 + nl * 32;
                            packed[dst..dst + 32].copy_from_slice(&b_bytes[src..src + 32]);
                        }
                    }
                }
                weight_offset = weight_offset
                    .checked_add(kp.checked_mul(n).ok_or(Int8DecodeError::SizeOverflow)?)
                    .ok_or(Int8DecodeError::SizeOverflow)?;
            }
        }
        bo.fini()?;
        let pack_ns = pack_start.elapsed().as_nanos();

        Ok(Int8PreparedWeights {
            device_fd: self.device.fd(),
            k,
            n,
            slices,
            bo,
            offsets,
            stats: Int8PreparedWeightStats {
                k_slices: slices,
                resident_bytes: weight_bytes,
                pack_ns,
            },
        })
    }

    /// Begin A[1,K] against already packed/resident INT8 weights. This stages
    /// the per-call BOs and submits the asynchronous Rocket job, but does not
    /// wait for the output fence/BO yet.
    pub fn begin_execute_prepared(
        &self,
        a_k: &[i8],
        weights: &Int8PreparedWeights,
    ) -> Result<Int8PendingDecode<'a>, Int8DecodeError> {
        let total_start = Instant::now();
        if self.device.fd() != weights.device_fd {
            return Err(Int8DecodeError::InvalidInput(
                "prepared weights belong to a different Rocket device",
            ));
        }
        let k = weights.k;
        let n = weights.n;
        validate_activation(a_k, k, n)?;
        let slices = weights.slices;
        if weights.offsets.len() != slices {
            return Err(Int8DecodeError::InvalidInput(
                "prepared weight slice metadata mismatch",
            ));
        }

        let regcmd_bytes = slices
            .checked_mul(REGCMD_BYTES)
            .ok_or(Int8DecodeError::SizeOverflow)?;
        let partial_values = slices.checked_mul(n).ok_or(Int8DecodeError::SizeOverflow)?;
        let partial_bytes = partial_values
            .checked_mul(4)
            .ok_or(Int8DecodeError::SizeOverflow)?;

        let alloc_start = Instant::now();
        let mut regcmd = self.device.alloc_buffer(regcmd_bytes)?;
        let mut input = self.device.alloc_buffer(k)?;
        let mut partials = self.device.alloc_buffer(partial_bytes)?;
        for bo in [&regcmd, &input, &partials] {
            check_dma32_range(bo.dma_address(), bo.len())?;
        }
        let alloc_ns = alloc_start.elapsed().as_nanos();

        let pack_start = Instant::now();
        let input_stage_start = Instant::now();
        input.prep_relative(0)?;
        for (dst, src) in input.as_mut_slice().iter_mut().zip(a_k.iter().copied()) {
            *dst = src as u8;
        }
        input.fini()?;
        let input_stage_ns = input_stage_start.elapsed().as_nanos();

        let partial_stage_start = Instant::now();
        if std::env::var_os("ROCKNPU_EXPERIMENT_NO_PARTIAL_PREZERO").is_none() {
            partials.prep_relative(0)?;
            partials.as_mut_slice().fill(0);
            partials.fini()?;
        }
        let partial_stage_ns = partial_stage_start.elapsed().as_nanos();

        let regcmd_stage_start = Instant::now();
        regcmd.prep_relative(0)?;
        regcmd.as_mut_slice().fill(0);
        let mut tasks = Vec::with_capacity(slices);
        for slice in 0..slices {
            let k0 = slice_k0(slices, slice);
            let kp = slice_kp(k, slices, slice);
            let output_offset = slice
                .checked_mul(n)
                .and_then(|v| v.checked_mul(4))
                .ok_or(Int8DecodeError::SizeOverflow)?;
            let weight_dma = weights
                .bo
                .dma_address()
                .checked_add(
                    u64::try_from(weights.offsets[slice])
                        .map_err(|_| Int8DecodeError::SizeOverflow)?,
                )
                .ok_or(Int8DecodeError::SizeOverflow)?;
            let ops = encode_int8_decode_m1(Int8DecodeDesc::new(
                kp,
                n,
                input.dma_address()
                    + u64::try_from(k0).map_err(|_| Int8DecodeError::SizeOverflow)?,
                weight_dma,
                partials.dma_address()
                    + u64::try_from(output_offset).map_err(|_| Int8DecodeError::SizeOverflow)?,
            ))?;
            let reg_offset = slice
                .checked_mul(REGCMD_BYTES)
                .ok_or(Int8DecodeError::SizeOverflow)?;
            write_regcmd_at(&mut regcmd, reg_offset, &ops)?;
            let reg_addr = regcmd.dma_address()
                + u64::try_from(reg_offset).map_err(|_| Int8DecodeError::SizeOverflow)?;
            tasks.push(Task {
                regcmd: u32::try_from(reg_addr)
                    .map_err(|_| Int8DecodeError::AddressAbove32Bit(reg_addr))?,
                regcmd_count: u32::try_from(ops.len())
                    .map_err(|_| Int8DecodeError::SizeOverflow)?,
            });
        }
        regcmd.fini()?;
        let regcmd_stage_ns = regcmd_stage_start.elapsed().as_nanos();
        let pack_ns = pack_start.elapsed().as_nanos();

        let submit_wait_start = Instant::now();
        let submit_start = Instant::now();
        self.device.submit(
            &tasks,
            &[input.handle(), weights.bo.handle(), regcmd.handle()],
            &[partials.handle()],
        )?;
        let submit_ns = submit_start.elapsed().as_nanos();

        Ok(Int8PendingDecode {
            _regcmd: regcmd,
            _input: input,
            partials,
            slices,
            n,
            npu_tasks: tasks.len(),
            pack_ns,
            alloc_ns,
            input_stage_ns,
            partial_stage_ns,
            regcmd_stage_ns,
            submit_ns,
            submit_wait_start,
            total_start,
        })
    }

    pub fn finish_execute_prepared(
        &self,
        pending: Int8PendingDecode<'a>,
    ) -> Result<Int8DecodeOutput, Int8DecodeError> {
        let wait_start = Instant::now();
        pending.partials.prep_relative(WAIT_NS)?;
        let wait_ns = wait_start.elapsed().as_nanos();
        let submit_wait_ns = pending.submit_wait_start.elapsed().as_nanos();

        let accum_start = Instant::now();
        let mut values = vec![0i32; pending.n];
        for slice in 0..pending.slices {
            for (col, sum) in values.iter_mut().enumerate() {
                *sum = sum
                    .checked_add(read_i32(
                        pending.partials.as_slice(),
                        slice * pending.n + col,
                    ))
                    .ok_or(Int8DecodeError::InvalidInput("int32 accumulation overflow"))?;
            }
        }
        let host_accum_ns = accum_start.elapsed().as_nanos();
        let output_fini_start = Instant::now();
        pending.partials.fini()?;
        let output_fini_ns = output_fini_start.elapsed().as_nanos();

        Ok(Int8DecodeOutput {
            values,
            stats: Int8DecodeStats {
                k_slices: pending.slices,
                npu_tasks: pending.npu_tasks,
                pack_ns: pending.pack_ns,
                alloc_ns: pending.alloc_ns,
                input_stage_ns: pending.input_stage_ns,
                partial_stage_ns: pending.partial_stage_ns,
                regcmd_stage_ns: pending.regcmd_stage_ns,
                output_fini_ns,
                submit_ns: pending.submit_ns,
                wait_ns,
                submit_wait_ns,
                host_accum_ns,
                total_ns: pending.total_start.elapsed().as_nanos(),
            },
        })
    }

    pub(crate) fn begin_execute_prepared_owned(
        &self,
        a_k: &[i8],
        weights: &Int8PreparedWeights,
        scratch_slot: &mut Option<Int8OwnedScratch>,
    ) -> Result<Int8OwnedPending, Int8DecodeError> {
        let total_start = Instant::now();
        if self.device.fd() != weights.device_fd {
            return Err(Int8DecodeError::InvalidInput(
                "prepared weights belong to a different Rocket device",
            ));
        }
        let k = weights.k;
        let n = weights.n;
        validate_activation(a_k, k, n)?;
        let slices = weights.slices;
        if weights.offsets.len() != slices {
            return Err(Int8DecodeError::InvalidInput(
                "prepared weight slice metadata mismatch",
            ));
        }
        let regcmd_bytes = slices
            .checked_mul(REGCMD_BYTES)
            .ok_or(Int8DecodeError::SizeOverflow)?;
        let partial_values = slices.checked_mul(n).ok_or(Int8DecodeError::SizeOverflow)?;
        let partial_bytes = partial_values
            .checked_mul(4)
            .ok_or(Int8DecodeError::SizeOverflow)?;

        let alloc_start = Instant::now();
        let needs_grow = scratch_slot.as_ref().is_none_or(|scratch| {
            scratch.regcmd.len() < regcmd_bytes
                || scratch.input.len() < k
                || scratch.partials.len() < partial_bytes
        });
        if needs_grow {
            let old = scratch_slot.as_ref();
            let regcmd_capacity = old.map_or(regcmd_bytes, |s| s.regcmd.len().max(regcmd_bytes));
            let input_capacity = old.map_or(k, |s| s.input.len().max(k));
            let partial_capacity =
                old.map_or(partial_bytes, |s| s.partials.len().max(partial_bytes));
            let scratch = Int8OwnedScratch {
                regcmd: self.device.alloc_owned_buffer(regcmd_capacity)?,
                input: self.device.alloc_owned_buffer(input_capacity)?,
                partials: self.device.alloc_owned_buffer(partial_capacity)?,
            };
            for (addr, len) in [
                (scratch.regcmd.dma_address(), scratch.regcmd.len()),
                (scratch.input.dma_address(), scratch.input.len()),
                (scratch.partials.dma_address(), scratch.partials.len()),
            ] {
                check_dma32_range(addr, len)?;
            }
            *scratch_slot = Some(scratch);
        }
        let alloc_ns = alloc_start.elapsed().as_nanos();
        let scratch = scratch_slot
            .as_mut()
            .ok_or(Int8DecodeError::InvalidInput("persistent scratch missing"))?;

        let pack_start = Instant::now();
        let input_stage_start = Instant::now();
        scratch.input.prep_relative(0)?;
        for (dst, src) in scratch.input.as_mut_slice()[..k]
            .iter_mut()
            .zip(a_k.iter().copied())
        {
            *dst = src as u8;
        }
        scratch.input.fini()?;
        let input_stage_ns = input_stage_start.elapsed().as_nanos();

        // Every INT8 WDMA task fully overwrites its assigned int32 output range.
        // The persistent direct-submit path therefore does not need to acquire,
        // zero, and release the partial BO before handing it to the device.
        let partial_stage_ns = 0;

        let regcmd_stage_start = Instant::now();
        scratch.regcmd.prep_relative(0)?;
        scratch.regcmd.as_mut_slice()[..regcmd_bytes].fill(0);
        let mut tasks = Vec::with_capacity(slices);
        for slice in 0..slices {
            let k0 = slice_k0(slices, slice);
            let kp = slice_kp(k, slices, slice);
            let output_offset = slice
                .checked_mul(n)
                .and_then(|v| v.checked_mul(4))
                .ok_or(Int8DecodeError::SizeOverflow)?;
            let weight_dma = weights
                .bo
                .dma_address()
                .checked_add(
                    u64::try_from(weights.offsets[slice])
                        .map_err(|_| Int8DecodeError::SizeOverflow)?,
                )
                .ok_or(Int8DecodeError::SizeOverflow)?;
            let ops = encode_int8_decode_m1(Int8DecodeDesc::new(
                kp,
                n,
                scratch.input.dma_address()
                    + u64::try_from(k0).map_err(|_| Int8DecodeError::SizeOverflow)?,
                weight_dma,
                scratch.partials.dma_address()
                    + u64::try_from(output_offset).map_err(|_| Int8DecodeError::SizeOverflow)?,
            ))?;
            let reg_offset = slice
                .checked_mul(REGCMD_BYTES)
                .ok_or(Int8DecodeError::SizeOverflow)?;
            write_regcmd_bytes(scratch.regcmd.as_mut_slice(), reg_offset, &ops)?;
            let reg_addr = scratch.regcmd.dma_address()
                + u64::try_from(reg_offset).map_err(|_| Int8DecodeError::SizeOverflow)?;
            tasks.push(Task {
                regcmd: u32::try_from(reg_addr)
                    .map_err(|_| Int8DecodeError::AddressAbove32Bit(reg_addr))?,
                regcmd_count: u32::try_from(ops.len())
                    .map_err(|_| Int8DecodeError::SizeOverflow)?,
            });
        }
        scratch.regcmd.fini()?;
        let regcmd_stage_ns = regcmd_stage_start.elapsed().as_nanos();
        let pack_ns = pack_start.elapsed().as_nanos();

        let submit_wait_start = Instant::now();
        let submit_start = Instant::now();
        self.device.submit(
            &tasks,
            &[
                scratch.input.handle(),
                weights.bo.handle(),
                scratch.regcmd.handle(),
            ],
            &[scratch.partials.handle()],
        )?;
        let submit_ns = submit_start.elapsed().as_nanos();

        Ok(Int8OwnedPending {
            slices,
            n,
            npu_tasks: tasks.len(),
            pack_ns,
            alloc_ns,
            input_stage_ns,
            partial_stage_ns,
            regcmd_stage_ns,
            submit_ns,
            submit_wait_start,
            total_start,
        })
    }

    pub(crate) fn finish_execute_prepared_owned_into(
        &self,
        pending: Int8OwnedPending,
        scratch_slot: &mut Option<Int8OwnedScratch>,
        values: &mut [i32],
    ) -> Result<Int8DecodeStats, Int8DecodeError> {
        if values.len() != pending.n {
            return Err(Int8DecodeError::InvalidInput(
                "persistent output length must equal prepared N",
            ));
        }
        let scratch = scratch_slot
            .as_mut()
            .ok_or(Int8DecodeError::InvalidInput("persistent scratch missing"))?;
        let wait_start = Instant::now();
        scratch.partials.prep_relative(WAIT_NS)?;
        let wait_ns = wait_start.elapsed().as_nanos();
        let submit_wait_ns = pending.submit_wait_start.elapsed().as_nanos();

        let accum_start = Instant::now();
        for slice in 0..pending.slices {
            for (col, sum) in values.iter_mut().enumerate() {
                *sum = sum
                    .checked_add(read_i32(
                        scratch.partials.as_slice(),
                        slice * pending.n + col,
                    ))
                    .ok_or(Int8DecodeError::InvalidInput("int32 accumulation overflow"))?;
            }
        }
        let host_accum_ns = accum_start.elapsed().as_nanos();
        let output_fini_start = Instant::now();
        scratch.partials.fini()?;
        let output_fini_ns = output_fini_start.elapsed().as_nanos();

        Ok(Int8DecodeStats {
            k_slices: pending.slices,
            npu_tasks: pending.npu_tasks,
            pack_ns: pending.pack_ns,
            alloc_ns: pending.alloc_ns,
            input_stage_ns: pending.input_stage_ns,
            partial_stage_ns: pending.partial_stage_ns,
            regcmd_stage_ns: pending.regcmd_stage_ns,
            output_fini_ns,
            submit_ns: pending.submit_ns,
            wait_ns,
            submit_wait_ns,
            host_accum_ns,
            total_ns: pending.total_start.elapsed().as_nanos(),
        })
    }

    /// Execute A[1,K] against already packed/resident INT8 weights. The static
    /// weight BO is neither CPU-touched nor repacked on this path.
    pub fn execute_prepared(
        &self,
        a_k: &[i8],
        weights: &Int8PreparedWeights,
    ) -> Result<Int8DecodeOutput, Int8DecodeError> {
        let pending = self.begin_execute_prepared(a_k, weights)?;
        self.finish_execute_prepared(pending)
    }
}

fn validate_shape(k: usize, n: usize) -> Result<(), Int8DecodeError> {
    if k == 0 || n == 0 {
        return Err(Int8DecodeError::InvalidInput("K and N must be non-zero"));
    }
    if !k.is_multiple_of(256) {
        return Err(Int8DecodeError::InvalidInput("K must be a multiple of 256"));
    }
    if !n.is_multiple_of(32) || n > N_MAX {
        return Err(Int8DecodeError::InvalidInput(
            "N must be a multiple of 32 and <=8192",
        ));
    }
    Ok(())
}

fn validate_activation(a: &[i8], k: usize, n: usize) -> Result<(), Int8DecodeError> {
    validate_shape(k, n)?;
    if a.len() != k {
        return Err(Int8DecodeError::InvalidInput("A length must equal K"));
    }
    Ok(())
}

fn validate_weights(b: &[i8], k: usize, n: usize) -> Result<(), Int8DecodeError> {
    validate_shape(k, n)?;
    if b.len() != n.checked_mul(k).ok_or(Int8DecodeError::SizeOverflow)? {
        return Err(Int8DecodeError::InvalidInput("B length must equal N*K"));
    }
    Ok(())
}

fn validate_inputs(a: &[i8], b: &[i8], k: usize, n: usize) -> Result<(), Int8DecodeError> {
    validate_activation(a, k, n)?;
    validate_weights(b, k, n)
}

fn slice_count(k: usize) -> usize {
    if k <= SINGLE_SUBMIT_K_MAX {
        1
    } else {
        k.div_ceil(WIDE_K_SLICE)
    }
}

fn slice_k0(slices: usize, slice: usize) -> usize {
    if slices == 1 { 0 } else { slice * WIDE_K_SLICE }
}

fn slice_kp(k: usize, slices: usize, slice: usize) -> usize {
    if slices == 1 {
        k
    } else {
        (k - slice * WIDE_K_SLICE).min(WIDE_K_SLICE)
    }
}

fn check_dma32_range(addr: u64, bytes: usize) -> Result<(), Int8DecodeError> {
    let last = addr
        .checked_add(
            u64::try_from(bytes.saturating_sub(1)).map_err(|_| Int8DecodeError::SizeOverflow)?,
        )
        .ok_or(Int8DecodeError::SizeOverflow)?;
    if last > u32::MAX as u64 {
        Err(Int8DecodeError::AddressAbove32Bit(last))
    } else {
        Ok(())
    }
}

fn write_regcmd_at(
    bo: &mut RocketBuffer<'_>,
    offset: usize,
    ops: &[u64; INT8_REGCMD_COUNT],
) -> Result<(), Int8DecodeError> {
    write_regcmd_bytes(bo.as_mut_slice(), offset, ops)
}

fn write_regcmd_bytes(
    bytes_out: &mut [u8],
    offset: usize,
    ops: &[u64; INT8_REGCMD_COUNT],
) -> Result<(), Int8DecodeError> {
    let bytes = ops
        .len()
        .checked_mul(8)
        .ok_or(Int8DecodeError::SizeOverflow)?;
    let end = offset
        .checked_add(bytes)
        .ok_or(Int8DecodeError::SizeOverflow)?;
    if end > bytes_out.len() {
        return Err(Int8DecodeError::InvalidInput("regcmd BO too small"));
    }
    for (chunk, word) in bytes_out[offset..end].chunks_exact_mut(8).zip(ops.iter()) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    Ok(())
}

fn read_i32(bytes: &[u8], index: usize) -> i32 {
    let p = index * 4;
    i32::from_le_bytes(
        bytes[p..p + 4]
            .try_into()
            .expect("validated partial output index"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_decode_shapes() {
        assert!(validate_inputs(&[0; 512], &[0; 512 * 32], 512, 32).is_ok());
        assert!(validate_inputs(&[0; 511], &[0; 512 * 32], 512, 32).is_err());
        assert!(validate_inputs(&[0; 512], &[0; 512 * 31], 512, 31).is_err());
        assert!(validate_inputs(&[0; 513], &[0; 513 * 32], 513, 32).is_err());
    }

    #[test]
    fn wide_k_slicing_matches_tinyllama_down_projection() {
        let k = 5632usize;
        let slices = slice_count(k);
        assert_eq!(slices, 6);
        assert_eq!(
            (0..slices)
                .map(|slice| slice_kp(k, slices, slice))
                .collect::<Vec<_>>(),
            vec![1024, 1024, 1024, 1024, 1024, 512]
        );
    }
}
