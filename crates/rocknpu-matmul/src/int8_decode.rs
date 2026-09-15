// SPDX-License-Identifier: MIT

use rocket_runtime::{RocketBuffer, RocketDevice, RocketOwnedBuffer, Task};
use rocknpu_regcmd::{
    INT8_REGCMD_COUNT, Int8DecodeDesc, Int8EncodeError, encode_int8_decode_m1,
    weight_i8_fullk_index,
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
    _guard: RocketBuffer<'a>,
}

impl<'a> Int8DecodeExecutor<'a> {
    pub fn new(device: &'a RocketDevice) -> Result<Self, Int8DecodeError> {
        let guard = device.alloc_buffer(4096)?;
        check_dma32_range(guard.dma_address(), guard.len())?;
        Ok(Self {
            device,
            _guard: guard,
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
        bo.as_mut_slice().fill(0);
        let mut offsets = Vec::with_capacity(slices);
        let mut weight_offset = 0usize;
        for slice in 0..slices {
            offsets.push(weight_offset);
            let k0 = slice_k0(slices, slice);
            let kp = slice_kp(k, slices, slice);
            for kk in 0..kp {
                for col in 0..n {
                    let dst = weight_offset + weight_i8_fullk_index(kp, n, kk, col);
                    bo.as_mut_slice()[dst] = b_nk[col * k + k0 + kk] as u8;
                }
            }
            weight_offset = weight_offset
                .checked_add(kp.checked_mul(n).ok_or(Int8DecodeError::SizeOverflow)?)
                .ok_or(Int8DecodeError::SizeOverflow)?;
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

    /// Execute A[1,K] against already packed/resident INT8 weights. The static
    /// weight BO is neither CPU-touched nor repacked on this path.
    pub fn execute_prepared(
        &self,
        a_k: &[i8],
        weights: &Int8PreparedWeights,
    ) -> Result<Int8DecodeOutput, Int8DecodeError> {
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

        let mut regcmd = self.device.alloc_buffer(regcmd_bytes)?;
        let mut input = self.device.alloc_buffer(k)?;
        let mut partials = self.device.alloc_buffer(partial_bytes)?;
        for bo in [&regcmd, &input, &partials] {
            check_dma32_range(bo.dma_address(), bo.len())?;
        }

        let pack_start = Instant::now();
        input.prep_relative(0)?;
        for (dst, src) in input.as_mut_slice().iter_mut().zip(a_k.iter().copied()) {
            *dst = src as u8;
        }
        input.fini()?;

        partials.prep_relative(0)?;
        partials.as_mut_slice().fill(0);
        partials.fini()?;

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
        let pack_ns = pack_start.elapsed().as_nanos();

        let submit_wait_start = Instant::now();
        let submit_start = Instant::now();
        self.device.submit(
            &tasks,
            &[input.handle(), weights.bo.handle(), regcmd.handle()],
            &[partials.handle()],
        )?;
        let submit_ns = submit_start.elapsed().as_nanos();
        let wait_start = Instant::now();
        partials.prep_relative(WAIT_NS)?;
        let wait_ns = wait_start.elapsed().as_nanos();
        let submit_wait_ns = submit_wait_start.elapsed().as_nanos();

        let accum_start = Instant::now();
        let mut values = vec![0i32; n];
        for slice in 0..slices {
            for (col, sum) in values.iter_mut().enumerate() {
                *sum = sum
                    .checked_add(read_i32(partials.as_slice(), slice * n + col))
                    .ok_or(Int8DecodeError::InvalidInput("int32 accumulation overflow"))?;
            }
        }
        let host_accum_ns = accum_start.elapsed().as_nanos();
        partials.fini()?;

        Ok(Int8DecodeOutput {
            values,
            stats: Int8DecodeStats {
                k_slices: slices,
                npu_tasks: tasks.len(),
                pack_ns,
                submit_ns,
                wait_ns,
                submit_wait_ns,
                host_accum_ns,
                total_ns: total_start.elapsed().as_nanos(),
            },
        })
    }
}

fn validate_shape(k: usize, n: usize) -> Result<(), Int8DecodeError> {
    if k == 0 || n == 0 {
        return Err(Int8DecodeError::InvalidInput("K and N must be non-zero"));
    }
    if !k.is_multiple_of(512) {
        return Err(Int8DecodeError::InvalidInput("K must be a multiple of 512"));
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
    let bytes = ops
        .len()
        .checked_mul(8)
        .ok_or(Int8DecodeError::SizeOverflow)?;
    let end = offset
        .checked_add(bytes)
        .ok_or(Int8DecodeError::SizeOverflow)?;
    if end > bo.len() {
        return Err(Int8DecodeError::InvalidInput("regcmd BO too small"));
    }
    for (chunk, word) in bo.as_mut_slice()[offset..end]
        .chunks_exact_mut(8)
        .zip(ops.iter())
    {
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
