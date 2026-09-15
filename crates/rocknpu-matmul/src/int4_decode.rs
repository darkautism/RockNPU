// SPDX-License-Identifier: MIT

use rocket_runtime::{RocketBuffer, RocketDevice, RocketOwnedBuffer, Task};
use rocknpu_regcmd::{INT4_REGCMD_COUNT, Int4DecodeDesc, Int4EncodeError, encode_int4_decode_m1};
use std::fmt;
use std::io;
use std::os::fd::RawFd;
use std::time::Instant;

const WAIT_NS: i64 = 2_000_000_000;
const REGCMD_BYTES: usize = 4096;
const K_MAX: usize = 10_752;
const N_MAX: usize = 8_192;

#[derive(Debug)]
pub enum Int4DecodeError {
    InvalidInput(&'static str),
    Encode(Int4EncodeError),
    Io(io::Error),
    AddressAbove32Bit(u64),
    SizeOverflow,
}

impl fmt::Display for Int4DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid native W4A4 M=1 MatMul input: {msg}"),
            Self::Encode(err) => err.fmt(f),
            Self::Io(err) => err.fmt(f),
            Self::AddressAbove32Bit(addr) => write!(
                f,
                "Rocket BO IOVA 0x{addr:x} exceeds the 32-bit RK3588 regcmd field"
            ),
            Self::SizeOverflow => write!(f, "native W4A4 M=1 MatMul size arithmetic overflow"),
        }
    }
}
impl std::error::Error for Int4DecodeError {}
impl From<Int4EncodeError> for Int4DecodeError {
    fn from(value: Int4EncodeError) -> Self {
        Self::Encode(value)
    }
}
impl From<io::Error> for Int4DecodeError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Int4PreparedWeightStats {
    pub resident_bytes: usize,
    pub pack_ns: u128,
}

pub struct Int4PreparedWeights {
    device_fd: RawFd,
    k: usize,
    n: usize,
    bo: RocketOwnedBuffer,
    stats: Int4PreparedWeightStats,
}

impl Int4PreparedWeights {
    pub fn k(&self) -> usize {
        self.k
    }
    pub fn n(&self) -> usize {
        self.n
    }
    pub fn stats(&self) -> Int4PreparedWeightStats {
        self.stats
    }
}

pub struct Int4GroupedPreparedWeights {
    device_fd: RawFd,
    k: usize,
    n: usize,
    group_size: usize,
    groups: usize,
    bo: RocketOwnedBuffer,
    offsets: Vec<usize>,
    stats: Int4PreparedWeightStats,
}

impl Int4GroupedPreparedWeights {
    pub fn k(&self) -> usize {
        self.k
    }
    pub fn n(&self) -> usize {
        self.n
    }
    pub fn group_size(&self) -> usize {
        self.group_size
    }
    pub fn groups(&self) -> usize {
        self.groups
    }
    pub fn stats(&self) -> Int4PreparedWeightStats {
        self.stats
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Int4DecodeStats {
    pub pack_ns: u128,
    pub submit_ns: u128,
    pub wait_ns: u128,
    pub submit_wait_ns: u128,
    pub total_ns: u128,
    pub saturated_outputs: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Int4DecodeOutput {
    pub values: Vec<i16>,
    pub stats: Int4DecodeStats,
}

/// Native RK3588 W4A4 M=1 executor.
///
/// Public activations and weights contain signed int4 codes in `i8` storage.
/// Static B[N,K] weights are nibble-packed into the validated native layout once
/// and kept resident. Each decode call packs only A[1,K], executes one Rocket
/// task, and returns the dense linear int16[N] accumulator surface.
pub struct Int4DecodeExecutor<'a> {
    device: &'a RocketDevice,
}

impl<'a> Int4DecodeExecutor<'a> {
    pub const fn new(device: &'a RocketDevice) -> Self {
        Self { device }
    }

    pub fn prepare_weights(
        &self,
        b_nk: &[i8],
        k: usize,
        n: usize,
    ) -> Result<Int4PreparedWeights, Int4DecodeError> {
        validate_weights(b_nk, k, n)?;
        let weight_bytes = k
            .checked_mul(n)
            .and_then(|v| v.checked_div(2))
            .ok_or(Int4DecodeError::SizeOverflow)?;
        let mut bo = self.device.alloc_owned_buffer(weight_bytes)?;
        check_dma32_range(bo.dma_address(), bo.len())?;

        let pack_start = Instant::now();
        bo.prep_relative(0)?;
        bo.as_mut_slice().fill(0);
        let kt = k / 32;
        for nb in 0..n / 64 {
            for k_tile in 0..kt {
                for n_lane in 0..64 {
                    for k_lane in 0..32 {
                        let packed_index = (((nb * kt + k_tile) * 64 + n_lane) * 32) + k_lane;
                        let col = nb * 64 + n_lane;
                        let row = k_tile * 32 + k_lane;
                        put_nibble(bo.as_mut_slice(), packed_index, b_nk[col * k + row]);
                    }
                }
            }
        }
        bo.fini()?;

        Ok(Int4PreparedWeights {
            device_fd: self.device.fd(),
            k,
            n,
            bo,
            stats: Int4PreparedWeightStats {
                resident_bytes: weight_bytes,
                pack_ns: pack_start.elapsed().as_nanos(),
            },
        })
    }

    pub fn prepare_grouped_weights(
        &self,
        b_nk: &[i8],
        k: usize,
        n: usize,
        group_size: usize,
    ) -> Result<Int4GroupedPreparedWeights, Int4DecodeError> {
        validate_weights(b_nk, k, n)?;
        if group_size == 0 || !group_size.is_multiple_of(32) || !k.is_multiple_of(group_size) {
            return Err(Int4DecodeError::InvalidInput(
                "group size must be a non-zero multiple of 32 that divides K",
            ));
        }
        let groups = k / group_size;
        let group_bytes = group_size
            .checked_mul(n)
            .and_then(|v| v.checked_div(2))
            .ok_or(Int4DecodeError::SizeOverflow)?;
        let weight_bytes = group_bytes
            .checked_mul(groups)
            .ok_or(Int4DecodeError::SizeOverflow)?;
        let mut bo = self.device.alloc_owned_buffer(weight_bytes)?;
        check_dma32_range(bo.dma_address(), bo.len())?;
        let pack_start = Instant::now();
        bo.prep_relative(0)?;
        bo.as_mut_slice().fill(0);
        let mut offsets = Vec::with_capacity(groups);
        for group in 0..groups {
            let offset = group
                .checked_mul(group_bytes)
                .ok_or(Int4DecodeError::SizeOverflow)?;
            offsets.push(offset);
            pack_weight_group(
                &mut bo.as_mut_slice()[offset..offset + group_bytes],
                b_nk,
                k,
                n,
                group * group_size,
                group_size,
            );
        }
        bo.fini()?;
        Ok(Int4GroupedPreparedWeights {
            device_fd: self.device.fd(),
            k,
            n,
            group_size,
            groups,
            bo,
            offsets,
            stats: Int4PreparedWeightStats {
                resident_bytes: weight_bytes,
                pack_ns: pack_start.elapsed().as_nanos(),
            },
        })
    }

    pub fn execute_prepared(
        &self,
        a_k: &[i8],
        weights: &Int4PreparedWeights,
    ) -> Result<Int4DecodeOutput, Int4DecodeError> {
        let total_start = Instant::now();
        if self.device.fd() != weights.device_fd {
            return Err(Int4DecodeError::InvalidInput(
                "prepared weights belong to a different Rocket device",
            ));
        }
        let k = weights.k;
        let n = weights.n;
        validate_activation(a_k, k, n)?;

        let input_bytes = k.checked_div(2).ok_or(Int4DecodeError::SizeOverflow)?;
        let output_bytes = n.checked_mul(2).ok_or(Int4DecodeError::SizeOverflow)?;
        let mut regcmd = self.device.alloc_buffer(REGCMD_BYTES)?;
        let mut input = self.device.alloc_buffer(input_bytes)?;
        let mut output = self.device.alloc_buffer(output_bytes)?;
        for bo in [&regcmd, &input, &output] {
            check_dma32_range(bo.dma_address(), bo.len())?;
        }

        let pack_start = Instant::now();
        input.prep_relative(0)?;
        input.as_mut_slice().fill(0);
        for (index, &value) in a_k.iter().enumerate() {
            put_nibble(input.as_mut_slice(), index, value);
        }
        input.fini()?;

        output.prep_relative(0)?;
        output.as_mut_slice().fill(0x7f);
        output.fini()?;

        let ops = encode_int4_decode_m1(Int4DecodeDesc::new(
            k,
            n,
            input.dma_address(),
            weights.bo.dma_address(),
            output.dma_address(),
        ))?;
        write_regcmd(&mut regcmd, &ops)?;
        let pack_ns = pack_start.elapsed().as_nanos();

        let regcmd_addr = regcmd.dma_address();
        let task = Task {
            regcmd: u32::try_from(regcmd_addr)
                .map_err(|_| Int4DecodeError::AddressAbove32Bit(regcmd_addr))?,
            regcmd_count: u32::try_from(INT4_REGCMD_COUNT)
                .map_err(|_| Int4DecodeError::SizeOverflow)?,
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

        let mut values = Vec::with_capacity(n);
        let mut saturated_outputs = 0usize;
        for index in 0..n {
            let value = read_i16(output.as_slice(), index);
            saturated_outputs += usize::from(value == i16::MIN || value == i16::MAX);
            values.push(value);
        }
        output.fini()?;

        Ok(Int4DecodeOutput {
            values,
            stats: Int4DecodeStats {
                pack_ns,
                submit_ns,
                wait_ns,
                submit_wait_ns,
                total_ns: total_start.elapsed().as_nanos(),
                saturated_outputs,
            },
        })
    }

    /// Run independently scaled K-groups as ordinary tasks in one Rocket job.
    /// Result layout is group-major `[groups, N]` int16 partials.
    pub fn execute_grouped_prepared(
        &self,
        a_k: &[i8],
        weights: &Int4GroupedPreparedWeights,
    ) -> Result<Int4DecodeOutput, Int4DecodeError> {
        let total_start = Instant::now();
        if self.device.fd() != weights.device_fd {
            return Err(Int4DecodeError::InvalidInput(
                "prepared weights belong to a different Rocket device",
            ));
        }
        validate_activation(a_k, weights.k, weights.n)?;
        if weights.offsets.len() != weights.groups {
            return Err(Int4DecodeError::InvalidInput(
                "grouped prepared weight metadata mismatch",
            ));
        }
        let partial_values = weights
            .groups
            .checked_mul(weights.n)
            .ok_or(Int4DecodeError::SizeOverflow)?;
        let mut regcmd = self.device.alloc_buffer(
            weights
                .groups
                .checked_mul(REGCMD_BYTES)
                .ok_or(Int4DecodeError::SizeOverflow)?,
        )?;
        let mut input = self.device.alloc_buffer(weights.k / 2)?;
        let mut output = self.device.alloc_buffer(
            partial_values
                .checked_mul(2)
                .ok_or(Int4DecodeError::SizeOverflow)?,
        )?;
        for bo in [&regcmd, &input, &output] {
            check_dma32_range(bo.dma_address(), bo.len())?;
        }

        let pack_start = Instant::now();
        input.prep_relative(0)?;
        input.as_mut_slice().fill(0);
        for (index, &value) in a_k.iter().enumerate() {
            put_nibble(input.as_mut_slice(), index, value);
        }
        input.fini()?;
        output.prep_relative(0)?;
        output.as_mut_slice().fill(0x7f);
        output.fini()?;
        regcmd.prep_relative(0)?;
        regcmd.as_mut_slice().fill(0);

        let mut tasks = Vec::with_capacity(weights.groups);
        for group in 0..weights.groups {
            let input_offset = group
                .checked_mul(weights.group_size / 2)
                .ok_or(Int4DecodeError::SizeOverflow)?;
            let output_offset = group
                .checked_mul(weights.n)
                .and_then(|v| v.checked_mul(2))
                .ok_or(Int4DecodeError::SizeOverflow)?;
            let weight_dma = weights
                .bo
                .dma_address()
                .checked_add(
                    u64::try_from(weights.offsets[group])
                        .map_err(|_| Int4DecodeError::SizeOverflow)?,
                )
                .ok_or(Int4DecodeError::SizeOverflow)?;
            let input_dma = input
                .dma_address()
                .checked_add(
                    u64::try_from(input_offset).map_err(|_| Int4DecodeError::SizeOverflow)?,
                )
                .ok_or(Int4DecodeError::SizeOverflow)?;
            let output_dma = output
                .dma_address()
                .checked_add(
                    u64::try_from(output_offset).map_err(|_| Int4DecodeError::SizeOverflow)?,
                )
                .ok_or(Int4DecodeError::SizeOverflow)?;
            let ops = encode_int4_decode_m1(Int4DecodeDesc::new(
                weights.group_size,
                weights.n,
                input_dma,
                weight_dma,
                output_dma,
            ))?;
            let reg_offset = group
                .checked_mul(REGCMD_BYTES)
                .ok_or(Int4DecodeError::SizeOverflow)?;
            write_regcmd_bytes(regcmd.as_mut_slice(), reg_offset, &ops)?;
            let regcmd_addr = regcmd
                .dma_address()
                .checked_add(u64::try_from(reg_offset).map_err(|_| Int4DecodeError::SizeOverflow)?)
                .ok_or(Int4DecodeError::SizeOverflow)?;
            tasks.push(Task {
                regcmd: u32::try_from(regcmd_addr)
                    .map_err(|_| Int4DecodeError::AddressAbove32Bit(regcmd_addr))?,
                regcmd_count: u32::try_from(INT4_REGCMD_COUNT)
                    .map_err(|_| Int4DecodeError::SizeOverflow)?,
            });
        }
        regcmd.fini()?;
        let pack_ns = pack_start.elapsed().as_nanos();
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

        let mut values = Vec::with_capacity(partial_values);
        let mut saturated_outputs = 0usize;
        for index in 0..partial_values {
            let value = read_i16(output.as_slice(), index);
            saturated_outputs += usize::from(value == i16::MIN || value == i16::MAX);
            values.push(value);
        }
        output.fini()?;
        Ok(Int4DecodeOutput {
            values,
            stats: Int4DecodeStats {
                pack_ns,
                submit_ns,
                wait_ns,
                submit_wait_ns,
                total_ns: total_start.elapsed().as_nanos(),
                saturated_outputs,
            },
        })
    }
}

fn validate_shape(k: usize, n: usize) -> Result<(), Int4DecodeError> {
    if k == 0 || n == 0 {
        return Err(Int4DecodeError::InvalidInput("K and N must be non-zero"));
    }
    if !k.is_multiple_of(32) || k > K_MAX {
        return Err(Int4DecodeError::InvalidInput(
            "K must be a multiple of 32 and <=10752",
        ));
    }
    if !n.is_multiple_of(64) || n > N_MAX {
        return Err(Int4DecodeError::InvalidInput(
            "N must be a multiple of 64 and <=8192",
        ));
    }
    Ok(())
}

fn validate_activation(a: &[i8], k: usize, n: usize) -> Result<(), Int4DecodeError> {
    validate_shape(k, n)?;
    if a.len() != k {
        return Err(Int4DecodeError::InvalidInput("A length must equal K"));
    }
    if a.iter().any(|&v| !(-8..=7).contains(&v)) {
        return Err(Int4DecodeError::InvalidInput(
            "A codes must fit signed int4",
        ));
    }
    Ok(())
}

fn validate_weights(b: &[i8], k: usize, n: usize) -> Result<(), Int4DecodeError> {
    validate_shape(k, n)?;
    if b.len() != n.checked_mul(k).ok_or(Int4DecodeError::SizeOverflow)? {
        return Err(Int4DecodeError::InvalidInput("B length must equal N*K"));
    }
    if b.iter().any(|&v| !(-8..=7).contains(&v)) {
        return Err(Int4DecodeError::InvalidInput(
            "B codes must fit signed int4",
        ));
    }
    Ok(())
}

fn pack_weight_group(dst: &mut [u8], b_nk: &[i8], full_k: usize, n: usize, k0: usize, kp: usize) {
    dst.fill(0);
    let kt = kp / 32;
    for nb in 0..n / 64 {
        for k_tile in 0..kt {
            for n_lane in 0..64 {
                for k_lane in 0..32 {
                    let packed_index = (((nb * kt + k_tile) * 64 + n_lane) * 32) + k_lane;
                    let col = nb * 64 + n_lane;
                    let row = k0 + k_tile * 32 + k_lane;
                    put_nibble(dst, packed_index, b_nk[col * full_k + row]);
                }
            }
        }
    }
}

fn put_nibble(dst: &mut [u8], index: usize, value: i8) {
    let nibble = (value as u8) & 0x0f;
    if index.is_multiple_of(2) {
        dst[index / 2] = (dst[index / 2] & 0xf0) | nibble;
    } else {
        dst[index / 2] = (dst[index / 2] & 0x0f) | (nibble << 4);
    }
}

fn check_dma32_range(addr: u64, bytes: usize) -> Result<(), Int4DecodeError> {
    let last = addr
        .checked_add(
            u64::try_from(bytes.saturating_sub(1)).map_err(|_| Int4DecodeError::SizeOverflow)?,
        )
        .ok_or(Int4DecodeError::SizeOverflow)?;
    if last > u32::MAX as u64 {
        Err(Int4DecodeError::AddressAbove32Bit(last))
    } else {
        Ok(())
    }
}

fn write_regcmd(
    bo: &mut RocketBuffer<'_>,
    ops: &[u64; INT4_REGCMD_COUNT],
) -> Result<(), Int4DecodeError> {
    bo.prep_relative(0)?;
    bo.as_mut_slice().fill(0);
    write_regcmd_bytes(bo.as_mut_slice(), 0, ops)?;
    bo.fini()?;
    Ok(())
}

fn write_regcmd_bytes(
    dst: &mut [u8],
    offset: usize,
    ops: &[u64; INT4_REGCMD_COUNT],
) -> Result<(), Int4DecodeError> {
    let bytes = ops
        .len()
        .checked_mul(8)
        .ok_or(Int4DecodeError::SizeOverflow)?;
    let end = offset
        .checked_add(bytes)
        .ok_or(Int4DecodeError::SizeOverflow)?;
    if end > dst.len() {
        return Err(Int4DecodeError::InvalidInput("regcmd BO too small"));
    }
    for (chunk, word) in dst[offset..end].chunks_exact_mut(8).zip(ops.iter()) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    Ok(())
}

fn read_i16(bytes: &[u8], index: usize) -> i16 {
    let p = index * 2;
    i16::from_le_bytes(
        bytes[p..p + 2]
            .try_into()
            .expect("validated native W4A4 output index"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_tinyllama_decode_shapes() {
        assert!(validate_shape(2048, 256).is_ok());
        assert!(validate_shape(2048, 2048).is_ok());
        assert!(validate_shape(2048, 5632).is_ok());
        assert!(validate_shape(5632, 2048).is_ok());
    }

    #[test]
    fn rejects_outside_native_single_program_envelope() {
        assert!(validate_shape(2048, 96).is_err());
        assert!(validate_shape(10784, 2048).is_err());
        assert!(validate_shape(2048, 8256).is_err());
    }
}
