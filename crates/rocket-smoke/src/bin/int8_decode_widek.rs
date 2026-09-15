use rocket_runtime::{RocketDevice, Task};
use rocknpu_regcmd::{Int8DecodeDesc, encode_int8_decode_m1, weight_i8_fullk_index};
use std::time::Instant;

const K: usize = 5632;
const N: usize = 2048;
const KS: usize = 1024;
const WAIT_NS: i64 = 2_000_000_000;
const REGCMD_BYTES: usize = 4096;

fn read_i32(bytes: &[u8], index: usize) -> i32 {
    let p = index * 4;
    i32::from_le_bytes(bytes[p..p + 4].try_into().unwrap())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = RocketDevice::open()?;
    let mut guard = device.alloc_buffer(4096)?;
    guard.prep_relative(0)?;
    guard.as_mut_slice().fill(0);
    guard.fini()?;

    let slices = K.div_ceil(KS);
    let mut regcmd = device.alloc_buffer(slices * REGCMD_BYTES)?;
    let mut input = device.alloc_buffer(K)?;
    let mut weights = device.alloc_buffer(K * N)?;
    let mut partials = device.alloc_buffer(slices * N * 4)?;
    for bo in [&regcmd, &input, &weights, &partials] {
        if bo.dma_address() >> 32 != 0 {
            return Err(format!(
                "BO IOVA 0x{:x} exceeds 32-bit regcmd window",
                bo.dma_address()
            )
            .into());
        }
    }

    let a: Vec<i8> = (0..K).map(|i| ((i * 7 + 3) % 9) as i8 - 4).collect();
    let b: Vec<i8> = (0..K * N).map(|i| ((i * 5 + 1) % 11) as i8 - 5).collect();
    let mut reference = vec![0i32; N];
    for col in 0..N {
        let mut sum = 0i32;
        for kk in 0..K {
            sum += i32::from(a[kk]) * i32::from(b[kk * N + col]);
        }
        reference[col] = sum;
    }

    input.prep_relative(0)?;
    for (dst, src) in input.as_mut_slice().iter_mut().zip(a.iter().copied()) {
        *dst = src as u8;
    }
    input.fini()?;

    weights.prep_relative(0)?;
    weights.as_mut_slice().fill(0);
    let mut weight_offset = 0usize;
    for slice in 0..slices {
        let k0 = slice * KS;
        let kp = (K - k0).min(KS);
        for kk in 0..kp {
            for col in 0..N {
                let dst = weight_offset + weight_i8_fullk_index(kp, N, kk, col);
                weights.as_mut_slice()[dst] = b[(k0 + kk) * N + col] as u8;
            }
        }
        weight_offset += kp * N;
    }
    weights.fini()?;

    partials.prep_relative(0)?;
    partials.as_mut_slice().fill(0xa5);
    partials.fini()?;

    regcmd.prep_relative(0)?;
    regcmd.as_mut_slice().fill(0);
    let mut tasks = Vec::with_capacity(slices);
    let mut weight_offset = 0usize;
    for slice in 0..slices {
        let k0 = slice * KS;
        let kp = (K - k0).min(KS);
        let ops = encode_int8_decode_m1(Int8DecodeDesc::new(
            kp,
            N,
            input.dma_address() + u64::try_from(k0)?,
            weights.dma_address() + u64::try_from(weight_offset)?,
            partials.dma_address() + u64::try_from(slice * N * 4)?,
        ))?;
        let reg_offset = slice * REGCMD_BYTES;
        for (chunk, word) in regcmd.as_mut_slice()[reg_offset..]
            .chunks_exact_mut(8)
            .zip(ops.iter())
        {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        tasks.push(Task {
            regcmd: u32::try_from(regcmd.dma_address() + u64::try_from(reg_offset)?)?,
            regcmd_count: u32::try_from(ops.len())?,
        });
        weight_offset += kp * N;
    }
    regcmd.fini()?;

    let inputs = [input.handle(), weights.handle(), regcmd.handle()];
    let outputs = [partials.handle()];

    // Warm once, then measure the identical Rocket multi-task submission. Each
    // task is one ork-style K slice; accumulation is deliberately on the host,
    // matching ork-driver's verified K>4096 decode strategy.
    device.submit(&tasks, &inputs, &outputs)?;
    partials.prep_relative(WAIT_NS)?;
    partials.fini()?;

    let start = Instant::now();
    device.submit(&tasks, &inputs, &outputs)?;
    partials.prep_relative(WAIT_NS)?;
    let submit_elapsed = start.elapsed();

    let accumulate_start = Instant::now();
    let mut got = vec![0i32; N];
    for slice in 0..slices {
        for (col, sum) in got.iter_mut().enumerate() {
            *sum += read_i32(partials.as_slice(), slice * N + col);
        }
    }
    let accumulate_elapsed = accumulate_start.elapsed();
    partials.fini()?;

    let mut mismatches = Vec::new();
    for (col, (&expected, &actual)) in reference.iter().zip(&got).enumerate() {
        if actual != expected && mismatches.len() < 10 {
            mismatches.push((col, expected, actual));
        }
    }
    if !mismatches.is_empty() {
        return Err(
            format!("INT8 WIDE-K M=1 K={K} N={N} mismatch examples: {mismatches:?}").into(),
        );
    }

    println!(
        "INT8 WIDE-K DECODE PASS M=1 K={K} N={N} slices={slices} outputs={N} submit_wait_us={:.1} host_accum_us={:.1}",
        submit_elapsed.as_secs_f64() * 1e6,
        accumulate_elapsed.as_secs_f64() * 1e6,
    );
    Ok(())
}
