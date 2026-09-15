use rocket_runtime::{RocketDevice, Task};
use rocknpu_regcmd::{Int8DecodeDesc, encode_int8_decode_m1, weight_i8_fullk_index};
use std::time::Instant;

const DEFAULT_K: usize = 2048;
const DEFAULT_N: usize = 2048;
const WAIT_NS: i64 = 2_000_000_000;

fn read_i32(bytes: &[u8], index: usize) -> i32 {
    let p = index * 4;
    i32::from_le_bytes(bytes[p..p + 4].try_into().unwrap())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let k = args.next().map_or(Ok(DEFAULT_K), |v| v.parse::<usize>())?;
    let n = args.next().map_or(Ok(DEFAULT_N), |v| v.parse::<usize>())?;
    if args.next().is_some() {
        return Err("usage: int8_decode_m1 [K] [N]".into());
    }

    let device = RocketDevice::open()?;
    let mut guard = device.alloc_buffer(4096)?;
    guard.prep_relative(0)?;
    guard.as_mut_slice().fill(0);
    guard.fini()?;

    let mut regcmd = device.alloc_buffer(4096)?;
    let mut input = device.alloc_buffer(k)?;
    let mut weights = device.alloc_buffer(k * n)?;
    let mut output = device.alloc_buffer(n * 4)?;
    for bo in [&regcmd, &input, &weights, &output] {
        if bo.dma_address() >> 32 != 0 {
            return Err(format!(
                "BO IOVA 0x{:x} exceeds 32-bit regcmd window",
                bo.dma_address()
            )
            .into());
        }
    }

    let a: Vec<i8> = (0..k).map(|i| ((i * 7 + 3) % 9) as i8 - 4).collect();
    let b: Vec<i8> = (0..k * n).map(|i| ((i * 5 + 1) % 11) as i8 - 5).collect();
    let mut reference = vec![0i32; n];
    for col in 0..n {
        let mut sum = 0i32;
        for kk in 0..k {
            sum += i32::from(a[kk]) * i32::from(b[kk * n + col]);
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
    for kk in 0..k {
        for col in 0..n {
            let dst = weight_i8_fullk_index(k, n, kk, col);
            weights.as_mut_slice()[dst] = b[kk * n + col] as u8;
        }
    }
    weights.fini()?;

    output.prep_relative(0)?;
    output.as_mut_slice().fill(0xa5);
    output.fini()?;

    let ops = encode_int8_decode_m1(Int8DecodeDesc::new(
        k,
        n,
        input.dma_address(),
        weights.dma_address(),
        output.dma_address(),
    ))?;
    regcmd.prep_relative(0)?;
    regcmd.as_mut_slice().fill(0);
    for (chunk, word) in regcmd.as_mut_slice().chunks_exact_mut(8).zip(ops.iter()) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    regcmd.fini()?;

    let task = Task {
        regcmd: u32::try_from(regcmd.dma_address())?,
        regcmd_count: u32::try_from(ops.len())?,
    };
    let inputs = [input.handle(), weights.handle(), regcmd.handle()];
    let outputs = [output.handle()];

    // ork-driver's rknpu path resets before entering int8 and treats the first
    // execution as warm-up. Rocket owns power/reset in-kernel rather than exposing
    // RKNPU_ACT_RESET, so run the same program twice and judge only the second.
    device.submit(&[task], &inputs, &outputs)?;
    output.prep_relative(WAIT_NS)?;
    output.fini()?;

    let start = Instant::now();
    device.submit(&[task], &inputs, &outputs)?;
    output.prep_relative(WAIT_NS)?;
    let elapsed = start.elapsed();

    let mut mismatches = Vec::new();
    for (col, &expected) in reference.iter().enumerate() {
        let got = read_i32(output.as_slice(), col);
        if got != expected && mismatches.len() < 10 {
            mismatches.push((col, expected, got));
        }
    }
    output.fini()?;

    if !mismatches.is_empty() {
        return Err(format!("INT8 M=1 K={k} N={n} mismatch examples: {mismatches:?}").into());
    }

    println!(
        "INT8 DECODE PASS M=1 K={k} N={n} outputs={n} regcmd_count={} submit_wait_us={:.1}",
        ops.len(),
        elapsed.as_secs_f64() * 1e6
    );
    Ok(())
}
