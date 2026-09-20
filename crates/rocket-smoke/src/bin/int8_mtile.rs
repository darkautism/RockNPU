use rocket_runtime::{RocketDevice, Task};
use rocknpu_regcmd::{Int8DecodeDesc, encode_int8_mtile};
use std::time::Instant;

const K: usize = 2048;
const N: usize = 2048;
const REGCMD_BYTES: usize = 4096;
const WAIT_NS: i64 = 2_000_000_000;

fn read_i32(bytes: &[u8], index: usize) -> i32 {
    let p = index * 4;
    i32::from_le_bytes(bytes[p..p + 4].try_into().unwrap())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let m = args.next().map_or(Ok(16usize), |s| s.parse::<usize>())?;
    if m != 16 || args.next().is_some() {
        return Err("usage: int8_mtile [16]".into());
    }

    let device = RocketDevice::open()?;
    let mut input = device.alloc_buffer(m * K)?;
    let mut weights = device.alloc_buffer(K * N)?;
    let mut output = device.alloc_buffer(m * N * 4)?;
    let mut regcmd = device.alloc_buffer(REGCMD_BYTES)?;

    input.prep_relative(0)?;
    for row in 0..m {
        let value = i8::try_from(row % 5 + 1)? as u8;
        input.as_mut_slice()[row * K..(row + 1) * K].fill(value);
    }
    input.fini()?;

    weights.prep_relative(0)?;
    weights.as_mut_slice().fill(1);
    weights.fini()?;

    output.prep_relative(0)?;
    output.as_mut_slice().fill(0xa5);
    output.fini()?;

    let desc = Int8DecodeDesc::new(
        K,
        N,
        input.dma_address(),
        weights.dma_address(),
        output.dma_address(),
    );
    let ops = encode_int8_mtile(m, desc)?;
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
    let mut timings_us = Vec::new();

    for rep in 0..5 {
        let start = Instant::now();
        device.submit(&[task], &inputs, &outputs)?;
        output.prep_relative(WAIT_NS)?;
        let elapsed = start.elapsed().as_secs_f64() * 1e6;
        for row in 0..m {
            let expected = i32::try_from(K * (row % 5 + 1))?;
            for col in 0..N {
                let got = read_i32(output.as_slice(), row * N + col);
                if got != expected {
                    return Err(format!(
                        "rep={rep} row={row} col={col}: expected {expected}, got {got}"
                    ).into());
                }
            }
        }
        output.fini()?;
        timings_us.push(elapsed);
    }

    println!("INT8 MTILE PASS M={m} K={K} N={N} submit_wait_us={timings_us:?}");
    Ok(())
}
