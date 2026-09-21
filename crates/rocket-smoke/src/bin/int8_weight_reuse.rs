use rocket_runtime::{RocketDevice, Task};
use rocknpu_regcmd::{
    INT8_REGCMD_COUNT, Int8DecodeDesc, encode_int8_mtile, encode_int8_mtile_weight_reuse,
};
use std::time::Instant;

const K: usize = 2048;
const N: usize = 64;
const REGCMD_STRIDE: usize = 4096;
const WAIT_NS: i64 = 2_000_000_000;

fn read_i32(bytes: &[u8], index: usize) -> i32 {
    let p = index * 4;
    i32::from_le_bytes(bytes[p..p + 4].try_into().unwrap())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let tile_m = args.next().map_or(Ok(64usize), |s| s.parse::<usize>())?;
    let total_m = args.next().map_or(Ok(128usize), |s| s.parse::<usize>())?;
    let reuse = args.next().is_some_and(|s| s == "reuse");
    if !matches!(tile_m, 16 | 32 | 64 | 128)
        || total_m == 0
        || !total_m.is_multiple_of(tile_m)
        || args.next().is_some()
    {
        return Err("usage: int8_weight_reuse [16|32|64|128] [total_m] [reuse]".into());
    }
    let task_count = total_m / tile_m;

    let device = RocketDevice::open()?;
    let mut input = device.alloc_buffer(total_m * K)?;
    let mut weights = device.alloc_buffer(K * N)?;
    let mut output = device.alloc_buffer(total_m * N * 4)?;
    let mut regcmd = device.alloc_buffer(task_count * REGCMD_STRIDE)?;

    input.prep_relative(0)?;
    for row in 0..total_m {
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

    regcmd.prep_relative(0)?;
    regcmd.as_mut_slice().fill(0);

    let mut tasks = Vec::with_capacity(task_count);
    for task_index in 0..task_count {
        let input_off = task_index * tile_m * K;
        let output_off = task_index * tile_m * N * 4;
        let desc = Int8DecodeDesc::new(
            K,
            N,
            input.dma_address() + input_off as u64,
            weights.dma_address(),
            output.dma_address() + output_off as u64,
        );
        let ops = if reuse && task_index != 0 {
            encode_int8_mtile_weight_reuse(tile_m, desc)?
        } else {
            encode_int8_mtile(tile_m, desc)?
        };
        let base = task_index * REGCMD_STRIDE;
        let bytes = &mut regcmd.as_mut_slice()[base..base + INT8_REGCMD_COUNT * 8];
        for (chunk, word) in bytes.chunks_exact_mut(8).zip(ops.iter()) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        tasks.push(Task {
            regcmd: u32::try_from(regcmd.dma_address() + base as u64)?,
            regcmd_count: u32::try_from(ops.len())?,
        });
    }
    regcmd.fini()?;

    let inputs = [input.handle(), weights.handle(), regcmd.handle()];
    let outputs = [output.handle()];
    let mut timings_us = Vec::new();

    for rep in 0..7 {
        let start = Instant::now();
        device.submit(&tasks, &inputs, &outputs)?;
        output.prep_relative(WAIT_NS)?;
        let elapsed = start.elapsed().as_secs_f64() * 1e6;

        for row in 0..total_m {
            let expected = i32::try_from(K * (row % 5 + 1))?;
            for col in 0..N {
                let got = read_i32(output.as_slice(), row * N + col);
                if got != expected {
                    return Err(format!(
                        "rep={rep} row={row} col={col}: expected {expected}, got {got}"
                    )
                    .into());
                }
            }
        }
        output.fini()?;
        timings_us.push(elapsed);
    }

    timings_us.sort_by(|a, b| a.total_cmp(b));
    println!(
        "INT8 WEIGHT_REUSE PASS tile_m={tile_m} total_m={total_m} tasks={task_count} K={K} N={N} reuse={reuse} median_us={:.3} samples_us={timings_us:?}",
        timings_us[timings_us.len() / 2]
    );
    Ok(())
}
