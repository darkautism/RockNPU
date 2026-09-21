use rocket_runtime::{RocketDevice, Task};
use rocknpu_regcmd::{encode_int8_mtile, Int8DecodeDesc, INT8_REGCMD_COUNT};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

const M: usize = 16;
const K: usize = 2048;
const REPS: usize = 40;
const REGCMD_BYTES: usize = INT8_REGCMD_COUNT * 8;
const WAIT_NS: i64 = 2_000_000_000;

fn split_n(n: usize, workers: usize) -> Vec<usize> {
    let blocks = n / 32;
    let active = workers.min(blocks);
    let base = blocks / active;
    let extra = blocks % active;
    (0..active)
        .map(|i| (base + usize::from(i < extra)) * 32)
        .collect()
}

fn run(n: usize, workers: usize) -> Result<f64, Box<dyn std::error::Error>> {
    let slices = split_n(n, workers);
    let barrier = Arc::new(Barrier::new(slices.len()));
    let mut handles = Vec::new();

    for (worker, nsub) in slices.into_iter().enumerate() {
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || -> Result<f64, String> {
            let device = RocketDevice::open().map_err(|e| e.to_string())?;
            let mut input = device.alloc_buffer(M * K).map_err(|e| e.to_string())?;
            let mut weights = device.alloc_buffer(K * nsub).map_err(|e| e.to_string())?;
            let output = device.alloc_buffer(M * nsub * 4).map_err(|e| e.to_string())?;
            let mut regcmd = device.alloc_buffer(REGCMD_BYTES).map_err(|e| e.to_string())?;

            input.prep_relative(0).map_err(|e| e.to_string())?;
            for row in 0..M {
                input.as_mut_slice()[row*K..(row+1)*K].fill((row % 5 + 1) as u8);
            }
            input.fini().map_err(|e| e.to_string())?;
            weights.prep_relative(0).map_err(|e| e.to_string())?;
            weights.as_mut_slice().fill(1);
            weights.fini().map_err(|e| e.to_string())?;

            let ops = encode_int8_mtile(M, Int8DecodeDesc::new(
                K, nsub, input.dma_address(), weights.dma_address(), output.dma_address()
            )).map_err(|e| e.to_string())?;
            regcmd.prep_relative(0).map_err(|e| e.to_string())?;
            regcmd.as_mut_slice().fill(0);
            for (chunk, word) in regcmd.as_mut_slice().chunks_exact_mut(8).zip(ops.iter()) {
                chunk.copy_from_slice(&word.to_le_bytes());
            }
            regcmd.fini().map_err(|e| e.to_string())?;

            let task = Task {
                regcmd: u32::try_from(regcmd.dma_address()).map_err(|e| e.to_string())?,
                regcmd_count: u32::try_from(ops.len()).map_err(|e| e.to_string())?,
            };
            let inputs = [input.handle(), weights.handle(), regcmd.handle()];
            let outputs = [output.handle()];

            // Per-core/output-buffer warmup, mirroring the empirical stale-first-write guard.
            for _ in 0..2 {
                device.submit(&[task], &inputs, &outputs).map_err(|e| e.to_string())?;
                output.prep_relative(WAIT_NS).map_err(|e| e.to_string())?;
                output.fini().map_err(|e| e.to_string())?;
            }

            barrier.wait();
            let start = Instant::now();
            for _ in 0..REPS {
                device.submit(&[task], &inputs, &outputs).map_err(|e| e.to_string())?;
                output.prep_relative(WAIT_NS).map_err(|e| e.to_string())?;
                output.fini().map_err(|e| e.to_string())?;
            }
            let elapsed = start.elapsed().as_secs_f64();

            output.prep_relative(WAIT_NS).map_err(|e| e.to_string())?;
            let bytes = output.as_slice();
            for row in 0..M {
                let expected = (K * (row % 5 + 1)) as i32;
                for col in 0..nsub {
                    let p = (row*nsub + col)*4;
                    let got = i32::from_le_bytes(bytes[p..p+4].try_into().unwrap());
                    if got != expected {
                        return Err(format!("worker={worker} row={row} col={col} got={got} expected={expected}"));
                    }
                }
            }
            output.fini().map_err(|e| e.to_string())?;
            Ok(elapsed)
        }));
    }

    let mut wall = 0.0f64;
    for h in handles {
        let elapsed = h.join().map_err(|_| "worker panic")??;
        wall = wall.max(elapsed);
    }
    Ok(REPS as f64 / wall)
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a,b| a.partial_cmp(b).unwrap());
    xs[xs.len()/2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for n in [256usize, 2048, 5632] {
        let mut samples = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for workers in [3usize,1,2,1,3,2,1,2,3] {
            let cps = run(n, workers)?;
            println!("sample M={M} K={K} N={n} workers={workers} full_projection_calls_s={cps:.3}");
            samples[workers].push(cps);
        }
        let s1 = median(samples[1].clone());
        let s2 = median(samples[2].clone());
        let s3 = median(samples[3].clone());
        println!("SUMMARY M={M} K={K} N={n} calls_s=[{s1:.3},{s2:.3},{s3:.3}] scaling=[{:.3}x,{:.3}x]", s2/s1, s3/s1);
    }
    Ok(())
}
