use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_matmul::{Fp16MatmulExecutor, cpu_reference_fp16};
use std::error::Error;

fn clock_note() -> String {
    let base = std::path::Path::new("/sys/class/devfreq/fdab0000.npu");
    if !base.exists() {
        return "clock_note=Rocket NPU devfreq unavailable (stock driver / uncontrolled clock)"
            .to_string();
    }
    let read = |name: &str| {
        std::fs::read_to_string(base.join(name))
            .map(|v| v.trim().to_string())
            .unwrap_or_else(|_| "?".to_string())
    };
    format!(
        "clock_note=devfreq cur={} target={} max={} governor={}",
        read("cur_freq"),
        read("target_freq"),
        read("max_freq"),
        read("governor")
    )
}

const M: usize = 256;
const K: usize = 512;
const N: usize = 128;
const REPS: usize = 41;

fn median(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("{}", clock_note());
    let a: Vec<f16> = (0..M * K)
        .map(|i| f16::from_f32(((i * 7 + 3) % 17) as f32 / 8.0 - 1.0))
        .collect();
    let b: Vec<f16> = (0..N * K)
        .map(|i| f16::from_f32(((i * 3 + 1) % 17) as f32 / 8.0 - 1.0))
        .collect();
    let reference = cpu_reference_fp16(&a, &b, M, K, N)?;
    let dev = RocketDevice::open()?;
    let mut ex = Fp16MatmulExecutor::new(&dev)?;
    let weights = ex.prepack_weights(&b, M, K, N)?;
    for _ in 0..5 {
        let out = ex.execute_prepacked(&a, &weights)?;
        if out.values != reference {
            return Err("frequency probe warm correctness mismatch".into());
        }
    }
    let allocs = ex.scratch_stats().bo_allocations;
    let mut total = Vec::with_capacity(REPS);
    let mut pack = Vec::with_capacity(REPS);
    let mut submit = Vec::with_capacity(REPS);
    let mut wait = Vec::with_capacity(REPS);
    let mut gather = Vec::with_capacity(REPS);
    for rep in 0..REPS {
        let out = ex.execute_prepacked(&a, &weights)?;
        if rep == REPS - 1 && out.values != reference {
            return Err("frequency probe final correctness mismatch".into());
        }
        let t = out.stats.timing;
        total.push(t.total_ns);
        pack.push(t.pack_ns);
        submit.push(t.submit_ns);
        wait.push(t.wait_ns);
        gather.push(t.gather_ns);
    }
    if ex.scratch_stats().bo_allocations != allocs {
        return Err("frequency probe allocated scratch after warmup".into());
    }
    let total = median(total);
    let pack = median(pack);
    let submit = median(submit);
    let wait = median(wait);
    let gather = median(gather);
    let flops = 2.0 * M as f64 * K as f64 * N as f64;
    println!(
        "freq_probe M{M} K{K} N{N} jobs=1 reps={REPS} total_ms={:.6} pack_ms={:.6} submit_ms={:.6} wait_ms={:.6} gather_ms={:.6} total_GFLOPs={:.2} wait_effective_GFLOPs={:.2} resident_bytes={} prepack_ms={:.3}",
        total as f64 / 1e6,
        pack as f64 / 1e6,
        submit as f64 / 1e6,
        wait as f64 / 1e6,
        gather as f64 / 1e6,
        flops / total as f64,
        flops / wait as f64,
        weights.stats().resident_bytes,
        weights.stats().pack_ns as f64 / 1e6
    );
    Ok(())
}
