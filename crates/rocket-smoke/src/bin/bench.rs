use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_matmul::{ExecutionTiming, Fp16MatmulExecutor};
use std::fs;
use std::path::PathBuf;

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

const REPS: usize = 5;

fn artifacts() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts")
}

fn data(m: usize, k: usize, n: usize) -> (Vec<f16>, Vec<f16>) {
    let a = (0..m * k)
        .map(|i| f16::from_f32((((i * 1103515245usize + 12345) >> 11) % 257) as f32 / 64.0 - 2.0))
        .collect();
    let b = (0..n * k)
        .map(|i| f16::from_f32((((i * 1664525usize + 1013904223) >> 9) % 257) as f32 / 64.0 - 2.0))
        .collect();
    (a, b)
}

fn median(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn packed_bytes(plan: &rocknpu_regcmd::Fp16MatmulPlan) -> usize {
    plan.tiles.iter().map(|t| (t.m * t.k + t.n * t.k) * 2).sum()
}

fn fmt_ms(ns: u128) -> f64 {
    ns as f64 / 1_000_000.0
}

fn print_best_phases(t: ExecutionTiming, pack_bytes: usize, ops: f64) -> String {
    let pack_gbps = if t.pack_ns == 0 {
        0.0
    } else {
        pack_bytes as f64 / t.pack_ns as f64
    };
    let wait_gflops = if t.wait_ns == 0 {
        0.0
    } else {
        ops / t.wait_ns as f64
    };
    format!(
        "phases_ms plan={:.3} scratch={:.3} pack={:.3} encode={:.3} regcmd={:.3} submit={:.3} wait={:.3} gather={:.3} unaccounted={:.3} pack_GBps={:.2} wait_effective_GFLOPs={:.2}",
        fmt_ms(t.plan_ns),
        fmt_ms(t.scratch_ns),
        fmt_ms(t.pack_ns),
        fmt_ms(t.encode_ns),
        fmt_ms(t.regcmd_write_ns),
        fmt_ms(t.submit_ns),
        fmt_ms(t.wait_ns),
        fmt_ms(t.gather_ns),
        fmt_ms(t.total_ns.saturating_sub(
            t.plan_ns
                + t.scratch_ns
                + t.pack_ns
                + t.encode_ns
                + t.regcmd_write_ns
                + t.submit_ns
                + t.wait_ns
                + t.gather_ns
        )),
        pack_gbps,
        wait_gflops
    )
}

fn bench_fp16(
    ex: &mut Fp16MatmulExecutor<'_>,
    tag: &str,
    m: usize,
    k: usize,
    n: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let (a, b) = data(m, k, n);
    let warm = ex.execute(&a, &b, m, k, n)?;
    let warm_bits: Vec<u16> = warm.values.iter().map(|v| v.to_bits()).collect();
    let alloc_after_warm = ex.scratch_stats().bo_allocations;
    let pack_bytes = packed_bytes(&warm.stats.plan);
    let ops = 2.0 * m as f64 * n as f64 * k as f64;
    let mut totals = Vec::with_capacity(REPS);
    let mut best = None::<ExecutionTiming>;

    for _ in 0..REPS {
        let out = ex.execute(&a, &b, m, k, n)?;
        if out
            .values
            .iter()
            .map(|v| v.to_bits())
            .ne(warm_bits.iter().copied())
        {
            return Err(format!("{tag}: repeated fp16 output changed").into());
        }
        totals.push(out.stats.timing.total_ns);
        if best.is_none_or(|b| out.stats.timing.total_ns < b.total_ns) {
            best = Some(out.stats.timing);
        }
    }
    let stats = ex.scratch_stats();
    if stats.bo_allocations != alloc_after_warm {
        return Err(format!(
            "{tag}: warmed benchmark allocated new BOs {} -> {}",
            alloc_after_warm, stats.bo_allocations
        )
        .into());
    }
    let best = best.unwrap();
    let med = median(totals);
    let gflops_best = ops / best.total_ns as f64;
    let gflops_med = ops / med as f64;
    Ok(format!(
        "{tag} fp16 M{m} K{k} N{n} jobs={} Mt={} Kt={} Nt={} reps={REPS} median_ms={:.3} best_ms={:.3} median_GFLOPs={:.2} best_GFLOPs={:.2} scratch_allocs={} scratch_grows={} packed_MB_per_run={:.3}\n  {}",
        warm.stats.jobs_submitted,
        warm.stats.plan.mt,
        warm.stats.plan.kt,
        warm.stats.plan.nt,
        fmt_ms(med),
        fmt_ms(best.total_ns),
        gflops_med,
        gflops_best,
        stats.bo_allocations,
        stats.bo_grows,
        pack_bytes as f64 / 1_000_000.0,
        print_best_phases(best, pack_bytes, ops)
    ))
}

fn bench_fp32(
    ex: &mut Fp16MatmulExecutor<'_>,
    tag: &str,
    m: usize,
    k: usize,
    n: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let (a, b) = data(m, k, n);
    let warm = ex.execute_f32(&a, &b, m, k, n)?;
    let warm_bits: Vec<u32> = warm.values.iter().map(|v| v.to_bits()).collect();
    let alloc_after_warm = ex.scratch_stats().bo_allocations;
    let pack_bytes = packed_bytes(&warm.stats.plan);
    let ops = 2.0 * m as f64 * n as f64 * k as f64;
    let mut totals = Vec::with_capacity(REPS);
    let mut best = None::<ExecutionTiming>;
    for _ in 0..REPS {
        let out = ex.execute_f32(&a, &b, m, k, n)?;
        if out
            .values
            .iter()
            .map(|v| v.to_bits())
            .ne(warm_bits.iter().copied())
        {
            return Err(format!("{tag}: repeated fp32 output changed").into());
        }
        totals.push(out.stats.timing.total_ns);
        if best.is_none_or(|b| out.stats.timing.total_ns < b.total_ns) {
            best = Some(out.stats.timing);
        }
    }
    let stats = ex.scratch_stats();
    if stats.bo_allocations != alloc_after_warm {
        return Err(format!("{tag}: warmed fp32 benchmark allocated new BOs").into());
    }
    let best = best.unwrap();
    let med = median(totals);
    Ok(format!(
        "{tag} fp32out M{m} K{k} N{n} jobs={} reps={REPS} median_ms={:.3} best_ms={:.3} median_GFLOPs={:.2} best_GFLOPs={:.2}\n  {}",
        warm.stats.jobs_submitted,
        fmt_ms(med),
        fmt_ms(best.total_ns),
        ops / med as f64,
        ops / best.total_ns as f64,
        print_best_phases(best, pack_bytes, ops)
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(artifacts())?;
    let dev = RocketDevice::open()?;
    let mut ex = Fp16MatmulExecutor::new(&dev)?;
    let mut lines = Vec::new();
    lines.push(clock_note());
    lines.push(bench_fp16(&mut ex, "single-ish", 256, 512, 128)?);
    lines.push(bench_fp16(&mut ex, "deep-k", 64, 4096, 128)?);
    lines.push(bench_fp16(&mut ex, "prefill-ish", 64, 4096, 512)?);
    lines.push(bench_fp16(&mut ex, "wide-tiled", 256, 1024, 256)?);
    lines.push(bench_fp32(&mut ex, "deep-k-accuracy", 64, 4096, 128)?);
    for line in &lines {
        println!("{line}");
    }
    fs::write(
        artifacts().join("executor-bench.txt"),
        lines.join("\n") + "\n",
    )?;
    println!(
        "PASS: warmed release-style executor benchmark; cases=5; no post-warm scratch allocations"
    );
    Ok(())
}
