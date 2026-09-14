use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_matmul::{Fp16MatmulExecutor, cpu_reference_executor_semantics, cpu_reference_fp16};
use std::fs;
use std::path::PathBuf;

fn artifacts() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts")
}
fn f16_bytes(v: &[f16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for &x in v {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    out
}
fn data(m: usize, k: usize, n: usize) -> (Vec<f16>, Vec<f16>) {
    let a = (0..m * k)
        .map(|i| f16::from_f32(((i * 7 + 3) % 5) as f32 - 2.0))
        .collect();
    let b = (0..n * k)
        .map(|i| f16::from_f32(((i * 3 + 1) % 5) as f32 - 2.0))
        .collect();
    (a, b)
}

fn run_case(
    ex: &mut Fp16MatmulExecutor<'_>,
    tag: &str,
    m: usize,
    k: usize,
    n: usize,
    expect: (usize, usize, usize, usize, usize),
) -> Result<String, Box<dyn std::error::Error>> {
    let (a, b) = data(m, k, n);
    let oracle = cpu_reference_executor_semantics(&a, &b, m, k, n)?;
    let result = ex.execute(&a, &b, m, k, n)?;
    let p = &result.stats.plan;
    let actual = (
        p.m_tiles(),
        p.k_tiles(),
        p.n_tiles(),
        result.stats.npu_kacc_groups,
        result.stats.host_kacc_groups,
    );
    if actual != expect {
        return Err(
            format!("{tag}: unexpected plan/stats actual={actual:?} expected={expect:?}").into(),
        );
    }
    let mut mismatches = 0usize;
    for i in 0..oracle.len() {
        if oracle[i].to_bits() != result.values[i].to_bits() {
            if mismatches < 10 {
                eprintln!(
                    "{tag} mismatch m={} n={} expected={} got={} ebits=0x{:04x} gbits=0x{:04x}",
                    i / n,
                    i % n,
                    oracle[i].to_f32(),
                    result.values[i].to_f32(),
                    oracle[i].to_bits(),
                    result.values[i].to_bits()
                );
            }
            mismatches += 1;
        }
    }
    if mismatches != 0 {
        return Err(format!("{tag}: executor/oracle mismatches={mismatches}").into());
    }

    let math = cpu_reference_fp16(&a, &b, m, k, n)?;
    let mut math_mismatches = 0usize;
    let mut max_abs = 0.0f32;
    for i in 0..math.len() {
        if math[i].to_bits() != result.values[i].to_bits() {
            math_mismatches += 1;
        }
        max_abs = max_abs.max((math[i].to_f32() - result.values[i].to_f32()).abs());
    }

    fs::write(
        artifacts().join(format!("executor-{tag}-output.bin")),
        f16_bytes(&result.values),
    )?;
    fs::write(
        artifacts().join(format!("executor-{tag}-oracle.bin")),
        f16_bytes(&oracle),
    )?;
    let summary = format!(
        "{tag} PASS M{m} K{k} N{n} Mt={} Kt={} Nt={} mtiles={} ktiles={} ntiles={} jobs={} npu_kacc_groups={} host_kacc_groups={} oracle_mismatches=0 math_mismatches={} max_math_abs_error={}",
        p.mt,
        p.kt,
        p.nt,
        p.m_tiles(),
        p.k_tiles(),
        p.n_tiles(),
        result.stats.jobs_submitted,
        result.stats.npu_kacc_groups,
        result.stats.host_kacc_groups,
        math_mismatches,
        max_abs
    );
    println!("{summary}");
    Ok(summary)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(artifacts())?;
    let device = RocketDevice::open()?;
    let mut ex = Fp16MatmulExecutor::new(&device)?;
    let mut lines = Vec::new();
    // Two N tiles, no K split: proves N-axis packing/gather and 16-channel ragged tail.
    lines.push(run_case(&mut ex, "n-only", 64, 512, 272, (1, 1, 2, 0, 0))?);
    // M tail=44, N tail=16, K=384+128: four output groups, each NPU ping-pong KACC.
    lines.push(run_case(
        &mut ex,
        "mnk-ragged",
        300,
        512,
        272,
        (2, 2, 2, 4, 0),
    )?);
    // EW is intentionally not trusted below M=12; K partials are NPU-produced then host-f32 accumulated.
    let tiny = run_case(&mut ex, "tiny-m-host-kacc", 4, 4096, 64, (1, 2, 1, 0, 1))?;
    lines.push(tiny);

    // A repeated shape must reuse the executor-owned scratch BOs without any new
    // CREATE_BO/mmap cycle. This is the hard gate for persistent scratch reuse.
    let before = ex.scratch_stats();
    let reuse = run_case(&mut ex, "reuse-same-shape", 4, 4096, 64, (1, 2, 1, 0, 1))?;
    let after = ex.scratch_stats();
    if after.bo_allocations != before.bo_allocations {
        return Err(format!(
            "same-shape scratch reuse allocated new BOs: before={} after={}",
            before.bo_allocations, after.bo_allocations
        )
        .into());
    }
    let reuse_line = format!(
        "{reuse} scratch_reuse=PASS bo_allocations={} bo_grows={}",
        after.bo_allocations, after.bo_grows
    );
    println!("{reuse_line}");
    lines.push(reuse_line);
    fs::write(
        artifacts().join("executor-run.log"),
        lines.join("\n") + "\n",
    )?;
    println!(
        "PASS: reusable single-core fp16 executor hardware gate; cases={}",
        lines.len()
    );
    Ok(())
}
