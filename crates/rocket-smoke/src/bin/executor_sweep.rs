use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_matmul::{Fp16MatmulExecutor, cpu_reference_executor_semantics};
use std::fs;
use std::path::PathBuf;

fn artifacts() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts")
}
fn lcg(mut s: u64) -> impl FnMut() -> f16 {
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Small integers keep the CPU f32 product/sum exact while randomized ordering
        // stresses packing, tails and K-split grouping far better than periodic fixtures.
        f16::from_f32(((s >> 32) % 5) as f32 - 2.0)
    }
}
fn data(m: usize, k: usize, n: usize, seed: u64) -> (Vec<f16>, Vec<f16>) {
    let mut next = lcg(seed);
    let a = (0..m * k).map(|_| next()).collect();
    let b = (0..n * k).map(|_| next()).collect();
    (a, b)
}
fn run_case(
    ex: &mut Fp16MatmulExecutor<'_>,
    tag: &str,
    m: usize,
    k: usize,
    n: usize,
    seed: u64,
    expected_modes: (usize, usize),
) -> Result<String, Box<dyn std::error::Error>> {
    let (a, b) = data(m, k, n, seed);
    let oracle = cpu_reference_executor_semantics(&a, &b, m, k, n)?;
    let got = ex.execute(&a, &b, m, k, n)?;
    if (got.stats.npu_kacc_groups, got.stats.host_kacc_groups) != expected_modes {
        return Err(format!(
            "{tag}: KACC mode counts {:?} != {:?}",
            (got.stats.npu_kacc_groups, got.stats.host_kacc_groups),
            expected_modes
        )
        .into());
    }
    let mut bad = 0usize;
    for i in 0..oracle.len() {
        if oracle[i].to_bits() != got.values[i].to_bits() {
            if bad < 8 {
                eprintln!(
                    "{tag}: mismatch m={} n={} expected={} got={} ebits=0x{:04x} gbits=0x{:04x}",
                    i / n,
                    i % n,
                    oracle[i].to_f32(),
                    got.values[i].to_f32(),
                    oracle[i].to_bits(),
                    got.values[i].to_bits()
                );
            }
            bad += 1;
        }
    }
    if bad != 0 {
        return Err(format!("{tag}: exact differential mismatches={bad}").into());
    }
    let p = &got.stats.plan;
    let line = format!(
        "{tag} seed=0x{seed:x} PASS M{m} K{k} N{n} Mt={} Kt={} Nt={} tiles={} jobs={} npu_kacc={} host_kacc={} mismatches=0",
        p.mt,
        p.kt,
        p.nt,
        p.tiles.len(),
        got.stats.jobs_submitted,
        got.stats.npu_kacc_groups,
        got.stats.host_kacc_groups
    );
    println!("{line}");
    Ok(line)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(artifacts())?;
    let dev = RocketDevice::open()?;
    let mut ex = Fp16MatmulExecutor::new(&dev)?;
    let mut log = Vec::new();
    for seed in [0x3588u64, 0xc0ffeeu64] {
        log.push(run_case(&mut ex, "boundary", 12, 96, 32, seed, (0, 0))?);
        log.push(run_case(
            &mut ex,
            "n-tail",
            68,
            224,
            272,
            seed ^ 0x1111,
            (0, 0),
        )?);
    }
    // mt=256, nt=256, kt=384 => 2 M x 2 N x 3 K = 12 jobs.
    // The M=256 groups use NPU EW accumulation; the M=4 tail groups use host f32.
    log.push(run_case(
        &mut ex,
        "mixed-kacc-tail",
        260,
        1024,
        272,
        0x5eed5eed,
        (2, 2),
    )?);
    fs::write(
        artifacts().join("executor-sweep.log"),
        log.join("\n") + "\n",
    )?;
    println!(
        "PASS: executor deterministic randomized hardware differential sweep; cases={}",
        log.len()
    );
    Ok(())
}
