use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_matmul::{Fp16MatmulExecutor, cpu_reference_fp32};
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
        let unit = ((s >> 32) as u32) as f32 / (u32::MAX as f32);
        f16::from_f32((unit * 2.0 - 1.0) * 4.0)
    }
}
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(v.len() * 4);
    for &x in v {
        o.extend_from_slice(&x.to_le_bytes());
    }
    o
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const M: usize = 64;
    const K: usize = 4096;
    const N: usize = 128;
    fs::create_dir_all(artifacts())?;
    let mut next = lcg(0x3588_f32);
    let a: Vec<f16> = (0..M * K).map(|_| next()).collect();
    let b: Vec<f16> = (0..N * K).map(|_| next()).collect();
    let reference = cpu_reference_fp32(&a, &b, M, K, N)?;

    let dev = RocketDevice::open()?;
    let mut ex = Fp16MatmulExecutor::new(&dev)?;
    let out16 = ex.execute(&a, &b, M, K, N)?;
    let out32 = ex.execute_f32(&a, &b, M, K, N)?;
    if out32.stats.plan.k_tiles() < 2 {
        return Err("fp32 accuracy gate must exercise multiple K tiles".into());
    }

    let mut max_ref = 0.0f64;
    let mut max16 = 0.0f64;
    let mut max32 = 0.0f64;
    let mut nonfinite = 0usize;
    for i in 0..reference.len() {
        let r = reference[i] as f64;
        let v16 = out16.values[i].to_f32() as f64;
        let v32 = out32.values[i] as f64;
        if !v16.is_finite() || !v32.is_finite() {
            nonfinite += 1;
            continue;
        }
        max_ref = max_ref.max(r.abs());
        max16 = max16.max((v16 - r).abs());
        max32 = max32.max((v32 - r).abs());
    }
    let norm16 = if max_ref > 0.0 {
        max16 / max_ref
    } else {
        max16
    };
    let norm32 = if max_ref > 0.0 {
        max32 / max_ref
    } else {
        max32
    };
    let improve = if norm32 > 0.0 {
        norm16 / norm32
    } else if norm16 > 0.0 {
        1.0e9
    } else {
        1.0
    };
    let p = &out32.stats.plan;
    let summary = format!(
        "fp32out M{M} K{K} N{N} Mt={} Kt={} Nt={} ktiles={} jobs16={} jobs32={} max_ref={:.6} max_abs16={:.6} max_abs32={:.6} norm16={:.8} norm32={:.8} improve={:.2}x nonfinite={}",
        p.mt,
        p.kt,
        p.nt,
        p.k_tiles(),
        out16.stats.jobs_submitted,
        out32.stats.jobs_submitted,
        max_ref,
        max16,
        max32,
        norm16,
        norm32,
        improve,
        nonfinite
    );
    println!("{summary}");
    fs::write(
        artifacts().join("fp32out-reference.bin"),
        f32_bytes(&reference),
    )?;
    fs::write(
        artifacts().join("fp32out-output.bin"),
        f32_bytes(&out32.values),
    )?;
    fs::write(artifacts().join("fp32out-run.log"), format!("{summary}\n"))?;
    if nonfinite != 0 {
        return Err(format!("nonfinite outputs={nonfinite}").into());
    }
    if norm32 >= 2.0e-4 {
        return Err(format!("fp32-output norm error {norm32} exceeds 2e-4").into());
    }
    if improve < 5.0 {
        return Err(format!("fp32-output improvement {improve}x is below 5x").into());
    }
    println!(
        "PASS: fp32-output executor tracks fp64 reference and improves over fp16-output by {improve:.2}x"
    );
    Ok(())
}
