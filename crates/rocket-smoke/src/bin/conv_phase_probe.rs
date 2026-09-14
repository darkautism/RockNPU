use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_conv::{Conv2dSpec, ConvExecutionTiming, Fp16Conv2dExecutor};
use std::{error::Error, fs};

fn read_f32(path: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let b = fs::read(path)?;
    Ok(b.chunks_exact(4)
        .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect())
}
fn median(v: &mut [u128]) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}
fn ms(v: u128) -> f64 {
    v as f64 / 1e6
}
fn run(
    dev: &RocketDevice,
    name: &str,
    spec: Conv2dSpec,
    ip: &str,
    wp: &str,
) -> Result<(), Box<dyn Error>> {
    let input: Vec<f16> = read_f32(ip)?.into_iter().map(f16::from_f32).collect();
    let weights: Vec<f16> = read_f32(wp)?.into_iter().map(f16::from_f32).collect();
    let mut ex = Fp16Conv2dExecutor::new(dev)?;
    let resident = ex.prepare_weights(&weights, spec)?;
    let baseline = ex.execute_prepared(&input, &resident)?;
    for _ in 0..8 {
        let _ = ex.execute_prepared(&input, &resident)?;
    }
    let mut timings = Vec::<ConvExecutionTiming>::with_capacity(101);
    for _ in 0..101 {
        let (out, t) = ex.execute_prepared_profiled(&input, &resident)?;
        if out
            .iter()
            .zip(&baseline)
            .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err(format!("{name} profiled output mismatch").into());
        }
        timings.push(t);
    }
    macro_rules! field {
        ($f:ident) => {{
            let mut v: Vec<u128> = timings.iter().map(|x| x.$f).collect();
            median(&mut v)
        }};
    }
    let total = field!(total_ns);
    let phases = [
        ("scratch", field!(scratch_ns)),
        ("input_pack", field!(input_pack_ns)),
        ("output_clear", field!(output_clear_ns)),
        ("encode", field!(encode_ns)),
        ("regcmd_write", field!(regcmd_write_ns)),
        ("submit", field!(submit_ns)),
        ("wait", field!(wait_ns)),
        ("gather", field!(gather_ns)),
    ];
    let accounted: u128 = phases.iter().map(|x| x.1).sum();
    println!(
        "{name} total_ms={:.4} jobs={} resident_bytes={} scratch={:?}",
        ms(total),
        timings[timings.len() / 2].jobs_submitted,
        resident.stats().resident_bytes,
        ex.scratch_stats()
    );
    for (n, v) in phases {
        println!(
            "  {n:13} ms={:.4} pct={:.1}%",
            ms(v),
            100.0 * v as f64 / total as f64
        );
    }
    println!(
        "  {:13} ms={:.4} pct={:.1}%",
        "unaccounted",
        ms(total.saturating_sub(accounted)),
        100.0 * total.saturating_sub(accounted) as f64 / total as f64
    );
    Ok(())
}
fn main() -> Result<(), Box<dyn Error>> {
    let dev = RocketDevice::open()?;
    run(
        &dev,
        "mnist8-conv1",
        Conv2dSpec {
            ic: 1,
            ih: 28,
            iw: 28,
            oc: 8,
            kh: 5,
            kw: 5,
            pad_top: 2,
            pad_left: 2,
            stride_y: 1,
            stride_x: 1,
        },
        "artifacts/mnist8-conv1-input-f32.bin",
        "artifacts/mnist8-conv1-weight-f32.bin",
    )?;
    run(
        &dev,
        "mnist8-conv2",
        Conv2dSpec {
            ic: 8,
            ih: 14,
            iw: 14,
            oc: 16,
            kh: 5,
            kw: 5,
            pad_top: 2,
            pad_left: 2,
            stride_y: 1,
            stride_x: 1,
        },
        "artifacts/mnist8-conv2-input-f32.bin",
        "artifacts/mnist8-conv2-weight-f32.bin",
    )?;
    Ok(())
}
