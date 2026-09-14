use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_conv::{Conv2dSpec, Fp16Conv2dExecutor};
use std::{error::Error, fs, time::Instant};
fn read_f32(path: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let b = fs::read(path)?;
    Ok(b.chunks_exact(4)
        .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect())
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
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
    let mut stream = Fp16Conv2dExecutor::new(dev)?;
    let mut prep = Fp16Conv2dExecutor::new(dev)?;
    let resident = prep.prepare_weights(&weights, spec)?;
    for _ in 0..8 {
        let _ = stream.execute(&input, &weights, spec)?;
        let _ = prep.execute_prepared(&input, &resident)?;
    }
    let mut a = Vec::new();
    let mut b = Vec::new();
    for i in 0..101 {
        if i % 2 == 0 {
            let t = Instant::now();
            let _ = stream.execute(&input, &weights, spec)?;
            a.push(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            let _ = prep.execute_prepared(&input, &resident)?;
            b.push(t.elapsed().as_secs_f64() * 1e3);
        } else {
            let t = Instant::now();
            let _ = prep.execute_prepared(&input, &resident)?;
            b.push(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            let _ = stream.execute(&input, &weights, spec)?;
            a.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    let sa = median(a);
    let sb = median(b);
    println!(
        "{name} streaming_ms={sa:.4} prepared_ms={sb:.4} stream_to_prepared={:.3}x resident_bytes={} pack_ns={} streaming_scratch={:?} prepared_scratch={:?}",
        sa / sb,
        resident.stats().resident_bytes,
        resident.stats().pack_ns,
        stream.scratch_stats(),
        prep.scratch_stats()
    );
    Ok(())
}
fn main() -> Result<(), Box<dyn Error>> {
    let d = RocketDevice::open()?;
    run(
        &d,
        "conv1",
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
        &d,
        "conv2",
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
