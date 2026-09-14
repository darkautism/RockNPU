use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_conv::{Conv2dSpec, Fp16Conv2dExecutor};
use std::{error::Error, fs};

fn read_f32(path: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let b = fs::read(path)?;
    if b.len() % 4 != 0 {
        return Err("bad f32 bytes".into());
    }
    Ok(b.chunks_exact(4)
        .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect())
}

fn run(
    dev: &RocketDevice,
    name: &str,
    spec: Conv2dSpec,
    input_path: &str,
    weight_path: &str,
) -> Result<(), Box<dyn Error>> {
    let input: Vec<f16> = read_f32(input_path)?
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let weights: Vec<f16> = read_f32(weight_path)?
        .into_iter()
        .map(f16::from_f32)
        .collect();

    let mut streaming = Fp16Conv2dExecutor::new(dev)?;
    let stream = streaming.execute(&input, &weights, spec)?;

    let mut prepared_exec = Fp16Conv2dExecutor::new(dev)?;
    let prepared = prepared_exec.prepare_weights(&weights, spec)?;
    let prep_stats = prepared.stats();
    let first = prepared_exec.execute_prepared(&input, &prepared)?;
    let scratch_first = prepared_exec.scratch_stats();
    let second = prepared_exec.execute_prepared(&input, &prepared)?;
    let scratch_second = prepared_exec.scratch_stats();

    let mut stream_diff = 0usize;
    let mut repeat_diff = 0usize;
    let mut max_abs = 0.0f32;
    for i in 0..stream.len() {
        if stream[i].to_bits() != first[i].to_bits() {
            stream_diff += 1;
        }
        if first[i].to_bits() != second[i].to_bits() {
            repeat_diff += 1;
        }
        max_abs = max_abs.max((stream[i].to_f32() - first[i].to_f32()).abs());
    }
    if stream_diff != 0 || repeat_diff != 0 || max_abs != 0.0 {
        return Err(format!("{name}: prepared output differs: stream_diff={stream_diff} repeat_diff={repeat_diff} max_abs={max_abs}").into());
    }
    if scratch_first != scratch_second {
        return Err(format!(
            "{name}: prepared scratch changed on repeat: {scratch_first:?} -> {scratch_second:?}"
        )
        .into());
    }
    if scratch_first.weight_bytes != 0 || scratch_first.allocations != 3 {
        return Err(format!(
            "{name}: prepared path unexpectedly owns streaming weight scratch: {scratch_first:?}"
        )
        .into());
    }
    let expected_resident =
        ((spec.oc + 15) / 16 * 16) * ((spec.ic + 31) / 32 * 32) * spec.kh * spec.kw * 2;
    if prep_stats.resident_bytes != expected_resident {
        return Err(format!(
            "{name}: resident bytes {} != expected {expected_resident}",
            prep_stats.resident_bytes
        )
        .into());
    }
    println!(
        "{name} prepared PASS outputs={} stream_diff=0 repeat_diff=0 resident_bytes={} pack_ns={} dma=0x{:x} scratch={:?}",
        stream.len(),
        prep_stats.resident_bytes,
        prep_stats.pack_ns,
        prepared.dma_address(),
        scratch_first
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
    println!(
        "PASS: real MNIST-8 Conv resident weights are bit-exact, repeat-stable, and remove per-run weight scratch/upload"
    );
    Ok(())
}
