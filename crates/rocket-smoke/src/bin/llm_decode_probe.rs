use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_ops::{
    ExecutionTarget, MatmulOutput, MatmulPrecision, MatmulSpec, SingleNpuBackend, execute_cpu,
};
use rocknpu_tensor::Matrix;
use std::time::Instant;

const REPS: usize = 7;

fn median(mut values: Vec<u128>) -> u128 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn ms(ns: u128) -> f64 {
    ns as f64 / 1_000_000.0
}

fn make_input(k: usize) -> Matrix<f16> {
    Matrix::from_vec(
        1,
        k,
        (0..k)
            .map(|i| f16::from_f32(((i * 17 + 11) % 37) as f32 / 19.0 - 0.9))
            .collect(),
    )
    .unwrap()
}

fn make_weights(n: usize, k: usize) -> Matrix<f16> {
    Matrix::from_vec(
        n,
        k,
        (0..n * k)
            .map(|i| f16::from_f32(((i * 13 + 7) % 29) as f32 / 17.0 - 0.8))
            .collect(),
    )
    .unwrap()
}

fn first_row_close(cpu: &[f16], padded_npu: &[f16], n: usize) -> bool {
    cpu.iter().zip(&padded_npu[..n]).all(|(&a, &b)| {
        let a = a.to_f32();
        let b = b.to_f32();
        let tol = 0.03f32.max(a.abs() * 0.003);
        (a - b).abs() <= tol
    })
}

fn probe(
    backend: &mut SingleNpuBackend<'_>,
    tag: &str,
    k: usize,
    n: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let a1 = make_input(k);
    let weights = make_weights(n, k);
    let cpu_spec = MatmulSpec::new(1, k, n, MatmulPrecision::Fp16Fast, ExecutionTarget::Cpu);

    let cpu_warm = match execute_cpu(cpu_spec, &a1, &weights)? {
        MatmulOutput::F16(values) => values,
        MatmulOutput::F32(_) => unreachable!("FP16-fast CPU returned FP32"),
    };
    let mut cpu_times = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let start = Instant::now();
        let out = execute_cpu(cpu_spec, &a1, &weights)?;
        cpu_times.push(start.elapsed().as_nanos());
        if !matches!(out, MatmulOutput::F16(_)) {
            return Err("FP16-fast CPU returned FP32".into());
        }
    }

    let prepare_start = Instant::now();
    let prepared = backend.prepare_fp16_compatible_m(&weights)?;
    let prepare_ns = prepare_start.elapsed().as_nanos();
    let weight_stats = prepared.weight_stats();

    let mut padded_values = vec![f16::ZERO; 4 * k];
    padded_values[..k].copy_from_slice(a1.values());
    let padded = Matrix::from_vec(4, k, padded_values)?;
    let npu_warm = backend.execute_prepared_fp16_compatible_m(&prepared, &padded)?;
    if !first_row_close(cpu_warm.values(), npu_warm.values(), n) {
        return Err(
            format!("{tag}: padded NPU first row differs from CPU beyond tolerance").into(),
        );
    }

    let mut npu_times = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let start = Instant::now();
        let out = backend.execute_prepared_fp16_compatible_m(&prepared, &padded)?;
        npu_times.push(start.elapsed().as_nanos());
        if !first_row_close(cpu_warm.values(), out.values(), n) {
            return Err(format!("{tag}: repeated padded NPU output changed").into());
        }
    }

    let cpu_med = median(cpu_times);
    let npu_med = median(npu_times);
    println!(
        "{tag} K={k} N={n} cpu_m1_ms={:.3} padded_npu_m4_ms={:.3} speedup={:.2}x prepare_ms={:.3} resident_mb={:.2} pack_ms={:.3}",
        ms(cpu_med),
        ms(npu_med),
        cpu_med as f64 / npu_med as f64,
        ms(prepare_ns),
        weight_stats.resident_bytes as f64 / (1024.0 * 1024.0),
        ms(weight_stats.pack_ns),
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = RocketDevice::open()?;
    let mut backend = SingleNpuBackend::new(&device)?;
    probe(&mut backend, "q_proj/o_proj", 2048, 2048)?;
    probe(&mut backend, "k_proj/v_proj", 2048, 256)?;
    probe(&mut backend, "gate/up", 2048, 5632)?;
    probe(&mut backend, "down", 5632, 2048)?;
    println!("PASS: TinyLlama decode projection CPU M=1 vs resident padded NPU M=4 probe");
    Ok(())
}
