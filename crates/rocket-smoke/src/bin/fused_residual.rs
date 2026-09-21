use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_ops::SingleNpuBackend;
use rocknpu_tensor::Matrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let m = args.next().map_or(Ok(16usize), |v| v.parse::<usize>())?;
    let k = args.next().map_or(Ok(256usize), |v| v.parse::<usize>())?;
    let n = args.next().map_or(Ok(64usize), |v| v.parse::<usize>())?;
    if args.next().is_some() {
        return Err("usage: fused_residual [M] [K] [N]".into());
    }

    let device = RocketDevice::open()?;
    let mut backend = SingleNpuBackend::new(&device)?;

    let a: Vec<f16> = (0..m * k)
        .map(|i| f16::from_f32(((i * 7 + 3) % 17) as f32 / 32.0 - 0.25))
        .collect();
    let b: Vec<f16> = (0..n * k)
        .map(|i| f16::from_f32(((i * 11 + 5) % 19) as f32 / 48.0 - 0.20))
        .collect();
    let residual: Vec<f16> = (0..m * n)
        .map(|i| f16::from_f32(((i * 13 + 1) % 23) as f32 / 64.0 - 0.15))
        .collect();

    let b = Matrix::from_vec(n, k, b)?;
    let prepared = backend.prepare_fp16_compatible_m(&b)?;
    let a = Matrix::from_vec(m, k, a)?;
    let residual_m = Matrix::from_vec(m, n, residual.clone())?;

    let plain = backend.execute_prepared_fp16_compatible_m(&prepared, &a)?;
    let fused =
        backend.execute_prepared_fp16_compatible_m_add(&prepared, &a, &residual_m)?;

    let mut max_abs = 0.0f32;
    let mut mismatches = 0usize;
    for ((&base, &add), &got) in plain.values().iter().zip(&residual).zip(fused.values()) {
        let want = f16::from_f32(base.to_f32() + add.to_f32()).to_f32();
        let err = (got.to_f32() - want).abs();
        max_abs = max_abs.max(err);
        if err > 0.002 {
            mismatches += 1;
        }
    }
    if mismatches != 0 {
        return Err(format!(
            "FUSED RESIDUAL FAIL M={m} K={k} N={n} mismatches={mismatches} max_abs={max_abs}"
        )
        .into());
    }

    println!("FUSED RESIDUAL PASS M={m} K={k} N={n} max_abs={max_abs:.6}");
    Ok(())
}
