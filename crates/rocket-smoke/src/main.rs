use half::f16;
use rocket_runtime::{RocketDevice, Task};
use rocknpu_regcmd::{Fp16MatmulDesc, encode_fp16_matmul, feature_data, weight_fp16};
use std::fs;
use std::path::PathBuf;

fn artifact_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts")
}

fn put_f16(dst: &mut [u8], index: usize, value: f16) {
    let p = index * 2;
    dst[p..p + 2].copy_from_slice(&value.to_bits().to_le_bytes());
}

fn get_f16(src: &[u8], index: usize) -> f16 {
    let p = index * 2;
    f16::from_bits(u16::from_le_bytes([src[p], src[p + 1]]))
}

fn f16_vec_bytes(v: &[f16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    out
}

fn u64_vec_bytes(v: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 8);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn run_case(
    device: &RocketDevice,
    m: usize,
    k: usize,
    n: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let tag = format!("M{m}-K{k}-N{n}");
    let mut regcmd = device.alloc_buffer(4096)?;
    let mut input = device.alloc_buffer(m * k * 2)?;
    let mut weights = device.alloc_buffer(n * k * 2)?;
    let mut output = device.alloc_buffer(m * n * 2)?;

    for bo in [&regcmd, &input, &weights, &output] {
        if bo.dma_address() >> 32 != 0 {
            return Err(format!(
                "{tag}: BO handle {} IOVA 0x{:x} exceeds 32-bit regcmd window",
                bo.handle(),
                bo.dma_address()
            )
            .into());
        }
    }

    let ops = encode_fp16_matmul(Fp16MatmulDesc::new(
        m,
        k,
        n,
        input.dma_address(),
        weights.dma_address(),
        output.dma_address(),
    ))?;
    regcmd.prep_relative(0)?;
    for (chunk, word) in regcmd.as_mut_slice().chunks_exact_mut(8).zip(ops.iter()) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    regcmd.fini()?;

    // Exact small integers keep fp16 products and the full fp32 accumulator exact,
    // so any bit mismatch is a real layout/encoder/hardware correctness failure.
    let a: Vec<f16> = (0..m * k)
        .map(|i| f16::from_f32(((i * 7 + 3) % 5) as f32 - 2.0))
        .collect();
    let b: Vec<f16> = (0..n * k)
        .map(|i| f16::from_f32(((i * 3 + 1) % 5) as f32 - 2.0))
        .collect();
    let mut cpu = vec![f16::ZERO; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[row * k + kk].to_f32() * b[col * k + kk].to_f32();
            }
            cpu[row * n + col] = f16::from_f32(sum);
        }
    }

    input.prep_relative(0)?;
    input.as_mut_slice().fill(0);
    for row in 1..=m {
        for kk in 1..=k {
            let idx = feature_data(k, m, 1, 8, kk, row, 1);
            put_f16(input.as_mut_slice(), idx, a[(row - 1) * k + (kk - 1)]);
        }
    }
    input.fini()?;

    weights.prep_relative(0)?;
    weights.as_mut_slice().fill(0);
    for col in 1..=n {
        for kk in 1..=k {
            let idx = weight_fp16(k, col, kk);
            put_f16(weights.as_mut_slice(), idx, b[(col - 1) * k + (kk - 1)]);
        }
    }
    weights.fini()?;

    output.prep_relative(0)?;
    output.as_mut_slice().fill(0xAA);
    output.fini()?;

    fs::write(
        artifact_dir().join(format!("rust-regcmd-{tag}.bin")),
        u64_vec_bytes(&ops),
    )?;
    fs::write(
        artifact_dir().join(format!("rust-input-{tag}.bin")),
        input.as_slice(),
    )?;
    fs::write(
        artifact_dir().join(format!("rust-weights-{tag}.bin")),
        weights.as_slice(),
    )?;
    fs::write(
        artifact_dir().join(format!("cpu-output-rowmajor-{tag}.bin")),
        f16_vec_bytes(&cpu),
    )?;

    let task = Task {
        regcmd: u32::try_from(regcmd.dma_address())?,
        regcmd_count: u32::try_from(ops.len())?,
    };
    let inputs = [input.handle(), weights.handle(), regcmd.handle()];
    let outputs = [output.handle()];
    device.submit(&[task], &inputs, &outputs)?;
    output.prep_relative(2_000_000_000)?;

    fs::write(
        artifact_dir().join(format!("rust-output-native-{tag}.bin")),
        output.as_slice(),
    )?;

    let mut mismatches = Vec::new();
    for row in 1..=m {
        for col in 1..=n {
            let native_idx = feature_data(n, m, 1, 8, col, row, 1);
            let actual = get_f16(output.as_slice(), native_idx);
            let expected = cpu[(row - 1) * n + (col - 1)];
            if actual.to_bits() != expected.to_bits() {
                if mismatches.len() < 10 {
                    mismatches.push(format!(
                        "row={row} col={col} expected={} actual={} expected_bits=0x{:04x} actual_bits=0x{:04x}",
                        expected.to_f32(),
                        actual.to_f32(),
                        expected.to_bits(),
                        actual.to_bits()
                    ));
                }
            }
        }
    }
    output.fini()?;

    if !mismatches.is_empty() {
        for line in &mismatches {
            eprintln!("{tag}: {line}");
        }
        return Err(format!("{tag}: NPU/CPU mismatch; first {} shown", mismatches.len()).into());
    }

    Ok(format!(
        "{tag} PASS regcmd_iova=0x{:x} input_iova=0x{:x} weights_iova=0x{:x} output_iova=0x{:x} regcmd_count={}",
        regcmd.dma_address(),
        input.dma_address(),
        weights.dma_address(),
        output.dma_address(),
        ops.len()
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(artifact_dir())?;
    let device = RocketDevice::open()?;

    // Reserve IOVA 0 for the lifetime of the full smoke run. The executable PC
    // base is a 32-bit field and our validated task stream keeps regcmd non-zero.
    let _guard = device.alloc_buffer(4096)?;

    let shapes = [
        (4, 32, 16),
        (12, 32, 16),
        (16, 64, 32),
        (32, 128, 48),
        (64, 256, 64),
        (256, 512, 128),
    ];

    let mut lines = Vec::new();
    lines.push("device=/dev/accel/accel0".to_string());
    for (m, k, n) in shapes {
        let line = run_case(&device, m, k, n)?;
        println!("{line}");
        lines.push(line);
    }
    lines.push(format!("cases={} failures=0", shapes.len()));
    fs::write(artifact_dir().join("rust-run.log"), lines.join("\n") + "\n")?;

    println!(
        "PASS: generic pure-Rust RK3588 fp16 MatMul encoder + Rocket UAPI + {} real NPU shapes + exact CPU compare",
        shapes.len()
    );
    Ok(())
}
