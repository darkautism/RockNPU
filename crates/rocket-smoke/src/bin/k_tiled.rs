use half::f16;
use rocket_runtime::{RocketDevice, Task};
use rocknpu_regcmd::{
    Fp16MatmulDesc, encode_fp16_matmul, feature_data, plan_fp16_matmul, weight_fp16,
};
use std::fs;
use std::path::PathBuf;

fn artifacts() -> PathBuf {
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

fn words_bytes(words: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 8);
    for &w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const M: usize = 64;
    const K: usize = 4096;
    const N: usize = 64;

    fs::create_dir_all(artifacts())?;
    let plan = plan_fp16_matmul(M, K, N)?;
    if plan.m_tiles() != 1 || plan.n_tiles() != 1 || plan.k_tiles() <= 1 {
        return Err(format!(
            "unexpected K-tiled gate plan: Mt={} Kt={} Nt={} mtiles={} ktiles={} ntiles={} tasks={}",
            plan.mt,
            plan.kt,
            plan.nt,
            plan.m_tiles(),
            plan.k_tiles(),
            plan.n_tiles(),
            plan.tiles.len()
        )
        .into());
    }

    let plan_text = format!(
        "shape=M{M} K{K} N{N}\nMt={} Kt={} Nt={}\nm_tiles={} k_tiles={} n_tiles={} tasks={}\n",
        plan.mt,
        plan.kt,
        plan.nt,
        plan.m_tiles(),
        plan.k_tiles(),
        plan.n_tiles(),
        plan.tiles.len()
    );
    fs::write(artifacts().join("k-tiled-plan.txt"), &plan_text)?;
    print!("{plan_text}");

    let a: Vec<f16> = (0..M * K)
        .map(|i| f16::from_f32(((i * 7 + 3) % 5) as f32 - 2.0))
        .collect();
    let b: Vec<f16> = (0..N * K)
        .map(|i| f16::from_f32(((i * 3 + 1) % 5) as f32 - 2.0))
        .collect();

    let device = RocketDevice::open()?;
    let _guard = device.alloc_buffer(4096)?;
    let mut npu_acc = vec![0.0f32; M * N];
    let mut cpu_tiled_acc = vec![0.0f32; M * N];
    let mut full_cpu = vec![0.0f32; M * N];
    let mut log = vec![plan_text.trim_end().to_string()];

    for m in 0..M {
        for n in 0..N {
            let mut acc = 0.0f32;
            for k in 0..K {
                acc += a[m * K + k].to_f32() * b[n * K + k].to_f32();
            }
            full_cpu[m * N + n] = acc;
        }
    }

    for (ti, tile) in plan.tiles.iter().enumerate() {
        let mut regcmd = device.alloc_buffer(4096)?;
        let mut input = device.alloc_buffer(tile.m * tile.k * 2)?;
        let mut weights = device.alloc_buffer(tile.n * tile.k * 2)?;
        let mut output = device.alloc_buffer(tile.m * tile.n * 2)?;

        let ops = encode_fp16_matmul(Fp16MatmulDesc::new(
            tile.m,
            tile.k,
            tile.n,
            input.dma_address(),
            weights.dma_address(),
            output.dma_address(),
        ))?;

        regcmd.prep_relative(0)?;
        for (chunk, word) in regcmd.as_mut_slice().chunks_exact_mut(8).zip(ops.iter()) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        regcmd.fini()?;

        input.prep_relative(0)?;
        input.as_mut_slice().fill(0);
        for m in 0..tile.m {
            for k in 0..tile.k {
                let dst = feature_data(tile.k, tile.m, 1, 8, k + 1, m + 1, 1);
                let src = (tile.m0 + m) * K + tile.k0 + k;
                put_f16(input.as_mut_slice(), dst, a[src]);
            }
        }
        input.fini()?;

        weights.prep_relative(0)?;
        weights.as_mut_slice().fill(0);
        for n in 0..tile.n {
            for k in 0..tile.k {
                let dst = weight_fp16(tile.k, n + 1, k + 1);
                let src = (tile.n0 + n) * K + tile.k0 + k;
                put_f16(weights.as_mut_slice(), dst, b[src]);
            }
        }
        weights.fini()?;

        output.prep_relative(0)?;
        output.as_mut_slice().fill(0xAA);
        output.fini()?;

        fs::write(
            artifacts().join(format!("k-tiled-regcmd-{ti}.bin")),
            words_bytes(&ops),
        )?;

        let task = Task {
            regcmd: u32::try_from(regcmd.dma_address())?,
            regcmd_count: u32::try_from(ops.len())?,
        };
        device.submit(
            &[task],
            &[input.handle(), weights.handle(), regcmd.handle()],
            &[output.handle()],
        )?;
        output.prep_relative(2_000_000_000)?;
        fs::write(
            artifacts().join(format!("k-tiled-output-native-{ti}.bin")),
            output.as_slice(),
        )?;

        let mut partial_mismatches = 0usize;
        for m in 0..tile.m {
            for n in 0..tile.n {
                let mut cpu_partial_f32 = 0.0f32;
                for k in 0..tile.k {
                    cpu_partial_f32 += a[(tile.m0 + m) * K + tile.k0 + k].to_f32()
                        * b[(tile.n0 + n) * K + tile.k0 + k].to_f32();
                }
                let cpu_partial = f16::from_f32(cpu_partial_f32);
                let src = feature_data(tile.n, tile.m, 1, 8, n + 1, m + 1, 1);
                let npu_partial = get_f16(output.as_slice(), src);
                if npu_partial.to_bits() != cpu_partial.to_bits() {
                    if partial_mismatches < 10 {
                        eprintln!(
                            "tile={ti} partial mismatch m={} n={} expected={} actual={} ebits=0x{:04x} abits=0x{:04x}",
                            m,
                            n,
                            cpu_partial.to_f32(),
                            npu_partial.to_f32(),
                            cpu_partial.to_bits(),
                            npu_partial.to_bits()
                        );
                    }
                    partial_mismatches += 1;
                }
                let idx = (tile.m0 + m) * N + tile.n0 + n;
                npu_acc[idx] += npu_partial.to_f32();
                cpu_tiled_acc[idx] += cpu_partial.to_f32();
            }
        }
        output.fini()?;
        if partial_mismatches != 0 {
            return Err(format!("tile {ti}: partial mismatches={partial_mismatches}").into());
        }

        let line = format!(
            "tile={ti} k0={} K{} partials_exact=PASS regcmd_iova=0x{:x}",
            tile.k0,
            tile.k,
            regcmd.dma_address()
        );
        println!("{line}");
        log.push(line);
    }

    let mut acc_mismatches = 0usize;
    let mut max_full_error = 0.0f32;
    for i in 0..npu_acc.len() {
        if npu_acc[i].to_bits() != cpu_tiled_acc[i].to_bits() {
            acc_mismatches += 1;
        }
        max_full_error = max_full_error.max((npu_acc[i] - full_cpu[i]).abs());
    }
    if acc_mismatches != 0 {
        return Err(format!(
            "host accumulated NPU partials differ from tiled CPU oracle: {acc_mismatches}"
        )
        .into());
    }

    fs::write(
        artifacts().join("k-tiled-npu-acc-f32.bin"),
        f32_bytes(&npu_acc),
    )?;
    fs::write(
        artifacts().join("k-tiled-cpu-oracle-f32.bin"),
        f32_bytes(&cpu_tiled_acc),
    )?;
    fs::write(
        artifacts().join("k-tiled-full-cpu-f32.bin"),
        f32_bytes(&full_cpu),
    )?;

    let done = format!(
        "PASS: K-tiled pure-Rust RK3588 fp16 MatMul M{M} K{K} N{N}; Kt={}; tasks={}; partial_mismatches=0; host_acc_mismatches=0; max_vs_full_fp32_error={max_full_error}",
        plan.kt,
        plan.tiles.len()
    );
    println!("{done}");
    log.push(done);
    fs::write(artifacts().join("k-tiled-run.log"), log.join("\n") + "\n")?;
    Ok(())
}
