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

fn f16_bytes(values: &[f16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const M: usize = 512;
    const K: usize = 512;
    const N: usize = 128;

    fs::create_dir_all(artifacts())?;
    let plan = plan_fp16_matmul(M, K, N)?;
    if plan.k_tiles() != 1 || plan.tiles.len() != 2 {
        return Err(format!(
            "unexpected first tiled gate plan: Mt={} Kt={} Nt={} mtiles={} ktiles={} ntiles={} tasks={}",
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
    fs::write(artifacts().join("tiled-plan.txt"), &plan_text)?;
    print!("{plan_text}");

    let a: Vec<f16> = (0..M * K)
        .map(|i| f16::from_f32(((i * 7 + 3) % 5) as f32 - 2.0))
        .collect();
    let b: Vec<f16> = (0..N * K)
        .map(|i| f16::from_f32(((i * 3 + 1) % 5) as f32 - 2.0))
        .collect();

    let mut cpu = vec![f16::ZERO; M * N];
    for m in 0..M {
        for n in 0..N {
            let mut acc = 0.0f32;
            for k in 0..K {
                acc += a[m * K + k].to_f32() * b[n * K + k].to_f32();
            }
            cpu[m * N + n] = f16::from_f32(acc);
        }
    }

    let device = RocketDevice::open()?;
    let _guard = device.alloc_buffer(4096)?;
    let mut got = vec![f16::ZERO; M * N];
    let mut log = vec![plan_text.trim_end().to_string()];

    for (ti, tile) in plan.tiles.iter().enumerate() {
        if tile.k0 != 0 || tile.k != K {
            return Err("first tiled hardware gate unexpectedly needs K accumulation".into());
        }

        let mut regcmd = device.alloc_buffer(4096)?;
        let mut input = device.alloc_buffer(tile.m * tile.k * 2)?;
        let mut weights = device.alloc_buffer(tile.n * tile.k * 2)?;
        let mut output = device.alloc_buffer(tile.m * tile.n * 2)?;

        for bo in [&regcmd, &input, &weights, &output] {
            if bo.dma_address() >> 32 != 0 {
                return Err(format!(
                    "tile {ti}: IOVA 0x{:x} exceeds 32-bit field",
                    bo.dma_address()
                )
                .into());
            }
        }

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
        for tm in 0..tile.m {
            for tk in 0..tile.k {
                let dst = feature_data(tile.k, tile.m, 1, 8, tk + 1, tm + 1, 1);
                let src = (tile.m0 + tm) * K + tile.k0 + tk;
                put_f16(input.as_mut_slice(), dst, a[src]);
            }
        }
        input.fini()?;

        weights.prep_relative(0)?;
        weights.as_mut_slice().fill(0);
        for tn in 0..tile.n {
            for tk in 0..tile.k {
                let dst = weight_fp16(tile.k, tn + 1, tk + 1);
                let src = (tile.n0 + tn) * K + tile.k0 + tk;
                put_f16(weights.as_mut_slice(), dst, b[src]);
            }
        }
        weights.fini()?;

        output.prep_relative(0)?;
        output.as_mut_slice().fill(0xAA);
        output.fini()?;

        fs::write(
            artifacts().join(format!("tiled-regcmd-{ti}.bin")),
            words_bytes(&ops),
        )?;
        fs::write(
            artifacts().join(format!("tiled-input-{ti}.bin")),
            input.as_slice(),
        )?;
        fs::write(
            artifacts().join(format!("tiled-weights-{ti}.bin")),
            weights.as_slice(),
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
            artifacts().join(format!("tiled-output-native-{ti}.bin")),
            output.as_slice(),
        )?;

        for tm in 0..tile.m {
            for tn in 0..tile.n {
                let src = feature_data(tile.n, tile.m, 1, 8, tn + 1, tm + 1, 1);
                got[(tile.m0 + tm) * N + tile.n0 + tn] = get_f16(output.as_slice(), src);
            }
        }
        output.fini()?;

        let line = format!(
            "tile={ti} m0={} n0={} k0={} M{} K{} N{} PASS regcmd_iova=0x{:x}",
            tile.m0,
            tile.n0,
            tile.k0,
            tile.m,
            tile.k,
            tile.n,
            regcmd.dma_address()
        );
        println!("{line}");
        log.push(line);
    }

    fs::write(
        artifacts().join("tiled-cpu-output-rowmajor.bin"),
        f16_bytes(&cpu),
    )?;
    fs::write(
        artifacts().join("tiled-got-output-rowmajor.bin"),
        f16_bytes(&got),
    )?;

    let mut mismatches = 0usize;
    for i in 0..got.len() {
        if got[i].to_bits() != cpu[i].to_bits() {
            if mismatches < 10 {
                eprintln!(
                    "mismatch m={} n={} expected={} actual={} ebits=0x{:04x} abits=0x{:04x}",
                    i / N,
                    i % N,
                    cpu[i].to_f32(),
                    got[i].to_f32(),
                    cpu[i].to_bits(),
                    got[i].to_bits()
                );
            }
            mismatches += 1;
        }
    }
    if mismatches != 0 {
        return Err(format!("tiled NPU/CPU mismatches={mismatches}").into());
    }

    let done = format!(
        "PASS: tiled pure-Rust RK3588 fp16 MatMul M{M} K{K} N{N}; tasks={}; mismatches=0",
        plan.tiles.len()
    );
    println!("{done}");
    log.push(done);
    fs::write(artifacts().join("tiled-run.log"), log.join("\n") + "\n")?;
    Ok(())
}
