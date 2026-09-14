use half::f16;
use rocket_runtime::{RocketDevice, Task};
use rocknpu_regcmd::{
    Fp16MatmulDesc, encode_fp16_matmul, encode_fp16_matmul_accumulate, feature_data,
    plan_fp16_matmul, weight_fp16,
};
use std::fs;
use std::path::PathBuf;

fn artifacts() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts")
}
fn put_f16(dst: &mut [u8], i: usize, v: f16) {
    let p = i * 2;
    dst[p..p + 2].copy_from_slice(&v.to_bits().to_le_bytes());
}
fn get_f16(src: &[u8], i: usize) -> f16 {
    let p = i * 2;
    f16::from_bits(u16::from_le_bytes([src[p], src[p + 1]]))
}
fn words_bytes(v: &[u64]) -> Vec<u8> {
    let mut o = Vec::with_capacity(v.len() * 8);
    for &x in v {
        o.extend_from_slice(&x.to_le_bytes());
    }
    o
}
fn f16_bytes(v: &[f16]) -> Vec<u8> {
    let mut o = Vec::with_capacity(v.len() * 2);
    for &x in v {
        o.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    o
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const M: usize = 64;
    const K: usize = 4096;
    const N: usize = 64;
    fs::create_dir_all(artifacts())?;
    let plan = plan_fp16_matmul(M, K, N)?;
    if plan.m_tiles() != 1 || plan.n_tiles() != 1 || plan.k_tiles() != 3 {
        return Err(format!(
            "unexpected KACC plan Mt={} Kt={} Nt={} tasks={}",
            plan.mt,
            plan.kt,
            plan.nt,
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
    print!("{plan_text}");
    fs::write(artifacts().join("kacc-plan.txt"), &plan_text)?;

    let a: Vec<f16> = (0..M * K)
        .map(|i| f16::from_f32(((i * 7 + 3) % 5) as f32 - 2.0))
        .collect();
    let b: Vec<f16> = (0..N * K)
        .map(|i| f16::from_f32(((i * 3 + 1) % 5) as f32 - 2.0))
        .collect();

    // CPU oracle follows the hardware contract exactly: each K partial is first
    // narrowed to fp16, then every EW add writes a new fp16 running value.
    let mut staged = vec![f16::ZERO; M * N];
    let mut full = vec![0.0f32; M * N];
    for m in 0..M {
        for n in 0..N {
            for k in 0..K {
                full[m * N + n] += a[m * K + k].to_f32() * b[n * K + k].to_f32();
            }
        }
    }
    for (ti, tile) in plan.tiles.iter().enumerate() {
        for m in 0..M {
            for n in 0..N {
                let mut part = 0.0f32;
                for k in 0..tile.k {
                    part += a[m * K + tile.k0 + k].to_f32() * b[n * K + tile.k0 + k].to_f32();
                }
                let pf = f16::from_f32(part);
                let idx = m * N + n;
                staged[idx] = if ti == 0 {
                    pf
                } else {
                    f16::from_f32(staged[idx].to_f32() + pf.to_f32())
                };
            }
        }
    }

    let device = RocketDevice::open()?;
    let _guard = device.alloc_buffer(4096)?;
    let mut out_a = device.alloc_buffer(M * N * 2)?;
    let mut out_b = device.alloc_buffer(M * N * 2)?;
    out_a.prep_relative(0)?;
    out_a.as_mut_slice().fill(0xAA);
    out_a.fini()?;
    out_b.prep_relative(0)?;
    out_b.as_mut_slice().fill(0x55);
    out_b.fini()?;
    let mut log = vec![plan_text.trim_end().to_string()];

    for (ti, tile) in plan.tiles.iter().enumerate() {
        let mut regcmd = device.alloc_buffer(4096)?;
        let mut input = device.alloc_buffer(tile.m * tile.k * 2)?;
        let mut weights = device.alloc_buffer(tile.n * tile.k * 2)?;
        let (dst_dma, dst_h, src_dma, src_h) = if ti % 2 == 0 {
            (
                out_a.dma_address(),
                out_a.handle(),
                out_b.dma_address(),
                out_b.handle(),
            )
        } else {
            (
                out_b.dma_address(),
                out_b.handle(),
                out_a.dma_address(),
                out_a.handle(),
            )
        };
        let desc = Fp16MatmulDesc::new(
            tile.m,
            tile.k,
            tile.n,
            input.dma_address(),
            weights.dma_address(),
            dst_dma,
        );
        let ops = if ti == 0 {
            encode_fp16_matmul(desc)?
        } else {
            encode_fp16_matmul_accumulate(desc, src_dma)?
        };

        regcmd.prep_relative(0)?;
        for (chunk, w) in regcmd.as_mut_slice().chunks_exact_mut(8).zip(ops.iter()) {
            chunk.copy_from_slice(&w.to_le_bytes());
        }
        regcmd.fini()?;
        input.prep_relative(0)?;
        input.as_mut_slice().fill(0);
        for m in 0..tile.m {
            for k in 0..tile.k {
                let d = feature_data(tile.k, tile.m, 1, 8, k + 1, m + 1, 1);
                put_f16(input.as_mut_slice(), d, a[(tile.m0 + m) * K + tile.k0 + k]);
            }
        }
        input.fini()?;
        weights.prep_relative(0)?;
        weights.as_mut_slice().fill(0);
        for n in 0..tile.n {
            for k in 0..tile.k {
                let d = weight_fp16(tile.k, n + 1, k + 1);
                put_f16(
                    weights.as_mut_slice(),
                    d,
                    b[(tile.n0 + n) * K + tile.k0 + k],
                );
            }
        }
        weights.fini()?;

        if ti % 2 == 0 {
            out_a.prep_relative(0)?;
            out_a.fini()?;
        } else {
            out_b.prep_relative(0)?;
            out_b.fini()?;
        }
        let task = Task {
            regcmd: u32::try_from(regcmd.dma_address())?,
            regcmd_count: u32::try_from(ops.len())?,
        };
        let mut inputs = vec![input.handle(), weights.handle(), regcmd.handle()];
        if ti > 0 {
            inputs.push(src_h);
        }
        device.submit(&[task], &inputs, &[dst_h])?;
        if ti % 2 == 0 {
            out_a.prep_relative(2_000_000_000)?;
            out_a.fini()?;
        } else {
            out_b.prep_relative(2_000_000_000)?;
            out_b.fini()?;
        }

        fs::write(
            artifacts().join(format!("kacc-regcmd-{ti}.bin")),
            words_bytes(&ops),
        )?;
        let line = format!(
            "tile={ti} k0={} K{} mode={} dst=0x{dst_dma:x} src={} PASS",
            tile.k0,
            tile.k,
            if ti == 0 { "plain" } else { "ew-add" },
            if ti == 0 {
                "none".to_string()
            } else {
                format!("0x{src_dma:x}")
            }
        );
        println!("{line}");
        log.push(line);
    }

    let fin = &if (plan.tiles.len() - 1) % 2 == 0 {
        out_a
    } else {
        out_b
    };
    fin.prep_relative(0)?;
    let mut got = vec![f16::ZERO; M * N];
    let mut mismatches = 0usize;
    let mut max_full = 0.0f32;
    for m in 0..M {
        for n in 0..N {
            let idx = m * N + n;
            let src = feature_data(N, M, 1, 8, n + 1, m + 1, 1);
            got[idx] = get_f16(fin.as_slice(), src);
            if got[idx].to_bits() != staged[idx].to_bits() {
                if mismatches < 10 {
                    eprintln!(
                        "mismatch m={m} n={n} staged={} got={} sbits=0x{:04x} gbits=0x{:04x}",
                        staged[idx].to_f32(),
                        got[idx].to_f32(),
                        staged[idx].to_bits(),
                        got[idx].to_bits()
                    );
                }
                mismatches += 1;
            }
            max_full = max_full.max((got[idx].to_f32() - full[idx]).abs());
        }
    }
    fin.fini()?;
    fs::write(artifacts().join("kacc-final-rowmajor.bin"), f16_bytes(&got))?;
    fs::write(
        artifacts().join("kacc-cpu-staged-rowmajor.bin"),
        f16_bytes(&staged),
    )?;
    if mismatches != 0 {
        return Err(format!("NPU EW staged oracle mismatches={mismatches}").into());
    }
    let done = format!(
        "PASS: NPU EW K-accumulation M{M} K{K} N{N}; Kt={}; tasks={}; staged_mismatches=0; max_vs_full_fp32_error={max_full}",
        plan.kt,
        plan.tiles.len()
    );
    println!("{done}");
    log.push(done);
    fs::write(artifacts().join("kacc-run.log"), log.join("\n") + "\n")?;
    Ok(())
}
