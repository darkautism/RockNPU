use half::f16;
use rocket_runtime::{RocketDevice, Task};
use rocknpu_regcmd::{Fp16Conv2dDesc, encode_fp16_conv2d, feature_data, weight_conv_fp16};
use std::error::Error;

const IC: usize = 32;
const IH: usize = 8;
const IW: usize = 8;
const OC: usize = 16;
const KH: usize = 5;
const KW: usize = 5;
const PAD: usize = 2;
const OH: usize = 8;
const OW: usize = 8;

fn put_f16(dst: &mut [u8], idx: usize, v: f16) {
    let p = idx * 2;
    dst[p..p + 2].copy_from_slice(&v.to_bits().to_le_bytes());
}
fn get_f16(src: &[u8], idx: usize) -> f16 {
    let p = idx * 2;
    f16::from_bits(u16::from_le_bytes([src[p], src[p + 1]]))
}
fn cpu_conv(input: &[f16], weights: &[f16]) -> Vec<f16> {
    let mut out = vec![f16::ZERO; OC * OH * OW];
    for oc in 0..OC {
        for oy in 0..OH {
            for ox in 0..OW {
                let mut acc = 0.0f32;
                for ic in 0..IC {
                    for ky in 0..KH {
                        let iy = oy + ky;
                        if iy < PAD {
                            continue;
                        }
                        let iy = iy - PAD;
                        if iy >= IH {
                            continue;
                        }
                        for kx in 0..KW {
                            let ix = ox + kx;
                            if ix < PAD {
                                continue;
                            }
                            let ix = ix - PAD;
                            if ix >= IW {
                                continue;
                            }
                            acc += input[(ic * IH + iy) * IW + ix].to_f32()
                                * weights[((oc * IC + ic) * KH + ky) * KW + kx].to_f32();
                        }
                    }
                }
                out[(oc * OH + oy) * OW + ox] = f16::from_f32(acc);
            }
        }
    }
    out
}
fn main() -> Result<(), Box<dyn Error>> {
    let input: Vec<f16> = (0..IC * IH * IW)
        .map(|i| f16::from_f32(((i * 7 + 3) % 3) as f32 - 1.0))
        .collect();
    let weights: Vec<f16> = (0..OC * IC * KH * KW)
        .map(|i| f16::from_f32(((i * 5 + 1) % 3) as f32 - 1.0))
        .collect();
    let reference = cpu_conv(&input, &weights);
    let dev = RocketDevice::open()?;
    let _guard = dev.alloc_buffer(4096)?;
    let mut regcmd = dev.alloc_buffer(4096)?;
    let mut inbo = dev.alloc_buffer(IC * IH * IW * 2)?;
    let mut wtbo = dev.alloc_buffer(OC * IC * KH * KW * 2)?;
    let mut outbo = dev.alloc_buffer(OC * OH * OW * 2)?;
    for (name, dma) in [
        ("regcmd", regcmd.dma_address()),
        ("input", inbo.dma_address()),
        ("weights", wtbo.dma_address()),
        ("output", outbo.dma_address()),
    ] {
        if dma > u32::MAX as u64 {
            return Err(format!("{name} dma above 32-bit: 0x{dma:x}").into());
        }
    }
    inbo.prep_relative(0)?;
    inbo.as_mut_slice().fill(0);
    for ic in 0..IC {
        for y in 0..IH {
            for x in 0..IW {
                let src = input[(ic * IH + y) * IW + x];
                let di = feature_data(IC, IH, IW, 8, ic + 1, y + 1, x + 1);
                put_f16(inbo.as_mut_slice(), di, src);
            }
        }
    }
    inbo.fini()?;
    wtbo.prep_relative(0)?;
    wtbo.as_mut_slice().fill(0);
    for oc in 0..OC {
        for ic in 0..IC {
            for ky in 0..KH {
                for kx in 0..KW {
                    let src = weights[((oc * IC + ic) * KH + ky) * KW + kx];
                    let di = weight_conv_fp16(OC, IC, KH, KW, oc + 1, ic + 1, ky + 1, kx + 1);
                    put_f16(wtbo.as_mut_slice(), di, src);
                }
            }
        }
    }
    wtbo.fini()?;
    outbo.prep_relative(0)?;
    outbo.as_mut_slice().fill(0);
    outbo.fini()?;
    let ops = encode_fp16_conv2d(Fp16Conv2dDesc {
        input_h: IH,
        input_w: IW,
        input_channels: IC,
        output_channels: OC,
        kernel_h: KH,
        kernel_w: KW,
        pad_top: PAD,
        pad_left: PAD,
        stride_y: 1,
        stride_x: 1,
        input_dma: inbo.dma_address(),
        weights_dma: wtbo.dma_address(),
        output_dma: outbo.dma_address(),
    })?;
    regcmd.prep_relative(0)?;
    regcmd.as_mut_slice().fill(0);
    for (chunk, w) in regcmd.as_mut_slice().chunks_exact_mut(8).zip(ops.iter()) {
        chunk.copy_from_slice(&w.to_le_bytes());
    }
    regcmd.fini()?;
    let task = Task {
        regcmd: u32::try_from(regcmd.dma_address())?,
        regcmd_count: ops.len() as u32,
    };
    dev.submit(
        &[task],
        &[inbo.handle(), wtbo.handle(), regcmd.handle()],
        &[outbo.handle()],
    )?;
    outbo.prep_relative(2_000_000_000)?;
    let mut got = vec![f16::ZERO; OC * OH * OW];
    for oc in 0..OC {
        for y in 0..OH {
            for x in 0..OW {
                let si = feature_data(OC, OH, OW, 8, oc + 1, y + 1, x + 1);
                got[(oc * OH + y) * OW + x] = get_f16(outbo.as_slice(), si);
            }
        }
    }
    outbo.fini()?;
    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    let mut first = Vec::new();
    for i in 0..got.len() {
        let e = (got[i].to_f32() - reference[i].to_f32()).abs();
        max_abs = max_abs.max(e);
        if got[i].to_bits() != reference[i].to_bits() {
            mismatches += 1;
            if first.len() < 8 {
                first.push((i, reference[i].to_f32(), got[i].to_f32()));
            }
        }
    }
    println!(
        "conv-fp16 IC{IC} {IH}x{IW} OC{OC} K{KH}x{KW} pad{PAD} mismatches={mismatches}/{} max_abs={max_abs:.8} first={first:?} dmas=[0x{:x},0x{:x},0x{:x}]",
        got.len(),
        inbo.dma_address(),
        wtbo.dma_address(),
        outbo.dma_address()
    );
    if mismatches != 0 {
        return Err("project-owned FP16 Conv hardware mismatch".into());
    }
    println!("PASS: project-owned RK3588 FP16 5x5 Conv matches independent CPU oracle bit-exact");
    Ok(())
}
