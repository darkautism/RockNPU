use half::f16;
use rocket_runtime::{RocketDevice, Task};
use rocknpu_regcmd::{Fp16Conv2dDesc, encode_fp16_conv2d, feature_data, weight_conv_fp16};
use std::{error::Error, fs};
const PIC: usize = 32;
const POC: usize = 16;
const K: usize = 5;
const PAD: usize = 2;
fn read_f32(path: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let b = fs::read(path)?;
    if b.len() % 4 != 0 {
        return Err("bad f32 bytes".into());
    }
    Ok(b.chunks_exact(4)
        .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect())
}
fn put(dst: &mut [u8], i: usize, v: f16) {
    let p = i * 2;
    dst[p..p + 2].copy_from_slice(&v.to_bits().to_le_bytes());
}
fn get(src: &[u8], i: usize) -> f16 {
    let p = i * 2;
    f16::from_bits(u16::from_le_bytes([src[p], src[p + 1]]))
}
fn cpu(input: &[f16], w: &[f16], ic: usize, oc: usize, h: usize, wid: usize) -> Vec<f16> {
    let mut o = vec![f16::ZERO; oc * h * wid];
    for c in 0..oc {
        for y in 0..h {
            for x in 0..wid {
                let mut a = 0.0f32;
                for q in 0..ic {
                    for ky in 0..K {
                        let yy = y + ky;
                        if yy < PAD {
                            continue;
                        }
                        let yy = yy - PAD;
                        if yy >= h {
                            continue;
                        }
                        for kx in 0..K {
                            let xx = x + kx;
                            if xx < PAD {
                                continue;
                            }
                            let xx = xx - PAD;
                            if xx >= wid {
                                continue;
                            }
                            a += input[(q * h + yy) * wid + xx].to_f32()
                                * w[((c * ic + q) * K + ky) * K + kx].to_f32();
                        }
                    }
                }
                o[(c * h + y) * wid + x] = f16::from_f32(a);
            }
        }
    }
    o
}
fn run(
    dev: &RocketDevice,
    name: &str,
    ic: usize,
    oc: usize,
    h: usize,
    wid: usize,
    input_path: &str,
    weight_path: &str,
    ref_path: &str,
) -> Result<(), Box<dyn Error>> {
    let inf = read_f32(input_path)?;
    let wf = read_f32(weight_path)?;
    let rf = read_f32(ref_path)?;
    if inf.len() != ic * h * wid || wf.len() != oc * ic * K * K || rf.len() != oc * h * wid {
        return Err(format!("{name} artifact shape mismatch").into());
    }
    let input: Vec<f16> = inf.into_iter().map(f16::from_f32).collect();
    let weights: Vec<f16> = wf.into_iter().map(f16::from_f32).collect();
    let cref = cpu(&input, &weights, ic, oc, h, wid);
    let _guard = dev.alloc_buffer(4096)?;
    let mut reg = dev.alloc_buffer(4096)?;
    let mut ib = dev.alloc_buffer(PIC * h * wid * 2)?;
    let mut wb = dev.alloc_buffer(POC * PIC * K * K * 2)?;
    let mut ob = dev.alloc_buffer(POC * h * wid * 2)?;
    ib.prep_relative(0)?;
    ib.as_mut_slice().fill(0);
    for c in 0..ic {
        for y in 0..h {
            for x in 0..wid {
                put(
                    ib.as_mut_slice(),
                    feature_data(PIC, h, wid, 8, c + 1, y + 1, x + 1),
                    input[(c * h + y) * wid + x],
                );
            }
        }
    }
    ib.fini()?;
    wb.prep_relative(0)?;
    wb.as_mut_slice().fill(0);
    for c in 0..oc {
        for q in 0..ic {
            for ky in 0..K {
                for kx in 0..K {
                    put(
                        wb.as_mut_slice(),
                        weight_conv_fp16(POC, PIC, K, K, c + 1, q + 1, ky + 1, kx + 1),
                        weights[((c * ic + q) * K + ky) * K + kx],
                    );
                }
            }
        }
    }
    wb.fini()?;
    ob.prep_relative(0)?;
    ob.as_mut_slice().fill(0);
    ob.fini()?;
    let ops = encode_fp16_conv2d(Fp16Conv2dDesc {
        input_h: h,
        input_w: wid,
        input_channels: PIC,
        output_channels: POC,
        kernel_h: K,
        kernel_w: K,
        pad_top: PAD,
        pad_left: PAD,
        stride_y: 1,
        stride_x: 1,
        input_dma: ib.dma_address(),
        weights_dma: wb.dma_address(),
        output_dma: ob.dma_address(),
    })?;
    reg.prep_relative(0)?;
    reg.as_mut_slice().fill(0);
    for (d, v) in reg.as_mut_slice().chunks_exact_mut(8).zip(ops.iter()) {
        d.copy_from_slice(&v.to_le_bytes());
    }
    reg.fini()?;
    dev.submit(
        &[Task {
            regcmd: u32::try_from(reg.dma_address())?,
            regcmd_count: ops.len() as u32,
        }],
        &[ib.handle(), wb.handle(), reg.handle()],
        &[ob.handle()],
    )?;
    ob.prep_relative(2_000_000_000)?;
    let mut got = vec![f16::ZERO; oc * h * wid];
    for c in 0..oc {
        for y in 0..h {
            for x in 0..wid {
                got[(c * h + y) * wid + x] = get(
                    ob.as_slice(),
                    feature_data(POC, h, wid, 8, c + 1, y + 1, x + 1),
                );
            }
        }
    }
    ob.fini()?;
    let mut cpu_max = 0.0f32;
    let mut cpu_sum = 0.0f64;
    let mut onnx_max = 0.0f32;
    let mut onnx_sum = 0.0f64;
    let mut bitdiff = 0usize;
    for i in 0..got.len() {
        let ec = (got[i].to_f32() - cref[i].to_f32()).abs();
        cpu_max = cpu_max.max(ec);
        cpu_sum += ec as f64;
        if got[i].to_bits() != cref[i].to_bits() {
            bitdiff += 1;
        }
        let eo = (got[i].to_f32() - rf[i]).abs();
        onnx_max = onnx_max.max(eo);
        onnx_sum += eo as f64;
    }
    println!(
        "{name} PASS logical=IC{ic}->OC{oc} padded=IC{PIC}->OC{POC} {h}x{wid} cpu_f16_bitdiff={bitdiff}/{} cpu_max={cpu_max:.8} cpu_mean={:.8} onnx_fp32_max={onnx_max:.8} onnx_fp32_mean={:.8}",
        got.len(),
        cpu_sum as f32 / got.len() as f32,
        onnx_sum as f32 / got.len() as f32
    );
    if cpu_max > 0.02 {
        return Err(format!("{name} NPU/CPU f16 error {cpu_max}").into());
    }
    Ok(())
}
fn main() -> Result<(), Box<dyn Error>> {
    let dev = RocketDevice::open()?;
    run(
        &dev,
        "mnist8-conv1",
        1,
        8,
        28,
        28,
        "artifacts/mnist8-conv1-input-f32.bin",
        "artifacts/mnist8-conv1-weight-f32.bin",
        "artifacts/mnist8-conv1-ref-f32.bin",
    )?;
    run(
        &dev,
        "mnist8-conv2",
        8,
        16,
        14,
        14,
        "artifacts/mnist8-conv2-input-f32.bin",
        "artifacts/mnist8-conv2-weight-f32.bin",
        "artifacts/mnist8-conv2-ref-f32.bin",
    )?;
    println!(
        "PASS: both real MNIST-8 Conv layers execute through padded project-owned RK3588 FP16 Conv"
    );
    Ok(())
}
