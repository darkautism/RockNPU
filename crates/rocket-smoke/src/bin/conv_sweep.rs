use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_conv::{Conv2dSpec, Fp16Conv2dExecutor, cpu_reference_fp16};
use rocknpu_regcmd::{Fp16Conv2dDesc, encode_fp16_conv2d};
use std::error::Error;

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    spec: Conv2dSpec,
    seed: u64,
}
fn next(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s
}
fn data(n: usize, seed: u64, modulus: i32, bias: i32) -> Vec<f16> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            let v = ((next(&mut s) >> 32) as u32 % modulus as u32) as i32 - bias;
            f16::from_f32(v as f32)
        })
        .collect()
}
fn main() -> Result<(), Box<dyn Error>> {
    let cases = [
        Case {
            name: "k1-low-rect",
            spec: Conv2dSpec {
                ic: 1,
                ih: 8,
                iw: 12,
                oc: 8,
                kh: 1,
                kw: 1,
                pad_top: 0,
                pad_left: 0,
                stride_y: 1,
                stride_x: 1,
            },
            seed: 0x101,
        },
        Case {
            name: "k3-low-same",
            spec: Conv2dSpec {
                ic: 8,
                ih: 10,
                iw: 14,
                oc: 16,
                kh: 3,
                kw: 3,
                pad_top: 1,
                pad_left: 1,
                stride_y: 1,
                stride_x: 1,
            },
            seed: 0x302,
        },
        Case {
            name: "k5-base-rect",
            spec: Conv2dSpec {
                ic: 32,
                ih: 8,
                iw: 12,
                oc: 16,
                kh: 5,
                kw: 5,
                pad_top: 2,
                pad_left: 2,
                stride_y: 1,
                stride_x: 1,
            },
            seed: 0x503,
        },
        Case {
            name: "k3-stride2",
            spec: Conv2dSpec {
                ic: 16,
                ih: 8,
                iw: 8,
                oc: 16,
                kh: 3,
                kw: 3,
                pad_top: 1,
                pad_left: 1,
                stride_y: 2,
                stride_x: 2,
            },
            seed: 0x321,
        },
        Case {
            name: "k5-stride2-rect",
            spec: Conv2dSpec {
                ic: 32,
                ih: 8,
                iw: 12,
                oc: 16,
                kh: 5,
                kw: 5,
                pad_top: 2,
                pad_left: 2,
                stride_y: 2,
                stride_x: 2,
            },
            seed: 0x522,
        },
        Case {
            name: "k3-stride-y2-x1",
            spec: Conv2dSpec {
                ic: 32,
                ih: 8,
                iw: 12,
                oc: 32,
                kh: 3,
                kw: 3,
                pad_top: 1,
                pad_left: 1,
                stride_y: 2,
                stride_x: 1,
            },
            seed: 0x231,
        },
        Case {
            name: "k3-two-groups",
            spec: Conv2dSpec {
                ic: 33,
                ih: 12,
                iw: 16,
                oc: 17,
                kh: 3,
                kw: 3,
                pad_top: 1,
                pad_left: 1,
                stride_y: 1,
                stride_x: 1,
            },
            seed: 0x334,
        },
        Case {
            name: "k5-valid-64x32",
            spec: Conv2dSpec {
                ic: 64,
                ih: 12,
                iw: 16,
                oc: 32,
                kh: 5,
                kw: 5,
                pad_top: 0,
                pad_left: 0,
                stride_y: 1,
                stride_x: 1,
            },
            seed: 0x645,
        },
        Case {
            name: "k5-four-oc-groups",
            spec: Conv2dSpec {
                ic: 64,
                ih: 16,
                iw: 20,
                oc: 64,
                kh: 5,
                kw: 5,
                pad_top: 2,
                pad_left: 2,
                stride_y: 1,
                stride_x: 1,
            },
            seed: 0x646,
        },
        Case {
            name: "k5-near-cbuf",
            spec: Conv2dSpec {
                ic: 64,
                ih: 24,
                iw: 28,
                oc: 80,
                kh: 5,
                kw: 5,
                pad_top: 2,
                pad_left: 2,
                stride_y: 1,
                stride_x: 1,
            },
            seed: 0x648,
        },
    ];
    let dev = RocketDevice::open()?;
    let mut ex = Fp16Conv2dExecutor::new(&dev)?;
    for c in cases {
        let s = c.spec;
        let input = data(s.ic * s.ih * s.iw, c.seed, 5, 2);
        let weights = data(s.oc * s.ic * s.kh * s.kw, c.seed ^ 0x3588, 3, 1);
        let reference = cpu_reference_fp16(&input, &weights, s)?;
        let got = ex.execute(&input, &weights, s)?;
        let mut bitdiff = 0usize;
        let mut max_abs = 0.0f32;
        for (a, b) in got.iter().zip(&reference) {
            if a.to_bits() != b.to_bits() {
                bitdiff += 1;
            }
            max_abs = max_abs.max((a.to_f32() - b.to_f32()).abs());
        }
        println!(
            "{} PASS logical=IC{} OC{} input={}x{} kernel={}x{} pad={},{} output={}x{} elems={} bitdiff={}/{} max_abs={:.8}",
            c.name,
            s.ic,
            s.oc,
            s.ih,
            s.iw,
            s.kh,
            s.kw,
            s.pad_top,
            s.pad_left,
            s.output_h(),
            s.output_w(),
            got.len(),
            bitdiff,
            got.len(),
            max_abs
        );
        if bitdiff != 0 {
            return Err(format!("{} bitdiff={} max_abs={}", c.name, bitdiff, max_abs).into());
        }
    }
    // Aligned native geometry deliberately exceeds the 12-bank single-task budget:
    // feature=3 banks, weights=10 banks for IC64/OC96/K5 on 24x28.
    let rejected = encode_fp16_conv2d(Fp16Conv2dDesc {
        input_h: 24,
        input_w: 28,
        input_channels: 64,
        output_channels: 96,
        kernel_h: 5,
        kernel_w: 5,
        pad_top: 2,
        pad_left: 2,
        stride_y: 1,
        stride_x: 1,
        input_dma: 0x1000,
        weights_dma: 0x2000,
        output_dma: 0x3000,
    });
    if !matches!(
        rejected,
        Err(rocknpu_regcmd::EncodeError::CbufUnsupported { .. })
    ) {
        return Err(format!("over-budget Conv was not CBUF-rejected: {rejected:?}").into());
    }
    println!("over-budget-cbuf PASS low-level encoder rejected IC64 OC96 24x28 K5");

    // The executor must recover by splitting independent OC16 groups. The full
    // packed weight cube stays resident/unchanged; each tile selects a contiguous
    // OC-group byte range and writes/gathers its own output channels.
    let tiled = Conv2dSpec {
        ic: 64,
        ih: 24,
        iw: 28,
        oc: 96,
        kh: 5,
        kw: 5,
        pad_top: 2,
        pad_left: 2,
        stride_y: 1,
        stride_x: 1,
    };
    let input = data(tiled.ic * tiled.ih * tiled.iw, 0x649, 5, 2);
    let weights = data(
        tiled.oc * tiled.ic * tiled.kh * tiled.kw,
        0x649 ^ 0x3588,
        3,
        1,
    );
    let reference = cpu_reference_fp16(&input, &weights, tiled)?;
    let stream = ex.execute(&input, &weights, tiled)?;
    let resident = ex.prepare_weights(&weights, tiled)?;
    let prepared = ex.execute_prepared(&input, &resident)?;
    let (profiled, timing) = ex.execute_prepared_profiled(&input, &resident)?;
    if timing.jobs_submitted <= 1 {
        return Err(format!("over-budget Conv did not profile as tiled: {timing:?}").into());
    }
    let stream_diff = stream
        .iter()
        .zip(&reference)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let prepared_diff = prepared
        .iter()
        .zip(&reference)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let cross_diff = stream
        .iter()
        .zip(&prepared)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let profiled_diff = profiled
        .iter()
        .zip(&reference)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    if stream_diff != 0 || prepared_diff != 0 || cross_diff != 0 || profiled_diff != 0 {
        return Err(format!(
            "OC-tiled Conv mismatch stream={stream_diff} prepared={prepared_diff} cross={cross_diff} profiled={profiled_diff}"
        )
        .into());
    }
    println!(
        "over-budget-oc-tiled PASS IC64 OC96 24x28 K5 outputs={} jobs={} stream_diff=0 prepared_diff=0 profiled_diff=0 cross_diff=0 resident_bytes={}",
        reference.len(),
        timing.jobs_submitted,
        resident.stats().resident_bytes
    );
    println!(
        "PASS: deterministic FP16 Conv hardware sweep covers kernels, strides, rectangular geometry, channel padding/groups, single-task CBUF boundary, and executor OC tiling"
    );
    Ok(())
}
