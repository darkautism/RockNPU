use crate::{EncodeError, FP16_REGCMD_COUNT};

const OP_REG_PC: u16 = 0x0101;
const OP_REG_CNA: u16 = 0x0201;
const OP_REG_CORE: u16 = 0x0801;
const OP_REG_DPU: u16 = 0x1001;
const OP_REG_DPU_RDMA: u16 = 0x2001;
const OP_40: u16 = 0x0041;
const OP_ENABLE: u16 = 0x0081;

const CBUF_BANK_SIZE: u64 = 32 * 1024;
const CBUF_BANKS: u32 = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp16MatmulDesc {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub input_dma: u64,
    pub weights_dma: u64,
    pub output_dma: u64,
}

impl Fp16MatmulDesc {
    pub const fn new(
        m: usize,
        k: usize,
        n: usize,
        input_dma: u64,
        weights_dma: u64,
        output_dma: u64,
    ) -> Self {
        Self {
            m,
            k,
            n,
            input_dma,
            weights_dma,
            output_dma,
        }
    }
}

#[inline]
fn npuop(op: u16, value: u32, reg: u16) -> u64 {
    ((op as u64) << 48) | ((value as u64) << 16) | reg as u64
}

fn address32(v: u64) -> Result<u32, EncodeError> {
    u32::try_from(v).map_err(|_| EncodeError::AddressAbove32Bit(v))
}

fn div_ceil_u64(n: u64, d: u64) -> u64 {
    n.div_ceil(d)
}

/// Emit one RK3588 fp16 MatMul task using the CNA -> CORE -> DPU path.
///
/// This is the plain, single-K-tile path: fp16 inputs/weights and fp16 output,
/// with no EW accumulation or CBUF operand reuse yet. The emitted task is always
/// 126 register-command words; only descriptor-derived register values vary.
pub fn encode_fp16_matmul(desc: Fp16MatmulDesc) -> Result<[u64; FP16_REGCMD_COUNT], EncodeError> {
    if desc.m == 0 || desc.k == 0 || desc.n == 0 {
        return Err(EncodeError::InvalidShape {
            m: desc.m,
            k: desc.k,
            n: desc.n,
            reason: "dimensions must be non-zero",
        });
    }
    if desc.m % 4 != 0 || desc.k % 32 != 0 || desc.n % 16 != 0 {
        return Err(EncodeError::InvalidShape {
            m: desc.m,
            k: desc.k,
            n: desc.n,
            reason: "requires M%4==0, K%32==0, N%16==0",
        });
    }
    if desc.m + 1 > 0x3ff {
        return Err(EncodeError::InvalidShape {
            m: desc.m,
            k: desc.k,
            n: desc.n,
            reason: "M+1 exceeds CNA feature_grains 10-bit field",
        });
    }
    if desc.k > u16::MAX as usize {
        return Err(EncodeError::InvalidShape {
            m: desc.m,
            k: desc.k,
            n: desc.n,
            reason: "K exceeds CNA 16-bit channel field",
        });
    }
    if desc.n > 0x2000 {
        return Err(EncodeError::InvalidShape {
            m: desc.m,
            k: desc.k,
            n: desc.n,
            reason: "N exceeds DPU 13-bit channel-minus-one field",
        });
    }

    let input_dma = address32(desc.input_dma)?;
    let weights_dma = address32(desc.weights_dma)?;
    let output_dma = address32(desc.output_dma)?;

    let m = u32::try_from(desc.m).expect("validated M fits u32");
    let k = u32::try_from(desc.k).expect("validated K fits u32");
    let n = u32::try_from(desc.n).expect("validated N fits u32");

    let weight_bytes_per_kernel = (desc.k as u64)
        .checked_mul(2)
        .ok_or(EncodeError::SizeOverflow)?;
    if weight_bytes_per_kernel > CBUF_BANK_SIZE {
        return Err(EncodeError::CbufUnsupported {
            feature_banks: 0,
            weight_banks: 0,
            reason: "one fp16 weight kernel exceeds one CBUF bank; K tiling is required",
        });
    }
    let weight_bytes = weight_bytes_per_kernel
        .checked_mul(desc.n as u64)
        .ok_or(EncodeError::SizeOverflow)?;
    let weight_bytes = u32::try_from(weight_bytes).map_err(|_| EncodeError::SizeOverflow)?;

    let feature_bytes = (desc.m as u64)
        .checked_mul(desc.k as u64)
        .and_then(|v| v.checked_mul(2))
        .ok_or(EncodeError::SizeOverflow)?;
    let feature_banks = div_ceil_u64(feature_bytes, CBUF_BANK_SIZE) as u32;
    let weight_banks = div_ceil_u64(weight_bytes as u64, CBUF_BANK_SIZE) as u32;
    if feature_banks + weight_banks > CBUF_BANKS {
        return Err(EncodeError::CbufUnsupported {
            feature_banks,
            weight_banks,
            reason: "input and weight tiles together exceed the RK3588 CBUF budget; tiling is required",
        });
    }
    let data_bank = feature_banks;
    let weight_bank = CBUF_BANKS - feature_banks;
    let data_entries = k.div_ceil(32);

    // The RK3588 matmul feature path uses a four-byte CNA row stride even for fp16.
    let line_stride = 4u32;
    let surf_stride = line_stride * ((m / 4).saturating_sub(1));

    let core_h = m - 1;
    let core_w = 0u32;
    let core_c = n - 1;

    let mut ops = [0u64; FP16_REGCMD_COUNT];
    let mut i = 0usize;
    let mut emit = |op: u16, value: u32, reg: u16| {
        ops[i] = npuop(op, value, reg);
        i += 1;
    };

    // DPU/DPU_RDMA single-register groups.
    emit(OP_REG_DPU, 0xE, 0x4004);
    emit(OP_REG_DPU_RDMA, 0xE, 0x5004);

    // CNA: 1x1 direct convolution encoding of MatMul.
    emit(OP_REG_CNA, (2 << 7) | (2 << 4), 0x100C);
    emit(OP_REG_CNA, ((m + 1) & 0x3ff) << 4, 0x1010);
    emit(OP_REG_CNA, 0x9, 0x1014);
    emit(OP_REG_CNA, (1 << 16) | (m & 0x7ff), 0x1020);
    emit(OP_REG_CNA, ((k - 1) << 16) | k, 0x1024);
    emit(OP_REG_CNA, 1, 0x1028);
    emit(OP_REG_CNA, m & 0x3ffff, 0x102C);
    emit(OP_REG_CNA, weight_bytes, 0x1030);
    emit(
        OP_REG_CNA,
        (weight_bytes_per_kernel as u32) & 0x7ffff,
        0x1034,
    );
    emit(OP_REG_CNA, (1 << 24) | (1 << 16) | (n & 0x3fff), 0x1038);
    emit(OP_REG_CNA, (weight_bank << 4) | data_bank, 0x1040);
    emit(OP_REG_CNA, data_entries & 0x1fff, 0x1044);
    emit(OP_REG_CNA, 0xB, 0x104C);
    emit(OP_REG_CNA, 1 << 16, 0x1050);
    emit(OP_REG_CNA, 1 << 16, 0x1054);
    emit(OP_REG_CNA, 1 << 16, 0x1058);
    emit(OP_REG_CNA, 1 << 16, 0x105C);
    emit(OP_REG_CNA, 0, 0x1060);
    emit(OP_REG_CNA, 0, 0x1064);
    emit(OP_REG_CNA, 0, 0x1068);
    emit(OP_REG_CNA, input_dma, 0x1070);
    emit(OP_REG_CNA, 0, 0x1074);
    emit(OP_REG_CNA, (0xF << 16) | 0xF, 0x1078);
    emit(OP_REG_CNA, line_stride, 0x107C);
    emit(OP_REG_CNA, surf_stride, 0x1080);
    emit(OP_REG_CNA, (1 << 16) | (m & 0x7ff), 0x1084);
    emit(OP_REG_CNA, k & 0xffff, 0x1088);
    emit(OP_REG_CNA, 0, 0x1100);
    emit(OP_REG_CNA, 0, 0x1104);
    emit(OP_REG_CNA, weights_dma, 0x1110);
    for reg in [
        0x1140, 0x1144, 0x1148, 0x114C, 0x1150, 0x1154, 0x1158, 0x115C, 0x1160, 0x1164, 0x1168,
        0x116C, 0x1170, 0x1174, 0x1178, 0x117C, 0x1180, 0x1184,
    ] {
        emit(OP_REG_CNA, 0, reg);
    }

    // CORE accumulator geometry.
    emit(OP_REG_CORE, (2 << 8) | 1, 0x3010);
    emit(OP_REG_CORE, (core_h << 16) | core_w, 0x3014);
    emit(OP_REG_CORE, core_c, 0x3018);
    emit(OP_REG_CORE, 0, 0x301C);
    emit(OP_REG_CORE, 0, 0x3030);

    // DPU: fp16 output writer with BS/BN/EW bypassed.
    emit(OP_REG_DPU, (0xF << 5) | (2 << 1), 0x400C);
    emit(OP_REG_DPU, (2 << 29) | (2 << 26) | 2, 0x4010);
    emit(OP_REG_DPU, 0, 0x4014);
    emit(OP_REG_DPU, output_dma, 0x4020);
    emit(OP_REG_DPU, m << 4, 0x4024);
    emit(OP_REG_DPU, core_w, 0x4030);
    emit(OP_REG_DPU, core_h, 0x4034);
    emit(OP_REG_DPU, 0, 0x4038);
    emit(OP_REG_DPU, (core_c << 16) | core_c, 0x403C);
    emit(OP_REG_DPU, 0x53, 0x4040);
    emit(OP_REG_DPU, 0, 0x4044);
    emit(OP_REG_DPU, 0, 0x4048);
    emit(OP_REG_DPU, 0, 0x404C);
    emit(OP_REG_DPU, 0x126, 0x4050);
    emit(OP_REG_DPU, 0, 0x4054);
    emit(OP_REG_DPU, core_c, 0x4058);
    emit(OP_REG_DPU, core_h << 16, 0x405C);
    emit(OP_REG_DPU, 0x53, 0x4060);
    emit(OP_REG_DPU, 0, 0x4064);
    emit(OP_REG_DPU, 0, 0x4068);
    emit(OP_REG_DPU, 0, 0x406C);
    emit(OP_REG_DPU, 0x383, 0x4070);
    emit(OP_REG_DPU, 0, 0x4074);
    emit(OP_REG_DPU, 1, 0x4078);
    emit(OP_REG_DPU, 0, 0x407C);
    emit(OP_REG_DPU, 0, 0x4080);
    emit(OP_REG_DPU, 0x0001_0001, 0x4084);
    emit(OP_REG_DPU, 0, 0x4088);
    for reg in [
        0x4090, 0x4094, 0x4098, 0x409C, 0x40A0, 0x40A4, 0x40A8, 0x40AC,
    ] {
        emit(OP_REG_DPU, 0, reg);
    }
    emit(OP_REG_DPU, (m * 2) << 4, 0x40C0);
    emit(OP_REG_DPU, 0, 0x40C4);
    for reg in [
        0x4100, 0x4104, 0x4108, 0x410C, 0x4110, 0x4114, 0x4118, 0x411C, 0x4120, 0x4124, 0x4128,
        0x412C,
    ] {
        emit(OP_REG_DPU, 0, reg);
    }

    // DPU_RDMA is present in the enabled block mask but both read paths are bypassed.
    emit(OP_REG_DPU_RDMA, core_w, 0x500C);
    emit(OP_REG_DPU_RDMA, core_h, 0x5010);
    emit(OP_REG_DPU_RDMA, core_c, 0x5014);
    emit(OP_REG_DPU_RDMA, 0, 0x5018);
    emit(OP_REG_DPU_RDMA, 0, 0x501C);
    emit(OP_REG_DPU_RDMA, 0, 0x5020);
    emit(OP_REG_DPU_RDMA, 0, 0x5028);
    emit(OP_REG_DPU_RDMA, 0, 0x502C);
    emit(OP_REG_DPU_RDMA, 1, 0x5034);
    emit(OP_REG_DPU_RDMA, 0, 0x5038);
    emit(OP_REG_DPU_RDMA, 0, 0x5040);
    emit(OP_REG_DPU_RDMA, (0xF << 11) | (1 << 4) | (1 << 3), 0x5044);
    emit(OP_REG_DPU_RDMA, 0, 0x5048);
    emit(OP_REG_DPU_RDMA, 0, 0x504C);
    emit(OP_REG_DPU_RDMA, 0, 0x5064);
    emit(OP_REG_DPU_RDMA, 0x0101_0101, 0x5068);
    emit(OP_REG_DPU_RDMA, 0, 0x506C);

    // Program trailer and operation enable mask: PC + CNA + DPU + DPU_RDMA.
    emit(0, 0, 0);
    emit(OP_REG_PC, 0, 0x0014);
    emit(OP_40, 0, 0);
    emit(OP_ENABLE, 0x1D, 0x0008);

    debug_assert_eq!(i, FP16_REGCMD_COUNT);
    Ok(ops)
}

/// Emit fp16-input MatMul with the DPU writing the full fp32 accumulator.
/// The output cube uses 4-byte elements with C2=4. This variant differs from the
/// validated fp16-output stream in exactly five output-side register values.
pub fn encode_fp16_matmul_fp32_output(
    desc: Fp16MatmulDesc,
) -> Result<[u64; FP16_REGCMD_COUNT], EncodeError> {
    let m = u32::try_from(desc.m).map_err(|_| EncodeError::SizeOverflow)?;
    let surf_bytes = m.checked_mul(4).ok_or(EncodeError::SizeOverflow)?;
    let mut ops = encode_fp16_matmul(desc)?;
    patch_register(&mut ops, 0x4010, 0xa800_0002);
    patch_register(&mut ops, 0x4050, 0x0000_036e);
    patch_register(&mut ops, 0x4084, 0x0000_0001);
    patch_register(&mut ops, 0x40c0, surf_bytes << 4);
    patch_register(&mut ops, 0x5044, 0x0000_7810);
    Ok(ops)
}

pub(crate) fn patch_register(ops: &mut [u64; FP16_REGCMD_COUNT], reg: u16, value: u32) {
    let word = ops
        .iter_mut()
        .find(|word| (**word as u16) == reg)
        .unwrap_or_else(|| panic!("validated regcmd is missing register 0x{reg:04x}"));
    *word = (*word & 0xffff_0000_0000_ffff) | ((value as u64) << 16);
}

/// Emit the RK3588 fp16 MatMul task that adds a previous fp16 output cube through
/// the DPU EW/ERDMA path. `add_dma` must name a distinct ping-pong buffer from
/// `desc.output_dma`; in-place WDMA/ERDMA is not supported by the validated path.
pub fn encode_fp16_matmul_accumulate(
    desc: Fp16MatmulDesc,
    add_dma: u64,
) -> Result<[u64; FP16_REGCMD_COUNT], EncodeError> {
    if desc.m < 12 {
        return Err(EncodeError::InvalidShape {
            m: desc.m,
            k: desc.k,
            n: desc.n,
            reason: "EW accumulation is currently gated to M>=12 until tiny-M surface-stride geometry is independently validated",
        });
    }
    if add_dma == desc.output_dma {
        return Err(EncodeError::InvalidShape {
            m: desc.m,
            k: desc.k,
            n: desc.n,
            reason: "EW accumulation requires distinct source and destination ping-pong buffers",
        });
    }
    let add = address32(add_dma)?;
    let base_off = u32::try_from(desc.m.checked_mul(16).ok_or(EncodeError::SizeOverflow)?)
        .map_err(|_| EncodeError::SizeOverflow)?;
    let ew_base = add
        .checked_add(base_off)
        .ok_or(EncodeError::AddressAbove32Bit(add_dma + base_off as u64))?;
    let ew_stride = u32::try_from(desc.m.max(12))
        .map_err(|_| EncodeError::SizeOverflow)?
        .checked_shl(4)
        .ok_or(EncodeError::SizeOverflow)?;

    let mut ops = encode_fp16_matmul(desc)?;
    // DPU EW add: per-pixel fp16 source, ADD ALU active, bypass ReLU/LUT.
    patch_register(&mut ops, 0x4070, 0x1082_02c0);
    // RDMA: MRDMA and ERDMA read the previous output cube; ERDMA starts one
    // 8-channel fp16 surface later and uses the validated surface stride/notch.
    patch_register(&mut ops, 0x5018, add);
    patch_register(&mut ops, 0x5034, 0x4000_0008);
    patch_register(&mut ops, 0x5038, ew_base);
    patch_register(&mut ops, 0x5040, ew_stride);
    patch_register(&mut ops, 0x5044, 0x0001_7d40);
    patch_register(&mut ops, 0x504c, ew_stride);
    patch_register(&mut ops, 0x506c, ew_stride);
    Ok(ops)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_misaligned_shapes() {
        for (m, k, n) in [(3, 32, 16), (4, 48, 16), (4, 32, 24)] {
            assert!(matches!(
                encode_fp16_matmul(Fp16MatmulDesc::new(m, k, n, 0x1000, 0x2000, 0x3000)),
                Err(EncodeError::InvalidShape { .. })
            ));
        }
    }

    #[test]
    fn rejects_single_kernel_that_exceeds_cbuf_bank() {
        let err = encode_fp16_matmul(Fp16MatmulDesc::new(4, 16_416, 16, 0x1000, 0x2000, 0x3000))
            .unwrap_err();
        assert!(matches!(err, EncodeError::CbufUnsupported { .. }));
    }

    #[test]
    fn rejects_combined_feature_and_weight_cbuf_overflow() {
        let err = encode_fp16_matmul(Fp16MatmulDesc::new(256, 512, 256, 0x1000, 0x2000, 0x3000))
            .unwrap_err();
        assert!(matches!(
            err,
            EncodeError::CbufUnsupported {
                feature_banks: 8,
                weight_banks: 8,
                ..
            }
        ));
    }

    #[test]
    fn accumulation_changes_only_expected_registers() {
        let d = Fp16MatmulDesc::new(64, 256, 64, 0x1000, 0x2000, 0x3000);
        let plain = encode_fp16_matmul(d).unwrap();
        let accum = encode_fp16_matmul_accumulate(d, 0x4000).unwrap();
        let diffs: Vec<u16> = plain
            .iter()
            .zip(accum.iter())
            .filter_map(|(a, b)| (a != b).then_some(*b as u16))
            .collect();
        assert_eq!(
            diffs,
            vec![
                0x4070, 0x5018, 0x5034, 0x5038, 0x5040, 0x5044, 0x504c, 0x506c
            ]
        );
    }

    #[test]
    fn accumulation_refuses_in_place_output() {
        let d = Fp16MatmulDesc::new(64, 256, 64, 0x1000, 0x2000, 0x3000);
        assert!(matches!(
            encode_fp16_matmul_accumulate(d, 0x3000),
            Err(EncodeError::InvalidShape { .. })
        ));
    }

    #[test]
    fn fp32_output_changes_only_validated_output_registers() {
        let d = Fp16MatmulDesc::new(64, 256, 64, 0x1111_1000, 0x2222_2000, 0x3333_3000);
        let plain = encode_fp16_matmul(d).unwrap();
        let f32o = encode_fp16_matmul_fp32_output(d).unwrap();
        let diffs: Vec<(u16, u32)> = plain
            .iter()
            .zip(f32o.iter())
            .filter_map(|(a, b)| {
                (a != b).then_some(((*b & 0xffff) as u16, ((*b >> 16) & 0xffff_ffff) as u32))
            })
            .collect();
        assert_eq!(
            diffs,
            vec![
                (0x4010, 0xa800_0002),
                (0x4050, 0x0000_036e),
                (0x4084, 0x0000_0001),
                (0x40c0, 0x0000_1000),
                (0x5044, 0x0000_7810),
            ]
        );
    }
}
