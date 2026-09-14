//! Minimal RK3588 register-command encoder for the first hardware vertical slice.
//!
//! RK3588 register-command encoding for the first fp16 MatMul path.
//! The production encoder derives geometry/bank/stride/register values from a descriptor.
//! A 4x32x16 byte-for-byte capture remains only as a regression golden.

pub const FP16_M: usize = 4;
pub const FP16_K: usize = 32;
pub const FP16_N: usize = 16;
pub const FP16_REGCMD_COUNT: usize = 126;

mod conv;
mod general;
mod planner;
pub use conv::{Fp16Conv2dDesc, encode_fp16_conv2d};
pub use general::{
    Fp16MatmulDesc, encode_fp16_matmul, encode_fp16_matmul_accumulate,
    encode_fp16_matmul_fp32_output,
};
pub use planner::{
    Fp16MatmulPlan, Fp16MatmulTile, plan_fp16_matmul, plan_fp16_matmul_compatible_m,
};

/// FP16 direct-Conv native weight cube index (1-based logical coordinates).
/// The ordering follows Mesa's MIT-licensed coefficient packing loop:
/// OC-group -> IC-group -> kernel H/W -> OC-inner -> IC-inner. FP16 group sizes
/// are OC16/IC32, fixed by the already validated 1x1 `weight_fp16` layout.
pub fn weight_conv_fp16(
    oc_total: usize,
    ic_total: usize,
    kh_total: usize,
    kw_total: usize,
    oc: usize,
    ic: usize,
    kh: usize,
    kw: usize,
) -> usize {
    debug_assert!(
        oc >= 1
            && oc <= oc_total
            && ic >= 1
            && ic <= ic_total
            && kh >= 1
            && kh <= kh_total
            && kw >= 1
            && kw <= kw_total
    );
    let oc1 = (oc - 1) / 16;
    let oc2 = (oc - 1) % 16;
    let ic1 = (ic - 1) / 32;
    let ic2 = (ic - 1) % 32;
    let ic_groups = ic_total.div_ceil(32);
    (((((oc1 * ic_groups + ic1) * kh_total + (kh - 1)) * kw_total + (kw - 1)) * 16 + oc2) * 32)
        + ic2
}

#[cfg(test)]
const GOLDEN_4X32X16: [u64; FP16_REGCMD_COUNT] = [
    0x10010000000e4004,
    0x20010000000e5004,
    0x020100000120100c,
    0x0201000000501010,
    0x0201000000091014,
    0x0201000100041020,
    0x0201001f00201024,
    0x0201000000011028,
    0x020100000004102c,
    0x0201000004001030,
    0x0201000000401034,
    0x0201010100101038,
    0x0201000000b11040,
    0x0201000000011044,
    0x02010000000b104c,
    0x0201000100001050,
    0x0201000100001054,
    0x0201000100001058,
    0x020100010000105c,
    0x0201000000001060,
    0x0201000000001064,
    0x0201000000001068,
    0x0201111110001070,
    0x0201000000001074,
    0x0201000f000f1078,
    0x020100000004107c,
    0x0201000000001080,
    0x0201000100041084,
    0x0201000000201088,
    0x0201000000001100,
    0x0201000000001104,
    0x0201222220001110,
    0x0201000000001140,
    0x0201000000001144,
    0x0201000000001148,
    0x020100000000114c,
    0x0201000000001150,
    0x0201000000001154,
    0x0201000000001158,
    0x020100000000115c,
    0x0201000000001160,
    0x0201000000001164,
    0x0201000000001168,
    0x020100000000116c,
    0x0201000000001170,
    0x0201000000001174,
    0x0201000000001178,
    0x020100000000117c,
    0x0201000000001180,
    0x0201000000001184,
    0x0801000002013010,
    0x0801000300003014,
    0x08010000000f3018,
    0x080100000000301c,
    0x0801000000003030,
    0x1001000001e4400c,
    0x1001480000024010,
    0x1001000000004014,
    0x1001333330004020,
    0x1001000000404024,
    0x1001000000004030,
    0x1001000000034034,
    0x1001000000004038,
    0x1001000f000f403c,
    0x1001000000534040,
    0x1001000000004044,
    0x1001000000004048,
    0x100100000000404c,
    0x1001000001264050,
    0x1001000000004054,
    0x10010000000f4058,
    0x100100030000405c,
    0x1001000000534060,
    0x1001000000004064,
    0x1001000000004068,
    0x100100000000406c,
    0x1001000003834070,
    0x1001000000004074,
    0x1001000000014078,
    0x100100000000407c,
    0x1001000000004080,
    0x1001000100014084,
    0x1001000000004088,
    0x1001000000004090,
    0x1001000000004094,
    0x1001000000004098,
    0x100100000000409c,
    0x10010000000040a0,
    0x10010000000040a4,
    0x10010000000040a8,
    0x10010000000040ac,
    0x10010000008040c0,
    0x10010000000040c4,
    0x1001000000004100,
    0x1001000000004104,
    0x1001000000004108,
    0x100100000000410c,
    0x1001000000004110,
    0x1001000000004114,
    0x1001000000004118,
    0x100100000000411c,
    0x1001000000004120,
    0x1001000000004124,
    0x1001000000004128,
    0x100100000000412c,
    0x200100000000500c,
    0x2001000000035010,
    0x20010000000f5014,
    0x2001000000005018,
    0x200100000000501c,
    0x2001000000005020,
    0x2001000000005028,
    0x200100000000502c,
    0x2001000000015034,
    0x2001000000005038,
    0x2001000000005040,
    0x2001000078185044,
    0x2001000000005048,
    0x200100000000504c,
    0x2001000000005064,
    0x2001010101015068,
    0x200100000000506c,
    0x0000000000000000,
    0x0101000000000014,
    0x0041000000000000,
    0x00810000001d0008,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeError {
    AddressAbove32Bit(u64),
    InvalidShape {
        m: usize,
        k: usize,
        n: usize,
        reason: &'static str,
    },
    CbufUnsupported {
        feature_banks: u32,
        weight_banks: u32,
        reason: &'static str,
    },
    SizeOverflow,
}

impl core::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AddressAbove32Bit(v) => write!(
                f,
                "NPU IOVA 0x{v:x} exceeds RK3588 regcmd 32-bit address field"
            ),
            Self::InvalidShape { m, k, n, reason } => {
                write!(
                    f,
                    "unsupported fp16 MatMul shape M={m} K={k} N={n}: {reason}"
                )
            }
            Self::CbufUnsupported {
                feature_banks,
                weight_banks,
                reason,
            } => write!(
                f,
                "fp16 MatMul tile does not fit the RK3588 12-bank CBUF budget (feature_banks={feature_banks}, weight_banks={weight_banks}): {reason}"
            ),
            Self::SizeOverflow => write!(f, "fp16 MatMul descriptor size arithmetic overflow"),
        }
    }
}
impl std::error::Error for EncodeError {}

pub fn encode_fp16_4x32x16(
    input_dma: u64,
    weights_dma: u64,
    output_dma: u64,
) -> Result<[u64; FP16_REGCMD_COUNT], EncodeError> {
    encode_fp16_matmul(Fp16MatmulDesc::new(
        FP16_M,
        FP16_K,
        FP16_N,
        input_dma,
        weights_dma,
        output_dma,
    ))
}

/// NC1HWC2 feature index; arguments c/h/w are 1-based to match the validated reference.
pub fn feature_data(
    _c_total: usize,
    h_total: usize,
    w_total: usize,
    c2: usize,
    c: usize,
    h: usize,
    w: usize,
) -> usize {
    let plane = (c - 1) / c2;
    let src = plane * h_total * w_total * c2;
    let offset = (c - 1) % c2;
    src + c2 * ((h - 1) * w_total + (w - 1)) + offset
}

/// RK3588 fp16 matmul weight cube index; k/c are 1-based.
pub fn weight_fp16(c_total: usize, k: usize, c: usize) -> usize {
    let kpg = (k - 1) / 16;
    let cpg = (c - 1) / 32;
    (cpg * 32) * 16 + (kpg * 16 * c_total) + ((c - 1) % 32) + (((k - 1) % 16) * 32)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn conv_weight_k1_collapses_to_matmul_layout() {
        for oc in 1..=32 {
            for ic in 1..=64 {
                assert_eq!(
                    weight_conv_fp16(32, 64, 1, 1, oc, ic, 1, 1),
                    weight_fp16(64, oc, ic)
                );
            }
        }
    }

    #[test]
    fn conv_weight_spatial_is_between_channel_groups_and_inner_atoms() {
        assert_eq!(weight_conv_fp16(16, 32, 5, 5, 1, 1, 1, 1), 0);
        assert_eq!(weight_conv_fp16(16, 32, 5, 5, 1, 1, 1, 2), 16 * 32);
        assert_eq!(weight_conv_fp16(16, 32, 5, 5, 2, 1, 1, 1), 32);
        assert_eq!(weight_conv_fp16(16, 32, 5, 5, 1, 2, 1, 1), 1);
    }

    #[test]
    fn sentinel_capture_matches_reference_template() {
        let ops = encode_fp16_4x32x16(0x1111_1000, 0x2222_2000, 0x3333_3000).unwrap();
        assert_eq!(ops, GOLDEN_4X32X16);
        assert_eq!(ops.len(), 126);
    }
    #[test]
    fn known_layout_indices() {
        assert_eq!(feature_data(32, 4, 1, 8, 1, 1, 1), 0);
        assert_eq!(feature_data(32, 4, 1, 8, 9, 1, 1), 32);
        assert_eq!(weight_fp16(32, 1, 1), 0);
        assert_eq!(weight_fp16(32, 16, 32), 511);
    }
}
