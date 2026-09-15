//! RK3588 W8A8 full-K register-command encoder for decode-shaped M=1 matmul.
//!
//! The int8 baseline template and geometry formulas in this module are derived
//! from oRKLLM/ork-driver commit 81428a705007b4858506d8cc245873a793ce90e9,
//! Copyright (c) 2025-2026 Michael Fischer / oRKLLM, under the ISC License.
//! The required permission notice is preserved in `docs/licenses/ork-driver-ISC.txt`.
//! RockNPU keeps only the hardware-programming knowledge here; buffer allocation
//! and submission remain Rust + upstream Rocket UAPI.

pub const INT8_REGCMD_COUNT: usize = 112;
const RK3588_CBUF_ELEMS: usize = 57_344;
const RK3588_NMAX: usize = 8_192;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Int8DecodeDesc {
    pub k: usize,
    pub n: usize,
    pub input_dma: u64,
    pub weights_dma: u64,
    pub output_dma: u64,
}

impl Int8DecodeDesc {
    pub const fn new(
        k: usize,
        n: usize,
        input_dma: u64,
        weights_dma: u64,
        output_dma: u64,
    ) -> Self {
        Self {
            k,
            n,
            input_dma,
            weights_dma,
            output_dma,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Int8EncodeError {
    AddressAbove32Bit(u64),
    InvalidShape {
        k: usize,
        n: usize,
        reason: &'static str,
    },
    SizeOverflow,
}

impl core::fmt::Display for Int8EncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AddressAbove32Bit(v) => write!(
                f,
                "NPU IOVA 0x{v:x} exceeds RK3588 regcmd 32-bit address field"
            ),
            Self::InvalidShape { k, n, reason } => {
                write!(
                    f,
                    "unsupported int8 decode MatMul shape M=1 K={k} N={n}: {reason}"
                )
            }
            Self::SizeOverflow => write!(f, "int8 decode MatMul size arithmetic overflow"),
        }
    }
}
impl std::error::Error for Int8EncodeError {}

fn address32(v: u64) -> Result<u32, Int8EncodeError> {
    u32::try_from(v).map_err(|_| Int8EncodeError::AddressAbove32Bit(v))
}

fn patch_register(ops: &mut [u64; INT8_REGCMD_COUNT], reg: u16, value: u32) {
    let word = ops
        .iter_mut()
        .find(|word| (**word as u16) == reg)
        .unwrap_or_else(|| panic!("validated int8 regcmd is missing register 0x{reg:04x}"));
    *word = (*word & 0xffff_0000_0000_ffff) | ((value as u64) << 16);
}

/// Index into the RK3588 full-K int8 weight layout `[N/32][K/32][32][32]`.
/// `k_index` and `n_index` are zero-based logical coordinates of B[K,N].
pub fn weight_i8_fullk_index(k: usize, n: usize, k_index: usize, n_index: usize) -> usize {
    debug_assert!(k.is_multiple_of(32) && n.is_multiple_of(32));
    debug_assert!(k_index < k && n_index < n);
    let kt = k / 32;
    let nt = n_index / 32;
    let nl = n_index % 32;
    let kb = k_index / 32;
    let kk = k_index % 32;
    nt * kt * 32 * 32 + kb * 32 * 32 + nl * 32 + kk
}

/// Encode the current production-style W8A8 decode primitive:
/// `C[1,N] i32 = A[1,K] i8 x B[K,N] i8`.
///
/// This intentionally exposes only the first hardware-proven envelope we need
/// for LLM decode. Wider K can later use the ork-derived K-split/full-K policies,
/// but is not silently accepted here until it has its own Rocket hardware gate.
pub fn encode_int8_decode_m1(
    desc: Int8DecodeDesc,
) -> Result<[u64; INT8_REGCMD_COUNT], Int8EncodeError> {
    if desc.k == 0 || desc.n == 0 {
        return Err(Int8EncodeError::InvalidShape {
            k: desc.k,
            n: desc.n,
            reason: "dimensions must be non-zero",
        });
    }
    if !desc.k.is_multiple_of(512) || desc.k > 4096 {
        return Err(Int8EncodeError::InvalidShape {
            k: desc.k,
            n: desc.n,
            reason: "first full-K Rocket gate requires K%512==0 and K<=4096",
        });
    }
    if !desc.n.is_multiple_of(32) || desc.n > RK3588_NMAX {
        return Err(Int8EncodeError::InvalidShape {
            k: desc.k,
            n: desc.n,
            reason: "requires N%32==0 and N<=8192",
        });
    }

    let input_dma = address32(desc.input_dma)?;
    let weights_dma = address32(desc.weights_dma)?;
    let output_dma = address32(desc.output_dma)?;
    let k = u32::try_from(desc.k).map_err(|_| Int8EncodeError::SizeOverflow)?;
    let n = u32::try_from(desc.n).map_err(|_| Int8EncodeError::SizeOverflow)?;
    let weight_elems = desc
        .k
        .checked_mul(desc.n)
        .ok_or(Int8EncodeError::SizeOverflow)?;
    let weight_elems = u32::try_from(weight_elems).map_err(|_| Int8EncodeError::SizeOverflow)?;

    let mut ops = INT8_TEMPLATE;

    // CNA contraction and M=1 geometry.
    patch_register(&mut ops, 0x1024, ((k - 1) << 16) | k);
    patch_register(&mut ops, 0x1030, weight_elems);
    patch_register(&mut ops, 0x1034, k);
    patch_register(&mut ops, 0x1044, k.div_ceil(64));
    patch_register(&mut ops, 0x107c, k / 16);
    patch_register(&mut ops, 0x1084, 0x0001_0001);
    patch_register(&mut ops, 0x1088, k);
    patch_register(&mut ops, 0x1020, 0x0001_0001);
    patch_register(&mut ops, 0x102c, 1);

    // Output-column geometry for int32 C[1,N].
    patch_register(&mut ops, 0x1038, 0x0101_0000 | n);
    patch_register(&mut ops, 0x3018, n - 1);
    patch_register(&mut ops, 0x403c, ((n - 1) << 16) | (n - 1));
    patch_register(&mut ops, 0x4058, n - 1);
    patch_register(&mut ops, 0x4038, (((n / 4) - 1) << 16) | ((n / 4) - 1));

    // sched=1 from ork-driver's current production synthesizer. For M=1 the
    // row-count hint is two rows when the CBUF permits it; 0x1040 is the
    // K-dependent reduction schedule (0xb1 only at K=512, not a universal M=1 literal).
    let mut r = (2 * RK3588_CBUF_ELEMS) / desc.k;
    r = r.max(1);
    let mut r_pow2 = 1usize;
    while r_pow2.saturating_mul(2) <= r {
        r_pow2 *= 2;
    }
    let rows = 2usize.min(r_pow2);
    patch_register(
        &mut ops,
        0x1010,
        u32::try_from(16 * rows).map_err(|_| Int8EncodeError::SizeOverflow)?,
    );

    let scale = desc.k / 512;
    let base =
        177i32 - 15i32 * (i32::try_from(scale).map_err(|_| Int8EncodeError::SizeOverflow)? - 1);
    let v = base.max(0x1b) as u32;
    patch_register(&mut ops, 0x1040, v);

    patch_register(&mut ops, 0x1070, input_dma);
    patch_register(&mut ops, 0x1110, weights_dma);
    patch_register(&mut ops, 0x4020, output_dma);

    // M=1 output height/write geometry.
    patch_register(&mut ops, 0x4034, 0);
    patch_register(&mut ops, 0x405c, 0);
    patch_register(&mut ops, 0x3014, 0);

    Ok(ops)
}

// ork-driver's ISC-licensed RKNN_INT8_MM_INT8_TO_INT32 4x32x32 baseline,
// represented in RockNPU's native u64 `{op,value,reg}` encoding. Geometry and
// addresses are overwritten by `encode_int8_decode_m1` before submission.
const INT8_TEMPLATE: [u64; INT8_REGCMD_COUNT] = [
    0x0201000000b11040,
    0x0201000000001104,
    0x0201000000001100,
    0x020120000000100c,
    0x10010000000e4004,
    0x020120000000100c,
    0x0201000000501010,
    0x0201000000091014,
    0x0201000100041020,
    0x0201001f00201024,
    0x0201000000011028,
    0x020100000004102c,
    0x0201000004001030,
    0x0201000000201034,
    0x0201010100201038,
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
    0x0201fffe40001070,
    0x0201000000001074,
    0x0201000f000f1078,
    0x020100000002107c,
    0x0201000000001080,
    0x0201000100041084,
    0x0201000000201088,
    0x0201000000001100,
    0x0201000000001104,
    0x0201fffe50001110,
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
    0x0801000000013010,
    0x0801000300003014,
    0x08010000001f3018,
    0x080100000000301c,
    0x0801000000003030,
    0x1001000001e4400c,
    0x1001800000004010,
    0x1001000000004014,
    0x1001fffe20004020,
    0x1001000000104024,
    0x1001000000004030,
    0x1001000000034034,
    0x1001000700074038,
    0x1001001f001f403c,
    0x1001000000534040,
    0x1001000000004044,
    0x1001000000004048,
    0x100100000000404c,
    0x1001000007fc4050,
    0x1001000000004054,
    0x10010000001f4058,
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
    0x1001000000014084,
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
    0x0000000000000000,
    0x0101000000000014,
    0x0041000000000000,
    0x00810000000d0008,
];

#[cfg(test)]
mod tests {
    use super::*;

    fn reg_value(ops: &[u64], reg: u16) -> u32 {
        let word = ops.iter().find(|word| (**word as u16) == reg).unwrap();
        ((*word >> 16) & 0xffff_ffff) as u32
    }

    #[test]
    fn m1_k2048_n2048_geometry_matches_ork_synth_contract() {
        let ops = encode_int8_decode_m1(Int8DecodeDesc::new(
            2048,
            2048,
            0x1111_1000,
            0x2222_2000,
            0x3333_3000,
        ))
        .unwrap();
        assert_eq!(ops.len(), INT8_REGCMD_COUNT);
        assert_eq!(reg_value(&ops, 0x1024), 0x07ff_0800);
        assert_eq!(reg_value(&ops, 0x1030), 2048 * 2048);
        assert_eq!(reg_value(&ops, 0x1038), 0x0101_0800);
        assert_eq!(reg_value(&ops, 0x1010), 0x20);
        assert_eq!(reg_value(&ops, 0x1040), 0x84);
        assert_eq!(reg_value(&ops, 0x107c), 128);
        assert_eq!(reg_value(&ops, 0x1084), 0x0001_0001);
        assert_eq!(reg_value(&ops, 0x1070), 0x1111_1000);
        assert_eq!(reg_value(&ops, 0x1110), 0x2222_2000);
        assert_eq!(reg_value(&ops, 0x4020), 0x3333_3000);
        assert_eq!(reg_value(&ops, 0x4034), 0);
    }

    #[test]
    fn rejects_unproven_fullk_shapes() {
        for (k, n) in [(5632, 2048), (2048, 2047), (2048, 8224), (1500, 2048)] {
            assert!(encode_int8_decode_m1(Int8DecodeDesc::new(k, n, 1, 2, 3)).is_err());
        }
    }

    #[test]
    fn fullk_weight_index_matches_32x32_tile_order() {
        assert_eq!(weight_i8_fullk_index(2048, 2048, 0, 0), 0);
        assert_eq!(weight_i8_fullk_index(2048, 2048, 31, 31), 1023);
        assert_eq!(weight_i8_fullk_index(2048, 2048, 32, 0), 1024);
        assert_eq!(weight_i8_fullk_index(2048, 2048, 0, 32), 2048 * 32);
    }
}
