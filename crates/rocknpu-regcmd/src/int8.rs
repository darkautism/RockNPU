// SPDX-License-Identifier: MIT
//
//! RK3588 W8A8 full-K register-command encoder for decode-shaped M=1 matmul.
//!
//! ork-driver's public RK35xx research informed this implementation. The direct
//! source-derived baseline register template is deliberately isolated in
//! `int8/ork_isc.rs` under its original ISC notice; the encoder/API here is MIT.

mod ork_isc;

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

fn patch_register_all(ops: &mut [u64; INT8_REGCMD_COUNT], reg: u16, value: u32) {
    let mut patched = 0usize;
    for word in ops.iter_mut().filter(|word| (**word as u16) == reg) {
        *word = (*word & 0xffff_0000_0000_ffff) | ((value as u64) << 16);
        patched += 1;
    }
    assert!(
        patched > 0,
        "validated int8 regcmd is missing register 0x{reg:04x}"
    );
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

/// Encode the hardware-validated research W8A8 multi-row tile.
/// Production still routes M=16 by default. M=32/48/64 are exposed only through
/// explicit research routing until the wider path is promoted.
pub fn encode_int8_mtile(
    m: usize,
    desc: Int8DecodeDesc,
) -> Result<[u64; INT8_REGCMD_COUNT], Int8EncodeError> {
    if !matches!(m, 16 | 32 | 48 | 64) || desc.k == 0 || desc.n == 0 {
        return Err(Int8EncodeError::InvalidShape {
            k: desc.k,
            n: desc.n,
            reason: "research M-tile requires M in {16,32,48,64} and non-zero K/N",
        });
    }
    if !desc.k.is_multiple_of(512) || desc.k > 4096 {
        return Err(Int8EncodeError::InvalidShape {
            k: desc.k,
            n: desc.n,
            reason: "M-tile full-K gate requires K%512==0 and K<=4096",
        });
    }
    if !desc.n.is_multiple_of(32) || desc.n > RK3588_NMAX {
        return Err(Int8EncodeError::InvalidShape {
            k: desc.k,
            n: desc.n,
            reason: "M-tile requires N%32==0 and N<=8192",
        });
    }

    let input_dma = address32(desc.input_dma)?;
    let weights_dma = address32(desc.weights_dma)?;
    let output_dma = address32(desc.output_dma)?;
    let m_u32 = u32::try_from(m).map_err(|_| Int8EncodeError::SizeOverflow)?;
    let k = u32::try_from(desc.k).map_err(|_| Int8EncodeError::SizeOverflow)?;
    let n = u32::try_from(desc.n).map_err(|_| Int8EncodeError::SizeOverflow)?;
    let weight_elems = u32::try_from(
        desc.k
            .checked_mul(desc.n)
            .ok_or(Int8EncodeError::SizeOverflow)?,
    )
    .map_err(|_| Int8EncodeError::SizeOverflow)?;

    let mut ops = ork_isc::INT8_TEMPLATE;
    patch_register(&mut ops, 0x1024, ((k - 1) << 16) | k);
    patch_register(&mut ops, 0x1030, weight_elems);
    patch_register(&mut ops, 0x1034, k);
    patch_register(&mut ops, 0x1044, k.div_ceil(64));
    patch_register(&mut ops, 0x107c, k / 16);
    patch_register(&mut ops, 0x1084, 0x0001_0000 | m_u32);
    patch_register(&mut ops, 0x1088, k);
    patch_register(&mut ops, 0x1020, 0x0001_0000 | m_u32);
    patch_register(&mut ops, 0x102c, m_u32);

    patch_register(&mut ops, 0x1038, 0x0101_0000 | n);
    patch_register(&mut ops, 0x3018, n - 1);
    patch_register(&mut ops, 0x403c, ((n - 1) << 16) | (n - 1));
    patch_register(&mut ops, 0x4058, n - 1);
    patch_register(&mut ops, 0x4038, (((n / 4) - 1) << 16) | ((n / 4) - 1));

    let mut r = (2 * RK3588_CBUF_ELEMS) / desc.k;
    r = r.max(1);
    let mut r_pow2 = 1usize;
    while r_pow2.saturating_mul(2) <= r {
        r_pow2 *= 2;
    }
    let rows = (m + 1).min(r_pow2);
    patch_register(
        &mut ops,
        0x1010,
        u32::try_from(16 * rows).map_err(|_| Int8EncodeError::SizeOverflow)?,
    );

    let scale = desc.k / 512;
    let base =
        177i32 - 15i32 * (i32::try_from(scale).map_err(|_| Int8EncodeError::SizeOverflow)? - 1);
    let slope = 15i32 * i32::try_from(scale).map_err(|_| Int8EncodeError::SizeOverflow)?;
    let mg = m.div_ceil(64).max(1);
    let v = (base - slope * (i32::try_from(mg).map_err(|_| Int8EncodeError::SizeOverflow)? - 1))
        .max(0x1b) as u32;
    patch_register_all(&mut ops, 0x1040, v);

    patch_register(&mut ops, 0x1070, input_dma);
    patch_register(&mut ops, 0x1110, weights_dma);
    patch_register(&mut ops, 0x4020, output_dma);

    let m_minus_1 = m_u32 - 1;
    patch_register(&mut ops, 0x4034, m_minus_1);
    patch_register(&mut ops, 0x405c, m_minus_1 << 16);
    patch_register(&mut ops, 0x3014, m_minus_1 << 16);
    Ok(ops)
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
    if !desc.k.is_multiple_of(256) || desc.k > 4096 {
        return Err(Int8EncodeError::InvalidShape {
            k: desc.k,
            n: desc.n,
            reason: "M=1 full-K Rocket gate requires K%256==0 and K<=4096",
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

    let mut ops = ork_isc::INT8_TEMPLATE;

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

#[cfg(test)]
mod tests {
    use super::*;

    fn reg_value(ops: &[u64], reg: u16) -> u32 {
        let word = ops.iter().find(|word| (**word as u16) == reg).unwrap();
        ((*word >> 16) & 0xffff_ffff) as u32
    }

    #[test]
    fn mtile_updates_every_repeated_reduction_schedule_register() {
        let ops = encode_int8_mtile(
            64,
            Int8DecodeDesc::new(2048, 2048, 0x1111_1000, 0x2222_2000, 0x3333_3000),
        )
        .unwrap();
        let values = ops
            .iter()
            .filter(|word| (**word as u16) == 0x1040)
            .map(|word| ((*word >> 16) & 0xffff_ffff) as u32)
            .collect::<Vec<_>>();
        assert!(
            values.len() >= 2,
            "INT8 template must retain repeated 0x1040 writes"
        );
        assert!(values.iter().all(|&value| value == values[0]));
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
    fn m1_k256_n64_geometry_is_encodable() {
        let ops = encode_int8_decode_m1(Int8DecodeDesc::new(
            256,
            64,
            0x1111_1000,
            0x2222_2000,
            0x3333_3000,
        ))
        .unwrap();
        assert_eq!(reg_value(&ops, 0x1024), 0x00ff_0100);
        assert_eq!(reg_value(&ops, 0x1030), 256 * 64);
        assert_eq!(reg_value(&ops, 0x1040), 0xc0);
        assert_eq!(reg_value(&ops, 0x107c), 16);
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
