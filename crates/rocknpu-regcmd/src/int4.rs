// SPDX-License-Identifier: MIT
//
//! RK3588 native W4A4 single-program M=1 register-command encoder.
//!
//! The source-derived baseline template is isolated in `int4/ork_isc.rs` under
//! ork-driver's ISC notice. Geometry/address patching here is RockNPU MIT code.

mod ork_isc;

pub const INT4_REGCMD_COUNT: usize = 116;
const RK3588_INT4_KMAX: usize = 10_752;
const RK3588_NMAX: usize = 8_192;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Int4DecodeDesc {
    pub k: usize,
    pub n: usize,
    pub input_dma: u64,
    pub weights_dma: u64,
    pub output_dma: u64,
}

impl Int4DecodeDesc {
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
pub enum Int4EncodeError {
    AddressAbove32Bit(u64),
    InvalidShape {
        k: usize,
        n: usize,
        reason: &'static str,
    },
    SizeOverflow,
}

impl core::fmt::Display for Int4EncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AddressAbove32Bit(v) => write!(
                f,
                "NPU IOVA 0x{v:x} exceeds RK3588 regcmd 32-bit address field"
            ),
            Self::InvalidShape { k, n, reason } => write!(
                f,
                "unsupported native W4A4 decode shape M=1 K={k} N={n}: {reason}"
            ),
            Self::SizeOverflow => write!(f, "native W4A4 decode size arithmetic overflow"),
        }
    }
}
impl std::error::Error for Int4EncodeError {}

fn address32(v: u64) -> Result<u32, Int4EncodeError> {
    u32::try_from(v).map_err(|_| Int4EncodeError::AddressAbove32Bit(v))
}

fn baseline() -> [u64; INT4_REGCMD_COUNT] {
    let mut out = [0u64; INT4_REGCMD_COUNT];
    let mut i = 0;
    while i < INT4_REGCMD_COUNT {
        out[i] = u64::from(ork_isc::INT4_TEMPLATE_U32[i * 2])
            | (u64::from(ork_isc::INT4_TEMPLATE_U32[i * 2 + 1]) << 32);
        i += 1;
    }
    out
}

fn patch_register(ops: &mut [u64; INT4_REGCMD_COUNT], reg: u16, value: u32) {
    let word = ops
        .iter_mut()
        .find(|word| (**word as u16) == reg)
        .unwrap_or_else(|| panic!("validated int4 regcmd is missing register 0x{reg:04x}"));
    *word = (*word & 0xffff_0000_0000_ffff) | (u64::from(value) << 16);
}

/// Current ork-derived native W4A4 M=1 single-program envelope.
/// Output is a dense linear `int16[N]` accumulator surface.
pub fn encode_int4_decode_m1(
    desc: Int4DecodeDesc,
) -> Result<[u64; INT4_REGCMD_COUNT], Int4EncodeError> {
    if desc.k == 0 || desc.n == 0 {
        return Err(Int4EncodeError::InvalidShape {
            k: desc.k,
            n: desc.n,
            reason: "dimensions must be non-zero",
        });
    }
    if !desc.k.is_multiple_of(32) || desc.k > RK3588_INT4_KMAX {
        return Err(Int4EncodeError::InvalidShape {
            k: desc.k,
            n: desc.n,
            reason: "requires K%32==0 and K<=10752",
        });
    }
    if !desc.n.is_multiple_of(64) || desc.n > RK3588_NMAX {
        return Err(Int4EncodeError::InvalidShape {
            k: desc.k,
            n: desc.n,
            reason: "single program requires N%64==0 and N<=8192",
        });
    }

    let input_dma = address32(desc.input_dma)?;
    let weights_dma = address32(desc.weights_dma)?;
    let output_dma = address32(desc.output_dma)?;
    let k = u32::try_from(desc.k).map_err(|_| Int4EncodeError::SizeOverflow)?;
    let n = u32::try_from(desc.n).map_err(|_| Int4EncodeError::SizeOverflow)?;
    let weight_bytes = desc
        .k
        .checked_mul(desc.n)
        .and_then(|v| v.checked_div(2))
        .ok_or(Int4EncodeError::SizeOverflow)?;
    let weight_bytes = u32::try_from(weight_bytes).map_err(|_| Int4EncodeError::SizeOverflow)?;

    let mut ops = baseline();
    patch_register(&mut ops, 0x1024, ((k - 1) << 16) | k);
    patch_register(&mut ops, 0x1030, weight_bytes);
    patch_register(&mut ops, 0x1034, k / 2);
    patch_register(&mut ops, 0x1044, k.div_ceil(128));
    patch_register(&mut ops, 0x1088, k);
    patch_register(&mut ops, 0x1038, 0x0101_0000 | n);
    patch_register(&mut ops, 0x3018, n - 1);
    patch_register(&mut ops, 0x403c, ((n - 1) << 16) | (n - 1));
    patch_register(&mut ops, 0x4058, n - 1);
    patch_register(&mut ops, 0x1070, input_dma);
    patch_register(&mut ops, 0x1110, weights_dma);
    patch_register(&mut ops, 0x4020, output_dma);
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
    fn tinyllama_projection_geometry() {
        let ops = encode_int4_decode_m1(Int4DecodeDesc::new(
            2048,
            2048,
            0x1111_1000,
            0x2222_2000,
            0x3333_3000,
        ))
        .unwrap();
        assert_eq!(ops.len(), INT4_REGCMD_COUNT);
        assert_eq!(reg_value(&ops, 0x1024), 0x07ff_0800);
        assert_eq!(reg_value(&ops, 0x1030), 2048 * 2048 / 2);
        assert_eq!(reg_value(&ops, 0x1034), 1024);
        assert_eq!(reg_value(&ops, 0x1044), 16);
        assert_eq!(reg_value(&ops, 0x1038), 0x0101_0800);
        assert_eq!(reg_value(&ops, 0x1070), 0x1111_1000);
        assert_eq!(reg_value(&ops, 0x1110), 0x2222_2000);
        assert_eq!(reg_value(&ops, 0x4020), 0x3333_3000);
    }

    #[test]
    fn rejects_shapes_outside_single_program_envelope() {
        for (k, n) in [(0, 64), (33, 64), (10784, 64), (2048, 96), (2048, 8256)] {
            assert!(encode_int4_decode_m1(Int4DecodeDesc::new(k, n, 1, 2, 3)).is_err());
        }
    }
}
