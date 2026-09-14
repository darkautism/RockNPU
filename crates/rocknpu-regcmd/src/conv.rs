use crate::general::{Fp16MatmulDesc, encode_fp16_matmul, patch_register};
use crate::{EncodeError, FP16_REGCMD_COUNT};

const CBUF_BANK_SIZE: u64 = 32 * 1024;
const CBUF_BANKS: u32 = 12;

/// First direct RK3588 FP16 Conv2D descriptor.
///
/// This intentionally starts with the subset required by the first hardware
/// vertical slice and MNIST-8: direct (non-depthwise), dilation=1, stride=1,
/// symmetric top/left padding, IC multiple of 32 and OC multiple of 16.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp16Conv2dDesc {
    pub input_h: usize,
    pub input_w: usize,
    pub input_channels: usize,
    pub output_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub pad_top: usize,
    pub pad_left: usize,
    pub stride_y: usize,
    pub stride_x: usize,
    pub input_dma: u64,
    pub weights_dma: u64,
    pub output_dma: u64,
}

impl Fp16Conv2dDesc {
    pub const fn output_h(self) -> usize {
        (self.input_h + 2 * self.pad_top - self.kernel_h) / self.stride_y + 1
    }
    pub const fn output_w(self) -> usize {
        (self.input_w + 2 * self.pad_left - self.kernel_w) / self.stride_x + 1
    }
}

#[inline]
fn div_ceil_u64(n: u64, d: u64) -> u64 {
    n.div_ceil(d)
}

/// Encode one direct FP16 Conv2D task.
///
/// The DPU/RDMA precision/bypass program is the already hardware-validated
/// FP16 MatMul stream. A 1x1 MatMul is the same CNA direct-convolution datapath;
/// this function replaces only geometry/layout registers whose meanings are
/// independently documented in Mesa's MIT-licensed Rocket register/task code.
pub fn encode_fp16_conv2d(desc: Fp16Conv2dDesc) -> Result<[u64; FP16_REGCMD_COUNT], EncodeError> {
    let ih = desc.input_h;
    let iw = desc.input_w;
    let ic = desc.input_channels;
    let oc = desc.output_channels;
    let kh = desc.kernel_h;
    let kw = desc.kernel_w;
    if ih < 4 || iw == 0 || ic == 0 || oc == 0 || kh == 0 || kw == 0 {
        return Err(EncodeError::InvalidShape {
            m: ih.saturating_mul(iw),
            k: ic,
            n: oc,
            reason: "Conv dimensions must be nonzero and input_h>=4",
        });
    }
    if ic % 32 != 0 || oc % 16 != 0 {
        return Err(EncodeError::InvalidShape {
            m: ih.saturating_mul(iw),
            k: ic,
            n: oc,
            reason: "direct fp16 Conv requires IC%32==0 and OC%16==0",
        });
    }
    if kh > 31
        || kw > 31
        || desc.pad_top > 15
        || desc.pad_left > 15
        || !(1..=2).contains(&desc.stride_y)
        || !(1..=2).contains(&desc.stride_x)
    {
        return Err(EncodeError::InvalidShape {
            m: ih.saturating_mul(iw),
            k: ic,
            n: oc,
            reason: "kernel/padding/stride exceeds CNA register field",
        });
    }
    if ih + 2 * desc.pad_top < kh || iw + 2 * desc.pad_left < kw {
        return Err(EncodeError::InvalidShape {
            m: ih.saturating_mul(iw),
            k: ic,
            n: oc,
            reason: "kernel larger than padded input",
        });
    }
    let oh = desc.output_h();
    let ow = desc.output_w();
    let spatial = oh.checked_mul(ow).ok_or(EncodeError::SizeOverflow)?;
    if spatial == 0 || spatial % 4 != 0 {
        return Err(EncodeError::InvalidShape {
            m: spatial,
            k: ic,
            n: oc,
            reason: "current Conv vertical slice requires output spatial %4==0",
        });
    }
    if ih > 0x7ff
        || iw > 0x7ff
        || oh > 0x1fff
        || ow > 0x1fff
        || ic > u16::MAX as usize
        || oc > 0x2000
    {
        return Err(EncodeError::InvalidShape {
            m: spatial,
            k: ic,
            n: oc,
            reason: "Conv geometry exceeds register fields",
        });
    }

    let weight_bytes_per_kernel = (kh as u64)
        .checked_mul(kw as u64)
        .and_then(|v| v.checked_mul(ic as u64))
        .and_then(|v| v.checked_mul(2))
        .ok_or(EncodeError::SizeOverflow)?;
    if weight_bytes_per_kernel > CBUF_BANK_SIZE {
        return Err(EncodeError::CbufUnsupported {
            feature_banks: 0,
            weight_banks: 0,
            reason: "one Conv kernel exceeds one CBUF bank; IC/kernel tiling required",
        });
    }
    let weight_bytes = weight_bytes_per_kernel
        .checked_mul(oc as u64)
        .ok_or(EncodeError::SizeOverflow)?;
    let feature_bytes = (ih as u64)
        .checked_mul(iw as u64)
        .and_then(|v| v.checked_mul(ic as u64))
        .and_then(|v| v.checked_mul(2))
        .ok_or(EncodeError::SizeOverflow)?;
    let feature_banks = div_ceil_u64(feature_bytes, CBUF_BANK_SIZE) as u32;
    let weight_banks = div_ceil_u64(weight_bytes, CBUF_BANK_SIZE) as u32;
    if feature_banks + weight_banks > CBUF_BANKS {
        return Err(EncodeError::CbufUnsupported {
            feature_banks,
            weight_banks,
            reason: "Conv input and weight tensors exceed single-task CBUF budget",
        });
    }
    let data_bank = feature_banks;
    let weight_bank = CBUF_BANKS - feature_banks;
    let weight_bytes_u32 = u32::try_from(weight_bytes).map_err(|_| EncodeError::SizeOverflow)?;
    let wpk = u32::try_from(weight_bytes_per_kernel).map_err(|_| EncodeError::SizeOverflow)?;

    // Start from the already validated direct-FP16 CNA/CORE/DPU/RDMA program.
    // MatMul is used only as the fixed 126-op FP16 template here. Use the smallest
    // legal dummy M so Conv geometry is not accidentally constrained by MatMul's
    // FEATURE_GRAINS interpretation; every M/spatial/CBUF-dependent register is
    // explicitly replaced below with Conv semantics.
    let mut ops = encode_fp16_matmul(Fp16MatmulDesc::new(
        4,
        ic,
        oc,
        desc.input_dma,
        desc.weights_dma,
        desc.output_dma,
    ))?;

    let feature_grains = u32::try_from(ih + 1).map_err(|_| EncodeError::SizeOverflow)?;
    let input_line_stride = u32::try_from(iw.checked_mul(4).ok_or(EncodeError::SizeOverflow)?)
        .map_err(|_| EncodeError::SizeOverflow)?;
    let surf_num = input_line_stride
        .checked_mul(u32::try_from(ih - 4).map_err(|_| EncodeError::SizeOverflow)?)
        .ok_or(EncodeError::SizeOverflow)?;
    if surf_num % 4 != 0 {
        return Err(EncodeError::InvalidShape {
            m: spatial,
            k: ic,
            n: oc,
            reason: "input surface stride is not representable for current FP16 geometry",
        });
    }
    let input_surface_stride = surf_num / 4;
    let data_entries = u32::try_from(
        (iw as u64)
            .checked_mul(ic as u64)
            .and_then(|v| v.checked_mul(2))
            .ok_or(EncodeError::SizeOverflow)?
            .div_ceil(64),
    )
    .map_err(|_| EncodeError::SizeOverflow)?;
    let core_h = u32::try_from(oh - 1).map_err(|_| EncodeError::SizeOverflow)?;
    let core_w = u32::try_from(ow - 1).map_err(|_| EncodeError::SizeOverflow)?;
    let core_c = u32::try_from(oc - 1).map_err(|_| EncodeError::SizeOverflow)?;
    let out_surface = u32::try_from(spatial.checked_mul(16).ok_or(EncodeError::SizeOverflow)?)
        .map_err(|_| EncodeError::SizeOverflow)?;
    let out_pair_surface = u32::try_from(spatial.checked_mul(32).ok_or(EncodeError::SizeOverflow)?)
        .map_err(|_| EncodeError::SizeOverflow)?;

    // CNA geometry. Field semantics are from Mesa registers.xml/rkt_task/rkt_regcmd.
    // FEATURE_GRAINS=IH+1 is an empirical RK3588 FP16 relation, independently
    // checked across multiple IH/stride/kernel probes and then hardware-gated.
    patch_register(&mut ops, 0x1010, (feature_grains & 0x3ff) << 4);
    patch_register(
        &mut ops,
        0x1014,
        ((desc.stride_y as u32) << 3) | (desc.stride_x as u32),
    );
    patch_register(&mut ops, 0x1020, ((iw as u32) << 16) | (ih as u32));
    patch_register(&mut ops, 0x1028, ow as u32);
    patch_register(&mut ops, 0x102c, spatial as u32);
    patch_register(&mut ops, 0x1030, weight_bytes_u32);
    patch_register(&mut ops, 0x1034, wpk & 0x7ffff);
    patch_register(
        &mut ops,
        0x1038,
        ((kw as u32) << 24) | ((kh as u32) << 16) | (oc as u32),
    );
    patch_register(&mut ops, 0x1040, (weight_bank << 4) | data_bank);
    patch_register(&mut ops, 0x1044, data_entries & 0x3fff);
    patch_register(
        &mut ops,
        0x1068,
        ((desc.pad_left as u32) << 4) | (desc.pad_top as u32),
    );
    patch_register(&mut ops, 0x107c, input_line_stride);
    patch_register(&mut ops, 0x1080, input_surface_stride);
    patch_register(&mut ops, 0x1084, ((iw as u32) << 16) | (ih as u32));

    // CORE + DPU + RDMA output cube geometry.
    patch_register(&mut ops, 0x3014, (core_h << 16) | core_w);
    patch_register(&mut ops, 0x3018, core_c);
    patch_register(&mut ops, 0x4024, out_surface);
    patch_register(&mut ops, 0x4030, core_w);
    patch_register(&mut ops, 0x4034, core_h);
    patch_register(&mut ops, 0x403c, (core_c << 16) | core_c);
    patch_register(&mut ops, 0x4058, core_c);
    patch_register(&mut ops, 0x405c, (core_h << 16) | core_w);
    patch_register(&mut ops, 0x40c0, out_pair_surface);
    patch_register(&mut ops, 0x500c, core_w);
    patch_register(&mut ops, 0x5010, core_h);
    patch_register(&mut ops, 0x5014, core_c);
    Ok(ops)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn reg(ops: &[u64; FP16_REGCMD_COUNT], r: u16) -> u32 {
        let w = *ops.iter().find(|w| (**w as u16) == r).unwrap();
        ((w >> 16) & 0xffff_ffff) as u32
    }
    #[test]
    fn k5_8x8_geometry_matches_documented_fields() {
        let d = Fp16Conv2dDesc {
            input_h: 8,
            input_w: 8,
            input_channels: 32,
            output_channels: 16,
            kernel_h: 5,
            kernel_w: 5,
            pad_top: 2,
            pad_left: 2,
            stride_y: 1,
            stride_x: 1,
            input_dma: 0x100000,
            weights_dma: 0x200000,
            output_dma: 0x300000,
        };
        let o = encode_fp16_conv2d(d).unwrap();
        assert_eq!(reg(&o, 0x1010), 0x90);
        assert_eq!(reg(&o, 0x1020), 0x0008_0008);
        assert_eq!(reg(&o, 0x1030), 25 * 32 * 16 * 2);
        assert_eq!(reg(&o, 0x1034), 25 * 32 * 2);
        assert_eq!(reg(&o, 0x1038), 0x0505_0010);
        assert_eq!(reg(&o, 0x1068), 0x22);
        assert_eq!(reg(&o, 0x3014), 0x0007_0007);
        assert_eq!(reg(&o, 0x4024), 8 * 8 * 16);
    }
    #[test]
    fn stride2_updates_conv_and_output_geometry() {
        let d = Fp16Conv2dDesc {
            input_h: 8,
            input_w: 12,
            input_channels: 32,
            output_channels: 16,
            kernel_h: 5,
            kernel_w: 5,
            pad_top: 2,
            pad_left: 2,
            stride_y: 2,
            stride_x: 2,
            input_dma: 0x100000,
            weights_dma: 0x200000,
            output_dma: 0x300000,
        };
        let o = encode_fp16_conv2d(d).unwrap();
        assert_eq!(d.output_h(), 4);
        assert_eq!(d.output_w(), 6);
        assert_eq!(reg(&o, 0x1014), 0x12);
        assert_eq!(reg(&o, 0x1028), 6);
        assert_eq!(reg(&o, 0x102c), 24);
        assert_eq!(reg(&o, 0x3014), 0x0003_0005);
        assert_eq!(reg(&o, 0x4024), 24 * 16);
    }

    #[test]
    fn spatial_32x32_is_not_limited_by_matmul_feature_grains() {
        let d = Fp16Conv2dDesc {
            input_h: 32,
            input_w: 32,
            input_channels: 32,
            output_channels: 16,
            kernel_h: 3,
            kernel_w: 3,
            pad_top: 1,
            pad_left: 1,
            stride_y: 1,
            stride_x: 1,
            input_dma: 0x100000,
            weights_dma: 0x200000,
            output_dma: 0x300000,
        };
        let o = encode_fp16_conv2d(d).unwrap();
        assert_eq!(d.output_h() * d.output_w(), 1024);
        assert_eq!(reg(&o, 0x1010), 33 << 4);
        assert_eq!(reg(&o, 0x102c), 1024);
        assert_eq!(reg(&o, 0x3014), 0x001f_001f);
        assert_eq!(reg(&o, 0x4024), 1024 * 16);
    }

    #[test]
    fn rejects_unaligned_channels() {
        let mut d = Fp16Conv2dDesc {
            input_h: 8,
            input_w: 8,
            input_channels: 32,
            output_channels: 16,
            kernel_h: 5,
            kernel_w: 5,
            pad_top: 2,
            pad_left: 2,
            stride_y: 1,
            stride_x: 1,
            input_dma: 0x1000,
            weights_dma: 0x2000,
            output_dma: 0x3000,
        };
        d.input_channels = 8;
        assert!(matches!(
            encode_fp16_conv2d(d),
            Err(EncodeError::InvalidShape { .. })
        ));
    }
}
