use half::f16;
use rocket_runtime::{RocketBuffer, RocketDevice, Task};
use rocknpu_regcmd::{Fp16Conv2dDesc, encode_fp16_conv2d, weight_conv_fp16};
use std::time::Instant;
use std::{fmt, io};

const WAIT_NS: i64 = 2_000_000_000;
const REGCMD_BYTES: usize = 4096;

#[derive(Debug)]
pub enum ConvError {
    InvalidInput(&'static str),
    Io(io::Error),
    Encode(rocknpu_regcmd::EncodeError),
    AddressAbove32Bit(u64),
    Internal(&'static str),
}
impl fmt::Display for ConvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(s) => write!(f, "invalid fp16 Conv2D input: {s}"),
            Self::Io(e) => e.fmt(f),
            Self::Encode(e) => e.fmt(f),
            Self::AddressAbove32Bit(v) => {
                write!(f, "Rocket BO IOVA 0x{v:x} exceeds 32-bit Conv regcmd field")
            }
            Self::Internal(s) => write!(f, "fp16 Conv2D executor invariant failed: {s}"),
        }
    }
}
impl std::error::Error for ConvError {}
impl From<io::Error> for ConvError {
    fn from(v: io::Error) -> Self {
        Self::Io(v)
    }
}
impl From<rocknpu_regcmd::EncodeError> for ConvError {
    fn from(v: rocknpu_regcmd::EncodeError) -> Self {
        Self::Encode(v)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv2dSpec {
    pub ic: usize,
    pub ih: usize,
    pub iw: usize,
    pub oc: usize,
    pub kh: usize,
    pub kw: usize,
    pub pad_top: usize,
    pub pad_left: usize,
    pub stride_y: usize,
    pub stride_x: usize,
}
impl Conv2dSpec {
    pub const fn output_h(self) -> usize {
        (self.ih + 2 * self.pad_top - self.kh) / self.stride_y + 1
    }
    pub const fn output_w(self) -> usize {
        (self.iw + 2 * self.pad_left - self.kw) / self.stride_x + 1
    }
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConvScratchStats {
    pub allocations: usize,
    pub grows: usize,
    pub regcmd_bytes: usize,
    pub input_bytes: usize,
    pub weight_bytes: usize,
    pub output_bytes: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConvExecutionTiming {
    pub scratch_ns: u128,
    pub input_pack_ns: u128,
    pub output_clear_ns: u128,
    pub encode_ns: u128,
    pub regcmd_write_ns: u128,
    pub submit_ns: u128,
    pub wait_ns: u128,
    pub gather_ns: u128,
    pub total_ns: u128,
    pub jobs_submitted: usize,
}
impl ConvExecutionTiming {
    pub const fn accounted_ns(self) -> u128 {
        self.scratch_ns
            + self.input_pack_ns
            + self.output_clear_ns
            + self.encode_ns
            + self.regcmd_write_ns
            + self.submit_ns
            + self.wait_ns
            + self.gather_ns
    }
    pub const fn overhead_ns(self) -> u128 {
        self.total_ns.saturating_sub(self.accounted_ns())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PreparedConvWeightStats {
    pub resident_bytes: usize,
    pub pack_ns: u128,
}

pub struct Fp16PreparedConvWeights<'d> {
    device_fd: i32,
    spec: Conv2dSpec,
    padded_ic: usize,
    padded_oc: usize,
    bo: RocketBuffer<'d>,
    stats: PreparedConvWeightStats,
}
impl Fp16PreparedConvWeights<'_> {
    pub const fn spec(&self) -> Conv2dSpec {
        self.spec
    }
    pub const fn padded_ic(&self) -> usize {
        self.padded_ic
    }
    pub const fn padded_oc(&self) -> usize {
        self.padded_oc
    }
    pub const fn stats(&self) -> PreparedConvWeightStats {
        self.stats
    }
    pub fn dma_address(&self) -> u64 {
        self.bo.dma_address()
    }
    pub fn handle(&self) -> u32 {
        self.bo.handle()
    }
}

struct Scratch<'d> {
    regcmd: Option<RocketBuffer<'d>>,
    input: Option<RocketBuffer<'d>>,
    weights: Option<RocketBuffer<'d>>,
    output: Option<RocketBuffer<'d>>,
    allocations: usize,
    grows: usize,
}
impl Scratch<'_> {
    fn stats(&self) -> ConvScratchStats {
        ConvScratchStats {
            allocations: self.allocations,
            grows: self.grows,
            regcmd_bytes: self.regcmd.as_ref().map_or(0, |x| x.len()),
            input_bytes: self.input.as_ref().map_or(0, |x| x.len()),
            weight_bytes: self.weights.as_ref().map_or(0, |x| x.len()),
            output_bytes: self.output.as_ref().map_or(0, |x| x.len()),
        }
    }
}

pub struct Fp16Conv2dExecutor<'d> {
    device: &'d RocketDevice,
    _guard: RocketBuffer<'d>,
    scratch: Scratch<'d>,
}
impl<'d> Fp16Conv2dExecutor<'d> {
    pub fn new(device: &'d RocketDevice) -> Result<Self, ConvError> {
        let guard = device.alloc_buffer(4096)?;
        Ok(Self {
            device,
            _guard: guard,
            scratch: Scratch {
                regcmd: None,
                input: None,
                weights: None,
                output: None,
                allocations: 0,
                grows: 0,
            },
        })
    }
    pub fn scratch_stats(&self) -> ConvScratchStats {
        self.scratch.stats()
    }
    pub fn prepare_weights(
        &self,
        weights: &[f16],
        spec: Conv2dSpec,
    ) -> Result<Fp16PreparedConvWeights<'d>, ConvError> {
        validate_spec(spec)?;
        let expected = spec
            .oc
            .checked_mul(spec.ic)
            .and_then(|v| v.checked_mul(spec.kh))
            .and_then(|v| v.checked_mul(spec.kw))
            .ok_or(ConvError::InvalidInput("weight length overflow"))?;
        if weights.len() != expected {
            return Err(ConvError::InvalidInput("weight length mismatch"));
        }
        let pic = align_up(spec.ic, 32)?;
        let poc = align_up(spec.oc, 16)?;
        let wt_bytes = poc
            .checked_mul(pic)
            .and_then(|v| v.checked_mul(spec.kh))
            .and_then(|v| v.checked_mul(spec.kw))
            .and_then(|v| v.checked_mul(2))
            .ok_or(ConvError::InvalidInput(
                "resident weight cube size overflow",
            ))?;
        let mut bo = self.device.alloc_buffer(wt_bytes.max(1))?;
        let last = bo
            .dma_address()
            .checked_add((wt_bytes.max(1) - 1) as u64)
            .ok_or(ConvError::AddressAbove32Bit(u64::MAX))?;
        if last > u32::MAX as u64 {
            return Err(ConvError::AddressAbove32Bit(last));
        }
        let start = Instant::now();
        bo.prep_relative(0)?;
        bo.as_mut_slice().fill(0);
        pack_weights_native(bo.as_mut_slice(), weights, spec, pic, poc);
        bo.fini()?;
        Ok(Fp16PreparedConvWeights {
            device_fd: self.device.fd(),
            spec,
            padded_ic: pic,
            padded_oc: poc,
            bo,
            stats: PreparedConvWeightStats {
                resident_bytes: wt_bytes,
                pack_ns: start.elapsed().as_nanos(),
            },
        })
    }

    pub fn execute_prepared(
        &mut self,
        input: &[f16],
        prepared: &Fp16PreparedConvWeights<'d>,
    ) -> Result<Vec<f16>, ConvError> {
        if prepared.device_fd != self.device.fd() {
            return Err(ConvError::InvalidInput(
                "prepared Conv weights belong to a different Rocket fd",
            ));
        }
        let spec = prepared.spec;
        validate_spec(spec)?;
        let expected = spec
            .ic
            .checked_mul(spec.ih)
            .and_then(|v| v.checked_mul(spec.iw))
            .ok_or(ConvError::InvalidInput("input length overflow"))?;
        if input.len() != expected {
            return Err(ConvError::InvalidInput("input length mismatch"));
        }
        let pic = prepared.padded_ic;
        let poc = prepared.padded_oc;
        let oh = spec.output_h();
        let ow = spec.output_w();
        let in_bytes = pic
            .checked_mul(spec.ih)
            .and_then(|v| v.checked_mul(spec.iw))
            .and_then(|v| v.checked_mul(2))
            .ok_or(ConvError::InvalidInput("input cube size overflow"))?;
        let out_bytes = poc
            .checked_mul(oh)
            .and_then(|v| v.checked_mul(ow))
            .and_then(|v| v.checked_mul(2))
            .ok_or(ConvError::InvalidInput("output cube size overflow"))?;
        ensure(
            self.device,
            &mut self.scratch.regcmd,
            REGCMD_BYTES,
            &mut self.scratch.allocations,
            &mut self.scratch.grows,
        )?;
        ensure(
            self.device,
            &mut self.scratch.input,
            in_bytes,
            &mut self.scratch.allocations,
            &mut self.scratch.grows,
        )?;
        ensure(
            self.device,
            &mut self.scratch.output,
            out_bytes,
            &mut self.scratch.allocations,
            &mut self.scratch.grows,
        )?;
        let Scratch {
            regcmd: Some(reg),
            input: Some(ib),
            output: Some(ob),
            ..
        } = &mut self.scratch
        else {
            return Err(ConvError::Internal("prepared scratch allocation"));
        };
        ib.prep_relative(0)?;
        ib.as_mut_slice().fill(0);
        pack_input_native(ib.as_mut_slice(), input, spec, pic);
        ib.fini()?;
        let tiles = encode_conv_tiles(
            spec,
            pic,
            poc,
            ib.dma_address(),
            prepared.bo.dma_address(),
            ob.dma_address(),
        )?;
        let out = execute_conv_tiles(self.device, reg, ib, prepared.bo.handle(), ob, spec, &tiles)?;
        Ok(out)
    }
    pub fn execute_prepared_profiled(
        &mut self,
        input: &[f16],
        prepared: &Fp16PreparedConvWeights<'d>,
    ) -> Result<(Vec<f16>, ConvExecutionTiming), ConvError> {
        let total_started = Instant::now();
        if prepared.device_fd != self.device.fd() {
            return Err(ConvError::InvalidInput(
                "prepared Conv weights belong to a different Rocket fd",
            ));
        }
        let spec = prepared.spec;
        validate_spec(spec)?;
        let expected = spec
            .ic
            .checked_mul(spec.ih)
            .and_then(|v| v.checked_mul(spec.iw))
            .ok_or(ConvError::InvalidInput("input length overflow"))?;
        if input.len() != expected {
            return Err(ConvError::InvalidInput("input length mismatch"));
        }
        let pic = prepared.padded_ic;
        let poc = prepared.padded_oc;
        let oh = spec.output_h();
        let ow = spec.output_w();
        let in_bytes = pic
            .checked_mul(spec.ih)
            .and_then(|v| v.checked_mul(spec.iw))
            .and_then(|v| v.checked_mul(2))
            .ok_or(ConvError::InvalidInput("input cube size overflow"))?;
        let out_bytes = poc
            .checked_mul(oh)
            .and_then(|v| v.checked_mul(ow))
            .and_then(|v| v.checked_mul(2))
            .ok_or(ConvError::InvalidInput("output cube size overflow"))?;
        let mut timing = ConvExecutionTiming::default();

        let phase = Instant::now();
        ensure(
            self.device,
            &mut self.scratch.regcmd,
            REGCMD_BYTES,
            &mut self.scratch.allocations,
            &mut self.scratch.grows,
        )?;
        ensure(
            self.device,
            &mut self.scratch.input,
            in_bytes,
            &mut self.scratch.allocations,
            &mut self.scratch.grows,
        )?;
        ensure(
            self.device,
            &mut self.scratch.output,
            out_bytes,
            &mut self.scratch.allocations,
            &mut self.scratch.grows,
        )?;
        timing.scratch_ns = phase.elapsed().as_nanos();
        let Scratch {
            regcmd: Some(reg),
            input: Some(ib),
            output: Some(ob),
            ..
        } = &mut self.scratch
        else {
            return Err(ConvError::Internal("prepared scratch allocation"));
        };

        let phase = Instant::now();
        ib.prep_relative(0)?;
        ib.as_mut_slice().fill(0);
        pack_input_native(ib.as_mut_slice(), input, spec, pic);
        ib.fini()?;
        timing.input_pack_ns = phase.elapsed().as_nanos();

        // Output is a write-only device destination.
        timing.output_clear_ns = 0;

        let phase = Instant::now();
        let tiles = encode_conv_tiles(
            spec,
            pic,
            poc,
            ib.dma_address(),
            prepared.bo.dma_address(),
            ob.dma_address(),
        )?;
        timing.encode_ns = phase.elapsed().as_nanos();
        timing.jobs_submitted = tiles.len();

        let mut out = vec![f16::ZERO; spec.oc * oh * ow];
        for tile in &tiles {
            let phase = Instant::now();
            write_regcmd(reg, &tile.ops)?;
            timing.regcmd_write_ns += phase.elapsed().as_nanos();
            let task = Task {
                regcmd: u32::try_from(reg.dma_address())
                    .map_err(|_| ConvError::AddressAbove32Bit(reg.dma_address()))?,
                regcmd_count: tile.ops.len() as u32,
            };

            let phase = Instant::now();
            self.device.submit(
                &[task],
                &[ib.handle(), prepared.bo.handle(), reg.handle()],
                &[ob.handle()],
            )?;
            timing.submit_ns += phase.elapsed().as_nanos();

            let phase = Instant::now();
            ob.prep_relative(WAIT_NS)?;
            timing.wait_ns += phase.elapsed().as_nanos();

            let phase = Instant::now();
            gather_output_tile(ob.as_slice(), spec, tile.channels, tile.oc0, &mut out);
            ob.fini()?;
            timing.gather_ns += phase.elapsed().as_nanos();
        }
        timing.total_ns = total_started.elapsed().as_nanos();
        Ok((out, timing))
    }

    pub fn execute(
        &mut self,
        input: &[f16],
        weights: &[f16],
        spec: Conv2dSpec,
    ) -> Result<Vec<f16>, ConvError> {
        validate(spec, input, weights)?;
        let pic = align_up(spec.ic, 32)?;
        let poc = align_up(spec.oc, 16)?;
        let oh = spec.output_h();
        let ow = spec.output_w();
        let in_bytes = pic
            .checked_mul(spec.ih)
            .and_then(|v| v.checked_mul(spec.iw))
            .and_then(|v| v.checked_mul(2))
            .ok_or(ConvError::InvalidInput("input cube size overflow"))?;
        let wt_bytes = poc
            .checked_mul(pic)
            .and_then(|v| v.checked_mul(spec.kh))
            .and_then(|v| v.checked_mul(spec.kw))
            .and_then(|v| v.checked_mul(2))
            .ok_or(ConvError::InvalidInput("weight cube size overflow"))?;
        let out_bytes = poc
            .checked_mul(oh)
            .and_then(|v| v.checked_mul(ow))
            .and_then(|v| v.checked_mul(2))
            .ok_or(ConvError::InvalidInput("output cube size overflow"))?;
        ensure(
            self.device,
            &mut self.scratch.regcmd,
            REGCMD_BYTES,
            &mut self.scratch.allocations,
            &mut self.scratch.grows,
        )?;
        ensure(
            self.device,
            &mut self.scratch.input,
            in_bytes,
            &mut self.scratch.allocations,
            &mut self.scratch.grows,
        )?;
        ensure(
            self.device,
            &mut self.scratch.weights,
            wt_bytes,
            &mut self.scratch.allocations,
            &mut self.scratch.grows,
        )?;
        ensure(
            self.device,
            &mut self.scratch.output,
            out_bytes,
            &mut self.scratch.allocations,
            &mut self.scratch.grows,
        )?;
        let Scratch {
            regcmd: Some(reg),
            input: Some(ib),
            weights: Some(wb),
            output: Some(ob),
            ..
        } = &mut self.scratch
        else {
            return Err(ConvError::Internal("scratch allocation"));
        };
        ib.prep_relative(0)?;
        ib.as_mut_slice().fill(0);
        pack_input_native(ib.as_mut_slice(), input, spec, pic);
        ib.fini()?;
        wb.prep_relative(0)?;
        wb.as_mut_slice().fill(0);
        pack_weights_native(wb.as_mut_slice(), weights, spec, pic, poc);
        wb.fini()?;
        let tiles = encode_conv_tiles(
            spec,
            pic,
            poc,
            ib.dma_address(),
            wb.dma_address(),
            ob.dma_address(),
        )?;
        let out = execute_conv_tiles(self.device, reg, ib, wb.handle(), ob, spec, &tiles)?;
        Ok(out)
    }
}

pub fn cpu_reference_fp16(
    input: &[f16],
    weights: &[f16],
    spec: Conv2dSpec,
) -> Result<Vec<f16>, ConvError> {
    validate(spec, input, weights)?;
    let oh = spec.output_h();
    let ow = spec.output_w();
    let mut out = vec![f16::ZERO; spec.oc * oh * ow];
    for oc in 0..spec.oc {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut acc = 0.0f32;
                for ic in 0..spec.ic {
                    for ky in 0..spec.kh {
                        let yy = oy * spec.stride_y + ky;
                        if yy < spec.pad_top {
                            continue;
                        }
                        let yy = yy - spec.pad_top;
                        if yy >= spec.ih {
                            continue;
                        }
                        for kx in 0..spec.kw {
                            let xx = ox * spec.stride_x + kx;
                            if xx < spec.pad_left {
                                continue;
                            }
                            let xx = xx - spec.pad_left;
                            if xx >= spec.iw {
                                continue;
                            }
                            acc += input[(ic * spec.ih + yy) * spec.iw + xx].to_f32()
                                * weights[((oc * spec.ic + ic) * spec.kh + ky) * spec.kw + kx]
                                    .to_f32();
                        }
                    }
                }
                out[(oc * oh + oy) * ow + ox] = f16::from_f32(acc);
            }
        }
    }
    Ok(out)
}
struct EncodedConvTile {
    oc0: usize,
    channels: usize,
    ops: [u64; rocknpu_regcmd::FP16_REGCMD_COUNT],
}

fn encode_conv_tiles(
    spec: Conv2dSpec,
    pic: usize,
    poc: usize,
    input_dma: u64,
    weights_dma: u64,
    output_dma: u64,
) -> Result<Vec<EncodedConvTile>, ConvError> {
    let base = Fp16Conv2dDesc {
        input_h: spec.ih,
        input_w: spec.iw,
        input_channels: pic,
        output_channels: poc,
        kernel_h: spec.kh,
        kernel_w: spec.kw,
        pad_top: spec.pad_top,
        pad_left: spec.pad_left,
        stride_y: spec.stride_y,
        stride_x: spec.stride_x,
        input_dma,
        weights_dma,
        output_dma,
    };
    match encode_fp16_conv2d(base) {
        Ok(ops) => {
            return Ok(vec![EncodedConvTile {
                oc0: 0,
                channels: poc,
                ops,
            }]);
        }
        Err(rocknpu_regcmd::EncodeError::CbufUnsupported { .. }) => {}
        Err(e) => return Err(e.into()),
    }

    // OC tiles are independent reductions and native weights are contiguous by
    // complete OC16 group, so this requires no accumulator or weight repacking.
    let mut tile_channels = poc.saturating_sub(16);
    while tile_channels >= 16 {
        let mut probe = base;
        probe.output_channels = tile_channels;
        if encode_fp16_conv2d(probe).is_ok() {
            break;
        }
        tile_channels -= 16;
    }
    if tile_channels < 16 {
        // Re-run the full descriptor to preserve the encoder's precise failure.
        return Err(encode_fp16_conv2d(base).unwrap_err().into());
    }

    let mut tiles = Vec::new();
    let mut oc0 = 0usize;
    while oc0 < poc {
        let channels = tile_channels.min(poc - oc0);
        debug_assert_eq!(oc0 % 16, 0);
        debug_assert_eq!(channels % 16, 0);
        let weight_elem_offset = weight_conv_fp16(poc, pic, spec.kh, spec.kw, oc0 + 1, 1, 1, 1);
        let weight_byte_offset = weight_elem_offset
            .checked_mul(2)
            .ok_or(ConvError::InvalidInput("Conv weight tile offset overflow"))?;
        let tile_weight_dma = weights_dma
            .checked_add(weight_byte_offset as u64)
            .ok_or(ConvError::AddressAbove32Bit(u64::MAX))?;
        let mut desc = base;
        desc.output_channels = channels;
        desc.weights_dma = tile_weight_dma;
        let ops = encode_fp16_conv2d(desc)?;
        tiles.push(EncodedConvTile { oc0, channels, ops });
        oc0 += channels;
    }
    Ok(tiles)
}

fn execute_conv_tiles(
    device: &RocketDevice,
    reg: &mut RocketBuffer<'_>,
    input: &RocketBuffer<'_>,
    weight_handle: u32,
    output: &mut RocketBuffer<'_>,
    spec: Conv2dSpec,
    tiles: &[EncodedConvTile],
) -> Result<Vec<f16>, ConvError> {
    let spatial = spec.output_h() * spec.output_w();
    let mut out = vec![f16::ZERO; spec.oc * spatial];
    for tile in tiles {
        write_regcmd(reg, &tile.ops)?;
        let task = Task {
            regcmd: u32::try_from(reg.dma_address())
                .map_err(|_| ConvError::AddressAbove32Bit(reg.dma_address()))?,
            regcmd_count: tile.ops.len() as u32,
        };
        device.submit(
            &[task],
            &[input.handle(), weight_handle, reg.handle()],
            &[output.handle()],
        )?;
        output.prep_relative(WAIT_NS)?;
        gather_output_tile(output.as_slice(), spec, tile.channels, tile.oc0, &mut out);
        output.fini()?;
    }
    Ok(out)
}

fn pack_input_native(dst: &mut [u8], input: &[f16], spec: Conv2dSpec, pic: usize) {
    debug_assert_eq!(pic % 8, 0);
    let spatial = spec.ih * spec.iw;
    // Native FP16 feature cube is NC1HWC2 with C2=8:
    // [channel_plane][spatial][channel_inner]. Input is NCHW channel-major.
    for c in 0..spec.ic {
        let plane = c / 8;
        let inner = c % 8;
        let src = &input[c * spatial..(c + 1) * spatial];
        let native_base = plane * spatial * 8 + inner;
        for (pos, &v) in src.iter().enumerate() {
            put_f16(dst, native_base + pos * 8, v);
        }
    }
}
fn pack_weights_native(dst: &mut [u8], weights: &[f16], spec: Conv2dSpec, pic: usize, poc: usize) {
    for oc in 0..spec.oc {
        for ic in 0..spec.ic {
            for ky in 0..spec.kh {
                for kx in 0..spec.kw {
                    put_f16(
                        dst,
                        weight_conv_fp16(
                            poc,
                            pic,
                            spec.kh,
                            spec.kw,
                            oc + 1,
                            ic + 1,
                            ky + 1,
                            kx + 1,
                        ),
                        weights[((oc * spec.ic + ic) * spec.kh + ky) * spec.kw + kx],
                    );
                }
            }
        }
    }
}
fn gather_output_tile(
    src: &[u8],
    spec: Conv2dSpec,
    tile_channels: usize,
    oc0: usize,
    dst: &mut [f16],
) {
    let spatial = spec.output_h() * spec.output_w();
    let logical = spec.oc.saturating_sub(oc0).min(tile_channels);
    for local_c in 0..logical {
        let plane = local_c / 8;
        let inner = local_c % 8;
        let native_base = plane * spatial * 8 + inner;
        let out_c = oc0 + local_c;
        let out = &mut dst[out_c * spatial..(out_c + 1) * spatial];
        for (pos, v) in out.iter_mut().enumerate() {
            *v = get_f16(src, native_base + pos * 8);
        }
    }
}

fn write_regcmd(bo: &mut RocketBuffer<'_>, ops: &[u64]) -> Result<(), ConvError> {
    if ops.len() * 8 > bo.len() {
        return Err(ConvError::Internal("regcmd BO too small"));
    }
    bo.prep_relative(0)?;
    bo.as_mut_slice().fill(0);
    for (d, v) in bo.as_mut_slice().chunks_exact_mut(8).zip(ops.iter()) {
        d.copy_from_slice(&v.to_le_bytes());
    }
    bo.fini()?;
    Ok(())
}
fn validate_spec(s: Conv2dSpec) -> Result<(), ConvError> {
    if s.ic == 0
        || s.oc == 0
        || s.ih < 4
        || s.iw == 0
        || s.kh == 0
        || s.kw == 0
        || !(1..=2).contains(&s.stride_y)
        || !(1..=2).contains(&s.stride_x)
    {
        return Err(ConvError::InvalidInput(
            "nonzero dimensions, IH>=4, and stride 1..=2 required",
        ));
    }
    if s.ih + 2 * s.pad_top < s.kh || s.iw + 2 * s.pad_left < s.kw {
        return Err(ConvError::InvalidInput("kernel exceeds padded input"));
    }
    Ok(())
}
fn validate(s: Conv2dSpec, input: &[f16], weights: &[f16]) -> Result<(), ConvError> {
    validate_spec(s)?;
    if input.len() != s.ic * s.ih * s.iw || weights.len() != s.oc * s.ic * s.kh * s.kw {
        return Err(ConvError::InvalidInput("input/weight length mismatch"));
    }
    Ok(())
}
fn align_up(v: usize, a: usize) -> Result<usize, ConvError> {
    v.checked_add(a - 1)
        .map(|x| (x / a) * a)
        .ok_or(ConvError::InvalidInput("alignment overflow"))
}
fn ensure<'d>(
    dev: &'d RocketDevice,
    slot: &mut Option<RocketBuffer<'d>>,
    need: usize,
    alloc: &mut usize,
    grows: &mut usize,
) -> Result<(), ConvError> {
    if slot.as_ref().is_some_and(|b| b.len() >= need) {
        return Ok(());
    }
    let replacing = slot.is_some();
    let bo = dev.alloc_buffer(need.max(1))?;
    let last = bo
        .dma_address()
        .checked_add((need.max(1) - 1) as u64)
        .ok_or(ConvError::AddressAbove32Bit(u64::MAX))?;
    if last > u32::MAX as u64 {
        return Err(ConvError::AddressAbove32Bit(last));
    }
    *slot = Some(bo);
    *alloc += 1;
    if replacing {
        *grows += 1;
    }
    Ok(())
}
fn put_f16(dst: &mut [u8], idx: usize, v: f16) {
    let p = idx * 2;
    dst[p..p + 2].copy_from_slice(&v.to_bits().to_le_bytes());
}
fn get_f16(src: &[u8], idx: usize) -> f16 {
    let p = idx * 2;
    f16::from_bits(u16::from_le_bytes([src[p], src[p + 1]]))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cpu_shape_and_padding_math() {
        let s = Conv2dSpec {
            ic: 1,
            ih: 8,
            iw: 8,
            oc: 2,
            kh: 5,
            kw: 5,
            pad_top: 2,
            pad_left: 2,
            stride_y: 1,
            stride_x: 1,
        };
        let a = vec![f16::ONE; 64];
        let w = vec![f16::ONE; 2 * 25];
        let o = cpu_reference_fp16(&a, &w, s).unwrap();
        assert_eq!(o.len(), 2 * 64);
        assert_eq!(o[3 * 8 + 3].to_f32(), 25.0);
    }
}
