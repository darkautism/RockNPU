#![forbid(unsafe_op_in_unsafe_fn)]

use bytemuck::pod_read_unaligned;
use half::f16;
use llama_gguf::tensor::quant::{dequantize_q4_k, dequantize_q6_k, BlockQ4K, BlockQ6K};
use rayon::prelude::*;
use rocket_runtime::RocketDevice;
use rocknpu_matmul::{
    Fp16MatmulExecutor, Fp16MatmulPool, Fp16MatmulPoolPreparedWeights, Int4DecodeExecutor,
    Int4DecodePool, Int4DecodePoolPreparedWeights, Int4GroupedPreparedWeights, Int4PreparedWeights,
    Int8DecodeExecutor, Int8DecodePool, Int8DecodePoolPreparedWeights, Int8DecodePoolStats,
    Int8DecodeSplit, Int8MtileScratch, Int8PreparedWeights, WorkerSlice,
};
use std::collections::{hash_map::Entry, HashMap};
use std::env;
use std::mem::size_of;
use std::ptr;
use std::slice;
use std::sync::Arc;
use std::time::Instant;

const STATUS_OK: i32 = 0;
const STATUS_INVALID_ARGUMENT: i32 = -1;
const STATUS_EXECUTION_ERROR: i32 = -3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DecodeWeightKind {
    Q4K,
    Q6K,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DecodeWeightKey {
    address: usize,
    bytes: usize,
    k: usize,
    n: usize,
    kind: DecodeWeightKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DecodePairKey {
    first: DecodeWeightKey,
    second: DecodeWeightKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DecodeTripleKey {
    first: DecodeWeightKey,
    second: DecodeWeightKey,
    third: DecodeWeightKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DecodeChoice {
    split: Int8DecodeSplit,
    workers: usize,
}

struct CachedW8A8Weight {
    prepared: Int8DecodePoolPreparedWeights,
    scales: Vec<f32>,
    choice: DecodeChoice,
}

struct CachedW8MtileWeight {
    prepared: Int8PreparedWeights,
    scales: Vec<f32>,
}

struct CachedW8MtilePoolWeight {
    prepared: Int8DecodePoolPreparedWeights,
    scales: Vec<f32>,
}

struct CachedW8MtilePairPoolWeight {
    first: Int8DecodePoolPreparedWeights,
    second: Int8DecodePoolPreparedWeights,
    first_scales: Vec<f32>,
    second_scales: Vec<f32>,
}

struct CachedGroupedW8MtileWeight {
    prepared: Vec<Int8PreparedWeights>,
    scales: Vec<f32>,
}

struct CachedGroupedW8MtilePoolWeight {
    prepared: Vec<Int8DecodePoolPreparedWeights>,
    scales: Vec<f32>,
}

struct CachedW4A4Weight {
    prepared: Int4PreparedWeights,
    scales: Vec<f32>,
}

struct CachedGroupedW4A4Weight {
    prepared: Int4GroupedPreparedWeights,
    scales: Vec<f32>,
}

struct CachedPoolW4A4Weight {
    prepared: Int4DecodePoolPreparedWeights,
    scales: Vec<f32>,
    workers: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum M1ProfileKind {
    Single,
    Pair,
    Triple,
}

impl M1ProfileKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Pair => "pair",
            Self::Triple => "triple",
        }
    }
}

#[derive(Default)]
struct M1ShapeProfile {
    calls: usize,
    npu_tasks: usize,
    worker_calls: [usize; 3],
    ksplit_calls: usize,
    quant_ns: u128,
    execute_wall_ns: u128,
    alloc_ns: u128,
    input_stage_ns: u128,
    partial_stage_ns: u128,
    regcmd_stage_ns: u128,
    output_fini_ns: u128,
    submit_ns: u128,
    wait_ns: u128,
    host_accum_ns: u128,
    rescale_ns: u128,
    total_ns: u128,
}

#[derive(Default)]
struct M1Profile {
    shapes: HashMap<(M1ProfileKind, usize, usize), M1ShapeProfile>,
}

#[derive(Default)]
struct MtileProfile {
    calls: usize,
    ksplit_calls: usize,
    cache_hits: usize,
    cache_misses: usize,
    weight_prepare_ns: u128,
    weight_pack_ns: u128,
    quant_ns: u128,
    alloc_ns: u128,
    input_stage_ns: u128,
    regcmd_stage_ns: u128,
    submit_ns: u128,
    wait_ns: u128,
    host_accum_ns: u128,
    execute_total_ns: u128,
    rescale_ns: u128,
}

#[derive(Default)]
struct PrefillProfile {
    cache_hits: usize,
    cache_misses: usize,
    resident_bytes: usize,
    pool_create_ns: u128,
    dequant_ns: u128,
    host_to_arc_ns: u128,
    prepare_call_ns: u128,
    prepare_pool_wall_ns: u128,
    prepare_worker_critical_ns: u128,
    plan_critical_ns: u128,
    layout_critical_ns: u128,
    alloc_mmap_critical_ns: u128,
    prep_critical_ns: u128,
    zero_critical_ns: u128,
    tile_pack_critical_ns: u128,
    fini_critical_ns: u128,
    pack_critical_ns: u128,
    pack_worker_sum_ns: u128,
    activation_convert_ns: u128,
    execute_ns: u128,
    output_copy_ns: u128,
}

pub struct RockNpuContext {
    device: RocketDevice,
    prefill_pool: Option<Fp16MatmulPool>,
    prefill_weights: HashMap<DecodeWeightKey, (usize, Fp16MatmulPoolPreparedWeights)>,
    prefill_profile: Option<PrefillProfile>,
    mtile_profile: Option<MtileProfile>,
    m1_profile: Option<M1Profile>,
    decode_pool: Int8DecodePool,
    decode_worker_cache: HashMap<(usize, usize), DecodeChoice>,
    decode_weights: HashMap<DecodeWeightKey, CachedW8A8Weight>,
    decode_mtile_weights: HashMap<DecodeWeightKey, CachedW8MtileWeight>,
    decode_mtile_pool_weights: HashMap<DecodeWeightKey, CachedW8MtilePoolWeight>,
    decode_mtile_pair_pool_weights: HashMap<DecodePairKey, CachedW8MtilePairPoolWeight>,
    decode_grouped_mtile_weights: HashMap<(DecodeWeightKey, usize), CachedGroupedW8MtileWeight>,
    decode_grouped_mtile_pool_weights:
        HashMap<(DecodeWeightKey, usize), CachedGroupedW8MtilePoolWeight>,
    decode_mtile_scratch: HashMap<(usize, usize), Option<Int8MtileScratch>>,
    decode_pair_weights: HashMap<DecodePairKey, CachedW8A8Weight>,
    decode_triple_weights: HashMap<DecodeTripleKey, CachedW8A8Weight>,
    decode_w4a4_weights: HashMap<DecodeWeightKey, CachedW4A4Weight>,
    decode_grouped_w4a4_weights: HashMap<(DecodeWeightKey, usize, bool), CachedGroupedW4A4Weight>,
    decode_pool_w4a4_weights: HashMap<(DecodeWeightKey, bool), CachedPoolW4A4Weight>,
    w4a4_pool: Option<Int4DecodePool>,
    w4a4_worker_cache: HashMap<(usize, usize), usize>,
    w4a4_enabled: bool,
    w4a4_calls: usize,
    w4a4_saturated_calls: usize,
    w4a4_saturated_outputs: usize,
    decode_cache_hits: usize,
    decode_cache_misses: usize,
    decode_cache_hit_ns: u64,
    decode_cache_miss_ns: u64,
    decode_worker_calls: [usize; 3],
    decode_ksplit_calls: usize,
    /// Reused int32 [M,N] accumulator for the direct M-tile path.
    mtile_i32: Vec<i32>,
    /// Same-input projections concatenated along N for the direct M-tile path.
    decode_mtile_concat_weights: HashMap<Vec<DecodeWeightKey>, CachedW8MtilePoolWeight>,
    /// One-shot host work to run while the next M=1 NPU projection is in
    /// flight (see `rocknpu_context_set_overlap`).
    overlap: Option<(OverlapFn, usize)>,
}

/// Shared mutable output for direct M-tile sinks: each NPU core's sink
/// writes a disjoint set of elements concurrently.
#[derive(Clone, Copy)]
struct SharedMut<T> {
    ptr: *mut T,
    len: usize,
}
// SAFETY: only used for disjoint concurrent writes coordinated by the caller.
unsafe impl<T: Send> Send for SharedMut<T> {}
unsafe impl<T: Send> Sync for SharedMut<T> {}
impl<T> SharedMut<T> {
    fn new(slice: &mut [T]) -> Self {
        Self { ptr: slice.as_mut_ptr(), len: slice.len() }
    }
    /// # Safety
    /// Concurrent callers must use non-overlapping ranges.
    unsafe fn range(&self, start: usize, len: usize) -> &mut [T] {
        assert!(start + len <= self.len);
        unsafe { slice::from_raw_parts_mut(self.ptr.add(start), len) }
    }
}

/// Host callback executed between NPU submission and completion wait.
pub type OverlapFn = unsafe extern "C" fn(user_data: *mut core::ffi::c_void);

/// Take the pending one-shot overlap callback as a closure for the pool.
fn take_overlap(overlap: &mut Option<(OverlapFn, usize)>) -> Option<impl FnMut()> {
    overlap.take().map(|(callback, user_data)| {
        move || {
            // SAFETY: the caller of rocknpu_context_set_overlap guarantees the
            // callback and user data stay valid until the next NPU call returns.
            unsafe { callback(user_data as *mut core::ffi::c_void) }
        }
    })
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RockNpuDecodeCacheStats {
    pub hits: usize,
    pub misses: usize,
    pub entries: usize,
    pub resident_bytes: usize,
    pub hit_ns: u64,
    pub miss_ns: u64,
    pub tuned_shapes: usize,
    pub worker1_calls: usize,
    pub worker2_calls: usize,
    pub worker3_calls: usize,
    pub ksplit_calls: usize,
}

fn env_enabled_default(name: &str, default: bool) -> bool {
    env::var(name)
        .map(|value| !value.is_empty() && value != "0")
        .unwrap_or(default)
}

fn env_enabled(name: &str) -> bool {
    env_enabled_default(name, false)
}

fn host_neon_enabled() -> bool {
    // Queried per activation row on the hot path; read the environment once.
    #[cfg(target_arch = "aarch64")]
    {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            env_enabled_default(
                "ROCKNPU_HOST_NEON",
                std::arch::is_aarch64_feature_detected!("neon"),
            )
        })
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

fn mtile_qo_group_size(k: usize, n: usize) -> Option<usize> {
    if k != 2048 || n != 2048 {
        return None;
    }
    let group = env::var("ROCKNPU_MTILE_QO_GROUP")
        .ok()?
        .parse::<usize>()
        .ok()?;
    (group >= 512 && group <= k && group.is_multiple_of(512) && k.is_multiple_of(group))
        .then_some(group)
}

fn mtile_shape_enabled(k: usize, n: usize) -> bool {
    if !env_enabled("ROCKNPU_W8_MTILE") {
        return false;
    }
    match env::var("ROCKNPU_W8_MTILE_SCOPE").ok().as_deref() {
        None | Some("") | Some("all") => true,
        Some("kv") => k == 2048 && n == 256,
        Some("qo") => k == 2048 && n == 2048,
        Some("ffn") => (k == 2048 && n == 5632) || (k == 5632 && n == 2048),
        Some("attn") => k == 2048 && (n == 256 || n == 2048),
        Some("safe") => k == 2048 && (n == 256 || n == 5632),
        Some("none") => false,
        Some(_) => false,
    }
}

fn ns_to_ms(value: u128) -> f64 {
    value as f64 / 1.0e6
}

fn record_m1_profile(
    profile: &mut Option<M1Profile>,
    kind: M1ProfileKind,
    k: usize,
    n: usize,
    choice: DecodeChoice,
    stats: &Int8DecodePoolStats,
    quant_ns: u128,
    rescale_ns: u128,
    total_ns: u128,
) {
    let Some(profile) = profile.as_mut() else {
        return;
    };
    let entry = profile.shapes.entry((kind, k, n)).or_default();
    entry.calls = entry.calls.saturating_add(1);
    entry.npu_tasks = entry.npu_tasks.saturating_add(stats.npu_tasks);
    if let Some(calls) = entry.worker_calls.get_mut(choice.workers.saturating_sub(1)) {
        *calls = calls.saturating_add(1);
    }
    if choice.split == Int8DecodeSplit::K {
        entry.ksplit_calls = entry.ksplit_calls.saturating_add(1);
    }
    entry.quant_ns = entry.quant_ns.saturating_add(quant_ns);
    entry.execute_wall_ns = entry.execute_wall_ns.saturating_add(stats.wall_ns);
    entry.rescale_ns = entry.rescale_ns.saturating_add(rescale_ns);
    entry.total_ns = entry.total_ns.saturating_add(total_ns);
    for worker in &stats.worker_stats {
        entry.alloc_ns = entry.alloc_ns.saturating_add(worker.alloc_ns);
        entry.input_stage_ns = entry.input_stage_ns.saturating_add(worker.input_stage_ns);
        entry.partial_stage_ns = entry
            .partial_stage_ns
            .saturating_add(worker.partial_stage_ns);
        entry.regcmd_stage_ns = entry.regcmd_stage_ns.saturating_add(worker.regcmd_stage_ns);
        entry.output_fini_ns = entry.output_fini_ns.saturating_add(worker.output_fini_ns);
        entry.submit_ns = entry.submit_ns.saturating_add(worker.submit_ns);
        entry.wait_ns = entry.wait_ns.saturating_add(worker.wait_ns);
        entry.host_accum_ns = entry.host_accum_ns.saturating_add(worker.host_accum_ns);
    }
}

const PARALLEL_DEQUANT_MIN_VALUES: usize = 1 << 20;

fn dequantize_q4_k_prefill_f16(
    weight_bytes: &[u8],
    weight_values: usize,
    parallel: bool,
) -> Vec<f16> {
    const VALUES_PER_BLOCK: usize = 256;
    if parallel && weight_values >= PARALLEL_DEQUANT_MIN_VALUES {
        let mut weights = vec![f16::from_bits(0); weight_values];
        weight_bytes
            .par_chunks_exact(size_of::<BlockQ4K>())
            .zip(weights.par_chunks_mut(VALUES_PER_BLOCK))
            .for_each(|(bytes, dst)| {
                let block: BlockQ4K = pod_read_unaligned(bytes);
                let mut decoded = [0.0f32; VALUES_PER_BLOCK];
                dequantize_q4_k(&block, &mut decoded);
                for (out, value) in dst.iter_mut().zip(decoded) {
                    *out = f16::from_f32(value);
                }
            });
        weights
    } else {
        let mut weights = Vec::with_capacity(weight_values);
        let mut decoded = [0.0f32; VALUES_PER_BLOCK];
        for bytes in weight_bytes.chunks_exact(size_of::<BlockQ4K>()) {
            let block: BlockQ4K = pod_read_unaligned(bytes);
            dequantize_q4_k(&block, &mut decoded);
            weights.extend(decoded.iter().copied().map(f16::from_f32));
        }
        weights
    }
}

fn dequantize_q6_k_prefill_f16(
    weight_bytes: &[u8],
    weight_values: usize,
    parallel: bool,
) -> Vec<f16> {
    const VALUES_PER_BLOCK: usize = 256;
    if parallel && weight_values >= PARALLEL_DEQUANT_MIN_VALUES {
        let mut weights = vec![f16::from_bits(0); weight_values];
        weight_bytes
            .par_chunks_exact(size_of::<BlockQ6K>())
            .zip(weights.par_chunks_mut(VALUES_PER_BLOCK))
            .for_each(|(bytes, dst)| {
                let block: BlockQ6K = pod_read_unaligned(bytes);
                let mut decoded = [0.0f32; VALUES_PER_BLOCK];
                dequantize_q6_k(&block, &mut decoded);
                for (out, value) in dst.iter_mut().zip(decoded) {
                    *out = f16::from_f32(value);
                }
            });
        weights
    } else {
        let mut weights = Vec::with_capacity(weight_values);
        let mut decoded = [0.0f32; VALUES_PER_BLOCK];
        for bytes in weight_bytes.chunks_exact(size_of::<BlockQ6K>()) {
            let block: BlockQ6K = pod_read_unaligned(bytes);
            dequantize_q6_k(&block, &mut decoded);
            weights.extend(decoded.iter().copied().map(f16::from_f32));
        }
        weights
    }
}

fn w4a4_shape_enabled(k: usize, n: usize) -> bool {
    match env::var("ROCKNPU_W4A4_SCOPE").as_deref() {
        Ok("ffn") => k == 2048 && n == 5632,
        Ok("attn") => k == 2048 && (n == 2048 || n == 256),
        Ok("proj2048") => k == 2048 && n == 2048,
        Ok("kv") => k == 2048 && n == 256,
        Ok("all") => true,
        _ => false,
    }
}

fn w4a4_group_size(k: usize) -> Option<usize> {
    let group = env::var("ROCKNPU_W4A4_GROUP").ok()?.parse::<usize>().ok()?;
    (group != 0 && group.is_multiple_of(32) && k.is_multiple_of(group)).then_some(group)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn max_abs_finite_neon(values: &[f32]) -> Option<f32> {
    use std::arch::aarch64::*;

    let mut i = 0usize;
    let mut max_abs;
    unsafe {
        let finite_limit = vdupq_n_f32(f32::MAX);
        let mut max_v = vdupq_n_f32(0.0);
        let mut finite_v = vdupq_n_u32(u32::MAX);
        while i + 4 <= values.len() {
            let value = vld1q_f32(values.as_ptr().add(i));
            let abs = vabsq_f32(value);
            finite_v = vandq_u32(finite_v, vcleq_f32(abs, finite_limit));
            max_v = vmaxq_f32(max_v, abs);
            i += 4;
        }
        if vminvq_u32(finite_v) != u32::MAX {
            return None;
        }
        max_abs = vmaxvq_f32(max_v);
    }
    while i < values.len() {
        let value = values[i];
        if !value.is_finite() {
            return None;
        }
        max_abs = max_abs.max(value.abs());
        i += 1;
    }
    Some(max_abs)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn quantize_symmetric_neon_into(values: &[f32], scale: f32, out: &mut [i8]) {
    use std::arch::aarch64::*;

    debug_assert_eq!(values.len(), out.len());
    let mut i = 0usize;
    unsafe {
        let inv_scale_v = vdupq_n_f32(1.0 / scale);
        let min_v = vdupq_n_f32(-127.0);
        let max_v = vdupq_n_f32(127.0);
        while i + 8 <= values.len() {
            let a = vld1q_f32(values.as_ptr().add(i));
            let b = vld1q_f32(values.as_ptr().add(i + 4));
            let a = vmaxq_f32(
                min_v,
                vminq_f32(max_v, vrndaq_f32(vmulq_f32(a, inv_scale_v))),
            );
            let b = vmaxq_f32(
                min_v,
                vminq_f32(max_v, vrndaq_f32(vmulq_f32(b, inv_scale_v))),
            );
            let a32 = vcvtq_s32_f32(a);
            let b32 = vcvtq_s32_f32(b);
            let packed16 = vcombine_s16(vqmovn_s32(a32), vqmovn_s32(b32));
            let packed8 = vqmovn_s16(packed16);
            vst1_s8(out.as_mut_ptr().add(i), packed8);
            i += 8;
        }
    }
    while i < values.len() {
        out[i] = (values[i] / scale).round().clamp(-127.0, 127.0) as i8;
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn quantize_symmetric_neon_second_pass(values: &[f32], scale: f32) -> Vec<i8> {
    let mut out = vec![0i8; values.len()];
    // SAFETY: caller already established NEON support and output has exact length.
    unsafe { quantize_symmetric_neon_into(values, scale, &mut out) };
    out
}

fn quantize_symmetric(values: &[f32]) -> Option<(Vec<i8>, f32)> {
    #[cfg(target_arch = "aarch64")]
    let max_abs = if host_neon_enabled() {
        // SAFETY: RK3588 is AArch64/ASIMD; the helper handles arbitrary tails.
        unsafe { max_abs_finite_neon(values)? }
    } else {
        let mut max_abs = 0.0f32;
        for &value in values {
            if !value.is_finite() {
                return None;
            }
            max_abs = max_abs.max(value.abs());
        }
        max_abs
    };
    #[cfg(not(target_arch = "aarch64"))]
    let max_abs = {
        let mut max_abs = 0.0f32;
        for &value in values {
            if !value.is_finite() {
                return None;
            }
            max_abs = max_abs.max(value.abs());
        }
        max_abs
    };
    if max_abs == 0.0 {
        return Some((vec![0; values.len()], 1.0));
    }
    let scale = max_abs / 127.0;
    #[cfg(target_arch = "aarch64")]
    if host_neon_enabled() {
        // SAFETY: RK3588 is AArch64/ASIMD; the helper handles arbitrary slice tails.
        let quantized = unsafe { quantize_symmetric_neon_second_pass(values, scale) };
        return Some((quantized, scale));
    }
    let quantized = values
        .iter()
        .map(|&value| (value / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    Some((quantized, scale))
}

fn quantize_symmetric_i4(values: &[f32]) -> Option<(Vec<i8>, f32)> {
    let mut max_abs = 0.0f32;
    for &value in values {
        if !value.is_finite() {
            return None;
        }
        max_abs = max_abs.max(value.abs());
    }
    if max_abs == 0.0 {
        return Some((vec![0; values.len()], 1.0));
    }
    let scale = max_abs / 7.0;
    let quantized = values
        .iter()
        .map(|&value| (value / scale).round().clamp(-7.0, 7.0) as i8)
        .collect();
    Some((quantized, scale))
}

fn quantize_symmetric_into(values: &[f32], out: &mut [i8]) -> Option<f32> {
    if values.len() != out.len() {
        return None;
    }
    #[cfg(target_arch = "aarch64")]
    let max_abs = if host_neon_enabled() {
        // SAFETY: RK3588 is AArch64/ASIMD; the helper handles arbitrary tails.
        unsafe { max_abs_finite_neon(values)? }
    } else {
        let mut max_abs = 0.0f32;
        for &value in values {
            if !value.is_finite() {
                return None;
            }
            max_abs = max_abs.max(value.abs());
        }
        max_abs
    };
    #[cfg(not(target_arch = "aarch64"))]
    let max_abs = {
        let mut max_abs = 0.0f32;
        for &value in values {
            if !value.is_finite() {
                return None;
            }
            max_abs = max_abs.max(value.abs());
        }
        max_abs
    };
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
    #[cfg(target_arch = "aarch64")]
    if host_neon_enabled() {
        // SAFETY: output slice exactly matches the input length and NEON is available.
        unsafe { quantize_symmetric_neon_into(values, scale, out) };
        return Some(scale);
    }
    for (out, &value) in out.iter_mut().zip(values) {
        *out = (value / scale).round().clamp(-127.0, 127.0) as i8;
    }
    Some(scale)
}

fn append_quantized_symmetric(values: &[f32], dst: &mut Vec<i8>) -> Option<f32> {
    let start = dst.len();
    dst.resize(start + values.len(), 0);
    quantize_symmetric_into(values, &mut dst[start..])
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn rescale_i32_row_neon(
    values: &[i32],
    weight_scales: &[f32],
    activation_scale: f32,
    output: &mut [f32],
) {
    use std::arch::aarch64::*;

    debug_assert_eq!(values.len(), weight_scales.len());
    debug_assert_eq!(values.len(), output.len());
    let mut i = 0usize;
    unsafe {
        let a_scale = vdupq_n_f32(activation_scale);
        while i + 4 <= values.len() {
            let v = vld1q_s32(values.as_ptr().add(i));
            let v = vcvtq_f32_s32(v);
            let w = vld1q_f32(weight_scales.as_ptr().add(i));
            let scaled = vmulq_f32(vmulq_f32(v, w), a_scale);
            vst1q_f32(output.as_mut_ptr().add(i), scaled);
            i += 4;
        }
    }
    while i < values.len() {
        output[i] = values[i] as f32 * activation_scale * weight_scales[i];
        i += 1;
    }
}

fn rescale_i32_row(
    values: &[i32],
    weight_scales: &[f32],
    activation_scale: f32,
    output: &mut [f32],
) {
    debug_assert_eq!(values.len(), weight_scales.len());
    debug_assert_eq!(values.len(), output.len());
    #[cfg(target_arch = "aarch64")]
    if host_neon_enabled() {
        // SAFETY: all slices have equal length and RK3588 exposes AArch64 NEON.
        unsafe { rescale_i32_row_neon(values, weight_scales, activation_scale, output) };
        return;
    }
    for ((out, &value), &weight_scale) in output.iter_mut().zip(values).zip(weight_scales) {
        *out = value as f32 * activation_scale * weight_scale;
    }
}

fn prepare_q4_k_w8a8(weight_bytes: &[u8], k: usize, n: usize) -> Option<(Vec<i8>, Vec<f32>)> {
    const VALUES: usize = 256;
    let row_bytes = (k / VALUES).checked_mul(size_of::<BlockQ4K>())?;
    if weight_bytes.len() != n.checked_mul(row_bytes)? {
        return None;
    }
    let total = n.checked_mul(k)?;
    if total >= PARALLEL_DEQUANT_MIN_VALUES {
        let mut weights = vec![0i8; total];
        let mut scales = vec![0.0f32; n];
        weight_bytes
            .par_chunks_exact(row_bytes)
            .zip(weights.par_chunks_mut(k))
            .zip(scales.par_iter_mut())
            .try_for_each_init(
                || (vec![0.0f32; k], [0.0f32; VALUES]),
                |state, ((encoded_row, output_row), output_scale)| -> Option<()> {
                    let (row, decoded) = state;
                    for (block_index, bytes) in
                        encoded_row.chunks_exact(size_of::<BlockQ4K>()).enumerate()
                    {
                        let block: BlockQ4K = pod_read_unaligned(bytes);
                        dequantize_q4_k(&block, decoded);
                        let start = block_index * VALUES;
                        row[start..start + VALUES].copy_from_slice(decoded);
                    }
                    *output_scale = quantize_symmetric_into(row, output_row)?;
                    Some(())
                },
            )?;
        return Some((weights, scales));
    }
    let mut weights = Vec::with_capacity(total);
    let mut scales = Vec::with_capacity(n);
    let mut row = Vec::with_capacity(k);
    let mut decoded = [0.0f32; VALUES];
    for encoded_row in weight_bytes.chunks_exact(row_bytes) {
        row.clear();
        for bytes in encoded_row.chunks_exact(size_of::<BlockQ4K>()) {
            let block: BlockQ4K = pod_read_unaligned(bytes);
            dequantize_q4_k(&block, &mut decoded);
            row.extend_from_slice(&decoded);
        }
        scales.push(append_quantized_symmetric(&row, &mut weights)?);
    }
    (weights.len() == total && scales.len() == n).then_some((weights, scales))
}

fn prepare_q4_k_grouped_w8a8(
    weight_bytes: &[u8],
    k: usize,
    n: usize,
    group_size: usize,
) -> Option<(Vec<i8>, Vec<f32>)> {
    const VALUES: usize = 256;
    if group_size == 0 || !group_size.is_multiple_of(512) || !k.is_multiple_of(group_size) {
        return None;
    }
    let groups = k / group_size;
    let row_bytes = (k / VALUES).checked_mul(size_of::<BlockQ4K>())?;
    if weight_bytes.len() != n.checked_mul(row_bytes)? {
        return None;
    }
    let total = n.checked_mul(k)?;
    let mut weights = vec![0i8; total];
    let mut scales = vec![0.0f32; groups.checked_mul(n)?];
    let mut row = Vec::with_capacity(k);
    let mut decoded = [0.0f32; VALUES];
    let mut quantized = Vec::with_capacity(group_size);
    for (output_channel, encoded_row) in weight_bytes.chunks_exact(row_bytes).enumerate() {
        row.clear();
        for bytes in encoded_row.chunks_exact(size_of::<BlockQ4K>()) {
            let block: BlockQ4K = pod_read_unaligned(bytes);
            dequantize_q4_k(&block, &mut decoded);
            row.extend_from_slice(&decoded);
        }
        for (group_index, group) in row.chunks_exact(group_size).enumerate() {
            quantized.clear();
            scales[group_index * n + output_channel] =
                append_quantized_symmetric(group, &mut quantized)?;
            let start = (group_index * n + output_channel).checked_mul(group_size)?;
            weights[start..start + group_size].copy_from_slice(&quantized);
        }
    }
    Some((weights, scales))
}

fn fwht_norm_in_place(values: &mut [f32]) -> bool {
    if values.is_empty() || !values.len().is_power_of_two() {
        return false;
    }
    let mut width = 1usize;
    while width < values.len() {
        let span = width * 2;
        for base in (0..values.len()).step_by(span) {
            for offset in 0..width {
                let a = values[base + offset];
                let b = values[base + offset + width];
                values[base + offset] = a + b;
                values[base + offset + width] = a - b;
            }
        }
        width = span;
    }
    let scale = (values.len() as f32).sqrt().recip();
    for value in values {
        *value *= scale;
    }
    true
}

fn quantize_symmetric_i4_grouped(
    values: &[f32],
    group_size: usize,
    hadamard: bool,
) -> Option<(Vec<i8>, Vec<f32>)> {
    if group_size == 0 || !values.len().is_multiple_of(group_size) {
        return None;
    }
    let mut transformed = Vec::new();
    let source = if hadamard {
        transformed.extend_from_slice(values);
        if !fwht_norm_in_place(&mut transformed) {
            return None;
        }
        transformed.as_slice()
    } else {
        values
    };
    let mut quantized = Vec::with_capacity(source.len());
    let mut scales = Vec::with_capacity(source.len() / group_size);
    for group in source.chunks_exact(group_size) {
        scales.push(append_quantized_symmetric_i4(group, &mut quantized)?);
    }
    Some((quantized, scales))
}

fn append_quantized_symmetric_i4(values: &[f32], dst: &mut Vec<i8>) -> Option<f32> {
    let mut max_abs = 0.0f32;
    for &value in values {
        if !value.is_finite() {
            return None;
        }
        max_abs = max_abs.max(value.abs());
    }
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
    dst.extend(
        values
            .iter()
            .map(|&value| (value / scale).round().clamp(-7.0, 7.0) as i8),
    );
    Some(scale)
}

fn prepare_q4_k_w4a4(weight_bytes: &[u8], k: usize, n: usize) -> Option<(Vec<i8>, Vec<f32>)> {
    const VALUES: usize = 256;
    let row_bytes = (k / VALUES).checked_mul(size_of::<BlockQ4K>())?;
    if weight_bytes.len() != n.checked_mul(row_bytes)? {
        return None;
    }
    let mut weights = Vec::with_capacity(n.checked_mul(k)?);
    let mut scales = Vec::with_capacity(n);
    let mut row = Vec::with_capacity(k);
    let mut decoded = [0.0f32; VALUES];
    for encoded_row in weight_bytes.chunks_exact(row_bytes) {
        row.clear();
        for bytes in encoded_row.chunks_exact(size_of::<BlockQ4K>()) {
            let block: BlockQ4K = pod_read_unaligned(bytes);
            dequantize_q4_k(&block, &mut decoded);
            row.extend_from_slice(&decoded);
        }
        scales.push(append_quantized_symmetric_i4(&row, &mut weights)?);
    }
    (weights.len() == n.checked_mul(k)? && scales.len() == n).then_some((weights, scales))
}

fn prepare_q4_k_grouped_w4a4(
    weight_bytes: &[u8],
    k: usize,
    n: usize,
    group_size: usize,
    hadamard: bool,
) -> Option<(Vec<i8>, Vec<f32>)> {
    const VALUES: usize = 256;
    if group_size == 0 || !group_size.is_multiple_of(32) || !k.is_multiple_of(group_size) {
        return None;
    }
    let groups = k / group_size;
    let row_bytes = (k / VALUES).checked_mul(size_of::<BlockQ4K>())?;
    if weight_bytes.len() != n.checked_mul(row_bytes)? {
        return None;
    }
    let mut weights = Vec::with_capacity(n.checked_mul(k)?);
    let mut scales = vec![0.0f32; groups.checked_mul(n)?];
    let mut row = Vec::with_capacity(k);
    let mut decoded = [0.0f32; VALUES];
    for (output_channel, encoded_row) in weight_bytes.chunks_exact(row_bytes).enumerate() {
        row.clear();
        for bytes in encoded_row.chunks_exact(size_of::<BlockQ4K>()) {
            let block: BlockQ4K = pod_read_unaligned(bytes);
            dequantize_q4_k(&block, &mut decoded);
            row.extend_from_slice(&decoded);
        }
        if hadamard && !fwht_norm_in_place(&mut row) {
            return None;
        }
        for (group_index, group) in row.chunks_exact(group_size).enumerate() {
            scales[group_index * n + output_channel] =
                append_quantized_symmetric_i4(group, &mut weights)?;
        }
    }
    (weights.len() == n.checked_mul(k)?).then_some((weights, scales))
}

fn prepare_q6_k_w8a8(weight_bytes: &[u8], k: usize, n: usize) -> Option<(Vec<i8>, Vec<f32>)> {
    const VALUES: usize = 256;
    let row_bytes = (k / VALUES).checked_mul(size_of::<BlockQ6K>())?;
    if weight_bytes.len() != n.checked_mul(row_bytes)? {
        return None;
    }
    let mut weights = Vec::with_capacity(n.checked_mul(k)?);
    let mut scales = Vec::with_capacity(n);
    let mut row = Vec::with_capacity(k);
    let mut decoded = [0.0f32; VALUES];
    for encoded_row in weight_bytes.chunks_exact(row_bytes) {
        row.clear();
        for bytes in encoded_row.chunks_exact(size_of::<BlockQ6K>()) {
            let block: BlockQ6K = pod_read_unaligned(bytes);
            dequantize_q6_k(&block, &mut decoded);
            row.extend_from_slice(&decoded);
        }
        scales.push(append_quantized_symmetric(&row, &mut weights)?);
    }
    (weights.len() == n.checked_mul(k)? && scales.len() == n).then_some((weights, scales))
}

fn prepare_q6_k_grouped_w8a8(
    weight_bytes: &[u8],
    k: usize,
    n: usize,
    group_size: usize,
) -> Option<(Vec<i8>, Vec<f32>)> {
    const VALUES: usize = 256;
    if group_size == 0 || !group_size.is_multiple_of(512) || !k.is_multiple_of(group_size) {
        return None;
    }
    let groups = k / group_size;
    let row_bytes = (k / VALUES).checked_mul(size_of::<BlockQ6K>())?;
    if weight_bytes.len() != n.checked_mul(row_bytes)? {
        return None;
    }
    let total = n.checked_mul(k)?;
    let mut weights = vec![0i8; total];
    let mut scales = vec![0.0f32; groups.checked_mul(n)?];
    let mut row = Vec::with_capacity(k);
    let mut decoded = [0.0f32; VALUES];
    let mut quantized = Vec::with_capacity(group_size);
    for (output_channel, encoded_row) in weight_bytes.chunks_exact(row_bytes).enumerate() {
        row.clear();
        for bytes in encoded_row.chunks_exact(size_of::<BlockQ6K>()) {
            let block: BlockQ6K = pod_read_unaligned(bytes);
            dequantize_q6_k(&block, &mut decoded);
            row.extend_from_slice(&decoded);
        }
        for (group_index, group) in row.chunks_exact(group_size).enumerate() {
            quantized.clear();
            scales[group_index * n + output_channel] =
                append_quantized_symmetric(group, &mut quantized)?;
            let start = (group_index * n + output_channel).checked_mul(group_size)?;
            weights[start..start + group_size].copy_from_slice(&quantized);
        }
    }
    Some((weights, scales))
}

fn decode_worker_candidates(
    pool: &Int8DecodePool,
    k: usize,
    n: usize,
) -> Result<Vec<DecodeChoice>, ()> {
    let mut candidates = vec![DecodeChoice {
        split: Int8DecodeSplit::N,
        workers: 1,
    }];
    let mut last_effective_n = 1usize;
    for requested in 2..=pool.workers() {
        let effective = pool.effective_workers_for_n(n, requested).map_err(|_| ())?;
        if effective > last_effective_n {
            candidates.push(DecodeChoice {
                split: Int8DecodeSplit::N,
                workers: requested,
            });
            last_effective_n = effective;
        }
    }
    if k > 4096 && !env_enabled("ROCKNPU_M1_FORCE_N_SPLIT") {
        let mut last_effective_k = 1usize;
        for requested in 2..=pool.workers() {
            let effective = pool.effective_workers_for_k(k, requested).map_err(|_| ())?;
            if effective > last_effective_k {
                candidates.push(DecodeChoice {
                    split: Int8DecodeSplit::K,
                    workers: requested,
                });
                last_effective_k = effective;
            }
        }
    }
    Ok(candidates)
}

fn select_decode_worker_index(medians: &[u128]) -> Option<usize> {
    const REQUIRED_IMPROVEMENT_PERCENT: u128 = 5;
    let (&first, rest) = medians.split_first()?;
    let mut selected_index = 0usize;
    let mut selected_median = first;
    for (offset, &candidate_median) in rest.iter().enumerate() {
        let candidate_index = offset + 1;
        let candidate_scaled = candidate_median.saturating_mul(100);
        let required_scaled = selected_median.saturating_mul(100 - REQUIRED_IMPROVEMENT_PERCENT);
        if candidate_scaled <= required_scaled {
            selected_index = candidate_index;
            selected_median = candidate_median;
        }
    }
    Some(selected_index)
}

fn tune_decode_workers(
    pool: &mut Int8DecodePool,
    weights: Arc<[i8]>,
    activation: Arc<[i8]>,
    k: usize,
    n: usize,
) -> Result<(DecodeChoice, Int8DecodePoolPreparedWeights), ()> {
    let candidates = decode_worker_candidates(pool, k, n)?;
    let mut prepared = Vec::with_capacity(candidates.len());
    for &choice in &candidates {
        match pool.prepare_weights_m1_with_split(
            Arc::clone(&weights),
            k,
            n,
            choice.workers,
            choice.split,
        ) {
            Ok(handle) => prepared.push((choice, handle)),
            Err(_) => {
                for (_, handle) in prepared {
                    let _ = pool.release_prepared(&handle);
                }
                return Err(());
            }
        }
    }

    let tuning = (|| -> Result<usize, ()> {
        for (_, handle) in &prepared {
            let _ = pool
                .execute_prepared(Arc::clone(&activation), handle)
                .map_err(|_| ())?;
        }
        let mut samples: Vec<Vec<u128>> = prepared.iter().map(|_| Vec::with_capacity(3)).collect();
        for round in 0..3 {
            if round % 2 == 0 {
                for (index, (_, handle)) in prepared.iter().enumerate() {
                    let start = Instant::now();
                    let _ = pool
                        .execute_prepared(Arc::clone(&activation), handle)
                        .map_err(|_| ())?;
                    samples[index].push(start.elapsed().as_nanos());
                }
            } else {
                for (index, (_, handle)) in prepared.iter().enumerate().rev() {
                    let start = Instant::now();
                    let _ = pool
                        .execute_prepared(Arc::clone(&activation), handle)
                        .map_err(|_| ())?;
                    samples[index].push(start.elapsed().as_nanos());
                }
            }
        }
        let mut medians = Vec::with_capacity(samples.len());
        for values in &mut samples {
            values.sort_unstable();
            medians.push(values[values.len() / 2]);
        }
        select_decode_worker_index(&medians).ok_or(())
    })();

    let best_index = match tuning {
        Ok(index) => index,
        Err(()) => {
            for (_, handle) in prepared {
                let _ = pool.release_prepared(&handle);
            }
            return Err(());
        }
    };

    let mut winner = None;
    for (index, (choice, handle)) in prepared.into_iter().enumerate() {
        if index == best_index {
            winner = Some((choice, handle));
        } else if pool.release_prepared(&handle).is_err() {
            return Err(());
        }
    }
    winner.ok_or(())
}

fn tune_w4a4_workers(
    pool: &mut Int4DecodePool,
    weights: Arc<[i8]>,
    activation: Arc<[i8]>,
    k: usize,
    n: usize,
) -> Result<(usize, Int4DecodePoolPreparedWeights), ()> {
    let mut candidates = vec![1usize];
    let mut last_effective = 1usize;
    for requested in 2..=pool.workers() {
        let effective = pool.effective_workers_for_n(n, requested).map_err(|_| ())?;
        if effective > last_effective {
            candidates.push(requested);
            last_effective = effective;
        }
    }

    let mut prepared = Vec::with_capacity(candidates.len());
    for &workers in &candidates {
        match pool.prepare_weights(Arc::clone(&weights), k, n, workers) {
            Ok(handle) => prepared.push((workers, handle)),
            Err(_) => {
                for (_, handle) in prepared {
                    let _ = pool.release_prepared(&handle);
                }
                return Err(());
            }
        }
    }

    let tuning = (|| -> Result<usize, ()> {
        for (_, handle) in &prepared {
            let _ = pool
                .execute_prepared(Arc::clone(&activation), handle)
                .map_err(|_| ())?;
        }
        let mut samples: Vec<Vec<u128>> = prepared.iter().map(|_| Vec::with_capacity(3)).collect();
        for round in 0usize..3 {
            if round.is_multiple_of(2) {
                for (index, (_, handle)) in prepared.iter().enumerate() {
                    let start = Instant::now();
                    let _ = pool
                        .execute_prepared(Arc::clone(&activation), handle)
                        .map_err(|_| ())?;
                    samples[index].push(start.elapsed().as_nanos());
                }
            } else {
                for (index, (_, handle)) in prepared.iter().enumerate().rev() {
                    let start = Instant::now();
                    let _ = pool
                        .execute_prepared(Arc::clone(&activation), handle)
                        .map_err(|_| ())?;
                    samples[index].push(start.elapsed().as_nanos());
                }
            }
        }
        let mut medians = Vec::with_capacity(samples.len());
        for values in &mut samples {
            values.sort_unstable();
            medians.push(values[values.len() / 2]);
        }
        let selected = select_decode_worker_index(&medians).ok_or(())?;
        if env_enabled("ROCKNPU_W4A4_TRACE") {
            let medians_us = medians
                .iter()
                .map(|&ns| ns as f64 / 1.0e3)
                .collect::<Vec<_>>();
            eprintln!(
                "ROCKNPU W4A4 tune K={} N={} candidates={:?} medians_us={:?} selected_workers={}",
                k, n, candidates, medians_us, candidates[selected]
            );
        }
        Ok(selected)
    })();

    let best_index = match tuning {
        Ok(index) => index,
        Err(()) => {
            for (_, handle) in prepared {
                let _ = pool.release_prepared(&handle);
            }
            return Err(());
        }
    };
    let mut winner = None;
    for (index, (workers, handle)) in prepared.into_iter().enumerate() {
        if index == best_index {
            winner = Some((workers, handle));
        } else if pool.release_prepared(&handle).is_err() {
            return Err(());
        }
    }
    winner.ok_or(())
}

fn execute_cached_pool_w4a4_m1<F>(
    context: &mut RockNpuContext,
    key: DecodeWeightKey,
    hadamard: bool,
    activations_k_f32: &[f32],
    output_n_f32: &mut [f32],
    prepare: F,
) -> i32
where
    F: FnOnce() -> Option<(Vec<i8>, Vec<f32>)>,
{
    let call_start = Instant::now();
    let Some(pool) = context.w4a4_pool.as_mut() else {
        return STATUS_EXECUTION_ERROR;
    };
    let Some((activations_i4, activation_scales)) =
        quantize_symmetric_i4_grouped(activations_k_f32, key.k, hadamard)
    else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(&activation_scale) = activation_scales.first() else {
        return STATUS_INVALID_ARGUMENT;
    };
    if activation_scales.len() != 1 || output_n_f32.len() != key.n {
        return STATUS_INVALID_ARGUMENT;
    }
    let activation: Arc<[i8]> = Arc::from(activations_i4);
    let cache_key = (key, hadamard);

    let (cached, cache_hit) = match context.decode_pool_w4a4_weights.entry(cache_key) {
        Entry::Occupied(entry) => {
            context.decode_cache_hits = context.decode_cache_hits.saturating_add(1);
            (entry.into_mut(), true)
        }
        Entry::Vacant(entry) => {
            let Some((weights_i4, scales)) = prepare() else {
                return STATUS_INVALID_ARGUMENT;
            };
            if scales.len() != key.n {
                return STATUS_INVALID_ARGUMENT;
            }
            let weights: Arc<[i8]> = Arc::from(weights_i4);
            let shape = (key.k, key.n);
            let (workers, prepared) = if let Some(&workers) = context.w4a4_worker_cache.get(&shape)
            {
                let prepared =
                    match pool.prepare_weights(Arc::clone(&weights), key.k, key.n, workers) {
                        Ok(prepared) => prepared,
                        Err(_) => return STATUS_EXECUTION_ERROR,
                    };
                (workers, prepared)
            } else {
                let (workers, prepared) = match tune_w4a4_workers(
                    pool,
                    Arc::clone(&weights),
                    Arc::clone(&activation),
                    key.k,
                    key.n,
                ) {
                    Ok(result) => result,
                    Err(()) => return STATUS_EXECUTION_ERROR,
                };
                context.w4a4_worker_cache.insert(shape, workers);
                (workers, prepared)
            };
            context.decode_cache_misses = context.decode_cache_misses.saturating_add(1);
            (
                entry.insert(CachedPoolW4A4Weight {
                    prepared,
                    scales,
                    workers,
                }),
                false,
            )
        }
    };

    let result = match pool.execute_prepared(Arc::clone(&activation), &cached.prepared) {
        Ok(result) => result,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    context.w4a4_calls = context.w4a4_calls.saturating_add(1);
    if let Some(calls) = context
        .decode_worker_calls
        .get_mut(cached.workers.saturating_sub(1))
    {
        *calls = calls.saturating_add(1);
    }
    if result.stats.saturated_outputs != 0 {
        context.w4a4_saturated_calls = context.w4a4_saturated_calls.saturating_add(1);
        context.w4a4_saturated_outputs = context
            .w4a4_saturated_outputs
            .saturating_add(result.stats.saturated_outputs);
    }
    for ((dst, &acc), &weight_scale) in output_n_f32
        .iter_mut()
        .zip(&result.values)
        .zip(&cached.scales)
    {
        *dst = f32::from(acc) * activation_scale * weight_scale;
    }
    let elapsed_ns = u64::try_from(call_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    if cache_hit {
        context.decode_cache_hit_ns = context.decode_cache_hit_ns.saturating_add(elapsed_ns);
    } else {
        context.decode_cache_miss_ns = context.decode_cache_miss_ns.saturating_add(elapsed_ns);
    }
    STATUS_OK
}

fn execute_cached_grouped_w4a4_m1<F>(
    context: &mut RockNpuContext,
    key: DecodeWeightKey,
    group_size: usize,
    hadamard: bool,
    activations_k_f32: &[f32],
    output_n_f32: &mut [f32],
    prepare: F,
) -> i32
where
    F: FnOnce() -> Option<(Vec<i8>, Vec<f32>)>,
{
    let call_start = Instant::now();
    if activations_k_f32.len() != key.k
        || output_n_f32.len() != key.n
        || group_size == 0
        || !group_size.is_multiple_of(32)
        || !key.k.is_multiple_of(group_size)
        || !key.n.is_multiple_of(64)
        || key.n > 8192
    {
        return STATUS_INVALID_ARGUMENT;
    }
    let Some((activations_i4, activation_scales)) =
        quantize_symmetric_i4_grouped(activations_k_f32, group_size, hadamard)
    else {
        return STATUS_INVALID_ARGUMENT;
    };
    let groups = key.k / group_size;
    if activation_scales.len() != groups {
        return STATUS_INVALID_ARGUMENT;
    }

    let RockNpuContext {
        device,
        decode_grouped_w4a4_weights,
        w4a4_calls,
        w4a4_saturated_calls,
        w4a4_saturated_outputs,
        decode_cache_hits,
        decode_cache_misses,
        decode_cache_hit_ns,
        decode_cache_miss_ns,
        ..
    } = context;
    let executor = Int4DecodeExecutor::new(device);
    let cache_key = (key, group_size, hadamard);
    let (cached, cache_hit) = match decode_grouped_w4a4_weights.entry(cache_key) {
        Entry::Occupied(entry) => {
            *decode_cache_hits = decode_cache_hits.saturating_add(1);
            (entry.into_mut(), true)
        }
        Entry::Vacant(entry) => {
            let Some((weights_i4, scales)) = prepare() else {
                return STATUS_INVALID_ARGUMENT;
            };
            if scales.len() != groups.saturating_mul(key.n) {
                return STATUS_INVALID_ARGUMENT;
            }
            let prepared =
                match executor.prepare_grouped_weights(&weights_i4, key.k, key.n, group_size) {
                    Ok(prepared) => prepared,
                    Err(_) => return STATUS_EXECUTION_ERROR,
                };
            *decode_cache_misses = decode_cache_misses.saturating_add(1);
            (
                entry.insert(CachedGroupedW4A4Weight { prepared, scales }),
                false,
            )
        }
    };

    let result = match executor.execute_grouped_prepared(&activations_i4, &cached.prepared) {
        Ok(result) => result,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    if result.values.len() != groups.saturating_mul(key.n) {
        return STATUS_EXECUTION_ERROR;
    }
    *w4a4_calls = w4a4_calls.saturating_add(1);
    if result.stats.saturated_outputs != 0 {
        *w4a4_saturated_calls = w4a4_saturated_calls.saturating_add(1);
        *w4a4_saturated_outputs =
            w4a4_saturated_outputs.saturating_add(result.stats.saturated_outputs);
    }
    output_n_f32.fill(0.0);
    for group in 0..groups {
        let activation_scale = activation_scales[group];
        let partials = &result.values[group * key.n..(group + 1) * key.n];
        let weight_scales = &cached.scales[group * key.n..(group + 1) * key.n];
        for ((dst, &acc), &weight_scale) in output_n_f32.iter_mut().zip(partials).zip(weight_scales)
        {
            *dst += f32::from(acc) * activation_scale * weight_scale;
        }
    }
    let elapsed_ns = u64::try_from(call_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    if cache_hit {
        *decode_cache_hit_ns = decode_cache_hit_ns.saturating_add(elapsed_ns);
    } else {
        *decode_cache_miss_ns = decode_cache_miss_ns.saturating_add(elapsed_ns);
    }
    STATUS_OK
}

fn execute_cached_w4a4_m1<F>(
    context: &mut RockNpuContext,
    key: DecodeWeightKey,
    activations_k_f32: &[f32],
    output_n_f32: &mut [f32],
    prepare: F,
) -> i32
where
    F: FnOnce() -> Option<(Vec<i8>, Vec<f32>)>,
{
    let call_start = Instant::now();
    if activations_k_f32.len() != key.k
        || output_n_f32.len() != key.n
        || !key.k.is_multiple_of(32)
        || key.k > 10_752
        || !key.n.is_multiple_of(64)
        || key.n > 8192
    {
        return STATUS_INVALID_ARGUMENT;
    }
    let Some((activations_i4, activation_scale)) = quantize_symmetric_i4(activations_k_f32) else {
        return STATUS_INVALID_ARGUMENT;
    };

    let RockNpuContext {
        device,
        decode_w4a4_weights,
        w4a4_calls,
        w4a4_saturated_calls,
        w4a4_saturated_outputs,
        decode_cache_hits,
        decode_cache_misses,
        decode_cache_hit_ns,
        decode_cache_miss_ns,
        ..
    } = context;
    let executor = Int4DecodeExecutor::new(device);

    let (cached, cache_hit) = match decode_w4a4_weights.entry(key) {
        Entry::Occupied(entry) => {
            *decode_cache_hits = decode_cache_hits.saturating_add(1);
            (entry.into_mut(), true)
        }
        Entry::Vacant(entry) => {
            let Some((weights_i4, scales)) = prepare() else {
                return STATUS_INVALID_ARGUMENT;
            };
            let prepared = match executor.prepare_weights(&weights_i4, key.k, key.n) {
                Ok(prepared) => prepared,
                Err(_) => return STATUS_EXECUTION_ERROR,
            };
            *decode_cache_misses = decode_cache_misses.saturating_add(1);
            (entry.insert(CachedW4A4Weight { prepared, scales }), false)
        }
    };

    let result = match executor.execute_prepared(&activations_i4, &cached.prepared) {
        Ok(result) => result,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    *w4a4_calls = w4a4_calls.saturating_add(1);
    if result.stats.saturated_outputs != 0 {
        *w4a4_saturated_calls = w4a4_saturated_calls.saturating_add(1);
        *w4a4_saturated_outputs =
            w4a4_saturated_outputs.saturating_add(result.stats.saturated_outputs);
        if env_enabled("ROCKNPU_W4A4_TRACE") {
            eprintln!(
                "ROCKNPU W4A4 saturation K={} N={} saturated_outputs={}",
                key.k, key.n, result.stats.saturated_outputs
            );
        }
    }
    for ((dst, &acc), &weight_scale) in output_n_f32
        .iter_mut()
        .zip(&result.values)
        .zip(&cached.scales)
    {
        *dst = f32::from(acc) * activation_scale * weight_scale;
    }
    let elapsed_ns = u64::try_from(call_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    if cache_hit {
        *decode_cache_hit_ns = decode_cache_hit_ns.saturating_add(elapsed_ns);
    } else {
        *decode_cache_miss_ns = decode_cache_miss_ns.saturating_add(elapsed_ns);
    }
    STATUS_OK
}

fn execute_cached_w8a8_m1<F>(
    context: &mut RockNpuContext,
    key: DecodeWeightKey,
    activations_k_f32: &[f32],
    output_n_f32: &mut [f32],
    prepare: F,
) -> i32
where
    F: FnOnce() -> Option<(Vec<i8>, Vec<f32>)>,
{
    let call_start = Instant::now();
    if activations_k_f32.len() != key.k
        || output_n_f32.len() != key.n
        || !key.k.is_multiple_of(512)
        || !key.n.is_multiple_of(32)
        || key.n > 8192
    {
        return STATUS_INVALID_ARGUMENT;
    }
    let quant_started = context.m1_profile.as_ref().map(|_| Instant::now());
    let Some((activations_i8, activation_scale)) = quantize_symmetric(activations_k_f32) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let quant_ns = quant_started.map_or(0, |started| started.elapsed().as_nanos());
    let activation: Arc<[i8]> = Arc::from(activations_i8);

    let RockNpuContext {
        device: _,
        decode_pool,
        decode_worker_cache,
        decode_weights,
        decode_cache_hits,
        decode_cache_misses,
        decode_cache_hit_ns,
        decode_cache_miss_ns,
        decode_worker_calls,
        decode_ksplit_calls,
        m1_profile,
        overlap,
        ..
    } = context;

    let (cached, cache_hit) = match decode_weights.entry(key) {
        Entry::Occupied(entry) => {
            *decode_cache_hits = decode_cache_hits.saturating_add(1);
            (entry.into_mut(), true)
        }
        Entry::Vacant(entry) => {
            let Some((weights_i8, scales)) = prepare() else {
                return STATUS_INVALID_ARGUMENT;
            };
            let weights: Arc<[i8]> = Arc::from(weights_i8);
            let shape = (key.k, key.n);
            let (choice, prepared) = if let Some(&choice) = decode_worker_cache.get(&shape) {
                let prepared = match decode_pool.prepare_weights_m1_with_split(
                    Arc::clone(&weights),
                    key.k,
                    key.n,
                    choice.workers,
                    choice.split,
                ) {
                    Ok(prepared) => prepared,
                    Err(_) => return STATUS_EXECUTION_ERROR,
                };
                (choice, prepared)
            } else {
                let (choice, prepared) = match tune_decode_workers(
                    decode_pool,
                    Arc::clone(&weights),
                    Arc::clone(&activation),
                    key.k,
                    key.n,
                ) {
                    Ok(result) => result,
                    Err(()) => return STATUS_EXECUTION_ERROR,
                };
                decode_worker_cache.insert(shape, choice);
                (choice, prepared)
            };
            *decode_cache_misses = decode_cache_misses.saturating_add(1);
            (
                entry.insert(CachedW8A8Weight {
                    prepared,
                    scales,
                    choice,
                }),
                false,
            )
        }
    };

    let mut overlap_fn = take_overlap(overlap);
    let result = match decode_pool.execute_prepared_overlap(
        Arc::clone(&activation),
        &cached.prepared,
        overlap_fn.as_mut().map(|f| f as &mut dyn FnMut()),
    ) {
        Ok(result) => result,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    if let Some(calls) = decode_worker_calls.get_mut(cached.choice.workers.saturating_sub(1)) {
        *calls = calls.saturating_add(1);
    }
    if cached.choice.split == Int8DecodeSplit::K {
        *decode_ksplit_calls = decode_ksplit_calls.saturating_add(1);
    }
    let rescale_started = m1_profile.as_ref().map(|_| Instant::now());
    for ((dst, &acc), &weight_scale) in output_n_f32
        .iter_mut()
        .zip(&result.values)
        .zip(&cached.scales)
    {
        *dst = acc as f32 * activation_scale * weight_scale;
    }
    let rescale_ns = rescale_started.map_or(0, |started| started.elapsed().as_nanos());
    let elapsed_total_ns = call_start.elapsed().as_nanos();
    let elapsed_ns = u64::try_from(elapsed_total_ns).unwrap_or(u64::MAX);
    if cache_hit {
        *decode_cache_hit_ns = decode_cache_hit_ns.saturating_add(elapsed_ns);
        record_m1_profile(
            m1_profile,
            M1ProfileKind::Single,
            key.k,
            key.n,
            cached.choice,
            &result.stats,
            quant_ns,
            rescale_ns,
            elapsed_total_ns,
        );
    } else {
        *decode_cache_miss_ns = decode_cache_miss_ns.saturating_add(elapsed_ns);
    }
    STATUS_OK
}

fn execute_cached_w8a8_mtile_pool<F>(
    context: &mut RockNpuContext,
    key: DecodeWeightKey,
    m: usize,
    split: Int8DecodeSplit,
    activations_mk_f32: &[f32],
    output_mn_f32: &mut [f32],
    prepare: F,
) -> i32
where
    F: FnOnce() -> Option<(Vec<i8>, Vec<f32>)>,
{
    if !matches!(m, 4 | 8 | 12 | 16 | 32 | 48 | 64 | 128) {
        return STATUS_INVALID_ARGUMENT;
    }
    let Some(expected_a) = m.checked_mul(key.k) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(expected_out) = m.checked_mul(key.n) else {
        return STATUS_INVALID_ARGUMENT;
    };
    const MAX_POOL_K: usize = 3 * 4096;
    const MIN_POOL_N: usize = MTILE_POOL_MIN_N;
    const MAX_POOL_N: usize = 8192;
    let split_shape_valid = match split {
        Int8DecodeSplit::N => key.k <= 4096 && key.n >= MIN_POOL_N,
        Int8DecodeSplit::K => key.k > 4096 && key.k <= MAX_POOL_K,
    };
    if activations_mk_f32.len() != expected_a
        || output_mn_f32.len() != expected_out
        || key.k == 0
        || !key.k.is_multiple_of(512)
        || key.n == 0
        || !key.n.is_multiple_of(32)
        || key.n > MAX_POOL_N
        || !split_shape_valid
    {
        return STATUS_INVALID_ARGUMENT;
    }

    let quant_started = context.mtile_profile.as_ref().map(|_| Instant::now());
    let mut activations_i8 = vec![0i8; expected_a];
    let mut activation_scales = vec![0.0f32; m];
    let quantized_ok = activations_mk_f32
        .par_chunks_exact(key.k)
        .zip(activations_i8.par_chunks_exact_mut(key.k))
        .zip(activation_scales.par_iter_mut())
        .all(|((row, out), scale)| match quantize_symmetric_into(row, out) {
            Some(row_scale) => {
                *scale = row_scale;
                true
            }
            None => false,
        });
    if !quantized_ok {
        return STATUS_INVALID_ARGUMENT;
    }
    if let (Some(profile), Some(started)) = (context.mtile_profile.as_mut(), quant_started) {
        profile.quant_ns += started.elapsed().as_nanos();
    }

    let RockNpuContext {
        decode_pool,
        decode_mtile_pool_weights,
        mtile_profile,
        mtile_i32,
        ..
    } = context;
    let direct = decode_pool.direct_enabled();

    let mut cache_miss = false;
    let mut weight_prepare_ns = 0u128;
    let mut weight_pack_ns = 0u128;
    let cached = match decode_mtile_pool_weights.entry(key) {
        Entry::Occupied(entry) => entry.into_mut(),
        Entry::Vacant(entry) => {
            cache_miss = true;
            let started = Instant::now();
            let Some((weights_i8, scales)) = prepare() else {
                return STATUS_INVALID_ARGUMENT;
            };
            if weights_i8.len() != key.n.saturating_mul(key.k) || scales.len() != key.n {
                return STATUS_INVALID_ARGUMENT;
            }
            weight_prepare_ns = started.elapsed().as_nanos();
            let started = Instant::now();
            let prepared = match if direct {
                decode_pool.prepare_weights_mtile_direct_with_split(
                    Arc::<[i8]>::from(weights_i8),
                    key.k,
                    key.n,
                    3,
                    split,
                )
            } else {
                decode_pool.prepare_weights_mtile_with_split(
                    Arc::<[i8]>::from(weights_i8),
                    key.k,
                    key.n,
                    3,
                    split,
                )
            } {
                Ok(prepared) => prepared,
                Err(err) => {
                    if env_enabled("ROCKNPU_MTILE_TRACE") {
                        eprintln!("ROCKNPU MTILE ERROR prepare M={m} K={} N={} split={split:?}: {err}", key.k, key.n);
                    }
                    return STATUS_EXECUTION_ERROR;
                }
            };
            weight_pack_ns = started.elapsed().as_nanos();
            entry.insert(CachedW8MtilePoolWeight { prepared, scales })
        }
    };

    if let Some(profile) = mtile_profile.as_mut() {
        if cache_miss {
            profile.cache_misses += 1;
            profile.weight_prepare_ns += weight_prepare_ns;
            profile.weight_pack_ns += weight_pack_ns;
        } else {
            profile.cache_hits += 1;
        }
    }

    if cached.prepared.is_direct() {
        let started = Instant::now();
        let n = key.n;
        let scales = &cached.scales;
        let a_scales = &activation_scales;
        let workers = cached.prepared.slices().len();
        let result = match split {
            Int8DecodeSplit::N => {
                // Rescale each core's [M, nsub] block straight into its columns.
                let out = SharedMut::new(output_mn_f32);
                let sink = |_: usize, slice: WorkerSlice, values: &[i32]| {
                    for row in 0..m {
                        // SAFETY: cores own disjoint column ranges of each row.
                        let dst = unsafe { out.range(row * n + slice.n0, slice.nsub) };
                        rescale_i32_row(
                            &values[row * slice.nsub..(row + 1) * slice.nsub],
                            &scales[slice.n0..slice.n0 + slice.nsub],
                            a_scales[row],
                            dst,
                        );
                    }
                };
                decode_pool.execute_prepared_mtile_direct(m, &activations_i8, &cached.prepared, &sink)
            }
            Int8DecodeSplit::K => {
                // Each core returns a full [M, N] partial over its K range.
                mtile_i32.resize(workers * expected_out, 0);
                let partials = SharedMut::new(mtile_i32);
                let sink = |worker: usize, _: WorkerSlice, values: &[i32]| {
                    // SAFETY: each core writes its own partial region.
                    unsafe { partials.range(worker * expected_out, expected_out) }
                        .copy_from_slice(&values[..expected_out]);
                };
                let result = decode_pool.execute_prepared_mtile_direct(
                    m,
                    &activations_i8,
                    &cached.prepared,
                    &sink,
                );
                if result.is_ok() {
                    let partials = &mtile_i32[..];
                    output_mn_f32
                        .par_chunks_exact_mut(n)
                        .enumerate()
                        .for_each(|(row, out)| {
                            let mut acc = [0i32; 64];
                            for c0 in (0..n).step_by(64) {
                                let len = (n - c0).min(64);
                                acc[..len].fill(0);
                                for w in 0..workers {
                                    let base = w * expected_out + row * n + c0;
                                    for (a, &v) in acc[..len].iter_mut().zip(&partials[base..base + len]) {
                                        *a = a.wrapping_add(v);
                                    }
                                }
                                rescale_i32_row(
                                    &acc[..len],
                                    &scales[c0..c0 + len],
                                    a_scales[row],
                                    &mut out[c0..c0 + len],
                                );
                            }
                        });
                }
                result
            }
        };
        let timings = match result {
            Ok(timings) => timings,
            Err(err) => {
                if env_enabled("ROCKNPU_MTILE_TRACE") {
                    eprintln!("ROCKNPU MTILE ERROR direct M={m} K={} N={} split={split:?}: {err}", key.k, key.n);
                }
                return STATUS_EXECUTION_ERROR;
            }
        };
        if let Some(profile) = mtile_profile.as_mut() {
            profile.calls += 1;
            profile.execute_total_ns += started.elapsed().as_nanos();
            profile.input_stage_ns += timings.stage_submit_ns;
            profile.wait_ns += timings.wait_ns;
            profile.host_accum_ns += timings.consume_ns;
        }
        return STATUS_OK;
    }

    let result = match decode_pool.execute_prepared_mtile(
        m,
        Arc::<[i8]>::from(activations_i8),
        &cached.prepared,
    ) {
        Ok(result) => result,
        Err(err) => {
            if env_enabled("ROCKNPU_MTILE_TRACE") {
                eprintln!("ROCKNPU MTILE ERROR execute M={m} K={} N={} split={split:?}: {err}", key.k, key.n);
            }
            return STATUS_EXECUTION_ERROR;
        }
    };

    if let Some(profile) = mtile_profile.as_mut() {
        profile.calls += 1;
        profile.execute_total_ns += result.stats.wall_ns;
        for stats in &result.stats.worker_stats {
            profile.alloc_ns += stats.alloc_ns;
            profile.input_stage_ns += stats.input_stage_ns;
            profile.regcmd_stage_ns += stats.regcmd_stage_ns;
            profile.submit_ns += stats.submit_ns;
            profile.wait_ns += stats.wait_ns;
            profile.host_accum_ns += stats.host_accum_ns;
        }
    }

    let rescale_started = mtile_profile.as_ref().map(|_| Instant::now());
    for row in 0..m {
        let start = row * key.n;
        let end = start + key.n;
        rescale_i32_row(
            &result.values[start..end],
            &cached.scales,
            activation_scales[row],
            &mut output_mn_f32[start..end],
        );
    }
    if let (Some(profile), Some(started)) = (mtile_profile.as_mut(), rescale_started) {
        profile.rescale_ns += started.elapsed().as_nanos();
    }
    STATUS_OK
}

/// Tile heights the native W8A8 M-tile path executes directly.
const MTILE_ROWS: [usize; 8] = [4, 8, 12, 16, 32, 48, 64, 128];

/// Prompt projections at least this wide split N across the three NPU cores
/// (>= 256 columns per core); narrower ones run on one core.
const MTILE_POOL_MIN_N: usize = 768;

fn native_mtile_routes_enabled() -> bool {
    env_enabled("ROCKNPU_NATIVE_MTILE")
        && env_enabled("ROCKNPU_MTILE_PERSIST")
        && env_enabled("ROCKNPU_W8_MTILE")
        && env_enabled("ROCKNPU_MTILE_MC")
}

/// Two-part activation encoding for prompt projections: each activation row
/// is quantized to int8 as usual, and its quantization residual is stacked
/// as an extra row. The int8 NPU then computes both with the same weights
/// and the two output halves are added, which gives ~15-bit activation
/// precision. Models with large activation outliers (Qwen) lose most of
/// their W8A8 error this way, at the cost of twice the NPU rows.
///
/// `ROCKNPU_PREFILL_HILO`: `0` off (default), `1` every prompt projection,
/// `down` only wide-K projections (FFN down), whose SwiGLU inputs carry the
/// largest per-token outliers.
fn hilo_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match env::var("ROCKNPU_PREFILL_HILO").ok().as_deref() {
        Some("1") | Some("all") => 2,
        Some("down") => 1,
        _ => 0,
    })
}

thread_local! {
    static HILO_INNER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct HiloInner;

impl HiloInner {
    fn enter() -> Self {
        HILO_INNER.with(|inner| inner.set(true));
        HiloInner
    }
}

impl Drop for HiloInner {
    fn drop(&mut self) {
        HILO_INNER.with(|inner| inner.set(false));
    }
}

/// Rows `[x; residual(x)]` for the two-part encoding, or `None` when it is
/// off or not needed.
fn hilo_stack(activations: &[f32], m: usize, k: usize) -> Option<Vec<f32>> {
    let mode = hilo_mode();
    if mode == 0 || m < 2 || k == 0 || activations.len() != m * k || HILO_INNER.with(|inner| inner.get()) {
        return None;
    }
    if mode == 1 && k <= 4096 && k.is_multiple_of(512) {
        return None;
    }
    let mut stacked = vec![0.0f32; 2 * m * k];
    let (hi, lo) = stacked.split_at_mut(m * k);
    hi.copy_from_slice(activations);
    let ok = activations
        .par_chunks_exact(k)
        .zip(lo.par_chunks_exact_mut(k))
        .all(|(row, residual)| {
            let mut q = vec![0i8; k];
            let Some(scale) = quantize_symmetric_into(row, &mut q) else {
                return false;
            };
            for ((r, &x), &qv) in residual.iter_mut().zip(row).zip(&q) {
                *r = x - f32::from(qv) * scale;
            }
            true
        });
    ok.then_some(stacked)
}

/// `output[i] = stacked[i] + stacked[len + i]` for the two output halves.
fn hilo_sum(output: &mut [f32], stacked: &[f32]) {
    let (hi, lo) = stacked.split_at(output.len());
    output
        .par_chunks_mut(4096)
        .zip(hi.par_chunks(4096).zip(lo.par_chunks(4096)))
        .for_each(|(out, (h, l))| {
            for ((o, &a), &b) in out.iter_mut().zip(h).zip(l) {
                *o = a + b;
            }
        });
}

/// Split a prompt-batch projection the M-tile kernels cannot take in one
/// piece. Returns `None` when the shape needs no split.
///
/// * Tiles taller than 64 rows allow at most 2048 K per core, so a 128-row
///   tile whose K is neither <= 2048 nor K-splittable into 2048-wide core
///   slices (4096 < K <= 6144) runs as two 64-row halves.
/// * Projections wider than one tile (N > 8192, e.g. an 8960-wide FFN) run
///   as column chunks of at most 8192 weight rows.
///
/// `call(weights, bytes, activations, output, rows, cols)` runs one piece.
///
/// # Safety
/// `weights` must hold `weights_bytes` bytes of `n` equally sized rows,
/// `activations` `m * k` floats and `output` `m * n` floats.
unsafe fn split_wide_mtile(
    m: usize,
    k: usize,
    n: usize,
    weights: *const u8,
    weights_bytes: usize,
    activations: *const f32,
    output: *mut f32,
    call: impl Fn(*const u8, usize, *const f32, *mut f32, usize, usize) -> i32,
) -> Option<i32> {
    let kp = k.next_multiple_of(512);
    if m == 128 && kp > 2048 && !(kp > 4096 && kp <= 3 * 2048) {
        for half in 0..2 {
            let status = unsafe {
                call(
                    weights,
                    weights_bytes,
                    activations.add(half * 64 * k),
                    output.add(half * 64 * n),
                    64,
                    n,
                )
            };
            if status != STATUS_OK {
                return Some(status);
            }
        }
        return Some(STATUS_OK);
    }
    if n > 8192 && n.is_multiple_of(32) && weights_bytes.is_multiple_of(n) {
        let row_bytes = weights_bytes / n;
        let per = (n / 32).div_ceil(n.div_ceil(8192)) * 32;
        let mut tmp = vec![0.0f32; m * per];
        let mut n0 = 0;
        while n0 < n {
            let cols = per.min(n - n0);
            let status = call(
                unsafe { weights.add(n0 * row_bytes) },
                cols * row_bytes,
                activations,
                tmp.as_mut_ptr(),
                m,
                cols,
            );
            if status != STATUS_OK {
                return Some(status);
            }
            for row in 0..m {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        tmp.as_ptr().add(row * cols),
                        output.add(row * n + n0),
                        cols,
                    );
                }
            }
            n0 += cols;
        }
        return Some(STATUS_OK);
    }
    None
}

/// Prompt projections whose K is a multiple of 256 but not of 512 (e.g. an
/// 8960-wide FFN down projection) run with K zero-padded to the next
/// multiple of 512. Exact: the padded weights and activations are zero.
fn execute_padded_k_mtile<F>(
    context: &mut RockNpuContext,
    key: DecodeWeightKey,
    m: usize,
    activations_mk_f32: &[f32],
    output_mn_f32: &mut [f32],
    prepare: F,
) -> i32
where
    F: FnOnce() -> Option<(Vec<i8>, Vec<f32>)>,
{
    let k = key.k;
    let kp = k.next_multiple_of(512);
    if activations_mk_f32.len() != m * k {
        return STATUS_INVALID_ARGUMENT;
    }
    let mut padded = vec![0.0f32; m * kp];
    for (dst, src) in padded.chunks_exact_mut(kp).zip(activations_mk_f32.chunks_exact(k)) {
        dst[..k].copy_from_slice(src);
    }
    let key = DecodeWeightKey { k: kp, ..key };
    let prepare = || {
        let (weights, scales) = prepare()?;
        let mut padded_weights = vec![0i8; key.n * kp];
        for (dst, src) in padded_weights.chunks_exact_mut(kp).zip(weights.chunks_exact(k)) {
            dst[..k].copy_from_slice(src);
        }
        Some((padded_weights, scales))
    };
    if kp > 4096 {
        execute_cached_w8a8_mtile_pool(context, key, m, Int8DecodeSplit::K, &padded, output_mn_f32, prepare)
    } else if key.n >= MTILE_POOL_MIN_N {
        execute_cached_w8a8_mtile_pool(context, key, m, Int8DecodeSplit::N, &padded, output_mn_f32, prepare)
    } else {
        execute_cached_w8a8_mtile(context, key, m, &padded, output_mn_f32, prepare)
    }
}

/// Run a row-independent M-row projection as native M-tiles: full 128-row
/// tiles plus a tail zero-padded up to the next supported tile height.
/// `run(activations, outputs, tile_m)` executes one tile.
///
/// # Safety
/// `activations` must hold `m * k` floats and `outputs[i]` `m * ns[i]` floats.
unsafe fn run_mtile_row_chunks(
    m: usize,
    k: usize,
    activations: *const f32,
    outputs: &[*mut f32],
    ns: &[usize],
    mut run: impl FnMut(*const f32, &[*mut f32], usize) -> i32,
) -> i32 {
    let mut row = 0usize;
    while row < m {
        let rows = (m - row).min(128);
        let tile = MTILE_ROWS.iter().copied().find(|&t| t >= rows).unwrap_or(128);
        let a = unsafe { activations.add(row * k) };
        let status = if tile == rows {
            let outs: Vec<*mut f32> = outputs
                .iter()
                .zip(ns)
                .map(|(&out, &n)| unsafe { out.add(row * n) })
                .collect();
            run(a, &outs, tile)
        } else {
            let mut padded_a = vec![0.0f32; tile * k];
            padded_a[..rows * k].copy_from_slice(unsafe { slice::from_raw_parts(a, rows * k) });
            let mut padded_out: Vec<Vec<f32>> = ns.iter().map(|&n| vec![0.0f32; tile * n]).collect();
            let outs: Vec<*mut f32> = padded_out.iter_mut().map(|o| o.as_mut_ptr()).collect();
            let status = run(padded_a.as_ptr(), &outs, tile);
            if status == STATUS_OK {
                for ((&out, &n), tmp) in outputs.iter().zip(ns).zip(&padded_out) {
                    unsafe { slice::from_raw_parts_mut(out.add(row * n), rows * n) }
                        .copy_from_slice(&tmp[..rows * n]);
                }
            }
            status
        };
        if status != STATUS_OK {
            return status;
        }
        row += rows;
    }
    STATUS_OK
}

/// Direct-submit M-tile execution of 1..=4 same-input projections whose W8
/// rows are concatenated along N into one resident matrix: activations are
/// quantized, staged and submitted once, and the output columns are rescaled
/// back into each projection's own [M, n_i] buffer.
fn execute_cached_w8a8_mtile_concat<F>(
    context: &mut RockNpuContext,
    keys: Vec<DecodeWeightKey>,
    m: usize,
    activations_mk_f32: &[f32],
    outputs: &mut [&mut [f32]],
    prepare: F,
) -> i32
where
    F: FnOnce() -> Option<(Vec<i8>, Vec<f32>)>,
{
    if !matches!(m, 4 | 8 | 12 | 16 | 32 | 48 | 64 | 128)
        || keys.is_empty()
        || keys.len() != outputs.len()
        || !context.decode_pool.direct_enabled()
    {
        return STATUS_INVALID_ARGUMENT;
    }
    let k = keys[0].k;
    let total_n: usize = keys.iter().map(|key| key.n).sum();
    if keys.iter().any(|key| key.k != k || !key.n.is_multiple_of(32))
        || k == 0
        || !k.is_multiple_of(512)
        || k > if m > 64 { 2048 } else { 4096 }
        || total_n < 96
        || total_n > 3 * 8192
        || activations_mk_f32.len() != m.saturating_mul(k)
        || outputs
            .iter()
            .zip(&keys)
            .any(|(out, key)| out.len() != m.saturating_mul(key.n))
    {
        return STATUS_INVALID_ARGUMENT;
    }

    let quant_started = context.mtile_profile.as_ref().map(|_| Instant::now());
    let mut activations_i8 = vec![0i8; m * k];
    let mut activation_scales = vec![0.0f32; m];
    let quantized_ok = activations_mk_f32
        .par_chunks_exact(k)
        .zip(activations_i8.par_chunks_exact_mut(k))
        .zip(activation_scales.par_iter_mut())
        .all(|((row, out), scale)| match quantize_symmetric_into(row, out) {
            Some(row_scale) => {
                *scale = row_scale;
                true
            }
            None => false,
        });
    if !quantized_ok {
        return STATUS_INVALID_ARGUMENT;
    }
    if let (Some(profile), Some(started)) = (context.mtile_profile.as_mut(), quant_started) {
        profile.quant_ns += started.elapsed().as_nanos();
    }

    let RockNpuContext {
        decode_pool,
        decode_mtile_concat_weights,
        mtile_profile,
        ..
    } = context;
    let cached = match decode_mtile_concat_weights.entry(keys.clone()) {
        Entry::Occupied(entry) => {
            if let Some(profile) = mtile_profile.as_mut() {
                profile.cache_hits += 1;
            }
            entry.into_mut()
        }
        Entry::Vacant(entry) => {
            let started = Instant::now();
            let Some((weights_i8, scales)) = prepare() else {
                return STATUS_INVALID_ARGUMENT;
            };
            if weights_i8.len() != total_n.saturating_mul(k) || scales.len() != total_n {
                return STATUS_INVALID_ARGUMENT;
            }
            let workers = (total_n / 32).min(3);
            let prepared = match decode_pool.prepare_weights_mtile_direct_with_split(
                Arc::<[i8]>::from(weights_i8),
                k,
                total_n,
                workers,
                Int8DecodeSplit::N,
            ) {
                Ok(prepared) => prepared,
                Err(err) => {
                    if env_enabled("ROCKNPU_MTILE_TRACE") {
                        eprintln!("ROCKNPU MTILE ERROR concat prepare M={m} K={k} N={total_n}: {err}");
                    }
                    return STATUS_EXECUTION_ERROR;
                }
            };
            if let Some(profile) = mtile_profile.as_mut() {
                profile.cache_misses += 1;
                profile.weight_prepare_ns += started.elapsed().as_nanos();
            }
            entry.insert(CachedW8MtilePoolWeight { prepared, scales })
        }
    };
    if !cached.prepared.is_direct() {
        return STATUS_EXECUTION_ERROR;
    }

    let started = Instant::now();
    let scales = &cached.scales;
    let a_scales = &activation_scales;
    // Column start of each member within the concatenated N.
    let mut member_starts = Vec::with_capacity(keys.len());
    let mut column = 0usize;
    for key in &keys {
        member_starts.push(column);
        column += key.n;
    }
    let targets: Vec<SharedMut<f32>> = outputs.iter_mut().map(|out| SharedMut::new(out)).collect();
    let sink = |_: usize, slice: WorkerSlice, values: &[i32]| {
        for (member, key) in keys.iter().enumerate() {
            let lo = member_starts[member].max(slice.n0);
            let hi = (member_starts[member] + key.n).min(slice.n0 + slice.nsub);
            if lo >= hi {
                continue;
            }
            for row in 0..m {
                let src = row * slice.nsub + (lo - slice.n0);
                // SAFETY: cores own disjoint concatenated column ranges.
                let dst = unsafe {
                    targets[member].range(row * key.n + (lo - member_starts[member]), hi - lo)
                };
                rescale_i32_row(&values[src..src + (hi - lo)], &scales[lo..hi], a_scales[row], dst);
            }
        }
    };
    let timings = match decode_pool.execute_prepared_mtile_direct(
        m,
        &activations_i8,
        &cached.prepared,
        &sink,
    ) {
        Ok(timings) => timings,
        Err(err) => {
            if env_enabled("ROCKNPU_MTILE_TRACE") {
                eprintln!("ROCKNPU MTILE ERROR concat execute M={m} K={k} N={total_n}: {err}");
            }
            return STATUS_EXECUTION_ERROR;
        }
    };
    if let Some(profile) = mtile_profile.as_mut() {
        profile.calls += 1;
        profile.execute_total_ns += started.elapsed().as_nanos();
        profile.input_stage_ns += timings.stage_submit_ns;
        profile.wait_ns += timings.wait_ns;
        profile.host_accum_ns += timings.consume_ns;
    }
    STATUS_OK
}

/// Run 1..=4 same-input Q4_K/Q6_K projections as one concatenated-N direct
/// M-tile NPU call. `kinds` holds 4 (Q4_K) or 6 (Q6_K) per projection.
/// Returns an error status when the grouped path is unavailable so callers
/// can fall back to per-projection execution.
///
/// # Safety
/// Every pointer array must hold `count` entries whose buffers satisfy the
/// stated lengths (weights `bytes[i]`, outputs `m * ns[i]` floats) and the
/// activations `m * k` floats, for the duration of this synchronous call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocknpu_matmul_q_concat_f32_f32_mtile(
    context: *mut RockNpuContext,
    count: usize,
    weights: *const *const u8,
    bytes: *const usize,
    kinds: *const u32,
    ns: *const usize,
    activations_mk_f32: *const f32,
    outputs: *const *mut f32,
    m: usize,
    k: usize,
) -> i32 {
    if context.is_null()
        || !(1..=4).contains(&count)
        || weights.is_null()
        || bytes.is_null()
        || kinds.is_null()
        || ns.is_null()
        || activations_mk_f32.is_null()
        || outputs.is_null()
        || m == 0
        || k == 0
    {
        return STATUS_INVALID_ARGUMENT;
    }
    {
        let activations = unsafe { slice::from_raw_parts(activations_mk_f32, m * k) };
        if let Some(stacked) = hilo_stack(activations, m, k) {
            let (ns_slice, outs_slice) =
                unsafe { (slice::from_raw_parts(ns, count), slice::from_raw_parts(outputs, count)) };
            let mut stacked_outs: Vec<Vec<f32>> = ns_slice.iter().map(|&cols| vec![0.0f32; 2 * m * cols]).collect();
            let ptrs: Vec<*mut f32> = stacked_outs.iter_mut().map(|out| out.as_mut_ptr()).collect();
            let status = {
                let _inner = HiloInner::enter();
                unsafe {
                    rocknpu_matmul_q_concat_f32_f32_mtile(
                        context,
                        count,
                        weights,
                        bytes,
                        kinds,
                        ns,
                        stacked.as_ptr(),
                        ptrs.as_ptr(),
                        2 * m,
                        k,
                    )
                }
            };
            if status == STATUS_OK {
                for ((&out, &cols), stacked_out) in outs_slice.iter().zip(ns_slice).zip(&stacked_outs) {
                    if out.is_null() {
                        return STATUS_INVALID_ARGUMENT;
                    }
                    hilo_sum(unsafe { slice::from_raw_parts_mut(out, m * cols) }, stacked_out);
                }
            }
            return status;
        }
    }
    if !MTILE_ROWS.contains(&m) {
        let (ns_slice, outs_slice) =
            unsafe { (slice::from_raw_parts(ns, count), slice::from_raw_parts(outputs, count)) };
        return unsafe {
            run_mtile_row_chunks(m, k, activations_mk_f32, outs_slice, ns_slice, |a, outs, tile| {
                rocknpu_matmul_q_concat_f32_f32_mtile(
                    context, count, weights, bytes, kinds, ns, a, outs.as_ptr(), tile, k,
                )
            })
        };
    }
    if m == 128 && k > 2048 {
        // Tiles taller than 64 rows allow at most 2048 K: two 64-row halves.
        let (ns_slice, outs_slice) =
            unsafe { (slice::from_raw_parts(ns, count), slice::from_raw_parts(outputs, count)) };
        for half in 0..2 {
            let outs: Vec<*mut f32> = outs_slice
                .iter()
                .zip(ns_slice)
                .map(|(&out, &cols)| out.wrapping_add(half * 64 * cols))
                .collect();
            let status = unsafe {
                rocknpu_matmul_q_concat_f32_f32_mtile(
                    context,
                    count,
                    weights,
                    bytes,
                    kinds,
                    ns,
                    activations_mk_f32.add(half * 64 * k),
                    outs.as_ptr(),
                    64,
                    k,
                )
            };
            if status != STATUS_OK {
                return status;
            }
        }
        return STATUS_OK;
    }
    let (weights, bytes, kinds, ns, outputs_raw) = unsafe {
        (
            slice::from_raw_parts(weights, count),
            slice::from_raw_parts(bytes, count),
            slice::from_raw_parts(kinds, count),
            slice::from_raw_parts(ns, count),
            slice::from_raw_parts(outputs, count),
        )
    };
    let mut keys = Vec::with_capacity(count);
    let mut sources = Vec::with_capacity(count);
    for i in 0..count {
        if weights[i].is_null() || outputs_raw[i].is_null() || ns[i] == 0 {
            return STATUS_INVALID_ARGUMENT;
        }
        let kind = match kinds[i] {
            4 => DecodeWeightKind::Q4K,
            6 => DecodeWeightKind::Q6K,
            _ => return STATUS_INVALID_ARGUMENT,
        };
        let source = unsafe { slice::from_raw_parts(weights[i], bytes[i]) };
        keys.push(DecodeWeightKey {
            address: source.as_ptr() as usize,
            bytes: bytes[i],
            k,
            n: ns[i],
            kind,
        });
        sources.push(source);
    }
    let Some(a_len) = m.checked_mul(k) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let activations = unsafe { slice::from_raw_parts(activations_mk_f32, a_len) };
    let mut outputs: Vec<&mut [f32]> = Vec::with_capacity(count);
    for i in 0..count {
        let Some(len) = m.checked_mul(ns[i]) else {
            return STATUS_INVALID_ARGUMENT;
        };
        outputs.push(unsafe { slice::from_raw_parts_mut(outputs_raw[i], len) });
    }
    let context = unsafe { &mut *context };
    let prepare_keys = keys.clone();
    execute_cached_w8a8_mtile_concat(context, keys, m, activations, &mut outputs, move || {
        let mut all_weights = Vec::new();
        let mut all_scales = Vec::new();
        for (key, source) in prepare_keys.iter().zip(sources) {
            let (w, sc) = match key.kind {
                DecodeWeightKind::Q4K => prepare_q4_k_w8a8(source, key.k, key.n)?,
                DecodeWeightKind::Q6K => prepare_q6_k_w8a8(source, key.k, key.n)?,
            };
            all_weights.extend_from_slice(&w);
            all_scales.extend_from_slice(&sc);
        }
        Some((all_weights, all_scales))
    })
}

fn execute_cached_w8a8_mtile_pair_pool<F>(
    context: &mut RockNpuContext,
    key: DecodePairKey,
    m: usize,
    activations_mk_f32: &[f32],
    first_output_mn_f32: &mut [f32],
    second_output_mn_f32: &mut [f32],
    prepare: F,
) -> i32
where
    F: FnOnce() -> Option<((Vec<i8>, Vec<f32>), (Vec<i8>, Vec<f32>))>,
{
    if !matches!(m, 4 | 8 | 12 | 16 | 32 | 48 | 64 | 128)
        || key.first.k != key.second.k
        || key.first.n != key.second.n
        || key.first.k == 0
        || key.first.k > 4096
        || !key.first.k.is_multiple_of(512)
        || key.first.n < 2048
        || key.first.n > 8192
        || !key.first.n.is_multiple_of(32)
    {
        return STATUS_INVALID_ARGUMENT;
    }
    let k = key.first.k;
    let n = key.first.n;
    let Some(expected_a) = m.checked_mul(k) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(expected_out) = m.checked_mul(n) else {
        return STATUS_INVALID_ARGUMENT;
    };
    if activations_mk_f32.len() != expected_a
        || first_output_mn_f32.len() != expected_out
        || second_output_mn_f32.len() != expected_out
    {
        return STATUS_INVALID_ARGUMENT;
    }

    let quant_started = context.mtile_profile.as_ref().map(|_| Instant::now());
    let mut activations_i8 = vec![0i8; expected_a];
    let mut activation_scales = vec![0.0f32; m];
    for ((row, out), scale) in activations_mk_f32
        .chunks_exact(k)
        .zip(activations_i8.chunks_exact_mut(k))
        .zip(activation_scales.iter_mut())
    {
        let Some(row_scale) = quantize_symmetric_into(row, out) else {
            return STATUS_INVALID_ARGUMENT;
        };
        *scale = row_scale;
    }
    if let (Some(profile), Some(started)) = (context.mtile_profile.as_mut(), quant_started) {
        profile.quant_ns += started.elapsed().as_nanos();
    }
    let activation: Arc<[i8]> = Arc::from(activations_i8);

    let RockNpuContext {
        decode_pool,
        decode_mtile_pair_pool_weights,
        mtile_profile,
        ..
    } = context;

    let mut cache_miss = false;
    let mut weight_prepare_ns = 0u128;
    let mut weight_pack_ns = 0u128;
    let cached = match decode_mtile_pair_pool_weights.entry(key) {
        Entry::Occupied(entry) => entry.into_mut(),
        Entry::Vacant(entry) => {
            cache_miss = true;
            let started = Instant::now();
            let Some(((first_weights, first_scales), (second_weights, second_scales))) = prepare() else {
                return STATUS_INVALID_ARGUMENT;
            };
            if first_weights.len() != n.saturating_mul(k)
                || second_weights.len() != n.saturating_mul(k)
                || first_scales.len() != n
                || second_scales.len() != n
            {
                return STATUS_INVALID_ARGUMENT;
            }
            weight_prepare_ns = started.elapsed().as_nanos();
            let started = Instant::now();
            let first = match decode_pool.prepare_weights_mtile_with_split(
                Arc::<[i8]>::from(first_weights),
                k,
                n,
                3,
                Int8DecodeSplit::N,
            ) {
                Ok(prepared) => prepared,
                Err(_) => return STATUS_EXECUTION_ERROR,
            };
            let second = match decode_pool.prepare_weights_mtile_with_split(
                Arc::<[i8]>::from(second_weights),
                k,
                n,
                3,
                Int8DecodeSplit::N,
            ) {
                Ok(prepared) => prepared,
                Err(_) => return STATUS_EXECUTION_ERROR,
            };
            weight_pack_ns = started.elapsed().as_nanos();
            entry.insert(CachedW8MtilePairPoolWeight {
                first,
                second,
                first_scales,
                second_scales,
            })
        }
    };

    if let Some(profile) = mtile_profile.as_mut() {
        if cache_miss {
            profile.cache_misses += 2;
            profile.weight_prepare_ns += weight_prepare_ns;
            profile.weight_pack_ns += weight_pack_ns;
        } else {
            profile.cache_hits += 2;
        }
    }

    let activations = [Arc::clone(&activation), activation];
    let weights = [&cached.first, &cached.second];
    let result = match decode_pool.execute_prepared_mtile_batch(m, &activations, &weights) {
        Ok(result) => result,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };

    if let Some(profile) = mtile_profile.as_mut() {
        profile.calls += 2;
        profile.execute_total_ns += result.stats.wall_ns;
        for stats in &result.stats.worker_stats {
            profile.alloc_ns += stats.alloc_ns;
            profile.input_stage_ns += stats.input_stage_ns;
            profile.regcmd_stage_ns += stats.regcmd_stage_ns;
            profile.submit_ns += stats.submit_ns;
            profile.wait_ns += stats.wait_ns;
            profile.host_accum_ns += stats.host_accum_ns;
        }
    }

    let rescale_started = mtile_profile.as_ref().map(|_| Instant::now());
    for row in 0..m {
        let start = row * n;
        let end = start + n;
        rescale_i32_row(
            &result.values[0][start..end],
            &cached.first_scales,
            activation_scales[row],
            &mut first_output_mn_f32[start..end],
        );
        rescale_i32_row(
            &result.values[1][start..end],
            &cached.second_scales,
            activation_scales[row],
            &mut second_output_mn_f32[start..end],
        );
    }
    if let (Some(profile), Some(started)) = (mtile_profile.as_mut(), rescale_started) {
        profile.rescale_ns += started.elapsed().as_nanos();
    }
    STATUS_OK
}

fn execute_cached_w8a8_m16_pool<F>(
    context: &mut RockNpuContext,
    key: DecodeWeightKey,
    activations_mk_f32: &[f32],
    output_mn_f32: &mut [f32],
    prepare: F,
) -> i32
where
    F: FnOnce() -> Option<(Vec<i8>, Vec<f32>)>,
{
    execute_cached_w8a8_mtile_pool(
        context,
        key,
        16,
        Int8DecodeSplit::N,
        activations_mk_f32,
        output_mn_f32,
        prepare,
    )
}

fn execute_cached_w8a8_mtile<F>(
    context: &mut RockNpuContext,
    key: DecodeWeightKey,
    m: usize,
    activations_mk_f32: &[f32],
    output_mn_f32: &mut [f32],
    prepare: F,
) -> i32
where
    F: FnOnce() -> Option<(Vec<i8>, Vec<f32>)>,
{
    if !matches!(m, 4 | 8 | 12 | 16 | 32 | 48 | 64 | 128) {
        return STATUS_INVALID_ARGUMENT;
    }
    let Some(expected_a) = m.checked_mul(key.k) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(expected_out) = m.checked_mul(key.n) else {
        return STATUS_INVALID_ARGUMENT;
    };
    if activations_mk_f32.len() != expected_a
        || output_mn_f32.len() != expected_out
        || !(key.k <= 4096 || key.k == 5632)
        || !key.k.is_multiple_of(512)
        || !key.n.is_multiple_of(32)
        || key.n > 8192
    {
        return STATUS_INVALID_ARGUMENT;
    }

    let quant_started = context.mtile_profile.as_ref().map(|_| Instant::now());
    let mut activations_i8 = vec![0i8; expected_a];
    let mut activation_scales = vec![0.0f32; m];
    for ((row, out), scale) in activations_mk_f32
        .chunks_exact(key.k)
        .zip(activations_i8.chunks_exact_mut(key.k))
        .zip(activation_scales.iter_mut())
    {
        let Some(row_scale) = quantize_symmetric_into(row, out) else {
            return STATUS_INVALID_ARGUMENT;
        };
        *scale = row_scale;
    }
    if let (Some(profile), Some(started)) = (context.mtile_profile.as_mut(), quant_started) {
        profile.quant_ns += started.elapsed().as_nanos();
    }

    let executor = Int8DecodeExecutor::from_externally_guarded_device(&context.device);
    let mut cache_miss = false;
    let mut weight_prepare_ns = 0u128;
    let mut weight_pack_ns = 0u128;
    let cached = match context.decode_mtile_weights.entry(key) {
        Entry::Occupied(entry) => entry.into_mut(),
        Entry::Vacant(entry) => {
            cache_miss = true;
            let started = Instant::now();
            let Some((weights_i8, scales)) = prepare() else {
                return STATUS_INVALID_ARGUMENT;
            };
            weight_prepare_ns = started.elapsed().as_nanos();
            let started = Instant::now();
            let prepared = match executor.prepare_weights(&weights_i8, key.k, key.n) {
                Ok(prepared) => prepared,
                Err(_) => return STATUS_EXECUTION_ERROR,
            };
            weight_pack_ns = started.elapsed().as_nanos();
            entry.insert(CachedW8MtileWeight { prepared, scales })
        }
    };
    if let Some(profile) = context.mtile_profile.as_mut() {
        if cache_miss {
            profile.cache_misses += 1;
            profile.weight_prepare_ns += weight_prepare_ns;
            profile.weight_pack_ns += weight_pack_ns;
        } else {
            profile.cache_hits += 1;
        }
    }

    let result = if key.k > 4096 {
        if m != 16 {
            return STATUS_INVALID_ARGUMENT;
        }
        match executor.execute_prepared_m16(&activations_i8, &cached.prepared) {
            Ok(result) => result,
            Err(_) => return STATUS_EXECUTION_ERROR,
        }
    } else if env_enabled("ROCKNPU_MTILE_PERSIST") {
        let scratch_slot = context
            .decode_mtile_scratch
            .entry((key.k, key.n))
            .or_insert(None);
        match executor.execute_prepared_mtile_persistent(
            m,
            &activations_i8,
            &cached.prepared,
            scratch_slot,
        ) {
            Ok(result) => result,
            Err(_) => return STATUS_EXECUTION_ERROR,
        }
    } else if m == 16 {
        match executor.execute_prepared_m16_single(&activations_i8, &cached.prepared) {
            Ok(result) => result,
            Err(_) => return STATUS_EXECUTION_ERROR,
        }
    } else {
        return STATUS_INVALID_ARGUMENT;
    };
    if let Some(profile) = context.mtile_profile.as_mut() {
        profile.calls += 1;
        profile.ksplit_calls += usize::from(result.stats.k_slices > 1);
        profile.alloc_ns += result.stats.alloc_ns;
        profile.input_stage_ns += result.stats.input_stage_ns;
        profile.regcmd_stage_ns += result.stats.regcmd_stage_ns;
        profile.submit_ns += result.stats.submit_ns;
        profile.wait_ns += result.stats.wait_ns;
        profile.host_accum_ns += result.stats.host_accum_ns;
        profile.execute_total_ns += result.stats.total_ns;
    }
    let rescale_started = context.mtile_profile.as_ref().map(|_| Instant::now());
    for row in 0..m {
        let start = row * key.n;
        let end = start + key.n;
        rescale_i32_row(
            &result.values[start..end],
            &cached.scales,
            activation_scales[row],
            &mut output_mn_f32[start..end],
        );
    }
    if let (Some(profile), Some(started)) = (context.mtile_profile.as_mut(), rescale_started) {
        profile.rescale_ns += started.elapsed().as_nanos();
    }
    STATUS_OK
}

fn execute_cached_w8a8_m16<F>(
    context: &mut RockNpuContext,
    key: DecodeWeightKey,
    activations_mk_f32: &[f32],
    output_mn_f32: &mut [f32],
    prepare: F,
) -> i32
where
    F: FnOnce() -> Option<(Vec<i8>, Vec<f32>)>,
{
    execute_cached_w8a8_mtile(context, key, 16, activations_mk_f32, output_mn_f32, prepare)
}

fn execute_cached_grouped_w8a8_m16_pool<F>(
    context: &mut RockNpuContext,
    key: DecodeWeightKey,
    group_size: usize,
    activations_mk_f32: &[f32],
    output_mn_f32: &mut [f32],
    prepare: F,
) -> i32
where
    F: FnOnce() -> Option<(Vec<i8>, Vec<f32>)>,
{
    const M: usize = 16;
    let groups = key.k / group_size;
    let Some(expected_a) = M.checked_mul(key.k) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(expected_out) = M.checked_mul(key.n) else {
        return STATUS_INVALID_ARGUMENT;
    };
    if groups < 2
        || !group_size.is_multiple_of(512)
        || !key.k.is_multiple_of(group_size)
        || group_size > 4096
        || key.n < 2048
        || !key.n.is_multiple_of(32)
        || key.n > 8192
        || activations_mk_f32.len() != expected_a
        || output_mn_f32.len() != expected_out
    {
        return STATUS_INVALID_ARGUMENT;
    }

    let quant_started = context.mtile_profile.as_ref().map(|_| Instant::now());
    let mut grouped_activations = (0..groups)
        .map(|_| Vec::with_capacity(M * group_size))
        .collect::<Vec<_>>();
    let mut activation_scales = vec![0.0f32; groups * M];
    for (row_index, row) in activations_mk_f32.chunks_exact(key.k).enumerate() {
        for (group_index, group) in row.chunks_exact(group_size).enumerate() {
            let Some((q, scale)) = quantize_symmetric(group) else {
                return STATUS_INVALID_ARGUMENT;
            };
            grouped_activations[group_index].extend_from_slice(&q);
            activation_scales[group_index * M + row_index] = scale;
        }
    }
    if let (Some(profile), Some(started)) = (context.mtile_profile.as_mut(), quant_started) {
        profile.quant_ns += started.elapsed().as_nanos();
    }

    let RockNpuContext {
        decode_pool,
        decode_grouped_mtile_pool_weights,
        mtile_profile,
        ..
    } = context;
    let cache_key = (key, group_size);
    let mut cache_miss = false;
    let mut weight_prepare_ns = 0u128;
    let mut weight_pack_ns = 0u128;
    let cached = match decode_grouped_mtile_pool_weights.entry(cache_key) {
        Entry::Occupied(entry) => entry.into_mut(),
        Entry::Vacant(entry) => {
            cache_miss = true;
            let started = Instant::now();
            let Some((weights_i8, scales)) = prepare() else {
                return STATUS_INVALID_ARGUMENT;
            };
            weight_prepare_ns = started.elapsed().as_nanos();
            if weights_i8.len() != key.n.saturating_mul(key.k)
                || scales.len() != groups.saturating_mul(key.n)
            {
                return STATUS_INVALID_ARGUMENT;
            }

            let group_matrix = key.n.saturating_mul(group_size);
            let mut prepared = Vec::with_capacity(groups);
            let started = Instant::now();
            for group_index in 0..groups {
                let start = group_index.saturating_mul(group_matrix);
                let end = start.saturating_add(group_matrix);
                let Some(group_weights) = weights_i8.get(start..end) else {
                    return STATUS_INVALID_ARGUMENT;
                };
                let group_prepared = match decode_pool.prepare_weights_mtile_with_split(
                    Arc::<[i8]>::from(group_weights),
                    group_size,
                    key.n,
                    3,
                    Int8DecodeSplit::N,
                ) {
                    Ok(prepared) => prepared,
                    Err(_) => return STATUS_EXECUTION_ERROR,
                };
                prepared.push(group_prepared);
            }
            weight_pack_ns = started.elapsed().as_nanos();
            entry.insert(CachedGroupedW8MtilePoolWeight { prepared, scales })
        }
    };

    if let Some(profile) = mtile_profile.as_mut() {
        if cache_miss {
            profile.cache_misses += 1;
            profile.weight_prepare_ns += weight_prepare_ns;
            profile.weight_pack_ns += weight_pack_ns;
        } else {
            profile.cache_hits += 1;
        }
    }

    if cached.prepared.len() != groups {
        return STATUS_EXECUTION_ERROR;
    }
    output_mn_f32.fill(0.0);

    for group_index in 0..groups {
        let result = match decode_pool.execute_prepared_m16(
            Arc::<[i8]>::from(grouped_activations[group_index].as_slice()),
            &cached.prepared[group_index],
        ) {
            Ok(result) => result,
            Err(_) => return STATUS_EXECUTION_ERROR,
        };
        if let Some(profile) = mtile_profile.as_mut() {
            profile.calls += 1;
            profile.execute_total_ns += result.stats.wall_ns;
            for stats in &result.stats.worker_stats {
                profile.alloc_ns += stats.alloc_ns;
                profile.input_stage_ns += stats.input_stage_ns;
                profile.regcmd_stage_ns += stats.regcmd_stage_ns;
                profile.submit_ns += stats.submit_ns;
                profile.wait_ns += stats.wait_ns;
                profile.host_accum_ns += stats.host_accum_ns;
            }
        }

        let rescale_started = mtile_profile.as_ref().map(|_| Instant::now());
        let weight_scales = &cached.scales[group_index * key.n..(group_index + 1) * key.n];
        for row in 0..M {
            let activation_scale = activation_scales[group_index * M + row];
            let row_start = row * key.n;
            for col in 0..key.n {
                output_mn_f32[row_start + col] +=
                    result.values[row_start + col] as f32 * activation_scale * weight_scales[col];
            }
        }
        if let (Some(profile), Some(started)) = (mtile_profile.as_mut(), rescale_started) {
            profile.rescale_ns += started.elapsed().as_nanos();
        }
    }

    STATUS_OK
}

fn execute_cached_grouped_w8a8_m16<F>(
    context: &mut RockNpuContext,
    key: DecodeWeightKey,
    group_size: usize,
    activations_mk_f32: &[f32],
    output_mn_f32: &mut [f32],
    prepare: F,
) -> i32
where
    F: FnOnce() -> Option<(Vec<i8>, Vec<f32>)>,
{
    const M: usize = 16;
    let groups = key.k / group_size;
    let Some(expected_a) = M.checked_mul(key.k) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(expected_out) = M.checked_mul(key.n) else {
        return STATUS_INVALID_ARGUMENT;
    };
    if groups < 2
        || !group_size.is_multiple_of(512)
        || !key.k.is_multiple_of(group_size)
        || group_size > 4096
        || activations_mk_f32.len() != expected_a
        || output_mn_f32.len() != expected_out
    {
        return STATUS_INVALID_ARGUMENT;
    }

    let quant_started = context.mtile_profile.as_ref().map(|_| Instant::now());
    let mut grouped_activations = (0..groups)
        .map(|_| Vec::with_capacity(M * group_size))
        .collect::<Vec<_>>();
    let mut activation_scales = vec![0.0f32; groups * M];
    for (row_index, row) in activations_mk_f32.chunks_exact(key.k).enumerate() {
        for (group_index, group) in row.chunks_exact(group_size).enumerate() {
            let Some((q, scale)) = quantize_symmetric(group) else {
                return STATUS_INVALID_ARGUMENT;
            };
            grouped_activations[group_index].extend_from_slice(&q);
            activation_scales[group_index * M + row_index] = scale;
        }
    }
    if let (Some(profile), Some(started)) = (context.mtile_profile.as_mut(), quant_started) {
        profile.quant_ns += started.elapsed().as_nanos();
    }

    let persistent = env_enabled("ROCKNPU_MTILE_PERSIST");
    let RockNpuContext {
        device,
        decode_grouped_mtile_weights,
        decode_mtile_scratch,
        ..
    } = context;
    let executor = Int8DecodeExecutor::from_externally_guarded_device(device);
    let cache_key = (key, group_size);
    let mut cache_miss = false;
    let mut weight_prepare_ns = 0u128;
    let mut weight_pack_ns = 0u128;
    let cached = match decode_grouped_mtile_weights.entry(cache_key) {
        Entry::Occupied(entry) => entry.into_mut(),
        Entry::Vacant(entry) => {
            cache_miss = true;
            let started = Instant::now();
            let Some((weights_i8, scales)) = prepare() else {
                return STATUS_INVALID_ARGUMENT;
            };
            weight_prepare_ns = started.elapsed().as_nanos();
            if weights_i8.len() != key.n.saturating_mul(key.k)
                || scales.len() != groups.saturating_mul(key.n)
            {
                return STATUS_INVALID_ARGUMENT;
            }
            let group_matrix = key.n.saturating_mul(group_size);
            let mut prepared = Vec::with_capacity(groups);
            let started = Instant::now();
            for group_index in 0..groups {
                let start = group_index.saturating_mul(group_matrix);
                let end = start.saturating_add(group_matrix);
                let Some(group_weights) = weights_i8.get(start..end) else {
                    return STATUS_INVALID_ARGUMENT;
                };
                match executor.prepare_weights(group_weights, group_size, key.n) {
                    Ok(group_prepared) => prepared.push(group_prepared),
                    Err(_) => return STATUS_EXECUTION_ERROR,
                }
            }
            weight_pack_ns = started.elapsed().as_nanos();
            entry.insert(CachedGroupedW8MtileWeight { prepared, scales })
        }
    };
    if let Some(profile) = context.mtile_profile.as_mut() {
        if cache_miss {
            profile.cache_misses += 1;
            profile.weight_prepare_ns += weight_prepare_ns;
            profile.weight_pack_ns += weight_pack_ns;
        } else {
            profile.cache_hits += 1;
        }
    }

    if cached.prepared.len() != groups {
        return STATUS_EXECUTION_ERROR;
    }
    output_mn_f32.fill(0.0);
    for group_index in 0..groups {
        let result = if persistent {
            let scratch_slot = decode_mtile_scratch
                .entry((group_size, key.n))
                .or_insert(None);
            match executor.execute_prepared_m16_persistent(
                &grouped_activations[group_index],
                &cached.prepared[group_index],
                scratch_slot,
            ) {
                Ok(result) => result,
                Err(_) => return STATUS_EXECUTION_ERROR,
            }
        } else {
            match executor.execute_prepared_m16_single(
                &grouped_activations[group_index],
                &cached.prepared[group_index],
            ) {
                Ok(result) => result,
                Err(_) => return STATUS_EXECUTION_ERROR,
            }
        };
        if let Some(profile) = context.mtile_profile.as_mut() {
            profile.calls += 1;
            profile.ksplit_calls += usize::from(result.stats.k_slices > 1);
            profile.alloc_ns += result.stats.alloc_ns;
            profile.input_stage_ns += result.stats.input_stage_ns;
            profile.regcmd_stage_ns += result.stats.regcmd_stage_ns;
            profile.submit_ns += result.stats.submit_ns;
            profile.wait_ns += result.stats.wait_ns;
            profile.host_accum_ns += result.stats.host_accum_ns;
            profile.execute_total_ns += result.stats.total_ns;
        }
        let rescale_started = context.mtile_profile.as_ref().map(|_| Instant::now());
        let weight_scales = &cached.scales[group_index * key.n..(group_index + 1) * key.n];
        for row in 0..M {
            let activation_scale = activation_scales[group_index * M + row];
            let row_start = row * key.n;
            for col in 0..key.n {
                output_mn_f32[row_start + col] +=
                    result.values[row_start + col] as f32 * activation_scale * weight_scales[col];
            }
        }
        if let (Some(profile), Some(started)) = (context.mtile_profile.as_mut(), rescale_started) {
            profile.rescale_ns += started.elapsed().as_nanos();
        }
    }
    STATUS_OK
}

fn execute_cached_w8a8_pair_m1<F>(
    context: &mut RockNpuContext,
    key: DecodePairKey,
    activations_k_f32: &[f32],
    output_first_f32: &mut [f32],
    output_second_f32: &mut [f32],
    prepare: F,
) -> i32
where
    F: FnOnce() -> Option<(Vec<i8>, Vec<f32>)>,
{
    let call_start = Instant::now();
    if key.first.k != key.second.k
        || activations_k_f32.len() != key.first.k
        || output_first_f32.len() != key.first.n
        || output_second_f32.len() != key.second.n
        || !key.first.k.is_multiple_of(512)
        || !key.first.n.is_multiple_of(32)
        || !key.second.n.is_multiple_of(32)
    {
        return STATUS_INVALID_ARGUMENT;
    }
    let Some(total_n) = key.first.n.checked_add(key.second.n) else {
        return STATUS_INVALID_ARGUMENT;
    };
    if total_n > 16384 {
        return STATUS_INVALID_ARGUMENT;
    }
    let quant_started = context.m1_profile.as_ref().map(|_| Instant::now());
    let Some((activations_i8, activation_scale)) = quantize_symmetric(activations_k_f32) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let quant_ns = quant_started.map_or(0, |started| started.elapsed().as_nanos());
    let activation: Arc<[i8]> = Arc::from(activations_i8);

    let RockNpuContext {
        device: _,
        decode_pool,
        decode_worker_cache,
        decode_pair_weights,
        decode_cache_hits,
        decode_cache_misses,
        decode_cache_hit_ns,
        decode_cache_miss_ns,
        decode_worker_calls,
        decode_ksplit_calls,
        m1_profile,
        overlap,
        ..
    } = context;

    let (cached, cache_hit) = match decode_pair_weights.entry(key) {
        Entry::Occupied(entry) => {
            *decode_cache_hits = decode_cache_hits.saturating_add(1);
            (entry.into_mut(), true)
        }
        Entry::Vacant(entry) => {
            let Some((weights_i8, scales)) = prepare() else {
                return STATUS_INVALID_ARGUMENT;
            };
            if weights_i8.len() != total_n.saturating_mul(key.first.k) || scales.len() != total_n {
                return STATUS_INVALID_ARGUMENT;
            }
            let weights: Arc<[i8]> = Arc::from(weights_i8);
            let shape = (key.first.k, total_n);
            let (choice, prepared) = if let Some(&choice) = decode_worker_cache.get(&shape) {
                let prepared = match decode_pool.prepare_weights_m1_with_split(
                    Arc::clone(&weights),
                    key.first.k,
                    total_n,
                    choice.workers,
                    choice.split,
                ) {
                    Ok(prepared) => prepared,
                    Err(_) => return STATUS_EXECUTION_ERROR,
                };
                (choice, prepared)
            } else {
                let (choice, prepared) = if total_n > 8192 {
                    let choice = DecodeChoice {
                        split: Int8DecodeSplit::N,
                        workers: decode_pool.workers().min(3),
                    };
                    let prepared = match decode_pool.prepare_weights_m1_with_split(
                        Arc::clone(&weights),
                        key.first.k,
                        total_n,
                        choice.workers,
                        choice.split,
                    ) {
                        Ok(prepared) => prepared,
                        Err(_) => return STATUS_EXECUTION_ERROR,
                    };
                    (choice, prepared)
                } else {
                    match tune_decode_workers(
                        decode_pool,
                        Arc::clone(&weights),
                        Arc::clone(&activation),
                        key.first.k,
                        total_n,
                    ) {
                        Ok(result) => result,
                        Err(()) => return STATUS_EXECUTION_ERROR,
                    }
                };
                decode_worker_cache.insert(shape, choice);
                (choice, prepared)
            };
            *decode_cache_misses = decode_cache_misses.saturating_add(1);
            (
                entry.insert(CachedW8A8Weight {
                    prepared,
                    scales,
                    choice,
                }),
                false,
            )
        }
    };

    let mut overlap_fn = take_overlap(overlap);
    let result = match decode_pool.execute_prepared_overlap(
        Arc::clone(&activation),
        &cached.prepared,
        overlap_fn.as_mut().map(|f| f as &mut dyn FnMut()),
    ) {
        Ok(result) => result,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    if let Some(calls) = decode_worker_calls.get_mut(cached.choice.workers.saturating_sub(1)) {
        *calls = calls.saturating_add(1);
    }
    if cached.choice.split == Int8DecodeSplit::K {
        *decode_ksplit_calls = decode_ksplit_calls.saturating_add(1);
    }
    let rescale_started = m1_profile.as_ref().map(|_| Instant::now());
    for i in 0..key.first.n {
        output_first_f32[i] = result.values[i] as f32 * activation_scale * cached.scales[i];
    }
    for i in 0..key.second.n {
        let j = key.first.n + i;
        output_second_f32[i] = result.values[j] as f32 * activation_scale * cached.scales[j];
    }
    let rescale_ns = rescale_started.map_or(0, |started| started.elapsed().as_nanos());
    let elapsed_total_ns = call_start.elapsed().as_nanos();
    let elapsed_ns = u64::try_from(elapsed_total_ns).unwrap_or(u64::MAX);
    if cache_hit {
        *decode_cache_hit_ns = decode_cache_hit_ns.saturating_add(elapsed_ns);
        record_m1_profile(
            m1_profile,
            M1ProfileKind::Pair,
            key.first.k,
            total_n,
            cached.choice,
            &result.stats,
            quant_ns,
            rescale_ns,
            elapsed_total_ns,
        );
    } else {
        *decode_cache_miss_ns = decode_cache_miss_ns.saturating_add(elapsed_ns);
    }
    STATUS_OK
}

fn execute_cached_w8a8_triple_m1<F>(
    context: &mut RockNpuContext,
    key: DecodeTripleKey,
    activations_k_f32: &[f32],
    output_first_f32: &mut [f32],
    output_second_f32: &mut [f32],
    output_third_f32: &mut [f32],
    prepare: F,
) -> i32
where
    F: FnOnce() -> Option<(Vec<i8>, Vec<f32>)>,
{
    let call_start = Instant::now();
    if key.first.k != key.second.k
        || key.first.k != key.third.k
        || activations_k_f32.len() != key.first.k
        || output_first_f32.len() != key.first.n
        || output_second_f32.len() != key.second.n
        || output_third_f32.len() != key.third.n
        || !key.first.k.is_multiple_of(512)
        || !key.first.n.is_multiple_of(32)
        || !key.second.n.is_multiple_of(32)
        || !key.third.n.is_multiple_of(32)
    {
        return STATUS_INVALID_ARGUMENT;
    }
    let Some(total_n) = key
        .first
        .n
        .checked_add(key.second.n)
        .and_then(|n| n.checked_add(key.third.n))
    else {
        return STATUS_INVALID_ARGUMENT;
    };
    if total_n > 16384 {
        return STATUS_INVALID_ARGUMENT;
    }
    let quant_started = context.m1_profile.as_ref().map(|_| Instant::now());
    let Some((activations_i8, activation_scale)) = quantize_symmetric(activations_k_f32) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let quant_ns = quant_started.map_or(0, |started| started.elapsed().as_nanos());
    let activation: Arc<[i8]> = Arc::from(activations_i8);

    let RockNpuContext {
        device: _,
        decode_pool,
        decode_worker_cache,
        decode_weights,
        decode_pair_weights,
        decode_triple_weights,
        decode_cache_hits,
        decode_cache_misses,
        decode_cache_hit_ns,
        decode_cache_miss_ns,
        decode_worker_calls,
        decode_ksplit_calls,
        m1_profile,
        overlap,
        ..
    } = context;

    let (cached, cache_hit) = match decode_triple_weights.entry(key) {
        Entry::Occupied(entry) => {
            *decode_cache_hits = decode_cache_hits.saturating_add(1);
            (entry.into_mut(), true)
        }
        Entry::Vacant(entry) => {
            let Some((weights_i8, scales)) = prepare() else {
                return STATUS_INVALID_ARGUMENT;
            };
            if weights_i8.len() != total_n.saturating_mul(key.first.k) || scales.len() != total_n {
                return STATUS_INVALID_ARGUMENT;
            }
            let weights: Arc<[i8]> = Arc::from(weights_i8);
            let shape = (key.first.k, total_n);
            let (choice, prepared) = if let Some(&choice) = decode_worker_cache.get(&shape) {
                let prepared = match decode_pool.prepare_weights_m1_with_split(
                    Arc::clone(&weights),
                    key.first.k,
                    total_n,
                    choice.workers,
                    choice.split,
                ) {
                    Ok(prepared) => prepared,
                    Err(_) => return STATUS_EXECUTION_ERROR,
                };
                (choice, prepared)
            } else {
                let (choice, prepared) = match tune_decode_workers(
                    decode_pool,
                    Arc::clone(&weights),
                    Arc::clone(&activation),
                    key.first.k,
                    total_n,
                ) {
                    Ok(result) => result,
                    Err(()) => return STATUS_EXECUTION_ERROR,
                };
                decode_worker_cache.insert(shape, choice);
                (choice, prepared)
            };
            *decode_cache_misses = decode_cache_misses.saturating_add(1);
            (
                entry.insert(CachedW8A8Weight {
                    prepared,
                    scales,
                    choice,
                }),
                false,
            )
        }
    };

    let mut overlap_fn = take_overlap(overlap);
    let result = match decode_pool.execute_prepared_overlap(
        Arc::clone(&activation),
        &cached.prepared,
        overlap_fn.as_mut().map(|f| f as &mut dyn FnMut()),
    ) {
        Ok(result) => result,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    if let Some(calls) = decode_worker_calls.get_mut(cached.choice.workers.saturating_sub(1)) {
        *calls = calls.saturating_add(1);
    }
    if cached.choice.split == Int8DecodeSplit::K {
        *decode_ksplit_calls = decode_ksplit_calls.saturating_add(1);
    }
    let rescale_started = m1_profile.as_ref().map(|_| Instant::now());
    for i in 0..key.first.n {
        output_first_f32[i] = result.values[i] as f32 * activation_scale * cached.scales[i];
    }
    for i in 0..key.second.n {
        let j = key.first.n + i;
        output_second_f32[i] = result.values[j] as f32 * activation_scale * cached.scales[j];
    }
    for i in 0..key.third.n {
        let j = key.first.n + key.second.n + i;
        output_third_f32[i] = result.values[j] as f32 * activation_scale * cached.scales[j];
    }
    let rescale_ns = rescale_started.map_or(0, |started| started.elapsed().as_nanos());

    // Once the combined Q/V/K resident entry has executed successfully, the
    // old standalone-Q and V/K-pair prepared weights are redundant. Removing
    // them keeps the production steady-state resident footprint near the
    // pre-triple baseline. If QKV is later disabled at runtime, the existing
    // lazy cache path will rebuild those fallback entries on demand.
    decode_weights.remove(&key.first);
    decode_pair_weights.remove(&DecodePairKey {
        first: key.second,
        second: key.third,
    });

    let elapsed_total_ns = call_start.elapsed().as_nanos();
    let elapsed_ns = u64::try_from(elapsed_total_ns).unwrap_or(u64::MAX);
    if cache_hit {
        *decode_cache_hit_ns = decode_cache_hit_ns.saturating_add(elapsed_ns);
        record_m1_profile(
            m1_profile,
            M1ProfileKind::Triple,
            key.first.k,
            total_n,
            cached.choice,
            &result.stats,
            quant_ns,
            rescale_ns,
            elapsed_total_ns,
        );
    } else {
        *decode_cache_miss_ns = decode_cache_miss_ns.saturating_add(elapsed_ns);
    }
    STATUS_OK
}

// Cache one exact-M packed layout per immutable GGML weight. A new batch size
// replaces the previous layout so changing prompt lengths cannot multiply the
// model's resident footprint. Handles belong to the context-owned worker pool.
fn execute_cached_prefill<F>(
    context: &mut RockNpuContext,
    key: DecodeWeightKey,
    m: usize,
    activations: &[f32],
    output: &mut [f32],
    decode: F,
) -> i32
where
    F: FnOnce() -> Vec<f16>,
{
    if context.prefill_pool.is_none() {
        let started = context.prefill_profile.as_ref().map(|_| Instant::now());
        context.prefill_pool = match Fp16MatmulPool::new(3) {
            Ok(pool) => Some(pool),
            Err(_) => return STATUS_EXECUTION_ERROR,
        };
        if let (Some(profile), Some(started)) = (context.prefill_profile.as_mut(), started) {
            profile.pool_create_ns += started.elapsed().as_nanos();
        }
    }
    if context
        .prefill_weights
        .get(&key)
        .is_some_and(|(old_m, _)| *old_m != m)
    {
        let (_, old) = context.prefill_weights.remove(&key).unwrap();
        if context
            .prefill_pool
            .as_mut()
            .unwrap()
            .release_prepared(&old)
            .is_err()
        {
            return STATUS_EXECUTION_ERROR;
        }
    }
    if !context.prefill_weights.contains_key(&key) {
        if let Some(profile) = context.prefill_profile.as_mut() {
            profile.cache_misses += 1;
        }
        let started = context.prefill_profile.as_ref().map(|_| Instant::now());
        let weights = decode();
        if let (Some(profile), Some(started)) = (context.prefill_profile.as_mut(), started) {
            profile.dequant_ns += started.elapsed().as_nanos();
        }
        if weights.len() != key.k.saturating_mul(key.n) {
            return STATUS_INVALID_ARGUMENT;
        }
        let started = context.prefill_profile.as_ref().map(|_| Instant::now());
        let weights = Arc::new(weights);
        if let (Some(profile), Some(started)) = (context.prefill_profile.as_mut(), started) {
            profile.host_to_arc_ns += started.elapsed().as_nanos();
        }
        let started = context.prefill_profile.as_ref().map(|_| Instant::now());
        let prepared = match context
            .prefill_pool
            .as_mut()
            .unwrap()
            .prepare_weights_f32_vec(weights, m, key.k, key.n)
        {
            Ok(prepared) => prepared,
            Err(_) => return STATUS_EXECUTION_ERROR,
        };
        if let Some(profile) = context.prefill_profile.as_mut() {
            if let Some(started) = started {
                profile.prepare_call_ns += started.elapsed().as_nanos();
            }
            let stats = prepared.stats();
            profile.resident_bytes = profile.resident_bytes.saturating_add(stats.resident_bytes);
            profile.prepare_pool_wall_ns += stats.prepare_wall_ns;
            profile.prepare_worker_critical_ns += stats.worker_total_ns_max;
            profile.plan_critical_ns += stats.plan_ns_max;
            profile.layout_critical_ns += stats.layout_ns_max;
            profile.alloc_mmap_critical_ns += stats.alloc_mmap_ns_max;
            profile.prep_critical_ns += stats.prep_ns_max;
            profile.zero_critical_ns += stats.zero_ns_max;
            profile.tile_pack_critical_ns += stats.tile_pack_ns_max;
            profile.fini_critical_ns += stats.fini_ns_max;
            profile.pack_critical_ns += stats.pack_ns_max;
            profile.pack_worker_sum_ns += stats.pack_ns_sum;
        }
        context.prefill_weights.insert(key, (m, prepared));
    } else if let Some(profile) = context.prefill_profile.as_mut() {
        profile.cache_hits += 1;
    }
    let started = context.prefill_profile.as_ref().map(|_| Instant::now());
    let a: Arc<[f16]> = activations.iter().copied().map(f16::from_f32).collect();
    if let (Some(profile), Some(started)) = (context.prefill_profile.as_mut(), started) {
        profile.activation_convert_ns += started.elapsed().as_nanos();
    }
    let started = context.prefill_profile.as_ref().map(|_| Instant::now());
    let result = match context.prefill_pool.as_mut().unwrap().execute_prepared_f32(
        a,
        m,
        &context.prefill_weights[&key].1,
    ) {
        Ok(result) => result,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    if let (Some(profile), Some(started)) = (context.prefill_profile.as_mut(), started) {
        profile.execute_ns += started.elapsed().as_nanos();
    }
    let started = context.prefill_profile.as_ref().map(|_| Instant::now());
    output.copy_from_slice(&result.values);
    if let (Some(profile), Some(started)) = (context.prefill_profile.as_mut(), started) {
        profile.output_copy_ns += started.elapsed().as_nanos();
    }
    STATUS_OK
}

/// Return the number of RockNPU devices currently usable by the runtime.
///
/// The initial RK3588 backend exposes one default Rocket device at
/// `/dev/accel/accel0`. The C ABI intentionally reports availability rather
/// than exposing Rocket file descriptors or Linux-specific details.
#[unsafe(no_mangle)]
pub extern "C" fn rocknpu_device_count() -> usize {
    usize::from(RocketDevice::open().is_ok())
}

#[unsafe(no_mangle)]
pub extern "C" fn rocknpu_context_create() -> *mut RockNpuContext {
    let w4a4_enabled = env_enabled("ROCKNPU_W4A4");
    let w4a4_pool = if w4a4_enabled {
        Int4DecodePool::new(3).ok()
    } else {
        None
    };
    match (RocketDevice::open(), Int8DecodePool::new(3)) {
        (Ok(device), Ok(decode_pool)) => Box::into_raw(Box::new(RockNpuContext {
            device,
            prefill_pool: None,
            prefill_weights: HashMap::new(),
            prefill_profile: env_enabled("ROCKNPU_PREFILL_PROFILE").then(PrefillProfile::default),
            mtile_profile: env_enabled("ROCKNPU_MTILE_PROFILE").then(MtileProfile::default),
            m1_profile: env_enabled("ROCKNPU_M1_PROFILE").then(M1Profile::default),
            decode_pool,
            decode_worker_cache: HashMap::new(),
            decode_weights: HashMap::new(),
            decode_mtile_weights: HashMap::new(),
            decode_mtile_pool_weights: HashMap::new(),
            decode_mtile_pair_pool_weights: HashMap::new(),
            decode_grouped_mtile_weights: HashMap::new(),
            decode_grouped_mtile_pool_weights: HashMap::new(),
            decode_mtile_scratch: HashMap::new(),
            decode_pair_weights: HashMap::new(),
            decode_triple_weights: HashMap::new(),
            decode_w4a4_weights: HashMap::new(),
            decode_grouped_w4a4_weights: HashMap::new(),
            decode_pool_w4a4_weights: HashMap::new(),
            w4a4_pool,
            w4a4_worker_cache: HashMap::new(),
            w4a4_enabled,
            w4a4_calls: 0,
            w4a4_saturated_calls: 0,
            w4a4_saturated_outputs: 0,
            decode_cache_hits: 0,
            decode_cache_misses: 0,
            decode_cache_hit_ns: 0,
            decode_cache_miss_ns: 0,
            decode_worker_calls: [0; 3],
            decode_ksplit_calls: 0,
            mtile_i32: Vec::new(),
            decode_mtile_concat_weights: HashMap::new(),
            overlap: None,
        })),
        _ => ptr::null_mut(),
    }
}

/// Read lazy W8A8 decode-cache statistics for diagnostics.
///
/// # Safety
/// `context` must be a live RockNPU context and `out` must point to writable
/// storage for one [`RockNpuDecodeCacheStats`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocknpu_context_decode_cache_stats(
    context: *const RockNpuContext,
    out: *mut RockNpuDecodeCacheStats,
) -> i32 {
    if context.is_null() || out.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: both pointers were checked above and are required by the C ABI to
    // remain valid for this synchronous call.
    let context = unsafe { &*context };
    let w8_resident_bytes = context
        .decode_weights
        .values()
        .chain(context.decode_pair_weights.values())
        .chain(context.decode_triple_weights.values())
        .fold(0usize, |acc, cached| {
            acc.saturating_add(cached.prepared.stats().resident_bytes)
        });
    let w4_resident_bytes = context
        .decode_w4a4_weights
        .values()
        .fold(w8_resident_bytes, |acc, cached| {
            acc.saturating_add(cached.prepared.stats().resident_bytes)
        });
    let grouped_w4_resident_bytes = context
        .decode_grouped_w4a4_weights
        .values()
        .fold(w4_resident_bytes, |acc, cached| {
            acc.saturating_add(cached.prepared.stats().resident_bytes)
        });
    let resident_bytes = context
        .decode_pool_w4a4_weights
        .values()
        .fold(grouped_w4_resident_bytes, |acc, cached| {
            acc.saturating_add(cached.prepared.stats().resident_bytes)
        });
    // SAFETY: out is non-null and caller provides writable storage for one stats value.
    unsafe {
        *out = RockNpuDecodeCacheStats {
            hits: context.decode_cache_hits,
            misses: context.decode_cache_misses,
            entries: context
                .decode_weights
                .len()
                .saturating_add(context.decode_pair_weights.len())
                .saturating_add(context.decode_triple_weights.len())
                .saturating_add(context.decode_w4a4_weights.len())
                .saturating_add(context.decode_grouped_w4a4_weights.len())
                .saturating_add(context.decode_pool_w4a4_weights.len()),
            resident_bytes,
            hit_ns: context.decode_cache_hit_ns,
            miss_ns: context.decode_cache_miss_ns,
            tuned_shapes: context
                .decode_worker_cache
                .len()
                .saturating_add(context.w4a4_worker_cache.len()),
            worker1_calls: context.decode_worker_calls[0],
            worker2_calls: context.decode_worker_calls[1],
            worker3_calls: context.decode_worker_calls[2],
            ksplit_calls: context.decode_ksplit_calls,
        };
    }
    STATUS_OK
}

/// Register a one-shot host callback for the next M=1 W8 projection
/// (single, pair or triple). The NPU path invokes it after submitting the
/// work and before waiting for completion, so host work overlaps the NPU.
/// It runs at most once; a null callback clears it. Callers must check
/// whether it ran and run it themselves otherwise (e.g. on errors).
///
/// # Safety
/// `context` must be a live context. The callback and `user_data` must stay
/// valid until the next NPU projection call returns or the callback is cleared.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocknpu_context_set_overlap(
    context: *mut RockNpuContext,
    callback: Option<OverlapFn>,
    user_data: *mut core::ffi::c_void,
) -> i32 {
    if context.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }
    let context = unsafe { &mut *context };
    context.overlap = callback.map(|callback| (callback, user_data as usize));
    STATUS_OK
}

/// Destroy a context returned by [`rocknpu_context_create`].
///
/// # Safety
///
/// `context` must be null or a live pointer returned by `rocknpu_context_create`
/// that has not already been destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocknpu_context_destroy(context: *mut RockNpuContext) {
    if context.is_null() {
        return;
    }
    // SAFETY: the C API requires a context returned by rocknpu_context_create,
    // and ownership is transferred back exactly once at destroy time.
    unsafe {
        let context = Box::from_raw(context);
        if env_enabled("ROCKNPU_W4A4_TRACE") && context.w4a4_calls != 0 {
            eprintln!(
                "ROCKNPU W4A4 summary calls={} cache_entries={} grouped_cache_entries={} pool_cache_entries={} tuned_shapes={} saturated_calls={} saturated_outputs={}",
                context.w4a4_calls,
                context.decode_w4a4_weights.len(),
                context.decode_grouped_w4a4_weights.len(),
                context.decode_pool_w4a4_weights.len(),
                context.w4a4_worker_cache.len(),
                context.w4a4_saturated_calls,
                context.w4a4_saturated_outputs
            );
        }
        if let Some(profile) = &context.m1_profile {
            let mut shapes: Vec<_> = profile.shapes.iter().collect();
            shapes.sort_by_key(|((kind, k, n), _)| (*kind, *k, *n));
            for ((kind, k, n), shape) in shapes {
                let accounted_ns = shape
                    .quant_ns
                    .saturating_add(shape.execute_wall_ns)
                    .saturating_add(shape.rescale_ns);
                let other_ns = shape.total_ns.saturating_sub(accounted_ns);
                eprintln!(
                    "ROCKNPU M1 PROFILE kind={} k={} n={} calls={} npu_tasks={} workers=[{},{},{}] ksplit_calls={} total_ms={:.3} avg_call_ms={:.3} quant_ms={:.3} execute_wall_ms={:.3} rescale_ms={:.3} other_ms={:.3} worker_alloc_ms={:.3} worker_input_ms={:.3} worker_partial_ms={:.3} worker_regcmd_ms={:.3} worker_output_fini_ms={:.3} worker_submit_ms={:.3} worker_wait_ms={:.3} worker_host_accum_ms={:.3}",
                    kind.as_str(),
                    k,
                    n,
                    shape.calls,
                    shape.npu_tasks,
                    shape.worker_calls[0],
                    shape.worker_calls[1],
                    shape.worker_calls[2],
                    shape.ksplit_calls,
                    ns_to_ms(shape.total_ns),
                    if shape.calls == 0 {
                        0.0
                    } else {
                        ns_to_ms(shape.total_ns) / shape.calls as f64
                    },
                    ns_to_ms(shape.quant_ns),
                    ns_to_ms(shape.execute_wall_ns),
                    ns_to_ms(shape.rescale_ns),
                    ns_to_ms(other_ns),
                    ns_to_ms(shape.alloc_ns),
                    ns_to_ms(shape.input_stage_ns),
                    ns_to_ms(shape.partial_stage_ns),
                    ns_to_ms(shape.regcmd_stage_ns),
                    ns_to_ms(shape.output_fini_ns),
                    ns_to_ms(shape.submit_ns),
                    ns_to_ms(shape.wait_ns),
                    ns_to_ms(shape.host_accum_ns),
                );
            }
        }
        if let Some(profile) = &context.mtile_profile {
            eprintln!(
                "ROCKNPU MTILE PROFILE calls={} ksplit_calls={} cache_hits={} cache_misses={} weight_prepare_ms={:.3} weight_pack_ms={:.3} quant_ms={:.3} alloc_ms={:.3} input_stage_ms={:.3} regcmd_ms={:.3} submit_ms={:.3} wait_ms={:.3} host_accum_ms={:.3} execute_total_ms={:.3} rescale_ms={:.3}",
                profile.calls,
                profile.ksplit_calls,
                profile.cache_hits,
                profile.cache_misses,
                ns_to_ms(profile.weight_prepare_ns),
                ns_to_ms(profile.weight_pack_ns),
                ns_to_ms(profile.quant_ns),
                ns_to_ms(profile.alloc_ns),
                ns_to_ms(profile.input_stage_ns),
                ns_to_ms(profile.regcmd_stage_ns),
                ns_to_ms(profile.submit_ns),
                ns_to_ms(profile.wait_ns),
                ns_to_ms(profile.host_accum_ns),
                ns_to_ms(profile.execute_total_ns),
                ns_to_ms(profile.rescale_ns),
            );
        }
        if let Some(profile) = &context.prefill_profile {
            eprintln!(
                "ROCKNPU PREFILL PROFILE hits={} misses={} resident_mb={:.2} pool_create_ms={:.3} dequant_ms={:.3} host_to_arc_ms={:.3} prepare_call_ms={:.3} prepare_pool_wall_ms={:.3} prepare_call_overhead_ms={:.3} prepare_worker_critical_ms={:.3} prepare_pool_residual_ms={:.3} plan_critical_ms={:.3} layout_critical_ms={:.3} alloc_mmap_critical_ms={:.3} prep_critical_ms={:.3} zero_critical_ms={:.3} tile_pack_critical_ms={:.3} fini_critical_ms={:.3} pack_critical_ms={:.3} pack_worker_sum_ms={:.3} activation_convert_ms={:.3} execute_ms={:.3} output_copy_ms={:.3}",
                profile.cache_hits,
                profile.cache_misses,
                profile.resident_bytes as f64 / (1024.0 * 1024.0),
                ns_to_ms(profile.pool_create_ns),
                ns_to_ms(profile.dequant_ns),
                ns_to_ms(profile.host_to_arc_ns),
                ns_to_ms(profile.prepare_call_ns),
                ns_to_ms(profile.prepare_pool_wall_ns),
                ns_to_ms(
                    profile
                        .prepare_call_ns
                        .saturating_sub(profile.prepare_pool_wall_ns)
                ),
                ns_to_ms(profile.prepare_worker_critical_ns),
                ns_to_ms(
                    profile
                        .prepare_pool_wall_ns
                        .saturating_sub(profile.prepare_worker_critical_ns)
                ),
                ns_to_ms(profile.plan_critical_ns),
                ns_to_ms(profile.layout_critical_ns),
                ns_to_ms(profile.alloc_mmap_critical_ns),
                ns_to_ms(profile.prep_critical_ns),
                ns_to_ms(profile.zero_critical_ns),
                ns_to_ms(profile.tile_pack_critical_ns),
                ns_to_ms(profile.fini_critical_ns),
                ns_to_ms(profile.pack_critical_ns),
                ns_to_ms(profile.pack_worker_sum_ns),
                ns_to_ms(profile.activation_convert_ns),
                ns_to_ms(profile.execute_ns),
                ns_to_ms(profile.output_copy_ns),
            );
        }
        drop(context);
    }
}

fn prewarm_quantized_m16_with<F>(
    context: &mut RockNpuContext,
    key: DecodeWeightKey,
    prepare: F,
) -> i32
where
    F: FnOnce() -> Option<(Vec<i8>, Vec<f32>)>,
{
    let supported_shape =
        (key.k == 2048 && matches!(key.n, 256 | 2048 | 5632)) || (key.k == 5632 && key.n == 2048);
    if !supported_shape || !mtile_shape_enabled(key.k, key.n) {
        return STATUS_OK;
    }

    if env_enabled("ROCKNPU_MTILE_MC") && key.n >= 2048 {
        if context.decode_mtile_pool_weights.contains_key(&key) {
            return STATUS_OK;
        }
        let Some((weights_i8, scales)) = prepare() else {
            return STATUS_INVALID_ARGUMENT;
        };
        if weights_i8.len() != key.n.saturating_mul(key.k) || scales.len() != key.n {
            return STATUS_INVALID_ARGUMENT;
        }
        let split = if key.k > 4096 {
            Int8DecodeSplit::K
        } else {
            Int8DecodeSplit::N
        };
        let prepared = match context.decode_pool.prepare_weights_mtile_with_split(
            Arc::<[i8]>::from(weights_i8),
            key.k,
            key.n,
            3,
            split,
        ) {
            Ok(prepared) => prepared,
            Err(_) => return STATUS_EXECUTION_ERROR,
        };
        context
            .decode_mtile_pool_weights
            .insert(key, CachedW8MtilePoolWeight { prepared, scales });
        return STATUS_OK;
    }

    if context.decode_mtile_weights.contains_key(&key) {
        return STATUS_OK;
    }
    let Some((weights_i8, scales)) = prepare() else {
        return STATUS_INVALID_ARGUMENT;
    };
    if weights_i8.len() != key.n.saturating_mul(key.k) || scales.len() != key.n {
        return STATUS_INVALID_ARGUMENT;
    }
    let executor = Int8DecodeExecutor::from_externally_guarded_device(&context.device);
    let prepared = match executor.prepare_weights(&weights_i8, key.k, key.n) {
        Ok(prepared) => prepared,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    context
        .decode_mtile_weights
        .insert(key, CachedW8MtileWeight { prepared, scales });
    STATUS_OK
}

/// Precompute resident W8/M16 decode weights for router-kept NPU shapes.
///
/// The operation only performs host conversion and resident weight packing;
/// it submits no NPU work. Repeated calls reuse the normal decode cache key.
///
/// # Safety
/// The context must be live and weights must reference exactly weights_bytes bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocknpu_prewarm_quantized_m16(
    context: *mut RockNpuContext,
    weights: *const u8,
    weights_bytes: usize,
    quant_kind: u32,
    k: usize,
    n: usize,
) -> i32 {
    if context.is_null() || weights.is_null() || k == 0 || n == 0 {
        return STATUS_INVALID_ARGUMENT;
    }
    let (kind, block_bytes) = match quant_kind {
        4 => (DecodeWeightKind::Q4K, size_of::<BlockQ4K>()),
        6 => (DecodeWeightKind::Q6K, size_of::<BlockQ6K>()),
        _ => return STATUS_INVALID_ARGUMENT,
    };
    if !k.is_multiple_of(256) {
        return STATUS_INVALID_ARGUMENT;
    }
    let Some(expected_bytes) = n
        .checked_mul(k / 256)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
    else {
        return STATUS_INVALID_ARGUMENT;
    };
    if weights_bytes != expected_bytes {
        return STATUS_INVALID_ARGUMENT;
    }

    // SAFETY: pointer/null/byte-count contracts are validated above.
    let (weight_bytes, context) =
        unsafe { (slice::from_raw_parts(weights, weights_bytes), &mut *context) };
    let key = DecodeWeightKey {
        address: weight_bytes.as_ptr() as usize,
        bytes: weight_bytes.len(),
        k,
        n,
        kind,
    };
    match kind {
        DecodeWeightKind::Q4K => {
            prewarm_quantized_m16_with(context, key, || prepare_q4_k_w8a8(weight_bytes, k, n))
        }
        DecodeWeightKind::Q6K => {
            prewarm_quantized_m16_with(context, key, || prepare_q6_k_w8a8(weight_bytes, k, n))
        }
    }
}

/// Execute C[M,N] = A[M,K] x B[N,K]^T on RockNPU.
///
/// `weights_nk_f16_bits` contains IEEE-754 binary16 bit patterns in row-major
/// [N,K] order. `activations_mk_f32` and `output_mn_f32` are row-major.
///
/// # Safety
///
/// `context` must point to a live RockNPU context. The three tensor pointers
/// must reference readable/writable storage for at least `N*K`, `M*K`, and
/// `M*N` elements respectively for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocknpu_matmul_f16_f32_f32(
    context: *mut RockNpuContext,
    weights_nk_f16_bits: *const u16,
    activations_mk_f32: *const f32,
    output_mn_f32: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) -> i32 {
    if context.is_null()
        || weights_nk_f16_bits.is_null()
        || activations_mk_f32.is_null()
        || output_mn_f32.is_null()
        || m == 0
        || k == 0
        || n == 0
    {
        return STATUS_INVALID_ARGUMENT;
    }

    let Some(a_len) = m.checked_mul(k) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(b_len) = n.checked_mul(k) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(out_len) = m.checked_mul(n) else {
        return STATUS_INVALID_ARGUMENT;
    };

    // SAFETY: non-null pointers and element counts are validated above. The C
    // caller owns these buffers and must provide at least the documented sizes.
    let (weights_bits, activations, output, context) = unsafe {
        (
            slice::from_raw_parts(weights_nk_f16_bits, b_len),
            slice::from_raw_parts(activations_mk_f32, a_len),
            slice::from_raw_parts_mut(output_mn_f32, out_len),
            &mut *context,
        )
    };

    let weights = weights_bits
        .iter()
        .copied()
        .map(f16::from_bits)
        .collect::<Vec<_>>();
    let activations = activations
        .iter()
        .copied()
        .map(f16::from_f32)
        .collect::<Vec<_>>();

    let mut executor = match Fp16MatmulExecutor::new(&context.device) {
        Ok(executor) => executor,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    let result = match executor.execute_f32(&activations, &weights, m, k, n) {
        Ok(result) => result,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    output.copy_from_slice(&result.values);
    STATUS_OK
}

/// Execute two M=1 quantized projections that share one activation by
/// concatenating their W8 rows along N and issuing one prepared NPU matmul.
/// `kind` is 4 for Q4_K and 6 for Q6_K. Outputs preserve the same per-row
/// W8 dequantization semantics as independent calls.
///
/// # Safety
/// All pointers must satisfy their documented byte/element lengths for the
/// duration of this synchronous call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocknpu_matmul_w8a8_f32_f32_m1(
    context: *mut RockNpuContext,
    weights_nk_i8: *const i8,
    weight_scales_n_f32: *const f32,
    activations_k_f32: *const f32,
    output_n_f32: *mut f32,
    k: usize,
    n: usize,
) -> i32 {
    if context.is_null()
        || weights_nk_i8.is_null()
        || weight_scales_n_f32.is_null()
        || activations_k_f32.is_null()
        || output_n_f32.is_null()
        || k == 0
        || n == 0
    {
        return STATUS_INVALID_ARGUMENT;
    }
    let Some(weight_len) = k.checked_mul(n) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let (weights, scales, activations, output, context) = unsafe {
        (
            slice::from_raw_parts(weights_nk_i8, weight_len),
            slice::from_raw_parts(weight_scales_n_f32, n),
            slice::from_raw_parts(activations_k_f32, k),
            slice::from_raw_parts_mut(output_n_f32, n),
            &mut *context,
        )
    };
    if scales.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return STATUS_INVALID_ARGUMENT;
    }
    let key = DecodeWeightKey {
        address: weights.as_ptr() as usize,
        bytes: weight_len,
        k,
        n,
        kind: DecodeWeightKind::Q4K,
    };
    execute_cached_w8a8_m1(context, key, activations, output, || {
        Some((weights.to_vec(), scales.to_vec()))
    })
}

/// Execute a wide M=1 W8A8 projection as bounded N chunks.
///
/// The resident cache and Rocket command path remain unchanged for each
/// chunk; only the adapter-facing entry point is split. This is used for
/// vocab/output heads whose N exceeds the native M=1 N limit.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocknpu_matmul_w8a8_f32_f32_m1_nsplit(
    context: *mut RockNpuContext,
    weights_nk_i8: *const i8,
    weight_scales_n_f32: *const f32,
    activations_k_f32: *const f32,
    output_n_f32: *mut f32,
    k: usize,
    n: usize,
    chunk_n: usize,
) -> i32 {
    if context.is_null()
        || weights_nk_i8.is_null()
        || weight_scales_n_f32.is_null()
        || activations_k_f32.is_null()
        || output_n_f32.is_null()
        || k == 0
        || n == 0
        || chunk_n == 0
        || !chunk_n.is_multiple_of(32)
        || !n.is_multiple_of(32)
    {
        return STATUS_INVALID_ARGUMENT;
    }

    let mut offset = 0usize;
    while offset < n {
        let current = chunk_n.min(n - offset);
        let status = unsafe {
            rocknpu_matmul_w8a8_f32_f32_m1(
                context,
                weights_nk_i8.add(offset * k),
                weight_scales_n_f32.add(offset),
                activations_k_f32,
                output_n_f32.add(offset),
                k,
                current,
            )
        };
        if status != STATUS_OK {
            return status;
        }
        offset += current;
    }
    STATUS_OK
}

/// Execute a quality-equivalent M=1 W8A8 FFN sequence through one adapter
/// boundary: gate/up projections, F32 SwiGLU, and the down projection.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocknpu_matmul_w8a8_swiglu_down_f32(
    context: *mut RockNpuContext,
    gate_weights_nk_i8: *const i8,
    gate_scales_n_f32: *const f32,
    up_weights_nk_i8: *const i8,
    up_scales_n_f32: *const f32,
    down_weights_nk_i8: *const i8,
    down_scales_n_f32: *const f32,
    activations_k_f32: *const f32,
    output_n_f32: *mut f32,
    projection_k: usize,
    ffn_n: usize,
    output_n: usize,
) -> i32 {
    if context.is_null()
        || gate_weights_nk_i8.is_null()
        || gate_scales_n_f32.is_null()
        || up_weights_nk_i8.is_null()
        || up_scales_n_f32.is_null()
        || down_weights_nk_i8.is_null()
        || down_scales_n_f32.is_null()
        || activations_k_f32.is_null()
        || output_n_f32.is_null()
        || projection_k == 0
        || ffn_n == 0
        || output_n == 0
    {
        return STATUS_INVALID_ARGUMENT;
    }
    let Some(gate_len) = projection_k.checked_mul(ffn_n) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(down_len) = ffn_n.checked_mul(output_n) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let (gate_weights, gate_scales, up_weights, up_scales, down_weights, down_scales, activation, output) = unsafe {
        (
            slice::from_raw_parts(gate_weights_nk_i8, gate_len),
            slice::from_raw_parts(gate_scales_n_f32, ffn_n),
            slice::from_raw_parts(up_weights_nk_i8, gate_len),
            slice::from_raw_parts(up_scales_n_f32, ffn_n),
            slice::from_raw_parts(down_weights_nk_i8, down_len),
            slice::from_raw_parts(down_scales_n_f32, output_n),
            slice::from_raw_parts(activations_k_f32, projection_k),
            slice::from_raw_parts_mut(output_n_f32, output_n),
        )
    };
    let mut gate = vec![0.0f32; ffn_n];
    let mut up = vec![0.0f32; ffn_n];
    let status = unsafe {
        rocknpu_matmul_w8a8_pair_f32_f32_m1(
            context,
            gate_weights.as_ptr(),
            gate_scales.as_ptr(),
            ffn_n,
            up_weights.as_ptr(),
            up_scales.as_ptr(),
            ffn_n,
            activation.as_ptr(),
            gate.as_mut_ptr(),
            up.as_mut_ptr(),
            projection_k,
        )
    };
    if status != STATUS_OK {
        return status;
    }
    for index in 0..ffn_n {
        let gate_value = gate[index];
        let up_value = up[index];
        let sigmoid = 1.0 / (1.0 + (-gate_value).exp());
        gate[index] = gate_value * sigmoid * up_value;
    }
    unsafe {
        rocknpu_matmul_w8a8_f32_f32_m1(
            context,
            down_weights.as_ptr(),
            down_scales.as_ptr(),
            gate.as_ptr(),
            output.as_mut_ptr(),
            ffn_n,
            output_n,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocknpu_matmul_w8a8_f32_f32_m16(
    context: *mut RockNpuContext,
    weights_nk_i8: *const i8,
    weight_scales_n_f32: *const f32,
    activations_mk_f32: *const f32,
    output_mn_f32: *mut f32,
    k: usize,
    n: usize,
) -> i32 {
    const M: usize = 16;
    if context.is_null()
        || weights_nk_i8.is_null()
        || weight_scales_n_f32.is_null()
        || activations_mk_f32.is_null()
        || output_mn_f32.is_null()
        || k == 0
        || n == 0
    {
        return STATUS_INVALID_ARGUMENT;
    }
    let Some(weight_len) = k.checked_mul(n) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(activation_len) = M.checked_mul(k) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(output_len) = M.checked_mul(n) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let (weights, scales, activations, output, context) = unsafe {
        (
            slice::from_raw_parts(weights_nk_i8, weight_len),
            slice::from_raw_parts(weight_scales_n_f32, n),
            slice::from_raw_parts(activations_mk_f32, activation_len),
            slice::from_raw_parts_mut(output_mn_f32, output_len),
            &mut *context,
        )
    };
    if scales.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return STATUS_INVALID_ARGUMENT;
    }
    let key = DecodeWeightKey {
        address: weights.as_ptr() as usize,
        bytes: weight_len,
        k,
        n,
        kind: DecodeWeightKind::Q4K,
    };
    execute_cached_w8a8_m16(context, key, activations, output, || {
        Some((weights.to_vec(), scales.to_vec()))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocknpu_matmul_w8a8_pair_f32_f32_m1(
    context: *mut RockNpuContext,
    first_weights_nk_i8: *const i8,
    first_scales_n_f32: *const f32,
    first_n: usize,
    second_weights_nk_i8: *const i8,
    second_scales_n_f32: *const f32,
    second_n: usize,
    activations_k_f32: *const f32,
    first_output_f32: *mut f32,
    second_output_f32: *mut f32,
    k: usize,
) -> i32 {
    if context.is_null()
        || first_weights_nk_i8.is_null()
        || first_scales_n_f32.is_null()
        || second_weights_nk_i8.is_null()
        || second_scales_n_f32.is_null()
        || activations_k_f32.is_null()
        || first_output_f32.is_null()
        || second_output_f32.is_null()
        || k == 0
        || first_n == 0
        || second_n == 0
    {
        return STATUS_INVALID_ARGUMENT;
    }
    let Some(first_len) = k.checked_mul(first_n) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(second_len) = k.checked_mul(second_n) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let (first_w, first_s, second_w, second_s, a, out1, out2, context) = unsafe {
        (
            slice::from_raw_parts(first_weights_nk_i8, first_len),
            slice::from_raw_parts(first_scales_n_f32, first_n),
            slice::from_raw_parts(second_weights_nk_i8, second_len),
            slice::from_raw_parts(second_scales_n_f32, second_n),
            slice::from_raw_parts(activations_k_f32, k),
            slice::from_raw_parts_mut(first_output_f32, first_n),
            slice::from_raw_parts_mut(second_output_f32, second_n),
            &mut *context,
        )
    };
    if first_s
        .iter()
        .chain(second_s)
        .any(|v| !v.is_finite() || *v <= 0.0)
    {
        return STATUS_INVALID_ARGUMENT;
    }
    let key = DecodePairKey {
        first: DecodeWeightKey {
            address: first_w.as_ptr() as usize,
            bytes: first_len,
            k,
            n: first_n,
            kind: DecodeWeightKind::Q4K,
        },
        second: DecodeWeightKey {
            address: second_w.as_ptr() as usize,
            bytes: second_len,
            k,
            n: second_n,
            kind: DecodeWeightKind::Q4K,
        },
    };
    execute_cached_w8a8_pair_m1(context, key, a, out1, out2, || {
        let mut w = Vec::with_capacity(first_len + second_len);
        w.extend_from_slice(first_w);
        w.extend_from_slice(second_w);
        let mut sc = Vec::with_capacity(first_n + second_n);
        sc.extend_from_slice(first_s);
        sc.extend_from_slice(second_s);
        Some((w, sc))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocknpu_matmul_w8a8_triple_f32_f32_m1(
    context: *mut RockNpuContext,
    first_weights_nk_i8: *const i8,
    first_scales_n_f32: *const f32,
    first_n: usize,
    second_weights_nk_i8: *const i8,
    second_scales_n_f32: *const f32,
    second_n: usize,
    third_weights_nk_i8: *const i8,
    third_scales_n_f32: *const f32,
    third_n: usize,
    activations_k_f32: *const f32,
    first_output_f32: *mut f32,
    second_output_f32: *mut f32,
    third_output_f32: *mut f32,
    k: usize,
) -> i32 {
    if context.is_null()
        || first_weights_nk_i8.is_null()
        || first_scales_n_f32.is_null()
        || second_weights_nk_i8.is_null()
        || second_scales_n_f32.is_null()
        || third_weights_nk_i8.is_null()
        || third_scales_n_f32.is_null()
        || activations_k_f32.is_null()
        || first_output_f32.is_null()
        || second_output_f32.is_null()
        || third_output_f32.is_null()
        || k == 0
        || first_n == 0
        || second_n == 0
        || third_n == 0
    {
        return STATUS_INVALID_ARGUMENT;
    }
    let Some(first_len) = k.checked_mul(first_n) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(second_len) = k.checked_mul(second_n) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(third_len) = k.checked_mul(third_n) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let (w1, s1, w2, s2, w3, s3, a, o1, o2, o3, context) = unsafe {
        (
            slice::from_raw_parts(first_weights_nk_i8, first_len),
            slice::from_raw_parts(first_scales_n_f32, first_n),
            slice::from_raw_parts(second_weights_nk_i8, second_len),
            slice::from_raw_parts(second_scales_n_f32, second_n),
            slice::from_raw_parts(third_weights_nk_i8, third_len),
            slice::from_raw_parts(third_scales_n_f32, third_n),
            slice::from_raw_parts(activations_k_f32, k),
            slice::from_raw_parts_mut(first_output_f32, first_n),
            slice::from_raw_parts_mut(second_output_f32, second_n),
            slice::from_raw_parts_mut(third_output_f32, third_n),
            &mut *context,
        )
    };
    if s1
        .iter()
        .chain(s2)
        .chain(s3)
        .any(|v| !v.is_finite() || *v <= 0.0)
    {
        return STATUS_INVALID_ARGUMENT;
    }
    let key = DecodeTripleKey {
        first: DecodeWeightKey {
            address: w1.as_ptr() as usize,
            bytes: first_len,
            k,
            n: first_n,
            kind: DecodeWeightKind::Q4K,
        },
        second: DecodeWeightKey {
            address: w2.as_ptr() as usize,
            bytes: second_len,
            k,
            n: second_n,
            kind: DecodeWeightKind::Q4K,
        },
        third: DecodeWeightKey {
            address: w3.as_ptr() as usize,
            bytes: third_len,
            k,
            n: third_n,
            kind: DecodeWeightKind::Q4K,
        },
    };
    execute_cached_w8a8_triple_m1(context, key, a, o1, o2, o3, || {
        let mut w = Vec::with_capacity(first_len + second_len + third_len);
        w.extend_from_slice(w1);
        w.extend_from_slice(w2);
        w.extend_from_slice(w3);
        let mut sc = Vec::with_capacity(first_n + second_n + third_n);
        sc.extend_from_slice(s1);
        sc.extend_from_slice(s2);
        sc.extend_from_slice(s3);
        Some((w, sc))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocknpu_matmul_q_pair_f32_f32_m1(
    context: *mut RockNpuContext,
    first_weights: *const u8,
    first_bytes: usize,
    first_kind: u32,
    first_n: usize,
    second_weights: *const u8,
    second_bytes: usize,
    second_kind: u32,
    second_n: usize,
    activations_k_f32: *const f32,
    first_output_f32: *mut f32,
    second_output_f32: *mut f32,
    k: usize,
) -> i32 {
    if context.is_null()
        || first_weights.is_null()
        || second_weights.is_null()
        || activations_k_f32.is_null()
        || first_output_f32.is_null()
        || second_output_f32.is_null()
        || k == 0
        || first_n == 0
        || second_n == 0
    {
        return STATUS_INVALID_ARGUMENT;
    }
    let first_kind = match first_kind {
        4 => DecodeWeightKind::Q4K,
        6 => DecodeWeightKind::Q6K,
        _ => return STATUS_INVALID_ARGUMENT,
    };
    let second_kind = match second_kind {
        4 => DecodeWeightKind::Q4K,
        6 => DecodeWeightKind::Q6K,
        _ => return STATUS_INVALID_ARGUMENT,
    };
    let (first, second, activations, first_output, second_output, context) = unsafe {
        (
            slice::from_raw_parts(first_weights, first_bytes),
            slice::from_raw_parts(second_weights, second_bytes),
            slice::from_raw_parts(activations_k_f32, k),
            slice::from_raw_parts_mut(first_output_f32, first_n),
            slice::from_raw_parts_mut(second_output_f32, second_n),
            &mut *context,
        )
    };
    let key = DecodePairKey {
        first: DecodeWeightKey {
            address: first.as_ptr() as usize,
            bytes: first.len(),
            k,
            n: first_n,
            kind: first_kind,
        },
        second: DecodeWeightKey {
            address: second.as_ptr() as usize,
            bytes: second.len(),
            k,
            n: second_n,
            kind: second_kind,
        },
    };
    execute_cached_w8a8_pair_m1(
        context,
        key,
        activations,
        first_output,
        second_output,
        || {
            let (mut weights, mut scales) = match first_kind {
                DecodeWeightKind::Q4K => prepare_q4_k_w8a8(first, k, first_n)?,
                DecodeWeightKind::Q6K => prepare_q6_k_w8a8(first, k, first_n)?,
            };
            let (second_weights, second_scales) = match second_kind {
                DecodeWeightKind::Q4K => prepare_q4_k_w8a8(second, k, second_n)?,
                DecodeWeightKind::Q6K => prepare_q6_k_w8a8(second, k, second_n)?,
            };
            weights.extend_from_slice(&second_weights);
            scales.extend_from_slice(&second_scales);
            Some((weights, scales))
        },
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocknpu_matmul_q_pair_f32_f32_mtile(
    context: *mut RockNpuContext,
    first_weights: *const u8,
    first_bytes: usize,
    first_kind: u32,
    first_n: usize,
    second_weights: *const u8,
    second_bytes: usize,
    second_kind: u32,
    second_n: usize,
    activations_mk_f32: *const f32,
    first_output_mn_f32: *mut f32,
    second_output_mn_f32: *mut f32,
    m: usize,
    k: usize,
) -> i32 {
    if context.is_null()
        || first_weights.is_null()
        || second_weights.is_null()
        || activations_mk_f32.is_null()
        || first_output_mn_f32.is_null()
        || second_output_mn_f32.is_null()
        || m == 0
        || k == 0
        || first_n == 0
        || second_n == 0
    {
        return STATUS_INVALID_ARGUMENT;
    }
    let first_kind = match first_kind {
        4 => DecodeWeightKind::Q4K,
        6 => DecodeWeightKind::Q6K,
        _ => return STATUS_INVALID_ARGUMENT,
    };
    let second_kind = match second_kind {
        4 => DecodeWeightKind::Q4K,
        6 => DecodeWeightKind::Q6K,
        _ => return STATUS_INVALID_ARGUMENT,
    };
    let Some(a_len) = m.checked_mul(k) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(first_out_len) = m.checked_mul(first_n) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(second_out_len) = m.checked_mul(second_n) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let (first, second, activations, first_output, second_output, context) = unsafe {
        (
            slice::from_raw_parts(first_weights, first_bytes),
            slice::from_raw_parts(second_weights, second_bytes),
            slice::from_raw_parts(activations_mk_f32, a_len),
            slice::from_raw_parts_mut(first_output_mn_f32, first_out_len),
            slice::from_raw_parts_mut(second_output_mn_f32, second_out_len),
            &mut *context,
        )
    };
    let key = DecodePairKey {
        first: DecodeWeightKey {
            address: first.as_ptr() as usize,
            bytes: first.len(),
            k,
            n: first_n,
            kind: first_kind,
        },
        second: DecodeWeightKey {
            address: second.as_ptr() as usize,
            bytes: second.len(),
            k,
            n: second_n,
            kind: second_kind,
        },
    };
    execute_cached_w8a8_mtile_pair_pool(
        context,
        key,
        m,
        activations,
        first_output,
        second_output,
        || {
            let first_prepared = match first_kind {
                DecodeWeightKind::Q4K => prepare_q4_k_w8a8(first, k, first_n)?,
                DecodeWeightKind::Q6K => prepare_q6_k_w8a8(first, k, first_n)?,
            };
            let second_prepared = match second_kind {
                DecodeWeightKind::Q4K => prepare_q4_k_w8a8(second, k, second_n)?,
                DecodeWeightKind::Q6K => prepare_q6_k_w8a8(second, k, second_n)?,
            };
            Some((first_prepared, second_prepared))
        },
    )
}

/// Execute three M=1 quantized projections that share one activation by
/// concatenating their W8 rows along N and issuing one prepared NPU matmul.
/// Kinds are 4 for Q4_K and 6 for Q6_K. Outputs preserve the same per-row
/// W8 dequantization semantics as independent calls.
///
/// # Safety
/// All pointers must satisfy their documented byte/element lengths for the
/// duration of this synchronous call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocknpu_matmul_q_triple_f32_f32_m1(
    context: *mut RockNpuContext,
    first_weights: *const u8,
    first_bytes: usize,
    first_kind: u32,
    first_n: usize,
    second_weights: *const u8,
    second_bytes: usize,
    second_kind: u32,
    second_n: usize,
    third_weights: *const u8,
    third_bytes: usize,
    third_kind: u32,
    third_n: usize,
    activations_k_f32: *const f32,
    first_output_f32: *mut f32,
    second_output_f32: *mut f32,
    third_output_f32: *mut f32,
    k: usize,
) -> i32 {
    if context.is_null()
        || first_weights.is_null()
        || second_weights.is_null()
        || third_weights.is_null()
        || activations_k_f32.is_null()
        || first_output_f32.is_null()
        || second_output_f32.is_null()
        || third_output_f32.is_null()
        || k == 0
        || first_n == 0
        || second_n == 0
        || third_n == 0
    {
        return STATUS_INVALID_ARGUMENT;
    }
    let decode_kind = |kind| match kind {
        4 => Some(DecodeWeightKind::Q4K),
        6 => Some(DecodeWeightKind::Q6K),
        _ => None,
    };
    let Some(first_kind) = decode_kind(first_kind) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(second_kind) = decode_kind(second_kind) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(third_kind) = decode_kind(third_kind) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let (first, second, third, activations, first_output, second_output, third_output, context) = unsafe {
        (
            slice::from_raw_parts(first_weights, first_bytes),
            slice::from_raw_parts(second_weights, second_bytes),
            slice::from_raw_parts(third_weights, third_bytes),
            slice::from_raw_parts(activations_k_f32, k),
            slice::from_raw_parts_mut(first_output_f32, first_n),
            slice::from_raw_parts_mut(second_output_f32, second_n),
            slice::from_raw_parts_mut(third_output_f32, third_n),
            &mut *context,
        )
    };
    let weight_key = |bytes: &[u8], n, kind| DecodeWeightKey {
        address: bytes.as_ptr() as usize,
        bytes: bytes.len(),
        k,
        n,
        kind,
    };
    let key = DecodeTripleKey {
        first: weight_key(first, first_n, first_kind),
        second: weight_key(second, second_n, second_kind),
        third: weight_key(third, third_n, third_kind),
    };
    execute_cached_w8a8_triple_m1(
        context,
        key,
        activations,
        first_output,
        second_output,
        third_output,
        || {
            let prepare_one = |bytes: &[u8], kind, n| match kind {
                DecodeWeightKind::Q4K => prepare_q4_k_w8a8(bytes, k, n),
                DecodeWeightKind::Q6K => prepare_q6_k_w8a8(bytes, k, n),
            };
            let (mut weights, mut scales) = prepare_one(first, first_kind, first_n)?;
            let (second_weights, second_scales) = prepare_one(second, second_kind, second_n)?;
            let (third_weights, third_scales) = prepare_one(third, third_kind, third_n)?;
            weights.extend_from_slice(&second_weights);
            weights.extend_from_slice(&third_weights);
            scales.extend_from_slice(&second_scales);
            scales.extend_from_slice(&third_scales);
            Some((weights, scales))
        },
    )
}

/// Execute C[M,N] = A[M,K] x B[N,K]^T with GGML-compatible Q4_K weights.
///
/// Q4_K is decoded in this Rust bridge into the same FP16 weight contract used
/// by the existing RockNPU executor. The external GGML adapter only forwards
/// raw block bytes and never owns quantization math.
///
/// # Safety
///
/// `context` must point to a live RockNPU context. `weights_nk_q4_k` must
/// reference exactly `weights_bytes` readable bytes encoding N contiguous rows
/// of K Q4_K values. Activation/output pointers must provide `M*K` readable and
/// `M*N` writable f32 elements respectively for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocknpu_matmul_q4_k_f32_f32(
    context: *mut RockNpuContext,
    weights_nk_q4_k: *const u8,
    weights_bytes: usize,
    activations_mk_f32: *const f32,
    output_mn_f32: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) -> i32 {
    const Q4_K_VALUES_PER_BLOCK: usize = 256;

    if context.is_null()
        || weights_nk_q4_k.is_null()
        || activations_mk_f32.is_null()
        || output_mn_f32.is_null()
        || m == 0
        || k == 0
        || n == 0
        || !k.is_multiple_of(Q4_K_VALUES_PER_BLOCK)
    {
        return STATUS_INVALID_ARGUMENT;
    }

    let Some(a_len) = m.checked_mul(k) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(weight_values) = n.checked_mul(k) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(out_len) = m.checked_mul(n) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let blocks_per_row = k / Q4_K_VALUES_PER_BLOCK;
    let Some(block_count) = n.checked_mul(blocks_per_row) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(expected_weight_bytes) = block_count.checked_mul(size_of::<BlockQ4K>()) else {
        return STATUS_INVALID_ARGUMENT;
    };
    if weights_bytes != expected_weight_bytes {
        return STATUS_INVALID_ARGUMENT;
    }

    // SAFETY: pointer/null/size contracts are validated above. The caller owns
    // the buffers for the duration of this synchronous call.
    if m > 1 && native_mtile_routes_enabled() {
        let activations = unsafe { slice::from_raw_parts(activations_mk_f32, a_len) };
        if let Some(stacked) = hilo_stack(activations, m, k) {
            let mut stacked_out = vec![0.0f32; 2 * out_len];
            let status = {
                let _inner = HiloInner::enter();
                unsafe {
                    rocknpu_matmul_q4_k_f32_f32(
                        context,
                        weights_nk_q4_k,
                        weights_bytes,
                        stacked.as_ptr(),
                        stacked_out.as_mut_ptr(),
                        2 * m,
                        k,
                        n,
                    )
                }
            };
            if status == STATUS_OK {
                hilo_sum(unsafe { slice::from_raw_parts_mut(output_mn_f32, out_len) }, &stacked_out);
            }
            return status;
        }
    }
    if m > 1 && !MTILE_ROWS.contains(&m) && native_mtile_routes_enabled() {
        // Other prompt/batch sizes run as native tiles: 128-row tiles plus a
        // zero-padded tail.
        return unsafe {
            run_mtile_row_chunks(m, k, activations_mk_f32, &[output_mn_f32], &[n], |a, outs, tile| {
                rocknpu_matmul_q4_k_f32_f32(context, weights_nk_q4_k, weights_bytes, a, outs[0], tile, k, n)
            })
        };
    }
    if m > 1 && native_mtile_routes_enabled() {
        let split = unsafe {
            split_wide_mtile(
                m,
                k,
                n,
                weights_nk_q4_k,
                weights_bytes,
                activations_mk_f32,
                output_mn_f32,
                |w, bytes, a, out, rows, cols| {
                    rocknpu_matmul_q4_k_f32_f32(context, w, bytes, a, out, rows, k, cols)
                },
            )
        };
        if let Some(status) = split {
            return status;
        }
    }

    let (weight_bytes, activations, output, context) = unsafe {
        (
            slice::from_raw_parts(weights_nk_q4_k, weights_bytes),
            slice::from_raw_parts(activations_mk_f32, a_len),
            slice::from_raw_parts_mut(output_mn_f32, out_len),
            &mut *context,
        )
    };

    if m == 1 {
        let key = DecodeWeightKey {
            address: weight_bytes.as_ptr() as usize,
            bytes: weight_bytes.len(),
            k,
            n,
            kind: DecodeWeightKind::Q4K,
        };
        if context.w4a4_enabled
            && w4a4_shape_enabled(k, n)
            && k <= 10_752
            && n.is_multiple_of(64)
            && n <= 8192
        {
            if let Some(group_size) = w4a4_group_size(k) {
                let hadamard = env_enabled("ROCKNPU_W4A4_HADAMARD");
                if group_size == k && context.w4a4_pool.is_some() {
                    return execute_cached_pool_w4a4_m1(
                        context,
                        key,
                        hadamard,
                        activations,
                        output,
                        || prepare_q4_k_grouped_w4a4(weight_bytes, k, n, group_size, hadamard),
                    );
                }
                return execute_cached_grouped_w4a4_m1(
                    context,
                    key,
                    group_size,
                    hadamard,
                    activations,
                    output,
                    || prepare_q4_k_grouped_w4a4(weight_bytes, k, n, group_size, hadamard),
                );
            }
            return execute_cached_w4a4_m1(context, key, activations, output, || {
                prepare_q4_k_w4a4(weight_bytes, k, n)
            });
        }
        return execute_cached_w8a8_m1(context, key, activations, output, || {
            prepare_q4_k_w8a8(weight_bytes, k, n)
        });
    }

    if matches!(m, 4 | 8 | 12 | 16 | 32 | 48 | 64 | 128)
        && !k.is_multiple_of(512)
        && k.next_multiple_of(512) <= 3 * 4096
        && n.is_multiple_of(32)
        && n <= 8192
        && native_mtile_routes_enabled()
        && env_enabled("ROCKNPU_NATIVE_MTILE_DOWN")
    {
        let key = DecodeWeightKey {
            address: weight_bytes.as_ptr() as usize,
            bytes: weight_bytes.len(),
            k,
            n,
            kind: DecodeWeightKind::Q4K,
        };
        return execute_padded_k_mtile(context, key, m, activations, output, || {
            prepare_q4_k_w8a8(weight_bytes, k, n)
        });
    }

    if matches!(m, 4 | 8 | 12 | 16 | 32 | 48 | 64 | 128)
        && k > 4096
        && k <= 3 * 4096
        && k.is_multiple_of(512)
        && n.is_multiple_of(32)
        && n <= 8192
        && env_enabled("ROCKNPU_NATIVE_MTILE")
        && env_enabled("ROCKNPU_NATIVE_MTILE_DOWN")
        && env_enabled("ROCKNPU_MTILE_PERSIST")
        && env_enabled("ROCKNPU_MTILE_MC")
        && mtile_shape_enabled(k, n)
    {
        let key = DecodeWeightKey {
            address: weight_bytes.as_ptr() as usize,
            bytes: weight_bytes.len(),
            k,
            n,
            kind: DecodeWeightKind::Q4K,
        };
        return execute_cached_w8a8_mtile_pool(
            context,
            key,
            m,
            Int8DecodeSplit::K,
            activations,
            output,
            || prepare_q4_k_w8a8(weight_bytes, k, n),
        );
    }

    if matches!(m, 4 | 8 | 12 | 32 | 48 | 64 | 128)
        && env_enabled("ROCKNPU_NATIVE_MTILE")
        && env_enabled("ROCKNPU_MTILE_PERSIST")
        && k <= 4096
        && k.is_multiple_of(512)
        && n.is_multiple_of(32)
        && n <= 8192
        && mtile_shape_enabled(k, n)
    {
        let key = DecodeWeightKey {
            address: weight_bytes.as_ptr() as usize,
            bytes: weight_bytes.len(),
            k,
            n,
            kind: DecodeWeightKind::Q4K,
        };
        if env_enabled("ROCKNPU_MTILE_MC") && n >= MTILE_POOL_MIN_N {
            return execute_cached_w8a8_mtile_pool(
                context,
                key,
                m,
                Int8DecodeSplit::N,
                activations,
                output,
                || prepare_q4_k_w8a8(weight_bytes, k, n),
            );
        }
        return execute_cached_w8a8_mtile(context, key, m, activations, output, || {
            prepare_q4_k_w8a8(weight_bytes, k, n)
        });
    }

    if m == 16
        && k <= 4096
        && k.is_multiple_of(512)
        && n.is_multiple_of(32)
        && n <= 8192
        && mtile_shape_enabled(k, n)
    {
        let key = DecodeWeightKey {
            address: weight_bytes.as_ptr() as usize,
            bytes: weight_bytes.len(),
            k,
            n,
            kind: DecodeWeightKind::Q4K,
        };
        if let Some(group_size) = mtile_qo_group_size(k, n) {
            if env_enabled("ROCKNPU_MTILE_MC") {
                return execute_cached_grouped_w8a8_m16_pool(
                    context,
                    key,
                    group_size,
                    activations,
                    output,
                    || prepare_q4_k_grouped_w8a8(weight_bytes, k, n, group_size),
                );
            }
            return execute_cached_grouped_w8a8_m16(
                context,
                key,
                group_size,
                activations,
                output,
                || prepare_q4_k_grouped_w8a8(weight_bytes, k, n, group_size),
            );
        }
        if env_enabled("ROCKNPU_MTILE_MC") && n >= 2048 {
            return execute_cached_w8a8_m16_pool(context, key, activations, output, || {
                prepare_q4_k_w8a8(weight_bytes, k, n)
            });
        }
        return execute_cached_w8a8_m16(context, key, activations, output, || {
            prepare_q4_k_w8a8(weight_bytes, k, n)
        });
    }

    if env_enabled("ROCKNPU_PREFILL_CACHE") {
        let key = DecodeWeightKey {
            address: weight_bytes.as_ptr() as usize,
            bytes: weight_bytes.len(),
            k,
            n,
            kind: DecodeWeightKind::Q4K,
        };
        return execute_cached_prefill(context, key, m, activations, output, || {
            dequantize_q4_k_prefill_f16(
                weight_bytes,
                weight_values,
                env_enabled_default("ROCKNPU_PREFILL_PARALLEL_DEQUANT", true),
            )
        });
    }

    let mut weights = Vec::with_capacity(weight_values);
    let mut decoded = [0.0f32; Q4_K_VALUES_PER_BLOCK];
    for bytes in weight_bytes.chunks_exact(size_of::<BlockQ4K>()) {
        let block: BlockQ4K = pod_read_unaligned(bytes);
        dequantize_q4_k(&block, &mut decoded);
        weights.extend(decoded.iter().copied().map(f16::from_f32));
    }
    if weights.len() != weight_values {
        return STATUS_INVALID_ARGUMENT;
    }
    let activations = activations
        .iter()
        .copied()
        .map(f16::from_f32)
        .collect::<Vec<_>>();

    let mut executor = match Fp16MatmulExecutor::new(&context.device) {
        Ok(executor) => executor,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    let result = match executor.execute_f32(&activations, &weights, m, k, n) {
        Ok(result) => result,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    output.copy_from_slice(&result.values);
    STATUS_OK
}

/// Execute C[M,N] = A[M,K] x B[N,K]^T with GGML-compatible Q6_K weights.
///
/// Like the Q4_K bridge, Q6_K is decoded in Rust into the existing FP16
/// executor contract. The external GGML adapter only forwards raw block bytes.
///
/// # Safety
///
/// `context` must point to a live RockNPU context. `weights_nk_q6_k` must
/// reference exactly `weights_bytes` readable bytes encoding N contiguous rows
/// of K Q6_K values. Activation/output pointers must provide `M*K` readable and
/// `M*N` writable f32 elements respectively for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocknpu_matmul_q6_k_f32_f32(
    context: *mut RockNpuContext,
    weights_nk_q6_k: *const u8,
    weights_bytes: usize,
    activations_mk_f32: *const f32,
    output_mn_f32: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) -> i32 {
    const Q6_K_VALUES_PER_BLOCK: usize = 256;

    if context.is_null()
        || weights_nk_q6_k.is_null()
        || activations_mk_f32.is_null()
        || output_mn_f32.is_null()
        || m == 0
        || k == 0
        || n == 0
        || !k.is_multiple_of(Q6_K_VALUES_PER_BLOCK)
    {
        return STATUS_INVALID_ARGUMENT;
    }

    let Some(a_len) = m.checked_mul(k) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(weight_values) = n.checked_mul(k) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(out_len) = m.checked_mul(n) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let blocks_per_row = k / Q6_K_VALUES_PER_BLOCK;
    let Some(block_count) = n.checked_mul(blocks_per_row) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(expected_weight_bytes) = block_count.checked_mul(size_of::<BlockQ6K>()) else {
        return STATUS_INVALID_ARGUMENT;
    };
    if weights_bytes != expected_weight_bytes {
        return STATUS_INVALID_ARGUMENT;
    }

    // SAFETY: pointer/null/size contracts are validated above. The caller owns
    // the buffers for the duration of this synchronous call.
    if m > 1 && native_mtile_routes_enabled() {
        let activations = unsafe { slice::from_raw_parts(activations_mk_f32, a_len) };
        if let Some(stacked) = hilo_stack(activations, m, k) {
            let mut stacked_out = vec![0.0f32; 2 * out_len];
            let status = {
                let _inner = HiloInner::enter();
                unsafe {
                    rocknpu_matmul_q6_k_f32_f32(
                        context,
                        weights_nk_q6_k,
                        weights_bytes,
                        stacked.as_ptr(),
                        stacked_out.as_mut_ptr(),
                        2 * m,
                        k,
                        n,
                    )
                }
            };
            if status == STATUS_OK {
                hilo_sum(unsafe { slice::from_raw_parts_mut(output_mn_f32, out_len) }, &stacked_out);
            }
            return status;
        }
    }
    if m > 1 && !MTILE_ROWS.contains(&m) && native_mtile_routes_enabled() {
        // Other prompt/batch sizes run as native tiles: 128-row tiles plus a
        // zero-padded tail.
        return unsafe {
            run_mtile_row_chunks(m, k, activations_mk_f32, &[output_mn_f32], &[n], |a, outs, tile| {
                rocknpu_matmul_q6_k_f32_f32(context, weights_nk_q6_k, weights_bytes, a, outs[0], tile, k, n)
            })
        };
    }
    if m > 1 && native_mtile_routes_enabled() {
        let split = unsafe {
            split_wide_mtile(
                m,
                k,
                n,
                weights_nk_q6_k,
                weights_bytes,
                activations_mk_f32,
                output_mn_f32,
                |w, bytes, a, out, rows, cols| {
                    rocknpu_matmul_q6_k_f32_f32(context, w, bytes, a, out, rows, k, cols)
                },
            )
        };
        if let Some(status) = split {
            return status;
        }
    }

    let (weight_bytes, activations, output, context) = unsafe {
        (
            slice::from_raw_parts(weights_nk_q6_k, weights_bytes),
            slice::from_raw_parts(activations_mk_f32, a_len),
            slice::from_raw_parts_mut(output_mn_f32, out_len),
            &mut *context,
        )
    };

    if m == 1 {
        let key = DecodeWeightKey {
            address: weight_bytes.as_ptr() as usize,
            bytes: weight_bytes.len(),
            k,
            n,
            kind: DecodeWeightKind::Q6K,
        };
        return execute_cached_w8a8_m1(context, key, activations, output, || {
            prepare_q6_k_w8a8(weight_bytes, k, n)
        });
    }

    if matches!(m, 4 | 8 | 12 | 16 | 32 | 48 | 64 | 128)
        && !k.is_multiple_of(512)
        && k.next_multiple_of(512) <= 3 * 4096
        && n.is_multiple_of(32)
        && n <= 8192
        && native_mtile_routes_enabled()
        && env_enabled("ROCKNPU_NATIVE_MTILE_DOWN")
    {
        let key = DecodeWeightKey {
            address: weight_bytes.as_ptr() as usize,
            bytes: weight_bytes.len(),
            k,
            n,
            kind: DecodeWeightKind::Q6K,
        };
        return execute_padded_k_mtile(context, key, m, activations, output, || {
            prepare_q6_k_w8a8(weight_bytes, k, n)
        });
    }

    if matches!(m, 4 | 8 | 12 | 16 | 32 | 48 | 64 | 128)
        && k > 4096
        && k <= 3 * 4096
        && k.is_multiple_of(512)
        && n.is_multiple_of(32)
        && n <= 8192
        && env_enabled("ROCKNPU_NATIVE_MTILE")
        && env_enabled("ROCKNPU_NATIVE_MTILE_DOWN")
        && env_enabled("ROCKNPU_MTILE_PERSIST")
        && env_enabled("ROCKNPU_MTILE_MC")
        && mtile_shape_enabled(k, n)
    {
        let key = DecodeWeightKey {
            address: weight_bytes.as_ptr() as usize,
            bytes: weight_bytes.len(),
            k,
            n,
            kind: DecodeWeightKind::Q6K,
        };
        return execute_cached_w8a8_mtile_pool(
            context,
            key,
            m,
            Int8DecodeSplit::K,
            activations,
            output,
            || prepare_q6_k_w8a8(weight_bytes, k, n),
        );
    }

    if matches!(m, 4 | 8 | 12 | 32 | 48 | 64 | 128)
        && env_enabled("ROCKNPU_NATIVE_MTILE")
        && env_enabled("ROCKNPU_MTILE_PERSIST")
        && k <= 4096
        && k.is_multiple_of(512)
        && n.is_multiple_of(32)
        && n <= 8192
        && mtile_shape_enabled(k, n)
    {
        let key = DecodeWeightKey {
            address: weight_bytes.as_ptr() as usize,
            bytes: weight_bytes.len(),
            k,
            n,
            kind: DecodeWeightKind::Q6K,
        };
        if env_enabled("ROCKNPU_MTILE_MC") && n >= MTILE_POOL_MIN_N {
            return execute_cached_w8a8_mtile_pool(
                context,
                key,
                m,
                Int8DecodeSplit::N,
                activations,
                output,
                || prepare_q6_k_w8a8(weight_bytes, k, n),
            );
        }
        return execute_cached_w8a8_mtile(context, key, m, activations, output, || {
            prepare_q6_k_w8a8(weight_bytes, k, n)
        });
    }

    if m == 16
        && k <= 4096
        && k.is_multiple_of(512)
        && n.is_multiple_of(32)
        && n <= 8192
        && mtile_shape_enabled(k, n)
    {
        let key = DecodeWeightKey {
            address: weight_bytes.as_ptr() as usize,
            bytes: weight_bytes.len(),
            k,
            n,
            kind: DecodeWeightKind::Q6K,
        };
        if let Some(group_size) = mtile_qo_group_size(k, n) {
            if env_enabled("ROCKNPU_MTILE_MC") {
                return execute_cached_grouped_w8a8_m16_pool(
                    context,
                    key,
                    group_size,
                    activations,
                    output,
                    || prepare_q6_k_grouped_w8a8(weight_bytes, k, n, group_size),
                );
            }
            return execute_cached_grouped_w8a8_m16(
                context,
                key,
                group_size,
                activations,
                output,
                || prepare_q6_k_grouped_w8a8(weight_bytes, k, n, group_size),
            );
        }
        if env_enabled("ROCKNPU_MTILE_MC") && n >= 2048 {
            return execute_cached_w8a8_m16_pool(context, key, activations, output, || {
                prepare_q6_k_w8a8(weight_bytes, k, n)
            });
        }
        return execute_cached_w8a8_m16(context, key, activations, output, || {
            prepare_q6_k_w8a8(weight_bytes, k, n)
        });
    }

    if env_enabled("ROCKNPU_PREFILL_CACHE") {
        let key = DecodeWeightKey {
            address: weight_bytes.as_ptr() as usize,
            bytes: weight_bytes.len(),
            k,
            n,
            kind: DecodeWeightKind::Q6K,
        };
        return execute_cached_prefill(context, key, m, activations, output, || {
            dequantize_q6_k_prefill_f16(
                weight_bytes,
                weight_values,
                env_enabled_default("ROCKNPU_PREFILL_PARALLEL_DEQUANT", true),
            )
        });
    }

    let mut weights = Vec::with_capacity(weight_values);
    let mut decoded = [0.0f32; Q6_K_VALUES_PER_BLOCK];
    for bytes in weight_bytes.chunks_exact(size_of::<BlockQ6K>()) {
        let block: BlockQ6K = pod_read_unaligned(bytes);
        dequantize_q6_k(&block, &mut decoded);
        weights.extend(decoded.iter().copied().map(f16::from_f32));
    }
    if weights.len() != weight_values {
        return STATUS_INVALID_ARGUMENT;
    }
    let activations = activations
        .iter()
        .copied()
        .map(f16::from_f32)
        .collect::<Vec<_>>();

    let mut executor = match Fp16MatmulExecutor::new(&context.device) {
        Ok(executor) => executor,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    let result = match executor.execute_f32(&activations, &weights, m, k, n) {
        Ok(result) => result,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    output.copy_from_slice(&result.values);
    STATUS_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_quantize_second_pass_matches_scalar_bits() {
        let mut values = vec![
            -12.75f32,
            -7.5,
            -3.5,
            -1.5,
            -0.5,
            -0.499_999_97,
            0.0,
            0.499_999_97,
            0.5,
            1.5,
            3.5,
            7.5,
            12.75,
        ];
        let mut state = 0x1234_5678u32;
        for _ in 0..2051 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / ((1u32 << 24) - 1) as f32;
            values.push((unit * 2.0 - 1.0) * 17.0);
        }
        let max_abs = values.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = max_abs / 127.0;
        let scalar = values
            .iter()
            .map(|&value| (value / scale).round().clamp(-127.0, 127.0) as i8)
            .collect::<Vec<_>>();
        // SAFETY: the test runs only on AArch64 and the helper supports arbitrary tails.
        let neon = unsafe { quantize_symmetric_neon_second_pass(&values, scale) };
        assert_eq!(neon, scalar);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_max_abs_matches_scalar_and_rejects_nonfinite() {
        let mut values = (0..2051)
            .map(|i| (((i * 7_919) % 65_521) as f32 - 32_760.0) * 0.001)
            .collect::<Vec<_>>();
        let scalar = values.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        // SAFETY: the test runs only on AArch64 and the helper handles arbitrary tails.
        let neon = unsafe { max_abs_finite_neon(&values) }.expect("finite input");
        assert_eq!(neon.to_bits(), scalar.to_bits());
        values[1024] = f32::INFINITY;
        assert!(unsafe { max_abs_finite_neon(&values) }.is_none());
        values[1024] = f32::NAN;
        assert!(unsafe { max_abs_finite_neon(&values) }.is_none());
    }

    #[test]
    fn q4_k_block_size_matches_ggml_abi() {
        assert_eq!(size_of::<BlockQ4K>(), 144);
    }

    #[test]
    fn q6_k_block_size_matches_ggml_abi() {
        assert_eq!(size_of::<BlockQ6K>(), 210);
    }

    #[test]
    fn parallel_q4_k_prefill_dequant_matches_serial() {
        let blocks = PARALLEL_DEQUANT_MIN_VALUES / 256;
        let bytes = vec![0u8; blocks * size_of::<BlockQ4K>()];
        let serial = dequantize_q4_k_prefill_f16(&bytes, PARALLEL_DEQUANT_MIN_VALUES, false);
        let parallel = dequantize_q4_k_prefill_f16(&bytes, PARALLEL_DEQUANT_MIN_VALUES, true);
        assert_eq!(parallel, serial);
    }

    #[test]
    fn parallel_q6_k_prefill_dequant_matches_serial() {
        let blocks = PARALLEL_DEQUANT_MIN_VALUES / 256;
        let bytes = vec![0u8; blocks * size_of::<BlockQ6K>()];
        let serial = dequantize_q6_k_prefill_f16(&bytes, PARALLEL_DEQUANT_MIN_VALUES, false);
        let parallel = dequantize_q6_k_prefill_f16(&bytes, PARALLEL_DEQUANT_MIN_VALUES, true);
        assert_eq!(parallel, serial);
    }

    #[test]
    fn normalized_fwht_preserves_dot_product() {
        let mut a = vec![1.0f32, -2.0, 3.5, 0.25, -1.5, 2.25, 0.5, -0.75];
        let mut b = vec![-0.5f32, 1.25, 2.0, -3.0, 0.75, -1.0, 4.0, 0.5];
        let before: f32 = a.iter().zip(&b).map(|(&x, &y)| x * y).sum();
        assert!(fwht_norm_in_place(&mut a));
        assert!(fwht_norm_in_place(&mut b));
        let after: f32 = a.iter().zip(&b).map(|(&x, &y)| x * y).sum();
        assert!(
            (before - after).abs() < 1.0e-5,
            "before={before} after={after}"
        );
    }

    #[test]
    fn decode_worker_selector_ignores_noise_level_gain() {
        assert_eq!(select_decode_worker_index(&[450, 446, 447]), Some(0));
    }

    #[test]
    fn decode_worker_selector_accepts_material_two_worker_gain() {
        assert_eq!(select_decode_worker_index(&[1598, 985, 1200]), Some(1));
    }

    #[test]
    fn decode_worker_selector_accepts_material_three_worker_gain() {
        assert_eq!(select_decode_worker_index(&[3960, 2220, 1647]), Some(2));
    }

    #[test]
    fn decode_worker_selector_requires_nonempty_samples() {
        assert_eq!(select_decode_worker_index(&[]), None);
    }

    #[test]
    fn matmul_rejects_null_context_before_pointer_access() {
        // SAFETY: null pointers are intentionally supplied to verify that the
        // FFI guard rejects them before any dereference.
        let status = unsafe {
            rocknpu_matmul_f16_f32_f32(
                ptr::null_mut(),
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
                4,
                32,
                16,
            )
        };
        assert_eq!(status, STATUS_INVALID_ARGUMENT);
    }
}
