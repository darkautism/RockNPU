#![forbid(unsafe_op_in_unsafe_fn)]

use bytemuck::pod_read_unaligned;
use half::f16;
use llama_gguf::tensor::quant::{BlockQ4K, BlockQ6K, dequantize_q4_k, dequantize_q6_k};
use rocket_runtime::RocketDevice;
use rocknpu_matmul::{Fp16MatmulExecutor, Int8DecodeExecutor};
use std::mem::size_of;
use std::ptr;
use std::slice;

const STATUS_OK: i32 = 0;
const STATUS_INVALID_ARGUMENT: i32 = -1;
const STATUS_EXECUTION_ERROR: i32 = -3;

pub struct RockNpuContext {
    device: RocketDevice,
}

fn quantize_symmetric(values: &[f32]) -> Option<(Vec<i8>, f32)> {
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
    let scale = max_abs / 127.0;
    let quantized = values
        .iter()
        .map(|&value| (value / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    Some((quantized, scale))
}

fn execute_w8a8_m1(
    context: &RockNpuContext,
    weights_nk_f32: &[f32],
    activations_k_f32: &[f32],
    output_n_f32: &mut [f32],
    k: usize,
    n: usize,
) -> i32 {
    if weights_nk_f32.len() != n.saturating_mul(k)
        || activations_k_f32.len() != k
        || output_n_f32.len() != n
        || !k.is_multiple_of(512)
        || !n.is_multiple_of(32)
        || n > 8192
    {
        return STATUS_INVALID_ARGUMENT;
    }

    let Some((activations_i8, activation_scale)) = quantize_symmetric(activations_k_f32) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let mut weights_i8 = Vec::with_capacity(weights_nk_f32.len());
    let mut weight_scales = Vec::with_capacity(n);
    for row in weights_nk_f32.chunks_exact(k) {
        let Some((quantized, scale)) = quantize_symmetric(row) else {
            return STATUS_INVALID_ARGUMENT;
        };
        weights_i8.extend_from_slice(&quantized);
        weight_scales.push(scale);
    }

    let executor = match Int8DecodeExecutor::new(&context.device) {
        Ok(executor) => executor,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    let result = match executor.execute(&activations_i8, &weights_i8, k, n) {
        Ok(result) => result,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    for ((dst, &acc), &weight_scale) in output_n_f32
        .iter_mut()
        .zip(&result.values)
        .zip(&weight_scales)
    {
        *dst = acc as f32 * activation_scale * weight_scale;
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
    match RocketDevice::open() {
        Ok(device) => Box::into_raw(Box::new(RockNpuContext { device })),
        Err(_) => ptr::null_mut(),
    }
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
        drop(Box::from_raw(context));
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
    let (weight_bytes, activations, output, context) = unsafe {
        (
            slice::from_raw_parts(weights_nk_q4_k, weights_bytes),
            slice::from_raw_parts(activations_mk_f32, a_len),
            slice::from_raw_parts_mut(output_mn_f32, out_len),
            &mut *context,
        )
    };

    if m == 1 {
        let mut weights = Vec::with_capacity(weight_values);
        let mut decoded = [0.0f32; Q4_K_VALUES_PER_BLOCK];
        for bytes in weight_bytes.chunks_exact(size_of::<BlockQ4K>()) {
            let block: BlockQ4K = pod_read_unaligned(bytes);
            dequantize_q4_k(&block, &mut decoded);
            weights.extend_from_slice(&decoded);
        }
        if weights.len() != weight_values {
            return STATUS_INVALID_ARGUMENT;
        }
        return execute_w8a8_m1(context, &weights, activations, output, k, n);
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
    let (weight_bytes, activations, output, context) = unsafe {
        (
            slice::from_raw_parts(weights_nk_q6_k, weights_bytes),
            slice::from_raw_parts(activations_mk_f32, a_len),
            slice::from_raw_parts_mut(output_mn_f32, out_len),
            &mut *context,
        )
    };

    if m == 1 {
        let mut weights = Vec::with_capacity(weight_values);
        let mut decoded = [0.0f32; Q6_K_VALUES_PER_BLOCK];
        for bytes in weight_bytes.chunks_exact(size_of::<BlockQ6K>()) {
            let block: BlockQ6K = pod_read_unaligned(bytes);
            dequantize_q6_k(&block, &mut decoded);
            weights.extend_from_slice(&decoded);
        }
        if weights.len() != weight_values {
            return STATUS_INVALID_ARGUMENT;
        }
        return execute_w8a8_m1(context, &weights, activations, output, k, n);
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

    #[test]
    fn q4_k_block_size_matches_ggml_abi() {
        assert_eq!(size_of::<BlockQ4K>(), 144);
    }

    #[test]
    fn q6_k_block_size_matches_ggml_abi() {
        assert_eq!(size_of::<BlockQ6K>(), 210);
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
