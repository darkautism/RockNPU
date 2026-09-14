#![forbid(unsafe_op_in_unsafe_fn)]

use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_matmul::Fp16MatmulExecutor;
use std::ptr;
use std::slice;

const STATUS_OK: i32 = 0;
const STATUS_INVALID_ARGUMENT: i32 = -1;
const STATUS_EXECUTION_ERROR: i32 = -3;

pub struct RockNpuContext {
    device: RocketDevice,
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

#[cfg(test)]
mod tests {
    use super::*;

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
