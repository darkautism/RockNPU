#ifndef ROCKNPU_H
#define ROCKNPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RockNpuContext rocknpu_context;

enum rocknpu_status {
    ROCKNPU_STATUS_OK = 0,
    ROCKNPU_STATUS_INVALID_ARGUMENT = -1,
    ROCKNPU_STATUS_DEVICE_ERROR = -2,
    ROCKNPU_STATUS_EXECUTION_ERROR = -3,
};

size_t rocknpu_device_count(void);
rocknpu_context * rocknpu_context_create(void);
void rocknpu_context_destroy(rocknpu_context * context);

int rocknpu_matmul_f16_f32_f32(
    rocknpu_context * context,
    const uint16_t * weights_nk_f16_bits,
    const float * activations_mk_f32,
    float * output_mn_f32,
    size_t m,
    size_t k,
    size_t n);

int rocknpu_matmul_q4_k_f32_f32(
    rocknpu_context * context,
    const uint8_t * weights_nk_q4_k,
    size_t weights_bytes,
    const float * activations_mk_f32,
    float * output_mn_f32,
    size_t m,
    size_t k,
    size_t n);

#ifdef __cplusplus
}
#endif

#endif
