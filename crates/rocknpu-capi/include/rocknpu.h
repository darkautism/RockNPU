#ifndef ROCKNPU_H
#define ROCKNPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RockNpuContext rocknpu_context;

typedef struct rocknpu_decode_cache_stats {
    size_t hits;
    size_t misses;
    size_t entries;
    size_t resident_bytes;
    uint64_t hit_ns;
    uint64_t miss_ns;
    size_t tuned_shapes;
    size_t worker1_calls;
    size_t worker2_calls;
    size_t worker3_calls;
    size_t ksplit_calls;
} rocknpu_decode_cache_stats;

enum rocknpu_status {
    ROCKNPU_STATUS_OK = 0,
    ROCKNPU_STATUS_INVALID_ARGUMENT = -1,
    ROCKNPU_STATUS_DEVICE_ERROR = -2,
    ROCKNPU_STATUS_EXECUTION_ERROR = -3,
};

size_t rocknpu_device_count(void);
rocknpu_context * rocknpu_context_create(void);
int rocknpu_context_decode_cache_stats(
    const rocknpu_context * context,
    rocknpu_decode_cache_stats * out);
void rocknpu_context_destroy(rocknpu_context * context);

/* One-shot host callback run between NPU submission and completion wait of
 * the next M=1 W8 projection (single/pair/triple). Null clears it. The
 * caller must run the work itself if the callback did not run. */
typedef void (*rocknpu_overlap_fn)(void * user_data);
int rocknpu_context_set_overlap(rocknpu_context * context, rocknpu_overlap_fn callback, void * user_data);

int rocknpu_prewarm_quantized_m16(
    rocknpu_context * context,
    const uint8_t * weights,
    size_t weights_bytes,
    uint32_t quant_kind,
    size_t k,
    size_t n);

int rocknpu_matmul_f16_f32_f32(
    rocknpu_context * context,
    const uint16_t * weights_nk_f16_bits,
    const float * activations_mk_f32,
    float * output_mn_f32,
    size_t m,
    size_t k,
    size_t n);

int rocknpu_matmul_w8a8_f32_f32_m1(
    rocknpu_context * context,
    const int8_t * weights_nk_i8,
    const float * weight_scales_n_f32,
    const float * activations_k_f32,
    float * output_n_f32,
    size_t k,
    size_t n);

int rocknpu_matmul_w8a8_f32_f32_m16(
    rocknpu_context * context,
    const int8_t * weights_nk_i8,
    const float * weight_scales_n_f32,
    const float * activations_mk_f32,
    float * output_mn_f32,
    size_t k,
    size_t n);

int rocknpu_matmul_w8a8_f32_f32_m1_nsplit(
    rocknpu_context * context,
    const int8_t * weights_nk_i8,
    const float * weight_scales_n_f32,
    const float * activations_k_f32,
    float * output_n_f32,
    size_t k,
    size_t n,
    size_t chunk_n);

int rocknpu_matmul_w8a8_swiglu_down_f32(
    rocknpu_context * context,
    const int8_t * gate_weights_nk_i8,
    const float * gate_scales_n_f32,
    const int8_t * up_weights_nk_i8,
    const float * up_scales_n_f32,
    const int8_t * down_weights_nk_i8,
    const float * down_scales_n_f32,
    const float * activations_k_f32,
    float * output_n_f32,
    size_t projection_k,
    size_t ffn_n,
    size_t output_n);

int rocknpu_matmul_w8a8_pair_f32_f32_m1(
    rocknpu_context * context,
    const int8_t * first_weights_nk_i8,
    const float * first_scales_n_f32,
    size_t first_n,
    const int8_t * second_weights_nk_i8,
    const float * second_scales_n_f32,
    size_t second_n,
    const float * activations_k_f32,
    float * first_output_f32,
    float * second_output_f32,
    size_t k);

int rocknpu_matmul_q_pair_f32_f32_mtile(
    rocknpu_context * context,
    const uint8_t * first_weights,
    size_t first_bytes,
    uint32_t first_kind,
    size_t first_n,
    const uint8_t * second_weights,
    size_t second_bytes,
    uint32_t second_kind,
    size_t second_n,
    const float * activations_mk_f32,
    float * first_output_mn_f32,
    float * second_output_mn_f32,
    size_t m,
    size_t k);

int rocknpu_matmul_w8a8_triple_f32_f32_m1(
    rocknpu_context * context,
    const int8_t * first_weights_nk_i8, const float * first_scales_n_f32, size_t first_n,
    const int8_t * second_weights_nk_i8, const float * second_scales_n_f32, size_t second_n,
    const int8_t * third_weights_nk_i8, const float * third_scales_n_f32, size_t third_n,
    const float * activations_k_f32,
    float * first_output_f32, float * second_output_f32, float * third_output_f32,
    size_t k);

int rocknpu_matmul_q_pair_f32_f32_m1(
    rocknpu_context * context,
    const uint8_t * first_weights,
    size_t first_bytes,
    uint32_t first_kind,
    size_t first_n,
    const uint8_t * second_weights,
    size_t second_bytes,
    uint32_t second_kind,
    size_t second_n,
    const float * activations_k_f32,
    float * first_output_f32,
    float * second_output_f32,
    size_t k);

int rocknpu_matmul_q_triple_f32_f32_m1(
    rocknpu_context * context,
    const uint8_t * first_weights,
    size_t first_bytes,
    uint32_t first_kind,
    size_t first_n,
    const uint8_t * second_weights,
    size_t second_bytes,
    uint32_t second_kind,
    size_t second_n,
    const uint8_t * third_weights,
    size_t third_bytes,
    uint32_t third_kind,
    size_t third_n,
    const float * activations_k_f32,
    float * first_output_f32,
    float * second_output_f32,
    float * third_output_f32,
    size_t k);

int rocknpu_matmul_q4_k_f32_f32(
    rocknpu_context * context,
    const uint8_t * weights_nk_q4_k,
    size_t weights_bytes,
    const float * activations_mk_f32,
    float * output_mn_f32,
    size_t m,
    size_t k,
    size_t n);

int rocknpu_matmul_q6_k_f32_f32(
    rocknpu_context * context,
    const uint8_t * weights_nk_q6_k,
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
