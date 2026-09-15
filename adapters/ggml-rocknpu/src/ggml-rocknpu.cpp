#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "rocknpu.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr size_t ROCKNPU_QKV_MAX_LAYERS = 256;

struct rocknpu_qkv_weight_ref {
    const uint8_t * data = nullptr;
    size_t bytes = 0;
    uint32_t kind = 0;
};

struct rocknpu_qkv_layer_state {
    rocknpu_qkv_weight_ref q;
    rocknpu_qkv_weight_ref v;
    rocknpu_qkv_weight_ref k;
    size_t stable_observations = 0;
    const float * activation = nullptr;
    bool pending = false;
    float v_output[256] = {};
    float k_output[256] = {};
};

struct rocknpu_backend_context {
    rocknpu_context * runtime;
    size_t q4_k_mul_mat_calls = 0;
    size_t q6_k_mul_mat_calls = 0;
    size_t f16_mul_mat_calls = 0;
    size_t w4a4_m1_mul_mat_calls = 0;
    size_t w8a8_m1_mul_mat_calls = 0;
    size_t vk_pair_calls = 0;
    size_t ffn_pair_calls = 0;
    size_t qkv_triple_calls = 0;
    size_t qkv_stash_hits = 0;
    rocknpu_qkv_layer_state qkv[ROCKNPU_QKV_MAX_LAYERS] = {};
};

bool rocknpu_env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool rocknpu_env_enabled_default(const char * name, bool default_value) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    return value[0] != '0';
}

bool rocknpu_vk_pair_enabled() {
    return rocknpu_env_enabled_default("ROCKNPU_VK_PAIR", true);
}

bool rocknpu_ffn_pair_enabled() {
    return rocknpu_env_enabled_default("ROCKNPU_FFN_PAIR", true);
}

bool rocknpu_qkv_triple_enabled() {
    return rocknpu_env_enabled_default("ROCKNPU_QKV_TRIPLE", true);
}

bool rocknpu_quant_kind(const ggml_tensor * weights, uint32_t * kind) {
    if (weights->type == GGML_TYPE_Q4_K) {
        *kind = 4;
        return true;
    }
    if (weights->type == GGML_TYPE_Q6_K) {
        *kind = 6;
        return true;
    }
    return false;
}

bool rocknpu_weight_name_contains(const ggml_tensor * weights, const char * needle) {
    return weights != nullptr && std::strstr(weights->name, needle) != nullptr;
}

bool rocknpu_attention_layer(const ggml_tensor * weights, const char * needle, size_t * layer) {
    if (!rocknpu_weight_name_contains(weights, needle)) {
        return false;
    }
    size_t parsed = 0;
    if (std::sscanf(weights->name, "blk.%zu.", &parsed) != 1 || parsed >= ROCKNPU_QKV_MAX_LAYERS) {
        return false;
    }
    *layer = parsed;
    return true;
}

bool rocknpu_trace_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("ROCKNPU_GGML_TRACE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

bool rocknpu_w4a4_enabled() {
    const char * value = std::getenv("ROCKNPU_W4A4");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool rocknpu_w4a4_shape_enabled(size_t k, size_t n) {
    if (!rocknpu_w4a4_enabled()) {
        return false;
    }
    const char * scope = std::getenv("ROCKNPU_W4A4_SCOPE");
    if (scope == nullptr || scope[0] == '\0') {
        return false;
    }
    if (std::strcmp(scope, "all") == 0) {
        return true;
    }
    if (std::strcmp(scope, "ffn") == 0) {
        return k == 2048 && n == 5632;
    }
    if (std::strcmp(scope, "attn") == 0) {
        return k == 2048 && (n == 2048 || n == 256);
    }
    if (std::strcmp(scope, "proj2048") == 0) {
        return k == 2048 && n == 2048;
    }
    if (std::strcmp(scope, "kv") == 0) {
        return k == 2048 && n == 256;
    }
    return false;
}

const char * rocknpu_device_name(ggml_backend_dev_t) {
    return "ROCKNPU0";
}

const char * rocknpu_device_description(ggml_backend_dev_t) {
    return "RockNPU RK3588";
}

void rocknpu_device_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    // RK3588 uses shared system memory; do not present it as dedicated VRAM.
    *free = 0;
    *total = 0;
}

enum ggml_backend_dev_type rocknpu_device_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

void rocknpu_device_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = rocknpu_device_name(dev);
    props->description = rocknpu_device_description(dev);
    rocknpu_device_memory(dev, &props->memory_free, &props->memory_total);
    props->type = rocknpu_device_type(dev);
    props->device_id = nullptr;
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        /* .mmap_support         = */ false,
    };
}

bool rocknpu_mul_mat_supported(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_MUL_MAT || op->src[0] == nullptr || op->src[1] == nullptr) {
        return false;
    }

    const ggml_tensor * weights = op->src[0];
    const ggml_tensor * activations = op->src[1];
    if ((weights->type != GGML_TYPE_F16 && weights->type != GGML_TYPE_Q4_K &&
         weights->type != GGML_TYPE_Q6_K) ||
        activations->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(activations) || !ggml_is_contiguous(op)) {
        return false;
    }
    if (weights->ne[2] != 1 || weights->ne[3] != 1 || activations->ne[2] != 1 || activations->ne[3] != 1) {
        return false;
    }
    if (weights->ne[0] != activations->ne[0]) {
        return false;
    }

    const int64_t k = weights->ne[0];
    const int64_t n = weights->ne[1];
    const int64_t m = activations->ne[1];
    const bool quantized = weights->type == GGML_TYPE_Q4_K || weights->type == GGML_TYPE_Q6_K;
    if (m == 1) {
        return quantized && k > 0 && n > 0 && k % 512 == 0 && n % 32 == 0 && n <= 8192;
    }
    const int64_t k_alignment = quantized ? 256 : 32;
    return m > 0 && k > 0 && n > 0 && m % 4 == 0 && k % k_alignment == 0 && n % 16 == 0;
}

const char * rocknpu_backend_name(ggml_backend_t) {
    return "ROCKNPU";
}

void rocknpu_backend_free(ggml_backend_t backend) {
    auto * context = static_cast<rocknpu_backend_context *>(backend->context);
    if (rocknpu_trace_enabled()) {
        std::fprintf(stderr,
            "ROCKNPU GGML TRACE summary q4_K_mul_mat=%zu q6_K_mul_mat=%zu f16_mul_mat=%zu w4a4_m1_mul_mat=%zu w8a8_m1_mul_mat=%zu vk_pair_calls=%zu ffn_pair_calls=%zu qkv_triple_calls=%zu qkv_stash_hits=%zu\n",
            context->q4_k_mul_mat_calls,
            context->q6_k_mul_mat_calls,
            context->f16_mul_mat_calls,
            context->w4a4_m1_mul_mat_calls,
            context->w8a8_m1_mul_mat_calls,
            context->vk_pair_calls,
            context->ffn_pair_calls,
            context->qkv_triple_calls,
            context->qkv_stash_hits);
        rocknpu_decode_cache_stats cache = {};
        if (rocknpu_context_decode_cache_stats(context->runtime, &cache) == ROCKNPU_STATUS_OK) {
            const double hit_ms = static_cast<double>(cache.hit_ns) / 1.0e6;
            const double miss_ms = static_cast<double>(cache.miss_ns) / 1.0e6;
            std::fprintf(stderr,
                "ROCKNPU GGML TRACE decode_cache hits=%zu misses=%zu entries=%zu resident_mb=%.2f hit_ms=%.2f hit_avg_ms=%.3f miss_ms=%.2f miss_avg_ms=%.3f tuned_shapes=%zu worker_calls=[%zu,%zu,%zu] ksplit_calls=%zu\n",
                cache.hits,
                cache.misses,
                cache.entries,
                static_cast<double>(cache.resident_bytes) / (1024.0 * 1024.0),
                hit_ms,
                cache.hits == 0 ? 0.0 : hit_ms / static_cast<double>(cache.hits),
                miss_ms,
                cache.misses == 0 ? 0.0 : miss_ms / static_cast<double>(cache.misses),
                cache.tuned_shapes,
                cache.worker1_calls,
                cache.worker2_calls,
                cache.worker3_calls,
                cache.ksplit_calls);
        }
    }
    rocknpu_context_destroy(context->runtime);
    delete context;
    delete backend;
}

enum ggml_status rocknpu_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto * context = static_cast<rocknpu_backend_context *>(backend->context);
    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT: {
                if (!rocknpu_mul_mat_supported(node)) {
                    return GGML_STATUS_FAILED;
                }
                if (rocknpu_qkv_triple_enabled()) {
                    const ggml_tensor * q_weights = node->src[0];
                    const ggml_tensor * q_activations = node->src[1];
                    size_t q_layer = 0;
                    uint32_t q_kind = 0;
                    if (rocknpu_attention_layer(q_weights, ".attn_q.weight", &q_layer) &&
                        q_activations->ne[1] == 1 &&
                        q_weights->ne[0] == 2048 && q_weights->ne[1] == 2048 &&
                        rocknpu_quant_kind(q_weights, &q_kind)) {
                        auto & state = context->qkv[q_layer];
                        const rocknpu_qkv_weight_ref current_q {
                            static_cast<const uint8_t *>(q_weights->data), ggml_nbytes(q_weights), q_kind
                        };
                        const bool same_q =
                            state.q.data == current_q.data && state.q.bytes == current_q.bytes && state.q.kind == current_q.kind;
                        if (!same_q) {
                            state.q = current_q;
                            state.stable_observations = 0;
                            state.pending = false;
                        } else if (state.stable_observations >= 2 && state.v.data != nullptr && state.k.data != nullptr) {
                            state.pending = false;
                            const int status = rocknpu_matmul_q_triple_f32_f32_m1(
                                context->runtime,
                                static_cast<const uint8_t *>(q_weights->data),
                                ggml_nbytes(q_weights),
                                q_kind,
                                2048,
                                state.v.data,
                                state.v.bytes,
                                state.v.kind,
                                256,
                                state.k.data,
                                state.k.bytes,
                                state.k.kind,
                                256,
                                static_cast<const float *>(q_activations->data),
                                static_cast<float *>(node->data),
                                state.v_output,
                                state.k_output,
                                2048);
                            if (status == ROCKNPU_STATUS_OK) {
                                state.activation = static_cast<const float *>(q_activations->data);
                                state.pending = true;
                                context->qkv_triple_calls++;
                                context->w8a8_m1_mul_mat_calls += 3;
                                if (q_weights->type == GGML_TYPE_Q4_K) context->q4_k_mul_mat_calls++; else context->q6_k_mul_mat_calls++;
                                if (state.v.kind == 4) context->q4_k_mul_mat_calls++; else context->q6_k_mul_mat_calls++;
                                if (state.k.kind == 4) context->q4_k_mul_mat_calls++; else context->q6_k_mul_mat_calls++;
                                if (rocknpu_trace_enabled()) {
                                    std::fprintf(stderr,
                                        "ROCKNPU GGML TRACE qkv_triple layer=%zu K=2048 N=2048+256+256 kinds=%u/%u/%u\n",
                                        q_layer, q_kind, state.v.kind, state.k.kind);
                                }
                                break;
                            }
                        }
                    }
                }
                if (rocknpu_ffn_pair_enabled() && i + 1 < graph->n_nodes) {
                    ggml_tensor * second = graph->nodes[i + 1];
                    if (second != nullptr && second->op == GGML_OP_MUL_MAT && rocknpu_mul_mat_supported(second)) {
                        const ggml_tensor * first_weights = node->src[0];
                        const ggml_tensor * second_weights = second->src[0];
                        const ggml_tensor * first_activations = node->src[1];
                        const ggml_tensor * second_activations = second->src[1];
                        uint32_t first_kind = 0;
                        uint32_t second_kind = 0;
                        const bool ffn_pair =
                            rocknpu_weight_name_contains(first_weights, ".ffn_gate.weight") &&
                            rocknpu_weight_name_contains(second_weights, ".ffn_up.weight") &&
                            first_activations == second_activations &&
                            first_activations->ne[1] == 1 &&
                            first_weights->ne[0] == second_weights->ne[0] &&
                            first_weights->ne[1] == 5632 && second_weights->ne[1] == 5632 &&
                            rocknpu_quant_kind(first_weights, &first_kind) &&
                            rocknpu_quant_kind(second_weights, &second_kind);
                        if (ffn_pair) {
                            const size_t k_pair = static_cast<size_t>(first_weights->ne[0]);
                            const int status = rocknpu_matmul_q_pair_f32_f32_m1(
                                context->runtime,
                                static_cast<const uint8_t *>(first_weights->data),
                                ggml_nbytes(first_weights),
                                first_kind,
                                5632,
                                static_cast<const uint8_t *>(second_weights->data),
                                ggml_nbytes(second_weights),
                                second_kind,
                                5632,
                                static_cast<const float *>(first_activations->data),
                                static_cast<float *>(node->data),
                                static_cast<float *>(second->data),
                                k_pair);
                            if (status != ROCKNPU_STATUS_OK) {
                                return GGML_STATUS_FAILED;
                            }
                            context->ffn_pair_calls++;
                            context->w8a8_m1_mul_mat_calls += 2;
                            if (first_weights->type == GGML_TYPE_Q4_K) context->q4_k_mul_mat_calls++; else context->q6_k_mul_mat_calls++;
                            if (second_weights->type == GGML_TYPE_Q4_K) context->q4_k_mul_mat_calls++; else context->q6_k_mul_mat_calls++;
                            if (rocknpu_trace_enabled()) {
                                std::fprintf(stderr,
                                    "ROCKNPU GGML TRACE ffn_pair first=%s second=%s K=%zu N=5632+5632 kinds=%u/%u\n",
                                    first_weights->name, second_weights->name, k_pair, first_kind, second_kind);
                            }
                            i += 1;
                            break;
                        }
                    }
                }
                if (rocknpu_vk_pair_enabled() && i + 2 < graph->n_nodes) {
                    ggml_tensor * second = graph->nodes[i + 2];
                    if (second != nullptr && second->op == GGML_OP_MUL_MAT && rocknpu_mul_mat_supported(second)) {
                        const ggml_tensor * first_weights = node->src[0];
                        const ggml_tensor * second_weights = second->src[0];
                        const ggml_tensor * first_activations = node->src[1];
                        const ggml_tensor * second_activations = second->src[1];
                        uint32_t first_kind = 0;
                        uint32_t second_kind = 0;
                        const bool vk_pair =
                            rocknpu_weight_name_contains(first_weights, ".attn_v.weight") &&
                            rocknpu_weight_name_contains(second_weights, ".attn_k.weight") &&
                            first_activations == second_activations &&
                            first_activations->ne[1] == 1 &&
                            first_weights->ne[0] == second_weights->ne[0] &&
                            first_weights->ne[1] == 256 && second_weights->ne[1] == 256 &&
                            rocknpu_quant_kind(first_weights, &first_kind) &&
                            rocknpu_quant_kind(second_weights, &second_kind);
                        if (vk_pair) {
                            size_t vk_layer = 0;
                            rocknpu_qkv_layer_state * qkv_state = nullptr;
                            if (rocknpu_attention_layer(first_weights, ".attn_v.weight", &vk_layer)) {
                                qkv_state = &context->qkv[vk_layer];
                                const rocknpu_qkv_weight_ref current_v {
                                    static_cast<const uint8_t *>(first_weights->data), ggml_nbytes(first_weights), first_kind
                                };
                                const rocknpu_qkv_weight_ref current_k {
                                    static_cast<const uint8_t *>(second_weights->data), ggml_nbytes(second_weights), second_kind
                                };
                                const bool same_refs =
                                    qkv_state->v.data == current_v.data && qkv_state->v.bytes == current_v.bytes && qkv_state->v.kind == current_v.kind &&
                                    qkv_state->k.data == current_k.data && qkv_state->k.bytes == current_k.bytes && qkv_state->k.kind == current_k.kind;
                                qkv_state->v = current_v;
                                qkv_state->k = current_k;
                                qkv_state->stable_observations = same_refs ? qkv_state->stable_observations + 1 : 1;
                                if (rocknpu_qkv_triple_enabled() && qkv_state->pending) {
                                    const bool activation_matches =
                                        qkv_state->activation == static_cast<const float *>(first_activations->data);
                                    if (activation_matches) {
                                        std::memcpy(node->data, qkv_state->v_output, sizeof(qkv_state->v_output));
                                        std::memcpy(second->data, qkv_state->k_output, sizeof(qkv_state->k_output));
                                        qkv_state->pending = false;
                                        context->qkv_stash_hits++;
                                        if (rocknpu_trace_enabled()) {
                                            std::fprintf(stderr,
                                                "ROCKNPU GGML TRACE qkv_stash layer=%zu first=%s second=%s\n",
                                                vk_layer, first_weights->name, second_weights->name);
                                        }
                                        i += 2;
                                        break;
                                    }
                                    qkv_state->pending = false;
                                }
                            }
                            const size_t k_pair = static_cast<size_t>(first_weights->ne[0]);
                            const int status = rocknpu_matmul_q_pair_f32_f32_m1(
                                context->runtime,
                                static_cast<const uint8_t *>(first_weights->data),
                                ggml_nbytes(first_weights),
                                first_kind,
                                256,
                                static_cast<const uint8_t *>(second_weights->data),
                                ggml_nbytes(second_weights),
                                second_kind,
                                256,
                                static_cast<const float *>(first_activations->data),
                                static_cast<float *>(node->data),
                                static_cast<float *>(second->data),
                                k_pair);
                            if (status != ROCKNPU_STATUS_OK) {
                                return GGML_STATUS_FAILED;
                            }
                            context->vk_pair_calls++;
                            context->w8a8_m1_mul_mat_calls += 2;
                            if (first_weights->type == GGML_TYPE_Q4_K) context->q4_k_mul_mat_calls++; else context->q6_k_mul_mat_calls++;
                            if (second_weights->type == GGML_TYPE_Q4_K) context->q4_k_mul_mat_calls++; else context->q6_k_mul_mat_calls++;
                            if (rocknpu_trace_enabled()) {
                                std::fprintf(stderr,
                                    "ROCKNPU GGML TRACE vk_pair first=%s second=%s K=%zu N=256+256 kinds=%u/%u\n",
                                    first_weights->name, second_weights->name, k_pair, first_kind, second_kind);
                            }
                            i += 2;
                            break;
                        }
                    }
                }
                const ggml_tensor * weights = node->src[0];
                const ggml_tensor * activations = node->src[1];
                const size_t k = static_cast<size_t>(weights->ne[0]);
                const size_t n = static_cast<size_t>(weights->ne[1]);
                const size_t m = static_cast<size_t>(activations->ne[1]);
                const char * weight_type = "f16";
                if (weights->type == GGML_TYPE_Q4_K) {
                    context->q4_k_mul_mat_calls++;
                    weight_type = "q4_K";
                } else if (weights->type == GGML_TYPE_Q6_K) {
                    context->q6_k_mul_mat_calls++;
                    weight_type = "q6_K";
                } else {
                    context->f16_mul_mat_calls++;
                }
                const bool w4a4_m1 = m == 1 && weights->type == GGML_TYPE_Q4_K &&
                    k <= 10752 && n % 64 == 0 && n <= 8192 && rocknpu_w4a4_shape_enabled(k, n);
                if (w4a4_m1) {
                    context->w4a4_m1_mul_mat_calls++;
                } else if (m == 1 && (weights->type == GGML_TYPE_Q4_K || weights->type == GGML_TYPE_Q6_K)) {
                    context->w8a8_m1_mul_mat_calls++;
                }
                if (rocknpu_trace_enabled()) {
                    std::fprintf(stderr,
                        "ROCKNPU GGML TRACE mul_mat weight=%s type=%s M=%zu K=%zu N=%zu path=%s\n",
                        weights->name,
                        weight_type,
                        m,
                        k,
                        n,
                        m == 1 ? (w4a4_m1 ? "w4a4_m1" : "w8a8_m1") : "fp16_bridge");
                }
                int status;
                if (weights->type == GGML_TYPE_Q4_K) {
                    status = rocknpu_matmul_q4_k_f32_f32(
                        context->runtime,
                        static_cast<const uint8_t *>(weights->data),
                        ggml_nbytes(weights),
                        static_cast<const float *>(activations->data),
                        static_cast<float *>(node->data),
                        m,
                        k,
                        n);
                } else if (weights->type == GGML_TYPE_Q6_K) {
                    status = rocknpu_matmul_q6_k_f32_f32(
                        context->runtime,
                        static_cast<const uint8_t *>(weights->data),
                        ggml_nbytes(weights),
                        static_cast<const float *>(activations->data),
                        static_cast<float *>(node->data),
                        m,
                        k,
                        n);
                } else {
                    status = rocknpu_matmul_f16_f32_f32(
                        context->runtime,
                        static_cast<const uint16_t *>(weights->data),
                        static_cast<const float *>(activations->data),
                        static_cast<float *>(node->data),
                        m,
                        k,
                        n);
                }
                if (status != ROCKNPU_STATUS_OK) {
                    return GGML_STATUS_FAILED;
                }
                break;
            }
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            default:
                return GGML_STATUS_FAILED;
        }
    }

    return GGML_STATUS_SUCCESS;
}

const ggml_backend_i rocknpu_backend_iface = {
    /* .get_name            = */ rocknpu_backend_name,
    /* .free                = */ rocknpu_backend_free,
    /* .set_tensor_async    = */ nullptr,
    /* .get_tensor_async    = */ nullptr,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ nullptr,
    /* .synchronize         = */ nullptr,
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ rocknpu_backend_graph_compute,
    /* .event_record        = */ nullptr,
    /* .event_wait          = */ nullptr,
    /* .graph_optimize      = */ nullptr,
};

ggml_guid_t rocknpu_backend_guid() {
    static ggml_guid guid = {
        0x52, 0x4f, 0x43, 0x4b, 0x4e, 0x50, 0x55, 0x00,
        0x35, 0x38, 0x38, 0x00, 0x00, 0x00, 0x00, 0x01,
    };
    return &guid;
}

ggml_backend_t rocknpu_device_init(ggml_backend_dev_t dev, const char *) {
    rocknpu_context * runtime = rocknpu_context_create();
    if (runtime == nullptr) {
        return nullptr;
    }

    auto * context = new rocknpu_backend_context {};
    context->runtime = runtime;
    return new ggml_backend {
        /* .guid    = */ rocknpu_backend_guid(),
        /* .iface   = */ rocknpu_backend_iface,
        /* .device  = */ dev,
        /* .context = */ context,
    };
}

ggml_backend_buffer_type_t rocknpu_device_buffer_type(ggml_backend_dev_t) {
    return ggml_backend_cpu_buffer_type();
}

bool rocknpu_device_supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            return rocknpu_mul_mat_supported(op);
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

bool rocknpu_device_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);
}

const ggml_backend_device_i rocknpu_device_iface = {
    /* .get_name             = */ rocknpu_device_name,
    /* .get_description      = */ rocknpu_device_description,
    /* .get_memory           = */ rocknpu_device_memory,
    /* .get_type             = */ rocknpu_device_type,
    /* .get_props            = */ rocknpu_device_props,
    /* .init_backend         = */ rocknpu_device_init,
    /* .get_buffer_type      = */ rocknpu_device_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ rocknpu_device_supports_op,
    /* .supports_buft        = */ rocknpu_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

const char * rocknpu_reg_name(ggml_backend_reg_t) {
    return "ROCKNPU";
}

size_t rocknpu_reg_device_count(ggml_backend_reg_t) {
    return rocknpu_device_count() > 0 ? 1 : 0;
}

ggml_backend_dev_t rocknpu_reg_device(ggml_backend_reg_t reg, size_t index) {
    if (index != 0) {
        return nullptr;
    }

    static ggml_backend_device device = {
        /* .iface   = */ rocknpu_device_iface,
        /* .reg     = */ nullptr,
        /* .context = */ nullptr,
    };
    device.reg = reg;
    return &device;
}

void * rocknpu_reg_proc_address(ggml_backend_reg_t, const char *) {
    return nullptr;
}

const ggml_backend_reg_i rocknpu_reg_iface = {
    /* .get_name         = */ rocknpu_reg_name,
    /* .get_device_count = */ rocknpu_reg_device_count,
    /* .get_device       = */ rocknpu_reg_device,
    /* .get_proc_address = */ rocknpu_reg_proc_address,
};

int rocknpu_backend_score() {
    return rocknpu_device_count() > 0 ? 1 : 0;
}

} // namespace

ggml_backend_reg_t ggml_backend_rocknpu_reg() {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ rocknpu_reg_iface,
        /* .context     = */ nullptr,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_rocknpu_reg)
GGML_BACKEND_DL_SCORE_IMPL(rocknpu_backend_score)
