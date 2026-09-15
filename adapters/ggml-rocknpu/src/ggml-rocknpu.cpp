#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "rocknpu.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

struct rocknpu_backend_context {
    rocknpu_context * runtime;
    size_t q4_k_mul_mat_calls = 0;
    size_t q6_k_mul_mat_calls = 0;
    size_t f16_mul_mat_calls = 0;
    size_t w8a8_m1_mul_mat_calls = 0;
};

bool rocknpu_trace_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("ROCKNPU_GGML_TRACE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
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
            "ROCKNPU GGML TRACE summary q4_K_mul_mat=%zu q6_K_mul_mat=%zu f16_mul_mat=%zu w8a8_m1_mul_mat=%zu\n",
            context->q4_k_mul_mat_calls,
            context->q6_k_mul_mat_calls,
            context->f16_mul_mat_calls,
            context->w8a8_m1_mul_mat_calls);
        rocknpu_decode_cache_stats cache = {};
        if (rocknpu_context_decode_cache_stats(context->runtime, &cache) == ROCKNPU_STATUS_OK) {
            const double hit_ms = static_cast<double>(cache.hit_ns) / 1.0e6;
            const double miss_ms = static_cast<double>(cache.miss_ns) / 1.0e6;
            std::fprintf(stderr,
                "ROCKNPU GGML TRACE decode_cache hits=%zu misses=%zu entries=%zu resident_mb=%.2f hit_ms=%.2f hit_avg_ms=%.3f miss_ms=%.2f miss_avg_ms=%.3f\n",
                cache.hits,
                cache.misses,
                cache.entries,
                static_cast<double>(cache.resident_bytes) / (1024.0 * 1024.0),
                hit_ms,
                cache.hits == 0 ? 0.0 : hit_ms / static_cast<double>(cache.hits),
                miss_ms,
                cache.misses == 0 ? 0.0 : miss_ms / static_cast<double>(cache.misses));
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
                if (m == 1 && (weights->type == GGML_TYPE_Q4_K || weights->type == GGML_TYPE_Q6_K)) {
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
                        m == 1 ? "w8a8_m1" : "fp16_bridge");
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

    auto * context = new rocknpu_backend_context { runtime, 0, 0, 0, 0 };
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
