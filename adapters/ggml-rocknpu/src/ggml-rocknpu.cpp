#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "rocknpu.h"

#include <chrono>
#include <cstddef>
#include <thread>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr size_t ROCKNPU_QKV_MAX_LAYERS = 256;

struct rocknpu_qkv_weight_ref {
    const uint8_t * data = nullptr;
    size_t bytes = 0;
    uint32_t kind = 0;
    const ggml_tensor * tensor = nullptr;
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

struct rocknpu_w8_tensor {
    bool attempted = false;
    bool valid = false;
    size_t k = 0;
    size_t n = 0;
    std::vector<int8_t> weights;
    std::vector<float> scales;
};

struct rocknpu_shape_profile {
    size_t calls = 0;
    uint64_t ns = 0;
};

// Concurrent CPU/NPU N-split for M=1 projections: the NPU computes the first
// rows of a projection from the resident W8 weights while the frontend's own
// CPU backend computes the remaining rows from the original GGUF blocks on the
// calling thread, between NPU submission and completion wait.
struct rocknpu_backend_context {
    rocknpu_context * runtime;
    ggml_backend_t hybrid_cpu = nullptr;
    std::vector<uint8_t> hybrid_graph_mem;
    uint64_t hybrid_npu_ns = 0;
    uint64_t hybrid_cpu_ns = 0;
    size_t hybrid_calls = 0;
    ggml_backend_t cpu_fallback = nullptr;
    size_t cpu_fallback_calls = 0;
    size_t q4_k_mul_mat_calls = 0;
    size_t q6_k_mul_mat_calls = 0;
    size_t f16_mul_mat_calls = 0;
    size_t w4a4_m1_mul_mat_calls = 0;
    size_t w8a8_m1_mul_mat_calls = 0;
    size_t vk_pair_calls = 0;
    size_t ffn_pair_calls = 0;
    size_t qkv_triple_calls = 0;
    size_t qkv_stash_hits = 0;
    size_t mtile_group_calls = 0;
    size_t native_w8_calls = 0;
    size_t m4_calls = 0;
    size_t m8_calls = 0;
    size_t m12_calls = 0;
    size_t m16_calls = 0;
    size_t m_other_calls = 0;
    size_t graph_compute_calls = 0;
    size_t graph_compute_nodes = 0;
    uint64_t graph_compute_ns = 0;
    std::vector<uint64_t> graph_compute_samples_ns;
    std::unordered_map<std::string, rocknpu_shape_profile> shape_profile;
    std::unordered_map<std::string, rocknpu_w8_tensor> w8_sidecar;
    rocknpu_qkv_layer_state qkv[ROCKNPU_QKV_MAX_LAYERS] = {};
    bool ffn_composite_pending = false;
    std::vector<float> ffn_composite_activation;
    const rocknpu_w8_tensor * ffn_composite_gate = nullptr;
    const rocknpu_w8_tensor * ffn_composite_up = nullptr;
    size_t ffn_composite_k = 0;
    size_t ffn_composite_n = 0;
};

bool rocknpu_trace_enabled();

bool rocknpu_read_exact(const std::string & path, void * dst, size_t bytes) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || static_cast<size_t>(file.tellg()) != bytes) return false;
    file.seekg(0, std::ios::beg);
    file.read(static_cast<char *>(dst), static_cast<std::streamsize>(bytes));
    return file.good() || static_cast<size_t>(file.gcount()) == bytes;
}

uint64_t rocknpu_source_sample_fingerprint(const uint8_t * data, size_t bytes) {
    constexpr uint64_t offset_basis = 14695981039346656037ULL;
    constexpr uint64_t prime = 1099511628211ULL;
    uint64_t hash = offset_basis;
    const uint64_t byte_count = static_cast<uint64_t>(bytes);
    for (unsigned shift = 0; shift < 64; shift += 8) {
        hash ^= static_cast<uint8_t>(byte_count >> shift);
        hash *= prime;
    }
    const auto update = [&](size_t offset, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            hash ^= data[offset + i];
            hash *= prime;
        }
    };
    constexpr size_t sample_bytes = 4096;
    if (bytes <= sample_bytes * 3) {
        update(0, bytes);
    } else {
        update(0, sample_bytes);
        update((bytes - sample_bytes) / 2, sample_bytes);
        update(bytes - sample_bytes, sample_bytes);
    }
    return hash;
}

bool rocknpu_read_source_fingerprint(const std::string & path, uint64_t * value) {
    std::ifstream file(path);
    std::string encoded;
    if (!file || !(file >> encoded) || encoded.size() != 16) return false;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(encoded.c_str(), &end, 16);
    if (end != encoded.c_str() + encoded.size()) return false;
    *value = static_cast<uint64_t>(parsed);
    return true;
}

bool rocknpu_env_enabled_default(const char * name, bool default_value);

// Symmetric per-output-channel int8 from the GGUF source rows. Identical
// semantics to scripts/make_tinyllama_w8_sidecar_gguf.py (dequantize, scale =
// max|row|/127, round half away from zero), so a sidecar is only a cache.
bool rocknpu_w8_convert(rocknpu_w8_tensor & tensor, const uint8_t * source, size_t source_bytes,
                        ggml_type type, size_t k, size_t n) {
    if (source == nullptr || type >= GGML_TYPE_COUNT) return false;
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    if (type != GGML_TYPE_F32 && (traits == nullptr || traits->to_float == nullptr)) return false;
    const size_t row_bytes = ggml_row_size(type, static_cast<int64_t>(k));
    if (row_bytes * n != source_bytes) return false;
    tensor.weights.resize(k * n);
    tensor.scales.resize(n);
    const size_t workers = std::max<size_t>(1, std::min<size_t>(4, std::thread::hardware_concurrency()));
    const auto convert_rows = [&](size_t first, size_t last) {
        std::vector<float> row(k);
        for (size_t r = first; r < last; ++r) {
            const uint8_t * src = source + r * row_bytes;
            if (type == GGML_TYPE_F32) {
                std::memcpy(row.data(), src, k * sizeof(float));
            } else {
                traits->to_float(src, row.data(), static_cast<int64_t>(k));
            }
            float max_abs = 0.0f;
            for (float v : row) max_abs = std::max(max_abs, std::fabs(v));
            const float scale = max_abs == 0.0f ? 1.0f : max_abs / 127.0f;
            int8_t * dst = tensor.weights.data() + r * k;
            for (size_t i = 0; i < k; ++i) {
                const float q = std::round(row[i] / scale);
                dst[i] = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, q)));
            }
            tensor.scales[r] = scale;
        }
    };
    std::vector<std::thread> threads;
    const size_t per = (n + workers - 1) / workers;
    for (size_t w = 0; w < workers; ++w) {
        const size_t first = w * per;
        const size_t last = std::min(n, first + per);
        if (first < last) threads.emplace_back(convert_rows, first, last);
    }
    for (auto & thread : threads) thread.join();
    for (float scale : tensor.scales) {
        if (!std::isfinite(scale) || scale <= 0.0f) return false;
    }
    return true;
}

rocknpu_w8_tensor * rocknpu_w8_sidecar_get(
    rocknpu_backend_context * context, const char * name, size_t k, size_t n,
    const uint8_t * source_data, size_t source_bytes, ggml_type type = GGML_TYPE_COUNT) {
    if (name == nullptr || name[0] == '\0') return nullptr;
    const char * dir = std::getenv("ROCKNPU_W8_SIDECAR_DIR");
    const bool convert = rocknpu_env_enabled_default("ROCKNPU_W8_CONVERT", true);
    if ((dir == nullptr || dir[0] == '\0') && !convert) return nullptr;
    auto & tensor = context->w8_sidecar[std::string(name)];
    if (!tensor.attempted) {
        tensor.attempted = true;
        tensor.k = k;
        tensor.n = n;
        const size_t weight_count = k * n;
        if (dir != nullptr && dir[0] != '\0') {
            const std::string base = std::string(dir) + "/" + name;
            uint64_t expected_fingerprint = 0;
            const bool source_matches = source_data != nullptr && source_bytes != 0 &&
                rocknpu_read_source_fingerprint(base + ".source.fnv1a64", &expected_fingerprint) &&
                expected_fingerprint == rocknpu_source_sample_fingerprint(source_data, source_bytes);
            if (!source_matches) {
                if (rocknpu_trace_enabled()) {
                    std::fprintf(stderr, "ROCKNPU GGML TRACE native_w8_rejected_source weight=%s\n", name);
                }
            } else {
                tensor.weights.resize(weight_count);
                tensor.scales.resize(n);
                if (rocknpu_read_exact(base + ".w8", tensor.weights.data(), weight_count) &&
                    rocknpu_read_exact(base + ".scale.f32", tensor.scales.data(), n * sizeof(float))) {
                    tensor.valid = true;
                    for (float scale : tensor.scales) {
                        if (!std::isfinite(scale) || scale <= 0.0f) { tensor.valid = false; break; }
                    }
                }
            }
        }
        if (!tensor.valid && convert) {
            tensor.valid = rocknpu_w8_convert(tensor, source_data, source_bytes, type, k, n);
            if (tensor.valid && rocknpu_trace_enabled()) {
                std::fprintf(stderr, "ROCKNPU GGML TRACE native_w8_converted weight=%s K=%zu N=%zu\n", name, k, n);
            }
        }
        if (!tensor.valid) {
            tensor.weights.clear();
            tensor.scales.clear();
        } else if (rocknpu_trace_enabled()) {
            std::fprintf(stderr, "ROCKNPU GGML TRACE native_w8_loaded weight=%s K=%zu N=%zu\n", name, k, n);
        }
    }
    if (!tensor.valid || tensor.k != k || tensor.n != n) return nullptr;
    return &tensor;
}

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

bool rocknpu_w8_mtile_shape_enabled(size_t k, size_t n) {
    if (!rocknpu_env_enabled("ROCKNPU_W8_MTILE")) {
        return false;
    }
    const char * scope = std::getenv("ROCKNPU_W8_MTILE_SCOPE");
    if (scope == nullptr || scope[0] == '\0' || std::strcmp(scope, "all") == 0) {
        return true;
    }
    if (std::strcmp(scope, "kv") == 0) return k == 2048 && n == 256;
    if (std::strcmp(scope, "qo") == 0) return k == 2048 && n == 2048;
    if (std::strcmp(scope, "ffn") == 0) return (k == 2048 && n == 5632) || (k == 5632 && n == 2048);
    if (std::strcmp(scope, "attn") == 0) return k == 2048 && (n == 256 || n == 2048);
    if (std::strcmp(scope, "safe") == 0) return k == 2048 && (n == 256 || n == 5632);
    return false;
}

bool rocknpu_small_native_mtile_supported(size_t m, size_t k, size_t n) {
    if (!(m == 4 || m == 8 || m == 12) ||
        !rocknpu_env_enabled("ROCKNPU_NATIVE_MTILE") ||
        !rocknpu_env_enabled("ROCKNPU_MTILE_PERSIST") ||
        !rocknpu_w8_mtile_shape_enabled(k, n)) {
        return false;
    }
    if (k == 5632 && n == 2048) {
        return rocknpu_env_enabled("ROCKNPU_NATIVE_MTILE_DOWN") &&
               rocknpu_env_enabled("ROCKNPU_MTILE_MC");
    }
    return k > 0 && k <= 4096 && k % 512 == 0 && n > 0 && n <= 8192 && n % 32 == 0;
}

// Where single-token (M=1) decode projections run.
//   cpu    - llama.cpp's own CPU kernels (fastest single-stream generation on
//            LPDDR4X RK3588 boards: decode is memory-bound and Q4_K moves
//            fewer bytes than W8).
//   npu    - resident W8 weights on the NPU (frees the CPU cores).
//   hybrid - rows split between NPU (W8) and CPU (original GGUF) concurrently.
// Batched decode and prompt processing (M>1) always use the NPU.
enum class rocknpu_decode { cpu, npu, hybrid };

rocknpu_decode rocknpu_decode_mode() {
    static const rocknpu_decode mode = [] {
        const char * value = std::getenv("ROCKNPU_DECODE");
        if (value == nullptr || value[0] == '\0') return rocknpu_decode::cpu;
        if (std::strcmp(value, "npu") == 0 || std::strcmp(value, "1") == 0) return rocknpu_decode::npu;
        if (std::strcmp(value, "hybrid") == 0) return rocknpu_decode::hybrid;
        return rocknpu_decode::cpu;
    }();
    return mode;
}

double rocknpu_hybrid_share_env(const char * name, double fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    const double parsed = std::atof(value);
    return parsed > 0.0 && parsed < 1.0 ? parsed : 0.0;
}

// NPU share of the output rows for a concurrent CPU/NPU M=1 projection.
// 0 disables the hybrid split for that shape.
double rocknpu_hybrid_npu_share(size_t k, size_t n) {
    static const double global = rocknpu_hybrid_share_env(
        "ROCKNPU_HYBRID", rocknpu_decode_mode() == rocknpu_decode::hybrid ? 0.3 : 0.0);
    static const double qkv = rocknpu_hybrid_share_env("ROCKNPU_HYBRID_QKV", global);
    static const double out = rocknpu_hybrid_share_env("ROCKNPU_HYBRID_O", global);
    static const double gate_up = rocknpu_hybrid_share_env("ROCKNPU_HYBRID_GATEUP", global);
    static const double down = rocknpu_hybrid_share_env("ROCKNPU_HYBRID_DOWN", global);
    if (k == 5632) return down;
    if (n == 5632) return gate_up;
    if (n == 2048) return out;
    return qkv;
}

bool rocknpu_hybrid_enabled() {
    static const bool enabled = rocknpu_hybrid_share_env("ROCKNPU_HYBRID", 0.0) > 0.0;
    return enabled;
}

bool rocknpu_native_mtile_routes() {
    return rocknpu_env_enabled("ROCKNPU_NATIVE_MTILE") && rocknpu_env_enabled("ROCKNPU_MTILE_PERSIST") &&
           rocknpu_env_enabled("ROCKNPU_W8_MTILE") && rocknpu_env_enabled("ROCKNPU_MTILE_MC");
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

struct rocknpu_hybrid_item {
    const ggml_tensor * weights;
    const rocknpu_w8_tensor * w8;
    float * output;
    size_t n;
    size_t n_npu = 0;
};

bool rocknpu_hybrid_init(rocknpu_backend_context * context) {
    if (context->hybrid_cpu != nullptr) return true;
    context->hybrid_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (context->hybrid_cpu == nullptr) return false;
    int threads = 4;
    if (const char * value = std::getenv("ROCKNPU_HYBRID_THREADS")) {
        const int parsed = std::atoi(value);
        if (parsed > 0) threads = parsed;
    }
    ggml_backend_dev_t cpu_dev = ggml_backend_get_device(context->hybrid_cpu);
    ggml_backend_reg_t cpu_reg = cpu_dev != nullptr ? ggml_backend_dev_backend_reg(cpu_dev) : nullptr;
    if (cpu_reg != nullptr) {
        auto set_n_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_set_n_threads"));
        if (set_n_threads != nullptr) set_n_threads(context->hybrid_cpu, threads);
    }
    return true;
}

// Concurrent CPU/NPU execution of 1..3 M=1 projections that share one input.
// The NPU computes the first n_npu rows of every projection as one fused
// submission (single/pair/triple W8 path); the frontend CPU backend computes
// the remaining rows from the original GGUF blocks at the same time.
// Returns -1 when the group is not eligible so the caller keeps its NPU-only path.
int rocknpu_hybrid_group(
        rocknpu_backend_context * context, const ggml_tensor * activations, size_t k, double share,
        rocknpu_hybrid_item * items, size_t count) {
    if (share <= 0.0 || count == 0 || count > 3 || activations->type != GGML_TYPE_F32 ||
        activations->ne[1] != 1 || !ggml_is_contiguous(activations)) {
        return -1;
    }
    for (size_t i = 0; i < count; ++i) {
        if (items[i].w8 == nullptr) return -1;
        items[i].n_npu = static_cast<size_t>(static_cast<double>(items[i].n) * share) / 32 * 32;
        if (items[i].n_npu < 32 || items[i].n_npu >= items[i].n) return -1;
    }
    if (!rocknpu_hybrid_init(context)) return -1;
    const size_t mem = ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false) + 1024;
    if (context->hybrid_graph_mem.size() < mem) context->hybrid_graph_mem.resize(mem);
    ggml_init_params params = { mem, context->hybrid_graph_mem.data(), true };
    ggml_context * graph_ctx = ggml_init(params);
    if (graph_ctx == nullptr) return -1;
    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx, 16, false);
    ggml_tensor * a = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, static_cast<int64_t>(k), 1);
    a->data = activations->data;
    for (size_t i = 0; i < count; ++i) {
        const ggml_tensor * weights = items[i].weights;
        const size_t n_cpu = items[i].n - items[i].n_npu;
        ggml_tensor * w = ggml_new_tensor_2d(graph_ctx, weights->type, static_cast<int64_t>(k), static_cast<int64_t>(n_cpu));
        w->data = static_cast<char *>(weights->data) + items[i].n_npu * weights->nb[1];
        ggml_tensor * d = ggml_mul_mat(graph_ctx, w, a);
        d->data = items[i].output + items[i].n_npu;
        ggml_build_forward_expand(graph, d);
    }

    struct overlap_state {
        ggml_backend_t backend;
        ggml_cgraph * graph;
        ggml_status status;
        bool ran;
        uint64_t ns;
    } overlap { context->hybrid_cpu, graph, GGML_STATUS_FAILED, false, 0 };
    const auto run_cpu = [](void * user_data) {
        auto * state = static_cast<overlap_state *>(user_data);
        const auto started = std::chrono::steady_clock::now();
        state->status = ggml_backend_graph_compute(state->backend, state->graph);
        state->ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count());
        state->ran = true;
    };
    const float * input = static_cast<const float *>(activations->data);
    const auto started = std::chrono::steady_clock::now();
    rocknpu_context_set_overlap(context->runtime, run_cpu, &overlap);
    int npu_status;
    const rocknpu_hybrid_item & x = items[0];
    if (count == 1) {
        npu_status = rocknpu_matmul_w8a8_f32_f32_m1(
            context->runtime, x.w8->weights.data(), x.w8->scales.data(), input, x.output, k, x.n_npu);
    } else if (count == 2) {
        const rocknpu_hybrid_item & y = items[1];
        npu_status = rocknpu_matmul_w8a8_pair_f32_f32_m1(
            context->runtime,
            x.w8->weights.data(), x.w8->scales.data(), x.n_npu,
            y.w8->weights.data(), y.w8->scales.data(), y.n_npu,
            input, x.output, y.output, k);
    } else {
        const rocknpu_hybrid_item & y = items[1];
        const rocknpu_hybrid_item & z = items[2];
        npu_status = rocknpu_matmul_w8a8_triple_f32_f32_m1(
            context->runtime,
            x.w8->weights.data(), x.w8->scales.data(), x.n_npu,
            y.w8->weights.data(), y.w8->scales.data(), y.n_npu,
            z.w8->weights.data(), z.w8->scales.data(), z.n_npu,
            input, x.output, y.output, z.output, k);
    }
    rocknpu_context_set_overlap(context->runtime, nullptr, nullptr);
    if (!overlap.ran) run_cpu(&overlap);
    const uint64_t total_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count());
    context->hybrid_cpu_ns += overlap.ns;
    context->hybrid_npu_ns += total_ns;
    context->hybrid_calls++;
    ggml_free(graph_ctx);
    if (overlap.status != GGML_STATUS_SUCCESS) return ROCKNPU_STATUS_INVALID_ARGUMENT;
    return npu_status;
}

int rocknpu_hybrid_m1(
        rocknpu_backend_context * context, const ggml_tensor * weights, const ggml_tensor * activations,
        ggml_tensor * node, const rocknpu_w8_tensor * w8, size_t k, size_t n) {
    rocknpu_hybrid_item item { weights, w8, static_cast<float *>(node->data), n };
    return rocknpu_hybrid_group(context, activations, k, rocknpu_hybrid_npu_share(k, n), &item, 1);
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
    // RK3588 NPU allocations live in shared system memory rather than dedicated
    // VRAM. Report Linux MemAvailable/MemTotal so frontends that require a
    // non-zero accelerator memory budget (for example Ollama) can schedule the
    // device without pretending that this is discrete VRAM.
    size_t available_kib = 0;
    size_t total_kib = 0;
    size_t free_kib = 0;

    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    size_t value = 0;
    std::string unit;
    while (meminfo >> key >> value >> unit) {
        if (key == "MemTotal:") {
            total_kib = value;
        } else if (key == "MemAvailable:") {
            available_kib = value;
        } else if (key == "MemFree:") {
            free_kib = value;
        }
    }

    if (available_kib == 0) {
        available_kib = free_kib;
    }
    *free = available_kib * 1024;
    *total = total_kib * 1024;
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
        /* .host_buffer          = */ true,
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
    // The native W8 M-tile path is validated only up to N=8192. In particular,
    // TinyLlama's N=32000 vocabulary head is dramatically slower through the
    // generic FP16 bridge, so leave it on the optimized CPU backend.
    if (quantized && m > 1 && n > 8192) {
        return false;
    }
    // Large quantized prompt batches are dramatically slower through the
    // uncached FP16 bridge.  The persistent NPU prefill path is an explicit
    // opt-in; otherwise leave M>128 prompt work on the optimized CPU backend.
    if (quantized && m > 128 && !rocknpu_env_enabled("ROCKNPU_PREFILL_CACHE")) {
        return false;
    }
    if (quantized && m > 1 && rocknpu_env_enabled("ROCKNPU_SCHED_FFN_ONLY")) {
        const bool gate_up = k == 2048 && n == 5632;
        const bool down = k == 5632 && n == 2048;
        const char * part = std::getenv("ROCKNPU_SCHED_FFN_PART");
        const bool ffn = part != nullptr && std::strcmp(part, "gateup") == 0 ? gate_up
            : part != nullptr && std::strcmp(part, "down") == 0 ? down
            : gate_up || down;
        if (!ffn) {
            return false;
        }
    }
    if (rocknpu_env_enabled("ROCKNPU_SCHED_CPU_QO") && quantized && m == 16 && k == 2048 && n == 2048) {
        return false;
    }
    if (rocknpu_env_enabled("ROCKNPU_M16_ONLY") && m != 16) {
        return false;
    }
    if (m == 1) {
        // Let optimized CPU kernels handle decode when NPU acceleration is
        // useful only for prompt processing. Decide before graph assignment.
        if (rocknpu_decode_mode() == rocknpu_decode::cpu) {
            return false;
        }
        if (quantized && n > 8192) {
            return rocknpu_env_enabled("ROCKNPU_NPU_OUTPUT_HEAD") &&
                k > 0 && k % 512 == 0 && n % 32 == 0 && n <= 32768;
        }
        if (rocknpu_env_enabled("ROCKNPU_SCHED_CPU_QO") && k == 2048 && n == 2048) {
            return false;
        }
        if (rocknpu_env_enabled("ROCKNPU_SCHED_CPU_KV") && k == 2048 && n == 256) {
            return false;
        }
        if (rocknpu_env_enabled("ROCKNPU_SCHED_CPU_GATEUP") && k == 2048 && n == 5632) {
            return false;
        }
        if (rocknpu_env_enabled("ROCKNPU_SCHED_CPU_FFN") &&
            ((k == 2048 && n == 5632) || (k == 5632 && n == 2048))) {
            return false;
        }
        if (rocknpu_env_enabled("ROCKNPU_SCHED_CPU_DOWN") && k == 5632 && n == 2048) {
            return false;
        }
        return quantized && k > 0 && n > 0 && k % 512 == 0 && n % 32 == 0 && n <= 8192;
    }
    if (quantized && rocknpu_native_mtile_routes()) {
        // Any batch size runs as native W8A8 M-tiles (the runtime tiles and
        // pads rows); the full-K tile needs K%512 and N within one tile.
        return k % 512 == 0 && n % 32 == 0 && n <= 8192;
    }
    if (quantized && (m == 4 || m == 8 || m == 12)) {
        // Small-M decode is only useful through the validated persistent W8 path.
        // Never assign these graphs to RockNPU merely to fall back to the FP16 bridge.
        return rocknpu_small_native_mtile_supported(
            static_cast<size_t>(m), static_cast<size_t>(k), static_cast<size_t>(n));
    }
    const int64_t k_alignment = quantized ? 256 : 32;
    return m > 0 && k > 0 && n > 0 && m % 4 == 0 && k % k_alignment == 0 && n % 16 == 0;
}

const char * rocknpu_backend_name(ggml_backend_t) {
    return "ROCKNPU";
}

void rocknpu_backend_free(ggml_backend_t backend) {
    auto * context = static_cast<rocknpu_backend_context *>(backend->context);
    if (rocknpu_trace_enabled() || rocknpu_env_enabled("ROCKNPU_DISPATCH_SUMMARY")) {
        std::fprintf(stderr,
            "ROCKNPU GGML TRACE summary q4_K_mul_mat=%zu q6_K_mul_mat=%zu f16_mul_mat=%zu w4a4_m1_mul_mat=%zu w8a8_m1_mul_mat=%zu native_w8_calls=%zu vk_pair_calls=%zu ffn_pair_calls=%zu qkv_triple_calls=%zu qkv_stash_hits=%zu mtile_group_calls=%zu\n",
            context->q4_k_mul_mat_calls,
            context->q6_k_mul_mat_calls,
            context->f16_mul_mat_calls,
            context->w4a4_m1_mul_mat_calls,
            context->w8a8_m1_mul_mat_calls,
            context->native_w8_calls,
            context->vk_pair_calls,
            context->ffn_pair_calls,
            context->qkv_triple_calls,
            context->qkv_stash_hits,
            context->mtile_group_calls);
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
    if (rocknpu_env_enabled("ROCKNPU_SPLIT_PROFILE")) {
        const double graph_ms = static_cast<double>(context->graph_compute_ns) / 1.0e6;
        size_t slow_5ms = 0;
        size_t slow_20ms = 0;
        size_t last_slow_5ms = 0;
        uint64_t max_ns = 0;
        uint64_t quarter_ns[4] = {};
        size_t quarter_slow20[4] = {};
        uint64_t first_batch_ns = 0;
        uint64_t warm_ns = 0;
        const size_t sample_count = context->graph_compute_samples_ns.size();
        for (size_t i = 0; i < sample_count; ++i) {
            const uint64_t ns = context->graph_compute_samples_ns[i];
            max_ns = std::max(max_ns, ns);
            if (ns >= 5'000'000) {
                ++slow_5ms;
                last_slow_5ms = i + 1;
            }
            slow_20ms += size_t(ns >= 20'000'000);
            const size_t quarter = sample_count == 0 ? 0 : std::min<size_t>(3, (i * 4) / sample_count);
            quarter_ns[quarter] += ns;
            quarter_slow20[quarter] += size_t(ns >= 20'000'000);
            if (i < 111) first_batch_ns += ns; else warm_ns += ns;
        }
        for (const auto & entry : context->shape_profile) {
            const auto & profile = entry.second;
            std::fprintf(stderr,
                "ROCKNPU SHAPE PROFILE %s calls=%zu total_ms=%.3f avg_ms=%.3f\n",
                entry.first.c_str(), profile.calls,
                static_cast<double>(profile.ns) / 1.0e6,
                profile.calls == 0 ? 0.0 : static_cast<double>(profile.ns) / 1.0e6 / static_cast<double>(profile.calls));
        }
        std::fprintf(stderr,
            "ROCKNPU SPLIT PROFILE graph_calls=%zu graph_nodes=%zu graph_ms=%.3f avg_call_ms=%.3f avg_nodes=%.2f cpu_fallback=%zu q4=%zu q6=%zu f16=%zu native_w8=%zu m=[4:%zu,8:%zu,12:%zu,16:%zu,other:%zu] slow5=%zu slow20=%zu last_slow5=%zu max_ms=%.3f first111_ms=%.3f rest_ms=%.3f q_ms=[%.3f,%.3f,%.3f,%.3f] q_slow20=[%zu,%zu,%zu,%zu]\n",
            context->graph_compute_calls,
            context->graph_compute_nodes,
            graph_ms,
            context->graph_compute_calls == 0 ? 0.0 : graph_ms / static_cast<double>(context->graph_compute_calls),
            context->graph_compute_calls == 0 ? 0.0 : static_cast<double>(context->graph_compute_nodes) / static_cast<double>(context->graph_compute_calls),
            context->cpu_fallback_calls,
            context->q4_k_mul_mat_calls,
            context->q6_k_mul_mat_calls,
            context->f16_mul_mat_calls,
            context->native_w8_calls,
            context->m4_calls,
            context->m8_calls,
            context->m12_calls,
            context->m16_calls,
            context->m_other_calls,
            slow_5ms,
            slow_20ms,
            last_slow_5ms,
            static_cast<double>(max_ns) / 1.0e6,
            static_cast<double>(first_batch_ns) / 1.0e6,
            static_cast<double>(warm_ns) / 1.0e6,
            static_cast<double>(quarter_ns[0]) / 1.0e6,
            static_cast<double>(quarter_ns[1]) / 1.0e6,
            static_cast<double>(quarter_ns[2]) / 1.0e6,
            static_cast<double>(quarter_ns[3]) / 1.0e6,
            quarter_slow20[0], quarter_slow20[1], quarter_slow20[2], quarter_slow20[3]);
    }
    if (context->hybrid_calls > 0 && (rocknpu_trace_enabled() || rocknpu_env_enabled("ROCKNPU_DISPATCH_SUMMARY"))) {
        std::fprintf(stderr,
            "ROCKNPU GGML TRACE hybrid calls=%zu total_avg_ms=%.3f cpu_avg_ms=%.3f\n",
            context->hybrid_calls,
            static_cast<double>(context->hybrid_npu_ns) / 1.0e6 / static_cast<double>(context->hybrid_calls),
            static_cast<double>(context->hybrid_cpu_ns) / 1.0e6 / static_cast<double>(context->hybrid_calls));
    }
    if (context->hybrid_cpu != nullptr) {
        ggml_backend_free(context->hybrid_cpu);
    }
    if (context->cpu_fallback != nullptr) {
        ggml_backend_free(context->cpu_fallback);
    }
    rocknpu_context_destroy(context->runtime);
    delete context;
    delete backend;
}

// Prefill-sized quantized projection eligible for same-input grouping.
bool rocknpu_mtile_group_m1_candidate(const ggml_tensor * node) {
    static const bool enabled = rocknpu_env_enabled_default("ROCKNPU_MTILE_GROUP", true);
    if (!enabled || node == nullptr || node->op != GGML_OP_MUL_MAT) return false;
    const ggml_tensor * w = node->src[0];
    const ggml_tensor * a = node->src[1];
    if (w == nullptr || a == nullptr || (w->type != GGML_TYPE_Q4_K && w->type != GGML_TYPE_Q6_K)) return false;
    const int64_t m = a->ne[1];
    const int64_t k = w->ne[0];
    // Larger batches are tiled into <=128-row NPU tiles by the runtime.
    return m >= 32 && a->type == GGML_TYPE_F32 &&
           node->type == GGML_TYPE_F32 && ggml_is_contiguous(a) && ggml_is_contiguous(node) &&
           a->ne[2] == 1 && a->ne[3] == 1 && k % 512 == 0 && k <= (m > 64 ? 2048 : 4096) &&
           w->ne[1] % 32 == 0;
}

// Run the projection at graph node `first` together with the later
// projections of this split that read the same activations, as one
// concatenated-N NPU M-tile call. Returns how many projections ran (0 = none).
int rocknpu_mtile_group(rocknpu_backend_context * context, ggml_cgraph * graph, int first,
                        std::vector<const ggml_tensor *> & done) {
    ggml_tensor * lead = graph->nodes[first];
    const ggml_tensor * activations = lead->src[1];
    ggml_tensor * members[4] = { lead };
    int count = 1;
    for (int j = first + 1; j < graph->n_nodes && count < 4; ++j) {
        ggml_tensor * candidate = graph->nodes[j];
        if (candidate == nullptr || (candidate->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) continue;
        if (candidate->op == GGML_OP_MUL_MAT && candidate->src[1] == activations &&
            candidate->src[0]->ne[0] == lead->src[0]->ne[0] &&
            rocknpu_mtile_group_m1_candidate(candidate) && rocknpu_mul_mat_supported(candidate)) {
            members[count++] = candidate;
        }
    }
    int64_t total_n = 0;
    for (int g = 0; g < count; ++g) total_n += members[g]->src[0]->ne[1];
    // A single wide projection keeps its dedicated (K-split capable) path.
    if (count == 1) return 0;
    if (total_n > 3 * 8192) return 0;
    const uint8_t * weights[4];
    size_t bytes[4];
    uint32_t kinds[4];
    size_t ns[4];
    float * outputs[4];
    for (int g = 0; g < count; ++g) {
        const ggml_tensor * w = members[g]->src[0];
        weights[g] = static_cast<const uint8_t *>(w->data);
        bytes[g] = ggml_nbytes(w);
        kinds[g] = w->type == GGML_TYPE_Q4_K ? 4 : 6;
        ns[g] = static_cast<size_t>(w->ne[1]);
        outputs[g] = static_cast<float *>(members[g]->data);
    }
    const size_t m = static_cast<size_t>(activations->ne[1]);
    const size_t k = static_cast<size_t>(lead->src[0]->ne[0]);
    const int status = rocknpu_matmul_q_concat_f32_f32_mtile(
        context->runtime, static_cast<size_t>(count), weights, bytes, kinds, ns,
        static_cast<const float *>(activations->data), outputs, m, k);
    if (status != ROCKNPU_STATUS_OK) return 0;
    for (int g = 0; g < count; ++g) {
        const ggml_tensor * w = members[g]->src[0];
        if (w->type == GGML_TYPE_Q4_K) context->q4_k_mul_mat_calls++; else context->q6_k_mul_mat_calls++;
        if (g > 0) done.push_back(members[g]);
    }
    context->m_other_calls += static_cast<size_t>(count);
    context->mtile_group_calls++;
    if (rocknpu_trace_enabled()) {
        std::fprintf(stderr, "ROCKNPU GGML TRACE mtile_group lead=%s count=%d M=%zu K=%zu N=%lld\n",
            lead->src[0]->name, count, m, k, static_cast<long long>(total_n));
    }
    return count;
}

enum ggml_status rocknpu_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto * context = static_cast<rocknpu_backend_context *>(backend->context);
    if (context->cpu_fallback != nullptr) {
        bool has_mul_mat = false;
        bool all_m16 = true;
        bool all_w8_m16 = true;
        const bool split_m32 = rocknpu_env_enabled("ROCKNPU_M32_AS_2X16");
        const bool chunk_m16 = rocknpu_env_enabled("ROCKNPU_M16_CHUNK_BATCH");
        const bool native_mtile = rocknpu_env_enabled("ROCKNPU_NATIVE_MTILE");
        const bool prewarm_mtile = rocknpu_env_enabled("ROCKNPU_RUNTIME_M16_PREWARM");
        std::vector<const ggml_tensor *> prewarm_weights;
        bool force_cpu_shape = false;
        const bool cpu_kv = rocknpu_env_enabled("ROCKNPU_RUNTIME_CPU_KV");
        const bool cpu_qo = rocknpu_env_enabled("ROCKNPU_RUNTIME_CPU_QO");
        const bool cpu_ffn = rocknpu_env_enabled("ROCKNPU_RUNTIME_CPU_FFN");
        for (int i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * node = graph->nodes[i];
            if (node == nullptr || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || node->op != GGML_OP_MUL_MAT) {
                continue;
            }
            has_mul_mat = true;
            const ggml_tensor * weights = node->src[0];
            const ggml_tensor * activations = node->src[1];
            const bool quantized = weights != nullptr &&
                (weights->type == GGML_TYPE_Q4_K || weights->type == GGML_TYPE_Q6_K);
            if (prewarm_mtile && quantized) {
                const size_t k = static_cast<size_t>(weights->ne[0]);
                const size_t n = static_cast<size_t>(weights->ne[1]);
                const bool prewarm_shape =
                    (k == 2048 && (n == 256 || n == 2048 || n == 5632)) ||
                    (k == 5632 && n == 2048);
                if (prewarm_shape) {
                    prewarm_weights.push_back(weights);
                }
            }
            const bool native_small_batch = quantized && activations != nullptr &&
                rocknpu_small_native_mtile_supported(
                    static_cast<size_t>(activations->ne[1]),
                    static_cast<size_t>(weights->ne[0]),
                    static_cast<size_t>(weights->ne[1]));
            const bool native_batch = native_small_batch ||
                (native_mtile && quantized && activations != nullptr &&
                 activations->ne[1] >= 32 && activations->ne[1] <= 128 && activations->ne[1] % 16 == 0);
            const bool supported_batch = activations != nullptr &&
                (activations->ne[1] == 16 || native_batch ||
                 (split_m32 && activations->ne[1] == 32) ||
                 (chunk_m16 && activations->ne[1] >= 32 && activations->ne[1] <= 128 &&
                  activations->ne[1] % 16 == 0));
            if (!supported_batch) {
                all_m16 = false;
            }
            const bool w8_shape = quantized && activations != nullptr &&
                (activations->ne[1] == 16 || native_batch) &&
                weights->ne[0] > 0 && weights->ne[0] <= 4096 && weights->ne[0] % 512 == 0 &&
                weights->ne[1] > 0 && weights->ne[1] <= 8192 && weights->ne[1] % 32 == 0;
            all_w8_m16 = all_w8_m16 && w8_shape;
            const bool routed_batch = activations != nullptr &&
                (activations->ne[1] == 16 || native_batch ||
                 (split_m32 && activations->ne[1] == 32) ||
                 (chunk_m16 && activations->ne[1] >= 32 && activations->ne[1] <= 128 &&
                  activations->ne[1] % 16 == 0));
            if (quantized && routed_batch && weights->ne[0] == 2048) {
                force_cpu_shape = force_cpu_shape || (cpu_kv && weights->ne[1] == 256);
                force_cpu_shape = force_cpu_shape || (cpu_qo && weights->ne[1] == 2048);
                force_cpu_shape = force_cpu_shape || (cpu_ffn && weights->ne[1] == 5632);
            }
        }
        const bool route_w8_only = rocknpu_env_enabled("ROCKNPU_RUNTIME_M16_W8_ONLY");
        if (has_mul_mat && (!all_m16 || (route_w8_only && !all_w8_m16) || force_cpu_shape)) {
            if (prewarm_mtile) {
                for (const ggml_tensor * weights : prewarm_weights) {
                    uint32_t kind = 0;
                    if (!rocknpu_quant_kind(weights, &kind)) {
                        continue;
                    }
                    const int status = rocknpu_prewarm_quantized_m16(
                        context->runtime,
                        static_cast<const uint8_t *>(weights->data),
                        ggml_nbytes(weights),
                        kind,
                        static_cast<size_t>(weights->ne[0]),
                        static_cast<size_t>(weights->ne[1]));
                    if (status != ROCKNPU_STATUS_OK) {
                        return GGML_STATUS_FAILED;
                    }
                }
            }
            context->cpu_fallback_calls++;
            return ggml_backend_graph_compute(context->cpu_fallback, graph);
        }
    }
    struct graph_compute_timer {
        rocknpu_backend_context * context;
        std::chrono::steady_clock::time_point start;
        ~graph_compute_timer() {
            const auto elapsed = std::chrono::steady_clock::now() - start;
            const uint64_t elapsed_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
            context->graph_compute_calls++;
            context->graph_compute_ns += elapsed_ns;
            if (rocknpu_env_enabled("ROCKNPU_SPLIT_PROFILE")) {
                context->graph_compute_samples_ns.push_back(elapsed_ns);
            }
        }
    } timer { context, std::chrono::steady_clock::now() };
    context->graph_compute_nodes += static_cast<size_t>(graph->n_nodes);
    // Projections already computed by a same-input group earlier in this split.
    std::vector<const ggml_tensor *> grouped_done;
    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (!grouped_done.empty() &&
            std::find(grouped_done.begin(), grouped_done.end(), node) != grouped_done.end()) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT: {
                if (!rocknpu_mul_mat_supported(node)) {
                    return GGML_STATUS_FAILED;
                }
                if (rocknpu_mtile_group_m1_candidate(node)) {
                    const int grouped = rocknpu_mtile_group(context, graph, i, grouped_done);
                    if (grouped > 0) break;
                }
                if (rocknpu_env_enabled("ROCKNPU_NPU_OUTPUT_HEAD")) {
                    const ggml_tensor * output_weights = node->src[0];
                    const ggml_tensor * output_activations = node->src[1];
                    const size_t output_m = static_cast<size_t>(output_activations->ne[1]);
                    const size_t output_k = static_cast<size_t>(output_weights->ne[0]);
                    const size_t output_n = static_cast<size_t>(output_weights->ne[1]);
                    if (output_m == 1 && output_n > 8192 &&
                        rocknpu_weight_name_contains(output_weights, "output.weight") &&
                        (output_weights->type == GGML_TYPE_Q4_K || output_weights->type == GGML_TYPE_Q6_K)) {
                        rocknpu_w8_tensor * output_w8 = rocknpu_w8_sidecar_get(
                            context, output_weights->name, output_k, output_n,
                            static_cast<const uint8_t *>(output_weights->data), ggml_nbytes(output_weights), output_weights->type);
                        if (output_w8 == nullptr) {
                            return GGML_STATUS_FAILED;
                        }
                        const int status = rocknpu_matmul_w8a8_f32_f32_m1_nsplit(
                            context->runtime,
                            output_w8->weights.data(), output_w8->scales.data(),
                            static_cast<const float *>(output_activations->data),
                            static_cast<float *>(node->data), output_k, output_n, 8192);
                        if (status != ROCKNPU_STATUS_OK) {
                            return GGML_STATUS_FAILED;
                        }
                        const size_t chunks = (output_n + 8191) / 8192;
                        context->w8a8_m1_mul_mat_calls++;
                        context->native_w8_calls += chunks;
                        if (output_weights->type == GGML_TYPE_Q4_K) context->q4_k_mul_mat_calls++; else context->q6_k_mul_mat_calls++;
                        if (rocknpu_trace_enabled()) {
                            std::fprintf(stderr,
                                "ROCKNPU GGML TRACE output_head_split name=%s type=%s K=%zu N=%zu chunks=%zu path=w8a8_m1_nsplit\n",
                                output_weights->name,
                                output_weights->type == GGML_TYPE_Q4_K ? "q4_K" : "q6_K",
                                output_k, output_n, chunks);
                        }
                        break;
                    }
                }
                if (rocknpu_env_enabled("ROCKNPU_FFN_COMPOSITE") && context->ffn_composite_pending) {
                    const ggml_tensor * down_weights = node->src[0];
                    const ggml_tensor * down_activations = node->src[1];
                    const size_t down_m = static_cast<size_t>(down_activations->ne[1]);
                    const size_t down_k = static_cast<size_t>(down_weights->ne[0]);
                    const size_t down_n = static_cast<size_t>(down_weights->ne[1]);
                    if (down_m == 1 && down_k == context->ffn_composite_n &&
                        down_n == 2048 && rocknpu_weight_name_contains(down_weights, ".ffn_down.weight") &&
                        (down_weights->type == GGML_TYPE_Q4_K || down_weights->type == GGML_TYPE_Q6_K) &&
                        context->ffn_composite_gate != nullptr && context->ffn_composite_up != nullptr) {
                        rocknpu_w8_tensor * down_w8 = rocknpu_w8_sidecar_get(
                            context, down_weights->name, down_k, down_n,
                            static_cast<const uint8_t *>(down_weights->data), ggml_nbytes(down_weights), down_weights->type);
                        if (down_w8 != nullptr) {
                            const rocknpu_w8_tensor * gate_w8 = context->ffn_composite_gate;
                            const rocknpu_w8_tensor * up_w8 = context->ffn_composite_up;
                            const int status = rocknpu_matmul_w8a8_swiglu_down_f32(
                                context->runtime,
                                gate_w8->weights.data(), gate_w8->scales.data(),
                                up_w8->weights.data(), up_w8->scales.data(),
                                down_w8->weights.data(), down_w8->scales.data(),
                                context->ffn_composite_activation.data(), static_cast<float *>(node->data),
                                context->ffn_composite_k, context->ffn_composite_n, down_n);
                            if (status != ROCKNPU_STATUS_OK) {
                                return GGML_STATUS_FAILED;
                            }
                            context->ffn_composite_pending = false;
                            context->ffn_pair_calls++;
                            context->w8a8_m1_mul_mat_calls += 3;
                            context->native_w8_calls += 3;
                            context->q4_k_mul_mat_calls += 2;
                            context->q6_k_mul_mat_calls += 1;
                            if (rocknpu_trace_enabled()) {
                                std::fprintf(stderr,
                                    "ROCKNPU GGML TRACE ffn_composite K=%zu N=%zu down_N=%zu path=w8a8_swiglu_down\n",
                                    context->ffn_composite_k, context->ffn_composite_n, down_n);
                            }
                            break;
                        }
                    }
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
                            const std::string v_name = "blk." + std::to_string(q_layer) + ".attn_v.weight";
                            const std::string k_name = "blk." + std::to_string(q_layer) + ".attn_k.weight";
                            rocknpu_w8_tensor * q_w8 = rocknpu_w8_sidecar_get(
                                context, q_weights->name, 2048, 2048,
                                static_cast<const uint8_t *>(q_weights->data), ggml_nbytes(q_weights), q_weights->type);
                            rocknpu_w8_tensor * v_w8 = rocknpu_w8_sidecar_get(
                                context, v_name.c_str(), 2048, 256, state.v.data, state.v.bytes, state.v.tensor != nullptr ? state.v.tensor->type : GGML_TYPE_COUNT);
                            rocknpu_w8_tensor * k_w8 = rocknpu_w8_sidecar_get(
                                context, k_name.c_str(), 2048, 256, state.k.data, state.k.bytes, state.k.tensor != nullptr ? state.k.tensor->type : GGML_TYPE_COUNT);
                            const bool native_w8 = q_w8 != nullptr && v_w8 != nullptr && k_w8 != nullptr;
                            const ggml_tensor * v_tensor = state.v.tensor;
                            const ggml_tensor * k_tensor = state.k.tensor;
                            rocknpu_hybrid_item qkv_items[3] = {
                                { q_weights, q_w8, static_cast<float *>(node->data), 2048 },
                                { v_tensor, v_w8, state.v_output, 256 },
                                { k_tensor, k_w8, state.k_output, 256 },
                            };
                            const int hybrid_status = native_w8 && v_tensor != nullptr && k_tensor != nullptr
                                ? rocknpu_hybrid_group(context, q_activations, 2048,
                                    rocknpu_hybrid_npu_share(2048, 2560), qkv_items, 3)
                                : -1;
                            const int status = hybrid_status != -1 ? hybrid_status : native_w8 ? rocknpu_matmul_w8a8_triple_f32_f32_m1(
                                context->runtime,
                                q_w8->weights.data(), q_w8->scales.data(), 2048,
                                v_w8->weights.data(), v_w8->scales.data(), 256,
                                k_w8->weights.data(), k_w8->scales.data(), 256,
                                static_cast<const float *>(q_activations->data),
                                static_cast<float *>(node->data), state.v_output, state.k_output, 2048)
                                : rocknpu_matmul_q_triple_f32_f32_m1(
                                context->runtime,
                                static_cast<const uint8_t *>(q_weights->data), ggml_nbytes(q_weights), q_kind, 2048,
                                state.v.data, state.v.bytes, state.v.kind, 256,
                                state.k.data, state.k.bytes, state.k.kind, 256,
                                static_cast<const float *>(q_activations->data),
                                static_cast<float *>(node->data), state.v_output, state.k_output, 2048);
                            if (status == ROCKNPU_STATUS_OK) {
                                state.activation = static_cast<const float *>(q_activations->data);
                                state.pending = true;
                                context->qkv_triple_calls++;
                                context->w8a8_m1_mul_mat_calls += 3;
                                if (native_w8) context->native_w8_calls += 3;
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
                        const size_t pair_m = static_cast<size_t>(first_activations->ne[1]);
                        const bool native_mtile_pair =
                            (pair_m == 4 || pair_m == 8 || pair_m == 12 || pair_m == 16) &&
                            rocknpu_env_enabled("ROCKNPU_NATIVE_MTILE") &&
                            rocknpu_env_enabled("ROCKNPU_W8_MTILE") &&
                            rocknpu_env_enabled("ROCKNPU_MTILE_PERSIST") &&
                            rocknpu_env_enabled("ROCKNPU_MTILE_MC");
                        const bool ffn_pair =
                            rocknpu_weight_name_contains(first_weights, ".ffn_gate.weight") &&
                            rocknpu_weight_name_contains(second_weights, ".ffn_up.weight") &&
                            first_activations == second_activations &&
                            (pair_m == 1 || native_mtile_pair) &&
                            first_weights->ne[0] == second_weights->ne[0] &&
                            first_weights->ne[1] == 5632 && second_weights->ne[1] == 5632 &&
                            rocknpu_quant_kind(first_weights, &first_kind) &&
                            rocknpu_quant_kind(second_weights, &second_kind);
                        if (ffn_pair) {
                            const size_t k_pair = static_cast<size_t>(first_weights->ne[0]);
                            bool native_w8 = false;
                            int status = ROCKNPU_STATUS_INVALID_ARGUMENT;
                            if (pair_m == 1) {
                                rocknpu_w8_tensor * first_w8 = rocknpu_w8_sidecar_get(
                                    context, first_weights->name, k_pair, 5632,
                                    static_cast<const uint8_t *>(first_weights->data), ggml_nbytes(first_weights), first_weights->type);
                                rocknpu_w8_tensor * second_w8 = rocknpu_w8_sidecar_get(
                                    context, second_weights->name, k_pair, 5632,
                                    static_cast<const uint8_t *>(second_weights->data), ggml_nbytes(second_weights), second_weights->type);
                                native_w8 = first_w8 != nullptr && second_w8 != nullptr;
                                if (pair_m == 1 && native_w8 &&
                                    rocknpu_env_enabled("ROCKNPU_FFN_COMPOSITE")) {
                                    context->ffn_composite_pending = true;
                                    context->ffn_composite_activation.assign(
                                        static_cast<const float *>(first_activations->data),
                                        static_cast<const float *>(first_activations->data) + k_pair);
                                    context->ffn_composite_gate = first_w8;
                                    context->ffn_composite_up = second_w8;
                                    context->ffn_composite_k = k_pair;
                                    context->ffn_composite_n = 5632;
                                    if (rocknpu_trace_enabled()) {
                                        std::fprintf(stderr,
                                            "ROCKNPU GGML TRACE ffn_composite_defer K=%zu N=5632\n", k_pair);
                                    }
                                    i += 1;
                                    break;
                                }
                                rocknpu_hybrid_item ffn_items[2] = {
                                    { first_weights, first_w8, static_cast<float *>(node->data), 5632 },
                                    { second_weights, second_w8, static_cast<float *>(second->data), 5632 },
                                };
                                const int hybrid_status = native_w8
                                    ? rocknpu_hybrid_group(context, first_activations, k_pair,
                                        rocknpu_hybrid_npu_share(k_pair, 5632), ffn_items, 2)
                                    : -1;
                                status = hybrid_status != -1 ? hybrid_status : native_w8 ? rocknpu_matmul_w8a8_pair_f32_f32_m1(
                                    context->runtime,
                                    first_w8->weights.data(), first_w8->scales.data(), 5632,
                                    second_w8->weights.data(), second_w8->scales.data(), 5632,
                                    static_cast<const float *>(first_activations->data),
                                    static_cast<float *>(node->data), static_cast<float *>(second->data), k_pair)
                                    : rocknpu_matmul_q_pair_f32_f32_m1(
                                    context->runtime,
                                    static_cast<const uint8_t *>(first_weights->data), ggml_nbytes(first_weights), first_kind, 5632,
                                    static_cast<const uint8_t *>(second_weights->data), ggml_nbytes(second_weights), second_kind, 5632,
                                    static_cast<const float *>(first_activations->data),
                                    static_cast<float *>(node->data), static_cast<float *>(second->data), k_pair);
                            } else {
                                status = rocknpu_matmul_q_pair_f32_f32_mtile(
                                    context->runtime,
                                    static_cast<const uint8_t *>(first_weights->data), ggml_nbytes(first_weights), first_kind, 5632,
                                    static_cast<const uint8_t *>(second_weights->data), ggml_nbytes(second_weights), second_kind, 5632,
                                    static_cast<const float *>(first_activations->data),
                                    static_cast<float *>(node->data), static_cast<float *>(second->data),
                                    pair_m, k_pair);
                            }
                            if (status != ROCKNPU_STATUS_OK) {
                                return GGML_STATUS_FAILED;
                            }
                            context->ffn_pair_calls++;
                            if (pair_m == 1) {
                                context->w8a8_m1_mul_mat_calls += 2;
                                if (native_w8) context->native_w8_calls += 2;
                            } else if (pair_m == 4) {
                                context->m4_calls += 2;
                            } else if (pair_m == 8) {
                                context->m8_calls += 2;
                            } else if (pair_m == 12) {
                                context->m12_calls += 2;
                            } else if (pair_m == 16) {
                                context->m16_calls += 2;
                            } else {
                                context->m_other_calls += 2;
                            }
                            if (first_weights->type == GGML_TYPE_Q4_K) context->q4_k_mul_mat_calls++; else context->q6_k_mul_mat_calls++;
                            if (second_weights->type == GGML_TYPE_Q4_K) context->q4_k_mul_mat_calls++; else context->q6_k_mul_mat_calls++;
                            if (rocknpu_trace_enabled()) {
                                std::fprintf(stderr,
                                    "ROCKNPU GGML TRACE ffn_pair first=%s second=%s M=%zu K=%zu N=5632+5632 kinds=%u/%u path=%s\n",
                                    first_weights->name, second_weights->name, pair_m, k_pair, first_kind, second_kind,
                                    pair_m == 1 ? "m1_pair" : "mtile_pair");
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
                                    static_cast<const uint8_t *>(first_weights->data), ggml_nbytes(first_weights), first_kind, first_weights
                                };
                                const rocknpu_qkv_weight_ref current_k {
                                    static_cast<const uint8_t *>(second_weights->data), ggml_nbytes(second_weights), second_kind, second_weights
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
                            rocknpu_w8_tensor * first_w8 = rocknpu_w8_sidecar_get(
                                context, first_weights->name, k_pair, 256,
                                static_cast<const uint8_t *>(first_weights->data), ggml_nbytes(first_weights), first_weights->type);
                            rocknpu_w8_tensor * second_w8 = rocknpu_w8_sidecar_get(
                                context, second_weights->name, k_pair, 256,
                                static_cast<const uint8_t *>(second_weights->data), ggml_nbytes(second_weights), second_weights->type);
                            const bool native_w8 = first_w8 != nullptr && second_w8 != nullptr;
                            const int status = native_w8 ? rocknpu_matmul_w8a8_pair_f32_f32_m1(
                                context->runtime,
                                first_w8->weights.data(), first_w8->scales.data(), 256,
                                second_w8->weights.data(), second_w8->scales.data(), 256,
                                static_cast<const float *>(first_activations->data),
                                static_cast<float *>(node->data), static_cast<float *>(second->data), k_pair)
                                : rocknpu_matmul_q_pair_f32_f32_m1(
                                context->runtime,
                                static_cast<const uint8_t *>(first_weights->data), ggml_nbytes(first_weights), first_kind, 256,
                                static_cast<const uint8_t *>(second_weights->data), ggml_nbytes(second_weights), second_kind, 256,
                                static_cast<const float *>(first_activations->data),
                                static_cast<float *>(node->data), static_cast<float *>(second->data), k_pair);
                            if (status != ROCKNPU_STATUS_OK) {
                                return GGML_STATUS_FAILED;
                            }
                            context->vk_pair_calls++;
                            context->w8a8_m1_mul_mat_calls += 2;
                            if (native_w8) context->native_w8_calls += 2;
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
                if (m == 4) context->m4_calls++;
                else if (m == 8) context->m8_calls++;
                else if (m == 12) context->m12_calls++;
                else if (m == 16) context->m16_calls++;
                else context->m_other_calls++;
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
                        m == 1 ? (w4a4_m1 ? "w4a4_m1" : "w8a8_m1") :
                        (rocknpu_small_native_mtile_supported(m, k, n) ? "native_mtile" : "fp16_bridge"));
                }
                const auto node_started = std::chrono::steady_clock::now();
                int status;
                const bool quantized_mtile = weights->type == GGML_TYPE_Q4_K || weights->type == GGML_TYPE_Q6_K;
                const bool native_mtile = quantized_mtile &&
                    (rocknpu_small_native_mtile_supported(m, k, n) ||
                     (m >= 32 && m <= 128 && m % 16 == 0 && rocknpu_env_enabled("ROCKNPU_NATIVE_MTILE")));
                const bool split_m32 = !native_mtile && m == 32 && rocknpu_env_enabled("ROCKNPU_M32_AS_2X16");
                const bool split_chunked = !native_mtile && m >= 32 && m <= 128 && m % 16 == 0 &&
                    rocknpu_env_enabled("ROCKNPU_M16_CHUNK_BATCH");
                if (split_m32 || split_chunked) {
                    const float * input = static_cast<const float *>(activations->data);
                    float * output = static_cast<float *>(node->data);
                    const size_t input_stride = 16 * k;
                    const size_t output_stride = 16 * n;
                    const auto run_chunk = [&](const float * chunk_input, float * chunk_output) -> int {
                        if (weights->type == GGML_TYPE_Q4_K) {
                            return rocknpu_matmul_q4_k_f32_f32(
                                context->runtime,
                                static_cast<const uint8_t *>(weights->data),
                                ggml_nbytes(weights),
                                chunk_input,
                                chunk_output,
                                16,
                                k,
                                n);
                        }
                        if (weights->type == GGML_TYPE_Q6_K) {
                            return rocknpu_matmul_q6_k_f32_f32(
                                context->runtime,
                                static_cast<const uint8_t *>(weights->data),
                                ggml_nbytes(weights),
                                chunk_input,
                                chunk_output,
                                16,
                                k,
                                n);
                        }
                        return rocknpu_matmul_f16_f32_f32(
                            context->runtime,
                            static_cast<const uint16_t *>(weights->data),
                            chunk_input,
                            chunk_output,
                            16,
                            k,
                            n);
                    };
                    status = ROCKNPU_STATUS_OK;
                    for (size_t chunk = 0; chunk < m / 16 && status == ROCKNPU_STATUS_OK; ++chunk) {
                        status = run_chunk(
                            input + chunk * input_stride,
                            output + chunk * output_stride);
                    }
                } else {
                    // M=16 must use the q4/q6 C API below so ROCKNPU_MTILE_MC can
                    // select the persistent multi-core M-tile pool.  The legacy
                    // sidecar M16 entry point is single-executor and is slower at
                    // server concurrency despite avoiding first-use conversion.
                    rocknpu_w8_tensor * native_w8 = m == 1 ? rocknpu_w8_sidecar_get(
                        context, weights->name, k, n,
                        static_cast<const uint8_t *>(weights->data), ggml_nbytes(weights), weights->type) : nullptr;
                    const int hybrid_status = native_w8 != nullptr && m == 1
                        ? rocknpu_hybrid_m1(context, weights, activations, node, native_w8, k, n)
                        : -1;
                    if (hybrid_status != -1) {
                        status = hybrid_status;
                        if (status == ROCKNPU_STATUS_OK) context->native_w8_calls++;
                    } else if (native_w8 != nullptr) {
                    status = m == 16
                        ? rocknpu_matmul_w8a8_f32_f32_m16(
                            context->runtime, native_w8->weights.data(), native_w8->scales.data(),
                            static_cast<const float *>(activations->data), static_cast<float *>(node->data), k, n)
                        : rocknpu_matmul_w8a8_f32_f32_m1(
                            context->runtime, native_w8->weights.data(), native_w8->scales.data(),
                            static_cast<const float *>(activations->data), static_cast<float *>(node->data), k, n);
                    if (status == ROCKNPU_STATUS_OK) context->native_w8_calls++;
                } else if (weights->type == GGML_TYPE_Q4_K) {
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
                }
                if (rocknpu_env_enabled("ROCKNPU_SPLIT_PROFILE") && context->graph_compute_calls >= 111) {
                    const uint64_t node_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - node_started).count());
                    char key[96];
                    std::snprintf(key, sizeof(key), "M=%zu K=%zu N=%zu", m, k, n);
                    auto & profile = context->shape_profile[std::string(key)];
                    profile.calls++;
                    profile.ns += node_ns;
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
    if (rocknpu_env_enabled("ROCKNPU_RUNTIME_M16_ROUTER")) {
        context->cpu_fallback = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (context->cpu_fallback != nullptr) {
            int threads = 4;
            if (const char * value = std::getenv("ROCKNPU_CPU_FALLBACK_THREADS")) {
                const int parsed = std::atoi(value);
                if (parsed > 0) threads = parsed;
            }
            ggml_backend_dev_t cpu_dev = ggml_backend_get_device(context->cpu_fallback);
            ggml_backend_reg_t cpu_reg = cpu_dev != nullptr ? ggml_backend_dev_backend_reg(cpu_dev) : nullptr;
            if (cpu_reg != nullptr) {
                auto set_n_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
                    ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_set_n_threads"));
                if (set_n_threads != nullptr) {
                    set_n_threads(context->cpu_fallback, threads);
                }
            }
        }
    }
    return new ggml_backend {
        /* .guid    = */ rocknpu_backend_guid(),
        /* .iface   = */ rocknpu_backend_iface,
        /* .device  = */ dev,
        /* .context = */ context,
    };
}

static const char * rocknpu_host_buffer_type_get_name(ggml_backend_buffer_type_t) {
    return "ROCKNPU_HOST";
}

static ggml_backend_buffer_t rocknpu_host_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    if (buffer != nullptr) {
        buffer->buft = buft;
    }
    return buffer;
}

static size_t rocknpu_host_buffer_type_get_alignment(ggml_backend_buffer_type_t) {
    return ggml_backend_cpu_buffer_type()->iface.get_alignment(ggml_backend_cpu_buffer_type());
}

static size_t rocknpu_host_buffer_type_get_alloc_size(
        ggml_backend_buffer_type_t, const ggml_tensor * tensor) {
    auto cpu = ggml_backend_cpu_buffer_type();
    return cpu->iface.get_alloc_size != nullptr
        ? cpu->iface.get_alloc_size(cpu, tensor)
        : ggml_nbytes(tensor);
}

static bool rocknpu_host_buffer_type_is_host(ggml_backend_buffer_type_t) {
    return true;
}

static const ggml_backend_buffer_type_i rocknpu_host_buffer_type_iface = {
    /* .get_name         = */ rocknpu_host_buffer_type_get_name,
    /* .alloc_buffer     = */ rocknpu_host_buffer_type_alloc_buffer,
    /* .get_alignment    = */ rocknpu_host_buffer_type_get_alignment,
    /* .get_max_size     = */ nullptr,
    /* .get_alloc_size   = */ rocknpu_host_buffer_type_get_alloc_size,
    /* .is_host          = */ rocknpu_host_buffer_type_is_host,
};

ggml_backend_buffer_type_t rocknpu_device_buffer_type(ggml_backend_dev_t dev) {
    static ggml_backend_buffer_type host_type = {
        /* .iface    = */ rocknpu_host_buffer_type_iface,
        /* .device   = */ nullptr,
        /* .context  = */ nullptr,
    };
    host_type.device = dev;
    return &host_type;
}

ggml_backend_buffer_type_t rocknpu_device_host_buffer_type(ggml_backend_dev_t dev) {
    return rocknpu_device_buffer_type(dev);
}

bool rocknpu_device_supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    if (rocknpu_env_enabled("ROCKNPU_SUPPORT_TRACE") && op != nullptr && op->op == GGML_OP_FLASH_ATTN_EXT) {
        const ggml_tensor * q = op->src[0];
        const ggml_tensor * k = op->src[1];
        const ggml_tensor * v = op->src[2];
        const ggml_tensor * mask = op->src[3];
        std::fprintf(stderr,
            "ROCKNPU SUPPORT FLASH name=%s q=%s[%lld,%lld,%lld,%lld] k=%s[%lld,%lld,%lld,%lld] v=%s[%lld,%lld,%lld,%lld] mask=%s[%lld,%lld,%lld,%lld] dst=%s[%lld,%lld,%lld,%lld]\n",
            op->name,
            q ? ggml_type_name(q->type) : "null", q ? (long long) q->ne[0] : 0, q ? (long long) q->ne[1] : 0, q ? (long long) q->ne[2] : 0, q ? (long long) q->ne[3] : 0,
            k ? ggml_type_name(k->type) : "null", k ? (long long) k->ne[0] : 0, k ? (long long) k->ne[1] : 0, k ? (long long) k->ne[2] : 0, k ? (long long) k->ne[3] : 0,
            v ? ggml_type_name(v->type) : "null", v ? (long long) v->ne[0] : 0, v ? (long long) v->ne[1] : 0, v ? (long long) v->ne[2] : 0, v ? (long long) v->ne[3] : 0,
            mask ? ggml_type_name(mask->type) : "null", mask ? (long long) mask->ne[0] : 0, mask ? (long long) mask->ne[1] : 0, mask ? (long long) mask->ne[2] : 0, mask ? (long long) mask->ne[3] : 0,
            ggml_type_name(op->type), (long long) op->ne[0], (long long) op->ne[1], (long long) op->ne[2], (long long) op->ne[3]);
    }
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
    /* .get_host_buffer_type = */ rocknpu_device_host_buffer_type,
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

// Validated execution routes that used to be opt-in research flags. They are
// set only when the user has not chosen a value, so any of them can still be
// disabled explicitly with "=0".
void rocknpu_apply_default_env() {
    static const char * const defaults[] = {
        "ROCKNPU_W8_DIRECT_SUBMIT",          // drive all NPU cores from the calling thread
        "ROCKNPU_EXPERIMENT_DIRECT_SCRATCH", // persistent per-core scratch + cached regcmds
        "ROCKNPU_PREFILL_CACHE",             // resident prompt-processing weights
        "ROCKNPU_NATIVE_MTILE",              // native W8A8 M-tile prompt processing
        "ROCKNPU_W8_MTILE",
        "ROCKNPU_MTILE_PERSIST",
        "ROCKNPU_MTILE_MC",                  // multi-core M-tile
        "ROCKNPU_NATIVE_MTILE_DOWN",         // K-split FFN down projection
    };
    for (const char * name : defaults) setenv(name, "1", 0);
}

ggml_backend_reg_t ggml_backend_rocknpu_reg() {
    static const bool defaults_applied = (rocknpu_apply_default_env(), true);
    (void) defaults_applied;
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ rocknpu_reg_iface,
        /* .context     = */ nullptr,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_rocknpu_reg)
GGML_BACKEND_DL_SCORE_IMPL(rocknpu_backend_score)
