# RockNPU architecture

RockNPU is organized around a frontend-neutral userspace execution core.

## 1. High-level flow

    llama.cpp / GGUF / ONNX / future framework frontend
                         |
                         v
                    adapter/importer
                         |
                         v
                     rocknpu-ir
                         |
                         v
              executable preparation
       shape / partition / lowering / placement
       layout / tiling / quantization / residency
                         |
               +---------+---------+
               |                   |
               v                   v
          CPU fallback       RK3588 NPU backend
                                   |
                                   v
                         existing accel device

The architectural center is Graph -> Executable -> Session, not any one model format.

## 2. Public userspace boundary

Frontends should not need to know:

- RK3588 register addresses;
- native feature/weight layouts;
- tile geometry;
- buffer residency rules;
- task construction;
- worker topology;
- prepared-weight lifetime.

Those details stay below the RockNPU core/backend boundary.

The current frontend-neutral path is:

    Graph
      |
      v
    Executable::compile
      |
      v
    Session::from_executable
      |
      +--> CPU fallback
      |
      +--> RK3588 NPU execution

The ONNX loader and GGML backend are adapters over this lower-level contract.

## 3. Workspace responsibilities

### rocknpu

High-level Graph / Executable / Session API.

### rocknpu-ir

Frontend-neutral graph, node, constant, tensor metadata, and operator attributes.

### rocknpu-tensor

Minimal project-owned tensor/layout contracts.

### rocknpu-regcmd

RK3588 register-command encoders and geometry planners.

This crate owns hardware-programming knowledge that belongs in userspace.

### rocket-uapi

Narrow ABI definitions required to talk to the existing accelerator device interface.

It is intentionally a wrapper, not a system implementation.

### rocket-runtime

Safe userspace ownership around device handles, buffers, submit, and completion/wait operations.

### rocknpu-matmul

MatMul executors, prepared/resident weights, tiling, W8A8 decode, M-tile execution, accumulation, scratch reuse, and worker pools.

### rocknpu-conv

FP16 Conv2D execution and resident weights.

### rocknpu-ops

Operator/backend dispatch contracts.

### rocknpu-onnx

Small ONNX importer/reference graph executor.

The long-term direction is to keep model-format-specific parsing here while shared execution moves through the frontend-neutral core.

### rocknpu-llm

Transformer-oriented userspace runtime pieces:

- RMSNorm reference path;
- RoPE;
- GQA/MHA reference attention;
- SwiGLU;
- residual glue;
- KV-cache container;
- hybrid/resident linear projections.

### rocknpu-capi

Narrow C ABI used by external backend integrations.

### adapters/ggml-rocknpu

Out-of-tree GGML backend loaded by stock llama.cpp.

It is responsible for:

- supported-op discovery;
- graph-facing routing;
- W8 sidecar integration;
- projection grouping;
- backend-local caches/stashes;
- calling the C ABI.

### rocket-smoke

Real-hardware primitive and model gates.

## 4. Data-layout ownership

RockNPU owns all transformations required between framework tensors and RK3588-native execution layouts.

Examples include:

- row-major activation staging;
- prepacked static weights;
- W8 sidecar layout;
- M-tile packing;
- output gather;
- K/N split aggregation.

The model frontend should not implement these transformations independently.

## 5. Residency

Prepared static weights should stay resident when the workload reuses them.

Current validated resident patterns include:

- prepacked FP16 projection weights;
- W8A8 decode weights;
- persistent scratch;
- prepared M-tile layouts;
- provenance-bound W8 sidecar data.

Cold preparation and hot execution must be measured separately.

## 6. Userspace scheduling

RockNPU may decide, in userspace:

- worker count;
- N split vs K split;
- projection grouping;
- task grouping within the public submit contract;
- prewarm policy;
- cache residency/eviction;
- CPU vs NPU placement.

Topology must be measured by shape. Avoid model-specific hard-coded worker tables when a small runtime tuner can make the decision.

## 7. LLM execution shape

Current TinyLlama-style decode is mixed execution.

Important NPU regions:

- Q/K/V/O projections;
- gate/up/down projections;
- compatible prefill projections;
- larger M-tile verifier work.

Important current CPU regions:

- LM head;
- attention glue not yet promoted to an NPU dataflow;
- model operations whose NPU path is not quality/performance validated.

Existing projection grouping reduces userspace boundaries:

- V + K;
- gate + up;
- Q + V + K using a backend-local stash across partitions.

## 8. Correctness architecture

Primitive tests use mathematical CPU oracles.

Model integration uses independent framework/model references.

Promotion sequence for a risky new path:

1. shape/encoding unit test;
2. primitive hardware oracle;
3. layer/tensor differential;
4. deterministic model output;
5. whole-model performance A/B.

No step may be skipped merely because a local microbenchmark is fast.

## 9. Research boundary

Userspace reverse engineering is in scope when it produces:

- register-command knowledge;
- layout knowledge;
- quantization/dataflow knowledge;
- safe use of the existing device interface.

System-driver development is not part of the architecture.

The canonical research direction is docs/research-status.md.
