# RockNPU goal

RockNPU is a Rust-native RK3588 NPU userspace compiler/runtime/backend.

Its job is to take model/frontend work, lower supported regions into RK3588 NPU operations, manage layouts/residency/scheduling in userspace, and expose a reusable backend to frameworks such as llama.cpp, ONNX frontends, and future Candle integration.

Target flow:

    ONNX / GGUF / framework frontend
                  |
                  v
             frontend adapter
                  |
                  v
             RockNPU IR
                  |
                  v
       partition / shape / lowering
       layout / tiling / quantization
       residency / userspace scheduling
              /              \
       CPU fallback       RK3588 NPU
                              |
                              v
                    existing device interface

RockNPU is not a training framework, autograd engine, general tensor framework, or model-conversion toolkit.

## Project boundary

RockNPU owns:

- frontend-neutral graph and tensor contracts;
- RK3588 operation lowering;
- register-command synthesis;
- tensor/weight packing and layout;
- quantization contracts used by RockNPU paths;
- resident buffers and prepared weights;
- userspace task/job construction;
- model/backend integration;
- correctness and performance validation.

RockNPU does not own system-space implementation. If the host exposes the required accelerator device interface, RockNPU uses it. If a required capability is not available there, the feature remains unsupported or pending rather than becoming a system-space subproject.

## Correctness rule

Every promoted NPU path needs an independent correctness oracle appropriate to its scope.

Examples:

- CPU mathematical oracle for a primitive;
- ONNX ReferenceEvaluator for imported models;
- llama.cpp / llama-gguf for TinyLlama model behavior;
- deterministic token sequences plus layer/tensor differentials for quantized paths.

Agreement between two RockNPU-owned paths alone is not sufficient evidence.

## Performance rule

A local operator win is not enough.

Promotion requires:

1. correctness;
2. a real workload that exercises the path;
3. a measurable whole-model or request-level improvement when performance is the purpose.

For small gains, use warm same-process or interleaved A/B measurements.

## Current LLM objective

The current LLM target is TinyLlama-class inference.

Validated work includes:

- stock llama.cpp dynamic backend integration;
- resident W8A8 M=1 projections;
- full-K K=5632 decode support;
- adaptive multi-worker routing;
- Q/V/K and gate/up userspace grouping;
- native M16/M32/M48/M64/M128 execution;
- prefill weight residency;
- fused residual for compatible FP16 prefill paths.

The primary remaining performance problem is ordinary M=1 autoregressive decode.

Canonical current hypotheses and closed experiments are maintained in docs/research-status.md and trialanderror.md.
