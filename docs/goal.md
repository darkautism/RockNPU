# RockNPU Goal

Build a **Rust-native RK3588 NPU userspace inference compiler/runtime/backend** that drives the NPU through Linux mainline `drivers/accel/rocket`, without requiring Rockchip's closed userspace runtimes or toolkits.

Target flow:

```text
ONNX / GGUF / Candle
        |
        v
 project frontend / adapter
        |
        v
 project-owned tensor + graph IR
        |
        v
 shape / layout / tiling / quantization / lowering
        |
        +--> RK3588 NPU regions --> regcmd compiler --> accel/rocket
        |
        +--> CPU fallback regions
```

RockNPU is **not** a new general-purpose Candle/PyTorch. It does not aim to provide training, autograd, optimizers, a broad GPU abstraction, or operator coverage for its own sake. Tensor/graph APIs stay as small as the inference compiler/runtime needs. Candle may later use RockNPU as an RK3588 backend.

The default design does not require `librknnrt.so`, `librkllmrt.so`, RKNN Toolkit, RKLLM Toolkit, `.rknn`, or `.rkllm`. The kernel strategy is mainline Rocket rather than a long-lived private `rknpu` fork.

## Correctness rule

Every NPU lowering needs an independent CPU/standard-format oracle. A model is not accepted merely because RockNPU's own CPU and NPU paths agree. Real model outputs must be compared with an independent implementation such as ONNX ReferenceEvaluator/NumPy, with an explicit precision/tolerance policy.

## First-stage success criterion

A real pretrained standard-format model must load from its original `.onnx`, execute meaningful inference on RK3588 with major compute operators genuinely dispatched to the NPU, and produce independently verified outputs. Unsupported regions may execute on CPU.

This criterion is now met by the external pretrained Pico-CNN MNIST MLP gate: four Gemm layers execute on RK3588 NPU and the first 50 canonical MNIST test images have identical top-1 predictions to ONNX ReferenceEvaluator.

## Runtime boundary and reference frontend

RockNPU's primary product boundary is the **middle layer**, not a standalone inference frontend. The intended architecture is:

```text
Candle / ONNX frontend / GGUF frontend / application runtime
                         |
                         v
                 frontend adapter
                         |
                         v
              RockNPU core contract
        graph/IR + partition + lowering
        layout/tiling + residency + scheduling
                  /              \
          CPU fallback       RK3588 backend
                                  |
                                  v
                         Linux accel/rocket
```

A frontend or inference framework should call RockNPU; it should not need to know RK3588 register commands, Rocket BO/IOVA details, native tensor packing, or NPU scheduling rules.

The current `rocknpu::Session::load("model.onnx")` and `rocknpu run model.onnx --input input.npy --output output.npy` paths are useful **bootstrap/reference frontends**. They prove the complete stack and provide a reproducible integration/debug harness, but ONNX parsing and CLI file I/O are not the core runtime boundary.

The first frontend-neutral contract is now implemented as `Graph -> Executable -> Session`. `rocknpu-ir` owns the shared graph/node/constant/tensor representation; `Executable::compile` validates the current runtime contract and derives device-independent preparation geometry; `Session` binds the executable to CPU or Rocket/NPU resources. The ONNX adapter now exposes `import_graph`, and imported model bytes can be discarded before runtime creation.

ONNX import, GGUF/LLM frontends, and Candle/framework integration should all use project-owned contracts and the same preparation, placement, residency, scheduling, CPU fallback, and Rocket backend machinery. `rocknpu-llm` now provides a hybrid transformer runtime with resident NPU prefill projections around CPU-reference RMSNorm, architecture-correct Normal/NeoX RoPE, GQA attention, SwiGLU and residual glue, plus per-layer K/V state for incremental decode. The real-GGUF autoregressive correctness target is now met: TinyLlama-1.1B Q4_K_M runs all 22 transformer layers with 154 prefill projection Linears on the RK3588 NPU and a 32-token greedy sequence exactly matches independent `llama-gguf` and llama.cpp oracles on a stable high-margin prompt. Native quantized NPU kernels and M=1 decode/GEMV optimization are now the primary LLM performance targets; cross-runtime near-tie logit flips remain an explicit numerical-precision concern rather than a hidden correctness assumption.
