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

## User-facing runtime milestone

The first ONNX command-line path is now implemented on top of the high-level Session runtime:

```text
rocknpu run model.onnx --input input.npy --output output.npy
```

For the currently supported ONNX subset, the CLI accepts C-order `float32` NumPy tensors, prepares static NPU weights once, executes the model through Rocket, writes a standard `.npy` output, and exposes CPU execution as an explicit target. Official MNIST-8 and edge-infer CIFAR-10 have both passed this path on real RK3588 hardware with independent reference checks.

This is a product milestone, not the end state. The next long-term interface goals are:

```text
rocknpu run model.gguf
Candle / framework adapter -> same RockNPU runtime/compiler
multi-input / multi-output and broader dtype / quantized model contracts
```

Those interfaces should feed the same project-owned compiler/backend rather than duplicate RK3588 register-command, layout, residency, or Rocket submission logic.
