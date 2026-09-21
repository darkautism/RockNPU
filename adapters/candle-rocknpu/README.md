# candle-rocknpu

Thin Candle frontend adapter for RockNPU.

The boundary is intentionally strict:

- Candle owns eager tensors and Module semantics.
- Shared execution uses RockNPU userspace crates: rocknpu-ops, rocknpu-tensor, and rocket-runtime.
- RockNPU core crates do not depend on Candle.
- The adapter does not depend on the ONNX importer/runtime path.

## Current support

### RockNpuLinear

RockNpuLinear implements Candle's Module trait.

Candle weights use [out_features, in_features]. Static weights and optional bias are rounded once to RockNPU's current FP16 prepared-matmul contract.

F32 Candle inputs may have arbitrary leading dimensions. The last dimension is in_features; leading dimensions are flattened into M for execution and restored on output.

NPU execution currently requires aligned static feature sizes:

- in_features % 32 == 0
- out_features % 16 == 0

The adapter pads the dynamic M dimension to a multiple of four internally and crops the output, so callers do not have to pad batch/token dimensions.

Example:

    use candle_core::{Device, Module, Tensor};
    use candle_rocknpu::RockNpuLinear;

    let device = Device::Cpu;
    let weight = Tensor::zeros((32, 32), candle_core::DType::F32, &device)?;
    let linear = RockNpuLinear::new(&weight, None)?;
    let input = Tensor::zeros((3, 32), candle_core::DType::F32, &device)?;
    let output = linear.forward(&input)?;

A CPU-reference constructor is also available:

    let linear = RockNpuLinear::cpu(&weight, None)?;

It uses the same RockNPU FP16 matmul contract without requiring NPU hardware.

## Build note on AArch64

Candle 0.11 pulls `gemm-f16`. Its current AArch64 inline-assembly path requires the architectural FP16 target feature at compile time. This adapter is a standalone Cargo workspace, and its local `.cargo/config.toml` enables `+fp16` only here; the RockNPU root workspace and core crates are unaffected.

## Validation

From `adapters/candle-rocknpu`:

CPU-only adapter tests:

    cargo test --lib

RK3588 NPU correctness gate:

    cargo test npu_linear_matches_candle_cpu -- --ignored --nocapture

The hardware gate compares Candle -> RockNPU NPU output against Candle CPU matmul using the same F16-rounded input and static weights.
