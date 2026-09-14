# RockNPU

**Open-source Rust userspace runtime and compiler for Rockchip NPUs on mainline Linux.**

RockNPU is building the missing open userspace layer between standard machine-learning models and the Rockchip NPU exposed by Linux `drivers/accel/rocket`.

The goal is simple:

```text
standard model
    |
    v
  RockNPU
    |
    v
Linux accel/rocket
    |
    v
 RK3588 NPU
```

RockNPU does **not** depend on Rockchip's proprietary `librknnrt.so`, `librkllmrt.so`, RKNN Toolkit, RKLLM Toolkit, or a mandatory `.rknn` / `.rkllm` model conversion step.

> **Status: developer preview.** Real pretrained models already run on real RK3588 hardware, but operator coverage and the high-level end-user CLI/API are still under active development.

## Why RockNPU?

An RK3588 board contains a capable NPU, but the traditional software path is controlled by a vendor-specific userspace stack:

```text
PyTorch / ONNX
      |
      v
RKNN / RKLLM Toolkit
      |
      v
.rknn / .rkllm
      |
      v
proprietary userspace runtime
      |
      v
vendor kernel driver
      |
      v
Rockchip NPU
```

Mainline Linux now has an open Rockchip NPU kernel driver, **Rocket**, but Rocket intentionally remains a thin kernel driver. It powers the NPU, allocates/maps buffers, submits jobs, and handles synchronization. It does not parse ONNX graphs, choose tensor layouts, tile operators, pack weights, or generate RK3588 register commands.

RockNPU fills that gap:

```text
ONNX / future framework frontends
              |
              v
          RockNPU
   +-----------------------+
   | model import          |
   | graph execution       |
   | CPU fallback          |
   | tensor layout         |
   | padding / tiling      |
   | resident weights      |
   | NPU kernel planning   |
   | RK3588 regcmd encode  |
   | Rocket BO / submit    |
   +-----------------------+
              |
              v
     Linux accel/rocket
              |
              v
          RK3588 NPU
```

This makes RockNPU closer to an **NPU userspace compiler/runtime/backend** than to a machine-learning framework.

## RockNPU is not another Candle

RockNPU is not intended to replace Candle, PyTorch, ONNX Runtime, or other model/framework APIs.

Those projects answer questions such as:

- How does an application define or load a model?
- What tensor/operator API does the application use?
- How are models composed?

RockNPU answers a lower-level question:

> How do we execute that computation correctly and efficiently on the RK3588 NPU through the open Linux Rocket driver?

A future integration can therefore look like:

```text
Candle
  |
  v
candle-rocknpu backend
  |
  v
RockNPU
  |
  v
Rocket
  |
  v
RK3588 NPU
```

## Kernel requirement: you need Rocket

RockNPU talks directly to the Linux DRM accelerator **Rocket** UAPI. A kernel without Rocket support cannot use the NPU backend.

### Supported kernel/hardware today

Current RockNPU hardware development and validation targets:

- **SoC:** Rockchip RK3588
- **Kernel driver:** `drivers/accel/rocket`
- **Upstream availability:** Rocket entered mainline Linux in **6.18**
- **Verified RockNPU host:** Linux `6.18.43-current-rockchip64` on an RK3588 Orange Pi 5-class board
- **Device:** `/dev/accel/accel0`

The upstream kernel documentation currently lists RK3588 as the supported Rocket hardware:

- https://docs.kernel.org/accel/rocket/index.html

A distribution kernel older than Linux 6.18 can only work if it has a compatible Rocket backport plus the required RK3588 device-tree support.

RockNPU does not currently claim end-to-end support for other Rockchip NPU SoCs. Support should be treated as RK3588-only until a target has both a usable Rocket kernel path and real RockNPU hardware validation.

### 1. Check the kernel configuration

Your kernel needs the DRM accelerator core and Rocket driver enabled:

```text
CONFIG_DRM_ACCEL=y
CONFIG_DRM_ACCEL_ROCKET=y
```

or Rocket may be built as a module:

```text
CONFIG_DRM_ACCEL=y
CONFIG_DRM_ACCEL_ROCKET=m
```

On distributions that install the running kernel config under `/boot`:

```sh
grep -E 'CONFIG_DRM_ACCEL|CONFIG_DRM_ACCEL_ROCKET' /boot/config-"$(uname -r)"
```

Expected output resembles:

```text
CONFIG_DRM_ACCEL=y
CONFIG_DRM_ACCEL_ROCKET=m
```

If your kernel exposes its config through `/proc/config.gz` instead:

```sh
zgrep -E 'CONFIG_DRM_ACCEL|CONFIG_DRM_ACCEL_ROCKET' /proc/config.gz
```

If `CONFIG_DRM_ACCEL_ROCKET` is missing, install/build a kernel that contains Rocket. RockNPU cannot compensate for a missing kernel driver.

### 2. Load Rocket when built as a module

```sh
sudo modprobe rocket
```

Then verify that the module is present:

```sh
lsmod | grep '^rocket'
```

If Rocket is built into the kernel (`CONFIG_DRM_ACCEL_ROCKET=y`), there is no module to load.

### 3. Check for the accelerator device

The Linux accel subsystem exposes compute accelerators as `/dev/accel/accel*`.

```sh
ls -l /dev/accel/
```

For the current RK3588 setup, RockNPU expects:

```text
/dev/accel/accel0
```

A quick check:

```sh
test -r /dev/accel/accel0 -a -w /dev/accel/accel0 && echo "Rocket device is accessible"
```

If the kernel config is correct but `/dev/accel/accel0` does not appear, inspect probe/device-tree errors:

```sh
dmesg | grep -i rocket
ls /sys/bus/platform/drivers/rocket/
```

The RK3588 NPU nodes, clocks, resets and IOMMU mappings must be described correctly by the device tree. Having `rocket.ko` installed is not sufficient if the hardware never probes.

### 4. Give the user access to `/dev/accel/accel0`

On the verified host, the accel node belongs to the `render` group:

```sh
ls -l /dev/accel/accel0
```

Typical output:

```text
crw-rw---- root render ... /dev/accel/accel0
```

Check your groups:

```sh
id
```

If your distribution also assigns the Rocket node to `render`, add your user to that group:

```sh
sudo usermod -aG render "$USER"
```

Then log out and back in so the new group membership takes effect.

Do not run normal RockNPU applications as root merely to bypass device permissions; fix the device/group permissions instead.

### 5. Minimum Rocket readiness checklist

Before reporting a RockNPU NPU execution failure, these should all be true:

```text
[ ] RK3588 target (the hardware validated by RockNPU today)
[ ] kernel contains drivers/accel/rocket
[ ] CONFIG_DRM_ACCEL=y
[ ] CONFIG_DRM_ACCEL_ROCKET=y or =m
[ ] Rocket probes successfully
[ ] /dev/accel/accel0 exists
[ ] current user can read/write /dev/accel/accel0
```

## Build RockNPU

RockNPU's production path is Rust. It does not require Rockchip's RKNN/RKLLM runtime libraries.

Install a current stable Rust toolchain, then from the repository root:

```sh
cargo build --workspace
cargo test --workspace
```

The normal Rust runtime path uses the Rocket UAPI directly. Some **validation-only** reference tools documented under `docs/repro.md` additionally use C/libdrm; they are not required to link or run normal RockNPU Rust code.

## Using RockNPU today

RockNPU is currently a **developer preview**, so the supported user interface is still the Rust workspace and hardware validation binaries rather than a polished `rocknpu` CLI.

### Basic real-hardware smoke test

With a working `/dev/accel/accel0`:

```sh
cargo run --release -p rocket-smoke
```

This exercises the project-owned Rust Rocket UAPI/runtime/register-command path on the real NPU and compares results with CPU references.

### Real pretrained model gates

The repository currently has hardware gates for real pretrained ONNX models used during development. When their corresponding model/input artifacts are present, examples include:

```sh
# Pretrained MNIST MLP: dense layers on the NPU
cargo run --release -p rocket-smoke --bin prepared_mnist
python3 scripts/verify_real_mnist.py prepared-mnist-npu

# Official MNIST-8 CNN: Conv + dense compute on the NPU
cargo run --release -p rocket-smoke --bin mnist8_cnn_prepared
python3 scripts/verify_mnist8.py mnist8-prepared

# RGB CIFAR-10 CNN: 3 Conv + 2 Gemm nodes on the NPU
cargo run --release -p rocket-smoke --bin cifar10_edgeinfer
python3 scripts/verify_cifar10_edgeinfer.py
```

The Python scripts are **independent validation oracles**. They use the original ONNX model and ONNX `ReferenceEvaluator` to compare intermediate tensors; RockNPU does not use them to execute the production NPU path.

For the complete hardware/reproduction matrix, see [`docs/repro.md`](docs/repro.md).

## What works today

The project has already demonstrated, on real RK3588 hardware:

- direct Rocket buffer allocation/mapping/submission/synchronization from Rust;
- FP16 MatMul/Gemm register-command generation;
- FP32-output MatMul mode for higher-accuracy accumulation;
- MatMul M/N/K tiling and CPU/NPU K-accumulation policies;
- FP16 Conv2D register-command generation;
- logical low-channel Conv (`IC=1/3/8`, etc.) through transparent hardware channel padding;
- 32x32 RGB Conv execution;
- resident/prepacked static weights;
- one-, two- and three-worker Rocket execution using separate DRM fds/IOMMU domains;
- adaptive worker-count measurement/cache for streaming MatMul;
- ONNX graph execution for the operator subset required by the validated real models;
- CPU fallback for supported graph operators that are not yet lowered to the NPU;
- independent node-by-node correctness checks against ONNX `ReferenceEvaluator`.

Validated real-model classes currently include:

```text
pretrained dense MNIST MLP
pretrained MNIST CNN
pretrained RGB CIFAR-10 CNN
```

This is evidence that the end-to-end architecture works. It is **not** a claim of general ONNX compatibility yet.

## Current ONNX scope

Operator support is intentionally model-driven rather than an attempt to implement the whole ONNX specification up front.

The current validated CNN/dense path includes the subset needed by the real-model gates, including:

- `MatMul`
- restricted `Gemm`
- `Conv`
- `Add`
- `Relu`
- `MaxPool`
- `Reshape`

Some operators execute on the RK3588 NPU and some currently use CPU fallback. Unsupported models/operators should fail explicitly rather than silently invoking an external proprietary runtime.

## High-level Rust Session API

A first developer-preview `rocknpu::Session` API is now implemented:

```rust,ignore
use rocknpu::{Session, Tensor};

let session = Session::load("model.onnx")?;
let input = Tensor::from_f32(vec![1, 3, 32, 32], pixels)?;
let output = session.run(input)?;
println!("{:?}", output.stats());
```

`Session::load` parses the ONNX graph once, validates the current single-`FLOAT` input/output contract, opens the Rocket backend, and eagerly prepares supported static `Conv`/`MatMul`/`Gemm` weights into resident NPU buffers. Repeated `run()` calls reuse that model state rather than reparsing or repacking the model. `Session::prepare_stats()` exposes resident-weight preparation statistics, while each `RunOutput` carries explicit operator placement and timing statistics.

The Session currently targets the same deliberately small ONNX subset listed above. CPU fallback remains part of the execution plan for supported small operators such as `Add`, `Relu`, `MaxPool`, and `Reshape`; unsupported graph structures fail explicitly. `SessionOptions::cpu()` is also available for an explicit CPU session.

The same runtime is now exposed through a first developer-preview CLI:

```sh
rocknpu run model.onnx --input input.npy --output output.npy
```

From the workspace, the equivalent development invocation is:

```sh
cargo run -p rocknpu -- run model.onnx --input input.npy --output output.npy
```

The CLI accepts C-order NumPy `.npy` tensors with `float32` elements, targets `/dev/accel/accel0` by default, supports `--target cpu` for explicit CPU execution, and accepts `--device <path>` for an alternate Rocket device. It uses the same eager Session preparation and reports resident-weight plus NPU-placement statistics after each run. The emitted `.npy` output is readable by standard NumPy.

Both the Rust API and CLI are developer previews rather than stability guarantees, but they are real hardware-tested interfaces rather than conceptual placeholders. Current real-model CLI gates cover official MNIST-8 and edge-infer CIFAR-10 on RK3588.

Frameworks such as Candle should eventually be able to use RockNPU as an RK3588 NPU backend without reimplementing RK3588 register commands, layouts, memory management and Rocket submission themselves.

## Architecture

Current workspace crates:

```text
rocknpu           high-level Session/Tensor API and model runtime ownership
rocket-uapi       Linux Rocket UAPI structs/ioctl wrappers
rocket-runtime    safe Rocket device/BO/submit/wait ownership layer
rocknpu-regcmd    RK3588 register-command encoders and planners
rocknpu-conv      FP16 Conv2D executor and resident weights
rocknpu-matmul    FP16/FP32 MatMul executor, tiling and worker pool
rocknpu-tensor    minimal project-owned tensor contracts
rocknpu-ops       operator/backend dispatch contracts
rocknpu-onnx      deliberately small ONNX frontend/graph executor
rocket-smoke      real-hardware correctness/performance gates
```

Detailed architecture and hardware findings are in [`docs/architecture.md`](docs/architecture.md).

The project goal/non-goals are in [`docs/goal.md`](docs/goal.md).

## Correctness policy

RockNPU treats hardware correctness as a first-class requirement.

A successful `cargo build` is not considered proof that an NPU operator works.

For NPU paths we prefer the following evidence chain:

```text
original standard model
        |
        v
independent standard reference
        |
        +---- intermediate tensor differential
        |
        v
RockNPU on a real RK3588 NPU
```

CPU-vs-NPU agreement inside RockNPU alone is not considered sufficient because both paths could share the same importer/layout bug.

For FP16 execution, intermediate CPU/NPU tensors are not required to be bit-identical when hardware accumulation order legitimately differs. Error is measured against an independent reference with explicit numerical bounds; classification/top-k behavior is checked separately where appropriate.

## What RockNPU does not do

RockNPU is not trying to become:

- a training framework;
- an autograd engine;
- an optimizer library;
- another general-purpose PyTorch/Candle replacement;
- a compatibility layer for proprietary `.rknn` / `.rkllm` internals;
- a reason to maintain a permanent private kernel fork.

The preferred kernel boundary is upstream Linux `drivers/accel/rocket`.

## Project maturity / limitations

Please assume all of the following today:

- API stability is not guaranteed yet.
- ONNX coverage is incomplete.
- Quantized INT8 execution is not yet the primary production path.
- GGUF/LLM frontend support is a goal, not a completed feature.
- Performance tuning is still ongoing.
- RK3588 is the hardware target with real end-to-end validation today.
- A working Rocket-enabled kernel/device tree is mandatory for NPU execution.

CPU-only unit tests can run elsewhere, but hardware claims require a real RK3588 Rocket device.

## Documentation

- [`docs/goal.md`](docs/goal.md) — project goal and non-goals
- [`docs/architecture.md`](docs/architecture.md) — architecture, Rocket UAPI and hardware findings
- [`docs/repro.md`](docs/repro.md) — detailed real-hardware reproduction commands and validation gates

## Upstream references

RockNPU is designed around the upstream Linux Rocket kernel boundary:

- Linux Rocket documentation: https://docs.kernel.org/accel/rocket/index.html
- Linux accelerator subsystem: https://docs.kernel.org/accel/index.html

The project also uses pinned public references during hardware research and validation. Their role and licensing boundaries are documented in `docs/architecture.md`; validation/reference code is not silently copied into the production Rust path.
