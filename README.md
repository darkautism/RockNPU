# RockNPU

**Open-source Rust userspace runtime and compiler for Rockchip NPUs on mainline Linux.**

> **You may also like:** [`ork-driver`](https://github.com/oRKLLM/ork-driver) — a clean-room userspace matmul library for Rockchip NPUs and one of the most advanced open research projects around RK35xx regcmd, quantized matmul, decode layouts, and multi-core execution.

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

For TinyLlama on native ARM llama.cpp, try the opt-in **NPU prompt processing +
CPU generation** mode: `ROCKNPU_PREFILL_CACHE=1 ROCKNPU_DECODE=0`. It retains
prompt weights on the NPU and uses FP32 partial outputs with f64 K accumulation.
On the controlled RK3588 test, two CPU/NPU ABBA blocks measured **12.16% higher
throughput** for a 512-token prompt plus 128 generated tokens; all 24 requests succeeded.
See the [build and usage instructions](adapters/ggml-rocknpu/README.md#npu-prompt-processing-with-cpu-generation)
and [same-machine CPU comparison, raw results and limits](docs/benchmarks/2026-09-20/README.md).
This mode does not claim faster NPU-only decode or CPU-identical generated tokens.


## RK3588 driver scope

RockNPU is a **userspace NPU runtime / reverse-engineering project**. It does not maintain a custom Rocket kernel driver, kernel fork, DVFS stack, or board-voltage policy.

The supported baseline is the packaged `rocket` driver provided by the host kernel. Historical benchmark work used external GPL Rocket variants to characterize the silicon at higher clocks or to test kernel-side hypotheses; those results remain research records only and are not RockNPU installation requirements or production dependencies.

Kernel-module swaps, custom IOMMU-domain caches, scheduler experiments, voltage probes, and DVFS-driver maintenance are outside the project scope. If future userspace performance work requires a kernel capability, prefer an externally maintained/public driver and keep that dependency explicitly separate from RockNPU.

For CPU-side benchmark reproducibility on RK3588, record all CPU cpufreq policies and the shared DSU state; on this board `policy0` affects memory bandwidth even when llama.cpp is pinned to Cortex-A76 cores.

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

### Configure RockNPU as a llama.cpp / GGML backend

RockNPU can be loaded by **stock, unmodified llama.cpp** as an out-of-tree GGML dynamic backend. You do not need to patch llama.cpp or maintain a RockNPU-specific fork.

There are three steps:

```text
build llama.cpp with dynamic backends enabled
        |
        v
build libggml-rocknpu.so against that GGML source tree
        |
        v
set GGML_BACKEND_PATH before starting llama.cpp
```

Assume:

```sh
export ROCKNPU_DIR=/path/to/RockNPU
export LLAMA_CPP_DIR=/path/to/llama.cpp
```

First build llama.cpp with GGML dynamic backend loading enabled:

```sh
cmake -S "$LLAMA_CPP_DIR" \
  -B "$LLAMA_CPP_DIR/build-rocknpu" \
  -DGGML_BACKEND_DL=ON
cmake --build "$LLAMA_CPP_DIR/build-rocknpu" -j
```

Then build the RockNPU GGML plugin from the RockNPU repository:

```sh
cd "$ROCKNPU_DIR"
cmake -S adapters/ggml-rocknpu \
  -B target/ggml-rocknpu \
  -DGGML_SOURCE_DIR="$LLAMA_CPP_DIR/ggml"
cmake --build target/ggml-rocknpu -j
```

Point stock llama.cpp at the resulting shared library:

```sh
export GGML_BACKEND_PATH="$ROCKNPU_DIR/target/ggml-rocknpu/libggml-rocknpu.so"
```

Verify that llama.cpp can discover the NPU backend:

```sh
"$LLAMA_CPP_DIR/build-rocknpu/bin/llama-cli" --list-devices
```

On a usable RK3588/Rocket host, the device list should include:

```text
ROCKNPU0: RockNPU RK3588
```

After that, start llama.cpp normally with `GGML_BACKEND_PATH` still set. For example:

```sh
GGML_BACKEND_PATH="$GGML_BACKEND_PATH" \
  "$LLAMA_CPP_DIR/build-rocknpu/bin/llama-completion" \
  -fit off -ngl 0 \
  -m /path/to/model.gguf \
  -p "The capital of" -n 2 --temp 0 --no-warmup
```

You do **not** need to modify llama.cpp source or pass a RockNPU-specific `--device` workaround. The GGML scheduler discovers `ROCKNPU0` through the plugin and sends supported operations to it; unsupported operations remain available to the other registered backends rather than being silently emulated inside the RockNPU adapter.

The currently validated GGML slice is deliberately narrow: contiguous `GGML_OP_MUL_MAT` with F16, Q4_K, or Q6_K weights and F32 activations. Real TinyLlama Q4_K_M prefill and autoregressive decode are hardware-proven. The `M=4` prefill projections use the FP16 correctness bridge; quantized Q4_K/Q6_K `M=1` projections with `K % 512 == 0`, `N % 32 == 0`, and `N <= 8192` use the W8A8/INT8 decode path through upstream Rocket. Static decode weights are lazily dequantized/requantized once and kept in worker-local Rocket-resident INT8 BOs. Routing is measured rather than model-specific: every `(K,N)` shape tunes effective 1/2/3-worker N-splits, and wide-K shapes that would otherwise require multiple sequential K tasks also tune 2/3-worker K-splits. Candidates are warmed, measured in forward/reverse-interleaved rounds, and a more complex choice must improve median latency by at least 5%; the winner is cached and losing resident BOs are released. With prompt `The capital of` and greedy `-n 3`, stock CPU and RockNPU both produce `" the United States"`. Two identical final traced runs selected the same topology (`worker_calls=[0,88,223]`, `ksplit_calls=45`) and measured cache-hit averages of `1.411 ms` and `1.314 ms`, versus `1.485 ms` before K-split and `2.573 ms` on the older single-fd resident path. Profiling shows the remaining hot-path cost is overwhelmingly NPU submit/wait (>90% on representative projections), not host packing or BO setup. The `N=32000` output head remains on CPU by design; further decode gains now require kernel/submit efficiency or broader operation offload rather than a hard-coded worker table.

An **experimental native W4A4 M=1 path** is also available for Q4_K research, but it is intentionally **not the default**. It nibble-packs signed int4 activations/weights, keeps static weights resident, supports an orthonormal Walsh-Hadamard rotation before quantization, and can auto-tune 1/2/3-way N-splits on the three RK3588 NPU fds. Real-hardware integer gates are exact: for `K=2048,N=5632`, three-way W4A4 N-split measured about `0.97 ms` versus `2.08 ms` single-worker, with zero int16 saturation. The same real-model tuner selected three workers at roughly `2.081/1.251/0.859 ms` for 1/2/3 workers. This does **not** yet translate into a material whole-model win: same-setting TinyLlama `tg8` measured `5.09 ± 0.13 tok/s` for the experimental W4 FFN path versus `5.02 ± 0.17 tok/s` for default W8A8. More importantly, longer deterministic generation diverges from W8A8 despite zero saturation, so the remaining error is quantization quality, not hardware correctness. Therefore W4A4 requires both `ROCKNPU_W4A4=1` and an explicit `ROCKNPU_W4A4_SCOPE=...`; absence of a scope leaves W8A8 routing unchanged. See the adapter/repro docs before enabling it.

For exact backend ABI constraints, correctness tests and the pinned llama.cpp validation revision, see [`adapters/ggml-rocknpu/README.md`](adapters/ggml-rocknpu/README.md) and [`docs/repro.md`](docs/repro.md).

### Basic real-hardware smoke test

With a working `/dev/accel/accel0`:

```sh
cargo run --release -p rocket-smoke
```

This exercises the project-owned Rust Rocket UAPI/runtime/register-command path on the real NPU and compares results with CPU references.

The W8A8 M=1 decode gates use the ISC-licensed ork-driver register-command/layout work while keeping allocation and submission on RockNPU's Rust + upstream Rocket path:

```sh
# TinyLlama-sized M=1 projection geometries
cargo run --release -p rocket-smoke --bin int8_decode_m1 -- 2048 256
cargo run --release -p rocket-smoke --bin int8_decode_m1 -- 2048 2048
cargo run --release -p rocket-smoke --bin int8_decode_m1 -- 2048 5632

# TinyLlama FFN-down: K=5632 split into 1024/512 partials, then exact host int32 accumulation
cargo run --release -p rocket-smoke --bin int8_decode_widek

# Persistent N-split characterization; optional third argument selects 1..3 workers
cargo run --release -p rocket-smoke --bin int8_decode_multicore -- 2048 5632 3
cargo run --release -p rocket-smoke --bin int8_decode_multicore -- 2048 2048 2
cargo run --release -p rocket-smoke --bin int8_decode_multicore -- 5632 2048 3
```

The single-worker gates bit-match their CPU int32 references on the verified RK3588 host, and `int8_decode_multicore` exact-checks the gathered N slices against the same int32 oracle. Isolated characterization measured up to `2.40x` for `K=2048,N=5632` and `2.33x` for `K=5632,N=2048`; smaller shapes can prefer fewer workers, which is why production measures and caches worker count instead of baking these observations into policy. The same prepared/pool primitives back quantized stock GGML `M=1` execution; see the adaptive resident-cache TinyLlama `-n 3` gate in [`docs/repro.md`](docs/repro.md).

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
- W8A8/INT8 M=1 decode MatMul through Rocket, including TinyLlama-sized `K=2048` projections and `K=5632` K-split/host-accumulation;
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

## Frontend-neutral middle-layer contract

The first model-format-neutral boundary is now implemented. A framework or model frontend can produce RockNPU IR and enter the runtime without passing ONNX bytes or using the CLI:

```rust,ignore
use rocknpu::{Executable, Graph, Session};

let graph: Graph = frontend.lower_to_rocknpu()?;
let executable = Executable::compile(graph)?;
let session = Session::from_executable(executable)?;
let output = session.run(input)?;
```

`rocknpu-ir` owns the shared `Graph`, `Node`, tensor metadata, constants and current operator attributes. `Executable::compile` validates the current runtime contract and derives device-independent preparation geometry. `Session` then binds that executable to CPU or Rocket/NPU resources and performs resident-weight preparation.

The ONNX adapter exposes `rocknpu_onnx::import_graph(bytes) -> Graph`. The imported graph owns its constants, so the source model bytes can be dropped before `Executable` or `Session` is created. `Session::from_graph` is a convenience form of `Executable::compile` followed by `Session::from_executable`.

This is the first real framework-facing contract, not a claim that compiler separation is finished. The validated CNN graph executor still lives inside `rocknpu-onnx` internally; moving that executor ownership fully below the model-format adapter is the next cleanup step. The public boundary no longer requires an ONNX model, however, and the same `Graph -> Executable -> Session` path is available to future Candle/GGUF adapters.

## Reference ONNX frontend and Session façade

A first developer-preview `rocknpu::Session` façade is now implemented for the current ONNX reference frontend:

```rust,ignore
use rocknpu::{Session, Tensor};

let session = Session::load("model.onnx")?;
let input = Tensor::from_f32(vec![1, 3, 32, 32], pixels)?;
let output = session.run(input)?;
println!("{:?}", output.stats());
```

`Session::load` is the ONNX convenience path: it imports the model once into the same shared `Graph`, compiles it to `Executable`, opens the Rocket backend, and eagerly prepares supported static `Conv`/`MatMul`/`Gemm` weights into resident NPU buffers. Repeated `run()` calls reuse that model state rather than reparsing or repacking the model. `Session::prepare_stats()` exposes resident-weight preparation statistics, while each `RunOutput` carries explicit operator placement and timing statistics.

The Session currently targets the same deliberately small ONNX subset listed above. CPU fallback remains part of the execution plan for supported small operators such as `Add`, `Relu`, `MaxPool`, and `Reshape`; unsupported graph structures fail explicitly. `SessionOptions::cpu()` is also available for an explicit CPU session.

For end-to-end integration and debugging, the same ONNX reference frontend is also exposed through a thin developer-preview CLI:

```sh
rocknpu run model.onnx --input input.npy --output output.npy
```

From the workspace, the equivalent development invocation is:

```sh
cargo run -p rocknpu -- run model.onnx --input input.npy --output output.npy
```

The CLI accepts C-order NumPy `.npy` tensors with `float32` elements, targets `/dev/accel/accel0` by default, supports `--target cpu` for explicit CPU execution, and accepts `--device <path>` for an alternate Rocket device. It uses the same eager Session preparation and reports resident-weight plus NPU-placement statistics after each run. The emitted `.npy` output is readable by standard NumPy.

Both paths are hardware-tested developer previews. Current real-model CLI gates cover official MNIST-8 and edge-infer CIFAR-10 on RK3588. The CLI is a **reference frontend / integration harness**, not the architectural center of RockNPU.

The core boundary now begins below model-format parsing with `Graph -> Executable -> Session`. It is still intentionally small, but future ONNX, GGUF/LLM, Candle, or other inference frontends can target that same contract rather than duplicating partitioning, lowering, tensor layout, preparation/residency, scheduling, CPU fallback, or RK3588/Rocket execution.

## LLM / transformer status

The first model-format-independent LLM runtime slice now exists in `rocknpu-llm`. It implements CPU-reference RMSNorm, explicit Llama/TinyLlama `Normal` RoPE and Qwen2 `NeoX` RoPE, causal MHA/GQA attention, SwiGLU, residuals and a bounded KV-cache container, plus `HybridLinear` resident FP16 weights over the existing RockNPU MatMul backend.

The intended first-stage placement is now hardware-proven on RK3588:

```text
large aligned prefill projections -> resident RockNPU NPU MatMul
RMSNorm / RoPE / GQA attention    -> CPU initially
SwiGLU / residual glue            -> CPU initially
M=1 decode projections / LM head  -> CPU until GEMV is benchmarked/proven
```

The real GGUF path now performs autoregressive generation. RockNPU loads `TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf`, tokenizes the prompt, dequantizes each transformer layer once to FP16, executes the prompt prefill through all 22 blocks, and retains a separate K/V cache for every layer. All seven projections per layer use resident NPU MatMul during aligned prefill, so 154 Linear operations run on the real RK3588 NPU. Current M=1 decode projections and the LM head intentionally remain on CPU until a dedicated GEMV path is validated.

The accepted 32-token hardware gate uses the low-ambiguity prompt `1, 2, 3, 4, 5, 6, 7, 8,`. RockNPU generates ` 9, 10, 11, 12, 13, 14, 15, 16, ` with all 32 generated token IDs exactly matching both `llama-gguf`'s independent CPU model path and llama.cpp commit `391fac16460f15233a7740550d858ac96df3419d` in raw greedy-completion mode. A semantic gate also produces `The capital of France is` -> ` Paris.` and matches both references for the first eight generated tokens.

Cross-runtime greedy token identity is only treated as a hard gate when the reference top-logit margin is numerically stable. The Q4_K_M model exposes genuine near ties: after the France prompt, token 9 is `C` vs ` C`; llama.cpp prefers `C` by about 0.125 logit while the FP16-dequantized Rust reference prefers ` C` by about 0.040. This is recorded as a precision divergence between native quantized and FP16-dequantized execution, not silently called a KV-cache failure. Stable-sequence gates, hidden-state/cache differential tests, and decoded semantic output are used together.

The current GGUF path remains deliberately correctness-first: common GGUF F16/BF16 and Q4/Q5/Q8/K-quant weights are converted to FP16 before RockNPU preparation, and the external `llama-gguf` crate is used for GGUF/tokenizer/format support rather than as the RockNPU execution backend. Native quantized NPU kernels and an NPU/GEMV path for M=1 decode remain the major performance work.

## Architecture

Current workspace crates:

```text
rocknpu           high-level Graph/Executable/Session runtime API
rocknpu-capi      narrow C ABI for external framework/backend adapters
rocknpu-ir        frontend-neutral graph, node, constant and tensor metadata IR
rocknpu-llm       hybrid Llama/Qwen transformer primitives and runtime building blocks
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

An out-of-tree GGML adapter now proves the external backend boundary without modifying llama.cpp: stock llama.cpp dynamically loads `libggml-rocknpu.so`, discovers the real Rocket-backed `ROCKNPU0` device through `rocknpu-capi`, and stock `test-backend-ops` validates aligned F16, Q4_K, and Q6_K/F32 `MUL_MAT` against its independent CPU reference. A real TinyLlama Q4_K_M prefill through stock llama.cpp executes all 151 NPU-eligible block projection MatMuls at the RockNPU `graph_compute` boundary (131 Q4_K + 20 Q6_K) and matches the stock CPU greedy next token for the validated four-token prompt. GGML ABI details remain confined to `adapters/ggml-rocknpu`; see its README and `docs/repro.md` for the exact gates.

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
- Real TinyLlama GGUF loading and autoregressive generation are hardware-proven; decode performance and model/operator coverage remain incomplete.
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

The project also uses pinned public references during hardware research and validation. Where production code is directly derived from third-party implementation work, the boundary and attribution are kept explicit rather than hidden inside the Rust rewrite.

## Thanks

Special thanks to [`ork-driver`](https://github.com/oRKLLM/ork-driver) and the oRKLLM project for their pioneering open research on Rockchip NPUs. Their work on RK35xx register-command synthesis, INT8/W8A8 and INT4 execution, decode-oriented layouts, resident weights, and multi-core scheduling provided RockNPU with major inspiration and practical hardware knowledge.

RockNPU takes a different kernel/runtime path — Rust userspace over upstream Linux `drivers/accel/rocket` — but the public research done by ork-driver has materially accelerated this project.

## License

RockNPU original code is licensed under the [MIT License](LICENSE).

RockNPU keeps the ISC boundary deliberately narrow: `crates/rocknpu-regcmd/src/int8/ork_isc.rs` contains the directly source-derived baseline regcmd template from ISC-licensed [`ork-driver`](https://github.com/oRKLLM/ork-driver). That private file carries the original ISC notice; the surrounding Rust encoder/API and the `rocknpu-regcmd` crate remain MIT. A copy of the upstream notice is also preserved in [`docs/licenses/ork-driver-ISC.txt`](docs/licenses/ork-driver-ISC.txt).
