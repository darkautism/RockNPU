# RockNPU

Open-source Rust userspace runtime/compiler/backend for Rockchip RK3588 NPUs.

RockNPU fills the userspace gap between standard model/framework frontends and the RK3588 NPU. It owns graph lowering, tensor layouts, quantization, resident weights, register-command generation, userspace scheduling, and framework integration.

## Quick start

### llama.cpp

```sh
git clone --depth 1 -b b10969 https://github.com/ggml-org/llama.cpp.git llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF && cmake --build llama.cpp/build --target llama-cli -j
cmake -S adapters/ggml-rocknpu -B target/ggml-rocknpu -DCMAKE_BUILD_TYPE=Release -DGGML_SOURCE_DIR="$PWD/llama.cpp/ggml" -DGGML_CPU_LIBRARY="$PWD/llama.cpp/build/bin/libggml-cpu.so" && cmake --build target/ggml-rocknpu -j
export GGML_BACKEND_PATH="$PWD/target/ggml-rocknpu/libggml-rocknpu.so"
./llama.cpp/build/bin/llama-cli --list-devices && ./llama.cpp/build/bin/llama-cli -dev ROCKNPU0 -m /path/to/model.gguf
```

A usable RK3588 should list `ROCKNPU0: RockNPU RK3588`.

### Ollama

Build the RockNPU backend once with the llama.cpp steps above, then:

```sh
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl stop ollama 2>/dev/null || true
GGML_BACKEND_PATH="$PWD/target/ggml-rocknpu/libggml-rocknpu.so" LLAMA_ARG_DEVICE=ROCKNPU0 ollama serve
# in another shell:
ollama run tinyllama:1.1b-chat-v1-q4_K_M
```

Current Ollama 0.34.x pins llama.cpp `b10969` (commit `391fac16460f15233a7740550d858ac96df3419d`), the same llama.cpp revision used by the validated RockNPU GGML backend. No Ollama source patch is required. `LLAMA_ARG_DEVICE=ROCKNPU0` is the standard llama.cpp device selector inherited by Ollama's runner. The optional W8 sidecar described later improves repeat startup/decode preparation but is not part of the basic install. Real llama.cpp/Ollama CPU-vs-NPU checks are recorded in [the frontend benchmark](docs/benchmarks/2026-09-22/frontend-cpu-npu.md).

> You may also like oRKLLM/ork-driver, an important open reverse-engineering reference for RK35xx regcmd, quantized matmul, decode layouts, and multi-core execution.

## Project scope

RockNPU is a userspace project.

It owns:

- frontend adapters;
- frontend-neutral IR;
- tensor/layout contracts;
- RK3588 operation lowering;
- register-command synthesis;
- resident buffers and prepared weights;
- quantization/dataflow choices;
- userspace worker/task scheduling;
- framework/backend integration;
- correctness and performance validation.

It does not own or maintain system-space implementation.

If the host exposes a compatible accelerator device at /dev/accel/accel0, RockNPU uses it. If a desired capability is unavailable there, the feature remains unsupported or pending instead of becoming a system-space subproject.

## Current status

Developer preview.

Real pretrained models run on real RK3588 hardware.

Validated areas include:

- FP16 MatMul and Conv2D;
- resident/prepacked static weights;
- ONNX dense/CNN subsets;
- stock llama.cpp dynamic backend integration;
- stock Ollama dynamic backend integration with no Ollama source patch;
- Candle eager/module adapter with a real prepared RockNpuLinear NPU path;
- TinyLlama Q4_K_M prefill and decode;
- W8A8 M=1 decode;
- K=5632 full-K decode;
- adaptive 1/2/3-worker routing;
- V/K, gate/up, and Q/V/K userspace projection grouping;
- native M16/M32/M48/M64/M128 execution;
- fused FP16 residual for compatible prefill projections;
- same-job INT8 weight reuse primitive;
- independent CPU/model correctness oracles.

The largest current LLM performance problem is ordinary M=1 autoregressive decode.

The canonical research direction is docs/research-status.md.

## Architecture

    llama.cpp / GGUF / Candle / ONNX / future framework frontend
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

The frontend-neutral public direction is:

    Graph -> Executable -> Session

Detailed crate ownership is documented in docs/architecture.md.

## Build

Install a current stable Rust toolchain.

    cargo build --workspace
    cargo test --workspace

Normal RockNPU code does not require proprietary RKNN/RKLLM runtime libraries.

## Device requirement

The NPU backend currently targets RK3588 and expects an accessible accelerator device:

    /dev/accel/accel0

Check:

    test -r /dev/accel/accel0 -a -w /dev/accel/accel0 && echo "accelerator available"

CPU-only tests can run elsewhere, but hardware claims require a real RK3588 NPU.

## Basic hardware smoke

    cargo run --release -p rocket-smoke

This exercises the Rust buffer/submit/register-command path and compares NPU results with CPU references.

Useful decode gates:

    cargo run --release -p rocket-smoke --bin int8_decode_m1 -- 2048 256
    cargo run --release -p rocket-smoke --bin int8_decode_m1 -- 2048 2048
    cargo run --release -p rocket-smoke --bin int8_decode_m1 -- 5632 2048
    cargo run --release -p rocket-smoke --bin int8_mtile -- 128
    cargo run --release -p rocket-smoke --bin fused_residual -- 16 2048 2048
    cargo run --release -p rocket-smoke --bin int8_weight_reuse -- 128 256 reuse

See docs/repro.md for the current validation matrix.

## llama.cpp / GGML backend

RockNPU can be loaded by stock, unmodified llama.cpp as an out-of-tree dynamic GGML backend.

Assume:

    export ROCKNPU_DIR=/path/to/RockNPU
    export LLAMA_CPP_DIR=/path/to/llama.cpp

Build llama.cpp with dynamic backends:

    cmake -S "$LLAMA_CPP_DIR" \
      -B "$LLAMA_CPP_DIR/build-rocknpu" \
      -DGGML_BACKEND_DL=ON \
      -DGGML_NATIVE=ON
    cmake --build "$LLAMA_CPP_DIR/build-rocknpu" -j

Build the RockNPU plugin:

    cmake -S "$ROCKNPU_DIR/adapters/ggml-rocknpu" \
      -B "$ROCKNPU_DIR/target/ggml-rocknpu" \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_SOURCE_DIR="$LLAMA_CPP_DIR/ggml" \
      -DGGML_CPU_LIBRARY="$LLAMA_CPP_DIR/build-rocknpu/bin/libggml-cpu.so"
    cmake --build "$ROCKNPU_DIR/target/ggml-rocknpu" -j

Load it:

    export GGML_BACKEND_PATH="$ROCKNPU_DIR/target/ggml-rocknpu/libggml-rocknpu.so"

Verify:

    "$LLAMA_CPP_DIR/build-rocknpu/bin/llama-cli" --list-devices

Expected:

    ROCKNPU0: RockNPU RK3588

The currently validated GGML slice is intentionally model-driven rather than broad for its own sake.

Important TinyLlama paths:

- Q4_K/Q6_K projection import;
- W8 sidecar;
- resident decode weights;
- adaptive M=1 worker routing;
- V/K pair;
- gate/up pair;
- Q/V/K triple grouping;
- native M-tile verifier/prefill route.

The N=32000 LM head remains on CPU.

## W8 sidecar

The sidecar is provenance-bound to the source GGUF.

Current canonical TinyLlama model SHA-256:

    5c66751b61537f9e55177b1b67e06af88e0e2df88f86de4909f5bf87fb1ae583

Do not reuse a sidecar generated from another model revision.

## TinyLlama status

RockNPU has a real autoregressive TinyLlama path.

Validated model-level work includes:

- all transformer layers;
- NPU projection execution;
- resident decode weights;
- CPU/NPU mixed execution;
- deterministic greedy comparison with independent llama.cpp / llama-gguf references;
- explicit handling of near-tie numerical divergences.

Ordinary M=1 decode remains slower than the native CPU reference and is the highest-priority performance target.

Current clean CPU reference:

- native ARM llama.cpp tg128: about 33.88 tok/s on the validated board/settings.

Old NPU absolute throughput results from mixed experimental environments are not current project toplines.

## Current validated userspace performance conclusions

### M128 vs M64

Fresh A-B-B-A at 100% acceptance:

- M64: 126.174 / 122.309 tok/s;
- M128: 137.932 / 136.478 tok/s.

The useful conclusion is roughly +10.4% at the two-run centers for M128.

### Projection grouping

Default userspace grouping includes:

- V + K;
- gate + up;
- Q + V + K.

These execute as larger combined projections rather than merely changing graph labels.

### Weight reuse

Same-job repeated-weight reuse is a real primitive.

Do not force wide TinyLlama projections into narrow column segments merely to trigger it; measured wide-N production attempts regressed substantially.

### W4A4

W4A4 hardware execution is real and locally fast, but tested TinyLlama routes did not preserve model behavior well enough and did not establish a whole-model speed win.

It remains research-only.

## Real pretrained ONNX gates

When corresponding artifacts are available:

    cargo run --release -p rocket-smoke --bin prepared_mnist
    python3 scripts/verify_real_mnist.py prepared-mnist-npu

    cargo run --release -p rocket-smoke --bin mnist8_cnn_prepared
    python3 scripts/verify_mnist8.py mnist8-prepared

    cargo run --release -p rocket-smoke --bin cifar10_edgeinfer
    python3 scripts/verify_cifar10_edgeinfer.py

The Python paths are independent validation oracles.

## Frontend-neutral API

A frontend can lower into RockNPU without using ONNX bytes at runtime:

    use rocknpu::{Executable, Graph, Session};

    let graph: Graph = frontend.lower_to_rocknpu()?;
    let executable = Executable::compile(graph)?;
    let session = Session::from_executable(executable)?;
    let output = session.run(input)?;

ONNX import is one frontend over this contract.

GGML integration is another.

Candle integration is a third: Candle tensors/modules stay in the adapter, while execution uses the same frontend-neutral RockNPU userspace ops/runtime primitives. The first supported Candle module is prepared Linear; this is not yet a full Candle graph compiler.

## Correctness policy

A successful build is not proof that an NPU path works.

Preferred evidence chain:

    independent mathematical/model reference
                    |
                    v
          tensor / primitive differential
                    |
                    v
              real RK3588 NPU
                    |
                    v
            model-level behavior

For risky quantized/dataflow changes:

1. primitive oracle;
2. layer differential;
3. deterministic model output;
4. whole-model performance A/B.

A local microbenchmark is never enough to promote a model path.

## Performance policy

For small expected gains:

- warm both variants;
- use same-process or tightly interleaved ABBA;
- record selected worker topology;
- separate cold preparation from hot execution;
- keep raw evidence;
- measure the whole model/request, not only the local operator.

The helper script scripts/bench_llama_cpu_npu.py stores CPU/NPU ABBA evidence and hashes.

## Current research priorities

1. profile current ordinary M=1 decode;
2. find the next real userspace dataflow/projection reduction after existing QKV and gate/up grouping;
3. prototype an exact end-to-end NPU attention slice;
4. find a quality-equivalent FFN intermediate representation;
5. improve speculative proposer acceptance so M128 verifier capacity is useful;
6. revisit output head/cold-start work only after the above.

See docs/research-status.md for hypotheses, evidence, and closed directions.

## Workspace

    rocknpu           high-level Graph/Executable/Session API
    rocknpu-capi      C ABI for external adapters
    rocknpu-ir        frontend-neutral graph/tensor metadata
    rocknpu-llm       transformer runtime building blocks
    rocket-uapi       narrow accelerator ABI wrappers
    rocket-runtime    safe userspace device/buffer/submit layer
    rocknpu-regcmd    RK3588 register-command encoders/planners
    rocknpu-conv      Conv2D execution
    rocknpu-matmul    MatMul/decode/tiling/residency/worker pools
    rocknpu-tensor    tensor contracts
    rocknpu-ops       operator/backend dispatch
    rocknpu-onnx      ONNX importer/reference executor
    rocket-smoke      real-hardware gates
    adapters/ggml-rocknpu
                      stock llama.cpp dynamic backend
    adapters/candle-rocknpu
                      Candle Tensor/Module frontend adapter

## Documentation

- docs/goal.md — project boundary and success criteria
- docs/architecture.md — userspace architecture
- docs/research-status.md — canonical current research direction
- trialanderror.md — closed/validated experiment ledger
- docs/repro.md — reproducible userspace gates

## Thanks

Special thanks to oRKLLM/ork-driver for pioneering open RK35xx NPU reverse engineering.

RockNPU uses that work as hardware/regcmd research input while maintaining its own Rust userspace architecture.

## License

RockNPU original code is MIT licensed.

The directly source-derived ork-driver regcmd baseline remains isolated with its original ISC attribution under crates/rocknpu-regcmd/src/int8/ork_isc.rs, with the notice also preserved in docs/licenses/ork-driver-ISC.txt.
