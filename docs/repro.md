# RockNPU reproduction guide

This guide covers reproducible userspace validation only.

Prerequisites:

- RK3588 host;
- accessible accelerator device at /dev/accel/accel0;
- current stable Rust toolchain;
- model/test artifacts required by the selected gate.

No system-driver modification is part of this guide.

## 1. Workspace build

    cargo build --workspace
    cargo test --workspace

For a quick userspace ABI/runtime check:

    cargo test -p rocket-uapi
    cargo test -p rocket-runtime
    cargo test -p rocknpu-regcmd

## 2. Basic hardware smoke

    cargo run --release -p rocket-smoke

This exercises the Rust buffer/submit/register-command path and compares NPU output with a CPU reference.

## 3. W8A8 M=1 decode gates

TinyLlama-sized projection examples:

    cargo run --release -p rocket-smoke --bin int8_decode_m1 -- 2048 256
    cargo run --release -p rocket-smoke --bin int8_decode_m1 -- 2048 2048
    cargo run --release -p rocket-smoke --bin int8_decode_m1 -- 2048 5632
    cargo run --release -p rocket-smoke --bin int8_decode_m1 -- 5632 2048

The result must pass the exact int32 CPU oracle.

## 4. Native M-tile gate

Examples:

    cargo run --release -p rocket-smoke --bin int8_mtile -- 64
    cargo run --release -p rocket-smoke --bin int8_mtile -- 128

M128 is a promoted capability. Do not infer model speed from this primitive alone.

## 5. Fused residual gate

    cargo run --release -p rocket-smoke --bin fused_residual -- 16 2048 2048

The fused result is compared against plain NPU matmul plus CPU FP16 residual addition.

Validated max-abs error for this shape is approximately 0.000610.

## 6. Same-job weight reuse gate

    cargo run --release -p rocket-smoke --bin int8_weight_reuse -- 128 256 reuse

This validates repeated same-weight M128 tasks.

Do not force wide TinyLlama projections into N64 column splits merely to use this feature; that production experiment regressed substantially.

## 7. Real pretrained ONNX gates

When the model/input artifacts are present:

    cargo run --release -p rocket-smoke --bin prepared_mnist
    python3 scripts/verify_real_mnist.py prepared-mnist-npu

    cargo run --release -p rocket-smoke --bin mnist8_cnn_prepared
    python3 scripts/verify_mnist8.py mnist8-prepared

    cargo run --release -p rocket-smoke --bin cifar10_edgeinfer
    python3 scripts/verify_cifar10_edgeinfer.py

The Python tools are independent validation oracles, not production execution backends.

## 8. Build the stock-llama.cpp dynamic backend

Assume:

    export ROCKNPU_DIR=/path/to/RockNPU
    export LLAMA_CPP_DIR=/path/to/llama.cpp

Build llama.cpp with dynamic backends:

    cmake -S "$LLAMA_CPP_DIR" \
      -B "$LLAMA_CPP_DIR/build-rocknpu" \
      -DGGML_BACKEND_DL=ON \
      -DGGML_NATIVE=ON
    cmake --build "$LLAMA_CPP_DIR/build-rocknpu" -j

Build the RockNPU backend:

    cmake -S "$ROCKNPU_DIR/adapters/ggml-rocknpu" \
      -B "$ROCKNPU_DIR/target/ggml-rocknpu" \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_SOURCE_DIR="$LLAMA_CPP_DIR/ggml"
    cmake --build "$ROCKNPU_DIR/target/ggml-rocknpu" -j

Load it:

    export GGML_BACKEND_PATH="$ROCKNPU_DIR/target/ggml-rocknpu/libggml-rocknpu.so"

Verify discovery:

    "$LLAMA_CPP_DIR/build-rocknpu/bin/llama-cli" --list-devices

Expected device:

    ROCKNPU0: RockNPU RK3588

## 9. TinyLlama W8 sidecar

The W8 sidecar must match the exact source GGUF.

Before performance testing, verify the sidecar manifest source hash equals the model SHA-256.

Current canonical TinyLlama model hash:

    5c66751b61537f9e55177b1b67e06af88e0e2df88f86de4909f5bf87fb1ae583

Do not reuse a sidecar generated from another GGUF revision.

## 10. Decode benchmark

The repository script runs interleaved CPU/NPU measurements and stores raw evidence.

Example:

    python3 scripts/bench_llama_cpu_npu.py \
      --bench "$LLAMA_CPP_DIR/build-rocknpu/bin/llama-bench" \
      --plugin "$ROCKNPU_DIR/target/ggml-rocknpu/libggml-rocknpu.so" \
      --model /path/to/TinyLlama.gguf \
      --sidecar /path/to/w8-sidecar \
      --output /tmp/rocknpu-abba \
      --mode decode \
      --tokens 128 \
      --reps 3 \
      --blocks 2

Before timing, set CPU policies 0/4/6 to performance so host-side comparison is reproducible.

The benchmark script records:

- command;
- model/plugin/binary hashes;
- CPU policy/frequency snapshot;
- temperature snapshot;
- per-process raw stdout/stderr;
- result rows;
- block speedups.

## 11. Small performance claims

For expected gains below about 5%, independent process-to-process comparisons are too noisy.

Use:

1. one model/context/backend process when possible;
2. warm baseline and candidate;
3. interleaved ABBA order;
4. enough repetitions to inspect block direction;
5. deterministic correctness gate in the same code revision.

Projection-pair work historically required this discipline.

## 12. Cold vs hot execution

Always state whether timing includes:

- model loading;
- sidecar loading;
- first dequantization;
- prepared-weight creation;
- first cache fill;
- steady-state cache hits.

A cold-start optimization is not a decode optimization.

## 13. Current clean reference checks

The 2026-09-22 consolidation revalidated current main with the ordinary public device interface after a fresh reboot.

Confirmed again:

- M=1 K=5632 N=2048 exact decode;
- M128 K=2048 N=2048 exact M-tile;
- fused residual M16/K2048/N2048;
- repeated M128 weight-reuse gate;
- M128 > M64 on a 100%-acceptance lookup A-B-B-A.

See docs/research-status.md for the current interpretation and priority order.

## 14. Evidence policy

A promoted claim must include:

- RockNPU commit;
- model hash;
- llama.cpp revision;
- plugin/binary hashes;
- exact environment variables;
- command;
- correctness result;
- raw timing evidence.

Do not use old absolute NPU results from mixed experimental environments as current toplines.
