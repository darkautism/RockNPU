# ggml-rocknpu

Out-of-tree GGML dynamic backend adapter for RockNPU. It is loaded by an unmodified llama.cpp process through `GGML_BACKEND_PATH`; no llama.cpp source patch is required.

The adapter is the only layer allowed to depend on GGML backend ABI types. RockNPU core/runtime crates remain GGML-independent. C++ calls the narrow `rocknpu-capi` ABI, which owns the Rust/Rocket boundary.

Validated against unmodified llama.cpp commit `391fac16460f15233a7740550d858ac96df3419d`.

For normal `GGML_BACKEND_PATH` auto-loading, build llama.cpp with its stock dynamic-backend option enabled (`-DGGML_BACKEND_DL=ON`). A build with static CPU registration (`GGML_BACKEND_DL=OFF`) can still load this plugin through tools that explicitly call `ggml_backend_load_all`, such as `--list-devices` and `test-backend-ops`, but ordinary completion does not automatically load the environment-provided plugin in that configuration.

## Build and discovery

```sh
cmake -S adapters/ggml-rocknpu \
  -B target/ggml-rocknpu \
  -DGGML_SOURCE_DIR=/build/llama.cpp-reference/ggml
cmake --build target/ggml-rocknpu -j

GGML_BACKEND_PATH="$PWD/target/ggml-rocknpu/libggml-rocknpu.so" \
  /build/llama.cpp-reference/build/bin/llama-cli --list-devices
```

Acceptance on a usable RK3588/Rocket host:

```text
ROCKNPU0: RockNPU RK3588
```

Device registration is not a C++ stub: `ggml_backend_score` and the registry call `rocknpu_device_count()`, which probes availability through Rust `RocketDevice::open()`.

## Supported compute slice

The deliberately narrow operation is contiguous, unbatched `GGML_OP_MUL_MAT`:

```text
src0 weights:     F16, Q4_K, or Q6_K [N,K]
src1 activations: F32 [M,K]
dst:              F32 [M,N]

prefill / M >= 4:
  M % 4 == 0, N % 16 == 0
  F16: K % 32 == 0
  Q4_K / Q6_K: K % 256 == 0

quantized decode / M == 1:
  Q4_K or Q6_K only
  K % 512 == 0
  N % 32 == 0, N <= 8192
```

Q4_K and Q6_K blocks are forwarded unchanged across the C++ adapter boundary and decoded in Rust by `rocknpu-capi`. For `M >= 4`, they are converted to the existing FP16 weight contract and executed by `Fp16MatmulExecutor`. For supported `M == 1`, Rust performs a correctness-first symmetric W8A8 conversion (per-tensor activation scale, per-output-channel weight scales), runs `Int8DecodeExecutor` through Rocket, and rescales the int32 result to F32. The current M=1 bridge redoes dequantization, requantization, allocation, and weight packing on every call; persistent/prepacked weights are the next performance step.

Everything else is rejected by `supports_op` rather than silently falling back inside the adapter.

Stock llama.cpp's own backend test provides the independent correctness oracle:

```sh
GGML_BACKEND_PATH="$PWD/target/ggml-rocknpu/libggml-rocknpu.so" \
  /build/llama.cpp-reference/build/bin/test-backend-ops \
  test -b ROCKNPU0 -o MUL_MAT \
  -p type_a=f16,type_b=f32,m=16,n=4,k=256
```

Accepted result:

```text
MUL_MAT(...m=16,n=4,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3]...): OK
MUL_MAT(...bs=[3,2]...): not supported [ROCKNPU]
MUL_MAT(...per=[0,3,2,1]...): not supported [ROCKNPU]
1/1 tests passed
Backend ROCKNPU: OK
```

The execution path is:

```text
stock GGML
  -> libggml-rocknpu.so
  -> rocknpu-capi
  -> Fp16MatmulExecutor
  -> RocketDevice
  -> RK3588 NPU
```

The Q4_K-specific stock oracle is:

```sh
GGML_BACKEND_PATH="$PWD/target/ggml-rocknpu/libggml-rocknpu.so" \
  /build/llama.cpp-reference/build-dl/bin/test-backend-ops \
  test -b ROCKNPU0 -o MUL_MAT -p q4_K
```

Accepted on RK3588: `7/7 tests passed`. The stock oracle's built-in M=1 sample uses `K=256`, so it remains intentionally rejected by RockNPU's `K % 512` W8A8 contract; the real TinyLlama gate below is the M=1 oracle.

The Q6_K-specific stock oracle is:

```sh
GGML_BACKEND_PATH="$PWD/target/ggml-rocknpu/libggml-rocknpu.so" \
  /build/llama.cpp-reference/build-dl/bin/test-backend-ops \
  test -b ROCKNPU0 -o MUL_MAT -p q6_K
```

Accepted on RK3588: `3/3 tests passed`, `Backend ROCKNPU: OK`.

## Real TinyLlama gate

Configure the same unmodified llama.cpp source with dynamic backends:

```sh
cmake -S /build/llama.cpp-reference \
  -B /build/llama.cpp-reference/build-dl \
  -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON
cmake --build /build/llama.cpp-reference/build-dl --target llama-completion test-backend-ops -j 8
```

Then run a four-token prompt and request two greedy output tokens so the second token requires a real autoregressive `M=1` decode step:

```sh
ROCKNPU_GGML_TRACE=1 \
GGML_BACKEND_PATH="$PWD/target/ggml-rocknpu/libggml-rocknpu.so" \
  /build/llama.cpp-reference/build-dl/bin/llama-completion \
  -fit off -ngl 0 \
  -m artifacts/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf \
  -no-cnv -p "The capital of" -n 2 --temp 0 --no-warmup
```

The independent stock CPU run greedily produces `" the United"`. The RockNPU run produces the same continuation. The opt-in trace is emitted at the actual `graph_compute` boundary and includes both paths:

```text
ROCKNPU GGML TRACE mul_mat weight=blk.0.attn_q.weight type=q4_K M=4 K=2048 N=2048 path=fp16_bridge
...
ROCKNPU GGML TRACE mul_mat weight=blk.21.ffn_gate.weight type=q4_K M=1 K=2048 N=5632 path=w8a8_m1
ROCKNPU GGML TRACE mul_mat weight=blk.0.attn_q.weight type=q4_K M=1 K=2048 N=2048 path=w8a8_m1
...
ROCKNPU GGML TRACE summary q4_K_mul_mat=268 q6_K_mul_mat=40 f16_mul_mat=0 w8a8_m1_mul_mat=157
```

The 157 W8A8 calls are exactly three final-layer FFN projections that stock llama.cpp output-prunes to `M=1` during the four-token prefill plus all `22 x 7 = 154` block projections in the following autoregressive decode step. The model output head has `N=32000` and remains on CPU because RockNPU deliberately caps the current W8A8 path at `N<=8192`. Batching, permutations, F16 M=1, and additional GGML ops remain unsupported. This milestone proves true stock llama.cpp autoregressive decode through RockNPU/Rocket/RK3588; it does not yet claim competitive decode speed because static weights are currently rebuilt for every call.
