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
M % 4 == 0, N % 16 == 0
F16: K % 32 == 0
Q4_K / Q6_K: K % 256 == 0
```

Q4_K and Q6_K blocks are forwarded unchanged across the C++ adapter boundary, decoded in Rust by `rocknpu-capi`, converted to the existing FP16 weight contract, and then executed by `Fp16MatmulExecutor`. This is a correctness-first bridge to real quantized GGUF models, not a native quantized NPU kernel or a performance claim.

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

Accepted on RK3588: `6/6 tests passed`; unsupported M=1/non-multiple-of-4, batched, permuted, and F16-activation variants remain explicitly rejected.

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

Then run a four-token prompt so the current NPU M alignment is satisfied:

```sh
ROCKNPU_GGML_TRACE=1 \
GGML_BACKEND_PATH="$PWD/target/ggml-rocknpu/libggml-rocknpu.so" \
  /build/llama.cpp-reference/build-dl/bin/llama-completion \
  -fit off -ngl 0 \
  -m artifacts/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf \
  -no-cnv -p "The capital of" -n 1 --temp 0 --no-warmup
```

The independent stock CPU run greedily produces `" the"`. The RockNPU run produces the same continuation and reports:

```text
ROCKNPU GGML TRACE mul_mat weight=blk.0.attn_q.weight type=q4_K M=4 K=2048 N=2048
ROCKNPU GGML TRACE mul_mat weight=blk.0.attn_v.weight type=q6_K M=4 K=2048 N=256
...
ROCKNPU GGML TRACE summary q4_K_mul_mat=131 q6_K_mul_mat=20 f16_mul_mat=0
```

The trace is opt-in and emitted at the actual RockNPU `graph_compute` boundary, so this proves real TinyLlama prefill nodes entered the backend rather than merely being accepted by `supports_op`. The four-token gate has 151 NPU-eligible block projection MatMuls and all 151 execute through RockNPU. Stock llama.cpp output pruning presents `blk.21.ffn_gate`, `blk.21.ffn_up`, `blk.21.ffn_down`, and `output.weight` as `M=1`; those correctly remain on another backend because the current RK3588 MatMul path does not correctly support M=1. Batching, permutations, and additional GGML ops also remain unsupported. Native quantized execution and persistent/prepacked GGML weights are future performance work.
