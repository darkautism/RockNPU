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

Q4_K and Q6_K blocks are forwarded unchanged across the C++ adapter boundary and decoded in Rust by `rocknpu-capi`. For `M >= 4`, they are converted to the existing FP16 weight contract and executed by `Fp16MatmulExecutor`. For supported `M == 1`, the **default** path performs symmetric W8A8 conversion (per-tensor activation scale, per-output-channel weight scales), runs `Int8DecodeExecutor` through Rocket, and rescales the int32 result to F32. Decode weights are lazily converted once per backend context and retained as native 32x32-packed Rocket BOs; subsequent calls reuse the resident weight and only stage the activation/regcmd/output scratch. `ROCKNPU_GGML_TRACE=1` reports cache entries, resident bytes, hits/misses, and hit/miss timing.

### Experimental native W4A4 decode

Q4_K `M == 1` can optionally use RockNPU's native signed-W4A4 RK3588 path. This is a research/characterization path, **not a drop-in quality-equivalent replacement for W8A8**. Enable it only with an explicit scope; `ROCKNPU_W4A4=1` without `ROCKNPU_W4A4_SCOPE` leaves the default W8A8 route in place.

A representative FFN experiment is:

```sh
ROCKNPU_W4A4=1 \
ROCKNPU_W4A4_SCOPE=ffn \
ROCKNPU_W4A4_GROUP=2048 \
ROCKNPU_W4A4_HADAMARD=1 \
ROCKNPU_W4A4_TRACE=1 \
GGML_BACKEND_PATH="$PWD/target/ggml-rocknpu/libggml-rocknpu.so" \
  /build/llama.cpp-reference/build-dl/bin/llama-completion \
  -fit off -ngl 0 \
  -m artifacts/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf \
  -no-cnv -p "The capital of" -n 3 --temp 0 --no-warmup
```

`ROCKNPU_W4A4_GROUP` chooses the K quantization group size; when it equals K, RockNPU's persistent W4 pool measures 1/2/3-worker N-splits and caches only the winner. `ROCKNPU_W4A4_HADAMARD=1` applies the same orthonormal FWHT to activation and each static weight row before symmetric int4 quantization. `ROCKNPU_W4A4_SCOPE` accepts `ffn`, `attn`, `proj2048`, `kv`, or explicit `all`.

On the validated RK3588, `K=2048,N=5632` native W4A4 is exact against the integer CPU oracle and three-way N-split measured about `2.08 -> 0.97 ms`; the real TinyLlama FFN tuner measured approximately `[2081,1251,859] us` for 1/2/3 workers and selected three. No saturation was observed. The short `-n 3` TinyLlama gate can match W8A8, but a 16-token deterministic continuation diverges, and Q/O-only W4A4 also diverges over the longer gate. Same-setting `llama-bench tg8` results (`5.09 ± 0.13 tok/s` W4 FFN vs `5.02 ± 0.17 tok/s` default W8A8) are within board/run noise. Consequently the W4 path is shipped only to make native int4 hardware work reproducible and available for further quantization research; it carries no whole-model speed or token-identity claim.

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

Then run a four-token prompt and request three greedy output tokens so one autoregressive step populates the resident decode cache and the following step reuses it:

```sh
ROCKNPU_GGML_TRACE=1 \
GGML_BACKEND_PATH="$PWD/target/ggml-rocknpu/libggml-rocknpu.so" \
  /build/llama.cpp-reference/build-dl/bin/llama-completion \
  -fit off -ngl 0 \
  -m artifacts/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf \
  -no-cnv -p "The capital of" -n 3 --temp 0 --no-warmup
```

The independent stock CPU run greedily produces `" the United States"`. The RockNPU run produces the same continuation. The opt-in trace is emitted at the actual `graph_compute` boundary and includes both paths plus resident-cache statistics:

```text
ROCKNPU GGML TRACE mul_mat weight=blk.0.attn_q.weight type=q4_K M=4 K=2048 N=2048 path=fp16_bridge
...
ROCKNPU GGML TRACE mul_mat weight=blk.21.ffn_gate.weight type=q4_K M=1 K=2048 N=5632 path=w8a8_m1
ROCKNPU GGML TRACE mul_mat weight=blk.0.attn_q.weight type=q4_K M=1 K=2048 N=2048 path=w8a8_m1
...
ROCKNPU GGML TRACE summary q4_K_mul_mat=402 q6_K_mul_mat=60 f16_mul_mat=0 w8a8_m1_mul_mat=311
ROCKNPU GGML TRACE decode_cache hits=157 misses=154 entries=154 resident_mb=924.00 hit_ms=206.36 hit_avg_ms=1.314 miss_ms=11304.78 miss_avg_ms=73.408 tuned_shapes=4 worker_calls=[0,88,223] ksplit_calls=45
```

The cache contains all 154 (`22 x 7`) transformer-block projection weights. The three final-layer FFN projections first appear as M=1 during output-pruned prefill; the first autoregressive step finishes populating the cache, and the next step reuses it. Each distinct `(K,N)` decode geometry tunes effective 1/2/3-worker N-splits using warmed, forward/reverse-interleaved median samples; shapes whose K would otherwise require multiple sequential tasks also tune 2/3-worker K-splits. The selector requires at least a 5% measured improvement before accepting the next candidate, then caches both split topology and worker count and releases losing resident candidates. Two identical final runs made the same choice distribution, `worker_calls=[0,88,223]` with `ksplit_calls=45`; cache-hit averages were `1.411 ms` and `1.314 ms`, both below the previous N-split-only `1.485 ms`. The output head has `N=32000` and remains on CPU because the current W8A8 path caps N at 8192. Batching, permutations, F16 M=1, and additional GGML ops remain unsupported. Real timing decomposition shows submit/wait dominates representative worker calls (>90%), so the next performance work is kernel/submit efficiency rather than host packing, scratch allocation, or a fixed routing table.

### V/K pair projection

For M=1 decode, adjacent attention V/K projections with the same activation and
K=2048,N=256 are concatenated along N and issued as one N=512 W8A8 projection.
This path is enabled by default and preserves the independent per-row scales of
Q4_K/Q6_K inputs. Set `ROCKNPU_VK_PAIR=0` only to disable it for A/B debugging.
