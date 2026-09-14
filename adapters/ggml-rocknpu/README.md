# ggml-rocknpu

Out-of-tree GGML dynamic backend adapter for RockNPU. It is loaded by an unmodified llama.cpp process through `GGML_BACKEND_PATH`; no llama.cpp source patch is required.

The adapter is the only layer allowed to depend on GGML backend ABI types. RockNPU core/runtime crates remain GGML-independent. C++ calls the narrow `rocknpu-capi` ABI, which owns the Rust/Rocket boundary.

Validated against llama.cpp commit `391fac16460f15233a7740550d858ac96df3419d`.

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

## First compute gate

The first deliberately narrow operation is contiguous, unbatched `GGML_OP_MUL_MAT`:

```text
src0 weights:     F16 [N,K]
src1 activations: F32 [M,K]
dst:              F32 [M,N]
M % 4 == 0, K % 32 == 0, N % 16 == 0
```

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

This is an integration/correctness gate, not broad GGML coverage. Quantized weights, M=1 decode, batching, permutations, and additional GGML ops remain unsupported here until each gets its own external gate.
