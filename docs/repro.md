# Reproducing RK3588 Rocket/Rust MatMul milestones

Run on the RK3588 host with `/dev/accel/accel0` accessible to the current user.

## Rust unit/ABI gates

```sh
cargo test --manifest-path /home/kautism/rocknpu/Cargo.toml --workspace
```

Expected: `rocket-uapi` ABI layout test plus `rocknpu-regcmd` generic-encoder/golden/layout/shape-limit tests pass.

## Pure Rust hardware gate

```sh
cargo run --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke
```

Expected summary:

```text
M4-K32-N16 PASS ... regcmd_count=126
M12-K32-N16 PASS ... regcmd_count=126
M16-K64-N32 PASS ... regcmd_count=126
M32-K128-N48 PASS ... regcmd_count=126
M64-K256-N64 PASS ... regcmd_count=126
M256-K512-N128 PASS ... regcmd_count=126
PASS: generic pure-Rust RK3588 fp16 MatMul encoder + Rocket UAPI + 6 real NPU shapes + exact CPU compare
```

The gate is exact: deterministic small-integer inputs are accumulated in CPU FP32, narrowed to FP16, then compared bit-for-bit with every NPU output element. Shape-tagged command/input/weight/output/reference evidence is written under `artifacts/`.

## Generic encoder byte gate against the public reference

The validation-only helpers compare the Rust stream with the pinned GPL reference generator using identical sentinel IOVAs:

```sh
cc -O2 -std=gnu11 -I reference/rocket-userspace/include \
  artifacts/dump_fp16_shape.c reference/rocket-userspace/build-ref/librocketnpu.a \
  -lm -ldrm -o artifacts/dump_fp16_shape
cargo build -p rocknpu-regcmd --example dump_fp16
```

The six hardware-gate shapes must each produce 126 identical 64-bit command words in both dumpers. The C helper is validation/reference material, not production Rust code.

## Tiled hardware gates

The first M-tiled gate exceeds the one-task CBUF envelope and must produce two NPU tasks:

```sh
cargo run --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin tiled
```

Expected tail:

```text
shape=M512 K512 N128
Mt=256 Kt=512 Nt=128
m_tiles=2 k_tiles=1 n_tiles=1 tasks=2
PASS: tiled pure-Rust RK3588 fp16 MatMul M512 K512 N128; tasks=2; mismatches=0
```

The K-split host-accumulation baseline is:

```sh
cargo run --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin k_tiled
```

Expected planning: `M64/K4096/N64 -> Kt=1536 -> 1536+1536+1024`. Every NPU partial must bit-match the CPU partial and host accumulation must match the tiled CPU oracle.

The NPU EW/ERDMA K-accumulation gate is:

```sh
cargo run --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin kacc
```

Expected tail:

```text
tile=0 ... mode=plain  ... PASS
tile=1 ... mode=ew-add ... PASS
tile=2 ... mode=ew-add ... PASS
PASS: NPU EW K-accumulation M64 K4096 N64; Kt=1536; tasks=3; staged_mismatches=0
```

This gate intentionally uses two output BOs in ping-pong order. Do not change it to in-place EW accumulation.

## EW accumulation byte gate against the public reference

```sh
cc -O2 -std=gnu11 -I reference/rocket-userspace/include \\
  artifacts/dump_fp16_accum_shape.c reference/rocket-userspace/build-ref/librocketnpu.a \\
  -lm -ldrm -o artifacts/dump_fp16_accum_shape
cargo build -p rocknpu-regcmd --example dump_fp16_accum
```

Validated accumulation shapes are `16x32x16`, `64x256x64`, `64x1536x64`, `128x512x128`, and `256x512x128`; all must produce 126 command words byte-identical to the public C generator.


## Reusable single-core executor gates

The consolidated executor owns planner-driven packing, submission, K-accumulation policy, and row-major gather:

```sh
cargo run --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin executor
```

Expected cases include:

```text
n-only PASS M64 K512 N272 ... jobs=2 ... mismatches=0
mnk-ragged PASS M300 K512 N272 ... jobs=8 npu_kacc_groups=4 ... mismatches=0
tiny-m-host-kacc PASS M4 K4096 N64 ... jobs=2 host_kacc_groups=1 ... mismatches=0
PASS: reusable single-core fp16 executor hardware gate; cases=3
```

The deterministic pseudo-random hardware differential sweep is:

```sh
cargo run --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin executor_sweep
```

It must pass two independent seeds for boundary/N-tail cases and the mixed-policy case:

```text
mixed-kacc-tail seed=0x5eed5eed PASS M260 K1024 N272   Mt=256 Kt=384 Nt=256 tiles=12 jobs=12 npu_kacc=2 host_kacc=2 mismatches=0
PASS: executor deterministic randomized hardware differential sweep; cases=5
```

## FP32-output byte gate and accuracy gate

The FP32-output encoder is a five-register variant of the validated FP16 stream. Build the validation-only reference helper and Rust dumper:

```sh
cc -O2 -std=gnu11 -I reference/rocket-userspace/include \
  artifacts/dump_fp16_f32out_shape.c reference/rocket-userspace/build-ref/librocketnpu.a \
  -lm -ldrm -o artifacts/dump_fp16_f32out_shape
cargo build -p rocknpu-regcmd --example dump_fp16_f32out
```

Validated byte-identical shapes are `16x32x16`, `64x256x64`, `128x512x128`, `256x384x128`, and `44x128x16`. Each stream has 126 command words.

Run the real accuracy gate:

```sh
cargo run --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin fp32out
```

Observed on this RK3588:

```text
M64 K4096 N128, Mt=64 Kt=1024 Nt=128, ktiles=4
FP16-output max abs error = 0.893799
FP32-output max abs error = 0.000183
FP32 normalized error     = 0.00000014
improvement               = 4881.33x
PASS
```

`execute_f32` emits FP32 NPU partials (`C2=4`) and accumulates K partials on the host in FP64; it does not depend on an unvalidated FP32 EW mode.

## Scratch reuse and warmed release benchmark

The executor owns grow-only reusable scratch BOs. The normal executor gate now repeats the tiny-M shape and must report `scratch_reuse=PASS`; the second same-shape call must not increase the BO allocation counter.

Phase timing and representative warmed release measurements are produced by:

```sh
cargo run --release --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin bench
```

The benchmark checks deterministic repeated outputs and zero post-warm allocations. Evidence is written to `artifacts/executor-bench.txt`. Benchmark binaries now report devfreq state dynamically: stock Rocket reports devfreq unavailable, while an experimental DVFS driver reports `cur/target/max/governor`. Historical stock examples after safe block/bytemuck packing were about `52 GFLOP/s` end-to-end for `M64/K4096/N512` and about `77 GFLOP/s` for `M256/K1024/N256`; controlled-clock results are documented separately below.

## RK3588 multicore scheduling and persistent pool

One DRM fd does not expose true three-core fan-out: its scheduling entity serializes queued work onto one core. The multicore gate therefore gives every worker its own Rocket fd, BOs, and executor. It preconditions the NPU first, samples worker counts in interleaved order, and uses medians to avoid governor-ramp artifacts:

```sh
cargo run --release --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin multicore
```

Current expected evidence is approximately:

```text
M256 K384 N256, 40 calls/worker
1 worker   ~32 aggregate GFLOP/s
2 workers  ~63 aggregate GFLOP/s   ~1.9-2.0x
3 workers  ~94 aggregate GFLOP/s   ~2.9x
PASS
```

Do not use the discarded cold ordered run that briefly reported >3x scaling; that was a governor/clock-ramp artifact. `artifacts/multicore-probe.txt` contains the preconditioned/interleaved evidence.

The persistent API gate is:

```sh
cargo run --release --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin pool
```

It creates a long-lived `Fp16MatmulPool` with three worker threads/fds, shares A/B through `Arc<[f16]>`, partitions N on 16-channel boundaries, and gathers row-major output. `M256/K384/N768` must match the CPU reference bit-for-bit on both the first and repeated call; worker scratch allocation counts must remain unchanged on the repeat. Repeated warmed runs observed `5.334-5.653 ms` with one worker vs `1.906-2.138 ms` with three (`2.64-2.80x`). Evidence is `artifacts/pool-run.txt`.


## Resident/prepacked weight gates

Static B weights can be packed once into a resident Rocket BO and reused without touching B on every inference call:

```sh
cargo run --release --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin prepacked
cargo run --release --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin prepacked_bench
```

`prepacked` covers single-tile, simultaneous ragged M/N/K, deep-K EW accumulation, and tiny-M host accumulation. Every result must bit-match `cpu_reference_executor_semantics`, and the repeated call must not increase the scratch allocation counter. `M300/K512/N272` has eight compute jobs but only four unique resident N/K weight tiles because M-axis duplicates are deduplicated.

`prepacked_bench` warms both streaming and resident paths and excludes the one-time prepack cost from repeated-call timing while reporting it separately. Representative observations showed resident weights improving deep-K and prefill workloads by roughly `1.2-1.5x` at the stock 200 MHz-class configuration, with larger packing-phase reductions.

The resident multicore characterization is:

```sh
cargo run --release --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin multicore_prepacked
```

Every worker owns a separate Rocket fd and its own resident B BO. Correctness and zero-post-warm-allocation checks remain enabled.

## Controlled NPU-frequency characterization (experimental reference only)

Stock Armbian Rocket has no `/sys/class/devfreq/fdab0000.npu` node on this host. The live NPU DT nodes carry a 200 MHz assigned clock and no NPU OPP table. To characterize the silicon without changing the project kernel policy, a public GPL-2.0 Rocket DVFS research tree was cloned at `reference/rk3588-npu-gpu`, commit `ed52a89afa8e68fedf636c8e891bd8fc47e82d26`, built against exact `6.18.43-current-rockchip64` headers, loaded temporarily, then removed. It is not production project code.

Safety conditions used for this experiment:

- no DTB or boot-service changes,
- original `vdd_npu_s0 = 800 mV` left unchanged,
- userspace governor only, with `max_freq` capped to the requested point,
- tested only 200, 600, and 700 MHz; no >700 MHz point because the research driver requires a higher-voltage guard above 700 MHz,
- exact resident hardware gate run before performance measurement at 600 and 700 MHz,
- custom driver lowered to 200 MHz before unload; packaged stock Rocket restored and exact gate rerun afterward.

The focused phase-separated probe is:

```sh
cargo run --release --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin freq_probe
```

With the appropriate externally controlled frequency, `M256/K512/N128`, one resident job, 41 repetitions produced:

```text
200 MHz  wait=0.399286 ms                 wait-effective=84.04 GFLOP/s
600 MHz  wait=0.181122-0.186955 ms        wait-effective=179.48-185.26 GFLOP/s
700 MHz  wait=0.125123-0.130957 ms        wait-effective=256.22-268.17 GFLOP/s
```

Do not infer clock scaling from total executor wall time alone: CPU pack/gather and scheduler noise can move independently. The fence-wait phase is the preferred clock-scaling evidence. `artifacts/dvfs-fp16-summary.txt` records the experiment. After reproduction, restore stock Rocket; normal project tests do not require the external DVFS module.


## Project-owned tensor/op contract gate

The high-level MatMul boundary can be tested without exposing Rocket commands to the caller:

```sh
cargo test --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocknpu-tensor -p rocknpu-ops
cargo run --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin ops_contract
```

The gate requires all of the following:

```text
ops single-fp16 PASS shape=64x256x64
ops single-fp32 PASS shape=64x256x64 max_abs_error=0
ops pool-fp16 PASS shape=64x256x96 workers=3
ops auto-cpu-fallback PASS shape=3x7x5
PASS: project-owned tensor/op MatMul contract hardware gate
```

`Auto` must not pad an unsupported shape: the unaligned case proves it routes to the independent `rocknpu-ops` CPU implementation. `NpuPool + Fp32Accurate` remains intentionally unsupported until that path has its own hardware gate.

## C/open reference UAPI gate

Pinned source: `reference/rocket-userspace` at commit `1a181cd69fd98fafa998cfeca963d35cb5f43c46`.

```sh
cmake -S reference/rocket-userspace -B reference/rocket-userspace/build-ref -G Ninja -DROCKETNPU_BUILD_TESTS=ON
cmake --build reference/rocket-userspace/build-ref --target uapi_selftest_rocket matmul_fp16_rocket -j 4
reference/rocket-userspace/build-ref/uapi_selftest_rocket
ROCKET_TEST_SEED=0x3588 reference/rocket-userspace/build-ref/matmul_fp16_rocket 4 32 16
```

Observed on this host: UAPI self-test `14 checks, 0 failed`; deterministic FP16 MatMul `OK: [4,32]x[16,32]`.


## First ONNX hybrid model gate

The first model-format gate is a real serialized ONNX model, not an in-memory graph:

```sh
cargo run --release --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin tiny_model
```

Expected graph:

```text
input [4,32]
MatMul -> Add -> Relu -> MatMul -> Add
output [4,16]
```

The two MatMul nodes run through the stock-Rocket RK3588 NPU backend; Add and Relu run on CPU. The binary compares against the same graph forced to CPU and requires `mismatches=0`. It writes `artifacts/tiny-mlp.onnx`, CPU/NPU FP16 output binaries, and `artifacts/tiny-mlp-run.txt`.

Independent format validation:

```sh
python3 -c "import onnx; m=onnx.load('/home/kautism/rocknpu/artifacts/tiny-mlp.onnx'); onnx.checker.check_model(m); print('PASS')"
```

Observed: Python ONNX 1.17 checker PASS, IR 9, opset 13, operations `MatMul/Add/Relu/MatMul/Add`, input `[4,32]`, output `[4,16]`. The Rust ONNX dependency uses `onnx-protobuf 0.2.3`; protobuf is intentionally pinned to exactly `3.4.0` because that generated crate performs a compile-time protobuf-version check.

## Environment evidence

```sh
cat artifacts/environment.txt
cat artifacts/rust-run.log
```

Do not treat a successful compile as the hardware gate. `rocket-smoke` must run on RK3588 and report zero mismatches.

## Independent tiny-model numerical verification

After `cargo run --release -p rocket-smoke --bin tiny_model`, run:

```sh
python3 scripts/verify_tiny_model.py
```

Expected: all five ONNX node outputs report `fp16_bits=True`, `fp32_exact=True`, `max_abs=0`, followed by `PASS: ONNX ReferenceEvaluator == NumPy node trace == RK3588 NPU, max_abs_error=0`. This verifier is intentionally outside the Rust importer/backend path.


## External pretrained MNIST MLP gate

Source model: Pico-CNN `data/mnist_mlp/mnist_mlp.onnx` (BSD-3-Clause), SHA-256 `967612db6a1724e85101d5e11aaed3322d7d52ddd65f9af910fa8c71cf88c7cd`. Canonical MNIST test data comes from the CVDF mirror of the original MNIST dataset. Pico-CNN's example normalizes input pixels to `[0,1]`; the gate uses the first 50 test images, matching the model's declared batch.

After fetching the model/data and producing the independent ONNX reference artifacts, run:

```sh
cargo run --release --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin real_mnist
python3 /home/kautism/rocknpu/scripts/verify_real_mnist.py
```

Expected hardware evidence on stock Rocket:

```text
model_nodes=7
gemm_npu=4
padded_npu=4
relu_cpu=3
ref_pred_match=50/50
ref_accuracy=49/50
npu_accuracy=49/50
ref_npu_max_abs=0.056854248
ref_npu_mean_abs=0.011868
```

The independent verifier must report seven trace nodes, NumPy final output exactly equal to the saved ONNX ReferenceEvaluator FP32 output, and `npu_top1_vs_ref=50/50`. Current per-node maximum absolute FP32-reference errors are approximately `0.01724, 0.00622, 0.01764, 0.01753, 0.06614, 0.04184, 0.05685`. These are FP16-lowering errors; top-1 must remain identical for this gate.


## Prepared pretrained MNIST model-session gate

Prepare the four Gemm weights once into resident Rocket BOs, drop the parsed ONNX model, and repeatedly execute using only the prepared session plus activation input:

```sh
cargo run --release --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin prepared_mnist
```

Current accepted evidence is approximately:

```text
prepared MNIST PASS
dense=4 padded=4
resident_MB=3.218 resident_tiles=21
prepare_wall_ms=3.52-3.56 weight_pack_ms=1.42-1.44
streaming_median_ms=6.53-6.56
prepared_median_ms=5.02-5.10
speedup=1.28-1.31x
scratch_allocs=5 scratch_grows=0
top1_ref=50/50 accuracy=49/50
bit_identical_streaming=true
model_dropped_before_prepared_runs=true
```

Then independently validate every prepared-session node against the original FLOAT ONNX graph:

```sh
python3 scripts/verify_real_mnist.py prepared-mnist-npu
```

Expected final lines include `numpy_final_vs_onnx_reference_exact=True`, `npu_top1_vs_ref=50/50`, and `PASS`. The per-node error profile must match the accepted streaming path; the current final Gemm is `max_abs=0.05685425`, `mean_abs=0.01186811`. Evidence files are `artifacts/prepared-mnist-run.txt`, `artifacts/prepared-mnist-npu-trace.tsv`, and `artifacts/prepared-mnist-npu-trace-f16.bin`.

This gate proves that resident/prepacked static weights survive after the parsed source model is dropped and that warmed inference performs no new executor scratch allocation. It does not yet prove reuse across a different batch/M geometry; that is the next residency milestone.


## Dynamic-batch resident MNIST gate

Prepare one M-compatible resident model and reuse exactly the same static BOs for several batch sizes:

```sh
cargo run --release --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin dynamic_batch_mnist
python3 scripts/verify_real_mnist.py dynamic-mnist-npu
```

Accepted stock-Rocket evidence:

```text
dynamic MNIST PASS batches=[1,4,16,50]
dense=4 m_compatible=4 resident_MB=3.213 resident_tiles=31
batch 1/4/16/50 top1_ref = 1/1, 4/4, 16/16, 50/50
post-max-M-warm scratch allocation/growth unchanged
batch50 final max_abs=0.08248138 mean_abs=0.01110571
```

The larger error than fixed-M preparation is an accepted consequence of the conservative M-invariant K tiling, not a license to skip the independent oracle.

## Worker-local resident pool gates

First validate the pool resident protocol directly:

```sh
cargo run --release --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin pool_prepared
```

Expected current evidence for `M256/K384/N768`: three worker-local resident copies, exact CPU-oracle results at `M64/128/256`, no post-warm worker scratch growth, explicit release PASS, and roughly `1.11x` resident-vs-streaming warmed improvement on this stock run.

Then run the full prepared pool model:

```sh
cargo run --release --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin pool_model_mnist
python3 scripts/verify_real_mnist.py pool-mnist-npu
```

Accepted evidence:

```text
pool model MNIST PASS workers=3 dense=4
resident_copies=10 resident_MB=3.217 resident_tiles=37
model_dropped=true release=true
batch 1/4/16/50 top1_ref = 1/1, 4/4, 16/16, 50/50
batch50_median_ms=5.343
independent final max_abs=0.06208801 mean_abs=0.01178154
```

Do not infer that three workers should be Auto-selected for this model: the full-model median is not materially better than the single-worker prepared path.


## Persistent FP32 pool gate

```sh
cargo run --release --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin pool_fp32
cargo run --release --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin ops_contract
```

Expected characteristics: integer `M64/K256/N96` is exact; deterministic fractional `M64/K4096/N192` stays near `1e-7` normalized RMS versus the f64 CPU reference, repeated output is stable, and scratch allocation/grow counters do not change after the deep-shape warmup. `ops_contract` must also report `ops pool-fp32 PASS` through the public `PoolNpuBackend` API.

## Adaptive streaming Auto policy gate

```sh
cargo run --release --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin auto_worker_probe
cargo run --release --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin auto_tuned
```

`auto_worker_probe` is characterization evidence, not a fixed expected winner table. `auto_tuned` must show that each aligned shape receives a cached 1/2/3-worker choice, a repeated call reuses the cache, FP16/FP32 differential thresholds pass, and an unaligned Auto request falls back to CPU without changing cache size.

## Resident lifecycle / low-4-GiB stress

```sh
cargo run --release --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin lifecycle_stress
```

Current bounded gate alternates warmed shapes for 128 executions, performs 128 resident prepare/drop cycles, holds 96 `K1024/N512` resident copies concurrently (about 100.7 MB), drops them, reallocates and executes again, then performs 48 pool prepare/release cycles and requires a released handle to be rejected. Every resident DMA range must remain below `u32::MAX`.


## Official MNIST-8 CNN gate

Source model: ONNX Model Zoo / `onnxmodelzoo/mnist-8`, SHA-256 `2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf`. It is the pretrained CNTK MNIST CNN published by the model zoo (IR 3, opset 8). The gate uses the first 100 canonical MNIST test images normalized to `[0,1]`.

First validate the project-owned FP16 Conv lowering and the two real model layers:

```sh
cargo run --release --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin conv_fp16
cargo run --release --manifest-path /home/kautism/rocknpu/Cargo.toml -p rocket-smoke --bin conv_mnist8_layers
```

`conv_fp16` must report zero mismatches for `IC32/H8/W8 -> OC16`, 5x5 pad2. The real-layer gate must show Conv1 (`IC1->OC8`, padded to `IC32->OC16`) bit-exact to the FP16 CPU oracle and Conv2 (`IC8->OC16`, padded to `IC32->OC16`) within the accepted one-result / `7.63e-6` FP16 rounding difference.

Then run the full graph with both Conv nodes and the final MatMul on NPU:

```sh
cargo run --release --manifest-path /build/rocknpu/Cargo.toml -p rocket-smoke --bin mnist8_cnn_npu
python3 /build/rocknpu/scripts/verify_mnist8.py mnist8-allnpu
```

Accepted stock-Rocket evidence:

```text
MNIST-8 CNN ALL-NPU-COMPUTE PASS nodes=12
conv_nodes=2 conv_npu=2 pool_cpu=2 reshape_cpu=2 add_cpu=3 relu_cpu=2
matmul_npu=1 padded_npu=1
top1_ref=100/100
ref_accuracy=98/100 npu_accuracy=98/100
max_abs=0.02100563 mean_abs=0.00299615
```

The Python verifier must report `onnx_checker=PASS node_count=12 trace_count=12`, compare every intermediate output requested directly from ONNX `ReferenceEvaluator`, and finish with `final_top1_match 100 /100` plus PASS. Current sample-0 maximum errors include first NPU Conv `0.00177240`, second NPU Conv `0.00469589`, NPU MatMul `0.00604439`, and final Add `0.01325989`. Across all 100 samples the final maximum is `0.02100563`.

Prepared Conv residency has separate low-level and full-graph gates:

```sh
cargo run --release --manifest-path /build/rocknpu/Cargo.toml -p rocket-smoke --bin conv_prepared
cargo run --release --manifest-path /build/rocknpu/Cargo.toml -p rocket-smoke --bin mnist8_cnn_prepared
python3 /build/rocknpu/scripts/verify_mnist8.py mnist8-prepared
cargo run --release --manifest-path /build/rocknpu/Cargo.toml -p rocket-smoke --bin conv_prepared_bench
cargo run --release --manifest-path /build/rocknpu/Cargo.toml -p rocket-smoke --bin mnist8_conv_bench
```

The prepared model must report two resident Conv tensors / 51,200 bytes, Conv scratch `weight_bytes=0`, no post-first-inference scratch growth, the same 100/100 top-1/reference trace result, and bit-identical streaming/prepared layer outputs. Representative warmed layer medians were `0.2080 -> 0.1945 ms` and `0.1207 -> 0.1070 ms`; a 40-inference block full-model comparison measured `0.9626 -> 0.8066 ms` (`1.194x`). Treat single sub-millisecond samples as scheduler-noisy and prefer the block result.

## Edge-infer CIFAR-10 pretrained model gate

Run the real RGB pretrained model on stock RK3588 Rocket:

```sh
cargo run --release --manifest-path /build/rocknpu/Cargo.toml -p rocket-smoke --bin cifar10_edgeinfer
```

Then independently validate all 13 intermediate tensors and final prediction against the original ONNX model:

```sh
cd /build/rocknpu
python3 scripts/verify_cifar10_edgeinfer.py
```

Expected high-level result: three Conv and two Gemm nodes execute on NPU, prediction is class 8 (`ship`), each trace node remains within the verifier's ONNX-reference bound, CPU/NPU cross-backend max absolute difference is <= `0.002`, and final FP16 logits are bit-identical. The current measured cross-backend maximum is `0.00097656`.

## High-level `rocknpu::Session` hardware gate

Run the public Session API against the same real models from the formal build workspace:

```sh
cd /build/rocknpu
cargo run -p rocket-smoke --bin session_models
```

This gate constructs each model through `rocknpu::Session::load`, so ONNX parsing and supported static weight preparation happen once before inference. It then exercises repeated `Session::run` calls without exposing `RocketDevice`, IOVA, register commands, or executor internals to the caller.

Current accepted RK3588 evidence:

```text
MNIST-8:
  eager resident weights = 2 Conv + 1 dense, 59,392 bytes
  NPU placement = 2 Conv + 1 MatMul
  top1 vs saved ONNX reference = 100/100
  accuracy = 98/100
  final max_abs = 0.02100563

edge-infer CIFAR-10:
  eager resident weights = 3 Conv + 2 dense, 104,448 bytes
  NPU placement = 3 Conv + 2 Gemm
  prediction = 8 (ship), reference = 8
  final max_abs = 0.00475883
```

The Session gate checks final outputs against saved independent FP32 reference artifacts and checks the execution statistics/placement contract. It does not replace the trace-level standard oracle. After changing Session/frontend/backend behavior, also run the existing independent verifiers:

```sh
cargo run -p rocket-smoke --bin mnist8_cnn_prepared
python3 scripts/verify_mnist8.py mnist8-prepared
cargo run -p rocket-smoke --bin cifar10_edgeinfer
python3 scripts/verify_cifar10_edgeinfer.py
cargo run -p rocket-smoke --bin prepared_mnist
python3 scripts/verify_real_mnist.py prepared-mnist-npu
```

All generated evidence remains under `/build/rocknpu/artifacts/` and is gitignored.

## High-level `rocknpu run` CLI gate

The CLI uses standard C-order `float32` NumPy `.npy` files and the same Session runtime as the Rust API. Build the two checked input files from the existing canonical artifacts:

```sh
cd /build/rocknpu
python3 -c "import numpy as np; x=np.fromfile('artifacts/mnist8-input100-f32.bin',dtype='<f4').reshape(100,1,28,28); np.save('artifacts/cli-mnist8-input.npy',x[:1]); x=np.fromfile('artifacts/cifar10-edgeinfer-input-f32.bin',dtype='<f4').reshape(1,3,32,32); np.save('artifacts/cli-cifar10-input.npy',x)"
```

First validate the file/CLI contract without NPU dependence:

```sh
cargo run -p rocknpu -- run artifacts/mnist-8.onnx --input artifacts/cli-mnist8-input.npy --output artifacts/cli-mnist8-cpu.npy --target cpu
cargo run -p rocknpu -- run artifacts/cifar10-edgeinfer.onnx --input artifacts/cli-cifar10-input.npy --output artifacts/cli-cifar10-cpu.npy --target cpu
```

Then run the identical interface on Rocket/NPU:

```sh
cargo run -p rocknpu -- run artifacts/mnist-8.onnx --input artifacts/cli-mnist8-input.npy --output artifacts/cli-mnist8-npu.npy
cargo run -p rocknpu -- run artifacts/cifar10-edgeinfer.onnx --input artifacts/cli-cifar10-input.npy --output artifacts/cli-cifar10-npu.npy
```

Current accepted debug-build RK3588 evidence:

```text
MNIST-8 CLI NPU:
  resident_bytes=59392
  npu_conv=2 npu_dense=1
  output shape=(1,10)
  max_abs vs saved FP32 reference=0.0132598877

CIFAR-10 CLI NPU:
  resident_bytes=104448
  npu_conv=3 npu_dense=2
  output shape=(1,10)
  prediction=8 (ship)
  max_abs vs saved FP32 reference=0.0047588348
```

Finally prove NumPy can read the RockNPU-produced files and that top-1 remains equal to the saved references:

```sh
python3 -c "import numpy as np; a=np.load('artifacts/cli-mnist8-npu.npy'); r=np.fromfile('artifacts/mnist8-ref100-f32.bin',dtype='<f4').reshape(100,10)[:1]; assert a.dtype==np.float32 and a.shape==(1,10) and a.argmax(1)[0]==r.argmax(1)[0]; a=np.load('artifacts/cli-cifar10-npu.npy'); r=np.fromfile('artifacts/cifar10-edgeinfer-ref-f32.bin',dtype='<f4').reshape(1,10); assert a.dtype==np.float32 and a.shape==(1,10) and a.argmax(1)[0]==r.argmax(1)[0]==8; print('PASS: rocknpu CLI NumPy round-trip + RK3588 inference')"
```

This gate proves the documented end-user ONNX command path itself, rather than only the lower-level smoke binaries. It does not expand the ONNX operator set or claim general NumPy dtype/layout support.

## TinyLlama GGUF autoregressive hardware gates

The first real LLM gate uses `TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf`. The validated artifact is 667,814,880 bytes and can be fetched from the public second-state TinyLlama GGUF mirror:

```sh
cd /build/rocknpu
curl -L --fail --retry 3 \
  -o artifacts/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf \
  https://huggingface.co/second-state/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf
stat -c '%s %n' artifacts/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf
```

Expected size:

```text
667814880 artifacts/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf
```

Run the RockNPU first-token path on the real RK3588 NPU:

```sh
cargo run -p rocket-smoke --bin llm_gguf -- \
  artifacts/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf Hello npu
```

Accepted model metadata and result:

```text
vocab_size=32000 hidden_size=2048 intermediate_size=5632
layers=22 heads=32 kv_heads=4 head_dim=64 context=2048
rope_theta=10000 rope_style=Normal
LLM GGUF FIRST TOKEN PASS target=npu prompt_tokens=2 padded_tokens=4 token_id=29892 text="," npu_linears=154 cpu_linears=1
```

The 154 NPU Linears are exactly seven projections across each of 22 transformer blocks. The current LM head is the one CPU Linear because M=1 decode/GEMV has not yet been optimized for the NPU. Quantized GGUF layer weights are converted to FP16 before preparation; only one block's prepared projection weights need to be resident at a time.

First compare with the separate `llama-gguf` CPU model implementation:

```sh
cargo run -p rocknpu-llm --example gguf_reference -- \
  artifacts/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf Hello
```

Expected:

```text
LLAMA-GGUF REFERENCE FIRST TOKEN prompt_tokens=2 token_id=29892 text=","
```

A stronger oracle uses an external llama.cpp checkout outside the RockNPU tree. The accepted run used llama.cpp commit `391fac16460f15233a7740550d858ac96df3419d`; it is validation material, not a RockNPU dependency:

```sh
git clone https://github.com/ggml-org/llama.cpp.git /build/llama.cpp-reference
git -C /build/llama.cpp-reference checkout 391fac16460f15233a7740550d858ac96df3419d
cmake -S /build/llama.cpp-reference -B /build/llama.cpp-reference/build \
  -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF
cmake --build /build/llama.cpp-reference/build --target llama-completion -j 8
/build/llama.cpp-reference/build/bin/llama-completion \
  -m /build/rocknpu/artifacts/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf \
  -no-cnv -p Hello -n 1 --temp 0 --no-warmup --verbose-prompt --log-verbosity 0
```

The raw completion is `Hello,`, so the first generated token text is again `","`. `-no-cnv` is required: allowing the model's chat template changes the prompt semantics and is not the same gate.

### Autoregressive KV-cache gate

The runtime now retains one K/V cache per transformer layer and performs incremental one-token decode. During prefill, only the real prompt rows are cached even when end-padding is added to make the NPU MatMul row count legal. After each layer's prefill, its Rocket resident-weight BOs are released while the one-time dequantized FP16 matrices are retained for the current CPU M=1 decode path.

Use a high-margin sequence to make exact cross-runtime greedy comparison meaningful:

```sh
cargo run --release -p rocket-smoke --bin llm_gguf -- \
  artifacts/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf \
  "1, 2, 3, 4, 5, 6, 7, 8," npu 32
```

Accepted RK3588 result:

```text
prompt_tokens=25 padded_tokens=28
text=" 9, 10, 11, 12, 13, 14, 15, 16, "
prefill_npu=154 prefill_cpu=0
decode_npu=0 decode_cpu=4774 lm_head_cpu=32
```

The exact 32 generated token IDs are:

```text
[29871, 29929, 29892, 29871, 29896, 29900, 29892, 29871,
 29896, 29896, 29892, 29871, 29896, 29906, 29892, 29871,
 29896, 29941, 29892, 29871, 29896, 29946, 29892, 29871,
 29896, 29945, 29892, 29871, 29896, 29953, 29892, 29871]
```

Run the independent Rust CPU model with the same prompt and token budget:

```sh
cargo run --release -p rocknpu-llm --example gguf_reference -- \
  artifacts/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf \
  "1, 2, 3, 4, 5, 6, 7, 8," 32
```

It must reproduce all 32 IDs and the same decoded string. The separately built llama.cpp raw-completion oracle must also produce the same sequence; the validated numeric prompt has large top-1/top-2 margins throughout, making it suitable for an exact greedy gate.

A semantic prompt is also exercised:

```text
The capital of France is -> " Paris.\n\n2. B."
```

RockNPU, `llama-gguf`, and llama.cpp agree on the first eight generated tokens. Extending that particular greedy sequence exposes a useful precision boundary at token 9: llama.cpp's native Q4_K path ranks token `29907` (`C`) above token `315` (` C`) by about `0.125`, while the FP16-dequantized Rust reference ranks `315` above `29907` by about `0.040`. Once either near-tied token is chosen, later greedy history naturally diverges.

Therefore exact greedy token identity is a hard cross-runtime requirement only on numerically stable reference steps. Near-tied candidates must be investigated with logits and reported as numerical precision divergence rather than mislabeled as a K/V-state failure. The project also keeps a unit differential where cached single-token decode matches full-sequence recomputation for the same transformer block.

This remains a correctness milestone rather than a performance claim. Q4_K_M weights are currently converted to FP16, prefill uses the NPU, and M=1 projection/LM-head decode stays on CPU. The next performance work is native quantized execution and a validated NPU GEMV/decode path.

## Stock GGML dynamic-backend gate

RockNPU can be loaded by an unmodified llama.cpp build as an out-of-tree GGML backend. The validated external ABI is llama.cpp commit `391fac16460f15233a7740550d858ac96df3419d`; GGML ABI types remain confined to `adapters/ggml-rocknpu`.

Build the adapter and its Rust C ABI sidecar:

```sh
cmake -S adapters/ggml-rocknpu \
  -B target/ggml-rocknpu \
  -DGGML_SOURCE_DIR=/build/llama.cpp-reference/ggml
cmake --build target/ggml-rocknpu -j 8
```

First prove stock llama.cpp discovers the real Rocket-backed device:

```sh
GGML_BACKEND_PATH="$PWD/target/ggml-rocknpu/libggml-rocknpu.so" \
  /build/llama.cpp-reference/build/bin/llama-cli --list-devices
```

Expected device:

```text
ROCKNPU0: RockNPU RK3588
```

Then run one deliberately narrow stock GGML correctness test:

```sh
GGML_BACKEND_PATH="$PWD/target/ggml-rocknpu/libggml-rocknpu.so" \
  /build/llama.cpp-reference/build/bin/test-backend-ops \
  test -b ROCKNPU0 -o MUL_MAT \
  -p type_a=f16,type_b=f32,m=16,n=4,k=256
```

Accepted result:

```text
MUL_MAT(type_a=f16,type_b=f32,m=16,n=4,k=256,...): OK
1/1 tests passed
Backend ROCKNPU: OK
```

The test data and CPU reference are owned by stock llama.cpp. The RockNPU path is `GGML -> libggml-rocknpu.so -> rocknpu-capi -> Fp16MatmulExecutor -> RocketDevice -> RK3588 NPU`.

### Q4_K / Q6_K and real TinyLlama through stock GGML

For ordinary `GGML_BACKEND_PATH` auto-loading, configure the same unmodified llama.cpp source with its dynamic-backend option enabled:

```sh
cmake -S /build/llama.cpp-reference \
  -B /build/llama.cpp-reference/build-dl \
  -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON
cmake --build /build/llama.cpp-reference/build-dl \
  --target llama-completion test-backend-ops -j 8
```

The older reference build used `GGML_BACKEND_DL=OFF`; with a statically registered CPU backend, ordinary `llama_backend_init()` does not call `ggml_backend_load_all()`. That build remains valid for tools such as `test-backend-ops` that explicitly load all backends, but `build-dl` is the clean external-plugin reproduction path.

Run the stock Q4_K oracle:

```sh
GGML_BACKEND_PATH="$PWD/target/ggml-rocknpu/libggml-rocknpu.so" \
  /build/llama.cpp-reference/build-dl/bin/test-backend-ops \
  test -b ROCKNPU0 -o MUL_MAT -p q4_K
```

Accepted result: `6/6 tests passed`, `Backend ROCKNPU: OK`. M=1/non-multiple-of-4 rows, batched/permuted layouts, and F16 activations are explicitly not supported.

Run the stock Q6_K oracle:

```sh
GGML_BACKEND_PATH="$PWD/target/ggml-rocknpu/libggml-rocknpu.so" \
  /build/llama.cpp-reference/build-dl/bin/test-backend-ops \
  test -b ROCKNPU0 -o MUL_MAT -p q6_K
```

Accepted result: `3/3 tests passed`, `Backend ROCKNPU: OK`.

The Q4_K/Q6_K bridge forwards raw GGML quantized blocks into Rust, dequantizes there, converts to RockNPU's existing FP16 weight contract, then runs the real Rocket/NPU MatMul. It is correctness-first and does not claim native quantized-kernel performance.

For a real-model gate, first run stock CPU with the four-token prompt `The capital of`; greedy generation produces `" the"`. Then run:

```sh
ROCKNPU_GGML_TRACE=1 \
GGML_BACKEND_PATH="$PWD/target/ggml-rocknpu/libggml-rocknpu.so" \
  /build/llama.cpp-reference/build-dl/bin/llama-completion \
  -fit off -ngl 0 \
  -m artifacts/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf \
  -no-cnv -p "The capital of" -n 1 --temp 0 --no-warmup --verbose-prompt
```

The RockNPU run produces the same `" the"` continuation. Opt-in execution tracing at the actual backend boundary reports 131 Q4_K plus 20 Q6_K MatMuls:

```text
ROCKNPU GGML TRACE mul_mat weight=blk.0.attn_q.weight type=q4_K M=4 K=2048 N=2048
ROCKNPU GGML TRACE mul_mat weight=blk.0.attn_v.weight type=q6_K M=4 K=2048 N=256
...
ROCKNPU GGML TRACE summary q4_K_mul_mat=131 q6_K_mul_mat=20 f16_mul_mat=0
```

This is all 151 block-projection MatMuls that stock llama.cpp presents with the NPU-eligible `M=4` shape in this gate. Output pruning reduces the final layer's `ffn_gate`, `ffn_up`, and `ffn_down` plus `output.weight` to `M=1`; those remain on CPU/another scheduler backend. This establishes real Q4_K/Q6_K TinyLlama prefill through stock GGML into RockNPU on RK3588 without claiming M=1 NPU decode or broader GGML operator/layout coverage.
