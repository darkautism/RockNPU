# RK3588 TinyLlama decode checkpoint — 2026-09-21

## Scope

This checkpoint consolidates the M16 verifier-split work, native wider INT8 M-tile work, FFN-down K-split work, and decode-cache prewarm experiments into one self-contained research branch: `research/native-m64-prewarm-0921`.

Target workload:
- TinyLlama GGUF
- speculative decode with `--spec-draft-n-max 63`
- `-n 512`
- `-t 4 -tb 4`
- `taskset -c 4-7`
- deterministic sampling: `--temp 0 --top-k 1 --seed 1`
- model and llama binaries copied to `/tmp` tmpfs for benchmark stability
- no eMMC migration; `/build` remains the intentional NVMe/TCP workspace

No kernel source or Rocket kernel module changes are part of this checkpoint.

## Current best path

Required research toggles:

```text
ROCKNPU_RUNTIME_M16_ROUTER=1
ROCKNPU_NATIVE_MTILE=1
ROCKNPU_NATIVE_MTILE_DOWN=1
ROCKNPU_RUNTIME_M16_PREWARM=1
ROCKNPU_MTILE_MC=1
ROCKNPU_W8_MTILE=1
ROCKNPU_W8_MTILE_SCOPE=all
ROCKNPU_MTILE_PERSIST=1
ROCKNPU_PREFILL_CACHE=1
ROCKNPU_DECODE=0
```

Do not enable `ROCKNPU_MTILE_QO_GROUP` for the current best configuration.

For controlled comparison only, the benchmark temporarily fixed:
- NPU minimum frequency to 1 GHz through the existing devfreq node
- **all CPU cpufreq policies** (`policy0`, `policy4`, `policy6`) to `performance`; do not set only the A76 clusters

The benchmark command restored the previous governors afterward.

### RK3588 DSU governor trap

On this board, `policy0` is not irrelevant just because the workload is pinned to A76 cores 4-7. A controlled 2x2 check on 2026-09-21 found that changing only the A55 `policy0` governor changes the shared SCMI DSU clock and roughly doubles available memory bandwidth:

| State | NPU clock | `policy0` | `scmi_clk_dsu` | 4xA76 aggregate sequential read | native CPU `tg128` |
|---|---:|---|---:|---:|---:|
| low-DSU | 200 MHz | `ondemand` | 1.2 GHz | 12.47 GB/s | 14.62 tok/s |
| high-DSU | 200 MHz | `performance` | 1.8 GHz | 25.29 GB/s | 34.29 tok/s |
| low-DSU control | 700 MHz | `ondemand` | 1.2 GHz | - | 14.19 tok/s |
| high-DSU control | 700 MHz | `performance` | 1.8 GHz | - | 34.25 tok/s |

The NPU clock does not cause the CPU throughput change. DDR remained at 2.112 GHz and the A76 cores remained at 2.4 GHz. The performance switch is the A55 cpufreq policy indirectly raising the shared DSU/L3 path. Therefore every CPU/NPU comparison on RK3588 must set **all** CPU policies to `performance` or explicitly record and compare the DSU state; setting only the big cores invalidates the comparison.

## Verified results

All results below used the same prompt, draft limit 63, and reported 100% acceptance with `n_accept=504`.

| Path | Encode | Decode | Decode throughput |
|---|---:|---:|---:|
| CPU, controlled CPU state | 1.223 s | 9.749 s | 52.619 tok/s |
| Native M64 + FFN-down K-split, no prewarm | 1.833 s | 5.988 s | 85.666 tok/s |
| Native M64 + FFN-down K-split + prewarm, run 1 | 3.486 s | 3.448 s | 148.761 tok/s |
| Native M64 + FFN-down K-split + prewarm, repeat | 3.509 s | 4.034 s | 127.181 tok/s |

Conservative repeated **lookup-speculative verifier/decode** result: **127.181 tok/s**, about **2.42x** the measured CPU throughput under the same lookup-speculative workload. This is not a claim that ordinary M=1 autoregressive decode runs at 127 tok/s: the benchmark drafted 504 tokens in 63-token lookup batches and accepted 100% of them.

The highest observed controlled lookup-speculative result is **148.761 tok/s**, about **2.83x** the same CPU workload, but this is not yet treated as the stable floor because the repeated run was lower.

Prompt + decode total time:
- CPU: 10.972 s
- NPU no-prewarm: 7.821 s
- NPU prewarm run 1: 6.934 s
- NPU prewarm repeat: 7.543 s

Prewarm therefore moves substantial preparation work before decode, but it did not merely hide the cost: both controlled prewarm runs were still faster end-to-end than CPU, and both were faster end-to-end than the measured no-prewarm NPU run. On the repeated run, total prompt+decode time was 7.543 s versus 10.972 s for CPU, about a 1.45x end-to-end request speedup; the 2.42x figure applies to the lookup-speculative decode phase only.

A separate ordinary non-speculative `llama-bench tg128` sanity run in the speculative-checkpoint environment measured **13.79 +/- 0.85 tok/s** on RockNPU versus **10.45 +/- 0.09 tok/s** on the four-core CPU for this Q4_K_M GGUF. That CPU number is **not a valid native CPU baseline** because this checkpoint did not preserve a complete all-policy cpufreq contract; on this RK3588 board, leaving `policy0` on `ondemand` throttles the shared DSU/L3 path even when the A76 clusters are fixed at 2.4 GHz. This was a sanity measurement under the speculative-checkpoint environment, **not the best known ordinary-generation configuration**. Earlier validated M=1 work on the same project reached about **16 tok/s** with caller-thread direct submit + persistent scratch, and about **19.3-19.4 tok/s** in the research-driver configuration with per-core IOMMU-domain caching plus the validated A55 IRQ-latency tune at 700 MHz / 800 mV. The native-M64 path only applies to M=32..64 and does not accelerate ordinary M=1 generation. Therefore 13.79 must not be used as the project-wide ordinary-generation ceiling. None of these ordinary-generation figures is directly interchangeable with the 127-149 tok/s lookup-speculative verifier numbers.

## Confirmed findings

### Native M64 is real and correct

The previous M32 wrap failure was caused by RockNPU patching only the first matching `0x1040` register in the INT8 template. The template contains repeated `0x1040` writes.

Changing the research M-tile path to patch every matching `0x1040` entry made M32 and M64 pass the on-silicon bit-exact smoke test.

Latest smoke:

```text
INT8 MTILE PASS M=16 K=2048 N=2048
INT8 MTILE PASS M=32 K=2048 N=2048
INT8 MTILE PASS M=64 K=2048 N=2048
```

The merge candidate intentionally exposes only M=16/32/48/64. Wider M values were not promoted merely because the register formula accepts them.

### Native M64 removes repeated M16 orchestration

On the same adapter/runtime binary, short A/B:

- native M64: 34.696 tok/s
- 4x sequential M16: 28.652 tok/s
- improvement: about 21%
- acceptance: 100% on both

Profile delta:

- projection calls: 1056 -> 264
- wait time: 1862 ms -> 890 ms
- execute time: 946 ms -> 517 ms

Long A/B:

- native M64: 73.664 tok/s
- 4x sequential M16: 57.000 tok/s
- improvement: about 29%
- acceptance: 100%

### M64 is the useful native batch size for this workload

Measured native long runs:

- M32 / draft31: 61.661 tok/s
- M48 / draft47: 57.355 tok/s
- M64 / draft63: 73.664 tok/s

A hybrid M80 attempt using M64 + M16 fell to 25.049 tok/s and was stopped. Do not continue increasing speculative batch merely to create a larger M.

### FFN-down K=5632 is now accelerated with 3-core K-split

The previous single-core M16 FFN-down K-split experiment was slower because it allocated partial buffers and performed host accumulation on an already-small M16 tile.

The new path uses native M64 and splits K=5632 across three NPU workers:

```text
K slices = 2048 + 2048 + 1536
```

Each worker runs a hardware-validated M64 tile, and the pool accumulates the three int32 partial outputs.

n128 A/B:
- base native M64: 38.514 tok/s
- + native FFN-down K-split: 47.600 tok/s
- about +23.6%
- acceptance 100%

n512 A/B under normal ondemand conditions:
- base: 72.943 tok/s
- + FFN-down K-split: 82.071 tok/s
- about +12.5%
- acceptance 100%

Controlled fixed-frequency run without prewarm:
- 85.666 tok/s
- acceptance 100%

### Prewarm is now effective

The earlier prewarm attempt was ineffective because:
- the adapter hook was not on the actual split/router path used by the best benchmark;
- it only considered a subset of K/N shapes;
- it did not prepare the new K=5632,N=2048 FFN-down K-split form.

The consolidated path prewarms:
- K=2048,N=256
- K=2048,N=2048
- K=2048,N=5632
- K=5632,N=2048

and selects N-split or K-split resident preparation to match the runtime path.

Measured decode cache:
- before effective prewarm: 154 misses
- after effective prewarm: 3 misses, 1229 hits

This is the main reason decode throughput jumped from the mid-80s to 127-149 tok/s in controlled runs.

## Negative / closed-for-now experiments

These should not be reintroduced without a new reason:

- grouped Q/O M16 (`ROCKNPU_MTILE_QO_GROUP=1024`) — worse than ungrouped native routing
- process/global/little-core/phase affinity experiments — worse
- NEON + prefetch weight-layout experiment — microbenchmark gain did not survive real A/B
- Q6 W8 preparation via Rayon — reduced preparation time but hurt end-to-end runtime through CPU/NPU contention
- M80 hybrid speculative batch — major regression
- old single-core `ROCKNPU_MTILE_DOWN` path — superseded by native M64 + 3-core K-split

## Pending hypotheses

### P1: explain 127-149 tok/s controlled-run variance

Both runs used fixed NPU/CPU frequency and 100% acceptance, but decode varied materially.

The profile difference is dominated by wait/input timing rather than cache misses. Investigate:
- per-core Rocket wait variance
- worker scheduling/wakeup latency
- NPU core synchronization behavior
- whether the three K-split workers create occasional serialization
- IRQ / kernel worker placement

Do not change the kernel/module merely to investigate this; profile userspace and existing driver behavior first.

### P2: eliminate the remaining three decode cache misses

Prewarm reduces 154 misses to 3. Identify exactly which weights/shapes remain cold. If they are stable graph weights, add them to prewarm. If they are genuinely dynamic, leave them alone.

### P3: reduce prewarm cost without reintroducing contention

Prewarm improves decode dramatically but increases encode/prefill time from ~1.8 s to ~3.5 s in the measured NPU run.

Potential directions:
- move preparation earlier than first prompt execution when model lifetime allows
- persist prepared W8 layout safely across repeated sessions/process lifetime
- avoid duplicate host conversions between N-split and K-split cache forms
- reuse prepared source weights when the same tensor is needed by multiple execution geometries

Do not repeat the rejected Q6 Rayon conversion as-is.

### P4: promote native M64 out of research gating

Before changing defaults:
- repeat across different prompts
- repeat after process restart
- validate at least one additional compatible model/shape set
- keep M32/M48/M64 bit-exact smoke in CI or hardware-gated validation

### P5: PRIME/shared-BO remains pending

The earlier cross-fd PRIME/shared-BO experiment functioned but teardown coincided with system-wide userspace instability on o16g. It remains pending, not rejected.

Do not revisit until lifetime/teardown ownership is explicitly designed and tested.

## External research

ORK research was useful for re-checking the INT8 M-tile assumptions and the duplicated register patch behavior. This checkpoint does not copy external kernel/driver code into the MIT RockNPU tree.

### External ordinary-generation comparison

These are **reference points, not direct A/B results**. The **13.79 +/- 0.85 tok/s** `tg128` result above is only the ordinary-generation sanity run taken under the speculative checkpoint environment; it is not the project's best known M=1 path. For ordinary-generation context, retain the separately validated ~16 tok/s userspace direct-submit/persistent-scratch result and the ~19.3-19.4 tok/s research-driver result documented in `docs/repro.md`.

| Runtime / path | Model / quantization | Published TG | Relation | Comparability |
|---|---|---:|---:|---|
| RockNPU local sanity | TinyLlama 1.1B, Q4_K_M GGUF | 13.79 tok/s | sanity only | speculative-checkpoint env; not best M=1 configuration |
| Radxa RKLLM reference | TinyLlama 1.1B, RK3588 | 15.03 tok/s | not direct A/B | older external reference; conversion/runtime details incomplete |
| RKLLM independent reproduction | TinyLlama 1.1B, plain W8A8 | 23.08 +/- 0.37 tok/s | not direct A/B | different quantization/runtime; ROCK 5B+, working DDR/DMC scaling |
| Rockchip-derived benchmark table | TinyLLAMA 1.1B, W8A8, seqlen 128, 64 new tokens | 24.26-24.49 tok/s | not direct A/B | external maximum-frequency reference, different quantization/runtime |
| llama.cpp PanVK report | Llama-3.2-1B-Instruct Q4_1, Mali-G610 | ~3.6 tok/s | not direct A/B | **different model/quantization**; GPU-direction evidence only, not a TinyLlama A/B |

Sources:
- Rockchip RKLLM / rknn-llm benchmark discussion and independent reproduction: https://github.com/airockchip/rknn-llm/issues/532
- Rockchip-derived benchmark table: https://github.com/Pelochus/ezrknn-llm/blob/main/benchmark.md
- Seeed RK3588 benchmark table: https://sensecraft.seeed.cc/ai-lab/en/tutorials/rk/benchmark/rk3576-and-rk3588-llm-and-vlm-performance-benchmarks
- Radxa RKLLM reference: https://docs.radxa.com/en/som/nx/nx5/ai-dev/rkllm-usage
- Mali-G610 / PanVK llama.cpp report: https://github.com/ggml-org/llama.cpp/issues/17783

The 2026 RKLLM reproduction is especially useful because it explains why older community results were much lower. Plain W8A8 versus grouped W8A8 and functional DDR/DMC frequency scaling materially change decode throughput. It is a useful external target range, but it is not valid to turn it into a precise RockNPU percentage gap without matching quantization, memory-clock state, token lengths, and runtime conditions.

There is still no comparable public RKLLM lookup-speculative benchmark. The local 127-149 tok/s M64 verifier result must therefore stay in a separate category; it cannot be divided by RKLLM ordinary-generation throughput.

The available Mali-G610/PanVK evidence strongly suggests the current RockNPU ordinary path is already ahead of that open GPU path, but the public report uses Llama-3.2-1B-Instruct Q4_1 rather than the exact TinyLlama/Q4_K_M workload. Treat this as directional evidence only, not a formal win.

## Safe benchmark notes

- Keep model/binaries in `/tmp` tmpfs when measuring; this avoids NVMe/TCP read stalls.
- Do not move the workload to eMMC.
- Do not persist experimental clock/governor settings as part of the code change.
- Do not reload/replace Rocket kernel modules for ordinary M64/prewarm validation.

## Consolidated-branch validation

The final self-contained branch was rebuilt from a fresh CMake build directory using its own CAPI source tree. Final gates:

- repeated-register regression test: PASS
- hardware smoke M16: PASS
- hardware smoke M32: PASS
- hardware smoke M64: PASS
- consolidated plugin short sanity: 155.356 tok/s, `n_accept=126`, 100% acceptance, 3 cache misses
- no external RockNPU source override in CMake

## Stage conclusion

This checkpoint changes the project conclusion materially:

- RockNPU no longer merely edges out CPU on TinyLlama decode.
- Native M64 is hardware-correct and materially reduces submit/wait overhead.
- FFN-down K=5632 is profitably accelerated with 3-core K-split.
- Effective prewarm removes almost all first-decode M-tile cache misses.
- A conservative repeated controlled result is 127.181 tok/s versus 52.619 tok/s CPU, with 100% speculative acceptance.

The next optimization stage should focus on variance and the final three misses, not larger speculative M values or another round of generic host micro-optimizations.
