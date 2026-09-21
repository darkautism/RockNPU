# H1: cold resident-prefill phase decomposition

This note is diagnostic evidence only. The profiling build is **not** used for formal throughput claims.

## Current truth and provenance

- Base main: `643f0eecd4ee0a4c1c9101be19b29b7253b31995`.
- Profiler commit: `9a6edc6e491b0193f326f8d57b0de10d3b791b7c`.
- llama.cpp: `391fac16460f15233a7740550d858ac96df3419d`.
- Model: `TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf`.
- Model SHA-256: `5c66751b61537f9e55177b1b67e06af88e0e2df88f86de4909f5bf87fb1ae583`.
- Workload: fresh process, `pp512+tg128`, `--no-warmup`, `taskset -c 4-7`, four CPU threads, flash attention on.
- Hybrid: `ROCKNPU_PREFILL_CACHE=1 ROCKNPU_DECODE=0`.
- Profiling is enabled only with `ROCKNPU_PREFILL_PROFILE=1`; normal stderr remains unchanged when it is unset.

o8g ran at NPU 700 MHz with all CPU policies on `performance`. Its external Rocket research tree was clean at `ed52a89afa8e68fedf636c8e891bd8fc47e82d26`; NPU IRQs 97-99 were on CPU0.

o16g independently rebuilt the same profiler commit, also at NPU 700 MHz with all CPU policies on `performance`. Its Rocket tree was a **RockNPU-created local GPL research modification made to test higher-throughput decode**, including an IOMMU-domain cache; the local tip was recorded as `3345d5f472e30b66b6c0d9640518c5315c99add5`. That commit was never published, so the hash is not a reproducible checkout target by itself. The three NPU IRQs were distributed over CPU0/CPU1/CPU2. This remains useful only as a historical cross-machine check across different driver/IRQ states.

Targeted correctness before profiling passed independently on both boards: rocknpu-capi 8/8 and rocknpu-matmul 20/20.

## Diagnostic results

The resident cache had 151 misses and 1782 MiB of resident weight storage on each fresh request.

| Cold one-time phase | o8g | o16g |
| --- | ---: | ---: |
| worker-pool creation | 0.550 ms | 0.472 ms |
| GGUF Q4/Q6 dequant to FP16 | 3063.543 ms | 2942.869 ms |
| decoded `Vec<f16>` -> `Arc<[f16]>` | 768.905 ms | 703.253 ms |
| resident prepare call | 1132.204 ms | 1271.573 ms |
| **total measured one-time setup** | **4965.202 ms** | **4918.167 ms** |

Thus dequant plus the host Vec-to-Arc materialization is 3832.448 ms on o8g and 3646.122 ms on o16g, roughly three quarters of the measured cold setup. Worker creation is irrelevant at this scale.

The resident prepare call itself decomposes as follows:

| Resident prepare detail | o8g | o16g |
| --- | ---: | ---: |
| pool internal wall | 926.395 ms | 1044.216 ms |
| worker critical path | 885.083 ms | 1030.533 ms |
| pool dispatch/wait residual | 41.312 ms | 13.684 ms |
| outer-call overhead after pool wall | 205.810 ms | 227.357 ms |
| BO allocation + mmap, critical maxima | 290.449 ms | 500.179 ms |
| PREP ioctl, critical maxima | 12.549 ms | 12.631 ms |
| zero/page-touch, critical maxima | 250.088 ms | 239.181 ms |
| tile packing, critical maxima | 340.253 ms | 292.048 ms |
| FINI, critical maxima | 20.780 ms | 27.682 ms |
| planner + layout, critical maxima | 1.821 ms | 1.783 ms |

Per-phase maxima can name different workers, so their sum is not expected to equal the per-weight worker critical path exactly.

For context only, the instrumented diagnostic requests themselves were 16.416 s on o8g and 15.823 s on o16g. Do not use those as formal throughput measurements.

The clean current-main evidence already stored in this benchmark directory remains the performance reference: warmed/steady-state `pp512+tg128` averaged 12.854186 s CPU versus 11.460154 s hybrid on the formal o8 ABBA, while the small cold `--no-warmup` check on o16 averaged 12.147630 s CPU versus 15.494643 s hybrid. Warm and cold evidence are intentionally kept separate.

## Mechanism conclusion

The cold regression is not explained by worker creation, planner cost, or pool synchronization.

The dominant first-use work is:

1. serial GGUF block dequantization into FP16;
2. another large host-memory materialization converting the decoded Vec into an Arc slice;
3. resident BO creation/page-touch/tile packing.

The cross-machine total is remarkably stable despite different Rocket revisions and IRQ routing: about 4.9-5.0 seconds of measured one-time setup.

## Verdict and next hypothesis

**KEEP EXPERIMENTAL.** The instrumentation is useful research evidence but is not a performance candidate by itself.

Next priority is **H1-C: a minimal persistent/offline representation prototype**.

The first prototype should stay small: cache only resident-prefill tensor materialization, bind it to the exact GGUF SHA/size/tensor identity plus a RockNPU format/runtime version, and safely fall back on mismatch. The first measurable target is to remove the ~3.0 s dequant path and avoid the ~0.7 s Vec-to-Arc copy by feeding an mmap/borrowed representation into preparation. Do not begin with a large general-purpose cache format.

If that succeeds, measure the remaining ~1 s resident prepare separately before deciding whether hardware-specific prepacked tile persistence is worth the extra format complexity.

**H1-B parallel preparation is second priority.** It can attack some of the remaining preparation/dequant wall time, but the current dominant work is memory-heavy and may be bandwidth-limited.

**H1-A eager prepare is third priority for performance.** It can improve first-user-request latency by moving the cost into initialization, but it does not reduce process-total cold startup and must report that trade explicitly.

## H1-C result: raw FP16 persistence is a dead end on this board/storage

A minimal provenance-bound FP16 sidecar was prototyped from the exact GGUF. It contained all 154 projection tensors, recorded the source GGUF size/SHA-256 and per-tensor source fingerprint, and occupied about **1.9 GiB** on disk. Runtime fingerprint mismatch or missing files fell back to the normal quantized path. No model weights were altered.

Two loading mechanisms were tested on o8g with fresh-process `pp512`, `--no-warmup`:

| H1-C variant | pp512 time | dominant new cost | verdict |
| --- | ---: | ---: | --- |
| heap-resident FP16 sidecar | 72.007 s | 64.583 s reading 1782 MiB sidecar | REJECT |
| per-tensor mmap -> pack -> munmap | 72.601 s | 65.680 s worker tile-pack/page-fault path | REJECT |

The heap-resident version successfully removed runtime GGUF dequantization (`0.008 ms`) and the Vec-to-Arc copy, while resident prepare itself remained about `0.908 s`. However, retaining roughly 1.78 GiB of host FP16 beside roughly 1.78 GiB of resident BOs caused severe memory pressure on the 7.7 GiB board; during the run only about 238 MiB was free and swap usage was about 922 MiB.

The mmap version removed that host residency and its mmap/open setup was only `3.380 ms`, but this did **not** remove storage traffic. The workers faulted/read the FP16 pages while packing: `prepare_call=66.208 s`, `worker_critical=66.186 s`, `tile_pack_critical=65.680 s`. `/proc/<pid>/io` near completion showed roughly 1.46 GiB of physical reads. Thus mmap merely moved the raw-FP16 read cost into the resident packing phase.

Mechanistically, the existing runtime conversion is better matched to this system: it reads the compact ~668 MB quantized GGUF and spends about 3 seconds dequantizing it while expanding toward resident form. Persisting the expanded FP16 representation nearly triples storage bytes and trades a few seconds of CPU conversion for tens of seconds of storage/page-fault traffic. The 1.9 GiB disk footprint is also a material regression.

**H1-C is therefore rejected.** Do not retry raw row-major FP16 persistence unless the storage/memory mechanism changes materially (for example a genuinely device-ready compact representation with evidence that total bytes read are lower). The failed production prototype should not be promoted or retained in main.

The next useful experiment is **H1-B**, specifically parallelizing the stable ~3 second Q4/Q6 dequant phase while leaving the already-parallel resident worker packing unchanged.

## H1-B result: parallel Q4_K/Q6_K prefill dequant

Candidate commit: `7e3c16249267a140d66be896f2fa1f259cebf2cd`.

The candidate keeps serial dequant as the default and enables the experiment with `ROCKNPU_PREFILL_PARALLEL_DEQUANT=1`. Large projection tensors (at least 1,048,576 values) dequantize Q4_K/Q6_K blocks in parallel with Rayon. Formal runs used `RAYON_NUM_THREADS=4` and `taskset -c 4-7`, matching the four A76 cores. Small tensors remain serial.

Correctness gates passed independently on o8g and o16g:

- `rocknpu-capi`: 10/10, including new bit-for-bit serial-vs-parallel Q4_K and Q6_K dequant tests;
- `rocknpu-matmul`: 20/20;
- full `pp512+tg128` requests completed normally;
- documented TinyLlama deterministic 24-token gate with runtime-W8 decode enabled produced stdout SHA-256 `08730f9092a465cc9915db41d7ba8f999504c968e4937d73b6d9f068dcae8f8d`, exactly matching the current contract.

### o8g mechanism diagnostic

Fresh-process `pp512 --no-warmup`, same build, profiling enabled only for diagnosis:

| phase/workload | serial A | parallel B | effect |
| --- | ---: | ---: | ---: |
| dequant sample 1 | 3036.620 ms | 1257.234 ms | |
| dequant sample 2 | 3023.819 ms | 1333.937 ms | |
| **dequant mean** | **3030.220 ms** | **1295.586 ms** | **-57.24%** |
| pp512 sample 1 | 10.955095 s | 9.211561 s | |
| pp512 sample 2 | 10.961316 s | 9.482908 s | |
| **pp512 mean** | **10.958205 s** | **9.347235 s** | **-14.70%**, `1.172x` |

The resident preparation and NPU execute phases stayed in the same range. The improvement tracks the targeted CPU dequant phase rather than a scheduler/cache artifact.

### o8g formal cold full request

Tracing/profile off, fresh process, `pp512+tg128`, `--no-warmup`, A/B/B/A:

- serial A: `15.761268`, `15.725260` s;
- parallel B: `13.980560`, `14.342719` s;
- serial mean/median: `15.743264` s;
- parallel mean/median: `14.161640` s;
- population stddev: serial `0.018004` s, parallel `0.181079` s;
- absolute delta: `1.581624` s/request;
- latency reduction: **10.05%**;
- speedup: **1.112x**.

### o16g independent validator

o16g independently fetched and detached at the exact candidate commit, rebuilt the plugin, and re-ran targeted tests. Hardware was NPU 700 MHz with all CPU policies on `performance`; model SHA-256 remained `5c66751b61537f9e55177b1b67e06af88e0e2df88f86de4909f5bf87fb1ae583`.

A parallel-only diagnostic measured `dequant=1193.468 ms`; the prior same-board H1 serial diagnostic measured `2942.869 ms`, confirming the same mechanism reduction.

Formal tracing-off fresh-process `pp512+tg128`, `--no-warmup`, reversed order B/A/A/B:

- parallel B: `13.884932`, `13.779607` s;
- serial A: `15.891719`, `15.729858` s;
- serial mean/median: `15.810788` s;
- parallel mean/median: `13.832270` s;
- population stddev: serial `0.080931` s, parallel `0.052662` s;
- absolute delta: `1.978519` s/request;
- latency reduction: **12.51%**;
- speedup: **1.143x**.

### Verdict

**PROMOTE.** H1-B passes correctness, mechanism, o8g gain, o16g reproduction, whole-request gain above noise, and introduces no model-format/license change. It specifically improves cold first-use preparation; warmed resident-cache behavior should remain unchanged after preparation.

One measured cold cost remains conspicuous after parallel dequant: converting the decoded `Vec<f16>` into `Arc<[f16]>` still costs roughly `0.7-0.8 s` per model load. A follow-up child experiment may remove that copy by borrowing the decoded Vec synchronously while the persistent workers pack resident BOs. Keep `7e3c162` as the validated fallback candidate if that extra optimization does not reproduce.

## H1-B child result: reuse the dequantized Vec allocation

Candidate commit: `970c80069ee0896c353f271b31144fefba3ce0a7`.

The validated H1-B path still spent roughly 0.7-0.8 seconds converting each freshly decoded `Vec<f16>` collection into `Arc<[f16]>`. This child keeps the decoded allocation intact by sharing `Arc<Vec<f16>>` with the resident preparation workers instead of bulk-copying into a new Arc slice. No tensor values, GGUF parsing, NPU kernel, scheduling policy, model format, or driver code changes.

Correctness passed independently on both boards:

- o8g: `rocknpu-capi` 10/10 and `rocknpu-matmul` 20/20;
- o16g: `rocknpu-capi` 10/10 and `rocknpu-matmul` 20/20;
- o16g documented TinyLlama deterministic 24-token continuation exactly matched the 82-byte contract, SHA-256 `08730f9092a465cc9915db41d7ba8f999504c968e4937d73b6d9f068dcae8f8d`.

### Mechanism diagnostic

Fresh-process o8g `pp512 --no-warmup` with profiling enabled measured:

- dequant: `1009.966 ms`;
- `host_to_arc`: **`0.125 ms`**;
- resident prepare call: `889.945 ms`;
- prepare-pool wall: `887.607 ms`;
- outer prepare-call overhead: **`2.338 ms`**;
- diagnostic pp512 wall: `7.766798 s`.

The former host materialization cost is therefore effectively eliminated rather than hidden in another phase.

### Whole-request adjacent A/B

Tracing/profile disabled, fresh process, `pp512+tg128`, `--no-warmup`, H1-B (`7e3c162`) versus the Arc/Vec child (`970c800`):

**o8g, A/B/B/A**

- H1-B A: `13.981163`, `14.169145` s;
- Arc/Vec B: `12.701498`, `12.742539` s;
- H1-B mean: `14.075154 s`, population stddev `0.093991 s`;
- Arc/Vec mean: `12.722019 s`, population stddev `0.020520 s`;
- absolute delta: `1.353135 s/request`;
- latency reduction: **9.61%**;
- speedup: **1.106x**.

**o16g, A/B/B/A**

- H1-B A: `13.507430`, `13.741281` s;
- Arc/Vec B: `12.266595`, `12.335495` s;
- H1-B mean: `13.624355 s`, population stddev `0.116926 s`;
- Arc/Vec mean: `12.301045 s`, population stddev `0.034450 s`;
- absolute delta: `1.323310 s/request`;
- latency reduction: **9.71%**;
- speedup: **1.108x**.

**Verdict: PROMOTE.** The mechanism is directly measured, both boards independently reproduce a whole-request gain far above run-to-run noise, and correctness is unchanged.

## H1-A result: load-time eager prepare is not a useful cold-throughput candidate at the current GGML boundary

H1-A proposed moving resident prefill preparation out of the first user request and into initialization. It was intentionally lower priority because it does not remove work; it only changes when the work is paid.

The current stock-GGML integration boundary makes a true model-load eager implementation materially different from the hypothesis:

- RockNPU is loaded as an out-of-tree dynamic backend and deliberately returns the stock CPU host buffer type;
- the GGML backend/device callbacks provide graph execution, op support, buffer, copy and synchronization surfaces, but no model-loaded/preload lifecycle callback;
- because RockNPU does not own a custom model buffer, it does not receive model-weight `set_tensor` callbacks during normal llama.cpp loading;
- the first point where the adapter has the complete weight pointer, activation geometry and exact prompt M for the existing exact-M resident plan is request graph scheduling/execution, i.e. the latency path H1-A intended to avoid.

A custom RockNPU buffer type could in principle take ownership of model tensors and perform partial work during load, but that is a new memory/lifecycle architecture, not a minimal eager-policy experiment. It would also need to solve unknown exact-M planning and could duplicate host model residency. Patching llama.cpp to add a RockNPU-specific model-load hook would violate the project's non-invasive stock-frontend boundary.

**Verdict: REJECT as a cold-throughput hypothesis for the current GGML backend.** No production code is added. If a future upstream-neutral backend lifecycle/preparation hook appears, eager preparation may be reconsidered strictly as a latency-placement policy, with startup time and memory reported separately; it should not be credited as reducing process-total cold work.

## H1 family final status

- H1-C raw persistent FP16 representation: **REJECT** — storage/page-fault cost dominates and disk footprint regresses.
- H1-B parallel Q4_K/Q6_K dequant: **PROMOTE** — large mechanism and whole-request gains reproduced on two boards.
- H1-B child Arc/Vec allocation reuse: **PROMOTE** — removes the remaining host bulk copy and yields an additional ~9.6-9.7% cold full-request latency reduction on two boards.
- H1-A load-time eager prepare: **REJECT for current stock-GGML integration** — shifts rather than removes work and lacks a non-invasive model-load hook with the current host-buffer design.

### Production default and fallback check

After promotion, parallel Q4_K/Q6_K prefill dequant is enabled by default and
`ROCKNPU_PREFILL_PARALLEL_DEQUANT=0` remains an explicit serial fallback.

On o8g with the documented `taskset -c 4-7` affinity and `RAYON_NUM_THREADS`
unset, the parallel path measured `dequant=1008.950 ms`, essentially identical to
the explicit four-thread diagnostic (`1009.966 ms`). The serial opt-out measured
`dequant=3188.919 ms`, confirming the fallback still selects the intended path.
The Arc/Vec reuse remained active in both cases (`host_to_arc=0.125-0.127 ms`).

A fresh-process production-default `pp512+tg128` diagnostic on o8g measured
`13.302210 s`. A separate same-setting CPU diagnostic measured `11.949711 s`.
These are single diagnostics rather than a formal CPU/hybrid ABBA, so they are not
used as a promotion statistic; they confirm only that the cold CPU gap is narrowed
but not yet eliminated. The existing warmed/steady-state result remains a separate
claim.
