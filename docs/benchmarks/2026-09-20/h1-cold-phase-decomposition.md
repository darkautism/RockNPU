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

o16g independently rebuilt the same profiler commit, also at NPU 700 MHz with all CPU policies on `performance`. Its Rocket tree was at `3345d5f472e30b66b6c0d9640518c5315c99add5`; the three NPU IRQs were distributed over CPU0/CPU1/CPU2. This gives a useful cross-machine check across different driver/IRQ states.

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
