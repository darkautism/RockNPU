# oRKLLM ork-driver capability audit — 2026-09-21

Reference: oRKLLM/ork-driver wiki and README capability matrix, reviewed 2026-09-21.

This document records which reverse-engineered RK3588 NPU mechanisms are already present in RockNPU, which were added during this audit, and which remain pending. It is a capability comparison, not a claim that every ork-driver mechanism is automatically beneficial to TinyLlama.

## Added and validated in this audit

### M=1 wide full-K W8A8

The previous RockNPU M=1 contract stopped at K<=4096. ork-driver documents a validated RK3588 full-K decode envelope to raw K=10752 and uses that mechanism as part of its ~96% closed-runtime decode result.

RockNPU now has a dedicated M=1 full-K resident packing path up to K=10752 without changing the M-tile K-split contract.

On-silicon exact checks:

- M=1 K=5632 N=64: PASS.
- M=1 K=5632 N=2048: PASS.
- 3-core N-split, each core full-K K=5632: PASS vs CPU reference.

Standalone K5632/N2048:
- single-core median: 4.370 ms
- 3-core N-split median: 2.780 ms
- speedup: 1.57x

In the real TinyLlama ordinary-generation path, the existing worker tuner now compares N-split full-K against K-split rather than forcing the old K3 workaround. A short trace after tuning reported worker_calls=[0,44,792], ksplit_calls=0, confirming that most calls selected 3-core N-split/full-K. Correct-governor tg128 measured 17.69 +/- 0.16 tok/s versus the prior 17.11 +/- 0.69 checkpoint.

### Native W8A8 M128

The previous native M-tile contract stopped at M64. ork-driver documents the larger RK3588 int8 M envelope.

On-silicon:
- M=128 K=2048 N=2048: bit-exact PASS.
- five submit/wait samples: ~3.46-3.66 ms.

The M128 shape is now admitted through the existing persistent M-tile, pool, C ABI, and ggml-rocknpu native-Mtile router. M128 is restricted to per-submit K<=2048; K5632 FFN-down still uses the validated 3-core K-split, whose individual slices fit the M128 envelope.

100%-acceptance lookup-speculative A/B, all CPU policies=performance and NPU=1 GHz:
- M64 / draft-max 63: 174.324 tok/s, 504/504 accepted.
- M128 / draft-max 127: 190.577 tok/s, 508/508 accepted.
- fresh M128 repeat: 190.289 tok/s, 508/508 accepted.

M128 therefore improves this verifier workload by about 9.2-9.3% over M64.

### Fused FP16 matmul + residual add

RockNPU already had an independently implemented MIT `encode_fp16_matmul_accumulate()` path used internally for NPU K-split accumulation, but it was not exposed as a model-level fused residual operation.

The prepacked M-compatible executor now accepts an optional row-major residual and seeds the existing DPU EW ping-pong path with it. This works both for one-K-tile projections and for NPU K-split accumulation; tiny-M host-accumulation geometry remains rejected because its EW surface mapping is not validated.

On-silicon comparisons against `plain NPU matmul + CPU fp16 residual add`:
- M=16 K=256 N=64: PASS, max_abs=0.000244.
- M=16 K=2048 N=64: PASS, max_abs=0.000488.
- M=16 K=2048 N=2048: PASS, max_abs=0.000610.

`rocknpu-llm` prefill now uses the fused path for the attention output projection residual and FFN down projection residual when M geometry permits. Decode M=1 intentionally keeps the CPU residual fallback.

### INT8 CBUF WEIGHT_REUSE

The ork reverse-engineering record identifies CNA_CBUF_CON0 (0x1040) bit13 as WEIGHT_REUSE: a later task in the same Rocket job may reuse the previous task's weight tile from CBUF instead of fetching it from DRAM again.

An important negative result came first: setting bit13 inside a standalone M128 task is incorrect (first output observed expected 2048, got -3851). WEIGHT_REUSE is a cross-task contract, not an internal M-group flag. The correct contract requires:
- the reuse task follows a task with the same weight tile;
- identical M/N/K tile geometry so the CBUF bank split is unchanged;
- both tasks execute in the same Rocket job;
- the full KxN weight segment fits the configured WEIGHT_BANK capacity.

RockNPU now exposes a fail-closed `encode_int8_mtile_weight_reuse()` encoder and a hardware smoke that submits multiple same-weight tasks in one job. The encoder sets bit13 only after validating the weight tile against the task's 0x1040 bank geometry.

On-silicon exact checks on o16g:
- 2 x M64, K2048/N64: PASS with and without reuse.
- 4 x M32, K2048/N64: PASS with and without reuse.
- 8 x M16, K2048/N64: PASS with and without reuse.
- 2 x M128 = total M256, K2048/N64: PASS with and without reuse.

CPU-big-core performance-governor ABBA medians for total M128:
- 2 x M64: baseline 388.8/411.5 us, reuse 356.4/391.4 us.
- 4 x M32: baseline 480.4/480.4 us, reuse 390.8/386.2 us.
- 8 x M16: baseline 539.3/610.5 us, reuse 438.1/459.4 us.

The deeper reuse runs show the expected larger benefit because more weight DMA fetches are skipped.

For the immediately useful M256 case (2 x M128, K2048/N64), controlled ABBA medians were:
- baseline 771.2 / 753.7 us;
- WEIGHT_REUSE 694.4 / 701.7 us.

That is about 8.4% lower latency, or roughly 9.2% higher throughput, while remaining bit-exact. The next production step is to lower M>128 verifier batches into same-job M128 tasks and enable WEIGHT_REUSE after the first task.

### NONBLOCK doorbell status

The live stock Rocket UAPI/kernel headers expose no Rocket/RKNPU NONBLOCK submit flag, and the loaded module exposes no matching runtime parameter. RockNPU already separates submit and completion through `begin_execute_prepared()` / `finish_execute_prepared()`, but that is not equivalent to ork-driver's NONBLOCK doorbell capability. This remains a driver-side capability gap; no kernel/module change was made during this audit.

## Existing RockNPU capabilities that already overlap ork-driver

- resident packed W8 weights / reuse across calls;
- direct submit with separated begin/finish;
- persistent activation/output scratch;
- up to 3 NPU workers and N/K split;
- W8A8 M=1 and native M16/M32/M48/M64/M128;
- W4A4 primitive and grouped experiments;
- Q/K/V and gate/up same-input grouping at the ggml adapter level;
- prewarm / resident prepared layouts;
- int32 host accumulation for K-split;
- zero-copy-style reuse of owned Rocket buffers inside the process;
- register-command probes and hardware smoke tests.

## Capability gaps still to evaluate

The current ork-driver matrix exposes several mechanisms RockNPU does not yet have as complete production surfaces:

- resident KV on the NPU;
- general hardware PC-chain / M-fold chain;
- NONBLOCK doorbell at the driver level; userspace begin/finish overlap already exists;
- general batched GEMM surface;
- fused nonlinear activation output stage;
- standalone SDP activation path;
- standalone NPU elementwise add and multiply beyond the validated fused residual-add path;
- per-channel multiply;
- general zero-copy dma-buf import/adopt;
- persistent serialized packed-weight format / streaming pool;
- output zero-copy into caller-provided GGML storage.

These are not all equally valuable for TinyLlama. The next audit order is:
1. resident KV and attention data movement;
2. elementwise/add/multiply and residual/RMSNorm boundary reduction;
3. hardware chain / M-fold, especially gate+up / FFN glue;
4. NONBLOCK/async only where it removes a real host wait;
5. zero-copy import/output and persistent packed-weight load;
6. batched GEMM where a real model graph exposes a matching workload.

Kernel/module changes remain out of scope unless explicitly approved. GPL driver code must not be copied into this MIT repository.
