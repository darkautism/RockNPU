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

In the real TinyLlama ordinary-generation path, the existing worker tuner compares N-split full-K against K-split rather than forcing the old K3 workaround. The previously quoted 17.69 +/- 0.16 tok/s whole-model result is invalidated because it was measured after NPU timeouts in the same boot. After rebooting back to the packaged stock Rocket driver, the current code was rebuilt from clean main and M=1 K=5632 N=2048 passed an exact prepared-decode gate again. Keep the capability; remeasure any whole-model topline only on a clean boot.

### Native W8A8 M128

The previous native M-tile contract stopped at M64. ork-driver documents the larger RK3588 int8 M envelope.

On-silicon:
- M=128 K=2048 N=2048: bit-exact PASS.
- five submit/wait samples: ~3.46-3.66 ms.

The M128 shape is now admitted through the existing persistent M-tile, pool, C ABI, and ggml-rocknpu native-Mtile router. M128 is restricted to per-submit K<=2048; K5632 FFN-down still uses the validated 3-core K-split, whose individual slices fit the M128 envelope.

The later 1 GHz lookup-speculative A/B that reported 174.324 tok/s for M64 and about 190.3-190.6 tok/s for M128 is invalidated as performance evidence because the board had already experienced NPU timeouts earlier in that boot. The relative userspace mechanism remains worth testing, but those percentages are no longer authoritative.

After rebooting to the packaged stock Rocket driver, M=128 K=2048 N=2048 was rebuilt from clean main and passed bit-exact again, with five stock-driver submit/wait samples around 3.40-3.53 ms. A fresh-stock A-B-B-A lookup test at 100% acceptance measured M64 at 126.174 / 122.309 tok/s and M128 at 137.932 / 136.478 tok/s, or about +10.4% by the two-run centers. This clean-driver relative result replaces the discarded post-timeout +9.2-9.3% claim; the absolute values are stock-200-MHz characterization, not full-speed toplines.

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

Historical stock-driver exact checks included:
- 2 x M64, K2048/N64: PASS with and without reuse.
- 4 x M32, K2048/N64: PASS with and without reuse.
- 8 x M16, K2048/N64: PASS with and without reuse.
- 2 x M128 = total M256, K2048/N64: PASS with and without reuse.

A fresh post-reboot o8g stock-driver rebuild revalidated the 2 x M128 = total M256 reuse path exactly.

CPU-big-core performance-governor ABBA medians for total M128:
- 2 x M64: baseline 388.8/411.5 us, reuse 356.4/391.4 us.
- 4 x M32: baseline 480.4/480.4 us, reuse 390.8/386.2 us.
- 8 x M16: baseline 539.3/610.5 us, reuse 438.1/459.4 us.

The deeper reuse runs show the expected larger benefit because more weight DMA fetches are skipped.

For the immediately useful M256 case (2 x M128, K2048/N64), controlled ABBA medians were:
- baseline 771.2 / 753.7 us;
- WEIGHT_REUSE 694.4 / 701.7 us.

That is about 8.4% lower latency, or roughly 9.2% higher throughput, while remaining bit-exact.

A production follow-up tested whether TinyLlama's real wider projections could profit by first column-splitting them into reuse-safe N64 segments. They do not:

- M256/K2048/N2048: two ordinary full-N M128 tasks = 6.491 ms median; N64 colsplit + WEIGHT_REUSE = 16.248 ms median (about 2.5x slower).
- M256/K2048/N256: two ordinary full-N M128 tasks = 1.313 ms median; N64 colsplit + WEIGHT_REUSE = 2.265 ms median (about 1.73x slower).

The extra task/scheduler cost overwhelms the saved weight DMA on these RockNPU int8 shapes. Therefore WEIGHT_REUSE remains a validated primitive for naturally segmented narrow-N workloads, but it is **not** routed into current TinyLlama Q/K/V/O/FFN production paths. Revisit only if another mechanism already forces compatible column segmentation, or if batched/PC-chain dispatch reduces per-task overhead enough to change this cost model.

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

## Remaining architecture-level gaps and disposition

The remaining ork-driver matrix differences are real, but most are no longer ambiguous "small missing features":

- **Resident KV on NPU:** current `rocknpu-llm` keeps K/V in host `Vec<f16>`, and its QK^T / softmax / AV consumer is also CPU-side. Adding only an NPU KV mirror would add a copy without removing CPU work. Treat resident KV as part of a future NPU-attention subsystem, not as a standalone optimization.
- **Hardware PC-chain / M-fold:** RockNPU already has a PC trailer encoder, and historical prototypes were run. Production o16g exposes no `rocket_batch_submit` capability; stock Rocket per-task kicking does not satisfy the true chained-job contract. Earlier whole-model A/B found no material gain. Keep this blocked on an explicit driver capability rather than retrying self-linked chains on stock UAPI.
- **NONBLOCK doorbell:** driver capability missing on the production kernel. Userspace begin/finish overlap already exists.
- **Batched GEMM:** no dedicated production surface today, but current TinyLlama/llama.cpp execution has no direct hot BMM replacement. Attention is handled by CPU/llama.cpp flash-attention rather than a missing generic BMM call.
- **Fused nonlinear activation / standalone SDP / EW mul / per-channel multiply:** hardware primitives have been explored. Fused SiLU is locally fast but conflicts with the current per-output-channel quantization semantics; standalone pure-SDP completion incurs the stock Rocket ~500 ms fence timeout; naive fused SwiGLU and RMSNorm-boundary experiments were rejected. Revisit only with a new quality-equivalent scaling domain or a safe chained completion contract.
- **Zero-copy dma-buf import/adopt:** stock DRM/RKNPU can import foreign dma-bufs, but the active Rocket accel UAPI exposes no corresponding import ioctl. Existing staging measurements are sub-ms versus the tens-of-ms NPU wait path, so a kernel/UAPI change is not justified by the present bottleneck.
- **Persistent serialized native-packed weights:** useful only for cold start. Steady-state already uses resident prepared weights; prior large FP16 sidecar/mmap approaches were dominated by storage traffic or memory pressure. The provenance-bound W8 sidecar already removes much of the model-format conversion cost without changing the steady-state kernel.
- **Output zero-copy into GGML storage:** still absent, but it targets a copy-sized cost rather than the dominant submit/wait cost. Do not redesign GGML buffer ownership until profiling shows this copy has become material.

### Current priority after this audit

1. Keep the validated wide full-K M=1, M128, fused residual, and WEIGHT_REUSE primitives.
2. Do **not** route WEIGHT_REUSE through forced N64 colsplit for current TinyLlama shapes; measured N256/N2048 cases regress badly.
3. Do not resurrect stock-kernel PC-chain, pure-SDP completion, fused-int8 SwiGLU, RMSNorm-boundary, or zero-copy-driver work without new contradictory evidence.
4. The next genuinely new high-upside direction is an end-to-end NPU attention subsystem (resident KV + QK^T + softmax + AV) or a new quality-equivalent fused FFN scale domain, not another isolated transport flag.

Kernel/module changes remain out of scope unless explicitly approved. GPL driver code must not be copied into this MIT repository.
