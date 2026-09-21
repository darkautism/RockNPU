# RockNPU research status

Last consolidated: 2026-09-22.

This is the canonical research-direction document. RockNPU is a userspace RK3588 NPU runtime/compiler/backend project. Research is limited to model integration, graph partitioning, tensor/layout work, register-command generation, device submission through the existing public interface, quantization, residency, userspace scheduling, and model-level performance/correctness.

Historical system-driver tuning is intentionally excluded from this document and from the project roadmap.

## Current objective

Improve real TinyLlama-class inference on RK3588 while preserving reproducibility and model quality.

The immediate performance problem is ordinary autoregressive M=1 decode. Large-M verifier/prefill execution is substantially healthier than M=1 decode, so work should not optimize a local primitive merely because it benchmarks well in isolation.

Success requires all three:

1. deterministic or tolerance-bounded correctness against an independent reference;
2. a measurable whole-model improvement;
3. an implementation that belongs in the userspace RockNPU stack.

## Authoritative baseline

Model used for the current LLM work:

- TinyLlama-1.1B-Chat-v1.0 Q4_K_M;
- model SHA-256: 5c66751b61537f9e55177b1b67e06af88e0e2df88f86de4909f5bf87fb1ae583.

Fresh post-reboot CPU reference with native ARM llama.cpp and CPU policies 0/4/6 on performance:

- tg128: 33.88 +/- 0.08 tok/s.

Old NPU absolute throughput values collected in mixed experimental system configurations are not project toplines. Use current-main, same-process or tightly interleaved relative A/B tests for userspace decisions.

## Validated userspace capabilities

### Frontend integration

- ONNX imports into the shared RockNPU IR.
- Stock llama.cpp can dynamically load libggml-rocknpu.so; no llama.cpp fork is required.
- Stock Ollama 0.34.2 loads the same GGML backend with no Ollama source patch. `GGML_BACKEND_PATH` supplies the plugin and the standard llama.cpp `LLAMA_ARG_DEVICE=ROCKNPU0` selector routes execution to it.
- Candle 0.11 has a separate thin adapter whose first real module is prepared RockNpuLinear; Candle types remain outside core crates, and the adapter executes through shared rocknpu-ops / rocknpu-tensor / rocket-runtime primitives.
- The Candle Linear path is validated both against Candle CPU semantics and on the real RK3588 NPU.
- Supported frontend slices are explicit. Unsupported work remains outside RockNPU rather than being silently emulated.
- Q4_K and Q6_K TinyLlama projections are exercised through the real GGML plugin.

### Resident W8A8 M=1 decode

- Static decode weights are prepared once and kept resident.
- Persistent scratch and direct submit are active.
- The active M=1 route supports TinyLlama Q/K/V/O/gate/up/down projections.
- K=5632,N=2048 FFN-down has a validated full-K M=1 path and multi-worker N split.
- Worker topology is measured and cached by shape rather than hard-coded.

### Native M-tile verifier/prefill shapes

Validated native W8A8 M shapes include M16, M32, M48, M64, and M128.

Fresh A-B-B-A lookup verification at 100% acceptance measured:

- M64: 126.174, 122.309 tok/s;
- M128: 137.932, 136.478 tok/s.

The two-run centers show roughly +10.4% for M128 over M64. The absolute values are characterization numbers; the relative M128 result is the useful conclusion.

### Userspace projection grouping

The adapter already reduces repeated activation work and projection boundaries by combining compatible same-input projections.

Validated patterns:

- V + K concat-N pair;
- gate + up concat-N pair;
- Q + V + K concat-N triple across the GGML partition boundary using a backend-local stash.

The combined paths preserve the existing per-output W8 scaling semantics. Historical same-process ABBA showed small but repeatable whole-token gains for V/K and gate/up, and their combined default-on configuration.

Do not reimplement these as superficial grouping: they already execute as larger combined matmuls.

### FP16 fused residual

The prepacked FP16 executor can fuse residual addition into the validated NPU elementwise accumulation path.

Exact/tolerance gates:

- M16/K256/N64: max abs 0.000244;
- M16/K2048/N64: 0.000488;
- M16/K2048/N2048: 0.000610.

rocknpu-llm uses this for prefill attention-output and FFN-down residuals when geometry permits.

### INT8 CBUF weight reuse

Cross-task weight reuse is a real hardware primitive when consecutive tasks in one job use an identical resident weight tile and compatible geometry.

Validated exact cases include repeated M16/M32/M64/M128 tasks.

Important production result:

- forcing N64 column splits solely to obtain weight reuse is a loss;
- M256/K2048/N2048 became about 2.5x slower;
- M256/K2048/N256 became about 1.73x slower.

Therefore keep the primitive, but do not route current TinyLlama wide-N projections through artificial column segmentation.

### Prefill residency

- Static prefill weights can remain prepared/resident across calls.
- Parallel Q4_K/Q6_K dequant preparation and child allocation reuse are already promoted.
- FP16 residual fusion is available for compatible prefill projections.

### Host preparation

Architecture-specific host preparation improvements that reduce real userspace packing/conversion cost are allowed. Existing NEON host-preparation work is part of this category.

## Validated negative results

Do not rerun these without a materially new mechanism.

### Plain FP16 M=1

- Native plain FP16 M=1 geometry is not a valid production decode path.
- Padding M=1 to M=4 can be numerically correct but loses badly at whole-model scale.

Use W8A8/native quantized decode instead.

### Forced W4A4 production decode

- Integer W4A4 primitives are real and locally fast.
- TinyLlama deterministic generation diverges from the W8 baseline under tested W4 schemes.
- Whole-model W4 did not show a stable speed win.

The blocker is quantization/model quality, not worker count.

### Full-int8 fused FFN

- Hardware-level pieces exist.
- Tested compact-int16 and int8 activation routes changed model semantics.
- A lossless raw-int32 intermediate recovered deterministic quality but regressed whole-token throughput.

Reopen only with a new, quality-equivalent scaling/intermediate representation.

### Output-head offload

- Wide W8 output-head routing failed the quality requirement.
- Padded high-precision M4 output-head execution passed correctness but was slower.

Keep the LM head on CPU until a genuinely efficient high-precision M=1 mechanism exists.

### Generic CPU-op delegation inside the plugin

Moving the same CPU work behind the RockNPU backend label did not remove dependencies and was slower. Do not optimize backend labels; remove or fuse real work.

### Activation quantization cache

Same-input quantization reuse is real, but measured host quantization cost is too small relative to whole-token latency to justify complex pointer/generation caching now.

### Artificial weight-reuse column splitting

Rejected as described above. The task-count cost dominates.

### RMSNorm / RoPE boundary experiments

Previous attempts to absorb small transformer glue operations into the RockNPU partition did not establish a reliable correctness-plus-throughput win. Reimplement only if a new graph/dataflow mechanism removes substantial work, not merely a partition boundary.

### Same-model speculative draft models

- Q2/Q4_0 requantized drafts were not useful.
- 11-layer alternating draft: fast standalone, effectively zero acceptance.
- 16/18-layer pruned drafts: very low acceptance.
- tested n-gram proposers were neutral or negative.

The verifier is capable; proposer acceptance remains the unresolved speculative-decoding problem.

### Raw FP16 persistence / giant sidecars

Large raw-FP16 persistence increased footprint/storage traffic too much. Do not trade steady-state memory and I/O for avoidable preparation work.

## Current-main M=1 profile — 2026-09-22

The first current-main profile closes the question of whether host quantization/rescale is still worth micro-optimizing.

Steady resident decode converges to exactly 88 prepared projection entries: 22 layers times four effective projection forms:

1. Q/V/K combined;
2. attention output;
3. gate/up combined;
4. FFN down.

For four hot decode tokens, 352 cache hits were observed: exactly 88 per token.

Per-call hot averages:

| Projection | Shape | Avg call |
| --- | --- | ---: |
| Q/V/K triple | K2048, total N2560 | 0.802 ms |
| attention output | K2048, N2048 | 0.784 ms |
| gate/up pair | K2048, total N11264 | 2.804 ms |
| FFN down | K5632, N2048 | 1.557 ms |

At 22 layers this is about 130.8 ms/token of projection time. Gate/up plus down account for about 95.9 ms/token, roughly 73% of the projection hot path.

The same profile measured only about 0.68 ms/token of activation quantization and about 0.85 ms/token of output rescale across the effective steady projection set. Host-side quantize/rescale work is therefore no longer a meaningful primary lever.

Conclusion: same-input projection grouping is already near its structural limit, and host micro-optimization of M=1 quantize/rescale is closed. Future gains must remove or restructure real model dataflow.

The opt-in `ROCKNPU_M1_PROFILE=1` diagnostic records hot-cache M=1 costs by shape without changing routing.

## Current hypotheses

### H1 — quality-equivalent userspace FFN dataflow

Highest priority.

FFN is about 73% of the measured M=1 projection hot path. Current execution is:

`gate/up W8 projection -> F32 SwiGLU -> down W8 projection`.

The next experiment should keep the exact current W8 projection semantics and F32 SwiGLU math, but execute the whole FFN sequence through one RockNPU userspace path so intermediate ownership and graph/backend handoffs can be removed.

This is deliberately different from the rejected fully quantized FFN experiments. Do not introduce a new hidden-state quantization domain in the first attempt.

Required order:

1. define a frontend-neutral F32 SwiGLU numerical contract;
2. add a direct FFN primitive/oracle using current gate/up and down W8 paths;
3. prove output equivalence against the existing graph path;
4. integrate behind an opt-in route;
5. deterministic TinyLlama generation gate;
6. same-process whole-token A/B.

A narrower 2026-09-22 subexperiment only moved split F32 SwiGLU into the
RockNPU GGML graph so gate/up, SwiGLU, and down no longer crossed a scheduler
partition. An initial adapter-side prototype called llama.cpp's
`ggml_vec_swiglu_f32`; that implementation was rejected because shared
execution must not depend on a frontend library. The experiment was then
reimplemented as a frontend-neutral Rust/C-ABI SwiGLU primitive. Its 8-token
greedy output was byte-identical to the baseline, but short adjacent n=8
performance pairs were inconsistent: about +3.84% in one order and -0.28% in
the reverse order, with a two-pair center of only about +1.8%. This is not a
validated speed win and the code was discarded.

Do not repeat the scheduler-boundary-only variant unchanged. H1 remains useful
only if the next composite FFN path removes more real work or intermediate
ownership than merely relabeling the CPU SwiGLU under the RockNPU backend.

### H2 — end-to-end NPU attention

Second priority.

TinyLlama GQA gives a useful mapping for decode: eight query heads share one KV head. Treating those heads as an M=8 batch can map attention matmuls onto the already validated FP16 geometry instead of the invalid FP16 M=1 path.

Candidate first slice for one GQA group:

- QK^T: M=8, K=64, N padded to 16;
- AV: M=8, K=context padded to 32, N=64;
- softmax remains CPU initially.

Start with a small exact/tolerance hardware oracle. Do not mirror KV state unless its consumers move with it.

### H3 — make M128 verifier capacity useful

M128 is a validated userspace win over M64. The remaining issue is workload generation: real speculative decoding needs a proposer with enough acceptance to exploit larger verification batches.

Focus on proposer mechanisms, not further verifier micro-optimization, unless profiling shows verifier cost again dominates.

### H4 — high-precision M=1 output head

Lower priority. Revisit only if a new M=1 high-precision mapping can avoid the four-row padding cost and the quality loss of W8.

### H5 — persistent packed format for cold start

Only a cold-start project. It is not a steady-state decode priority. Any format must remain provenance-bound to the source GGUF and must not duplicate the model with an excessive footprint.

## Priority order

1. quality-equivalent userspace FFN dataflow;
2. exact/tolerance M=8 GQA attention slice;
3. improve speculative proposer acceptance so M128 matters in general generation;
4. high-precision M=1 output head;
5. cold-start packed format.

## Benchmark rules

- Correctness first.
- Use independent references where possible.
- Record RockNPU commit, model hash, llama.cpp/plugin hashes, environment variables, CPU policy state, and test command.
- Separate cold preparation from hot execution.
- Distinguish local operator wins from whole-token wins.
- For small improvements, prefer same-process ABBA.
- Do not compare stale branch throughput with current main.
- Do not retain branches for failed hypotheses; record the conclusion and delete the branch/worktree.
- Keep temporary generated models and large scratch artifacts bounded.

## Branch/worktree policy

- main carries validated userspace mechanisms.
- At most one short-lived research worktree for a live hypothesis.
- Failed experiment: record result, revert/delete.
- Successful experiment: review, merge, delete the research branch/worktree.
- Do not use branches as research memory.
