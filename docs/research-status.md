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

### Dynamic llama.cpp / GGML backend

- Stock llama.cpp can dynamically load libggml-rocknpu.so; no llama.cpp fork is required.
- Supported model paths are explicit. Unsupported work remains outside the RockNPU backend instead of being silently emulated.
- Q4_K and Q6_K TinyLlama projections are exercised through the real plugin.

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

## Current hypotheses

### H1 — reduce ordinary M=1 projection cost from userspace

Highest priority.

Existing QKV and gate/up grouping prove that larger same-input work can help. Find remaining opportunities to reduce real projection work or repeated data movement while preserving current W8 semantics.

Test requirements:

- same current-main binary family;
- deterministic output gate;
- whole-token A/B;
- for gains below 5%, use same-process warm/interleaved ABBA.

### H2 — end-to-end NPU attention

Potentially high upside, but only useful as a complete dataflow change.

A useful implementation would keep K/V in a representation directly consumed by QK^T, softmax or an equivalent numerically validated attention step, and AV.

Simply mirroring KV state without moving its consumers is expected to lose.

Start with a small exact attention oracle before integrating with TinyLlama.

### H3 — quality-equivalent FFN intermediate retention

Goal: avoid unnecessary host round-trips between gate/up, activation, multiply, and down projection.

The next attempt must define a scale/intermediate domain that preserves model quality. Local integer exactness is insufficient.

Required order:

1. numerical contract;
2. primitive oracle;
3. layer differential;
4. deterministic generation;
5. whole-token A/B.

### H4 — make M128 verifier capacity useful

M128 is a validated userspace win over M64. The remaining issue is workload generation: real speculative decoding needs a proposer with enough acceptance to exploit larger verification batches.

Focus on proposer mechanisms, not further verifier micro-optimization, unless profiling shows verifier cost again dominates.

### H5 — high-precision M=1 output head

Lower priority. Revisit only if a new M=1 high-precision mapping can avoid the four-row padding cost and the quality loss of W8.

### H6 — persistent packed format for cold start

Only a cold-start project. It is not a steady-state decode priority. Any format must remain provenance-bound to the source GGUF and must not duplicate the model with an excessive footprint.

## Priority order

1. profile current ordinary M=1 decode on current main;
2. measure existing QKV / gate-up grouping and find the next userspace dataflow reduction;
3. prototype a small exact NPU-attention slice;
4. investigate a quality-equivalent FFN intermediate domain;
5. improve speculative proposer acceptance so M128 matters in general generation;
6. output-head and cold-start work only after the above.

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
