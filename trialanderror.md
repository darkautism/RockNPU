# RockNPU trial and error ledger

Purpose: preserve userspace decisions so future work does not repeat closed experiments.

Confidence:
- C5: repeated real-hardware evidence or independent model validation.
- C4: strong repeated evidence, but narrower scope.
- C3: useful evidence with material remaining uncertainty.

Disposition:
- KEEP: validated production direction.
- CLOSED: do not repeat without a materially new mechanism.
- OPEN: worth further work.

## Validated production directions

| Decision | Confidence | Status |
| --- | ---: | --- |
| Use W8A8/native quantized M=1 for TinyLlama decode. Plain FP16 M=1 is not the production route. | C5 | KEEP |
| Keep decode weights resident and reuse persistent scratch. Repacking/dequantizing every call is obsolete. | C5 | KEEP |
| Tune worker count/split by shape instead of hard-coding a topology. | C5 | KEEP |
| Keep V/K concat-N pairing default-on. It is bit-exact under the existing W8 semantics and has repeatable whole-token benefit. | C5 correctness / C4 perf | KEEP |
| Keep gate/up concat-N pairing default-on. It is bit-exact under the existing W8 semantics and has repeatable whole-token benefit. | C5 correctness / C4 perf | KEEP |
| Keep Q/V/K triple grouping. It executes as one larger projection and uses backend-local output stashing to bridge the GGML partition boundary. | C5 | KEEP |
| Keep native M16/M32/M48/M64/M128 routes. Fresh M128 vs M64 A-B-B-A remains positive by about 10.4% at the two-run centers. | C5 correctness / C4 perf | KEEP |
| Keep the full-K M=1 K=5632 FFN-down path and measured multi-worker N split. | C5 | KEEP |
| Keep FP16 fused residual for compatible prefill projections. | C5 | KEEP |
| Keep same-job INT8 weight reuse as a primitive, but only when the natural workload already satisfies its geometry. | C5 | KEEP |
| Keep parallel Q4_K/Q6_K prefill preparation and child allocation reuse. | C4 | KEEP |
| Keep architecture-specific host preparation such as the validated NEON path when it reduces measured userspace work. | C4 | KEEP |
| Stock llama.cpp dynamic backend loading is sufficient; do not maintain a llama.cpp fork merely to load RockNPU. | C5 | KEEP |
| Correctness gates must include an independent model/reference path when model semantics are involved. | C5 | KEEP |
| Replay cached regcmds per resident weight and skip redundant BO syncs in the direct M=1 path (NPU decode 18.5 → 20.9 tok/s, bit-identical). | C4 | KEEP |
| Convert GGUF weights to W8 in-process; the sidecar is only a startup cache (KL bit-identical). | C5 | KEEP |
| Direct-submit M-tile prefill with per-core parallel staging, fused rescale, same-input grouping and any-M tiling (pp128 116 → 496 at defaults; bit-identical KL). | C5 correctness / C4 perf | KEEP |
| Recommend `GOMP_SPINCOUNT=20000` for frontends (pp128 496 → 551; CPU decode unchanged). | C4 | KEEP |
| Default single-sequence decode on the CPU (`ROCKNPU_DECODE=cpu`); decode is DRAM-bound on LPDDR4X and Q4_K moves fewer bytes than W8. NPU/hybrid stay opt-in. | C4 | KEEP |

## Closed experiments

| Experiment | Result | Confidence | Status |
| --- | --- | ---: | --- |
| Plain FP16 M=1 | Large mismatches; invalid production geometry. | C5 | CLOSED |
| Pad FP16 M=1 to M=4 | Can be correct, but loses badly at whole-model scale. | C5 | CLOSED |
| Generic CPU delegation inside RockNPU backend | Preserves work instead of removing it; slower. | C4 | CLOSED |
| GGML-only SwiGLU boundary collapse | Directly calling libggml-cpu from the adapter violates the frontend-neutral core direction. A corrected Rust/C-ABI SwiGLU version preserved an 8-token greedy output byte-for-byte, but adjacent n=8 A/B orders were inconsistent (+3.84% then -0.28%; about +1.8% two-pair center). No validated speed win. | C4 | CLOSED |
| Same-input activation quantization cache | Real reuse opportunity, but too small to justify complexity at current latency. Current-main M=1 profiling puts all steady activation quantization at only about 0.68 ms/token. | C5 | CLOSED |
| M=1 host quantize/rescale micro-optimization | Current-main hot profile shows about 0.68 ms/token quantize and 0.85 ms/token rescale across the effective steady projection set; not a primary lever. | C5 | CLOSED |
| Forced N64 segmentation for weight reuse | M256/N2048 about 2.5x slower; M256/N256 about 1.73x slower. | C5 | CLOSED |
| W8 output-head offload | Fails quality requirement. | C5 | CLOSED |
| Padded high-precision M4 output head | Correct, but slower. | C4 | CLOSED |
| W4A4 default decode | Primitive is fast, but generation quality diverges and whole-model speedup is not established. | C5 | CLOSED |
| Compact-int16 FFN hidden state | Hardware exact under constructed integer oracle, but TinyLlama semantics diverge. | C5 | CLOSED |
| Raw-int32 hidden FFN route | Restores deterministic quality but regresses whole-token throughput. | C4 | CLOSED |
| Naive fused SwiGLU / fully quantized FFN | Current scale/activation semantics do not preserve model quality. | C5 | CLOSED |
| Q-only RoPE routing | No useful whole-model result. | C4 | CLOSED |
| RMSNorm boundary-collapse experiments | No reliable correctness plus throughput win. | C4 | CLOSED |
| Giant raw-FP16 sidecar / persistence | Storage and memory footprint dominate. | C4 | CLOSED |
| Q2/Q4_0 speculative drafts | No useful general speculative route. | C4 | CLOSED |
| 11-layer alternating speculative draft | Fast standalone, essentially zero acceptance. | C5 | CLOSED |
| 16/18-layer pruned speculative drafts | Very low acceptance. | C4 | CLOSED |
| Tested n-gram proposers | Neutral or negative on representative prompts. | C4 | CLOSED |
| CPU thread-count escalation for TinyLlama | More threads are not a stable win; four A76 threads remain the reference. | C4 | CLOSED |
| Head-only extra CPU threads | Exact but slower. | C4 | CLOSED |
| Hybrid CPU/NPU M=1 row split as default decode | Best 26.2 tok/s vs CPU 32–33; CPU share slows ~1.4× under NPU DMA (serial/overlap A/B). Kept opt-in. | C4 | CLOSED |
| Hybrid helper thread (big cores or A55 cluster) | Oversubscribes the OpenMP team or slows NPU submission; replaced by the overlap callback. | C4 | CLOSED |
| NPU 1 GHz (850 mV) for LLM work | Decode primitive and pp128 unchanged within 2 % vs 700 MHz. | C4 | CLOSED |
| `-fa off` with NPU prefill | pp128 452 → 326. | C4 | CLOSED |
| Wide shapes on the native M-tile path (K-split to 12288, 64-row halves, N chunks, K zero-padding, pool from N ≥ 768) | Llama‑3.2‑1B pp128 260 → 580, Qwen2.5‑1.5B 31 → 311; TinyLlama KL bit-identical. | C4 | KEEP |
| Two-part activation encoding (`ROCKNPU_PREFILL_HILO`) as default | KLD 4–5× lower but prefill speed halves (`down`: ~2× lower KLD, −23 %). Opt-in only. | C4 | CLOSED as default |
| LLM.int8()-style fixed outlier channels for W8A8 prefill | Outliers are per-token over tens–hundreds of channels; scale gain only 2–4×. | C4 | CLOSED |
| Prefill Q+V+K as one concatenated M-tile call (Q node stashes V/K for the later split) | Steady pp128 556 → ~567 (+2 %), but the first measured run drops to 290 (concat weight build) and a third resident W8 copy of Q/K/V is kept (~115 MB TinyLlama, ~440 MB at 3B). Not worth the memory on 8 GB boards. | C4 | CLOSED |

## Important measurement lessons

1. A local primitive win is not a model win.
2. Small whole-token deltas require warm, interleaved A/B. Adjacent independent processes are too noisy.
3. Auto-tuned worker topology can change between runs; record the chosen topology.
4. Separate first-use preparation/cache fill from hot execution.
5. Do not infer execution capability from a packed-weight storage envelope.
6. A short deterministic prompt is not a complete quantization-quality proof.
7. Exact integer primitive results do not prove model-level equivalence.
8. Keep model hash, plugin hash, llama.cpp revision, environment variables, and CPU policy state with benchmark evidence.
9. Old absolute NPU throughput collected under mixed system configurations is not an authoritative current baseline.
10. Failed experiment code should be deleted after the conclusion is recorded.

## Userspace history

### GGML / llama.cpp integration

The backend progressed from a narrow proof-of-concept to a real stock-llama.cpp dynamic plugin.

Important milestones:

- F16, Q4_K and Q6_K MUL_MAT routes validated against independent references.
- resident W8 decode cache established;
- persistent direct-submit scratch established;
- V/K and gate/up same-input pairing promoted;
- Q/V/K triple grouping promoted across the partition boundary;
- native M-tile path expanded through M128.

### TinyLlama correctness

The real GGUF path has repeatedly matched independent llama.cpp / llama-gguf references on stable greedy sequences. Near-tie logit flips are treated as numerical-precision cases rather than hidden correctness failures.

### M=1 decode evolution

The initial FP16 route was rejected. W8A8 became the useful decode route, then gained:

- resident weights;
- persistent scratch;
- multi-worker shape tuning;
- full-K K=5632 support;
- grouped projections;
- native sidecar support.

Current-main profiling on 2026-09-22 found exactly 88 steady prepared projection entries, or four effective projection forms per layer. Per-call hot averages were about 0.802 ms Q/V/K, 0.784 ms attention-output, 2.804 ms gate/up, and 1.557 ms FFN-down. Gate/up plus down account for about 73% of projection time. Host quantize/rescale is negligible by comparison.

Therefore ordinary M=1 remains the largest performance gap, but further host quantize/rescale tuning is closed. The next target is real FFN/dataflow work.

### Prefill / verifier evolution

Prefill and verifier work gained:

- resident/prepacked weights;
- parallel preparation;
- native M16/M32/M48/M64/M128;
- fused residual;
- controlled M128 improvement over M64.

This area is no longer the first bottleneck unless profiling says otherwise.

### Quantization experiments

W4A4 demonstrated real hardware capability and strong local primitive speed, but model quality and whole-model throughput did not justify promotion. Future W4 work must begin with a better quantization/calibration contract rather than another worker-count sweep.

### Speculative decoding

Verifier capacity is not enough. General speculative decoding remains blocked by proposer quality/acceptance. The failed draft-model and n-gram experiments should not be repeated unchanged.

## Open work

See docs/research-status.md for the canonical current hypotheses and priority order.
