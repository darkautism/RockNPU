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

## Closed experiments

| Experiment | Result | Confidence | Status |
| --- | --- | ---: | --- |
| Plain FP16 M=1 | Large mismatches; invalid production geometry. | C5 | CLOSED |
| Pad FP16 M=1 to M=4 | Can be correct, but loses badly at whole-model scale. | C5 | CLOSED |
| Generic CPU delegation inside RockNPU backend | Preserves work instead of removing it; slower. | C4 | CLOSED |
| Same-input activation quantization cache | Real reuse opportunity, but too small to justify complexity at current latency. | C4 | CLOSED |
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

Ordinary M=1 decode remains the largest performance gap.

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
