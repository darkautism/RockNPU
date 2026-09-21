# RK3588 decode research consolidation — 2026-09-21

This checkpoint closes the branch/worktree sprawl accumulated during the TinyLlama RK3588 decode investigation. The rule after this checkpoint is simple: keep validated production mechanisms on `main`, keep durable negative/pending conclusions in documentation, and do not keep stale implementation branches merely as research memory.

## Authoritative benchmark state

All comparable RK3588 CPU/NPU measurements must set **all** CPU cpufreq policies (`policy0`, `policy4`, `policy6`) to `performance`. On this board, `policy0` also changes the shared SCMI DSU clock:

- `policy0=ondemand`: DSU 1.2 GHz, 4xA76 sequential read about 12.47 GB/s, native CPU `tg128` about 14.62 tok/s.
- `policy0=performance`: DSU 1.8 GHz, 4xA76 sequential read about 25.29 GB/s, native CPU `tg128` about 34.29 tok/s.
- Changing the NPU clock between 200 and 700 MHz did not cause this CPU throughput change. DDR remained at 2.112 GHz.

With the corrected all-policy performance state and NPU fixed at 1 GHz, the repeated 100%-acceptance lookup-speculative M64 workload measured:

- RockNPU run 1: 513 decoded tokens in 2.940 s = **174.507 tok/s**, 504/504 accepted.
- RockNPU run 2: 513 decoded tokens in 2.886 s = **177.756 tok/s**, 504/504 accepted.
- CPU target: 513 decoded tokens in 8.257 s = **62.130 tok/s**, 504/504 accepted.

Therefore the corrected verifier/decode speedup is about **2.81x to 2.86x CPU** on this workload. The two fresh-process RockNPU runs center around ~176 tok/s. This supersedes the older 127-149 tok/s checkpoint figures that were taken without a complete DSU/cpufreq contract.

For ordinary one-token generation, the corrected native CPU baseline is about **34.2-34.4 tok/s**. The current RockNPU M=1 direct-submit/persistent-scratch path measured **17.11 +/- 0.69 tok/s** in the corrected governor state. Ordinary M=1 remains a separate problem from the validated M64 verifier win.

## Branch/worktree audit

### Already promoted to main

The following mechanisms are already represented in current `main`; their old research worktrees/branches are redundant and should be removed:

- native W8 direct-submit and persistent scratch;
- H1-B parallel Q4_K/Q6_K prefill dequant;
- H1-B child `Arc<Vec<f16>>` allocation reuse;
- W8 M-tile routing and scope control;
- RK3588 host NEON preparation;
- native M16/M32/M48/M64 verifier routing;
- native M64 FFN-down three-core K-split and prewarm.

### Closed / do not resurrect without new evidence

The source prototypes can be deleted because their durable conclusions are already captured here and/or in `trialanderror.md`:

- batched M=1 K-split / PC-chain prototypes: enabling primitive only; no material whole-model gain, and true chained execution requires an explicit userspace/kernel batched-job contract;
- FFN-middle custom EWMUL/PC-chain prototypes: not productionized; stale against the later M64/K-split path;
- CPU SDOT K/V routing: host dot-product work did not improve the relevant whole-model path;
- raw-hidden FFN / compact-int16 hidden fusion: semantic or throughput regression; raw-int32 hidden path restored deterministic quality but regressed whole-token throughput;
- W4A4/W4 FFN quality experiments: not model-quality equivalent and no material whole-model win;
- naive fused SwiGLU routing: rejected;
- Q-only RoPE routing: rejected;
- RMSNorm boundary-collapse experiment: rejected;
- H1-C raw FP16 persistence: rejected because storage/page-fault cost and disk footprint dominate;
- H1-A eager prepare: rejected as a throughput improvement in the current stock-GGML integration.

### Pending idea retained as documentation only

`research/resnorm-bridge-0920` commit `bd53fe2` attempted to reduce the historical ~221 GGML backend splits by handling TinyLlama decode `ADD`, `RMS_NORM`, and norm `MUL` inside the RockNPU partition. No reliable correctness + throughput result was found for that branch. Its implementation is based on a stale pre-M64 tree and should **not** be merged or benchmarked directly against current main. Keep the hypothesis only: if backend-split overhead becomes material again after current M64 work, reimplement the minimal bridge on current main and measure it under the corrected DSU contract.

## Speculative proposer results after DSU correction

The verifier is no longer the primary uncertainty. Proposer quality determines whether the M64 win applies to general generation.

Observed general-prompt lookup results remain low-acceptance. A Rust prompt with lookup cap 47 improved from 13.72 tok/s under the bad DSU state to 20.75 tok/s after fixing `policy0`, with the same ~16.45% acceptance. This confirms the host/DSU state was suppressing throughput, but lookup proposer quality still prevents a general win.

Same-model draft experiments were closed:

- Q2_K and Q4_0 requantized drafts did not provide a useful path;
- an 11/22-layer alternating draft ran ~61.6 tok/s standalone but produced 0/480 accepted tokens in a short speculative gate;
- a 16-layer tail-pruned draft accepted ~0.95%; an 18-layer tail-pruned draft was worse;
- middle-layer pruning was not pursued because the pinned `llama-quantize --prune-layers` path failed for non-tail surgery and there was no evidence justifying custom tooling.

N-gram experiments:

- `ngram-mod` approximately preserved a normal Rust baseline (31.2 vs 31.4 tok/s) but regressed a repeated prompt (24.9 vs 30.5 tok/s);
- `ngram-simple` with a 16-token exact match preserved the normal baseline and did not materially accelerate the repeated prompt (31.0 vs 30.5 tok/s).

Do not retain branches for these negative proposer experiments.

## Working rules after cleanup

1. Do not create a new long-lived branch for every performance hypothesis.
2. Use one current research branch/worktree at a time; negative experiments are reverted and recorded, not archived as branches.
3. Promote validated code to `main` quickly; after merge, remove the research branch and worktree.
4. Every benchmark must record all CPU policies, DSU state when available, NPU frequency, model hash, binary/plugin provenance, and exact environment.
5. Do not compare stale branch throughput directly against current main after major verifier changes; port the mechanism forward first if the hypothesis still has merit.
6. Keep `/tmp` bounded. Temporary draft GGUFs filled the tmpfs during this investigation and were removed after the experiments closed.
