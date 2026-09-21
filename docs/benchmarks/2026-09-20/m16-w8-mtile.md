# W8A8 M=16 tile experiment

## Scope

This experiment adds an opt-in RK3588 W8A8 `M=16` matmul path for TinyLlama-sized prompt/speculative-verification batches. Production `M=1` routing is unchanged.

Validated environment on o8g:

- RockNPU base: `7507e07`
- llama.cpp: `391fac16460f15233a7740550d858ac96df3419d`
- model: `TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf`
- model SHA256: `5c66751b61537f9e55177b1b67e06af88e0e2df88f86de4909f5bf87fb1ae583`
- NPU: 700 MHz
- A76 policies: `performance`
- model-level opt-in: `ROCKNPU_W8_MTILE=1`
- final warmed configuration also uses `ROCKNPU_MTILE_PERSIST=1 ROCKNPU_PREFILL_CACHE=1`

## Hardware correctness

The M-tile encoder is restricted to the hardware-proven `M=16` envelope. A larger `M=32` probe wrapped at row 16; a six-register rkllm-style override did not fix it, so no M32 API remains in the candidate.

Final clean-source gates:

- `M=16 K=2048 N=2048`: exact all-rows int32 oracle PASS. Submit/wait samples: `582.158, 509.534, 449.451, 541.617, 540.158 us`.
- `M=16 K=5632 N=64`: six-slice differential PASS against 16 independent M=1 executions. `m16_us=829.486`, `submit_wait_us=183.455`, `host_accum_us=64.749`.

The K=5632 implementation is retained as research evidence only. Routing TinyLlama's down projection through it reduced model-level pp16 throughput, so the final model route remains `K<=4096` and leaves the down projection on the resident prefill fallback.

## Model-level performance

Same-board pure CPU `pp16`:

- r=3: `71.4097, 71.3701, 71.2533 tok/s`
- mean: `71.344361 tok/s`
- later r=1 confirmation: `71.411107 tok/s`

Final warmed RockNPU configuration:

```text
ROCKNPU_W8_MTILE=1
ROCKNPU_MTILE_PERSIST=1
ROCKNPU_PREFILL_CACHE=1
```

`pp16,r=5` samples:

```text
77.8041
78.2193
77.4184
78.0906
76.1117 tok/s
```

Mean: `77.528817 tok/s`.

Against the r=3 same-board CPU mean, this is approximately `1.0867x` / `+8.7%` throughput.

For context, the old M>1 FP16 bridge measured only about `3.91 tok/s` at pp16. The M16 W8 path therefore removes the main small-M NPU utilization failure, rather than merely shaving host overhead.

## Negative variants

- Routing `K=5632` down through M16 K-split was numerically correct but model-level pp16 fell to `64.758086 tok/s`; keep down on the fallback path.
- M32 general geometry repeats/wraps rows after row 15. The attempted rkllm-style six-register override did not restore correctness; reject M32 until a new mechanism is found.
- M16 persistent scratch without resident fallback was misleadingly slow because the remaining FP16 fallback dominated the request. The final performant configuration requires resident prefill cache for the unsupported down projection.
- Batched M=1 K-split PC-chain is a useful enabling primitive but reversed whole-model A/B showed no material throughput gain by itself.

## Independent o16 reproduction

Exact candidate: `d6497fa67dfdb3bef51c112f53b381685aaf161e` in detached `/build/rocknpu-w8-mtile-validator`.

Metadata:

- model SHA256 identical: `5c66751b61537f9e55177b1b67e06af88e0e2df88f86de4909f5bf87fb1ae583`
- NPU: 700 MHz
- llama.cpp: same `391fac164...` commit

Hardware gates independently passed:

- `M=16 K=2048 N=2048`: exact row oracle PASS, submit/wait `517.991, 505.741, 517.699, 477.741, 426.118 us`.
- `M=16 K=5632 N=64`: six-slice M16-vs-M1 differential PASS, `m16_us=821.027`, `submit_wait_us=178.206`, `host_accum_us=60.374`.

Clean sequential pp16 comparison (an earlier CPU run overlapped the candidate and is discarded):

- CPU r=5: mean `73.674390 tok/s`; samples `73.8521, 73.7054, 73.5859, 73.6230, 73.6056`.
- M16 candidate r=5: mean `86.478190 tok/s`; samples `88.6623, 87.9665, 86.6163, 82.4952, 86.6507`.
- Independent speedup: about `1.174x` / `+17.4%` throughput.

## Model-level 16-token quality gate

Prompt:

```text
The capital of France is Paris, and the capital of Germany is Berlin.
```

Pinned llama.cpp reports this prompt as exactly 16 tokens. With temperature 0:

- pure CPU and the same RockNPU candidate with `ROCKNPU_W8_MTILE=0` produced the same 24-token continuation through `Washington, D.C.` and continued with `3. The capital of ...`;
- `ROCKNPU_W8_MTILE=1` shared the same prefix through `Washington, D.C.` but then diverged and continued with `and the capital of Canada is Ott...`.

The candidate output remained semantically coherent, but the deterministic continuation is **not byte-exact** once the M16 W8 prompt path is enabled. Therefore the M16 path has a real model-level numerical effect even though integer matmul hardware gates are exact. This is consistent with row-wise activation/weight quantization changing logits, not a broken Rocket matmul.

## Projection-scope diagnosis and groupwise Q/O recovery

The known divergent 16-token France/Germany prompt was used to isolate M16 W8 quantization sensitivity by projection shape:

- `kv` only (`K=2048,N=256`): matched the CPU continuation; 44 M16 calls.
- `qo` only (`K=2048,N=2048`): reproduced the `Canada is Ott...` divergence; Q/O is a clear sensitive class.
- `ffn` only (`K=2048,N=5632` gate/up): matched the CPU continuation; 42 M16 calls, but pp16 averaged only `44.685219 tok/s`, so simply falling sensitive projections back to FP16 loses too much throughput.
- `safe` (`kv+ffn`, Q/O on FP16): still diverged. Individually tolerable quantization errors can accumulate across projection classes; a simple shape allowlist is not a complete quality bridge.

A new opt-in Q/O groupwise W8 path therefore splits `K=2048` into two `K=1024` groups. Each group has independent activation and per-output-channel weight scales, runs the same exact M16 INT8 primitive, and the scaled partials are accumulated in FP32. `ROCKNPU_MTILE_QO_GROUP=1024` enables this research path.

With all other eligible projections still using M16 W8, the 2-way Q/O path restored the previously divergent France/Germany continuation exactly to the CPU output. It also preserved the second exact-16-token France/China continuation that already matched CPU.

On o8, before grouped persistent scratch, pp16 r=3 averaged `72.132268 tok/s` (`71.7717, 72.0200, 72.6051`), already slightly above the same-board CPU baseline. After reusing persistent M16 scratch across the two Q/O groups, pp16 r=5 averaged `75.989331 tok/s` (`75.1190, 74.9143, 74.9151, 77.7588, 77.2395`), approximately `+6.5%` over the o8 CPU mean `71.344361 tok/s` while retaining both deterministic 16-token quality gates.

Independent o16 validation used exact follow-up commit `38ae896a792efdd311f377e9b45aaaf3af538d86`. The previously divergent France/Germany prompt again matched the CPU continuation exactly with 2-way Q/O grouping. Same-board pp16 r=5 averaged `84.051609 tok/s` (`84.5751, 84.4015, 83.2001, 83.1550, 84.9263`) versus the established o16 CPU mean `73.917624 tok/s`, approximately `+13.7%` throughput. The quality-restoring mechanism and performance gain therefore reproduce independently on both RK3588 boards.

## Verdict

**KEEP EXPERIMENTAL, but the quality/performance blocker now has a viable mechanism.**

The original ungrouped M16 path remains unsuitable for promotion because model-level divergence is real. The 2-way Q/O groupwise path now restores the known divergent deterministic gate on both o8 and o16, preserves the second o8 deterministic gate, and remains faster than CPU on both boards. Broader numerical/quality coverage is still required before promoting the model route as a default.

The next high-value gate is broader M16 numerical/quality coverage, followed by speculative/batched target verification. Further M=1 micro-optimization is lower priority.

