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

## Verdict

**KEEP EXPERIMENTAL pending independent o16 reproduction and a model-level quality gate that exercises an actual 16-token batch.**

The performance mechanism is strong: `M=16` W8A8 changes NPU utilization enough to exceed same-board CPU pp16 throughput by about 8.7%. The next architectural use is speculative/batched target verification, not further M=1 micro-optimization.

