# 2026-09-25 NPU/CPU benchmark summary

## Environment

- Boards: o8 and o16, RK3588.
- CPU: native A76 build, policies 0/4/6 `performance`, four A76 workers.
- NPU: README-pinned external DVFS module commit `ed52a89afa8e68fedf636c8e891bd8fc47e82d26`, verified at 700 MHz.
- Model: TinyLlama-1.1B-Chat-v1.0-Q4_K_M, SHA-256 `5c66751b61537f9e55177b1b67e06af88e0e2df88f86de4909f5bf87fb1ae583`.
- Workload: fixed `Paris` prompt, temperature 0, 8 or 32 generated tokens, warm resident state, same native no-repack llama binary.

## Correctness and integration

- llama.cpp no-repack NPU smoke on both boards: backend/device `ROCKNPU0`, native W8 dispatch 1386 per smoke, model hash and sidecar v2 hash matched.
- Ollama 0.34.4 isolated stock runtime on both boards: `ROCKNPU0` discovered, `ROCKNPU_HOST` used for KV/compute and, with the documented prefill-cache route, 549.40 MiB of model weights. Per-op trace showed `path=w8a8_m1` (66 o8 operations and 179 o16 operations in the recorded short request).
- Ollama CPU and NPU deterministic 8-token responses were identical: `Yes, the French capital has a rich`.
- The host-buffer adapter change is a correctness/routing fix: it gives RockNPU an independent host buffer identity and prevents the frontend from silently using `CPU_REPACK` as a false NPU path.

## Three-block hot A/B

| Board/path | CPU tok/s center | NPU tok/s center | NPU/CPU | Gate |
|---|---:|---:|---:|---|
| llama.cpp o8, 32 tokens | 32.68 | 16.35 | 0.50x | FAIL |
| llama.cpp o16, 32 tokens | 33.96 | 16.59 | 0.49x | FAIL |
| llama.cpp c8 diagnostic, 8 requests | 32.32 | 32.46 | 1.003x | FAIL (<5%) |
| Ollama c1 hot diagnostic, 8 tokens | 40.1 | 19.1 | 0.48x | FAIL |
| Ollama c16 diagnostic | not completed | not completed | — | INVALID/timeout |

Raw A/B files:

- `artifacts/bench-700-cpu-npu-final-o8/`
- `artifacts/bench-700-cpu-npu-final-o16/`
- `/tmp/c8-*` and `/tmp/c16-*` diagnostic logs (not promoted).

## Profiling conclusion

At 700 MHz, the M1 profile attributes most time to NPU execute/wait:
K2048/N2048 about 0.28 ms, K5632/N2048 about 0.53 ms, and the FFN pair
K2048/N11264 about 0.91 ms. Host quantization/rescale is not the dominant
cost. Direct-submit, scratch, scheduler-routing, K-split, M8 and concurrency
variants did not produce a stable 5% whole-model NPU win.

## Boundary check

Only RockNPU userspace middleware/adapter, benchmark tooling and docs were
changed. No frontend source and no kernel-driver source were modified. The
700 MHz module is an external environment dependency and its source was not
edited.

## Status

Correctness and integration gates pass. The required NPU-over-CPU performance
objective is not met on either board; this document records the blocker rather
than claiming completion.
