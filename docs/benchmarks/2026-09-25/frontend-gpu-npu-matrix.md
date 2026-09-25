# Frontend GPU/NPU benchmark matrix

## Scope and promotion rule

This matrix covers request-level LLM frontends that the repository can run on
RK3588. A row is a valid promotion candidate only when both GPU and NPU are
real backend/device paths under the same model, quantization, input, output
length, thread count, frequency, warmup, thermal policy, and timing method.
The default promotion threshold is NPU throughput >= 1.05x the same-board
GPU, with no generation-quality regression. CPU results are diagnostics and
must not be substituted for a missing GPU backend.

## Fixed conditions

- Boards: o8 and o16, RK3588.
- Model: TinyLlama-1.1B-Chat-v1.0-Q4_K_M.
- Model SHA-256: `5c66751b61537f9e55177b1b67e06af88e0e2df88f86de4909f5bf87fb1ae583`.
- llama.cpp uses the existing Vulkan build at commit `391fac164`, native A76,
  `-t 4`, `-fa on`, four CPU workers, and one resident request.
- NPU is checked at `fdab0000.npu/cur_freq=target_freq=700000000`.
- GPU is checked at `fb000000.gpu/cur_freq=target_freq=1000000000`.
- Thermal snapshots are taken before and after every timed process.
- llama.cpp formal workload: prompt length 0 (decode-only), 32 generated
  tokens, `-r 1`, three interleaved ABBA blocks (12 processes/board).
- The quality command is separately rerunnable with a fixed `Paris` request,
  temperature 0, fixed seed, and independent CPU oracle response.

## Full frontend rows

| Frontend | GPU backend/device | NPU backend/device | Model/quant | Workload and timing | Dispatch/quality evidence | Status |
|---|---|---|---|---|---|---|
| stock llama.cpp | Vulkan / `Vulkan0`, Mali-G610 | `ROCKNPU0` with W8 sidecar | TinyLlama Q4_K_M; exact SHA above | 0 prompt + 32 decode tokens; 3 ABBA blocks; 4 threads; default llama-bench warmup; model load excluded | `raw/llama-gpu-npu-repro-o8/` and `raw/llama-gpu-npu-repro-o16/`; each NPU run has nonzero `w8a8_m1` dispatch; GPU/NPU quality harness is the next gate | Formal performance baseline complete; quality gate pending |
| stock Ollama | **Unavailable in the isolated stock runtime**: no `libggml-vulkan.so` and no valid Vulkan device row; see `raw/ollama-gpu-discovery-o8.txt` / `...o16.txt` | `ROCKNPU0` route exists, but formal quality comparison is currently failed | TinyLlama Q4_K_M; exact SHA above | 8-token API diagnostic; no valid same-frontend GPU baseline can be run in this environment | NPU device/dispatch evidence exists, but it cannot be compared to a GPU row until a stock Ollama Vulkan backend is supplied | GPU comparison blocked/unsupported; do not count CPU as GPU |

Candle `RockNpuLinear` and the ONNX importer are adapter/operator slices,
not complete request-level LLM frontends, and are not rows in this matrix.

## llama.cpp formal results

| Board | GPU tok/s | NPU tok/s | NPU/GPU | Block ratios | Interpretation |
|---|---:|---:|---:|---|---|
| o8 | 6.148 | 7.314 | 1.190x | 1.887, 1.034, 0.979 | Mean clears 1.05x, but variance is too high for promotion |
| o16 | 6.076 | 6.577 | 1.082x | 1.918, 0.849, 0.870 | Mean clears 1.05x, but two blocks are below GPU |

These are performance results only. They are not a promotion until the
GPU/NPU quality oracle passes and the block-level stability rule is applied.
Every command, environment, frequency/thermal snapshot, stdout/stderr,
backend row, JSON result, and dispatch summary is versioned in the two raw
 directories.

## Required next evidence

1. Run `scripts/bench_llama_gpu_quality.py` on both boards with Vulkan0 as the
   GPU reference and `ROCKNPU0` as NPU; compare both against a fresh CPU
   oracle.
2. Add the quality raw files and hashes to the board evidence index.
3. Investigate the o16 block variance and require a predeclared stability
   rule before considering promotion.
4. Obtain a stock Ollama runtime with a real Vulkan backend before comparing
   Ollama NPU against GPU; do not modify Ollama frontend source to create one.

## Current conclusion

The llama.cpp NPU path is a plausible GPU-acceleration candidate, but the
present evidence is a baseline matrix, not a completed promotion. Ollama has
no valid GPU denominator in the current stock environment, so its GPU/NPU
promotion status is explicitly blocked rather than inferred from CPU.
