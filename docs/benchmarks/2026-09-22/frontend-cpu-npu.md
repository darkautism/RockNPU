# Frontend CPU/NPU check — 2026-09-22

Purpose: verify that RockNPU works through real user-facing frontends, not only primitive smoke tests.

Environment:

- RK3588 / o8g;
- packaged stock Rocket driver;
- NPU in the stock approximately 200 MHz class configuration;
- TinyLlama-1.1B-Chat-v1.0-Q4_K_M;
- model SHA-256: `5c66751b61537f9e55177b1b67e06af88e0e2df88f86de4909f5bf87fb1ae583`;
- llama.cpp tag `b10969`, commit `391fac16460f15233a7740550d858ac96df3419d`;
- Ollama 0.34.2, which pins the same llama.cpp tag;
- CPU policies 0/4/6 set to `performance` for timed comparisons.

These are compatibility/current-state numbers, not a full-speed NPU claim. The stock device configuration is intentionally kept unchanged.

## llama.cpp

Frontend: `llama-bench`.

Workload:

- decode only, 32 generated tokens;
- two repetitions in one process;
- four Cortex-A76 threads;
- CPU and NPU use the same GGUF and llama.cpp build.

| Path | Throughput |
| --- | ---: |
| CPU | 34.3449 tok/s |
| RockNPU | 7.2081 tok/s |

The RockNPU run used the real dynamic `libggml-rocknpu.so` backend with the validated W8 sidecar/direct-submit decode path.

## Ollama

Ollama integration requires no source patch. The current Ollama runner inherits:

- `GGML_BACKEND_PATH=/path/to/libggml-rocknpu.so`;
- `LLAMA_ARG_DEVICE=ROCKNPU0`.

RockNPU reports RK3588 shared system memory through the GGML device metadata so Ollama does not discard the accelerator as a zero-memory pseudo-device.

A trace run confirmed real RockNPU dispatch through Ollama: Q/V/K triples, attention-output projections, gate/up pairs, and FFN-down projections all executed through the W8A8 M=1 RockNPU path.

### Fixed-length warm decode comparison

For the final comparison, both paths used the same raw prompt:

`Paris` repeated 16 times.

Each server was warmed once, then measured twice with exactly 32 generated tokens, temperature 0, and four CPU threads. The NPU server used the validated W8 sidecar/direct-submit path. Both CPU and NPU completed all 32 requested tokens.

| Path | Run 1 | Run 2 | Two-run center |
| --- | ---: | ---: | ---: |
| CPU | 36.3885 tok/s | 36.2809 tok/s | **36.3347 tok/s** |
| RockNPU | 7.1327 tok/s | 7.1350 tok/s | **7.1339 tok/s** |

The stock-configuration NPU center is about **19.6% of CPU throughput** on this ordinary decode workload.

Prompt processing showed the same direction:

- CPU: about 597-600 prompt tok/s;
- RockNPU: about 116.9-117.0 prompt tok/s.

The deterministic continuations were not token-identical. The CPU continued repeating `Paris`; the current W8A8 RockNPU decode diverged after several tokens. This is consistent with the existing model-quality limitation of the approximate W8 decode path and must not be hidden by the frontend integration result.

### Interpretation

The frontend objective is successful:

- stock Ollama 0.34.2 loads the unmodified RockNPU GGML backend;
- a minimal Ollama smoke with only `GGML_BACKEND_PATH` and `LLAMA_ARG_DEVICE=ROCKNPU0` (no sidecar or direct-submit tuning) completed generation and produced real RockNPU FFN dispatch;
- Ollama discovery recognizes `ROCKNPU0`;
- no Ollama source patch is required;
- `LLAMA_ARG_DEVICE=ROCKNPU0` selects RockNPU through llama.cpp's standard device interface;
- trace evidence confirms real QKV, attention-output, gate/up, and FFN-down NPU dispatch.

The performance objective is not yet met: at the packaged stock NPU frequency, ordinary decode remains far behind the native ARM CPU.
