# Frontend NPU/CPU benchmark matrix — 2026-09-25

## Scope

This matrix is the reproducibility contract for the active goal. A full-LLM
frontend passes only when the unmodified frontend can run the same complete
request/decode workload on the native CPU backend and on RockNPU, with genuine
backend execution and an independent quality gate.

Candle `RockNpuLinear` and the ONNX importer are adapter/operator slices, not
complete LLM frontends. They are not counted as full-LLM passes unless this
repository gains a complete request-level benchmark for them.

## Shared environment

- Boards: `o8` and `o16`.
- SoC: RK3588.
- Model: TinyLlama-1.1B-Chat-v1.0-Q4_K_M.
- Model SHA-256:
  `5c66751b61537f9e55177b1b67e06af88e0e2df88f86de4909f5bf87fb1ae583`.
- llama.cpp: `391fac164`, native A76 build, `GGML_NATIVE=ON`,
  `GGML_CPU_REPACK=OFF`; the same executable is used for CPU and NPU.
- NPU benchmark environment: README-pinned Rocket DVFS checkout
  `ed52a89afa8e68fedf636c8e891bd8fc47e82d26`, requested and verified at
  **700 MHz**. The external module is an environment dependency and is not
  modified by this project.
- CPU: policies 0/4/6 set to `performance`; four Cortex-A76 workers; policy 0
  at 1.8 GHz and policies 4/6 at 2.4 GHz when available.
- llama.cpp prompt: `Paris` repeated 16 times; Ollama API prompt: `Paris`.
- Decode: 32 generated tokens for llama.cpp and 8 generated tokens for the fixed Ollama quality/A-B diagnostic, temperature 0, fixed context and model hash.
- Initial concurrency: one request per frontend. Additional concurrency points
  use the same fixed workload at each tested value.
- Timing: resident caches warm; model loading and first preparation are
  excluded. At least three interleaved hot A/B blocks are required.
- Acceptance: NPU hot decode throughput is at least 5% above CPU and the
  quality gate does not regress.

## Full-LLM frontend rows

| Frontend | Request path | CPU baseline | RockNPU path | Required quality evidence | Status |
|---|---|---|---|---|---|
| stock llama.cpp | `llama-bench`, `llama-server`, `llama-cli` | `-dev none`, no RockNPU plugin | `-dev ROCKNPU0`, W8 sidecar, resident decode | deterministic continuation plus int32/model oracle | three-block A/B completed; throughput gate failed |
| stock Ollama | `/api/generate` | stock runner, CPU-only, no RockNPU plugin | stock runner + `GGML_BACKEND_PATH` RockNPU plugin and `-dev ROCKNPU0` equivalence | deterministic response compared with CPU response and model oracle | stock 0.34.4 isolated install and three-block evidence completed; quality/throughput gates failed |

## Backend gates

### CPU

- backend row must be CPU only;
- device selection must be `none`;
- no RockNPU or other accelerator dispatch;
- exact model hash and four A76 threads required.

### RockNPU

- backend/device row must contain `ROCKNPU0`;
- trace must report nonzero native W8 matmul dispatches;
- exact sidecar v2 model hash required;
- no CPU fallback may satisfy a missing NPU operation.

## Required evidence per row

Each result directory contains:

1. exact commands, versions, binary/plugin/model hashes and source commit;
2. NPU/CPU/GPU frequency, governor, temperature and memory snapshots;
3. warmup policy and hot measurement window;
4. raw CPU/NPU stdout, stderr and structured JSON (llama.cpp per-run files are under `raw/llama-cpu-npu-o8/` and `raw/llama-cpu-npu-o16/`);
5. NPU dispatch summary and backend/device row;
6. deterministic response/continuation and independent quality result;
7. every raw A/B block and computed aggregate.

Canonical llama.cpp evidence directories:

- `artifacts/bench-700-cpu-npu-formal-o8/`
- `docs/benchmarks/2026-09-25/raw/ollama-cpu-npu-o8.json`
- `docs/benchmarks/2026-09-25/raw/profile-o8.log`
- `artifacts/bench-700-cpu-npu-formal-o16/`
- `docs/benchmarks/2026-09-25/raw/ollama-cpu-npu-o16.json`
- `docs/benchmarks/2026-09-25/raw/profile-o16.log`

## Boundary

Only RockNPU userspace middleware, benchmark tooling and documentation may be
changed. Frontend source and kernel-driver source are out of scope. Loading the
README-pinned external DVFS module is permitted only to establish the 700 MHz
benchmark environment; no module source is modified.
