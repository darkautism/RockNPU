# Frontend NPU/CPU benchmark matrix — 2026-09-25

## Scope

This is the reproducibility contract for the full-LLM frontends benchmarked
by the repository. A frontend is counted only when the unmodified frontend
can execute the same complete request/decode workload on the native CPU path
and on RockNPU, with a real backend/device check and an independent quality
oracle. Candle `RockNpuLinear` and the ONNX importer are operator/adapter
slices, not complete request-level LLM frontends, and are not counted as
full-LLM rows.

## Fixed environment

- Boards: o8 and o16, RK3588.
- Model: TinyLlama-1.1B-Chat-v1.0-Q4_K_M.
- Model SHA-256: `5c66751b61537f9e55177b1b67e06af88e0e2df88f86de4909f5bf87fb1ae583`.
- llama.cpp: native A76 build at commit `391fac164`, `GGML_NATIVE=ON`,
  `GGML_CPU_REPACK=OFF`; CPU and NPU use the same executable.
- NPU: README-pinned external DVFS module
  `ed52a89afa8e68fedf636c8e891bd8fc47e82d26`, requested and recorded at
  **700 MHz** (`cur_freq` and `target_freq`).
- CPU: policies 0/4/6 set to `performance`; four A76 workers; current
  frequencies are recorded per process.
- llama.cpp formal workload: 32 generated tokens, temperature 0, three
  four-process blocks, warm resident state. Ollama formal workload: fixed
  `Paris` API prompt, temperature 0, 8 generated tokens, context 2048, four
  threads, one warmup, three interleaved blocks.
- Acceptance: NPU hot throughput at least 5% above CPU and quality does not
  regress. Instrumented trace runs verify dispatch but are not used as fair
  timing numbers.

## Full-LLM frontend rows

| Frontend | Request path | CPU baseline | RockNPU path | Independent quality evidence | Reproducibility/status |
|---|---|---|---|---|---|
| stock llama.cpp | `llama-bench`, `llama-server` | `-dev none`, no plugin | `-dev ROCKNPU0`, W8 sidecar, resident decode | fresh CPU `llama-server` oracle; 3 quality blocks match | `raw/llama-formal-repro-o8/`, `raw/llama-formal-repro-o16/`, `raw/llama-quality-repro-o8/`, `raw/llama-quality-repro-o16/`; A/B complete, quality PASS, throughput FAIL |
| stock Ollama | `/api/generate` | isolated Ollama CPU service, no plugin | isolated Ollama + `GGML_BACKEND_PATH` + `LLAMA_ARG_DEVICE=ROCKNPU0` | fresh CPU Ollama oracle; formal response comparison | `raw/ollama-formal-repro-o8/`, `raw/ollama-formal-repro-o16/`, `raw/ollama-trace-repro-o8/`, `raw/ollama-trace-repro-o16/`; integration/dispatch complete, quality FAIL, throughput FAIL |

The Ollama fair run intentionally has no per-op trace in the timed path. A
same-configuration `--trace` three-block run is retained separately and
contains 2,467 `path=w8a8_m1` lines on each board, preventing device
 discovery or CPU fallback from being counted as the NPU result.

## Backend gates

### CPU

- Backend/device row is CPU only.
- Device selection is `none`.
- No RockNPU/accelerator dispatch.
- Exact model hash, four workers, and frequency/governor snapshots are saved.

### RockNPU

- Backend/device row contains `ROCKNPU0`.
- Native W8 dispatch is nonzero; llama.cpp formal metadata records 30,492
  dispatches per board and the Ollama trace run records 2,467 W8 M1 lines.
- Sidecar v2 source hash matches the model.
- Host-buffer routing is recorded; CPU_REPACK is not accepted as a substitute.

## Required evidence

Each board's `raw/evidence-index-<board>.json` is the authoritative manifest.
It indexes all raw files and records fixed conditions, commands, environments,
frequency snapshots, quality results, dispatch counts, profiles, candidate
rows, and c8/c16 diagnostics. The raw evidence includes:

1. exact argv and complete relevant environment;
2. model, frontend/binary, plugin, sidecar and source hashes;
3. NPU `cur_freq`/`target_freq`, CPU governor/current frequency, and thermal
   snapshots before/after each run;
4. warmup policy and hot timing window;
5. raw CPU/NPU stdout, stderr, JSON, HTTP request/response and `/api/ps` data;
6. independent CPU oracle response and hash;
7. native trace/dispatch summary and raw log hash;
8. all A/B blocks, candidate rows, concurrency rows, and computed aggregates.

The old ignored `artifacts/` paths are not required for review; the complete
review package is the versioned `raw/` tree.

## Blocker/candidate evidence

`raw/candidate-matrix-o8/` and `raw/candidate-matrix-o16/` retain direct,
scratch, scheduler-routing, K64, M-tile and grouped-QO candidate stdout/stderr,
JSON, commands, frequency snapshots and hashes. `raw/ollama-concurrency-o8/`
and `raw/ollama-concurrency-o16/` retain c8/c16 raw requests and service logs.
`raw/llama-profile-repro-o8/` and `raw/llama-profile-repro-o16/` retain the
700 MHz M1 execute/wait profile with exact provenance.

No row passes the joint promotion gate. The 700 MHz execute/wait profile and
whole-model A/B are the recorded blocker; c8/c16 are diagnostics, not a
replacement for fair CPU/NPU A/B.

## Boundary

Only RockNPU userspace middleware/adapter glue, benchmark tooling, raw
evidence, and documentation may change. Frontend source and kernel-driver
source are out of scope. The external DVFS module is an environment dependency
and is not modified.
