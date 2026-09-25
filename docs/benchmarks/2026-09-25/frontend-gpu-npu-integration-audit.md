# Frontend NPU integration audit — 2026-09-25

This audit separates device discovery from actual NPU execution. A row is
not considered integrated merely because `ROCKNPU0` appears in a log.

## llama.cpp

- Device discovery: `llama-bench` JSON reports `backends=Vulkan` /
  `devices=Vulkan0` for GPU and `backends` containing `ROCKNPU` /
  `devices=ROCKNPU0` for NPU.
- Buffer routing: NPU run metadata records the GPU host/NPU frequency nodes;
  NPU stderr records `ROCKNPU_HOST` model/KV/compute buffers and the W8
  sidecar path. The formal raw result is under
  `raw/llama-gpu-npu-repro-o8/` and `...o16/`.
- Dispatch: each NPU run records nonzero `w8a8_m1` counts; the quality
  instrumented run records 2,518 NPU W8 lines on each board.
- Readback: the 32-token JSON result is read from the NPU process itself;
  the quality artifact contains raw API responses and a CPU oracle.

**Status: integrated and measured.**

## stock Ollama

- Device discovery: isolated `llama-server --list-devices` reports
  `Vulkan0: Mali-G610 MC4` when the external Vulkan backend is supplied.
  The Ollama parent also logs `offloaded 23/23 layers to GPU` in the GPU
  smoke.
- NPU buffer routing: NPU service logs report `ROCKNPU_HOST` model, KV and
  compute buffers, and the NPU trace/raw response artifacts are retained.
- Dispatch: NPU quality/trace raw logs contain `path=w8a8_m1` and model
  readback responses; no CPU fallback is counted as NPU work.
- GPU execution blocker: repeated Ollama GPU generation fails with
  `vk::CommandBuffer::end: ErrorOutOfDeviceMemory` under both Flash Attention
  settings and with `OLLAMA_KV_CACHE_TYPE=q8_0`. Raw diagnostics are in
  `raw/ollama-gpu-npu-repro-o16-v2/` and
  `raw/ollama-gpu-q8-probe-o16.log`, `raw/ollama-gpu-wrapper3-o16.log`, and `raw/llama-direct-vulkan-stable-o16.log`.
- Consequence: Ollama has a real discoverable GPU backend but no valid hot
  GPU generation denominator in this environment. CPU results are not a
  substitute; the Ollama GPU/NPU A/B remains blocked until the Vulkan/PanVK
  resource failure is resolved.

## Boundary

Only RockNPU adapter/userspace, benchmark scripts, and evidence were used.
No Ollama/llama.cpp frontend source or kernel driver was modified.
