# Goal blocker record

Status: **blocked pending a valid Ollama GPU generation path**.

The llama.cpp frontend has a reproducible GPU/NPU candidate:
o8 direct+scratch `1.546x` and o16 direct+scratch `1.735x`, with quality
oracle PASS. This does not complete the multi-frontend goal.

The Ollama frontend repeatedly fails GPU generation with
`vk::CommandBuffer::end: ErrorOutOfDeviceMemory` under:

- context 2048 and context 512;
- Flash Attention on and off;
- q8 KV cache;
- temporary `num_ctx`/`num_batch` Modelfile parameters;
- reversible llama-server argv wrapper reducing batch/ubatch and disabling
  context shift/continuous batching.

The isolated Ollama parent discovers `Vulkan0`, and a direct llama-server with
the same Vulkan backend and conservative memory settings succeeds. Therefore
the blocker is the Ollama runner's generated GPU allocation/execution path,
not model absence or device discovery. CPU results are never substituted for
the missing GPU denominator.

Raw evidence is retained in `raw/ollama-gpu-npu-repro-o16-v2/`,
`raw/ollama-gpu-q8-probe-o16.log`,
`raw/ollama-gpu-wrapper3-o16.log`, and
`raw/llama-direct-vulkan-stable-o16.log`.

Unblock only when a stock Ollama GPU service can complete the same hot A/B
without CPU fallback or Vulkan OOM, with response quality and raw evidence.

## Final source-level audit

Ollama v0.34.4 source confirms `OLLAMA_LLM_LIBRARY` selects a named
variant directory, while integrated GPUs require `OLLAMA_IGPU_ENABLE=1`.
A temporary `vulkan` variant symlink and `OLLAMA_LLM_LIBRARY=vulkan` made
Ollama report `library=Vulkan`, `name=Vulkan0`, `type=iGPU`, and load the
model on Vulkan, but its runner still returned HTTP 500 with
`ErrorOutOfDeviceMemory`. Replacing only the runner with the same-commit
native llama-server did not change the result. The installed runtime was
restored after the probes.
