# Ollama GPU runtime diagnostics

The isolated Ollama 0.34.4 parent can discover `Vulkan0` when the external
`libggml-vulkan.so` is supplied. The following conditions were tested with the
same TinyLlama blob:

- default context/batch and Flash Attention off;
- context 512 and q8 KV cache;
- context 512, q8 KV cache, Flash Attention on;
- a temporary model parameter layer (`num_ctx=512`, `num_batch=256`);
- a reversible `llama-server` argv wrapper reducing batch/ubatch and disabling
  context shift/continuous batching.

Every completed GPU generation path ended in
`vk::CommandBuffer::end: ErrorOutOfDeviceMemory` or an HTTP 500 with the same
server error. A direct `llama-server` using the same Vulkan backend and
`ctx512/batch128/ubatch128/no-context-shift/no-cont-batching/q8 KV` succeeds;
therefore the blocker is the Ollama runner's generated execution/allocation
path, not discovery, model loading, or absence of a Vulkan device.

Raw diagnostics:

- `raw/ollama-gpu-npu-repro-o16-v2/`
- `raw/ollama-gpu-q8-probe-o16.log`
- `raw/ollama-gpu-wrapper3-o16.log`
- `raw/llama-direct-vulkan-stable-o16.log`

No CPU result is used as a GPU substitute. Ollama GPU/NPU promotion remains
blocked until a valid stock Ollama GPU generation path is available.
