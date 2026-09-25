#!/bin/sh
set -eu
# npu service
env GGML_BACKEND_PATH=/build/rocknpu/target/ggml-rocknpu-700/libggml-rocknpu.so LD_LIBRARY_PATH=/opt/ollama-0.34.4/lib/ollama LLAMA_ARG_DEVICE=ROCKNPU0 OLLAMA_HOST=127.0.0.1:11440 OLLAMA_KEEP_ALIVE=30m OLLAMA_MODELS=/build/ollama-models OLLAMA_NUM_GPU=999 ROCKNPU_DECODE=1 ROCKNPU_DISPATCH_SUMMARY=1 ROCKNPU_EXPERIMENT_DIRECT_SCRATCH=1 ROCKNPU_GGML_TRACE=1 ROCKNPU_PREFILL_CACHE=1 ROCKNPU_W8_DIRECT_SUBMIT=1 ROCKNPU_W8_SIDECAR_DIR=/build/w8a8-models/native-w8-gguf /opt/ollama-0.34.4/bin/ollama serve >ollama-cpu-npu-trace-o8-npu.stdout.log 2>ollama-cpu-npu-trace-o8-npu.stderr.log &
npu_pid=$!
# cpu service
env LD_LIBRARY_PATH=/opt/ollama-0.34.4/lib/ollama LLAMA_ARG_DEVICE=none OLLAMA_HOST=127.0.0.1:11441 OLLAMA_KEEP_ALIVE=30m OLLAMA_MODELS=/build/ollama-models OLLAMA_NUM_GPU=0 /opt/ollama-0.34.4/bin/ollama serve >ollama-cpu-npu-trace-o8-cpu.stdout.log 2>ollama-cpu-npu-trace-o8-cpu.stderr.log &
cpu_pid=$!
# oracle service
env LD_LIBRARY_PATH=/opt/ollama-0.34.4/lib/ollama LLAMA_ARG_DEVICE=none OLLAMA_HOST=127.0.0.1:11442 OLLAMA_KEEP_ALIVE=30m OLLAMA_MODELS=/build/ollama-models OLLAMA_NUM_GPU=0 /opt/ollama-0.34.4/bin/ollama serve >ollama-cpu-npu-trace-o8-oracle.stdout.log 2>ollama-cpu-npu-trace-o8-oracle.stderr.log &
oracle_pid=$!
# Run requests using the Python harness; this file records the exact service launch commands.
wait
