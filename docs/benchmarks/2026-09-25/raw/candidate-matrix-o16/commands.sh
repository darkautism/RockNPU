#!/bin/sh
set -eu
# cpu-reference
taskset -c 4-7 /build/llama.cpp-reference/build-native-norepack-rocknpu/bin/llama-bench -m /build/model-cache/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf -p 0 -n 8 -r 1 -t 4 -fa on -dev none -nopo 1 -o json
# baseline
taskset -c 4-7 /build/llama.cpp-reference/build-native-norepack-rocknpu/bin/llama-bench -m /build/model-cache/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf -p 0 -n 8 -r 1 -t 4 -fa on -dev ROCKNPU0 -nopo 0 -o json
# direct-submit
taskset -c 4-7 /build/llama.cpp-reference/build-native-norepack-rocknpu/bin/llama-bench -m /build/model-cache/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf -p 0 -n 8 -r 1 -t 4 -fa on -dev ROCKNPU0 -nopo 0 -o json
# direct-scratch
taskset -c 4-7 /build/llama.cpp-reference/build-native-norepack-rocknpu/bin/llama-bench -m /build/model-cache/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf -p 0 -n 8 -r 1 -t 4 -fa on -dev ROCKNPU0 -nopo 0 -o json
# scheduler-cpu-qo
taskset -c 4-7 /build/llama.cpp-reference/build-native-norepack-rocknpu/bin/llama-bench -m /build/model-cache/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf -p 0 -n 8 -r 1 -t 4 -fa on -dev ROCKNPU0 -nopo 0 -o json
# scheduler-ffn-only
taskset -c 4-7 /build/llama.cpp-reference/build-native-norepack-rocknpu/bin/llama-bench -m /build/model-cache/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf -p 0 -n 8 -r 1 -t 4 -fa on -dev ROCKNPU0 -nopo 0 -o json
# k64-candidate
taskset -c 4-7 /build/llama.cpp-reference/build-native-norepack-rocknpu/bin/llama-bench -m /build/model-cache/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf -p 0 -n 8 -r 1 -t 4 -fa on -dev ROCKNPU0 -nopo 0 -o json
# m8-mtile
taskset -c 4-7 /build/llama.cpp-reference/build-native-norepack-rocknpu/bin/llama-bench -m /build/model-cache/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf -p 0 -n 8 -r 1 -t 4 -fa on -dev ROCKNPU0 -nopo 0 -o json
# m16-qo-group
taskset -c 4-7 /build/llama.cpp-reference/build-native-norepack-rocknpu/bin/llama-bench -m /build/model-cache/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf -p 0 -n 8 -r 1 -t 4 -fa on -dev ROCKNPU0 -nopo 0 -o json
