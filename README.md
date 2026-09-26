# RockNPU

**English** | [繁體中文](README-zh-TW.md)

**Run large language models through the RK3588 NPU from stock llama.cpp and Ollama.**  
Open source, no RKNN/RKLLM proprietary runtime required, no llama.cpp/Ollama source patch required, and no model conversion: use your existing GGUF models.

---

## What it does

Measured with TinyLlama-1.1B Q4_K_M on Orange Pi 5 (RK3588, LPDDR4X):

| Workload | CPU only (llama.cpp) | **RockNPU** | Reference: published RKLLM result* |
|---|---:|---:|---:|
| Prefill, 128 tokens | 73 tok/s | **≈ 570 tok/s** | ≈ 525 tok/s (TTFT 244 ms) |
| Prefill, 512 tokens | 69 tok/s | **≈ 420 tok/s** | — |
| Decode / generation | 32–33 tok/s | **≈ 31–33 tok/s** | 24.4 tok/s |
| Output quality | reference | CPU-exact decode; W8A8 prefill | W8A8 |

\* RKLLM cannot run in the current validation environment without replacing the kernel, so its numbers are taken from the [official airockchip/rknn-llm benchmark](https://github.com/airockchip/rknn-llm/blob/main/benchmark.md) (W8A8, maximum CPU/NPU clocks).  
RockNPU numbers use a 700 MHz NPU, llama-bench, and four Cortex-A76 threads. See [架構.md](架構.md#效能與量測方法) for methodology and raw-data notes.

Other models (tok/s; “CPU only” means stock llama.cpp defaults):

| Model (Q4_K_M) | Prefill 128 CPU | **RockNPU** | RKLLM* | Decode CPU | **RockNPU** | RKLLM* |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B-Instruct | 71 | **≈ 580** | — | 26 | 23 | — |
| Qwen2.5-1.5B-Instruct | 55 | **≈ 350** | ≈ 340 | 22.5 | 19–20 | 16.7 |
| Qwen2.5-0.5B-Instruct | 71 | 71 (no acceleration; see FAQ) | — | — | — | 41.6 (Qwen2 0.5B) |

For long documents and long conversation history, prompt processing is typically **6–8× faster**. Single-stream decode defaults to the CPU because RK3588 memory bandwidth makes the CPU Q4_K path faster than resident W8 NPU decode. Compared with stock llama.cpp, decode can still be 5–15% slower because RockNPU must disable CPU weight repacking; see the FAQ below.

---

## Requirements

- An RK3588 / RK3588S board, such as Orange Pi 5 or Rock 5.
- A Linux kernel exposing the mainline-style `rocket` accelerator interface, typically Linux 6.18 or newer.
  Check with: `ls /dev/accel/accel0`
- About 2 GB of free disk space for building.

## Install in 3 steps

```sh
# 1. Clone RockNPU
git clone https://github.com/darkautism/RockNPU.git
cd RockNPU

# 2. Install build dependencies, Rust, and RockNPU.
#    Add --with-llama to also build llama-server when llama.cpp is not already installed.
./scripts/install.sh --with-llama

# 3. Load the generated environment.
#    Add this line to ~/.bashrc if desired.
. ~/.local/share/rocknpu/rocknpu.env
```

If `/dev/accel/accel0` exists but is not accessible, run `sudo usermod -aG render $USER` and log in again.

### Best-performance settings

**Always load the generated environment before starting llama.cpp or Ollama:**

```sh
. ~/.local/share/rocknpu/rocknpu.env
```

`install.sh` writes the validated settings into this file: the RockNPU backend/device, `LLAMA_ARG_REPACK=false`, `GOMP_SPINCOUNT=20000`, four CPU threads, and flash attention. The validated NPU execution routes are already enabled by the backend itself; **you do not need to copy experimental `ROCKNPU_*` flags from benchmark notes.** If you launch `llama-server` without loading this environment, you can easily end up benchmarking stock CPU instead of RockNPU.

For maximum performance, also pin the process to the four Cortex-A76 cores (`taskset -c 4-7`) and use a supported K-quant GGUF such as `Q4_K_M`.

### Optional but strongly recommended: run the NPU at 700 MHz

```sh
sudo ./scripts/rocknpu-tune.sh dvfs      # build and load the external devfreq-capable driver; kernel headers required
sudo ./scripts/rocknpu-tune.sh install   # apply tuning at boot; restore with: sudo ./scripts/rocknpu-tune.sh restore
```

The mainline `rocket` driver does not currently expose frequency scaling on these systems, so the NPU may remain at its 200 MHz boot clock and prompt processing can be roughly 40% slower.

`dvfs` directly downloads, builds, and loads the community-maintained [rk3588-npu-gpu](https://github.com/sky-rk3588/rk3588-npu-gpu) module. **RockNPU does not modify or maintain that kernel module.** It does not change the NPU voltage or replace the boot kernel; `restore` followed by a reboot returns to the distribution driver.

For the RockNPU LLM path, performance is already near saturation around 700 MHz. Testing at 1 GHz showed no reproducible throughput gain, so 700 MHz is the recommended performance target.

## Usage

### llama.cpp / llama-server

```sh
. ~/.local/share/rocknpu/rocknpu.env
taskset -c 4-7 llama-server -m your-model.gguf -t 4
```

Open `http://BOARD_IP:8080`.

Confirm the backend is visible:

```sh
llama-server --list-devices
```

Expected output includes:

```text
ROCKNPU0: RockNPU RK3588
```

Pinning llama.cpp to the four large cores improves prompt processing by roughly another 10% on the validated systems. The environment already requests four threads; `taskset` additionally prevents those threads from migrating onto the A55 cores.

The first request after model loading is slower because RockNPU prepares 8-bit NPU weights. For TinyLlama this takes roughly one second; steady-state requests are faster.

### Ollama

```sh
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl stop ollama
. ~/.local/share/rocknpu/rocknpu.env
ollama serve

# In another terminal:
ollama run tinyllama:1.1b-chat-v1-q4_K_M
```

Use a K-quant model such as `Q4_K_M`. Ollama's default tags are often `Q4_0`, which RockNPU does not accelerate.

Ollama 0.34.x uses the same llama.cpp line validated by RockNPU (`b10969`); 0.34.4 has been tested without source patches. For another Ollama/llama.cpp revision, rebuild with:

```sh
LLAMA_REF=<matching llama.cpp tag> ./scripts/install.sh
```

The generated environment config also applies the two settings needed for good Ollama performance:

- `LLAMA_ARG_THREADS=4`, avoiding the A55 cores that otherwise stall each step.
- `OLLAMA_FLASH_ATTENTION=1`, keeping flash attention enabled.

Measured with TinyLlama Q4_K_M and a 329-token prompt, Ollama reaches roughly **300 tok/s prefill** and **30 tok/s decode** through RockNPU. Stock CPU-only Ollama on the same setup is roughly 100 / 23 tok/s.

> Ollama itself occupies about 2 GB. On boards booting from eMMC or SD, keep Ollama and model storage on NVMe/SSD where possible.

## FAQ

**`--list-devices` does not show `ROCKNPU0`. What should I check?**  
Make sure `. ~/.local/share/rocknpu/rocknpu.env` has been loaded and `/dev/accel/accel0` exists with read/write permission.

**The NPU appears, but performance is similar to CPU. Why?**  
Make sure `LLAMA_ARG_REPACK=false` is active. The install script already sets it. llama.cpp's CPU repacking transforms weights into a private CPU-only layout that RockNPU cannot consume. When invoking llama.cpp manually, `--no-repack` is equivalent.

**Is NPU prefill bit-identical to CPU?**  
No. Prompt projections use W8A8 integer execution, the same broad quantization class used by RKLLM. On the validated models, the next-token top-1 result matches CPU roughly 91–94% of the time. Decode defaults to CPU and is therefore identical to the corresponding no-repack CPU path.

For higher prompt accuracy:

```sh
export ROCKNPU_PREFILL_HILO=down
```

This roughly halves quantization error for about a 23% prefill throughput cost.

Or:

```sh
export ROCKNPU_PREFILL_HILO=1
```

This reduces the measured KLD by roughly 4–5× while cutting prefill throughput by about half; it is still substantially faster than CPU on supported models.

**Why is generation a little slower than stock llama.cpp?**  
Typically by about 5–15%, depending on the model. Stock llama.cpp can repack weights into a CPU-specific layout and decode faster. RockNPU needs `LLAMA_ARG_REPACK=false` so the original GGUF layout remains available to the NPU.

If your workload is almost entirely short prompts followed by very long outputs, pure CPU with repacking may be faster overall.

**Can the NPU perform generation too?**  
Yes:

```sh
export ROCKNPU_DECODE=npu
```

This is roughly 20 tok/s for TinyLlama while leaving the CPU mostly idle.

Or:

```sh
export ROCKNPU_DECODE=hybrid
```

This runs CPU and NPU concurrently and reaches roughly 26 tok/s. The default `cpu` mode is fastest for single-stream generation.

**Which models are supported?**  
RockNPU accelerates Q4_K / Q6_K GGUF projection weights, including common `Q4_K_M` models, when the relevant hidden dimensions match the supported NPU geometry. Llama 3.x, TinyLlama, and Qwen2.5 1.5B and larger are representative supported cases.

Unsupported operations and formats remain on the CPU, so models that llama.cpp can run still work; they simply receive less or no NPU acceleration.

Qwen2.5-0.5B has hidden width 896 and its GGUF projection weights do not use the supported K-quant path, so it currently stays on CPU.

The NPU keeps an additional 8-bit copy of accelerated weights. Budget roughly one extra byte per parameter, around 1 GB for a 1B-parameter model.

**Does NPU frequency matter? Should I overclock to 1 GHz?**  
200 → 700 MHz matters: measured TinyLlama prefill rose from roughly 326 to 556 tok/s.  
700 MHz → 1 GHz, which also requires raising the NPU rail to about 850 mV, did not produce a meaningful reproducible improvement. The bottleneck has already moved to memory/host-side work, so 1 GHz is not recommended for this workload.

See [架構.md](架構.md#npu-頻率) for details.

## More documentation

- Architecture, settings, and performance methodology: [架構.md](架構.md)
- Research status and validated/closed experiments: [docs/research-status.md](docs/research-status.md)
- Trial-and-error ledger: [trialanderror.md](trialanderror.md)

## Credits and license

[oRKLLM/ork-driver](https://github.com/oRKLLM/ork-driver) is an important reverse-engineering reference for RK35xx hardware behavior and register-command research.

Original RockNPU code is MIT licensed. Regcmd reference material directly derived from ork-driver remains under its ISC license in `crates/rocknpu-regcmd/src/int8/ork_isc.rs`; see `docs/licenses/ork-driver-ISC.txt`.
