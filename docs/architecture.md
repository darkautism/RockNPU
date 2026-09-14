# rocknpu architecture and first hardware vertical slice

Status: first RK3588 hardware vertical slice is working on 2026-09-14.

## 1. Goal and non-goals

The target stack is deliberately split into frontend adapters, a frontend-neutral RockNPU core, and the RK3588/Rocket backend:

```text
Candle / ONNX frontend / GGUF frontend / application runtime
                         |
                         v
                  frontend adapters
                         |
                         v
              RockNPU core compiler/runtime
        graph IR / partition / shape / placement
        layout / tiling / residency / scheduling
                  /                 \
          CPU fallback         RK3588 backend
                                    |
                                    v
                         Linux DRM accel Rocket UAPI
                                    |
                                    v
                                RK3588 NPU
```

The inference frontend is a caller of RockNPU. Model-format parsing belongs in adapters/frontends; RK3588 register commands, native layouts, residency, scheduling and Rocket submission belong behind the RockNPU core/backend boundary. The public middle-layer entry is now `Graph -> Executable -> Session`: adapters produce the shared `rocknpu-ir::Graph`, `Executable` performs device-independent contract/shape validation, and `Session` binds it to CPU or Rocket/NPU state. `Session::load(ONNX)` and the CLI remain reference-frontend conveniences over that path.

The crate split is one step behind the public boundary: the currently validated CNN executor implementation still resides in `rocknpu-onnx` and can also be constructed from a neutral `Graph`. Its model-format-independent execution code should be moved below the ONNX adapter in the next internal refactor; new frontend APIs must target the shared IR rather than add another format-specific executor.

The project does not depend on `librknnrt.so`, `librkllmrt.so`, RKNN Toolkit, or RKLLM Toolkit. `.rknn` and `.rkllm` compatibility is explicitly outside the first architecture milestone. The kernel driver is not forked: use upstream `drivers/accel/rocket/` unless a demonstrated hardware limitation makes that impossible.

## 2. Verified target host

Hardware/OS evidence is captured in `artifacts/environment.txt`.

Current test host:

- Orange Pi 5 class RK3588 host, aarch64.
- Kernel: `6.18.43-current-rockchip64` (`Linux orangepi5-8g`).
- `CONFIG_DRM_ACCEL=y`.
- `CONFIG_DRM_ACCEL_ROCKET=m`.
- In-tree module: `/lib/modules/6.18.43-current-rockchip64/kernel/drivers/accel/rocket/rocket.ko`.
- Device node: `/dev/accel/accel0`, group `render`; the current test user is in `render`.
- Three Rocket-bound NPU platform cores: `fdab0000.npu`, `fdac0000.npu`, `fdad0000.npu`.
- Direct sysfs IOMMU assignments: groups 8, 9, and 10 respectively.
- DRM driver identity checked at runtime: `rocket`, interface version currently reported as `0.0.0` by this kernel.

This means DT/probe/power/IOMMU are not current bring-up blockers. The device is usable from an unprivileged user in the `render` group.

## 3. Linux Rocket kernel/userspace boundary

Rocket is deliberately thin. The current public UAPI exposes four Rocket-specific ioctls:

- `DRM_IOCTL_ROCKET_CREATE_BO`
- `DRM_IOCTL_ROCKET_SUBMIT`
- `DRM_IOCTL_ROCKET_PREP_BO`
- `DRM_IOCTL_ROCKET_FINI_BO`

GEM close uses standard `DRM_IOCTL_GEM_CLOSE`.

The kernel owns:

- NPU runtime power management and reset.
- GEM/shmem BO allocation.
- per-DRM-file NPU IOVA allocation and IOMMU mappings.
- job dependency tracking via BO reservation objects/fences.
- per-core DRM scheduler queues.
- attaching a job's IOMMU domain to the selected NPU core.
- programming the PC block with the supplied regcmd base/count.
- completion IRQ handling, fence signalling, timeout/reset.

The kernel does **not** parse ONNX/TFLite/GGUF graphs, infer tensor layouts, tile operations, quantize models, or compile neural-network operations. Those belong in userspace.

## 4. Rocket UAPI details

### CREATE_BO

`drm_rocket_create_bo` contains:

```text
u32 size
u32 handle
u64 dma_address
u64 offset
```

The kernel creates shmem pages, obtains an SG table, allocates an IOVA from a per-file `drm_mm`, maps the pages into that file's IOMMU domain, returns a GEM handle, NPU-visible `dma_address`, and DRM mmap offset.

Important RK3588 fact verified on hardware: the first BO on a fresh fd legitimately receives NPU IOVA `0x0`. Zero is therefore not an invalid-address sentinel. Our executable regcmd is deliberately placed after a throwaway guard BO so PC `BASE_ADDRESS` is nonzero for the first milestone.

### mmap

Userspace calls `mmap(MAP_SHARED)` on the accel fd using the offset returned by `CREATE_BO`. The Rust runtime keeps the mapping and GEM handle paired in one `RocketBuffer`; `Drop` unmaps and closes the GEM handle.

### PREP_BO / FINI_BO

`PREP_BO` starts CPU ownership. In the current kernel it:

1. waits on the BO reservation object's write fence,
2. then synchronizes the SG table for CPU access.

`FINI_BO` synchronizes caches for device/NPU access.

The kernel's `timeout_ns` is an **absolute CLOCK_MONOTONIC deadline**, not a duration. `rocket-runtime::RocketBuffer::prep_relative()` therefore converts a relative duration to an absolute deadline. The C reference conformance test verified this against a real in-flight job: a past deadline returned `EBUSY`, and a relative 3-second wait converted by userspace completed successfully.

### SUBMIT

A submit contains an array of jobs. Each job contains:

- an array of `drm_rocket_task`,
- input BO handles,
- output BO handles.

A task is only:

```text
u32 regcmd
u32 regcmd_count
```

`regcmd` is the NPU IOVA of the register-command stream. This is a 32-bit field in the current UAPI/hardware programming path; buffers referenced directly by regcmd fields must therefore remain in the low 4 GiB IOVA window.

The current Rust vertical slice submits one job containing one task.

## 5. BO lifetime and IOMMU

BO ownership is tied to the DRM fd.

On `CREATE_BO`, the kernel obtains/creates the file's Rocket IOMMU domain, allocates an IOVA node, maps the BO SG table, and returns that IOVA. On BO destruction it unmaps the IOVA, removes the `drm_mm` node, releases the domain reference, and frees the GEM/shmem object.

At execution time the scheduler selects a Rocket core, attaches the job's domain to that core's IOMMU group, starts the task, and detaches on completion/reset. The three RK3588 cores on this host are currently in IOMMU groups 8, 9, and 10.

The Rust API therefore must not expose a device address with a lifetime independent of its BO/fd. `RocketBuffer<'device>` intentionally borrows the device.

## 6. Synchronization and scheduling

Input/output BO handle lists are semantically important, not bookkeeping. Rocket uses them to acquire implicit dependencies from GEM reservation objects. Output BOs receive the job completion fence.

`DRM_IOCTL_ROCKET_SUBMIT` is asynchronous: successful return means the job was queued, not that the NPU finished. The first Rust milestone waits by calling `PREP_BO` on the output with a real deadline.

Within one Rocket job, tasks execute sequentially on the same core so CBUF/SRAM residency can be reused. Separate jobs may be scheduled independently across the per-core DRM schedulers. This distinction must survive into the future compiler: `CommandBuffer`, `Task`, and `Job` are different concepts.

## 7. Register command path

Mesa packs one register command into a 64-bit word:

```text
bits 63:48  target/op selector
bits 47:16  32-bit register value
bits 15:0   register offset
```

The command program configures CNA, CORE, DPU and DPU-RDMA registers, followed by the PC trailer that enables execution.

The Rocket kernel does not interpret those neural-network registers. It points the NPU PC at the command buffer and gives the PC the command count. Therefore register knowledge, tensor layout, tiling and operation lowering are userspace responsibilities.

## 8. Mesa Rocket path

Pinned Mesa reference used for this round:

```text
f447fb30acceedd33c996301a866c63832bb9195
```

Relevant files:

- `src/gallium/drivers/rocket/rkt_ml.c`
- `rkt_task.c`
- `rkt_regcmd.c`
- `rkt_coefs.c`
- `rkt_device.c`
- `src/gallium/frontends/teflon/tfl_device.c`

Current path:

```text
Teflon / pipe_ml operation
        |
        v
rkt_ml_subgraph_create
        |
        +-- lower convolution/add
        +-- create Rocket tensor BOs
        +-- rkt_split_tasks (CBUF-aware splitting)
        +-- rkt_fill_regcmd
        v
DRM Rocket job/task arrays
        v
DRM_IOCTL_ROCKET_SUBMIT
```

Current Mesa `rkt_ml_operation_supported()` explicitly supports `PIPE_ML_OPERATION_TYPE_CONVOLUTION` and a restricted `ADD`. There is no direct Gallium ML MatMul/GEMM operation/lowering in this path today.

Mesa also demonstrates the important data transformations: tensor metadata is lowered into hardware-native feature/weight layouts before a regcmd is submitted. Teflon is useful as a graph/quantization/partitioning reference, but it should not become our model frontend dependency.

## 9. MatMul conclusion

The absence of MatMul in Mesa Rocket must **not** be interpreted as the RK3588 hardware lacking MatMul capability.

A separate public Rocket userspace implementation (reference commit `1a181cd69fd98fafa998cfeca963d35cb5f43c46`) implements MatMul as a 1x1 convolution through the RK3588 CNA -> CORE -> DPU datapath. We built and ran its deterministic FP16 test on this exact host:

```text
M=4 K=32 N=16
regcmd ops=126
ROCKET_SUBMIT=0
CPU/NPU result: exact match
```

Therefore the current gap is primarily:

1. no direct Gallium ML MatMul op path in Mesa/Teflon,
2. no Mesa MatMul lowering/compiler path,
3. **not** a known absence of RK3588 register-command knowledge.

The shortest path for this project is to own MatMul lowering directly in Rust rather than first teaching Gallium/Teflon MatMul.

For LLMs, another important observation is that large-M prefill behaves as GEMM and is the attractive NPU target, while `M=1` decode is GEMV-like and may be inefficient. Do not assume decode belongs on the NPU before benchmarking it.

## 10. First Pure Rust vertical slice

Workspace currently contains:

```text
crates/rocket-uapi       repr(C) UAPI structs + ioctl numbers/wrappers
crates/rocket-runtime    safe device/BO/submit/wait API; unsafe at ioctl/mmap edge
crates/rocknpu-regcmd    RK3588 fp16/fp32-output register-command encoding + planner
crates/rocknpu-matmul    reusable row-major FP16/FP32-output MatMul executor + persistent 1-3 worker pool
crates/rocket-smoke      deterministic real-hardware single-task / tiling / executor gates
```

`rocket-uapi` has ABI-size tests against Linux v6.18 structures.

`rocket-runtime` provides the first safe API shape:

```rust
let device = RocketDevice::open()?;
let buffer = device.alloc_buffer(size)?;
device.submit(&tasks, &input_handles, &output_handles)?;
output.prep_relative(timeout_ns)?;
```

`rocknpu-regcmd` now owns a descriptor-driven RK3588 FP16 MatMul encoder for one hardware task. It emits all 126 CNA/CORE/DPU/DPU-RDMA/PC register-command words from `M/K/N`, CBUF geometry, strides, and the three IOVAs; production code no longer patches a fixed command template. The original `M=4,K=32,N=16` capture is retained only under `cfg(test)` as a byte-for-byte regression golden.

The current single-task legal envelope is deliberately conservative and explicit:

- `M > 0`, `M % 4 == 0`, and `M + 1 <= 1023` (`CNA feature_grains`),
- `K > 0`, `K % 32 == 0`, `K <= 16384` for this no-K-tiling path because one kernel's FP16 weights must fit one 32 KiB CBUF bank,
- `N > 0`, `N % 16 == 0`, `N <= 8192` from the DPU channel-minus-one field,
- `ceil(M*K*2 / 32768) + ceil(N*K*2 / 32768) <= 12`; input and weight tiles must fit the twelve 32 KiB CBUF banks together,
- every IOVA directly encoded into a register command must fit in 32 bits.

Before hardware execution, six descriptor shapes were compared word-for-word against the pinned public C generator using identical sentinel IOVAs. Every stream was exactly 126 words and byte-identical:

```text
M4 K32 N16       MATCH
M12 K32 N16      MATCH
M16 K64 N32      MATCH
M32 K128 N48     MATCH
M64 K256 N64     MATCH
M256 K512 N128   MATCH
```

The same six shapes then executed through the pure-Rust Rocket path on the RK3588. Every output element matched the deterministic CPU FP32-accumulate -> FP16 reference bit-for-bit:

```text
M4-K32-N16       PASS
M12-K32-N16      PASS
M16-K64-N32      PASS
M32-K128-N48     PASS
M64-K256-N64     PASS
M256-K512-N128   PASS
cases=6 failures=0
PASS: generic pure-Rust RK3588 fp16 MatMul encoder + Rocket UAPI + 6 real NPU shapes + exact CPU compare
```

Artifacts written by the Rust gate are shape-tagged, for example:

- `artifacts/rust-regcmd-M256-K512-N128.bin`
- `artifacts/rust-input-M256-K512-N128.bin`
- `artifacts/rust-weights-M256-K512-N128.bin`
- `artifacts/rust-output-native-M256-K512-N128.bin`
- `artifacts/cpu-output-rowmajor-M256-K512-N128.bin`
- `artifacts/rust-run.log`
- `artifacts/environment.txt`

This satisfies the first hardware path:

```text
CPU deterministic tensors
        |
        v
Rust native-layout packing
        |
        v
Rust command encoding
        |
        v
Rocket CREATE_BO/mmap/PREP/FINI
        |
        v
Rocket SUBMIT + output fence wait
        |
        v
RK3588 NPU
        |
        v
Rust readback -> exact fp16 CPU comparison
```

### Tiled MatMul progress

The first correctness-first planner is now in Rust and shares the same explicit RK3588 CBUF model as the encoder:

- 12 CBUF banks x 32 KiB,
- `Mt <= 256`, `Nt <= 256` as the currently validated RK3588 planning ceiling,
- `Kt <= 16384`, 32-aligned,
- choose the largest `Kt` for which `banks(Mt,Kt) + banks(Nt,Kt) <= 12`,
- every returned tile is revalidated by the single-task encoder before it is accepted into the plan.

The stronger combined-CBUF check closed an important hole in the initial generic encoder: a task with an individually legal feature tile and one-kernel weight size can still be unsafe if the complete feature and weight tiles exceed the twelve-bank budget. For example `M256/K512/N256` is now rejected as `8 + 8 = 16` banks and must be tiled.

Two real tiled paths have now passed on RK3588.

First, an M-tiled case that cannot fit as one task:

```text
shape=M512 K512 N128
Mt=256 Kt=512 Nt=128
m_tiles=2 k_tiles=1 n_tiles=1 tasks=2
tile 0: M256 K512 N128 PASS
tile 1: M256 K512 N128 PASS
PASS: gather vs full CPU reference, mismatches=0
```

Second, a K-split baseline:

```text
shape=M64 K4096 N64
Mt=64 Kt=1536 Nt=64
K tiles: 1536 + 1536 + 1024
```

Each plain NPU partial matched its CPU FP16 partial bit-for-bit. Host FP32 accumulation of those NPU partials also matched a tiled CPU oracle bit-for-bit. Compared with one uninterrupted full-K CPU FP32 sum, the observed maximum difference was `1`, which is the expected consequence of narrowing each partial to FP16 before host accumulation.

The DPU EW/ERDMA accumulation path was then encoded in Rust. Accumulation changes exactly eight register values (`0x4070`, `0x5018`, `0x5034`, `0x5038`, `0x5040`, `0x5044`, `0x504c`, `0x506c`). Five accumulation shapes were byte-for-byte identical to the pinned public C generator.

Real NPU-side K accumulation uses separate output BOs because in-place WDMA/ERDMA is unsafe. The three K jobs are separately fenced and ping-pong `A -> B -> A`:

```text
tile=0 K1536 mode=plain  dst=A             PASS
tile=1 K1536 mode=ew-add dst=B src=A       PASS
tile=2 K1024 mode=ew-add dst=A src=B       PASS
PASS: staged fp16 CPU oracle mismatches=0
```

The final NPU EW result matched a CPU model of the same staged FP16 accumulation semantics bit-for-bit. Its maximum difference from a single uninterrupted full-K FP32 CPU sum was `2`; that is the measurable precision cost of staged FP16 K accumulation, not a correctness failure.

### Reusable single-core MatMul executor

The focused smoke paths have now been consolidated into `crates/rocknpu-matmul`. `Fp16MatmulExecutor` accepts row-major `A[M,K]` and `B[N,K]`, owns planning/native packing/submission/wait/gather, and returns row-major `C[M,N]`.

The executor handles arbitrary planner output rather than one special-case topology:

- no-K-split tiles execute directly,
- K-split output tiles with `M >= 12` use separately fenced NPU EW/ERDMA ping-pong accumulation,
- tiny-M (`M < 12`) K-split tiles deliberately use host FP32 accumulation of NPU-produced FP16 partials because the EW surface-stride floor below 12 rows is not accepted as a production hardware contract,
- M/N ragged aligned tails are packed and gathered independently per output tile.

Three dedicated RK3588 executor cases passed bit-for-bit against a CPU oracle that mirrors the executor precision semantics:

```text
n-only PASS M64 K512 N272
  Mt=64 Kt=512 Nt=256, jobs=2, N tail=16, mismatches=0

mnk-ragged PASS M300 K512 N272
  Mt=256 Kt=384 Nt=256, 2x2x2=8 jobs
  M tail=44, N tail=16, K=384+128
  npu_kacc_groups=4, mismatches=0

tiny-m-host-kacc PASS M4 K4096 N64
  Mt=4 Kt=2816 Nt=64, jobs=2
  host_kacc_groups=1, mismatches=0
```

A deterministic pseudo-random hardware differential sweep then exercised two independent seeds plus a mixed policy case. Small integer operands preserve exact FP16 comparison while changing data ordering:

```text
boundary M12 K96 N32                         PASS x2 seeds
n-tail M68 K224 N272                        PASS x2 seeds
mixed-kacc-tail M260 K1024 N272             PASS
  Mt=256 Kt=384 Nt=256, tiles=12, jobs=12
  npu_kacc_groups=2, host_kacc_groups=2
  mismatches=0
```

The mixed case is important: one matrix simultaneously contains large-M output groups accumulated by NPU EW and an `M=4` tail accumulated on the host, proving the policy boundary does not corrupt tile grouping or gather.

### FP32-output high-accuracy path

The high-accuracy path follows the public RK3588 reference's conservative design rather than inventing an FP32 EW mode. Each K tile is computed by the NPU with the full FP32 accumulator written to DRAM (native output cube `C2=4`), then K partials are summed on the host in FP64 and returned as FP32.

Relative to the validated FP16-output command stream, the FP32-output variant changes exactly five register values while keeping the same 126-word program length:

```text
0x4010  0x48000002 -> 0xa8000002   DPU output precision FP32
0x4050  0x00000126 -> 0x0000036e   FP32 output element sizing
0x4084  0x00010001 -> 0x00000001   disable FP32->FP16 narrowing
0x40c0  surface*2  -> surface*4    four bytes per output element
0x5044  0x00007818 -> 0x00007810   RDMA mode delta
```

`encode_fp16_matmul_fp32_output()` was compared word-for-word against the pinned public C generator for five shapes, including a ragged M tile; all were byte-identical:

```text
M16 K32 N16       MATCH
M64 K256 N64      MATCH
M128 K512 N128    MATCH
M256 K384 N128    MATCH
M44 K128 N16      MATCH
```

The reusable executor exposes this as opt-in `execute_f32`. Real RK3588 accuracy validation at `M64/K4096/N128` (four K tiles) with deterministic fractional FP16 inputs produced:

```text
max|reference| = 1335.893799
FP16-output max abs error = 0.893799
FP32-output max abs error = 0.000183
FP16 normalized error     = 6.6906e-4
FP32 normalized error     = 1.4e-7
accuracy improvement      = 4881.33x
nonfinite outputs         = 0
PASS
```

This is now the proven high-accuracy deep-K policy. The normal FP16 executor remains useful when readback/bandwidth matters; the two policies should be selected explicitly rather than silently changing numerical behavior.

### Persistent scratch, packing, timing, and multicore

`Fp16MatmulExecutor` now owns grow-only reusable scratch BOs for regcmd, input, weights, and two output ping-pong slots. A slot is reallocated only when a later tile requires more capacity. The hardware executor gate repeats `M4/K4096/N64` after warmup and verifies the scratch allocation counter does not change; therefore repeated same-shape execution no longer performs a `CREATE_BO + mmap + GEM_CLOSE` cycle per tile.

Native FP16 packing was also rewritten from per-element generic index calls into block-oriented copies matching the RK3588 layouts (`C2=8` feature blocks and `N16/K32` weight blocks). On little-endian hosts the contiguous 8/32-element blocks use safe `bytemuck::cast_slice` copies; `half::f16` is `Pod` when its `bytemuck` feature is enabled. No new unsafe code was added. Pure tests compare every packed element against `feature_data()` / `weight_fp16()` reference indices, and all randomized hardware gates remained exact.

The executor records plan/scratch/pack/encode/regcmd-write/submit/wait/gather/total phase times. A warmed release benchmark (`artifacts/executor-bench.txt`) asserts zero post-warm scratch allocations. The host does **not** expose a Rocket NPU devfreq node under `/sys/class/devfreq`, so these numbers are not clock-locked and must not be presented as frequency-controlled silicon maxima. The latest observed representative results after bulk packing were:

```text
shape                    best wall    end-to-end     pack      wait
M64 K4096 N128 FP16       1.345-2.903 ms   ~23-50 GF/s
M64 K4096 N512 FP16       5.100-5.172 ms   ~52 GF/s
M256 K1024 N256 FP16      1.739-1.743 ms   ~77 GF/s
M64 K4096 N128 FP32out    1.576-1.640 ms   ~41-43 GF/s
```

This changed the dominant cost for wide/prefill shapes: userspace packing was reduced enough that NPU fence wait is now the largest measured phase in the `N512` and `M256/N256` examples.

RK3588 multicore scheduling has an important userspace contract: multiple jobs on one DRM fd share one scheduling entity and serialize on one NPU core while that entity has queued work. True three-core fan-out requires independent DRM fds/entities and therefore independent BO/IOMMU domains. A Rust probe with one `RocketDevice + Fp16MatmulExecutor` per worker was preconditioned and sampled in interleaved worker-count order to avoid mistaking governor ramp for scaling. Median results for `M256/K384/N256`, 40 calls per worker, were:

```text
1 worker   31.96-32.86 aggregate GFLOP/s   1.00x
2 workers  62.93-63.68 aggregate GFLOP/s   1.91-1.99x
3 workers  93.58-94.11 aggregate GFLOP/s   2.86-2.93x
```

The first cold ordered probe produced a physically impossible apparent `5.14x` three-worker speedup and was explicitly rejected as a clock/governor artifact; only the preconditioned interleaved medians above are retained as scaling evidence.

`Fp16MatmulPool` turns that proof into a persistent runtime API. It owns 1-3 long-lived worker threads, each with its own Rocket fd, executor, and scratch BOs. Inputs are shared as `Arc<[f16]>`; N is partitioned on 16-channel boundaries without copying A/B, and row-major outputs are gathered after workers finish. Real hardware validation at `M256/K384/N768` split three ways into `N256` produced an exact CPU match, reused `[4,4,4]` scratch allocations unchanged on the second call, and observed warmed one-worker-to-three-worker wall-time changes from `5.334-5.653 ms` to `1.906-2.138 ms` (`2.64-2.80x`) across repeated runs.

### Resident/prepacked FP16 weights

`Fp16MatmulExecutor::prepack_weights()` converts a static row-major `B[N,K]` once into one resident low-4-GiB Rocket BO. The exact mode remains bound to one `M/K/N` planner geometry for best per-shape K tiling. A second `prepack_weights_compatible_m()` mode deliberately chooses Kt against the worst validated 256-row M tile, making N/K tile boundaries invariant across aligned M so one resident B can be reused across batch changes without repacking. Native weight tiles are keyed by `(n0,k0,n,k)`, so M-axis tiling does not duplicate resident weights: `M512/K512/N128` has two compute tiles but one resident weight tile, while `M300/K512/N272` has eight compute jobs but only four unique N/K resident tiles. Each tile starts at a 4096-byte-aligned BO offset and regcmds address `resident_dma + offset` directly.

`execute_prepacked()` supports the same correctness policies as the streaming executor: direct tiles, FP16 EW ping-pong K accumulation, tiny-M host-FP32 K accumulation, and simultaneous ragged M/N/K tails. Four real RK3588 topologies were run twice against the executor CPU oracle and remained bit-exact; the second call performed no new scratch BO allocation:

```text
single          M256 K512  N128   PASS
mnk-ragged      M300 K512  N272   PASS   8 jobs / 4 unique weight tiles
deep-k-ew       M64  K4096 N128   PASS   4 resident K tiles
tiny-m-host     M4   K4096 N64    PASS
```

A warmed release comparison showed the expected repeated-inference benefit. Representative stock-clock observations included `M64/K4096/N128` improving from about `3.02 ms` streaming to `2.03 ms` resident in one run, and `M64/K4096/N512` from about `5.08 ms` to `4.35 ms`. Under the controlled 600 MHz experiment below, resident prefill reached about `118 GFLOP/s` versus about `61 GFLOP/s` at the controlled 200 MHz baseline. One-time prepack cost is explicitly reported and is not included in repeated-call timing.

### Controlled Rocket DVFS experiment

The stock Armbian/mainline Rocket path on this host exposes no NPU devfreq device. Live DT gives all three NPU nodes `assigned-clock-rates = <200000000>` and no NPU `operating-points-v2`; stock Rocket enables/disables the clocks but does not own a devfreq policy. Therefore earlier stock measurements should be treated as the stock 200 MHz-class configuration rather than an automatically scaling NPU.

For characterization only, a public GPL-2.0 Rocket DVFS research tree was cloned under `reference/rk3588-npu-gpu` at commit `ed52a89afa8e68fedf636c8e891bd8fc47e82d26`. It is **reference/experimental kernel code, not project production code and not a new kernel fork we intend to maintain**. Its module was built against the exact running `6.18.43-current-rockchip64` headers and had matching vermagic. The driver registers dynamic 200-1000 MHz OPPs, treats 200 MHz as the safe power-domain transition rate, guards genpd transitions, and refuses rates above 700 MHz unless the NPU rail is at least 850 mV.

The temporary test deliberately did **not** alter the DTB, boot configuration, or NPU voltage. `vdd_npu_s0` remained at its original `800 mV`; `max_freq` was capped to each requested point. Exact resident MatMul gates passed at 200, 600, and 700 MHz. A focused single-job resident probe (`M256/K512/N128`, 41 repetitions) separated NPU fence-wait from CPU pack/gather noise:

```text
requested   median wait       wait-effective FP16 throughput
200 MHz     0.399286 ms       84.04 GFLOP/s
600 MHz     0.181122-0.186955 179.48-185.26 GFLOP/s
700 MHz     0.125123-0.130957 256.22-268.17 GFLOP/s
```

Thus the validated no-voltage 700 MHz point reduces this one-job NPU wait by about `3.05-3.19x` relative to 200 MHz. Whole-executor totals are less monotonic because CPU scheduling/governor noise affects packing and gather; use the phase-separated wait measurement for clock scaling. At 600 MHz, warmed resident examples reached about `106 GFLOP/s` for `M64/K4096/N128`, `118 GFLOP/s` for `M64/K4096/N512`, and `119 GFLOP/s` for `M256/K1024/N256`. A three-fd resident `M256/K384/N256` probe measured about `120.7 aggregate GFLOP/s` at 600 MHz and still scaled about `2.86x` from one to three workers, showing that this smaller per-core shape is increasingly limited by shared/host costs.

No >700 MHz test was performed because that crosses the research driver's voltage guard. After the experiment, the custom module was returned to 200 MHz, unloaded, and the packaged stock Rocket module was restored. The stock state was verified (`/sys/class/devfreq/fdab0000.npu` absent, `/dev/accel/accel0` present) and the resident exact hardware gate passed again. The NPU thermal zone stayed about `38.8 C` during these short tests. Evidence is summarized in `artifacts/dvfs-fp16-summary.txt`.

Evidence includes `artifacts/tiled-*`, `artifacts/k-tiled-*`, `artifacts/kacc-*`, `artifacts/executor-*`, `artifacts/executor-run.log`, `artifacts/executor-sweep.log`, `artifacts/fp32out-*`, `artifacts/executor-bench.txt`, `artifacts/multicore-probe.txt`, `artifacts/multicore-prepacked.txt`, `artifacts/pool-run.txt`, and `artifacts/dvfs-fp16-summary.txt`.

## 11. Provenance caveat for MatMul register knowledge

Mesa/Linux are permissively licensed references for the driver/UAPI path. The public `rocket-userspace` project used to validate FP16 MatMul is GPL-3.0-or-later.

The original fixed-shape golden was generated from that project's executable generator output, and the new descriptor encoder was cross-checked against its generator while the RK3588 register geometry was being understood. This is sufficient for research/hardware validation but does not by itself settle provenance for redistribution under an incompatible license. Before such distribution, re-derive or audit the production register encoder against permissively licensed Mesa/register documentation plus independent hardware experiments; keep GPL validation helpers clearly separated.

This does not affect the independently written Rust Rocket UAPI/runtime layer.

## 12. Hardware abstraction direction

Do not split into many crates before the next generic operation works. The eventual separation should be conceptual first:

```text
rocket-uapi       raw ABI only
rocket-runtime    Device / Buffer / Task / Job / synchronization
rocknpu-tensor    Shape / dtype / quantization / layout
rocknpu-regcmd    RK3588 register encoding
rocknpu-ops       operation contracts + CPU references
rocknpu-compiler  tiling / layout selection / lowering / partitioning
rocknpu-onnx      ONNX parser + shape/constants import
rocknpu-llm       GGUF/safetensors + transformer execution
```

Frontend code must never construct ioctls or raw register words.


### Project-owned tensor and MatMul operation contract

The first backend-independent operation boundary now exists above `rocknpu-matmul`. `rocknpu-tensor` owns validated rank-2 `Matrix<T>` values with explicit `MatrixShape` and `RowMajor` layout. `rocknpu-ops` defines the currently proven logical operation explicitly as `C[M,N] = A[M,K] x B[N,K]^T`; the transpose is not hidden as generic ONNX MatMul semantics.

The operation policy is explicit:

- `MatmulPrecision::Fp16Fast` returns an FP16 matrix and permits the validated fast NPU accumulation path,
- `MatmulPrecision::Fp32Accurate` returns FP32 and maps to the validated FP32-output NPU path or the independent CPU FP64-accumulate reference,
- `ExecutionTarget::{Cpu,NpuSingle,NpuPool,Auto}` makes backend choice visible,
- `Auto` uses a supplied single-NPU backend only when `M%4==0`, `K%32==0`, and `N%16==0`; otherwise it falls back to CPU without padding or changing the requested numerical policy,
- requesting an unavailable or unsupported backend is an error rather than a silent policy change.

The CPU implementation lives in `rocknpu-ops` rather than calling a CPU helper from the Rocket backend crate, so unsupported graph regions can eventually execute without depending on backend implementation details. The persistent pool now supports both `Fp16Fast` and `Fp32Accurate`. FP32 workers reuse the validated single-executor FP32-output regcmd path, gather FP32 N-slices without narrowing, and host-accumulate K partials in f64 inside each worker.

A stock-Rocket hardware gate exercised this boundary directly:

```text
ops single-fp16 PASS shape=64x256x64
ops single-fp32 PASS shape=64x256x64 max_abs_error=0
ops pool-fp16   PASS shape=64x256x96 workers=3
ops auto-cpu-fallback PASS shape=3x7x5
PASS: project-owned tensor/op MatMul contract hardware gate
```

This is the boundary an eventual graph importer should target. ONNX parsing still does not belong in the Rocket/runtime crates.

### First ONNX model execution

`rocknpu-onnx` is now a minimal project-owned ONNX frontend. It uses `onnx-protobuf 0.2.3` only for protobuf messages (MPL-2.0); execution remains in `rocknpu-ops` / the Rocket backend. Because that crate's generated code hard-checks protobuf 3.4.0 while its Cargo range would otherwise resolve newer incompatible protobuf releases, the workspace pins `protobuf = =3.4.0`.

The importer remains deliberately narrow and topologically ordered, but now accepts `MatMul`, the external model's exact `Gemm` subset (`alpha=beta=1`, `transA=0`, `transB=1`, initializer weight/bias), bias-broadcast `Add`, and `Relu`. Standard ONNX MatMul weights are stored `[K,N]`; the importer transposes them once at execution into the project's logical `B[N,K]^T` MatMul contract. `MatMul` is dispatched through `rocknpu-ops`; `Add` and `Relu` are CPU operations for now. Unsupported operators fail explicitly rather than silently falling back to a third-party runtime. Explicit `NpuSingle` dense execution now zero-pads M/K/N to the validated 4/32/16 hardware alignment and crops the result back to the ONNX-visible shape; `Auto` retains its existing no-padding CPU-fallback policy.

The first serialized model is `artifacts/tiny-mlp.onnx`, a typed opset-13 MLP:

```text
input [4,32]
  -> MatMul [32,32]   (RK3588 NPU)
  -> Add [32]         (CPU)
  -> Relu             (CPU)
  -> MatMul [32,16]   (RK3588 NPU)
  -> Add [16]         (CPU)
output [4,16]
```

The smoke gate writes the ONNX file, reloads the serialized bytes through `rocknpu-onnx`, runs a CPU-only graph and a stock-Rocket RK3588 hybrid graph, and requires bit-identical FP16 output. Observed result: 5 graph nodes, 2 NPU MatMuls, 2 CPU Adds, 1 CPU Relu, `mismatches=0`. Independent Python ONNX 1.17 validation also passes `onnx.checker.check_model()` with IR 9 / opset 13 and concrete input/output shapes `[4,32] -> [4,16]`. This is the first end-to-end standard model-format milestone, not a register-command test.

Evidence: `artifacts/tiny-mlp.onnx`, `artifacts/tiny-mlp-cpu-f16.bin`, `artifacts/tiny-mlp-npu-f16.bin`, and `artifacts/tiny-mlp-run.txt`.


### First external pretrained ONNX model

The first real-model gate uses the BSD-3-Clause Pico-CNN pretrained MNIST MLP from `data/mnist_mlp/mnist_mlp.onnx`, downloaded unchanged with SHA-256 `967612db6a1724e85101d5e11aaed3322d7d52ddd65f9af910fa8c71cf88c7cd`. ONNX checker reports IR 4 / opset 9. Its graph is `Gemm -> Relu -> Gemm -> Relu -> Gemm -> Relu -> Gemm` with dense widths `784 -> 800 -> 800 -> 400 -> 10` and declared batch 50.

All four Gemm nodes execute on stock-Rocket RK3588 NPU. Every layer requires transparent padding because M=50 is not a multiple of four; the first layer also pads K=784 to 800, and the last layer pads K=400 to 416 and N=10 to 16 before cropping. The three ReLU nodes remain CPU.

The independent oracle uses the canonical MNIST test set, normalized to `[0,1]` as Pico-CNN's example does. ONNX `ReferenceEvaluator` predicts 49/50 of the first 50 labels correctly. RockNPU NPU execution matches the ONNX top-1 prediction on all 50 samples and therefore also scores 49/50. Final-logit error versus the original FP32 ONNX reference is `max_abs=0.05685425`, `mean_abs=0.01186811`; this is expected from the current explicit FP16 lowering, not an assertion of FP32-exact execution. `scripts/verify_real_mnist.py` independently re-evaluates every Gemm/Relu with NumPy and compares every captured node tensor.


### Prepared ONNX model session with resident weights

The external pretrained MNIST model now has a first-class fixed-shape `PreparedNpuModel`. Preparation resolves the graph for a concrete input shape, performs ONNX-to-project weight orientation exactly once, pads each static dense weight to the validated NPU geometry, calls the `rocknpu-ops` prepared FP16 API, and leaves the native packed tiles resident in Rocket BOs. Repeated execution pads only the changing activation, calls `execute_prepacked`, crops the hardware-visible padded output, and applies the small CPU bias/ReLU operations.

For the `50x784` Pico-CNN MNIST batch, all four Gemm layers are resident:

```text
dense nodes                4
padded dense nodes         4
resident weight storage    3.218 MB
resident native tiles      21
one-time prepare wall      3.52-3.56 ms
weight packing inside prep 1.42-1.44 ms
streaming median           6.53-6.56 ms
prepared median            5.02-5.10 ms
observed wall improvement  1.28-1.31x
```

The correctness gate is stronger than a timing comparison. `prepared_mnist` first warms and records the old streaming output, prepares the session, then explicitly drops the parsed `TinyOnnxModel` and its original initializer storage before any prepared benchmark repetitions. The prepared result is bit-identical to the old streaming FP16 result, matches ONNX ReferenceEvaluator top-1 on all 50 samples, preserves the model's `49/50` accuracy on those samples, and performs no new executor scratch BO allocation after warmup (`bo_allocations=5`, `bo_grows=0`).

The prepared session also emits its own seven-node trace. `scripts/verify_real_mnist.py prepared-mnist-npu` independently evaluates the original FLOAT ONNX graph with NumPy semantics and observes the same per-node errors as the streaming path, ending at `max_abs=0.05685425`, `mean_abs=0.01186811`, and `50/50` top-1 agreement. Therefore resident packing changes storage/lifetime and repeated-inference work, not the accepted numerical contract.


### Dynamic-batch resident model session

`PreparedNpuModel` now has two explicit policies rather than silently trading speed for flexibility. `prepare_npu()` keeps the fixed-M exact planner path above. `prepare_npu_dynamic_batch()` uses `plan_fp16_matmul_compatible_m`: N/K resident tile geometry is chosen against a worst-case `Mt=256`, while each call is free to choose a different aligned/padded M. Only activations are repadded; static B is never repacked.

The real Pico-CNN MNIST model was prepared once, the parsed ONNX model was dropped, and the same four resident weight objects were then used for batch `1`, `4`, `16`, and `50` on stock Rocket. Every subset matched the independent ONNX top-1 prediction exactly. After warming the largest batch, smaller/later batches caused no new scratch allocation or growth. Observed preparation state was `3.213 MB`, 31 native resident tiles, about `4.60 ms` prepare wall and `1.50 ms` summed weight-pack work.

This flexibility has an explicit numerical/performance trade-off. The conservative K geometry creates more native tiles than the fixed session (31 vs 21 for this model) and changes FP16 K-split rounding. Batch-50 final-logit error versus the FP32 ONNX reference rose from fixed-session `max_abs=0.05685` to dynamic-session `0.08248`, while top-1 remained `50/50`. The independent seven-node NumPy oracle passes. This policy is therefore selectable, not a replacement for the fixed-M fast path.

### Worker-local resident pool model

Persistent multicore preparation now respects the real Rocket ownership model: every worker owns a separate fd/IOMMU domain and therefore its own resident BO copy. `Fp16MatmulPool` supports `Prepare`, `RunPrepared`, and `Release` commands keyed by a weight id. Preparation partitions N exactly as execution will, uploads each worker's slice once, and later commands carry only activation plus the resident id. `rocknpu-ops` exposes this as `PreparedPoolFp16Matmul`; `rocknpu-onnx` exposes a dynamic-batch `PreparedPoolNpuModel`.

The lower-level `M256/K384/N768` three-worker gate prepared three resident copies (`0.590 MB` total), reused them exactly for `M64/128/256`, kept worker scratch fixed at `[(4,0),(4,0),(4,0)]`, explicitly released them, and measured warmed streaming `1.779 ms` vs prepared `1.610 ms` (`1.11x`).

The full pretrained MNIST model creates 10 worker-local resident copies across four Gemm layers (large layers use three workers; the padded 16-wide classifier uses one), totaling `3.217 MB` / 37 native tiles. After the source model was dropped, batch `1/4/16/50` all matched ONNX top-1; batch-50 full-model median was `5.343 ms`; explicit release passed. The independent pool trace ends at `max_abs=0.06208801`, `mean_abs=0.01178154`, with `50/50` top-1 agreement.

For this small model, three-worker pool execution is not faster than the roughly 5 ms single-worker prepared session. This is positive policy evidence: worker count must be selected by measured graph/shape cost and must not default blindly to three.

### Persistent FP32Accurate pool

`Fp16MatmulPool` now has a typed FP32-output path in addition to the existing FP16 path. A `RunF32` worker command invokes the already validated `execute_f32()` on each worker's N-slice; responses carry `Fp32MatmulOutput`, and the pool gathers row-major `f32` slices without narrowing. `rocknpu-ops::PoolNpuBackend` maps `MatmulPrecision::Fp32Accurate` to this path instead of rejecting it.

Stock-Rocket hardware evidence:

```text
pool-fp32 integer PASS M64 K256 N96 workers=3 jobs=3 exact=true
pool-fp32 deep PASS M64 K4096 N192 workers=3 jobs=9
  max_abs=0.00001526
  normalized_rms=1.298e-7
  single normalized_rms=1.009e-7
  repeated output stable; scratch state unchanged after warmup
```

The public `rocknpu-ops` boundary independently passed `M64/K4096/N96` against its own CPU f64-accumulate reference with `max_abs=0.00001526`, `normalized_rms=1.508e-7`. Pool N-slicing can change planner geometry and therefore tiny FP32 rounding details, but accuracy remains in the same order as the single-NPU path.

### Benchmark-driven adaptive Auto policy

A persistent pool can now choose `1..=3` requested workers per streaming call. Worker-count probes intentionally use one already-created three-worker pool and interleave candidate order after warmup. Measurements showed that a static FLOP threshold is not reliable: for example, large MNIST dense layers and `M256/K384/N768` strongly favored three workers, while many small shapes favored direct single execution, and deep-K widths around N=128..320 changed winner across host/NPU states. Two workers also win in some boundary cases. Therefore the runtime does **not** encode today's benchmark numbers as magic shape thresholds.

`rocknpu-ops::AutoTunedNpuBackend` instead owns a direct `SingleNpuBackend`, a persistent pool, and a cache keyed by `(M,K,N,precision)`. On the first aligned occurrence it warms every distinct effective candidate, collects three interleaved wall-time samples for direct single and usable pool worker counts, caches the fastest candidate, and then executes with that backend. Subsequent identical shapes use the cached choice. Unaligned `Auto` still falls back to CPU and does not populate the NPU policy cache.

One stock-Rocket gate selected:

```text
M64  K256  N96  FP16 -> 1 worker; first/calibration 3.925 ms, cached 0.197 ms
M256 K384  N768 FP16 -> 3 workers; first/calibration 37.187 ms, cached 1.839 ms
M64  K4096 N192 FP32 -> 1 worker; first/calibration 49.471 ms, cached 2.711 ms
```

The selected worker counts are observations, not fixed expectations; the gate checks cache behavior and numerical correctness, not a machine-specific winner. This adaptive policy currently applies to streaming MatMul. Prepared-model resident slices are still fixed by their prepare-time worker partition, so prepared-model Auto worker-count selection remains separate future work.

### Resident lifecycle and low-4-GiB IOVA stress

A bounded stock-Rocket lifecycle gate now stresses the state-management assumptions directly:

```text
shape-change scratch: 128 alternating warmed executions, allocation/grow counters unchanged
resident churn:        128 prepare/drop cycles across K/N shapes
                        only 2 unique DMA bases observed
concurrent pressure:   96 resident copies, 100.7 MB total
                        DMA span 0x00096000..0x06095fff (< 4 GiB)
post-release:           next resident returned to 0x00096000 and executed successfully
pool lifecycle:         48 prepare/release cycles; use-after-release rejected
```

This is bounded stress, not a proof that the complete 32-bit IOVA aperture can never fragment or exhaust. It does establish that current GEM-close/drop paths release mappings in practice, resident allocation is not monotonically consuming the aperture in these cycles, grow-only executor scratch stabilizes after the largest warmed shapes, and pool resident IDs become unusable after explicit release.

### Second external pretrained model: official MNIST-8 CNN

The second real-model gate is the ONNX Model Zoo `mnist-8` CNN mirrored by `onnxmodelzoo/mnist-8`. The downloaded model is 26,454 bytes with SHA-256 `2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf`, matching the published Git-LFS object. ONNX checker reports IR 3 / opset 8. The graph has twelve nodes:

```text
Reshape(weight)
Conv -> Add -> Relu -> MaxPool
Conv -> Add -> Relu -> MaxPool
Reshape -> MatMul -> Add
```

This model deliberately expands graph/runtime scope without expanding the backend into a framework. `rocknpu-onnx::cnn` owns a small internal rank-N FP16 value for graph execution and CPU fallback. It implements the exact currently needed subsets of `Conv` (NCHW/OIHW, group 1, dilation 1, SAME_UPPER/explicit pads), NumPy-style `Add` broadcast, `Relu`, 2-D `MaxPool`, `Reshape`, and `MatMul`. `rocknpu-tensor` remains the project-native rank-2 backend boundary; the final `[1,256] x [256,10]` MatMul is converted into the existing Matrix contract and explicit NPU execution pads it to `M4/K256/N16` before cropping back to `[1,10]`.

The two 5x5 Conv nodes now execute through the project-owned `rocknpu-conv` / `rocknpu-regcmd` FP16 direct-convolution path on stock Rocket. The production encoder was derived from permissive Mesa register descriptions/task geometry plus independent hardware experiments; the separate GPL reference was used only as an isolated host-side black-box regcmd oracle during bring-up. Low logical channel counts are zero-padded to the hardware-validated direct geometry: Conv1 `IC1->OC8` executes as `IC32->OC16`, and Conv2 `IC8->OC16` executes as `IC32->OC16`. A standalone `IC32/H8/W8 -> OC16`, 5x5 pad2 gate was bit-exact over all 1024 outputs. With the real model weights, Conv1 was bit-exact to the FP16 CPU oracle over 6272 outputs; Conv2 differed in one FP16 result by only `7.63e-6` over 3136 outputs.

On stock Rocket, the canonical MNIST test-set first 100 images now produce:

```text
nodes=12
Conv RK3588 NPU  2
MaxPool CPU      2
Reshape CPU      2
Add CPU          3
Relu CPU         2
MatMul RK3588 NPU 1 (padded)
ONNX-reference top-1 agreement 100/100
ONNX ReferenceEvaluator accuracy 98/100
RockNPU hybrid accuracy          98/100
final max_abs                   0.02100563
final mean_abs                  0.00299615
```

The numerical oracle remains independent. `scripts/verify_mnist8.py` asks ONNX `ReferenceEvaluator` for all twelve intermediate node outputs for sample 0 and compares each tensor to the raw Rust/RK3588 FP16 trace. Per-node maximum absolute errors remain `0.00177` at the first NPU Conv, `0.00470` at the second NPU Conv, `0.00604` at the NPU MatMul, and `0.01326` after its bias Add. Across all 100 final outputs, top-1 agreement remains 100/100 and the final maximum error is `0.02100563`.

`CnnPreparedConvState` additionally keeps static Conv weights resident in low-4-GiB Rocket BOs. MNIST-8 owns two resident Conv tensors totaling 51,200 bytes; after the first inference the Conv executor uses only regcmd/input/output scratch (`weight_bytes=0`) and repeated calls do not grow scratch. Streaming and prepared Conv outputs are bit-identical on both real layers, and the full prepared graph independently passes the same 12-node / 100-sample ONNX ReferenceEvaluator gate. Layer-level warmed medians improved from `0.2080 -> 0.1945 ms` for Conv1 and `0.1207 -> 0.1070 ms` for Conv2. A 40-inference block benchmark measured the full graph at `0.9626 ms` streaming-Conv versus `0.8066 ms` prepared-Conv (`1.194x`), while single-inference medians are noisier at this sub-millisecond scale.

## 13. ONNX frontend plan

Do not implement protobuf or the whole ONNX spec from scratch. Evaluate a pure-Rust parser/graph implementation (including `tract` components if useful) for:

- graph topology,
- tensor shapes/dtypes,
- constants,
- attributes.

Import that into a project-owned IR. Execution must not become inseparable from a third-party CPU backend.

Partitioning should exist from the first model milestone:

```text
ONNX graph
   |
   v
internal IR
   |
   +-- supported connected region -> RK3588 compiler/backend
   +-- unsupported op/region      -> CPU executor
```

Initial operator coverage should follow operations that have validated register lowering, not a speculative ONNX checklist. CNN/MLP test models are preferred before large transformers.

## 14. CPU fallback plan

CPU fallback is part of the architecture, not an error path.

Every operation contract needs a CPU reference implementation usable for:

- deterministic unit vectors,
- randomized differential tests,
- unsupported graph regions,
- edge-shape verification,
- optimization regressions.

The partitioner owns transitions between CPU row-major tensors and NPU-native layouts. Layout conversion cost must become part of partition cost later; correctness comes first.

## 15. GGUF / LLM direction

Do not start the LLM frontend until generic MatMul/GEMM works across multiple shapes.

First LLM family should be one of Qwen or Llama. The minimum useful execution plan is allowed to be hybrid:

```text
MatMul / large projections -> NPU
RMSNorm                  -> CPU initially
RoPE                     -> CPU initially
attention glue           -> CPU initially
sampling/tokenizer       -> CPU
```

Move bottlenecks only after each operation has a hardware/CPU differential test.

The first LLM correctness milestone remains: load one GGUF, accept a prompt, generate the correct next token(s). Performance comes after correctness.

## 16. Unknown hardware capabilities / blockers

Confirmed:

- Rocket mainline UAPI works on this Armbian 6.18 kernel.
- three RK3588 NPU cores are bound and IOMMU-isolated.
- BO allocation/mmap/cache sync works from Rust.
- async submit + fence wait works.
- generic single-task FP16 MatMul executes correctly through a 1x1-convolution lowering across six tested shapes from `M4/K32/N16` through `M256/K512/N128`.
- the Rust planner correctly rejects combined CBUF overflow and emits legal M/N/K tiles.
- real two-task M tiling (`M512/K512/N128`) gathers exactly to the full CPU result.
- real three-task K tiling (`M64/K4096/N64`) has exact plain-partial CPU agreement.
- DPU EW/ERDMA fp16 K accumulation works from Rust with separately fenced ping-pong output buffers and exact staged-FP16 CPU agreement.
- `rocknpu-matmul` provides a reusable single-core executor with arbitrary planner-driven M/N/K tiling, aligned ragged tails, row-major packing/gather, explicit K-accumulation policy, grow-only scratch BO reuse, and per-phase timing.
- real N-axis tiling (`M64/K512/N272`) and simultaneous M/N/K ragged tiling (`M300/K512/N272`) are exact on hardware.
- one matrix can safely mix NPU EW K accumulation for large-M tiles with host-FP32 K accumulation for an `M=4` tail (`M260/K1024/N272`, 12 jobs, zero mismatches).
- deterministic pseudo-random differential hardware coverage passes across boundary and N-tail cases with multiple seeds.
- block-oriented/bytemuck FP16 packing is byte-equivalent to the reference layouts and remains exact on hardware.
- the FP32-output register stream is byte-identical to the public reference across five shapes; `execute_f32` returns NPU FP32 partials accumulated on the host in FP64.
- `M64/K4096/N128` FP32-output hardware validation achieved normalized error `1.4e-7`, about `4881x` better than the staged-FP16 path on the same data.
- true RK3588 multicore requires separate Rocket fds/entities; preconditioned Rust measurements scale `1.91x` at two workers and `2.86x` at three workers for `M256/K384/N256`.
- `Fp16MatmulPool` provides persistent 1-3 worker fan-out and is exact on hardware; `M256/K384/N768` observed `2.80x` warmed wall-time speedup with unchanged worker scratch allocations across repeated calls.
- resident/prepacked FP16 weights are exact across direct, ragged M/N/K, EW-KACC, and tiny-M host-KACC paths; M-axis duplicate weight tiles are stored only once.
- a temporary exact-vermagic DVFS research module established controlled 200/600/700 MHz behavior without changing the 800 mV rail; `M256/K512/N128` fence-wait fell from `0.399 ms` at 200 MHz to `0.125-0.131 ms` at 700 MHz.
- the packaged stock Rocket module was restored after DVFS characterization and the resident exact gate passed again.
- `rocknpu-tensor` and `rocknpu-ops` provide a backend-independent rank-2 tensor/MatMul contract with explicit precision/target policy and independent CPU fallback; single FP16, single FP32, three-worker FP16, and unaligned Auto->CPU paths are hardware/CPU gated.
- the official ONNX Model Zoo MNIST-8 CNN runs end-to-end on 100 canonical MNIST samples with both 5x5 Conv nodes plus the padded final MatMul on RK3588 NPU; MaxPool/Reshape/Add/Relu remain CPU, top-1 matches ONNX ReferenceEvaluator 100/100, and all 12 intermediate tensors have an independent trace comparison.
- project-owned direct FP16 Conv is hardware-proven for the MNIST geometry, including low-channel zero-padding; prepared Conv graph state holds two 25,600-byte resident weight cubes, removes per-run weight scratch/upload, remains trace-equivalent, and gives a measurable warmed block-level model speedup.

Not yet established by our own Rust compiler/runtime:

- prepared-model worker-count Auto policy: streaming MatMul now has adaptive benchmark-driven Auto selection, but worker-local resident slices are still fixed at model-prepare time.
- upstream/mainline DVFS integration and a production clock policy; the 200/600/700 MHz characterization used an external GPL research module only and does not change the project policy of avoiding a maintained kernel fork.
- INT8/INT4 correctness and exact packing/accumulator/requant rules.
- robust low-4-GiB IOVA exhaustion strategy.
- the remaining permissively documented register/packing rules required to extend the proven direct FP16 Conv path to stride/dilation/depthwise/tiling cases, and the separate Add/activation offload rules.
- whether any upstream Rocket UAPI version differences after the host's v0.0 interface need compatibility handling.

## 17. Next milestone

The resident-state foundation is now substantially proven: fixed/dynamic-M single-NPU residency, worker-local pool residency, FP32Accurate persistent pool execution, streaming adaptive Auto selection, and bounded prepare/release/IOVA lifecycle stress all pass on stock Rocket.

The second real pretrained model is now proven with its major compute on the RK3588 NPU: both MNIST-8 5x5 Conv nodes and the final MatMul execute through project-owned Rust backends, with 100/100 top-1 agreement and a 12-node ONNX ReferenceEvaluator trace. The exact same oracle also passes with prepared resident Conv weights.

The next useful milestone is **make the Conv path robust beyond this one model before adding speculative operators**. The current encoder is deliberately a narrow direct-FP16, stride-1, single-task path; low channels are safely zero-padded, but broader spatial/CBUF geometry still needs planner coverage and randomized hardware evidence.

Focused follow-ups remain:

1. add deterministic randomized CPU-vs-NPU Conv differentials across 1x1/3x3/5x5 kernels, rectangular H/W, padding, multiple IC32/OC16 groups, and boundary CBUF sizes,
2. validate stride-2 and then add spatial/channel tiling for Conv shapes that do not fit one CBUF task; do not guess depthwise/dilation rules before hardware gates,
3. profile the remaining full-graph time and keep MaxPool/Reshape/Add/Relu on CPU unless measurements show that offload beats layout/submit overhead,
4. preserve prepared resident Conv state for larger models where weight packing/upload is material; add shape-compatibility or worker-local copies only when a real graph needs them,
5. keep low-4-GiB pressure bounded and add explicit allocator failure/recovery behavior if larger real models approach the aperture,
6. do not expand into training/autograd/general framework APIs.

A third CNN model becomes useful after this Conv robustness sweep because it can then test generalization rather than introduce another one-off lowering. GGUF/transformer work remains later.

## 18. Host setup changes made during this round

The o8g host was missing C reference build tooling. Installed from Debian packages:

- `cmake`
- `ninja-build`
- `libdrm-dev`
- dependencies pulled by those packages (`libpciaccess-dev`, CMake runtime libraries, etc.)

`pkg-config` was already installed.

Debian `libdrm-dev 2.4.124` did not provide `drm/rocket_accel.h`, so the open Rocket UAPI header from the pinned Mesa checkout was copied to:

```text
/usr/local/include/drm/rocket_accel.h
```

- `python3-onnx=1.17.0-3+b1` (plus distro NumPy/protobuf/ONNX shared-library dependencies) for independent `onnx.checker` validation; mirror on o16g if reproducing this validation gate.

These system changes are test-host setup only and are not runtime dependencies of the Rust UAPI implementation.
For out-of-tree DVFS characterization only, the exact running-kernel headers were also installed:

- `linux-headers-current-rockchip64=26.8.1` for `6.18.43-current-rockchip64`
- header build dependencies installed by apt: `m4`, `flex`, `bison`, `libzstd-dev`, `libelf-dev`, `libssl-dev`
- apt also updated `libssl3t64`, `openssl`, and `openssl-provider-legacy` from the Debian repository during that transaction.

Record these on o16g if reproducing module builds there. The header/module experiment did not upgrade `linux-image-current-rockchip64`, modify DTB/boot configuration, or leave the external Rocket module loaded.


### Independent numerical oracle for the first ONNX model

The tiny MLP is not accepted merely because RockNPU CPU and NPU paths agree. `scripts/verify_tiny_model.py` independently loads `artifacts/tiny-mlp.onnx` with Python ONNX 1.17, validates it with `onnx.checker`, evaluates the declared FLOAT graph with ONNX `ReferenceEvaluator`, and separately re-evaluates every MatMul/Add/Relu node with NumPy. It then compares those independent per-node tensors against raw FP16 bits captured from the RK3588 NPU/hybrid run. Current result: every node and the final output have `max_abs_error=0`, and ONNX ReferenceEvaluator equals the NumPy trace exactly.

This exact result is specific to the deterministic fixture: its integer-valued inputs, weights, and intermediate magnitudes are exactly representable in FP16. It does **not** establish exact FP32 semantics for arbitrary ONNX FLOAT models, because the current importer intentionally converts FLOAT constants/activations into the FP16 execution path. Real pretrained models must therefore be checked against an independent FP32/FLOAT16 oracle with an explicit numerical tolerance/precision policy; shared RockNPU CPU/NPU agreement alone is never sufficient.

## Third pretrained model: RGB CIFAR-10 CNN

RockNPU now runs the external `edge-infer` pretrained CIFAR-10 CNN (reference commit `5eac90cc33ef06130571688aae012dd797d024ac`, model SHA-256 `fb5d665103dab8658e773267e5423cc4f338f395041d05e4cc37f15efd6fd341`). The bundled normalized CIFAR-10 test sample is class 8 (`ship`) and is used without inventing new preprocessing.

The opset-18 graph has 13 nodes: three `Conv`, three `Relu`, three `MaxPool`, one `Reshape`, and two `Gemm`. On stock RK3588 Rocket, all three Conv nodes and both Gemm nodes execute on the NPU; pooling, activation and reshape remain CPU fallback. The final RockNPU prediction is class 8, matching the original ONNX ReferenceEvaluator.

The importer additions are deliberately narrow: optional Conv bias, Conv kernel inference from OIHW weights when `kernel_shape` is omitted, and Gemm with `alpha=beta=1`, `transA=0`, `transB=1`. Standard exporter attributes used by this model (`MaxPool` dilation 1 / ceil_mode 0 / storage_order 0, `Reshape allowzero=1` with target `[-1,512]`) are explicitly validated.

The first 32x32 Conv exposed an artificial scaffold restriction: the Conv encoder used MatMul only as a fixed FP16 register-stream template and inherited MatMul's 10-bit `M+1` validation, incorrectly rejecting output spatial 1024. Conv's real `FEATURE_GRAINS` programming is based on input height (`IH+1`, 33 for 32x32), so the base template now uses a legal dummy MatMul `M=4` and all Conv geometry fields are explicitly patched as before. A 32x32 regression and the real CIFAR-10 hardware run validate this path.

Correctness is judged against the independent ONNX ReferenceEvaluator per node, not by requiring the CPU and NPU implementations to be bit-identical internally. For this model the 13-node traces both pass the external oracle; the largest node error versus FP32 ONNX is about `0.00638962`. CPU/NPU FP16 traces differ in only 16 of 32,906 elements, with maximum cross-backend absolute difference `0.00097656`; all final 10 logits are bit-identical and top-1 is identical. These differences are accepted as bounded FP16 accumulation rounding.
