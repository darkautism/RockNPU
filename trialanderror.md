# trialanderror.md

Purpose: stop future agents from re-running already-decided RockNPU/RK3588 experiments. Prefer new HW evidence over argument.

## 0. Symbols

- `HW`: real RK3588 + Linux Rocket (`/dev/accel/accel0`), not simulator/compile-only.
- `GGML`: stock unmodified llama.cpp dynamic backend path.
- `M/K/N`: RockNPU matmul convention: activation `[M,K]`, weight `[N,K]`, output `[M,N]`. GGML printed axes may differ.
- `W8A8`: int8 weight + int8 activation decode path.
- `PC`: hardware program/PC-chain; multiple regcmd tasks in one NPU job.
- `hot`: prepared/resident weight path; excludes first-pack/prepare cost.
- `C5`: repeated real-HW exact/correctness evidence or direct production trace.
- `C4`: real-HW benchmark repeated enough for direction, but timing may jitter.
- `C3`: single HW probe and/or strong source inspection; enough to guide, not immutable.
- `C2`: plausible hypothesis; requires proof before production.
- `X`: do not overturn without contradictory real-HW evidence on current Rocket path.
- `R`: revisit only after a material architecture/kernel/backend change, not by repeating same test.
- `T`: intentionally open; worth testing.

## 1. Trial/error ledger

| 內容描述 | 信心指數 | 後人應不應該推翻 |
|---|---:|---|
| Plain FP16 `M=1` conv-as-matmul is **incorrect** on RK3588. Relaxing `M%4==0` to allow `M==1` produced large mismatches; old rocket-userspace `M=1` reference also prints FAILED. | C5 | X |
| `M=1 -> pad M=4` FP16 works numerically and beats RockNPU scalar CPU per projection, but is catastrophically worse than optimized llama.cpp CPU decode at whole-model scale. Do not productionize this padding trick. | C5 | X |
| Current useful decode route is `W8A8 M=1`, not plain FP16 M=1. GGML TinyLlama decode traces show real `w8a8_m1` execution for Q/O/K/V/gate/up/down. | C5 | X |
| Wide-K `K=5632` down projection must remain split/accumulated with current hardware programming. ork `Bf` storage supporting larger K does **not** imply ordinary full-K execution supports it; production execution gates still reject `K>4096`. | C5 | X |
| Do not infer an execution capability from a packed-weight/storage envelope. `Bf<=10752` was a false lead for full-K decode. | C5 | X |
| GGML backend is not a loader-only stub. `GGML_OP_MUL_MAT` already executes on RK3588; real TinyLlama prefill previously sent 131 Q4_K matmuls through RockNPU. | C5 | X |
| Stock llama.cpp dynamic loading is sufficient. No llama.cpp fork/patch/upstream PR is required for the backend. `GGML_BACKEND_PATH + GGML_BACKEND_DL=ON` is the intended integration. | C5 | X |
| Adding narrow `GGML_OP_GLU/SWIGLU` support causes stock llama.cpp to naturally group decode FFN into one RockNPU graph: `gate MM -> up MM -> SWIGLU -> down MM`, observed as `graph_compute nodes=4`. No scheduler fork required. | C5 | X |
| Temporary C++ F32 SWIGLU implementation was **only a scheduler/correctness probe**, not the target production implementation. `-n2` remained `The capital of the United`; 23 M=1 SWIGLU calls hit RockNPU. | C5 | X |
| V/K concat-N pair fusion is bit-exact. Stock local K=2048, N=256+256 measured `0.866 ms` separate vs `0.334 ms` fused 2-worker (`2.59x`; fused 3-worker `0.604 ms`). The old separate-process whole-token `+1.78%/+0.80%` verdict is superseded by same-process ABBA: experimental board `1.052582x`, stock board `1.029613x`, with all three ABBA blocks positive on both. Keep it in the combination pool; stock evidence is positive but sub-5% and decays in later blocks, so do not make a standalone production claim yet. | C4 | T |
| Gate/up concat-N pair is bit-exact and preserves the same W8A8 per-channel projection math. Stock local K=2048, N=5632+5632 measured `2.955 -> 2.843 ms` (~4%). Stock same-process ABBA measured `1.018275x`; all three blocks were positive, but blocks 2/3 were only ~`1.0063x/1.0081x`. Combination-only for now, not a standalone production win. | C4 | T |
| V/K + gate/up together stack positively rather than canceling in the current measurements: same-process ABBA `both` measured `1.037348x` on the experimental board and `1.045015x` on the stock board; all three blocks were positive on both. The gains are not additive, but the stock result is the strongest production-kernel evidence for retaining both pair optimizations as a combination candidate. | C4 | T |
| Small whole-token deltas must not be judged by adjacent independent llama-bench processes. The board showed order-dependent swings from roughly `-11%` to `+7.8%` without thermal throttling. For <5% claims use one process/context/backend, warm both variants, ABBA interleave, >=12 reps, and inspect block direction as well as aggregate median. | C5 | X |
| Same-input activation quantization reuse is real in the current W8 path, but after `both` pairing the remaining safe duplicate is mainly Q vs the V/K pair: one extra K=2048 F32->I8 quantization per transformer layer. A Release-equivalent host microbench measured ~`14.1–14.5 us` for K=2048 and ~`15.6 us` for K=5632; conservatively charging `15.8 us` gives <`0.2%` whole-token upper bound for 22 saved K=2048 quantizations/token at current TinyLlama latency. Do not add pointer/generation activation-cache complexity now; revisit only if decode latency falls enough for this host work to become material. | C4 | R |
| Two-op task chaining saves little because Rocket submit is already cheap: V/K ~1.10x, gate/up ~1.03x. Chaining merely to remove one submit is not a major lever. | C4 | R |
| Rocket submit ioctl measured ~`0.006–0.008 ms`; BO/fence completion ~`1.397–1.416 ms`. Host submit overhead is not the main decode bottleneck. | C4 | R |
| Copying ork's NONBLOCK/doorbell idea is low value on current Rocket path: RockNPU submit is already effectively enqueue-then-wait. Attack NPU work/dataflow, not ioctl microseconds. | C4 | R |
| Pinning workers/process to RK3588 A76 big cores did **not** help current RockNPU path. Repeated A/B: little ~`1.82–1.88 ms`, big ~`2.03–2.14 ms`. Do not cargo-cult ork affinity policy. | C4 | R |
| Auto-tuner may choose different topology between runs; raw single-run fused/unfused comparisons can be polluted by topology choice. Use interleaved repeated A/B and inspect selected topology. | C4 | X |
| Decode hot-cache timing has meaningful board jitter. Do not claim small (<~few %) wins from one run. | C4 | X |
| Prepared/resident decode weights are already part of the active path. A TinyLlama trace showed `decode_cache entries=154`, ~`924 MiB` resident after first fills. Do not return to dequant/repack-every-call design. | C5 | X |
| Current first-token/cache-fill cost can dwarf hot decode; distinguish `miss/prepare` from `hit/hot` in every benchmark. | C5 | X |
| ork-driver binary cannot be used directly as an apples-to-apples runtime on this Rocket kernel: its rknpu CREATE ioctl fails `EINVAL`. Use ork as hardware/regcmd knowledge, not as a drop-in userspace baseline here. | C5 | R |
| ork source contains a validated W8A8 FFN hardware chain: `MM_I8 -> SILU_I8 -> MM_I8 -> EWMUL_I8 -> MM_I8`. This is the right reference architecture for eliminating FFN F32 round-trips. | C5 | X |
| Critical hazard: ork reports **separate-submit `MM_I8 -> SILU_I8` hangs**, while the same transition is validated when kept inside one HW PC-chain. Never implement this as two independent submits just because both ops work separately. | C5 | X |
| Same chain constraints matter: current ork generic sequence eligibility requires int8 matmul `K%512==0 && K<=4096`; therefore TinyLlama `down K=5632` cannot simply be pasted into that generic chain unchanged. | C5 | X |
| A true Rocket PC-chain is a **joint userspace/kernel contract**. Stock Rocket executes a multi-task job as per-task kicks (`TASK_NUMBER=1`); self-linking those regcmds without batched-kernel support can corrupt/stall. Require `rocket_batch_submit!=0` or an explicitly versioned equivalent before setting `JOB_BATCHED`. | C5 | X |
| Current RK3588 executor kernel has `/dev/accel/accel0` but no `/sys/module/rocket/parameters/rocket_batch_submit`; do **not** run the dangerous PC-chain proof on this installed module. This is a kernel-capability blocker, not evidence against the hardware chain itself. | C5 | R |
| The next safe hardware proof remains Rocket `MM_I8 -> SILU_I8` in one true batched PC-chain with exact CPU oracle. Userspace fail-closed `JOB_BATCHED` plumbing and PC-trailer linking compile and pass targeted Rust tests (`rocket-uapi`, `rocket-runtime`, `rocknpu-regcmd`); HW validation is still blocked because the installed kernel lacks the batched capability signal. | C3 | T |
| For full TinyLlama FFN, likely architecture is: keep gate/up/SWIGLU intermediate inside RockNPU, then bridge to the already-working wide-K split down path rather than forcing unsupported full-K chain execution. | C2 | T |
| Native/quantized intermediate retention is a larger lever than shaving submit calls: current FFN crosses F32/CPU boundaries around SWIGLU/down. | C3 | T |
| Q4_K->FP16 correctness bridge is valid but not a performance endpoint. Do not optimize around preserving that bridge if a native quantized/resident path becomes provable. | C5 correctness / C3 perf direction | T |
| Q6_K exists in TinyLlama and is already seen in current traces; do not assume a Q4_K-only model when analyzing actual projection coverage. | C5 | X |
| Native W4A4 M=1 hardware is real and exact at the integer primitive: K2048/N5632 three-way N-split measured ~`0.97 ms` vs ~`2.08 ms` single worker with zero saturation; real-model tuner chose three workers (`~2081/1251/859 us`). | C5 | X |
| Native W4A4 is **not model-quality equivalent** yet. FFN full-K+Hadamard, FFN G=512+Hadamard, and Q/O-only W4 all diverged from the default W8 deterministic continuation on the 16-token gate despite zero saturation. Keep W4 explicit opt-in; do not make it default from short-gate success. | C5 | X |
| Whole-model W4 speed is currently noise-level vs W8: same-setting TinyLlama `tg8` measured `5.09 ± 0.13` vs `5.02 ± 0.17 tok/s`. Do not advertise a decode speedup from the much faster local W4 FFN primitive. | C4 | X |
| W4 quantization quality, not Rocket execution/overflow, is now the blocking problem. Hadamard helps short-gate quality but does not preserve longer token identity. Future W4 work should target better quantization/calibration or quantized dataflow, not another blind worker-count/group-size sweep. | C5 | T |
| `ROCKNPU_GGML_TRACE=1` is the required first diagnostic for claims about routing. Verify weight name, type, M/K/N, chosen path, cache hit/miss, graph node grouping before changing kernels. | C5 | X |
| Do not judge a reference binary only by exit code. Old reference M=1 test printed FAILED/mismatches while returning 0. Inspect correctness text/data. | C5 | X |

## 2. 2026-09-15 same-process pair ABBA evidence

All whole-token rows below used llama.cpp `391fac16460f15233a7740550d858ac96df3419d`, TinyLlama-1.1B-Chat-v1.0-Q4_K_M, `tg32`, `r=12`, one model/context/backend process, warm baseline+candidate, and `B,A,A,B` interleave. The GGML plugin build artifacts on both boards contain `-O3 -DNDEBUG -fPIC`. `B` is pair(s) disabled; `A` is the named candidate.

The first board is exploration-only because its loaded module exposes experimental Rocket parameters (`rocket_batch_submit=Y`, `rocket_force_core0=N`). The second board lacks those parameters and is the stock production-kernel validator.

- Experimental V/K: baseline samples `[6355285676, 6563720148, 6443800844, 6391513469, 6438364227, 6526746411]`; candidate `[6218370374, 6111478134, 6328047050, 6125652262, 6112986243, 6093505280]`; medians `6441082535 -> 6119319252 ns`; speedup `1.052582x`; block speedups `1.0478x, 1.0306x, 1.0621x`.
- Stock FFN pair: baseline `[6938423457, 6590359038, 6639709940, 6620918158, 6726314187, 6667731585]`; candidate `[6396010572, 6452092152, 6656396191, 6521481652, 6547130946, 6739062416]`; medians `6653720762 -> 6534306299 ns`; speedup `1.018275x`; block speedups `1.0530x, 1.0063x, 1.0081x`.
- Stock V/K pair: baseline `[6551763654, 6502699026, 6550185944, 6499084060, 6379214169, 6432363366]`; candidate `[6113958276, 6296337698, 6331495429, 6269001866, 6357148750, 6332921532]`; medians `6500891543 -> 6313916563 ns`; speedup `1.029613x`; block speedups `1.0519x, 1.0356x, 1.0096x`.
- Experimental BOTH: baseline `[6343612563, 6252070036, 6202569324, 6263966119, 6191903499, 6258387101]`; candidate `[6128375958, 6026424712, 6053667633, 5887300198, 6027502704, 6032537273]`; medians `6255228568 -> 6030019988 ns`; speedup `1.037348x`; block speedups `1.0363x, 1.0440x, 1.0324x`.
- Stock BOTH: baseline `[7107273605, 6670265913, 6863762311, 6912329792, 6766908576, 6666913087]`; candidate `[6408075269, 6700032956, 6654328583, 6443878125, 6599636630, 6430685882]`; medians `6815335443 -> 6521757377 ns`; speedup `1.045015x`; block speedups `1.0511x, 1.0518x, 1.0310x`.

The pair implementations remain opt-in experimental/runtime-toggle paths. These measurements justify retaining them and testing them as a bundle; they do not justify changing main/default behavior yet.

## 3. Current dirty/research state

Committed baseline when this log was created:

`092d70da235ad7802f534dcbc711ad42092c52e6` — `Auto-tune wide-K W8A8 split topology`

Uncommitted research files at that point:

- `adapters/ggml-rocknpu/src/ggml-rocknpu.cpp` — temporary SWIGLU scheduler probe + graph tracing; do not mistake C++ F32 SWIGLU for final design.
- `crates/rocknpu-matmul/src/int8_decode.rs` — experimental instrumentation/changes from decode investigations.
- `crates/rocket-smoke/src/bin/int8_decode_multicore.rs` — instrumentation/benchmark edits.
- `crates/rocket-smoke/src/bin/int8_decode_chain.rs` — task-chain experiment.

Before committing any of these, separate durable instrumentation/probes from rejected production optimizations.

## 4. Decision rule for future agents

1. If an item is `X`, do not rerun the same idea with renamed code. Require contradictory current-HW evidence.
2. If an item is `R`, only reopen after a material change (kernel/UAPI/regcmd model/quantization/dataflow), and state what changed.
3. Spend effort on `T` items first.
4. Every perf claim must preserve exact/acceptable correctness and be measured at the **whole decode/token** level, not only a local operator microbench.
5. For risky SDP/int8 mode transitions, prefer known-safe PC-chain topology. A separately valid op does not imply a safe cross-submit transition.
