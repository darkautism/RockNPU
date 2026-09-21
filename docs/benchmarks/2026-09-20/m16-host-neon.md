# RK3588 host NEON for grouped M16 W8

## Motivation

The quality-restoring M16 path uses 2-way Q/O grouping (`K=2048 -> 2x1024`). Profiling on o8 showed that host activation quantization and scaled output accumulation had become material parts of pp16 latency even though the dot-product itself runs on the NPU.

Both validation boards report Cortex-A76 big cores with AArch64 ASIMD/NEON and `asimddp` (dotprod). RockNPU otherwise builds the host Rust code as generic AArch64; there was no Cortex-A76 or dotprod-specific host path.

## Baseline profile

Grouped M16 configuration:

```text
ROCKNPU_W8_MTILE=1
ROCKNPU_W8_MTILE_SCOPE=all
ROCKNPU_MTILE_QO_GROUP=1024
ROCKNPU_MTILE_PERSIST=1
ROCKNPU_PREFILL_CACHE=1
```

Scalar-profile totals over the same benchmark window:

- M16 calls: 696
- activation quantize: `51.860 ms`
- NPU wait: `336.621 ms`
- executor total: `424.357 ms`
- scaled output accumulation: `42.649 ms`
- pp16: approximately `75.01 tok/s` under profiling

## Attempt 1: explicit second-pass quantize + explicit rescale NEON

The first implementation kept scalar max/finite reduction and used AArch64 intrinsics for the f32->i8 second pass and int32->f32 scaled accumulation.

Direct differential tests made quantized bytes and rescale FP32 bits identical to scalar. However the profile showed little benefit:

- quantize: `51.860 -> 50.252 ms`
- rescale: `42.649 -> 44.222 ms` (regression)
- pp16 profile: `74.75 tok/s`

Conclusion: LLVM already vectorizes those arithmetic loops adequately. The explicit rescale path was removed.

## Attempt 2: NEON max-abs + finite reduction

The remaining likely scalar bottleneck was the first quantization pass:

```text
finite check + max(abs(x))
```

The branch now uses a NEON reduction only when `ROCKNPU_HOST_NEON=1`:

- 4-wide absolute value and finite mask
- vector max reduction
- exact scalar tail
- existing scalar fallback unchanged
- bit-exact vector FDIV/round/narrow second pass retained

Direct AArch64 tests:

- NEON max-abs equals scalar bit-for-bit and rejects `Inf`/`NaN`: PASS
- NEON quantize second pass equals scalar i8 bytes exactly, including tail and boundary-style values: PASS

Profile after the max-reduction change:

- activation quantize: `51.860 -> 25.859 ms` (~2.01x faster)
- NPU wait: `336.123 ms` (unchanged)
- scaled accumulation: `42.563 ms` (scalar path restored; unchanged)
- pp16 profile: `77.87 tok/s`

No-profile o8 pp16 r=5:

```text
78.3469
78.8673
78.8423
79.1020
76.2222 tok/s
```

Mean: `78.276165 tok/s`.

Compared with the grouped-Q/O candidate before host NEON (`75.989331 tok/s`), this is about `+3.0%`. Compared with the same-board CPU mean (`71.344361 tok/s`), the grouped-M16 + host-NEON path is about `+9.7%`.

The known France/Germany 16-token deterministic quality gate still produced the same CPU continuation after enabling host NEON.

## Cortex-A76 + dotprod compiler tuning

As a separate no-source-change experiment, only `rocknpu-capi` was rebuilt with:

```text
-C target-cpu=cortex-a76 -C target-feature=+dotprod
```

No-profile pp16 r=5 averaged `78.247307 tok/s`, statistically indistinguishable from generic-AArch64 codegen plus the explicit NEON reduction (`78.276165 tok/s`).

This is expected: the current host hotspot performs reduction/quantization, not INT8 dot products. `dotprod`/SDOT remains interesting for future CPU-routed small matmuls, but it does not improve this NPU-front-end path.

## Independent o16 reproduction

Exact candidate `401d22538ca198187471e84be5add78f7db4501f` was validated independently on o16.

- direct NEON max-abs/finite differential: PASS
- direct NEON quantize byte-for-byte differential: PASS
- known France/Germany exact-16-token deterministic model gate: identical CPU continuation
- grouped-M16 + host-NEON pp16 r=5: `88.4580, 84.8920, 91.0175, 93.0577, 92.1631 tok/s`
- mean: `89.917679 tok/s`

The pre-NEON grouped-Q/O candidate on o16 averaged `84.051609 tok/s`, so host NEON adds about `+7.0%` independently on the second board. Against the o16 CPU mean `73.917624 tok/s`, the combined grouped-M16 + host-NEON path is about `+21.6%`.

## Default-on policy

After cross-board reproduction, AArch64 builds now enable the host NEON path by default when `is_aarch64_feature_detected!("neon")` succeeds. `ROCKNPU_HOST_NEON=0` remains an explicit scalar opt-out for diagnostics/nonstandard hosts.

An o8 pp16 run with no `ROCKNPU_HOST_NEON` variable set averaged `79.205144 tok/s` (`79.7357, 76.8097, 81.0700`), confirming runtime feature detection selected the fast path automatically.

## Verdict

**PROMOTE.**

The useful mechanism is explicit NEON max-abs/finite reduction, not generic Cortex-A76/dotprod compiler tuning and not explicit NEON rescale. It is bit-exact against the scalar quantizer, preserves the grouped-M16 model quality gate, and produces reproducible whole-model pp16 gains on both RK3588 boards.

