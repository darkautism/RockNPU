# Userspace checkpoint — 2026-09-22

This checkpoint replaces the mixed-environment NPU benchmark notes that were removed during the project-scope cleanup.

Only conclusions that remain useful to the userspace RockNPU project are retained here.

## Fresh current-main revalidation

Current userspace code was rebuilt from a clean main checkout after reboot and revalidated through the ordinary accelerator device interface.

Exact/correctness gates:

- M=1 K=5632 N=2048 prepared decode: PASS.
- M=128 K=2048 N=2048: PASS.
- fused residual M16/K2048/N2048: PASS, max abs 0.000610.
- repeated M128 same-weight reuse: PASS.

## M128 relative result

100%-acceptance lookup A-B-B-A:

- M64: 126.174 tok/s
- M128: 137.932 tok/s
- M128 repeat: 136.478 tok/s
- M64 repeat: 122.309 tok/s

Using the two-run centers, M128 is approximately 10.4% faster than M64 for this verifier workload.

The relative result is the important conclusion. These absolute numbers are not promoted as a general full-speed TinyLlama topline.

## CPU reference

Native ARM llama.cpp, TinyLlama Q4_K_M, tg128:

- 33.88 +/- 0.08 tok/s.

This is the current clean CPU reference for ordinary decode comparisons.

## Interpretation

- M128 is a real userspace improvement and remains promoted.
- Full-K K=5632 M=1 remains valid.
- fused residual remains valid.
- same-job weight reuse remains a valid primitive.
- ordinary M=1 model throughput still needs a new current-main baseline before making a new NPU-vs-CPU claim.

Canonical direction: ../../research-status.md
