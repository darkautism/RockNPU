#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def round_away_from_zero(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0.0, np.floor(values + 0.5), np.ceil(values - 0.5))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_sample_fnv1a64(values: np.ndarray) -> str:
    raw = memoryview(np.ascontiguousarray(values)).cast("B")
    offset_basis = 14695981039346656037
    prime = 1099511628211
    mask = (1 << 64) - 1
    h = offset_basis
    byte_count = len(raw)
    for shift in range(0, 64, 8):
        h ^= (byte_count >> shift) & 0xFF
        h = (h * prime) & mask
    sample_bytes = 4096
    if byte_count <= sample_bytes * 3:
        ranges = ((0, byte_count),)
    else:
        ranges = (
            (0, sample_bytes),
            ((byte_count - sample_bytes) // 2, sample_bytes),
            (byte_count - sample_bytes, sample_bytes),
        )
    for start, count in ranges:
        for value in raw[start : start + count]:
            h ^= value
            h = (h * prime) & mask
    return f"{h:016x}"


def quantize_rows(values: np.ndarray, out_base: Path, chunk_rows: int) -> tuple[int, int]:
    if values.ndim != 2 or values.dtype != np.float32:
        raise ValueError(f"expected rank-2 float32 tensor, got shape={values.shape} dtype={values.dtype}")
    n, k = map(int, values.shape)
    scales = np.empty(n, dtype=np.float32)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(out_base) + ".w8").open("wb") as wf:
        for r0 in range(0, n, chunk_rows):
            r1 = min(n, r0 + chunk_rows)
            rows = values[r0:r1]
            max_abs = np.max(np.abs(rows), axis=1).astype(np.float32)
            scale = np.where(
                max_abs == 0.0,
                np.float32(1.0),
                max_abs / np.float32(127.0),
            ).astype(np.float32)
            normalized = (rows / scale[:, None]).astype(np.float32)
            q = round_away_from_zero(normalized).clip(-127, 127).astype(np.int8)
            wf.write(q.tobytes(order="C"))
            scales[r0:r1] = scale
    scales.tofile(str(out_base) + ".scale.f32")
    return n, k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--gguf-py", type=Path, required=True)
    ap.add_argument("--layers", type=int, default=22)
    ap.add_argument("--chunk-rows", type=int, default=64)
    ap.add_argument(
        "--include-output-head",
        action="store_true",
        help="also prepare the wide output/vocabulary projection for N-split decode",
    )
    args = ap.parse_args()

    sys.path.insert(0, str(args.gguf_py))
    from gguf import GGUFReader, dequantize

    reader = GGUFReader(str(args.gguf), "r")
    by_name = {tensor.name: tensor for tensor in reader.tensors}
    suffixes = [
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    ]
    if args.include_output_head:
        suffixes.append("output.weight")
    manifest = {
        "format": "rocknpu-w8-sidecar-v2",
        "source": str(args.gguf),
        "source_size": args.gguf.stat().st_size,
        "source_sha256": sha256_file(args.gguf),
        "source_semantics": "GGUF tensor dequantize -> symmetric-per-output-channel-int8",
        "rounding": "half-away-from-zero",
        "tensors": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for layer in range(args.layers):
        for suffix in suffixes:
            name = f"blk.{layer}.{suffix}"
            tensor = by_name.get(name)
            if tensor is None:
                raise KeyError(name)
            source_fingerprint = source_sample_fnv1a64(tensor.data)
            values = dequantize(tensor.data, tensor.tensor_type)
            out_base = args.output_dir / name
            n, k = quantize_rows(values, out_base, args.chunk_rows)
            Path(str(out_base) + ".source.fnv1a64").write_text(source_fingerprint + "\n")
            expected_k, expected_n = map(int, tensor.shape)
            if (k, n) != (expected_k, expected_n):
                raise ValueError(
                    f"{name}: dequant orientation mismatch: got K={k} N={n}, expected K={expected_k} N={expected_n}"
                )
            manifest["tensors"][name] = {
                "k": k,
                "n": n,
                "scheme": "symmetric-per-output-channel-int8",
                "ggml_type": int(tensor.tensor_type),
                "source_sample_fnv1a64": source_fingerprint,
            }
            print(f"{name} K={k} N={n} type={int(tensor.tensor_type)}", flush=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(manifest['tensors'])} tensors to {args.output_dir}")


if __name__ == "__main__":
    main()
