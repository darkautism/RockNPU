#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path

import numpy as np

MAP = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}

def read_header(path: Path):
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return 8 + header_len, header

def bf16_rows(path: Path, offset: int, shape):
    n, k = shape
    raw = np.memmap(path, dtype=np.uint16, mode="r", offset=offset, shape=(n, k))
    return raw

def quantize_tensor(src: Path, data_base: int, meta, out_base: Path, chunk_rows: int):
    if meta["dtype"] != "BF16" or len(meta["shape"]) != 2:
        raise ValueError(f"expected rank-2 BF16, got {meta}")
    n, k = map(int, meta["shape"])
    begin, end = map(int, meta["data_offsets"])
    if end - begin != n * k * 2:
        raise ValueError("safetensors byte range does not match BF16 shape")
    raw = bf16_rows(src, data_base + begin, (n, k))
    scales = np.empty(n, dtype=np.float32)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    with (Path(str(out_base) + ".w8")).open("wb") as wf:
        for r0 in range(0, n, chunk_rows):
            r1 = min(n, r0 + chunk_rows)
            u32 = raw[r0:r1].astype(np.uint32) << np.uint32(16)
            values = u32.view(np.float32)
            max_abs = np.max(np.abs(values), axis=1)
            scale = np.where(max_abs == 0.0, np.float32(1.0), max_abs / np.float32(127.0)).astype(np.float32)
            q = np.rint(values / scale[:, None]).clip(-127, 127).astype(np.int8)
            wf.write(q.tobytes(order="C"))
            scales[r0:r1] = scale
    scales.tofile(str(out_base) + ".scale.f32")
    return n, k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("safetensors", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--layers", type=int, default=22)
    ap.add_argument("--chunk-rows", type=int, default=64)
    args = ap.parse_args()
    data_base, header = read_header(args.safetensors)
    manifest = {"format": "rocknpu-w8-sidecar-v1", "source": str(args.safetensors), "tensors": {}}
    for layer in range(args.layers):
        for hf_suffix, ggml_suffix in MAP.items():
            hf = f"model.layers.{layer}.{hf_suffix}.weight"
            ggml = f"blk.{layer}.{ggml_suffix}.weight"
            if hf not in header:
                raise KeyError(hf)
            n, k = quantize_tensor(args.safetensors, data_base, header[hf], args.output_dir / ggml, args.chunk_rows)
            manifest["tensors"][ggml] = {"k": k, "n": n, "scheme": "symmetric-per-output-channel-int8"}
            print(f"{ggml} K={k} N={n}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(manifest['tensors'])} tensors to {args.output_dir}")

if __name__ == "__main__":
    main()
