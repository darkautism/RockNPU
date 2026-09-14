#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import onnx
from onnx import numpy_helper

root = Path(__file__).resolve().parents[1]
art = root / "artifacts"
model_path = art / "external/pico-cnn-mnist-mlp.onnx"
prefix = sys.argv[1] if len(sys.argv) > 1 else "real-mnist-npu"

model = onnx.load(model_path)
onnx.checker.check_model(model)
x = np.fromfile(art / "external/mnist-first50-f32.bin", dtype="<f4").reshape(50, 784)
ref = np.fromfile(art / "external/mnist-first50-onnx-ref-f32.bin", dtype="<f4").reshape(50, 10)
labels = np.fromfile(art / "external/mnist-first50-labels-u8.bin", dtype=np.uint8)
initializers = {t.name: numpy_helper.to_array(t).astype(np.float32) for t in model.graph.initializer}

meta = []
for line in (art / f"{prefix}-trace.tsv").read_text().splitlines():
    op, name, rows, cols, off, length = line.split("\t")
    meta.append((op, name, int(rows), int(cols), int(off), int(length)))
raw = np.fromfile(art / f"{prefix}-trace-f16.bin", dtype="<f2")
trace = {
    name: raw[off : off + length].reshape(rows, cols).astype(np.float32)
    for op, name, rows, cols, off, length in meta
}

values = {"input.1": x}
print(f"trace_prefix={prefix}")
print("onnx_checker=PASS")
print(f"node_count={len(model.graph.node)} trace_count={len(meta)}")
for index, node in enumerate(model.graph.node):
    if node.op_type == "Gemm":
        # This external model declares alpha=beta=1, transA=0, transB=1.
        y = values[node.input[0]] @ initializers[node.input[1]].T + initializers[node.input[2]]
    elif node.op_type == "Relu":
        y = np.maximum(values[node.input[0]], 0)
    else:
        raise RuntimeError(f"unsupported verification op {node.op_type}")
    y = y.astype(np.float32)
    values[node.output[0]] = y
    got = trace[node.output[0]]
    err = np.abs(got - y)
    print(
        f"node{index} op={node.op_type} name={node.output[0]} shape={tuple(y.shape)} "
        f"max_abs={err.max():.8f} mean_abs={err.mean():.8f} rms={(np.mean(err*err)**0.5):.8f}"
    )

final = values[model.graph.output[0].name]
if not np.array_equal(final, ref):
    raise SystemExit("FAIL: NumPy ONNX semantics differ from saved ONNX ReferenceEvaluator output")
npu = trace[model.graph.output[0].name]
ref_pred = ref.argmax(axis=1)
npu_pred = npu.argmax(axis=1)
match = int(np.sum(ref_pred == npu_pred))
ref_correct = int(np.sum(ref_pred == labels))
npu_correct = int(np.sum(npu_pred == labels))
max_abs = float(np.max(np.abs(npu - ref)))
mean_abs = float(np.mean(np.abs(npu - ref)))
print("numpy_final_vs_onnx_reference_exact=True")
print(f"npu_top1_vs_ref={match}/50")
print(f"reference_accuracy={ref_correct}/50 npu_accuracy={npu_correct}/50")
print(f"npu_final_max_abs={max_abs:.8f} npu_final_mean_abs={mean_abs:.8f}")
if match != 50:
    raise SystemExit("FAIL: NPU top-1 differs from ONNX reference")
print("PASS: real pretrained MNIST model matches ONNX top-1 on all 50 samples")
