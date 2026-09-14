#!/usr/bin/env python3
from pathlib import Path
import onnx
import numpy as np
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator

root = Path(__file__).resolve().parent.parent / "artifacts"
model = onnx.load(root / "tiny-mlp.onnx")
onnx.checker.check_model(model)
bits = np.fromfile(root / "tiny-mlp-input-f16.bin", dtype="<u2")
inp = bits.view(np.float16).reshape(4, 32).astype(np.float32)
ref_final = ReferenceEvaluator(model).run(None, {"input": inp})[0]
env = {"input": inp}
for t in model.graph.initializer:
    env[t.name] = numpy_helper.to_array(t).astype(np.float32)
expected = {}
for node in model.graph.node:
    ins = [env[n] for n in node.input]
    if node.op_type == "MatMul": out = np.matmul(ins[0], ins[1])
    elif node.op_type == "Add": out = ins[0] + ins[1]
    elif node.op_type == "Relu": out = np.maximum(ins[0], 0)
    else: raise RuntimeError(node.op_type)
    env[node.output[0]] = out.astype(np.float32, copy=False)
    expected[node.output[0]] = env[node.output[0]]
assert np.array_equal(ref_final, env["output"])
for line in (root / "tiny-mlp-npu-trace.tsv").read_text().splitlines():
    op, name, r, c, hexes = line.split("\t")
    u = np.array([int(x, 16) for x in hexes.split(",")], dtype=np.uint16)
    got = u.view(np.float16).reshape(int(r), int(c))
    exp = expected[name]
    err = np.max(np.abs(got.astype(np.float32) - exp))
    bit_exact = np.array_equal(got.view(np.uint16), exp.astype(np.float16).view(np.uint16))
    fp32_exact = np.array_equal(got.astype(np.float32), exp)
    print(f"{op:6s} {name:8s} fp16_bits={bit_exact} fp32_exact={fp32_exact} max_abs={float(err):.9g}")
    assert bit_exact and fp32_exact and err == 0
npu = np.fromfile(root / "tiny-mlp-npu-f16.bin", dtype="<u2").view(np.float16).reshape(4, 16)
assert np.array_equal(npu.astype(np.float32), ref_final)
print("PASS: ONNX ReferenceEvaluator == NumPy node trace == RK3588 NPU, max_abs_error=0")
