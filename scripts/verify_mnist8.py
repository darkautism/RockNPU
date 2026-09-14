#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

root=Path(__file__).resolve().parent.parent
prefix = sys.argv[1] if len(sys.argv) > 1 else "mnist8-npu"
model=onnx.load(root/'artifacts/mnist-8.onnx')
onnx.checker.check_model(model)
x100=np.fromfile(root/'artifacts/mnist8-input100-f32.bin',dtype='<f4').reshape(100,1,28,28)
ref100=np.fromfile(root/'artifacts/mnist8-ref100-f32.bin',dtype='<f4').reshape(100,10)
labels=np.fromfile(root/'artifacts/mnist8-label100-u8.bin',dtype='u1')
npu100=np.fromfile(root/f'artifacts/{prefix}100-f16.bin',dtype='<f2').astype(np.float32).reshape(100,10)

meta=[]
with open(root/f'artifacts/{prefix}-trace.tsv') as f:
    next(f)
    for line in f:
        i,op,name,dims,offset,count=line.rstrip('\n').split('\t')
        meta.append((int(i),op,name,tuple(map(int,dims.split('x'))),int(offset),int(count)))
raw=np.fromfile(root/f'artifacts/{prefix}-trace-f16.bin',dtype='<f2').astype(np.float32)
names=[n.output[0] for n in model.graph.node]
ref_nodes=ReferenceEvaluator(model).run(names,{'Input3':x100[:1]})
assert len(meta)==len(names)==12
print(f'prefix={prefix} onnx_checker=PASS node_count=12 trace_count=12')
for (i,op,name,dims,offset,count),rname,ref in zip(meta,names,ref_nodes):
    assert i==len([x for x in meta if x[0]<i])
    assert name==rname
    got=raw[offset:offset+count].reshape(dims)
    ref=np.asarray(ref,dtype=np.float32)
    assert got.shape==ref.shape,(name,got.shape,ref.shape)
    err=np.abs(got-ref)
    rms=float(np.sqrt(np.mean((got-ref)**2)))
    print(f'node{i} op={op} name={name} shape={dims} max_abs={err.max():.8f} mean_abs={err.mean():.8f} rms={rms:.8f}')

ref_pred=ref100.argmax(1); npu_pred=npu100.argmax(1)
err=np.abs(npu100-ref100)
print('final_top1_match',int(np.sum(ref_pred==npu_pred)),'/100')
print('reference_accuracy',int(np.sum(ref_pred==labels)),'/100','npu_accuracy',int(np.sum(npu_pred==labels)),'/100')
print(f'final_max_abs={err.max():.8f} final_mean_abs={err.mean():.8f}')
assert np.all(ref_pred==npu_pred)
assert int(np.sum(ref_pred==labels))==98
assert int(np.sum(npu_pred==labels))==98
print('PASS: official MNIST-8 12-node trace and 100-sample RockNPU output match ONNX ReferenceEvaluator')
