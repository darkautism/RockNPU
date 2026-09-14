#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts'
MODEL=ART/'cifar10-edgeinfer.onnx'
INPUT=ART/'cifar10-edgeinfer-input-f32.bin'

def load_trace(prefix):
    lines=(ART/f'{prefix}-trace.tsv').read_text().strip().splitlines()
    raw=np.fromfile(ART/f'{prefix}-trace-f16.bin',dtype='<f2')
    rows=[]
    for line in lines[1:]:
        i,op,name,dims,off,count=line.split('\t')
        shape=tuple(map(int,dims.split('x'))); off=int(off); count=int(count)
        vals=raw[off:off+count].astype(np.float32).reshape(shape)
        rows.append((int(i),op,name,shape,vals))
    return rows,raw

def verify(prefix, refs, nodes):
    rows,raw=load_trace(prefix)
    if len(rows)!=len(refs): raise SystemExit(f'{prefix}: trace count {len(rows)} != {len(refs)}')
    max_seen=0.0
    print(f'prefix={prefix} trace_count={len(rows)}')
    for (idx,op,name,shape,got),ref,node in zip(rows,refs,nodes):
        ref=np.asarray(ref,dtype=np.float32)
        if idx>=len(nodes) or op!=node.op_type or name!=node.output[0] or shape!=ref.shape:
            raise SystemExit(f'{prefix}: node metadata mismatch at {idx}: {op} {name} {shape} vs {node.op_type} {node.output[0]} {ref.shape}')
        d=np.abs(got-ref)
        if not np.isfinite(got).all(): raise SystemExit(f'{prefix}: nonfinite at node {idx}')
        mx=float(d.max(initial=0)); mean=float(d.mean()); rms=float(np.sqrt(np.mean(d.astype(np.float64)**2)))
        max_seen=max(max_seen,mx)
        print(f'node{idx} op={op} name={name} shape={shape} max_abs={mx:.8f} mean_abs={mean:.8f} rms={rms:.8f}')
    final=rows[-1][4].reshape(-1); pred=int(final.argmax())
    ref_pred=int(np.asarray(refs[-1]).reshape(-1).argmax())
    if pred!=8 or ref_pred!=8: raise SystemExit(f'{prefix}: top1 got={pred} ref={ref_pred}, expected ship=8')
    if max_seen>=0.1: raise SystemExit(f'{prefix}: node max_abs {max_seen} exceeds 0.1')
    return raw,max_seen

def main():
    m=onnx.load(str(MODEL)); onnx.checker.check_model(m)
    x=np.fromfile(INPUT,dtype='<f4').reshape(1,3,32,32)
    nodes=list(m.graph.node); names=[n.output[0] for n in nodes]
    refs=ReferenceEvaluator(m).run(names,{'input':x})
    cpu,cpu_max=verify('cifar10-edgeinfer-cpu',refs,nodes)
    npu,npu_max=verify('cifar10-edgeinfer-npu',refs,nodes)
    if cpu.shape!=npu.shape: raise SystemExit('CPU/NPU trace size mismatch')
    cpu_bits=cpu.view(np.uint16); npu_bits=npu.view(np.uint16)
    bitdiff=int(np.count_nonzero(cpu_bits!=npu_bits))
    cpu_f=cpu.astype(np.float32); npu_f=npu.astype(np.float32)
    if not np.isfinite(cpu_f).all() or not np.isfinite(npu_f).all():
        raise SystemExit('non-finite CPU/NPU trace value')
    cross_max=float(np.max(np.abs(cpu_f-npu_f), initial=0.0))
    final_count=int(np.prod(refs[-1].shape))
    final_bitdiff=int(np.count_nonzero(cpu_bits[-final_count:]!=npu_bits[-final_count:]))
    print(f'cpu_npu_trace_bitdiff={bitdiff}/{cpu.size} cpu_npu_cross_max_abs={cross_max:.8f} final_bitdiff={final_bitdiff}/{final_count} cpu_max_abs={cpu_max:.8f} npu_max_abs={npu_max:.8f}')
    if cross_max>0.002: raise SystemExit(f'CPU/NPU trace max_abs {cross_max} exceeds 0.002')
    if final_bitdiff!=0: raise SystemExit(f'CPU/NPU final logits differ in {final_bitdiff} fp16 elements')
    print('PASS: edge-infer CIFAR-10 13-node CPU/NPU traces independently match ONNX ReferenceEvaluator; CPU/NPU rounding stays tightly bounded and final logits are bit-identical')
if __name__=='__main__': main()
