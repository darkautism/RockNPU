#!/usr/bin/env python3
"""Run a fixed-condition, raw-preserving candidate matrix for the RockNPU adapter."""
import argparse, hashlib, json, os, re, shlex, subprocess, sys, time
from pathlib import Path

from bench_llama_cpu_npu import snapshot, sha256

CANDIDATES = {
    "cpu-reference": {},
    "baseline": {},
    "direct-submit": {"ROCKNPU_W8_DIRECT_SUBMIT": "1"},
    "direct-scratch": {"ROCKNPU_W8_DIRECT_SUBMIT": "1", "ROCKNPU_EXPERIMENT_DIRECT_SCRATCH": "1"},
    "scheduler-cpu-qo": {"ROCKNPU_W8_DIRECT_SUBMIT": "1", "ROCKNPU_EXPERIMENT_DIRECT_SCRATCH": "1", "ROCKNPU_SCHED_CPU_QO": "1"},
    "scheduler-ffn-only": {"ROCKNPU_W8_DIRECT_SUBMIT": "1", "ROCKNPU_EXPERIMENT_DIRECT_SCRATCH": "1", "ROCKNPU_SCHED_FFN_ONLY": "1"},
    "k64-candidate": {"ROCKNPU_W8_DIRECT_SUBMIT": "1", "ROCKNPU_EXPERIMENT_DIRECT_SCRATCH": "1", "ROCKNPU_EXPERIMENT_K64": "1"},
    "m8-mtile": {"ROCKNPU_W8_DIRECT_SUBMIT": "1", "ROCKNPU_EXPERIMENT_DIRECT_SCRATCH": "1", "ROCKNPU_W8_MTILE": "1", "ROCKNPU_NATIVE_MTILE": "1", "ROCKNPU_W8_MTILE_SCOPE": "all"},
    "m16-qo-group": {"ROCKNPU_W8_DIRECT_SUBMIT": "1", "ROCKNPU_EXPERIMENT_DIRECT_SCRATCH": "1", "ROCKNPU_MTILE_QO_GROUP": "1024"},
}

def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--bench',type=Path,required=True);ap.add_argument('--plugin',type=Path,required=True);ap.add_argument('--model',type=Path,required=True);ap.add_argument('--sidecar',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--board',required=True);ap.add_argument('--expected-npu-freq',type=int,default=700000000);ap.add_argument('--tokens',type=int,default=8);ap.add_argument('--reps',type=int,default=1);a=ap.parse_args()
    for n in ('bench','plugin','model','sidecar'):setattr(a,n,getattr(a,n).resolve(strict=True))
    a.output.mkdir(parents=True,exist_ok=False); initial=snapshot(); base={k:v for k,v in os.environ.items() if not k.startswith('ROCKNPU_') and k not in ('GGML_BACKEND_PATH','GGML_SCHED_DEBUG')}
    rows=[]; commands=[]
    for name,extra in CANDIDATES.items():
        is_cpu=name=='cpu-reference'; env=base.copy()
        if not is_cpu:
            env.update({'GGML_BACKEND_PATH':str(a.plugin),'ROCKNPU_W8_SIDECAR_DIR':str(a.sidecar),'ROCKNPU_DECODE':'1','ROCKNPU_PREFILL_CACHE':'1','ROCKNPU_DISPATCH_SUMMARY':'1'})
        env.update(extra)
        device='none' if is_cpu else 'ROCKNPU0'
        cmd=['taskset','-c','4-7',str(a.bench),'-m',str(a.model),'-p','0','-n',str(a.tokens),'-r',str(a.reps),'-t','4','-fa','on','-dev',device,'-nopo','1' if is_cpu else '0','-o','json']
        before=snapshot(); freq={k:before.get(k) for k in ('/sys/class/devfreq/fdab0000.npu/cur_freq','/sys/class/devfreq/fdab0000.npu/target_freq')}
        if not is_cpu and (freq['/sys/class/devfreq/fdab0000.npu/cur_freq']!=str(a.expected_npu_freq) or freq['/sys/class/devfreq/fdab0000.npu/target_freq']!=str(a.expected_npu_freq)):
            raise RuntimeError(f'{name}: NPU frequency gate failed: {freq}')
        prefix=a.output/name; prefix.parent.mkdir(parents=True,exist_ok=True)
        start=time.monotonic()
        with (prefix.with_suffix('.stdout')).open('wb') as out,(prefix.with_suffix('.stderr')).open('wb') as err:
            p=subprocess.run(cmd,env=env,stdout=out,stderr=err,timeout=1800)
        after=snapshot(); stdout=prefix.with_suffix('.stdout').read_text(errors='replace'); stderr=prefix.with_suffix('.stderr').read_text(errors='replace')
        try: result=json.loads(stdout)
        except Exception: result=None
        summaries=re.findall(r'ROCKNPU GGML TRACE summary .*',stderr)
        row={'name':name,'flags':extra,'command':cmd,'environment':{k:v for k,v in sorted(env.items()) if k.startswith('ROCKNPU_') or k in ('GGML_BACKEND_PATH','LD_LIBRARY_PATH')},'before':before,'after':after,'frequency_before':freq,'frequency_after':{k:after.get(k) for k in freq},'exit_code':p.returncode,'wall_s':time.monotonic()-start,'stdout_file':str(prefix.with_suffix('.stdout')),'stderr_file':str(prefix.with_suffix('.stderr')),'stdout_sha256':hashlib.sha256(stdout.encode()).hexdigest(),'stderr_sha256':hashlib.sha256(stderr.encode()).hexdigest(),'json':result,'dispatch_summary_lines':summaries}
        write(prefix.with_suffix('.record.json'),row); rows.append(row); commands.append('# '+name+'\n'+' '.join(shlex.quote(x) for x in cmd))
        if not is_cpu and (freq['/sys/class/devfreq/fdab0000.npu/cur_freq']!=str(a.expected_npu_freq) or after.get('/sys/class/devfreq/fdab0000.npu/target_freq')!=str(a.expected_npu_freq)):
            raise RuntimeError(f'{name}: NPU frequency changed during run')
    manifest={'schema':'rocknpu-candidate-matrix-v1','board':a.board,'argv':sys.argv,'model_sha256':sha256(a.model),'plugin_sha256':sha256(a.plugin),'bench_sha256':sha256(a.bench),'sidecar':str(a.sidecar),'expected_npu_freq':a.expected_npu_freq,'initial_snapshot':initial,'tokens':a.tokens,'reps':a.reps,'rows':rows}
    write(a.output/'candidate-matrix.json',manifest); (a.output/'commands.sh').write_text('#!/bin/sh\nset -eu\n'+'\n'.join(commands)+'\n'); (a.output/'commands.sh').chmod(0o755)
    files={str(p.relative_to(a.output)):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in sorted(a.output.rglob('*')) if p.is_file()};write(a.output/'artifact-manifest.json',{'board':a.board,'files':files})
    print(json.dumps({'board':a.board,'candidates':len(rows),'failed':[r['name'] for r in rows if r['exit_code'] or not r['json']],'files':len(files)}))
if __name__=='__main__':main()
