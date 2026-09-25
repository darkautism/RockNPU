#!/usr/bin/env python3
"""Repeated GPU/NPU candidate A/B for adapter optimization decisions."""
import argparse,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
from bench_llama_cpu_npu import snapshot,sha256
CANDIDATES={'baseline':{},'direct-scratch':{'ROCKNPU_W8_DIRECT_SUBMIT':'1','ROCKNPU_EXPERIMENT_DIRECT_SCRATCH':'1'}}
def wj(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def check(s):
 for n,v in [('fdab0000.npu','700000000'),('fb000000.gpu','1000000000')]:
  for f in ('cur_freq','target_freq'):
   if s.get(f'/sys/class/devfreq/{n}/{f}')!=v:raise RuntimeError(f'{n}/{f} changed')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bench',type=Path,required=True);ap.add_argument('--plugin',type=Path,required=True);ap.add_argument('--model',type=Path,required=True);ap.add_argument('--sidecar',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--board',required=True);ap.add_argument('--blocks',type=int,default=3);a=ap.parse_args()
 for n in ('bench','plugin','model','sidecar'):setattr(a,n,getattr(a,n).resolve(strict=True))
 a.output.mkdir(parents=True,exist_ok=False);initial=snapshot();check(initial);base={k:v for k,v in os.environ.items() if not k.startswith('ROCKNPU_') and k not in ('GGML_BACKEND_PATH','GGML_SCHED_DEBUG')};rows=[]
 for cand,extra in CANDIDATES.items():
  for block in range(a.blocks):
   for kind in ('gpu','npu','npu','gpu'):
    e=base.copy(); e['LD_LIBRARY_PATH']=str(a.bench.parent)+(':'+e['LD_LIBRARY_PATH'] if e.get('LD_LIBRARY_PATH') else '')
    if kind=='npu':e.update({'GGML_BACKEND_PATH':str(a.plugin),'ROCKNPU_W8_SIDECAR_DIR':str(a.sidecar),'ROCKNPU_DECODE':'1','ROCKNPU_PREFILL_CACHE':'1','ROCKNPU_DISPATCH_SUMMARY':'1',**extra})
    else:e.pop('GGML_BACKEND_PATH',None)
    before=snapshot();check(before);cmd=['taskset','-c','4-7',str(a.bench),'-m',str(a.model),'-p','0','-n','8','-r','1','-t','4','-fa','on','-dev','Vulkan0' if kind=='gpu' else 'ROCKNPU0','-nopo','0','-o','json'];p=a.output/f'{cand}-b{block+1}-{kind}-{len(rows)+1:02d}';start=time.monotonic()
    with p.with_suffix('.stdout').open('wb') as out,p.with_suffix('.stderr').open('wb') as err:proc=subprocess.run(cmd,env=e,stdout=out,stderr=err,timeout=900)
    raw=p.with_suffix('.stdout').read_text(errors='replace');err=p.with_suffix('.stderr').read_text(errors='replace');
    try:data=json.loads(raw)
    except Exception:data=None
    if proc.returncode or not data:raise RuntimeError(f'{cand}/{kind} failed {proc.returncode}: {p}.stderr')
    row=data[0]
    if kind=='gpu' and (row.get('backends')!='Vulkan' or row.get('devices')!='Vulkan0'):raise RuntimeError('GPU device mismatch')
    if kind=='npu' and ('ROCKNPU' not in row.get('backends','') or row.get('devices')!='ROCKNPU0'):raise RuntimeError('NPU device mismatch')
    disp=sum(int(x) for x in re.findall(r'w8a8_m1_mul_mat=(\d+)',err)) if kind=='npu' else 0
    if kind=='npu' and disp<=0:raise RuntimeError('zero NPU dispatch')
    rec={'candidate':cand,'block':block+1,'kind':kind,'command':cmd,'flags':extra,'before':before,'after':snapshot(),'result':row,'dispatch':disp,'stdout_sha256':hashlib.sha256(raw.encode()).hexdigest(),'stderr_sha256':hashlib.sha256(err.encode()).hexdigest()};wj(p.with_suffix('.record.json'),rec);rows.append(rec)
 means={c:{k:sum(x['result']['avg_ts'] for x in rows if x['candidate']==c and x['kind']==k)/sum(x['candidate']==c and x['kind']==k for x in rows) for k in ('gpu','npu')} for c in CANDIDATES};summary={'means':means,'npu_over_gpu':{c:means[c]['npu']/means[c]['gpu'] for c in CANDIDATES}}
 wj(a.output/'summary.json',summary);wj(a.output/'run-manifest.json',{'board':a.board,'argv':sys.argv,'initial_snapshot':initial,'model':sha256(a.model),'plugin':sha256(a.plugin),'rows':rows,'summary':summary});files={str(p.relative_to(a.output)):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in sorted(a.output.rglob('*')) if p.is_file()};wj(a.output/'artifact-manifest.json',{'board':a.board,'files':files});print(json.dumps({'board':a.board,'summary':summary,'files':len(files)}))
if __name__=='__main__':main()
