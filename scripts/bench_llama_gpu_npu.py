#!/usr/bin/env python3
"""Fixed-condition GPU/NPU ABBA benchmark for stock llama.cpp on RK3588."""
import argparse, hashlib, json, os, platform, re, subprocess, sys, time
from pathlib import Path


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def snap():
    r={'timestamp_unix':time.time(),'machine':platform.machine()}
    paths=[]
    for p in Path('/sys/devices/system/cpu/cpufreq').glob('policy*'):
        for f in ('scaling_governor','scaling_cur_freq','cpuinfo_cur_freq','cpuinfo_min_freq','cpuinfo_max_freq','related_cpus'):
            paths.append(p/f)
    for n in Path('/sys/class/devfreq').glob('*'):
        for f in ('governor','cur_freq','target_freq','min_freq','max_freq','available_frequencies'):paths.append(n/f)
    paths += list(Path('/sys/class/thermal').glob('thermal_zone*/temp'))
    for p in paths:
        try:r[str(p)]=p.read_text().strip()
        except OSError as e:r[str(p)]=f'<unavailable: {e}>'
    return r

def write(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')

def check(s,node,expected):
    for f in ('cur_freq','target_freq'):
        got=s.get(f'/sys/class/devfreq/{node}/{f}')
        if got!=str(expected):raise RuntimeError(f'{node}/{f}={got}, expected {expected}')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bench',type=Path,required=True);ap.add_argument('--plugin',type=Path,required=True);ap.add_argument('--model',type=Path,required=True);ap.add_argument('--sidecar',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--board',required=True);ap.add_argument('--blocks',type=int,default=3);ap.add_argument('--tokens',type=int,default=32);ap.add_argument('--prompt',type=int,default=0);ap.add_argument('--npu-freq',type=int,default=700000000);ap.add_argument('--gpu-freq',type=int,default=1000000000);a=ap.parse_args()
    for n in ('bench','plugin','model','sidecar'):setattr(a,n,getattr(a,n).resolve(strict=True))
    if a.blocks<3:ap.error('at least three blocks required')
    a.output.mkdir(parents=True,exist_ok=False); initial=snap(); check(initial,'fdab0000.npu',a.npu_freq);check(initial,'fb000000.gpu',a.gpu_freq)
    base={k:v for k,v in os.environ.items() if not k.startswith('ROCKNPU_') and k not in ('GGML_BACKEND_PATH','GGML_SCHED_DEBUG')}
    rows=[]
    for i,kind in enumerate(['gpu','npu','npu','gpu']*a.blocks,1):
        env=base.copy();env.update({'LD_LIBRARY_PATH':str(a.bench.parent)+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')})
        if kind=='npu':env.update({'GGML_BACKEND_PATH':str(a.plugin),'ROCKNPU_W8_SIDECAR_DIR':str(a.sidecar),'ROCKNPU_W8_DIRECT_SUBMIT':'1','ROCKNPU_EXPERIMENT_DIRECT_SCRATCH':'1','ROCKNPU_DECODE':'1','ROCKNPU_PREFILL_CACHE':'1','ROCKNPU_DISPATCH_SUMMARY':'1'})
        cmd=['taskset','-c','4-7',str(a.bench),'-m',str(a.model),'-p',str(a.prompt),'-n',str(a.tokens),'-r','1','-t','4','-fa','on','-dev','Vulkan0' if kind=='gpu' else 'ROCKNPU0','-nopo','0','-o','json']
        before=snap();check(before,'fdab0000.npu',a.npu_freq);check(before,'fb000000.gpu',a.gpu_freq);p=a.output/f'{i:02d}-{kind}';start=time.monotonic()
        with p.with_suffix('.stdout').open('wb') as out,p.with_suffix('.stderr').open('wb') as err:proc=subprocess.run(cmd,env=env,stdout=out,stderr=err,timeout=1800)
        after=snap();check(after,'fdab0000.npu',a.npu_freq);check(after,'fb000000.gpu',a.gpu_freq)
        raw=p.with_suffix('.stdout').read_text(errors='replace');err=p.with_suffix('.stderr').read_text(errors='replace');
        try:data=json.loads(raw)
        except Exception:data=None
        if proc.returncode:raise RuntimeError(f'{kind} failed {proc.returncode}; {p}.stderr')
        if not data or len(data)!=1:raise RuntimeError(f'invalid JSON {p}.stdout')
        row=data[0]
        if kind=='gpu' and (row.get('backends')!='Vulkan' or row.get('devices')!='Vulkan0'):raise RuntimeError('GPU backend/device mismatch')
        if kind=='npu' and ('ROCKNPU' not in row.get('backends','') or row.get('devices')!='ROCKNPU0'):raise RuntimeError('NPU backend/device mismatch')
        disp=0
        if kind=='npu':
            for m in re.finditer(r'ROCKNPU GGML TRACE summary .*?w8a8_m1_mul_mat=(\d+)',err):disp+=int(m.group(1))
            if disp<=0:raise RuntimeError(f'zero NPU W8 dispatch: {p}.stderr')
        rec={'index':i,'kind':kind,'command':cmd,'environment':{k:v for k,v in sorted(env.items()) if k.startswith('ROCKNPU_') or k in ('GGML_BACKEND_PATH','LD_LIBRARY_PATH')},'before':before,'after':after,'wall_s':time.monotonic()-start,'exit_code':proc.returncode,'result':row,'npu_dispatch_count':disp,'stdout_sha256':hashlib.sha256(raw.encode()).hexdigest(),'stderr_sha256':hashlib.sha256(err.encode()).hexdigest()};write(p.with_suffix('.meta.json'),rec);rows.append(rec)
    def mean(k):return sum(x['result']['avg_ts'] for x in rows if x['kind']==k)/sum(x['kind']==k for x in rows)
    summary={'mean_tok_s':{'gpu':mean('gpu'),'npu':mean('npu')},'npu_over_gpu_tok_s':mean('npu')/mean('gpu'),'npu_exceeds_gpu':mean('npu')>=1.05*mean('gpu')}
    for j in range(a.blocks):
        b=rows[j*4:(j+1)*4];g=[x for x in b if x['kind']=='gpu'];n=[x for x in b if x['kind']=='npu'];summary.setdefault('block_npu_over_gpu_tok_s',[]).append((sum(x['result']['avg_ts'] for x in n)/len(n))/(sum(x['result']['avg_ts'] for x in g)/len(g)))
    write(a.output/'summary.json',summary);manifest={'schema':'rocknpu-llama-gpu-npu-v1','board':a.board,'argv':sys.argv,'settings':vars(a)|{k:str(getattr(a,k)) for k in ('bench','plugin','model','sidecar','output')},'hashes':{'model':sha256(a.model),'bench':sha256(a.bench),'plugin':sha256(a.plugin),'sidecar_manifest':sha256(a.sidecar/'manifest.json')},'initial_snapshot':initial,'rows':rows,'summary':summary};write(a.output/'run-manifest.json',manifest)
    files={str(p.relative_to(a.output)):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in sorted(a.output.rglob('*')) if p.is_file()};write(a.output/'artifact-manifest.json',{'board':a.board,'files':files});print(json.dumps({'board':a.board,'summary':summary,'files':len(files)}))
if __name__=='__main__':main()
