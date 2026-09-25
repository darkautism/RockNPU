#!/usr/bin/env python3
"""GPU/NPU llama-server quality oracle for the fixed RK3588 workload."""
import argparse, hashlib, json, os, platform, re, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path


def snap():
    r={'timestamp_unix':time.time(),'machine':platform.machine()}
    ps=[]
    for p in Path('/sys/devices/system/cpu/cpufreq').glob('policy*'):
        for f in ('scaling_governor','scaling_cur_freq','cpuinfo_cur_freq','related_cpus'):ps.append(p/f)
    for n in Path('/sys/class/devfreq').glob('*'):
        for f in ('governor','cur_freq','target_freq','min_freq','max_freq','available_frequencies'):ps.append(n/f)
    ps += list(Path('/sys/class/thermal').glob('thermal_zone*/temp'))
    for p in ps:
        try:r[str(p)]=p.read_text().strip()
        except OSError as e:r[str(p)]=f'<unavailable: {e}>'
    return r

def write(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(s,node,expected):
    for f in ('cur_freq','target_freq'):
        if s.get(f'/sys/class/devfreq/{node}/{f}')!=str(expected):raise RuntimeError(f'{node}/{f} not {expected}')

def start(name,server,model,port,device,plugin,sidecar,out,trace=False):
    e=os.environ.copy()
    for k in list(e):
        if k.startswith('ROCKNPU_') or k in ('GGML_BACKEND_PATH','GGML_SCHED_DEBUG'):e.pop(k,None)
    e['LD_LIBRARY_PATH']=str(Path(server).parent)+(':'+e['LD_LIBRARY_PATH'] if e.get('LD_LIBRARY_PATH') else '')
    cmd=[str(server),'--model',str(model),'--host','127.0.0.1','--port',str(port),'--no-webui','--offline','--ctx-size','2048','--threads','4','--flash-attn','off','--load-mode','none','--device',device,'--no-jinja','--chat-template','chatml','--log-verbosity','4','--no-log-prefix','--no-log-timestamps','--keep','4']
    if device=='ROCKNPU0':
        e.update({'GGML_BACKEND_PATH':str(plugin),'ROCKNPU_W8_SIDECAR_DIR':str(sidecar),'ROCKNPU_W8_DIRECT_SUBMIT':'1','ROCKNPU_EXPERIMENT_DIRECT_SCRATCH':'1','ROCKNPU_PREFILL_CACHE':'1','ROCKNPU_DECODE':'1','ROCKNPU_DISPATCH_SUMMARY':'1'})
        if trace:e['ROCKNPU_GGML_TRACE']='1'
    out=Path(out);out.mkdir(parents=True,exist_ok=True);o=(out/'stdout.log').open('wb');er=(out/'stderr.log').open('wb');p=subprocess.Popen(cmd,env=e,stdout=o,stderr=er,start_new_session=True);o.close();er.close()
    return {'name':name,'pid':p.pid,'process':p,'command':cmd,'env':{k:v for k,v in sorted(e.items()) if k.startswith('ROCKNPU_') or k in ('GGML_BACKEND_PATH','LD_LIBRARY_PATH')},'stdout':str(out/'stdout.log'),'stderr':str(out/'stderr.log')}

def stop(x):
    p=x['process']
    if p.poll() is None:
        p.terminate()
        try:p.wait(10)
        except subprocess.TimeoutExpired:p.kill();p.wait(10)

def wait(port,p):
    end=time.monotonic()+180
    while time.monotonic()<end:
        if p.poll() is not None:raise RuntimeError('server exited '+str(p.returncode))
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as r:return r.status
        except Exception:time.sleep(.25)
    raise TimeoutError(port)

def request(port,payload,stem):
    stem=Path(stem);stem.parent.mkdir(parents=True,exist_ok=True);write(stem.with_suffix('.request.json'),payload);rawreq=json.dumps(payload).encode();q=urllib.request.Request(f'http://127.0.0.1:{port}/completion',data=rawreq,headers={'Content-Type':'application/json'},method='POST');before=snap();t=time.monotonic()
    try:
        with urllib.request.urlopen(q,timeout=300) as r:raw=r.read();status=r.status
    except urllib.error.HTTPError as e:raw=e.read();status=e.code
    wall=time.monotonic()-t;rawp=stem.with_suffix('.response.raw');rawp.write_bytes(raw)
    try:d=json.loads(raw);write(stem.with_suffix('.response.json'),d)
    except Exception:d=None
    rec={'request':str(stem.with_suffix('.request.json')),'response_raw':str(rawp),'response_json':str(stem.with_suffix('.response.json')) if d else None,'status':status,'wall_s':wall,'before':before,'after':snap(),'raw_sha256':hashlib.sha256(raw).hexdigest(),'response':d}
    if isinstance(d,dict):rec['text']=d.get('content','');rec['tokens_evaluated']=d.get('tokens_evaluated');rec['stop_type']=d.get('stop_type')
    write(stem.with_suffix('.record.json'),rec);return rec

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--server',type=Path,required=True);ap.add_argument('--model',type=Path,required=True);ap.add_argument('--plugin',type=Path,required=True);ap.add_argument('--sidecar',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--board',required=True);ap.add_argument('--gpu-port',type=int,default=11540);ap.add_argument('--npu-port',type=int,default=11541);ap.add_argument('--oracle-port',type=int,default=11542);ap.add_argument('--blocks',type=int,default=3);ap.add_argument('--tokens',type=int,default=8);a=ap.parse_args()
    for n in ('server','model','plugin','sidecar'):setattr(a,n,getattr(a,n).resolve(strict=True))
    a.output.mkdir(parents=True,exist_ok=False);initial=snap();check(initial,'fdab0000.npu',700000000);check(initial,'fb000000.gpu',1000000000);payload={'prompt':'Paris','n_predict':a.tokens,'temperature':0,'top_k':1,'seed':1234,'stream':False,'cache_prompt':False};services=[]
    try:
        gpu=start('gpu',a.server,a.model,a.gpu_port,'Vulkan0',a.plugin,a.sidecar,a.output/'gpu-server');services.append(gpu);npu=start('npu',a.server,a.model,a.npu_port,'ROCKNPU0',a.plugin,a.sidecar,a.output/'npu-server',True);services.append(npu);oracle=start('oracle',a.server,a.model,a.oracle_port,'none',a.plugin,a.sidecar,a.output/'oracle-server');services.append(oracle)
        for x in (gpu,npu,oracle):wait({'gpu':a.gpu_port,'npu':a.npu_port,'oracle':a.oracle_port}[x['name']],x['process'])
        for name,port in (('gpu',a.gpu_port),('npu',a.npu_port),('oracle',a.oracle_port)):request(port,payload,a.output/'warmup'/name)
        oracle_rec=request(a.oracle_port,payload,a.output/'oracle'/'request');blocks=[]
        for b in range(a.blocks):
            order=('npu','gpu','npu','gpu') if b%2==0 else ('gpu','npu','gpu','npu');req=[]
            for i,k in enumerate(order,1):req.append({'kind':k,'order':i,'record':request(a.npu_port if k=='npu' else a.gpu_port,payload,a.output/'blocks'/f'block-{b+1:02d}'/f'{k}-{i:02d}')})
            blocks.append({'block':b+1,'order':list(order),'requests':req})
        nt=[x['record'].get('text') for b in blocks for x in b['requests'] if x['kind']=='npu'];gt=[x['record'].get('text') for b in blocks for x in b['requests'] if x['kind']=='gpu'];ot=oracle_rec.get('text');nerr=Path(npu['stderr']).read_text(errors='replace');gerr=Path(gpu['stderr']).read_text(errors='replace')
        result={'board':a.board,'quality_gate':{'oracle':ot,'gpu_responses':gt,'npu_responses':nt,'npu_matches_gpu':all(x==y for x,y in zip(nt,gt)),'npu_matches_oracle':all(x==ot for x in nt),'gpu_matches_oracle':all(x==ot for x in gt),'status':'passed' if all(x==y for x,y in zip(nt,gt)) and all(x==ot for x in nt) else 'failed'},'dispatch':{'npu_trace_lines':nerr.count('ROCKNPU GGML TRACE'),'npu_w8a8_m1_lines':nerr.count('path=w8a8_m1'),'npu_stderr_sha256':hashlib.sha256(nerr.encode()).hexdigest(),'gpu_stderr_sha256':hashlib.sha256(gerr.encode()).hexdigest()},'blocks':blocks,'oracle':oracle_rec}
        write(a.output/'result.json',result);files={str(p.relative_to(a.output)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(a.output.rglob('*')) if p.is_file()};write(a.output/'artifact-manifest.json',{'board':a.board,'files':files});print(json.dumps({'board':a.board,'quality':result['quality_gate']['status'],'dispatch':result['dispatch'],'files':len(files)}))
    finally:
        for x in reversed(services):stop(x)
if __name__=='__main__':main()
