#!/usr/bin/env python3
"""Independent CPU oracle and NPU quality gate for the stock llama-server frontend."""
import argparse, hashlib, json, os, platform, shlex, subprocess, sys, time
from pathlib import Path
import urllib.error, urllib.request


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def snap():
    r={'timestamp_unix':time.time(),'machine':platform.machine(),'uname':' '.join(platform.uname())}
    paths=[]
    for p in sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*')):
        for f in ('scaling_governor','scaling_cur_freq','cpuinfo_cur_freq','cpuinfo_min_freq','cpuinfo_max_freq','related_cpus'):
            paths.append(p/f)
    for n in sorted(Path('/sys/class/devfreq').glob('*')):
        for f in ('governor','cur_freq','target_freq','min_freq','max_freq','available_frequencies'): paths.append(n/f)
    paths += sorted(Path('/sys/class/thermal').glob('thermal_zone*/temp'))
    for p in paths:
        try:r[str(p)]=p.read_text().strip()
        except OSError as e:r[str(p)]=f'<unavailable: {e}>'
    return r

def envsafe(e):
    return {k:('<redacted>' if any(x in k.upper() for x in ('TOKEN','PASSWORD','SECRET','API_KEY')) else v) for k,v in sorted(e.items())}

def wj(p,v): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')

def wait(url,proc,limit=180):
    end=time.monotonic()+limit; last=None
    while time.monotonic()<end:
        if proc.poll() is not None: raise RuntimeError(f'server exited {proc.returncode}')
        try:
            with urllib.request.urlopen(url,timeout=2) as r:return r.status,r.read()
        except Exception as e:last=e; time.sleep(.25)
    raise TimeoutError(f'{url}: {last}')

def start(name,server,model,port,device,plugin,sidecar,out,trace=False):
    e=os.environ.copy(); e.update({'LD_LIBRARY_PATH':str(plugin.parent)})
    e.pop('ROCKNPU_GGML_TRACE',None)
    cmd=[str(server),'--model',str(model),'--host','127.0.0.1','--port',str(port),'--no-webui','--offline','--ctx-size','2048','--threads','4','--flash-attn','off','--load-mode','none','--device',device,'--no-jinja','--chat-template','chatml','--log-verbosity','4','--no-log-prefix','--no-log-timestamps','--keep','4']
    if device=='ROCKNPU0':
        e.update({'GGML_BACKEND_PATH':str(plugin),'ROCKNPU_W8_SIDECAR_DIR':str(sidecar),'ROCKNPU_W8_DIRECT_SUBMIT':'1','ROCKNPU_EXPERIMENT_DIRECT_SCRATCH':'1','ROCKNPU_PREFILL_CACHE':'1','ROCKNPU_DECODE':'1','ROCKNPU_DISPATCH_SUMMARY':'1'})
        if trace:e['ROCKNPU_GGML_TRACE']='1'
    o=Path(out).with_suffix('.stdout.log').open('wb'); er=Path(out).with_suffix('.stderr.log').open('wb')
    p=subprocess.Popen(cmd,env=e,stdout=o,stderr=er,start_new_session=True); o.close(); er.close()
    return {'name':name,'process':p,'pid':p.pid,'command':cmd,'environment':envsafe(e),'stdout':str(Path(out).with_suffix('.stdout.log')),'stderr':str(Path(out).with_suffix('.stderr.log'))}

def stop(x):
    p=x['process']
    if p.poll() is None:
        p.terminate()
        try:p.wait(10)
        except subprocess.TimeoutExpired:p.kill();p.wait(10)

def request(port,payload,stem):
    stem=Path(stem); stem.parent.mkdir(parents=True,exist_ok=True); wj(stem.with_suffix('.request.json'),payload)
    body=json.dumps(payload).encode(); req=urllib.request.Request(f'http://127.0.0.1:{port}/completion',data=body,headers={'Content-Type':'application/json'},method='POST'); before=snap(); t=time.monotonic()
    try:
        with urllib.request.urlopen(req,timeout=300) as q: raw=q.read(); status=q.status
    except urllib.error.HTTPError as e:raw=e.read();status=e.code
    wall=time.monotonic()-t; rawp=stem.with_suffix('.response.raw'); rawp.write_bytes(raw)
    try: parsed=json.loads(raw);wj(stem.with_suffix('.response.json'),parsed)
    except Exception:parsed=None
    rec={'request_file':str(stem.with_suffix('.request.json')),'response_raw_file':str(rawp),'response_json_file':str(stem.with_suffix('.response.json')) if parsed else None,'status':status,'wall_s':wall,'before':before,'after':snap(),'raw_sha256':hashlib.sha256(raw).hexdigest(),'response':parsed}
    if isinstance(parsed,dict):
        rec['response_text']=parsed.get('content',''); rec['tokens_evaluated']=parsed.get('tokens_evaluated'); rec['stop_type']=parsed.get('stop_type')
    wj(stem.with_suffix('.record.json'),rec); return rec

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--server',type=Path,required=True);ap.add_argument('--model',type=Path,required=True);ap.add_argument('--plugin',type=Path,required=True);ap.add_argument('--sidecar',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--board',required=True);ap.add_argument('--npu-port',type=int,default=11540);ap.add_argument('--cpu-port',type=int,default=11541);ap.add_argument('--oracle-port',type=int,default=11542);ap.add_argument('--prompt',default='Paris');ap.add_argument('--tokens',type=int,default=8);ap.add_argument('--blocks',type=int,default=3);a=ap.parse_args()
    for n in ('server','model','plugin','sidecar'):setattr(a,n,getattr(a,n).resolve(strict=True))
    a.output.mkdir(parents=True,exist_ok=False); initial=snap(); payload={'prompt':a.prompt,'n_predict':a.tokens,'temperature':0,'top_k':1,'seed':1234,'stream':False,'cache_prompt':False}
    manifest={'board':a.board,'argv':sys.argv,'server_hash':sha256(a.server),'model_hash':sha256(a.model),'plugin_hash':sha256(a.plugin),'workload':payload,'initial_snapshot':initial,'environment':envsafe(os.environ)}
    services=[]
    try:
        npu=start('npu',a.server,a.model,a.npu_port,'ROCKNPU0',a.plugin,a.sidecar,a.output/'npu-server',True); services.append(npu)
        cpu=start('cpu',a.server,a.model,a.cpu_port,'none',a.plugin,a.sidecar,a.output/'cpu-server',False); services.append(cpu)
        wait(f'http://127.0.0.1:{a.npu_port}/health',npu['process']);wait(f'http://127.0.0.1:{a.cpu_port}/health',cpu['process'])
        for kind,port in (('npu',a.npu_port),('cpu',a.cpu_port)):request(port,payload,a.output/'warmup'/kind)
        oracle=start('oracle',a.server,a.model,a.oracle_port,'none',a.plugin,a.sidecar,a.output/'oracle-server',False);services.append(oracle);wait(f'http://127.0.0.1:{a.oracle_port}/health',oracle['process']);oracle_rec=request(a.oracle_port,payload,a.output/'oracle'/'request')
        blocks=[]
        for b in range(a.blocks):
            order=('npu','cpu','npu','cpu') if b%2==0 else ('cpu','npu','cpu','npu'); reqs=[]
            for i,k in enumerate(order,1):reqs.append({'kind':k,'order':i,'record':request(a.npu_port if k=='npu' else a.cpu_port,payload,a.output/'blocks'/f'block-{b+1:02d}'/f'{k}-{i:02d}')})
            blocks.append({'block':b+1,'order':list(order),'requests':reqs})
        npu_text=Path(npu['stderr']).read_text(errors='replace');cpu_text=Path(cpu['stderr']).read_text(errors='replace'); nr=[x['record']['response_text'] for b in blocks for x in b['requests'] if x['kind']=='npu'];cr=[x['record']['response_text'] for b in blocks for x in b['requests'] if x['kind']=='cpu'];oracle_text=oracle_rec.get('response_text')
        result={'board':a.board,'manifest':manifest,'blocks':blocks,'quality_gate':{'status':'passed' if nr and all(x==oracle_text for x in nr) else 'failed','cpu_oracle_response':oracle_text,'npu_responses':nr,'cpu_responses':cr,'npu_matches_oracle':bool(nr) and all(x==oracle_text for x in nr),'cpu_matches_oracle':bool(cr) and all(x==oracle_text for x in cr)},'npu_trace':{'trace_lines':npu_text.count('ROCKNPU GGML TRACE'),'w8a8_m1_lines':npu_text.count('path=w8a8_m1'),'host_lines':npu_text.count('ROCKNPU_HOST'),'dispatch_summary_lines':npu_text.count('ROCKNPU GGML TRACE summary'),'stderr_sha256':hashlib.sha256(npu_text.encode()).hexdigest()},'cpu_log_sha256':hashlib.sha256(cpu_text.encode()).hexdigest()}
        wj(a.output/'result.json',result)
        files={str(p.relative_to(a.output)):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in sorted(a.output.rglob('*')) if p.is_file()};wj(a.output/'artifact-manifest.json',{'board':a.board,'files':files})
        print(json.dumps({'quality':result['quality_gate']['status'],'trace':result['npu_trace'],'files':len(files)}))
    finally:
        for x in reversed(services):stop(x)
if __name__=='__main__':main()
