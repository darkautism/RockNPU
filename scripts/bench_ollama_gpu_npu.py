#!/usr/bin/env python3
"""Fixed-condition stock Ollama GPU/NPU ABBA benchmark with CPU quality oracle."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
from bench_ollama_cpu_npu import snapshot, safe_env, write_json, sha256, api_get, generate, start_service, stop_service


def env_for(args, port, kind):
    e=os.environ.copy()
    for k in list(e):
        if k.startswith('ROCKNPU_') or k in ('GGML_BACKEND_PATH','GGML_SCHED_DEBUG'):e.pop(k,None)
    e.update({'OLLAMA_HOST':f'127.0.0.1:{port}','OLLAMA_MODELS':str(args.models),'OLLAMA_NUM_GPU':'0' if kind=='oracle' else '999','OLLAMA_KEEP_ALIVE':'30m','OLLAMA_VULKAN':'true' if kind=='gpu' else 'false','LLAMA_ARG_DEVICE':'none' if kind=='oracle' else ('Vulkan0' if kind=='gpu' else 'ROCKNPU0')})
    ld=str(args.lib_dir)
    if kind=='gpu':
        e['GGML_BACKEND_PATH']=str(args.gpu_backend);ld=str(args.gpu_backend.parent)+':'+ld
    elif kind=='npu':
        e.update({'GGML_BACKEND_PATH':str(args.plugin),'ROCKNPU_W8_SIDECAR_DIR':str(args.sidecar),'ROCKNPU_W8_DIRECT_SUBMIT':'1','ROCKNPU_EXPERIMENT_DIRECT_SCRATCH':'1','ROCKNPU_PREFILL_CACHE':'1','ROCKNPU_DECODE':'1','ROCKNPU_DISPATCH_SUMMARY':'1'})
        if args.trace:e['ROCKNPU_GGML_TRACE']='1'
        ld=str(args.plugin.parent)+':'+ld
    e['LD_LIBRARY_PATH']=ld+((':'+e['LD_LIBRARY_PATH']) if e.get('LD_LIBRARY_PATH') else '')
    return e

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ollama',type=Path,required=True);ap.add_argument('--lib-dir',type=Path,required=True);ap.add_argument('--models',type=Path,required=True);ap.add_argument('--model',required=True);ap.add_argument('--model-blob',type=Path,required=True);ap.add_argument('--plugin',type=Path,required=True);ap.add_argument('--sidecar',type=Path,required=True);ap.add_argument('--gpu-backend',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--board',required=True);ap.add_argument('--gpu-port',type=int,default=11460);ap.add_argument('--npu-port',type=int,default=11461);ap.add_argument('--oracle-port',type=int,default=11462);ap.add_argument('--blocks',type=int,default=3);ap.add_argument('--warmups',type=int,default=2);ap.add_argument('--tokens',type=int,default=8);ap.add_argument('--trace',action='store_true');a=ap.parse_args()
    for n in ('ollama','lib_dir','models','model_blob','plugin','sidecar','gpu_backend'):setattr(a,n,getattr(a,n).resolve(strict=True))
    a.output.mkdir(parents=True,exist_ok=False);initial=snapshot();payload={'model':a.model,'prompt':'Paris','stream':False,'options':{'temperature':0,'num_predict':a.tokens,'num_ctx':2048,'num_thread':4}}
    services=[]
    try:
        ge=env_for(a,a.gpu_port,'gpu');ne=env_for(a,a.npu_port,'npu');oe=env_for(a,a.oracle_port,'oracle')
        gpu=start_service('gpu',a.ollama,ge,a.output/'gpu-server');npu=start_service('npu',a.ollama,ne,a.output/'npu-server');oracle=start_service('oracle',a.ollama,oe,a.output/'oracle-server');services=[gpu,npu,oracle]
        import urllib.request
        for svc,port in ((gpu,a.gpu_port),(npu,a.npu_port),(oracle,a.oracle_port)):
            end=time.monotonic()+180
            while time.monotonic()<end:
                if svc['process'].poll() is not None:raise RuntimeError(f"{svc['name']} exited")
                try:urllib.request.urlopen(f'http://127.0.0.1:{port}/api/tags',timeout=2).close();break
                except Exception:time.sleep(.25)
            else:raise TimeoutError(svc['name'])
            api_get(port,'/api/ps',a.output/f"{svc['name']}-ps-before.json")
        for kind,port in (('gpu',a.gpu_port),('npu',a.npu_port),('oracle',a.oracle_port)):
            for i in range(a.warmups):generate(port,payload,a.output/'warmup'/f'{kind}-{i+1:02d}')
        oracle_rec=generate(a.oracle_port,payload,a.output/'oracle'/'request');blocks=[]
        for b in range(a.blocks):
            order=('npu','gpu','npu','gpu') if b%2==0 else ('gpu','npu','gpu','npu');req=[]
            for i,k in enumerate(order,1):req.append({'kind':k,'order':i,'record':generate(a.npu_port if k=='npu' else a.gpu_port,payload,a.output/'blocks'/f'block-{b+1:02d}'/f'{k}-{i:02d}')})
            blocks.append({'block':b+1,'order':list(order),'requests':req})
        for svc,port in ((gpu,a.gpu_port),(npu,a.npu_port),(oracle,a.oracle_port)):api_get(port,'/api/ps',a.output/f"{svc['name']}-ps-after.json")
        nt=[x['record'].get('response_text') for b in blocks for x in b['requests'] if x['kind']=='npu'];gt=[x['record'].get('response_text') for b in blocks for x in b['requests'] if x['kind']=='gpu'];ot=oracle_rec.get('response_text');nerr=Path(npu['stderr']).read_text(errors='replace');gerr=Path(gpu['stderr']).read_text(errors='replace')
        samples={k:[x['record'].get('tok_s') for b in blocks for x in b['requests'] if x['kind']==k] for k in ('npu','gpu')}
        valid={k:[v for v in vals if isinstance(v,(int,float)) and v>0] for k,vals in samples.items()}
        invalid={k:len(vals)-len(valid[k]) for k,vals in samples.items()}
        means={k:(sum(valid[k])/len(valid[k]) if valid[k] else None) for k in ('npu','gpu')}
        ratio=(means['npu']/means['gpu']) if means['npu'] is not None and means['gpu'] else None
        status='passed' if not any(invalid.values()) and all(x==y for x,y in zip(nt,gt)) and all(x==ot for x in nt) else 'failed'
        result={'board':a.board,'warmups':a.warmups,'instrumentation':{'trace':a.trace},'quality_gate':{'oracle':ot,'npu_responses':nt,'gpu_responses':gt,'npu_matches_gpu':all(x==y for x,y in zip(nt,gt)),'npu_matches_oracle':all(x==ot for x in nt),'gpu_matches_oracle':all(x==ot for x in gt),'invalid_request_counts':invalid,'status':status},'throughput':{'npu_tok_s':means['npu'],'gpu_tok_s':means['gpu'],'npu_over_gpu':ratio},'dispatch':{'npu_trace_lines':nerr.count('ROCKNPU GGML TRACE'),'npu_w8a8_m1_lines':nerr.count('path=w8a8_m1'),'npu_host_lines':nerr.count('ROCKNPU_HOST'),'npu_stderr_sha256':hashlib.sha256(nerr.encode()).hexdigest(),'gpu_stderr_sha256':hashlib.sha256(gerr.encode()).hexdigest()},'blocks':blocks,'oracle':oracle_rec}
        write_json(a.output/'result.json',result);files={str(p.relative_to(a.output)):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in sorted(a.output.rglob('*')) if p.is_file()};write_json(a.output/'artifact-manifest.json',{'board':a.board,'files':files});write_json(a.output/'run-manifest.json',{'board':a.board,'argv':sys.argv,'hashes':{'ollama':sha256(a.ollama),'model':sha256(a.model_blob),'plugin':sha256(a.plugin),'gpu_backend':sha256(a.gpu_backend)},'payload':payload,'initial_snapshot':initial,'service_envs':{'gpu':safe_env(ge),'npu':safe_env(ne),'oracle':safe_env(oe)},'blocks':a.blocks,'warmups':a.warmups});print(json.dumps({'board':a.board,'quality':result['quality_gate']['status'],'throughput':result['throughput'],'dispatch':result['dispatch']}))
    finally:
        for x in reversed(services):stop_service(x)
if __name__=='__main__':main()
