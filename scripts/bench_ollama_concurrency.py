#!/usr/bin/env python3
"""Raw c8/c16 Ollama concurrency diagnostics for blocker evidence."""
import argparse, hashlib, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from bench_ollama_cpu_npu import snapshot, safe_env, write_json, start_service, stop_service, generate, api_get, service_env, sha256


def one_request(port, payload, stem):
    try:
        return {"status": "ok", "record": generate(port, payload, stem)}
    except Exception as e:
        write_json(Path(str(stem) + ".error.json"), {"error": repr(e)})
        return {"status": "error", "error": repr(e)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--ollama',type=Path,required=True);ap.add_argument('--lib-dir',type=Path,required=True);ap.add_argument('--models',type=Path,required=True);ap.add_argument('--model',required=True);ap.add_argument('--plugin',type=Path,required=True);ap.add_argument('--sidecar',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--board',required=True);ap.add_argument('--port',type=int,default=11450);ap.add_argument('--concurrency',type=int,nargs='+',default=[8,16]);ap.add_argument('--num-predict',type=int,default=8);a=ap.parse_args()
    for n in ('ollama','lib_dir','models','plugin','sidecar'):setattr(a,n,getattr(a,n).resolve(strict=True))
    a.output.mkdir(parents=True,exist_ok=False); initial=snapshot(); payload={'model':a.model,'prompt':'Paris','stream':False,'options':{'temperature':0,'num_predict':a.num_predict,'num_ctx':2048,'num_thread':4}}
    manifest={'schema':'rocknpu-ollama-concurrency-v1','board':a.board,'argv':sys.argv,'binary_hash':sha256(a.ollama),'model':a.model,'plugin_hash':sha256(a.plugin),'payload':payload,'initial_snapshot':initial,'concurrency':a.concurrency,'environment':safe_env(os.environ),'runs':[]}; write_json(a.output/'run-manifest.json',manifest)
    for c in a.concurrency:
        run_dir=a.output/f'c{c}'; run_dir.mkdir()
        env=service_env(a,a.port,'npu'); env.update({'OLLAMA_NUM_PARALLEL':str(c),'OLLAMA_MAX_QUEUE':str(c*2),'OLLAMA_MAX_LOADED_MODELS':'1'})
        service=start_service(f'c{c}',a.ollama,env,run_dir/'server')
        run={'concurrency':c,'service':{k:v for k,v in service.items() if k!='process'},'environment':safe_env(env),'before':snapshot()}
        try:
            import urllib.request
            end=time.monotonic()+180
            while time.monotonic()<end:
                try:
                    urllib.request.urlopen(f'http://127.0.0.1:{a.port}/api/tags',timeout=2).close();break
                except Exception:time.sleep(.25)
            one_request(a.port,payload,run_dir/'warmup')
            api_get(a.port,'/api/ps',run_dir/'ps-before.json')
            start=time.monotonic()
            with ThreadPoolExecutor(max_workers=c) as pool:
                futures=[pool.submit(one_request,a.port,payload,run_dir/f'request-{i:02d}') for i in range(c)]
                results=[f.result() for f in as_completed(futures)]
            wall=time.monotonic()-start
            api_get(a.port,'/api/ps',run_dir/'ps-after.json')
            run.update({'requests':sorted(results,key=lambda x:x.get('record',{}).get('request_file','')),'wall_s':wall,'after':snapshot(),'status':'complete' if all(x['status']=='ok' for x in results) else 'partial_or_timeout'})
        except Exception as e:
            run.update({'status':'error','error':repr(e),'after':snapshot()})
        finally:
            stop_service(service)
        write_json(run_dir/'run.json',run); manifest['runs'].append(run); write_json(a.output/'run-manifest.json',manifest)
    files={str(p.relative_to(a.output)):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in sorted(a.output.rglob('*')) if p.is_file()};write_json(a.output/'artifact-manifest.json',{'board':a.board,'files':files})
    print(json.dumps({'board':a.board,'runs':[(x['concurrency'],x['status'],x.get('wall_s')) for x in manifest['runs']],'files':len(files)}))
if __name__=='__main__':main()
