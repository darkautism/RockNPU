#!/usr/bin/env python3
"""Normalize an existing llama-bench results.json into explicit time and tok/s summaries."""
import argparse, hashlib, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('directory',type=Path); a=ap.parse_args(); d=a.directory
    rows=json.loads((d/'results.json').read_text()); means={}
    for kind in ('cpu','npu'):
        vals=[r['result'] for r in rows if r['kind']==kind]
        means[kind]={'mean_time_ns':sum(x['avg_ns'] for x in vals)/len(vals),'mean_tok_s':sum(x['avg_ts'] for x in vals)/len(vals)}
    out={'mean_time_ns':{k:v['mean_time_ns'] for k,v in means.items()},'mean_tok_s':{k:v['mean_tok_s'] for k,v in means.items()},'npu_over_cpu_latency_ratio':means['npu']['mean_time_ns']/means['cpu']['mean_time_ns'],'npu_over_cpu_tok_s':means['npu']['mean_tok_s']/means['cpu']['mean_tok_s'],'block_speedups':[]}
    for i in range(0,len(rows),4):
        b=rows[i:i+4]; c=[x['result'] for x in b if x['kind']=='cpu']; n=[x['result'] for x in b if x['kind']=='npu']; out['block_speedups'].append({'npu_over_cpu_latency_ratio':(sum(x['avg_ns'] for x in n)/len(n))/(sum(x['avg_ns'] for x in c)/len(c)),'npu_over_cpu_tok_s':(sum(x['avg_ts'] for x in n)/len(n))/(sum(x['avg_ts'] for x in c)/len(c))})
    (d/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    files={}
    for p in sorted(d.rglob('*')):
        if p.is_file() and p.name!='artifact-manifest.json': files[str(p.relative_to(d))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (d/'artifact-manifest.json').write_text(json.dumps({'directory':str(d),'files':files},indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
