"""Inspect PetShop's measured aggregates; no inferred queue events or unit conversion."""
from pathlib import Path
import json, hashlib, argparse
import numpy as np
import pandas as pd
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-dir', type=Path, required=True, help='Directory containing dataset/')
parser.add_argument('--output-dir', type=Path, required=True)
args=parser.parse_args()
root=args.data_dir
out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
records=[];issues=[];curves=[]
for scenario in ['low_traffic','high_traffic','temporal_traffic1','temporal_traffic2']:
 p=root/'dataset'/scenario/'noissue/metrics.csv'
 d=pd.read_csv(p,header=[0,1,2],index_col=0)
 n=d[('PetSite','requests','Sum')]; y=d[('PetSite','latency','Average')]
 if not np.all(np.diff(d.index) == 300) or not np.all(np.isfinite(n)) or not np.all(np.isfinite(y)) or (n < 0).any() or (y < 0).any():
  raise ValueError('Invalid cadence, counts, or latency values')
 g=pd.read_csv(root/'dataset'/scenario/'graph.csv',index_col=0)
 records.append(dict(scenario=scenario,rows=len(d),columns=len(d.columns),nodes=len(g),edges=int(g.values.sum()),first_utc=pd.to_datetime(d.index.min(),unit='s',utc=True).isoformat(),last_utc=pd.to_datetime(d.index.max(),unit='s',utc=True).isoformat(),cadence_seconds=sorted(set(np.diff(d.index))),requests_per_window_mean=n.mean(),request_rate_min=n.min()/300,request_rate_max=n.max()/300,latency_raw_mean=y.mean(),latency_raw_request_weighted_mean=(y*n).sum()/n.sum(),availability_mean=d[('PetSite','availability','Average')].mean(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 q=pd.DataFrame(dict(timestamp=d.index,requests=n.values,requests_per_second=n.values/300,latency_average_raw=y.values,availability_percent=d[('PetSite','availability','Average')].values))
 q.to_csv(out/f'{scenario}_petsite.csv',index=False)
 # descriptive bins only; these are not inferential confidence bounds
 q['load_bin']=pd.cut(q.requests_per_second,bins=np.arange(0,6.25,.25),right=False)
 for b,z in q.groupby('load_bin',observed=True):
  curves.append(dict(scenario=scenario,load_bin=str(b),windows=len(z),requests=z.requests.sum(),rate_mean=z.requests_per_second.mean(),latency_weighted_raw=np.average(z.latency_average_raw,weights=z.requests)))
 for t in (root/'dataset'/scenario).glob('*/*/target.json'):
  meta=json.load(open(t));m=pd.read_csv(t.parent/'metrics.csv',header=[0,1,2],index_col=0);ts=meta['target']['timestamp'];a=m[('PetSite','latency','Average')];post=m.index>=ts
  issues.append(dict(scenario=scenario,split=t.parts[-3],issue=t.parts[-2],rows=len(m),metric=meta['target']['metric'],agg=meta['target']['agg'],root_cause=meta['root_cause']['node'],injection=ts,first=m.index.min(),last=m.index.max(),before_windows=int((~post).sum()),after_windows=int(post.sum()),latency_before_raw=a[~post].mean(),latency_after_raw=a[post].mean(),command=meta['metadata'].get('reproducibility',{}).get('command','')))
json.dump(records,open(out/'summary.json','w'),indent=2)
pd.DataFrame(issues).to_csv(out/'issues.csv',index=False)
pd.DataFrame(curves).to_csv(out/'descriptive_curves.csv',index=False)
print(json.dumps(records,indent=2));print(pd.DataFrame(issues).groupby(['scenario','split','metric']).size());print('total issues',len(issues))
