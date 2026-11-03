"""Aggregate completed matched runs; keep all seed values and make scientific plots."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    paths=sorted(Path(a.runs).glob('seed*'))
    if not paths: raise ValueError('no seed runs')
    manifests=[json.loads((r/'manifest.json').read_text()) for r in paths]
    if any(m['status']!='complete' for m in manifests): raise ValueError('incomplete run')
    for k in ['source_sha256','data_seeds','train','val','test','width','epochs','batch','balance','shortcut']:
        if any(m[k]!=manifests[0][k] for m in manifests): raise ValueError(f'unmatched {k}')
    if len({m['seed'] for m in manifests})!=len(manifests): raise ValueError('duplicate seeds')
    groups={}
    for path in paths:
        for row in csv.DictReader((path/'metrics.csv').open()):
            key=(row['model'],row['family'],float(row['strength']),int(row['regime']))
            groups.setdefault(key,[]).append(row)
    records=[]
    for key,rows in groups.items():
        if len(rows)!=len(paths): raise ValueError('missing conditions')
        for field in ['00','11','10','01','expert_term','routing_term','total','interaction','switch_rate']:
            values=[float(r[field]) for r in rows if r.get(field,'')!='']
            if not values: continue
            records.append(dict(model=key[0],family=key[1],strength=key[2],regime=key[3],metric=field,
                                seeds=len(values),mean=float(np.mean(values)),
                                seed_sd=float(np.std(values,ddof=1)) if len(values)>1 else None,
                                minimum=min(values),maximum=max(values),seed_values=values))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'aggregate.json').write_text(json.dumps(records,indent=2))
    with (out/'aggregate.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,family in zip(axs,['noise','mask']):
        for model in ['moe','dense_active','dense_total']:
            rows=sorted([r for r in records if r['model']==model and r['family']==family and r['regime']==-1 and r['metric']=='11'],key=lambda r:r['strength'])
            ax.errorbar([r['strength'] for r in rows],[r['mean'] for r in rows],yerr=[r['seed_sd'] or 0 for r in rows],marker='o',capsize=3,label=model)
        ax.set_xlabel('Nuisance noise standard deviation' if family=='noise' else 'Fraction of relevant features masked')
        ax.set_ylabel('Normalized-target test MSE')
        ax.set_title('Nuisance perturbation' if family=='noise' else 'Relevant information loss')
        ax.legend();ax.grid(alpha=.2)
    fig.suptitle(f'Controlled synthetic pilot | {len(paths)} training seeds | bars: seed SD, not confidence intervals',fontsize=10)
    fig.savefig(out/'robustness.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    rows=sorted([r for r in records if r['model']=='moe' and r['family']=='noise' and r['regime']==-1 and r['metric']=='routing_term'],key=lambda r:r['strength'])
    ax.errorbar([r['strength'] for r in rows],[r['mean'] for r in rows],yerr=[r['seed_sd'] or 0 for r in rows],marker='o',capsize=3)
    ax.axhline(0,color='black',linewidth=.8);ax.set_xlabel('Nuisance noise standard deviation')
    ax.set_ylabel('Routing intervention effect: L11 - L10 (MSE)')
    ax.set_title('Positive: clean routing helps; negative: clean routing hurts\nBars: between-seed SD, not confidence intervals',fontsize=10)
    ax.grid(alpha=.2);fig.savefig(out/'routing_effect.png',dpi=180);plt.close(fig)
    print(json.dumps({'completed_seeds':[m['seed'] for m in manifests],'rows':len(records),'out':str(out)}))

if __name__=='__main__': main()
