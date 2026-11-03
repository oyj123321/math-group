import argparse,json,csv
from pathlib import Path
import numpy as np
from analyze_v2 import crossed_bootstrap


def contrasts(losses):
    m00,m01,m10,m11=[losses[k] for k in ['00','01','10','11']]
    return dict(identity_restore=m11-m01,amplitude_restore=m11-m10,
                joint_restore=m11-m00,interaction=m11-m10-m01+m00)


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--out',required=True)
    p.add_argument('--draws',type=int,default=1000);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);results=[];seedwise=[]
    for pi,profile in enumerate(['independent','shortcut']):
        for ai,arm in enumerate(['clean','augmentation','consistency']):
            fs=[np.load(Path(a.runs)/profile/f'seed{s}'/f'{arm}.npz') for s in range(5)]
            for f in fs:
                for key in ['source_id','target','task']:assert np.array_equal(f[key],fs[0][key])
            losses={k:np.stack([np.mean([(f[f'r{r}_pred{k}'].astype('float64')-f['target'])**2
                                         for r in range(3)],axis=0) for f in fs]) for k in ['00','01','10','11']}
            effects=contrasts(losses)
            assert np.allclose(effects['joint_restore'],effects['identity_restore']+effects['amplitude_restore']-effects['interaction'])
            metrics={f'M{k}':v for k,v in losses.items()};metrics.update(effects)
            metrics['id_switch_rate']=np.stack([np.mean([f[f'r{r}_clean_ids']!=f[f'r{r}_noisy_ids'] for r in range(3)],axis=0) for f in fs])
            metrics['gate_absolute_change']=np.stack([np.mean([abs(f[f'r{r}_clean_weight'].astype('float64')-f[f'r{r}_noisy_weight']) for r in range(3)],axis=0) for f in fs])
            ci=crossed_bootstrap(np.stack(list(metrics.values())),a.draws,942+pi*3+ai)
            for i,(name,values) in enumerate(metrics.items()):
                results.append(dict(profile=profile,arm=arm,metric=name,mean=float(values.mean()),
                                    ci_low=float(ci[0,i]),ci_high=float(ci[1,i]),seed_values=values.mean(1).tolist()))
                for task in range(4):
                    ix=fs[0]['task']==task
                    for seed in range(5):seedwise.append(dict(profile=profile,arm=arm,metric=name,task=task,seed=seed,value=float(values[seed,ix].mean())))
            for f in fs:f.close()
            print(profile,arm,'bootstrap complete',flush=True)
    (out/'results.json').write_text(json.dumps(results,indent=2))
    (out/'seedwise_task_metrics.json').write_text(json.dumps(seedwise,indent=2))
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=[k for k in results[0] if k!='seed_values']);w.writeheader()
        w.writerows({k:v for k,v in r.items() if k!='seed_values'} for r in results)
    meta=dict(date='2026-10-20',status='exploratory follow-up after E4',models=30,
              training_runs=0,draws=a.draws,interval='pointwise paired crossed bootstrap of 5 seeds and 20000 sources; average 3 perturbation draws within source first',
              sign='Positive restoration contrast favors restoration; signed effects and interactions are retained.',
              warning='Privileged route components; off-support crossed cells; not unique causal attribution; no CPU/GPU cell mixing.')
    (out/'analysis_metadata.json').write_text(json.dumps(meta,indent=2))


if __name__=='__main__':main()
