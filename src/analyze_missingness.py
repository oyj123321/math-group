import argparse
import csv
import json
from pathlib import Path
import numpy as np
from analyze_v2 import crossed_bootstrap

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--out',required=True)
    p.add_argument('--draws',type=int,default=1000);a=p.parse_args()
    root=Path(a.runs);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    results=[]
    for profile in ['independent','shortcut']:
        cases=sorted((root/profile).glob('seed*'))
        if len(cases)!=5:raise ValueError('need five seeds')
        files=[np.load(d/'paired_missingness.npz') for d in cases]
        y=files[0]['target']
        for f in files:
            if not np.array_equal(f['target'],y):raise ValueError('unpaired target')
        reps=sorted(int(k.split('_r')[1].split('_')[0]) for k in files[0].files if k.startswith('0.125_r') and k.endswith('_bayes'))
        for si,strength in enumerate((.125,.25,.5)):
            metrics={}
            def loss(key):return np.stack([np.mean([(f[f'{strength}_r{r}_{key}'].astype('float64')-y)**2 for r in reps],axis=0) for f in files])
            oracle=loss('bayes')
            metrics['empirical_bayes_mse']=oracle
            for model in ['moe','dense_active','dense_total']:
                marked=loss('marked_'+model);blind=loss('blind_'+model)
                metrics[model+'_marked_mse']=marked;metrics[model+'_blind_mse']=blind
                metrics[model+'_marker_penalty']=marked-blind
                metrics[model+'_blind_excess_over_bayes']=blind-oracle
            metrics['marked_routing_term']=metrics['moe_marked_mse']-loss('marked_moe_restored')
            metrics['blind_routing_term']=metrics['moe_blind_mse']-loss('blind_moe_restored')
            names=list(metrics);values=np.stack([metrics[k] for k in names])
            ci=crossed_bootstrap(values,a.draws,900+si)
            for i,name in enumerate(names):
                results.append(dict(profile=profile,strength=strength,metric=name,mean=float(values[i].mean()),
                                    ci_low=float(ci[0,i]),ci_high=float(ci[1,i]),seed_values=values[i].mean(1).tolist()))
            print(profile,strength,'paired missingness analyzed',flush=True)
        if profile=='shortcut':
            # Same trained model and target; remove only training nuisance correlations.
            diffs=np.stack([np.stack([(f[model+'_clean'].astype('float64')-y)**2-(f[model+'_in_distribution'].astype('float64')-y)**2 for f in files]) for model in ['moe','dense_active','dense_total']])
            bounds=crossed_bootstrap(diffs,a.draws,990)
            for i,model in enumerate(['moe','dense_active','dense_total']):
                results.append(dict(profile=profile,strength=0.,metric=model+'_shortcut_break_delta',mean=float(diffs[i].mean()),ci_low=float(bounds[0,i]),ci_high=float(bounds[1,i]),seed_values=diffs[i].mean(1).tolist()))
        for f in files:f.close()
    (out/'paired_controls.json').write_text(json.dumps(results,indent=2))
    with (out/'paired_controls.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    print('Paired control analysis complete',flush=True)

if __name__=='__main__':main()
