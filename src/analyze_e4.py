"""Matched E4 analysis; primary comparisons fixed before training."""
import argparse
import json
import csv
from pathlib import Path
import numpy as np
from analyze_v2 import crossed_bootstrap, paired_conditions


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--out',required=True)
    p.add_argument('--draws',type=int,default=1000);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    results=[];loads=[]
    for profile in ['independent','shortcut']:
        paths=[Path(a.runs)/profile/f'seed{i}' for i in range(5)]
        manifests=[json.loads((d/'manifest.json').read_text()) for d in paths]
        for i,m in enumerate(manifests):
            assert m['status']=='complete' and m['seed']==i
            for key in ['dataset_sha256','source_sha256','epochs','batch','width','noise_sd','kl_weight']:
                assert m[key]==manifests[0][key],key
        arms=['clean','augmentation','consistency']
        files={arm:[np.load(d/f'{arm}_predictions.npz') for d in paths] for arm in arms}
        ref=files['clean'][0]
        for fs in files.values():
            for f in fs:
                for key in ['source_id','target','task']:
                    assert np.array_equal(f[key],ref[key]),key
        settings={'clean':[0]};settings.update(paired_conditions(ref))
        for sid,(setting,reps) in enumerate(settings.items()):
            errors={};switches={}
            for arm,fs in files.items():
                if setting=='clean':
                    errors[arm]=np.stack([(f['clean'].astype('float64')-f['target'])**2 for f in fs])
                else:
                    errors[arm]=np.stack([np.mean([(f[f'{setting}_r{r}_11'].astype('float64')-f['target'])**2
                                                  for r in reps],axis=0) for f in fs])
                    switches[arm]=np.stack([np.mean([f[f'{setting}_r{r}_noisy_ids']!=f['clean_ids']
                                                    for r in reps],axis=0) for f in fs])
            metrics={arm+'_mse':v for arm,v in errors.items()}
            metrics['consistency_minus_augmentation']=errors['consistency']-errors['augmentation']
            metrics['augmentation_minus_clean']=errors['augmentation']-errors['clean']
            metrics['consistency_minus_clean']=errors['consistency']-errors['clean']
            if setting!='clean':
                baselines={arm:np.stack([(f['clean'].astype('float64')-f['target'])**2 for f in fs])
                           for arm,fs in files.items()}
                for arm in arms:metrics[arm+'_degradation']=errors[arm]-baselines[arm]
                metrics['consistency_minus_augmentation_degradation']=(errors['consistency']-baselines['consistency'])-(errors['augmentation']-baselines['augmentation'])
            for arm,value in switches.items():metrics[arm+'_switch_rate']=value
            names=list(metrics)
            ci=crossed_bootstrap(np.stack(list(metrics.values())),a.draws,761+sid)
            for i,name in enumerate(names):
                results.append(dict(profile=profile,condition=setting,metric=name,
                                    mean=float(metrics[name].mean()),ci_low=float(ci[0,i]),ci_high=float(ci[1,i]),
                                    seed_values=metrics[name].mean(axis=1).tolist(),
                                    primary=setting=='noise_1.0' and name=='consistency_minus_augmentation',
                                    source_items=len(ref['target']),seeds=5))
            print(profile,setting,'done',flush=True)
        for arm in arms:
            for seed,d in enumerate(paths):
                t=json.loads((d/f'{arm}_training.json').read_text())
                load=np.array(t['clean_test_expert_load'])
                loads.append(dict(profile=profile,arm=arm,seed=seed,loads=load.tolist(),
                                  max_load=float(load.max()),active_experts=int((load>0).sum()),
                                  entropy=float(-(load[load>0]*np.log(load[load>0])).sum()),
                                  best_validation_mse=min(r['validation_clean_mse'] for r in t['history']),
                                  selected_epoch=min(t['history'],key=lambda r:r['validation_clean_mse'])['epoch'],
                                  elapsed_seconds=t['elapsed_seconds']))
        for fs in files.values():
            for f in fs:f.close()
    (out/'results.json').write_text(json.dumps(results,indent=2))
    (out/'expert_loads.json').write_text(json.dumps(loads,indent=2))
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=[k for k in results[0] if k!='seed_values']);w.writeheader()
        w.writerows({k:v for k,v in r.items() if k!='seed_values'} for r in results)
    meta=dict(training_runs=30,profiles=2,seeds=5,draws=a.draws,
              interval='pointwise paired crossed bootstrap, conditional on shared fixed training data',
              primary='consistency minus augmentation MSE, nuisance noise SD 1, separately by profile',
              frozen_kl=.1,training_noise_sd=.5,selection='minimum clean validation MSE')
    meta['degradation_difference']='Exploratory diagnostic added after scalar-mean inspection; not a primary endpoint.'
    (out/'analysis_metadata.json').write_text(json.dumps(meta,indent=2))


if __name__=='__main__':main()
