"""Paired crossed bootstrap: training seeds and source IDs; perturbations averaged first."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def crossed_bootstrap(values, draws=1000, seed=427):
    # values: metric x training seed x source item. Source IDs are shared across seeds.
    if values.ndim != 3 or min(values.shape) < 1:
        raise ValueError('expected nonempty metric/seed/source tensor')
    if not np.isfinite(values).all():
        raise ValueError('nonfinite values')
    rng=np.random.default_rng(seed)
    _, seeds, sources=values.shape
    result=np.empty((draws,values.shape[0]))
    for b in range(draws):
        seed_ids=rng.integers(seeds,size=seeds)
        source_ids=rng.integers(sources,size=sources)
        result[b]=values[:,seed_ids,:].mean(axis=1)[:,source_ids].mean(axis=1)
    return np.quantile(result,[.025,.975],axis=0)


def paired_conditions(archive):
    groups={}
    for key in archive.files:
        if key.endswith('_11') and '_r' in key:
            setting,rep=key[:-3].rsplit('_r',1)
            groups.setdefault(setting,[]).append(int(rep))
    return {k:sorted(v) for k,v in groups.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--out',required=True)
    p.add_argument('--draws',type=int,default=1000);a=p.parse_args()
    if a.draws < 100: p.error('at least 100 bootstrap draws required')
    paths=sorted(Path(a.runs).glob('seed*'))
    if len(paths)!=5: raise ValueError('five matched seeds required')
    manifests=[json.loads((d/'manifest.json').read_text()) for d in paths]
    for field in ['source_sha256','dataset_sha256','model_budgets','train','val','test','width','epochs','batch','balance','shortcut','perturb_reps']:
        if any(m[field]!=manifests[0][field] for m in manifests): raise ValueError('unmatched '+field)
    if any(m['status']!='complete' for m in manifests) or len({m['seed'] for m in manifests})!=5:
        raise ValueError('missing or duplicate complete seeds')
    archives={model:[np.load(d/f'{model}_predictions.npz') for d in paths]
              for model in ['moe','dense_active','dense_total']}
    reference=archives['moe'][0]
    for files in archives.values():
        for file in files:
            for key in ['source_id','target','task']:
                if not np.array_equal(file[key],reference[key]): raise ValueError('unpaired '+key)
    settings=paired_conditions(reference)
    records=[];seed_rows=[]
    for ci_id,(setting,reps) in enumerate(settings.items()):
        family,strength=setting.rsplit('_',1)
        errors={}
        for model,files in archives.items():
            clean=np.stack([(f['clean'].astype('float64')-f['target'])**2 for f in files])
            errors[(model,'clean')]=clean
            for cell in (['11','10','01','id_restore','random_fixed_gate'] if model=='moe' else ['11']):
                errors[(model,cell)]=np.stack([
                    np.mean([(f[f'{setting}_r{r}_{cell}'].astype('float64')-f['target'])**2 for r in reps],axis=0)
                    for f in files])
        base=errors[('moe','clean')];e11=errors[('moe','11')];e10=errors[('moe','10')];e01=errors[('moe','01')]
        metrics={
            'moe_clean_mse':base,'moe_corrupt_mse':e11,
            'total_degradation':e11-base,'expert_input_term':e10-base,
            'routing_term':e11-e10,'id_only_routing_term':e11-errors[('moe','id_restore')],
            'interaction':e11-e10-e01+base,
            'dense_active_corrupt_mse':errors[('dense_active','11')],
            'dense_total_corrupt_mse':errors[('dense_total','11')],
            'moe_minus_dense_active':e11-errors[('dense_active','11')],
            'moe_minus_dense_total':e11-errors[('dense_total','11')],
        }
        if not np.allclose(metrics['total_degradation'],metrics['expert_input_term']+metrics['routing_term']):
            raise ValueError('decomposition invariant violated')
        # Only primary population-average metrics receive bootstrap intervals here.
        names=list(metrics);stack=np.stack([metrics[k] for k in names])
        ci=crossed_bootstrap(stack,a.draws,427+ci_id)
        for k,key in enumerate(names):
            seed_values=metrics[key].mean(axis=1)
            records.append(dict(family=family,strength=float(strength),metric=key,mean=float(seed_values.mean()),
                                ci_low=float(ci[0,k]),ci_high=float(ci[1,k]),training_seeds=5,
                                perturbation_replicates=len(reps),source_items=stack.shape[-1],
                                seed_values=seed_values.tolist()))
        for regime in [-1,0,1,2,3]:
            idx=np.ones(len(reference['task']),dtype=bool) if regime==-1 else reference['task']==regime
            for name,array in metrics.items():
                for seed in range(5):
                    seed_rows.append(dict(family=family,strength=float(strength),regime=regime,seed=manifests[seed]['seed'],
                                          metric=name,value=float(array[seed,idx].mean()),n=int(idx.sum())))
        print(setting,'bootstrap complete',flush=True)
    specialization=[]
    task=reference['task']
    dominant_effects=[]
    for d,file in zip(paths,archives['moe']):
        clean=(file['clean'].astype('float64')-file['target'])**2
        info=json.loads((d/'specialization.json').read_text())
        dominant=np.zeros(len(task))
        for regime in range(4):
            idx=task==regime
            expert=int(np.argmax(np.bincount(file['clean_ids'][idx],minlength=4)))
            removed=(file[f'expert_remove_{expert}'].astype('float64')-file['target'])**2
            dominant[idx]=(removed-clean)[idx]
        dominant_effects.append(dominant)
        for expert in range(4):
            remove=(file[f'expert_remove_{expert}'].astype('float64')-file['target'])**2
            replace=(file[f'expert_replace_{expert}'].astype('float64')-file['target'])**2
            for regime in range(4):
                idx=task==regime
                specialization.append(dict(seed=int(d.name[4:]),expert=expert,regime=regime,
                    clean_mse=float(clean[idx].mean()),forced_expert_mse=float(replace[idx].mean()),
                    removal_delta_mse=float((remove-clean)[idx].mean()),
                    routing_fraction=float((file['clean_ids'][idx]==expert).mean()),
                    mutual_information_nats=info['mutual_information_nats']))
    dominant_effects=np.stack(dominant_effects)
    bounds=crossed_bootstrap(dominant_effects[None,:,:],a.draws,799)
    records.append(dict(family='specialization',strength=0.,metric='dominant_expert_removal_delta',
                        mean=float(dominant_effects.mean()),ci_low=float(bounds[0,0]),ci_high=float(bounds[1,0]),
                        training_seeds=5,perturbation_replicates=1,source_items=len(task),
                        seed_values=dominant_effects.mean(1).tolist()))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('confidence_intervals',records),('seedwise_metrics',seed_rows),('specialization',specialization)]:
        (out/f'{name}.json').write_text(json.dumps(rows,indent=2))
        with (out/f'{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'analysis_metadata.json').write_text(json.dumps(dict(bootstrap_draws=a.draws,
        method='paired crossed seed/source bootstrap; perturbation replicas averaged within source',
        interval='pointwise percentile 95%; exploratory, no multiplicity correction',
        limitations=['five seeds give limited tail precision','seed resampling is not an independent dataset replication',
                     'synthetic tasks and observable regime cue','linear MACs omit routing/activation overhead'],
        source_sha256=manifests[0]['source_sha256'],configuration=manifests[0]),indent=2))
    for files in archives.values():
        for f in files:f.close()
    print('Analysis complete:',out,flush=True)

if __name__=='__main__':main()
