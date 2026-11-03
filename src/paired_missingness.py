"""Local checkpoint re-evaluation with identical masks and a Gaussian Bayes reference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from run import data, perturb, MoE, budget, matched_dense, predict


def bayes_prediction(z, task):
    # Independent standard-normal relevant features; cue is intact in this control.
    u=z[:,4:12]
    mask=z[:,20:28]
    missing=mask.sum(1)
    raw=torch.stack([u.sum(1)/np.sqrt(8),
        (u[:,:4].sum(1)-u[:,4:].sum(1))/np.sqrt(8),
        (u[:,0]*u[:,1]+u[:,2]*u[:,3])/np.sqrt(2),
        torch.exp(-missing/16)*torch.sin(u.sum(1)/np.sqrt(8))],dim=1)
    return raw.gather(1,task[:,None]).squeeze(1)


def theoretical_risk(k,std):
    q=k/8
    raw=np.array([q,q,1-((8-k)*(7-k))/56,
                  (1-np.exp(-2)-np.exp(-q)+np.exp(-2+q))/2])
    return raw / np.asarray(std)**2


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    source=Path(a.runs)
    cases=sorted(source.glob('*/seed*'))
    if not cases:raise ValueError('no run cases')
    for directory in cases:
        m=json.loads((directory/'manifest.json').read_text())
        x,raw,task=data(m['test'],m['data_seeds']['test'])
        mean=torch.tensor(m['target_mean']);std=torch.tensor(m['target_std'])
        y=(raw-mean[task])/std[task]
        with np.load(directory/'moe_predictions.npz') as saved:
            stored_y=saved['target'].copy()
        if not np.allclose(y.numpy(),stored_y,atol=1e-6):raise ValueError('target reconstruction mismatch')
        if hashlib.sha256(x.numpy().tobytes()+stored_y.tobytes()).hexdigest()!=m['dataset_sha256'][2]:
            raise ValueError('input reconstruction hash mismatch')
        y=torch.from_numpy(stored_y)
        moe=MoE(m['width']);b=budget(moe)
        models={'moe':moe,'dense_active':matched_dense(b['active_linear_macs_per_item'],'macs'),
                'dense_total':matched_dense(b['parameters'],'parameters')}
        records=dict(source_id=np.arange(len(x)),target=y.numpy(),task=task.numpy())
        for name,model in models.items():
            model.load_state_dict(torch.load(directory/f'{name}.pt',map_location='cpu',weights_only=True))
            model.eval()
            with torch.no_grad():records[f'{name}_clean']=predict(model,x).numpy()
            if m['shortcut']:
                id_x=x.clone();id_x[:,12:16]+=2*x[:,:4]
                with torch.no_grad():records[f'{name}_in_distribution']=predict(model,id_x).numpy()
        notes=[]
        with torch.no_grad():
            clean_route=moe.routing(x)
            for strength_id,strength in enumerate((.125,.25,.5)):
                for replicate in range(m['perturb_reps']):
                    # Original v2 mask condition IDs: 9 noise + 3 shift, then masks.
                    perturb_seed=9000+3*m['perturb_reps']+3+strength_id*m['perturb_reps']+replicate
                    marked=perturb(x,'mask',strength,perturb_seed)
                    blind=marked.clone();blind[:,20:28]=0
                    assert torch.equal(marked[:,:20],blind[:,:20])
                    prefix=f'{strength}_r{replicate}'
                    oracle_raw=bayes_prediction(marked,task)
                    records[f'{prefix}_bayes']=((oracle_raw-mean[task])/std[task]).numpy()
                    for flag,z in [('marked',marked),('blind',blind)]:
                        for name,model in models.items():
                            records[f'{prefix}_{flag}_{name}']=predict(model,z).numpy()
                        records[f'{prefix}_{flag}_moe_restored']=moe(z,clean_route)[0].numpy()
                notes.append(dict(fraction=strength,features_removed=round(8*strength),
                    theoretical_normalized_risk_per_task=theoretical_risk(round(8*strength),m['target_std']).tolist()))
        target=out/directory.parent.name/directory.name
        target.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(target/'paired_missingness.npz',**records)
        (target/'metadata.json').write_text(json.dumps(dict(seed=m['seed'],shortcut=m['shortcut'],
            training_source_sha256=m['source_sha256'],
            local_analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            checkpoint_sha256={name:hashlib.sha256((directory/f'{name}.pt').read_bytes()).hexdigest() for name in models},
            theory=notes,
            interpretation='Oracle knows missing locations and intact task cue; Gaussian-data-specific conditional expectation, not deployable medical model.'),indent=2))
        print('Paired masks and Bayes reference:',target,flush=True)

if __name__=='__main__':main()
