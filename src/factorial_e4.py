"""Frozen 2x2 ID/amplitude interventions; all cells recomputed on CPU."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import torch
from run import MoE, data, perturb


def factorial_predictions(model, x, z):
    j0,w0,p0=model.routing(x)
    j1,w1,p1=model.routing(z)
    # Hold expert input and computation fixed while independently crossing scalars.
    e0=model(z,(j0,torch.ones_like(w0),p0))[0]
    e1=model(z,(j1,torch.ones_like(w1),p1))[0]
    cells={'00':e0*w0,'01':e0*w1,'10':e1*w0,'11':e1*w1}
    return cells,dict(clean_ids=j0,noisy_ids=j1,clean_weight=w0,noisy_weight=w1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    notes=[]
    for profile in ['independent','shortcut']:
        for seed in range(5):
            source=Path(a.runs)/profile/f'seed{seed}'
            manifest=json.loads((source/'manifest.json').read_text())
            assert manifest['status']=='complete' and manifest['perturb_reps']==3
            x,_,task=data(manifest['test'],30000)
            for arm in ['clean','augmentation','consistency']:
                with np.load(source/f'{arm}_predictions.npz') as f:
                    y=f['target'].copy();archived_clean=f['clean'].copy()
                assert hashlib.sha256(x.numpy().tobytes()+y.tobytes()).hexdigest()==manifest['dataset_sha256'][2]
                model=MoE(manifest['width']);model.load_state_dict(torch.load(source/f'{arm}.pt',map_location='cpu',weights_only=True));model.eval()
                records=dict(source_id=np.arange(len(x)),target=y,task=task.numpy())
                with torch.no_grad():
                    clean=model(x)[0]
                    records['clean_prediction']=clean.numpy()
                    for replicate in range(3):
                        z=perturb(x,'noise',1.,9006+replicate)
                        cells,routes=factorial_predictions(model,x,z)
                        assert torch.equal(cells['11'],model(z)[0])
                        same=routes['clean_ids']==routes['noisy_ids']
                        assert torch.equal(cells['00'][same],cells['10'][same])
                        assert torch.equal(cells['01'][same],cells['11'][same])
                        for key,value in cells.items():records[f'r{replicate}_pred{key}']=value.numpy()
                        for key,value in routes.items():records[f'r{replicate}_{key}']=value.numpy()
                dest=out/profile/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest/f'{arm}.npz',**records)
                error=records['clean_prediction']-archived_clean
                notes.append(dict(profile=profile,seed=seed,arm=arm,input_hash_verified=True,
                                  checkpoint_sha256=hashlib.sha256((source/f'{arm}.pt').read_bytes()).hexdigest(),
                                  clean_cpu_gpu_prediction_difference_mse=float(np.mean(error**2))))
                print(profile,seed,arm,'done',flush=True)
    meta=dict(date='2026-10-20',backend='CPU, float32 inference; float64 statistical losses',
              numpy=np.__version__,torch=torch.__version__,platform=platform.platform(),
              perturbation='nuisance Gaussian SD 1',perturbation_seeds=[9006,9007,9008],
              cell_definition='M_ab = E[(w_b h_{j_a}(z)-y)^2]; 0 clean route component, 1 noisy component',
              distinction='All cells share corrupted expert input; different from E3 expert-input/routing four cells.',
              status='complete',checks=notes,
              source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                             for n in ['factorial_e4.py','run.py']})
    (out/'manifest.json').write_text(json.dumps(meta,indent=2))


if __name__=='__main__':main()
