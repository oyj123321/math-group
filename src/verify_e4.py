import argparse
import hashlib
import json
import tarfile
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--sha256',required=True)
p.add_argument('--out',required=True);a=p.parse_args()
archive=Path(a.archive);digest=hashlib.sha256(archive.read_bytes()).hexdigest()
assert digest==a.sha256,'archive checksum'
out=Path(a.out).resolve();out.mkdir(exist_ok=False)
with tarfile.open(archive) as t:
    for m in t.getmembers():
        target=(out/m.name).resolve()
        assert out in target.parents and (m.isfile() or m.isdir()),'unsafe tar entry'
    t.extractall(out)
hashes={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ['e4.py','run.py']}
verified=[]
for profile in ['independent','shortcut']:
    for seed in range(5):
        d=out/'runs'/profile/f'seed{seed}';m=json.loads((d/'manifest.json').read_text())
        assert m['source_sha256']==hashes and m['status']=='complete' and m['seed']==seed
        assert m['shortcut']==(profile=='shortcut')
        for arm in ['clean','augmentation','consistency']:
            for suffix in ['.pt','_predictions.npz','_training.json','_metrics.json']:
                assert (d/f'{arm}{suffix}').is_file()
            with np.load(d/f'{arm}_predictions.npz') as f:
                assert len(f['target'])==20000 and np.isfinite(f['clean']).all()
                assert np.isfinite(f['noise_1.0_r0_11']).all()
        verified.append(dict(profile=profile,seed=seed))
(out/'VERIFIED.json').write_text(json.dumps(dict(archive_sha256=digest,source_sha256=hashes,
                                                completed_runs=verified),indent=2))
print('Verified',len(verified),'seed-profile runs / 30 model checkpoints',flush=True)
