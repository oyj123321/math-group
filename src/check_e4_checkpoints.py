"""Reconstruct inputs and check archived checkpoints against stored clean outputs."""
from pathlib import Path
import json, hashlib
import numpy as np
import torch
from run import data, MoE

torch.set_num_threads(4)
root=Path('work/moe_e4_results/runs');records=[]
for profile in ['independent','shortcut']:
    for seed in range(5):
        d=root/profile/f'seed{seed}';m=json.loads((d/'manifest.json').read_text())
        x,_,_=data(m['test'],30000)
        for arm in ['clean','augmentation','consistency']:
            with np.load(d/f'{arm}_predictions.npz') as f:
                y=f['target'];stored=f['clean'];saved_ids=torch.from_numpy(f['clean_ids'].copy());saved_weight=torch.from_numpy(f['clean_weight'].copy())
                assert hashlib.sha256(x.numpy().tobytes()+y.tobytes()).hexdigest()==m['dataset_sha256'][2]
            model=MoE(m['width'])
            model.load_state_dict(torch.load(d/f'{arm}.pt',map_location='cpu',weights_only=True));model.eval()
            with torch.no_grad():
                route=model.routing(x)
                pred=model(x,route)[0].numpy()
                replay=model(x,(saved_ids,saved_weight,route[2]))[0].numpy()
            changed=route[0]!=saved_ids
            selected_gap=route[1]-route[2].gather(1,saved_ids[:,None]).squeeze(1)
            assert float(selected_gap.max())<1e-5,'route changed away from a numerical near-tie'
            error=pred-stored
            replay_error=replay-stored
            assert np.allclose(replay,stored,atol=5e-5,rtol=2e-5),(profile,seed,arm,float(abs(replay_error).max()))
            records.append(dict(profile=profile,seed=seed,arm=arm,max_absolute_difference=float(abs(error).max()),
                                prediction_difference_mse=float(np.mean(error**2)),input_hash_verified=True,
                                near_tie_id_changes=int(changed.sum()),largest_stored_route_probability_gap=float(selected_gap.max()),
                                stored_route_replay_max_difference=float(abs(replay_error).max())))
Path('outputs/moe_e4/checkpoint_replay.json').write_text(json.dumps(records,indent=2))
print('30 stored-route checkpoint replays passed; numerical near-tie ID changes',sum(r['near_tie_id_changes'] for r in records),
      'max stored-route replay difference',max(r['stored_route_replay_max_difference'] for r in records))
