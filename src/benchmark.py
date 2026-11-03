"""One-device synchronized forward profiling; not an independent accuracy experiment."""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from run import MoE, matched_dense, budget, data, predict

p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True)
p.add_argument('--device',default='cuda:0');a=p.parse_args()
if not a.device.startswith('cuda'):raise ValueError('GPU benchmark required')
torch.set_num_threads(4)
directory=Path(a.run);manifest=json.loads((directory/'manifest.json').read_text())
moe=MoE(manifest['width']);b=budget(moe)
models={'moe':moe,'dense_active':matched_dense(b['active_linear_macs_per_item'],'macs'),
        'dense_total':matched_dense(b['parameters'],'parameters')}
rows=[]
with torch.no_grad():
    for name,model in models.items():
        model.load_state_dict(torch.load(directory/f'{name}.pt',map_location='cpu',weights_only=True))
        model.to(a.device).eval()
        for batch in [1,1024]:
            x=data(batch,5444)[0].to(a.device)
            for _ in range(10):predict(model,x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            gpu_ms=[];wall_ms=[]
            for _ in range(30):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize();t0=time.perf_counter();start.record()
                predict(model,x);end.record();torch.cuda.synchronize()
                wall_ms.append((time.perf_counter()-t0)*1000);gpu_ms.append(start.elapsed_time(end))
            rows.append(dict(model=name,batch=batch,repeats=30,wall_median_ms=float(np.median(wall_ms)),
                gpu_event_median_ms=float(np.median(gpu_ms)),wall_min_ms=min(wall_ms),wall_max_ms=max(wall_ms),
                peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,**budget(model)))
            del x
        model.cpu();torch.cuda.empty_cache()
result=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),
    source_training_sha256=manifest['source_sha256'],rows=rows,
    limitations=['one checkpoint per architecture, one profiling session',
                 'batch 1 contains task regime 0; batch 1024 is balanced across regimes',
                 'times include synchronization/dispatch; no data transfer or loading',
                 'peak allocated is PyTorch allocation, not complete driver/context memory',
                 'shared server CPU load was not controlled'])
Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
