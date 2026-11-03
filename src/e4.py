"""E4: matched two-view training; fixed KL weight, no test-driven tuning."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from run import MoE, data, normalize, perturb, evaluate, budget


def noisy_view(x, sd, generator):
    z = x.clone()
    z[:, 12:20] += sd * torch.randn((len(x), 8), generator=generator, device=x.device)
    return z


def kl_consistency(clean, noisy):
    # Clean teacher detached; KL(p_clean || p_noisy), averaged over inputs.
    p = clean.detach().clamp_min(1e-8)
    return (p * (p.log() - noisy.clamp_min(1e-8).log())).sum(-1).mean()


def balance(prob, selected):
    freq = torch.nn.functional.one_hot(selected, 4).float().mean(0)
    return 4 * (freq * prob.mean(0)).sum()


def fit_e4(model, dataset, validation, args, arm):
    device = torch.device(args.device)
    model.to(device)
    x, y, _ = [a.to(device) for a in dataset]
    vx, vy, _ = [a.to(device) for a in validation]
    # Independent streams prevent stochastic calls in one arm altering minibatches.
    order_gen = torch.Generator(device=device).manual_seed(100000 + args.seed)
    noise_gen = torch.Generator(device=device).manual_seed(200000 + args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history, best, state = [], float('inf'), None
    for epoch in range(args.epochs):
        model.train()
        sums = np.zeros(3)
        for idx in torch.randperm(len(x), device=device, generator=order_gen).split(args.batch):
            clean = x[idx]
            noise = noisy_view(clean, args.noise_sd, noise_gen)
            second = clean if arm == 'clean' else noise
            opt.zero_grad()
            a, pa, ia = model(clean)
            b, pb, ib = model(second)
            mse = .5 * ((a-y[idx]).square().mean() + (b-y[idx]).square().mean())
            lb = .5 * (balance(pa, ia) + balance(pb, ib))
            kl = kl_consistency(pa, pb)
            loss = mse + args.balance*lb + (args.kl_weight*kl if arm == 'consistency' else 0)
            loss.backward()
            opt.step()
            sums += np.array([mse.item(), lb.item(), kl.item()]) * len(idx)
        model.eval()
        with torch.no_grad():
            score = (model(vx)[0]-vy).square().mean().item()
        history.append(dict(epoch=epoch+1, validation_clean_mse=score,
                            train_mse=sums[0]/len(x), balance=sums[1]/len(x), kl=sums[2]/len(x)))
        if score < best:
            best = score
            state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if (epoch+1) % 25 == 0:
            print(arm, epoch+1, 'clean validation', score, flush=True)
    model.load_state_dict(state)
    return history


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--shortcut',action='store_true')
    p.add_argument('--device',default='cpu')
    for name,value in [('train',80000),('val',10000),('test',20000),('width',128),
                       ('epochs',100),('batch',1024),('perturb-reps',3)]:
        p.add_argument('--'+name,type=int,default=value)
    p.add_argument('--lr',type=float,default=.002)
    p.add_argument('--balance',type=float,default=.1)
    p.add_argument('--kl-weight',type=float,default=.1)
    p.add_argument('--noise-sd',type=float,default=.5)
    args=p.parse_args()
    if min(args.train,args.val,args.test)<4 or min(args.width,args.epochs,args.batch,args.perturb_reps)<1:
        p.error('invalid sizes')
    if args.kl_weight<0 or args.noise_sd<0:
        p.error('negative coefficient')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    sets,mean,std=normalize(data(args.train,10000,args.shortcut,True),
                            data(args.val,20000,args.shortcut,True),data(args.test,30000))
    torch.manual_seed(args.seed)
    initial=MoE(args.width).state_dict()
    manifest=dict(vars(args),status='running',protocol='E4-fixed-0.1-v1',
                  torch=torch.__version__,gpu=torch.cuda.get_device_name() if 'cuda' in args.device else None,
                  source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                                 for n in ['e4.py','run.py']},
                  target_mean=mean.tolist(),target_std=std.tolist(),
                  dataset_sha256=[hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest() for x,y,_ in sets],
                  initial_sha256=hashlib.sha256(b''.join(v.numpy().tobytes() for v in initial.values())).hexdigest(),
                  checkpoint_selection='lowest clean validation MSE, same for all arms',
                  arms=['clean','augmentation','consistency'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    start=time.time()
    for arm in manifest['arms']:
        t0=time.time(); model=MoE(args.width);model.load_state_dict(initial)
        history=fit_e4(model,sets[0],sets[1],args,arm)
        torch.save(model.cpu().state_dict(),out/f'{arm}.pt');model.to(args.device)
        rows=evaluate(model,sets[2],args,torch.device(args.device),out,arm)
        (out/f'{arm}_metrics.json').write_text(json.dumps(rows))
        with torch.no_grad():
            _,_,ids=model(sets[2][0].to(args.device))
            load=torch.bincount(ids,minlength=4).float()/len(ids)
        (out/f'{arm}_training.json').write_text(json.dumps(dict(history=history,elapsed_seconds=time.time()-t0,
                       clean_test_expert_load=load.cpu().tolist(),budget=budget(model)),indent=2))
        print('DONE',arm,'seconds',time.time()-t0,flush=True)
    manifest.update(status='complete',elapsed_seconds=time.time()-start)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
