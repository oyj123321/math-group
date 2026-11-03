"""Controlled MoE pilot. Run locally or on an allocated GPU; no network services."""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


def data(n, seed, shortcut=False, train=False):
    if n < 1:
        raise ValueError('dataset size must be positive')
    rng = np.random.default_rng(seed)
    task = np.arange(n) % 4
    rng.shuffle(task)
    u = rng.normal(size=(n, 8)).astype('float32')
    v = rng.normal(size=(n, 8)).astype('float32')
    if shortcut and train:
        v[:, :4] += 2 * np.eye(4, dtype='float32')[task]
    targets = np.stack([
        u.sum(1) / np.sqrt(8),
        (u[:, :4].sum(1) - u[:, 4:].sum(1)) / np.sqrt(8),
        (u[:, 0] * u[:, 1] + u[:, 2] * u[:, 3]) / np.sqrt(2),
        np.sin(u.sum(1) / np.sqrt(8)),
    ], axis=1).astype('float32')
    x = np.concatenate([np.eye(4, dtype='float32')[task], u, v,
                        np.zeros((n, 8), dtype='float32')], axis=1)
    return torch.from_numpy(x), torch.from_numpy(targets[np.arange(n), task]), torch.from_numpy(task)


def normalize(train, *others):
    x, y, task = train
    mean = torch.stack([y[task == k].mean() for k in range(4)])
    std = torch.stack([y[task == k].std(unbiased=False).clamp_min(1e-6) for k in range(4)])
    return [(xx, (yy - mean[tt]) / std[tt], tt) for xx, yy, tt in (train,) + others], mean, std


def perturb(x, family, strength, seed):
    if strength < 0:
        raise ValueError('negative perturbation strength')
    rng = np.random.default_rng(seed)
    z = x.clone()
    if family == 'noise':
        z[:, 12:20] += torch.from_numpy(rng.normal(0, strength, (len(x), 8)).astype('float32'))
    elif family == 'shift':
        z[:, 12:20] += strength
    elif family in ('mask', 'mask_blind'):
        if strength > 1:
            raise ValueError('mask fraction must be at most one')
        count = round(8 * strength)
        mask = np.zeros((len(x), 8), dtype='float32')
        positions = np.argsort(rng.random((len(x), 8)), axis=1)[:, :count]
        np.put_along_axis(mask, positions, 1, axis=1)
        m = torch.from_numpy(mask)
        z[:, 4:12] *= 1 - m
        if family == 'mask':
            z[:, 20:28] = m
    elif family == 'cue':
        if strength > 1:
            raise ValueError('cue removal fraction must be at most one')
        selected = rng.choice(len(x), round(len(x) * strength), replace=False)
        z[selected, :4] = 0
    else:
        raise ValueError(f'unknown perturbation {family}')
    return z


class MoE(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.router = nn.Sequential(nn.Linear(28, width), nn.Tanh(), nn.Linear(width, 4))
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(28, width), nn.ReLU(),
                                     nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1)) for _ in range(4)])

    def routing(self, x):
        p = self.router(x).softmax(-1)
        weight, index = p.max(-1)
        return index, weight, p

    def forward(self, x, route=None):
        index, weight, p = self.routing(x) if route is None else route
        # Execute selected experts only; bypassing capacity limits avoids token-drop confounding.
        out = torch.zeros(len(x), device=x.device)
        for i, expert in enumerate(self.experts):
            selected = index == i
            if selected.any():
                out[selected] = expert(x[selected]).squeeze(-1)
        return out * weight, p, index


class Dense(nn.Module):
    def __init__(self, width, second_width=None):
        super().__init__()
        h = width if second_width is None else second_width
        self.net = nn.Sequential(nn.Linear(28, width), nn.ReLU(), nn.Linear(width, h),
                                 nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def predict(model, x):
    return model(x)[0] if isinstance(model, MoE) else model(x)


def budget(model):
    parameters = sum(p.numel() for p in model.parameters())
    def macs(net):
        return sum(m.in_features * m.out_features for m in net.modules() if isinstance(m, nn.Linear))
    active_macs = macs(model.router) + macs(model.experts[0]) if isinstance(model, MoE) else macs(model)
    return {'parameters': parameters, 'active_linear_macs_per_item': active_macs}


def matched_dense(target, metric):
    # Search integer rectangular widths with comparable aspect ratio, no dummy parameters.
    candidates=[]
    for first in range(4, int(np.sqrt(target) * 2) + 1):
        second = round((target - 28 * first) / (first + 1)) if metric == 'macs' else round((target - 29 * first - 1) / (first + 2))
        if second < 4 or not 0.5 <= first / second <= 2:
            continue
        cost = 28 * first + first * second + second if metric == 'macs' else 29 * first + first * second + 2 * second + 1
        candidates.append((abs(cost-target), abs(np.log(first/second)), first, second))
    if not candidates:
        raise ValueError('no matching dense architecture')
    _, _, first, second = min(candidates)
    return Dense(first, second)


def fit(model, train, val, args, device):
    model.to(device)
    tx, ty, _ = [t.to(device) for t in train]
    vx, vy, _ = [t.to(device) for t in val]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history, best, state = [], float('inf'), None
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(tx), device=device)
        for idx in order.split(args.batch):
            opt.zero_grad()
            if isinstance(model, MoE):
                pred, prob, selected = model(tx[idx])
                freq = nn.functional.one_hot(selected, 4).float().mean(0)
                loss = (pred - ty[idx]).square().mean() + args.balance * 4 * (freq * prob.mean(0)).sum()
            else:
                loss = (model(tx[idx]) - ty[idx]).square().mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            score = (predict(model, vx) - vy).square().mean().item()
        history.append({'epoch': epoch + 1, 'validation_mse': score})
        if score < best:
            best = score
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return history


def evaluate(model, dataset, args, device, directory, name):
    x, y, task = dataset
    settings = [('noise', s) for s in (0.25, 0.5, 1.0)] + [('shift', s) for s in (0.25, 0.5, 1.0)]
    settings += [(f, s) for f in ('mask', 'mask_blind') for s in (0.125, 0.25, 0.5)]
    settings += [('cue', s) for s in (0.25, 0.5, 0.75)]
    conditions = [(f, s, r) for f, s in settings for r in range(1 if f == 'shift' else args.perturb_reps)]
    rows, records = [], {}
    model.eval()
    with torch.no_grad():
        xd, yd = x.to(device), y.to(device)
        base = predict(model, xd)
        cleanroute = model.routing(xd) if isinstance(model, MoE) else None
        records.update(source_id=np.arange(len(x)), target=y.numpy(), task=task.numpy(), clean=base.cpu().numpy())
        if cleanroute is not None:
            records.update(clean_ids=cleanroute[0].cpu().numpy(), clean_weight=cleanroute[1].cpu().numpy())
        for condition_id, (family, strength, replicate) in enumerate(conditions):
            prefix = f'{family}_{strength}_r{replicate}'
            z = perturb(x, family, strength, 9000 + condition_id).to(device)
            noisy = predict(model, z)
            predictions = {'00': base, '11': noisy}
            if cleanroute is not None:
                noisyroute = model.routing(z)
                predictions['10'] = model(z, cleanroute)[0]
                predictions['01'] = model(xd, noisyroute)[0]
                idonly = (cleanroute[0], noisyroute[1], noisyroute[2])
                predictions['id_restore'] = model(z, idonly)[0]
                gen = torch.Generator(device=device).manual_seed(8000 + condition_id)
                random_ids = torch.randint(4, (len(x),), generator=gen, device=device)
                random_weights = noisyroute[2].gather(1, random_ids[:, None]).squeeze(1)
                predictions['random'] = model(z, (random_ids, random_weights, noisyroute[2]))[0]
                predictions['random_fixed_gate'] = model(z, (random_ids, noisyroute[1], noisyroute[2]))[0]
                switch = (cleanroute[0] != noisyroute[0]).float()
                records[f'{prefix}_noisy_ids'] = noisyroute[0].cpu().numpy()
            else:
                switch = None
            for regime in [-1, 0, 1, 2, 3]:
                idx = torch.ones(len(x), dtype=torch.bool, device=device) if regime == -1 else task.to(device) == regime
                scores = {key: (pred[idx] - yd[idx]).square().mean().item() for key, pred in predictions.items()}
                row = dict(model=name, seed=args.seed, family=family, strength=strength, replicate=replicate,
                           regime=regime, n=int(idx.sum()), switch_rate=None if switch is None else switch[idx].mean().item(), **scores)
                if cleanroute is not None:
                    row.update(expert_term=scores['10'] - scores['00'], routing_term=scores['11'] - scores['10'],
                               total=scores['11'] - scores['00'], interaction=scores['11'] - scores['10'] - scores['01'] + scores['00'])
                    assert abs(row['expert_term'] + row['routing_term'] - row['total']) < 1e-6
                rows.append(row)
            for key, pred in predictions.items():
                if key != '00':
                    records[f'{prefix}_{key}'] = pred.cpu().numpy()
        if cleanroute is not None:
            # Replace only the expert identity, retaining the original gate weight.
            replacement = []
            for expert in range(4):
                route = (torch.full_like(cleanroute[0], expert), cleanroute[1], cleanroute[2])
                pred = model(xd, route)[0]
                records[f'expert_replace_{expert}'] = pred.cpu().numpy()
                replacement.append([float((pred[task.to(device) == k] - yd[task.to(device) == k]).square().mean()) for k in range(4)])
                rerouted = cleanroute[2].clone()
                rerouted[:, expert] = -1
                alternate = rerouted.argmax(-1)
                records[f'expert_remove_{expert}'] = model(xd, (alternate, cleanroute[1], cleanroute[2]))[0].cpu().numpy()
            (directory / 'expert_replacement.json').write_text(json.dumps(replacement, indent=2))
            joint = np.zeros((4, 4))
            ids = cleanroute[0].cpu().numpy()
            np.add.at(joint, (task.numpy(), ids), 1)
            joint /= joint.sum()
            expected = joint.sum(1)[:, None] * joint.sum(0)[None, :]
            ok = joint > 0
            mi = float((joint[ok] * np.log(joint[ok] / expected[ok])).sum())
            (directory / 'specialization.json').write_text(json.dumps({'task_expert_joint': joint.tolist(), 'mutual_information_nats': mi}, indent=2))
    np.savez_compressed(directory / f'{name}_predictions.npz', **records)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--train', type=int, default=8000)
    p.add_argument('--val', type=int, default=2000)
    p.add_argument('--test', type=int, default=2000)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=0.002)
    p.add_argument('--balance', type=float, default=0.01)
    p.add_argument('--device', default='cpu')
    p.add_argument('--shortcut', action='store_true')
    p.add_argument('--perturb-reps', type=int, default=3)
    args = p.parse_args()
    if min(args.train, args.val, args.test) < 4 or min(args.epochs, args.batch, args.width, args.perturb_reps) < 1:
        p.error('positive dimensions and at least four items per split required')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    datasets, mean, std = normalize(data(args.train, 10000, args.shortcut, True),
                                   data(args.val, 20000, args.shortcut, True),
                                   data(args.test, 30000))
    metadata = dict(vars(args), torch=torch.__version__, python=sys.version, platform=platform.platform(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    target_mean=mean.tolist(), target_std=std.tolist(),
                    data_seeds={'train':10000, 'validation':20000, 'test':30000},
                    cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                    protocol_version=2, perturbation_seed_base=9000,
                    dataset_sha256=[hashlib.sha256(xx.numpy().tobytes()+yy.numpy().tobytes()).hexdigest() for xx,yy,_ in datasets],
                    status='running')
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2))
    rows, t0 = [], time.time()
    moe = MoE(args.width)
    costs = budget(moe)
    models = [('moe', moe),
              ('dense_active', matched_dense(costs['active_linear_macs_per_item'], 'macs')),
              ('dense_total', matched_dense(costs['parameters'], 'parameters'))]
    metadata['model_budgets'] = {name: budget(model) for name, model in models}
    for name, model in models:
        start = time.time()
        history = fit(model, datasets[0], datasets[1], args, device)
        torch.save(model.cpu().state_dict(), out / f'{name}.pt')
        model.to(device)
        rows.extend(evaluate(model, datasets[2], args, device, out, name))
        (out / f'{name}_training.json').write_text(json.dumps({'history': history, 'parameters': sum(x.numel() for x in model.parameters()),
                                                            'elapsed_seconds': time.time() - start}, indent=2))
        print(name, 'best validation MSE', min(x['validation_mse'] for x in history), flush=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metadata.update(status='complete', elapsed_seconds=time.time() - t0)
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2))
    print('Completed', out, metadata['elapsed_seconds'], 'seconds', flush=True)


if __name__ == '__main__':
    main()
