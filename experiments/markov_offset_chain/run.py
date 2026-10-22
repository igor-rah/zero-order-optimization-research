"""Markov-data pilot. MeZO (fp32 / bf16-RTN loss value) and first-order SGD on TinyLM,
with a *lazy* data chain: each minibatch is reused for tau consecutive steps
(the lazy Markov chain of Prokhorov et al., 2026, mixing time ~ tau).
Per-step logging of the true two-point values, the bf16 grid offset of the stencil
centre c_k = frac(mean/ulp) and ties, to study the rounding-offset chain.
"""
import argparse, json, time, math, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'rounded_oracle_mezo'))
import torch, numpy as np
from common import load_corpora, get_batch, TinyLM, bf16_rtn, bf16_ulp, OUT as PRE

ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='fp32')          # fp32 | rtn | sgd
ap.add_argument('--tau', type=int, default=1)
ap.add_argument('--mu', type=float, default=1e-3)
ap.add_argument('--lr', type=float, default=5e-4)
ap.add_argument('--steps', type=int, default=2000)
ap.add_argument('--bs', type=int, default=32)
ap.add_argument('--eval_every', type=int, default=100)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--tag', default='')
a = ap.parse_args()

code, ft_tr, ft_ev, V = load_corpora()
m = TinyLM(V); m.load_state_dict(torch.load(f'{PRE}/pretrained.pt')); m.eval()
params = [p for p in m.parameters()]
gb = torch.Generator().manual_seed(1000 + a.seed)


def perturb(scale, seed):
    gz = torch.Generator().manual_seed(seed)
    for p in params:
        p.add_(torch.randn(p.shape, generator=gz) * scale)


def evaluate():
    ge = torch.Generator().manual_seed(123)
    with torch.no_grad():
        return float(np.mean([m(*get_batch(ft_ev, 64, ge)).item() for _ in range(10)]))


log = {'args': vars(a), 'eval': [(0, evaluate())], 'step': {k: [] for k in ('lp', 'lm', 'op', 'om', 'ulp', 'bid')}}
t0 = time.time(); bid = -1
for it in range(1, a.steps + 1):
    if (it - 1) % a.tau == 0:
        x, y = get_batch(ft_tr, a.bs, gb); bid += 1
    if a.mode == 'sgd':
        m.zero_grad(set_to_none=True)
        loss = m(x, y); loss.backward()
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p.add_(p.grad, alpha=-a.lr)
        lp = lm = loss.item(); op = om = lp
    else:
        with torch.no_grad():
            zseed = 10_000_000 * (a.seed + 1) + it
            perturb(+a.mu, zseed); lp = m(x, y).item()
            perturb(-2 * a.mu, zseed); lm = m(x, y).item()
            perturb(+a.mu, zseed)
            op, om = (bf16_rtn(lp), bf16_rtn(lm)) if a.mode == 'rtn' else (lp, lm)
            g = (op - om) / (2 * a.mu)
            perturb(-a.lr * g, zseed)
    S = log['step']
    S['lp'].append(lp); S['lm'].append(lm); S['op'].append(op); S['om'].append(om)
    S['ulp'].append(bf16_ulp((lp + lm) / 2)); S['bid'].append(bid)
    if it % a.eval_every == 0:
        ev = evaluate()
        if not math.isfinite(ev):
            log['eval'].append((it, float('nan'))); print('diverged'); break
        log['eval'].append((it, ev))
        print(f"{a.mode} tau={a.tau} it {it} eval {ev:.4f} ({time.time()-t0:.0f}s)", flush=True)
name = a.tag or f'{a.mode}_tau{a.tau}_mu{a.mu:g}_lr{a.lr:g}_s{a.seed}'
json.dump(log, open(os.path.join(HERE, 'runs', f'{name}.json'), 'w'))
print('saved', name, f'{time.time()-t0:.0f}s')
