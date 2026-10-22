"""Stage 2: MeZO fine-tuning (two-point ZO, Gaussian directions) on prose, with the loss VALUE
optionally rounded to bfloat16. The forward pass itself is fp32; only the returned scalar is rounded,
which isolates the 'rounded oracle value' mechanism.

modes: fp32 | rtn (round-to-nearest bf16) | sr (stochastic rounding bf16) | rtn_adapt (rtn + adaptive mu)
"""
import argparse, json, time, math, torch, numpy as np
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='fp32')
ap.add_argument('--mu', type=float, default=1e-3)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--steps', type=int, default=3000)
ap.add_argument('--bs', type=int, default=32)
ap.add_argument('--eval_every', type=int, default=150)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--tag', default='')
a = ap.parse_args()

code, ft_tr, ft_ev, V = load_corpora()
m = TinyLM(V); m.load_state_dict(torch.load(f'{OUT}/pretrained.pt')); m.eval()
params = [p for p in m.parameters()]
gb = torch.Generator().manual_seed(1000 + a.seed)   # batches: identical across modes
nr = np.random.default_rng(7 + a.seed)             # stochastic rounding coin flips


def perturb(scale, seed):
    gz = torch.Generator().manual_seed(seed)
    for p in params:
        p.add_(torch.randn(p.shape, generator=gz) * scale)


def evaluate():
    ge = torch.Generator().manual_seed(123)
    return float(np.mean([m(*get_batch(ft_ev, 64, ge)).item() for _ in range(10)]))


def oracle(v):
    if a.mode == 'fp32':
        return v
    if a.mode in ('rtn', 'rtn_adapt'):
        return bf16_rtn(v)
    if a.mode == 'sr':
        return bf16_sr(v, nr)
    raise ValueError(a.mode)


log = {'args': vars(a), 'eval': [], 'win': []}
mu = a.mu
ties = spans = 0; mus = []
t0 = time.time()
with torch.no_grad():
    log['eval'].append((0, evaluate()))
    for it in range(1, a.steps + 1):
        x, y = get_batch(ft_tr, a.bs, gb)
        zseed = 10_000_000 * (a.seed + 1) + it   # identical directions across modes
        perturb(+mu, zseed); lp = m(x, y).item()
        perturb(-2 * mu, zseed); lm = m(x, y).item()
        perturb(+mu, zseed)
        ulp = bf16_ulp((lp + lm) / 2)
        op, om = oracle(lp), oracle(lm)
        ties += (op == om)
        spans += abs(lp - lm) / ulp          # how many bf16 cells the stencil spans (true values)
        mus.append(mu)
        g = (op - om) / (2 * mu)
        perturb(-a.lr * g, zseed)            # theta <- theta - lr * g * z
        if a.mode == 'rtn_adapt':
            mu = min(mu * 1.5, 100 * a.mu) if abs(op - om) < 2 * ulp else max(a.mu, mu / 1.5)
        if it % a.eval_every == 0:
            ev = evaluate()
            if not math.isfinite(ev):
                print('diverged'); log['eval'].append((it, float('nan'))); break
            log['eval'].append((it, ev))
            log['win'].append({'it': it, 'tie_frac': ties / a.eval_every,
                               'mean_span_cells': spans / a.eval_every, 'mean_mu': float(np.mean(mus))})
            ties = spans = 0; mus = []
            print(f"{a.mode:9s} it {it:5d} eval {ev:.4f} tie {log['win'][-1]['tie_frac']:.2f} "
                  f"span {log['win'][-1]['mean_span_cells']:.2f} mu {log['win'][-1]['mean_mu']:.1e} "
                  f"({time.time()-t0:.0f}s)", flush=True)
name = a.tag or f'{a.mode}_mu{a.mu:g}_lr{a.lr:g}_s{a.seed}'
json.dump(log, open(f'{OUT}/runs/{name}.json', 'w'))
