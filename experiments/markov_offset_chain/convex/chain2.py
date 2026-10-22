"""Rounding-offset chain on a convex problem with known optimum.
l2-logistic regression (d=50, n=500), f-values rounded to p=8 significant bits (bf16-like).
Two-point ZO-GD, B=5 directions on the sphere per step (same as zo_convex/exp.py).
Data modes:
  full      : deterministic f (offset chain driven only by the iterate)
  mb<tau>   : minibatch of size b=50, the SAME minibatch is used for both points (two-point
              feedback) and is reused for tau consecutive steps (lazy Markov data chain).
Logged: final accuracy f(x_bar)-f*, per-step centre offset c_k = frac(f_S(x_k)/ulp), tie fraction.
"""
import json, os, sys, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'rounded_oracle_convex'))
from exp import make_problem, round_p, ulp_p

d, n, p, T, B, b = 50, 500, 8, 10000, 5, 50
mus = np.logspace(-5, 0.5, 12)

def run(mode, tau, seed):
    f, fstar, xstar, L, lam = make_problem(d, n, seed)
    r = np.random.default_rng(seed); A = r.standard_normal((n, d)) / np.sqrt(d)
    w = r.standard_normal(d) * 2.0; y = np.sign(A @ w + 0.5 * r.standard_normal(n))
    M = -(y[:, None] * A)                                   # same data as make_problem
    def fS(X, idx):
        Z = M[idx] @ X
        return np.mean(np.logaddexp(0, Z), 0) + 0.5 * lam * np.sum(X * X, 0)
    rng = np.random.default_rng(10_000 + seed); drng = np.random.default_rng(20_000 + seed)
    K = len(mus); mu = mus.copy()
    x0 = np.random.default_rng(30_000 + seed).standard_normal(d); x0 /= np.linalg.norm(x0)
    X = np.repeat(x0[:, None], K, 1); Xbar = np.zeros((d, K)); cnt = 0
    eta = 0.5 / (L * (d / B + 1))
    C = np.zeros((T, K), np.float32); TIE = np.zeros((T, K), np.float32); HOLD = np.zeros((T, K), np.float32)
    idx = np.arange(n)
    for t in range(T):
        if mode == 'mb' and t % tau == 0:
            idx = drng.choice(n, b, replace=False)
        E = rng.standard_normal((d, B)); E /= np.linalg.norm(E, axis=0)
        P = (X[:, :, None] + mu[None, :, None] * E[:, None, :]).reshape(d, -1)
        Q = (X[:, :, None] - mu[None, :, None] * E[:, None, :]).reshape(d, -1)
        fp = fS(P, idx).reshape(K, B); fm = fS(Q, idx).reshape(K, B)
        rp, rm = round_p(fp, p), round_p(fm, p)
        fc = fS(X, idx); C[t] = (fc / ulp_p(fc, p)) % 1.0
        TIE[t] = np.mean(rp == rm, 1); HOLD[t] = np.all(rp == rm, 1)
        G = (d / (2 * mu))[:, None] * (rp - rm)
        X -= eta * (E @ G.T) / B
        if t >= T // 2:
            Xbar += X; cnt += 1
    floor = f(Xbar / cnt) - fstar
    lags = [1, 4, 16, 64, 256]
    Ce = C[:2000]
    rho = {l: np.abs(np.mean(np.exp(2j * np.pi * (Ce[l:] - Ce[:-l])), 0)).tolist() for l in lags}
    sF = np.std([fS(x0[:, None], drng.choice(n, b, replace=False))[0] for _ in range(400)]) / ulp_p(f(x0[:, None])[0], p)
    return dict(mode=mode, tau=tau, seed=seed, mus=mus.tolist(), floor=floor.tolist(),
                f0=float(f(x0[:, None])[0] - fstar), sF_over_u=float(sF), hold=HOLD.mean(0).tolist(), tie=TIE.mean(0).tolist(), rho=rho,
                C_head={str(k): C[:400, k].tolist() for k in range(K)})

if __name__ == '__main__':
    out = {}; t0 = time.time()
    for seed in (0, 1):
        for mode, tau in (('full', 0), ('mb', 1), ('mb', 16), ('mb', 256)):
            out[f'{mode}{tau}_{seed}'] = run(mode, tau, seed)
            print(mode, tau, seed, f'{time.time()-t0:.0f}s', flush=True)
    json.dump(out, open(os.path.join(HERE, 'chain2.json'), 'w'))
