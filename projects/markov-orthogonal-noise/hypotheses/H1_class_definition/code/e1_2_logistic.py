"""E1.2  Logistic regression with rough Markov noise (tau = 1): the target is reached for every Delta,
and the budget needed grows as Delta^2 (no error floor).

Problem: l2-regularised logistic regression as in experiments/rounded_oracle_convex/exp.py
(make_problem copied verbatim, d = 10, n = 50, lambda = 1e-2; instance seed = run seed).
Domain: ball B_R, R = 2|x*|; iterates projected on B_{R-gamma}; x_1 = 0.
Noise:  state s = (k, sigma), k ~ uniform 64-bit key, sigma = +-1, drawn i.i.d. every round (tau = 1);
        delta(x, s) = sigma * Delta * eta(hash(cell(x), k)),  eta = +-1,  cells: cubes of side h = gamma/(2 sqrt d).
        E_pi delta(x, .) = 0 for every x (antithetic sigma) => m = 0, class condition for all radii.
Method: Algorithm 1 of the report with t_0 = 0 (tau = 1): per iteration B rounds, each round one direction
        e_j and the pair x +- gamma e_j answered in the same state; step eta = 0.5/L; N = 300 iterations;
        output x_bar = mean of the last N/2 iterates (Polyak-Ruppert tail averaging).
Grid:   Delta in {0, 1, 10, 100} * Delta_ref, Delta_ref = M gamma / sqrt d;  B in {4^0, ..., 4^b(Delta)},
        b = 7, 7, 8, 9 (the upper end grows with Delta only to bound the CPU cost).
Control (outside the class): frozen state (tau = infinity), noise Delta*sign(sin(sqrt d <v,x>/gamma + phi)),
        Delta = 10 Delta_ref -> a bias of order sqrt(d) Delta / gamma that batching cannot remove.
Pre-registered (README, report Sec. 5): primary target eps = 0.03; prediction: T_eps(Delta) finite for all
Delta and log-log slope of T_eps over {10,100} Delta_ref equal to 2 +- 0.3; refutation: an error floor that
does not decrease with B (as in the control).
Output: data/e1_2.json
"""
import json, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import sphere, cell_sign, hash_coefs

D, NS, LAM = 10, 50, 1e-2
GAMMA = 0.05
N_IT = 300
BGRID = [4 ** i for i in range(10)]
BMAX = {0: 7, 1: 7, 10: 8, 100: 9, 'control': 8}   # largest exponent run per Delta (cost); see README
DMULT = [0, 1, 10, 100]
SEEDS = range(5)
CHUNK = 1 << 15

def make_problem(d, n, seed, lam=1e-2):          # verbatim from rounded_oracle_convex/exp.py (f vectorised by rows)
    r = np.random.default_rng(seed)
    A = r.standard_normal((n, d)) / np.sqrt(d)
    w = r.standard_normal(d) * 2.0
    y = np.sign(A @ w + 0.5 * r.standard_normal(n))
    M = -(y[:, None] * A)
    x = np.zeros(d)
    for _ in range(60):
        z = M @ x; s = 1 / (1 + np.exp(-z))
        g = M.T @ s / n + lam * x
        H = (M.T * (s * (1 - s))) @ M / n + lam * np.eye(d)
        x -= np.linalg.solve(H, g)
    def f(X):                                     # X: (k, d) -> (k,)
        return np.mean(np.logaddexp(0, X @ M.T), 1) + 0.5 * lam * np.sum(X * X, 1)
    fstar = f(x[None])[0]
    L = np.linalg.eigvalsh(M.T @ M / n).max() / 4 + lam
    return f, fstar, x, L, M

def run(seed, Delta, B, mode, prob, coef, extra):
    f, fstar, xstar, L, Mmat, R = prob
    rng = np.random.default_rng([seed, B, int(round(Delta * 1e9)), {'markov': 0, 'frozen': 1}[mode]])
    h = GAMMA / (2 * np.sqrt(D))
    eta = 0.5 / L
    x = np.zeros(D); xs = np.zeros(D); cnt = 0
    Rp = R - GAMMA
    for it in range(N_IT):
        g = np.zeros(D)
        for c0 in range(0, B, CHUNK):
            b = min(CHUNK, B - c0)
            E = sphere(rng, b, D)
            P = x + GAMMA * E; Q = x - GAMMA * E
            vp, vm = f(P), f(Q)
            if Delta > 0:
                if mode == 'markov':
                    keys = rng.integers(0, 2 ** 63, size=b, dtype=np.int64).astype(np.uint64)
                    sig = rng.choice([-1.0, 1.0], size=b)
                    vp = vp + sig * Delta * cell_sign(P, h, keys, coef)
                    vm = vm + sig * Delta * cell_sign(Q, h, keys, coef)
                else:
                    v, ph = extra
                    vp = vp + Delta * np.sign(np.sin(np.sqrt(D) * (P @ v) / GAMMA + ph))
                    vm = vm + Delta * np.sign(np.sin(np.sqrt(D) * (Q @ v) / GAMMA + ph))
            g += ((D / (2 * GAMMA)) * (vp - vm)) @ E
        g /= B
        x = x - eta * g
        nx = np.linalg.norm(x)
        if nx > Rp:
            x *= Rp / nx
        if it >= N_IT // 2:
            xs += x; cnt += 1
    return float(f((xs / cnt)[None])[0] - fstar)

def main():
    out = dict(meta=dict(d=D, n=NS, lam=LAM, gamma=GAMMA, N=N_IT, Bgrid=BGRID, Dmult=DMULT, seeds=list(SEEDS)))
    coef = hash_coefs(D)
    t0 = time.time()
    for seed in SEEDS:
        f, fstar, xstar, L, Mmat = make_problem(D, NS, seed, LAM)
        R = 2 * float(np.linalg.norm(xstar))
        Mlip = float(np.linalg.norm(Mmat, 2) / np.sqrt(NS) + LAM * R)
        Dref = Mlip * GAMMA / np.sqrt(D)
        prob = (f, fstar, xstar, L, Mmat, R)
        rv = np.random.default_rng(555 + seed); v = rv.standard_normal(D); v /= np.linalg.norm(v)
        extra = (v, rv.uniform(0, 2 * np.pi))
        res = dict(fstar=float(fstar), R=R, M=Mlip, L=float(L), Dref=float(Dref), gap0=float(np.log(2) - fstar),
                   err={}, control={})
        for k in DMULT:
            res['err'][str(k)] = [run(seed, k * Dref, B, 'markov', prob, coef, extra) for B in BGRID[:BMAX[k] + 1]]
            print(f'seed {seed} Delta={k}Dref {np.round(res["err"][str(k)], 5)} {time.time()-t0:.0f}s', flush=True)
        res['control']['10'] = [run(seed, 10 * Dref, B, 'frozen', prob, coef, extra) for B in BGRID[:BMAX['control'] + 1]]
        print(f'seed {seed} control {np.round(res["control"]["10"], 5)} {time.time()-t0:.0f}s', flush=True)
        out[str(seed)] = res
        json.dump(out, open('data/e1_2.json', 'w'))

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'time':
        coef = hash_coefs(D); f, fs, xs, L, Mm = make_problem(D, NS, 0)
        R = 2 * np.linalg.norm(xs); prob = (f, fs, xs, L, Mm, R)
        for B in (64, 1024, 4096):
            t = time.time(); e = run(0, 0.085, B, 'markov', prob, coef, None); print(B, e, time.time() - t)
    elif len(sys.argv) == 1:
        main()


def collapse():
    """E1.2b: direct scaling test without interpolation.  If the budget needed scales as Delta^2, then at
    B' = B * (Delta'/Delta)^2 the error distribution at Delta' must equal that at (Delta, B).
    Pairs: (100 Dref, 4^k) vs (10 Dref, round(4^k/100)), k = 6..9  [reuses the main-run values at 100 Dref]."""
    D2 = json.load(open('data/e1_2.json'))
    coef = hash_coefs(D)
    out = dict(meta=dict(note='Delta=10Dref at B=round(4^k/100), k=6..9; compare with e1_2.json Delta=100Dref at 4^k'))
    t0 = time.time()
    for seed in SEEDS:
        f, fstar, xstar, L, Mmat = make_problem(D, NS, seed, LAM)
        R = 2 * float(np.linalg.norm(xstar))
        prob = (f, fstar, xstar, L, Mmat, R)
        Dref = D2[str(seed)]['Dref']
        Bs = [int(round(4 ** k / 100)) for k in range(6, 10)]
        out[str(seed)] = dict(B=Bs, err10=[run(seed, 10 * Dref, B, 'markov', prob, coef, None) for B in Bs],
                              err100=D2[str(seed)]['err']['100'][6:10])
        print('collapse', seed, np.round(out[str(seed)]['err10'], 4), np.round(out[str(seed)]['err100'], 4),
              f'{time.time()-t0:.0f}s', flush=True)
    json.dump(out, open('data/e1_2_collapse.json', 'w'))


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'collapse':
    collapse()
