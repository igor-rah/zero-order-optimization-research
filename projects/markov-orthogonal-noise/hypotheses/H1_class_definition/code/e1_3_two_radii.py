"""E1.3  A wave tuned to a Bessel zero for radius gamma1 is tested at a second radius gamma2.

m(x) = Delta cos(<w,x>+phi), |w| = j_{d/2,k}/gamma1 (k = 1,2,3), so m is in the class for gamma1.
The gradient-estimate bias at radius gamma2 (two-point estimator, Prop. 4.13 of the report):
    beta(gamma2) = (d/gamma2) E_e[m(x+gamma2 e) e] = grad m_hat_{gamma2}(x) = -Delta Psi_d(gamma2|w|) sin(.) w,
so along w and at sin(.) = 1:  beta = -Delta |w| Psi_d(gamma2 |w|).
Prediction: |beta| = Delta|w||Psi_d(gamma2|w|)|, zero exactly at gamma2/gamma1 = j_l/j_k (ratios of zeros),
nonzero at ratios in general position (e.g. sqrt 2).  WLOG gamma1 = Delta = 1, w = |w| e_(1).
Seeds 0..4, N directions per seed.  Output: data/e1_3.json
"""
import json, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import Psi, bessel_zeros, sphere

DIMS = (5, 20, 100)
SEEDS = range(5)
N = 1_000_000
CHUNK = 250_000

def main():
    out = dict(meta=dict(N=N, seeds=list(SEEDS)))
    t0 = time.time()
    for d in DIMS:
        z = bessel_zeros(d / 2, 12)
        base = np.logspace(np.log10(0.05), np.log10(3.2), 70)
        dips = np.array([z[l] / z[k] for k in range(3) for l in range(12) if 0.05 <= z[l] / z[k] <= 3.2])
        r = np.unique(np.concatenate([base, dips, [np.sqrt(2)]]))
        W = z[:3]                                   # |w| for k = 1,2,3 (gamma1 = 1)
        TT = (r[None, :] * W[:, None]).ravel()      # gamma2*|w| for all (k, r)
        fac = (d / r)[None, :].repeat(3, 0).ravel() # d/gamma2
        theory = (-W[:, None] * Psi(d, r[None, :] * W[:, None])).ravel()
        est, se = [], []
        for seed in SEEDS:
            rng = np.random.default_rng(7000 + 10 * d + seed)
            s1 = np.zeros(TT.size); s2 = np.zeros(TT.size); done = 0
            while done < N:
                k = min(CHUNK, N - done)
                e1 = sphere(rng, k, d)[:, 0]
                c = -np.sin(np.outer(e1, TT)) * e1[:, None]        # m(x+g2 e) e_1 at sin=1
                s1 += c.sum(0); s2 += (c * c).sum(0); done += k
            mu = s1 / N; sd = np.sqrt(s2 / N - mu ** 2)
            est.append((fac * mu).tolist()); se.append((fac * sd / np.sqrt(N)).tolist())
            print(f'd={d} seed={seed} {time.time()-t0:.0f}s', flush=True)
        out[str(d)] = dict(zeros=z.tolist(), ratios=r.tolist(), w=W.tolist(), theory=theory.tolist(),
                           est=est, se=se, shape=[3, len(r)])
    json.dump(out, open('data/e1_3.json', 'w'))

if __name__ == '__main__':
    main()
