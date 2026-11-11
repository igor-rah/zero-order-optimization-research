"""E3.2  Cost of noise vs its spatial scale ell (same amplitude Delta).

Noise  delta(x, s) = Delta sin(<v_s, x>/ell + phi_s), v_s ~ Unif(S^{d-1}), phi_s ~ U[0, 2 pi)  (E_pi delta = 0).
Pre-registered:
  (a) estimator level (exact, no free parameter): the noise part of the two-point estimator has
      E||n||^2 = (d^2 Delta^2 / (4 gamma^2)) * (1 - Phi_d(2 gamma/ell))   (Lemma 4.10),
      Phi_d = sphere multiplier.  Refutation: deviation beyond Monte Carlo error.
  (b) optimisation level: logistic regression, gamma = 0.05, Delta = 30 Delta_ref, T = 1e5, ZO-SGD with a
      common step (tuned to the rough limit ell -> 0); the excess error err(ell) - err(no noise), normalised by
      its value at ell = 0.01 gamma, follows 1 - Phi_d(2 gamma/ell) (within a factor 2 where it is >= 0.05).
      The formulation of the hypothesis predicted min{1, gamma/ell}^2; the two differ by the knee position
      (ell ~ gamma/sqrt(d) vs ell ~ gamma).  Refutation (formulation of the hypothesis): the cost does not depend on ell.
Run: python3 e32_smooth_noise.py  ->  data/e32a.json, data/e32b_d<d>.json
"""
import json, os, time, sys
import numpy as np
from common import Logistic, ZOSGD, Job, Phi_sphere, unit_sphere

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
RATIOS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
NREP = 5
BUDGET, T_START = 240.0, time.time()


def run(name, factory):
    job = Job(os.path.join(DATA, f'.job_e32_{name}.pkl'), factory)
    if not job.step(BUDGET - (time.time() - T_START)):
        print(name, 'paused at t =', job.obj.t, flush=True)
        sys.exit(3)
    return job.obj.result()


# ---------------------------------------------------------------- (a) estimator level
res_a = {}
gamma, Delta = 1.0, 1.0
DONE_A = os.path.exists(os.path.join(DATA, 'e32a.json'))
fine = np.logspace(-2, 2.5, 28)
for d in (() if DONE_A else (20, 80)):
    rows = []
    for rr in fine:
        ell = rr * gamma
        vals = []
        for rep in range(NREP):
            rng = np.random.default_rng(32000 + 100 * d + rep + int(1000 * np.log10(rr) + 5000))
            N = 200000
            e = unit_sphere(rng, N, d)
            v = unit_sphere(rng, N, d)
            ph = rng.uniform(0, 2 * np.pi, N)
            x = np.zeros(d)   # by rotation/translation invariance in law, x is irrelevant
            Dd = Delta * (np.sin(gamma * (e * v).sum(1) / ell + ph) - np.sin(-gamma * (e * v).sum(1) / ell + ph))
            n2 = (d / (2 * gamma)) ** 2 * Dd ** 2
            vals.append([n2.mean(), n2.std() / np.sqrt(N)])
        vals = np.array(vals)
        rows.append(dict(ratio=rr, mean=vals[:, 0].tolist(), se=vals[:, 1].tolist(),
                         theory=(d ** 2 * Delta ** 2 / (4 * gamma ** 2)) * (1 - Phi_sphere(d, 2 * gamma / ell)[0])))
    res_a[str(d)] = rows
if not DONE_A:
    json.dump(dict(gamma=gamma, Delta=Delta, N=200000, nrep=NREP, rows=res_a), open(os.path.join(DATA, 'e32a.json'), 'w'))
    print('e32a done', flush=True)

# ---------------------------------------------------------------- (b) optimisation level
T = 10 ** 5
CPS = [10 ** 4, 3 * 10 ** 4, 10 ** 5]
for d in (20, 80):
    if os.path.exists(os.path.join(DATA, f'e32b_d{d}.json')):
        continue
    t0 = time.time()
    gamma = 0.05
    prob = Logistic(d, 200, 1e-2, 0)
    Dref = prob.M * gamma / np.sqrt(d)
    Delta = 30 * Dref
    G_common = d * prob.M / np.sqrt(d - 1) + d * Delta / (2 * gamma)        # rough sinusoid limit: omega_bar = Delta
    ells = np.repeat(np.r_[RATIOS, 1e9] * gamma, NREP)                      # 1e9: smooth limit (sanity)
    Dl = np.full(len(ells), Delta)
    err_sin = run(f'sin{d}', lambda: ZOSGD(prob, gamma, Dl, 'sin', T, CPS, len(ells), seed=32500 + d, G=G_common, ell=ells))
    err_none = run(f'none{d}', lambda: ZOSGD(prob, gamma, 0.0, 'none', T, CPS, NREP, seed=32600 + d, G=G_common))
    # rough hash noise with the same mean-square oscillation as the rough sinusoid: omega_bar = Delta
    Db = Delta / np.sqrt(2 / 7)
    err_beta = run(f'beta{d}', lambda: ZOSGD(prob, gamma, Db, 'beta', T, CPS, NREP, seed=32700 + d, G=G_common))
    out = dict(d=d, gamma=gamma, Delta=Delta, Delta_ref=Dref, R=prob.R, M=prob.M, T=T, checkpoints=CPS,
               ratios=RATIOS + [1e9], ells=ells.tolist(), nrep=NREP, G_common=G_common,
               err_sin=err_sin.tolist(), err_none=err_none.tolist(), err_beta=err_beta.tolist(),
               theory_cost=[float(1 - Phi_sphere(d, 2 / r)[0]) for r in RATIOS + [1e9]],
               seconds=time.time() - t0)
    json.dump(out, open(os.path.join(DATA, f'e32b_d{d}.json'), 'w'))
    print('e32b', d, f'{time.time() - t0:.0f}s', flush=True)
