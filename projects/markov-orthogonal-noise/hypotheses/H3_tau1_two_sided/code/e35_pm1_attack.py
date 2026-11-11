"""E3.5 (new)  Why the +-1 hash noise proposed in the formulation of the hypothesis (claim H3.2) cannot give an upper bound (Prop. 4.9).

With eta = +-1 the pair difference is D = g + Delta (eta+ - eta-), with eta+ - eta- in {-2, 0, 2}.
If |g| <= 2 M gamma < Delta, the decoder g = D - 2 Delta round(D / (2 Delta)) is exact: the noise is removed.
Pre-registered prediction: ZO-SGD on decoded differences reproduces the noiseless trajectory to rounding
error (same direction stream), at every Delta > 2 M gamma; raw ZO-SGD with +-1 noise and with Beta(3,3)
noise of the same amplitude is much worse.  Refutation: decoded error differs from noiseless.
Logistic, d = 20, gamma = 0.05, Delta/Delta_ref in {10, 100} (2 M gamma = 8.9 Delta_ref), T = 1e5, 5 runs.
Run: python3 e35_pm1_attack.py  ->  data/e35.json
"""
import json, os, sys, time
import numpy as np
from common import Logistic, ZOSGD, Job

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
d, gamma, T = 20, 0.05, 10 ** 5
CPS = [10 ** 3, 3 * 10 ** 3, 10 ** 4, 3 * 10 ** 4, 10 ** 5]
prob = Logistic(d, 200, 1e-2, 0)
BUDGET, T_START = 240.0, time.time()


def zo_sgd(*a, name, **kw):
    job = Job(os.path.join(DATA, f'.job_e35_{name}.pkl'), lambda: ZOSGD(*a, **kw))
    if not job.step(BUDGET - (time.time() - T_START)):
        print(name, 'paused at t =', job.obj.t, flush=True)
        sys.exit(3)
    return job.obj.result()


Dref = prob.M * gamma / np.sqrt(d)
G0 = d * prob.M / np.sqrt(d - 1)
out = dict(d=d, gamma=gamma, T=T, checkpoints=CPS, Delta_ref=Dref, two_M_gamma=2 * prob.M * gamma, runs={})
out['runs']['none'] = zo_sgd(prob, gamma, 0.0, 'none', T, CPS, 5, seed=35000, G=G0, name='none').tolist()
for lev in (10, 100):
    Delta = lev * Dref
    out['runs'][f'pm1_decoded_{lev}'] = zo_sgd(prob, gamma, Delta, 'pm1', T, CPS, 5, seed=35000, G=G0,
                                               decode=True, name=f'dec{lev}').tolist()
    out['runs'][f'pm1_raw_{lev}'] = zo_sgd(prob, gamma, Delta, 'pm1', T, CPS, 5, seed=35000, name=f'pm{lev}').tolist()
    # Beta(3,3) hash noise with the same amplitude Delta (not decodable: continuous law)
    out['runs'][f'beta_raw_{lev}'] = zo_sgd(prob, gamma, Delta, 'beta', T, CPS, 5, seed=35000, name=f'beta{lev}').tolist()
    print(lev, flush=True)
json.dump(out, open(os.path.join(DATA, 'e35.json'), 'w'))
