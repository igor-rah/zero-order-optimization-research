"""E3.1 sensitivity: logistic d = 20 with (i) another problem instance, (ii) +-1 hash noise, (iii) gamma x 2,
(iv) l1 regression d = 20 with gamma x 5.  T <= 2e5, same Delta grid and estimator as e31_collapse.py.
(v) [added after the first version of the report] 'pm1_eqomega': the +-1 variant (ii) with every noise level
    multiplied by sqrt(2/7)/sqrt(2) = 0.378, so that the mean-square pair oscillation omega_bar equals that of
    the Beta(3,3) runs at the nominal levels; same seed as (ii) (common random numbers), so the only change
    with respect to (ii) is the level.  Question: is the poor fit of (ii) (a = 0.67) caused by saturation?
    Prediction: a within [0.4, 0.6] with the CI containing 1/2 on the all-points fit (if saturation explains it).
    Refutation: the same criterion as E3.1.  NOT a blind pre-registration: this configuration had already been run
    when this header was written.
    The JSON keeps the nominal levels in 'levels' and the factor in 'level_scale';
    Delta = levels * level_scale * Delta_ref.
Run: python3 e31_sensitivity.py  ->  data/e31s_<variant>.json
"""
import json, os, time, sys
import numpy as np
from common import Logistic, L1Reg, ZOSGD, Job, omega_bar

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
Tmax = 2 * 10 ** 5
CPS = [1000, 2000, 5000, 10 ** 4, 2 * 10 ** 4, 5 * 10 ** 4, 10 ** 5, 2 * 10 ** 5]
LEVELS = [0, 3, 10, 30, 100, 300]
NREP = 5
BUDGET, T_START = 240.0, time.time()
variants = [('instance1', 'logistic', 1, 0.05, 'beta'), ('pm1', 'logistic', 0, 0.05, 'pm1'),
            ('gamma2', 'logistic', 0, 0.10, 'beta'), ('l1_gamma5', 'l1', 0, 0.01, 'beta'),
            ('pm1_eqomega', 'logistic', 0, 0.05, 'pm1')]
LEVEL_SCALE = {'pm1_eqomega': np.sqrt(2 / 7) / np.sqrt(2)}     # omega_bar(pm1) * scale = omega_bar(beta)
SEED = {'pm1_eqomega': 31700 + len('pm1')}                      # same streams as the 'pm1' variant
for tag, name, inst, gamma, kind in variants:
    d = 20
    prob = Logistic(d, 200, 1e-2, inst) if name == 'logistic' else L1Reg(d, 200, inst)
    Dref = prob.M * gamma / np.sqrt(d)
    lev = np.repeat(LEVELS, NREP).astype(float)
    scale = LEVEL_SCALE.get(tag, 1.0)
    Delta = lev * scale * Dref
    G = d * prob.M / np.sqrt(d - 1) + d * omega_bar(kind, Delta) / (2 * gamma)
    final = os.path.join(DATA, f'e31s_{tag}.json')
    if os.path.exists(final):
        continue
    t0 = time.time()
    job = Job(os.path.join(DATA, f'.job_e31s_{tag}.pkl'),
              lambda: ZOSGD(prob, gamma, Delta, kind, Tmax, CPS, len(Delta), seed=SEED.get(tag, 31700 + len(tag))))
    if not job.step(BUDGET - (time.time() - T_START)):
        print(tag, 'paused at t =', job.obj.t, flush=True)
        sys.exit(3)
    err = job.obj.result()
    json.dump(dict(variant=tag, problem=name, d=d, gamma=gamma, noise=kind, R=prob.R, M=prob.M, f0=prob.f0,
                   fstar=prob.fstar, Delta_ref=Dref, levels=lev.tolist(), level_scale=scale, Delta=Delta.tolist(), G=G.tolist(),
                   checkpoints=CPS, err=err.tolist(), nrep=NREP),
              open(final, 'w'))
    print(tag, f'{time.time() - t0:.0f}s', flush=True)
