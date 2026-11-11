"""E3.4 (new)  The hard instance of Thm 4.6 against ZO-SGD (a member of the class A_gamma).

Instance: f_v(x) = a <v, x> on B_R (R = 1), v uniform on {-1,1}^d, rough hash noise with U ~ 2Beta(3,3)-1
(Fisher information I = 10), Delta = 1, gamma = 0.01, i.i.d. states.
Pre-registered prediction (Thm 4.6 / Lemma 4.8 with I = 10): for every algorithm of the class, and so for
ZO-SGD with any step,   E f_v(xhat) - f* >= LB(a) := (a R / (4 sqrt d)) * (d - 2 gamma a sqrt(d I T)/Delta)_+ .
At a* = sqrt(d) Delta / (4 gamma sqrt(I T)) this is a* R sqrt(d)/8.
Refutation: mean error (over v and runs, 95% CI) below LB(a) for some a.
Also reported: the error of ZO-SGD relative to LB (how loose the bound is for this method).
Grid: d in {20, 80}, T in {1e3, 1e4, 1e5}, a = a* x {1/16, 1/4, 1, 4, 16}, step scale in {0.1, 1, 10}
(best per a reported, which is still a single algorithm per a), 20 runs = 5 seeds x 4 random v.
Run: python3 e34_hard_instance.py  ->  data/e34.json
"""
import json, os, time, sys
import numpy as np
from common import Linear, ZOSGD, Job, omega_bar, BETA33_FISHER

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
I = BETA33_FISHER
gamma, Delta, R = 0.01, 1.0, 1.0
AMULT = [1 / 16, 1 / 4, 1, 4, 16]
ETAS = [0.1, 1.0, 10.0]
NRUN = 20
BUDGET, T_START = 240.0, time.time()
res = []
for d in (20, 80):
    for T in (10 ** 3, 10 ** 4, 10 ** 5):
        t0 = time.time()
        astar = np.sqrt(d) * Delta / (4 * gamma * np.sqrt(I * T))
        rng = np.random.default_rng(34000 + d + T)
        avals, etas, V = [], [], []
        for am in AMULT:
            for es in ETAS:
                Vv = rng.choice([-1.0, 1.0], size=(NRUN, d))
                V.append(Vv)
                avals += [am * astar] * NRUN
                etas += [es] * NRUN
        V = np.concatenate(V)
        avals = np.array(avals)
        K = len(avals)
        prob = Linear(V, avals, R)
        G = d * prob.M / np.sqrt(d - 1) + d * omega_bar('beta', Delta) / (2 * gamma)
        job = Job(os.path.join(DATA, f'.job_e34_{d}_{T}.pkl'),
                  lambda: ZOSGD(prob, gamma, Delta, 'beta', T, [T], K, seed=34500 + d + T, G=G,
                                eta_scale=np.array(etas), Rball=R))
        if not job.step(BUDGET - (time.time() - T_START)):
            print(d, T, 'paused at t =', job.obj.t, flush=True)
            sys.exit(3)
        err = job.obj.result()[0]
        LB = (avals * R / (4 * np.sqrt(d))) * np.maximum(d - 2 * gamma * avals * np.sqrt(d * I * T) / Delta, 0)
        res.append(dict(d=d, T=T, astar=astar, a=avals.tolist(), eta_scale=etas, err=err.tolist(),
                        LB=LB.tolist(), seconds=time.time() - t0))
        print(d, T, f'{time.time() - t0:.0f}s', flush=True)
json.dump(dict(gamma=gamma, Delta=Delta, R=R, I=I, amult=AMULT, etas=ETAS, nrun=NRUN, rows=res),
          open(os.path.join(DATA, 'e34.json'), 'w'))
