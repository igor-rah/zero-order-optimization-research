"""E3.1  Collapse in Delta/sqrt(T) for ZO-SGD (two-point, radius gamma, batch 1, tau = 1).

Pre-registered (formulation of the hypothesis, fixed before the study, + this header, written before the run):
  setup      logistic (d in {20, 80}, n = 200, lambda = 1e-2, gamma = 0.05) and l1 regression
             (d in {20, 80}, n = 200, gamma = 0.002); rough grid-cell hash noise with U ~ 2Beta(3,3)-1
             (cells of diameter gamma < 2 gamma, fresh state every round); Delta/Delta_ref in
             {0, 3, 10, 30, 100, 300}, Delta_ref = M gamma / sqrt(d); 5 independent runs per cell;
             checkpoints T in [1e3, 1e6]; step eta_t = R/(G sqrt t), G = dM/sqrt(d-1) + d*omega_bar/(2 gamma).
  prediction in the noise-dominated region (Delta >= 10 Delta_ref and error >= 5x the noiseless error at the
             same T and <= 0.5 (f(0) - f*)) the error depends on (Delta, T) only through Delta/sqrt(T):
             fitted log err = c + alpha log Delta + beta log T gives a := -beta/alpha = 1/2;
             and err <= 3 R G / sqrt(T) + smoothing floor (Thm 4.2 anytime version).
  refutation a outside [0.4, 0.6] with its 95% bootstrap CI excluding 1/2.
Revision note (after the first version of the report): the clause "and <= 0.5 (f(0) - f*)" in the prediction above defines the
  'progress' subset of make_figs.py, while the report's primary ('all') fit does not use it, and an earlier
  version of the report described the progress fit as added after the run.  The archive does not record which text came
  first.  The revised report therefore claims neither fit as blind: it reports both ('all' = the selection of
  report section 5.1 without the progress clause; 'progress' = the selection of this header), each against the
  report's criterion and against the criterion fixed before the study (collapse in Delta/T^a with a != 1/2
  refutes).  The header text above is left as archived.
Run: python3 e31_collapse.py [Tmax]   ->  data/e31_<problem>_d<d>.json
"""
import sys, json, time, os
import numpy as np
from common import Logistic, L1Reg, ZOSGD, Job, omega_bar

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
Tmax = int(float(sys.argv[1])) if len(sys.argv) > 1 else 10**6
CPS = sorted({int(c) for c in [1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6] if c <= Tmax} | {Tmax})
LEVELS = [0, 3, 10, 30, 100, 300]
NREP = 5
only = sys.argv[2] if len(sys.argv) > 2 else None
BUDGET = 240.0   # seconds per invocation; the job is pickled and resumed on the next call (exit code 3)
T_START = time.time()

configs = [('logistic', 20, 0.05), ('logistic', 80, 0.05), ('l1', 20, 0.002), ('l1', 80, 0.002)]
for name, d, gamma in configs:
    tag = f'{name}_d{d}'
    if only and only != tag:
        continue
    prob = Logistic(d, 200, 1e-2, 0) if name == 'logistic' else L1Reg(d, 200, 0)
    Dref = prob.M * gamma / np.sqrt(d)
    lev = np.repeat(LEVELS, NREP).astype(float)
    Delta = lev * Dref
    K = len(Delta)
    G = d * prob.M / np.sqrt(d - 1) + d * omega_bar('beta', Delta) / (2 * gamma)
    final = os.path.join(DATA, f'e31_{tag}.json')
    if os.path.exists(final) and json.load(open(final)).get('checkpoints', [0])[-1] == CPS[-1]:
        continue
    t0 = time.time()
    job = Job(os.path.join(DATA, f'.job_e31_{tag}_{Tmax}.pkl'),
              lambda: ZOSGD(prob, gamma, Delta, 'beta', Tmax, CPS, K, seed=31000 + d + (0 if name == 'logistic' else 7)))
    if not job.step(BUDGET - (time.time() - T_START)):
        print(tag, 'paused at t =', job.obj.t, flush=True)
        sys.exit(3)
    err = job.obj.result()
    out = dict(problem=name, d=d, n=200, gamma=gamma, R=prob.R, M=prob.M, fstar=prob.fstar, f0=prob.f0,
               Delta_ref=Dref, levels=lev.tolist(), Delta=Delta.tolist(), G=G.tolist(), checkpoints=CPS,
               err=err.tolist(), nrep=NREP, seconds=time.time() - t0, noise='beta', cell_diameter=gamma)
    json.dump(out, open(final, 'w'))
    print(tag, f'{time.time() - t0:.0f}s', flush=True)
