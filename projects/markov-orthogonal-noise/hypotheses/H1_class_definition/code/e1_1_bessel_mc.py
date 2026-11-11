"""E1.1  Monte Carlo check of the ball-Stokes identity for Bessel plane waves.

m(x) = Delta*cos(<w,x> + phi).  Claim (Lemma 4.3 and Cor. 4.7 of the report):
    E_e[m(x+gamma e) e] = (gamma/d) grad m_hat_gamma(x) = -(gamma/d) Delta Psi_d(gamma|w|) sin(<w,x>+phi) w,
so the vector vanishes iff J_{d/2}(gamma|w|) = 0.
WLOG gamma = Delta = 1, w = t * e_(1) (sampling is rotation invariant), x chosen with <w,x>+phi = pi/2.
Part A: sweep of t, along-w component vs theory -(t/d)Psi_d(t), plus chi^2 test of the orthogonal part.
Part B: first three zeros, large n, chi^2_d test of the whole vector against 0.
Part C: superpositions of 10 waves with |w_i| at the first three zeros (random directions, phases),
        tested at 20 random points; positive control: the same waves detuned to 0.9*zero.
Seeds 0..4.  Output: data/e1_1.json
"""
import json, sys, time, os
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from common import Psi, bessel_zeros, sphere

DIMS = (5, 20, 100)
SEEDS = range(5)
N_SWEEP = 400_000          # directions per seed in the sweep (common random numbers across t)
N_ZERO = 2_000_000         # directions per seed at the zeros
N_SUP = 400_000            # directions per (seed, point) for superpositions
CHUNK = 200_000

def mc_vector(rng, d, n, fun):
    """Estimate E[fun(e) e] (vector) and per-component standard errors; fun maps (k,d)->(k,ncol)."""
    s1 = s2 = None; done = 0
    while done < n:
        k = min(CHUNK, n - done)
        e = sphere(rng, k, d)
        c = fun(e)                                   # (k, ncol)
        a = e.T @ c                                  # (d, ncol)
        b = (e * e).T @ (c * c)
        s1 = a if s1 is None else s1 + a
        s2 = b if s2 is None else s2 + b
        done += k
    mean = s1 / n
    var = s2 / n - mean ** 2
    return mean, np.sqrt(var / n)

def main():
    out = dict(meta=dict(N_SWEEP=N_SWEEP, N_ZERO=N_ZERO, N_SUP=N_SUP, seeds=list(SEEDS), dims=DIMS))
    t0 = time.time()
    for d in DIMS:
        z = bessel_zeros(d / 2, 3)
        ts = np.unique(np.concatenate([np.linspace(0.25, z[2] + 2.0, 57), z]))
        res = dict(zeros=z.tolist(), t=ts.tolist(), theory=(-(ts / d) * Psi(d, ts)).tolist(),
                   along=[], se_along=[], chi2_orth=[], zero_vec=[], sup=[], sup_detuned=[])
        for seed in SEEDS:
            rng = np.random.default_rng(1000 * d + seed)
            # Part A: along component at <w,x>+phi = pi/2  ->  m(x+e) = cos(pi/2 + t e_1) = -sin(t e_1)
            mean, se = mc_vector(rng, d, N_SWEEP, lambda e: -np.sin(np.outer(e[:, 0], ts)))
            res['along'].append(mean[0].tolist()); res['se_along'].append(se[0].tolist())
            res['chi2_orth'].append(((mean[1:] / se[1:]) ** 2).sum(0).tolist())   # ~ chi2_{d-1} each t
            # Part B: zeros, whole vector against 0 (random phase point a = <w,x>+phi)
            a = rng.uniform(0, 2 * np.pi)
            mean, se = mc_vector(rng, d, N_ZERO, lambda e: np.cos(a + np.outer(e[:, 0], z)))
            chi2 = ((mean / se) ** 2).sum(0)
            res['zero_vec'].append(dict(norm=np.linalg.norm(mean, axis=0).tolist(),
                                        noise_norm=np.sqrt((se ** 2).sum(0)).tolist(),
                                        chi2=chi2.tolist(), p=stats.chi2.sf(chi2, d).tolist()))
            # Part C: superposition of 10 waves (amplitudes Delta/10), tuned vs detuned
            for key, scale in (('sup', 1.0), ('sup_detuned', 0.9)):
                W = rng.standard_normal((10, d)); W /= np.linalg.norm(W, axis=1, keepdims=True)
                W *= (z[np.arange(10) % 3] * scale)[:, None]
                ph = rng.uniform(0, 2 * np.pi, 10)
                X = rng.standard_normal((20, d))
                A = X @ W.T + ph                                    # (20 points, 10 waves)
                f = lambda e: (np.cos(A[None, :, :] + (e @ W.T)[:, None, :]).sum(2) / 10)  # (k,20)
                mean, se = mc_vector(rng, d, N_SUP, f)
                # theory: E[m(x+e)e] = -(1/d) sum_i a_i Psi_d(|w_i|) sin(<w_i,x>+phi_i) w_i
                th = -(1 / d) * (np.sin(A) * Psi(d, np.linalg.norm(W, axis=1))[None, :] / 10) @ W  # (20,d)
                chi2 = (((mean.T - th) / se.T) ** 2).sum(1)
                chi2_0 = ((mean.T / se.T) ** 2).sum(1)
                res[key].append(dict(chi2_vs_theory=chi2.tolist(), p_vs_theory=stats.chi2.sf(chi2, d).tolist(),
                                     chi2_vs_zero=chi2_0.tolist(), p_vs_zero=stats.chi2.sf(chi2_0, d).tolist(),
                                     theory_norm=np.linalg.norm(th, axis=1).tolist(),
                                     est_norm=np.linalg.norm(mean, axis=0).tolist(),
                                     noise_norm=np.sqrt((se ** 2).sum(0)).tolist()))
            print(f'd={d} seed={seed} {time.time()-t0:.0f}s', flush=True)
        out[str(d)] = res
    os.makedirs('data', exist_ok=True)
    json.dump(out, open('data/e1_1.json', 'w'))

if __name__ == '__main__':
    main()
