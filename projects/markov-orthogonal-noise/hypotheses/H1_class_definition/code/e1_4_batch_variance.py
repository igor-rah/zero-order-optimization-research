"""E1.4 (new)  Second moment of the batched estimator of Algorithm 1 under a Markov chain: additivity d + tau.

Noise (state s = (k, sigma), k in [K], sigma = +-1):
    delta(x, s) = sigma * Delta * sign(sin(sqrt(d) <v_k, x>/gamma + phi_k)),
with random unit v_k and phases phi_k.  E_pi delta(x,.) = 0 for every x (antithetic sigma), so the class
condition holds for all radii, while at a fixed state the bias has the maximal order sqrt(d) Delta/gamma.
Chain: lazy resampling, stay with prob. alpha, otherwise redraw from pi (uniform on [K]x{+-1}).
Estimator at x = 0 with f = 0 (isolates the noise):
    g_bar = (1/B) sum_j (d/(2 gamma)) (delta(gamma e_j, s_j) - delta(-gamma e_j, s_j)) e_j,  s_1 ~ pi.
Exact prediction (no free parameters; derived in Sec. 5.4 of the report from the proof of Lemma 4.14):
    B * E|g_bar|^2 = A + S * [1 + 2 sum_{t=1}^{B-1} (1 - t/B) alpha^t],
    S = E_pi |b(s)|^2,  A = E|(d/2gamma)(delta_+ - delta_-) e|^2 - S,
computed by exact one-dimensional integrals over the law of e_1 (incomplete beta function).
Upper bound of Lemma 4.14:  B * E|g_bar|^2 <= (d + 8 tau_mix - 3) d Delta^2 / gamma^2.
Output: data/e1_4.json
"""
import json, sys, os, time
import numpy as np
from scipy.special import betaln
sys.path.insert(0, os.path.dirname(__file__))
from common import sphere, e1_cdf

DIMS = (10, 40, 160)
TAUS = (1, 4, 16, 64, 256)        # nominal 1/(1-alpha)
SEEDS = range(5)
K = 512
REPS = 300

def e1_moment_interval(d, l, r):
    """int_l^r u p(u) du for the density p of e_1 on S^{d-1}."""
    a = (d - 3) / 2
    c = np.exp(-betaln(0.5, (d - 1) / 2))
    F = lambda u: -(1 - u * u) ** (a + 1) / (2 * (a + 1))
    return c * (F(r) - F(l))

def exact_AS(d, phis):
    sd = np.sqrt(d)
    g, P = [], []
    for ph in phis:
        m = np.arange(np.floor((ph - sd) / np.pi) - 1, np.ceil((ph + sd) / np.pi) + 2)
        bp = np.unique(np.clip(np.concatenate([[-1, 1], (m * np.pi - ph) / sd]), -1, 1))
        mid = (bp[:-1] + bp[1:]) / 2
        sgn = np.sign(np.sin(sd * mid + ph))
        g.append(np.sum(sgn * e1_moment_interval(d, bp[:-1], bp[1:])))
        bp2 = np.unique(np.clip(np.concatenate([[-1, 1], (m * np.pi - ph) / sd, (ph - m * np.pi) / sd]), -1, 1))
        mid2 = (bp2[:-1] + bp2[1:]) / 2
        diff = np.sign(np.sin(sd * mid2 + ph)) != np.sign(np.sin(-sd * mid2 + ph))
        P.append(np.sum(diff * (e1_cdf(d, bp2[1:]) - e1_cdf(d, bp2[:-1]))))
    g, P = np.array(g), np.array(P)
    S = d ** 2 * np.mean(g ** 2)            # Delta = gamma = 1
    A = d ** 2 * np.mean(P) - S
    return A, S

def tmix(alpha):
    if alpha == 0:
        return 1
    dt = lambda t: alpha ** t * (1 - 1 / (2 * K))
    t = 1
    while dt(t) > 0.25:
        t += 1
    return t

def kappa(alpha, B):
    t = np.arange(1, B)
    return 1 + 2 * np.sum((1 - t / B) * alpha ** t)

def main():
    out = dict(meta=dict(K=K, REPS=REPS, seeds=list(SEEDS), taus=TAUS, dims=DIMS))
    t0 = time.time()
    for d in DIMS:
        rs = np.random.default_rng(99 + d)
        V = rs.standard_normal((K, d)); V /= np.linalg.norm(V, axis=1, keepdims=True)
        PH = rs.uniform(0, 2 * np.pi, K)
        A, S = exact_AS(d, PH)
        rows = []
        for tau in TAUS:
            alpha = 1 - 1 / tau
            tm = tmix(alpha)
            B = 32 * tm + 64
            pred = A + S * kappa(alpha, B)
            per_seed = []
            for seed in SEEDS:
                rng = np.random.default_rng(100_000 * d + 100 * tau + seed)
                acc = []
                for _ in range(REPS):
                    jump = rng.random(B) >= alpha; jump[0] = True
                    draws = rng.integers(0, 2 * K, B)
                    last = np.maximum.accumulate(np.where(jump, np.arange(B), 0))
                    st = draws[last]
                    k, sig = st % K, np.where(st < K, 1.0, -1.0)
                    e = sphere(rng, B, d)
                    p = np.einsum('ij,ij->i', V[k], e) * np.sqrt(d)
                    dp = np.sign(np.sin(p + PH[k])); dm = np.sign(np.sin(-p + PH[k]))
                    g = ((d / 2) * sig * (dp - dm))[:, None] * e
                    gb = g.mean(0)
                    acc.append(B * gb @ gb)
                acc = np.array(acc)
                per_seed.append(dict(mean=acc.mean(), se=acc.std(ddof=1) / np.sqrt(REPS)))
            rows.append(dict(tau_nominal=tau, alpha=alpha, tmix=tm, B=B, pred=pred,
                             bound=(d + 8 * tm - 3) * d, per_seed=per_seed))
            print(f'd={d} tau={tau} tmix={tm} B={B} pred={pred:.1f} got={np.mean([p["mean"] for p in per_seed]):.1f} '
                  f'{time.time()-t0:.0f}s', flush=True)
        out[str(d)] = dict(A=A, S=S, rows=rows)
    json.dump(out, open('data/e1_4.json', 'w'))

if __name__ == '__main__':
    main()
