"""Numerical verification suite for the H1 report (numpy/scipy only).  Writes data/verify.json.

V1  Psi_d(t) = E_{u~Unif(B)} cos(t u_1)                         (Fourier transform of the ball average)
V2  Phi_d'(t) = -(t/d) Psi_d(t)                                  (sphere vs ball multiplier; Stokes in Fourier form)
V3  Ball-Stokes identity (d/gamma) E_e[g(x+gamma e) e] = grad g_hat_gamma(x)   (the planned 08_check.py):
    (a) smooth g (random waves), (b) rough g (grid-hash noise, discontinuous on hyperplanes, class G_gamma)
V4  |b(x,s)| <= sqrt(d) Delta/gamma, via E|e_1| = Gamma(d/2)/(sqrt(pi) Gamma((d+1)/2)) <= 1/sqrt(d)
V5  Poincare bound E|g_f|^2 <= d^2 M^2/(d-1) for the two-point estimator of an M-Lipschitz f
V6  Mixing: d(l*tmix) <= 2^{-l} and sum_{t>=1} d(t) <= 2 tmix - 1 (lazy chains);  sum_{t>=1} 2^{-floor(t/tau)} = 2tau-1
V7  Pointwise failure off G_gamma: g = 1{|y| = gamma} sign(y_1): identity fails at x = 0 only
V8  Two-radius exceptional set Q_d = {j_k/j_l}: generic ratios are not in Q_d; Q_d accumulates (dense)
V9  Explicit budget T(Delta) of Theorem 4.15: T = O(Delta^2) (log-log slope -> 2), value at Delta = 0
V10 Unbiasedness of the batched estimator under a lazy chain for both protocols (q = 2: pair in one state;
    q = 1: the two queries in consecutive states)
"""
import json, sys, os
import numpy as np
from scipy import integrate
from scipy.special import gammaln
sys.path.insert(0, os.path.dirname(__file__))
from common import Psi, Phi, bessel_zeros, sphere, cell_sign, hash_coefs

R = {}
def record(name, ok, **info):
    R[name] = dict(ok=bool(ok), **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()})
    print(('PASS ' if ok else 'FAIL ') + name, {k: v for k, v in info.items() if not isinstance(v, (list, np.ndarray))})

def ball(rng, n, d):
    return sphere(rng, n, d) * rng.random(n)[:, None] ** (1 / d)

# V1 ------------------------------------------------------------------------------------------
err = 0
for d in (2, 3, 5, 20):
    c = 1 / integrate.quad(lambda s: (1 - s * s) ** ((d - 1) / 2), -1, 1)[0]
    for t in (0.3, 1.7, 5.0, 11.3):
        q = c * integrate.quad(lambda s: np.cos(t * s) * (1 - s * s) ** ((d - 1) / 2), -1, 1, limit=200, epsabs=1e-14)[0]
        err = max(err, abs(q - Psi(d, np.array([t]))[0]))
record('V1_Psi_is_ball_FT', err < 1e-10, max_abs_err=err)

# V2 ------------------------------------------------------------------------------------------
err = 0
for d in (2, 3, 5, 20, 100):
    t = np.linspace(0.5, 60, 300); h = 1e-5
    lhs = (Phi(d, t + h) - Phi(d, t - h)) / (2 * h)
    err = max(err, np.max(np.abs(lhs + (t / d) * Psi(d, t))))
record('V2_PhiPrime_eq_minus_t_over_d_Psi', err < 1e-8, max_abs_err=err)

# V3 ------------------------------------------------------------------------------------------
rng = np.random.default_rng(3)
d, gam, n = 5, 0.7, 4_000_000
W = rng.standard_normal((6, d)) * 2.0; ph = rng.uniform(0, 2 * np.pi, 6)
g_s = lambda Y: np.cos(Y @ W.T + ph).sum(1) / 6
coef = hash_coefs(d, 7); hcell = 0.1 * gam
g_r = lambda Y: cell_sign(Y, hcell, np.uint64(42), coef)
x = rng.standard_normal(d) * 0.3
out = {}
for name, g in (('smooth', g_s), ('rough', g_r)):
    # LHS by Monte Carlo on the sphere (chunks)
    s1 = np.zeros(d); s2 = np.zeros(d)
    for _ in range(n // 500_000):
        e = sphere(rng, 500_000, d); v = g(x + gam * e)[:, None] * e * (d / gam)
        s1 += v.sum(0); s2 += (v * v).sum(0)
    lhs = s1 / n; lse = np.sqrt(s2 / n - lhs ** 2) / np.sqrt(n)
    # RHS: central finite differences of the ball average with common random numbers
    hfd = 0.02 * gam
    rhs = np.zeros(d); rse = np.zeros(d)
    for i in range(d):
        acc = [];
        for _ in range(n // 500_000):
            u = ball(rng, 500_000, d) * gam
            ei = np.zeros(d); ei[i] = hfd
            acc.append((g(x + ei + u) - g(x - ei + u)) / (2 * hfd))
        a = np.concatenate(acc); rhs[i] = a.mean(); rse[i] = a.std() / np.sqrt(a.size)
    if name == 'smooth':   # exact: grad g_hat = -sum Psi(gam|w|) sin(<w,x>+ph) w / 6
        exact = -(Psi(d, gam * np.linalg.norm(W, axis=1)) * np.sin(W @ x + ph) / 6) @ W
        z_exact = np.max(np.abs(lhs - exact) / lse)
    else:
        exact = None; z_exact = None
    z = np.max(np.abs(lhs - rhs) / np.sqrt(lse ** 2 + rse ** 2))
    out[name] = dict(lhs=lhs, rhs_fd=rhs, se_lhs=lse, se_rhs=rse, max_z=z, max_z_vs_exact=z_exact)
ok = out['smooth']['max_z'] < 4 and out['smooth']['max_z_vs_exact'] < 4 and out['rough']['max_z'] < 4
record('V3_ball_Stokes_MC_vs_FD', ok, smooth_max_z=float(out['smooth']['max_z']),
       smooth_max_z_exact=float(out['smooth']['max_z_vs_exact']), rough_max_z=float(out['rough']['max_z']),
       rough_lhs=out['rough']['lhs'], rough_rhs=out['rough']['rhs_fd'])

# V4 ------------------------------------------------------------------------------------------
vals = {}
ok = True
for d in (2, 5, 20, 100, 1000):
    Ee1 = np.exp(gammaln(d / 2) - gammaln((d + 1) / 2)) / np.sqrt(np.pi)
    vals[d] = [Ee1, 1 / np.sqrt(d), np.sqrt(2 / (np.pi * d))]
    ok &= Ee1 <= 1 / np.sqrt(d)
e = sphere(np.random.default_rng(4), 2_000_000, 20)
worst = 20 * np.mean(np.sign(e[:, 0]) * e[:, 0])      # (d/gamma) E[Delta sign(e_1) e_1], Delta = gamma = 1
record('V4_bias_bound_sqrt_d', ok and worst <= np.sqrt(20), E_abs_e1=vals, worst_bias_d20=worst, bound_d20=np.sqrt(20))

# V5 ------------------------------------------------------------------------------------------
rng = np.random.default_rng(5); rat = {}
for d in (2, 5, 20):
    fs = {'abs_x1': lambda Y: np.abs(Y[:, 0]), 'norm': lambda Y: np.linalg.norm(Y, axis=1),
          'max_x1_x2': lambda Y: np.maximum(Y[:, 0], Y[:, 1]), 'linear': lambda Y: Y[:, 0]}
    for nm, f in fs.items():
        e = sphere(rng, 1_000_000, d); x0 = np.zeros(d)
        g2 = (d / 2) ** 2 * (f(x0 + e) - f(x0 - e)) ** 2          # gamma = 1, M = 1, |e| = 1
        rat[f'{d}_{nm}'] = float(g2.mean() / (d ** 2 / (d - 1)))
record('V5_Poincare_second_moment', max(rat.values()) <= 1.0, ratio_to_bound=rat, max_ratio=max(rat.values()))

# V6 ------------------------------------------------------------------------------------------
ok = True; info = {}
for tau in (1, 2, 3, 7, 50):
    s = sum(2.0 ** (-np.floor(t / tau)) for t in range(1, 4000))
    ok &= abs(s - (2 * tau - 1)) < 1e-9
for alpha in (0.5, 0.9, 0.99, 0.999):
    K = 512; dt = lambda t: alpha ** t * (1 - 1 / K)
    tm = next(t for t in range(1, 10 ** 6) if dt(t) <= 0.25)
    ok &= all(dt(l * tm) <= 2.0 ** (-l) for l in range(1, 30))
    ssum = sum(dt(t) for t in range(1, 200 * tm))
    ok &= ssum <= 2 * tm - 1
    info[str(alpha)] = dict(tmix=tm, sum_d=ssum, bound=2 * tm - 1)
record('V6_mixing_sums', ok, lazy=info)

# V7 ------------------------------------------------------------------------------------------
d, gam = 5, 1.0
g = lambda Y: np.where(np.abs(np.linalg.norm(Y, axis=1) - gam) < 1e-9, np.sign(Y[:, 0]), 0.0)
rng = np.random.default_rng(7)
e = sphere(rng, 1_000_000, d)
at0 = (d / gam) * (g(0 * e + gam * e)[:, None] * e).mean(0)
xr = rng.standard_normal(d) * 0.3
atx = (d / gam) * (g(xr + gam * e)[:, None] * e).mean(0)
ballavg = g(ball(rng, 1_000_000, d) * gam).mean()
expected0 = d * np.exp(gammaln(d / 2) - gammaln((d + 1) / 2)) / np.sqrt(np.pi)
record('V7_pointwise_failure_off_G', abs(at0[0] - expected0) < 0.01 and np.abs(atx).max() == 0 and ballavg == 0,
       spherical_at_0=at0, expected_first=expected0, spherical_at_random_x=atx, ball_average=ballavg)

# V8 ------------------------------------------------------------------------------------------
d = 5
z = bessel_zeros(d / 2, 400)
info = {}
for K in (20, 100, 400):
    Q = (z[:K, None] / z[None, :K]).ravel()
    info[K] = {str(r): float(np.min(np.abs(Q - r))) for r in (np.sqrt(2), 1.5, np.pi / 2)}
ok = all(v > 0 for v in info[400].values()) and info[400][str(np.sqrt(2))] < info[20][str(np.sqrt(2))]
record('V8_two_radius_exceptional_set', ok, min_dist=info,
       note='distances >0 (generic ratios not in Q_d) and shrink with K (Q_d accumulates)')

# V9 ------------------------------------------------------------------------------------------
def budget(Delta, eps, d, tau, gamma, M, R):
    V = 2 * d ** 2 * M ** 2 / (d - 1) + (2 * d + 8 * tau - 3) * d * Delta ** 2 / gamma ** 2
    B = int(np.ceil(V / M ** 2))
    N = int(np.ceil(36.25 * M ** 2 * R ** 2 / eps ** 2))
    t0 = 0 if Delta == 0 else tau * max(0, int(np.ceil(np.log2(16 * R * np.sqrt(d) * Delta / (gamma * eps)))))
    return N * (B + t0), N, B, t0
d, tau, M, Rr, eps = 20, 16, 1.0, 1.0, 0.05; gamma = eps / (4 * M)
Ds = np.logspace(-4, 3, 15)
Ts = np.array([budget(D_, eps, d, tau, gamma, M, Rr)[0] for D_ in Ds])
sl = np.polyfit(np.log(Ds[-5:]), np.log(Ts[-5:]), 1)[0]
T0 = d * M ** 2 * Rr ** 2 / eps ** 2
record('V9_budget_polynomial', abs(sl - 2) < 0.02, slope_large_Delta=sl, T_at_0_over_T0=budget(0, eps, d, tau, gamma, M, Rr)[0] / T0,
       Delta=Ds, T=Ts)

# V10 -----------------------------------------------------------------------------------------
d, K, alpha, gam, B, reps = 10, 64, 1 - 1 / 16, 1.0, 400, 3000
rng = np.random.default_rng(10)
V = rng.standard_normal((K, d)); V /= np.linalg.norm(V, axis=1, keepdims=True); PH = rng.uniform(0, 2 * np.pi, K)
a = rng.standard_normal(d)                        # f(x) = <a,x>  =>  grad f_gamma = a
noise = lambda Y, st: np.where(st < K, 1.0, -1.0) * np.sign(np.sin(np.sqrt(d) * np.einsum('ij,ij->i', V[st % K], Y) / gam + PH[st % K]))
def chain(n):
    jump = rng.random(n) >= alpha; jump[0] = True
    draws = rng.integers(0, 2 * K, n)
    return draws[np.maximum.accumulate(np.where(jump, np.arange(n), 0))]
res = {}
for q in (2, 1):
    G = []
    for _ in range(reps):
        e = sphere(rng, B, d)
        st = chain(B * (3 - q))                   # q=2: B states; q=1: 2B states (one per call)
        sp, sm = (st, st) if q == 2 else (st[0::2], st[1::2])
        yp, ym = gam * e, -gam * e
        Op = yp @ a + noise(yp, sp); Om = ym @ a + noise(ym, sm)
        G.append(((d / (2 * gam)) * (Op - Om)) @ e / B)
    G = np.array(G); m = G.mean(0); se = G.std(0) / np.sqrt(reps)
    res[q] = dict(max_z=float(np.max(np.abs(m - a) / se)), B_times_mse=float(B * np.mean(np.sum((G - a) ** 2, 1))))
record('V10_unbiased_both_protocols', res[2]['max_z'] < 4.5 and res[1]['max_z'] < 4.5, q2=res[2], q1=res[1])

json.dump(R, open('data/verify.json', 'w'), indent=1, default=float)
print('ALL PASS' if all(v['ok'] for v in R.values()) else 'SOME FAIL')
