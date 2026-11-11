"""Checks V1-V17 of the H3 report (symbolic with sympy, numeric with numpy/scipy).
Run: python3 verify.py  ->  data/verify.json   (all checks print PASS/FAIL)
"""
import json, os
import numpy as np
import sympy as sp
from scipy import integrate, stats
from common import HashNoise, Phi_sphere, unit_sphere, Logistic, omega_bar

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
R = {}


def rec(key, ok, **kw):
    kw = {k: (float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v) for k, v in kw.items()}
    R[key] = dict(ok=bool(ok), **kw)
    print(('PASS ' if ok else 'FAIL ') + key, kw)


t = sp.symbols('t', real=True)
# V1 Fisher information and normalisation of the two densities
p_cos = sp.cos(sp.pi * t / 2) ** 2
p_b = sp.Rational(15, 16) * (1 - t ** 2) ** 2
I_cos = sp.simplify(sp.integrate(sp.diff(sp.sqrt(p_cos).rewrite(sp.cos), t) ** 2 * 4, (t, -1, 1)))
I_cos = sp.integrate(4 * (sp.pi / 2) ** 2 * sp.sin(sp.pi * t / 2) ** 2, (t, -1, 1))   # sqrt(p)=cos(pi t/2)
I_b = sp.integrate(sp.simplify(sp.diff(p_b, t) ** 2 / p_b), (t, -1, 1))
rec('V1_fisher', sp.simplify(I_cos - sp.pi ** 2) == 0 and I_b == 10
    and sp.integrate(p_cos, (t, -1, 1)) == 1 and sp.integrate(p_b, (t, -1, 1)) == 1,
    I_cos=str(I_cos), I_beta=str(I_b))
# V2 variances
v_cos = sp.simplify(sp.integrate(t ** 2 * p_cos, (t, -1, 1)))
v_b = sp.integrate(t ** 2 * p_b, (t, -1, 1))
rec('V2_variances', sp.simplify(v_cos - (sp.Rational(1, 3) - 2 / sp.pi ** 2)) == 0 and v_b == sp.Rational(1, 7),
    var_cos=str(v_cos), var_beta=str(v_b))

# V3 Hellinger bound H^2(b) <= b^2 I / 4 (location shift), and for the difference of two independent copies
def sq_hell(dens, b, lo=-1, hi=1):
    f = lambda x: (np.sqrt(max(dens(x - b), 0)) - np.sqrt(max(dens(x), 0))) ** 2
    return integrate.quad(f, lo, hi + b, limit=400, points=[1, -1 + b])[0]

dc = lambda x: np.cos(np.pi * x / 2) ** 2 if abs(x) <= 1 else 0.0
db = lambda x: 15 / 16 * (1 - x * x) ** 2 if abs(x) <= 1 else 0.0
worst = []
for dens, I in ((dc, np.pi ** 2), (db, 10.0)):
    for b in (1e-3, 1e-2, 0.1, 0.3, 1.0):
        h2 = sq_hell(dens, b)
        worst.append(h2 / (b * b * I / 4))
rec('V3_hellinger_shift', max(worst) <= 1 + 1e-6, max_ratio=max(worst), ratio_small_b=worst[0])
# difference of two copies: density of U - U' by convolution (Beta case) -> bound must still hold (data processing)
xs = np.linspace(-2, 2, 8001)
dx = xs[1] - xs[0]
pb = np.array([db(x) for x in np.linspace(-1, 1, 4001)])
conv = np.convolve(pb, pb[::-1]) * (2 / 4000)
xc = np.linspace(-2, 2, len(conv))
ratios = []
for b in (0.01, 0.1, 0.5):
    sh = np.interp(xc - b, xc, conv, left=0, right=0)
    h2 = np.trapezoid((np.sqrt(sh) - np.sqrt(conv)) ** 2, xc) + b * 0
    ratios.append(h2 / (b * b * 10 / 4))
rec('V4_hellinger_difference', max(ratios) <= 1.0, ratios=ratios)

# V5 hash noise: median of 5 hashed uniforms is Beta(3,3); pair points in different cells -> independent
rng = np.random.default_rng(5)
d, gam = 20, 0.05
hn = HashNoise(d, gam, 'beta', rng)
N = 100000
x = rng.uniform(-1, 1, (N, d))
e = unit_sphere(rng, N, d)
key = rng.integers(0, 2**63, size=N, dtype=np.int64).astype(np.uint64)
up, um = hn.eta(x + gam * e, key), hn.eta(x - gam * e, key)
ks = stats.kstest((up + 1) / 2, stats.beta(3, 3).cdf)
same = (hn.cells(x + gam * e) == hn.cells(x - gam * e)).all(axis=1).sum()
rec('V5_hash_noise', ks.pvalue > 0.01 and abs(np.corrcoef(up, um)[0, 1]) < 0.01 and same == 0
    and abs(up.mean()) < 0.005, ks_p=ks.pvalue, corr=np.corrcoef(up, um)[0, 1], pairs_in_same_cell=int(same),
    mean=up.mean(), var=up.var())

# V6 separation lemma: a(<v,x> + R sqrt d) >= a R/(2 sqrt d) #{j: v_j x_j >= 0} on B_R; and sqrt d - sqrt(d-i) >= i/(2 sqrt d)
okk = all(np.sqrt(dd) - np.sqrt(dd - i) >= i / (2 * np.sqrt(dd)) - 1e-12 for dd in range(1, 400) for i in range(dd + 1))
worst = np.inf
for dd in (2, 5, 20, 80):
    v = rng.choice([-1.0, 1.0], size=dd)
    X = unit_sphere(rng, 20000, dd) * rng.uniform(0, 1, (20000, 1)) ** (1 / dd)
    X[:5000] = -v / np.sqrt(dd) + 0.3 * X[:5000]
    X = X / np.maximum(1, np.linalg.norm(X, axis=1, keepdims=True))
    gap = X @ v + np.sqrt(dd)
    W = (X * v >= 0).sum(1)
    worst = min(worst, (gap - W / (2 * np.sqrt(dd))).min())
rec('V6_separation', okk and worst >= -1e-12, min_slack=worst)

# V7 sequential Bhattacharyya bound + TV <= sqrt(1 - BC^2) on a two-round adaptive toy (exact enumeration)
def toy(seed):
    r = np.random.default_rng(seed)
    nq, no = 3, 4
    A1 = r.dirichlet(np.ones(nq))
    A2 = r.dirichlet(np.ones(nq), size=(nq, no))            # adaptive second query
    P = r.dirichlet(np.ones(no), size=nq); Q = r.dirichlet(np.ones(no), size=nq)
    bc = lambda p, q: np.sqrt(p * q).sum()
    BC = TVv = 0.0
    xbar1 = 1 - sum(A1[a] * bc(P[a], Q[a]) for a in range(nq))
    xbar2 = 0.0
    for a in range(nq):
        for o in range(no):
            xbar2 = max(xbar2, 1 - sum(A2[a, o, b] * bc(P[b], Q[b]) for b in range(nq)))
            for b in range(nq):
                for o2 in range(no):
                    pp = A1[a] * P[a, o] * A2[a, o, b] * P[b, o2]
                    qq = A1[a] * Q[a, o] * A2[a, o, b] * Q[b, o2]
                    BC += np.sqrt(pp * qq); TVv += abs(pp - qq) / 2
    return BC, TVv, (1 - xbar1) * (1 - xbar2)
oks = []
for s in range(200):
    BC, TVv, lb = toy(s)
    oks.append(BC >= lb - 1e-12 and TVv <= np.sqrt(1 - BC ** 2) + 1e-12)
rec('V7_sequential_BC', all(oks), n_cases=len(oks))

# V8 E_e sin^2(k e_1) = (1 - Phi_d(2k))/2 (quadrature over the law of e_1) and Phi_d(t) = E cos(t e_1)
from scipy.special import gammaln
errs = []
for dd in (3, 20, 80):
    c = np.exp(gammaln(dd / 2) - gammaln((dd - 1) / 2) - 0.5 * np.log(np.pi))
    dens = lambda s: c * (1 - s * s) ** ((dd - 3) / 2)
    for k in (0.1, 1.0, 5.0, 30.0):
        lhs = integrate.quad(lambda s: np.sin(k * s) ** 2 * dens(s), -1, 1, limit=400)[0]
        errs.append(abs(lhs - (1 - Phi_sphere(dd, 2 * k)[0]) / 2))
rec('V8_sphere_multiplier', max(errs) < 1e-8, max_abs_err=max(errs))

# V9 constants
dd = np.arange(2, 10001)
rec('V9_constants', (np.sqrt(24) / 2 - np.sqrt(dd / (dd - 1)) >= 1).all(),
    min_factor_C24=float((np.sqrt(24) / 2 - np.sqrt(dd / (dd - 1))).min()), c_upper_cos=32 * np.pi,
    c_upper_beta=32 * np.sqrt(10), c_any=16 * np.sqrt(2) * np.pi, gap_omega=32 * np.pi * np.sqrt(2 * (1 / 3 - 2 / np.pi ** 2)) * 2 / 2)

# V10 rounding with a uniform offset: P(tie) = 1 - frac, E[(D delta)^2] = u^2 r'(1 - r')
u = 1.0
ok10, rows10 = True, []
for g in (0.0, 0.1, 0.37, 0.5, 0.8, 1.3, 2.6):
    c = rng.uniform(0, u, 400000)
    F1 = c + g; F0 = c
    Q1, Q0 = np.round(F1 / u) * u, np.round(F0 / u) * u
    Dd = (Q1 - F1) - (Q0 - F0)
    rp = (g / u) % 1
    tie = (Q1 == Q0).mean()
    m2 = (Dd ** 2).mean()
    rows10.append([g, tie, m2, u * u * rp * (1 - rp)])
    ok10 &= abs(m2 - u * u * rp * (1 - rp)) < 3e-3 and (g >= u or abs(tie - (1 - g / u)) < 3e-3) and np.abs(Dd).max() <= u
rec('V10_rounding', ok10, rows=rows10)

# V11 second moment of the batch-1 two-point estimator (Lemma 4.1): E||Y||^2 <= G^2, logistic + beta noise
prob = Logistic(20, 200, 1e-2, 0)
gam, D = 0.05, 30 * prob.M * 0.05 / np.sqrt(20)
N = 200000
X0 = prob.xstar[None] + 0.5 * unit_sphere(rng, 1, 20)
e = unit_sphere(rng, N, 20)
yp, ym = X0 + gam * e, X0 - gam * e
hn = HashNoise(20, gam, 'beta', rng)
key = rng.integers(0, 2**63, size=N, dtype=np.int64).astype(np.uint64)
Dv = prob.f(yp) - prob.f(ym) + D * (hn.eta(yp, key) - hn.eta(ym, key))
EY2 = ((20 / (2 * gam)) ** 2 * Dv ** 2).mean()
G = 20 * prob.M / np.sqrt(19) + 20 * omega_bar('beta', D) / (2 * gam)
rec('V11_second_moment', EY2 <= G ** 2, EY2=EY2, G2=G ** 2, ratio=EY2 / G ** 2)

# V12 Lipschitz-noise corollary: f + delta(.,s) is (M+Lambda)-Lipschitz -> E||Y||^2 <= d^2 (M+L)^2/(d-1)
ell = 0.2
v = unit_sphere(rng, N, 20); ph = rng.uniform(0, 2 * np.pi, N)
Dv = prob.f(yp) - prob.f(ym) + D * (np.sin((yp * v).sum(1) / ell + ph) - np.sin((ym * v).sum(1) / ell + ph))
EY2 = ((20 / (2 * gam)) ** 2 * Dv ** 2).mean()
bound = 400 * (prob.M + D / ell) ** 2 / 19
rec('V12_lipschitz_noise', EY2 <= bound, EY2=EY2, bound=bound)

# V13 the +-1 decoder is exact when |g| < Delta
g = rng.uniform(-0.9, 0.9, 100000)
eta = rng.choice([-1.0, 1.0], size=(2, 100000))
Dd = g + (eta[0] - eta[1])
rec('V13_pm1_decoder', np.abs(Dd - 2 * np.round(Dd / 2) - g).max() < 1e-12)

# V14 convolution does not increase Fisher information: I(U - U') <= I(U) = 10 (numeric, Beta case)
p = conv
dp = np.gradient(p, xc)
mask = p > 1e-12
Iconv = np.trapezoid(np.where(mask, dp ** 2 / np.where(mask, p, 1), 0), xc)
rec('V14_fisher_difference', Iconv <= 10, I_difference=Iconv)

# ---------------------------------------------------------------- V15-V17
# V15 Cor. 4.3(b) with the idle first round: D_low(24 T0) = (M gamma/sqrt d)(sqrt(6 (1 - 1/(24 T0))) - sqrt(d/(d-1)));
#     eps < MR/8 gives T0 = d M^2 R^2/eps^2 > 64 d; the factor is increasing in T0, so its minimum over T0 > 64 d
#     is approached at T0 = 64 d; minimum over 2 <= d <= 1e4.
dd = np.arange(2, 10001, dtype=float)
fac15 = np.sqrt(6 * (1 - 1 / (24 * 64 * dd))) - np.sqrt(dd / (dd - 1))
dsym = sp.symbols('d', positive=True)
# symbolic: eps sqrt(T-1)/(2R) at T = 24 T0, in units of sqrt(d) M:  sqrt(24 T0 - 1)/(2 sqrt(T0))
Ms, Rs, es = sp.symbols('M R epsilon', positive=True)
T0e = dsym * Ms ** 2 * Rs ** 2 / es ** 2
lhs = sp.simplify(es * sp.sqrt(24 * T0e - 1) / (2 * Rs) / (sp.sqrt(dsym) * Ms))
target = sp.sqrt(6 * (1 - 1 / (24 * T0e)))
ok15 = sp.simplify(lhs ** 2 - target ** 2) == 0 and fac15.min() >= 1.034
rec('V15_cp_constant_idle_round', ok15, min_factor=float(fac15.min()), argmin_d=float(dd[fac15.argmin()]),
    factor_without_idle=float(np.sqrt(6) - np.sqrt(2)))

# V16 Cor. 4.17 (range-normalised bandit rates).  Budget for the bandit route: T - 2 >= 4 C_b d^k S^2/eps^2 with
#     S = 2MR + 4M gamma + 4 Delta; solved for Delta; the two crossovers.
Cb, k, T, D_, g_ = sp.symbols('C_b k T Delta gamma', positive=True)
S_ = 2 * Ms * Rs + 4 * Ms * g_ + 4 * D_
Dband = sp.solve(sp.Eq(4 * Cb * dsym ** k * S_ ** 2 / es ** 2, T - 2), D_)
Dband = [sol for sol in Dband if sol.has(sp.sqrt(T - 2)) and sp.simplify(sol.diff(T)).subs(
    {Cb: 1, dsym: 20, k: 3, T: 10 ** 9, es: 0.1, Ms: 1, Rs: 1, g_: 0.025}) > 0][0]
claimed = es * sp.sqrt(T - 2) / (8 * sp.sqrt(Cb) * dsym ** (k / 2)) - Ms * Rs / 2 - Ms * g_
ok16a = sp.simplify(Dband - claimed) == 0
# first crossover: eps sqrt(T-2)/(16 sqrt(C_b) d^{k/2}) > 8 pi eps^2 sqrt(T)/(d M R)  <=>  MR/eps > 128 pi sqrt(C_b) d^{k/2-1} sqrt(T/(T-2))
x_ = sp.symbols('x', positive=True)                     # x = MR/eps
cross1 = sp.solve(sp.Eq(sp.sqrt(T - 2) / (16 * sp.sqrt(Cb) * dsym ** (k / 2)), 8 * sp.pi * sp.sqrt(T) / (dsym * x_)), x_)[0]
ok16b = sp.simplify(cross1 - 128 * sp.pi * sp.sqrt(Cb) * dsym ** (k / 2 - 1) * sp.sqrt(T) / sp.sqrt(T - 2)) == 0
# second crossover: eps sqrt(T)/(8 sqrt(C_b) d^{k/2}) <= eps^2 sqrt(T-1)/(16 d M R)  <=>  MR/eps <= (sqrt(C_b)/2) d^{k/2-1} sqrt((T-1)/T)
cross2 = sp.solve(sp.Eq(sp.sqrt(T) / (8 * sp.sqrt(Cb) * dsym ** (k / 2)), sp.sqrt(T - 1) / (16 * dsym * x_)), x_)[0]
ok16c = sp.simplify(cross2 - sp.sqrt(Cb) / 2 * dsym ** (k / 2 - 1) * sp.sqrt((T - 1) / T)) == 0
# clean form: for T - 2 >= 256 C_b d^k (MR/eps)^2 and M gamma <= MR/2, Delta_band >= eps sqrt(T-2)/(16 sqrt(C_b) d^{k/2})
ok16d = True
for d_ in (2, 20, 80):
    for kk in (3.0, 3.5):
        for xr in (1.5, 10.0, 1e3):
            Tm = 2 + 256 * d_ ** kk * xr ** 2
            eps_, M_, R_ = 1.0, 1.0, xr
            gam = eps_ / (4 * M_)
            db = eps_ * np.sqrt(Tm - 2) / (8 * d_ ** (kk / 2)) - M_ * R_ / 2 - M_ * gam
            ok16d &= db >= eps_ * np.sqrt(Tm - 2) / (16 * d_ ** (kk / 2)) - 1e-12
rec('V16_bandit_rescaled', ok16a and ok16b and ok16c and ok16d, Delta_band=str(Dband), cross1=str(cross1),
    cross2=str(cross2))

# V17 round-1 bias under a non-stationary initial law (Thm. 4.2, why ZO idles in round 1):
#     delta(x, s) = Delta s sign(x_1), s = +-1, pi uniform (E_pi delta = 0), s_1 = +1, y = 0:
#     E[Y_1]_1 = (d/gamma) Delta E|e_1|,  E|e_1| = Gamma(d/2)/(sqrt(pi) Gamma((d+1)/2)).
from scipy.special import gammaln as _gl
rows17, ok17 = [], True
for d_ in (20, 80):
    gam, Dl, N17 = 0.05, 1.0, 200000
    e17 = unit_sphere(rng, N17, d_)
    Y1 = (d_ / (2 * gam)) * Dl * (np.sign(e17[:, 0]) - np.sign(-e17[:, 0]))
    mc = (Y1 * e17[:, 0]).mean()
    se = (Y1 * e17[:, 0]).std() / np.sqrt(N17)
    exact = d_ / gam * Dl * np.exp(_gl(d_ / 2) - _gl((d_ + 1) / 2)) / np.sqrt(np.pi)
    rows17.append([d_, mc, se, exact, np.sqrt(2 * d_ / np.pi) * Dl / gam])
    ok17 &= abs(mc - exact) < 4 * se
rec('V17_round1_bias', ok17, rows=rows17)

json.dump(R, open(os.path.join(DATA, 'verify.json'), 'w'), indent=1)
print('all pass:', all(v['ok'] for v in R.values()))
