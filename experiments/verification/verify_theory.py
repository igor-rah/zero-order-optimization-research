"""Numerical verification of every closed-form statement used in the paper.
Prints max discrepancies; exits non-zero if any check fails."""
import numpy as np, sympy as sp, json, sys
from scipy.stats import norm
from scipy import integrate
rng = np.random.default_rng(0)
OK = True
def check(name, err, tol):
    global OK
    flag = err <= tol
    OK &= flag
    print(f"[{'PASS' if flag else 'FAIL'}] {name}: err={err:.3e} (tol {tol:.0e})")
    return err

Q = lambda v, u: u * np.round(v / u)
def SR(v, u, r):
    lo = np.floor(v / u) * u
    return lo + u * (r.random(v.shape) < (v - lo) / u)

res = {}
# ---- 1. tie probability, RTN, random offset: P = (1-delta)_+
u = 1.0
for delta in [0.0, 0.1, 0.37, 0.8, 1.0, 1.6]:
    U = rng.random(2_000_000) * u
    p = np.mean(Q(U, u) == Q(U + delta, u))
    check(f"RTN tie prob delta={delta}", abs(p - max(0, 1 - delta)), 2e-3)
# distribution of D = (Q(b)-Q(a))/u : mean delta, var (d-n)(n+1-d)
for delta in [0.3, 1.45, 2.7]:
    U = rng.random(2_000_000)
    D = (Q(U + delta, 1) - Q(U, 1))
    n = np.floor(delta)
    check(f"RTN offset-mean delta={delta}", abs(D.mean() - delta), 3e-3)
    check(f"RTN offset-var delta={delta}", abs(D.var() - (delta - n) * (n + 1 - delta)), 3e-3)

# ---- 2. SR tie probability = autocorrelation of tent (cubic B-spline)
y, d_ = sp.symbols('y delta', real=True)
tent = lambda t: sp.Piecewise((1 - sp.Abs(t), sp.Abs(t) <= 1), (0, True))
# delta in [0,1]: integrate over y in [-1, 1-delta]... do by pieces with delta symbolic
I01 = sp.integrate((1 + y) * (1 + y + d_), (y, -1, -d_)) + sp.integrate((1 + y) * (1 - y - d_), (y, -d_, 0)) \
    + sp.integrate((1 - y) * (1 - y - d_), (y, 0, 1 - d_))
I01 = sp.simplify(sp.expand(I01))
I12 = sp.simplify(sp.integrate((1 + y) * (1 - y - d_), (y, -1, 1 - d_)))
print("SR tie, 0<=delta<=1 :", sp.factor(I01), "  expected 2/3 - d^2 + d^3/2 ->", sp.simplify(I01 - (sp.Rational(2, 3) - d_**2 + d_**3 / 2)))
print("SR tie, 1<=delta<=2 :", sp.factor(I12), "  expected (2-d)^3/6 ->", sp.simplify(I12 - (2 - d_)**3 / 6))
def Bsp(dl):
    dl = np.abs(dl)
    return np.where(dl <= 1, 2 / 3 - dl**2 + dl**3 / 2, np.where(dl <= 2, (2 - dl)**3 / 6, 0.0))
for delta in [0.0, 0.25, 0.7, 1.0, 1.3, 1.9, 2.4]:
    U = rng.random(2_000_000) * 3
    p = np.mean(SR(U, 1, rng) == SR(U + delta, 1, rng))
    check(f"SR tie prob delta={delta}", abs(p - Bsp(delta)), 2e-3)
# SR offset-averaged second moment: delta^2 + 1/3
for delta in [0.0, 0.4, 1.7]:
    U = rng.random(2_000_000)
    D = SR(U + delta, 1, rng) - SR(U, 1, rng)
    check(f"SR offset second moment delta={delta}", abs(np.mean(D**2) - (delta**2 + 1 / 3)), 5e-3)
# RTN offset-averaged second moment: delta^2 + frac(1-frac)
for delta in [0.2, 0.9, 2.3]:
    U = rng.random(2_000_000)
    D = Q(U + delta, 1) - Q(U, 1)
    fr = delta - np.floor(delta)
    check(f"RTN offset second moment delta={delta}", abs(np.mean(D**2) - (delta**2 + fr * (1 - fr))), 5e-3)

# ---- 3. Gaussian tie fraction: tau_RTN(s) = 2Phi(1/s)-1-2s(phi(0)-phi(1/s))
def tau_rtn(s):
    return 2 * norm.cdf(1 / s) - 1 - 2 * s * (norm.pdf(0) - norm.pdf(1 / s))
def tau_sr(s):
    return integrate.quad(lambda w: 2 * norm.pdf(w) * Bsp(s * w), 0, 3 / s if s > 0 else 50, limit=200)[0]
for s in [0.05, 0.15, 0.5, 1.0, 2.6, 6.0]:
    W = rng.standard_normal(2_000_000)
    U = rng.random(W.shape)
    p = np.mean(Q(U, 1) == Q(U + s * W, 1))
    check(f"tau_RTN closed form s={s}", abs(p - tau_rtn(s)), 2e-3)
    p2 = np.mean(SR(U, 1, rng) == SR(U + s * W, 1, rng))
    check(f"tau_SR quadrature s={s}", abs(p2 - tau_sr(s)), 2e-3)

# ---- 4. exact mean of rounded MeZO estimator for affine f:  E g = kappa(c,r) a
def kappa(c, r, u=1.0):      # wrapped-Gaussian form
    n = np.arange(-60, 61)
    return np.sum(np.exp(-((c / u - n - 0.5) ** 2) / (2 * r**2))) / (r * np.sqrt(2 * np.pi))
def kappa_fourier(c, r, u=1.0):
    k = np.arange(1, 200)
    return 1 + 2 * np.sum(np.exp(-2 * np.pi**2 * k**2 * r**2) * np.cos(2 * np.pi * k * (c / u - 0.5)))
for c in [0.0, 0.13, 0.31, 0.5]:
    for r in [0.08, 0.2, 0.5, 1.2]:
        check(f"Poisson identity kappa c={c} r={r}", abs(kappa(c, r) - kappa_fourier(c, r)) / max(1, kappa(c, r)), 1e-9)
d = 20
a = rng.standard_normal(d); a /= np.linalg.norm(a)
for (c, r) in [(0.0, 0.15), (0.45, 0.15), (0.25, 0.3), (0.1, 0.6), (0.4, 1.0)]:
    mu = r * 1.0 / np.linalg.norm(a)           # r = mu |a| / u, u=1
    N = 4_000_000
    Z = rng.standard_normal((N, d)).astype(np.float32)
    t = mu * (Z @ a)
    g = ((Q(c + t, 1) - Q(c - t, 1)) / (2 * mu))[:, None] * Z
    Eg = g.mean(0)
    par, perp = Eg @ a, np.linalg.norm(Eg - (Eg @ a) * a)
    se = g.std(0).max() / np.sqrt(N)
    check(f"E[g] parallel = kappa (c={c},r={r}) kappa={kappa(c, r):.4f} MC={par:.4f}", abs(par - kappa(c, r)), 6 * se * np.sqrt(d) + 1e-3)
    check(f"E[g] orthogonal = 0 (c={c},r={r})", perp, 8 * se * np.sqrt(d))
# bound |kappa-1| <= 2q/(1-q^3), q = exp(-2 pi^2 r^2)
for r in np.linspace(0.2, 1.5, 14):
    q = np.exp(-2 * np.pi**2 * r**2)
    worst = max(abs(kappa(c, r) - 1) for c in np.linspace(0, 1, 401))
    check(f"|kappa-1|<=2q/(1-q^3) r={r:.2f}", max(0, worst - 2 * q / (1 - q**3)), 1e-12)
# average over c equals 1
for r in [0.05, 0.3, 1.0]:
    avg = integrate.quad(lambda c: kappa(c, r), 0, 1, limit=400)[0]
    check(f"offset-average of kappa = 1 (r={r})", abs(avg - 1), 1e-8)

# ---- 5. Gaussian moments
d = 7
Z = rng.standard_normal((4_000_000, d))
v = rng.standard_normal(d)
m1 = np.mean((Z @ v) ** 2 * np.sum(Z**2, 1)); m2 = np.mean(np.sum(Z**2, 1) ** 3)
check("E[<v,z>^2|z|^2]=(d+2)|v|^2", abs(m1 / ((d + 2) * v @ v) - 1), 1e-2)
check("E|z|^6 = d(d+2)(d+4)", abs(m2 / (d * (d + 2) * (d + 4)) - 1), 1e-2)
check("E|<z,v>| = sqrt(2/pi)|v|", abs(np.mean(np.abs(Z @ v)) / np.linalg.norm(v) - np.sqrt(2 / np.pi)), 2e-3)
# sphere: E|e_1| <= 1/sqrt(d)
E = rng.standard_normal((2_000_000, d)); E /= np.linalg.norm(E, axis=1, keepdims=True)
check("sphere E|e1| <= 1/sqrt(d)", max(0, np.mean(np.abs(E[:, 0])) - 1 / np.sqrt(d)), 1e-12)

print("\nALL CHECKS PASSED" if OK else "\nSOME CHECKS FAILED")
sys.exit(0 if OK else 1)
