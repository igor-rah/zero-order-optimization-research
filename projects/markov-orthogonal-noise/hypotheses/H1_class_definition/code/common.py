"""Shared helpers for the H1 report (ball multiplier, Bessel zeros, sphere sampling, CIs, hash noise)."""
import numpy as np
from scipy.special import jv, gammaln, betainc
from scipy.optimize import brentq
from scipy import stats

def Psi(d, t):
    """Ball multiplier Psi_d(t) = Gamma(d/2+1) (2/t)^{d/2} J_{d/2}(t) = E_{u~Unif(B)} exp(i t u_1); Psi_d(0)=1."""
    t = np.asarray(t, float)
    out = np.ones_like(t)
    nz = t != 0
    tt = np.abs(t[nz])
    out[nz] = np.exp(gammaln(d / 2 + 1) + (d / 2) * np.log(2 / tt)) * jv(d / 2, tt)
    return out

def Phi(d, t):
    """Sphere multiplier Phi_d(t) = E_{e~Unif(S)} exp(i t e_1) = Gamma(d/2)(2/t)^{d/2-1} J_{d/2-1}(t)."""
    t = np.asarray(t, float)
    out = np.ones_like(t)
    nz = t != 0
    tt = np.abs(t[nz])
    out[nz] = np.exp(gammaln(d / 2) + (d / 2 - 1) * np.log(2 / tt)) * jv(d / 2 - 1, tt)
    return out

def bessel_zeros(nu, k):
    """First k positive zeros of J_nu (any real nu >= 0), by sign changes + Brent."""
    zs, a = [], max(1e-6, nu * 0.5)
    ts = np.linspace(1e-6, nu + 3.5 * k + 10 + 4 * np.pi * k, 200000)
    f = jv(nu, ts)
    idx = np.where(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
    for i in idx[:k]:
        zs.append(brentq(lambda t: jv(nu, t), ts[i], ts[i + 1], xtol=1e-14))
    assert len(zs) == k
    return np.array(zs)

def sphere(rng, n, d, dtype=np.float64):
    e = rng.standard_normal((n, d)).astype(dtype)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return e

def e1_cdf(d, x):
    """CDF of the first coordinate of e~Unif(S^{d-1}): P(e_1<=x) = I_{(1+x)/2}((d-1)/2,(d-1)/2)."""
    x = np.clip(np.asarray(x, float), -1, 1)
    return betainc((d - 1) / 2, (d - 1) / 2, (1 + x) / 2)

def t_ci(x, level=0.95):
    """Mean and half-width of a Student-t CI over seeds."""
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    if n < 2:
        return m, np.nan
    h = stats.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, h

def loglog_slope(x, y, level=0.95):
    """OLS slope of log y on log x with a t-CI."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    res = stats.linregress(lx, ly)
    n = len(lx)
    h = stats.t.ppf(0.5 + level / 2, n - 2) * res.stderr if n > 2 else np.nan
    return res.slope, h

# ---------- 64-bit hash noise on grid cells --------------------------------------------------
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)

def splitmix(z):
    z = z.astype(np.uint64)
    z = (z ^ (z >> np.uint64(30))) * _M1
    z = (z ^ (z >> np.uint64(27))) * _M2
    return z ^ (z >> np.uint64(31))

def cell_sign(X, h, key, coef):
    """eta(cell(x), key) in {-1,+1}: X (n,d), cells of side h, key uint64 per row (n,) or scalar."""
    with np.errstate(over='ignore'):
        C = np.floor(X / h).astype(np.int64).astype(np.uint64)
        z = (C * coef[None, :]).sum(1, dtype=np.uint64) + np.asarray(key, np.uint64) * np.uint64(0x9E3779B97F4A7C15)
        z = splitmix(z)
    return np.where((z >> np.uint64(63)) == 1, 1.0, -1.0)

def hash_coefs(d, seed=12345):
    r = np.random.default_rng(seed)
    return (r.integers(1, 2**62, size=d, dtype=np.int64) * 2 + 1).astype(np.uint64)
