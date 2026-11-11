"""Shared code for the H3 experiments (tau = 1, two-point methods of radius gamma).

Everything is vectorised over K independent runs (rows).  Seeds are fixed by the callers.
Noises (all have E_pi delta(x, .) = 0 for every x, hence condition (2) for all radii, pointwise):
  * 'beta'  : rough grid-cell hash noise  delta(y,s) = Delta * U_{cell(y)}(s),  U ~ 2*Beta(3,3)-1,
              cells = cubes of side r/sqrt(d) (diameter r = gamma < 2 gamma); a fresh state
              (64-bit key) every round.  Beta(3,3) = median of 5 hashed uniforms.  Fisher info of U: 10.
  * 'pm1'   : the same hash with U = +-1 (the construction literally proposed in the formulation of the hypothesis, claim H3.2).
  * 'sin'   : smooth noise Delta * sin(<v_s, y>/ell + phi_s), v_s ~ Unif(S^{d-1}), phi_s ~ U[0, 2pi).
"""
import os
import numpy as np
from scipy.special import gammaln, jv

U64 = np.uint64
_M1, _M2, _G = U64(0xBF58476D1CE4E5B9), U64(0x94D049BB133111EB), U64(0x9E3779B97F4A7C15)
np.seterr(over='ignore')


def splitmix(h):
    h = (h + _G)
    h = (h ^ (h >> U64(30))) * _M1
    h = (h ^ (h >> U64(27))) * _M2
    return h ^ (h >> U64(31))


def to_unif(h):
    return (h >> U64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


class HashNoise:
    """delta(y, s) = Delta * eta(cell(y), s); the state s is a 64-bit key per run and round."""
    SALTS = [U64(s) for s in (0x1234567, 0x89ABCDE, 0x2468ACE, 0x13579BD, 0x0F0F0F0)]

    def __init__(self, d, r, kind, rng):
        self.scale = np.sqrt(d) / r            # cell side r/sqrt(d) -> diameter r
        self.mult = (rng.integers(1, 2**62, size=d, dtype=np.int64) * 2 + 1).astype(np.uint64)
        self.kind = kind

    def cells(self, y):
        return np.floor(y * self.scale).astype(np.int64)

    def eta(self, y, key):
        z = self.cells(y).astype(np.uint64)
        h = splitmix((z * self.mult).sum(axis=1) ^ key)
        if self.kind == 'pm1':
            return np.where((h >> U64(7)) & U64(1), 1.0, -1.0)
        u = np.stack([to_unif(splitmix(h ^ s)) for s in self.SALTS], axis=0)
        u.sort(axis=0)
        return 2.0 * u[2] - 1.0      # median of 5 uniforms ~ Beta(3,3); U = 2*Beta(3,3) - 1


# ----------------------------------------------------------------------------- problems
class Logistic:
    name = 'logistic'

    def __init__(self, d, n, lam, seed):
        rng = np.random.default_rng(1000 + seed)
        A = rng.standard_normal((n, d))
        w0 = rng.standard_normal(d) / np.sqrt(d) * 2
        p = 1 / (1 + np.exp(-A @ w0))
        b = np.where(rng.random(n) < p, 1.0, -1.0)
        self.A, self.lam, self.n, self.d = A * b[:, None], lam, n, d
        from scipy.optimize import minimize
        r = minimize(lambda x: self.f(x[None])[0], np.zeros(d), jac=self.grad, method='L-BFGS-B',
                     options=dict(gtol=1e-12, ftol=1e-15, maxiter=10000))
        self.xstar, self.fstar = r.x, r.fun
        self.R = 2 * np.linalg.norm(self.xstar)
        self.M = np.linalg.norm(A, 2) / np.sqrt(n) + lam * self.R
        self.f0 = self.f(np.zeros((1, d)))[0]

    def f(self, X):          # X: (K, d)
        z = X @ self.A.T
        l = np.maximum(-z, 0.0) + np.log1p(np.exp(-np.abs(z)))   # = log(1 + exp(-z)), stable
        return l.mean(axis=1) + 0.5 * self.lam * (X ** 2).sum(axis=1)

    def grad(self, x):
        z = self.A @ x
        return -(self.A.T @ (1 / (1 + np.exp(z)))) / self.n + self.lam * x


class L1Reg:
    name = 'l1'

    def __init__(self, d, n, seed):
        rng = np.random.default_rng(2000 + seed)
        A = rng.standard_normal((n, d))
        x0 = rng.standard_normal(d) / np.sqrt(d)
        b = A @ x0 + rng.laplace(size=n) * 0.5
        self.A, self.b, self.n, self.d = A, b, n, d
        from scipy.optimize import linprog
        c = np.r_[np.zeros(d), np.ones(n) / n]
        I = np.eye(n)
        r = linprog(c, A_ub=np.block([[A, -I], [-A, -I]]), b_ub=np.r_[b, -b],
                    bounds=[(None, None)] * d + [(0, None)] * n, method='highs')
        self.xstar = r.x[:d]
        self.fstar = self.f(self.xstar[None])[0]
        self.R = 2 * np.linalg.norm(self.xstar)
        self.M = np.linalg.norm(A, 2) / np.sqrt(n)
        self.f0 = self.f(np.zeros((1, d)))[0]

    def f(self, X):
        return np.abs(X @ self.A.T - self.b).mean(axis=1)


class Linear:
    """f_v(x) = a <v, x> on B_R (the hard family of Thm 4.6)."""
    name = 'linear'

    def __init__(self, V, a, R):          # a: scalar or one slope per run (row of V)
        self.V, self.a, self.R, self.d = V, np.asarray(a, float), R, V.shape[1]
        self.M = self.a * np.sqrt(self.d)
        self.fstar = -self.a * R * np.sqrt(self.d)

    def f(self, X):
        K = self.V.shape[0]
        if X.shape[0] == K:
            return self.a * (X * self.V).sum(axis=1)
        return np.tile(self.a, X.shape[0] // K) * (X * np.tile(self.V, (X.shape[0] // K, 1))).sum(axis=1)


# ----------------------------------------------------------------------------- helpers
def unit_sphere(rng, K, d):
    e = rng.standard_normal((K, d))
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def proj_ball(X, Rp):
    nr = np.linalg.norm(X, axis=1, keepdims=True)
    return X * np.minimum(1.0, Rp / np.maximum(nr, 1e-300))


def Phi_sphere(d, t):
    """Sphere multiplier E_e cos(t e_1) = Gamma(d/2) (2/t)^{d/2-1} J_{d/2-1}(t)."""
    t = np.atleast_1d(np.asarray(t, float))
    out = np.ones_like(t)
    nz = t > 1e-3
    out[~nz] = 1 - t[~nz] ** 2 / (2 * d) + t[~nz] ** 4 / (8 * d * (d + 2))   # Taylor series
    nu = d / 2 - 1
    out[nz] = np.exp(gammaln(d / 2) + nu * np.log(2 / t[nz])) * jv(nu, t[nz])
    return out


BETA33_VAR = 1.0 / 7.0          # Var(2*Beta(3,3)-1)
BETA33_FISHER = 10.0            # Fisher information of 2*Beta(3,3)-1


def omega_bar(kind, Delta, d=None, gamma=None, ell=None):
    """sqrt(E[(delta(y+ge,s)-delta(y-ge,s))^2]) (pair in different cells) for the noises above."""
    Delta = np.asarray(Delta, float)
    if kind == 'beta':
        return Delta * np.sqrt(2 * BETA33_VAR)
    if kind == 'pm1':
        return Delta * np.sqrt(2.0)
    if kind == 'sin':          # E (2 Delta cos(.) sin(g<v,e>/ell))^2 = Delta^2 (1 - Phi_d(2 g/ell))
        return Delta * np.sqrt(1 - Phi_sphere(d, 2 * gamma / ell)[0])
    if kind == 'none':
        return 0.0 * Delta
    raise ValueError(kind)


class ZOSGD:
    """Projected ZO-SGD (batch 1) on B_{R-gamma}; anytime step eta_t = eta_scale*R/(G sqrt t);
    output = average of all iterates.  Records f(xbar_t) - f* at the checkpoints (array (C, K)).
    Delta / G / eta_scale / ell may be arrays of length K (one value per run).
    Directions and noise use separate random streams (seed, seed + 10**6), so two runs with the same
    seed see the same directions.  The object is picklable: run(t_end) can be resumed after a restart."""

    def __init__(self, prob, gamma, Delta, kind, T, checkpoints, K, seed, G=None, eta_scale=1.0, ell=None,
                 decode=False, Rball=None):
        d = prob.d
        self.prob, self.gamma, self.kind, self.T, self.K, self.decode = prob, gamma, kind, T, K, decode
        self.rng = np.random.default_rng(seed)
        self.rng_n = np.random.default_rng(seed + 10**6)
        self.R = prob.R if Rball is None else Rball
        self.Delta = np.broadcast_to(np.asarray(Delta, float), (K,)).copy()
        self.ell = np.broadcast_to(np.asarray(ell, float), (K,)).copy() if kind == 'sin' else None
        if G is None:
            G = d * prob.M / np.sqrt(d - 1) + d * omega_bar(kind, self.Delta, d, gamma, self.ell) / (2 * gamma)
        self.G = np.broadcast_to(np.asarray(G, float), (K,)).copy()
        self.c0 = np.broadcast_to(np.asarray(eta_scale, float), (K,)) * self.R / self.G
        self.X = np.zeros((K, d))
        self.S = np.zeros((K, d))
        self.noise = HashNoise(d, gamma, kind, self.rng_n) if kind in ('beta', 'pm1') else None
        self.cps = sorted(checkpoints)
        self.out = []
        self.t = 0

    def run(self, t_end=None, seconds=None):
        import time
        t_end = self.T if t_end is None else min(t_end, self.T)
        prob, gamma, K, d = self.prob, self.gamma, self.K, self.prob.d
        Rp = self.R - gamma
        cps = set(self.cps)
        t0 = time.time()
        while self.t < t_end:
            self.t += 1
            t = self.t
            e = unit_sphere(self.rng, K, d)
            yp, ym = self.X + gamma * e, self.X - gamma * e
            fv = prob.f(np.concatenate([yp, ym]))
            D = fv[:K] - fv[K:]
            if self.noise is not None:
                key = self.rng_n.integers(0, 2**63, size=K, dtype=np.int64).astype(np.uint64)
                D = D + self.Delta * (self.noise.eta(yp, key) - self.noise.eta(ym, key))
            elif self.kind == 'sin':
                v = unit_sphere(self.rng_n, K, d)
                ph = self.rng_n.uniform(0, 2 * np.pi, K)
                D = D + self.Delta * (np.sin((yp * v).sum(1) / self.ell + ph) - np.sin((ym * v).sum(1) / self.ell + ph))
            if self.decode:   # +-1 attack: |true difference| <= 2 M gamma < Delta => remove the multiple of 2 Delta
                D = D - 2 * self.Delta * np.round(D / (2 * self.Delta))
            self.X = proj_ball(self.X - ((self.c0 / np.sqrt(t)) * (d / (2 * gamma)) * D)[:, None] * e, Rp)
            self.S += self.X
            if t in cps:
                self.out.append(prob.f(self.S / t) - prob.fstar)
            if seconds is not None and (t & 255) == 0 and time.time() - t0 > seconds:
                break
        return self.t >= self.T

    def result(self):
        return np.array(self.out)


def zo_sgd(*args, **kw):
    z = ZOSGD(*args, **kw)
    z.run()
    return z.result()


class Job:
    """Resumable job: pickles a ZOSGD (or anything with .run(seconds=) and .result()) between invocations."""

    def __init__(self, path, factory):
        import pickle
        self.path = path
        if os.path.exists(path):
            self.obj = pickle.load(open(path, 'rb'))
        else:
            self.obj = factory()

    def step(self, seconds):
        import pickle
        done = self.obj.run(seconds=seconds)
        tmp = self.path + '.tmp'
        pickle.dump(self.obj, open(tmp, 'wb'))
        os.replace(tmp, self.path)
        return done
