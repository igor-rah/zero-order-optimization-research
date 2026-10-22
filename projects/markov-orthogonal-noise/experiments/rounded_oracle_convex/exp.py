"""Convex check: where does rounding of f-values put a floor on ZO accuracy, and how does the
window of good smoothing radii mu shrink as the target accuracy eps decreases?

Problem: l2-regularised logistic regression (smooth, strongly convex), f* known to ~1e-15.
Method: two-point ZO gradient descent, B random directions on the unit sphere per step,
  g = mean_b d/(2mu) (f~(x+mu e_b) - f~(x-mu e_b)) e_b,   f~ = f rounded to p significant bits.
Output: f(x_bar) - f*, x_bar = average of the last half of iterates (removes variance floor,
keeps bias floor). All mu values run in one vectorised batch.
"""
import json, os, sys, time, numpy as np

def make_problem(d, n, seed, lam=1e-2, kind='logreg'):
    if kind == 'huber':
        return make_huber(d, n, seed, lam)
    r = np.random.default_rng(seed)
    A = r.standard_normal((n, d)) / np.sqrt(d)
    w = r.standard_normal(d) * 2.0
    y = np.sign(A @ w + 0.5 * r.standard_normal(n))
    M = -(y[:, None] * A)                      # f(x) = mean log(1+exp(M x)) + lam/2 |x|^2
    def f(X):                                  # X: (d, k) -> (k,)
        Z = M @ X
        return np.mean(np.logaddexp(0, Z), 0) + 0.5 * lam * np.sum(X * X, 0)
    # Newton for f*
    x = np.zeros(d)
    for _ in range(60):
        z = M @ x; s = 1 / (1 + np.exp(-z))
        g = M.T @ s / n + lam * x
        H = (M.T * (s * (1 - s))) @ M / n + lam * np.eye(d)
        x -= np.linalg.solve(H, g)
    fstar = f(x[:, None])[0]
    L = np.linalg.eigvalsh(M.T @ M / n).max() / 4 + lam
    return f, fstar, x, L, lam

def make_huber(d, n, seed, lam, kh=0.1):
    """Huber regression: gradient Lipschitz, Hessian discontinuous (smoothness order beta = 2)."""
    r = np.random.default_rng(seed)
    A = r.standard_normal((n, d)) / np.sqrt(d)
    w = r.standard_normal(d) * 2.0
    b = A @ w + 0.5 * r.standard_normal(n)
    def hub(z):
        a = np.abs(z)
        return np.where(a <= kh, 0.5 * z * z, kh * (a - 0.5 * kh))
    def f(X):
        return np.mean(hub(A @ X - b[:, None]), 0) + 0.5 * lam * np.sum(X * X, 0) + 0.5
    L = np.linalg.eigvalsh(A.T @ A / n).max() + lam
    x = np.zeros(d)
    for _ in range(20000):
        z = A @ x - b
        g = A.T @ np.clip(z, -kh, kh) / n + lam * x
        x -= g / L
    fstar = f(x[:, None])[0]
    return f, fstar, x, L, lam

def round_p(v, p):
    if p is None:
        return v
    e = np.floor(np.log2(np.abs(v)))
    ulp = 2.0 ** (e - p + 1)
    return np.round(v / ulp) * ulp

def ulp_p(v, p):
    return 2.0 ** (np.floor(np.log2(np.abs(v))) - p + 1)

def run(d, n, seed, p, mus, T, B=5, adaptive=False, mu_min=None, mu_max=None, kind='logreg'):
    f, fstar, xstar, L, lam = make_problem(d, n, seed, kind=kind)
    rng = np.random.default_rng(10_000 + seed)
    K = len(mus)
    mu = np.array(mus, float)
    X = np.zeros((d, K)); Xbar = np.zeros((d, K)); cnt = 0
    eta = 0.5 / (L * (d / B + 1))
    curve = []
    for t in range(T):
        E = rng.standard_normal((d, B)); E /= np.linalg.norm(E, axis=0)
        # points: for each k and b, x_k +- mu_k e_b
        P = X[:, :, None] + mu[None, :, None] * E[:, None, :]          # d,K,B
        Q = X[:, :, None] - mu[None, :, None] * E[:, None, :]
        fp = f(P.reshape(d, -1)).reshape(K, B); fm = f(Q.reshape(d, -1)).reshape(K, B)
        rp, rm = round_p(fp, p), round_p(fm, p)
        G = (d / (2 * mu))[:, None] * (rp - rm)                         # K,B
        X -= eta * (E @ G.T) / B
        if adaptive and p is not None:
            small = np.mean(np.abs(rp - rm) < 2 * ulp_p((fp + fm) / 2, p), 1) > 0.5
            mu = np.where(small, np.minimum(mu * 1.5, mu_max), np.maximum(mu / 1.5, mu_min))
        if t >= T // 2:
            Xbar += X; cnt += 1
        if t % (T // 60) == 0:
            curve.append((t, (f(X) - fstar).tolist()))
    floor = f(Xbar / cnt) - fstar
    last = f(X) - fstar
    return dict(d=d, n=n, seed=seed, p=p, mus=list(mus), floor=floor.tolist(), last=last.tolist(),
                fstar=fstar, L=L, lam=lam, R=float(np.linalg.norm(xstar)), curve=curve,
                final_mu=mu.tolist())

if __name__ == '__main__':
    job = sys.argv[1]
    out = {}
    t0 = time.time()
    mus = np.logspace(-6, 1.5, 23)
    if job in ('main', 'huber'):
        kind = 'logreg' if job == 'main' else 'huber'
        for seed in ((0, 1, 2) if job == 'main' else (0, 1)):
            for p in (None, 14, 11, 8):
                out[f'{p}_{seed}'] = run(50, 500, seed, p, mus, T=10000, kind=kind)
                print(job, seed, p, f'{time.time()-t0:.0f}s', flush=True)
            for p in (11, 8):
                out[f'adapt{p}_{seed}'] = run(50, 500, seed, p, [1e-6], T=10000, adaptive=True,
                                              mu_min=1e-6, mu_max=30.0, kind=kind)
    elif job == 'dim':
        mus_d = np.logspace(-2, 1.5, 15)
        for d in (12, 25, 50, 100):
            for seed in (0, 1):
                out[f'{d}_{seed}'] = run(d, 10 * d, seed, 11, mus_d, T=160 * d)
                print(job, d, seed, f'{time.time()-t0:.0f}s', flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{job}.json'), 'w'))
