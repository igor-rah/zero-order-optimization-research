"""Numerical checks of the closed forms used in the Markov paper (Monte Carlo)."""
import numpy as np
from scipy.stats import norm
rng = np.random.default_rng(0)
ok = True
def check(name, a, b, tol):
    global ok
    good = abs(a - b) <= tol
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} {name}: closed={a:.6f} mc={b:.6f} tol={tol}")

# Prop A: conditional tie kernel P(tie|omega) = 2 Phi((1/2-|omega|)/r) - 1  (RTN, affine, Gaussian dir)
N = 2_000_000
for r in (0.065, 0.15, 0.5, 1.0):
    for om in (0.0, 0.2, 0.45, -0.3):
        W = rng.standard_normal(N)
        fp, fm = om + r * W, om - r * W                  # values in grid units, grid = Z
        tie = np.mean(np.round(fp) == np.round(fm))
        check(f"kernel r={r} omega={om}", 2 * norm.cdf((0.5 - abs(om)) / r) - 1, tie, 4e-3)
# its uniform average equals tau_rn(s), s = 2r
for r in (0.065, 0.25, 0.5, 1.3):
    s = 2 * r
    om = np.linspace(-0.5, 0.5, 200001)
    avg = np.trapezoid(2 * norm.cdf((0.5 - np.abs(om)) / r) - 1, om)
    tau = 2 * norm.cdf(1 / s) - 1 - 2 * s * (norm.pdf(0) - norm.pdf(1 / s))
    check(f"average kernel = tie law, r={r}", tau, avg, 1e-6)
# Prop B: no move in N steps with B directions = P_tie^(B N)
r, om, B, Nst = 0.15, 0.2, 5, 4
p = (2 * norm.cdf((0.5 - om) / r) - 1) ** (B * Nst)
W = rng.standard_normal((200000, B * Nst))
mc = np.mean(np.all(np.round(om + r * W) == np.round(om - r * W), 1))
check("absorption probability", p, mc, 3e-3)
# Prop C: dither bound TV(wrapped N(0,sig^2), Unif) <= q/(1-q^3), q = exp(-2 pi^2 sig^2)
for sig in (0.3, 0.5, 0.75):
    q = np.exp(-2 * np.pi ** 2 * sig ** 2)
    t = np.linspace(0, 1, 20001)[:-1]
    dens = 1 + 2 * sum(np.exp(-2 * np.pi ** 2 * k * k * sig ** 2) * np.cos(2 * np.pi * k * t) for k in range(1, 50))
    tv = 0.5 * np.mean(np.abs(dens - 1))
    bound = q / (1 - q ** 3)
    good = tv <= bound + 1e-12; ok &= good
    print(f"{'PASS' if good else 'FAIL'} dither TV sig={sig}: TV={tv:.3e} <= bound={bound:.3e}")
    # first Fourier coefficient = autocorrelation contribution
    X = rng.normal(0, sig, 1_000_000) % 1.0
    check(f"|E exp(2 pi i c)| sig={sig}", q, abs(np.mean(np.exp(2j * np.pi * X))), 3e-3)
print("ALL PASS" if ok else "SOME FAILED")
