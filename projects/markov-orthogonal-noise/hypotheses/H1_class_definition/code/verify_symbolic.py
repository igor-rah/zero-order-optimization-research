"""Symbolic / high-precision checks (sympy + mpmath).  Writes data/verify_symbolic.json.
Run where sympy is installed.

S1  d/dt[t^{-nu} J_nu(t)] = -t^{-nu} J_{nu+1}(t)  =>  Phi_d'(t) = -(t/d) Psi_d(t)   (symbolic, general d)
S2  Psi_d is the Fourier transform of the uniform law on the unit ball: exact for d = 1, 3; 40 digits for d = 2, 5
S3  Plane-wave identity E[sin(t e_1) e_1] = t Psi_d(t)/d at 40 digits for d = 100 near the first zeros
    (the regime where Monte Carlo in E1.1 cannot resolve the signal)
S4  Zeros of J_{d/2} are simple and Psi_d'(j) != 0 (d = 5, 20, 100; first three zeros)
S5  sum_{t>=1} 2^{-floor(t/tau)} = 2 tau - 1 (symbolic in tau);  (9/8)^2 + 1 = 145/64;  16*145/64 = 36.25
S6  Single-layer step of Theorem 3: rho * delta'(rho) = -delta(rho) as distributions (sympy DiracDelta)
S7  E|e_1| = Gamma(d/2)/(sqrt(pi) Gamma((d+1)/2)) <= 1/sqrt(d)  (exact for d = 2..200 at 30 digits)
"""
import json
import sympy as sp
import mpmath as mp

mp.mp.dps = 50
R = {}
def rec(name, ok, **kw):
    R[name] = dict(ok=bool(ok), **{k: str(v) for k, v in kw.items()})
    print(('PASS ' if ok else 'FAIL ') + name, kw)

t, d, tau, nu = sp.symbols('t d tau nu', positive=True)
# S1
expr = sp.diff(t ** (-nu) * sp.besselj(nu, t), t) + t ** (-nu) * sp.besselj(nu + 1, t)
s1 = sp.simplify(sp.expand_func(expr).rewrite(sp.besselj))
Psi = sp.gamma(d / 2 + 1) * (2 / t) ** (d / 2) * sp.besselj(d / 2, t)
Phi = sp.gamma(d / 2) * (2 / t) ** (d / 2 - 1) * sp.besselj(d / 2 - 1, t)
s1b = sp.simplify(sp.diff(Phi, t) + t / d * Psi)
num = [abs(complex(s1b.subs({d: dv, t: tv}).evalf(30))) for dv in (2, 3, 7, 50) for tv in (0.7, 4.1, 33.0)]
rec('S1_Phi_prime', s1 == 0 and max(num) < 1e-25, recurrence=s1, max_num_residual=max(num))

# S2
u = sp.symbols('u', real=True)
P1 = sp.simplify(sp.integrate(sp.cos(t * u), (u, -1, 1)) / 2 - Psi.subs(d, 1))
P3 = sp.simplify(sp.Rational(3, 4) * sp.integrate((1 - u ** 2) * sp.cos(t * u), (u, -1, 1)) - Psi.subs(d, 3))
err = 0
for dv in (2, 5):
    c = 1 / mp.quad(lambda s: (1 - s * s) ** (mp.mpf(dv - 1) / 2), [-1, 1])
    for tv in (mp.mpf('0.9'), mp.mpf('6.3'), mp.mpf('17.5')):
        q = c * mp.quad(lambda s: mp.cos(tv * s) * (1 - s * s) ** (mp.mpf(dv - 1) / 2), [-1, 0, 1])
        ps = mp.gamma(mp.mpf(dv) / 2 + 1) * (2 / tv) ** (mp.mpf(dv) / 2) * mp.besselj(mp.mpf(dv) / 2, tv)
        err = max(err, abs(q - ps))
rec('S2_Psi_ball_FT', sp.simplify(P1.rewrite(sp.sin)) == 0 and sp.simplify(P3.rewrite(sp.sin)) == 0 and err < 1e-35,
    d1=sp.simplify(P1.rewrite(sp.sin)), d3=sp.simplify(P3.rewrite(sp.sin)), max_err_d2_d5=err)

# S3
dv = 100
c = 1 / mp.quad(lambda s: (1 - s * s) ** (mp.mpf(dv - 3) / 2), [-1, 1])
rows = []
ok = True
for k in (1, 2, 3):
    j = mp.besseljzero(mp.mpf(dv) / 2, k)
    for tv in (j * mp.mpf('0.95'), j, j * mp.mpf('1.05')):
        lhs = c * mp.quad(lambda s: mp.sin(tv * s) * s * (1 - s * s) ** (mp.mpf(dv - 3) / 2), mp.linspace(-1, 1, 41))
        rhs = tv / dv * mp.gamma(mp.mpf(dv) / 2 + 1) * (2 / tv) ** (mp.mpf(dv) / 2) * mp.besselj(mp.mpf(dv) / 2, tv)
        rel = abs(lhs - rhs) / max(abs(rhs), mp.mpf(10) ** -45)
        ok &= rel < 1e-20 or abs(lhs - rhs) < 1e-40
        rows.append((float(tv), mp.nstr(lhs, 8), mp.nstr(rhs, 8)))
rec('S3_plane_wave_d100_high_precision', ok, rows=rows)

# S4
ok = True; info = []
for dv in (5, 20, 100):
    for k in (1, 2, 3):
        j = mp.besseljzero(mp.mpf(dv) / 2, k)
        jp1 = mp.besselj(mp.mpf(dv) / 2 + 1, j)
        ok &= abs(jp1) > mp.mpf(10) ** -5
        info.append((dv, k, mp.nstr(j, 12), mp.nstr(jp1, 6)))
rec('S4_simple_zeros', ok, zeros_and_J_nu_plus_1=info)

# S5
l = sp.symbols('l', integer=True, nonnegative=True)
s5 = sp.simplify(tau * sp.summation(sp.Rational(1, 2) ** l, (l, 0, sp.oo)) - 1 - (2 * tau - 1))
c1 = sp.Rational(9, 8) ** 2 + 1
rec('S5_constants', s5 == 0 and c1 == sp.Rational(145, 64) and 16 * c1 == sp.Rational(145, 4),
    blocksum_residual=s5, G2_over_M2=c1, N_const=16 * c1)

# S6
r_, x_ = sp.symbols('rho x', real=True)
phi = sp.exp(-(r_ - sp.Rational(1, 3)) ** 2) * sp.cos(r_)        # test function
lhs = sp.integrate(r_ * sp.DiracDelta(r_, 1) * phi, (r_, -sp.oo, sp.oo))
rhs = -sp.integrate(sp.DiracDelta(r_) * phi, (r_, -sp.oo, sp.oo))
rec('S6_rho_delta_prime', sp.simplify(lhs - rhs) == 0, lhs=lhs, rhs=rhs)

# S7
mp.mp.dps = 30
ok = all(mp.gamma(mp.mpf(k) / 2) / (mp.sqrt(mp.pi) * mp.gamma(mp.mpf(k + 1) / 2)) <= 1 / mp.sqrt(k) for k in range(2, 201))
rec('S7_E_abs_e1', ok)

json.dump(R, open('data/verify_symbolic.json', 'w'), indent=1)
print('ALL PASS' if all(v['ok'] for v in R.values()) else 'SOME FAIL')
