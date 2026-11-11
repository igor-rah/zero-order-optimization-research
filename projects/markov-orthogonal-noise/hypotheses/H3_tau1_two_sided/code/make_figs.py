"""Build every figure of the H3 report from data/*.json and write data/summary_stats.json.
Run: python3 make_figs.py
"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import Phi_sphere, omega_bar

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
FIGS = os.path.join(HERE, '..', 'figs')
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9.5, 'legend.fontsize': 7.5, 'figure.dpi': 150,
                     'axes.grid': True, 'grid.alpha': 0.3})
S = {}
B = 4000
rs = np.random.default_rng(0)
LEVCOL = {3: '#7fb3e6', 10: '#2a78d6', 30: '#e8743b', 100: '#19a979', 300: '#945ecf'}


def load(name):
    p = os.path.join(DATA, name)
    return json.load(open(p)) if os.path.exists(p) else None


def ci(x):
    x = np.asarray(x, float)
    m = x.mean()
    bs = rs.choice(x, size=(B, len(x))).mean(1)
    return m, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


# ======================================================================================== E3.1
def e31_analysis(r, rng=None):
    rng = rs if rng is None else rng             # separate generator for added variants, so that
    sc = r.get('level_scale', 1.0)               # every number of the earlier version is reproduced unchanged
    err = np.array(r['err'])                     # (C, K)
    T = np.array(r['checkpoints'], float)
    lev = np.array(r['levels'])
    n = r['nrep']
    L = sorted(set(lev))
    E = np.stack([err[:, lev == l] for l in L], axis=1)        # (C, nL, n)
    noiseless = np.exp(np.log(np.maximum(E[:, 0, :], 1e-300)).mean(1))   # geometric mean, Delta = 0
    gap0 = r['f0'] - r['fstar']
    pts = []
    for il, l in enumerate(L):
        if l < 10:
            continue
        for ic in range(len(T)):
            for k in range(n):
                e = E[ic, il, k]
                if e >= 5 * noiseless[ic]:
                    pts.append((np.log(l * sc * r['Delta_ref']), np.log(T[ic]), np.log(e), k, e <= 0.5 * gap0))
    P = np.array(pts)

    def fit(mask, idx=None):
        Q = P[mask] if idx is None else np.concatenate([P[mask & (P[:, 3] == k)] for k in idx])
        X = np.c_[np.ones(len(Q)), Q[:, 0], Q[:, 1]]
        c, *_ = np.linalg.lstsq(X, Q[:, 2], rcond=None)
        res = Q[:, 2] - X @ c
        return -c[2] / c[1], c[1], 1 - res.var() / Q[:, 2].var()

    out = {}
    for tag, mask in (('all', np.ones(len(P), bool)), ('progress', P[:, 4] > 0)):
        if mask.sum() < 6:
            continue
        a, al, r2 = fit(mask)
        boots = np.array([fit(mask, rng.integers(0, n, n))[:2] for _ in range(1000)])
        out[tag] = dict(a=a, a_lo=np.percentile(boots[:, 0], 2.5), a_hi=np.percentile(boots[:, 0], 97.5),
                        alpha=al, alpha_lo=np.percentile(boots[:, 1], 2.5), alpha_hi=np.percentile(boots[:, 1], 97.5),
                        R2=r2, npts=int(mask.sum()))
    # bound check: err <= 3 R G / sqrt(T) + 2 M gamma
    G = np.array(r['G'])
    bound = 3 * r['R'] * G[None, :] / np.sqrt(T)[:, None] + 2 * r['M'] * r['gamma']
    out['max_err_over_bound'] = float((err / bound).max())
    out['median_err_over_bound_noisedom'] = float(np.median((err / bound)[:, lev >= 10]))
    return out, E, T, L, noiseless


def plot_e31(files, fname, title, rng=None):
    rr = [load(f) for f in files]
    rr = [r for r in rr if r is not None]
    if not rr:
        return
    fig, axs = plt.subplots(2, len(rr), figsize=(3.1 * len(rr), 5.6), squeeze=False)
    for j, r in enumerate(rr):
        st, E, T, L, noiseless = e31_analysis(r, rng)
        S.setdefault(f"e31_{r.get('variant', r['problem'] + '_d' + str(r['d']))}", st)
        sc = r.get('level_scale', 1.0)
        wfac = float(omega_bar(r.get('noise', 'beta'), 1.0))    # omega_bar per unit Delta (per noise)
        ax = axs[0, j]
        ax.loglog(T, noiseless, 'k-', lw=1.2, label=r'$\Delta=0$')
        for il, l in enumerate(L):
            if l == 0:
                continue
            g = np.exp(np.log(E[:, il, :]).mean(1))
            lo, hi = np.exp(np.log(E[:, il, :]).min(1)), np.exp(np.log(E[:, il, :]).max(1))
            ax.loglog(T, g, 'o-', ms=2.5, color=LEVCOL[int(l)], label=fr'{int(l)}$\Delta_{{\rm ref}}$')
            ax.fill_between(T, lo, hi, color=LEVCOL[int(l)], alpha=0.2, lw=0)
            Gl = np.array(r['G'])[np.array(r['levels']) == l][0]
            ax.loglog(T, 3 * r['R'] * Gl / np.sqrt(T) + 2 * r['M'] * r['gamma'], '--', lw=0.8, color=LEVCOL[int(l)])
        ax.axhline(r['f0'] - r['fstar'], color='gray', lw=0.6, ls=':')
        ax.set_title(f"{r['problem']}, d={r['d']}" + (f" [{r['variant']}]" if 'variant' in r else '')
                     + (f"\nlevels x{sc:.3f}" if sc != 1.0 else ''))
        ax.set_xlabel('rounds $T$ (2 calls each)')
        ax.set_ylabel(r'$f(\bar x_T)-f^*$')
        ax.set_ylim(max(noiseless.min() / 3, 1e-9), None)
        if j == 0:
            ax.legend(loc='lower left', ncol=2)
        ax = axs[1, j]
        for il, l in enumerate(L):
            if l < 10:
                continue
            u = l * sc / np.sqrt(T)
            for k in range(E.shape[2]):
                e = E[:, il, k]
                nd = e >= 5 * noiseless
                ax.loglog(u[nd], e[nd], 'o', ms=2.6, color=LEVCOL[int(l)], alpha=0.8)
                ax.loglog(u[~nd], e[~nd], 'o', ms=2.6, mfc='none', color=LEVCOL[int(l)], alpha=0.5)
        uu = np.logspace(np.log10(10 * sc / np.sqrt(T.max())), np.log10(300 * sc / np.sqrt(T.min())), 50)
        theory = 3 * r['R'] * r['d'] * wfac * r['Delta_ref'] * uu / (2 * r['gamma'])
        ax.loglog(uu, theory, 'k--', lw=1, label=r'lower-bound theorem, noise part $3R\,d\bar\omega/(2\gamma\sqrt{T})$')
        if 'all' in st:
            ax.text(0.03, 0.97, f"$a={st['all']['a']:.3f}$ [{st['all']['a_lo']:.3f}, {st['all']['a_hi']:.3f}]\n"
                    f"slope in $\\Delta/\\sqrt{{T}}$: {st['all']['alpha']:.2f}", transform=ax.transAxes, va='top',
                    fontsize=7.5, bbox=dict(fc='white', ec='0.7', lw=0.5))
        ax.set_xlabel(r'$(\Delta/\Delta_{\rm ref})/\sqrt{T}$')
        ax.set_ylabel(r'$f(\bar x_T)-f^*$')
        if j == 0:
            ax.legend(loc='lower right', fontsize=6.5)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, fname))
    plt.close(fig)


plot_e31(['e31_logistic_d20.json', 'e31_logistic_d80.json', 'e31_l1_d20.json', 'e31_l1_d80.json'], 'e31.pdf',
         r'E3.1: top, error vs $T$ (dashed: theorem bound); bottom, collapse in $\Delta/\sqrt{T}$ (filled: noise-dominated)')
plot_e31(['e31s_instance1.json', 'e31s_pm1.json', 'e31s_gamma2.json', 'e31s_l1_gamma5.json'], 'e31_sensitivity.pdf',
         'E3.1 sensitivity (d = 20, T <= 2e5): other instance, $\\pm1$ noise, $\\gamma\\times2$, $\\ell_1$ with $\\gamma\\times5$')
# +-1 noise at nominal levels vs. at levels x0.378 (same omega_bar as the Beta runs)
plot_e31(['e31s_pm1.json', 'e31s_pm1_eqomega.json'], 'e31_pm1_eqomega.pdf',
         'E3.1, $\\pm1$ noise (d = 20): nominal levels vs. levels with the $\\bar\\omega$ of the Beta runs',
         rng=np.random.default_rng(1))

# ======================================================================================== E3.2
a = load('e32a.json')
b20, b80 = load('e32b_d20.json'), load('e32b_d80.json')
if a is not None:
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.5))
    ax = axs[0]
    zmax = 0
    for d, col in (('20', '#2a78d6'), ('80', '#e8743b')):
        rows = a['rows'][d]
        rr = np.array([r_['ratio'] for r_ in rows])
        norm = int(d) ** 2 * a['Delta'] ** 2 / (4 * a['gamma'] ** 2)
        m = np.array([np.mean(r_['mean']) for r_ in rows]) / norm
        se = np.array([np.sqrt(np.sum(np.square(r_['se']))) / len(r_['se']) for r_ in rows]) / norm
        th = np.array([r_['theory'] for r_ in rows]) / norm
        z = (m - th) / se
        zmax = max(zmax, np.abs(z).max())
        S[f'e32a_d{d}'] = dict(max_abs_z=float(np.abs(z).max()), max_rel_dev=float(np.abs(m / th - 1).max()))
        ax.errorbar(rr, m, yerr=1.96 * se, fmt='o', ms=3, color=col, label=f'MC, d={d} (5 seeds, 95% CI)')
        xx = np.logspace(-2, 2.5, 300)
        ax.loglog(xx, 1 - Phi_sphere(int(d), 2 / xx), '-', color=col, lw=1, label=fr'$1-\Phi_{{{d}}}(2\gamma/\ell)$ (exact)')
        ax.loglog(xx, np.minimum(1, 2 / (int(d) * xx ** 2)), ':', color=col, lw=0.8)
    xx = np.logspace(-2, 2.5, 300)
    ax.loglog(xx, np.minimum(1, 1 / xx ** 2), 'k--', lw=0.9, label=r'hypothesis $\min\{1,(\gamma/\ell)^2\}$')
    ax.set_xlabel(r'$\ell/\gamma$')
    ax.set_ylabel(r'$\mathrm{E}\|n\|^2\,/\,(d^2\Delta^2/4\gamma^2)$')
    ax.set_title('E3.2a: noise part of the estimator (dotted: $\\min\\{1,2\\gamma^2/(d\\ell^2)\\}$)')
    ax.set_ylim(1e-6, 3)
    ax.legend(fontsize=6.3, loc='lower left')
    ax = axs[1]
    for bb, col in ((b20, '#2a78d6'), (b80, '#e8743b')):
        if bb is None:
            continue
        d = bb['d']
        es = np.array(bb['err_sin'])[-1]                    # T = 1e5
        en = np.array(bb['err_none'])[-1]
        n = bb['nrep']
        ratios = np.array(bb['ratios'])
        es = es.reshape(len(ratios), n)
        base = es[0] - en.mean()                              # ell = 0.01 gamma: rough limit
        rows = []
        for i, rt in enumerate(ratios[:-1]):
            c = (es[i] - en.mean()) / base.mean()
            m, lo, hi = ci(c)
            rows.append([rt, m, lo, hi, bb['theory_cost'][i]])
        rows = np.array(rows)
        S[f'e32b_d{d}'] = dict(rows=rows.tolist(), err_none=float(en.mean()), err_rough=float(es[0].mean()),
                               err_smooth_limit=float(es[-1].mean()), err_beta=float(np.mean(np.array(bb['err_beta'])[-1])))
        ax.errorbar(rows[:, 0], rows[:, 1], yerr=[rows[:, 1] - rows[:, 2], rows[:, 3] - rows[:, 1]], fmt='o', ms=3,
                    color=col, label=f'logistic, d={d}, T=$10^5$')
        ax.semilogx(xx, 1 - Phi_sphere(d, 2 / xx), '-', color=col, lw=1)
        ax.axhline((np.mean(np.array(bb['err_beta'])[-1]) - en.mean()) / base.mean(), color=col, ls='-.', lw=0.7)
    ax.semilogx(xx, np.minimum(1, 1 / xx ** 2), 'k--', lw=0.9, label=r'hypothesis $\min\{1,(\gamma/\ell)^2\}$')
    ax.set_xlabel(r'$\ell/\gamma$')
    ax.set_ylabel('excess error / excess error at $\\ell=0.01\\gamma$')
    ax.set_title('E3.2b: cost of noise in optimisation (dash-dot: Beta hash, same $\\bar\\omega$)')
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'e32.pdf'))
    plt.close(fig)

# ======================================================================================== E3.4
r = load('e34.json')
if r is not None:
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.5))
    rows_s = []
    for row in r['rows']:
        d, T = row['d'], row['T']
        ax = axs[0 if d == 20 else 1]
        A = np.array(row['a']); err = np.array(row['err']); LB = np.array(row['LB']); eta = np.array(row['eta_scale'])
        am = A / row['astar']
        best = []
        for m_ in r['amult']:
            cand = []
            for es in r['etas']:
                sel = (np.isclose(am, m_)) & (eta == es)
                cand.append(ci(err[sel]) + (es,))
            bm = min(cand, key=lambda c: c[0])
            lb = LB[np.isclose(am, m_)][0]
            best.append((m_, *bm[:3], lb, bm[3]))
            rows_s.append(dict(d=d, T=T, amult=m_, err=bm[0], lo=bm[1], hi=bm[2], LB=lb, eta=bm[3],
                               ratio=bm[1] / lb if lb > 0 else None))
        best = np.array(best)
        col = {1000: '#7fb3e6', 10000: '#2a78d6', 100000: '#e8743b'}[T]
        scale = row['astar'] * r['R'] * np.sqrt(d)
        ax.errorbar(best[:, 0], best[:, 1] / scale, yerr=[(best[:, 1] - best[:, 2]) / scale, (best[:, 3] - best[:, 1]) / scale],
                    fmt='o-', ms=3, color=col, lw=0.8, label=f'ZO-SGD, best step, T={T:g}')
        xx = np.logspace(np.log10(1 / 16), np.log10(16), 200)
        aa = xx * row['astar']
        lbc = (aa * r['R'] / (4 * np.sqrt(d))) * np.maximum(d - 2 * r['gamma'] * aa * np.sqrt(d * r['I'] * T) / r['Delta'], 0)
        ax.loglog(xx, lbc / scale, '--', color=col, lw=0.9)
    for k, ax in enumerate(axs):
        ax.loglog([1 / 16, 16], [1 / 16, 16], color='0.6', lw=0.6, ls=':')
        ax.set_xlabel(r'$a/a^*$,  $a^*=\sqrt{d}\,\Delta/(4\gamma\sqrt{IT})$')
        ax.set_ylabel(r'$\mathrm{E}[f_v(\hat x)-f^*]\;/\;(a^*R\sqrt{d})$')
        ax.set_title(f'E3.4: hard instance, d={20 if k == 0 else 80} (dashed: Assouad lower bound)')
        ax.legend(fontsize=6.5, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'e34.pdf'))
    plt.close(fig)
    S['e34'] = rows_s
    valid = [x for x in rows_s if x['LB'] > 0]
    S['e34_min_lo_over_LB'] = min(x['lo'] / x['LB'] for x in valid)
    S['e34_at_astar'] = [x for x in rows_s if x['amult'] == 1]

# ======================================================================================== E3.5
r = load('e35.json')
if r is not None:
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    T = np.array(r['checkpoints'])
    sty = {'none': ('k', '-', 'noiseless'), 'pm1_decoded_10': ('#19a979', 'o', r'$\pm1$, decoded, 10$\Delta_{\rm ref}$'),
           'pm1_decoded_100': ('#19a979', 's', r'$\pm1$, decoded, 100$\Delta_{\rm ref}$'),
           'pm1_raw_10': ('#e8743b', '--', r'$\pm1$, raw, 10$\Delta_{\rm ref}$'), 'pm1_raw_100': ('#e8743b', ':', r'$\pm1$, raw, 100$\Delta_{\rm ref}$'),
           'beta_raw_10': ('#2a78d6', '--', r'Beta, raw, 10$\Delta_{\rm ref}$'), 'beta_raw_100': ('#2a78d6', ':', r'Beta, raw, 100$\Delta_{\rm ref}$')}
    for k, v in r['runs'].items():
        v = np.array(v)
        g = np.exp(np.log(v).mean(1))
        c, ls, lab = sty[k]
        if ls in ('o', 's'):
            ax.loglog(T, g, ls, color=c, mfc='none', ms=6, label=lab)
        else:
            ax.loglog(T, g, ls, color=c, lw=1.2, label=lab)
    dn = np.array(r['runs']['none'])
    S['e35'] = dict(max_abs_diff_decoded_vs_none=float(max(np.abs(np.array(r['runs'][f'pm1_decoded_{l}']) - dn).max() for l in (10, 100))),
                    final={k: float(np.exp(np.log(np.array(v)[-1]).mean())) for k, v in r['runs'].items()},
                    two_M_gamma_over_Dref=r['two_M_gamma'] / r['Delta_ref'])
    ax.set_xlabel('rounds $T$')
    ax.set_ylabel(r'$f(\bar x_T)-f^*$ (geo. mean, 5 runs)')
    ax.set_title(r'E3.5: $\pm1$ hash noise is decodable when $\Delta>2M\gamma$')
    ax.legend(fontsize=6.5, ncol=1, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'e35.pdf'))
    plt.close(fig)

# ======================================================================================== E3.3 (analytic)
fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.3))
x = np.logspace(0.5, 4, 200)            # MR/eps
for ax, d in zip(axs, (20, 80)):
    lo2 = 1 / (4 * d) * (1 / x) / 4    # eps gamma sqrt T/(4 d R) with gamma = eps/(4M): (1/(16 d)) (eps/MR)
    up2 = 32 * np.pi / d * (1 / x) / 4  # 32 pi eps gamma sqrt T/(d R) with gamma = eps/(4M)
    ax.fill_between(x, lo2, up2, color='#2a78d6', alpha=0.25, lw=0, label=r'two-point, radius $\varepsilon/4M$: proven bracket')
    # the cited bandit rates are for losses in [0,1]; after rescaling (Cor. 4.17) the bandit route
    # guarantees Delta_band(T) = eps sqrt(T-2)/(8 sqrt(C_b) d^{k/2}) - MR/2 - M gamma (zero if negative).
    # Drawn at the budget T = kappa d^k (MR/eps)^2 (C_b = 1, gamma = eps/(4M), T-2 ~ T), in units of eps sqrt(T):
    # (1/d^{k/2}) (1/8 - 1/(2 sqrt(kappa)) - 1/(4 sqrt(kappa) MR/eps)); identically zero for kappa <= 16.
    for k, ls in ((3.0, '-'), (3.5, '--')):
        for kappa, lw in ((256, 1.1), (64, 0.7)):
            yb = d ** (-k / 2) * (1 / 8 - 1 / (2 * np.sqrt(kappa)) - 1 / (4 * np.sqrt(kappa) * x))
            ax.loglog(x, np.where(yb > 0, yb, np.nan), ls, color='#e8743b', lw=lw,
                      label=fr'bandit route (cited), $k={k:g}$, $T={kappa}\,d^{{k}}(MR/\varepsilon)^2$, $C_b=1$')
    ax.loglog(x, np.full_like(x, 16 * np.sqrt(2) * np.pi / np.sqrt(d)), 'k-', lw=1, label='any algorithm, upper bound (Hellinger-Assouad)')
    ax.set_xlabel(r'$MR/\varepsilon$')
    ax.set_ylabel(r'admissible $\Delta\,/\,(\varepsilon\sqrt{T})$')
    ax.set_title(f'E3.3 (analytic, not an experiment), d={d}')
    ax.axvline(np.sqrt(d), color='0.5', lw=0.6, ls=':')
h_, l_ = axs[0].get_legend_handles_labels()
fig.legend(h_, l_, loc='lower center', ncol=3, fontsize=6.8, frameon=False)
fig.tight_layout(rect=(0, 0.17, 1, 1))
fig.savefig(os.path.join(FIGS, 'e33_analytic.pdf'))
plt.close(fig)

json.dump(S, open(os.path.join(DATA, 'summary_stats.json'), 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in S.items() if not k.startswith('e34')}, indent=1, default=float)[:4000])
