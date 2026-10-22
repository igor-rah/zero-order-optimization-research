"""Figures and numbers.json for the Markov paper. Uses ONLY logged runs."""
import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures')
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#ffffff'
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
TAUC = {0: INK2, 1: '#86b6ef', 16: '#2a78d6', 256: '#104281'}
plt.rcParams.update({'font.size': 7.5, 'font.family': 'serif', 'mathtext.fontset': 'cm',
    'axes.edgecolor': '#b8b7b2', 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'axes.titlesize': 8, 'axes.titleweight': 'bold'})
def grid(a): a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
NUM = {}
gm = lambda a: float(np.exp(np.mean(np.log(np.asarray(a)), 0))) if np.ndim(a) == 1 else np.exp(np.mean(np.log(np.asarray(a)), 0))

# ======================= convex =======================
R = json.load(open(os.path.join(HERE, 'convex', 'chain2.json')))
E = json.load(open(os.path.join(HERE, 'convex', 'chain3_exact.json')))
mus = np.array(R['full0_0']['mus'])
keys = [('full0', 'full batch', 0), ('mb1', r'$\tau_{\rm data}=1$', 1), ('mb16', r'$\tau_{\rm data}=16$', 16), ('mb256', r'$\tau_{\rm data}=256$', 256)]
conv = {}
for k, lab, t in keys:
    g = lambda D, f: np.array([D[f'{k}_{s}'][f] for s in (0, 1)])
    conv[k] = dict(floor=gm(g(R, 'floor')), floor_exact=gm(np.maximum(g(E, 'floor'), 1e-16)), tie=g(R, 'tie').mean(0), hold=g(R, 'hold').mean(0),
                   rho={l: np.mean([R[f'{k}_{s}']['rho'][l] for s in (0, 1)], 0) for l in ('1', '4', '16', '64', '256')},
                   f0=float(np.mean(g(R, 'f0'))), sF=float(np.mean(g(R, 'sF_over_u'))))
NUM['convex'] = {k: {kk: (v.tolist() if isinstance(v, np.ndarray) else ({l: x.tolist() for l, x in v.items()} if isinstance(v, dict) else v))
                     for kk, v in d.items()} for k, d in conv.items()}
NUM['convex']['mus'] = mus.tolist()

fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7))
for k, lab, t in keys:
    c = TAUC[t]
    fl = np.array(conv[k]['floor']); div = fl > 1.0
    ax[0].plot(mus, np.where(div, np.nan, fl), color=c, lw=1.6, marker='o', ms=2.5, label=lab)
    ax[0].scatter(mus[div], np.full(div.sum(), 30.0), color=c, marker='x', s=14, lw=1.2)
    ax[0].plot(mus, conv[k]['floor_exact'], color=c, lw=1.0, ls='--')
ax[0].axhline(conv['full0']['f0'], color=INK2, lw=0.7, ls=':')
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_ylim(1e-16, 200)
ax[0].set_xlabel(r'radius $\mu$'); ax[0].set_ylabel(r'$f(\bar x)-f^\star$'); ax[0].set_title('(a) accuracy')
ax[0].legend(fontsize=6, loc='lower left'); grid(ax[0])
for k, lab, t in keys:
    ax[1].plot(mus, conv[k]['hold'], color=TAUC[t], lw=1.6, marker='o', ms=2.5, label=lab)
ax[1].set_xscale('log'); ax[1].set_ylim(-0.03, 1.03); ax[1].set_xlabel(r'radius $\mu$'); ax[1].set_ylabel('fraction of steps with $x_{k+1}=x_k$')
ax[1].set_title('(b) laziness'); grid(ax[1])
lags = np.array([1, 4, 16, 64, 256]); ll = np.logspace(0, np.log10(300), 100)
i_small = int(np.argmin(np.abs(mus - 1e-4)))
for k, lab, t in keys:
    rho = [conv[k]['rho'][str(l)][i_small] for l in lags]
    ax[2].plot(lags, rho, color=TAUC[t], lw=0, marker='o', ms=4, label=lab)
    if t > 0:
        ax[2].plot(ll, np.clip(1 - ll / t, 0, None), color=TAUC[t], lw=1.0, ls='--')
ax[2].set_xscale('log'); ax[2].set_ylim(-0.03, 1.05); ax[2].set_xlabel(r'lag $\ell$'); ax[2].set_ylabel(r'offset autocorrelation $\rho(\ell)$')
ax[2].set_title(r'(c) offset memory at $\mu=10^{-4}$'); grid(ax[2])
fig.tight_layout(); fig.savefig(f'{OUT}/fig_convex.pdf'); fig.savefig(f'{OUT}/fig_convex.png', dpi=150); plt.close(fig)

# ======================= MeZO pilot =======================
S = json.load(open(os.path.join(HERE, 'summary.json')))
NUM['mezo'] = {k: {kk: v for kk, v in d.items() if kk not in ('curve', 'kappa_win', 'tie_win')} for k, d in S.items()}
def run(mode, tau, mu=1e-3, lr=5e-4):
    return S.get(f'{mode}_tau{tau}_mu{mu:g}_lr{lr:g}_s0')
fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.2)); ax = ax.ravel()
for tau in (1, 16, 256):
    for mode, ls in (('fp32', '-'), ('rtn', '--')):
        r = run(mode, tau)
        if r is None: continue
        cv = np.array(r['curve'])
        ax[0].plot(cv[:, 0], cv[:, 1], color=TAUC[tau], ls=ls, lw=1.5, label=f'{mode.upper() if mode=="rtn" else mode}, ' + r'$\tau_{\rm data}=$' + f'{tau}')
ax[0].set_xlabel('MeZO step'); ax[0].set_ylabel('eval loss'); ax[0].set_title(r'(a) MeZO, $\mu=10^{-3}$: fp32 (solid) vs bf16 RTN (dashed)')
ax[0].legend(fontsize=5.8, ncol=2); grid(ax[0])
for tau in (1, 16, 256, 2000):
    r = run('rtn', tau)
    if r is None: continue
    col = TAUC.get(tau, INK)
    rho = [r['rho'][str(l)] for l in lags]
    ax[1].plot(lags, rho, color=col, lw=0, marker='o', ms=4, label=r'$\tau_{\rm data}=$' + (f'{tau}' if tau < 2000 else r'$\infty$'))
    if 1 < tau < 2000:
        ax[1].plot(ll, np.clip(1 - ll / tau, 0, None), color=col, lw=1.0, ls='--')
ax[1].set_xscale('log'); ax[1].set_ylim(-0.03, 1.05); ax[1].set_xlabel(r'lag $\ell$'); ax[1].set_ylabel(r'$\rho(\ell)$')
ax[1].set_title('(b) offset memory follows the data chain'); ax[1].legend(fontsize=6); grid(ax[1])
DR = json.load(open(os.path.join(HERE, 'drift.json')))
xs4 = [1, 16, 256, 2000]; xl = ['1', '16', '256', r'$\infty$']; xi = np.arange(4)
tie_o = [run('rtn', t)['tie'] for t in xs4]; tie_p = [run('rtn', t)['tie_pred_uniform'] for t in xs4]
kap = [run('rtn', t)['kappa_hat'] for t in xs4]; intr = [DR[f'rtn_tau{t}_mu0.001_lr0.0005_s0']['frac_interior'] for t in xs4]
ax[2].plot(xi, tie_o, color=C2, lw=1.6, marker='o', ms=4, label='tie fraction (observed)')
ax[2].plot(xi, tie_p, color=C2, lw=1.0, ls='--', marker='o', ms=3, mfc='w', label='tie fraction (uniform offset)')
ax[2].plot(xi, kap, color=C1, lw=1.6, marker='s', ms=4, label=r'gain $\hat\kappa$')
ax[2].plot(xi, intr, color=C3, lw=1.6, marker='^', ms=4, label=r'offsets with $|\omega|<1/4$')
ax[2].axhline(0.5, color=C3, lw=0.6, ls=':')
ax[2].set_xticks(xi); ax[2].set_xticklabels(xl); ax[2].set_ylim(-0.03, 1.05)
ax[2].set_xlabel(r'data mixing time $\tau_{\rm data}$'); ax[2].set_ylabel('fraction / gain')
ax[2].set_title(r'(c) bf16 RTN, $\mu=10^{-3}$: the iterate freezes its offset'); ax[2].legend(fontsize=5.8, loc='center right', bbox_to_anchor=(1.0, 0.42)); grid(ax[2])
xs = np.array([1, 16, 256])
series = (('fp32', 1e-3, 5e-4, C1, '-', r'MeZO fp32, $\mu=10^{-3}$'), ('rtn', 1e-2, 5e-4, C2, '-', r'MeZO bf16 RTN, $\mu=10^{-2}$'),
          ('sgd', 1e-3, 7e-4, C3, '-', r'SGD, $\eta=7\cdot10^{-4}$ (small step)'), ('sgd', 1e-3, 0.1, C3, '--', r'SGD, $\eta=0.1$ (tuned, 1000 steps)'))
NUM['ratios'] = {}
for mode, mu, lr, col, ls, lab in series:
    ys = np.array([np.nan if run(mode, t, mu, lr) is None else run(mode, t, mu, lr)['drop'] for t in xs])
    ax[3].plot(xs, ys / ys[0], color=col, ls=ls, lw=1.6, marker='o', ms=4, label=lab)
    NUM['ratios'][f'{mode}_mu{mu:g}_lr{lr:g}'] = dict(drops=ys.tolist(), ratio=(ys / ys[0]).tolist())
ax[3].axhline(1, color=INK2, lw=0.7, ls=':'); ax[3].axhline(0, color=INK2, lw=0.5)
ax[3].set_xscale('log'); ax[3].set_xlabel(r'data mixing time $\tau_{\rm data}$'); ax[3].set_ylabel(r'loss drop relative to $\tau_{\rm data}=1$')
ax[3].set_title('(d) sensitivity to Markovian data'); ax[3].legend(fontsize=5.8, loc='lower left'); grid(ax[3])
fig.tight_layout(); fig.savefig(f'{OUT}/fig_mezo.pdf'); fig.savefig(f'{OUT}/fig_mezo.png', dpi=150); plt.close(fig)
json.dump(NUM, open(os.path.join(HERE, 'figures', 'numbers.json'), 'w'), indent=1)
print('ok')
