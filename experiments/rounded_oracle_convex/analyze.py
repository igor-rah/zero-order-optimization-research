import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from exp import ulp_p

import os
D = os.path.dirname(os.path.abspath(__file__))
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
COL = {None: '#52514e', 14: '#86b6ef', 11: '#2a78d6', 8: '#104281'}
LAB = {None: 'exact values', 14: 'p = 14 bits', 11: 'p = 11 (≈ fp16)', 8: 'p = 8 (≈ bf16)'}
MK = {None: 'o', 14: 's', 11: '^', 8: 'D'}
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.facecolor': SURF,
                     'figure.facecolor': SURF, 'axes.spines.top': False, 'axes.spines.right': False})
P = [None, 14, 11, 8]


def load(name, seeds):
    J = json.load(open(f'{D}/{name}.json'))
    mus = np.array(J['None_0']['mus'])
    fl = {p: np.array([J[f'{p}_{s}']['floor'] for s in seeds]) for p in P}
    fstar = np.mean([J[f'None_{s}']['fstar'] for s in seeds])
    Delta = {p: ulp_p(fstar, p) / 2 for p in (14, 11, 8)}
    med = {p: np.median(np.maximum(fl[p], 1e-17), 0) for p in P}
    emin = {p: med[p].min() for p in P}
    imin = {p: int(np.argmin(med[p])) for p in P}
    ad = {p: np.median([J[f'adapt{p}_{s}']['floor'][0] for s in seeds]) for p in (11, 8)}
    return dict(J=J, mus=mus, fl=fl, med=med, emin=emin, imin=imin, Delta=Delta, ad=ad, seeds=seeds)


def width(mus, med, eps):
    ok = med <= eps
    if not ok.any():
        return np.nan
    i0 = int(np.argmin(med)); lo = hi = i0
    while lo - 1 >= 0 and ok[lo - 1]: lo -= 1
    while hi + 1 < len(med) and ok[hi + 1]: hi += 1
    return np.log10(mus[hi] / mus[lo]) + np.log10(mus[1] / mus[0])


S = {'logreg': load('main', (0, 1, 2)), 'huber': load('huber', (0, 1))}
TITLE = {'logreg': 'logistic regression (very smooth)', 'huber': 'Huber regression (β = 2: only the gradient is Lipschitz)'}

fig, axs = plt.subplots(2, 3, figsize=(17, 10))
ax = axs.ravel()
for j, key in enumerate(['logreg', 'huber']):
    s = S[key]; a = ax[j]
    for p in P:
        a.fill_between(s['mus'], np.maximum(s['fl'][p].min(0), 1e-17), np.maximum(s['fl'][p].max(0), 1e-17), color=COL[p], alpha=0.15, lw=0)
        a.plot(s['mus'], s['med'][p], '-', marker=MK[p], ms=4, color=COL[p], lw=2, label=LAB[p])
    a.set_xscale('log'); a.set_yscale('log'); a.set_ylim(1e-17, 1)
    a.set_title(f'{"AB"[j]}. f(x̄) − f* vs μ\n{TITLE[key]}', loc='left', color=INK, fontsize=11)
    a.set_xlabel('μ (stencil radius)'); a.set_ylabel('f(x̄) − f*  (median over instances)')
    a.grid(color=GRID, lw=0.8); a.legend(frameon=False, fontsize=9, loc='lower left')

# C: eps_min vs Delta
slopes = {}
for key, ls, mfc in (('logreg', '-', None), ('huber', '--', 'none')):
    s = S[key]
    ds = np.array([s['Delta'][p] for p in (14, 11, 8)]); es = np.array([s['emin'][p] for p in (14, 11, 8)])
    k, b = np.polyfit(np.log(ds), np.log(es), 1); slopes[key] = k
    for p in (14, 11, 8):
        ax[2].plot(s['Delta'][p], s['emin'][p], MK[p], ms=9, color=COL[p], mfc=mfc if mfc else COL[p], mew=1.8)
    xx = np.logspace(np.log10(ds.min()) - 0.2, np.log10(ds.max()) + 0.2, 20)
    ax[2].plot(xx, np.exp(b) * xx ** k, ls, color=INK2, lw=1.5, label=f'{"logreg" if key=="logreg" else "Huber"}: slope {k:.2f}')
ax[2].plot([], [], 'o', color=INK2, label='filled: logreg'); ax[2].plot([], [], 'o', color=INK2, mfc='none', label='hollow: Huber')
ax[2].set_xscale('log'); ax[2].set_yscale('log')
ax[2].set_title('C. ε_min vs rounding level Δ\n(marker = precision p, as in A/B)', loc='left', color=INK, fontsize=11)
ax[2].set_xlabel('Δ = half grid step near f*'); ax[2].set_ylabel('ε_min = min over μ')
ax[2].grid(color=GRID, lw=0.8); ax[2].legend(frameon=False, fontsize=9, loc='upper left')

# D: window width vs eps
eps_grid = np.logspace(-16, -1, 150)
for key, ls in (('logreg', '-'), ('huber', '--')):
    s = S[key]
    for p in (14, 11, 8):
        ax[3].plot(eps_grid, [width(s['mus'], s['med'][p], e) for e in eps_grid], ls, color=COL[p], lw=2,
                   label=f'{LAB[p]}, {"logreg" if key=="logreg" else "Huber"}')
ax[3].set_xscale('log')
ax[3].set_title('D. Width of the window of good μ for target ε\n(decades; line ends = window closed, ε < ε_min)', loc='left', color=INK, fontsize=11)
ax[3].set_xlabel('target accuracy ε'); ax[3].set_ylabel('window width, decades of μ')
ax[3].grid(color=GRID, lw=0.8); ax[3].legend(frameon=False, fontsize=8, loc='upper left', ncol=2)

# E: dimension
Dm = json.load(open(f'{D}/dim.json'))
dims = sorted({int(k.split('_')[0]) for k in Dm})
ratio, edge = [], []
for d in dims:
    em = [np.min(Dm[f'{d}_{s}']['floor']) for s in (0, 1)]
    ix = [int(np.argmin(Dm[f'{d}_{s}']['floor'])) for s in (0, 1)]
    fs = np.mean([Dm[f'{d}_{s}']['fstar'] for s in (0, 1)])
    ratio.append(np.median(em) / (ulp_p(fs, 11) / 2)); edge.append(max(ix) == len(Dm[f'{d}_0']['mus']) - 1)
kd, bd = np.polyfit(np.log(dims), np.log(ratio), 1)
ax[4].plot(dims, ratio, '^', ms=9, color=COL[11], label='p = 11, logreg: ε_min / Δ')
dd = np.array([dims[0] / 1.3, dims[-1] * 1.3])
ax[4].plot(dd, np.exp(bd) * dd ** kd, '-', color=INK2, lw=1.5, label=f'fit: slope {kd:.2f}')
ax[4].plot(dd, ratio[0] * (dd / dims[0]) ** 0.5, ':', color=INK2, lw=1.5, label='slope 1/2 (√d), reference')
ax[4].set_xscale('log'); ax[4].set_yscale('log')
ax[4].set_title('E. ε_min / Δ vs dimension d', loc='left', color=INK, fontsize=11)
ax[4].set_xlabel('dimension d (n = 10 d)'); ax[4].set_ylabel('ε_min / Δ')
ax[4].grid(color=GRID, lw=0.8); ax[4].legend(frameon=False, fontsize=9, loc='upper left')

# F: adaptive mu vs best fixed mu
lab, best, adap, colors = [], [], [], []
for key in ('logreg', 'huber'):
    for p in (11, 8):
        lab.append(f'{"logreg" if key=="logreg" else "Huber"}\n{LAB[p]}')
        best.append(S[key]['emin'][p]); adap.append(S[key]['ad'][p])
x = np.arange(len(lab)); w = 0.38
ax[5].bar(x - w / 2, best, w * 0.92, color='#2a78d6', label='best fixed μ (grid-tuned)')
ax[5].bar(x + w / 2, adap, w * 0.92, color='#eb6834', label='adaptive μ (no tuning)')
for xi, v in zip(x - w / 2, best): ax[5].text(xi, v * 1.3, f'{v:.1e}', ha='center', fontsize=8, color=INK2)
for xi, v in zip(x + w / 2, adap): ax[5].text(xi, v * 1.3, f'{v:.1e}', ha='center', fontsize=8, color=INK2)
ax[5].set_yscale('log'); ax[5].set_xticks(x, lab, fontsize=8.5)
ax[5].set_title('F. Adaptive μ vs best fixed μ', loc='left', color=INK, fontsize=11)
ax[5].set_ylabel('f(x̄) − f*'); ax[5].grid(axis='y', color=GRID, lw=0.8); ax[5].legend(frameon=False, fontsize=9, loc='upper left')
for a_ in ax: a_.set_axisbelow(True)
fig.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.08, hspace=0.42, wspace=0.25)
fig.savefig(f'{D}/zo_convex_rounding.png', dpi=150)

# ---- numbers ----
for key, s in S.items():
    print('==', key, 'Delta', {p: f'{v:.2e}' for p, v in s['Delta'].items()})
    for p in P:
        i = s['imin'][p]
        print(f"  p={p}: eps_min={s['emin'][p]:.2e} at mu={s['mus'][i]:.3g} (edge={i in (0, len(s['mus'])-1)})")
    print('  slope eps_min~Delta^k: k =', round(slopes[key], 3))
    for p in (11, 8):
        med = s['med'][p]; i = s['imin'][p]
        sel = [j for j in range(i) if med[j] < 0.3 * med[0] and med[j] > 5 * med[i]]
        if len(sel) >= 2:
            print(f'  p={p} left-branch slope (floor~mu^k):', round(np.polyfit(np.log(s['mus'][sel]), np.log(med[sel]), 1)[0], 2), 'npts', len(sel))
    med = s['med'][None]; i0 = s['imin'][14]
    sel = [j for j in range(len(med)) if s['mus'][j] >= s['mus'][s['imin'][8]] and med[j] > 1e-14]
    if len(sel) >= 2:
        print('  exact right-branch slope (smoothing bias, floor~mu^k):', round(np.polyfit(np.log(s['mus'][sel]), np.log(med[sel]), 1)[0], 2), 'npts', len(sel))
    for p in (14, 11, 8):
        print(f'  p={p} window widths (dec) at eps=1e-4,1e-6,1e-8:', [round(width(s['mus'], s['med'][p], e), 2) for e in (1e-4, 1e-6, 1e-8)])
    print('  adaptive:', {p: f"{s['ad'][p]:.2e}" for p in (11, 8)})
print('dims', dims, 'ratio', [f'{r:.2e}' for r in ratio], 'edge', edge, 'slope', round(kd, 2))
