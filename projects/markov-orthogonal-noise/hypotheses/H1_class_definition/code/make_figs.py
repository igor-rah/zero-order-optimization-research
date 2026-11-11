"""Build every figure of the report from data/*.json (no recomputation of experiments).
Also writes data/summary_stats.json with the numbers quoted in the report text."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from common import Psi, loglog_slope, t_ci

C = ['#2a78d6', '#eb6834', '#1baf7a']          # categorical slots 1-3 (validated all-pairs)
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
                     'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'axes.edgecolor': INK2, 'axes.linewidth': 0.6, 'grid.color': GRID, 'grid.linewidth': 0.5,
                     'lines.linewidth': 1.4, 'savefig.dpi': 200, 'figure.dpi': 110})
os.makedirs('figs', exist_ok=True)
S = {}

def save(fig, name):
    fig.savefig(f'figs/{name}.pdf', bbox_inches='tight'); fig.savefig(f'figs/{name}.png', bbox_inches='tight')
    plt.close(fig)

# ---------------- E1.1 ---------------------------------------------------------------------------
D = json.load(open('data/e1_1.json'))
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=False)
S['E1.1'] = {}
for ax, d in zip(axs, ('5', '20', '100')):
    r = D[d]; t = np.array(r['t']); th = np.array(r['theory'])
    al = np.array(r['along']); se = np.array(r['se_along']); ns = len(al)
    m = al.mean(0); sem = np.sqrt((se ** 2).sum(0)) / ns
    z = (m - th) / sem
    floor = 2 * sem
    tt = np.linspace(0.05, t.max(), 2000)
    ax.semilogy(tt, np.abs(tt / int(d) * Psi(int(d), tt)), color=INK, lw=1.2, label=r'theory $|t\Psi_d(t)|/d$')
    res = np.abs(m) > 2 * sem
    ax.errorbar(t[res], np.abs(m[res]), yerr=1.96 * sem[res], fmt='o', ms=3.2, color=C[0], ecolor=C[0],
                elinewidth=0.8, label='MC mean, 95% CI')
    ax.plot(t[~res], np.maximum(np.abs(m[~res]), 1e-12), 'o', ms=3.2, mfc='white', mec=C[0], mew=0.8,
            label=r'MC, $|\mathrm{est}|<2$ s.e.')
    ax.plot(t, floor, ls='--', color=INK2, lw=0.9, label='MC resolution (2 s.e.)')
    for zz in r['zeros']:
        ax.axvline(zz, color=C[1], lw=0.8, ls=':')
    ax.set_ylim(max(1e-9, floor.min() / 20), 0.5)
    ax.set_title(f'$d={d}$'); ax.set_xlabel(r'$t=\gamma|\omega|$'); ax.grid(True, which='major')
    S['E1.1'][d] = dict(max_abs_z=float(np.abs(z).max()), frac_abs_z_gt2=float(np.mean(np.abs(z) > 2)),
                        n_t=int(len(t)))
axs[0].set_ylabel(r'$|\langle \mathbb{E}_e[m(x+\gamma e)e],\hat\omega\rangle|$  (units of $\Delta$)')
h_, l_ = axs[0].get_legend_handles_labels()
h_.append(plt.Line2D([], [], color=C[1], ls=':', lw=0.8)); l_.append('zeros $j_{d/2,k}$, $k=1,2,3$')
fig.legend(h_, l_, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=7)
fig.subplots_adjust(wspace=0.3); save(fig, 'e1_1_bessel_sweep')

# zero tests and superpositions -> table numbers
for d in ('5', '20', '100'):
    r = D[d]
    pz = np.array([p for s in r['zero_vec'] for p in s['p']])
    ratio = np.array([np.array(s['norm']) / np.array(s['noise_norm']) for s in r['zero_vec']]).ravel()
    sup = {}
    for key in ('sup', 'sup_detuned'):
        pv = np.concatenate([s['p_vs_zero'] for s in r[key]]); pt = np.concatenate([s['p_vs_theory'] for s in r[key]])
        thn = np.concatenate([s['theory_norm'] for s in r[key]]); nn = np.concatenate([s['noise_norm'] for s in r[key]])
        sup[key] = dict(frac_reject_zero=float(np.mean(pv < 0.05)), frac_reject_theory=float(np.mean(pt < 0.05)),
                        median_theory_norm=float(np.median(thn)), median_noise_norm=float(np.median(nn)), n=int(pv.size))
    S['E1.1'][d].update(zero_p_min=float(pz.min()), zero_frac_p_lt_005=float(np.mean(pz < 0.05)), zero_n=int(pz.size),
                        zero_norm_over_noise_median=float(np.median(ratio)), zero_norm_over_noise_max=float(ratio.max()),
                        sup=sup)

# ---------------- E1.3 ---------------------------------------------------------------------------
D3 = json.load(open('data/e1_3.json'))
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.6))
S['E1.3'] = {}
for ax, d in zip(axs, ('5', '20', '100')):
    r = D3[d]; rr = np.array(r['ratios']); W = np.array(r['w'])
    est = np.array(r['est']).reshape(len(r['est']), 3, -1); se = np.array(r['se']).reshape(len(r['se']), 3, -1)
    ns = est.shape[0]; m = est.mean(0); sem = np.sqrt((se ** 2).sum(0)) / ns
    th = np.array(r['theory']).reshape(3, -1)
    xx = np.logspace(np.log10(rr.min()), np.log10(rr.max()), 3000)
    zs = []
    for k in range(3):
        norm = W[k]                                     # plot |beta|/(Delta |w|)
        ax.loglog(xx, np.abs(Psi(int(d), xx * W[k])), color=C[k], lw=1.1,
                  label=f'$k={k+1}$ theory' if d == '5' else None)
        ok = np.abs(m[k]) > 2 * sem[k]
        ax.errorbar(rr[ok], np.abs(m[k][ok]) / norm, yerr=1.96 * sem[k][ok] / norm, fmt='o', ms=2.4, color=C[k],
                    elinewidth=0.7, label=f'$k={k+1}$ MC' if d == '5' else None)
        zs.append((m[k] - th[k]) / sem[k])
    floor = 2 * sem.mean(0) / W.mean()
    ax.plot(rr, floor, ls='--', color=INK2, lw=0.8, label='MC resolution' if d == '5' else None)
    ax.axvline(np.sqrt(2), color=INK2, lw=0.7, ls=':')
    ax.set_ylim(1e-7, 2); ax.set_xlabel(r'$\gamma_2/\gamma_1$'); ax.set_title(f'$d={d}$'); ax.grid(True, which='major')
    zs = np.array(zs)
    i2 = int(np.argmin(np.abs(rr - np.sqrt(2))))
    S['E1.3'][d] = dict(max_abs_z=float(np.abs(zs).max()), frac_abs_z_gt2=float(np.mean(np.abs(zs) > 2)),
                        n=int(zs.size), at_sqrt2_rel=[float(np.abs(m[k][i2]) / W[k]) for k in range(3)],
                        at_sqrt2_theory=[float(abs(Psi(int(d), np.array([np.sqrt(2) * W[k]]))[0])) for k in range(3)],
                        at_sqrt2_se_rel=[float(sem[k][i2] / W[k]) for k in range(3)],
                        small_radius_rel=[float(np.abs(Psi(int(d), np.array([0.1 * W[k]]))[0])) for k in range(3)])
axs[0].set_ylabel(r'$|\beta(\gamma_2)|/(\Delta|\omega|)=|\Psi_d(\gamma_2|\omega|)|$')
h_, l_ = axs[0].get_legend_handles_labels()
h_.append(plt.Line2D([], [], color=INK2, ls=':', lw=0.7)); l_.append(r'$\gamma_2/\gamma_1=\sqrt{2}$')
fig.legend(h_, l_, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=4, frameon=False, fontsize=7)
fig.subplots_adjust(wspace=0.3); save(fig, 'e1_3_two_radii')

# ---------------- E1.4 ---------------------------------------------------------------------------
D4 = json.load(open('data/e1_4.json'))
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
S['E1.4'] = {}
allz = []
for i, d in enumerate(('10', '40', '160')):
    r = D4[d]; dd = int(d); A, Sv = r['A'], r['S']
    tm = np.array([row['tmix'] for row in r['rows']])
    mean = np.array([np.mean([p['mean'] for p in row['per_seed']]) for row in r['rows']])
    sep = np.array([np.sqrt(np.sum([p['se'] ** 2 for p in row['per_seed']])) / len(row['per_seed']) for row in r['rows']])
    pred = np.array([row['pred'] for row in r['rows']]); bound = np.array([row['bound'] for row in r['rows']])
    z = (mean - pred) / sep; allz += list(z)
    nrm = dd                                               # units d Delta^2/gamma^2
    axs[0].errorbar(tm, mean / nrm, yerr=1.96 * sep / nrm, fmt='o', ms=3.5, color=C[i], label=f'$d={d}$ data')
    axs[0].plot(tm, pred / nrm, '-', color=C[i], lw=1.0)
    axs[0].plot(tm, bound / nrm, '--', color=C[i], lw=0.8)
    kap = (mean - A) / Sv; kap_se = sep / Sv
    alpha = np.array([row['alpha'] for row in r['rows']]); B = np.array([row['B'] for row in r['rows']])
    axs[1].errorbar(tm, kap, yerr=1.96 * kap_se, fmt='o', ms=3.5, color=C[i], label=f'$d={d}$')
    sl, h = loglog_slope(tm[1:], np.maximum(kap[1:], 1e-9))
    S['E1.4'][d] = dict(A=A, S=Sv, A_over_d2=A / dd ** 2, S_over_d=Sv / dd, z=z.tolist(),
                        rel_dev=((mean - pred) / pred).tolist(), slope_kappa_vs_tmix=[sl, h],
                        max_ratio_to_bound=float(np.max(mean / bound)))
tt = np.logspace(0, np.log10(400), 100)
al_ = 1 - 1 / np.array([1, 4, 16, 64, 256.])
axs[1].plot([row['tmix'] for row in D4['10']['rows']], [(1 + a) / (1 - a) for a in al_], '-', color=INK, lw=1.0,
            label=r'$(1+\alpha)/(1-\alpha)$')
axs[0].set_xscale('log'); axs[0].set_yscale('log'); axs[1].set_xscale('log'); axs[1].set_yscale('log')
axs[0].set_xlabel(r'mixing time $\tau=t_{\rm mix}(1/4)$'); axs[1].set_xlabel(r'mixing time $\tau=t_{\rm mix}(1/4)$')
axs[0].set_ylabel(r'$B\,\mathbb{E}\|\bar g\|^2$  (units of $d\Delta^2/\gamma^2$)')
axs[1].set_ylabel(r'$(B\,\mathbb{E}\|\bar g\|^2-A)/S$')
axs[0].set_title('solid: exact prediction; dashed: bound $d+8\\tau-3$', fontsize=8)
axs[1].set_title('state part, all $d$ collapse', fontsize=8)
for ax in axs: ax.grid(True, which='major')
axs[0].legend(frameon=False, loc='upper left'); axs[1].legend(frameon=False, loc='upper left')
S['E1.4']['all_z'] = allz
save(fig, 'e1_4_batch_variance')

# ---------------- E1.2 ---------------------------------------------------------------------------
if os.path.exists('data/e1_2.json'):
    D2 = json.load(open('data/e1_2.json'))
    meta = D2['meta']; seeds = [k for k in D2 if k != 'meta']; Ngrid = np.array(meta['Bgrid'])
    Nit = meta['N']; EPS = 0.03
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1.0, 0.8]))
    S['E1.2'] = dict(seeds=len(seeds))
    Teps = {}
    cols = {'0': INK2, '1': C[0], '10': C[1], '100': C[2]}
    for k in ('0', '1', '10', '100'):
        E = np.array([D2[s]['err'][k] for s in seeds])          # seeds x nB
        E = np.maximum(E, 1e-9)
        T = 2 * Nit * Ngrid[:E.shape[1]]
        lg = np.log10(E); mu = lg.mean(0); h = stats.t.ppf(0.975, len(seeds) - 1) * lg.std(0, ddof=1) / np.sqrt(len(seeds))
        axs[0].plot(T, 10 ** mu, 'o-', ms=3.2, color=cols[k], label=rf'$\Delta={k}\,\Delta_{{\rm ref}}$')
        axs[0].fill_between(T, 10 ** (mu - h), 10 ** (mu + h), color=cols[k], alpha=0.18, lw=0)
        # per-seed first crossing of EPS (log-log interpolation) -> T_eps
        te = []
        for row in E:
            idx = np.where(row <= EPS)[0]
            # first index after which the error stays below EPS
            ok = [j for j in idx if np.all(row[j:] <= EPS)]
            if not ok:
                te.append(np.nan); continue
            j = ok[0]
            if j == 0:
                te.append(T[0]); continue
            x0, x1 = np.log(T[j - 1]), np.log(T[j]); y0, y1 = np.log(row[j - 1]), np.log(row[j])
            te.append(float(np.exp(x0 + (np.log(EPS) - y0) * (x1 - x0) / (y1 - y0))))
        Teps[k] = np.array(te)
        S['E1.2'][k] = dict(final_err_median=float(np.median(E[:, -1])), final_err_max=float(E[:, -1].max()),
                            T_eps=list(map(float, te)), reached=int(np.sum(np.isfinite(te))))
    Ec = np.maximum(np.array([D2[s]['control']['10'] for s in seeds]), 1e-9)
    Tc = 2 * Nit * Ngrid[:Ec.shape[1]]
    lg = np.log10(Ec); axs[0].plot(Tc, 10 ** lg.mean(0), 's--', ms=3, color=INK, lw=1.0,
                                   label=r'control: frozen state, $10\Delta_{\rm ref}$')
    S['E1.2']['control'] = dict(err_geo_mean=(10 ** lg.mean(0)).tolist(),
                                slope_last4=loglog_slope(Tc[-4:], 10 ** lg.mean(0)[-4:]))
    axs[0].axhline(EPS, color=INK2, lw=0.8, ls=':', label=r'target $\varepsilon=0.03$')
    axs[0].set_xscale('log'); axs[0].set_yscale('log'); axs[0].set_ylim(1e-6, 2)
    axs[0].set_xlabel('queries $T=2NB$'); axs[0].set_ylabel(r'$f(\bar x)-f^*$ (geo. mean, 95% CI)')
    axs[0].legend(frameon=False, fontsize=6.5, loc='lower left'); axs[0].grid(True, which='major')
    # right: T_eps vs Delta
    ks = np.array([1, 10, 100]); med = np.array([np.nanmedian(Teps[str(k)]) for k in ks])
    lo = np.array([np.nanmin(Teps[str(k)]) for k in ks]); hi = np.array([np.nanmax(Teps[str(k)]) for k in ks])
    axs[1].errorbar(ks, med, yerr=[med - lo, hi - med], fmt='o', ms=4, color=C[1], capsize=2, label='$T_\\varepsilon$: median, min–max over seeds')
    # (unused) theory shape from Theorem 4.15: T ~ V(Delta) = 2d^2/(d-1) M^2 + (2d+5) d Delta^2/gamma^2 (tau=1), Delta = k M gamma/sqrt d
    d = meta['d']; kk = np.logspace(-0.3, 2.3, 200)
    Vk = 2 * d ** 2 / (d - 1) + (2 * d + 5) * kk ** 2
    # one multiplicative constant fitted on the Delta = 100 Delta_ref median (log scale)
    axs[1].plot(kk, med[-1] * (kk / 100) ** 2, '--', color=INK, lw=1.0, label=r'$\propto\Delta^2$ (noise-dominated law)')
    # slope over seeds via log-log regression on all finite (k, T_eps) pairs with k in {10,100}
    xs = np.concatenate([[k] * len(Teps[str(k)]) for k in (10, 100)]); ys = np.concatenate([Teps['10'], Teps['100']])
    fin = np.isfinite(ys); sl, h = loglog_slope(xs[fin], ys[fin])
    # paired per-seed slopes
    ps = np.log(Teps['100'] / Teps['10']) / np.log(10); m_, h_ = t_ci(ps[np.isfinite(ps)])
    S['E1.2']['slope_10_100'] = dict(ols=[sl, h], paired_mean=[m_, h_], per_seed=ps.tolist())
    axs[1].set_xscale('log'); axs[1].set_yscale('log'); axs[1].set_ylim(5e2, 3e9)
    axs[1].set_xlabel(r'$\Delta/\Delta_{\rm ref}$,  $\Delta_{\rm ref}=M\gamma/\sqrt{d}$'); axs[1].set_ylabel(r'queries to reach $\varepsilon=0.03$')
    axs[1].set_title(f'slope on $[10,100]\\Delta_{{\\rm ref}}$: {m_:.2f} $\\pm$ {h_:.2f}', fontsize=8)
    axs[1].legend(frameon=False, fontsize=6, loc='upper left'); axs[1].grid(True, which='major')
    # sensitivity: other targets
    sens = {}
    for eps2 in (0.02, 0.05, 0.1):
        tt = {}
        for k in ('10', '100'):
            E = np.maximum(np.array([D2[s]['err'][k] for s in seeds]), 1e-9); T = 2 * Nit * Ngrid[:E.shape[1]]
            te = []
            for row in E:
                ok = [j for j in range(len(row)) if np.all(row[j:] <= eps2)]
                if not ok: te.append(np.nan); continue
                j = ok[0]
                if j == 0: te.append(T[0]); continue
                x0, x1 = np.log(T[j - 1]), np.log(T[j]); y0, y1 = np.log(row[j - 1]), np.log(row[j])
                te.append(float(np.exp(x0 + (np.log(eps2) - y0) * (x1 - x0) / (y1 - y0))))
            tt[k] = np.array(te)
        ps2 = np.log(tt['100'] / tt['10']) / np.log(10); fin2 = np.isfinite(ps2)
        sens[str(eps2)] = dict(n=int(fin2.sum()), slope=t_ci(ps2[fin2]) if fin2.sum() > 1 else [float('nan')] * 2)
    S['E1.2']['sensitivity_eps'] = sens
    # E1.2b collapse panel
    Cc = json.load(open('data/e1_2_collapse.json'))
    e10 = np.array([Cc[s]['err10'] for s in seeds]); e100 = np.array([Cc[s]['err100'] for s in seeds])
    for j in range(e10.shape[1]):
        axs[2].plot(e10[:, j], e100[:, j], 'o', ms=3.5, color=[C[0], C[1], C[2], INK2][j],
                    label=f'$B={4**(j+6)}$ vs ${int(round(4**(j+6)/100))}$')
    lim = [3e-3, 1.5]; axs[2].plot(lim, lim, '-', color=INK, lw=0.9)
    axs[2].set_xscale('log'); axs[2].set_yscale('log'); axs[2].set_xlim(lim); axs[2].set_ylim(lim)
    axs[2].set_xlabel(r'error, $10\Delta_{\rm ref}$, batch $B/100$'); axs[2].set_ylabel(r'error, $100\Delta_{\rm ref}$, batch $B$')
    axs[2].set_title('collapse test (E1.2b)', fontsize=8); axs[2].legend(frameon=False, fontsize=5.5, loc='upper left')
    axs[2].grid(True, which='major')
    lr = np.log10(e100 / e10)
    S['E1.2']['collapse'] = dict(per_k=[t_ci(lr[:, j]) for j in range(lr.shape[1])], pooled=t_ci(lr.ravel()),
                                 note='log10(err100/err10) at matched B ratio 100; 0 means exact Delta^2 scaling')
    fig.subplots_adjust(wspace=0.42)
    save(fig, 'e1_2_logistic')

json.dump(S, open('data/summary_stats.json', 'w'), indent=1, default=float)
print(json.dumps(S, indent=1, default=float)[:6000])
