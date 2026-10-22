import json, glob, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = os.path.dirname(os.path.abspath(__file__))
R, T = {}, {}
for f in glob.glob(f'{D}/runs/*.json'):
    if 'sweep' in f:
        continue
    j = json.load(open(f)); a = j['args']
    if a['tag']:
        T[a['tag']] = j
        continue
    R[(a['mode'], a['mu'], a['seed'])] = j

COL = {'fp32': '#2a78d6', 'rtn': '#eb6834', 'sr': '#1baf7a', 'rtn_adapt': '#eda100'}
LS = {'fp32': '-', 'rtn': '--', 'sr': '-.', 'rtn_adapt': ':'}
LAB = {'fp32': 'fp32 (reference)', 'rtn': 'bf16, round-to-nearest', 'sr': 'bf16, stochastic rounding',
       'rtn_adapt': 'bf16, RTN + adaptive μ'}
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.facecolor': SURF,
                     'figure.facecolor': SURF, 'axes.spines.top': False, 'axes.spines.right': False})


def curve(mode, mu=1e-3):
    runs = [R[k] for k in R if k[0] == mode and k[1] == mu]
    it = np.array([e[0] for e in runs[0]['eval']])
    ev = np.array([[e[1] for e in r['eval']] for r in runs])
    return it, ev, runs


fig, axs = plt.subplots(2, 2, figsize=(13, 10))
ax = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]

# A: eval loss
summary = {}
for md in ['fp32', 'rtn', 'sr', 'rtn_adapt']:
    it, ev, runs = curve(md)
    mean = ev.mean(0)
    ax[0].fill_between(it, ev.min(0), ev.max(0), color=COL[md], alpha=0.15, lw=0)
    ax[0].plot(it, mean, LS[md], color=COL[md], lw=2, label=LAB[md])
    summary[md] = (mean[0], mean[-1], ev[:, -1])
ax[0].set_title('A. Eval loss during MeZO fine-tuning (μ = 1e-3)', loc='left', color=INK, fontsize=11)
ax[0].set_xlabel('step'); ax[0].set_ylabel('cross-entropy, nats (fp32 eval)')
ax[0].grid(axis='y', color=GRID, lw=0.8); ax[0].legend(frameon=False, fontsize=9, loc='upper left')

# B: tie fraction
for md in ['rtn', 'sr', 'rtn_adapt']:
    _, _, runs = curve(md)
    its = [w['it'] for w in runs[0]['win']]
    tf = np.mean([[w['tie_frac'] for w in r['win']] for r in runs], 0)
    ax[1].plot(its, tf, LS[md], color=COL[md], lw=2, label=LAB[md])
ax[1].set_ylim(0, 1.02)
ax[1].set_title('B. Fraction of steps with Q(f(x+μz)) = Q(f(x−μz))\n(ties after rounding)', loc='left', color=INK, fontsize=11)
ax[1].set_xlabel('step'); ax[1].set_ylabel('tie fraction')
ax[1].grid(axis='y', color=GRID, lw=0.8); ax[1].legend(frameon=False, fontsize=9, loc='center left', bbox_to_anchor=(0.0, 0.41))

# C: improvement vs mu
mus = [1e-3, 1e-2, 1e-1]
w = 0.36
xs = np.arange(len(mus))
for k, md in enumerate(['fp32', 'rtn']):
    imp, span = [], []
    for mu in mus:
        r = R[(md, mu, 0)]
        imp.append(r['eval'][0][1] - r['eval'][-1][1])
        span.append(np.mean([q['mean_span_cells'] for q in r['win']]))
    b = ax[2].bar(xs + (k - 0.5) * w, imp, w * 0.92, color=COL[md], label=LAB[md])
    for x, v, s in zip(xs + (k - 0.5) * w, imp, span):
        ax[2].text(x, v + 0.002, f'{v:.3f}', ha='center', va='bottom', fontsize=8, color=INK2)
    if md == 'rtn':
        spans_rtn = span
ax[2].axhline(0, color=INK2, lw=0.8)
ax[2].set_xticks(xs, [f'μ = {m:g}\n(≈{sp:.2f} cells)' for m, sp in zip(mus, spans_rtn)])
ax[2].set_title('C. Eval-loss drop over 2000 steps\nvs radius μ (seed 0)', loc='left', color=INK, fontsize=11)
ax[2].set_ylabel('loss(0) − loss(2000), nats')
ax[2].grid(axis='y', color=GRID, lw=0.8); ax[2].legend(frameon=False, fontsize=9)

# D: lr sensitivity (seed 0, mu=1e-3): does a smaller step rescue rounding?
lrs = [5e-4, 5e-5, 5e-6]
def get(md, lr):
    if lr == 5e-4:
        return R.get((md, 1e-3, 0))
    return T.get(f'{md}_lr{lr:.0e}'.replace('e-0', 'e-'))
xs = np.arange(len(lrs)); w = 0.27
for k, md in enumerate(['fp32', 'rtn', 'sr']):
    vals = []
    for lr in lrs:
        r = get(md, lr)
        vals.append(np.nan if r is None else r['eval'][0][1] - r['eval'][-1][1])
    ax[3].bar(xs + (k - 1) * w, np.nan_to_num(vals), w * 0.92, color=COL[md], label=LAB[md])
    for x, v in zip(xs + (k - 1) * w, vals):
        ax[3].text(x, (0 if np.isnan(v) else v) + (0.004 if (np.isnan(v) or v >= 0) else -0.004),
                   '—' if np.isnan(v) else f'{v:.3f}', ha='center',
                   va='bottom' if (np.isnan(v) or v >= 0) else 'top', fontsize=8, color=INK2)
ax[3].axhline(0, color=INK2, lw=0.8)
ax[3].set_xticks(xs, [f'lr = {l:g}' for l in lrs])
ax[3].set_ylim(-0.12, None)
ax[3].set_title('D. Eval-loss drop vs learning rate (μ = 1e-3, seed 0)\nnegative = got worse (clipped)', loc='left', color=INK, fontsize=11)
ax[3].set_ylabel('loss(0) − loss(2000), nats')
ax[3].grid(axis='y', color=GRID, lw=0.8); ax[3].legend(frameon=False, fontsize=9)

print('lr table (drop):')
for md in ['fp32', 'rtn', 'sr']:
    print(md, [None if get(md, lr) is None else round(get(md, lr)['eval'][0][1] - get(md, lr)['eval'][-1][1], 4) for lr in lrs],
          'tie last', [None if get(md, lr) is None else round(get(md, lr)['win'][-1]['tie_frac'], 2) for lr in lrs])

for a_ in ax: a_.set_axisbelow(True)
fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.08, hspace=0.42, wspace=0.22)
fig.text(0.07, 0.01, '“cells”: mean number of bf16 grid cells spanned by f(x+μz) − f(x−μz);  “—”: run diverged', fontsize=8.5, color=INK2)
fig.savefig(f'{D}/zo_bf16_rounding.png', dpi=160)

print('mode       start   end(mean)  end per seed  drop')
for md, (s, e, per) in summary.items():
    print(f'{md:10s} {s:.4f}  {e:.4f}   {np.round(per,4)}  {s-e:.4f}')
for md in ['rtn', 'sr', 'rtn_adapt']:
    _, _, runs = curve(md)
    print(md, 'tie frac first/last window:', round(np.mean([r['win'][0]['tie_frac'] for r in runs]), 2),
          round(np.mean([r['win'][-1]['tie_frac'] for r in runs]), 2),
          'mean mu', round(float(np.mean([q['mean_mu'] for r in runs for q in r['win']])), 5))
for md in ['fp32', 'rtn']:
    for mu in mus:
        r = R[(md, mu, 0)]
        print(md, mu, 'drop', round(r['eval'][0][1] - r['eval'][-1][1], 4),
              'span', round(np.mean([q['mean_span_cells'] for q in r['win']]), 3))
