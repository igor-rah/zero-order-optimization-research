import json, glob, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'runs')
def load(n):
    p = f'{R}/{n}.json'
    return json.load(open(p)) if os.path.exists(p) else None
def chain_stats(L):
    S = {k: np.array(v) for k, v in L['step'].items()}
    lp, lm, op, om, u = S['lp'], S['lm'], S['op'], S['om'], S['ulp']
    df, dq = lp - lm, op - om
    c = (((lp + lm) / 2) / u) % 1.0
    out = {}
    out['tie'] = float(np.mean(dq == 0)) if L['args']['mode'] == 'rtn' else float(np.mean(np.round(lp/u) == np.round(lm/u)))
    out['tie_pred_uniform'] = float(np.mean(np.clip(1 - np.abs(df) / u, 0, None)))
    out['span'] = float(np.mean(np.abs(df) / u))
    if L['args']['mode'] == 'rtn':
        out['kappa_hat'] = float(np.sum(dq * df) / np.sum(df * df))
        W = 100
        kw = [np.sum(dq[i:i+W]*df[i:i+W]) / np.sum(df[i:i+W]**2) for i in range(0, len(df), W)]
        tw = [np.mean(dq[i:i+W] == 0) for i in range(0, len(df), W)]
        out['kappa_win_sd'] = float(np.std(kw)); out['tie_win_sd'] = float(np.std(tw))
        out['kappa_win'] = [float(v) for v in kw]; out['tie_win'] = [float(v) for v in tw]
    out['rho'] = {l: float(np.abs(np.mean(np.exp(2j*np.pi*(c[l:]-c[:-l]))))) for l in (1, 4, 16, 64, 256)}
    fc = (lp + lm) / 2
    bid = S['bid']
    first = np.r_[True, bid[1:] != bid[:-1]]
    out['sigmaF_over_u'] = float(np.std(fc[first]) / np.median(u)) if first.sum() > 3 else None
    within = np.diff(fc)[(bid[1:] == bid[:-1])]
    out['within_block_step_over_u'] = float(np.median(np.abs(within)) / np.median(u)) if len(within) else None
    out['u_median'] = float(np.median(u))
    return out
res = {}
for f in sorted(glob.glob(f'{R}/*.json')):
    n = os.path.basename(f)[:-5]
    if n.startswith('sgdsweep'): continue
    L = json.load(open(f)); ev = L['eval']
    r = dict(args=L['args'], eval0=ev[0][1], final=ev[-1][1], drop=ev[0][1]-ev[-1][1], curve=ev)
    if L['args']['mode'] != 'sgd':
        r.update(chain_stats(L))
    res[n] = r
    print(f"{n:32s} final {r['final']:.4f} drop {r['drop']:+.4f}",
          '' if L['args']['mode']=='sgd' else f"tie {r['tie']:.3f} pred {r['tie_pred_uniform']:.3f} span {r['span']:.3f} kappa {r.get('kappa_hat',float('nan')):.3f} "
          f"rho1 {r['rho'][1]:.3f} rho16 {r['rho'][16]:.3f} rho64 {r['rho'][64]:.3f} sF/u {r['sigmaF_over_u']} within {r['within_block_step_over_u']}")
json.dump(res, open(os.path.join(HERE, 'summary.json'), 'w'), indent=1)
