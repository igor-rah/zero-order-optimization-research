"""Compare logged tie fractions of the TinyLM MeZO runs with Corollary (tau as function of mean span)."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import json, glob, os, numpy as np
from scipy.stats import norm
from scipy import integrate
def Bsp(dl):
    dl = np.abs(dl)
    return np.where(dl <= 1, 2/3 - dl**2 + dl**3/2, np.where(dl <= 2, (2 - dl)**3/6, 0.0))
tau_rtn = lambda s: 2*norm.cdf(1/s) - 1 - 2*s*(norm.pdf(0) - norm.pdf(1/s))
tau_sr = lambda s: integrate.quad(lambda w: 2*norm.pdf(w)*Bsp(s*w), 0, 3/s, limit=200)[0]
rows = []
for f in sorted(glob.glob(os.path.join(HERE, 'runs', '*.json'))):
    if 'sweep' in f: continue
    j = json.load(open(f)); a = j['args']
    if a['mode'] == 'fp32': continue
    name = os.path.basename(f)[:-5]
    W = j['win']
    m = np.array([w['mean_span_cells'] for w in W]); t = np.array([w['tie_frac'] for w in W])
    s = m*np.sqrt(np.pi/2)
    pred = np.array([tau_rtn(x) if a['mode'] in ('rtn','rtn_adapt') else tau_sr(x) for x in s])
    rows.append(dict(run=name, mode=a['mode'], mu=a['mu'], lr=a['lr'], seed=a['seed'],
                     span=float(m.mean()), tie=float(t.mean()), pred=float(pred.mean()),
                     mae=float(np.mean(np.abs(t-pred))), n=len(W), m_all=m.tolist(), t_all=t.tolist()))
for r in rows:
    print(f"{r['run']:32s} span={r['span']:.3f} tie={r['tie']:.3f} pred={r['pred']:.3f} |err|win={r['mae']:.3f}")
allerr = np.concatenate([np.abs(np.array(r['t_all'])-np.array([tau_rtn(x*np.sqrt(np.pi/2)) if r['mode']!='sr' else tau_sr(x*np.sqrt(np.pi/2)) for x in r['m_all']])) for r in rows])
print('overall window-level MAE', allerr.mean(), 'max', allerr.max(), 'windows', len(allerr))
json.dump(rows, open(os.path.join(HERE, 'ties_vs_logs.json'), 'w'))
