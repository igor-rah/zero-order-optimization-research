"""Decision-dependent drift: where does a non-tie step send the offset (same data block)?"""
import json, os, numpy as np, sys
HERE = os.path.dirname(os.path.abspath(__file__))
out = {}
for name in sys.argv[1:]:
    L = json.load(open(os.path.join(HERE, 'runs', f'{name}.json'))); S = {k: np.array(v) for k, v in L['step'].items()}
    lp, lm, op, om, u, bid = S['lp'], S['lm'], S['op'], S['om'], S['ulp'], S['bid']
    F = (lp + lm) / 2; w = F / u - np.round(F / u)             # offset in [-1/2,1/2)
    tie = (op == om) if L['args']['mode'] == 'rtn' else (np.round(lp / u) == np.round(lm / u))
    same = bid[1:] == bid[:-1]
    nt = (~tie[:-1]) & same; tt = tie[:-1] & same
    res = dict(n_nontie_within=int(nt.sum()),
               mean_abs_w_before=float(np.mean(np.abs(w[:-1][nt]))) if nt.any() else None,
               mean_abs_w_after=float(np.mean(np.abs(w[1:][nt]))) if nt.any() else None,
               p_tie_after_nontie=float(np.mean(tie[1:][nt])) if nt.any() else None,
               p_tie_after_tie=float(np.mean(tie[1:][tt])) if tt.any() else None,
               mean_abs_w_all=float(np.mean(np.abs(w))),
               frac_interior=float(np.mean(np.abs(w) < 0.25)))
    # offsets at the first step of each block vs at the last step of each block
    first = np.r_[True, bid[1:] != bid[:-1]]; last = np.r_[bid[1:] != bid[:-1], True]
    res['abs_w_block_first'] = float(np.mean(np.abs(w[first]))); res['abs_w_block_last'] = float(np.mean(np.abs(w[last])))
    out[name] = res; print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})
json.dump(out, open(os.path.join(HERE, 'drift.json'), 'w'), indent=1)
