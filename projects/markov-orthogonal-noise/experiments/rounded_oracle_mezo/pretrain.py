"""Stage 1: first-order pretraining of TinyLM on Python code (stand-in for a pretrained LLM)."""
import time, torch, numpy as np
from common import *

code, ft_tr, ft_ev, V = load_corpora()
print(f'code {len(code)/1e6:.2f}M chars, ft-train {len(ft_tr)/1e6:.2f}M, ft-eval {len(ft_ev)/1e6:.2f}M, V={V}')
torch.manual_seed(0)
m = TinyLM(V)
print('params', sum(p.numel() for p in m.parameters()))
opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
g = torch.Generator().manual_seed(0)
t0 = time.time()
for it in range(1, 1501):
    x, y = get_batch(code, 32, g)
    loss = m(x, y)
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 250 == 0:
        with torch.no_grad():
            ge = torch.Generator().manual_seed(123)
            ev = np.mean([m(*get_batch(ft_ev, 64, ge)).item() for _ in range(8)])
        print(f'it {it} code-loss {loss.item():.3f} prose-eval {ev:.3f} ({time.time()-t0:.0f}s)', flush=True)
torch.save(m.state_dict(), f'{OUT}/pretrained.pt')
