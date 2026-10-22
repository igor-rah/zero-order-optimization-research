"""Shared pieces: data, tiny char-level transformer, bf16 rounding helpers."""
import glob, os, math, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(int(os.environ.get("NT", 2)))
CTX = 64
OUT = os.path.dirname(os.path.abspath(__file__))


def _read(files, limit):
    buf, n = [], 0
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
                t = fh.read()
        except (IsADirectoryError, PermissionError):
            continue
        buf.append(t); n += len(t)
        if n > limit:
            break
    return ''.join(buf)


def load_corpora():
    rnd = random.Random(0)
    code_files = sorted(glob.glob('/usr/lib/python3*/**/*.py', recursive=True))
    rnd.shuffle(code_files)
    code = _read(code_files, 4_000_000)
    prose_files = sorted(glob.glob('/usr/share/common-licenses/*') +
                         glob.glob('/usr/lib/python3*/**/LICENSE*', recursive=True) +
                         glob.glob('/usr/lib/python3*/**/*.txt', recursive=True))
    prose = _read(prose_files, 10_000_000)
    keep = set(chr(i) for i in range(32, 127)) | {'\n'}
    code = ''.join(c for c in code if c in keep)
    prose = ''.join(c for c in prose if c in keep)
    vocab = sorted(keep)
    stoi = {c: i for i, c in enumerate(vocab)}
    enc = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long)
    code_t = enc(code)
    prose_t = enc(prose)
    n = len(prose_t)
    return code_t, prose_t[: int(0.8 * n)], prose_t[int(0.8 * n):], len(vocab)


def get_batch(data, bs, gen):
    ix = torch.randint(len(data) - CTX - 1, (bs,), generator=gen)
    x = torch.stack([data[i:i + CTX] for i in ix])
    y = torch.stack([data[i + 1:i + CTX + 1] for i in ix])
    return x, y


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.register_buffer('mask', torch.triu(torch.ones(CTX, CTX, dtype=torch.bool), 1))

    def forward(self, x):
        h = self.ln1(x)
        T = x.shape[1]
        x = x + self.attn(h, h, h, attn_mask=self.mask[:T, :T], need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, V, d=128, L=4, h=4):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.zeros(1, CTX, d))
        self.blocks = nn.Sequential(*[Block(d, h) for _ in range(L)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)

    def forward(self, x, y):
        z = self.tok(x) + self.pos[:, : x.shape[1]]
        z = self.head(self.ln(self.blocks(z)))
        return F.cross_entropy(z.reshape(-1, z.shape[-1]), y.reshape(-1))


# ---------- bf16 rounding of a scalar loss value ----------
def bf16_rtn(v: float) -> float:
    """Round-to-nearest(-even) to bfloat16, back to float."""
    return torch.tensor(v, dtype=torch.float32).to(torch.bfloat16).float().item()


def bf16_neighbors(v: float):
    """Two consecutive bf16 numbers lo <= v <= hi (positive v)."""
    r = bf16_rtn(v)
    ulp = bf16_ulp(r)
    if r <= v:
        return r, r + ulp
    return r - bf16_ulp(r - ulp / 2), r


def bf16_ulp(v: float) -> float:
    m, e = math.frexp(abs(v))  # v = m*2^e, m in [0.5,1)
    return 2.0 ** (e - 1 - 7)   # 8 significant bits (7 stored + implicit)


def bf16_sr(v: float, gen: np.random.Generator) -> float:
    lo, hi = bf16_neighbors(v)
    if hi == lo:
        return lo
    p = (v - lo) / (hi - lo)
    return hi if gen.random() < p else lo
