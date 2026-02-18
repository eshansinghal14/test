import numpy as np,\
        scipy as sp,\
        pandas as pd,\
        torch as th,\
        torchvision as thv,\
        torch.nn as nn,\
        torch.nn.functional as F,\
        os,random,inspect,glob,math,pickle,time,fire
from pprint import pprint
from functools import partial
from itertools import chain
import pdb

import nltk
import sentencepiece as sp

import seaborn as sns
import matplotlib.pyplot as plt
import argparse


dev = (
    'cuda' if th.cuda.is_available() else
    'mps' if hasattr(th.backends, 'mps') and th.backends.mps.is_available() else
    'cpu'
)
maskid, padid = None, None


class WeezlTokenizerTSV:
    def __init__(self, path):
        self.path = path
        self.pattern_to_code, self.code_to_pattern = self._load_dictionary(path)
        self._base_vocab = max(self.code_to_pattern.keys(), default=0) + 1
        self.pad_id = self._base_vocab
        self.unk_id = self._base_vocab + 1
        self.mask_id = self._base_vocab + 2

    def vocab_size(self):
        return self._base_vocab + 3

    def encode(self, text):
        chars = list(text)
        tokens = []
        i = 0
        while i < len(chars):
            current = ''
            best_code = None
            best_len = 0
            j = i
            while j < len(chars):
                current += chars[j]
                code = self.pattern_to_code.get(current)
                if code is not None:
                    best_code = code
                    best_len = j - i + 1
                    j += 1
                else:
                    break

            if best_code is not None:
                tokens.append(best_code)
                i += best_len
            else:
                tokens.append(self.unk_id)
                i += 1
        return tokens

    def decode(self, ids):
        out = []
        for tid in ids:
            if tid == self.pad_id:
                continue
            if tid == self.unk_id:
                out.append('�')
                continue
            if tid == self.mask_id:
                out.append('[MASK]')
                continue
            pat = self.code_to_pattern.get(int(tid))
            if pat is not None:
                out.append(pat)
            else:
                out.append('�')
        return ''.join(out)

    @staticmethod
    def _unescape(s):
        return (s.replace('\\\\', '\x00').replace('\\t', '\t')
                .replace('\\n', '\n').replace('\\r', '\r')
                .replace('\\0', '\0').replace('\x00', '\\'))

    @classmethod
    def _load_dictionary(cls, path):
        pattern_to_code = {}
        code_to_pattern = {0: ''}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t', 3)
                if len(parts) < 4:
                    continue
                code = int(parts[0])
                pattern = cls._unescape(parts[3])
                pattern_to_code[pattern] = code
                code_to_pattern[code] = pattern
        return pattern_to_code, code_to_pattern

def setup(s):
    th.manual_seed(s)
    random.seed(s)
    np.random.seed(s)
    nltk.download('punkt_tab')

class kv_cache_t:
    def __init__(s, bsz, nblock, nhead, T, hdim):
        super().__init__()
        s.shape = (nblock, 2, bsz, nhead, 2*T, hdim//nhead)
        s.kv = None
        s.p = 0

    def reset(s):
        s.p = 0

    def insert(s, l, k, v):

        if s.kv is None:
            s.kv = th.empty(s.shape, dtype=k.dtype, device=dev)
        b, _, T, _ = k.size()
        t0, t1 = s.p, s.p + T

        s.kv[l, 0, :, :, t0:t1, :] = k
        s.kv[l, 1, :, :, t0:t1, :] = v
        k = s.kv[l, 0, :, :, :t1, :]
        v = s.kv[l, 1, :, :, :t1, :]
        if l == s.kv.size(0)-1:
            s.p = t1
        return k, v

class attn_t(nn.Module):
    def __init__(s, hdim, T, nhead, drop):
        super().__init__()
        s.hdim, s.T, s.nhead = hdim, T, nhead
        s.qkv = nn.Linear(hdim, 3*hdim, bias=False)
        s.op = nn.Linear(hdim, hdim, bias=False)
        s.dropout = nn.Dropout(drop)

    def forward(s, x, φ, idx, cache=None):
        b, T, hdim = x.size()
        x = x + φ
        q, k, v = s.qkv(x).split(s.hdim, dim=2)

        q, k = F.rms_norm(q, (hdim, )), F.rms_norm(k, (hdim, ))
        v = F.rms_norm(v, (hdim, ))
        q = q.view(b, T, s.nhead, hdim//s.nhead).transpose(1,2)
        k = k.view(b, T, s.nhead, hdim//s.nhead).transpose(1,2)
        v = v.view(b, T, s.nhead, hdim//s.nhead).transpose(1,2)

        if cache is not None:
            k, v = cache.insert(idx, k, v)

        nq, nk = q.size(2), k.size(2)
        func = partial(F.scaled_dot_product_attention,
                       query=q, key=k, value=v)
        if cache is None or nq == nk:
            y = func(is_causal=True)
        elif nq == 1:
            y = func(is_causal=False)
        else:
            mask = th.zeros((nq, nk), dtype=th.bool, device=dev)
            T0 = nk - nq
            mask[:, :T0] = True
            mask[:, T0:] = th.tril(th.ones((nq,nq), dtype=th.bool, device=dev))
            y = func(attn_mask=mask)

        y = y.transpose(1,2).contiguous().view(b, T, hdim)
        y = s.op(y)
        return y

class block_t(nn.Module):
    def __init__(s, hdim, T, nhead, drop, idx):
        super().__init__()
        s.idx = idx
        s.ln = partial(F.rms_norm, normalized_shape=(hdim,))
        s.attn = attn_t(hdim, T, nhead, drop)
        s.m = nn.ModuleDict(dict(
            fc = nn.Linear(hdim, 4*hdim, bias=False),
            proj = nn.Linear(4*hdim, hdim, bias=False),
            act = nn.GELU(),
            drop = nn.Dropout(drop),
            ))
        s.mlp = lambda x: s.m.drop(s.m.proj(s.m.act(s.m.fc(x))))

    def forward(s, x, φ, cache):
        x = x + s.ln(s.attn(s.ln(x), φ, s.idx, cache))
        x = x + s.ln(s.mlp(s.ln(x)))
        return x

class gpt_t(nn.Module):
    def __init__(s, nblock=4, vdim=1024, hdim=32,
                 T=128, nhead=2, drop=0.25):
        super().__init__()
        s.nblock, s.vdim, s.hdim, s.T, s.nhead = nblock, vdim, hdim, T, nhead
        s.idx = th.arange(0, 2*T, dtype=th.long).unsqueeze(0).to(dev)

        s.register_buffer('mask',th.tril(th.ones(T, T)).view(1,1,T,T))

        s.t = nn.ModuleDict(dict(
                wte = nn.Embedding(vdim, hdim),
                wpe = nn.Embedding(T, hdim),
                drop = nn.Dropout(drop),
                m = nn.ModuleList([block_t(hdim, T, nhead, drop, idx) \
                        for idx in range(nblock)])
            ))
        s.ln = partial(F.rms_norm, normalized_shape=(hdim,))
        s.op = nn.Linear(hdim, vdim, bias=False)
        nn.init.normal_(s.t.wte.weight, mean=0, std=hdim**-0.5)
        s.op.weight = s.t.wte.weight

        np = sum(p.numel() for p in s.t.parameters())
        print(f'# params: {np/1e6:.2f} M')

    def forward(s, x, y, cache=None):
        b, T = x.size()

        T0 = 0 if cache is None else cache.p
        h = s.t.wte(x)
        φ = s.t.wpe(s.idx[:,T0:T0+T])
        h = s.t.drop(h+φ)
        for blk in s.t.m:
            h = blk(h, φ, cache)
        h = s.ln(h)
        yh = s.op(h)

        ell = None
        if y is not None:
            ignore = -100
            if padid is not None:
                ignore = int(padid)
            ell = F.cross_entropy(yh.view(-1, yh.size(-1)),
                                  y.view(-1),
                                  ignore_index=ignore,
                                  label_smoothing=0.0)

        return yh, ell

    @th.inference_mode()
    def generate(s, x, T, temp=1, topk=None, cache=None):
        t0 = x.size(1)
        xp, r = x.clone(), x.clone()
        for _ in range(t0, T):
            yh, ell = s.forward(xp, None, cache=cache)
            yh = yh[:, -1, :]
            if topk is not None:
                v, _ = th.topk(yh, min(topk, yh.size(-1)))
                yh[yh < v[:, [-1]]] = -float('inf')
            if temp > 0:
                yh = yh/temp
                p = F.softmax(yh, -1)
                xp = th.multinomial(p, 1)
            else:
                xp = th.argmax(yh, -1, keepdim=True)
            r = th.cat((r, xp), dim=1)
        return r


def _nll_and_bytes(m, x, y, tok):
    yh, _ = m(x, y)
    v = yh.size(-1)
    nll = F.cross_entropy(
        yh.view(-1, v),
        y.view(-1),
        reduction='none',
        ignore_index=int(padid) if padid is not None else -100,
    ).view_as(y)

    if padid is None:
        mask = th.ones_like(y, dtype=th.bool)
    else:
        mask = y.ne(int(padid))

    nll_sum = nll.masked_select(mask).sum().detach().item()

    byte_sum = 0
    y_cpu = y.detach().to('cpu')
    for i in range(y_cpu.size(0)):
        ids = y_cpu[i].tolist()
        if padid is not None:
            ids = [t for t in ids if t != int(padid)]
        txt = tok.decode(ids)
        byte_sum += len(txt.encode('utf-8', errors='replace'))

    token_count = int(mask.sum().detach().item())
    return nll_sum, token_count, byte_sum

def get_data(
    f='/tmp/10.txt.utf-8',
    m=0,
    reset=False,
    tokenizer='sp',
    weezl_dict_path=None,
    vocab_size=1024,
):
    global maskid, padid
 
    if tokenizer == 'weezl':
        if weezl_dict_path is None:
            raise ValueError('weezl_dict_path is required when tokenizer="weezl"')
        tok = WeezlTokenizerTSV(weezl_dict_path)
        padid, maskid = tok.pad_id, tok.mask_id
 
        cache_path = f"d_weezl_{os.path.basename(weezl_dict_path)}.p"
        if not os.path.exists(cache_path) or reset:
            d = {}
            d['vdim'] = tok.vocab_size()
            d['x'] = []
            x = open(f, 'r', encoding='utf-8', errors='replace').read()
            x = nltk.sent_tokenize(x)
            for _ in range(max(1, m)):
                for sent in x:
                    d['x'] += tok.encode(sent)
            pickle.dump(d, open(cache_path, 'wb'))
        else:
            d = pickle.load(open(cache_path, 'rb'))
        return d
 
    if not os.path.exists('tok.model') or reset:
        sp.SentencePieceTrainer.train(input=f,
                                      model_prefix='tok',
                                      model_type='bpe',
                                      vocab_size=vocab_size,
                                      normalization_rule_name='nmt_nfkc_cf')
                                      #user_defined_symbols=['[mask]','[pad]'])
 
    s = sp.SentencePieceProcessor(model_file='tok.model')
    padid = 0
    maskid = None
    if not os.path.exists('d.p') or reset:
        d = {}
        d['vdim'] = s.vocab_size()
        d['x'] = []
        x = open(f, 'r', encoding='utf-8', errors='replace').read()
        x = nltk.sent_tokenize(x)
        for _ in range(max(1,m)):
            t = s.encode(x,
                         enable_sampling=True if m > 0 else False,
                         alpha=0.2)
            d['x'] += list(chain(*t))
        pickle.dump(d, open('d.p', 'wb'))
    else:
        d = pickle.load(open('d.p', 'rb'))
    return d

def get_batch(ds, b, T, overfit=False):
    N = len(ds['x'])
    ns = th.randint(N, (b,))
    if overfit:
        ns = th.arange(b)

    x = (padid + th.zeros(b, T)).long()
    y = (padid + th.zeros(b, T)).long()
    for i,n in enumerate(ns):
        s = n
        e = min(s+T+1, N)

        a = th.tensor(ds['x'][s:e]).long()
        x[i,:len(a)-1] = a[:-1]
        y[i,:len(a)-1] = a[1:]
    return x.to(dev), y.to(dev)

def main(seed=42,
         bsz=32,
         E=int(1e4),
         lr=0.025,
         wd=0.001,
         drop=0.001,
         T=64,
         hdim=128,
         nhead=2,
         nblock=2,
         generate=False,
         reset=False,
         data_file='jungle_book.txt',
         tokenizer='sp',
         weezl_dict_path=None,
         vocab_size=1024):

    ds = get_data(data_file, reset=reset, tokenizer=tokenizer, weezl_dict_path=weezl_dict_path, vocab_size=vocab_size)
    tok = None
    if tokenizer == 'sp':
        tok = sp.SentencePieceProcessor(model_file='tok.model')
    elif tokenizer == 'weezl':
        tok = WeezlTokenizerTSV(weezl_dict_path)
    vdim = ds['vdim']
    m = gpt_t(nblock=nblock,
              vdim=vdim, hdim=hdim, T=T,
              nhead=nhead, drop=drop)

    m.to(dev)
    opt = th.optim.Muon(m.parameters(), lr=lr,
                         weight_decay=wd)
    sched = th.optim.lr_scheduler.CosineAnnealingLR(opt, E)

    if generate:
        print('[Loading]...')
        t = th.load('m.p', map_location=dev)

        cache = kv_cache_t(bsz, nblock, nhead, T, hdim)
        m.load_state_dict(t)

        m.eval()
        x, _ = get_batch(ds, bsz, T)
        xp = m.generate(x[:,:T//8], T, temp=0, cache=cache)
        for i in range(bsz):
            if tokenizer == 'sp':
                print('x: ', tok.decode(x[i].tolist()))
                print('xp: ', tok.decode(xp[i].tolist()))
            else:
                print('x: ', tok.decode(x[i].tolist()))
                print('xp: ', tok.decode(xp[i].tolist()))
            print()
        pickle.dump(cache, open('cache.p', 'wb'))

    if not generate:
        ℓ=10;dt=0
        bpb_ema = None
        cache = kv_cache_t(bsz, nblock, nhead, T, hdim)
        for t in range(E):
            m.train()
            ts = time.time()
            x, y = get_batch(ds, bsz, T, False)

            yh, ell = m(x, y)
            m.zero_grad(set_to_none=True)
            ell.backward()
            opt.step()
            sched.step()

            ℓ = (1-0.1)*ℓ + 0.1*ell.detach().item()

            nll_sum, _, byte_sum = _nll_and_bytes(m, x, y, tok)
            if byte_sum > 0:
                bpb = (nll_sum / byte_sum) / math.log(2)
                if bpb_ema is None:
                    bpb_ema = bpb
                else:
                    bpb_ema = (1-0.1)*bpb_ema + 0.1*bpb

            dt = (1-0.1)*dt + 0.1*(time.time()-ts)
            if t%100 == 0 and t > 0:
                bpb_str = 'n/a' if bpb_ema is None else f'{bpb_ema:.3f}'
                print(f'[{t:04d}/{E:04d}] ℓ: {ℓ:.3f} p: {np.exp(ℓ):.1f} bpb: {bpb_str} dt: {dt*1000:3.1f} ms')
                th.save(m.state_dict(), 'm.p')

                if False:
                    m.eval()
                    cache.reset()
                    xp = m.generate(x[:,:T//2], T, temp=0, cache=cache)
                    #print('x: ', x[0].tolist())
                    #print('xp: ', xp[0].tolist())
                    if tokenizer == 'sp':
                        print('x: ', tok.decode(x[0].tolist()))
                        print('xp: ', tok.decode(xp[0].tolist()))
                    else:
                        print('x: ', tok.decode(x[0].tolist()))
                        print('xp: ', tok.decode(xp[0].tolist()))


if __name__=='__main__':
    setup(44)
    fire.Fire(main)
