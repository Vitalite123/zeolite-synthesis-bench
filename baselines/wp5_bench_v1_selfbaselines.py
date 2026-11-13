# -*- coding: utf-8 -*-
"""
ZeoliteSynthesisBench v1 — Track A 自实现基线生成（KDE-QT / TabDDPM-MLP / Random-Uniform）
================================================================================
预注册: Documentation/ZeoliteSynthesisBench_v1_规格_骨架.md (v1.0, FROZEN 2026-10-23)
协议: 123 合格体系 (test n_true>=5, zeo!=Dense/Amorphous) × 200 样本/体系, seed 42
产出 (幂等覆盖写, gitignored):
  Data/wp5_bench_v1_kde_syn.parquet
  Data/wp5_bench_v1_tabddpm_syn.parquet
  Data/wp5_bench_v1_random_syn.parquet
  Data/wp5_bench_v1_selfbaselines_manifest.json
  Data/wp5_bench_v1_selfbaselines_report.txt
用法: python wp5_bench_v1_selfbaselines.py [--skip-tabddpm]
"""
import os, sys, json, time, math, hashlib, argparse, collections, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZG = os.path.join(BASE, 'vendor', 'zeosyn_gen')
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'metrics'))
sys.path.insert(0, os.path.join(BASE, 'baselines')); sys.path.insert(0, ZG)

import wp5_metrics_api_v0 as api            # 注入官方 utils（pkl 反序列化所需）；会 chdir(ZG)
M = api.ZeoSynMetrics()
import wp3_ddpm_v0 as w3

DS_PT = os.path.join(BASE, 'weights', 'wp3_ddpm_v0_dataset.pt')
KDE_PQ = os.path.join(BASE, 'data', 'samples', 'wp5_bench_v1_kde_syn.parquet')
TAB_PQ = os.path.join(BASE, 'data', 'samples', 'wp5_bench_v1_tabddpm_syn.parquet')
RND_PQ = os.path.join(BASE, 'data', 'samples', 'wp5_bench_v1_random_syn.parquet')
TAB_PT = os.path.join(BASE, 'weights', 'wp5_bench_v1_tabddpm_model.pt')
MANIFEST = os.path.join(BASE, 'data', 'wp5_bench_v1_selfbaselines_manifest.json')
REPORT = os.path.join(BASE, 'data', 'wp5_bench_v1_selfbaselines_report.txt')

NS = 200
SEED = 42
KDE_MIN_TRAIN = 30          # 骨架 train 行 < 30 → 回退全 train KDE（预注册）
TAB_STEPS = 1000
TAB_EPOCHS = 500
TAB_BATCH = 1024
TAB_HID = 1024
TAB_LR = 5e-4
TAB_EMA = 0.999

L = []
def log(s=''):
    L.append(s); print(s, flush=True)

def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

def load_payload():
    p = torch.load(DS_PT, weights_only=False)
    X = np.asarray(p['syn_raw'], dtype=np.float64)
    Z = np.asarray(p['zeo_key']); O = np.asarray(p['osda_key'])
    tr = p['train_idx'].numpy(); te = p['test_idx'].numpy()
    c = collections.Counter(zip(Z[te], O[te]))
    elig = sorted([k for k in c if c[k] >= 5 and k[0] != 'Dense/Amorphous'])
    log('载荷: %d 行 | train %d / test %d | 合格体系 %d' % (len(X), len(tr), len(te), len(elig)))
    return p, X, Z, O, tr, te, elig

def cond_matrix(p, Z, O, elig, rows0=None):
    """每个合格体系取首个 test 行条件向量（与 G3 一致）。"""
    te = p['test_idx'].numpy()
    zf = p['zeo_feat']; of = p['osda_feat']
    if rows0 is None:
        rows0 = []
        for z_, o_ in elig:
            m = np.where((Z == z_) & (O == o_))[0]
            m = m[np.isin(m, te)]
            rows0.append(int(m[0]))
    return zf[rows0].float(), of[rows0].float()

def save_pred(Xraw, elig, path, method):
    df = pd.DataFrame(Xraw, columns=M.cols)
    df['zeo'] = np.repeat([k[0] for k in elig], NS)
    df['osda'] = np.repeat([k[1] for k in elig], NS)
    df.to_parquet(path, index=False)
    log('%s: %d 行 → %s' % (method, len(df), os.path.basename(path)))

# =====================================================================
# KDE-QT: 骨架条件 KDE（QuantileTransformer 空间；n<30 回退全 train）
# =====================================================================
def run_kde(X, Z, O, tr, te, elig):
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.neighbors import KernelDensity
    t0 = time.time()
    qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal',
                             subsample=len(tr), random_state=SEED)
    Xtr_q = qt.fit_transform(X[tr])
    global_kde = KernelDensity(bandwidth='scott', kernel='gaussian').fit(Xtr_q)
    log('KDE: 全局拟合 %d 行；骨架条件拟合开始' % len(tr))

    by_sys = collections.defaultdict(list)
    for pos, gi in enumerate(tr):
        by_sys[Z[gi]].append(pos)

    rng = np.random.RandomState(SEED)
    out, n_global, n_local = [], 0, 0
    for z_, o_ in elig:
        idx = by_sys.get(z_, [])
        if len(idx) < KDE_MIN_TRAIN:
            kde = global_kde; n_global += 1
        else:
            kde = KernelDensity(bandwidth='scott', kernel='gaussian').fit(Xtr_q[idx])
            n_local += 1
        s = kde.sample(NS, random_state=rng.randint(0, 2**31 - 1))
        out.append(qt.inverse_transform(s))
    Xgen = np.vstack(out)
    save_pred(Xgen, elig, KDE_PQ, 'KDE-QT')
    log('KDE: 骨架条件 %d 体系 / 全局回退 %d 体系；用时 %.1f min'
        % (n_local, n_global, (time.time() - t0) / 60))
    return {'n_local': n_local, 'n_global': n_global, 'min_train': KDE_MIN_TRAIN}

# =====================================================================
# TabDDPM-MLP: 12 列全连续（payload syn_qt）+ 条件 MLP 去噪
# =====================================================================
class Denoiser(nn.Module):
    def __init__(self, d_x, d_c, h=TAB_HID):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_x + d_c + 1, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, d_x))
    def forward(self, x, c, t):
        return self.net(torch.cat([x, c, t], dim=1))

def cosine_alpha_bar(T):
    s = 0.008
    t = np.linspace(0, 1, T + 1)
    f = np.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return (f / f[0]).astype(np.float32)

def run_tabddpm(p, X, Z, O, tr, te, elig, device, reuse=False):
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)
    Xq = np.asarray(p['syn_qt'], dtype=np.float32)
    C = torch.cat([p['zeo_feat'], p['osda_feat']], dim=1).numpy().astype(np.float32)
    d_x, d_c = Xq.shape[1], C.shape[1]
    log('TabDDPM: d_x=%d, d_c=%d, steps=%d, epochs=%d' % (d_x, d_c, TAB_STEPS, TAB_EPOCHS))

    ab = torch.tensor(cosine_alpha_bar(TAB_STEPS), device=device)      # (T+1,)
    model = Denoiser(d_x, d_c).to(device)
    import copy
    ema = copy.deepcopy(model).eval()
    for q in ema.parameters():
        q.requires_grad_(False)
    n = len(tr)
    if reuse and os.path.exists(TAB_PT):
        ck = torch.load(TAB_PT, map_location=device, weights_only=False)
        ema.load_state_dict(ck['state_dict'])
        log('TabDDPM: 复用已训 ckpt (%d 行训练), 跳过训练' % ck['train_rows'])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=TAB_LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TAB_EPOCHS)
        x_tr = torch.tensor(Xq[tr], device=device)
        c_tr = torch.tensor(C[tr], device=device)
        for ep in range(TAB_EPOCHS):
            perm = torch.randperm(n, device=device)
            tot = 0.0; nb = 0
            for k in range(0, n, TAB_BATCH):
                b = perm[k:k + TAB_BATCH]
                t = torch.randint(1, TAB_STEPS + 1, (len(b),), device=device)
                a = ab[t].view(-1, 1)
                eps = torch.randn_like(x_tr[b])
                xt = a.sqrt() * x_tr[b] + (1 - a).sqrt() * eps
                pred = model(xt, c_tr[b], (t.float() / TAB_STEPS).view(-1, 1))
                loss = ((pred - eps) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(TAB_EMA).add_(pm, alpha=1 - TAB_EMA)
                tot += float(loss); nb += 1
            sched.step()
            if (ep + 1) % 50 == 0 or ep == 0:
                log('  epoch %3d/%d  loss %.4f  (%.1f min)' % (ep + 1, TAB_EPOCHS, tot / nb, (time.time() - t0) / 60))
        torch.save({'state_dict': ema.state_dict(), 'd_x': d_x, 'd_c': d_c,
                    'steps': TAB_STEPS, 'seed': SEED, 'train_rows': int(n)}, TAB_PT)

    # ---- 采样: 每体系 200（标准 DDPM 后验采样，x0 截断稳定）----
    zc, oc = cond_matrix(p, Z, O, elig)
    c_all = torch.cat([zc, oc], dim=1).to(device)
    c_rep = c_all.repeat_interleave(NS, dim=0)
    ema.eval()
    outs = []
    with torch.no_grad():
        for k in range(0, len(c_rep), 2048):
            cb = c_rep[k:k + 2048]
            x = torch.randn(len(cb), d_x, device=device)
            for t in range(TAB_STEPS, 0, -1):
                tb = torch.full((len(cb), 1), t / TAB_STEPS, device=device)
                eps = ema(x, cb, tb)
                a = ab[t]; a_prev = ab[t - 1]
                x0 = torch.clamp((x - (1 - a).sqrt() * eps) / a.sqrt(), 0, 1)
                if t > 1:
                    alpha_t = a / a_prev                     # α_t = ᾱ_t / ᾱ_{t-1}
                    beta = 1 - alpha_t
                    # 后验 q(x_{t-1}|x_t,x_0): 系数须用 sqrt(α_t)=sqrt(ᾱ_t/ᾱ_{t-1}),
                    # 用 sqrt(ᾱ_t) 会在高噪声段 (ᾱ_t→0) 抹掉 x_t 项 → 方差逐级塌缩
                    mean = (a_prev.sqrt() * beta / (1 - a)) * x0 + \
                           (alpha_t.sqrt() * (1 - a_prev) / (1 - a)) * x
                    var = (beta * (1 - a_prev) / (1 - a)).clamp(min=0)
                    x = mean + var.sqrt() * torch.randn_like(x)
                else:
                    x = x0
            outs.append(x.cpu().numpy())
    Xq_gen = np.vstack(outs)
    Xgen = w3.inv_qt(Xq_gen, p['qts'])
    save_pred(np.asarray(Xgen), elig, TAB_PQ, 'TabDDPM-MLP')
    log('TabDDPM: 训练+采样 %.1f min' % ((time.time() - t0) / 60))
    return {'epochs': TAB_EPOCHS, 'steps': TAB_STEPS, 'hidden': TAB_HID,
            'batch': TAB_BATCH, 'lr': TAB_LR, 'model_md5': md5(TAB_PT)}

# =====================================================================
# Random-Uniform: train 全局 [min,max]
# =====================================================================
def run_random(X, tr, elig):
    rng = np.random.RandomState(SEED)
    lo, hi = X[tr].min(0), X[tr].max(0)
    Xgen = rng.uniform(lo, hi, size=(len(elig) * NS, len(M.cols)))
    save_pred(Xgen, elig, RND_PQ, 'Random-Uniform')
    return {'lo': lo.tolist(), 'hi': hi.tolist()}

# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-tabddpm', action='store_true')
    ap.add_argument('--skip-kde', action='store_true')
    ap.add_argument('--reuse-tabddpm', action='store_true', help='复用已训 ckpt, 只重采样')
    args = ap.parse_args()
    t0 = time.time()
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    log('ZeoliteSynthesisBench v1 — Track A 自实现基线 | %s | device %s' % (
        time.strftime('%Y-%m-%d %H:%M:%S'), dev))
    p, X, Z, O, tr, te, elig = load_payload()
    meta = {'task': 'ZeoliteSynthesisBench v1 Track A self-implemented baselines',
            'spec': 'Documentation/ZeoliteSynthesisBench_v1_规格_骨架.md v1.1 FROZEN',
            'protocol': {'systems': len(elig), 'ns': NS, 'seed': SEED,
                         'universe': 'test n_true>=5, zeo!=Dense/Amorphous'},
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_md5': md5(DS_PT)}
    if not args.skip_kde:
        meta['kde'] = run_kde(X, Z, O, tr, te, elig)
    if not args.skip_tabddpm:
        meta['tabddpm'] = run_tabddpm(p, X, Z, O, tr, te, elig, dev, reuse=args.reuse_tabddpm)
    meta['random'] = run_random(X, tr, elig)
    meta['duration_min'] = round((time.time() - t0) / 60, 1)
    with open(MANIFEST, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    log('完成: %.1f min | manifest %s' % (meta['duration_min'], os.path.basename(MANIFEST)))

if __name__ == '__main__':
    main()
