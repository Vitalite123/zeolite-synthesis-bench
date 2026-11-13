# -*- coding: utf-8 -*-
"""
WP3/T3.2 DDPM v0.3 cryst_time 分布修正 (G3 提升路径第②项, EMA)
================================================================================
背景 (v0.1 eval + 尾部结构探针 2026-10-17): cryst_time std 比 v0 3.52 / v0.1 3.67 持续过散;
  探针定位 = 顶部 qt bin (0.999-1.0) 离群质量主导 — 裁剪到 train p99.9 后 std 比 0.87 (略欠散),
  >5000h 尾部占比 gen 0.26% vs train 0.06%, qt>0.99 质量占比 gen 0.74% < train 0.81%
  → 机理 = 稀疏重尾列 inv_qt 非线性放大边界游走 (qt 顶部 bin 跨 3600→14496h, 0.001 qt 误差
  → 数千小时 raw 离群), 非整体发散。
修正: EMA 权重平均 (decay 0.999, 标准 ramp warmup) — 平滑去噪器 → 减少边界游走样本;
  v0.1 登记的 v2 候选 ("EMA/更强加权") 中 EMA 路线, 单杠杆归因 (更强加权留 v2 备选)。
协议纪律: v0.2 遗产保持 (R1 qt 全量 / R2 per-col 加权 / 框架逆频过采样); 唯一自变量 = EMA。
  best-val 选模以 EMA 权重为准 (采样模型 = EMA); raw val 同步登记 (可比性注记)。
eval: v0.2 同协议 + cryst_time 尾部诊断直读 (尾部占比 / 裁剪敏感性 — EMA 对症性验证)。
用法: python wp3_ddpm_v0_3.py train|eval|all
"""
import os, sys, json, time, argparse, warnings
from collections import OrderedDict
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZG = os.path.join(BASE, 'vendor', 'zeosyn_gen')
os.chdir(ZG); sys.path.insert(0, ZG)

import torch

import wp3_ddpm_v0 as w3
import wp3_ddpm_v0_1 as v01
import wp3_ddpm_v0_2 as v02                      # 框架权重 / 路径惯例复用
from wp3_ddpm_v0_1 import build_diffusion_v01
from wp3_ddpm_v0_2 import framework_weights, ALPHA, S_REF

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------- 路径 (v0.3 专属) ----------------
DS_PT_V03    = v01.DS_PT_V01                     # dataset 逐位复用
MODEL_PT_V03 = os.path.join(BASE, 'weights', 'wp3_ddpm_v03_model.pt')
LOSS_CSV_V03 = os.path.join(BASE, 'data', 'wp3_ddpm_v03_loss.csv')
MANIFEST_V03 = os.path.join(BASE, 'data', 'wp3_ddpm_v03_manifest.json')
TRAIN_REP_V03 = os.path.join(BASE, 'data', 'wp3_ddpm_v03_train_report.txt')
EVAL_REP_V03  = os.path.join(BASE, 'data', 'wp3_ddpm_v03_eval_report.txt')
WSD_CSV_V03   = os.path.join(BASE, 'data', 'wp3_ddpm_v03_wsd_systems.csv')
LOSS_PNG_V03  = os.path.join(BASE, 'Documentation', 'wp3_ddpm_v03_loss.png')
DIST_PNG_V03  = os.path.join(BASE, 'Documentation', 'wp3_ddpm_v03_distcheck.png')
PRED_PARQUET  = os.path.join(BASE, 'data', 'wp3_ddpm_v03_syn_pred.parquet')
EMA_DECAY     = 0.999
V02_STD = None                                    # eval 时从 v0.2 manifest 读

L = []
def log(s=''):
    L.append(s); print(s, flush=True)

# =====================================================================
# EMA 工具
# =====================================================================
def ema_state_dict(diffusion, ema):
    """EMA 参数 + 当前 buffer → 可直接 load_state_dict 的完整 state_dict。"""
    sd = OrderedDict(diffusion.state_dict())
    for k, v in ema.items():
        sd[k] = v
    return sd

def val_with_ema(diffusion, ema, Xva, Zva, Ova, autocast_ctx):
    """EMA 权重下的 val loss (换入-计算-换回; 采样模型 = EMA → 选模以此为准)。"""
    backup = {k: v.detach().clone() for k, v in diffusion.named_parameters()}
    with torch.no_grad():
        for k, p in diffusion.named_parameters():
            p.copy_(ema[k])
    with torch.no_grad(), autocast_ctx():
        vl = float(diffusion(Xva.unsqueeze(1), zeo=Zva, osda=Ova).item())
    with torch.no_grad():
        for k, p in diffusion.named_parameters():
            p.copy_(backup[k])
    return vl

# =====================================================================
# train: v0.3 = v0.2 + EMA (唯一自变量 = EMA)
# =====================================================================
def train():
    t0 = time.time()
    payload = torch.load(DS_PT_V03, weights_only=False)
    meta = payload['meta']
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0); np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    diffusion, n_par, cw = build_diffusion_v01(device)
    log('==================== WP3/T3.2 DDPM v0.3 train (EMA decay %.4f) ====================' % EMA_DECAY)
    log('模型: MLPUnet %.2fM + WeightedColDiffusion1D (v0.2 遗产: qt 全量 / per-col 加权 / 框架逆频过采样)' % (n_par / 1e6))
    log('超参: 与 v0.2 逐位一致 — 唯一自变量 = EMA 权重平均 (选模与采样均用 EMA)')

    X = payload['syn_qt'].to(device); Z = payload['zeo_feat'].to(device); O = payload['osda_feat'].to(device)
    tr = payload['train_idx'].to(device); va = payload['val_idx'].to(device)
    zeo_key = np.asarray(payload['zeo_key'])
    tr_np = payload['train_idx'].numpy()
    Xtr, Ztr, Otr = X[tr], Z[tr], O[tr]
    Xva, Zva, Ova = X[va], Z[va], O[va]
    n_tr = Xtr.shape[0]

    w_np, sup = framework_weights(zeo_key, tr_np)
    W = torch.from_numpy(w_np).float().to(device)
    n_low = int((np.array([sup.get(z, 0) for z in zeo_key[tr_np]]) < S_REF).sum())
    log('数据: train %d / val %d | 低支撑批量份额 %.1f%% → %.1f%% (v0.2 同款)'
        % (n_tr, len(va), 100 * n_low / n_tr, 100 * w_np[w_np > 1].sum() / w_np.sum()))

    opt = torch.optim.Adam(diffusion.parameters(), lr=1e-4, betas=(0.9, 0.99))
    ema = {k: v.detach().clone() for k, v in diffusion.named_parameters()}
    train_log, val_log_raw, val_log_ema = [], [], []
    best_val, best_step = float('inf'), -1
    autocast_ctx = lambda: torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device != 'cpu'))

    for step in range(1, w3.STEPS + 1):
        idx = torch.multinomial(W, w3.BATCH, replacement=True)
        with autocast_ctx():
            loss = diffusion(Xtr[idx].unsqueeze(1), zeo=Ztr[idx], osda=Otr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        # EMA 更新 (ramp warmup: 前期等效更快收敛, ~10k 步后达到目标 decay)
        d = min(EMA_DECAY, (1 + step) / (10 + step))
        with torch.no_grad():
            for k, p in diffusion.named_parameters():
                ema[k].mul_(d).add_(p.detach(), alpha=1 - d)
        if step % w3.LOG_EVERY == 0:
            train_log.append((step, float(loss.item())))
        if step % w3.VAL_EVERY == 0:
            with torch.no_grad(), autocast_ctx():
                vl_raw = float(diffusion(Xva.unsqueeze(1), zeo=Zva, osda=Ova).item())
            vl_ema = val_with_ema(diffusion, ema, Xva, Zva, Ova, autocast_ctx)
            val_log_raw.append((step, vl_raw)); val_log_ema.append((step, vl_ema))
            if vl_ema < best_val:
                best_val, best_step = vl_ema, step
                torch.save({'step': step, 'val_loss': vl_ema, 'val_loss_raw': vl_raw,
                            'state_dict': ema_state_dict(diffusion, ema),
                            'n_params': n_par, 'col_weights': cw.tolist(),
                            'ema_decay': EMA_DECAY,
                            'sample_weights': {'alpha': ALPHA, 's_ref': S_REF},
                            'meta': {k: meta[k] for k in ('data_version', 'n_pop', 'split')}},
                           MODEL_PT_V03)
            pd.DataFrame({'step': [s for s, _ in train_log], 'train_loss': [l for _, l in train_log]}
                         ).merge(pd.DataFrame({'step': [s for s, _ in val_log_raw], 'val_loss_raw': [l for _, l in val_log_raw]}),
                                 on='step', how='outer'
                                 ).merge(pd.DataFrame({'step': [s for s, _ in val_log_ema], 'val_loss_ema': [l for _, l in val_log_ema]}),
                                         on='step', how='outer').sort_values('step').to_csv(LOSS_CSV_V03, index=False)
            if step % 2500 == 0:
                el = time.time() - t0
                print('step %d/%d | train %.4f | val raw %.4f / EMA %.4f (best %.4f @%d) | %.1f min, ETA %.1f min'
                      % (step, w3.STEPS, train_log[-1][1], vl_raw, vl_ema, best_val, best_step,
                         el / 60, el / step * (w3.STEPS - step) / 60), flush=True)

    # ---- loss 曲线 PNG ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if train_log:
        ts = np.array([s for s, _ in train_log]); ls = np.array([l for _, l in train_log])
        em = pd.Series(ls).ewm(alpha=0.005).mean().to_numpy()
        ax.plot(ts, ls, lw=0.4, alpha=0.25, color='tab:blue', label='train (raw)')
        ax.plot(ts, em, lw=1.6, color='tab:blue', label='train (EMA)')
    if val_log_raw:
        ax.plot([s for s, _ in val_log_raw], [l for _, l in val_log_raw], lw=1.2, color='tab:red',
                marker='.', ms=2.5, label='val raw')
        ax.plot([s for s, _ in val_log_ema], [l for _, l in val_log_ema], lw=1.2, color='tab:green',
                marker='.', ms=2.5, label='val EMA (选模)')
        ax.axvline(best_step, color='gray', ls=':', lw=1, label='best EMA-val @%d' % best_step)
    ax.set_xlabel('step'); ax.set_ylabel('diffusion loss (pred_v, l1 col-weighted)')
    ax.set_yscale('log'); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title('WP3/T3.2 DDPM v0.3 (v0.2 + EMA, zeobench_db_v1_8)')
    fig.tight_layout(); fig.savefig(LOSS_PNG_V03, dpi=160); plt.close(fig)

    # ---- manifest ----
    manifest = {
        'task': 'WP3/T3.2 DDPM v0.3 cryst_time 分布修正 (G3 提升路径②, EMA)',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'data_version': 'zeobench_db_v1_8', 'dataset_md5': w3.md5(DS_PT_V03),
        'data_md5': w3.md5(w3.DB_PARQUET),
        'code_md5': {'wp3_ddpm_v0_3.py': w3.md5(os.path.abspath(__file__)),
                     'wp3_ddpm_v0_2.py': w3.md5(v02.__file__),
                     'wp3_ddpm_v0_1.py': w3.md5(v01.__file__),
                     'wp3_ddpm_v0.py': w3.md5(os.path.join(BASE, 'wp3_ddpm_v0.py'))},
        'git_commit': w3.git_head(),
        'changes': {'v03': 'EMA 权重平均 decay %.3f (ramp warmup); 选模与采样均用 EMA — '
                           '对症 cryst_time 顶部 qt bin 边界游走 (探针: 裁 p99.9 后 std 比 0.87, '
                           '过散全由 ~0.26%% >5000h 离群贡献)' % EMA_DECAY,
                    'inherited': 'v0.2 框架逆频过采样 + v0.1 R1/R2'},
        'model': {'denoiser': 'MLPUnet (v0 同款)', 'params': int(n_par),
                  'interface': 'WeightedColDiffusion1D',
                  'col_weights': {c: float(cw[i]) for i, c in enumerate(w3.COLS)}},
        'hyperparams': {'lr': 1e-4, 'betas': [0.9, 0.99], 'batch': w3.BATCH, 'steps': w3.STEPS,
                        'cond_drop_prob': 0.1, 'ema_decay': EMA_DECAY, 'ema_ramp': '(1+step)/(10+step)',
                        'sample_weights': {'alpha': ALPHA, 's_ref': S_REF},
                        'precision': 'TF32 + bf16 autocast', 'selection': 'best-EMA-val checkpoint'},
        'best_val': {'step': best_step, 'val_loss_ema': best_val,
                     'val_loss_raw_at_best': val_log_raw[-1][1] if val_log_raw else None},
        'train_duration_min': round((time.time() - t0) / 60, 1),
    }
    with open(MANIFEST_V03, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    R = []
    R.append('WP3/T3.2 DDPM v0.3 训练报告 — %s' % time.strftime('%Y-%m-%d %H:%M'))
    R.append('=' * 78)
    R.append('【修正项】EMA decay %.3f (ramp warmup, 选模/采样均用 EMA); v0.2 遗产保持;' % EMA_DECAY)
    R.append('        协议其余与 v0.2 逐位一致 (唯一自变量 = EMA)')
    R.append('【收敛】train 首 %.4f → 末 %.4f; val EMA best %.4f @ step %d (选模口径; raw val 同步登记);' 
             ' 用时 %.1f min' % (train_log[0][1], train_log[-1][1], best_val, best_step, (time.time() - t0) / 60))
    R.append('【产出】%s / %s / %s' % (MODEL_PT_V03, LOSS_CSV_V03, LOSS_PNG_V03))
    with open(TRAIN_REP_V03, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n\n' + '\n'.join(R) + '\n')
    log('\n'.join(R))
    log('train 完成: best EMA-val %.4f @ step %d' % (best_val, best_step))

# =====================================================================
# eval: v0.2 同协议 + cryst_time 尾部诊断 (EMA 对症性直读)
# =====================================================================
def w1d(a, b):
    from scipy.stats import wasserstein_distance
    return wasserstein_distance(a, b)

def eval():
    t0 = time.time()
    payload = torch.load(DS_PT_V03, weights_only=False)
    meta, qts = payload['meta'], payload['qts']
    pop_max = np.array([meta['pop_max'][c] for c in w3.COLS])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    diffusion, n_par, _ = build_diffusion_v01(device)
    ck = torch.load(MODEL_PT_V03, map_location=device, weights_only=False)
    diffusion.load_state_dict(ck['state_dict']); diffusion.eval()
    log('==================== WP3/T3.2 DDPM v0.3 eval ====================')
    log('模型加载: best-EMA-val @ step %d (EMA val %.4f / raw %.4f), cond_scale=%.2f'
        % (ck['step'], ck['val_loss'], ck.get('val_loss_raw', float('nan')), w3.COND_SCALE))

    # 前版 std 比对照 (v0.1 eval 登记 / v0.2 manifest)
    prev_std = {'K/T': 1.78, 'cryst_time': 3.67}          # v0.1
    v02_mf = os.path.join(BASE, 'data', 'wp3_ddpm_v02_manifest.json')
    if os.path.exists(v02_mf):
        with open(v02_mf, encoding='utf-8') as f:
            prev_std = json.load(f).get('eval_v02', {}).get('std_ratio', prev_std)

    X_raw = payload['syn_raw']; Z = payload['zeo_feat'].to(device); O = payload['osda_feat'].to(device)
    tr_idx = payload['train_idx'].numpy(); te_idx = payload['test_idx'].numpy()
    zeo_key = payload['zeo_key']; osda_key = payload['osda_key']
    R = []

    # ---------- 1) 采样自检 ----------
    rng = np.random.RandomState(42)
    sel = rng.choice(tr_idx, size=min(2000, len(tr_idx)), replace=False)
    torch.manual_seed(42)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device != 'cpu')):
        syn = diffusion.sample(zeo=Z[sel], osda=O[sel], cond_scale=w3.COND_SCALE,
                               batch_size=len(sel)).squeeze(1).cpu().numpy()
    syn_raw = w3.inv_qt(syn, qts)
    tr_raw = X_raw[tr_idx]
    R.append('一、采样自检 (%d 训练行 × 1 样本; v0.3 = v0.2 + EMA)' % len(sel))
    R.append('%-12s %10s %8s %8s | %-24s | %-24s | %s' % ('特征', 'W1', '越界%', 'std比',
             'train p1/p50/p99', 'syn p1/p50/p99', 'v0.2 std比'))
    oob_flags, col_flags, std_map = [], [], {}
    for i, c in enumerate(w3.COLS):
        a, b = tr_raw[:, i], syn_raw[:, i]
        w1 = w1d(a, b)
        oob = float(((b < a.min()) | (b > a.max())).mean() * 100)
        sr = float(np.std(b) / (np.std(a) + 1e-12))
        std_map[c] = sr
        qa = np.quantile(a, [0.01, 0.5, 0.99]); qb = np.quantile(b, [0.01, 0.5, 0.99])
        R.append('%-12s %10.4f %8.2f %8.3f | %7.3g/%7.3g/%7.3g | %7.3g/%7.3g/%7.3g | %s'
                 % (c, w1, oob, sr, *qa, *qb, ('%.2f' % prev_std[c]) if c in prev_std else '—'))
        if oob > 5: oob_flags.append((c, oob))
        if sr < 0.5 or sr > 2: col_flags.append((c, sr))
    R.append('越界判定 (>5%%): %s; 塌缩/发散判定 (std 比 <0.5 或 >2): %s'
             % ('无' if not oob_flags else str(oob_flags), '无' if not col_flags else str(col_flags)))

    # cryst_time 尾部诊断 (EMA 对症性直读; v0.1 探针基线: gen >5000h 0.26% / 裁 p99.9 后 0.87)
    ct_i = w3.COLS.index('cryst_time')
    a, b = tr_raw[:, ct_i], syn_raw[:, ct_i]
    cap = float(np.quantile(a, 0.999))
    R.append('cryst_time 尾部诊断 (EMA 对症性): >2000h train %.2f%% / gen %.2f%% | '
             '>5000h train %.2f%% / gen %.2f%% (v0.1 探针: 0.26%%) | 裁 p99.9 (%.0fh) 后 std 比 %.2f (v0.1 探针: 0.87)'
             % (100 * (a > 2000).mean(), 100 * (b > 2000).mean(),
                100 * (a > 5000).mean(), 100 * (b > 5000).mean(),
                cap, float(np.minimum(b, cap).std() / (np.std(a) + 1e-12))))

    # PNG
    fig, axes = plt.subplots(4, 3, figsize=(13.5, 12))
    for i, c in enumerate(w3.COLS):
        ax = axes[i // 3][i % 3]
        a, b = tr_raw[:, i], syn_raw[:, i]
        use_log = i < 10 and (np.nanmax(a) > 20 * max(1e-12, np.nanmedian(a[a > 0]) if (a > 0).any() else 1))
        aa, bb = (np.log10(np.maximum(a, 1e-9)) if use_log else a,
                  np.log10(np.maximum(b, 1e-9)) if use_log else b)
        ax.hist(aa, bins=50, density=True, alpha=0.5, color='tab:blue', label='train')
        ax.hist(bb, bins=50, density=True, alpha=0.5, color='tab:red', label='sampled')
        ax.set_title(c + (' (log10)' if use_log else ''), fontsize=10)
        ax.set_yticks([])
        if i == 0: ax.legend(fontsize=8)
    fig.suptitle('WP3/T3.2 DDPM v0.3 sampling self-check (v0.2 + EMA)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(DIST_PNG_V03, dpi=160); plt.close(fig)
    log('分布自检 PNG: %s' % DIST_PNG_V03)

    # ---------- 2) WSD 同体系对照 + 分层 ----------
    n_te = len(te_idx); n_samp = 25
    chunks = []
    for c0 in range(0, n_te, 1000):
        rows = te_idx[c0:c0 + 1000]
        z = Z[rows].repeat_interleave(n_samp, dim=0)
        o = O[rows].repeat_interleave(n_samp, dim=0)
        torch.manual_seed(43 + c0 // 1000)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device != 'cpu')):
            s = diffusion.sample(zeo=z, osda=o, cond_scale=w3.COND_SCALE,
                                 batch_size=z.shape[0]).squeeze(1).cpu().numpy()
        chunks.append(w3.inv_qt(s, qts))
        log('WSD 采样: 测试行 %d/%d (每行 %d 样本)' % (min(c0 + 1000, n_te), n_te, n_samp))
    P = np.concatenate(chunks)
    syn_pred = pd.DataFrame(P, columns=w3.COLS)
    syn_pred['zeo'] = np.repeat(zeo_key[te_idx], n_samp)
    syn_pred['osda'] = np.repeat(osda_key[te_idx], n_samp)
    syn_pred.to_parquet(PRED_PARQUET, index=False)
    log('采样产物落盘: %s (%d 行)' % (os.path.basename(PRED_PARQUET), len(syn_pred)))
    syn_true = pd.DataFrame(X_raw[te_idx], columns=w3.COLS)
    syn_true['zeo'] = zeo_key[te_idx]; syn_true['osda'] = osda_key[te_idx]
    ps = syn_pred[w3.COLS] / pop_max
    ts = syn_true[w3.COLS] / pop_max

    rows = []
    for zeo, osda in syn_true[['zeo', 'osda']].value_counts().index:
        if zeo == 'Dense/Amorphous':
            continue
        m = (syn_pred['zeo'] == zeo) & (syn_pred['osda'] == osda)
        mt = (syn_true['zeo'] == zeo) & (syn_true['osda'] == osda)
        rows.append(dict(zeo=zeo, osda=osda, n_true=int(mt.sum()),
                         WSD=float(w3.wasserstein_distance(ps[m], ts[mt])),
                         MMD=float(w3.maximum_mean_discrepancy(ps[m], ts[mt]))))
    w = pd.DataFrame(rows)
    w.to_csv(WSD_CSV_V03, index=False)

    sup = pd.Series(zeo_key[tr_idx]).value_counts().to_dict()
    w['sup'] = [sup.get(z, 0) for z in w['zeo']]
    w['tier'] = np.where(w['sup'] < 5, 'few(<5)', np.where(w['sup'] < 50, 'mid(5-49)', 'rich(≥50)'))
    tier_v03 = w.groupby('tier')['WSD'].agg(['count', 'mean'])
    tier_prev = None
    for prev_csv, label in [(os.path.join(BASE, 'data', 'wp3_ddpm_v02_wsd_systems.csv'), 'v0.2')]:
        if os.path.exists(prev_csv):
            wp = pd.read_csv(prev_csv)
            wp['sup'] = [sup.get(z, 0) for z in wp['zeo']]
            wp['tier'] = np.where(wp['sup'] < 5, 'few(<5)', np.where(wp['sup'] < 50, 'mid(5-49)', 'rich(≥50)'))
            tier_prev = (label, wp.groupby('tier')['WSD'].agg(['count', 'mean']))

    R.append('')
    R.append('二、同体系 WSD 对照 (25 样本/行, cond_scale 0.75, 官方协议镜像)')
    R.append('体系数: %d (明细 %s)' % (len(w), os.path.basename(WSD_CSV_V03)))
    R.append('WSD mean ± std: %.4f ± %.4f  (v0.1: 0.4064; v0: 0.4142; 官方锚 0.4237)'
             % (w['WSD'].mean(), w['WSD'].std()))
    R.append('')
    R.append('三、WSD 框架支撑分层对照 (v0.2 遗产效应跟踪)')
    R.append('%-12s %6s | %10s %10s | %s' % ('tier', 'n_sys', 'v0.2 WSD', 'v0.3 WSD', 'Δ'))
    for t in ['few(<5)', 'mid(5-49)', 'rich(≥50)']:
        if tier_prev is not None and t in tier_prev[1].index and t in tier_v03.index:
            a_, b_ = float(tier_prev[1].loc[t, 'mean']), float(tier_v03.loc[t, 'mean'])
            R.append('%-12s %6d | %10.4f %10.4f | %+.4f' % (t, int(tier_v03.loc[t, 'count']), a_, b_, b_ - a_))

    R.append('')
    R.append('四、结论')
    R.append('1) EMA 对症性: cryst_time std 比 %.2f (v0.2 %s, v0.1 3.67; 目标 <2) + 尾部诊断行'
             % (std_map['cryst_time'], ('%.2f' % prev_std['cryst_time']) if 'cryst_time' in prev_std else '—'))
    R.append('2) K/T std 比 %.2f (v0.2 %s)' % (std_map['K/T'], ('%.2f' % prev_std['K/T']) if 'K/T' in prev_std else '—'))
    R.append('3) 采样产物 %s 供 WP5 正式读数' % os.path.basename(PRED_PARQUET))
    R.append('4) 数据版本 zeobench_db_v1_8; eval 用时 %.1f min' % ((time.time() - t0) / 60))
    with open(EVAL_REP_V03, 'w', encoding='utf-8') as f:
        f.write('\n'.join(R) + '\n')
    with open(MANIFEST_V03, encoding='utf-8') as f:
        manifest = json.load(f)
    manifest['eval_v03'] = {'wsd_systems': int(len(w)), 'wsd_mean': float(w['WSD'].mean()),
                            'wsd_std': float(w['WSD'].std()),
                            'wsd_tier': {t: {'n': int(tier_v03.loc[t, 'count']),
                                             'mean': float(tier_v03.loc[t, 'mean'])} for t in tier_v03.index},
                            'best_val_step': int(ck['step']),
                            'std_ratio': {c: std_map[c] for c in ('K/T', 'cryst_time')},
                            'ct_tail': {'gt2000_pct': float(100 * (syn_raw[:, ct_i] > 2000).mean()),
                                        'gt5000_pct': float(100 * (syn_raw[:, ct_i] > 5000).mean())},
                            'n_pred_rows': int(len(syn_pred)),
                            'eval_duration_min': round((time.time() - t0) / 60, 1)}
    with open(MANIFEST_V03, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log('eval 完成: 报告 %s' % EVAL_REP_V03)

# =====================================================================
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['train', 'eval', 'all'])
    a = ap.parse_args()
    if a.stage in ('train', 'all'):
        train()
    if a.stage in ('eval', 'all'):
        eval()
