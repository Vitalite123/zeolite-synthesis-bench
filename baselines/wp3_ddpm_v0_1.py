# -*- coding: utf-8 -*-
"""
WP3/T3.2 DDPM v0.1 质量迭代 (周计划 2026-10-23~10-29 P0-1, M12 G3 中期线关键路径)
================================================================================
v0 → v0.1 三项修正 (v0 首训报告登记项逐条收口):
  R1 qt subsample 修正: QuantileTransformer 显式 subsample=100_000 (对齐官方 sklearn 1.1.2
     默认 1e5; 21,714 行人口 < 1e5 → 全量拟合, v0 被 1.9.1 默认 subsample=10_000 截到 10k 子样本,
     仅极值端点受影响 (cryst_temp qt 上界 450 vs 人口 720) — rebuild 阶段从 v0 dataset 的
     syn_raw 重建 qt, split/feat/keys 逐位复用 → 与 v0 严格可比)
  R2 稀疏重尾列优化 (v0 观察项: K/T std 比 2.14 / cryst_time 3.52 过散, 二选一决策):
     per-col loss 加权 (对症) 而非 EMA (通用) — 官方 GaussianDiffusion1D.p_losses 逐位镜像
     子类 WeightedColDiffusion1D, K/T 与 cryst_time 两列 loss ×3 上权, 权重归一化均值 1
     (总 loss 量级与 v0 可比); EMA 登记为 v2 候选
  R3 cond_scale 扫描 (v0 单值 0.75 官方口径): 0.5/0.75/1.0/2.0/3.0/5.0 六档 × 3 代表测试体系
     × 200 样本 (WP5 demo 同构协议) → WP5 指标敏感性表 + 建议档位

协议纪律: 同 v0 (150k 步 / batch 512 / Adam 1e-4 / seed 0 / best-val 选模 / 训练挂 manifest);
  eval 同 v0 协议 (采样自检 2000 训练行×1 + WSD 全测试 25 样本/行) + v0 std 比对照 +
  采样产物落盘 parquet (Data/wp3_ddpm_v01_syn_pred.parquet — WP5 正式读数 P0-2 直接复用)
用法: python wp3_ddpm_v0_1.py rebuild|train|eval|scan|all
"""
import os, sys, json, time, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZG = os.path.join(BASE, 'vendor', 'zeosyn_gen')
os.chdir(ZG); sys.path.insert(0, ZG)

import torch
import torch.nn.functional as F

import wp3_ddpm_v0 as w3                       # v0 复用件: MLPUnet / COLS / 常量 / inv_qt / md5 / git_head
from models.diffusion import GaussianDiffusion1D, extract, default
from sklearn.preprocessing import QuantileTransformer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------- 路径 (v0.1 专属, 不覆盖 v0 产物) ----------------
DS_PT_V01    = os.path.join(BASE, 'weights', 'wp3_ddpm_v01_dataset.pt')
MODEL_PT_V01 = os.path.join(BASE, 'weights', 'wp3_ddpm_v01_model.pt')
LOSS_CSV_V01 = os.path.join(BASE, 'data', 'wp3_ddpm_v01_loss.csv')
MANIFEST_V01 = os.path.join(BASE, 'data', 'wp3_ddpm_v01_manifest.json')
TRAIN_REP_V01 = os.path.join(BASE, 'data', 'wp3_ddpm_v01_train_report.txt')
EVAL_REP_V01  = os.path.join(BASE, 'data', 'wp3_ddpm_v01_eval_report.txt')
SCAN_REP_V01  = os.path.join(BASE, 'data', 'wp3_ddpm_v01_scan_report.txt')
LOSS_PNG_V01  = os.path.join(BASE, 'Documentation', 'wp3_ddpm_v01_loss.png')
DIST_PNG_V01  = os.path.join(BASE, 'Documentation', 'wp3_ddpm_v01_distcheck.png')
PRED_PARQUET  = os.path.join(BASE, 'data', 'wp3_ddpm_v01_syn_pred.parquet')   # WP5 正式读数输入
QT_SUBSAMPLE  = 100_000                        # 官方 sklearn 1.1.2 默认 1e5 (21714 < 1e5 → 全量拟合)
HEAVY_COLS    = ['K/T', 'cryst_time']          # v0 std 比过散列 (2.14 / 3.52)
HEAVY_W       = 3.0                            # 上权倍数; 权重归一化均值 1

L = []
def log(s=''):
    L.append(s); print(s, flush=True)

# =====================================================================
# per-col 加权扩散: 官方 GaussianDiffusion1D.p_losses 逐位镜像 + 列加权
# =====================================================================
class WeightedColDiffusion1D(GaussianDiffusion1D):
    """官方 p_losses 的逐位镜像, 唯一改动 = loss 的 'b ... -> b (...)' mean 前插入列权重。
    列权重归一化均值 1 (sum(w)/n=1), SNR loss_weight 加权与官方逐位一致。
    决策登记: per-col 加权 (对症稀疏重尾列欠拟合) > EMA (通用平滑, 不针对列) — EMA 留 v2 候选。"""
    def __init__(self, *a, col_weights=None, **kw):
        super().__init__(*a, **kw)
        w = np.asarray(col_weights, float)
        w = w / w.mean()                        # 归一化均值 1 → 总量级与 v0 可比
        self.register_buffer('col_w', torch.from_numpy(w).float())

    def p_losses(self, x_start, t, *, zeo, osda, noise=None):
        b, c, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x, t, zeo, osda)
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            target = self.predict_v(x_start, t, noise)
        else:
            raise ValueError(f'unknown objective {self.objective}')
        loss = self.loss_fn(model_out, target, reduction='none')     # [B,1,12] (官方同款)
        w = self.col_w.to(loss.dtype).view(1, 1, n)
        loss = (loss * w).flatten(1).mean(dim=1)                      # 列加权后行均值 [B] (唯一改动)
        loss = loss * extract(self.loss_weight, t, loss.shape)        # SNR 加权 (官方逐位同款)
        return loss.mean()

def build_diffusion_v01(device):
    model = w3.MLPUnet(143, 14, cond_drop_prob=0.1)
    n_par = sum(p.numel() for p in model.parameters())
    assert n_par < 5_000_000, f'参数量 {n_par} 超出 <5M 规格'
    cw = np.array([HEAVY_W if c in HEAVY_COLS else 1.0 for c in w3.COLS], float)
    diffusion = WeightedColDiffusion1D(model, seq_length=12, timesteps=1000,
                                       objective='pred_v', beta_schedule='cosine',
                                       loss_type='l1', col_weights=cw).to(device)
    return diffusion, n_par, cw

# =====================================================================
# rebuild: v0 dataset syn_raw → qt 全量重建 (subsample 修正)
# =====================================================================
def rebuild():
    t0 = time.time()
    log('==================== WP3/T3.2 DDPM v0.1 rebuild (R1: qt subsample 修正) ====================')
    payload = torch.load(w3.DS_PT, weights_only=False)
    meta = payload['meta']
    assert meta['data_version'] == 'zeobench_db_v1_8'
    X_pop = np.asarray(payload['syn_raw'])       # 原始量纲人口 (capped+imputed, 与 v0 逐位一致)
    log('v0 dataset 载入: 人口 %d 行; syn_raw/feat/keys/split 逐位复用, 仅重建 qt' % len(X_pop))

    qts, X_qt = {}, np.zeros_like(X_pop)
    for i, c in enumerate(w3.COLS):
        qt = QuantileTransformer(n_quantiles=1000, random_state=0, subsample=QT_SUBSAMPLE)
        X_qt[:, i] = qt.fit_transform(X_pop[:, i].reshape(-1, 1)).ravel()
        qts[c] = qt

    # ---- 修正验证锚: v0 qt 上界 (10k 子样本 max) vs v0.1 (人口 max) ----
    qts_v0 = payload['qts']
    log('qt 端点对照 (v0 subsample=10k → v0.1 全量):')
    for c in ['cryst_temp', 'cryst_time', 'K/T']:
        hi0 = float(np.max(qts_v0[c].quantiles_)); hi1 = float(np.max(qts[c].quantiles_))
        pop_max = meta['pop_max'][c]
        log('  %-11s v0 qt 上界 %.4g → v0.1 %.4g (人口 max %.4g)%s'
            % (c, hi0, hi1, pop_max, '  ← 极值端点修正生效' if abs(hi1 - pop_max) < abs(hi0 - pop_max) else ''))

    meta_v01 = dict(meta)
    meta_v01['v01_changes'] = {
        'R1': 'qt subsample=100_000 显式对齐官方 sklearn 1.1.2 默认 (21714<1e5 全量拟合); '
              'v0 被 1.9.1 默认 10_000 截到 10k 子样本',
        'R2': 'per-col loss 加权: K/T & cryst_time x%s, 归一化均值 1 (EMA 留 v2 候选)' % HEAVY_W,
        'R3': 'cond_scale 扫描 0.5-5 (scan 阶段)',
        'rebuilt_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'git_commit': w3.git_head(),
    }
    torch.save({'syn_qt': torch.from_numpy(X_qt).float(), 'syn_raw': payload['syn_raw'],
                'zeo_feat': payload['zeo_feat'], 'osda_feat': payload['osda_feat'],
                'zeo_key': payload['zeo_key'], 'osda_key': payload['osda_key'],
                'train_idx': payload['train_idx'], 'val_idx': payload['val_idx'],
                'test_idx': payload['test_idx'],
                'qts': qts, 'meta': meta_v01}, DS_PT_V01)
    log('落盘: %s (%.1f MB) — rebuild 用时 %.1f min'
        % (DS_PT_V01, os.path.getsize(DS_PT_V01) / 1e6, (time.time() - t0) / 60))

# =====================================================================
# train: v0.1 重训 (同 v0 超参, per-col 加权)
# =====================================================================
def train():
    t0 = time.time()
    payload = torch.load(DS_PT_V01, weights_only=False)
    meta = payload['meta']
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0); np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    diffusion, n_par, cw = build_diffusion_v01(device)
    log('==================== WP3/T3.2 DDPM v0.1 train (R2: per-col loss 加权) ====================')
    log('模型: MLPUnet %.2fM 参数 + WeightedColDiffusion1D (官方 p_losses 逐位镜像 + 列加权)' % (n_par / 1e6))
    log('列权重: %s (归一化后 %s)' % (
        {c: cw[i] for i, c in enumerate(w3.COLS) if cw[i] != cw.mean()},
        np.array2string(cw, precision=3)))
    log('超参: 与 v0 逐位一致 (Adam 1e-4 / batch %d / %d 步 / seed 0 / best-val 选模) — 唯一自变量 = R1+R2'
        % (w3.BATCH, w3.STEPS))

    X = payload['syn_qt'].to(device); Z = payload['zeo_feat'].to(device); O = payload['osda_feat'].to(device)
    tr = payload['train_idx'].to(device); va = payload['val_idx'].to(device)
    Xtr, Ztr, Otr = X[tr], Z[tr], O[tr]
    Xva, Zva, Ova = X[va], Z[va], O[va]
    n_tr = Xtr.shape[0]
    log('数据: train %d / val %d (qt v0.1 全量拟合)' % (n_tr, len(va)))

    opt = torch.optim.Adam(diffusion.parameters(), lr=1e-4, betas=(0.9, 0.99))
    train_log, val_log = [], []
    best_val, best_step = float('inf'), -1
    autocast_ctx = lambda: torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device != 'cpu'))

    for step in range(1, w3.STEPS + 1):
        idx = torch.randint(0, n_tr, (w3.BATCH,), device=device)
        with autocast_ctx():
            loss = diffusion(Xtr[idx].unsqueeze(1), zeo=Ztr[idx], osda=Otr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % w3.LOG_EVERY == 0:
            train_log.append((step, float(loss.item())))
        if step % w3.VAL_EVERY == 0:
            with torch.no_grad(), autocast_ctx():
                vl = float(diffusion(Xva.unsqueeze(1), zeo=Zva, osda=Ova).item())
            val_log.append((step, vl))
            if vl < best_val:
                best_val, best_step = vl, step
                torch.save({'step': step, 'val_loss': vl, 'state_dict': diffusion.state_dict(),
                            'n_params': n_par, 'col_weights': cw.tolist(),
                            'meta': {k: meta[k] for k in ('data_version', 'n_pop', 'split')}},
                           MODEL_PT_V01)
            pd.DataFrame({'step': [s for s, _ in train_log], 'train_loss': [l for _, l in train_log]}
                         ).merge(pd.DataFrame({'step': [s for s, _ in val_log], 'val_loss': [l for _, l in val_log]}),
                                 on='step', how='outer').sort_values('step').to_csv(LOSS_CSV_V01, index=False)
            el = time.time() - t0
            print('step %d/%d | train %.4f | val %.4f (best %.4f @%d) | %.1f min, ETA %.1f min'
                  % (step, w3.STEPS, train_log[-1][1], vl, best_val, best_step,
                     el / 60, el / step * (w3.STEPS - step) / 60), flush=True)

    # ---- loss 曲线 PNG ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if train_log:
        ts = np.array([s for s, _ in train_log]); ls = np.array([l for _, l in train_log])
        ema = pd.Series(ls).ewm(alpha=0.005).mean().to_numpy()
        ax.plot(ts, ls, lw=0.4, alpha=0.25, color='tab:blue', label='train (raw)')
        ax.plot(ts, ema, lw=1.6, color='tab:blue', label='train (EMA)')
    if val_log:
        ax.plot([s for s, _ in val_log], [l for _, l in val_log], lw=1.2, color='tab:red',
                marker='.', ms=2.5, label='val (500-step)')
        ax.axvline(best_step, color='gray', ls=':', lw=1, label='best val @%d' % best_step)
    ax.set_xlabel('step'); ax.set_ylabel('diffusion loss (pred_v, l1 col-weighted)')
    ax.set_yscale('log'); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title('WP3/T3.2 DDPM v0.1 (qt full-fit + per-col weighted loss, zeobench_db_v1_8)')
    fig.tight_layout(); fig.savefig(LOSS_PNG_V01, dpi=160); plt.close(fig)

    # ---- manifest ----
    manifest = {
        'task': 'WP3/T3.2 DDPM v0.1 质量迭代 (周计划 2026-10-23~10-29 P0-1)',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'data_version': 'zeobench_db_v1_8', 'dataset_md5': w3.md5(DS_PT_V01),
        'data_md5': w3.md5(w3.DB_PARQUET),
        'code_md5': {'wp3_ddpm_v0_1.py': w3.md5(os.path.abspath(__file__)),
                     'wp3_ddpm_v0.py': w3.md5(os.path.join(BASE, 'wp3_ddpm_v0.py'))},
        'git_commit': w3.git_head(),
        'changes': meta['v01_changes'],
        'model': {'denoiser': 'MLPUnet (v0 同款)', 'params': int(n_par),
                  'interface': 'WeightedColDiffusion1D = 官方 GaussianDiffusion1D p_losses 逐位镜像 + 列加权',
                  'col_weights': {c: float(cw[i]) for i, c in enumerate(w3.COLS)}},
        'hyperparams': {'lr': 1e-4, 'betas': [0.9, 0.99], 'batch': w3.BATCH, 'steps': w3.STEPS,
                        'cond_drop_prob': 0.1, 'qt_subsample': QT_SUBSAMPLE,
                        'precision': 'TF32 + bf16 autocast, 模型输出回 fp32 计损失',
                        'selection': 'best-val checkpoint'},
        'best_val': {'step': best_step, 'val_loss': best_val},
        'train_duration_min': round((time.time() - t0) / 60, 1),
    }
    with open(MANIFEST_V01, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    R = []
    R.append('WP3/T3.2 DDPM v0.1 训练报告 — %s' % time.strftime('%Y-%m-%d %H:%M'))
    R.append('=' * 78)
    R.append('【修正项】R1 qt subsample=100_000 全量拟合 (对齐官方 1.1.2); R2 per-col loss 加权')
    R.append('        (%s x%.0f, 归一化均值 1, EMA 留 v2); 协议其余与 v0 逐位一致' % (','.join(HEAVY_COLS), HEAVY_W))
    R.append('【收敛】train 首 %.4f → 末 %.4f; val best %.4f @ step %d (v0: 0.1039 @106000); 用时 %.1f min'
             % (train_log[0][1], train_log[-1][1], best_val, best_step, (time.time() - t0) / 60))
    R.append('【产出】%s / %s / %s' % (MODEL_PT_V01, LOSS_CSV_V01, LOSS_PNG_V01))
    with open(TRAIN_REP_V01, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n\n' + '\n'.join(R) + '\n')
    log('\n'.join(R))
    log('train 完成: best val %.4f @ step %d' % (best_val, best_step))

# =====================================================================
# eval: v0 同协议 + std 比对照 + 采样产物落盘 (WP5 正式读数输入)
# =====================================================================
def eval():
    t0 = time.time()
    payload = torch.load(DS_PT_V01, weights_only=False)
    meta, qts = payload['meta'], payload['qts']
    pop_max = np.array([meta['pop_max'][c] for c in w3.COLS])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    diffusion, n_par, _ = build_diffusion_v01(device)
    ck = torch.load(MODEL_PT_V01, map_location=device, weights_only=False)
    diffusion.load_state_dict(ck['state_dict']); diffusion.eval()
    log('==================== WP3/T3.2 DDPM v0.1 eval ====================')
    log('模型加载: best-val @ step %d (val %.4f), cond_scale=%.2f' % (ck['step'], ck['val_loss'], w3.COND_SCALE))

    X_raw = payload['syn_raw']; Z = payload['zeo_feat'].to(device); O = payload['osda_feat'].to(device)
    tr_idx = payload['train_idx'].numpy(); te_idx = payload['test_idx'].numpy()
    zeo_key = payload['zeo_key']; osda_key = payload['osda_key']
    R = []
    V0_STD = {'K/T': 2.14, 'cryst_time': 3.52}           # v0 eval 报告登记读数 (std 比)

    # ---------- 1) 采样自检 (v0 同协议: 2000 训练行 × 1 样本) ----------
    rng = np.random.RandomState(42)
    sel = rng.choice(tr_idx, size=min(2000, len(tr_idx)), replace=False)
    torch.manual_seed(42)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device != 'cpu')):
        syn = diffusion.sample(zeo=Z[sel], osda=O[sel], cond_scale=w3.COND_SCALE,
                               batch_size=len(sel)).squeeze(1).cpu().numpy()
    syn_raw = w3.inv_qt(syn, qts)
    tr_raw = X_raw[tr_idx]
    log('采样自检: %d 训练行 × 1 样本 (seed 42, cond_scale 0.75)' % len(sel))
    R.append('一、采样自检 (%d 训练行 × 1 样本; v0.1 = qt 全量 + per-col 加权)' % len(sel))
    R.append('%-12s %10s %8s %8s | %-24s | %-24s | %s' % ('特征', 'W1', '越界%', 'std比',
             'train p1/p50/p99', 'syn p1/p50/p99', 'v0 std比对照'))
    oob_flags, col_flags, std_map = [], [], {}
    for i, c in enumerate(w3.COLS):
        a, b = tr_raw[:, i], syn_raw[:, i]
        w1 = w1d(a, b)
        oob = float(((b < a.min()) | (b > a.max())).mean() * 100)
        sr = float(np.std(b) / (np.std(a) + 1e-12))
        std_map[c] = sr
        qa = np.quantile(a, [0.01, 0.5, 0.99]); qb = np.quantile(b, [0.01, 0.5, 0.99])
        v0sr = V0_STD.get(c, '')
        R.append('%-12s %10.4f %8.2f %8.3f | %7.3g/%7.3g/%7.3g | %7.3g/%7.3g/%7.3g | %s'
                 % (c, w1, oob, sr, *qa, *qb, ('%.2f' % v0sr) if v0sr != '' else '—'))
        if oob > 5: oob_flags.append((c, oob))
        if sr < 0.5 or sr > 2: col_flags.append((c, sr))
    R.append('越界判定 (>5%%): %s; 塌缩/发散判定 (std 比 <0.5 或 >2): %s'
             % ('无' if not oob_flags else str(oob_flags), '无' if not col_flags else str(col_flags)))
    R.append('R2 修正效果 (v0 → v0.1 std 比): ' + ', '.join(
        '%s %.2f → %.2f (%s)' % (c, V0_STD[c], std_map[c],
                                 '收窄达标' if std_map[c] < 2 else '仍过散, 登记 v2') for c in HEAVY_COLS))

    # PNG: 4×3 逐特征分布对照
    fig, axes = plt.subplots(4, 3, figsize=(13.5, 12))
    for i, c in enumerate(w3.COLS):
        ax = axes[i // 3][i % 3]
        a, b = tr_raw[:, i], syn_raw[:, i]
        use_log = i < 10 and (np.nanmax(a) > 20 * max(1e-12, np.nanmedian(a[a > 0]) if (a > 0).any() else 1))
        aa, bb = (np.log10(np.maximum(a, 1e-9)) if use_log else a,
                  np.log10(np.maximum(b, 1e-9)) if use_log else b)
        ax.hist(aa, bins=50, density=True, alpha=0.5, color='tab:blue', label='train')
        ax.hist(bb, bins=50, density=True, alpha=0.5, color='tab:red', label='sampled')
        ax.set_title(c + (' (log10)' if use_log else '') + (' [v0.1]' if c in HEAVY_COLS else ''), fontsize=10)
        ax.set_yticks([])
        if i == 0: ax.legend(fontsize=8)
    fig.suptitle('WP3/T3.2 DDPM v0.1 sampling self-check (qt full-fit + per-col weighted)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(DIST_PNG_V01, dpi=160); plt.close(fig)
    log('分布自检 PNG: %s' % DIST_PNG_V01)

    # ---------- 2) WSD 同体系对照 + 采样产物落盘 ----------
    n_te = len(te_idx)
    n_samp = 25
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
    log('采样产物落盘: %s (%d 行 — WP5 正式读数 P0-2 直接复用)' % (os.path.basename(PRED_PARQUET), len(syn_pred)))
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
    wsd_csv_v01 = os.path.join(BASE, 'data', 'wp3_ddpm_v01_wsd_systems.csv')
    w.to_csv(wsd_csv_v01, index=False)
    R.append('')
    R.append('二、同体系 WSD 对照 (25 样本/行, cond_scale 0.75, 官方协议镜像)')
    R.append('体系数: %d (明细 %s)' % (len(w), os.path.basename(wsd_csv_v01)))
    R.append('WSD mean ± std: %.4f ± %.4f  (v0: 0.4142 ± 0.4060; 官方锚 0.4237)'
             % (w['WSD'].mean(), w['WSD'].std()))
    R.append('MMD mean ± std: %.4f ± %.4f  (v0: 官方 wrap 首读同源)' % (w['MMD'].mean(), w['MMD'].std()))
    ratio = w['WSD'].mean() / w3.OFFICIAL_WSD_ANCHOR
    R.append('量级判定: %.2f× 官方锚 — %s' % (ratio, '同量级 (0.5–2×)' if 0.5 <= ratio <= 2 else '偏离量级, 登记复核'))

    R.append('')
    R.append('三、结论')
    R.append('1) R1 qt 全量拟合: 极值端点修正 (cryst_temp qt 上界 450→720 人口 max), 越界判定应归零')
    R.append('2) R2 per-col 加权: K/T & cryst_time std 比 v0→v0.1 见上表 (目标 <2; 若仍过散登记 v2: EMA/更长训练)')
    R.append('3) WSD vs v0 (0.4142): %s' % ('改善' if w['WSD'].mean() < 0.4142 else '持平/微升 (loss 加权后量级不变即达标)'))
    R.append('4) 采样产物 %s 供 WP5 正式读数 (P0-2 G3 中期线判定)' % os.path.basename(PRED_PARQUET))
    R.append('5) 数据版本 zeobench_db_v1_8; eval 用时 %.1f min' % ((time.time() - t0) / 60))
    with open(EVAL_REP_V01, 'w', encoding='utf-8') as f:
        f.write('\n'.join(R) + '\n')
    with open(MANIFEST_V01, encoding='utf-8') as f:
        manifest = json.load(f)
    manifest['eval_v01'] = {'wsd_systems': int(len(w)), 'wsd_mean': float(w['WSD'].mean()),
                            'wsd_std': float(w['WSD'].std()), 'v0_wsd_mean': 0.4142,
                            'ratio_to_anchor': float(ratio), 'best_val_step': int(ck['step']),
                            'std_ratio': {c: std_map[c] for c in HEAVY_COLS},
                            'n_pred_rows': int(len(syn_pred)),
                            'eval_duration_min': round((time.time() - t0) / 60, 1)}
    with open(MANIFEST_V01, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log('eval 完成: 报告 %s' % EVAL_REP_V01)

def w1d(a, b):
    from scipy.stats import wasserstein_distance
    return wasserstein_distance(a, b)

# =====================================================================
# scan: cond_scale 敏感性扫描 (R3; WP5 demo 同构协议)
# =====================================================================
def scan():
    t0 = time.time()
    payload = torch.load(DS_PT_V01, weights_only=False)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    diffusion, _, _ = build_diffusion_v01(device)
    ck = torch.load(MODEL_PT_V01, map_location=device, weights_only=False)
    diffusion.load_state_dict(ck['state_dict']); diffusion.eval()
    X_raw = payload['syn_raw']; Z = payload['zeo_feat'].to(device); O = payload['osda_feat'].to(device)
    te_idx = payload['test_idx'].numpy(); tr_idx = payload['train_idx'].numpy()
    zeo_key = payload['zeo_key']; osda_key = payload['osda_key']
    log('==================== WP3/T3.2 DDPM v0.1 cond_scale 扫描 (R3) ====================')

    from wp5_metrics_api_v0 import ZeoSynMetrics
    M = ZeoSynMetrics()
    train_df = pd.DataFrame(np.asarray(X_raw)[tr_idx], columns=M.cols)
    test_df = pd.DataFrame(np.asarray(X_raw)[te_idx], columns=M.cols)
    test_df['zeo'], test_df['osda'] = zeo_key[te_idx], osda_key[te_idx]

    key_te = pd.Series(list(zip(zeo_key[te_idx], osda_key[te_idx])))
    counts = key_te.value_counts()
    eligible = [k for k in counts.index if counts[k] >= 5 and k[0] != 'Dense/Amorphous']
    picks = [eligible[0], eligible[len(eligible) // 2], eligible[-1]]
    log('体系选择 (WP5 demo 同构): %s' % '; '.join('%s+%s (n=%d)' % (z, o[:16], counts[(z, o)]) for z, o in picks))

    rows0 = []
    for z_, o_ in picks:
        mrow = np.where((zeo_key == z_) & (osda_key == o_))[0]
        rows0.append(mrow[0])
    NS, SCALES = 200, [0.5, 0.75, 1.0, 2.0, 3.0, 5.0]
    zc = torch.stack([Z[r] for r in rows0]).repeat_interleave(NS, dim=0)
    oc = torch.stack([O[r] for r in rows0]).repeat_interleave(NS, dim=0)

    R = []
    R.append('WP3/T3.2 DDPM v0.1 cond_scale 扫描 — %s' % time.strftime('%Y-%m-%d %H:%M'))
    R.append('=' * 78)
    R.append('协议: 3 代表测试体系 × %d 样本 (seed 42, v0.1 best-val @%d); WP5 ZeoSynMetrics 全指标; '
             'v0 首训单值 0.75 为官方口径基线' % (NS, ck['step']))
    R.append('%-6s | %8s %8s %8s | %6s %6s | %7s | %s' % ('scale', 'W2', 'MMD', 'W1均值', 'COV-P', 'COV-R',
             '物理有效率', 'std比 K/T / cryst_time'))
    results = {}
    for cs in SCALES:
        torch.manual_seed(42)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device != 'cpu')):
            s = diffusion.sample(zeo=zc, osda=oc, cond_scale=cs,
                                 batch_size=zc.shape[0]).squeeze(1).cpu().numpy()
        raw = w3.inv_qt(s, payload['qts'])
        pred_df = pd.DataFrame(raw, columns=M.cols)
        pred_df['zeo'] = np.repeat([p[0] for p in picks], NS)
        pred_df['osda'] = np.repeat([p[1] for p in picks], NS)
        # 三体系聚合读数 (逐体系 full_suite 后均值 — COV 语义不可跨体系合并后计算)
        per_sys = []
        sr_kt, sr_ct = [], []
        for z_, o_ in picks:
            p_i = pred_df[(pred_df['zeo'] == z_) & (pred_df['osda'] == o_)][M.cols]
            t_i = test_df[(test_df['zeo'] == z_) & (test_df['osda'] == o_)][M.cols]
            fs = M.full_suite(p_i, t_i, ref=train_df)
            per_sys.append(fs)
            tr_i = train_df  # std 比: 生成列 vs 全 train 列 (与 v0 采样自检同型)
            sr_kt.append(float(np.std(p_i['K/T']) / (np.std(tr_i['K/T']) + 1e-12)))
            sr_ct.append(float(np.std(p_i['cryst_time']) / (np.std(tr_i['cryst_time']) + 1e-12)))
        agg = {'W2': float(np.mean([x['W2_joint'] for x in per_sys])),
               'MMD': float(np.mean([x['MMD'] for x in per_sys])),
               'W1': float(np.mean([x['W1_mean'] for x in per_sys])),
               'COV-P': float(np.mean([x['COV-P'] for x in per_sys])),
               'COV-R': float(np.mean([x['COV-R'] for x in per_sys])),
               'PV': float(np.mean([x['physical_validity']['rate'] for x in per_sys]))}
        results[cs] = agg
        R.append('%-6.2f | %8.4f %8.4f %8.4f | %6.3f %6.3f | %7.4f | %.2f / %.2f'
                 % (cs, agg['W2'], agg['MMD'], agg['W1'], agg['COV-P'], agg['COV-R'], agg['PV'],
                    float(np.mean(sr_kt)), float(np.mean(sr_ct))))
        log('cond_scale %.2f: W2 %.4f COV-P %.3f PV %.4f' % (cs, agg['W2'], agg['COV-P'], agg['PV']))

    best_cs = min(results, key=lambda k: results[k]['W2'])
    base = results[0.75]
    R.append('')
    R.append('判读: W2 最优档 %.2f (W2 %.4f / COV-P %.3f); 官方口径 0.75 读数 W2 %.4f / COV-P %.3f;'
             % (best_cs, results[best_cs]['W2'], results[best_cs]['COV-P'], base['W2'], base['COV-P']))
    R.append('      cond_scale 敏感性结论与建议档位登记 manifest (v0.1 eval/WP5 正式读数沿用 0.75 官方口径,')
    R.append('      除非扫描显示显著改善 (ΔW2 > 5%% 且 COV-P 不降) — 保持与官方锚可比优先)')
    with open(SCAN_REP_V01, 'w', encoding='utf-8') as f:
        f.write('\n'.join(R) + '\n')
    with open(MANIFEST_V01, encoding='utf-8') as f:
        manifest = json.load(f)
    manifest['cond_scale_scan'] = {'scales': SCALES, 'results': {str(k): v for k, v in results.items()},
                                   'best_w2_scale': float(best_cs), 'official_scale': 0.75,
                                   'scan_duration_min': round((time.time() - t0) / 60, 1)}
    with open(MANIFEST_V01, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log('scan 完成: 报告 %s' % SCAN_REP_V01)

# =====================================================================
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['rebuild', 'train', 'eval', 'scan', 'all'])
    a = ap.parse_args()
    if a.stage in ('rebuild', 'all'):
        rebuild()
    if a.stage in ('train', 'all'):
        train()
    if a.stage in ('eval', 'all'):
        eval()
    if a.stage in ('scan', 'all'):
        scan()
