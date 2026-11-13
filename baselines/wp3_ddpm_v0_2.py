# -*- coding: utf-8 -*-
"""
WP3/T3.2 DDPM v0.2 低支撑体系修复 (G3 提升路径第①项, 接续 P0-2 判定报告 §四)
================================================================================
背景 (G3 正式读数 2026-10-17): COV-P v0.1 0.7286 未过 0.75 线, 分层归因 = few/mid
  低支撑 27 体系 (COV-P 0.641/0.687) 拉低等权均值 ~2.2pp = 主差距量级;
  rich 96 体系 0.749 已逼近线。
机理: 低支撑框架 (train 行 <50) 的 zeo_feat 条件方向欠训练 → 条件生成质量受限
  (COV-P 为 precision 口径 — 生成样本落不到该框架典型配方区)。
修正: 框架级逆频过采样 — 训练批量采样概率 ∝ 行权 w(s) = (50/s)^0.5 (s<50) / 1 (s≥50),
  即 inverse-sqrt-frequency 平衡 (标准温和档); 探针读数: 低支撑批量份额 10.3% → 16.4%,
  few 行权 3.5-7.1×。α=0.5 选择依据: α=0.75/1.0 的 1-3 行框架行权 6.6-50× 会记忆化
  (点质量条件 → 若 test 配方偏离, precision 反降), 温和档规避。
结构性天花板登记: 123 合格体系中 7 个 few 体系框架 train 支撑 = 0 (纯描述子外推,
  过采样不可及 — 提升上限受此约束, 判定报告归因)。
协议纪律: 数据/超参/评价与 v0.1 逐位一致 (唯一自变量 = 批量采样分布);
  R1 qt 全量 + R2 per-col 加权保持 (v0.1 遗产); dataset 复用 DS_PT_V01 不重建。
eval: v0.1 同协议 (采样自检 + WSD 全测试) + WSD 框架支撑分层对照 vs v0.1 (修复效果直读)
  + 采样产物落盘 parquet (WP5 正式读数复用)。
用法: python wp3_ddpm_v0_2.py train|eval|all
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

import wp3_ddpm_v0 as w3
import wp3_ddpm_v0_1 as v01                      # dataset / build_diffusion_v01 复用
from wp3_ddpm_v0_1 import build_diffusion_v01

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------- 路径 (v0.2 专属) ----------------
DS_PT_V02    = v01.DS_PT_V01                     # dataset 逐位复用 (不重建)
MODEL_PT_V02 = os.path.join(BASE, 'weights', 'wp3_ddpm_v02_model.pt')
LOSS_CSV_V02 = os.path.join(BASE, 'data', 'wp3_ddpm_v02_loss.csv')
MANIFEST_V02 = os.path.join(BASE, 'data', 'wp3_ddpm_v02_manifest.json')
TRAIN_REP_V02 = os.path.join(BASE, 'data', 'wp3_ddpm_v02_train_report.txt')
EVAL_REP_V02  = os.path.join(BASE, 'data', 'wp3_ddpm_v02_eval_report.txt')
WSD_CSV_V02   = os.path.join(BASE, 'data', 'wp3_ddpm_v02_wsd_systems.csv')
LOSS_PNG_V02  = os.path.join(BASE, 'Documentation', 'wp3_ddpm_v02_loss.png')
DIST_PNG_V02  = os.path.join(BASE, 'Documentation', 'wp3_ddpm_v02_distcheck.png')
PRED_PARQUET  = os.path.join(BASE, 'data', 'wp3_ddpm_v02_syn_pred.parquet')
ALPHA         = 0.5                              # 逆频指数 (温和档)
S_REF         = 50                               # 支撑参照 (rich 分界)

L = []
def log(s=''):
    L.append(s); print(s, flush=True)

# =====================================================================
# 框架级采样权重
# =====================================================================
def framework_weights(zeo_key, tr_idx):
    """w(s) = (S_REF/s)^ALPHA if s<S_REF else 1 — inverse-sqrt-frequency 温和平衡。
    返回 (train 行权 np.array, 框架支撑 dict)。"""
    sup = pd.Series(zeo_key[tr_idx]).value_counts().to_dict()
    s_rows = np.array([sup.get(z, 0) for z in zeo_key[tr_idx]], float)
    w = np.where(s_rows < S_REF, (S_REF / np.maximum(s_rows, 1)) ** ALPHA, 1.0)
    return w, sup

# =====================================================================
# train: v0.2 重训 (唯一自变量 = 批量采样分布; 其余与 v0.1 逐位一致)
# =====================================================================
def train():
    t0 = time.time()
    payload = torch.load(DS_PT_V02, weights_only=False)
    meta = payload['meta']
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0); np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    diffusion, n_par, cw = build_diffusion_v01(device)
    log('==================== WP3/T3.2 DDPM v0.2 train (框架级逆频过采样 α=%.1f) ====================' % ALPHA)
    log('模型: MLPUnet %.2fM + WeightedColDiffusion1D (v0.1 同款, R1/R2 遗产保持)' % (n_par / 1e6))
    log('超参: 与 v0.1 逐位一致 — 唯一自变量 = 批量采样分布 (uniform → inverse-sqrt-frequency)')

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
    log('数据: train %d / val %d | 低支撑框架 (<%d 行) 覆盖 %d train 行, 批量份额 %.1f%% → %.1f%%'
        % (n_tr, len(va), S_REF, n_low, 100 * n_low / n_tr, 100 * w_np[w_np > 1].sum() / w_np.sum()))
    log('行权分布: few(支撑1-4) %.1f-%.1f× | mid(5-49) %.2f-%.2f× | rich 1.0×'
        % ((S_REF / 4) ** ALPHA, (S_REF / 1) ** ALPHA, (S_REF / 49) ** ALPHA, (S_REF / 5) ** ALPHA))

    opt = torch.optim.Adam(diffusion.parameters(), lr=1e-4, betas=(0.9, 0.99))
    train_log, val_log = [], []
    best_val, best_step = float('inf'), -1
    autocast_ctx = lambda: torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device != 'cpu'))

    for step in range(1, w3.STEPS + 1):
        idx = torch.multinomial(W, w3.BATCH, replacement=True)     # ← 唯一改动 (v0.1: randint uniform)
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
                            'sample_weights': {'alpha': ALPHA, 's_ref': S_REF},
                            'meta': {k: meta[k] for k in ('data_version', 'n_pop', 'split')}},
                           MODEL_PT_V02)
            pd.DataFrame({'step': [s for s, _ in train_log], 'train_loss': [l for _, l in train_log]}
                         ).merge(pd.DataFrame({'step': [s for s, _ in val_log], 'val_loss': [l for _, l in val_log]}),
                                 on='step', how='outer').sort_values('step').to_csv(LOSS_CSV_V02, index=False)
            if step % 2500 == 0:
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
    ax.set_title('WP3/T3.2 DDPM v0.2 (framework inverse-sqrt oversampling, zeobench_db_v1_8)')
    fig.tight_layout(); fig.savefig(LOSS_PNG_V02, dpi=160); plt.close(fig)

    # ---- manifest ----
    manifest = {
        'task': 'WP3/T3.2 DDPM v0.2 低支撑体系修复 (G3 提升路径①, G3 判定报告 §四)',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'data_version': 'zeobench_db_v1_8', 'dataset_md5': w3.md5(DS_PT_V02),
        'data_md5': w3.md5(w3.DB_PARQUET),
        'code_md5': {'wp3_ddpm_v0_2.py': w3.md5(os.path.abspath(__file__)),
                     'wp3_ddpm_v0_1.py': w3.md5(v01.__file__),
                     'wp3_ddpm_v0.py': w3.md5(os.path.join(BASE, 'wp3_ddpm_v0.py'))},
        'git_commit': w3.git_head(),
        'changes': {'v02': '框架级逆频过采样: 批量采样 ∝ (50/s)^0.5 (s<50)/1 (s≥50), '
                           '低支撑批量份额 10.3%%→16.4%%; 7 个 few 体系框架支撑=0 不可及 (结构性天花板登记)',
                    'inherited': meta['v01_changes']},
        'model': {'denoiser': 'MLPUnet (v0 同款)', 'params': int(n_par),
                  'interface': 'WeightedColDiffusion1D (R2 per-col 加权保持)',
                  'col_weights': {c: float(cw[i]) for i, c in enumerate(w3.COLS)}},
        'hyperparams': {'lr': 1e-4, 'betas': [0.9, 0.99], 'batch': w3.BATCH, 'steps': w3.STEPS,
                        'cond_drop_prob': 0.1, 'sample_weights': {'alpha': ALPHA, 's_ref': S_REF,
                                                                  'scheme': 'inverse-sqrt-frequency'},
                        'precision': 'TF32 + bf16 autocast', 'selection': 'best-val checkpoint'},
        'best_val': {'step': best_step, 'val_loss': best_val},
        'train_duration_min': round((time.time() - t0) / 60, 1),
    }
    with open(MANIFEST_V02, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    R = []
    R.append('WP3/T3.2 DDPM v0.2 训练报告 — %s' % time.strftime('%Y-%m-%d %H:%M'))
    R.append('=' * 78)
    R.append('【修正项】框架级逆频过采样 α=%.1f (低支撑批量份额 10.3%%→16.4%%); R1/R2 v0.1 遗产保持;' % ALPHA)
    R.append('        协议其余与 v0.1 逐位一致 (唯一自变量 = 批量采样分布)')
    R.append('【收敛】train 首 %.4f → 末 %.4f; val best %.4f @ step %d (v0.1: 0.1101 @106000); 用时 %.1f min'
             % (train_log[0][1], train_log[-1][1], best_val, best_step, (time.time() - t0) / 60))
    R.append('【产出】%s / %s / %s' % (MODEL_PT_V02, LOSS_CSV_V02, LOSS_PNG_V02))
    with open(TRAIN_REP_V02, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n\n' + '\n'.join(R) + '\n')
    log('\n'.join(R))
    log('train 完成: best val %.4f @ step %d' % (best_val, best_step))

# =====================================================================
# eval: v0.1 同协议 + WSD 框架支撑分层对照 (修复效果直读)
# =====================================================================
def w1d(a, b):
    from scipy.stats import wasserstein_distance
    return wasserstein_distance(a, b)

def eval():
    t0 = time.time()
    payload = torch.load(DS_PT_V02, weights_only=False)
    meta, qts = payload['meta'], payload['qts']
    pop_max = np.array([meta['pop_max'][c] for c in w3.COLS])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    diffusion, n_par, _ = build_diffusion_v01(device)
    ck = torch.load(MODEL_PT_V02, map_location=device, weights_only=False)
    diffusion.load_state_dict(ck['state_dict']); diffusion.eval()
    log('==================== WP3/T3.2 DDPM v0.2 eval ====================')
    log('模型加载: best-val @ step %d (val %.4f), cond_scale=%.2f' % (ck['step'], ck['val_loss'], w3.COND_SCALE))

    X_raw = payload['syn_raw']; Z = payload['zeo_feat'].to(device); O = payload['osda_feat'].to(device)
    tr_idx = payload['train_idx'].numpy(); te_idx = payload['test_idx'].numpy()
    zeo_key = payload['zeo_key']; osda_key = payload['osda_key']
    R = []
    V01_STD = {'K/T': 1.78, 'cryst_time': 3.67}          # v0.1 eval 登记读数

    # ---------- 1) 采样自检 (v0 同协议) ----------
    rng = np.random.RandomState(42)
    sel = rng.choice(tr_idx, size=min(2000, len(tr_idx)), replace=False)
    torch.manual_seed(42)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device != 'cpu')):
        syn = diffusion.sample(zeo=Z[sel], osda=O[sel], cond_scale=w3.COND_SCALE,
                               batch_size=len(sel)).squeeze(1).cpu().numpy()
    syn_raw = w3.inv_qt(syn, qts)
    tr_raw = X_raw[tr_idx]
    R.append('一、采样自检 (%d 训练行 × 1 样本; v0.2 = v0.1 + 框架逆频过采样)' % len(sel))
    R.append('%-12s %10s %8s %8s | %-24s | %-24s | %s' % ('特征', 'W1', '越界%', 'std比',
             'train p1/p50/p99', 'syn p1/p50/p99', 'v0.1 std比'))
    oob_flags, col_flags, std_map = [], [], {}
    for i, c in enumerate(w3.COLS):
        a, b = tr_raw[:, i], syn_raw[:, i]
        w1 = w1d(a, b)
        oob = float(((b < a.min()) | (b > a.max())).mean() * 100)
        sr = float(np.std(b) / (np.std(a) + 1e-12))
        std_map[c] = sr
        qa = np.quantile(a, [0.01, 0.5, 0.99]); qb = np.quantile(b, [0.01, 0.5, 0.99])
        R.append('%-12s %10.4f %8.2f %8.3f | %7.3g/%7.3g/%7.3g | %7.3g/%7.3g/%7.3g | %s'
                 % (c, w1, oob, sr, *qa, *qb, ('%.2f' % V01_STD[c]) if c in V01_STD else '—'))
        if oob > 5: oob_flags.append((c, oob))
        if sr < 0.5 or sr > 2: col_flags.append((c, sr))
    R.append('越界判定 (>5%%): %s; 塌缩/发散判定 (std 比 <0.5 或 >2): %s'
             % ('无' if not oob_flags else str(oob_flags), '无' if not col_flags else str(col_flags)))

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
    fig.suptitle('WP3/T3.2 DDPM v0.2 sampling self-check (framework oversampling)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(DIST_PNG_V02, dpi=160); plt.close(fig)
    log('分布自检 PNG: %s' % DIST_PNG_V02)

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
    w.to_csv(WSD_CSV_V02, index=False)

    # 分层对照 (框架支撑 tier, vs v0.1 同口径)
    sup = pd.Series(zeo_key[tr_idx]).value_counts().to_dict()
    for df in (w,):
        df['sup'] = [sup.get(z, 0) for z in df['zeo']]
        df['tier'] = np.where(df['sup'] < 5, 'few(<5)', np.where(df['sup'] < 50, 'mid(5-49)', 'rich(≥50)'))
    v01_csv = os.path.join(BASE, 'data', 'wp3_ddpm_v01_wsd_systems.csv')
    tier_v01 = None
    if os.path.exists(v01_csv):
        w1 = pd.read_csv(v01_csv)
        w1['sup'] = [sup.get(z, 0) for z in w1['zeo']]
        w1['tier'] = np.where(w1['sup'] < 5, 'few(<5)', np.where(w1['sup'] < 50, 'mid(5-49)', 'rich(≥50)'))
        tier_v01 = w1.groupby('tier')['WSD'].agg(['count', 'mean'])

    R.append('')
    R.append('二、同体系 WSD 对照 (25 样本/行, cond_scale 0.75, 官方协议镜像)')
    R.append('体系数: %d (明细 %s)' % (len(w), os.path.basename(WSD_CSV_V02)))
    R.append('WSD mean ± std: %.4f ± %.4f  (v0.1: 0.4064 ± 0.4089; v0: 0.4142; 官方锚 0.4237)'
             % (w['WSD'].mean(), w['WSD'].std()))
    R.append('')
    R.append('三、WSD 框架支撑分层对照 (修复效果直读; v0.1 同口径重算)')
    R.append('%-12s %6s | %10s %10s | %s' % ('tier', 'n_sys', 'v0.1 WSD', 'v0.2 WSD', 'Δ'))
    tier_v02 = w.groupby('tier')['WSD'].agg(['count', 'mean'])
    for t in ['few(<5)', 'mid(5-49)', 'rich(≥50)']:
        if tier_v01 is not None and t in tier_v01.index and t in tier_v02.index:
            a, b = float(tier_v01.loc[t, 'mean']), float(tier_v02.loc[t, 'mean'])
            R.append('%-12s %6d | %10.4f %10.4f | %+.4f' % (t, int(tier_v02.loc[t, 'count']), a, b, b - a))
    n_zero = int((w[w['tier'] == 'few(<5)']['sup'] == 0).sum())
    R.append('登记: few 层含 %d 个框架支撑=0 体系 (过采样不可及, 纯描述子外推 — 结构性天花板)' % n_zero)

    R.append('')
    R.append('四、结论')
    R.append('1) 低支撑修复: few/mid 层 WSD vs v0.1 见分层表 (目标: 双层改善且 rich 不塌)')
    R.append('2) std 比: K/T %.2f (v0.1 1.78) / cryst_time %.2f (v0.1 3.67, v0.3 EMA 对症)' % (std_map['K/T'], std_map['cryst_time']))
    R.append('3) 采样产物 %s 供 WP5 正式读数' % os.path.basename(PRED_PARQUET))
    R.append('4) 数据版本 zeobench_db_v1_8; eval 用时 %.1f min' % ((time.time() - t0) / 60))
    with open(EVAL_REP_V02, 'w', encoding='utf-8') as f:
        f.write('\n'.join(R) + '\n')
    with open(MANIFEST_V02, encoding='utf-8') as f:
        manifest = json.load(f)
    manifest['eval_v02'] = {'wsd_systems': int(len(w)), 'wsd_mean': float(w['WSD'].mean()),
                            'wsd_std': float(w['WSD'].std()),
                            'wsd_tier': {t: {'n': int(tier_v02.loc[t, 'count']),
                                             'mean': float(tier_v02.loc[t, 'mean'])} for t in tier_v02.index},
                            'best_val_step': int(ck['step']),
                            'std_ratio': {c: std_map[c] for c in ('K/T', 'cryst_time')},
                            'n_pred_rows': int(len(syn_pred)),
                            'eval_duration_min': round((time.time() - t0) / 60, 1)}
    with open(MANIFEST_V02, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log('eval 完成: 报告 %s' % EVAL_REP_V02)

# =====================================================================
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['train', 'eval', 'all'])
    a = ap.parse_args()
    if a.stage in ('train', 'all'):
        train()
    if a.stage in ('eval', 'all'):
        eval()
