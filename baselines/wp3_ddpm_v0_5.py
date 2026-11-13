# -*- coding: utf-8 -*-
"""
WP3/T3.2 DDPM v0.5 低支撑体系定向过采样 (P0#3 生成器 v1 COV-P 差距处置, v0.5 实跑)
================================================================================
背景 (FASE_执行计划.md P0#3 / G3 判定报告 §3.1 §6.3 §6.4): G3 中期线 COV-P GAP, 主差距
  由 27 个低支撑体系 (few/mid 层) 承载; v0.2 框架级逆频过采样 (α=0.5) 是唯一结构性改善
  (few 层 +6.6pp) 但被 mid 层 -3.7pp 抵消 → 净效应噪声级。
修正 (v0.5 新变量 = 体系级定向过采样, 叠加 v0.2 框架级遗产):
  w_row = min(w_fw × w_sys, CAP=7.071)
    w_fw  = (S_REF/s_fw)^ALPHA (s_fw<50) / 1 (v0.2 框架级遗产, 未改)
    w_sys = (S_SYS_REF/s_sys)^BETA (s_sys<5, 仅 few/mid 层框架) / 1 — 体系 = (zeo_code, osda_key) 对;
            直接定向「低支撑体系」而非仅「低支撑框架」(v0.2 mid 层欠定向的对症修正)
  温和档纪律: β=0.25 → w_sys max 1.50; cap = v0.2 框架级上界 7.071 → few 层不超出 v0.2 (尊重
  v0.2「α=0.75/1.0 会记忆化, 温和档规避」结论); rich 层权重不受 w_sys 扰动。
  机理: 低支撑体系的生成质量受「少样本条件方向欠训练」限制; 框架级仅按 zeo 支撑加权,
  对同框架下更稀有的体系欠定向 → 体系级二次加权直接补偿 (mid 层尤其受益)。
协议纪律: v0.3 遗产保持 (R1 qt 全量 / R2 per-col 加权 / v0.2 框架逆频过采样 / EMA 0.999);
  与 v0.3 唯一自变量 = 体系级过采样因子 w_sys。不含 Si/Al 条件通道 (保持 G3 null 档谱系可比);
  数据版本 zeobench_db_v1_8。
eval: v0.3 同协议 (采样自检 + WSD 体系级 + 框架支撑分层对照 vs v0.3) + 采样产物落盘 (WP5 复用)。
用法: python wp3_ddpm_v0_5.py train|eval|all
产出: Data/wp3_ddpm_v05_{model.pt,loss.csv,manifest.json,train_report.txt,eval_report.txt,
        wsd_systems.csv,syn_pred.parquet} (gitignored)
      Documentation/wp3_ddpm_v05_{loss,distcheck}.png (入 git)
"""
import os, sys, json, time, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZG = os.path.join(BASE, 'vendor', 'zeosyn_gen')
os.chdir(ZG); sys.path.insert(0, ZG); sys.path.insert(0, BASE)

import torch

import wp3_ddpm_v0 as w3
import wp3_ddpm_v0_1 as v01
import wp3_ddpm_v0_2 as v02                      # framework_weights / ALPHA / S_REF 复用
import wp3_ddpm_v0_3 as v03                      # EMA 工具复用
from wp3_ddpm_v0_1 import build_diffusion_v01
from wp3_ddpm_v0_2 import framework_weights, ALPHA, S_REF
from wp3_ddpm_v0_3 import EMA_DECAY, ema_state_dict, val_with_ema

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------- 路径 (v0.5 专属) ----------------
DS_PT_V05     = v01.DS_PT_V01                     # dataset 逐位复用 (含 v0.1 qt 全量)
MODEL_PT_V05  = os.path.join(BASE, 'weights', 'wp3_ddpm_v05_model.pt')
LOSS_CSV_V05  = os.path.join(BASE, 'data', 'wp3_ddpm_v05_loss.csv')
MANIFEST_V05  = os.path.join(BASE, 'data', 'wp3_ddpm_v05_manifest.json')
TRAIN_REP_V05 = os.path.join(BASE, 'data', 'wp3_ddpm_v05_train_report.txt')
EVAL_REP_V05  = os.path.join(BASE, 'data', 'wp3_ddpm_v05_eval_report.txt')
WSD_CSV_V05   = os.path.join(BASE, 'data', 'wp3_ddpm_v05_wsd_systems.csv')
LOSS_PNG_V05  = os.path.join(BASE, 'Documentation', 'wp3_ddpm_v05_loss.png')
DIST_PNG_V05  = os.path.join(BASE, 'Documentation', 'wp3_ddpm_v05_distcheck.png')
PRED_PARQUET  = os.path.join(BASE, 'data', 'wp3_ddpm_v05_syn_pred.parquet')
S_SYS_REF = 5                                    # 体系支撑参照 (与 few<5 分层口径一致)
BETA      = 0.25                                 # 体系级逆频指数 (温和档: w_sys max 1.50; 尊重 v0.2 "温和档规避记忆化" 结论)
CAP       = S_REF ** ALPHA                       # 复合权重上限 = v0.2 框架级上界 (7.071) — few 层不超出 v0.2 温和档

L = []
def log(s=''):
    L.append(s); print(s, flush=True)

def sys_key(zeo_key, osda_key):
    return np.array(['%s||%s' % (a, b) for a, b in zip(zeo_key, osda_key)])

# =====================================================================
# 定向复合权重: 框架级 (v0.2) × 体系级 (v0.5 新变量)
# =====================================================================
def targeted_weights(zeo_key, osda_key, tr_idx):
    """w_row = min(w_fw × w_sys, CAP)。
    w_fw  = (S_REF/s_fw)^ALPHA (s_fw<50)/1 (v0.2 框架级逆频, 未改);
    w_sys = (S_SYS_REF/s_sys)^BETA (s_sys<S_SYS_REF)/1 — 体系级定向 (v0.5 新变量),
            仅作用于 few/mid 层框架行 (s_fw<S_REF), rich 层不受扰动。
    返回 (train 复合行权, 框架级行权, 框架支撑 dict, 体系支撑 dict)。"""
    w_fw, sup_fw = framework_weights(zeo_key, tr_idx)
    s_fw = np.array([sup_fw.get(z, 0) for z in zeo_key[tr_idx]], float)
    sk = sys_key(zeo_key[tr_idx], osda_key[tr_idx])
    sup_sys = pd.Series(sk).value_counts().to_dict()
    s_rows = np.array([sup_sys[k] for k in sk], float)
    w_sys = np.where((s_fw < S_REF) & (s_rows < S_SYS_REF),
                     (S_SYS_REF / np.maximum(s_rows, 1)) ** BETA, 1.0)
    w = np.minimum(w_fw * w_sys, CAP)
    return w, w_fw, sup_fw, sup_sys

# =====================================================================
# train: v0.5 = v0.3 + 体系级定向过采样 (唯一自变量 = w_sys)
# =====================================================================
def train():
    t0 = time.time()
    payload = torch.load(DS_PT_V05, weights_only=False)
    meta = payload['meta']
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0); np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    diffusion, n_par, cw = build_diffusion_v01(device)
    log('==================== WP3/T3.2 DDPM v0.5 train (低支撑体系定向过采样) ====================')
    log('模型: MLPUnet %.2fM + WeightedColDiffusion1D (v0.3 遗产: qt 全量 / per-col 加权 / 框架逆频 / EMA %.3f)'
        % (n_par / 1e6, EMA_DECAY))
    log('超参: 与 v0.3 逐位一致 — 唯一自变量 = 体系级定向过采样 (w_sys, S_SYS_REF=%d, β=%.2f, cap=%.1f)'
        % (S_SYS_REF, BETA, CAP))

    X = payload['syn_qt'].to(device); Z = payload['zeo_feat'].to(device); O = payload['osda_feat'].to(device)
    tr = payload['train_idx'].to(device); va = payload['val_idx'].to(device)
    zeo_key = np.asarray(payload['zeo_key']); osda_key = np.asarray(payload['osda_key'])
    tr_np = payload['train_idx'].numpy()
    Xtr, Ztr, Otr = X[tr], Z[tr], O[tr]
    Xva, Zva, Ova = X[va], Z[va], O[va]
    n_tr = Xtr.shape[0]

    w_np, w_fw_np, sup_fw, sup_sys = targeted_weights(zeo_key, osda_key, tr_np)
    W = torch.from_numpy(w_np).float().to(device)
    s_fw_rows = np.array([sup_fw.get(z, 0) for z in zeo_key[tr_np]])
    low_fw = s_fw_rows < S_REF
    n_low = int(low_fw.sum())
    sk = sys_key(zeo_key[tr_np], osda_key[tr_np])
    s_sys_rows = np.array([sup_sys[k] for k in sk], float)
    n_low_sys = int((low_fw & (s_sys_rows < S_SYS_REF)).sum())
    n_cap = int((w_np >= CAP).sum())
    share_raw = 100 * n_low / n_tr
    share_v02 = 100 * w_fw_np[w_fw_np > 1].sum() / w_fw_np.sum()      # 纯框架级 (可比 v0.2 基线)
    share_v05 = 100 * w_np[w_np > 1].sum() / w_np.sum()              # 复合 (含体系级)
    log('数据: train %d / val %d | 低支撑框架 (<%d 行) 覆盖 %d train 行 (原始占比 %.1f%%)'
        % (n_tr, len(va), S_REF, n_low, share_raw))
    log('批量份额: 低支撑框架 %.1f%% → 复合 %.1f%% (纯框架级 %.1f%%, 对齐 v0.2 基线; 体系级净增 %+.1fpp)'
        % (share_raw, share_v05, share_v02, share_v05 - share_v02))
    log('体系级定向 (v0.5 新变量): 体系数 %d | 低支撑体系 (<%d 行, few/mid 层) 覆盖 %d train 行 | 权重触顶 %d 行'
        % (len(sup_sys), S_SYS_REF, n_low_sys, n_cap))
    log('权重范围: w_fw (%.2f-%.2f) × w_sys (1.00-%.2f) → 复合 (%.2f-%.2f, cap %.1f)'
        % (w_fw_np.min(), w_fw_np.max(), (S_SYS_REF / 1) ** BETA, w_np.min(), w_np.max(), CAP))

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
                            'sample_weights': {'alpha': ALPHA, 's_ref': S_REF,
                                               's_sys_ref': S_SYS_REF, 'beta': BETA, 'cap': CAP,
                                               'scheme': 'framework x system targeted'},
                            'meta': {k: meta[k] for k in ('data_version', 'n_pop', 'split')}},
                           MODEL_PT_V05)
            pd.DataFrame({'step': [s for s, _ in train_log], 'train_loss': [l for _, l in train_log]}
                         ).merge(pd.DataFrame({'step': [s for s, _ in val_log_raw], 'val_loss_raw': [l for _, l in val_log_raw]}),
                                 on='step', how='outer'
                                 ).merge(pd.DataFrame({'step': [s for s, _ in val_log_ema], 'val_loss_ema': [l for _, l in val_log_ema]}),
                                         on='step', how='outer').sort_values('step').to_csv(LOSS_CSV_V05, index=False)
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
    ax.set_title('WP3/T3.2 DDPM v0.5 (framework x system targeted oversampling, zeobench_db_v1_8)')
    fig.tight_layout(); fig.savefig(LOSS_PNG_V05, dpi=160); plt.close(fig)

    # ---- manifest ----
    manifest = {
        'task': 'WP3/T3.2 DDPM v0.5 低支撑体系定向过采样 (P0#3 COV-P 差距处置)',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'data_version': 'zeobench_db_v1_8', 'dataset_md5': w3.md5(DS_PT_V05),
        'data_md5': w3.md5(w3.DB_PARQUET),
        'code_md5': {'wp3_ddpm_v0_5.py': w3.md5(os.path.abspath(__file__)),
                     'wp3_ddpm_v0_3.py': w3.md5(v03.__file__),
                     'wp3_ddpm_v0_2.py': w3.md5(v02.__file__),
                     'wp3_ddpm_v0_1.py': w3.md5(v01.__file__),
                     'wp3_ddpm_v0.py': w3.md5(os.path.join(BASE, 'wp3_ddpm_v0.py'))},
        'git_commit': w3.git_head(),
        'changes': {'v05': '体系级定向过采样 (新变量): w_row = min(w_fw × w_sys, %.1f); '
                           'w_sys = (S_SYS_REF/s_sys)^β, S_SYS_REF=%d, β=%.2f — 直接定向低支撑体系 '
                           '(v0.2 仅框架级, mid 层欠定向的对症修正); 无 Si/Al 条件通道 (G3 null 档谱系可比)'
                           % (CAP, S_SYS_REF, BETA),
                    'inherited': 'v0.3 EMA + v0.2 框架逆频过采样 + v0.1 R1/R2'},
        'model': {'denoiser': 'MLPUnet (v0 同款)', 'params': int(n_par),
                  'interface': 'WeightedColDiffusion1D',
                  'col_weights': {c: float(cw[i]) for i, c in enumerate(w3.COLS)}},
        'hyperparams': {'lr': 1e-4, 'betas': [0.9, 0.99], 'batch': w3.BATCH, 'steps': w3.STEPS,
                        'cond_drop_prob': 0.1, 'ema_decay': EMA_DECAY,
                        'sample_weights': {'alpha': ALPHA, 's_ref': S_REF, 's_sys_ref': S_SYS_REF,
                                           'beta': BETA, 'cap': CAP, 'scheme': 'framework x system targeted'},
                        'precision': 'TF32 + bf16 autocast', 'selection': 'best-EMA-val checkpoint'},
        'best_val': {'step': best_step, 'val_loss_ema': best_val,
                     'val_loss_raw_at_best': val_log_raw[-1][1] if val_log_raw else None},
        'train_duration_min': round((time.time() - t0) / 60, 1),
    }
    with open(MANIFEST_V05, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    R = []
    R.append('WP3/T3.2 DDPM v0.5 训练报告 — %s' % time.strftime('%Y-%m-%d %H:%M'))
    R.append('=' * 78)
    R.append('【修正项】体系级定向过采样 (新变量): w_row = min(w_fw × w_sys, %.1f);' % CAP)
    R.append('        w_sys = (S_SYS_REF/s_sys)^β (S_SYS_REF=%d, β=%.2f); v0.3 遗产保持' % (S_SYS_REF, BETA))
    R.append('【收敛】train 首 %.4f → 末 %.4f; val EMA best %.4f @ step %d (v0.3: 0.1080 @%d); 用时 %.1f min'
             % (train_log[0][1], train_log[-1][1], best_val, best_step, 106000, (time.time() - t0) / 60))
    R.append('【产出】%s / %s / %s' % (MODEL_PT_V05, LOSS_CSV_V05, LOSS_PNG_V05))
    with open(TRAIN_REP_V05, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n\n' + '\n'.join(R) + '\n')
    log('\n'.join(R))
    log('train 完成: best EMA-val %.4f @ step %d' % (best_val, best_step))

# =====================================================================
# eval: v0.3 同协议 + 框架支撑分层对照 (定向效应直读)
# =====================================================================
def w1d(a, b):
    from scipy.stats import wasserstein_distance
    return wasserstein_distance(a, b)

def eval():
    t0 = time.time()
    payload = torch.load(DS_PT_V05, weights_only=False)
    meta, qts = payload['meta'], payload['qts']
    pop_max = np.array([meta['pop_max'][c] for c in w3.COLS])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    diffusion, n_par, _ = build_diffusion_v01(device)
    ck = torch.load(MODEL_PT_V05, map_location=device, weights_only=False)
    diffusion.load_state_dict(ck['state_dict']); diffusion.eval()
    log('==================== WP3/T3.2 DDPM v0.5 eval ====================')
    log('模型加载: best-EMA-val @ step %d (EMA val %.4f / raw %.4f), cond_scale=%.2f'
        % (ck['step'], ck['val_loss'], ck.get('val_loss_raw', float('nan')), w3.COND_SCALE))

    prev_std = {'K/T': 1.27, 'cryst_time': 2.70}          # v0.3 eval 登记读数
    v03_mf = os.path.join(BASE, 'data', 'wp3_ddpm_v03_manifest.json')
    if os.path.exists(v03_mf):
        with open(v03_mf, encoding='utf-8') as f:
            prev_std = json.load(f).get('eval_v03', {}).get('std_ratio', prev_std)

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
    R.append('一、采样自检 (%d 训练行 × 1 样本; v0.5 = v0.3 + 体系级定向过采样)' % len(sel))
    R.append('%-12s %10s %8s %8s | %-24s | %-24s | %s' % ('特征', 'W1', '越界%', 'std比',
             'train p1/p50/p99', 'syn p1/p50/p99', 'v0.3 std比'))
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
    fig.suptitle('WP3/T3.2 DDPM v0.5 sampling self-check (framework x system targeted oversampling)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(DIST_PNG_V05, dpi=160); plt.close(fig)
    log('分布自检 PNG: %s' % DIST_PNG_V05)

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
    w.to_csv(WSD_CSV_V05, index=False)

    sup = pd.Series(zeo_key[tr_idx]).value_counts().to_dict()
    w['sup'] = [sup.get(z, 0) for z in w['zeo']]
    w['tier'] = np.where(w['sup'] < 5, 'few(<5)', np.where(w['sup'] < 50, 'mid(5-49)', 'rich(≥50)'))
    tier_v05 = w.groupby('tier')['WSD'].agg(['count', 'mean'])
    tier_prev = None
    prev_csv = os.path.join(BASE, 'data', 'wp3_ddpm_v03_wsd_systems.csv')
    if os.path.exists(prev_csv):
        wp = pd.read_csv(prev_csv)
        wp['sup'] = [sup.get(z, 0) for z in wp['zeo']]
        wp['tier'] = np.where(wp['sup'] < 5, 'few(<5)', np.where(wp['sup'] < 50, 'mid(5-49)', 'rich(≥50)'))
        tier_prev = wp.groupby('tier')['WSD'].agg(['count', 'mean'])

    R.append('')
    R.append('二、同体系 WSD 对照 (25 样本/行, cond_scale %.2f, 官方协议镜像)' % w3.COND_SCALE)
    R.append('体系数: %d (明细 %s)' % (len(w), os.path.basename(WSD_CSV_V05)))
    R.append('WSD mean ± std: %.4f ± %.4f  (v0.3: 0.3931; v0.1: 0.4064; v0: 0.4142; 官方锚 0.4237)'
             % (w['WSD'].mean(), w['WSD'].std()))
    R.append('')
    R.append('三、WSD 框架支撑分层对照 (定向效应直读)')
    R.append('%-12s %6s | %10s %10s | %s' % ('tier', 'n_sys', 'v0.3 WSD', 'v0.5 WSD', 'Δ'))
    for t in ['few(<5)', 'mid(5-49)', 'rich(≥50)']:
        if tier_prev is not None and t in tier_prev.index and t in tier_v05.index:
            a_, b_ = float(tier_prev.loc[t, 'mean']), float(tier_v05.loc[t, 'mean'])
            R.append('%-12s %6d | %10.4f %10.4f | %+.4f' % (t, int(tier_v05.loc[t, 'count']), a_, b_, b_ - a_))

    R.append('')
    R.append('四、结论')
    R.append('1) 定向效应: few/mid 层 WSD vs v0.3 见分层表 (目标: 双层改善且 rich 不塌)')
    R.append('2) std 比: K/T %.2f / cryst_time %.2f (v0.3 遗产对照)' % (std_map['K/T'], std_map['cryst_time']))
    R.append('3) 采样产物 %s 供 WP5 G3 正式读数 (v0.5 COV-P 判定)' % os.path.basename(PRED_PARQUET))
    R.append('4) 数据版本 zeobench_db_v1_8; eval 用时 %.1f min' % ((time.time() - t0) / 60))
    with open(EVAL_REP_V05, 'w', encoding='utf-8') as f:
        f.write('\n'.join(R) + '\n')
    with open(MANIFEST_V05, encoding='utf-8') as f:
        manifest = json.load(f)
    manifest['eval_v05'] = {'wsd_systems': int(len(w)), 'wsd_mean': float(w['WSD'].mean()),
                            'wsd_std': float(w['WSD'].std()),
                            'wsd_tier': {t: {'n': int(tier_v05.loc[t, 'count']),
                                             'mean': float(tier_v05.loc[t, 'mean'])} for t in tier_v05.index},
                            'best_val_step': int(ck['step']),
                            'std_ratio': {c: std_map[c] for c in ('K/T', 'cryst_time')},
                            'n_pred_rows': int(len(syn_pred)),
                            'eval_duration_min': round((time.time() - t0) / 60, 1)}
    with open(MANIFEST_V05, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log('eval 完成: 报告 %s' % EVAL_REP_V05)

# =====================================================================
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['train', 'eval', 'all'])
    a = ap.parse_args()
    if a.stage in ('train', 'all'):
        train()
    if a.stage in ('eval', 'all'):
        eval()
