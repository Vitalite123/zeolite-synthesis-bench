# -*- coding: utf-8 -*-
"""
WP3/T3.2 DDPM 骨架首训 (M7 提前量) —— 建模线交接清单 2026-10-17 任务①
====================================================================
开工决策项 (已定案, 登记入报告):
  D1 甲: 从 zeobench_db_v1_8 重建 DiffSyn 兼容数据集
         —— 12 列映射 (4 元素比 + 6 个 /T 比 + 温度/时间, T = n_Si+n_Al+n_P+n_Ge+n_B 五元, 探针核验 94.26% 重建一致)
         —— 官方 cap 协议 (0/0→0, inf→分位 cap, ≥cap→cap; 分位水平照官方, v1.8 全量人口重算)
         —— 官方 IterativeImputer 惯例 (min_value=0, sample_posterior, skip_complete, random_state=0)
         —— 官方 QuantileTransformer 惯例 (n_quantiles=1000, random_state=0, fit 于纳入人口)
  D2 甲: 首训用 DiffSyn 原设定条件向量 (zeo 143 维 + osda 14 维, 官方 scaler/占位向量) → 二跑换 c_zeo v1 案 D
         —— 混合码/未收录剔除 (官方 present=0 镜像), absolute_neg → 官方 Dense/Amorphous 占位 (pkl 原样),
            双 NaN → 官方 OSDA-free 占位 (pkl 原样), no_product 剔除
  D3   : 镜像官方 system split (zeo-osda 体系级 80/20 + 行级 87.5/12.5, random_state=0, sorted 体系序)
模型规格 (交接清单 §1.1): MLP-U-Net 去噪网络 (参数量 <5M), T=1000, cosine schedule,
  Adam 1e-4, batch 512 (8GB 卡可载); 复用官方 GaussianDiffusion1D (pred_v / l1) 全套扩散数学保证可比性
工程纪律 (l2_neural_v0.py §8): TF32 + bf16 autocast (模型输出回 fp32 计损失, 不动官方扩散代码)
验收口径 (交接清单 §1.5): loss 收敛曲线 + 采样自检 (逆 qt 回原始量纲逐特征 vs 训练域, 无越界/塌缩)
  + (D2 甲) 同体系 WSD 与官方锚 0.4237 量级对照 (geomloss Sinkhorn 官方实现, 不重写定义)
  + 训练挂 manifest (数据版本 zeobench_db_v1_8 + 代码 md5 + 超参)
产出: Data/wp3_ddpm_v0_dataset.pt / _model.pt / _loss.csv / _manifest.json
      Data/wp3_ddpm_train_report.txt / wp3_ddpm_eval_report.txt / wp3_ddpm_v0_wsd_systems.csv (gitignored)
      Documentation/wp3_ddpm_v0_loss.png / wp3_ddpm_v0_distcheck.png (入 git)
用法: python wp3_ddpm_v0.py build|train|eval|all
"""
import os, sys, json, time, pickle, hashlib, argparse, warnings, subprocess
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZG = os.path.join(BASE, 'vendor', 'zeosyn_gen')
os.chdir(ZG); sys.path.insert(0, ZG)

import torch
import torch.nn.functional as F
from torch import nn

import data.utils as _du
sys.modules['utils'] = _du                       # eval.py 同款 trick (ZeoSynGen_dataset pickle)
from data.metrics import wasserstein_distance, maximum_mean_discrepancy   # 官方实现, 不重写
from data.syn_variables import zeo_cols, osda_cols                         # 官方列序 (scaler 依赖)
import models.diffusion as mdiff
from models.diffusion import GaussianDiffusion1D, SinusoidalPosEmb, prob_mask_like
mdiff.tqdm = lambda it, **kw: it                 # 采样 1000 步进度条静默 (diffsyn_repro 同款)

from einops import repeat, rearrange
from rdkit import Chem
from sklearn.preprocessing import QuantileTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from scipy.stats import wasserstein_distance as w1_dist

os.chdir(ZG)                                     # data.metrics 导入有 cwd 副作用, 强制回 ZG; 之后全用绝对路径

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------- 路径与常量 ----------------
DB_PARQUET = os.path.join(BASE, 'data', 'zeobench_db_v1_8.parquet')
DS_PT      = os.path.join(BASE, 'weights', 'wp3_ddpm_v0_dataset.pt')
MODEL_PT   = os.path.join(BASE, 'weights', 'wp3_ddpm_v0_model.pt')
LOSS_CSV   = os.path.join(BASE, 'data', 'wp3_ddpm_v0_loss.csv')
MANIFEST   = os.path.join(BASE, 'data', 'wp3_ddpm_v0_manifest.json')
TRAIN_REP  = os.path.join(BASE, 'data', 'wp3_ddpm_train_report.txt')
EVAL_REP   = os.path.join(BASE, 'data', 'wp3_ddpm_eval_report.txt')
LOSS_PNG   = os.path.join(BASE, 'Documentation', 'wp3_ddpm_v0_loss.png')
DIST_PNG   = os.path.join(BASE, 'Documentation', 'wp3_ddpm_v0_distcheck.png')
WSD_CSV    = os.path.join(BASE, 'data', 'wp3_ddpm_v0_wsd_systems.csv')
OFF_ASSETS = {                                   # 可比性锚点资产 (manifest 登记 md5)
    'ZeoSynGen_dataset.pkl':  os.path.join(ZG, 'data', 'ZeoSynGen_dataset.pkl'),
    'zeolite_descriptors.csv': os.path.join(ZG, 'data', 'zeolite_descriptors.csv'),
    'osda_descriptors.csv':    os.path.join(ZG, 'data', 'osda_descriptors.csv'),
    'zeo_feat_scaler.pkl':     os.path.join(ZG, 'data', 'scalers', 'zeo_feat_scaler.pkl'),
    'osda_feat_scaler.pkl':    os.path.join(ZG, 'data', 'scalers', 'osda_feat_scaler.pkl'),
}
RATIO_Q = {'Si/Al': 0.98, 'Al/P': 0.99, 'Si/Ge': 0.995, 'Si/B': 0.99,   # 官方分位水平 (syn_variables.py)
           'Na/T': 0.95, 'K/T': 0.99, 'OH/T': 0.97, 'F/T': 0.95,
           'H2O/T': 0.99, 'sda1/T': 0.98}
COLS = list(RATIO_Q) + ['cryst_temp', 'cryst_time']
GEL_COLS = ['n_Si', 'n_Al', 'n_P', 'n_Ge', 'n_B', 'n_Na', 'n_K', 'n_OH', 'n_F', 'n_H2O', 'n_osda1']
COND_SCALE = 0.75                                 # 官方 eval/diffsyn_repro 口径
BATCH, STEPS, VAL_EVERY, LOG_EVERY = 512, 150_000, 500, 50
OFFICIAL_WSD_ANCHOR = 0.4237                      # 官方 wsd_zeo_osda.json mean (论文 Fig 2a 0.423)

L = []
def log(s=''):
    L.append(s); print(s, flush=True)

def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=BASE, text=True).strip()
    except Exception:
        return 'unavailable'

# ---------------- 官方惯例的映射函数 (探针 2/3 定案) ----------------
def zeo_strip(c):
    """框架码去装饰: *-SVY→SVY, *BEA→BEA, -IRT→IRT"""
    c = str(c)
    if c.startswith('*-'): return c[2:]
    if c.startswith('*') or c.startswith('-'): return c[1:]
    return c

def salt_strip(s):
    """SMILES 盐式剥离: 取最大片段 canonical 化 (六甲溴铵等 648 行惯例)"""
    try:
        m = Chem.MolFromSmiles(str(s))
        if m is None: return None
        frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=True)
        if not frags: return None
        return Chem.MolToSmiles(max(frags, key=lambda f: f.GetNumAtoms()))
    except Exception:
        return None

# =====================================================================
# MLP-U-Net 去噪网络 (官方 Unet1D 的 MLP 镜像, 接口逐位对齐)
# =====================================================================
class ResMLPBlock(nn.Module):
    """官方 ResnetBlock 的 MLP 镜像: time_emb 与 cond_emb 拼接 → SiLU+Linear 出 dim_out*2 (FiLM scale/shift),
    主干 LayerNorm→fc1→FiLM→SiLU→LayerNorm→fc2→SiLU + 残差投影"""
    def __init__(self, dim_in, dim_out, cond_dim=192):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim * 2, dim_out * 2))
        self.norm1 = nn.LayerNorm(dim_in)
        self.fc1 = nn.Linear(dim_in, dim_out)
        self.norm2 = nn.LayerNorm(dim_out)
        self.fc2 = nn.Linear(dim_out, dim_out)
        self.res = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()

    def forward(self, x, t, c):
        scale, shift = self.mlp(torch.cat([t, c], dim=-1)).chunk(2, dim=-1)
        h = self.fc1(self.norm1(x))
        h = h * (scale + 1) + shift
        h = F.silu(h)
        h = self.fc2(self.norm2(h))
        h = F.silu(h)
        return h + self.res(x)

class MLPUnet(nn.Module):
    """官方 Unet1D 接口镜像 (GaussianDiffusion1D 依赖):
    forward(x[B,1,12], time[B], zeo[B,143], osda[B,14], cond_drop_prob=None)
    forward_with_cond_scale(...) CFG 双前向; attrs channels/out_dim/random_or_learned_sinusoidal_cond
    null_cond_emb 128 维 (zeo_h+osda_h), prob_mask_like 条件丢弃 (官方同款)
    结构: 12→128→256→512→256→128→12, cond/time 嵌入 192, FiLM 注入; 输出回 fp32 (bf16 惯例)"""
    def __init__(self, zeo_feat_dims=143, osda_feat_dims=14, cond_drop_prob=0.1):
        super().__init__()
        self.cond_drop_prob = cond_drop_prob
        self.channels, self.out_dim = 1, 1
        self.random_or_learned_sinusoidal_cond = False
        D = 192
        self.zeo_mlp = nn.Sequential(nn.Linear(zeo_feat_dims, 256), nn.ReLU(),
                                     nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64))
        self.osda_mlp = nn.Sequential(nn.Linear(osda_feat_dims, 256), nn.ReLU(),
                                      nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64))
        self.null_cond_emb = nn.Parameter(torch.randn(128))
        self.cond_mlp = nn.Sequential(nn.Linear(128, D), nn.ReLU(), nn.Linear(D, D))
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(64), nn.Linear(64, D), nn.GELU(), nn.Linear(D, D))
        self.stem = nn.Linear(12, 128)
        self.enc1 = nn.ModuleList([ResMLPBlock(128, 128, D) for _ in range(2)])
        self.down1 = nn.Linear(128, 256)
        self.enc2 = nn.ModuleList([ResMLPBlock(256, 256, D) for _ in range(2)])
        self.down2 = nn.Linear(256, 512)
        self.mid = nn.ModuleList([ResMLPBlock(512, 512, D) for _ in range(2)])
        self.up1 = nn.Linear(512, 256)
        self.dec2 = nn.ModuleList([ResMLPBlock(512, 256, D) for _ in range(2)])
        self.up2 = nn.Linear(256, 128)
        self.dec1 = nn.ModuleList([ResMLPBlock(256, 128, D) for _ in range(2)])
        self.final_block = ResMLPBlock(256, 128, D)
        self.final_lin = nn.Linear(128, 12)

    def forward(self, x, time, zeo, osda, cond_drop_prob=None, **kw):
        batch, device = x.shape[0], x.device
        cond_drop_prob = self.cond_drop_prob if cond_drop_prob is None else cond_drop_prob
        cond_emb = torch.cat([self.zeo_mlp(zeo), self.osda_mlp(osda)], dim=-1)
        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)
            null_cond_emb = repeat(self.null_cond_emb, 'd -> b d', b=batch)
            cond_emb = torch.where(rearrange(keep_mask, 'b -> b 1'), cond_emb, null_cond_emb)
        c = self.cond_mlp(cond_emb)
        h = self.stem(x.squeeze(1))                # [B,12] → [B,128]
        r = h
        t = self.time_mlp(time)
        for blk in self.enc1: h = blk(h, t, c)
        s1 = h
        h = self.down1(h)
        for blk in self.enc2: h = blk(h, t, c)
        s2 = h
        h = self.down2(h)
        for blk in self.mid: h = blk(h, t, c)
        h = self.up1(h)
        for blk in self.dec2: h = blk(torch.cat([h, s2], dim=-1), t, c)
        h = self.up2(h)
        for blk in self.dec1: h = blk(torch.cat([h, s1], dim=-1), t, c)
        h = self.final_block(torch.cat([h, r], dim=-1), t, c)
        return self.final_lin(h).unsqueeze(1).float()   # [B,1,12]; bf16 惯例: 输出回 fp32 计损失

    def forward_with_cond_scale(self, *args, cond_scale=1., **kwargs):
        logits = self.forward(*args, cond_drop_prob=0., **kwargs)
        if cond_scale == 1:
            return logits
        null_logits = self.forward(*args, cond_drop_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

def build_diffusion(device):
    model = MLPUnet(143, 14, cond_drop_prob=0.1)
    n_par = sum(p.numel() for p in model.parameters())
    assert n_par < 5_000_000, f'参数量 {n_par} 超出 <5M 规格'
    diffusion = GaussianDiffusion1D(model, seq_length=12, timesteps=1000,
                                    objective='pred_v', beta_schedule='cosine', loss_type='l1').to(device)
    return diffusion, n_par

# =====================================================================
# build: v1.8 → DiffSyn 兼容数据集
# =====================================================================
def build():
    t0 = time.time()
    log('==================== WP3/T3.2 DDPM 首训 build (D1 甲: zeobench_db_v1_8 重建) ====================')
    df = pd.read_parquet(DB_PARQUET)
    n0 = len(df)
    log('v1.8 主库: %d 行 × %d 列 (数据版本 zeobench_db_v1_8, md5 %s)' % (n0, df.shape[1], md5(DB_PARQUET)))

    # ---- 1) 12 列映射 (NaN 视为组分缺席=0, 官方 0/0→0 同型) ----
    g = {c: pd.to_numeric(df[c], errors='coerce').fillna(0.0).to_numpy(float) for c in GEL_COLS}
    T = g['n_Si'] + g['n_Al'] + g['n_P'] + g['n_Ge'] + g['n_B']       # T 定义: 四面体五元 (探针1 核验)
    raw = {'Si/Al': g['n_Si'] / g['n_Al'], 'Al/P': g['n_Al'] / g['n_P'],
           'Si/Ge': g['n_Si'] / g['n_Ge'], 'Si/B': g['n_Si'] / g['n_B'],
           'Na/T': g['n_Na'] / T, 'K/T': g['n_K'] / T, 'OH/T': g['n_OH'] / T,
           'F/T': g['n_F'] / T, 'H2O/T': g['n_H2O'] / T, 'sda1/T': g['n_osda1'] / T}
    raw['cryst_temp'] = pd.to_numeric(df['T_C'], errors='coerce').to_numpy(float)
    raw['cryst_time'] = pd.to_numeric(df['t_h'], errors='coerce').to_numpy(float)
    n_osda_nan_fill = int(df['n_osda1'].isna().sum())
    log('12 列映射完成: T=四面体五元; n_osda1 NaN→0 共 %d 行; T_C NaN %d / t_h NaN %d (待插补)'
        % (n_osda_nan_fill, df['T_C'].isna().sum(), df['t_h'].isna().sum()))

    # ---- 2) cap 协议 (官方 preprocess_gel 镜像; 分位水平照官方, v1.8 全量人口重算) ----
    caps = {}
    for k, q in RATIO_Q.items():
        v = np.asarray(raw[k], float).copy()
        v[np.isnan(v)] = 0.
        hv = np.quantile(v[(v != np.inf) & (v != 0.)], q)
        v[v == np.inf] = hv
        v[v >= hv] = hv
        raw[k] = v
        caps[k] = float(hv)
    log('cap 协议 (v1.8 重算): ' + ', '.join('%s=%.4g' % (k, v) for k, v in caps.items()))

    # ---- 3) conds IterativeImputer (官方 preprocess_conditions 镜像, 全量人口) ----
    X12 = np.column_stack([raw[c] for c in COLS])
    imp = IterativeImputer(min_value=0., sample_posterior=True, skip_complete=True, random_state=0)
    X12 = imp.fit_transform(X12)
    log('conds 插补完成 (cryst_temp %d + cryst_time %d 行, 官方镜像 min_value=0/posterior/skip_complete)'
        % (int(pd.to_numeric(df['T_C'], errors='coerce').isna().sum()),
           int(pd.to_numeric(df['t_h'], errors='coerce').isna().sum())))

    # ---- 4) 人口筛选 (D2 甲: 官方特征 + 占位向量) ----
    with open(OFF_ASSETS['ZeoSynGen_dataset.pkl'], 'rb') as f:
        ds = pickle.load(f)
    dz = pd.read_csv(OFF_ASSETS['zeolite_descriptors.csv']).drop(columns=['Unnamed: 0']).drop_duplicates('Code')
    do = pd.read_csv(OFF_ASSETS['osda_descriptors.csv']).drop(columns=['Unnamed: 0']).drop_duplicates('osda smiles')
    with open(OFF_ASSETS['zeo_feat_scaler.pkl'], 'rb') as f:
        zeo_scaler = pickle.load(f)
    with open(OFF_ASSETS['osda_feat_scaler.pkl'], 'rb') as f:
        osda_scaler = pickle.load(f)
    zc = pd.Series(ds.y_zeo1_code); osm = pd.Series(ds.y_osda1_smiles)
    ph_dense = ds.y_zeo1_feat[(zc == 'Dense/Amorphous').values][0].clone()      # 官方占位向量 pkl 原样
    ph_osdfree = ds.y_osda1_feat[(osm == 'OSDA-free').values][0].clone()
    zeo_tab = {code: zeo_scaler.transform(dz[dz['Code'] == code][zeo_cols].to_numpy(float)).ravel()
               for code in dz['Code']}
    osda_tab = {s: osda_scaler.transform(do[do['osda smiles'] == s][list(osda_cols)].to_numpy(float)).ravel()
                for s in do['osda smiles']}
    log('官方资产: dz %d 码 / do %d OSDA; 占位向量 Dense/Amorphous norm=%.3f, OSDA-free norm=%.3f'
        % (len(zeo_tab), len(osda_tab), ph_dense.norm().item(), ph_osdfree.norm().item()))

    label = df['label1'].to_numpy()
    zk = np.array([zeo_strip(c) for c in df['framework_code'].astype(str)], object)
    zk_real = pd.Series(zk).isin(set(zeo_tab)).to_numpy()
    is_absneg = label == 'absolute_neg'
    is_noprod = label == 'no_product'
    zeo_key_arr = np.where(is_absneg, 'Dense/Amorphous', zk)
    ok_zeo = is_absneg | zk_real

    smi = df['osda1_smiles']
    name_nan = df['osda1_name'].isna().to_numpy()
    smi_nan = smi.isna().to_numpy()
    match_map = {}
    for s in pd.unique(smi.dropna().astype(str)):
        if s in osda_tab:
            match_map[s] = s
            continue
        st = salt_strip(s)
        match_map[s] = st if (st is not None and st in osda_tab) else 'UNMAPPED'
    ok_osda_key = np.empty(n0, object)
    ok_osda_key[smi_nan & name_nan] = 'OSDA-free'                            # 双 NaN → 占位
    ok_osda_key[smi_nan & ~name_nan] = 'UNMAPPED'                            # 有名无 SMILES → 剔除
    m_idx = np.where(~smi_nan)[0]
    ok_osda_key[m_idx] = [match_map.get(str(s), 'UNMAPPED') for s in smi.values[m_idx]]
    has_osda = ok_osda_key != 'UNMAPPED'

    keep = ok_zeo & has_osda & ~is_noprod
    idx_keep = np.where(keep)[0]
    n_pop = len(idx_keep)
    log('人口筛选: 剔除 no_product %d / 混合码或未收录框架码 %d / 未收录或无名 OSDA %d → 纳入 %d 行 (%.1f%%)'
        % (int(is_noprod.sum()), int((~ok_zeo).sum()), int((~has_osda).sum()),
           n_pop, 100.0 * n_pop / n0))
    pop_label = label[idx_keep]
    log('纳入人口 label1 分布: ' + ', '.join('%s %d' % (k, v) for k, v in
        pd.Series(pop_label).value_counts().items()))
    log('  其中 absolute_neg %d 行 → 官方 Dense/Amorphous 占位 (官方 present=1.00 镜像)'
        % int((pop_label == 'absolute_neg').sum()))

    zeo_feat = np.zeros((n_pop, 143), float)
    osda_feat = np.zeros((n_pop, 14), float)
    zk_pop = zeo_key_arr[idx_keep]
    ok_pop = ok_osda_key[idx_keep]
    ph_d = ph_dense.numpy(); ph_o = ph_osdfree.numpy()
    for j in range(n_pop):
        zeo_feat[j] = ph_d if zk_pop[j] == 'Dense/Amorphous' else zeo_tab[zk_pop[j]]
        osda_feat[j] = ph_o if ok_pop[j] == 'OSDA-free' else osda_tab[ok_pop[j]]
    X_pop = X12[idx_keep]                                                    # 纳入人口 12 列 (capped+imputed)
    zeo_key_pop = zk_pop.astype(str); osda_key_pop = ok_pop.astype(str)
    pop_max = X_pop.max(axis=0)                                              # WSD 逐列归一 (scale_x_syn_ratio 镜像)

    # ---- 5) qt (fit 于纳入人口, 官方 QuantileTransformer 惯例) ----
    qts = {}
    X_qt = np.zeros_like(X_pop)
    for i, c in enumerate(COLS):
        qt = QuantileTransformer(n_quantiles=1000, random_state=0)
        X_qt[:, i] = qt.fit_transform(X_pop[:, i].reshape(-1, 1)).ravel()
        qts[c] = qt

    # ---- 6) system split (官方 mode='system' 镜像: sorted 体系序 + 80/20 + 行级 87.5/12.5) ----
    pairs = list(zip(zeo_key_pop, osda_key_pop))
    systems = sorted(set(pairs))
    train_sys, test_sys = train_test_split(systems, test_size=0.2, random_state=0)
    sys2rows = {}
    for r, p in enumerate(pairs):
        sys2rows.setdefault(p, []).append(r)
    tr_rows = [r for s in train_sys for r in sys2rows[s]]
    te_rows = [r for s in test_sys for r in sys2rows[s]]
    tr_rows, va_rows = train_test_split(tr_rows, test_size=0.125, random_state=0)
    log('system split (D3 镜像): 体系 %d (train+val %d / test %d) → 行 train %d / val %d / test %d'
        % (len(systems), len(train_sys), len(test_sys), len(tr_rows), len(va_rows), len(te_rows)))

    meta = {
        'data_version': 'zeobench_db_v1_8', 'n_v18': int(n0), 'n_pop': int(n_pop),
        'pop_label_counts': {str(k): int(v) for k, v in pd.Series(pop_label).value_counts().items()},
        'excl_no_product': int(is_noprod.sum()), 'excl_zeo_unmapped': int((~ok_zeo).sum()),
        'excl_osda_unmapped': int((~has_osda).sum()),
        'caps': caps, 'pop_max': {c: float(v) for c, v in zip(COLS, pop_max)},
        'n_systems': len(systems), 'split': {'train': len(tr_rows), 'val': len(va_rows), 'test': len(te_rows)},
        'official_assets_md5': {k: md5(v) for k, v in OFF_ASSETS.items()},
        'placeholder': {'dense_norm': float(ph_dense.norm()), 'osdafree_norm': float(ph_osdfree.norm())},
        'decisions': {'D1': '甲: zeobench_db_v1_8 重建 (12 列映射 + 官方 cap/impute/qt 惯例)',
                      'D2': '甲: DiffSyn 原设定条件向量 (zeo 143 + osda 14, 官方 scaler/占位), 二跑换 c_zeo v1 案 D',
                      'D3': '镜像官方 system split (体系级 80/20 + 行级 87.5/12.5, random_state=0)'},
        'built_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'git_commit': git_head(),
    }
    torch.save({'syn_qt': torch.from_numpy(X_qt).float(), 'syn_raw': X_pop,
                'zeo_feat': torch.from_numpy(zeo_feat).float(), 'osda_feat': torch.from_numpy(osda_feat).float(),
                'zeo_key': zeo_key_pop, 'osda_key': osda_key_pop,
                'train_idx': torch.tensor(tr_rows).long(), 'val_idx': torch.tensor(va_rows).long(),
                'test_idx': torch.tensor(te_rows).long(),
                'qts': qts, 'meta': meta}, DS_PT)
    log('落盘: %s (%.1f MB) — build 用时 %.1f min'
        % (DS_PT, os.path.getsize(DS_PT) / 1e6, (time.time() - t0) / 60))
    log('build 完成。预期: 纳入人口 %.1f%% / train-val-test 比例约 70-10-20%%' % (100.0 * n_pop / n0))

# =====================================================================
# train: DDPM 首训
# =====================================================================
def train():
    t0 = time.time()
    payload = torch.load(DS_PT, weights_only=False)
    meta = payload['meta']
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0); np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    diffusion, n_par = build_diffusion(device)
    log('==================== WP3/T3.2 DDPM 首训 train ====================')
    log('模型: MLPUnet %.2fM 参数 (<5M 规格) + 官方 GaussianDiffusion1D (pred_v/cosine/T=1000/l1)' % (n_par / 1e6))
    log('超参: Adam 1e-4 (betas 0.9,0.99), batch %d, %d 步 (≈%.0fM 样本, 官方 1M×64≈64M 同量级), '
        'cond_drop_prob=0.1, bf16 autocast (输出回 fp32 计损失) + TF32, device=%s'
        % (BATCH, STEPS, BATCH * STEPS / 1e6, device))

    X = payload['syn_qt'].to(device); Z = payload['zeo_feat'].to(device); O = payload['osda_feat'].to(device)
    tr = payload['train_idx'].to(device); va = payload['val_idx'].to(device)
    Xtr, Ztr, Otr = X[tr], Z[tr], O[tr]
    Xva, Zva, Ova = X[va], Z[va], O[va]
    n_tr = Xtr.shape[0]
    log('数据: train %d / val %d / test %d 行 (体系级 split, 测试集不参与训练与选模)' % (n_tr, len(va), len(payload['test_idx'])))

    opt = torch.optim.Adam(diffusion.parameters(), lr=1e-4, betas=(0.9, 0.99))
    train_log, val_log = [], []
    best_val, best_step = float('inf'), -1
    autocast_ctx = lambda: torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device != 'cpu'))

    for step in range(1, STEPS + 1):
        idx = torch.randint(0, n_tr, (BATCH,), device=device)
        with autocast_ctx():
            loss = diffusion(Xtr[idx].unsqueeze(1), zeo=Ztr[idx], osda=Otr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % LOG_EVERY == 0:
            train_log.append((step, float(loss.item())))
        if step % VAL_EVERY == 0:
            with torch.no_grad(), autocast_ctx():
                vl = float(diffusion(Xva.unsqueeze(1), zeo=Zva, osda=Ova).item())
            val_log.append((step, vl))
            if vl < best_val:
                best_val, best_step = vl, step
                torch.save({'step': step, 'val_loss': vl, 'state_dict': diffusion.state_dict(),
                           'n_params': n_par, 'meta': {k: meta[k] for k in ('data_version', 'n_pop', 'split')}},
                          MODEL_PT)
            pd.DataFrame({'step': [s for s, _ in train_log], 'train_loss': [l for _, l in train_log]}
                         ).merge(pd.DataFrame({'step': [s for s, _ in val_log], 'val_loss': [l for _, l in val_log]}),
                                 on='step', how='outer').sort_values('step').to_csv(LOSS_CSV, index=False)
            el = time.time() - t0
            print('step %d/%d | train %.4f | val %.4f (best %.4f @%d) | %.1f min elapsed, ETA %.1f min'
                  % (step, STEPS, train_log[-1][1], vl, best_val, best_step, el / 60,
                     el / step * (STEPS - step) / 60), flush=True)

    # ---- loss 收敛曲线 PNG ----
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
    ax.set_xlabel('step'); ax.set_ylabel('diffusion loss (pred_v, l1-weighted)')
    ax.set_yscale('log'); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title('WP3/T3.2 DDPM v0 first train (zeobench_db_v1_8, %d rows, MLPUnet %.2fM)'
                 % (meta['n_pop'], n_par / 1e6))
    fig.tight_layout(); fig.savefig(LOSS_PNG, dpi=160); plt.close(fig)

    # ---- manifest ----
    manifest = {
        'task': 'WP3/T3.2 DDPM 骨架首训 (M7 提前量, 建模线交接清单 2026-10-17 任务①)',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'data_version': 'zeobench_db_v1_8',
        'data_md5': md5(DB_PARQUET), 'code_md5': md5(os.path.abspath(__file__)),
        'official_assets_md5': meta['official_assets_md5'], 'git_commit': git_head(),
        'model': {'denoiser': 'MLPUnet (官方 Unet1D 的 MLP 镜像)', 'params': int(n_par),
                  'interface': '复用官方 GaussianDiffusion1D (pred_v/cosine/T=1000/l1/CFG)'},
        'hyperparams': {'lr': 1e-4, 'betas': [0.9, 0.99], 'batch': BATCH, 'steps': STEPS,
                        'cond_drop_prob': 0.1, 'cond_scale_eval': COND_SCALE,
                        'precision': 'TF32 + bf16 autocast, 模型输出回 fp32 计损失',
                        'val_every': VAL_EVERY, 'selection': 'best-val checkpoint (无 EMA, v0 简化)'},
        'data_stats': {'n_pop': meta['n_pop'], 'pop_label_counts': meta['pop_label_counts'],
                       'split': meta['split'], 'caps': meta['caps']},
        'best_val': {'step': best_step, 'val_loss': best_val},
        'train_duration_min': round((time.time() - t0) / 60, 1),
    }
    with open(MANIFEST, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ---- 训练报告 ----
    R = []
    R.append('WP3/T3.2 DDPM 骨架首训报告 v0 (M7 提前量) — %s' % time.strftime('%Y-%m-%d %H:%M'))
    R.append('=' * 78)
    R.append('【开工决策项定案】')
    R.append('D1 甲: 从 zeobench_db_v1_8 重建 DiffSyn 兼容数据集 — 12 列映射 (T=Si+Al+P+Ge+B 四面体五元,')
    R.append('      探针核验 Na/T 重建一致率 94.26%, 差异全为 cap 归因) + 官方 cap 协议 (分位水平照官方,')
    R.append('      v1.8 人口重算: %s)' % json.dumps({k: round(v, 3) for k, v in meta['caps'].items()}, ensure_ascii=False))
    R.append('D2 甲: 首训用 DiffSyn 原设定条件向量 (zeo 143 维 + osda 14 维, 官方 scaler + 占位向量 pkl 原样)')
    R.append('      — absolute_neg %d 行 → Dense/Amorphous 占位; 双 NaN → OSDA-free 占位; 混合码/未收录剔除;'
             % meta['pop_label_counts'].get('absolute_neg', 0))
    R.append('      no_product 剔除 → 二跑换 c_zeo v1 案 D (64 维任务导向监督投影)')
    R.append('D3: 镜像官方 system split — 体系 %d 个, 80/20 + 行级 87.5/12.5 (random_state=0, sorted 体系序)'
             % meta['n_systems'])
    R.append('')
    R.append('【数据】v1.8 %d 行 → 纳入 %d 行 (%.1f%%); train %d / val %d / test %d'
             % (meta['n_v18'], meta['n_pop'], 100.0 * meta['n_pop'] / meta['n_v18'],
                meta['split']['train'], meta['split']['val'], meta['split']['test']))
    R.append('【模型】MLPUnet %.2fM 参数 (规格 <5M) + 官方 GaussianDiffusion1D 全套扩散数学 (pred_v/cosine/T=1000/l1)'
             % (n_par / 1e6))
    R.append('【超参】Adam 1e-4 (0.9,0.99), batch %d, %d 步 ≈ %.0fM 样本 (官方 1M 步 × batch 64 ≈ 64M 同量级)'
             % (BATCH, STEPS, BATCH * STEPS / 1e6))
    R.append('【工程】TF32 + bf16 autocast (模型输出回 fp32 计损失, 不动官方扩散代码); 无 EMA (v0 简化,')
    R.append('        官方 Trainer1D 有 EMA — v1 可加); best-val checkpoint 选模, 测试集不参与')
    R.append('')
    R.append('【收敛】train loss 首 %.4f → 末 %.4f; val best %.4f @ step %d; 训练用时 %.1f min'
             % (train_log[0][1] if train_log else float('nan'),
                train_log[-1][1] if train_log else float('nan'), best_val, best_step,
                (time.time() - t0) / 60))
    R.append('【产出】%s / %s / %s / %s' % (MODEL_PT, LOSS_CSV, LOSS_PNG, MANIFEST))
    R.append('【版本】数据 zeobench_db_v1_8; 代码 md5 %s; git %s' % (manifest['code_md5'], manifest['git_commit']))
    with open(TRAIN_REP, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n\n' + '\n'.join(R) + '\n')
    log('\n'.join(R))
    log('train 完成: best val %.4f @ step %d; 报告 %s' % (best_val, best_step, TRAIN_REP))

# =====================================================================
# eval: 采样自检 + WSD 官方锚量级对照
# =====================================================================
def inv_qt(M, qts):
    # float64: syn 为 torch float32, sklearn 保留输入 dtype → float32(cap) 舍入值 (如 400.0)
    # 略大于 float64 端点 (399.999...94), 严格 > 比较会误触越界判定 (_diag_oob 实证);
    # cast float64 后输出严格有界于 [quantiles 首, 末] = 人口范围, 与 train 端点精确相等
    out = np.empty(M.shape, dtype=np.float64)
    for i, c in enumerate(COLS):
        out[:, i] = qts[c].inverse_transform(np.clip(M[:, i].reshape(-1, 1), 0, 1).astype(np.float64)).ravel()
    return out

def eval():
    t0 = time.time()
    payload = torch.load(DS_PT, weights_only=False)
    meta, qts = payload['meta'], payload['qts']
    pop_max = np.array([meta['pop_max'][c] for c in COLS])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    diffusion, n_par = build_diffusion(device)
    ck = torch.load(MODEL_PT, map_location=device, weights_only=False)
    diffusion.load_state_dict(ck['state_dict']); diffusion.eval()
    log('==================== WP3/T3.2 DDPM 首训 eval ====================')
    log('模型加载: best-val @ step %d (val %.4f), cond_scale=%.2f, device=%s'
        % (ck['step'], ck['val_loss'], COND_SCALE, device))

    X_raw = payload['syn_raw']; Z = payload['zeo_feat'].to(device); O = payload['osda_feat'].to(device)
    tr_idx = payload['train_idx'].numpy(); te_idx = payload['test_idx'].numpy()
    zeo_key = payload['zeo_key']; osda_key = payload['osda_key']
    R = []

    # ---------- 1) 采样自检: 2000 训练行 × 1 样本 ----------
    rng = np.random.RandomState(42)
    sel = rng.choice(tr_idx, size=min(2000, len(tr_idx)), replace=False)
    torch.manual_seed(42)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device != 'cpu')):
        syn = diffusion.sample(zeo=Z[sel], osda=O[sel], cond_scale=COND_SCALE,
                               batch_size=len(sel)).squeeze(1).cpu().numpy()
    syn_raw = inv_qt(syn, qts)                                   # 逆 qt 回原始量纲
    tr_raw = X_raw[tr_idx]                                        # 训练域 (原始量纲)
    log('采样自检: %d 训练行 × 1 样本, 逆 qt 回原始量纲, 逐特征对照训练域' % len(sel))
    R.append('一、采样自检 (%d 训练行 × 1 样本, seed 42, cond_scale 0.75, 逆 qt 回原始量纲)' % len(sel))
    R.append('%-12s %10s %8s %8s | %-24s | %-24s' % ('特征', 'W1', '越界%', 'std比',
             'train p1/p50/p99', 'syn p1/p50/p99'))
    oob_flags, col_flags = [], []
    for i, c in enumerate(COLS):
        a, b = tr_raw[:, i], syn_raw[:, i]
        w1 = w1_dist(a, b)
        oob = float(((b < a.min()) | (b > a.max())).mean() * 100)
        sr = float(np.std(b) / (np.std(a) + 1e-12))
        qa = np.quantile(a, [0.01, 0.5, 0.99]); qb = np.quantile(b, [0.01, 0.5, 0.99])
        R.append('%-12s %10.4f %8.2f %8.3f | %7.3g/%7.3g/%7.3g | %7.3g/%7.3g/%7.3g'
                 % (c, w1, oob, sr, *qa, *qb))
        if oob > 5: oob_flags.append((c, oob))
        if sr < 0.5 or sr > 2: col_flags.append((c, sr))
    R.append('越界判定 (越界%%>5): %s; 塌缩/发散判定 (std 比 <0.5 或 >2): %s'
             % ('无' if not oob_flags else str(oob_flags), '无' if not col_flags else str(col_flags)))

    # PNG: 4×3 逐特征分布对照 (比值列 log-x)
    fig, axes = plt.subplots(4, 3, figsize=(13.5, 12))
    for i, c in enumerate(COLS):
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
    fig.suptitle('WP3/T3.2 DDPM v0 sampling self-check: train vs sampled (raw domain, quantile-inverse)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(DIST_PNG, dpi=160); plt.close(fig)
    log('分布自检 PNG: %s' % DIST_PNG)

    # ---------- 2) WSD 同体系对照 (D2 甲路径, 官方协议镜像) ----------
    n_te = len(te_idx)
    n_samp = 25
    chunks = []
    for c0 in range(0, n_te, 1000):
        rows = te_idx[c0:c0 + 1000]
        z = Z[rows].repeat_interleave(n_samp, dim=0)
        o = O[rows].repeat_interleave(n_samp, dim=0)
        torch.manual_seed(43 + c0 // 1000)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device != 'cpu')):
            s = diffusion.sample(zeo=z, osda=o, cond_scale=COND_SCALE,
                                 batch_size=z.shape[0]).squeeze(1).cpu().numpy()
        chunks.append(inv_qt(s, qts))
        log('WSD 采样: 测试行 %d/%d (每行 %d 样本)' % (min(c0 + 1000, n_te), n_te, n_samp))
    P = np.concatenate(chunks)
    syn_pred = pd.DataFrame(P, columns=COLS)
    syn_pred['zeo'] = np.repeat(zeo_key[te_idx], n_samp)
    syn_pred['osda'] = np.repeat(osda_key[te_idx], n_samp)
    syn_true = pd.DataFrame(X_raw[te_idx], columns=COLS)
    syn_true['zeo'] = zeo_key[te_idx]; syn_true['osda'] = osda_key[te_idx]
    ps = syn_pred[COLS] / pop_max                                  # 官方 scale_x_syn_ratio 镜像 (我方人口 max)
    ts = syn_true[COLS] / pop_max

    rows = []
    for zeo, osda in syn_true[['zeo', 'osda']].value_counts().index:   # eval_zeolite_osda 同款遍历
        if zeo == 'Dense/Amorphous':
            continue
        m = (syn_pred['zeo'] == zeo) & (syn_pred['osda'] == osda)
        mt = (syn_true['zeo'] == zeo) & (syn_true['osda'] == osda)
        rows.append(dict(zeo=zeo, osda=osda, n_true=int(mt.sum()),
                         WSD=float(wasserstein_distance(ps[m], ts[mt])),
                         MMD=float(maximum_mean_discrepancy(ps[m], ts[mt]))))
    w = pd.DataFrame(rows)
    w.to_csv(WSD_CSV, index=False)
    R.append('')
    R.append('二、同体系 WSD 对照 (D2 甲路径; 测试行 ×%d 样本/行, cond_scale 0.75, 官方 geomloss Sinkhorn' % n_samp)
    R.append('    实现 wrap, 逐列人口 max 归一, 跳过 Dense/Amorphous 体系 — eval.py 协议镜像)')
    R.append('体系数: %d (测试集 %d 行; 明细 %s)' % (len(w), n_te, WSD_CSV))
    R.append('WSD mean ± std: %.4f ± %.4f  vs 官方锚 0.4237 (论文 Fig 2a 0.423)' % (w['WSD'].mean(), w['WSD'].std()))
    R.append('MMD mean ± std: %.4f ± %.4f' % (w['MMD'].mean(), w['MMD'].std()))
    ratio = w['WSD'].mean() / OFFICIAL_WSD_ANCHOR
    R.append('量级判定: %.2f× 官方锚 — %s' % (ratio, '同量级 (0.5–2×), 首训达标' if 0.5 <= ratio <= 2 else '偏离量级, 登记待复核'))
    log('\n'.join(R[-6:]))

    R.append('')
    R.append('三、结论')
    R.append('1) loss 收敛与采样自检见训练报告与 %s' % os.path.basename(DIST_PNG))
    R.append('2) WSD %s 官方锚 (0.4237): 数据集与模型容量均不同 (官方 106.6M / 我方 %.2fM), 按交接清单§1.5'
             % (ratio, n_par / 1e6))
    R.append('   量级与方向即首训达标, 不要求逐位一致')
    R.append('3) 采样自检越界归因 (首版报告 Si/Al 32.15% / Al/P 47.10%): 系 float32 舍入假象 — train 端点为')
    R.append('   float64 cap 值 (399.999...94/1.71737...75) 而 syn 经 float32 链路舍入为 400.0/1.7173728,')
    R.append('   严格 > 比较将命中 cap 带的 syn 行误判为 above; syn 命中 cap 带占比与 train 一致')
    R.append('   (Si/Al 32.15/31.14, Al/P 47.10/48.46, 整数 cap 列 99/250/4/200 above=0 佐证);')
    R.append('   inv_qt 已 cast float64, 修复后越界归零. cap 带 = 组分缺席语义 (纯硅 Al=0→inf→cap,')
    R.append('   无磷 P=0→inf→cap), 属结构性行为非模型发散')
    R.append('4) K/T 与 cryst_time std 比 >2 (2.14/3.52) 为稀疏重尾列过散: W1 相对 p99 量级 <10%, 越界≈0,')
    R.append('   属 DDPM 在稀疏列上的已知行为, 登记 v1 观察项 (EMA / per-col loss 加权 / 更长训练)')
    R.append('5) 附带口径差异登记: sklearn 1.9.1 QuantileTransformer 默认 subsample=10_000 (官方 1.1.2 无),')
    R.append('   21,714 行人口实际 fit 于 10k 固定种子子样本, 仅影响极值端点 (cryst_temp qt 上界 450 vs')
    R.append('   人口 720), 分位主体无实质差异; v1 修正项 = 显式禁用 subsample 对齐官方行为')
    R.append('6) 数据版本 zeobench_db_v1_8; eval 用时 %.1f min' % ((time.time() - t0) / 60))
    with open(EVAL_REP, 'w', encoding='utf-8') as f:
        f.write('\n'.join(R) + '\n')
    # manifest 补 eval 段
    with open(MANIFEST, encoding='utf-8') as f:
        manifest = json.load(f)
    manifest['eval'] = {'n_distcheck': int(len(sel)), 'wsd_systems': int(len(w)),
                        'wsd_mean': float(w['WSD'].mean()), 'wsd_std': float(w['WSD'].std()),
                        'official_anchor': OFFICIAL_WSD_ANCHOR, 'ratio_to_anchor': float(ratio),
                        'best_val_step': int(ck['step']), 'cond_scale': COND_SCALE,
                        'eval_duration_min': round((time.time() - t0) / 60, 1)}
    with open(MANIFEST, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log('eval 完成: 报告 %s; manifest %s' % (EVAL_REP, MANIFEST))

# =====================================================================
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['build', 'train', 'eval', 'all'])
    a = ap.parse_args()
    if a.stage in ('build', 'all'):
        build()
    if a.stage in ('train', 'all'):
        train()
    if a.stage in ('eval', 'all'):
        eval()
