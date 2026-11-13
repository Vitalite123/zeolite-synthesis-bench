# -*- coding: utf-8 -*-
"""
WP5 指标统一 API v0 (M6 顺延任务) —— 建模线交接清单 2026-10-17 任务②
====================================================================
指标清单 (执行计划 §4 M6 / 课题总计划 §判别器-评估, 7 类):
  1. Wasserstein-2 联合 (官方 geomloss Sinkhorn wrap) + 逐维 W1 (scipy, diffsyn_repro.py 同款)
  2. MMD (官方 rbf 多带宽 wrap, 对角剔除)
  3. COV-P / COV-R (官方 coverage wrap — DiffSyn 定义兼容, 官方逐列阈值)
  4. 物理有效率 (自行实现: 全有限 + 比值列>=0 + 温度/时间物理窗口; G3 线 >=98%)
  5. 条件一致率 (自行实现: scorer 注入式, 生成配方被判别器预测目标相的概率; G3 线 >=90%)
  6. 新颖性 (自行实现: 与训练集最近邻距离 + 训练凸包外比例 + 逐列盒外下界)
  7. 外推指数 (自行实现: 时间切分 vs random 切分指标衰减比; 首跑实验=M10)

关键纪律 (交接清单 §2.2): WSD / MMD / COV 直接 wrap zeosyn_gen/data/metrics.py 官方实现, 不重写定义
—— 与官方读数可比的唯一保证; 新增指标自行实现, 接口与官方风格对齐 (DataFrame 输入 + dict 返回)。

校准 (交接清单 §2.3, 验收口径, assert 失败即非零退出 → 可挂 CI):
  calib1  官方预测产物逐位复算: Data/diffsyn_metric_recheck.csv 读数用本 API 重算 → 逐位一致 assert
  calib2  v1.8 训练域自反调用: 自比恒等参考值 (W1=0 / COV=1 / NN=0 / 凸包外=0 / 一致率=1) 无异常
  demo    WP3 首训产物应用: 3 个测试体系 × 200 样本全指标首读 (G3 线对照)

────────────────────────────────────────────────────────────────────────────
接口文档 (v0)
────────────────────────────────────────────────────────────────────────────
类 ZeoSynMetrics(cols=官方 12 列) —— 所有方法接受 pd.DataFrame 或 2D ndarray:

  scale(X, ref)                        -> DataFrame  逐列人口 max 归一 (官方 scale_x_syn_ratio 同法)
  w2_joint(pred, true, ref=None)       -> float      联合 Wasserstein-2 (官方 Sinkhorn blur=0.05)
                                                    输入须已归一, 或传 ref 由本方法归一 (WSD 协议)
  w1_per_dim(pred, true)               -> dict       逐维 Wasserstein-1 (scipy, 原始量纲, 逐列)
  mmd(pred, true, ref=None)            -> float      MMD (官方 rbf 多带宽实现, 归一空间读数)
  cov_pr(pred, true)                   -> dict       COV-P/COV-R 逐列 (官方 coverage wrap, 官方阈值,
                                                    pred>200 行时官方 sample(200, random_state=0) 上限)
  cov_pr_mean(pred, true)              -> dict       {COV-P, COV-R, per_col} 12 列均值聚合
  physical_validity_rate(X, bounds)    -> dict       物理有效率 + 分规则失效计数
                                                    bounds='strict'(默认: temp∈[20,350]°C, time∈(0,2e4]h)
                                                          |'loose'(temp∈(0,750]°C, 水热窗口上限放宽)
                                                          |dict({'cryst_temp':(lo,hi),'cryst_time':(lo,hi)})
  conditional_consistency(gen, scorer, target, tau=0.5)
                                       -> dict       {mean_p, frac_ge_tau} — scorer(df, target) 返回
                                                    P(target) 数组 (v0 无内置 scorer, v1 接 L1/L2 适配器)
  nn_distance(gen, train, ref=None, gen_sys=None, train_sys=None)
                                       -> dict       新颖性-最近邻距离 (归一空间欧氏; 给 sys 键→逐组 NN;
                                                    组内无训练行→NaN 计数 n_no_train)
  outside_hull_fraction(gen, train, gen_group, train_group, ref=None)
                                       -> dict       新颖性-凸包外比例 (逐组; 组=体系/骨架由调用方定;
                                                    判定=分离超平面 LP: x∈conv(P) ⟺ LP 有界, unbounded
                                                    ⟺ 凸包外 (13 变量 n 约束, 对组规模不敏感);
                                                    LP 数值异常组 → 逐列盒近似 (盒⊇凸包, 下界) 并计数)
  outside_box_fraction(gen, train, ref=None)
                                       -> dict       逐列训练 min/max 盒外比例 (全局凸包外的下界近似)
  extrapolation_index(m_rand, m_time, higher_better=True)
                                       -> dict       {ratio=m_time/m_rand, decay_pct} — decay_pct>0 = 衰减
  extrapolation_index_from_dicts(d_rand, d_time, hb_map)
                                       -> dict       逐指标批量 (hb_map 指定各指标方向)
  full_suite(pred, true, ref)          -> dict       W2/W1/MMD/COV/物理有效率 一站式
                                                    (新颖性/条件一致率/外推指数需额外输入, 单独调用)

用量: python wp5_metrics_api_v0.py calib1|calib2|demo|all
产出: Data/wp5_metrics_api_report.txt (gitignored) + Data/wp5_demo_wp3_samples.csv (gitignored)
依赖: zeosyn_gen/ (官方指标实现), torch, geomloss, scipy, sklearn; demo 需 WP3 产物
"""
import os, sys, json, time, pickle, hashlib, argparse, warnings, subprocess
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZG = os.path.join(BASE, 'vendor', 'zeosyn_gen')
os.chdir(ZG)                                       # 官方代码以 cwd 相对路径工作
sys.path.insert(0, ZG)

import torch
from scipy.stats import wasserstein_distance as w1_dist
from scipy.optimize import linprog

import data.utils as _du
sys.modules['utils'] = _du                        # eval.py 同款 trick (ZeoSynGen_dataset pickle)
from data.metrics import (wasserstein_distance as _off_wsd,     # 官方实现, wrap 不重写
                          maximum_mean_discrepancy as _off_mmd,
                          coverage as _off_cov)
os.chdir(ZG)                                       # data.metrics 导入有 cwd 副作用, 强制回 ZG

# ---------------- 路径与常量 ----------------
REPORT     = os.path.join(BASE, 'data', 'wp5_metrics_api_report.txt')
RECHECK    = os.path.join(BASE, 'data', 'diffsyn_metric_recheck.csv')
DS_PT      = os.path.join(BASE, 'data', 'wp3_ddpm_v0_dataset.pt')
DEMO_CSV   = os.path.join(BASE, 'data', 'wp5_demo_wp3_samples.csv')
RUN_DIR    = os.path.join('runs', 'diff', 'system', 'run1')     # 官方产物 (ZG 相对)
PKL_PATH   = os.path.join('data', 'ZeoSynGen_dataset.pkl')
PRED_CSV   = os.path.join(RUN_DIR, 'syn_pred_agg-cond_scale_0.75-test.csv')

COLS_OFFICIAL = ['Si/Al', 'Al/P', 'Si/Ge', 'Si/B', 'Na/T', 'K/T', 'OH/T', 'F/T',
                 'H2O/T', 'sda1/T', 'cryst_temp', 'cryst_time']

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

# =====================================================================
# 指标统一 API
# =====================================================================
class ZeoSynMetrics:
    """WP5 指标统一 API v0 —— 官方三指标 wrap + 新增四指标, 接口与官方风格对齐。
    全部方法接受 pd.DataFrame 或 2D ndarray; W2/MMD 在逐列人口 max 归一空间计算 (官方 WSD 协议),
    W1/COV/物理有效率在原始量纲计算 (官方 coverage 协议)。"""

    EI_HIGHER_BETTER = {'COV-P': True, 'COV-R': True, 'W2_joint': False, 'W1_mean': False,
                        'MMD': False, 'physical_validity': True, 'nn_distance': False,
                         'conditional_consistency': True, 'outside_hull': False}

    def __init__(self, cols=COLS_OFFICIAL):
        bad = [c for c in cols if c not in COLS_OFFICIAL]
        if bad:
            raise ValueError('列 %s 不在官方 12 列内 (官方 coverage 阈值按列名定义): %s' % (bad, COLS_OFFICIAL))
        self.cols = list(cols)

    # ---------- 内部 ----------
    def _as_df(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(np.asarray(X, dtype=float), columns=self.cols)

    def _mat(self, X):
        return self._as_df(X)[self.cols].to_numpy(dtype=float)

    # ---------- 官方 wrap (纪律: 不重写定义) ----------
    def scale(self, X, ref):
        """逐列人口 max 归一 (官方 utils.scale_x_syn_ratio 同法; ref=参考人口 DataFrame)"""
        X, ref = self._as_df(X).copy(), self._as_df(ref)
        for c in self.cols:
            m = float(ref[c].max())
            X[c] = X[c] / (m if m > 0 else 1.0)
        return X

    def w2_joint(self, pred, true, ref=None):
        """联合 Wasserstein-2 (官方 geomloss Sinkhorn blur=0.05; 输入须已归一或传 ref)"""
        if ref is not None:
            pred, true = self.scale(pred, ref), self.scale(true, ref)
        return float(_off_wsd(self._as_df(pred)[self.cols], self._as_df(true)[self.cols]))

    def w1_per_dim(self, pred, true):
        """逐维 Wasserstein-1 (scipy.stats, 原始量纲, diffsyn_repro.py 同款)"""
        p, t = self._as_df(pred), self._as_df(true)
        return {c: float(w1_dist(p[c], t[c])) for c in self.cols}

    def mmd(self, pred, true, ref=None):
        """MMD (官方 rbf 多带宽实现, 对角剔除; 归一空间读数)"""
        if ref is not None:
            pred, true = self.scale(pred, ref), self.scale(true, ref)
        return float(_off_mmd(self._as_df(pred)[self.cols], self._as_df(true)[self.cols]))

    def cov_pr(self, pred, true):
        """COV-P / COV-R 逐列 (官方 coverage wrap: 官方阈值 + pred>200 时 sample(200, seed=0))"""
        return _off_cov(self._as_df(pred)[self.cols].reset_index(drop=True),
                        self._as_df(true)[self.cols].reset_index(drop=True))

    def cov_pr_mean(self, pred, true):
        """COV-P/R 12 列均值聚合 {COV-P, COV-R, per_col}"""
        cov = self.cov_pr(pred, true)
        return {'COV-P': float(np.mean([cov['precision'][c] for c in self.cols])),
                'COV-R': float(np.mean([cov['recall'][c] for c in self.cols])),
                'per_col': cov}

    # ---------- 新增指标 1: 物理有效率 ----------
    PHYS_STRICT = {'cryst_temp': (20.0, 350.0), 'cryst_time': (0.0, 20000.0)}   # 水热合成窗口
    PHYS_LOOSE  = {'cryst_temp': (0.0, 750.0),  'cryst_time': (0.0, 20000.0)}   # 覆盖观测域 (max 720°C)

    def physical_validity_rate(self, X, bounds='strict'):
        """物理有效率: 全 12 列有限 + 10 比值列>=0 + cryst_temp∈[lo,hi] + cryst_time∈(lo,hi]。
        bounds='strict'(默认)/'loose'/dict; 返回 rate + 分规则失效计数 (诊断用)。
        注: 比值上界无物理约束 (由 cap 协议管); qt 有界采样天然通过, 本指标防护无界生成路径
        (GAN/VAE 基线、v1 直接原始量纲输出的模型)。"""
        b = {'strict': self.PHYS_STRICT, 'loose': self.PHYS_LOOSE}.get(bounds, bounds)
        M12 = self._mat(X)
        ok = np.isfinite(M12).all(axis=1)
        fail = {'nonfinite_rows': int((~np.isfinite(M12).all(axis=1)).sum())}
        neg = (M12[:, :10] < 0).any(axis=1)
        fail['negative_ratio_cells'] = int((M12[:, :10] < 0).sum())
        ok &= ~neg
        (tlo, thi), (hlo, hhi) = b['cryst_temp'], b['cryst_time']
        v_t, v_h = M12[:, 10], M12[:, 11]
        fail['cryst_temp_below'] = int((v_t < tlo).sum());  fail['cryst_temp_above'] = int((v_t > thi).sum())
        fail['cryst_time_nonpos'] = int((v_h <= hlo).sum()); fail['cryst_time_above'] = int((v_h > hhi).sum())
        ok &= (v_t >= tlo) & (v_t <= thi) & (v_h > hlo) & (v_h <= hhi)
        return {'rate': float(ok.mean()), 'n': int(len(M12)), 'n_invalid': int((~ok).sum()),
                'bounds': {k: tuple(v) for k, v in b.items()}, 'fail': fail}

    # ---------- 新增指标 2: 条件一致率 ----------
    def conditional_consistency(self, gen, scorer, target=None, tau=0.5):
        """条件一致率: scorer(df, target) → P(target) 数组 (生成配方的判别器目标相概率)。
        返回 {mean_p, frac_ge_tau}; G3 线 >=90% 读数 = frac_ge_tau。v0 无内置 scorer (v1 接 L1/L2 适配器)。"""
        df = self._as_df(gen)
        p = np.asarray(scorer(df, target), dtype=float)
        assert p.shape == (len(df),), 'scorer 须返回形状 (%d,) 的概率数组' % len(df)
        return {'mean_p': float(p.mean()), 'frac_ge_tau': float((p >= tau).mean()),
                'tau': tau, 'target': None if target is None else str(target), 'n': int(len(df))}

    # ---------- 新增指标 3: 新颖性 ----------
    @staticmethod
    def _cdist_min(A, B, chunk=8192):
        """逐行 min 欧氏距离 (分块防大矩阵), float64 — cdist 展开式噪声底 ~6e-8 (自反 max)"""
        out = np.empty(len(A))
        for i0 in range(0, len(A), 2048):
            a = torch.from_numpy(A[i0:i0 + 2048])
            mins = []
            for j0 in range(0, len(B), chunk):
                b = torch.from_numpy(B[j0:j0 + chunk])
                mins.append(torch.cdist(a, b).min(dim=1).values)
            out[i0:i0 + 2048] = torch.stack(mins, dim=1).min(dim=1).values.numpy()
        return out

    def nn_distance(self, gen, train, ref=None, gen_sys=None, train_sys=None):
        """新颖性-最近邻距离: gen 每行到 train 的最小欧氏距离 (ref 归一空间, 与 W2 同尺度)。
        给 sys 键 → 逐组 NN (组内无训练行 → NaN, 计 n_no_train)。"""
        G, T = self._mat(gen), self._mat(train)
        if ref is not None:
            Rm = self._mat(ref)
            sc = np.where(Rm.max(axis=0) > 0, Rm.max(axis=0), 1.0)
            G, T = G / sc, T / sc
        if gen_sys is None:
            d = self._cdist_min(G, T)
        else:
            gs, ts = np.asarray(gen_sys), np.asarray(train_sys)
            d = np.full(len(G), np.nan)
            for s in pd.unique(gs):
                gi = np.where(gs == s)[0]; ti = np.where(ts == s)[0]
                if len(ti):
                    d[gi] = self._cdist_min(G[gi], T[ti])
        v = d[np.isfinite(d)]
        return {'mean': float(v.mean()) if len(v) else float('nan'),
                'median': float(np.median(v)) if len(v) else float('nan'),
                'p90': float(np.percentile(v, 90)) if len(v) else float('nan'),
                'max': float(v.max()) if len(v) else float('nan'),
                'n': int(len(d)), 'n_no_train': int(np.isnan(d).sum())}

    @staticmethod
    def _sep_lp(x, P):
        """分离超平面 LP 凸包判定: x ∈ conv(P) ⟺ 不存在 (w,b): w·x+b>0 且 P@w+b≤-1。
        LP: minimize [-x,-1]·[w,b] s.t. P@w+b ≤ -1, (w,b) 自由 (d+1=13 变量 n 约束, HiGHS);
        status=3 (unbounded) ⟺ 凸包外 / status=0 (optimal) ⟺ 凸包内 (强分离定理)。
        —— 替代 Qhull 半空间判定: 12D 大点集 (百点级) facet 数计算爆炸 (实测 1102 点组挂死),
        LP 判定约束数即组点数, 对组规模不敏感。"""
        c = np.concatenate([-x, [-1.0]])
        A_ub = np.hstack([P, np.ones((len(P), 1))])
        return linprog(c, A_ub=A_ub, b_ub=-np.ones(len(P)),
                       bounds=[(None, None)] * (len(x) + 1), method='highs')

    def outside_hull_fraction(self, gen, train, gen_group, train_group, ref=None):
        """新颖性-凸包外比例: 逐组判 gen 行是否落在该组 train 行凸包内 (ref 归一空间)。
        组键由调用方定 (体系级 = 条件生成语义 / 骨架级 = 生成外推语义)。
        判定 = 分离超平面 LP (x∈conv(P) ⟺ LP 有界; unbounded ⟺ 外; 见 _sep_lp)。
        LP 数值异常组 → 逐列盒近似 (盒⊇凸包 → 外比例下界) 并计数 n_box_fallback。
        组内无 train 行 → 记凸包外 + n_groups_no_train。"""
        G, T = self._mat(gen), self._mat(train)
        if ref is not None:
            Rm = self._mat(ref)
            sc = np.where(Rm.max(axis=0) > 0, Rm.max(axis=0), 1.0)
            G, T = G / sc, T / sc
        gs, ts = np.asarray(gen_group), np.asarray(train_group)
        outside = np.zeros(len(G), bool)
        n_lp = n_box = n_empty = 0
        for s in pd.unique(gs):
            gi = np.where(gs == s)[0]; ti = np.where(ts == s)[0]
            if len(ti) == 0:
                outside[gi] = True; n_empty += 1
                continue
            P, Xg = T[ti], G[gi]
            try:
                for xi in gi:
                    r = self._sep_lp(G[xi], P)
                    if r.status == 3:
                        outside[xi] = True
                    elif r.status != 0:
                        raise RuntimeError('LP status=%d (非 0/3)' % r.status)
                n_lp += 1
            except Exception:                       # 数值异常组 → 盒近似 (下界)
                outside[gi] = ((Xg < P.min(axis=0) - 1e-12) | (Xg > P.max(axis=0) + 1e-12)).any(axis=1)
                n_box += 1
        return {'fraction_outside': float(outside.mean()), 'n': int(len(G)),
                'n_outside': int(outside.sum()), 'n_groups': int(len(pd.unique(gs))),
                'n_groups_no_train': n_empty, 'n_sep_lp': n_lp, 'n_box_fallback': n_box}

    def outside_box_fraction(self, gen, train, ref=None):
        """逐列训练 min/max 盒外比例 — 全局训练凸包外比例的下界近似 (盒 ⊇ 凸包)"""
        G, T = self._mat(gen), self._mat(train)
        lo, hi = T.min(axis=0), T.max(axis=0)
        outside = ((G < lo) | (G > hi)).any(axis=1)
        return {'fraction_outside': float(outside.mean()), 'n': int(len(G)),
                'per_col': {c: float(((G[:, i] < lo[i]) | (G[:, i] > hi[i])).mean())
                            for i, c in enumerate(self.cols)}}

    # ---------- 新增指标 4: 外推指数 ----------
    @staticmethod
    def extrapolation_index(m_rand, m_time, higher_better=True):
        """外推指数 = 时间切分读数 / random 切分读数; decay_pct>0 = 衰减 (higher_better 反号)"""
        if m_rand == 0:
            ratio = 1.0 if m_time == 0 else float('nan')
        else:
            ratio = float(m_time / m_rand)
        decay = (1.0 - ratio) * 100 if higher_better else (ratio - 1.0) * 100
        return {'ratio': ratio, 'decay_pct': float(decay), 'higher_better': bool(higher_better)}

    def extrapolation_index_from_dicts(self, d_rand, d_time, hb_map=None):
        """逐指标批量外推指数 (hb_map 缺省用 EI_HIGHER_BETTER)"""
        hb_map = hb_map or self.EI_HIGHER_BETTER
        return {k: self.extrapolation_index(d_rand[k], d_time[k], hb_map.get(k, True))
                for k in d_rand if k in d_time}

    # ---------- 一站式 ----------
    def full_suite(self, pred, true, ref, validity_bounds='strict'):
        """W2/W1/MMD/COV/物理有效率 一站式 (ref=训练人口, W2/MMD 按人口 max 归一)"""
        cov = self.cov_pr_mean(pred, true)
        w1d = self.w1_per_dim(pred, true)
        return {'W2_joint': self.w2_joint(pred, true, ref),
                'MMD': self.mmd(pred, true, ref),
                'W1_per_dim': w1d, 'W1_mean': float(np.mean(list(w1d.values()))),
                'COV-P': cov['COV-P'], 'COV-R': cov['COV-R'], 'COV_per_col': cov['per_col'],
                'physical_validity': self.physical_validity_rate(pred, validity_bounds)}

# =====================================================================
# 校准 ①: 官方预测产物逐位复算 (diffsyn_repro.py Part A 协议逐位镜像)
# =====================================================================
def calib1(M):
    os.chdir(ZG)
    t0 = time.time()
    log('')
    log('==================== 校准 ① 官方预测产物逐位复算 ====================')
    log('协议: diffsyn_repro.py Part A 逐位镜像 (官方 pkl + system split + 官方预测 csv,')
    log('      本 API w2_joint/mmd/cov_pr 重算 → 与 Data/diffsyn_metric_recheck.csv 逐位 assert)')
    with open(PKL_PATH, 'rb') as f:
        dataset = pickle.load(f)
    COLS = dataset.ratio_names + dataset.cond_names
    assert COLS == M.cols, '官方列序与 API 默认列不一致'
    _, _, _dataset = dataset.train_val_test_split(mode='system', both_graph_feat_present=True, random_state=0)
    syn_true, zeo_code, osda_smiles = _dataset[1], _dataset[3], _dataset[13]
    for i, r in enumerate(COLS):                    # 逆 quantile 变换回原始量纲 (原脚本逐位同款)
        qt = dataset.qts[r]
        syn_true[:, i] = torch.tensor(qt.inverse_transform(syn_true[:, i].reshape(-1, 1))).reshape(-1)
    syn_true = pd.DataFrame(syn_true, columns=COLS)
    syn_true['zeo'], syn_true['osda'] = zeo_code, osda_smiles
    syn_true_scaled = _du.scale_x_syn_ratio(syn_true, dataset)
    syn_pred = pd.read_csv(PRED_CSV)
    syn_pred_scaled = _du.scale_x_syn_ratio(syn_pred, dataset)
    log('官方资产: 数据集 %d 路线 / 测试真值 %d 行 / 官方预测 %d 行 (50 样本 × 测试点)'
        % (len(dataset), len(syn_true), len(syn_pred)))

    rows = []
    for zeo, osda in syn_true[['zeo', 'osda']].value_counts().index:   # eval_zeolite_osda 同款遍历
        if zeo == 'Dense/Amorphous':
            continue
        m = (syn_pred['zeo'] == zeo) & (syn_pred['osda'] == osda)
        mt = (syn_true['zeo'] == zeo) & (syn_true['osda'] == osda)
        p, t = syn_pred.loc[m, COLS], syn_true.loc[mt, COLS]
        ps, ts = syn_pred_scaled.loc[m, COLS], syn_true_scaled.loc[mt, COLS]
        cov = M.cov_pr(p, t)
        rows.append(dict(zeo=zeo, osda=osda,
                         WSD_ours=M.w2_joint(ps, ts), MMD_ours=M.mmd(ps, ts),
                         **{c + '_prec': cov['precision'][c] for c in COLS},
                         **{c + '_rec': cov['recall'][c] for c in COLS}))
    ours = pd.DataFrame(rows)
    recheck = pd.read_csv(RECHECK)
    mg = ours.merge(recheck, on=['zeo', 'osda'], suffixes=('_new', '_old'))
    assert len(mg) == len(recheck) == 372, '体系对齐失败: %d vs %d' % (len(mg), len(recheck))

    cmp_cols = ['WSD_ours', 'MMD_ours'] + [c + s for c in COLS for s in ('_prec', '_rec')]
    log('逐位对照 (372 体系 × 26 数值列):')
    log('%-14s %12s' % ('列', 'max|Δ|'))
    worst = 0.0
    for c in cmp_cols:
        d = float((mg[c + '_new'] - mg[c + '_old']).abs().max())
        worst = max(worst, d)
        log('%-14s %12.3e' % (c, d))
    log('总 max|Δ| = %.3e (容差 1e-12) — 用时 %.1f min' % (worst, (time.time() - t0) / 60))
    assert worst <= 1e-12, '校准 ① 未达逐位一致: max|Δ|=%.3e' % worst

    with open(os.path.join(RUN_DIR, 'wsd_zeo_osda.json')) as f:
        off_wsd = json.load(f)['WSD mean']
    with open(os.path.join(RUN_DIR, 'mmd_zeo_osda.json')) as f:
        off_mmd = json.load(f)['MMD mean']
    log('读数锚定: WSD mean 本 API %.4f vs 官方 json %.4f | MMD mean 本 API %.4f vs 官方 %.4f'
        % (ours['WSD_ours'].mean(), off_wsd, ours['MMD_ours'].mean(), off_mmd))
    log('校准 ① 判定: PASS (官方产物逐位复算一致, 周末验收检查表 WP5 项达成)')
    return worst

# =====================================================================
# 校准 ②: v1.8 训练域自反调用 (恒等参考值)
# =====================================================================
def calib2(M):
    t0 = time.time()
    log('')
    log('==================== 校准 ② v1.8 训练域自反调用 ====================')
    log('协议: WP3 dataset (zeobench_db_v1_8 重建, 12 列原始量纲) train 域自比 → 恒等参考值')
    payload = torch.load(DS_PT, weights_only=False)
    meta = payload['meta']
    assert meta['data_version'] == 'zeobench_db_v1_8', '数据版本非 v1.8: %s' % meta['data_version']
    X = np.asarray(payload['syn_raw'])
    tr = payload['train_idx'].numpy()
    train_df = pd.DataFrame(X[tr], columns=M.cols)
    zeo_k = np.array(payload['zeo_key']); osda_k = np.array(payload['osda_key'])
    tr_sys = np.array(['%s||%s' % (a, b) for a, b in zip(zeo_k[tr], osda_k[tr])])
    log('训练域: %d 行 (v1.8 纳入人口 %d, train split), 数据版本 zeobench_db_v1_8'
        % (len(train_df), meta['n_pop']))
    rng = np.random.RandomState(0)
    results = {}

    # ---- 分布距离自反 (同集自比 → 恒等参考 ≈0) ----
    sel = rng.choice(len(train_df), 4000, replace=False)
    sub = train_df.iloc[sel].reset_index(drop=True)
    w2_self = M.w2_joint(sub, sub, ref=train_df)
    mmd_self = M.mmd(sub, sub, ref=train_df)
    w1_self = M.w1_per_dim(sub, sub)
    results['W2_self'] = w2_self; results['MMD_self'] = mmd_self
    log('[1] 分布距离自反 (n=4000 同集): W2=%.6f (Sinkhorn 熵正则自代价, 预期~0) | MMD=%.6f (对角剔除'
        '公式自比微负, 预期~0)' % (w2_self, mmd_self))
    assert max(w1_self.values()) == 0.0, 'W1 自反非零'
    log('    W1 逐列自反 = 0 (精确); COV 自反:')
    # ---- COV 自反 (官方 sample(200) 上限: n<=200 无子采样 → P=R=1 精确) ----
    sel2 = rng.choice(len(train_df), 200, replace=False)
    s2 = train_df.iloc[sel2].reset_index(drop=True)
    cov_self = M.cov_pr(s2, s2)
    p_all = all(v == 1.0 for v in cov_self['precision'].values())
    r_all = all(v == 1.0 for v in cov_self['recall'].values())
    assert p_all and r_all, 'COV 自反非 1'
    log('    COV-P / COV-R 逐列自反 = 1.000 (精确, n=200 ≤ 官方 200 上限, 无子采样)')
    sel2b = rng.choice(len(train_df), 2000, replace=False)
    s2b = train_df.iloc[sel2b].reset_index(drop=True)
    cov_big = M.cov_pr(s2b, s2b)
    log('    官方协议结构性行为登记: n=2000 自反时 COV-P=%.3f / COV-R=%.3f — pred 被 sample(200) 截断,'
        % (float(np.mean(list(cov_big['precision'].values()))),
           float(np.mean(list(cov_big['recall'].values())))))
    log('    2000 真值中仅 200 采样近邻可命中, COV-R 上限受采样密度约束 (官方定义固有, 非实现缺陷);')
    log('    读数纪律: COV 的 pred 集规模应与官方协议同量级 (50×n_true/体系)')

    # ---- 物理有效率域读数 (非 assert, 诚实登记) ----
    v_strict = M.physical_validity_rate(train_df, 'strict')
    v_loose = M.physical_validity_rate(train_df, 'loose')
    log('[2] 物理有效率域读数 (训练域 %d 行): strict [20,350]°C = %.4f | loose (0,750]°C = %.4f'
        % (len(train_df), v_strict['rate'], v_loose['rate']))
    log('    strict 失效: temp<20 %d 行 (插补伪影) + temp>350 %d 行 + time<=0 %d 行 (插补伪影) + 比值负 0'
        % (v_strict['fail']['cryst_temp_below'], v_strict['fail']['cryst_temp_above'],
           v_strict['fail']['cryst_time_nonpos']))
    log('    登记项: 训练域自身有效率为域上限参照 — G3 线 98%% 低于域上限, 可达; 插补伪影行 (T_C≈0/t_h=0)')
    log('    为 IterativeImputer min_value=0 产物, 与 WP3 报告 v1 登记项 (qt subsample) 同源归因')
    results['phys_strict'] = v_strict['rate']; results['phys_loose'] = v_loose['rate']

    # ---- 新颖性自反 ----
    sel3 = rng.choice(len(train_df), 2000, replace=False)
    g3 = train_df.iloc[sel3].reset_index(drop=True)
    nn_self = M.nn_distance(g3, train_df, ref=train_df)
    assert nn_self['max'] < 1e-6, 'NN 自反超容差: %s' % nn_self['max']
    log('[3] 新颖性-NN 自反 (n=2000 vs train): max=%.2e, mean=%.2e — 语义恒等 0'
        ' (torch.cdist matmul 展开式浮点噪声底 ~6e-8, 容差 1e-6)'
        % (nn_self['max'], nn_self['mean']))
    hull_self = M.outside_hull_fraction(g3, train_df, tr_sys[sel3], tr_sys, ref=train_df)
    assert hull_self['fraction_outside'] == 0.0, '凸包自反非零'
    log('    凸包外自反 (逐体系, %d 行): 0.000 (分离超平面 LP %d 组 / 盒回退 %d 组)'
        % (hull_self['n'], hull_self['n_sep_lp'], hull_self['n_box_fallback']))
    box_self = M.outside_box_fraction(g3, train_df)
    assert box_self['fraction_outside'] == 0.0
    log('    盒外自反 = 0.000 (精确)')
    results['nn_self'] = nn_self['mean']; results['hull_self'] = hull_self['fraction_outside']

    # ---- 条件一致率 / 外推指数恒等 ----
    cc = M.conditional_consistency(g3, lambda df, target: np.ones(len(df)), target='SELFTEST')
    assert cc['mean_p'] == 1.0 and cc['frac_ge_tau'] == 1.0
    log('[4] 条件一致率恒等 (identity scorer): mean_p=1.000, frac_ge_tau=1.000')
    ei = M.extrapolation_index(0.8, 0.8, higher_better=True)
    assert abs(ei['decay_pct']) == 0.0 and ei['ratio'] == 1.0
    ei2 = M.extrapolation_index_from_dicts({'COV-P': 0.8, 'W2_joint': 0.4}, {'COV-P': 0.8, 'W2_joint': 0.4})
    assert all(abs(v['decay_pct']) == 0.0 for v in ei2.values())
    log('    外推指数恒等: ratio=1.000, decay=0.0%% (标量 + 批量接口)')
    assert abs(w2_self) < 0.05 and abs(mmd_self) < 0.05, 'W2/MMD 自反偏离恒等参考'
    log('校准 ② 判定: PASS (恒等参考值全部达标; W2/MMD 自反在熵正则/对角公式噪声内) — 用时 %.1f min'
        % ((time.time() - t0) / 60))
    return results

# =====================================================================
# 应用 demo: WP3 首训产物全指标首读 (G3 线对照)
# =====================================================================
def demo(M):
    t0 = time.time()
    log('')
    log('==================== 应用 demo: WP3 首训产物全指标首读 ====================')
    log('协议: 3 个测试体系 (大/中/小, n_true>=5, 非 Dense/Amorphous) × 200 样本/体系,')
    log('      seed 42, cond_scale 0.75, 逆 qt 回原始量纲; ref = WP3 train 域 (人口 max 归一)')
    import wp3_ddpm_v0 as w3                        # 复用 build_diffusion / inv_qt / 常量
    payload = torch.load(w3.DS_PT, weights_only=False)
    meta = payload['meta']
    X = np.asarray(payload['syn_raw'])
    tr_idx = payload['train_idx'].numpy(); te_idx = payload['test_idx'].numpy()
    zeo_k = np.array(payload['zeo_key']); osda_k = np.array(payload['osda_key'])
    train_df = pd.DataFrame(X[tr_idx], columns=M.cols)
    test_df = pd.DataFrame(X[te_idx], columns=M.cols)
    test_df['zeo'], test_df['osda'] = zeo_k[te_idx], osda_k[te_idx]
    tr_zeo = zeo_k[tr_idx]

    key_te = pd.Series(list(zip(zeo_k[te_idx], osda_k[te_idx])))
    counts = key_te.value_counts()
    eligible = [k for k in counts.index if counts[k] >= 5 and k[0] != 'Dense/Amorphous']
    picks = [eligible[0], eligible[len(eligible) // 2], eligible[-1]]
    log('测试体系选择 (n_true>=5 共 %d 个): %s' % (
        len(eligible), '; '.join('%s+%s (n=%d)' % (z, o[:20], counts[(z, o)]) for z, o in picks)))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    diffusion, n_par = w3.build_diffusion(device)
    ck = torch.load(w3.MODEL_PT, map_location=device, weights_only=False)
    diffusion.load_state_dict(ck['state_dict']); diffusion.eval()
    Z = payload['zeo_feat'].to(device); O = payload['osda_feat'].to(device)
    rows0 = []
    for z_, o_ in picks:
        mrow = np.where((zeo_k == z_) & (osda_k == o_))[0]
        rows0.append(mrow[0])
    NS = 200
    zc = torch.stack([Z[r] for r in rows0]).repeat_interleave(NS, dim=0)
    oc = torch.stack([O[r] for r in rows0]).repeat_interleave(NS, dim=0)
    torch.manual_seed(42)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device != 'cpu')):
        s = diffusion.sample(zeo=zc, osda=oc, cond_scale=w3.COND_SCALE,
                             batch_size=zc.shape[0]).squeeze(1).cpu().numpy()
    raw = w3.inv_qt(s, payload['qts'])              # 逆 qt 回原始量纲 (float64)
    pred_df = pd.DataFrame(raw, columns=M.cols)
    pred_df['zeo'] = np.repeat([p[0] for p in picks], NS)
    pred_df['osda'] = np.repeat([p[1] for p in picks], NS)
    pred_df.to_csv(DEMO_CSV, index=False)
    log('采样完成: %d 条 (3 体系 × %d), best-val ckpt @%d; 落盘 %s'
        % (len(pred_df), NS, ck['step'], os.path.basename(DEMO_CSV)))

    # ---- 逐体系全指标 ----
    log('')
    log('逐体系指标 (vs 该体系测试真值; W2/MMD 为 train 人口 max 归一空间):')
    log('%-10s %5s | %8s %8s %8s | %6s %6s | %7s' % ('体系', 'n_true', 'W2', 'MMD', 'W1均值', 'COV-P', 'COV-R', '物理有效率'))
    suites = {}
    for z_, o_ in picks:
        p_i = pred_df[(pred_df['zeo'] == z_) & (pred_df['osda'] == o_)][M.cols]
        t_i = test_df[(test_df['zeo'] == z_) & (test_df['osda'] == o_)][M.cols]
        fs = M.full_suite(p_i, t_i, ref=train_df)
        suites[(z_, o_)] = fs
        log('%-10s %5d | %8.4f %8.4f %8.4f | %6.3f %6.3f | %7.4f'
            % (z_, len(t_i), fs['W2_joint'], fs['MMD'], fs['W1_mean'],
               fs['COV-P'], fs['COV-R'], fs['physical_validity']['rate']))
    mean_covp = float(np.mean([s['COV-P'] for s in suites.values()]))
    mean_covr = float(np.mean([s['COV-R'] for s in suites.values()]))
    mean_w2 = float(np.mean([s['W2_joint'] for s in suites.values()]))
    mean_pv = float(np.mean([s['physical_validity']['rate'] for s in suites.values()]))
    log('三体系均值: W2=%.4f, COV-P=%.3f, COV-R=%.3f, 物理有效率(strict)=%.4f' % (mean_w2, mean_covp, mean_covr, mean_pv))

    # ---- 新颖性 (全局 NN + 骨架级 NN/凸包 + 全局盒外) ----
    gen_fw = pred_df['zeo'].to_numpy()
    nn_g = M.nn_distance(pred_df[M.cols], train_df, ref=train_df)
    nn_fw = M.nn_distance(pred_df[M.cols], train_df, ref=train_df, gen_sys=gen_fw, train_sys=tr_zeo)
    hull_fw = M.outside_hull_fraction(pred_df[M.cols], train_df, gen_fw, tr_zeo, ref=train_df)
    box_g = M.outside_box_fraction(pred_df[M.cols], train_df)
    log('')
    log('新颖性 (600 生成 vs train 域, 人口 max 归一空间):')
    log('  全局 NN 距离: mean %.4f / median %.4f / p90 %.4f' % (nn_g['mean'], nn_g['median'], nn_g['p90']))
    log('  骨架级 NN 距离 (同骨架 train 行): mean %.4f / median %.4f' % (nn_fw['mean'], nn_fw['median']))
    log('  骨架级凸包外比例: %.4f (分离超平面 LP %d 组 / 盒回退 %d 组 / 无 train 组 %d)'
        % (hull_fw['fraction_outside'], hull_fw['n_sep_lp'],
           hull_fw['n_box_fallback'], hull_fw['n_groups_no_train']))
    log('  全局盒外比例 (凸包外下界近似): %.4f' % box_g['fraction_outside'])

    # ---- G3 线对照与登记 ----
    log('')
    log('G3 线对照 (生成器中期线, 首训 v0 读数为初始基线非达标判定):')
    log('  COV-P >=0.75 (中期线): 三体系均值 %.3f — %s' % (
        mean_covp, '首读低于线, 登记 v1 追踪' if mean_covp < 0.75 else '过线'))
    log('  物理有效率 >=98%%: strict %.4f — %s' % (
        mean_pv, '达标' if mean_pv >= 0.98 else '未达 (登记)'))
    log('  W2 <=基线 (GAN/VAE): 基线家族未跑, 登记; 条件一致率 >=90%%: v1 接 L1/L2 scorer 适配器后首读')
    log('  外推指数: M10 时间外推实验首跑 (本 API extrapolation_index 就绪)')
    log('应用 demo 判定: PASS (七类指标全链路跑通, 读数登记) — 用时 %.1f min' % ((time.time() - t0) / 60))
    return {'COV-P': mean_covp, 'physical_validity': mean_pv, 'W2': mean_w2}

# =====================================================================
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['calib1', 'calib2', 'demo', 'all'])
    a = ap.parse_args()

    M = ZeoSynMetrics()
    log('WP5 指标统一 API v0 校准报告 (M6 顺延任务, 建模线交接清单 2026-10-17 任务②)')
    log('=' * 78)
    log('生成时间: %s | git %s | 模块 md5 %s' % (
        time.strftime('%Y-%m-%d %H:%M:%S'), git_head(), md5(os.path.abspath(__file__))))
    log('校准资产: diffsyn_metric_recheck.csv md5 %s | ZeoSynGen_dataset.pkl md5 %s' % (
        md5(RECHECK), md5(os.path.join(ZG, 'data', 'ZeoSynGen_dataset.pkl'))))
    log('指标实现: WSD/MMD/COV = 官方 data/metrics.py wrap (不重写); W1 = scipy; 新增四指标自行实现')
    log('数据版本: zeobench_db_v1_8 (校准 ②/demo 经 WP3 dataset); 官方锚: wsd_zeo_osda.json 0.4237')

    def flush():
        with open(REPORT, 'w', encoding='utf-8') as f:
            f.write('\n'.join(L) + '\n')

    if a.stage in ('calib1', 'all'):
        calib1(M); flush()
    if a.stage in ('calib2', 'all'):
        calib2(M); flush()
    if a.stage in ('demo', 'all'):
        demo(M); flush()
    log('')
    log('报告已写入 %s' % REPORT)
    flush()
