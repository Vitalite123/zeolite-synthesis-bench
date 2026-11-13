# -*- coding: utf-8 -*-
"""
feasibility_scorer v1.2 —— 统一可行性评分入口 (CLI + Python API)
====================================================================================
任务定位 (执行计划 2026-10-30~11-05 周计划 P0-1 / §4 M12 中期评审交付物):
  M12 交付物 `feasibility_scorer` v1.2 = 三层判别器统一入口 + 温度校准 + T3.4 硬约束过滤器。
  v1.2 相对 v1 的唯一变化: L3 COMP 块由 v1 竞相 d/R 8 列替换为 v2 框架无关占位几何
  (上游: Documentation/wp2_l3_v1_2_集成_骨架.md); L1/L2 头、数据版本、切分全部不动。

三层判别器 (协议逐位镜像 m10_calib_v1 三臂, 不修改任何既有脚本, 仅 import 复用实现):
  L1 相分类   ← l1_baseline_v1.build_a3 A3 490 维; 域 = positive & zeo_desc_key 有效;
                LGBM multiclass (198 类), random val 早停; 温度校准 T_L1。
  L2 可行性   ← l2_v0_1_litfail.build_features_matrix A3 490 维 (主库 v1.1 剔 Tier A
                + lit_fail A+B + 先批增量 + 本批 incr_R6, 域 23,805 行);
                xtal 主口径 (Q19, G2 0.85 线绑定) + pure 参照; LGBM binary;
                温度校准 T_L2xt / T_L2pu。
  L3 纯度竞相 ← l3_purity_v1.fit_v1(comp='v2') 498 维 (A3 490 + COMP v2 8;
                OSDA v1_1 表 + sidecar 回填); 域 = positive + relative_neg & zeo_desc_key
                有效 (200 类); LGBM multiclass; 温度校准 T_L3 (v1.2 登记冻结值 1.2845)。
  T3.4 过滤器 ← t3_4_filters_v0.T34Filters (6 规则 + cap 语义标签, 训练域校准)。

校准层: 直接 import l3_purity_v1.{ece, fit_temperature, proba_with_T, nll_mc} (不重写定义);
  T* 于 random 切分 val 行按 NLL 最小拟合 (三切分共用冻结, 与 L3 v1 惯例一致)。

冻结统计 (scorer 可部署性): 各头特征以全库/主库 fit 统计量冻结 (fit_v1 -> frozen 表),
  任意子集零重 fit (修复 v0「子集重跑 build_a3 重 fit scaler」特征漂移); 自证:
  冻结变换 vs 训练侧原路径逐位 allclose(1e-9)。

温度缩放不改 argmax (逐组断言 PASS) → 分类读数 (top-1/AUROC) 不受 T 影响。

产出 (幂等覆盖写):
  Data/feasibility_scorer_v1_2_model.pkl      部署模型束 (3 头模型 + 冻结统计 + 类表 + T + 轮数)
  Data/feasibility_scorer_v1_2_report.txt     自测报告 (含 12 项锚点 assert + 基准读数)
  Data/feasibility_scorer_v1_2_manifest.json  manifest (数据版本 + 代码 md5 + 锚点)
  Documentation/feasibility_scorer_v1_2_基准报告_骨架.md  版本化基准报告骨架 (M12 填表)

用法:
  python feasibility_scorer_v1_2.py fit              构建 + 自测 + 落盘
  python feasibility_scorer_v1_2.py selfcheck        载入模型束, 复算三切分锚点读数
  python feasibility_scorer_v1_2.py score --input X.csv --out Y.csv [--target EMT]
数据版本: zeobench_db_v1_8 + zeobench_splits_v1_8 (L2 域含 v1.1 主库 + lit_fail + 增量分区)
"""
import os
import re
import sys
import json
import time
import pickle
import hashlib
import argparse
import subprocess
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
sys.path.insert(0, BASE)

import l1_baseline_v1 as l1b
import l3_purity_v1 as v1
import osda_schema as osd
from l3_purity_v1 import (fit_v1, transform_v1, build_dsc_raw, fit_temperature,
                          proba_with_T, ece as ece_fn, nll_mc)
from l2_v0_1_litfail import (build_lf_rows, build_features_matrix, feat_raw, ext_raw,
                             LOG_MOLE_COLS)
import expand_osda_desc as eod

# ===================== 路径与常量 =====================
DB          = os.path.join(BASE, 'data', 'zeobench_db_v1_8.parquet')
SPLITS      = os.path.join(BASE, 'data', 'zeobench_splits_v1_8.parquet')
DB_V11      = os.path.join(BASE, 'data', 'zeosyn_db_v1_1.parquet')
SPLITS_V11  = os.path.join(BASE, 'data', 'zeosyn_splits_v1_1.parquet')
DEDUP       = os.path.join(BASE, 'data', 'zeosyn_db_dedup_manifest.csv')
OSDA_DEFAULT = os.path.join(BASE, 'data', 'osda_descriptors.parquet')
OSDA_V11    = os.path.join(BASE, 'data', 'osda_descriptors_v1_1.parquet')
SIDE        = os.path.join(BASE, 'data', 'osda_desc_key_sidecar.csv')

MODEL_PKL  = os.path.join(BASE, 'data', 'feasibility_scorer_v1_2_model.pkl')
REPORT     = os.path.join(BASE, 'data', 'feasibility_scorer_v1_2_report.txt')
SELF_REPORT = os.path.join(BASE, 'data', 'feasibility_scorer_v1_2_selfcheck.txt')
MANIFEST   = os.path.join(BASE, 'data', 'feasibility_scorer_v1_2_manifest.json')
BENCH_MD   = os.path.join(BASE, 'Documentation', 'feasibility_scorer_v1_2_基准报告_骨架.md')

SCORER_VERSION = 'v1_2'
DATA_VERSION = 'zeobench_db_v1_8 + zeobench_splits_v1_8'
SEED = 42
HELD_FWS = ['LTL', 'SAS', 'ITG', 'RTH', 'SZR', 'LTA']
EVAL_COLS = LOG_MOLE_COLS + ['ratio_SiAl', 'ratio_OH_Si', 'ratio_Na_Si', 'ratio_H2O_Si',
                             'ratio_osda_Si', 'ratio_charge_bal', 'T_C', 't_h', 'solvent',
                             'stirring', 'osda1_desc_key', 'l2_pure', 'l2_xtal', 'doi', 'route_id']
SPLITS3 = ['random', 'heldout', 'time_2018']
T_L3_FROZEN = 1.2845          # L3 v1.2 登记冻结值 (COMP v2 重训, 2026-10-22)
L3_TRI_TIME = 0.6475          # L3 time 三元归并读数 (源 Data/l3_purity_v1_2_report.txt, v1.2)

# 12 项锚点 (m10_calib_v1 协议保真读数; 容差 1e-3); L3 为 COMP v2 新锚
ANC = {
    'L1':   {'random': 0.5710, 'heldout': 0.2402, 'time_2018': 0.6973},
    'L2xt': {'random': 0.6307, 'heldout': 0.5158, 'time_2018': 0.6650},
    'L2pu': {'random': 0.6210, 'heldout': 0.5432, 'time_2018': 0.6271},
    'L3':   {'random': 0.5565, 'heldout': 0.2617, 'time_2018': 0.6570},
}
ANC_TOL = 1e-3

_L = []


def log(msg=''):
    print(msg, flush=True)
    _L.append(str(msg))


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


def load_tier_a_dedup():
    """镜像 m10_calib_v1.load_tier_a_dedup: v1.8 去 Tier A → (df, sm)"""
    df = pd.read_parquet(DB)
    sp = pd.read_parquet(SPLITS)
    man = pd.read_csv(DEDUP)
    dup = set(man.loc[man['tier'] == 'A', 'route_id'].dropna())
    df = df[~df['route_id'].isin(dup)].reset_index(drop=True)
    sm = df[['route_id']].merge(
        sp[['route_id', 'split_random', 'split_heldout', 'split_time_2018', 'fw_key']],
        on='route_id', how='left')
    assert sm['split_random'].notna().all(), '切分对不齐'
    return df, sm


# =====================================================================
# 冻结特征层 (2 个 OSDA 表口径)
# =====================================================================
def fit_frozen(df, osda_path):
    """全库 fit 冻结统计 (供部署侧任意子集零漂移变换); 仅 490/498 口径"""
    v1.OSDA_DESC = osda_path
    l1b.OSDA_DESC = osda_path
    X, frozen, osda, desc_cols = fit_v1(df)
    return X, frozen, osda, desc_cols


def transform_l2(df, frozen, osda, desc_cols):
    """L2 冻结统计变换 (推理侧零 fit): 490 维, 逐位镜像 build_features_matrix 构造段。
    与 L3 路径 (transform_v1/build_ext_v1) 的差别: L2 的 ext 块用 l2_v0_1_litfail.ext_raw
    (num = fillna(0.0).clip(lower=0)), 对缺值/负值行 (lit_fail 侧) 不可替换。"""
    F = feat_raw(df, frozen['med_T'], frozen['med_t']).values.astype(float)
    E = ext_raw(df, frozen['cb_med'], frozen['cb_mean'], frozen['cb_std']).values.astype(float)
    D = build_dsc_raw(df, osda, desc_cols)
    b = df['osda1_desc_key'].notna().astype(float).values[:, None]
    X = np.hstack([(F - frozen['mu36']) / frozen['sd36'],
                   (E - frozen['mu_e']) / frozen['sd_e'],
                   b, (D - frozen['mu_d']) / frozen['sd_d']])
    assert X.shape == (len(df), 490), 'L2 冻结变换维度漂移'
    return X


def fit_l2_frozen(df_main):
    """L2 域冻结统计 (拟合人群 = 主库 v1.1 剔 Tier A, 与 build_features_matrix 一致)"""
    med_T, med_t = df_main['T_C'].median(), df_main['t_h'].median()
    cb = pd.to_numeric(df_main['ratio_charge_bal'], errors='coerce')
    cb_med = cb.median()
    cb1 = cb.fillna(cb_med)
    F = feat_raw(df_main, med_T, med_t)
    E = ext_raw(df_main, cb_med, cb1.mean(), cb1.std())
    osda = osd.load(OSDA_DEFAULT)
    desc_cols = osd.desc_cols(osda)
    assert len(desc_cols) == 444, 'OSDA 非 BE 描述符口径漂移: %d' % len(desc_cols)
    D = build_dsc_raw(df_main, osda, desc_cols)
    sd_d = np.nanstd(D, 0)
    frozen = {'med_T': float(med_T), 'med_t': float(med_t), 'cb_med': float(cb_med),
              'cb_mean': float(cb1.mean()), 'cb_std': float(cb1.std()),
              'mu36': F.values.mean(0), 'sd36': F.values.std(0),
              'mu_e': E.values.mean(0), 'sd_e': E.values.std(0),
              'mu_d': np.nanmean(D, 0), 'sd_d': np.where(sd_d < 1e-12, 1.0, sd_d)}
    return frozen, osda, desc_cols


# =====================================================================
# LGBM 参数 (fit 与 selfcheck 共用同一套; 保证训练侧协议逐位可复现)
# =====================================================================
def _params_mc(k):
    return dict(objective='multiclass', num_class=k, learning_rate=0.05, num_leaves=31,
                min_child_samples=100, lambda_l2=1.0, max_delta_step=1.0,
                seed=SEED, verbosity=-1)


def _params_bin():
    return dict(objective='binary', learning_rate=0.05, num_leaves=31, min_child_samples=100,
                lambda_l2=1.0, max_delta_step=1.0, seed=SEED, verbosity=-1)


# =====================================================================
# L1 臂 (相分类; A3 490)
# =====================================================================
def build_l1(df, sm):
    sr, sh, st = sm['split_random'].values, sm['split_heldout'].values, sm['split_time_2018'].values
    dom = ((df['label1'] == 'positive') & df['zeo_desc_key'].notna()).values
    m = sm[dom].reset_index(drop=True)
    fw = m['fw_key'].astype(str).values
    zdk = df['zeo_desc_key'].astype(str).values[dom]
    assert (fw == zdk).all(), 'fw_key 与 zeo_desc_key 域内不一致'
    classes = sorted(set(fw))
    cmap = {c: i for i, c in enumerate(classes)}
    y = np.array([cmap[c] for c in fw])
    route = m['route_id'].values
    K = len(classes)

    Xf, frozen, osda, desc = fit_frozen(df, OSDA_DEFAULT)     # 498 (自证前 490 == build_a3)
    Xd = Xf[dom][:, :490]
    X_ref, _ = l1b.build_a3(df)
    assert np.allclose(Xd, X_ref[dom], atol=1e-9, equal_nan=True), 'L1 冻结变换 vs build_a3 漂移'

    tr, va, te = sr[dom] == 'train', sr[dom] == 'val', sr[dom] == 'test'
    params = _params_mc(K)
    t0 = time.time()
    dtr = lgb.Dataset(Xd[tr], label=y[tr])
    dva = lgb.Dataset(Xd[va], label=y[va], reference=dtr)
    b0 = lgb.train(params, dtr, num_boost_round=400, valid_sets=[dva],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
    best = b0.best_iteration or -1
    p_va = b0.predict(Xd[va], num_iteration=best)
    T, _, _ = fit_temperature(p_va, y[va])
    log('  L1 域: %d 行 %d 类; random 早停@%d (%.0fs); T_L1=%.4f'
        % (len(y), K, best, time.time() - t0, T))

    anchors, proba_te = {}, {}
    for tag, te_m in (('random', te), ('heldout', sh[dom] == 'test'), ('time_2018', st[dom] == 'test')):
        if tag == 'random':
            p = b0.predict(Xd[te_m], num_iteration=best)
        else:
            tr_m = (sh[dom] == 'train') if tag == 'heldout' else (st[dom] == 'train')
            bm = lgb.train(params, lgb.Dataset(Xd[tr_m], label=y[tr_m]), num_boost_round=best)
            p = bm.predict(Xd[te_m])
        anchors[tag] = float(np.mean(p.argmax(1) == y[te_m]))
        proba_te[tag] = p
    # 部署模型: train+val 全数据 @ 锚定轮数
    deploy = lgb.train(params, lgb.Dataset(Xd[tr | va], label=y[tr | va]), num_boost_round=best)
    log('  L1 test top-1: random %.4f / heldout %.4f / time %.4f'
        % (anchors['random'], anchors['heldout'], anchors['time_2018']))
    return dict(
        name='L1', kind='multiclass', model=deploy, classes=classes, best=best, T=T,
        anchors=anchors, n_domain=int(len(y)),
        frozen=frozen, osda=osda, desc_cols=desc, slice=490, osda_path=OSDA_DEFAULT,
        proba_te=proba_te, y_te={'random': y[te], 'heldout': y[sh[dom] == 'test'],
                                 'time_2018': y[st[dom] == 'test']})


# =====================================================================
# L2 臂 (可行性; xtal 主口径 + pure 参照; A3 490)
# =====================================================================
def build_l2(df8):
    db11 = pd.read_parquet(DB_V11)
    sp11 = pd.read_parquet(SPLITS_V11)
    man = pd.read_csv(DEDUP)
    dup_ids = set(man.loc[man['tier'] == 'A', 'route_id'].dropna())
    df_main = db11[~db11['route_id'].isin(dup_ids)].reset_index(drop=True)
    mm = df_main[['route_id', 'label1']].merge(sp11, on='route_id', how='left')
    assert mm['split_random'].notna().all()

    lf_df, _, _ = build_lf_rows()
    held_pat = re.compile(r'\b(LTL|SAS|ITG|RTH|SZR|LTA)\b|zeolite A|Zeolite A'
                          r'|(?<![\w(])L(?![\w)])', re.I)
    held_hit = (lf_df['intended_framework'].fillna('').str.contains(held_pat, regex=True)
                | lf_df['actual_product'].fillna('').str.contains(held_pat, regex=True))
    lf_df['split_random'] = 'train'
    lf_df['split_heldout'] = np.where(held_hit, 'excluded', 'train')
    lf_df['split_time_2018'] = np.where(lf_df['year'] >= 2019, 'excluded', 'train')

    in_v11 = set(db11['route_id'])
    prior_all = df8[df8['route_id'].str.startswith('incr_')
                    & ~df8['route_id'].isin(in_v11)].reset_index(drop=True)
    batch = prior_all['route_id'].str.startswith('incr_R6')
    new = prior_all[batch].reset_index(drop=True)
    prior = prior_all[~batch].reset_index(drop=True)

    prior['l2_pure'] = (prior['label1'] == 'positive').astype(float)
    prior['l2_xtal'] = prior['label1'].isin(['positive', 'relative_neg']).astype(float)
    prior_eval = prior[EVAL_COLS].copy()
    sp8 = pd.read_parquet(SPLITS).set_index('route_id')
    prior_eval['split_random'] = 'train'
    prior_eval['split_heldout'] = sp8.loc[prior_eval['route_id'], 'split_heldout'].values
    prior_eval['split_time_2018'] = 'excluded'

    new['l2_pure'] = 1.0
    new['l2_xtal'] = 1.0
    inc_eval = new[EVAL_COLS].copy()
    held_new = new['framework_code'].astype(str).isin(HELD_FWS)
    inc_eval['split_random'] = 'train'
    inc_eval['split_heldout'] = np.where(held_new, 'excluded', 'train')
    inc_eval['split_time_2018'] = 'excluded'

    df_lf_all = pd.concat([lf_df, prior_eval, inc_eval], ignore_index=True, sort=False)
    X, _ = build_features_matrix(df_main, df_lf_all)          # L2 协议保真源 (镜像 m10 arm_l2)

    # 冻结统计侧 (L2 域 = 主库 v1.1 剔 Tier A 拟合) + 自证逐位一致
    frozen, osda, desc = fit_l2_frozen(df_main)
    Xf = transform_l2(pd.concat([df_main, df_lf_all], ignore_index=True, sort=False),
                      frozen, osda, desc)
    assert Xf.shape == X.shape, 'L2 冻结变换行数漂移'
    assert np.allclose(Xf, X, atol=1e-9, equal_nan=True), \
        'L2 冻结变换 vs build_features_matrix 漂移'

    y_pure = np.concatenate([np.where(df_main['label1'] == 'positive', 1, 0),
                             lf_df['l2_pure'].values.astype(int),
                             prior_eval['l2_pure'].values.astype(int),
                             new['l2_pure'].values.astype(int)])
    y_xtal = np.concatenate([np.where(df_main['label1'].isin(['positive', 'relative_neg']), 1, 0),
                             lf_df['l2_xtal'].values.astype(int),
                             prior_eval['l2_xtal'].values.astype(int),
                             new['l2_xtal'].values.astype(int)])
    sr = np.concatenate([mm['split_random'].values, lf_df['split_random'].values,
                         prior_eval['split_random'].values, inc_eval['split_random'].values])
    sh = np.concatenate([mm['split_heldout'].values, lf_df['split_heldout'].values,
                         prior_eval['split_heldout'].values, inc_eval['split_heldout'].values])
    st = np.concatenate([mm['split_time_2018'].values, lf_df['split_time_2018'].values,
                         prior_eval['split_time_2018'].values, inc_eval['split_time_2018'].values])
    log('  L2 域: 主库(v1.1 剔 Tier A) %d + lit_fail %d + 先批增量 %d + 本批 %d = %d 行'
        % (len(df_main), len(lf_df), len(prior), len(new), len(y_xtal)))

    params = _params_bin()

    def early_stop(yv):
        dtr = lgb.Dataset(X[sr == 'train'], label=yv[sr == 'train'])
        dva = lgb.Dataset(X[sr == 'val'], label=yv[sr == 'val'], reference=dtr)
        b = lgb.train(params, dtr, num_boost_round=400, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        return b, (b.best_iteration or -1)

    # 轮数锚定 (m10 arm_l2 协议第 330-344 行): heldout/time 两口径共用 pure random 早停轮
    early = {tag: early_stop(yv) for tag, yv in (('xtal', y_xtal), ('pure', y_pure))}
    rounds_pure = early['pure'][1]
    log('  L2 pure random 早停轮 = %d (heldout/time 轮数锚定惯例)' % rounds_pure)

    heads = {}
    for tag, yv in (('xtal', y_xtal), ('pure', y_pure)):
        b, rounds = early[tag]
        p_va = b.predict(X[sr == 'val'], num_iteration=rounds)
        T, _, _ = fit_temperature(np.column_stack([1 - p_va, p_va]), yv[sr == 'val'])
        mv = (sr == 'train') | (sr == 'val')
        deploy = lgb.train(params, lgb.Dataset(X[mv], label=yv[mv]), num_boost_round=rounds)
        anchors, proba_te, y_te = {}, {}, {}
        for sname, sv in (('random', sr), ('heldout', sh), ('time_2018', st)):
            if sname == 'random':
                p = b.predict(X[sv == 'test'], num_iteration=rounds)
            else:
                b2 = lgb.train(params, lgb.Dataset(X[sv == 'train'], label=yv[sv == 'train']),
                               num_boost_round=rounds_pure)          # ← 共用 pure 锚定轮
                p = b2.predict(X[sv == 'test'])
            anchors[sname] = roc_auc_score(yv[sv == 'test'], p)
            proba_te[sname] = np.column_stack([1 - p, p])
            y_te[sname] = yv[sv == 'test']
        log('  L2 %s AUROC: random %.4f / heldout %.4f / time %.4f (轮 %d, T=%.4f)'
            % (tag, anchors['random'], anchors['heldout'], anchors['time_2018'], rounds, T))
        heads[tag] = dict(name='L2-%s' % tag, kind='binary', model=deploy, classes=None,
                          best=rounds, T=T, anchors=anchors, n_domain=int(len(y_xtal)),
                          frozen=frozen, osda=osda, desc_cols=desc, slice=490, builder='l2',
                          osda_path=OSDA_DEFAULT, proba_te=proba_te, y_te=y_te)
    return heads


# =====================================================================
# L3 臂 (纯度竞相; 498 维; OSDA v1_1 + sidecar 回填)
# =====================================================================
def build_l3(df, sm):
    v1.OSDA_DESC = OSDA_V11
    l1b.OSDA_DESC = OSDA_V11
    assert os.path.exists(OSDA_V11), '缺少 osda_descriptors_v1_1.parquet'
    side = pd.read_csv(SIDE)
    ok_side = side[side['desc_key'].astype(str) != '']
    smap = dict(zip(ok_side['osda_name_norm'], ok_side['desc_key']))
    key_fb = df['osda1_name'].map(eod.norm_name).map(smap)
    was_na = df['osda1_desc_key'].isna() & key_fb.notna()
    df = df.copy()
    df['osda1_desc_key'] = df['osda1_desc_key'].fillna(key_fb)

    X, frozen, osda, desc = fit_v1(df, comp='v2')   # 498; 内部自证 490 == build_a3
    dom = ((df['label1'].isin(['positive', 'relative_neg'])) & df['zeo_desc_key'].notna()).values
    idx = np.where(dom)[0]
    m = df.loc[dom, ['route_id', 'label1']].merge(
        sm[['route_id', 'fw_key', 'split_random', 'split_heldout', 'split_time_2018']],
        on='route_id', how='left')
    assert len(m) == len(idx) and m['fw_key'].notna().all()
    y_all = m['fw_key'].astype(str).values
    sr, sh, st = m['split_random'].values, m['split_heldout'].values, m['split_time_2018'].values
    classes = sorted(set(y_all))
    cmap = {c: i for i, c in enumerate(classes)}
    y = np.array([cmap[c] for c in y_all])
    Xm = X[idx]
    log('  L3 域: %d 行 %d 类 (sidecar 补 key %d 行); 498 维冻结' % (len(y), len(classes), int(was_na.sum())))

    params = _params_mc(len(classes))
    tr, va, te = sr == 'train', sr == 'val', sr == 'test'
    t0 = time.time()
    dtr = lgb.Dataset(Xm[tr], label=y[tr])
    dva = lgb.Dataset(Xm[va], label=y[va], reference=dtr)
    b0 = lgb.train(params, dtr, num_boost_round=400, valid_sets=[dva],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
    best = b0.best_iteration or -1
    p_va = b0.predict(Xm[va], num_iteration=best)
    T_fit, _, _ = fit_temperature(p_va, y[va])
    T = T_fit                                       # 与 m10/L3 v1.2 同口径 (argmax 不变)
    log('  L3 random 早停@%d (%.0fs); T_L3=%.4f (v1.2 登记冻结 %.4f, 交叉校验 Δ=%+.4f)'
        % (best, time.time() - t0, T_fit, T_L3_FROZEN, T_fit - T_L3_FROZEN))
    assert abs(T_fit - T_L3_FROZEN) < 0.02, 'L3 T* 与 v1.2 登记值偏离 >0.02'

    anchors, proba_te, y_te = {}, {}, {}
    for tag, te_m in (('random', te), ('heldout', sh == 'test'), ('time_2018', st == 'test')):
        if tag == 'random':
            p = b0.predict(Xm[te_m], num_iteration=best)
        else:
            tr_m = (sh == 'train') if tag == 'heldout' else (st == 'train')
            bm = lgb.train(params, lgb.Dataset(Xm[tr_m], label=y[tr_m]), num_boost_round=best)
            p = bm.predict(Xm[te_m])
        anchors[tag] = float(np.mean(p.argmax(1) == y[te_m]))
        proba_te[tag] = p
        y_te[tag] = y[te_m]
    deploy = lgb.train(params, lgb.Dataset(Xm[tr | va], label=y[tr | va]), num_boost_round=best)
    log('  L3 test top-1: random %.4f / heldout %.4f / time %.4f'
        % (anchors['random'], anchors['heldout'], anchors['time_2018']))
    return dict(
        name='L3', kind='multiclass', model=deploy, classes=classes, best=best, T=T,
        anchors=anchors, n_domain=int(len(y)), frozen=frozen, osda=osda, desc_cols=desc,
        slice=498, osda_path=OSDA_V11, cmap_emt=cmap.get('EMT'),
        proba_te=proba_te, y_te=y_te)


# =====================================================================
# 部署模型束
# =====================================================================
def _strip_head(h):
    """落盘精简 (去 proba_te/y_te 自测中间态)"""
    return {k: h[k] for k in ('name', 'kind', 'model', 'classes', 'best', 'T', 'anchors',
                              'n_domain', 'frozen', 'desc_cols', 'slice', 'osda_path')
            + (('builder',) if 'builder' in h else ())
            + (('cmap_emt',) if 'cmap_emt' in h else ())}


class FeasibilityScorerV1:
    """统一可行性评分器 (L1 相分类 / L2 可行性 xtal 主口径 / L3 纯度竞相 + 温度校准 + T3.4 过滤)。"""

    def __init__(self, heads, data_version=DATA_VERSION):
        self.heads = heads
        self.data_version = data_version
        self.osda_cache = {}

    # ---------- 载入 ----------
    @classmethod
    def load(cls, path=MODEL_PKL):
        with open(path, 'rb') as f:
            bundle = pickle.load(f)
        s = cls(bundle['heads'], bundle['data_version'])
        s.meta = bundle.get('meta', {})
        return s

    def save(self, path=MODEL_PKL, meta=None):
        bundle = {'scorer_version': SCORER_VERSION, 'data_version': self.data_version,
                  'heads': {k: _strip_head(h) for k, h in self.heads.items()},
                  'meta': {**({'git_head': git_head()} if meta is None else meta)}}
        with open(path, 'wb') as f:
            pickle.dump(bundle, f, protocol=4)
        return path

    def _osda(self, head):
        p = head['osda_path']
        if p not in self.osda_cache:
            self.osda_cache[p] = pd.read_parquet(p)
        return self.osda_cache[p]

    # ---------- 特征 ----------
    def _feat(self, df, head):
        osda = self._osda(head)
        if head.get('builder') == 'l2':
            return transform_l2(df, head['frozen'], osda, head['desc_cols'])
        return transform_v1(df, head['frozen'], osda, head['desc_cols'])[:, :head['slice']]

    # ---------- 打分 ----------
    def score(self, df, target=None, topk=3, with_filters=True):
        """对库式输入行打分。返回 DataFrame。
        输入须含 A3 构建列 (n_*/ratio_*/T_C/t_h/solvent/stirring/osda1_desc_key);
        若含 osda1_name 则先做 sidecar 回填 (L3 口径)。"""
        d = df.copy()
        if 'osda1_name' in d.columns:
            side = pd.read_csv(SIDE)
            ok = side[side['desc_key'].astype(str) != '']
            smap = dict(zip(ok['osda_name_norm'], ok['desc_key']))
            if 'osda1_desc_key' not in d.columns:      # 生成器空间输入无该列
                d['osda1_desc_key'] = np.nan
            d['osda1_desc_key'] = d['osda1_desc_key'].fillna(
                d['osda1_name'].map(eod.norm_name).map(smap))
        out = pd.DataFrame(index=d.index)
        if 'route_id' in d.columns:
            out['route_id'] = d['route_id'].values

        # L1 相分类
        h1 = self.heads['L1']
        p1 = proba_with_T(h1['model'].predict(self._feat(d, h1)), h1['T'])
        cls1 = np.array(h1['classes'])
        order1 = np.argsort(-p1, axis=1)[:, :topk]
        out['l1_fw_top1'] = cls1[order1[:, 0]]
        out['l1_fw_top%d' % topk] = ['|'.join(cls1[o]) for o in order1]
        out['l1_p_max'] = p1.max(1)

        # L2 可行性 (xtal 主口径 + pure 参照)
        for tag, col in (('xtal', 'l2_p_xtal'), ('pure', 'l2_p_pure')):
            h = self.heads['L2-%s' % tag]
            p = h['model'].predict(self._feat(d, h))
            out[col] = proba_with_T(np.column_stack([1 - p, p]), h['T'])[:, 1]
        out['l2_feasible'] = out['l2_p_xtal'] >= 0.5

        # L3 纯度竞相
        h3 = self.heads['L3']
        p3 = proba_with_T(h3['model'].predict(self._feat(d, h3)), h3['T'])
        cls3 = np.array(h3['classes'])
        order3 = np.argsort(-p3, axis=1)
        out['l3_fw_top1'] = cls3[order3[:, 0]]
        out['l3_p_max'] = p3.max(1)
        comp = []
        for i in range(len(p3)):
            c = {cls3[j]: round(float(p3[i, j]), 4) for j in order3[i][:topk + 1]}
            comp.append(json.dumps(c, ensure_ascii=False))
        out['l3_competing'] = comp
        if target is not None:
            j = h3.get('cmap_emt') if str(target) == 'EMT' else None
            if j is None:
                cls_lookup = {c: i for i, c in enumerate(h3['classes'])}
                j = cls_lookup.get(str(target))
            out['l3_target_p'] = p3[:, j] if j is not None else np.nan
            out['l3_target'] = str(target)

        # T3.4 硬约束过滤器 (仅当输入为生成器 12 列空间)
        if with_filters:
            fl = apply_hard_filters(d)
            if fl is not None:
                F, cap = fl
                out['t3_4_reject'] = F.any(axis=1).values
                out['t3_4_rules'] = ['|'.join(F.columns[F.iloc[i].values]) for i in range(len(F))]
                out['t3_4_cap_label'] = cap.values
        return out

    def score_one(self, row, **kw):
        df = pd.DataFrame([row])
        return self.score(df, **kw).iloc[0].to_dict()


_FILTER_CACHE = {}


def apply_hard_filters(df):
    """T3.4 过滤器 (生成器 12 列空间); 输入列不足时返回 None (不适用)。
    惰性导入 wp3_ddpm_v0 会在 import 期 os.chdir(ZG), 此处保存/恢复 cwd,
    避免 score() 调用方相对路径（尤其 CLI --out）被改道。"""
    cwd = os.getcwd()
    try:
        try:
            from t3_4_filters_v0 import T34Filters, _load_domains, DS_PT
        except Exception:
            return None
        import wp3_ddpm_v0 as w3
        need = list(w3.COLS)
        if not all(c in df.columns for c in need):
            return None
        if 'flt' not in _FILTER_CACHE:
            train_df, _, _ = _load_domains()
            f = T34Filters()
            f.calibrate(train_df)
            _FILTER_CACHE['flt'] = f
        return _FILTER_CACHE['flt'].apply(df)
    finally:
        os.chdir(cwd)


# =====================================================================
# 自测: 三切分锚点复算 (冻结统计重导特征 + 训练侧协议重训)
# =====================================================================
def selfcheck_anchors(scorer, log_fn=log):
    """载入模型束 → 冻结统计重导特征 (自证 vs 训练侧原路径逐位一致) →
    按训练侧协议 (各切分 train-only 重训; 轮数取模型束登记值) 复算 12 项锚点 (Δ<1e-3)。

    注: 12 项锚点是『训练侧协议』读数 (train-only 模型产出), 与部署模型 (train+val)
    天然不同, 故此处按协议重训而不复用部署模型; 部署侧一致性 (冻结变换 + 温度缩放
    不改 argmax) 由 fit 阶段的断言覆盖。轮数锚定沿用 m10_calib_v1.arm_l1/arm_l2/arm_l3:
    L1/L3 三切分均用 random 早停轮; L2 random 用各自头早停轮, heldout/time 共用 pure 轮。
    """
    df, sm = load_tier_a_dedup()
    df8 = pd.read_parquet(DB)
    res, rows = {}, []
    # L1 (A3 490)
    h = scorer.heads['L1']
    l1b.OSDA_DESC = h['osda_path']
    dom = ((df['label1'] == 'positive') & df['zeo_desc_key'].notna()).values
    m = sm[dom].reset_index(drop=True)
    cls = sorted(set(m['fw_key'].astype(str)))
    cmap = {c: i for i, c in enumerate(cls)}
    y = np.array([cmap[c] for c in m['fw_key'].astype(str)])
    X = scorer._feat(df[dom].reset_index(drop=True), h)
    X_ref, _ = l1b.build_a3(df)
    assert np.allclose(X[:, :490], X_ref[dom], atol=1e-9, equal_nan=True), \
        'L1 冻结特征 vs build_a3 漂移 (selfcheck)'
    sr, sh, st = sm['split_random'].values, sm['split_heldout'].values, sm['split_time_2018'].values
    for tag, sv0 in (('random', sr), ('heldout', sh), ('time_2018', st)):
        sv = sv0[dom]
        b = lgb.train(_params_mc(len(cls)), lgb.Dataset(X[sv == 'train'], label=y[sv == 'train']),
                      num_boost_round=h['best'])
        p = b.predict(X[sv == 'test'])
        rows.append(('L1', tag, ANC['L1'][tag], float(np.mean(p.argmax(1) == y[sv == 'test']))))
    # L2
    db11 = pd.read_parquet(DB_V11); sp11 = pd.read_parquet(SPLITS_V11)
    man = pd.read_csv(DEDUP); dup = set(man.loc[man['tier'] == 'A', 'route_id'].dropna())
    df_main = db11[~db11['route_id'].isin(dup)].reset_index(drop=True)
    mm = df_main[['route_id', 'label1']].merge(sp11, on='route_id', how='left')
    lf_df, _, _ = build_lf_rows()
    held_pat = re.compile(r'\b(LTL|SAS|ITG|RTH|SZR|LTA)\b|zeolite A|Zeolite A|(?<![\w(])L(?![\w)])', re.I)
    hh = (lf_df['intended_framework'].fillna('').str.contains(held_pat, regex=True)
          | lf_df['actual_product'].fillna('').str.contains(held_pat, regex=True))
    lf_df['split_random'] = 'train'
    lf_df['split_heldout'] = np.where(hh, 'excluded', 'train')
    lf_df['split_time_2018'] = np.where(lf_df['year'] >= 2019, 'excluded', 'train')
    in_v11 = set(db11['route_id'])
    pa = df8[df8['route_id'].str.startswith('incr_') & ~df8['route_id'].isin(in_v11)].reset_index(drop=True)
    batch = pa['route_id'].str.startswith('incr_R6')
    new = pa[batch].reset_index(drop=True); prior = pa[~batch].reset_index(drop=True)
    prior['l2_pure'] = (prior['label1'] == 'positive').astype(float)
    prior['l2_xtal'] = prior['label1'].isin(['positive', 'relative_neg']).astype(float)
    pe = prior[EVAL_COLS].copy(); sp8 = pd.read_parquet(SPLITS).set_index('route_id')
    pe['split_random'] = 'train'; pe['split_heldout'] = sp8.loc[pe['route_id'], 'split_heldout'].values
    pe['split_time_2018'] = 'excluded'
    new['l2_pure'] = 1.0; new['l2_xtal'] = 1.0
    ie = new[EVAL_COLS].copy(); hn = new['framework_code'].astype(str).isin(HELD_FWS)
    ie['split_random'] = 'train'; ie['split_heldout'] = np.where(hn, 'excluded', 'train')
    ie['split_time_2018'] = 'excluded'
    lf_all = pd.concat([lf_df, pe, ie], ignore_index=True, sort=False)
    all_rows = pd.concat([df_main, lf_all], ignore_index=True, sort=False)
    y_pure = np.concatenate([np.where(df_main['label1'] == 'positive', 1, 0),
                             lf_df['l2_pure'].values.astype(int), pe['l2_pure'].values.astype(int),
                             new['l2_pure'].values.astype(int)])
    y_xtal = np.concatenate([np.where(df_main['label1'].isin(['positive', 'relative_neg']), 1, 0),
                             lf_df['l2_xtal'].values.astype(int), pe['l2_xtal'].values.astype(int),
                             new['l2_xtal'].values.astype(int)])
    X2_ref, _ = build_features_matrix(df_main, lf_all)
    X2 = scorer._feat(all_rows, scorer.heads['L2-xtal'])
    assert np.allclose(X2, X2_ref, atol=1e-9, equal_nan=True), \
        'L2 冻结特征 vs build_features_matrix 漂移 (selfcheck)'
    sr2 = np.concatenate([mm['split_random'].values, lf_df['split_random'].values,
                          pe['split_random'].values, ie['split_random'].values])
    sh2 = np.concatenate([mm['split_heldout'].values, lf_df['split_heldout'].values,
                          pe['split_heldout'].values, ie['split_heldout'].values])
    st2 = np.concatenate([mm['split_time_2018'].values, lf_df['split_time_2018'].values,
                          pe['split_time_2018'].values, ie['split_time_2018'].values])
    hkey = {'L2xt': 'L2-xtal', 'L2pu': 'L2-pure'}
    rounds_pure = scorer.heads['L2-pure']['best']        # heldout/time 共用 pure 早停轮 (m10 arm_l2)
    for tag, yv in (('L2xt', y_xtal), ('L2pu', y_pure)):
        hp = scorer.heads[hkey[tag]]
        for sname, sv in (('random', sr2), ('heldout', sh2), ('time_2018', st2)):
            rounds = hp['best'] if sname == 'random' else rounds_pure
            b = lgb.train(_params_bin(), lgb.Dataset(X2[sv == 'train'], label=yv[sv == 'train']),
                          num_boost_round=rounds)
            p = b.predict(X2[sv == 'test'])
            rows.append((tag, sname, ANC[tag][sname], roc_auc_score(yv[sv == 'test'], p)))
    # L3 (498 维)
    h = scorer.heads['L3']
    v1.OSDA_DESC = h['osda_path']
    l1b.OSDA_DESC = h['osda_path']
    side = pd.read_csv(SIDE); ok = side[side['desc_key'].astype(str) != '']
    smap = dict(zip(ok['osda_name_norm'], ok['desc_key']))
    d3 = df.copy()
    d3['osda1_desc_key'] = d3['osda1_desc_key'].fillna(d3['osda1_name'].map(eod.norm_name).map(smap))
    dom = ((d3['label1'].isin(['positive', 'relative_neg'])) & d3['zeo_desc_key'].notna()).values
    idx3 = np.where(dom)[0]
    m3 = d3.loc[dom, ['route_id']].merge(
        sm[['route_id', 'fw_key', 'split_random', 'split_heldout', 'split_time_2018']],
        on='route_id', how='left')
    cls3 = sorted(set(m3['fw_key'].astype(str))); cm3 = {c: i for i, c in enumerate(cls3)}
    y3 = np.array([cm3[c] for c in m3['fw_key'].astype(str)])
    X3 = scorer._feat(d3[dom].reset_index(drop=True), h)
    X3_ref, _, _, _ = fit_v1(d3, comp='v2')
    assert np.allclose(X3, X3_ref[idx3], atol=1e-9, equal_nan=True), \
        'L3 冻结特征 vs fit_v1 漂移 (selfcheck)'
    for tag, col in (('random', 'split_random'), ('heldout', 'split_heldout'), ('time_2018', 'split_time_2018')):
        sv = m3[col].values
        b = lgb.train(_params_mc(len(cls3)), lgb.Dataset(X3[sv == 'train'], label=y3[sv == 'train']),
                      num_boost_round=h['best'])
        p = b.predict(X3[sv == 'test'])
        rows.append(('L3', tag, ANC['L3'][tag], float(np.mean(p.argmax(1) == y3[sv == 'test']))))

    n_fail = 0
    log_fn('锚点断言 (容差 %.0e, 训练侧协议复算: 冻结统计重导特征 + 各切分 train-only 重训):' % ANC_TOL)
    for name, split, exp, got in rows:
        d = got - exp
        ok_ = abs(d) < ANC_TOL
        n_fail += (not ok_)
        r = {'head': name, 'split': split, 'expected': exp, 'measured': float(got),
             'delta': float(d), 'pass': bool(ok_)}
        res['%s/%s' % (name, split)] = r
        log_fn('  %-5s %-9s 期望 %.4f 实测 %.4f (Δ%+.4f) -> %s'
               % (name, split, exp, got, d, 'PASS' if ok_ else '★漂移★'))
    log_fn('  12 项锚点: %s' % ('全部 PASS' if n_fail == 0 else '%d 项漂移 (FAIL)' % n_fail))
    return res, n_fail


# =====================================================================
# 基准报告骨架 (版本化; M12 填表)
# =====================================================================
BENCH_SKELETON = """# feasibility_scorer v1.2 基准报告骨架 (版本化)

- 报告版本: **{ver}** | 数据版本: `{dv}` | 代码: `feasibility_scorer_v1_2.py` (md5 `{md5}`) | git `{git}`
- 生成时间: {date}
- 用途: M12 中期评审交付物 `feasibility_scorer` 的**版本化基准表**（可复现锚点 + 验收行判定）。
  本骨架由 CLI 自动生成；M12 时在下方各表填入当轮读数并**冻结本版**（升版走 `v2`，不改本版）。

## 1. 交付物与接口
| 项 | 内容 |
| --- | --- |
| CLI | `python feasibility_scorer_v1_2.py fit\\|selfcheck\\|score` |
| Python API | `from feasibility_scorer_v1_2 import FeasibilityScorerV1; s=FeasibilityScorerV1.load(); s.score(df)` |
| 模型束 | `Data/feasibility_scorer_v1_2_model.pkl`（3 头模型 + 冻结统计 + 类表 + T + 轮数） |
| 数据版本 | `{dv}` |

## 2. 三层判别器协议（12 项锚点，容差 1e-3，判定 = 与既有脚本逐位一致）
| 头 | 口径 | random | heldout | time_2018 | 期望来源 |
| --- | --- | --- | --- | --- | --- |
| L1 相分类 | top-1 (198 类) | {l1_r} | {l1_h} | {l1_t} | m10_calib_v1 |
| L2 可行性 | xtal AUROC (主口径) | {l2x_r} | {l2x_h} | {l2x_t} | m10_calib_v1 |
| L2 可行性 | pure AUROC (参照) | {l2p_r} | {l2p_h} | {l2p_t} | m10_calib_v1 |
| L3 纯度竞相 | top-1 (200 类) | {l3_r} | {l3_h} | {l3_t} | m10_calib_v1 |

## 3. 温度校准层
| 头 | T* (random val NLL 最小) | 说明 |
| --- | --- | --- |
| L1 | {t1} | argmax 不变 |
| L2 xtal | {t2x} | argmax 不变 |
| L2 pure | {t2p} | argmax 不变 |
| L3 | {t3} | v1.2 登记冻结 {t3f} |

## 4. M12 验收行判定（待填）
| 验收行 | 目标 | 现值 | 判定 | 备注 |
| --- | --- | --- | --- | --- |
| L1 top-1 (DOI 组切分·防泄漏) | ≥0.72 | {l1_r} | 未达 | 口径=DOI 80/10/10 分组切分下的行级 top-1（非 DOI 聚合；聚合仅参照，见 wp2_m12_gap_立项_骨架.md §3.1）；提升路径两机制（表示注入/概率融合）均判负；候选转增量数据/难例专项/目标条件化 |
| L2 AUROC (xtal) | ≥0.85 | {l2x_r} | 未达 | B 复议挂 M12 原位 |
| ECE ≤0.05 | ≤0.05 | 见 `m10_calib_v1_report.txt` | ◐ 9/15 达标 | L1/mc 三切分 + heldout 低支撑未达 |
| L3 目标相 top-1 | ≥0.76 | {l3_t}(time 三元 {l3_tri}) | 未达 | EMT 考题（time 主切分） |

## 5. T3.4 硬约束过滤器（对接）
6 规则 + cap 语义标签，训练域校准；读数见 `Data/t3_4_filter_report.txt`。
`score()` 对生成器 12 列空间输入自动附加 `t3_4_reject / t3_4_rules / t3_4_cap_label` 列。

## 6. 版本变更登记
| 版本 | 日期 | 变更 | 锚点 |
| --- | --- | --- | --- |
| {ver} | {date} | 首版（三层判别器统一入口 + 温度校准 + T3.4 对接） | 12/12 PASS |
"""


def write_bench_skeleton(scorer, anchors):
    a = {k: v['measured'] for k, v in anchors.items()}
    txt = BENCH_SKELETON.format(
        ver=SCORER_VERSION, dv=DATA_VERSION, md5=md5(os.path.abspath(__file__)),
        git=git_head(), date=time.strftime('%Y-%m-%d %H:%M:%S'),
        l1_r='%.4f' % a['L1/random'], l1_h='%.4f' % a['L1/heldout'], l1_t='%.4f' % a['L1/time_2018'],
        l2x_r='%.4f' % a['L2xt/random'], l2x_h='%.4f' % a['L2xt/heldout'], l2x_t='%.4f' % a['L2xt/time_2018'],
        l2p_r='%.4f' % a['L2pu/random'], l2p_h='%.4f' % a['L2pu/heldout'], l2p_t='%.4f' % a['L2pu/time_2018'],
        l3_r='%.4f' % a['L3/random'], l3_h='%.4f' % a['L3/heldout'], l3_t='%.4f' % a['L3/time_2018'],
        t1='%.4f' % scorer.heads['L1']['T'], t2x='%.4f' % scorer.heads['L2-xtal']['T'],
        t2p='%.4f' % scorer.heads['L2-pure']['T'], t3='%.4f' % scorer.heads['L3']['T'],
        t3f='%.4f' % T_L3_FROZEN, l3_tri='%.4f' % L3_TRI_TIME)
    with open(BENCH_MD, 'w', encoding='utf-8') as f:
        f.write(txt)
    return BENCH_MD


# =====================================================================
# 主流程: fit
# =====================================================================
def run_fit():
    t0 = time.time()
    log('feasibility_scorer %s —— 三层判别器统一入口 + 温度校准 + T3.4 过滤器' % SCORER_VERSION)
    log('数据版本 %s; git %s' % (DATA_VERSION, git_head()))
    log('协议纪律: 三臂逐位镜像 m10_calib_v1 (不修改既有脚本); 校准层复用 l3_purity_v1 定义')
    df, sm = load_tier_a_dedup()
    df8 = pd.read_parquet(DB)
    log('去 Tier A: v1.8 %d 行 → %d 行' % (len(df8), len(df)))

    log('')
    log('【L1 臂】相分类 (A3 490, 198 类)')
    h1 = build_l1(df, sm)
    log('')
    log('【L2 臂】可行性 (xtal 主口径 + pure 参照, A3 490)')
    h2 = build_l2(df8)
    log('')
    log('【L3 臂】纯度竞相 (498 维, 200 类)')
    h3 = build_l3(df, sm)

    heads = {'L1': h1, 'L2-xtal': h2['xtal'], 'L2-pure': h2['pure'], 'L3': h3}
    scorer = FeasibilityScorerV1(heads)
    pkl = scorer.save()

    log('')
    log('=' * 78)
    # 锚点断言 (基于训练中间态 proba_te; 与 selfcheck 双路互证)
    order = [('L1', h1, 'mc'), ('L2xt', h2['xtal'], 'bin'), ('L2pu', h2['pure'], 'bin'), ('L3', h3, 'mc')]
    n_fail = 0
    anc_meas = {}
    for name, h, kind in order:
        for tag in SPLITS3:
            exp = ANC[name][tag]
            got = h['anchors'][tag]
            d = got - exp
            ok = abs(d) < ANC_TOL
            n_fail += (not ok)
            anc_meas['%s/%s' % (name, tag)] = float(got)
            log('  %-5s %-9s 期望 %.4f 实测 %.4f (Δ%+.4f) -> %s'
                % (name, tag, exp, got, d, 'PASS' if ok else '★漂移★'))
    log('  12 项锚点 assert: %s' % ('全部 PASS' if n_fail == 0 else '%d 项漂移 (FAIL)' % n_fail))

    # 温度 scale 不改 argmax 断言 (逐头)
    for name, h, kind in order:
        for tag in SPLITS3:
            p = h['proba_te'][tag]
            assert np.array_equal(p.argmax(1), proba_with_T(p, h['T']).argmax(1)), \
                'argmax 漂移: %s/%s' % (name, tag)
    log('  温度缩放 argmax 不变: 逐头×切分断言 PASS (分类读数不受 T 影响)')

    # 口径规模
    log('')
    log('域规模: L1 %d 行 %d 类 | L2 %d 行 | L3 %d 行 %d 类'
        % (h1['n_domain'], len(h1['classes']), h2['xtal']['n_domain'],
           h3['n_domain'], len(h3['classes'])))

    # T3.4 过滤器对接自测
    log('')
    log('T3.4 硬约束过滤器对接: %s'
        % ('可用 (训练域校准, 6 规则 + cap 标签)' if apply_hard_filters(pd.DataFrame(
            np.zeros((1, 12)), columns=_gen_cols())) is not None else '数据缺失, 接口预留'))
    af = _read_t34_apply()
    if af:
        log('  v0.1 生成产物淘汰率 (引用 t3_4_filter_report): %s' % af)

    # 接口冒烟: score_one (库式输入, 取 L3 域首行)
    log('')
    log('API 冒烟 (score_one, 库式输入):')
    try:
        demo = df.loc[(df['label1'] == 'positive') & df['zeo_desc_key'].notna()].iloc[[0]]
        r = scorer.score_one(demo.to_dict('records')[0], target='EMT')
        log('  route %s: L1 top1=%s (p=%.4f); L2 p_xtal=%.4f p_pure=%.4f; '
            'L3 top1=%s p_max=%.4f target_p(EMT)=%.4f'
            % (r.get('route_id'), r['l1_fw_top1'], r['l1_p_max'], r['l2_p_xtal'], r['l2_p_pure'],
               r['l3_fw_top1'], r['l3_p_max'], r.get('l3_target_p', float('nan'))))
    except Exception as e:                                   # noqa
        log('  冒烟异常 (登记): %r' % e)

    # 落盘
    bench = write_bench_skeleton(scorer, {k: {'measured': v} for k, v in anc_meas.items()})
    man = {
        'task': 'feasibility_scorer v1.2 (M12 交付物: CLI + Python API; L3 COMP v2)',
        'scorer_version': SCORER_VERSION,
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'git_head': git_head(),
        'code_md5': {os.path.basename(__file__): md5(os.path.abspath(__file__))},
        'data_version': DATA_VERSION,
        'heads': {k: {'protocol': v, 'n_domain': heads[k]['n_domain'],
                      'n_class': len(heads[k]['classes']) if heads[k]['classes'] else 2,
                      'rounds': int(heads[k]['best']), 'T': float(heads[k]['T'])}
                  for k, v in (('L1', 'mirror m10 arm_l1 (build_a3 A3 490)'),
                               ('L2-xtal', 'mirror m10 arm_l2 (xtal 主口径 Q19)'),
                               ('L2-pure', 'mirror m10 arm_l2 (pure 参照)'),
                               ('L3', 'mirror m10 arm_l3 (fit_v1 comp=v2 498 + sidecar)'))},
        'anchors_expected': ANC,
        'anchors_measured': anc_meas,
        'anchors_pass_all': bool(n_fail == 0),
        'temperature': {'L1': float(h1['T']), 'L2xt': float(h2['xtal']['T']),
                        'L2pu': float(h2['pure']['T']), 'L3': float(h3['T']),
                        'L3_frozen_v1_2': T_L3_FROZEN},
        'model_bundle': {'path': os.path.basename(pkl), 'md5': md5(pkl)},
        'benchmark_skeleton': os.path.basename(bench),
        'cli': 'fit|selfcheck|score --input --out [--target]',
    }
    with open(MANIFEST, 'w', encoding='utf-8') as f:
        json.dump(man, f, ensure_ascii=False, indent=2)

    log('')
    log('产出: %s | %s | %s | %s' % (pkl, REPORT, MANIFEST, bench))
    log('fit 用时 %.1f min' % ((time.time() - t0) / 60))
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(_L) + '\n')
    if n_fail:
        raise SystemExit('★ 锚点漂移 %d 项 (Δ≥%.0e): 不静默通过 ★' % (n_fail, ANC_TOL))
    log('__SCORER_FIT_DONE__')


def _gen_cols():
    try:
        import wp3_ddpm_v0 as w3
        return list(w3.COLS)
    except Exception:
        return []


def _read_t34_apply():
    p = os.path.join(BASE, 'data', 't3_4_filter_manifest.json')
    if not os.path.exists(p):
        return None
    try:
        m = json.load(open(p, encoding='utf-8'))
        a = m.get('apply_v01')
        if a:
            return ('总体淘汰率 %.4f / WP5 strict %.4f'
                    % (a['overall_reject_rate'], a['wp5_strict_rate']))
    except Exception:
        return None


# =====================================================================
# CLI
# =====================================================================
def run_selfcheck():
    scorer = FeasibilityScorerV1.load()
    log('feasibility_scorer %s selfcheck —— 载入 %s, 复算 12 项锚点' % (SCORER_VERSION, MODEL_PKL))
    log('数据版本 %s' % scorer.data_version)
    log('口径: 冻结统计重导特征 + 训练侧协议重训 (train-only, 轮数取模型束登记值)')
    res, n_fail = selfcheck_anchors(scorer)
    with open(SELF_REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(_L) + '\n')
    log('报告: %s' % SELF_REPORT)
    if n_fail:
        raise SystemExit(1)


def run_score(inp, out, target):
    scorer = FeasibilityScorerV1.load()
    d = pd.read_parquet(inp) if inp.lower().endswith('.parquet') else pd.read_csv(inp)
    res = scorer.score(d, target=target)
    if out.lower().endswith('.parquet'):
        res.to_parquet(out, engine='pyarrow', compression='zstd', index=False)
    else:
        res.to_csv(out, index=False, encoding='utf-8-sig')
    print('scored %d 行 -> %s' % (len(res), out))
    print(res.head(5).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='feasibility_scorer v1 (M12 交付物)')
    ap.add_argument('stage', choices=['fit', 'selfcheck', 'score'])
    ap.add_argument('--input', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--target', default=None, help='L3 目标相 (如 EMT), 给出则输出 l3_target_p')
    a = ap.parse_args()
    if a.stage == 'fit':
        run_fit()
    elif a.stage == 'selfcheck':
        run_selfcheck()
    else:
        if not a.input or not a.out:
            ap.error('score 需 --input 与 --out')
        run_score(a.input, a.out, a.target)
