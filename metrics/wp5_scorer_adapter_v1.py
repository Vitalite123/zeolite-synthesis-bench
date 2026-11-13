# -*- coding: utf-8 -*-
"""
WP5 scorer 适配器 v1 — 条件一致率口径三角对照 (G3 报告 §6.4 候选执行)
================================================================================
目的: 分解一致率 gap (生成 frac@0.5 ~0.17 vs 真实参照 0.873) 中「198 类 softmax 先验挤压」
      「Dense/Amorphous 负类支撑」「生成质量」三者贡献 — M12 终线一致率可达性判定前置。
设计修正 (对 G3 报告原候选表述): 「接 L1/L2 概率头」的 A3 682 维空间路线不可行 — 生成行仅
      12 列, 缺 n_osda2/n_seed/solvent/stirring/ratio_charge_bal 等生成器未建模维度, 无法
      构造 A3; 适配改为同 12 列特征空间三口径对照 (v0.5 主口径协议冻结不动, 本脚本并行读数):
  口径① v0.5 主口径: SCORER_PKL 复用 (12 列→198 类全人口, Dense/Amorphous 负类支撑)
  口径② nneg 非负域头: 12 列→zeo_key 真码类 (去 Dense/Amorphous 支撑; 训练子集 =
        payload zeo_key != 'Dense/Amorphous', 即 absolute_neg 聚合负类全剔除), 同修正配置
        (300 轮/lr0.05/min_leaf200, seed 0) — 单变量 = 训练域
  口径③ calib-K 挤压校准 (零训练): v0.5 proba top-K 局部归一 p'=p_t/ΣtopK, K∈{5,10,20}
        — 直接量化「概率质量被 198 类全局分走」的先验挤压贡献
读数双侧: 真实行参照 (合格体系 test 行, 与 read_stage 同源) + 六模型生成行
        (Data/wp5_g3_formal_{tag}_syn.parquet, 协议产物直接复用, 不重采样)。
判定逻辑: 三口径保持率 (生成/真实) 全部低位 → 生成质量主导实锤; calib 口径大幅回升 →
        先验挤压主因 (M12 终线口径建议同步修订)。
用法: python wp5_scorer_adapter_v1.py
产出: Data/wp5_scorer_adapter_v1_report.txt / wp5_scorer_adapter_v1_manifest.json (gitignored)
"""
import os, sys, json, time, pickle, hashlib, subprocess, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZG = os.path.join(BASE, 'vendor', 'zeosyn_gen')
os.chdir(ZG); sys.path.insert(0, ZG); sys.path.insert(0, BASE)

import torch
import wp3_ddpm_v0 as w3
import wp3_ddpm_v0_1 as v01

SCORER_PKL = os.path.join(BASE, 'data', 'wp5_g3_scorer.pkl')
REPORT     = os.path.join(BASE, 'data', 'wp5_scorer_adapter_v1_report.txt')
MANIFEST   = os.path.join(BASE, 'data', 'wp5_scorer_adapter_v1_manifest.json')
CALIB_KS   = [5, 10, 20]

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

def pred_parq(tag):
    return os.path.join(BASE, 'data', 'wp5_g3_formal_%s_syn.parquet' % tag.replace('.', ''))

# ---------------- 打分核 (全量 proba 一次算, 逐体系 numpy 索引) ----------------
def proba_target(P, cls_idx, tgt):
    """P (n,C) proba, tgt (n,) 目标类名 → (p_target, argmax_hit)"""
    col = np.array([cls_idx[t] for t in tgt])
    return P[np.arange(len(tgt)), col], P.argmax(axis=1) == col

def topk_norm(P, K):
    """top-K 局部归一: p' = p / ΣtopK (逐行)"""
    part = np.partition(P, -K, axis=1)[:, -K:].sum(axis=1)
    return P / np.clip(part, 1e-12, None)[:, None]

def agg(p_t, hit):
    return dict(mean_p=float(p_t.mean()),
                frac05=float((p_t >= 0.5).mean()), frac02=float((p_t >= 0.2).mean()),
                top1=float(hit.mean()), n=int(len(p_t)))

def main():
    t0 = time.time()
    log('==================== scorer 适配器 v1 — 一致率口径三角对照 ====================')
    with open(SCORER_PKL, 'rb') as f:
        sc = pickle.load(f)
    clf = sc['model']
    cls_idx = {c: i for i, c in enumerate(clf.classes_)}
    cols12 = list(sc['cols'])

    # ---- 非负域头训练 (同修正配置, 单变量 = 训练域) ----
    import lightgbm as lgb
    payload = torch.load(v01.DS_PT_V01, weights_only=False)
    X_all = np.asarray(payload['syn_raw']); zk_all = np.asarray(payload['zeo_key'])
    nneg = zk_all != 'Dense/Amorphous'
    log('非负域头训练: %d 行 (全人口 %d 剔除 Dense/Amorphous %d) → %d 真码类 (300轮/lr0.05/mdl200, seed 0)'
        % (int(nneg.sum()), len(X_all), int((~nneg).sum()), len(pd.unique(zk_all[nneg]))))
    ts = time.time()
    clf_n = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                               min_data_in_leaf=200, random_state=0, n_jobs=-1, verbose=-1)
    clf_n.fit(X_all[nneg], zk_all[nneg])
    cls_idx_n = {c: i for i, c in enumerate(clf_n.classes_)}
    log('非负域头训练完成 (%.1f min)' % ((time.time() - ts) / 60))

    # ---- 真实行参照 (合格体系 test 行, 复刻 read_stage eligible 逻辑, v0 payload) ----
    pv0 = torch.load(w3.DS_PT, weights_only=False)
    zk0 = np.asarray(pv0['zeo_key']); ok0 = np.asarray(pv0['osda_key'])
    te0 = pv0['test_idx'].numpy()
    key_te = pd.Series(list(zip(zk0[te0], ok0[te0])))
    counts = key_te.value_counts()
    eligible = [k for k in counts.index if counts[k] >= 5 and k[0] != 'Dense/Amorphous']
    te_rows = te0[key_te.isin(eligible).to_numpy()]
    tgt_true = zk0[te_rows]
    X_true = X_all[te_rows]                       # payload 行序 v0/v01 逐位一致 (split 两版本同源)
    log('真实行参照: 合格体系 %d 个, test 行 %d' % (len(eligible), len(te_rows)))

    # 真实行三口径 proba
    Pt5 = clf.predict_proba(X_true)
    Ptn = clf_n.predict_proba(X_true)
    p5, h5 = proba_target(Pt5, cls_idx, tgt_true)
    pn, hn = proba_target(Ptn, cls_idx_n, tgt_true)
    ref = {'v0.5': agg(p5, h5), 'nneg': agg(pn, hn)}
    for K in CALIB_KS:
        pk, hk = proba_target(topk_norm(Pt5, K), cls_idx, tgt_true)
        ref['calib%d' % K] = agg(pk, hk)
    log('真实行参照 (%d 行):' % len(te_rows))
    for k in ['v0.5', 'nneg'] + ['calib%d' % K for K in CALIB_KS]:
        r = ref[k]
        log('  %-8s mean P=%.4f frac@0.5=%.4f frac@0.2=%.4f top1=%.4f'
            % (k, r['mean_p'], r['frac05'], r['frac02'], r['top1']))

    # ---- 六模型生成行三口径读数 ----
    tags = ['v0', 'v0.1', 'v0.2', 'v0.3', 'v0.4', 'v0.4.1']
    rows = []
    for tag in tags:
        fp = pred_parq(tag)
        if not os.path.exists(fp):
            log('跳过 %s (无产物)' % tag); continue
        tm = time.time()
        df = pd.read_parquet(fp)
        Xg = df[cols12].to_numpy(); tgt_g = df['zeo'].to_numpy()
        Pg5 = clf.predict_proba(Xg)
        Pgn = clf_n.predict_proba(Xg)
        g5, gh5 = proba_target(Pg5, cls_idx, tgt_g)
        gn, ghn = proba_target(Pgn, cls_idx_n, tgt_g)
        gr = {'v0.5': agg(g5, gh5), 'nneg': agg(gn, ghn)}
        for K in CALIB_KS:
            gk, ghk = proba_target(topk_norm(Pg5, K), cls_idx, tgt_g)
            gr['calib%d' % K] = agg(gk, ghk)
        rows.append(dict(tag=tag, **gr))
        log('%s 生成 (%d 行, %.1f min): frac@0.5 v0.5=%.4f nneg=%.4f calib10=%.4f'
            % (tag, len(df), (time.time() - tm) / 60,
               gr['v0.5']['frac05'], gr['nneg']['frac05'], gr['calib10']['frac05']))

    # ---- 报告 ----
    keys = ['v0.5', 'nneg'] + ['calib%d' % K for K in CALIB_KS]
    R = ['scorer 适配器 v1 — 一致率口径三角对照 — %s' % time.strftime('%Y-%m-%d %H:%M'),
         '=' * 78,
         '目的: 分解一致率 gap = 先验挤压 / 负类支撑域 / 生成质量 三因素贡献;',
         '      全口径同 12 列特征空间 (A3 空间路线不可行: 生成行缺 n_osda2/n_seed/solvent 等',
         '      生成器未建模维度); v0.5 主口径协议冻结, 本读数为并行诊断不替代主口径。',
         '口径① v0.5 = 12列→198 类全人口 (SCORER_PKL 复用); 口径② nneg = 非负域头',
         '      (剔 Dense/Amorphous %d 行, %d 真码类, 同修正配置); 口径③ calib-K = v0.5 proba'
         % (int((~nneg).sum()), len(clf_n.classes_)),
         '      top-K 局部归一 (K=5/10/20, 零训练). 数据版本 zeobench_db_v1_8', '']
    R.append('真实行参照 (合格体系 test %d 行):' % len(te_rows))
    R.append('  %-8s %8s %9s %9s %7s' % ('口径', 'mean P', 'frac@0.5', 'frac@0.2', 'top-1'))
    for k in keys:
        r = ref[k]
        R.append('  %-8s %8.4f %9.4f %9.4f %7.4f' % (k, r['mean_p'], r['frac05'], r['frac02'], r['top1']))
    R.append('')
    R.append('六模型生成行 (24600 行/模型):')
    R.append('  %-8s' % '模型' + ''.join(' %9s' % k for k in keys) + '   保持率% (v0.5/nneg/calib10)')
    for row in rows:
        keep = ['%.1f' % (100 * row[k]['frac05'] / max(ref[k]['frac05'], 1e-9)) for k in ['v0.5', 'nneg', 'calib10']]
        R.append('  %-8s' % row['tag'] + ''.join(' %9.4f' % row[k]['frac05'] for k in keys)
                 + '   %s' % '/'.join(keep))
    R.append('')
    R.append('其他档位 (生成行, 最新模型 %s):' % rows[-1]['tag'])
    for k in keys:
        r = rows[-1][k]
        R.append('  %-8s mean P=%.4f frac@0.2=%.4f top1=%.4f' % (k, r['mean_p'], r['frac02'], r['top1']))
    R.append('')
    # ---- 判定 ----
    last = rows[-1]
    lift_c = last['calib10']['frac05'] - last['v0.5']['frac05']
    lift_n = last['nneg']['frac05'] - last['v0.5']['frac05']
    keep_c = last['calib10']['frac05'] / max(ref['calib10']['frac05'], 1e-9)
    R.append('判定 (最新模型 %s):' % last['tag'])
    R.append('  先验挤压贡献: calib10 生成 frac@0.5 %+.4f (v0.5 %.4f → %.4f); 真实行参照同向 %.4f → %.4f'
             % (lift_c, last['v0.5']['frac05'], last['calib10']['frac05'],
                ref['v0.5']['frac05'], ref['calib10']['frac05']))
    R.append('  负类支撑域贡献: nneg 生成 %+.4f (v0.5 %.4f → %.4f); 真实行参照 %.4f → %.4f'
             % (lift_n, last['v0.5']['frac05'], last['nneg']['frac05'],
                ref['v0.5']['frac05'], ref['nneg']['frac05']))
    R.append('  校准口径保持率: %.1f%% (真实参照 %.4f, 生成 %.4f)' % (100 * keep_c, ref['calib10']['frac05'], last['calib10']['frac05']))
    if keep_c < 0.40:
        R.append('  → 三口径保持率均低位: 生成质量主导实锤 (先验挤压/域非主因), G3 报告归因维持')
    else:
        R.append('  → 校准口径保持率显著回升: 先验挤压主因, M12 终线一致率口径建议修订')
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n\n' + '\n'.join(R) + '\n')
    log('报告: %s' % REPORT)

    # ---- manifest ----
    man = {'task': 'WP5 scorer 适配器 v1 — 一致率口径三角对照 (G3 报告 §6.4 候选执行)',
           'date': time.strftime('%Y-%m-%d %H:%M:%S'), 'data_version': 'zeobench_db_v1_8',
           'git_commit': git_head(),
           'code_md5': {'wp5_scorer_adapter_v1.py': md5(os.path.abspath(__file__))},
           'config': {'nneg_head': dict(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                        min_data_in_leaf=200, random_state=0,
                                        n_train=int(nneg.sum()), n_classes=int(len(clf_n.classes_))),
                      'calib_K': CALIB_KS,
                      'scorer_v05': 'SCORER_PKL 复用 (协议冻结)'},
           'ref_true': ref,
           'models': {r['tag']: {k: r[k] for k in keys} for r in rows},
           'duration_min': round((time.time() - t0) / 60, 1)}
    with open(MANIFEST, 'w', encoding='utf-8') as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    log('完成: 用时 %.1f min' % ((time.time() - t0) / 60))

if __name__ == '__main__':
    main()
