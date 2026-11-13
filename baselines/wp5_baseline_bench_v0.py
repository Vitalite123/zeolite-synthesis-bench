# -*- coding: utf-8 -*-
"""
WP5 指标 API 基线接入 v0 —— 四类官方生成基线过七类指标 + DiffSyn 官方对照表
================================================================================
任务定位 (执行计划 2026-10-30~11-05 周计划 P0-2 / §4 M12 检查表「指标 API」项):
  计划点名 KDE / cVAE / WGAN-GP / TabDDPM 四类基线接入 —— 其中 KDE、TabDDPM 在官方
  zeosyn_gen 仓库无现成实现; 负责人裁定 (2026-10-20)「跳过计划点名的 KDE 与 TabDDPM,
  其他按原计划执行」→ 本脚本接入官方既有四类生成基线 + 官方 DiffSyn 参考:
    cVAE (cvae/system/v10) · GAN=WGAN-GP (gan/system/v3-3) · GMM (gmm/system/v0) · NF (nf/system/v0)
    另附平凡下界 NN (nn/system/v0) / random (random/system/v0) 与 DiffSyn 官方 run1 参考行。
  ⚠ 该裁定已于 2026-10-23 解除：KDE / TabDDPM 改为自实现并纳入 ZeoliteSynthesisBench v1
    Track A（见 wp5_bench_v1_selfbaselines.py 与规格 v1.1）；本脚本维持 50×n_true 官方口径存档。

纪律 (交接清单 §2.2 / §2.3):
  指标计算全部复用 wp5_metrics_api_v0.ZeoSynMetrics (官方 WSD/MMD/COV wrap 不重写);
  官方预测产物直接复用 (runs/*/*/syn_pred_agg.csv, 50×n_true/体系 —— 满足 COV pred 规模纪律,
  无需重载模型推理); 归一空间 = 官方 utils.scale_x_syn_ratio (逐列全人口 max, 与 API scale 同法);
  DiffSyn 官方对照表 = 各 run 自带 wsd_zeo_osda.json / mmd_zeo_osda.json / prec·rec_zeo_osda_df.csv。

七类指标覆盖 (②⑤⑦ 口径登记见报告):
  ① W2 联合 + 逐维 W1  ② MMD  ③ COV-P / COV-R  ④ 物理有效率 (strict)
  ⑤ 条件一致率 (scorer 注入 = wp5_g3_scorer.pkl 口径① v0.5 主口径, 12 列→zeo_key 全人口头)
  ⑥ 新颖性 (全局 NN / 骨架级 NN / 骨架级凸包外 / 全局盒外)
  ⑦ 外推指数 —— 官方基线仅 system 切分, 无 time 切分产物 → N/A (登记, M10 时间外推线另跑)

口径差异登记 (advisor 提示): 官方 eval.py 的 MMD/WSD 在 min-max 归一空间计算、coverage 在原始
  量纲计算; 本 API W2 联合用人口 max 归一、W1 用原始量纲、COV 原始量纲 —— 对照表显式标注,
  不隐式混用。

用法: python wp5_baseline_bench_v0.py
产出 (幂等覆盖写):
  Data/wp5_baseline_bench_v0_report.txt       全量日志报告 (gitignored)
  Data/wp5_baseline_bench_v0_table.csv        对照表 (本 API 读数 + 官方读数, gitignored)
  Data/wp5_baseline_bench_v0_manifest.json    manifest (数据版本 + 代码 md5 + 注册表)
  Documentation/wp5_baseline_bench_v0_对照表.md  版本化对照表 (入 git)
"""
import os, sys, json, time, pickle, hashlib, subprocess, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZG = os.path.join(BASE, 'vendor', 'zeosyn_gen')
sys.path.insert(0, BASE); sys.path.insert(0, ZG)

import torch
import wp5_metrics_api_v0 as api                 # 会 os.chdir(ZG) + 注入 utils (官方 pkl 反序列化)
M = api.ZeoSynMetrics()
_du = sys.modules['utils']

# ---------------- 路径与常量 ----------------
PKL        = os.path.join(ZG, 'data', 'ZeoSynGen_dataset.pkl')
SCORER_PKL = os.path.join(BASE, 'data', 'wp5_g3_scorer.pkl')
REPORT     = os.path.join(BASE, 'data', 'wp5_baseline_bench_v0_report.txt')
TABLE_CSV  = os.path.join(BASE, 'data', 'wp5_baseline_bench_v0_table.csv')
MANIFEST   = os.path.join(BASE, 'data', 'wp5_baseline_bench_v0_manifest.json')
DOC_MD     = os.path.join(BASE, 'Documentation', 'wp5_baseline_bench_v0_对照表.md')

DATA_VERSION = 'zeobench_db_v1_8 (官方 ZeoSynGen_dataset.pkl, system split)'
N_HULL = 3000          # 凸包外 LP 判定子采样上限 (LP 逐行成本约束; 分层子采样 seed 0)
HULL_SEED = 0
COND_SCALE_DIFF = 0.75 # DiffSyn 官方 run1 预测产物 cond_scale

# 官方基线注册表: (tag, 显示名, model_type, split, fname, pred_filename)
RUNS = [
    ('cvae',   'cVAE',      'cvae',   'system', 'v10',  'syn_pred_agg.csv'),
    ('gan',    'WGAN-GP',   'gan',    'system', 'v3-3', 'syn_pred_agg.csv'),
    ('gmm',    'GMM',       'gmm',    'system', 'v0',   'syn_pred_agg.csv'),
    ('nf',     'NF',        'nf',     'system', 'v0',   'syn_pred_agg.csv'),
    ('nn',     'NN(下界)',  'nn',     'system', 'v0',   'syn_pred_agg.csv'),
    ('random', 'random(下界)', 'random', 'system', 'v0', 'syn_pred_agg.csv'),
    ('diff',   'DiffSyn(参考)', 'diff', 'system', 'run1',
     'syn_pred_agg-cond_scale_%s-test.csv' % COND_SCALE_DIFF),
]
CORE4 = ['cvae', 'gan', 'gmm', 'nf']             # 计划四类 (KDE/TabDDPM 已裁跳) 的落地映射

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
# 真值构造 (逐位镜像 eval.py / calibration① 协议)
# =====================================================================
def build_true_df(ds):
    """官方 _dataset → 原始量纲 DataFrame: [1]=syn ratios (归一空间), [3]=zeo code, [13]=osda smiles"""
    X = np.array(ds[1], dtype=float)
    for i, r in enumerate(COLS):
        qt = dataset.qts[r]
        X[:, i] = torch.tensor(qt.inverse_transform(X[:, i].reshape(-1, 1))).reshape(-1)
    df = pd.DataFrame(X, columns=COLS)
    df['zeo'], df['osda'] = ds[3], ds[13]
    return df

# =====================================================================
# 单基线全指标
# =====================================================================
def bench_one(tag, name, model_type, split, fname, pred_name):
    path = os.path.join(ZG, 'runs', model_type, split, fname, pred_name)
    if not os.path.exists(path):
        log('  [跳过] %s: 无预测产物 %s' % (name, path)); return None, None
    pred = pd.read_csv(path)
    assert list(pred.columns[:len(COLS)]) == COLS, '%s 列序与官方 12 列不一致' % tag
    pred_scaled = _du.scale_x_syn_ratio(pred, dataset)

    # ---- ①②③④ 逐体系 (官方 eval_zeolite_osda 同款遍历 + 同款阈值/归一) ----
    rows = []
    for zeo, osda in syn_true[['zeo', 'osda']].value_counts().index:
        if zeo == 'Dense/Amorphous':
            continue
        m = (pred['zeo'] == zeo) & (pred['osda'] == osda)
        mt = (syn_true['zeo'] == zeo) & (syn_true['osda'] == osda)
        if m.sum() == 0:
            continue
        p, t = pred.loc[m, COLS], syn_true.loc[mt, COLS]
        ps, ts = pred_scaled.loc[m, COLS], syn_true_scaled.loc[mt, COLS]
        cov = M.cov_pr(p, t)
        rows.append(dict(zeo=zeo, osda=osda, n_true=int(mt.sum()), n_pred=int(m.sum()),
                         W2=M.w2_joint(ps, ts), MMD=M.mmd(ps, ts),
                         W1=float(np.mean(list(M.w1_per_dim(p, t).values()))),
                         COVP=float(np.mean(list(cov['precision'].values()))),
                         COVR=float(np.mean(list(cov['recall'].values()))),
                         PV=M.physical_validity_rate(p, 'strict')['rate']))
    sys_df = pd.DataFrame(rows)
    agg = {k: float(sys_df[k].mean()) for k in ['W2', 'MMD', 'W1', 'COVP', 'COVR', 'PV']}
    agg['n_systems'] = int(len(sys_df)); agg['n_pred_rows'] = int(len(pred))

    # ---- ④ 物理有效率 (全局, 便于与逐体系均值区分) ----
    agg['PV_global'] = M.physical_validity_rate(pred[COLS], 'strict')['rate']

    # ---- ⑥ 新颖性 (全局 NN + 骨架级 NN / 凸包外; 凸包子采样 N_HULL) ----
    tr = train_df[COLS]
    nn_g = M.nn_distance(pred[COLS], tr, ref=tr)
    nn_fw = M.nn_distance(pred[COLS], tr, ref=tr,
                          gen_sys=pred['zeo'].to_numpy(), train_sys=train_df['zeo'].to_numpy())
    rng = np.random.RandomState(HULL_SEED)
    sel = rng.choice(len(pred), min(N_HULL, len(pred)), replace=False)
    sub = pred.iloc[sel].reset_index(drop=True)
    hull = M.outside_hull_fraction(sub[COLS], tr, sub['zeo'].to_numpy(),
                                   train_df['zeo'].to_numpy(), ref=tr)
    box = M.outside_box_fraction(pred[COLS], tr)
    agg.update({'NN_mean': nn_g['mean'], 'NN_median': nn_g['median'],
                'NNfw_mean': nn_fw['mean'], 'HULL_out': hull['fraction_outside'],
                'HULL_n': hull['n'], 'BOX_out': box['fraction_outside']})

    # ---- ⑤ 条件一致率 (口径① v0.5 主口径: 12 列→zeo_key 全人口头) ----
    cc = cond_consistency(pred, pred['zeo'].to_numpy()) if scorer is not None else None
    if cc is not None:
        agg.update({'CC_meanP': cc['mean_p'], 'CC_frac05': cc['frac_ge_tau'], 'CC_n': cc['n']})

    # ---- DiffSyn 官方读数 (该 run 自带 json / csv) ----
    off = official_readings(model_type, split, fname)
    return agg, (sys_df, off)

def cond_consistency(df, target):
    """⑤ 条件一致率: 用 wp5_g3_scorer.pkl (口径①) 打分 → ZeoSynMetrics.conditional_consistency"""
    def _score(d, tgt):
        P = clf.predict_proba(d[cols12].to_numpy())
        col = np.array([cls_idx[x] for x in tgt])
        return P[np.arange(len(tgt)), col]
    tgt = [x for x in target]
    bad = [x for x in set(tgt) if x not in cls_idx]
    if bad:
        log('       [⑤] 目标码不在 scorer 类集, 剔除 %d 行 (示例 %s)' % (
            int(sum(1 for x in tgt if x not in cls_idx)), bad[:3]))
    keep = np.array([x in cls_idx for x in tgt])
    if keep.sum() == 0:
        return None
    return M.conditional_consistency(df[COLS].iloc[keep].reset_index(drop=True),
                                     _score, target=np.array(tgt)[keep], tau=0.5)

def official_readings(model_type, split, fname):
    d = os.path.join(ZG, 'runs', model_type, split, fname)
    out = {}
    for key, fn, field in [('off_WSD', 'wsd_zeo_osda.json', 'WSD mean'),
                           ('off_MMD', 'mmd_zeo_osda.json', 'MMD mean')]:
        fp = os.path.join(d, fn)
        if os.path.exists(fp):
            with open(fp) as f:
                out[key] = json.load(f)[field]
    for key, fn, sfx in [('off_COVP', 'prec_zeo_osda_df.csv', '_prec'),
                         ('off_COVR', 'rec_zeo_osda_df.csv', '_rec')]:
        fp = os.path.join(d, fn)
        if os.path.exists(fp):
            dfc = pd.read_csv(fp)
            cols = [c for c in dfc.columns if c.endswith(sfx)]
            out[key] = float(dfc[cols].mean().mean())
    return out

# =====================================================================
def main():
    t0 = time.time()
    global dataset, COLS, syn_true, syn_true_scaled, train_df, scorer, clf, cls_idx, cols12
    log('WP5 指标 API 基线接入 v0 —— 四类官方生成基线过七类指标 + DiffSyn 官方对照表')
    log('=' * 78)
    log('生成时间: %s | git %s | 模块 md5 %s' % (
        time.strftime('%Y-%m-%d %H:%M:%S'), git_head(), md5(os.path.abspath(__file__))))
    log('裁定: 计划点名 KDE / TabDDPM 无官方实现 → 跳过 (2026-10-20 负责人裁定); '
        '四类 = cVAE / WGAN-GP / GMM / NF (官方既有)')
    log('数据版本: %s' % DATA_VERSION)

    with open(PKL, 'rb') as f:
        dataset = pickle.load(f)
    COLS = dataset.ratio_names + dataset.cond_names
    assert COLS == M.cols, '官方列序与 API 默认列不一致'

    _, _, _test = dataset.train_val_test_split(mode='system', both_graph_feat_present=True, random_state=0)
    _train, _, _ = dataset.train_val_test_split(mode='system', both_graph_feat_present=True, random_state=0)
    syn_true = build_true_df(_test); syn_true_scaled = _du.scale_x_syn_ratio(syn_true, dataset)
    train_df = build_true_df(_train)
    log('官方资产: 数据集 %d 路线 | test 真值 %d 行 (%d 体系) | train 域 %d 行'
        % (len(dataset), len(syn_true), syn_true[['zeo', 'osda']].value_counts().shape[0], len(train_df)))

    # ⑤ scorer 载入 (口径① v0.5 主口径)
    scorer = clf = cls_idx = cols12 = None
    if os.path.exists(SCORER_PKL):
        with open(SCORER_PKL, 'rb') as f:
            sc = pickle.load(f)
        clf = sc['model']; cls_idx = {c: i for i, c in enumerate(clf.classes_)}; cols12 = list(sc['cols'])
        scorer = True
        log('⑤ scorer: %s (%d 类, 12 列头) — 口径① v0.5 主口径' % (os.path.basename(SCORER_PKL), len(clf.classes_)))
    else:
        log('⑤ scorer 缺失 (%s) → 条件一致率 N/A' % SCORER_PKL)

    # 真实行参照 (⑤一致性 / 说明口径)
    key_te = pd.Series(list(zip(syn_true['zeo'], syn_true['osda'])))
    counts = key_te.value_counts()
    eligible = [k for k in counts.index if counts[k] >= 5 and k[0] != 'Dense/Amorphous']
    ref_rows = syn_true[key_te.isin(eligible).to_numpy()].reset_index(drop=True)
    log('⑤ 真实行参照: 合格体系 %d 个, test 行 %d' % (len(eligible), len(ref_rows)))
    ref_cc = cond_consistency(ref_rows, ref_rows['zeo'].to_numpy()) if scorer is not None else None
    if ref_cc is not None:
        log('    真实行参照 ⑤: mean P=%.4f frac@0.5=%.4f' % (ref_cc['mean_p'], ref_cc['frac_ge_tau']))

    # ---- 逐基线 ----
    log('')
    log('逐基线读数 (①②③④ 逐体系均值 + ④全局 + ⑥新颖性 + ⑤一致率):')
    log('%-12s %5s %7s | %8s %8s %8s | %6s %6s | %6s | %8s %8s %7s | %7s %8s'
        % ('基线', '体系', 'n_pred', 'W2', 'MMD', 'W1均', 'COV-P', 'COV-R', '物理%',
           'NN均', 'NN骨架', '凸包外', '一致率', '官方WSD'))
    table, syscols, offcols = [], {}, {}
    for tag, name, mt, sp, fn, pf in RUNS:
        log('  --- %s (%s/%s/%s) ---' % (name, mt, sp, fn))
        agg, aux = bench_one(tag, name, mt, sp, fn, pf)
        if agg is None:
            continue
        sys_df, off = aux
        syscols[tag] = sys_df; offcols[tag] = off
        row = dict(tag=tag, name=name, run='%s/%s/%s' % (mt, sp, fn), core4=(tag in CORE4), **agg)
        table.append(row)
        log('%-12s %5d %7d | %8.4f %8.4f %8.4f | %6.3f %6.3f | %6.4f | %8.4f %8.4f %7.4f | %7s %8s'
            % (name, agg['n_systems'], agg['n_pred_rows'], agg['W2'], agg['MMD'], agg['W1'],
               agg['COVP'], agg['COVR'], agg['PV_global'], agg['NN_mean'], agg['NNfw_mean'],
               agg['HULL_out'], '%.4f' % agg.get('CC_frac05', float('nan')),
               '%.4f' % off.get('off_WSD', float('nan'))))

    tbl = pd.DataFrame(table)

    # ---- 报告 ----
    log('')
    log('=' * 78)
    log('对照表 (本 API 读数):')
    show = ['name', 'n_systems', 'W2', 'MMD', 'W1', 'COVP', 'COVR', 'PV_global',
            'NN_mean', 'HULL_out', 'BOX_out', 'CC_frac05']
    log(tbl[[c for c in show if c in tbl.columns]].to_string(index=False))
    log('')
    log('DiffSyn 官方对照表 (各 run 自带 wsd/mmd json + prec/rec csv, 与本 API 同口径并列):')
    log('%-12s | %8s %8s | %8s %8s | %8s %8s' % ('基线', '本W2', '官WSD', '本MMD', '官MMD', '本COV-P', '官COV-P'))
    for _, r in tbl.iterrows():
        off = offcols.get(r['tag'], {})
        log('%-12s | %8.4f %8s | %8.4f %8s | %8.4f %8s'
            % (r['name'], r['W2'], '%.4f' % off.get('off_WSD', float('nan')),
               r['MMD'], '%.4f' % off.get('off_MMD', float('nan')),
               r['COVP'], '%.4f' % off.get('off_COVP', float('nan'))))
    log('')
    log('口径登记: ②MMD/①W2 = 官方 utils.scale_x_syn_ratio 人口 max 归一空间 (与官方 eval.py 同);')
    log('          W1/COV/物理有效率 = 原始量纲; ⑤ 一致率口径① = 12 列→zeo_key 全人口 softmax 头;')
    log('          ⑥ 凸包外 = 骨架级分离超平面 LP 判定, 逐行 LP 成本约束 → 子采样上限 N_HULL=%d (seed %d);'
        % (N_HULL, HULL_SEED))
    log('          ⑦ 外推指数 N/A —— 官方基线仅 system 切分, 无 time 切分预测产物 (M10 时间外推线另跑)。')
    log('判定: PASS (四类基线 + 官方对照表全链路跑通; 读数登记) — 用时 %.1f min' % ((time.time() - t0) / 60))

    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    tbl.to_csv(TABLE_CSV, index=False)
    for tag, sdf in syscols.items():
        sdf.to_csv(os.path.join(BASE, 'data', 'wp5_baseline_bench_v0_sys_%s.csv' % tag), index=False)

    # ---- manifest ----
    man = {'task': 'WP5 指标 API 基线接入 v0 (执行计划 P0-2)',
           'data_version': DATA_VERSION, 'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
           'git': git_head(), 'code_md5': md5(os.path.abspath(__file__)),
           'asset_md5': {'ZeoSynGen_dataset.pkl': md5(PKL)},
           'ruling': 'KDE/TabDDPM 跳过 (官方无实现, 2026-10-20 裁定); 四类 = cVAE/WGAN-GP/GMM/NF',
           'registry': [{'tag': t, 'run': '%s/%s/%s' % (m_, s_, f_), 'pred': p_}
                        for t, _, m_, s_, f_, p_ in RUNS],
           'metrics': ['W2_joint', 'W1_per_dim', 'MMD', 'COV-P', 'COV-R', 'physical_validity',
                       'conditional_consistency', 'nn_distance', 'outside_hull', 'outside_box',
                       'extrapolation_index(NA)'],
           'scaling': 'W2/MMD 人口 max 归一 (官方 scale_x_syn_ratio); W1/COV 原始量纲',
           'cov_discipline': '50xn_true/体系 (官方 syn_pred_agg.csv 原生协议)'}
    with open(MANIFEST, 'w', encoding='utf-8') as f:
        json.dump(man, f, ensure_ascii=False, indent=2)

    # ---- 版本化对照表 (入 git) ----
    md = build_doc(tbl, offcols, ref_cc)
    with open(DOC_MD, 'w', encoding='utf-8') as f:
        f.write(md)
    log('产出: %s | %s | %s | %s' % (REPORT, TABLE_CSV, MANIFEST, DOC_MD))
    log('__WP5_BASELINE_BENCH_DONE__')

def build_doc(tbl, offcols, ref_cc, ):
    L2 = ['# WP5 指标 API 基线接入 v0 — 四类生成基线对照表', '',
          '> 执行计划 P0-2（M12 检查表「指标 API」项）。负责人裁定 2026-10-20：**跳过计划点名的 KDE / TabDDPM**'
          '（官方 zeosyn_gen 无现成实现）；接入官方既有四类基线 = **cVAE / WGAN-GP / GMM / NF**，'
          '另附平凡下界 NN / random 与 DiffSyn 官方 run1 参考。数据版本 `zeobench_db_v1_8`。', '',
          '## 表 A｜本 API 七类指标读数（逐体系均值，COV pred 规模 = 50×n_true/体系）', '',
          '| 基线 | run | 体系数 | W2 | MMD | W1均 | COV-P | COV-R | 物理有效率† | NN(全局) | 骨架凸包外 | 条件一致率‡ |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for _, r in tbl.iterrows():
        L2.append('| %s | `%s` | %d | %.4f | %.4f | %.4f | %.3f | %.3f | %.4f | %.4f | %.4f | %s |'
                  % (r['name'], r['run'], r['n_systems'], r['W2'], r['MMD'], r['W1'],
                     r['COVP'], r['COVR'], r['PV_global'], r['NN_mean'], r['HULL_out'],
                     ('%.4f' % r['CC_frac05']) if 'CC_frac05' in r and pd.notna(r.get('CC_frac05')) else 'n/a'))
    L2 += ['', '† 物理有效率 = 全局 strict [20,350]°C 水热窗口; 逐体系均值见 `Data/wp5_baseline_bench_v0_report.txt`。',
           '‡ 条件一致率 = frac@0.5 (scorer 口径①, 12 列→zeo_key 全人口头)%s。'
           % ('' if ref_cc is None else '；真实行参照 %.4f' % ref_cc['frac_ge_tau']),
           '', '## 表 B｜DiffSyn 官方对照表（各 run 自带 json/csv 与本 API 同口径并列）', '',
           '| 基线 | 本 API W2 | 官方 WSD | 本 API MMD | 官方 MMD | 本 API COV-P | 官方 COV-P |',
           '|---|---|---|---|---|---|---|']
    for _, r in tbl.iterrows():
        off = offcols.get(r['tag'], {})
        L2.append('| %s | %.4f | %s | %.4f | %s | %.4f | %s |'
                  % (r['name'], r['W2'], fmt(off.get('off_WSD')), r['MMD'], fmt(off.get('off_MMD')),
                     r['COVP'], fmt(off.get('off_COVP'))))
    L2 += ['', '注：本 API W2 联合 = 官方 `wasserstein_distance` (geomloss Sinkhorn) wrap，'
           'MMD/COV 同为官方实现 wrap；归一空间 = 官方 `utils.scale_x_syn_ratio`（逐列全人口 max）'
           '→ 与官方读数同口径可比。',
           '', '## 口径登记与 N/A 项', '',
           '- ② MMD / ① W2：官方 `scale_x_syn_ratio` 人口 max 归一空间（与官方 `eval.py` 一致）。',
           '- W1 / COV-P·R / 物理有效率：原始量纲。',
           '- ⑤ 条件一致率：scorer 口径①（12 列 → zeo_key 全人口 softmax 头，`wp5_g3_scorer.pkl`）。',
           '- ⑥ 凸包外：骨架级分离超平面 LP 判定；因逐行 LP 成本约束，读数取**子采样上限 3000 行**（seed 0）；'
           '少数系统 LP 数值异常 → 逐列盒近似下界（API `n_box_fallback` 计数）。',
           '- ⑦ 外推指数：**N/A** —— 官方基线仅 `system` 切分，无 `time` 切分预测产物（M10 时间外推线另跑）。',
           '- KDE / TabDDPM：按裁定跳过（官方仓库无实现）。', '',
           '> 复现：`python wp5_baseline_bench_v0.py`（幂等覆盖写）。']
    return '\n'.join(L2) + '\n'

def fmt(v):
    return 'n/a' if v is None or (isinstance(v, float) and np.isnan(v)) else '%.4f' % v

if __name__ == '__main__':
    main()
