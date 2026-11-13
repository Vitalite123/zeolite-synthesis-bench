# -*- coding: utf-8 -*-
"""
WP5 指标正式读数 (周计划 2026-10-23~10-29 P0-2) —— G3 中期线 COV-P ≥0.75 首正式判定
================================================================================
输入: DDPM v0 首训基线 + v0.1 迭代 (双模型同协议采样, 严格可比)
协议: 正式体系集 = test 侧 (zeo,osda) 体系 n_true≥5 且非 Dense/Amorphous 全体 (探针: 123 体系,
      2087 真实行) × 200 样本/体系 (seed 42, cond_scale 0.75 官方口径) —— COV pred 集规模纪律
      与官方协议同量级 (官方 coverage 对 pred>200 强制 sample(200), 200=官方截断上限)
七类指标 (wp5_metrics_api_v0.py ZeoSynMetrics):
  WSD(W2_joint 官方 Sinkhorn)/MMD/W1/COV-P·COV-R (逐体系 full_suite → 等权均值聚合,
  COV 语义不可跨体系合并计算 — scan 协议同型) + 物理有效率 (strict, 全体池单读, 域上限 0.9842
  参照) + 条件一致率 (scorer v0.5) + 新颖性 (骨架级 NN + 骨架级凸包外 LP + 全局盒外下界)
  + 外推指数 (骨架支撑度分层代理 v0 — 正式 time 切分衰减比留 M10 WP5 时间外推实验首跑)

条件一致率 scorer v0.5 (= L3-diffsyn, 自训自用评估工具):
  LGBM 多分类 12 列原始量纲 → zeo_key 198 类 (含 Dense/Amorphous 负类 4588 行支撑);
  全人口训练 (test 体系 train 侧 0 支撑 — system split 严格体系级 heldout, 全人口训练保证
  target 类有真实配方支撑, 判别器仲裁语义与官方 DiffSyn 一致); 读数 = P_softmax(target),
  双档 tau (0.5 主 / 0.2 参照) + top-1 一致率; 真实行对照三件套 (全人口自反上限 +
  test 侧真实行未见参照 — 生成读数以 vs 真实上限保持率归因, 类先验偏差已知限制登记)
  ⚠ 配置修正 (探针实证): 初版 200轮/lr0.1/min_leaf20 在 198 类全人口上发生 softmax margin
  塌缩 — rowmax 均值 1.0000 + in-sample top1 仅 0.135, 大支撑类 (MFI 2214 行) 预测数 0,
  argmax 锁死在 AFI/BEA/TON/BEC 共激发类 (lr×轮数预算过大 + 叶粒度过细, 非目标类树协同
  饱和后真类梯度被 1/198 稀释无法翻盘); 修正 = 300轮/lr0.05/min_leaf200 → in-sample
  top1 0.888 / frac@0.5 0.844 / rowmax 0.823 校准健康 (探针四配置对比选定, 病态配置确定性复现)
新颖性分组 = 骨架级 (同 zeo 码 train 行, 含不同 osda — test 体系体系级 train 支撑全 0,
  体系级分组无定义; 骨架级 = 生成外推语义, WP5 demo 同型)
用法: python wp5_g3_formal_read.py scorer|read|all
产出: Data/wp5_g3_scorer.pkl / wp5_g3_formal_{v0,v01}_syn.parquet / wp5_g3_formal_report.txt
      / wp5_g3_formal_manifest.json (gitignored) + Documentation/wp5_g3_covp_by_system.png (入 git)
"""
import os, sys, json, time, pickle, hashlib, argparse, warnings, subprocess
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZG = os.path.join(BASE, 'vendor', 'zeosyn_gen')
os.chdir(ZG); sys.path.insert(0, ZG)
sys.path.insert(0, BASE)                       # 任意导入上下文可解析 wp3 模块
sys.path.insert(0, os.path.join(BASE, 'metrics'))
sys.path.insert(0, os.path.join(BASE, 'baselines'))

import torch

import wp3_ddpm_v0 as w3
import wp3_ddpm_v0_1 as v01
from wp5_metrics_api_v0 import ZeoSynMetrics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------- 路径 ----------------
SCORER_PKL  = os.path.join(BASE, 'weights', 'wp5_g3_scorer.pkl')
REPORT      = os.path.join(BASE, 'data', 'wp5_g3_formal_report.txt')
MANIFEST    = os.path.join(BASE, 'data', 'wp5_g3_formal_manifest.json')
COVP_PNG    = os.path.join(BASE, 'figures', 'assets', 'wp5_g3_covp_by_system.png')

def pred_parq(tag):
    return os.path.join(BASE, 'data', 'samples', 'wp5_g3_formal_%s_syn.parquet' % tag.replace('.', ''))

def model_registry():
    """模型注册表 (渐进扩展: ckpt 存在即注册; v0.2+ 惰性导入)。
    条目 = (tag, dataset_path, ckpt_path, builder(device)->diffusion, osda_wrap|None)。
    osda_wrap: v0.4 起条件向量扩展 (osda 14→15, Si/Al 通道 null 哨兵) — G3 读数用 null 档
    (Si/Al 未指定, 与 v0-v0.3 任务定义可比); 可控性读数在 v0.4 脚本内另行执行。"""
    reg = [('v0', w3.DS_PT, w3.MODEL_PT, lambda d: w3.build_diffusion(d)[0], None),
           ('v0.1', v01.DS_PT_V01, v01.MODEL_PT_V01, lambda d: v01.build_diffusion_v01(d)[0], None)]
    for mod_name, tag, a_ds, a_ck, a_build in [
            ('wp3_ddpm_v0_2', 'v0.2', 'DS_PT_V02', 'MODEL_PT_V02', 'build_diffusion_v01'),
            ('wp3_ddpm_v0_3', 'v0.3', 'DS_PT_V03', 'MODEL_PT_V03', 'build_diffusion_v01'),
            ('wp3_ddpm_v0_4', 'v0.4', 'DS_PT_V04', 'MODEL_PT_V04', 'build_diffusion_v04'),
            ('wp3_ddpm_v0_4_1', 'v0.4.1', 'DS_PT_V041', 'MODEL_PT_V041', 'build_diffusion_v04'),
            ('wp3_ddpm_v0_5', 'v0.5', 'DS_PT_V05', 'MODEL_PT_V05', 'build_diffusion_v01')]:
        try:
            m = __import__(mod_name)
            ck = getattr(m, a_ck)
            if os.path.exists(ck):
                wrap = getattr(m, 'osda_null_wrap', None)
                reg.append((tag, getattr(m, a_ds), ck,
                            (lambda mm, ab: lambda d: getattr(mm, ab)(d)[0])(m, a_build), wrap))
        except Exception:
            pass
    return reg

NS          = 200                                  # 样本/体系 (官方 coverage 截断上限, COV 纪律)
COND_SCALE  = 0.75                                 # 官方口径 (scan 建议档位如有更优, 报告对照登记)
TAUS        = [0.5, 0.2]                           # 条件一致率双档 (API 默认 0.5 为主读数)

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
# scorer 阶段: L3-diffsyn 条件一致率 scorer v0.5
# =====================================================================
def scorer_stage():
    t0 = time.time()
    log('==================== WP5 G3 正式读数 — scorer v0.5 训练 (L3-diffsyn) ====================')
    payload = torch.load(v01.DS_PT_V01, weights_only=False)
    X = np.asarray(payload['syn_raw'])
    y = np.asarray(payload['zeo_key'])
    te_idx = payload['test_idx'].numpy()
    log('全人口 %d 行 × 12 列 → %d 类 (含 Dense/Amorphous 负类)' % (len(X), len(pd.unique(y))))

    import lightgbm as lgb
    # 300轮/lr0.05/min_leaf200: softmax margin 塌缩修正配置 (见文件头 ⚠ 注记)
    clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                             min_data_in_leaf=200,
                             random_state=0, n_jobs=-1, verbose=-1)
    clf.fit(X, y)
    cls_idx = {c: i for i, c in enumerate(clf.classes_)}

    # ---- 自反读数: 真实行 P(真实类) (全体拟合内上限 + test 侧未见参照) ----
    P = clf.predict_proba(X)
    p_true = P[np.arange(len(y)), np.array([cls_idx[t] for t in y])]
    top1 = (P.argmax(axis=1) == np.array([cls_idx[t] for t in y]))
    ref_all = {'mean_p': float(p_true.mean()),
               'frac_05': float((p_true >= 0.5).mean()), 'frac_02': float((p_true >= 0.2).mean()),
               'top1': float(top1.mean()), 'n': int(len(y))}
    p_te = p_true[te_idx]; top1_te = top1[te_idx]
    ref_test = {'mean_p': float(p_te.mean()),
                'frac_05': float((p_te >= 0.5).mean()), 'frac_02': float((p_te >= 0.2).mean()),
                'top1': float(top1_te.mean()), 'n': int(len(te_idx))}
    log('真实行自反上限 (全体拟合内): mean P=%.4f frac@0.5=%.4f frac@0.2=%.4f top1=%.4f'
        % (ref_all['mean_p'], ref_all['frac_05'], ref_all['frac_02'], ref_all['top1']))
    log('真实行未见参照 (test 侧):      mean P=%.4f frac@0.5=%.4f frac@0.2=%.4f top1=%.4f'
        % (ref_test['mean_p'], ref_test['frac_05'], ref_test['frac_02'], ref_test['top1']))

    with open(SCORER_PKL, 'wb') as f:
        pickle.dump({'model': clf, 'cols': list(w3.COLS), 'classes': list(clf.classes_),
                     'meta': {'data_version': 'zeobench_db_v1_8', 'n_pop': int(len(X)),
                              'n_classes': int(len(clf.classes_)),
                              'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                              'git_commit': git_head()}}, f)
    log('scorer 落盘: %s — 用时 %.1f min' % (os.path.basename(SCORER_PKL), (time.time() - t0) / 60))
    return ref_all, ref_test

def load_scorer():
    with open(SCORER_PKL, 'rb') as f:
        return pickle.load(f)

# =====================================================================
# read 阶段: 双模型正式读数 + G3 判定
# =====================================================================
def build_model(entry, device):
    tag, ds_pt, ckpt_pt, builder, osda_wrap = entry
    diffusion = builder(device)
    ck = torch.load(ckpt_pt, map_location=device, weights_only=False)
    diffusion.load_state_dict(ck['state_dict']); diffusion.eval()
    return diffusion, ck

def read_stage():
    t0 = time.time()
    M = ZeoSynMetrics()
    sc = load_scorer()
    clf = sc['model']; cls_idx = {c: i for i, c in enumerate(clf.classes_)}

    # ---- 正式体系集 (split 两版本逐位一致, 用 v0 payload 取) ----
    pv0 = torch.load(w3.DS_PT, weights_only=False)
    zk_all = np.asarray(pv0['zeo_key']); ok_all = np.asarray(pv0['osda_key'])
    tr_idx0 = pv0['train_idx'].numpy(); te_idx0 = pv0['test_idx'].numpy()
    key_te = pd.Series(list(zip(zk_all[te_idx0], ok_all[te_idx0])))
    counts = key_te.value_counts()
    eligible = [k for k in counts.index if counts[k] >= 5 and k[0] != 'Dense/Amorphous']
    log('==================== WP5 G3 正式读数 — %d 合格体系 × %d 样本 (seed 42, cond_scale %.2f) ===================='
        % (len(eligible), NS, COND_SCALE))
    n_true_map = {k: int(counts[k]) for k in eligible}

    # 骨架级 train 支撑 (外推代理分层维度)
    tr_zeo_counts = pd.Series(zk_all[tr_idx0]).value_counts().to_dict()
    sup_tier = {}
    for (z_, o_) in eligible:
        s = int(tr_zeo_counts.get(z_, 0))
        sup_tier[(z_, o_)] = 'rich(≥50)' if s >= 50 else ('mid(5-49)' if s >= 5 else 'few(<5)')
    log('骨架支撑分层: %s' % {v: sum(1 for x in sup_tier.values() if x == v)
                                for v in ['rich(≥50)', 'mid(5-49)', 'few(<5)']})

    # 真实行 test 参照 (合格体系 test 行, scorer 未见参照)
    te_mask = key_te.isin(eligible).to_numpy()
    true_te_rows = te_idx0[te_mask]
    ref_test = None
    if len(true_te_rows):
        Xraw0 = np.asarray(pv0['syn_raw'])
        Pt = clf.predict_proba(Xraw0[true_te_rows])
        tgt = zk_all[true_te_rows]
        p_te = Pt[np.arange(len(tgt)), np.array([cls_idx[t] for t in tgt])]
        top1_te = Pt.argmax(axis=1) == np.array([cls_idx[t] for t in tgt])
        ref_test = {'mean_p': float(p_te.mean()), 'frac_05': float((p_te >= 0.5).mean()),
                    'frac_02': float((p_te >= 0.2).mean()), 'top1': float(top1_te.mean()),
                    'n': int(len(tgt))}
        log('真实行未见参照 (合格体系 test %d 行): mean P=%.4f frac@0.5=%.4f top1=%.4f'
            % (ref_test['n'], ref_test['mean_p'], ref_test['frac_05'], ref_test['top1']))

    per_model = {}
    reg = model_registry()
    tags = [e[0] for e in reg]
    log('模型注册表 (%d): %s' % (len(tags), ', '.join(tags)))
    for entry in reg:
        tag, ds_pt, ckpt_pt, builder, osda_wrap = entry
        tm = time.time()
        payload = torch.load(ds_pt, weights_only=False)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        diffusion, ck = build_model(entry, device)
        X_raw = np.asarray(payload['syn_raw']); Z = payload['zeo_feat'].to(device); O = payload['osda_feat'].to(device)
        tr_idx = payload['train_idx'].numpy(); te_idx = payload['test_idx'].numpy()
        zk = np.asarray(payload['zeo_key']); ok = np.asarray(payload['osda_key'])
        qts = payload['qts']
        train_df = pd.DataFrame(X_raw[tr_idx], columns=M.cols)
        test_df = pd.DataFrame(X_raw[te_idx], columns=M.cols)
        test_df['zeo'], test_df['osda'] = zk[te_idx], ok[te_idx]
        log('--- 模型 %s (best-val @%d, val %.4f) ---' % (tag, ck['step'], ck['val_loss']))

        # ---- 采样: 每体系条件向量 (该体系首个 test 行) × NS 样本 ----
        rows0 = []
        for z_, o_ in eligible:
            mrow = np.where((zk == z_) & (ok == o_))[0]
            rows0.append(int(mrow[0]))
        zc = torch.stack([Z[r] for r in rows0]).repeat_interleave(NS, dim=0)
        oc = torch.stack([O[r] for r in rows0]).repeat_interleave(NS, dim=0)
        if osda_wrap is not None:                     # v0.4+: null 档条件包装 (osda 14→15, Si/Al 哨兵 -1.0)
            oc = osda_wrap(oc)
        torch.manual_seed(42)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device != 'cpu')):
            s = diffusion.sample(zeo=zc, osda=oc, cond_scale=COND_SCALE,
                                 batch_size=zc.shape[0]).squeeze(1).cpu().numpy()
        pred = w3.inv_qt(s, qts)
        gen_keys = np.repeat(np.arange(len(eligible)), NS)
        pred_df = pd.DataFrame(pred, columns=M.cols)
        pred_df['zeo'] = [eligible[i][0] for i in gen_keys]
        pred_df['osda'] = [eligible[i][1] for i in gen_keys]
        pred_df.to_parquet(pred_parq(tag), index=False)
        log('采样 %d 行落盘 %s (%.1f min)' % (len(pred_df), os.path.basename(pred_parq(tag)),
                                              (time.time() - tm) / 60))

        # ---- 逐体系 full_suite + 条件一致率 (API 调用, 七类全过 API 纪律) ----
        scorer_fn = lambda df, tgt: clf.predict_proba(df[M.cols].to_numpy())[:, cls_idx[tgt]]
        rows = []
        for i, (z_, o_) in enumerate(eligible):
            p_i = pred_df[(pred_df['zeo'] == z_) & (pred_df['osda'] == o_)][M.cols]
            t_i = test_df[(test_df['zeo'] == z_) & (test_df['osda'] == o_)][M.cols]
            fs = M.full_suite(p_i, t_i, ref=train_df)
            cc05 = M.conditional_consistency(p_i, scorer_fn, target=z_, tau=0.5)
            cc02 = M.conditional_consistency(p_i, scorer_fn, target=z_, tau=0.2)
            top1_hit = (clf.predict_proba(p_i.to_numpy()).argmax(axis=1) == cls_idx[z_])
            rows.append(dict(zeo=z_, osda=o_, n_true=n_true_map[(z_, o_)],
                             tier=sup_tier[(z_, o_)],
                             W2=fs['W2_joint'], MMD=fs['MMD'], W1=fs['W1_mean'],
                             COVP=fs['COV-P'], COVR=fs['COV-R'],
                             mean_p=cc05['mean_p'], frac05=cc05['frac_ge_tau'],
                             frac02=cc02['frac_ge_tau'], top1=float(top1_hit.mean())))
        per = pd.DataFrame(rows)
        per_model[tag] = per

        # ---- 新颖性 (骨架级; gen/train 全体一次调用) + 全局盒外 ----
        nn = M.nn_distance(pred_df, train_df, ref=train_df,
                           gen_sys=pred_df['zeo'].to_numpy(), train_sys=zk[tr_idx])
        hull = M.outside_hull_fraction(pred_df, train_df,
                                       gen_group=pred_df['zeo'].to_numpy(),
                                       train_group=zk[tr_idx], ref=train_df)
        box = M.outside_box_fraction(pred_df, train_df, ref=train_df)
        # ---- 物理有效率 (全体池) ----
        pv = M.physical_validity_rate(pred_df, 'strict')
        per_model[tag] = {'per': per, 'nn': nn, 'hull': hull, 'box': box, 'pv': pv,
                          'ckpt_step': int(ck['step'])}
        log('%s 聚合: COV-P %.4f / COV-R %.4f / W2 %.4f / PV(strict) %.4f / '
            'frac@0.5 %.4f / NN中位 %.4f / 凸包外 %.4f (%.1f min)'
            % (tag, per['COVP'].mean(), per['COVR'].mean(), per['W2'].mean(), pv['rate'],
               per['frac05'].mean(), nn['median'], hull['fraction_outside'],
               (time.time() - tm) / 60))

    # ================= 报告 =================
    latest = tags[-1]
    R = []
    R.append('WP5 G3 中期线正式读数 — %s' % time.strftime('%Y-%m-%d %H:%M'))
    R.append('=' * 78)
    R.append('协议: %d 合格体系 (test n_true≥5 非 Dense/Amorphous; 真实行 %d) × %d 样本/体系,'
             % (len(eligible), int(sum(n_true_map.values())), NS))
    R.append('      seed 42, cond_scale %.2f (官方口径); %d 模型同协议严格可比: %s;'
             % (COND_SCALE, len(tags), ' / '.join(tags)))
    R.append('      COV pred 集规模纪律 = 官方截断上限 200; 数据版本 zeobench_db_v1_8')
    R.append('scorer v0.5 = LGBM 多分类 12 列→198 类 (全人口训练, Dense/Amorphous 负类支撑);')
    R.append('      类先验偏差已知限制 — 一致率以 vs 真实行对照保持率归因')
    R.append('')
    cw = 10
    hdr = '%-22s' % '指标' + ''.join(' %10s' % t for t in tags) + ' %10s %12s' % ('G3 线', '判定(%s)' % latest)
    R.append(hdr); R.append('-' * len(hdr))
    def judged(x, line, higher=True):
        ok = (x >= line) if higher else (x <= line)
        return 'PASS' if ok else 'GAP'
    for key, label, line, higher, fmt in [
            ('COVP', 'COV-P (逐体系均值)', 0.75, True, '%.4f'),
            ('COVR', 'COV-R (逐体系均值)', None, None, '%.4f'),
            ('W2', 'W2_joint (体系级)', None, None, '%.4f'),
            ('MMD', 'MMD (体系级)', None, None, '%.4f'),
            ('W1', 'W1_mean (体系级)', None, None, '%.4f'),
            ('frac05', '一致率 frac@0.5', 0.90, True, '%.4f'),
            ('frac02', '一致率 frac@0.2', None, None, '%.4f'),
            ('top1', '一致率 top-1', None, None, '%.4f'),
            ('mean_p', '一致率 mean P(target)', None, None, '%.4f')]:
        vals = [float(per_model[t]['per'][key].mean()) for t in tags]
        g3 = '' if line is None else ('≥%.2f' % line if higher else '≤%.2f' % line)
        jd = '' if line is None else judged(vals[-1], line, higher)
        R.append('%-22s' % label + ''.join(' %10s' % (fmt % v) for v in vals)
                 + ' %10s %12s' % (g3, jd))
    for t in tags:
        pv = per_model[t]['pv']
        R.append('物理有效率 strict (%s): %.4f vs ≥0.98 → %-4s (域上限 0.9842 参照; 失效明细 %s)'
                 % (t, pv['rate'], judged(pv['rate'], 0.98, True), pv['fail']))
    R.append('')
    R.append('递进轨迹: COV-P ' + ' → '.join('%s %.4f' % (t, per_model[t]['per']['COVP'].mean()) for t in tags)
             + ' | frac@0.5 ' + ' → '.join('%s %.4f' % (t, per_model[t]['per']['frac05'].mean()) for t in tags))
    R.append('')
    R.append('新颖性 (骨架级分组 = 生成外推语义; 体系级 train 支撑全 0 — system split 严格 heldout):')
    for t in tags:
        d = per_model[t]
        R.append('  %s: NN mean %.4f / 中位 %.4f / p90 %.4f (无训练骨架组 %d) | 凸包外 %.4f '
                 '(LP 组 %d / 盒兜底 %d / 无训练组 %d) | 全局盒外 %.4f'
                 % (t, d['nn']['mean'], d['nn']['median'], d['nn']['p90'], d['nn']['n_no_train'],
                    d['hull']['fraction_outside'], d['hull']['n_sep_lp'], d['hull']['n_box_fallback'],
                    d['hull']['n_groups_no_train'], d['box']['fraction_outside']))
    R.append('')
    R.append('外推指数代理 v0 (骨架支撑度分层; 正式 time 切分衰减比留 M10 时间外推实验):')
    for metric, mlab in [('COVP', 'COV-P'), ('frac05', 'frac@0.5')]:
        R.append('  [%s]' % mlab + ''.join(' %10s' % t for t in tags))
        for tier in ['rich(≥50)', 'mid(5-49)', 'few(<5)']:
            vals = []
            n_sys = 0
            for t in tags:
                m = per_model[t]['per']; s = m[m['tier'] == tier]
                vals.append(float(s[metric].mean()) if len(s) else float('nan'))
                n_sys = len(s)
            if n_sys == 0:
                continue
            R.append('  %-12s %5d' % (tier, n_sys) + ''.join(' %10.4f' % v for v in vals))
    if ref_test is not None:
        R.append('')
        R.append('一致率归因 (真实行未见参照, 合格体系 test %d 行): mean P=%.4f frac@0.5=%.4f top1=%.4f'
                 % (ref_test['n'], ref_test['mean_p'], ref_test['frac_05'], ref_test['top1']))
        for t in tags:
            b = float(per_model[t]['per']['frac05'].mean())
            R.append('  %s 生成 frac@0.5=%.4f — 真实参照保持率 %.1f%% (%s)'
                     % (t, b, 100 * b / max(ref_test['frac_05'], 1e-9),
                        '生成质量受限' if b < 0.9 * ref_test['frac_05'] else '接近口径上限, 判定以保持率为准'))
    R.append('')
    R.append('G3 中期线判定 (%s = 最新):' % latest)
    cl = float(per_model[latest]['per']['COVP'].mean())
    R.append('  COV-P %.4f vs 0.75 中期线: %s' % (cl, '过线' if cl >= 0.75 else '差距 %.4f (差距归因见分层表)' % (0.75 - cl)))
    R.append('  物理有效率 %.4f vs 0.98: %s (域上限 0.9842 参照)' % (per_model[latest]['pv']['rate'],
             '过线' if per_model[latest]['pv']['rate'] >= 0.98 else '未达线'))
    w2s = [(t, float(per_model[t]['per']['W2'].mean())) for t in tags]
    R.append('  W2 递进: %s (逐行 25 样本口径的 WSD 对照另见各版 eval 报告)'
             % ' → '.join('%s %.4f' % (t, v) for t, v in w2s))
    R.append('  条件一致率 frac@0.5 见上表 (类先验口径限制登记; scorer 适配器 v1 接 L1/L2 后复核)')
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n\n' + '\n'.join(R) + '\n')
    log('报告: %s' % REPORT)

    # ---- PNG: 逐体系 COV-P 对比 (排序曲线; 各模型 per 行序一致 = eligible 序) ----
    order = np.argsort(per_model[tags[0]]['per']['COVP'].to_numpy())
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['tab:gray', 'tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown']
    for ti, t in enumerate(tags):
        m = per_model[t]['per'].iloc[order].reset_index(drop=True)
        ax.plot(np.arange(len(m)), m['COVP'], lw=1.2, marker='.', ms=3,
                color=colors[ti % len(colors)], label=t)
    ax.axhline(0.75, color='tab:red', ls='--', lw=1, label='G3 中期线 0.75')
    ax.set_xlabel('体系 (按 %s COV-P 排序, n=%d)' % (tags[0], len(m))); ax.set_ylabel('COV-P')
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title('G3 中期线正式读数: 逐体系 COV-P (123 合格测试体系 × 200 样本, zeobench_db_v1_8)')
    fig.tight_layout(); fig.savefig(COVP_PNG, dpi=160); plt.close(fig)
    log('PNG: %s' % COVP_PNG)

    # ---- manifest ----
    manifest = {
        'task': 'WP5 指标正式读数 — G3 中期线 COV-P≥0.75 首正式判定 (周计划 2026-10-23~10-29 P0-2)',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'data_version': 'zeobench_db_v1_8',
        'code_md5': {'wp5_g3_formal_read.py': md5(os.path.abspath(__file__)),
                     'wp5_metrics_api_v0.py': md5(os.path.join(BASE, 'wp5_metrics_api_v0.py')),
                     'wp3_ddpm_v0.py': md5(w3.__file__), 'wp3_ddpm_v0_1.py': md5(v01.__file__)},
        'git_commit': git_head(),
        'protocol': {'n_systems': len(eligible), 'n_samples_per_sys': NS, 'seed': 42,
                     'cond_scale': COND_SCALE, 'cov_pred_discipline': '200 = 官方 coverage 截断上限',
                     'aggregation': '逐体系 full_suite 等权均值 (COV 不可跨体系合并)'},
        'scorer_v05': {'type': 'LGBM multiclass 12列→zeo_key 198 类 (L3-diffsyn)',
                       'n_estimators': 300, 'num_leaves': 31, 'lr': 0.05,
                       'min_data_in_leaf': 200,
                       'note': 'margin 塌缩修正配置 (200/lr.1/mdl20 → rowmax 1.0/top1 0.135 病态, 见文件头)',
                       'trained_on': '全人口 (test 体系 train 侧 0 支撑)',
                       'ref_test_true_rows': ref_test},
        'models': {tag: {'ckpt_step': per_model[tag]['ckpt_step'],
                         'COV-P': float(per_model[tag]['per']['COVP'].mean()),
                         'COV-R': float(per_model[tag]['per']['COVR'].mean()),
                         'W2': float(per_model[tag]['per']['W2'].mean()),
                         'frac05': float(per_model[tag]['per']['frac05'].mean()),
                         'physical_validity_strict': per_model[tag]['pv']['rate'],
                         'nn_median': per_model[tag]['nn']['median'],
                         'outside_hull': per_model[tag]['hull']['fraction_outside'],
                         'pred_parquet_md5': md5(pred_parq(tag))}
                   for tag in tags},
        'duration_min': round((time.time() - t0) / 60, 1),
    }
    with open(MANIFEST, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log('read 完成: 用时 %.1f min' % ((time.time() - t0) / 60))

# =====================================================================
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['scorer', 'read', 'all'])
    a = ap.parse_args()
    if a.stage in ('scorer', 'all'):
        scorer_stage()
    if a.stage in ('read', 'all'):
        read_stage()
