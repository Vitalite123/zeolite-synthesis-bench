# -*- coding: utf-8 -*-
"""
ZeoliteSynthesisBench v1 — Track A 统一评估主表
================================================================================
预注册: Documentation/ZeoliteSynthesisBench_v1_规格_骨架.md v1.1 (FROZEN 2026-10-23)
协议: 123 合格体系 × 200 样本/体系；七类指标全部经 wp5_metrics_api_v0
方法: KDE-QT / TabDDPM-MLP / Random-Uniform / FASE-DDPM v0 / FASE-DDPM v0.5
产出:
  Data/wp5_bench_v1_table.csv
  Data/wp5_bench_v1_report.txt
  Data/wp5_bench_v1_manifest.json
  Data/wp5_bench_v1_by_system.csv
  Documentation/ZeoliteSynthesisBench_v1_基准主表_骨架.md
  opensource/fase/leaderboard/index.html
用法: python wp5_bench_v1_table.py
"""
import os, sys, json, time, math, hashlib, argparse, collections, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZG = os.path.join(BASE, 'vendor', 'zeosyn_gen')
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'metrics'))
sys.path.insert(0, os.path.join(BASE, 'baselines')); sys.path.insert(0, ZG)

import wp5_metrics_api_v0 as api
M = api.ZeoSynMetrics()
import wp3_ddpm_v0 as w3

DS_PT = os.path.join(BASE, 'weights', 'wp3_ddpm_v0_dataset.pt')
SCORER_PKL = os.path.join(BASE, 'weights', 'wp5_g3_scorer.pkl')
TABLE_CSV = os.path.join(BASE, 'data', 'wp5_bench_v1_table.csv')
SYS_CSV = os.path.join(BASE, 'data', 'wp5_bench_v1_by_system.csv')
REPORT = os.path.join(BASE, 'data', 'wp5_bench_v1_report.txt')
MANIFEST = os.path.join(BASE, 'data', 'wp5_bench_v1_manifest.json')
DOC_MD = os.path.join(BASE, 'data', 'bench_table_notes.md')
HTML = os.path.join(BASE, 'data', 'leaderboard.html')

NS = 200
METHODS = [
    ('FASE-DDPM v0',   os.path.join(BASE, 'data', 'samples', 'wp5_g3_formal_v0_syn.parquet'),  'diffusion'),
    ('FASE-DDPM v0.5', os.path.join(BASE, 'data', 'samples', 'wp5_g3_formal_v05_syn.parquet'), 'diffusion'),
    ('KDE-QT',         os.path.join(BASE, 'data', 'samples', 'wp5_bench_v1_kde_syn.parquet'),  'kde'),
    ('TabDDPM-MLP',    os.path.join(BASE, 'data', 'samples', 'wp5_bench_v1_tabddpm_syn.parquet'), 'tabddpm'),
    ('Random-Uniform', os.path.join(BASE, 'data', 'samples', 'wp5_bench_v1_random_syn.parquet'), 'random'),
]

L = []
def log(s=''):
    L.append(s); print(s, flush=True)

def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

def load_all():
    p = __import__('torch').load(DS_PT, weights_only=False)
    X = np.asarray(p['syn_raw'], dtype=np.float64)
    Z = np.asarray(p['zeo_key']); O = np.asarray(p['osda_key'])
    tr, te = p['train_idx'].numpy(), p['test_idx'].numpy()
    c = collections.Counter(zip(Z[te], O[te]))
    elig = sorted([k for k in c if c[k] >= 5 and k[0] != 'Dense/Amorphous'])
    train_df = pd.DataFrame(X[tr], columns=M.cols); train_df['zeo'] = Z[tr]
    test_df = pd.DataFrame(X[te], columns=M.cols); test_df['zeo'] = Z[te]; test_df['osda'] = O[te]
    tr_zeo = pd.Series(Z[tr]).value_counts().to_dict()
    tier = {k: ('rich(>=50)' if tr_zeo.get(k[0], 0) >= 50 else
                'mid(5-49)' if tr_zeo.get(k[0], 0) >= 5 else 'few(<5)') for k in elig}
    return p, train_df, test_df, elig, tier, {k: int(c[k]) for k in elig}

def eval_method(name, path, kind, train_df, test_df, elig, tier, n_true, clf, cls_idx):
    if not os.path.exists(path):
        log('[跳过] %s: 缺 %s' % (name, os.path.basename(path))); return None, None
    pred = pd.read_parquet(path)
    assert list(pred.columns[:len(M.cols)]) == M.cols, '%s 列序不符' % name
    cnt = pred.groupby(['zeo', 'osda']).size()
    assert all(cnt.get(k, 0) == NS for k in elig), '%s 存在非 %d 行的体系' % (name, NS)
    log('--- %s (%s, %d 行) ---' % (name, kind, len(pred)))

    rows = []
    for z_, o_ in elig:
        p_i = pred[(pred['zeo'] == z_) & (pred['osda'] == o_)][M.cols]
        t_i = test_df[(test_df['zeo'] == z_) & (test_df['osda'] == o_)][M.cols]
        fs = M.full_suite(p_i, t_i, ref=train_df)
        P = clf.predict_proba(p_i.to_numpy())
        tgt = np.array([cls_idx[z_]] * len(p_i))
        p_t = P[np.arange(len(p_i)), tgt]
        top1 = float((P.argmax(1) == tgt).mean())
        rows.append(dict(zeo=z_, osda=o_, tier=tier[(z_, o_)], n_true=n_true[(z_, o_)],
                         W2=fs['W2_joint'], MMD=fs['MMD'], W1=fs['W1_mean'],
                         COVP=fs['COV-P'], COVR=fs['COV-R'],
                         PV=M.physical_validity_rate(p_i, 'strict')['rate'],
                         CC05=float((p_t >= 0.5).mean()), CC02=float((p_t >= 0.2).mean()),
                         CCmeanP=float(p_t.mean()), top1=top1))
    per = pd.DataFrame(rows)

    pv = M.physical_validity_rate(pred[M.cols], 'strict')
    nn_g = M.nn_distance(pred[M.cols], train_df[M.cols], ref=train_df[M.cols])
    nn_s = M.nn_distance(pred[M.cols], train_df[M.cols], ref=train_df[M.cols],
                         gen_sys=pred['zeo'].to_numpy(), train_sys=train_df['zeo'].to_numpy())
    hull = M.outside_hull_fraction(pred[M.cols], train_df[M.cols], pred['zeo'].to_numpy(),
                                   train_df['zeo'].to_numpy(), ref=train_df[M.cols])
    box = M.outside_box_fraction(pred[M.cols], train_df[M.cols], ref=train_df[M.cols])
    agg = dict(method=name, kind=kind, n_systems=len(per), n_rows=len(pred),
               W2=per['W2'].mean(), MMD=per['MMD'].mean(), W1=per['W1'].mean(),
               COVP=per['COVP'].mean(), COVR=per['COVR'].mean(),
               PV_sys=per['PV'].mean(), PV_global=pv['rate'],
               NN_g_median=nn_g['median'], NN_g_mean=nn_g['mean'],
               NN_s_median=nn_s['median'], NN_s_mean=nn_s['mean'],
               HULL_out=hull['fraction_outside'], HULL_n=hull['n'],
               HULL_box_fallback=hull['n_box_fallback'], HULL_groups_no_train=hull['n_groups_no_train'],
               NN_s_no_train=nn_s['n_no_train'],
               BOX_out=box['fraction_outside'],
               CC_frac05=per['CC05'].mean(), CC_frac02=per['CC02'].mean(),
               CC_meanP=per['CCmeanP'].mean(), top1=per['top1'].mean())
    for t in ['rich(>=50)', 'mid(5-49)', 'few(<5)']:
        s = per[per['tier'] == t]
        agg['COVP_' + t.split('(')[0]] = (s['COVP'].mean() if len(s) else float('nan'))
    log('%s: COV-P %.4f | W2 %.4f | MMD %.4f | PV %.4f | CC@0.5 %.4f | NN中位 %.4f'
        % (name, agg['COVP'], agg['W2'], agg['MMD'], agg['PV_global'],
           agg['CC_frac05'], agg['NN_g_median']))
    return agg, per

def fmt(x, k=4):
    if isinstance(x, float) and (math.isnan(x)):
        return '—'
    return ('%.' + str(k) + 'f') % x if isinstance(x, float) else str(x)

def write_html(df):
    os.makedirs(os.path.dirname(HTML), exist_ok=True)
    cols = ['method', 'COVP', 'W2', 'MMD', 'PV_global', 'CC_frac05', 'NN_g_median', 'HULL_out']
    head = ''.join('<th>%s</th>' % c for c in cols)
    body = ''
    for _, r in df.iterrows():
        body += '<tr>' + ''.join('<td>%s</td>' % fmt(r[c]) for c in cols) + '</tr>'
    html = """<!doctype html><html lang="zh"><meta charset="utf-8">
<title>ZeoliteSynthesisBench v1 — Leaderboard (Track A)</title>
<style>body{font-family:system-ui,sans-serif;max-width:980px;margin:40px auto;padding:0 16px}
table{border-collapse:collapse;width:100%%}th,td{border-bottom:1px solid #ddd;padding:8px;text-align:right}
th:first-child,td:first-child{text-align:left}caption{text-align:left;font-weight:600;padding:8px 0}</style>
<h1>ZeoliteSynthesisBench v1 — Track A Leaderboard</h1>
<p>123 systems × 200 samples/system. Protocol: <code>Documentation/ZeoliteSynthesisBench_v1_规格_骨架.md</code> (frozen 2026-10-23).</p>
<table><caption>Main results (per-system mean unless noted)</caption><thead><tr>%s</tr></thead><tbody>%s</tbody></table>
<p>Generated %s from <code>Data/wp5_bench_v1_table.csv</code>.</p></html>""" % (
        head, body, time.strftime('%Y-%m-%d %H:%M'))
    with open(HTML, 'w', encoding='utf-8') as f:
        f.write(html)

def write_md(tab):
    """主表落盘 (版本化, 入 git) — 七类指标全列 + 外推分层 + 口径登记"""
    hdr = ['方法', '类型', '体系', 'W2', 'MMD', 'W1均', 'COV-P', 'COV-R',
           '物理有效†', 'NN(全局)', 'NN(骨架)', '骨架凸包外‡', '盒外',
           'CC@0.5', 'CC@0.2', 'top-1', 'COV-P rich', 'COV-P mid', 'COV-P few']
    keys = ['method', 'kind', 'n_systems', 'W2', 'MMD', 'W1', 'COVP', 'COVR',
            'PV_global', 'NN_g_median', 'NN_s_median', 'HULL_out', 'BOX_out',
            'CC_frac05', 'CC_frac02', 'top1', 'COVP_rich', 'COVP_mid', 'COVP_few']
    kind_zh = {'diffusion': '扩散', 'kde': 'KDE', 'tabddpm': '表格扩散', 'random': '均匀下界'}
    lines = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    for _, r in tab.iterrows():
        cells = []
        for k in keys:
            v = r[k]
            if k == 'kind':
                cells.append(kind_zh.get(v, v))
            elif k == 'n_systems':
                cells.append('%d' % int(v))
            elif k == 'method':
                cells.append(str(v))
            else:
                cells.append(fmt(float(v)))
        lines.append('| ' + ' | '.join(cells) + ' |')
    md = """# ZeoliteSynthesisBench v1 — Track A 主表（自实现基线 + FASE）

> 预注册规格：[ZeoliteSynthesisBench_v1_规格_骨架.md](ZeoliteSynthesisBench_v1_规格_骨架.md) v1.1（FROZEN 2026-10-23）。
> 运行时间 %s｜数据 `%s`（md5 `%s`）｜scorer `%s`（md5 `%s`）。

## 表 A｜Track A 主表（%d 体系 × %d 样本/体系，逐体系等权均值）

%s

† 物理有效率 = 全局 strict 水热窗口（cryst_temp ∈ [20,350] °C，cryst_time ∈ (0,20000]）。
‡ 骨架凸包外 = 骨架级（同 zeo 码 train 行）分离超平面 LP 判定，**全池 24,600 行**精确判定（与 G3 正式读数同口径，无子采样）。**已登记的结构性缺口**：7 个合格体系的 zeo 码在 train 侧无行（1,400 行）→ NN(骨架) 记 NaN、凸包外按定义记"外"（G3 读数同口径）；LP 数值异常组按组退化逐列盒判定（每方法 1–3 组，≤600 行 = 2.4%%），取外比例下界。
CC（条件一致率）= scorer v0.5（`wp5_g3_scorer.pkl`，12 列 → zeo_key），读数 = P(target) 的 frac@τ；真实行参照 = 合格体系 test 侧 2,087 行的 frac@0.5 = 0.8730（`Data/wp5_g3_formal_report.txt`）。

## 口径与复现

- W2/MMD：官方 `utils.scale_x_syn_ratio` 全人口 max 归一空间；W1/COV：原始量纲（与 G3 正式读数和 v0 对照表同口径）。
- 五方法同协议：123 合格体系（test 侧 n_true≥5 且 zeo≠Dense/Amorphous）× 200 样本，seed 42；FASE 行直接复用 G3 落盘 parquet。
- ⑦ 外推指数：Track A 正式定义留待 time 切分产物（M10 线），本表以 COV-P 骨架支撑分层（rich/mid/few）作代理登记。
- 复现：`python wp5_bench_v1_selfbaselines.py` → `python wp5_bench_v1_table.py`（幂等覆盖写；`--from-csv` 只由 `Data/wp5_bench_v1_table.csv` 重写本表与静态页）。
""" % (time.strftime('%Y-%m-%d %H:%M'), os.path.basename(DS_PT), md5(DS_PT),
       os.path.basename(SCORER_PKL), md5(SCORER_PKL),
       int(tab['n_systems'].max()), NS, '\n'.join(lines))
    with open(DOC_MD, 'w', encoding='utf-8') as f:
        f.write(md)

def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument('--from-csv', action='store_true', help='只由已有 CSV 重写主表/静态页 (不重算指标)')
    if ap.parse_args().from_csv:
        tab = pd.read_csv(TABLE_CSV)
        write_html(tab); write_md(tab)
        log('由 CSV 重写: %s + %s' % (os.path.relpath(DOC_MD, BASE), os.path.relpath(HTML, BASE)))
        return
    log('ZeoliteSynthesisBench v1 — Track A 统一评估 | %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
    p, train_df, test_df, elig, tier, n_true = load_all()
    import pickle
    with open(SCORER_PKL, 'rb') as f:
        sc = pickle.load(f)
    clf = sc['model']; cls_idx = {c: i for i, c in enumerate(clf.classes_)}
    log('协议: %d 体系 × %d 样本 | scorer %s (%d 类)'
        % (len(elig), NS, os.path.basename(SCORER_PKL), len(clf.classes_)))

    aggs, pers = [], []
    for name, path, kind in METHODS:
        a, per = eval_method(name, path, kind, train_df, test_df, elig, tier, n_true, clf, cls_idx)
        if a is not None:
            aggs.append(a); pers.append(per)
    tab = pd.DataFrame(aggs)
    tab.to_csv(TABLE_CSV, index=False)
    sysdf = pd.concat([p.assign(method=n) for n, p in zip(tab['method'], pers)], ignore_index=True)
    sysdf.to_csv(SYS_CSV, index=False)

    manifest = {'task': 'ZeoliteSynthesisBench v1 Track A unified table',
                'spec': 'Documentation/ZeoliteSynthesisBench_v1_规格_骨架.md v1.1 FROZEN',
                'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'protocol': {'systems': len(elig), 'ns': NS, 'seed': 42,
                             'universe': 'test n_true>=5, zeo!=Dense/Amorphous'},
                'dataset_md5': md5(DS_PT), 'scorer_md5': md5(SCORER_PKL),
                'methods': {m: md5(f) for m, f, _ in METHODS if os.path.exists(f)},
                'duration_min': round((time.time() - t0) / 60, 1)}
    with open(MANIFEST, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')

    write_html(tab)
    write_md(tab)
    log('完成: %.1f min | %s | %s | %s' % (manifest['duration_min'],
                                     os.path.basename(TABLE_CSV), os.path.relpath(DOC_MD, BASE),
                                     os.path.relpath(HTML, BASE)))

if __name__ == '__main__':
    main()
