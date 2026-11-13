# -*- coding: utf-8 -*-
"""
merge_increment.py — WP1/T1.1 Q28④ 增量入库 + 切分扩展 + 六锚点评估
================================================================================
依据: Documentation/WP1_增量数据滚动扩容实施计划.md §5.1 步骤5-7 / §6.3
用法: python merge_increment.py --batch R4   (R1/R2/R3 模式复现 v1.2-v1.4 收口)

输入 (按 --batch 派生; R3 = lit_extract_R3 + 主库 v1.3 → v1.4):
  Data/lit_extract_{B}.parquet      W3 双模板一致+溯源可定位 recipe
                                    (R2 起含 trace_ok/trace_miss_fields 列)
  Data/lit_pool_{B}_filtered.csv    DOI → year/journal/title
  Data/lit_extract_R1_upgrade.csv   ⑤ b_unmatched 缩围升格 2 行定义
                                    (仅 R2 批次接入, 克数换算链在 notes 留档)
  Data/zeosyn_db_v1_{n}.parquet     主库 (R2: 输入 v1.2 → 输出 v1.3)
  Data/zeosyn_splits_v1_{n}.parquet 切分 (值级冻结扩展)
  其余同 R1 (dedup manifest / lit_fail / osda_descriptors / qc_config)

评估架构 (关键, 特征非行不变所致):
  build_features_matrix 的 scaler/中位数/描述符统计均在 df_main 上拟合 →
  评估基座恒用 v1.1 (剔 Tier A 23,488 行), 保证 v0.1/v0.2/v0.3 测试集特征
  逐位一致 (§6.3 测试集冻结); 各批增量行经 df_lf 通道以主库拟合统计变换:
    R1: X = [main11; lf; inc_R1]           锚点域 v0.1 = main11+lf
    R2: X = [main11; lf; inc_R1; inc_R2; up]  锚点域 v0.2 = main11+lf+inc_R1
  锚点复核 seed=42 六读数 assert (R1→V01 含 AUCPR; R2→V02 仅 AUROC,
  v0.2 报告未录 AUCPR); 而后锚点域/全量 3-seed 对照 → 边际收益 Δ。

处理:
  1. 框架归一: 材料名→IZA 码 (merlinoite→MER / DD3R→DDR; R4: MMU-1→EDI,
     cancrinite→CAN); 致密矿 megakalsilite ('Meg') 与 kalsilite (2026-10-10
     证伪, 原 kalsilite→ABW 剔除) 按主库惯例 framework_code=材料名 +
     absolute_neg; R4 增 amorphous 分支 (framework_code='amorphous' +
     absolute_neg + label1_detail='特殊值', 主库 3,768 行惯例);
     product1=framework_code; label1=positive (成功路线, label1_detail='IZA 框架码')
  2. 组成映射 (金标准 remediation: gel_composition_str 为一等证据, 结构化列兜底):
     SiO2→n_Si 1:1, Al2O3→n_Al×2, NaOH/KOH/CsOH→n_Na/n_K/n_Cs+n_OH,
     Na2O→n_Na×2 (2026-10-10 暂定惯例, 登记周会复核), K2O→n_K×2, BaO→n_Ba,
     HF→n_F, OSDA-OH 型→n_OH, 次有机组分→n_osda2 (TMA-OH/EDA);
     缺项置 0+旗标 (B 级惯例); seed 用量未抽→n_seed=0+旗标;
     全 n_ 列(除 n_seed)归一化 sum=1 (主库口径, 含 n_Ba); ratio 列用归一前摩尔;
     电荷平衡 pos=n_Na+n_K+n_Li+n_Cs+q·n_osda1 (DATA_DICT: 阳离子+OSDA 电荷)
  3. route_id = incr_{B}_{doi_key}_{r##}; 149 列 schema 对齐 (qc 5 列按
     qc_config 标记; if_score/if_out 不重算, if_out=False 保守);
     R2 起 notes 记录 trace_ok (⑧ 溯源协议, 抽取值正文逐字可定位)
  4. DOI 泄漏审计: 增量 DOI × 输入主库 = 0 (assert)
  5. 切分扩展 splits_v1_{n+1}: 输入行值级冻结; 增量行 random=train(全),
     heldout=留出骨架行 excluded 其余 train, time_2018/2020=test
     (2022+ 天然考题, incr_zeo1 惯例; 评估侧 time 增量行全 excluded)
  6. ⑤ 升格行 (R2): lit_extract_R1_upgrade.csv 2 行经 build_upgrade_rows
     入库 (route_id=incr_R1_{doi_key}_up{##}, source_table=..._upgrade,
     n_Sn/n_P 直填, 归一化/QC 同 inc 口径)
  7. 六锚点评估: 锚点域 (上批口径) seed=42 六读数 assert → 锚点域/全量
     3-seed (42/43/44) 对照 → 边际收益 Δ (§6.3: 连续两批 <0.5pp 触发 G2 复议)

输出 (幂等覆盖写, 按 --batch 派生):
  R1: zeosyn_db_v1_2 / splits_v1_2 / l2_v0_2_incr_report.txt (收口态)
  R2: zeosyn_db_v1_3 / splits_v1_3 / l2_v0_3_incr_report.txt
  R3: zeosyn_db_v1_4 / splits_v1_4 / l2_v0_4_incr_report.txt
  R4: zeosyn_db_v1_5 / splits_v1_5 / l2_v0_5_incr_report.txt

工程偏离 (R2 决议 split_zeosyn_v11 惯例的两处调整, 均为 §6.3 测试集冻结服务):
  1. splits_v1_2 random 侧 incr 行全 train (偏离"新 DOI 池同协议同种子
     80/10/10 指派"): §6.3 边际收益监控要求测试集与 v0.1 基线逐位一致,
     36 行/22 DOI 规模不支撑 10% 池; 库级与评估口径就此统一无双轨
  2. heldout 侧 incr 含留出骨架行 excluded (lit_fail 先例) 而非整 DOI test:
     同为测试集冻结服务 (LTL 5 + LTA 6 = 11 行)

金标准 remediation (§5.4, 2026-10-10; 首跑 93.4%<95% 不通过后的修复重跑,
抽检结论留档 Data/lit_extract_R1_goldstandard_check.csv):
  1. chemmater.2c00773 七行剔除 (na_oh/solvent_mol 正文不可溯源, 系双模板
     同源推断; 逐样品组成在 SI Table S3, 列为 R2 SI 获取目标)
  2. 组成映射改由 gel_composition_str 解析 (修复: KOH/CsOH 误入 n_Na、
     K2O/BaO 丢失、Na2O 1:1+幻影 n_OH、HF n_F=0、EDA 不入归一化分母)
  3. 'Meg'→megakalsilite 致密矿 absolute_neg (原 'meg'→'MER' 假阳性)
  4. 零 Si 行拦截 (P1 r17: n_Si=n_Al=0 的 positive 行不入库)
"""
import json
import os
import re
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from comp_neg_knn import LOG_MOLE_COLS
from l2_v0_1_litfail import build_lf_rows, build_features_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 评估基座恒用 v1.1 (特征 scaler/统计拟合基准, 见 docstring 评估架构节)
DB_V11 = os.path.join(ROOT, 'data', 'zeosyn_db_v1_1.parquet')
SPLITS_V11 = os.path.join(ROOT, 'data', 'zeosyn_splits_v1_1.parquet')
MANIFEST = os.path.join(ROOT, 'data', 'zeosyn_db_dedup_manifest.csv')
UPGRADE_CSV = os.path.join(ROOT, 'data', 'lit_extract_R1_upgrade.csv')
QC_CFG = json.load(open(os.path.join(ROOT, 'qc_config.json'),
                        encoding='utf-8'))

HELD_FWS = ['LTL', 'SAS', 'ITG', 'RTH', 'SZR', 'LTA']   # v1.0 报告冻结清单
SEEDS = (42, 43, 44)

# v0.1 报告定稿读数 (seed=42, Data/l2_v0_1_litfail_report.txt 【六】节)
V01 = {'pure_random': 0.6034, 'pure_random_ap': 0.7388,
       'xtal_random': 0.6535, 'xtal_random_ap': 0.8876,
       'pure_heldout': 0.5258, 'xtal_heldout': 0.4919,
       'pure_time': 0.6221, 'xtal_time': 0.6814}

# v0.2 报告定稿读数 (Data/l2_v0_2_incr_report.txt 【四】节; 3-seed 均值
# std=0 系 LGBM 确定性 → seed=42 单 seed 复核可用; AUCPR 未记录于报告,
# R2 锚点复核以 AUROC 六读数为准, abs_neg FP 作参考读数)
V02 = {'pure_random': 0.5971, 'pure_heldout': 0.5264,
       'pure_time': 0.6237, 'xtal_random': 0.6639,
       'xtal_heldout': 0.4838, 'xtal_time': 0.6731, 'xtal_fp': 0.993}

# v0.3 报告定稿读数 (Data/l2_v0_3_incr_report.txt; R3 对照基线。
# AUCPR 未录于 v0.3 报告, 锚点复核仅 AUROC 六读数 + abs_neg FP 参考)
V03 = {'pure_random': 0.6059, 'pure_heldout': 0.5356,
       'pure_time': 0.6225, 'xtal_random': 0.6607,
       'xtal_heldout': 0.5177, 'xtal_time': 0.6893, 'xtal_fp': 0.995}

# v0.4 报告定稿读数 (Data/l2_v0_4_incr_report.txt; R4 对照基线。
# AUCPR 未录于 v0.4 报告, 锚点复核仅 AUROC 六读数 + abs_neg FP 参考)
V04 = {'pure_random': 0.6163, 'pure_heldout': 0.5427,
       'pure_time': 0.6240, 'xtal_random': 0.6475,
       'xtal_heldout': 0.5267, 'xtal_time': 0.6888, 'xtal_fp': 0.973}

# v0.5 报告定稿读数 (Data/l2_v0_5_incr_report.txt; R5 对照基线。
# AUCPR 未录于 v0.5 报告, 锚点复核仅 AUROC 六读数 + abs_neg FP 参考)
V05 = {'pure_random': 0.6083, 'pure_heldout': 0.5333,
       'pure_time': 0.6227, 'xtal_random': 0.6547,
       'xtal_heldout': 0.4987, 'xtal_time': 0.6836, 'xtal_fp': 0.993}

# v0.6 报告定稿读数 (Data/l2_v0_6_incr_report.txt; PP1 批 3-seed 均值)
V06 = {'pure_random': 0.6131, 'pure_heldout': 0.5429,
       'pure_time': 0.6230, 'xtal_random': 0.6557,
       'xtal_heldout': 0.5280, 'xtal_time': 0.6884, 'xtal_fp': 0.984}

# v0.7 报告定稿读数 (Data/l2_v0_7_incr_report.txt; PP2 批 3-seed 均值)
V07 = {'pure_random': 0.6277, 'pure_heldout': 0.5497,
       'pure_time': 0.6246, 'xtal_random': 0.6576,
       'xtal_heldout': 0.5304, 'xtal_time': 0.6922, 'xtal_fp': 0.991}

# ---- 批次配置 (--batch 派生) ----
# R2 金标准剔除集: §5.4 抽检 (≥95% 门槛) 不通过行在此登记后重跑;
# ⑧ 溯源协议已在 W3 聚合侧拦截 trace_fail 行, 此处为抽检二次兜底
BATCH_CFG = {
    'R1': dict(
        inc=os.path.join(ROOT, 'data', 'lit_extract_R1.parquet'),
        pool=os.path.join(ROOT, 'data', 'lit_pool_R1_filtered.csv'),
        db_in=DB_V11, sp_in=SPLITS_V11,
        db_out=os.path.join(ROOT, 'data', 'zeosyn_db_v1_2.parquet'),
        sp_out=os.path.join(ROOT, 'data', 'zeosyn_splits_v1_2.parquet'),
        rpt=os.path.join(ROOT, 'data', 'l2_v0_2_incr_report.txt'),
        prefix='incr_R1', src_table='lit_pool_R1_W3',
        anchor=V01, anchor_name='v0.1', new_name='v0.2',
        goldstd_exclude={'10.1021/acs.chemmater.2c00773'},
        goldstd_reason='goldstd fail: 组成不可溯源 (SI Table S3 未获取)',
        with_upgrade=False),
    'R2': dict(
        inc=os.path.join(ROOT, 'data', 'lit_extract_R2.parquet'),
        pool=os.path.join(ROOT, 'data', 'lit_pool_R2_filtered.csv'),
        db_in=os.path.join(ROOT, 'data', 'zeosyn_db_v1_2.parquet'),
        sp_in=os.path.join(ROOT, 'data', 'zeosyn_splits_v1_2.parquet'),
        db_out=os.path.join(ROOT, 'data', 'zeosyn_db_v1_3.parquet'),
        sp_out=os.path.join(ROOT, 'data', 'zeosyn_splits_v1_3.parquet'),
        rpt=os.path.join(ROOT, 'data', 'l2_v0_3_incr_report.txt'),
        prefix='incr_R2', src_table='lit_pool_R2_W3',
        anchor=V02, anchor_name='v0.2', new_name='v0.3',
        # §5.4 抽检 (2026-10-10): jacs.3c04328 T/t 为 Cu-MOR XANES 表征的
        # O2 活化条件 (450C/1h, Fig.9 图注) 非水热合成条件, 全文无凝胶配比,
        # MOR 为研究对象商用沸石 → 错抽行证伪剔除; ZMQ-1 6 条逐字溯源 PASS
        goldstd_exclude={'10.1021/jacs.3c04328'},
        goldstd_reason=('goldstd fail: T/t 为表征活化条件非合成条件'
                        ' (§5.4 溯源证伪)'),
        with_upgrade=True, upgrade_csv=UPGRADE_CSV,
        upgrade_prefix='incr_R1', upgrade_table='lit_pool_R1_W3_upgrade'),
    'R3': dict(
        inc=os.path.join(ROOT, 'data', 'lit_extract_R3.parquet'),
        pool=os.path.join(ROOT, 'data', 'lit_pool_R3_filtered.csv'),
        db_in=os.path.join(ROOT, 'data', 'zeosyn_db_v1_3.parquet'),
        sp_in=os.path.join(ROOT, 'data', 'zeosyn_splits_v1_3.parquet'),
        db_out=os.path.join(ROOT, 'data', 'zeosyn_db_v1_4.parquet'),
        sp_out=os.path.join(ROOT, 'data', 'zeosyn_splits_v1_4.parquet'),
        rpt=os.path.join(ROOT, 'data', 'l2_v0_4_incr_report.txt'),
        prefix='incr_R3', src_table='lit_pool_R3_W3',
        anchor=V03, anchor_name='v0.3', new_name='v0.4',
        goldstd_exclude=set(),   # §5.4 抽检 2/2 PASS (100%), 无剔除
        # ⑤ 升格 (2026-10-10 人工复核裁决): 63537-5 B#2-5 MOR-Cu 温度
        # 系列 4 行 (T=170/190/210/250); 5c00431 全部人工队列行不升格
        # (as-made FER 含 20-30 wt% Layered 次相, 双模板未捕获, 配比
        # x 值在 SI 层; 详见 lit_extract_R3_review 处置注记)
        with_upgrade=True,
        upgrade_csv=os.path.join(ROOT, 'data',
                                 'lit_extract_R3_upgrade.csv'),
        upgrade_prefix='incr_R3', upgrade_table='lit_pool_R3_W3_upgrade'),
    'R4': dict(
        inc=os.path.join(ROOT, 'data', 'lit_extract_R4.parquet'),
        pool=os.path.join(ROOT, 'data', 'lit_pool_R4_filtered.csv'),
        db_in=os.path.join(ROOT, 'data', 'zeosyn_db_v1_4.parquet'),
        sp_in=os.path.join(ROOT, 'data', 'zeosyn_splits_v1_4.parquet'),
        db_out=os.path.join(ROOT, 'data', 'zeosyn_db_v1_5.parquet'),
        sp_out=os.path.join(ROOT, 'data', 'zeosyn_splits_v1_5.parquet'),
        rpt=os.path.join(ROOT, 'data', 'l2_v0_5_incr_report.txt'),
        prefix='incr_R4', src_table='lit_pool_R4_W3',
        anchor=V04, anchor_name='v0.4', new_name='v0.5',
        # §5.4 抽检 (2026-10-10): 7/7 溯源 PASS (100%, 连续三批满分), 无剔除;
        # recipe_idx 5/8 Cs-free 行 gel 修正后入库 (名义 Cs2O 剔除, 见
        # lit_extract_R4_goldstandard_check.csv remediation 注记)
        goldstd_exclude=set(),
        # 人工复核裁决 (2026-10-10): B#7 mixture B 溶剂热 cancrinite 升格 1 行
        # (low_agreement 拆解: A#7 为 A#6 重复行弃); A#4/B#4 Cs-free solution A
        # FAU+GIS 混合相弃 (R3 混合相惯例, 增量管线无相对负样本路径)
        with_upgrade=True,
        upgrade_csv=os.path.join(ROOT, 'data',
                                 'lit_extract_R4_upgrade.csv'),
        upgrade_prefix='incr_R4', upgrade_table='lit_pool_R4_W3_upgrade'),
    # PP1: 挂账文献渠道补捞批次 (2026-10-15 负责人指令; 非滚动节律重启——
    # 不含 W1 新检索, 仅处理已入池挂账文献的补获全文: 304/924 挂账正文
    # [doubao 下载+渠道补捞] + 5 篇前批 no_syn_detail SI 升格复筛; R5 空批
    # 未消耗 v1.5 版本号, 本批 v1.5→v1.6 / 锚点 v0.5→v0.6)
    'PP1': dict(
        inc=os.path.join(ROOT, 'data', 'lit_extract_PP1.parquet'),
        pool=os.path.join(ROOT, 'data', 'lit_pool_PP1_filtered.csv'),
        db_in=os.path.join(ROOT, 'data', 'zeosyn_db_v1_5.parquet'),
        sp_in=os.path.join(ROOT, 'data', 'zeosyn_splits_v1_5.parquet'),
        db_out=os.path.join(ROOT, 'data', 'zeosyn_db_v1_6.parquet'),
        sp_out=os.path.join(ROOT, 'data', 'zeosyn_splits_v1_6.parquet'),
        rpt=os.path.join(ROOT, 'data', 'l2_v0_6_incr_report.txt'),
        prefix='incr_PP1', src_table='lit_pool_PP1_W3',
        anchor=V05, anchor_name='v0.5', new_name='v0.6',
        goldstd_exclude={'10.1002/advs.202306533',   # §5.4 抽检 3/20 FAIL:
                         #   叙述性 gel (Sn source, Si source...) 无配比,
                         #   si_source=1.0 幻觉归一值 (引文[37]凝胶 33.3
                         #   SiO2 非本文), 7 条一致行全叙述性 → 整论文剔除
                         '10.1021/jacs.3c00423'},    # §5.4 抽检 1/20 FAIL:
                         #   #1 time_h=240h 无原文锚 (原文 14 days=336h),
                         #   #3 为 post-treatment 非合成行 → 整论文剔除
        goldstd_reason=('goldstd fail: 叙述性 gel 幻觉归一值 / time_h 无原文锚'
                        ' (§5.4 抽检 20 条 16 pass 4 FAIL, 剔除后 100%)'),
        # 人工复核 LLM 辅助裁决 (pp_arbitrate.py): 704 行 review 队列 →
        #   orphan 454 trace 预检 + 244 disagree GLM 按篇仲裁 → 198 keep
        #   → 终检清洗 (goldstd_exclude 4 + n_Si=n_Al=0 零Si拦截 68) = 126 行
        #   全 positive; 升格定义留档 arbitrate_log.csv + upgrade_removed 留 %TEMP%
        with_upgrade=True,
        upgrade_csv=os.path.join(ROOT, 'data',
                                 'lit_extract_PP1_upgrade.csv'),
        upgrade_prefix='incr_PP1', upgrade_table='lit_pool_PP1_W3_upgrade'),
    # PP2: SI-only 挂账补捞批次 (2026-10-15 负责人指令; PP1 后续——42 篇
    # 仅 SI 无正文的挂账 DOI, 以纯 SI 文本为载体过管线; v1.6→v1.7 /
    # 锚点 v0.6→v0.7)
    'PP2': dict(
        inc=os.path.join(ROOT, 'data', 'lit_extract_PP2.parquet'),
        pool=os.path.join(ROOT, 'data', 'lit_pool_PP2_filtered.csv'),
        db_in=os.path.join(ROOT, 'data', 'zeosyn_db_v1_6.parquet'),
        sp_in=os.path.join(ROOT, 'data', 'zeosyn_splits_v1_6.parquet'),
        db_out=os.path.join(ROOT, 'data', 'zeosyn_db_v1_7.parquet'),
        sp_out=os.path.join(ROOT, 'data', 'zeosyn_splits_v1_7.parquet'),
        rpt=os.path.join(ROOT, 'data', 'l2_v0_7_incr_report.txt'),
        prefix='incr_PP2', src_table='lit_pool_PP2_W3',
        anchor=V06, anchor_name='v0.6', new_name='v0.7',
        # 金标准抽检 (13 条一致行全量溯源核验, R4 小批全查惯例):
        #   11/13 数值可定位 (含 388K=115°C / 7 days=168h 换算形态);
        #   114170 为动力学论文 (gel 固定×温度梯度×结晶时间), time_h 是行
        #   核心字段——2 条硬幻觉 (96h/264h 全文零命中) + 11 条中 t=12/24/48
        #   疑似巧合命中 ("stirred for 24 h" 老化语境非结晶语境), 时间字段
        #   溯源系统性失效 → 整论文剔除 (12 行), §5.4 论文级判决惯例
        goldstd_exclude={'10.1016/j.micromeso.2026.114170'},
        goldstd_reason=('goldstd fail: time_h 幻觉 (96h/264h 全文零命中,'
                        ' 动力学论文时间字段溯源系统性失效; 13 条抽检 11 可定位'
                        ' 但含巧合命中嫌疑)'),
        # 人工复核裁决 (pp_arbitrate.py): 50 行 review → 升格 6 → 终检清洗
        #   (goldstd DOI 覆盖 1 + 零 Si 拦截 4) = 1 行 (matt.2022.10.013
        #   framework='X' 占位码, merge 框架归一侧拦截)
        with_upgrade=True,
        upgrade_csv=os.path.join(ROOT, 'data',
                                 'lit_extract_PP2_upgrade.csv'),
        upgrade_prefix='incr_PP2', upgrade_table='lit_pool_PP2_W3_upgrade'),
    # R6 负样本定向批 (M12 裁定#4 重启后首批, 2026-10-16):
    #   W3 一致 recipe 0 条 (anie.202117698 trace_fail 系 al_source
    #   分数形态 1/38=0.0263 逐字定位盲区误伤, 人工核验 Table 1 后
    #   全量恢复; s41467 分子筛负载催化假阳性 / anie.202408275 商品
    #   沸石后处理 / accounts.3c00269 Perspective 二手配方, 三篇
    #   自然出局) → 增量全走升格通道: 8 positive + 2 amorphous
    #   absolute_neg (R6 负样本通道首批实证, AMORPHOUS_FW 分支);
    #   A#5 (MEL 杂相) / A#10 (ZSM-5 杂质) 转 lit_fail v2 不入主库
    'R6': dict(
        inc=os.path.join(ROOT, 'data', 'lit_extract_R6.parquet'),
        pool=os.path.join(ROOT, 'data', 'lit_pool_R6_filtered.csv'),
        db_in=os.path.join(ROOT, 'data', 'zeosyn_db_v1_7.parquet'),
        sp_in=os.path.join(ROOT, 'data', 'zeosyn_splits_v1_7.parquet'),
        db_out=os.path.join(ROOT, 'data', 'zeobench_db_v1_8.parquet'),
        sp_out=os.path.join(ROOT, 'data', 'zeobench_splits_v1_8.parquet'),
        rpt=os.path.join(ROOT, 'data', 'l2_v0_8_incr_report.txt'),
        prefix='incr_R6', src_table='lit_pool_R6_W3',
        anchor=V07, anchor_name='v0.7', new_name='v0.8',
        goldstd_exclude=set(), goldstd_reason='',
        with_upgrade=True,
        upgrade_csv=os.path.join(ROOT, 'data',
                                 'lit_extract_R6_upgrade.csv'),
        upgrade_prefix='incr_R6', upgrade_table='lit_pool_R6_W3_upgrade'),
}

# 材料名 → IZA 码 (R1 增量池实测出现的非码值)
# kalsilite→ABW 已证伪剔除 (2026-10-10 R2 复核: 源论文 chemmater.2c00773
# 定性 kalsilite 为 non-porous potassium aluminosilicate 致密相, 与
# Cs-ABW 对举为不同物相; IZA ABW 基准材料为 Li-A(BW)) → 并入
# DENSE_MINERALS 按致密矿惯例处置 (framework_code=材料名+absolute_neg)
MATER2IZA = {'merlinoite': 'MER',
             'dd3r': 'DDR', 'zeolite a': 'LTA', 'zeolite y': 'FAU',
             'zeolite x': 'FAU', 'zsm-5': 'MFI', 'ssz-13': 'CHA',
             # R4 增 (2026-10-10, cgd.3c00239 MMU-1 论文): MMU-1 为 EDI 拓扑
             # (原文摘要 'a new Cs-Na modification of an EDI-type zeolite
             # structure' + 结论 'MMU-1 of EDI framework topology type' 双重
             # 明示; NZ 为正文暂定名; 主库 EDI 29 行正中 T2 缺口目标);
             # cancrinite 为 IZA CAN 基准矿物名 (主库 CAN 17 行)
             'mmu1': 'EDI', 'cancrinite': 'CAN'}

# IZA 无码的注明沸石 → 材料名直入 framework_code (274 结晶微孔相惯例,
# 主库 JU96/122B 同构; R2: ZMQ-1 为 2024 Nature 新报道超大孔沸石)
# 键为 low 形态 (连字符已在 norm_framework 预清洗剥离: 'ZMQ-1'→'zmq1')
NAMED_ZEO = {'zmq1': 'ZMQ-1'}

# ---- 金标准 remediation 常量 (§5.4, 2026-10-10) ----
# ① 剔除 DOI: chemmater.2c00773 七行 na_oh/solvent_mol 正文不可溯源
#    (SI Table S3 未获取), 双模板同源推断 (R2 SI 获取目标)
GOLDSTD_EXCLUDE = {'10.1021/acs.chemmater.2c00773'}
# ② 致密矿: megakalsilite 为致密 KAlSiO4 (论文原文明示 dense 相), 按主库
#    惯例 framework_code=材料名 + absolute_neg (tridymite/quartz 同型);
#    kalsilite 2026-10-10 证伪后并入 (non-porous KAlSiO4, chemmater.2c00773
#    原文与 Cs-ABW 对举为不同物相)
DENSE_MINERALS = {'meg', 'megakalsilite', 'kalsilite'}
DENSE_FW = {'megakalsilite', 'kalsilite'}   # norm_framework 致密相返回值
# ③ 无定形产物 (R4 增, 2026-10-10): framework_code='amorphous' + absolute_neg
#    + label1_detail='特殊值' (主库 3,768 行惯例; R4 首现于 cgd.3c00239
#    Cs-free 溶剂热负样本, 对 A3 comp 空间混居判别有价值)
AMORPHOUS_FW = {'amorphous'}

# gel_composition_str 组分 → (库存键, 化学计量); 氢氧化物另计 OH
# R3 增 (2026-10-10, 63537-5 Cu-MOR 首现): 骨架 Cu 以 CuO 计入 n_Cu
# (Cu 取代骨架 Al, 主库 52 n_ 列含 n_Cu); 草酸矿化剂 C2H2O4 入 n_acid
# (主库酸列 334 行先例); 未列金属氧化物 (如 ZnO) 仍走 osda 分支,
# 出现于未来批次入库行时再扩展 (5c00431 Zn 行未入库, 暂不需要)
INORG = {'sio2': ('Si', 1.0), 'al2o3': ('Al', 2.0),
         'naoh': ('Na', 1.0), 'koh': ('K', 1.0), 'csoh': ('Cs', 1.0),
         'lioh': ('Li', 1.0), 'na2o': ('Na', 2.0), 'k2o': ('K', 2.0),
         'bao': ('Ba', 1.0), 'hf': ('F', 1.0), 'h2o': ('H2O', 1.0),
         'cuo': ('Cu', 1.0), 'c2h2o4': ('acid', 1.0),
         # R4 增 (2026-10-10, cgd.3c00239 首现): Cs2O 按 M2O 型 ×2 惯例
         # (同 Na2O/K2O); 未映射时被误入 OSDA 有机组分 (幻影 n_osda1)
         'cs2o': ('Cs', 2.0)}
HYDROXIDES = {'naoh', 'koh', 'csoh', 'lioh'}


def norm_framework(fc):
    """framework_code 归一: 致密矿优先 (金标准缺陷④: 'Meg' 曾被误映射
    'MER'/3 字母正则误判 'MEG'); 材料名映射次之; 再 IZA 码直取; 未知 None"""
    if not isinstance(fc, str) or not fc.strip():
        return None
    low = re.sub(r'[^a-z0-9 ]', '', fc.lower()).strip()
    if low in DENSE_MINERALS:
        # 致密相 → framework_code=材料名 + absolute_neg (主库材料名惯例);
        # 'meg' 为缩写归一全名, kalsilite 等直接用材料名
        return 'megakalsilite' if low == 'meg' else low
    if low in AMORPHOUS_FW:
        # 无定形产物 → 'amorphous' + absolute_neg (主库 3,768 行惯例)
        return 'amorphous'
    if low in MATER2IZA:
        return MATER2IZA[low]
    if low in NAMED_ZEO:
        return NAMED_ZEO[low]
    s = re.sub(r'[^A-Za-z0-9*]', '', fc).upper()
    if re.fullmatch(r'\*?[A-Z]{3,4}', s):      # IZA 码 (含 *BEA 装饰码)
        return s
    return None


def doi_key(doi):
    """DOI → route_id 片段: 去注册商前缀, 留字母数字, 截 24 字符"""
    s = re.sub(r'^10\.\d+/', '', str(doi))
    s = re.sub(r'[^A-Za-z0-9]', '', s)
    return s[:24] or 'unknown'


def split_top(s, sep=':'):
    """按 sep 切分, 括号内不切 (P6 r02 组分名内含冒号 'sodium aluminate: ...')"""
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth = max(0, depth - 1)
        if ch == sep and depth == 0:
            parts.append(''.join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append(''.join(cur).strip())
    return [p for p in parts if p]


def parse_gel(gel):
    """gel_composition_str 解析 (金标准 remediation: 一等组成证据)

    支持两种版式:
      A) '5.4 SiO2: 1 Al2O3: 27.2 KOH: 1093 H2O'    数值前置, 冒号分隔
      B) 'ADA: SiO2: EDA: H2O = 3: 100: 50: 4000'    名称前置, '=' 后数值列
    组分名去括号注记后小写匹配 INORG; 未识别的有机组分按出现序入 osda 列表
    (名称含 '-oh'/'hydroxide' 者另计 OH); 'x SiO2' 等非数值组分跳过。
    返回 dict {元素/组分: 摩尔, 'osda': [(名, 摩尔), ...]}; 无可解析组分返 None
    """
    if not isinstance(gel, str) or not gel.strip():
        return None
    comps = []                       # [(amount, 规范名)]
    if '=' in gel:
        lhs, _, rhs = gel.partition('=')
        names = [re.sub(r'\([^)]*\)', '', p).strip().lower()
                 for p in split_top(lhs)]
        amts = rhs.replace(':', ' ').split()
        if not amts or len(names) != len(amts):
            return None
        for nm, am in zip(names, amts):
            try:
                comps.append((float(am), nm, 0))
            except ValueError:
                return None
    else:
        for part in split_top(gel):
            m = re.match(r'^(\d+(?:\.\d+)?)\s*(.+)$', part)
            if not m:
                continue             # 'x SiO2' / 叙述句等非数值组分跳过
            raw = m.group(2)
            # OSDA-(OH)n 型: 括号内为氢氧根化学计量而非注记 (R2 适配:
            # Tri-Cy-dC8(OH)2 双季鏻盐二氢氧化物, n_OH=2×amt; R1 无此版式,
            # oh_n=0 路径行为不变)
            ohm = re.search(r'\((OH)\)(\d*)\s*$', raw, re.I)
            oh_n = (int(ohm.group(2)) if ohm and ohm.group(2)
                    else (1 if ohm else 0))
            base = (re.sub(r'\(OH\)\d*\s*$', '', raw, flags=re.I)
                    if ohm else raw)      # 尾缀数字不残留 (tri-cy-dc82 教训)
            comps.append((float(m.group(1)),
                          re.sub(r'\([^)]*\)', '', base).strip().lower(),
                          oh_n))
    if not comps:
        return None
    out = {'Si': 0.0, 'Al': 0.0, 'Na': 0.0, 'K': 0.0, 'Cs': 0.0, 'Li': 0.0,
           'Ba': 0.0, 'F': 0.0, 'H2O': 0.0, 'OH': 0.0,
           'Cu': 0.0, 'acid': 0.0, 'osda': []}
    for amt, nm, oh_n in comps:
        if nm in INORG:
            k, stoich = INORG[nm]
            out[k] += stoich * amt
            if nm in HYDROXIDES:
                out['OH'] += amt
        else:                        # 有机组分 (OSDA / 胺 / 醇)
            out['osda'].append((nm, amt))
            if oh_n > 0:
                out['OH'] += amt * oh_n
            elif nm.endswith('-oh') or 'hydroxide' in nm:
                out['OH'] += amt     # OSDA-OH 型: OH 随模板量计入
    return out


def load_desc_maps(db_path):
    """OSDA 描述符匹配 (l2_v0_1_litfail match_desc 复刻);
    返回 (chg_map: smiles→formal_charge, match_desc: 名称→smiles)"""
    osda_db = pd.read_parquet(os.path.join(ROOT, 'data',
                                           'osda_descriptors.parquet'),
                              columns=['osda_smiles', 'formal_charge'])
    chg_map = dict(zip(osda_db['osda_smiles'].astype(str),
                       osda_db['formal_charge']))
    db_cols = ['osda%d_%s' % (i, c) for i in (1, 2, 3)
               for c in ('name', 'smiles')]
    dbo = pd.read_parquet(db_path, columns=db_cols)
    name2smi = {}
    for i in (1, 2, 3):
        sub = dbo[['osda%d_name' % i, 'osda%d_smiles' % i]].dropna()
        for n, sm in zip(sub.iloc[:, 0], sub.iloc[:, 1]):
            name2smi[str(n).lower().strip()] = str(sm)
    smi_set = set(osda_db['osda_smiles'].astype(str))
    desc_keys = {nm: sm for nm, sm in name2smi.items() if sm in smi_set}

    def match_desc(name):
        if not name:
            return None
        sm = desc_keys.get(name)
        if sm:
            return sm
        subs = [k for k in desc_keys if name in k or k in name]
        if len(set(desc_keys[k] for k in subs)) == 1:
            return desc_keys[subs[0]]
        return None
    return chg_map, match_desc


def build_inc_rows(cfg):
    """增量 recipe → 主库 149 列行 + 评估辅助列 (批次参数化)"""
    inc = pd.read_parquet(cfg['inc'])
    pool = pd.read_csv(cfg['pool'])
    ym = dict(zip(pool['doi'], pool['publication_year']))
    tm = dict(zip(pool['doi'], pool['title']))
    db = pd.read_parquet(cfg['db_in'], columns=['route_id', 'doi',
                                                'framework_code', 'label1'])
    fw_known = set(db['framework_code'].dropna().astype(str))
    chg_map, match_desc = load_desc_maps(cfg['db_in'])
    has_trace = 'trace_ok' in inc.columns

    rows, drop_unmapped = [], []
    for r in inc.itertuples():
        # ---- 金标准剔除 (§5.4 抽检不通过行, 批次配置) ----
        if r.doi in cfg['goldstd_exclude']:
            drop_unmapped.append((r.doi, cfg['goldstd_reason']))
            continue
        fw = norm_framework(r.framework_code)
        if fw is None:
            drop_unmapped.append((r.doi, r.framework_code))
            continue
        dense = fw in DENSE_FW
        amorph = fw in AMORPHOUS_FW        # 无定形 (R4: Cs-free 边界负样本)
        named = fw in NAMED_ZEO.values()   # IZA 无码注明沸石 (材料名惯例)
        fl = []
        # ---- 组成映射 (gel_composition_str 一等证据, 结构化列兜底) ----
        gel = parse_gel(r.gel_composition_str)
        g = gel if gel else {}
        if gel is None:
            fl.append('gel_unparsed')
        n_Si = g.get('Si', 0.0) or (
            float(r.si_source) if pd.notna(r.si_source) else 0.0)
        if n_Si <= 0 and pd.isna(r.si_source):
            fl.append('si_missing')
        n_Al = g.get('Al', 0.0) or (
            2.0 * float(r.al_source) if pd.notna(r.al_source) else 0.0)
        if n_Al <= 0 and pd.isna(r.al_source) and n_Si > 0:
            fl.append('al_missing_or_puresilica')
        # 碱金属/碱土/氟: 仅 gel 可源 (na_oh 字段不分物种, 金标准缺陷①②③⑥⑧)
        n_Na = g.get('Na', 0.0)
        n_K = g.get('K', 0.0)
        n_Li = g.get('Li', 0.0)
        n_Cs = g.get('Cs', 0.0)
        n_Ba = g.get('Ba', 0.0)
        n_F = g.get('F', 0.0)
        n_OH = g.get('OH', 0.0)
        if n_F <= 0 and r.fluoride_used:
            fl.append('f_unquantified')
        if (n_Na <= 0 and n_K <= 0 and n_Li <= 0 and n_Cs <= 0
                and pd.notna(r.na_oh)):
            # gel 无碱金属可源 → 结构化 na_oh 兜底 (Na+OH 等量, 旧行为)
            n_Na = float(r.na_oh)
            n_OH += float(r.na_oh)
            fl.append('alkali_from_na_oh')
        n_H2O = g.get('H2O', 0.0) or (
            float(r.solvent_mol) if pd.notna(r.solvent_mol) else 0.0)
        if n_H2O <= 0 and pd.isna(r.solvent_mol):
            fl.append('h2o_missing')
        # OSDA: gel 首个有机组分对齐结构化 osda_1; 次有机组分 → osda2
        osda_name = str(r.osda_1).strip() if pd.notna(r.osda_1) else None
        osda_mol = float(r.osda_1_mol) if pd.notna(r.osda_1_mol) else 0.0
        g_osda = g.get('osda', [])
        if g_osda and osda_mol <= 0:
            osda_mol = g_osda[0][1]       # gel 补量 (结构化缺失时)
        if osda_name and osda_mol <= 0:
            fl.append('osda_unquantified')
        if (osda_name and osda_mol > 0 and not g_osda
                and re.search(r'\bOH\b|hydroxide', osda_name, re.I)):
            n_OH += osda_mol              # 兜底: gel 未解析时 OSDA-OH 计 OH
        osda2_name, osda2_mol = None, 0.0
        if len(g_osda) > 1:
            osda2_name, osda2_mol = g_osda[1]
        n_seed = 0.0
        if r.seed_used:
            fl.append('seed_present_unquantified')
        # ---- 金标准缺陷⑨: 零 Si 行拦截 (无硅源的沸石合成配方不入库) ----
        if n_Si <= 0:
            fl.append('si_zero')
            drop_unmapped.append((r.doi, 'si_zero (positive 行无硅源)'))
            continue
        # ---- 归一化 (全 n_ 列除 n_seed, 主库 sum=1 口径, 含 n_Ba) ----
        # n_Cu/n_acid: R3 组分映射扩展 (INORG cuo/c2h2o4, 见 INORG 注释)
        moles = {'n_Si': n_Si, 'n_Al': n_Al, 'n_Na': n_Na, 'n_K': n_K,
                 'n_Li': n_Li, 'n_Cs': n_Cs, 'n_Ca': 0.0, 'n_F': n_F,
                 'n_P': 0.0, 'n_B': 0.0, 'n_Ge': 0.0, 'n_Ti': 0.0,
                 'n_Ga': 0.0, 'n_Fe': 0.0, 'n_Zn': 0.0, 'n_Co': 0.0,
                 'n_Mg': 0.0, 'n_V': 0.0, 'n_Ba': n_Ba, 'n_H2O': n_H2O,
                 'n_Cu': g.get('Cu', 0.0), 'n_acid': g.get('acid', 0.0),
                 'n_OH': n_OH, 'n_osda1': osda_mol, 'n_osda2': osda2_mol,
                 'n_osda3': 0.0, 'n_seed': 0.0}
        tot = sum(v for k, v in moles.items() if k != 'n_seed')
        if tot <= 0:
            fl.append('all_moles_missing')
            drop_unmapped.append((r.doi, 'all moles null'))
            continue
        norm = {k: (v / tot if k != 'n_seed' else 0.0)
                for k, v in moles.items()}
        # ---- ratio (归一前摩尔, scale 不变) ----
        rSiAl = n_Si / n_Al if n_Al > 0 else np.nan
        key = match_desc(osda_name.lower().strip() if osda_name else None)
        q = chg_map.get(str(key), np.nan) if key else np.nan
        pos = (norm['n_Na'] + norm['n_K'] + norm['n_Li'] + norm['n_Cs']
               + (0.0 if pd.isna(q) else q) * norm['n_osda1'])
        neg = norm['n_Al'] + norm['n_OH'] + norm['n_F']
        cb = pos / neg if (neg > 0 and n_Al > 0) else np.nan
        T_C = float(r.temperature_c) if pd.notna(r.temperature_c) else np.nan
        t_h = float(r.time_h) if pd.notna(r.time_h) else np.nan
        year = int(ym.get(r.doi, 0))
        rid = '%s_%s_r%02d' % (cfg['prefix'], doi_key(r.doi),
                               int(r.recipe_idx) + 1)
        trace_note = ('' if not has_trace
                      else '; trace_ok=%s' % (bool(getattr(r, 'trace_ok',
                                                           True))))
        notes = ('%s-W3 双模板一致%s (agree=%.2f, %d 字段); gel: %s; '
                 'vessel: %s; screen 池 full_pass/needs_review; 旗标: %s'
                 % (cfg['prefix'].replace('incr_', ''), trace_note,
                    r.agree_rate, r.n_fields_compared,
                    r.gel_composition_str, r.vessel_type,
                    ';'.join(fl) or '无'))
        row = {'route_id': rid, 'src_row_id': np.nan, 'doi': r.doi,
               'year': float(year), 'normed': 'yes', **norm,
               'ratio_SiAl': rSiAl,
               'ratio_OH_Si': n_OH / n_Si if n_Si > 0 else np.nan,
               'ratio_Na_Si': n_Na / n_Si if n_Si > 0 else np.nan,
               'ratio_H2O_Si': n_H2O / n_Si if n_Si > 0 else np.nan,
               'ratio_osda_Si': (osda_mol / n_Si
                                 if (osda_mol > 0 and n_Si > 0) else np.nan),
               'ratio_charge_bal': cb, 'T_C': T_C, 't_h': t_h,
               'aging_h': np.nan, 'aging_T_C': np.nan, 'stirring': 'no',
               'react_vol': np.nan, 'pH': np.nan, 'solvent': 0.0,
               'seed_type': np.nan,
               'osda1_name': osda_name, 'osda1_smiles': np.nan,
               'osda1_iupac': np.nan, 'osda1_formula': np.nan,
               'osda1_synonyms': np.nan,
               'osda2_name': (osda2_name if osda2_name else np.nan),
               'osda2_smiles': np.nan,
               'osda2_iupac': np.nan, 'osda2_formula': np.nan,
               'osda2_synonyms': np.nan,
               'osda3_name': np.nan, 'osda3_smiles': np.nan,
               'osda3_iupac': np.nan, 'osda3_formula': np.nan,
               'osda3_synonyms': np.nan,
               'product1': fw, 'product2': np.nan, 'product3': np.nan,
               'framework_code': fw, 'framework_code2': np.nan,
               'framework_code3': np.nan,
               'label1': ('absolute_neg' if (dense or amorph) else 'positive'),
               'label1_detail': ('dense mineral (%s)' % fw if dense
                                 else ('特殊值' if amorph
                                       else ('材料名 (IZA 无码注明沸石)' if named
                                             else 'IZA 框架码'))),
               'label2': np.nan, 'label3': np.nan,
               'zeo_desc_key': np.nan,
               'osda1_desc_key': key,
               'osda2_desc_key': np.nan, 'osda3_desc_key': np.nan,
               'if_score': np.nan, 'if_out': False,
               'SiAl_reported': np.nan, 'yield': np.nan,
               'percent_cryst': np.nan, 'crystal_size': np.nan,
               'micropore_volume': np.nan, 'micropore_diameter': np.nan,
               'bet_area': np.nan, 'external_surface_area': np.nan,
               'precursors': np.nan, 'brands': np.nan,
               'title': tm.get(r.doi, np.nan),
               'abstract_keywords': np.nan, 'recipe_keywords': np.nan,
               'notes': notes, 'entry': np.nan, 'sample_no': np.nan,
               'type_role': np.nan,
               'si_source': (str(r.si_source_name)
                             if pd.notna(r.si_source_name) else np.nan),
               'al_source': (str(r.al_source_name)
                             if pd.notna(r.al_source_name) else np.nan),
               'oven_type': np.nan, 'h2o_initial': np.nan,
               'h2o_final': np.nan, 'phase_raw': np.nan,
               'fau_emt_fraction': np.nan, 'particle_size_nm': np.nan,
               'ar_uptake_001': np.nan, 'agg_log10_pc': np.nan,
               'campaign_target': np.nan, 'is_target_family': np.nan,
               'step_note': np.nan, 'output_backfilled': np.nan,
               'source_table': cfg['src_table'], 'paper_label': np.nan,
               'in_paper_clean': np.nan, 'si_size_nm': np.nan,
               'pre_dis_si': np.nan, 'pre_dis_al': np.nan,
               'clear_sol': np.nan, 'morphology': np.nan,
               'variant': np.nan, 'mass_product_g': np.nan,
               # ---- 评估辅助列 (不入主库, 落库前剔除) ----
               'l2_pure': 0 if (dense or amorph) else 1,
               'l2_xtal': 0 if (dense or amorph) else 1,
               'inc_flags': ';'.join(sorted(set(fl))),
               'agree_rate': float(r.agree_rate)}
        rows.append(row)
    df = pd.DataFrame(rows)
    if not rows:
        # R6 起负样本定向批可能 0 行一致 recipe (增量全走升格通道):
        # 无列空帧会令 apply_qc_flags (ratio_charge_bal 直取 / df.eval
        # where 引用 n_Si,n_P) 与 main 报告段 (doi/framework_code)
        # KeyError → 返回最小列集空帧 (QC 字段+where 列+报告列)
        df = pd.DataFrame(columns=[
            'doi', 'framework_code', 'inc_flags', 'agree_rate',
            'T_C', 't_h', 'ratio_SiAl', 'ratio_charge_bal',
            'n_Si', 'n_P', 'n_Al'])
    return apply_qc_flags(df), drop_unmapped, fw_known


def apply_qc_flags(df):
    """QC 标记 (qc_config 规则, qc_checks 同口径); inc 与升格行共用"""
    out_cols = []
    for field, rule in QC_CFG['checks'].items():
        lo, hi = rule.get('min'), rule.get('max')
        s = df[field] if field in df.columns else pd.Series(np.nan,
                                                            index=df.index)
        ins = df.eval(rule['where']) if rule.get('where') else \
            pd.Series(True, index=df.index)
        out = s.notna() & ins & ~s.between(lo, hi if hi is not None
                                           else np.inf)
        df['qc_%s_out' % field] = out
        out_cols.append('qc_%s_out' % field)
    # charge QC (电荷平衡公式已在构建侧按同口径计算)
    cb_rule = QC_CFG.get('charge_balance')
    cbv = df['ratio_charge_bal']
    df['qc_charge_out'] = cbv.notna() & ~cbv.between(cb_rule['ratio_min'],
                                                     cb_rule['ratio_max'])
    out_cols.append('qc_charge_out')
    df['qc_any_out'] = df[out_cols].any(axis=1)
    return df


def build_upgrade_rows(cfg):
    """⑤ b_unmatched 缩围升格行 (R2: lit_extract_R1_upgrade.csv 2 行;
    R3: lit_extract_R3_upgrade.csv 4 行 = 63537-5 MOR-Cu 温度系列)
    → 主库 149 列行; 摩尔量已由升格定义留档 (notes 证据链), 此处仅
    归一化+ratio+QC。n_ 列动态读取 (CSV 缺列按 0; n_Sn 历史性不读
    = v1.3 定稿口径, R1 CSV 的 n_Sn 值不回溯)"""
    up = pd.read_csv(cfg['upgrade_csv'])
    chg_map, match_desc = load_desc_maps(cfg['db_in'])
    N_UP = ('n_Si', 'n_Al', 'n_Na', 'n_K', 'n_Li', 'n_Cs', 'n_Ca',
            'n_F', 'n_P', 'n_B', 'n_Ge', 'n_Ti', 'n_Ga', 'n_Fe',
            'n_Cu', 'n_Zn', 'n_Co', 'n_Mg', 'n_V', 'n_Ba', 'n_H2O',
            'n_OH', 'n_osda1', 'n_osda2', 'n_osda3', 'n_acid')
    rows = []
    for r in up.itertuples():
        moles = {c: float(getattr(r, c, 0.0) or 0.0) for c in N_UP}
        moles['n_seed'] = 0.0
        tot = sum(v for k, v in moles.items() if k != 'n_seed')
        assert tot > 0, '升格行全摩尔缺失: %s' % r.doi
        norm = {k: (v / tot if k != 'n_seed' else 0.0)
                for k, v in moles.items()}
        n_Si, n_Al = moles['n_Si'], moles['n_Al']
        # NaN 防护 (R3 升格行无 OSDA: osda1_name 空值 → 库 NaN 非 'nan' 串)
        _osda = getattr(r, 'osda1_name', None)
        osda_name = ('' if _osda is None or pd.isna(_osda)
                     else str(_osda).strip())
        key = (match_desc(osda_name.lower().strip()) if osda_name
               else None)
        q = chg_map.get(str(key), np.nan) if key else np.nan
        pos = (norm['n_Na'] + norm['n_K'] + norm['n_Li'] + norm['n_Cs']
               + (0.0 if pd.isna(q) else q) * norm['n_osda1'])
        neg = norm['n_Al'] + norm['n_OH'] + norm['n_F']
        cb = pos / neg if (neg > 0 and n_Al > 0) else np.nan
        rid = '%s_%s_up%02d' % (cfg['upgrade_prefix'], doi_key(r.doi),
                                int(r.recipe_idx))
        row = {'route_id': rid, 'src_row_id': np.nan, 'doi': r.doi,
               'year': float(r.year), 'normed': 'yes', **norm,
               'ratio_SiAl': (n_Si / n_Al if n_Al > 0 else np.nan),
               'ratio_OH_Si': (moles['n_OH'] / n_Si if n_Si > 0 else np.nan),
               'ratio_Na_Si': (moles['n_Na'] / n_Si if n_Si > 0
                               else np.nan),
               'ratio_H2O_Si': (moles['n_H2O'] / n_Si if n_Si > 0
                                else np.nan),
               'ratio_osda_Si': (moles['n_osda1'] / n_Si
                                 if (moles['n_osda1'] > 0 and n_Si > 0)
                                 else np.nan),
               'ratio_charge_bal': cb,
               'T_C': (float(r.T_C) if pd.notna(r.T_C) else np.nan),
               't_h': (float(r.t_h) if pd.notna(r.t_h) else np.nan),
               'aging_h': np.nan, 'aging_T_C': np.nan, 'stirring': 'no',
               'react_vol': np.nan, 'pH': np.nan, 'solvent': 0.0,
               'seed_type': np.nan,
               'osda1_name': (osda_name or np.nan), 'osda1_smiles': np.nan,
               'osda1_iupac': np.nan, 'osda1_formula': np.nan,
               'osda1_synonyms': np.nan,
               'osda2_name': np.nan, 'osda2_smiles': np.nan,
               'osda2_iupac': np.nan, 'osda2_formula': np.nan,
               'osda2_synonyms': np.nan,
               'osda3_name': np.nan, 'osda3_smiles': np.nan,
               'osda3_iupac': np.nan, 'osda3_formula': np.nan,
               'osda3_synonyms': np.nan,
               'product1': r.framework_code, 'product2': np.nan,
               'product3': np.nan, 'framework_code': r.framework_code,
               'framework_code2': np.nan, 'framework_code3': np.nan,
               'label1': r.label1, 'label1_detail': r.label1_detail,
               'label2': np.nan, 'label3': np.nan, 'zeo_desc_key': np.nan,
               'osda1_desc_key': key, 'osda2_desc_key': np.nan,
               'osda3_desc_key': np.nan,
               'if_score': np.nan, 'if_out': False,
               'SiAl_reported': np.nan, 'yield': np.nan,
               'percent_cryst': np.nan, 'crystal_size': np.nan,
               'micropore_volume': np.nan, 'micropore_diameter': np.nan,
               'bet_area': np.nan, 'external_surface_area': np.nan,
               'precursors': np.nan, 'brands': np.nan,
               'title': np.nan, 'abstract_keywords': np.nan,
               'recipe_keywords': np.nan,
               'notes': r.notes, 'entry': np.nan, 'sample_no': np.nan,
               'type_role': np.nan,
               'si_source': (str(r.si_source) if pd.notna(r.si_source)
                             and str(r.si_source).strip() else np.nan),
               'al_source': (str(r.al_source) if pd.notna(r.al_source)
                             and str(r.al_source).strip() else np.nan),
               'oven_type': np.nan, 'h2o_initial': np.nan,
               'h2o_final': np.nan, 'phase_raw': np.nan,
               'fau_emt_fraction': np.nan, 'particle_size_nm': np.nan,
               'ar_uptake_001': np.nan, 'agg_log10_pc': np.nan,
               'campaign_target': np.nan, 'is_target_family': np.nan,
               'step_note': np.nan, 'output_backfilled': np.nan,
               'source_table': cfg['upgrade_table'],
               'paper_label': np.nan, 'in_paper_clean': np.nan,
               'si_size_nm': np.nan, 'pre_dis_si': np.nan,
               'pre_dis_al': np.nan, 'clear_sol': np.nan,
               'morphology': np.nan, 'variant': np.nan,
               'mass_product_g': np.nan,
               # ---- 评估辅助列 (不入主库) ----
               'l2_pure': 1, 'l2_xtal': 1,
               'inc_flags': str(r.upgrade_flags), 'agree_rate': np.nan}
        rows.append(row)
    if not rows:
        return pd.DataFrame(), []
    df = pd.DataFrame(rows)
    return apply_qc_flags(df), list(up['doi'])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', default='R2', choices=list(BATCH_CFG),
                    help='批次 (R2 默认; R1 复现 v1.2 收口)')
    args = ap.parse_args()
    B, cfg = args.batch, BATCH_CFG[args.batch]
    V = cfg['anchor']

    log = []

    def P(msg=''):
        print(msg)
        log.append(str(msg))

    t0 = time.time()
    # ==================== 一、增量行构建 ====================
    inc_df, drop_unmapped, fw_known = build_inc_rows(cfg)
    up_df, up_dois = (build_upgrade_rows(cfg) if cfg['with_upgrade']
                      else (pd.DataFrame(), []))
    n_inc, n_up = len(inc_df), len(up_df)
    all_new = (pd.concat([inc_df, up_df], ignore_index=True)
               if n_up else inc_df)
    if n_up and not n_inc:
        # R6 0 行 inc 场景: 空帧 (全 object 列) 与 up_df concat 会把
        # 数值列污染为 object → np.log10 ufunc 收到 Python float 报
        # TypeError (R1 prior_eval 同型教训) → 数值列回 cast float64
        for c in all_new.columns:
            if (c.startswith(('n_', 'ratio_')) or c in (
                    'T_C', 't_h', 'aging_h', 'aging_T_C', 'react_vol',
                    'pH', 'solvent', 'year', 'l2_pure', 'l2_xtal',
                    'agree_rate', 'if_score')):
                all_new[c] = pd.to_numeric(all_new[c], errors='coerce')
    db_in = pd.read_parquet(cfg['db_in'])
    sp_in = pd.read_parquet(cfg['sp_in'])
    # DOI 泄漏审计 (增量 + 升格 × 输入主库)
    doi_new = set(all_new['doi'])
    doi_ov = doi_new & set(db_in['doi'].dropna())
    assert not doi_ov, '增量 DOI 与主库重叠: %s' % doi_ov
    assert all_new['route_id'].is_unique and not (
        set(all_new['route_id']) & set(db_in['route_id']))

    P('WP1/T1.1 Q28④ %s-W3 增量入库 + 切分扩展 + 六锚点评估 (%s → %s)'
      % (B, os.path.basename(cfg['db_in']), os.path.basename(cfg['db_out'])))
    P('=' * 70)
    P('【一、增量行构建 (双模板一致+溯源可定位 recipe%s)】'
      % (' + ⑤ 升格行' if n_up else ''))
    P('  输入: %s (%d 行, %d 篇)%s → 入库 %d 行'
      % (os.path.basename(cfg['inc']), len(inc_df) + len(drop_unmapped),
         inc_df['doi'].nunique() + len({d for d, _ in drop_unmapped}),
         (' + %s (%d 行升格)' % (os.path.basename(cfg['upgrade_csv']),
                                 n_up))
         if n_up else '', n_inc + n_up))
    P('  剔除明细: %s' % (drop_unmapped or '无'))
    P('  框架归一: %s' % dict(inc_df['framework_code'].value_counts()))
    P('  材料名映射: merlinoite→MER / DD3R→DDR; Meg→megakalsilite 致密矿;')
    P('  kalsilite→致密矿 absolute_neg (2026-10-10 证伪, 原 kalsilite→ABW 剔除)')
    nsum = [c for c in inc_df.columns
            if c.startswith('n_') and c != 'n_seed']
    if n_inc:
        P('  归一化 sum=1 (全 n_ 列口径, 含 n_Ba): 均值 %.6f±%.1e'
          % (inc_df[nsum].sum(axis=1).mean(), inc_df[nsum].sum(axis=1).std()))
        P('  缺项旗标分布: %s' % dict(inc_df['inc_flags'].str.split(';')
                                      .explode().value_counts()))
        P('  OSDA 描述符命中: %d/%d (%.0f%%)'
          % (inc_df['osda1_desc_key'].notna().sum(), n_inc,
             100 * inc_df['osda1_desc_key'].notna().mean()))
        P('  QC 越界: %s' % {c: int(inc_df[c].sum())
                              for c in inc_df.columns
                              if c.startswith('qc_')})
    if n_up:
        P('  升格行 (⑤ b_unmatched 缩围, 用户决策 2026-10-10): %s'
          % dict(up_df['framework_code'].value_counts()))
        P('  升格旗标: %s' % dict(up_df['inc_flags'].str.split(';')
                                  .explode().value_counts()))
    P('  DOI 泄漏审计: 本批新增 × 主库重叠 = 0 (assert PASS)')

    # ==================== 二、主库落库 ====================
    eval_cols = ['l2_pure', 'l2_xtal', 'inc_flags', 'agree_rate']
    new_db = all_new.drop(columns=eval_cols)
    # 列对齐主库 149 列: 稀有元素 n_* 列未涉及置 0.0 (lit_fail 惯例),
    # 其余缺列 NaN
    for c in db_in.columns:
        if c not in new_db.columns:
            new_db[c] = 0.0 if c.startswith('n_') else np.nan
    new_db = new_db[db_in.columns]
    db_out = pd.concat([db_in, new_db], ignore_index=True)
    assert db_out.shape == (len(db_in) + n_inc + n_up, 149), db_out.shape
    assert not db_out['route_id'].duplicated().any()
    db_out.to_parquet(cfg['db_out'], engine='pyarrow', compression='zstd',
                      index=False)
    P('')
    P('【二、主库落库】%d×149 = %s %d + incr_%s %d%s; 列序对齐 assert PASS;'
      ' route_id 唯一 assert PASS'
      % (len(db_out), os.path.basename(cfg['db_in']), len(db_in), B, n_inc,
         (' + 升格 %d' % n_up) if n_up else ''))

    # ==================== 三、切分扩展 ====================
    all_new = all_new.reset_index(drop=True)
    fw_of_row = all_new['framework_code']
    held_rows = fw_of_row.isin(HELD_FWS)
    year_arr = all_new['year'].values
    new_sp = pd.DataFrame({
        'route_id': all_new['route_id'], 'doi': all_new['doi'],
        'fw_key': all_new['framework_code'].astype(str),
        'split_random': 'train',
        'split_heldout': np.where(held_rows, 'excluded', 'train'),
        'split_time_2020': np.where(year_arr > 2020, 'test', 'train'),
        'split_time_2018': np.where(year_arr > 2018, 'test', 'train')})
    sp_out = pd.concat([sp_in, new_sp[sp_in.columns]], ignore_index=True)
    assert sp_out.shape == (len(sp_in) + n_inc + n_up, 7), sp_out.shape
    assert sp_out['route_id'].is_unique
    pd.testing.assert_frame_equal(
        sp_out.iloc[:len(sp_in)].reset_index(drop=True),
        sp_in.reset_index(drop=True), check_dtype=False)
    sp_out.to_parquet(cfg['sp_out'], engine='pyarrow', compression='zstd',
                      index=False)
    P('【三、切分扩展 %s】%d×7 = %s %d (值级冻结 assert PASS) + 本批 %d'
      % (os.path.basename(cfg['sp_out']), len(sp_out),
         os.path.basename(cfg['sp_in']), len(sp_in), n_inc + n_up))
    P('  本批指派: random 全 train (工程偏离①: §6.3 测试集冻结优先于')
    P('  80/10/10 指派惯例); heldout 留出骨架行 excluded %d 其余 train %d;'
      ' time_2018/2020 全 test (2022+ 天然考题, incr_zeo1 惯例;'
      ' 评估侧 time 全 excluded)'
      % (int(held_rows.sum()), int((~held_rows).sum())))

    # ==================== 四、六锚点评估 ====================
    P('')
    P('【四、六锚点评估 (%s = %s + 本批训练行; 测试集逐位一致)】'
      % (cfg['new_name'], cfg['anchor_name']))
    # 评估基座恒 v1.1 (特征 scaler/统计拟合基准, docstring 评估架构节)
    db11 = pd.read_parquet(DB_V11)
    sp11 = pd.read_parquet(SPLITS_V11)
    man = pd.read_csv(MANIFEST)
    dup_ids = set(man.loc[man['tier'] == 'A', 'route_id'].dropna())
    df_main = db11[~db11['route_id'].isin(dup_ids)].reset_index(drop=True)
    mm = df_main[['route_id', 'label1']].merge(sp11, on='route_id',
                                               how='left')
    assert mm['split_random'].notna().all()
    # lit_fail 行 (v0.1 镜像: 复用 build_lf_rows 保证逐位一致)
    lf_df, _, _ = build_lf_rows()
    held_pat = re.compile(
        r'\b(LTL|SAS|ITG|RTH|SZR|LTA)\b|zeolite A|Zeolite A'
        r'|(?<![\w(])L(?![\w)])', re.I)
    held_hit = (lf_df['intended_framework'].fillna('').str.contains(
        held_pat, regex=True)
        | lf_df['actual_product'].fillna('').str.contains(held_pat,
                                                          regex=True))
    lf_df['split_random'] = 'train'
    lf_df['split_heldout'] = np.where(held_hit, 'excluded', 'train')
    lf_df['split_time_2018'] = np.where(lf_df['year'] >= 2019, 'excluded',
                                        'train')
    # 先批增量行 (锚点域成员): db_in 中 incr_ 行且不在 v1.1 (R2 时 = incr_R1)
    EVAL_COLS = LOG_MOLE_COLS + [
        'ratio_SiAl', 'ratio_OH_Si', 'ratio_Na_Si', 'ratio_H2O_Si',
        'ratio_osda_Si', 'ratio_charge_bal', 'T_C', 't_h', 'solvent',
        'stirring', 'osda1_desc_key', 'l2_pure', 'l2_xtal', 'doi',
        'route_id']
    in_v11 = set(db11['route_id'])
    prior = db_in[db_in['route_id'].str.startswith('incr_')
                  & ~db_in['route_id'].isin(in_v11)].reset_index(drop=True)
    if len(prior):
        prior['l2_pure'] = (prior['label1'] == 'positive').astype(float)
        prior['l2_xtal'] = prior['label1'].isin(
            ['positive', 'relative_neg']).astype(float)
        prior_eval = prior[EVAL_COLS].copy()
        prior_sp = sp_in.set_index('route_id')
        prior_eval['split_random'] = 'train'
        prior_eval['split_heldout'] = prior_sp.loc[
            prior_eval['route_id'], 'split_heldout'].values
        prior_eval['split_time_2018'] = 'excluded'  # 评估侧测试集冻结
    else:
        # R1 模式: db_in=v1.1 无先批增量行 → 空 prior_eval (零行列对齐帧;
        # l2_pure/l2_xtal 为评估期派生列, 149 列 schema 不含, 不可先切后赋;
        # split 三列为下游 .values 无条件访问点, 须一并入列集)
        prior_eval = pd.DataFrame(
            columns=EVAL_COLS + ['split_random', 'split_heldout',
                                 'split_time_2018'])
    # 本批评估行 (time 侧全 excluded = 测试集冻结)
    inc_eval = all_new[EVAL_COLS].copy()
    inc_eval['split_random'] = 'train'
    inc_eval['split_heldout'] = np.where(held_rows, 'excluded', 'train')
    inc_eval['split_time_2018'] = 'excluded'
    # concat 跳过空 prior_eval (全 object dtype 空帧会污染数值列 dtype,
    # np.log10 收到 Python float 报 ufunc TypeError, 2026-10-10 R1 回归实证)
    lf_frames = ([lf_df, prior_eval, inc_eval] if len(prior)
                 else [lf_df, inc_eval])
    df_lf_all = pd.concat(lf_frames, ignore_index=True, sort=False)
    X, cols = build_features_matrix(df_main, df_lf_all)
    n_m, n_lf, n_prior = len(df_main), len(lf_df), len(prior)
    X_new = X[n_m + n_lf + n_prior:]
    P('  特征: A3 682 维, 主库(v1.1 剔 Tier A) %d + lit_fail %d + 先批增量'
      ' %d + 本批 %d; 主库侧 build_a3 逐位一致 assert PASS'
      % (n_m, n_lf, n_prior, n_inc + n_up))
    # 标签/切分向量 (行序: main / lf / prior / 本批)
    y_pure_m = np.where(df_main['label1'] == 'positive', 1.0, 0.0)
    y_xtal_m = np.where(df_main['label1'].isin(['positive', 'relative_neg']),
                        1.0, 0.0)
    y_pure = np.concatenate([
        y_pure_m, lf_df['l2_pure'].values.astype(float),
        prior_eval['l2_pure'].values.astype(float),
        all_new['l2_pure'].values.astype(float)])
    y_xtal = np.concatenate([
        y_xtal_m, lf_df['l2_xtal'].values.astype(float),
        prior_eval['l2_xtal'].values.astype(float),
        all_new['l2_xtal'].values.astype(float)])
    sr = np.concatenate([mm['split_random'].values,
                         lf_df['split_random'].values,
                         prior_eval['split_random'].values,
                         inc_eval['split_random'].values])
    sh = np.concatenate([mm['split_heldout'].values,
                         lf_df['split_heldout'].values,
                         prior_eval['split_heldout'].values,
                         inc_eval['split_heldout'].values])
    st = np.concatenate([mm['split_time_2018'].values,
                         lf_df['split_time_2018'].values,
                         prior_eval['split_time_2018'].values,
                         inc_eval['split_time_2018'].values])
    absn_m = np.zeros(len(X), bool)          # 测试侧 absolute_neg 行 (主库)
    absn_m[:n_m] = (df_main['label1'] == 'absolute_neg').values
    # 锚点域位掩码 (上批口径复核 = 不含本批新增行)
    is_anchor = np.zeros(len(X), bool)
    is_anchor[:n_m + n_lf + n_prior] = True

    def run(tr, te, y, seed, rounds=None, va=None):
        params = dict(objective='binary', learning_rate=0.05,
                      num_leaves=31, min_child_samples=100, lambda_l2=1.0,
                      max_delta_step=1.0, seed=seed, verbosity=-1)
        dtr = lgb.Dataset(X[tr], label=y[tr])
        if va is not None:
            dva = lgb.Dataset(X[va], label=y[va], reference=dtr)
            b = lgb.train(params, dtr, num_boost_round=400,
                          valid_sets=[dva],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            rounds = b.best_iteration or -1
        else:
            b = lgb.train(params, dtr, num_boost_round=rounds)
        return b, rounds, b.predict(X[te], num_iteration=rounds)

    # ---- 锚点域复核 (seed=42, 六读数 assert; R1→v0.1 含 AUCPR, R2→v0.2
    #      仅 AUROC — v0.2 报告未录 AUCPR) ----
    P('  ---- %s 锚点复核 (seed=42, 锚点域=主库+lit_fail+先批增量, 不含本批)'
      ' ----' % cfg['anchor_name'])
    va_m = is_anchor
    anchor = {}
    for tag, y in (('pure', y_pure), ('xtal', y_xtal)):
        tr0 = (sr == 'train') & va_m
        va0 = (sr == 'val') & va_m
        te0 = (sr == 'test') & va_m
        _, best, p = run(tr0, te0, y, 42, va=va0)
        a = roc_auc_score(y[te0], p)
        ap = average_precision_score(y[te0], p)
        anchor[tag] = best
        apk = '%s_random_ap' % tag
        P('    %s: random 早停@%d AUROC=%.4f AUCPR=%.4f (%s: %.4f%s) -> %s'
          % (tag, best, a, ap, cfg['anchor_name'],
             V['%s_random' % tag],
             '/%.4f' % V[apk] if apk in V else '',
             '一致' if abs(a - V['%s_random' % tag]) < 1e-3 else '漂移!'))
        assert abs(a - V['%s_random' % tag]) < 1e-3, \
            '%s 锚点漂移' % cfg['anchor_name']
        if apk in V:
            assert abs(ap - V[apk]) < 1e-3, \
                '%s 锚点 AUCPR 漂移' % cfg['anchor_name']
    for tag, y in (('pure', y_pure), ('xtal', y_xtal)):
        for sname, sv in (('heldout', sh), ('time_2018', st)):
            tr2 = (sv == 'train') & va_m
            te2 = (sv == 'test') & va_m
            _, _, p2 = run(tr2, te2, y, 42, rounds=anchor['pure'])
            a2 = roc_auc_score(y[te2], p2)
            vkey = '%s_%s' % (tag, sname.split('_')[0] if sname == 'heldout'
                              else 'time')
            P('    %s %s: AUROC=%.4f (%s: %.4f) -> %s'
              % (tag, sname, a2, cfg['anchor_name'], V[vkey],
                 '一致' if abs(a2 - V[vkey]) < 1e-3 else '漂移!'))
            assert abs(a2 - V[vkey]) < 1e-3, \
                '%s 锚点漂移' % cfg['anchor_name']
    P('    %s 锚点六读数 assert 全部 PASS (|Δ|<1e-3)' % cfg['anchor_name'])

    # ---- 锚点域 / 本批全量 3-seed 对照 ----
    res = {}
    for tag, y in (('pure', y_pure), ('xtal', y_xtal)):
        for s in SEEDS:
            for cfgk in (cfg['anchor_name'], cfg['new_name']):
                msk = (is_anchor if cfgk == cfg['anchor_name']
                       else np.ones(len(X), bool))
                tr = (sr == 'train') & msk
                va, te = (sr == 'val') & msk, (sr == 'test') & msk
                _, best, p = run(tr, te, y, s, va=va)
                a = roc_auc_score(y[te], p)
                ap = average_precision_score(y[te], p)
                res[tag, s, cfgk, 'random'] = (a, ap, best)
                for sname, sv in (('heldout', sh), ('time_2018', st)):
                    tr2 = (sv == 'train') & msk
                    te2 = (sv == 'test') & msk
                    _, _, p2 = run(tr2, te2, y, s,
                                   rounds=res['pure', s, cfgk, 'random'][2])
                    res[tag, s, cfgk, sname] = \
                        (roc_auc_score(y[te2], p2),)
                # abs_neg FP (xtal random 测试侧 absolute_neg 行 p≥0.5)
                if tag == 'xtal':
                    sub = (absn_m & te)[te]   # te 内子集掩码, 与 p 对齐
                    res['fp', s, cfgk, 'random'] = (
                        float((p[sub] >= 0.5).mean())
                        if sub.any() else np.nan,)

    def agg(tag, cfgk, key, idx=0):
        vals = [res[tag, s, cfgk, key][idx] for s in SEEDS]
        return np.mean(vals), np.std(vals)

    P('  ---- 3-seed 对照 (42/43/44; 测试集与 %s 逐位一致) ----'
      % cfg['anchor_name'])
    for tag, label in (('xtal', 'xtal (主口径)'), ('pure', 'pure (参照)')):
        P('    口径 %s:' % label)
        for key, vn in (('random', 'random'), ('heldout', 'heldout'),
                        ('time_2018', 'time')):
            m1, s1 = agg(tag, cfg['anchor_name'], key)
            m2, s2 = agg(tag, cfg['new_name'], key)
            # 早停轮数留档 (R1 教训: 早停分岔致 AUROC 波动, 审计必需;
            # 3-seed 一致性 + 域间对比可判 Δ 读数是否轮数噪声主导)
            r1 = [int(res[tag, s, cfg['anchor_name'], 'random'][2])
                  for s in SEEDS] if key == 'random' else None
            r2 = [int(res[tag, s, cfg['new_name'], 'random'][2])
                  for s in SEEDS] if key == 'random' else None
            P('      %-9s %s %.4f±%.4f → %s %.4f±%.4f (Δ%+.2fpp)%s'
              % (vn, cfg['anchor_name'], m1, s1, cfg['new_name'], m2, s2,
                 100 * (m2 - m1),
                 (' [早停轮 %s→%s]' % (r1, r2))
                 if key == 'random' else ''))
        if tag == 'xtal':
            m1, s1 = agg('fp', cfg['anchor_name'], 'random')
            m2, s2 = agg('fp', cfg['new_name'], 'random')
            P('      abs_neg FP: %s %.1f%%±%.1f → %s %.1f%%±%.1f'
              ' (上批报告基线 %.1f%%; std=0 为 LGBM 确定性所致)'
              % (cfg['anchor_name'], 100 * m1, 100 * s1,
                 cfg['new_name'], 100 * m2, 100 * s2,
                 100 * V.get('xtal_fp', 0.986)))
    # ---- 组成空间覆盖 (锚点⑥: 本批新增→主库 NN 距离 vs 主库内部基线) ----
    rng = np.random.default_rng(42)
    ecols = [i for i, c in enumerate(cols) if c.startswith('lg_n_')]
    Xm = X[:n_m][:, ecols]
    ref = Xm[rng.choice(len(Xm), 2000, replace=False)]
    Xi = X_new[:, ecols]
    samp = ref[rng.choice(len(ref), 200, replace=False)]
    d_inc = np.sqrt(((Xi[:, None, :] - ref[None, :, :]) ** 2).sum(-1)).min(1)
    d_ref = np.array([np.sqrt(((r - ref) ** 2).sum(1))[
        np.argsort(np.sqrt(((r - ref) ** 2).sum(1)))[1]].real
        for r in samp])
    P('    组成空间覆盖 (lg_n_ 18 维欧氏): 本批→主库 NN 距离 中位 %.3f / '
      'p90 %.3f vs 主库内部 NN 基线 中位 %.3f / p90 %.3f'
      % (np.median(d_inc), np.percentile(d_inc, 90),
         np.median(d_ref), np.percentile(d_ref, 90)))
    # ==================== 五、判读与收口 ====================
    P('')
    P('【五、六锚点判读 (计划 §6.3)】')
    m_r1, s_r1 = agg('xtal', cfg['anchor_name'], 'random')
    m_r2, s_r2 = agg('xtal', cfg['new_name'], 'random')
    d_main = 100 * (m_r2 - m_r1)
    P('  ① xtal random AUROC: %.4f±%.4f (%s 3-seed %.4f±%.4f, Δ%+.2fpp;'
      ' 单 seed %s 基线 %.4f) — %s'
      % (m_r2, s_r2, cfg['anchor_name'], m_r1, s_r1, d_main,
         cfg['anchor_name'], V['xtal_random'],
         'Δ≥+1pp 继续' if d_main >= 1 else (
             'Δ<+0.5pp 触发复议' if d_main < 0.5 else '中间区间继续观察')))
    for key, vn in (('heldout', '②'), ('time_2018', '③')):
        m1, _ = agg('xtal', cfg['anchor_name'], key)
        m2, _ = agg('xtal', cfg['new_name'], key)
        vkey = '%s_%s' % ('xtal', key.split('_')[0] if key == 'heldout'
                          else 'time')
        P('  %s xtal %s: %.4f (%s 3-seed %.4f, Δ%+.2fpp; 单 seed 基线 %.4f)'
          % (vn, key, m2, cfg['anchor_name'], m1, 100 * (m2 - m1), V[vkey]))
    m1, _ = agg('fp', cfg['anchor_name'], 'random')
    m2, _ = agg('fp', cfg['new_name'], 'random')
    P('  ④ abs_neg FP: %.1f%% (%s 3-seed %.1f%%, 上批报告基线 %.1f%%) — %s'
      % (100 * m2, cfg['anchor_name'], 100 * m1,
         100 * V.get('xtal_fp', 0.986),
         '改善' if m2 < m1 - 1e-9 else '未改善 (A3 comp 空间混居根因未动)'))
    fw_new = sorted(set(all_new.loc[all_new['label1'] == 'positive',
                                    'framework_code'].astype(str))
                    - fw_known)
    P('  ⑤ 骨架覆盖: 新增 %d %s (对照输入主库已知骨架集; 致密矿负样本'
      ' 不计骨架)' % (len(fw_new), fw_new or ''))
    P('  ⑥ 组成空间: 本批新增 NN 距离见上 (主库近邻域基线对照 — %d 条'
      ' 规模所限)' % (n_inc + n_up))
    P('')
    P('  止损规则对照 (§6.3): 本批增量 %d + 升格 %d 条; xtal random Δ%+.2fpp'
      % (n_inc, n_up, d_main))
    if d_main < 0.5:
        P('  → Δ<0.5pp: 登记边际收益, 连续两批 <0.5pp 触发 G2 复议;')
        P('    本批先深入分析原因 (抽取质量/数据源调整) 再定后续批次')
    else:
        P('  → Δ≥0.5pp: 继续后续批次 (R1 实测 +1.04pp 基线对照)')
    P('')
    P('收口声明:')
    P('  1. 数据版本: 主库 %s = %s (%d) + incr_%s (%d, source_table=%s)'
      '%s; %s 同步 (输入值级冻结)。'
      % (os.path.basename(cfg['db_out']), os.path.basename(cfg['db_in']),
         len(db_in), B, n_inc, cfg['src_table'],
         (' + 升格 %d (source_table=%s)' % (n_up, cfg['upgrade_table']))
         if n_up else '', os.path.basename(cfg['sp_out'])))
    P('  2. 评估口径: %s = %s + 本批新增训练行 %d (全 train); 测试集与'
      '%s 逐位一致 (time 侧本批全 excluded; heldout 侧留出骨架行'
      ' excluded)。'
      % (cfg['new_name'], cfg['anchor_name'], n_inc + n_up,
         cfg['anchor_name']))
    n_pos = int((all_new['label1'] == 'positive').sum())
    n_dns = int((all_new['label1'] == 'absolute_neg').sum())
    P('  3. 标签: positive %d (成功路线, IZA 框架码) + absolute_neg %d'
      % (n_pos, n_dns))
    # 负样本构成说明 (R4 前硬编码致密矿文案, R4 首现 amorphous 行后动态化)
    n_dns_kind = int(all_new.loc[all_new['label1'] == 'absolute_neg',
                                 'framework_code'].isin(DENSE_FW).sum())
    P('     (致密矿 %d 行 framework_code=材料名, amorphous %d 行 '
      "framework_code='amorphous'); 缺项旗标留档 inc_flags。"
      % (n_dns_kind, n_dns - n_dns_kind))
    P('  4. 泄漏审计: DOI 重叠 0; route_id 唯一; 输入切分值级冻结 assert。')
    P('  5. LGBM 3-seed (42/43/44) 均值±std 双配置对照; %s seed=42 锚点'
      % cfg['anchor_name'])
    P('     六读数 assert PASS。')
    P('产出: %s / %s / %s; 耗时 %.0fs'
      % (cfg['db_out'], cfg['sp_out'], cfg['rpt'], time.time() - t0))
    with open(cfg['rpt'], 'w', encoding='utf-8') as f:
        f.write('\n'.join(log) + '\n')


if __name__ == '__main__':
    main()
