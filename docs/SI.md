# P2-1 Supporting Information v0.1 — Track B 分册

> 状态：**DRAFT v0.1（2026-10-24）**，按 `P2-1_TrackB参照表_立项_骨架.md` v0.1 产出。
> 范围：本分册只含 Track B 两表（S-B1 汇总、S-B2 逐参数）。S2 判别器参照表与 F1–F4 图件另册，不在此文件合并。
> 口径：所有 B1/B2 数值由 `p21_trackb_table.py` 生成；脚本 G0/A1/A2/A3 全通过，输入与 87 个已发布 run 文件在运行前后 md5 零漂移。

---

## S-B1. Track B baseline values (DiffSyn)

**Table S-B1.** Baselines published by DiffSyn for the released ZeoSyn system split. Values are **not** comparable with Table 4 (Track A): the sample budget, the aggregation of coverage, and the evaluated system set all differ.

| arm | seeds | n_systems | WSD† | MMD‡ | COV-P‡ | COV-R‡ |
|---|---:|---:|---:|---:|---:|---:|
| random | 1 | 372 | 1.6782 | 2.9344 | 0.0319 | 0.6235 |
| AMD | 3 | 370 | 0.5576 ± 0.0152 | 4.5435 ± 0.0438 | 0.6013 ± 0.0057 | 0.5247 ± 0.0053 |
| NN | 1 | 372 | 0.5313 | 4.4462 | 0.6109 | 0.5344 |
| BNN | 3 | 372 | 0.7132 ± 0.0147 | 5.0949 ± 0.1424 | 0.5174 ± 0.0099 | 0.4467 ± 0.0072 |
| GMM | 3 | 372 | 1.2610 ± 0.0038 | 2.5432 ± 0.0033 | 0.4599 ± 0.0007 | 0.8934 ± 0.0017 |
| WGAN-GP | 3 | 372 | 0.6488 ± 0.0022 | 5.0878 ± 0.0028 | 0.4922 ± 0.0034 | 0.4296 ± 0.0035 |
| Real NVP | 3 | 372 | 0.5875 ± 0.0055 | 1.9801 ± 0.0265 | 0.5765 ± 0.0032 | 0.9306 ± 0.0032 |
| cVAE | 3 | 372 | 0.5729 ± 0.0300 | 2.0453 ± 0.1333 | 0.5795 ± 0.0180 | 0.9028 ± 0.0141 |
| DiffSyn | 1 | 372 | 0.4237 | 1.8307 | 0.6602 | 0.8979 |

† Published value: element-wise identical to the released Fig. 2a source data (`fig2a-wasserstein_distance.json`); 21/21 seed values match to `1e-9`.
‡ Locally computed from the released run artifacts (`wsd/mmd_zeo_osda.json`, `prec/rec_zeo_osda_df.csv`); no published numeric value exists for MMD or COV-P/COV-R.

**Notes.**
1. Three system counts coexist and must not be conflated: **464** test systems in the released split (unfiltered); **372** systems actually scored after the `both_graph_feat_present=True` filter (`zeosyn_gen/data/utils.py:442-466`); **370** for AMD, whose precision/recall files contain 370 rows.
2. Version pinning follows the released plotting notebook (`compare_model_metrics.ipynb`, cell 2): random v0; AMD v0/v0-2/v0-3; NN v0; BNN v0/v1/v2; GMM v0/v0-2/v0-3; WGAN-GP v3/v3-2/v3-3; Real NVP v0/v0-2/v0-3; cVAE v10/v10-2/v10-3; DiffSyn v3. The released artifact labels these two arms `GAN` and `NF`. DiffSyn `v3` and `run1` are element-wise identical; `run1` additionally carries the released weights.
3. Across-seed `mean` and `std` use `ddof = 0`, matching `get_mean_and_std`. Single-seed arms (random, NN, DiffSyn) report no standard deviation.
4. COV-P/COV-R are the mean of the 12 per-parameter means, not the per-system coverage used in Table 4. The two are not interchangeable.
5. No cross-arm ranking, composite score or cross-track difference is implied by this table.
6. The published Fig. 2b "COV-F1" is implemented in the released notebook as the arithmetic mean `(COV-P + COV-R)/2`, not the harmonic mean described in the text; F1 is therefore not tabulated here. It is used only as an internal cross-check that the precision/recall files are the published ones (max |Δ| = 4.4e-16).

---

## S-B2. Per-parameter coverage (12 synthesis parameters)

**Table S-B2.** COV-P and COV-R per synthesis parameter, averaged across the pinned seeds of each arm. Values at four decimals; full precision in `Data/p21_trackb_b1_params.csv`.

| arm | metric | Si/Al | Al/P | Si/Ge | Si/B | Na/T | K/T | OH/T | F/T | H2O/T | sda1/T | cryst_temp | cryst_time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | COV-P | 0.0111 | 0.0516 | 0.0007 | 0.0004 | 0.0590 | 0.0122 | 0.0530 | 0.0641 | 0.0740 | 0.0279 | 0.0250 | 0.0042 |
| random | COV-R | 0.4501 | 0.9099 | 0.0652 | 0.0274 | 0.9315 | 0.6448 | 0.8871 | 0.9812 | 0.9410 | 0.7331 | 0.7467 | 0.1641 |
| AMD | COV-P | 0.5270 | 0.6396 | 0.8108 | 0.8252 | 0.7523 | 0.9342 | 0.6288 | 0.8171 | 0.4063 | 0.3423 | 0.2937 | 0.2378 |
| AMD | COV-R | 0.4342 | 0.5665 | 0.7525 | 0.7595 | 0.6721 | 0.9101 | 0.5409 | 0.7723 | 0.3041 | 0.2458 | 0.2100 | 0.1280 |
| NN | COV-P | 0.5645 | 0.6667 | 0.8145 | 0.8280 | 0.7581 | 0.9220 | 0.6290 | 0.8360 | 0.4005 | 0.3548 | 0.3145 | 0.2419 |
| NN | COV-R | 0.4658 | 0.5925 | 0.7612 | 0.7640 | 0.6826 | 0.8970 | 0.5450 | 0.7902 | 0.3029 | 0.2635 | 0.2273 | 0.1208 |
| BNN | COV-P | 0.2312 | 0.2858 | 0.7572 | 0.7796 | 0.7778 | 0.9382 | 0.6057 | 0.8226 | 0.2733 | 0.2778 | 0.2841 | 0.1756 |
| BNN | COV-R | 0.1701 | 0.2388 | 0.7009 | 0.7127 | 0.6891 | 0.9142 | 0.5347 | 0.7739 | 0.1727 | 0.1861 | 0.1884 | 0.0786 |
| GMM | COV-P | 0.4023 | 0.5185 | 0.5178 | 0.5344 | 0.6705 | 0.8577 | 0.5470 | 0.6966 | 0.3062 | 0.2411 | 0.0942 | 0.1327 |
| GMM | COV-R | 0.8271 | 0.9369 | 0.9373 | 0.9245 | 0.9215 | 0.9384 | 0.9124 | 0.9408 | 0.9137 | 0.8869 | 0.8832 | 0.6983 |
| WGAN-GP | COV-P | 0.0675 | 0.0968 | 0.7554 | 0.7796 | 0.7903 | 0.9409 | 0.6701 | 0.8333 | 0.2388 | 0.2770 | 0.3360 | 0.1209 |
| WGAN-GP | COV-R | 0.0566 | 0.0724 | 0.6975 | 0.7127 | 0.7015 | 0.9168 | 0.5990 | 0.7848 | 0.1362 | 0.2033 | 0.2368 | 0.0380 |
| Real NVP | COV-P | 0.4802 | 0.6248 | 0.7903 | 0.8004 | 0.7021 | 0.9165 | 0.6078 | 0.7789 | 0.3941 | 0.3684 | 0.2608 | 0.1933 |
| Real NVP | COV-R | 0.9131 | 0.9706 | 0.9474 | 0.9346 | 0.9547 | 0.9721 | 0.9439 | 0.9641 | 0.9166 | 0.9141 | 0.9195 | 0.8161 |
| cVAE | COV-P | 0.4957 | 0.6248 | 0.7745 | 0.8057 | 0.6950 | 0.8985 | 0.5796 | 0.7631 | 0.4190 | 0.3615 | 0.3044 | 0.2319 |
| cVAE | COV-R | 0.8625 | 0.9507 | 0.9504 | 0.9277 | 0.9283 | 0.9463 | 0.9225 | 0.9446 | 0.9110 | 0.8851 | 0.8442 | 0.7600 |
| DiffSyn | COV-P | 0.5718 | 0.7268 | 0.8156 | 0.8188 | 0.7196 | 0.9309 | 0.6828 | 0.8153 | 0.5571 | 0.4878 | 0.4813 | 0.3150 |
| DiffSyn | COV-R | 0.8804 | 0.9346 | 0.9312 | 0.9281 | 0.9520 | 0.9762 | 0.9247 | 0.9406 | 0.8600 | 0.8580 | 0.8207 | 0.7685 |

---

## Data files

| 文件 | 内容 |
|---|---|
| `Data/p21_trackb_b1_seeds.csv` | 21 行逐 seed 账本（WSD/MMD/COV-P/COV-R、体系数、F1 校验差） |
| `Data/p21_trackb_b1_table.csv` | 9 行汇总（全精度 + 4 位字符串） |
| `Data/p21_trackb_b1_params.csv` | 9 × 12 逐参数 COV-P/COV-R |
| `Data/p21_trackb_b1_manifest.json` | 输入/输出 md5、断言状态、版本 |
| `Data/p21_trackb_b1_report.txt` | 生成报告（G0/A1/A2/A3 日志） |
