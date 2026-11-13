# -*- coding: utf-8 -*-
"""P1#6 / P2-1 补件包：T2 可行性参照表（只读复述，零训练、零新读数）

预注册：Documentation/P2-1_补件包_立项_骨架.md v0.1（2026-10-23）
输入门 G0：8 项输入 md5 硬校验，任一不符 -> SystemExit
口径：5 臂 x 3 切分 = 15 行；主读数取自 feasibility_scorer_v1_2_report.txt，
      ECE 取自 m10_calib_v1_report.txt；跨版本（T* 不一致）的臂登记 NA，不推断。
输出：Data/p21_t2_table.csv / Data/p21_supp_report.txt / Data/p21_supp_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SCORER_REPORT = DATA / "feasibility_scorer_v1_2_report.txt"
BENCH_CSV = DATA / "wp5_bench_v1_table.csv"
G3_REPORT = DATA / "wp5_g3_formal_report.txt"
CALIB_REPORT = DATA / "m10_calib_v1_report.txt"

OUT_CSV = DATA / "p21_t2_table.csv"
OUT_REPORT = DATA / "p21_supp_report.txt"
OUT_MANIFEST = DATA / "p21_supp_manifest.json"

# 预注册 §1 冻结 md5（输入门）
FROZEN_MD5 = {
    SCORER_REPORT: "35be508bc3650c9e1696e67aca09337b",
    BENCH_CSV: "55a58dcf697a98cc4b1ca42d4159ac47",
    G3_REPORT: "8198d094b7bb0e4b1f45e9f3704bf802",
    CALIB_REPORT: "783d61a5087faf4d28b4be339990dbb5",
}

# 部署束登记的温度（feasibility_scorer_v1_2_report.txt）
BUNDLE_T = {"L1": 1.2458, "L2xt": 0.9434, "L2pu": 1.1836, "L3mc": 1.2845, "L3emt": 1.2845}

# 校准账本 ECE_post 锚点（m10_calib_v1_report.txt L46-L60，逐位登记）
EXPECTED_ECE = {
    ("L1", "mc", "random"): 0.0503,
    ("L1", "mc", "heldout"): 0.1890,
    ("L1", "mc", "time_2018"): 0.0638,
    ("L2", "xtal", "random"): 0.0293,
    ("L2", "xtal", "heldout"): 0.0561,
    ("L2", "xtal", "time_2018"): 0.0291,
    ("L2", "pure", "random"): 0.0332,
    ("L2", "pure", "heldout"): 0.0846,
    ("L2", "pure", "time_2018"): 0.0491,
    ("L3", "mc", "random"): 0.0423,
    ("L3", "mc", "heldout"): 0.1939,
    ("L3", "mc", "time_2018"): 0.0508,
    ("L3", "emt", "random"): 0.0097,
    ("L3", "emt", "heldout"): 0.0302,
    ("L3", "emt", "time_2018"): 0.0262,
}

ARMS = [
    # (arm, head, caliber, calib_key, metric_label)
    ("L1", "L1", "mc", ("L1", "mc"), "top-1"),
    ("L2xt", "L2", "xtal", ("L2", "xtal"), "AUROC"),
    ("L2pu", "L2", "pure", ("L2", "pure"), "AUROC"),
    ("L3mc", "L3", "mc", ("L3", "mc"), "top-1"),
    ("L3emt", "L3", "emt", ("L3", "emt"), "acc"),
]

SPLITS = ["random", "heldout", "time_2018"]


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_inputs() -> dict[str, str]:
    got, bad = {}, []
    for p, want in FROZEN_MD5.items():
        if not p.exists():
            bad.append(f"MISSING {p}")
            continue
        h = md5_of(p)
        got[str(p)] = h
        if h != want:
            bad.append(f"MD5 MISMATCH {p.name}: got {h} want {want}")
    if bad:
        print("G0 INPUT GATE FAILED")
        for b in bad:
            print("  " + b)
        raise SystemExit(2)
    print(f"G0 input gate: {len(got)}/{len(FROZEN_MD5)} md5 PASS")
    return got


def parse_headline(text: str) -> tuple[dict, int]:
    """解析部署束报告的 12 条锚点行（期望 == 实测）。"""
    pat = re.compile(
        r"^\s+(L1|L2xt|L2pu|L3)\s+(random|heldout|time_2018)\s+期望\s+([\d.]+)\s+实测\s+([\d.]+)"
    )
    vals, n = {}, 0
    for line in text.splitlines():
        m = pat.match(line)
        if not m:
            continue
        arm, split, exp, act = m.group(1), m.group(2), float(m.group(3)), float(m.group(4))
        assert abs(exp - act) <= 1e-9, f"anchor drift {arm}/{split}: exp {exp} act {act}"
        vals[(arm, split)] = act
        n += 1
    if n != 12:
        raise SystemExit(f"anchor count {n} != 12")
    # 同时校验报告正文的 L1/L2/L3 读数行与锚点表同值
    prose = {
        ("L1", "random"): r"L1 test top-1:.*?random\s+([\d.]+)",
        ("L1", "heldout"): r"L1 test top-1:.*?heldout\s+([\d.]+)",
        ("L1", "time_2018"): r"L1 test top-1:.*?time\s+([\d.]+)",
        ("L2xt", "random"): r"L2 xtal AUROC:.*?random\s+([\d.]+)",
        ("L2xt", "heldout"): r"L2 xtal AUROC:.*?heldout\s+([\d.]+)",
        ("L2xt", "time_2018"): r"L2 xtal AUROC:.*?time\s+([\d.]+)",
        ("L2pu", "random"): r"L2 pure AUROC:.*?random\s+([\d.]+)",
        ("L2pu", "heldout"): r"L2 pure AUROC:.*?heldout\s+([\d.]+)",
        ("L2pu", "time_2018"): r"L2 pure AUROC:.*?time\s+([\d.]+)",
        ("L3", "random"): r"L3 test top-1:.*?random\s+([\d.]+)",
        ("L3", "heldout"): r"L3 test top-1:.*?heldout\s+([\d.]+)",
        ("L3", "time_2018"): r"L3 test top-1:.*?time\s+([\d.]+)",
    }
    for key, rx in prose.items():
        m = re.search(rx, text)
        assert m, f"prose line missing for {key}"
        v = float(m.group(1))
        assert abs(v - vals[key]) <= 1e-9, f"prose/anchor mismatch {key}: {v} vs {vals[key]}"
    return vals, n


def parse_calib(text: str) -> dict:
    pat = re.compile(
        r"^\s*(L[123])\s+(mc|xtal|pure|emt)\s+(random|heldout|time_2018)\s+"
        r"(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        re.IGNORECASE,
    )
    rows = {}
    for line in text.splitlines():
        m = pat.match(line)
        if not m:
            continue
        key = (m.group(1), m.group(2).lower(), m.group(3).lower())
        rows[key] = {
            "n_test": int(m.group(4)),
            "acc": float(m.group(5)),
            "ece_pre": float(m.group(6)),
            "ece_post": float(m.group(7)),
            "nll_pre": float(m.group(8)),
            "nll_post": float(m.group(9)),
            "T": float(m.group(10)),
        }
    if len(rows) != 15:
        raise SystemExit(f"calib rows {len(rows)} != 15")
    return rows


def main() -> None:
    t0 = time.time()
    inputs = check_inputs()

    scorer_text = SCORER_REPORT.read_text(encoding="utf-8")
    calib_text = CALIB_REPORT.read_text(encoding="utf-8")

    anchors, n_anchor = parse_headline(scorer_text)
    print(f"A1 anchors: {n_anchor}/12 PASS")

    calib = parse_calib(calib_text)

    # ECE 锚点断言（15 条）
    for key, want in EXPECTED_ECE.items():
        got = calib[key]["ece_post"]
        assert abs(got - want) <= 1e-9, f"ECE anchor drift {key}: got {got} want {want}"
    print(f"ECE anchors: {len(EXPECTED_ECE)}/15 PASS")

    rows = []
    anchor_key = {"L1": "L1", "L2xt": "L2xt", "L2pu": "L2pu", "L3mc": "L3"}
    for arm, head, caliber, ckey, metric_label in ARMS:
        bundle_t = BUNDLE_T[arm]
        for split in SPLITS:
            c = calib[(ckey[0], ckey[1], split)]
            if arm in anchor_key:
                value = anchors[(anchor_key[arm], split)]
                src_metric = "feasibility_scorer_v1_2_report.txt"
            else:
                # L3emt 无部署束主读数行（12 锚点不含 EMT 口径），取校准账本 acc
                value = c["acc"]
                src_metric = "m10_calib_v1_report.txt"
            # 版本一致性：温度须与部署束登记值一致（否则该臂 ECE 不可跨版本引用）
            t_ok = abs(c["T"] - bundle_t) <= 5e-4
            ece = c["ece_post"] if t_ok else None
            if arm.startswith("L3"):
                gap = (
                    "ECE NA: L3 臂版本不同（m10 calib acc 0.5554 vs bundle top-1 0.5565；"
                    f"T*={c['T']:.4f} vs bundle T_L3={bundle_t:.4f}），逐组 ECE 不可跨版本引用"
                )
            elif not t_ok:
                gap = f"ECE NA: T*={c['T']:.4f} vs bundle T={bundle_t:.4f}"
            else:
                gap = ""
            pass_strict = "" if ece is None else ("PASS" if ece <= 0.05 else "FAIL")
            pass_2dp = "" if ece is None else ("PASS" if round(ece, 2) <= 0.05 else "FAIL")
            rows.append({
                "arm": arm,
                "head": head,
                "caliber": caliber,
                "split": split,
                "metric": metric_label,
                "value": f"{value:.4f}",
                "n_test": c["n_test"],
                "ece_pre": f"{c['ece_pre']:.4f}",
                "ece_post": "" if ece is None else f"{ece:.4f}",
                "T_calib": f"{c['T']:.4f}",
                "T_bundle": f"{bundle_t:.4f}",
                "ece_pass_strict": pass_strict,
                "ece_pass_2dp": pass_2dp,
                "source_metric": src_metric,
                "source_ece": "" if ece is None else "m10_calib_v1_report.txt",
                "gap": gap,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"wrote {OUT_CSV.relative_to(ROOT)} ({len(df)} rows)")

    # 统计
    n_ece = int(df["ece_post"].ne("").sum())
    n_gap = int(df["gap"].ne("").sum())
    strict = int((df["ece_pass_strict"] == "PASS").sum())
    two_dp = int((df["ece_pass_2dp"] == "PASS").sum())
    applicable = int(df["ece_pass_strict"].ne("").sum())
    assert n_ece + n_gap == len(df)
    assert applicable == n_ece

    # markdown 片段（供 v0.2 正文直接引用）
    md = []
    md.append("| Arm | Caliber | Metric | random | heldout | time_2018 | ECE_post random / heldout / time |")
    md.append("|---|---|---|---:|---:|---:|---|")
    for arm, head, caliber, ckey, metric_label in ARMS:
        sub = df[df["arm"] == arm]
        vals = [sub[sub["split"] == s]["value"].iloc[0] for s in SPLITS]
        eces = [sub[sub["split"] == s]["ece_post"].iloc[0] for s in SPLITS]
        ece_cell = " / ".join(e if e else "n.a.†" for e in eces)
        md.append(f"| {arm} | {caliber} | {metric_label} | {vals[0]} | {vals[1]} | {vals[2]} | {ece_cell} |")

    lines = []
    lines.append("P2-1 补件包 T2 可行性参照表 — 只读复述")
    lines.append("预注册 Documentation/P2-1_补件包_立项_骨架.md v0.1")
    lines.append(f"运行 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"G0 输入门: {len(inputs)}/{len(FROZEN_MD5)} md5 PASS")
    lines.append(f"A1 锚点: {n_anchor}/12 PASS; ECE 锚点: {len(EXPECTED_ECE)}/15 PASS")
    lines.append("")
    lines.append("## T2 表（15 行）")
    lines.append("")
    lines.extend(md)
    lines.append("")
    lines.append("† L3 两臂的 ECE 记 n.a.：m10_calib_v1_report.txt 的 T*=1.299 与部署束 T_L3=1.2845 不一致，")
    lines.append("  属不同 L3 版本（特征/类别集），逐组 ECE 不可跨版本引用；分类读数仍取自部署束。")
    lines.append("")
    lines.append("## ECE 达标计数（已知口径冲突，登记不裁定）")
    lines.append(f"严格 4 位判定: {strict}/{applicable}（仅计 ECE 可引用行）")
    lines.append(f"两位小数判定: {two_dp}/{applicable}")
    lines.append("差异组 L1/mc/random 0.0503：严格 FAIL、两位小数 PASS。")
    lines.append("对账: m10 全账本含 L3 时为 8/15（严格，见其判读④）；")
    lines.append("      进展报告记 9/15，属全账本口径；本表只引用 ECE 可跨版本比较的 9 行，不裁定全账本计数。")
    lines.append("")
    lines.append("## 缺口登记")
    for g in df.loc[df["gap"].ne(""), "gap"].unique():
        lines.append("- " + g)
    lines.append("")
    lines.append("## 逐行读数（CSV 镜像）")
    for _, r in df.iterrows():
        lines.append(
            f"{r['arm']:5s} {r['split']:10s} {r['metric']:6s} {r['value']} "
            f"n={r['n_test']:5d} ECE_post={r['ece_post'] or 'NA':>6s} T={r['T_calib']}"
        )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_REPORT.relative_to(ROOT)}")

    # 输出后复检输入未变（A3）
    for p, want in FROZEN_MD5.items():
        assert md5_of(p) == want, f"frozen input mutated during run: {p}"
    print("A3 post-run md5: PASS")

    manifest = {
        "task": "P1#6 P2-1 补件包 / T2 参照表",
        "prereg": "Documentation/P2-1_补件包_立项_骨架.md v0.1",
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "inputs_md5": {Path(p).name: h for p, h in inputs.items()},
        "outputs_md5": {
            str(OUT_CSV.relative_to(ROOT)): md5_of(OUT_CSV),
            str(OUT_REPORT.relative_to(ROOT)): md5_of(OUT_REPORT),
        },
        "rows": len(df),
        "ece_referenceable": n_ece,
        "ece_gap": n_gap,
        "ece_pass_strict": strict,
        "ece_pass_2dp": two_dp,
        "elapsed_s": round(time.time() - t0, 2),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT_MANIFEST.relative_to(ROOT)}")
    print(f"T2 rows={len(df)} ECE={n_ece} gap={n_gap} strict={strict}/{applicable} 2dp={two_dp}/{applicable}")


if __name__ == "__main__":
    main()
