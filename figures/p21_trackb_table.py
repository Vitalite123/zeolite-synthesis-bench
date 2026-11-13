#!/usr/bin/env python
"""P2-1 Track B 官方参照表（B1）生成脚本。

预注册：Documentation/P2-1_TrackB参照表_立项_骨架.md（v0.1, 2026-10-24）
性质：只读汇总 + 制表。不训练、不推理、不重采样、不联网、不新增指标定义。

判据：
  G0  7 项输入 + 对账清单内 87 项 run 文件 md5 全等，否则 SystemExit
  A1  WSD 21/21 与出版源逐位一致（1e-9）；逐参数 F1 与出版源差 ≤1e-12
  A2  9 行、seed∈{1,3}、n_systems∈{370,372}、4 位格式自洽、diff 唯一行
  A3  运行前后全部输入 md5 零漂移

输出：
  Data/p21_trackb_b1_seeds.csv    21 行逐 seed 账本
  Data/p21_trackb_b1_table.csv     9 行汇总（全精度 + 4 位字符串）
  Data/p21_trackb_b1_params.csv    9×12 逐参数 COV-P/COV-R
  Data/p21_trackb_b1_manifest.json
  Data/p21_trackb_b1_report.txt
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ZG = BASE / "vendor" / "zeosyn_gen"
RUNS = ZG / "runs"
NB = ZG / "notebooks" / "compare_model_metrics.ipynb"
PUB_A = ZG / "figs" / "main_fig_source_data" / "fig2a-wasserstein_distance.json"
PUB_B = ZG / "figs" / "main_fig_source_data" / "fig2b-f1.json"
RECON_JSON = BASE / "data" / "p21_trackb_pub_reconcile.json"
RECON_CSV = BASE / "data" / "p21_trackb_pub_reconcile.csv"

OUT_SEEDS = BASE / "data" / "p21_trackb_b1_seeds.csv"
OUT_TABLE = BASE / "data" / "p21_trackb_b1_table.csv"
OUT_PARAMS = BASE / "data" / "p21_trackb_b1_params.csv"
OUT_MANIFEST = BASE / "data" / "p21_trackb_b1_manifest.json"
OUT_REPORT = BASE / "data" / "p21_trackb_b1_report.txt"

FROZEN = {
    PUB_A: "9b459241572db09da0a1ff4c4ed9cef9",
    PUB_B: "26ad1d742bf0812b89e77de4cc7d5a5f",
    NB: "36e29261a11868f0523e8811c0828cf7",
    RECON_CSV: "43d136d1f6cb8e3b1cb03d47f551f1ff",
    RECON_JSON: "bb98e3250cee3f10a5f8b8469a5fe3be",
}

WSD_TOL = 1e-9
F1_TOL = 1e-12
PARAMS = ["Si/Al", "Al/P", "Si/Ge", "Si/B", "Na/T", "K/T", "OH/T", "F/T",
          "H2O/T", "sda1/T", "cryst_temp", "cryst_time"]
TRACK_A_METHODS = {"FASE-DDPM", "TabDDPM-MLP", "KDE-QT", "uniform"}


def md5_of(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_pins(text: str) -> list[tuple[str, list[str]]]:
    block = re.search(r"for model_type, fnames, split in \[(.*?)\]:", text, re.S)
    if not block:
        raise SystemExit("A1 FAIL: notebook 中找不到 run 列表")
    pins = re.findall(r"\('([\w\-]+)',\s*\[([^\]]*)\],\s*'system'\)", block.group(1))
    return [(m, [s.strip().strip("'\"") for s in f.split(",") if s.strip()]) for m, f in pins]


def flatten_f1(entry):
    if isinstance(entry, dict) and "0" in entry and isinstance(entry["0"], dict):
        return entry["0"]
    return entry


def check_g0() -> dict:
    got = {}
    for p, want in FROZEN.items():
        if not p.is_file():
            raise SystemExit(f"G0 FAIL: missing {p}")
        h = md5_of(p)
        if h != want:
            raise SystemExit(f"G0 FAIL: md5 drift {p.name}: {h} != {want}")
        got[str(p.relative_to(BASE))] = h
    recon = json.loads(RECON_JSON.read_text(encoding="utf-8"))
    bad = []
    for path, want in recon["inputs_md5"].items():
        p = Path(path)
        if not p.is_file() or md5_of(p) != want:
            bad.append(path)
    if bad:
        raise SystemExit(f"G0 FAIL: run artifacts drift: {bad}")
    return got, recon["inputs_md5"]


def main() -> None:
    t0 = time.time()
    frozen_md5, run_md5 = check_g0()

    pins_notebook = parse_pins(NB.read_text(encoding="utf-8"))
    pins_ledger = (pd.read_csv(RECON_CSV)
                   .groupby("arm", sort=False)["version"].apply(list).to_dict())
    order = [a for a, _ in pins_notebook]
    if set(order) != set(pins_ledger) or any(
            pins_ledger[a] != v for a, v in pins_notebook):
        raise SystemExit("A1 FAIL: notebook 钉定与对账账本不一致")

    pub_a = json.loads(PUB_A.read_text(encoding="utf-8"))
    pub_b = json.loads(PUB_B.read_text(encoding="utf-8"))

    seeds = []
    f1_max_global = 0.0
    for arm, versions in pins_notebook:
        if len(pub_a[arm]) != len(versions):
            raise SystemExit(f"A1 FAIL: {arm} 出版 seed 数与钉定不一致")
        for k, ver in enumerate(versions):
            d = RUNS / arm / "system" / ver
            wsd = json.loads((d / "wsd_zeo_osda.json").read_text(encoding="utf-8"))["WSD mean"]
            mmd = json.loads((d / "mmd_zeo_osda.json").read_text(encoding="utf-8"))["MMD mean"]
            prec = pd.read_csv(d / "prec_zeo_osda_df.csv").drop(columns=["zeo", "osda"])
            rec = pd.read_csv(d / "rec_zeo_osda_df.csv").drop(columns=["zeo", "osda"])
            p_mean = prec.mean(axis=0)
            r_mean = rec.mean(axis=0)

            # A1: WSD 出版对账
            delta = float(wsd) - float(pub_a[arm][k])
            if abs(delta) > WSD_TOL:
                raise SystemExit(f"A1 FAIL: {arm}/{ver} WSD Δ={delta:.3e}")

            # A1: 逐参数 F1 交叉校验（官方 notebook 口径：算术平均）
            f1 = pd.Series((p_mean.to_numpy() + r_mean.to_numpy()) / 2.0,
                           index=[c.replace("_prec", "") for c in p_mean.index])
            pub_f1 = flatten_f1(pub_b[arm][k])
            f1d = max(abs(float(v) - float(f1[PARAMS[i]]))
                      for i, v in enumerate(pub_f1[p] for p in PARAMS))
            f1_max_global = max(f1_max_global, f1d)
            if f1d > F1_TOL:
                raise SystemExit(f"A1 FAIL: {arm}/{ver} F1 Δ={f1d:.3e}")

            seeds.append({
                "arm": arm,
                "seed_index": k,
                "version": ver,
                "run_dir": str(d.relative_to(BASE)),
                "wsd": float(wsd),
                "mmd": float(mmd),
                "covp": float(p_mean.mean()),
                "covr": float(r_mean.mean()),
                "n_systems": int(len(prec)),
                "f1_max_abs_delta": f1d,
            })
            if len(rec) != len(prec):
                raise SystemExit(f"A2 FAIL: {arm}/{ver} prec/rec 行数不一致")

    seed_df = pd.DataFrame(seeds)
    seed_df.to_csv(OUT_SEEDS, index=False, encoding="utf-8-sig")

    # ---------------- 汇总 ----------------
    rows = []
    for arm in order:
        g = seed_df[seed_df.arm == arm]
        if g.n_systems.nunique() != 1:
            raise SystemExit(f"A2 FAIL: {arm} 各 seed 体系数不一致")
        n = len(g)
        if n not in (1, 3):
            raise SystemExit(f"A2 FAIL: {arm} seed 数={n}")
        row = {"arm": arm, "n_seeds": n, "n_systems": int(g.n_systems.iloc[0])}
        for col in ("wsd", "mmd", "covp", "covr"):
            v = g[col].to_numpy(dtype=float)
            row[f"{col}_mean"] = float(np.mean(v))
            row[f"{col}_std"] = float(np.std(v, ddof=0)) if n > 1 else None
            row[f"{col}_display"] = f"{np.mean(v):.4f}" + (
                "" if n == 1 else f" ± {np.std(v, ddof=0):.4f}")
        rows.append(row)
    table = pd.DataFrame(rows)
    if len(table) != 9 or set(table.arm) & TRACK_A_METHODS:
        raise SystemExit("A2 FAIL: 行集异常")
    if table[table.arm == "diff"].shape[0] != 1:
        raise SystemExit("A2 FAIL: diff 行不唯一")
    for r in table.itertuples():
        for col in ("wsd", "mmd", "covp", "covr"):
            m, s = getattr(r, f"{col}_mean"), getattr(r, f"{col}_std")
            if pd.isna(s):
                s = None
            expect = f"{m:.4f}" + ("" if s is None else f" ± {s:.4f}")
            if getattr(r, f"{col}_display") != expect:
                raise SystemExit(f"A2 FAIL: {r.arm}/{col} 格式自洽失败")
            if s is not None and f"{s:.4f}" == "0.0000":
                raise SystemExit(f"A2 FAIL: {r.arm}/{col} std 显示为 0.0000")
    table.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")

    # ---------------- 逐参数表 ----------------
    prows = []
    for arm in order:
        g = seed_df[seed_df.arm == arm]
        p_acc, r_acc = [], []
        for ver in g.version:
            d = RUNS / arm / "system" / ver
            prec = pd.read_csv(d / "prec_zeo_osda_df.csv").drop(columns=["zeo", "osda"])
            rec = pd.read_csv(d / "rec_zeo_osda_df.csv").drop(columns=["zeo", "osda"])
            p_acc.append(prec.mean(axis=0).to_numpy())
            r_acc.append(rec.mean(axis=0).to_numpy())
        pm = np.mean(np.vstack(p_acc), axis=0)
        rm = np.mean(np.vstack(r_acc), axis=0)
        for i, p in enumerate(PARAMS):
            prows.append({"arm": arm, "param": p,
                          "covp": float(pm[i]), "covr": float(rm[i]),
                          "covp_4dp": f"{pm[i]:.4f}", "covr_4dp": f"{rm[i]:.4f}"})
    params = pd.DataFrame(prows)
    if len(params) != 9 * 12:
        raise SystemExit("A2 FAIL: 逐参数行数异常")
    params.to_csv(OUT_PARAMS, index=False, encoding="utf-8-sig")

    # ---------------- A3 冻结复算 ----------------
    for p, want in FROZEN.items():
        if md5_of(p) != want:
            raise SystemExit(f"A3 FAIL: {p.name} 漂移")
    for path, want in run_md5.items():
        if md5_of(Path(path)) != want:
            raise SystemExit(f"A3 FAIL: {path} 漂移")

    # ---------------- 报告 ----------------
    L = []
    L.append("P2-1 Track B 官方参照表（B1）生成报告")
    L.append("=" * 88)
    L.append(f"运行 {time.strftime('%Y-%m-%dT%H:%M:%S')}  预注册 v0.1")
    L.append("G0 PASS: 7 项输入 + 87 项 run 文件 md5 全等")
    L.append(f"A1 PASS: WSD 21/21（容差 {WSD_TOL:g}）；逐参数 F1 max|Δ|={f1_max_global:.2e}")
    L.append("A2 PASS: 9 行 / seed∈{1,3} / n_systems∈{370,372} / 4 位格式自洽")
    L.append("A3 PASS: 运行前后输入 md5 零漂移")
    L.append("")
    L.append("## Table B1（† 出版值；‡ 本地官方指标；不可与 Table 5 比较）")
    L.append("")
    L.append("| arm | seeds | n_systems | WSD† | MMD‡ | COV-P‡ | COV-R‡ |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in table.itertuples():
        L.append(f"| {r.arm} | {r.n_seeds} | {r.n_systems} | {r.wsd_display} | "
                 f"{r.mmd_display} | {r.covp_display} | {r.covr_display} |")
    L.append("")
    L.append("口径：464 = 官方清单测试体系；372 = both_graph_feat_present=True 过滤后；"
             "370 = AMD 行数。COV-P/R 为 12 参数均值的均值。单 seed 臂不报 std。")
    L.append("")
    L.append("## Table S-B1（逐参数，COV-P / COV-R，4 位）")
    L.append("")
    L.append("| arm | " + " | ".join(PARAMS) + " |")
    L.append("|---|" + "---|" * len(PARAMS))
    for arm in order:
        g = params[params.arm == arm]
        L.append(f"| {arm} | " + " | ".join(g.covp_4dp) + " |")
    L.append("")
    L.append("（上行仅 COV-P；COV-R 见 Data/p21_trackb_b1_params.csv）")
    L.append("")
    L.append(f"产出: {OUT_SEEDS.name}, {OUT_TABLE.name}, {OUT_PARAMS.name}, {OUT_MANIFEST.name}")
    OUT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")

    manifest = {
        "task": "P2-1 Table B1 official-reported reference (read-only aggregation)",
        "preregistration": "Documentation/P2-1_TrackB参照表_立项_骨架.md",
        "preregistration_version": "v0.1",
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": f"{__import__('sys').version.split()[0]}",
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "tolerances": {"wsd": WSD_TOL, "f1": F1_TOL},
        "assertions": {"G0": "PASS", "A1": "PASS", "A2": "PASS", "A3": "PASS"},
        "f1_max_abs_delta": f1_max_global,
        "inputs_md5_frozen": frozen_md5,
        "inputs_md5_runs": run_md5,
        "outputs_md5": {str(p.relative_to(BASE)): md5_of(p)
                        for p in (OUT_SEEDS, OUT_TABLE, OUT_PARAMS, OUT_REPORT)},
        "notes": [
            "WSD 与 Fig. 2a 源数据逐位一致。",
            "MMD 与 COV-P/COV-R 无出版数值，标为本地官方指标。",
            "COV 为 12 参数均值的均值，与 Track A 逐体系口径不同。",
            "diff 采用 v3；run1 为同值权重副本。",
        ],
        "elapsed_s": round(time.time() - t0, 2),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
