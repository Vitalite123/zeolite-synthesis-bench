# -*- coding: utf-8 -*-
"""P1#6 / P2-1 补件包：图 F1–F5（确定性、headless、无新读数）

预注册：Documentation/P2-1_补件包_立项_骨架.md v0.1 §2.2 / A2
数据来源：
  F1 常量：DATA_DICT.md §22、规格 v1.1 §2、wp5_g3_formal_report.txt
  F2 常量：规格 v1.1 §2「不可合并的证据」（2026-10-23 审计）
  F3/F4：Data/wp5_bench_v1_table.csv；F4 真实参照 Data/wp5_g3_formal_report.txt
输出：Documentation/p21_f{1..4}_*.png，并把 figures 段并入 Data/p21_supp_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ASSETS = ROOT / "figures" / "assets"

INPUTS = {
    DATA / "feasibility_scorer_v1_2_report.txt": "35be508bc3650c9e1696e67aca09337b",
    DATA / "wp5_bench_v1_table.csv": "55a58dcf697a98cc4b1ca42d4159ac47",
    DATA / "wp5_g3_formal_report.txt": "8198d094b7bb0e4b1f45e9f3704bf802",
    DATA / "m10_calib_v1_report.txt": "783d61a5087faf4d28b4be339990dbb5",
}

F1 = ASSETS / "p21_f1_dataflow.png"
F2 = ASSETS / "p21_f2_dual_track.png"
F3 = ASSETS / "p21_f3_baselines.png"
F4 = ASSETS / "p21_f4_stratified_covp.png"
F5 = ASSETS / "p21_f5_cc_gap.png"

MANIFEST = DATA / "p21_supp_manifest.json"

DPI = 160
PUB_DPI = 300
TOC = ASSETS / "p21_toc_graphic_300dpi.png"
TOC_SIZE = (3.5, 1.375)
GA = ASSETS / "p21_ga_graphic_300dpi.png"
GA_PDF = ASSETS / "p21_ga_graphic.pdf"
GA_SIZE = (6.0, 2.4)  # 1800 x 720 px @300 dpi; >= Elsevier 1328 x 531 px (w x h), 2.5:1
PUBLICATIONS = [
    ("f1", "p21_f1_dataflow_300dpi.png", (8.69, 4.60), 0.852),
    ("f2", "p21_f2_dual_track_300dpi.png", (8.47, 3.58), 0.941),
    ("f3", "p21_f3_baselines_300dpi.png", (7.00, 3.85), 0.900),
    ("f4", "p21_f4_stratified_covp_300dpi.png", (8.06, 3.94), 0.937),
    ("f5", "p21_f5_cc_gap_300dpi.png", (7.91, 3.76), 0.941),
]
METHOD_ORDER = ["FASE-DDPM v0", "FASE-DDPM v0.5", "TabDDPM-MLP", "KDE-QT", "Random-Uniform"]
METHOD_LABEL = {
    "FASE-DDPM v0": "FASE v0",
    "FASE-DDPM v0.5": "FASE v0.5",
    "TabDDPM-MLP": "TabDDPM",
    "KDE-QT": "KDE-QT",
    "Random-Uniform": "Uniform",
}
COLORS = ["#2f5d8a", "#4f86b8", "#7aa6c2", "#a8c3d4", "#c9d6df"]
STRATA = [("COVP_rich", "rich (n=96)"), ("COVP_mid", "mid (n=9)"), ("COVP_few", "few (n=18)")]


def md5_of(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def gate() -> None:
    bad = []
    for p, want in INPUTS.items():
        if not p.exists():
            bad.append(f"MISSING {p}")
        elif md5_of(p) != want:
            bad.append(f"MD5 MISMATCH {p.name}")
    if bad:
        print("G0 INPUT GATE FAILED")
        for b in bad:
            print("  " + b)
        raise SystemExit(2)
    print(f"G0 input gate: {len(INPUTS)}/{len(INPUTS)} md5 PASS")


def box(ax, xy, w, h, text, fc="#eef3f8", ec="#2f5d8a", fs=8.5, bold=False, pad=0.02):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle=f"round,pad={pad},rounding_size=0.015",
                                linewidth=1.0, edgecolor=ec, facecolor=fc))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, p0, p1, color="#5c6b73"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=1.0, color=color, shrinkA=0, shrinkB=0,
                                 zorder=3))


def scale_text(fig, s: float) -> None:
    """Publication re-render: scale every text object so the layout is preserved
    when the physical width is reduced to the two-column limit."""
    from matplotlib.text import Text

    for t in fig.findobj(Text):
        t.set_fontsize(t.get_fontsize() * s)


def fig_f1(out: Path, size: tuple[float, float] = (10.2, 5.4), dpi: int = DPI,
           font_scale: float = 1.0, wide_corpus: bool = True) -> None:
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # layer 1: records container with the four outcome labels as inset boxes
    box(ax, (0.07, 0.755), 0.86, 0.220, "")
    ax.text(0.5, 0.930, "Hydrothermal records\n24,706 records × 149 fields · 3,236 DOIs (1966–2026)",
            ha="center", va="center", fontsize=8.5, fontweight="bold", linespacing=1.35)
    labels = [
        ("positive\n15,187", "#e3eef7"),
        ("relative_neg\n4,478", "#eaf1e4"),
        ("absolute_neg\n4,993", "#f6ece2"),
        ("no_product\n48", "#f0eef5"),
    ]
    for i, (t, c) in enumerate(labels):
        box(ax, (0.105 + i * 0.21, 0.775), 0.16, 0.085, t, fc=c, fs=8)

    # layer 2: three frozen splits, narrowed and centred to clear the left
    # channel that feeds the DDPM subset box
    splits = [("random\ntest 2,410", "#e3eef7"), ("heldout\ntest 3,067", "#eaf1e4"),
              ("time_2018\ntest 2,379", "#f6ece2")]
    w2, x2 = 0.19, 0.165
    centers = [x2 + i * (w2 + 0.05) + w2 / 2 for i in range(3)]
    for cx, (t, c) in zip(centers, splits):
        box(ax, (cx - w2 / 2, 0.590), w2, 0.090, t, fc=c, fs=8)
        arrow(ax, (cx, 0.735), (cx, 0.700))
    ax.text(0.5, 0.550, "three frozen splits · six leakage audits PASS",
            ha="center", va="center", fontsize=7.8, style="italic", color="#4a5a63")

    # layer 3: DDPM subset (generation line) and feasibility domain (right)
    box(ax, (0.055, 0.415), 0.36, 0.095,
        "DDPM subset (21,714 rows)\nsystem split seed 0 · test 3,362", fc="#dfeaf4", fs=7.8)
    box(ax, (0.60, 0.415), 0.33, 0.095,
        "Feasibility domain\n24,113 = 19,049 xtal + 5,064 neg\n490 features", fc="#e8f0e8", fs=8.2)
    # the DDPM subset is carved from the records container (left channel)
    arrow(ax, (0.10, 0.735), (0.10, 0.530))
    # labelled rows channel: records container -> feasibility domain
    arrow(ax, (0.92, 0.735), (0.92, 0.530))
    ax.text(0.935, 0.632, "labelled rows", ha="center", va="center",
            fontsize=6.4, rotation=90, color="#4a5a63")

    # layer 4: the two tracks, fed from layer 3
    box(ax, (0.055, 0.215), 0.36, 0.125,
        "Track A (canonical)\n123 systems × 200 samples\n2,087 of 3,362 real test rows", fc="#dfeaf4", fs=8.2, bold=True)
    box(ax, (0.53, 0.215), 0.27, 0.125,
        "Track B (ZeoSyn split)\n464 systems\nno cross-track inference", fc="#f1eee6", fs=8.2, bold=True)
    arrow(ax, (0.235, 0.395), (0.235, 0.360))

    # layer 5: external ZeoSyn source, feeding Track B from below
    ext = FancyBboxPatch((0.53, 0.090), 0.27, 0.060,
                         boxstyle="round,pad=0.02,rounding_size=0.015",
                         linewidth=1.0, edgecolor="#8a6d3b", facecolor="#f7f3ea",
                         linestyle=(0, (4, 2)))
    ax.add_patch(ext)
    ax.text(0.665, 0.120, "ZeoSyn corpus · external split\n(DiffSyn-published baselines)",
            ha="center", va="center", fontsize=7.0, linespacing=1.3)
    arrow(ax, (0.665, 0.170), (0.665, 0.195))

    ax.text(0.5, 0.025,
            "Sources: released data card · split manifest · benchmark report",
            ha="center", va="center", fontsize=7.2, color="#6b7b84")
    scale_text(fig, font_scale)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


def fig_f2(out: Path, size: tuple[float, float] = (9.0, 3.8), dpi: int = DPI,
           font_scale: float = 1.0) -> None:
    track_a = {"covered by DiffSyn predictions": 26, "inside ZeoSyn train/val": 97}
    track_b = {"inside our train": 350, "outside our train": 114}
    assert sum(track_a.values()) == 123 and sum(track_b.values()) == 464

    fig, ax = plt.subplots(figsize=size)
    rows = [
        ("Track A\n123 systems", track_a, ["#2f5d8a", "#b9cddd"], 0.62, 0.175),
        ("Track B\n464 systems", track_b, ["#c07a3a", "#e8d3bf"], 0.22, -0.175),
    ]
    for name, parts, cols, y, dy in rows:
        left = 0.0
        total = sum(parts.values())
        items = list(parts.items())
        for i, ((lab, n), c) in enumerate(zip(items, cols)):
            frac = n / total
            ax.barh(y, frac, left=left, height=0.26, color=c, edgecolor="white", linewidth=1.2)
            ax.text(left + frac / 2, y, f"{n}\n({frac * 100:.1f}%)", ha="center", va="center",
                    fontsize=9, color="white" if c in ("#2f5d8a", "#c07a3a") else "#243642")
            if i == 0 and frac < 0.40:
                tx, ha = left + 0.004, "left"
            elif i == len(items) - 1 and frac < 0.55:
                tx, ha = left + frac - 0.004, "right"
            else:
                tx, ha = left + frac / 2, "center"
            ax.text(tx, y + dy, lab, ha=ha, va="center", fontsize=7.6, color="#33454f")
            left += frac
        ax.text(-0.012, y, name, ha="right", va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(-0.16, 1.0)
    ax.set_ylim(0.0, 0.88)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_xlabel("share of systems", fontsize=8.5)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.set_title("Two tracks cannot be merged: evidence from the leakage audit",
                 fontsize=10, pad=10)
    ax.text(0.5, -0.30, "Source: released split manifest · benchmark audit report",
            transform=ax.transAxes, ha="center", fontsize=7.2, color="#6b7b84")
    scale_text(fig, font_scale)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


def fig_baselines(out: Path, size: tuple[float, float] = (10.4, 5.7), dpi: int = DPI,
                  font_scale: float = 1.0) -> None:
    """F3: generation paradigms of the five Track A baselines. Schematic only,
    no data readings; method facts follow Methods 2.4 and the registered
    training-line records (plain v0, accumulated recipe, v0.5 system-level
    oversampling; no Si/Al channel in the frozen variants)."""
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.015)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    BRAND = "#2f5d8a"
    lanes = [
        ("FASE-DDPM", "learned · in-house",
         [("Joint noising of", "recipe + conditioning"),
          ("MLP U-Net denoiser", "v-pred · 1,000 steps"),
          ("CFG sampling", "cond_scale 0.75")]),
        ("TabDDPM-MLP", "learned · independent",
         [("Quantile transform", "12 continuous columns"),
          ("Column-wise Gaussian", "diffusion"),
          ("MLP denoiser", "two-part conditioning")]),
        ("KDE-QT", "non-neural reference",
         [("Quantile transform", "of training rows"),
          ("Per-framework", "Gaussian KDE"),
          ("Global fallback", "if framework < 30 rows")]),
        ("Random-Uniform", "non-learned floor",
         [("Per-column", "min-max range"),
          ("Uniform draw", "per column · seed 42"),
          ("Non-learned floor", "anchors metric scale")]),
    ]
    w, gap = 0.205, 0.048
    x0 = (1 - (4 * w + 3 * gap)) / 2
    y_head, h_head = 0.779, 0.074
    y_s = [0.631, 0.482, 0.333]
    h_s = 0.099
    y_strip, h_strip = 0.114, 0.180
    y_score, h_score = 0.012, 0.066

    box(ax, (0.035, 0.911), 0.93, 0.069,
        "Target (zeo, osda) pair · framework + OSDA descriptors",
        fc="#e3eef7", bold=True, fs=8.2, pad=0.012)

    for i, (name, tag, stages) in enumerate(lanes):
        x = x0 + i * (w + gap)
        cx = x + w / 2
        main = i == 0
        fc = "#dfeaf4" if main else "#eef3f8"
        ec = BRAND if main else "#5c6b73"
        ax.add_patch(FancyBboxPatch((x, y_head), w, h_head,
                                    boxstyle="round,pad=0.012,rounding_size=0.012",
                                    linewidth=1.1 if main else 0.9, edgecolor=ec, facecolor=fc))
        ax.text(cx, y_head + h_head * 0.62, name, ha="center", va="center",
                fontsize=8.6, fontweight="bold", color=ec)
        ax.text(cx, y_head + h_head * 0.28, tag, ha="center", va="center",
                fontsize=7.3, color="#4a5a63")
        arrow(ax, (cx, 0.899), (cx, y_head + h_head + 0.012), color=ec)
        arrow(ax, (cx, y_head - 0.012), (cx, y_s[0] + h_s + 0.012), color=ec)
        for j, (l1, l2) in enumerate(stages):
            box(ax, (x, y_s[j]), w, h_s, f"{l1}\n{l2}", fc=fc, ec=ec, fs=7.3, pad=0.012)
            if j:
                arrow(ax, (cx, y_s[j - 1] - 0.012), (cx, y_s[j] + h_s + 0.012), color=ec)

    xs = x0
    cxs = xs + w / 2
    ax.add_patch(FancyBboxPatch((xs, y_strip), w, h_strip,
                                boxstyle="round,pad=0.012,rounding_size=0.012",
                                linewidth=1.0, edgecolor=BRAND, facecolor="none",
                                linestyle=(0, (4, 2))))
    ax.text(cxs, y_strip + h_strip / 2,
            "v0 \u2192 v0.5 training recipe\nv0 \u2014 plain run\nacc. qt · per-col loss\n"
            "+ inv-freq · EMA 0.999\nv0.5 + system oversampling",
            ha="center", va="center", fontsize=6.8, linespacing=1.4, color="#33454f")
    arrow(ax, (cxs, y_s[2] - 0.012), (cxs, y_strip + h_strip + 0.012), color=BRAND)

    for i in range(4):
        cx = x0 + i * (w + gap) + w / 2
        top = y_strip if i == 0 else y_s[2]
        arrow(ax, (cx, top - 0.012), (cx, y_score + h_score + 0.012),
              color=BRAND if i == 0 else "#5c6b73")
    box(ax, (0.035, y_score), 0.93, h_score,
        "Frozen scoring protocol (§2.3) · 12 metric functions · seed-42 rows",
        fc="#eef3f8", fs=8.6, pad=0.012)

    scale_text(fig, font_scale)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


def load_bench() -> pd.DataFrame:
    df = pd.read_csv(DATA / "wp5_bench_v1_table.csv")
    assert len(df) == 5, f"bench rows {len(df)} != 5"
    for c in ["COVP_rich", "COVP_mid", "COVP_few", "CC_frac05"]:
        assert c in df.columns, f"missing column {c}"
    assert set(df["method"]) == set(METHOD_ORDER), "unexpected method set"
    return df.set_index("method").loc[METHOD_ORDER].reset_index()


def fig_stratified(df: pd.DataFrame, out: Path, size: tuple[float, float] = (8.6, 4.2),
                   dpi: int = DPI, font_scale: float = 1.0) -> None:
    fig, ax = plt.subplots(figsize=size)
    x = range(len(df))
    width = 0.26
    shades = ["#2f5d8a", "#7aa6c2", "#c9d6df"]
    for i, ((col, lab), c) in enumerate(zip(STRATA, shades)):
        vals = df[col].tolist()
        pos = [xi + (i - 1) * width for xi in x]
        bars = ax.bar(pos, vals, width=width, color=c, label=lab, edgecolor="white", linewidth=0.8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.0, rotation=90)
    ax.set_xticks(list(x))
    ax.set_xticklabels([METHOD_LABEL[m] for m in df["method"]], fontsize=8.5)
    ax.set_ylabel("COV-P (per-system mean)", fontsize=9)
    ax.set_ylim(0, 0.92)
    ax.legend(title="train support stratum", fontsize=7.8, title_fontsize=7.8, frameon=False, ncol=3,
              loc="upper right")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title("Coverage precision by framework-support stratum (Track A, 123 systems × 200)",
                 fontsize=10, pad=10)
    ax.text(0.5, -0.22, "Source: Data/wp5_bench_v1_table.csv (COVP_rich / COVP_mid / COVP_few)",
            transform=ax.transAxes, ha="center", fontsize=7.2, color="#6b7b84")
    scale_text(fig, font_scale)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


def fig_cc_gap(df: pd.DataFrame, real_cc: float, out: Path, size: tuple[float, float] = (8.4, 4.0),
               dpi: int = DPI, font_scale: float = 1.0) -> None:
    vals = df["CC_frac05"].tolist()
    fig, ax = plt.subplots(figsize=size)
    bars = ax.bar(range(len(df)), vals, width=0.55, color=COLORS, edgecolor="white", linewidth=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.018, f"{v:.4f}\n({v / real_cc * 100:.0f}%)",
                ha="center", va="bottom", fontsize=7.6)
    ax.axhline(real_cc, color="#b03a2e", linewidth=1.4, linestyle="--")
    ax.text(len(df) - 0.4, real_cc + 0.022, f"real held-out recipes {real_cc:.4f}",
            ha="right", va="bottom", fontsize=8.0, color="#b03a2e")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([METHOD_LABEL[m] for m in df["method"]], fontsize=8.5)
    ax.set_ylabel("CC@0.5 (condition consistency)", fontsize=9)
    ax.set_ylim(0, 1.0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title("Generation–reality gap in condition consistency (n = 2,087 real rows)",
                 fontsize=10, pad=10)
    ax.text(0.5, -0.20, "Bars: Data/wp5_bench_v1_table.csv · reference: Data/wp5_g3_formal_report.txt; "
                        "percentages = method / real",
            transform=ax.transAxes, ha="center", fontsize=7.2, color="#6b7b84")
    scale_text(fig, font_scale)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


def toc_panel(ax, x: float, y: float, w: float, h: float, title: str, body: str,
              fc: str = "#eef3f8", title_fs: float = 5.2, body_fs: float = 5.6) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                linewidth=0.7, edgecolor="#2f5d8a", facecolor=fc))
    ax.text(x + w / 2, y + h - 0.055, title, ha="center", va="center",
            fontsize=title_fs, style="italic", color="#3d4d57")
    ax.text(x + w / 2, y + (h - 0.075) / 2 - 0.01, body, ha="center", va="center",
            fontsize=body_fs, linespacing=1.4)


def fig_toc(out: Path = TOC, size: tuple[float, float] = TOC_SIZE, dpi: int = PUB_DPI) -> None:
    """TOC graphic, 3.5 x 1.375 in (JCIM C5). Numbers are restricted to those
    already drawn in F1/F2; no new reading is introduced."""
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    toc_panel(ax, 0.012, 0.545, 0.262, 0.405, "hydrothermal records",
              "24,706 records\n3,236 DOIs · 149 fields")
    toc_panel(ax, 0.348, 0.545, 0.268, 0.405, "four outcome labels",
              "15,187 positive\n4,478 rel-neg\n4,993 abs-neg\n48 no-product",
              fc="#eaf1e4", body_fs=5.2)
    toc_panel(ax, 0.690, 0.545, 0.298, 0.405, "three frozen splits",
              "test rows\n2,410 / 3,067 / 2,379\nsix leakage audits PASS",
              fc="#f6ece2", body_fs=5.2)

    arrow(ax, (0.276, 0.745), (0.345, 0.745))
    arrow(ax, (0.618, 0.745), (0.687, 0.745))

    toc_panel(ax, 0.012, 0.135, 0.415, 0.315, "Track A · canonical leaderboard",
              "123 systems × 200 samples\n2,087 real test rows", fc="#dfeaf4", title_fs=5.4)
    toc_panel(ax, 0.573, 0.135, 0.415, 0.315, "Track B · ZeoSyn split",
              "464 systems\nno cross-track inference", fc="#f1eee6", title_fs=5.4)

    ax.plot([0.4995, 0.4995], [0.155, 0.430], linestyle=(0, (2, 2)), linewidth=0.8,
            color="#b03a2e")
    ax.text(0.4995, 0.2925, "not merged", ha="center", va="center", fontsize=4.6,
            style="italic", color="#b03a2e", rotation=90,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    ax.text(0.5, 0.060,
            "26/123 covered by DiffSyn predictions · 97 inside ZeoSyn train/val · "
            "350/464 inside our train",
            ha="center", va="center", fontsize=4.6, color="#4a5a63")

    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"wrote {out.name}")


def fig_ga(out: Path = GA, size: tuple[float, float] = GA_SIZE, dpi: int = PUB_DPI) -> None:
    """Graphical abstract for CMS (Elsevier): minimum 1328 x 531 px (w x h),
    2.5:1, readable at 5 x 13 cm. Numbers are frozen readings only (no new one)."""
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # pipeline: literature -> audited benchmark -> dual-track evaluation
    toc_panel(ax, 0.012, 0.600, 0.290, 0.360, "hydrothermal literature",
              "24,706 records · 3,236 DOIs\n1966–2026 · four outcome labels\nfailed routes kept as negatives",
              fc="#eef3f8", title_fs=7.6, body_fs=7.0)
    toc_panel(ax, 0.362, 0.600, 0.290, 0.360, "leakage-audited benchmark",
              "three frozen splits\nsix audits PASS · two tasks\nseven metric families · one API",
              fc="#eaf1e4", title_fs=7.6, body_fs=7.0)
    toc_panel(ax, 0.702, 0.600, 0.286, 0.360, "dual-track evaluation",
              "Track A 123 systems × 200 samples\nTrack B 464 ZeoSyn systems\nfeasibility heads on three splits",
              fc="#f6ece2", title_fs=7.6, body_fs=7.0)

    arrow(ax, (0.314, 0.780), (0.350, 0.780))
    arrow(ax, (0.664, 0.780), (0.696, 0.780))

    # second row: generator-context panel (left) + findings band (right).
    # the left panel names the tested generator categories so the model
    # names quoted in the findings read as part of one protocol.
    toc_panel(ax, 0.012, 0.145, 0.262, 0.300, "generators evaluated",
              "3 diffusion generators:\nFASE-DDPM v0 · v0.5 · TabDDPM\nKDE-QT (kernel density)\nuniform sampler (lower bound)",
              fc="#f1eee6", title_fs=7.6, body_fs=6.6)

    # findings band, narrowed and shifted right, fed by the pipeline
    band = FancyBboxPatch((0.306, 0.145), 0.682, 0.300,
                          boxstyle="round,pad=0.006,rounding_size=0.012",
                          linewidth=0.7, edgecolor="#2f5d8a", facecolor="#f7f4ee")
    ax.add_patch(band)
    ax.text(0.647, 0.398, "what the benchmark shows", ha="center", va="center",
            fontsize=7.6, style="italic", color="#3d4d57")
    cols = [
        "FASE-DDPM leads\nCOV-P 0.7359",
        "coverage falls\n15–19% rel.\nrich → few",
        "CC 0.1672 vs\n0.8730 real\ngap: generation",
        "feasibility near\nchance on heldout",
    ]
    for i, body in enumerate(cols):
        ax.text(0.414 + i * 0.1555, 0.245, body, ha="center", va="center",
                fontsize=6.8, linespacing=1.45)
    # arrow from the bottom-centre of the middle pipeline panel, vertically
    # down to the findings band (which now spans underneath)
    arrow(ax, (0.507, 0.594), (0.507, 0.451))

    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"wrote {out.name}")


def main() -> None:
    gate()
    df = load_bench()

    g3 = (DATA / "wp5_g3_formal_report.txt").read_text(encoding="utf-8")
    m = re.search(r"frac@0\.5=([\d.]+)\s+top1=([\d.]+)", g3)
    assert m, "real CC anchor missing"
    real_cc = float(m.group(1))
    assert abs(real_cc - 0.8730) <= 1e-9, f"real CC drift {real_cc}"
    assert m.group(2) == "0.9109"
    print(f"F4 anchor: real CC@0.5={real_cc:.4f} PASS")

    publication = "--publication" in sys.argv
    toc = "--toc" in sys.argv
    ga = "--ga" in sys.argv
    if toc:
        from PIL import Image

        fig_toc(TOC, TOC_SIZE, PUB_DPI)
        with Image.open(TOC) as im:
            w_px, h_px = im.size
            dpi_x = im.info["dpi"][0]
        w_in, h_in = w_px / PUB_DPI, h_px / PUB_DPI
        assert abs(dpi_x - PUB_DPI) < 0.01, f"toc: dpi {dpi_x}"
        assert abs(w_in - 3.5) <= 0.01, f"toc: width {w_in:.3f} in"
        assert abs(h_in - 1.375) <= 0.01, f"toc: height {h_in:.3f} in"
        print(f"  {TOC.name}: {w_px}x{h_px} px = {w_in:.3f} x {h_in:.3f} in @ {PUB_DPI} dpi")
    elif ga:
        from PIL import Image

        fig_ga(GA, GA_SIZE, PUB_DPI)
        fig_ga(GA_PDF, GA_SIZE, PUB_DPI)
        with Image.open(GA) as im:
            w_px, h_px = im.size
            dpi_x = im.info["dpi"][0]
        w_in, h_in = w_px / PUB_DPI, h_px / PUB_DPI
        assert abs(dpi_x - PUB_DPI) < 0.01, f"ga: dpi {dpi_x}"
        assert w_px >= 1328 and h_px >= 531, f"ga: {w_px}x{h_px} px < Elsevier 1328x531"
        print(f"  {GA.name}: {w_px}x{h_px} px = {w_in:.3f} x {h_in:.3f} in @ {PUB_DPI} dpi "
              f"(>= 1328x531 PASS)")
        print(f"  {GA_PDF.name}: vector copy for submission")
    elif publication:
        from PIL import Image

        outs = []
        for key, name, size, fs in PUBLICATIONS:
            out = ASSETS / name
            if key == "f1":
                fig_f1(out, size, PUB_DPI, fs, wide_corpus=True)
            elif key == "f2":
                fig_f2(out, size, PUB_DPI, fs)
            elif key == "f3":
                fig_baselines(out, size, PUB_DPI, fs)
            elif key == "f4":
                fig_stratified(df, out, size, PUB_DPI, fs)
            else:
                fig_cc_gap(df, real_cc, out, size, PUB_DPI, fs)
            with Image.open(out) as im:
                w_px, h_px = im.size
                dpi_x = im.info["dpi"][0]
            w_in, h_in = w_px / PUB_DPI, h_px / PUB_DPI
            assert abs(dpi_x - PUB_DPI) < 0.01, f"{name}: dpi {dpi_x}"
            assert 4.1 <= w_in <= 7.05, f"{name}: width {w_in:.2f} in"
            assert h_in <= 9.1, f"{name}: depth {h_in:.2f} in"
            print(f"  {name}: {w_px}x{h_px} px = {w_in:.2f} x {h_in:.2f} in @ {PUB_DPI} dpi")
            outs.append((out, w_in, h_in))
    else:
        fig_f1(F1)
        fig_f2(F2)
        fig_baselines(F3)
        fig_stratified(df, F4)
        fig_cc_gap(df, real_cc, F5)

    for p, want in INPUTS.items():
        assert md5_of(p) == want, f"frozen input mutated: {p}"
    print("A3 post-run md5: PASS")

    figs = {}
    if MANIFEST.exists():
        figs = json.loads(MANIFEST.read_text(encoding="utf-8"))
    SOURCES = {"p21_f1_dataflow.png": "data/DATASHEET.md · data/wp5_g3_formal_report.txt",
               "p21_f2_dual_track.png": "ZeoSyn/DiffSyn non-merge audit (Methods 2.2)",
               "p21_f3_baselines.png": "Methods 2.4 · registered training-line records (G3)",
               "p21_f4_stratified_covp.png": "data/wp5_bench_v1_table.csv",
               "p21_f5_cc_gap.png": "data/wp5_bench_v1_table.csv · data/wp5_g3_formal_report.txt"}
    if toc:
        figs["toc_graphic"] = {
            TOC.name: {"md5": md5_of(TOC), "dpi": PUB_DPI,
                       "width_in": round(w_in, 3), "height_in": round(h_in, 3),
                       "source": "F1/F2 constants only (data/DATASHEET.md · Methods 2.2)"}
        }
        print("TOC graphic: PASS")
    elif publication:
        figs["publication_figures"] = {
            out.name: {"md5": md5_of(out), "dpi": PUB_DPI,
                       "width_in": round(w_in, 3), "height_in": round(h_in, 3),
                       "source": SOURCES[out.name.replace("_300dpi", "")]}
            for out, w_in, h_in in outs
        }
        print("publication figures: 5/5 PASS")
    elif ga:
        figs["ga_graphic"] = {
            GA.name: {"md5": md5_of(GA), "dpi": PUB_DPI,
                      "width_in": round(w_in, 3), "height_in": round(h_in, 3),
                      "source": "F1/F2 constants only (data/DATASHEET.md · Methods 2.2/2.3)"},
            GA_PDF.name: {"md5": md5_of(GA_PDF), "format": "pdf",
                          "source": "vector copy of p21_ga_graphic_300dpi.png"},
        }
        print("Graphical abstract: PASS")
    else:
        figs["figures"] = {
            p.name: {"md5": md5_of(p), "dpi": DPI, "source": SOURCES[p.name]}
            for p in [F1, F2, F3, F4, F5]
        }
        print("FIGS: 5/5 PASS")
    MANIFEST.write_text(json.dumps(figs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"updated {MANIFEST.name}")


if __name__ == "__main__":
    main()
