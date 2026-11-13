# -*- coding: utf-8 -*-
"""P1#6 / P2-1 补件包：图 F1–F4（确定性、headless、无新读数）

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
F3 = ASSETS / "p21_f3_stratified_covp.png"
F4 = ASSETS / "p21_f4_cc_gap.png"

MANIFEST = DATA / "p21_supp_manifest.json"

DPI = 160
PUB_DPI = 300
TOC = ASSETS / "p21_toc_graphic_300dpi.png"
TOC_SIZE = (3.5, 1.375)
PUBLICATIONS = [
    ("f1", "p21_f1_dataflow_300dpi.png", (8.69, 4.60), 0.852),
    ("f2", "p21_f2_dual_track_300dpi.png", (8.47, 3.58), 0.941),
    ("f3", "p21_f3_stratified_covp_300dpi.png", (8.06, 3.94), 0.937),
    ("f4", "p21_f4_cc_gap_300dpi.png", (7.91, 3.76), 0.941),
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


def box(ax, xy, w, h, text, fc="#eef3f8", ec="#2f5d8a", fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.015",
                                linewidth=1.0, edgecolor=ec, facecolor=fc))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=1.0, color="#5c6b73", shrinkA=0, shrinkB=0))


def scale_text(fig, s: float) -> None:
    """Publication re-render: scale every text object so the layout is preserved
    when the physical width is reduced to the two-column limit."""
    from matplotlib.text import Text

    for t in fig.findobj(Text):
        t.set_fontsize(t.get_fontsize() * s)


def fig_f1(out: Path, size: tuple[float, float] = (10.2, 5.4), dpi: int = DPI,
           font_scale: float = 1.0, wide_corpus: bool = False) -> None:
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cx, cw = (0.24, 0.52) if wide_corpus else (0.30, 0.40)
    box(ax, (cx, 0.845), cw, 0.115,
        "Hydrothermal records\n24,706 records × 149 fields · 3,236 DOIs (1966–2026)", bold=True)

    labels = [
        ("positive\n15,187", "#e3eef7"),
        ("relative_neg\n4,478", "#eaf1e4"),
        ("absolute_neg\n4,993", "#f6ece2"),
        ("no_product\n48", "#f0eef5"),
    ]
    w, gap = 0.215, 0.023
    x0 = (1 - (4 * w + 3 * gap)) / 2
    for i, (t, c) in enumerate(labels):
        x = x0 + i * (w + gap)
        box(ax, (x, 0.665), w, 0.095, t, fc=c, fs=8)
        arrow(ax, (0.5, 0.845), (x + w / 2, 0.760))

    splits = [("random\ntest 2,410", "#e3eef7"), ("heldout\ntest 3,067", "#eaf1e4"),
              ("time_2018\ntest 2,379", "#f6ece2")]
    w2, gap2 = 0.27, 0.045
    x2 = (1 - (3 * w2 + 2 * gap2)) / 2
    for i, (t, c) in enumerate(splits):
        x = x2 + i * (w2 + gap2)
        box(ax, (x, 0.475), w2, 0.095, t, fc=c, fs=8)
    ax.text(0.5, 0.135, "three frozen splits · six leakage audits PASS",
            ha="center", va="center", fontsize=7.8, style="italic", color="#4a5a63")

    box(ax, (0.055, 0.205), 0.30, 0.145,
        "Track A (canonical)\n123 systems × 200 samples\n2,087 real test rows\nselected from 3,362 test rows", fc="#dfeaf4", fs=7.8)
    box(ax, (0.385, 0.205), 0.24, 0.145,
        "Track B (ZeoSyn split)\n464 systems\nno cross-track inference", fc="#f1eee6", fs=8.2)
    box(ax, (0.655, 0.205), 0.29, 0.145,
        "Feasibility domain\n24,113 = 19,049 xtal + 5,064 neg\n490 features", fc="#e8f0e8", fs=8.2)

    arrow(ax, (0.2125, 0.475), (0.205, 0.350))
    arrow(ax, (0.4775, 0.475), (0.505, 0.350))
    arrow(ax, (0.8125, 0.475), (0.800, 0.350))

    ax.text(0.5, 0.065,
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


def load_bench() -> pd.DataFrame:
    df = pd.read_csv(DATA / "wp5_bench_v1_table.csv")
    assert len(df) == 5, f"bench rows {len(df)} != 5"
    for c in ["COVP_rich", "COVP_mid", "COVP_few", "CC_frac05"]:
        assert c in df.columns, f"missing column {c}"
    assert set(df["method"]) == set(METHOD_ORDER), "unexpected method set"
    return df.set_index("method").loc[METHOD_ORDER].reset_index()


def fig_f3(df: pd.DataFrame, out: Path, size: tuple[float, float] = (8.6, 4.2),
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
    ax.set_title("Coverage precision by skeleton-support stratum (Track A, 123 systems × 200)",
                 fontsize=10, pad=10)
    ax.text(0.5, -0.22, "Source: Data/wp5_bench_v1_table.csv (COVP_rich / COVP_mid / COVP_few)",
            transform=ax.transAxes, ha="center", fontsize=7.2, color="#6b7b84")
    scale_text(fig, font_scale)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


def fig_f4(df: pd.DataFrame, real_cc: float, out: Path, size: tuple[float, float] = (8.4, 4.0),
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
                fig_f3(df, out, size, PUB_DPI, fs)
            else:
                fig_f4(df, real_cc, out, size, PUB_DPI, fs)
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
        fig_f3(df, F3)
        fig_f4(df, real_cc, F4)

    for p, want in INPUTS.items():
        assert md5_of(p) == want, f"frozen input mutated: {p}"
    print("A3 post-run md5: PASS")

    figs = {}
    if MANIFEST.exists():
        figs = json.loads(MANIFEST.read_text(encoding="utf-8"))
    SOURCES = {"p21_f1_dataflow.png": "data/DATASHEET.md · data/wp5_g3_formal_report.txt",
               "p21_f2_dual_track.png": "ZeoSyn/DiffSyn non-merge audit (Methods 2.2)",
               "p21_f3_stratified_covp.png": "data/wp5_bench_v1_table.csv",
               "p21_f4_cc_gap.png": "data/wp5_bench_v1_table.csv · data/wp5_g3_formal_report.txt"}
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
        print("publication figures: 4/4 PASS")
    else:
        figs["figures"] = {
            p.name: {"md5": md5_of(p), "dpi": DPI, "source": SOURCES[p.name]}
            for p in [F1, F2, F3, F4]
        }
        print("FIGS: 4/4 PASS")
    MANIFEST.write_text(json.dumps(figs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"updated {MANIFEST.name}")


if __name__ == "__main__":
    main()
