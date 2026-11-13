# -*- coding: utf-8 -*-
"""Re-run the six leakage audits for the frozen v1.8 splits.

Inputs:  data/zeobench_db_v1_8.parquet, data/zeobench_splits_v1_8.parquet
Output:  one PASS/FAIL line per audit; exit code 1 if any audit fails.

    python splits/audit_splits.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "zeobench_db_v1_8.parquet"
SPLITS = ROOT / "data" / "zeobench_splits_v1_8.parquet"

HELDOUT_FRAMEWORKS = ["LTL", "SAS", "ITG", "RTH", "SZR", "LTA"]
SPLIT_COLUMNS = ["split_random", "split_heldout", "split_time_2018", "split_time_2020"]


def load() -> pd.DataFrame:
    db = pd.read_parquet(DB, columns=["route_id", "doi", "year", "src_row_id"])
    sp = pd.read_parquet(SPLITS).drop(columns=["doi"])
    df = db.merge(sp, on="route_id", how="inner", validate="one_to_one")
    if len(df) != len(sp):
        raise SystemExit(f"route_id join lost rows: {len(sp)} -> {len(df)}")
    return df


def audit_doi_purity(df: pd.DataFrame) -> tuple[int, str]:
    bad = 0
    for col in SPLIT_COLUMNS:
        part = df[df[col].isin(["train", "val", "test"])]
        spans = part.groupby("doi")[col].nunique()
        bad += int((spans > 1).sum())
    return bad, "DOIs spanning more than one set"


def audit_group_purity(df: pd.DataFrame) -> tuple[int, str]:
    bad = 0
    for col in SPLIT_COLUMNS:
        part = df[df[col].isin(["train", "val", "test"])]
        spans = part.groupby(["fw_key", "doi"])[col].nunique()
        bad += int((spans > 1).sum())
    return bad, "(framework, DOI) groups spanning more than one set"


def audit_heldout_absence(df: pd.DataFrame) -> tuple[int, str]:
    train = df[df["split_heldout"] == "train"]
    bad = int(train["fw_key"].isin(HELDOUT_FRAMEWORKS).sum())
    return bad, "held-out framework rows in heldout training"


def audit_increment_disjoint(df: pd.DataFrame) -> tuple[int, str]:
    increment = df["src_row_id"].isna()
    overlap = set(df.loc[increment, "doi"]) & set(df.loc[~increment, "doi"])
    return len(overlap), "increment DOIs also present in the base corpus"


def audit_time_monotonic(df: pd.DataFrame) -> tuple[int, str]:
    bad = 0
    for cutoff in (2018, 2020):
        col = f"split_time_{cutoff}"
        train = df.loc[df[col] == "train", "year"]
        test = df.loc[df[col] == "test", "year"]
        if train.isna().any() or test.isna().any():
            bad += 1
        if not train.max() <= cutoff < test.min():
            bad += 1
    return bad, "time-boundary violations (train <= cutoff < test)"


def audit_jzo(df: pd.DataFrame) -> tuple[int, str]:
    jzo = df[df["fw_key"] == "JZO"]
    bad = int((jzo["split_time_2018"] != "test").sum())
    return bad, f"JZO rows off the time-test side (n={len(jzo)})"


def run_audits() -> list[tuple[str, int, str]]:
    df = load()
    checks = [
        ("DOI purity", audit_doi_purity),
        ("group purity", audit_group_purity),
        ("held-out framework absence", audit_heldout_absence),
        ("increment disjointness", audit_increment_disjoint),
        ("time boundary monotonicity", audit_time_monotonic),
        ("JZO placement", audit_jzo),
    ]
    return [(name, *fn(df)) for name, fn in checks]


def main() -> int:
    results = run_audits()
    for name, violations, detail in results:
        status = "PASS" if violations == 0 else "FAIL"
        print(f"{status}  {name:<28} violations={violations}  ({detail})")
    failed = [name for name, violations, _ in results if violations]
    print(f"\n{len(results) - len(failed)}/{len(results)} audits pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
