"""Frozen-corpus contract: shape, provenance and the counts quoted in the manuscript."""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.fixture(scope="module")
def db() -> pd.DataFrame:
    return pd.read_parquet(DATA / "zeobench_db_v1_8.parquet")


@pytest.fixture(scope="module")
def splits() -> pd.DataFrame:
    return pd.read_parquet(DATA / "zeobench_splits_v1_8.parquet")


def test_shape(db):
    assert db.shape == (24706, 149)


def test_route_ids_unique(db):
    assert db["route_id"].is_unique


def test_dois_and_years(db):
    assert db["doi"].nunique() == 3236
    assert db["year"].min() == 1966
    assert db["year"].max() == 2026


def test_label_counts(db):
    counts = db["label1"].value_counts().to_dict()
    assert counts == {"positive": 15187, "relative_neg": 4478,
                      "absolute_neg": 4993, "no_product": 48}


def test_increment_rows(db):
    assert int(db["src_row_id"].isna().sum()) == 745


@pytest.mark.parametrize("column,expected", [
    ("split_random", {"train": 19870, "val": 2426, "test": 2410}),
    ("split_heldout", {"train": 21616, "test": 3067, "excluded": 23}),
    ("split_time_2018", {"train": 22067, "test": 2379, "excluded": 260}),
    ("split_time_2020", {"train": 23662, "test": 784, "excluded": 260}),
])
def test_split_counts(splits, column, expected):
    assert splits[column].value_counts().to_dict() == expected


def test_split_rows_align(splits, db):
    assert len(splits) == len(db)
    assert set(splits["route_id"]) == set(db["route_id"])
