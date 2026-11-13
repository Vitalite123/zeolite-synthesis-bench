"""Figure manifest: every shipped PNG must match its recorded md5."""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ASSETS = ROOT / "figures" / "assets"
SECTIONS = ["figures", "publication_figures", "toc_graphic"]


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((DATA / "p21_supp_manifest.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("section", SECTIONS)
def test_section_present(manifest, section):
    assert section in manifest and manifest[section]


def test_figure_md5(manifest):
    entries = {}
    for section in SECTIONS:
        entries.update(manifest.get(section, {}))
    assert entries, "manifest carries no figure entries"
    for name, meta in entries.items():
        path = ASSETS / name
        assert path.exists(), f"missing figure asset: {name}"
        assert md5_of(path) == meta["md5"], f"md5 drift: {name}"


def test_figure_dpi(manifest):
    entries = {}
    for section in SECTIONS:
        entries.update(manifest.get(section, {}))
    for name, meta in entries.items():
        assert meta["dpi"] in (160, 300), f"{name}: unexpected dpi {meta['dpi']}"
