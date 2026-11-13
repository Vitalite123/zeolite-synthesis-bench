# -*- coding: utf-8 -*-
"""Contract tests for the frozen generated-sample parquets in data/samples/.

The md5 anchors come from data/wp5_bench_v1_manifest.json, produced by
baselines/wp5_bench_v1_table.py. Weight md5s are checked only when weights/ has
been populated (see baselines/download_weights.py).
"""
import hashlib
import importlib.util
import json
import os

import pandas as pd
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMP = os.path.join(BASE, 'data', 'samples')

METHOD_FILES = {
    'FASE-DDPM v0': 'wp5_g3_formal_v0_syn.parquet',
    'FASE-DDPM v0.5': 'wp5_g3_formal_v05_syn.parquet',
    'KDE-QT': 'wp5_bench_v1_kde_syn.parquet',
    'TabDDPM-MLP': 'wp5_bench_v1_tabddpm_syn.parquet',
    'Random-Uniform': 'wp5_bench_v1_random_syn.parquet',
}

RECIPE_COLS = ['Si/Al', 'Al/P', 'Si/Ge', 'Si/B', 'Na/T', 'K/T', 'OH/T', 'F/T',
               'H2O/T', 'sda1/T', 'cryst_temp', 'cryst_time']


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _load_weights_manifest():
    spec = importlib.util.spec_from_file_location(
        'download_weights', os.path.join(BASE, 'baselines', 'download_weights.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FILES, mod.md5


def test_samples_manifest_md5():
    with open(os.path.join(SAMP, 'wp5_bench_v1_manifest.json'), encoding='utf-8') as f:
        man = json.load(f)
    assert man['protocol']['systems'] == 123
    assert man['protocol']['ns'] == 200
    for method, fname in METHOD_FILES.items():
        path = os.path.join(SAMP, fname)
        assert os.path.exists(path), 'missing frozen sample parquet: %s' % fname
        assert md5(path) == man['methods'][method], 'md5 mismatch: %s' % fname


@pytest.mark.parametrize('fname', sorted(METHOD_FILES.values()))
def test_sample_shape(fname):
    df = pd.read_parquet(os.path.join(SAMP, fname))
    assert len(df) == 123 * 200
    for c in RECIPE_COLS:
        assert c in df.columns


def test_weight_md5_when_present():
    files, md5fn = _load_weights_manifest()
    wgt = os.path.join(BASE, 'weights')
    if not os.path.isdir(wgt) or not os.listdir(wgt):
        pytest.skip('weights/ not populated; run baselines/download_weights.py first')
    for name, want in files.items():
        p = os.path.join(wgt, name)
        if os.path.exists(p) and want:
            assert md5fn(p) == want, 'weight md5 mismatch: %s' % name
