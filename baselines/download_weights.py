# -*- coding: utf-8 -*-
"""
Download the trained checkpoints and scorer bundles for ZeoliteSynthesisBench v1
into weights/. The binaries are hosted as GitHub Release assets (tag weights-v1)
because they are too large for the git tree.

Files and their md5 (as recorded in data/wp5_bench_v1_manifest.json and the
companion reports):
  wp3_ddpm_v0_model.pt              FASE-DDPM v0 checkpoint
  wp3_ddpm_v0_dataset.pt            frozen 21,714-row dataset payload (seed 0 split)
  wp3_ddpm_v01_model.pt             FASE-DDPM v0.1 checkpoint
  wp3_ddpm_v01_dataset.pt           v0.1 dataset payload (full-population qt)
  wp5_bench_v1_tabddpm_model.pt     TabDDPM-MLP checkpoint
  wp5_g3_scorer.pkl                 condition-consistency scorer (198-class head)
  feasibility_scorer_v1_2_model.pkl feasibility prediction model bundle

Usage:
  python baselines/download_weights.py            # download what is missing
  python baselines/download_weights.py --check    # verify md5 only
"""
import argparse
import hashlib
import os
import sys
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WGT = os.path.join(BASE, 'weights')
URL = ('https://github.com/Vitalite123/zeolite-synthesis-bench/releases/'
       'download/weights-v1/')

FILES = {
    'wp3_ddpm_v0_model.pt': 'd9ed94ba909f7e415f6e05d66fcc6649',
    'wp3_ddpm_v0_dataset.pt': 'ab46eab3288711da80fe5a53a70a63b5',
    'wp3_ddpm_v01_model.pt': '31981bda24223dd3218520bd32ef5442',
    'wp3_ddpm_v01_dataset.pt': 'fe35a54b2bb2c004da7cd460cbba17ed',
    'wp5_bench_v1_tabddpm_model.pt': '89f5a6947335f5b069da414a963fcd93',
    'wp5_g3_scorer.pkl': 'd2142afea965fac1c930b32719924435',
    'feasibility_scorer_v1_2_model.pkl': 'fa0889f9db9118ef7d6b39448ac6f12a',
}


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='verify md5 of existing files only')
    args = ap.parse_args()

    os.makedirs(WGT, exist_ok=True)
    missing = [n for n in FILES if not os.path.exists(os.path.join(WGT, n))]

    if args.check:
        bad = 0
        for n, want in FILES.items():
            p = os.path.join(WGT, n)
            if not os.path.exists(p):
                print('MISSING   %s' % n); bad += 1
            elif want and md5(p) != want:
                print('MISMATCH  %s (md5 %s, expected %s)' % (n, md5(p), want)); bad += 1
            else:
                print('OK        %s' % n)
        return 1 if bad else 0

    if not missing:
        print('weights/ complete (%d files).' % len(FILES))
    for n in missing:
        dst = os.path.join(WGT, n)
        print('downloading %s ...' % n, flush=True)
        urllib.request.urlretrieve(URL + n, dst)
        print('  -> %d bytes, md5 %s' % (os.path.getsize(dst), md5(dst)))
    if not missing:
        return 0
    print('done. %d files in weights/.' % len(FILES))
    return 0


if __name__ == '__main__':
    sys.exit(main())
