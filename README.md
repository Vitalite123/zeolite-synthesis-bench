# ZeoliteSynthesisBench v1

A leakage-audited benchmark for zeolite synthesis recipe generation and feasibility prediction.

> **Status.** The companion manuscript is under double-anonymous review. Keep this repository
> private until acceptance, or serve an anonymised copy (for example via anonymous.4open.science).
> Author names and affiliations are intentionally absent.

## What this is

24,706 hydrothermal synthesis records (3,236 DOIs, 1966–2026) with four outcome labels, three frozen
splits that pass six leakage audits, two tasks and seven metric families. Track A (123 systems × 200
samples) is the canonical leaderboard. Track B (464 systems from the released ZeoSyn split) is
reported separately because merging the two splits would leak training data into the test side.

Full counts, audit expectations and known biases are in [data/DATASHEET.md](data/DATASHEET.md).

## Layout

```
data/            frozen corpus, frozen splits, headline readouts, manifest, datasheet
metrics/         metric API, scorer adapter, feasibility scorer
splits/          split builder (provenance) and the six-audit checker
baselines/       Track A baseline readout and official-baseline adapter
figures/         figure scripts and rendered assets (160 and 300 dpi)
docs/            manuscript, Supporting Information, bibliography
tests/           data contract, manifest and split-audit tests
```

`data/` holds everything needed to verify the reported numbers and regenerate the figures.
Training checkpoints and intermediate prediction files are not included.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python -m pytest -q             # data contract + manifest + audit tests
python splits/audit_splits.py   # six leakage audits, prints PASS/FAIL per audit
python figures/p21_supp_figs.py # regenerate F1–F4 at 160 dpi
python figures/p21_supp_figs.py --publication
python figures/p21_supp_figs.py --toc
python figures/p21_supp_t2.py   # Table S2 and manifest refresh
```

`make test`, `make audit`, `make figures` wrap the same commands.

## What reproduces here

| Artefact | Source | Needs |
|---|---|---|
| Data contract (24,706 × 149, label and split counts) | `tests/test_data_contract.py` | pandas, pyarrow |
| Six leakage audits | `splits/audit_splits.py` | pandas, pyarrow |
| Figure md5 / size manifest | `tests/test_manifest.py`, `data/p21_supp_manifest.json` | hashlib, Pillow |
| Figures F1–F4 and TOC graphic | `figures/p21_supp_figs.py` | matplotlib, pandas, Pillow |
| Table S2 | `figures/p21_supp_t2.py` | pandas |
| Track B baseline table | `figures/p21_trackb_table.py` | `vendor/zeosyn_gen` checkout |
| Metric API and baseline readout | `metrics/`, `baselines/` | `vendor/zeosyn_gen` checkout, trained checkpoints |

The metric API wraps the official implementations shipped with ZeoSynGen for W2, MMD and COV, so
those three numbers are computed by the upstream code rather than reimplemented here. To run the
API-level code, clone the ZeoSynGen repository into `vendor/zeosyn_gen/` and download its released
dataset pickle and run directories. The scripts document their expected paths in their headers.

## Headline readouts

Track A, 123 systems × 200 samples per system, 12 metrics. See `data/wp5_bench_v1_table.csv` for the
full table.

| Method | COV-P | CC@0.5 | PV (global) |
|---|---:|---:|---:|
| FASE-DDPM v0 | 0.7359 | 0.1532 | 0.9867 |
| FASE-DDPM v0.5 | 0.7317 | 0.1672 | 0.9915 |
| TabDDPM-MLP | 0.6518 | 0.0646 | 0.9667 |
| KDE-QT | 0.5872 | 0.0559 | 0.9469 |
| Random-Uniform | 0.0451 | 0.0127 | 0.5507 |
| real test-side recipes (n = 2,087) | — | 0.8730 | — |

Condition consistency for real recipes averages 0.8337 with top-1 0.9109. Every number above is
reproduced in `data/wp5_bench_v1_table.csv` and `data/wp5_g3_formal_report.txt`; the manuscript
quotes them bit-for-bit.

## Data flow

```
24,706 routes ──┬─ minus 48 no_product, unmapped framework codes, unnamed OSDAs
                │     └─ 21,714 rows → DDPM training/validation/test (16,058 / 2,294 / 3,362, seed 0)
                ├─ minus 901 Tier-A duplicates
                │     └─ 24,113 feasibility rows (19,049 xtal-positive / 5,064 negative)
                └─ three frozen splits (seed 42)
                      random 19,870 / 2,426 / 2,410
                      heldout 21,616 / 0 / 3,067 (+23 excluded)
                      time_2018 22,067 / 0 / 2,379 (+260 excluded)
                            └─ Track A: 123 test-side systems, 2,087 real rows, 200 samples each
```

## Citation

See `CITATION.cff`. Until the manuscript is published, cite the paper title and the repository URL.

## Licence

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`LICENSE-DATA`). The corpus is a derived annotation of
published hydrothermal routes; consult the original articles for the underlying experiments.

## Contact

Correspondence is withheld during double-anonymous review. Open an issue in this repository instead.
