# Datasheet — ZeoliteSynthesisBench v1

## Identity

| Item | Value |
|---|---|
| Benchmark | ZeoliteSynthesisBench v1 |
| Data version | v1.8 (frozen) |
| Records | 24,706 hydrothermal routes x 149 fields |
| DOIs | 3,236 |
| Publication years | 1966–2026 |
| Split seed | 42 |
| Frozen files | `zeobench_db_v1_8.parquet`, `zeobench_splits_v1_8.parquet` |

The corpus assembles 23,961 source routes, four literature increments (428 rows), seven rolling
extraction batches (net 173 rows) and 144 upgrade rows. Increment rows are identified by a null
`src_row_id` (745 rows in total). Every route keeps its provenance chain (source table, DOI, batch,
extraction flags).

## Schema

Composition is stored as normalised element mole fractions that sum to one. Temperature is stored
in degrees Celsius and time in hours. Each OSDA carries 444 RDKit descriptors plus an availability
flag. The 149 fields fall into six groups: route identity, composition, ratios, conditions, OSDA
descriptors and product/label fields.

## Outcome labels

| Label | Definition | n |
|---|---|---:|
| `positive` | target framework is the only crystalline product | 15,187 |
| `relative_neg` | target framework plus one secondary crystalline phase | 4,478 |
| `absolute_neg` | amorphous, dense, layered or unknown product | 4,993 |
| `no_product` | no solid product reported | 48 |

The feasibility task uses `xtal` (`positive` ∪ `relative_neg`) as the primary label and `pure`
(`positive` only) as a reference. Negative labels are heterogeneous by construction: a failed
crystallisation may reflect an unrecorded variable rather than an infeasible recipe.

## Frozen splits

All splits are defined at DOI level for `random` and at `(framework, DOI)` group level for the two
hold-outs. Test counts are rows, not systems.

| Split | Rule | train | val | test | excluded |
|---|---|---:|---:|---:|---:|
| `split_random` | DOI 80/10/10 | 19,870 | 2,426 | 2,410 | 0 |
| `split_heldout` | LTL, SAS, ITG, RTH, SZR, LTA withheld | 21,616 | 0 | 3,067 | 23 |
| `split_time_2018` | trained on records published in 2018 or earlier | 22,067 | 0 | 2,379 | 260 |
| `split_time_2020` | trained on records published in 2020 or earlier | 23,662 | 0 | 784 | 260 |

Excluded rows are DOI conflicts (`heldout`) or rows with no recorded publication year
(`time_*`). The paper reports only `time_2018`; `split_time_2020` ships for completeness.

## Six leakage audits

`python splits/audit_splits.py` re-runs these from the two frozen parquet files.

| Audit | Expected |
|---|---|
| DOI appears in more than one split set | 0 |
| `(framework, DOI)` group appears in more than one split set | 0 |
| held-out frameworks present in training | 0 |
| increment DOI also present in the base corpus | 0 |
| time boundary monotonic (train ≤ cutoff < test) | holds for 2018 and 2020 |
| JZO rows on the time-test side | 32 of 32 |

## Derived subsets

| Subset | Definition | Size |
|---|---|---|
| Feasibility domain | 24,706 − 901 Tier-A duplicates | 24,113 (19,049 xtal-positive, 5,064 negative) |
| Feasibility features | log mole/ratio/condition columns + ratio and charge-balance terms + OSDA flag + 444 descriptors | 490 |
| DDPM subset | drops 48 `no_product`, unmapped or mixed framework codes, unmapped or unnamed OSDAs | 21,714 rows, system split seed 0 (16,058 / 2,294 / 3,362) |
| Track A | test-side systems with at least five real rows, excluding Dense and Amorphous | 123 systems × 200 samples; 2,087 real rows of 3,362 |
| Track B | systems in the released ZeoSyn split | 464 reported, 372 pass the released feature filter (370 for AMD) |

Of the 901 removed Tier-A duplicates, 616 are xtal-positive and 285 xtal-negative.

## Known biases and limitations

- The `random` test split contains only 443 xtal-negative rows, so negative-class power is limited.
- Literature coverage is skewed towards open-access sources.
- Most routes are unseeded syntheses; unrecorded seeding is treated as unseeded.
- Label noise is one-sided: failed routes are rarely reported, so negatives under-represent genuine
  infeasibility.
- The benchmark evaluates offline only; no synthesis is performed here.
- Single-seed evaluation is part of the frozen protocol. The 10-seed subsample check (sd 0.86 pp)
  bounds evaluation-side noise; it is not a seed-robustness claim.

## Physical validity window

`PV` counts a generated recipe as valid when all values are finite, ratio columns are non-negative,
temperature lies in [20, 350] °C and time lies in (0, 20,000] h.

## Licence

Data are released under CC BY 4.0 (`LICENSE-DATA`). Source code is MIT (`LICENSE`). The corpus is a
derived annotation of published hydrothermal routes; consult the original publications for the
underlying experimental records.
