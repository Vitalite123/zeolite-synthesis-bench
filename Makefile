PY ?= python

BASEURL := https://github.com/Vitalite123/zeolite-synthesis-bench/releases/download/weights-v1

.PHONY: test audit figures tables weights all

all: test audit figures tables

test:
	$(PY) -m pytest -q

audit:
	$(PY) splits/audit_splits.py

figures:
	$(PY) figures/p21_supp_figs.py
	$(PY) figures/p21_supp_figs.py --publication
	$(PY) figures/p21_supp_figs.py --toc

tables:
	$(PY) figures/p21_supp_t2.py

weights:
	$(PY) baselines/download_weights.py
