PY ?= python

.PHONY: test audit figures tables all

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
