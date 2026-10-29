"""Paths and settings of the failure-type analysis, read from the environment so one script serves every K.

  FT_K             K of the explained runs (10, 20 or 30)                                     (required)
  FT_FOLD          fold the runs explained: test (default, the report fold) or validation
  FT_EXPLAIN_ROOT  folder holding top<K>/<method>_seed<N>/ explanation records              (required)
  FT_SUMMARY_DIR   output of `comparison.explain.summarize` for that folder (parity check)   (required)
  FT_RUNS          root of the trained runs, top<K>/<method>_seed<N>/ with binding.json and <fold>.npz
                   (default comparison/standardized/clinical_runs_fair_v10t_20261011)
  FT_WORK          working folder: data/ (per-visit tables with anonymous indices, never commit) and out/ (aggregates)
This package lives outside the hashed code roots (core, the method folders, comparison/<package>), so adding or
changing it does not change the code identity of the explanation runs.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _required(name):
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set (see analysis/failure_types/_paths.py)")
    return value


K = int(_required("FT_K"))
FOLD = os.environ.get("FT_FOLD", "test")
EXPLAIN_ROOT = Path(_required("FT_EXPLAIN_ROOT"))
SUMMARY_DIR = Path(_required("FT_SUMMARY_DIR"))
RUNS = Path(os.environ.get("FT_RUNS", str(REPO / "comparison/standardized/clinical_runs_fair_v10t_20261011")))
WORK = Path(_required("FT_WORK"))
