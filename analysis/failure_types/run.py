"""Run the failure-type analysis (step 1 then the stratified report) for one or more K.

    PYTHONPATH=. python -m analysis.failure_types.run --explanations <root with top<K>/...> \
        --summary <output of comparison.explain.summarize for that root> --runs <trained runs root> \
        --work <new dir> [--ks 10,20,30] [--fold test]

Each K gets its own working folder `<work>/k<K>` (per-visit tables in `data/`, never commit; aggregates in `out/`).
Settings reach the two steps through the FT_* environment (see `_paths.py`).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STEPS = ("analysis.failure_types.step1_visit_tables", "analysis.failure_types.strata_report")
DEFAULT_RUNS = "comparison/standardized/clinical_runs_fair_v10t_20261011"


def plan(args) -> list[tuple[dict, list[str]]]:
    """[(environment, command)] for every K and step."""
    steps = []
    for k in args.ks:
        env = {"FT_K": str(k), "FT_FOLD": args.fold, "FT_EXPLAIN_ROOT": str(args.explanations),
               "FT_SUMMARY_DIR": str(args.summary), "FT_RUNS": str(args.runs), "FT_WORK": str(Path(args.work) / f"k{k}")}
        for module in STEPS:
            steps.append((env, [sys.executable, "-m", module]))
    return steps


def main(argv=None, run=subprocess.run) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--explanations", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--runs", default=DEFAULT_RUNS, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--ks", default="10,20,30")
    parser.add_argument("--fold", default="test", choices=("test", "validation"))
    args = parser.parse_args(argv)
    args.ks = [int(k) for k in args.ks.split(",")]
    if Path(args.work).exists() and any(Path(args.work).iterdir()):
        raise SystemExit(f"--work must be new or empty: {args.work}")
    for env, command in plan(args):
        print(f"K={env['FT_K']}: {' '.join(command[1:])}", flush=True)
        result = run(command, env={**os.environ, **env, "PYTHONPATH": str(REPO)}, cwd=REPO)
        if result.returncode != 0:
            print(f"failed with exit status {result.returncode}", file=sys.stderr)
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
