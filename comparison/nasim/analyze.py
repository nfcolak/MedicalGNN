"""Analysis of Section-2 experiments: arms vs a reference, fair_v10t runs (the shared code is comparison/ablation/arms_vs_ref.py).

    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 comparison/nasim/analyze.py \
        --ref final=<.../clinical_runs_fair_v10t_20261011/top20> --arm x=<dir> [--arm y=<dir>] \
        [--method gsat] --out <new dir>

Every folder holds `<method>_seed<s>/` run dirs (result.json, binding.json, validation.npz, test.npz). The
method key is the run-dir name without `_seed<s>` (so `gsat`, `cei_mp2`, `gchm_pna`). Read-only on the runs;
`--out` must not exist. Statistics, reading rule and tables: see comparison/ablation/arms_vs_ref.py (test first).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from comparison.ablation import arms_vs_ref  # noqa: E402


def main(argv=None):
    return arms_vs_ref.main(argv, title='Section-2 analysis')


if __name__ == '__main__':
    sys.exit(main())
