"""Regenerate the three final comparison tables (K = 10, 20, 30) from the stored fair_v10t runs, optionally proving
byte-identity with the stored outputs. Read-only on the runs; writes only into the new `--out` folder.

Usage (from the repo root):
    python3 -m comparison.final.compare_all --runs comparison/standardized/clinical_runs_fair_v10t_20261011 --out <new dir>
    python3 -m comparison.final.compare_all --check --out <new dir>
    python3 -m comparison.final.compare_all --check <stored dir> --out <new dir>

Calls `comparison.top3.compare.main` in-process once per K, with the runs passed in the recorded order (method order, then
seeds 1234, 2025, 7). Output layout: `<out>/compare_top<K>.txt` (the table printed by compare.py) and
`<out>/compare_top<K>.json/comparison.json`. Since 2026-10-19 the tables list the methods by TEST macro-F1 (reported) with
the validation rank (decision) as a labelled column; the stored copy of this layout is
`comparison/standardized/final_compare_fv10t_test_20261019`. The folder `final_compare_fv10t_20261012` keeps the
pre-2026-10-19 layout (methods in validation order) and is never edited; the same numbers, a different order.

`--methods` default = cei_mp2,protgnn,gchm_pna,gsat,graphcare. This default exists ONLY to regenerate the published
tables byte-for-byte (they were made in that order and include GCHM-PNA); it is not a method choice for new analyses.
For anything new pass `--methods` without gchm_pna.

`--check [DIR]`: every produced file is compared byte-for-byte with DIR/<same name> (default
`comparison/standardized/final_compare_fv10t_test_20261019`); differing or missing files are listed and the exit code is 1. One allowance: a stored .txt was captured with stderr merged, so it may begin with the
interpreter's warning lines (a local library path, not part of the table); those leading lines, up to the first line of
the table ("Ranking by ..."), are ignored for the .txt comparison and reported as such.
"""
import argparse
import contextlib
import io
import sys
from pathlib import Path

from comparison.top3 import compare

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = 'comparison/standardized/clinical_runs_fair_v10t_20261011'
RECORDED_METHODS = 'cei_mp2,protgnn,gchm_pna,gsat,graphcare'   # published order (includes GCHM-PNA); parity only
SEEDS = (1234, 2025, 7)
DEFAULT_CHECK = 'comparison/standardized/final_compare_fv10t_test_20261019'
TABLE_STARTS = (b'Test ranking', b'Ranking by ')


def rel(p):
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return p.name


def strip_warning_prefix(data):
    hits = [i for i in (data.find(t) for t in TABLE_STARTS) if i >= 0]
    i = min(hits) if hits else 0
    return data[i:] if i > 0 else data


def main(argv=None):
    ap = argparse.ArgumentParser(prog='comparison.final.compare_all', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default=DEFAULT_RUNS, help=f'folder holding top<K>/<method>_seed<S> (default {DEFAULT_RUNS})')
    ap.add_argument('--out', required=True, help='NEW folder for the regenerated tables (refused if it exists)')
    ap.add_argument('--ks', default='10,20,30', help='comma-separated K (default 10,20,30)')
    ap.add_argument('--methods', default=RECORDED_METHODS,
                    help=f'run order = recorded published order, only to regenerate the published tables byte-for-byte '
                         f'(default {RECORDED_METHODS})')
    ap.add_argument('--check', metavar='STORED_DIR', nargs='?', const=DEFAULT_CHECK,
                    help=f'stored compare folder; compare every produced file byte-for-byte (default {DEFAULT_CHECK})')
    a = ap.parse_args(argv)

    out, runs = Path(a.out), Path(a.runs)
    if out.exists():
        print(f'REFUSED: --out already exists: {rel(out)}', file=sys.stderr)
        return 2
    ks = [int(k) for k in a.ks.split(',')]
    methods = [m for m in a.methods.split(',') if m]
    out.mkdir(parents=True)
    produced = []
    for K in ks:
        run_args = []
        for m in methods:
            for s in SEEDS:
                d = runs / f'top{K}' / f'{m}_seed{s}'
                if not d.is_dir():
                    print(f'REFUSED: missing run {rel(d)}', file=sys.stderr)
                    return 2
                run_args += ['--run', str(d)]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = compare.main(run_args + ['--out', str(out / f'compare_top{K}.json')])
        if rc != 0:
            print(f'compare.py returned {rc} for K={K}', file=sys.stderr)
            return rc
        (out / f'compare_top{K}.txt').write_bytes(buf.getvalue().encode('utf-8'))
        produced += [f'compare_top{K}.txt', f'compare_top{K}.json/comparison.json']
        print(f'K={K}: {len(methods)} methods x {len(SEEDS)} seeds -> {rel(out)}/compare_top{K}.txt, compare_top{K}.json/')

    if not a.check:
        return 0
    stored = Path(a.check)
    bad, same = [], 0
    for name in produced:
        mine, theirs = (out / name).read_bytes(), stored / name
        note = ''
        if not theirs.is_file():
            bad.append(f'{name}: missing in stored folder')
            continue
        other = theirs.read_bytes()
        if name.endswith('.txt') and other != mine:
            stripped = strip_warning_prefix(other)
            if stripped == mine:
                other, note = stripped, ' (stored interpreter-warning prefix ignored)'
        if other == mine:
            same += 1
            print(f'identical  {name}{note}')
        else:
            bad.append(f'{name}: differs')
            print(f'DIFFERENT  {name}')
    print(f'byte-identical: {same}/{len(produced)} files against {rel(stored)}')
    for b in bad:
        print('  ' + b)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
