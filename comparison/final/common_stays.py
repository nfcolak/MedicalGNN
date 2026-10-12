"""Common stays of the 4 fair_v3 cohorts INCLUDING test rows (the fair_v10t test-label restriction; same intersection
rule as the old common_stays_v5 file, which skipped test). Checks that the train+validation part equals the old file.

Usage: python3 -m comparison.final.common_stays OUT_DIR [--v5-root EVENT_INPUTS_DIR]
The two old cohorts are read from the directory in the env var MEDGNN_OLD_EVENT_INPUTS; the two v5 cohorts and the old
common-stay file are read from --v5-root (default: the env var MEDGNN_V5_EVENT_INPUTS).
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

OLD_ROOT = Path(os.environ['MEDGNN_OLD_EVENT_INPUTS']) if os.environ.get('MEDGNN_OLD_EVENT_INPUTS') else None
V5_ROOT = Path(os.environ['MEDGNN_V5_EVENT_INPUTS']) if os.environ.get('MEDGNN_V5_EVENT_INPUTS') else None
NAMES = ['old_firstlab', 'old_plus60', 'v5_firstlab', 'v5_plus60']


def main(argv=None):
    ap = argparse.ArgumentParser(prog='comparison.final.common_stays')
    ap.add_argument('output')
    ap.add_argument('--v5-root', default=str(V5_ROOT) if V5_ROOT else None)
    a = ap.parse_args(argv)
    if OLD_ROOT is None:
        sys.exit('set MEDGNN_OLD_EVENT_INPUTS to the directory holding the two old event_inputs cohorts')
    if not a.v5_root:
        sys.exit('set MEDGNN_V5_EVENT_INPUTS or pass --v5-root with the directory holding the two v5 event_inputs cohorts')
    v5 = Path(a.v5_root)
    out = Path(a.output)
    paths = [OLD_ROOT / 'first_recorded_lab_all_visits_v2_max6/cohort.csv',
             OLD_ROOT / 'first_recorded_lab_plus60_max6_20261005/cohort.csv',
             v5 / 'first_recorded_lab_all_visits_max6_v5_20261009/cohort.csv',
             v5 / 'first_recorded_lab_plus60_max6_v5_20261009/cohort.csv']
    cohorts = {}
    for name, path in zip(NAMES, paths):
        with open(path, newline='') as f:
            cohorts[name] = {r['stay_id']: (r['sample_id'], r['subject_id'], r['split']) for r in csv.DictReader(f)}
    common = set.intersection(*[set(r) for r in cohorts.values()])
    ref = cohorts[NAMES[0]]
    for name in NAMES[1:]:
        bad = [s for s in common if cohorts[name][s] != ref[s]]
        if bad:
            sys.exit(f'{name}: sample_id/subject/split differ for {len(bad)} common stays')
    old_file = v5 / 'common_stays_v5_20261009/stays.txt'
    if old_file.exists():
        old = set(old_file.read_text().split())
        trval = {s for s in common if ref[s][2] != 'test'}
        if trval != old:
            sys.exit(f'train+validation common stays differ from {old_file}: +{len(trval - old)} -{len(old - trval)}')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'stays.txt').write_text(''.join(s + '\n' for s in sorted(common)))
    report = {'scope': 'train, validation AND test rows (test opened, design decision 2026-10-11)',
              'common_stays': len(common), 'train_validation_equal_old_file': old_file.exists(),
              'common_by_split': {sp: sum(1 for s in common if ref[s][2] == sp) for sp in ('train', 'validation', 'test')},
              'cohorts': {n: {'path': str(p), 'by_split': {sp: sum(1 for v in cohorts[n].values() if v[2] == sp)
                                                           for sp in ('train', 'validation', 'test')}}
                          for n, p in zip(NAMES, paths)}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
