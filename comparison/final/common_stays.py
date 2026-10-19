"""Common stays of the 3 cohorts INCLUDING test rows (the fair_v10t test-label restriction).

The legacy plus-60 cohort has exactly the same stays as the legacy max-6 cohort, so the intersection of the
legacy max-6, v5 max-6 and v5 plus-60 cohorts equals the intersection of all 4 cohorts used before.
Test rows are kept so that every reported result can be stated on test; decisions (checkpoint, HPO, baseline choice) use validation.

Usage: python3 -m comparison.final.common_stays OUT_DIR --legacy-max6 DIR --v5-max6 DIR --v5-plus60 DIR
                                                [--check-against FILE]
Each DIR is an event-input cohort directory holding cohort.csv. Writes OUT_DIR/stays.txt and OUT_DIR/report.json.
--check-against FILE: optional stays file; its stay ids must equal the train+validation part of the common stays.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

NAMES = ['legacy_max6', 'v5_max6', 'v5_plus60']
SPLITS = ('train', 'validation', 'test')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='comparison.final.common_stays')
    ap.add_argument('output')
    ap.add_argument('--legacy-max6', required=True, help='legacy cohort (first_lab all-numeric-known-unit, max 6 visits)')
    ap.add_argument('--v5-max6', required=True, help='v5 cohort (representable-v5, max 6 visits)')
    ap.add_argument('--v5-plus60', required=True, help='v5 +60 min cohort')
    ap.add_argument('--check-against', default=None, help='optional stays file to compare with the train+validation part')
    a = ap.parse_args(argv)
    out = Path(a.output)
    paths = [Path(a.legacy_max6) / 'cohort.csv', Path(a.v5_max6) / 'cohort.csv', Path(a.v5_plus60) / 'cohort.csv']
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
    checked = None
    if a.check_against:
        old = set(Path(a.check_against).read_text().split())
        trval = {s for s in common if ref[s][2] != 'test'}
        if trval != old:
            sys.exit(f'train+validation common stays differ from {a.check_against}: +{len(trval - old)} -{len(old - trval)}')
        checked = True
    out.mkdir(parents=True, exist_ok=False)
    (out / 'stays.txt').write_text(''.join(s + '\n' for s in sorted(common)))
    report = {'scope': 'train, validation AND test rows',
              'common_stays': len(common), 'train_validation_equal_check_file': checked,
              'common_by_split': {sp: sum(1 for s in common if ref[s][2] == sp) for sp in SPLITS},
              'cohorts': {n: {'path': str(p), 'by_split': {sp: sum(1 for v in cohorts[n].values() if v[2] == sp)
                                                           for sp in SPLITS}}
                          for n, p in zip(NAMES, paths)}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
