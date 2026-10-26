"""Split check: verify the frozen comparison/canonical_split.json without ever regenerating or writing it.

Origin (found by reading the old MedGNN history, read-only): derived from MedGNN commit 0a61bed1 (identical tree in
21904b15, 2026-06-29) comparison/build_split.py, which used shared/lib/splits.py stratified_split_indices
(numpy default_rng(1234), per-class shuffle, round(0.8 n) train / round(0.1 n) validation / rest test) on the
single-visit rows of merged_ed.csv with a disease_1 label and at least one code (med/symptom/cc), subjects unique.

Re-derivation: done by comparison/final/build_split.py, which ports that rule and rebuilds the file
byte-identically from merged_ed.csv (the one-row-per-patient output of the frozen raw preprocessing
data/s1_clean/merge_ed.py). This script needs no input table; it checks the frozen file's internal consistency and
its agreement with the final targets (read-only):
  1. file structure: keys, split rule text, seed, ratio, 30 sorted unique classes, n_kept == number of subjects
  2. no subject in two folds (duplicate JSON keys are detected), folds only 0/1/2, one fold entry per subject
  3. fold sizes against the 80/10/10 ratio
  4. targets.csv: every subject has a single split name, equal to the frozen fold (train=0, validation=1, test=2)
  5. stratification: per-class fold shares of labelled subjects, class labels of the targets against the 30 classes

Usage: python -m comparison.final.split_check [--split comparison/canonical_split.json] [--targets FILE]
Prints aggregated counts only (no subject ids). Opens every input read-only and writes nothing.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_SPLIT = 'comparison/canonical_split.json'
DEFAULT_TARGETS = 'comparison/standardized/event_inputs/targets_primary_v4_v5plus60_v5_test_20261011/targets.csv'
KEYS = {'fold', 'classes', 'n_kept', 'split', 'seed', 'ratio'}
FOLD_NAME = {'train': 0, 'validation': 1, 'test': 2}
RATIO_TOL = 0.001
RESULTS = []


def check(name, ok, detail):
    RESULTS.append(ok)
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: {detail}")


def load_split(path):
    dup = []

    def hook(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            dup.append(len(keys) - len(set(keys)))
        return dict(pairs)

    with open(path, 'r') as f:
        data = json.load(f, object_pairs_hook=hook)
    return data, sum(dup)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='comparison.final.split_check')
    ap.add_argument('--split', default=DEFAULT_SPLIT, help='frozen split file (read only, never rewritten)')
    ap.add_argument('--targets', default=DEFAULT_TARGETS,
                    help='targets.csv with subject_id, split, label (read only); "none" skips the cross-check')
    a = ap.parse_args(argv)

    data, dup_keys = load_split(a.split)
    fold, classes, ratio = data['fold'], data['classes'], data['ratio']
    n = len(fold)
    print(f'split file: {a.split}')
    print('origin: derived from MedGNN commit 0a61bed1 comparison/build_split.py (+ shared/lib/splits.py stratified_split_indices)')
    print('re-derivation: comparison.final.build_split (byte-identical rebuild from merged_ed.csv); this script checks consistency')

    print('structure')
    check('keys', set(data) == KEYS, sorted(data))
    check('split rule text', data['split'] == 'stratified (subjects unique = single-visit)', data['split'])
    check('seed', data['seed'] == 1234, data['seed'])
    check('ratio', len(ratio) == 3 and abs(sum(ratio) - 1) < 1e-9 and ratio == [0.8, 0.1, 0.1], ratio)
    check('classes', len(classes) == 30 and len(set(classes)) == 30 and classes == sorted(classes),
          f'{len(classes)} unique, sorted={classes == sorted(classes)}')
    check('n_kept == subjects', data['n_kept'] == n, f"n_kept={data['n_kept']} subjects={n}")
    check('subjects unique (single-visit)', dup_keys == 0 and all(k.isdigit() for k in fold),
          f'duplicate keys={dup_keys}, subject entries={n}')

    print('folds')
    counts = Counter(fold.values())
    check('fold values', set(counts) <= {0, 1, 2}, dict(sorted(counts.items())))
    for k, name in enumerate(('train', 'validation', 'test')):
        share = counts.get(k, 0) / n
        check(f'fold {k} ({name}) share', abs(share - ratio[k]) <= RATIO_TOL,
              f'{counts.get(k, 0)}/{n} = {share:.4f} vs ratio {ratio[k]}')

    if a.targets.lower() != 'none':
        print(f'targets cross-check: {a.targets}')
        sub_splits, sub_labels = defaultdict(set), defaultdict(set)
        rows = 0
        with open(a.targets, 'r', newline='') as f:
            for r in csv.DictReader(f):
                rows += 1
                sub_splits[r['subject_id']].add(r['split'])
                if r['label']:
                    sub_labels[r['subject_id']].add(r['label'])
        in_split = [s for s in sub_splits if s in fold]
        multi = sum(len(sub_splits[s]) > 1 for s in sub_splits)
        match = sum(1 for s in in_split if len(sub_splits[s]) == 1 and FOLD_NAME[next(iter(sub_splits[s]))] == fold[s])
        print(f'  targets rows={rows}, subjects={len(sub_splits)}, subjects also in frozen file={len(in_split)}/{n}')
        check('one split per subject in targets', multi == 0, f'{multi} subjects with more than one split')
        check('fold equals targets split', match == len(in_split),
              f'matched {match}/{len(in_split)} subjects; by fold '
              f'{dict(sorted(Counter(fold[s] for s in in_split).items()))}')
        labels = set().union(*sub_labels.values()) if sub_labels else set()
        print(f'  labelled subjects={len(sub_labels)}, distinct labels={len(labels)}')
        check('target labels within frozen classes', labels <= set(classes) or len(labels) == len(classes),
              f'{len(labels)} labels vs {len(classes)} classes')
        per_class = defaultdict(Counter)
        for s, ls in sub_labels.items():
            if s in fold:
                for lab in ls:
                    per_class[lab][fold[s]] += 1
        if per_class:
            tr = [c[0] / sum(c.values()) for c in per_class.values()]
            te = [c[2] / sum(c.values()) for c in per_class.values()]
            sizes = [sum(c.values()) for c in per_class.values()]
            print(f'  per-class (labelled subjects, {len(per_class)} classes, class size {min(sizes)}..{max(sizes)}): '
                  f'train share {min(tr):.3f}..{max(tr):.3f}, test share {min(te):.3f}..{max(te):.3f}')
            big = [c for c in per_class.values() if sum(c.values()) >= 500]
            check('big classes stratified (>=500 subjects)',
                  all(abs(c[0] / sum(c.values()) - ratio[0]) < 0.03 for c in big),
                  f'{len(big)} classes, max train-share deviation '
                  f'{max((abs(c[0] / sum(c.values()) - ratio[0]) for c in big), default=0):.3f}')

    ok = all(RESULTS)
    print(f'checks passed {sum(RESULTS)}/{len(RESULTS)}')
    print('conclusion: consistency checked' if ok else 'conclusion: consistency checks FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
