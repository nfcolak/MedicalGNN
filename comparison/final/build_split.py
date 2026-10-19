"""Rebuild the patient split comparison/canonical_split.json from its input table, by the original rule.

Port of the June-2026 generator (MedGNN commit 0a61bed1: comparison/build_split.py, shared/lib/splits.py
stratified_split_indices, graphcare_analysis/build_kg.py vocab_and_presence, med_min_prev 0.01), kept
line-for-line in behaviour so that the output is byte-identical to the frozen file:

  input   merged_ed.csv (one row per patient, output of the frozen raw preprocessing data/s1_clean/merge_ed.py,
          expected sha256 below)
  keep    rows with a disease_1 label and at least one code (medication class with prevalence >= 1%, symptom or
          chief complaint)
  classes sorted distinct disease_1 values (30)
  split   per class: numpy default_rng(1234) shuffle, round(0.8 n) train, round(0.1 n) validation, rest test
          (subjects are unique, so the subject-aware branch of the original code is never taken)

Modes (the frozen file is never written):
  --check     (default) rebuild in memory and compare with --split byte for byte; exit 0 only if identical
  --out PATH  also write the rebuilt split to PATH; refuses an existing path and the frozen file itself

Usage: python -m comparison.final.build_split [--merged-ed FILE] [--split FILE] [--out NEW_PATH]
Prints aggregated counts only (no subject ids).
"""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT = 'comparison/standardized/event_inputs/split_input_merged_ed/merged_ed.csv'
DEFAULT_SPLIT = 'comparison/canonical_split.json'
INPUT_SHA256 = '95b055d53c5f879e5a5784b9d44cd3e6f8b0a073798fadc6bb25e5a228eaaf8b'
SEED = 1234
SPLIT_RATIO = [0.8, 0.1, 0.1]
MED_MIN_PREV = 0.01


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _multihot_vocab(df, cols):
    """Unique non-empty cell values across cols, and their per-row presence."""
    counter = {}
    if not cols:
        return [], np.zeros((len(df), 0), dtype=bool)
    rows = df[cols].fillna('').astype(str).values.tolist()
    for row in rows:
        for s in row:
            s = s.strip()
            if s:
                counter[s] = counter.get(s, 0) + 1
    vocab = sorted(counter)
    index = {v: j for j, v in enumerate(vocab)}
    present = np.zeros((len(df), len(vocab)), dtype=bool)
    for i, row in enumerate(rows):
        for s in row:
            s = s.strip()
            if s in index:
                present[i, index[s]] = True
    return vocab, present


def code_counts(df):
    """Number of codes per row: medication classes with prevalence >= 1%, symptoms, chief complaints."""
    med_all = sorted(c for c in df.columns if c.startswith(('med_', 'pyx_')))
    med_bin = (df[med_all].apply(pd.to_numeric, errors='coerce').fillna(0).values > 0)
    med_present = med_bin[:, med_bin.mean(0) >= MED_MIN_PREV]
    _, sym_present = _multihot_vocab(df, sorted(c for c in df.columns if c.startswith('symptom_')))
    _, cc_present = _multihot_vocab(df, sorted(c for c in df.columns if c.startswith('chiefcomplaint_')))
    return np.concatenate([med_present, sym_present, cc_present], axis=1).astype(bool).sum(1)


def stratified_split_indices(labels, split_sizes, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    train, eval_, test = [], [], []
    for label in np.unique(labels):
        li = indices[labels == label]
        rng.shuffle(li)
        n = len(li)
        n_tr = int(round(split_sizes[0] * n))
        n_ev = int(round(split_sizes[1] * n))
        if n_tr + n_ev > n:
            n_ev = max(0, n - n_tr)
        train += li[:n_tr].tolist()
        eval_ += li[n_tr:n_tr + n_ev].tolist()
        test += li[n_tr + n_ev:].tolist()
    rng.shuffle(train); rng.shuffle(eval_); rng.shuffle(test)
    return train, eval_, test


def build(merged_ed):
    df = pd.read_csv(merged_ed, low_memory=False)
    n_codes = code_counts(df)
    dis = df['disease_1'].fillna('').astype(str)
    classes = sorted(d for d in dis.unique() if d.strip())
    c2i = {d: i for i, d in enumerate(classes)}
    y = dis.map(c2i).fillna(-1).astype(int).values
    subj = df['subject_id'].values
    keep = [i for i in range(len(df)) if y[i] >= 0 and n_codes[i] > 0]
    groups_k = subj[keep]
    if len(np.unique(groups_k)) < len(groups_k):
        raise SystemExit('REFUSED: repeated subject_id in the input; the frozen split was built from one row per '
                         'patient (the original subject-aware branch is not ported)')
    tr, va, te = stratified_split_indices(y[keep], SPLIT_RATIO, SEED)
    fold = {}
    for part, k in ((tr, 0), (va, 1), (te, 2)):
        for local in part:
            fold[str(int(subj[keep[local]]))] = k
    data = {'fold': fold, 'classes': classes, 'n_kept': len(keep),
            'split': 'stratified (subjects unique = single-visit)', 'seed': SEED, 'ratio': SPLIT_RATIO}
    buf = io.StringIO()
    json.dump(data, buf)
    return data, buf.getvalue().encode(), len(df), (len(tr), len(va), len(te))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='comparison.final.build_split')
    ap.add_argument('--merged-ed', default=DEFAULT_INPUT, help='input table (one row per patient), read only')
    ap.add_argument('--split', default=DEFAULT_SPLIT, help='frozen split to compare with (read only, never written)')
    ap.add_argument('--out', default=None, help='optional NEW path for the rebuilt split (never the frozen file)')
    a = ap.parse_args(argv)

    frozen = Path(a.split).resolve()
    if a.out is not None:
        out = Path(a.out).resolve()
        if out == frozen or out == Path(DEFAULT_SPLIT).resolve():
            raise SystemExit('REFUSED: --out points at the frozen split; it is never regenerated')
        if out.exists():
            raise SystemExit(f'REFUSED: {a.out} exists; write to a new path')

    digest = sha256(a.merged_ed)
    print(f'input: {a.merged_ed} sha256 {digest[:16]} '
          f"({'expected' if digest == INPUT_SHA256 else 'DIFFERENT from expected ' + INPUT_SHA256[:16]})")
    data, blob, n_rows, (n_tr, n_va, n_te) = build(a.merged_ed)
    print(f"kept {data['n_kept']} / {n_rows} patients, {len(data['classes'])} classes; "
          f'train={n_tr} validation={n_va} test={n_te}')

    ref = frozen.read_bytes()
    old = json.loads(ref)
    common = set(old['fold']).intersection(data['fold'])
    mismatch = sum(old['fold'][s] != data['fold'][s] for s in common)
    print(f"frozen: {a.split} sha256 {hashlib.sha256(ref).hexdigest()[:16]}; subjects frozen={len(old['fold'])} "
          f"rebuilt={len(data['fold'])} common={len(common)} fold mismatches={mismatch}; "
          f"classes {'same' if old['classes'] == data['classes'] else 'DIFFERENT'}")
    same = blob == ref
    print(f"rebuilt split is {'BYTE-IDENTICAL to' if same else 'DIFFERENT from'} the frozen file "
          f'(rebuilt sha256 {hashlib.sha256(blob).hexdigest()[:16]})')
    if a.out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'xb') as f:
            f.write(blob)
        print(f'wrote {a.out}')
    return 0 if same else 1


if __name__ == '__main__':
    sys.exit(main())
