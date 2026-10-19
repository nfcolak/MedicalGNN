"""Generic analysis of arms vs a reference, fair_v10t runs (taken unchanged from the section-2 analysis, nasim_fixes ed7f97d).

    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 comparison/nasim/analyze.py \
        --ref final=<.../clinical_runs_fair_v10t_20261011/top20> --arm x=<dir> [--arm y=<dir>] \
        [--method gsat] --out <new dir>

Every folder holds `<method>_seed<s>/` run dirs (result.json, binding.json, validation.npz, test.npz). The
method key is the run-dir name without `_seed<s>` (so `gsat`, `cei_mp2`, `gchm_pna`); the binding method is
recorded alongside. Read-only on the runs; `--out` must not exist.

Decisions use the selected VALIDATION fold only. The test fold (scored once per run at the selected
checkpoint) is shown as "reported only, not used for decisions".

Per run: selected validation macro-F1/accuracy (result.json, recomputed from validation.npz with the repo
definition `f1_score(labels=arange(K), zero_division=0)` and asserted equal to 1e-6), selected epoch, test
macro-F1/accuracy. Per arm and method: mean +- SD (ddof=1) over seeds. Arm vs reference over the matched
seeds: per-seed deltas and a two-level bootstrap of the macro-F1 difference (arm - reference):
  level 1 resamples the matched seeds with replacement; level 2 resamples patients (npz `subjects`, all their
  rows with multiplicity; rows paired between arm and reference by `sample_ids`); one patient resample per
  replicate is shared by all seeds drawn and by arm and reference (the evaluation fold is one fixed sample).
  Statistic = mean over the drawn seeds of F1(arm) - F1(reference) on the resampled patients. B = 2000,
  fixed rng. With one matched seed level 1 is trivial and this is the plain patient bootstrap against the
  reference run of that seed. A 'patients only' interval (seeds fixed at the matched seeds) is given too.
With 3 seeds the seed level is a weak estimate of seed variance: read it with the per-seed deltas.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from core.contracts import sample_ids_sha256  # noqa: E402

PROTOCOL = 'fair_v10t'
B = 2000
SEED_PATIENTS = {'validation': 20261012, 'test': 20261013}
SEED_SEEDS = 20261014
TOL = 1e-6
RUN_RE = re.compile(r'^(?P<method>.+)_seed(?P<seed>\d+)$')
SAME_KEYS = ('protocol', 'top_k_labels', 'num_classes', 'split_sample_ids_sha256', 'kept_label_indices',
             'artifact_graphs_sha256', 'targets_sha256')
TEST_NOTE = 'reported only, not used for decisions'


class Refusal(Exception):
    pass


def macro_f1_from_conf(conf):
    tp = np.diagonal(conf, axis1=-2, axis2=-1).astype(np.float64)
    fp = conf.sum(axis=-2) - tp
    fn = conf.sum(axis=-1) - tp
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
    return f1.mean(axis=-1)


def repo_metrics(proba, y):
    """Macro-F1 and accuracy with the repo definition (core/train.py metrics)."""
    from sklearn.metrics import accuracy_score, f1_score
    K = proba.shape[1]
    pred = proba.argmax(1)
    return (float(f1_score(y, pred, labels=np.arange(K), average='macro', zero_division=0)),
            float(accuracy_score(y, pred)))


def load_fold(path, name):
    z = np.load(Path(path) / name)
    return {'proba': z['proba'].astype(np.float64), 'y': z['y'].astype(np.int64),
            'subjects': z['subjects'].astype(str), 'sample_ids': z['sample_ids'].astype(str)}


def load_run(path, method, seed):
    path = Path(path)
    result = json.loads((path / 'result.json').read_text())
    binding = json.loads((path / 'binding.json').read_text())
    if result.get('status') != 'completed':
        raise Refusal(f"{path}: status {result.get('status')!r}")
    if binding.get('protocol') != PROTOCOL:
        raise Refusal(f"{path}: protocol {binding.get('protocol')!r}, not {PROTOCOL!r}")
    if binding.get('protocol_overrides') != []:
        raise Refusal(f"{path}: protocol_overrides {binding.get('protocol_overrides')!r}, not []")
    if result.get('selection_fold') != 'validation':
        raise Refusal(f"{path}: selection_fold {result.get('selection_fold')!r}, not 'validation'")
    if binding.get('seed') not in (None, seed):
        raise Refusal(f"{path}: binding seed {binding.get('seed')!r} != dir seed {seed}")
    folds = {'validation': load_fold(path, 'validation.npz')}
    ids_hash = (binding.get('split_sample_ids_sha256') or {})
    for fold in ('validation', 'test'):
        if fold == 'test':
            if not (path / 'test.npz').is_file():
                continue
            if result.get('test_evaluations') != 1:
                raise Refusal(f"{path}: test_evaluations {result.get('test_evaluations')!r}, not 1")
            folds['test'] = load_fold(path, 'test.npz')
        f = folds[fold]
        K = f['proba'].shape[1]
        if len(f['y']) and (f['y'].min() < 0 or f['y'].max() >= K):
            raise Refusal(f'{path}: {fold} y outside [0, {K})')
        if len(set(f['sample_ids'])) != len(f['sample_ids']):
            raise Refusal(f'{path}: duplicate {fold} sample_ids')
        if ids_hash.get(fold) and sample_ids_sha256(f['sample_ids'].tolist()) != ids_hash[fold]:
            raise Refusal(f'{path}: {fold} sample_ids differ from binding split_sample_ids_sha256')
    if len({f['proba'].shape[1] for f in folds.values()}) != 1:
        raise Refusal(f'{path}: validation and test class counts differ')
    # recomputation of every stored number this script reports
    row = {'path': str(path), 'method': method, 'seed': seed, 'binding_method': binding.get('method'),
           'binding_conv': binding.get('conv'), 'selected_epoch': result.get('selected_epoch'),
           'best_epoch_is_last': result.get('best_epoch_is_last'), 'K': int(folds['validation']['proba'].shape[1])}
    v_f1, v_acc = repo_metrics(folds['validation']['proba'], folds['validation']['y'])
    stored = result.get('metrics') or {}
    for name, got in (('macro_f1', v_f1), ('accuracy', v_acc)):
        if not isinstance(stored.get(name), (int, float)) or abs(got - stored[name]) > TOL:
            raise Refusal(f"{path}: recomputed validation {name} {got:.6f} != result.json {stored.get(name)!r}")
    row.update(val_macro_f1=v_f1, val_accuracy=v_acc)
    if 'test' in folds:
        t_f1, t_acc = repo_metrics(folds['test']['proba'], folds['test']['y'])
        stored_t = result.get('test') or {}
        for name, got in (('macro_f1', t_f1), ('accuracy', t_acc)):
            if not isinstance(stored_t.get(name), (int, float)) or abs(got - stored_t[name]) > TOL:
                raise Refusal(f"{path}: recomputed test {name} {got:.6f} != result.json {stored_t.get(name)!r}")
        row.update(test_macro_f1=t_f1, test_accuracy=t_acc)
    return {'row': row, 'binding': binding, 'folds': folds}


def discover(root, methods):
    """{(method, seed): dir} of completed-looking run dirs, plus names skipped for lack of result.json."""
    root = Path(root)
    if not root.is_dir():
        raise Refusal(f'not a directory: {root}')
    found, skipped = {}, []
    for d in sorted(root.iterdir()):
        m = RUN_RE.match(d.name)
        if not (d.is_dir() and m):
            continue
        if methods and m['method'] not in methods:
            continue
        if not (d / 'result.json').is_file():
            skipped.append(d.name)
            continue
        found[(m['method'], int(m['seed']))] = d
    return found, skipped


def mean_sd(values):
    v = np.asarray(values, dtype=float)
    return {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)) if len(v) > 1 else None, 'n': int(len(v))}


def fmt(ms, nd=4):
    if ms is None:
        return 'n/a'
    return f"{ms['mean']:.{nd}f}" + (f" ± {ms['sd']:.{nd}f}" if ms['sd'] is not None else '')


def bootstrap_f1(runs_by_key, fold):
    """{key: array[B]} macro-F1 of each run on B shared patient resamples of the fold (rows aligned to the first)."""
    keys = list(runs_by_key)
    first = runs_by_key[keys[0]]['folds'][fold]
    order = {s: i for i, s in enumerate(first['sample_ids'])}
    aligned = {}
    for k in keys:
        f = runs_by_key[k]['folds'][fold]
        if set(f['sample_ids']) != set(order):
            raise Refusal(f'{fold} sample_id sets differ between runs ({k})')
        idx = np.argsort([order[s] for s in f['sample_ids']])
        a = {n: f[n][idx] for n in ('proba', 'y', 'subjects', 'sample_ids')}
        if not (np.array_equal(a['y'], first['y']) and np.array_equal(a['subjects'], first['subjects'])):
            raise Refusal(f'{fold} y/subjects disagree after alignment by sample_id ({k})')
        aligned[k] = a
    y = first['y']
    K = first['proba'].shape[1]
    subj, inv = np.unique(first['subjects'], return_inverse=True)
    cell = {k: y * K + aligned[k]['proba'].argmax(1) for k in keys}
    rng = np.random.default_rng(SEED_PATIENTS[fold])
    out = {k: np.empty(B) for k in keys}
    for b in range(B):
        w = np.bincount(rng.integers(0, len(subj), len(subj)), minlength=len(subj))[inv].astype(np.float64)
        for k in keys:
            conf = np.bincount(cell[k], weights=w, minlength=K * K).reshape(K, K)
            out[k][b] = macro_f1_from_conf(conf)
    return out, {'rows': int(len(y)), 'patients': int(len(subj)), 'K': int(K)}


def two_level(delta):
    """delta: array [n_seeds, B] of F1(arm)-F1(ref) per matched seed and patient resample."""
    n = delta.shape[0]
    rng = np.random.default_rng(SEED_SEEDS)
    pick = rng.integers(0, n, size=(B, n))
    stat = delta[pick, np.arange(B)[:, None]].mean(axis=1)
    patients_only = delta.mean(axis=0)
    out = {}
    for name, d in (('seeds_and_patients', stat), ('patients_only', patients_only)):
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[name] = {'ci95': [float(lo), float(hi)], 'p_delta_gt_0': float((d > 0).mean())}
    return out


def analyze(ref, arms, methods):
    loaded, notes = {}, []
    dirs = {}
    for name, root in [('__ref__', ref[1])] + [(n, r) for n, r in arms]:
        found, skipped = discover(root, methods)
        if skipped:
            notes.append(f'{name if name != "__ref__" else ref[0]}: no result.json (skipped): ' + ', '.join(skipped))
        dirs[name] = found
        for (m, s), d in found.items():
            loaded[(name, m, s)] = load_run(d, m, s)
    if not dirs['__ref__']:
        raise Refusal('the reference has no runs for this method filter')
    # all runs must be one task: same split, class set, targets, graph
    keys = list(loaded)
    base = loaded[keys[0]]['binding']
    for k in keys:
        for key in SAME_KEYS:
            if loaded[k]['binding'].get(key) != base.get(key):
                raise Refusal(f'{key} differs between {loaded[keys[0]]["row"]["path"]} and {loaded[k]["row"]["path"]}')
    results = {}
    for fold in ('validation', 'test'):
        have = {k: v for k, v in loaded.items() if fold in v['folds']}
        if len(have) != len(loaded):
            if fold == 'validation':
                raise Refusal('a run lacks validation.npz')
            notes.append('test: some runs lack test.npz; test statistics skipped')
            continue
        results[fold] = bootstrap_f1(have, fold)
    fold_info = {f: results[f][1] for f in results}

    doc = {'protocol': PROTOCOL, 'reference': ref[0], 'arms': [n for n, _ in arms],
           'method_filter': sorted(methods) or None, 'task': {
               'top_k_labels': base.get('top_k_labels'), 'num_classes': base.get('num_classes'),
               'folds': fold_info},
           'bootstrap': {'B': B, 'rng_patients': SEED_PATIENTS, 'rng_seeds': SEED_SEEDS,
                         'levels': 'seeds (matched, with replacement), then patients (by subjects, rows paired by '
                                   'sample_ids; one patient resample per replicate shared by seeds and arms)',
                         'statistic': 'mean over drawn seeds of macro-F1(arm) - macro-F1(reference)',
                         'ci': '95% percentile'},
           'test_note': TEST_NOTE, 'notes': notes, 'groups': []}
    all_methods = sorted({m for (_, m, _) in loaded})
    for arm_name in ['__ref__'] + [n for n, _ in arms]:
        label = ref[0] if arm_name == '__ref__' else arm_name
        for m in all_methods:
            seeds = sorted(s for (a, mm, s) in loaded if a == arm_name and mm == m)
            if not seeds:
                continue
            runs = [loaded[(arm_name, m, s)]['row'] for s in seeds]
            g = {'arm': label, 'is_reference': arm_name == '__ref__', 'method': m, 'seeds': seeds, 'runs': runs,
                 'validation_macro_f1': mean_sd([r['val_macro_f1'] for r in runs]),
                 'validation_accuracy': mean_sd([r['val_accuracy'] for r in runs]),
                 'selected_epoch': mean_sd([r['selected_epoch'] for r in runs])}
            if all('test_macro_f1' in r for r in runs):
                g['test_macro_f1'] = mean_sd([r['test_macro_f1'] for r in runs])
                g['test_accuracy'] = mean_sd([r['test_accuracy'] for r in runs])
            if arm_name != '__ref__':
                ref_seeds = {s for (a, mm, s) in loaded if a == '__ref__' and mm == m}
                matched = [s for s in seeds if s in ref_seeds]
                g['matched_seeds'] = matched
                g['unmatched_seeds'] = [s for s in seeds if s not in ref_seeds]
                if not matched:
                    notes.append(f'{label} {m}: no seed in common with the reference; no comparison')
                else:
                    cmp_ = {'n_matched_seeds': len(matched), 'per_seed': []}
                    for fold in results:
                        boot = results[fold][0]
                        d = np.stack([boot[(arm_name, m, s)] - boot[('__ref__', m, s)] for s in matched])
                        key = 'validation' if fold == 'validation' else 'test'
                        rows = []
                        for s in matched:
                            a, r = loaded[(arm_name, m, s)]['row'], loaded[('__ref__', m, s)]['row']
                            pre = 'val' if fold == 'validation' else 'test'
                            rows.append({'seed': s, 'macro_f1_delta': a[f'{pre}_macro_f1'] - r[f'{pre}_macro_f1'],
                                         'accuracy_delta': a[f'{pre}_accuracy'] - r[f'{pre}_accuracy']})
                        deltas = [x['macro_f1_delta'] for x in rows]
                        cmp_[key] = {'per_seed': rows, 'macro_f1_delta': mean_sd(deltas),
                                     'accuracy_delta': mean_sd([x['accuracy_delta'] for x in rows]),
                                     'bootstrap': two_level(d)}
                        if fold == 'test':
                            cmp_[key]['note'] = TEST_NOTE
                    g['vs_reference'] = cmp_
            doc['groups'].append(g)
    return doc


def ci_txt(b):
    return f"[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]"


def render(doc):
    L = []
    t = doc['task']
    L += ['# Arms vs reference', '',
          f"Protocol {doc['protocol']} (validation selection; test scored once, {TEST_NOTE}). "
          f"K={t['top_k_labels']}; reference `{doc['reference']}`; arms: {', '.join(doc['arms']) or 'none'}; "
          f"methods: {', '.join(doc['method_filter']) if doc['method_filter'] else 'all found'}.",
          f"Validation {t['folds']['validation']['rows']} rows / {t['folds']['validation']['patients']} patients"
          + (f"; test {t['folds']['test']['rows']} rows / {t['folds']['test']['patients']} patients"
             if 'test' in t['folds'] else '') + '.',
          f"Bootstrap B={doc['bootstrap']['B']}: {doc['bootstrap']['levels']}; {doc['bootstrap']['statistic']}; "
          f"{doc['bootstrap']['ci']}.", '']
    for n in doc['notes']:
        L.append(f'NOTE: {n}')
    if doc['notes']:
        L.append('')
    L += ['## Overview (delta = arm - reference, matched seeds)', '',
          '| arm | method | seeds | val macro-F1 | d val [CI seeds+patients] | P(d>0) | CI patients only | '
          'test macro-F1 (reported only) | d test [CI] (reported only) |',
          '|---|---|---|---|---|---|---|---|---|']
    for g in doc['groups']:
        seeds = ','.join(map(str, g['seeds']))
        tm = fmt(g.get('test_macro_f1'))
        if g['is_reference']:
            L.append(f"| {g['arm']} (ref) | {g['method']} | {seeds} | {fmt(g['validation_macro_f1'])} | - | - | - | {tm} | - |")
            continue
        c = g.get('vs_reference')
        if not c:
            L.append(f"| {g['arm']} | {g['method']} | {seeds} | {fmt(g['validation_macro_f1'])} | no matched seed | - | - | {tm} | - |")
            continue
        v, te = c['validation'], c.get('test')
        tcol = (f"{te['macro_f1_delta']['mean']:+.4f} {ci_txt(te['bootstrap']['seeds_and_patients'])}" if te else 'n/a')
        L.append(f"| {g['arm']} | {g['method']} | {seeds} | {fmt(g['validation_macro_f1'])} | "
                 f"{v['macro_f1_delta']['mean']:+.4f} {ci_txt(v['bootstrap']['seeds_and_patients'])} | "
                 f"{v['bootstrap']['seeds_and_patients']['p_delta_gt_0']:.3f} | {ci_txt(v['bootstrap']['patients_only'])} | "
                 f"{tm} | {tcol} |")
    L += ['', 'Intervals are conditional on the fixed evaluation fold; the seed level rests on the matched seeds '
              'only (3 for final-style arms), so a CI that excludes 0 is not seed-robust proof: check per-seed deltas.', '']
    for g in doc['groups']:
        L += [f"## {g['arm']}{' (reference)' if g['is_reference'] else ''} / {g['method']}", '',
              '| seed | val macro-F1 | val accuracy | selected epoch | test macro-F1 (reported only) | test accuracy (reported only) |',
              '|---|---|---|---|---|---|']
        for r in g['runs']:
            L.append(f"| {r['seed']} | {r['val_macro_f1']:.4f} | {r['val_accuracy']:.4f} | {r['selected_epoch']} | "
                     + (f"{r['test_macro_f1']:.4f} | {r['test_accuracy']:.4f} |" if 'test_macro_f1' in r else 'n/a | n/a |'))
        L.append(f"| mean ± SD | {fmt(g['validation_macro_f1'])} | {fmt(g['validation_accuracy'])} | "
                 f"{fmt(g['selected_epoch'], 1)} | {fmt(g.get('test_macro_f1'))} | {fmt(g.get('test_accuracy'))} |")
        c = g.get('vs_reference')
        if c:
            L += ['', f"vs reference `{doc['reference']}`, matched seeds {g['matched_seeds']}"
                      + (f", arm-only seeds {g['unmatched_seeds']}" if g['unmatched_seeds'] else '') + ':', '',
                  '| fold | seed | d macro-F1 | d accuracy |', '|---|---|---|---|']
            for fold in ('validation', 'test'):
                if fold not in c:
                    continue
                tag = fold if fold == 'validation' else f'test ({TEST_NOTE})'
                for x in c[fold]['per_seed']:
                    L.append(f"| {tag} | {x['seed']} | {x['macro_f1_delta']:+.4f} | {x['accuracy_delta']:+.4f} |")
                b = c[fold]['bootstrap']
                L.append(f"| {tag} | mean | {fmt(c[fold]['macro_f1_delta'])} | {fmt(c[fold]['accuracy_delta'])} |")
                L.append(f"| {tag} | CI seeds+patients | {ci_txt(b['seeds_and_patients'])} "
                         f"(P>0 {b['seeds_and_patients']['p_delta_gt_0']:.3f}) | patients only {ci_txt(b['patients_only'])} |")
        L.append('')
    return '\n'.join(L)


def parse_named(items, what):
    out = []
    for it in items:
        name, sep, path = it.partition('=')
        if not sep or not name or not path:
            raise Refusal(f'{what} must be NAME=DIR, got {it!r}')
        out.append((name, path))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--ref', required=True, help='NAME=DIR of the reference runs')
    ap.add_argument('--arm', action='append', default=[], help='NAME=DIR (repeatable)')
    ap.add_argument('--method', action='append', default=[], help='method key(s) (run-dir prefix); repeat or comma-separate')
    ap.add_argument('--out', required=True, help='new output directory (must not exist)')
    args = ap.parse_args(argv)
    try:
        out = Path(args.out)
        if out.exists():
            raise Refusal(f'--out already exists: {out}')
        ref = parse_named([args.ref], '--ref')[0]
        arms = parse_named(args.arm, '--arm')
        names = [ref[0]] + [n for n, _ in arms]
        if len(set(names)) != len(names):
            raise Refusal('arm/ref names must be unique')
        methods = {m for x in args.method for m in x.split(',') if m}
        doc = analyze(ref, arms, methods)
    except Refusal as e:
        print(f'REFUSED: {e}', file=sys.stderr)
        return 2
    out.mkdir(parents=True)
    (out / 'summary.json').write_text(json.dumps(doc, indent=1))
    (out / 'summary.md').write_text(render(doc) + '\n')
    print(f'wrote {out}/summary.md and summary.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
