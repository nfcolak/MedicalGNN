"""Generic analysis of arms vs a reference, fair_v10t runs (shared by the section-2, ablation and split-replication analyses).

    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m comparison.ablation.arms_vs_ref \
        --ref final=<.../clinical_runs_fair_v10t_20261011/top20> --arm x=<dir> [--arm y=<dir>] \
        [--method gsat] --out <new dir>

Every folder holds `<method>_seed<s>/` run dirs (result.json, binding.json, validation.npz, test.npz). The
method key is the run-dir name without `_seed<s>` (so `gsat`, `cei_mp2`, `gchm_pna`); the binding method is
recorded alongside. Read-only on the runs; `--out` must not exist.

Decisions (checkpoint, HPO, baseline choice) use VALIDATION; every reported result is on TEST (scored once per run
at the validation-selected checkpoint). Tables lead with the test fold; validation follows as the decision fold.

Per run: test macro-F1/accuracy (result.json, recomputed from test.npz; a run without test.npz is refused), selected
validation macro-F1/accuracy (recomputed from validation.npz with the repo definition
`f1_score(labels=arange(K), zero_division=0)` and asserted equal to 1e-6), selected epoch. Per arm and method: mean +- SD (ddof=1) over seeds. Arm vs reference over the matched
seeds: per-seed deltas and a two-level bootstrap of the macro-F1 difference (arm - reference):
  level 1 resamples the matched seeds with replacement; level 2 resamples patients (npz `subjects`, all their
  rows with multiplicity; rows paired between arm and reference by `sample_ids`); one patient resample per
  replicate is shared by all seeds drawn and by arm and reference (the evaluation fold is one fixed sample).
  Statistic = mean over the drawn seeds of F1(arm) - F1(reference) on the resampled patients. B = 2000,
  fixed rng. With one matched seed level 1 is trivial and this is the plain patient bootstrap against the
  reference run of that seed. A 'patients only' interval (seeds fixed at the matched seeds) is given too.
Reading rule (same on both folds): drop = reference - arm; 'matters' = drop >= 0.005 and the seeds+patients CI
excludes 0; 'no measurable effect' = the CI includes 0 (noted when drop >= 0.005); 'small effect, not resolved' =
the CI excludes 0 and drop < 0.005.
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
MATTERS = 0.005
PROTOCOL_NOTE = ('Decisions (checkpoint, HPO, baseline choice) use VALIDATION; every reported result is on TEST '
                 '(scored once per run at the validation-selected checkpoint).')


class Refusal(Exception):
    pass


def reading(mean_delta, ci):
    """drop = reference - arm, CI = seeds+patients 95% interval of the delta.
    Returns (reading, note); the note flags a drop >= MATTERS whose CI still includes 0."""
    drop = -mean_delta
    excludes0 = ci[0] > 0 or ci[1] < 0
    if drop >= MATTERS and excludes0:
        return 'matters', ''
    if not excludes0:
        return 'no measurable effect', 'not resolved at this precision' if drop >= MATTERS else ''
    return 'small effect, not resolved', ''


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
                raise Refusal(f'{path}: no test.npz')
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
            loaded[(name, m, s)]['row']['path'] = f'{ref[0] if name == "__ref__" else name}/{d.name}'
    if not dirs['__ref__']:
        raise Refusal('the reference has no runs for this method filter')
    # all runs must be one task: same split, class set, targets, graph
    keys = list(loaded)
    base = loaded[keys[0]]['binding']
    for k in keys:
        for key in SAME_KEYS:
            if loaded[k]['binding'].get(key) != base.get(key):
                raise Refusal(f'{key} differs between {loaded[keys[0]]["row"]["path"]} and {loaded[k]["row"]["path"]}')
    results = {fold: bootstrap_f1(loaded, fold) for fold in ('validation', 'test')}
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
           'protocol_note': PROTOCOL_NOTE, 'notes': notes, 'groups': []}
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
                 'selected_epoch': mean_sd([r['selected_epoch'] for r in runs]),
                 'test_macro_f1': mean_sd([r['test_macro_f1'] for r in runs]),
                 'test_accuracy': mean_sd([r['test_accuracy'] for r in runs])}
            if arm_name != '__ref__':
                ref_seeds = {s for (a, mm, s) in loaded if a == '__ref__' and mm == m}
                matched = [s for s in seeds if s in ref_seeds]
                g['matched_seeds'] = matched
                g['unmatched_seeds'] = [s for s in seeds if s not in ref_seeds]
                if not matched:
                    notes.append(f'{label} {m}: no seed in common with the reference; no comparison')
                else:
                    cmp_ = {'n_matched_seeds': len(matched), 'per_seed': []}
                    for fold in ('test', 'validation'):
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
                        cmp_[key]['reading'], cmp_[key]['reading_note'] = reading(
                            cmp_[key]['macro_f1_delta']['mean'], cmp_[key]['bootstrap']['seeds_and_patients']['ci95'])
                    g['vs_reference'] = cmp_
            doc['groups'].append(g)
    return doc


def ci_txt(b):
    return f"[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]"


def stat_cells(c):
    """reading, d [CI seeds+patients], P(d>0), CI patients only of one fold's comparison."""
    b = c['bootstrap']
    return [c['reading'] + (f" ({c['reading_note']})" if c['reading_note'] else ''),
            f"{c['macro_f1_delta']['mean']:+.4f} {ci_txt(b['seeds_and_patients'])}",
            f"{b['seeds_and_patients']['p_delta_gt_0']:.3f}", ci_txt(b['patients_only'])]


def render(doc, title='Arms vs reference'):
    L = []
    t = doc['task']
    L += [f'# {title}', '',
          f"Protocol {doc['protocol']}. {doc['protocol_note']} "
          f"K={t['top_k_labels']}; reference `{doc['reference']}`; arms: {', '.join(doc['arms']) or 'none'}; "
          f"methods: {', '.join(doc['method_filter']) if doc['method_filter'] else 'all found'}.",
          f"Test {t['folds']['test']['rows']} rows / {t['folds']['test']['patients']} patients; "
          f"validation (decision fold) {t['folds']['validation']['rows']} rows / {t['folds']['validation']['patients']} patients.",
          f"Bootstrap B={doc['bootstrap']['B']}: {doc['bootstrap']['levels']}; {doc['bootstrap']['statistic']}; "
          f"{doc['bootstrap']['ci']}.",
          f"Reading rule (both folds): matters = drop (reference - arm) >= {MATTERS} and the seeds+patients CI excludes 0; "
          f"no measurable effect = CI includes 0; small effect, not resolved = CI excludes 0 and drop < {MATTERS}.", '']
    for n in doc['notes']:
        L.append(f'NOTE: {n}')
    if doc['notes']:
        L.append('')
    L += ['## Overview (delta = arm - reference, matched seeds)', '',
          '| arm | method | seeds | test macro-F1 | test reading | d test [CI seeds+patients] | P(d>0) | CI patients only | '
          'validation macro-F1 (decision fold) | validation reading | d val [CI seeds+patients] | P(d>0) | CI patients only |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for g in doc['groups']:
        seeds = ','.join(map(str, g['seeds']))
        tm, vm = fmt(g['test_macro_f1']), fmt(g['validation_macro_f1'])
        if g['is_reference']:
            L.append(f"| {g['arm']} (ref) | {g['method']} | {seeds} | {tm} | - | - | - | - | {vm} | - | - | - | - |")
            continue
        c = g.get('vs_reference')
        if not c:
            L.append(f"| {g['arm']} | {g['method']} | {seeds} | {tm} | no matched seed | - | - | - | {vm} | - | - | - | - |")
            continue
        L.append(f"| {g['arm']} | {g['method']} | {seeds} | {tm} | " + ' | '.join(stat_cells(c['test']))
                 + f" | {vm} | " + ' | '.join(stat_cells(c['validation'])) + ' |')
    n_matched = sorted({g['vs_reference']['n_matched_seeds'] for g in doc['groups'] if g.get('vs_reference')})
    n_txt = '/'.join(map(str, n_matched)) or '0'
    L += ['', 'Intervals are conditional on the fixed test fold; the seed level rests on the matched seeds '
              f'only ({n_txt} per arm here), so a CI that excludes 0 is not seed-robust proof: check per-seed deltas.', '']
    for g in doc['groups']:
        L += [f"## {g['arm']}{' (reference)' if g['is_reference'] else ''} / {g['method']}", '',
              '| seed | test macro-F1 | test accuracy | validation macro-F1 (decision fold) | validation accuracy | selected epoch |',
              '|---|---|---|---|---|---|']
        for r in g['runs']:
            L.append(f"| {r['seed']} | {r['test_macro_f1']:.4f} | {r['test_accuracy']:.4f} | {r['val_macro_f1']:.4f} | "
                     f"{r['val_accuracy']:.4f} | {r['selected_epoch']} |")
        L.append(f"| mean ± SD | {fmt(g['test_macro_f1'])} | {fmt(g['test_accuracy'])} | {fmt(g['validation_macro_f1'])} | "
                 f"{fmt(g['validation_accuracy'])} | {fmt(g['selected_epoch'], 1)} |")
        c = g.get('vs_reference')
        if c:
            L += ['', f"vs reference `{doc['reference']}`, matched seeds {g['matched_seeds']}"
                      + (f", arm-only seeds {g['unmatched_seeds']}" if g['unmatched_seeds'] else '') + ':', '',
                  '| fold | seed | d macro-F1 | d accuracy |', '|---|---|---|---|']
            for fold in ('test', 'validation'):
                tag = fold if fold == 'test' else 'validation (decision fold)'
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


def main(argv=None, title='Arms vs reference'):
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
    (out / 'summary.md').write_text(render(doc, title) + '\n')
    print(f'wrote {out}/summary.md and summary.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
