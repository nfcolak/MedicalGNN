"""Fair-protocol comparison: any number of methods; fair_v10t is ranked and reported on test, selected on validation.

Accepted runs (no protocol overrides, one protocol and one top_k_labels per comparison, each its own group):
  * `fair_v10t` (`core.train.PROTOCOL_NAME`, `core.train.FAIR_PROTOCOL` with a per-run `top_k_labels` in
    ALLOWED_TOP_K): decisions (checkpoint) use VALIDATION; every reported result is on TEST. The ranking, the
    mean +- SD and the patient-cluster paired bootstrap of all method pairs (ordered by test rank) use each run's
    `test.npz` (scored once); the mean selected VALIDATION macro-F1 and its rank stay as a labelled selection
    column. A run without `test.npz`, with `test_evaluations != 1` or `test_evaluated` not true is refused.
  * `fair_v10` (full TRAIN minus a 5000-row dev split, dev selection) and legacy `fair_v3` (Top-10): ranked by mean
    selected DEV macro-F1, validation fold (single evaluation) reported, test never evaluated.
Runs of one comparison must share (protocol, top_k_labels, kept_label_indices, split hashes); the class
count K is taken from the data.

Ablation arms: several arms of one method (cei_gnn_mp2) can be compared. A run's arm label is its method name plus
the ablation options that differ from the final CEI recipe (ARM_DEFAULTS), read from binding['method_config']:
`cei_gnn_mp2` is the Full model, `cei_gnn_mp2[ev_pair=0]` an arm. `--label NAME` (once per `--run`, same order)
overrides the derived label. Arms of one method must agree on everything else (the other method options, lr,
dropout, weight decay, hidden size, optimizer, batch size); a run that differs there is refused.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from core.contracts import sample_ids_sha256
from core.train import (FAIR_PROTOCOL, LEGACY_V10_FAIR_PROTOCOL, LEGACY_V10_PROTOCOL_NAME,
                        PROTOCOL_NAME)

ALLOWED_TOP_K = (10, 20, 30)
LEGACY_PROTOCOL = 'fair_v3'
LEGACY_FAIR_PROTOCOL = dict(train_limit=10000, sample_seed=1234, top_k_labels=10,
                            weights='sqrt_inverse', selection_fold='dev', dev_limit=5000,
                            final_eval='validation', epochs=40, patience=40)
N_BOOT = 1000
SEED = 2026
FOLDS = ('train', 'dev', 'validation')
FOLDS_TEST_PROTOCOL = ('train', 'validation', 'test')
CAVEAT_TEST = ('Fair protocol fair_v10t: whole TRAIN fold of the kept classes, checkpoint selected per run on the '
               'validation fold (selection column = mean selected validation macro-F1); every reported number and '
               'the ranking are on the test fold, scored exactly once per run at that checkpoint and never used for '
               'selection; CI is conditional on these training seeds and the fixed test fold.')
CAVEAT = ('Fair protocol fair_v3: one fixed 10k train sample and 5k dev draw; ranking uses the '
          'dev macro-F1 selected per run, CI is conditional on these training seeds; '
          'validation fold only (single evaluation per run); test fold never evaluated.')


# Ablation options of cei_gnn_mp2 and their value in the final (Full) recipe; a run is an arm of Full iff
# some of them differ. A key missing from a binding (runs made before the switch existed) has the Full value.
ARM_DEFAULTS = {'arm': 'C', 'mp_feed': 'dual', 'ev_combine': 'sum', 'ev_pair': 1, 'ev_edge': 1,
                'readout': 'evidence', 'pool_hidden': 0, 'encoder_depth': 1, 'comorbid_block': 0}
# binding keys that must agree between arms of one method
ARM_SAME_KEYS = ('lr', 'dropout', 'weight_decay', 'hidden', 'optimizer', 'batch_size', 'layers', 'epochs',
                 'patience', 'weights')


class Refusal(Exception):
    pass


def arm_options(binding):
    """{ablation option: value} of one run (Full values for anything the binding does not record)."""
    config = binding.get('method_config') or {}
    options = dict(config.get('mp2_options') or {})
    values = dict(ARM_DEFAULTS)
    for key in values:
        if key in options:
            values[key] = options[key]
        elif key in config:
            values[key] = config[key]
    return values


def arm_label(binding, method):
    """`method` for the Full recipe, else `method[opt=value,...]` listing the differing ablation options."""
    if binding.get('method') != 'cei_gnn_mp2':
        return method
    values = arm_options(binding)
    diff = [f'{k}={values[k]}' for k in ARM_DEFAULTS if values[k] != ARM_DEFAULTS[k]
            and not (k == 'pool_hidden' and values['readout'] != 'pool')]
    if values['readout'] == 'pool':   # always show the pooled head width
        diff = [d for d in diff if not d.startswith('pool_hidden=')] + [f"pool_hidden={values['pool_hidden']}"]
    return method + (f"[{','.join(diff)}]" if diff else '')


def arm_rest(binding):
    """Everything an arm of one method must share: the non-ablation method options and the common settings."""
    config = binding.get('method_config') or {}
    options = {k: v for k, v in (config.get('mp2_options') or {}).items() if k not in ARM_DEFAULTS}
    return {'options': options, **{k: binding.get(k) for k in ARM_SAME_KEYS}}


def caveat_for(protocol):
    if protocol == PROTOCOL_NAME:
        return CAVEAT_TEST
    return CAVEAT.replace('fair_v3', protocol) if protocol != LEGACY_PROTOCOL else CAVEAT


def uses_test(protocol):
    """fair_v10t runs report their (single) test scoring; the older protocols never evaluate test."""
    return protocol == PROTOCOL_NAME


def expected_resolved(binding, path='run'):
    """The protocol_resolved a clean run of binding['protocol'] must carry."""
    protocol = binding.get('protocol')
    if protocol == LEGACY_PROTOCOL:
        return dict(LEGACY_FAIR_PROTOCOL)
    if protocol in (PROTOCOL_NAME, LEGACY_V10_PROTOCOL_NAME):
        resolved = binding.get('protocol_resolved')
        k = resolved.get('top_k_labels') if isinstance(resolved, dict) else None
        if k not in ALLOWED_TOP_K:
            raise Refusal(f'{path}: top_k_labels {k!r} not in {ALLOWED_TOP_K}')
        if binding.get('top_k_labels') != k:
            raise Refusal(f"{path}: binding top_k_labels {binding.get('top_k_labels')!r} "
                          f'!= protocol_resolved {k!r}')
        return dict(FAIR_PROTOCOL if protocol == PROTOCOL_NAME else LEGACY_V10_FAIR_PROTOCOL, top_k_labels=k)
    raise Refusal(f'{path}: protocol is {protocol!r}, not {PROTOCOL_NAME!r}, {LEGACY_V10_PROTOCOL_NAME!r} '
                  f'or {LEGACY_PROTOCOL!r}')


def class_count(proba, y):
    """K from the data; y must be valid class indices."""
    K = int(proba.shape[1])
    if len(y) != len(proba):
        raise Refusal(f'proba has {len(proba)} rows but y has {len(y)}')
    if len(y) and (int(y.max()) >= K or int(y.min()) < 0):
        raise Refusal(f'y has class index outside [0, {K})')
    return K


def macro_f1_from_conf(conf):
    tp = np.diagonal(conf, axis1=-2, axis2=-1).astype(np.float64)
    fp = conf.sum(axis=-2) - tp
    fn = conf.sum(axis=-1) - tp
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
    return f1.mean(axis=-1)


def point_metrics(proba, y):
    K = class_count(proba, y)
    pred = proba.argmax(1)
    conf = np.bincount(y * K + pred, minlength=K * K).reshape(K, K)
    recall = np.diagonal(conf) / np.maximum(conf.sum(1), 1)
    present = conf.sum(1) > 0
    top3 = np.argsort(-proba, axis=1)[:, :3]
    return {
        'macro_f1': float(macro_f1_from_conf(conf)),
        'accuracy': float((pred == y).mean()),
        'balanced_accuracy': float(recall[present].mean()),
        'top3_accuracy': float((top3 == y[:, None]).any(1).mean()),
    }


def _load_npz(path, name, binding, fold, check_hash=True):
    """proba, y, subjects, sample_ids of one stored fold, bound to the binding's sample-id hash."""
    z = np.load(Path(path) / name)
    proba, y = z['proba'].astype(np.float64), z['y'].astype(np.int64)
    ids = z['sample_ids']
    expected = (binding.get('split_sample_ids_sha256') or {}).get(fold)
    if check_hash and expected is not None and sample_ids_sha256(ids.tolist()) != expected:
        raise Refusal(f'{path}: {name} sample_ids differ from binding split_sample_ids_sha256[{fold!r}]')
    return proba, y, z['subjects'], ids


def load_run(path, label=None):
    path = Path(path)
    for name in ('result.json', 'binding.json', 'validation.npz'):
        if not (path / name).is_file():
            raise Refusal(f'{path}: missing {name}')
    result = json.loads((path / 'result.json').read_text())
    binding = json.loads((path / 'binding.json').read_text())
    if result.get('status') != 'completed':
        raise Refusal(f"{path}: status is {result.get('status')!r}, not 'completed'")
    protocol = binding.get('protocol')
    test_protocol = uses_test(protocol)
    if test_protocol:
        if not (path / 'test.npz').is_file():
            raise Refusal(f'{path}: missing test.npz')
        if binding.get('test_evaluated') is not True:
            raise Refusal(f'{path}: binding test_evaluated is not true')
        if result.get('test_evaluations') != 1:
            raise Refusal(f"{path}: test_evaluations is {result.get('test_evaluations')!r}, not 1")
    elif binding.get('test_evaluated') is not False:
        raise Refusal(f'{path}: binding test_evaluated is not false')
    expected = expected_resolved(binding, path)
    if binding.get('protocol_overrides') != []:
        raise Refusal(f"{path}: protocol_overrides is {binding.get('protocol_overrides')!r}, "
                      'not an empty list')
    if binding.get('protocol_resolved') != expected:
        raise Refusal(f"{path}: protocol_resolved {binding.get('protocol_resolved')!r} "
                      f'differs from the fair protocol {expected!r}')
    hashes = binding.get('split_sample_ids_sha256')
    folds = FOLDS_TEST_PROTOCOL if test_protocol else FOLDS
    if not isinstance(hashes, dict) or any(not hashes.get(f) for f in folds):
        raise Refusal(f"{path}: split_sample_ids_sha256 lacks {', '.join(folds)}")
    if test_protocol:
        if result.get('selection_fold') != 'validation':
            raise Refusal(f"{path}: result.json selection_fold is {result.get('selection_fold')!r}, "
                          "not 'validation'")
        select = (result.get('metrics') or {}).get('macro_f1')
        if not isinstance(select, (int, float)):
            raise Refusal(f'{path}: result.json has no selected validation macro_f1')
    else:
        select = (result.get('dev_metrics') or {}).get('macro_f1')
        if result.get('selection_fold') != 'dev' or not isinstance(select, (int, float)):
            raise Refusal(f'{path}: result.json has no dev-selected macro_f1')
    method = binding['method']
    if method == 'clinical_gnn':
        method = f"{method}:{binding.get('conv')}"
    # `proba/y/subjects/sample_ids` are the REPORTED evaluation set: test.npz for fair_v10t, else validation.npz
    proba, y, subjects, sample_ids = _load_npz(path, 'test.npz' if test_protocol else 'validation.npz',
                                               binding, 'test' if test_protocol else 'validation',
                                               check_hash=test_protocol)
    K = class_count(proba, y)
    if test_protocol:
        v_proba, v_y, _, _ = _load_npz(path, 'validation.npz', binding, 'validation')
        if v_proba.shape[1] != K:
            raise Refusal(f'{path}: validation.npz and test.npz class counts differ')
        v_f1 = point_metrics(v_proba, v_y)['macro_f1']
        if abs(v_f1 - select) > 1e-6:
            raise Refusal(f'{path}: recomputed validation macro_f1 {v_f1:.6f} != result.json {select}')
        if (result.get('test') or {}).get('n') != len(y):
            raise Refusal(f"{path}: result.json test n {(result.get('test') or {}).get('n')!r} "
                          f'!= test.npz rows {len(y)}')
    if 'num_classes' in binding and binding['num_classes'] != K:
        raise Refusal(f"{path}: binding num_classes {binding['num_classes']!r} != proba.shape[1] {K}")
    kept = binding.get('kept_label_indices')
    if kept is not None and len(kept) != K:
        raise Refusal(f'{path}: len(kept_label_indices) {len(kept)} != proba.shape[1] {K}')
    if binding.get('protocol') != LEGACY_PROTOCOL and K != binding['top_k_labels']:
        raise Refusal(f"{path}: proba.shape[1] {K} != top_k_labels {binding['top_k_labels']}")
    base_method = method
    method = label or arm_label(binding, method)
    return {'path': str(path), 'method': method, 'base_method': base_method, 'seed': binding['seed'],
            'binding': binding,
            'select_macro_f1': float(select), 'test_protocol': test_protocol,
            'result': result, 'proba': proba, 'y': y,
            'subjects': subjects, 'sample_ids': sample_ids}


def check_and_align(runs):
    ref = runs[0]
    for key in ('protocol', 'top_k_labels', 'num_classes', 'split_sample_ids_sha256',
                'kept_label_indices', 'artifact_graphs_sha256', 'targets_sha256'):
        for r in runs:
            if r['binding'].get(key) != ref['binding'].get(key):
                raise Refusal(f"{key} differs between {ref['path']} and {r['path']}")
    for r in runs:
        if r['proba'].shape[1] != ref['proba'].shape[1]:
            raise Refusal(f"class count differs between {ref['path']} and {r['path']}")
    for r in runs:
        if len(set(r['sample_ids'])) != len(r['sample_ids']):
            raise Refusal(f"{r['path']}: duplicate evaluation sample_ids")
    first_of = {}
    for r in runs:   # arms of one method differ only in their ablation options
        rest = first_of.setdefault(r['base_method'], r)
        if arm_rest(r['binding']) != arm_rest(rest['binding']):
            diff = sorted(k for k, v in arm_rest(r['binding']).items() if v != arm_rest(rest['binding'])[k])
            raise Refusal(f"{r['base_method']}: {r['path']} differs from {rest['path']} in more than the ablation "
                          f'options ({diff})')
    seen = set()
    for r in runs:
        ident = (r['method'], r['seed'])
        if ident in seen:
            raise Refusal(f'duplicate method/seed {ident}')
        seen.add(ident)
    if all(np.array_equal(r['sample_ids'], ref['sample_ids']) for r in runs):
        _check_labels(runs, ref)
        return
    if any(set(r['sample_ids']) != set(ref['sample_ids']) for r in runs):
        raise Refusal('evaluation sample_id sets differ between runs')
    order = {s: i for i, s in enumerate(ref['sample_ids'])}
    for r in runs:
        idx = np.argsort([order[s] for s in r['sample_ids']])
        for k in ('proba', 'y', 'subjects', 'sample_ids'):
            r[k] = r[k][idx]
    _check_labels(runs, ref)


def _check_labels(runs, ref):
    for r in runs:
        if not (np.array_equal(r['y'], ref['y']) and np.array_equal(r['subjects'], ref['subjects'])):
            raise Refusal('y/subjects disagree after alignment by sample_id')


def rank_by_dev(runs, methods):
    """Methods by mean selected macro-F1 over their seeds (dev; validation for fair_v10t), best first."""
    mean = {m: float(np.mean([r['select_macro_f1'] for r in runs if r['method'] == m]))
            for m in methods}
    return sorted(methods, key=lambda m: (-mean[m], m)), mean


def bootstrap(runs, methods, ref):
    subj, inv = np.unique(ref['subjects'], return_inverse=True)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(subj), size=(N_BOOT, len(subj)))
    counts = np.stack([np.bincount(d, minlength=len(subj)) for d in draws])  # [B, S]
    row_w = counts[:, inv].astype(np.float64)  # [B, N]
    y = ref['y']
    K = class_count(ref['proba'], y)
    per_method = {m: [] for m in methods}
    for r in runs:
        idx = y * K + r['proba'].argmax(1)
        conf = np.stack([np.bincount(idx, weights=w, minlength=K * K) for w in row_w])
        per_method[r['method']].append(macro_f1_from_conf(conf.reshape(N_BOOT, K, K)))
    return {m: np.mean(v, axis=0) for m, v in per_method.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(prog='comparison.top3.compare')
    ap.add_argument('--run', action='append', required=True)
    ap.add_argument('--label', action='append', default=None,
                    help='arm label of the matching --run (give one per --run, in order); default: derived from the run')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    try:
        out = Path(args.out)
        if out.exists():
            raise Refusal(f'--out already exists: {out}')
        if args.label is not None and len(args.label) != len(args.run):
            raise Refusal(f'--label given {len(args.label)} times for {len(args.run)} --run')
        runs = [load_run(p, (args.label or [None] * len(args.run))[i]) for i, p in enumerate(args.run)]
        check_and_align(runs)
        for r in runs:
            r['metrics'] = point_metrics(r['proba'], r['y'])
            rep = (r['result']['test'] if r['test_protocol'] else r['result']['metrics'])['macro_f1']
            if abs(r['metrics']['macro_f1'] - rep) > 1e-6:
                raise Refusal(f"{r['path']}: recomputed macro_f1 {r['metrics']['macro_f1']:.6f} "
                              f'!= result.json {rep}')
    except Refusal as e:
        print(f'REFUSED: {e}', file=sys.stderr)
        return 2

    methods = list(dict.fromkeys(r['method'] for r in runs))
    test_protocol = runs[0]['test_protocol']
    fold = 'test' if test_protocol else 'validation'
    select_fold = 'validation' if test_protocol else 'dev'
    names = ('macro_f1', 'accuracy', 'balanced_accuracy', 'top3_accuracy')
    summary = {}
    for m in methods:
        rs = [r for r in runs if r['method'] == m]
        summary[m] = {'n_seeds': len(rs), 'seeds': [r['seed'] for r in rs]}
        for n in names:
            v = np.array([r['metrics'][n] for r in rs])
            summary[m][n] = {'mean': float(v.mean()),
                             'sd': float(v.std(ddof=1)) if len(v) > 1 else None}

    ranking, dev_mean = rank_by_dev(runs, methods)
    test_ranking = sorted(methods, key=lambda m: (-summary[m]['macro_f1']['mean'], m))
    for m in methods:
        dev = np.array([r['select_macro_f1'] for r in runs if r['method'] == m])
        summary[m][f'{select_fold}_macro_f1_selected' if test_protocol else 'dev_macro_f1'] = {
            'mean': dev_mean[m], 'sd': float(dev.std(ddof=1)) if len(dev) > 1 else None}
    top = list(test_ranking if test_protocol else ranking)   # fair_v10t: pairs ordered by test rank
    ref = runs[0]
    boot = bootstrap([r for r in runs if r['method'] in top], top, ref)
    pairs = []
    for a, b in itertools.combinations(top, 2):
        d = boot[a] - boot[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        pairs.append({'a': a, 'b': b,
                      'observed_delta_macro_f1': summary[a]['macro_f1']['mean'] - summary[b]['macro_f1']['mean'],
                      'ci95': [float(lo), float(hi)], 'share_delta_gt_0': float((d > 0).mean())})

    doc = {
        'inputs': [{'run': r['path'], 'method': r['method'], 'seed': r['seed'],
                    **{k: r['binding'].get(k) for k in (
                        'split_sample_ids_sha256', 'kept_label_indices', 'artifact_graphs_sha256',
                        'targets_sha256')}} for r in runs],
        f'{fold}_sample_ids_sha256': sample_ids_sha256(ref['sample_ids'].tolist()),
        f'n_{fold}_rows': int(len(ref['y'])),
        f'n_{fold}_subjects': int(len(np.unique(ref['subjects']))),
        'per_run': [{'run': r['path'], 'method': r['method'], 'seed': r['seed'], **r['metrics']}
                    for r in runs],
        'protocol': ref['binding']['protocol'],
        'protocol_resolved': expected_resolved(ref['binding']),
        **({'ranking_by_test_macro_f1': [
            {'rank': i + 1, 'method': m, 'test_macro_f1_mean': summary[m]['macro_f1']['mean']}
            for i, m in enumerate(test_ranking)]} if test_protocol else {}),
        f'ranking_by_{select_fold}_macro_f1': [
            {'rank': i + 1, 'method': m, f'{select_fold}_macro_f1_mean': dev_mean[m]}
            for i, m in enumerate(ranking)],
        'bootstrap_methods': top,
        'per_method': summary,
        'pairs': pairs,
        'settings': {'bootstrap_resamples': N_BOOT, 'rng': f'numpy default_rng({SEED})',
                     'resampling_unit': 'subjects (all rows, with multiplicity)',
                     'ranking': (f'mean {fold} macro-F1 over seeds, descending (reported); selection column: mean '
                                 f'selected {select_fold} macro-F1 over seeds' if test_protocol
                                 else f'mean selected {select_fold} macro-F1 over seeds, descending'),
                     'bootstrap_scope': (f'all pairs among all compared methods, ordered by {fold} rank' if test_protocol
                                         else f'all pairs among all compared methods, ranked by {select_fold}'),
                     'statistic': f'{fold} macro-F1 averaged over each method\'s seeds; delta = A - B',
                     'ci': '95% percentile', 'sd': 'sample SD over seeds (ddof=1)'},
        'caveat': caveat_for(ref['binding']['protocol']),
    }
    if ref['binding']['protocol'] != LEGACY_PROTOCOL:
        doc['top_k_labels'] = ref['binding']['top_k_labels']
        doc['n_classes'] = int(ref['proba'].shape[1])
    out.mkdir(parents=True)
    (out / 'comparison.json').write_text(json.dumps(doc, indent=2))

    if test_protocol:
        val_rank = {m: i + 1 for i, m in enumerate(ranking)}
        print('Test ranking (reported): mean test macro-F1 over seeds, scored once per run')
        print(f"{'rank':>4}  {'method':<24}{'seeds':>6}  {'macro-F1':>16}  {'accuracy':>16}  {'top-3':>16}  {'validation rank':>15}")
        for i, m in enumerate(test_ranking):
            s = summary[m]
            cell = lambda n: f"{s[n]['mean']:.4f} ± {s[n]['sd'] if s[n]['sd'] is not None else float('nan'):.4f}"
            print(f"{i + 1:>4}  {m:<24}{s['n_seeds']:>6}  {cell('macro_f1'):>16}  {cell('accuracy'):>16}  "
                  f"{cell('top3_accuracy'):>16}  {val_rank[m]:>15}")
        print()
        print('Ranking chosen on validation (selection column, decides the checkpoint): mean selected VALIDATION macro-F1')
        for i, m in enumerate(ranking):
            sd = summary[m]['validation_macro_f1_selected']['sd']
            print(f"{i + 1:>3}. {m:<24}{dev_mean[m]:.4f} ± {sd if sd is not None else float('nan'):.4f}"
                  + (f"   (test rank {test_ranking.index(m) + 1})"))
    else:
        print(f'Ranking by mean selected {select_fold.upper()} macro-F1')
        for i, m in enumerate(ranking):
            sd = summary[m]['dev_macro_f1']['sd']
            print(f"{i + 1:>3}. {m:<24}{dev_mean[m]:.4f} ± {sd if sd is not None else float('nan'):.4f}"
                  + ('   (bootstrap)' if m in top else ''))
        print()
        print('Validation (single evaluation per run)')
        print(f"{'method':<24}{'seeds':>6}  {'macro-F1':>16}  {'accuracy':>16}  {'top-3':>16}")
        for m in ranking:
            s = summary[m]
            cell = lambda n: f"{s[n]['mean']:.4f} ± {s[n]['sd'] if s[n]['sd'] is not None else float('nan'):.4f}"
            print(f"{m:<24}{s['n_seeds']:>6}  {cell('macro_f1'):>16}  {cell('accuracy'):>16}  {cell('top3_accuracy'):>16}")
    print()
    print(f'Paired {fold} bootstrap, all pairs among all {len(top)} methods'
          + (' (ordered by test rank)' if test_protocol else ''))
    print(f"{'A vs B':<38}{'delta':>9}  {'95% CI':>20}  {'P(d>0)':>7}")
    for p in pairs:
        print(f"{p['a'] + ' - ' + p['b']:<38}{p['observed_delta_macro_f1']:>+9.4f}  "
              f"[{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]  {p['share_delta_gt_0']:>7.3f}")
    print('\n' + caveat_for(ref['binding']['protocol']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
