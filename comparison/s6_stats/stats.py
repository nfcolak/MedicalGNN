"""Statistical reliability of the stored final runs (fair_v10t): aggregated tables only, nothing is trained.

Reads `result.json`, `binding.json`, `validation.npz` and `test.npz` of the stored runs
(`<runs>/top<K>/<method>_seed<S>/`) and writes aggregated tables into a NEW output directory:
  * parity: macro-F1 and accuracy of every run recomputed from the stored probabilities on both folds
    must equal `result.json` within 1e-6 (otherwise exit 2 before anything is written);
  * per-seed and summary tables; macro-F1 variants (fixed / present-class / >= 30 rows) with rankings;
  * patient-only bootstrap of `comparison/top3/compare.py` reproduced exactly (B=1000, rng 2026, one draw shared
    by all runs of a K, statistic = mean over the seeds of the per-seed macro-F1 on the resampled confusion
    matrix) and a two-level seed x patient bootstrap (level 1: seeds resampled per method; level 2: the shared
    patient draw), for CEI-GNN against every baseline and against the mean of the baselines;
  * per-class precision / recall / F1 / support (mean, SD over seeds) and the class spread of the CEI gain.
Validation decides ("best baseline" = highest mean validation macro-F1); test numbers are reported only.
Outputs hold no sample or subject ids.
"""
import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

KS = (10, 20, 30)
SEEDS = (1234, 2025, 7)
DEFAULT_METHODS = ('cei_mp2', 'gsat', 'graphcare', 'protgnn')
CEI = 'cei_mp2'
FOLDS = ('validation', 'test')
FOLD_FILE = {'validation': 'validation.npz', 'test': 'test.npz'}
FOLD_RESULT_KEY = {'validation': 'metrics', 'test': 'test'}
NAME = {'cei_mp2': 'CEI-GNN mp2', 'gsat': 'GSAT', 'graphcare': 'GraphCare-style (adapted)',
        'protgnn': 'ProtGNN', 'gchm_pna': 'GCHM-PNA'}
# compare.py names (method of the binding) -> method names of this module
COMPARE_NAMES = {'cei_gnn_mp2': 'cei_mp2', 'clinical_gnn:gchm': 'gchm_pna'}
N_PATIENT_BOOT = 1000          # compare.py N_BOOT
N_TWO_LEVEL_BOOT = 2000
PATIENT_RNG = 2026             # compare.py SEED
TWO_LEVEL_RNG_BASE = 7777      # nf-s6-son s6a_analysis.py: default_rng(7777 + K) draws the seed picks
BONFERRONI_M = 9               # 3 baselines x 3 K
MIN_SUPPORT = 30               # classes below this are marked "not evaluable"
TOL = 1e-6
TEST_NOTE = 'test fold: reported only, not used for decisions'


class Refusal(Exception):
    pass


def display_path(p):
    """Repo-relative path for printing/storing; never an absolute path."""
    p = Path(p)
    try:
        return str(p.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return p.name


def pretty(m):
    return NAME.get(m, m)


# ---------------------------------------------------------------------------------------------- metrics
def macro_f1_from_conf(conf):
    """Identical to comparison.top3.compare.macro_f1_from_conf."""
    tp = np.diagonal(conf, axis1=-2, axis2=-1).astype(np.float64)
    fp = conf.sum(axis=-2) - tp
    fn = conf.sum(axis=-1) - tp
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
    return f1.mean(axis=-1)


def per_class_from_conf(conf):
    tp = np.diagonal(conf).astype(np.float64)
    support = conf.sum(axis=1).astype(np.float64)
    n_pred = conf.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, n_pred, out=np.zeros_like(tp), where=n_pred > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    den = 2 * tp + (n_pred - tp) + (support - tp)
    f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
    return precision, recall, f1, support.astype(np.int64), n_pred.astype(np.int64)


def run_metrics(y, pred, K):
    conf = np.bincount(y * K + pred, minlength=K * K).reshape(K, K)
    precision, recall, f1, support, n_pred = per_class_from_conf(conf)
    present = support > 0
    ge = support >= MIN_SUPPORT
    return {
        'macro_f1': float(macro_f1_from_conf(conf)),
        'macro_f1_sklearn': float(f1_score(y, pred, labels=np.arange(K), average='macro', zero_division=0)),
        'accuracy': float((pred == y).mean()),
        'balanced_accuracy': float(recall[present].mean()),
        'precision': precision, 'recall': recall, 'f1': f1, 'support': support, 'n_pred': n_pred,
        'fixed': float(f1.mean()),
        'present': float(f1[present].mean()) if present.any() else float('nan'),
        'ge30': float(f1[ge].mean()) if ge.any() else float('nan'),
        'n_present': int(present.sum()), 'n_ge30': int(ge.sum()),
    }


def mean_sd(values):
    v = np.asarray(values, dtype=np.float64)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float('nan'))


# ---------------------------------------------------------------------------------------------- loading
def load_fold(path, name):
    z = np.load(path / name)
    return z['proba'], z['y'].astype(np.int64), z['subjects'], z['sample_ids']


def align(items, where):
    """items: list of (key, pred, y, subjects, ids); returns reference y, subjects and aligned preds."""
    ref = items[0]
    ref_ids = ref[4]
    out = {}
    order = {s: i for i, s in enumerate(ref_ids)}
    for key, pred, y, subj, ids in items:
        if len(set(ids)) != len(ids):
            raise Refusal(f'{where} {key}: duplicate sample_ids')
        if not np.array_equal(ids, ref_ids):
            if set(ids) != set(ref_ids):
                raise Refusal(f'{where} {key}: sample_id set differs from the reference run')
            idx = np.argsort([order[s] for s in ids])
            pred, y, subj = pred[idx], y[idx], subj[idx]
        if not (np.array_equal(y, ref[2]) and np.array_equal(subj, ref[3])):
            raise Refusal(f'{where} {key}: y/subjects disagree after alignment by sample_id')
        out[key] = pred
    return ref[2], ref[3], out


def load_all(runs_dir, methods, ks, seeds):
    data = {}
    for K in ks:
        labels = None
        stored = {}
        fold_items = {f: [] for f in FOLDS}
        for m in methods:
            for s in seeds:
                d = runs_dir / f'top{K}' / f'{m}_seed{s}'
                for name in ('result.json', 'binding.json', 'validation.npz', 'test.npz'):
                    if not (d / name).is_file():
                        raise Refusal(f'missing {display_path(d)}/{name}')
                result = json.loads((d / 'result.json').read_text())
                binding = json.loads((d / 'binding.json').read_text())
                if result.get('status') != 'completed' or result.get('test_evaluations') != 1:
                    raise Refusal(f'top{K}/{m}_seed{s}: run not completed or test_evaluations != 1')
                lo = binding.get('label_order')
                if lo is None or len(lo) != K or binding.get('num_classes', K) != K:
                    raise Refusal(f'top{K}/{m}_seed{s}: label_order does not have {K} entries')
                if labels is None:
                    labels = list(lo)
                elif labels != list(lo):
                    raise Refusal(f'top{K}/{m}_seed{s}: label_order differs inside K={K}')
                stored[(m, s)] = result
                for fold in FOLDS:
                    proba, y, subj, ids = load_fold(d, FOLD_FILE[fold])
                    if proba.shape[1] != K:
                        raise Refusal(f'top{K}/{m}_seed{s}: {fold} proba has {proba.shape[1]} classes, not {K}')
                    if len(y) and (int(y.max()) >= K or int(y.min()) < 0):
                        raise Refusal(f'top{K}/{m}_seed{s}: {fold} y outside [0, {K})')
                    fold_items[fold].append(((m, s), proba.argmax(1).astype(np.int64), y, subj, ids))
        folds = {}
        for fold in FOLDS:
            y, subj, preds = align(fold_items[fold], f'top{K} {fold}')
            u, inv = np.unique(subj, return_inverse=True)
            folds[fold] = {'y': y, 'inv': inv, 'S': len(u), 'N': len(y), 'preds': preds}
        data[K] = {'labels': labels, 'stored': stored, 'folds': folds}
    return data


# ---------------------------------------------------------------------------------------------- parity
def parity_check(data, methods, seeds):
    rows, bad = [], []
    for K, dk in data.items():
        for fold in FOLDS:
            fd = dk['folds'][fold]
            for m in methods:
                for s in seeds:
                    met = run_metrics(fd['y'], fd['preds'][(m, s)], K)
                    dk.setdefault('metrics', {})[(fold, m, s)] = met
                    st = dk['stored'][(m, s)][FOLD_RESULT_KEY[fold]]
                    d_f1 = abs(met['macro_f1'] - st['macro_f1'])
                    d_sk = abs(met['macro_f1_sklearn'] - st['macro_f1'])
                    d_acc = abs(met['accuracy'] - st['accuracy'])
                    ok = max(d_f1, d_sk, d_acc) <= TOL
                    row = {'K': K, 'fold': fold, 'method': m, 'seed': s, 'n_rows': fd['N'],
                           'macro_f1_recomputed': met['macro_f1'], 'macro_f1_recomputed_sklearn': met['macro_f1_sklearn'],
                           'macro_f1_stored': st['macro_f1'], 'macro_f1_abs_diff': d_f1,
                           'accuracy_recomputed': met['accuracy'], 'accuracy_stored': st['accuracy'],
                           'accuracy_abs_diff': d_acc, 'ok': bool(ok)}
                    rows.append(row)
                    if not ok:
                        bad.append(row)
    return rows, bad


# ---------------------------------------------------------------------------------------------- bootstrap
def patient_bootstrap(fd, K, keys, n_boot, rng_seed):
    """compare.py `bootstrap`: one subject draw (with replacement) shared by all runs; row weight = multiplicity.
    Returns per-run macro-F1 of the resampled confusion matrix, [n_boot] per run."""
    S = fd['S']
    rng = np.random.default_rng(rng_seed)
    draws = rng.integers(0, S, size=(n_boot, S))
    counts = np.stack([np.bincount(d, minlength=S) for d in draws])
    row_w = counts[:, fd['inv']].astype(np.float64)
    out = {}
    for k in keys:
        idx = fd['y'] * K + fd['preds'][k]
        conf = np.stack([np.bincount(idx, weights=w, minlength=K * K) for w in row_w])
        out[k] = macro_f1_from_conf(conf.reshape(n_boot, K, K))
    return out


def two_level(per_run_boot, order, seeds, K, n_boot):
    """nf-s6-son: independent seed resampling per method (len(seeds) draws with replacement) on top of the shared
    patient draw; seed picks from default_rng(7777 + K) of shape [B, methods, seeds]."""
    rng = np.random.default_rng(TWO_LEVEL_RNG_BASE + K)
    pick = rng.integers(0, len(seeds), size=(n_boot, len(order), len(seeds)))
    out = {}
    for mi, m in enumerate(order):
        arr = np.stack([per_run_boot[(m, s)] for s in seeds], axis=1)
        out[m] = np.take_along_axis(arr, pick[:, mi, :], axis=1).mean(axis=1)
    return out


def interval(d, level_alpha):
    lo, hi = np.percentile(d, [100 * level_alpha / 2, 100 * (1 - level_alpha / 2)])
    return float(lo), float(hi)


def compare_json_check(compare_dir, K, meanboot):
    """Max abs difference of the patient-only test intervals and P(d>0) to the stored compare_top<K>.json."""
    p = Path(compare_dir) / f'compare_top{K}.json'
    if p.is_dir():
        p = p / 'comparison.json'
    if not p.is_file():
        raise Refusal(f'--compare-dir has no compare_top{K}.json')
    doc = json.loads(p.read_text())
    norm = lambda n: COMPARE_NAMES.get(n, n)
    diffs, n = [], 0
    for pair in doc['pairs']:
        a, b = norm(pair['a']), norm(pair['b'])
        if a not in meanboot or b not in meanboot:
            continue
        d = meanboot[a] - meanboot[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        diffs.append(max(abs(float(lo) - pair['ci95'][0]), abs(float(hi) - pair['ci95'][1]),
                         abs(float((d > 0).mean()) - pair['share_delta_gt_0'])))
        n += 1
    return {'n_pairs_compared': n, 'n_pairs_stored': len(doc['pairs']),
            'max_abs_diff': float(max(diffs)) if diffs else None}


# ---------------------------------------------------------------------------------------------- formatting
def f4(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 'n/a'
    return f'{x:.4f}'


def md_table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    for r in rows:
        lines.append('| ' + ' | '.join(c if isinstance(c, str) else (f4(c) if isinstance(c, float) else str(c))
                                        for c in r) + ' |')
    return '\n'.join(lines) + '\n'


def csv_text(fields, rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator='\n')
    w.writeheader()
    for r in rows:
        w.writerow({k: (round(v, 6) if isinstance(v, float) and not np.isnan(v) else ('' if isinstance(v, float) else v))
                    for k, v in r.items()})
    return buf.getvalue()


def json_clean(o):
    if isinstance(o, dict):
        return {str(k): json_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def identity_line(K, fold, data, seeds):
    fd = data[K]['folds'][fold]
    note = f' ({TEST_NOTE})' if fold == 'test' else ' (decision fold)'
    return f'K={K}, {fold} fold{note}: {fd["N"]} rows, {fd["S"]} patients, seeds {", ".join(map(str, seeds))}'


# ---------------------------------------------------------------------------------------------- main analysis
def build(args):
    methods = list(dict.fromkeys(args.methods))
    if CEI not in methods or len(methods) < 2:
        raise Refusal(f'--methods must contain {CEI} and at least one baseline')
    order = [CEI] + sorted(m for m in methods if m != CEI)   # canonical order (also the seed-pick order)
    baselines = order[1:]
    seeds, ks = list(args.seeds), list(args.ks)
    runs_dir = Path(args.runs)
    if not runs_dir.is_dir():
        raise Refusal(f'--runs is not a directory: {display_path(runs_dir)}')
    print(f'Loading {len(ks) * len(order) * len(seeds)} runs from {display_path(runs_dir)}', flush=True)
    data = load_all(runs_dir, order, ks, seeds)

    parity_rows, bad = parity_check(data, order, seeds)
    if bad:
        for r in bad:
            print(f"PARITY FAIL K={r['K']} {r['fold']} {r['method']} seed {r['seed']}: macro-F1 "
                  f"{r['macro_f1_recomputed']:.6f} vs stored {r['macro_f1_stored']}, accuracy "
                  f"{r['accuracy_recomputed']:.6f} vs stored {r['accuracy_stored']}", file=sys.stderr)
        raise Refusal(f'{len(bad)} of {len(parity_rows)} metric checks differ from result.json by more than {TOL}')
    max_diff = max(max(r['macro_f1_abs_diff'], r['accuracy_abs_diff']) for r in parity_rows)
    print(f'Parity: {len(parity_rows)} run/fold checks within {TOL} of result.json (max diff {max_diff:.2e})', flush=True)
    parity = {'tolerance': TOL, 'n_checks': len(parity_rows), 'all_ok': True, 'max_abs_diff': max_diff,
              'definition': 'macro_f1 = f1_score(y, argmax, labels=arange(K), average=macro, zero_division=0), also '
                            'from the KxK confusion matrix; accuracy = mean(argmax == y)',
              'runs': parity_rows, 'compare_json': {}}

    def met(K, fold, m, s):
        return data[K]['metrics'][(fold, m, s)]

    # ---- per-seed and summary tables
    per_seed, summary = [], []
    for K in ks:
        for m in order:
            for s in seeds:
                v, t = met(K, 'validation', m, s), met(K, 'test', m, s)
                per_seed.append({'K': K, 'method': m, 'seed': s,
                                 'validation_macro_f1': v['macro_f1'], 'validation_accuracy': v['accuracy'],
                                 'test_macro_f1': t['macro_f1'], 'test_accuracy': t['accuracy'],
                                 'test_balanced_accuracy': t['balanced_accuracy'],
                                 'selected_epoch': data[K]['stored'][(m, s)].get('selected_epoch'),
                                 'validation_rows': data[K]['folds']['validation']['N'],
                                 'test_rows': data[K]['folds']['test']['N']})
    val_mean = {}
    for K in ks:
        for m in order:
            val_mean[(K, m)] = mean_sd([met(K, 'validation', m, s)['macro_f1'] for s in seeds])[0]
    best_baseline = {K: max(baselines, key=lambda m: (val_mean[(K, m)], m)) for K in ks}
    for K in ks:
        for fold in FOLDS:
            fd = data[K]['folds'][fold]
            means = {m: mean_sd([met(K, fold, m, s)['macro_f1'] for s in seeds])[0] for m in order}
            ranked = sorted(order, key=lambda m: (-means[m], m))
            for m in order:
                f1m, f1s = mean_sd([met(K, fold, m, s)['macro_f1'] for s in seeds])
                am, asd = mean_sd([met(K, fold, m, s)['accuracy'] for s in seeds])
                bm, bsd = mean_sd([met(K, fold, m, s)['balanced_accuracy'] for s in seeds])
                summary.append({'K': K, 'fold': fold, 'method': m, 'n_seeds': len(seeds), 'rows': fd['N'],
                                'patients': fd['S'], 'macro_f1_mean': f1m, 'macro_f1_sd': f1s,
                                'accuracy_mean': am, 'accuracy_sd': asd, 'balanced_accuracy_mean': bm,
                                'balanced_accuracy_sd': bsd, 'rank_in_fold': ranked.index(m) + 1,
                                'best_baseline_by_validation': m == best_baseline[K]})

    # ---- bootstrap: patient-only (compare.py) and two-level, both folds
    uncertainty = []
    for K in ks:
        for fold in FOLDS:
            fd = data[K]['folds'][fold]
            keys = [(m, s) for m in order for s in seeds]
            print(f'Bootstrap K={K} {fold} ({fd["S"]} patients): patient-only B={args.b_patient}, '
                  f'two-level B={args.b_two_level}', flush=True)
            prb = patient_bootstrap(fd, K, keys, args.b_patient, PATIENT_RNG)
            meanboot = {m: np.mean([prb[(m, s)] for s in seeds], axis=0) for m in order}
            if args.b_two_level == args.b_patient:
                prb2 = prb
            else:
                prb2 = patient_bootstrap(fd, K, keys, args.b_two_level, PATIENT_RNG)
            two = two_level(prb2, order, seeds, K, args.b_two_level)
            point = {m: mean_sd([met(K, fold, m, s)['macro_f1'] for s in seeds])[0] for m in order}
            if fold == 'test' and args.compare_dir:
                chk = compare_json_check(args.compare_dir, K, meanboot)
                parity['compare_json'][f'K{K}'] = chk
                shown = 'no common pairs' if chk['max_abs_diff'] is None else f"{chk['max_abs_diff']:.3e}"
                print(f"  patient-only test intervals vs stored compare_top{K}.json: max abs difference {shown} "
                      f"({chk['n_pairs_compared']} of {chk['n_pairs_stored']} stored pairs)", flush=True)
            cmp_rows = [(b, meanboot[CEI] - meanboot[b], two[CEI] - two[b], point[CEI] - point[b]) for b in baselines]
            mean_po = np.mean([meanboot[b] for b in baselines], axis=0)
            mean_tl = np.mean([two[b] for b in baselines], axis=0)
            cmp_rows.append(('mean_of_baselines', meanboot[CEI] - mean_po, two[CEI] - mean_tl,
                             point[CEI] - float(np.mean([point[b] for b in baselines]))))
            for b, d_po, d_tl, obs in cmp_rows:
                po_lo, po_hi = interval(d_po, 0.05)
                tl_lo, tl_hi = interval(d_tl, 0.05)
                bf_lo, bf_hi = interval(d_tl, 0.05 / BONFERRONI_M)
                se_po, se_tl = float(d_po.std(ddof=1)), float(d_tl.std(ddof=1))
                uncertainty.append({
                    'K': K, 'fold': fold, 'baseline': b, 'rows': fd['N'], 'patients': fd['S'],
                    'seeds': len(seeds), 'observed_delta': obs,
                    'patient_only_lo': po_lo, 'patient_only_hi': po_hi, 'patient_only_p_delta_gt_0': float((d_po > 0).mean()),
                    'patient_only_se': se_po,
                    'two_level_lo': tl_lo, 'two_level_hi': tl_hi, 'two_level_p_delta_gt_0': float((d_tl > 0).mean()),
                    'two_level_se': se_tl, 'se_ratio_two_level_over_patient_only': se_tl / se_po,
                    'bonferroni_lo': bf_lo, 'bonferroni_hi': bf_hi,
                    'best_baseline_by_validation': b == best_baseline[K]})

    # ---- macro-F1 variants
    variants_run, variants = [], []
    for K in ks:
        for fold in FOLDS:
            fd = data[K]['folds'][fold]
            for m in order:
                for s in seeds:
                    r = met(K, fold, m, s)
                    variants_run.append({'K': K, 'fold': fold, 'method': m, 'seed': s,
                                         'fixed': r['fixed'], 'present': r['present'], 'ge30': r['ge30'],
                                         'n_present': r['n_present'], 'n_ge30': r['n_ge30'],
                                         'n_zero_support': K - r['n_present']})
            for var, ncls in (('fixed', lambda r: K), ('present', lambda r: r['n_present']), ('ge30', lambda r: r['n_ge30'])):
                stats = {m: mean_sd([met(K, fold, m, s)[var] for s in seeds]) for m in order}
                ranked = sorted(order, key=lambda m: (-stats[m][0], m))
                n_cls = ncls(met(K, fold, order[0], seeds[0]))
                for m in order:
                    vals = [met(K, fold, m, s)[var] for s in seeds]
                    variants.append({'K': K, 'fold': fold, 'variant': var, 'n_classes': n_cls, 'method': m,
                                     'mean': stats[m][0], 'sd': stats[m][1], 'rank': ranked.index(m) + 1,
                                     'per_seed': ' '.join(f'{v:.4f}' for v in vals), 'rows': fd['N'], 'patients': fd['S']})

    # ---- per-class tables
    pc_run, pc = [], []
    pc_f1 = {}      # (K, fold, m) -> mean F1 vector over seeds
    pc_sup = {}
    for K in ks:
        labels = data[K]['labels']
        for fold in FOLDS:
            for m in order:
                mats = {k: np.stack([met(K, fold, m, s)[k] for s in seeds]).astype(np.float64)
                        for k in ('precision', 'recall', 'f1', 'n_pred')}
                sup = met(K, fold, m, seeds[0])['support']
                for s in seeds:
                    r = met(K, fold, m, s)
                    if not np.array_equal(r['support'], sup):
                        raise Refusal(f'K={K} {fold}: class support differs between runs')
                    for i in range(K):
                        pc_run.append({'K': K, 'fold': fold, 'method': m, 'seed': s, 'class_index': i,
                                       'label': labels[i], 'support': int(sup[i]), 'n_pred': int(r['n_pred'][i]),
                                       'precision': float(r['precision'][i]), 'recall': float(r['recall'][i]),
                                       'f1': float(r['f1'][i])})
                pc_f1[(K, fold, m)] = mats['f1'].mean(axis=0)
                pc_sup[(K, fold)] = sup
                sd = lambda a: a.std(axis=0, ddof=1) if len(a) > 1 else np.full(a.shape[1], np.nan)
                for i in range(K):
                    pc.append({'K': K, 'fold': fold, 'method': m, 'class_index': i, 'label': labels[i],
                               'support': int(sup[i]), 'evaluable': bool(sup[i] >= MIN_SUPPORT),
                               'n_pred_mean': float(mats['n_pred'][:, i].mean()),
                               'precision_mean': float(mats['precision'][:, i].mean()), 'precision_sd': float(sd(mats['precision'])[i]),
                               'recall_mean': float(mats['recall'][:, i].mean()), 'recall_sd': float(sd(mats['recall'])[i]),
                               'f1_mean': float(mats['f1'][:, i].mean()), 'f1_sd': float(sd(mats['f1'])[i]),
                               'n_seeds': len(seeds)})

    # ---- class spread (best baseline chosen on VALIDATION, applied to both folds)
    spread = {}
    for K in ks:
        labels = data[K]['labels']
        bb = best_baseline[K]
        for fold in FOLDS:
            sup = pc_sup[(K, fold)]
            cei = pc_f1[(K, fold, CEI)]
            base_means = {b: float(pc_f1[(K, fold, b)].mean()) for b in baselines}
            delta_best = cei - pc_f1[(K, fold, bb)]
            stack = np.stack([pc_f1[(K, fold, b)] for b in baselines])
            delta_max = cei - stack.max(axis=0)
            delta_mean = cei - stack.mean(axis=0)
            ev = sup >= MIN_SUPPORT
            gap = float(delta_best.sum() / K)
            idx_order = np.argsort(-delta_best)
            top3 = idx_order[:3]
            share = float(delta_best[top3].sum() / K / gap) if gap != 0 else float('nan')
            pos = float(delta_best[delta_best > 0].sum() / K)
            share_pos = float(delta_best[top3].sum() / K / pos) if pos else float('nan')
            terciles = np.array_split(np.argsort(sup), 3)
            strata = []
            for j, ix in enumerate(terciles):
                strata.append({'tercile': j + 1, 'support_min': int(sup[ix].min()), 'support_max': int(sup[ix].max()),
                               'n_classes': int(len(ix)), 'mean_delta_vs_best_baseline': float(delta_best[ix].mean()),
                               'cei_mean_f1': float(cei[ix].mean()),
                               'best_baseline_mean_f1': float(pc_f1[(K, fold, bb)][ix].mean())})
            fd = data[K]['folds'][fold]
            spread[f'{fold}_K{K}'] = {
                'K': K, 'fold': fold, 'rows': fd['N'], 'patients': fd['S'], 'seeds': seeds,
                'use': TEST_NOTE if fold == 'test' else 'decision fold',
                'best_baseline_by_validation': bb, 'validation_macro_f1_mean_of_baselines': {b: val_mean[(K, b)] for b in baselines},
                'baseline_macro_means_in_fold': base_means, 'cei_macro_mean': float(cei.mean()),
                'macro_gap_cei_minus_best': gap,
                'macro_gap_cei_minus_mean_of_baselines': float(delta_mean.sum() / K),
                'n_classes': K, 'n_classes_evaluable': int(ev.sum()), 'n_classes_not_evaluable': int((~ev).sum()),
                'n_classes_cei_beats_best_baseline': int((delta_best > 0).sum()),
                'n_classes_cei_below_best_baseline': int((delta_best < 0).sum()),
                'n_classes_cei_beats_every_baseline': int((delta_max > 0).sum()),
                'n_classes_cei_below_some_baseline': int((delta_max < 0).sum()),
                'evaluable_n_classes_cei_beats_best_baseline': int(((delta_best > 0) & ev).sum()),
                'evaluable_n_classes_cei_beats_every_baseline': int(((delta_max > 0) & ev).sum()),
                'top3_classes': [{'label': labels[i], 'support': int(sup[i]), 'evaluable': bool(ev[i]),
                                  'cei_f1': float(cei[i]), 'best_baseline_f1': float(pc_f1[(K, fold, bb)][i]),
                                  'delta': float(delta_best[i])} for i in top3],
                'top3_share_of_macro_gap': share, 'top3_share_of_positive_deltas': share_pos,
                'bottom3_classes': [{'label': labels[i], 'support': int(sup[i]), 'evaluable': bool(ev[i]),
                                     'delta': float(delta_best[i])} for i in idx_order[-3:]],
                'support_terciles': strata,
                'per_class_delta': [{'label': labels[i], 'support': int(sup[i]), 'evaluable': bool(ev[i]),
                                     'delta_vs_best_baseline': float(delta_best[i]),
                                     'delta_vs_classwise_max': float(delta_max[i]),
                                     'delta_vs_mean_of_baselines': float(delta_mean[i])} for i in idx_order]}

    return {'order': order, 'baselines': baselines, 'seeds': seeds, 'ks': ks, 'data': data, 'parity': parity,
            'per_seed': per_seed, 'summary': summary, 'uncertainty': uncertainty, 'variants': variants,
            'variants_run': variants_run, 'pc_run': pc_run, 'pc': pc, 'spread': spread,
            'best_baseline': best_baseline, 'val_mean': val_mean, 'runs_dir': runs_dir}


# ---------------------------------------------------------------------------------------------- rendering
def render(res, args):
    order, baselines, seeds, ks, data = res['order'], res['baselines'], res['seeds'], res['ks'], res['data']
    files = {}
    ident = ('Final runs, protocol fair_v10t; seeds ' + ', '.join(map(str, seeds)) + '; methods '
             + ', '.join(pretty(m) for m in order) + '. Validation decides; ' + TEST_NOTE + '.\n')

    # per-seed
    fields = list(res['per_seed'][0].keys())
    files['per_seed.csv'] = csv_text(fields, res['per_seed'])
    md = '# Per-seed results\n\n' + ident + '\n'
    for K in ks:
        md += f'## K={K}\n\n' + ' / '.join(
            f'{fold}: {data[K]["folds"][fold]["N"]} rows, {data[K]["folds"][fold]["S"]} patients' for fold in FOLDS) + '\n\n'
        md += md_table(['method', 'seed', 'val macro-F1', 'val acc', 'test macro-F1 (reported only)', 'test acc',
                        'test bal. acc', 'selected epoch'],
                       [[pretty(r['method']), str(r['seed']), r['validation_macro_f1'], r['validation_accuracy'],
                         r['test_macro_f1'], r['test_accuracy'], r['test_balanced_accuracy'], str(r['selected_epoch'])]
                        for r in res['per_seed'] if r['K'] == K])
        md += '\n'
    files['per_seed.md'] = md

    # summary
    files['summary.csv'] = csv_text(list(res['summary'][0].keys()), res['summary'])
    md = '# Summary over seeds (mean and SD, ddof=1)\n\n' + ident + '\n'
    for K in ks:
        for fold in FOLDS:
            md += f'## {identity_line(K, fold, data, seeds)}\n\n'
            md += md_table(['method', 'macro-F1', 'SD', 'accuracy', 'SD', 'bal. acc', 'SD', 'rank', 'best baseline (validation)'],
                           [[pretty(r['method']), r['macro_f1_mean'], r['macro_f1_sd'], r['accuracy_mean'], r['accuracy_sd'],
                             r['balanced_accuracy_mean'], r['balanced_accuracy_sd'], str(r['rank_in_fold']),
                             'yes' if r['best_baseline_by_validation'] else ''] for r in res['summary']
                            if r['K'] == K and r['fold'] == fold])
            md += '\n'
    files['summary.md'] = md

    # uncertainty
    files['uncertainty.csv'] = csv_text(list(res['uncertainty'][0].keys()), res['uncertainty'])
    md = ('# CEI-GNN mp2 against each baseline: patient-only and two-level intervals\n\n' + ident
          + f'\nDelta = CEI-GNN mp2 minus baseline in macro-F1 (mean over seeds). Patient-only: the interval of '
            f'comparison/top3/compare.py (B={args.b_patient}, rng {PATIENT_RNG}, resampling patients, seeds fixed). '
            f'Two-level: seeds resampled per method (level 1) and the same patient draw (level 2), B={args.b_two_level}. '
            f'SE ratio = two-level SE / patient-only SE. Bonferroni: two-level interval at level 1 - 0.05/{BONFERRONI_M} '
            f'({BONFERRONI_M} = 3 baselines x 3 K), labelled as such.\n\n')
    for K in ks:
        for fold in FOLDS:
            md += f'## {identity_line(K, fold, data, seeds)}\n\n'
            md += md_table(['baseline', 'delta', 'patient-only 95% CI', 'P(d>0)', 'two-level 95% CI', 'P(d>0)', 'SE ratio',
                            f'Bonferroni CI (1-0.05/{BONFERRONI_M})'],
                           [[pretty(r['baseline']) + (' (best by validation)' if r['best_baseline_by_validation'] else '')
                             if r['baseline'] != 'mean_of_baselines' else 'mean of the baselines',
                             r['observed_delta'], f"[{r['patient_only_lo']:+.4f}, {r['patient_only_hi']:+.4f}]",
                             r['patient_only_p_delta_gt_0'], f"[{r['two_level_lo']:+.4f}, {r['two_level_hi']:+.4f}]",
                             r['two_level_p_delta_gt_0'], r['se_ratio_two_level_over_patient_only'],
                             f"[{r['bonferroni_lo']:+.4f}, {r['bonferroni_hi']:+.4f}]"]
                            for r in res['uncertainty'] if r['K'] == K and r['fold'] == fold])
            md += '\n'
    files['uncertainty.md'] = md

    # variants
    files['macro_f1_variants.csv'] = csv_text(list(res['variants'][0].keys()), res['variants'])
    files['macro_f1_variants_per_run.csv'] = csv_text(list(res['variants_run'][0].keys()), res['variants_run'])
    md = ('# Macro-F1 variants\n\n' + ident + '\nfixed = mean F1 over all K classes (the reported metric, primary); '
          f'present = mean F1 over classes with at least 1 true row in the fold; ge30 = mean F1 over classes with at least '
          f'{MIN_SUPPORT} true rows in the fold. Mean +- SD over seeds, per-seed values in brackets; ranking best first.\n\n')
    for K in ks:
        for fold in FOLDS:
            md += f'## {identity_line(K, fold, data, seeds)}\n\n'
            rows = []
            for var in ('fixed', 'present', 'ge30'):
                vs = [v for v in res['variants'] if v['K'] == K and v['fold'] == fold and v['variant'] == var]
                cells = [f"{v['mean']:.4f} +- {v['sd']:.4f} [{v['per_seed'].replace(' ', ', ')}]" for v in vs]
                rank = ' > '.join(pretty(v['method']) for v in sorted(vs, key=lambda v: v['rank']))
                rows.append([var, str(vs[0]['n_classes'])] + cells + [rank])
            md += md_table(['variant', 'n classes'] + [pretty(m) for m in order] + ['ranking'], rows) + '\n'
    files['macro_f1_variants.md'] = md

    # per-class
    files['per_class.csv'] = csv_text(list(res['pc'][0].keys()), res['pc'])
    files['per_class_per_run.csv'] = csv_text(list(res['pc_run'][0].keys()), res['pc_run'])
    if 30 in ks:
        files['per_class_k30_test.md'] = k30_test_md(res, args)

    # spread
    files['class_spread.json'] = json.dumps(json_clean(res['spread']), indent=1) + '\n'
    md = ('# Class spread of the CEI-GNN mp2 gain\n\n' + ident + f'\nBest baseline per K = highest mean VALIDATION macro-F1 '
          f'(applied to both folds). Classes with support < {MIN_SUPPORT} in the fold are not evaluable. Top-3 share = '
          f'share of the macro-F1 gap carried by the 3 classes with the largest CEI minus best-baseline F1 delta.\n\n')
    rows = []
    for key, v in res['spread'].items():
        rows.append([str(v['K']), v['fold'], f"{v['rows']} / {v['patients']}", pretty(v['best_baseline_by_validation']),
                     v['macro_gap_cei_minus_best'], v['macro_gap_cei_minus_mean_of_baselines'],
                     f"{v['n_classes_cei_beats_best_baseline']} of {v['n_classes']}",
                     f"{v['n_classes_cei_beats_every_baseline']} of {v['n_classes']}",
                     f"{v['n_classes_not_evaluable']}", v['top3_share_of_macro_gap']])
    md += md_table(['K', 'fold', 'rows / patients', 'best baseline (validation)', 'macro gap vs best', 'macro gap vs mean of baselines',
                    'classes CEI > best', 'classes CEI > every baseline', 'not evaluable', 'top-3 share of gap'], rows) + '\n'
    for key, v in res['spread'].items():
        md += f"## K={v['K']}, {v['fold']} fold ({v['use']}): top-3 classes and support terciles\n\n"
        md += md_table(['class', 'support', 'CEI F1', 'best baseline F1', 'delta'],
                       [[t['label'] + ('' if t['evaluable'] else ' (not evaluable)'), str(t['support']), t['cei_f1'],
                         t['best_baseline_f1'], t['delta']] for t in v['top3_classes']]) + '\n'
        md += md_table(['support tercile', 'support range', 'classes', 'CEI mean F1', 'best baseline mean F1', 'mean delta'],
                       [[str(t['tercile']), f"{t['support_min']}-{t['support_max']}", str(t['n_classes']), t['cei_mean_f1'],
                         t['best_baseline_mean_f1'], t['mean_delta_vs_best_baseline']] for t in v['support_terciles']]) + '\n'
    files['class_spread.md'] = md

    files['parity.json'] = json.dumps(json_clean(res['parity']), indent=1) + '\n'
    files['summary.json'] = json.dumps(json_clean(summary_doc(res, args)), indent=1) + '\n'
    return files


def k30_test_md(res, args):
    data, seeds, order, baselines = res['data'], res['seeds'], res['order'], res['baselines']
    K, fold = 30, 'test'
    bb = res['best_baseline'][K]
    others = [b for b in baselines if b != bb]
    rows_pc = {}
    for r in res['pc']:
        if r['K'] == K and r['fold'] == fold:
            rows_pc.setdefault(r['class_index'], {})[r['method']] = r
    md = ('# Per-class results, K=30, test fold (reported only, not used for decisions)\n\n'
          + identity_line(K, fold, data, seeds) + f'. F1 mean (SD over seeds); P/R = precision/recall mean. Compared baseline = '
          f'{pretty(bb)} (highest mean validation macro-F1). Classes with support < {MIN_SUPPORT} are not evaluable.\n\n')
    header = ['class', 'support', 'CEI P/R', 'CEI F1', f'{pretty(bb)} P/R', f'{pretty(bb)} F1'] + \
             [f'{pretty(b)} F1' for b in others] + ['CEI - best']
    cell = lambda r: f"{r['f1_mean']:.4f} ({r['f1_sd']:.4f})"
    pr = lambda r: f"{r['precision_mean']:.4f}/{r['recall_mean']:.4f}"
    body = []
    for i in sorted(rows_pc, key=lambda i: (-rows_pc[i][CEI]['support'], i)):
        d = rows_pc[i]
        c, b = d[CEI], d[bb]
        name = c['label'] + ('' if c['evaluable'] else ' (not evaluable)')
        body.append([name, str(c['support']), pr(c), cell(c), pr(b), cell(b)] + [cell(d[o]) for o in others]
                    + [f"{c['f1_mean'] - b['f1_mean']:+.4f}"])
    macro = lambda m: float(np.mean([rows_pc[i][m]['f1_mean'] for i in rows_pc]))
    body.append(['macro (fixed)', '', '', f4(macro(CEI)), '', f4(macro(bb))] + [f4(macro(o)) for o in others]
                + [f'{macro(CEI) - macro(bb):+.4f}'])
    return md + md_table(header, body)


def summary_doc(res, args):
    data = res['data']
    return {
        'protocol': 'fair_v10t', 'test_note': TEST_NOTE,
        'runs_dir': display_path(res['runs_dir']),
        'methods': res['order'], 'cei': CEI, 'baselines': res['baselines'], 'seeds': res['seeds'], 'ks': res['ks'],
        'task_identity': {f'K{K}': {'n_classes': K, 'class_labels': data[K]['labels'],
                                    **{fold: {'rows': data[K]['folds'][fold]['N'], 'patients': data[K]['folds'][fold]['S']}
                                       for fold in FOLDS}} for K in res['ks']},
        'best_baseline_by_validation': {f'K{K}': res['best_baseline'][K] for K in res['ks']},
        'validation_macro_f1_mean': {f'K{K}': {m: res['val_mean'][(K, m)] for m in res['order']} for K in res['ks']},
        'bootstrap': {
            'patient_only': {'B': args.b_patient, 'rng': f'numpy default_rng({PATIENT_RNG})', 'source': 'comparison/top3/compare.py',
                             'unit': 'unique subjects with replacement, row weight = multiplicity, one draw shared by all runs of a K',
                             'statistic': 'mean over seeds of the per-seed macro-F1 on the resampled confusion matrix'},
            'two_level': {'B': args.b_two_level, 'patient_rng': f'numpy default_rng({PATIENT_RNG})',
                          'seed_pick_rng': f'numpy default_rng({TWO_LEVEL_RNG_BASE} + K)',
                          'seed_pick_shape': '[B, methods (cei_mp2 first, then alphabetical), seeds]',
                          'definition': 'level 1 resamples the seeds with replacement independently per method, level 2 uses the '
                                        'shared patient draw; statistic = mean over the resampled seeds of the per-run bootstrap macro-F1'},
            'interval': '95%% percentile; Bonferroni column at level 1 - 0.05/%d' % BONFERRONI_M},
        'macro_f1': 'f1_score(y, argmax, labels=arange(K), average=macro, zero_division=0); variants present (support >= 1) and ge30 (support >= 30)',
        'min_support_evaluable': MIN_SUPPORT,
        'parity': {'tolerance': TOL, 'all_ok': res['parity']['all_ok'], 'max_abs_diff': res['parity']['max_abs_diff'],
                   'n_checks': res['parity']['n_checks'], 'compare_json': res['parity']['compare_json']},
        'outputs': sorted(['per_seed.csv', 'per_seed.md', 'summary.csv', 'summary.md', 'uncertainty.csv', 'uncertainty.md',
                           'macro_f1_variants.csv', 'macro_f1_variants.md', 'macro_f1_variants_per_run.csv', 'per_class.csv',
                           'per_class_per_run.csv', 'per_class_k30_test.md', 'class_spread.json', 'class_spread.md',
                           'parity.json', 'summary.json']),
    }


def parse_ints(s):
    return tuple(int(x) for x in s.split(',') if x)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='comparison.s6_stats.stats', description=__doc__.split('\n\n')[0])
    ap.add_argument('--runs', required=True, help='directory with top<K>/<method>_seed<S>/ run folders')
    ap.add_argument('--out', required=True, help='NEW output directory (must not exist)')
    ap.add_argument('--methods', default=','.join(DEFAULT_METHODS),
                    help='comma list; default excludes gchm_pna (pass it only for parity checks with older numbers)')
    ap.add_argument('--ks', default=','.join(map(str, KS))),
    ap.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    ap.add_argument('--compare-dir', default=None, help='folder with compare_top<K>.json: print the max difference of the '
                                                         'patient-only test intervals to the stored values')
    ap.add_argument('--b-patient', type=int, default=N_PATIENT_BOOT)
    ap.add_argument('--b-two-level', type=int, default=N_TWO_LEVEL_BOOT)
    args = ap.parse_args(argv)
    args.methods = [m for m in args.methods.split(',') if m]
    args.ks, args.seeds = parse_ints(args.ks), parse_ints(args.seeds)
    out = Path(args.out)
    if out.exists():
        print(f'REFUSED: --out already exists: {display_path(out)}', file=sys.stderr)
        return 2
    try:
        res = build(args)
        files = render(res, args)
    except (Refusal, FileNotFoundError) as e:
        print(f'REFUSED: {e}', file=sys.stderr)
        return 2
    out.mkdir(parents=True)
    for name, text in files.items():
        (out / name).write_text(text)
    print(f'Wrote {len(files)} files to {display_path(out)}')
    for K in res['ks']:
        print(f"K={K}: best baseline by validation macro-F1 = {pretty(res['best_baseline'][K])}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
