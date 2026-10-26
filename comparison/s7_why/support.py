"""Class-support analysis: per-class CEI-minus-baseline deltas against train support.

Reads only stored predictions (validation.npz, test.npz), binding.json and result.json of the final runs.
No checkpoints and no graphs are loaded.

Decisions (checkpoint, HPO, baseline choice) use VALIDATION; every reported result is on TEST (scored once per run
at the validation-selected checkpoint). The best baseline of a K is the baseline with the highest mean validation
macro-F1 over seeds (chosen on validation). Tables and figure panels list the test fold first, validation second.

Definitions
  per-class F1      2*TP / (support + predicted), 0 when the denominator is 0 (as in comparison/top3/compare.py).
  macro-F1          mean per-class F1 over the classes with support > 0 in the fold (all K classes in practice).
  seed SD           sample SD (ddof=1) of the per-class F1 over the seeds.
  SD of a delta     sqrt(sdA^2/n + sdB^2/n) for independent seeds (n = number of seeds); for the
                    mean of m baselines: sqrt(sdA^2/n + sum_b sd_b^2/(n*m^2)).
  bootstrap         resampling of subjects (patients) of the fold with replacement; per-class F1 is
                    recomputed with the resampled row weights; percentile 95% interval.

Usage
  python -m comparison.s7_why.support --runs <runs root> --out <new dir> [--methods cei_mp2,gsat,graphcare,protgnn] [--b 1000]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
from scipy import stats

DEFAULT_METHODS = ['cei_mp2', 'gsat', 'graphcare', 'protgnn']
FOCUS = 'cei_mp2'
SEEDS = [1234, 2025, 7]
KS = [10, 20, 30]
FOLDS = ['validation', 'test']          # computation order (one bootstrap stream is shared across folds); do not reorder
REPORT_FOLDS = ['test', 'validation']   # presentation order: reported fold first, decision fold second
BOOT_SEED = 20261017
MIN_ROWS = 30


class Refusal(SystemExit):
    pass


def refuse(msg):
    raise Refusal('REFUSED: ' + msg)


# ----------------------------------------------------------------------------- loading

def load_k(runs, K, methods):
    """binding + validation/test predictions for every (method, seed) of one K."""
    out = {'pred': {}, 'folds': {}}
    ref_binding = None
    for m in methods:
        for s in SEEDS:
            d = os.path.join(runs, 'top%d' % K, '%s_seed%d' % (m, s))
            if not os.path.isdir(d):
                refuse('missing run directory top%d/%s_seed%d' % (K, m, s))
            b = json.load(open(os.path.join(d, 'binding.json')))
            key = (list(b['label_order']), [int(x) for x in b['class_counts_train']])
            if ref_binding is None:
                ref_binding = key
            elif key != ref_binding:
                refuse('label_order/class_counts_train differ across runs of K=%d (%s_seed%d)' % (K, m, s))
    out['labels'], tr = ref_binding
    out['train'] = np.array(tr)
    ref = {}
    for fold in FOLDS:
        out['pred'][fold] = {m: [] for m in methods}
        for m in methods:
            for s in SEEDS:
                z = np.load(os.path.join(runs, 'top%d' % K, '%s_seed%d' % (m, s), fold + '.npz'), allow_pickle=True)
                if fold not in ref:
                    ref[fold] = z
                    out['folds'][fold] = {'y': z['y'], 'subjects': z['subjects']}
                elif not (np.array_equal(z['sample_ids'], ref[fold]['sample_ids']) and np.array_equal(z['y'], ref[fold]['y'])):
                    refuse('rows differ across runs: K=%d %s %s_seed%d' % (K, fold, m, s))
                out['pred'][fold][m].append(z['proba'].argmax(1))
        out['pred'][fold] = {m: np.array(v) for m, v in out['pred'][fold].items()}
    return out


def f1_per_class(y, p, K, w=None):
    if w is None:
        w = np.ones(len(y))
    hit = y == p
    tp = np.bincount(y[hit], weights=w[hit], minlength=K)
    sup = np.bincount(y, weights=w, minlength=K)
    pc = np.bincount(p, weights=w, minlength=K)
    den = sup + pc
    return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0), sup


def seed_f1(y, preds, K, w=None):
    """(n_seeds, K) per-class F1 and (K,) support."""
    F = []
    sup = None
    for p in preds:
        f, sup = f1_per_class(y, p, K, w)
        F.append(f)
    return np.array(F), sup


# ----------------------------------------------------------------------------- helpers

def sd_delta(sd_a, sd_b_list, n):
    m = len(sd_b_list)
    return np.sqrt(sd_a ** 2 / n + sum(s ** 2 for s in sd_b_list) / (n * m * m))


def rnd(x, nd=6):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def spearman(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float('nan')
    return float(stats.spearmanr(x, y)[0])


def pct(a):
    a = np.asarray(a, dtype=float)
    return float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))


def report_order(rows):
    """Stable sort of rows by presentation order of the fold (test first)."""
    return sorted(rows, key=lambda r: REPORT_FOLDS.index(r['fold']))


def write_csv(path, rows, fields):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in fields})


def write_md(path, title, note, rows, fields):
    with open(path, 'w') as f:
        f.write('# %s\n\n%s\n\n' % (title, note))
        f.write('| ' + ' | '.join(fields) + ' |\n|' + '---|' * len(fields) + '\n')
        for r in rows:
            f.write('| ' + ' | '.join('' if r.get(k) is None else str(r.get(k)) for k in fields) + ' |\n')


def ci_str(lo, hi):
    return '[%.4f, %.4f]' % (lo, hi)


# ----------------------------------------------------------------------------- per K / fold stats

def mean_f1(D, K, fold, methods):
    y = D[K]['folds'][fold]['y']
    F = {}
    for m in methods:
        F[m], sup = seed_f1(y, D[K]['pred'][fold][m], K)
    return F, sup


def choose_best(D, methods, baselines):
    """best baseline per K: highest mean validation macro-F1 (never test)."""
    best, val_means = {}, {}
    for K in KS:
        F, sup = mean_f1(D, K, 'validation', methods)
        keep = sup > 0
        # seed-mean of the macro-F1 (identical to the macro of the seed-mean per-class F1)
        val_means[K] = {m: float(F[m][:, keep].mean()) for m in methods}
        best[K] = max(baselines, key=lambda m: val_means[K][m])
    return best, val_means


def analyse(D, methods, baselines, best, B, log):
    n = len(SEEDS)
    rows, terc_rows, sp_rows = [], [], []
    summary_fold = {}
    rng = np.random.default_rng(BOOT_SEED)
    for fold in FOLDS:
        for K in KS:
            y = D[K]['folds'][fold]['y']
            subj = D[K]['folds'][fold]['subjects']
            tr = D[K]['train']
            F, sup = mean_f1(D, K, fold, methods)
            keep = sup > 0
            Fm = {m: F[m].mean(0) for m in methods}
            Fs = {m: F[m].std(0, ddof=1) for m in methods}
            bb = best[K]
            mb = np.mean([Fm[m] for m in baselines], 0)
            d_best = Fm[FOCUS] - Fm[bb]
            d_mean = Fm[FOCUS] - mb
            sd_best = sd_delta(Fs[FOCUS], [Fs[bb]], n)
            sd_mean = sd_delta(Fs[FOCUS], [Fs[m] for m in baselines], n)
            for c in range(K):
                r = {'fold': fold, 'K': K, 'class': D[K]['labels'][c], 'train_support': int(tr[c]),
                     'eval_support': int(sup[c])}
                for m in methods:
                    r['f1_' + m] = rnd(Fm[m][c])
                    r['f1sd_' + m] = rnd(Fs[m][c])
                r['best_baseline'] = bb
                r['delta_best'] = rnd(d_best[c])
                r['delta_best_sd'] = rnd(sd_best[c])
                r['delta_mean'] = rnd(d_mean[c])
                r['delta_mean_sd'] = rnd(sd_mean[c])
                for m in baselines:
                    r['delta_vs_' + m] = rnd(Fm[FOCUS][c] - Fm[m][c])
                rows.append(r)

            # ---- Spearman vs log10(train support) with subject bootstrap
            lt = np.log10(tr.astype(float))
            ge = sup >= MIN_ROWS
            subsets = {'all': keep, 'ge%d' % MIN_ROWS: ge}
            pt = {}
            for dn, dv in (('delta_best', d_best), ('delta_mean', d_mean)):
                for sn, mk in subsets.items():
                    pt[(dn, sn)] = spearman(lt[mk], dv[mk])
            us, inv = np.unique(subj, return_inverse=True)
            boots = {k: [] for k in pt}
            for b in range(B):
                cnt = np.bincount(rng.integers(0, len(us), len(us)), minlength=len(us))
                w = cnt[inv].astype(float)
                Fb, supb = {}, None
                for m in methods:
                    fb, supb = seed_f1(y, D[K]['pred'][fold][m], K, w)
                    Fb[m] = fb.mean(0)
                mbb = np.mean([Fb[m] for m in baselines], 0)
                dvs = {'delta_best': Fb[FOCUS] - Fb[bb], 'delta_mean': Fb[FOCUS] - mbb}
                for dn, dv in dvs.items():
                    for sn, mk in subsets.items():
                        mk2 = (supb > 0) if sn == 'all' else mk
                        boots[(dn, sn)].append(spearman(lt[mk2], dv[mk2]))
            for (dn, sn), v in pt.items():
                lo, hi = pct(boots[(dn, sn)])
                sp_rows.append({'fold': fold, 'K': K, 'delta': dn, 'classes': sn,
                                'n_classes': int(subsets[sn].sum()), 'rho': rnd(v, 4),
                                'ci_lo': rnd(lo, 4), 'ci_hi': rnd(hi, 4), 'B': B,
                                'n_subjects': int(len(us)), 'n_rows': int(len(y))})
            log('  %s K=%d rho(delta_mean, all)=%.3f %s' % (fold, K, pt[('delta_mean', 'all')],
                ci_str(*pct(boots[('delta_mean', 'all')]))))

            # ---- terciles of train support (classes with eval support > 0)
            idx = np.where(keep)[0][np.argsort(tr[keep], kind='stable')]
            for name, ids in zip(('low', 'mid', 'high'), np.array_split(idx, 3)):
                terc_rows.append({
                    'fold': fold, 'K': K, 'tercile': name, 'n_classes': int(len(ids)),
                    'train_support_min': int(tr[ids].min()), 'train_support_max': int(tr[ids].max()),
                    'f1_cei': rnd(Fm[FOCUS][ids].mean(), 4), 'f1_mean_baselines': rnd(mb[ids].mean(), 4),
                    'f1_best_baseline': rnd(Fm[bb][ids].mean(), 4),
                    'delta_best': rnd(d_best[ids].mean(), 4), 'delta_mean': rnd(d_mean[ids].mean(), 4),
                    'n_cei_above_best': int((d_best[ids] > 0).sum()), 'best_baseline': bb})
            summary_fold[(fold, K)] = {
                'n_rows': int(len(y)), 'n_patients': int(len(us)), 'n_classes_with_support': int(keep.sum()),
                'n_classes_ge%d_rows' % MIN_ROWS: int(ge.sum()),
                'macro_f1_by_method': {m: rnd(Fm[m][keep].mean(), 4) for m in methods},
                'macro_f1_fixed_k_by_method': {m: rnd(Fm[m].mean(), 4) for m in methods},
                'strongest_baseline_in_fold': max(baselines, key=lambda m: float(Fm[m][keep].mean()))}
    return rows, sp_rows, terc_rows, summary_fold


def same_classes(D, methods, baselines, best, B, log):
    """K-way macro-F1 restricted to fixed class sets, per K, with subject-bootstrap CIs."""
    n_all = len(KS)
    labels = {K: D[K]['labels'] for K in KS}
    for a, b in ((10, 20), (20, 30)):
        if not set(labels[a]) <= set(labels[b]):
            refuse('class set of K=%d is not contained in K=%d' % (a, b))
    sets = {'top10': (labels[10], [10, 20, 30]), 'top20': (labels[20], [20, 30]), 'all': (None, [10, 20, 30])}
    rows = []
    rng = np.random.default_rng(BOOT_SEED)
    for fold in FOLDS:
        allsubj = np.unique(np.concatenate([D[K]['folds'][fold]['subjects'] for K in KS]))
        sidx = {K: np.searchsorted(allsubj, D[K]['folds'][fold]['subjects']) for K in KS}
        cidx = {}
        for sn, (names, ks) in sets.items():
            for K in ks:
                cidx[(sn, K)] = np.arange(K) if names is None else np.array([labels[K].index(c) for c in names])

        def stat(wmap):
            """{(set, K): {method: seed-mean macro-F1 of the set's classes in the K-way task}}"""
            res = {}
            fc = {}
            for K in KS:
                y = D[K]['folds'][fold]['y']
                w = wmap[sidx[K]]
                fc[K] = {m: seed_f1(y, D[K]['pred'][fold][m], K, w)[0].mean(0) for m in methods}
            for (sn, K), ci in cidx.items():
                res[(sn, K)] = {m: float(fc[K][m][ci].mean()) for m in methods}
            return res

        base = stat(np.ones(len(allsubj)))
        boots = [stat(np.bincount(rng.integers(0, len(allsubj), len(allsubj)), minlength=len(allsubj)).astype(float))
                 for _ in range(B)]
        npatients = {K: int(len(np.unique(D[K]['folds'][fold]['subjects']))) for K in KS}
        for sn, (names, ks) in sets.items():
            first = ks[0]
            for K in ks:
                bb = best[K]
                v = base[(sn, K)]
                db = v[FOCUS] - v[bb]
                dm = v[FOCUS] - np.mean([v[m] for m in baselines])
                bdb = np.array([x[(sn, K)][FOCUS] - x[(sn, K)][bb] for x in boots])
                bdm = np.array([x[(sn, K)][FOCUS] - np.mean([x[(sn, K)][m] for m in baselines]) for x in boots])
                r = {'fold': fold, 'class_set': sn, 'K': K, 'n_classes_in_set': int(len(cidx[(sn, K)])),
                     'n_rows_fold': int(len(D[K]['folds'][fold]['y'])), 'n_patients_fold': npatients[K],
                     'f1_cei': rnd(v[FOCUS], 4), 'f1_mean_baselines': rnd(np.mean([v[m] for m in baselines]), 4),
                     'f1_best_baseline': rnd(v[bb], 4), 'best_baseline': bb,
                     'delta_mean': rnd(dm, 4), 'delta_mean_ci_lo': rnd(pct(bdm)[0], 4), 'delta_mean_ci_hi': rnd(pct(bdm)[1], 4),
                     'delta_best': rnd(db, 4), 'delta_best_ci_lo': rnd(pct(bdb)[0], 4), 'delta_best_ci_hi': rnd(pct(bdb)[1], 4),
                     'B': B}
                for m in baselines:
                    r['delta_vs_' + m] = rnd(v[FOCUS] - v[m], 4)
                if K != first:
                    v0 = base[(sn, first)]
                    for nm, cur, f0 in (
                            ('delta_mean', dm, v0[FOCUS] - np.mean([v0[m] for m in baselines])),
                            ('delta_best', db, v0[FOCUS] - v0[bb])):
                        bd = np.array([(x[(sn, K)][FOCUS] - (np.mean([x[(sn, K)][m] for m in baselines]) if nm == 'delta_mean' else x[(sn, K)][bb]))
                                       - (x[(sn, first)][FOCUS] - (np.mean([x[(sn, first)][m] for m in baselines]) if nm == 'delta_mean' else x[(sn, first)][bb]))
                                       for x in boots])
                        lo, hi = pct(bd)
                        r['%s_change_vs_K%d' % (nm, first)] = rnd(cur - f0, 4)
                        r['%s_change_ci_lo' % nm] = rnd(lo, 4)
                        r['%s_change_ci_hi' % nm] = rnd(hi, 4)
                rows.append(r)
        log('  same-classes %s done' % fold)
    return rows


# ----------------------------------------------------------------------------- figures

def figures(rows, sp_rows, out, methods):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rho = {(r['fold'], r['K']): r for r in sp_rows if r['delta'] == 'delta_mean' and r['classes'] == 'all'}
    for K in KS:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, fold in zip(axes, REPORT_FOLDS):
            rr = [r for r in rows if r['fold'] == fold and r['K'] == K and r['eval_support'] > 0]
            x = np.log10([r['train_support'] for r in rr])
            ax.errorbar(x, [r['delta_mean'] for r in rr], yerr=[r['delta_mean_sd'] for r in rr], fmt='o', ms=4,
                        elinewidth=0.8, capsize=2, color='#1f4e79', ecolor='#8aa6c1')
            ax.axhline(0, color='black', lw=0.8)
            s = rho[(fold, K)]
            name = 'validation (decision fold)' if fold == 'validation' else 'test'
            ax.set_title('%s: rho=%.2f [%.2f, %.2f]' % (name, s['rho'], s['ci_lo'], s['ci_hi']), fontsize=10)
            ax.set_xlabel('log10 train support (rows per class)')
        axes[0].set_ylabel('CEI minus mean of baselines (F1, seed SD bars)')
        fig.suptitle('Top-%d: per-class delta vs train support (baselines: %s)' % (K, ', '.join(m for m in methods if m != FOCUS)), fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'fig_support_k%d.png' % K), dpi=150)
        plt.close(fig)


# ----------------------------------------------------------------------------- parity (optional)

def parity(ref_dir, rows, same_rows, methods, log):
    """Compare with a reference per_class_delta_vs_support.csv / partAC_results.json (same methods)."""
    ref = {}
    with open(os.path.join(ref_dir, 'per_class_delta_vs_support.csv')) as f:
        for r in csv.DictReader(f):
            ref[(r['fold'], int(r['K']), r['class'])] = r
    mine = {(r['fold'], r['K'], r['class']): r for r in rows}
    colmap = {'train_support': 'train_support', 'eval_support': 'eval_support', 'delta_best': 'delta_vs_best',
              'delta_mean': 'delta_vs_mean_of_4'}
    short = {'cei_mp2': 'cei', 'gchm_pna': 'gchm'}
    for m in methods:
        colmap['f1_' + m] = 'f1_' + short.get(m, m)
        if m != FOCUS:
            colmap['delta_vs_' + m] = 'delta_vs_' + short.get(m, m)
    worst = {}
    out = {}
    miss = sum(1 for k in ref if k not in mine)
    for k, rr in ref.items():
        if k not in mine:
            continue
        for mc, rc in colmap.items():
            if rc not in rr:
                continue
            dif = abs(float(mine[k][mc]) - float(rr[rc]))
            key = (k[0], k[1], mc)
            worst[key] = max(worst.get(key, 0.0), dif)
    log('parity per-class (max abs diff per fold/K/column; reference is rounded to 4 d.p.; rows missing: %d)' % miss)
    for fold in FOLDS:
        for K in KS:
            line = ' '.join('%s=%.6f' % (mc, worst.get((fold, K, mc), float('nan'))) for mc in colmap)
            log('  %s K=%d %s' % (fold, K, line))
    out['per_class_max_abs_diff'] = max(worst.values()) if worst else None
    out['per_class_max_by_key'] = {'%s_K%d_%s' % k: v for k, v in worst.items()}
    pc = json.load(open(os.path.join(ref_dir, 'partAC_results.json')))['partC']
    refkey = {'top10': 'top10', 'top20': 'top20_old'}
    rows_by = {(r['fold'], r['class_set'], r['K']): r for r in same_rows}
    sworst = {}
    for fold in FOLDS:
        for sn, rs in refkey.items():
            tab = pc['%s_%s_kway' % (fold, rs)]
            for Ks, e in tab.items():
                r = rows_by[(fold, sn, int(Ks))]
                pairs = [('f1_cei', 'cei'), ('f1_mean_baselines', 'meanB'), ('f1_best_baseline', 'best'),
                         ('delta_best', 'd_best'), ('delta_mean', 'd_meanB')]
                for m in methods:
                    if m != FOCUS:
                        pairs.append(('delta_vs_' + m, 'd_' + m))
                for mc, rc in pairs:
                    sworst[(fold, sn, mc)] = max(sworst.get((fold, sn, mc), 0.0), abs(float(r[mc]) - float(e[rc])))
                if 'd_best_minus_d_best_at_K10' in e or 'd_best_minus_d_best_at_K20' in e:
                    rk = [k for k in e if k.startswith('d_best_minus_d_best_at_K')][0]
                    sworst[(fold, sn, 'delta_best_change')] = max(sworst.get((fold, sn, 'delta_best_change'), 0.0),
                                                                  abs(float(r['delta_best_change_vs_K%d' % int(rk.split('K')[-1])]) - float(e[rk])))
                if e.get('best_name') != r['best_baseline']:
                    log('  WARNING best baseline differs: %s %s K=%s ref=%s mine=%s' % (fold, sn, Ks, e.get('best_name'), r['best_baseline']))
    log('parity S7-05 (same classes across K, K-way variant; max abs diff over K)')
    for k, v in sorted(sworst.items()):
        log('  %s %s %s %.6f' % (k[0], k[1], k[2], v))
    out['same_classes_max_abs_diff'] = max(sworst.values()) if sworst else None
    log('parity max abs diff: per-class %.6f, S7-05 %.6f' % (out['per_class_max_abs_diff'], out['same_classes_max_abs_diff']))
    return out


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', required=True, help='runs root containing top10/ top20/ top30/ (read-only)')
    ap.add_argument('--out', required=True, help='new output directory (an existing one is refused)')
    ap.add_argument('--methods', default=','.join(DEFAULT_METHODS),
                    help='comma list; cei_mp2 is the focus method, the others are baselines (default: %(default)s)')
    ap.add_argument('--b', type=int, default=1000, help='bootstrap resamples of subjects (default 1000)')
    ap.add_argument('--parity-ref', default=None,
                    help='optional directory with a reference per_class_delta_vs_support.csv and partAC_results.json; prints max differences')
    a = ap.parse_args(argv)

    methods = [m.strip() for m in a.methods.split(',') if m.strip()]
    if FOCUS not in methods or len(methods) < 2 or len(set(methods)) != len(methods):
        refuse('--methods must contain %s and at least one distinct baseline' % FOCUS)
    methods = [FOCUS] + sorted(m for m in methods if m != FOCUS)
    baselines = methods[1:]
    if os.path.exists(a.out):
        refuse('--out %s already exists; write to a new path' % a.out)
    if not os.path.isdir(a.runs):
        refuse('--runs is not a directory')
    log = lambda s: print(s, flush=True)

    log('methods: %s | baselines: %s | seeds %s | B=%d' % (methods, baselines, SEEDS, a.b))
    D = {K: load_k(a.runs, K, methods) for K in KS}
    best, val_means = choose_best(D, methods, baselines)
    for K in KS:
        log('K=%d best baseline by mean validation macro-F1: %s (%.4f); validation means %s' % (
            K, best[K], val_means[K][best[K]], {m: round(v, 4) for m, v in val_means[K].items()}))

    rows, sp_rows, terc_rows, summary_fold = analyse(D, methods, baselines, best, a.b, log)
    same_rows = same_classes(D, methods, baselines, best, a.b, log)

    os.makedirs(a.out)
    pc_fields = (['fold', 'K', 'class', 'train_support', 'eval_support'] + ['f1_' + m for m in methods] +
                 ['f1sd_' + m for m in methods] + ['best_baseline', 'delta_best', 'delta_best_sd', 'delta_mean', 'delta_mean_sd'] +
                 ['delta_vs_' + m for m in baselines])
    rows, sp_rows, terc_rows, same_rows = (report_order(x) for x in (rows, sp_rows, terc_rows, same_rows))
    write_csv(os.path.join(a.out, 'per_class_delta.csv'), rows, pc_fields)

    sp_fields = ['fold', 'K', 'delta', 'classes', 'n_classes', 'rho', 'ci_lo', 'ci_hi', 'B', 'n_subjects', 'n_rows']
    write_csv(os.path.join(a.out, 'spearman.csv'), sp_rows, sp_fields)
    write_md(os.path.join(a.out, 'spearman.md'), 'Spearman rho of CEI-minus-baseline delta vs log10 train support',
             'Test = reported result (listed first); validation = decision fold. '
             'delta_best: CEI minus the best baseline (chosen on validation, also used on test); delta_mean: CEI minus the mean of the baselines. '
             'classes: all = classes with rows in the fold; ge%d = classes with at least %d rows in the fold. '
             '95%% interval: subject bootstrap (B=%d, seed %d).' % (MIN_ROWS, MIN_ROWS, a.b, BOOT_SEED),
             sp_rows, sp_fields)

    tf = ['fold', 'K', 'tercile', 'n_classes', 'train_support_min', 'train_support_max', 'f1_cei', 'f1_mean_baselines',
          'f1_best_baseline', 'best_baseline', 'delta_best', 'delta_mean', 'n_cei_above_best']
    write_csv(os.path.join(a.out, 'terciles.csv'), terc_rows, tf)
    write_md(os.path.join(a.out, 'terciles.md'), 'Terciles of train support: mean per-class delta',
             'Classes with rows in the fold, split into equal-size terciles by train support (low, mid, high). Test = reported result '
             '(listed first); validation = decision fold. best_baseline was chosen on validation.',
             terc_rows, tf)

    sf = (['fold', 'class_set', 'K', 'n_classes_in_set', 'n_rows_fold', 'n_patients_fold', 'f1_cei', 'f1_mean_baselines', 'f1_best_baseline',
           'best_baseline', 'delta_mean', 'delta_mean_ci_lo', 'delta_mean_ci_hi', 'delta_best', 'delta_best_ci_lo', 'delta_best_ci_hi'] +
          ['delta_vs_' + m for m in baselines] +
          ['delta_mean_change_vs_K10', 'delta_mean_change_vs_K20', 'delta_mean_change_ci_lo', 'delta_mean_change_ci_hi',
           'delta_best_change_vs_K10', 'delta_best_change_vs_K20', 'delta_best_change_ci_lo', 'delta_best_change_ci_hi', 'B'])
    write_csv(os.path.join(a.out, 'same_classes_across_k.csv'), same_rows, sf)
    md_f = ['fold', 'class_set', 'K', 'n_classes_in_set', 'f1_cei', 'f1_mean_baselines', 'f1_best_baseline', 'best_baseline',
            'delta_mean', 'delta_mean_ci', 'delta_best', 'delta_best_ci', 'delta_mean_change', 'delta_mean_change_ci',
            'delta_best_change', 'delta_best_change_ci']
    md_rows = []
    for r in same_rows:
        q = dict(r)
        q['delta_mean_ci'] = ci_str(r['delta_mean_ci_lo'], r['delta_mean_ci_hi'])
        q['delta_best_ci'] = ci_str(r['delta_best_ci_lo'], r['delta_best_ci_hi'])
        for nm in ('delta_mean', 'delta_best'):
            ck = [k for k in r if k.startswith(nm + '_change_vs_K')]
            if ck and r.get(ck[0]) is not None:
                q[nm + '_change'] = '%+.4f (vs %s)' % (r[ck[0]], ck[0].split('_')[-1])
                q[nm + '_change_ci'] = ci_str(r[nm + '_change_ci_lo'], r[nm + '_change_ci_hi'])
        md_rows.append(q)
    write_md(os.path.join(a.out, 'same_classes_across_k.md'), 'Same classes across K (K-way task)',
             'F1 averaged over a fixed class set (top10: the K=10 classes, top20: the K=20 classes, all: all K classes), '
             'computed inside each K-way task. Test = reported result (listed first); validation = decision fold. '
             'delta = CEI minus the mean of the baselines (delta_mean) or minus the best baseline chosen on validation '
             '(delta_best); the change is relative to the first K that contains the set, with the same best baseline (chosen at the later K) at both ends. '
             'class_set=all is the official K-way macro-F1 gap. 95%% interval: subject bootstrap over the pooled subjects of the fold '
             '(B=%d, seed %d).' % (a.b, BOOT_SEED), md_rows, md_f)

    figures(rows, sp_rows, a.out, methods)

    summary = {
        'methods': methods, 'focus': FOCUS, 'baselines': baselines, 'seeds': SEEDS, 'K': KS, 'folds': REPORT_FOLDS,
        'fold_note': {'test': 'reported result (scored once per run at the validation-selected checkpoint)',
                      'validation': 'decision fold (checkpoint, HPO, best-baseline choice)'},
        'bootstrap': {'unit': 'subjects of the fold (pooled across K for same_classes_across_k)', 'B': a.b, 'seed': BOOT_SEED},
        'sd_note': 'seed SD ddof=1; SD of a delta = sqrt(sdA^2/n + sdB^2/n), mean of m baselines: sqrt(sdA^2/n + sum sd_b^2/(n m^2)), n=%d' % len(SEEDS),
        'runs_root': os.path.basename(os.path.normpath(a.runs)),
        'best_baseline': {str(K): {'method': best[K], 'mean_validation_macro_f1': rnd(val_means[K][best[K]], 4),
                                   'validation_macro_f1_by_method': {m: rnd(v, 4) for m, v in val_means[K].items()}} for K in KS},
        'tasks': {'%s_K%d' % k: dict(v, class_set='top%d' % k[1], train_support_equal_across_runs=True)
                  for k, v in summary_fold.items()},
        'spearman': sp_rows,
    }
    if a.parity_ref:
        summary['parity'] = parity(a.parity_ref, rows, same_rows, methods, log)
    with open(os.path.join(a.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    log('wrote per_class_delta.csv, spearman.csv|md, terciles.csv|md, same_classes_across_k.csv|md, fig_support_k{10,20,30}.png, summary.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
