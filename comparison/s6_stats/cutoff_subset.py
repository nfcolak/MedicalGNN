"""Cut-off subset: how do the stored final models score on the stays whose cut-off equals the ED discharge time?

Inference-free analysis of stored predictions. For every stored final run (`<runs>/top<K>/<method>_seed<S>/`, 5 methods
x 3 K x 3 seeds) the scored fold (`--fold`, default test, the reporting fold) is split by a per-stay flag:
subset = the cut-off of the stay equals the ED discharge (`outtime`) time, i.e. the prediction is made at discharge
(cutoff = min(first representable lab storetime + 60 min, ED outtime)); rest = every other stay. The cut-off comes from
the graph artifact's `cohort.csv`, the discharge time from the event index (`events.sqlite`, table `visits`, column
`finish`); rows are matched to the stored predictions by sample id. Nothing is trained, no checkpoint or graph is
loaded, and the outputs hold aggregates only (no sample, subject or stay ids).

    python -m comparison.s6_stats.cutoff_subset --runs <runs root> --cohort <graph dir>/cohort.csv \
        --events <event index dir>/events.sqlite --out <new dir> [--fold test|validation] [--ks 10,20,30] \
        [--methods cei_mp2,gsat,graphcare,protgnn,gchm_pna] [--seeds 1234,2025,7] [--b 1000]

Metrics per run: accuracy and macro-F1 (mean per-class F1 over the classes with support > 0 in the scored rows, so a class
that does not occur in the small subset is left out; the parity check against result.json uses its K-class mean). Gap = subset minus rest, averaged over the seeds. Interval: patient bootstrap (resampling
subjects with replacement; one draw shared by all runs of a K, B draws, rng 2026 as in comparison/top3/compare.py; the
statistic is the seed-mean gap on the resampled rows); percentile 95%. "Overall without the subset" = the metric on the
rest. Parity: the whole-fold accuracy and macro-F1 recomputed here must equal the stored result.json (tolerance 1e-6),
otherwise exit 2 before anything is written. Decisions (checkpoint, HPO, baseline choice) use VALIDATION; every reported
result is on the scored fold (test for the reported tables; `--fold validation` is the labelled decision fold and
reproduces the earlier validation statement of +8.5 to +11 accuracy points on the subset (K = 20)). --out must not exist.
"""
import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from comparison.s6_stats.stats import (DEFAULT_METHODS, FOLD_FILE, FOLD_RESULT_KEY, KS, NAME, PATIENT_RNG,  # noqa: E402
                                       SEEDS, TOL, json_clean, md_table)

METHODS = DEFAULT_METHODS + ('gchm_pna',)
FOLDS = ('test', 'validation')
STORED_STATEMENT = (8.5, 11.0)      # validation accuracy points on the subset, every method (stored statement)


class Refusal(Exception):
    pass


def load_flags(cohort, events):
    """{sample_id: (split, flag)}; flag = cut-off equals the ED discharge time of the stay."""
    con = sqlite3.connect(f'file:{events}?mode=ro', uri=True)
    finish = dict(con.execute('select stay, finish from visits'))
    con.close()
    flags = {}
    with open(cohort, newline='') as fh:
        for r in csv.DictReader(fh):
            # timestamps are ISO strings to the second in both files; compare normalised to 'YYYY-MM-DDTHH:MM:SS'
            f = finish.get(r['stay_id'])
            if f is None:
                raise Refusal('a cohort stay has no visit row in the event index')
            flags[r['sample_id']] = (r['split'], f.replace(' ', 'T')[:19] == r['cutoff'].replace(' ', 'T')[:19])
    return flags


def weighted_metrics(y, pred, K, W, mask, fixed=False):
    """Accuracy and macro-F1 of the rows in `mask`, one value per weight column. Macro-F1 averages the classes with
    weighted support > 0, or all K classes with fixed=True (the stored result.json definition).
    y, pred: [n]; W: [n, B] row weights; mask: [n] bool. Returns (accuracy [B], macro_f1 [B])."""
    Wm = W * mask[:, None]
    yh = np.eye(K)[y]
    ph = np.eye(K)[pred]
    support = yh.T @ Wm
    n_pred = ph.T @ Wm
    tp = (yh * ph).T @ Wm
    den = support + n_pred
    f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
    present = np.ones_like(support, dtype=bool) if fixed else support > 0
    macro = (f1 * present).sum(0) / np.maximum(present.sum(0), 1)
    total = Wm.sum(0)
    acc = np.divide((Wm * (pred == y)[:, None]).sum(0), total, out=np.full(total.shape, np.nan), where=total > 0)
    return acc, macro


def load_runs(runs, methods, ks, seeds):
    data = {}
    for K in ks:
        for m in methods:
            for s in seeds:
                d = Path(runs) / f'top{K}' / f'{m}_seed{s}'
                if not (d / 'result.json').is_file():
                    raise Refusal(f'missing finished run top{K}/{m}_seed{s}')
                data[(K, m, s)] = d
    return data


def read_fold(d, fold):
    z = np.load(d / FOLD_FILE[fold])
    return z['proba'].argmax(1).astype(np.int64), z['y'].astype(np.int64), z['subjects'], z['sample_ids']


def ci(draws, alpha=0.05):
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def ms(v):
    v = np.asarray(v, dtype=np.float64)
    return {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)) if len(v) > 1 else None}


def analyse(args):
    methods = [m for m in args.methods.split(',') if m]
    ks = [int(k) for k in args.ks.split(',') if k]
    seeds = [int(s) for s in args.seeds.split(',') if s]
    flags = load_flags(args.cohort, args.events)
    runs = load_runs(args.runs, methods, ks, seeds)
    res = {'fold': args.fold, 'ks': ks, 'methods': methods, 'seeds': seeds, 'b': args.b, 'sizes': {}, 'parity': [],
           'cells': {}}
    # cohort-level share per split (all stays of the graph artifact)
    share = {}
    for split, flag in flags.values():
        a = share.setdefault(split, [0, 0])
        a[0] += int(flag)
        a[1] += 1
    res['cohort_share'] = {k: {'subset': v[0], 'stays': v[1], 'share': v[0] / v[1]} for k, v in sorted(share.items())}
    for K in ks:
        for fold in FOLDS:     # sizes for both folds, the scored fold decides the rest
            _, y, _, ids = read_fold(runs[(K, methods[0], seeds[0])], fold)
            sub = np.array([flags[i][1] for i in ids])
            res['sizes'][f'{K}/{fold}'] = {'rows': int(len(y)), 'subset': int(sub.sum()),
                                           'share': float(sub.mean())}
        rows = {}
        for m in methods:
            for s in seeds:
                pred, y, subj, ids = read_fold(runs[(K, m, s)], args.fold)
                rows[(m, s)] = (pred, y, subj, ids)
        ref_ids = rows[(methods[0], seeds[0])][3]
        for key, (_, y, subj, ids) in rows.items():
            if not np.array_equal(ids, ref_ids):
                raise Refusal(f'top{K} {key}: sample ids differ from the reference run')
        _, y, subj, ids = rows[(methods[0], seeds[0])]
        sub = np.array([flags[i][1] for i in ids])
        rest = ~sub
        allm = np.ones(len(y), dtype=bool)
        uniq, inv = np.unique(subj, return_inverse=True)
        rng = np.random.default_rng(PATIENT_RNG)
        draws = rng.integers(0, len(uniq), size=(args.b, len(uniq)))
        counts = np.stack([np.bincount(d, minlength=len(uniq)) for d in draws])
        W = counts[:, inv].T.astype(np.float64)        # [n, B]
        ones = np.ones((len(y), 1))
        for m in methods:
            per_seed, boot_gap = [], {'accuracy': [], 'macro_f1': []}
            for s in seeds:
                pred = rows[(m, s)][0]
                point = {name: weighted_metrics(y, pred, K, ones, mk) for name, mk in
                         (('all', allm), ('subset', sub), ('rest', rest))}
                st = json.loads((runs[(K, m, s)] / 'result.json').read_text())[FOLD_RESULT_KEY[args.fold]]
                d_acc = abs(float(point['all'][0][0]) - st['accuracy'])
                d_f1 = abs(float(weighted_metrics(y, pred, K, ones, allm, fixed=True)[1][0]) - st['macro_f1'])
                res['parity'].append({'K': K, 'method': m, 'seed': s, 'accuracy_abs_diff': d_acc,
                                      'macro_f1_abs_diff': d_f1})
                if max(d_acc, d_f1) > TOL:
                    raise Refusal(f'parity failed for top{K} {m} seed {s}: accuracy {d_acc:.2e}, macro-F1 {d_f1:.2e}')
                per_seed.append({n: {'accuracy': float(v[0][0]), 'macro_f1': float(v[1][0])} for n, v in point.items()})
                bs, br = weighted_metrics(y, pred, K, W, sub), weighted_metrics(y, pred, K, W, rest)
                boot_gap['accuracy'].append(bs[0] - br[0])
                boot_gap['macro_f1'].append(bs[1] - br[1])
            cell = {'seeds': seeds}
            for part in ('all', 'subset', 'rest'):
                cell[part] = {met: ms([p[part][met] for p in per_seed]) for met in ('accuracy', 'macro_f1')}
            for met in ('accuracy', 'macro_f1'):
                gaps = [p['subset'][met] - p['rest'][met] for p in per_seed]
                bg = np.mean(boot_gap[met], axis=0)
                lo, hi = ci(bg)
                cell.setdefault('gap', {})[met] = {**ms(gaps), 'ci95': [lo, hi],
                                                   'all_minus_rest': float(np.mean([p['all'][met] - p['rest'][met]
                                                                                    for p in per_seed]))}
            res['cells'][f'{K}/{m}'] = cell
    res['max_abs_diff'] = {k: max(p[k] for p in res['parity']) for k in ('accuracy_abs_diff', 'macro_f1_abs_diff')}
    return res


def render(res):
    fold = res['fold']
    pts = lambda x: f'{100 * x:+.1f}'          # noqa: E731
    L = [f'# Cut-off = ED discharge subset ({fold} fold)', '',
         f'Reported fold: {fold}' + (' (decision fold; labelled "validation")' if fold == 'validation' else '') +
         f'. Methods {", ".join(NAME.get(m, m) for m in res["methods"])}; K {", ".join(map(str, res["ks"]))}; seeds '
         f'{", ".join(map(str, res["seeds"]))} ({len(res["methods"]) * len(res["ks"]) * len(res["seeds"])} stored runs). '
         'Subset = the cut-off of the stay equals the ED discharge time; rest = all other stays. '
         'Gap = subset minus rest in points (seed mean), interval = patient bootstrap '
         f'(B={res["b"]}, shared subject draw, percentile 95%).', '',
         '## Subset size', '',
         md_table(['K', 'fold', 'rows scored', 'rows in subset', 'share'],
                  [[str(K), f, str(v['rows']), str(v['subset']), f"{100 * v['share']:.1f}%"]
                   for K in res['ks'] for f in FOLDS for v in [res['sizes'][f'{K}/{f}']]]),
         'All stays of the graph artifact (before the class filter): '
         + '; '.join(f"{k} {v['subset']}/{v['stays']} = {100 * v['share']:.2f}%" for k, v in res['cohort_share'].items())
         + '.', '']
    for met, title in (('accuracy', 'Accuracy'), ('macro_f1', 'Macro-F1')):
        L += [f'## {title}: subset, rest and gap ({fold})', '']
        rows = []
        for K in res['ks']:
            for m in res['methods']:
                c = res['cells'][f'{K}/{m}']
                g = c['gap'][met]
                rows.append([str(K), NAME.get(m, m), f"{c['subset'][met]['mean']:.4f}", f"{c['rest'][met]['mean']:.4f}",
                             f"{c['all'][met]['mean']:.4f}", f"{pts(g['mean'])} [{pts(g['ci95'][0])}, {pts(g['ci95'][1])}]",
                             f"{pts(g['all_minus_rest'])}"])
        L += [md_table(['K', 'method', 'subset', 'rest = overall without the subset', 'overall', 'gap subset - rest, points [95% CI]',
                        'overall - rest, points'], rows)]
    L += ['## Reproduction of the stored validation statement', '',
          f'Stored statement: +{STORED_STATEMENT[0]} to +{STORED_STATEMENT[1]:g} validation accuracy points on the subset '
          'for every method (about +0.6 overall), made for the K = 20 task. Accuracy gap per K over the methods:', '']
    rows = []
    for K in res['ks']:
        cells = [res['cells'][f'{K}/{m}']['gap']['accuracy'] for m in res['methods']]
        g = [c['mean'] * 100 for c in cells]
        o = [c['all_minus_rest'] * 100 for c in cells]
        rows.append([str(K), f'{min(g):+.1f} to {max(g):+.1f}', f'{min(o):+.2f} to {max(o):+.2f}'])
    L += [md_table(['K', 'subset - rest accuracy gap, points (min to max over methods)',
                    'overall - rest accuracy, points'], rows)]
    if 20 in res['ks']:
        g20 = [res['cells'][f'20/{m}']['gap']['accuracy']['mean'] * 100 for m in res['methods']]
        inside = STORED_STATEMENT[0] - 0.05 <= min(g20) and max(g20) <= STORED_STATEMENT[1] + 0.05
        L += [f'K = 20, {fold} fold: {min(g20):+.1f} to {max(g20):+.1f} points; inside the stored range '
              f'(+{STORED_STATEMENT[0]} to +{STORED_STATEMENT[1]:g}): {"yes" if inside else "no"}'
              + ('.' if fold == 'validation' else ' (the stored statement is a validation statement; this is the test fold).'), '']
    L += [f"Parity with the stored result.json (whole fold, {len(res['parity'])} runs): max abs diff accuracy "
          f"{res['max_abs_diff']['accuracy_abs_diff']:.2e}, macro-F1 {res['max_abs_diff']['macro_f1_abs_diff']:.2e}.", '']
    return '\n'.join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--runs', required=True, help='final runs root (holds top<K>/<method>_seed<S>)')
    ap.add_argument('--cohort', required=True, help='cohort.csv of the graph artifact (sample_id, stay_id, cutoff, split)')
    ap.add_argument('--events', required=True, help='events.sqlite of the event index (table visits: stay, finish)')
    ap.add_argument('--out', required=True, help='new output directory (must not exist)')
    ap.add_argument('--fold', choices=FOLDS, default='test')
    ap.add_argument('--ks', default=','.join(map(str, KS)))
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    ap.add_argument('--b', type=int, default=1000, help='patient bootstrap draws')
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        print(f'REFUSED: --out already exists: {out.name}', file=sys.stderr)
        return 2
    try:
        res = analyse(a)
    except Refusal as e:
        print(f'REFUSED: {e}', file=sys.stderr)
        return 2
    out.mkdir(parents=True)
    (out / 'cutoff_subset.json').write_text(json.dumps(json_clean(res), indent=1))
    (out / 'cutoff_subset.md').write_text(render(res))
    print(f'wrote {out.name}/cutoff_subset.md and cutoff_subset.json (fold {a.fold})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
