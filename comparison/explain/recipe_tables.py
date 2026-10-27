"""Explanation results of the top-20 final runs next to the CEI-GNN training-recipe variants (test fold), read-only.

    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m comparison.explain.recipe_tables \
        --final-summary <summary dir of the 12 final runs> --final-records <explanation dir of the 12 final runs> \
        --variant R1_ls0=<summary dir> [--variant R2_ls0_wd=<summary dir> ...] --ls0-records <explanation dir of R1_ls0> \
        [--root DIR] --out <new dir>

Inputs: the summary folders written by comparison.explain.summarize (summary.csv), the explanation record folders
written by comparison.explain.run (per-visit JSON records; only the R1_ls0 and final CEI-GNN records are read), and
each trained run's stored test predictions (test.npz) under DIR/comparison/standardized (default: env
MEDGNN_DATA_ROOT or the repo root). Run folders: final runs clinical_runs_fair_v10t_20261011/top20, recipe arms
clinical_runs_nasim_s2_<date>/<arm>/top20 (dates as in comparison/nasim/recipe_control.py).

Writes <out>/comparison_table.md and <out>/comparison_table.json (aggregates only, no identifiers):
  1. one row per model: test macro-F1 (mean ± seed SD), accuracy, balanced accuracy (mean per-class recall), top-3
     accuracy and mean confidence on all test visits; the deletion-vs-random headline of IG, Grad, GNNExplainer and
     the built-in explanation (mean ± seed SD); fidelity+, fidelity- and mass90 sparsity of the built-in explanation;
  2. per explanation source: deletion headline, fidelity+, fidelity-, mass90 sparsity, their ranks and the mean rank;
  3. per class: test F1 of the final CEI-GNN and of R1_ls0 (label smoothing 0), and the IG and built-in headline by
     true class on the explained visits.
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from comparison.explain.summarize import source_metrics  # noqa: E402

K = 20
SEEDS = (1234, 2025, 7)
HEADLINE = 'deletion_mean_change_vs_random'
SOURCES = {'IG': 'IntegratedGradExplainer', 'Grad': 'GradExplainer', 'GNNExplainer': 'GNNExplainer', 'Built-in': 'builtin'}
FINAL_METHODS = [  # label, method key in summary.csv, run-folder prefix
    ('GSAT', 'gsat', 'gsat'),
    ('GraphCare-style', 'graphcare', 'graphcare'),
    ('ProtGNN', 'protgnn', 'protgnn'),
    ('CEI-GNN final', 'cei_gnn_mp2', 'cei_mp2'),
]
ARMS = {  # arm -> (label, training date of its run folder, label smoothing, weight decay 0.01, DropEdge 0.2)
    'R4_wd': ('CEI-GNN, weight decay removed', '20261025', 1, 0, 1),
    'R5_de0': ('CEI-GNN, DropEdge removed', '20261025', 1, 1, 0),
    'R1_ls0': ('CEI-GNN, label smoothing removed', '20261012', 0, 1, 1),
    'R2_ls0_wd': ('CEI-GNN, label smoothing and weight decay removed', '20261012', 0, 0, 1),
    'R3_ls0_wd_de0': ('CEI-GNN, all three removed', '20261012', 0, 0, 0),
}
GRADED = [  # metric, header, sign (+1: higher is better)
    (HEADLINE, 'deletion vs random', +1), ('fidelity_plus', 'fidelity+', +1),
    ('fidelity_minus', 'fidelity-', -1), ('mass90_sparsity', 'sparsity', +1),
]


def summary_rows(summary_dir, method):
    rows = [r for r in csv.DictReader(open(Path(summary_dir) / 'summary.csv'))
            if r['method'] == method and r['table'] == 'all_visits']
    out = {}
    for short, src in SOURCES.items():
        r = [x for x in rows if x['source'] == src]
        if len(r) != 1:
            raise SystemExit(f'{summary_dir}: expected one all_visits row for {method}/{src}, found {len(r)}')
        r = r[0]
        out[short] = {k: float(r[f'{k}_mean']) for k, _, _ in GRADED}
        out[short].update({f'{k}_sd': float(r[f'{k}_sd']) for k, _, _ in GRADED})
        out[short]['visits_explained'] = int(r['visits_explained'])
    return out


def performance(run_glob):
    dirs = sorted(glob.glob(run_glob))
    if len(dirs) != len(SEEDS):
        raise SystemExit(f'{run_glob}: expected {len(SEEDS)} runs, found {len(dirs)}')
    f1, acc, bacc, top3, conf = [], [], [], [], []
    for d in dirs:
        z = np.load(f'{d}/test.npz', allow_pickle=True)
        y, p = np.asarray(z['y']).astype(int), z['proba']
        pred = p.argmax(1)
        f1.append(f1_score(y, pred, average='macro'))
        acc.append(float((pred == y).mean()))
        bacc.append(float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)])))
        top3.append(float(np.mean((np.argsort(-p, axis=1)[:, :3] == y[:, None]).any(1))))
        conf.append(float(p.max(1).mean()))
    return {'macro_f1': float(np.mean(f1)), 'macro_f1_sd': float(np.std(f1, ddof=1)), 'accuracy': float(np.mean(acc)),
            'balanced_accuracy': float(np.mean(bacc)), 'top3_accuracy': float(np.mean(top3)),
            'mean_confidence': float(np.mean(conf)), 'test_visits': int(len(y))}


def per_class(final_runs, ls0_runs, final_records, ls0_records):
    names = json.load(open(f'{final_runs}/cei_mp2_seed{SEEDS[0]}/binding.json'))['label_order']
    f1 = {}
    for arm, root in (('final', final_runs), ('ls0', ls0_runs)):
        f1[arm] = np.array([f1_score(np.asarray(z['y']).astype(int), z['proba'].argmax(1), average=None,
                                     labels=list(range(len(names))))
                            for z in (np.load(f'{root}/cei_mp2_seed{s}/test.npz', allow_pickle=True) for s in SEEDS)])
    support = np.bincount(np.asarray(np.load(f'{final_runs}/cei_mp2_seed{SEEDS[0]}/test.npz', allow_pickle=True)['y'])
                          .astype(int), minlength=len(names))
    expl, nvis = {}, np.zeros(len(names), int)
    for arm, root in (('final', final_records), ('ls0', ls0_records)):
        for short in ('IG', 'Built-in'):
            mat = np.full((len(SEEDS), len(names)), np.nan)
            for i, s in enumerate(SEEDS):
                acc = defaultdict(list)
                for path in glob.glob(f'{root}/top20/cei_mp2_seed{s}/subject_*.json'):
                    rec = json.load(open(path))
                    m = source_metrics((rec.get('fidelity_v2') or {}).get(SOURCES[short]))
                    if m and HEADLINE in m:
                        acc[int(rec['true_class_id'])].append(m[HEADLINE])
                    if arm == 'final' and short == 'IG' and i == 0:
                        nvis[int(rec['true_class_id'])] += 1
                for c, v in acc.items():
                    mat[i, c] = np.mean(v)
            expl[(arm, short)] = mat
    d_f1 = f1['final'].mean(0) - f1['ls0'].mean(0)
    se = np.sqrt(f1['final'].var(0, ddof=1) / len(SEEDS) + f1['ls0'].var(0, ddof=1) / len(SEEDS))
    rows = []
    for c in sorted(range(len(names)), key=lambda c: -d_f1[c]):
        rows.append({'class': names[c], 'test_visits': int(support[c]), 'explained_visits': int(nvis[c]),
                     'f1_final': float(f1['final'][:, c].mean()), 'f1_ls0': float(f1['ls0'][:, c].mean()),
                     'f1_gain_from_ls': float(d_f1[c]), 'f1_gain_se': float(se[c]),
                     'ig_final': float(np.nanmean(expl[('final', 'IG')][:, c])),
                     'ig_ls0': float(np.nanmean(expl[('ls0', 'IG')][:, c])),
                     'builtin_final': float(np.nanmean(expl[('final', 'Built-in')][:, c])),
                     'builtin_ls0': float(np.nanmean(expl[('ls0', 'Built-in')][:, c]))})

    def spearman(a, b):
        return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
    d_ig = np.array([r['ig_ls0'] - r['ig_final'] for r in rows])
    d_bi = np.array([r['builtin_ls0'] - r['builtin_final'] for r in rows])
    gain = np.array([r['f1_gain_from_ls'] for r in rows])
    return rows, {'spearman_f1_gain_vs_ig_cost': spearman(gain, d_ig),
                  'spearman_f1_gain_vs_builtin_cost': spearman(gain, d_bi)}


def fmt(v, d=3):
    return f'{v:.{d}f}'


def write_markdown(models, graded, classes, corr, path):
    L = ['# Explanation results, top-20, test fold: final methods and CEI-GNN training-recipe variants', '',
         'Protocol fair_v10t, K = 20. Performance: all test visits (stored test predictions of each run). Explanations: '
         'the same 1,000 random test patients (1,345 visits) in every run, intervention contract v6. Mean over the '
         'three seeds (1234, 2025, 7); ± is the seed SD. Deletion vs random: how much more the predicted-class '
         'probability drops when the explanation\'s top-ranked evidence is deleted than when random evidence of the '
         'same size is deleted (higher is better). Fidelity+: drop when the top 20% are deleted (higher is better). '
         'Fidelity-: drop when only the top 20% are kept (lower is better). Sparsity: share of nodes that can be '
         'dropped while keeping 90% of the importance mass.', '',
         'LS = label smoothing 0.1, WD = decoupled weight decay 0.01 on the weight matrices, DE = DropEdge 0.2 '
         '(1 = used as in the final recipe, 0 = removed). The recipe variants were explained after the test '
         'explanation results of the final runs had been seen; they are a diagnostic, not a model selection. The '
         'reported model stays CEI-GNN final.', '',
         '## 1. Performance and explanation scores', '',
         '| Model | LS | WD | DE | Macro-F1 | Accuracy | Balanced accuracy | Top-3 accuracy | Mean confidence '
         '| IG | Grad | GNNExplainer | Built-in | Built-in fidelity+ | Built-in fidelity- | Built-in sparsity |',
         '|' + '---|' * 16]
    for m in models:
        e, p = m['explain'], m['performance']
        recipe = ['own recipe', '', ''] if m['ls'] is None else [str(m['ls']), str(m['wd']), str(m['de'])]
        L.append('| ' + ' | '.join([m['label'], *recipe, f"{fmt(p['macro_f1'], 4)} ± {fmt(p['macro_f1_sd'], 4)}",
                                    fmt(p['accuracy'], 4), fmt(p['balanced_accuracy'], 4), fmt(p['top3_accuracy'], 4),
                                    fmt(p['mean_confidence'])]
                                   + [f"{fmt(e[s][HEADLINE])} ± {fmt(e[s][HEADLINE + '_sd'])}" for s in SOURCES]
                                   + [fmt(e['Built-in']['fidelity_plus']), fmt(e['Built-in']['fidelity_minus']),
                                      fmt(e['Built-in']['mass90_sparsity'])]) + ' |')
    L += ['', 'GraphCare-style and ProtGNN: the built-in explanation is not better than random, so its side metrics '
          'are not comparable. ProtGNN built-in: visit-runs whose prototype subgraph gives zero importance to every '
          'evidence node are left out of its mean (see visits explained in the summary).', '',
          '## 2. Per explanation source: scores, ranks (1 = best of the rows) and mean rank', '']
    for short in SOURCES:
        L += [f'### {short}', '', '| Model | ' + ' | '.join(h for _, h, _ in GRADED) + ' | Mean rank |',
              '|' + '---|' * (len(GRADED) + 2)]
        for g in graded[short]:
            L.append('| ' + ' | '.join([g['label']] + [f"{fmt(g[k])} ± {fmt(g[k + '_sd'])} ({g[k + '_rank']})"
                                                       for k, _, _ in GRADED] + [f"{g['mean_rank']:.2f}"]) + ' |')
        L.append('')
    L += ['## 3. Per class: final CEI-GNN vs label smoothing removed (R1_ls0)', '',
          'F1 on all test visits; explanation score (deletion vs random) by true class on the explained visits. '
          'F1 gain from LS = F1 final - F1 without LS; se = seed standard error of that difference.', '',
          '| Class | Test visits | Explained visits | F1 final | F1 without LS | F1 gain from LS | se | IG final '
          '| IG without LS | Built-in final | Built-in without LS |', '|' + '---|' * 11]
    for r in classes:
        L.append('| ' + ' | '.join([r['class'], str(r['test_visits']), str(r['explained_visits']), fmt(r['f1_final']),
                                    fmt(r['f1_ls0']), f"{r['f1_gain_from_ls']:+.3f}", fmt(r['f1_gain_se']),
                                    fmt(r['ig_final']), fmt(r['ig_ls0']), fmt(r['builtin_final']),
                                    fmt(r['builtin_ls0'])]) + ' |')
    L += ['', f"Spearman correlation across classes: F1 gain from LS vs IG cost of LS "
              f"{corr['spearman_f1_gain_vs_ig_cost']:.2f}; vs built-in cost {corr['spearman_f1_gain_vs_builtin_cost']:.2f}.",
          '']
    Path(path).write_text('\n'.join(L).replace('-0.', '−0.').replace(' -', ' −'))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--final-summary', required=True)
    ap.add_argument('--final-records', required=True)
    ap.add_argument('--variant', action='append', default=[], metavar='ARM=SUMMARY_DIR')
    ap.add_argument('--ls0-records', required=True)
    ap.add_argument('--root', default=os.environ.get('MEDGNN_DATA_ROOT', str(REPO)))
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'{out} exists; give a new path')
    std = Path(a.root) / 'comparison/standardized'
    final_runs = std / 'clinical_runs_fair_v10t_20261011/top20'
    variants = dict(v.split('=', 1) for v in a.variant)
    unknown = sorted(set(variants) - set(ARMS))
    if unknown:
        raise SystemExit(f'unknown arm(s) {unknown}; known: {list(ARMS)}')

    models = []
    for label, method, prefix in FINAL_METHODS:
        models.append({'label': label, 'ls': 1 if prefix == 'cei_mp2' else None, 'wd': 1, 'de': 1,
                       'performance': performance(f'{final_runs}/{prefix}_seed*'),
                       'explain': summary_rows(a.final_summary, method)})
    for arm in ARMS:
        if arm not in variants:
            continue
        label, date, ls, wd, de = ARMS[arm]
        runs = std / f'clinical_runs_nasim_s2_{date}/{arm}/top20'
        models.append({'label': label, 'arm': arm, 'ls': ls, 'wd': wd, 'de': de,
                       'performance': performance(f'{runs}/cei_mp2_seed*'),
                       'explain': summary_rows(variants[arm], 'cei_gnn_mp2')})

    graded = {}
    for short in SOURCES:
        rows = []
        for m in models:
            row = {'label': m['label']}
            row.update({k: m['explain'][short][k] for k, _, _ in GRADED})
            row.update({k + '_sd': m['explain'][short][k + '_sd'] for k, _, _ in GRADED})
            rows.append(row)
        for k, _, sign in GRADED:
            order = sorted((r[k] for r in rows), key=lambda v: -sign * v)
            for r in rows:
                r[k + '_rank'] = 1 + order.index(r[k])
        for r in rows:
            r['mean_rank'] = sum(r[k + '_rank'] for k, _, _ in GRADED) / len(GRADED)
        graded[short] = sorted(rows, key=lambda r: r['mean_rank'])

    classes, corr = per_class(final_runs, std / 'clinical_runs_nasim_s2_20261012/R1_ls0/top20',
                              a.final_records, a.ls0_records)
    out.mkdir(parents=True)
    write_markdown(models, graded, classes, corr, out / 'comparison_table.md')
    (out / 'comparison_table.json').write_text(json.dumps(
        {'k': K, 'fold': 'test', 'seeds': list(SEEDS), 'headline': HEADLINE, 'models': models, 'graded': graded,
         'per_class_label_smoothing': classes, 'per_class_correlation': corr}, indent=1))
    print(out / 'comparison_table.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
