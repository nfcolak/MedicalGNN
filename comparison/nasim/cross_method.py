"""Section 2 cross-method analysis: every recipe/fidelity arm minus every final baseline, K=20.

Analysis from stored predictions only (no training, no model is loaded). Arms: the CEI-GNN mp2 training-recipe
controls R1-R3, the ProtGNN projection-selected arm P1, the GCHM-PNA cosine-schedule arm C1 and the six-cell GSAT
grid G (one seed each); the final CEI-GNN mp2 is carried as an anchor row. Baselines: ProtGNN final and
projection-selected, GraphCare, GSAT, GCHM-PNA final and cosine. Every pair is the paired two-level bootstrap of
`comparison.ablation.arms_vs_ref` (matched seeds, then patients, one shared patient resample per replicate;
`load_run`, `bootstrap_f1`, `two_level` are reused, not copied) on difference = arm - baseline. An arm is never
compared with its own runs (P1 vs ProtGNN projection-selected, C1 vs GCHM-PNA cosine).

    python -m comparison.nasim.cross_method --runs <comparison/standardized> --out <new dir> \
        [--fold test|validation] [--stored-bolum2 <bolum2-sonuc.md>]

`--fold` names the fold that leads the tables (default test, the reporting fold); both folds are always computed
and written, the other one labelled "validation (decision fold)" or "test". Decisions (checkpoint, HPO, which
baseline) use VALIDATION; every reported result is on TEST. `--stored-bolum2` compares the validation fold with the
earlier validation tables (baseline means, arm means and every arm-vs-baseline difference with its interval) and
writes `validation_reproduction.json`. `--out` must not exist; run folders are read-only.

Reading of a pair: ahead / behind when the seeds+patients 95% interval excludes 0, otherwise unresolved. With
three matched seeds (one for the G grid) the seed level is a weak estimate of seed variance: read the interval with
the per-seed differences and the matched-seed win counts.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from comparison.ablation import arms_vs_ref as A  # noqa: E402

FOLDS = ('test', 'validation')
FOLD_TAG = {'test': 'test', 'validation': 'validation (decision fold)'}
FINAL = 'clinical_runs_fair_v10t_20261011/top20'
S2 = 'clinical_runs_nasim_s2_20261012'
P1_DIR, C1_DIR = f'{S2}/P1_proj10_select/top20', f'{S2}/C1_cosine/top20'
GRID = [(ib, r) for ib in ('1.0', '0.1', '0.01') for r in ('0.5', '0.7')]
# label -> (run folder under --runs, method key)
ARMS = {'Final CEI (anchor)': (FINAL, 'cei_mp2'),
        'R1': (f'{S2}/R1_ls0/top20', 'cei_mp2'),
        'R2': (f'{S2}/R2_ls0_wd/top20', 'cei_mp2'),
        'R3': (f'{S2}/R3_ls0_wd_de0/top20', 'cei_mp2'),
        'P1': (P1_DIR, 'protgnn'),
        'C1': (C1_DIR, 'gchm_pna')}
for _ib, _r in GRID:
    ARMS[f'G ib{_ib} r{_r}'] = (f'{S2}/G_ib{_ib}_r{_r}/top20', 'gsat')
BASES = {'ProtGNN final': (FINAL, 'protgnn'), 'ProtGNN proj-select': (P1_DIR, 'protgnn'),
         'GraphCare': (FINAL, 'graphcare'), 'GSAT': (FINAL, 'gsat'),
         'GCHM-PNA final': (FINAL, 'gchm_pna'), 'GCHM-PNA cosine': (C1_DIR, 'gchm_pna')}
# the five baseline columns of the earlier validation table (ProtGNN final was not a column there)
EARLIER_FIVE = ('GCHM-PNA final', 'GCHM-PNA cosine', 'ProtGNN proj-select', 'GraphCare', 'GSAT')
PROTOCOL_NOTE = A.PROTOCOL_NOTE


def load_all(runs_root):
    """{(folder, method, seed): loaded run} for every distinct (folder, method) used by an arm or a baseline."""
    loaded, notes = {}, []
    for folder, method in sorted(set(ARMS.values()) | set(BASES.values())):
        found, skipped = A.discover(Path(runs_root) / folder, {method})
        if skipped:
            notes.append(f'{folder}: no result.json (skipped): ' + ', '.join(skipped))
        if not found:
            raise A.Refusal(f'no {method} runs in {folder}')
        for (m, s), d in found.items():
            loaded[(folder, m, s)] = A.load_run(d, m, s)
            loaded[(folder, m, s)]['row']['path'] = f'{folder}/{d.name}'
    keys = list(loaded)
    base = loaded[keys[0]]['binding']
    for k in keys:
        for key in A.SAME_KEYS:
            if loaded[k]['binding'].get(key) != base.get(key):
                raise A.Refusal(f'{key} differs between {keys[0]} and {k}: not one task')
    return loaded, base, notes


def verdict(ci):
    return 'ahead' if ci[0] > 0 else 'behind' if ci[1] < 0 else 'unresolved'


def entity_stats(loaded, spec):
    folder, method = spec
    seeds = sorted(s for (f, m, s) in loaded if f == folder and m == method)
    rows = [loaded[(folder, method, s)]['row'] for s in seeds]
    return {'seeds': seeds, 'folder': folder, 'method': method,
            'validation_macro_f1': A.mean_sd([r['val_macro_f1'] for r in rows]),
            'test_macro_f1': A.mean_sd([r['test_macro_f1'] for r in rows]),
            'validation_accuracy': A.mean_sd([r['val_accuracy'] for r in rows]),
            'test_accuracy': A.mean_sd([r['test_accuracy'] for r in rows])}


def analyze(runs_root):
    loaded, base, notes = load_all(runs_root)
    boots, info = {}, {}
    for fold in FOLDS:
        boots[fold], info[fold] = A.bootstrap_f1(loaded, fold)
    doc = {'protocol': A.PROTOCOL, 'protocol_note': PROTOCOL_NOTE, 'notes': notes,
           'task': {'top_k_labels': base.get('top_k_labels'), 'num_classes': base.get('num_classes'), 'folds': info},
           'bootstrap': {'B': A.B, 'rng_patients': A.SEED_PATIENTS, 'rng_seeds': A.SEED_SEEDS,
                         'statistic': 'mean over matched seeds of macro-F1(arm) - macro-F1(baseline), '
                                      'two-level (seeds, patients) percentile 95% interval'},
           'arms': {n: entity_stats(loaded, s) for n, s in ARMS.items()},
           'baselines': {n: entity_stats(loaded, s) for n, s in BASES.items()}, 'pairs': [], 'wins': {}}
    for arm, aspec in ARMS.items():
        for bname, bspec in BASES.items():
            if aspec == bspec:
                continue
            matched = sorted(set(doc['arms'][arm]['seeds']) & set(doc['baselines'][bname]['seeds']))
            pair = {'arm': arm, 'baseline': bname, 'matched_seeds': matched}
            for fold in FOLDS:
                pre = 'val' if fold == 'validation' else 'test'
                delta = np.stack([boots[fold][(*aspec, s)] - boots[fold][(*bspec, s)] for s in matched])
                two = A.two_level(delta)
                per_seed = [loaded[(*aspec, s)]['row'][f'{pre}_macro_f1'] - loaded[(*bspec, s)]['row'][f'{pre}_macro_f1']
                            for s in matched]
                sp = two['seeds_and_patients']
                pair[fold] = {'delta': float(np.mean(per_seed)), 'per_seed_delta': per_seed,
                              'ci_seeds_patients': sp['ci95'], 'p_delta_gt_0': sp['p_delta_gt_0'],
                              'ci_patients_only': two['patients_only']['ci95'],
                              'seeds_ahead': int(sum(x > 0 for x in per_seed)), 'verdict': verdict(sp['ci95'])}
            doc['pairs'].append(pair)
    for arm in ARMS:
        for fold in FOLDS:
            w = {}
            for name, subset in (('all_baselines', tuple(BASES)), ('earlier_five', EARLIER_FIVE)):
                ps = [p for p in doc['pairs'] if p['arm'] == arm and p['baseline'] in subset]
                vals = [x for p in ps for x in p[fold]['per_seed_delta']]
                w[name] = {'ahead': int(sum(x > 0 for x in vals)), 'of': len(vals),
                           'min': min(vals) if vals else None, 'max': max(vals) if vals else None}
            doc['wins'].setdefault(arm, {})[fold] = w
    return doc


def pair_of(doc, arm, base):
    return next((p for p in doc['pairs'] if p['arm'] == arm and p['baseline'] == base), None)


def cell(p, fold):
    if p is None:
        return 'same runs'
    x = p[fold]
    star = '*' if x['verdict'] != 'unresolved' else ''
    return f"{x['delta']:+.4f} [{x['ci_seeds_patients'][0]:+.4f}, {x['ci_seeds_patients'][1]:+.4f}]{star} {x['seeds_ahead']}/{len(x['per_seed_delta'])}"


def render(doc, lead):
    order = [lead] + [f for f in FOLDS if f != lead]
    t = doc['task']
    L = ['# Section 2 cross-method analysis (arm minus final baseline)', '',
         f"Protocol {doc['protocol']}. {doc['protocol_note']} K={t['top_k_labels']} ({t['num_classes']} classes). "
         f"Test {t['folds']['test']['rows']} rows / {t['folds']['test']['patients']} patients; validation (decision fold) "
         f"{t['folds']['validation']['rows']} rows / {t['folds']['validation']['patients']} patients.",
         f"Difference = arm - baseline in macro-F1, mean over the matched seeds; [95% interval of the paired two-level "
         f"bootstrap, B={doc['bootstrap']['B']}: matched seeds, then patients]; `*` = interval excludes 0; the last "
         f"number is the matched seeds in which the arm is ahead / the matched seeds. Intervals are conditional on the "
         f"fixed fold; the seed level rests on 3 matched seeds (1 for the G grid cells), so read them with the "
         f"per-seed counts.", '']
    for n in doc['notes']:
        L.append(f'NOTE: {n}')
    for fold in order:
        L += ['', f'## {FOLD_TAG[fold].capitalize()}', '', 'Macro-F1 (mean ± SD over seeds):', '',
              '| run set | method | seeds | macro-F1 |', '|---|---|---|---|']
        for grp, label in (('arms', 'arm'), ('baselines', 'baseline')):
            for n, e in doc[grp].items():
                L.append(f"| {n} ({label}) | {e['method']} | {','.join(map(str, e['seeds']))} | "
                         f"{A.fmt(e[fold + '_macro_f1'])} |")
        L += ['', 'Arm minus baseline:', '', '| arm | ' + ' | '.join(BASES) + ' |', '|---|' + '---|' * len(BASES)]
        for arm in ARMS:
            L.append(f'| {arm} | ' + ' | '.join(cell(pair_of(doc, arm, b), fold) for b in BASES) + ' |')
        L += ['', 'Matched-seed win counts (seeds in which the arm is ahead / matched-seed comparisons; min and max of the '
                  'per-seed differences):', '', '| arm | all six baselines | the five baselines of the earlier validation table |',
              '|---|---|---|']
        for arm in ARMS:
            w = doc['wins'][arm][fold]
            L.append(f"| {arm} | " + ' | '.join(
                f"{w[k]['ahead']}/{w[k]['of']} ({w[k]['min']:+.4f} to {w[k]['max']:+.4f})"
                for k in ('all_baselines', 'earlier_five')) + ' |')
    L += ['', '## Pair detail', '', '| arm | baseline | matched seeds | fold | d | CI seeds+patients | P(d>0) | CI patients only '
          '| per-seed d | reading |', '|---|---|---|---|---|---|---|---|---|---|']
    for p in doc['pairs']:
        for fold in order:
            x = p[fold]
            L.append(f"| {p['arm']} | {p['baseline']} | {','.join(map(str, p['matched_seeds']))} | {FOLD_TAG[fold]} | "
                     f"{x['delta']:+.4f} | [{x['ci_seeds_patients'][0]:+.4f}, {x['ci_seeds_patients'][1]:+.4f}] | "
                     f"{x['p_delta_gt_0']:.3f} | [{x['ci_patients_only'][0]:+.4f}, {x['ci_patients_only'][1]:+.4f}] | "
                     + ', '.join(f'{v:+.4f}' for v in x['per_seed_delta']) + f" | {x['verdict']} |")
    return '\n'.join(L) + '\n'


NUM = r'([+-]?\d\.\d{4})'
DELTA_CELL = re.compile(NUM + r' \[' + NUM + ', ' + NUM + r'\]')


def parse_stored(text):
    """Stored validation table of the earlier analysis: baseline means (header), arm means, arm - baseline cells."""
    sec = text.split('\n## 2.', 1)[1].split('\n## 3.', 1)[0]
    head = next(l for l in sec.splitlines() if l.startswith('| CEI kolu'))
    bases = [(m.group(1).strip(), float(m.group(2))) for m in re.finditer(r'\|?\s*([^|()]+?) \((0\.\d{4})\)', head)]
    cells = {}
    for l in sec.splitlines():
        parts = [c.strip() for c in l.strip().strip('|').split('|')]
        if len(parts) == len(bases) + 1 and parts[0] in ('Final', 'R1', 'R2', 'R3'):
            for (b, _), c in zip(bases, parts[1:]):
                m = DELTA_CELL.fullmatch(c)
                cells[(parts[0], b)] = tuple(float(x) for x in m.groups())
    sec1 = text.split('\n## 1.', 1)[1].split('\n## 2.', 1)[0]
    arm_means = {}
    for l in sec1.splitlines():
        parts = [c.strip() for c in l.strip().strip('|').split('|')]
        if len(parts) >= 5 and parts[0] in ('Final CEI', 'R1', 'R2', 'R3'):
            arm_means['Final' if parts[0] == 'Final CEI' else parts[0]] = float(parts[2].split('±')[0])
    return {'baseline_means': dict(bases), 'arm_means': arm_means, 'cells': cells}


def reproduce(doc, stored_path):
    st = parse_stored(Path(stored_path).read_text())
    rows, diffs_pt, diffs_ci = [], [], []
    for b, m in st['baseline_means'].items():
        got = doc['baselines'][b]['validation_macro_f1']['mean']
        rows.append({'what': f'validation mean {b}', 'stored': m, 'got': round(got, 4), 'diff': round(got, 4) - m})
        diffs_pt.append(abs(round(got, 4) - m))
    for a, m in st['arm_means'].items():
        key = 'Final CEI (anchor)' if a == 'Final' else a
        got = doc['arms'][key]['validation_macro_f1']['mean']
        rows.append({'what': f'validation mean {a}', 'stored': m, 'got': round(got, 4), 'diff': round(got, 4) - m})
        diffs_pt.append(abs(round(got, 4) - m))
    for (a, b), (d, lo, hi) in st['cells'].items():
        p = pair_of(doc, 'Final CEI (anchor)' if a == 'Final' else a, b)['validation']
        got = (p['delta'], *p['ci_seeds_patients'])
        diffs_pt.append(abs(round(got[0], 4) - d))
        ci = [abs(round(g, 4) - s) for g, s in zip(got[1:], (lo, hi))]
        diffs_ci += ci
        rows.append({'what': f'{a} - {b}', 'stored': [d, lo, hi], 'got': [round(g, 4) for g in got],
                     'diff_point': round(got[0], 4) - d, 'diff_ci': [round(got[1], 4) - lo, round(got[2], 4) - hi]})
    return {'compared_cells': len(st['cells']), 'max_abs_diff_point_estimates_4dp': max(diffs_pt),
            'max_abs_diff_ci_bounds_4dp': max(diffs_ci), 'rows': rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--runs', default=str(ROOT / 'comparison' / 'standardized'), help='folder holding the run folders')
    ap.add_argument('--out', required=True, help='new output directory (must not exist)')
    ap.add_argument('--fold', choices=FOLDS, default='test', help='fold that leads the tables (both are computed)')
    ap.add_argument('--stored-bolum2', help='earlier validation tables (markdown) to reproduce')
    args = ap.parse_args(argv)
    try:
        out = Path(args.out)
        if out.exists():
            raise A.Refusal(f'--out already exists: {out}')
        doc = analyze(args.runs)
        rep = reproduce(doc, args.stored_bolum2) if args.stored_bolum2 else None
    except A.Refusal as e:
        print(f'REFUSED: {e}', file=sys.stderr)
        return 2
    out.mkdir(parents=True)
    (out / 'cross_method.json').write_text(json.dumps(doc, indent=1))
    (out / 'cross_method.md').write_text(render(doc, args.fold))
    if rep:
        (out / 'validation_reproduction.json').write_text(json.dumps(rep, indent=1))
        print(f"validation reproduction: {rep['compared_cells']} cells, max abs diff point estimates "
              f"{rep['max_abs_diff_point_estimates_4dp']:.4f}, CI bounds {rep['max_abs_diff_ci_bounds_4dp']:.4f} (4 dp)")
    print(f'wrote {out}/cross_method.md and cross_method.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
