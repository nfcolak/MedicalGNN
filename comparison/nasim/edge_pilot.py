"""Edge-direction pilot (10k training rows, K=20, seed 1234): bidirectional minus forward per baseline, and CEI-GNN mp2 minus each baseline arm.

Analysis from stored predictions only (no training, no model is loaded). The pilot trained GraphCare-style
(adapted), GSAT and ProtGNN once with forward edges and once with bidirectional edges, plus CEI-GNN mp2 (which
already reads both directions internally), all on the first 10000 training rows (protocol override `train_limit`)
and scored at the validation-selected checkpoint. Per pair the macro-F1 difference gets a patient bootstrap
(B=2000, patients resampled with all their rows, one resample shared by all arms, rng 20261016) built on
`comparison.ablation.arms_vs_ref.bootstrap_f1` (reused; only its patient seed is set for the pilot). One seed, so
the intervals are conditional on that seed and on the fixed fold.

    python -m comparison.nasim.edge_pilot --runs <clinical_runs_bidir_pilot10k folder> --out <new dir> \
        [--fold test|validation] [--stored-pilot results.json] [--stored-cei cei_vs_baselines.json]

`--fold` names the fold that leads the tables (default test, the reporting fold); both folds are always computed.
Decisions (which edge direction to adopt) use VALIDATION; every reported result is on TEST. `--stored-pilot` and
`--stored-cei` compare the validation fold with the earlier validation results and write
`validation_reproduction.json`. Each run is checked: protocol fair_v10t with overrides ['train_limit'], completed,
test scored once, identical split and 10000 training rows in all arms, macro-F1 and accuracy recomputed from the
stored predictions equal to result.json (1e-6). The binding of the two arms of a baseline may differ only in the
edge-direction keys; anything else is listed. `--out` must not exist; run folders are read-only.
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
from core.contracts import sample_ids_sha256  # noqa: E402

FOLDS = ('test', 'validation')
FOLD_TAG = {'test': 'test', 'validation': 'validation (decision fold)'}
PILOT_RNG = 20261016
BASELINES = {'graphcare': 'GraphCare', 'gsat': 'GSAT', 'protgnn': 'ProtGNN'}
RUNS = {'cei_mp2': 'cei_mp2_forward_seed1234'}
for _m in BASELINES:
    for _d in ('forward', 'bidirectional'):
        RUNS[f'{_m}_{_d}'] = f'{_m}_{_d}_seed1234'
# binding keys that legitimately differ between the two edge-direction arms of one baseline
EXPECTED_DIFF = re.compile(r'^(edge_direction|edge_dim|edge_feature_layout|meta_relations|num_meta_relations|num_relations|'
                           r'parameter_count|active_parameter_count|selected_(validation|test)\..*|method_config\.architecture\.'
                           r'(edge_dim|num_relations|num_triples|.*parameter_count))$')


def flat(o, p=''):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flat(v, f'{p}.{k}' if p else k))
    else:
        out[p] = json.dumps(o, sort_keys=True)
    return out


def load_pilot_run(path, name):
    path = Path(path)
    result = json.loads((path / 'result.json').read_text())
    binding = json.loads((path / 'binding.json').read_text())
    if result.get('status') != 'completed':
        raise A.Refusal(f"{name}: status {result.get('status')!r}")
    if binding.get('protocol') != A.PROTOCOL or binding.get('protocol_overrides') != ['train_limit']:
        raise A.Refusal(f"{name}: protocol {binding.get('protocol')!r} overrides {binding.get('protocol_overrides')!r}")
    if result.get('selection_fold') != 'validation' or result.get('test_evaluations') != 1:
        raise A.Refusal(f'{name}: selection fold / test evaluations not (validation, 1)')
    folds = {'validation': A.load_fold(path, 'validation.npz'), 'test': A.load_fold(path, 'test.npz')}
    for fold, f in folds.items():
        want = (binding.get('split_sample_ids_sha256') or {}).get(fold)
        if want and sample_ids_sha256(f['sample_ids'].tolist()) != want:
            raise A.Refusal(f'{name}: {fold} sample_ids differ from the binding hash')
    row = {'name': name, 'selected_epoch': result.get('selected_epoch'), 'parameter_count': binding.get('parameter_count'),
           'train_rows': binding['counts']['train'], 'minutes': round(result['total_seconds'] / 60, 1),
           'edge_direction': binding.get('edge_direction')}
    for fold, pre, stored in (('validation', 'val', result.get('metrics') or {}), ('test', 'test', result.get('test') or {})):
        f1, acc = A.repo_metrics(folds[fold]['proba'], folds[fold]['y'])
        for what, got in (('macro_f1', f1), ('accuracy', acc)):
            if not isinstance(stored.get(what), (int, float)) or abs(got - stored[what]) > A.TOL:
                raise A.Refusal(f'{name}: recomputed {fold} {what} {got:.6f} != result.json {stored.get(what)!r}')
        row[f'{pre}_macro_f1'], row[f'{pre}_accuracy'] = f1, acc
    return {'row': row, 'binding': binding, 'folds': folds}


def analyze(runs_root):
    loaded = {n: load_pilot_run(Path(runs_root) / d, n) for n, d in RUNS.items()}
    ref = loaded['cei_mp2']
    for n, x in loaded.items():
        if x['binding']['counts'] != ref['binding']['counts']:
            raise A.Refusal(f'{n}: split/train counts differ from the CEI run')
        for key in ('top_k_labels', 'num_classes', 'split_sample_ids_sha256', 'kept_label_indices', 'targets_sha256'):
            if x['binding'].get(key) != ref['binding'].get(key):
                raise A.Refusal(f'{n}: {key} differs from the CEI run (not one task)')
    notes = []
    diffs = {}
    for m in BASELINES:
        f, b = flat(loaded[f'{m}_forward']['binding']), flat(loaded[f'{m}_bidirectional']['binding'])
        diffs[m] = sorted(k for k in set(f) | set(b) if f.get(k) != b.get(k))
        odd = [k for k in diffs[m] if not EXPECTED_DIFF.match(k)]
        if odd:
            notes.append(f'{BASELINES[m]}: unexpected binding differences between the arms: ' + ', '.join(odd))
        if loaded[f'{m}_forward']['binding'].get('edge_direction') != 'forward' or \
                loaded[f'{m}_bidirectional']['binding'].get('edge_direction') != 'bidirectional':
            raise A.Refusal(f'{m}: edge_direction not (forward, bidirectional)')
    saved = dict(A.SEED_PATIENTS)
    A.SEED_PATIENTS.update({f: PILOT_RNG for f in FOLDS})
    try:
        boots, info = {}, {}
        for fold in FOLDS:
            boots[fold], info[fold] = A.bootstrap_f1(loaded, fold)
    finally:
        A.SEED_PATIENTS.clear()
        A.SEED_PATIENTS.update(saved)
    pairs = [(f'{BASELINES[m]} bidirectional - forward', f'{m}_bidirectional', f'{m}_forward') for m in BASELINES]
    pairs += [(f'CEI-GNN mp2 - {BASELINES[m]} {d}', 'cei_mp2', f'{m}_{d}') for d in ('bidirectional', 'forward')
              for m in BASELINES]
    doc = {'protocol': A.PROTOCOL, 'protocol_note': A.PROTOCOL_NOTE, 'notes': notes,
           'task': {'top_k_labels': ref['binding'].get('top_k_labels'), 'num_classes': ref['binding'].get('num_classes'),
                    'seed': 1234, 'train_rows': ref['binding']['counts']['train'], 'folds': info},
           'bootstrap': {'B': A.B, 'rng_patients': PILOT_RNG, 'levels': 'patients only (one seed)'},
           'binding_differences_between_arms': diffs,
           'runs': {n: {k: v for k, v in x['row'].items()} for n, x in loaded.items()}, 'pairs': []}
    for label, a, b in pairs:
        p = {'pair': label, 'a': a, 'b': b}
        for fold in FOLDS:
            pre = 'val' if fold == 'validation' else 'test'
            two = A.two_level((boots[fold][a] - boots[fold][b])[None])['seeds_and_patients']
            p[fold] = {'f1_a': loaded[a]['row'][f'{pre}_macro_f1'], 'f1_b': loaded[b]['row'][f'{pre}_macro_f1'],
                       'delta': loaded[a]['row'][f'{pre}_macro_f1'] - loaded[b]['row'][f'{pre}_macro_f1'],
                       'ci95': two['ci95'], 'p_delta_gt_0': two['p_delta_gt_0']}
        doc['pairs'].append(p)
    return doc


def render(doc, lead):
    order = [lead] + [f for f in FOLDS if f != lead]
    t = doc['task']
    L = ['# Edge-direction pilot (10k training rows)', '',
         f"Protocol {doc['protocol']} with the `train_limit` override ({t['train_rows']} training rows), K={t['top_k_labels']} "
         f"({t['num_classes']} classes), seed {t['seed']}, one run per arm. {doc['protocol_note']} Test "
         f"{t['folds']['test']['rows']} rows / {t['folds']['test']['patients']} patients; validation (decision fold) "
         f"{t['folds']['validation']['rows']} rows / {t['folds']['validation']['patients']} patients.",
         f"Difference = first minus second in macro-F1; [95% patient-bootstrap interval, B={doc['bootstrap']['B']}, one "
         f"resample shared by all arms, rng {doc['bootstrap']['rng_patients']}]; P = share of resamples with a positive "
         f"difference. One seed: the intervals are conditional on that seed and on the fixed fold.", '']
    for n in doc['notes']:
        L.append(f'NOTE: {n}')
    L += ['## Runs', '', '| run | edge direction | parameters | selected epoch | test macro-F1 | test accuracy | '
          'validation macro-F1 (decision fold) | validation accuracy |', '|---|---|---|---|---|---|---|---|']
    for n, r in doc['runs'].items():
        L.append(f"| {n} | {r['edge_direction']} | {r['parameter_count']} | {r['selected_epoch']} | {r['test_macro_f1']:.4f} | "
                 f"{r['test_accuracy']:.4f} | {r['val_macro_f1']:.4f} | {r['val_accuracy']:.4f} |")
    for title, sel in (('Bidirectional minus forward, per baseline', lambda p: 'bidirectional - forward' in p['pair']),
                       ('CEI-GNN mp2 minus each baseline arm', lambda p: p['pair'].startswith('CEI'))):
        L += ['', f'## {title}', '', '| pair | ' + ' | '.join(
            f"{FOLD_TAG[f]} a / b | d [CI] | P(d>0)" for f in order) + ' |', '|---|' + '---|---|---|' * len(order)]
        for p in filter(sel, doc['pairs']):
            L.append(f"| {p['pair']} | " + ' | '.join(
                f"{p[f]['f1_a']:.4f} / {p[f]['f1_b']:.4f} | {p[f]['delta']:+.4f} [{p[f]['ci95'][0]:+.4f}, {p[f]['ci95'][1]:+.4f}]"
                f" | {p[f]['p_delta_gt_0']:.3f}" for f in order) + ' |')
    L += ['', '## Binding differences between the two arms of a baseline', '']
    for m, d in doc['binding_differences_between_arms'].items():
        L.append(f"- {BASELINES[m]}: {', '.join(d)}")
    return '\n'.join(L) + '\n'


def reproduce(doc, pilot_path, cei_path):
    rows, diffs = [], []
    if pilot_path:
        for r in json.loads(Path(pilot_path).read_text()):
            p = next(x for x in doc['pairs'] if x['a'] == f"{r['method']}_bidirectional" and x['b'] == f"{r['method']}_forward")
            v = p['validation']
            got = {'f1_fwd': v['f1_b'], 'f1_bid': v['f1_a'], 'delta': v['delta'], 'ci_lo': v['ci95'][0],
                   'ci_hi': v['ci95'][1], 'p_pos': v['p_delta_gt_0']}
            want = {'f1_fwd': r['f1_fwd'], 'f1_bid': r['f1_bid'], 'delta': r['delta'], 'ci_lo': r['ci'][0],
                    'ci_hi': r['ci'][1], 'p_pos': r['p_pos']}
            run = doc['runs'][f"{r['method']}_forward"], doc['runs'][f"{r['method']}_bidirectional"]
            got.update(ep_fwd=run[0]['selected_epoch'], ep_bid=run[1]['selected_epoch'],
                       test_f1_fwd=run[0]['test_macro_f1'], test_f1_bid=run[1]['test_macro_f1'])
            want.update(ep_fwd=r['ep_fwd'], ep_bid=r['ep_bid'], test_f1_fwd=r['test_f1_fwd'], test_f1_bid=r['test_f1_bid'])
            d = {k: abs(got[k] - want[k]) for k in got}
            rows.append({'source': 'results.json', 'method': r['method'], 'max_abs_diff': max(d.values()), 'diffs': d})
            diffs.append(max(d.values()))
    if cei_path:
        for r in json.loads(Path(cei_path).read_text()):
            if r['run'] == 'cei_mp2':
                continue
            p = next(x for x in doc['pairs'] if x['a'] == 'cei_mp2' and x['b'] == r['run'])
            v = p['validation']
            got = {'cei_minus': v['delta'], 'ci_lo': v['ci95'][0], 'ci_hi': v['ci95'][1], 'p_cei_ahead': v['p_delta_gt_0'],
                   'val_f1': v['f1_b']}
            want = {'cei_minus': r['cei_minus'], 'ci_lo': r['ci'][0], 'ci_hi': r['ci'][1], 'p_cei_ahead': r['p_cei_ahead'],
                    'val_f1': r['val_f1']}
            d = {k: abs(got[k] - want[k]) for k in got}
            rows.append({'source': 'cei_vs_baselines.json', 'run': r['run'], 'max_abs_diff': max(d.values()), 'diffs': d})
            diffs.append(max(d.values()))
    return {'max_abs_diff': max(diffs), 'rows': rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--runs', default=str(ROOT / 'comparison' / 'standardized' / 'clinical_runs_bidir_pilot10k_20261016'),
                    help='folder holding the pilot run folders')
    ap.add_argument('--out', required=True, help='new output directory (must not exist)')
    ap.add_argument('--fold', choices=FOLDS, default='test', help='fold that leads the tables (both are computed)')
    ap.add_argument('--stored-pilot', help='earlier validation results.json (bidirectional vs forward) to reproduce')
    ap.add_argument('--stored-cei', help='earlier validation cei_vs_baselines.json to reproduce')
    args = ap.parse_args(argv)
    try:
        out = Path(args.out)
        if out.exists():
            raise A.Refusal(f'--out already exists: {out}')
        doc = analyze(args.runs)
        rep = reproduce(doc, args.stored_pilot, args.stored_cei) if (args.stored_pilot or args.stored_cei) else None
    except A.Refusal as e:
        print(f'REFUSED: {e}', file=sys.stderr)
        return 2
    out.mkdir(parents=True)
    (out / 'edge_pilot.json').write_text(json.dumps(doc, indent=1))
    (out / 'edge_pilot.md').write_text(render(doc, args.fold))
    if rep:
        (out / 'validation_reproduction.json').write_text(json.dumps(rep, indent=1))
        print(f"validation reproduction: {len(rep['rows'])} rows, max abs diff {rep['max_abs_diff']:.2e}")
    print(f'wrote {out}/edge_pilot.md and edge_pilot.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
