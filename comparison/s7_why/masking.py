"""Source masking: how much does each stored final model lean on history, labs or triage?

Inference only. Stored final runs (best.pt) of CEI-GNN mp2 and the two strongest validation baselines
(among GSAT, GraphCare-style (adapted), ProtGNN) score VALIDATION graphs whose history, laboratory or
triage information was removed from an in-memory copy of each raw graph record BEFORE encoding. Nothing is
refitted (the run's stored preprocessing.json is loaded), nothing is trained, no file of the inputs is
written, and the test fold is never opened. Decisions (which baselines) use VALIDATION only.

    python -m comparison.s7_why.masking --runs <runs root> --artifact <graph dir> --targets <targets.csv> \
        --out <new dir> [--ks 10,20,30] [--seeds 1234,2025,7] [--masks none,history,labs,triage] \
        [--limit N] [--execute] [--threads 2]

Without --execute the plan is printed (methods per K with their validation means, rows, masks, forward
passes) and nothing is loaded or written. With --execute --out must not exist.

Record format (read from core/tensorize.py, core/contracts.py, data/s4_graph/graph.py)
---------------------------------------------------------------------------------------
graph record (graphs.jsonl line): `sample_id`, `nodes`, `edges` (the encoder reads nothing else).
  node: `id` ('n<position>'), `kind` in {patient, visit, complaint, vital, measurement, analyte, knowledge,
        diagnosis}, `token`; measurements additionally carry `scope` ('index' | 'prior').
  edge: `source`, `target` (node ids), `relation`, `informative`, payload fields.
visit-membership sidecar record (visit_membership.jsonl, aligned line by line): `visit_ordinals` = [0..n-1]
  (source visits in time order; the INDEX visit is the last ordinal n-1; the artifact contains exactly one
  visit NODE, `visit:index`; prior visits exist only as ordinals), `membership_pairs` = sorted
  [ordinal, node position] pairs, `global_node_mask` (True only for the patient node). `num_visits` of the
  encoded tensors is len(visit_ordinals); CEI reads the within-visit membership pairs.
Scope of a node = its membership set: index-scope nodes contain ordinal n-1; prior-scope nodes (prior lab
  measurements `scope == 'prior'`, prior diagnoses, knowledge reached only through them) contain only
  ordinals < n-1. Complaints, vitals, index measurements and the visit node belong to the index ordinal
  only; analytes and knowledge nodes may belong to several ordinals.

Mask rules (applied identically for every method; the graph is copied, never edited in place)
-----------------------------------------------------------------------------------------------
  none     unchanged (parity reference).
  history  remove every non-global node whose membership does not contain the index ordinal (prior
           measurements, prior diagnoses and everything attached only to them) plus all incident edges;
           surviving nodes keep only the index ordinal; visit_ordinals becomes [0] (num_visits = 1).
           Patient and index visit are kept.
  labs     remove every measurement and analyte node (index and prior) and incident edges.
  triage   remove every vital and complaint node (index and prior) and incident edges, and blank the
           index visit's triage acuity (acuity -> None, read by the encoder as a missing value, as for
           visits without a recorded acuity): ED triage = vitals, chief complaint and acuity.
After the removal every node left without any edge is dropped (patient and index visit excepted); nodes
and edges are re-indexed (ids 'n0'..'n<m-1>' in the original order); membership pairs are remapped to the
new positions; ordinals that no longer hold any non-global node are dropped (index ordinal always kept)
and the remaining ordinals are renumbered 0..v-1, so num_visits stays consistent with the remaining
content (history -> 1; labs/triage may drop prior visits whose only content was removed). Fields that a
record does not provide in the expected form (index visit membership, prior-scope measurements outside the
history set) stop the run with an error instead of being guessed.

CEI-GNN reads absence evidence ("items not observed"); removing labs therefore also changes that block
(intended: absence of labs is part of the lab source).

Metrics: macro-F1 over the K classes of the run (repo definition, core.train.metrics) and accuracy per
run x mask; drop = metric(none) - metric(mask). The CEI-minus-baseline difference of seed-mean drops gets a
patient bootstrap 95% CI (B=1000, rng 20261017, one shared draw of patients for every method/seed/mask of a
K; negative = CEI drops less). Parity gate: unmasked probabilities must equal the stored validation.npz
rows (aligned by sample_id) within 1e-4 with 100% argmax agreement, otherwise that run is not scored and
the failure is written to parity.json.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SEED_RNG = 20261017
BOOTSTRAP_B = 1000
PARITY_TOL = 1e-4
FOLD = 'validation'
ALL_SEEDS = (1234, 2025, 7)
BASELINES = ('gsat', 'graphcare', 'protgnn')
MASKS = ('none', 'history', 'labs', 'triage')
PRETTY = {'cei_mp2': 'CEI-GNN mp2', 'gsat': 'GSAT', 'graphcare': 'GraphCare-style (adapted)',
          'protgnn': 'ProtGNN', 'gchm_pna': 'GCHM-PNA'}
V3_STATE = 'v3_state_L4_v5c_v10t_top{K}_20261011/K4.json'
LAB_KINDS = frozenset({'measurement', 'analyte'})
TRIAGE_KINDS = frozenset({'vital', 'complaint'})
ABSENCE_NOTE = ('CEI-GNN reads an absence block ("items not observed"); removing labs therefore also changes '
                'its absence evidence (intended).')


def shown(path):
    """Repo-relative text for a path; outside the repo only the final name (never an absolute path)."""
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return p.name


# ------------------------------------------------------------------------------------------------ masks
def apply_mask(graph, vm, mask):
    """Return (graph', vm', stats) for one raw record; inputs are never modified."""
    nodes, edges = graph['nodes'], graph['edges']
    ordinals = vm['visit_ordinals']
    nv, last = len(ordinals), len(ordinals) - 1
    n = len(nodes)
    if mask == 'none':
        return graph, vm, dict(nodes_removed=0, edges_removed=0, visits_removed=0, acuity_removed=False,
                               changed=False)
    member = [set() for _ in range(n)]
    for ordinal, pos in vm['membership_pairs']:
        member[pos].add(ordinal)
    glob = vm['global_node_mask']
    visit_pos = [i for i, x in enumerate(nodes) if x['kind'] == 'visit']
    if len(visit_pos) != 1 or member[visit_pos[0]] != {last}:
        raise ValueError('mask undefined: index visit node is not the unique member of the last ordinal')
    index_visit = visit_pos[0]
    if mask == 'history':
        remove = {i for i in range(n) if not glob[i] and last not in member[i]}
        if any(x.get('scope') == 'prior' and i not in remove for i, x in enumerate(nodes)):
            raise ValueError('mask undefined: prior-scope measurement belongs to the index visit')
        if any(x.get('scope') == 'index' and i in remove for i, x in enumerate(nodes)):
            raise ValueError('mask undefined: index-scope measurement outside the index visit')
    elif mask == 'labs':
        remove = {i for i in range(n) if nodes[i]['kind'] in LAB_KINDS}
    elif mask == 'triage':
        remove = {i for i in range(n) if nodes[i]['kind'] in TRIAGE_KINDS}
    else:
        raise ValueError(f'unknown mask {mask!r}')
    pos_of = {x['id']: i for i, x in enumerate(nodes)}
    kept_edges = [e for e in edges if pos_of[e['source']] not in remove and pos_of[e['target']] not in remove]
    degree = [0] * n
    for e in kept_edges:
        degree[pos_of[e['source']]] += 1
        degree[pos_of[e['target']]] += 1
    for i in range(n):
        if i not in remove and not glob[i] and i != index_visit and degree[i] == 0:
            remove.add(i)
    keep = [i for i in range(n) if i not in remove]
    new_pos = {old: new for new, old in enumerate(keep)}
    new_id = {nodes[old]['id']: f'n{new}' for old, new in new_pos.items()}
    new_nodes = [{**nodes[old], 'id': f'n{new}'} for old, new in new_pos.items()]
    acuity_removed = False
    if mask == 'triage':   # triage acuity is a field of the index visit node
        for node in new_nodes:
            if node['kind'] == 'visit' and node.get('acuity') is not None:
                node['acuity'] = None
                acuity_removed = True
    new_edges = [{**e, 'source': new_id[e['source']], 'target': new_id[e['target']]}
                 for e in edges if e['source'] in new_id and e['target'] in new_id]
    if mask == 'history':
        live = {last}
        member = [({last} & m) for m in member]
    else:
        live = {last} | {o for old in keep for o in member[old] if not glob[old]}
    order = sorted(live)
    remap = {o: i for i, o in enumerate(order)}
    pairs = sorted({(remap[o], new_pos[old]) for old in keep for o in member[old] if o in remap})
    new_vm = {'contract_version': vm['contract_version'], 'sample_id': vm['sample_id'],
              'visit_ordinals': list(range(len(order))),
              'membership_pairs': [list(p) for p in pairs],
              'global_node_mask': [bool(glob[old]) for old in keep]}
    new_graph = {'sample_id': graph['sample_id'], 'nodes': new_nodes, 'edges': new_edges}
    stats = dict(nodes_removed=n - len(keep), edges_removed=len(edges) - len(new_edges),
                 visits_removed=nv - len(order), acuity_removed=acuity_removed,
                 changed=(n - len(keep) > 0 or len(edges) != len(new_edges) or nv != len(order)
                          or acuity_removed))
    return new_graph, new_vm, stats


# ------------------------------------------------------------------------------------------------ metrics
def fast_metrics(y, pred, k, w=None):
    """Macro-F1 over labels 0..k-1 (zero_division=0) and accuracy from a (weighted) confusion matrix."""
    cm = np.bincount(y * k + pred, weights=w, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    den = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    return float(f1.mean()), float(tp.sum() / cm.sum())


# ------------------------------------------------------------------------------------------------ planning
def run_dir(runs, k, method, seed):
    return Path(runs) / f'top{k}' / f'{method}_seed{seed}'


def read_json(path):
    with open(path) as stream:
        return json.load(stream)


def subsample_rows(sample_ids, subjects, limit):
    """Deterministic patient-level subsample (rng 20261017): whole patients, rows kept in stored order."""
    if limit is None or limit >= len(sample_ids):
        return np.arange(len(sample_ids))
    uniq, inverse = np.unique(subjects, return_inverse=True)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(SEED_RNG)
    total, chosen = 0, np.zeros(len(uniq), dtype=bool)
    for p in rng.permutation(len(uniq)):
        if total + counts[p] <= limit:
            chosen[p] = True
            total += counts[p]
            if total == limit:
                break
    return np.flatnonzero(chosen[inverse])


def build_plan(a):
    plan = {'fold': FOLD, 'masks': list(a.masks), 'seeds': list(a.seeds), 'limit': a.limit,
            'threads': a.threads, 'bootstrap': {'B': BOOTSTRAP_B, 'rng': SEED_RNG, 'unit': 'patient'},
            'baseline_selection': f'highest mean validation macro-F1 over seeds {list(ALL_SEEDS)} among '
                                  f'{list(BASELINES)} (result.json, selection fold validation)',
            'per_k': {}}
    for k in a.ks:
        stats = {}
        for method in a.pool:
            vals = []
            for s in ALL_SEEDS:
                r = read_json(run_dir(a.runs, k, method, s) / 'result.json')
                if r.get('status') != 'completed' or r.get('selection_fold') != FOLD:
                    raise ValueError(f'top{k}/{method}_seed{s}: run is not a completed validation-selected run')
                vals.append(float(r['metrics']['macro_f1']))
            stats[method] = float(np.mean(vals))
        if a.methods:
            methods = list(a.methods)
        else:
            ranked = sorted((m for m in BASELINES), key=lambda m: -stats[m])
            methods = ['cei_mp2'] + ranked[:2]
        ref = None
        rows = {}
        for method in methods:
            for s in a.seeds:
                z = np.load(run_dir(a.runs, k, method, s) / 'validation.npz', allow_pickle=True)
                ids = [str(x) for x in z['sample_ids']]
                if ref is None:
                    ref = (ids, np.asarray(z['subjects']).astype(str), np.asarray(z['y']))
                elif sorted(ids) != sorted(ref[0]):
                    raise ValueError(f'top{k}: validation rows differ between runs')
        sel = subsample_rows(ref[0], ref[1], a.limit)
        rows['sample_ids'] = [ref[0][i] for i in sel]
        rows['subjects'] = ref[1][sel]
        rows['y'] = ref[2][sel]
        binding = read_json(run_dir(a.runs, k, 'cei_mp2', a.seeds[0]) / 'binding.json')
        names = binding['label_order']
        if len(names) != k:
            names = [names[i] for i in binding['kept_label_indices']]
        plan['per_k'][k] = {
            'task': {'K': k, 'class_set': list(names), 'fold': FOLD,
                     'rows': len(sel), 'rows_full_fold': len(ref[0]),
                     'patients': int(len(np.unique(rows['subjects']))), 'seeds': list(a.seeds)},
            'methods': methods,
            'validation_mean_macro_f1': {m: round(v, 6) for m, v in stats.items()},
            'runs': len(methods) * len(a.seeds),
            'forward_passes': len(methods) * len(a.seeds) * len(a.masks),
            'graph_forwards': len(methods) * len(a.seeds) * len(a.masks) * len(sel),
            '_rows': rows}
    return plan


def public_plan(plan):
    out = {k: v for k, v in plan.items() if k != 'per_k'}
    out['per_k'] = {str(k): {kk: vv for kk, vv in v.items() if kk != '_rows'} for k, v in plan['per_k'].items()}
    return out


def print_plan(plan):
    print(f"plan: fold {plan['fold']}, masks {plan['masks']}, seeds {plan['seeds']}, limit {plan['limit']}, "
          f"threads {plan['threads']}")
    print(f"baseline choice: {plan['baseline_selection']}")
    for k, v in plan['per_k'].items():
        t = v['task']
        means = ', '.join(f'{m} {x:.4f}' for m, x in v['validation_mean_macro_f1'].items())
        print(f"K={k}: rows {t['rows']} of {t['rows_full_fold']} ({t['patients']} patients), "
              f"methods {v['methods']}")
        print(f"   validation mean macro-F1 (3 seeds): {means}")
        print(f"   runs {v['runs']}, forward passes {v['forward_passes']} "
              f"({v['graph_forwards']} graph forwards), no checkpoint loaded in plan mode")


# ------------------------------------------------------------------------------------------------ execute
def check_targets(targets, rows):
    """Aggregate check: every scored row is a validation row of the targets file with the same subject."""
    want = dict(zip(rows['sample_ids'], rows['subjects']))
    seen = 0
    with open(targets, newline='') as stream:
        for rec in csv.DictReader(stream):
            sid = rec['sample_id']
            if sid in want:
                if rec['split'] != FOLD or str(rec['subject_id']) != str(want[sid]):
                    raise ValueError('scored row is not a validation row of the targets file')
                seen += 1
    if seen != len(want):
        raise ValueError('scored rows missing from the targets file')


def load_graphs(artifact, membership_name, sample_ids):
    from core.contracts import iter_graphs_with_membership
    wanted = set(sample_ids)
    got = {g['sample_id']: (g, vm) for g, vm in iter_graphs_with_membership(
        Path(artifact) / 'graphs.jsonl', Path(artifact) / membership_name, wanted_ids=wanted)}
    if len(got) != len(wanted):
        raise ValueError('some scored rows are absent from the graph artifact')
    return [got[s] for s in sample_ids]


class Shim:
    """Stands in for comparison.final.run.Paths: only the CEI v3 state path is read by args()."""

    def __init__(self, events):
        self.events = events

    def v3_state(self, k):
        return self.events / V3_STATE.format(K=k)


def make_args(T, final_args, shim, a, method, k, seed):
    cli = final_args(shim, method, k) + ['--top-k-labels', str(k), '--artifact', str(a.artifact),
                                         '--targets', str(a.targets), '--output', 'unused_no_output',
                                         '--seed', str(seed), '--cpu-threads', str(a.threads)]
    p = T.parser()
    return T.normalize_method_args(p.parse_args(cli), p)


def build_model(T, M, method, args, prep, data0, k, state_dict):
    if method == 'gchm_pna':
        model = M['ClinicalGNN'](
            num_classes=k, num_tokens=len(prep['vocabulary']), node_dim=data0.x.size(1),
            edge_dim=data0.edge_attr.size(1), hidden=args.hidden, layers=args.layers, dropout=args.dropout,
            token_dim=args.token_dim, use_edge_payload=not args.no_edge_payload,
            use_message_passing=not args.no_message_passing, conv=args.conv,
            num_triples=M['triple_count'](prep, args.edge_direction), payload_dim=M['PAYLOAD_WIDTH'],
            heads=args.heads, degree_histogram=state_dict['degree_histogram'], modulation=args.modulation)
    else:
        model = M['build_method'](
            args.method, num_tokens=len(prep['vocabulary']), node_dim=data0.x.size(1),
            edge_dim=data0.edge_attr.size(1), num_classes=k, hidden=args.hidden, layers=args.layers,
            dropout=args.dropout, token_dim=args.token_dim,
            num_triples=M['triple_count'](prep, args.edge_direction), args=args)
        hook = getattr(model, 'verify_run_preprocessing', None)
        if callable(hook):
            hook(T.preprocessing_state(prep))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def execute(a, plan):
    import torch
    from torch_geometric.loader import DataLoader
    from core import train as T
    from core.registry import ClinicalMethodAdapter, build_method
    from core.model import ClinicalGNN
    from core.tensorize import (PAYLOAD_WIDTH, encode_graph, load_preprocessing, preprocessing_state,
                                triple_count)
    from comparison.final.run import args as final_args
    M = dict(build_method=build_method, ClinicalGNN=ClinicalGNN, triple_count=triple_count,
             PAYLOAD_WIDTH=PAYLOAD_WIDTH)
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'plan.json').write_text(json.dumps(public_plan(plan), indent=1, sort_keys=True) + '\n')
    shim = Shim(Path(a.artifact).resolve().parent)

    parity_rows, csv_rows, mask_stats, metric_check = [], [], {}, []
    summary = {'fold': FOLD, 'test_fold_touched': False, 'ks': {}, 'note': ABSENCE_NOTE,
               'drop_definition': 'metric(none) - metric(mask); difference = CEI drop - baseline drop '
                                  '(negative = CEI drops less)',
               'bootstrap': plan['bootstrap'], 'seeds': list(a.seeds), 'limit': a.limit, 'threads': a.threads}
    for k, kplan in plan['per_k'].items():
        rows = kplan['_rows']
        ids, y = rows['sample_ids'], np.asarray(rows['y'], dtype=np.int64)
        check_targets(a.targets, rows)
        binding0 = read_json(run_dir(a.runs, k, kplan['methods'][0], a.seeds[0]) / 'binding.json')
        membership_name = binding0['artifact_visit_membership_file']
        raw = load_graphs(a.artifact, membership_name, ids)
        preds = {}
        mask_acc = {m: dict(graphs=len(raw), changed=0, nodes=0, edges=0, visits=0, acuity=0, nodes_before=0,
                            edges_before=0)
                    for m in a.masks}
        stats_done = set()
        alive = {}   # (method, seed) -> True while the parity gate holds
        for seed in a.seeds:
            groups = {}
            for method in kplan['methods']:
                d = run_dir(a.runs, k, method, seed)
                sha = hashlib.sha256((d / 'preprocessing.json').read_bytes()).hexdigest()
                groups.setdefault(sha, []).append(method)
            for sha, methods in groups.items():
                models, margs, refs = {}, {}, {}
                prep_state = read_json(run_dir(a.runs, k, methods[0], seed) / 'preprocessing.json')
                prep = load_preprocessing(prep_state)
                roundtrip = preprocessing_state(prep) == prep_state
                for method in methods:
                    d = run_dir(a.runs, k, method, seed)
                    z = np.load(d / 'validation.npz', allow_pickle=True)
                    pos = {str(s): i for i, s in enumerate(z['sample_ids'])}
                    ix = np.array([pos[s] for s in ids])
                    if not np.array_equal(np.asarray(z['y'])[ix], y):
                        raise ValueError('stored validation labels differ from the scored rows')
                    refs[method] = np.asarray(z['proba'])[ix]
                    margs[method] = make_args(T, final_args, shim, a, method, k, seed)
                for mask in a.masks:
                    if mask != 'none' and not models:
                        break
                    data = []
                    for i, (g, vm) in enumerate(raw):
                        g2, vm2, st = apply_mask(g, vm, mask)
                        if mask not in stats_done:
                            ma = mask_acc[mask]
                            ma['changed'] += int(st['changed'])
                            ma['nodes'] += st['nodes_removed']
                            ma['edges'] += st['edges_removed']
                            ma['visits'] += st['visits_removed']
                            ma['acuity'] += int(st['acuity_removed'])
                            ma['nodes_before'] += len(g['nodes'])
                            ma['edges_before'] += len(g['edges'])
                        d0 = encode_graph(g2, prep, vm2, margs[methods[0]].edges, (), (), seed,
                                          edge_direction=margs[methods[0]].edge_direction)
                        d0.y = torch.tensor([int(y[i])], dtype=torch.long)
                        data.append(d0)
                    stats_done.add(mask)
                    if mask == 'none':
                        for method in methods:
                            sd = torch.load(run_dir(a.runs, k, method, seed) / 'best.pt', map_location='cpu')
                            torch.manual_seed(seed)
                            models[method] = build_model(T, M, method, margs[method], prep, data[0], k, sd)
                    batches = list(DataLoader(data, batch_size=margs[methods[0]].batch_size))
                    for method in list(models):
                        model = models[method]
                        is_adapter = isinstance(model, ClinicalMethodAdapter)
                        with torch.no_grad():
                            logits = [(model(b, epoch=0).logits if is_adapter else model(b)) for b in batches]
                        proba = torch.softmax(torch.cat(logits), 1).numpy()
                        if mask == 'none':
                            diff = float(np.abs(proba - refs[method]).max())
                            agree = float((proba.argmax(1) == refs[method].argmax(1)).mean())
                            ok = bool(roundtrip and diff <= PARITY_TOL and agree == 1.0)
                            msg = ('ok' if ok else
                                   f'parity failed: max_abs_diff {diff:.3g} (tolerance {PARITY_TOL}), '
                                   f'argmax agreement {agree:.4f}, preprocessing_roundtrip_equal {roundtrip}')
                            metr = T.metrics(y, proba, k)
                            ours = fast_metrics(y, proba.argmax(1), k)
                            metric_check.append(max(abs(ours[0] - metr['macro_f1']), abs(ours[1] - metr['accuracy'])))
                            parity_rows.append({'K': k, 'method': method, 'seed': seed, 'rows': len(ids),
                                                'fold': FOLD, 'threads': a.threads, 'max_abs_diff': diff,
                                                'argmax_agreement': agree, 'tolerance': PARITY_TOL,
                                                'preprocessing_roundtrip_equal': roundtrip,
                                                'passed': ok, 'message': msg})
                            print(f'K={k} {method} seed {seed}: parity max_abs_diff {diff:.3g}, '
                                  f'argmax agreement {agree:.4f} -> {"pass" if ok else "FAIL"}', flush=True)
                            if not ok:
                                del models[method]
                                continue
                            alive[(method, seed)] = True
                        preds[(method, seed, mask)] = proba.argmax(1)
                    del batches, data
                (out / 'parity.json').write_text(json.dumps(parity_rows, indent=1) + '\n')
        # ---- per-K tables
        pat = np.unique(rows['subjects'], return_inverse=True)[1]
        for mask, ma in mask_acc.items():
            g = ma['graphs']
            mask_stats.setdefault(str(k), {})[mask] = {
                'graphs': g, 'graphs_changed': ma['changed'],
                'mean_nodes_removed': ma['nodes'] / g, 'mean_edges_removed': ma['edges'] / g,
                'mean_visits_removed': ma['visits'] / g, 'graphs_acuity_removed': ma['acuity'],
                'mean_nodes_before': ma['nodes_before'] / g, 'mean_edges_before': ma['edges_before'] / g}
        kres = analyse(k, kplan, y, pat, preds, alive, a, csv_rows)
        summary['ks'][str(k)] = kres
    (out / 'parity.json').write_text(json.dumps(parity_rows, indent=1) + '\n')
    (out / 'mask_stats.json').write_text(json.dumps(mask_stats, indent=1, sort_keys=True) + '\n')
    cols = ['K', 'fold', 'method', 'seed', 'mask', 'rows', 'patients', 'macro_f1', 'accuracy',
            'macro_f1_drop', 'accuracy_drop']
    with open(out / 'masking.csv', 'w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=cols)
        w.writeheader()
        w.writerows(csv_rows)
    summary['parity'] = {'tolerance': PARITY_TOL,
                         'runs_checked': len(parity_rows), 'runs_passed': sum(r['passed'] for r in parity_rows),
                         'max_abs_diff_over_runs': max((r['max_abs_diff'] for r in parity_rows), default=None),
                         'metric_implementation_max_diff_vs_core_train': max(metric_check, default=None)}
    summary['mask_stats'] = mask_stats
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, sort_keys=True) + '\n')
    (out / 'masking_summary.md').write_text(render_md(plan, summary))
    print('written: plan.json parity.json masking.csv masking_summary.md mask_stats.json summary.json')


def analyse(k, kplan, y, pat, preds, alive, a, csv_rows):
    """Metrics, drops, seed means and shared patient bootstrap for one K."""
    n_pat = int(pat.max()) + 1
    rng = np.random.default_rng(SEED_RNG)
    counts = np.stack([np.bincount(rng.integers(0, n_pat, size=n_pat), minlength=n_pat)
                       for _ in range(BOOTSTRAP_B)]).astype(float)
    weights = counts[:, pat]
    methods = kplan['methods']
    seeds_ok = {m: [s for s in a.seeds if alive.get((m, s))] for m in methods}
    point, boot = {}, {}
    for m in methods:
        for s in seeds_ok[m]:
            for mask in a.masks:
                p = preds[(m, s, mask)]
                point[(m, s, mask)] = fast_metrics(y, p, k)
                boot[(m, s, mask)] = np.array([fast_metrics(y, p, k, w) for w in weights])
    for (m, s, mask), (f1, acc) in point.items():
        f1_0, acc_0 = point[(m, s, 'none')]
        csv_rows.append({'K': k, 'fold': FOLD, 'method': m, 'seed': s, 'mask': mask, 'rows': len(y),
                         'patients': n_pat, 'macro_f1': round(f1, 6), 'accuracy': round(acc, 6),
                         'macro_f1_drop': round(f1_0 - f1, 6), 'accuracy_drop': round(acc_0 - acc, 6)})
    res = {'task': kplan['task'], 'methods': methods, 'seeds_scored': {m: seeds_ok[m] for m in methods},
           'seed_mean': {}, 'cei_minus_baseline_drop': {}}
    for m in methods:
        ss = seeds_ok[m]
        if not ss:
            continue
        for mask in a.masks:
            f = np.mean([point[(m, s, mask)][0] for s in ss])
            c = np.mean([point[(m, s, mask)][1] for s in ss])
            f0 = np.mean([point[(m, s, 'none')][0] for s in ss])
            c0 = np.mean([point[(m, s, 'none')][1] for s in ss])
            res['seed_mean'].setdefault(m, {})[mask] = {
                'macro_f1': round(float(f), 6), 'accuracy': round(float(c), 6),
                'macro_f1_drop': round(float(f0 - f), 6), 'accuracy_drop': round(float(c0 - c), 6),
                'seeds': len(ss)}
    if 'cei_mp2' in methods:
        for base in methods:
            if base == 'cei_mp2':
                continue
            common = [s for s in seeds_ok['cei_mp2'] if s in seeds_ok[base]]
            if not common:
                continue
            for mask in a.masks:
                if mask == 'none':
                    continue
                entry = {'seeds': common}
                for idx, metric in ((0, 'macro_f1'), (1, 'accuracy')):
                    def drop(m, s, src):
                        d = lambda key: src[key][idx] if src is point else src[key][:, idx]
                        return d((m, s, 'none')) - d((m, s, mask))
                    est = np.mean([drop('cei_mp2', s, point) - drop(base, s, point) for s in common])
                    bt = np.mean([drop('cei_mp2', s, boot) - drop(base, s, boot) for s in common], axis=0)
                    lo, hi = np.percentile(bt, [2.5, 97.5])
                    entry[metric] = {'difference': round(float(est), 6), 'ci95': [round(float(lo), 6),
                                                                               round(float(hi), 6)]}
                res['cei_minus_baseline_drop'].setdefault(base, {})[mask] = entry
    return res


def render_md(plan, summary):
    L = ['# Source masking (fold: validation; test never touched)', '',
         f"Seeds {summary['seeds']}, limit {summary['limit']}, threads {summary['threads']}. "
         f"Drop = metric(none) - metric(mask). Difference = CEI drop - baseline drop (negative: CEI drops "
         f"less); 95% CI from a patient bootstrap (B={BOOTSTRAP_B}, rng {SEED_RNG}, shared draw, seed-mean "
         "statistic). Baselines chosen on validation macro-F1.", '', summary['note'], '']
    for k, r in summary['ks'].items():
        t = r['task']
        L += [f"## K={k} ({len(t['class_set'])} classes, {t['rows']} validation rows, {t['patients']} patients, "
              f"seeds {t['seeds']})", '',
              f"Classes: {', '.join(t['class_set'])}", '',
              '| method | mask | macro-F1 | accuracy | macro-F1 drop | accuracy drop | seeds |',
              '|---|---|---|---|---|---|---|']
        for m, per in r['seed_mean'].items():
            for mask, v in per.items():
                L.append(f"| {PRETTY[m]} | {mask} | {v['macro_f1']:.4f} | {v['accuracy']:.4f} | "
                         f"{v['macro_f1_drop']:+.4f} | {v['accuracy_drop']:+.4f} | {v['seeds']} |")
        L += ['', '| baseline | mask | macro-F1 drop difference (CEI - baseline) | 95% CI | '
                  'accuracy drop difference | 95% CI |', '|---|---|---|---|---|---|']
        for base, per in r['cei_minus_baseline_drop'].items():
            for mask, v in per.items():
                f, c = v['macro_f1'], v['accuracy']
                L.append(f"| {PRETTY[base]} | {mask} | {f['difference']:+.4f} | "
                         f"[{f['ci95'][0]:+.4f}, {f['ci95'][1]:+.4f}] | {c['difference']:+.4f} | "
                         f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] |")
        L.append('')
    p = summary['parity']
    L += [f"Parity gate (unmasked vs stored validation probabilities, tolerance {p['tolerance']}): "
          f"{p['runs_passed']}/{p['runs_checked']} runs passed, max abs diff {p['max_abs_diff_over_runs']}.", '']
    return '\n'.join(L)


# ------------------------------------------------------------------------------------------------ cli
def ints(text):
    return tuple(int(x) for x in text.split(',') if x.strip())


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m comparison.s7_why.masking',
                                 description=__doc__.split('\n\n')[0],
                                 epilog='Plan mode (default) prints the plan and loads no checkpoint.')
    ap.add_argument('--runs', required=True, help='runs root (top<K>/<method>_seed<S>/)')
    ap.add_argument('--artifact', required=True, help='graph artifact directory (graphs.jsonl, manifest.json)')
    ap.add_argument('--targets', required=True, help='targets.csv')
    ap.add_argument('--out', required=True, help='new output directory (must not exist; execute mode)')
    ap.add_argument('--ks', default='10,20,30')
    ap.add_argument('--seeds', default=','.join(map(str, ALL_SEEDS)))
    ap.add_argument('--masks', default=','.join(MASKS))
    ap.add_argument('--limit', type=int, default=None, help='deterministic patient-level subsample of rows')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--methods', default=None,
                    help='explicit method list (parity checks only, e.g. cei_mp2,gchm_pna); default: CEI + '
                         'two strongest validation baselines per K')
    ap.add_argument('--execute', action='store_true', help='load checkpoints and score (inference only)')
    a = ap.parse_args(argv)
    a.ks, a.seeds = ints(a.ks), ints(a.seeds)
    a.masks = tuple(x for x in a.masks.split(',') if x)
    if not set(a.masks) <= set(MASKS) or 'none' not in a.masks:
        ap.error(f'--masks must include none and be a subset of {list(MASKS)}')
    a.masks = ('none',) + tuple(m for m in MASKS if m in a.masks and m != 'none')
    if not set(a.seeds) <= set(ALL_SEEDS) or not a.ks:
        ap.error(f'--seeds must be a subset of {list(ALL_SEEDS)}')
    a.methods = tuple(x for x in a.methods.split(',') if x) if a.methods else ()
    a.pool = tuple(dict.fromkeys(('cei_mp2',) + BASELINES + a.methods))
    if a.limit is not None and a.limit < 1:
        ap.error('--limit must be positive')
    if 'standardized' in Path(a.out).resolve().parts:
        ap.error('--out must not be under comparison/standardized (inputs are read-only)')
    if a.execute and Path(a.out).exists():
        ap.error('--out exists; choose a new directory')
    plan = build_plan(a)
    print_plan(plan)
    if not a.execute:
        return 0
    execute(a, plan)
    return 0


if __name__ == '__main__':
    sys.exit(main())
