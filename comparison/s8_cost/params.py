"""Parameter counts per method x K from the stored run bindings; optional rebuild check of the seed-1234 models.

Usage (from the repo root):
    python3 -m comparison.s8_cost.params --runs comparison/standardized/clinical_runs_fair_v10t_20261011 --out NEW_DIR
    python3 -m comparison.s8_cost.params --runs ... --out NEW_DIR --execute      # also rebuilds + loads best.pt (no training)

Without --execute only binding.json is read: parameter_count, active_parameter_count, batch_size, hidden/layers, checked
equal over all seeds found. With --execute the seed-1234 model of each (K, method) is rebuilt with the repo's own
construction code (core.train.parser/normalize_method_args, core.registry.build_method), best.pt is loaded with
strict=True, the trainable parameters are recounted and split into explanation-specific groups. A recount that differs
from binding.json (or a strict-load failure) exits non-zero. --out must not exist; outputs are aggregated only.
Explanation-specific parameters: GSAT extractor; ProtGNN prototype vectors + prototype classifier (last layer);
CEI-GNN none (its evidence heads are the prediction path); GraphCare none (no built-in explanation).
"""
import argparse, csv, json, os, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_METHODS = 'cei_mp2,gsat,graphcare,protgnn'
DEFAULT_KS = '10,20,30'
EXPLANATION_PREFIXES = {   # state-dict name prefixes of explanation-specific trainable parameters
    'gsat': ('extractor.',),
    'protgnn': ('prototype_vectors', 'prototype_classifier.'),
    'cei_mp2': (),
    'graphcare': (),
    'gchm_pna': (),
}
EXPLANATION_NOTE = {
    'gsat': 'edge-attention extractor MLP',
    'protgnn': 'prototype vectors + prototype classifier (last layer)',
    'cei_mp2': 'none: evidence heads are the prediction path',
    'graphcare': 'none: no built-in explanation',
    'gchm_pna': 'none: no built-in explanation',
}


def rel(path):
    """Repo-relative text for a path (never an absolute path)."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        try:
            return p.relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return p.name


def run_dirs(runs, K, method):
    base = Path(runs) / f'top{K}'
    return sorted((d for d in base.glob(f'{method}_seed*') if re.fullmatch(rf'{method}_seed\d+', d.name)),
                  key=lambda d: int(d.name.rsplit('seed', 1)[1]))


def read_bindings(runs, ks, methods):
    recs = []
    for K in ks:
        for m in methods:
            dirs = run_dirs(runs, K, m)
            if not dirs:
                raise SystemExit(f'no runs for top{K}/{m} under {rel(runs)}')
            bs = {int(d.name.rsplit('seed', 1)[1]): json.load(open(d / 'binding.json')) for d in dirs}
            pc = {s: b['parameter_count'] for s, b in bs.items()}
            ac = {s: b['active_parameter_count'] for s, b in bs.items()}
            first = bs[min(bs)]
            arch = (first.get('method_config') or {}).get('architecture') or {}
            recs.append(dict(
                K=K, method=m, seeds_checked=sorted(bs), parameter_count=pc[min(pc)], active_parameter_count=ac[min(ac)],
                seeds_equal=len(set(pc.values())) == 1 and len(set(ac.values())) == 1,
                batch_size=first.get('batch_size'), hidden=first.get('hidden'), layers=first.get('layers'),
                node_dim=first.get('node_dim', arch.get('node_dim')), edge_dim=first.get('edge_dim'),
                num_classes=first.get('top_k_labels'), train_rows=first['counts']['train'],
                validation_rows=first['counts']['validation'], test_rows=first['counts']['test'],
                explanation_params=None, explanation_note=EXPLANATION_NOTE.get(m), rebuilt_trainable=None,
                rebuild_matches_binding=None, strict_load=None))
    return recs


def rebuild(runs, rec, date):
    """Rebuild the seed-1234 model of one (K, method), load best.pt strictly, return (trainable, explanation, buffers)."""
    import torch
    torch.set_num_threads(2)
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from core import train as T
    from core.model import ClinicalGNN
    from core.registry import build_method
    from core.tensorize import PAYLOAD_WIDTH, load_preprocessing, triple_count
    from comparison.final.run import Paths, args as final_args

    K, m = rec['K'], rec['method']
    run_dir = Path(runs) / f'top{K}' / f'{m}_seed1234'
    P = Paths(str(REPO), date)
    cli = final_args(P, m, K) + ['--top-k-labels', str(K), '--artifact', str(P.graph), '--targets', str(P.targets),
                                 '--output', 'unused_rebuild_output', '--seed', '1234', '--cpu-threads', '2']
    p = T.parser()
    args = T.normalize_method_args(p.parse_args(cli), p)
    prep = load_preprocessing(json.load(open(run_dir / 'preprocessing.json')))
    sd = torch.load(run_dir / 'best.pt', map_location='cpu')
    node_dim, edge_dim = rec['node_dim'], rec['edge_dim']
    num_triples = triple_count(prep, args.edge_direction)
    if m == 'gchm_pna':
        model = ClinicalGNN(num_classes=K, num_tokens=len(prep['vocabulary']), node_dim=node_dim, edge_dim=edge_dim,
                            hidden=args.hidden, layers=args.layers, dropout=args.dropout, token_dim=args.token_dim,
                            use_edge_payload=not args.no_edge_payload, use_message_passing=not args.no_message_passing,
                            conv=args.conv, num_triples=num_triples, payload_dim=PAYLOAD_WIDTH, heads=args.heads,
                            degree_histogram=sd['degree_histogram'], modulation=args.modulation)
    else:
        model = build_method(args.method, num_tokens=len(prep['vocabulary']), node_dim=node_dim, edge_dim=edge_dim,
                             num_classes=K, hidden=args.hidden, layers=args.layers, dropout=args.dropout,
                             token_dim=args.token_dim, num_triples=num_triples, args=args)
        hook = getattr(model, 'verify_run_preprocessing', None)
        if callable(hook):
            hook(T.preprocessing_state(prep))
    model.load_state_dict(sd, strict=True)
    trainable = sum(q.numel() for q in model.parameters() if q.requires_grad)
    prefixes = EXPLANATION_PREFIXES.get(m, ())
    expl = sum(q.numel() for n, q in model.named_parameters() if q.requires_grad and n.startswith(prefixes)) if prefixes else 0
    buffers = sum(b.numel() for b in model.buffers())
    return trainable, expl, buffers


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', required=True, help='final runs root, e.g. comparison/standardized/clinical_runs_fair_v10t_20261011')
    ap.add_argument('--out', required=True, help='NEW output directory (an existing one is refused)')
    ap.add_argument('--methods', default=DEFAULT_METHODS,
                    help=f'comma-separated (default {DEFAULT_METHODS}); gchm_pna only on explicit request (parity checks)')
    ap.add_argument('--ks', default=DEFAULT_KS)
    ap.add_argument('--execute', action='store_true',
                    help='rebuild the seed-1234 models, load best.pt strictly, recount (no training); exit 1 on any mismatch')
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'refusing existing --out {rel(out)}')
    ks = [int(k) for k in a.ks.split(',')]
    methods = [m for m in a.methods.split(',') if m]
    unknown = [m for m in methods if m not in EXPLANATION_PREFIXES]
    if unknown:
        raise SystemExit(f'unknown methods {unknown}')
    m_date = re.search(r'(\d{8})$', Path(a.runs).resolve().name)
    date = m_date.group(1) if m_date else '20261011'

    recs = read_bindings(a.runs, ks, methods)
    failures = [f"top{r['K']}/{r['method']}: parameter/active counts differ across seeds {r['seeds_checked']}"
                for r in recs if not r['seeds_equal']]
    if a.execute:
        for r in recs:
            try:
                trainable, expl, buffers = rebuild(a.runs, r, date)
            except Exception as e:   # strict-load or construction failure counts as a mismatch
                r['strict_load'] = False
                r['rebuild_matches_binding'] = False
                failures.append(f"top{r['K']}/{r['method']}: rebuild/strict load failed: {type(e).__name__}: {str(e)[:200]}")
                continue
            r.update(strict_load=True, rebuilt_trainable=trainable, explanation_params=expl, buffer_elements=buffers,
                     rebuild_matches_binding=trainable == r['parameter_count'])
            if trainable != r['parameter_count']:
                failures.append(f"top{r['K']}/{r['method']}: rebuilt {trainable} != binding {r['parameter_count']}")

    out.mkdir(parents=True)
    summary = dict(
        task='parameter counts per method x K from stored final-run bindings' + (' + strict rebuild of seed-1234 models' if a.execute else ''),
        runs=rel(a.runs), methods=methods, ks=ks, execute=a.execute,
        task_identity=dict(seeds=sorted({s for r in recs for s in r['seeds_checked']}), epochs=40,
                           rows={f"K{r['K']}": dict(train=r['train_rows'], validation=r['validation_rows'], test=r['test_rows'])
                                 for r in recs if r['method'] == methods[0]}),
        explanation_definition=EXPLANATION_NOTE, records=recs, failures=failures)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    cols = ['K', 'method', 'parameter_count', 'active_parameter_count', 'explanation_params', 'batch_size', 'hidden', 'layers',
            'seeds_checked', 'seeds_equal', 'rebuilt_trainable', 'rebuild_matches_binding']
    with open(out / 'params.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in recs:
            w.writerow([' '.join(map(str, r[c])) if c == 'seeds_checked' else r[c] for c in cols])
    print('K  method     total   active  expl    batch  seeds        rebuilt')
    for r in recs:
        print(f"{r['K']:<2} {r['method']:<10} {r['parameter_count']:<7} {r['active_parameter_count']:<7} "
              f"{str(r['explanation_params']):<7} {r['batch_size']:<6} {','.join(map(str, r['seeds_checked'])):<12} "
              f"{r['rebuilt_trainable'] if a.execute else '-'}")
    print('wrote', rel(out))
    if failures:
        print('FAILURES:\n' + '\n'.join(failures))
        sys.exit(1)


if __name__ == '__main__':
    main()
