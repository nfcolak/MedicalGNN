"""Inference cost of the stored final models (forward pass, tensorization, collation, memory, explanation extraction).

Reads the stored best.pt of the final fair_v10t runs and scores the same random VALIDATION graphs with every method
and K; no optimizer, no backward, no training. Each (method, K) runs in its OWN subprocess so peak memory is not
shared between models. Predictions are checked against the run's validation.npz (softmax max abs diff must stay
<= 1e-4; otherwise the command exits non-zero after writing its outputs).

Usage (from the repo root):
    python3 -m comparison.s8_cost.bench                          plan only: prints one line per (K, method), loads nothing
    python3 -m comparison.s8_cost.bench --execute --out DIR      measures (reads checkpoints and graphs; needs approval)

Graphs: --graphs ids drawn with random.Random(--sample-seed) from the K=10 validation fold of the seed-1234 gsat run
(the K=10 validation fold is a subset of the K=20 and K=30 folds). The graph lines are copied to a private temporary
directory for the run and deleted at the end; the output holds aggregates only (no sample or subject ids).
Timings: one warm-up pass, then --rounds rounds of --passes timed passes; reported per graph as median/min/max over
rounds. Load average is recorded before and after each subprocess; above 2 the record is flagged contended.
The timings and memory numbers do not depend on the fold: their inputs are 500 validation graphs (--graphs default).
"""
import argparse
import json
import os
import random
import re
import resource
import shutil
import statistics as st
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = 'comparison/standardized/clinical_runs_fair_v10t_20261011'
DEFAULT_ARTIFACT = 'comparison/standardized/event_inputs/clinical_graph_L4_plus60_v5c_20261009'
DEFAULT_METHODS = 'cei_mp2,gsat,graphcare,protgnn'
ALL_METHODS = ('cei_mp2', 'gsat', 'graphcare', 'protgnn', 'gchm_pna')
DEFAULT_KS = '10,20,30'
PARITY_TOLERANCE = 1e-4
LOAD_LIMIT = 2.0
RUN_DATE = '20261011'
PROTOCOL = 'fair_v10t'
REFERENCE_IDS_RUN = 'top10/gsat_seed1234'
_SAMPLE_ID = re.compile(r'"sample_id"\s*:\s*("(?:[^"\\]|\\.)*")')


# ------------------------------------------------------------------------------------------------ shared helpers
def rel(path):
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return p.name


def scrub(text):
    """Remove local absolute paths from a message before it is printed or stored."""
    text = text.replace(str(REPO), '.')
    return re.sub(r'/(?:Users|home|private|var)/[^\s\'",)]+', '<path>', text)


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024   # macOS reports bytes


def spread(values, scale=1.0):
    return {'median': scale * st.median(values), 'min': scale * min(values), 'max': scale * max(values),
            'n_rounds': len(values)}


def parse_list(text, cast=str):
    return [cast(x) for x in text.split(',') if x.strip()]


# ------------------------------------------------------------------------------------------------ worker
def worker(a):
    """One method x K in this process. Prints a single 'RESULT {json}' line (aggregates only)."""
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    from torch_geometric.loader import DataLoader
    from core import train as T
    from core.contracts import iter_graphs_with_membership
    from core.model import ClinicalGNN
    from core.registry import ClinicalMethodAdapter, build_method
    from core.tensorize import PAYLOAD_WIDTH, encode_graph, load_preprocessing, triple_count
    from comparison.final.run import Paths, args as final_args

    method, K, seed, bs = a.worker_method, a.worker_k, a.checkpoint_seed, a.batch_size
    runs, artifact = REPO / a.runs, REPO / a.artifact
    run_dir = runs / f'top{K}' / f'{method}_seed{seed}'
    P = Paths(REPO, RUN_DATE)
    P.graph, P.events, P.runs = artifact, artifact.parent, runs
    cli = final_args(P, method, K) + ['--top-k-labels', str(K), '--artifact', str(artifact), '--targets', str(P.targets),
                                      '--output', '/nonexistent/bench', '--seed', str(seed),
                                      '--cpu-threads', str(a.threads)]
    parser = T.parser()
    args = T.normalize_method_args(parser.parse_args(cli), parser)
    binding = json.loads((run_dir / 'binding.json').read_text())
    r_base = rss_mb()

    prep = load_preprocessing(json.loads((run_dir / 'preprocessing.json').read_text()))
    val = np.load(run_dir / 'validation.npz', allow_pickle=True)
    pos = {str(s): i for i, s in enumerate(val['sample_ids'])}
    parsed, rows_idx = [], []
    for graph, vm in iter_graphs_with_membership(Path(a.data_dir) / 'graphs.jsonl',
                                                 Path(a.data_dir) / 'visit_membership.jsonl'):
        parsed.append((graph, vm))
        rows_idx.append(pos[graph['sample_id']])
    n = len(parsed)

    def tensorize_all():
        t0 = time.perf_counter()
        out = [encode_graph(g, prep, vm, args.edges, (), (), seed, edge_direction=args.edge_direction)
               for g, vm in parsed]
        return out, time.perf_counter() - t0

    data, t_first = tensorize_all()
    for d, i in zip(data, rows_idx):
        d.y = torch.tensor([int(val['y'][i])], dtype=torch.long)
    node_dim, edge_dim = data[0].x.size(1), data[0].edge_attr.size(1)
    num_triples = triple_count(prep, args.edge_direction)
    sd = torch.load(run_dir / 'best.pt', map_location='cpu')
    if method == 'gchm_pna':
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
    model.eval()
    is_adapter = isinstance(model, ClinicalMethodAdapter)
    r_after_load = rss_mb()

    def collate():
        t0 = time.perf_counter()
        out = list(DataLoader(data, batch_size=bs))
        return out, time.perf_counter() - t0

    batches, t_coll_first = collate()

    def fwd(b):
        return model(b, epoch=0).logits if is_adapter else model(b)

    @torch.no_grad()
    def one_pass():
        outs = []
        t0 = time.perf_counter()
        for b in batches:
            outs.append(fwd(b))
        dt = time.perf_counter() - t0
        return dt, torch.softmax(torch.cat(outs), 1).numpy()

    _, proba = one_pass()                       # warm-up; also the parity pass
    ref = val['proba'][rows_idx]
    parity = float(np.abs(proba - ref).max())
    agree = float((proba.argmax(1) == ref.argmax(1)).mean())

    fwd_r, tens_r, coll_r = [], [], []
    for r in range(a.rounds):
        if r == 0:
            tens_r.append(t_first)
            coll_r.append(t_coll_first)
        else:
            _, dt = tensorize_all()
            tens_r.append(dt)
            _, dt = collate()
            coll_r.append(dt)
        fwd_r.append(sum(one_pass()[0] for _ in range(a.passes)) / a.passes)
    r_peak_inf = rss_mb()

    out = dict(method=method, K=K, graphs=n, batch_size=bs, threads=torch.get_num_threads(),
               checkpoint_seed=seed, rounds=a.rounds, passes=a.passes,
               parameter_count=binding['parameter_count'], active_parameter_count=binding['active_parameter_count'],
               params_trainable_counted=sum(q.numel() for q in model.parameters() if q.requires_grad),
               training_batch_size=binding.get('batch_size'),
               parity_max_abs_diff=parity, argmax_agreement=agree, parity_pass=bool(parity <= PARITY_TOLERANCE),
               forward_ms_per_graph=spread(fwd_r, 1000 / n),
               tensorize_ms_per_graph=spread(tens_r, 1000 / n),
               collate_ms_per_graph=spread(coll_r, 1000 / n),
               rss_mb_base=round(r_base, 1), rss_mb_after_load=round(r_after_load, 1),
               rss_mb_peak_inference=round(r_peak_inf, 1),
               nodes_mean=float(np.mean([d.num_nodes for d in data])),
               edges_mean=float(np.mean([d.edge_index.size(1) for d in data])))

    # ------------------------------------------------------------ explanation overhead (no_grad, same graphs)
    reps = max(a.passes, 1)

    def timed(fn):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return 1000 * st.median(ts) / n

    expl = {'available': False, 'overhead_ms_per_graph': None, 'what': 'no built-in explanation head (n/a)'}
    with torch.no_grad():
        if method == 'gsat':
            def plain():
                for b in batches:
                    model(b, epoch=0)

            def with_expl():
                for b in batches:
                    model(b, epoch=0)
                    ea = model.last_edge_attention
                    g = b.batch[b.edge_index[0]]
                    for gi in range(b.num_graphs):      # per-graph top-5 attended edges
                        v = ea[g == gi]
                        if v.numel():
                            v.topk(min(5, v.numel()))
            plain()
            p_ms, e_ms = timed(plain), timed(with_expl)
            expl = dict(available=True, overhead_ms_per_graph=e_ms - p_ms, plain_forward_ms_per_graph=p_ms,
                        with_extraction_ms_per_graph=e_ms,
                        what='edge attention is produced inside every forward (last_edge_attention); extraction = per-graph top-5 edges; overhead = with extraction - plain forward')
        elif method == 'protgnn':
            def plain():
                for b in batches:
                    model(b, epoch=0)

            def sims():
                for b in batches:
                    _, ge, _ = model._node_and_graph_embeddings(b)
                    act, _ = model._prototype_activations(ge)
                    act.topk(3, dim=1)
            plain()
            p_ms, s_ms = timed(plain), timed(sims)
            expl = dict(available=True, overhead_ms_per_graph=s_ms - p_ms, plain_forward_ms_per_graph=p_ms,
                        with_extraction_ms_per_graph=s_ms,
                        what='top-3 prototype similarities (embedding + prototype activations + top-3); overhead = that pass - plain forward (the similarities are computed inside the forward as well)')
            if a.protgnn_mcts > 0:
                sel = batches[0]
                node_state, ge, gi_ = model._node_and_graph_embeddings(sel)
                act, _ = model._prototype_activations(ge)
                t0 = time.perf_counter()
                done = 0
                for g in range(min(a.protgnn_mcts, sel.num_graphs)):
                    cand = model._projection_candidate(sel, g, node_state, gi_)
                    if cand is None:
                        continue
                    model._mcts_project(cand, model.prototype_vectors[int(act[g].argmax())])
                    done += 1
                expl['mcts_node_explanation_ms_per_graph'] = 1000 * (time.perf_counter() - t0) / max(done, 1)
                expl['mcts_graphs'] = done
        elif method == 'cei_mp2':
            net = model.network
            pre = []
            for b in batches:
                clinical, membership, vg = model._read(b)
                pre.append((clinical, membership, vg, net.continuous_inputs(clinical)))

            def f_plain():
                for clinical, membership, vg, feats in pre:
                    net.forward_continuous(feats, clinical.edge_index, clinical, membership,
                                           return_parts=False, visit_graph=vg)

            def f_parts():
                for clinical, membership, vg, feats in pre:
                    net.forward_continuous(feats, clinical.edge_index, clinical, membership,
                                           return_parts=True, visit_graph=vg)

            def extract():
                for clinical, membership, vg, feats in pre:
                    parts = net.forward_continuous(feats, clinical.edge_index, clinical, membership,
                                                   return_parts=True, visit_graph=vg)
                    ng, C = clinical.graph_count, K
                    bi, ei = clinical.batch_index, clinical.edge_index
                    pred = parts['logits'].argmax(1)

                    def own(values, owner):
                        tot = torch.zeros((ng, C))
                        if values.numel():
                            tot.index_add_(0, owner.long(), values)
                        return tot
                    own(parts['node_contributions'], bi)
                    own(parts['edge_contributions'], bi[ei[0]] if ei.numel() else ei.new_zeros(0))
                    own(parts['pair_contributions'], bi[parts['pairs'][0]] if parts['pairs'].numel() else bi.new_zeros(0))
                    if parts['absence_items'].numel():
                        own(parts['absence_contributions'], parts['absence_items'][0])
                    if 'nbr_contributions' in parts:
                        own(parts['nbr_contributions'], bi)
                    for gi in range(ng):                # per-graph top-5 items of the predicted class
                        c = int(pred[gi])
                        vals = torch.cat([parts['node_contributions'][bi == gi, c],
                                          parts['edge_contributions'][(bi[ei[0]] == gi), c]])
                        if vals.numel():
                            vals.topk(min(5, vals.numel()))
            f_plain()
            extract()
            lg, pt, ex = timed(f_plain), timed(f_parts), timed(extract)
            expl = dict(available=True, overhead_ms_per_graph=ex - lg, logits_only_ms_per_graph=lg,
                        return_parts_ms_per_graph=pt, ledger_extraction_ms_per_graph=ex,
                        overhead_vs_return_parts_ms_per_graph=ex - pt,
                        what='network.forward_continuous: logits only vs return_parts=True vs parts + per-graph block sums and top-5 items; overhead = extraction - logits only (inputs pre-read; the adapter forward itself already returns the parts)')
    out['explanation'] = expl
    out['rss_mb_peak'] = round(rss_mb(), 1)
    print('RESULT ' + json.dumps(out))


# ------------------------------------------------------------------------------------------------ orchestration
def plan_lines(a):
    lines = []
    for K in parse_list(a.ks, int):
        for m in parse_list(a.methods):
            run = f'{a.runs}/top{K}/{m}_seed{a.checkpoint_seed}'
            lines.append(f'K={K} {m.ljust(9)} fold=validation graphs={a.graphs} seed={a.checkpoint_seed} '
                         f'threads={a.threads} batch={a.batch_size} rounds={a.rounds} passes={a.passes} run={run}')
    return lines


def extract_graphs(a, dest):
    """Copy the chosen validation graphs' lines (patient data: private temp dir only) and return their count."""
    import numpy as np
    runs, artifact = REPO / a.runs, REPO / a.artifact
    ids = [str(x) for x in np.load(runs / REPO_IDS(a) / 'validation.npz', allow_pickle=True)['sample_ids']]
    pick = set(random.Random(a.sample_seed).sample(ids, a.graphs))
    n = 0
    with open(artifact / 'graphs.jsonl') as g, open(artifact / 'visit_membership.jsonl') as m, \
            open(dest / 'graphs.jsonl', 'w') as go, open(dest / 'visit_membership.jsonl', 'w') as mo:
        for gl, ml in zip(g, m):
            mt = _SAMPLE_ID.findall(gl)
            if len(mt) == 1 and json.loads(mt[0]) in pick:
                go.write(gl)
                mo.write(ml)
                n += 1
    if n != a.graphs:
        raise SystemExit(f'expected {a.graphs} graphs, extracted {n}')
    return n


def REPO_IDS(a):
    return Path(REFERENCE_IDS_RUN.replace('seed1234', f'seed{a.checkpoint_seed}'))


def run_worker(a, method, K, data_dir):
    cmd = [sys.executable, '-m', 'comparison.s8_cost.bench', '--worker-method', method, '--worker-k', str(K),
           '--data-dir', str(data_dir), '--runs', a.runs, '--artifact', a.artifact, '--graphs', str(a.graphs),
           '--rounds', str(a.rounds), '--passes', str(a.passes), '--threads', str(a.threads),
           '--batch-size', str(a.batch_size), '--checkpoint-seed', str(a.checkpoint_seed),
           '--protgnn-mcts', str(a.protgnn_mcts)]
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    load_before = os.getloadavg()[0]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    wall = time.time() - t0
    load_after = os.getloadavg()[0]
    rec = {'method': method, 'K': K, 'fold': 'validation', 'loadavg_1min_before': round(load_before, 2),
           'loadavg_1min_after': round(load_after, 2), 'subprocess_wall_seconds': round(wall, 1),
           'contended': bool(max(load_before, load_after) > LOAD_LIMIT)}
    line = next((x for x in reversed(proc.stdout.splitlines()) if x.startswith('RESULT ')), None)
    if proc.returncode != 0 or line is None:
        rec['error'] = scrub((proc.stderr or proc.stdout).strip()[-600:])
        return rec
    rec.update(json.loads(line[7:]))
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default=DEFAULT_RUNS, help='root of the final runs (repo-relative)')
    ap.add_argument('--artifact', default=DEFAULT_ARTIFACT, help='graph artifact directory (repo-relative)')
    ap.add_argument('--out', help='new output directory (must not exist); required with --execute')
    ap.add_argument('--methods', default=DEFAULT_METHODS, help='comma list; gchm_pna only for parity checks')
    ap.add_argument('--ks', default=DEFAULT_KS)
    ap.add_argument('--graphs', type=int, default=500)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--passes', type=int, default=3)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--checkpoint-seed', type=int, default=1234)
    ap.add_argument('--sample-seed', type=int, default=0)
    ap.add_argument('--protgnn-mcts', type=int, default=0, help='also time MCTS node explanations for N graphs (0 = off)')
    ap.add_argument('--execute', action='store_true', help='measure; without it only the plan is printed')
    ap.add_argument('--worker-method', help=argparse.SUPPRESS)
    ap.add_argument('--worker-k', type=int, help=argparse.SUPPRESS)
    ap.add_argument('--data-dir', help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.worker_method:
        return worker(a)
    methods, ks = parse_list(a.methods), parse_list(a.ks, int)
    bad = [m for m in methods if m not in ALL_METHODS]
    if bad or min(a.graphs, a.rounds, a.passes, a.threads, a.batch_size) < 1:
        ap.error(f'unknown methods {bad} or a count below 1')
    if 'gchm_pna' in methods:
        print('note: gchm_pna is excluded from new analyses; it runs here only for a parity check')
    print(f'bench plan: protocol {PROTOCOL}, fold validation, {a.graphs} graphs (sample seed {a.sample_seed}, '
          f'drawn from {REPO_IDS(a)} validation), checkpoint seed {a.checkpoint_seed}')
    for line in plan_lines(a):
        print(line)
    if not a.execute:
        print('plan only (no --execute): no checkpoint or graph was loaded')
        return 0
    if not a.out:
        ap.error('--execute needs --out')
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'refusing an existing --out: {rel(out)}')
    for p in (REPO / a.runs, REPO / a.artifact):
        if not p.is_dir():
            raise SystemExit(f'missing input directory: {rel(p)}')
    out.mkdir(parents=True, mode=0o700)

    tmp = Path(tempfile.mkdtemp(prefix='s8bench_'))
    records, load_start = [], os.getloadavg()[0]
    try:
        t0 = time.time()
        extract_graphs(a, tmp)
        print(f'extracted {a.graphs} validation graphs in {time.time() - t0:.0f}s')
        for K in ks:
            for m in methods:
                rec = run_worker(a, m, K, tmp)
                records.append(rec)
                if 'error' in rec:
                    print(f'K={K} {m}: FAILED {rec["error"][-200:]}')
                    continue
                flag = ' CONTENDED' if rec['contended'] else ''
                print(f'K={K} {m.ljust(9)} parity {rec["parity_max_abs_diff"]:.2e} '
                      f'agree {rec["argmax_agreement"]:.3f} {"PASS" if rec["parity_pass"] else "FAIL"} '
                      f'fwd {rec["forward_ms_per_graph"]["median"]:.3f} ms/graph '
                      f'tensorize {rec["tensorize_ms_per_graph"]["median"]:.3f} '
                      f'collate {rec["collate_ms_per_graph"]["median"]:.3f} rss peak {rec["rss_mb_peak"]} MB '
                      f'load {rec["loadavg_1min_before"]}->{rec["loadavg_1min_after"]}{flag}')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    contended = [r for r in records if r.get('contended')]
    if contended:
        print(f'WARNING: load average above {LOAD_LIMIT:g} during {len(contended)} of {len(records)} measurements '
              '(flagged contended); timings are not clean')
    failed = [r for r in records if 'error' in r or not r.get('parity_pass', False)]
    summary = {
        'task': {'protocol': PROTOCOL, 'fold': 'validation', 'ks': ks, 'methods': methods,
                 'graphs': a.graphs, 'sample_seed': a.sample_seed,
                 'graphs_drawn_from': f'{REPO_IDS(a)} validation fold (K=10, a subset of the K=20 and K=30 folds)',
                 'checkpoint_seed': a.checkpoint_seed, 'threads': a.threads, 'batch_size': a.batch_size,
                 'rounds': a.rounds, 'passes': a.passes, 'warmup_passes': 1,
                 'runs': rel(REPO / a.runs), 'artifact': rel(REPO / a.artifact),
                 'parity_tolerance_max_abs_diff': PARITY_TOLERANCE, 'load_limit_1min': LOAD_LIMIT,
                 'rss_unit': 'MB (ru_maxrss, bytes on macOS / 1048576)', 'test_fold': 'not used'},
        'machine_loadavg_1min_at_start': round(load_start, 2),
        'any_contended': bool(contended), 'all_parity_pass': not failed,
        'records': records}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(f'wrote {rel(out)}/summary.json')
    if failed:
        print(f'FAIL: {len(failed)} record(s) failed parity (> {PARITY_TOLERANCE:g}) or errored')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
