"""Ledger probes of the final CEI-GNN mp2 runs: block shares, node-removal gap, pair second difference.

Inference only. The stored final runs (best.pt) are rebuilt exactly as core/train.py builds them and score the first
N graphs of one fold (`--fold`, default test, the reporting fold) in forward passes: no training, no optimizer step,
no file of the inputs is written, nothing but aggregates is stored (no sample, subject or stay ids). Decisions
(checkpoint) use VALIDATION; every reported result is on the scored fold (test for the reported tables;
`--fold validation` reproduces the earlier validation tables).

    python -m comparison.s3_ledger.probe {blocks,removal,pairsecond} --out <new dir> [--fold test|validation]
        [--ks 10,20,30] [--seeds 1234,2025,7] [--runs-spec 10:7,20:2025,30:1234] [--n-graphs N] [--execute]
        [--runs <runs root>] [--artifact <graph dir>] [--targets <targets.csv>] [--threads 2]

Without --execute the plan is printed (runs, fold rows, graphs per run) and nothing is loaded or written. With
--execute --out must not exist. `--runs-spec` lists explicit K:seed runs and replaces the --ks x --seeds product.
Defaults point at the repository copies under comparison/standardized/ (final runs, graph, targets, CEI state).

Modes (n-graphs default: 300 for blocks and for the pairsecond selection, 200 for removal; pairsecond scores the first 40
  stored rows of its selection and 4000 sampled pairs)
  blocks      per run: item-level absolute-mass share of the five ledger blocks (node, edge, pair, nbr, absence;
              bias excluded), the block-total variant, the signed block contribution to the predicted-class
              logit and the exactness of the ledger (max |logit - bias - sum of parts|).
              Files: block_shares.csv (primary), block_shares_blocktotal.csv, block_shares_extra.csv.
  removal     per run: node-block removal gap (true removal effect against the ledger-naive removal, relative
              to the ledger effect) and the nbr mass share on the same graphs.
              File: removal_probe_K<K>_s<seed>.json.
  pairsecond  per run: second difference f(zi+zj) - f(zi) - f(zj) + f(0) of the pair vote on sampled pairs
              (zero iff the vote is endpoint-additive). File: pairsecond_K<K>_s<seed>.json.

Graph selection: the first N sample ids of the scored fold after sorting the id strings (identical across seeds of
a K; independent of the stored row order). Parity gate: the reloaded softmax of every probed graph must equal the
stored `<fold>.npz` row within 1e-4 with 100% argmax agreement, and the ledger must reproduce the logits within
1e-4, otherwise the run stops and nothing is written for it.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
STD = REPO / 'comparison/standardized'
FOLDS = ('test', 'validation')
ALL_SEEDS = (1234, 2025, 7)
MODES = ('blocks', 'removal', 'pairsecond')
BLOCKS = ('node', 'edge', 'pair', 'nbr', 'absence')
PARITY_TOL = 1e-4
LEDGER_TOL = 1e-4
DEFAULT_N = {'blocks': 300, 'removal': 200, 'pairsecond': 300}
PAIRS_GRAPHS = 40   # pairsecond scores the first 40 stored rows among the selected graphs
PAIRS_SAMPLED = 4000
DEFAULTS = {
    'runs': STD / 'clinical_runs_fair_v10t_20261011',
    'artifact': STD / 'event_inputs/clinical_graph_L4_plus60_v5c_20261009',
    'targets': STD / 'event_inputs/targets_primary_v4_v5plus60_v5_test_20261011/targets.csv',
}
V3_STATE = 'event_inputs/v3_state_L4_v5c_v10t_top{K}_20261011/K4.json'


def shown(path):
    """Repo-relative text for a path; outside the repo only the final name (never an absolute path)."""
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return p.name


def ints(text):
    return tuple(int(x) for x in text.split(',') if x)


def run_list(a):
    if a.runs_spec:
        return tuple(tuple(int(v) for v in item.split(':')) for item in a.runs_spec.split(',') if item)
    return tuple((k, s) for k in a.ks for s in a.seeds)


def run_dir(a, k, seed):
    return Path(a.runs) / f'top{k}' / f'cei_mp2_seed{seed}'


def pick_graphs(ids, n):
    """First n sample ids of the fold after sorting the id strings -> {sample_id: stored row index}."""
    chosen = sorted(range(len(ids)), key=lambda i: ids[i])[:n]
    return chosen, {ids[i]: i for i in chosen}


def print_plan(a, runs):
    n = a.n_graphs
    print(f'plan: mode {a.mode}, fold {a.fold}, graphs per run {n}, {len(runs)} runs')
    print(f'  runs   {shown(a.runs)}  (cei_mp2, best.pt, {a.fold}.npz)')
    print(f'  graph  {shown(a.artifact)}')
    print(f'  CEI state per K from {shown(STD)}/' + V3_STATE)
    for k, s in runs:
        d = run_dir(a, k, s)
        z = np.load(d / f'{a.fold}.npz', allow_pickle=True)
        print(f'  K={k:<2} seed={s:<4} {a.fold} rows {len(z["sample_ids"]):<5} probed graphs {min(n, len(z["sample_ids"]))}')
    print('  forward passes only; --execute loads the checkpoints and writes --out (must not exist)')


def load_run(a, k, seed, modules):
    """Rebuild one final run (model + preprocessing) and encode its first N fold graphs."""
    torch, np_, T, contracts, tensorize, registry = modules
    run = run_dir(a, k, seed)
    graph_dir = Path(a.artifact)
    state = STD / V3_STATE.format(K=k)
    argv = ('--method cei_gnn_mp2 --method-option arm=C --method-option k=4 '
            f'--method-option v3_state={state} --method-option mp_feed=dual --method-option mp_hub_max=1 '
            '--method-option mp_dedup=1 --method-option mp_ctx=table --method-option mp_root=bias '
            '--method-option mp_optimizer=adamw --method-option mp_lr_schedule=cosine '
            '--method-option mp_label_smoothing=0.1 --lr 0.00237 --weight-decay 1.03e-6 --dropout 0.493 '
            f'--top-k-labels {k} --artifact {graph_dir} --targets {a.targets} --output /nonexistent/x '
            f'--seed {seed}').split()
    args = T.normalize_method_args(T.parser().parse_args(argv))
    binding = json.loads((run / 'binding.json').read_text())
    prep_bytes = (run / 'preprocessing.json').read_bytes()
    assert hashlib.sha256(prep_bytes).hexdigest() == binding['preprocessing_sha256'], 'preprocessing hash'
    prep = tensorize.load_preprocessing(json.loads(prep_bytes))
    z = np.load(run / f'{a.fold}.npz', allow_pickle=True)
    ids = [str(x) for x in z['sample_ids']]
    y_all, proba_all = z['y'], z['proba']
    n = min(a.n_graphs, len(ids))
    pick, want = pick_graphs(ids, n)
    data = {}
    for graph, vm in contracts.iter_graphs_with_membership(graph_dir / 'graphs.jsonl',
                                                           graph_dir / contracts.VISIT_MEMBERSHIP_FILENAME,
                                                           wanted_ids=set(want)):
        d = tensorize.encode_graph(graph, prep, vm, args.edges, (), (), args.seed,
                                   edge_direction=args.edge_direction)
        i = want[graph['sample_id']]
        d.y = torch.tensor([int(y_all[i])], dtype=torch.long)
        data[i] = d
        if len(data) == len(want):
            break
    order = sorted(data)
    assert len(order) == n and set(order) == set(pick), 'graphs missing from the artifact'
    model = registry.build_method(
        args.method, num_tokens=len(prep['vocabulary']), node_dim=data[order[0]].x.size(1),
        edge_dim=data[order[0]].edge_attr.size(1), num_classes=binding['num_classes'], hidden=args.hidden,
        layers=args.layers, dropout=args.dropout, token_dim=args.token_dim,
        num_triples=tensorize.triple_count(prep, args.edge_direction), args=args)
    model.load_state_dict(torch.load(run / 'best.pt', map_location='cpu', weights_only=True), strict=True)
    model.eval()
    assert sum(p.numel() for p in model.parameters()) == binding['parameter_count'], 'parameter count'
    return dict(model=model, net=model.network, data=data, order=order, y=y_all, stored=proba_all,
                classes_in_sample=len(set(y_all[order].tolist())))


def forward_parts(model, net, batch):
    clinical, membership, visit_graph = model._read(batch)
    feats = net.continuous_inputs(clinical)
    parts = net.forward_continuous(feats, clinical.edge_index, clinical, membership, return_parts=True,
                                   visit_graph=visit_graph)
    return clinical, membership, visit_graph, feats, parts


def owners(clinical, p):
    b = clinical.batch_index
    return {'node': b, 'edge': b[clinical.edge_index[0]],
            'pair': b[p['pairs'][0]] if p['pairs'].size(1) else b[:0],
            'nbr': b, 'absence': p['absence_items'][0]}


def check_parity(softmax, stored, label):
    diff = float(np.abs(softmax - stored).max())
    agree = float((softmax.argmax(1) == stored.argmax(1)).mean())
    if diff > PARITY_TOL or agree < 1.0:
        raise SystemExit(f'parity failed for {label}: max abs diff {diff:.3e}, argmax agreement {agree:.4f}')
    return diff, agree


def probe_blocks(R, k, seed, torch, DataLoader):
    model, net, data, order = R['model'], R['net'], R['data'], R['order']
    n, c = len(order), net.num_classes
    loader = DataLoader([data[i] for i in order], batch_size=50)
    logit_rows, led_rows = [], []
    tot = {b: [] for b in BLOCKS}      # block totals [G, C]
    absum = {b: [] for b in BLOCKS}    # sum over items of |part| per class [G, C]
    with torch.no_grad():
        for batch in loader:
            clinical, _, _, _, p = forward_parts(model, net, batch)
            g = int(clinical.graph_count)
            own = owners(clinical, p)
            led = p['bias'].double().expand(g, c).clone()
            for b in BLOCKS:
                part = p[f'{b}_contributions'].double()
                t = torch.zeros((g, c), dtype=torch.float64).index_add(0, own[b], part)
                s = torch.zeros((g, c), dtype=torch.float64).index_add(0, own[b], part.abs())
                tot[b].append(t)
                absum[b].append(s)
                led = led + t
            logit_rows.append(p['logits'].double())
            led_rows.append(led)
    logits, ledger = torch.cat(logit_rows), torch.cat(led_rows)
    bt = {b: torch.cat(v) for b, v in tot.items()}
    ab = {b: torch.cat(v) for b, v in absum.items()}
    softmax = torch.softmax(logits.float(), dim=1).numpy()
    parity, agree = check_parity(softmax, R['stored'][order], f'K={k} seed={seed} blocks')
    ledger_diff = float((logits - ledger).abs().max())
    if ledger_diff > LEDGER_TOL:
        raise SystemExit(f'ledger not exact for K={k} seed={seed}: max |logit - bias - parts| {ledger_diff:.3e}')
    # item-level |part| mass (denominator: the five blocks and all classes), mean of per-graph shares
    m_item = {b: ab[b].sum(1) for b in BLOCKS}
    d_item = sum(m_item.values()).clamp_min(1e-12)
    share_item = {b: float((m_item[b] / d_item).mean()) for b in BLOCKS}
    # block-total mass (|block total| summed over classes): cancellation inside a block is allowed
    m_bt = {b: bt[b].abs().sum(1) for b in BLOCKS}
    d_bt = sum(m_bt.values()).clamp_min(1e-12)
    share_bt = {b: float((m_bt[b] / d_bt).mean()) for b in BLOCKS}
    top = logits.argmax(1)
    ar = torch.arange(len(top))
    pred = {b: float(bt[b][ar, top].mean()) for b in BLOCKS}
    pred_bias = float(net.bias.detach().double()[top].mean())
    pred_logit = float(logits[ar, top].mean())
    assert abs(sum(pred.values()) + pred_bias - pred_logit) < 1e-4
    base = {'K': k, 'seed': seed, 'n_graphs': n, 'parity_maxdiff': f'{parity:.3e}', 'ledger_maxdiff': f'{ledger_diff:.3e}'}
    tail = {f'pred_{b}': f'{pred[b]:.6f}' for b in BLOCKS}
    tail['pred_bias'] = f'{pred_bias:.6f}'
    main = {**base, **{f'mass_{b}': f'{share_item[b]:.6f}' for b in BLOCKS}, **tail}
    blocktotal = {**base, **{f'mass_{b}': f'{share_bt[b]:.6f}' for b in BLOCKS}, **tail}
    extra = {'K': k, 'seed': seed, 'n_graphs': n, 'pred_logit': f'{pred_logit:.6f}', 'argmax_agree': f'{agree:.4f}',
             'classes_in_sample': R['classes_in_sample']}
    print(f'K={k} seed={seed} n={n} classes_in_sample={R["classes_in_sample"]} parity={parity:.3e} '
          f'ledger={ledger_diff:.3e} argmax_agree={agree:.4f}', flush=True)
    return {'block_shares.csv': main, 'block_shares_blocktotal.csv': blocktotal, 'block_shares_extra.csv': extra}


def probe_removal(R, k, seed, torch, DataLoader):
    """Node block: deleting node j changes only the denominator and its own part (h0 is node-local):
    true removal total = (sum g*y - g_j*y_j) / (den - g_j); ledger-naive removal = total - part_j;
    gap = sum |d_true - d_ledger| / sum d_ledger  (d_ledger = |part_j|)."""
    model, net, data, order = R['model'], R['net'], R['data'], R['order']
    gaps, gap_pred, nbr_item, nbr_bt, softmax = [], [], [], [], []
    with torch.no_grad():
        for i in order:
            batch = next(iter(DataLoader([data[i]], batch_size=1)))
            _, _, _, _, p = forward_parts(model, net, batch)
            gate = p['node_gates'].double()
            den = p['node_denominator'][0].double()
            part = p['node_contributions'].double()
            num = (part * den).sum(0)
            total = num / den
            naive = total[None, :] - part
            true = (num[None, :] - part * den) / (den[None, :] - gate)
            d_led = (total[None] - naive).abs()
            d_true = (total[None] - true).abs()
            gaps.append(float((d_true - d_led).abs().sum() / d_led.sum().clamp_min(1e-12)))
            cls = int(p['logits'][0].argmax())
            dl, dt = d_led[:, cls], d_true[:, cls]
            gap_pred.append(float((dt - dl).abs().sum() / dl.sum().clamp_min(1e-12)))
            m = {b: p[f'{b}_contributions'].double().abs().sum() for b in BLOCKS}
            nbr_item.append(float(m['nbr'] / sum(m.values())))
            t = {b: p[f'{b}_contributions'].double().sum(0).abs().sum() for b in BLOCKS}
            nbr_bt.append(float(t['nbr'] / sum(t.values())))
            softmax.append(torch.softmax(p['logits'][0].float(), dim=0).numpy())
    parity, _ = check_parity(np.stack(softmax), R['stored'][order], f'K={k} seed={seed} removal')
    res = {'K': k, 'seed': seed, 'n_graphs': len(order),
           'node_block_removal_gap_rel_mean_allclasses': float(np.mean(gaps)),
           'node_block_removal_gap_rel_median_allclasses': float(np.median(gaps)),
           'node_block_removal_gap_rel_mean_predclass': float(np.mean(gap_pred)),
           'nbr_mass_share_item_level_mean': float(np.mean(nbr_item)),
           'nbr_mass_share_block_total_mean': float(np.mean(nbr_bt)),
           'parity_maxdiff': parity}
    print(json.dumps(res), flush=True)
    return {f'removal_probe_K{k}_s{seed}.json': res}


def probe_pairsecond(R, k, seed, torch, DataLoader):
    """Second difference of the pair vote: f(zi+zj) - f(zi) - f(zj) + f(0); zero iff endpoint-additive."""
    model, net, data, order = R['model'], R['net'], R['data'], R['order'][:PAIRS_GRAPHS]
    with torch.no_grad():
        batch = next(iter(DataLoader([data[i] for i in order], batch_size=len(order))))
        clinical, _, _, feats, p = forward_parts(model, net, batch)
        softmax = torch.softmax(p['logits'].float(), dim=1).numpy()
        parity, _ = check_parity(softmax, R['stored'][order], f'K={k} seed={seed} pairsecond')
        h0, _, _ = net._encode(feats, clinical, clinical.edge_index)
        zz = torch.tanh(net.pair_projection(h0))
        left, right = p['pairs']
        gen = torch.Generator().manual_seed(0)
        sel = torch.randperm(left.numel(), generator=gen)[:PAIRS_SAMPLED]
        zi, zj = zz[left[sel]], zz[right[sel]]
        f = net.pair_vote
        cross = f(zi + zj) - f(zi) - f(zj) + f(torch.zeros_like(zi))
        res = {'K': k, 'seed': seed, 'pairs_sampled': int(sel.numel()),
               'pair_second_difference_max_abs': float(cross.abs().max()),
               'pair_vote_mean_abs': float(f(zi + zj).abs().mean()),
               'pair_vote_head_is_linear': type(f).__name__, 'n_graphs': len(order), 'parity_maxdiff': parity}
    print(json.dumps(res), flush=True)
    return {f'pairsecond_K{k}_s{seed}.json': res}


PROBES = {'blocks': probe_blocks, 'removal': probe_removal, 'pairsecond': probe_pairsecond}


def write_outputs(out, rows):
    """rows: {file name: [row dict, ...] (csv) | dict (json)}. Runs are written sorted by (K, seed)."""
    out.mkdir(parents=True, exist_ok=False)
    for name, content in rows.items():
        if name.endswith('.json'):
            (out / name).write_text(json.dumps(content, indent=1) + '\n')
            continue
        content = sorted(content, key=lambda r: (int(r['K']), int(r['seed'])))
        with open(out / name, 'w', newline='') as stream:
            w = csv.DictWriter(stream, fieldnames=list(content[0].keys()))
            w.writeheader()
            w.writerows(content)


def execute(a, runs):
    import torch
    torch.set_num_threads(a.threads)
    sys.path.insert(0, str(REPO))
    from core import train as T
    from core import contracts, registry, tensorize
    from torch_geometric.loader import DataLoader
    modules = (torch, np, T, contracts, tensorize, registry)
    files = {}
    t0 = time.time()
    for k, seed in runs:
        t1 = time.time()
        R = load_run(a, k, seed, modules)
        got = PROBES[a.mode](R, k, seed, torch, DataLoader)
        for name, content in got.items():
            if name.endswith('.json'):
                files[name] = content
            else:
                files.setdefault(name, []).append(content)
        print(f'  run K={k} seed={seed} done in {time.time() - t1:.0f}s', flush=True)
    write_outputs(Path(a.out), files)
    print(f'wrote {len(files)} file(s) to {shown(a.out)} in {time.time() - t0:.0f}s')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m comparison.s3_ledger.probe', description=__doc__.split('\n\n')[0],
                                 epilog='Plan mode (default) prints the plan and loads no checkpoint.')
    ap.add_argument('mode', choices=MODES)
    ap.add_argument('--out', required=True, help='new output directory (must not exist; execute mode)')
    ap.add_argument('--fold', default='test', choices=FOLDS,
                    help='fold to score (default test = reported result; validation reproduces the earlier tables)')
    ap.add_argument('--ks', default='10,20,30')
    ap.add_argument('--seeds', default=','.join(map(str, ALL_SEEDS)))
    ap.add_argument('--runs-spec', default=None, help='explicit K:seed list (e.g. 10:7,20:2025,30:1234)')
    ap.add_argument('--n-graphs', type=int, default=None, help='graphs per run (default 300 blocks and pairsecond, 200 removal)')
    ap.add_argument('--runs', default=str(DEFAULTS['runs']), help='runs root (top<K>/cei_mp2_seed<S>/)')
    ap.add_argument('--artifact', default=str(DEFAULTS['artifact']), help='graph artifact directory')
    ap.add_argument('--targets', default=str(DEFAULTS['targets']), help='targets.csv')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--execute', action='store_true', help='load checkpoints and probe (inference only)')
    a = ap.parse_args(argv)
    a.ks, a.seeds = ints(a.ks), ints(a.seeds)
    if a.n_graphs is None:
        a.n_graphs = DEFAULT_N[a.mode]
    if a.n_graphs < 1:
        ap.error('--n-graphs must be positive')
    runs = run_list(a)
    if not runs or not all(s in ALL_SEEDS for _, s in runs):
        ap.error(f'runs must be K:seed with seed in {list(ALL_SEEDS)}')
    if 'standardized' in Path(a.out).resolve().parts:
        ap.error('--out must not be under comparison/standardized (inputs are read-only)')
    if a.execute and Path(a.out).exists():
        ap.error('--out exists; choose a new directory')
    print_plan(a, runs)
    if not a.execute:
        return 0
    execute(a, runs)
    return 0


if __name__ == '__main__':
    sys.exit(main())
