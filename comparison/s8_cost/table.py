"""One cost table per K (cei_mp2, gsat, graphcare, protgnn) from the outputs of params.py, bench.py and train_cost.py.

Usage (from the repo root):
    python3 -m comparison.s8_cost.table --params DIR --bench DIR --train DIR --out NEW_DIR [--methods cei_mp2,gsat,graphcare,protgnn] [--ks 10,20,30]

Reads only the summary.json of each input directory (no checkpoint, graph or training file is touched). Writes
top<K>.md, top<K>.csv per K and summary.json into NEW_DIR (must not exist). Outputs are aggregated: no sample ids,
subject ids, per-row values or absolute paths; input directories are recorded by name only. Caveat lines are derived
from the inputs (load average, parity, contended epochs, batch sizes, missing ablation memory). Validation fold for
timings; test numbers are not used anywhere.
"""
import argparse, csv, json, sys
from pathlib import Path

DEFAULT_METHODS = 'cei_mp2,gsat,graphcare,protgnn'
DEFAULT_KS = '10,20,30'
DISPLAY = {'cei_mp2': 'CEI-GNN mp2', 'gsat': 'GSAT', 'graphcare': 'GraphCare-style (adapted)', 'protgnn': 'ProtGNN'}
NA = 'n/a'

# (key, markdown header, CSV column, decimals or None for integers)
COLUMNS = [
    ('params_total', 'Params total', 'parameters_total', None),
    ('params_active', 'Params active', 'parameters_active', None),
    ('explanation_params', 'Explanation params', 'explanation_params', None),
    ('forward_ms', 'Forward ms/graph', 'forward_ms_per_graph', 2),
    ('tensorize_ms', 'Tensorize ms/graph', 'tensorize_ms_per_graph', 2),
    ('rss_mb', 'Inference peak RSS MB', 'inference_peak_rss_mb', 1),
    ('s_per_epoch', 'Median s/epoch (contended)', 'median_s_per_epoch', 1),
    ('total_min', 'Total train min', 'total_train_minutes', 1),
    ('epochs_95', 'Epochs to 95%', 'epochs_to_95_mean', 1),
    ('epochs_99', 'Epochs to 99%', 'epochs_to_99_mean', 1),
    ('selected', 'Selected epoch (mean, min-max)', 'selected_epoch_mean', 1),
    ('expl_ms', 'Explanation overhead ms/graph', 'explanation_overhead_ms_per_graph', 2),
    ('train_batch', 'Train batch', 'train_batch_size', None),
]


def load(directory, name):
    p = Path(directory) / 'summary.json'
    if not p.is_file():
        raise SystemExit(f'missing {name} summary.json in {Path(directory).name}')
    return json.loads(p.read_text())


def fmt(v, nd):
    if v is None:
        return NA
    if nd is None:
        return f'{int(v):,}'
    return f'{v:,.{nd}f}'


def build_rows(K, methods, params, bench, train):
    """One dict per method for this K; every value is raw (None = missing)."""
    pmap = {(r['K'], r['method']): r for r in params['records']}
    bmap = {(r['K'], r['method']): r for r in bench['records'] if 'error' not in r}
    tmap = {(r['K'], r['method']): r for r in train['records']}
    rows = []
    for m in methods:
        p, b, t = pmap.get((K, m)), bmap.get((K, m)), tmap.get((K, m))
        expl = (b or {}).get('explanation') or {}
        sel_mean = t['selected_epoch_mean'] if t else None
        rows.append(dict(
            method=m, display=DISPLAY.get(m, m),
            params_total=p['parameter_count'] if p else None,
            params_active=p['active_parameter_count'] if p else None,
            explanation_params=p['explanation_params'] if p else None,
            explanation_note=p.get('explanation_note') if p else None,
            forward_ms=b['forward_ms_per_graph']['median'] if b else None,
            tensorize_ms=b['tensorize_ms_per_graph']['median'] if b else None,
            rss_mb=b['rss_mb_peak'] if b else None,
            bench_graphs=b['graphs'] if b else None,
            bench_batch=b['batch_size'] if b else None,
            bench_contended=b['contended'] if b else None,
            bench_load=(b['loadavg_1min_before'], b['loadavg_1min_after']) if b else None,
            bench_parity_pass=b['parity_pass'] if b else None,
            bench_parity_diff=b['parity_max_abs_diff'] if b else None,
            s_per_epoch=t['median_s_per_epoch'] if t else None,
            total_min=t['total_minutes'] if t else None,
            epochs_95=t['epochs_to_95_mean'] if t else None,
            epochs_99=t['epochs_to_99_mean'] if t else None,
            selected=sel_mean,
            selected_min=t['selected_epoch_min'] if t else None,
            selected_max=t['selected_epoch_max'] if t else None,
            boundary_count=t['boundary_count'] if t else None,
            n_runs=t['n_runs'] if t else None,
            contended_epochs=t['contended_epochs'] if t else None,
            train_batch=p['batch_size'] if p else (t or {}).get('batch_size'),
            expl_ms=expl.get('overhead_ms_per_graph') if expl.get('available') else None,
        ))
    return rows


def caveats(methods, ks, params, bench, train, ablation_status):
    out = []
    sel = [(K, m) for K in ks for m in methods]
    bench_recs = [r for r in bench['records'] if (r['K'], r['method']) in sel]
    bench_err = [f"K={r['K']} {r['method']}" for r in bench['records'] if 'error' in r and (r['K'], r['method']) in sel]
    bt = bench['task']
    if bench_recs:
        contended = [r for r in bench_recs if r.get('contended')]
        loads = [x for r in bench_recs for x in (r['loadavg_1min_before'], r['loadavg_1min_after'])]
        if contended:
            out.append(f"Inference timings were measured on a loaded machine: 1-minute load average {min(loads):.2f}-{max(loads):.2f} "
                       f"(limit {bt['load_limit_1min']:g}) during {len(contended)} of {len(bench_recs)} measurements; "
                       "the ms/graph and RSS columns are indicative only.")
        else:
            out.append(f"No inference measurement was flagged contended (1-minute load {min(loads):.2f}-{max(loads):.2f}, "
                       f"limit {bt['load_limit_1min']:g}).")
        out.append(f"Inference bench sample: {bt['graphs']} validation graphs (sample seed {bt['sample_seed']}, drawn from the "
                   f"K=10 validation fold of the seed-{bt['checkpoint_seed']} GSAT run; the same graphs for every K), "
                   f"checkpoint seed {bt['checkpoint_seed']}, {bt['threads']} threads, batch {bt['batch_size']}, "
                   f"{bt['rounds']} round(s) x {bt['passes']} timed pass(es) after 1 warm-up.")
        fails = [f"K={r['K']} {r['method']}" for r in bench_recs if not r.get('parity_pass')]
        worst = max(r['parity_max_abs_diff'] for r in bench_recs)
        if fails or bench_err:
            out.append(f"Bench parity FAILED for {', '.join(fails + bench_err)} (tolerance {bt['parity_tolerance_max_abs_diff']:g}).")
        else:
            out.append(f"Bench parity PASSED for all {len(bench_recs)} records (max abs diff of softmax vs stored validation "
                       f"predictions {worst:.1e}, tolerance {bt['parity_tolerance_max_abs_diff']:g}).")
        neg = [f"K={r['K']} {r['method']}" for r in bench_recs
               if (r.get('explanation') or {}).get('available') and (r['explanation'].get('overhead_ms_per_graph') or 0) < 0]
        if neg:
            out.append(f"Negative explanation overhead ({', '.join(neg)}) is timing noise; read as about 0 ms/graph.")
    for r in bench_err:
        out.append(f"Bench record missing (error) for {r}; its cells are n/a.")

    tt = train['task_identity']
    tr_recs = [r for r in train['records'] if (r['K'], r['method']) in sel]
    if tr_recs:
        cont = {}
        for r in tr_recs:
            cont.setdefault(r['K'], 0)
            cont[r['K']] += r['contended_epochs']
        per_k = ', '.join(f"K={K}: {cont.get(K, 0)}" for K in ks if K in cont)
        out.append(f"Training runs were executed {tt['machine']}. Flagged contended epochs (epoch > 1.6 x run median) per K: {per_k}.")
        flagged = [K for K in ks if cont.get(K, 0) > 0]
        if flagged:
            out.append(f"Training s/epoch for {', '.join(f'K={K}' for K in flagged)} comes from contended runs "
                       "(6 concurrent runs x 3 threads; overlapping runners); it is not an uncontended speed.")
        bnd = [f"K={r['K']} {r['method']} {r['boundary_count']}/{r['n_runs']}" for r in tr_recs if r['boundary_count']]
        if bnd:
            out.append(f"Selected epoch at the budget boundary (38-40 of 40): {', '.join(bnd)} runs.")
        out.append(f"Training statistics: seeds {', '.join(map(str, tt['seeds']))}, {tt['epochs']} epochs, selection by best "
                   "validation macro-F1; epochs to 95%/99% and selected epoch are means over seeds.")
    batches = {}
    for r in params['records']:
        if (r['K'], r['method']) in sel:
            batches.setdefault(r['batch_size'], set()).add(DISPLAY.get(r['method'], r['method']))
    if batches:
        parts = '; '.join(f"{', '.join(sorted(v))} {k}" for k, v in sorted(batches.items()))
        out.append(f"Training batch sizes differ, read from the run bindings: {parts}. "
                   f"The inference bench uses batch {bt['batch_size']} for every method.")
    abl_text = {'read': 'read, but are not shown in this table',
                'not requested': 'not used',
                'not found': 'not available (the given ablation extra_metrics file was not found)'}.get(
        ablation_status, f'not available (status: {ablation_status})')
    if ablation_status == 'not requested':
        out.append('Training memory was not recorded in the final runs; the memory column is inference memory only.')
    else:
        out.append(f'CEI-GNN training memory is not in this table: the ablation extra metrics were {abl_text}.')
    pt = params.get('execute')
    if pt:
        ok = all(r.get('rebuild_matches_binding') for r in params['records'] if (r['K'], r['method']) in sel)
        out.append('Parameter counts: seed-1234 models rebuilt and strict-loaded; recount '
                   + ('matches the bindings for every row.' if ok else 'DIFFERS from the bindings for some rows (see params summary).'))
    no_head = sorted({DISPLAY.get(r['method'], r['method']) for r in params['records']
                      if (r['K'], r['method']) in sel and r.get('explanation_params') == 0})
    out.append('Explanation columns: n/a means no built-in explanation head; explanation parameters are 0 for '
               f"{', '.join(no_head) or 'none'} (CEI-GNN mp2: its evidence heads are the prediction path).")
    out.append('Test fold: not used in any number of this table.')
    return out


def write_md(path, K, rows, caveat_lines, identity):
    head = [f'# Model cost, K={K} (top-{K} labels)', '',
            f"Task identity: K={K}; validation fold (timings: {identity['graphs']} graphs, sample seed {identity['sample_seed']}); "
            f"checkpoints seed {identity['checkpoint_seed']} (parameters, bench); training statistics seeds "
            f"{', '.join(map(str, identity['seeds']))}, {identity['epochs']} epochs; rows train {identity['train_rows']}, "
            f"validation {identity['validation_rows']}; test not used.", '']
    hdr = '| Method | ' + ' | '.join(c[1] for c in COLUMNS) + ' |'
    sep = '|' + '---|' * (len(COLUMNS) + 1)
    lines = head + [hdr, sep]
    for r in rows:
        cells = [r['display']]
        for key, _, _, nd in COLUMNS:
            if key == 'selected':
                cells.append(NA if r['selected'] is None else
                             f"{r['selected']:.1f} ({r['selected_min']}-{r['selected_max']})")
            else:
                raw = {'params_total': r['params_total'], 'params_active': r['params_active'],
                       'explanation_params': r['explanation_params'], 'forward_ms': r['forward_ms'],
                       'tensorize_ms': r['tensorize_ms'], 'rss_mb': r['rss_mb'], 's_per_epoch': r['s_per_epoch'],
                       'total_min': r['total_min'], 'epochs_95': r['epochs_95'], 'epochs_99': r['epochs_99'],
                       'expl_ms': r['expl_ms'], 'train_batch': r['train_batch']}[key]
                cells.append(fmt(raw, nd))
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'Notes: forward, tensorize and RSS are inference (bench, validation graphs, 2 threads). Median s/epoch, total '
              'minutes, epochs to 95%/99% and selected epoch are training runs (seeds as above). Explanation overhead: GSAT edge '
              'top-5 extraction, ProtGNN top-3 prototype similarities, CEI-GNN mp2 ledger extraction over logits-only; '
              'n/a = no built-in explanation.', '', '## Caveats', '']
    lines += [f'- {c}' for c in caveat_lines]
    path.write_text('\n'.join(lines) + '\n')


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'display_name'] + [c[2] for c in COLUMNS[:6]] + ['selected_epoch_min', 'selected_epoch_max',
                   'epochs_to_95_mean', 'epochs_to_99_mean', 'median_s_per_epoch', 'total_train_minutes',
                   'explanation_overhead_ms_per_graph', 'train_batch_size', 'bench_batch_size', 'bench_graphs'])
        for r in rows:
            w.writerow([r['method'], r['display'], r['params_total'], r['params_active'], r['explanation_params'],
                        r['forward_ms'], r['tensorize_ms'], r['rss_mb'], r['selected_min'], r['selected_max'],
                        r['epochs_95'], r['epochs_99'], r['s_per_epoch'], r['total_min'], r['expl_ms'],
                        r['train_batch'], r['bench_batch'], r['bench_graphs']])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--params', required=True, help='params.py output directory')
    ap.add_argument('--bench', required=True, help='bench.py output directory')
    ap.add_argument('--train', required=True, help='train_cost.py output directory')
    ap.add_argument('--out', required=True, help='NEW output directory (an existing one is refused)')
    ap.add_argument('--methods', default=DEFAULT_METHODS,
                    help=f'comma-separated (default {DEFAULT_METHODS})')
    ap.add_argument('--ks', default=DEFAULT_KS, help=f'comma-separated K values (default {DEFAULT_KS})')
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'refusing existing --out {out.name}')
    methods = [m for m in a.methods.split(',') if m]
    ks = [int(k) for k in a.ks.split(',') if k]
    if not methods or not ks:
        raise SystemExit('need at least one method and one K')

    params, bench, train = load(a.params, 'params'), load(a.bench, 'bench'), load(a.train, 'train')
    ablation_status = train.get('ablation_extra', {}).get('status', 'not requested')
    cav = caveats(methods, ks, params, bench, train, ablation_status)
    bt = bench['task']

    out.mkdir(parents=True)
    per_k = {}
    for K in ks:
        rows = build_rows(K, methods, params, bench, train)
        rk = params['task_identity']['rows'].get(f'K{K}', {})
        identity = dict(graphs=bt['graphs'], sample_seed=bt['sample_seed'], checkpoint_seed=bt['checkpoint_seed'],
                        seeds=train['task_identity']['seeds'], epochs=train['task_identity']['epochs'],
                        train_rows=rk.get('train'), validation_rows=rk.get('validation'))
        write_md(out / f'top{K}.md', K, rows, cav, identity)
        write_csv(out / f'top{K}.csv', rows)
        per_k[str(K)] = dict(identity=identity, rows=[{k: v for k, v in r.items()} for r in rows])

    summary = dict(
        task='one cost table per K: parameters, explanation parameters, inference (bench), training time and convergence',
        task_identity=dict(protocol=bt['protocol'], timing_fold='validation', timing_graphs=bt['graphs'],
                           timing_sample_seed=bt['sample_seed'], checkpoint_seed=bt['checkpoint_seed'],
                           bench_threads=bt['threads'], bench_batch_size=bt['batch_size'],
                           training_seeds=train['task_identity']['seeds'], epochs=train['task_identity']['epochs'],
                           test_fold='not used'),
        inputs={'params': Path(a.params).name, 'bench': Path(a.bench).name, 'train': Path(a.train).name},
        methods=methods, ks=ks, display_names={m: DISPLAY.get(m, m) for m in methods},
        ablation_extra_status=ablation_status, caveats=cav, per_k=per_k)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n')
    print(f'wrote {out.name}: ' + ', '.join(f'top{K}.md, top{K}.csv' for K in ks) + ', summary.json')


if __name__ == '__main__':
    sys.exit(main())
