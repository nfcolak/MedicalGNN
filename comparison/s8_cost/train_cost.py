"""Training time and convergence of the stored final runs, derived from history.json / result.json (no training).

Usage (from the repo root):
    python3 -m comparison.s8_cost.train_cost --runs comparison/standardized/clinical_runs_fair_v10t_20261011 --out NEW_DIR
        [--ablation-extra extra_metrics.json] [--print-probe-plan]

history.json `seconds` is CUMULATIVE wall time since process start: epoch 1 includes data loading, preprocessing fit and
model build (reported as start-up and left out of the epoch statistics); epoch e >= 2 costs seconds[e] - seconds[e-1]
(includes the validation pass and the train-eval subset pass). Runs were executed 6 at a time x 3 threads under nice 5, so
wall times carry contention: an epoch is flagged contended when it exceeds 1.6 x the run's median of epochs 2-6 (count
reported; progress.txt START/adopted lines are summarised when the file is present next to the runs).
Convergence: epochs to reach 95% / 99% of the run's best validation macro-F1, selected epoch (best validation macro-F1),
and the number of runs whose selected epoch is at the 40-epoch boundary (>= 38).
--ablation-extra (optional): CEI-GNN ablation-run training peak RSS and s/epoch from extra_metrics.json (group a1_full),
labelled "CEI-GNN, ablation run, shared machine".
--print-probe-plan prints (never runs) one uncontended 1-epoch K=30 training command per method under /usr/bin/time -l,
marked NEEDS TRAINING APPROVAL. --out must not exist; outputs are aggregated only (no sample ids, no per-row values).
"""
import argparse, csv, json, re, shlex, statistics as st, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_METHODS = 'cei_mp2,gsat,graphcare,protgnn'
DEFAULT_KS = '10,20,30'
CONTENTION_FACTOR = 1.6
BASELINE_EPOCHS = (2, 6)      # epochs whose median is the run's uncontended reference
BOUNDARY_EPOCH = 38           # selected epoch >= 38 of 40 counts as at the budget boundary
PROBE_K = 30
PROBE_METHODS = ('gsat', 'graphcare', 'cei_mp2', 'protgnn')


def rel(path):
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


def first_reach(f1, frac):
    best = max(f1)
    return next(i + 1 for i, v in enumerate(f1) if v >= frac * best)


def analyse_run(d):
    hist = json.load(open(d / 'history.json'))
    res = json.load(open(d / 'result.json'))
    cum = [h['seconds'] for h in hist]
    ep = [b - a for a, b in zip(cum[:-1], cum[1:])]                 # epochs 2..N
    ref = st.median(ep[BASELINE_EPOCHS[0] - 2:BASELINE_EPOCHS[1] - 1])
    f1 = [h['macro_f1'] for h in hist]
    sel = res['selected_epoch']
    return dict(seed=int(d.name.rsplit('seed', 1)[1]), epochs=len(hist), startup_s=cum[0],
                epoch_median_s=st.median(ep), epoch_min_s=min(ep), epoch_max_s=max(ep), epoch_mean_s=st.mean(ep),
                total_min=res['total_seconds'] / 60, reference_epoch_s=ref,
                contended_epochs=sum(1 for x in ep if x > CONTENTION_FACTOR * ref),
                best_val_macro_f1=max(f1), epochs_to_95=first_reach(f1, 0.95), epochs_to_99=first_reach(f1, 0.99),
                selected_epoch=sel, at_boundary=sel >= BOUNDARY_EPOCH)


def agg(runs):
    def col(k):
        return [r[k] for r in runs]
    return dict(
        seeds=col('seed'), epochs=sorted(set(col('epochs'))),
        median_s_per_epoch=round(st.mean(col('epoch_median_s')), 1),       # mean over seeds of each run's median
        min_s_per_epoch=round(min(col('epoch_min_s')), 1), max_s_per_epoch=round(max(col('epoch_max_s')), 1),
        startup_s=round(st.mean(col('startup_s')), 1),
        total_minutes=round(st.mean(col('total_min')), 1),
        contended_epochs=sum(col('contended_epochs')), contended_epochs_per_seed=col('contended_epochs'),
        epochs_to_95_mean=round(st.mean(col('epochs_to_95')), 1), epochs_to_99_mean=round(st.mean(col('epochs_to_99')), 1),
        epochs_to_95=col('epochs_to_95'), epochs_to_99=col('epochs_to_99'),
        selected_epoch=col('selected_epoch'), selected_epoch_mean=round(st.mean(col('selected_epoch')), 1),
        selected_epoch_min=min(col('selected_epoch')), selected_epoch_max=max(col('selected_epoch')),
        boundary_count=sum(col('at_boundary')), n_runs=len(runs))


def progress_summary(runs):
    p = Path(runs) / 'progress.txt'
    if not p.exists():
        return None
    starts, adopted, finished = [], 0, 0
    for line in open(p):
        line = line.rstrip('\n')
        if ' START ' in line:
            starts.append(line.split(' START ')[0].strip() + ' START ' + line.split(' START ')[1].split(',')[0])
        if 'adopted from earlier runner' in line:
            adopted += 1
        if 'ALL RUNS FINISHED' in line:
            finished += 1
    return dict(start_lines=starts, adopted_from_earlier_runner=adopted, finished_lines=finished,
                overlapping_runners=len(starts) > 1)


def ablation(path):
    """CEI-GNN ablation-run cost numbers (training peak RSS, s/epoch) of group a1_full; tolerant of the file layout."""
    data = json.load(open(path))
    def groups(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == 'a1_full':
                    yield v
                else:
                    yield from groups(v)
        elif isinstance(node, list):
            for v in node:
                yield from groups(v)
    found = list(groups(data))
    keep = re.compile(r'rss|seconds|sec_per|s_per_epoch|epoch_s|time', re.I)
    def flat(node, prefix=''):
        if isinstance(node, dict):
            for k, v in node.items():
                yield from flat(v, f'{prefix}{k}.')
        elif isinstance(node, list):
            for i, v in enumerate(node):
                yield from flat(v, f'{prefix}{i}.')
        elif isinstance(node, (int, float)) and not isinstance(node, bool) and keep.search(prefix):
            yield prefix.rstrip('.'), node
    return dict(label='CEI-GNN, ablation run, shared machine', group='a1_full', groups_found=len(found),
                fields={k: v for g in found for k, v in flat(g)})


def probe_plan(date):
    sys.path.insert(0, str(REPO))
    from comparison.final.run import Paths, args as final_args
    P = Paths(str(REPO), date)
    lines = []
    for m in PROBE_METHODS:
        c = ['/usr/bin/time', '-l', 'env', '-u', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE=1', '/usr/bin/python3', '-m', 'core.train'] + \
            final_args(P, m, PROBE_K) + ['--top-k-labels', str(PROBE_K), '--artifact', rel(P.graph), '--targets', rel(P.targets),
                                         '--output', f'<NEW_EMPTY_DIR>/top{PROBE_K}/{m}_probe_1ep', '--seed', '1234',
                                         '--cpu-threads', '3', '--epochs', '1', '--execute']
        # the v3_state method option carries an absolute path; print it repo-relative
        c = [x.replace(str(P.events), rel(P.events)) if str(P.events) in x else x for x in c]
        lines.append(f'{m}: ' + ' '.join(shlex.quote(x) if '<' not in x else x for x in c))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True, help='NEW output directory (an existing one is refused)')
    ap.add_argument('--methods', default=DEFAULT_METHODS, help=f'default {DEFAULT_METHODS}; gchm_pna only on explicit request')
    ap.add_argument('--ks', default=DEFAULT_KS)
    ap.add_argument('--ablation-extra', default=None, help='extra_metrics.json of the CEI ablation runs (optional)')
    ap.add_argument('--print-probe-plan', action='store_true',
                    help='print (never run) one uncontended 1-epoch K=30 training command per method; NEEDS TRAINING APPROVAL')
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'refusing existing --out {rel(out)}')
    ks = [int(k) for k in a.ks.split(',')]
    methods = [m for m in a.methods.split(',') if m]
    m_date = re.search(r'(\d{8})$', Path(a.runs).resolve().name)
    date = m_date.group(1) if m_date else '20261011'

    records, per_run = [], []
    for K in ks:
        for m in methods:
            dirs = run_dirs(a.runs, K, m)
            if not dirs:
                raise SystemExit(f'no runs for top{K}/{m} under {rel(a.runs)}')
            rs = [analyse_run(d) for d in dirs]
            first = json.load(open(dirs[0] / 'binding.json'))
            records.append(dict(K=K, method=m, batch_size=first.get('batch_size'), train_rows=first['counts']['train'],
                                validation_rows=first['counts']['validation'], **agg(rs)))
            per_run += [dict(K=K, method=m, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}) for r in rs]

    progress = progress_summary(a.runs)
    abl = None
    abl_status = 'not requested'
    if a.ablation_extra:
        if Path(a.ablation_extra).is_file():
            abl = ablation(a.ablation_extra)
            abl_status = 'read'
        else:
            abl_status = 'not found'

    out.mkdir(parents=True)
    summary = dict(
        task='training time and convergence of the stored final runs (derived from history.json / result.json; nothing trained)',
        runs=rel(a.runs), methods=methods, ks=ks,
        task_identity=dict(seeds=sorted({s for r in records for s in r['seeds']}), epochs=40, selection='best validation macro-F1',
                           rows={f"K{r['K']}": dict(train=r['train_rows'], validation=r['validation_rows'])
                                 for r in records if r['method'] == methods[0]},
                           machine='6 concurrent runs x 3 CPU threads, nice 5 (wall times carry contention)'),
        definitions=dict(
            seconds='history.json seconds is cumulative since process start; epoch 1 = start-up (data load + preprocessing + model build)',
            median_s_per_epoch='mean over seeds of each run median of epochs 2-N',
            total_minutes='mean over seeds of result.json total_seconds / 60 (includes start-up and test scoring)',
            contended_epoch=f'epoch > {CONTENTION_FACTOR} x the run median of epochs {BASELINE_EPOCHS[0]}-{BASELINE_EPOCHS[1]}',
            boundary_count=f'runs with selected epoch >= {BOUNDARY_EPOCH} of 40'),
        progress_txt=progress if progress is not None else 'not present next to the runs',
        ablation_extra=dict(status=abl_status, **(abl or {})),
        records=records)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    with open(out / 'train_cost_per_run.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_run[0]))
        w.writeheader()
        w.writerows(per_run)

    print('K  method     s/ep(med) min    startup total_min contended e95  e99  sel_mean(range)  boundary')
    for r in records:
        print(f"{r['K']:<2} {r['method']:<10} {r['median_s_per_epoch']:<9} {r['min_s_per_epoch']:<6} {r['startup_s']:<7} "
              f"{r['total_minutes']:<9} {r['contended_epochs']:<9} {r['epochs_to_95_mean']:<4} {r['epochs_to_99_mean']:<4} "
              f"{r['selected_epoch_mean']}({r['selected_epoch_min']}-{r['selected_epoch_max']})  {r['boundary_count']}/{r['n_runs']}")
    print('progress.txt:', 'absent' if progress is None else json.dumps(progress))
    print('ablation extra_metrics:', abl_status)
    print('wrote', rel(out))
    if a.print_probe_plan:
        print('\nPROBE PLAN (printed only, nothing is run) -- NEEDS TRAINING APPROVAL')
        print('one uncontended 1-epoch K=30 training per method; run alone on an idle machine; replace <NEW_EMPTY_DIR> by a new empty directory')
        for line in probe_plan(date):
            print('NEEDS TRAINING APPROVAL ' + line)


if __name__ == '__main__':
    main()
