"""CEI-GNN mp2 ablation runs (Nasim section 4): 8 arms x K x seeds, same fair_v10t protocol and final CEI recipe.

Every arm REPLACES the listed method options of the final cei_mp2 command (comparison/final/run.py); nothing else
changes (lr 0.00237, dropout 0.493, 40 epochs, patience 40, validation selection, test scored once).
Usage (from the repo root):
    python3 -m comparison.ablation.run [--plan] [--ks 20] [--seeds 1234,2025,7] [--arms a1_full,a3_noabsence]
        --plan is the default: no --execute, prints one line per run (resolved protocol, overrides, epochs).
    python3 -m comparison.ablation.run --execute [...]       real training; needs the user's approval
Output: <root>/comparison/standardized/clinical_runs_fair_v10t_ablation_<date>/top<K>/<arm>_seed<S>. A non-empty
output directory is refused, never resumed or overwritten (use a new --date). `--root DIR` (or MEDGNN_DATA_ROOT)
reads inputs from and writes runs under DIR/comparison/standardized.
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GRAPH = 'clinical_graph_L4_plus60_v5c_20261009'
TARGETS = 'targets_primary_v4_v5plus60_v5_test_20261011/targets.csv'
V3_STATE = 'v3_state_L4_v5c_v10t_top{K}_20261011/K4.json'
WORKERS, THREADS = 6, 3
SEEDS = (1234, 2025, 7, 42, 3407)
# Full (a1) already exists for these seeds: the 45 final runs (same code, bit-identical logits); a1 is not rerun for them
FINAL_SEEDS = (1234, 2025, 7)
DEFAULT_KS = (20,)
# pooled-head width (arm a6) that matches Full's LIVE parameter count at each K: a6's 8,960 mp_ctx=table context
# parameters get no gradient, so they are not counted (user decision 2026-10-16; solved by building both)
POOL_HIDDEN = {10: 211, 20: 251, 30: 287}
# a6 trains with its own tuned lr/dropout/optimizer/label smoothing (user decision 2026-10-16: own HPO,
# `python3 -m comparison.final.hpo cei_a6_pool 12 --name ...`). None until that HPO is done: a6 is refused.
A6_HPARAMS = {'lr': 0.002638042027543271, 'dropout': 0.26086698080653065, 'optimizer': 'adam', 'label_smoothing': 0.1, 'weight_decay': 0.00017788368941377665}   # HPO ablation_a6_k17_20261017, trial 9
# encoder depth (1 + residual blocks) that matches Full's parameter count for the no-MP arm
MATCHED_ENCODER_DEPTH = 7

ARMS = {
    'a1_full': {'desc': 'Full CEI-GNN (final recipe)', 'options': lambda K: {}},
    'a2_mlp': {'desc': 'without evidence decomposition: MLP over the five block totals instead of their sum',
               'options': lambda K: {'ev_combine': 'mlp'}},
    'a3_noabsence': {'desc': 'without absence evidence (arm=B)', 'options': lambda K: {'arm': 'B'}},
    'a4_nopair': {'desc': 'without interaction (within-visit pair) evidence', 'options': lambda K: {'ev_pair': 0}},
    'a4b_nopair_noedge': {'desc': 'without pair and edge evidence', 'options': lambda K: {'ev_pair': 0, 'ev_edge': 0}},
    'a5_nomp': {'desc': 'without message passing', 'options': lambda K: {'mp_feed': 'none'}},
    'a5b_nomp_matched': {'desc': 'without message passing, capacity matched by a deeper encoder',
                         'options': lambda K: {'mp_feed': 'none', 'encoder_depth': MATCHED_ENCODER_DEPTH}},
    'a6_pool': {'desc': 'plain GNN, same encoder + message passing, pooled MLP head, no evidence mechanism',
                'options': lambda K: {'mp_feed': 'all', 'readout': 'pool', 'pool_hidden': POOL_HIDDEN[K]}},
}


class Paths:
    def __init__(self, root, date):
        std = Path(root) / 'comparison/standardized'
        self.events = std / 'event_inputs'
        self.graph = self.events / GRAPH
        self.targets = self.events / TARGETS
        self.runs = std / f'clinical_runs_fair_v10t_ablation_{date}'

    def v3_state(self, K):
        return self.events / V3_STATE.format(K=K)


def method_options(P, arm, K):
    """Ordered {option: value} of one arm: the final recipe with the arm's options REPLACED (never appended)."""
    options = {'arm': 'C', 'k': '4', 'v3_state': str(P.v3_state(K)), 'mp_feed': 'dual', 'mp_hub_max': '1',
               'mp_dedup': '1', 'mp_ctx': 'table', 'mp_root': 'bias', 'mp_optimizer': 'adamw',
               'mp_lr_schedule': 'cosine', 'mp_label_smoothing': '0.1'}
    options.update({k: str(v) for k, v in ARMS[arm]['options'](K).items()})
    if arm == 'a6_pool':
        aw = A6_HPARAMS['optimizer'] == 'adamw_cosine'
        options.update({'mp_optimizer': 'adamw' if aw else 'adam', 'mp_lr_schedule': 'cosine' if aw else 'none',
                        'mp_label_smoothing': str(A6_HPARAMS['label_smoothing'])})
    return options


def hparams(arm):
    """(lr, weight decay, dropout) of one arm: the final CEI values, or a6's own tuned values."""
    if arm == 'a6_pool':
        return str(A6_HPARAMS['lr']), str(A6_HPARAMS.get('weight_decay', 1.03e-6)), str(A6_HPARAMS['dropout'])
    return '0.00237', '1.03e-6', '0.493'


def cmd(P, arm, K, seed, out, execute, threads=THREADS):
    # executed runs go through /usr/bin/time -l: peak RSS and CPU time land at the end of the run's log
    c = (['/usr/bin/time', '-l'] if execute else []) + ['nice', '-n', '5', 'env', '-u', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE=1', '/usr/bin/python3', '-m', 'core.train',
         '--method', 'cei_gnn_mp2']
    for k, v in method_options(P, arm, K).items():
        c += ['--method-option', f'{k}={v}']
    lr, wd, dropout = hparams(arm)
    c += ['--lr', lr, '--weight-decay', wd, '--dropout', dropout, '--top-k-labels', str(K),
          '--artifact', str(P.graph), '--targets', str(P.targets), '--output', str(out), '--seed', str(seed),
          '--cpu-threads', str(threads)]
    return c + ['--execute'] if execute else c


def refused(out):
    """A directory with anything inside is filled: never resumed, never overwritten."""
    return out.exists() and (not out.is_dir() or any(out.iterdir()))


def plan_one(P, arm, K, seed):
    out = P.runs / f'top{K}' / f'{arm}_seed{seed}'
    if refused(out):
        return f'{K} {arm} seed{seed} REFUSED output dir is filled: {out}'
    o = subprocess.run(cmd(P, arm, K, seed, out, False), cwd=REPO, capture_output=True, text=True)
    t = o.stdout
    try:
        p = json.loads(t[t.index('{'):])
        pr = p['protocol_resolved']
        return (f"{K} {arm.ljust(18)} seed{str(seed).ljust(5)} {p.get('status')} {p.get('protocol')} "
                f"overrides {p.get('protocol_overrides')} epochs {pr['epochs']} topk {pr['top_k_labels']} "
                f"selection {pr['selection_fold']} final {pr['final_eval']}")
    except Exception:
        return f'{K} {arm} seed{seed} ERROR {(o.stdout + o.stderr).strip()[-400:]}'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', action='store_true', help='print the resolved protocol of every run and exit (default)')
    ap.add_argument('--execute', action='store_true', help='really train (needs the user\'s approval)')
    ap.add_argument('--ks', default=','.join(map(str, DEFAULT_KS)), help='comma-separated top-K label counts (default 20)')
    ap.add_argument('--seeds', default=','.join(map(str, SEEDS)), help='comma-separated seeds (default 1234,2025,7,42,3407)')
    ap.add_argument('--arms', default=','.join(ARMS), help='comma-separated arm ids (default all): ' + ', '.join(ARMS))
    ap.add_argument('--root', default=os.environ.get('MEDGNN_DATA_ROOT', str(REPO)))
    ap.add_argument('--date', default=time.strftime('%Y%m%d'), help='run-folder date suffix (default today)')
    ap.add_argument('--workers', type=int, default=WORKERS)
    ap.add_argument('--threads', type=int, default=THREADS)
    a = ap.parse_args(argv)
    if a.plan and a.execute:
        ap.error('--plan and --execute exclude each other')
    ks = tuple(int(k) for k in a.ks.split(','))
    seeds = tuple(int(s) for s in a.seeds.split(','))
    arms = tuple(x.strip() for x in a.arms.split(','))
    bad = [x for x in arms if x not in ARMS]
    if bad:
        ap.error(f'unknown arm(s) {bad}; known: {list(ARMS)}')
    if bad := [k for k in ks if k not in POOL_HIDDEN]:
        ap.error(f'K must be one of {sorted(POOL_HIDDEN)}, got {bad}')
    if len(set(seeds)) != len(seeds):
        ap.error('duplicate seeds')
    if 'a6_pool' in arms and A6_HPARAMS is None:
        ap.error('a6_pool needs its own HPO first (A6_HPARAMS is None); leave it out with --arms until then')
    P = Paths(a.root, a.date)
    skipped = [(arm, K, s) for K in ks for arm in arms for s in seeds if arm == 'a1_full' and s in FINAL_SEEDS]
    # seed-major order: every arm starts in the first waves, so an arm that fails on real data shows up early
    jobs = [(arm, K, s) for K in ks for s in seeds for arm in arms if (arm, K, s) not in skipped]
    for arm, K, s in skipped:
        print(f'{K} {arm.ljust(18)} seed{str(s).ljust(5)} SKIPPED reused from the final runs (clinical_runs_fair_v10t_20261011)')
    if not a.execute:
        with ThreadPoolExecutor(max(1, min(a.workers, 6))) as ex:   # plan lines are independent short subprocesses
            lines = list(ex.map(lambda job: plan_one(P, *job), jobs))
        print('\n'.join(lines))
        print(f'{len(lines)} runs planned (nothing executed); output root {P.runs}')
        return 0 if all(' ERROR ' not in x and ' REFUSED ' not in x for x in lines) else 1
    bad = [str(P.runs / f'top{K}' / f'{arm}_seed{s}') for arm, K, s in jobs if refused(P.runs / f'top{K}' / f'{arm}_seed{s}')]
    if bad:
        print('REFUSED: output dir(s) already filled (use a new --date):\n' + '\n'.join(bad), file=sys.stderr)
        return 1

    def log(msg):
        P.runs.mkdir(parents=True, exist_ok=True)
        with open(P.runs / 'progress.txt', 'a') as f:
            f.write(time.strftime('%F %T ') + msg + '\n')

    def run(job):
        arm, K, seed = job
        out = P.runs / f'top{K}' / f'{arm}_seed{seed}'
        out.parent.mkdir(parents=True, exist_ok=True)
        (P.runs / 'logs').mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(P.runs / 'logs' / f'top{K}_{arm}_seed{seed}.log', 'w') as f:
            rc = subprocess.run(cmd(P, arm, K, seed, out, True, a.threads), cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode
        ok = rc == 0 and (out / 'result.json').is_file()   # a run counts only with its result.json
        log(f'{"DONE" if ok else "FAIL"} top{K} {arm} seed{seed} rc={rc} {round((time.time() - t0) / 60, 1)} min')
        return ok

    log(f'START {len(jobs)} runs K={ks} seeds={seeds} arms={arms}, {a.workers} workers x {a.threads} threads, HEAD ' +
        subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip())
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(run, jobs))
    failed = [f'top{K} {arm} seed{s}' for (arm, K, s), ok in zip(jobs, results) if not ok]
    if failed:   # non-zero exit: a failed run must never pass silently into the analysis
        log(f'ALL RUNS FINISHED, {len(failed)} FAILED: ' + ', '.join(failed))
        print(f'{len(failed)} run(s) failed: ' + ', '.join(failed), file=sys.stderr)
        return 1
    log('ALL RUNS FINISHED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
