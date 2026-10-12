"""Final fair_v10t runs: 5 tuned methods x K (default 10,20,30) x seeds 1234/2025/7, 6 at a time, 3 CPU threads each.

Heavy jobs first. Skips runs whose result.json exists (resume-safe); a failed run dir is moved aside, never overwritten.
Usage (from the repo root):
    python3 -m comparison.final.run --plan [--ks 10,20,30]      no --execute, prints one line per (K, method)
    python3 -m comparison.final.run [--ks 30,10]                runs (real training; needs the user's approval)
Everything is repo-relative (`comparison/standardized/...`). `--root DIR` (or env MEDGNN_DATA_ROOT) reads inputs from and
writes runs under DIR/comparison/standardized instead; the code always runs from this repo.
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATE = '20261011'
GRAPH = 'clinical_graph_L4_plus60_v5c_20261009'
TARGETS = 'targets_primary_v4_v5plus60_v5_test_20261011/targets.csv'
V3_STATE = 'v3_state_L4_v5c_v10t_top{K}_20261011/K4.json'
WORKERS, THREADS = 6, 3
SEEDS = (1234, 2025, 7)
METHODS = ['cei_mp2', 'protgnn', 'gchm_pna', 'gsat', 'graphcare']   # heavy first
PLAN_ORDER = ['gsat', 'graphcare', 'cei_mp2', 'protgnn', 'gchm_pna']


class Paths:
    def __init__(self, root, date):
        std = Path(root) / 'comparison/standardized'
        self.events = std / 'event_inputs'
        self.graph = self.events / GRAPH
        self.targets = self.events / TARGETS
        self.runs = std / f'clinical_runs_fair_v10t_{date}'

    def v3_state(self, K):
        return self.events / V3_STATE.format(K=K)


def args(P, m, K):   # HPO-tuned settings (comparison/final/hpo.py), the same for every K
    return {
        'gsat': '--method gsat --optimizer adamw_cosine --lr 0.00369 --weight-decay 6.17e-5 --dropout 0.137 --gsat-info-loss-coef 0.104',
        'graphcare': '--method graphcare --optimizer adamw_cosine --lr 0.00281 --weight-decay 1.05e-6 --dropout 0.286 --graphcare-decay-rate 0.0116',
        'cei_mp2': (f'--method cei_gnn_mp2 --method-option arm=C --method-option k=4 --method-option v3_state={P.v3_state(K)} '
                    '--method-option mp_feed=dual --method-option mp_hub_max=1 --method-option mp_dedup=1 --method-option mp_ctx=table '
                    '--method-option mp_root=bias --method-option mp_optimizer=adamw --method-option mp_lr_schedule=cosine '
                    '--method-option mp_label_smoothing=0.1 --lr 0.00237 --weight-decay 1.03e-6 --dropout 0.493'),
        'protgnn': '--method protgnn --optimizer adamw_cosine --lr 0.00261 --weight-decay 4.42e-6 --dropout 0.345 --protgnn-prototypes-per-class 5',
        'gchm_pna': '--method clinical_gnn --conv gchm',
    }[m].split()


def cmd(P, m, K, seed, out, plan):
    c = ['nice', '-n', '5', 'env', '-u', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE=1', '/usr/bin/python3', '-m', 'core.train'] + args(P, m, K) + [
        '--top-k-labels', str(K), '--artifact', str(P.graph), '--targets', str(P.targets), '--output', str(out),
        '--seed', str(seed), '--cpu-threads', str(THREADS)]
    return c if plan else c + ['--execute']


def plan_check(P, ks):
    lines = []
    for K in ks:
        for m in PLAN_ORDER:
            o = subprocess.run(cmd(P, m, K, 1234, P.runs / f'top{K}' / f'{m}_seed1234', True), cwd=REPO, capture_output=True, text=True)
            t = o.stdout
            try:
                p = json.loads(t[t.index('{'):]); pr = p['protocol_resolved']
                lines.append(f"{K} {m.ljust(9)} {p.get('status')} {p.get('protocol')} overrides {p.get('protocol_overrides')} "
                             f"epochs {pr['epochs']} topk {pr['top_k_labels']} selection {pr['selection_fold']} final {pr['final_eval']}")
            except Exception:
                lines.append(f'{K} {m} ERROR {(o.stdout + o.stderr).strip()[-300:]}')
    print('\n'.join(lines))
    return all('ERROR' not in line for line in lines)


def live_pid(out):
    p = subprocess.run(['pgrep', '-f', '--', f'--output {out} '], capture_output=True, text=True)
    return [x for x in p.stdout.split() if x]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', action='store_true', help='no --execute: print the resolved protocol of every (K, method) and exit')
    ap.add_argument('--ks', default='10,20,30', help='comma-separated top-K label counts, run in this order (default 10,20,30)')
    ap.add_argument('--root', default=os.environ.get('MEDGNN_DATA_ROOT', str(REPO)),
                    help='directory holding comparison/standardized (default: this repo)')
    ap.add_argument('--date', default=DATE, help=f'run-folder date suffix (default {DATE})')
    a = ap.parse_args()
    ks = tuple(int(k) for k in a.ks.split(','))
    P = Paths(a.root, a.date)
    if a.plan:
        sys.exit(0 if plan_check(P, ks) else 1)

    def log(msg):
        P.runs.mkdir(parents=True, exist_ok=True)
        with open(P.runs / 'progress.txt', 'a') as f:
            f.write(time.strftime('%F %T ') + msg + '\n')

    def run(job):
        m, K, seed = job
        out = P.runs / f'top{K}' / f'{m}_seed{seed}'
        if (out / 'result.json').exists():
            return
        if out.exists() and live_pid(out):   # started by an earlier runner and still training: wait for it, do not restart
            t0 = time.time()
            while live_pid(out):
                time.sleep(30)
            ok = (out / 'result.json').exists()
            log(f'{"DONE" if ok else "FAIL"} top{K} {m} seed{seed} (adopted from earlier runner) +{round((time.time() - t0) / 60, 1)} min')
            return
        if out.exists():
            out.rename(out.with_name(out.name + f'_failed_{int(time.time())}'))
        out.parent.mkdir(parents=True, exist_ok=True)
        (P.runs / 'logs').mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(P.runs / 'logs' / f'top{K}_{m}_seed{seed}.log', 'w') as f:
            rc = subprocess.run(cmd(P, m, K, seed, out, False), cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode
        log(f'{"DONE" if rc == 0 else "FAIL"} top{K} {m} seed{seed} rc={rc} {round((time.time() - t0) / 60, 1)} min')

    jobs = [(m, K, s) for K in ks for m in METHODS for s in SEEDS]
    log(f'START {len(jobs)} runs K={ks}, {WORKERS} workers x {THREADS} threads, HEAD ' +
        subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip())
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(run, jobs))
    log('ALL RUNS FINISHED')


if __name__ == '__main__':
    main()
