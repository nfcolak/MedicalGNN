"""Nasim feedback 2.3 (ProtGNN): does selecting the checkpoint after prototype projection change ProtGNN?

Arm P1_proj10_select = the final ProtGNN command of comparison/final/run.py (args()) plus
`--protgnn-select-after-projection --protgnn-proj-interval 10` (proj_epochs stays 20: projections at the start of
epoch indices 20 and 30 of 40; the flag itself is added by unit s2-protgnn-code). K=20, seeds 1234/2025/7.
All jobs keep the fair_v10t protocol (overrides must be []): validation selection, test scored once, never used to decide.

Usage (repo root):
    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 comparison/nasim/protgnn_proj.py [--root DIR] [--date YYYYMMDD] --plan
    ... --jobs-json     print the jobs (with --execute) for the scheduler; this runner never trains itself
`--without-new-flag` (debug only) drops --protgnn-select-after-projection so the plan works before s2-protgnn-code is merged.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from comparison.final.run import Paths, args, THREADS   # noqa: E402  (GRAPH, TARGETS, V3_STATE are used through Paths)

DEFAULT_DATE = '20261012'
METHOD = 'protgnn'
KS = (20,)
SEEDS = (1234, 2025, 7)
NEW_FLAG = '--protgnn-select-after-projection'
ARMS = {'P1_proj10_select': [NEW_FLAG, '--protgnn-proj-interval', '10']}
PREFIX = ['nice', '-n', '5', 'env', '-u', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE=1', '/usr/bin/python3', '-m', 'core.train']


def jobs(P, base, date, without_new_flag):
    std = Path(base) / 'comparison/standardized' / f'clinical_runs_nasim_s2_{date}'
    for arm, extra in ARMS.items():
        extra = [x for x in extra if not (without_new_flag and x == NEW_FLAG)]
        for K in KS:
            for seed in SEEDS:
                name = f'{arm}_top{K}_{METHOD}_seed{seed}'
                out = std / arm / f'top{K}' / f'{METHOD}_seed{seed}'
                log = std / 'logs' / f'{arm}_top{K}_{METHOD}_seed{seed}.log'
                argv = PREFIX + args(P, METHOD, K) + extra + [
                    '--top-k-labels', str(K), '--artifact', str(P.graph), '--targets', str(P.targets),
                    '--output', str(out), '--seed', str(seed), '--cpu-threads', str(THREADS)]
                yield dict(name=name, arm=arm, method=METHOD, K=K, seed=seed, out=str(out), log=str(log),
                           argv=argv, extra=extra)


def plan(job_list):
    ok = True
    for j in job_list:
        o = subprocess.run(j['argv'], cwd=REPO, capture_output=True, text=True)   # no --execute
        head = f"{j['arm']} top{j['K']} {j['method']} seed{j['seed']}:"
        flags = 'flags: ' + ' '.join(j['extra'])
        try:
            t = o.stdout
            p = json.loads(t[t.index('{'):]); pr = p['protocol_resolved']
            if o.returncode != 0 or p.get('protocol_overrides') != []:
                ok = False
            print(f"{head} {p.get('status')} {p.get('protocol')} overrides {p.get('protocol_overrides')} epochs {pr['epochs']} "
                  f"selection {pr['selection_fold']} final {pr['final_eval']} | {flags}")
        except Exception:
            ok = False
            print(f"{head} ERROR {(o.stdout + o.stderr).strip()[-300:]} | {flags}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=os.environ.get('MEDGNN_DATA_ROOT', str(REPO)),
                    help='directory holding comparison/standardized (default: env MEDGNN_DATA_ROOT or the repo root)')
    ap.add_argument('--date', default=DEFAULT_DATE, help=f'run-folder date suffix (default {DEFAULT_DATE})')
    ap.add_argument('--plan', action='store_true', help='run core.train without --execute for every job and print one line per job')
    ap.add_argument('--jobs-json', action='store_true', help='print the jobs (full argv with --execute) as JSON; skips finished jobs')
    ap.add_argument('--without-new-flag', action='store_true', help=f'DEBUG ONLY: drop {NEW_FLAG} (not yet merged)')
    a = ap.parse_args()
    if a.plan == a.jobs_json:
        ap.error('give exactly one of --plan, --jobs-json')
    if a.without_new_flag:
        print(f'WARNING: {NEW_FLAG} dropped (debug); these jobs are NOT the P1 arm', file=sys.stderr)
    P = Paths(a.root, a.date)
    job_list = list(jobs(P, a.root, a.date, a.without_new_flag))
    if a.plan:
        sys.exit(0 if plan(job_list) else 1)
    out = []
    for j in job_list:
        if (Path(j['out']) / 'result.json').exists():
            continue
        out.append(dict(name=j['name'], arm=j['arm'], method=j['method'], K=j['K'], seed=j['seed'], out=j['out'], log=j['log'],
                        cmd=j['argv'] + ['--execute'], cwd=str(REPO)))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
