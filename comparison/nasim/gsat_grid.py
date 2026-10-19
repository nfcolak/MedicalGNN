"""Nasim Section 2, item 2: GSAT information-bottleneck grid (plan / job list only; jobs are scheduled by hand).

Arms: GSAT, K=20, seed 1234, the final tuned command of comparison/final/run.py args() with
`--gsat-info-loss-coef` in {0.01, 0.1, 1.0} (replaces the final 0.104) x `--gsat-final-r` in {0.5, 0.7}.
Decay schedule (gsat/adapter.py r_for_epoch, 0-based epoch, 40 epochs): r = max(init_r - (epoch // decay_interval) * decay_r, final_r)
with init_r 0.9, decay_r 0.1, decay_interval 10. That reaches 0.7 at epoch 20 but only 0.6 at epoch 30-39, so final_r 0.5 would
never be reached. The final_r 0.5 arms therefore add `--gsat-decay-interval 5` (0.5 from epoch 20, the same epoch at which the
r 0.7 arms reach their final r, so both r values train 20 epochs at final_r).
All jobs keep the fair_v10t protocol (validation selection, test scored once, reported only).

    python3 comparison/nasim/gsat_grid.py --plan [--root DIR]         no --execute, one line per job; exit 1 on any error
    python3 comparison/nasim/gsat_grid.py --jobs-json [--root DIR]    JSON job list (full argv with --execute), skips finished jobs
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from comparison.final.run import Paths, args, GRAPH, TARGETS, V3_STATE, THREADS   # noqa: E402,F401

DATE = '20261012'
METHOD, K, SEED = 'gsat', 20, 1234
IB = ('0.01', '0.1', '1.0')
FINAL_R = ('0.5', '0.7')
DECAY_INTERVAL = {'0.5': '5'}   # final_r 0.5 is unreachable in 40 epochs with the default interval 10 (r bottoms out at 0.6)
PREFIX = ['nice', '-n', '5', 'env', '-u', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE=1', '/usr/bin/python3', '-m', 'core.train']


def arms():
    return [(f'G_ib{ib}_r{r}', ib, r) for r in FINAL_R for ib in IB]


def arm_flags(ib, r):
    """The flags that differ from comparison/final/run.py args(): the new value of --gsat-info-loss-coef plus the appended ones."""
    return ['--gsat-info-loss-coef', ib, '--gsat-final-r', r] + (['--gsat-decay-interval', DECAY_INTERVAL[r]] if r in DECAY_INTERVAL else [])


def arm_args(P, ib, r):
    base = args(P, METHOD, K)
    i = base.index('--gsat-info-loss-coef')   # replace, never append a duplicate flag
    assert base[i + 1] and base.count('--gsat-info-loss-coef') == 1
    base = base[:i] + base[i + 2:]
    return base + arm_flags(ib, r)


def jobs(P, root, date):
    runs = Path(root) / 'comparison/standardized' / f'clinical_runs_nasim_s2_{date}'
    out = []
    for arm, ib, r in arms():
        o = runs / arm / f'top{K}' / f'{METHOD}_seed{SEED}'
        log = runs / 'logs' / f'{arm}_top{K}_{METHOD}_seed{SEED}.log'
        tail = ['--top-k-labels', str(K), '--artifact', str(P.graph), '--targets', str(P.targets), '--output', str(o),
                '--seed', str(SEED), '--cpu-threads', str(THREADS)]
        out.append(dict(name=f'{arm}_top{K}_{METHOD}_seed{SEED}', arm=arm, method=METHOD, K=K, seed=SEED, out=str(o), log=str(log),
                        flags=arm_flags(ib, r), plan_cmd=PREFIX + arm_args(P, ib, r) + tail))
    return out


def plan_check(P, root, date):
    lines = []
    for j in jobs(P, root, date):
        o = subprocess.run(j['plan_cmd'], cwd=REPO, capture_output=True, text=True)
        t = o.stdout
        try:
            p = json.loads(t[t.index('{'):]); pr = p['protocol_resolved']
            lines.append(f"{j['arm']} top{j['K']} {j['method']} seed{j['seed']}: {p.get('status')} {p.get('protocol')} "
                         f"overrides {p.get('protocol_overrides')} epochs {pr['epochs']} selection {pr['selection_fold']} "
                         f"final {pr['final_eval']} | flags: {' '.join(j['flags'])}")
            if p.get('protocol_overrides') != [] or o.returncode != 0:
                lines[-1] += ' ERROR protocol overrides or nonzero exit'
        except Exception:
            lines.append(f"{j['arm']} top{j['K']} {j['method']} seed{j['seed']}: ERROR {(o.stdout + o.stderr).strip()[-300:]}")
    print('\n'.join(lines))
    return all('ERROR' not in line for line in lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=os.environ.get('MEDGNN_DATA_ROOT', str(REPO)),
                    help='directory holding comparison/standardized (default: env MEDGNN_DATA_ROOT or the repo root)')
    ap.add_argument('--date', default=DATE, help=f'run-folder date suffix (default {DATE})')
    ap.add_argument('--plan', action='store_true', help='no --execute: print the resolved protocol of every job; exit 1 on error')
    ap.add_argument('--jobs-json', action='store_true', help='print the JSON job list (cmd includes --execute); finished jobs skipped')
    a = ap.parse_args()
    P = Paths(a.root, '20261011')   # inputs only; the run folder of this runner is built in jobs()
    if a.plan:
        sys.exit(0 if plan_check(P, a.root, a.date) else 1)
    if a.jobs_json:
        js = []
        for j in jobs(P, a.root, a.date):
            if (Path(j['out']) / 'result.json').exists():
                continue
            j['cmd'] = j.pop('plan_cmd') + ['--execute']
            j.pop('flags')
            j['cwd'] = str(REPO)
            js.append(j)
        print(json.dumps(js, indent=1))
        return
    ap.error('give --plan or --jobs-json (this runner never trains by itself)')


if __name__ == '__main__':
    main()
