"""Nasim feedback 2026-10-12, Section 2 item 4: GCHM-PNA with the AdamW + cosine schedule the other baselines used.

The final GCHM-PNA runs (comparison/final/run.py: `--method clinical_gnn --conv gchm`) trained at constant lr (native
defaults: AdamW, lr 1e-3, weight decay 1e-5, dropout 0.1, no scheduler). Arm C1_cosine adds only `--optimizer adamw_cosine`
(core/train.py: same AdamW over model.parameters(), same lr / weight decay / dropout, plus the per-epoch cosine factor
0.1-1.0 x lr). K=20, seeds 1234, 2025, 7; fair_v10t protocol (validation selection, test scored once; no overrides).

    python3 comparison/nasim/gchm_cosine.py --plan [--root DIR] [--date YYYYMMDD]   no --execute, one line per job
    python3 comparison/nasim/gchm_cosine.py --jobs-json [--root DIR]                job list with full argv (cwd = repo root)
Inputs are read from <root>/comparison/standardized; runs go to <root>/comparison/standardized/clinical_runs_nasim_s2_<date>/.
This script never trains; the printed jobs are scheduled by hand.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from comparison.final.run import Paths, args as final_args, THREADS  # noqa: E402  (GRAPH/TARGETS/V3_STATE live in Paths)

DATE = '20261012'
METHOD = 'gchm_pna'
ARMS = {'C1_cosine': ['--optimizer', 'adamw_cosine']}   # the only flags that differ from comparison/final/run.py args()
KS = (20,)
SEEDS = (1234, 2025, 7)
PREFIX = ['nice', '-n', '5', 'env', '-u', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE=1', '/usr/bin/python3', '-m', 'core.train']


def jobs(root, date):
    P = Paths(root, date)
    base = Path(root) / 'comparison/standardized' / f'clinical_runs_nasim_s2_{date}'
    out = []
    for arm, extra in ARMS.items():
        for K in KS:
            for seed in SEEDS:
                o = base / arm / f'top{K}' / f'{METHOD}_seed{seed}'
                c = PREFIX + final_args(P, METHOD, K) + extra + [
                    '--top-k-labels', str(K), '--artifact', str(P.graph), '--targets', str(P.targets), '--output', str(o),
                    '--seed', str(seed), '--cpu-threads', str(THREADS)]
                out.append(dict(name=f'{arm}_top{K}_{METHOD}_seed{seed}', arm=arm, method=METHOD, K=K, seed=seed, out=str(o),
                                log=str(base / 'logs' / f'{arm}_top{K}_{METHOD}_seed{seed}.log'), cmd=c, flags=extra))
    return out


def plan(js):
    ok = True
    for j in js:
        o = subprocess.run(j['cmd'], cwd=REPO, capture_output=True, text=True)   # no --execute
        t = o.stdout
        try:
            p = json.loads(t[t.index('{'):]); pr = p['protocol_resolved']
            line = (f"{j['arm']} top{j['K']} {j['method']} seed{j['seed']}: {p.get('status')} {p.get('protocol')} "
                    f"overrides {p.get('protocol_overrides')} epochs {pr['epochs']} selection {pr['selection_fold']} "
                    f"final {pr['final_eval']} | flags: {' '.join(j['flags'])}")
            ok &= p.get('protocol_overrides') == []
        except Exception:
            line = f"{j['arm']} top{j['K']} {j['method']} seed{j['seed']}: ERROR {(o.stdout + o.stderr).strip()[-300:]}"
            ok = False
        print(line)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=os.environ.get('MEDGNN_DATA_ROOT', str(REPO)),
                    help='directory holding comparison/standardized (default: env MEDGNN_DATA_ROOT or this repo)')
    ap.add_argument('--date', default=DATE, help=f'run-folder date suffix (default {DATE})')
    ap.add_argument('--plan', action='store_true', help='run core.train without --execute and print one line per job')
    ap.add_argument('--jobs-json', action='store_true', help='print the not-yet-finished jobs as JSON (with --execute)')
    a = ap.parse_args()
    js = jobs(a.root, a.date)
    if a.plan:
        sys.exit(0 if plan(js) else 1)
    if a.jobs_json:
        print(json.dumps([dict(j, cmd=j['cmd'] + ['--execute']) for j in js if not (Path(j['out']) / 'result.json').exists()], indent=1))
        return
    ap.error('give --plan or --jobs-json (this script never trains)')


if __name__ == '__main__':
    main()
