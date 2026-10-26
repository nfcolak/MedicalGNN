"""Nasim feedback 2.1 recipe control: CEI-GNN mp2 at K=20 with the label smoothing (and the weight decay) of the final recipe removed.

Arms (everything else is exactly the final CEI command of comparison/final/run.py args()):
    R1_ls0      mp_label_smoothing=0.0 (final: 0.1)
    R2_ls0_wd   mp_label_smoothing=0.0 and mp_weight_decay=1.03e-6 (final: AdamW decoupled decay 0.01); the baseline-like recipe
    R3_ls0_wd_de0  R2 plus mp_dropedge=0.0 (final: default 0.2); the full baseline-like recipe: no label smoothing, small decay, no DropEdge
The reference arm is the existing final run set (clinical_runs_fair_v10t_20261011/top20/cei_mp2_seed*), not rerun.
Each arm REPLACES the mp_label_smoothing option token of the final args (core.train keeps only the last value of a repeated key);
mp_weight_decay is inserted right after it. Protocol stays fair_v10t (overrides []): validation selection, test scored once.

Usage (repo root, env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 comparison/nasim/recipe_control.py ...):
    --plan         no --execute: one line per job, exit 1 on any error
    --jobs-json    JSON list of jobs (full argv with --execute); skips jobs whose out/result.json exists
    --root DIR     inputs from DIR/comparison/standardized (default env MEDGNN_DATA_ROOT or the repo root)
    --date YYYYMMDD  run-folder suffix (default 20261012)
    --arms A[,B...]  only these arms in --plan / --jobs-json (default: all)
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from comparison.final.run import Paths, args as final_args, THREADS   # noqa: E402  (GRAPH/TARGETS/V3_STATE are used through Paths)

DATE = '20261012'
K = 20
METHOD = 'cei_mp2'
SEEDS = (1234, 2025, 7)
FINAL_LS = 'mp_label_smoothing=0.1'
ARMS = {   # arm -> option tokens that replace FINAL_LS in the final args
    'R1_ls0': ['mp_label_smoothing=0.0'],
    'R2_ls0_wd': ['mp_label_smoothing=0.0', 'mp_weight_decay=1.03e-6'],
    'R3_ls0_wd_de0': ['mp_label_smoothing=0.0', 'mp_weight_decay=1.03e-6', 'mp_dropedge=0.0'],
}
PREFIX = ['nice', '-n', '5', 'env', '-u', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE=1', '/usr/bin/python3', '-m', 'core.train']


def arm_args(P, arm):
    """Final CEI args with the arm's options replacing the mp_label_smoothing=0.1 option (no duplicate key)."""
    base = final_args(P, METHOD, K)
    idx = [i for i, t in enumerate(base) if t == FINAL_LS]
    if len(idx) != 1 or base[idx[0] - 1] != '--method-option':
        raise SystemExit(f'final CEI args no longer contain exactly one --method-option {FINAL_LS}')
    i = idx[0]
    new = []
    for opt in ARMS[arm]:
        new += ['--method-option', opt]
    out = base[:i - 1] + new + base[i + 1:]
    keys = [t.partition('=')[0] for a, t in zip(out, out[1:]) if a == '--method-option']
    assert len(keys) == len(set(keys)), f'duplicate --method-option key in {keys}'
    return out


def flag_diff(P, arm):
    """The flags that differ from the final CEI command, as written in the command line."""
    base, mine = final_args(P, METHOD, K), arm_args(P, arm)
    drop = [t for t in base if t not in mine]
    add = [t for t in mine if t not in base]
    return f"replaces --method-option {' '.join(drop)} with " + ' '.join(f'--method-option {t}' for t in add)


def jobs(P, runs):
    res = []
    for arm in ARMS:
        for seed in SEEDS:
            out = runs / arm / f'top{K}' / f'{METHOD}_seed{seed}'
            log = runs / 'logs' / f'{arm}_top{K}_{METHOD}_seed{seed}.log'
            cmd = PREFIX + arm_args(P, arm) + ['--top-k-labels', str(K), '--artifact', str(P.graph), '--targets', str(P.targets),
                                                '--output', str(out), '--seed', str(seed), '--cpu-threads', str(THREADS)]
            res.append({'name': f'{arm}_top{K}_{METHOD}_seed{seed}', 'arm': arm, 'method': METHOD, 'K': K, 'seed': seed,
                        'out': str(out), 'log': str(log), 'cmd': cmd + ['--execute']})
    return res


def plan(P, js):
    ok = True
    for j in js:
        o = subprocess.run(j['cmd'][:-1], cwd=REPO, capture_output=True, text=True)   # without --execute
        t = o.stdout
        try:
            p = json.loads(t[t.index('{'):])
            pr = p['protocol_resolved']
            if p.get('protocol_overrides'):
                raise ValueError('protocol_overrides not empty')
            line = (f"{j['arm']} top{j['K']} {j['method']} seed{j['seed']}: {p.get('status')} {p.get('protocol')} overrides "
                    f"{p.get('protocol_overrides')} epochs {pr['epochs']} selection {pr['selection_fold']} final {pr['final_eval']}"
                    f" | flags: {flag_diff(P, j['arm'])}")
        except Exception as e:
            ok = False
            line = f"{j['arm']} top{j['K']} {j['method']} seed{j['seed']}: ERROR {e} {(o.stdout + o.stderr).strip()[-300:]}"
        print(line)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=os.environ.get('MEDGNN_DATA_ROOT', str(REPO)), help='directory holding comparison/standardized')
    ap.add_argument('--date', default=DATE, help=f'run-folder date suffix (default {DATE})')
    ap.add_argument('--plan', action='store_true', help='no --execute: print one line per job and exit')
    ap.add_argument('--jobs-json', action='store_true', help='print the jobs (argv with --execute) as JSON, skipping finished ones')
    ap.add_argument('--arms', default=None, help='comma-separated arm names to include (default: all)')
    a = ap.parse_args()
    P = Paths(a.root, '')   # only graph/targets/v3_state are used
    runs = Path(a.root) / 'comparison/standardized' / f'clinical_runs_nasim_s2_{a.date}'
    js = jobs(P, runs)
    if a.arms:
        want = [x for x in a.arms.split(',') if x]
        bad = [x for x in want if x not in ARMS]
        if bad:
            ap.error(f'unknown arm(s) {bad}; known: {list(ARMS)}')
        js = [j for j in js if j['arm'] in want]
    if a.plan:
        sys.exit(0 if plan(P, js) else 1)
    if a.jobs_json:
        print(json.dumps([j for j in js if not (Path(j['out']) / 'result.json').exists()], indent=1))
        return
    ap.error('give --plan or --jobs-json (this runner never trains by itself)')


if __name__ == '__main__':
    main()
