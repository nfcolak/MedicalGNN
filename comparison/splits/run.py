"""Split-replication runs: a1_full, a5_nomp, a5b_nomp_matched, a6_pool x seeds on a new patient split (K=20).

Arm options, recipe and a6 hyper-parameters are imported from comparison/ablation/run.py (nothing is copied); only the
inputs change: the split's own graph, targets, v3 state and split file. Unlike the ablation runner, a1_full is trained for
all five seeds (the 45 final runs belong to the canonical split).
Usage (from the repo root):
    python3 -m comparison.splits.run --splits r1,r2 [--plan] [--seeds 7,42,1234,2025,3407] [--arms a1_full,a6_pool]
        --plan is the default: no --execute, prints one line per run (resolved protocol, overrides, epochs) and lists the
        inputs that do not exist yet (they exist only after comparison/splits/pipeline.sh has run for that split).
    python3 -m comparison.splits.run --splits r1,r2 --execute [...]      real training; needs the user's approval
Inputs of split <id> with tag <id>_<inputs-date> (as written by comparison/splits/pipeline.sh; --split-file/--graph/--targets/--v3-state
override them for one split), all under <root>/comparison/standardized/event_inputs/:
    split_<tag>/split.json, clinical_graph_L4_plus60_v5c_<tag>, targets_primary_v4_v5plus60_v5_test_<tag>/targets.csv,
    v3_state_L4_v5c_v10t_top20_<tag>/K4.json
Output: <root>/comparison/standardized/clinical_runs_fair_v10t_splits_<date>/<id>/top20/<arm>_seed<S>/ with the usual run files
plus split_binding.json (split id, split file path and sha256, graph/targets/v3 state paths and sha256; written after the run,
checked against the hashes taken before it). A non-empty output directory is refused, never resumed or overwritten.
"""
import argparse, hashlib, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from comparison.ablation import run as ablation

REPO = ablation.REPO
K = 20
ARMS = ('a1_full', 'a5_nomp', 'a5b_nomp_matched', 'a6_pool')
SEEDS = (7, 42, 1234, 2025, 3407)


class Paths:
    """Inputs of one split (ablation.cmd and ablation.method_options read .graph, .targets and .v3_state(K))."""
    def __init__(self, root, split_id, inputs_date, date, split_file=None, graph=None, targets=None, v3_state=None):
        std = Path(root) / 'comparison/standardized'
        ei, tag = std / 'event_inputs', f'{split_id}_{inputs_date}'
        self.split_id = split_id
        self.split_file = Path(split_file) if split_file else ei / f'split_{tag}/split.json'
        self.graph = Path(graph) if graph else ei / f'clinical_graph_L4_plus60_v5c_{tag}'
        self.targets = Path(targets) if targets else ei / f'targets_primary_v4_v5plus60_v5_test_{tag}/targets.csv'
        self._v3_state = Path(v3_state) if v3_state else ei / f'v3_state_L4_v5c_v10t_top20_{tag}/K4.json'
        self.runs = std / f'clinical_runs_fair_v10t_splits_{date}' / split_id

    def v3_state(self, k):
        if k != K:
            raise ValueError(f'splits runs are K={K} only, got {k}')
        return self._v3_state

    def missing(self):
        return [f'{n} {p}' for n, p in (('split-file', self.split_file), ('graph', self.graph / 'manifest.json'),
                                         ('targets', self.targets), ('v3-state', self._v3_state)) if not p.is_file()]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def input_hashes(P):
    return {'split_file': str(P.split_file), 'split_file_sha256': sha256(P.split_file),
            'graph': str(P.graph), 'graph_manifest_sha256': sha256(P.graph / 'manifest.json'),
            'graph_cohort_sha256': sha256(P.graph / 'cohort.csv'),
            'targets': str(P.targets), 'targets_sha256': sha256(P.targets),
            'v3_state': str(P._v3_state), 'v3_state_sha256': sha256(P._v3_state)}


def cmd(P, arm, seed, out, execute, threads=ablation.THREADS):
    """The ablation command of the arm plus --canonical <this split's file> (class order check only; folds come from the targets)."""
    c = ablation.cmd(P, arm, K, seed, out, execute, threads)
    at = len(c) - 1 if execute else len(c)   # `--execute` stays last
    return c[:at] + ['--canonical', str(P.split_file)] + c[at:]


def plan_one(P, arm, seed):
    out = P.runs / f'top{K}' / f'{arm}_seed{seed}'
    tag = f'{P.split_id} {K} {arm.ljust(18)} seed{str(seed).ljust(5)}'
    if ablation.refused(out):
        return f'{tag} REFUSED output dir is filled: {out}'
    if P.missing():
        return f'{tag} INPUTS MISSING (run comparison/splits/pipeline.sh first) -> {out}'
    o = subprocess.run(cmd(P, arm, seed, out, False), cwd=REPO, capture_output=True, text=True)
    t = o.stdout
    try:
        p = json.loads(t[t.index('{'):])
        pr = p['protocol_resolved']
        return (f"{tag} {p.get('status')} {p.get('protocol')} overrides {p.get('protocol_overrides')} epochs {pr['epochs']} "
                f"topk {pr['top_k_labels']} selection {pr['selection_fold']} final {pr['final_eval']}")
    except Exception:
        return f'{tag} ERROR {(o.stdout + o.stderr).strip()[-400:]}'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', action='store_true', help='print the resolved protocol of every run and exit (default)')
    ap.add_argument('--execute', action='store_true', help='really train (needs the user\'s approval)')
    ap.add_argument('--splits', required=True, help='comma-separated split ids, e.g. r1,r2')
    ap.add_argument('--seeds', default=','.join(map(str, SEEDS)), help='comma-separated seeds (default 7,42,1234,2025,3407)')
    ap.add_argument('--arms', default=','.join(ARMS), help='comma-separated arm ids (default all): ' + ', '.join(ARMS))
    ap.add_argument('--root', default=os.environ.get('MEDGNN_DATA_ROOT', str(REPO)))
    ap.add_argument('--date', default=time.strftime('%Y%m%d'), help='run-folder date suffix (default today)')
    ap.add_argument('--inputs-date', default='20261018', help='date suffix of the split input folders (default 20261018)')
    ap.add_argument('--split-file'), ap.add_argument('--graph'), ap.add_argument('--targets'), ap.add_argument('--v3-state')
    ap.add_argument('--workers', type=int, default=ablation.WORKERS)
    ap.add_argument('--threads', type=int, default=ablation.THREADS)
    a = ap.parse_args(argv)
    if a.plan and a.execute:
        ap.error('--plan and --execute exclude each other')
    splits = tuple(x.strip() for x in a.splits.split(','))
    seeds = tuple(int(s) for s in a.seeds.split(','))
    arms = tuple(x.strip() for x in a.arms.split(','))
    if bad := [x for x in arms if x not in ARMS]:
        ap.error(f'unknown arm(s) {bad}; known: {list(ARMS)}')
    if len(set(seeds)) != len(seeds) or len(set(splits)) != len(splits):
        ap.error('duplicate seeds or splits')
    if 'r0' in splits:
        ap.error('r0 is the canonical split: its runs already exist (comparison.ablation.run, final runs); not rerun here')
    if any([a.split_file, a.graph, a.targets, a.v3_state]) and len(splits) != 1:
        ap.error('--split-file/--graph/--targets/--v3-state override the paths of one split: pass exactly one --splits id')
    if 'a6_pool' in arms and ablation.A6_HPARAMS is None:
        ap.error('a6_pool needs its own HPO first (A6_HPARAMS is None)')
    Ps = {s: Paths(a.root, s, a.inputs_date, a.date, a.split_file, a.graph, a.targets, a.v3_state) for s in splits}
    # seed-major order inside each split: every arm starts in the first waves, so an arm that fails shows up early
    jobs = [(s, arm, seed) for s in splits for seed in seeds for arm in arms]
    if not a.execute:
        with ThreadPoolExecutor(max(1, min(a.workers, 6))) as ex:   # plan lines are independent short subprocesses
            lines = list(ex.map(lambda job: plan_one(Ps[job[0]], job[1], job[2]), jobs))
        print('\n'.join(lines))
        for s, P in Ps.items():
            for m in P.missing():
                print(f'{s} MISSING INPUT {m}')
            print(f'{s} output root {P.runs}')
        print(f'{len(lines)} runs planned (nothing executed)')
        return 0 if all(' ERROR ' not in x and ' REFUSED ' not in x for x in lines) else 1
    if miss := [f'{s} {m}' for s, P in Ps.items() for m in P.missing()]:
        print('REFUSED: inputs missing:\n' + '\n'.join(miss), file=sys.stderr)
        return 1
    if bad := [str(P.runs / f'top{K}' / f'{arm}_seed{seed}') for s, arm, seed in jobs
               for P in [Ps[s]] if ablation.refused(P.runs / f'top{K}' / f'{arm}_seed{seed}')]:
        print('REFUSED: output dir(s) already filled (use a new --date):\n' + '\n'.join(bad), file=sys.stderr)
        return 1
    before = {s: input_hashes(P) for s, P in Ps.items()}

    def log(P, msg):
        P.runs.mkdir(parents=True, exist_ok=True)
        with open(P.runs / 'progress.txt', 'a') as f:
            f.write(time.strftime('%F %T ') + msg + '\n')

    def run(job):
        s, arm, seed = job
        P = Ps[s]
        out = P.runs / f'top{K}' / f'{arm}_seed{seed}'
        out.parent.mkdir(parents=True, exist_ok=True)
        (P.runs / 'logs').mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(P.runs / 'logs' / f'top{K}_{arm}_seed{seed}.log', 'w') as f:
            rc = subprocess.run(cmd(P, arm, seed, out, True, a.threads), cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode
        ok = rc == 0 and (out / 'result.json').is_file()   # a run counts only with its result.json
        if ok:
            after = input_hashes(P)
            ok = after == before[s]   # an input that changed during the run voids it
            (out / 'split_binding.json').write_text(json.dumps(
                {'split_id': s, 'arm': arm, 'seed': seed, 'top_k_labels': K, 'inputs_unchanged_during_run': ok, **after}, indent=2) + '\n')
        log(P, f'{"DONE" if ok else "FAIL"} {s} top{K} {arm} seed{seed} rc={rc} {round((time.time() - t0) / 60, 1)} min')
        return ok

    head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    for P in Ps.values():
        log(P, f'START {len(jobs)} runs (all splits) splits={splits} seeds={seeds} arms={arms}, {a.workers} workers x {a.threads} threads, HEAD {head}')
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(run, jobs))
    failed = [f'{s} top{K} {arm} seed{seed}' for (s, arm, seed), ok in zip(jobs, results) if not ok]
    for P in Ps.values():
        log(P, f'ALL RUNS FINISHED, {len(failed)} FAILED' + (': ' + ', '.join(failed) if failed else ''))
    if failed:   # non-zero exit: a failed run must never pass silently into the analysis
        print(f'{len(failed)} run(s) failed: ' + ', '.join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
