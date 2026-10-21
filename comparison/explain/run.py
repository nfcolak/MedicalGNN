"""Explanation driver: one runner call per stored final run, planned first, executed only on request.

Walks the final runs (`<runs>/top<K>/<method>_seed<seed>/`) and builds, for each selected run, the command of the
explanation runner (`core.explain.runner`) that replays the stored predictions of one fold and explains the sampled
visits. Nothing is trained and no fold is fitted or selected on: the explanations are post hoc on the frozen
checkpoints. The default fold is test (the reporting fold, `fold_role` report); `--fold validation` is the labelled
decision fold.

    python -m comparison.explain.run [--runs <runs root>] [--ks 10,20,30] [--methods gsat,graphcare,protgnn,cei_mp2] \
        [--seeds 1234,2025,7] [--fold test|validation] [--sample-patients 500] --out <new dir> (--plan | --execute)

Without --plan or --execute the plan is printed and the exit status is 2. --plan prints one runner command per run
and its expected output directory and executes nothing. --execute runs the commands one after another, stops with a
non-zero exit status on the first failing run and writes `<out>/driver_manifest.json` (also after a failure, with
`complete` false). --out must be a new or empty directory; a filled one is refused.

Before anything is printed or executed, every selected run's `binding.json` is checked: `protocol` is `fair_v10t`,
`top_k_labels` equals K, `final_eval` is `test` and `method` matches the run-dir prefix. One mismatch stops the driver.

Layout of the outputs: `<out>/top<K>/<method>_seed<seed>/` holds the runner's files (`cohort.json`,
`subject_<sample_id>.json`, `explanation_manifest.json`, `binding.json`); `comparison.explain.summarize` reads it.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = 'comparison/standardized/clinical_runs_fair_v10t_20261011'
PROTOCOL = 'fair_v10t'
FINAL_EVAL = 'test'
DEFAULT_KS = (10, 20, 30)
DEFAULT_SEEDS = (1234, 2025, 7)
DEFAULT_METHODS = ('gsat', 'graphcare', 'protgnn', 'cei_mp2')
BINDING_METHOD = {'gsat': 'gsat', 'graphcare': 'graphcare', 'protgnn': 'protgnn', 'cei_mp2': 'cei_gnn_mp2'}
FOLDS = ('test', 'validation')
FOLD_ROLE = {'test': 'report', 'validation': 'decision'}
DROPPED = {'gchm_pna': 'GCHM-PNA is not part of the explanation study'}
MANIFEST_NAME = 'driver_manifest.json'


def shown(path):
    """Repo-relative text for a path; outside the repo only the final name (never an absolute path)."""
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return p.name


def parse_list(text, cast, what):
    try:
        values = [cast(part) for part in text.split(',') if part.strip()]
    except ValueError:
        raise SystemExit(f'--{what}: cannot read {text!r}')
    if not values:
        raise SystemExit(f'--{what}: empty list')
    return values


def check_methods(methods):
    for method in methods:
        if method in DROPPED:
            raise SystemExit(f'--methods: {method}: {DROPPED[method]}')
        if method not in BINDING_METHOD:
            raise SystemExit(f'--methods: unknown method {method!r}; choose from {", ".join(DEFAULT_METHODS)}')


def check_out(out):
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise SystemExit(f'--out must be a new or empty directory: {shown(out)}')


def check_binding(run_dir, k, method):
    """The run's binding.json must be a final fair_v10t run of this K and method; returns its seed."""
    path = run_dir / 'binding.json'
    if not path.is_file():
        raise SystemExit(f'{shown(run_dir)}: binding.json missing')
    binding = json.loads(path.read_text())
    expected = {'protocol': PROTOCOL, 'top_k_labels': k, 'final_eval': FINAL_EVAL,
                'method': BINDING_METHOD[method]}
    for key, want in expected.items():
        if binding.get(key) != want:
            raise SystemExit(f'{shown(run_dir)}: binding {key} is {binding.get(key)!r}, expected {want!r}')
    return binding.get('seed')


def plan_runs(runs, ks, methods, seeds, fold, sample_patients, out):
    """One entry per selected run, all bindings checked; nothing is executed."""
    runs, out = Path(runs), Path(out)
    entries = []
    for k in ks:
        for method in methods:
            for seed in seeds:
                name = f'{method}_seed{seed}'
                run_dir = runs / f'top{k}' / name
                seed_bound = check_binding(run_dir, k, method)
                if seed_bound is not None and seed_bound != seed:
                    raise SystemExit(f'{shown(run_dir)}: binding seed is {seed_bound!r}, expected {seed}')
                target = out / f'top{k}' / name
                command = [sys.executable, '-m', 'core.explain.runner', '--run-dir', str(run_dir),
                           '--output', str(target), '--fold', fold, '--sample-patients', str(sample_patients)]
                entries.append({'k': k, 'method': method, 'seed': seed, 'run': f'top{k}/{name}',
                                'output': f'top{k}/{name}', 'command': command})
    return entries


def run_command(command, env):
    """Default executor: the runner as a child process; returns its exit status."""
    return subprocess.run(command, env=env, cwd=REPO).returncode


def write_manifest(out, args, entries, results, complete):
    manifest = {
        'schema': 'medgnn.explanation_driver_manifest', 'protocol': PROTOCOL, 'fold': args.fold,
        'fold_role': FOLD_ROLE[args.fold], 'ks': args.ks, 'methods': args.methods, 'seeds': args.seeds,
        'sample_patients': args.sample_patients, 'planned_runs': len(entries), 'complete': complete,
        'runs': [{'k': e['k'], 'method': e['method'], 'seed': e['seed'], 'output': e['output'],
                  'returncode': results.get(e['output'])} for e in entries],
    }
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main(argv=None, *, executor=run_command):
    """CLI entry; `executor(command, env) -> returncode` is the injectable runner call."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', default=DEFAULT_RUNS)
    parser.add_argument('--ks', default=','.join(map(str, DEFAULT_KS)))
    parser.add_argument('--methods', default=','.join(DEFAULT_METHODS))
    parser.add_argument('--seeds', default=','.join(map(str, DEFAULT_SEEDS)))
    parser.add_argument('--fold', choices=FOLDS, default='test')
    parser.add_argument('--sample-patients', type=int, default=500)
    parser.add_argument('--out', required=True, help='a new, empty directory')
    parser.add_argument('--plan', action='store_true', help='print the runner commands, execute nothing')
    parser.add_argument('--execute', action='store_true', help='run the commands one after another')
    args = parser.parse_args(argv)
    args.ks = parse_list(args.ks, int, 'ks')
    args.methods = parse_list(args.methods, str, 'methods')
    args.seeds = parse_list(args.seeds, int, 'seeds')
    check_methods(args.methods)
    check_out(args.out)
    runs = Path(args.runs)
    if not runs.is_absolute():
        runs = REPO / runs
    entries = plan_runs(runs, args.ks, args.methods, args.seeds, args.fold, args.sample_patients, args.out)

    if args.plan or not args.execute:
        print(f'fold {args.fold} ({FOLD_ROLE[args.fold]} fold), protocol {PROTOCOL}, {len(entries)} runs')
        for entry in entries:
            print(' '.join(entry['command']))
            print(f'  -> {entry["output"]}')
        return 0 if args.plan else 2

    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    results = {}
    for entry in entries:
        code = executor(entry['command'], env)
        results[entry['output']] = code
        if code != 0:
            write_manifest(args.out, args, entries, results, complete=False)
            print(f'run {entry["output"]} failed with exit status {code}; stopped', file=sys.stderr)
            return code if code > 0 else 1
    write_manifest(args.out, args, entries, results, complete=True)
    print(f'{len(entries)} runs explained; manifest {MANIFEST_NAME}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
