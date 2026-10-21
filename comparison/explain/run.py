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

--keep-going continues after a failing run (exit status 1 if any run failed). --resume skips planned runs whose folder
already holds `explanation_manifest.json` (recorded as `skipped_existing`), keeps the previous driver manifest as
`driver_manifest.<its started_at>.json` and refuses an interrupted run folder or any file outside the plan. A finished
folder is skipped only if it belongs to the current plan: its manifest and `binding.json` must name the same fold,
cohort sample size, method and run (`top<K>/<method>_seed<seed>`), and its recorded `code.source_hashes` must equal the
hashes of the code roots of this checkout (every `.py` under the training code roots, keyed `<root>/<relative path>`,
names starting `._` skipped); one difference refuses the resume with the folder's name. `--allow-mixed-code` skips only
the code comparison (the summary then needs its own `--allow-mixed-code`).
--jobs N runs up to N runner processes at once, each with OMP_NUM_THREADS and MKL_NUM_THREADS = max(1, cpus // N).
Every `driver_manifest.json` entry carries `seconds`, `started_at`, `status` (ok, failed, skipped_existing) and
`max_rss_mb` (peak memory of the child, null when unknown).

Before anything is printed or executed, every selected run's `binding.json` is checked: `protocol` is `fair_v10t`,
`top_k_labels` equals K, `final_eval` is `test`, `method` matches the run-dir prefix, `protocol_overrides` is an empty
list and `seed` is recorded and equals the planned seed. One mismatch stops the driver.

Layout of the outputs: `<out>/top<K>/<method>_seed<seed>/` holds the runner's files (`cohort.json`,
`subject_<sample_id>.json`, `explanation_manifest.json`, `binding.json`); `comparison.explain.summarize` reads it.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
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
RUN_MANIFEST = 'explanation_manifest.json'


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


def check_binding(run_dir, k, method, seed):
    """The run's binding.json must be a final fair_v10t run of this K, method and seed with no protocol override."""
    path = run_dir / 'binding.json'
    if not path.is_file():
        raise SystemExit(f'{shown(run_dir)}: binding.json missing')
    binding = json.loads(path.read_text())
    expected = {'protocol': PROTOCOL, 'top_k_labels': k, 'final_eval': FINAL_EVAL,
                'method': BINDING_METHOD[method], 'protocol_overrides': [], 'seed': seed}
    for key, want in expected.items():
        if key not in binding:
            raise SystemExit(f'{shown(run_dir)}: binding carries no {key}, expected {want!r}')
        if binding[key] != want:
            raise SystemExit(f'{shown(run_dir)}: binding {key} is {binding[key]!r}, expected {want!r}')
    return binding['seed']


def current_source_hashes():
    """sha256 of every `.py` under the training code roots of this checkout, keyed `<root>/<relative path>`.

    Names starting `._` (copy-tool leftovers) are skipped. The runner records the same mapping under
    `code.source_hashes` in its manifest. Loaded lazily: planning needs no hashing."""
    from core.paths import CODE_ROOTS
    from core.schema import sha256
    hashes = {}
    for root in CODE_ROOTS:
        base = REPO / root
        for path in sorted(base.rglob('*.py')):
            if '__pycache__' in path.parts or path.name.startswith('._'):
                continue
            hashes[f'{root}/{path.relative_to(base).as_posix()}'] = sha256(path)
    return hashes


def plan_runs(runs, ks, methods, seeds, fold, sample_patients, out):
    """One entry per selected run, all bindings checked; nothing is executed."""
    runs, out = Path(runs), Path(out)
    entries = []
    for k in ks:
        for method in methods:
            for seed in seeds:
                name = f'{method}_seed{seed}'
                run_dir = runs / f'top{k}' / name
                check_binding(run_dir, k, method, seed)
                target = out / f'top{k}' / name
                command = [sys.executable, '-m', 'core.explain.runner', '--run-dir', str(run_dir),
                           '--output', str(target), '--fold', fold, '--sample-patients', str(sample_patients)]
                entries.append({'k': k, 'method': method, 'seed': seed, 'run': f'top{k}/{name}',
                                'output': f'top{k}/{name}', 'command': command})
    return entries


def utc_now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def cpu_count():
    return os.cpu_count() or 1


def run_command(command, env):
    """Default executor: the runner as a child process; returns its exit status and peak memory.

    The child is reaped with `os.wait4` to read `ru_maxrss` (KiB on Linux, bytes on macOS)."""
    process = subprocess.Popen(command, env=env, cwd=REPO)
    _, status, usage = os.wait4(process.pid, 0)
    code = os.waitstatus_to_exitcode(status)
    process.returncode = code
    divisor = 1024 * 1024 if sys.platform == 'darwin' else 1024
    return {'returncode': code, 'max_rss_mb': round(usage.ru_maxrss / divisor, 1)}


def write_manifest(out, args, entries, results, complete, started_at=None):
    """`results` maps a run's output name to its record (returncode, seconds, started_at, status, max_rss_mb)."""
    runs = []
    for e in entries:
        r = results.get(e['output'], {})
        runs.append({'k': e['k'], 'method': e['method'], 'seed': e['seed'], 'output': e['output'],
                     'returncode': r.get('returncode'), 'status': r.get('status'), 'seconds': r.get('seconds'),
                     'started_at': r.get('started_at'), 'max_rss_mb': r.get('max_rss_mb')}
                    | ({k: r[k] for k in ('explained_count', 'failed_count') if k in r}))
    manifest = {
        'schema': 'medgnn.explanation_driver_manifest', 'protocol': PROTOCOL, 'fold': args.fold,
        'fold_role': FOLD_ROLE[args.fold], 'ks': args.ks, 'methods': args.methods, 'seeds': args.seeds,
        'sample_patients': args.sample_patients, 'planned_runs': len(entries), 'complete': complete,
        'started_at': started_at, 'runs': runs,
    }
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def check_finished(entry, folder, manifest, *, fold, sample_patients, hashes):
    """A finished folder must belong to the current plan: fold, sample size, method, run name and code.

    `hashes` is the code mapping of this checkout, or None to skip the code comparison."""
    name = shown(folder)
    binding_path = folder / 'binding.json'
    if not binding_path.is_file():
        raise SystemExit(f'--resume: {name} holds no binding.json; move it away and run again')
    binding = json.loads(binding_path.read_text())
    model_run = binding.get('model_run') or {}
    sampling = (binding.get('cohort') or {}).get('sampling') or {}
    run_name = '/'.join(Path(str(model_run.get('run_dir', ''))).parts[-2:])
    found = {'fold': manifest.get('fold'), 'fold_role': manifest.get('fold_role'),
             'sample size': sampling.get('sample_patients'), 'method': manifest.get('method'),
             'run': run_name, 'seed': model_run.get('seed'), 'top_k_labels': model_run.get('top_k_labels')}
    planned = {'fold': fold, 'fold_role': FOLD_ROLE[fold], 'sample size': sample_patients,
               'method': BINDING_METHOD[entry['method']], 'run': entry['run'], 'seed': entry['seed'],
               'top_k_labels': entry['k']}
    for key, want in planned.items():
        if found[key] != want:
            raise SystemExit(f'--resume: {name} was written for {key} {found[key]!r}, the plan says {want!r}; '
                             'move it away and run again')
    if hashes is not None:
        recorded = (manifest.get('code') or {}).get('source_hashes')
        if not isinstance(recorded, dict) or not recorded:
            raise SystemExit(f'--resume: {name} records no code hashes (code.source_hashes); '
                             'move it away and run again')
        if recorded != hashes:
            changed = sorted(key for key in set(recorded) | set(hashes) if recorded.get(key) != hashes.get(key))
            raise SystemExit(f'--resume: {name} was written by other code than this checkout '
                             f'({len(changed)} files differ, e.g. {", ".join(changed[:3])}); move it away and '
                             'run again, or pass --allow-mixed-code to accept it')


def check_resume_out(out, entries, *, fold, sample_patients, hashes):
    """Under --resume the output folder may be filled, but only by planned run folders and driver manifests.

    Returns the planned runs that are finished (hold `explanation_manifest.json`) and belong to the plan; an
    interrupted run folder or a finished one of another plan is refused with its name (the user moves it; nothing
    is deleted)."""
    out = Path(out)
    if out.exists() and not out.is_dir():
        raise SystemExit(f'--out must be a directory: {shown(out)}')
    planned = {e['output'] for e in entries}
    if out.is_dir():
        for item in sorted(out.iterdir()):
            if item.is_file() and (item.name == MANIFEST_NAME or
                                   (item.name.startswith('driver_manifest.') and item.name.endswith('.json'))):
                continue
            if item.is_dir() and item.name.startswith('top'):
                for child in sorted(item.iterdir()):
                    if f'{item.name}/{child.name}' not in planned or not child.is_dir():
                        raise SystemExit(f'--resume: {shown(child)} is outside the planned runs')
                continue
            raise SystemExit(f'--resume: {shown(item)} is outside the planned runs')
    finished = {}
    for e in entries:
        folder = out / e['output']
        if not folder.exists():
            continue
        if not (folder / RUN_MANIFEST).is_file():
            raise SystemExit(f'--resume: {shown(folder)} holds no {RUN_MANIFEST} (interrupted run); '
                             'move it away and run again')
        finished[e['output']] = json.loads((folder / RUN_MANIFEST).read_text())
        check_finished(e, folder, finished[e['output']], fold=fold, sample_patients=sample_patients, hashes=hashes)
    return finished


def keep_previous_manifest(out):
    """Rename the previous driver manifest to `driver_manifest.<started_at>.json`; returns its run records."""
    path = Path(out) / MANIFEST_NAME
    if not path.is_file():
        return {}
    previous = json.loads(path.read_text())
    stamp = previous.get('started_at')
    if not stamp:
        stamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    target = Path(out) / f'driver_manifest.{stamp.replace(":", "-")}.json'
    if target.exists():
        raise SystemExit(f'--resume: {shown(target)} already exists')
    path.rename(target)
    return {r.get('output'): r for r in previous.get('runs', [])}


def execute_one(entry, executor, env):
    """One run through the executor hook (int return code, or dict with returncode and max_rss_mb)."""
    started_at, t0 = utc_now(), time.monotonic()
    result = executor(entry['command'], env)
    seconds = round(time.monotonic() - t0, 1)
    if isinstance(result, dict):
        code, rss = result['returncode'], result.get('max_rss_mb')
    else:
        code, rss = result, None
    return {'returncode': code, 'seconds': seconds, 'started_at': started_at,
            'status': 'ok' if code == 0 else 'failed', 'max_rss_mb': rss}


def run_entries(todo, executor, env, jobs, keep_going, record):
    """Run `todo` with up to `jobs` at once; `record(entry, result)` is called under the caller's lock.

    Returns the first failing return code (None when nothing failed) and the number of failures."""
    first_failure, failures = None, 0

    def finish(entry, result):
        nonlocal first_failure, failures
        record(entry, result)
        if result['returncode'] != 0:
            failures += 1
            if first_failure is None:
                first_failure = result['returncode']
            print(f'run {entry["output"]} failed with exit status {result["returncode"]}'
                  + ('' if keep_going else '; stopped'), file=sys.stderr)

    if jobs <= 1:
        for entry in todo:
            finish(entry, execute_one(entry, executor, env))
            if first_failure is not None and not keep_going:
                break
        return first_failure, failures
    pending, queue = {}, list(todo)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        while queue or pending:
            while queue and len(pending) < jobs and (keep_going or first_failure is None):
                entry = queue.pop(0)
                pending[pool.submit(execute_one, entry, executor, env)] = entry
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                finish(pending.pop(future), future.result())
    return first_failure, failures


def main(argv=None, *, executor=run_command):
    """CLI entry; `executor(command, env)` is the injectable runner call.

    It returns the exit status, or a dict `{returncode, max_rss_mb}`; with `--jobs` above 1 it is called from worker
    threads."""
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
    parser.add_argument('--keep-going', action='store_true', help='continue after a failing run; exit 1 if any failed')
    parser.add_argument('--resume', action='store_true',
                        help='skip planned runs that already hold an explanation manifest')
    parser.add_argument('--jobs', type=int, default=1, help='runner processes at once (default 1)')
    parser.add_argument('--allow-mixed-code', action='store_true',
                        help='with --resume: accept finished runs written by other code than this checkout')
    args = parser.parse_args(argv)
    args.ks = parse_list(args.ks, int, 'ks')
    args.methods = parse_list(args.methods, str, 'methods')
    args.seeds = parse_list(args.seeds, int, 'seeds')
    if args.jobs < 1:
        raise SystemExit('--jobs: must be at least 1')
    check_methods(args.methods)
    if not args.resume:
        check_out(args.out)
    runs = Path(args.runs)
    if not runs.is_absolute():
        runs = REPO / runs
    entries = plan_runs(runs, args.ks, args.methods, args.seeds, args.fold, args.sample_patients, args.out)
    finished = {}
    if args.resume:
        hashes = None if args.allow_mixed_code else current_source_hashes()
        finished = check_resume_out(args.out, entries, fold=args.fold, sample_patients=args.sample_patients,
                                    hashes=hashes)

    if args.plan or not args.execute:
        print(f'fold {args.fold} ({FOLD_ROLE[args.fold]} fold), protocol {PROTOCOL}, {len(entries)} runs')
        for entry in entries:
            print(' '.join(entry['command']))
            print(f'  -> {entry["output"]}' + ('  (finished, skipped on --resume)' if entry['output'] in finished else ''))
        return 0 if args.plan else 2

    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    if args.jobs > 1:
        threads = str(max(1, cpu_count() // args.jobs))
        env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
    started_at = utc_now()
    previous = keep_previous_manifest(args.out) if args.resume else {}
    results = {}
    for entry in entries:
        if entry['output'] in finished:
            old = previous.get(entry['output'], {})
            results[entry['output']] = {
                'returncode': 0, 'status': 'skipped_existing', 'seconds': old.get('seconds'),
                'started_at': old.get('started_at'), 'max_rss_mb': old.get('max_rss_mb'),
                'explained_count': finished[entry['output']].get('explained_count'),
                'failed_count': finished[entry['output']].get('failed_count')}
    todo = [e for e in entries if e['output'] not in finished]
    lock = threading.Lock()

    def record(entry, result):
        with lock:
            results[entry['output']] = result
            write_manifest(args.out, args, entries, results, complete=False, started_at=started_at)

    write_manifest(args.out, args, entries, results, complete=False, started_at=started_at)
    first_failure, failures = run_entries(todo, executor, env, args.jobs, args.keep_going, record)
    complete = failures == 0 and all(results.get(e['output']) for e in entries)
    write_manifest(args.out, args, entries, results, complete=complete, started_at=started_at)
    if failures:
        if args.keep_going:
            print(f'{failures} of {len(todo)} runs failed; manifest {MANIFEST_NAME}', file=sys.stderr)
            return 1
        return first_failure if first_failure > 0 else 1
    print(f'{len(entries)} runs explained ({len(finished)} skipped); manifest {MANIFEST_NAME}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
