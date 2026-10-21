"""Aggregate explanation tables: mean and SD over seeds per (K, method, importance source).

Reads the finished explanation outputs of `comparison.explain.run` (`<explanations>/top<K>/<method>_seed<seed>/`
with `explanation_manifest.json` and one record file per explained visit) and writes aggregates only. Inference-free:
nothing is loaded except those JSON files, nothing is trained, no checkpoint is touched.

    python -m comparison.explain.summarize --explanations <driver out> --out <new dir>

Outputs (`--out` must be new or empty): `summary.csv`, `summary.json`, `summary.md`.

One row per (K, method, importance source); the sources are the GraphXAI algorithms of the records plus the model's
built-in explanation (`builtin`). Per run the metrics are first averaged over its explained visits, then the run means
are averaged over the seeds (mean and sample SD; SD is empty with a single run). Metrics: fidelity+ and fidelity- (drop
of the predicted-class probability when the top 20% of evidence nodes are removed, respectively only they are kept),
deletion and insertion AUC, mass-90 sparsity, and for each of the four fidelity numbers the difference to the random
removal reference of the same graph and the fraction of random draws it beats. Also per row: runs, explained visits,
failed visits (runner failures) and failed source evaluations.

Every table carries its fold: `test (report fold)` for the reporting fold, `validation (decision fold)` for the
labelled decision fold. The summarizer refuses outputs that mix folds, fold roles or protocols, and writes no sample,
subject or stay identifiers and no absolute paths.
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

FOLD_LABEL = {'test': 'test (report fold)', 'validation': 'validation (decision fold)',
              'dev': 'dev (decision fold)'}
FOLD_ROLE = {'test': 'report', 'validation': 'decision', 'dev': 'decision'}
RUN_NAME = re.compile(r'^(?P<method>.+)_seed(?P<seed>\d+)$')
TOP_NAME = re.compile(r'^top(?P<k>\d+)$')
RANDOM_KEYS = ('fidelity_plus', 'fidelity_minus', 'deletion_auc', 'insertion_auc')
METRICS = ('fidelity_plus', 'fidelity_minus', 'deletion_auc', 'insertion_auc', 'sparsity') \
    + tuple(f'{key}_vs_random' for key in RANDOM_KEYS) + tuple(f'{key}_beats_random' for key in RANDOM_KEYS)
MD_COLUMNS = ('fidelity_plus', 'fidelity_minus', 'deletion_auc', 'insertion_auc', 'sparsity',
              'deletion_auc_vs_random', 'insertion_auc_vs_random')


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def source_metrics(entry):
    """The scalar metrics of one source in one record, or None when that evaluation failed."""
    if not isinstance(entry, dict) or entry.get('status') == 'failed':
        return None
    try:
        out = {'fidelity_plus': float(entry['fidelity_plus']['prob']),
               'fidelity_minus': float(entry['fidelity_minus']['prob']),
               'deletion_auc': float(entry['deletion_curve']['auc']),
               'insertion_auc': float(entry['insertion_curve']['auc']),
               'sparsity': float(entry['sparsity']['mass90_sparsity'])}
    except (KeyError, TypeError):
        return None
    vs_random = entry.get('vs_random') or {}
    for key in RANDOM_KEYS:
        part = vs_random.get(key)
        if isinstance(part, dict):
            out[f'{key}_vs_random'] = float(part['difference'])
            out[f'{key}_beats_random'] = float(part['beats_random_fraction'])
    return out


def load_run(run_dir):
    """Manifest facts and the per-source metric lists of one run; ids are used to open files, never kept."""
    manifest = json.loads((run_dir / 'explanation_manifest.json').read_text())
    binding_path = run_dir / 'binding.json'
    binding = json.loads(binding_path.read_text()) if binding_path.is_file() else {}
    fold = manifest.get('fold')
    if fold not in FOLD_LABEL:
        raise SystemExit(f'{run_dir.name}: unknown fold {fold!r}')
    role = manifest.get('fold_role', FOLD_ROLE[fold])
    if role != FOLD_ROLE[fold]:
        raise SystemExit(f'{run_dir.name}: fold_role {role!r} does not belong to fold {fold!r}')
    sources, source_failures = {}, {}
    for filename in manifest.get('record_files', []):
        record = json.loads((run_dir / filename).read_text())
        for name, entry in (record.get('fidelity_v2') or {}).items():
            metrics = source_metrics(entry)
            if metrics is None:
                source_failures[name] = source_failures.get(name, 0) + 1
                sources.setdefault(name, [])
            else:
                sources.setdefault(name, []).append(metrics)
    return {'method': manifest['method'], 'fold': fold, 'role': role,
            'final_eval': (binding.get('model_run') or {}).get('final_eval'),
            'visits': int(manifest.get('explained_count', len(manifest.get('record_files', [])))),
            'failed': int(manifest.get('failed_count', 0)),
            'sources': sources, 'source_failures': source_failures}


def collect(root):
    root = Path(root)
    protocol, driver_fold = None, None
    driver = root / 'driver_manifest.json'
    if driver.is_file():
        header = json.loads(driver.read_text())
        protocol, driver_fold = header.get('protocol'), header.get('fold')
    runs = []
    for top in sorted(root.iterdir()):
        match = TOP_NAME.match(top.name)
        if not (top.is_dir() and match):
            continue
        for run_dir in sorted(top.iterdir()):
            name = RUN_NAME.match(run_dir.name)
            if name and (run_dir / 'explanation_manifest.json').is_file():
                run = load_run(run_dir)
                run.update(k=int(match['k']), seed=int(name['seed']))
                runs.append(run)
    if not runs:
        raise SystemExit('no explanation_manifest.json found under top<K>/<method>_seed<seed>')
    for key, what in (('fold', 'folds'), ('role', 'fold roles'), ('final_eval', 'final_eval values')):
        found = sorted({str(run[key]) for run in runs})
        if len(found) > 1:
            raise SystemExit(f'refusing to mix {what} in one table: {", ".join(found)}')
    if driver_fold is not None and driver_fold != runs[0]['fold']:
        raise SystemExit(f'refusing to mix folds in one table: driver {driver_fold}, runs {runs[0]["fold"]}')
    return runs, protocol


def aggregate(runs):
    """Rows keyed by (K, method, source); mean over visits per run, then mean and SD over the run means."""
    groups = {}
    for run in runs:
        for source, per_visit in run['sources'].items():
            groups.setdefault((run['k'], run['method'], source), []).append((run, per_visit))
    rows = []
    for (k, method, source), members in sorted(groups.items()):
        row = {'k': k, 'method': method, 'source': source, 'runs': len(members),
               'visits': sum(len(per_visit) for _, per_visit in members),
               'failed_visits': sum(run['failed'] for run, _ in members),
               'failed_sources': sum(run['source_failures'].get(source, 0) for run, _ in members)}
        for metric in METRICS:
            run_means = []
            for _, per_visit in members:
                values = [m[metric] for m in per_visit if metric in m]
                if values:
                    run_means.append(sum(values) / len(values))
            row[f'{metric}_mean'], row[f'{metric}_sd'] = mean_sd(run_means)
        rows.append(row)
    return rows


def fmt(value, digits=4):
    return '' if value is None else f'{value:.{digits}f}'


def pm(row, metric):
    mean, sd = row[f'{metric}_mean'], row[f'{metric}_sd']
    if mean is None:
        return ''
    return fmt(mean) if sd is None else f'{fmt(mean)} ± {fmt(sd)}'


def write_outputs(out, rows, fold, role, protocol):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    label = FOLD_LABEL[fold]
    columns = ['fold', 'k', 'method', 'source', 'runs', 'visits', 'failed_visits', 'failed_sources'] \
        + [f'{m}_{part}' for m in METRICS for part in ('mean', 'sd')]
    with open(out / 'summary.csv', 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([label if c == 'fold' else ('' if row[c] is None else row[c]) for c in columns])
    payload = {'schema': 'medgnn.explanation_summary', 'fold': fold, 'fold_label': label, 'fold_role': role,
               'protocol': protocol or 'not_recorded', 'aggregation': 'mean over visits per run, then mean and '
               'sample SD over seeds', 'rows': rows}
    (out / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    lines = [f'# Explanation summary, fold: {label}', '',
             f'Protocol: {payload["protocol"]}. Mean ± SD over seeds of the per-run visit means; '
             'fidelity numbers are drops of the predicted-class probability. No identifiers.', '']
    for k in sorted({r['k'] for r in rows}):
        lines += [f'## K = {k} · fold: {label}', '',
                  '| method | source | runs | visits | failed | ' + ' | '.join(MD_COLUMNS) + ' |',
                  '|---|---|---|---|---|' + '---|' * len(MD_COLUMNS)]
        for row in (r for r in rows if r['k'] == k):
            lines.append(f'| {row["method"]} | {row["source"]} | {row["runs"]} | {row["visits"]} | '
                         f'{row["failed_visits"]}+{row["failed_sources"]} | '
                         + ' | '.join(pm(row, m) for m in MD_COLUMNS) + ' |')
        lines.append('')
    (out / 'summary.md').write_text('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--explanations', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path, help='a new, empty directory')
    args = parser.parse_args(argv)
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        raise SystemExit(f'--out must be a new or empty directory: {args.out.name}')
    runs, protocol = collect(args.explanations)
    rows = aggregate(runs)
    write_outputs(args.out, rows, runs[0]['fold'], runs[0]['role'], protocol)
    print(f'{len(runs)} runs, {len(rows)} rows, fold {FOLD_LABEL[runs[0]["fold"]]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
