"""Aggregate explanation tables: mean and SD over seeds per (K, method, importance source).

Reads the finished explanation outputs of `comparison.explain.run` (`<explanations>/top<K>/<method>_seed<seed>/`
with `explanation_manifest.json` and one record file per explained visit) and writes aggregates only. Inference-free:
nothing is loaded except those JSON files, nothing is trained, no checkpoint is touched.

    python -m comparison.explain.summarize --explanations <driver out> --out <new dir> [--allow-incomplete]

Outputs (`--out` must be new or empty): `summary.csv`, `summary.json`, `summary.md`.

One row per (K, method, importance source); the sources are the GraphXAI algorithms of the records plus the model's
built-in explanation (`builtin`). Per run the metrics are first averaged over its explained visits, then the run means
are averaged over the seeds (mean and sample SD; SD is empty with a single run). Metrics: fidelity+ and fidelity- (drop
of the predicted-class probability when the top 20% of evidence nodes are removed, respectively only they are kept),
for both curves the raw AUC and the width-normalised `mean_change` (AUC divided by the curve's explained fraction, which
is comparable between a sparse and a dense explanation), the explained fraction, mass-90 sparsity, and for each fidelity
and curve number the difference to the random removal reference of the same graph and the fraction of random draws it
beats. Also per row: runs, explained visits, failed visits (runner failures) and failed source evaluations.
`HEADLINE_CURVE` only sets which curve measure comes first in the markdown columns; both are always written.

Direction: higher is better for fidelity+ and the deletion curve; lower is better for fidelity- and the insertion curve;
a negative insertion difference to random is better than random.

Every table carries its fold: `test (report fold)` for the reporting fold, `validation (decision fold)` for the
labelled decision fold, and the intervention contract version of its records. The summarizer refuses outputs that mix
folds, fold roles, protocols or contract versions, records or bindings without a version, and a driver run that is not
complete (a run without return code 0) unless `--allow-incomplete` is given; then the runs left out are listed in a
"runs not summarised" table (K, method, seed, return code). It writes no sample, subject or stay identifiers and no
absolute paths.
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

HEADLINE_CURVE = 'auc'    # 'auc' or 'mean_change': which curve measure leads the markdown columns
FOLD_LABEL = {'test': 'test (report fold)', 'validation': 'validation (decision fold)',
              'dev': 'dev (decision fold)'}
FOLD_ROLE = {'test': 'report', 'validation': 'decision', 'dev': 'decision'}
RUN_NAME = re.compile(r'^(?P<method>.+)_seed(?P<seed>\d+)$')
TOP_NAME = re.compile(r'^top(?P<k>\d+)$')
CURVE_MEASURES = ('deletion_auc', 'insertion_auc', 'deletion_mean_change', 'insertion_mean_change')
RANDOM_KEYS = ('fidelity_plus', 'fidelity_minus') + CURVE_MEASURES
METRICS = ('fidelity_plus', 'fidelity_minus') + CURVE_MEASURES + ('explained_fraction', 'sparsity') \
    + tuple(f'{key}_vs_random' for key in RANDOM_KEYS) + tuple(f'{key}_beats_random' for key in RANDOM_KEYS)
DIRECTION_NOTE = ('Direction: higher is better for fidelity_plus and the deletion columns; lower is better for '
                  'fidelity_minus and the insertion columns; a negative insertion difference to random '
                  '(`*_vs_random`) is better than random, a positive deletion difference is.')


def md_columns(headline=None):
    """Markdown columns; the headline curve measure leads, the other follows, both always shown."""
    headline = headline or HEADLINE_CURVE
    if headline not in ('auc', 'mean_change'):
        raise ValueError(f'HEADLINE_CURVE must be "auc" or "mean_change", not {headline!r}')
    other = 'mean_change' if headline == 'auc' else 'auc'
    curves = [f'{kind}_{measure}' for measure in (headline, other) for kind in ('deletion', 'insertion')]
    return ['fidelity_plus', 'fidelity_minus'] + curves + ['explained_fraction', 'sparsity'] \
        + [f'{c}_vs_random' for c in curves]


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
        deletion, insertion = entry['deletion_curve'], entry['insertion_curve']
        out = {'fidelity_plus': float(entry['fidelity_plus']['prob']),
               'fidelity_minus': float(entry['fidelity_minus']['prob']),
               'deletion_auc': float(deletion['auc']),
               'insertion_auc': float(insertion['auc']),
               'sparsity': float(entry['sparsity']['mass90_sparsity'])}
    except (KeyError, TypeError):
        return None
    for name, curve in (('deletion_mean_change', deletion), ('insertion_mean_change', insertion)):
        if curve.get('mean_change') is not None:
            out[name] = float(curve['mean_change'])
    if deletion.get('explained_fraction') is not None:
        out['explained_fraction'] = float(deletion['explained_fraction'])
    vs_random = entry.get('vs_random') or {}
    for key in RANDOM_KEYS:
        part = vs_random.get(key)
        if isinstance(part, dict):
            out[f'{key}_vs_random'] = float(part['difference'])
            out[f'{key}_beats_random'] = float(part['beats_random_fraction'])
    return out


def contract_version(run_name, record_versions, binding):
    """The run's intervention contract version from its records and its binding; refuses a missing
    version or a disagreement between records, or between the records and the binding."""
    if None in record_versions:
        raise SystemExit(f'{run_name}: a record carries no intervention contract version '
                         '(fidelity_v2_intervention.version)')
    if len(record_versions) > 1:
        raise SystemExit(f'{run_name}: records disagree on the intervention contract version: '
                         + ', '.join(str(v) for v in sorted(record_versions)))
    bound = ((binding or {}).get('explainer_config') or {}).get('intervention_contract_version')
    if binding and bound is None:
        raise SystemExit(f'{run_name}: binding.json carries no intervention contract version '
                         '(explainer_config.intervention_contract_version)')
    if record_versions and bound is not None and record_versions != {bound}:
        raise SystemExit(f'{run_name}: record contract version {sorted(record_versions)[0]} does not match '
                         f'its binding ({bound})')
    version = next(iter(record_versions), bound)
    if version is None:
        raise SystemExit(f'{run_name}: no intervention contract version in its records or binding')
    return version


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
    sources, source_failures, record_versions = {}, {}, set()
    for filename in manifest.get('record_files', []):
        record = json.loads((run_dir / filename).read_text())
        record_versions.add((record.get('fidelity_v2_intervention') or {}).get('version'))
        for name, entry in (record.get('fidelity_v2') or {}).items():
            metrics = source_metrics(entry)
            if metrics is None:
                source_failures[name] = source_failures.get(name, 0) + 1
                sources.setdefault(name, [])
            else:
                sources.setdefault(name, []).append(metrics)
    return {'method': manifest['method'], 'fold': fold, 'role': role,
            'final_eval': (binding.get('model_run') or {}).get('final_eval'),
            'version': contract_version(run_dir.name, record_versions, binding),
            'visits': int(manifest.get('explained_count', len(manifest.get('record_files', [])))),
            'failed': int(manifest.get('failed_count', 0)),
            'sources': sources, 'source_failures': source_failures}


def read_driver(root):
    """The driver manifest of an output folder, or None when it has none."""
    driver = Path(root) / 'driver_manifest.json'
    return json.loads(driver.read_text()) if driver.is_file() else None


def run_succeeded(entry):
    status = entry.get('status')
    return entry.get('returncode') == 0 or (entry.get('returncode') is None and status == 'skipped_existing')


def not_summarised(header):
    """The planned runs of a driver manifest that did not finish with return code 0."""
    rows = []
    for entry in (header or {}).get('runs', []):
        if not run_succeeded(entry):
            rows.append({'k': entry.get('k'), 'method': entry.get('method'), 'seed': entry.get('seed'),
                         'returncode': entry.get('returncode')})
    return rows


def collect(root, allow_incomplete=False):
    root = Path(root)
    protocol, driver_fold, left_out = None, None, []
    header = read_driver(root)
    if header is not None:
        protocol, driver_fold = header.get('protocol'), header.get('fold')
        left_out = not_summarised(header)
        if (header.get('complete') is not True or left_out) and not allow_incomplete:
            raise SystemExit('the driver run is not complete (' + (
                ', '.join(f'K={r["k"]} {r["method"]} seed {r["seed"]}: return code {r["returncode"]}'
                          for r in left_out) or 'complete is not true') +
                '); finish it with --resume or pass --allow-incomplete to summarise the finished runs')
    skip = {(r['k'], r['method'], r['seed']) for r in left_out}
    runs = []
    for top in sorted(root.iterdir()):
        match = TOP_NAME.match(top.name)
        if not (top.is_dir() and match):
            continue
        for run_dir in sorted(top.iterdir()):
            name = RUN_NAME.match(run_dir.name)
            if name and (run_dir / 'explanation_manifest.json').is_file():
                if (int(match['k']), name['method'], int(name['seed'])) in skip:
                    continue
                run = load_run(run_dir)
                run.update(k=int(match['k']), seed=int(name['seed']))
                runs.append(run)
    if not runs:
        raise SystemExit('no explanation_manifest.json found under top<K>/<method>_seed<seed>')
    for key, what in (('fold', 'folds'), ('role', 'fold roles'), ('final_eval', 'final_eval values'),
                      ('version', 'intervention contract versions')):
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


def write_outputs(out, rows, fold, role, protocol, version, left_out=()):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    label = FOLD_LABEL[fold]
    left_out = list(left_out)
    columns = ['fold', 'k', 'method', 'source', 'runs', 'visits', 'failed_visits', 'failed_sources'] \
        + [f'{m}_{part}' for m in METRICS for part in ('mean', 'sd')]
    with open(out / 'summary.csv', 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([label if c == 'fold' else ('' if row[c] is None else row[c]) for c in columns])
    payload = {'schema': 'medgnn.explanation_summary', 'fold': fold, 'fold_label': label, 'fold_role': role,
               'protocol': protocol or 'not_recorded', 'intervention_contract_version': version,
               'headline_curve': HEADLINE_CURVE, 'aggregation': 'mean over visits per run, then mean and '
               'sample SD over seeds', 'rows': rows, 'runs_not_summarised': left_out}
    (out / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    shown = md_columns()
    lines = [f'# Explanation summary, fold: {label}', '',
             f'Protocol: {payload["protocol"]}. Intervention contract version {version}. Mean ± SD over seeds of '
             'the per-run visit means; fidelity numbers are drops of the predicted-class probability; '
             '`*_mean_change` is the curve AUC divided by the explained fraction. No identifiers.', '']
    for k in sorted({r['k'] for r in rows}):
        lines += [f'## K = {k} · fold: {label} · intervention contract v{version}', '',
                  '| method | source | runs | visits | failed | ' + ' | '.join(shown) + ' |',
                  '|---|---|---|---|---|' + '---|' * len(shown)]
        for row in (r for r in rows if r['k'] == k):
            lines.append(f'| {row["method"]} | {row["source"]} | {row["runs"]} | {row["visits"]} | '
                         f'{row["failed_visits"]}+{row["failed_sources"]} | '
                         + ' | '.join(pm(row, m) for m in shown) + ' |')
        lines += ['', DIRECTION_NOTE, '']
    if left_out:
        lines += ['## Runs not summarised', '', '| K | method | seed | return code |', '|---|---|---|---|']
        for r in left_out:
            code = 'not run' if r['returncode'] is None else r['returncode']
            lines.append(f'| {r["k"]} | {r["method"]} | {r["seed"]} | {code} |')
        lines.append('')
    (out / 'summary.md').write_text('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--explanations', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path, help='a new, empty directory')
    parser.add_argument('--allow-incomplete', action='store_true',
                        help='summarise the finished runs of an incomplete driver run and list the others')
    args = parser.parse_args(argv)
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        raise SystemExit(f'--out must be a new or empty directory: {args.out.name}')
    runs, protocol = collect(args.explanations, allow_incomplete=args.allow_incomplete)
    rows = aggregate(runs)
    left_out = not_summarised(read_driver(args.explanations)) if args.allow_incomplete else []
    write_outputs(args.out, rows, runs[0]['fold'], runs[0]['role'], protocol, runs[0]['version'], left_out)
    print(f'{len(runs)} runs, {len(rows)} rows, fold {FOLD_LABEL[runs[0]["fold"]]}, '
          f'contract version {runs[0]["version"]}' + (f', {len(left_out)} runs not summarised' if left_out else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
