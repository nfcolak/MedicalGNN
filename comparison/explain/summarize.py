"""Aggregate explanation tables: mean and SD over seeds per (K, method, importance source).

Reads the finished explanation outputs of `comparison.explain.run` (`<explanations>/top<K>/<method>_seed<seed>/`
with `explanation_manifest.json`, `binding.json` and one record file per explained visit) and writes aggregates only.
Inference-free: nothing is loaded except those JSON files, nothing is trained, no checkpoint is touched.

    python -m comparison.explain.summarize --explanations <driver out> --out <new dir> \
        [--allow-incomplete] [--allow-mixed-code]

Outputs (`--out` must be new or empty): `summary.csv`, `summary.json`, `summary.md`.

What is read. Only explanation record schema 3 with intervention contract version 6 (node deletion with combo
provenance and concept cascade, emptied prior visits dropped, tie-aware scoring, width-matched random reference);
anything else is refused with the run's name.

Rows. One row per (K, method, importance source); the sources are the GraphXAI algorithms of the records plus the
model's built-in explanation (`builtin`). The vendored zero-baseline integrated gradients
(`IntegratedGradExplainerZeroBaseline`) stays in the records and is left out of every table. Per run the metrics are
first averaged over its explained visits, then the run means are averaged over the seeds (mean and sample SD; SD is
empty with a single run).

Headline. `deletion_mean_change_vs_random` and `insertion_mean_change_vs_random`: the width-matched difference of a
curve to the random-removal reference of the same graph (positive deletion difference and negative insertion
difference are better than random). `mean_change` is the curve AUC divided by the curve's explained fraction, so a
sparse and a dense explanation are comparable; the raw AUC, `explained_fraction`, fidelity+ and fidelity- (drop of the
predicted-class probability when the top 20% of evidence nodes are removed, respectively only they are kept) and the
fraction of random draws beaten stay as side columns. `HEADLINE_CURVE` only sets which curve measure leads the
markdown columns; both are always written. Sparsity is written under two names: `mass90_sparsity` (share of eligible
nodes that can be dropped while keeping 90% of the importance mass) and `one_minus_support_fraction` (share of eligible
nodes with zero importance).

Missing sources are counted, never dropped silently. Per row: `visits` (explained visits of the runs that carry the
source), `visits_explained` (visits with metrics, the ones the means are over), `visits_unsupported` and
`visits_failed_explainer` (the explainer returned no explanation), `visits_failed_fidelity` (the fidelity evaluation
failed), `visits_unavailable` (the model's built-in explanation is unavailable) and `visits_no_eligible` (the
explanation holds no eligible evidence node); the six add up to `visits`. `failed_visits` counts visits the runner
could not explain at all. `visits_vs_random` counts the explained visits whose metrics carry the difference to the
random reference (the `*_vs_random` and `*_beats_random` columns are means over these visits only; a visit whose
random reference failed or was not computed has none); a headline cell whose runs rest on fewer visits than
`visits_explained` is marked with `*` in the markdown, and the missing reference statuses are listed per row
(`random_reference_missing`). Also per row: the share of integrated-gradients visits that met the completeness tolerance,
the share of visits whose importance is constant over the eligible nodes, the share where the evaluation cut fell
inside a group of equal importance, and the mean size of the largest tie group. A second table holds the
visits every source of a method has (the paired table); a third lists the sources that are not available on any visit,
with their reasons.

Caveats. Under each K table, per method: the intervention, built-in and explainer disclosure that the run manifests
carry, the settings, the replay rule that passed, and for ProtGNN how many of its seeds had their prototypes projected.

Checks (all refuse with a message that names the run): fold and fold role, `final_eval`, intervention contract
version, record schema; K and seed from the folder name against the run's binding; the protocol from the bindings
against the driver header; `protocol_overrides` an empty list in every binding; a driver run that is not complete (a run without return code 0) unless `--allow-incomplete`
(the runs left out are listed in a "runs not summarised" table); the run folders against the driver plan (a planned run
whose folder is missing, a folder that is not in the plan, a cohort sample size other than the plan's, with 0 and
null both meaning every patient); and, unless `--allow-mixed-code`, runs of different
explanation code, protocols, cohort sample, or settings, and, within one (K, method), different disclosures. Without a
driver header the run set cannot be checked and the markdown says so.

Every table carries its fold: `test (report fold)` for the reporting fold, `validation (decision fold)` for the
labelled decision fold, and the intervention contract version. The summary writes no sample, subject or stay
identifiers and no absolute paths.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

HEADLINE_CURVE = 'mean_change'    # 'auc' or 'mean_change': which curve measure leads the markdown columns
RECORD_SCHEMA_VERSION = 3
CONTRACT_VERSION = 6
FOLD_LABEL = {'test': 'test (report fold)', 'validation': 'validation (decision fold)',
              'dev': 'dev (decision fold)'}
FOLD_ROLE = {'test': 'report', 'validation': 'decision', 'dev': 'decision'}
RUN_NAME = re.compile(r'^(?P<method>.+)_seed(?P<seed>\d+)$')
TOP_NAME = re.compile(r'^top(?P<k>\d+)$')
IG_SOURCE = 'IntegratedGradExplainer'
ZERO_BASELINE_IG = 'IntegratedGradExplainerZeroBaseline'
EXCLUDED_SOURCES = {ZERO_BASELINE_IG: 'vendored zero-baseline integrated gradients (no completeness check)'}
CURVE_MEASURES = ('deletion_auc', 'insertion_auc', 'deletion_mean_change', 'insertion_mean_change')
RANDOM_KEYS = ('fidelity_plus', 'fidelity_minus') + CURVE_MEASURES
METRICS = ('fidelity_plus', 'fidelity_minus') + CURVE_MEASURES \
    + ('explained_fraction', 'mass90_sparsity', 'one_minus_support_fraction') \
    + tuple(f'{key}_vs_random' for key in RANDOM_KEYS) + tuple(f'{key}_beats_random' for key in RANDOM_KEYS)
CATEGORIES = ('explained', 'unsupported', 'failed_explainer', 'failed_fidelity', 'unavailable', 'no_eligible')
COUNT_COLUMNS = tuple(f'visits_{c}' for c in CATEGORIES)
# a visit's metrics carry the difference to random exactly when this key is present
VS_RANDOM_KEY = 'fidelity_plus_vs_random'
RANDOM_REFERENCE_STATUSES = ('computed', 'failed', 'not_requested', 'no_eligible_evidence')
POOLED = ('ig_completeness_met_share', 'constant_importance_share', 'tied_at_cut_share', 'largest_tie_group_mean')
# settings that must agree across every run of one table; the chosen IG step count is run-specific and not here
IG_COMPARABLE = ('baseline', 'tolerance', 'step_candidates', 'candidates', 'escalation')
SAMPLING_COMPARABLE = ('sample_patients', 'sample_seed', 'max_subjects')
REASON_LIMIT = 200
DIRECTION_NOTE = ('Direction: higher is better for fidelity_plus and the deletion columns; lower is better for '
                  'fidelity_minus and the insertion columns; a negative insertion difference to random '
                  '(`*_vs_random`) is better than random, a positive deletion difference is. The headline is the '
                  'width-matched difference to random (`deletion_mean_change_vs_random`, '
                  '`insertion_mean_change_vs_random`).')


def md_columns(headline=None):
    """Markdown columns; the headline difference to random leads, then the headline curve measure, then the other."""
    headline = headline or HEADLINE_CURVE
    if headline not in ('auc', 'mean_change'):
        raise ValueError(f'HEADLINE_CURVE must be "auc" or "mean_change", not {headline!r}')
    other = 'mean_change' if headline == 'auc' else 'auc'
    curves = [f'{kind}_{measure}' for measure in (headline, other) for kind in ('deletion', 'insertion')]
    lead = [f'deletion_{headline}_vs_random', f'insertion_{headline}_vs_random']
    rest = [f'{c}_vs_random' for c in curves if f'{c}_vs_random' not in lead]
    return lead + ['fidelity_plus', 'fidelity_minus'] + curves \
        + ['explained_fraction', 'mass90_sparsity', 'one_minus_support_fraction'] + rest


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def source_metrics(entry):
    """The scalar metrics of one source in one record; None when the entry is not a metrics dict."""
    try:
        deletion, insertion = entry['deletion_curve'], entry['insertion_curve']
        out = {'fidelity_plus': float(entry['fidelity_plus']['prob']),
               'fidelity_minus': float(entry['fidelity_minus']['prob']),
               'deletion_auc': float(deletion['auc']),
               'insertion_auc': float(insertion['auc']),
               'mass90_sparsity': float(entry['sparsity']['mass90_sparsity'])}
    except (KeyError, TypeError):
        return None
    support = entry['sparsity'].get('support_fraction')
    if support is not None:
        out['one_minus_support_fraction'] = 1.0 - float(support)
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


SAMPLE_NAMES = {'sample_patients': 'size', 'sample_seed': 'seed', 'max_subjects': 'cap'}


def sample_facts(sampling):
    """The cohort sampling of a run under neutral names (size, seed, cap)."""
    return {SAMPLE_NAMES[k]: sampling[k] for k in SAMPLING_COMPARABLE if k in sampling}


def sample_size(value):
    """One reading of a cohort sample size: 0 and None both mean every patient of the fold (the runner records
    None for the driver's `--sample-patients 0`)."""
    return None if value in (0, None) else int(value)


def short_reason(text):
    """A fixed model message (never an identifier); long texts are cut."""
    text = ' '.join(str(text).split())
    return text if len(text) <= REASON_LIMIT else text[:REASON_LIMIT - 3] + '...'


def classify_record(record, where):
    """{source: (category, metrics or None, reason or None, extras)} for every tabulated source of one record."""
    graphxai = record.get('graphxai') or {}
    fidelity = record.get('fidelity_v2') or {}
    builtin_status = record.get('builtin_status')
    if builtin_status not in ('available', 'unavailable', 'no_eligible_evidence'):
        raise SystemExit(f'{where}: a record carries builtin_status {builtin_status!r}')
    reference = (record.get('random_reference') or {}).get('status')
    if reference is not None and reference not in RANDOM_REFERENCE_STATUSES:
        raise SystemExit(f'{where}: a record carries random_reference status {reference!r}')
    names =[n for n in dict.fromkeys([*graphxai, *fidelity, 'builtin']) if n not in EXCLUDED_SOURCES]
    out = {}
    for name in names:
        extras = {}
        entry = fidelity.get(name)
        if name == 'builtin':
            if builtin_status == 'unavailable':
                out[name] = ('unavailable', None, short_reason(record.get('builtin_unavailable_reason') or
                                                                 'no reason recorded'), extras)
                continue
            if builtin_status == 'no_eligible_evidence':
                out[name] = ('no_eligible', None, None, extras)
                continue
            extras['tie_summary'] = record.get('builtin_tie_summary')
        else:
            payload = graphxai.get(name)
            status = None if payload is None else payload.get('status')
            if payload is not None and status == 'unsupported':
                out[name] = ('unsupported', None, short_reason(payload.get('unsupported_reason') or
                                                               'no reason recorded'), extras)
                continue
            if payload is not None and status == 'failed':
                out[name] = ('failed_explainer', None, None, extras)
                continue
            if payload is not None and status != 'success':
                raise SystemExit(f'{where}: {name} carries status {status!r}')
            provenance = (payload or {}).get('provenance') or {}
            if (name == IG_SOURCE and status == 'success' and 'completeness_met' not in provenance
                    and 'ig_clinical' in str(provenance.get('implementation'))):
                raise SystemExit(f'{where}: {name} (clinical IG) carries no completeness_met')
            if name == IG_SOURCE and 'completeness_met' in provenance:
                extras['ig_completeness_met'] = bool(provenance['completeness_met'])
            if name == IG_SOURCE:
                extras['implementation'] = provenance.get('implementation')
        if not isinstance(entry, dict):
            raise SystemExit(f'{where}: {name} has no fidelity_v2 entry')
        if entry.get('status') == 'failed':
            out[name] = ('failed_fidelity', None, None, extras)
        elif entry.get('status') == 'no_eligible_evidence':
            out[name] = ('no_eligible', None, None, extras)
        else:
            metrics = source_metrics(entry)
            if metrics is None:
                raise SystemExit(f'{where}: {name} has a malformed fidelity_v2 entry')
            ties = entry.get('tie_summary') or extras.get('tie_summary') or {}
            if 'constant_over_eligible' in ties:
                extras['constant_importance'] = bool(ties['constant_over_eligible'])
            if ties.get('largest_tie_group_eligible') is not None:
                extras['largest_tie_group'] = float(ties['largest_tie_group_eligible'])
            tied = (entry.get('fidelity_plus') or {}).get('tied_at_cut')
            if tied is not None:
                extras['tied_at_cut'] = bool(tied)
            if VS_RANDOM_KEY not in metrics:
                extras['random_reference_missing'] = reference or 'not recorded'
            out[name] = ('explained', metrics, None, extras)
    return out


def contract_version(run_name, record_versions, binding):
    """The run's intervention contract version from its records and its binding; refuses a missing
    version, a disagreement between records, or between the records and the binding, and any version
    other than the one this summary reads."""
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
    if version != CONTRACT_VERSION:
        raise SystemExit(f'{run_name}: intervention contract version {version}; this summary reads only '
                         f'version {CONTRACT_VERSION}')
    return version


def canonical(value):
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)


def code_identity(code):
    """A short digest over the hashed explanation code and the vendored GraphXAI tree."""
    if not isinstance(code, dict) or not isinstance(code.get('source_hashes'), dict) or not code['source_hashes']:
        return None
    body = {'source_hashes': code['source_hashes'], 'vendored': code.get('vendored_graphxai_tree_sha256')}
    return hashlib.sha256(canonical(body).encode()).hexdigest()[:12]


def comparable_settings(manifest, binding):
    """The settings every run of one table must share (run-specific choices such as the chosen IG start count are not)."""
    settings = dict(manifest.get('settings') or {})
    ig = settings.pop('ig', None)
    if isinstance(ig, dict):
        settings['ig'] = {key: ig[key] for key in IG_COMPARABLE if key in ig}
    sampling = ((binding.get('cohort') or {}).get('sampling')) or {}
    settings['cohort_sampling'] = {key: sampling.get(key) for key in SAMPLING_COMPARABLE}
    return settings


def method_disclosure(manifest, folder_method):
    block = manifest.get('disclosure')
    if not isinstance(block, dict):
        return None
    for key in (manifest.get('method'), folder_method):
        if isinstance(block.get(key), dict):
            return block[key]
    return None


def load_run(run_dir, k, folder_method, seed):
    """Manifest facts and the per-source metric dicts of one run; ids are used to open files, never kept."""
    name = run_dir.name
    manifest = json.loads((run_dir / 'explanation_manifest.json').read_text())
    binding_path = run_dir / 'binding.json'
    binding = json.loads(binding_path.read_text()) if binding_path.is_file() else {}
    model_run = binding.get('model_run') or {}
    fold = manifest.get('fold')
    if fold not in FOLD_LABEL:
        raise SystemExit(f'{name}: unknown fold {fold!r}')
    role = manifest.get('fold_role', FOLD_ROLE[fold])
    if role != FOLD_ROLE[fold]:
        raise SystemExit(f'{name}: fold_role {role!r} does not belong to fold {fold!r}')
    for key, folder in (('top_k_labels', k), ('seed', seed)):
        if model_run.get(key) is None:
            raise SystemExit(f'{name}: binding.json model_run carries no {key}')
        if model_run[key] != folder:
            raise SystemExit(f'{name}: folder says {key} {folder}, its binding says {model_run[key]!r}')
    if model_run.get('protocol') is None:
        raise SystemExit(f'{name}: binding.json model_run carries no protocol')
    if model_run.get('protocol_overrides') != []:
        raise SystemExit(f'{name}: binding.json protocol_overrides must be an empty list, not '
                         f'{model_run.get("protocol_overrides")!r}')
    if manifest.get('method') != model_run.get('method'):
        raise SystemExit(f'{name}: manifest method {manifest.get("method")!r} differs from its binding '
                         f'({model_run.get("method")!r})')
    code = manifest.get('code')
    if code_identity(code) is None:
        raise SystemExit(f'{name}: the manifest carries no hashed explanation code (code.source_hashes)')
    disclosure = method_disclosure(manifest, folder_method)
    if disclosure is None:
        raise SystemExit(f'{name}: the manifest carries no disclosure for method {manifest.get("method")!r}')
    record_versions, statuses, reasons, metrics, extras = set(), {}, {}, {}, {}
    for index, filename in enumerate(manifest.get('record_files', [])):
        record = json.loads((run_dir / filename).read_text())
        if record.get('schema_version') != RECORD_SCHEMA_VERSION:
            raise SystemExit(f'{name}: a record has schema version {record.get("schema_version")!r}; this '
                             f'summary reads only schema {RECORD_SCHEMA_VERSION}')
        record_versions.add((record.get('fidelity_v2_intervention') or {}).get('version'))
        for source, (category, values, reason, more) in classify_record(record, name).items():
            counts = statuses.setdefault(source, dict.fromkeys(CATEGORIES, 0))
            counts[category] += 1
            if reason is not None:
                reasons.setdefault(source, {}).setdefault(category, set()).add(reason)
            if values is not None:
                metrics.setdefault(source, {})[index] = values
            extras.setdefault(source, {})[index] = more
    version = contract_version(name, record_versions, binding)
    return {'method': manifest['method'], 'fold': fold, 'role': role, 'k': k, 'seed': seed,
            'folder_method': folder_method, 'name': f'top{k}/{folder_method}_seed{seed}',
            'final_eval': model_run.get('final_eval'), 'protocol': model_run['protocol'], 'version': version,
            'visits': int(manifest.get('explained_count', len(manifest.get('record_files', [])))),
            'failed': int(manifest.get('failed_count', 0)),
            'sources': metrics, 'status': statuses, 'reasons': reasons, 'extras': extras,
            'code': code, 'code_id': code_identity(code), 'disclosure': disclosure,
            'builtin_explanation': manifest.get('builtin_explanation'),
            'projection': manifest.get('prototype_projection'),
            'replay_mode': (manifest.get('replay') or {}).get('mode'),
            'settings': comparable_settings(manifest, binding), 'settings_shown': manifest.get('settings') or {},
            'sampling': ((binding.get('cohort') or {}).get('sampling')) or {}}


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


def describe_groups(groups, limit=3):
    parts = []
    for value, names in groups:
        shown = ', '.join(names[:limit]) + (f' and {len(names) - limit} more' if len(names) > limit else '')
        parts.append(f'{value} ({shown})')
    return '; '.join(parts)


def flat_items(value, prefix=''):
    """{dotted key: canonical leaf} of nested mappings and lists."""
    if isinstance(value, dict) and value:
        items = value.items()
    elif isinstance(value, list) and value:
        items = enumerate(value)
    else:
        return {prefix: canonical(value)}
    out = {}
    for key, item in items:
        out.update(flat_items(item, f'{prefix}.{key}' if prefix else str(key)))
    return out


def differing_items(values):
    """For mappings, each value reduced to the dotted keys whose leaves are not the same in all of them
    (`absent` where a value lacks the key); other values unchanged."""
    if not all(isinstance(value, dict) for value in values):
        return [canonical(value) for value in values]
    flat = [flat_items(value) for value in values]
    keys = sorted({key for items in flat for key in items})
    differ = [key for key in keys if len({items.get(key) for items in flat}) > 1]
    return ['; '.join(f'{key}: {items.get(key, "absent")}' for key in differ) for items in flat]


def check_mixed(runs, label, getter, allow):
    """Refuse (or, when allowed, report) runs that disagree on `getter(run)`; for mappings only the
    entries that differ are named."""
    found, values = {}, {}
    for run in runs:
        value = getter(run)
        key = canonical(value)
        found.setdefault(key, []).append(run['name'])
        values.setdefault(key, value)
    if len(found) < 2:
        return None
    keys = sorted(found)
    shown = differing_items([values[key] for key in keys])
    groups = [(text, found[key]) for text, key in zip(shown, keys)]
    if not allow:
        raise SystemExit(f'refusing to mix {label} in one table: {describe_groups(groups)}; '
                         'pass --allow-mixed-code to summarise them anyway')
    return {'what': label, 'values': [{'value': value, 'runs': len(names)} for value, names in groups]}


def check_driver_plan(header, runs, left_out):
    """The summarised folders must be exactly the driver's planned runs minus the listed left-out runs."""
    expected = {(e.get('k'), e.get('method'), e.get('seed')): e for e in header.get('runs', [])}
    expected = {key: entry for key, entry in expected.items() if run_succeeded(entry)}
    found = {(r['k'], r['folder_method'], r['seed']) for r in runs}
    missing = sorted(set(expected) - found, key=str)
    extra = sorted(found - set(expected), key=str)
    if missing:
        raise SystemExit('the driver plan lists runs whose folder is missing: ' + ', '.join(
            f'top{k}/{m}_seed{s}' for k, m, s in missing))
    if extra:
        raise SystemExit('run folders outside the driver plan: ' + ', '.join(
            f'top{k}/{m}_seed{s}' for k, m, s in extra))
    if 'sample_patients' in header:
        planned = sample_size(header['sample_patients'])
        wrong = [r['name'] for r in runs if sample_size(r['sampling'].get('sample_patients')) != planned]
        if wrong:
            raise SystemExit(f'cohort sample differs from the driver plan '
                             f'({"every patient" if planned is None else planned}): ' + ', '.join(wrong))


def collect(root, allow_incomplete=False, allow_mixed_code=False):
    """(runs, meta): every summarised run, and the table-level facts (protocol, run-set check, mixed warnings)."""
    root = Path(root)
    driver_fold, left_out = None, []
    header = read_driver(root)
    if header is not None:
        driver_fold = header.get('fold')
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
                key = (int(match['k']), name['method'], int(name['seed']))
                if key in skip:
                    continue
                runs.append(load_run(run_dir, key[0], key[1], key[2]))
    if not runs:
        raise SystemExit('no explanation_manifest.json found under top<K>/<method>_seed<seed>')
    for key, what in (('fold', 'folds'), ('role', 'fold roles'), ('final_eval', 'final_eval values'),
                      ('version', 'intervention contract versions')):
        found = sorted({str(run[key]) for run in runs})
        if len(found) > 1:
            raise SystemExit(f'refusing to mix {what} in one table: {", ".join(found)}')
    if driver_fold is not None and driver_fold != runs[0]['fold']:
        raise SystemExit(f'refusing to mix folds in one table: driver {driver_fold}, runs {runs[0]["fold"]}')
    if header is not None:
        check_driver_plan(header, runs, left_out)
    mixed = [m for m in (
        check_mixed(runs, 'protocols', lambda r: r['protocol'], allow_mixed_code),
        check_mixed(runs, 'explanation code', lambda r: r['code_id'], allow_mixed_code),
        check_mixed(runs, 'settings', lambda r: r['settings'], allow_mixed_code),
    ) if m]
    for k, method in sorted({(r['k'], r['method']) for r in runs}):
        group = [r for r in runs if (r['k'], r['method']) == (k, method)]
        for label, getter in (('disclosures', lambda r: r['disclosure']),
                              ('built-in explanation settings', lambda r: r['builtin_explanation'])):
            found = check_mixed(group, f'{label} within K={k} {method}', getter, allow_mixed_code)
            if found:
                mixed.append(found)
    protocols = sorted({r['protocol'] for r in runs})
    driver_protocol = (header or {}).get('protocol')
    if driver_protocol is not None and protocols != [driver_protocol] and not allow_mixed_code:
        raise SystemExit(f'the driver header says protocol {driver_protocol}, the run bindings say '
                         + ', '.join(protocols))
    meta = {'protocol': protocols[0] if len(protocols) == 1 else 'mixed: ' + ', '.join(protocols),
            'run_set_checked': header is not None, 'mixed': mixed, 'left_out': left_out,
            'planned_seeds': planned_seeds(header)}
    return runs, meta


def planned_seeds(header):
    """{(K, driver method name): number of planned seeds} from a driver header."""
    counts = {}
    for entry in (header or {}).get('runs', []):
        key = (entry.get('k'), entry.get('method'))
        counts[key] = counts.get(key, 0) + 1
    return counts


def pair_sources(runs):
    """{(K, method): sources with at least one explained visit}; the sources the paired table compares."""
    sources = {}
    for run in runs:
        for source, per_visit in run['sources'].items():
            if per_visit:
                sources.setdefault((run['k'], run['method']), set()).add(source)
    return sources


def source_label(source, implementations):
    if source == IG_SOURCE and implementations and all('ig_clinical' in str(i) for i in implementations):
        return 'IntegratedGradExplainer (clinical baseline)'
    return source


def share(values):
    return sum(values) / len(values) if values else None


def aggregate(runs, planned=None, paired=False):
    """Rows keyed by (K, method, source); mean over visits per run, then mean and SD over the run means.

    With `paired`, only the visits that every source of the method has are used."""
    common_sources = pair_sources(runs) if paired else {}
    groups = {}
    for run in runs:
        for source in run['status']:
            groups.setdefault((run['k'], run['method'], source), []).append(run)
    rows = []
    for (k, method, source), members in sorted(groups.items()):
        if paired and source not in common_sources.get((k, method), ()):
            continue
        used = {}
        for run in members:
            visits = run['sources'].get(source, {})
            if paired:
                shared = [set(run['sources'].get(s, {})) for s in common_sources[(k, method)]]
                visits = {i: m for i, m in visits.items() if all(i in s for s in shared)}
            used[run['name']] = visits
        counts = {c: sum(run['status'][source][c] for run in members) for c in CATEGORIES}
        implementations = {e.get('implementation') for run in members
                           for e in run['extras'].get(source, {}).values() if e.get('implementation')}
        row = {'table': 'paired_visits' if paired else 'all_visits', 'k': k, 'method': method, 'source': source,
               'source_label': source_label(source, implementations), 'runs': len(members),
               'runs_planned': None if planned is None else planned.get((k, members[0]['folder_method'])),
               'visits': None if paired else sum(counts.values()),
               'failed_visits': None if paired else sum(run['failed'] for run in members),
               'paired_visits': sum(len(v) for v in used.values()) if paired else None}
        for category in CATEGORIES:
            row[f'visits_{category}'] = None if paired else counts[category]
        row['visits_vs_random'] = sum(1 for visits in used.values() for m in visits.values() if VS_RANDOM_KEY in m)
        row['random_reference_missing'] = sorted({
            extra['random_reference_missing'] for run in members
            for i, extra in run['extras'].get(source, {}).items()
            if i in used[run['name']] and 'random_reference_missing' in extra})
        reasons = {}
        for run in members:
            for category, texts in run['reasons'].get(source, {}).items():
                reasons.setdefault(category, set()).update(texts)
        row['reasons'] = {c: sorted(t) for c, t in sorted(reasons.items())}
        picked = [(run, run['extras'].get(source, {}), used[run['name']]) for run in members]
        for column, key in (('ig_completeness_met_share', 'ig_completeness_met'),
                            ('constant_importance_share', 'constant_importance'),
                            ('tied_at_cut_share', 'tied_at_cut')):
            row[column] = share([float(extra[key]) for _, extras, visits in picked
                                 for i, extra in extras.items() if i in visits and key in extra])
        sizes = [extra['largest_tie_group'] for _, extras, visits in picked
                 for i, extra in extras.items() if i in visits and 'largest_tie_group' in extra]
        row['largest_tie_group_mean'] = share(sizes)
        for metric in METRICS:
            run_means = []
            for visits in used.values():
                values = [m[metric] for m in visits.values() if metric in m]
                if values:
                    run_means.append(sum(values) / len(values))
            row[f'{metric}_mean'], row[f'{metric}_sd'] = mean_sd(run_means)
        rows.append(row)
    return rows


def split_available(rows):
    """(rows with at least one explained visit, rows of sources that are not available on any visit)."""
    return [r for r in rows if r['visits_explained'] > 0], [r for r in rows if r['visits_explained'] == 0]


def fmt(value, digits=4):
    return '' if value is None else f'{value:.{digits}f}'


def visits_with_metrics(row):
    """The visits a row's means are over: explained visits (all-visits table) or paired visits."""
    return row['paired_visits'] if row['table'] == 'paired_visits' else row['visits_explained']


def pm(row, metric):
    mean, sd = row[f'{metric}_mean'], row[f'{metric}_sd']
    if mean is None:
        return ''
    text = fmt(mean) if sd is None else f'{fmt(mean)} ± {fmt(sd)}'
    if metric.endswith(('_vs_random', '_beats_random')) and row['visits_vs_random'] < visits_with_metrics(row):
        text += ' *'    # a mean over fewer visits than the row's other columns
    return text


def render(value):
    """Plain text of a disclosure value (a string, a list or a mapping)."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return '; '.join(f'{k}: {render(v)}' for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return '; '.join(render(v) for v in value)
    return json.dumps(value, sort_keys=True)


def projection_line(group):
    """'k of n seeds projected' for the methods whose manifests carry a prototype projection status."""
    states = [(r['seed'], r['projection']) for r in group if isinstance(r['projection'], dict)]
    if not states:
        return None
    projected = sorted(seed for seed, p in states if p.get('prototypes_projected'))
    unprojected = sorted(seed for seed, p in states if not p.get('prototypes_projected'))
    text = f'Prototype projection: {len(projected)} of {len(states)} seeds projected'
    if unprojected:
        text += (f' (seed {", ".join(map(str, unprojected))}: selected before the first projection, '
                 'explained like the others and averaged with them)')
    late = [f'seed {seed} {p["epochs_trained_after_last_projection"]}' for seed, p in sorted(states, key=lambda s: s[0])
            if p.get('epochs_trained_after_last_projection') is not None]
    if late:
        text += '; epochs trained after the last projection: ' + ', '.join(late)
    return text


def caveat_lines(group):
    """The caveat bullet list of one (K, method): disclosure of the run manifests, settings, replay, projection."""
    first = group[0]
    disclosure = first['disclosure']
    lines = []
    for key, title in (('intervention', 'Intervention'), ('builtin', 'Built-in explanation'),
                       ('explainers', 'Explainers')):
        if disclosure.get(key) is not None:
            lines.append(f'{title}: {render(disclosure[key])}')
    lines += [render(c) for c in disclosure.get('caveats') or []]
    line = projection_line(group)
    if line:
        lines.append(line)
    modes = sorted({r['replay_mode'] for r in group if r['replay_mode']})
    if modes:
        lines.append('Replay rule that passed: ' + ', '.join(modes))
    if first['builtin_explanation']:
        lines.append('Built-in explanation source: ' + render(first['builtin_explanation']))
    shown = first['settings_shown']
    if shown:
        lines.append('Settings: ' + render(shown))
    sampling = first['sampling']
    if sampling:
        lines.append('Cohort sample: ' + render(sample_facts(sampling)))
    return lines


def csv_columns():
    return ['table', 'fold', 'k', 'method', 'source', 'runs', 'runs_planned', 'visits', *COUNT_COLUMNS,
            'visits_vs_random', 'failed_visits', 'paired_visits', *POOLED] + [f'{m}_{part}' for m in METRICS for part in ('mean', 'sd')]


def md_table(rows, shown, label=lambda r: r['source_label']):
    lines = ['| method | source | runs | visits | explained | failed | ' + ' | '.join(shown) + ' |',
             '|---|---|---|---|---|---|' + '---|' * len(shown)]
    for row in rows:
        runs = f'{row["runs"]}' if row['runs_planned'] in (None, row['runs']) \
            else f'{row["runs"]} of {row["runs_planned"]}'
        lines.append(f'| {row["method"]} | {label(row)} | {runs} | {row["visits"]} | {row["visits_explained"]} | '
                     f'{row["failed_visits"]} | ' + ' | '.join(pm(row, m) for m in shown) + ' |')
    return lines


def write_outputs(out, rows, paired_rows, runs, meta):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fold, role, version = runs[0]['fold'], runs[0]['role'], runs[0]['version']
    label = FOLD_LABEL[fold]
    left_out = list(meta['left_out'])
    available, missing = split_available(rows)
    paired_available = [r for r in paired_rows]
    columns = csv_columns()
    with open(out / 'summary.csv', 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in available + paired_available:
            writer.writerow([label if c == 'fold' else ('' if row.get(c) is None else row[c]) for c in columns])
    codes = {}
    for run in runs:
        codes.setdefault(run['code_id'], []).append(run)
    caveats = {}
    for k, method in sorted({(r['k'], r['method']) for r in runs}):
        group = [r for r in runs if (r['k'], r['method']) == (k, method)]
        caveats[f'K={k} {method}'] = caveat_lines(group)
    sampling = runs[0]['sampling']
    payload = {'schema': 'medgnn.explanation_summary', 'fold': fold, 'fold_label': label, 'fold_role': role,
               'protocol': meta['protocol'], 'intervention_contract_version': version,
               'record_schema_version': RECORD_SCHEMA_VERSION,
               'headline': ['deletion_mean_change_vs_random', 'insertion_mean_change_vs_random'],
               'headline_curve': HEADLINE_CURVE, 'aggregation': 'mean over visits per run, then mean and '
               'sample SD over seeds', 'rows': available, 'paired_rows': paired_available,
               'sources_not_available': missing, 'excluded_sources': EXCLUDED_SOURCES,
               'runs_not_summarised': left_out, 'run_set_checked_against_driver_plan': meta['run_set_checked'],
               'cohort_sample': sample_facts(sampling),
               'code': [{'id': code_id, 'runs': len(group), 'git_head': (group[0]['code'] or {}).get('git_head'),
                         'git_dirty': (group[0]['code'] or {}).get('git_dirty')} for code_id, group in codes.items()],
               'mixed': meta['mixed'], 'caveats': caveats}
    (out / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    shown = md_columns()
    lines = [f'# Explanation summary, fold: {label}', '',
             f'Protocol: {meta["protocol"]}. Intervention contract version {version}, record schema '
             f'{RECORD_SCHEMA_VERSION}. Mean ± SD over seeds of the per-run visit means; fidelity numbers are drops '
             'of the predicted-class probability; `*_mean_change` is the curve AUC divided by the explained '
             'fraction; the headline is the difference of `*_mean_change` to random removal. No identifiers.', '']
    if sampling:
        lines += ['Cohort sample: ' + render(sample_facts(sampling)) + '.', '']
    lines.append('Explanation code: ' + '; '.join(
        f'{code_id} ({len(group)} runs, git head {(group[0]["code"] or {}).get("git_head") or "not recorded"}'
        f'{", dirty" if (group[0]["code"] or {}).get("git_dirty") else ""})' for code_id, group in codes.items()) + '.')
    lines.append('')
    lines.append('Run set: ' + ('checked against the driver plan.' if meta['run_set_checked']
                               else 'no driver manifest: run set not checked.'))
    lines.append('')
    for entry in meta['mixed']:
        lines += [f'**WARNING: mixed {entry["what"]}** (--allow-mixed-code): ' + '; '.join(
            f'{v["value"]} ({v["runs"]} runs)' for v in entry['values']), '']
    for k in sorted({r['k'] for r in rows}):
        k_rows = [r for r in available if r['k'] == k]
        lines += [f'## K = {k} · fold: {label} · intervention contract v{version}', '']
        if k_rows:
            lines += md_table(k_rows, shown) + ['']
        lines += ['### Visit accounting', '',
                  '| method | source | visits | explained | unsupported | explainer failed | fidelity failed | '
                  'built-in unavailable | no eligible evidence | with random reference | IG complete | '
                  'constant importance | tied at cut | largest tie group |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for row in (r for r in rows if r['k'] == k):
            lines.append(f'| {row["method"]} | {row["source_label"]} | {row["visits"]} | '
                         + ' | '.join(str(row[c]) for c in COUNT_COLUMNS) + f' | {row["visits_vs_random"]} | '
                         + ' | '.join(fmt(row[c], 3) for c in POOLED[:3]) + f' | {fmt(row[POOLED[3]], 2)} |')
        lines.append('')
        short = [r for r in rows if r['k'] == k and r['visits_vs_random'] < r['visits_explained']]
        if short:
            lines += ['`*`: the difference to random of these rows rests on fewer visits than the other columns '
                      '(random reference ' + ', '.join(sorted({s for r in short for s in r['random_reference_missing']}))
                      + '): ' + '; '.join(f'{r["method"]} {r["source_label"]} {r["visits_vs_random"]} of '
                                          f'{r["visits_explained"]}' for r in short) + '.', '']
        p_rows = [r for r in paired_available if r['k'] == k]
        if p_rows:
            lines += ['### Paired comparison (visits every source of the method has)', '',
                      '| method | source | runs | paired visits | ' + ' | '.join(shown) + ' |',
                      '|---|---|---|---|' + '---|' * len(shown)]
            for row in p_rows:
                lines.append(f'| {row["method"]} | {row["source_label"]} | {row["runs"]} | {row["paired_visits"]} | '
                             + ' | '.join(pm(row, m) for m in shown) + ' |')
            lines.append('')
        m_rows = [r for r in missing if r['k'] == k]
        if m_rows:
            lines += ['### Sources not available', '', '| method | source | visits | unsupported | explainer failed | '
                      'fidelity failed | built-in unavailable | no eligible evidence | reasons |',
                      '|---|---|---|---|---|---|---|---|---|']
            for row in m_rows:
                reasons = '; '.join(sorted({t for texts in row['reasons'].values() for t in texts}))
                lines.append(f'| {row["method"]} | {row["source_label"]} | {row["visits"]} | '
                             + ' | '.join(str(row[c]) for c in COUNT_COLUMNS[1:]) + f' | {reasons} |')
            lines.append('')
        for key, bullets in caveats.items():
            if key.startswith(f'K={k} '):
                lines += [f'### Caveats: {key[len(f"K={k} "):]} (K = {k})', ''] + [f'- {b}' for b in bullets] + ['']
        lines += [DIRECTION_NOTE, '']
    lines += ['Left out of every table: ' + '; '.join(f'`{n}` ({why}); it stays in the records'
                                                      for n, why in EXCLUDED_SOURCES.items()) + '.', '']
    if left_out:
        lines += ['## Runs not summarised', '', '| K | method | seed | return code |', '|---|---|---|---|']
        for r in left_out:
            code = 'not run' if r['returncode'] is None else r['returncode']
            lines.append(f'| {r["k"]} | {r["method"]} | {r["seed"]} | {code} |')
        lines.append('')
    (out / 'summary.md').write_text('\n'.join(lines))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--explanations', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path, help='a new, empty directory')
    parser.add_argument('--allow-incomplete', action='store_true',
                        help='summarise the finished runs of an incomplete driver run and list the others')
    parser.add_argument('--allow-mixed-code', action='store_true',
                        help='summarise runs of different explanation code, protocols or settings (listed as a warning)')
    args = parser.parse_args(argv)
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        raise SystemExit(f'--out must be a new or empty directory: {args.out.name}')
    runs, meta = collect(args.explanations, allow_incomplete=args.allow_incomplete,
                         allow_mixed_code=args.allow_mixed_code)
    planned = meta['planned_seeds'] or None
    rows = aggregate(runs, planned)
    paired_rows = aggregate(runs, planned, paired=True)
    write_outputs(args.out, rows, paired_rows, runs, meta)
    print(f'{len(runs)} runs, {len(rows)} rows, fold {FOLD_LABEL[runs[0]["fold"]]}, '
          f'contract version {runs[0]["version"]}'
          + (f', {len(meta["left_out"])} runs not summarised' if meta['left_out'] else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
