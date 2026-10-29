"""Graph-type analysis of the top-20 explanations, step 1: per-visit tables, checks and parity with the stored summaries.

Run from the repo root:  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 comparison/explain/graph_types/step1_visit_tables.py
Paths: _paths.py in this folder (W = working folder with data/ and out/; data/ holds per-visit tables with anonymous indices and is never committed).

Reads (all read only): the explanation records of the 9 model keys x 3 seeds, each run's test.npz, the stored
summary.csv files, and the graph inputs through `core.explain.runner.rebuild_split` of the cei_final seed-1234
binding (loads no model). Writes data/visit_features.csv, data/visit_scores.csv, data/top_kinds.csv,
data/parity.md and out/step1-report.md. Stdout and the two .md files carry aggregates only; visit_idx and
patient_idx are anonymous running indices and the id map is never written. Counts of 1..9 print as "<10".
"""
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
CODE = str(REPO)
sys.path.insert(0, CODE)
from comparison.explain.graph_types._paths import STD, EXPLAIN as ART, WORK as W  # noqa: E402

import core  # noqa: E402
from comparison.explain import summarize  # noqa: E402
from core.explain import fidelity_v2 as fv2  # noqa: E402
from core.explain.runner import rebuild_split  # noqa: E402
from core.tensorize import ALL_RELATIONS, REVERSE_RELATIONS  # noqa: E402

SEEDS = (1234, 2025, 7)
MAIN = ART / 'explain-k20-p1000-6782d4b-20261024'
FAIR = STD / 'clinical_runs_fair_v10t_20261011/top20'
NASIM12 = STD / 'clinical_runs_nasim_s2_20261012'
NASIM25 = STD / 'clinical_runs_nasim_s2_20261025'
# key -> (explain run dir, folder prefix of the records, method name in its summary.csv, trained-run dir prefix)
KEYS = {
    'gsat': (MAIN, 'gsat', 'gsat', FAIR / 'gsat'),
    'graphcare': (MAIN, 'graphcare', 'graphcare', FAIR / 'graphcare'),
    'protgnn': (MAIN, 'protgnn', 'protgnn', FAIR / 'protgnn'),
    'cei_final': (MAIN, 'cei_mp2', 'cei_gnn_mp2', FAIR / 'cei_mp2'),
    'cei_R1_ls0': (ART / 'explain-k20-p1000-cei-R1_ls0-6782d4b-20261025b', 'cei_mp2', 'cei_gnn_mp2',
                   NASIM12 / 'R1_ls0/top20/cei_mp2'),
    'cei_R2_ls0_wd': (ART / 'explain-k20-p1000-cei-R2_ls0_wd-6782d4b-20261025b', 'cei_mp2', 'cei_gnn_mp2',
                      NASIM12 / 'R2_ls0_wd/top20/cei_mp2'),
    'cei_R3_ls0_wd_de0': (ART / 'explain-k20-p1000-cei-R3_ls0_wd_de0-6782d4b-20261025b', 'cei_mp2', 'cei_gnn_mp2',
                          NASIM12 / 'R3_ls0_wd_de0/top20/cei_mp2'),
    'cei_R4_wd': (ART / 'explain-k20-p1000-cei-R4_wd-6782d4b-20261025', 'cei_mp2', 'cei_gnn_mp2',
                  NASIM25 / 'R4_wd/top20/cei_mp2'),
    'cei_R5_de0': (ART / 'explain-k20-p1000-cei-R5_de0-6782d4b-20261025', 'cei_mp2', 'cei_gnn_mp2',
                   NASIM25 / 'R5_de0/top20/cei_mp2'),
}
GRAPH_BINDING = FAIR / 'cei_mp2_seed1234/binding.json'      # the only binding rebuild_split is called on
SOURCES = {'IntegratedGradExplainer': 'IG', 'GradExplainer': 'Grad', 'GNNExplainer': 'GNNExp', 'builtin': 'builtin'}
EVIDENCE = ('complaint', 'measurement', 'vital', 'diagnosis')
HEADLINE = 'deletion_mean_change_vs_random'
PARITY_METRICS = (HEADLINE, 'fidelity_plus', 'fidelity_minus', 'mass90_sparsity')
SCORE_METRICS = tuple(summarize.METRICS)       # includes every column source_metrics can return
TOLERANCE = 1e-9
NUM_VISITS_EXPECTED = 1345
NUM_PATIENTS_EXPECTED = 1000

DATA, OUT = W / 'data', W / 'out'
STARTED = time.time()


def say(text=''):
    print(text, flush=True)


def small(n):
    """A count for printing: 1..9 are shown as <10 (privacy), 0 and >= 10 as they are."""
    return '<10' if 0 < n < 10 else str(n)


def rel_col(name):
    return 'n_rel_' + name.replace(':', '_')


# --------------------------------------------------------------------------- graph structure
def graph_features():
    """{sample_id: features} for every row of the cei_final seed-1234 test split, as training built it."""
    binding = json.loads(GRAPH_BINDING.read_text())
    started = time.time()
    splits = rebuild_split(binding, fold='test')
    say(f'rebuild_split: {len(splits["test"])} test rows in {time.time() - started:.0f} s')
    kinds = list(core.NODE_KINDS)
    removable_ids = {kinds.index(k) for k in fv2.EVIDENCE_NODE_KINDS}
    prior_dx = ALL_RELATIONS.index('has_prior_diagnosis')
    diagnosis = kinds.index('diagnosis')
    n_forward = len(ALL_RELATIONS)
    out, reverse_edges = {}, 0
    for d in splits['test']:
        node_type = d.node_type.numpy().astype(np.int64)
        edge_relation = d.edge_relation.numpy().astype(np.int64)
        n_nodes, n_edges = int(node_type.size), int(edge_relation.size)
        counts = np.bincount(edge_relation, minlength=n_forward + len(REVERSE_RELATIONS))
        reverse_edges += int(counts[n_forward:].sum())
        ends = d.edge_index.numpy()[:, edge_relation == prior_dx].ravel()
        row = {'n_nodes': n_nodes, 'n_edges': n_edges,
               'n_evidence': int(np.isin(node_type, list(removable_ids)).sum()),
               'n_prior_dx': int(np.unique(ends[node_type[ends] == diagnosis]).size) if ends.size else 0,
               'density': n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0,
               'num_visits': int(d.num_visits.view(-1)[0]), 'true_class_id': int(d.y.view(-1)[0]),
               'subject': str(d.subject), 'node_type': node_type}
        for i, kind in enumerate(kinds):
            row['n_kind_' + kind] = int((node_type == i).sum())
        for i, name in enumerate(ALL_RELATIONS):
            row[rel_col(name)] = int(counts[i])
        out[str(d.sample_id)] = row
    say(f'graph features: {len(out)} rows, reverse-relation edges {reverse_edges}')
    return out, binding, reverse_edges


# --------------------------------------------------------------------------- records
def top_weights(values, k):
    """Weight of every evidence node in the top k by |importance|; the tie group at the cut shares the
    remaining slots equally (the expectation over random tie orders)."""
    cut = np.sort(values)[::-1][k - 1]
    above = values > cut
    tied = values == cut
    return above.astype(float) + tied * ((k - int(above.sum())) / int(tied.sum()))


def kind_rows(values, evidence_type, k):
    """top_share / mass_share / top_rate per evidence kind for one explanation (values = |importance|)."""
    weights = top_weights(values, k)
    total = float(values.sum())
    out = {}
    for kind in EVIDENCE:
        mask = evidence_type == core.NODE_KINDS.index(kind)
        available = int(mask.sum())
        out['top_share_' + kind] = float(weights[mask].sum() / k)
        out['mass_share_' + kind] = float(values[mask].sum() / total) if total > 0 else ''
        out['top_rate_' + kind] = float(weights[mask].sum() / available) if available else ''
    return out


def vector(record, source):
    if source == 'builtin':
        return record.get('builtin_node_importance')
    payload = (record.get('graphxai') or {}).get(source)
    if payload and payload.get('status') == 'success' and 'node_explanation' in payload:
        return payload['node_explanation']['node_importance']
    return None


def read_runs(graph):
    """Stream every record of every (key, seed); returns score rows, top-kind rows, per-run test.npz facts and
    the check counters. Rows carry the raw sample_id until the visit index is assigned."""
    scores, tops, runs, checks = [], [], {}, {}
    for key, (explain_dir, folder, _method, trained) in KEYS.items():
        chk = checks[key] = {'vectors': 0, 'length_mismatch': 0, 'unknown_visit': 0, 'duplicate': 0,
                             'class_mismatch': 0, 'num_visits_mismatch': 0, 'k_mismatch': 0, 'k_checked': 0,
                             'pred_vs_npz_mismatch': 0, 'npz_missing': 0, 'status': {}, 'visit_sets': {}}
        for seed in SEEDS:
            run = explain_dir / 'top20' / f'{folder}_seed{seed}'
            npz = np.load(f'{trained}_seed{seed}/test.npz', allow_pickle=False)
            row_of = {str(s): i for i, s in enumerate(npz['sample_ids'])}
            proba, y = npz['proba'].astype(np.float64), np.asarray(npz['y']).astype(np.int64).reshape(-1)
            manifest = json.loads((run / 'explanation_manifest.json').read_text())
            seen = set()
            for filename in manifest['record_files']:
                record = json.loads((run / filename).read_text())
                sid = str(record['sample_id'])
                if sid in seen:
                    chk['duplicate'] += 1
                seen.add(sid)
                info = graph.get(sid)
                if info is None:
                    chk['unknown_visit'] += 1
                    continue
                if record['true_class_id'] != info['true_class_id']:
                    chk['class_mismatch'] += 1
                if record.get('num_visits') != info['num_visits']:
                    chk['num_visits_mismatch'] += 1
                if str(record['subject_id']) != info['subject']:
                    chk['class_mismatch'] += 1
                npz_row = row_of.get(sid)
                if npz_row is None:
                    chk['npz_missing'] += 1
                    continue
                pred_prob = float(proba[npz_row].max())
                if int(proba[npz_row].argmax()) != record['prediction_class_id']:
                    chk['pred_vs_npz_mismatch'] += 1
                classified = summarize.classify_record(record, f'{key} seed {seed}')
                node_type = info['node_type']
                evidence_mask = np.isin(node_type, [core.NODE_KINDS.index(k) for k in EVIDENCE])
                n_ev = int(evidence_mask.sum())
                for source, short in SOURCES.items():
                    category, metrics, _reason, _extras = classified.get(source, ('absent', None, None, {}))
                    chk['status'].setdefault(short, {}).setdefault(category, 0)
                    chk['status'][short][category] += 1
                    row = {'sample_id': sid, 'key': key, 'seed': seed, 'source': short,
                           'ok': int(metrics is not None), 'status': category,
                           'pred_class': record['prediction_class_id'], 'pred_prob': pred_prob}
                    for name in SCORE_METRICS:
                        row[name] = metrics[name] if metrics and name in metrics else ''
                    scores.append(row)
                    vec = vector(record, source)
                    if vec is None:
                        continue
                    chk['vectors'] += 1
                    if len(vec) != node_type.size:
                        chk['length_mismatch'] += 1
                        continue
                    values = np.abs(np.asarray(vec, dtype=float))[evidence_mask]
                    support = int((values > 0).sum())
                    if n_ev == 0 or support == 0:
                        continue
                    k_calc = min(fv2.resolve_k(n_ev), support)
                    k_rec = ((((record.get('fidelity_v2') or {}).get(source) or {}).get('fidelity_plus') or {})
                             .get('k'))
                    if k_rec is not None:
                        chk['k_checked'] += 1
                        chk['k_mismatch'] += int(int(k_rec) != k_calc)
                    k = int(k_rec) if k_rec is not None else k_calc
                    tops.append({'sample_id': sid, 'key': key, 'seed': seed, 'source': short, 'k': k,
                                 **kind_rows(values, node_type[evidence_mask], k)})
            chk['visit_sets'][seed] = seen
            runs[(key, seed)] = (proba, y, row_of)
            say(f'  {key} seed {seed}: {len(seen)} records')
    return scores, tops, runs, checks


# --------------------------------------------------------------------------- tables
def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_tables(graph, scores, tops, runs, binding):
    visits = sorted({r['sample_id'] for r in scores})
    patients = sorted({graph[s]['subject'] for s in visits})
    visit_idx = {s: i for i, s in enumerate(visits)}
    patient_idx = {p: i for i, p in enumerate(patients)}
    names = binding['label_order']
    kinds = list(core.NODE_KINDS)
    rel_names = [rel_col(n) for n in ALL_RELATIONS]
    feature_rows = []
    for sid in visits:
        info = graph[sid]
        row = {'visit_idx': visit_idx[sid], 'patient_idx': patient_idx[info['subject']],
               'true_class_id': info['true_class_id'], 'true_class': names[info['true_class_id']],
               'num_visits': info['num_visits'], 'n_nodes': info['n_nodes'], 'n_edges': info['n_edges'],
               'n_evidence': info['n_evidence']}
        for kind in kinds:
            row['n_kind_' + kind] = info['n_kind_' + kind]
        for col in rel_names:
            row[col] = info[col]
        row['n_prior_dx'], row['density'] = info['n_prior_dx'], info['density']
        for key in KEYS:
            conf, correct, preds = [], [], []
            for seed in SEEDS:
                proba, y, row_of = runs[(key, seed)]
                i = row_of[sid]
                conf.append(float(proba[i].max()))
                correct.append(int(proba[i].argmax() == info['true_class_id']))
                preds.append(int(proba[i].argmax()))
            row['conf_' + key] = sum(conf) / len(conf)
            row['correct_' + key] = sum(correct) / len(correct)
            row['pred_agree_' + key] = int(len(set(preds)) == 1)
        feature_rows.append(row)
    feature_cols = list(feature_rows[0].keys())
    write_csv(DATA / 'visit_features.csv', feature_cols, feature_rows)

    key_order, source_order = list(KEYS), list(SOURCES.values())
    for row in scores:
        row['visit_idx'] = visit_idx[row.pop('sample_id')]
    scores.sort(key=lambda r: (r['visit_idx'], key_order.index(r['key']), SEEDS.index(r['seed']),
                               source_order.index(r['source'])))
    score_cols = ['visit_idx', 'key', 'seed', 'source', 'ok', 'status', 'pred_class', 'pred_prob', *SCORE_METRICS]
    write_csv(DATA / 'visit_scores.csv', score_cols, scores)

    for row in tops:
        row['visit_idx'] = visit_idx[row.pop('sample_id')]
    tops.sort(key=lambda r: (r['visit_idx'], key_order.index(r['key']), SEEDS.index(r['seed']),
                             source_order.index(r['source'])))
    top_cols = (['visit_idx', 'key', 'seed', 'source', 'k'] + ['top_share_' + k for k in EVIDENCE]
                + ['mass_share_' + k for k in EVIDENCE] + ['top_rate_' + k for k in EVIDENCE])
    write_csv(DATA / 'top_kinds.csv', top_cols, tops)
    sizes = {'visit_features': (len(feature_rows), len(feature_cols)), 'visit_scores': (len(scores), len(score_cols)),
             'top_kinds': (len(tops), len(top_cols))}
    return sizes, {'visit_features': feature_cols, 'visit_scores': score_cols, 'top_kinds': top_cols}, \
        len(visits), len(patients)


# --------------------------------------------------------------------------- parity
def stored_means(key):
    explain_dir, _folder, method, _trained = KEYS[key]
    path = Path(str(explain_dir) + '-summary') / 'summary.csv'
    out = {}
    with open(path, newline='') as handle:
        for row in csv.DictReader(handle):
            if row['table'] == 'all_visits' and row['k'] == '20' and row['method'] == method:
                out[row['source']] = row
    return out


def recomputed_means():
    """{(key, short source, metric): mean over seeds of per-run mean over visits}, read back from the written
    visit_scores.csv, with the same arithmetic as summarize.aggregate."""
    values = {}
    with open(DATA / 'visit_scores.csv', newline='') as handle:
        for row in csv.DictReader(handle):
            for metric in PARITY_METRICS:
                if row[metric] != '':
                    values.setdefault((row['key'], row['source'], int(row['seed']), metric), []).append(float(row[metric]))
    out = {}
    for (key, source, seed, metric), vals in values.items():
        out.setdefault((key, source, metric), []).append(sum(vals) / len(vals))
    return {k: sum(v) / len(v) for k, v in out.items()}


def parity_rows():
    recomputed = recomputed_means()
    long_of = {v: k for k, v in SOURCES.items()}
    rows, worst, failing = [], 0.0, 0
    for key in KEYS:
        stored = stored_means(key)
        for short in SOURCES.values():
            summary_row = stored.get(long_of[short])
            for metric in PARITY_METRICS:
                mine = recomputed.get((key, short, metric))
                theirs = None if summary_row is None or summary_row[f'{metric}_mean'] == '' \
                    else float(summary_row[f'{metric}_mean'])
                if mine is None and theirs is None:
                    diff, status = 0.0, 'both empty'
                elif mine is None or theirs is None:
                    diff, status = math.inf, 'ONE SIDE EMPTY'
                else:
                    diff = abs(mine - theirs)
                    status = 'ok' if diff < TOLERANCE else 'FAIL'
                if status in ('FAIL', 'ONE SIDE EMPTY'):
                    failing += 1
                if math.isfinite(diff):
                    worst = max(worst, diff)
                rows.append((key, short, metric, mine, theirs, diff, status))
    return rows, worst, failing


def fmt(value):
    return '' if value is None else f'{value:.12g}'


def write_parity(checks, graph_n, sizes, rows, worst, failing, stopped):
    lines = ['# Step 1: checks and parity', '',
             'Task K=20, primary_dx_v4, fair_v10t, TEST fold. Aggregates only; counts of 1..9 are shown as <10.', '',
             '## Visit sets and node counts (task 1.2)', '',
             f'Graph structure comes from `rebuild_split` of the cei_final seed-1234 binding ({graph_n} test rows '
             'decoded). Each record vector (builtin + successful GraphXAI explainers) is compared with the node count '
             'of its visit.', '',
             '| key | seed | records | same visit set as reference | importance vectors checked | length != node count | '
             'unknown visit | duplicate | class/subject mismatch | num_visits mismatch | not in test.npz | '
             'record prediction != argmax(test.npz) | record k != recomputed k (of checked) |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    reference = checks['gsat']['visit_sets'][SEEDS[0]]
    for key, chk in checks.items():
        for seed in SEEDS:
            same = chk['visit_sets'][seed] == reference
            cells = [key, seed, len(chk['visit_sets'][seed]), 'yes' if same else 'NO']
            if seed == SEEDS[0]:
                cells += [f'{chk["vectors"]} (all seeds)', f'{small(chk["length_mismatch"])} (all seeds)',
                          small(chk['unknown_visit']), small(chk['duplicate']), small(chk['class_mismatch']),
                          small(chk['num_visits_mismatch']), small(chk['npz_missing']),
                          small(chk['pred_vs_npz_mismatch']),
                          f'{small(chk["k_mismatch"])} of {chk["k_checked"]} (all seeds)']
            else:
                cells += [''] * 9
            lines.append('| ' + ' | '.join(str(c) for c in cells) + ' |')
    lines += ['', f'Visits: {len(reference)} per run; patients in the visit set: see report.', '']
    if stopped:
        lines += ['**STOPPED**: a key has importance vectors whose length differs from the visit node count, or its '
                  'visit set differs from the reference; the CSV tables were not written.', '']
    else:
        lines += ['## Parity with the stored summaries (task 1.3)', '',
                  f'Recomputed = mean over the 3 seeds of the per-run mean over the visits that carry the metric, read '
                  f'back from the written `visit_scores.csv`. Stored = `<metric>_mean` of the `all_visits` row of the '
                  f'run dir\'s `-summary/summary.csv`. Tolerance: absolute difference < {TOLERANCE:g}.', '',
                  f'Pairs compared: {len(rows)}. Failing pairs: {failing}. Largest absolute difference: {worst:.3e}.', '',
                  '| key | source | metric | recomputed | stored | abs diff | status |', '|---|---|---|---|---|---|---|']
        for key, source, metric, mine, theirs, diff, status in rows:
            lines.append(f'| {key} | {source} | {metric} | {fmt(mine)} | {fmt(theirs)} | '
                         f'{"inf" if diff == math.inf else f"{diff:.3e}"} | {status} |')
    (DATA / 'parity.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


# --------------------------------------------------------------------------- report
def write_report(checks, sizes, columns, n_visits, n_patients, rows, worst, failing, stopped, reverse_edges):
    runtime = time.time() - STARTED
    lines = ['# Step 1 report: per-visit tables and parity', '',
             f'Built by `comparison/explain/graph_types/step1_visit_tables.py` (runtime {runtime:.0f} s). Aggregates only; counts of 1..9 are '
             'shown as <10. Test fold, descriptive.', '']
    if stopped:
        lines += ['**The build stopped at the node-count / visit-set check; no CSV table was written.** Details in '
                  '`data/parity.md`.', '']
    else:
        lines += ['## Table sizes', '', '| file | rows | columns |', '|---|---|---|']
        for name, (r, c) in sizes.items():
            lines.append(f'| data/{name}.csv | {r} | {c} |')
        lines += ['', f'Visits: {n_visits} (visit_idx 0..{n_visits - 1}); patients: {n_patients}; keys: {len(KEYS)}; '
                  f'seeds: {len(SEEDS)}; sources: {len(SOURCES)}. Reverse-relation edges in the graphs: {reverse_edges} '
                  '(the cei_final binding uses forward edges).', '', '## Columns', '']
        for name, cols in columns.items():
            lines += [f'**{name}.csv**: ' + ', '.join(f'`{c}`' for c in cols), '']
        lines += ['`visit_scores.status` is the summarize category of the source on that visit: explained, unsupported, '
                  'failed_explainer, failed_fidelity, unavailable, no_eligible, or absent (source not in the record); '
                  '`ok` = 1 exactly when it is `explained`. `top_kinds` ties at the top-k cut share the remaining '
                  'slots equally (expectation over random tie orders); `k` is the record\'s fidelity+ k, with the '
                  'recomputed k checked in `data/parity.md`.', '',
                  '## Dropped / not-ok visit counts per key and source', '',
                  'Counts over the 3 seeds x 1345 visits of each status; the last two columns count visits (of 1345) '
                  'that are ok in at least 2 and in all 3 seeds (the inclusion rule of the analysis).', '',
                  '| key | source | explained | unsupported | failed explainer | failed fidelity | unavailable | '
                  'no eligible | absent | visits ok in >=2 seeds | visits ok in 3 seeds |',
                  '|---|---|---|---|---|---|---|---|---|---|---|']
        ok_counts = {}
        with open(DATA / 'visit_scores.csv', newline='') as handle:
            for row in csv.DictReader(handle):
                ok_counts[(row['key'], row['source'], row['visit_idx'])] = \
                    ok_counts.get((row['key'], row['source'], row['visit_idx']), 0) + int(row['ok'])
        for key in KEYS:
            for short in SOURCES.values():
                st = checks[key]['status'].get(short, {})
                per_visit = [v for (k, s, _), v in ok_counts.items() if k == key and s == short]
                cats = ('explained', 'unsupported', 'failed_explainer', 'failed_fidelity', 'unavailable',
                        'no_eligible', 'absent')
                lines.append(f'| {key} | {short} | ' + ' | '.join(small(st.get(c, 0)) for c in cats)
                             + f' | {small(sum(v >= 2 for v in per_visit))} | {small(sum(v == 3 for v in per_visit))} |')
        lines += ['', '## Parity summary', '',
                  f'{len(rows)} (key, source, metric) pairs compared with the stored `summary.csv` values; failing '
                  f'pairs: {failing}; largest absolute difference: {worst:.3e} (tolerance {TOLERANCE:g}). Full table in '
                  '`data/parity.md`.', '']
    lines += ['## Node-count and visit-set checks', '',
              f'All 9 keys x 3 seeds hold {NUM_VISITS_EXPECTED} records. Importance vectors checked against the node '
              'count of their visit: ' + ', '.join(f'{k} {c["vectors"]} (mismatch {small(c["length_mismatch"])})'
                                                     for k, c in checks.items()) + '.', '',
              '## Command', '',
              '`env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 comparison/explain/graph_types/step1_visit_tables.py`', '']
    if stopped:
        lines.append('STATUS: incomplete: node-count or visit-set check failed (see data/parity.md)')
    elif failing:
        lines.append(f'STATUS: incomplete: {failing} parity pairs differ by >= {TOLERANCE:g} (largest {worst:.3e})')
    else:
        lines.append('STATUS: complete')
    (OUT / 'step1-report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    graph, binding, reverse_edges = graph_features()
    say('reading records and test.npz of 9 keys x 3 seeds')
    scores, tops, runs, checks = read_runs(graph)
    reference = checks['gsat']['visit_sets'][SEEDS[0]]
    stopped = (any(c['length_mismatch'] or c['unknown_visit'] or c['duplicate'] or c['npz_missing']
                   for c in checks.values())
               or any(c['visit_sets'][s] != reference for c in checks.values() for s in SEEDS)
               or len(reference) != NUM_VISITS_EXPECTED)
    say(f'checks: {sum(c["vectors"] for c in checks.values())} vectors, length mismatches '
        f'{sum(c["length_mismatch"] for c in checks.values())}, visit sets equal: '
        f'{all(c["visit_sets"][s] == reference for c in checks.values() for s in SEEDS)}, stopped: {stopped}')
    if stopped:
        write_parity(checks, len(graph), {}, [], 0.0, 0, True)
        write_report(checks, {}, {}, len(reference), 0, [], 0.0, 0, True, reverse_edges)
        say('STOPPED: see data/parity.md')
        return 2
    sizes, columns, n_visits, n_patients = build_tables(graph, scores, tops, runs, binding)
    for name, (r, c) in sizes.items():
        say(f'{name}.csv: {r} rows x {c} columns')
    say(f'visits {n_visits}, patients {n_patients}')
    rows, worst, failing = parity_rows()
    say(f'parity: {len(rows)} pairs, failing {failing}, largest abs diff {worst:.3e}')
    write_parity(checks, len(graph), sizes, rows, worst, failing, False)
    write_report(checks, sizes, columns, n_visits, n_patients, rows, worst, failing, False, reverse_edges)
    say(f'runtime {time.time() - STARTED:.0f} s')
    return 1 if failing or n_visits != NUM_VISITS_EXPECTED or n_patients != NUM_PATIENTS_EXPECTED else 0


if __name__ == '__main__':
    sys.exit(main())
