"""Limitation counts: one read-only streaming pass over the final graph artifact, aggregated counts only.

Usage: python3 -m comparison.s9_repro.scan_graphs --artifact DIR --split comparison/canonical_split.json --out NEW_DIR
                                                  [--limit N] [--execute] [--compare REF_graph_scan.json]
Without --execute only the plan is printed and graphs.jsonl is not read. With --execute the artifact is read once and
OUT/graph_scan.json and OUT/graph_scan.md are written (OUT must not exist). Nothing in the outputs is a sample id,
subject id or per-row value; the fold of a graph is the `split` field stored in the graph (train / validation / test).

Counted, overall and per fold:
 - duplicate edges: exact copies of (source, target, relation), per relation, graphs affected, multiplicity histogram;
   comorbid_with copies; medical:indicates copies (same source/target, split by distinct vs repeated rule_id)
 - censored measurement nodes ('<' / '>') by sign, scope (index / prior), kind, top lab tokens with censored share
 - edges flagged delta_is_bound; medical:interacts_with edge count and every relation's total (unused relations show 0)
 - nodes with total degree >= 10 by kind and the share covered by the message-passing hub kinds (visit, complaint, patient)
--compare REF: print every difference against an earlier scan's graph_scan.json (flat counts, relation totals,
duplicate histogram, degree table, censored top tokens).
"""
import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

try:
    import orjson
    loads = orjson.loads
except ImportError:  # pure-json fallback, about 2-3x slower
    loads = json.loads

GRAPHS = 'graphs.jsonl'
FOLDS = ('train', 'validation', 'test')
HUB_KINDS = ('visit', 'complaint', 'patient')
DEG_MIN = 10
MEASURE_KINDS = ('measurement', 'vital')
EXPECTED_RELATIONS = ('medical:interacts_with',)  # reported even when absent (zero)
BYTES_PER_SEC_ORJSON = 170e6  # measured once on this artifact (about 6.4 GB in 37 s)
BYTES_PER_SEC_JSON = 55e6


def rel(path):
    p = Path(path).resolve()
    try:
        return str(p.relative_to(Path.cwd().resolve()))
    except ValueError:
        return p.name


def scan(path, limit):
    C = collections.Counter()
    dup_hist = collections.Counter()
    cens_tok = collections.Counter()
    meas_tok = collections.Counter()
    cens_tok_fold = collections.Counter()
    meas_tok_fold = collections.Counter()
    rel_total = collections.Counter()
    rel_total_fold = collections.Counter()
    deg10 = collections.Counter()
    deg10_fold = collections.Counter()
    t0 = time.time()
    with open(path, 'rb') as f:
        for line in f:
            if limit and C['graphs'] >= limit:
                break
            g = loads(line)
            sp = g['split']
            C['graphs'] += 1
            C[f'graphs:{sp}'] += 1
            nodes = g['nodes']
            edges = g['edges']
            kind = {n['id']: n['kind'] for n in nodes}
            seen = collections.Counter()
            deg = collections.Counter()
            gdup = collections.Counter()
            rules = collections.defaultdict(set)
            for e in edges:
                r = e['relation']
                rel_total[r] += 1
                rel_total_fold[(sp, r)] += 1
                key = (e['source'], e['target'], r)
                seen[key] += 1
                if r == 'medical:indicates':
                    rules[key].add(e.get('rule_id'))
                deg[e['source']] += 1
                deg[e['target']] += 1
                if e.get('delta_is_bound'):
                    C['edges_delta_is_bound'] += 1
                    C[f'edges_delta_is_bound:{sp}'] += 1
            C['edges'] += len(edges)
            C[f'edges:{sp}'] += len(edges)
            extra = 0
            for k, c in seen.items():
                if c > 1:
                    extra += c - 1
                    gdup[k[2]] += c - 1
                    dup_hist[(k[2], c)] += 1
                    dup_hist[(k[2], c, sp)] += 1
                    if k[2] == 'medical:indicates':
                        # same source/target, different rule_id (distinct rules) vs the very same rule repeated
                        if len(rules[k]) > 1:
                            C['indicates_dup_pairs_distinct_rule'] += 1
                            C[f'indicates_dup_pairs_distinct_rule:{sp}'] += 1
                        else:
                            C['indicates_dup_pairs_same_rule'] += 1
                            C[f'indicates_dup_pairs_same_rule:{sp}'] += 1
            if extra:
                C['graphs_with_dup_edge'] += 1
                C[f'graphs_with_dup_edge:{sp}'] += 1
                C['extra_edge_copies'] += extra
                C[f'extra_edge_copies:{sp}'] += extra
                for r, c in gdup.items():
                    C[f'extra_copies_rel:{r}'] += c
                    C[f'extra_copies_rel:{r}:{sp}'] += c
                    C[f'graphs_with_dup_rel:{r}'] += 1
                    C[f'graphs_with_dup_rel:{r}:{sp}'] += 1
            if any(k[2] == 'comorbid_with' for k in seen):
                C['graphs_with_comorbid'] += 1
                C[f'graphs_with_comorbid:{sp}'] += 1
            for nid, d in deg.items():
                if d >= DEG_MIN:
                    deg10[kind[nid]] += 1
                    deg10_fold[(sp, kind[nid])] += 1
            gc = 0
            nm = 0
            for n in nodes:
                if n['kind'] in MEASURE_KINDS:
                    nm += 1
                    key = n['kind'] + ':' + str(n.get('scope'))
                    C[f'nodes:{key}'] += 1
                    C[f'nodes:{n["kind"]}'] += 1
                    tok = n.get('token')
                    meas_tok[tok] += 1
                    meas_tok_fold[(sp, tok)] += 1
                    cs = n.get('censored')
                    if cs is not None:
                        C[f'censored:{cs}'] += 1
                        C[f'censored:{cs}:{sp}'] += 1
                        C[f'censored_nodes:{n["kind"]}'] += 1
                        C[f'censored_nodes:{key}'] += 1
                        C[f'censored_nodes:{key}:{sp}'] += 1
                        C[f'censored_nodes:split:{sp}'] += 1
                        C[f'censored_level:{n.get("level")}'] += 1
                        cens_tok[(tok, cs)] += 1
                        cens_tok_fold[(sp, tok, cs)] += 1
                        gc += 1
            C[f'meas_nodes_split:{sp}'] += nm
            if gc:
                C['graphs_with_censored'] += 1
                C[f'graphs_with_censored:{sp}'] += 1
                C['censored_nodes_total'] += gc
            if C['graphs'] % 20000 == 0:
                print(C['graphs'], round(time.time() - t0), 's', flush=True)
    return dict(C=C, dup_hist=dup_hist, cens_tok=cens_tok, meas_tok=meas_tok, cens_tok_fold=cens_tok_fold,
                meas_tok_fold=meas_tok_fold, rel_total=rel_total, rel_total_fold=rel_total_fold, deg10=deg10,
                deg10_fold=deg10_fold, seconds=round(time.time() - t0))


def hub_share(deg):
    total = sum(deg.values())
    hub = sum(deg.get(k, 0) for k in HUB_KINDS)
    return {'nodes_deg_ge10': total, 'in_hub_kinds': hub, 'share': round(hub / total, 4) if total else None}


def build_report(s, a, size, n_subjects):
    C = s['C']
    folds = [fo for fo in FOLDS if C[f'graphs:{fo}']]
    relations = sorted(set(s['rel_total']) | set(EXPECTED_RELATIONS))
    top_tokens = []
    for (tok, sign), c in s['cens_tok'].most_common(15):
        top_tokens.append({'token': str(tok), 'sign': sign, 'censored': c, 'token_nodes': s['meas_tok'][tok],
                           'censored_share': round(c / s['meas_tok'][tok], 4)})

    def tok_share(tok, fold=None):
        if fold is None:
            c = sum(v for (t, _), v in s['cens_tok'].items() if t == tok)
            n = s['meas_tok'][tok]
        else:
            c = sum(v for (f, t, _), v in s['cens_tok_fold'].items() if f == fold and t == tok)
            n = s['meas_tok_fold'][(fold, tok)]
        return {'censored': c, 'token_nodes': n, 'censored_share': round(c / n, 4) if n else None}

    def per_fold(prefix):
        return {fo: C[f'{prefix}:{fo}'] for fo in folds}

    rep = {
        'task_identity': {
            'artifact': rel(a.artifact), 'graphs': C['graphs'], 'graphs_by_fold': per_fold('graphs'),
            'edges': C['edges'], 'limit': a.limit or None, 'fold_source': 'split field of each graph',
            'canonical_split': rel(a.split), 'canonical_split_subjects_by_fold_index': n_subjects,
            'reported_only_not_used_for_decisions': 'all counts describe the stored graphs; no model selection uses them'},
        'duplicate_edges': {
            'any_duplicate_edge': {'graphs': C['graphs_with_dup_edge'], 'extra_copies': C['extra_edge_copies'],
                                   'graphs_by_fold': per_fold('graphs_with_dup_edge'),
                                   'extra_copies_by_fold': per_fold('extra_edge_copies')},
            'by_relation': {r: {'extra_copies': C[f'extra_copies_rel:{r}'], 'graphs': C[f'graphs_with_dup_rel:{r}'],
                                'extra_copies_by_fold': {fo: C[f'extra_copies_rel:{r}:{fo}'] for fo in folds},
                                'graphs_by_fold': {fo: C[f'graphs_with_dup_rel:{r}:{fo}'] for fo in folds}}
                            for r in sorted({k[0] for k in s['dup_hist'] if len(k) == 2})},
            'comorbid_with': {'extra_copies': C['extra_copies_rel:comorbid_with'],
                              'graphs': C['graphs_with_dup_rel:comorbid_with'],
                              'graphs_with_any_comorbid_edge': C['graphs_with_comorbid']},
            'medical_indicates': {'extra_copies': C['extra_copies_rel:medical:indicates'],
                                  'graphs': C['graphs_with_dup_rel:medical:indicates'],
                                  'duplicate_pairs_distinct_rule_id': C['indicates_dup_pairs_distinct_rule'],
                                  'duplicate_pairs_same_rule_id': C['indicates_dup_pairs_same_rule']},
            'multiplicity_histogram': {f'{k[0]}|x{k[1]}': v for k, v in sorted(s['dup_hist'].items()) if len(k) == 2},
            'multiplicity_histogram_by_fold': {fo: {f'{k[0]}|x{k[1]}': v for k, v in sorted(s['dup_hist'].items())
                                                    if len(k) == 3 and k[2] == fo} for fo in folds}},
        'censored_measurements': {
            'measurement_and_vital_nodes': C['nodes:measurement'] + C['nodes:vital'],
            'censored_nodes': C['censored_nodes_total'],
            'graphs_with_censored': C['graphs_with_censored'],
            'graphs_with_censored_share': round(C['graphs_with_censored'] / C['graphs'], 4),
            'by_sign': {'<': C['censored:<'], '>': C['censored:>'],
                        **{f'{sg}:{fo}': C[f'censored:{sg}:{fo}'] for sg in ('<', '>') for fo in folds}},
            'by_scope': {sc: C[f'censored_nodes:measurement:{sc}'] for sc in ('index', 'prior')},
            'by_scope_and_fold': {f'{sc}:{fo}': C[f'censored_nodes:measurement:{sc}:{fo}']
                                  for sc in ('index', 'prior') for fo in folds},
            'censored_nodes_by_fold': per_fold('censored_nodes:split'),
            'graphs_with_censored_by_fold': per_fold('graphs_with_censored'),
            'measurement_nodes_by_fold': per_fold('meas_nodes_split'),
            'top_tokens': top_tokens,
            'troponin_lab_51003': {'all_folds': tok_share('lab:51003'), **{fo: tok_share('lab:51003', fo) for fo in folds}}},
        'delta_is_bound_edges': {'total': C['edges_delta_is_bound'], 'by_fold': per_fold('edges_delta_is_bound')},
        'relations': {'total_edges': {r: s['rel_total'].get(r, 0) for r in relations},
                      'by_fold': {fo: {r: s['rel_total_fold'].get((fo, r), 0) for r in relations} for fo in folds},
                      'interacts_with_edges': s['rel_total'].get('medical:interacts_with', 0),
                      'zero_edge_relations': [r for r in relations if not s['rel_total'].get(r, 0)]},
        'hub_coverage': {'degree_min': DEG_MIN, 'hub_kinds': list(HUB_KINDS), 'degree': 'source + target edge endpoints per graph',
                         'nodes_by_kind': dict(s['deg10']), **hub_share(s['deg10']),
                         'by_fold': {fo: {'nodes_by_kind': {k: v for (f, k), v in s['deg10_fold'].items() if f == fo},
                                          **hub_share({k: v for (f, k), v in s['deg10_fold'].items() if f == fo})}
                                     for fo in folds}},
        'scan_seconds': s['seconds'],
        'input_bytes': size,
    }
    # flat view with the same key names as the earlier scan, used by --compare
    rep['flat'] = {'counts': dict(C), 'relation_total_edges': dict(s['rel_total']),
                   'dup_multiplicity_hist': rep['duplicate_edges']['multiplicity_histogram'],
                   'deg_ge10_nodes_by_kind': dict(s['deg10']),
                   'censored_top_tokens': [[t['token'], t['sign'], t['censored'], t['token_nodes']] for t in top_tokens]}
    return rep


def render_md(rep):
    ti, d, c, h = rep['task_identity'], rep['duplicate_edges'], rep['censored_measurements'], rep['hub_coverage']
    fo = list(ti['graphs_by_fold'])
    L = ['# Limitation counts of the final graphs', '',
         f"Artifact `{ti['artifact']}`: {ti['graphs']:,} graphs, {ti['edges']:,} edges; folds = graph `split` field "
         f"({', '.join(f'{f} {n:,}' for f, n in ti['graphs_by_fold'].items())}). Counts only, reported only, "
         'not used for decisions.', '', '## Duplicate edges', '',
         '| scope | extra copies | graphs | ' + ' | '.join(f'graphs {f}' for f in fo) + ' |',
         '|---|---|---|' + '---|' * len(fo)]
    a = d['any_duplicate_edge']
    L.append(f"| any relation | {a['extra_copies']:,} | {a['graphs']:,} | " + ' | '.join(f"{a['graphs_by_fold'][f]:,}" for f in fo) + ' |')
    for r, v in d['by_relation'].items():
        L.append(f"| {r} | {v['extra_copies']:,} | {v['graphs']:,} | " + ' | '.join(f"{v['graphs_by_fold'][f]:,}" for f in fo) + ' |')
    mi = d['medical_indicates']
    L += ['', f"medical:indicates duplicate pairs: {mi['duplicate_pairs_distinct_rule_id']:,} with different rule_id, "
              f"{mi['duplicate_pairs_same_rule_id']:,} with the same rule_id. Multiplicity histogram: "
              + ', '.join(f'{k} = {v:,}' for k, v in d['multiplicity_histogram'].items()) + '.', '',
          '## Censored measurement values', '',
          f"{c['censored_nodes']:,} of {c['measurement_and_vital_nodes']:,} measurement/vital nodes are censored, in "
          f"{c['graphs_with_censored']:,} graphs ({100 * c['graphs_with_censored_share']:.1f}% of graphs). "
          f"By sign: `<` {c['by_sign']['<']:,}, `>` {c['by_sign']['>']:,}. By scope: index {c['by_scope']['index']:,}, "
          f"prior {c['by_scope']['prior']:,}.", '',
          '| fold | censored nodes | graphs with censored | measurement nodes |', '|---|---|---|---|']
    for f in fo:
        L.append(f"| {f} | {c['censored_nodes_by_fold'][f]:,} | {c['graphs_with_censored_by_fold'][f]:,} | {c['measurement_nodes_by_fold'][f]:,} |")
    L += ['', '| token | sign | censored | token nodes | censored share |', '|---|---|---|---|---|']
    for t in c['top_tokens']:
        L.append(f"| {t['token']} | {t['sign']} | {t['censored']:,} | {t['token_nodes']:,} | {100 * t['censored_share']:.1f}% |")
    tp = c['troponin_lab_51003']
    L += ['', 'Troponin (lab:51003) censored share: ' + ', '.join(
        f"{k} {100 * v['censored_share']:.1f}% ({v['censored']:,}/{v['token_nodes']:,})" for k, v in tp.items() if v['censored_share'] is not None) + '.',
          '', '## Edges flagged delta_is_bound and relation totals', '',
          f"delta_is_bound edges: {rep['delta_is_bound_edges']['total']:,} ("
          + ', '.join(f'{f} {n:,}' for f, n in rep['delta_is_bound_edges']['by_fold'].items()) + ').',
          f"medical:interacts_with edges: {rep['relations']['interacts_with_edges']:,}. Relations with zero edges: "
          + (', '.join(rep['relations']['zero_edge_relations']) or 'none') + '.', '',
          '| relation | edges | ' + ' | '.join(fo) + ' |', '|---|---|' + '---|' * len(fo)]
    for r, n in rep['relations']['total_edges'].items():
        L.append(f'| {r} | {n:,} | ' + ' | '.join(f"{rep['relations']['by_fold'][f][r]:,}" for f in fo) + ' |')
    L += ['', '## Hub-kind coverage', '',
          f"Nodes with total degree >= {h['degree_min']}, covered by the message-passing hub kinds "
          f"({', '.join(h['hub_kinds'])}): {h['in_hub_kinds']:,} of {h['nodes_deg_ge10']:,} ({100 * h['share']:.1f}%).", '',
          '| scope | ' + ' | '.join(sorted(h['nodes_by_kind'])) + ' | hub share |',
          '|---|' + '---|' * (len(h['nodes_by_kind']) + 1)]
    kinds = sorted(h['nodes_by_kind'])
    L.append('| all | ' + ' | '.join(f"{h['nodes_by_kind'][k]:,}" for k in kinds) + f" | {100 * h['share']:.1f}% |")
    for f in fo:
        b = h['by_fold'][f]
        L.append(f'| {f} | ' + ' | '.join(f"{b['nodes_by_kind'].get(k, 0):,}" for k in kinds) + f" | {100 * b['share']:.1f}% |")
    return '\n'.join(L) + '\n'


def compare(flat, ref_path):
    """Print every difference between this scan and an earlier graph_scan.json; returns the number of differences."""
    ref = json.load(open(ref_path))
    diffs = 0
    for section in ('counts', 'relation_total_edges', 'dup_multiplicity_hist', 'deg_ge10_nodes_by_kind'):
        mine, theirs = flat[section], ref.get(section, {})
        for k in sorted(set(theirs)):
            ok = mine.get(k, 0) == theirs[k]
            print(f"parity {section}.{k}: {'equal' if ok else 'DIFFERENT'} ref={theirs[k]} new={mine.get(k, 0)}")
            diffs += not ok
    mine_t, ref_t = flat['censored_top_tokens'], ref.get('censored_top_tokens', [])
    ok = mine_t == ref_t
    print(f"parity censored_top_tokens: {'equal' if ok else 'DIFFERENT'} (ties may reorder) ref={ref_t[:3]}... new={mine_t[:3]}...")
    diffs += not ok
    extra = sorted(set(flat['counts']) - set(ref.get('counts', {})))
    print(f'parity: {len(extra)} keys only in the new scan (finer per-fold splits, not compared)')
    print(f'parity differences: {diffs}')
    return diffs


def main(argv=None):
    ap = argparse.ArgumentParser(prog='comparison.s9_repro.scan_graphs', description=__doc__.split('\n')[0])
    ap.add_argument('--artifact', required=True, help='graph directory holding graphs.jsonl (read only)')
    ap.add_argument('--split', required=True, help='canonical split json (key fold: subject -> fold index), read only')
    ap.add_argument('--out', required=True, help='new output directory; an existing one is refused')
    ap.add_argument('--limit', type=int, default=0, help='scan only the first N graphs (0 = all)')
    ap.add_argument('--execute', action='store_true', help='read graphs.jsonl and write the outputs; absent = plan only')
    ap.add_argument('--compare', default=None, help='graph_scan.json of an earlier scan to print differences against')
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        sys.exit(f'refusing existing --out: {rel(out)}')
    path = Path(a.artifact) / GRAPHS
    if not path.is_file():
        sys.exit(f'missing {rel(path)}')
    size = os.path.getsize(path)
    speed = BYTES_PER_SEC_ORJSON if loads is not json.loads else BYTES_PER_SEC_JSON
    print(f'plan: scan {rel(path)} once, read only: {size / 1e9:.2f} GB, '
          f"limit {a.limit or 'none'}, parser {'orjson' if loads is not json.loads else 'json'}, "
          f'expected pass time about {round(size / speed)} s (about {round(size / speed / 60, 1)} min)')
    print(f'plan: write {rel(out / "graph_scan.json")} and {rel(out / "graph_scan.md")}; counts only, no ids')
    if not a.execute:
        print('plan only (no --execute): graphs.jsonl was not read')
        return 0
    split = json.load(open(a.split))
    n_subjects = dict(sorted(collections.Counter(str(v) for v in split['fold'].values()).items()))
    s = scan(path, a.limit)
    rep = build_report(s, a, size, n_subjects)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'graph_scan.json').write_text(json.dumps(rep, indent=1, sort_keys=True) + '\n')
    (out / 'graph_scan.md').write_text(render_md(rep))
    print(f'done in {rep["scan_seconds"]} s: {rel(out / "graph_scan.json")}, {rel(out / "graph_scan.md")}')
    if a.compare:
        compare(rep['flat'], a.compare)
    return 0


if __name__ == '__main__':
    sys.exit(main())
