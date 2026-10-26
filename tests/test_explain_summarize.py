"""Aggregate explanation tables from synthetic schema-3 runner outputs in the runner's layout (numbers checked by hand)."""
import csv
import json
import math

import pytest

from comparison.explain import summarize

GX_NAMES = ('GradExplainer', 'GNNExplainer', 'IntegratedGradExplainer', 'IntegratedGradExplainerZeroBaseline')


def source(plus, minus, dauc, iauc, sparsity, vs=None, fraction=0.5, support=None, ties=None, tied_at_cut=None):
    # mean_change = AUC / explained fraction, as the curves record it
    entry = {'fidelity_plus': {'prob': plus}, 'fidelity_minus': {'prob': minus},
             'deletion_curve': {'auc': dauc, 'mean_change': dauc / fraction, 'explained_fraction': fraction},
             'insertion_curve': {'auc': iauc, 'mean_change': iauc / fraction, 'explained_fraction': fraction},
             'sparsity': {'mass90_sparsity': sparsity}}
    if support is not None:
        entry['sparsity']['support_fraction'] = support
    if ties is not None:
        entry['tie_summary'] = ties
    if tied_at_cut is not None:
        entry['fidelity_plus']['tied_at_cut'] = tied_at_cut
    if vs is not None:
        entry['vs_random'] = {key: {'difference': vs, 'beats_random_fraction': 0.5} for key in
                              ('fidelity_plus', 'fidelity_minus', 'deletion_auc', 'insertion_auc',
                               'deletion_mean_change', 'insertion_mean_change')}
    return entry


def gx_payload(name, entry):
    """The `graphxai` entry that goes with a `fidelity_v2` entry."""
    return {'status': 'success', 'provenance': {'implementation': 'core.explain.ig_clinical', 'completeness_met': True}
            if name == 'IntegratedGradExplainer' else {}}


DISCLOSURE = {'intervention': 'node deletion with concept cascade', 'builtin': 'the model\'s own explanation',
              'explainers': 'GradExplainer, IntegratedGradExplainer, GNNExplainer',
              'caveats': ['CEI pairs and absence states are not reachable by edge masks']}
SETTINGS = {'curve_points': 10, 'random_repeats': 50, 'tie_draws': 10, 'gnnexplainer_epochs': 50,
            'gnnexplainer_seed_rule': 'per visit', 'gnnexplainer_feature_penalty': 'mean',
            'ig': {'tolerance': 0.05, 'step_candidates': [8, 16, 32], 'steps_chosen': 16}}


def make_record(index, fidelity, extra=None, *, schema=3, version=6):
    record = {'schema_version': schema, 'sample_id': f'secret{index}', 'subject_id': f'patient{index}',
              'builtin_status': 'available', 'builtin_tie_summary': None,
              'graphxai': {n: gx_payload(n, e) for n, e in fidelity.items() if n != 'builtin'},
              'fidelity_v2': fidelity}
    if version is not None:
        record['fidelity_v2_intervention'] = {'version': version}
    record.update(extra or {})
    return record


def write_run(root, k, name, visits, *, method='cei_gnn_mp2', fold='test', failed=0, role=None, final_eval='test',
              version=6, binding_version='same', record_versions=None, schema=3, protocol='fair_v10t', code='A',
              settings=None, disclosure=None, projection=None, sample=500, overrides=(), model_k=None,
              model_seed=None, replay='exact_hash', with_code=True, with_disclosure=True):
    """visits: a fidelity dict per record, or (fidelity dict, extra record keys)."""
    run_dir = root / f'top{k}' / name
    run_dir.mkdir(parents=True)
    seed = int(name.rsplit('seed', 1)[1])
    files = []
    for i, visit in enumerate(visits):
        fidelity, extra = visit if isinstance(visit, tuple) else (visit, None)
        version_i = version if record_versions is None else record_versions[i]
        filename = f'subject_secret{k}{name}{i}.json'
        (run_dir / filename).write_text(json.dumps(make_record(i, fidelity, extra, schema=schema, version=version_i)))
        files.append(filename)
    manifest = {'method': method, 'fold': fold, 'explained_count': len(visits), 'failed_count': failed,
                'failures': [{'sample_id': 'secret9', 'error': 'boom'}] * failed, 'record_files': files,
                'run_dir': 'top/x', 'replay': {'mode': replay}, 'settings': settings or SETTINGS,
                'prototype_projection': projection}
    if with_disclosure:
        manifest['disclosure'] = {method: disclosure or DISCLOSURE}
    if with_code:
        manifest['code'] = {'source_hashes': {'core/a.py': code}, 'vendored_graphxai_tree_sha256': 'v',
                            'git_head': 'abc1234', 'git_dirty': False}
    if role:
        manifest['fold_role'] = role
    (run_dir / 'explanation_manifest.json').write_text(json.dumps(manifest))
    bound = version if binding_version == 'same' else binding_version
    explainer = {} if bound is None else {'intervention_contract_version': bound}
    (run_dir / 'binding.json').write_text(json.dumps({
        'model_run': {'final_eval': final_eval, 'protocol': protocol, 'top_k_labels': model_k or k,
                      'seed': model_seed or seed, 'method': method, 'protocol_overrides': list(overrides)},
        'cohort': {'sampling': {'sample_patients': sample, 'sample_seed': 1234, 'max_subjects': None}},
        'explainer_config': explainer}))


def driver_manifest(root, runs, *, complete=True, fold='test', protocol='fair_v10t', sample=500):
    """runs: (k, method, seed, returncode) tuples."""
    (root / 'driver_manifest.json').write_text(json.dumps({
        'protocol': protocol, 'fold': fold, 'complete': complete, 'planned_runs': len(runs),
        'sample_patients': sample,
        'runs': [{'k': k, 'method': m, 'seed': s, 'output': f'top{k}/{m}_seed{s}',
                  'returncode': code} for k, m, s, code in runs]}))


def build(root, **kw):
    # seed 1234: builtin plus .2/.4 (mean .3); seed 2025: .5/.7 (mean .6) -> mean .45, SD .15*sqrt(2)
    write_run(root, 10, 'cei_mp2_seed1234', [
        {'builtin': source(.2, .1, .3, .2, .5, vs=.1), 'GradExplainer': source(.1, .1, .1, .1, .9)},
        ({'builtin': source(.4, .3, .5, .4, .7, vs=.3)},
         {'graphxai': {'GradExplainer': {'status': 'success'}},
          'fidelity_v2': {'builtin': source(.4, .3, .5, .4, .7, vs=.3),
                          'GradExplainer': {'status': 'failed', 'error': 'x'}}})],
        failed=1, **kw)
    write_run(root, 10, 'cei_mp2_seed2025', [
        {'builtin': source(.5, .2, .4, .3, .5, vs=.2), 'GradExplainer': source(.3, .1, .2, .2, .8)},
        {'builtin': source(.7, .4, .6, .5, .9, vs=.4), 'GradExplainer': source(.5, .1, .4, .2, .6)}], **kw)


CEI_RUNS = [(10, 'cei_mp2', 1234, 0), (10, 'cei_mp2', 2025, 0)]
ONE = [{'builtin': source(.2, .1, .3, .2, .5)}]


def rows_of(rows):
    return {(r['k'], r['method'], r['source']): r for r in rows}


def run_main(root, tmp_path, name='out', *extra):
    out = tmp_path / name
    assert summarize.main(['--explanations', str(root), '--out', str(out), *extra]) == 0
    return out


def test_mean_and_sd_over_seeds_by_hand(tmp_path):
    build(tmp_path)
    runs, _ = summarize.collect(tmp_path)
    rows = rows_of(summarize.aggregate(runs))
    builtin = rows[(10, 'cei_gnn_mp2', 'builtin')]
    assert builtin['runs'] == 2 and builtin['visits'] == 4 and builtin['visits_explained'] == 4
    assert builtin['fidelity_plus_mean'] == pytest.approx(0.45)
    assert builtin['fidelity_plus_sd'] == pytest.approx(0.15 * math.sqrt(2))
    assert builtin['fidelity_minus_mean'] == pytest.approx(0.25)
    assert builtin['mass90_sparsity_mean'] == pytest.approx(((0.5 + 0.7) / 2 + (0.5 + 0.9) / 2) / 2)
    assert builtin['deletion_auc_vs_random_mean'] == pytest.approx(((0.1 + 0.3) / 2 + (0.2 + 0.4) / 2) / 2)
    assert builtin['fidelity_plus_beats_random_mean'] == pytest.approx(0.5)
    assert builtin['failed_visits'] == 1 and builtin['visits_failed_fidelity'] == 0
    grad = rows[(10, 'cei_gnn_mp2', 'GradExplainer')]
    # seed 1234 has one failed evaluation: its run mean is the single good visit (.1)
    assert grad['visits'] == 4 and grad['visits_explained'] == 3 and grad['visits_failed_fidelity'] == 1
    assert grad['fidelity_plus_mean'] == pytest.approx((0.1 + (0.3 + 0.5) / 2) / 2)
    assert grad['deletion_auc_vs_random_mean'] is None


def test_single_run_has_no_sd(tmp_path):
    write_run(tmp_path, 20, 'gsat_seed7', ONE, method='gsat')
    runs, _ = summarize.collect(tmp_path)
    row = summarize.aggregate(runs)[0]
    assert row['fidelity_plus_mean'] == pytest.approx(0.2) and row['fidelity_plus_sd'] is None


def test_outputs_name_the_fold_and_hold_no_ids_or_paths(tmp_path):
    build(tmp_path / 'in')
    driver_manifest(tmp_path / 'in', CEI_RUNS)
    out = run_main(tmp_path / 'in', tmp_path)
    assert sorted(p.name for p in out.iterdir()) == ['summary.csv', 'summary.json', 'summary.md']
    text = ''.join((out / name).read_text() for name in ('summary.csv', 'summary.json', 'summary.md'))
    assert 'test (report fold)' in text and 'fair_v10t' in text
    assert '## K = 10 · fold: test (report fold)' in (out / 'summary.md').read_text()
    for forbidden in ('secret', 'patient', '/home/', 'sample_id', 'subject'):
        assert forbidden not in text
    with open(out / 'summary.csv') as handle:
        table = list(csv.DictReader(handle))
    assert {r['fold'] for r in table} == {'test (report fold)'}
    assert [r['table'] for r in table].count('all_visits') == 2 and len(table) == 4
    data = json.loads((out / 'summary.json').read_text())
    assert data['fold_role'] == 'report' and len(data['rows']) == 2 and data['record_schema_version'] == 3
    assert data['run_set_checked_against_driver_plan'] is True


def test_validation_is_labelled_decision_fold(tmp_path):
    build(tmp_path / 'in', fold='validation', role='decision')
    out = run_main(tmp_path / 'in', tmp_path)
    assert 'validation (decision fold)' in (out / 'summary.md').read_text()


def test_refuses_mixed_folds_roles_and_final_eval(tmp_path):
    build(tmp_path / 'folds')
    write_run(tmp_path / 'folds', 10, 'gsat_seed7', ONE, method='gsat', fold='validation')
    with pytest.raises(SystemExit, match='mix folds'):
        summarize.collect(tmp_path / 'folds')
    write_run(tmp_path / 'roles', 10, 'gsat_seed7', ONE, method='gsat', role='decision')
    with pytest.raises(SystemExit, match='fold_role'):
        summarize.collect(tmp_path / 'roles')
    build(tmp_path / 'proto')
    write_run(tmp_path / 'proto', 20, 'gsat_seed7', ONE, method='gsat', final_eval='validation')
    with pytest.raises(SystemExit, match='final_eval'):
        summarize.collect(tmp_path / 'proto')
    build(tmp_path / 'driver')
    driver_manifest(tmp_path / 'driver', CEI_RUNS, fold='validation')
    with pytest.raises(SystemExit, match='mix folds'):
        summarize.collect(tmp_path / 'driver')


def test_filled_out_dir_and_empty_input_are_refused(tmp_path):
    build(tmp_path / 'in')
    (tmp_path / 'out').mkdir()
    (tmp_path / 'out' / 'x').write_text('x')
    with pytest.raises(SystemExit, match='new or empty'):
        summarize.main(['--explanations', str(tmp_path / 'in'), '--out', str(tmp_path / 'out')])
    (tmp_path / 'blank').mkdir()
    with pytest.raises(SystemExit, match='no explanation_manifest'):
        summarize.collect(tmp_path / 'blank')


# ------------------------------------------------------------------ curve columns, versions, sparsity names
def test_both_curve_measures_and_the_explained_fraction_by_hand(tmp_path):
    build(tmp_path)
    runs, _ = summarize.collect(tmp_path)
    builtin = rows_of(summarize.aggregate(runs))[(10, 'cei_gnn_mp2', 'builtin')]
    # seed 1234: AUC .3/.5 -> mean_change .6/1.0 (mean .8); seed 2025: .4/.6 -> .8/1.2 (mean 1.0)
    assert builtin['deletion_mean_change_mean'] == pytest.approx((0.8 + 1.0) / 2)
    assert builtin['deletion_mean_change_sd'] == pytest.approx(0.1 * math.sqrt(2))
    assert builtin['insertion_mean_change_mean'] == pytest.approx(2 * builtin['insertion_auc_mean'])
    assert builtin['deletion_auc_mean'] == pytest.approx((0.4 + 0.5) / 2)
    assert builtin['explained_fraction_mean'] == pytest.approx(0.5)
    assert builtin['deletion_mean_change_vs_random_mean'] == pytest.approx(((0.1 + 0.3) / 2 + (0.2 + 0.4) / 2) / 2)
    assert builtin['insertion_mean_change_beats_random_mean'] == pytest.approx(0.5)


def test_sparsity_is_written_under_both_names_and_the_old_name_is_gone(tmp_path):
    write_run(tmp_path / 'in', 10, 'gsat_seed7',
              [{'builtin': source(.2, .1, .3, .2, .6, support=0.25)}, {'builtin': source(.2, .1, .3, .2, .8, support=0.5)}],
              method='gsat')
    out = run_main(tmp_path / 'in', tmp_path)
    row = json.loads((out / 'summary.json').read_text())['rows'][0]
    assert row['mass90_sparsity_mean'] == pytest.approx(0.7)
    assert row['one_minus_support_fraction_mean'] == pytest.approx((0.75 + 0.5) / 2)
    assert 'sparsity_mean' not in row
    header = next(csv.reader(open(out / 'summary.csv')))
    assert 'mass90_sparsity_mean' in header and 'one_minus_support_fraction_mean' in header
    assert 'sparsity_mean' not in header


def test_markdown_leads_with_the_width_matched_difference_and_has_direction_notes(tmp_path, monkeypatch):
    build(tmp_path / 'in')
    assert summarize.HEADLINE_CURVE == 'mean_change'
    monkeypatch.setattr(summarize, 'HEADLINE_CURVE', 'auc')
    out = run_main(tmp_path / 'in', tmp_path, 'auc')
    text = (out / 'summary.md').read_text()
    header = next(line for line in text.splitlines() if line.startswith('| method'))
    cells = [c.strip() for c in header.strip('|').split('|')]
    assert cells.index('deletion_auc') < cells.index('deletion_mean_change')
    assert {'insertion_auc', 'insertion_mean_change', 'explained_fraction', 'mass90_sparsity',
            'one_minus_support_fraction', 'deletion_mean_change_vs_random', 'insertion_auc_vs_random'} <= set(cells)
    assert text.count('Direction: higher is better for fidelity_plus') == 1      # one note per K table
    assert 'lower is better for fidelity_minus' in text and 'negative insertion difference' in text
    assert 'intervention contract v6' in text
    monkeypatch.setattr(summarize, 'HEADLINE_CURVE', 'mean_change')
    out = run_main(tmp_path / 'in', tmp_path, 'mean')
    header = next(line for line in (out / 'summary.md').read_text().splitlines() if line.startswith('| method'))
    cells = [c.strip() for c in header.strip('|').split('|')]
    first_metric = cells.index('deletion_mean_change_vs_random')
    assert cells[first_metric:first_metric + 2] == ['deletion_mean_change_vs_random',
                                                    'insertion_mean_change_vs_random']
    assert first_metric < cells.index('fidelity_plus') < cells.index('deletion_mean_change') < cells.index('deletion_auc')
    csv_columns = next(csv.reader(open(out / 'summary.csv')))
    assert {'deletion_auc_mean', 'deletion_mean_change_mean', 'explained_fraction_mean'} <= set(csv_columns)
    assert json.loads((out / 'summary.json').read_text())['intervention_contract_version'] == 6


def test_missing_contract_version_is_refused_and_names_the_run(tmp_path):
    write_run(tmp_path / 'record', 10, 'gsat_seed7', ONE, method='gsat', version=None, binding_version=6)
    with pytest.raises(SystemExit, match=r'gsat_seed7.*no intervention contract version'):
        summarize.collect(tmp_path / 'record')
    write_run(tmp_path / 'binding', 10, 'gsat_seed7', ONE, method='gsat', version=6, binding_version=None)
    with pytest.raises(SystemExit, match=r'gsat_seed7.*binding.*no intervention contract version'):
        summarize.collect(tmp_path / 'binding')


def test_disagreeing_versions_are_refused(tmp_path):
    visits = ONE * 2
    write_run(tmp_path / 'records', 10, 'gsat_seed7', visits, method='gsat', record_versions=[5, 6], binding_version=6)
    with pytest.raises(SystemExit, match=r'gsat_seed7.*records disagree.*5, 6'):
        summarize.collect(tmp_path / 'records')
    write_run(tmp_path / 'binding', 10, 'gsat_seed7', visits, method='gsat', version=6, binding_version=5)
    with pytest.raises(SystemExit, match=r'gsat_seed7.*record contract version 6 does not match its binding \(5\)'):
        summarize.collect(tmp_path / 'binding')


@pytest.mark.parametrize('version', [3, 4, 5, 7])
def test_only_contract_version_6_is_read(tmp_path, version):
    write_run(tmp_path, 10, 'gsat_seed7', ONE, method='gsat', version=version)
    with pytest.raises(SystemExit, match=rf'gsat_seed7.*version {version}.*only version 6'):
        summarize.collect(tmp_path)


@pytest.mark.parametrize('schema', [None, 1, 2, 4])
def test_only_record_schema_3_is_read(tmp_path, schema):
    write_run(tmp_path, 10, 'gsat_seed7', ONE, method='gsat', schema=schema)
    with pytest.raises(SystemExit, match=r'gsat_seed7.*schema version.*only schema 3'):
        summarize.collect(tmp_path)


# ------------------------------------------------------------------ driver header, incomplete runs
def test_incomplete_driver_run_is_refused_unless_allowed_and_lists_the_left_out_runs(tmp_path):
    root = tmp_path / 'in'
    build(root)
    write_run(root, 10, 'gsat_seed1234', ONE, method='gsat')
    driver_manifest(root, [(10, 'cei_mp2', 1234, 0), (10, 'cei_mp2', 2025, 0), (10, 'gsat', 1234, 0),
                           (10, 'protgnn', 1234, 3), (10, 'graphcare', 1234, None)], complete=False)
    with pytest.raises(SystemExit, match=r'not complete.*K=10 protgnn seed 1234: return code 3'):
        summarize.collect(root)
    with pytest.raises(SystemExit, match='not complete'):
        summarize.main(['--explanations', str(root), '--out', str(tmp_path / 'refused')])
    assert not (tmp_path / 'refused').exists()
    out = run_main(root, tmp_path, 'out', '--allow-incomplete')
    text = (out / 'summary.md').read_text()
    section = text.split('## Runs not summarised')[1]
    assert '| 10 | protgnn | 1234 | 3 |' in section and '| 10 | graphcare | 1234 | not run |' in section
    assert 'gsat' not in section and 'cei_mp2' not in section
    assert json.loads((out / 'summary.json').read_text())['runs_not_summarised'] == [
        {'k': 10, 'method': 'protgnn', 'seed': 1234, 'returncode': 3},
        {'k': 10, 'method': 'graphcare', 'seed': 1234, 'returncode': None}]


def test_a_failed_run_with_a_stray_manifest_is_left_out_when_incomplete_is_allowed(tmp_path):
    root = tmp_path / 'in'
    build(root)
    write_run(root, 10, 'gsat_seed1234', [{'builtin': source(.9, .9, .9, .9, .9)}], method='gsat')
    driver_manifest(root, CEI_RUNS + [(10, 'gsat', 1234, 1)], complete=False)
    runs, _ = summarize.collect(root, allow_incomplete=True)
    assert sorted(run['method'] for run in runs) == ['cei_gnn_mp2', 'cei_gnn_mp2']


def test_a_complete_driver_run_needs_no_flag_and_writes_no_left_out_table(tmp_path):
    root = tmp_path / 'in'
    build(root)
    driver_manifest(root, CEI_RUNS)
    out = run_main(root, tmp_path)
    assert 'Runs not summarised' not in (out / 'summary.md').read_text()
    assert json.loads((out / 'summary.json').read_text())['runs_not_summarised'] == []
    skipped = tmp_path / 'resumed'
    build(skipped)
    (skipped / 'driver_manifest.json').write_text(json.dumps({
        'protocol': 'fair_v10t', 'fold': 'test', 'complete': True,
        'runs': [{'k': 10, 'method': 'cei_mp2', 'seed': 1234, 'returncode': None, 'status': 'skipped_existing'},
                 {'k': 10, 'method': 'cei_mp2', 'seed': 2025, 'returncode': 0, 'status': 'ok'}]}))
    runs, _ = summarize.collect(skipped)
    assert len(runs) == 2


# ------------------------------------------------------------------ every missing source is counted
def graphcare_with_gaps(root):
    """GNNExplainer succeeds on visit 0, is unsupported on 1 and failed on 2; the built-in is unavailable on all."""
    unavailable = {'builtin_status': 'unavailable', 'builtin_unavailable_reason': 'no eligible attention'}
    visits = [
        ({'GradExplainer': source(.1, .1, .1, .1, .5), 'GNNExplainer': source(.4, .1, .1, .1, .5)}, unavailable),
        ({'GradExplainer': source(.3, .1, .1, .1, .5)},
         dict(unavailable, graphxai={'GradExplainer': {'status': 'success'},
                                     'GNNExplainer': {'status': 'unsupported',
                                                      'unsupported_reason': 'edge mask does not reach the output'}})),
        ({'GradExplainer': source(.8, .1, .1, .1, .5)},
         dict(unavailable, graphxai={'GradExplainer': {'status': 'success'},
                                     'GNNExplainer': {'status': 'failed', 'error': 'boom'}}))]
    write_run(root, 10, 'graphcare_seed1234', visits, method='graphcare')


def test_unsupported_failed_and_unavailable_sources_are_counted(tmp_path):
    graphcare_with_gaps(tmp_path)
    runs, _ = summarize.collect(tmp_path)
    rows = rows_of(summarize.aggregate(runs))
    gnn = rows[(10, 'graphcare', 'GNNExplainer')]
    assert (gnn['visits'], gnn['visits_explained'], gnn['visits_unsupported'], gnn['visits_failed_explainer']) \
        == (3, 1, 1, 1)
    assert gnn['reasons'] == {'unsupported': ['edge mask does not reach the output']}
    assert gnn['fidelity_plus_mean'] == pytest.approx(0.4)
    grad = rows[(10, 'graphcare', 'GradExplainer')]
    assert grad['visits'] == 3 and grad['visits_explained'] == 3
    assert grad['fidelity_plus_mean'] == pytest.approx((0.1 + 0.3 + 0.8) / 3)
    builtin = rows[(10, 'graphcare', 'builtin')]
    assert (builtin['visits'], builtin['visits_explained'], builtin['visits_unavailable']) == (3, 0, 3)
    assert builtin['reasons'] == {'unavailable': ['no eligible attention']}
    for row in rows.values():
        assert row['visits'] == sum(row[c] for c in summarize.COUNT_COLUMNS)


def test_a_source_missing_everywhere_is_listed_with_its_reason_and_the_pair_uses_common_visits(tmp_path):
    graphcare_with_gaps(tmp_path / 'in')
    out = run_main(tmp_path / 'in', tmp_path)
    data = json.loads((out / 'summary.json').read_text())
    assert [(r['source'], r['visits_unavailable']) for r in data['sources_not_available']] == [('builtin', 3)]
    assert {r['source'] for r in data['rows']} == {'GradExplainer', 'GNNExplainer'}     # no empty builtin row
    text = (out / 'summary.md').read_text()
    section = text.split('### Sources not available')[1].split('###')[0]
    assert 'builtin' in section and 'no eligible attention' in section
    paired = {r['source']: r for r in data['paired_rows']}
    # the one visit both explainers have is visit 0: Grad is judged there only (.1), not on its three visits (.4)
    assert {s: r['paired_visits'] for s, r in paired.items()} == {'GradExplainer': 1, 'GNNExplainer': 1}
    assert paired['GradExplainer']['fidelity_plus_mean'] == pytest.approx(0.1)
    assert paired['GNNExplainer']['fidelity_plus_mean'] == pytest.approx(0.4)
    assert '### Paired comparison (visits every source of the method has)' in text


def test_no_eligible_evidence_and_failed_fidelity_have_their_own_columns(tmp_path):
    write_run(tmp_path, 10, 'protgnn_seed7', [
        {'builtin': source(.2, .1, .3, .2, .5), 'GradExplainer': source(.1, .1, .1, .1, .5)},
        ({'GradExplainer': source(.2, .1, .1, .1, .5), 'builtin': {'status': 'no_eligible_evidence', 'support': 0}},
         {'builtin_status': 'no_eligible_evidence'}),
        {'builtin': {'status': 'failed', 'error': 'x'}, 'GradExplainer': source(.3, .1, .1, .1, .5)}], method='protgnn')
    runs, _ = summarize.collect(tmp_path)
    builtin = rows_of(summarize.aggregate(runs))[(10, 'protgnn', 'builtin')]
    assert (builtin['visits'], builtin['visits_explained'], builtin['visits_no_eligible'],
            builtin['visits_failed_fidelity']) == (3, 1, 1, 1)
    assert builtin['fidelity_plus_mean'] == pytest.approx(0.2)


def test_ig_completeness_share_ties_and_the_zero_baseline_exclusion(tmp_path):
    ties = {'distinct_values_eligible': 1, 'largest_tie_group_eligible': 6, 'constant_over_eligible': True}
    varied = {'distinct_values_eligible': 5, 'largest_tie_group_eligible': 2, 'constant_over_eligible': False}
    ig = lambda met: {'status': 'success', 'provenance': {'implementation': 'core.explain.ig_clinical (clinical IG)',
                                                          'completeness_met': met}}
    visits = []
    for met, tie, tied in ((True, varied, False), (True, ties, True), (False, varied, False)):
        fidelity = {'IntegratedGradExplainer': source(.2, .1, .1, .1, .5, ties=tie, tied_at_cut=tied),
                    'IntegratedGradExplainerZeroBaseline': source(.9, .9, .9, .9, .9),
                    'GradExplainer': source(.3, .1, .1, .1, .5)}
        visits.append((fidelity, {'builtin_status': 'unavailable', 'graphxai': {'IntegratedGradExplainer': ig(met), 'GradExplainer': {'status': 'success'},
                                               'IntegratedGradExplainerZeroBaseline': {'status': 'success'}}}))
    write_run(tmp_path / 'in', 10, 'gsat_seed7', visits, method='gsat')
    out = run_main(tmp_path / 'in', tmp_path)
    data = json.loads((out / 'summary.json').read_text())
    rows = {r['source']: r for r in data['rows']}
    assert set(rows) == {'IntegratedGradExplainer', 'GradExplainer'} and 'IntegratedGradExplainerZeroBaseline' in \
        data['excluded_sources']
    ig_row = rows['IntegratedGradExplainer']
    assert ig_row['ig_completeness_met_share'] == pytest.approx(2 / 3)
    assert ig_row['constant_importance_share'] == pytest.approx(1 / 3)
    assert ig_row['tied_at_cut_share'] == pytest.approx(1 / 3)
    assert ig_row['largest_tie_group_mean'] == pytest.approx((2 + 6 + 2) / 3)
    assert ig_row['source_label'] == 'IntegratedGradExplainer (clinical baseline)'
    assert rows['GradExplainer']['ig_completeness_met_share'] is None
    text = (out / 'summary.md').read_text()
    assert 'IntegratedGradExplainer (clinical baseline)' in text and 'ZeroBaseline` (vendored zero-baseline' in text
    assert '| gsat | IntegratedGradExplainerZeroBaseline' not in text


def test_the_builtin_tie_summary_of_the_record_fills_the_tie_columns(tmp_path):
    ties = {'distinct_values_eligible': 1, 'largest_tie_group_eligible': 4, 'constant_over_eligible': True}
    write_run(tmp_path, 10, 'graphcare_seed7', [({'builtin': source(.2, .1, .3, .2, .5)},
                                                {'builtin_tie_summary': ties})], method='graphcare')
    runs, _ = summarize.collect(tmp_path)
    row = summarize.aggregate(runs)[0]
    assert row['constant_importance_share'] == 1.0 and row['largest_tie_group_mean'] == 4.0


# ------------------------------------------------------------------ caveats, code, settings
PROJECTED = {'selected_epoch': 21, 'first_projection_epoch': 21, 'prototypes_projected': True,
             'last_projection_epoch': 21, 'epochs_trained_after_last_projection': 0}


def protgnn_seeds(root, states):
    for seed, state in zip((1234, 2025, 7), states):
        write_run(root, 10, f'protgnn_seed{seed}', ONE, method='protgnn', projection=state,
                  disclosure=dict(DISCLOSURE, caveats=['ProtGNN+ style per-prediction subgraph']))


def test_protgnn_seeds_projected_are_counted_in_the_caveat_block(tmp_path):
    protgnn_seeds(tmp_path / 'in', [PROJECTED, PROJECTED, dict(PROJECTED, prototypes_projected=False,
                                                                first_projection_epoch=None,
                                                                last_projection_epoch=None,
                                                                epochs_trained_after_last_projection=None)])
    out = run_main(tmp_path / 'in', tmp_path)
    text = (out / 'summary.md').read_text()
    assert '### Caveats: protgnn (K = 10)' in text
    assert 'Prototype projection: 2 of 3 seeds projected (seed 7: selected before the first projection' in text
    assert 'ProtGNN+ style per-prediction subgraph' in text
    assert 'Replay rule that passed: exact_hash' in text
    assert json.loads((out / 'summary.json').read_text())['caveats']['K=10 protgnn']


def test_every_method_prints_its_own_disclosure(tmp_path):
    build(tmp_path / 'in')
    out = run_main(tmp_path / 'in', tmp_path)
    text = (out / 'summary.md').read_text()
    block = text.split('### Caveats: cei_gnn_mp2 (K = 10)')[1]
    assert 'Intervention: node deletion with concept cascade' in block
    assert 'CEI pairs and absence states are not reachable by edge masks' in block
    assert 'Settings:' in block and 'gnnexplainer_epochs: 50' in block and 'Cohort sample:' in block
    assert 'projection' not in block


def test_a_disclosure_that_differs_within_a_method_is_refused(tmp_path):
    write_run(tmp_path, 10, 'gsat_seed1234', ONE, method='gsat')
    write_run(tmp_path, 10, 'gsat_seed7', ONE, method='gsat', disclosure=dict(DISCLOSURE, intervention='other'))
    with pytest.raises(SystemExit, match=r'mix disclosures within K=10 gsat'):
        summarize.collect(tmp_path)
    runs, meta = summarize.collect(tmp_path, allow_mixed_code=True)
    assert meta['mixed'][0]['what'].startswith('disclosures within K=10 gsat')


def test_runs_without_code_hashes_or_disclosure_are_refused(tmp_path):
    write_run(tmp_path / 'code', 10, 'gsat_seed7', ONE, method='gsat', with_code=False)
    with pytest.raises(SystemExit, match=r'gsat_seed7.*no hashed explanation code'):
        summarize.collect(tmp_path / 'code')
    write_run(tmp_path / 'disclosure', 10, 'gsat_seed7', ONE, method='gsat', with_disclosure=False)
    with pytest.raises(SystemExit, match=r'gsat_seed7.*no disclosure'):
        summarize.collect(tmp_path / 'disclosure')


def test_mixed_code_hashes_and_cohort_samples_are_refused_unless_allowed(tmp_path):
    """Two runs with other code and another cohort sample are never summarised into one row without a word."""
    write_run(tmp_path / 'in', 10, 'gsat_seed1234', ONE, method='gsat', code='A', sample=500)
    write_run(tmp_path / 'in', 10, 'gsat_seed7', ONE, method='gsat', code='B', sample=10)
    with pytest.raises(SystemExit, match=r'refusing to mix explanation code.*top10/gsat_seed1234.*top10/gsat_seed7'):
        summarize.collect(tmp_path / 'in')
    with pytest.raises(SystemExit, match='--allow-mixed-code'):
        summarize.main(['--explanations', str(tmp_path / 'in'), '--out', str(tmp_path / 'refused')])
    assert not (tmp_path / 'refused').exists()
    out = run_main(tmp_path / 'in', tmp_path, 'allowed', '--allow-mixed-code')
    text = (out / 'summary.md').read_text()
    assert '**WARNING: mixed explanation code**' in text and '**WARNING: mixed settings**' in text
    data = json.loads((out / 'summary.json').read_text())
    assert {m['what'] for m in data['mixed']} == {'explanation code', 'settings'} and len(data['code']) == 2


def test_the_cohort_sample_alone_is_enough_to_refuse(tmp_path):
    write_run(tmp_path, 10, 'gsat_seed1234', ONE, method='gsat', sample=500)
    write_run(tmp_path, 10, 'gsat_seed7', ONE, method='gsat', sample=10)
    with pytest.raises(SystemExit, match='refusing to mix settings'):
        summarize.collect(tmp_path)


def test_mixed_explainer_settings_are_refused_but_the_chosen_ig_step_count_is_not_a_setting(tmp_path):
    write_run(tmp_path / 'steps', 10, 'gsat_seed1234', ONE, method='gsat')
    other = dict(SETTINGS, ig=dict(SETTINGS['ig'], steps_chosen=64))
    write_run(tmp_path / 'steps', 10, 'gsat_seed7', ONE, method='gsat', settings=other)
    assert len(summarize.collect(tmp_path / 'steps')[0]) == 2
    write_run(tmp_path / 'epochs', 10, 'gsat_seed1234', ONE, method='gsat')
    write_run(tmp_path / 'epochs', 10, 'gsat_seed7', ONE, method='gsat', settings=dict(SETTINGS, tie_draws=3))
    with pytest.raises(SystemExit, match=r'refusing to mix settings in one table: tie_draws: 10 \(top10/gsat_seed1234\); '
                                         r'tie_draws: 3 \(top10/gsat_seed7\);'):
        summarize.collect(tmp_path / 'epochs')
    write_run(tmp_path / 'tolerance', 10, 'gsat_seed1234', ONE, method='gsat')
    write_run(tmp_path / 'tolerance', 10, 'gsat_seed7', ONE, method='gsat',
              settings=dict(SETTINGS, ig=dict(SETTINGS['ig'], tolerance=0.5)))
    with pytest.raises(SystemExit, match=r'refusing to mix settings in one table: ig.tolerance: 0.05 .*ig.tolerance: 0.5 '):
        summarize.collect(tmp_path / 'tolerance')


def test_a_mixed_settings_or_disclosure_message_names_only_what_differs(tmp_path):
    """Long settings and disclosures share their first characters; the refusal and the published warning
    name the differing entries, not a common prefix."""
    for seed in (1234, 2025):
        write_run(tmp_path / 'in', 10, f'gsat_seed{seed}', ONE, method='gsat')
    write_run(tmp_path / 'in', 10, 'gsat_seed7', ONE, method='gsat', settings=dict(SETTINGS, random_repeats=20),
              disclosure=dict(DISCLOSURE, caveats=[*DISCLOSURE['caveats'], 'an added caveat']))
    with pytest.raises(SystemExit) as refused:
        summarize.collect(tmp_path / 'in')
    message = str(refused.value)
    assert 'random_repeats: 20 (top10/gsat_seed7)' in message
    assert 'random_repeats: 50 (top10/gsat_seed1234, top10/gsat_seed2025)' in message
    assert 'curve_points' not in message and 'cohort_sampling' not in message
    out = run_main(tmp_path / 'in', tmp_path, 'allowed', '--allow-mixed-code')
    text = (out / 'summary.md').read_text()
    assert '**WARNING: mixed settings** (--allow-mixed-code): random_repeats: 20 (1 runs); random_repeats: 50 (2 runs)' in text
    warning = next(line for line in text.splitlines() if line.startswith('**WARNING: mixed disclosures'))
    assert 'caveats.1: an added caveat (1 runs)' in warning and 'caveats.1: absent (2 runs)' in warning
    assert 'intervention' not in warning
    mixed = {m['what']: m['values'] for m in json.loads((out / 'summary.json').read_text())['mixed']}
    assert mixed['settings'] == [{'value': 'random_repeats: 20', 'runs': 1}, {'value': 'random_repeats: 50', 'runs': 2}]


# ------------------------------------------------------------------ protocol, K and seed from the bindings
def test_protocol_comes_from_the_bindings_when_there_is_no_driver_header(tmp_path):
    build(tmp_path / 'in')
    out = run_main(tmp_path / 'in', tmp_path)
    text = (out / 'summary.md').read_text()
    assert 'Protocol: fair_v10t.' in text and 'not_recorded' not in text
    assert json.loads((out / 'summary.json').read_text())['protocol'] == 'fair_v10t'


def test_a_folder_whose_k_or_seed_disagrees_with_its_binding_is_refused(tmp_path):
    write_run(tmp_path / 'k', 20, 'gsat_seed7', ONE, method='gsat', model_k=10)
    with pytest.raises(SystemExit, match=r'gsat_seed7.*top_k_labels 20, its binding says 10'):
        summarize.collect(tmp_path / 'k')
    write_run(tmp_path / 'seed', 10, 'gsat_seed7', ONE, method='gsat', model_seed=2025)
    with pytest.raises(SystemExit, match=r'gsat_seed7.*seed 7, its binding says 2025'):
        summarize.collect(tmp_path / 'seed')


def test_binding_without_protocol_k_seed_or_with_overrides_is_refused(tmp_path):
    for field in ('protocol', 'top_k_labels', 'seed'):
        root = tmp_path / field
        write_run(root, 10, 'gsat_seed7', ONE, method='gsat')
        path = root / 'top10' / 'gsat_seed7' / 'binding.json'
        binding = json.loads(path.read_text())
        del binding['model_run'][field]
        path.write_text(json.dumps(binding))
        with pytest.raises(SystemExit, match=rf'model_run carries no {field}'):
            summarize.collect(root)
    write_run(tmp_path / 'overrides', 10, 'gsat_seed7', ONE, method='gsat', overrides=['--lr 0.1'])
    with pytest.raises(SystemExit, match='protocol_overrides must be an empty list'):
        summarize.collect(tmp_path / 'overrides')
    # a training binding without the key gives null in the runner's binding: unknown overrides are not final
    for name, edit in (('null', lambda m: m.update(protocol_overrides=None)),
                       ('absent', lambda m: m.pop('protocol_overrides'))):
        root = tmp_path / name
        write_run(root, 10, 'gsat_seed7', ONE, method='gsat')
        path = root / 'top10' / 'gsat_seed7' / 'binding.json'
        binding = json.loads(path.read_text())
        edit(binding['model_run'])
        path.write_text(json.dumps(binding))
        with pytest.raises(SystemExit, match=r'gsat_seed7.*protocol_overrides must be an empty list, not None'):
            summarize.collect(root)


def test_mixed_protocols_and_a_driver_protocol_that_disagrees_are_refused(tmp_path):
    write_run(tmp_path / 'mixed', 10, 'gsat_seed1234', ONE, method='gsat')
    write_run(tmp_path / 'mixed', 10, 'gsat_seed7', ONE, method='gsat', protocol='fair_v9')
    with pytest.raises(SystemExit, match=r'mix protocols.*fair_v10t.*fair_v9'):
        summarize.collect(tmp_path / 'mixed')
    build(tmp_path / 'driver')
    driver_manifest(tmp_path / 'driver', CEI_RUNS, protocol='fair_v9')
    with pytest.raises(SystemExit, match='driver header says protocol fair_v9'):
        summarize.collect(tmp_path / 'driver')


# ------------------------------------------------------------------ run set against the driver plan
def test_a_planned_run_without_a_folder_is_refused(tmp_path):
    """The driver says two runs ok, one folder is missing."""
    write_run(tmp_path, 10, 'gsat_seed1234', ONE, method='gsat')
    driver_manifest(tmp_path, [(10, 'gsat', 1234, 0), (10, 'gsat', 7, 0)])
    with pytest.raises(SystemExit, match=r'folder is missing: top10/gsat_seed7'):
        summarize.collect(tmp_path)


def test_a_folder_outside_the_plan_is_refused(tmp_path):
    write_run(tmp_path, 10, 'gsat_seed1234', ONE, method='gsat')
    write_run(tmp_path, 10, 'gsat_seed7', ONE, method='gsat')
    driver_manifest(tmp_path, [(10, 'gsat', 1234, 0)])
    with pytest.raises(SystemExit, match=r'outside the driver plan: top10/gsat_seed7'):
        summarize.collect(tmp_path)


def test_a_cohort_sample_other_than_the_plan_is_refused(tmp_path):
    write_run(tmp_path, 10, 'gsat_seed1234', ONE, method='gsat', sample=10)
    driver_manifest(tmp_path, [(10, 'gsat', 1234, 0)], sample=500)
    with pytest.raises(SystemExit, match=r'cohort sample differs from the driver plan \(500\): top10/gsat_seed1234'):
        summarize.collect(tmp_path)


def test_an_every_patient_driver_plan_is_summarised(tmp_path):
    """The driver's --sample-patients 0 (header null, or 0 in an older header) and the runner's null are one size."""
    for header in (None, 0):
        root = tmp_path / f'every_{header}'
        write_run(root, 10, 'gsat_seed1234', ONE, method='gsat', sample=None)
        driver_manifest(root, [(10, 'gsat', 1234, 0)], sample=header)
        assert len(summarize.collect(root)[0]) == 1
    root = tmp_path / 'every_vs_500'
    write_run(root, 10, 'gsat_seed1234', ONE, method='gsat', sample=500)
    driver_manifest(root, [(10, 'gsat', 1234, 0)], sample=None)
    with pytest.raises(SystemExit, match=r'cohort sample differs from the driver plan \(every patient\)'):
        summarize.collect(root)


def test_without_a_driver_header_the_markdown_says_the_run_set_was_not_checked(tmp_path):
    write_run(tmp_path / 'in', 10, 'gsat_seed7', ONE, method='gsat')
    out = run_main(tmp_path / 'in', tmp_path)
    assert 'no driver manifest: run set not checked' in (out / 'summary.md').read_text()
    assert json.loads((out / 'summary.json').read_text())['run_set_checked_against_driver_plan'] is False


def test_a_row_with_fewer_runs_than_planned_is_flagged(tmp_path):
    root = tmp_path / 'in'
    write_run(root, 10, 'gsat_seed1234', ONE, method='gsat')
    driver_manifest(root, [(10, 'gsat', 1234, 0), (10, 'gsat', 7, 1)], complete=False)
    out = run_main(root, tmp_path, 'out', '--allow-incomplete')
    row = json.loads((out / 'summary.json').read_text())['rows'][0]
    assert (row['runs'], row['runs_planned']) == (1, 2)
    assert '| 1 of 2 |' in (out / 'summary.md').read_text()


# ------------------------------------------------------------------ the difference to random and the other guards
def edit_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_the_difference_to_random_counts_the_visits_it_rests_on(tmp_path):
    """One visit's random reference failed: its metrics carry no `*_vs_random`, so the headline is a mean over the
    other visits; the row says over how many and the markdown marks the headline cells."""
    failed = {'random_reference': {'status': 'failed', 'repeats': 50, 'seed': 1, 'error': 'boom'}}
    computed = {'random_reference': {'status': 'computed', 'repeats': 50, 'seed': 1}}
    write_run(tmp_path / 'in', 10, 'gsat_seed7', [
        ({'GradExplainer': source(.2, .1, .3, .2, .5, vs=.1)}, dict(computed, builtin_status='unavailable')),
        ({'GradExplainer': source(.4, .1, .3, .2, .5, vs=.3)}, dict(computed, builtin_status='unavailable')),
        ({'GradExplainer': source(.6, .1, .3, .2, .5)}, dict(failed, builtin_status='unavailable'))], method='gsat')
    out = run_main(tmp_path / 'in', tmp_path)
    data = json.loads((out / 'summary.json').read_text())
    row = {r['source']: r for r in data['rows']}['GradExplainer']
    assert (row['visits_explained'], row['visits_vs_random']) == (3, 2)
    assert row['random_reference_missing'] == ['failed']
    assert row['fidelity_plus_mean'] == pytest.approx(0.4)                 # over all three visits
    assert row['deletion_mean_change_vs_random_mean'] == pytest.approx(0.2)  # over the two with a reference
    paired = {r['source']: r for r in data['paired_rows']}['GradExplainer']
    assert (paired['paired_visits'], paired['visits_vs_random']) == (3, 2)
    assert 'visits_vs_random' in next(csv.reader(open(out / 'summary.csv')))
    text = (out / 'summary.md').read_text()
    assert '| with random reference |' in text and '0.2000 *' in text
    assert '`*`: the difference to random of these rows rests on fewer visits than the other columns ' \
           '(random reference failed): gsat GradExplainer 2 of 3.' in text
    # a full row is not marked
    write_run(tmp_path / 'full', 10, 'gsat_seed7', [
        ({'GradExplainer': source(.2, .1, .3, .2, .5, vs=.1)}, dict(computed, builtin_status='unavailable'))],
        method='gsat')
    text = (run_main(tmp_path / 'full', tmp_path, 'out_full') / 'summary.md').read_text()
    assert ' *' not in text and '`*`' not in text


def test_built_in_explanation_settings_that_differ_within_a_method_are_refused_and_printed(tmp_path):
    settings = {'method': 'protgnn', 'algorithm': 'MCTS', 'settings': {'rollout': 10}}
    for seed, rollout in ((1234, 10), (7, 20)):
        write_run(tmp_path / 'in', 10, f'protgnn_seed{seed}', ONE, method='protgnn')
        edit_json(tmp_path / 'in' / 'top10' / f'protgnn_seed{seed}' / 'explanation_manifest.json',
                  lambda m, r=rollout: m.update(builtin_explanation=dict(settings, settings={'rollout': r})))
    with pytest.raises(SystemExit, match=r'mix built-in explanation settings within K=10 protgnn'):
        summarize.collect(tmp_path / 'in')
    write_run(tmp_path / 'same', 10, 'protgnn_seed7', ONE, method='protgnn')
    edit_json(tmp_path / 'same' / 'top10' / 'protgnn_seed7' / 'explanation_manifest.json',
              lambda m: m.update(builtin_explanation=settings))
    text = (run_main(tmp_path / 'same', tmp_path) / 'summary.md').read_text()
    assert '- Built-in explanation source: method: protgnn; algorithm: MCTS; settings: rollout: 10' in text


def test_a_manifest_method_other_than_the_binding_method_is_refused(tmp_path):
    write_run(tmp_path, 10, 'gsat_seed7', ONE, method='gsat')
    edit_json(tmp_path / 'top10' / 'gsat_seed7' / 'binding.json', lambda b: b['model_run'].update(method='graphcare'))
    with pytest.raises(SystemExit, match=r"gsat_seed7: manifest method 'gsat' differs from its binding \('graphcare'\)"):
        summarize.collect(tmp_path)


def test_runs_that_differ_only_in_the_vendored_graphxai_tree_are_refused(tmp_path):
    write_run(tmp_path, 10, 'gsat_seed1234', ONE, method='gsat')
    write_run(tmp_path, 10, 'gsat_seed7', ONE, method='gsat')
    edit_json(tmp_path / 'top10' / 'gsat_seed7' / 'explanation_manifest.json',
              lambda m: m['code'].update(vendored_graphxai_tree_sha256='other'))
    with pytest.raises(SystemExit, match=r'refusing to mix explanation code.*gsat_seed1234.*gsat_seed7'):
        summarize.collect(tmp_path)


@pytest.mark.parametrize('extra, match', [
    ({'builtin_status': 'maybe'}, r"gsat_seed7: a record carries builtin_status 'maybe'"),
    ({'graphxai': {'GradExplainer': {'status': 'pending'}}}, r"gsat_seed7: GradExplainer carries status 'pending'"),
    ({'fidelity_v2': {}}, r'gsat_seed7: GradExplainer has no fidelity_v2 entry'),
    ({'fidelity_v2': {'GradExplainer': {'fidelity_plus': {}}}}, r'gsat_seed7: GradExplainer has a malformed'),
    ({'random_reference': {'status': 'lost'}}, r"gsat_seed7: a record carries random_reference status 'lost'"),
    ({'graphxai': {'GradExplainer': {'status': 'success'},
                   'IntegratedGradExplainer': {'status': 'success',
                                               'provenance': {'implementation': 'core.explain.ig_clinical'}}},
      'fidelity_v2': {'GradExplainer': source(.2, .1, .3, .2, .5),
                      'IntegratedGradExplainer': source(.2, .1, .3, .2, .5)}},
     r'gsat_seed7: IntegratedGradExplainer \(clinical IG\) carries no completeness_met'),
])
def test_records_the_summary_cannot_read_are_refused_with_the_run(tmp_path, extra, match):
    write_run(tmp_path, 10, 'gsat_seed7', [({'GradExplainer': source(.2, .1, .3, .2, .5)},
                                           dict({'builtin_status': 'unavailable'}, **extra))], method='gsat')
    with pytest.raises(SystemExit, match=match):
        summarize.collect(tmp_path)


def test_an_unknown_fold_is_refused_with_the_run(tmp_path):
    write_run(tmp_path, 10, 'gsat_seed7', ONE, method='gsat')
    edit_json(tmp_path / 'top10' / 'gsat_seed7' / 'explanation_manifest.json', lambda m: m.update(fold='train'))
    with pytest.raises(SystemExit, match=r"gsat_seed7: unknown fold 'train'"):
        summarize.collect(tmp_path)


def test_a_graphxai_source_without_eligible_evidence_is_counted(tmp_path):
    write_run(tmp_path, 10, 'gsat_seed7', [
        ({'GradExplainer': source(.2, .1, .3, .2, .5)}, {'builtin_status': 'unavailable'}),
        ({'GradExplainer': {'status': 'no_eligible_evidence', 'support': 0}},
         {'builtin_status': 'unavailable', 'graphxai': {'GradExplainer': {'status': 'success'}}})], method='gsat')
    runs, _ = summarize.collect(tmp_path)
    row = rows_of(summarize.aggregate(runs))[(10, 'gsat', 'GradExplainer')]
    assert (row['visits'], row['visits_explained'], row['visits_no_eligible']) == (2, 1, 1)
    assert row['fidelity_plus_mean'] == pytest.approx(0.2)


def test_summarize_does_not_import_torch_or_the_fidelity_module():
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    code = ('import sys; sys.path.insert(0, sys.argv[1]); import comparison.explain.summarize; '
            'sys.exit(int("torch" in sys.modules or "core.explain.fidelity_v2" in sys.modules))')
    assert subprocess.run([sys.executable, '-I', '-c', code, str(repo)], check=False).returncode == 0
