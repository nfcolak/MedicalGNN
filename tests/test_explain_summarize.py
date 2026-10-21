"""Aggregate explanation tables from synthetic runner outputs in the runner's layout (numbers checked by hand)."""
import csv
import json
import math

import pytest

from comparison.explain import summarize


def source(plus, minus, dauc, iauc, sparsity, vs=None, fraction=0.5):
    # mean_change = AUC / explained fraction, as the curves record it
    entry = {'fidelity_plus': {'prob': plus}, 'fidelity_minus': {'prob': minus},
             'deletion_curve': {'auc': dauc, 'mean_change': dauc / fraction, 'explained_fraction': fraction},
             'insertion_curve': {'auc': iauc, 'mean_change': iauc / fraction, 'explained_fraction': fraction},
             'sparsity': {'mass90_sparsity': sparsity}}
    if vs is not None:
        entry['vs_random'] = {key: {'difference': vs, 'beats_random_fraction': 0.5} for key in
                              ('fidelity_plus', 'fidelity_minus', 'deletion_auc', 'insertion_auc',
                               'deletion_mean_change', 'insertion_mean_change')}
    return entry


def write_run(root, k, name, visits, *, method='cei_gnn_mp2', fold='test', failed=0, role=None,
              final_eval='test', version=4, binding_version='same', record_versions=None):
    run_dir = root / f'top{k}' / name
    run_dir.mkdir(parents=True)
    files = []
    for i, fidelity in enumerate(visits):
        filename = f'subject_secret{k}{name}{i}.json'
        record = {'sample_id': f'secret{i}', 'subject_id': f'patient{i}', 'fidelity_v2': fidelity}
        record_version = version if record_versions is None else record_versions[i]
        if record_version is not None:
            record['fidelity_v2_intervention'] = {'version': record_version}
        (run_dir / filename).write_text(json.dumps(record))
        files.append(filename)
    manifest = {'method': method, 'fold': fold, 'explained_count': len(visits), 'failed_count': failed,
                'failures': [{'sample_id': 'secret9', 'error': 'boom'}] * failed, 'record_files': files,
                'run_dir': '/home/user/runs/x'}
    if role:
        manifest['fold_role'] = role
    (run_dir / 'explanation_manifest.json').write_text(json.dumps(manifest))
    bound = version if binding_version == 'same' else binding_version
    explainer = {} if bound is None else {'intervention_contract_version': bound}
    (run_dir / 'binding.json').write_text(json.dumps({'model_run': {'final_eval': final_eval},
                                                      'explainer_config': explainer}))


def driver_manifest(root, runs, *, complete=True, fold='test', protocol='fair_v10t'):
    """runs: (k, method, seed, returncode) tuples."""
    (root / 'driver_manifest.json').write_text(json.dumps({
        'protocol': protocol, 'fold': fold, 'complete': complete, 'planned_runs': len(runs),
        'runs': [{'k': k, 'method': m, 'seed': s, 'output': f'/home/user/out/top{k}/{m}_seed{s}',
                  'returncode': code} for k, m, s, code in runs]}))


def build(root, **kw):
    # seed 1234: builtin plus .2/.4 (mean .3); seed 2025: .5/.7 (mean .6) -> mean .45, SD .15*sqrt(2)
    write_run(root, 10, 'cei_mp2_seed1234', [
        {'builtin': source(.2, .1, .3, .2, .5, vs=.1), 'GradExplainer': source(.1, .1, .1, .1, .9)},
        {'builtin': source(.4, .3, .5, .4, .7, vs=.3), 'GradExplainer': {'status': 'failed', 'error': 'x'}}],
        failed=1, **kw)
    write_run(root, 10, 'cei_mp2_seed2025', [
        {'builtin': source(.5, .2, .4, .3, .5, vs=.2), 'GradExplainer': source(.3, .1, .2, .2, .8)},
        {'builtin': source(.7, .4, .6, .5, .9, vs=.4), 'GradExplainer': source(.5, .1, .4, .2, .6)}], **kw)


CEI_RUNS = [(10, 'cei_mp2', 1234, 0), (10, 'cei_mp2', 2025, 0)]


def rows_of(rows):
    return {(r['k'], r['method'], r['source']): r for r in rows}


def test_mean_and_sd_over_seeds_by_hand(tmp_path):
    build(tmp_path)
    runs, _ = summarize.collect(tmp_path)
    rows = rows_of(summarize.aggregate(runs))
    builtin = rows[(10, 'cei_gnn_mp2', 'builtin')]
    assert builtin['runs'] == 2 and builtin['visits'] == 4
    assert builtin['fidelity_plus_mean'] == pytest.approx(0.45)
    assert builtin['fidelity_plus_sd'] == pytest.approx(0.15 * math.sqrt(2))
    assert builtin['fidelity_minus_mean'] == pytest.approx(0.25)
    assert builtin['sparsity_mean'] == pytest.approx(((0.5 + 0.7) / 2 + (0.5 + 0.9) / 2) / 2)
    assert builtin['deletion_auc_vs_random_mean'] == pytest.approx(((0.1 + 0.3) / 2 + (0.2 + 0.4) / 2) / 2)
    assert builtin['fidelity_plus_beats_random_mean'] == pytest.approx(0.5)
    assert builtin['failed_visits'] == 1 and builtin['failed_sources'] == 0
    grad = rows[(10, 'cei_gnn_mp2', 'GradExplainer')]
    # seed 1234 has one failed evaluation: its run mean is the single good visit (.1)
    assert grad['visits'] == 3 and grad['failed_sources'] == 1
    assert grad['fidelity_plus_mean'] == pytest.approx((0.1 + (0.3 + 0.5) / 2) / 2)
    assert grad['deletion_auc_vs_random_mean'] is None


def test_single_run_has_no_sd(tmp_path):
    write_run(tmp_path, 20, 'gsat_seed7', [{'builtin': source(.2, .1, .3, .2, .5)}], method='gsat')
    runs, _ = summarize.collect(tmp_path)
    row = summarize.aggregate(runs)[0]
    assert row['fidelity_plus_mean'] == pytest.approx(0.2) and row['fidelity_plus_sd'] is None


def test_outputs_name_the_fold_and_hold_no_ids_or_paths(tmp_path):
    build(tmp_path / 'in')
    driver_manifest(tmp_path / 'in', CEI_RUNS)
    out = tmp_path / 'out'
    assert summarize.main(['--explanations', str(tmp_path / 'in'), '--out', str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ['summary.csv', 'summary.json', 'summary.md']
    text = ''.join((out / name).read_text() for name in ('summary.csv', 'summary.json', 'summary.md'))
    assert 'test (report fold)' in text and 'fair_v10t' in text
    assert '## K = 10 · fold: test (report fold)' in (out / 'summary.md').read_text()
    for forbidden in ('secret', 'patient', '/home/', 'sample_id', 'subject'):
        assert forbidden not in text
    with open(out / 'summary.csv') as handle:
        table = list(csv.DictReader(handle))
    assert {r['fold'] for r in table} == {'test (report fold)'} and len(table) == 2
    data = json.loads((out / 'summary.json').read_text())
    assert data['fold_role'] == 'report' and len(data['rows']) == 2


def test_validation_is_labelled_decision_fold(tmp_path):
    build(tmp_path / 'in', fold='validation', role='decision')
    assert summarize.main(['--explanations', str(tmp_path / 'in'), '--out', str(tmp_path / 'out')]) == 0
    assert 'validation (decision fold)' in (tmp_path / 'out' / 'summary.md').read_text()


def test_refuses_mixed_folds_roles_and_protocols(tmp_path):
    build(tmp_path / 'folds')
    write_run(tmp_path / 'folds', 10, 'gsat_seed7', [{'builtin': source(.2, .1, .3, .2, .5)}], method='gsat',
              fold='validation')
    with pytest.raises(SystemExit, match='mix folds'):
        summarize.collect(tmp_path / 'folds')
    write_run(tmp_path / 'roles', 10, 'gsat_seed7', [{'builtin': source(.2, .1, .3, .2, .5)}], method='gsat',
              role='decision')
    with pytest.raises(SystemExit, match='fold_role'):
        summarize.collect(tmp_path / 'roles')
    build(tmp_path / 'proto')
    write_run(tmp_path / 'proto', 20, 'gsat_seed7', [{'builtin': source(.2, .1, .3, .2, .5)}], method='gsat',
              final_eval='validation')
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


# ------------------------------------------------------------------ curve columns, versions, driver
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


def test_markdown_has_both_columns_in_headline_order_and_direction_notes(tmp_path, monkeypatch):
    build(tmp_path / 'in')
    assert summarize.HEADLINE_CURVE == 'mean_change'     # user decision 2026-10-21
    monkeypatch.setattr(summarize, 'HEADLINE_CURVE', 'auc')
    summarize.main(['--explanations', str(tmp_path / 'in'), '--out', str(tmp_path / 'auc')])
    text = (tmp_path / 'auc' / 'summary.md').read_text()
    header = next(line for line in text.splitlines() if line.startswith('| method'))
    cells = [c.strip() for c in header.strip('|').split('|')]
    assert cells.index('deletion_auc') < cells.index('deletion_mean_change')
    assert {'insertion_auc', 'insertion_mean_change', 'explained_fraction',
            'deletion_mean_change_vs_random', 'insertion_auc_vs_random'} <= set(cells)
    assert text.count('Direction: higher is better for fidelity_plus') == 1      # one note per K table
    assert 'lower is better for fidelity_minus' in text and 'negative insertion difference' in text
    assert 'intervention contract v4' in text
    monkeypatch.setattr(summarize, 'HEADLINE_CURVE', 'mean_change')
    summarize.main(['--explanations', str(tmp_path / 'in'), '--out', str(tmp_path / 'mean')])
    header = next(line for line in (tmp_path / 'mean' / 'summary.md').read_text().splitlines()
                  if line.startswith('| method'))
    cells = [c.strip() for c in header.strip('|').split('|')]
    assert cells.index('deletion_mean_change') < cells.index('deletion_auc')
    assert {'deletion_auc', 'insertion_auc', 'deletion_mean_change', 'insertion_mean_change'} <= set(cells)
    csv_columns = next(csv.reader(open(tmp_path / 'mean' / 'summary.csv')))
    assert {'deletion_auc_mean', 'deletion_mean_change_mean', 'explained_fraction_mean'} <= set(csv_columns)
    assert json.loads((tmp_path / 'auc' / 'summary.json').read_text())['intervention_contract_version'] == 4


def test_missing_contract_version_is_refused_and_names_the_run(tmp_path):
    visits = [{'builtin': source(.2, .1, .3, .2, .5)}]
    write_run(tmp_path / 'record', 10, 'gsat_seed7', visits, method='gsat', version=None, binding_version=4)
    with pytest.raises(SystemExit, match=r'gsat_seed7.*no intervention contract version'):
        summarize.collect(tmp_path / 'record')
    write_run(tmp_path / 'binding', 10, 'gsat_seed7', visits, method='gsat', version=4, binding_version=None)
    with pytest.raises(SystemExit, match=r'gsat_seed7.*binding.*no intervention contract version'):
        summarize.collect(tmp_path / 'binding')


def test_disagreeing_versions_are_refused(tmp_path):
    visits = [{'builtin': source(.2, .1, .3, .2, .5)}] * 2
    write_run(tmp_path / 'records', 10, 'gsat_seed7', visits, method='gsat', record_versions=[3, 4],
              binding_version=4)
    with pytest.raises(SystemExit, match=r'gsat_seed7.*records disagree.*3, 4'):
        summarize.collect(tmp_path / 'records')
    write_run(tmp_path / 'binding', 10, 'gsat_seed7', visits, method='gsat', version=4, binding_version=3)
    with pytest.raises(SystemExit, match=r'gsat_seed7.*record contract version 4 does not match its binding \(3\)'):
        summarize.collect(tmp_path / 'binding')
    write_run(tmp_path / 'runs', 10, 'gsat_seed7', visits[:1], method='gsat', version=3)
    write_run(tmp_path / 'runs', 10, 'gsat_seed1', visits[:1], method='gsat', version=4)
    with pytest.raises(SystemExit, match=r'mix intervention contract versions in one table: 3, 4'):
        summarize.collect(tmp_path / 'runs')


def test_incomplete_driver_run_is_refused_unless_allowed_and_lists_the_left_out_runs(tmp_path):
    root = tmp_path / 'in'
    build(root)
    write_run(root, 10, 'gsat_seed1234', [{'builtin': source(.2, .1, .3, .2, .5)}], method='gsat')
    driver_manifest(root, [(10, 'cei_mp2', 1234, 0), (10, 'cei_mp2', 2025, 0), (10, 'gsat', 1234, 0),
                           (10, 'protgnn', 1234, 3), (10, 'graphcare', 1234, None)], complete=False)
    with pytest.raises(SystemExit, match=r'not complete.*K=10 protgnn seed 1234: return code 3'):
        summarize.collect(root)
    with pytest.raises(SystemExit, match='not complete'):
        summarize.main(['--explanations', str(root), '--out', str(tmp_path / 'refused')])
    assert not (tmp_path / 'refused').exists()
    assert summarize.main(['--explanations', str(root), '--out', str(tmp_path / 'out'),
                           '--allow-incomplete']) == 0
    text = (tmp_path / 'out' / 'summary.md').read_text()
    section = text.split('## Runs not summarised')[1]
    assert '| 10 | protgnn | 1234 | 3 |' in section and '| 10 | graphcare | 1234 | not run |' in section
    assert 'gsat' not in section and 'cei_mp2' not in section
    assert json.loads((tmp_path / 'out' / 'summary.json').read_text())['runs_not_summarised'] == [
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
    assert summarize.main(['--explanations', str(root), '--out', str(tmp_path / 'out')]) == 0
    assert 'Runs not summarised' not in (tmp_path / 'out' / 'summary.md').read_text()
    assert json.loads((tmp_path / 'out' / 'summary.json').read_text())['runs_not_summarised'] == []
    skipped = tmp_path / 'resumed'
    build(skipped)
    (skipped / 'driver_manifest.json').write_text(json.dumps({
        'protocol': 'fair_v10t', 'fold': 'test', 'complete': True,
        'runs': [{'k': 10, 'method': 'cei_mp2', 'seed': 1234, 'returncode': None, 'status': 'skipped_existing'},
                 {'k': 10, 'method': 'cei_mp2', 'seed': 2025, 'returncode': 0, 'status': 'ok'}]}))
    runs, _ = summarize.collect(skipped)
    assert len(runs) == 2


def test_summarize_does_not_import_torch_or_the_fidelity_module():
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    code = ('import sys; sys.path.insert(0, sys.argv[1]); import comparison.explain.summarize; '
            'sys.exit(int("torch" in sys.modules or "core.explain.fidelity_v2" in sys.modules))')
    assert subprocess.run([sys.executable, '-I', '-c', code, str(repo)], check=False).returncode == 0
