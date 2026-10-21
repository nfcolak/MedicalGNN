"""Aggregate explanation tables from synthetic runner outputs in the runner's layout (numbers checked by hand)."""
import csv
import json
import math

import pytest

from comparison.explain import summarize


def source(plus, minus, dauc, iauc, sparsity, vs=None):
    entry = {'fidelity_plus': {'prob': plus}, 'fidelity_minus': {'prob': minus},
             'deletion_curve': {'auc': dauc}, 'insertion_curve': {'auc': iauc},
             'sparsity': {'mass90_sparsity': sparsity}}
    if vs is not None:
        entry['vs_random'] = {key: {'difference': vs, 'beats_random_fraction': 0.5} for key in
                              ('fidelity_plus', 'fidelity_minus', 'deletion_auc', 'insertion_auc')}
    return entry


def write_run(root, k, name, visits, *, method='cei_gnn_mp2', fold='test', failed=0, role=None,
              final_eval='test'):
    run_dir = root / f'top{k}' / name
    run_dir.mkdir(parents=True)
    files = []
    for i, fidelity in enumerate(visits):
        filename = f'subject_secret{k}{name}{i}.json'
        (run_dir / filename).write_text(json.dumps({'sample_id': f'secret{i}', 'subject_id': f'patient{i}',
                                                   'fidelity_v2': fidelity}))
        files.append(filename)
    manifest = {'method': method, 'fold': fold, 'explained_count': len(visits), 'failed_count': failed,
                'failures': [{'sample_id': 'secret9', 'error': 'boom'}] * failed, 'record_files': files,
                'run_dir': '/home/user/runs/x'}
    if role:
        manifest['fold_role'] = role
    (run_dir / 'explanation_manifest.json').write_text(json.dumps(manifest))
    (run_dir / 'binding.json').write_text(json.dumps({'model_run': {'final_eval': final_eval}}))


def build(root, **kw):
    # seed 1234: builtin plus .2/.4 (mean .3); seed 2025: .5/.7 (mean .6) -> mean .45, SD .15*sqrt(2)
    write_run(root, 10, 'cei_mp2_seed1234', [
        {'builtin': source(.2, .1, .3, .2, .5, vs=.1), 'GradExplainer': source(.1, .1, .1, .1, .9)},
        {'builtin': source(.4, .3, .5, .4, .7, vs=.3), 'GradExplainer': {'status': 'failed', 'error': 'x'}}],
        failed=1, **kw)
    write_run(root, 10, 'cei_mp2_seed2025', [
        {'builtin': source(.5, .2, .4, .3, .5, vs=.2), 'GradExplainer': source(.3, .1, .2, .2, .8)},
        {'builtin': source(.7, .4, .6, .5, .9, vs=.4), 'GradExplainer': source(.5, .1, .4, .2, .6)}], **kw)


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
    (tmp_path / 'in' / 'driver_manifest.json').write_text(json.dumps({'protocol': 'fair_v10t', 'fold': 'test'}))
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
    (tmp_path / 'driver' / 'driver_manifest.json').write_text(json.dumps({'protocol': 'fair_v10t', 'fold': 'validation'}))
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
