"""Explanation driver: plan, binding checks, no-overwrite and execution with a stubbed runner call."""
import json
import sys

import pytest

from comparison.explain import run as driver


def make_runs(root, ks=(10,), methods=('gsat', 'cei_mp2'), seeds=(1234, 7), **overrides):
    for k in ks:
        for method in methods:
            for seed in seeds:
                run_dir = root / f'top{k}' / f'{method}_seed{seed}'
                run_dir.mkdir(parents=True)
                binding = {'protocol': 'fair_v10t', 'top_k_labels': k, 'final_eval': 'test', 'seed': seed,
                           'method': driver.BINDING_METHOD[method]}
                binding.update(overrides)
                (run_dir / 'binding.json').write_text(json.dumps(binding))
    return root


def args(runs, out, *extra, **kw):
    return ['--runs', str(runs), '--out', str(out), '--ks', kw.get('ks', '10'),
            '--methods', kw.get('methods', 'gsat,cei_mp2'), '--seeds', kw.get('seeds', '1234,7'), *extra]


def test_plan_prints_one_command_per_run_and_executes_nothing(tmp_path, capsys):
    runs = make_runs(tmp_path / 'runs')
    calls = []
    code = driver.main(args(runs, tmp_path / 'out', '--plan'), executor=lambda c, e: calls.append(c) or 0)
    text = capsys.readouterr().out
    assert code == 0 and calls == []
    commands = [line for line in text.splitlines() if 'core.explain.runner' in line]
    assert len(commands) == 4
    assert all('--fold test' in c and '--sample-patients 500' in c and sys.executable in c for c in commands)
    assert 'top10/cei_mp2_seed7' in text
    assert 'fold test (report fold)' in text
    assert not (tmp_path / 'out').exists()


def test_without_flag_prints_plan_and_exits_2(tmp_path, capsys):
    runs = make_runs(tmp_path / 'runs')
    assert driver.main(args(runs, tmp_path / 'out'), executor=lambda c, e: 0) == 2
    assert 'core.explain.runner' in capsys.readouterr().out


def test_validation_fold_is_labelled_decision(tmp_path, capsys):
    runs = make_runs(tmp_path / 'runs')
    driver.main(args(runs, tmp_path / 'out', '--plan', '--fold', 'validation'))
    text = capsys.readouterr().out
    assert 'validation (decision fold)' in text and '--fold validation' in text


@pytest.mark.parametrize('field, value', [('protocol', 'fair_v9'), ('top_k_labels', 20),
                                          ('final_eval', 'validation'), ('method', 'graphcare')])
def test_binding_mismatch_is_refused_before_anything_runs(tmp_path, field, value):
    runs = make_runs(tmp_path / 'runs', methods=('gsat',), **{field: value})
    calls = []
    with pytest.raises(SystemExit) as info:
        driver.main(args(runs, tmp_path / 'out', '--execute', methods='gsat'),
                    executor=lambda c, e: calls.append(c) or 0)
    assert field in str(info.value) and calls == []


def test_missing_binding_and_wrong_seed_are_refused(tmp_path):
    runs = make_runs(tmp_path / 'runs', methods=('gsat',), seeds=(1234,))
    with pytest.raises(SystemExit, match='binding.json missing'):
        driver.main(args(runs, tmp_path / 'out', '--plan', methods='gsat', seeds='1234,7'))
    bound = json.loads((runs / 'top10' / 'gsat_seed1234' / 'binding.json').read_text())
    bound['seed'] = 7
    (runs / 'top10' / 'gsat_seed1234' / 'binding.json').write_text(json.dumps(bound))
    with pytest.raises(SystemExit, match='seed'):
        driver.main(args(runs, tmp_path / 'out', '--plan', methods='gsat', seeds='1234'))


def test_filled_out_dir_is_refused(tmp_path):
    runs = make_runs(tmp_path / 'runs')
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'something').write_text('x')
    calls = []
    with pytest.raises(SystemExit, match='new or empty'):
        driver.main(args(runs, out, '--execute'), executor=lambda c, e: calls.append(c) or 0)
    assert calls == []
    out2 = tmp_path / 'empty'
    out2.mkdir()
    assert driver.main(args(runs, out2, '--plan')) == 0


def test_dropped_and_unknown_methods_are_refused(tmp_path):
    runs = make_runs(tmp_path / 'runs')
    with pytest.raises(SystemExit, match='not part of the explanation study'):
        driver.main(args(runs, tmp_path / 'out', '--plan', methods='gchm_pna'))
    with pytest.raises(SystemExit, match='unknown method'):
        driver.main(args(runs, tmp_path / 'out', '--plan', methods='nope'))


def test_execute_runs_in_order_and_writes_manifest(tmp_path):
    runs = make_runs(tmp_path / 'runs')
    out = tmp_path / 'out'
    seen = []

    def executor(command, env):
        seen.append((command, env))
        return 0

    assert driver.main(args(runs, out, '--execute'), executor=executor) == 0
    assert [c[c.index('--output') + 1].split('/')[-1] for c, _ in seen] == \
        ['gsat_seed1234', 'gsat_seed7', 'cei_mp2_seed1234', 'cei_mp2_seed7']
    assert all(e['PYTHONDONTWRITEBYTECODE'] == '1' for _, e in seen)
    manifest = json.loads((out / 'driver_manifest.json').read_text())
    assert manifest['complete'] is True and manifest['fold'] == 'test' and manifest['fold_role'] == 'report'
    assert manifest['planned_runs'] == 4 and [r['returncode'] for r in manifest['runs']] == [0, 0, 0, 0]
    assert str(tmp_path) not in json.dumps(manifest)


def test_execute_stops_on_first_failing_run(tmp_path):
    runs = make_runs(tmp_path / 'runs')
    out = tmp_path / 'out'
    seen = []

    def executor(command, env):
        seen.append(command)
        return 3 if len(seen) == 2 else 0

    assert driver.main(args(runs, out, '--execute'), executor=executor) == 3
    assert len(seen) == 2
    manifest = json.loads((out / 'driver_manifest.json').read_text())
    assert manifest['complete'] is False
    assert [r['returncode'] for r in manifest['runs']] == [0, 3, None, None]
