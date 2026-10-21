"""Explanation driver: plan, binding checks, no-overwrite and execution with a stubbed runner call."""
import json
import os
import sys
import threading
import time

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


# --- keep-going, resume, jobs, timing -------------------------------------------------------------------------

RUN_FIELDS = {'k', 'method', 'seed', 'output', 'returncode', 'status', 'seconds', 'started_at', 'max_rss_mb'}


def name_of(command):
    return '/'.join(command[command.index('--output') + 1].split('/')[-2:])


def load(out, name='driver_manifest.json'):
    return json.loads((out / name).read_text())


def test_manifest_fields_for_every_run(tmp_path):
    runs = make_runs(tmp_path / 'runs')
    out = tmp_path / 'out'
    driver.main(args(runs, out, '--execute'), executor=lambda c, e: 0)
    manifest = load(out)
    assert manifest['started_at'].endswith('Z')
    for run in manifest['runs']:
        assert RUN_FIELDS <= set(run)
        assert run['status'] == 'ok' and run['returncode'] == 0
        assert isinstance(run['seconds'], float) and run['started_at'].endswith('Z') and run['max_rss_mb'] is None


def test_dict_executor_records_peak_memory(tmp_path):
    runs = make_runs(tmp_path / 'runs')
    out = tmp_path / 'out'
    driver.main(args(runs, out, '--execute'), executor=lambda c, e: {'returncode': 0, 'max_rss_mb': 123.4})
    assert [r['max_rss_mb'] for r in load(out)['runs']] == [123.4] * 4


def test_keep_going_continues_after_failure_and_exits_1(tmp_path):
    runs = make_runs(tmp_path / 'runs')
    out = tmp_path / 'out'
    seen = []

    def executor(command, env):
        seen.append(command)
        return 3 if len(seen) == 2 else 0

    assert driver.main(args(runs, out, '--execute', '--keep-going'), executor=executor) == 1
    assert len(seen) == 4
    manifest = load(out)
    assert manifest['complete'] is False
    assert [r['returncode'] for r in manifest['runs']] == [0, 3, 0, 0]
    assert [r['status'] for r in manifest['runs']] == ['ok', 'failed', 'ok', 'ok']


def test_without_keep_going_keeps_stop_message(tmp_path, capsys):
    runs = make_runs(tmp_path / 'runs')
    out = tmp_path / 'out'
    assert driver.main(args(runs, out, '--execute'), executor=lambda c, e: 5) == 5
    assert 'top10/gsat_seed1234 failed with exit status 5; stopped' in capsys.readouterr().err
    assert [r['status'] for r in load(out)['runs']] == ['failed', None, None, None]


def finished_run(out, name, explained=7, failed=1):
    folder = out / name
    folder.mkdir(parents=True)
    (folder / 'explanation_manifest.json').write_text(json.dumps({'explained_count': explained,
                                                                  'failed_count': failed}))


def test_resume_skips_finished_runs_and_keeps_old_manifest(tmp_path):
    runs = make_runs(tmp_path / 'runs')
    out = tmp_path / 'out'
    out.mkdir()
    finished_run(out, 'top10/gsat_seed1234')
    finished_run(out, 'top10/gsat_seed7', explained=9, failed=0)
    old = {'started_at': '2026-10-21T10:00:00Z', 'runs': [
        {'output': 'top10/gsat_seed1234', 'seconds': 12.3, 'started_at': '2026-10-21T10:00:01Z',
         'max_rss_mb': 800.0}]}
    (out / 'driver_manifest.json').write_text(json.dumps(old))
    seen = []
    assert driver.main(args(runs, out, '--execute', '--resume'),
                       executor=lambda c, e: seen.append(name_of(c)) or 0) == 0
    assert seen == ['top10/cei_mp2_seed1234', 'top10/cei_mp2_seed7']
    assert load(out, 'driver_manifest.2026-10-21T10-00-00Z.json') == old
    runs_out = load(out)['runs']
    assert [r['status'] for r in runs_out] == ['skipped_existing', 'skipped_existing', 'ok', 'ok']
    first = runs_out[0]
    assert (first['seconds'], first['started_at'], first['max_rss_mb']) == (12.3, '2026-10-21T10:00:01Z', 800.0)
    assert (first['explained_count'], first['failed_count']) == (7, 1)
    assert runs_out[1]['explained_count'] == 9 and runs_out[1]['seconds'] is None
    assert load(out)['complete'] is True


def test_resume_refuses_interrupted_folder_and_stray_files(tmp_path):
    runs = make_runs(tmp_path / 'runs')
    out = tmp_path / 'out'
    (out / 'top10' / 'gsat_seed1234').mkdir(parents=True)
    calls = []
    with pytest.raises(SystemExit, match='gsat_seed1234.*interrupted'):
        driver.main(args(runs, out, '--execute', '--resume'), executor=lambda c, e: calls.append(c) or 0)
    (out / 'top10' / 'gsat_seed1234').rmdir()
    (out / 'notes.txt').write_text('x')
    with pytest.raises(SystemExit, match='notes.txt.*outside the planned runs'):
        driver.main(args(runs, out, '--execute', '--resume'), executor=lambda c, e: calls.append(c) or 0)
    (out / 'notes.txt').unlink()
    (out / 'top10' / 'other_seed1').mkdir()
    with pytest.raises(SystemExit, match='other_seed1'):
        driver.main(args(runs, out, '--execute', '--resume'), executor=lambda c, e: calls.append(c) or 0)
    assert calls == []


def test_resume_keeps_stop_at_first_failure(tmp_path):
    runs = make_runs(tmp_path / 'runs')
    out = tmp_path / 'out'
    out.mkdir()
    finished_run(out, 'top10/gsat_seed1234')
    seen = []

    def executor(command, env):
        seen.append(command)
        return 4

    assert driver.main(args(runs, out, '--execute', '--resume'), executor=executor) == 4
    assert len(seen) == 1


def test_jobs_limit_concurrency_and_set_thread_env(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'cpu_count', lambda: 8)
    runs = make_runs(tmp_path / 'runs', methods=('gsat',), seeds=(1, 2, 3, 4, 5))
    out = tmp_path / 'out'
    lock, state, envs = threading.Lock(), {'now': 0, 'max': 0}, []

    def executor(command, env):
        with lock:
            envs.append(env)
            state['now'] += 1
            state['max'] = max(state['max'], state['now'])
        time.sleep(0.05)
        with lock:
            state['now'] -= 1
        return 0

    code = driver.main(args(runs, out, '--execute', '--jobs', '3', methods='gsat', seeds='1,2,3,4,5'),
                       executor=executor)
    assert code == 0 and state['max'] == 3
    assert all(e['OMP_NUM_THREADS'] == '2' and e['MKL_NUM_THREADS'] == '2' for e in envs)
    assert [r['returncode'] for r in load(out)['runs']] == [0] * 5


def test_jobs_write_manifest_after_each_finished_run_with_simultaneous_finishes(tmp_path, monkeypatch):
    runs = make_runs(tmp_path / 'runs', methods=('gsat',), seeds=(1, 2, 3))
    out = tmp_path / 'out'
    barrier = threading.Barrier(2)
    writes, real = [], driver.write_manifest

    def spy(out_dir, a, entries, results, complete, started_at=None):
        writes.append(sum(1 for r in results.values() if r.get('status')))
        return real(out_dir, a, entries, results, complete, started_at)

    monkeypatch.setattr(driver, 'write_manifest', spy)

    def executor(command, env):
        if name_of(command).endswith(('seed1', 'seed2')):
            barrier.wait(timeout=5)
        return 0

    assert driver.main(args(runs, out, '--execute', '--jobs', '2', methods='gsat', seeds='1,2,3'),
                       executor=executor) == 0
    # initial write, one per finished run, final write: finished runs seen never skip a count
    assert writes == [0, 1, 2, 3, 3]
    assert [r['status'] for r in load(out)['runs']] == ['ok', 'ok', 'ok']


def test_jobs_failure_stops_new_runs_without_keep_going(tmp_path):
    runs = make_runs(tmp_path / 'runs', methods=('gsat',), seeds=(1, 2, 3, 4))
    out = tmp_path / 'out'
    started = []

    def executor(command, env):
        started.append(name_of(command))
        if started[-1].endswith('seed1'):
            return 6
        time.sleep(0.1)
        return 0

    code = driver.main(args(runs, out, '--execute', '--jobs', '2', methods='gsat', seeds='1,2,3,4'),
                       executor=executor)
    assert code == 6 and len(started) == 2
    assert [r['status'] for r in load(out)['runs']] == ['failed', 'ok', None, None]


def test_jobs_one_matches_sequential_order_and_manifest(tmp_path):
    runs = make_runs(tmp_path / 'runs')
    order = {}
    for label, extra in (('plain', ()), ('jobs1', ('--jobs', '1'))):
        out = tmp_path / label
        seen = []
        driver.main(args(runs, out, '--execute', *extra), executor=lambda c, e: seen.append((name_of(c), e)) or 0)
        manifest = load(out)
        order[label] = ([n for n, _ in seen], [r['output'] for r in manifest['runs']],
                        [(r['returncode'], r['status']) for r in manifest['runs']],
                        [e.get('OMP_NUM_THREADS') for _, e in seen], manifest['complete'])
    assert order['plain'] == order['jobs1']


def test_jobs_must_be_positive(tmp_path):
    runs = make_runs(tmp_path / 'runs')
    with pytest.raises(SystemExit, match='--jobs'):
        driver.main(args(runs, tmp_path / 'out', '--plan', '--jobs', '0'))


def test_default_executor_reports_status_and_peak_memory():
    result = driver.run_command([sys.executable, '-c', 'import sys; sys.exit(3)'], dict(os.environ))
    assert result['returncode'] == 3 and result['max_rss_mb'] > 0
