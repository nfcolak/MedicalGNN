"""build_delay passes the v5 build options (level source, diagnosis policy, age policy) to build.run."""
import json
import sys

import pytest

from data.s4_graph import build, build_delay

V5_FLAGS = ['--level-source', 'row-range', '--diagnosis-policy', 'v5', '--age-policy', 'visit']


def make_delay_artifact(root):
    """Minimal first_lab_delay artifact: manifest.json + verification.json (synthetic, no data)."""
    root.mkdir()
    (root / 'manifest.json').write_text(json.dumps({
        'cutoff_policy': 'first_lab_plus_delay', 'delay_minutes': 60,
        'temporal_clean': False, 'early_triage_eligible': False,
        'inherits_index_sha256': 'a' * 64, 'event_index_sha256': 'b' * 64,
        'cutoff_clipped_at_outtime': 0}))
    (root / 'verification.json').write_text(json.dumps({'status': 'verified', 'delay_minutes': 60}))
    return root


def run_main(monkeypatch, tmp_path, extra_flags, execute):
    inherited = make_delay_artifact(tmp_path / 'delay')
    argv = ['build_delay', '--inherit-from', str(inherited), '--raw-root', str(tmp_path / 'raw'),
            '--canonical', str(tmp_path / 'canonical.json'), '--knowledge', str(tmp_path / 'knowledge.csv'),
            '--output', str(tmp_path / 'out')] + extra_flags
    if execute:
        argv.append('--execute')
    monkeypatch.setattr(sys, 'argv', argv)
    captured = {}

    def fake_run(args, **kwargs):
        captured['args'] = args
        captured['kwargs'] = kwargs

    monkeypatch.setattr(build, 'run', fake_run)
    monkeypatch.setattr(build, 'plan_thresholds', lambda path: {'path': str(path)})
    build_delay.main()
    return captured


def test_execute_passes_v5_options_to_build_run(monkeypatch, tmp_path):
    captured = run_main(monkeypatch, tmp_path,
                        V5_FLAGS + ['--with-diagnosis', '--thresholds', str(tmp_path / 't.csv')], execute=True)
    args = captured['args']
    assert args.level_source == 'row-range'
    assert args.diagnosis_policy == 'v5'
    assert args.age_policy == 'visit'


def test_execute_defaults_are_csv_v3_anchor(monkeypatch, tmp_path):
    captured = run_main(monkeypatch, tmp_path, [], execute=True)
    args = captured['args']
    assert (args.level_source, args.diagnosis_policy, args.age_policy) == ('csv', 'v3', 'anchor')


def test_execute_keeps_delay_policy_and_manifest_extra(monkeypatch, tmp_path):
    captured = run_main(monkeypatch, tmp_path, [], execute=True)
    assert captured['kwargs']['inherited_validator'] is build_delay.check_delay_manifest
    assert captured['kwargs']['manifest_extra']['cutoff_policy'] == 'first_lab_plus_delay'
    assert captured['kwargs']['manifest_extra']['delay_minutes'] == 60


def test_plan_lists_v5_options_and_runs_nothing(monkeypatch, tmp_path, capsys):
    captured = run_main(monkeypatch, tmp_path,
                        V5_FLAGS + ['--with-diagnosis', '--thresholds', str(tmp_path / 't.csv')], execute=False)
    plan = json.loads(capsys.readouterr().out)
    assert plan['status'] == 'not_executed'
    assert plan['options'] == {'level_source': 'row-range', 'diagnosis_policy': 'v5', 'age_policy': 'visit'}
    assert 'args' not in captured
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('flags, options', [
    (['--age-policy', 'visit'], {'level_source': 'csv', 'diagnosis_policy': 'v3', 'age_policy': 'visit'}),
    (['--level-source', 'row-range', '--thresholds', 'T'],
     {'level_source': 'row-range', 'diagnosis_policy': 'v3', 'age_policy': 'anchor'}),
    (['--diagnosis-policy', 'v5', '--with-diagnosis'],
     {'level_source': 'csv', 'diagnosis_policy': 'v5', 'age_policy': 'anchor'}),
], ids=['age-policy-only', 'level-source-only', 'diagnosis-policy-only'])
def test_plan_lists_options_when_a_single_flag_is_non_default(monkeypatch, tmp_path, capsys, flags, options):
    flags = [str(tmp_path / 't.csv') if f == 'T' else f for f in flags]
    run_main(monkeypatch, tmp_path, flags, execute=False)
    plan = json.loads(capsys.readouterr().out)
    assert plan['options'] == options


def test_plan_without_options_is_unchanged(monkeypatch, tmp_path, capsys):
    run_main(monkeypatch, tmp_path, [], execute=False)
    plan = json.loads(capsys.readouterr().out)
    assert 'options' not in plan
    assert plan['pipeline'] == 'data.s4_graph.build.run'


def test_v5_diagnosis_policy_requires_with_diagnosis(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, ['--diagnosis-policy', 'v5'], execute=False)
    assert '--diagnosis-policy v5 needs --with-diagnosis' in capsys.readouterr().err


def test_row_range_requires_thresholds(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, ['--level-source', 'row-range'], execute=False)
    assert '--level-source row-range needs --thresholds' in capsys.readouterr().err


def test_unknown_age_policy_is_rejected(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, ['--age-policy', 'bogus'], execute=False)
    assert "invalid choice: 'bogus'" in capsys.readouterr().err
