"""compare.py: K from the data, fair_v10 grouping by top_k_labels, fair_v3 stays readable."""
import json

import numpy as np
import pytest

from comparison.top3 import compare as cmp

LEGACY = dict(train_limit=10000, sample_seed=1234, top_k_labels=10, weights='sqrt_inverse',
              selection_fold='dev', dev_limit=5000, final_eval='validation', epochs=40, patience=40)
H = {'train': 'a' * 64, 'dev': 'b' * 64, 'validation': 'c' * 64}


def make_run(tmp_path, name, K, protocol='fair_v3', seed=1, n=60, overrides=(), method='protgnn'):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % K
    proba = rng.random((n, K))
    proba[np.arange(n), y] += 0.8
    path = tmp_path / name
    path.mkdir()
    conf = np.bincount(y * K + proba.argmax(1), minlength=K * K).reshape(K, K)
    f1 = float(cmp.macro_f1_from_conf(conf))
    resolved = dict(cmp.FAIR_PROTOCOL if protocol == cmp.PROTOCOL_NAME and protocol != 'fair_v3'
                    else LEGACY, top_k_labels=K)
    if protocol == 'fair_v3':
        resolved = dict(LEGACY)
    binding = {'protocol': protocol, 'protocol_overrides': list(overrides), 'protocol_resolved': resolved,
               'test_evaluated': False, 'split_sample_ids_sha256': H, 'method': method, 'seed': seed,
               'top_k_labels': K, 'num_classes': K, 'kept_label_indices': list(range(K))}
    (path / 'binding.json').write_text(json.dumps(binding))
    (path / 'result.json').write_text(json.dumps({'status': 'completed', 'selection_fold': 'dev',
                                                  'dev_metrics': {'macro_f1': 0.5},
                                                  'metrics': {'macro_f1': f1}}))
    np.savez(path / 'validation.npz', proba=proba, y=y, subjects=np.arange(n) // 2,
             sample_ids=np.array([f's{i}' for i in range(n)]))
    return path


def test_fair_v3_k10_accepted_and_k_from_data(tmp_path):
    r = cmp.load_run(make_run(tmp_path, 'a', 10))
    assert cmp.point_metrics(r['proba'], r['y'])['macro_f1'] > 0


def test_fair_v10_k17_and_k30(tmp_path, monkeypatch):
    monkeypatch.setattr(cmp, 'PROTOCOL_NAME', 'fair_v10')
    monkeypatch.setattr(cmp, 'FAIR_PROTOCOL', dict(LEGACY, train_eval_subset=1000))
    for K in (17, 30):
        r = cmp.load_run(make_run(tmp_path, f'k{K}', K, protocol='fair_v10'))
        assert cmp.point_metrics(r['proba'], r['y'])['macro_f1'] > 0
        r['seed'] = 1
        b = cmp.bootstrap([r], [r['method']], r)
        assert b[r['method']].shape == (cmp.N_BOOT,)


def test_refusals(tmp_path, monkeypatch):
    monkeypatch.setattr(cmp, 'PROTOCOL_NAME', 'fair_v10')
    monkeypatch.setattr(cmp, 'FAIR_PROTOCOL', dict(LEGACY, train_eval_subset=1000))
    with pytest.raises(cmp.Refusal):
        cmp.load_run(make_run(tmp_path, 'ov', 17, protocol='fair_v10', overrides=['epochs']))
    with pytest.raises(cmp.Refusal):  # K not allowed
        cmp.load_run(make_run(tmp_path, 'k12', 12, protocol='fair_v10'))
    p = make_run(tmp_path, 'bad', 17, protocol='fair_v10')
    b = json.loads((p / 'binding.json').read_text())
    b['num_classes'] = 10
    (p / 'binding.json').write_text(json.dumps(b))
    with pytest.raises(cmp.Refusal):
        cmp.load_run(p)


def test_groups_not_mixed(tmp_path, monkeypatch):
    monkeypatch.setattr(cmp, 'PROTOCOL_NAME', 'fair_v10')
    monkeypatch.setattr(cmp, 'FAIR_PROTOCOL', dict(LEGACY, train_eval_subset=1000))
    a = cmp.load_run(make_run(tmp_path, 'a', 17, protocol='fair_v10', seed=1))
    b = cmp.load_run(make_run(tmp_path, 'b', 25, protocol='fair_v10', seed=2))
    with pytest.raises(cmp.Refusal):
        cmp.check_and_align([a, b])
    c = cmp.load_run(make_run(tmp_path, 'c', 10, seed=3))
    d = cmp.load_run(make_run(tmp_path, 'd', 10, protocol='fair_v10', seed=4))
    with pytest.raises(cmp.Refusal):
        cmp.check_and_align([c, d])


def test_y_disagreement_refused_same_order(tmp_path):
    a = cmp.load_run(make_run(tmp_path, 'a', 10, seed=1))
    b = cmp.load_run(make_run(tmp_path, 'b', 10, seed=2))
    b['y'] = (b['y'] + 1) % 10
    with pytest.raises(cmp.Refusal):
        cmp.check_and_align([a, b])


def test_bootstrap_covers_all_pairs_of_all_methods(tmp_path):
    names = ['gsat', 'graphcare', 'cei_gnn_mp2', 'protgnn', 'gchm_pna']
    for n in (4, 5):
        paths = [str(make_run(tmp_path, f'n{n}_{m}', 10, seed=i + 1, method=m))
                 for i, m in enumerate(names[:n])]
        out = tmp_path / f'out{n}'
        assert cmp.main([a for p in paths for a in ('--run', p)] + ['--out', str(out)]) == 0
        doc = json.loads((out / 'comparison.json').read_text())
        assert sorted(doc['bootstrap_methods']) == sorted(names[:n])
        assert len(doc['pairs']) == n * (n - 1) // 2
