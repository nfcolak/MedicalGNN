"""fair_v10 protocol: top-K as a run parameter, new defaults, grad-clip record, early test-fold skip."""
import json

import numpy as np
import pytest
import torch.nn as nn

from core import contracts, train
from tests.test_clinical_inputs import graph, membership


def normalized(*argv):
    parser = train.parser()
    base = ['--artifact', 'a', '--targets', 't', '--output', 'o']
    return train.normalize_method_args(parser.parse_args(base + list(argv)), parser)


def test_default_protocol_is_fair_v10_with_train_eval_subset():
    assert train.PROTOCOL_NAME == 'fair_v10'
    assert train.FAIR_PROTOCOL['train_eval_subset'] == 1000
    assert train.FAIR_PROTOCOL['top_k_labels'] == 10 and train.FAIR_PROTOCOL['epochs'] == 40
    for method in ('protgnn', 'gsat', 'graphcare', 'clinical_gnn'):
        args = normalized('--method', method)
        assert args.protocol_overrides == []
        assert args.protocol_resolved == train.FAIR_PROTOCOL and args.train_eval_subset == 1000


@pytest.mark.parametrize('k', [10, 17, 25, 30])
def test_allowed_top_k_is_a_parameter_not_an_override(k):
    args = normalized('--method', 'protgnn', '--top-k-labels', str(k))
    assert args.protocol_overrides == [] and args.protocol_resolved['top_k_labels'] == k


@pytest.mark.parametrize('extra,name', [(['--top-k-labels', '12'], 'top_k_labels'),
                                        (['--epochs', '25'], 'epochs'),
                                        (['--train-eval-subset', '0'], 'train_eval_subset'),
                                        (['--train-limit', '5000'], 'train_limit')])
def test_other_changes_stay_overrides(extra, name):
    assert normalized('--method', 'protgnn', *extra).protocol_overrides == [name]


def test_top_k_and_epoch_override_together_list_only_the_override():
    args = normalized('--method', 'protgnn', '--top-k-labels', '17', '--epochs', '25')
    assert args.protocol_overrides == ['epochs']


def test_absent_class_gets_count_one_weight():
    w = train.class_weights(np.array([0, 0, 1, 1, 1, 1]), 'sqrt_inverse', 3)
    assert w[2] == pytest.approx(np.sqrt(6 / 3))   # count 1 -> sqrt(N/K), not 1.0
    assert w[0] == pytest.approx(np.sqrt(6 / (3 * 2)))


def test_grad_clip_settings_match_clip_gradients():
    model = nn.Linear(2, 2)
    assert train.grad_clip_settings('clinical_gnn', model) == {'value': None, 'norm': 5.0}
    assert train.grad_clip_settings('protgnn', model) == {'value': 2.0, 'norm': None}
    assert train.grad_clip_settings('gsat', model) == {'value': None, 'norm': None}


def test_wanted_ids_skip_decoding_other_rows(tmp_path):
    graphs = [graph(f's{i}') for i in range(5)]
    gpath, mpath = tmp_path / 'g.jsonl', tmp_path / 'm.jsonl'
    gpath.write_text(''.join(json.dumps(g) + '\n' for g in graphs))
    mpath.write_text(''.join(json.dumps(membership(g)) + '\n' for g in graphs))
    full = list(contracts.iter_graphs_with_membership(gpath, mpath))
    picked = list(contracts.iter_graphs_with_membership(gpath, mpath, wanted_ids={'s1', 's3'}))
    assert [g['sample_id'] for g, _ in picked] == ['s1', 's3']
    assert picked == [full[1], full[3]]
    # a corrupted graph body of an unwanted row is never decoded; the id envelope still is checked
    lines = gpath.read_text().splitlines()
    lines[0] = '{"sample_id": "s0", "nodes": BROKEN'
    gpath.write_text('\n'.join(lines) + '\n')
    assert [g['sample_id'] for g, _ in contracts.iter_graphs_with_membership(
        gpath, mpath, wanted_ids={'s1'})] == ['s1']
    with pytest.raises(ValueError):
        list(contracts.iter_graphs_with_membership(gpath, mpath))
    mpath.write_text(mpath.read_text().replace('"s0"', '"zz"', 1))
    with pytest.raises(ValueError, match='mismatch'):
        list(contracts.iter_graphs_with_membership(gpath, mpath, wanted_ids={'s1'}))


def test_run_records_fair_v10_binding_and_result_keys(monkeypatch, tmp_path):
    from tests.test_cei_mp2_train_hooks import TinyHooks, run_stub
    binding, history, out, _ = run_stub(monkeypatch, tmp_path, '--top-k-labels', '17', adapter=TinyHooks)
    result = json.loads((out / 'result.json').read_text())
    assert binding['protocol'] == 'fair_v10' and binding['protocol_overrides'] == ['dev_limit', 'epochs']   # stub: 2 epochs, dev 12 (K=17 is none)
    assert binding['protocol_resolved']['top_k_labels'] == 17 and binding['top_k_labels'] == 17
    assert binding['num_classes'] == 6 and binding['kept_label_indices'] == list(range(6))   # stub keeps 6
    assert binding['grad_clip'] == {'value': None, 'norm': None}
    assert binding['graph_sha256'] == 'g' and len(binding['graph_manifest_sha256']) == 64
    assert 'level_source' not in binding and 'threshold_semantics' not in binding   # absent in the manifest
    assert binding['train_eval_subset'] == 24   # protocol default 1000, capped at the 24 stub graphs
    assert 'train_eval_macro_f1' in history[0]
    selected = result['selected_epoch']
    assert result['best_epoch_is_last'] == (selected == len(history))
    assert result['last5_dev_macro_f1_mean'] == pytest.approx(
        float(np.mean([row['macro_f1'] for row in history[-5:]])), abs=1e-6)


def _split_targets():
    # 60 TRAIN patients with 1-3 rows each (class by index), 5 validation patients
    targets = {}
    for p in range(60):
        for v in range(1 + p % 3):
            targets[f'p{p:02d}v{v}'] = (p % 3, 'train', f'p{p:02d}')
    for p in range(5):
        targets[f'val{p}'] = (p % 3, 'validation', f'v{p}')
    return targets


def test_full_train_is_the_protocol_default_and_numeric_limit_an_override():
    assert train.FAIR_PROTOCOL['train_limit'] is None
    assert normalized('--method', 'protgnn').protocol_overrides == []
    assert normalized('--method', 'protgnn').train_limit is None
    assert normalized('--method', 'protgnn', '--train-limit', '10000').protocol_overrides == ['train_limit']


def test_split_full_train_dev_is_whole_patients_disjoint_and_complete():
    from data.s5_filter_split.selection import split_full_train_dev
    targets = _split_targets()
    tr, dev = split_full_train_dev(targets, 40, 1234)
    assert len(dev) == 40 and not (tr & dev)
    subj = lambda ids: {targets[s][2] for s in ids}
    assert not (subj(tr) & subj(dev))
    train_fold = {s for s, e in targets.items() if e[1] == 'train'}
    assert tr | dev | {s for s in train_fold if targets[s][2] in subj(dev)} == train_fold
    assert tr == {s for s in train_fold if targets[s][2] not in subj(dev)}   # every other TRAIN row trains
    assert (tr, dev) == split_full_train_dev(targets, 40, 1234)
    assert dev != split_full_train_dev(targets, 40, 99)[1]
    elig = {s for s in train_fold if s.endswith('v0')}
    tr2, dev2 = split_full_train_dev(targets, 10, 1234, elig)
    assert dev2 <= elig and not (subj(tr2) & subj(dev2))
    with pytest.raises(ValueError):
        split_full_train_dev(targets, 10 ** 6, 1234)


def test_resolve_train_dev_ids_numeric_path_is_the_old_draw():
    from data.s5_filter_split.selection import (resolve_train_dev_ids, sample_train_ids,
                                                select_dev_ids)
    targets = _split_targets()
    ids = sample_train_ids(targets, 15, 7)
    assert resolve_train_dev_ids(targets, 15, 10, 7) == (ids, select_dev_ids(targets, ids, 10, 7))
    assert resolve_train_dev_ids(targets, None, None, 7) == (sample_train_ids(targets, None, 7), frozenset())
