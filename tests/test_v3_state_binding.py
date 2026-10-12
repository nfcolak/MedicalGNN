"""fv10-m3-03: a CEI v3 state must belong to the run (top-k, TRAIN sample, vocabulary tokens, preprocessing hash).

Synthetic only: no data file is opened (the sample check runs on a tiny targets CSV written here).
"""
from __future__ import annotations

import hashlib
import json
from argparse import Namespace

import pytest
import torch

from cei import plugin_cei_gnn_v3 as plugin
from cei.studies import cei_v3_study as study
from tests.test_cei_gnn_v3_core import (CLASSES, EDGE_DIM, HIDDEN, K, NODE_DIM, NUM_TOKENS, TOKEN_DIM, TOKENS,
                                        TRIPLES, _state_document, _write_state)


def _build(state_path, **namespace):
    from core.registry import METHOD_REGISTRY
    args = Namespace(method_options={'arm': 'C', 'v3_state': str(state_path), 'k': K}, seed=1234, **namespace)
    torch.manual_seed(1234)
    return METHOD_REGISTRY['cei_gnn_v3'](
        num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_classes=CLASSES, hidden=HIDDEN,
        layers=1, dropout=0.0, token_dim=TOKEN_DIM, num_triples=TRIPLES, args=args)


def _targets(tmp_path, labels):
    path = tmp_path / 'targets.csv'
    rows = ['sample_id,subject_id,split,target']
    for i, label in enumerate(labels):
        rows.append(f's{i:03d},p{i:03d},train,{label}')
    path.write_text('\n'.join(rows) + '\n')
    return path


def test_old_state_without_new_fields_still_loads(tmp_path):
    state = plugin.load_v3_state(_write_state(tmp_path))
    assert state.top_k_labels is None and state.train_sample_ids_sha256 is None
    adapter = _build(_write_state(tmp_path), top_k_labels=17)   # nothing to compare -> accepted
    assert 'v3_state_top_k_labels' not in adapter.run_config()


def test_state_top_k_must_equal_the_run_top_k(tmp_path):
    document = _state_document()
    document['top_k_labels'] = 10
    path = _write_state(tmp_path, document)
    adapter = _build(path, top_k_labels=10)
    assert adapter.run_config()['v3_state_top_k_labels'] == 10
    with pytest.raises(ValueError, match=r'top_k_labels=10.*top_k_labels=17'):
        _build(path, top_k_labels=17)


def test_state_fitted_on_another_train_sample_is_refused(tmp_path):
    targets = _targets(tmp_path, [0, 1] * 20)
    from core import train as train_module
    loaded = train_module.load_targets(targets)
    ids = train_module.sample_train_ids(loaded, 10, 1234)
    document = _state_document()
    document['train_sample_ids_sha256'] = study.train_sample_ids_sha256(ids)
    good = _write_state(tmp_path, document)
    run = dict(targets=str(targets), train_limit=10, sample_seed=1234)
    _build(good, **run)
    document['train_sample_ids_sha256'] = hashlib.sha256(b'other').hexdigest()
    with pytest.raises(ValueError, match='another TRAIN sample'):
        _build(_write_state(tmp_path, document, name='other.json'), **run)


def test_tokens_and_preprocessing_hash_are_compared_with_the_run(tmp_path):
    state = plugin.load_v3_state(_write_state(tmp_path))
    prep = {'vocabulary': {'tokens': list(TOKENS), 'min_count': 20},
            'node_feature_layout': list(state.node_feature_layout)}
    digest = hashlib.sha256((json.dumps(prep, indent=2, sort_keys=True) + '\n').encode()).hexdigest()
    with pytest.raises(ValueError, match='preprocessing_sha256'):
        plugin.check_state_against_preprocessing(state, prep)       # synthetic state hash != this prep
    matching = _state_document()
    matching['preprocessing_sha256'] = digest
    state = plugin.load_v3_state(_write_state(tmp_path, matching, name='m.json'))
    plugin.check_state_against_preprocessing(state, prep)
    shifted = dict(prep, vocabulary={'tokens': list(reversed(TOKENS)), 'min_count': 20})
    with pytest.raises(ValueError, match='another vocabulary'):
        plugin.check_state_against_preprocessing(state, shifted)    # same count, other tokens
    adapter = _build(_write_state(tmp_path, matching, name='m.json'))
    adapter.verify_run_preprocessing(prep)


def test_build_state_records_new_fields_only_when_given():
    from tests.test_cei_gnn_v3_core import _knot_table, _universe, LAYOUT
    base = dict(K=K, knot_table=_knot_table(), universe=_universe(), vocabulary_tokens=TOKENS,
                vocabulary_min_count=20, preprocessing_sha256='0' * 64, node_feature_layout=LAYOUT)
    assert 'top_k_labels' not in plugin.build_v3_state(**base)
    document = plugin.build_v3_state(**base, top_k_labels=17, train_sample_ids_sha256='1' * 64)
    assert document['top_k_labels'] == 17 and document['train_sample_ids_sha256'] == '1' * 64


def test_studies_refuse_a_non_top10_binding_with_a_clear_message():
    with pytest.raises(ValueError, match='Top-10 only'):
        study.validate_v3_binding({'top_k_labels': 17, 'num_classes': 17}, Namespace(), )


def test_train_calls_the_model_preprocessing_hook_with_the_run_state():
    import inspect
    from core import train
    src = inspect.getsource(train.run)
    assert src.index('state = preprocessing_state(prep)') < src.index("getattr(model, 'verify_run_preprocessing'")
    seen = []

    class Fake:
        def verify_run_preprocessing(self, state):
            seen.append(state)

    hook = getattr(Fake(), 'verify_run_preprocessing', None)
    assert callable(hook)
    hook({'a': 1})
    assert seen == [{'a': 1}]


def test_state_hash_of_full_train_run_matches_split_and_refuses_the_numeric_sample(tmp_path):
    targets = _targets(tmp_path, [0, 1] * 20)
    from core import train as train_module
    loaded = train_module.load_targets(targets)
    ids, _dev = train_module.resolve_train_dev_ids(loaded, None, 8, 1234)
    assert len(ids) == 32
    document = _state_document()
    document['train_sample_ids_sha256'] = study.train_sample_ids_sha256(ids)
    run = dict(targets=str(targets), train_limit=None, dev_limit=8, sample_seed=1234)
    assert plugin.train_sample_ids_sha256_of_run(Namespace(**run)) == study.train_sample_ids_sha256(ids)
    _build(_write_state(tmp_path, document), **run)
    with pytest.raises(ValueError, match='another TRAIN sample'):
        _build(_write_state(tmp_path, document, name='n.json'), **dict(run, train_limit=10))
