"""mp2 hook regions of core/train.py (adapter build_optimizer / lr_scheduler, --train-eval-subset,
binding and history records).

Synthetic only: the data layer of ``core.train.run`` is replaced by in-memory random graphs and the adapter is a
tiny dropout classifier, so the real training loop runs without any clinical data.
"""
import json
import math
import random

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from core import registry as methods, train
from core.contracts import sample_ids_sha256
from core.method_base import ClinicalMethodAdapter, MethodOutput

NODE_DIM, EDGE_DIM, CLASSES = 6, 3, 30
CREATED = []   # every adapter instance built by the stub runs, in creation order


class Tiny(ClinicalMethodAdapter):
    """Mean-pooled linear classifier with dropout (dropout draws from the global torch RNG)."""
    runner_defaults = dict(hidden=8, layers=1, dropout=0.5, lr=1e-2, weight_decay=0.0,
                           batch_size=8, epochs=2, patience=40, min_delta=0.0)

    def __init__(self, **kw):
        super().__init__()
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(NODE_DIM, kw['num_classes'])
        CREATED.append(self)

    def forward(self, batch, *, epoch):
        h = self.drop(global_mean_pool(batch.x, batch.batch))
        return MethodOutput(logits=self.fc(h), auxiliary_loss=torch.zeros(()), diagnostics={})

    def on_epoch_start(self, epoch, train_loader):
        pass

    def optimizer_groups(self, args):
        return [{"params": list(self.parameters())}]

    def run_config(self):
        n = sum(p.numel() for p in self.parameters())
        return {"method": "tiny", "architecture": {"parameter_count": n}}


class TinyHooks(Tiny):
    """Tiny + the hook shapes of the mp2 adapter: AdamW (decoupled decay) and a cosine LambdaLR."""

    def build_optimizer(self, args):
        return torch.optim.AdamW(self.optimizer_groups(args), lr=args.lr, weight_decay=0.01)

    def lr_scheduler(self, optimizer, epochs):
        epochs = max(int(epochs), 1)
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda e: 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(e, epochs) / epochs)))


class TinyAdamHooks(Tiny):
    """mp2 DEFAULT shape: build_optimizer returns plain Adam with the runner's weight decay."""

    def build_optimizer(self, args):
        return torch.optim.Adam(self.optimizer_groups(args), lr=args.lr, weight_decay=args.weight_decay)


class Gated(Tiny):
    def early_stopping_start(self):
        return 2


def _graphs(n, offset, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        d = Data(x=torch.randn(5, NODE_DIM, generator=g), edge_index=torch.zeros((2, 0), dtype=torch.long),
                 edge_attr=torch.zeros((0, EDGE_DIM)))
        d.y = torch.tensor([i % 6])
        d.sample_id = f's{offset + i}'
        d.subject = offset + i
        d.rewired_edge_count = 0
        out.append(d)
    return out


def run_stub(monkeypatch, tmp_path, *extra, adapter=Tiny, name='tiny'):
    """Run the real ``core.train.run`` on the stub; return (binding, history, out_dir, splits)."""
    monkeypatch.setitem(methods.METHOD_REGISTRY, name, adapter)
    art = tmp_path / 'a'
    art.mkdir()
    (art / 'manifest.json').write_text(json.dumps({
        'graphs_sha256': 'g', 'visit_membership_sha256': 'v', 'schema_version': 's', 'logic_contract_version': 'l',
        'temporal_clean': True}))
    tgt = tmp_path / 't'
    tgt.mkdir()
    (tgt / 'targets.csv').write_text('x\n')
    (tgt / 'binding_manifest.json').write_text(json.dumps({'label_policy_version': 'primary_dx_v4'}))
    canon = tmp_path / 'canon.json'
    canon.write_text(json.dumps({'classes': [f'c{i}' for i in range(CLASSES)]}))
    splits = {'train': _graphs(24, 0, 1), 'validation': _graphs(12, 100, 2), 'dev': _graphs(12, 200, 3)}
    prep = {'vocabulary': {'a': 1}, 'triples': ['t1', 't2']}
    for fn, val in (('validate_artifact_manifest', lambda m: None), ('verify_graph_file', lambda a, m: None),
                    ('verify_visit_membership_file', lambda a, m: None),
                    ('verify_target_binding', lambda m, t, l: 'b' * 64),
                    ('load_targets', lambda p: {}),
                    ('select_top_labels', lambda t, k: (t, list(range(6)), {})),
                    ('build_dataset', lambda *a, **k: (splits, prep)),
                    ('preprocessing_state', lambda p: {'preprocessing_version': train.PREPROCESSING_VERSION}),
                    ('triple_count', lambda p, d='forward': 3),
                    ('edge_feature_layout', lambda p, d='forward': []),
                    ('code_source_hashes', lambda: {})):
        monkeypatch.setattr(train, fn, val)
    argv = ['--artifact', str(art), '--targets', str(tgt / 'targets.csv'), '--canonical', str(canon),
            '--output', str(tmp_path / 'o'), '--method', name, '--epochs', '2', '--batch-size', '8',
            '--dev-limit', '12', '--selection-fold', 'dev', '--final-eval', 'validation', *extra]
    parser = train.parser()
    args = train.normalize_method_args(parser.parse_args(argv), parser)
    train.run(args)
    out = tmp_path / 'o'
    return (json.loads((out / 'binding.json').read_text()), json.loads((out / 'history.json').read_text()), out,
            splits)


def _train_and_weights(monkeypatch, tmp_path, *extra, adapter):
    CREATED.clear()
    binding, history, out, splits = run_stub(monkeypatch, tmp_path, *extra, adapter=adapter, name='tiny')
    final = {k: v.clone() for k, v in CREATED[-1].state_dict().items()}   # the model as training left it
    return binding, history, final


# ----------------------------------------------------------------

@pytest.mark.parametrize('adapter', [Tiny, TinyHooks], ids=['plain-adam', 'adamw-cosine'])
def test_tsc_h5_01_train_eval_subset_does_not_change_training(monkeypatch, tmp_path_factory, adapter):
    _, plain_hist, plain_w = _train_and_weights(monkeypatch, tmp_path_factory.mktemp('plain'), adapter=adapter)
    _, sub_hist, sub_w = _train_and_weights(monkeypatch, tmp_path_factory.mktemp('sub'),
                                            '--train-eval-subset', '8', adapter=adapter)
    drop = ('seconds', 'train_eval_macro_f1', 'train_eval_ce', 'dev_ce')   # timing and the flag's own columns
    strip = lambda rows: [{k: v for k, v in r.items() if k not in drop} for r in rows]
    assert strip(plain_hist) == strip(sub_hist), 'history (loss, metrics, lr) differs with --train-eval-subset'
    assert all('train_eval_ce' in r for r in sub_hist)   # the flag really ran
    for key in plain_w:
        assert torch.equal(plain_w[key], sub_w[key]), f'final weight {key} differs with --train-eval-subset'


def test_control_two_plain_runs_are_identical(monkeypatch, tmp_path_factory):
    _, a_hist, a_w = _train_and_weights(monkeypatch, tmp_path_factory.mktemp('a'), adapter=Tiny)
    _, b_hist, b_w = _train_and_weights(monkeypatch, tmp_path_factory.mktemp('b'), adapter=Tiny)
    strip = lambda rows: [{k: v for k, v in r.items() if k != 'seconds'} for r in rows]
    assert strip(a_hist) == strip(b_hist)
    assert all(torch.equal(a_w[k], b_w[k]) for k in a_w)


def test_tsc_h5_01_train_eval_subset_does_not_advance_the_global_rng(monkeypatch, tmp_path):
    """The eval-only train-subset pass must not draw from the global torch RNG (the training stream)."""
    draws = []
    real_evaluate = train.evaluate

    def spy(model, loader, device, *, epoch=0):
        if len(loader.dataset) != 8:   # selection / validation loaders: not under test
            return real_evaluate(model, loader, device, epoch=epoch)
        before = torch.get_rng_state()
        out = real_evaluate(model, loader, device, epoch=epoch)
        draws.append(not torch.equal(before, torch.get_rng_state()))
        return out

    monkeypatch.setattr(train, 'evaluate', spy)
    run_stub(monkeypatch, tmp_path, '--train-eval-subset', '8', adapter=Tiny, name='tiny')
    assert len(draws) == 2, 'train-subset evaluation should run once per epoch'
    assert not any(draws), f'train-subset evaluation advanced the global RNG: {draws}'


# ----------------------------------------------------------------

BASE_ADAPTER_DEV_SELECTION = (
    'best dev macro_f1 on a patient-disjoint split of unused TRAIN patients; validation evaluated once at the '
    'selected checkpoint; adapter early stopping on dev with configured patience/min_delta after method schedule gate')


def test_tsc_h5_03_selection_text_stays_the_cohort_compatible_base_text(monkeypatch, tmp_path):
    # design decision 2026-10-11: binding['selection'] is a core/aggregate.py cohort key and every existing fair_v3 run
    # carries the base text, so it must not change even when early stopping cannot fire (patience >= epochs).
    b, _, _, _ = run_stub(monkeypatch, tmp_path, '--epochs', '2', '--patience', '40')
    assert b['early_stopping'].startswith('disabled')   # the early_stopping key carries the truth
    assert b['selection'] == BASE_ADAPTER_DEV_SELECTION, b['selection']


def test_tsc_h5_03_selection_text_is_the_base_text_when_early_stopping_can_fire(monkeypatch, tmp_path):
    b, _, _, _ = run_stub(monkeypatch, tmp_path, '--epochs', '2', '--patience', '1')
    assert 'early_stopping' not in b
    assert b['selection'] == BASE_ADAPTER_DEV_SELECTION, b['selection']


# ----------------------------------------------------------------

def test_tsc_h5_04_early_stopping_key_counts_the_start_gate(monkeypatch, tmp_path):
    # start gate 2, epochs 3, patience 2: only epoch index 2 is eligible, the counter stays 0 < patience
    b, h, _, _ = run_stub(monkeypatch, tmp_path, '--epochs', '3', '--patience', '2', adapter=Gated, name='gated')
    assert len(h) == 3
    assert b['early_stopping_start_epoch_index'] == 2
    assert b.get('early_stopping', '').startswith('disabled'), b.get('early_stopping')
    assert b['selection'] == BASE_ADAPTER_DEV_SELECTION, b['selection']   # text kept (cohort key)


def test_tsc_h5_04_early_stopping_key_absent_when_the_gated_run_can_still_stop(monkeypatch, tmp_path):
    # start gate 2, epochs 4, patience 1: eligible epochs 2 and 3, a no-gain on epoch 3 can stop the run
    b, _, _, _ = run_stub(monkeypatch, tmp_path, '--epochs', '4', '--patience', '1', adapter=Gated, name='gated')
    assert 'early_stopping' not in b
    assert b['selection'] == BASE_ADAPTER_DEV_SELECTION, b['selection']


def test_tsc_h5_04_early_stopping_key_at_the_boundary_patience_equals_epochs_minus_start(monkeypatch, tmp_path):
    # start gate 2, epochs 4, patience 2 = epochs - start. Eligible epochs 2 and 3, so
    # the no-gain counter reaches 1 at most and never 2: early stopping cannot fire, the key must say so.
    b, h, _, _ = run_stub(monkeypatch, tmp_path, '--epochs', '4', '--patience', '2', adapter=Gated, name='gated')
    assert len(h) == 4
    assert b['early_stopping_start_epoch_index'] == 2
    assert b.get('early_stopping', '').startswith('disabled'), b.get('early_stopping')


# ----------------------------------------------------------------

def test_tsc_h5_05_binding_weight_decay_is_a_value_that_is_applied(monkeypatch, tmp_path):
    b, _, _, _ = run_stub(monkeypatch, tmp_path, '--weight-decay', '4.3e-05', adapter=TinyHooks, name='tinyh')
    applied = {g['weight_decay'] for g in b['optimizer_param_groups']}
    assert b['weight_decay'] in applied, f"binding weight_decay={b['weight_decay']} but groups apply {sorted(applied)}"


def test_binding_weight_decay_unchanged_for_adam_hook_and_hookless_adapters(monkeypatch, tmp_path_factory):
    """DEFAULT/LEGACY mp2 (build_optimizer returns Adam with args.weight_decay) and hookless adapters."""
    for adapter in (TinyAdamHooks, Tiny):
        b, _, _, _ = run_stub(monkeypatch, tmp_path_factory.mktemp(adapter.__name__), '--weight-decay', '4.3e-05',
                              adapter=adapter, name='tiny')
        assert b['weight_decay'] == 4.3e-05


# ---------------------------------------------------------------- guards for items that need no code change

def test_history_lr_column_follows_the_scheduler(monkeypatch, tmp_path):
    """mp2v3-fixes-06 / mp2v2-train-07: optimizer_param_groups[*].lr is the epoch-0 lr, the per-epoch lr is in history."""
    _, h, _, _ = run_stub(monkeypatch, tmp_path, '--epochs', '4', '--patience', '40', '--lr', '0.01',
                          adapter=TinyHooks, name='tinyh')
    expected = [0.01 * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * e / 4))) for e in range(4)]
    assert [r['lr'] for r in h] == pytest.approx(expected, rel=1e-9)


def test_tsc_h5_07_train_eval_subset_digest_is_the_sorted_id_digest_and_seed_independent(monkeypatch,
                                                                                       tmp_path_factory):
    """Pins the convention (digest of the SORTED ids of the subset); nothing consumes it in loader order."""
    b1, _, _, splits = run_stub(monkeypatch, tmp_path_factory.mktemp('s1'), '--train-eval-subset', '8',
                                '--seed', '1')
    b2, _, _, _ = run_stub(monkeypatch, tmp_path_factory.mktemp('s2'), '--train-eval-subset', '8', '--seed', '2')
    assert b1['train_eval_subset'] == 8
    assert b1['train_eval_subset_sample_ids_sha256'] == b2['train_eval_subset_sample_ids_sha256']
    picked = sorted(random.Random(b1['sample_seed']).sample(range(len(splits['train'])), 8))
    sorted_ids = sorted(splits['train'][i].sample_id for i in picked)
    assert b1['train_eval_subset_sample_ids_sha256'] == sample_ids_sha256(sorted_ids)
