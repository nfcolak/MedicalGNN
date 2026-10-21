"""ProtGNN built-in explanation metadata: projection counts after a reload."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch

from protgnn.adapter import ProtGNNAdapter
from tests.graphxai_fixtures import (
    EDGE_DIM, NODE_DIM, NUM_CLASSES, NUM_TOKENS, NUM_TRIPLES, method_args,
    synthetic_graph,
)

ROOT = Path(__file__).resolve().parents[1]


def _kwargs():
    return dict(
        num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM,
        num_classes=NUM_CLASSES, hidden=8, layers=1, dropout=0.0, token_dim=4,
        num_triples=NUM_TRIPLES, args=method_args(min_atoms=2, max_atoms=4))


def _graphs():
    return [
        synthetic_graph(sample_id=f"g{index}", label=index % NUM_CLASSES,
                        node_count=5 + index % 2, num_visits=1,
                        membership=[0] * (5 + index % 2),
                        global_mask=[False] * (5 + index % 2))
        for index in range(4)
    ]


def _fresh_model():
    torch.manual_seed(0)
    return ProtGNNAdapter(**_kwargs())


def _projected_model():
    model = _fresh_model()
    loader = [Batch.from_data_list(_graphs())]
    model._project_from_training_loader(loader, epoch=3)
    return model


def _detail(model):
    model.eval()
    return model.explain_detail(Batch.from_data_list(_graphs()[:1]))


def test_projected_model_reports_real_counts():
    model = _projected_model()
    detail = _detail(model)
    summary = model.projection_summary
    assert summary["source"] == "computed_in_process"
    assert detail["projection_summary_source"] == "computed_in_process"
    assert isinstance(detail["prototypes_projected_onto_training_subgraphs"], int)
    assert isinstance(detail["prototypes_unmatched"], int)
    assert (detail["prototypes_projected_onto_training_subgraphs"]
            + detail["prototypes_unmatched"]) == model.num_prototypes
    assert detail["prototypes_projected_onto_training_subgraphs"] > 0


def test_reloaded_model_reports_null_counts_and_source():
    state = _projected_model().state_dict()
    rebuilt = _fresh_model()
    rebuilt.load_state_dict(state, strict=True)
    detail = _detail(rebuilt)
    assert detail["prototypes_projected_onto_training_subgraphs"] is None
    assert detail["prototypes_unmatched"] is None
    assert detail["projection_summary_source"] == "not_recorded_in_checkpoint"


def test_untouched_model_does_not_claim_zero_projected():
    detail = _detail(_fresh_model())
    assert detail["prototypes_projected_onto_training_subgraphs"] is None
    assert detail["prototypes_unmatched"] is None


def _main_class():
    try:
        source = subprocess.run(
            ["git", "show", "main:protgnn/adapter.py"], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("main-branch protgnn/adapter.py is not available")
    spec = importlib.util.spec_from_loader("protgnn_adapter_main", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        exec(compile(source, "protgnn_adapter_main", "exec"), module.__dict__)
    finally:
        sys.modules.pop(spec.name, None)
    return module.ProtGNNAdapter


def test_state_dict_keys_match_main_branch_class():
    torch.manual_seed(0)
    reference = _main_class()(**_kwargs())
    current = _fresh_model()
    assert list(current.state_dict()) == list(reference.state_dict())
    assert list(_projected_model().state_dict()) == list(reference.state_dict())
    reference.load_state_dict(current.state_dict(), strict=True)
