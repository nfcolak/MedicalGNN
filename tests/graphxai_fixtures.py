"""Fixtures shared by the GraphXAI tests.

Copied verbatim from tests/test_clinical_method_adapters.py, a former clinical_gnn test module and
tests/test_real_graphxai.py (as of 0a855d55), which the initial release removed. Only the
helpers the GraphXAI tests use are kept; nothing else of those modules is restored.
"""
from __future__ import annotations

import json
import torch
from argparse import Namespace
from core import train
from core.contracts import VISIT_MEMBERSHIP_CONTRACT_VERSION, VISIT_MEMBERSHIP_FILENAME
from core.schema import sha256
from core.tensorize import ALL_RELATIONS, ClinicalGraphData, PAYLOAD_WIDTH
from torch_geometric.nn import GCNConv, global_mean_pool


# --- from tests/test_clinical_method_adapters.py @ 0a855d55 ---
NODE_DIM = 6
EDGE_DIM = len(ALL_RELATIONS) + PAYLOAD_WIDTH
NUM_TOKENS = 12
NUM_TRIPLES = 5
NUM_CLASSES = 2
def synthetic_graph(*, sample_id, label, node_count, num_visits, membership, global_mask):
    generator = torch.Generator().manual_seed(100 + label + node_count)
    x = torch.randn((node_count, NODE_DIM), generator=generator)
    edge_pairs = [(index, (index + 1) % node_count) for index in range(node_count)]
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_attr = torch.randn((len(edge_pairs), EDGE_DIM), generator=generator)
    edge_relation = torch.arange(len(edge_pairs), dtype=torch.long) % len(ALL_RELATIONS)
    edge_triple = torch.arange(len(edge_pairs), dtype=torch.long) % NUM_TRIPLES
    data = ClinicalGraphData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        visit_membership_index=torch.tensor(membership, dtype=torch.long),
        num_visits=torch.tensor([num_visits], dtype=torch.long),
        global_node_mask=torch.tensor(global_mask, dtype=torch.bool),
    )
    data.token = torch.arange(1, node_count + 1, dtype=torch.long) % NUM_TOKENS
    data.node_type = torch.arange(node_count, dtype=torch.long) % 8
    data.edge_relation = edge_relation
    data.edge_triple = edge_triple
    data.y = torch.tensor([label], dtype=torch.long)
    data.sample_id = sample_id
    return data

def method_args(**overrides):
    values = {
        "epochs": 4,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 2,
        "patience": 2,
        "min_delta": 0.0,
        "warm_epochs": 1,
        "proj_epochs": 1,
        "proj_interval": 5,
        "nearest_graphs": 2,
        "prototypes_per_class": 1,
        "rollout": 1,
        "min_atoms": 2,
        "max_atoms": 3,
        "expand_atoms": 2,
        "c_puct": 1.0,
        "gsat_extractor_dropout": 0.0,
        "graphcare_message_dropout": 0.0,
    }
    values.update(overrides)
    return Namespace(**values)

# --- from a former clinical_gnn test module @ 0a855d55 ---
def clinical_graph(sample_id, value=70.0, *, extra_measurement=True, prior_visits=0):
    """One contract-valid graph: patient, index visit, complaint, vital, lab + analyte."""
    nodes = [
        {'id': 'p', 'kind': 'patient', 'token': 'patient', 'age': 40,
         'gender': 'F', 'race': 'A', 'arrival_transport': 'WALK IN'},
        {'id': 'v', 'kind': 'visit', 'token': 'visit:index', 'acuity': 2},
        {'id': 'c', 'kind': 'complaint', 'token': 'cc:chest pain'},
        {'id': 'hr', 'kind': 'vital', 'token': 'vital:heartrate', 'unit': 'bpm',
         'value': value, 'time_hours': -1.0, 'available_hours': -1.0},
    ]
    edges = [
        {'source': 'p', 'target': 'v', 'relation': 'has_visit', 'informative': False},
        {'source': 'v', 'target': 'p', 'relation': 'index_visit_of', 'informative': False},
        {'source': 'v', 'target': 'c', 'relation': 'reports_complaint', 'informative': False},
        {'source': 'v', 'target': 'hr', 'relation': 'observed_vital', 'informative': False},
    ]
    if extra_measurement:
        nodes += [{'id': 'm', 'kind': 'measurement', 'token': 'lab:creatinine', 'unit': 'mg/dL',
                   'value': value / 50, 'time_hours': -2.0, 'available_hours': -1.5},
                  {'id': 'a', 'kind': 'analyte', 'token': 'lab:creatinine'}]
        edges += [{'source': 'v', 'target': 'm', 'relation': 'measured_in', 'informative': False},
                  {'source': 'm', 'target': 'a', 'relation': 'instance_of', 'informative': False}]
    return {'sample_id': sample_id, 'nodes': nodes, 'edges': edges,
            'coverage': {'complaints': 1, 'index_measurements': int(extra_measurement),
                         'prior_visits': prior_visits, 'informative_edges': 0}}

def membership(graph):
    """Sidecar row: patient global, every other node in the single index visit."""
    kinds = [node['kind'] for node in graph['nodes']]
    return {'contract_version': VISIT_MEMBERSHIP_CONTRACT_VERSION,
            'sample_id': graph['sample_id'], 'visit_ordinals': [0],
            'membership_pairs': [[0, i] for i, kind in enumerate(kinds) if kind != 'patient'],
            'global_node_mask': [kind == 'patient' for kind in kinds]}

def artifact(tmp_path, count=24, test_rows=2):
    """Contract-valid artifact + bound target sidecar; one patient per sample id."""
    root = tmp_path / 'artifact'
    root.mkdir()
    graphs = [clinical_graph(f'sample-{i:02d}', 55.0 + 3 * i) for i in range(count)]
    splits = (['train'] * (count - 4 - test_rows) + ['validation'] * 4 + ['test'] * test_rows)
    graphs_path, membership_path = root / 'graphs.jsonl', root / VISIT_MEMBERSHIP_FILENAME
    graphs_path.write_text(''.join(json.dumps(g) + '\n' for g in graphs))
    membership_path.write_text(''.join(json.dumps(membership(g)) + '\n' for g in graphs))
    labels = [f'synthetic_label_{i}' for i in range(train.NUM_CLASSES)]
    canonical = tmp_path / 'canonical.json'
    canonical.write_text(json.dumps({'classes': labels}))
    manifest = {'status': 'completed', 'schema_version': 'clinical_graph_v2',
                'logic_contract_version': 'clinical_graph_logic_v2', 'temporal_clean': False,
                'graphs_sha256': sha256(graphs_path), 'inherited_cohort_sha256': 'synthetic',
                'visit_membership_contract_version': VISIT_MEMBERSHIP_CONTRACT_VERSION,
                'visit_membership_file': VISIT_MEMBERSHIP_FILENAME,
                'visit_membership_rows': count,
                'visit_membership_sha256': sha256(membership_path),
                'limitations': ['Synthetic fixture; not a research result.']}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    targets = tmp_path / 'targets.csv'
    targets.write_text('sample_id,target,split,subject_id\n' + ''.join(
        f'sample-{i:02d},{i % 2},{split},person-{i:02d}\n' for i, split in enumerate(splits)))
    (tmp_path / 'binding_manifest.json').write_text(json.dumps(
        {'status': 'completed', 'labels': labels, 'cohort_sha256': 'synthetic',
         'label_policy_version': 'primary_dx_v3',
         'artifact_files': {'targets.csv': sha256(targets)}}))
    return root, targets, canonical

def run_args(root, targets, canonical, output, **overrides):
    # Default = the final protocol's shape: selection on validation, the test fold scored once
    # (4 validation rows, `test_rows` test rows). A dev run (selection_fold='dev', dev_limit=3,
    # final_eval='validation') draws a bounded TRAIN sample so unused TRAIN patients remain for
    # the patient-disjoint dev split.
    values = dict(artifact=str(root), targets=str(targets), canonical=str(canonical),
                  output=str(output), method='gsat', conv=None, seed=7,
                  top_k_labels=2, drop_relation=None, rewire_relation=None, edges='all',
                  train_limit=12, token_min_count=1, min_prior_visits=0, device='cpu',
                  batch_size=4, hidden=8, layers=1, dropout=0.0, token_dim=4, heads=2,
                  modulation='multiplicative', weights='none', lr=0.01, epochs=2,
                  no_edge_payload=False, no_message_passing=False,
                  edge_direction='bidirectional', selection_fold='validation', dev_limit=None,
                  final_eval='test', sample_seed=None)
    if overrides.get('final_eval') in ('none', 'validation') and 'selection_fold' not in overrides:
        # a run that never scores test is a dev-selection run (train.py: --final-eval none needs it)
        values.update(selection_fold='dev', dev_limit=3)
    values.update(overrides)
    # Start from the real parser so every flag the current train.py reads has its default
    # (optimizer, protocol keys, ...), then apply this fixture's small-run values.
    parser = train.parser()
    args = parser.parse_args(['--artifact', values['artifact'], '--targets', values['targets'],
                              '--canonical', values['canonical'], '--output', values['output']])
    for name, value in values.items():
        setattr(args, name, value)
    return train.normalize_method_args(args, parser)

# --- from tests/test_real_graphxai.py @ 0a855d55 ---
class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = GCNConv(2, 2)
    def forward(self, x, edge_index, batch=None):
        return global_mean_pool(self.conv(x, edge_index), batch)

