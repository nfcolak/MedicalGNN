"""CEI-GNN v3 adapter: v2 additive evidence + PLE value basis + "no recorded result" block.

Method id ``cei_gnn_v3``. Options (closed set, extensions spec §9 U3 / E18):
``arm`` ∈ {A, B, C}; ``v3_state=<path>`` (v3 §12 F18) and ``k=<int>`` (must equal the
state's K); ``encoder_depth`` (E2d residual blocks, spec §2.2; the runner's ``--layers``
maps to it) and ``comorbid_block`` ∈ {0, 1} (E6b: lazily imports unit X5's
``cei_v3_ext.comorbid_block.build_block`` through the ``extra_blocks`` protocol).
NOTE: ``encoder_depth >= 2`` / ``layers >= 2`` are supported by the v3 adapter
and map to U3x E2d residual blocks.

v3_state file (JSON, produced by U5 ``fit_v3_state``; see ``build_v3_state`` for the exact
schema): {version, K, knot_table (KnotTable.state()), knot_table_sha256,
universe (Universe.state()), universe_sha256, vocabulary {tokens, min_count},
preprocessing_sha256, node_feature_layout}. ``load_v3_state`` validates every field and the
embedded hashes; the file's own SHA-256 is bound in ``run_config()['v3_state_sha256']``.
The file is derived from medical data and lives under the git-ignored output root.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

from core import NODE_KINDS
from .studies.cei_v3_absence import Universe
from .studies.cei_v3_ple import KnotTable
from core.tensorize import relation_vocabulary
from core.method_base import (ClinicalMethodAdapter, MethodOutput, diagnostic_float, method_option,
                   parameter_count, read_clinical_batch, relation_count,
                   reject_unknown_options)
from .cei_gnn_v3 import ARMS, EvidenceNetworkV3
from .plugin_cei_gnn_v2 import _INDEX_FIELDS, _INTEGER_DTYPES

KNOWN_OPTIONS = frozenset(("arm", "v3_state", "k", "encoder_depth", "comorbid_block"))
# Unit X5 owns this module; the adapter imports it lazily for comorbid_block=1 only.
COMORBID_BLOCK_MODULE = '.studies.cei_v3_ext.comorbid_block'
STATE_VERSION = 'cei_v3_state_v1'
# Optional fields (fv10-m3-03): written by fit_k4.py, absent in older states, validated when present.
STATE_OPTIONAL_FIELDS = ('top_k_labels', 'train_sample_ids_sha256')
STATE_FIELDS = ('version', 'K', 'knot_table', 'knot_table_sha256', 'universe',
                'universe_sha256', 'vocabulary', 'preprocessing_sha256', 'node_feature_layout')
# Arm C of the study is trained at the v2 default width (extensions spec §2.3); an arm at
# another width cannot share C's initial values (E2w is the stated exception).
CONTROL_HIDDEN = 128
PAIR_RANK = 16  # v2 default (`plugin_cei_gnn_v2.py` line 44); v3 does not expose it as an option
_SHA256 = re.compile(r'^[0-9a-f]{64}$')

__all__ = ['KNOWN_OPTIONS', 'STATE_VERSION', 'STATE_FIELDS', 'CONTROL_HIDDEN', 'PAIR_RANK',
           'V3State', 'build_v3_state', 'load_v3_state', 'check_state_against_preprocessing',
           'train_sample_ids_sha256_of_run', 'EvidenceAdapterV3', 'REGISTER']


@dataclass(frozen=True)
class V3State:
    """Validated content of one v3_state file."""

    path: str
    sha256: str
    K: int
    knot_table: KnotTable
    knot_table_sha256: str
    universe: Universe
    universe_sha256: str
    vocabulary_tokens: Tuple[str, ...]
    vocabulary_min_count: int
    preprocessing_sha256: str
    node_feature_layout: Tuple[str, ...]
    top_k_labels: Optional[int] = None              # absent in states fitted before fv10-m3-03
    train_sample_ids_sha256: Optional[str] = None   # sha256 of the sorted TRAIN sample ids, same

    @property
    def num_tokens(self) -> int:
        return len(self.vocabulary_tokens) + 1  # index 0 = UNK (tensorize.Vocabulary)

    def knot_row_of_token(self) -> torch.Tensor:
        """long[num_tokens]: knot-table row of each token index, -1 when it has no row."""
        index = {token: i + 1 for i, token in enumerate(self.vocabulary_tokens)}
        rows = torch.full((self.num_tokens,), -1, dtype=torch.long)
        for item, row in self.knot_table.rows().items():
            rows[index[item]] = row
        return rows


def build_v3_state(*, K, knot_table, universe, vocabulary_tokens, vocabulary_min_count,
                   preprocessing_sha256, node_feature_layout, top_k_labels=None,
                   train_sample_ids_sha256=None) -> dict:
    """The exact JSON document ``load_v3_state`` accepts (U5 writes it with ``json.dumps``).

    ``K`` is the PLE knot count (knots per measurement), not the number of diagnosis labels;
    ``top_k_labels`` / ``train_sample_ids_sha256`` bind the state to its Top-k TRAIN sample.
    """
    if int(K) != knot_table.K:
        raise ValueError(f'K {K} differs from the knot table K {knot_table.K}')
    document = {
        'version': STATE_VERSION,
        'K': int(K),
        'knot_table': knot_table.state(),
        'knot_table_sha256': knot_table.sha256(),
        'universe': universe.state(),
        'universe_sha256': universe.sha256(),
        'vocabulary': {'tokens': list(vocabulary_tokens), 'min_count': int(vocabulary_min_count)},
        'preprocessing_sha256': str(preprocessing_sha256),
        'node_feature_layout': list(node_feature_layout),
    }
    if top_k_labels is not None:
        document['top_k_labels'] = int(top_k_labels)
    if train_sample_ids_sha256 is not None:
        document['train_sample_ids_sha256'] = str(train_sample_ids_sha256)
    return document


def load_v3_state(path) -> V3State:
    """Read and validate a v3_state file; every violation raises ValueError naming the field."""
    path = Path(path)
    raw = path.read_bytes()
    document = json.loads(raw.decode('utf-8'))
    if not isinstance(document, dict):
        raise ValueError('v3_state must be a JSON object')
    missing = [field for field in STATE_FIELDS if field not in document]
    if missing:
        raise ValueError(f'v3_state missing fields: {missing}')
    if document['version'] != STATE_VERSION:
        raise ValueError(f'unsupported v3_state version {document["version"]!r}')
    K = document['K']
    if isinstance(K, bool) or not isinstance(K, int) or K < 1:
        raise ValueError(f'v3_state K must be a positive integer, got {K!r}')
    knot_table = KnotTable.load(document['knot_table'])
    if knot_table.K != K:
        raise ValueError(f'v3_state K {K} differs from the knot table K {knot_table.K}')
    if knot_table.sha256() != document['knot_table_sha256']:
        raise ValueError('v3_state knot_table_sha256 does not match the embedded knot table')
    universe = Universe.load(document['universe'])
    if universe.sha256() != document['universe_sha256']:
        raise ValueError('v3_state universe_sha256 does not match the embedded universe')
    vocabulary = document['vocabulary']
    if (not isinstance(vocabulary, dict) or not isinstance(vocabulary.get('tokens'), list)
            or not all(isinstance(token, str) for token in vocabulary['tokens'])):
        raise ValueError('v3_state vocabulary must be {"tokens": [str, ...], "min_count": int}')
    tokens = tuple(vocabulary['tokens'])
    if len(set(tokens)) != len(tokens):
        raise ValueError('v3_state vocabulary tokens must be unique')
    index = {token: i + 1 for i, token in enumerate(tokens)}
    if len(tokens) + 1 != universe.num_tokens:
        raise ValueError('v3_state vocabulary size differs from the universe num_tokens')
    for item, token_index in zip(universe.items, universe.token_index):
        if index.get(item) != token_index:
            raise ValueError(f'universe item {item!r} does not map to its vocabulary index')
    unknown = sorted(item for item in knot_table.items if item not in index)
    if unknown:
        raise ValueError(f'knot table items outside the vocabulary: {unknown}')
    if knot_table.token_min_count != int(vocabulary.get('min_count', -1)):
        raise ValueError('knot table token_min_count differs from the vocabulary min_count')
    preprocessing_sha256 = document['preprocessing_sha256']
    if not isinstance(preprocessing_sha256, str) or not _SHA256.match(preprocessing_sha256):
        raise ValueError('v3_state preprocessing_sha256 must be a hex SHA-256')
    layout = document['node_feature_layout']
    if not isinstance(layout, list) or not all(isinstance(name, str) for name in layout):
        raise ValueError('v3_state node_feature_layout must be a list of column names')
    for column in ('scaled_value', 'has_value'):
        if layout.count(column) != 1:
            raise ValueError(f'v3_state node_feature_layout must contain exactly one {column!r}')
    top_k_labels = document.get('top_k_labels')
    if top_k_labels is not None and (isinstance(top_k_labels, bool) or not isinstance(top_k_labels, int)
                                     or top_k_labels < 1):
        raise ValueError(f'v3_state top_k_labels must be a positive integer, got {top_k_labels!r}')
    sample_sha256 = document.get('train_sample_ids_sha256')
    if sample_sha256 is not None and (not isinstance(sample_sha256, str) or not _SHA256.match(sample_sha256)):
        raise ValueError('v3_state train_sample_ids_sha256 must be a hex SHA-256')
    return V3State(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), K=K,
                   knot_table=knot_table, knot_table_sha256=document['knot_table_sha256'],
                   universe=universe, universe_sha256=document['universe_sha256'],
                   vocabulary_tokens=tokens, vocabulary_min_count=int(vocabulary['min_count']),
                   preprocessing_sha256=preprocessing_sha256, node_feature_layout=tuple(layout),
                   top_k_labels=top_k_labels, train_sample_ids_sha256=sample_sha256)


def check_state_against_preprocessing(state: V3State, preprocessing: dict) -> None:
    """Refuse a state that was not fitted on this run's preprocessing (fv10-m3-03).

    ``preprocessing`` is the run's ``core.tensorize.preprocessing_state(prep)`` dict (what is
    written to ``preprocessing.json``): the vocabulary TOKEN LIST must equal the state's, the
    canonical file hash must equal ``state.preprocessing_sha256`` and the node layout must match.
    """
    tokens = tuple(preprocessing['vocabulary']['tokens'])
    if tokens != state.vocabulary_tokens:
        both = len(set(tokens).intersection(state.vocabulary_tokens))
        raise ValueError(
            f'v3_state {state.path} was fitted on another vocabulary: the run has {len(tokens)} tokens, '
            f'the state {len(state.vocabulary_tokens)} ({both} in common); refit it with '
            'comparison.fair_v3.fit_k4 --top-k-labels <the run\'s K>')
    digest = hashlib.sha256((json.dumps(preprocessing, indent=2, sort_keys=True) + '\n')
                            .encode('utf-8')).hexdigest()
    if digest != state.preprocessing_sha256:
        raise ValueError(f'v3_state {state.path} preprocessing_sha256 {state.preprocessing_sha256[:12]} '
                         f'differs from the run\'s preprocessing {digest[:12]}')
    if tuple(preprocessing['node_feature_layout']) != state.node_feature_layout:
        raise ValueError(f'v3_state {state.path} node_feature_layout differs from the run\'s')


def _arg(args, name, default=None):
    value = args.get(name, default) if isinstance(args, dict) else getattr(args, name, default)
    return default if value is None else value


def train_sample_ids_sha256_of_run(args):
    """sha256 of the sorted TRAIN sample ids the run will draw, or None if ``args`` lacks the inputs.

    Uses the same functions as ``core.train`` (load_targets -> select_top_labels -> resolve_train_dev_ids;
    train_limit None + dev_limit = the fair_v10 full-TRAIN rule);
    reads the targets sidecar only, never a graph.
    """
    targets_path = _arg(args, 'targets')
    if targets_path is None or _arg(args, 'min_prior_visits', 0):
        return None
    from core import train as train_module
    from .studies.cei_v3_study import train_sample_ids_sha256
    targets = train_module.load_targets(targets_path)
    top_k = _arg(args, 'top_k_labels')
    if top_k is not None:
        targets, _kept, _dropped = train_module.select_top_labels(targets, int(top_k))
    seed = _arg(args, 'sample_seed', _arg(args, 'seed'))
    ids, _dev = train_module.resolve_train_dev_ids(targets, _arg(args, 'train_limit'),
                                                   _arg(args, 'dev_limit'), seed)
    return train_sample_ids_sha256(ids)


def _seed_of(args) -> int:
    seed = args.get('seed') if isinstance(args, dict) else getattr(args, 'seed', None)
    if seed is None or isinstance(seed, bool) or int(seed) != seed:
        raise ValueError('cei_gnn_v3 requires the runner seed (args.seed) for per-tensor initialisation')
    return int(seed)


class EvidenceAdapterV3(ClinicalMethodAdapter):
    """CEI-GNN v3: v2 additive pairs + fixed PLE value basis + not-measured evidence."""

    adaptation_version = "clinical_graph_v2_cei_gnn_v3_value_encoding"
    runner_defaults = dict(hidden=128, layers=1, dropout=0.3, lr=1.79e-3,
                           weight_decay=4.3e-5, batch_size=128, epochs=40,
                           patience=10, min_delta=0.005)
    grad_clip_value = 2.0
    known_options = KNOWN_OPTIONS
    network_class = EvidenceNetworkV3

    @classmethod
    def validate_method_options(cls, args) -> None:
        """Plan-mode check of the method options (no data, no network): names, arm, v3_state file, k, depth."""
        reject_unknown_options(args, cls.known_options)
        arm = method_option(args, "arm", None, str)
        if arm not in ARMS:
            raise ValueError(f"method option arm must be one of {list(ARMS)} (got {arm!r})")
        state_path = method_option(args, "v3_state", None, str)
        if not state_path:
            raise ValueError("method option v3_state=<path> is required (v3 §12 F18)")
        state = load_v3_state(state_path)
        k = method_option(args, "k", None, int)
        if k is None:
            raise ValueError("method option k=<int> is required and must equal the v3_state K")
        if k != state.K:
            raise ValueError(f"method option k={k} differs from the v3_state K={state.K}")
        run_top_k = _arg(args, 'top_k_labels')
        if state.top_k_labels is not None and run_top_k is not None and state.top_k_labels != int(run_top_k):
            raise ValueError(f"v3_state {state.path} was fitted for top_k_labels={state.top_k_labels}, "
                             f"this run uses top_k_labels={int(run_top_k)}")
        method_option(args, "encoder_depth", 1, int, minimum=1)
        method_option(args, "comorbid_block", 0, int, minimum=0, maximum=1)

    def __init__(self, *, num_tokens, node_dim, edge_dim, num_classes, hidden,
                 layers, dropout, token_dim, num_triples, args):
        super().__init__()
        reject_unknown_options(args, self.known_options)
        dimensions = (num_tokens, node_dim, edge_dim, num_classes, hidden, token_dim, num_triples)
        if any(isinstance(value, bool) or int(value) != value or int(value) < 1
               for value in dimensions):
            raise ValueError("model dimensions must be positive integers")
        if isinstance(layers, bool) or int(layers) != layers or int(layers) < 1:
            raise ValueError("layers must be a positive integer")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        self.num_tokens, self.node_dim, self.edge_dim = map(int, (num_tokens, node_dim, edge_dim))
        self.num_classes, self.hidden = map(int, (num_classes, hidden))
        self.layers_count, self.dropout_rate = int(layers), float(dropout)
        self.token_dim, self.num_triples = int(token_dim), int(num_triples)
        self.num_relations = relation_count(args)
        direction = args.get('edge_direction') if isinstance(args, dict) else getattr(args, 'edge_direction', None)
        self.edge_direction = str(direction or 'forward')
        self.pair_rank = PAIR_RANK
        self.seed = _seed_of(args)

        self.arm = method_option(args, "arm", None, str)
        if self.arm not in ARMS:
            raise ValueError(f"method option arm must be one of {list(ARMS)} (got {self.arm!r})")
        state_path = method_option(args, "v3_state", None, str)
        if not state_path:
            raise ValueError("method option v3_state=<path> is required (v3 §12 F18)")
        self.state = load_v3_state(state_path)
        self.k = method_option(args, "k", None, int)
        if self.k is None:
            raise ValueError("method option k=<int> is required and must equal the v3_state K")
        if self.k != self.state.K:
            raise ValueError(f"method option k={self.k} differs from the v3_state K={self.state.K} "
                             "(k = PLE knots per measurement, not the number of diagnosis labels)")
        self._check_state_belongs_to_run(args)
        self.encoder_depth = method_option(args, "encoder_depth", self.layers_count, int, minimum=1)
        self.comorbid_block = method_option(args, "comorbid_block", 0, int, minimum=0, maximum=1)

        if self.state.num_tokens != self.num_tokens:
            raise ValueError(f"v3_state vocabulary gives num_tokens={self.state.num_tokens}, "
                             f"the runner passed num_tokens={self.num_tokens}")
        if len(self.state.node_feature_layout) != self.node_dim:
            raise ValueError("v3_state node_feature_layout length differs from node_dim")
        self.control_shapes = self._control_shapes()
        extra_blocks = ()
        if self.comorbid_block == 1:
            extra_blocks = (self._build_comorbid_block(),)
        knots, active, _ = self.state.knot_table.tensor()
        self.network = self.network_class(
            num_tokens=self.num_tokens, node_dim=self.node_dim, edge_dim=self.edge_dim,
            num_classes=self.num_classes, hidden=self.hidden, token_dim=self.token_dim,
            num_triples=self.num_triples, num_relations=self.num_relations,
            dropout=self.dropout_rate, pair_rank=self.pair_rank, num_node_types=len(NODE_KINDS),
            arm=self.arm, knots=knots, knot_active=active,
            slot_of_token=self.state.universe.slot_of_token(self.num_tokens),
            universe_size=len(self.state.universe), feature_layout=self.state.node_feature_layout,
            seed=self.seed, encoder_depth=self.encoder_depth, extra_blocks=extra_blocks,
            knot_row_of_token=self.state.knot_row_of_token(), control_shapes=self.control_shapes,
            **self._network_kwargs(args))

    def _check_state_belongs_to_run(self, args):
        """fv10-m3-03: refuse a state fitted for another Top-k / TRAIN sample (old states skip the new fields)."""
        state = self.state
        run_top_k = _arg(args, 'top_k_labels')
        if state.top_k_labels is not None and run_top_k is not None and state.top_k_labels != int(run_top_k):
            raise ValueError(f"v3_state {state.path} was fitted for top_k_labels={state.top_k_labels}, "
                             f"this run uses top_k_labels={int(run_top_k)}; fit one state per K "
                             "(comparison.fair_v3.fit_k4 --top-k-labels)")
        if state.top_k_labels is not None and run_top_k is None and self.num_classes != state.top_k_labels:
            raise ValueError(f"v3_state {state.path} was fitted for top_k_labels={state.top_k_labels}, "
                             f"this run has num_classes={self.num_classes}")
        if state.train_sample_ids_sha256 is not None:
            run_sha256 = train_sample_ids_sha256_of_run(args)
            if run_sha256 is not None and run_sha256 != state.train_sample_ids_sha256:
                raise ValueError(f"v3_state {state.path} was fitted on another TRAIN sample than this run's "
                                 f"(sample ids sha256 {state.train_sample_ids_sha256[:12]} vs {run_sha256[:12]})")

    def verify_run_preprocessing(self, preprocessing: dict) -> None:
        """Token list + preprocessing hash check; call with ``preprocessing_state(prep)`` once it exists."""
        check_state_against_preprocessing(self.state, preprocessing)

    # ---------------------------------------------------------- extension hooks

    def _network_kwargs(self, args):
        """Extra constructor arguments of a subclass's ``network_class`` (none for v3)."""
        del args
        return {}

    def _control_shapes(self):
        """Arm C's (forward-view) shapes of the view-dependent tensors, or None.

        E6a widens exactly the relation table (+R reverse ids), the triple table
        (T + t for the reverse of fitted triple t) and the payload projection input
        (+R one-hot columns) — ``tensorize.py`` lines 76–93, 359. When the runner's shapes
        match that widening the control shapes are recoverable; any other shape under the
        bidirectional view is reported as non-comparable (``common_init_identical_to_c``
        False) instead of guessed.
        """
        if self.edge_direction != 'bidirectional':
            return None
        forward = len(relation_vocabulary('forward'))
        reverse = self.num_relations - forward
        fitted_triples, remainder = divmod(self.num_triples - 1, 2)
        if reverse <= 0 or remainder != 0 or fitted_triples < 0:
            return None
        if self.edge_dim <= reverse:
            return None
        return {'num_relations': forward, 'num_triples': fitted_triples + 1,
                'edge_dim': self.edge_dim - reverse}

    def _relation_layout(self):
        """Relation name -> id of this arm's edge view (forward ids never renumbered)."""
        names = relation_vocabulary(self.edge_direction)
        if len(names) != self.num_relations:
            raise ValueError("relation vocabulary size differs from num_relations")
        return {name: index for index, name in enumerate(names)}

    def _build_comorbid_block(self):
        """Lazily import X5's block builder; a clear error names X5 when it is absent."""
        try:
            module = importlib.import_module(COMORBID_BLOCK_MODULE, __package__)
        except ImportError as error:
            raise ValueError(
                f"comorbid_block=1 (U3x extra_blocks hook) needs unit X5: module "
                f"{COMORBID_BLOCK_MODULE} providing build_block(hidden, num_classes, *, "
                f"relation_layout, seed) is not available ({error})") from error
        build_block = getattr(module, 'build_block', None)
        if build_block is None:
            raise ValueError(f"comorbid_block=1 (U3x extra_blocks hook) needs unit X5: "
                             f"{COMORBID_BLOCK_MODULE} has no build_block(hidden, num_classes, *, "
                             "relation_layout, seed)")
        return build_block(self.hidden, self.num_classes, relation_layout=self._relation_layout(),
                           seed=self.seed)

    # --------------------------------------------------------------- batch reading

    def _read(self, batch):
        for field in _INDEX_FIELDS:
            value = getattr(batch, field, None)
            if value is not None and (not torch.is_tensor(value) or value.dtype not in _INTEGER_DTYPES):
                raise ValueError(f"{field} must use an integer dtype")
        membership = getattr(batch, "visit_membership_index", None)
        if membership is None:
            raise ValueError("cei_gnn_v3 requires visit_membership_index")
        visit_counts = getattr(batch, "num_visits", None)
        if not torch.is_tensor(visit_counts) or visit_counts.dtype not in _INTEGER_DTYPES:
            raise ValueError("num_visits must be an integer tensor")
        clinical = read_clinical_batch(
            batch, method="cei_gnn_v3", node_dim=self.node_dim, edge_dim=self.edge_dim,
            num_tokens=self.num_tokens, num_triples=self.num_triples,
            num_relations=self.num_relations)
        visit_counts = visit_counts.to(device=clinical.x.device).view(-1)
        if visit_counts.numel() != clinical.graph_count or (visit_counts < 1).any():
            raise ValueError("num_visits must contain one positive count per graph")
        visit_graph = torch.repeat_interleave(
            torch.arange(clinical.graph_count, device=clinical.x.device), visit_counts.long())
        return clinical, membership.long(), visit_graph

    def continuous_inputs(self, batch) -> torch.Tensor:
        clinical, _, _ = self._read(batch)
        return self.network.continuous_inputs(clinical)

    def forward_continuous(self, features, edge_index, metadata, *, return_parts=False):
        clinical, membership, visit_graph = self._read(metadata)
        if edge_index.shape != clinical.edge_index.shape or not torch.equal(
                edge_index.to(clinical.edge_index.device), clinical.edge_index):
            raise ValueError("edge_index differs from the fixed metadata edge list")
        return self.network.forward_continuous(features, clinical.edge_index, clinical,
                                               membership, return_parts=return_parts,
                                               visit_graph=visit_graph)

    def forward(self, batch, *, epoch: int) -> MethodOutput:
        del epoch
        clinical, membership, visit_graph = self._read(batch)
        features = self.network.continuous_inputs(clinical)
        parts = self.network.forward_continuous(features, clinical.edge_index, clinical,
                                                membership, return_parts=True,
                                                visit_graph=visit_graph)
        logits = parts["logits"]
        auxiliary_loss = logits.sum() * 0.0
        graphs = max(int(clinical.graph_count), 1)
        return MethodOutput(logits=logits, auxiliary_loss=auxiliary_loss, diagnostics={
            "auxiliary_loss": diagnostic_float(auxiliary_loss, "cei_gnn_v3"),
            "pairs_per_graph": float(parts["pairs"].size(1) / graphs),
            "absent_items_per_graph": float(parts["absence_items"].size(1) / graphs)})

    def on_epoch_start(self, epoch: int, train_loader) -> None:
        del train_loader
        if isinstance(epoch, bool) or int(epoch) != epoch or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")

    def optimizer_groups(self, args):
        del args
        return [{"params": list(self.parameters())}]

    # ------------------------------------------------------------------ binding

    def inactive_parameter_count(self) -> int:
        return self.network.inactive_parameter_count()

    def common_init_identical_to_c(self) -> bool:
        """§2.2: True iff every tensor C also has starts from C's initial values.

        Width must equal the control's (E2w is the stated exception); under the
        bidirectional view the widened tensors must have been registered at C's shapes
        (``control_shapes`` recovered from the runner's dimensions), otherwise the later
        common tensors were drawn at a different point of the global stream.
        """
        if self.hidden != CONTROL_HIDDEN:
            return False
        if self.edge_direction != 'forward' and self.control_shapes is None:
            return False
        return True

    def run_config(self) -> dict:
        total, inactive = parameter_count(self), self.inactive_parameter_count()
        inventory = {name: {"shape": list(shape), "active": bool(active)}
                     for name, (shape, active) in self.network.parameter_inventory().items()}
        effective = {"arm": self.arm, "k": self.k, "v3_state": self.state.path,
                     "encoder_depth": self.encoder_depth, "comorbid_block": self.comorbid_block}
        return {
            "method": "cei_gnn_v3",
            "adaptation_version": self.adaptation_version,
            "native_defaults": dict(self.runner_defaults),
            "effective_settings": effective,
            "arm": self.arm,
            "k": self.k,
            "v3_state_path": self.state.path,
            "v3_state_sha256": self.state.sha256,
            "knot_table_sha256": self.state.knot_table_sha256,
            "universe_sha256": self.state.universe_sha256,
            "universe_size": len(self.state.universe),
            "preprocessing_sha256": self.state.preprocessing_sha256,
            **({"v3_state_top_k_labels": self.state.top_k_labels}
               if self.state.top_k_labels is not None else {}),
            **({"v3_state_train_sample_ids_sha256": self.state.train_sample_ids_sha256}
               if self.state.train_sample_ids_sha256 is not None else {}),
            "encoder_depth": self.encoder_depth,
            "comorbid_block": self.comorbid_block,
            "edge_direction": self.edge_direction,
            "hidden": self.hidden,
            "seed": self.seed,
            "pair_mode": self.network.pair_mode,
            "pair_rank": self.pair_rank,
            "parameter_count": total,
            "active_parameter_count": total - inactive,
            "inactive_parameter_count": inactive,
            "ple_active": self.network.ple_active,
            "absence_active": self.network.absence_active,
            # Extensions spec §2.2: an arm at the control width whose common tensors were
            # registered at C's shapes shares C's initial values (per-tensor generators);
            # E2w does not, nor does a bidirectional arm whose shapes cannot be mapped to C's.
            "common_init_identical_to_c": self.common_init_identical_to_c(),
            "control_shapes": self.control_shapes,
            "widened_tensors": list(self.network.widened_tensors),
            "extra_blocks": list(self.network.extra_block_names),
            "parameter_inventory": inventory,
            "architecture": {
                "node_dim": self.node_dim, "edge_dim": self.edge_dim,
                "num_tokens": self.num_tokens, "num_triples": self.num_triples,
                "num_relations": self.num_relations, "num_node_types": len(NODE_KINDS),
                "num_classes": self.num_classes, "hidden": self.hidden,
                "layers": self.layers_count, "encoder_depth": self.encoder_depth,
                "dropout": self.dropout_rate, "token_dim": self.token_dim,
                "pair_rank": self.pair_rank, "k": self.k,
                "universe_size": len(self.state.universe),
                "parameter_count": total, "active_parameter_count": total - inactive,
                "inactive_parameter_count": inactive,
            },
        }


REGISTER = {"cei_gnn_v3": EvidenceAdapterV3}
