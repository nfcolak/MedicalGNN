"""CEI-GNN with configurable message passing: ``EvidenceNetworkMP2`` (method ``cei_gnn_mp2``).

Network-side half of ``cei_mp2``: encoder variants, how the message-passing state ``h`` reaches the
evidence heads, the readout variants and the training-only extras. The message-passing stack itself
lives in ``cei/layers.py`` (``MPStack``).

Forward (``h0`` = encoder output, ``h`` = message-passing state)::

    h0   = encoder(features)                      # RDO-05a/RDO-07/RDO-06/LCL-02 variants
    h    = MPStack(h0)                            # or mp_branches narrow stacks (ERR-04), one state each
    h    = softmax-mix of the layer states        # mp_jk (REG-06)
    rows = heads(views of h0 / h)                 # mp_feed, mp_dual_read, readout options; with
                                                  # branches: rows, extra blocks, LCL-02/LCL-04
                                                  # terms are computed per branch state, then averaged
    logits = bias + sum of the rows               # ledger_keys() lists the additive row blocks

Under the LEGACY option values every operation, the global-RNG consumption of construction and of
the forward, the state-dict names (``mp_layers.`` -> ``mp_stack.layers.``) and the values reproduce
``cei_mp.cei_gnn_mp.EvidenceNetworkMP``. Replacement modules are created with ``skip_init`` and
``tensor_generator(seed, '<qualified name>')``; every random draw of this file comes from a
generator owned by the network (``mp_mask``, ``row_drop``).
"""
from __future__ import annotations

import contextlib
import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from core import NODE_KINDS
from cei.cei_gnn_v2 import kind_pair_index, within_visit_pairs
from cei.cei_gnn_v3 import EvidenceNetworkV3, tensor_generator
from cei.studies.cei_v3_ple import MIN_ACTIVE_KNOTS, ple_basis
from .layers import MPStack

__all__ = ['EvidenceNetworkMP2']

FEEDS = ('all', 'dual', 'node', 'gates')
NODE_READOUTS = ('global', 'kind_budget')
PLE_MODES = ('hat', 'robust_cum')
PLE_CLIP = 10.0                       # NUMERIC_POLICY zscore_clip_10
PLE_FALLBACK_Z = (-2.0, 2.0)          # fixed z-grid {-2,-1,0,1,2} for K=4: linspace(-2, 2, K+1)
BUDGET_OFFSET = math.log(math.e - 1.0)   # softplus(BUDGET_OFFSET) == 1 (ERR-03 identity)
# options this file reads (full validated dict is passed in)
REQUIRED_OPTIONS = ('mp_layers', 'mp_time_decay', 'mp_feed', 'mp_dual_read', 'mp_jk', 'mp_branches',
                    'mp_branch_width', 'row_drop', 'shared_gate', 'node_readout', 'edge_head_rank',
                    'kind_norm', 'mp_shrink', 'mp_split_rows', 'mp_mask_rate', 'ple_mode',
                    'mp_dropout')   # (mp2v2-net-06): read below, so required like the rest


def _linear(in_features, out_features, *, seed, name, bias, zero=False):
    """``nn.Linear`` initialised from per-tensor generators (global RNG untouched)."""
    layer = nn.utils.skip_init(nn.Linear, in_features=in_features, out_features=out_features,
                               bias=bias)
    with torch.no_grad():
        if zero:
            layer.weight.zero_()
            if bias:
                layer.bias.zero_()
        else:
            bound = 1.0 / float(in_features) ** 0.5
            layer.weight.uniform_(-bound, bound, generator=tensor_generator(seed, f'{name}.weight'))
            if bias:
                layer.bias.uniform_(-bound, bound, generator=tensor_generator(seed, f'{name}.bias'))
    return layer


def robust_knot_table(knots, clip=PLE_CLIP):
    """RDO-06: ``(knots float[items, K+1], active bool[items])`` derived from the fitted table.

    Items with fewer than K+1 finite knots use the fixed z-grid linspace(-2, 2, K+1) clipped to
    the item's knot range (ties after clipping collapse; an item needs >= 3 distinct knots to be
    active). When the clipped grid collapses below that (the item's knots lie outside the grid)
    the item keeps its own fitted knots, so an item active under ``hat`` never becomes inactive.
    A top knot equal to the clip bound is replaced by the midpoint between it and the previous
    knot. Padding is a trailing +inf suffix as in the input.
    """
    knots = torch.as_tensor(knots, dtype=torch.float32)
    items, width = knots.shape
    grid = torch.linspace(PLE_FALLBACK_Z[0], PLE_FALLBACK_Z[1], width, dtype=torch.float32)
    out = torch.full((items, width), float('inf'), dtype=torch.float32)
    active = torch.zeros((items,), dtype=torch.bool)
    for index in range(items):
        row = knots[index]
        finite = row[torch.isfinite(row)]
        if finite.numel() == 0:
            continue
        fitted = None
        if finite.numel() < width:
            fitted = torch.unique(grid.clamp(float(finite[0]), float(finite[-1])))
            if fitted.numel() < MIN_ACTIVE_KNOTS:   # RDO-06: grid collapsed -> own knots
                fitted = None
        if fitted is None:
            fitted = finite.clone()
            if fitted.numel() >= 2 and float(fitted[-1]) >= clip - 1e-6:
                fitted[-1] = (fitted[-1] + fitted[-2]) / 2.0
        if fitted.numel() >= MIN_ACTIVE_KNOTS:
            out[index, :fitted.numel()] = fitted
            active[index] = True
    return out, active


class EvidenceNetworkMP2(EvidenceNetworkV3):
    """``EvidenceNetworkV3`` + configurable message passing (see module docstring)."""

    def __init__(self, *, mp2_options, relation_names, **kwargs):
        options = dict(mp2_options)
        missing = [key for key in REQUIRED_OPTIONS if key not in options]
        if missing:
            raise ValueError(f'mp2_options is missing {missing}; pass the validated option dict')
        self._check_options(options)
        super().__init__(**kwargs)
        self.mp2_options = options
        self.relation_names = tuple(str(name) for name in relation_names)
        o, seed, hidden, classes = options, self.seed, self.hidden, self.num_classes
        self.mp_feed, self.mp_dual_read = str(o['mp_feed']), bool(int(o['mp_dual_read']))
        self.mp_branches = int(o['mp_branches'])
        self.mp_layer_count = int(o['mp_layers'])
        self.shared_gate, self.node_readout = bool(int(o['shared_gate'])), str(o['node_readout'])
        self.row_drop, self.kind_norm_on = float(o['row_drop']), bool(int(o['kind_norm']))
        self.mp_shrink, self.mp_split_rows = float(o['mp_shrink']), bool(int(o['mp_split_rows']))
        self.mp_mask_rate, self.ple_mode = float(o['mp_mask_rate']), str(o['ple_mode'])
        self.mp_time_decay, self.mp_jk = bool(int(o['mp_time_decay'])), bool(int(o['mp_jk']))
        node_types = self.node_type_embedding.num_embeddings
        self.num_node_types = int(node_types)

        # ---- message-passing stack(s)
        stack_args = dict(dropout=(self.dropout_rate if float(options['mp_dropout']) < 0 else float(options['mp_dropout'])), seed=seed, num_relations=self.num_relations,
                          num_triples=self.num_triples, num_node_types=node_types,
                          relation_names=self.relation_names, node_kinds=tuple(NODE_KINDS),
                          options=options)
        if self.mp_branches == 1:
            self.mp_stack = MPStack(hidden, name='mp_layers', **stack_args)
        else:   # ERR-04: K narrow stacks, h_k = h0 + up_k(MP_k(down_k(h0)))
            width = min(int(o['mp_branch_width']), hidden)
            self.branch_width = width
            self.mp_branch_stacks = nn.ModuleList(
                [MPStack(width, name=f'mp_branch.{k}', **stack_args) for k in range(self.mp_branches)])
            self.branch_down = nn.ModuleList(
                [_linear(hidden, width, seed=seed, name=f'branch_down.{k}', bias=True)
                 for k in range(self.mp_branches)])
            # mp2rev-net-04: zero-init (as nbr_head) so every branch starts at h0 = v3 and the
            # ensemble only adds learned, message-driven change (h already holds down_k(h0))
            self.branch_up = nn.ModuleList(
                [_linear(width, hidden, seed=seed, name=f'branch_up.{k}', bias=True, zero=True)
                 for k in range(self.mp_branches)])
        if self.mp_jk:   # REG-06: learned softmax mix of [h0, h1..hL], uniform at init
            self.jk_logits = nn.Parameter(torch.zeros(self.mp_branches, self.mp_layer_count + 1))

        # ---- RDO-02 shared gate: one scalar gate per node row / edge row
        if self.shared_gate:
            self.node_head = _linear(hidden, classes + 1, seed=seed, name='node_head', bias=True)
            self.edge_context_gate = _linear(hidden, 1, seed=seed, name='edge_context_gate', bias=True)
        gate_width = self.edge_context_gate.out_features
        # ---- RDO-04 low-rank edge endpoint heads
        rank = int(o['edge_head_rank'])
        self._legacy_edge_heads = not 0 < rank < hidden   # v3's own heads (global-RNG dropout)
        if 0 < rank < hidden:
            self.edge_source = self._edge_head('edge_source', rank)
            self.edge_target = self._edge_head('edge_target', rank)
        # ---- ERR-03 per-kind contribution budget: softplus(raw) = 1 at init
        if self.node_readout == 'kind_budget':
            # budget = softplus(raw + BUDGET_OFFSET); raw = 0 is the identity (budget 1), so weight
            # decay on raw pulls the budget toward 1, not away from it (REG-06 / ERR-03)
            self.node_kind_budget = nn.Parameter(torch.zeros(self.num_node_types, classes))
        # ---- RDO-07 per-kind affine replaces the shared LayerNorm affine (identity at init):
        # gamma = 1 + kind_norm_gamma, so weight decay pulls the scale toward 1
        if self.kind_norm_on:
            self.node_norm = nn.LayerNorm(hidden, eps=self.node_norm.eps, elementwise_affine=False)
            self.kind_norm_gamma = nn.Parameter(torch.zeros(self.num_node_types, hidden))
            self.kind_norm_beta = nn.Parameter(torch.zeros(self.num_node_types, hidden))
        # ---- ERR-01 second head set on the message state
        if self.mp_dual_read:
            self.node_head_mp = _linear(hidden, self.node_head.out_features, seed=seed,
                                        name='node_head_mp', bias=True)
            self.edge_source_mp = self._edge_head('edge_source_mp', rank)
            self.edge_target_mp = self._edge_head('edge_target_mp', rank)
            self.pair_projection_mp = _linear(hidden, self.pair_rank, seed=seed,
                                              name='pair_projection_mp', bias=False)
            self.pair_vote_mp = _linear(self.pair_rank, classes, seed=seed, name='pair_vote_mp',
                                        bias=True)
        # ---- EVD-01 neighbourhood head (zero-init: starts as v3), EVD-04 edge-gate term (zero-init)
        if self.mp_feed == 'dual':
            # (mp2v2-net-04): shared_gate=1 -> one scalar gate per nbr row (classes + 1 outputs)
            self.nbr_head = _linear(hidden, (classes + 1) if self.shared_gate else 2 * classes, seed=seed,
                                    name='nbr_head', bias=True, zero=True)
        if self.mp_feed == 'gates':
            self.edge_gate_nbr = _linear(hidden, gate_width, seed=seed, name='edge_gate_nbr',
                                         bias=False, zero=True)
        # ---- LCL-02 masked-node reconstruction head and its generator
        if self.mp_mask_rate > 0.0:
            self.mask_proj = _linear(hidden, self.token_dim, seed=seed, name='mask_proj', bias=False)
        self._mask_generator = tensor_generator(seed, 'mp_mask')
        self._site_generators = {}      # network-owned dropout generators of the new head sites
        self._row_drop_generator = tensor_generator(seed, 'row_drop')

        # ---- RDO-06 robust PLE: derived knots + cumulative (thermometer) projection
        if self.ple_mode == 'robust_cum':
            if not self.ple_active:
                raise ValueError(f"ple_mode='robust_cum' acts on the PLE path, which is inactive in "
                                 f"arm {self.arm!r} (arms B and C only)")
            robust, robust_active = robust_knot_table(self.knots)
            self.register_buffer('robust_knots', robust, persistent=False)
            self.register_buffer('robust_knot_active', robust_active, persistent=False)
            # K increment columns instead of v3's K+1 absolute bin embeddings (-hidden params);
            # a hat-mode [hidden, K+1] checkpoint tensor therefore fails the shape check.
            self.ple_projection = _linear(self.K, hidden, seed=seed, name='ple_projection', bias=False)
        # ---- RDO-05a: columns whose value is one replicated per-graph scalar
        layout = self.feature_layout
        self._time_columns = tuple(layout.index(name) for name in ('time_signed_log', 'has_time')
                                   if name in layout)
        self._tsl_column = layout.index('time_signed_log') if 'time_signed_log' in layout else None
        self._has_time_column = layout.index('has_time') if 'has_time' in layout else None
        if self.mp_time_decay and self._tsl_column is None:
            raise ValueError("mp_time_decay=1 needs a 'time_signed_log' column in feature_layout")
        self.register_buffer('time_zero_kinds', torch.tensor(
            [NODE_KINDS.index(kind) for kind in ('visit', 'complaint', 'vital')], dtype=torch.long),
            persistent=False)
        self.register_buffer('mask_skip_kinds', torch.tensor(
            [NODE_KINDS.index(kind) for kind in ('patient', 'visit')], dtype=torch.long),
            persistent=False)

    # ------------------------------------------------------------------ construction helpers

    @staticmethod
    def _check_options(o):
        if str(o['mp_feed']) not in FEEDS:
            raise ValueError(f"mp_feed must be one of {list(FEEDS)}, got {o['mp_feed']!r}")
        if str(o['node_readout']) not in NODE_READOUTS:
            raise ValueError(f"node_readout must be one of {list(NODE_READOUTS)}")
        if str(o['ple_mode']) not in PLE_MODES:
            raise ValueError(f"ple_mode must be one of {list(PLE_MODES)}")
        if int(o['mp_dual_read']) and str(o['mp_feed']) != 'all':
            raise ValueError("mp_dual_read=1 contradicts mp_feed!='all' (dual read already feeds "
                             "h0 and h to separate head sets)")
        if int(o['mp_branches']) < 1 or int(o['mp_layers']) < 1:
            raise ValueError('mp_branches and mp_layers must be >= 1')
        for key in ('row_drop', 'mp_mask_rate'):
            if not 0.0 <= float(o[key]) <= 0.5:
                raise ValueError(f'{key} must be in [0, 0.5]')
        if float(o['mp_shrink']) < 0.0:
            raise ValueError('mp_shrink must be >= 0')
        for key in ('mp_dual_read', 'mp_jk', 'shared_gate', 'kind_norm', 'mp_split_rows',
                    'mp_time_decay'):
            if int(o[key]) not in (0, 1):
                raise ValueError(f'{key} must be 0 or 1, got {o[key]!r}')
        if int(o['mp_branch_width']) < 1:
            raise ValueError('mp_branch_width must be >= 1')
        if int(o['edge_head_rank']) < 0:
            raise ValueError('edge_head_rank must be >= 0')

    def _edge_head(self, name, rank):
        width = rank if 0 < rank < self.hidden else self.hidden
        # slot 2 is the dropout of v3's head; here it is applied by ``_edge_apply`` from a
        # network-owned generator instead of the global RNG
        return nn.Sequential(
            _linear(self.hidden, width, seed=self.seed, name=f'{name}.0', bias=True), nn.GELU(),
            nn.Identity(),
            _linear(width, self.num_classes, seed=self.seed, name=f'{name}.3', bias=True))

    def _dropout(self, x, tag, noise):
        """Dropout of a new head site: mask from the network-owned generator ``<tag>.dropout``."""
        rate = self.dropout_rate
        if not noise or rate <= 0.0:
            return x
        generator = self._site_generators.get(tag)
        if generator is None:
            generator = self._site_generators[tag] = tensor_generator(self.seed, f'{tag}.dropout')
        keep = torch.rand(x.shape, generator=generator) >= rate
        return x * keep.to(device=x.device, dtype=x.dtype) / (1.0 - rate)

    def _edge_apply(self, head, x, tag, noise):
        """``head(x)``; ``tag=None`` keeps v3's own dropout module (legacy site, global RNG)."""
        if tag is None:
            return head(x)
        return head[3](self._dropout(head[1](head[0](x)), tag, noise))

    def ledger_keys(self):
        """Additive row blocks of this configuration: ``logits = bias + sum of their rows``."""
        keys = ['node_contributions', 'edge_contributions', 'pair_contributions']
        if self.mp_dual_read:
            keys += ['node_contributions_message', 'edge_contributions_message',
                     'pair_contributions_message']
        if self.mp_feed == 'dual':
            keys.append('nbr_contributions')
        keys.append('absence_contributions')
        # extra blocks (comorbid_block, ...) add their total to the logits; their rows follow the
        # '<block>_contributions' convention of the extra-block protocol
        keys += [f'{name}_contributions' for name in self.extra_block_names]
        return tuple(keys)

    # ------------------------------------------------------------------ encoder (RDO-05a, 07, 06; LCL-02)

    def _ple_term(self, features, metadata):
        if self.ple_mode != 'robust_cum':
            return super()._ple_term(features, metadata)
        values = features[:, self.scaled_value_column]
        has_value = features[:, self.has_value_column]
        knot_row = self.knot_row_of_token[metadata.token]
        basis = ple_basis(values, has_value, knot_row, self.robust_knots, self.robust_knot_active)
        # RDO-06: ple_projection stores the increments between neighbouring knot bins; the lowest
        # bin embedding is 0 (thermometer code), so the response is basis[:, 1:] @ cumsum(inc).T
        return F.linear(basis[:, 1:], self.ple_projection.weight.cumsum(dim=1))

    def _partner_nodes(self, draw, edge_index, metadata):
        """mp2rev-net-07: the other endpoint (bool[N]) of every ``instance_of`` edge touching a drawn node."""
        relation_ids = [index for index, name in enumerate(self.relation_names)
                        if name.startswith('instance_of')]
        partner = torch.zeros_like(draw)
        if edge_index is None or not relation_ids or edge_index.size(1) == 0:
            return partner
        edge = torch.isin(metadata.edge_relation, torch.tensor(relation_ids, device=draw.device))
        src, dst = edge_index[0][edge], edge_index[1][edge]
        # both directions, any kind of partner (a drawn analyte hides its measurements too)
        partner[dst[draw[src]]] = True
        partner[src[draw[dst]]] = True
        return partner

    def _encode(self, features, metadata, edge_index=None):
        """``(h0, node_time or None, masked node index or None)``."""
        numeric, token_vectors, type_vectors = torch.split(
            features, (self.node_dim, self.token_dim, self.hidden), dim=-1)
        node_type, masked = metadata.node_type, None
        if self.training and self.mp_mask_rate > 0.0:   # LCL-02: mask inputs of evidence nodes
            draw = torch.rand((features.size(0),), generator=self._mask_generator) < self.mp_mask_rate
            draw = draw.to(features.device) & ~torch.isin(node_type, self.mask_skip_kinds)
            keep = (~draw).to(features.dtype)[:, None]
            # mp2rev-net-07: the analyte partner (instance_of edge) names the same lab, so its token
            # is hidden too; otherwise one hop reveals the reconstruction target
            partner_keep = (~(draw | self._partner_nodes(draw, edge_index, metadata))
                            ).to(features.dtype)[:, None]
            numeric, token_vectors = numeric * keep, token_vectors * partner_keep
            masked = draw.nonzero(as_tuple=True)[0]
        node_time = None
        if self.mp_time_decay:
            time = numeric[:, self._tsl_column]
            if self._has_time_column is not None:
                time = torch.where(numeric[:, self._has_time_column] > 0, time, torch.zeros_like(time))
            node_time = time
            # RDO-05 a: the replicated per-graph time of visit/complaint/vital leaves the encoder
            kind = torch.isin(node_type, self.time_zero_kinds).to(numeric.dtype)[:, None]
            column = torch.zeros((1, self.node_dim), dtype=numeric.dtype, device=numeric.device)
            column[0, list(self._time_columns)] = 1.0
            numeric = numeric * (1.0 - kind * column)
        encoder_input = torch.cat((numeric, token_vectors, type_vectors), dim=-1)
        pre_activation = self.node_encoder(encoder_input)
        if self.ple_active:
            pre_activation = pre_activation + self._ple_term(encoder_input, metadata)
        normed = self.node_norm(pre_activation)
        if self.kind_norm_on:   # RDO-07
            normed = normed * (1.0 + self.kind_norm_gamma[node_type]) + self.kind_norm_beta[node_type]
        h = F.gelu(normed)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        if self.encoder_depth > 1:
            for block in self.encoder_blocks:
                h = block(h)
        return h, node_time, masked

    # ------------------------------------------------------------------ message passing

    def _edge_weight(self):
        """Explanation mask on the messages exactly as ``cei_gnn_mp`` (or None)."""
        aggregator = self.edge_aggregator
        mask = aggregator._edge_mask
        if mask is None:
            return None
        if getattr(aggregator, '_apply_sigmoid', False):
            mask = mask.sigmoid()
        return mask

    def _jk_mix(self, states, branch):
        if len(states) != self.jk_logits.size(1):
            raise ValueError(f'mp_jk: stack returned {len(states)} states, expected '
                             f'{self.jk_logits.size(1)} (mp_layers + 1)')
        weights = torch.softmax(self.jk_logits[branch], dim=0)
        mixed = weights[0] * states[0]
        for index in range(1, len(states)):
            mixed = mixed + weights[index] * states[index]
        return mixed

    def _message_passing(self, h0, edge_index, context, metadata, node_time):
        """``(list of per-branch h, list of stack infos)``."""
        edge_weight = self._edge_weight()
        if self.mp_branches == 1:
            h, info = self.mp_stack(h0, edge_index, context, metadata, edge_weight=edge_weight,
                                    node_time=node_time)
            if self.mp_jk:
                h = self._jk_mix(info['states'], 0)
            return [h], [info]
        outputs, infos = [], []
        for k, stack in enumerate(self.mp_branch_stacks):
            # the edge context reuses the branch's down-projection weight (no extra parameters)
            h, info = stack(self.branch_down[k](h0), edge_index,
                            F.linear(context, self.branch_down[k].weight),
                            metadata, edge_weight=edge_weight, node_time=node_time)
            if self.mp_jk:
                h = self._jk_mix(info['states'], k)
            outputs.append(h0 + self.branch_up[k](h))
            infos.append(info)
        return outputs, infos

    # ------------------------------------------------------------------ evidence streams

    def _node_stream(self, rc, vote_h, gate_h, head):
        classes = self.num_classes
        if vote_h is gate_h:
            out = head(vote_h)
            vote, gate = torch.split(out, (classes, out.size(-1) - classes), dim=-1)
        else:   # EVD-04: votes from one state, gates from the other
            vote = head(vote_h)[:, :classes]
            gate = head(gate_h)[:, classes:]
        gate = gate.sigmoid()
        if gate.size(-1) != classes:    # RDO-02: one scalar gate per row
            gate = gate.expand(-1, classes)
        if rc.keep_n is not None:       # RDO-03
            gate = gate * rc.keep_n
        return self._node_rows(rc, vote, gate)

    def _node_rows(self, rc, vote, gate):
        """Gate-normalised node rows (global, or per (graph, kind) with the ERR-03 budget)."""
        classes = self.num_classes
        weighted = gate * vote
        batch, graphs = rc.batch_index, rc.graph_count
        if self.node_readout == 'kind_budget':   # ERR-03: normalise within (graph, kind)
            key = batch * self.num_node_types + rc.node_type
            den = vote.new_ones((graphs * self.num_node_types, classes)).index_add(0, key, gate)
            budget = F.softplus(self.node_kind_budget + BUDGET_OFFSET)[rc.node_type]
            # mp2rev-net-06: the initial budget is 1/(kinds present in the graph), so at init the
            # node total is a mean over the present kinds (same scale as 'global'); only learned
            # budgets change the weighting
            present = (vote.new_zeros((graphs * self.num_node_types,)).index_add(
                0, key, vote.new_ones((key.numel(),))) > 0).to(vote.dtype)
            kinds_present = present.view(graphs, self.num_node_types).sum(-1).clamp_min(1.0)
            parts = budget * weighted / den[key] / kinds_present[batch][:, None]
            total = vote.new_zeros((graphs, classes)).index_add(0, batch, parts)
            return {'parts': parts, 'total': total, 'gate': gate, 'den': den}
        num = vote.new_zeros((graphs, classes)).index_add(0, batch, weighted)
        den = vote.new_ones((graphs, classes)).index_add(0, batch, gate)
        return {'parts': weighted / den[batch], 'total': num / den, 'gate': gate, 'den': den}

    def _edge_stream(self, rc, edge_vote, edge_gate):
        classes, graphs, batch = self.num_classes, rc.graph_count, rc.batch_index
        if rc.keep_e is not None:       # RDO-03: before the aggregator, numerator and denominator
            edge_gate = edge_gate * rc.keep_e
        vote_sum, gate_sum = self.edge_aggregator(rc.edge_index, edge_vote, edge_gate, rc.node_count)
        vote_graph = edge_vote.new_zeros((graphs, classes)).index_add(0, batch, vote_sum)
        gate_graph = edge_vote.new_zeros((graphs, classes)).index_add(0, batch, gate_sum)
        denominator = 1.0 + gate_graph
        edge_mask = self.edge_aggregator._edge_mask
        if edge_mask is None:
            effective = edge_gate * edge_vote
        else:
            edge_mask = edge_mask.to(device=edge_gate.device, dtype=edge_gate.dtype)
            if getattr(self.edge_aggregator, '_apply_sigmoid', False):
                edge_mask = edge_mask.sigmoid()
            effective = edge_mask[:, None] * edge_gate * edge_vote
        parts = effective / denominator[batch[rc.src]] if rc.edge_count else edge_vote
        return {'parts': parts, 'total': vote_graph / denominator}

    def _pair_stream(self, rc, pair_h, projection, vote_head, tag=None):
        classes, graphs = self.num_classes, rc.graph_count
        if rc.pair_count == 0:
            return {'parts': pair_h.new_zeros((0, classes)), 'total': pair_h.new_zeros((graphs, classes)),
                    'gates': pair_h.new_empty((0,)), 'den': pair_h.new_ones((graphs, classes))}
        z = torch.tanh(projection(pair_h))
        left, right = rc.pairs
        q = z[left] + z[right]
        if tag is None:                 # v3's pair-dropout site (global RNG)
            q = F.dropout(q, p=self.dropout_rate, training=rc.noise)
        else:
            q = self._dropout(q, tag, rc.noise)
        vote = vote_head(q)
        gate = self.pair_gate[rc.pair_kind].sigmoid()
        if rc.keep_p is not None:       # RDO-03
            gate = gate * rc.keep_p
        pair_graph = rc.batch_index[left]
        num = pair_h.new_zeros((graphs, classes)).index_add(0, pair_graph, gate * vote)
        den = pair_h.new_ones((graphs, classes)).index_add(0, pair_graph, gate)
        return {'parts': (gate * vote) / den[pair_graph], 'total': num / den, 'gates': gate, 'den': den}

    def _readout(self, rc, h0, h, *, feed, dual_read, branch=0):
        """All evidence rows of one message state ``h`` (``h0`` = node-local state).

        The first branch keeps v3's dropout sites (global RNG, legacy parity); every other
        dropout site of this method draws from a network-owned generator.
        """
        src, dst = rc.src, rc.dst
        if feed == 'all':
            # ERR-01: with a dual read the original heads are the node-local set R0 (on h0) and
            # the *_mp heads read the message state
            vote_h = gate_h = edge_h = h0 if dual_read else h
        elif feed == 'node':            # EVD-06
            vote_h = gate_h = h
            edge_h = h0
        elif feed == 'gates':           # EVD-04
            vote_h, gate_h, edge_h = h0, h, h0
        else:                           # 'dual' (EVD-01): v3 heads on h0, extra nbr row on h
            vote_h = gate_h = edge_h = h0
        pair_h = edge_h
        node = self._node_stream(rc, vote_h, gate_h, self.node_head)
        gate_logit = rc.edge_gate_logit
        if feed == 'gates':
            gate_logit = gate_logit + self.edge_gate_nbr(h[src] + h[dst])
        edge_gate = gate_logit.sigmoid()
        if edge_gate.size(-1) != self.num_classes:
            edge_gate = edge_gate.expand(-1, self.num_classes)
        owned = not (self._legacy_edge_heads and branch == 0)
        edge_vote = (self._edge_apply(self.edge_source, edge_h[src],
                                      f'edge_source.b{branch}' if owned else None, rc.noise)
                     + self._edge_apply(self.edge_target, edge_h[dst],
                                        f'edge_target.b{branch}' if owned else None, rc.noise)
                     + rc.edge_context_vote)
        edge = self._edge_stream(rc, edge_vote, edge_gate)
        pair = self._pair_stream(rc, pair_h, self.pair_projection, self.pair_vote,
                                 tag=None if branch == 0 else f'pair.b{branch}')
        rows = {'node_contributions': node['parts'], 'node_total': node['total'],
                'node_gates': node['gate'], 'node_denominator': node['den'],
                'edge_contributions': edge['parts'],
                'edge_total': edge['total'], 'pair_contributions': pair['parts'],
                'pair_total': pair['total'], 'pair_gates': pair['gates'],
                'pair_denominator': pair['den']}
        if dual_read:                   # ERR-01: second head set on h, separate additive rows
            node_m = self._node_stream(rc, h, h, self.node_head_mp)
            edge_m = self._edge_stream(
                rc, (self._edge_apply(self.edge_source_mp, h[src], f'edge_source_mp.b{branch}', rc.noise)
                     + self._edge_apply(self.edge_target_mp, h[dst], f'edge_target_mp.b{branch}',
                                        rc.noise)), edge_gate)
            pair_m = self._pair_stream(rc, h, self.pair_projection_mp, self.pair_vote_mp,
                                       tag=f'pair_mp.b{branch}')
            rows.update({'node_contributions_message': node_m['parts'], 'node_total_message': node_m['total'],
                         'edge_contributions_message': edge_m['parts'], 'edge_total_message': edge_m['total'],
                         'pair_contributions_message': pair_m['parts'], 'pair_total_message': pair_m['total']})
        if feed == 'dual':              # EVD-01
            rows.update(self._nbr_rows(rc, h))
        return rows

    def _nbr_rows(self, rc, h):
        """EVD-01 neighbourhood row block on the message state ``h``."""
        classes = self.num_classes
        vote, gate = torch.split(self.nbr_head(h), (classes, self.nbr_head.out_features - classes), dim=-1)
        gate = gate.sigmoid()
        if gate.size(-1) != classes:    # shared_gate, expanded like _node_stream
            gate = gate.expand(-1, classes)
        if rc.keep_n is not None:
            gate = gate * rc.keep_n
        nbr = self._node_rows(rc, vote, gate)   # node-row normalisation rule (and ERR-03 budget); own gates and denominator
        return {'nbr_contributions': nbr['parts'], 'nbr_total': nbr['total'],
                'nbr_gates': gate, 'nbr_denominator': nbr['den']}

    @staticmethod
    def _mean_rows(rows_list):
        """ERR-04: every float row block is the mean of the branches' blocks."""
        if len(rows_list) == 1:
            return rows_list[0]
        return {key: (torch.stack([rows[key] for rows in rows_list]).mean(0)
                      if rows_list[0][key].is_floating_point() else rows_list[0][key])
                for key in rows_list[0]}

    def _row_keeps(self, rc, reference):
        """RDO-03 keep masks from the network's own generator (train only)."""
        if not (self.training and self.row_drop > 0.0):
            return
        def draw(count):
            keep = torch.rand((count, 1), generator=self._row_drop_generator) >= self.row_drop
            return keep.to(device=reference.device, dtype=reference.dtype)
        rc.keep_n, rc.keep_e = draw(rc.node_count), draw(rc.edge_count)
        if rc.pair_count:
            rc.keep_p = draw(rc.pair_count)

    def _exported_keys(self):
        """every ``return_parts`` key of this configuration (any mode), for the extra-block collision check."""
        keys = {'logits', 'node_contributions', 'edge_contributions', 'pair_contributions', 'pairs', 'bias',
                'pair_gates', 'pair_denominator', 'node_gates', 'node_denominator', 'absence_contributions',
                'absence_items', 'absence_gates', 'absence_denominator', 'mp_layers', 'mp_edge_weights',
                'mp_alpha'}
        if self.mp_dual_read:
            keys |= {f'{kind}_contributions_{suffix}' for kind in ('node', 'edge', 'pair')
                     for suffix in ('message', 'local')} | {'aux_local_logits'}
        if self.mp_feed == 'dual':
            keys |= {'nbr_contributions', 'nbr_gates', 'nbr_denominator'}
        if self.mp_branches > 1:
            keys.add('mp_branch_alpha')
        if self.mp_jk:
            keys.add('mp_jk_weights')
        if self.mp_shrink > 0.0:
            keys.add('mp_shrink_loss')
        if self.mp_mask_rate > 0.0:
            keys.add('mask_recon_loss')
        if self.mp_split_rows:
            keys |= {f'{prefix}_{kind}_contributions' for prefix in ('local', 'nbr')
                     for kind in ('node', 'edge', 'pair')} | {'message_share', 'message_share_kind'}
        return keys

    @contextlib.contextmanager
    def _quiet(self):
        """Switch every ``nn.Dropout`` off (the local pass of ``mp_split_rows`` is noise-free)."""
        modules = [m for m in self.modules() if isinstance(m, nn.Dropout)]
        states = [m.training for m in modules]
        for module in modules:
            module.training = False
        try:
            yield
        finally:
            for module, state in zip(modules, states):
                module.training = state

    # ------------------------------------------------------------------ forward

    def forward_continuous(self, features, edge_index, metadata, membership, *,
                           return_parts=False, visit_graph=None):
        self._validate(features, edge_index, metadata)
        graph_count, classes = int(metadata.graph_count), self.num_classes
        h0, node_time, masked = self._encode(features, metadata, edge_index)
        context = (self.relation_embedding(metadata.edge_relation)
                   + self.triple_embedding(metadata.edge_triple)
                   + self.edge_feature_projection(metadata.edge_attr))
        h_list, infos = self._message_passing(h0, edge_index, context, metadata, node_time)
        # mp2rev-net-05: with ERR-04 branches every term below is evaluated per branch state and
        # averaged (as the ledger rows are); there is no 'mean of branch states' anywhere
        node_count, batch_index = int(features.size(0)), metadata.batch_index
        pairs = within_visit_pairs(
            membership, metadata.node_type, node_count,
            **({'node_graph': batch_index, 'visit_graph': visit_graph}
               if visit_graph is not None else {}))
        rc = SimpleNamespace(
            edge_index=edge_index, src=edge_index[0], dst=edge_index[1], node_count=node_count,
            edge_count=edge_index.size(1), graph_count=graph_count, batch_index=batch_index,
            node_type=metadata.node_type, pairs=pairs, pair_count=pairs.size(1),
            pair_kind=kind_pair_index(metadata.node_type, pairs) if pairs.size(1) else None,
            edge_context_vote=self.edge_context_vote(context),
            edge_gate_logit=self.edge_context_gate(context),
            keep_n=None, keep_e=None, keep_p=None, noise=self.training)
        self._row_keeps(rc, h0)
        # (mp2v2-net-03): with mp_feed='dual' the v3 rows read h0 only, so they are
        # evaluated once (branch 0 keeps the global-RNG sites) and only the nbr row is per branch
        if self.mp_feed == 'dual' and len(h_list) > 1:
            rows = self._readout(rc, h0, h_list[0], feed='dual', dual_read=False, branch=0)
            nbr_keys = ('nbr_contributions', 'nbr_total', 'nbr_gates', 'nbr_denominator')
            rows.update(self._mean_rows([{key: rows[key] for key in nbr_keys}]
                                        + [self._nbr_rows(rc, h) for h in h_list[1:]]))
        else:
            rows = self._mean_rows([self._readout(rc, h0, h, feed=self.mp_feed, dual_read=self.mp_dual_read,
                                                  branch=k) for k, h in enumerate(h_list)])

        logits = self.bias + rows['node_total'] + rows['edge_total'] + rows['pair_total']
        if self.mp_dual_read:
            logits = (logits + rows['node_total_message'] + rows['edge_total_message']
                      + rows['pair_total_message'])
        if self.mp_feed == 'dual':
            logits = logits + rows['nbr_total']
        absence_total = None
        if self.absence_active:
            absence_total, absence_parts = self._absence_block(features, metadata, membership, visit_graph)
            total = absence_total
        else:
            total = None
            absence_parts = {
                'absence_contributions': features.new_zeros((0, classes)),
                'absence_items': torch.zeros((2, 0), dtype=torch.long, device=features.device),
                'absence_gates': features.new_zeros((0, classes)),
                'absence_denominator': features.new_ones((graph_count, classes))}
        extra_total, extra_parts = None, {}
        if self.extra_block_names:
            extra_total, extra_parts = self._branch_extra_terms(
                h_list, edge_index, metadata, self._exported_keys())
            total = extra_total if total is None else total + extra_total
        if total is not None:
            logits = logits + total
        if not return_parts:
            return logits

        result = {'logits': logits, 'node_contributions': rows['node_contributions'],
                  'edge_contributions': rows['edge_contributions'],
                  'pair_contributions': rows['pair_contributions'], 'pairs': pairs, 'bias': self.bias,
                  'pair_gates': rows['pair_gates'], 'pair_denominator': rows['pair_denominator']}
        # mp2v2-net-05: the node rows' gates and denominator are exported like the nbr / pair / absence ones
        # (under node_readout='kind_budget' the denominator is per (graph, kind), ERR-03)
        result['node_gates'] = rows['node_gates']
        result['node_denominator'] = rows['node_denominator']
        result.update(absence_parts)
        result.update(extra_parts)
        for key in ('node_contributions_message', 'edge_contributions_message',
                    'pair_contributions_message', 'nbr_contributions', 'nbr_gates', 'nbr_denominator'):
            if key in rows:
                result[key] = rows[key]
        if self.mp_dual_read:   # ERR-01 names; the local rows are the unsuffixed ledger rows
            result['node_contributions_local'] = rows['node_contributions']
            result['edge_contributions_local'] = rows['edge_contributions']
            result['pair_contributions_local'] = rows['pair_contributions']
            # ERR-01: CE(bias + R0 terms); R0 reads h0 only (absence / extra blocks are not R0)
            result['aux_local_logits'] = (self.bias + rows['node_total'] + rows['edge_total']
                                          + rows['pair_total'])
        result['mp_layers'] = torch.full((1,), self.mp_layer_count, dtype=torch.long,
                                         device=features.device)
        shares = [info['edge_share'] for info in infos]
        result['mp_edge_weights'] = shares[0] if len(shares) == 1 else torch.stack(shares).mean(0)
        alphas = [info['alpha'] for info in infos if info.get('alpha') is not None]
        if alphas:   # EVD-02; ERR-04 reports the branch mean and the per-branch stack
            result['mp_alpha'] = alphas[0] if len(alphas) == 1 else torch.stack(alphas).mean(0)
            if len(alphas) > 1:
                result['mp_branch_alpha'] = torch.stack(alphas)
        if self.mp_jk:
            result['mp_jk_weights'] = torch.softmax(self.jk_logits, dim=-1).detach()
        if self.training and self.mp_shrink > 0.0:   # LCL-04 (unweighted)
            # mp2rev-net-05: per branch state, averaged; mp2rev-train-06: masked nodes (LCL-02)
            # must move away from their emptied h0, so they are excluded from the ratio
            result['mp_shrink_loss'] = torch.stack(
                [self._shrink_ratio(h, h0, masked) for h in h_list]).mean(0)
        if self.training and self.mp_mask_rate > 0.0:   # LCL-02 (unweighted)
            result['mask_recon_loss'] = torch.stack(
                [self._mask_loss(h, metadata, masked) for h in h_list]).mean(0)
        if self.mp_split_rows and not self.training:   # LCL-04: explanation-time only, no training cost
            result.update(self._split_rows(rc, h0, rows, logits, graph_count, extra_total))
        return result

    # ------------------------------------------------------------------ training extras / reporting

    def _shrink_ratio(self, h, h0, masked):
        """LCL-04 ratio mean over the nodes that were not masked by LCL-02 (all nodes if none)."""
        ratio = (h - h0).pow(2).sum(-1) / (h0.pow(2).sum(-1) + 1e-6)
        if masked is None or masked.numel() == 0:
            return ratio.mean()
        keep = torch.ones(ratio.size(0), dtype=torch.bool, device=ratio.device)
        keep[masked] = False
        return ratio[keep].mean() if bool(keep.any()) else ratio.sum() * 0.0

    def _branch_extra_terms(self, h_list, edge_index, metadata, existing_keys):
        """Extra blocks read every branch state; totals and float parts are averaged like the ledger."""
        if len(h_list) == 1:
            return self._extra_block_terms(h_list[0], edge_index, metadata, existing_keys)
        results = [self._extra_block_terms(h, edge_index, metadata, existing_keys) for h in h_list]
        total = torch.stack([r[0] for r in results]).mean(0)
        parts = {key: (torch.stack([r[1][key] for r in results]).mean(0)
                       if value.is_floating_point() else value)
                 for key, value in results[0][1].items()}
        return total, parts

    def _mask_loss(self, h, metadata, masked):
        target = metadata.token[masked] if masked is not None else None
        if target is None or target.numel() == 0:
            return h.sum() * 0.0
        keep = target > 0
        if not bool(keep.any()):
            return h.sum() * 0.0
        index, target = masked[keep], target[keep] - 1
        logits = self.mask_proj(h[index]) @ self.token_embedding.weight[1:].t()
        return F.cross_entropy(logits, target)

    def _split_rows(self, rc, h0, rows, logits, graph_count, extra_total=None):
        """EVD-05/LCL-04: rows(h0) vs the remainder; ``bias + local + nbr + absence + extra blocks = logits``.

        The extra blocks' rows are exported as ``<block>_contributions`` and are in neither
        piece, so ``message_share`` is taken against ``|logits - bias - extra_total|`` (the evidence the
        split covers).
        """
        with torch.no_grad(), self._quiet():
            local_rc = SimpleNamespace(**{**vars(rc), 'noise': False})
            local = self._readout(local_rc, h0, h0, feed='all', dual_read=False)
        actual = {'node': rows['node_contributions'], 'edge': rows['edge_contributions'],
                  'pair': rows['pair_contributions']}
        if self.mp_dual_read:
            for kind in actual:
                actual[kind] = actual[kind] + rows[f'{kind}_contributions_message']
        if self.mp_feed == 'dual':
            actual['node'] = actual['node'] + rows['nbr_contributions']
        out, difference = {}, logits.new_zeros((graph_count, self.num_classes))
        owner = {'node': rc.batch_index, 'edge': rc.batch_index[rc.src],
                 'pair': rc.batch_index[rc.pairs[0]] if rc.pair_count else rc.batch_index[:0]}
        for kind, rows_actual in actual.items():
            local_rows = local[f'{kind}_contributions']
            neighbourhood = (rows_actual - local_rows).detach()
            out[f'local_{kind}_contributions'] = local_rows
            out[f'nbr_{kind}_contributions'] = neighbourhood
            difference = difference.index_add(0, owner[kind], neighbourhood)
        covered = logits.detach() - self.bias.detach()
        if extra_total is not None:
            covered = covered - extra_total.detach()
        denominator = covered.abs().sum(-1).clamp_min(1e-12)
        out['message_share'] = difference.abs().sum(-1) / denominator
        # EVD-05 per-kind share: the same ratio for the rows owned by nodes of one kind
        # (node rows: the node; edge rows: the source node; pair rows: the first member)
        kinds = {'node': rc.node_type, 'edge': rc.node_type[rc.src],
                 'pair': rc.node_type[rc.pairs[0]] if rc.pair_count else rc.node_type[:0]}
        count = self.num_node_types
        by_kind = logits.new_zeros((graph_count * count, self.num_classes))
        for kind, rows_actual in actual.items():
            neighbourhood = (rows_actual - local[f'{kind}_contributions']).detach()
            by_kind = by_kind.index_add(0, owner[kind] * count + kinds[kind], neighbourhood)
        # fv10-m1-04: per-kind shares are NOT additive. Each is |sum of that kind's rows| (signed
        # rows cancel inside a kind) over the same denominator, so their sum differs from
        # message_share (up to ~0.3 on real graphs); message_share itself is exact.
        out['message_share_kind'] = (by_kind.abs().sum(-1) / denominator.repeat_interleave(count)
                                     ).view(graph_count, count)
        return out
