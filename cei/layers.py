"""Message-passing stack of ``cei_gnn_mp2``: the ``cei_gnn_mp`` layer generalised by option.

One ``MPStack`` replaces ``EvidenceNetworkMP.mp_layers``. Under the LEGACY option values it is the
``cei_gnn_mp`` stack bit for bit (state-dict names ``layers.<i>.<w_fwd|c_fwd|w_rev|c_rev|transform|
glob|norm>.*``, generator names ``<name>.<i>.<module>.weight|bias`` and ``<name>.<i>.dropout``,
identical operation order). Every option below is a proposal id of the 2026-10-10 MP study; the
mean-type kinds (gine, rgcn, diag) share one aggregation helper, the attention kinds (simplehgn,
hgt) share one segment softmax.

Determinism contract: tensors with a random init come from ``tensor_generator(seed, '<qualified
name>')`` (constants and zeros need no generator), every training-time draw (dropout, DropEdge,
KG-edge drop, DropMessage, DropNode, DropPath, attention dropout) comes from a generator owned by
the layer (``<name>.<i>.<purpose>``), the global torch RNG is never touched, eval is deterministic
and nothing is stored in persistent buffers.

Per-graph statistics only: all graph-level reductions index by ``batch_index`` (global mean,
anchor visit, GraphNorm, DropPath), so no graph influences another.
"""
from __future__ import annotations

import math
import warnings
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from cei.cei_gnn_v3 import tensor_generator
from core.tensorize import PAYLOAD_WIDTH

__all__ = ['MPStack']

# Stack-owned options with their DEFAULT column values (the network passes the full validated dict).
# Note (fv10-m1-02): mp_hub_kinds defaults to visit,complaint,patient. On the clinical graphs
# the patient (max degree 2) and complaint (max degree ~10) nodes are no hubs, so the mp_hub_max
# max term effectively reaches only the single index visit per graph; the high-degree diagnosis
# and knowledge nodes get no max channel unless they are listed in mp_hub_kinds.
_STACK_DEFAULTS = {
    'mp_layers': 2, 'mp_kind': 'gine', 'mp_msg_dim': 64, 'mp_rank': 0, 'mp_tie_directions': 0,
    'mp_root': 'linear', 'mp_share_layers': 0, 'mp_residual': 'channel', 'mp_residual_init': 0.1,
    'mp_residual_bound': 0, 'mp_aggr': 'relation', 'mp_degree_norm': 'mean', 'mp_edge_gate': 0,
    'mp_rev_attention': 0, 'mp_att_dropout': 0.0, 'mp_rev_agg': 'mean', 'mp_hub_max': 0,
    'mp_hub_kinds': 'visit,complaint,patient', 'mp_relation_film': 0, 'mp_rel_gate': 0,
    'mp_global': 0, 'mp_dedup': 0, 'mp_ctx': 'matrix', 'mp_bases': 2, 'mp_heads': 4,
    'mp_dropedge': 0.2, 'mp_dropedge_exempt': 'has_visit,index_visit_of', 'mp_kg_edge_drop': 0.0,
    'mp_dropmsg': 0.0, 'mp_dropnode': 0.0, 'mp_droppath': 0.0, 'mp_norm': 'layer',
    'mp_update_kinds': 'all', 'mp_time_decay': 0, 'mp_shortcut': 0}
_CHOICES = {
    'mp_kind': ('gine', 'rgcn', 'diag', 'simplehgn', 'hgt'), 'mp_root': ('linear', 'bias'),
    'mp_residual': ('none', 'scalar', 'channel', 'kind', 'kind_vec'),
    'mp_aggr': ('mean', 'relation', 'triple'), 'mp_degree_norm': ('mean', 'sym', 'null'),
    'mp_rev_agg': ('mean', 'meanmax'), 'mp_ctx': ('matrix', 'table'),
    'mp_norm': ('layer', 'graph', 'graph_kind')}
_KG_PREFIX = 'medical:'
_REV_PREFIX = 'rev:'                      # typed reverse copies of the bidirectional view
_SHORTCUT_INNER, _SHORTCUT_OUTER = 'instance_of', ('medical:member_of', 'medical:measures',
                                                   'medical:assesses')
_NULL_INIT = math.log(math.expm1(0.05))  # softplus(_NULL_INIT) == 0.05: a small learnable null mass (RDO-08)
_DK = 8                                  # HET-06 per-head key width
_CTX_RANK = 16                           # HET-02 / HET-04 low-rank context path
_GATE_BIAS = 2.0                         # MSG-03: sigmoid(2) ~ 0.88


# ------------------------------------------------------------------------------ small builders
def _linear(in_features, out_features, *, seed, name, bias):
    """Exactly ``cei_mp.cei_gnn_mp._linear``."""
    layer = nn.utils.skip_init(nn.Linear, in_features=in_features, out_features=out_features,
                               bias=bias)
    bound = 1.0 / float(in_features) ** 0.5
    with torch.no_grad():
        layer.weight.uniform_(-bound, bound, generator=tensor_generator(seed, f'{name}.weight'))
        if bias:
            layer.bias.uniform_(-bound, bound, generator=tensor_generator(seed, f'{name}.bias'))
    return layer


def _zero_linear(in_features, out_features, *, bias=False, bias_value=0.0):
    layer = nn.utils.skip_init(nn.Linear, in_features=in_features, out_features=out_features,
                               bias=bias)
    with torch.no_grad():
        layer.weight.zero_()
        if bias:
            layer.bias.fill_(bias_value)
    return layer


def _embedding(rows, dim, *, seed=None, name=None):
    """Zero-initialised table, or uniform(+-1/sqrt(dim)) from a named generator."""
    table = nn.utils.skip_init(nn.Embedding, num_embeddings=rows, embedding_dim=dim)
    with torch.no_grad():
        if name is None:
            table.weight.zero_()
        else:
            bound = 1.0 / float(dim) ** 0.5
            table.weight.uniform_(-bound, bound, generator=tensor_generator(seed, f'{name}.weight'))
    return table


class _Factorized(nn.Module):
    """REG-04 / LRB-01: ``Linear(in, rank, no bias) -> Linear(rank, out)`` replacing an in x out map."""

    def __init__(self, in_features, out_features, rank, *, seed, name, bias):
        super().__init__()
        self.a = _linear(in_features, rank, seed=seed, name=f'{name}.a', bias=False)
        self.b = _linear(rank, out_features, seed=seed, name=f'{name}.b', bias=bias)

    def forward(self, x):
        return self.b(self.a(x))


def _map(in_features, out_features, *, rank, seed, name, bias):
    if 0 < rank < min(in_features, out_features):
        return _Factorized(in_features, out_features, rank, seed=seed, name=name, bias=bias)
    return _linear(in_features, out_features, seed=seed, name=name, bias=bias)


class _GraphNorm(nn.Module):
    """REG-08 / LRB-04: GraphNorm with statistics per graph (or per graph x node kind)."""

    def __init__(self, width, *, per_kind, num_kinds):
        super().__init__()
        self.per_kind, self.num_kinds = bool(per_kind), int(num_kinds)
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))
        self.shift = nn.Parameter(torch.ones(width))   # the learnable mean-shift alpha of GraphNorm

    def forward(self, x, batch_index, graph_count, node_type):
        if self.per_kind:
            segment, size = batch_index * self.num_kinds + node_type, graph_count * self.num_kinds
        else:
            segment, size = batch_index, graph_count
        counts = x.new_zeros((size,)).index_add(0, segment, x.new_ones((x.size(0),))).clamp_min(1.0)
        mean = x.new_zeros((size, x.size(1))).index_add(0, segment, x) / counts[:, None]
        centred = x - self.shift * mean[segment]
        var = x.new_zeros((size, x.size(1))).index_add(0, segment, centred * centred) / counts[:, None]
        # a ONE-node graph has mean == node under either statistic, so its output would be the bias
        # alone whatever the state or the messages; such nodes are normalised over their channels instead
        # (mean-centred, unit variance, the same affine), like LayerNorm.
        chan = x - x.mean(1, keepdim=True)
        channel = self.weight * chan / torch.sqrt((chan * chan).mean(1, keepdim=True) + 1e-5) + self.bias
        if not self.per_kind:
            lone = (counts[segment] == 1.0)[:, None]
            return torch.where(lone, channel, self.weight * centred / torch.sqrt(var[segment] + 1e-5) + self.bias)
        # LRB-04: a (graph, kind) segment with ONE node has mean == node, so its centred value would be 0
        # and the output the bias alone; those nodes use the graph-level statistics instead.
        single = (counts[segment] == 1.0)[:, None]
        g_counts = x.new_zeros((graph_count,)).index_add(0, batch_index, x.new_ones((x.size(0),))).clamp_min(1.0)
        g_mean = x.new_zeros((graph_count, x.size(1))).index_add(0, batch_index, x) / g_counts[:, None]
        g_centred = x - self.shift * g_mean[batch_index]
        g_var = x.new_zeros((graph_count, x.size(1))).index_add(
            0, batch_index, g_centred * g_centred) / g_counts[:, None]
        centred = torch.where(single, g_centred, centred)
        var_node = torch.where(single, g_var[batch_index], var[segment])
        lone = (g_counts[batch_index] == 1.0)[:, None]
        return torch.where(lone, channel, self.weight * centred / torch.sqrt(var_node + 1e-5) + self.bias)


def _make_norm(width, kind, num_kinds):
    if kind == 'layer':
        norm = nn.utils.skip_init(nn.LayerNorm, normalized_shape=width)
        with torch.no_grad():
            norm.weight.fill_(1.0)
            norm.bias.zero_()
        return norm
    return _GraphNorm(width, per_kind=kind == 'graph_kind', num_kinds=num_kinds)


def _csv(text):
    return tuple(part.strip() for part in str(text).split(',') if part.strip())


def _masked_peak(score, key, size, support):
    """Per-key maximum of ``score`` over the supported entries (``-inf`` where a key has none)."""
    shape = (-1,) + (1,) * (score.dim() - 1)
    masked = torch.where(support.view(shape), score, score.new_full((), float('-inf')))
    peak = score.new_full((size,) + tuple(score.shape[1:]), float('-inf'))
    peak = peak.scatter_reduce(0, key.view(shape).expand_as(score), masked, 'amax', include_self=True)
    return masked, peak


def _segment_softmax(score, weight, recv, nodes, null=None):
    """Softmax of ``score [E,H]`` over the edges of each receiver (HET-05 / HET-06).

    Edges with weight 0 (DropEdge, KG drop, hard mask) leave the softmax, so the kept edges are
    renormalised. The edge weight is a soft mask with the SAME semantics as in the mean kinds
    (``_Layer._aggregate``): it enters the softmax as ``weight * exp(score)`` and the receiver's total
    mass is ``mass / max(mass, 1)`` with ``mass`` the summed weight, so a uniform soft mask leaves every
    receiver with ``mass >= 1`` unchanged and only attenuates receivers whose summed weight is below 1;
    with weights in {0, 1} (the training path, hard masks) this is the plain softmax over kept edges.
    ``null`` (RDO-08) is the logit of an extra null slot that absorbs attention mass (like the mean
    kinds' ``degree + softplus(c)`` denominator).
    """
    heads = score.size(1)
    masked, peak = _masked_peak(score, recv, nodes, weight > 0)
    if null is not None:
        peak = torch.maximum(peak, null.detach())
    peak = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak)).detach()
    exp = torch.exp(masked - peak[recv]) * weight[:, None]
    total = score.new_zeros((nodes, heads)).index_add(0, recv, exp)
    if null is not None:
        total = total + torch.exp(null - peak)
        return exp / total[recv].clamp_min(1e-12)
    mass = weight.new_zeros((nodes,)).index_add(0, recv, weight)
    return exp / total[recv].clamp_min(1e-12) * (mass / mass.clamp_min(1.0))[recv][:, None]


# ------------------------------------------------------------------------------ option parsing
def _parse(options, node_kinds):
    raw = dict(_STACK_DEFAULTS)
    raw.update({key: options[key] for key in _STACK_DEFAULTS if key in options})

    def number(key, kind, low=None, high=None, *, strict_high=False):
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{key} must be a number, got {value!r}')
        value = kind(value)
        if low is not None and value < low:
            raise ValueError(f'{key} must be >= {low}, got {value!r}')
        if high is not None and (value >= high if strict_high else value > high):
            raise ValueError(f'{key} must be {"<" if strict_high else "<="} {high}, got {value!r}')
        return value

    for key, choices in _CHOICES.items():
        if raw[key] not in choices:
            raise ValueError(f'{key} must be one of {choices}, got {raw[key]!r}')
    kinds = tuple(node_kinds)
    hub = _csv(raw['mp_hub_kinds'])
    update = None if str(raw['mp_update_kinds']).strip() == 'all' else _csv(raw['mp_update_kinds'])
    for key, names in (('mp_hub_kinds', hub), ('mp_update_kinds', update or ())):
        unknown = [name for name in names if name not in kinds]
        if unknown:
            raise ValueError(f'{key} names unknown node kinds {unknown}; known: {kinds}')
    o = SimpleNamespace(
        layers=number('mp_layers', int, 1), kind=raw['mp_kind'], msg_dim=number('mp_msg_dim', int, 0),
        rank=number('mp_rank', int, 0), tie=number('mp_tie_directions', int, 0, 1),
        root=raw['mp_root'], share=number('mp_share_layers', int, 0, 2), residual=raw['mp_residual'],
        res_init=number('mp_residual_init', float, 0.0, 1.0), res_bound=number('mp_residual_bound', int, 0, 1),
        aggr=raw['mp_aggr'], dnorm=raw['mp_degree_norm'], gate=number('mp_edge_gate', int, 0, 1),
        rev_att=number('mp_rev_attention', int, 0, 1),
        att_drop=number('mp_att_dropout', float, 0.0, 1.0, strict_high=True),
        rev_agg=raw['mp_rev_agg'], hub_max=number('mp_hub_max', int, 0, 1), hub_kinds=hub,
        film=number('mp_relation_film', int, 0, 1), rel_gate=number('mp_rel_gate', int, 0, 1),
        glob=number('mp_global', int, 0, 2), dedup=number('mp_dedup', int, 0, 1), ctx=raw['mp_ctx'],
        bases=number('mp_bases', int, 1), heads=number('mp_heads', int, 1),
        dropedge=number('mp_dropedge', float, 0.0, 1.0, strict_high=True),
        exempt=_csv(raw['mp_dropedge_exempt']),
        kg_drop=number('mp_kg_edge_drop', float, 0.0, 1.0, strict_high=True),
        dropmsg=number('mp_dropmsg', float, 0.0, 1.0, strict_high=True),
        dropnode=number('mp_dropnode', float, 0.0, 1.0, strict_high=True),
        droppath=number('mp_droppath', float, 0.0, 1.0, strict_high=True), norm=raw['mp_norm'],
        update_kinds=update, time_decay=number('mp_time_decay', int, 0, 1),
        shortcut=number('mp_shortcut', int, 0, 1))
    if o.res_bound and o.residual == 'none':
        raise ValueError("mp_residual_bound=1 contradicts mp_residual='none' (no residual scale to bound)")
    if o.kind == 'gine':
        if o.rev_att and o.rev_agg == 'meanmax':
            raise ValueError("mp_rev_attention=1 and mp_rev_agg='meanmax' both define the reverse "
                             'aggregate; use one of them')
    else:
        # gine-only message-function options are ignored (not an error) for the other kinds
        o.msg_dim = o.rank = o.tie = o.gate = o.rev_att = o.hub_max = o.film = 0
        o.root, o.rev_agg, o.ctx = 'linear', 'mean', 'matrix'
    # the dead `o.ignored` attribute (written, never read) is removed; the ignored-option
    # record is cei.options.ignored_options
    if o.att_drop > 0.0 and not (o.rev_att or o.kind in ('simplehgn', 'hgt')):
        # MSG-04 / HET-05 / HET-06: there is no attention to drop out; say so instead of staying silent
        warnings.warn(f'mp_att_dropout={o.att_drop} has no effect for mp_kind={o.kind!r} without '
                      'mp_rev_attention=1 (no attention in this stack)', UserWarning, stacklevel=3)
        o.att_drop = 0.0
    return o


# ------------------------------------------------------------------------------ layer base
class _Layer(nn.Module):
    """Tail shared by every kind: root, global, norm, GELU, dropouts, update mask, residual scale."""

    def __init__(self, width, *, dropout, seed, name, o, dims):
        super().__init__()
        self.width, self.o, self.name, self.seed = width, o, name, seed
        self.num_kinds, self.rel_slots, self.triple_slots = dims
        if o.root == 'linear':
            # LRB-01: mp_rank factorises the four message maps and glob; the root transform stays dense
            self.transform = _linear(width, width, seed=seed, name=f'{name}.transform', bias=True)
        else:   # LRB-01: the residual already carries the node itself; keep a bias vector only
            self.root_bias = nn.Parameter(torch.zeros(width))
        if o.glob:
            self.glob = _map(width, width, rank=o.rank, seed=seed, name=f'{name}.glob', bias=False)
        self.norm = _make_norm(width, o.norm, self.num_kinds)
        self.dropout_rate = float(dropout)
        self._dropout_generator = tensor_generator(seed, f'{name}.dropout')
        self._generators = {}
        if o.residual != 'none':   # MSG-06 / REG-02 / LRB-02 / EVD-02 / ERR-02
            shape = {'scalar': (1,), 'channel': (width,), 'kind': (self.num_kinds,),
                     'kind_vec': (self.num_kinds, width)}[o.residual]
            self.alpha = nn.Parameter(torch.full(shape, float(o.res_init)))
        if o.dnorm == 'null':      # RDO-08: null slot (mean kinds: softplus weight; attention kinds: logit)
            self.null_c = nn.Parameter(torch.full(
                (1,), 0.0 if o.kind in ('simplehgn', 'hgt') else _NULL_INIT))
        if o.rel_gate:             # LCL-03: 2 * sigmoid(0) = 1 at init
            self.rel_gate = nn.Parameter(torch.zeros(self.rel_slots))
        if o.time_decay:           # RDO-05 (b): softplus(-2) ~ 0.13 per source node kind
            self.decay_theta = nn.Parameter(torch.full((self.num_kinds,), -2.0))
        self.groups = {'mean': 1, 'relation': self.rel_slots, 'triple': self.triple_slots}[o.aggr]

    # -------------------------------------------------------------- random helpers
    def _generator(self, purpose):
        if purpose not in self._generators:
            self._generators[purpose] = tensor_generator(self.seed, f'{self.name}.{purpose}')
        return self._generators[purpose]

    def _draw(self, purpose, shape, like):
        return torch.rand(shape, generator=self._generator(purpose)).to(like.device)

    def _dropmsg(self, message):   # REG-07 / LRB-03
        rate = self.o.dropmsg
        if not self.training or rate <= 0.0:
            return message
        keep = self._draw('dropmsg', message.shape, message) >= rate
        return message * keep.to(message.dtype) / (1.0 - rate)

    # -------------------------------------------------------------- weights
    def _edge_weights(self, h, g):
        o, w_f, w_r = self.o, g.w_f, g.w_r
        copies = None if g.dup is None else g.dup.copies
        if self.training:
            keep = None
            if o.dropedge > 0.0:   # REG-01 / STR-02: no rescale, the weighted mean renormalises
                keep = (self._draw('dropedge', (g.src.numel(),), h) >= o.dropedge) | g.exempt
            if o.kg_drop > 0.0:    # LCL-03: knowledge-graph edges only
                kg_keep = (self._draw('kgdrop', (g.src.numel(),), h) >= o.kg_drop) | ~g.kg
                keep = kg_keep if keep is None else keep & kg_keep
            if keep is not None:
                keep = keep.to(h.dtype)
                if g.dup is None:
                    w_f, w_r = w_f * keep, w_r * keep
                else:
                    # (mp2v3-model-03): dedup splits a group of k parallel copies into k weights of
                    # 1/k; a dropped copy must hand its share to the survivors, otherwise the survivor delivers
                    # 1/k of its message in training and all of it at eval. Re-split over the surviving copies.
                    alive = keep.new_zeros((g.dup.groups,)).index_add(0, g.dup.inverse, keep)
                    w_f = g.dup.base * keep / alive[g.dup.inverse].clamp_min(1.0)
                    w_r = w_f * g.dup.no_reverse
                    copies = alive[g.dup.inverse].clamp_min(1.0)
            if o.dropnode > 0.0:   # LRB-03: a dropped node is silent as a sender for this layer
                alive = (self._draw('dropnode', (h.size(0),), h) >= o.dropnode).to(h.dtype)
                w_f, w_r = w_f * alive[g.src], w_r * alive[g.dst]
        # (mp2v2-stack-02): the max terms weight a message by its edge weight WITHOUT the 1/k dedup
        # split (k identical copies must not shrink a max); soft masks and drops still scale/silence it.
        g.peak_f, g.peak_r = (w_f, w_r) if copies is None else (w_f * copies, w_r * copies)
        return w_f, w_r

    def _rel_scale(self, g):
        """LCL-03: learned per-relation scale [E, 1] of the MESSAGE (not of the edge weight: a weight
        scale cancels in the per-group weighted mean). Normalisation is unchanged, like time decay."""
        if not self.o.rel_gate:
            return None
        return (2.0 * torch.sigmoid(self.rel_gate))[g.rel][:, None]

    def _decay(self, h, g):
        if not self.o.time_decay or g.node_time is None:
            return None
        rate = F.softplus(self.decay_theta)[g.node_type]
        return torch.exp(-rate * F.relu(-g.node_time))

    # -------------------------------------------------------------- aggregation
    def _coef(self, weight, recv, send, group, g, floor):
        """Per-edge coefficient of the aggregate, its receiver key and the number of keys."""
        nodes, o = g.nodes, self.o
        if o.dnorm == 'sym':   # LCL-05: node-level weighted degrees over all relations, plain sum
            r_deg = weight.new_zeros((nodes,)).index_add(0, recv, weight)
            s_deg = weight.new_zeros((nodes,)).index_add(0, send, weight)
            return weight / torch.sqrt(r_deg[recv].clamp_min(1.0) * s_deg[send].clamp_min(1.0)), recv, nodes
        if group is None:
            rkey, size = recv, nodes
        else:
            rkey, size = recv * self.groups + group, nodes * self.groups
        degree = weight.new_zeros((size,)).index_add(0, rkey, weight)
        denom = degree + F.softplus(self.null_c) if o.dnorm == 'null' else degree.clamp_min(floor)
        if group is None:
            return weight / denom[recv], rkey, size
        # soft group presence, clamp(group weight sum, max=1) summed per receiver. It equals the hard
        # (degree > 0) count for {0,1} weights (every non-empty group has degree >= 1) and is continuous in a
        # fractional edge weight, so 1e-6 -> 0 no longer doubles the aggregate (design decision 2026-10-11).
        present = degree.clamp(min=0.0, max=1.0).view(nodes, self.groups).sum(1).clamp_min(1.0)
        return weight / (denom[rkey] * present[recv]), rkey, size

    def _aggregate(self, message, weight, recv, send, group, g, floor):
        """Weighted mean of ``message`` into ``recv``; returns ``(aggregate, per-edge share)``.

        mean: STR-01/HET-01 groups the mean per relation/triple (sum over present groups divided
        by their number); degree_norm 'sym' (LCL-05) sums with 1/sqrt(deg_recv * deg_send) over the
        node degrees of all relations (the group is ignored); 'null' (RDO-08) divides by
        softplus(c) + degree instead of max(degree, floor).
        """
        nodes, o = g.nodes, self.o
        if group is None and o.dnorm != 'sym':   # the cei_gnn_mp operation order
            degree = weight.new_zeros((nodes,)).index_add(0, recv, weight)
            denom = degree + F.softplus(self.null_c) if o.dnorm == 'null' else degree.clamp_min(floor)
            agg = message.new_zeros((nodes, message.size(1))).index_add(0, recv, weight[:, None] * message)
            return agg / denom[:, None], weight / denom[recv]
        coef = self._coef(weight, recv, send, group, g, floor)[0]
        return message.new_zeros((nodes, message.size(1))).index_add(0, recv, coef[:, None] * message), coef

    def _group(self, g):
        return {'mean': None, 'relation': g.rel, 'triple': g.triple}[self.o.aggr]

    # -------------------------------------------------------------- forward
    def _norm_apply(self, norm, pre, g):
        if isinstance(norm, _GraphNorm):
            return norm(pre, g.batch_index, g.graph_count, g.node_type)
        return norm(pre)

    def _scale(self, g):
        if self.o.residual == 'none':
            return None
        alpha = torch.tanh(self.alpha) if self.o.res_bound else self.alpha   # ERR-02
        if self.o.residual == 'kind':
            return alpha[g.node_type][:, None]
        if self.o.residual == 'kind_vec':
            return alpha[g.node_type]
        return alpha

    def effective_alpha(self):
        if self.o.residual == 'none':
            return None
        return (torch.tanh(self.alpha) if self.o.res_bound else self.alpha).detach()

    def _update_gate(self, g):
        return None

    def _global(self, h, g):
        if self.o.glob == 1:   # PLN-01 / RDO-01: legacy graph mean
            counts = h.new_zeros((g.graph_count,)).index_add(
                0, g.batch_index, h.new_ones((h.size(0),))).clamp_min(1.0)
            mean = h.new_zeros((g.graph_count, h.size(1))).index_add(0, g.batch_index, h) / counts[:, None]
            return self.glob(mean)[g.batch_index]
        # STR-05: anchor on the index visit (graphs without one read zeros)
        valid = (g.anchor >= 0).to(h.dtype)[:, None]
        return self.glob(h[g.anchor.clamp_min(0)] * valid)[g.batch_index]

    def neighbours(self, h, g, w_f, w_r, decay):
        raise NotImplementedError

    def forward(self, h, g, *, norm=None, depth=0, total=1):
        """Returns ``(h_next, forward share float[E_total])``."""
        o = self.o
        w_f, w_r = self._edge_weights(h, g)
        g.rel_scale = self._rel_scale(g)   # read by the message functions of this layer
        terms, share = self.neighbours(h, g, w_f, w_r, self._decay(h, g))
        pre = self.transform(h) if o.root == 'linear' else self.root_bias.expand(h.size(0), -1)
        for term in terms:
            pre = pre + term
        if o.glob:
            pre = pre + self._global(h, g)
        update = F.gelu(self._norm_apply(norm if norm is not None else self.norm, pre, g))
        if self.training and self.dropout_rate > 0.0:
            keep = torch.rand(update.shape, generator=self._dropout_generator) >= self.dropout_rate
            update = update * keep.to(device=update.device, dtype=update.dtype) / (1.0 - self.dropout_rate)
        if self.training and o.droppath > 0.0:   # REG-05 / LRB-05: whole graph skips the update
            rate = o.droppath * (depth + 1) / max(total, 1)
            alive = (self._draw('droppath', (g.graph_count,), h) >= rate).to(h.dtype)
            update = update * alive[g.batch_index][:, None] / (1.0 - rate)
        gate = self._update_gate(g)
        if gate is not None:
            update = update * gate
        if g.update_mask is not None:            # EVD-03: masked kinds keep their state
            update = update * g.update_mask[:, None].to(update.dtype)
        scale = self._scale(g)
        return (h + update if scale is None else h + scale * update), share


# ------------------------------------------------------------------------------ gine
class GineLayer(_Layer):
    """cei_gnn_mp's layer generalised by MSG-01..05, REG-04, STR-04, LRB-01, HET-03, ERR-05."""

    def __init__(self, width, *, dropout, seed, name, o, dims):
        super().__init__(width, dropout=dropout, seed=seed, name=name, o=o, dims=dims)
        m = o.msg_dim if 0 < o.msg_dim < width else width   # MSG-01 / REG-04
        self.m, self.bottleneck, self.tied = m, m != width, bool(o.tie)
        table = o.ctx == 'table'
        self.w_fwd = _map(width, m, rank=o.rank, seed=seed, name=f'{name}.w_fwd', bias=False)
        if not self.tied:
            self.w_rev = _map(width, m, rank=o.rank, seed=seed, name=f'{name}.w_rev', bias=False)
        for side in ('fwd',) if self.tied else ('fwd', 'rev'):
            if table:   # HET-03: table[triple] + payload projection instead of a context matrix
                noise = torch.empty(self.triple_slots, m).normal_(
                    0.0, 1.0 / float(width) ** 0.5, generator=tensor_generator(seed, f'{name}.table_{side}'))
                setattr(self, f'table_{side}', nn.Parameter(noise))
                setattr(self, f'pay_{side}', _linear(PAYLOAD_WIDTH, m, seed=seed,
                                                     name=f'{name}.pay_{side}', bias=False))
            else:
                setattr(self, f'c_{side}', _map(width, m, rank=o.rank, seed=seed,
                                                name=f'{name}.c_{side}', bias=False))
        if self.tied:   # STR-04 / LRB-01: one W and one C; asymmetry from two direction vectors dir_bias[2, m]
            self.dir_bias = nn.Parameter(torch.zeros(2, m))
        if self.bottleneck:
            self.up_fwd = _linear(m, width, seed=seed, name=f'{name}.up_fwd', bias=False)
            if o.rev_agg == 'meanmax':   # MSG-05: mix_rev = [up_rev, 0, 0] replaces up_rev
                up = _linear(m, width, seed=seed, name=f'{name}.up_rev', bias=False)
                mix = _zero_linear(2 * m + 1, width)
                with torch.no_grad():
                    mix.weight[:, :m].copy_(up.weight)   # the layer's own up_rev init: starts as DEFAULT
                self.mix_rev = mix
            else:
                self.up_rev = _linear(m, width, seed=seed, name=f'{name}.up_rev', bias=False)
        elif o.rev_agg == 'meanmax':
            mix = _zero_linear(2 * m + 1, width)
            with torch.no_grad():
                mix.weight[:, :m].copy_(torch.eye(m))
            self.mix_rev = mix
        if o.film:   # MSG-02
            self.gamma_fwd = _embedding(self.rel_slots, m)
            self.gamma_rev = _embedding(self.rel_slots, m)
        if o.gate:   # MSG-03
            self.gate_fwd = _zero_linear(3 * m, 1, bias=True, bias_value=_GATE_BIAS)
            self.gate_rev = _zero_linear(3 * m, 1, bias=True, bias_value=_GATE_BIAS)
        if o.rev_att:   # MSG-04
            self.att = _zero_linear(m, 1)
        if o.hub_max:   # ERR-05
            self.w_max = _zero_linear(m, width)

    def _context(self, g):
        o = self.o
        if o.ctx == 'table':
            c_f = self.table_fwd[g.triple] + self.pay_fwd(g.payload)
            if self.tied:
                return c_f, c_f
            return c_f, self.table_rev[g.triple] + self.pay_rev(g.payload)
        c_f = self.c_fwd(g.ctx)
        return c_f, (c_f if self.tied else self.c_rev(g.ctx))

    def neighbours(self, h, g, w_f, w_r, decay):
        o, src, dst = self.o, g.src, g.dst
        a_f = self.w_fwd(h)[src]   # project the N nodes once, then gather (E > N)
        a_r = (self.w_fwd if self.tied else self.w_rev)(h)[dst]
        c_f, c_r = self._context(g)
        if o.gate:   # MSG-03: scalar gate from content + context + the other endpoint
            gate_f = torch.sigmoid(self.gate_fwd(torch.cat((a_f, c_f, a_r), dim=-1)))[:, 0]
            gate_r = torch.sigmoid(self.gate_rev(torch.cat((a_r, c_r, a_f), dim=-1)))[:, 0]
        raw_f, raw_r = a_f, a_r
        if o.film:   # MSG-02: relation-wise diagonal modulation of the content term
            raw_f = a_f * (1.0 + torch.tanh(self.gamma_fwd(g.rel)))
            raw_r = a_r * (1.0 + torch.tanh(self.gamma_rev(g.rel)))
        pre_f, pre_r = raw_f + c_f, raw_r + c_r
        if self.tied:
            pre_f = pre_f + self.dir_bias[0]
            pre_r = pre_r + self.dir_bias[1]
        m_f, m_r = F.relu(pre_f), F.relu(pre_r)
        if decay is not None:   # RDO-05: old senders count less, the degree is unchanged
            m_f, m_r = m_f * decay[src][:, None], m_r * decay[dst][:, None]
        scale = getattr(g, 'rel_scale', None)
        if scale is not None:   # LCL-03: after normalisation, so the scale does not cancel in the mean
            m_f, m_r = m_f * scale, m_r * scale
        if o.gate:   # MSG-03: the gate scales the message; the denominator stays the un-gated weight
            m_f, m_r = m_f * gate_f[:, None], m_r * gate_r[:, None]
        m_f, m_r = self._dropmsg(m_f), self._dropmsg(m_r)
        floor = 1.0
        group = self._group(g)
        agg_f, share = self._aggregate(m_f, w_f, dst, src, group, g, floor)
        if o.gate:   # the reported share is the effective coefficient of the gated message
            share = share * gate_f
        terms = []
        # reverse channel: receiver is src, sender is dst
        if o.rev_att:   # MSG-04: GATv2 attention replaces the reverse mean
            coef, rkey, size = self._coef(w_r, src, dst, group, g, floor)
            score = self.att(F.leaky_relu(a_r + c_r + a_f, 0.2))[:, 0]
            masked, peak = _masked_peak(score, rkey, size, coef > 0)
            peak = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak)).detach()
            exp = torch.exp(masked - peak[rkey])
            # attention re-weights the mean coefficients inside each (receiver[, relation]) group and keeps
            # the group's total mass, so uniform attention (att zero-init) is exactly the DEFAULT aggregate
            mass = coef.new_zeros((size,)).index_add(0, rkey, coef)
            kept = coef.new_zeros((size,)).index_add(0, rkey, coef * exp)
            alpha = coef * exp * (mass / kept.clamp_min(1e-12))[rkey]
            if self.training and o.att_drop > 0.0:
                keep = self._draw('attdrop', alpha.shape, alpha) >= o.att_drop
                alpha = alpha * keep.to(alpha.dtype) / (1.0 - o.att_drop)
            agg_r = m_r.new_zeros((g.nodes, self.m)).index_add(0, src, alpha[:, None] * m_r)
        else:
            agg_r, _ = self._aggregate(m_r, w_r, src, dst, group, g, floor)
        terms.append(self.up_fwd(agg_f) if self.bottleneck else agg_f)
        if o.rev_agg == 'meanmax':   # MSG-05: mean + max + log-degree scaler
            peak = m_r.new_zeros((g.nodes, self.m)).scatter_reduce(
                0, src[:, None].expand(-1, self.m), g.peak_r[:, None] * m_r, 'amax', include_self=True)
            scaler = torch.log1p(w_r.new_zeros((g.nodes,)).index_add(0, src, w_r)) / 4.0
            terms.append(self.mix_rev(torch.cat((agg_r, peak, scaler[:, None]), dim=-1)))
        else:
            terms.append(self.up_rev(agg_r) if self.bottleneck else agg_r)
        if o.hub_max:   # ERR-05: max over the messages a hub node receives (forward into dst, reverse into src)
            peak_f = m_f.new_zeros((g.nodes, self.m)).scatter_reduce(
                0, dst[:, None].expand(-1, self.m), g.peak_f[:, None] * m_f, 'amax', include_self=True)
            peak_r = m_r.new_zeros((g.nodes, self.m)).scatter_reduce(
                0, src[:, None].expand(-1, self.m), g.peak_r[:, None] * m_r, 'amax', include_self=True)
            peak = torch.maximum(peak_f, peak_r)
            terms.append(self.w_max(peak * g.hub_mask[:, None].to(peak.dtype)))
        return terms, share


# ------------------------------------------------------------------------------ rgcn / diag
class RGCNLayer(_Layer):
    """HET-02: R-GCN basis decomposition over triples with a rank-16 context path."""

    def __init__(self, width, *, dropout, seed, name, o, dims):
        super().__init__(width, dropout=dropout, seed=seed, name=name, o=o, dims=dims)
        bases, rank = o.bases, max(1, min(_CTX_RANK, width // 2))
        bound = 1.0 / float(width) ** 0.5
        self.V = nn.Parameter(torch.empty(bases, width, width).uniform_(
            -bound, bound, generator=tensor_generator(seed, f'{name}.V')))
        for side in ('fwd', 'rev'):
            noise = torch.empty(self.triple_slots, bases).normal_(
                0.0, 1.0 / float(bases) ** 0.5, generator=tensor_generator(seed, f'{name}.a_{side}'))
            setattr(self, f'a_{side}', nn.Parameter(1.0 / float(bases) ** 0.5 + 0.1 * noise))
            setattr(self, f'c_{side}', _Factorized(width, width, rank, seed=seed,
                                                   name=f'{name}.c_{side}', bias=False))

    def neighbours(self, h, g, w_f, w_r, decay):
        projected = torch.einsum('nd,bde->nbe', h, self.V)
        m_f = F.relu((projected[g.src] * self.a_fwd[g.triple][:, :, None]).sum(1) + self.c_fwd(g.ctx))
        m_r = F.relu((projected[g.dst] * self.a_rev[g.triple][:, :, None]).sum(1) + self.c_rev(g.ctx))
        return _mean_channels(self, m_f, m_r, w_f, w_r, g, decay)


class DiagLayer(_Layer):
    """HET-04: one shared W per layer, triple-typed diagonal gains on the sender (CompGCN-style)."""

    def __init__(self, width, *, dropout, seed, name, o, dims):
        super().__init__(width, dropout=dropout, seed=seed, name=name, o=o, dims=dims)
        rank = max(1, min(_CTX_RANK, width // 2))
        self.w = _linear(width, width, seed=seed, name=f'{name}.w', bias=False)
        self.g_fwd = nn.Parameter(torch.zeros(self.triple_slots, width))
        self.g_rev = nn.Parameter(torch.zeros(self.triple_slots, width))
        for side in ('fwd', 'rev'):
            setattr(self, f'c_{side}', _Factorized(width, width, rank, seed=seed,
                                                   name=f'{name}.c_{side}', bias=False))

    def neighbours(self, h, g, w_f, w_r, decay):
        m_f = F.relu(self.w(h[g.src] * (1.0 + self.g_fwd[g.triple])) + self.c_fwd(g.ctx))
        m_r = F.relu(self.w(h[g.dst] * (1.0 + self.g_rev[g.triple])) + self.c_rev(g.ctx))
        return _mean_channels(self, m_f, m_r, w_f, w_r, g, decay)


def _mean_channels(layer, m_f, m_r, w_f, w_r, g, decay):
    if decay is not None:
        m_f, m_r = m_f * decay[g.src][:, None], m_r * decay[g.dst][:, None]
    if getattr(g, 'rel_scale', None) is not None:   # LCL-03: message scale, after normalisation
        m_f, m_r = m_f * g.rel_scale, m_r * g.rel_scale
    m_f, m_r = layer._dropmsg(m_f), layer._dropmsg(m_r)
    group = layer._group(g)
    agg_f, share = layer._aggregate(m_f, w_f, g.dst, g.src, group, g, 1.0)
    agg_r, _ = layer._aggregate(m_r, w_r, g.src, g.dst, group, g, 1.0)
    return [agg_f, agg_r], share


# ------------------------------------------------------------------------------ attention kinds
class _AttentionLayer(_Layer):
    """Common half-edge machinery: every edge sends to both endpoints, one softmax per receiver."""

    def _attend(self, score, w_f, w_r, g):
        o = self.o
        weight = torch.cat((w_f, w_r))
        null = self.null_c if o.dnorm == 'null' else None   # RDO-08: null slot inside the softmax
        alpha = _segment_softmax(score, weight, g.recv2, g.nodes, null)
        if o.dnorm == 'sym':   # LCL-05: hub damping sqrt(deg_recv / deg_send) on the attention coefficients
            degree = weight.new_zeros((g.nodes,)).index_add(0, g.recv2, weight).clamp_min(1.0)
            alpha = alpha * torch.sqrt(degree[g.recv2] / degree[g.send2])[:, None]
        if self.training and self.o.att_drop > 0.0:   # HET-05 / HET-06
            keep = self._draw('attdrop', alpha.shape, alpha) >= self.o.att_drop
            alpha = alpha * keep.to(alpha.dtype) / (1.0 - self.o.att_drop)
        return alpha

    def _values(self, values, decay, g):
        if decay is not None:
            values = values * decay[g.send2][:, None]
        if getattr(g, 'rel_scale', None) is not None:   # LCL-03: message scale (both half-edge directions)
            values = values * torch.cat((g.rel_scale, g.rel_scale))
        return self._dropmsg(values)

    def _collect(self, values, alpha, g, heads):
        edges = values.size(0)
        weighted = values.view(edges, heads, values.size(1) // heads) * alpha[:, :, None]
        agg = values.new_zeros((g.nodes, heads, weighted.size(2))).index_add(0, g.recv2, weighted)
        return agg.view(g.nodes, -1), alpha[:g.src.numel()].mean(1)


class SimpleHGNLayer(_AttentionLayer):
    """HET-05: one softmax over forward + reverse half-edges, edge-type term from the context."""

    def __init__(self, width, *, dropout, seed, name, o, dims):
        super().__init__(width, dropout=dropout, seed=seed, name=name, o=o, dims=dims)
        heads = o.heads
        self.heads, self.head_dim = heads, width // heads
        bound = 1.0 / float(self.head_dim) ** 0.5
        self.w = _linear(width, width, seed=seed, name=f'{name}.w', bias=False)
        for side in ('l', 'r'):
            setattr(self, f'a_{side}', nn.Parameter(torch.empty(heads, self.head_dim).uniform_(
                -bound, bound, generator=tensor_generator(seed, f'{name}.a_{side}'))))
        self.edge_logit = _linear(width, heads, seed=seed, name=f'{name}.edge_logit', bias=True)
        self.dir_emb = nn.Parameter(torch.zeros(2, width))

    def neighbours(self, h, g, w_f, w_r, decay):
        wh = self.w(h)
        heads = wh.view(h.size(0), self.heads, self.head_dim)
        left, right = (heads * self.a_l).sum(-1), (heads * self.a_r).sum(-1)
        ctx2 = torch.cat((g.ctx + self.dir_emb[0], g.ctx + self.dir_emb[1]))
        score = F.leaky_relu(left[g.recv2] + right[g.send2] + self.edge_logit(ctx2), 0.2)
        alpha = self._attend(score, w_f, w_r, g)
        values = self._values(wh[g.send2], decay, g)
        agg, share = self._collect(values, alpha, g, self.heads)
        return [agg], share


class HGTLayer(_AttentionLayer):
    """HET-06: dot-product attention, triple-typed diagonal key/value maps, per-triple prior mu,
    per-kind skip gate on the update."""

    def __init__(self, width, *, dropout, seed, name, o, dims):
        super().__init__(width, dropout=dropout, seed=seed, name=name, o=o, dims=dims)
        heads = o.heads
        self.heads, self.head_dim, self.key_dim = heads, width // heads, _DK
        types = 2 * self.triple_slots
        self.q = _linear(width, heads * _DK, seed=seed, name=f'{name}.q', bias=False)
        self.k = _linear(width, heads * _DK, seed=seed, name=f'{name}.k', bias=False)
        self.v = _linear(width, width, seed=seed, name=f'{name}.v', bias=False)
        self.out = _linear(width, width, seed=seed, name=f'{name}.out', bias=False)
        self.gk = nn.Parameter(torch.zeros(types, heads * _DK))
        self.gv = nn.Parameter(torch.zeros(types, width))
        self.mu = nn.Parameter(torch.ones(types, heads))
        self.skip = nn.Parameter(torch.zeros(self.num_kinds))
        self.edge_logit = _zero_linear(width, heads)   # edge context (relation, triple, payload) -> per-head logit

    def _update_gate(self, g):   # 2 * sigmoid(0) = 1: the skip starts neutral
        return (2.0 * torch.sigmoid(self.skip))[g.node_type][:, None]

    def neighbours(self, h, g, w_f, w_r, decay):
        nodes, heads = h.size(0), self.heads
        q = self.q(h).view(nodes, heads, _DK)
        k = self.k(h).view(nodes, heads, _DK)
        v = self.v(h)
        kind = torch.cat((g.triple, g.triple + self.triple_slots))
        k_e = k[g.send2] * (1.0 + self.gk[kind].view(-1, heads, _DK))
        score = (q[g.recv2] * k_e).sum(-1) / math.sqrt(_DK) * self.mu[kind]
        score = score + self.edge_logit(torch.cat((g.ctx, g.ctx)))
        alpha = self._attend(score, w_f, w_r, g)
        values = v[g.send2] * (1.0 + self.gv[kind])
        agg, share = self._collect(self._values(values, decay, g), alpha, g, heads)
        return [self.out(agg)], share


_LAYERS = {'gine': GineLayer, 'rgcn': RGCNLayer, 'diag': DiagLayer, 'simplehgn': SimpleHGNLayer,
           'hgt': HGTLayer}


# ------------------------------------------------------------------------------ stack
class MPStack(nn.Module):
    def __init__(self, width, *, dropout, seed, name, num_relations, num_triples, num_node_types,
                 relation_names, node_kinds, options):
        super().__init__()
        width = int(width)
        self.o = o = _parse(options, node_kinds)
        if o.kind in ('simplehgn', 'hgt') and width % o.heads:
            raise ValueError(f"mp_kind={o.kind!r} needs width {width} divisible by mp_heads={o.heads}")
        self.width, self.name, self.seed = width, name, seed
        self.num_relations, self.num_triples = int(num_relations), int(num_triples)
        self.relation_names, self.node_kinds = tuple(relation_names), tuple(node_kinds)
        self.num_kinds = int(num_node_types)
        extra = 1 if o.shortcut else 0   # STR-03: one extra relation id and triple id
        dims = (self.num_kinds, self.num_relations + extra, self.num_triples + extra)
        self.iterations = o.layers
        count = 1 if o.share else o.layers   # REG-03 / LCL-01 / LRB-01: tie the layers
        cls = _LAYERS[o.kind]
        self.layers = nn.ModuleList(
            [cls(width, dropout=dropout, seed=seed, name=f'{name}.{index}', o=o, dims=dims)
             for index in range(count)])
        if o.kind == 'rgcn':   # HET-02: one basis bank V shared by all layers (mp_share=1 is the rgcn default there)
            for layer in list(self.layers)[1:]:
                layer.V = self.layers[0].V
        if o.share == 2:   # own norm per iteration on shared weights
            self.iter_norms = nn.ModuleList(
                [_make_norm(width, o.norm, self.num_kinds) for _ in range(1, o.layers)])
        if o.shortcut and o.ctx != 'table':   # the table path reads table[triple], never g.ctx
            bound = 1.0 / float(width) ** 0.5
            self.shortcut_context = nn.Parameter(torch.empty(width).uniform_(
                -bound, bound, generator=tensor_generator(seed, f'{name}.shortcut_context')))
        # name-based lookups; absent names are ignored, ids never exceed the table
        ids = {rel: index for index, rel in enumerate(self.relation_names)}
        size = max(self.num_relations, len(self.relation_names)) + 1
        self._ids = ids
        exempt, kg = torch.zeros(size, dtype=torch.bool), torch.zeros(size, dtype=torch.bool)
        for rel in o.exempt:
            if rel in ids:
                exempt[ids[rel]] = True
        for rel, index in ids.items():
            if rel.startswith(_KG_PREFIX) or rel.startswith(_REV_PREFIX + _KG_PREFIX):   # LCL-03: both directions
                kg[index] = True
        self.register_buffer('_exempt_rel', exempt, persistent=False)
        self.register_buffer('_kg_rel', kg, persistent=False)
        hub = torch.zeros(self.num_kinds, dtype=torch.bool)
        update = torch.ones(self.num_kinds, dtype=torch.bool)
        if o.update_kinds is not None:
            update.zero_()
            for kind in o.update_kinds:
                update[self.node_kinds.index(kind)] = True
        for kind in o.hub_kinds:
            hub[self.node_kinds.index(kind)] = True
        self.register_buffer('_hub_kind', hub, persistent=False)
        self.register_buffer('_update_kind', update, persistent=False)
        self._visit_kind = self.node_kinds.index('visit') if 'visit' in self.node_kinds else None

    # -------------------------------------------------------------- per-forward graph tensors
    def _shortcut_edges(self, src, dst, rel, weight, nodes):
        """STR-03: measurement -> knowledge through instance_of o (member_of | measures | assesses)."""
        inner = self._ids.get(_SHORTCUT_INNER)
        outer = [self._ids[name] for name in _SHORTCUT_OUTER if name in self._ids]
        if inner is None or not outer:
            return None
        first = torch.nonzero(rel == inner, as_tuple=True)[0]
        second = torch.nonzero(torch.isin(rel, torch.tensor(outer, device=rel.device)), as_tuple=True)[0]
        if first.numel() == 0 or second.numel() == 0:
            return None
        order = torch.argsort(src[second])
        second, keys = second[order], src[second][order]
        counts = torch.bincount(keys, minlength=nodes)
        starts = torch.cumsum(counts, 0) - counts
        middle = dst[first]
        repeat = counts[middle]
        total = int(repeat.sum())
        if total == 0:
            return None
        owner = torch.repeat_interleave(torch.arange(first.numel(), device=src.device), repeat)
        within = torch.arange(total, device=src.device) - torch.repeat_interleave(
            torch.cumsum(repeat, 0) - repeat, repeat)
        pick = second[starts[middle][owner] + within]
        return src[first][owner], dst[pick], weight[first][owner] * weight[pick]

    def _graph(self, h, edge_index, context, metadata, edge_weight, node_time):
        o, nodes = self.o, h.size(0)
        if o.time_decay and node_time is None:   # RDO-05: a missing time column must not switch the option off
            raise ValueError('mp_time_decay=1 needs node_time (float[N]); got None')
        src, dst = edge_index[0], edge_index[1]
        rel, triple = metadata.edge_relation, metadata.edge_triple
        edges = src.numel()
        weight = h.new_ones((edges,)) if edge_weight is None else edge_weight.to(device=h.device, dtype=h.dtype)
        ctx = context
        top = self._kg_rel.numel() - 1
        kg, exempt = self._kg_rel[rel.clamp_max(top)], self._exempt_rel[rel.clamp_max(top)]
        payload = None
        if o.ctx == 'table':   # HET-03: the last PAYLOAD_WIDTH edge columns, left-padded with zeros
            attr = metadata.edge_attr.to(h.dtype)[:, -PAYLOAD_WIDTH:]
            payload = F.pad(attr, (PAYLOAD_WIDTH - attr.size(1), 0))
        if o.shortcut:
            extra = self._shortcut_edges(src, dst, rel, weight, nodes)
            if extra is not None:
                s_src, s_dst, s_weight = extra
                count = s_src.numel()
                fill = torch.zeros(count, dtype=torch.bool, device=h.device)
                src, dst = torch.cat((src, s_src)), torch.cat((dst, s_dst))
                rel = torch.cat((rel, rel.new_full((count,), self.num_relations)))
                triple = torch.cat((triple, triple.new_full((count,), self.num_triples)))
                ctx = torch.cat((ctx, ctx.new_zeros((count, ctx.size(1))) if o.ctx == 'table'
                                 else self.shortcut_context.to(ctx.dtype).expand(count, -1)))
                weight = torch.cat((weight, s_weight))
                kg, exempt = torch.cat((kg, fill)), torch.cat((exempt, fill))
                if payload is not None:
                    payload = torch.cat((payload, payload.new_zeros((count, PAYLOAD_WIDTH))))
        w_f = w_r = weight
        dup = None
        # fv10-m1-03: the mirror test (has_reverse) looks only at the (dst, src) NODE pair of ANY
        # relation, not at the paired relation: has_visit <-> index_visit_of and
        # has_prior_diagnosis <-> recurrence_of count as mirrors too (~5.8% of edges lose their
        # reverse message on the real graphs), not only same-relation pairs such as comorbid_with.
        if o.dedup:   # STR-07: one message per symmetric pair, parallel edges count once
            top_rel = int(rel.max()) + 1 if rel.numel() else 1
            key = src * nodes + dst
            _, inverse, counts = torch.unique(key * top_rel + rel, return_inverse=True, return_counts=True)
            w_f = weight / counts[inverse].to(weight.dtype)
            has_reverse = torch.isin(dst * nodes + src, key) & (src != dst)
            w_r = w_f * (~has_reverse).to(weight.dtype)
            dup = SimpleNamespace(base=weight, inverse=inverse, groups=counts.numel(),
                                  copies=counts[inverse].to(weight.dtype),
                                  no_reverse=(~has_reverse).to(weight.dtype))
        g = SimpleNamespace(
            nodes=nodes, src=src, dst=dst, rel=rel, triple=triple, ctx=ctx, payload=payload,
            w_f=w_f, w_r=w_r, dup=dup, kg=kg, exempt=exempt, batch_index=metadata.batch_index,
            graph_count=int(metadata.graph_count), node_type=metadata.node_type,
            node_time=None if (node_time is None or not o.time_decay) else node_time.to(h.dtype),
            update_mask=None if o.update_kinds is None else self._update_kind[metadata.node_type],
            hub_mask=self._hub_kind[metadata.node_type] if o.hub_max else None,
            anchor=self._anchor(src, rel, w_f, metadata, nodes) if o.glob == 2 else None,
            recv2=torch.cat((dst, src)), send2=torch.cat((src, dst)))
        return g

    def _anchor(self, src, rel, weight, metadata, nodes):
        """STR-05: index visit per graph = source of a live (weight > 0) index_visit_of edge, else the
        last visit node; so an all-zero explanation mask anchors like the edgeless graph."""
        graphs, batch = int(metadata.graph_count), metadata.batch_index
        if nodes >= 1 << 24:
            raise ValueError('mp_global=2 supports fewer than 2**24 nodes per batch')

        def last(index, values):   # per-graph maximum node id; float32 because MPS has no int64 amax
            base = torch.full((graphs,), -1.0, dtype=torch.float32, device=src.device)
            return base.scatter_reduce(0, index, values.to(torch.float32), 'amax', include_self=True).long()

        anchor = torch.full((graphs,), -1, dtype=torch.long, device=src.device)
        if self._visit_kind is not None:
            visits = torch.nonzero(metadata.node_type == self._visit_kind, as_tuple=True)[0]
            anchor = last(batch[visits], visits)
        index_rel = self._ids.get('index_visit_of')
        if index_rel is not None:
            chosen = torch.nonzero((rel == index_rel) & (weight > 0), as_tuple=True)[0]
            if chosen.numel():
                by_edge = last(batch[src[chosen]], src[chosen])
                anchor = torch.where(by_edge >= 0, by_edge, anchor)
        return anchor

    # -------------------------------------------------------------- forward
    def forward(self, h, edge_index, context, metadata, *, edge_weight=None, node_time=None):
        g = self._graph(h, edge_index, context, metadata, edge_weight, node_time)
        edges = edge_index.size(1)
        states, alphas, share = [h], [], None
        for index in range(self.iterations):
            layer = self.layers[0 if self.o.share else index]
            norm = self.iter_norms[index - 1] if (self.o.share == 2 and index > 0) else None
            h, share = layer(h, g, norm=norm, depth=index, total=self.iterations)
            states.append(h)
            alphas.append(layer.effective_alpha())
        alpha = None if alphas[0] is None else torch.stack(alphas)
        return h, {'edge_share': share[:edges], 'states': states, 'alpha': alpha}
