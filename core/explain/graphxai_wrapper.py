"""GraphXAI-compatible wrappers for the clinical_graph_v2 baseline adapters.

`ClinicalGraphXAIWrapper` covers gsat and protgnn and -- via `is_method_adapter=False` -- the
black-box `ClinicalGNN` baselines. `ClinicalGNN` isn't a ClinicalMethodAdapter (it's not in
METHOD_REGISTRY): its forward(data) returns a plain logits tensor, with no MethodOutput wrapper
and no `epoch` argument. `GraphCareGraphXAIWrapper` adds GraphCare's visit context
(visit_membership_index, num_visits, global_node_mask), which GraphXAI never supplies, so it is
held in the wrapper context like the node types and the edges.

Protocol (the same for every final method, CEI included):

- `set_context(batch) -> x`, with x = [numeric row (node_dim) | token vector (token_dim)] for an
  adapter with `embed_tokens` (gsat, protgnn, graphcare): the token block is the adapter's own
  token-embedding lookup, and `forward` hands it back to the adapter as `token_vectors`, so
  Grad/IG attribute node identity as well as values. The node type, the edges, the edge metadata
  and (GraphCare) the visit context stay fixed in the wrapper context. The black-box ClinicalGNN
  has no token hook; its x is the numeric row alone.
- `forward(x, edge_index, batch=None, edge_mask=None, edge_attr_override=None) -> logits [1, C]`
  (the black-box wrapper takes the mask through PyG's MessagePassing hook instead of the
  keyword, see `ClinicalGNNGraphXAIWrapper`).
- `verify(x, edge_index) -> (ok, max_abs_diff)`: the wrapper against the adapter's own forward
  on the RAW batch, so an error in the held context fails the check.
- `ig_feature_spec()`, `ig_feature_provenance()`, `ig_path()` (no-op here), `method`,
  `builtin_node_importance`, `builtin_detail`, `builtin_class_specific`, `inert_edge_mask()`,
  `deletion_forward(data)`.

Like every wrapper here these score exactly one graph at a time: graph_count must be 1.

`edge_mask` reaches the model through each adapter's own `external_edge_mask` hook:
- gsat: every message of the shared GIN in both passes (extraction and prediction), once each;
- protgnn: every `_RelationPayloadLayer` message and degree, once;
- graphcare: the BAT messages, subset with GraphCare's own global-node edge filter; edges with a
  global endpoint carry no message (`inert_edge_mask`).

Built-in explanations (the trained model's own):
- gsat: the deterministic node attention of a fresh eval pass (`adapter.explain`), with the
  lifted edge attention of that pass in the detail. Class-agnostic.
- protgnn: the adapter's `explain_detail` -- the own-class prototype with the largest
  classifier weight x activation and the adapter's MCTS (`_initial_coalition` + `_mcts_project`,
  the search that projected the prototypes) with the model's own settings; the node vector is
  that subgraph's binary mask. One search per visit and class. Class-specific.
- graphcare: the node attention of the last layer from a fresh unmasked eval forward,
  sum over membership pairs (v, n) of `last_visit_weight` (alpha * beta), then the float64 mean
  over the nodes with the same visit set (equal visit sets tie exactly). Class-agnostic.
"""
from __future__ import annotations

import types
from contextlib import contextmanager

import torch
import torch.nn as nn

from torch_geometric.explain.algorithm.utils import clear_masks, set_masks

from core.explain.fidelity_v2 import removable_node_mask, tie_summary
from core.method_base import ClinicalBatch, read_clinical_batch
from core.tensorize import PAYLOAD_WIDTH

# Methods with a per-node built-in readout in this wrapper. Others raise rather than
# fabricating a score.
_SUPPORTS_BUILTIN_EXPLAIN = frozenset({"gsat", "protgnn"})
_CLASS_SPECIFIC_BUILTIN = frozenset({"protgnn"})

PROTGNN_SEARCH_ALGORITHM = ("adapter MCTS (_initial_coalition + _mcts_project), the search that "
                            "projected the prototypes")
PROTGNN_OFFICIAL_SETTINGS = "official ProtGNN 10/5/10/10"
PROTGNN_SUBGRAPH_SCORING = ("induced subgraph re-encoded: excluded nodes and their edges are "
                            "dropped before encoding (the official code zero-fills excluded nodes)")
PROTGNN_EXPLANATION_STYLE = ("per-prediction subgraph nearest to the explained class's "
                             "most-contributing prototype (ProtGNN+ style)")
GRAPHCARE_VALUE_DEPENDENCE = (
    "the value depends only on the node's visit set (alpha_projection is linear; the node term "
    "cancels in the softmax over visits); float32 rounding makes the values of one multi-visit "
    "set differ slightly, so every node gets the float64 mean of its visit set "
    "(max_within_visit_set_spread)")
PROTGNN_SUBGRAPH_SIZE_NOTE = (
    "the subgraph has at most max_atoms nodes (structural hubs included); only its eligible "
    "evidence nodes are deleted, so its k is at most that count; subgraph_eligible_count is "
    "recorded per visit")
GSAT_EDGE_ATTENTION_RULE = "att[source] * att[target] (the lifted attention of the prediction pass)"


def _tie_stats(node_type, importance) -> dict:
    """`fidelity_v2.tie_summary` over the eligible nodes; all None when there is none."""
    try:
        return tie_summary(node_type, importance)
    except ValueError:
        return {"distinct_values_eligible": None, "largest_tie_group_eligible": None,
                "constant_over_eligible": None}


def set_keyword_mask(model, mask, edge_index) -> None:
    """PyG `set_masks` for a caller's mask tensor, keeping its autograd graph.

    After GNNExplainer, `clear_masks` leaves `_edge_mask` registered as a (None) parameter, and
    `set_masks` would then wrap a plain tensor in a new Parameter: a leaf with no path back to
    the caller's tensor. Such cleared registrations are dropped first, so the mask installed is
    the caller's tensor itself."""
    from torch_geometric.nn import MessagePassing

    for module in model.modules():
        if (isinstance(module, MessagePassing) and "_edge_mask" in module._parameters
                and module._parameters["_edge_mask"] is None):
            del module._parameters["_edge_mask"]
    set_masks(model, mask, edge_index, apply_sigmoid=False)


def _per_visit_set_mean(attention, membership):
    """(float64 attention, largest spread): every node with membership pairs gets the float64 mean
    of the float32 values of the nodes with the same visit set (the sorted visit ordinals of its
    pairs), so equal visit sets tie exactly; nodes without a pair keep their value (0). The spread
    is the largest max - min inside one visit set before the mean."""
    canonical = attention.to(torch.float64)
    visits_of: dict[int, list[int]] = {}
    for visit, node in zip(membership[0].tolist(), membership[1].tolist()):
        visits_of.setdefault(int(node), []).append(int(visit))
    members_of: dict[tuple, list[int]] = {}
    for node, visits in visits_of.items():
        members_of.setdefault(tuple(sorted(visits)), []).append(node)
    spread = 0.0
    for members in members_of.values():
        index = torch.tensor(sorted(members), dtype=torch.long, device=canonical.device)
        values = canonical[index]
        spread = max(spread, float((values.max() - values.min()).item()))
        canonical[index] = values.mean()
    return canonical, spread


class _ContextWrapper(nn.Module):
    """Context handling, x layout, verify and the shared protocol parts of the baseline wrappers."""

    builtin_class_specific = False

    def __init__(self, adapter, *, method: str, node_dim: int, edge_dim: int, num_tokens: int,
                 num_triples: int, num_relations: int, is_method_adapter: bool = True):
        super().__init__()
        self.adapter = adapter
        self._method = method
        self._node_dim = int(node_dim)
        self._edge_dim = int(edge_dim)
        self._num_tokens = int(num_tokens)
        self._num_triples = int(num_triples)
        self._num_relations = int(num_relations)
        self._is_method_adapter = is_method_adapter
        # The token block is explained wherever the adapter exposes its lookup.
        self._explains_tokens = is_method_adapter and hasattr(adapter, "embed_tokens")
        self._context: ClinicalBatch | None = None
        self._raw_context = None
        self._x_shape = None

    @property
    def method(self) -> str:
        return self._method

    def _read(self, batch) -> ClinicalBatch:
        return read_clinical_batch(
            batch, method=self._method, node_dim=self._node_dim, edge_dim=self._edge_dim,
            num_tokens=self._num_tokens, num_triples=self._num_triples,
            num_relations=self._num_relations,
        )

    def set_context(self, batch) -> torch.Tensor:
        clinical_batch = self._read(batch)
        if clinical_batch.graph_count != 1:
            raise ValueError(
                "this wrapper explains one graph at a time; got "
                f"graph_count={clinical_batch.graph_count}"
            )
        self._check_extra_context(batch)
        x = clinical_batch.x
        if self._explains_tokens:
            was_training = self.adapter.training
            self.adapter.eval()
            try:
                with torch.no_grad():
                    tokens = self.adapter.embed_tokens(batch).detach().float()
            finally:
                if was_training:
                    self.adapter.train()
            x = torch.cat([x, tokens.to(x.device)], dim=-1)
        self._context = clinical_batch
        self._raw_context = batch
        self._x_shape = tuple(x.shape)
        return x.detach().clone()

    def _check_extra_context(self, batch) -> None:
        del batch

    def _split(self, x):
        """(numeric row, token vectors or None) of an explained x."""
        if self._context is None:
            raise RuntimeError("call set_context(batch) before forward()/verify()")
        if tuple(x.shape) != self._x_shape:
            raise ValueError("x must keep the shape set_context() was given "
                             f"{self._x_shape}; got {tuple(x.shape)}")
        if not self._explains_tokens:
            return x, None
        return x[:, :self._node_dim], x[:, self._node_dim:]

    def _batch_with(self, x, edge_index, *, edge_attr=None):
        """The adapter's input view: the numeric block of x plus the held context."""
        numeric, _ = self._split(x)
        ctx = self._context
        if edge_index.shape != ctx.edge_index.shape or not torch.equal(
            edge_index.long(), ctx.edge_index
        ):
            raise ValueError("edge_index must match the graph passed to set_context()")
        if edge_attr is not None and edge_attr.shape != ctx.edge_attr.shape:
            raise ValueError("edge_attr override must keep the shape set_context() was given")
        used_edge_attr = ctx.edge_attr if edge_attr is None else edge_attr
        view = types.SimpleNamespace(
            x=numeric, token=ctx.token, node_type=ctx.node_type,
            edge_index=ctx.edge_index, edge_attr=used_edge_attr,
            edge_payload=used_edge_attr[:, -PAYLOAD_WIDTH:],
            edge_relation=ctx.edge_relation,
            edge_triple=ctx.edge_triple, batch=ctx.batch_index, num_graphs=ctx.graph_count,
        )
        self._add_extra_context(view)
        return view

    def _add_extra_context(self, view) -> None:
        del view

    def _call_adapter(self, view, *, edge_mask=None, token_vectors=None):
        was_training = self.adapter.training
        self.adapter.eval()
        try:
            if not self._is_method_adapter:
                return self.adapter(view)
            kwargs = {"epoch": 0, "external_edge_mask": edge_mask}
            if token_vectors is not None:
                kwargs["token_vectors"] = token_vectors
            return self.adapter(view, **kwargs).logits
        finally:
            if was_training:
                self.adapter.train()

    def forward(self, x, edge_index, batch=None, edge_mask=None, edge_attr_override=None,
                **kwargs):
        del batch, kwargs
        view = self._batch_with(x, edge_index, edge_attr=edge_attr_override)
        _, token_vectors = self._split(x)
        return self._call_adapter(view, edge_mask=edge_mask, token_vectors=token_vectors)

    def _reference_logits(self):
        """The adapter's own eval forward on the raw batch given to set_context()."""
        raw = self._raw_context
        if not self._is_method_adapter:
            # ClinicalGNN reads `edge_payload`, which a raw graph need not carry: rebuild the
            # view from the raw batch itself, never from the held context.
            fresh = self._read(raw)
            raw = types.SimpleNamespace(
                x=fresh.x, token=fresh.token, node_type=fresh.node_type,
                edge_index=fresh.edge_index, edge_attr=fresh.edge_attr,
                edge_payload=fresh.edge_attr[:, -PAYLOAD_WIDTH:],
                edge_relation=fresh.edge_relation, edge_triple=fresh.edge_triple,
                batch=fresh.batch_index, num_graphs=fresh.graph_count)
        return self._call_adapter(raw)

    @torch.no_grad()
    def verify(self, x, edge_index, batch=None, atol: float = 1e-5):
        """(is_faithful, max_abs_diff): the wrapper (no mask) on x vs the adapter's own eval
        forward on the raw batch given to set_context()."""
        got = self.forward(x, edge_index, batch=batch)
        ref = self._reference_logits()
        return bool(torch.allclose(got, ref, atol=atol)), float((got - ref).abs().max().item())

    @torch.no_grad()
    def deletion_forward(self, data):
        """The adapter's own eval forward on one (re-encoded) graph: logits [1, C]."""
        from torch_geometric.data import Batch

        batch = Batch.from_data_list([data])
        was_training = self.adapter.training
        self.adapter.eval()
        try:
            if not self._is_method_adapter:
                return self.adapter(batch)
            return self.adapter(batch, epoch=0).logits
        finally:
            if was_training:
                self.adapter.train()

    def inert_edge_mask(self):
        """Edges the model never reads; None when it reads every edge."""
        return None

    @contextmanager
    def ig_path(self):
        """No-op: the baselines' IG path is their ordinary forward."""
        yield self

    def ig_feature_spec(self) -> dict:
        """Keyword arguments of `ig_clinical.not_recorded_baseline`: the token block of x."""
        if not self._explains_tokens:
            return {}
        return {"token_columns": [self._node_dim, self._x_width()]}

    def _x_width(self) -> int:
        if self._explains_tokens:
            return self._node_dim + int(self.adapter.token_dim)
        return self._node_dim

    def _held_fixed(self) -> str:
        return "node types, edges and edge metadata (wrapper context)"

    def ig_feature_provenance(self) -> dict:
        """How the explained row reaches the logits."""
        width = self._x_width()
        columns = {"numeric": [0, self._node_dim]}
        if self._explains_tokens:
            columns["token_vector"] = [self._node_dim, width]
        return {**self.ig_feature_spec(), "explained_columns": columns,
                "token_vectors": ("passed to the adapter as token_vectors (its own lookup at "
                                  "set_context)" if self._explains_tokens
                                  else "not explained (looked up from the held tokens)"),
                "held_fixed": self._held_fixed()}


class ClinicalGraphXAIWrapper(_ContextWrapper):
    def __init__(self, adapter, *, method: str, node_dim: int, edge_dim: int,
                 num_tokens: int, num_triples: int, num_relations: int,
                 is_method_adapter: bool = True):
        super().__init__(adapter, method=method, node_dim=node_dim, edge_dim=edge_dim,
                         num_tokens=num_tokens, num_triples=num_triples,
                         num_relations=num_relations, is_method_adapter=is_method_adapter)
        # ProtGNN explain_detail per explained class: one search per visit and class, shared by
        # the node vector and the detail. Cleared by set_context().
        self._protgnn_cache: dict = {}

    @property
    def builtin_class_specific(self) -> bool:
        return self._method in _CLASS_SPECIFIC_BUILTIN

    def set_context(self, batch) -> torch.Tensor:
        self._protgnn_cache = {}
        return super().set_context(batch)

    def _require_builtin(self):
        if self._context is None:
            raise RuntimeError("call set_context(batch) before asking for the built-in explanation")
        if self._method not in _SUPPORTS_BUILTIN_EXPLAIN:
            raise NotImplementedError(
                f"{self._method} has no per-node built-in explanation in this adapter"
            )

    # --- ProtGNN -----------------------------------------------------------------------------

    def _protgnn_predicted_class(self) -> int:
        return int(self._call_adapter(self._raw_context).argmax(-1).item())

    def _protgnn_settings(self) -> dict:
        adapter = self.adapter
        return {"rollout": int(adapter.rollout), "min_atoms": int(adapter.min_atoms),
                "max_atoms": int(adapter.max_atoms), "expand_atoms": int(adapter.expand_atoms),
                "c_puct": float(adapter.c_puct)}

    def _protgnn_settings_source(self, settings) -> str:
        from protgnn.adapter import _NATIVE_DEFAULTS

        defaults = {name: type(value)(_NATIVE_DEFAULTS[name]) for name, value in settings.items()}
        if settings == defaults:
            return f"adaptation defaults; {PROTGNN_OFFICIAL_SETTINGS}"
        return f"the run's recorded settings (not the adaptation defaults); {PROTGNN_OFFICIAL_SETTINGS}"

    def _protgnn_search(self, target_class):
        """`adapter.explain_detail` on the raw batch, with the search provenance read off the same
        search: the root coalition (`_initial_coalition`) and the number of induced subgraphs
        scored (`_induced_graph_embedding` calls). Nothing of the adapter's search is changed."""
        adapter = self.adapter
        roots, scored = [], [0]
        initial, induced = adapter._initial_coalition, adapter._induced_graph_embedding

        def recording_initial(*args, **kwargs):
            result = initial(*args, **kwargs)
            roots.append([int(node) for node in result[0]])
            return result

        def counting_induced(*args, **kwargs):
            scored[0] += 1
            return induced(*args, **kwargs)

        adapter._initial_coalition = recording_initial
        adapter._induced_graph_embedding = counting_induced
        was_training = adapter.training
        try:
            detail = dict(adapter.explain_detail(self._raw_context, target_class))
        finally:
            del adapter._initial_coalition
            del adapter._induced_graph_embedding
            if was_training:
                adapter.train()
        settings = self._protgnn_settings()
        eligible = removable_node_mask(self._context.node_type).tolist()
        subgraph_eligible = [node for node in detail["subgraph_nodes"] if eligible[node]]
        detail["search"] = {
            "algorithm": PROTGNN_SEARCH_ALGORITHM,
            "settings": settings,
            "settings_source": self._protgnn_settings_source(settings),
            "subgraph_scoring": PROTGNN_SUBGRAPH_SCORING,
            "explanation_style": PROTGNN_EXPLANATION_STYLE,
            "subgraph_size": PROTGNN_SUBGRAPH_SIZE_NOTE,
            "states_scored": scored[0],
            "root_coalition": roots[0] if roots else [],
        }
        detail["subgraph_nodes_eligible"] = subgraph_eligible
        detail["subgraph_eligible_count"] = len(subgraph_eligible)
        detail["class_specific"] = True
        return detail

    def _protgnn_detail(self, target_class):
        explained = (self._protgnn_predicted_class() if target_class is None
                     else int(target_class))
        if explained not in self._protgnn_cache:
            self._protgnn_cache[explained] = self._protgnn_search(target_class)
        return self._protgnn_cache[explained]

    # --- protocol ----------------------------------------------------------------------------

    @torch.no_grad()
    def builtin_node_importance(self, x, edge_index, batch=None, target_class=None):
        """GSAT: the adapter's own `.explain(batch)` (a fresh unmasked eval pass; class-agnostic,
        `target_class` is ignored). ProtGNN: the binary mask of the adapter's own prototype
        subgraph for the explained class (`explain_detail`), raising RuntimeError when the model
        found no connected subgraph of at least `min_atoms` nodes. Returns a list of floats."""
        del x, edge_index, batch
        self._require_builtin()
        if self._method == "protgnn":
            detail = self._protgnn_detail(target_class)
            if not detail["subgraph_nodes"]:
                raise RuntimeError(
                    "ProtGNN found no connected subgraph of at least "
                    f"{self.adapter.min_atoms} nodes in this graph")
            importance = [0.0] * int(detail["node_count"])
            for node in detail["subgraph_nodes"]:
                importance[node] = 1.0
            return importance
        return [float(value) for value in self._gsat_pass()[0]]

    def _gsat_pass(self):
        """(node attention, edge attention) of one fresh eval pass of GSAT's own `explain`."""
        node_attention = self.adapter.explain(self._raw_context).detach().clone()
        edge_attention = self.adapter.last_edge_attention.detach().clone()
        return node_attention, edge_attention

    @torch.no_grad()
    def builtin_detail(self, target_class=None):
        """ProtGNN: the cached `explain_detail` of the explained class plus the search provenance
        (`search`) and the subgraph's eligible evidence nodes. GSAT: the node and edge attention
        of a fresh pass with the tie statistics over the eligible nodes. None for a method
        without a built-in."""
        if self._method not in _SUPPORTS_BUILTIN_EXPLAIN:
            return None
        self._require_builtin()
        if self._method == "protgnn":
            return self._protgnn_detail(target_class)
        node_attention, edge_attention = self._gsat_pass()
        return {
            "kind": "gsat_node_attention",
            "class_specific": False,
            "explained_class": None,
            "node_attention": [float(value) for value in node_attention],
            "edge_attention": [float(value) for value in edge_attention],
            "edge_attention_rule": GSAT_EDGE_ATTENTION_RULE,
            "saturated_nodes": int((node_attention == 1.0).sum().item()),
            **_tie_stats(self._context.node_type, node_attention),
        }


class ClinicalGNNGraphXAIWrapper(ClinicalGraphXAIWrapper):
    """Black-box baselines: `ClinicalGNN` with conv in (edge_conditioned, hgt).

    Their layers are genuine `torch_geometric.nn.MessagePassing` modules, so
    GNNExplainer's learned mask reaches the messages through PyG's own
    `set_masks` hook with no model change. Because of that this wrapper's
    `forward` deliberately does NOT declare an `edge_mask` parameter: if it did,
    CompatibleGNNExplainer would deliver the mask a second time through the
    keyword and the mask would be applied twice per edge; it must be applied
    exactly once. An explicit mask passed as a plain keyword is installed
    through the same PyG hook for that one call only.

    Soft-mask caveat, as for every other model here: the mask multiplies each
    message after it is computed. For HGT that is after the per-receiver softmax,
    so masked-out edges do not renormalise the remaining attention weights. A soft
    mask is not the same intervention as deleting the edge.
    `hgt` also reads relation identity from `edge_triple`, not from `edge_attr`.
    """

    def __init__(self, adapter, *, method: str, node_dim: int, edge_dim: int,
                 num_tokens: int, num_triples: int, num_relations: int):
        super().__init__(adapter, method=method, node_dim=node_dim, edge_dim=edge_dim,
                         num_tokens=num_tokens, num_triples=num_triples,
                         num_relations=num_relations, is_method_adapter=False)

    def forward(self, x, edge_index, batch=None, edge_attr_override=None, **kwargs):
        del batch
        removal_mask = kwargs.pop("edge_mask", None)
        view = self._batch_with(x, edge_index, edge_attr=edge_attr_override)
        installed = removal_mask is not None
        if installed:
            set_keyword_mask(self.adapter, removal_mask, edge_index.long())
        try:
            return self._call_adapter(view)
        finally:
            if installed:
                clear_masks(self.adapter)


class GraphCareGraphXAIWrapper(_ContextWrapper):
    builtin_class_specific = False

    def __init__(self, adapter, *, node_dim: int, edge_dim: int, num_tokens: int,
                 num_triples: int, num_relations: int):
        super().__init__(adapter, method="graphcare", node_dim=node_dim, edge_dim=edge_dim,
                         num_tokens=num_tokens, num_triples=num_triples,
                         num_relations=num_relations, is_method_adapter=True)

    def _read(self, batch) -> ClinicalBatch:
        return read_clinical_batch(
            batch, method="GraphCare", node_dim=self._node_dim, edge_dim=self._edge_dim,
            num_tokens=self._num_tokens, num_triples=self._num_triples,
            num_relations=self._num_relations,
        )

    def _check_extra_context(self, batch) -> None:
        for name in ("visit_membership_index", "num_visits", "global_node_mask"):
            if getattr(batch, name, None) is None:
                raise ValueError(f"GraphCare requires {name}")

    def _add_extra_context(self, view) -> None:
        raw = self._raw_context
        view.visit_membership_index = raw.visit_membership_index
        view.num_visits = raw.num_visits
        view.global_node_mask = raw.global_node_mask

    def _held_fixed(self) -> str:
        return ("node types, edges, edge metadata and the visit context (membership pairs, "
                "visit count, global-node mask) (wrapper context)")

    def inert_edge_mask(self):
        """Edges with a global endpoint: GraphCare's message pass drops them, so the model never
        reads them."""
        if self._context is None:
            raise RuntimeError("call set_context(batch) before inert_edge_mask()")
        global_mask = self._raw_context.global_node_mask.to(torch.bool).view(-1)
        source, target = self._context.edge_index
        return global_mask[source] | global_mask[target]

    def _attention_pass(self):
        """Node attention of the last layer from a fresh unmasked eval forward on the raw batch:
        node_attention[n] = sum over membership pairs (v, n) of last_visit_weight (alpha * beta),
        replaced by the float64 mean over the nodes with the same visit set; a global node has no
        membership pair and gets 0. Returns (attention, beta, max_within_visit_set_spread)."""
        if self._context is None:
            raise RuntimeError("call set_context(batch) before asking for the built-in explanation")
        self._call_adapter(self._raw_context)
        adapter = self.adapter
        visit_weight = adapter.last_visit_weight.detach().clone()
        beta = adapter.last_beta.detach().clone()
        node_count = self._context.x.size(0)
        membership = self._raw_context.visit_membership_index.to(torch.long)
        attention = torch.zeros(node_count, dtype=visit_weight.dtype, device=visit_weight.device)
        if membership.numel():
            attention.index_add_(0, membership[1].to(visit_weight.device), visit_weight)
        attention, spread = _per_visit_set_mean(attention, membership)
        return attention, beta, spread

    @torch.no_grad()
    def builtin_node_importance(self, x, edge_index, batch=None, target_class=None):
        """GraphCare's own node attention of the last layer (`_attention_pass`), as a list of
        floats. Class-agnostic: `target_class` is ignored."""
        del x, edge_index, batch, target_class
        attention, _, _ = self._attention_pass()
        return [float(value) for value in attention]

    @torch.no_grad()
    def builtin_detail(self, target_class=None):
        """The visit attention behind the built-in: beta per visit, the visit count, the largest
        float32 spread inside one visit set before the per-set mean, and the tie statistics over
        the eligible nodes."""
        del target_class
        attention, beta, spread = self._attention_pass()
        return {
            "kind": "graphcare_visit_attention",
            "layer": "last",
            "num_layers": len(self.adapter.layers),
            "class_specific": False,
            "explained_class": None,
            "num_visits": int(self._raw_context.num_visits.view(-1)[0].item()),
            "beta": [float(value) for value in beta],
            "node_attention": [float(value) for value in attention],
            "value_depends_on": GRAPHCARE_VALUE_DEPENDENCE,
            "max_within_visit_set_spread": spread,
            **_tie_stats(self._context.node_type, attention),
        }
