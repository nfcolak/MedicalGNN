"""CEI-GNN (v3 and mp2) under the clinical_graph_v2 explanation runner.

Built on the CEI study's own explanation code (cei/studies/cei_v3_graphxai.py); nothing of it is
reimplemented:

- the model-facing network is the study's `CEIV3GraphXAIWrapper` (a private copy of the network,
  edge identity checked on every call), with its cached, constant PLE term dropped: the network's
  own `_ple_term` runs on the features of every forward;
- the built-in explanation is the study's `native_importance`, the exact additive accounting of
  the logit (node + half of each incident edge/pair contribution), with bias and absence
  evidence exported separately rather than pushed onto nodes. For `cei_gnn_mp2` it reads
  every block of the network's evidence ledger (`ledger_keys()`), the neighbourhood rows
  included.

What this module adds is the runner's wrapper protocol, the same for every final method:
`set_context(batch) -> x`, `forward(x, edge_index, batch=None, edge_attr_override=None)`,
`verify`, `ig_feature_spec`, `ig_feature_provenance`, `ig_path`, `builtin_node_importance`,
`builtin_detail`, `builtin_class_specific`, `inert_edge_mask`, `deletion_forward`, plus
`build_cei_model(binding, state_dict)`.

Explained input. x = [numeric row (node_dim) | token vector (token_dim)], as for the three
baselines. The node-type vectors, the edge metadata, the membership pairs and the absence inputs
are held fixed in the wrapper context; `forward` appends the type vectors before the network's
own continuous forward, so x * s for any s is the network on [numeric * s | token * s | type].

PLE (piecewise-linear encoding) term. Every forward, gradient and GNNExplainer step uses the
network's own hard-gated term (`ple_basis` switches a row on at `has_value > 0`), recomputed from
the explained features. Only inside `wrapper.ig_path()` is the gate made linear
(`_smooth_ple_term`, identical at has_value 0 and 1): the IG path from the not-recorded baseline
would otherwise jump at its first step and IG completeness would fail. Each source records the
term it ran on (`ple_term`).

GNNExplainer reaches CEI through the edge aggregator it already routes through PyG's mask
state (read in cei_gnn_v3._evidence_blocks; mp2's network also hands that same mask to its
message-passing stack as `edge_weight`, cei/layers.py). `forward` therefore declares no
`edge_mask` parameter: the explainer has exactly one delivery path. A mask passed as the
`edge_mask` keyword (one call) goes through the same hook. An edge mask cannot reach the
within-visit pair rows, the absence block or the node-local rows (`EDGE_MASK_SCOPES`).

Fidelity removal does not go through x or an edge mask: `deletion_forward(data)` is the
adapter's own forward on a graph re-encoded without the removed nodes.
"""
from __future__ import annotations

from argparse import Namespace
from contextlib import contextmanager
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks

from core.registry import build_method

CEI_METHODS = ("cei_gnn_v3", "cei_gnn_mp2")

PLE_TERM_EXACT = ("the network's own hard-gated PLE term (has_value > 0), recomputed from the "
                  "explained features")
PLE_TERM_IG = ("inside IG only: the PLE basis computed with has_value treated as 1 and multiplied "
               "by the explained has_value (equal to the network's term at has_value 0 and 1, "
               "linear in between)")

HELD_FIXED = ("node-type vectors, edge metadata, within-visit membership pairs and absence "
              "inputs (wrapper context)")

_V3_EDGE_MASK_SCOPE = (
    "an edge mask scales the edge's evidence rows once (0 = edge deleted from them); "
    "within-visit pair contributions, the absence block and the node rows are not reachable "
    "by an edge mask.")
EDGE_MASK_SCOPES = {
    "cei_gnn_v3": _V3_EDGE_MASK_SCOPE,
    "cei_gnn_mp2": (
        "an edge mask (0) deletes the edge from the edge rows and from every message-passing "
        "layer (weighted means, relation groups, hub max, mp_dedup copy split and mirror rule); "
        "a fractional mask scales the edge once, and under a weighted mean a common scale of a "
        "receiver's incoming masks is partly invisible. Within-visit pair contributions, the "
        "absence block and the node-local rows are not reachable by an edge mask; a typed "
        "reverse edge of a bidirectional view is a separate edge."),
}


def _check_method(method) -> str:
    if method not in CEI_METHODS:
        raise ValueError(f"CEI explanation supports {list(CEI_METHODS)}; got method {method!r}")
    return method


def _adapter_method(adapter) -> str:
    return "cei_gnn_mp2" if hasattr(adapter, "mp2_options") else "cei_gnn_v3"


def build_cei_model(binding: dict, state_dict: dict):
    """CEI-GNN v3 / mp2 through the shared registry from the run's own recorded options (arm, k,
    v3_state file, mp2 options, ...), then proven to be the recorded configuration: the rebuilt
    adapter's run_config() must equal binding.json's method_config exactly.

    The adapter is built with the recorded dropout rate (part of the recorded configuration)
    and returned in eval mode, where every dropout site is off. The checkpoint loads strictly.
    """
    method = _check_method(binding.get("method"))
    config = binding["method_config"]
    if config.get("method") != method:
        raise ValueError(f"binding.json names method {method!r} but its method_config records "
                         f"{config.get('method')!r}; refusing to explain a mismatched run")
    arch = config["architecture"]
    model = build_method(
        method, num_tokens=binding["vocabulary_size"], node_dim=binding["node_dim"],
        edge_dim=binding["edge_dim"], num_classes=binding["num_classes"],
        hidden=binding["hidden"], layers=binding["layers"], dropout=float(arch.get("dropout", 0.0)),
        token_dim=int(arch["token_dim"]), num_triples=binding["num_meta_relations"],
        args=Namespace(edge_direction=binding["edge_direction"],
                       method_options=dict(config["effective_settings"]),
                       seed=binding.get("seed"), top_k_labels=binding.get("top_k_labels"),
                       weights=binding.get("weights")),
    )
    model.load_state_dict(state_dict)
    if _comparable_run_config(model.run_config(), config) != config:
        raise ValueError(
            "rebuilt CEI-GNN's configuration differs from binding.json's method_config "
            "(check that the recorded v3_state file is the one on disk); refusing to "
            "explain a model that cannot be shown to be the one that was trained")
    return model.eval()


# The sections of an mp2 run_config() that list option values by name.
_OPTION_SECTIONS = ("architecture", "effective_settings", "mp2_options")


def _comparable_run_config(rebuilt: dict, recorded: dict) -> dict:
    """The rebuilt run_config() without options the recorded run predates.

    Options added to cei_gnn_mp2 after a run was trained (e.g. the section-4 ablation switches)
    are missing from that run's binding.json. Such an option is dropped from the comparison only
    when the rebuilt model holds it at its documented default, which is the behaviour the run was
    trained with (cei/options.py: each later option defaults to the earlier behaviour); any other
    value, and every option the binding does record, is still compared exactly.
    """
    if rebuilt.get("method") != "cei_gnn_mp2":
        return rebuilt
    from cei.options import DEFAULTS

    comparable = dict(rebuilt)
    for section in _OPTION_SECTIONS:
        new, old = rebuilt.get(section), recorded.get(section)
        if not isinstance(new, dict) or not isinstance(old, dict):
            continue
        comparable[section] = {
            key: value for key, value in new.items()
            if key in old or key not in DEFAULTS or value != DEFAULTS[key]
        }
    return comparable


def _smooth_ple_term(network, features, metadata):
    """The network's PLE term with its hard has_value gate made continuous (IG path only).

    `ple_basis` switches a node's row on at `has_value > 0`, so along an IG path that starts at
    has_value = 0 the logit jumps and IG misses the jump. Here the basis is computed as if the
    value were recorded (has_value treated as 1 for every row with an active knot table, from
    the explained value column) and each row is then multiplied by the explained has_value
    column. At has_value in {0, 1} this is exactly the network's own term; in between it is
    linear in has_value. A non-finite value on a row whose has_value is 0 is read as 0 (the hard
    gate never looked at it); with has_value > 0 it still raises, as in `ple_basis`."""
    from cei.studies.cei_v3_ple import ple_basis

    values = features[:, network.scaled_value_column]
    gate = features[:, network.has_value_column]
    values = torch.where(torch.isfinite(values) | (gate > 0), values, torch.zeros_like(values))
    knot_row = network.knot_row_of_token[metadata.token]
    if getattr(network, "ple_mode", "hat") == "robust_cum":
        basis = ple_basis(values, torch.ones_like(gate), knot_row, network.robust_knots,
                          network.robust_knot_active) * gate[:, None]
        return F.linear(basis[:, 1:], network.ple_projection.weight.cumsum(dim=1))
    basis = ple_basis(values, torch.ones_like(gate), knot_row, network.knots,
                      network.knot_active) * gate[:, None]
    return network.ple_projection(basis)


def _install_ple(network, *, smooth: bool) -> None:
    """The network's own `_ple_term` (smooth=False) or `_smooth_ple_term` (smooth=True) on the
    private copy. The study wrapper's cached constant is dropped either way."""
    if "_graphxai_fixed_ple" in network._buffers:
        del network._buffers["_graphxai_fixed_ple"]
    network.__dict__.pop("_ple_term", None)          # instance override -> the class's own term
    if smooth:
        network._ple_term = MethodType(_smooth_ple_term, network)
    if hasattr(network, "_graphxai_fixed_ple"):
        raise RuntimeError("the fixed PLE term could not be removed from the explained network")
    own = type(network)._ple_term
    if not smooth and getattr(network._ple_term, "__func__", None) is not own:
        raise RuntimeError("the explained network does not run its own PLE term")


class CEIGraphXAIWrapper(nn.Module):
    builtin_class_specific = True

    def __init__(self, adapter, *, method: str):
        super().__init__()
        _check_method(method)
        if _adapter_method(adapter) != method:
            raise ValueError(f"method {method!r} does not match the adapter "
                             f"({_adapter_method(adapter)!r})")
        # Held in a plain list so the adapter's own network is NOT registered as a
        # submodule: GNNExplainer must install its mask only on the private copy inside
        # the wrapper that actually runs the forward pass.
        self._holder = [adapter]
        self._method = method
        self.edge_mask_scope = EDGE_MASK_SCOPES[method]
        self.inner = None
        self._raw_context = None
        self._features = None
        self._type_vectors = None
        self._smooth = False
        self._native = None

    @property
    def adapter(self):
        return self._holder[0]

    @property
    def method(self) -> str:
        return self._method

    @property
    def ple_term(self) -> str:
        """The PLE term the next forward runs on."""
        return PLE_TERM_IG if self._smooth else PLE_TERM_EXACT

    def _widths(self):
        adapter = self.adapter
        return int(adapter.node_dim), int(adapter.token_dim), int(adapter.hidden)

    def set_context(self, batch) -> torch.Tensor:
        from cei.studies.cei_v3_graphxai import CEIV3GraphXAIWrapper

        adapter = self.adapter.eval()
        clinical, _, _ = adapter._read(batch)
        if clinical.graph_count != 1:
            raise ValueError(
                f"this wrapper explains one graph at a time; got graph_count={clinical.graph_count}")
        self.inner = CEIV3GraphXAIWrapper(adapter, batch).eval()
        self._smooth = False
        _install_ple(self.inner.network, smooth=False)
        self._raw_context = batch
        self._native = None
        node_dim, token_dim, hidden = self._widths()
        with torch.no_grad():
            features = adapter.continuous_inputs(batch).detach().clone()
        if features.size(1) != node_dim + token_dim + hidden:
            raise ValueError(f"CEI continuous row has {features.size(1)} columns, expected "
                             f"{node_dim} + {token_dim} + {hidden}")
        self._features = features[:, :node_dim + token_dim].clone()
        self._type_vectors = features[:, node_dim + token_dim:].clone()
        return self._features.clone()

    @contextmanager
    def ig_path(self):
        """Inside this block the PLE has_value gate is linear (`_smooth_ple_term`); outside it the
        network's own hard-gated term runs. Nested use keeps the outer mode on exit."""
        if self.inner is None:
            raise RuntimeError("call set_context(batch) before ig_path()")
        network, previous = self.inner.network, self._smooth
        _install_ple(network, smooth=True)
        self._smooth = True
        try:
            yield self
        finally:
            _install_ple(network, smooth=previous)
            self._smooth = previous

    def forward(self, x, edge_index, batch=None, edge_attr_override=None, **kwargs):
        if self.inner is None:
            raise RuntimeError("call set_context(batch) before forward()/verify()")
        removal_mask = kwargs.pop("edge_mask", None)
        if kwargs:
            raise TypeError(f"unexpected keyword arguments {sorted(kwargs)}")
        if edge_attr_override is not None:
            raise NotImplementedError(
                "the CEI wrapper holds the edge metadata fixed in its context; an edge-attribute "
                "override (relation-vs-payload ablation) is not applicable")
        node_dim, token_dim, _ = self._widths()
        if x.ndim != 2 or x.size(1) != node_dim + token_dim:
            raise ValueError(f"x must be [numeric ({node_dim}) | token vector ({token_dim})]; "
                             f"got shape {tuple(x.shape)}")
        features = torch.cat((x, self._type_vectors.to(device=x.device, dtype=x.dtype)), dim=-1)
        installed = removal_mask is not None
        if installed:
            set_masks(self.inner, removal_mask, edge_index.long(), apply_sigmoid=False)
        try:
            return self.inner(features, edge_index, batch)
        finally:
            if installed:
                clear_masks(self.inner)

    @torch.no_grad()
    def verify(self, x, edge_index, batch=None, atol: float = 1e-5):
        """(is_faithful, max_abs_diff): this wrapper (no mask) vs the adapter's own forward on
        the raw batch."""
        got = self.forward(x, edge_index, batch=batch)
        ref = self.adapter(self._raw_context, epoch=0).logits
        return bool(torch.allclose(got, ref, atol=atol)), float((got - ref).abs().max().item())

    @torch.no_grad()
    def deletion_forward(self, data):
        """The adapter's own eval forward on one (re-encoded) graph: logits [1, C]."""
        from torch_geometric.data import Batch

        return self.adapter.eval()(Batch.from_data_list([data]), epoch=0).logits

    def inert_edge_mask(self):
        """Every CEI edge is read by the model."""
        return None

    @torch.no_grad()
    def builtin_node_importance(self, x, edge_index, batch=None, target_class=None):
        """The study's exact additive accounting for the explained class, as |net contribution|
        per node (list of floats). Raises NotImplementedError (a RuntimeError) when the
        configuration has no additive ledger."""
        from cei.studies.cei_v3_graphxai import native_importance

        del x, edge_index, batch
        if self.inner is None:
            raise RuntimeError("call set_context(batch) before builtin_node_importance()")
        network = self.adapter.network
        if self._method == "cei_gnn_mp2" and not network.ledger_additive:
            raise NotImplementedError(
                f"cei_gnn_mp2 with ev_combine={network.ev_combine!r}, readout={network.readout!r} "
                "has no additive evidence ledger, so it has no exact per-node built-in explanation")
        target = (int(target_class) if target_class is not None
                  else int(self.adapter(self._raw_context, epoch=0).logits.argmax(-1)))
        importance, provenance = native_importance(self.adapter, self._raw_context, target)
        if importance is None:
            raise NotImplementedError(
                f"{self._method} has no per-node built-in explanation in this adapter")
        self._native = dict(provenance, edge_mask_scope=self.edge_mask_scope,
                            explained_class=target, ple_term=PLE_TERM_EXACT)
        return [float(value) for value in importance.detach()]

    def builtin_detail(self, target_class=None):
        """Provenance of the built-in explanation (accounting check, bias, absence, shares).
        With `target_class`, the detail of that class's explanation (computed if the last one
        explained another class)."""
        if target_class is not None and (self._native is None
                                         or self._native["explained_class"] != int(target_class)):
            self.builtin_node_importance(None, None, target_class=target_class)
        return self._native

    def ig_feature_spec(self) -> dict:
        """Keyword arguments of `ig_clinical.not_recorded_baseline`: the token block of x."""
        node_dim, token_dim, _ = self._widths()
        return {"token_columns": [node_dim, node_dim + token_dim]}

    def ig_feature_provenance(self) -> dict:
        """How the explained row reaches the logits on the IG path. The baseline zeroes the value
        and has_value columns, so a node without a recorded value has a zero basis row and its
        PLE term is 0 (the projection has no bias); its token block is zero (the pad
        embedding)."""
        node_dim, token_dim, _ = self._widths()
        return {"ple_term": PLE_TERM_IG, **self.ig_feature_spec(),
                "explained_columns": {"numeric": [0, node_dim],
                                      "token_vector": [node_dim, node_dim + token_dim]},
                "held_fixed": HELD_FIXED}
