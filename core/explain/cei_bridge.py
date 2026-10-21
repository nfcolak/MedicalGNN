"""CEI-GNN (v3 and mp2) under the clinical_graph_v2 explanation runner.

Built ON TOP of Necati's own explanation code (cei/studies/cei_v3_graphxai.py); nothing of his is
reimplemented:

- the model-facing wrapper is his `CEIV3GraphXAIWrapper` (a private copy of the network, edge
  identity checked on every call), with one change: his cached, constant PLE term is dropped
  (`_recompute_ple`), so the network's own `_ple_term` runs on the features passed to `forward`;
- the built-in explanation is his `native_importance`, the exact additive accounting of the
  logit (node + half of each incident edge/pair contribution), with bias and absence
  evidence exported separately rather than pushed onto nodes. For `cei_gnn_mp2` it reads
  every block of the network's evidence ledger (`ledger_keys()`), the neighbourhood rows
  included.

What this module adds is only the runner's interface: `build_cei_model(binding, state_dict)`
and the wrapper's `set_context(batch)`, `forward(x, edge_index, batch=None, ...)`, `verify`,
`builtin_node_importance`, `builtin_detail`, `ig_feature_spec`, `intervention_scope`.

GNNExplainer reaches CEI through the edge aggregator it already routes through PyG's mask
state (read in cei_gnn_v3._evidence_blocks; mp2's network also hands that same mask to its
message-passing stack as `edge_weight`, cei/layers.py). Like `ClinicalGNNGraphXAIWrapper`,
`forward` therefore declares no `edge_mask` parameter: the explainer has exactly one delivery
path, and a removal mask from the fidelity code goes through the same hook for that one call.

The PLE (piecewise-linear encoding) term of the node encoder is a function of the explained
row's value and has_value columns (the knot row comes from the fixed tokens). It is recomputed
from the explained features, so gradients, IG and the removal interventions reach the value
column through it. His study cached the term at the original features: nothing in the network
forces that (`ple_basis` is built from clamp / gather / scatter and is differentiable in the
value), it was a choice that kept the PLE input out of the explained quantity. The hard
`has_value > 0` gate of `ple_basis` is replaced by a linear one (`_smooth_ple_term`), identical at
has_value 0 and 1: otherwise the IG path from the not-recorded baseline jumps at its first step
and IG completeness fails.

Departures from the contract the other models get, recorded rather than hidden:

- The explained input is CEI's continuous row (numeric features | token vectors |
  node-type vectors). Zeroing a row therefore also zeroes its token and type vectors, so
  `token_override` is ignored (there is no separate token to replace).
- Removal masks the node's incident EDGES, but CEI's within-visit PAIR contributions and
  its absence block keep seeing the node (the absence inputs are fixed, as in his study).
  Fidelity for CEI is therefore a partial removal and not strictly comparable with the
  other models'; see `intervention_scope`.
- The relation-vs-payload ablation cannot be applied (its edge metadata is fixed inside
  the bound wrapper); the runner records that as not applicable.
"""
from __future__ import annotations

from argparse import Namespace
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks

from core.registry import build_method

CEI_METHODS = ("cei_gnn_v3", "cei_gnn_mp2")

PLE_TERM = ("recomputed from the explained features (differentiable; has_value gate "
            "interpolated linearly, identical at 0 and 1)")

INTERVENTION_SCOPE = (
    "continuous node row zeroed (numeric + token + type vectors; the PLE term is recomputed "
    "from the zeroed row) and incident edges masked; "
    "within-visit pair contributions and the absence block still see the node. Partial "
    "removal: not strictly comparable with the other models' fidelity."
)
INTERVENTION_SCOPES = {
    "cei_gnn_v3": INTERVENTION_SCOPE,
    "cei_gnn_mp2": (
        "continuous node row zeroed (numeric + token + type vectors; the PLE term is recomputed "
        "from the zeroed row) and incident edges masked. A masked edge (0) is deleted from the edge "
        "rows and from every message-passing layer (weighted means, relation groups, hub max, "
        "mp_dedup copy split and mirror rule); a fractional mask scales the edge once. The node's "
        "own node and neighbourhood rows remain (its zeroed row still passes through the encoder), "
        "within-visit pair contributions and the absence block still see the node, and a typed "
        "reverse edge of a bidirectional view is a separate edge. Partial removal: not strictly "
        "comparable with the other models' fidelity."
    ),
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
    """The network's PLE term with its hard has_value gate made continuous.

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


def _recompute_ple(inner) -> None:
    """Undo the frozen PLE of his wrapper: drop the cached constant and install the continuous
    PLE term, which runs on the features of every forward."""
    network = inner.network
    if "_graphxai_fixed_ple" in network._buffers:
        del network._buffers["_graphxai_fixed_ple"]
    network._ple_term = MethodType(_smooth_ple_term, network)
    if hasattr(network, "_graphxai_fixed_ple"):
        raise RuntimeError("the fixed PLE term could not be removed from the explained network")


class CEIGraphXAIWrapper(nn.Module):
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
        self.intervention_scope = INTERVENTION_SCOPES[method]
        self.inner = None
        self._raw_context = None
        self._features = None
        self._native = None

    @property
    def adapter(self):
        return self._holder[0]

    def set_context(self, batch) -> torch.Tensor:
        from cei.studies.cei_v3_graphxai import CEIV3GraphXAIWrapper

        adapter = self.adapter.eval()
        clinical, _, _ = adapter._read(batch)
        if clinical.graph_count != 1:
            raise ValueError(
                f"this wrapper explains one graph at a time; got graph_count={clinical.graph_count}")
        self.inner = CEIV3GraphXAIWrapper(adapter, batch).eval()
        _recompute_ple(self.inner)
        self._raw_context = batch
        self._native = None
        with torch.no_grad():
            self._features = adapter.continuous_inputs(batch).detach().clone()
        return self._features.clone()

    def forward(self, x, edge_index, batch=None, token_override=None, edge_attr_override=None,
                **kwargs):
        if self.inner is None:
            raise RuntimeError("call set_context(batch) before forward()/verify()")
        removal_mask = kwargs.pop("edge_mask", None)
        installed = removal_mask is not None
        if installed:
            set_masks(self.inner, removal_mask, edge_index.long(), apply_sigmoid=False)
        try:
            return self.inner(x, edge_index, batch)
        finally:
            if installed:
                clear_masks(self.inner)

    @torch.no_grad()
    def verify(self, x, edge_index, batch=None, atol: float = 1e-5):
        """(is_faithful, max_abs_diff): this wrapper (no mask) vs the adapter's own forward."""
        got = self.forward(x, edge_index, batch=batch)
        ref = self.adapter(self._raw_context, epoch=0).logits
        return bool(torch.allclose(got, ref, atol=atol)), float((got - ref).abs().max().item())

    @torch.no_grad()
    def builtin_node_importance(self, x, edge_index, batch=None, target_class=None):
        """His exact additive accounting for the explained class, as |net contribution|."""
        from cei.studies.cei_v3_graphxai import native_importance

        del x, batch
        network = self.adapter.network
        if self._method == "cei_gnn_mp2" and not network.ledger_additive:
            raise NotImplementedError(
                f"cei_gnn_mp2 with ev_combine={network.ev_combine!r}, readout={network.readout!r} "
                "has no additive evidence ledger, so it has no exact per-node built-in explanation")
        target = (int(target_class) if target_class is not None
                  else int(self.forward(self._features, edge_index).argmax(-1)))
        importance, provenance = native_importance(self.adapter, self._raw_context, target)
        if importance is None:
            raise NotImplementedError(
                f"{self._method} has no per-node built-in explanation in this adapter")
        provenance = dict(provenance, intervention_scope=self.intervention_scope,
                          explained_class=target, ple_term=PLE_TERM, ple_inputs=PLE_TERM)
        self._native = provenance
        return importance.detach()

    @torch.no_grad()
    def builtin_detail(self):
        """Provenance of the last built-in explanation (accounting check, bias, absence)."""
        return self._native

    def ig_feature_spec(self) -> dict:
        """Widths of CEI's composite explained row, for the IG 'not recorded' baseline.

        Only the widths: the runner passes this dict on as keyword arguments of
        `ig_clinical.not_recorded_baseline`. The PLE provenance is `ig_feature_provenance()`."""
        adapter = self.adapter
        return {"numeric_width": int(adapter.node_dim), "zeroed_tail": int(adapter.token_dim),
                "kept_tail": int(adapter.hidden)}

    def ig_feature_provenance(self) -> dict:
        """How the explained row reaches the logits through the PLE term. The baseline zeroes
        the value and has_value columns, so a node without a recorded value has a zero basis row
        and its PLE term is 0 (the projection has no bias)."""
        return {"ple_term": PLE_TERM, **self.ig_feature_spec()}
