"""Integrated Gradients with a clinical baseline.

The vendored IG (external/GraphXAI-main/.../integrated_grad.py:142) uses an all-zero
baseline, with the authors' own "TODO: baseline all 0s, all 1s, ...?". For these
node features that is not a patient: a zero row has no node kind, no value and no
"was measured" flag, so the integration path runs through states that never occur.

The explained row x is the same for every final model: [numeric row | token vector]
(the wrapper's `ig_feature_spec()` names the token columns). The node type, the edges
and the edge metadata stay fixed in the wrapper context.

Baselines here (pre-registered; recorded in every record's provenance):

- ``not_recorded`` (primary). Each node keeps its own kind (what the node *is*) and
  loses everything recorded about it: value, time and availability scaled to 0 with
  their presence flags at 0, context numbers at 0, and the patient's categorical
  context at its explicit ``<missing>`` code. "Value 0 with has_value 0" is the data's
  own encoding of a missing value, so this is a state the encoder was built to see.
  The token block is zero, which is the pad embedding (``padding_idx=0``): the node's
  code identity is part of what IG attributes. IG then answers: "what did *recording*
  this node, value and code, contribute?".
- ``real_patient_refs`` (robustness check, expected-gradients style). Per node, the
  baseline row is drawn from real training nodes of the same token and kind; the
  attributions of several draws are averaged. Costs one IG pass per draw.

Completeness: IG attributions must sum to f(x) - f(baseline) (here f is the explained
class's logit). `integrated_gradients_until_complete` raises the step count per graph
until the relative error is within the tolerance or the largest candidate is reached.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from core import NODE_KINDS

PATIENT_KIND_ID = NODE_KINDS.index("patient")
DEFAULT_STEP_CANDIDATES = (8, 16, 32, 64, 128, 256)
DEFAULT_TOLERANCE = 0.05
# |f(x) - f(baseline)| below this is "no change to explain": relative error is
# measured against this floor so a near-zero denominator cannot fake a failure.
DELTA_FLOOR = 1e-3
NODE_REDUCTION = "sum_abs_over_columns"


def not_recorded_baseline(x: torch.Tensor, node_type: torch.Tensor, layout: Sequence[str],
                          *, token_columns: Sequence[int] | None = None) -> torch.Tensor:
    """The primary baseline: every node keeps its kind, loses what was recorded.

    `layout` names the numeric columns. With `token_columns` = [start, stop] (the wrapper's
    `ig_feature_spec()`), x is [numeric | token vector]: the numeric block must end at
    `start`, the token block at the last column, and the token block of the baseline is zero
    (the pad embedding). Without it, x is the numeric row alone."""
    if token_columns is None:
        start = stop = x.size(1)
    else:
        start, stop = (int(column) for column in token_columns)
    if len(layout) != start or not start <= stop == x.size(1):
        raise ValueError(f"layout has {len(layout)} names and token columns {token_columns}, "
                         f"but x has {x.size(1)} columns")
    baseline = torch.zeros_like(x)
    patient = node_type.long() == PATIENT_KIND_ID
    for column, name in enumerate(layout):
        if name in NODE_KINDS:
            baseline[:, column] = x[:, column]
        elif name.endswith(":<missing>"):
            baseline[patient, column] = 1.0
    return baseline


class ReferenceBank:
    """Real training node rows, keyed by (token, node kind)."""

    def __init__(self, rows: dict[tuple[int, int], torch.Tensor]):
        self.rows = rows

    @classmethod
    def from_graphs(cls, graphs, *, per_key_cap: int = 100) -> "ReferenceBank":
        collected: dict[tuple[int, int], list[torch.Tensor]] = {}
        for graph in graphs:
            for row, token, kind in zip(graph.x, graph.token.tolist(), graph.node_type.tolist()):
                bucket = collected.setdefault((int(token), int(kind)), [])
                if len(bucket) < per_key_cap:
                    bucket.append(row.detach().clone())
        return cls({key: torch.stack(rows) for key, rows in collected.items()})

    def draw(self, token: torch.Tensor, node_type: torch.Tensor, fallback: torch.Tensor,
             rng: np.random.Generator, *, fill: torch.Tensor | None = None) -> torch.Tensor:
        """One baseline: a real row for each node whose key has one, else `fallback`.

        The bank holds numeric rows; when the explained row is wider (a token block follows),
        a drawn node's remaining columns come from `fill` (the explained x: the same token has
        the same token vector)."""
        baseline = fallback.clone()
        for i, (t, k) in enumerate(zip(token.tolist(), node_type.tolist())):
            rows = self.rows.get((int(t), int(k)))
            if rows is not None:
                width = rows.size(1)
                baseline[i, :width] = rows[int(rng.integers(rows.size(0)))]
                if width < baseline.size(1):
                    if fill is None:
                        raise ValueError("bank rows are narrower than the baseline; pass fill")
                    baseline[i, width:] = fill[i, width:]
        return baseline


def integrated_gradients(wrapper, x: torch.Tensor, edge_index: torch.Tensor,
                         baseline: torch.Tensor, target_class: int, steps: int):
    """IG of the target logit w.r.t. x along the straight path from `baseline`.

    Trapezoid rule over steps+1 points; gradients are accumulated as a running sum
    (one copy, not steps+1). Returns (attribution [N, F], completeness). Every column of
    x is interpolated (numeric row and token vector); what the wrapper holds in its
    context (node type, edges, edge metadata) stays fixed.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    delta = x - baseline
    total = torch.zeros_like(x)
    for i in range(steps + 1):
        point = (baseline + (i / steps) * delta).detach().requires_grad_(True)
        logit = wrapper(point, edge_index)[0, target_class]
        (grad,) = torch.autograd.grad(logit, point)
        total += grad * (0.5 if i in (0, steps) else 1.0)
    attribution = delta * total / steps
    with torch.no_grad():
        f_x = float(wrapper(x, edge_index)[0, target_class])
        f_base = float(wrapper(baseline, edge_index)[0, target_class])
    change = f_x - f_base
    error = abs(float(attribution.sum()) - change)
    return attribution.detach(), {
        "target_logit_change": change,
        "attribution_sum": float(attribution.sum()),
        "completeness_error_abs": error,
        "completeness_error_rel": error / max(abs(change), DELTA_FLOOR),
    }


def integrated_gradients_until_complete(wrapper, x: torch.Tensor, edge_index: torch.Tensor,
                                        baseline: torch.Tensor, target_class: int, *,
                                        start_steps: int = DEFAULT_STEP_CANDIDATES[0],
                                        candidates: Sequence[int] = DEFAULT_STEP_CANDIDATES,
                                        tolerance: float = DEFAULT_TOLERANCE):
    """IG with the step count raised per graph until completeness holds.

    Tries the candidates from `start_steps` upward (it must be one of them) and stops at the
    first whose relative completeness error is within `tolerance`; if none is, the largest
    candidate's attribution is returned with `completeness_met` False. Only the IG identity
    f(x) - f(baseline) = sum of attributions decides, never a label. Returns (attribution,
    {"steps_used", "completeness_error_rel", "completeness_met", "tried", ...completeness})
    with `tried` = [{"steps", "completeness_error_rel"}, ...] in the order run."""
    ordered = sorted({int(steps) for steps in candidates})
    if int(start_steps) not in ordered:
        raise ValueError(f"start_steps {start_steps} is not one of the candidates {ordered}")
    tried = []
    for steps in ordered[ordered.index(int(start_steps)):]:
        attribution, completeness = integrated_gradients(
            wrapper, x, edge_index, baseline, target_class, steps)
        tried.append({"steps": steps,
                      "completeness_error_rel": completeness["completeness_error_rel"]})
        if completeness["completeness_error_rel"] <= tolerance:
            break
    return attribution, dict(completeness, steps_used=steps, tolerance=tolerance,
                             completeness_met=completeness["completeness_error_rel"] <= tolerance,
                             tried=tried)


def node_importance(attribution: torch.Tensor) -> torch.Tensor:
    """Sum of |attribution| over every column of x per node (NODE_REDUCTION). The vendored
    explainers' own default is the signed sum; every source here uses this absolute sum."""
    return attribution.abs().sum(dim=1)


def choose_steps(errors_by_steps: dict[int, Sequence[float]],
                 tolerance: float = DEFAULT_TOLERANCE) -> dict:
    """Smallest candidate whose WORST relative error over the pilot graphs is within
    `tolerance`; if none qualifies, the largest candidate, flagged as not met."""
    worst = {int(s): float(max(errs)) for s, errs in sorted(errors_by_steps.items())}
    for steps, error in worst.items():
        if error <= tolerance:
            return {"steps": steps, "completeness_met": True, "tolerance": tolerance,
                    "worst_relative_error_by_steps": worst}
    largest = max(worst)
    return {"steps": largest, "completeness_met": False, "tolerance": tolerance,
            "worst_relative_error_by_steps": worst}


def expected_gradients(wrapper, x, edge_index, token, node_type, bank: ReferenceBank,
                       fallback: torch.Tensor, target_class: int, steps: int, *,
                       draws: int, seed: int):
    """Average IG attributions over `draws` real-patient baselines."""
    if draws < 1:
        raise ValueError("draws must be positive")
    rng = np.random.default_rng(int(seed))
    total, errors = torch.zeros_like(x), []
    for _ in range(draws):
        baseline = bank.draw(token, node_type, fallback, rng, fill=x)
        attribution, completeness = integrated_gradients(
            wrapper, x, edge_index, baseline, target_class, steps)
        total += attribution
        errors.append(completeness["completeness_error_rel"])
    return total / draws, {"draws": draws, "mean_relative_error": float(np.mean(errors)),
                           "max_relative_error": float(np.max(errors))}
