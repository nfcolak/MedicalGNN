"""Real vendored GraphXAI graph algorithms, with explicit PyG compatibility fixes.

Upstream GNNExplainer's optimizer is retained. Its base prediction helper disables
all gradients and its legacy double-underscore masks are ignored by modern PyG;
this adapter repairs those two API defects, not the optimization algorithm. Two
settings differ from the vendored defaults and are recorded in every provenance:
the feature-mask size penalty is the MEAN of sigmoid(F) over the columns (PyG's
reduction; the vendored sum grows with the width of the explained row), and the
vendored module-level device is replaced by the device of the explained features.

GNNExplainer normally drives its learned edge mask into the model by patching
every `torch_geometric.nn.MessagePassing` submodule (`set_masks`/`clear_masks`,
below). Models that aggregate by hand instead of subclassing `MessagePassing`
(e.g. the clinical_graph_v2 adapters) have nothing for that patch to attach to,
so `set_masks` silently finds zero targets and the optimizer climbs a mask that
never touches a single message. `CompatibleGNNExplainer._predict` below adds a
second, explicit delivery path for exactly that case: if the wrapped model's
own `forward` declares an `edge_mask` parameter, the live mask (same Parameter,
every optimisation step, sigmoid-activated like the MessagePassing path) is
passed as that keyword argument. Models that don't declare `edge_mask` are
unaffected: the signature check is `False` for them, so the call shape is
unchanged. A model with both paths would receive the mask twice (a mask m
would act as m * m), so `CompatibleGNNExplainer._set_masks` refuses such a
model; no final wrapper has both.
"""
from contextlib import nullcontext
from pathlib import Path
import hashlib
import inspect
import os
import sys
import torch
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.explain.algorithm.utils import set_masks, clear_masks
from core.explain.explanation_contract import build_node_explanation

VENDOR = Path(__file__).resolve().parents[2] / 'external/GraphXAI-main'
sys.path.insert(0, str(VENDOR))
from graphxai.explainers import GradExplainer, IntegratedGradExplainer, GNNExplainer

ALGORITHMS = ('GradExplainer', 'IntegratedGradExplainer', 'GNNExplainer')
GRADIENT_NODE_REDUCTION = 'sum_abs_over_columns'
GNNEXPLAINER_FEATURE_PENALTY = 'mean'
GNNEXPLAINER_COEFF_NOTE = ('edge: size 0.005 * sum(sigmoid(M)), entropy 1.0 * mean; feature: '
                           'size 1.0 * mean(sigmoid(F)) over the columns, entropy 0.1 * mean')
SOFT_MASK_NOTE = ('a fractional edge mask is not a removal: hard 0/1 masks equal deletion and '
                  'keeping, soft values are compared by rank only')
# What GNNExplainer's edge mask can and cannot reach in each final model.
GNNEXPLAINER_DISCLOSURE = {
    'gsat': 'the mask scales the edge attention of both passes (extractor and predictor) once',
    'protgnn': ('weighted mean sum(w * msg) / max(sum(w), 1): a common scale of a receiver\'s '
                'incoming masks is invisible while their sum is at least 1'),
    'graphcare': ('edges with a global endpoint are never read by the model; their mask entries '
                  'are set by the regularisers alone and are excluded from the node reduction '
                  '(inert_edges)'),
    'cei_gnn_v3': ('within-visit pair contributions, the absence block and the node rows are not '
                   'reachable by an edge mask'),
    'cei_gnn_mp2': ('within-visit pair contributions, the absence block and the node-local rows '
                    'are not reachable by an edge mask; the message-passing weighted means make a '
                    'common scale of a receiver\'s incoming masks partly invisible'),
}


def _model_declares_edge_mask(model):
    try:
        return 'edge_mask' in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def _has_message_passing(model):
    return any(isinstance(module, MessagePassing) for module in model.modules())


def aggregate_reverse_edge_pairs(edge_imp, reversible_edge_pairs):
    """Collapse each (forward_index, reverse_index) pair to their mean, in
    place of two independently-learned mask weights for what is really one
    clinical fact (reverse edges are not separate evidence). A reverse edge
    (edge_direction='bidirectional') duplicates a single fact as two directed
    edges; left alone, that fact gets two chances to push its incident nodes'
    importance up under the incident-edge node reduction below, inflating anything
    touching a reversible relation relative to a forward-only one. `None` or
    `[]` is a no-op, returning `edge_imp` unchanged (the entire function is
    skipped by forward-only callers and every non-clinical_graph_v2 caller)."""
    if not reversible_edge_pairs:
        return edge_imp
    edge_imp = edge_imp.clone()
    for forward_i, reverse_i in reversible_edge_pairs:
        combined = 0.5 * (edge_imp[forward_i] + edge_imp[reverse_i])
        edge_imp[forward_i] = combined
        edge_imp[reverse_i] = combined
    return edge_imp


class CompatibleGNNExplainer(GNNExplainer):
    def __init__(self, model, coeff=None, *, feature_width=None):
        """`feature_width` (the explained row's width) turns the vendored feature-mask size
        penalty 1.0 * sum(sigmoid(F)) into 1.0 * mean(sigmoid(F))."""
        if coeff is None and feature_width is not None:
            coeff = {'edge': {'entropy': 1.0, 'size': 0.005},
                     'feature': {'entropy': 0.1, 'size': 1.0 / max(int(feature_width), 1)}}
        super().__init__(model, coeff)

    def get_explanation_graph(self, x, edge_index, **kwargs):
        # The vendored module moves x to a module-level `device` ('cuda' when available);
        # the explained model lives on the device of x.
        module = sys.modules[GNNExplainer.__module__]
        saved = module.device
        module.device = x.device
        try:
            return super().get_explanation_graph(x, edge_index, **kwargs)
        finally:
            module.device = saved

    def _predict(self, x, edge_index, return_type='label', forward_kwargs=None):
        kwargs = dict(forward_kwargs or {})
        mask = getattr(self, 'edge_mask', None)
        if mask is not None and _model_declares_edge_mask(self.model):
            kwargs['edge_mask'] = mask.sigmoid()
        out = self.model(x, edge_index, **kwargs)
        if return_type == 'label':
            return out.argmax(-1)
        if return_type == 'prob':
            return out.softmax(-1)
        if return_type == 'log_prob':
            return out.log_softmax(-1)
        raise ValueError('Unknown prediction return type')

    def _set_masks(self, x, edge_index, **kwargs):
        if _model_declares_edge_mask(self.model) and _has_message_passing(self.model):
            raise ValueError('the model both declares edge_mask and contains a MessagePassing '
                             'module; the mask would be applied twice')
        super()._set_masks(x, edge_index, **kwargs)
        set_masks(self.model, self.edge_mask, edge_index, apply_sigmoid=True)

    def _clear_masks(self):
        clear_masks(self.model)
        super()._clear_masks()

# GNNExplainer's budget: kept at the 50 epochs every earlier study used; the paper uses
# 100-300 and PyG defaults to 100 (disclosed in each record's provenance).
GNNEXPLAINER_EPOCHS_NOTE = '50 epochs (paper 100-300, PyG default 100)'


def _explained_columns(wrapper, width):
    spec = wrapper.ig_feature_spec() if hasattr(wrapper, 'ig_feature_spec') else {}
    if spec.get('token_columns'):
        start, stop = (int(column) for column in spec['token_columns'])
        return {'numeric': [0, start], 'token_vector': [start, stop]}
    return {'numeric': [0, int(width)]}


def _ig_path(wrapper):
    return wrapper.ig_path() if hasattr(wrapper, 'ig_path') else nullcontext()


def explain_algorithms(wrapper, x, edge_index, *, batch, steps=32, epochs=50,
                       reversible_edge_pairs=None, node_reduction='sum', target_class=None,
                       seed=None, inert_edge_mask=None, method=None, v1_fidelity=False):
    """GradExplainer, the vendored (zero-baseline) IntegratedGradExplainer and GNNExplainer.

    `seed` (optional) fixes GNNExplainer's random mask initialisation: the connectivity
    pre-check and the optimisation run inside one `torch.random.fork_rng()`, the optimisation
    after `torch.manual_seed(seed)`, so the same visit gives the same explanation and the
    caller's global torch RNG stream is left untouched (also with `seed=None`).

    `inert_edge_mask` (bool [E], e.g. the GraphCare wrapper's edges with a global endpoint,
    which the model never reads) excludes those edges from GNNExplainer's node reduction; their
    count and indices are recorded as `inert_edges`. `method` (default: `wrapper.method`) selects
    the GNNExplainer disclosure text. The vendored IG runs inside `wrapper.ig_path()` where the
    wrapper has one. `v1_fidelity` adds the v1 (x-row zeroing) fidelity to each node
    explanation; new explanation records leave it off."""
    if steps < 1 or epochs < 1:
        raise ValueError('positive explanation budgets required')
    # 'sum' is the historical reduction every existing caller (e.g. the CEI GraphXAI
    # studies) was run with; 'mean' divides by incident-edge count to remove the
    # degree bias and is opt-in so existing results stay reproducible.
    if node_reduction not in ('sum', 'mean'):
        raise ValueError("node_reduction must be 'sum' or 'mean'")
    if inert_edge_mask is not None:
        inert_edge_mask = torch.as_tensor(inert_edge_mask, dtype=torch.bool, device=x.device)
        if inert_edge_mask.shape != (edge_index.size(1),):
            raise ValueError(f'inert_edge_mask must have one entry per edge ({edge_index.size(1)}); '
                             f'got shape {tuple(inert_edge_mask.shape)}')
    method = method if method is not None else getattr(wrapper, 'method', None)
    wrapper.eval()
    with torch.no_grad():
        label = wrapper(x, edge_index, batch=batch).argmax(-1)
    # target_class=None explains the model's own prediction (every existing caller).
    # An explicit class explains that class's logit instead -- e.g. the TRUE class of a
    # misclassified patient. GNNExplainer cannot be redirected: the vendored optimiser
    # hard-codes the model's predicted class as its target.
    predicted_label = label
    if target_class is not None:
        label = torch.tensor([int(target_class)], device=label.device)
    # Attribute the predicted logit, rather than an unrelated training label.
    criterion = lambda logits, target: logits.gather(1, target.view(-1,1)).sum()
    aggregate = lambda values, dim: values.abs().sum(dim=dim)
    columns = _explained_columns(wrapper, x.size(1))
    result = {}
    sources = ['grad.py','integrated_grad.py','gnn_explainer.py','_base.py']
    hashes = {name:hashlib.sha256((VENDOR/'graphxai/explainers'/name).read_bytes()).hexdigest() for name in sources}
    for name in ALGORITHMS:
        wrapper.zero_grad(set_to_none=True)
        features = x.detach().clone()
        provenance = {'implementation':'vendored GraphXAI', 'source_sha256':hashes,
                      'torch':torch.__version__, 'pyg':torch_geometric.__version__,
                      'target':('predicted_logit' if target_class is None else f'class_{int(target_class)}_logit')
                               if name != 'GNNExplainer' else 'predicted_class_log_probability',
                      'steps':steps if name == 'IntegratedGradExplainer' else None,
                      'epochs':epochs if name == 'GNNExplainer' else None,
                      'explained_columns':columns}
        if name != 'GNNExplainer':
            provenance['node_reduction'] = GRADIENT_NODE_REDUCTION
        if name == 'IntegratedGradExplainer':
            provenance['baseline'] = 'all_zero'
        if name == 'GNNExplainer':
            provenance['seed'] = None if seed is None else int(seed)
            provenance['epochs_note'] = GNNEXPLAINER_EPOCHS_NOTE
            provenance['feature_penalty'] = GNNEXPLAINER_FEATURE_PENALTY
            provenance['coefficients'] = GNNEXPLAINER_COEFF_NOTE
            provenance['soft_mask'] = SOFT_MASK_NOTE
            provenance['disclosure'] = GNNEXPLAINER_DISCLOSURE.get(method)
        # Isolated per algorithm: one algorithm's failure (e.g. GNNExplainer's
        # disconnected-gradient refusal below) must not discard the other two
        # algorithms' already-computed results along with it.
        if name == 'GNNExplainer' and target_class is not None and int(label.item()) != int(predicted_label.item()):
            result[name] = {'status': 'unsupported', 'provenance': provenance,
                            'unsupported_reason': 'GNNExplainer always explains the model\'s own predicted '
                                                  'class; it cannot be directed at another class'}
            continue
        try:
            if name == 'GradExplainer':
                if hasattr(wrapper, 'ple_term'):
                    provenance['ple_term'] = wrapper.ple_term
                exp = GradExplainer(wrapper, criterion).get_explanation_graph(features,edge_index,label,
                    aggregate_node_imp=aggregate,forward_kwargs={'batch':batch})
            elif name == 'IntegratedGradExplainer':
                with _ig_path(wrapper):
                    if hasattr(wrapper, 'ple_term'):
                        provenance['ple_term'] = wrapper.ple_term
                    exp = IntegratedGradExplainer(wrapper,criterion).get_explanation_graph(edge_index,features,label,
                        steps=steps,node_agg=aggregate,forward_kwargs={'batch':batch})
            else:
                if hasattr(wrapper, 'ple_term'):
                    provenance['ple_term'] = wrapper.ple_term
                ex = CompatibleGNNExplainer(wrapper, feature_width=features.size(1))
                old = os.environ.get('GNNEXP_EPOCHS')
                try:
                    with torch.random.fork_rng(devices=[]):
                        ex._set_masks(features,edge_index,explain_feature=True,device=x.device)
                        # This connectivity pre-check calls `wrapper` directly, not
                        # through `_predict` (unlike the optimisation loop inside
                        # get_explanation_graph below) -- so it needs the same explicit
                        # edge_mask handoff repeated here, or it would (correctly, for
                        # the wrong reason) report "disconnected" for any model that
                        # only supports the mask via this keyword, before ever reaching
                        # the loop that actually uses it.
                        precheck_kwargs = {'batch': batch}
                        if _model_declares_edge_mask(wrapper):
                            precheck_kwargs['edge_mask'] = ex.edge_mask.sigmoid()
                        logits = wrapper(features,edge_index,**precheck_kwargs)
                        gradient = torch.autograd.grad(logits[0,label.item()],ex.edge_mask,allow_unused=True)[0]
                        edgeless = edge_index.size(1) == 0
                        if not edgeless and (gradient is None or not torch.isfinite(gradient).all()):
                            raise RuntimeError('GNNExplainer edge mask gradient disconnected or nonfinite; unsupported graph/model')
                        # Connected zero derivatives are legitimate flat regions. An empty
                        # graph has no edge derivative to verify; never label that verified.
                        provenance['edge_count'] = int(edge_index.size(1))
                        provenance['edge_gradient_verified'] = not edgeless
                        provenance['edge_gradient_status'] = 'not_applicable_edgeless' if edgeless else 'connected_finite'
                        provenance['edge_gradient_l1'] = float(gradient.abs().sum()) if gradient is not None else 0.0
                        provenance['zero_predictive_edge_gradient'] = provenance['edge_gradient_l1'] == 0.0
                        provenance['empty_edge_entropy'] = 'zero'
                        ex._clear_masks()
                        os.environ['GNNEXP_EPOCHS'] = str(epochs)
                        if seed is not None:
                            torch.manual_seed(int(seed))
                        exp = ex.get_explanation_graph(features,edge_index,forward_kwargs={'batch':batch})
                finally:
                    ex._clear_masks()
                    if old is None:
                        os.environ.pop('GNNEXP_EPOCHS',None)
                    else:
                        os.environ['GNNEXP_EPOCHS'] = old
                edge_imp = aggregate_reverse_edge_pairs(exp.edge_imp.detach(), reversible_edge_pairs)
                provenance['reverse_pairs_aggregated'] = len(reversible_edge_pairs or [])
                # Continuous incident-edge mass avoids upstream's binary node mask ties.
                # 'mean' (opt-in): summing raw mask mass over incident edges makes node
                # importance scale with node degree almost independently of which
                # edges the mask actually learned to weight highly -- a high-degree
                # hub accumulates the largest total even when no single incident edge
                # is individually important. Dividing by incident-edge count isolates
                # the mask's actual signal from this degree confound. Inert edges (never
                # read by the model) carry no mass and add no degree.
                read = (torch.ones_like(edge_imp) if inert_edge_mask is None
                        else (~inert_edge_mask).to(edge_imp.dtype))
                importance = torch.zeros(x.size(0),device=x.device)
                degree = torch.zeros(x.size(0),device=x.device)
                importance.index_add_(0,edge_index[0],edge_imp * read)
                importance.index_add_(0,edge_index[1],edge_imp * read)
                degree.index_add_(0,edge_index[0],read)
                degree.index_add_(0,edge_index[1],read)
                exp.node_imp = importance / degree.clamp_min(1.0) if node_reduction == 'mean' else importance
                provenance['node_reduction'] = f'{node_reduction}_incident_continuous_edge_mask'
                provenance['inert_edges'] = {
                    'count': 0 if inert_edge_mask is None else int(inert_edge_mask.sum()),
                    'indices': [] if inert_edge_mask is None
                               else inert_edge_mask.nonzero().flatten().tolist(),
                    'rule': 'excluded from the node reduction (never read by the model)'}
        except Exception as exc:
            result[name] = {'status':'failed','error':str(exc),'error_type':type(exc).__name__,
                           'provenance':provenance}
            continue
        result[name] = {'status':'success','provenance':provenance,
                       'node_explanation':build_node_explanation(wrapper,x,edge_index,batch=batch,
                                                                 node_importance=exp.node_imp,
                                                                 v1_fidelity=v1_fidelity)}
    wrapper.zero_grad(set_to_none=True)
    return result
