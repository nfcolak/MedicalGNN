"""CEI-GNN with configurable message passing: method ``cei_gnn_mp2``.

Same required options as ``cei_gnn_v3`` (``arm``, ``k``, ``v3_state``, optional ``encoder_depth``,
``comorbid_block``) plus every option of ``cei.options.SPECS``. The adapter owns the training
objective terms (ERR-01 dual-read, LCL-04 shrinkage, LCL-02 masked-node reconstruction, LCL-06
coarse-group CE, LRB-06 two-view consistency), the residual-scale weight-decay group (LRB-02) and
the pre-trained encoder/MP initialisation (LCL-07). Each term exists only when its option is on.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from cei.plugin_cei_gnn_v3 import EvidenceAdapterV3, KNOWN_OPTIONS
from core.method_base import MethodOutput, _method_options, diagnostic_float
from .network import EvidenceNetworkMP2
from .options import SPECS, check_build, effective_options, ignored_options, validate

__all__ = ['EvidenceAdapterMP2', 'REGISTER', 'PRETRAIN_PREFIXES', 'CONFIG_KEYS', 'checkpoint_state',
           'checkpoint_config', 'load_pretrained', 'AdamDecoupledGates', 'is_weight_matrix']

# Network tensors a pre-training checkpoint may carry (LCL-07): the node-local encoder, the edge
# context modules and the message-passing stacks (with the ERR-04 branch projections and the RDO-07
# per-kind affine they were trained with). Heads and absence/pair tensors are never loaded.
PRETRAIN_PREFIXES = ('token_embedding.', 'node_type_embedding.', 'node_encoder.', 'node_norm.',
                     'encoder_blocks.', 'ple_projection.', 'relation_embedding.', 'triple_embedding.',
                     'edge_feature_projection.', 'mp_stack.', 'mp_branch_stacks.', 'branch_down.',
                     'branch_up.', 'branch_ctx.', 'kind_norm_gamma', 'kind_norm_beta')
# What a checkpoint's tensors mean depends on these (recorded next to it by cei.pretrain, compared on load).
# mp_layers (a shallower checkpoint would leave the extra layers random) and mp_norm (LayerNorm
# affine tensors fit a GraphNorm of the same width) are part of what the tensors mean.
CONFIG_KEYS = ('ple_mode', 'edge_direction', 'hidden', 'mp_kind', 'mp_branches', 'mp_branch_width',
               'mp_share_layers', 'mp_layers', 'mp_norm')


def checkpoint_state(network) -> dict:
    """Detached CPU copy of the tensors ``mp_pretrain_checkpoint`` loads (same names as the network)."""
    return {name: tensor.detach().cpu().clone() for name, tensor in network.state_dict().items()
            if name.startswith(PRETRAIN_PREFIXES)}


def checkpoint_config(adapter) -> dict:
    """The facts that fix what the checkpoint tensors mean (``CONFIG_KEYS``) for ``adapter``'s network."""
    options = adapter.mp2_options
    effective = effective_options(options, hidden=adapter.hidden)
    return {'ple_mode': options['ple_mode'], 'edge_direction': adapter.edge_direction,
            'hidden': adapter.hidden, 'mp_kind': options['mp_kind'], 'mp_branches': options['mp_branches'],
            'mp_branch_width': effective['mp_branch_width'] or 0,
            'mp_share_layers': options['mp_share_layers'], 'mp_layers': options['mp_layers'],
            'mp_norm': options['mp_norm']}


def _recorded_config(path):
    """``config`` of the sidecar ``<checkpoint>.json`` written by ``cei.pretrain`` (None if absent)."""
    sidecar = path.with_suffix(path.suffix + '.json')
    if not sidecar.is_file():
        return None
    config = json.loads(sidecar.read_text(encoding='utf-8')).get('config')
    return config if isinstance(config, dict) else None


def load_pretrained(network, path, *, expect=None) -> dict:
    """Copy matching encoder/MP tensors of ``path`` into ``network``; returns a load summary.

    Names outside ``PRETRAIN_PREFIXES`` or absent from the network are skipped (and counted);
    a name that matches with another shape is refused; nothing matching is an error. ``expect``
    (``checkpoint_config`` of the receiving adapter) is compared with the config recorded beside the
    checkpoint: any difference is refused. Without a recorded config the ``ple_projection`` tensors
    (raw weights in ple_mode=hat, cumulative increments in robust_cum) load only into a hat network.
    """
    path = Path(path).expanduser()
    recorded = _recorded_config(path) if expect is not None else None
    if recorded is not None:
        differing = {key: (recorded[key], expect[key]) for key in CONFIG_KEYS
                     if key in recorded and key in expect and recorded[key] != expect[key]}
        if differing:
            raise ValueError(f'mp_pretrain_checkpoint {str(path)!r} was made for another configuration '
                             f'(key: (checkpoint, network)) {differing}')
    raw = path.read_bytes()
    state = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and isinstance(state.get('state_dict'), dict):
        state = state['state_dict']
    if not isinstance(state, dict) or not all(torch.is_tensor(t) for t in state.values()):
        raise ValueError(f'mp_pretrain_checkpoint {str(path)!r} must hold a dict of tensors')
    own = network.state_dict()
    loadable, skipped = {}, []
    for name, tensor in state.items():
        if not name.startswith(PRETRAIN_PREFIXES) or name not in own:
            skipped.append(name)
            continue
        if tuple(tensor.shape) != tuple(own[name].shape):
            raise ValueError(f'mp_pretrain_checkpoint tensor {name!r} has shape {tuple(tensor.shape)}, '
                             f'the network expects {tuple(own[name].shape)}')
        loadable[name] = tensor.to(dtype=own[name].dtype)
    if not loadable:
        raise ValueError(f'mp_pretrain_checkpoint {str(path)!r} has no tensor matching the network')
    if (recorded is None and expect is not None and expect['ple_mode'] != 'hat'
            and any(name.startswith('ple_projection.') for name in loadable)):
        raise ValueError(f'mp_pretrain_checkpoint {str(path)!r} holds ple_projection tensors without a recorded '
                         f"ple_mode (<checkpoint>.json 'config'); they are raw weights in ple_mode=hat but "
                         f"cumulative increments in robust_cum, so they cannot enter a ple_mode={expect['ple_mode']!r} "
                         'network')
    network.load_state_dict(loadable, strict=False)
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'loaded': len(loadable), 'skipped': len(skipped),
            'config_checked': recorded is not None}


# mp_optimizer=adamw (mp2rev-train-02): decoupled decay only on weight matrices. These leaf names are the
# matrices (nn.Linear weights, the rgcn basis tensor V) of the stacks and heads; everything else (biases, norm
# affine, the class-prior `bias`, residual `alpha`, embeddings, per-triple / per-type tables such as table_*,
# g_fwd, g_rev, gk, gv, vote/gate tables, jk_logits, attention vectors) is excluded.
# g_fwd, g_rev, gk, gv are lookup tables (zero-initialised gains), so they left this list.
_MATRIX_LEAVES = frozenset(('weight', 'V'))


def is_weight_matrix(name, parameter) -> bool:
    return (parameter.ndim >= 2 and name.rsplit('.', 1)[-1] in _MATRIX_LEAVES
            and 'embedding' not in name and 'norm' not in name)


class AdamDecoupledGates(torch.optim.Adam):
    """Adam (coupled L2 ``weight_decay`` as today) whose groups with ``decoupled_decay`` additionally get the
    AdamW rule ``p *= 1 - lr * decay`` before the step (mp_gate_decay, mp2rev-train-07)."""

    def step(self, closure=None):
        with torch.no_grad():
            for group in self.param_groups:
                decay = group.get('decoupled_decay', 0.0)
                if decay > 0.0:
                    for parameter in group['params']:
                        if parameter.grad is not None:
                            parameter.mul_(1.0 - group['lr'] * decay)
        return super().step(closure)


def _method_weights(args):
    """The runner's ``--weights`` policy (``none`` when the adapter is built without it)."""
    value = args.get('weights') if isinstance(args, dict) else getattr(args, 'weights', None)
    return value or 'none'


def _read_group_map(path, num_classes):
    """(group index per class long[C], group names, {class: group}, file sha256) of a JSON {"<class index>": "<group>"}."""
    raw = Path(path).expanduser().read_bytes()
    document = json.loads(raw.decode('utf-8'))
    if not isinstance(document, dict):
        raise ValueError('aux_group_map must be a JSON object {"<class index>": "<group>"}')
    mapping = {}
    for key, group in document.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            raise ValueError(f'aux_group_map key {key!r} is not a class index') from None
        if not 0 <= index < num_classes or str(index) != str(key).strip():
            raise ValueError(f'aux_group_map class index {key!r} is outside 0..{num_classes - 1}')
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f'aux_group_map group of class {key!r} must be a non-empty string')
        mapping[index] = group.strip()
    missing = [c for c in range(num_classes) if c not in mapping]
    if missing:
        raise ValueError(f'aux_group_map must cover every class; missing {missing}')
    names = sorted(set(mapping.values()))
    if len(names) < 2:
        raise ValueError('aux_group_map needs at least two groups (a single group has a constant loss)')
    groups = torch.tensor([names.index(mapping[c]) for c in range(num_classes)], dtype=torch.long)
    return groups, tuple(names), {str(c): mapping[c] for c in range(num_classes)}, hashlib.sha256(raw).hexdigest()


class EvidenceAdapterMP2(EvidenceAdapterV3):
    adaptation_version = "clinical_graph_v2_cei_gnn_mp2"
    known_options = KNOWN_OPTIONS | set(SPECS)
    network_class = EvidenceNetworkMP2

    @classmethod
    def validate_method_options(cls, args) -> None:
        """Plan-mode check: the v3 options plus every mp2 option (types, ranges, combinations, build facts)."""
        super().validate_method_options(args)
        supplied = _method_options(args)
        options = validate({name: supplied[name] for name in SPECS if name in supplied})
        defaults = cls.runner_defaults
        hidden = getattr(args, 'hidden', None) if not isinstance(args, dict) else args.get('hidden')
        dropout = getattr(args, 'dropout', None) if not isinstance(args, dict) else args.get('dropout')
        check_build(options, hidden=int(defaults['hidden'] if hidden is None else hidden),
                    dropout=float(defaults['dropout'] if dropout is None else dropout))
        EvidenceNetworkMP2._check_options(options)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        options = self.mp2_options
        self.group_names = ()
        self.group_info = None
        self.pretrain_info = None
        self.weight_policy = str(_method_weights(kwargs.get('args')))
        self.class_weight = None   # set by on_epoch_start from the train split, like the runner's criterion
        if options['aux_group_map']:
            group_of_class, self.group_names, assignment, digest = _read_group_map(
                options['aux_group_map'], self.num_classes)
            self.group_info = {'assignment': assignment, 'sha256': digest}
            member = F.one_hot(group_of_class, len(self.group_names)).bool()
            self.register_buffer('group_of_class', group_of_class, persistent=False)
            self.register_buffer('group_member', member, persistent=False)
        if options['mp_pretrain_checkpoint']:
            self.pretrain_info = load_pretrained(self.network, options['mp_pretrain_checkpoint'],
                                                 expect=checkpoint_config(self))

    def _network_kwargs(self, args):
        supplied = _method_options(args)
        self.mp2_options = validate({name: supplied[name] for name in SPECS if name in supplied})
        check_build(self.mp2_options, hidden=self.hidden, dropout=self.dropout_rate)
        return {'mp2_options': dict(self.mp2_options),
                'relation_names': tuple(self._relation_layout())}

    def on_epoch_start(self, epoch: int, train_loader) -> None:
        super().on_epoch_start(epoch, train_loader)
        if self.class_weight is None and self.weight_policy != 'none':
            from core.train import class_weights   # the runner's own policy on the same train split
            labels = [int(graph.y) for graph in train_loader.dataset]
            self.class_weight = torch.tensor(class_weights(labels, self.weight_policy, self.num_classes),
                                             dtype=torch.float32)

    # ------------------------------------------------------------------ objective terms

    def _sample_weight(self, y):
        """Per-graph weight of the runner's CrossEntropyLoss(weight=class_weights); None = all ones."""
        if self.weight_policy == 'none':
            return None
        if self.class_weight is None:
            raise ValueError(f"cei_gnn_mp2 auxiliary CE terms need the class weights of weights="
                             f"{self.weight_policy!r}: call on_epoch_start(epoch, train_loader) first")
        return self.class_weight.to(y.device)[y]

    def _weighted_mean(self, per_graph, y):
        """Weighted mean of per-graph losses with the normalisation of nn.CrossEntropyLoss(weight=...)."""
        weight = self._sample_weight(y)
        if weight is None:
            return per_graph.mean()
        return (per_graph * weight).sum() / weight.sum()

    def _weighted_ce(self, logits, y, smoothing=0.0):
        """Weighted CE; ``smoothing`` > 0 mixes in the uniform target: (1 - e) * CE + e * mean_c(-log p_c)."""
        per_graph = F.cross_entropy(logits, y, reduction='none')
        if smoothing > 0.0:
            per_graph = (1.0 - smoothing) * per_graph - smoothing * F.log_softmax(logits, dim=-1).mean(dim=-1)
        return self._weighted_mean(per_graph, y)

    def _group_loss(self, logits, y):
        """LCL-06: CE on log P(group) = logsumexp(group logits) - logsumexp(all logits)."""
        member = self.group_member.to(logits.device)
        masked = logits[:, :, None].masked_fill(~member[None], float('-inf'))
        log_group = masked.logsumexp(dim=1) - logits.logsumexp(dim=1, keepdim=True)
        target = self.group_of_class.to(logits.device)[y]
        return self._weighted_mean(F.nll_loss(log_group, target, reduction='none'), y)

    @staticmethod
    def _symmetric_kl(logits_a, logits_b):
        """LRB-06: 0.5 * (KL(a || b) + KL(b || a)) of the softmaxes, mean over graphs."""
        log_a, log_b = F.log_softmax(logits_a, dim=-1), F.log_softmax(logits_b, dim=-1)
        kl_ab = (log_a.exp() * (log_a - log_b)).sum(dim=-1)
        kl_ba = (log_b.exp() * (log_b - log_a)).sum(dim=-1)
        return 0.5 * (kl_ab + kl_ba).mean()

    def forward(self, batch, *, epoch: int) -> MethodOutput:
        del epoch
        options = self.mp2_options
        clinical, membership, visit_graph = self._read(batch)
        features = self.network.continuous_inputs(clinical)
        parts = self.network.forward_continuous(features, clinical.edge_index, clinical, membership,
                                                return_parts=True, visit_graph=visit_graph)
        logits = parts["logits"]
        auxiliary_loss = logits.sum() * 0.0
        terms = {}
        correction = None   # signed CE terms that turn the runner's plain CE into the intended one
        if self.training:
            y = None
            smoothing = options['mp_label_smoothing']
            if options['mp_dual_read'] or self.group_names or options['mp_consistency'] > 0 or smoothing > 0:
                if getattr(batch, 'y', None) is None:
                    raise ValueError('cei_gnn_mp2 label-based auxiliary terms need batch.y')
                y = batch.y.view(-1).long().to(logits.device)
            if options['mp_dual_read']:
                terms['aux_local_loss'] = self._weighted_ce(parts['aux_local_logits'], y)
                auxiliary_loss = auxiliary_loss + options['mp_aux_weight'] * terms['aux_local_loss']
            if options['mp_shrink'] > 0 and 'mp_shrink_loss' in parts:
                terms['mp_shrink_loss'] = parts['mp_shrink_loss']
                auxiliary_loss = auxiliary_loss + options['mp_shrink'] * parts['mp_shrink_loss']
            if options['mp_mask_rate'] > 0 and 'mask_recon_loss' in parts:
                terms['mask_recon_loss'] = parts['mask_recon_loss']
                auxiliary_loss = auxiliary_loss + options['mp_mask_weight'] * parts['mask_recon_loss']
            if self.group_names:
                terms['group_loss'] = self._group_loss(logits, y)
                auxiliary_loss = auxiliary_loss + options['aux_group_weight'] * terms['group_loss']
            if options['mp_consistency'] > 0:
                second = self.network.forward_continuous(
                    features, clinical.edge_index, clinical, membership, return_parts=False,
                    visit_graph=visit_graph)
                terms['consistency_loss'] = self._symmetric_kl(logits, second)
                auxiliary_loss = auxiliary_loss + options['mp_consistency'] * terms['consistency_loss']
                # The runner adds CE(view a); CE of the two views is averaged: 0.5 * (CE_a + CE_b)
                # = CE_a + 0.5 * (CE_b - CE_a), the same weighted CE on both views.
                terms['consistency_ce_b'] = self._weighted_ce(second, y, smoothing)
                correction = 0.5 * (terms['consistency_ce_b'] - self._weighted_ce(logits, y, smoothing))
            if smoothing > 0:   # runner CE (plain) -> smoothed weighted CE of the main view
                terms['label_smoothing_correction'] = (self._weighted_ce(logits, y, smoothing)
                                                       - self._weighted_ce(logits, y))
                correction = (terms['label_smoothing_correction'] if correction is None
                              else correction + terms['label_smoothing_correction'])
        # mp2rev-train-05: the CE corrections are loss, not regularisation; the reported value omits them
        reported = auxiliary_loss
        if correction is not None:
            auxiliary_loss = auxiliary_loss + correction
        graphs = max(int(clinical.graph_count), 1)
        diagnostics = {
            "auxiliary_loss": diagnostic_float(reported, "cei_gnn_mp2"),
            "pairs_per_graph": float(parts["pairs"].size(1) / graphs),
            "absent_items_per_graph": float(parts["absence_items"].size(1) / graphs)}
        if correction is not None:
            diagnostics["reported_auxiliary_loss"] = diagnostics["auxiliary_loss"]
            diagnostics["ce_correction"] = diagnostic_float(correction, "cei_gnn_mp2")
        for name, value in terms.items():
            diagnostics[name] = diagnostic_float(value, "cei_gnn_mp2")
        return MethodOutput(logits=logits, auxiliary_loss=auxiliary_loss, diagnostics=diagnostics)

    # ------------------------------------------------------------------ optimizer

    def optimizer_groups(self, args):
        del args
        options = self.mp2_options
        decay = options['mp_gate_decay']
        if options['mp_optimizer'] == 'adamw':   # mp2rev-train-02: explicit decoupled decay per group
            matrices, plain, gates = [], [], []
            for name, parameter in self.named_parameters():
                if decay > 0 and name.endswith('.alpha'):
                    gates.append(parameter)
                else:
                    (matrices if is_weight_matrix(name, parameter) else plain).append(parameter)
            groups = [{"params": matrices, "weight_decay": options['mp_weight_decay']}] if matrices else []
            if plain:
                groups.append({"params": plain, "weight_decay": 0.0})
            if gates:
                groups.append({"params": gates, "weight_decay": decay})   # LRB-02, decoupled under AdamW
            return groups
        if decay <= 0:
            return [{"params": list(self.parameters())}]
        gates, others = [], []
        for name, parameter in self.named_parameters():
            (gates if name.endswith('.alpha') else others).append(parameter)
        groups = [{"params": others}] if others else []
        if gates:
            # LRB-02; build_optimizer applies this decay decoupled (mp2rev-train-07), not as Adam's coupled L2
            groups.append({"params": gates, "weight_decay": decay, "gate_decay": True})
        return groups

    def build_optimizer(self, args):
        """Optimizer hook of ``core.train`` (default options: the same ``torch.optim.Adam`` as before)."""
        groups = self.optimizer_groups(args)
        if self.mp2_options['mp_optimizer'] == 'adamw':
            return torch.optim.AdamW(groups, lr=args.lr, weight_decay=self.mp2_options['mp_weight_decay'])
        if not any(group.get('gate_decay') for group in groups):
            return torch.optim.Adam(groups, lr=args.lr, weight_decay=args.weight_decay)
        groups = [({"params": g["params"], "weight_decay": 0.0, "decoupled_decay": g["weight_decay"]}
                   if g.get('gate_decay') else g) for g in groups]
        return AdamDecoupledGates(groups, lr=args.lr, weight_decay=args.weight_decay)

    def lr_scheduler(self, optimizer, epochs):
        """mp_lr_schedule=cosine: lr x (0.1 + 0.9 * (1 + cos(pi * e / epochs)) / 2), stepped once per epoch."""
        if self.mp2_options['mp_lr_schedule'] != 'cosine':
            return None
        epochs = max(int(epochs), 1)
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda e: 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(e, epochs) / epochs)))

    # ------------------------------------------------------------------ binding

    def run_config(self) -> dict:
        config = super().run_config()
        options = dict(self.mp2_options)
        config["method"] = "cei_gnn_mp2"
        config["mp2_options"] = options
        config["effective_settings"] = {**config["effective_settings"], **options}
        # (mp2v3-model-02): v3's 'layers' is the encoder depth, not the message-passing depth
        architecture = {**config["architecture"], **options}
        architecture["encoder_layers"] = architecture.pop("layers")
        config["architecture"] = architecture
        config["mp_layers"], config["mp_global"] = options["mp_layers"], options["mp_global"]   # as cei_gnn_mp
        config["mp2_effective"] = effective_options(options, hidden=self.hidden, dropout=self.dropout_rate)
        config["mp2_ignored_options"] = ignored_options(options)   # mp2rev-train-04
        if self.group_names:
            config["aux_group_names"] = list(self.group_names)
            config["aux_group_assignment"] = dict(self.group_info["assignment"])
            config["aux_group_map_sha256"] = self.group_info["sha256"]
        if self.weight_policy != "none":
            config["aux_ce_class_weights"] = self.weight_policy
        if self.pretrain_info is not None:
            config["mp_pretrain_checkpoint_sha256"] = self.pretrain_info["sha256"]
            config["mp_pretrain_loaded_tensors"] = self.pretrain_info["loaded"]
            config["mp_pretrain_skipped_tensors"] = self.pretrain_info["skipped"]
            config["mp_pretrain_config_checked"] = self.pretrain_info["config_checked"]
        return config

    def common_init_identical_to_c(self) -> bool:
        # a pre-training checkpoint overwrites encoder tensors that C's initial values would hold
        options = self.mp2_options
        # these options register other tensors under C's names (other shape or other values)
        replaces_c_tensor = (options['shared_gate'] == 1 or 0 < options['edge_head_rank'] < self.hidden
                             or options['kind_norm'] == 1 or options['ple_mode'] != 'hat')
        return super().common_init_identical_to_c() and self.pretrain_info is None and not replaces_c_tensor


REGISTER = {"cei_gnn_mp2": EvidenceAdapterMP2}
