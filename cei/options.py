"""Method options of ``cei_gnn_mp2``: one table, typed defaults, the legacy (= ``cei_gnn_mp``) values.

``SPECS[name]`` carries ``type``, ``default`` (the new method's default), ``legacy`` (the value that
reproduces ``cei_gnn_mp`` exactly), ``choices``, ``minimum``/``maximum`` (both inclusive: the seven
names in ``EXCLUSIVE_MAXIMUM`` have the open range [0, 1), stored as the largest float below 1.0),
``owner`` (``stack`` = layers.py, ``network`` = network.py, ``adapter`` = plugin_cei_gnn_mp2.py) and
``source`` (proposal ids). ``validate(raw)`` is the only way option values enter the method;
``check_build(options, hidden=, dropout=)`` is the second gate for the facts validate cannot see
(the width and the dropout rate), called by the adapter before the network is built.
"""
from __future__ import annotations

import math
import os
import warnings

from core import NODE_KINDS
from core.tensorize import relation_vocabulary

__all__ = ['SPECS', 'DEFAULTS', 'LEGACY', 'STACK_KEYS', 'NETWORK_KEYS', 'ADAPTER_KEYS',
           'EXCLUSIVE_MAXIMUM', 'validate', 'check_build', 'effective_options', 'ignored_options']

_HUB_KINDS = 'visit,complaint,patient'
_EXEMPT = 'has_visit,index_visit_of'
_RELATIONS = frozenset(relation_vocabulary('bidirectional'))
# Options whose range is [minimum, 1) rather than [minimum, 1]; SPECS stores the largest float below 1.0
# as their (inclusive) maximum, so a reader of SPECS['maximum'] sees an admissible bound.
EXCLUSIVE_MAXIMUM = frozenset(('mp_att_dropout', 'mp_dropedge', 'mp_kg_edge_drop', 'mp_dropmsg',
                               'mp_dropnode', 'mp_droppath', 'mp_dropout'))
_BELOW_ONE = math.nextafter(1.0, 0.0)

# name, type, DEFAULT, LEGACY, choices, minimum, maximum, owner, proposals
_TABLE = (
    ('mp_layers', int, 2, 2, None, 1, None, 'stack', 'STR-06,PLN-02'),
    ('mp_kind', str, 'gine', 'gine', ('gine', 'rgcn', 'diag', 'simplehgn', 'hgt'), None, None, 'stack',
     'HET-02,HET-04,HET-05,HET-06'),
    ('mp_msg_dim', int, 64, 0, None, 0, None, 'stack', 'MSG-01,REG-04'),
    ('mp_rank', int, 0, 0, None, 0, None, 'stack', 'REG-04,LRB-01'),
    ('mp_tie_directions', int, 0, 0, None, 0, 1, 'stack', 'STR-04,LRB-01'),
    ('mp_root', str, 'linear', 'linear', ('linear', 'bias'), None, None, 'stack', 'LRB-01'),
    ('mp_share_layers', int, 0, 0, None, 0, 2, 'stack', 'REG-03,LCL-01,MSG-01'),
    ('mp_residual', str, 'channel', 'none', ('none', 'scalar', 'channel', 'kind', 'kind_vec'), None, None,
     'stack', 'MSG-06,REG-02,LRB-02,EVD-02'),
    ('mp_residual_init', float, 0.1, 0.1, None, 0.0, 1.0, 'stack', 'MSG-06,REG-02'),
    ('mp_residual_bound', int, 0, 0, None, 0, 1, 'stack', 'ERR-02'),
    ('mp_aggr', str, 'relation', 'mean', ('mean', 'relation', 'triple'), None, None, 'stack', 'STR-01,HET-01'),
    ('mp_degree_norm', str, 'mean', 'mean', ('mean', 'sym', 'null'), None, None, 'stack', 'LCL-05,RDO-08'),
    ('mp_edge_gate', int, 0, 0, None, 0, 1, 'stack', 'MSG-03'),
    ('mp_rev_attention', int, 0, 0, None, 0, 1, 'stack', 'MSG-04'),
    ('mp_att_dropout', float, 0.0, 0.0, None, 0.0, 1.0, 'stack', 'MSG-04,HET-05,HET-06'),
    ('mp_rev_agg', str, 'mean', 'mean', ('mean', 'meanmax'), None, None, 'stack', 'MSG-05'),
    ('mp_hub_max', int, 0, 0, None, 0, 1, 'stack', 'ERR-05'),
    # mp_hub_kinds: node kinds that get the mp_hub_max max term. Default visit,complaint,patient;
    # on these graphs that effectively means the index visit only (patient/complaint are no
    # hubs; diagnosis/knowledge nodes are the high-degree ones but are not listed). fv10-m1-02.
    ('mp_hub_kinds', str, _HUB_KINDS, _HUB_KINDS, None, None, None, 'stack', 'ERR-05'),
    ('mp_relation_film', int, 0, 0, None, 0, 1, 'stack', 'MSG-02'),
    ('mp_rel_gate', int, 0, 0, None, 0, 1, 'stack', 'LCL-03'),
    ('mp_global', int, 0, 1, None, 0, 2, 'stack', 'PLN-01,RDO-01,STR-05'),
    # mp_dedup: the mirror test ignores the relation: any (dst, src) node pair of any relation
    # zeroes the reverse message, e.g. has_visit/index_visit_of too. fv10-m1-03.
    ('mp_dedup', int, 0, 0, None, 0, 1, 'stack', 'STR-07'),
    ('mp_ctx', str, 'matrix', 'matrix', ('matrix', 'table'), None, None, 'stack', 'HET-03'),
    ('mp_bases', int, 2, 2, None, 1, None, 'stack', 'HET-02'),
    ('mp_heads', int, 4, 4, None, 1, None, 'stack', 'HET-05,HET-06'),
    ('mp_dropedge', float, 0.2, 0.0, None, 0.0, 1.0, 'stack', 'REG-01,STR-02'),
    ('mp_dropedge_exempt', str, _EXEMPT, _EXEMPT, None, None, None, 'stack', 'STR-02'),
    ('mp_kg_edge_drop', float, 0.0, 0.0, None, 0.0, 1.0, 'stack', 'LCL-03'),
    ('mp_dropmsg', float, 0.0, 0.0, None, 0.0, 1.0, 'stack', 'REG-07,LRB-03'),
    ('mp_dropnode', float, 0.0, 0.0, None, 0.0, 1.0, 'stack', 'LRB-03'),
    ('mp_droppath', float, 0.0, 0.0, None, 0.0, 1.0, 'stack', 'REG-05,LRB-05'),
    ('mp_norm', str, 'layer', 'layer', ('layer', 'graph', 'graph_kind'), None, None, 'stack', 'REG-08,LRB-04'),
    ('mp_update_kinds', str, 'all', 'all', None, None, None, 'stack', 'EVD-03'),
    ('mp_time_decay', int, 0, 0, None, 0, 1, 'stack', 'RDO-05'),
    ('mp_shortcut', int, 0, 0, None, 0, 1, 'stack', 'STR-03'),
    # mp_feed='none' (ablation): no message-passing stack is built; every head reads the node-local state h0
    ('mp_feed', str, 'all', 'all', ('all', 'dual', 'node', 'gates', 'none'), None, None, 'network',
     'EVD-01,EVD-06,EVD-04,ABL-05'),
    ('mp_dual_read', int, 0, 0, None, 0, 1, 'network', 'ERR-01'),
    ('mp_aux_weight', float, 0.3, 0.3, None, 0.0, 10.0, 'adapter', 'ERR-01'),
    ('mp_jk', int, 0, 0, None, 0, 1, 'network', 'REG-06'),
    ('mp_branches', int, 1, 1, None, 1, None, 'network', 'ERR-04'),
    ('mp_branch_width', int, 64, 64, None, 1, None, 'network', 'ERR-04'),
    ('row_drop', float, 0.0, 0.0, None, 0.0, 0.5, 'network', 'RDO-03'),
    ('shared_gate', int, 0, 0, None, 0, 1, 'network', 'RDO-02'),
    ('node_readout', str, 'global', 'global', ('global', 'kind_budget'), None, None, 'network', 'ERR-03'),
    ('edge_head_rank', int, 0, 0, None, 0, None, 'network', 'RDO-04'),
    ('kind_norm', int, 0, 0, None, 0, 1, 'network', 'RDO-07'),
    ('mp_shrink', float, 0.0, 0.0, None, 0.0, None, 'network', 'LCL-04'),
    ('mp_split_rows', int, 0, 0, None, 0, 1, 'network', 'LCL-04,EVD-05'),
    ('mp_mask_rate', float, 0.0, 0.0, None, 0.0, 0.5, 'network', 'LCL-02'),
    ('mp_mask_weight', float, 0.015, 0.015, None, 0.0, None, 'adapter', 'LCL-02'),
    ('mp_gate_decay', float, 0.0, 0.0, None, 0.0, None, 'adapter', 'LRB-02'),
    ('aux_group_map', str, '', '', None, None, None, 'adapter', 'LCL-06'),
    ('aux_group_weight', float, 0.1, 0.1, None, 0.0, None, 'adapter', 'LCL-06'),
    ('mp_consistency', float, 0.0, 0.0, None, 0.0, None, 'adapter', 'LRB-06'),
    ('ple_mode', str, 'hat', 'hat', ('hat', 'robust_cum'), None, None, 'network', 'RDO-06'),
    ('mp_pretrain_checkpoint', str, '', '', None, None, None, 'adapter', 'LCL-07'),
    # Review fixes (mp2rev-train-01/02/05/perf-05). Owner 'adapter' = read by the adapter or by one line
    # of network.py; each default is today's behaviour (DEFAULT == LEGACY).
    ('mp_optimizer', str, 'adam', 'adam', ('adam', 'adamw'), None, None, 'adapter', 'mp2rev-train-02'),
    ('mp_weight_decay', float, 0.01, 0.01, None, 0.0, None, 'adapter', 'mp2rev-train-02'),
    ('mp_lr_schedule', str, 'none', 'none', ('none', 'cosine'), None, None, 'adapter', 'mp2rev-train-01'),
    ('mp_label_smoothing', float, 0.0, 0.0, None, 0.0, 0.3, 'adapter', 'mp2rev-train-01'),
    ('mp_dropout', float, -1.0, -1.0, None, -1.0, None, 'adapter', 'mp2rev-perf-05'),
    # Ablation switches (Nasim section 4). Each default is today's behaviour (DEFAULT == LEGACY), so the
    # final model is unchanged; see network.py for what each switch does to the computation.
    ('ev_combine', str, 'sum', 'sum', ('sum', 'mlp'), None, None, 'network', 'ABL-02'),
    ('ev_pair', int, 1, 1, None, 0, 1, 'network', 'ABL-04'),
    ('ev_edge', int, 1, 1, None, 0, 1, 'network', 'ABL-04b'),
    ('readout', str, 'evidence', 'evidence', ('evidence', 'pool'), None, None, 'network', 'ABL-06'),
    ('pool_hidden', int, 0, 0, None, 0, None, 'network', 'ABL-06'),
)

SPECS: dict[str, dict] = {
    name: {'type': kind, 'default': default, 'legacy': legacy, 'choices': choices, 'minimum': minimum,
           'maximum': _BELOW_ONE if name in EXCLUSIVE_MAXIMUM else maximum, 'owner': owner, 'source': source}
    for name, kind, default, legacy, choices, minimum, maximum, owner, source in _TABLE}
DEFAULTS: dict[str, object] = {name: spec['default'] for name, spec in SPECS.items()}
LEGACY: dict[str, object] = {name: spec['legacy'] for name, spec in SPECS.items()}
STACK_KEYS = tuple(name for name, spec in SPECS.items() if spec['owner'] == 'stack')
NETWORK_KEYS = tuple(name for name, spec in SPECS.items() if spec['owner'] == 'network')
ADAPTER_KEYS = tuple(name for name, spec in SPECS.items() if spec['owner'] == 'adapter')
_FILE_OPTIONS = ('aux_group_map', 'mp_pretrain_checkpoint')


def _convert(name, value, kind):
    if isinstance(value, bool):
        raise ValueError(f'option {name} must be a {kind.__name__}, got a bool')
    if kind is str:
        if not isinstance(value, str):
            raise ValueError(f'option {name} must be a string, got {value!r}')
        return value.strip()
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'option {name} must be a {kind.__name__}, got {value!r}') from None
    if not math.isfinite(number):
        raise ValueError(f'option {name} must be finite, got {value!r}')
    if kind is int:
        if number != int(number):
            raise ValueError(f'option {name} must be an integer, got {value!r}')
        return int(number)
    return number


def _comma_list(name, value, allowed, what, *, allow_all=False, allow_empty=False):
    items = [item.strip() for item in value.split(',')] if value else []
    if allow_all and 'all' in items:
        if items != ['all']:
            raise ValueError(f"option {name}: 'all' cannot be combined with other entries ({value!r})")
        return 'all'
    if not items and not allow_empty:
        raise ValueError(f'option {name} must list at least one {what}')
    if any(not item for item in items):
        raise ValueError(f'option {name} has an empty entry in {value!r}')
    unknown = [item for item in items if item not in allowed]
    if unknown:
        raise ValueError(f'option {name} names unknown {what}(s) {unknown}')
    if len(set(items)) != len(items):
        raise ValueError(f'option {name} lists a {what} twice: {value!r}')
    return ','.join(items)


def validate(raw: dict) -> dict:
    """Typed, range-checked copy of ``raw`` with every missing option set to its DEFAULT."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError('options must be a mapping')
    unknown = sorted(set(raw) - set(SPECS))
    if unknown:
        raise ValueError(f'unknown cei_gnn_mp2 option(s): {unknown}')
    out = dict(DEFAULTS)
    for name, value in raw.items():
        spec = SPECS[name]
        value = _convert(name, value, spec['type'])
        if spec['choices'] is not None and value not in spec['choices']:
            raise ValueError(f'option {name} must be one of {list(spec["choices"])}, got {value!r}')
        if spec['minimum'] is not None and value < spec['minimum']:
            raise ValueError(f'option {name} must be >= {spec["minimum"]}, got {value!r}')
        if spec['maximum'] is not None and value > spec['maximum']:
            bound = '< 1.0' if name in EXCLUSIVE_MAXIMUM else f'<= {spec["maximum"]}'
            raise ValueError(f'option {name} must be {bound}, got {value!r}')
        out[name] = value
    out['mp_hub_kinds'] = _comma_list('mp_hub_kinds', out['mp_hub_kinds'], NODE_KINDS, 'node kind')
    out['mp_update_kinds'] = _comma_list('mp_update_kinds', out['mp_update_kinds'], NODE_KINDS,
                                         'node kind', allow_all=True)
    out['mp_dropedge_exempt'] = _comma_list('mp_dropedge_exempt', out['mp_dropedge_exempt'], _RELATIONS,
                                            'relation', allow_empty=True)
    for name in _FILE_OPTIONS:
        if out[name] and not os.path.isfile(os.path.expanduser(out[name])):
            raise ValueError(f'option {name}: file {out[name]!r} does not exist')
    _check_combinations(out)
    return out


_ATTENTION_KINDS = ('simplehgn', 'hgt')
_GINE_ONLY = ('mp_msg_dim', 'mp_rank', 'mp_tie_directions', 'mp_edge_gate', 'mp_rev_attention', 'mp_hub_max',
              'mp_relation_film', 'mp_root', 'mp_rev_agg', 'mp_ctx')   # layers.py _parse, non-gine branch


def ignored_options(options: dict) -> dict:
    """{option: reason} for every option set away from its DEFAULT while the switch it needs is off."""
    ignored = {}

    def note(name, reason):
        if options[name] != DEFAULTS[name]:
            ignored[name] = reason
    if options['mp_dual_read'] != 1:
        note('mp_aux_weight', 'needs mp_dual_read=1')
    if not options['mp_mask_rate'] > 0:
        note('mp_mask_weight', 'needs mp_mask_rate>0')
    if not options['aux_group_map']:
        note('aux_group_weight', 'needs aux_group_map')
    # (mp2v3-fixes-04): judge mp_hub_kinds by the hub_max the stack really
    # uses (layers._parse resets it to 0 for every kind but gine)
    if options['mp_hub_max'] != 1:
        note('mp_hub_kinds', 'needs mp_hub_max=1')
    elif options['mp_kind'] != 'gine':
        note('mp_hub_kinds', "needs mp_hub_max=1 with mp_kind='gine' (the stack resets mp_hub_max for the other kinds)")
    # no residual scale exists under mp_residual='none'
    if options['mp_residual'] == 'none':
        note('mp_residual_init', "needs mp_residual != 'none'")
    # the stack ORs the exempt set in only inside `if dropedge > 0`;
    # the mp_kg_edge_drop branch uses `~kg` and never reads it
    if not options['mp_dropedge'] > 0:
        note('mp_dropedge_exempt', 'needs mp_dropedge>0')
    # (mp2v3-model-04, mp2rev-layers-08): mp_aggr only groups the mean aggregate
    if options['mp_degree_norm'] == 'sym':
        note('mp_aggr', "needs mp_degree_norm != 'sym' (sym normalises over all relations, no group is read)")
    elif options['mp_kind'] in _ATTENTION_KINDS:
        note('mp_aggr', "needs mp_kind in ('gine', 'rgcn', 'diag') (attention kinds use one softmax per receiver)")
    if options['mp_branches'] == 1:
        note('mp_branch_width', 'needs mp_branches>1')
    if options['mp_kind'] != 'rgcn':
        note('mp_bases', "needs mp_kind='rgcn'")
    if options['mp_kind'] not in _ATTENTION_KINDS:
        note('mp_heads', "needs mp_kind in ('simplehgn', 'hgt')")
    if options['mp_kind'] not in _ATTENTION_KINDS and not (options['mp_kind'] == 'gine'
                                                           and options['mp_rev_attention'] == 1):
        note('mp_att_dropout', 'needs an attention stack (mp_rev_attention=1 or simplehgn/hgt)')
    if options['mp_optimizer'] != 'adamw':
        note('mp_weight_decay', "needs mp_optimizer='adamw'")
    if options['mp_feed'] == 'none':   # no stack is built: every stack option is moot
        for name in _NEEDS_STACK:
            note(name, "needs a message-passing stack (mp_feed != 'none')")
        note('mp_aux_weight', "needs mp_feed != 'none'")
    if options['readout'] != 'pool':
        note('pool_hidden', "needs readout='pool'")
    if options['readout'] == 'pool':
        for name in ('node_readout', 'edge_head_rank', 'mp_hub_kinds'):
            note(name, "readout='pool' has no evidence heads")
    if options['ev_edge'] == 0:
        note('edge_head_rank', 'needs ev_edge=1')
    if options['mp_kind'] != 'gine':   # mp2v2-train-06: layers.py _parse resets these for the other kinds
        for name in _GINE_ONLY:
            note(name, "needs mp_kind='gine' (reset by the stack for rgcn, diag, simplehgn, hgt)")
    return ignored


# options that act on (or through) the message-passing stack: meaningless when the stack is not built
_NEEDS_STACK = tuple(STACK_KEYS)


def _check_ablation(options: dict) -> None:
    """Combinations of the ablation switches (ev_combine, ev_pair, ev_edge, readout, mp_feed='none')."""
    if options['mp_feed'] == 'none':
        for name, bad in (('mp_jk', 1), ('mp_split_rows', 1), ('mp_time_decay', 1)):
            if options[name] == bad:
                raise ValueError(f"{name}={bad} contradicts mp_feed='none' (no message-passing stack is built)")
        for name, limit in (('mp_branches', 1), ('mp_shrink', 0.0), ('mp_mask_rate', 0.0)):
            if options[name] != limit:
                raise ValueError(f"{name}={options[name]!r} contradicts mp_feed='none' (no message-passing stack is built)")
        if options['mp_pretrain_checkpoint']:
            raise ValueError("mp_pretrain_checkpoint contradicts mp_feed='none' (no message-passing stack to load into)")
    if options['readout'] == 'pool':
        if options['mp_feed'] != 'all':
            raise ValueError(f"readout='pool' needs mp_feed='all' (got {options['mp_feed']!r}): the pooled head reads "
                             'the message-passing state and no evidence row exists')
        if options['pool_hidden'] < 1:
            raise ValueError("readout='pool' needs pool_hidden >= 1")
        for name, off in (('ev_combine', 'sum'), ('ev_pair', 1), ('ev_edge', 1), ('mp_dual_read', 0),
                          ('mp_split_rows', 0), ('mp_branches', 1), ('shared_gate', 0), ('row_drop', 0.0)):
            if options[name] != off:
                raise ValueError(f"{name}={options[name]!r} contradicts readout='pool' (the pooled head has no "
                                 'evidence rows)')
    if options['ev_combine'] == 'mlp':
        for name, off in (('mp_dual_read', 0), ('mp_split_rows', 0)):
            if options[name] != off:
                raise ValueError(f"{name}={options[name]!r} contradicts ev_combine='mlp' (the block totals are not "
                                 'summed, so the additive ledger rows it needs do not exist)')
    if options['ev_pair'] == 0 and options['mp_dual_read'] == 1:
        raise ValueError('ev_pair=0 contradicts mp_dual_read=1 (the message-state pair head would stay)')
    if options['ev_edge'] == 0:
        for name, off in (('mp_dual_read', 0), ('shared_gate', 0)):
            if options[name] != off:
                raise ValueError(f"{name}={options[name]!r} contradicts ev_edge=0 (it needs the edge heads)")
        if options['mp_feed'] == 'gates':
            raise ValueError("mp_feed='gates' contradicts ev_edge=0 (the gates feed the edge gate)")


def _check_combinations(options: dict) -> None:
    _check_ablation(options)
    if options['mp_dual_read'] == 1 and options['mp_feed'] != 'all':
        raise ValueError(f"mp_dual_read=1 contradicts mp_feed={options['mp_feed']!r} "
                         "(dual read needs mp_feed=all)")
    if options['mp_residual_bound'] == 1 and options['mp_residual'] == 'none':
        raise ValueError("mp_residual_bound=1 contradicts mp_residual='none' (no residual scale to bound)")
    if options['mp_gate_decay'] > 0 and options['mp_residual'] == 'none':
        raise ValueError("mp_gate_decay>0 contradicts mp_residual='none' (no residual scale to decay)")
    if (options['mp_kind'] == 'gine' and options['mp_rev_attention'] == 1
            and options['mp_rev_agg'] == 'meanmax'):
        raise ValueError("mp_rev_attention=1 contradicts mp_rev_agg='meanmax' (both define the reverse "
                         'aggregate; use one of them)')
    if -1.0 < options['mp_dropout'] < 0.0:
        raise ValueError(f"mp_dropout must be -1 (= the network dropout) or in [0, 1), got {options['mp_dropout']!r}")
    if options['mp_shrink'] > 0 and options['mp_mask_rate'] > 0:   # mp2rev-train-06
        warnings.warn('mp_shrink>0 with mp_mask_rate>0: the shrink term pulls h_msg toward h0 at the masked '
                      'nodes too, while the masked-node reconstruction needs h_msg to move away from the '
                      'emptied h0 there (opposing pulls)', UserWarning, stacklevel=3)
    if options['mp_gate_decay'] > 0 and options['mp_shrink'] > 0:   # mp2rev-train-06
        warnings.warn('mp_gate_decay>0 with mp_shrink>0: both shrink alpha * update toward 0 (two knobs for '
                      'one quantity)', UserWarning, stacklevel=3)
    for name, reason in ignored_options(options).items():
        if name != 'mp_att_dropout':   # warned by the stack (layers.py) when it is built
            warnings.warn(f'{name}={options[name]!r} has no effect: {reason}', UserWarning, stacklevel=3)


def _stack_width(options: dict, hidden: int) -> int:
    return min(int(options['mp_branch_width']), int(hidden)) if options['mp_branches'] > 1 else int(hidden)


def check_build(options: dict, *, hidden: int, dropout: float) -> None:
    """Checks of a validated option dict that need the width or the dropout rate (adapter-side)."""
    width = _stack_width(options, hidden)
    if options['mp_kind'] in ('simplehgn', 'hgt') and width % options['mp_heads']:
        where = (f"min(mp_branch_width={options['mp_branch_width']}, hidden={hidden})"
                 if options['mp_branches'] > 1 else f'hidden={hidden}')
        raise ValueError(f"mp_kind={options['mp_kind']!r} needs the stack width {width} ({where}) "
                         f"divisible by mp_heads={options['mp_heads']}")
    if options['mp_consistency'] > 0:   # LRB-06 compares two stochastic views
        attention = options['mp_kind'] in ('simplehgn', 'hgt') or options['mp_rev_attention'] == 1
        sources = [name for name in ('mp_dropedge', 'mp_kg_edge_drop', 'mp_dropmsg', 'mp_dropnode',
                                     'mp_droppath', 'row_drop', 'mp_mask_rate') if options[name] > 0]
        if attention and options['mp_att_dropout'] > 0:
            sources.append('mp_att_dropout')
        if dropout > 0 or options['mp_dropout'] > 0:
            sources.append('dropout')
        if not sources:
            raise ValueError('mp_consistency>0 contradicts dropout=0 with mp_dropedge, mp_kg_edge_drop, '
                             'mp_dropmsg, mp_dropnode, mp_droppath, row_drop, mp_mask_rate all 0 (the two '
                             'views would be identical, the term is exactly 0)')


def effective_options(options: dict, *, hidden: int, dropout: float | None = None) -> dict:
    """Values the build actually uses where an option is capped or means "full width".

    ``dropout`` (the network rate) resolves ``mp_dropout=-1`` to the rate the stack applies.
    """
    width = _stack_width(options, hidden)

    def capped(value, limit):
        return int(value) if 0 < int(value) < limit else int(limit)

    gine = options['mp_kind'] == 'gine'   # the gine-only message options are ignored by the other kinds
    msg_dim = capped(options['mp_msg_dim'], width) if gine else width
    # layers._map factorises a map only if 0 < rank < min(in, out). The
    # message maps are width -> msg_dim, the global map (mp_global > 0) is width -> width; the widest of the
    # built maps is the "full rank" the value is capped at.
    widest = max(msg_dim, width if options['mp_global'] > 0 else 0)
    effective = {'mp_stack_width': width,
                 'mp_branch_width': width if options['mp_branches'] > 1 else None,
                 'mp_msg_dim': msg_dim,
                 'mp_rank': capped(options['mp_rank'], widest) if gine else width,
                 'edge_head_rank': capped(options['edge_head_rank'], hidden)}
    if dropout is not None:   # -1 is "the network dropout"
        effective['mp_dropout'] = float(dropout) if options['mp_dropout'] < 0 else float(options['mp_dropout'])
    return effective
