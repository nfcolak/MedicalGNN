"""Evidence removal as real node deletion, for the fidelity metrics (contract version 5).

A node is removed from the raw graph RECORD (graphs.jsonl line + visit-membership sidecar row), the
record is re-encoded with the run's own fitted preprocessing (`core.tensorize.encode_graph`) and the
trained model scores it with its own forward. Nothing of the removed node survives: not its feature
row, token, node type, visit membership, pooling share or (for CEI) its pairs, and an item it was the
only evidence of becomes absent at the index visit, as on a graph built without it.

Rules, in this order:

1. Only eligible evidence nodes (complaint, vital, measurement, diagnosis) may be selected.
2. The selected nodes and every edge with a deleted endpoint are dropped.
3. Cascade, repeated until nothing changes: an analyte with no `instance_of` edge left from a kept
   measurement is dropped (the graph builder creates an analyte only for a measured lab); a knowledge
   node with no edge left to a kept non-knowledge node is dropped (knowledge nodes exist only through
   their anchors: complaints, prior diagnoses, analytes, vitals and the measurements/vitals that fired
   a threshold state). Then the source-masking orphan rule of `comparison/s7_why/masking.py`
   (`apply_mask`): every non-global node other than the index visit left without any edge is dropped.
   The patient (global) node and the index visit node are never dropped.
4. Membership as if the removed evidence had never been recorded: the pairs of dropped nodes go, and
   every surviving concept node takes the visits of its surviving anchors by the graph builder's rule
   (an analyte: the visits of its measurements, prior labs included; a knowledge node: the visits of
   its anchors). Then the live-visit renumbering of `apply_mask` (non-history masks): the live
   ordinals are the index ordinal plus every ordinal of a kept non-global node, renumbered in order;
   node ids become 'n<position>' in the original order. With `cascade=False` concept nodes keep
   their recorded membership.

Remaining edges are not relinked: deleting the middle measurement of a prior-lab trajectory leaves the
chain broken (its predecessor gains no edge to its successor).
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

import torch

from core import NODE_KINDS
from core.explain.fidelity_v2 import EVIDENCE_NODE_KINDS

INSTANCE_OF = 'instance_of'           # measurement -> analyte (data/s4_graph/graph.py)
ANALYTE, KNOWLEDGE, MEASUREMENT = 'analyte', 'knowledge', 'measurement'
ENCODED_FIELDS = ('x', 'edge_index', 'edge_attr', 'token', 'node_type', 'edge_relation', 'edge_triple',
                  'edge_payload', 'visit_membership_index', 'num_visits', 'global_node_mask')


def _empty_info(graph: dict) -> dict[str, Any]:
    return {'selected': 0, 'cascaded': {'analyte': 0, 'knowledge': 0, 'orphan': 0},
            'visits_dropped': 0, 'nodes_after': len(graph['nodes']), 'edges_after': len(graph['edges'])}


def delete_from_record(graph: dict, membership: dict, selected: Iterable[int], *,
                       cascade: bool = True) -> tuple[dict, dict, dict]:
    """(graph', membership', info) with the nodes at the `selected` record positions deleted.

    Positions are the node indices of the encoded tensor (`encode_graph` keeps record order). The
    inputs are never modified.
    """
    nodes, edges = graph['nodes'], graph['edges']
    n = len(nodes)
    pos_of = {node['id']: i for i, node in enumerate(nodes)}
    if len(pos_of) != n:
        raise ValueError('graph record has duplicate node ids; positions are not node indices')
    if any(e['source'] not in pos_of or e['target'] not in pos_of for e in edges):
        raise ValueError('graph record has an edge with an unknown endpoint')
    glob = membership['global_node_mask']
    if len(glob) != n:
        raise ValueError('membership global_node_mask does not match the graph nodes')
    ordinals = membership['visit_ordinals']
    last = len(ordinals) - 1
    member: list[set[int]] = [set() for _ in range(n)]
    for ordinal, pos in membership['membership_pairs']:
        if not 0 <= pos < n:
            raise ValueError('membership pair points outside the graph nodes')
        member[pos].add(ordinal)
    visit_pos = [i for i, node in enumerate(nodes) if node['kind'] == 'visit']
    if len(visit_pos) != 1 or member[visit_pos[0]] != {last}:
        raise ValueError('deletion undefined: index visit node is not the unique member of the last ordinal')
    index_visit = visit_pos[0]

    remove: set[int] = set()
    for value in selected:
        if isinstance(value, bool) or int(value) != value or not 0 <= int(value) < n:
            raise ValueError(f'selected node {value!r} is not a node position of this graph')
        position = int(value)
        if nodes[position]['kind'] not in EVIDENCE_NODE_KINDS:
            raise ValueError(f"node {position} of kind {nodes[position]['kind']!r} is not eligible evidence "
                             f'(eligible: {list(EVIDENCE_NODE_KINDS)})')
        remove.add(position)
    if not remove:
        return graph, membership, _empty_info(graph)
    selected_count = len(remove)

    ends = [(pos_of[e['source']], pos_of[e['target']], e['relation']) for e in edges]

    def kept_edges():
        return [(s, t, r) for s, t, r in ends if s not in remove and t not in remove]

    cascaded = {'analyte': 0, 'knowledge': 0, 'orphan': 0}
    if cascade:
        while True:
            live = kept_edges()
            anchored_analytes = {t for s, t, r in live
                                 if r == INSTANCE_OF and nodes[s]['kind'] == MEASUREMENT}
            anchored_knowledge = set()
            for s, t, _ in live:
                if nodes[s]['kind'] != KNOWLEDGE and nodes[t]['kind'] == KNOWLEDGE:
                    anchored_knowledge.add(t)
                if nodes[t]['kind'] != KNOWLEDGE and nodes[s]['kind'] == KNOWLEDGE:
                    anchored_knowledge.add(s)
            gone = {'analyte': {i for i in range(n) if i not in remove and nodes[i]['kind'] == ANALYTE
                                and i not in anchored_analytes},
                    'knowledge': {i for i in range(n) if i not in remove and nodes[i]['kind'] == KNOWLEDGE
                                  and i not in anchored_knowledge}}
            if not gone['analyte'] and not gone['knowledge']:
                break
            for kind, positions in gone.items():
                cascaded[kind] += len(positions)
                remove |= positions
        degree = [0] * n
        for s, t, _ in kept_edges():
            degree[s] += 1
            degree[t] += 1
        orphans = {i for i in range(n)
                   if i not in remove and not glob[i] and i != index_visit and degree[i] == 0}
        cascaded['orphan'] = len(orphans)
        remove |= orphans
        # concept nodes take the visits of their surviving anchors, as the graph builder assigns them:
        # analytes first (visits of their measurements, prior labs included), then knowledge nodes
        # (visits of their anchors, an analyte anchor with its recomputed visits)
        live = kept_edges()
        member = list(member)
        for i in range(n):
            if i not in remove and nodes[i]['kind'] == ANALYTE:
                member[i] = set().union(*(member[s] for s, t, r in live
                                          if t == i and r == INSTANCE_OF and nodes[s]['kind'] == MEASUREMENT))
        for i in range(n):
            if i not in remove and nodes[i]['kind'] == KNOWLEDGE:
                anchors = ({s for s, t, _ in live if t == i} | {t for s, t, _ in live if s == i})
                member[i] = set().union(*(member[a] for a in anchors if nodes[a]['kind'] != KNOWLEDGE))

    keep = [i for i in range(n) if i not in remove]
    new_pos = {old: new for new, old in enumerate(keep)}
    new_id = {nodes[old]['id']: f'n{new}' for old, new in new_pos.items()}
    new_nodes = [{**nodes[old], 'id': f'n{new}'} for old, new in new_pos.items()]
    new_edges = [{**e, 'source': new_id[e['source']], 'target': new_id[e['target']]}
                 for e in edges if e['source'] in new_id and e['target'] in new_id]
    live_ordinals = {last} | {o for old in keep for o in member[old] if not glob[old]}
    order = sorted(live_ordinals)
    remap = {o: i for i, o in enumerate(order)}
    pairs = sorted({(remap[o], new_pos[old]) for old in keep for o in member[old] if o in remap})
    new_membership = {**membership,
                      'visit_ordinals': list(range(len(order))),
                      'membership_pairs': [list(p) for p in pairs],
                      'global_node_mask': [bool(glob[old]) for old in keep]}
    new_graph = {**graph, 'nodes': new_nodes, 'edges': new_edges}
    info = {'selected': selected_count, 'cascaded': cascaded, 'visits_dropped': len(ordinals) - len(order),
            'nodes_after': len(new_nodes), 'edges_after': len(new_edges)}
    return new_graph, new_membership, info


def encoding_differences(data, reference) -> list[str]:
    """Names of the encoded attributes in which two single-graph encodings differ (torch.equal)."""
    differing = []
    for name in ENCODED_FIELDS:
        a, b = getattr(data, name, None), getattr(reference, name, None)
        if a is None and b is None:
            continue
        if a is None or b is None or a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
            differing.append(name)
    return differing


class DeletionScorer:
    """Class probabilities of one visit after deleting a set of evidence nodes.

    `forward(data) -> logits [1, C]` is the model's own forward on one encoded graph (eval mode, the
    adapter on `Batch.from_data_list([data])`); `encode(graph, membership)` is the run's
    `encode_graph` with its fitted preprocessing and edge view. The untouched record is encoded once
    (`self.data`); `probs(deleted)` re-encodes the record without `deleted` (and its cascade) and is
    cached per node set. The empty set returns the untouched graph's probabilities.
    """

    def __init__(self, forward: Callable, graph: dict, membership: dict, encode: Callable, *, y=None):
        self.forward = forward
        self.graph, self.membership = graph, membership
        self.encode = encode
        self.y = None if y is None else int(y)
        self.data = self._encode(graph, membership)
        kinds = [NODE_KINDS.index(node['kind']) for node in graph['nodes']]
        if (self.data.num_nodes != len(kinds)
                or self.data.node_type.tolist() != kinds):
            raise ValueError('encoded node indices differ from the record node positions')
        self.node_type = self.data.node_type
        self.last_info = _empty_info(graph)
        self._cache: dict[frozenset, tuple[torch.Tensor, dict]] = {}
        self._cache[frozenset()] = (self._score(self.data), self.last_info)

    def _encode(self, graph, membership):
        data = self.encode(graph, membership)
        if self.y is not None:
            data.y = torch.tensor([self.y], dtype=torch.long)
        return data

    def _score(self, data) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(data)
        if logits.ndim != 2 or logits.size(0) != 1:
            raise ValueError('the deletion forward must score exactly one graph')
        return torch.softmax(logits, dim=-1)

    def check_against(self, reference) -> None:
        """Refuse when the record's encoding is not the tensor the model was evaluated on."""
        differing = encoding_differences(self.data, reference)
        if differing:
            raise ValueError(f'the re-encoded record differs from the split tensor in {differing}')

    def probs(self, deleted: Iterable[int] = ()) -> torch.Tensor:
        key = frozenset(int(i) for i in deleted)
        if key not in self._cache:
            graph, membership, info = delete_from_record(self.graph, self.membership, sorted(key))
            self._cache[key] = (self._score(self._encode(graph, membership)), info)
        probs, self.last_info = self._cache[key]
        return probs
