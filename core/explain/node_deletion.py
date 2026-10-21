"""Evidence removal as real node deletion, for the fidelity metrics (contract version 6).

A node is removed from the raw graph RECORD (graphs.jsonl line + visit-membership sidecar row), the
record is re-encoded with the run's own fitted preprocessing (`core.tensorize.encode_graph`) and the
trained model scores it with its own forward. Nothing of the removed node survives: not its feature
row, token, node type, visit membership, pooling share or (for CEI) its pairs, and a concept it was the
only provenance of becomes absent. The result is the record with these nodes deleted, not the record
the graph builder would produce without the events: prior labs of an analyte whose index labs are all
deleted stay (with the analyte and its knowledge nodes), where the builder pulls prior history only
for analytes measured at the index visit, and no edge is relinked (a `baseline_of` edge is not moved
to the next index lab or to the previous prior lab, a broken trajectory chain stays broken).

Rules, in this order:

1. Only eligible evidence nodes (complaint, vital, measurement, diagnosis) may be selected.
2. The selected nodes and every edge with a deleted endpoint are dropped.
3. Combo threshold states follow combo provenance. The graph builder fires a combo rule only when
   every row of the rule holds on the latest index value of its token, and then adds one
   `medical:indicates` edge per contributing node, all with the rule's `rule_id`. The edges of one
   (state node, rule_id) form a group: deleting any anchor of a group deletes the whole group. When a
   deleted node was the latest index value of a combo token, the rule is evaluated again on the
   surviving index values with the builder's own `_row_holds` (the next-latest value takes the
   deleted one's place); a rule that holds on the surviving values gets its edges (and its state node
   when the record has none) as the builder would add them, so deleting a latest value that failed a
   rule can add a state the record did not have (counted in `info['combos']['new_state_nodes']`).
   This needs the threshold table the graph artifact was built with
   (`load_combo_rules`) and every input the builder reads from the record (value, unit, time,
   censoring sign and reference bounds of a lab, the patient's gender); when an input is missing the
   group is deleted without re-evaluation and the case is reported in `info['combos']`. Without a
   table, a group whose anchors carry different tokens is treated as a combo and is never
   re-evaluated.
4. Cascade, repeated until nothing changes: an analyte with no `instance_of` edge left from a kept
   measurement is dropped (the graph builder creates an analyte only for a measured lab); a knowledge
   node with no edge left to a kept non-knowledge node is dropped (knowledge nodes exist only through
   their anchors: complaints, prior diagnoses, analytes, vitals and the measurements/vitals that fired
   a threshold state). Then the source-masking orphan rule of `comparison/s7_why/masking.py`
   (`apply_mask`): every non-global node other than the index visit left without any edge is dropped.
   The patient (global) node and the index visit node are never dropped.
5. Membership as if the removed evidence had never been recorded: the pairs of dropped nodes go, and
   every surviving concept node takes the visits of its surviving anchors by the graph builder's rule
   (an analyte: the visits of its measurements, prior labs included; a knowledge node: the visits of
   its anchors). Then the visits are renumbered: a prior visit the deletion emptied (it had a
   non-global node before and has none after) is dropped, every other visit is kept, including a
   prior visit that was empty in the record already (the builder numbers every prior stay); the kept
   ordinals are renumbered in order and node ids become 'n<position>' in the original order. With
   `cascade=False` steps 3 and 4 are skipped and concept nodes keep their recorded membership.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from core import NODE_KINDS
from core.explain.fidelity_v2 import EVIDENCE_NODE_KINDS
from core.schema import read_thresholds, sha256 as file_sha256
from data.s4_graph.graph import LEVEL_SOURCES, _row_holds

INSTANCE_OF = 'instance_of'           # measurement -> analyte (data/s4_graph/graph.py)
INDICATES = 'medical:indicates'       # measurement/vital -> threshold state, with the rule's rule_id
ANALYTE, KNOWLEDGE, MEASUREMENT, VITAL = 'analyte', 'knowledge', 'measurement', 'vital'
ENCODED_FIELDS = ('x', 'edge_index', 'edge_attr', 'token', 'node_type', 'edge_relation', 'edge_triple',
                  'edge_payload', 'visit_membership_index', 'num_visits', 'global_node_mask')
COMBO_SOURCE_TABLE = 'threshold_table'
COMBO_SOURCE_FALLBACK = 'distinct_anchor_tokens'
# what the builder reads from a lab record besides value/unit/time (present on lab nodes of v5 graphs)
LAB_COMBO_INPUTS = ('censored', 'ref_upper', 'ref_lower')
PROVENANCE_FIELDS = ('source_uri', 'source_version', 'reviewed_by')


class ComboRules:
    """The combo rows (`kind == combo`) of the threshold table a graph artifact was built with, and the
    artifact's `level_source`, which `_row_holds` needs to evaluate them as the builder did."""

    def __init__(self, rows: Iterable[dict], *, level_source: str, path: str | None = None,
                 sha256: str | None = None, sha256_checked: bool = False):
        if level_source not in LEVEL_SOURCES:
            raise ValueError(f'unknown level source {level_source!r} (known: {list(LEVEL_SOURCES)})')
        rows = tuple(dict(row) for row in rows)
        if any(row['kind'] != 'combo' for row in rows):
            raise ValueError('combo rules take only rows of kind combo')
        by_rule: dict[str, list[dict]] = defaultdict(list)
        for row in rows:          # table order within a rule, as the builder groups them
            by_rule[row['rule_id']].append(row)
        self.rows = rows
        self.by_rule = {rule_id: tuple(by_rule[rule_id]) for rule_id in sorted(by_rule)}
        self.level_source = level_source
        self.path, self.sha256, self.sha256_checked = path, sha256, sha256_checked

    def describe(self) -> dict[str, Any]:
        return {'source': COMBO_SOURCE_TABLE, 'path': self.path, 'sha256': self.sha256,
                'sha256_checked': self.sha256_checked, 'level_source': self.level_source,
                'combo_rules': len(self.by_rule), 'combo_rows': len(self.rows)}


def load_combo_rules(path, sha256: str | None, *, level_source: str) -> ComboRules:
    """The combo rules of the threshold table at `path` (read with `core.schema.read_thresholds`).

    `sha256` is the table's recorded hash; a file with another hash is refused. None means no hash
    was recorded: the file's own hash is then kept (`sha256_checked` False).
    """
    path = Path(path)
    actual = file_sha256(path)
    if sha256 is not None and actual != sha256:
        raise ValueError(f'threshold table {path.name} has sha256 {actual}, the recorded one is {sha256}')
    rows = [row for row in read_thresholds(path) if row['kind'] == 'combo']
    return ComboRules(rows, level_source=level_source, path=str(path), sha256=actual,
                      sha256_checked=sha256 is not None)


def combo_source(combo_rules: ComboRules | None) -> dict[str, Any]:
    """Which combo rule a deletion used, for the record."""
    if combo_rules is not None:
        return combo_rules.describe()
    return {'source': COMBO_SOURCE_FALLBACK,
            'rule': ('the medical:indicates edges of one (state node, rule_id) whose anchors carry '
                     'different tokens form a combo group; no re-evaluation on an earlier value')}


class _NotReevaluated(Exception):
    """A combo rule the deletion cannot evaluate again from the record; the message says why."""


def _combo_info(combo_rules: ComboRules | None) -> dict[str, Any]:
    return {'source': COMBO_SOURCE_TABLE if combo_rules is not None else COMBO_SOURCE_FALLBACK,
            'groups_deleted': 0, 'reevaluated': 0, 'refired': 0, 'new_state_nodes': 0,
            'not_reevaluated': []}


def _originally_empty(membership: dict) -> set[int]:
    glob = membership['global_node_mask']
    used = {ordinal for ordinal, pos in membership['membership_pairs'] if not glob[pos]}
    return set(range(len(membership['visit_ordinals']))) - used


def _empty_info(graph: dict, membership: dict, combo_rules: ComboRules | None = None) -> dict[str, Any]:
    return {'selected': 0, 'cascaded': {'analyte': 0, 'knowledge': 0, 'orphan': 0},
            'combos': _combo_info(combo_rules), 'visits_dropped': 0,
            'visits_originally_empty': len(_originally_empty(membership)),
            'nodes_after': len(graph['nodes']), 'edges_after': len(graph['edges'])}


def _patient_gender(nodes: list[dict]):
    patients = [node for node in nodes if node['kind'] == 'patient']
    if len(patients) != 1 or 'gender' not in patients[0]:
        raise _NotReevaluated('the patient gender is not in the record')
    return patients[0]['gender']


def _combo_fires(rows, candidates: list[int], nodes: list[dict], gender, level_source: str) -> dict[int, dict]:
    """{anchor position: row} when the rule fires on the index values at `candidates`, as the graph
    builder evaluates it (every row on the latest value of its token, ties: the later node), else {}."""
    contributing = {}
    for row in rows:
        same_token = [i for i in candidates if nodes[i]['token'] == row['source_token']]
        for i in same_token:
            if 'unit' not in nodes[i]:
                raise _NotReevaluated(f'the unit of an index value of {row["source_token"]} is not in the record')
        same_token = [i for i in same_token if nodes[i]['unit'] == row['unit']]
        if not same_token:
            return {}
        for i in same_token:
            if nodes[i].get('time_hours') is None:
                raise _NotReevaluated(f'the time of an index value of {row["source_token"]} is not in the record')
        latest = max(same_token, key=lambda i: (nodes[i]['time_hours'], i))
        node = nodes[latest]
        if 'value' not in node:
            raise _NotReevaluated(f'the value of the latest {row["source_token"]} is not in the record')
        extras = {key: None for key in LAB_COMBO_INPUTS}       # the builder passes None for a vital
        if node['kind'] == MEASUREMENT:
            missing = [key for key in LAB_COMBO_INPUTS if key not in node]
            if missing:
                raise _NotReevaluated(f'{", ".join(missing)} of the latest {row["source_token"]} '
                                      'not in the record')
            extras = {key: node[key] for key in LAB_COMBO_INPUTS}
        if not _row_holds(row, node['token'], node['unit'], node['value'], gender, extras['censored'],
                          node.get('level'), extras['ref_upper'], level_source, 'combo', extras['ref_lower']):
            return {}
        contributing[latest] = row
    return contributing


def _combo_groups(nodes: list[dict], work: list[tuple], remove: set[int],
                  combo_rules: ComboRules | None, info: dict) -> tuple[list[dict], list[tuple]]:
    """(nodes, edges) after rule 3: combo groups with a deleted anchor go, combo rules whose latest
    index value was deleted are evaluated again. `work` holds (source, target, relation, edge) by
    position; a new state node is appended to the nodes, new edges take the builder's place (combo
    rules in rule_id order, anchors in node order)."""
    nodes = list(nodes)
    dead: set[int] = set()
    inserted: dict[int, list[tuple]] = defaultdict(list)
    if combo_rules is None:
        groups: dict[tuple, list[int]] = defaultdict(list)
        for j, (s, t, r, e) in enumerate(work):
            if r == INDICATES and e.get('rule_id') is not None:
                groups[(t, e['rule_id'])].append(j)
        for (_, rule_id), members in groups.items():
            if len({nodes[work[j][0]]['token'] for j in members}) < 2:
                continue        # a state rule: its edges are alternatives
            if any(work[j][0] in remove for j in members):
                dead.update(members)
                info['groups_deleted'] += 1
                info['not_reevaluated'].append({'rule_id': rule_id, 'reason': 'no threshold table given'})
        return nodes, [entry for j, entry in enumerate(work) if j not in dead]

    groups = defaultdict(list)
    for j, (s, t, r, e) in enumerate(work):
        if r == INDICATES and e.get('rule_id') in combo_rules.by_rule:
            groups[e['rule_id']].append(j)
    candidates = [i for i, node in enumerate(nodes)
                  if node['kind'] == VITAL or (node['kind'] == MEASUREMENT
                                               and node.get('scope', 'index') == 'index')]
    deleted_tokens = {nodes[i]['token'] for i in candidates if i in remove}
    alive = [i for i in candidates if i not in remove]
    for rule_id, rows in combo_rules.by_rule.items():
        group = groups.get(rule_id, [])
        hit_anchor = any(work[j][0] in remove for j in group)
        if not hit_anchor and not deleted_tokens & {row['source_token'] for row in rows}:
            continue
        try:
            gender = _patient_gender(nodes)
            before = _combo_fires(rows, candidates, nodes, gender, combo_rules.level_source)
            if set(before) != {work[j][0] for j in group} or len({work[j][1] for j in group}) > 1:
                raise _NotReevaluated('the recorded edges differ from the rule evaluated on the record')
            after = _combo_fires(rows, alive, nodes, gender, combo_rules.level_source)
        except _NotReevaluated as reason:
            if hit_anchor:
                dead.update(group)
                info['groups_deleted'] += 1
            info['not_reevaluated'].append({'rule_id': rule_id, 'reason': str(reason)})
            continue
        info['reevaluated'] += 1
        if after == before:
            continue            # the deleted value was not the latest of its token
        dead.update(group)
        if not after:
            info['groups_deleted'] += int(bool(group))
            continue
        info['refired'] += 1
        target_token = rows[0]['target_token']
        if group:
            target = work[group[0]][1]
        else:
            existing = [i for i, node in enumerate(nodes)
                        if node['kind'] == KNOWLEDGE and node['token'] == target_token]
            if existing:
                target = existing[0]
            else:
                target = len(nodes)
                nodes.append({'id': f'new:{target_token}', 'kind': KNOWLEDGE, 'token': target_token,
                              'time_hours': None, 'available_hours': None})
                info['new_state_nodes'] += 1
        recorded = {work[j][0]: work[j][3] for j in group}
        entries = []
        for anchor in sorted(after):
            row = after[anchor]
            edge = recorded.get(anchor) or {
                'source': nodes[anchor]['id'], 'target': nodes[target]['id'], 'relation': INDICATES,
                'informative': True, 'rule_id': rule_id,
                'provenance': {key: row[key] for key in PROVENANCE_FIELDS}}
            entries.append((anchor, target, INDICATES, edge))
        later = [j for rid in combo_rules.by_rule if rid > rule_id for j in groups.get(rid, [])]
        slot = group[0] if group else min(later, default=len(work))
        inserted[slot].extend(entries)
    edges = []
    for j, entry in enumerate(work):
        edges.extend(inserted.get(j, ()))
        if j not in dead:
            edges.append(entry)
    edges.extend(inserted.get(len(work), ()))
    return nodes, edges


def delete_from_record(graph: dict, membership: dict, selected: Iterable[int], *,
                       cascade: bool = True, combo_rules: ComboRules | None = None
                       ) -> tuple[dict, dict, dict]:
    """(graph', membership', info) with the nodes at the `selected` record positions deleted.

    Positions are the node indices of the encoded tensor (`encode_graph` keeps record order). The
    inputs are never modified. `combo_rules` are the combo rows of the artifact's threshold table
    (`load_combo_rules`); None uses the fallback combo rule (rule 3). `info['combos']` says which was
    used, how many combo groups went, how many rules were evaluated again (and fired again), how many
    state nodes the deletion added that the record did not have (`new_state_nodes`) and which could
    not be evaluated again, with the reason.
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
        return graph, membership, _empty_info(graph, membership, combo_rules)
    selected_count = len(remove)

    work = [(pos_of[e['source']], pos_of[e['target']], e['relation'], e) for e in edges]
    combos = _combo_info(combo_rules)
    glob = list(glob)
    if cascade:
        nodes, work = _combo_groups(nodes, work, remove, combo_rules, combos)
        for _ in range(len(nodes) - n):           # a combo state node the builder would add
            member.append(set())
            glob.append(False)
        n = len(nodes)

    def kept_edges():
        return [(s, t, r) for s, t, r, _ in work if s not in remove and t not in remove]

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
    new_nodes = [{**nodes[old], 'id': f'n{new}'} for old, new in new_pos.items()]
    new_edges = [{**e, 'source': f'n{new_pos[s]}', 'target': f'n{new_pos[t]}'}
                 for s, t, _, e in work if s in new_pos and t in new_pos]
    # a prior visit is dropped only when the deletion emptied it; one empty in the record stays
    originally_empty = _originally_empty(membership)
    live_ordinals = ({last} | originally_empty
                     | {o for old in keep for o in member[old] if not glob[old]})
    order = sorted(live_ordinals)
    remap = {o: i for i, o in enumerate(order)}
    pairs = sorted({(remap[o], new_pos[old]) for old in keep for o in member[old] if o in remap})
    new_membership = {**membership,
                      'visit_ordinals': list(range(len(order))),
                      'membership_pairs': [list(p) for p in pairs],
                      'global_node_mask': [bool(glob[old]) for old in keep]}
    new_graph = {**graph, 'nodes': new_nodes, 'edges': new_edges}
    info = {'selected': selected_count, 'cascaded': cascaded, 'combos': combos,
            'visits_dropped': len(ordinals) - len(order),
            'visits_originally_empty': len(originally_empty),
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
    cached per node set. The empty set returns the untouched graph's probabilities. `combo_rules`
    (`load_combo_rules` on the artifact's threshold table) go to `delete_from_record`; None uses its
    fallback combo rule. `combo_source` says which was used.
    """

    def __init__(self, forward: Callable, graph: dict, membership: dict, encode: Callable, *, y=None,
                 combo_rules: ComboRules | None = None):
        self.forward = forward
        self.graph, self.membership = graph, membership
        self.encode = encode
        self.combo_rules = combo_rules
        self.combo_source = combo_source(combo_rules)
        self.y = None if y is None else int(y)
        self.data = self._encode(graph, membership)
        kinds = [NODE_KINDS.index(node['kind']) for node in graph['nodes']]
        if (self.data.num_nodes != len(kinds)
                or self.data.node_type.tolist() != kinds):
            raise ValueError('encoded node indices differ from the record node positions')
        self.node_type = self.data.node_type
        self.last_info = _empty_info(graph, membership, combo_rules)
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
            graph, membership, info = delete_from_record(self.graph, self.membership, sorted(key),
                                                         combo_rules=self.combo_rules)
            self._cache[key] = (self._score(self._encode(graph, membership)), info)
        probs, self.last_info = self._cache[key]
        return probs

    def deletion_infos(self) -> list[dict]:
        """The deletion info of every distinct node set scored so far (the untouched record first)."""
        return [self._cache[frozenset()][1]] + [info for key, (_, info) in self._cache.items() if key]
