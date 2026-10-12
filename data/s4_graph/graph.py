"""Materialize one decision-point graph per cohort visit.

Design rule, enforced by `audit.py` against the produced artifact: an edge earns its
place only if computing its `informative` payload requires looking at BOTH endpoints.
v1 failed this -- `contains_event` and `observes_concept` are membership facts that
a presence vector already encodes, and `next_event_time` orders nodes that each
carry their own timestamp. Those relations are kept here only where they are needed
to make the graph navigable, and are declared STRUCTURAL so no result may cite them.

The informative relations:

  baseline_of    prior-visit measurement -> index measurement of the SAME analyte,
                 carrying the signed delta, the elapsed interval and the rate. The
                 delta is a two-endpoint quantity; a fixed-width row holding "latest
                 creatinine" cannot express "creatinine rose 1.8 mg/dL in 11 days"
                 because the comparison partner differs per patient.
  trajectory_of  consecutive prior measurements of one analyte, same payload. Gives
                 message passing a real chain to walk instead of a star.
  co_complaint   complaint <-> complaint within one visit. Conjunction is the point:
                 "chest pain" with "dyspnea" is not the sum of the two presences.
  recurrence_of  prior diagnosis -> the visit, carrying how many prior encounters
                 recorded it and how long ago it was last seen. "COPD, 3 admissions,
                 last one 40 days ago" is not the same evidence as a history flag.
  comorbid_with  prior diagnosis <-> prior diagnosis recorded in the SAME encounter.
                 Co-occurring conditions constrain each other; their pair is not the
                 sum of two indicator columns.
  medical:*      concept -> external knowledge node, shared across patients.

Diagnoses come exclusively from encounters that FINISHED before the index encounter
began. The index encounter's own diagnoses are the label and are never admitted --
`diagnosis.DiagnosisIndex.prior_history` refuses the index stay, and `build.py`
re-verifies from the source after writing.

Everything is bounded by the inherited cutoff; no node may be dated after it.
"""
from collections import defaultdict
from itertools import combinations

from core import SCHEMA_VERSION
from core.contracts import VISIT_MEMBERSHIP_CONTRACT_VERSION
from core.schema import timestamp
from data.s3_labels.icd_mapping import ICD_MAPPING_POLICY

LOGIC_CONTRACT_VERSION = 'clinical_graph_logic_v2'


def temporal_contract(with_diagnosis):
    """Recorded proxy bounds are testable; actual clinician availability is not."""
    assumptions = ['triage_at_encounter_intime', 'storetime_as_availability_proxy']
    if with_diagnosis:
        assumptions.append('prior_diagnosis_available_at_encounter_completion')
    return {'temporal_clean': False, 'early_triage_eligible': False,
            'temporal_validation': 'proxy_bounds_only; availability_assumptions_unverified',
            'availability_assumptions': assumptions}


def assumed_timing(source, basis):
    return {'timestamp_source': source, 'timing_basis': basis,
            'source_timestamp_present': False,
            'availability_assumed': True, 'availability_verified': False}


def lab_availability():
    return {'availability_basis': 'database_storetime_proxy',
            'availability_verified': False, 'availability_source': 'labevents.csv:storetime'}


_COMPARE = {'<': lambda a, b: a < b, '<=': lambda a, b: a <= b,
            '>': lambda a, b: a > b, '>=': lambda a, b: a >= b}


def _sex_ok(row_sex, gender):
    return row_sex == 'any' or row_sex == gender


LEVEL_SOURCES = ('csv', 'row-range')
CENSOR_SIGNS = ('<', '>')
# Serum and blood-gas potassium / sodium: under 'row-range' their level keeps the textbook CSV
# limits (the record's own range, e.g. K 3.3-5.1 or Na 133-145, would move the clinical cut-off).
TEXTBOOK_LEVEL_TOKENS = {'lab:50971', 'lab:50983', 'lab:50822', 'lab:50824'}


def _holds(comparator, threshold, value, censored=None):
    """Does `value <comparator> threshold` hold; a censored value only when certain.

    A censored result stores its detection limit as `value`: '<x' means the true value
    lies in (-inf, x), '>x' in (x, +inf). A rule holds only if it holds for the whole
    interval, so '<x' never satisfies a '>'/'>=' rule and '>x' never satisfies a '<'/'<='
    rule. censored=None is the exact old comparison.
    """
    if censored is None:
        return _COMPARE[comparator](value, threshold)
    if censored == '<':
        return comparator in ('<', '<=') and value <= threshold
    if censored == '>':
        return comparator in ('>', '>=') and value >= threshold
    raise ValueError('Unknown censor sign: %r' % (censored,))


def _own_range(ref_lower, ref_upper):
    """The record's own reference range, or None unless both bounds exist and lower < upper.

    Equal bounds (5-5, and the 0-0 placeholder) carry no range information, so they fall
    back to the CSV limits like a missing range.
    """
    if ref_lower is None or ref_upper is None or not ref_lower < ref_upper:
        return None
    return ref_lower, ref_upper


def level_rows(thresholds, token, unit, gender):
    """The level row per side ('level:low' / 'level:high') that applies to this patient.

    A sex-specific row for the patient's gender replaces the `any` row of the same side.
    """
    chosen = {}
    for row in thresholds or ():
        if (row['kind'] != 'level' or row['source_token'] != token or row['unit'] != unit
                or not _sex_ok(row['sex'], gender)):
            continue
        side = row['target_token']
        current = chosen.get(side)
        if current is None or (current['sex'] == 'any' and row['sex'] != 'any'):
            chosen[side] = row
    return chosen


def level_for(thresholds, token, unit, value, gender, censored=None, ref_lower=None,
              ref_upper=None, level_source='csv'):
    """low | normal | high from level rows of this token+unit; None if no limit applies.

    A sex-specific row for the patient's gender replaces the `any` row of the same side.
    censored ('<' / '>' / None) is the sign of a result stored as its detection limit.
    level_source 'row-range': when the record carries both of its own reference bounds,
    they replace the CSV limits (value < ref_lower is low, value > ref_upper is high) for
    every side the CSV defines for this token+unit; otherwise the CSV row decides. Tokens in
    TEXTBOOK_LEVEL_TOKENS (serum / blood-gas K and Na) always use the CSV rows.
    """
    if level_source not in LEVEL_SOURCES:
        raise ValueError('Unknown level source: %r' % (level_source,))
    if not thresholds or value is None:
        return None
    chosen = level_rows(thresholds, token, unit, gender)
    if not chosen:
        return None
    own = (_own_range(ref_lower, ref_upper)
           if level_source == 'row-range' and token not in TEXTBOOK_LEVEL_TOKENS else None)
    for side in ('level:low', 'level:high'):
        row = chosen.get(side)
        if row is None:
            continue
        if own is not None:
            comparator, threshold = ('<', own[0]) if side == 'level:low' else ('>', own[1])
        else:
            comparator, threshold = row['comparator'], row['threshold']
        if _holds(comparator, threshold, value, censored):
            return side[6:]
    return 'normal'


# Tokens whose '>'/'>=' state/combo rows use the record's own upper limit under 'row-range'
# (anion gap: MIMIC rows with reference 8-20 include potassium, ~+4 mEq/L over the K-free CSV).
OWN_UPPER_TOKENS = {'lab:50868'}
_LOW_COMPARATORS, _HIGH_COMPARATORS = ('<', '<='), ('>', '>=')


def threshold_semantics():
    """JSON-able description of the row-range threshold rules (recorded in manifests)."""
    return {'state_gate': 'level_agreement',
            'textbook_level_tokens': sorted(TEXTBOOK_LEVEL_TOKENS),
            'own_upper_tokens': sorted(OWN_UPPER_TOKENS),
            'own_range_rule': 'both bounds present and lower < upper'}


def _row_holds(row, token, unit, value, gender, censored=None, level=None, ref_upper=None,
               level_source='csv', kind='state', ref_lower=None):
    """Does this state/combo row fire on the measurement.

    level_source 'csv' (default) is the old comparison. 'row-range': when a level was
    computed for the node, a STATE row ('<'/'<=') needs level 'low' and a '>'/'>=' state row
    needs 'high' (so a state never contradicts the node level). COMBO rows (`kind` 'combo')
    keep their CSV comparison: their criteria (SIRS HR > 90, qSOFA SBP <= 100, HHS
    bicarbonate > 18) deliberately sit inside the normal range. '>'/'>=' rows (state and
    combo) on OWN_UPPER_TOKENS use the record's own `ref_upper` as threshold when the record
    has a valid own range (`_own_range`: both bounds, ordered), else the CSV threshold.
    Censoring still follows `_holds`.
    """
    if not (row['source_token'] == token and row['unit'] == unit and value is not None
            and _sex_ok(row['sex'], gender)):
        return False
    comparator, threshold = row['comparator'], row['threshold']
    if level_source == 'row-range':
        wanted = 'low' if comparator in _LOW_COMPARATORS else 'high'
        if kind == 'state' and level is not None and level != wanted:
            return False
        if (comparator in _HIGH_COMPARATORS and token in OWN_UPPER_TOKENS
                and _own_range(ref_lower, ref_upper) is not None):
            threshold = ref_upper
    return _holds(comparator, threshold, value, censored)


def build_graph_with_visit_membership(store, sample, knowledge, vocabulary, diagnoses=None,
                                      max_nodes=20000, max_edges=200000, thresholds=None,
                                      level_source='csv', icd_mapping_policy=None):
    """Build one graph and its source-derived visit-membership sidecar row.

    `store` exposes the read-only v1 SQLite event index plus the v2 triage/arrival
    tables. `vocabulary` is the train-fitted complaint allowlist. `diagnoses` is an
    optional `DiagnosisIndex`; when absent the graph carries no diagnosis tier. No
    target is read or attached here: labels are bound in a separate, hashed sidecar.

    v5 options (defaults reproduce v3/v4 graphs byte-for-byte): `level_source` ('csv' |
    'row-range', see `level_for`); `icd_mapping_policy` overrides the per-graph policy
    string. Lab nodes carry censored/ref_lower/ref_upper/flag only when the event store
    has those columns (`store.has_v5_columns`); diagnosis nodes carry full_code only when
    the DiagnosisIndex was built under the v5 policy.
    """
    if level_source not in LEVEL_SOURCES:
        raise ValueError('Unknown level source: %r' % (level_source,))
    v5_events = bool(getattr(store, 'has_v5_columns', False))
    cutoff = sample['cutoff']
    index = store.visit(sample['stay_id'])
    if index is None or index['subject'] != sample['subject_id']:
        raise ValueError('Index visit lineage mismatch')
    if not index['start'] <= cutoff.isoformat() <= (index['finish'] or index['start']):
        raise ValueError('Cutoff falls outside the index encounter')

    prior_visits = store.prior_visits(sample['subject_id'], index['start'])
    ordered_stays = [*prior_visits, sample['stay_id']]
    stay_to_ordinal = {stay: ordinal for ordinal, stay in enumerate(ordered_stays)}
    if len(stay_to_ordinal) != len(ordered_stays):
        raise ValueError('Duplicate source stay in visit order')

    nodes, edges, lookup, times = [], [], {}, {}
    node_visits = defaultdict(set)
    global_nodes = set()
    dropped = defaultdict(int)

    def hours(value):
        return (timestamp(value) - cutoff).total_seconds() / 3600. if value else None

    def node(key, kind, token, **payload):
        if key not in lookup:
            if len(nodes) >= max_nodes:
                raise ValueError('Graph node budget exceeded; no silent truncation')
            node_id = 'n' + str(len(nodes))
            lookup[key] = node_id
            time_hours = hours(payload.pop('time', None))
            available_hours = hours(payload.pop('available', None))
            if time_hours is not None and time_hours > 0:
                raise ValueError('Node dated after the prediction cutoff')
            if available_hours is not None and available_hours > 0:
                raise ValueError('Node available only after the prediction cutoff')
            times[node_id] = time_hours
            nodes.append({'id': node_id, 'kind': kind, 'token': token,
                          'time_hours': time_hours, 'available_hours': available_hours,
                          **payload})
        return lookup[key]

    def edge(source, target, relation, informative, **payload):
        if len(edges) >= max_edges:
            raise ValueError('Graph edge budget exceeded; no silent truncation')
        edges.append({'source': source, 'target': target, 'relation': relation,
                      'informative': informative, **payload})

    def add_membership(node_id, stay):
        try:
            node_visits[node_id].add(stay_to_ordinal[stay])
        except KeyError as exc:
            raise ValueError('Node source stay is outside the sample visit lineage') from exc

    # ---------------------------------------------------------------- patient/visit
    arrival = store.arrival(sample['stay_id'])
    gender = arrival.get('gender')
    threshold_state_rows = [r for r in (thresholds or ()) if r['kind'] != 'level']
    # (node_id, token, unit, value, time, censored, level, ref_upper, ref_lower) of index
    # measurements + triage vitals; level / ref bounds drive the row-range state/combo decision.
    index_nodes = []

    def node_level(token, unit, value, record=None):
        record = record or {}
        return level_for(thresholds, token, unit, value, gender,
                         censored=record.get('censored'),
                         ref_lower=record.get('ref_lower'),
                         ref_upper=record.get('ref_upper'),
                         level_source=level_source)

    def level_payload(token, unit, value, record=None):
        # No `level` key at all without thresholds: graphs stay byte-identical.
        if thresholds is None:
            return {}
        return {'level': node_level(token, unit, value, record)}

    def lab_extras(record):
        # Only v5 event indexes: older artifacts keep their node payload unchanged.
        if not v5_events:
            return {}
        return {'censored': record.get('censored'), 'ref_lower': record.get('ref_lower'),
                'ref_upper': record.get('ref_upper'), 'flag': record.get('flag')}
    patient = node(('patient',), 'patient', 'patient',
                   age=arrival.get('age'), gender=arrival.get('gender'),
                   race=arrival.get('race'), arrival_transport=arrival.get('arrival_transport'))
    global_nodes.add(patient)
    visit = node(('visit', sample['stay_id']), 'visit', 'visit:index',
                 time=index['start'], available=index['start'],
                 acuity=arrival.get('acuity'),
                 acuity_timing=assumed_timing('edstays.csv:intime', 'encounter_intime_assumption'))
    add_membership(visit, sample['stay_id'])
    edge(patient, visit, 'has_visit', False)
    edge(visit, patient, 'index_visit_of', False)

    # ------------------------------------------------------------------- complaints
    # Triage has NO independent timestamp. Assigning intime is an assumption,
    # not evidence that complaints/vitals/acuity were actually visible by cutoff.
    complaint_ids = []
    for token in store.complaints(sample['stay_id']):
        if token not in vocabulary:
            dropped['complaint_out_of_vocabulary'] += 1
            continue
        cid = node(('complaint', token), 'complaint', 'cc:' + token,
                   time=index['start'], available=index['start'], source_table='triage.csv',
                   **assumed_timing('edstays.csv:intime', 'encounter_intime_assumption'))
        add_membership(cid, sample['stay_id'])
        edge(visit, cid, 'reports_complaint', False)
        complaint_ids.append(cid)
    # Conjunction, not summation: this is the pairwise term a presence vector lacks.
    for left, right in combinations(sorted(complaint_ids, key=lambda i: int(i[1:])), 2):
        edge(left, right, 'co_complaint', True)
        edge(right, left, 'co_complaint', True)

    # ---------------------------------------------------------------- triage vitals
    for field, value, unit, source_unit in store.triage_vitals(sample['stay_id']):
        vid = node(('vital', field), 'vital', 'vital:' + field, value=value, unit=unit,
                   source_unit=source_unit, time=index['start'], available=index['start'],
                   source_table='triage.csv', **level_payload('vital:' + field, unit, value),
                   **assumed_timing('edstays.csv:intime', 'encounter_intime_assumption'))
        add_membership(vid, sample['stay_id'])
        edge(visit, vid, 'observed_vital', False)
        index_nodes.append((vid, 'vital:' + field, unit, value, index['start'], None,
                            node_level('vital:' + field, unit, value) if thresholds is not None else None,
                            None, None))

    # ------------------------------------------------- index-visit lab measurements
    # One node per result, keyed by the source labevent id so repeated measurements
    # of one analyte stay distinct rather than being collapsed into a summary.
    analyte_nodes = {}
    index_by_analyte = defaultdict(list)
    for record in store.measurements(sample['stay_id'], cutoff):
        mid = node(('measure', record['id']), 'measurement', record['token'],
                   value=record['value'], unit=record['unit'],
                   time=record['time'], available=record['available'],
                   source_record=record['source'], timing_basis=record['timing_basis'],
                   scope='index', **lab_availability(), **lab_extras(record),
                   **level_payload(record['token'], record['unit'], record['value'], record))
        add_membership(mid, record['stay'])
        index_nodes.append((mid, record['token'], record['unit'], record['value'], record['time'],
                            record.get('censored'),
                            (node_level(record['token'], record['unit'], record['value'], record)
                             if thresholds is not None else None),
                            record.get('ref_upper'), record.get('ref_lower')))
        analyte =  analyte_nodes.get(record['token'])
        if analyte is None:
            analyte = node(('analyte', record['token']), 'analyte', record['token'])
            analyte_nodes[record['token']] = analyte
        add_membership(analyte, record['stay'])
        edge(visit, mid, 'measured_in', False)
        edge(mid, analyte, 'instance_of', False)
        index_by_analyte[record['token']].append((record['time'], mid, record))

    # ------------------------------------------- prior-visit baselines (informative)
    # Only analytes that ALSO occur in the index visit are pulled in. This is the
    # whole point: an unmatched historical value is a feature column, while a matched
    # one defines a delta that needs both endpoints. v1 dumped every historical event
    # regardless of match, producing 18 GB and, as measured, no signal.
    baseline_links = 0
    for analyte_token, index_records in index_by_analyte.items():
        history = store.analyte_history(sample['subject_id'], prior_visits, analyte_token, cutoff)
        if not history:
            continue
        previous = None
        history_ids = []
        for record in history:
            hid = node(('measure', record['id']), 'measurement', record['token'],
                       value=record['value'], unit=record['unit'],
                       time=record['time'], available=record['available'],
                       source_record=record['source'], timing_basis=record['timing_basis'],
                       scope='prior', **lab_availability(), **lab_extras(record),
                       **level_payload(record['token'], record['unit'], record['value'], record))
            add_membership(hid, record['stay'])
            add_membership(analyte_nodes[analyte_token], record['stay'])
            edge(hid, analyte_nodes[analyte_token], 'instance_of', False)
            history_ids.append((record, hid))
            if previous is not None:
                _link(edge, previous[1], hid, 'trajectory_of', previous[0], record)
            previous = (record, hid)
        earliest_index = min(index_records, key=lambda entry: entry[0])
        last_record, last_id = history_ids[-1]
        _link(edge, last_id, earliest_index[1], 'baseline_of', last_record, earliest_index[2])
        baseline_links += 1

    # ------------------------------------------ prior diagnosis tier (informative)
    # Strictly from encounters that finished before this one began. The index
    # encounter's diagnoses are the label; `prior_history` refuses the index stay.
    diagnosis_nodes = {}
    recurrence_links = 0
    comorbid_links = 0
    if diagnoses is not None:
        history = diagnoses.prior_history(sample['stay_id'], prior_visits)
        visit_end = {stay: store.visit(stay)['finish'] for stay in prior_visits}
        by_encounter = defaultdict(list)
        for record in history:
            ends = [visit_end[stay] for stay in record['occurrences'] if visit_end.get(stay)]
            last_seen = max(ends) if ends else None
            # v5 (diagnosis.normalize_v5): the token is a 3-char category spanning many
            # codes; `full_code` is the deterministic smallest one, all of them are listed.
            v5_dx = ({'full_code': record['full_code'],
                      'full_codes': record.get('full_codes', [record['full_code']]),
                      'source_versions': record.get('source_versions', [record['source_version']])}
                     if 'full_code' in record else {})
            did = node(('diagnosis', record['token']), 'diagnosis', record['token'],
                       source_version=record['source_version'],
                       source_code=record['source_code'],
                       approximate_mapping=record['approximate_mapping'],
                       **v5_dx,
                       prior_encounters=len(record['occurrences']), source_table='diagnosis.csv',
                       assumed_available_hours=hours(last_seen),
                       **assumed_timing('edstays.csv:outtime', 'completed_prior_encounter_end_assumption'))
            for stay in record['occurrences']:
                add_membership(did, stay)
            diagnosis_nodes[record['token']] = did
            edge(visit, did, 'has_prior_diagnosis', False)
            # Recency is the two-endpoint quantity: a condition coded three times,
            # most recently 40 days ago, is different evidence from a history flag.
            age_hours = hours(last_seen) if last_seen else None
            edge(did, visit, 'recurrence_of', True,
                 prior_encounters=len(record['occurrences']),
                 last_seen_hours=age_hours, recency_basis='completed_prior_encounter_end_assumption')
            recurrence_links += 1
            for stay in record['occurrences']:
                by_encounter[stay].append(record['token'])
        # Conditions coded together in one past encounter constrain each other.
        for stay, tokens in by_encounter.items():
            for left, right in combinations(sorted(set(tokens)), 2):
                edge(diagnosis_nodes[left], diagnosis_nodes[right], 'comorbid_with', True)
                edge(diagnosis_nodes[right], diagnosis_nodes[left], 'comorbid_with', True)
                comorbid_links += 1

    # ------------------------------------------------------- external knowledge tier
    knowledge_edges = 0
    for triple in knowledge:
        source = triple['source_token']
        if source.startswith(('cc:', 'dx:')):
            # Complaint / prior-diagnosis nodes carry the source token verbatim. Only
            # prior-visit diagnosis nodes exist (DiagnosisIndex.prior_history), so the
            # index visit's own diagnosis can never be an anchor.
            anchor = lookup.get(('complaint', source[3:])) if source.startswith('cc:') \
                else diagnosis_nodes.get(source)
        else:
            anchor = analyte_nodes.get(source) or lookup.get(('vital', source.split(':', 1)[-1]))
        if anchor is None:
            continue
        target = node(('knowledge', triple['target_token']), 'knowledge', triple['target_token'])
        anchor_visits = node_visits.get(anchor, set())
        if not anchor_visits:
            raise ValueError('Knowledge anchor has no source visit membership')
        node_visits[target].update(anchor_visits)
        edge(anchor, target, 'medical:' + triple['relation'], True,
             provenance={k: triple[k] for k in ('source_uri', 'source_version', 'reviewed_by')})
        knowledge_edges += 1

    # ------------------------------------------------ threshold state tier (informative)
    # Index-visit measurements and triage vitals only; prior-visit history carries the
    # `level` attribute but never state/combo edges.
    threshold_edges = 0
    fired_states = set()

    def indicate(anchor_id, row):
        nonlocal threshold_edges
        target = node(('knowledge', row['target_token']), 'knowledge', row['target_token'])
        anchor_visits = node_visits.get(anchor_id, set())
        if not anchor_visits:
            raise ValueError('Threshold anchor has no source visit membership')
        node_visits[target].update(anchor_visits)
        edge(anchor_id, target, 'medical:indicates', True, rule_id=row['rule_id'],
             provenance={k: row[k] for k in ('source_uri', 'source_version', 'reviewed_by')})
        threshold_edges += 1
        fired_states.add(row['target_token'])

    if thresholds is not None:
        for row in threshold_state_rows:
            if row['kind'] != 'state':
                continue
            for node_id, token, unit, value, _, censored, level, ref_upper, ref_lower in index_nodes:
                if _row_holds(row, token, unit, value, gender, censored, level, ref_upper,
                              level_source, 'state', ref_lower):
                    indicate(node_id, row)
        combos = defaultdict(list)
        for row in threshold_state_rows:
            if row['kind'] == 'combo':
                combos[row['rule_id']].append(row)
        for rule_id in sorted(combos):
            contributing = {}
            for row in combos[rule_id]:
                candidates = [c for c in index_nodes
                              if c[1] == row['source_token'] and c[2] == row['unit']]
                if not candidates:
                    break
                # Latest available index value of the token (ties: later node).
                latest = max(candidates, key=lambda c: (timestamp(c[4]), int(c[0][1:])))
                if not _row_holds(row, latest[1], latest[2], latest[3], gender, latest[5],
                                  latest[6], latest[7], level_source, 'combo', latest[8]):
                    break
                contributing[latest[0]] = row
            else:
                for node_id in sorted(contributing, key=lambda i: int(i[1:])):
                    indicate(node_id, contributing[node_id])

    informative = sum(1 for e in edges if e['informative'])
    graph = {
        'schema_version': SCHEMA_VERSION,
        'logic_contract_version': LOGIC_CONTRACT_VERSION,
        'icd_mapping_policy': icd_mapping_policy or ICD_MAPPING_POLICY,
        **temporal_contract(diagnoses is not None),
        'sample_id': sample['sample_id'],
        'split': sample['split'],
        'nodes': nodes,
        'edges': edges,
        'coverage': {
            'complaints': len(complaint_ids),
            'index_measurements': sum(len(v) for v in index_by_analyte.values()),
            'prior_visits': len(prior_visits),
            'baseline_links': baseline_links,
            'prior_diagnoses': len(diagnosis_nodes),
            'recurrence_links': recurrence_links,
            'comorbid_links': comorbid_links,
            'knowledge_edges': knowledge_edges,
            'informative_edges': informative,
            'structural_edges': len(edges) - informative,
            'dropped': dict(dropped),
        },
        'limitations': [
            'Recorded order is not causality.',
            'Triage complaint text is a surface-form allowlist, not a clinical ontology.',
            'Triage timestamps are unverified encounter-intime assumptions; checking intime <= cutoff does not prove triage availability.',
            'Laboratory availability uses storetime, a database proxy for clinician visibility.',
            'Diagnoses carry no source timestamp; completion-time availability is assumed, not verified. Structural prior-stay eligibility does not prove coding availability.',
            'ICD-9 codes are mapped to ICD-10 through the repository GEM table; approximate mappings are flagged, unmappable codes keep their ICD-9 namespace.',
            'Knowledge relations are source-checked, not clinically reviewed.',
        ],
    }
    if thresholds is not None:
        graph['coverage']['threshold_state_edges'] = threshold_edges
        graph['coverage']['threshold_states'] = sorted(fired_states)
        graph['limitations'].append('Threshold levels and states are source-checked limits, not clinical '
                                    'review; prior-visit measurements carry level only, never state edges.')
    if thresholds is not None and level_source == 'row-range':
        graph['limitations'].append('Level uses the result\'s own laboratory reference range where both bounds '
                                    'are recorded (sides taken from the rule CSV), else the CSV limit.')
        graph['limitations'].extend([
            'State rows fire only when the node level agrees with the rule direction '
            '(low rows need level low, high rows need level high); combo rows keep their CSV comparison.',
            'Serum and blood-gas potassium and sodium levels (%s) use the textbook CSV limits, '
            'not the result\'s own reference range.' % ', '.join(sorted(TEXTBOOK_LEVEL_TOKENS)),
            'DKA anion gap = above the lab\'s own upper limit (fallback > 12 when no valid own range).',
        ])
    membership_pairs = []
    global_node_mask = []
    for node_index, graph_node in enumerate(nodes):
        node_id = graph_node['id']
        is_global = node_id in global_nodes
        visits = node_visits.get(node_id, set())
        if is_global:
            if visits:
                raise ValueError('Global node also has visit-specific provenance')
        elif not visits:
            raise ValueError('Visit-specific graph node has no source membership')
        global_node_mask.append(is_global)
        membership_pairs.extend([ordinal, node_index] for ordinal in sorted(visits))
    sidecar_record = {
        'contract_version': VISIT_MEMBERSHIP_CONTRACT_VERSION,
        'sample_id': sample['sample_id'],
        'visit_ordinals': list(range(len(ordered_stays))),
        'membership_pairs': sorted(membership_pairs),
        'global_node_mask': global_node_mask,
    }
    return graph, sidecar_record


def build_graph(store, sample, knowledge, vocabulary, diagnoses=None,
                max_nodes=20000, max_edges=200000, thresholds=None,
                level_source='csv', icd_mapping_policy=None):
    """Compatibility wrapper preserving the original graph-only API."""
    graph, _ = build_graph_with_visit_membership(
        store, sample, knowledge, vocabulary, diagnoses, max_nodes, max_edges, thresholds,
        level_source, icd_mapping_policy)
    return graph


def _link(edge, source_id, target_id, relation, source_record, target_record):
    """Attach the two-endpoint payload that justifies the relation existing at all.

    delta / interval / rate are undefined for a single node: they are the reason a
    graph model can see something a per-node feature row cannot.
    """
    start = timestamp(source_record['time'])
    end = timestamp(target_record['time'])
    interval = (end - start).total_seconds() / 3600.
    if interval < 0:
        raise ValueError('Baseline/trajectory link must run forward in time')
    delta = None
    rate = None
    if source_record['value'] is not None and target_record['value'] is not None:
        if source_record['unit'] != target_record['unit']:
            # Refuse to compare across units rather than invent a conversion.
            delta = None
        else:
            delta = target_record['value'] - source_record['value']
            rate = delta / interval if interval > 0 else None
    # A censored endpoint is a bound, so the delta is a bound too (flagged, not dropped).
    bound = ({'delta_is_bound': True}
             if source_record.get('censored') or target_record.get('censored') else {})
    edge(source_id, target_id, relation, True,
         delta=delta, interval_hours=interval, rate_per_hour=rate,
         comparable_units=source_record['unit'] == target_record['unit'], **bound)
