"""Closed input contracts for the v2 clinical decision-point graph.

Every rule here is a refusal, not a repair. Identifiers stay metadata, timestamps
stay in source form, and no clinical mapping is invented: the complaint vocabulary
is a frequency allowlist fitted on the TRAIN fold only, never a curated symptom
ontology and never derived from the label list.
"""
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

from data.s2_events.schema import (  # reuse audited primitives
    LAB_IDS, identifier, numeric, rows, sha256, timestamp,
)

KNOWLEDGE_RELATIONS = {'is_a', 'member_of', 'measures', 'assesses', 'may_affect', 'interacts_with'}
# Source namespaces: lab/vital (analyte, triage vital), cc: (complaint), dx: (prior
# diagnosis). Targets must stay under `knowledge:` (existing check, which already
# admits every family); the families below are the ones the seed is expected to use.
KNOWLEDGE_SOURCE_PREFIXES = ('lab:', 'vital:', 'cc:', 'dx:')
KNOWLEDGE_TARGET_FAMILIES = ('system', 'panel', 'physiology', 'icd10_chapter',
                             'icd10_block', 'icd9_chapter')

# Triage fields recorded at arrival. Temperature is Fahrenheit in the raw ED table
# and is converted exactly once, here, with the source unit recorded alongside.
TRIAGE_VITALS = {'temperature': 'Celsius', 'heartrate': 'beats/min',
                 'resprate': 'breaths/min', 'o2sat': 'percent',
                 'sbp': 'mmHg', 'dbp': 'mmHg'}
# Plausibility gates transcribed from the audited legacy spec. Out-of-range values
# are dropped with a counter, never clipped into range.
VITAL_RANGES = {'temperature': (25., 45.), 'heartrate': (5., 300.), 'resprate': (2., 120.),
                'o2sat': (50., 100.), 'sbp': (40., 300.), 'dbp': (0., 200.)}

# Arrival-time encounter fields. `disposition` is deliberately absent: it is the
# outcome of the visit and is not knowable at the decision point.
ARRIVAL_FIELDS = ('gender', 'race', 'arrival_transport')
FORBIDDEN_EDSTAYS_FIELDS = {'disposition', 'hadm_id'}

_WHITESPACE = re.compile(r'\s+')


def normalize_complaint(value):
    """Casefold and collapse whitespace. No synonym mapping, no clinical grouping.

    Returning the surface form keeps the vocabulary auditable: every token in the
    artifact can be grepped back to raw triage rows. Any mapping of `abd pain` onto
    `abdominal pain` would be an unreviewed clinical judgement, so it is not made.
    """
    if not isinstance(value, str):
        return None
    token = _WHITESPACE.sub(' ', value.strip().casefold())
    return token or None


def split_complaints(value):
    """Split the raw comma-separated triage free-text field into surface tokens."""
    if not isinstance(value, str):
        return []
    seen, result = set(), []
    for part in value.split(','):
        token = normalize_complaint(part)
        if token and token not in seen:
            seen.add(token)
            result.append(token)
    return result


def fit_complaint_vocabulary(counts, min_count):
    """Frequency allowlist fitted on training-fold visits only.

    Rare surface forms are dropped and counted, never bucketed into an `other`
    node: an `other` node would make graphs comparable on a quantity that has no
    consistent meaning across patients.
    """
    if min_count < 1:
        raise ValueError('Minimum complaint count must be positive')
    kept = sorted(token for token, count in counts.items() if count >= min_count)
    if not kept:
        raise ValueError('Empty complaint vocabulary; lower --complaint-min-count')
    return kept


def read_cohort(path):
    """Read the inherited v1 cohort unchanged.

    Sample identity, visit lineage, cutoff and fold all come from the existing
    artifact. This producer is forbidden from re-deriving a cutoff: doing so would
    silently move the prediction moment and break comparability with v1.
    """
    required = {'sample_id', 'subject_id', 'stay_id', 'cutoff', 'split'}
    samples, ids, subject_fold, encounters, stay_subjects = [], set(), {}, set(), {}
    for row in rows(path, required):
        if not row['sample_id'] or row['sample_id'] in ids:
            raise ValueError('Empty or duplicate sample_id')
        ids.add(row['sample_id'])
        for key in ('subject_id', 'stay_id'):
            identifier(row[key])
        if stay_subjects.setdefault(row['stay_id'], row['subject_id']) != row['subject_id']:
            raise ValueError('One stay cannot belong to multiple subjects')
        if row['split'] not in {'train', 'validation', 'test'}:
            raise ValueError('split must be train, validation or test')
        if subject_fold.setdefault(row['subject_id'], row['split']) != row['split']:
            raise ValueError('One subject cannot cross folds')
        cutoff = timestamp(row['cutoff'])
        key = (row['stay_id'], cutoff)
        if key in encounters:
            raise ValueError('Duplicate stay/cutoff sample')
        encounters.add(key)
        samples.append({k: row[k] for k in required} | {'cutoff': cutoff})
    if not samples:
        raise ValueError('Empty cohort')
    return samples


def read_knowledge(path):
    """Versioned, provenance-bearing relation allowlist. Source-checked, not clinically reviewed."""
    result, seen = [], set()
    required = {'source_token', 'relation', 'target_token', 'source_uri',
                'source_version', 'reviewed_by'}
    for row in rows(path, required):
        if any(not row[key].strip() for key in required):
            raise ValueError('Every knowledge relation requires complete provenance')
        if row['relation'] not in KNOWLEDGE_RELATIONS:
            raise ValueError('Unsupported medical relation')
        if not row['source_token'].startswith(KNOWLEDGE_SOURCE_PREFIXES):
            raise ValueError('Knowledge source must be an observed concept namespace')
        if not row['target_token'].startswith('knowledge:'):
            raise ValueError('Knowledge target must use knowledge: namespace')
        if not row['source_uri'].startswith(('https://', 'http://', 'urn:')):
            raise ValueError('Knowledge requires a traceable URI')
        key = tuple(row[k] for k in ('source_token', 'relation', 'target_token'))
        if key in seen:
            raise ValueError('Duplicate knowledge triple')
        seen.add(key)
        result.append({key: row[key] for key in required})
    if not result:
        raise ValueError('Knowledge input is empty')
    return result


THRESHOLD_COLUMNS = ('rule_id', 'kind', 'source_token', 'unit', 'comparator', 'threshold', 'sex',
                     'target_token', 'relation', 'source_uri', 'source_version', 'reviewed_by',
                     'evidence')
THRESHOLD_KINDS = ('level', 'state', 'combo')
THRESHOLD_COMPARATORS = ('<', '<=', '>', '>=')
THRESHOLD_SEXES = ('any', 'F', 'M')
THRESHOLD_RELATIONS = {'level': 'level', 'state': 'indicates', 'combo': 'indicates'}
LEVEL_TARGETS = ('level:low', 'level:high')
_PLAIN_NUMBER = re.compile(r'^[+-]?(\d+(\.\d*)?|\.\d+)$')
_STATE_TARGET = re.compile(r'^knowledge:state:[a-z][a-z0-9_]*$')


def read_thresholds(path):
    """Clinical level / state / combo rules with per-row provenance (source-checked only).

    Level rows: per-analyte reference limit (target level:low or level:high).
    State rows: one measurement implies knowledge:state:<name>.
    Combo rows: several rows sharing a rule_id; the state needs every row to hold.
    """
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != THRESHOLD_COLUMNS:
            raise ValueError('Thresholds header must be exactly: ' + ','.join(THRESHOLD_COLUMNS))
        raw = list(reader)
    result, level_keys, rule_kinds = [], set(), {}
    for row in raw:
        if None in row or any(row[key] is None for key in THRESHOLD_COLUMNS):
            raise ValueError('Thresholds row has the wrong number of fields')
        row = {key: row[key].strip() for key in THRESHOLD_COLUMNS}
        if any(not row[key] for key in THRESHOLD_COLUMNS):
            raise ValueError('Every threshold row requires every field, incl. complete provenance')
        kind = row['kind']
        if kind not in THRESHOLD_KINDS:
            raise ValueError('Unsupported threshold kind')
        if row['comparator'] not in THRESHOLD_COMPARATORS:
            raise ValueError('Unsupported threshold comparator')
        if not _PLAIN_NUMBER.match(row['threshold']):
            raise ValueError('Threshold must be a plain number')
        value = float(row['threshold'])
        if not math.isfinite(value):
            raise ValueError('Threshold must be finite')
        if row['sex'] not in THRESHOLD_SEXES:
            raise ValueError('Threshold sex must be any, F or M')
        if row['relation'] != THRESHOLD_RELATIONS[kind]:
            raise ValueError(f'Threshold kind {kind} requires relation {THRESHOLD_RELATIONS[kind]}')
        if not row['source_token'].startswith(('lab:', 'vital:')):
            raise ValueError('Threshold source must be a lab: or vital: token')
        if not row['source_uri'].startswith(('https://', 'http://', 'urn:')):
            raise ValueError('Threshold requires a traceable URI')
        if rule_kinds.setdefault(row['rule_id'], kind) != kind:
            raise ValueError('rule_id is shared across kinds')
        if kind == 'level':
            if row['target_token'] not in LEVEL_TARGETS:
                raise ValueError('Level target must be level:low or level:high')
            if (row['target_token'] == 'level:low') != row['comparator'].startswith('<'):
                raise ValueError('level:low needs < or <=; level:high needs > or >=')
            key = (row['source_token'], row['unit'], row['sex'], row['target_token'])
            if key in level_keys:
                raise ValueError('Duplicate level limit for token/unit/sex')
            level_keys.add(key)
        elif not _STATE_TARGET.match(row['target_token']):
            raise ValueError('State target must be knowledge:state:<snake_name>')
        result.append(row | {'threshold': value})
    if not result:
        raise ValueError('Thresholds input is empty')
    by_rule = {}
    for row in result:
        by_rule.setdefault(row['rule_id'], []).append(row)
    for rule_id, group in by_rule.items():
        kind = group[0]['kind']
        if kind == 'combo':
            if len(group) < 2:
                raise ValueError('Combo rule needs at least 2 rows: ' + rule_id)
            if len({row['target_token'] for row in group}) != 1:
                raise ValueError('Combo rule needs exactly one target: ' + rule_id)
        elif len(group) != 1:
            raise ValueError('Duplicate rule_id for a non-combo rule: ' + rule_id)
    return result


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    Path(path).chmod(0o600)


__all__ = ['ARRIVAL_FIELDS', 'FORBIDDEN_EDSTAYS_FIELDS', 'KNOWLEDGE_RELATIONS',
           'LAB_IDS', 'TRIAGE_VITALS', 'VITAL_RANGES', 'fit_complaint_vocabulary',
           'identifier', 'normalize_complaint', 'numeric', 'read_cohort',
           'read_knowledge', 'read_thresholds', 'rows', 'save_json', 'sha256', 'split_complaints',
           'timestamp']
