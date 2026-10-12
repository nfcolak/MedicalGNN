"""First-lab-plus-delay cutoff (v4b): a NEW cutoff contract derived from a first_lab artifact.

cutoff = min(first_available + delay, ED outtime). `first_available` is the inherited
first_results value, so the 18 GB lab scan is never repeated: the inherited event index
already holds every numeric result with its storetime. The index is copied (events rows
unchanged) and its first_results table is recomputed as the new cutoff. sample_id,
stay_id, subject_id and split are inherited byte-for-byte; only the cutoff column moves.

Plan mode (no --execute) prints the contract and creates nothing.
"""
import argparse
import csv
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import shutil
import sqlite3

from core.schema import save_json, sha256, timestamp
from data.s2_events.event_store_v5 import EVENT_SCHEMA_V5, event_schema, first_rule_violations

POLICY = 'first_lab_plus_delay'
FOLDS = ('train', 'validation', 'test')
REPORTED_FOLDS = ('train', 'validation')  # the test fold is never counted in a report
DECISION_POLICY = ('Cutoff = min(first nonempty recorded lab result storetime + {delay} min, ED outtime). '
                   'The first result is the inherited first_lab cutoff (charttime<=storetime<=outtime, '
                   'intime<=charttime). Include all representable results with storetime<=cutoff, ties '
                   'inclusive. The delay is fixed for every stay and does not depend on which labs arrived. '
                   'Not an early-triage time; temporal_clean=false (storetime is an availability proxy).')


DECISION_POLICY_V5 = ('Cutoff = min(first representable lab result storetime + {delay} min, ED outtime). '
                      'The first result is the inherited representable-v5 first_lab cutoff (earliest storetime among '
                      'numeric results with a unit, censored results with their limit, and allowlisted unit-less '
                      'results; text-only and unit-less non-allowlisted results never set it; charttime<=storetime<='
                      'outtime, intime<=charttime). Include all representable results with storetime<=cutoff, ties '
                      'inclusive. The delay is fixed for every stay and does not depend on which labs arrived. '
                      'Not an early-triage time; temporal_clean=false (storetime is an availability proxy).')


def _ro(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro&immutable=1', uri=True)


def new_cutoff(first, finish, delay):
    """Pure rule shared by production and verification."""
    return min(timestamp(first) + timedelta(minutes=delay), timestamp(finish)).isoformat()


def load_inherited(inherit):
    from data.s2_events.repair_metadata import validate_index_ready
    inherit = Path(inherit).resolve()
    manifest = json.loads((inherit / 'manifest.json').read_text())
    if manifest.get('cutoff_policy') == POLICY:
        raise ValueError('Inherited artifact is already a first_lab_plus_delay artifact')
    validate_index_ready(inherit, manifest)
    db = _ro(inherit / 'events.sqlite')
    try:
        schema = event_schema(db)
    finally:
        db.close()
    if schema != manifest.get('event_schema'):
        raise ValueError('Inherited manifest event_schema %r does not match the index columns (%r)'
                         % (manifest.get('event_schema'), schema))
    for name in ('events.sqlite', 'cohort.csv', 'knowledge.csv', 'canonical_split.json', 'prepared_index.json'):
        if not (inherit / name).is_file():
            raise ValueError('Inherited artifact lacks ' + name)
    return inherit, manifest


def cutoff_table(inherit, delay):
    """stay -> (inherited first, new cutoff, start, finish), read-only."""
    db = _ro(inherit / 'events.sqlite')
    try:
        return {stay: (first, new_cutoff(first, finish, delay), start, finish)
                for stay, first, start, finish in db.execute(
                    'SELECT v.stay,f.available,v.start,v.finish FROM visits v JOIN first_results f USING(stay)')}
    finally:
        db.close()


def clip_share(inherit, table):
    """Share of stays whose cutoff is clipped at outtime; train and validation only."""
    totals = {name: [0, 0] for name in REPORTED_FOLDS}
    with (inherit / 'cohort.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['split'] in totals:
                first, cutoff, _, finish = table[row['stay_id']]
                totals[row['split']][0] += 1
                totals[row['split']][1] += int(cutoff == timestamp(finish).isoformat())
    return {name: {'stays': n, 'clipped_at_outtime': c, 'share': round(c / n, 6) if n else None}
            for name, (n, c) in totals.items()}


def plan(args):
    inherit, manifest = load_inherited(args.inherit_from)
    table = cutoff_table(inherit, args.delay_minutes)
    return {'status': 'not_executed', 'requires': '--execute', 'cutoff_policy': POLICY,
            'event_schema': manifest.get('event_schema'),
            'first_result_rule': ('representable results only (checked against the inherited events at --execute)'
                                  if manifest.get('event_schema') == EVENT_SCHEMA_V5 else
                                  'inherited first_results as-is (any nonempty result; v3/v4 contract)'),
            'rule': 'cutoff = min(first_available + delay_minutes, ED outtime)',
            'delay_minutes': args.delay_minutes, 'inherit_from': str(inherit),
            'inherits_index_sha256': manifest['event_index_sha256'],
            'inherits_cohort_sha256': manifest['cohort_sha256'],
            'output': str(args.output.resolve()), 'output_exists': args.output.exists(),
            'will_write': ['events.sqlite (copy; first_results recomputed)', 'cohort.csv (new cutoff column only)',
                           'knowledge.csv', 'canonical_split.json', 'prepared_index.json',
                           'verification.json', 'manifest.json'],
            'stays_with_first_result': len(table),
            'estimated_clipped_at_outtime': clip_share(inherit, table),
            'temporal_clean': False, 'early_triage_eligible': False}


def verify(root, inherit, delay, table):
    """Read the whole output and re-derive every cutoff from the inherited index."""
    out = _ro(root / 'events.sqlite')
    source = _ro(inherit / 'events.sqlite')
    schema = event_schema(out)
    if schema != event_schema(source) or \
            list(out.execute('PRAGMA table_info(events)')) != list(source.execute('PRAGMA table_info(events)')):
        raise ValueError('Output events schema differs from the inherited index')
    source.close()
    new_first = dict(out.execute('SELECT stay,available FROM first_results'))
    old_events = _ro(inherit / 'events.sqlite').execute('SELECT COUNT(*) FROM events').fetchone()[0]
    events = out.execute('SELECT COUNT(*) FROM events').fetchone()[0]
    out.close()
    if events != old_events or set(new_first) != set(table):
        raise ValueError('Event rows or first_results stays differ from the inherited index')
    with (inherit / 'cohort.csv').open(newline='') as a, (root / 'cohort.csv').open(newline='') as b:
        old_rows, new_rows = list(csv.DictReader(a)), list(csv.DictReader(b))
    if len(old_rows) != len(new_rows):
        raise ValueError('Cohort length changed')
    moved = 0
    for old, new in zip(old_rows, new_rows):
        if any(old[k] != new[k] for k in ('sample_id', 'subject_id', 'stay_id', 'split')):
            raise ValueError('Cohort identity or split changed')
        first, expected, start, finish = table[new['stay_id']]
        if first != old['cutoff']:
            raise ValueError('Inherited cutoff differs from inherited first_results')
        if new['cutoff'] != expected or new_first[new['stay_id']] != expected:
            raise ValueError('cutoff != min(first + delay, finish)')
        if not timestamp(start) <= timestamp(new['cutoff']) <= timestamp(finish) or new['cutoff'] < old['cutoff']:
            raise ValueError('Cutoff outside visit interval or earlier than inherited cutoff')
        moved += int(new['cutoff'] != old['cutoff'])
    return {'status': 'verified', 'cohort_rows': len(new_rows), 'events_rows': events,
            'cutoff_violations': 0, 'identity_violations': 0, 'rows_with_later_cutoff': moved,
            'delay_minutes': delay, 'event_schema': schema,
            'method': 'Full cohort readback against the inherited index: identity/split equality and cutoff == min(first+delay, finish).'}


def execute(args):
    os.umask(0o077)
    inherit, old = load_inherited(args.inherit_from)
    root = args.output.resolve()
    if sha256(inherit / 'events.sqlite') != old['event_index_sha256'] or sha256(inherit / 'cohort.csv') != old['cohort_sha256']:
        raise ValueError('Inherited index/cohort do not match their manifest')
    for suffix in ('-wal', '-shm', '-journal'):
        if (inherit / ('events.sqlite' + suffix)).exists():
            raise ValueError('SQLite sidecar present in inherited artifact')
    rule = None
    if old.get('event_schema') == EVENT_SCHEMA_V5:
        # v5 inheritance keeps the representable-only first-result rule: every inherited first
        # must equal the earliest representable lab event of its stay.
        db = _ro(inherit / 'events.sqlite')
        try:
            rule = first_rule_violations(db)
        finally:
            db.close()
        if rule['first_differs_from_earliest_event'] or rule['first_without_event'] or rule['event_without_first']:
            raise ValueError('Inherited first_results violate the representable-only rule: ' + json.dumps(rule))
    table = cutoff_table(inherit, args.delay_minutes)
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    shutil.copyfile(inherit / 'events.sqlite', root / 'events.sqlite')
    db = sqlite3.connect(root / 'events.sqlite')
    db.executemany('UPDATE first_results SET available=? WHERE stay=?',
                   ((cutoff, stay) for stay, (_, cutoff, _, _) in table.items()))
    db.commit()
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    db.close()
    with (inherit / 'cohort.csv').open(newline='') as source, (root / 'cohort.csv').open('x', newline='') as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            row['cutoff'] = table[row['stay_id']][1]
            writer.writerow(row)
    for name in ('knowledge.csv', 'canonical_split.json'):
        shutil.copyfile(inherit / name, root / name)
    verification = verify(root, inherit, args.delay_minutes, table)
    if rule is not None:
        verification['inherited_representable_first_rule'] = dict(rule, violations=0)
    save_json(root / 'verification.json', verification)
    index_sha, cohort_sha = sha256(root / 'events.sqlite'), sha256(root / 'cohort.csv')
    shares = clip_share(root, {k: (v[0], v[1], v[2], v[3]) for k, v in table.items()})
    prepared = json.loads((inherit / 'prepared_index.json').read_text())
    evidence = {'cohort_sha256': cohort_sha, 'event_index_sha256': index_sha,
                'cohort_policy': old['cohort_policy'], 'index_counts': old['index_counts'],
                'visit_filter': old['visit_filter'], 'status': 'verified',
                'delay_verification_sha256': sha256(root / 'verification.json')}
    manifest = dict(old)
    manifest.update(
        status='index_ready', cutoff_policy=POLICY, delay_minutes=args.delay_minutes,
        decision_policy=(DECISION_POLICY_V5 if rule is not None else DECISION_POLICY).format(delay=args.delay_minutes),
        inherits_from=str(inherit), inherits_index_sha256=old['event_index_sha256'],
        inherits_cohort_sha256=old['cohort_sha256'], inherits_manifest_sha256=sha256(inherit / 'manifest.json'),
        cutoff_clipped_at_outtime=shares, cutoff_clipped_at_outtime_note='train and validation only; test is never counted',
        temporal_clean=False, early_triage_eligible=False, graph_status='absent', graph_regeneration_required=True,
        event_index_sha256=index_sha, cohort_sha256=cohort_sha, index_verification=evidence,
        metadata_repair={'version': 'event_index_metadata_repair_v1',
                         'origin': 'first_lab_delay native index_ready receipt; no repair applied',
                         'source_code_sha256': sha256(Path(__file__))})
    prepared.update(status='index_ready', cohort_sha256=cohort_sha, event_index_sha256=index_sha,
                    temporal_clean=False, early_triage_eligible=False, graph_status='absent',
                    graph_regeneration_required=True, index_verification=evidence,
                    metadata_repair=manifest['metadata_repair'])
    save_json(root / 'prepared_index.json', prepared)
    save_json(root / 'manifest.json', manifest)
    from data.s2_events.repair_metadata import validate_index_ready
    validate_index_ready(root, json.loads((root / 'manifest.json').read_text()))
    print(json.dumps({'status': 'completed', 'output': str(root), 'cohort_sha256': cohort_sha,
                      'event_index_sha256': index_sha, 'clipped_at_outtime': shares}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inherit-from', type=Path, required=True,
                        help='index_ready first_lab artifact supplying events.sqlite, cohort and folds')
    parser.add_argument('--delay-minutes', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.delay_minutes < 1:
        parser.error('--delay-minutes must be positive')
    if args.output.exists():
        parser.error('Output exists; refusing to overwrite: ' + str(args.output))
    if args.execute:
        execute(args)
    else:
        print(json.dumps(plan(args), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
