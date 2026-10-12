"""Event index v5: lab events carry the censor sign and the lab's own reference range.

`EventStoreV5` behaves like `ingest.EventStore` (same constructor, `chunks`, `insert`,
`close`, `counts`) but its `events` table has four extra trailing columns:
censored TEXT ('<', '>' or NULL), ref_lower REAL, ref_upper REAL, flag TEXT.
`insert` accepts the old 10-tuple (extra columns NULL) or the 14-tuple.
The v5 contract also fixes how a lab row becomes an event (`classify_lab_row`) and
which events may set the first-lab cutoff (representable results only).
"""
import re

from .ingest import EventStore
from .schema import numeric

EVENT_SCHEMA_V5 = 'event_index_v5'
V5_COLUMNS = ('censored', 'ref_lower', 'ref_upper', 'flag')
LAB_COLUMNS_V5 = ['labevent_id', 'subject_id', 'itemid', 'charttime', 'storetime', 'value', 'valuenum',
                  'valueuom', 'ref_range_lower', 'ref_range_upper', 'flag', 'comments']

UNKNOWN_UNITS = {'', '?', 'unknown', 'unk', 'nan', 'none', 'null', 'n/a', 'na', 'not available'}
# Closed list of dimensionless analytes whose source unit is blank. The specimen indices
# 50934/50947/51678 (H/I/L) are deliberately absent: they stay dropped.
UNITLESS_ITEMS = {'51237': 'ratio', '51675': 'ratio', '51498': 'ratio'}

# Anchored: a result that merely contains '>75' mid-sentence (eGFR 50920) never matches.
CENSOR_RE = re.compile(
    r'^\s*(<|>|LESS THAN|GREATER THAN)\s*(?:OR EQUAL TO\s*)?=?\s*'
    r'([0-9]*\.?[0-9]+)(?![0-9])(?!,[0-9])(?![eE][0-9+-])', re.I)


class EventStoreV5(EventStore):
    def __init__(self, path):
        super().__init__(path)
        for column, kind in zip(V5_COLUMNS, ('TEXT', 'REAL', 'REAL', 'TEXT')):
            self.db.execute('ALTER TABLE events ADD COLUMN %s %s' % (column, kind))
        self.db.execute('CREATE TABLE event_schema (name TEXT PRIMARY KEY)')
        self.db.execute('INSERT INTO event_schema VALUES (?)', (EVENT_SCHEMA_V5,))
        self.db.commit()

    def insert(self, event):
        if len(event) == 10:
            event = tuple(event) + (None, None, None, None)
        self.db.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', event)
        self.counts['events:' + event[8].split(':')[0]] += 1


def event_schema(db):
    """EVENT_SCHEMA_V5 if the sqlite index has the v5 events columns, else None."""
    columns = {row[1] for row in db.execute('PRAGMA table_info(events)')}
    return EVENT_SCHEMA_V5 if set(V5_COLUMNS) <= columns else None


def parse_censored(text):
    """(sign, limit) for a text starting with a comparator and a number, else None."""
    match = CENSOR_RE.match(text or '')
    if not match:
        return None
    sign = '<' if match.group(1)[0] in 'lL<' else '>'
    return sign, float(match.group(2))


def classify_lab_row(row):
    """Return (outcome, value, censored, unit, ref_lower, ref_upper, flag, source).

    outcome is 'event' for a representable result; otherwise a drop reason:
    'text_result' (nonempty text, no limit), 'empty_result', 'nonnumeric_value',
    'unit_missing' (representable value, but no source unit and not allowlisted).
    `source` is 'numeric', 'censored_value', 'censored_comment' or None.
    Unit-less allowlisted rows get the allowlist unit and outcome 'event'
    (the caller counts them via the returned `unit` differing from valueuom).
    """
    item = row['itemid']
    ref_lower, ref_upper = numeric(row['ref_range_lower']), numeric(row['ref_range_upper'])
    flag = row['flag'].strip() or None
    value, censored, source = numeric(row['valuenum']), None, 'numeric'
    text = row['value'].strip()
    if value is None:
        found = None
        if text:
            found, source = parse_censored(text), 'censored_value'
            if found is None:
                return ('text_result', None, None, None, ref_lower, ref_upper, flag, None)
        else:
            found, source = parse_censored(row['comments']), 'censored_comment'
            if found is None:
                return ('empty_result', None, None, None, ref_lower, ref_upper, flag, None)
        censored, value = found
        if ref_upper is not None and abs(ref_upper - value) <= 1e-9 * max(1.0, abs(value)):
            value = ref_upper
    unit = row['valueuom']
    if unit.strip().casefold() in UNKNOWN_UNITS:
        unit = UNITLESS_ITEMS.get(item)
        if unit is None:
            return ('unit_missing', value, censored, None, ref_lower, ref_upper, flag, source)
    return ('event', value, censored, unit, ref_lower, ref_upper, flag, source)


def first_rule_violations(db):
    """Stays whose first_results.available differs from the earliest lab event they hold.

    v5 rule: first_results = min(available) over the stay's lab events (all representable)
    with available <= visit finish. Returns counts of mismatching / missing / extra stays.
    """
    sql = ('SELECT e.stay, MIN(e.available) m FROM events e JOIN visits v ON v.stay=e.stay '
           "WHERE e.source LIKE 'labevents.csv:%' AND v.finish IS NOT NULL AND e.available<=v.finish "
           'GROUP BY e.stay')
    expected = dict(db.execute(sql))
    actual = dict(db.execute('SELECT stay, available FROM first_results'))
    return {'stays_checked': len(actual),
            'first_differs_from_earliest_event': sum(1 for s in actual if s in expected and expected[s] != actual[s]),
            'first_without_event': sum(1 for s in actual if s not in expected),
            'event_without_first': sum(1 for s in expected if s not in actual)}
