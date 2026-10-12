"""Read-only access to the v1 event index plus a v2-local triage/arrival index.

The 18 GB `labevents.csv` scan is NOT repeated. The v1 artifact already produced a
source-bound SQLite index of every cutoff-eligible lab result, and the producer
verifies its sha256 against the v1 manifest before reading a single row. Only the
cheap tables v1 deliberately excluded -- triage and arrival demographics -- are
ingested here, into a separate v2-owned database that never mutates v1.
"""
from collections import Counter, defaultdict
import sqlite3

from core.schema import (ARRIVAL_FIELDS, FORBIDDEN_EDSTAYS_FIELDS, TRIAGE_VITALS,
                     VITAL_RANGES, identifier, normalize_complaint, numeric, rows,
                     split_complaints, timestamp)


V5_EVENT_COLUMNS = ('censored', 'ref_lower', 'ref_upper', 'flag')
AGE_POLICIES = ('anchor', 'visit')
AGE_CAP = 91


class ClinicalStore:
    """Joins the inherited read-only event index with the v2 triage index."""

    def __init__(self, event_index, triage_index):
        self.events = sqlite3.connect('file:' + str(event_index) + '?mode=ro', uri=True)
        self.triage = sqlite3.connect('file:' + str(triage_index) + '?mode=ro', uri=True)
        self.events.row_factory = sqlite3.Row
        self.triage.row_factory = sqlite3.Row
        self._visits = {}
        self._prior = {}
        # I2: v5 event indexes carry censored/ref_lower/ref_upper/flag. Older artifacts
        # do not; their records get None for these keys and graphs stay byte-identical.
        present = {r['name'] for r in self.events.execute('PRAGMA table_info(events)')}
        self.v5_columns = tuple(c for c in V5_EVENT_COLUMNS if c in present)
        self.has_v5_columns = len(self.v5_columns) == len(V5_EVENT_COLUMNS)
        self._event_select = ('id, stay, token, value, unit, time, available, source, timing_basis'
                              + ''.join(', ' + (c if c in present else 'NULL AS ' + c)
                                        for c in V5_EVENT_COLUMNS))

    @staticmethod
    def _record(r):
        return {'id': r['id'], 'stay': r['stay'],
                'token': r['token'], 'value': r['value'], 'unit': r['unit'],
                'time': r['time'], 'available': r['available'], 'source': r['source'],
                'timing_basis': r['timing_basis'],
                'censored': r['censored'], 'ref_lower': r['ref_lower'],
                'ref_upper': r['ref_upper'], 'flag': r['flag']}

    def close(self):
        self.events.close()
        self.triage.close()

    # ------------------------------------------------------------------ encounters
    def visit(self, stay):
        if stay not in self._visits:
            row = self.events.execute(
                'SELECT stay, subject, start, finish FROM visits WHERE stay=?', (stay,)).fetchone()
            self._visits[stay] = dict(row) if row else None
        return self._visits[stay]

    def prior_visits(self, subject, index_start):
        """Completed encounters strictly before the index encounter began.

        An overlapping or unfinished earlier encounter is not settled history, so
        `finish < index_start` is required as well as `start < index_start`.
        """
        key = (subject, index_start)
        if key not in self._prior:
            self._prior[key] = [r['stay'] for r in self.events.execute(
                'SELECT stay FROM visits WHERE subject=? AND start<? AND finish IS NOT NULL '
                'AND finish<? ORDER BY start, stay', (subject, index_start, index_start))]
        return self._prior[key]

    # ------------------------------------------------------------------ triage tier
    def visit_age(self, stay):
        """min(anchor_age + year(visit start) - anchor_year, 91) for a 'visit'-policy triage index.

        Re-derived from the raw anchors and the event index's visit start; None when either
        anchor, the visit or the table (anchor-policy index) is missing.
        """
        try:
            row = self.triage.execute(
                'SELECT anchor_age, anchor_year FROM age_source WHERE stay=?', (stay,)).fetchone()
        except sqlite3.OperationalError:
            return None
        visit = self.visit(stay)
        if row is None or visit is None or row['anchor_age'] is None or row['anchor_year'] is None:
            return None
        return min(row['anchor_age'] + timestamp(visit['start']).year - row['anchor_year'], AGE_CAP)

    def arrival(self, stay):
        row = self.triage.execute('SELECT * FROM arrival WHERE stay=?', (stay,)).fetchone()
        return dict(row) if row else {}

    def complaints(self, stay):
        return [r['token'] for r in self.triage.execute(
            'SELECT token FROM complaints WHERE stay=? ORDER BY ordinal', (stay,))]

    def triage_vitals(self, stay):
        return [(r['field'], r['value'], r['unit'], r['source_unit'])
                for r in self.triage.execute(
                    'SELECT field, value, unit, source_unit FROM vitals WHERE stay=? ORDER BY field', (stay,))]

    # ---------------------------------------------------------------- measurements
    def measurements(self, stay, cutoff):
        """Every representable result of this encounter available by the cutoff."""
        iso = cutoff.isoformat()
        return [self._record(r) for r in self.events.execute(
            'SELECT ' + self._event_select + ' '
            'FROM events WHERE stay=? AND time<=? AND available<=? ORDER BY time, id',
            (stay, iso, iso))]

    def analyte_history(self, subject, prior_stays, token, cutoff):
        """Prior-visit results of ONE analyte, oldest first.

        Restricting to a named analyte is what keeps this artifact small: v1 pulled
        every historical event of every kind, which is where its 18 GB went.
        """
        if not prior_stays:
            return []
        iso = cutoff.isoformat()
        placeholders = ','.join('?' * len(prior_stays))
        return [self._record(r) for r in self.events.execute(
            'SELECT ' + self._event_select + ' '
            'FROM events WHERE subject=? AND token=? AND stay IN (%s) '
            'AND time<=? AND available<=? ORDER BY time, id' % placeholders,
            (subject, token, *prior_stays, iso, iso))]


def build_triage_index(path, raw_root, stays, age_policy='anchor'):
    """Ingest triage + arrival demographics for the cohort stays only.

    Both tables are small (~39 MB and ~40 MB) and are read once. Values that fail a
    plausibility gate are counted and dropped, never clipped. `disposition` is
    refused outright: it is the visit's outcome, not decision-time evidence.

    age_policy 'anchor' (default, unchanged) stores patients.anchor_age. 'visit' stores
    anchor_age + year(edstays.intime) - anchor_year, capped at 91 (EVT-05/BLD-04).
    """
    if age_policy not in AGE_POLICIES:
        raise ValueError('Unknown age policy: %r' % (age_policy,))
    db = sqlite3.connect(path)
    path.chmod(0o600)
    db.executescript('''
        CREATE TABLE arrival (stay TEXT PRIMARY KEY, subject TEXT NOT NULL,
            gender TEXT, race TEXT, arrival_transport TEXT, age REAL, acuity REAL);
        CREATE TABLE complaints (stay TEXT NOT NULL, ordinal INTEGER NOT NULL,
            token TEXT NOT NULL, PRIMARY KEY (stay, ordinal));
        CREATE TABLE vitals (stay TEXT NOT NULL, field TEXT NOT NULL, value REAL NOT NULL,
            unit TEXT NOT NULL, source_unit TEXT NOT NULL, PRIMARY KEY (stay, field));
    ''')
    counts = Counter()

    ages, anchor_years = {}, {}
    patient_fields = {'subject_id', 'anchor_age'} | ({'anchor_year'} if age_policy == 'visit' else set())
    for row in rows(raw_root / 'patients.csv', patient_fields):
        value = numeric(row['anchor_age'])
        if value is not None:
            ages[row['subject_id']] = value
        if age_policy == 'visit':
            year = numeric(row['anchor_year'])
            if year is not None:
                anchor_years[row['subject_id']] = year
    counts['patients:with_anchor_age'] = len(ages)

    arrival_rows = []
    age_source_rows = []
    required_arrival = {'subject_id', 'stay_id', *ARRIVAL_FIELDS}
    if age_policy == 'visit':
        required_arrival.add('intime')
    if FORBIDDEN_EDSTAYS_FIELDS & required_arrival:
        raise ValueError('Outcome field requested from edstays; disposition is not decision-time evidence')
    for row in rows(raw_root / 'edstays.csv', required_arrival):
        if row['stay_id'] not in stays:
            continue
        identifier(row['subject_id'])
        identifier(row['stay_id'])
        age = ages.get(row['subject_id'])
        if age_policy == 'visit' and age is not None:
            anchor_year = anchor_years.get(row['subject_id'])
            if anchor_year is None:
                counts['age:visit_missing_anchor_year'] += 1
                age = None
            else:
                raw_age = age + timestamp(row['intime']).year - anchor_year
                age = min(raw_age, AGE_CAP)
                counts['age:visit_changed'] += int(age != ages[row['subject_id']])
                counts['age:visit_capped_91'] += int(raw_age > AGE_CAP)
        arrival_rows.append((row['stay_id'], row['subject_id'],
                             row['gender'].strip() or None, row['race'].strip() or None,
                             row['arrival_transport'].strip() or None,
                             age, None))
        if age_policy == 'visit':
            age_source_rows.append((row['stay_id'], ages.get(row['subject_id']),
                                    anchor_years.get(row['subject_id'])))
        counts['arrival:rows'] += 1
    db.executemany('INSERT INTO arrival VALUES (?,?,?,?,?,?,?)', arrival_rows)
    if age_policy == 'visit':
        # Raw anchors, so verify_artifact can re-derive the age without trusting arrival.age.
        db.execute('CREATE TABLE age_source (stay TEXT PRIMARY KEY, anchor_age REAL, anchor_year REAL)')
        db.executemany('INSERT INTO age_source VALUES (?,?,?)', age_source_rows)
    db.commit()

    complaint_rows, vital_rows, acuities = [], [], []
    for row in rows(raw_root / 'triage.csv', {'subject_id', 'stay_id', 'chiefcomplaint',
                                              'acuity', *TRIAGE_VITALS}):
        if row['stay_id'] not in stays:
            continue
        counts['triage:rows'] += 1
        tokens = split_complaints(row['chiefcomplaint'])
        if not tokens:
            counts['triage:no_complaint'] += 1
        for ordinal, token in enumerate(tokens):
            complaint_rows.append((row['stay_id'], ordinal, token))
        acuity = numeric(row['acuity'])
        if acuity is not None:
            acuities.append((acuity, row['stay_id']))
        for field, unit in TRIAGE_VITALS.items():
            value = numeric(row[field])
            if value is None:
                counts['vitals:missing:' + field] += 1
                continue
            source_unit = 'Fahrenheit' if field == 'temperature' else unit
            if field == 'temperature':
                value = (value - 32.) * 5. / 9.
            low, high = VITAL_RANGES[field]
            if not low <= value <= high:
                counts['vitals:out_of_range:' + field] += 1
                continue
            vital_rows.append((row['stay_id'], field, value, unit, source_unit))
    db.executemany('INSERT INTO complaints VALUES (?,?,?)', complaint_rows)
    db.executemany('INSERT INTO vitals VALUES (?,?,?,?,?)', vital_rows)
    db.executemany('UPDATE arrival SET acuity=? WHERE stay=?', acuities)
    db.commit()
    counts['complaints:rows'] = len(complaint_rows)
    counts['vitals:rows'] = len(vital_rows)
    counts['acuity:rows'] = len(acuities)
    db.close()
    return dict(counts)


def complaint_counts(triage_index, train_stays):
    """Count complaint surface forms over TRAIN-fold visits only."""
    db = sqlite3.connect('file:' + str(triage_index) + '?mode=ro', uri=True)
    counts = Counter()
    for stay, token in db.execute('SELECT stay, token FROM complaints'):
        if stay in train_stays:
            counts[token] += 1
    db.close()
    return counts
