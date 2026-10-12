"""Prior-visit diagnosis history for the v3 clinical graph.

The single hard rule: the index encounter's own diagnoses are the LABEL. They may
never enter a graph. This module therefore never accepts the index `stay_id` in a
diagnosis query, and `build.py` re-derives the same guarantee after writing by
counting how many emitted diagnosis nodes trace back to the index stay. A nonzero
count fails the production.

Timing caveat, stated because it cannot be resolved from the source: MIMIC's
`diagnosis.csv` carries no timestamp at all -- only `stay_id` and `seq_num`. So
"before the cutoff" is established structurally (the diagnosis belongs to an
encounter that had already FINISHED before the index encounter began), never by a
recorded diagnosis time. Availability at encounter completion is an unverified
assumption: this source does not prove when a billing diagnosis became visible.

ICD-9 and ICD-10 can occur in different encounters of the same patient. Comparing
raw codes across that boundary misses recurrences, so both history and targets
use the shared, historical target producer's GEM row-selection policy.
Approximate mappings are flagged on the node, never silently treated as exact.
"""
from collections import Counter, defaultdict
from data.s3_labels.icd_mapping import load_icd_map, normalize_icd10

from core.schema import rows

DIAGNOSIS_POLICIES = ('v3', 'v5')

# v5 single-era check: categories that legitimately exist in one coding era only.
SINGLE_ERA_ALLOWLIST = frozenset({'W22', 'Z3A', 'R87', 'J90', 'W00', 'L76', 'S46', 'J10',
                                  'I16', 'F17', 'T36', 'E13', 'B18', 'T07', 'E08'})
SINGLE_ERA_MIN_COUNT = 20


def _check_policy(policy):
    if policy not in DIAGNOSIS_POLICIES:
        raise ValueError('Unknown diagnosis policy: %r (expected one of %s)'
                         % (policy, ', '.join(DIAGNOSIS_POLICIES)))


def load_icd_map_for_policy(path, policy='v3'):
    """Return (mapping, approximate) for the policy. v3 = icd_mapping.load_icd_map.

    v5 goes through data.s3_labels.icd_mapping_v2 (GEM + tracked overrides); its
    mapping holds None for ICD-9 codes without a defensible single target.
    """
    _check_policy(policy)
    if policy == 'v3':
        return load_icd_map(path)
    from data.s3_labels.icd_mapping_v2 import load_icd_map_v2
    return load_icd_map_v2(path)


def normalize(version, code, mapping, approximate, policy='v3'):
    """Return (token, is_approximate). ICD-10 passes through; ICD-9 goes via GEM.

    An unmappable ICD-9 code keeps its own namespace instead of being forced into
    the ICD-10 space, so it can still match other ICD-9 occurrences of itself.
    policy='v5' returns only the token pair too; use `normalize_v5` for the full code.
    """
    _check_policy(policy)
    if policy == 'v5':
        token, _full, approx = normalize_v5(version, code, mapping, approximate)
        return token, approx
    code = code.strip().upper()
    target = normalize_icd10(version, code, mapping)
    if not code:
        return None, False
    if version == '10':
        return 'dx:icd10:' + code, False
    if version == '9':
        if target is None:
            return 'dx:icd9:' + code, False
        return 'dx:icd10:' + target, code in approximate


def normalize_v5(version, code, mapping, approximate):
    """Return (token, full_code, is_approximate) under the v5 policy.

    Token is dx:icd10:<3-char ICD-10 category>. ICD-9 codes go through the I4 mapping
    (icd_mapping_v2); a code mapped to None keeps dx:icd9:<code> (full_code = the
    ICD-9 code) so it can still match its own recurrences.
    """
    from data.s3_labels.icd_mapping_v2 import normalize_icd10_v2
    code = code.strip().upper()
    if not code:
        return None, None, False
    target = normalize_icd10_v2(version, code, mapping)
    if version == '10':
        return 'dx:icd10:' + code[:3], code, False
    if version == '9':
        if target is None:
            return 'dx:icd9:' + code, code, False
        target = target.strip().upper()
        return 'dx:icd10:' + target[:3], target, code in approximate
    raise ValueError('Unsupported ICD version: ' + str(version))


def single_era_tokens(era_counts, min_count=SINGLE_ERA_MIN_COUNT,
                      allowlist=SINGLE_ERA_ALLOWLIST):
    """Tokens seen >= min_count times from exactly one source_version, minus allowlist.

    era_counts: token -> Counter(source_version -> n). Returns a sorted list of
    {'token', 'source_version', 'n'} for build.py to record. Unmapped ICD-9
    (dx:icd9:*) tokens are single-era by construction and are skipped; allowlist
    entries are 3-char categories compared against the token's category.
    """
    out = []
    for token in sorted(era_counts):
        if token.startswith('dx:icd9:'):
            continue
        counter = era_counts[token]
        present = [v for v, n in counter.items() if n > 0]
        total = sum(counter.values())
        if len(present) != 1 or total < min_count:
            continue
        if token.rsplit(':', 1)[-1] in allowlist:
            continue
        out.append({'token': token, 'source_version': present[0], 'n': total})
    return out


class DiagnosisIndex:
    """Diagnoses of the cohort's encounters, keyed by stay.

    Holding index-encounter diagnoses in memory is deliberate: `build.py` needs them
    to PROVE none of them reached a graph. Access is through `prior_history`, which
    refuses the index stay outright, so the guarded and unguarded paths cannot be
    confused at a call site.
    """

    def __init__(self, path, stays, mapping, approximate, policy='v3'):
        _check_policy(policy)
        self.policy = policy
        self.by_stay = defaultdict(list)
        self.counts = defaultdict(int)
        # token -> Counter(source_version); filled under v5 only (see single_era_tokens)
        self.era_counts = defaultdict(Counter)
        self.mapping = mapping
        self.approximate = approximate
        for row in rows(path, {'subject_id', 'stay_id', 'seq_num', 'icd_code',
                               'icd_version', 'icd_title'}):
            stay = row['stay_id']
            if stay not in stays:
                continue
            self.counts['rows'] += 1
            full_code = None
            if policy == 'v5':
                token, full_code, approx = normalize_v5(row['icd_version'], row['icd_code'],
                                                        mapping, approximate)
            else:
                token, approx = normalize(row['icd_version'], row['icd_code'],
                                          mapping, approximate)
            if token is None:
                self.counts['empty_code'] += 1
                continue
            if token.startswith('dx:icd9:'):
                self.counts['unmapped_icd9'] += 1
            if approx:
                self.counts['approximate_mapping'] += 1
            if policy == 'v5':
                self.era_counts[token][row['icd_version']] += 1
            record = {
                'stay': stay,
                'token': token,
                'seq_num': int(row['seq_num']) if row['seq_num'].strip().isdigit() else None,
                'source_version': row['icd_version'],
                'source_code': row['icd_code'].strip(),
                'title': row['icd_title'].strip(),
                'approximate_mapping': approx,
            }
            if policy == 'v5':
                record['full_code'] = full_code
            self.by_stay[stay].append(record)

    def single_era_tokens(self, min_count=SINGLE_ERA_MIN_COUNT, allowlist=SINGLE_ERA_ALLOWLIST):
        """v5 single-era check over the indexed rows; empty list under v3."""
        return single_era_tokens(self.era_counts, min_count, allowlist)

    def index_tokens(self, stay):
        """Diagnoses of the index encounter -- the LABEL. Never emitted as nodes.

        Exposed only so the producer can verify that none of them appear in the graph.
        """
        return {d['token'] for d in self.by_stay.get(stay, ())}

    def prior_history(self, index_stay, prior_stays):
        """Deduplicated diagnosis history from COMPLETED prior encounters only.

        Refuses the index stay even if a caller passes it in the prior list, so the
        guarantee does not depend on the caller getting the list right.
        """
        prior_stays = list(dict.fromkeys(prior_stays))
        if index_stay in prior_stays:
            raise ValueError('Index encounter is not prior history; its diagnoses are the label')
        history = {}
        for stay in prior_stays:
            for record in self.by_stay.get(stay, ()):
                if record['stay'] == index_stay:
                    raise ValueError('Index-encounter diagnosis reached the history path')
                existing = history.get(record['token'])
                # Recurrence is a count of completed stays, not diagnosis rows.
                # Multiple source codes can also map to the same token in a stay.
                if existing is None:
                    history[record['token']] = {**record, 'occurrences': {stay},
                                                'titles': {record['title']},
                                                'approximate_mapping': record['approximate_mapping']}
                    if 'full_code' in record:
                        history[record['token']]['full_codes'] = {record['full_code']}
                        history[record['token']]['source_versions'] = {record['source_version']}
                else:
                    existing['occurrences'].add(stay)
                    existing['titles'].add(record['title'])
                    existing['approximate_mapping'] |= record['approximate_mapping']
                    if 'full_code' in record:
                        existing['full_codes'].add(record['full_code'])
                        existing['source_versions'].add(record['source_version'])
        result = []
        for token in sorted(history):
            item = {**history[token], 'occurrences': sorted(history[token]['occurrences'])}
            if 'full_codes' in item:
                # v5: one category token spans many full codes; expose them all and
                # make `full_code` the deterministic (smallest) one.
                item['full_codes'] = sorted(item['full_codes'])
                item['source_versions'] = sorted(item['source_versions'])
                item['full_code'] = item['full_codes'][0]
            result.append(item)
        return result
