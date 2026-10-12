"""T2 (SC2, SC3): threshold states/combos agree with the row-range level; anion gap uses
the record's own upper limit.

Synthetic store and threshold rows only: no patient data, no event index, no test fold.
The state/combo decision lives inside `build_graph`, so these tests drive it end to end.
"""
from core.schema import timestamp
from data.s4_graph.graph import build_graph

K, GLUCOSE, ANION_GAP, NO_LEVEL = 'lab:50822', 'lab:50931', 'lab:50868', 'lab:99999'
HYPO, HYPER, DKA, ODD = ('knowledge:state:hypokalemia', 'knowledge:state:hyperkalemia',
                         'knowledge:state:dka_pattern', 'knowledge:state:odd_low')

SAMPLE = {'stay_id': 'index', 'subject_id': 'subject', 'cutoff': timestamp('2020-01-03T01:00:00'),
          'sample_id': 'synthetic', 'split': 'train'}


class FakeStore:
    has_v5_columns = True

    def __init__(self, records):
        self.records = records

    def visit(self, stay):
        return {'subject': 'subject', 'start': '2020-01-03T00:00:00', 'finish': '2020-01-03T02:00:00'}

    def arrival(self, stay):
        return {'gender': 'F'}

    def complaints(self, stay):
        return []

    def triage_vitals(self, stay):
        return []

    def prior_visits(self, subject, start):
        return []

    def measurements(self, stay, cutoff):
        return self.records

    def analyte_history(self, subject, prior_visits, token, cutoff):
        return []


def lab(identifier, token, unit, value, ref_lower=None, ref_upper=None, censored=None, when='00:30'):
    stamp = '2020-01-03T%s:00' % when
    return {'id': identifier, 'source': identifier, 'token': token, 'time': stamp,
            'available': stamp, 'value': value, 'unit': unit, 'timing_basis': 'synthetic',
            'stay': 'index', 'censored': censored, 'ref_lower': ref_lower, 'ref_upper': ref_upper,
            'flag': None}


def rule(rule_id, kind, token, unit, comparator, threshold, target):
    return {'rule_id': rule_id, 'kind': kind, 'source_token': token, 'unit': unit,
            'comparator': comparator, 'threshold': float(threshold), 'sex': 'any',
            'target_token': target, 'relation': 'indicates', 'source_uri': 'synthetic',
            'source_version': 'synthetic', 'reviewed_by': 'synthetic'}


THRESHOLDS = [
    rule('lvl_k_low', 'level', K, 'mEq/L', '<', 3.5, 'level:low'),
    rule('lvl_k_high', 'level', K, 'mEq/L', '>', 5.5, 'level:high'),
    rule('state_k_hypo', 'state', K, 'mEq/L', '<', 3.5, HYPO),
    rule('state_k_hyper', 'state', K, 'mEq/L', '>', 5.5, HYPER),
    rule('lvl_ag_low', 'level', ANION_GAP, 'mEq/L', '<', 10, 'level:low'),
    rule('lvl_ag_high', 'level', ANION_GAP, 'mEq/L', '>', 18, 'level:high'),
    rule('combo_dka', 'combo', GLUCOSE, 'mg/dL', '>', 250, DKA),
    rule('combo_dka', 'combo', ANION_GAP, 'mEq/L', '>', 12, DKA),
    # A state on a token that has no level row at all.
    rule('state_odd', 'state', NO_LEVEL, 'u', '<', 5, ODD),
]


def build(records, level_source, thresholds=THRESHOLDS):
    return build_graph(FakeStore(records), SAMPLE, [], set(), thresholds=thresholds,
                       level_source=level_source)


def fired(graph):
    return set(graph['coverage']['threshold_states'])


def level_of(graph, token):
    return next(n['level'] for n in graph['nodes'] if n['token'] == token and n['kind'] == 'measurement')


def potassium(value, censored=None, lower=3.3, upper=5.1):
    return [lab('k', K, 'mEq/L', value, lower, upper, censored)]


def dka(anion_gap, lower=8.0, upper=20.0):  # lower/upper may be None (no own range)
    return [lab('g', GLUCOSE, 'mg/dL', 300.0),
            lab('ag', ANION_GAP, 'mEq/L', anion_gap, lower, upper)]


# ------------------------------------------------------------------ SC2: low / high states
def test_row_range_potassium_3_4_is_low_by_the_textbook_limit_and_fires_the_hypokalemia_row():
    # v5c (SC1): serum / blood-gas K and Na levels use the CSV limits, not the own range 3.3-5.1.
    graph = build(potassium(3.4), 'row-range')
    assert level_of(graph, K) == 'low'
    assert HYPO in fired(graph)


def test_row_range_low_potassium_fires_the_hypokalemia_row():
    graph = build(potassium(3.2), 'row-range')
    assert level_of(graph, K) == 'low'
    assert HYPO in fired(graph)


def test_row_range_potassium_5_6_with_own_upper_6_0_is_high_by_the_textbook_limit_and_fires():
    # 5.6 is above the CSV 5.0 / 5.5 although inside this record's own range 3.3-6.0 (v5c, SC1).
    graph = build(potassium(5.6, upper=6.0), 'row-range')
    assert level_of(graph, K) == 'high'
    assert HYPER in fired(graph)
    assert HYPER in fired(build(potassium(5.6, upper=6.0), 'csv'))
    high = build(potassium(5.6), 'row-range')
    assert level_of(high, K) == 'high' and HYPER in fired(high)


def test_row_range_potassium_ignores_an_own_range_that_would_make_it_high():
    # Own range 1.0-3.0 would make 3.2 HIGH; K uses the textbook limits (v5c, SC1): low, fires.
    graph = build(potassium(3.2, lower=1.0, upper=3.0), 'row-range')
    assert level_of(graph, K) == 'low'
    assert HYPO in fired(graph)


def test_csv_mode_is_unchanged_k_3_4_still_fires_hypokalemia():
    graph = build(potassium(3.4), 'csv')
    assert level_of(graph, K) == 'low'
    assert HYPO in fired(graph)


def test_csv_mode_ignores_the_record_range_for_states():
    graph = build(potassium(3.2, lower=1.0, upper=3.0), 'csv')
    assert HYPO in fired(graph)


def test_node_without_a_level_row_keeps_csv_behaviour_in_row_range_mode():
    records = [lab('odd', NO_LEVEL, 'u', 4.0, 5.0, 9.0)]  # below its own range, no level row
    row_range, csv = build(records, 'row-range'), build(records, 'csv')
    assert level_of(row_range, NO_LEVEL) is None
    assert ODD in fired(row_range) and fired(row_range) == fired(csv)


def test_no_thresholds_row_range_builds_without_states():
    graph = build(potassium(3.4), 'row-range', thresholds=None)
    assert 'threshold_states' not in graph['coverage']


# ------------------------------------------------------------------------- censored values
def test_censored_below_limit_is_low_by_the_textbook_limit_and_fires_in_row_range():
    # '<3.4': true value below 3.4, certainly below the CSV level limit 3.5 (v5c, SC1) -> low -> state.
    graph = build(potassium(3.4, censored='<'), 'row-range')
    assert level_of(graph, K) == 'low'
    assert HYPO in fired(graph)
    assert HYPO in fired(build(potassium(3.4, censored='<'), 'csv'))  # csv: _holds('<', 3.5, 3.4, '<')


def test_censored_below_limit_that_is_certainly_low_still_fires():
    graph = build(potassium(3.0, censored='<'), 'row-range')
    assert level_of(graph, K) == 'low'
    assert HYPO in fired(graph)


def test_censored_above_limit_never_satisfies_a_low_row():
    for mode in ('csv', 'row-range'):
        assert HYPO not in fired(build(potassium(3.2, censored='>'), mode))


def test_censored_above_limit_over_own_upper_is_high_and_fires_hyperkalemia_only_with_certainty():
    graph = build(potassium(6.0, censored='>'), 'row-range')
    assert level_of(graph, K) == 'high' and HYPER in fired(graph)
    # '>5.2' is above own upper 5.1 (high) but the CSV row needs > 5.5: _holds decides.
    assert HYPER not in fired(build(potassium(5.2, censored='>'), 'row-range'))


# ------------------------------------------------------------------- SC3: anion gap upper
def test_anion_gap_15_with_own_range_8_20_does_not_satisfy_the_dka_combo():
    graph = build(dka(15.0), 'row-range')
    assert level_of(graph, ANION_GAP) == 'normal'
    assert DKA not in fired(graph)


def test_anion_gap_21_with_own_range_8_20_satisfies_the_dka_combo():
    graph = build(dka(21.0), 'row-range')
    assert level_of(graph, ANION_GAP) == 'high'
    assert DKA in fired(graph)


def test_anion_gap_20_equals_own_upper_is_not_above_it():
    assert DKA not in fired(build(dka(20.0), 'row-range'))


def test_anion_gap_15_in_csv_mode_still_satisfies_the_dka_combo():
    assert DKA in fired(build(dka(15.0), 'csv'))


def test_anion_gap_without_own_range_row_range_uses_the_csv_limits():
    # No own bounds: combo keeps its CSV '> 12' (no level gating): 15 fires, 11 does not.
    assert DKA in fired(build(dka(15.0, None, None), 'row-range'))
    assert DKA not in fired(build(dka(11.0, None, None), 'row-range'))
    graph = build(dka(19.0, None, None), 'row-range')
    assert level_of(graph, ANION_GAP) == 'high' and DKA in fired(graph)


def test_own_upper_tokens_constant_is_the_anion_gap_only():
    from data.s4_graph import graph as graph_module
    assert graph_module.OWN_UPPER_TOKENS == {'lab:50868'}


def test_only_the_anion_gap_token_swaps_in_the_own_upper_limit():
    # Glucose has an own range too, but its combo row keeps the CSV threshold 250.
    records = [lab('g', GLUCOSE, 'mg/dL', 300.0, 70.0, 400.0), lab('ag', ANION_GAP, 'mEq/L', 21.0, 8.0, 20.0)]
    assert DKA in fired(build(records, 'row-range'))


def test_combo_uses_latest_value_of_the_anion_gap():
    records = [lab('g', GLUCOSE, 'mg/dL', 300.0, when='00:10'),
               lab('ag1', ANION_GAP, 'mEq/L', 25.0, 8.0, 20.0, when='00:20'),
               lab('ag2', ANION_GAP, 'mEq/L', 15.0, 8.0, 20.0, when='00:40')]
    assert DKA not in fired(build(records, 'row-range'))


# ------------------------------------------- SC2 amended: combo rows keep their CSV cut-offs
HR, SBP, BICARB = 'vital:heartrate', 'vital:sbp', 'lab:50882'
SIRS, QSOFA, HHS = 'knowledge:state:sirs', 'knowledge:state:qsofa', 'knowledge:state:hhs'

COMBO_THRESHOLDS = [
    rule('lvl_hr_low', 'level', HR, 'bpm', '<', 60, 'level:low'),
    rule('lvl_hr_high', 'level', HR, 'bpm', '>', 100, 'level:high'),
    rule('combo_sirs_hr', 'combo', HR, 'bpm', '>', 90, SIRS),
    rule('lvl_sbp_low', 'level', SBP, 'mmHg', '<', 90, 'level:low'),
    rule('lvl_sbp_high', 'level', SBP, 'mmHg', '>', 180, 'level:high'),
    rule('combo_qsofa_sbp', 'combo', SBP, 'mmHg', '<=', 100, QSOFA),
    rule('lvl_bic_low', 'level', BICARB, 'mEq/L', '<', 22, 'level:low'),
    rule('lvl_bic_high', 'level', BICARB, 'mEq/L', '>', 32, 'level:high'),
    rule('combo_hhs', 'combo', BICARB, 'mEq/L', '>', 18, HHS),
]


def combo_build(records, level_source='row-range'):
    return build(records, level_source, thresholds=COMBO_THRESHOLDS)


def test_combo_hr_95_satisfies_gt_90_even_though_level_row_is_gt_100():
    records = [lab('hr', HR, 'bpm', 95.0)]
    graph = combo_build(records)
    assert level_of(graph, HR) == 'normal'
    assert SIRS in fired(graph)
    assert SIRS in fired(combo_build(records, 'csv'))


def test_combo_sbp_95_satisfies_le_100_even_though_level_low_is_lt_90():
    records = [lab('sbp', SBP, 'mmHg', 95.0)]
    graph = combo_build(records)
    assert level_of(graph, SBP) == 'normal'
    assert QSOFA in fired(graph)


def test_combo_bicarbonate_24_satisfies_gt_18_even_though_level_high_is_gt_32():
    records = [lab('bic', BICARB, 'mEq/L', 24.0)]
    graph = combo_build(records)
    assert level_of(graph, BICARB) == 'normal'
    assert HHS in fired(graph)


def test_combo_rows_still_respect_their_own_csv_cutoff_in_row_range():
    graph = combo_build([lab('hr', HR, 'bpm', 85.0), lab('sbp', SBP, 'mmHg', 105.0),
                         lab('bic', BICARB, 'mEq/L', 17.0)])
    assert fired(graph).isdisjoint({SIRS, QSOFA, HHS})


def test_combo_anion_gap_gt_12_uses_own_upper_not_csv_and_not_level():
    assert DKA not in fired(build(dka(15.0), 'row-range'))
    assert DKA in fired(build(dka(21.0), 'row-range'))


def test_state_row_fires_alongside_combo_rows_when_the_textbook_level_agrees():
    # K 3.4 is low by the textbook limit (own range 3.3-5.1 is ignored, v5c SC1) -> '< 3.5' fires.
    assert HYPO in fired(build(potassium(3.4), 'row-range'))


# ------------------------------- SC3: the own-upper swap needs a valid own range (same rule as level_for)
def test_anion_gap_with_only_an_upper_bound_uses_the_csv_threshold():
    # ref_upper 10 but no ref_lower: not a valid own range -> CSV '> 12'. 11 does not fire, 13 does.
    assert DKA not in fired(build(dka(11.0, None, 10.0), 'row-range'))
    assert DKA in fired(build(dka(13.0, None, 10.0), 'row-range'))


def test_anion_gap_with_inverted_own_bounds_uses_the_csv_threshold():
    assert DKA not in fired(build(dka(11.0, 20.0, 10.0), 'row-range'))
    assert DKA in fired(build(dka(13.0, 20.0, 10.0), 'row-range'))


def test_anion_gap_with_a_valid_own_range_below_the_csv_threshold_uses_own_upper():
    assert DKA in fired(build(dka(11.0, 3.0, 10.0), 'row-range'))


# ----------------------------- SC2: every comparator against the real STATE rows (rule data only)
import pathlib

import pytest

from core.schema import read_thresholds
from data.s4_graph.graph import TEXTBOOK_LEVEL_TOKENS, _row_holds, level_rows

REAL_THRESHOLDS = read_thresholds(
    str(pathlib.Path(__file__).resolve().parent.parent / 'data/s2_events/knowledge_thresholds_v2.csv'))
REAL_STATES = [r for r in REAL_THRESHOLDS if r['kind'] == 'state']
# Same side, other inclusiveness: the real file has no '<=' state row, so derive it from rule data.
INCLUSIVE_TWIN = {'<': '<=', '<=': '<', '>': '>=', '>=': '>'}
COMPARATOR_CASES = [dict(r, comparator=c) for r in REAL_STATES for c in (r['comparator'], INCLUSIVE_TWIN[r['comparator']])]
SIDE = {'<': 'low', '<=': 'low', '>': 'high', '>=': 'high'}
OPPOSITE = {'low': 'high', 'high': 'low'}
STEP = 0.001


def case_id(row):
    return '%s%s%s' % (row['rule_id'], row['comparator'], row['threshold'])


def gender_of(row):
    return 'F' if row['sex'] == 'any' else row['sex']


def inside(row):
    """Value just on the firing side of the threshold."""
    return row['threshold'] - STEP if SIDE[row['comparator']] == 'low' else row['threshold'] + STEP


def fires(row, value, level):
    return _row_holds(row, row['source_token'], row['unit'], value, gender_of(row), None, level, None,
                      'row-range', 'state')


def test_real_state_rows_cover_all_four_comparators():
    assert {c['comparator'] for c in COMPARATOR_CASES} == {'<', '<=', '>', '>='}
    assert {r['comparator'] for r in REAL_STATES} >= {'<', '>', '>='}


@pytest.mark.parametrize('row', COMPARATOR_CASES, ids=case_id)
def test_state_row_fires_on_its_own_side_level_and_not_on_the_opposite_level(row):
    side = SIDE[row['comparator']]
    value = inside(row)
    assert fires(row, value, side)
    assert not fires(row, value, OPPOSITE[side])
    assert not fires(row, value, 'normal')
    assert fires(row, value, None)  # no level computed -> the CSV comparison alone decides


@pytest.mark.parametrize('row', COMPARATOR_CASES, ids=case_id)
def test_state_row_boundary_follows_the_comparator(row):
    side = SIDE[row['comparator']]
    inclusive = row['comparator'] in ('<=', '>=')
    assert fires(row, row['threshold'], side) is inclusive
    outside = row['threshold'] + STEP if side == 'low' else row['threshold'] - STEP
    assert not fires(row, outside, side)


@pytest.mark.parametrize('row', COMPARATOR_CASES, ids=case_id)
def test_state_row_comparator_in_csv_mode_ignores_the_level(row):
    value = inside(row)
    for level in ('low', 'normal', 'high', None):
        assert _row_holds(row, row['source_token'], row['unit'], value, gender_of(row), None, level,
                          None, 'csv', 'state')


@pytest.mark.parametrize('row', [r for r in COMPARATOR_CASES if r['source_token'].startswith('lab:')],
                         ids=case_id)
def test_lab_state_row_end_to_end_with_the_real_threshold_file(row):
    # Own range puts the value on the wanted side (low: below own lower; high: above own upper).
    side, value = SIDE[row['comparator']], inside(row)
    own_low = (value + 2.0, value + 3.0) if side == 'low' else (value - 3.0, value - 2.0)
    own_opposite = (value - 3.0, value - 2.0) if side == 'low' else (value + 2.0, value + 3.0)
    store_row = [lab('x', row['source_token'], row['unit'], value, *own_low)]
    opposite_row = [lab('x', row['source_token'], row['unit'], value, *own_opposite)]
    thresholds = [r for r in REAL_THRESHOLDS if r['kind'] != 'state'] + [row]
    sex_store = type('SexStore', (FakeStore,), {'arrival': lambda self, stay: {'gender': gender_of(row)}})
    # A token without a level row on the opposite side can only come out 'normal' there.
    opposite_level = (OPPOSITE[side] if 'level:' + OPPOSITE[side] in
                      level_rows(thresholds, row['source_token'], row['unit'], gender_of(row))
                      else 'normal')
    cases = ((store_row, side, True), (opposite_row, opposite_level, False))
    if row['source_token'] in TEXTBOOK_LEVEL_TOKENS:  # v5c: own range ignored, both records alike
        cases = ((store_row, side, True), (opposite_row, side, True))
    for records, expected_level, expected_fire in cases:
        graph = build_graph(sex_store(records), SAMPLE, [], set(), thresholds=thresholds,
                            level_source='row-range')
        assert level_of(graph, row['source_token']) == expected_level
        assert (row['target_token'] in fired(graph)) is expected_fire


# --------------------- F3: the triage-vital node's level is wired into the state decision
TEMP, FEVER = 'vital:temperature', 'knowledge:state:fever'


class VitalStore(FakeStore):
    def __init__(self, value):
        super().__init__([])
        self.value = value

    def triage_vitals(self, stay):
        return [('temperature', self.value, 'degC', 'degF')]


def vital_thresholds(level_cut):
    # State row: '> 37.0' fever. Level row: high above `level_cut` (disagrees with the state when > 37.0).
    return [rule('lvl_temp_low', 'level', TEMP, 'degC', '<', 35.0, 'level:low'),
            rule('lvl_temp_high', 'level', TEMP, 'degC', '>', level_cut, 'level:high'),
            rule('state_fever', 'state', TEMP, 'degC', '>', 37.0, FEVER)]


def vital_build(value, level_cut, level_source):
    return build_graph(VitalStore(value), SAMPLE, [], set(), thresholds=vital_thresholds(level_cut),
                       level_source=level_source)


def test_triage_vital_state_is_blocked_by_a_normal_vital_level_in_row_range():
    graph = vital_build(38.0, 39.0, 'row-range')  # state '> 37.0' holds, level cut 39.0 -> normal
    assert level_of_vital(graph) == 'normal'
    assert FEVER not in fired(graph)


def test_triage_vital_state_fires_when_the_vital_level_agrees_in_row_range():
    graph = vital_build(38.0, 37.5, 'row-range')  # level cut 37.5 -> high
    assert level_of_vital(graph) == 'high'
    assert FEVER in fired(graph)


def test_triage_vital_state_ignores_the_vital_level_in_csv_mode():
    assert FEVER in fired(vital_build(38.0, 39.0, 'csv'))


def level_of_vital(graph):
    return next(n['level'] for n in graph['nodes'] if n['token'] == TEMP)
