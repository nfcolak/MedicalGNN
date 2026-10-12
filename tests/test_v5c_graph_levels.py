"""T1 (SC1, SC2, SC3 graph part): K/Na textbook levels under row-range, equal-bound own
ranges, threshold limitation lines, threshold_semantics().

Synthetic store and threshold rows only: no patient data, no event index, no test fold.
(R1-C reads the real rule file knowledge_thresholds_v2.csv, which is rule data.)
Reuses the FakeStore pattern of test_v5_threshold_states.py.
"""
from pathlib import Path

import pytest

from data.s4_graph import graph as graph_module
from data.s4_graph.graph import _own_range, _row_holds, level_for
from test_v5_threshold_states import (ANION_GAP, DKA, GLUCOSE, SAMPLE, FakeStore, build, dka, fired,
                                      lab, level_of, rule)
from data.s4_graph.graph import build_graph

K_SERUM, K_GAS, NA_SERUM, NA_GAS = 'lab:50971', 'lab:50822', 'lab:50983', 'lab:50824'
CREATININE = 'lab:50912'
TEXTBOOK = [K_SERUM, K_GAS, NA_SERUM, NA_GAS]
HYPO, HYPER = 'knowledge:state:hypokalemia', 'knowledge:state:hyperkalemia'
HYPONA, HYPERNA = 'knowledge:state:hyponatremia', 'knowledge:state:hypernatremia'
CREAT_HIGH = 'knowledge:state:creatinine_high'


def textbook_rules():
    rows = []
    for token in (K_SERUM, K_GAS):
        rows += [rule('lvl_%s_low' % token, 'level', token, 'mEq/L', '<', 3.5, 'level:low'),
                 rule('lvl_%s_high' % token, 'level', token, 'mEq/L', '>', 5.0, 'level:high'),
                 rule('st_%s_hypo' % token, 'state', token, 'mEq/L', '<', 3.5, HYPO),
                 rule('st_%s_hyper' % token, 'state', token, 'mEq/L', '>', 5.5, HYPER)]
    for token in (NA_SERUM, NA_GAS):
        rows += [rule('lvl_%s_low' % token, 'level', token, 'mEq/L', '<', 135, 'level:low'),
                 rule('lvl_%s_high' % token, 'level', token, 'mEq/L', '>', 145, 'level:high'),
                 rule('st_%s_hypo' % token, 'state', token, 'mEq/L', '<', 135, HYPONA),
                 rule('st_%s_hyper' % token, 'state', token, 'mEq/L', '>', 145, HYPERNA)]
    rows += [rule('lvl_cr_low', 'level', CREATININE, 'mg/dL', '<', 0.5, 'level:low'),
             rule('lvl_cr_high', 'level', CREATININE, 'mg/dL', '>', 1.2, 'level:high'),
             rule('st_cr_high', 'state', CREATININE, 'mg/dL', '>', 1.2, CREAT_HIGH),
             rule('lvl_ag_low', 'level', ANION_GAP, 'mEq/L', '<', 10, 'level:low'),
             rule('lvl_ag_high', 'level', ANION_GAP, 'mEq/L', '>', 18, 'level:high'),
             rule('combo_dka', 'combo', GLUCOSE, 'mg/dL', '>', 250, DKA),
             rule('combo_dka', 'combo', ANION_GAP, 'mEq/L', '>', 12, DKA)]
    return rows


RULES = textbook_rules()


def level(token, unit, value, lower, upper, source='row-range', censored=None):
    return level_for(RULES, token, unit, value, 'F', censored=censored, ref_lower=lower,
                     ref_upper=upper, level_source=source)


def build_rules(records, source):
    return build(records, source, thresholds=RULES)


# ------------------------------------------------------------------ SC1: level_for
@pytest.mark.parametrize('token', [K_SERUM, K_GAS])
def test_potassium_3_4_with_own_range_3_3_to_5_1_is_low_under_row_range(token):
    assert level(token, 'mEq/L', 3.4, 3.3, 5.1) == 'low'


@pytest.mark.parametrize('token', [K_SERUM, K_GAS])
def test_potassium_3_2_is_low_and_5_2_with_own_upper_5_5_is_high(token):
    assert level(token, 'mEq/L', 3.2, 3.3, 5.1) == 'low'
    assert level(token, 'mEq/L', 5.2, 3.0, 5.5) == 'high'  # own range would say normal


@pytest.mark.parametrize('token', [NA_SERUM, NA_GAS])
def test_sodium_134_with_own_range_133_to_145_is_low_under_row_range(token):
    assert level(token, 'mEq/L', 134, 133, 145) == 'low'


@pytest.mark.parametrize('token', [NA_SERUM, NA_GAS])
def test_sodium_146_with_own_range_135_to_147_is_high_under_row_range(token):
    assert level(token, 'mEq/L', 146, 135, 147) == 'high'


@pytest.mark.parametrize('token', [NA_SERUM, NA_GAS])
def test_sodium_133_is_low_and_a_textbook_normal_value_stays_normal(token):
    assert level(token, 'mEq/L', 133, 133, 145) == 'low'
    # 136 is normal by the textbook limits although it lies below this record's own lower 137.
    assert level(token, 'mEq/L', 136, 137, 145) == 'normal'


@pytest.mark.parametrize('token', TEXTBOOK)
def test_textbook_tokens_are_unchanged_in_csv_mode(token):
    value = 3.4 if token in (K_SERUM, K_GAS) else 134
    own = (3.3, 5.1) if token in (K_SERUM, K_GAS) else (133, 145)
    assert level(token, 'mEq/L', value, *own, source='csv') == 'low'
    assert level(token, 'mEq/L', 4.0 if token in (K_SERUM, K_GAS) else 140, 4.1, 4.2, source='csv') == 'normal'


def test_non_textbook_token_still_uses_its_own_range_under_row_range():
    # Creatinine 1.1: CSV high is > 1.2 -> normal; own range 0.5-1.0 -> high.
    assert level(CREATININE, 'mg/dL', 1.1, 0.5, 1.0) == 'high'
    assert level(CREATININE, 'mg/dL', 1.1, 0.5, 1.0, source='csv') == 'normal'


def test_censored_textbook_potassium_uses_the_csv_limits():
    assert level(K_SERUM, 'mEq/L', 3.4, 3.3, 5.1, censored='<') == 'low'  # '<3.4' is certainly < 3.5
    assert level(K_SERUM, 'mEq/L', 3.6, 3.7, 5.1, censored='<') == 'normal'  # not certainly < 3.5


def test_textbook_level_tokens_constant():
    assert graph_module.TEXTBOOK_LEVEL_TOKENS == {'lab:50971', 'lab:50983', 'lab:50822', 'lab:50824'}


# ----------------------------------------------------------- SC1: _row_holds / end to end
def test_row_holds_state_row_still_needs_level_agreement_for_textbook_tokens():
    row = next(r for r in RULES if r['rule_id'] == 'st_%s_hypo' % K_SERUM)
    args = (row, K_SERUM, 'mEq/L', 3.4, 'F', None)
    assert _row_holds(*args, 'low', None, 'row-range', 'state', 3.3)
    assert not _row_holds(*args, 'normal', None, 'row-range', 'state', 3.3)
    assert not _row_holds(*args, 'high', None, 'row-range', 'state', 3.3)


def test_row_holds_csv_mode_ignores_the_level():
    row = next(r for r in RULES if r['rule_id'] == 'st_%s_hypo' % K_SERUM)
    assert _row_holds(row, K_SERUM, 'mEq/L', 3.4, 'F', None, 'normal', None, 'csv', 'state')


def test_state_rule_stricter_than_the_level_row_does_not_fire_on_a_normal_level():
    # State '< 3.8' vs level '< 3.5': K 3.6 is textbook normal, so the state must not fire.
    rows = [r for r in RULES if r['source_token'] != K_SERUM] + [
        rule('lvl_k_low', 'level', K_SERUM, 'mEq/L', '<', 3.5, 'level:low'),
        rule('st_k_wide', 'state', K_SERUM, 'mEq/L', '<', 3.8, HYPO)]
    graph = build(potassium_records(K_SERUM, 3.6, 3.3, 5.1), 'row-range', thresholds=rows)
    assert level_of(graph, K_SERUM) == 'normal' and HYPO not in fired(graph)


def potassium_records(token, value, lower, upper):
    return [lab('k', token, 'mEq/L', value, lower, upper)]


@pytest.mark.parametrize('token', [K_SERUM, K_GAS])
def test_graph_potassium_3_4_is_low_and_fires_hypokalemia(token):
    graph = build_rules(potassium_records(token, 3.4, 3.3, 5.1), 'row-range')
    assert level_of(graph, token) == 'low'
    assert HYPO in fired(graph)


@pytest.mark.parametrize('token', [K_SERUM, K_GAS])
def test_graph_potassium_3_6_with_own_lower_3_7_is_normal_and_fires_nothing(token):
    graph = build_rules(potassium_records(token, 3.6, 3.7, 5.1), 'row-range')
    assert level_of(graph, token) == 'normal'
    assert fired(graph).isdisjoint({HYPO, HYPER})


@pytest.mark.parametrize('token', [NA_SERUM, NA_GAS])
def test_graph_sodium_134_is_low_and_fires_hyponatremia(token):
    graph = build_rules([lab('na', token, 'mEq/L', 134, 133, 145)], 'row-range')
    assert level_of(graph, token) == 'low'
    assert HYPONA in fired(graph)


@pytest.mark.parametrize('token', [NA_SERUM, NA_GAS])
def test_graph_sodium_146_is_high_and_fires_hypernatremia(token):
    graph = build_rules([lab('na', token, 'mEq/L', 146, 135, 147)], 'row-range')
    assert level_of(graph, token) == 'high'
    assert HYPERNA in fired(graph)


def test_graph_creatinine_state_still_follows_the_own_range_level():
    # 1.3 passes the CSV state row '> 1.2' but is normal for own range 0.5-1.5 -> the gate blocks it.
    graph = build_rules([lab('cr', CREATININE, 'mg/dL', 1.3, 0.5, 1.5)], 'row-range')
    assert level_of(graph, CREATININE) == 'normal' and CREAT_HIGH not in fired(graph)
    assert CREAT_HIGH in fired(build_rules([lab('cr', CREATININE, 'mg/dL', 1.3, 0.5, 1.5)], 'csv'))


def test_graph_csv_mode_potassium_unchanged():
    graph = build_rules(potassium_records(K_SERUM, 3.4, 3.3, 5.1), 'csv')
    assert level_of(graph, K_SERUM) == 'low' and HYPO in fired(graph)


# ------------------------------------------------------------------ SC2: equal bounds
def test_own_range_rejects_equal_bounds_including_zero_zero():
    assert _own_range(5.0, 5.0) is None
    assert _own_range(0.0, 0.0) is None
    assert _own_range(0, 0) is None
    assert _own_range(3.0, 5.0) == (3.0, 5.0)
    assert _own_range(None, 5.0) is None and _own_range(5.0, 3.0) is None


@pytest.mark.parametrize('bounds', [(5.0, 5.0), (0.0, 0.0)])
def test_equal_bounds_fall_back_to_the_csv_level(bounds):
    # Creatinine 1.0 is normal by the CSV (0.5-1.2); "own range 5-5" / "0-0" must not be used.
    assert level(CREATININE, 'mg/dL', 1.0, *bounds) == 'normal'
    assert level(CREATININE, 'mg/dL', 1.5, *bounds) == 'high'
    assert level(CREATININE, 'mg/dL', 0.4, *bounds) == 'low'


@pytest.mark.parametrize('bounds', [(10.0, 10.0), (0.0, 0.0)])
def test_anion_gap_with_equal_bounds_uses_the_csv_combo_threshold(bounds):
    # CSV '> 12': 11 must not fire (own upper 10 would), 13 must fire.
    assert DKA not in fired(build_rules(dka(11.0, *bounds), 'row-range'))
    assert DKA in fired(build_rules(dka(13.0, *bounds), 'row-range'))


def test_anion_gap_with_a_valid_range_still_uses_the_own_upper():
    assert DKA in fired(build_rules(dka(11.0, 3.0, 10.0), 'row-range'))
    assert DKA not in fired(build_rules(dka(15.0, 8.0, 20.0), 'row-range'))


# ------------------------------------------------------------- SC3: limitation lines
GAP_LINE = "DKA anion gap = above the lab's own upper limit (fallback > 12 when no valid own range)"


def limitations(source, thresholds=RULES):
    records = potassium_records(K_SERUM, 3.4, 3.3, 5.1)
    return build_graph(FakeStore(records), SAMPLE, [], set(), thresholds=thresholds,
                       level_source=source)['limitations']


def test_row_range_graph_carries_the_three_threshold_limitation_lines():
    lines = limitations('row-range')
    state = [l for l in lines if 'State rows fire only when the node level agrees' in l]
    textbook = [l for l in lines if 'textbook' in l and all(t in l for t in TEXTBOOK)]
    gap = [l for l in lines if GAP_LINE in l]
    assert len(state) == 1 and len(textbook) == 1 and len(gap) == 1
    assert 'potassium' in textbook[0] and 'sodium' in textbook[0]


def test_csv_mode_limitation_list_is_the_old_list():
    csv_lines, row_range_lines = limitations('csv'), limitations('row-range')
    assert len(csv_lines) == 8
    assert not any('textbook' in l or 'anion gap' in l or 'node level agrees' in l for l in csv_lines)
    assert row_range_lines[:8] == csv_lines  # the old lines come first, unchanged
    assert len(row_range_lines) > 9


def test_graph_without_thresholds_has_no_threshold_limitation_lines():
    lines = build_graph(FakeStore([]), SAMPLE, [], set(), thresholds=None)['limitations']
    assert len(lines) == 7


# --------------------------------------------------------- threshold_semantics()
def test_threshold_semantics_content():
    semantics = graph_module.threshold_semantics()
    assert semantics == {
        'state_gate': 'level_agreement',
        'textbook_level_tokens': ['lab:50822', 'lab:50824', 'lab:50971', 'lab:50983'],
        'own_upper_tokens': ['lab:50868'],
        'own_range_rule': 'both bounds present and lower < upper'}


def test_threshold_semantics_is_json_able_and_stable():
    import json
    first = json.dumps(graph_module.threshold_semantics(), sort_keys=True)
    assert first == json.dumps(graph_module.threshold_semantics(), sort_keys=True)
    assert json.loads(first)['textbook_level_tokens'] == sorted(graph_module.TEXTBOOK_LEVEL_TOKENS)


# --------------------------------------------------------- R1-C: real rule file, state vs level (K/Na)
def _real_rules(kind):
    import csv
    path = Path(__file__).resolve().parents[1] / 'data/s2_events/knowledge_thresholds_v2.csv'
    with path.open(newline='') as handle:
        return [r for r in csv.DictReader(handle) if r['kind'] == kind and r['sex'] == 'any']


@pytest.mark.parametrize('token', sorted(graph_module.TEXTBOOK_LEVEL_TOKENS))
@pytest.mark.parametrize('side', ['low', 'high'])
def test_real_csv_state_threshold_is_at_or_beyond_the_level_threshold_for_textbook_tokens(token, side):
    """A K/Na state must imply its node level, or the level_agreement gate would silently drop it again."""
    low = side == 'low'
    strict, loose = ('<', '<=') if low else ('>', '>=')
    levels = [r for r in _real_rules('level') if r['source_token'] == token
              and r['comparator'] in (strict, loose) and r['target_token'] == 'level:' + side]
    states = [r for r in _real_rules('state') if r['source_token'] == token and r['comparator'] in (strict, loose)
              and ('hypo' if low else 'hyper') in r['target_token']]
    assert len(levels) == 1 and states, (token, side)  # non-vacuous: the rows exist
    level_thr = float(levels[0]['threshold'])
    for state in states:
        state_thr = float(state['threshold'])
        assert (state_thr <= level_thr) if low else (state_thr >= level_thr), (token, side, state['rule_id'])
        if state_thr == level_thr:  # same cut-off: the state comparator must not be the looser one
            assert state['comparator'] == strict or levels[0]['comparator'] == loose, (token, side)
