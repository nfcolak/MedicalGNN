"""Clinical graph build (data.s4_graph.build pipeline) on a first_lab_delay artifact (v4b).

Thin entrypoint: same arguments as data.s4_graph.build; --inherit-from must be a
data.s2_events.first_lab_delay output. Its own manifest hashes (cohort, event index) are the
inherited contract instead of the v1 manifest, and the delay is recorded in decision_policy.
Without --execute it prints the plan only and creates nothing.
"""
import argparse
import json
from pathlib import Path

from data.s4_graph import build
from data.s4_graph.build import DIAGNOSIS_TOKEN_POLICIES
from data.s4_graph.graph import LEVEL_SOURCES
from data.s4_graph.store import AGE_POLICIES

POLICY = 'first_lab_plus_delay'


def check_delay_manifest(inherited, manifest):
    """Reject anything that is not a verified first_lab_delay artifact."""
    if manifest.get('cutoff_policy') != POLICY:
        raise ValueError('Inherited artifact is not a first_lab_plus_delay artifact')
    if type(manifest.get('delay_minutes')) is not int or manifest['delay_minutes'] < 1:
        raise ValueError('Inherited manifest has no valid delay_minutes')
    if manifest.get('temporal_clean') is not False or manifest.get('early_triage_eligible') is not False:
        raise ValueError('Inherited manifest must declare temporal_clean=false and early_triage_eligible=false')
    if not manifest.get('inherits_index_sha256') or manifest['inherits_index_sha256'] == manifest.get('event_index_sha256'):
        raise ValueError('Inherited manifest lacks a distinct inherits_index_sha256')
    verification = json.loads((Path(inherited) / 'verification.json').read_text())
    if verification.get('status') != 'verified' or verification.get('delay_minutes') != manifest['delay_minutes']:
        raise ValueError('Delay artifact verification missing or for another delay')


def policy_text(manifest):
    if build.is_v5_event_manifest(manifest) and manifest.get('decision_policy'):
        return manifest['decision_policy']  # v5 event artifact: its own cutoff text is the truth
    return ('Cutoff = min(first nonempty recorded lab result storetime + %d min, ED outtime), '
            'inherited from the first_lab_plus_delay artifact (inherits_index_sha256 %s); every '
            'stay shares the same fixed delay. Triage evidence is admitted under an unverified '
            'arrival-time assumption. Checking intime <= cutoff verifies only proxy bounds. Not an '
            'early-triage time.' % (manifest['delay_minutes'], manifest['inherits_index_sha256']))


def extra(manifest):
    return {'cutoff_policy': POLICY, 'delay_minutes': manifest['delay_minutes'],
            'inherits_index_sha256': manifest['inherits_index_sha256'],
            'cutoff_clipped_at_outtime': manifest.get('cutoff_clipped_at_outtime'),
            'early_triage_eligible': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inherit-from', type=Path, required=True,
                        help='first_lab_delay output (index_ready)')
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--canonical', type=Path, required=True)
    parser.add_argument('--knowledge', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--thresholds', type=Path, default=None,
                        help='Optional clinical level/state/combo rules CSV')
    parser.add_argument('--complaint-min-count', type=int, default=50)
    parser.add_argument('--with-diagnosis', action='store_true')
    parser.add_argument('--level-source', choices=LEVEL_SOURCES, default='csv',
                        help="csv (default): CSV limits. row-range: a lab result's own reference "
                             'range replaces the CSV limit where both bounds were recorded '
                             '(needs --thresholds and a v5 event index)')
    parser.add_argument('--diagnosis-policy', choices=DIAGNOSIS_TOKEN_POLICIES, default='v3',
                        help='v3 (default): full-code dx tokens, GEM quicksort mapping. v5: '
                             'dx:icd10:<3-char category> via the override-aware ICD mapping')
    parser.add_argument('--age-policy', choices=AGE_POLICIES, default='anchor',
                        help='anchor (default): patients.anchor_age. visit: anchor_age + '
                             'year(intime) - anchor_year, capped at 91')
    parser.add_argument('--max-nodes', type=int, default=20000)
    parser.add_argument('--max-edges', type=int, default=200000)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if min(args.complaint_min_count, args.max_nodes, args.max_edges) < 1:
        parser.error('Budgets must be positive')
    if args.level_source == 'row-range' and args.thresholds is None:
        parser.error('--level-source row-range needs --thresholds')
    if args.diagnosis_policy == 'v5' and not args.with_diagnosis:
        parser.error('--diagnosis-policy v5 needs --with-diagnosis')
    if args.output.exists():
        parser.error('Output exists; refusing to overwrite: ' + str(args.output))
    manifest_path = args.inherit_from / 'manifest.json'
    if args.execute:
        manifest = json.loads(manifest_path.read_text())
        check_delay_manifest(args.inherit_from, manifest)
        build.run(args, inherited_validator=check_delay_manifest,
                  decision_policy=policy_text(manifest), manifest_extra=extra(manifest))
        return
    plan = {'status': 'not_executed', 'requires': '--execute', 'pipeline': 'data.s4_graph.build.run',
            'inherit_from': str(args.inherit_from.resolve()), 'output': str(args.output.resolve()),
            'knowledge': str(args.knowledge), 'with_diagnosis': args.with_diagnosis, 'limit': args.limit,
            'checks_at_execute': ['cutoff_policy == first_lab_plus_delay', 'delay_minutes int >= 1',
                                  'temporal_clean false, early_triage_eligible false',
                                  'inherits_index_sha256 != event_index_sha256', 'delay verification.json verified',
                                  'events.sqlite and cohort.csv sha256 == the delay manifest hashes']}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        check_delay_manifest(args.inherit_from, manifest)
        plan.update(inherit_from_manifest='valid', delay_minutes=manifest['delay_minutes'],
                    inherits_index_sha256=manifest['inherits_index_sha256'],
                    decision_policy=policy_text(manifest))
    else:
        plan.update(inherit_from_manifest='absent: run data.s2_events.first_lab_delay --execute first')
    if (args.level_source, args.diagnosis_policy, args.age_policy) != ('csv', 'v3', 'anchor'):
        plan['options'] = {'level_source': args.level_source,
                           'diagnosis_policy': args.diagnosis_policy,
                           'age_policy': args.age_policy}
    if args.thresholds is not None:
        plan['thresholds'] = build.plan_thresholds(args.thresholds)
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
