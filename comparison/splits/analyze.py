"""Split-replication analysis: arms vs Full on one or more patient splits, then combined across splits.

    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m comparison.splits.analyze \
        --split r1=<root>/r1/top20 --split r2=<root>/r2/top20 \
        [--split r0=<ablation runs top20> --final r0=<final runs top20>] --out <new dir>

A split is `ID=DIR`. A new split's DIR holds `<arm>_seed<S>/` run dirs (result.json, binding.json, validation.npz,
test.npz, split_binding.json), Full = `a1_full_seed<S>` for every seed. `--final ID=DIR` marks the canonical layout
(the split `r0`, no sidecar): DIR holds the final Full runs `cei_mp2_seed<S>` for the final seeds, the other Full
seeds are read from the ablation runs DIR. Read-only on the runs; `--out` must not exist.

Per split: exactly the two-level paired bootstrap of comparison/ablation/arms_vs_ref.py (its functions are
imported, not copied; arm - Full on matched seeds, B=2000, same rng seeds), so a split alone reproduces the
ablation analysis to the digit. Decisions use VALIDATION only; test is reported, never used.
Combined over splits (mean of the per-split mean deltas): a stratified bootstrap (inside each split the seeds and
the patients are resampled as above, independently per split, then averaged over the splits) and a split-level
line (between-split SD of the per-split deltas, t interval over splits, df = n_splits - 1).
Groups: primary r1+r2 (splits not seen when the rule was fixed), secondary every split given (r0+r1+r2); each
split alone is always reported. The reading rule is applied to each group and to each split alone.

Refused: unfinished or missing runs (Full and every arm need every seed), a run that is not fair_v10t, a run
whose recipe is not its arm's (comparison.ablation.analyze.parity_errors), runs of one split that disagree on
graph, targets, v3 state, K, kept classes or sample ids, a split without a consistent split_binding.json, two
splits that differ in protocol, recipe or K, and two splits that are the same fold assignment. The kept
classes (top-K by TRAIN counts) may differ between splits; they are recorded and reported, not refused.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from comparison.ablation import arms_vs_ref as AR  # noqa: E402
from comparison.ablation.analyze import ABLATION_DEFAULTS, parity_errors  # noqa: E402
from comparison.ablation.run import ARMS, FINAL_SEEDS, SEEDS  # noqa: E402

REF = 'a1_full'
DEFAULT_ARMS = ('a5_nomp', 'a5b_nomp_matched', 'a6_pool')
PRIMARY = ('r1', 'r2')
SIDECAR = 'split_binding.json'
RUN_RE = re.compile(r'^(?P<arm>.+)_seed(?P<seed>\d+)$')
# binding fields that must agree between the reference runs of different splits (the recipe, not the data)
RECIPE_KEYS = ('method', 'protocol', 'protocol_overrides', 'top_k_labels', 'num_classes', 'epochs', 'patience',
               'weights', 'batch_size', 'hidden', 'edge_direction', 'selection_fold', 'sample_seed', 'train_limit',
               'lr', 'dropout')
SIDECAR_RUN_KEYS = {'seed', 'arm', 'run', 'run_dir', 'run_path', 'out', 'output', 'created', 'date', 'time', 'timestamp'}
SEED_COMBINED = 20261015
# sidecar fields (comparison/splits/run.py) that must be present and equal across the runs of a split
SIDECAR_REQUIRED = ('split_file_sha256', 'graph_manifest_sha256', 'targets_sha256', 'v3_state_sha256')
MATTERS = 0.005
Refusal = AR.Refusal


def finished(d):
    return d.is_dir() and (d / 'result.json').is_file()


def find_runs(runs, final, arms, seeds, canonical):
    """{arm: {seed: dir}} for Full + arms; raises Refusal listing every missing run."""
    runs = Path(runs)
    out = {name: {} for name in (REF,) + tuple(arms)}
    missing = []
    for name in out:
        for s in seeds:
            d = runs / f'{name}_seed{s}'
            if canonical and name == REF and s in FINAL_SEEDS:
                d = Path(final) / f'cei_mp2_seed{s}'
            if finished(d):
                out[name][s] = d
            else:
                missing.append(str(d))
    if missing:
        raise Refusal(f'{len(missing)} expected run(s) missing or unfinished (no result.json):\n  ' + '\n  '.join(missing))
    return out


def read_sidecars(split_id, dirs):
    """The split_binding.json of every run of a new split; all must agree (run-specific keys ignored)."""
    docs = {}
    for d in dirs:
        p = Path(d) / SIDECAR
        if not p.is_file():
            raise Refusal(f'{d}: no {SIDECAR} (a new split needs the sidecar written by comparison.splits.run)')
        doc = json.loads(p.read_text())
        if not isinstance(doc, dict) or not doc:
            raise Refusal(f'{p}: not a non-empty JSON object')
        if doc.get('inputs_unchanged_during_run') is not True:
            raise Refusal(f"{p}: inputs_unchanged_during_run is {doc.get('inputs_unchanged_during_run')!r}, not true")
        docs[d] = doc
    first_dir = next(iter(docs))
    base = {k: v for k, v in docs[first_dir].items() if k not in SIDECAR_RUN_KEYS}
    if base.get('split_id') != split_id:
        raise Refusal(f"{first_dir / SIDECAR}: split_id {base.get('split_id')!r}, expected {split_id!r}")
    absent = [k for k in SIDECAR_REQUIRED if not base.get(k)]
    if absent:
        raise Refusal(f'{first_dir / SIDECAR}: missing {absent}')
    for d, doc in docs.items():
        other = {k: v for k, v in doc.items() if k not in SIDECAR_RUN_KEYS}
        if other != base:
            diff = sorted(k for k in set(base) | set(other) if base.get(k) != other.get(k))
            raise Refusal(f'{d / SIDECAR} disagrees with {first_dir / SIDECAR} on {diff}')
    return base


def check_sidecar_vs_binding(sidecar, binding, path):
    """Hashes the sidecar shares by name with the run binding must agree with it."""
    config = binding.get('method_config') or {}
    for key, got in (('targets_sha256', binding.get('targets_sha256')),
                     ('v3_state_sha256', config.get('v3_state_sha256'))):
        if key in sidecar and got is not None and sidecar[key] != got:
            raise Refusal(f'{path}: sidecar {key} {sidecar[key]!r} != binding {got!r}')


def load_split(split_id, runs, final, arms, seeds, k):
    canonical = final is not None
    dirs = find_runs(runs, final, arms, seeds, canonical)
    loaded = {}
    for name, by_seed in dirs.items():
        for s, d in by_seed.items():
            loaded[(name, s)] = AR.load_run(d, name, s)
    keys = list(loaded)
    base = loaded[keys[0]]['binding']
    for key in keys:
        for same in AR.SAME_KEYS:
            if loaded[key]['binding'].get(same) != base.get(same):
                raise Refusal(f'{split_id}: {same} differs between {loaded[keys[0]]["row"]["path"]} '
                              f'and {loaded[key]["row"]["path"]}')
    if base.get('top_k_labels') != k:
        raise Refusal(f"{split_id}: top_k_labels {base.get('top_k_labels')!r}, expected {k}")
    ref_seed = min(dirs[REF], key=lambda s: (s not in FINAL_SEEDS, s))
    errors = parity_errors(dirs[REF][ref_seed], [(name, d) for name, by_seed in dirs.items() for d in by_seed.values()])
    if errors:
        raise Refusal(f'{split_id}: {len(errors)} run(s) differ from their arm recipe:\n  ' + '\n  '.join(errors))
    sidecar = None
    if canonical:
        pass   # the canonical layout has no sidecar; --final marks it (the canonical split is r0)
    else:
        run_dirs = [d for by_seed in dirs.values() for d in by_seed.values()]
        sidecar = read_sidecars(split_id, run_dirs)
        for key in keys:
            check_sidecar_vs_binding(sidecar, loaded[key]['binding'], loaded[key]['row']['path'])
    return {'id': split_id, 'canonical': canonical, 'dirs': dirs, 'loaded': loaded, 'binding': base,
            'ref_binding': loaded[(REF, ref_seed)]['binding'], 'sidecar': sidecar, 'seeds': list(seeds)}


def split_boot(sp, arms):
    """Per fold: bootstrap F1 of every run on shared patient resamples (one call, as arms_vs_ref.analyze)."""
    boots, info = {}, {}
    for fold in ('validation', 'test'):
        have = {k: v for k, v in sp['loaded'].items() if fold in v['folds']}
        if len(have) != len(sp['loaded']):
            if fold == 'validation':
                raise Refusal(f"{sp['id']}: a run lacks validation.npz")
            continue
        boots[fold], info[fold] = AR.bootstrap_f1(have, fold)
    # delta[fold][arm]: array [n_seeds, B] of F1(arm) - F1(Full) per matched seed and resample
    delta = {fold: {arm: np.stack([boots[fold][(arm, s)] - boots[fold][(REF, s)] for s in sp['seeds']])
                    for arm in arms} for fold in boots}
    return delta, info


def per_split_result(sp, arms, delta, info):
    """Same numbers as arms_vs_ref.analyze for this split (macro-F1 and accuracy deltas, two-level bootstrap)."""
    out = {'seeds': sp['seeds'], 'canonical': sp['canonical'], 'folds': info,
           'kept_label_indices': sp['binding'].get('kept_label_indices'),
           'reference': {'validation_macro_f1': AR.mean_sd([sp['loaded'][(REF, s)]['row']['val_macro_f1'] for s in sp['seeds']]),
                         'test_macro_f1': AR.mean_sd([sp['loaded'][(REF, s)]['row']['test_macro_f1'] for s in sp['seeds']])
                         if all('test_macro_f1' in sp['loaded'][(REF, s)]['row'] for s in sp['seeds']) else None},
           'arms': {}}
    for arm in arms:
        res = {}
        for fold in delta:
            pre = 'val' if fold == 'validation' else 'test'
            rows = []
            for s in sp['seeds']:
                a, r = sp['loaded'][(arm, s)]['row'], sp['loaded'][(REF, s)]['row']
                rows.append({'seed': s, 'macro_f1_delta': a[f'{pre}_macro_f1'] - r[f'{pre}_macro_f1'],
                             'accuracy_delta': a[f'{pre}_accuracy'] - r[f'{pre}_accuracy']})
            res[fold] = {'per_seed': rows, 'macro_f1_delta': AR.mean_sd([x['macro_f1_delta'] for x in rows]),
                         'accuracy_delta': AR.mean_sd([x['accuracy_delta'] for x in rows]),
                         'bootstrap': AR.two_level(delta[fold][arm])}
            if fold == 'test':
                res[fold]['note'] = AR.TEST_NOTE
        out['arms'][arm] = res
    return out


def t_interval(means):
    n = len(means)
    if n < 2:
        return None
    from scipy import stats
    sd = float(np.std(means, ddof=1))
    half = float(stats.t.ppf(0.975, n - 1)) * sd / np.sqrt(n)
    mean = float(np.mean(means))
    return {'ci95': [mean - half, mean + half], 'df': n - 1}


def combine(ids, per_split, deltas, arm, fold):
    """Combined estimate of one arm over the splits `ids` (mean of per-split mean deltas)."""
    means = [per_split[i]['arms'][arm][fold]['macro_f1_delta']['mean'] for i in ids]
    acc = [per_split[i]['arms'][arm][fold]['accuracy_delta']['mean'] for i in ids]
    strat, strat_p = np.zeros(AR.B), np.zeros(AR.B)
    for j, i in enumerate(ids):
        d = deltas[i][fold][arm]
        n = d.shape[0]
        # independent seed picks per split; patient replicates are decoupled across splits by a fixed permutation
        pick = np.random.default_rng([SEED_COMBINED, j]).integers(0, n, size=(AR.B, n))
        cols = np.random.default_rng([SEED_COMBINED, j, 1]).permutation(AR.B)
        dj = d[:, cols]
        strat += dj[pick, np.arange(AR.B)[:, None]].mean(axis=1)
        strat_p += dj.mean(axis=0)
    strat, strat_p = strat / len(ids), strat_p / len(ids)
    out = {'splits': list(ids), 'per_split_mean_delta': dict(zip(ids, means)),
           'macro_f1_delta_mean': float(np.mean(means)), 'accuracy_delta_mean': float(np.mean(acc)),
           'between_split_sd': float(np.std(means, ddof=1)) if len(means) > 1 else None,
           'split_t_interval': t_interval(means), 'bootstrap': {}}
    for name, d in (('seeds_and_patients', strat), ('patients_only', strat_p)):
        lo, hi = np.percentile(d, [2.5, 97.5])
        out['bootstrap'][name] = {'ci95': [float(lo), float(hi)], 'p_delta_gt_0': float((d > 0).mean())}
    return out


def reading(mean_delta, ci):
    """PREREGISTRATION.md rule: drop = Full - arm, CI = seeds+patients 95% interval of the delta.
    Returns (reading, note); the note flags a drop >= MATTERS whose CI still includes 0."""
    drop = -mean_delta
    excludes0 = ci[0] > 0 or ci[1] < 0
    if drop >= MATTERS and excludes0:
        return 'matters', ''
    if not excludes0:
        return 'no measurable effect', 'not resolved at this precision' if drop >= MATTERS else ''
    return 'small effect, not resolved', ''


def groups_for(ids):
    out = {}
    primary = [i for i in PRIMARY if i in ids]
    if len(primary) == len(PRIMARY):
        out['primary r1+r2'] = primary
    if list(ids) != primary:
        out['secondary ' + '+'.join(ids)] = list(ids)
    return out


def full_mp2(binding):
    """mp2_options of a Full binding with absent ablation options filled by their Full default."""
    mp2 = dict((binding.get('method_config') or {}).get('mp2_options') or {})
    for key, value in ABLATION_DEFAULTS.items():
        if key not in ('arm', 'encoder_depth'):
            mp2.setdefault(key, value)
    return mp2


def analyze(splits, arms, seeds, k, allow_duplicate_splits):
    loaded = [load_split(i, runs, final, arms, seeds, k) for i, runs, final in splits]
    ids = [sp['id'] for sp in loaded]
    # one recipe across splits
    base = loaded[0]['ref_binding']
    for sp in loaded[1:]:
        for key in RECIPE_KEYS:
            if sp['ref_binding'].get(key) != base.get(key):
                raise Refusal(f"{key} differs between split {ids[0]} ({base.get(key)!r}) and {sp['id']} "
                              f"({sp['ref_binding'].get(key)!r})")
        # the final Full runs (r0) predate the ablation options and omit them; an absent option means its Full default
        if full_mp2(sp['ref_binding']) != full_mp2(base):
            raise Refusal(f"mp2_options of the Full runs differ between split {ids[0]} and {sp['id']}")
    hashes = [json.dumps(sp['ref_binding'].get('split_sample_ids_sha256'), sort_keys=True) for sp in loaded]
    if len(set(hashes)) != len(hashes) and not allow_duplicate_splits:
        raise Refusal('two splits carry the same sample ids (the same fold assignment); '
                      '--allow-duplicate-splits is for the self-check only')
    per_split, deltas, notes = {}, {}, []
    for sp in loaded:
        deltas[sp['id']], info = split_boot(sp, arms)
        per_split[sp['id']] = per_split_result(sp, arms, deltas[sp['id']], info)
    kept = {i: per_split[i]['kept_label_indices'] for i in ids}
    if len({json.dumps(v) for v in kept.values()}) > 1:
        notes.append('the kept class set (top-K by TRAIN counts) differs between splits: '
                     + '; '.join(f'{i}: {v}' for i, v in kept.items()))
    if allow_duplicate_splits:
        notes.append('--allow-duplicate-splits: duplicated split input, self-check only, not a result')
    groups = groups_for(ids)
    combined = {}
    for gname, gids in groups.items():
        combined[gname] = {arm: {fold: combine(gids, per_split, deltas, arm, fold) for fold in deltas[gids[0]]}
                           for arm in arms}
    for g in combined.values():
        for arm, folds in g.items():
            for fold, c in folds.items():
                if fold == 'validation':
                    c['reading'], c['reading_note'] = reading(c['macro_f1_delta_mean'], c['bootstrap']['seeds_and_patients']['ci95'])
    for i in ids:
        for arm in arms:
            v = per_split[i]['arms'][arm]['validation']
            v['reading'], v['reading_note'] = reading(v['macro_f1_delta']['mean'], v['bootstrap']['seeds_and_patients']['ci95'])
    return {'protocol': AR.PROTOCOL, 'reference': REF, 'arms': list(arms), 'seeds': list(seeds), 'top_k_labels': k,
            'splits': ids, 'canonical_split': [sp['id'] for sp in loaded if sp['canonical']] or None,
            'bootstrap': {'B': AR.B, 'rng_patients': AR.SEED_PATIENTS, 'rng_seeds': AR.SEED_SEEDS,
                          'rng_combined': SEED_COMBINED,
                          'per_split': 'arms_vs_ref two-level: matched seeds, then patients (rows paired by sample_ids)',
                          'combined': 'per split independent seed picks and decoupled patient resamples, then the '
                                      'mean over splits; 95% percentile',
                          'split_level': 'SD of the per-split mean deltas (ddof=1) and t interval, df = n_splits - 1'},
            'reading_rule': f'matters: drop (Full - arm) >= {MATTERS} and CI excludes 0; no measurable effect: CI '
                            f'includes 0 (note when drop >= {MATTERS}); small effect, not resolved: CI excludes 0 and '
                            f'drop < {MATTERS} (PREREGISTRATION.md)',
            'test_note': AR.TEST_NOTE, 'notes': notes, 'per_split': per_split, 'combined': combined}


def ci_txt(c):
    return f"[{c[0]:+.4f}, {c[1]:+.4f}]"


def render(doc):
    L = ['# Split replication: arms vs Full', '',
         f"Protocol {doc['protocol']}, K={doc['top_k_labels']}, reference `{doc['reference']}`, arms "
         f"{', '.join(doc['arms'])}, seeds {','.join(map(str, doc['seeds']))}, splits {', '.join(doc['splits'])}"
         + (f" (canonical layout, no sidecar: {', '.join(doc['canonical_split'])})" if doc['canonical_split'] else '')
         + f". Decisions use VALIDATION; test {doc['test_note']}.",
         f"Bootstrap B={doc['bootstrap']['B']}. Per split: {doc['bootstrap']['per_split']}. Combined: "
         f"{doc['bootstrap']['combined']}. Split level: {doc['bootstrap']['split_level']}.",
         f"Reading rule: {doc['reading_rule']}.", '']
    for n in doc['notes']:
        L.append(f'NOTE: {n}')
    L += ['', '## Kept classes per split (top-K by TRAIN counts)', '']
    for i, s in doc['per_split'].items():
        L.append(f"- {i}: {s['kept_label_indices']}")
    L += ['', '## Per split (delta = arm - Full, matched seeds)', '',
          '| split | arm | d val macro-F1 [CI seeds+patients] | P(d>0) | CI patients only | reading | note | '
          'd test [CI] (reported only) |', '|---|---|---|---|---|---|---|---|']
    for i, s in doc['per_split'].items():
        for arm, r in s['arms'].items():
            v, t = r['validation'], r.get('test')
            b = v['bootstrap']
            L.append(f"| {i} | {arm} | {v['macro_f1_delta']['mean']:+.4f} {ci_txt(b['seeds_and_patients']['ci95'])} | "
                     f"{b['seeds_and_patients']['p_delta_gt_0']:.3f} | {ci_txt(b['patients_only']['ci95'])} | {v['reading']} | {v['reading_note']} | "
                     + (f"{t['macro_f1_delta']['mean']:+.4f} {ci_txt(t['bootstrap']['seeds_and_patients']['ci95'])} |" if t else 'n/a |'))
    for gname, g in doc['combined'].items():
        L += ['', f'## Combined: {gname}', '',
              '| arm | mean d val | CI seeds+patients (stratified) | P(d>0) | CI patients only | between-split SD | '
              't interval over splits | reading | note | mean d test (reported only) |', '|---|---|---|---|---|---|---|---|---|---|']
        for arm, folds in g.items():
            c, t = folds['validation'], folds.get('test')
            b = c['bootstrap']
            sd = f"{c['between_split_sd']:.4f}" if c['between_split_sd'] is not None else 'n/a'
            ti = ci_txt(c['split_t_interval']['ci95']) + f" (df {c['split_t_interval']['df']})" if c['split_t_interval'] else 'n/a'
            L.append(f"| {arm} | {c['macro_f1_delta_mean']:+.4f} | {ci_txt(b['seeds_and_patients']['ci95'])} | "
                     f"{b['seeds_and_patients']['p_delta_gt_0']:.3f} | {ci_txt(b['patients_only']['ci95'])} | {sd} | {ti} | "
                     f"{c['reading']} | {c['reading_note']} | " + (f"{t['macro_f1_delta_mean']:+.4f}" if t else 'n/a') + ' |')
        L += ['', 'Per-split mean d val: ' + '; '.join(
            f"{arm}: " + ', '.join(f'{i} {x:+.4f}' for i, x in folds['validation']['per_split_mean_delta'].items())
            for arm, folds in g.items())]
    L.append('')
    return '\n'.join(L)


def parse_named(items, what):
    out = {}
    for it in items:
        name, sep, path = it.partition('=')
        if not sep or not name or not path:
            raise Refusal(f'{what} must be ID=DIR, got {it!r}')
        if name in out:
            raise Refusal(f'{what}: split id {name!r} given twice')
        out[name] = path
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--split', action='append', default=[], required=True, help='ID=DIR of the runs of one split (repeatable)')
    ap.add_argument('--final', action='append', default=[], help='ID=DIR of the final Full runs (canonical layout, split r0)')
    ap.add_argument('--arms', default=','.join(DEFAULT_ARMS), help='comma-separated arms compared with a1_full')
    ap.add_argument('--seeds', default=','.join(map(str, SEEDS)), help='seeds every arm and Full must have')
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--allow-duplicate-splits', action='store_true', help='self-check only: accept two identical splits')
    ap.add_argument('--out', required=True, help='new output directory (must not exist)')
    a = ap.parse_args(argv)
    try:
        out = Path(a.out)
        if out.exists():
            raise Refusal(f'--out already exists: {out}')
        splits, finals = parse_named(a.split, '--split'), parse_named(a.final, '--final')
        if unknown := sorted(set(finals) - set(splits)):
            raise Refusal(f'--final for unknown split(s) {unknown}')
        arms = tuple(x for x in a.arms.split(',') if x)
        if bad := sorted(set(arms) - set(ARMS)) or ([REF] if REF in arms else []):
            raise Refusal(f'unknown arm(s) or the reference in --arms: {bad}')
        seeds = tuple(sorted(int(s) for s in a.seeds.split(',') if s))   # sorted: the seed level of two_level is order-dependent
        doc = analyze([(i, p, finals.get(i)) for i, p in splits.items()], arms, seeds, a.k, a.allow_duplicate_splits)
    except Refusal as e:
        print(f'REFUSED: {e}', file=sys.stderr)
        return 2
    out.mkdir(parents=True)
    (out / 'summary.json').write_text(json.dumps(doc, indent=1))
    (out / 'summary.md').write_text(render(doc) + '\n')
    print(f'wrote {out}/summary.md and summary.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
