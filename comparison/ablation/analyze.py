"""Validation analysis of the ablation runs: every arm against Full (a1), matched seeds, two-level bootstrap.

    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m comparison.ablation.analyze \
        --runs <clinical_runs_fair_v10t_ablation_<date>/top<K>> \
        --final <MedicalGNN .../clinical_runs_fair_v10t_20261011/top<K>> --out <new dir>

Full = the final cei_mp2 runs (seeds 1234/2025/7, read from --final) plus the ablation folder's a1_full runs (new
seeds). Each arm's `<arm>_seed<S>` runs are staged as `cei_mp2_seed<S>` symlinks so that `arms_vs_ref.py` (the
section-2 analysis, unchanged) matches arm and Full by seed. Decisions use VALIDATION only; test is reported once.
Completeness: Full and every requested arm (default: all arms of the frozen list) must have a finished run
(result.json) for every seed of --seeds (default: the runner's 5 seeds); otherwise the analysis is REFUSED and the
missing runs are listed. `--allow-incomplete` runs it anyway and writes the missing runs to <out>/MISSING.txt.
Read-only on the runs; --out must not exist.
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from comparison.ablation import arms_vs_ref  # noqa: E402
from comparison.ablation.run import ARMS, FINAL_SEEDS, SEEDS  # noqa: E402

RUN_RE = re.compile(r'^(?P<arm>.+)_seed(?P<seed>\d+)$')


def finished(d):
    return d.is_dir() and (d / 'result.json').is_file()


def missing_runs(runs, final, arms, seeds):
    """Expected run dirs (Full + arms x seeds) that do not hold a finished run."""
    runs, final = Path(runs), Path(final)
    expected = [final / f'cei_mp2_seed{s}' if s in FINAL_SEEDS else runs / f'a1_full_seed{s}' for s in seeds]
    expected += [runs / f'{arm}_seed{s}' for arm in arms if arm != 'a1_full' for s in seeds]
    return [str(d) for d in expected if not finished(d)]


# ablation options (method_config['arm'/'encoder_depth'] or mp2_options) with their Full (final recipe) value
ABLATION_DEFAULTS = {'arm': 'C', 'mp_feed': 'dual', 'ev_combine': 'sum', 'ev_pair': 1, 'ev_edge': 1,
                     'readout': 'evidence', 'pool_hidden': 0, 'encoder_depth': 1}
A6_OPTIONS = ('mp_optimizer', 'mp_lr_schedule', 'mp_label_smoothing')   # a6 takes these from its own HPO
# binding fields every run (Full and arms) must share with the reference Full run
COMMON = ('method', 'protocol', 'protocol_overrides', 'top_k_labels', 'num_classes', 'kept_label_indices', 'epochs',
          'patience', 'weights', 'batch_size', 'hidden', 'edge_direction', 'selection_fold', 'split_sample_ids_sha256',
          'artifact_graphs_sha256', 'targets_sha256', 'sample_seed', 'train_limit')
CONFIG_COMMON = ('k', 'comorbid_block', 'v3_state_sha256', 'hidden')


def _binding(d):
    import json
    return json.loads((Path(d) / 'binding.json').read_text())


def ablation_values(b):
    config = b.get('method_config') or {}
    values = dict(ABLATION_DEFAULTS)
    values.update({k: v for k, v in (config.get('mp2_options') or {}).items() if k in values})
    values['arm'], values['encoder_depth'] = config.get('arm'), config.get('encoder_depth')
    return values


def parity_errors(ref_dir, runs):
    """Deny-by-default recipe check. runs: [(arm, dir)]. Each run must carry exactly its arm's ablation options
    (every other ablation option at the Full value), share every COMMON field and every non-ablation mp2 option with
    the reference Full run, and use the final lr/dropout (a6: A6_HPARAMS from comparison.ablation.run)."""
    from comparison.ablation import run as R
    ref = _binding(ref_dir)
    ref_options = (ref.get('method_config') or {}).get('mp2_options') or {}
    errors = []
    for arm, d in runs:
        b = _binding(d)
        K = b.get('top_k_labels')
        want = dict(ABLATION_DEFAULTS)
        try:
            want.update(ARMS[arm]['options'](K))
        except KeyError:
            errors.append(f'{d} ({arm}): top_k_labels {K!r} has no ablation recipe (runner K values only)')
            continue
        got = ablation_values(b)
        bad = [f'{k}={got[k]!r} (want {want[k]!r})' for k in want if str(got[k]) != str(want[k])]
        bad += [f'{k} {b.get(k)!r} != reference {ref.get(k)!r}' for k in COMMON if b.get(k) != ref.get(k)]
        config, ref_config = b.get('method_config') or {}, ref.get('method_config') or {}
        bad += [f'method_config.{k} {config.get(k)!r} != reference {ref_config.get(k)!r}'
                for k in CONFIG_COMMON if config.get(k) != ref_config.get(k)]
        options = config.get('mp2_options') or {}
        skip = set(ABLATION_DEFAULTS) | (set(A6_OPTIONS) if arm == 'a6_pool' else set())
        for k in sorted((set(options) | set(ref_options)) - skip):
            if options.get(k) != ref_options.get(k):
                bad.append(f'mp2_options.{k} {options.get(k)!r} != reference {ref_options.get(k)!r}')
        if arm == 'a6_pool':
            hp = R.A6_HPARAMS
            if hp is None:
                bad.append('A6_HPARAMS is None in comparison/ablation/run.py: the a6 hyper-parameters are not recorded')
            else:
                optimizer = ('adamw', 'cosine') if hp['optimizer'] == 'adamw_cosine' else ('adam', 'none')
                want_hp = {'lr': float(hp['lr']), 'dropout': float(hp['dropout']),
                           'mp_optimizer': optimizer[0], 'mp_lr_schedule': optimizer[1],
                           'mp_label_smoothing': float(hp['label_smoothing'])}
                have_hp = {'lr': b.get('lr'), 'dropout': b.get('dropout'), **{k: options.get(k) for k in A6_OPTIONS}}
                bad += [f'{k} {have_hp[k]!r} != A6_HPARAMS {v!r}' for k, v in want_hp.items()
                        if (abs(float(have_hp[k]) - v) > 1e-12 if isinstance(v, float) else have_hp[k] != v)]
        else:
            bad += [f'{k} {b.get(k)!r} != reference {ref.get(k)!r}' for k in ('lr', 'dropout') if b.get(k) != ref.get(k)]
        if bad:
            errors.append(f'{d} ({arm}): ' + '; '.join(bad))
    return errors


def peak_rss_gb(log):
    """Peak RSS of one executed run from the `/usr/bin/time -l` lines at the end of its runner log (None if absent)."""
    if not log.is_file():
        return None
    m = re.findall(r'^\s*(\d+)\s+maximum resident set size', log.read_text(errors='replace'), flags=re.M)
    return round(int(m[-1]) / 1e9, 2) if m else None


def extra_metrics(groups, logs_dir):
    """Pre-registered secondary numbers per arm (mean ± SD over seeds, from the stored result.json / validation.npz):
    balanced accuracy, top-3, per-class F1 (validation); test balanced accuracy and top-3 (reported only);
    seconds per epoch and peak RSS (Nasim section 8; runs shared the Mac, so not contention-free)."""
    import json
    import numpy as np
    from sklearn.metrics import f1_score
    doc = {}
    for name, dirs in groups.items():
        rows, per_class, labels = [], [], None
        for seed, d in sorted(dirs.items()):
            d = Path(d)
            r = json.loads((d / 'result.json').read_text())
            z = np.load(d / 'validation.npz')
            K = z['proba'].shape[1]
            per_class.append(f1_score(z['y'], z['proba'].argmax(1), labels=np.arange(K), average=None, zero_division=0))
            labels = labels or [c.get('label') for c in sorted(r.get('per_class') or [], key=lambda c: c['index'])]
            arm_name = d.name.rsplit('_seed', 1)[0]
            rows.append({'seed': seed, 'val_balanced_acc': r['metrics']['balanced_acc'], 'val_top3_acc': r['metrics']['top3_acc'],
                         'test_balanced_acc': (r.get('test') or {}).get('balanced_acc'),
                         'test_top3_acc': (r.get('test') or {}).get('top3_acc'),
                         'seconds_per_epoch': round(r['total_seconds'] / max(len(r['history']), 1), 1),
                         'peak_rss_gb': peak_rss_gb(Path(logs_dir) / f"top{K}_{arm_name}_seed{seed}.log")})
        pcf = np.vstack(per_class)

        def ms(key):
            v = [x[key] for x in rows if x[key] is not None]
            return {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else None, 'n': len(v)} if v else None
        doc[name] = {'seeds': sorted(dirs), 'runs': rows,
                     **{k: ms(k) for k in ('val_balanced_acc', 'val_top3_acc', 'test_balanced_acc', 'test_top3_acc',
                                           'seconds_per_epoch', 'peak_rss_gb')},
                     'val_per_class_f1_mean': pcf.mean(0).round(4).tolist(),
                     'val_per_class_f1_sd': (pcf.std(0, ddof=1).round(4).tolist() if len(pcf) > 1 else None),
                     'class_labels': labels}
    return doc


def render_extra(doc):
    def f(m, nd=4):
        return 'n/a' if m is None else f"{m['mean']:.{nd}f}" + (f" ± {m['sd']:.{nd}f}" if m['sd'] is not None else '')
    L = ['# Secondary metrics per arm (validation; test reported only, not used for decisions)', '',
         '| arm | seeds | val balanced acc | val top-3 | test balanced acc | test top-3 | s/epoch | peak RSS GB |',
         '|---|---|---|---|---|---|---|---|']
    for name, g in doc.items():
        L.append(f"| {name} | {','.join(map(str, g['seeds']))} | {f(g['val_balanced_acc'])} | {f(g['val_top3_acc'])} | "
                 f"{f(g['test_balanced_acc'])} | {f(g['test_top3_acc'])} | {f(g['seconds_per_epoch'], 1)} | "
                 f"{f(g['peak_rss_gb'], 2)} |")
    L += ['', 's/epoch and peak RSS: runs shared the Mac (several jobs at once), so they are not contention-free; the '
          'final Full runs (seeds 1234/2025/7) have no memory record.', '',
          '## Validation per-class F1 (mean over seeds)', '']
    names = list(doc)
    labels = next((g['class_labels'] for g in doc.values() if g['class_labels']), None) or []
    L += ['| class | ' + ' | '.join(names) + ' |', '|---|' + '---|' * len(names)]
    for i in range(len(doc[names[0]]['val_per_class_f1_mean'])):
        label = labels[i] if i < len(labels) else str(i)
        L.append(f'| {label} | ' + ' | '.join(f"{doc[n]['val_per_class_f1_mean'][i]:.3f}" for n in names) + ' |')
    return '\n'.join(L) + '\n'


def stage(runs, final, staging, arms):
    """Symlink folders {name: dir} holding cei_mp2_seed<S> links; Full gets final seeds + a1_full seeds."""
    out = {}
    full = staging / 'full'
    full.mkdir(parents=True)
    for s in FINAL_SEEDS:
        src = Path(final) / f'cei_mp2_seed{s}'
        if src.is_dir():
            (full / f'cei_mp2_seed{s}').symlink_to(src.resolve())
    for d in sorted(Path(runs).iterdir()) if Path(runs).is_dir() else []:
        m = RUN_RE.match(d.name)
        if not (m and d.is_dir()):
            continue
        arm, seed = m['arm'], int(m['seed'])
        if arm == 'a1_full':
            if seed in FINAL_SEEDS:
                raise SystemExit(f'a1_full seed {seed} exists in both the final and the ablation runs: {d}')
            (full / f'cei_mp2_seed{seed}').symlink_to(d.resolve())
            continue
        if arms and arm not in arms:
            continue
        if arm not in ARMS:
            raise SystemExit(f'unknown arm folder {d.name}')
        target = staging / arm
        target.mkdir(exist_ok=True)
        (target / f'cei_mp2_seed{seed}').symlink_to(d.resolve())
        out[arm] = target
    return full, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--runs', required=True, help='ablation folder of one K (holds <arm>_seed<S> run dirs)')
    ap.add_argument('--final', required=True, help='final runs folder of the same K (holds cei_mp2_seed<S>)')
    ap.add_argument('--arms', default='', help='comma-separated arm ids (default: every arm of the frozen list)')
    ap.add_argument('--seeds', default=','.join(map(str, SEEDS)),
                    help='seeds every arm and Full must have (default 1234,2025,7,42,3407)')
    ap.add_argument('--allow-incomplete', action='store_true',
                    help='analyse even if runs are missing (they are listed in <out>/MISSING.txt)')
    ap.add_argument('--out', required=True, help='new output directory (must not exist)')
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        print(f'REFUSED: --out already exists: {out}', file=sys.stderr)
        return 2
    arms = {x for x in a.arms.split(',') if x}
    if bad := sorted(arms - set(ARMS)):
        print(f'REFUSED: unknown arm(s) {bad}; known: {list(ARMS)}', file=sys.stderr)
        return 2
    seeds = tuple(int(s) for s in a.seeds.split(',') if s)
    missing = missing_runs(a.runs, a.final, [x for x in ARMS if not arms or x in arms], seeds)
    if missing and not a.allow_incomplete:
        print(f'REFUSED: {len(missing)} expected run(s) missing or unfinished (no result.json):\n  '
              + '\n  '.join(missing) + '\n(--allow-incomplete analyses anyway)', file=sys.stderr)
        return 2
    groups = collect(a.runs, a.final, [x for x in ARMS if not arms or x in arms])
    ref_seed = min(groups['a1_full'], key=lambda s: (s not in FINAL_SEEDS, s)) if groups.get('a1_full') else None
    if ref_seed is None:
        print('REFUSED: no finished Full run', file=sys.stderr)
        return 2
    errors = parity_errors(groups['a1_full'][ref_seed],
                           [(arm, d) for arm, dirs in groups.items() for d in dirs.values()])
    if errors:   # never compare a run whose recipe is not its arm's
        print(f'REFUSED: {len(errors)} run(s) differ from their arm recipe (reference '
              f"{groups['a1_full'][ref_seed]}):\n  " + '\n  '.join(errors), file=sys.stderr)
        return 2
    full, staged = stage(a.runs, a.final, out / '_staging', arms)
    if missing:
        (out / 'MISSING.txt').write_text('\n'.join(missing) + '\n')
        print(f'WARNING: {len(missing)} run(s) missing, listed in {out / "MISSING.txt"}', file=sys.stderr)
    order = [x for x in ARMS if x in staged]
    args = ['--ref', f'a1_full={full}', '--method', 'cei_mp2', '--out', str(out / 'report')]
    for arm in order:
        args += ['--arm', f'{arm}={staged[arm]}']
    rc = arms_vs_ref.main(args)
    if rc == 0:
        import json
        extra = extra_metrics({k: groups[k] for k in ARMS if groups.get(k)}, Path(a.runs).parent / 'logs')
        (out / 'report' / 'extra_metrics.json').write_text(json.dumps(extra, indent=1))
        (out / 'report' / 'extra_metrics.md').write_text(render_extra(extra))
        print(f'wrote {out}/report/extra_metrics.md and extra_metrics.json')
    return rc


def collect(runs, final, arms):
    """{arm: {seed: real dir}} of FINISHED runs; Full ('a1_full') = final cei_mp2 seeds + the ablation a1_full seeds."""
    groups = {'a1_full': {}}
    for s in FINAL_SEEDS:
        d = Path(final) / f'cei_mp2_seed{s}'
        if finished(d):
            groups['a1_full'][s] = d.resolve()
    for d in sorted(Path(runs).iterdir()) if Path(runs).is_dir() else []:
        m = RUN_RE.match(d.name)
        if m and finished(d) and m['arm'] in ARMS and (m['arm'] == 'a1_full' or m['arm'] in arms):
            groups.setdefault(m['arm'], {})[int(m['seed'])] = d.resolve()
    return groups


if __name__ == '__main__':
    sys.exit(main())
