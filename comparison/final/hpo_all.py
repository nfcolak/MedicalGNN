"""Recorded HPO of the four tuned final methods (gsat, graphcare, protgnn, cei_mp2): commands, study folders, best trials,
and a check of the best values against the tuned flags in `args()` of comparison/final/run.py.

Default = plan only: prints, per method, the comparison.final.hpo command, the recorded study folder, the best trial with
its values (best.json), and the matching `args()` line of run.py; any difference is flagged as MISMATCH and the exit
code is 1. Nothing is trained and nothing is written.
`--execute` runs comparison.final.hpo once per method (real training, TPE seed 2026, 12 trials each; needs the user's
approval). Setup of the recorded study: K=17, 10k TRAIN sample, 20 epochs, objective on the dev split, validation and
test never scored (see comparison/final/hpo.py).

Usage (from the repo root):
    python3 -m comparison.final.hpo_all [--name fv10_k17_20261011] [--execute]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from comparison.final import run

REPO = Path(__file__).resolve().parents[2]
NAME = 'fv10_k17_20261011'
N_TRIALS = 12
METHODS = ['gsat', 'graphcare', 'protgnn', 'cei_mp2']   # GCHM-PNA is not tuned in new analyses
SPECIFIC = {   # best.json param -> run.py flag
    'gsat': {'info_loss_coef': '--gsat-info-loss-coef'},
    'graphcare': {'decay_rate': '--graphcare-decay-rate'},
    'protgnn': {'prototypes_per_class': '--protgnn-prototypes-per-class'},
    'cei_mp2': {},
}


def flag_values(argv):
    """{'--lr': ['0.00369'], '--method-option': ['arm=C', ...]} from a flag list."""
    d, key = {}, None
    for tok in argv:
        if tok.startswith('--'):
            key = tok
            d.setdefault(key, [])
        elif key:
            d[key].append(tok)
    return d


def same(flag_value, v):
    """run.py rounds to 3 significant digits; best.json holds the full value."""
    return float(flag_value) == float(f'{v:.3g}')


def check(method, params, flags):
    """List of mismatch descriptions between best.json params and the args() flags."""
    bad = []
    for p, f in (('lr', '--lr'), ('weight_decay', '--weight-decay'), ('dropout', '--dropout'), *SPECIFIC[method].items()):
        got = flags.get(f)
        if not got:
            bad.append(f'{f} missing in run.py args()')
        elif p == 'prototypes_per_class':
            if int(got[0]) != params[p]:
                bad.append(f'{f}: run.py {got[0]} vs best {params[p]}')
        elif not same(got[0], params[p]):
            bad.append(f'{f}: run.py {got[0]} vs best {params[p]:.6g}')
    if method == 'cei_mp2':
        opts = dict(o.split('=', 1) for o in flags.get('--method-option', []))
        want = {'mp_optimizer': 'adamw', 'mp_lr_schedule': 'cosine'} if params['optimizer'] == 'adamw_cosine' else \
               {'mp_optimizer': 'adam', 'mp_lr_schedule': 'none'}
        for k, v in want.items():
            if opts.get(k) != v:
                bad.append(f'--method-option {k}: run.py {opts.get(k)} vs best {v}')
        if float(opts.get('mp_label_smoothing', 'nan')) != params['label_smoothing']:
            bad.append(f"--method-option mp_label_smoothing: run.py {opts.get('mp_label_smoothing')} vs best {params['label_smoothing']}")
    else:
        got = flags.get('--optimizer', [None])[0]
        if got != params['optimizer']:
            bad.append(f"--optimizer: run.py {got} vs best {params['optimizer']}")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(prog='comparison.final.hpo_all', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', default=NAME, help=f'study name suffix, folder comparison/standardized/hpo_<name> (default {NAME})')
    ap.add_argument('--execute', action='store_true',
                    help='run comparison.final.hpo per method (real training; needs the user\'s approval)')
    a = ap.parse_args(argv)
    P = run.Paths(REPO, run.DATE)
    mismatches = 0
    for m in METHODS:
        study = Path('comparison/standardized') / f'hpo_{a.name}' / m
        cmd = ['python3', '-m', 'comparison.final.hpo', m, str(N_TRIALS), '--name', a.name]
        print(f'== {m}')
        print('  hpo command :', ' '.join(cmd))
        print('  study folder:', study)
        best_file = REPO / study / 'best.json'
        argline = run.args(P, m, 17)
        shown = ' '.join(t.replace(str(P.events), 'comparison/standardized/event_inputs') for t in argline)
        print('  run.py args():', shown)
        if not best_file.is_file():
            print('  best.json    : MISSING')
            mismatches += 1
            continue
        b = json.loads(best_file.read_text())
        print(f"  best trial   : {b['trial']} of {b['finished_trials']} finished, plateau {b['plateau']:.4f} "
              f"(trial 0 default {b['default_plateau']:.4f}), best dev {b['best_dev']:.4f} at epoch {b['best_epoch']}")
        print('  best values  :', ', '.join(f'{k}={v:.6g}' if isinstance(v, float) else f'{k}={v}' for k, v in b['params'].items()))
        if b['finished_trials'] != N_TRIALS:
            print(f"  MISMATCH finished_trials {b['finished_trials']} != {N_TRIALS}")
            mismatches += 1
        bad = check(m, b['params'], flag_values(argline))
        for line in bad:
            print('  MISMATCH', line)
        mismatches += len(bad)
        if not bad:
            print('  run.py args() matches best.json (3 significant digits)')
    print(f'{mismatches} mismatches' if mismatches else '0 mismatches')

    if a.execute:
        for m in METHODS:
            rc = subprocess.run([sys.executable, '-m', 'comparison.final.hpo', m, str(N_TRIALS), '--name', a.name], cwd=REPO).returncode
            if rc != 0:
                print(f'hpo {m} failed rc={rc}', file=sys.stderr)
                return rc
    return 1 if mismatches else 0


if __name__ == '__main__':
    sys.exit(main())
