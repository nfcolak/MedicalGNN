"""Light HPO (Optuna TPE + median pruning) for one final method, as run for fair_v11_final (2026-10-11).

Setup of the recorded HPO: K=17, the old 10k TRAIN sample (seed 1234), 20 epochs, seed 1234, checkpoint/objective on the
5,000-row dev split, validation and test never scored, 3 CPU threads, nice 10, 12 trials, trial 0 = the method's
previous setting. Objective = mean dev macro-F1 over epochs 16-20. hidden/layers fixed. The tuned values are the
`args()` of comparison/final/run.py. K=17 and the 10k sample are protocol overrides here (allowed: tuning runs only).
Output: comparison/standardized/hpo_<name>/<method>/ (study.db, trial logs, trials.json, best.json).
Usage: python3 -m comparison.final.hpo gsat|graphcare|protgnn|gchm_pna|cei_mp2 N_TRIALS [--name fv10_k17_20261011]
Real-data training: only with the user's approval.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import optuna

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('MEDGNN_DATA_ROOT', REPO))
E = ROOT / 'comparison/standardized/event_inputs'
G = E / 'clinical_graph_L4_plus60_v5c_20261009'
T = E / 'targets_primary_v4_v5plus60_v5_test_20261011/targets.csv'   # train/validation rows = the HPO-time file
K4 = E / 'v3_state_L4_v5c_top17_20261011/K4.json'                     # fitted on the old 10k sample, K=17
EPOCHS, THREADS, K = 20, 3, 17

BASE = {
    'gsat': ['--method', 'gsat'],
    'graphcare': ['--method', 'graphcare'],
    'protgnn': ['--method', 'protgnn'],
    'gchm_pna': ['--method', 'clinical_gnn', '--conv', 'gchm'],
    'cei_mp2': ['--method', 'cei_gnn_mp2'] + sum((['--method-option', o] for o in [
        'arm=C', 'k=4', f'v3_state={K4}', 'mp_feed=dual', 'mp_hub_max=1', 'mp_dedup=1', 'mp_ctx=table',
        'mp_root=bias']), []),
}
# previous settings (trial 0); gsat weight_decay 0.0 -> 1e-6 (log-space floor)
DEFAULTS = {
    'gsat': dict(lr=1e-3, weight_decay=1e-6, dropout=0.3, optimizer='native', info_loss_coef=1.0),
    'graphcare': dict(lr=1e-3, weight_decay=1e-5, dropout=0.3, optimizer='native', decay_rate=0.03),
    'protgnn': dict(lr=1.79e-3, weight_decay=4.3e-5, dropout=0.51, optimizer='native', prototypes_per_class=3),
    'gchm_pna': dict(lr=1e-3, weight_decay=1e-5, dropout=0.1, optimizer='native', batch_size=64),
    'cei_mp2': dict(lr=1.79e-3, weight_decay=4.3e-5, dropout=0.3, optimizer='adamw_cosine', label_smoothing=0.0),
}


def space(model, trial):
    opts = ['adam', 'adamw_cosine'] if model == 'cei_mp2' else ['native', 'adamw_cosine']
    p = dict(lr=trial.suggest_float('lr', 3e-4, 5e-3, log=True),
             weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
             dropout=trial.suggest_float('dropout', 0.1, 0.6),
             optimizer=trial.suggest_categorical('optimizer', opts))
    if model == 'gsat':
        p['info_loss_coef'] = trial.suggest_float('info_loss_coef', 0.1, 2.0, log=True)
    elif model == 'graphcare':
        p['decay_rate'] = trial.suggest_float('decay_rate', 0.01, 0.1, log=True)
    elif model == 'protgnn':
        p['prototypes_per_class'] = trial.suggest_categorical('prototypes_per_class', [3, 5, 8])
    elif model == 'gchm_pna':
        p['batch_size'] = trial.suggest_categorical('batch_size', [32, 64, 128])
    else:
        p['label_smoothing'] = trial.suggest_categorical('label_smoothing', [0.0, 0.1])
    return p


def command(model, p, out):
    cmd = ['nice', '-n', '10', 'env', '-u', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE=1', '/usr/bin/python3',
           '-m', 'core.train'] + BASE[model] + [
           '--artifact', str(G), '--targets', str(T), '--output', str(out), '--seed', '1234',
           '--top-k-labels', str(K), '--train-limit', '10000', '--epochs', str(EPOCHS),
           '--selection-fold', 'dev', '--dev-limit', '5000', '--final-eval', 'none',
           '--cpu-threads', str(THREADS), '--execute',
           '--lr', str(p['lr']), '--weight-decay', str(p['weight_decay']), '--dropout', str(p['dropout'])]
    if model == 'cei_mp2':
        aw = p['optimizer'] == 'adamw_cosine'
        cmd += ['--method-option', 'mp_optimizer=' + ('adamw' if aw else 'adam'),
                '--method-option', 'mp_lr_schedule=' + ('cosine' if aw else 'none'),
                '--method-option', f"mp_label_smoothing={p['label_smoothing']}"]
    else:
        cmd += ['--optimizer', p['optimizer']]
    if model == 'gsat':
        cmd += ['--gsat-info-loss-coef', str(p['info_loss_coef'])]
    elif model == 'graphcare':
        cmd += ['--graphcare-decay-rate', str(p['decay_rate'])]
    elif model == 'protgnn':
        cmd += ['--protgnn-prototypes-per-class', str(p['prototypes_per_class'])]
    elif model == 'gchm_pna':
        cmd += ['--batch-size', str(p['batch_size'])]
    return cmd


def main(argv=None):
    ap = argparse.ArgumentParser(prog='comparison.final.hpo')
    ap.add_argument('model', choices=sorted(BASE))
    ap.add_argument('n_trials', type=int)
    ap.add_argument('--name', default='fv10_k17_20261011')
    a = ap.parse_args(argv)
    model = a.model
    X = ROOT / 'comparison/standardized' / f'hpo_{a.name}' / model
    X.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        p = space(model, trial)
        out = X / f'trial_{trial.number:03d}'
        log = open(X / f'trial_{trial.number:03d}.log', 'w')
        log.write(' '.join(command(model, p, out)) + '\n')
        proc = subprocess.Popen(command(model, p, out), cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        f1s, t0 = [], time.time()
        for line in proc.stdout:
            log.write(line); log.flush()
            if line.startswith('{"epoch"'):
                e = json.loads(line); f1s.append(e['macro_f1'])
                trial.report(max(f1s), e['epoch'])
                if trial.should_prune():
                    proc.kill(); proc.wait()
                    trial.set_user_attr('pruned_at', e['epoch']); trial.set_user_attr('best_dev', max(f1s))
                    raise optuna.TrialPruned()
        rc = proc.wait()
        if rc != 0 or len(f1s) < EPOCHS:
            raise RuntimeError(f'trial {trial.number} rc={rc} epochs={len(f1s)}')
        best = max(f1s)
        trial.set_user_attr('best_dev', best); trial.set_user_attr('best_epoch', f1s.index(best) + 1)
        trial.set_user_attr('minutes', round((time.time() - t0) / 60, 1))
        return sum(f1s[15:20]) / 5

    study = optuna.create_study(study_name=f'hpo_{model}', direction='maximize',
                                storage=f'sqlite:///{X}/study.db', load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=2026, n_startup_trials=5),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=10, interval_steps=2))
    if not study.trials:
        study.enqueue_trial(DEFAULTS[model])
    done = len([t for t in study.trials if t.state.is_finished()])
    study.optimize(objective, n_trials=max(0, a.n_trials - done), catch=(RuntimeError,))
    rows = [dict(n=t.number, state=t.state.name, plateau=t.value, **t.user_attrs, **t.params) for t in study.trials]
    json.dump(rows, open(X / 'trials.json', 'w'), indent=1)
    best, t0 = study.best_trial, study.trials[0]
    json.dump(dict(params=best.params, plateau=best.value, trial=best.number, **best.user_attrs,
                   default_plateau=t0.value, default_best_dev=t0.user_attrs.get('best_dev'),
                   finished_trials=len([t for t in study.trials if t.state.is_finished()])),
              open(X / 'best.json', 'w'), indent=1)
    print('BEST', model, best.number, best.value, best.params)


if __name__ == '__main__':
    main()
