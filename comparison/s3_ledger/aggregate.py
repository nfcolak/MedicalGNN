"""Aggregate the ledger probes into the tracked tables and compare the validation run with the stored one.

Reads the probe outputs of two folds (each a folder holding blocks/, removal/, pairsecond/ as written by
comparison.s3_ledger.probe), writes block_shares.md (test tables first, validation kept as the earlier fold),
the per-fold csv copies and, when the stored validation files are given, the max abs difference per file.
Aggregates only; no model or data is loaded.

    python -m comparison.s3_ledger.aggregate --test <dir> --validation <dir> --out <new dir> [--stored <dir>]

Test is the reported fold; the validation tables are labelled "validation (earlier fold)".
"""
import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

from comparison.s3_ledger.probe import BLOCKS, shown

KS = (10, 20, 30)
BLOCK_FILES = ('block_shares.csv', 'block_shares_blocktotal.csv', 'block_shares_extra.csv')
GATES = ('parity_maxdiff', 'ledger_maxdiff')


def load_csv(path):
    rows = list(csv.DictReader(open(path)))
    rows.sort(key=lambda r: (int(r['K']), int(r['seed'])))
    return rows


def load_json_dir(folder):
    out = {}
    for path in sorted(Path(folder).glob('*.json')):
        out[path.name] = json.loads(path.read_text())
    return out


def load_fold(folder):
    folder = Path(folder)
    return {'blocks': {n: load_csv(folder / 'blocks' / n) for n in BLOCK_FILES},
            'removal': load_json_dir(folder / 'removal'), 'pairsecond': load_json_dir(folder / 'pairsecond')}


def ms(values, dec):
    return f'{st.mean(values):.{dec}f} ± {st.stdev(values):.{dec}f}'


def share_table(rows, prefix):
    out = ['| K | ' + ' | '.join(BLOCKS) + ' |', '|---|' + '---|' * len(BLOCKS)]
    for k in KS:
        rr = [r for r in rows if int(r['K']) == k]
        out.append(f'| {k} | ' + ' | '.join(ms([float(r[f"{prefix}_{b}"]) * 100 for r in rr], 1) for b in BLOCKS) + ' |')
    return '\n'.join(out)


def pred_table(rows):
    cols = list(BLOCKS) + ['bias']
    out = ['| K | ' + ' | '.join(cols) + ' | sum = predicted-class logit |', '|---|' + '---|' * (len(cols) + 1)]
    for k in KS:
        rr = [r for r in rows if int(r['K']) == k]
        cells = [ms([float(r[f'pred_{c}']) for r in rr], 2) for c in cols]
        total = [sum(float(r[f'pred_{c}']) for c in cols) for r in rr]
        out.append(f'| {k} | ' + ' | '.join(cells) + f' | {ms(total, 2)} |')
    return '\n'.join(out)


def run_table(rows):
    out = ['| K | seed | parity max diff | ledger max diff | ' + ' | '.join(BLOCKS) + ' (item-level mass %) |',
           '|---|---|---|---|' + '---|' * len(BLOCKS)]
    for r in rows:
        out.append(f"| {r['K']} | {r['seed']} | {float(r['parity_maxdiff']):.1e} | {float(r['ledger_maxdiff']):.1e} | "
                   + ' | '.join(f"{float(r['mass_' + b]) * 100:.1f}" for b in BLOCKS) + ' |')
    return '\n'.join(out)


def probe_rows(fold):
    rows = []
    for name, res in fold['removal'].items():
        rows.append({'probe': 'removal', 'K': res['K'], 'seed': res['seed'], 'n_graphs': res['n_graphs'],
                     'gap_mean_allclasses': res['node_block_removal_gap_rel_mean_allclasses'],
                     'gap_median_allclasses': res['node_block_removal_gap_rel_median_allclasses'],
                     'gap_mean_predclass': res['node_block_removal_gap_rel_mean_predclass'],
                     'nbr_share_item_level': res['nbr_mass_share_item_level_mean'],
                     'nbr_share_block_total': res['nbr_mass_share_block_total_mean'],
                     'pair_second_difference_max_abs': '', 'pair_vote_mean_abs': ''})
    for name, res in fold['pairsecond'].items():
        rows.append({'probe': 'pairsecond', 'K': res['K'], 'seed': res['seed'], 'n_graphs': res['n_graphs'],
                     'gap_mean_allclasses': '', 'gap_median_allclasses': '', 'gap_mean_predclass': '',
                     'nbr_share_item_level': '', 'nbr_share_block_total': '',
                     'pair_second_difference_max_abs': res['pair_second_difference_max_abs'],
                     'pair_vote_mean_abs': res['pair_vote_mean_abs']})
    rows.sort(key=lambda r: (r['probe'], int(r['K']), int(r['seed'])))
    return rows


def fold_section(title, fold):
    main, bt, extra = (fold['blocks'][n] for n in BLOCK_FILES)
    classes = ', '.join(f'K={k}: {[r for r in extra if int(r["K"]) == k][0]["classes_in_sample"]}' for k in KS)
    parity = max(float(r['parity_maxdiff']) for r in main)
    ledger = [float(r['ledger_maxdiff']) for r in main]
    n = main[0]['n_graphs']
    lines = [f'## {title}', '',
             f'{len(main)} runs, first {n} sample ids of the fold per K after sorting the id strings (identical for the three seeds; '
             f'classes present: {classes}). Parity (reloaded softmax vs stored predictions of the fold): max abs diff {parity:.1e} '
             f'(limit 1e-4), argmax agreement 100% in every run. Ledger exactness (max |logit - bias - sum of parts|, float64): '
             f'max {max(ledger):.1e}, min {min(ledger):.1e}.', '',
             '### Absolute evidence mass share, % (item-level |parts|; mean ± SD over 3 seeds, n-1)', '', share_table(main, 'mass'), '',
             '### Same, block-total variant (cancellation inside a block allowed), %', '', share_table(bt, 'mass'), '',
             '### Contribution to the predicted-class logit (mean over graphs, logit units; mean ± SD over seeds)', '', pred_table(main), '',
             '### Per run', '', run_table(main), '']
    removal = sorted(fold['removal'].values(), key=lambda r: (r['K'], r['seed']))
    if removal:
        lines += ['### Node-block removal gap and nbr share (seed 1234)', '',
                  '| K | graphs | gap, all classes (mean) | gap, all classes (median) | gap, predicted class | nbr share item-level | nbr share block-total |',
                  '|---|---|---|---|---|---|---|']
        for r in removal:
            lines.append(f"| {r['K']} | {r['n_graphs']} | {r['node_block_removal_gap_rel_mean_allclasses']:.3f} | "
                         f"{r['node_block_removal_gap_rel_median_allclasses']:.3f} | {r['node_block_removal_gap_rel_mean_predclass']:.3f} | "
                         f"{r['nbr_mass_share_item_level_mean']:.3f} | {r['nbr_mass_share_block_total_mean']:.3f} |")
        lines.append('')
    pair = sorted(fold['pairsecond'].values(), key=lambda r: (r['K'], r['seed']))
    if pair:
        lines += ['### Pair vote second difference', '', '| K | seed | pairs sampled | max abs second difference | mean abs pair vote |', '|---|---|---|---|---|']
        for r in pair:
            lines.append(f"| {r['K']} | {r['seed']} | {r['pairs_sampled']} | {r['pair_second_difference_max_abs']:.2e} | {r['pair_vote_mean_abs']:.3f} |")
        lines.append('')
    return lines


def write_csv(path, rows):
    with open(path, 'w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def numeric_diff(new, old, skip=()):
    worst, key = 0.0, None
    for field, value in old.items():
        if field in skip or field not in new or value in ('', None):
            continue
        try:
            d = abs(float(new[field]) - float(value))
        except (TypeError, ValueError):
            continue
        if d > worst:
            worst, key = d, field
    return worst, key


def compare_stored(validation, stored):
    """Max abs difference of the validation re-run against the stored validation files."""
    stored = Path(stored)
    lines = ['| stored file | rows | max abs diff | column of the max |', '|---|---|---|---|']
    results = {}
    for name in BLOCK_FILES:
        old = load_csv(stored / name)
        new = {(int(r['K']), int(r['seed'])): r for r in validation['blocks'][name]}
        worst, col, n = 0.0, None, 0
        for o in old:
            key = (int(o['K']), int(o['seed']))
            if key not in new:
                continue
            d, c = numeric_diff(new[key], o, skip=GATES)
            n += 1
            if d > worst:
                worst, col = d, c
        results[name] = worst
        lines.append(f'| {name} | {n} of {len(old)} | {worst:.2e} | {col or "-"} |')
    gate = max(abs(float(n['ledger_maxdiff']) - float(o['ledger_maxdiff']))
               for n, o in ((validation['blocks']['block_shares.csv'][i], load_csv(stored / 'block_shares.csv')[i])
                            for i in range(len(validation['blocks']['block_shares.csv']))))
    lines.append(f'| ledger_maxdiff column (float32 round-off level, informational) | 9 of 9 | {gate:.2e} | ledger_maxdiff |')
    for family, label in (('removal', 'removal_probe'), ('pairsecond', 'pairsecond')):
        for path in sorted(stored.glob(f'{label}_*.json')):
            old = json.loads(path.read_text())
            new = validation[family].get(path.name)
            if new is None:
                lines.append(f'| {path.name} | not re-run | - | - |')
                continue
            d, c = numeric_diff(new, old)
            results[path.name] = d
            lines.append(f'| {path.name} | 1 | {d:.2e} | {c or "-"} |')
    return lines, results


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m comparison.s3_ledger.aggregate', description=__doc__.split('\n\n')[0])
    ap.add_argument('--test', required=True, help='probe outputs on the test fold (blocks/, removal/, pairsecond/)')
    ap.add_argument('--validation', required=True, help='probe outputs on the validation fold')
    ap.add_argument('--stored', default=None, help='stored validation files (csv/json) to compare the validation re-run with')
    ap.add_argument('--out', required=True, help='new output directory (must not exist)')
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        ap.error('--out exists; choose a new directory')
    test, validation = load_fold(a.test), load_fold(a.validation)
    out.mkdir(parents=True)
    md = ['# Evidence-ledger probes of the CEI-GNN mp2 final runs', '',
          'Final fair_v10t runs, K = 10, 20, 30 x seeds 1234, 2025, 7 (`best.pt`), forward passes only. Test is the reported fold; '
          'validation (the decision fold, the earlier numbers) is kept below. Probes: `comparison/s3_ledger/probe.py`.', '',
          '- Mass share (primary): per graph, mass_b = sum over all items of block b and all classes of |part|; denominator = sum of the five masses '
          '(bias excluded); averaged over graphs. Parts of opposite sign inside a block do not cancel: how much signed evidence a block puts on the '
          'table, not how much it moves the logit.',
          '- Block-total variant: |block total T_c| summed over classes; cancellation inside a block allowed.',
          '- Predicted-class contribution: mean block total at the argmax class (signed, logit units); with the bias the seven numbers add up to the mean predicted-class logit.',
          '- Removal gap (node block): relative difference between the true effect of deleting a node (denominator renormalised) and the ledger-naive effect (its own part).',
          '- Pair second difference: f(zi+zj) - f(zi) - f(zj) + f(0) of the pair vote; zero iff the vote is endpoint-additive.', '']
    md += fold_section('Test (reported fold)', test)
    md += fold_section('Validation (decision fold; earlier numbers)', validation)
    if a.stored:
        lines, _ = compare_stored(validation, a.stored)
        md += ['## Validation reproduction against the stored validation files', '',
               'Re-run with `--fold validation` against the stored validation files of the earlier probe (max abs difference of every shared number).', ''] + lines + ['']
    (out / 'block_shares.md').write_text('\n'.join(md) + '\n')
    for tag, fold in (('test', test), ('validation', validation)):
        for name in BLOCK_FILES:
            write_csv(out / f'{name[:-4]}_{tag}.csv', fold['blocks'][name])
        rows = probe_rows(fold)
        if rows:
            write_csv(out / f'removal_pairsecond_{tag}.csv', rows)
    print(f'wrote {len(list(out.iterdir()))} files to {shown(out)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
