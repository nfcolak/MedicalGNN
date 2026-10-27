"""Graph-type analysis, step 4: explanation content axis.

Run from the repo root:  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 comparison/explain/graph_types/step4_content.py

Reads only W/data/*.csv (written by step 1) inside this script; writes out/step4-report.md (aggregates only, every
cell n >= 10 visits, cells under 30 visits are listed but not interpreted) and scratch/step4/step4_numbers.json.
Paths: _paths.py in this folder (W = working folder with data/ and out/; data/ holds per-visit tables with anonymous indices and is never committed).

Fixed before looking at any difference:
  * unit = visit; a visit counts for (key, source) when ok in >= 2 of the 3 seeds; its value = mean over its ok seeds
    (top-kind rows exist only where the record carried an importance vector with support);
  * bootstrap = patient-cluster, 2,000 draws, seed 20261025, 95% percentile CI, one shared draw matrix;
  * class groups (CLASS_GROUP below, 6 groups covering all 20 classes);
  * graph-type bins: history depth (num_visits 1 / 2-3 / >=4), evidence size (tertiles of n_evidence), lab richness
    (tertiles of the availability share n_kind_measurement / n_evidence);
  * composition bins: tertiles over visits of the CEI model's own top_share_measurement (per source) for the
    stratified view; the regression view uses the continuous share (slope per +0.10);
  * confidence adjustment: (a) regression of d on the share plus conf_cei and conf_other (the other model), and
    (b) stratified: composition high-minus-low tertile contrast averaged over tertiles of conf_cei - conf_other.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from comparison.explain.graph_types._paths import WORK  # noqa: E402
W = WORK
DATA, OUT, SCRATCH = W / 'data', W / 'out', W / 'scratch' / 'step4'
STARTED = time.time()

SEEDS = (1234, 2025, 7)
KINDS = ('complaint', 'measurement', 'vital', 'diagnosis')
SOURCES = ('IG', 'Grad', 'GNNExp', 'builtin')
MODELS = ('gsat', 'graphcare', 'protgnn', 'cei_final', 'cei_R1_ls0')
CEI_KEYS = ('cei_final', 'cei_R1_ls0')
OTHERS = {'GSAT': 'gsat', 'GraphCare': 'graphcare'}
# the best post-hoc explainer of each comparator, used for the "built-in vs best post-hoc" pairs
BEST_POSTHOC = {'GSAT': 'IG', 'GraphCare': 'Grad'}
HEADLINE = 'deletion_mean_change_vs_random'
DRAWS, BOOT_SEED = 2000, 20261025
MIN_PRINT, MIN_INTERPRET = 10, 30

CLASS_GROUP = {
    'Heart failure': 'cardiovascular', 'Non-ST elevation (NSTEMI) myocardial infarction': 'cardiovascular',
    'Unspecified atrial fibrillation': 'cardiovascular',
    'Acute kidney failure': 'metabolic-renal-systemic', 'Anemia': 'metabolic-renal-systemic',
    'Diabetes mellitus': 'metabolic-renal-systemic', 'Hypokalemia': 'metabolic-renal-systemic',
    'Sepsis': 'metabolic-renal-systemic',
    'Acute pharyngitis': 'infection-respiratory', 'Lower respiratory disease': 'infection-respiratory',
    'Skin or soft-tissue infection': 'infection-respiratory', 'UTI or pyelonephritis': 'infection-respiratory',
    'Back or spine pain': 'injury-musculoskeletal', 'Head injury': 'injury-musculoskeletal',
    'Limb injury or pain': 'injury-musculoskeletal',
    'GI bleed': 'gastrointestinal', 'Unsp intestnl obst': 'gastrointestinal',
    'Alcohol abuse with intoxication': 'neuro-psych-substance', 'Epilepsy': 'neuro-psych-substance',
    'Major depressive disorder': 'neuro-psych-substance',
}


def say(text=''):
    print(text, flush=True)


# --------------------------------------------------------------------------- loading
def load_features():
    with open(DATA / 'visit_features.csv', newline='') as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda r: int(r['visit_idx']))
    assert [int(r['visit_idx']) for r in rows] == list(range(len(rows)))
    num = lambda col: np.array([float(r[col]) for r in rows])
    feat = {'patient': np.array([int(r['patient_idx']) for r in rows]), 'num_visits': num('num_visits'),
            'n_evidence': num('n_evidence'), 'n_prior_dx': num('n_prior_dx'),
            'class_group': np.array([CLASS_GROUP[r['true_class']] for r in rows])}
    for kind in KINDS:
        feat['avail_' + kind] = num('n_kind_' + kind) / feat['n_evidence']
    for key in MODELS:
        feat['conf_' + key] = num('conf_' + key)
    return feat


def load_cells(n_visits):
    """{(key, source): {'del': (n_visits,), 'top_share_<kind>': ..., 'mass_share_<kind>': ..., 'top_rate_<kind>': ...}}
    seed-averaged over ok seeds; NaN where fewer than 2 seeds are ok / carry a vector."""
    cols = [f'{p}_{k}' for p in ('top_share', 'mass_share', 'top_rate') for k in KINDS]
    ok = {}
    with open(DATA / 'visit_scores.csv', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['key'] in MODELS and row['ok'] == '1' and row[HEADLINE] != '':
                ok.setdefault((row['key'], row['source']), {}).setdefault(int(row['visit_idx']), {})[int(row['seed'])] = \
                    float(row[HEADLINE])
    top = {}
    with open(DATA / 'top_kinds.csv', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['key'] in MODELS:
                top.setdefault((row['key'], row['source']), {}).setdefault(int(row['visit_idx']), {})[int(row['seed'])] = \
                    [float(row[c]) if row[c] != '' else np.nan for c in cols]
    cells = {}
    for key in MODELS:
        for source in SOURCES:
            cell = {name: np.full(n_visits, np.nan) for name in ['del'] + cols}
            ok_seeds = ok.get((key, source), {})
            for v, by_seed in ok_seeds.items():
                if len(by_seed) >= 2:
                    cell['del'][v] = np.mean(list(by_seed.values()))
            for v, by_seed in top.get((key, source), {}).items():
                usable = [s for s in by_seed if s in ok_seeds.get(v, {})]
                if len(usable) >= 2:
                    arr = np.array([by_seed[s] for s in usable])
                    with np.errstate(all='ignore'):
                        mean = np.nanmean(arr, axis=0)
                    for i, c in enumerate(cols):
                        cell[c][v] = mean[i]
            cells[(key, source)] = cell
    return cells, cols


# --------------------------------------------------------------------------- bootstrap helpers
class Boot:
    """Patient-cluster bootstrap with one shared draw matrix: counts[b, p] = times patient p is drawn in draw b."""

    def __init__(self, patient):
        self.patient = patient
        self.n_patients = int(patient.max()) + 1
        rng = np.random.default_rng(BOOT_SEED)
        draws = rng.integers(0, self.n_patients, size=(DRAWS, self.n_patients))
        self.counts = np.stack([np.bincount(d, minlength=self.n_patients) for d in draws]).astype(float)
        self.w = self.counts[:, patient]                      # (DRAWS, n_visits) visit weights

    def n_patients_in(self, mask):
        return int(np.unique(self.patient[mask]).size)

    def mean(self, x, mask):
        """point estimate and bootstrap draws of the mean of x over the visits in mask (NaN visits dropped)."""
        use = mask & np.isfinite(x)
        xv = np.where(use, x, 0.0)
        den = self.w @ use.astype(float)
        num = self.w @ xv
        with np.errstate(all='ignore'):
            draws = num / den
        return float(x[use].mean()) if use.any() else np.nan, draws, int(use.sum())

    def interval(self, point, draws):
        d = draws[np.isfinite(draws)]
        if d.size < DRAWS * 0.95 or not np.isfinite(point):
            return np.nan, np.nan
        return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    def wls(self, y, columns, mask):
        """weighted least squares of y on [1, columns...] over mask; returns point coefs and (DRAWS, p) draws."""
        use = mask & np.isfinite(y)
        for c in columns:
            use &= np.isfinite(c)
        X = np.column_stack([np.ones(use.sum())] + [c[use] for c in columns])
        yy = y[use]
        beta = np.linalg.lstsq(X, yy, rcond=None)[0]
        w = self.w[:, use]
        p = X.shape[1]
        xtx = np.empty((DRAWS, p, p))
        for i in range(p):
            for j in range(i, p):
                xtx[:, i, j] = xtx[:, j, i] = w @ (X[:, i] * X[:, j])
        xty = np.stack([w @ (X[:, i] * yy) for i in range(p)], axis=1)
        with np.errstate(all='ignore'):
            try:
                draws = np.linalg.solve(xtx + 1e-9 * np.eye(p), xty[..., None])[..., 0]
            except np.linalg.LinAlgError:
                draws = np.full((DRAWS, p), np.nan)
        return beta, draws, int(use.sum())


def fmt(value, digits=3):
    return '' if value is None or not np.isfinite(value) else f'{value:+.{digits}f}'


def fmt_plain(value, digits=3):
    return '' if value is None or not np.isfinite(value) else f'{value:.{digits}f}'


def ci_text(lo, hi, digits=3):
    return '' if not (np.isfinite(lo) and np.isfinite(hi)) else f'[{lo:+.{digits}f}, {hi:+.{digits}f}]'


def excludes_zero(lo, hi):
    return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))


def cell_ok(n):
    """(printable, interpretable) under the cell-size rules."""
    return n >= MIN_PRINT, n >= MIN_INTERPRET


# --------------------------------------------------------------------------- analyses
def composition_overall(boot, cells, feat, mask_all, lines, numbers):
    """Table A: mean top_share / mass_share / top_rate per model, source, kind (patient-cluster CI)."""
    lines += ['## A. Composition of each model\'s top nodes (overall)', '',
              'top_share = share of the top-k evidence nodes that are of the kind (k = the record\'s top-20% rule); '
              'mass_share = share of total |importance| over evidence nodes that sits on the kind; top_rate = share of the '
              'available nodes of that kind that enter the top k. Seed-averaged per visit, then averaged over visits; '
              'CI = patient-cluster bootstrap. "availability" = share of evidence nodes of that kind in the graph '
              '(what a size-blind pick would give). n = visits with the explainer ok in >= 2 seeds and an importance '
              'vector with support.', '']
    avail = {k: boot.mean(feat['avail_' + k], mask_all) for k in KINDS}
    lines += ['Availability of evidence nodes by kind (mean over the 1,345 visits): ' + ', '.join(
        f'{k} {avail[k][0]:.3f}' for k in KINDS) + '.', '']
    for measure in ('top_share', 'mass_share', 'top_rate'):
        extra = (' top_rate is undefined for a kind the visit does not have (e.g. no complaint node), so each kind '
                 'averages over the visits that have it; its n is given in brackets.' if measure == 'top_rate' else '')
        lines += [f'### {measure}', '', extra.strip(), '' if extra else None,
                  '| source | model | n visits | n patients | ' + ' | '.join(
                      f'{k} (95% CI)' for k in KINDS) + ' |', '|---|---|---|---|' + '---|' * len(KINDS)]
        lines[:] = [x for x in lines if x is not None]
        for source in SOURCES:
            for key in MODELS:
                cell = cells[(key, source)]
                first = cell[f'top_share_{KINDS[0]}']
                visit_use = mask_all & np.isfinite(first)
                n = int(visit_use.sum())
                if n < MIN_PRINT:
                    continue
                parts = []
                for kind in KINDS:
                    use = visit_use & np.isfinite(cell[f'{measure}_{kind}'])
                    if use.sum() < MIN_PRINT:
                        parts.append('')
                        continue
                    pt, dr, nk = boot.mean(cell[f'{measure}_{kind}'], use)
                    lo, hi = boot.interval(pt, dr)
                    parts.append(f'{pt:.3f} {ci_text(lo, hi)}' + (f' [n={nk}]' if measure == 'top_rate' else ''))
                    numbers.setdefault('overall', {})[f'{measure}|{source}|{key}|{kind}'] = pt
                lines.append(f'| {source} | {key} | {n} | {boot.n_patients_in(visit_use)} | ' + ' | '.join(parts) + ' |')
        lines.append('')
    return avail


def composition_pairs(boot, cells, mask_all, lines, numbers):
    """Table B: paired composition difference (CEI - comparator) per source and kind, top_share and mass_share."""
    lines += ['## B. Paired composition difference (CEI - comparator), same visit, same explainer', '',
              'Mean over visits of (CEI value - comparator value); both must carry the vector in >= 2 seeds. '
              'A positive top_share difference for "measurement" means CEI\'s top nodes are more often labs than the '
              'comparator\'s on the same visit.', '']
    for measure in ('top_share', 'mass_share'):
        lines += [f'### {measure}', '', '| CEI | vs | source | n visits | n patients | ' + ' | '.join(
            f'{k} diff (95% CI)' for k in KINDS) + ' |', '|---|---|---|---|---|' + '---|' * len(KINDS)]
        for cei in CEI_KEYS:
            for name, other in OTHERS.items():
                for source in SOURCES:
                    if name == 'GraphCare' and source == 'builtin':
                        continue            # GraphCare's built-in is constant / tied: not an explanation
                    a, b = cells[(cei, source)], cells[(other, source)]
                    use = mask_all & np.isfinite(a[f'{measure}_{KINDS[0]}']) & np.isfinite(b[f'{measure}_{KINDS[0]}'])
                    n = int(use.sum())
                    if n < MIN_PRINT:
                        continue
                    parts = []
                    for kind in KINDS:
                        diff = a[f'{measure}_{kind}'] - b[f'{measure}_{kind}']
                        pt, dr, _ = boot.mean(diff, use)
                        lo, hi = boot.interval(pt, dr)
                        parts.append(f'{fmt(pt)} {ci_text(lo, hi)}')
                        numbers.setdefault('pairs', {})[f'{measure}|{cei}|{name}|{source}|{kind}'] = [pt, lo, hi]
                    lines.append(f'| {cei} | {name} | {source} | {n} | {boot.n_patients_in(use)} | ' + ' | '.join(parts) + ' |')
        lines.append('')


def tertile_edges(x, mask):
    return np.nanpercentile(x[mask & np.isfinite(x)], [100 / 3, 200 / 3])


def bin_labels(feat):
    """Graph-type bins (rule fixed here, before any difference is computed) -> {bin family: (labels array, order)}."""
    families = {}
    nv = feat['num_visits']
    families['history depth (num_visits)'] = (np.where(nv <= 1, '1', np.where(nv <= 3, '2-3', '4+')), ['1', '2-3', '4+'])
    for name, x in (('evidence size (n_evidence tertile)', feat['n_evidence']),
                    ('lab richness (measurement share of evidence, tertile)', feat['avail_measurement'])):
        e = np.percentile(x, [100 / 3, 200 / 3])
        lab = np.where(x <= e[0], 'T1 low', np.where(x <= e[1], 'T2 mid', 'T3 high'))
        families[name] = (lab, ['T1 low', 'T2 mid', 'T3 high'])
    families['class group'] = (feat['class_group'], sorted(set(feat['class_group'])))
    return families


def composition_by_stratum(boot, cells, feat, mask_all, lines, numbers, avail_all):
    lines += ['## C. Composition by class group and graph type', '',
              'Bin rules (fixed in the script before any difference was computed): history depth 1 / 2-3 / 4+; '
              'evidence size and lab richness by tertiles of the 1,345 visits; class groups as listed in '
              '`CLASS_GROUP`. Cells under 30 visits are shown but marked "(n<30, not interpreted)"; under 10 not shown. '
              'The table reports top_share of labs (`measurement`) with the bin\'s availability share next to it, for '
              'CEI (final) vs GSAT and GraphCare, per explainer; the other kinds are in the JSON next to this report.', '']
    families = bin_labels(feat)
    for family, (labels, order) in families.items():
        lines += [f'### {family}', '', '| bin | n visits | n patients | availability (labs) | source | CEI final top_share labs | '
                  'GSAT top_share labs | GraphCare top_share labs | CEI R1 top_share labs |',
                  '|---|---|---|---|---|---|---|---|---|']
        for b in order:
            in_bin = mask_all & (labels == b)
            nb = int(in_bin.sum())
            if nb < MIN_PRINT:
                continue
            flag = '' if nb >= MIN_INTERPRET else ' (n<30, not interpreted)'
            av = boot.mean(feat['avail_measurement'], in_bin)[0]
            for source in SOURCES:
                parts = []
                for key in ('cei_final', 'gsat', 'graphcare', 'cei_R1_ls0'):
                    if key == 'graphcare' and source == 'builtin':
                        parts.append('(tied, n/a)')
                        continue
                    x = cells[(key, source)]['top_share_measurement']
                    use = in_bin & np.isfinite(x)
                    if use.sum() < MIN_PRINT:
                        parts.append('')
                        continue
                    pt, dr, n = boot.mean(x, use)
                    lo, hi = boot.interval(pt, dr)
                    parts.append(f'{pt:.3f} {ci_text(lo, hi)} (n={n})')
                    for kind in KINDS:
                        pk, _, _ = boot.mean(cells[(key, source)][f'top_share_{kind}'], use)
                        numbers.setdefault('strata', {})[f'{family}|{b}|{source}|{key}|{kind}'] = [pk, n]
                lines.append(f'| {b}{flag} | {nb} | {boot.n_patients_in(in_bin)} | {av:.3f} | {source} | '
                             + ' | '.join(parts) + ' |')
        lines.append('')


def relate_to_difference(boot, cells, feat, mask_all, lines, numbers):
    """Section D: composition vs the paired score difference d = CEI - comparator."""
    lines += ['## D. Does composition relate to the paired score difference?', '',
              'd = CEI - comparator on the headline metric `deletion_mean_change_vs_random` (higher = CEI better), paired on '
              'the same visit, seed-averaged. The composition is CEI\'s own top_share of a kind for the same explainer. '
              'For the built-in comparison only GSAT is used (GraphCare\'s built-in is tied); in addition CEI built-in vs '
              'each model\'s best post-hoc explainer (IG for GSAT, Grad for GraphCare) is shown with the CEI built-in '
              'composition.', '',
              '**Raw slope** = coefficient of top_share (per +0.10) in a regression of d on top_share. **Adjusted slope** = '
              'the same coefficient with conf_cei and conf_other (the comparator\'s confidence) as extra regressors '
              '(confidence adjustment (a)). CIs are patient-cluster bootstrap (weighted least squares per draw).', '']
    slope_rows = []
    pairs = []
    for cei in CEI_KEYS:
        for name, other in OTHERS.items():
            for source in SOURCES:
                if name == 'GraphCare' and source == 'builtin':
                    continue
                pairs.append((cei, name, other, source, source, source))
            if name == 'GSAT':
                pass
        pairs.append((cei, 'GSAT (best post-hoc IG)', 'gsat', 'builtin', 'IG', 'builtin'))
        pairs.append((cei, 'GraphCare (best post-hoc Grad)', 'graphcare', 'builtin', 'Grad', 'builtin'))
    lines += ['### D1. Mean d overall (the paired difference the slopes are about)', '',
              '| CEI | comparator | CEI explainer | comparator explainer | n visits | n patients | mean d (95% CI) |',
              '|---|---|---|---|---|---|---|']
    d_cache = {}
    for cei, name, other, cei_source, other_source, _ in pairs:
        d = cells[(cei, cei_source)]['del'] - cells[(other, other_source)]['del']
        d_cache[(cei, name, cei_source)] = d
        use = mask_all & np.isfinite(d)
        if use.sum() < MIN_PRINT:
            continue
        pt, dr, n = boot.mean(d, use)
        lo, hi = boot.interval(pt, dr)
        lines.append(f'| {cei} | {name} | {cei_source} | {other_source} | {n} | {boot.n_patients_in(use)} | '
                     f'{fmt(pt)} {ci_text(lo, hi)} |')
        numbers.setdefault('d_overall', {})[f'{cei}|{name}|{cei_source}'] = [pt, lo, hi, n]
    lines.append('')

    lines += ['### D2. Slope of d on CEI top_share (per +0.10), raw and confidence-adjusted', '',
              '| CEI | comparator | explainer | kind | n visits | n patients | raw slope (95% CI) | adjusted slope (95% CI) |',
              '|---|---|---|---|---|---|---|---|']
    for cei, name, other, cei_source, other_source, _ in pairs:
        d = d_cache[(cei, name, cei_source)]
        conf_c, conf_o = feat['conf_' + cei], feat['conf_' + other]
        for kind in KINDS:
            share = cells[(cei, cei_source)][f'top_share_{kind}']
            use = mask_all & np.isfinite(d) & np.isfinite(share)
            n = int(use.sum())
            if n < MIN_PRINT:
                continue
            raw_b, raw_d, _ = boot.wls(d, [share * 10], use)
            adj_b, adj_d, _ = boot.wls(d, [share * 10, conf_c, conf_o], use)
            rlo, rhi = boot.interval(raw_b[1], raw_d[:, 1])
            alo, ahi = boot.interval(adj_b[1], adj_d[:, 1])
            slope_rows.append({'cei': cei, 'name': name, 'source': cei_source, 'kind': kind, 'n': n,
                               'raw': [float(raw_b[1]), rlo, rhi], 'adj': [float(adj_b[1]), alo, ahi]})
            lines.append(f'| {cei} | {name} | {cei_source} | {kind} | {n} | {boot.n_patients_in(use)} | '
                         f'{fmt(raw_b[1])} {ci_text(rlo, rhi)} | {fmt(adj_b[1])} {ci_text(alo, ahi)} |')
    lines.append('')
    numbers['slopes'] = slope_rows

    # D3: stratified view on labs: composition tertiles, with the confidence-stratified contrast
    lines += ['### D3. Mean d by tertile of CEI top_share of labs (stratified view)', '',
              'Tertiles over the visits with the vector (edges fixed from that distribution). "adjusted high-low" = average '
              'over tertiles of conf_cei - conf_other (confidence adjustment (b)) of the within-tertile high-minus-low '
              'composition contrast (equal weight per confidence tertile, cells under 30 visits dropped from the average). '
              'Rows where the share has fewer than 3 distinct values (all-tied) are skipped.', '',
              '| CEI | comparator | explainer | n visits | n patients | low tertile mean d (n) | mid (n) | high (n) | '
              'raw high-low (95% CI) | adjusted high-low (95% CI) |', '|---|---|---|---|---|---|---|---|---|---|']
    strat_rows = []
    for cei, name, other, cei_source, other_source, _ in pairs:
        d = d_cache[(cei, name, cei_source)]
        share = cells[(cei, cei_source)]['top_share_measurement']
        use = mask_all & np.isfinite(d) & np.isfinite(share)
        if use.sum() < MIN_INTERPRET or np.unique(np.round(share[use], 6)).size < 3:
            continue
        e = tertile_edges(share, use)
        terc = np.where(share <= e[0], 0, np.where(share <= e[1], 1, 2))
        cdiff = feat['conf_' + cei] - feat['conf_' + other]
        ce = tertile_edges(cdiff, use)
        cterc = np.where(cdiff <= ce[0], 0, np.where(cdiff <= ce[1], 1, 2))
        means, draws_t, counts = [], [], []
        for t in range(3):
            pt, dr, n = boot.mean(d, use & (terc == t))
            means.append(pt), draws_t.append(dr), counts.append(n)
        if min(counts) < MIN_PRINT:
            continue
        raw_pt = means[2] - means[0]
        raw_dr = draws_t[2] - draws_t[0]
        adj_pt_parts, adj_dr_parts = [], []
        for c in range(3):
            m_hi, d_hi, n_hi = boot.mean(d, use & (terc == 2) & (cterc == c))
            m_lo, d_lo, n_lo = boot.mean(d, use & (terc == 0) & (cterc == c))
            if n_hi >= MIN_INTERPRET and n_lo >= MIN_INTERPRET:
                adj_pt_parts.append(m_hi - m_lo)
                adj_dr_parts.append(d_hi - d_lo)
        adj_pt = float(np.mean(adj_pt_parts)) if adj_pt_parts else np.nan
        adj_dr = np.mean(adj_dr_parts, axis=0) if adj_dr_parts else np.full(DRAWS, np.nan)
        rlo, rhi = boot.interval(raw_pt, raw_dr)
        alo, ahi = boot.interval(adj_pt, adj_dr)
        strat_rows.append({'cei': cei, 'name': name, 'source': cei_source, 'means': means, 'counts': counts,
                           'raw': [raw_pt, rlo, rhi], 'adj': [adj_pt, alo, ahi], 'edges': [float(e[0]), float(e[1])]})
        flag = '' if min(counts) >= MIN_INTERPRET else ' (a cell n<30)'
        lines.append(f'| {cei} | {name} | {cei_source} | {int(use.sum())} | {boot.n_patients_in(use)} | '
                     f'{fmt(means[0])} ({counts[0]}) | {fmt(means[1])} ({counts[1]}) | {fmt(means[2])} ({counts[2]}) | '
                     f'{fmt(raw_pt)} {ci_text(rlo, rhi)}{flag} | {fmt(adj_pt)} {ci_text(alo, ahi)} |')
    lines.append('')
    numbers['strat'] = strat_rows

    # D4: does d differ across class groups / graph types where composition differs? Own-composition tertile edges per group
    return slope_rows, strat_rows


def supported_summary(slope_rows, strat_rows, lines):
    """Section E: apply the 'supported' rule fixed before the analysis to the slopes and the stratified contrast."""
    lines += ['## E. Which composition-to-d relations pass the "supported" rule?', '',
              'Rule: CI excludes 0 AND the sign repeats for both CEI versions (cei_final and '
              'cei_R1_ls0) and for at least two explainers. Here an explainer counts only if, for BOTH CEI versions, the '
              'raw and the confidence-adjusted slope CIs exclude 0 with one and the same sign. A comparator/kind with '
              '>= 2 such explainers of one sign is "supported"; one with >= 2 raw-only or adjusted-only explainers is '
              'labelled so (a sign that flips between raw and adjusted marks confidence as the driver). Evaluated per '
              'comparator and per kind on the slopes of D2.', '',
              '| comparator | kind | raw: explainers with both CEI versions excluding 0, same sign | '
              'adjusted: same | counts as supported (both raw and adjusted, same sign) | verdict |',
              '|---|---|---|---|---|---|']

    def side(got, tag):
        signs = []
        for cei in CEI_KEYS:
            pt, lo, hi = got[cei][tag]
            signs.append(int(np.sign(pt)) if excludes_zero(lo, hi) else 0)
        return signs[0] if signs[0] != 0 and signs[0] == signs[1] else 0

    out = []
    for name in sorted({r['name'] for r in slope_rows}):
        for kind in KINDS:
            raw_ok, adj_ok, both_ok = [], [], []
            for source in SOURCES:
                got = {r['cei']: r for r in slope_rows
                       if r['name'] == name and r['kind'] == kind and r['source'] == source}
                if set(got) != set(CEI_KEYS):
                    continue
                raw, adj = side(got, 'raw'), side(got, 'adj')
                tag = lambda s: '+' if s > 0 else '-'
                if raw:
                    raw_ok.append(f'{source}({tag(raw)})')
                if adj:
                    adj_ok.append(f'{source}({tag(adj)})')
                if raw and raw == adj:
                    both_ok.append((source, raw))
            if len(both_ok) >= 2 and len({s for _, s in both_ok}) == 1:
                verdict = 'supported'
            elif len(adj_ok) >= 2 and len(raw_ok) >= 2 and not both_ok:
                verdict = 'sign flips or changes with confidence adjustment: not supported'
            elif len(adj_ok) >= 2:
                verdict = 'adjusted only: exploratory'
            elif len(raw_ok) >= 2:
                verdict = 'raw only: exploratory (not after confidence adjustment)'
            else:
                verdict = 'not supported'
            out.append((name, kind, raw_ok, adj_ok, [f'{a}({"+" if s > 0 else "-"})' for a, s in both_ok], verdict))
            lines.append(f'| {name} | {kind} | {", ".join(raw_ok) or "none"} | {", ".join(adj_ok) or "none"} | '
                         f'{", ".join(f"{a}({chr(43) if s > 0 else chr(45)})" for a, s in both_ok) or "none"} | {verdict} |')
    lines.append('')
    return out


# --------------------------------------------------------------------------- report
def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    feat = load_features()
    n_visits = len(feat['patient'])
    say(f'features: {n_visits} visits, {feat["patient"].max() + 1} patients')
    cells, _cols = load_cells(n_visits)
    boot = Boot(feat['patient'])
    mask_all = np.ones(n_visits, dtype=bool)
    for key in MODELS:
        for source in SOURCES:
            c = cells[(key, source)]
            say(f'  {key:11s} {source:8s} visits with del {int(np.isfinite(c["del"]).sum()):5d}, with composition '
                f'{int(np.isfinite(c["top_share_measurement"]).sum()):5d}')
    numbers = {}
    lines = []
    avail = composition_overall(boot, cells, feat, mask_all, lines, numbers)
    composition_pairs(boot, cells, mask_all, lines, numbers)
    composition_by_stratum(boot, cells, feat, mask_all, lines, numbers, avail)
    slope_rows, strat_rows = relate_to_difference(boot, cells, feat, mask_all, lines, numbers)
    verdicts = supported_summary(slope_rows, strat_rows, lines)
    numbers['verdicts'] = [list(v) for v in verdicts]
    (SCRATCH / 'step4_body.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (SCRATCH / 'step4_numbers.json').write_text(json.dumps(numbers, default=float), encoding='utf-8')
    say(f'body written: {len(lines)} lines, runtime {time.time() - STARTED:.0f} s')
    command = ('`env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 comparison/explain/graph_types/step4_content.py` '
               '(repo root; reads data/*.csv, writes scratch/step4/step4_body.md, step4_numbers.json and out/step4-report.md).')
    report = '# Step 4: explanation content axis\n\n' + '\n'.join(lines) + '\n\n## Command\n\n' + command + '\n'
    (OUT / 'step4-report.md').write_text(report, encoding='utf-8')
    say('out/step4-report.md written')
    return 0


if __name__ == '__main__':
    sys.exit(main())
