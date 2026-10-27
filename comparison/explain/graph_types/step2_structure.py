"""Graph-type analysis, step 2: graph-structure axis.

Run from the repo root:  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 comparison/explain/graph_types/step2_structure.py

Reads data/visit_features.csv and data/visit_scores.csv (step 1; read only inside this script) and writes
out/step2-strata.csv (aggregate cells with >= 10 visits) and out/step2-report.md (generated tables).
Paths: _paths.py in this folder (W = working folder with data/ and out/; data/ holds per-visit tables with anonymous indices and is never committed).
Stdout carries aggregates only; no row, id or raw value is printed.

BIN RULE (fixed here, before any difference is computed; it depends only on the feature's own distribution over the
1,345 visits):
  * a feature with at most 6 distinct values gets one bin per value (num_visits 1..6, small counts);
  * otherwise the bins are cut at the lower quartile values q25 <= q50 <= q75 (np.sort index floor(q*(n-1)), duplicates
    merged, so features with many zeros get a "0" bin): (-inf, e1], (e1, e2], ..., (e_last, inf); empty bins dropped.
  * features: n_nodes, n_evidence, n_edges, density, num_visits, count and share (of n_evidence) of complaint /
    measurement / vital / diagnosis nodes, knowledge and analyte node counts, n_prior_dx, relation counts
    trajectory_of, recurrence_of, comorbid_with, co_complaint, baseline_of, medical_* (total and each relation).
CELL RULE: a cell under 10 visits is never printed; a cell under 30 visits is printed but never interpreted (marked +).

MEASURE: per visit, mean over the seeds that are ok (>= 2 seeds) of deletion_mean_change_vs_random ("del", headline).
PAIRED difference d = CEI source - other model's source on the same visit, CEI = cei_final and cei_R1_ls0 separately.
CONFIDENCE ADJUSTMENT: regression adjustment. d is regressed (weighted OLS, refit in every bootstrap draw) on dummies of
the quintile bins of the CEI model's confidence and of the other model's confidence (conf_<key>, edges fixed from all
visits); the adjusted d_i = d_i - fitted confidence part_i + its weighted mean, so the adjusted overall mean equals the
raw overall mean and a stratum mean shows what is left once confidence differences are removed (additively).
CI: patient-cluster bootstrap, 2,000 draws, seed 20261025, 95% percentile.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from comparison.explain.graph_types._paths import WORK  # noqa: E402
W = WORK
DATA, OUT, SCRATCH = W / 'data', W / 'out', W / 'scratch' / 'step2'
B, BOOT_SEED = 2000, 20261025
MIN_PRINT, MIN_INTERPRET = 10, 30
HEADLINE = 'deletion_mean_change_vs_random'
VERSIONS = ('cei_final', 'cei_R1_ls0')
SOURCES = ('IG', 'Grad', 'GNNExp', 'builtin')
MEDICAL = ('member_of', 'assesses', 'measures', 'is_a', 'may_affect', 'interacts_with', 'indicates')
STARTED = time.time()


def say(text=''):
    print(text, flush=True)


# --------------------------------------------------------------------------- inputs
def load_features():
    with open(DATA / 'visit_features.csv', newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    cols = {}
    for name in rows[0]:
        if name == 'true_class':
            continue
        cols[name] = np.array([float(r[name]) for r in rows])
    assert (cols['visit_idx'] == np.arange(len(rows))).all()
    return cols


def load_scores(n):
    """{(key, source): per-visit mean of the headline over the ok seeds (nan when ok in < 2 seeds)} and the counts of
    visits dropped for the keys used."""
    keep = {'gsat', 'graphcare', *VERSIONS}
    sums, cnts = {}, {}
    with open(DATA / 'visit_scores.csv', newline='', encoding='utf-8') as handle:
        for r in csv.DictReader(handle):
            if r['key'] not in keep or r['ok'] != '1' or r[HEADLINE] == '':
                continue
            k = (r['key'], r['source'])
            if k not in sums:
                sums[k], cnts[k] = np.zeros(n), np.zeros(n)
            i = int(r['visit_idx'])
            sums[k][i] += float(r[HEADLINE])
            cnts[k][i] += 1
    values, dropped = {}, {}
    for k in sums:
        ok = cnts[k] >= 2
        values[k] = np.where(ok, sums[k] / np.maximum(cnts[k], 1), np.nan)
        dropped[k] = int((~ok).sum())
    return values, dropped


# --------------------------------------------------------------------------- strata
def fmtv(x):
    return str(int(x)) if float(x).is_integer() else f'{x:.3g}'


def bins_of(v):
    """[(label, mask)] following the bin rule in the module docstring."""
    distinct = np.unique(v)
    if distinct.size <= 1:
        return []
    if distinct.size <= 6:
        return [(fmtv(x), v == x) for x in distinct]
    s = np.sort(v)
    edges = sorted({float(s[int(np.floor(q * (len(s) - 1)))]) for q in (0.25, 0.5, 0.75)})
    out, prev = [], -np.inf
    for i, e in enumerate(edges):
        mask = (v > prev) & (v <= e)
        out.append((f'<={fmtv(e)}' if i == 0 else f'({fmtv(prev)},{fmtv(e)}]', mask))
        prev = e
    out.append((f'>{fmtv(prev)}', v > prev))
    return [(label, m) for label, m in out if m.any()]


def feature_list(F):
    feats = []
    for name in ('n_nodes', 'n_evidence', 'n_edges', 'density', 'num_visits'):
        feats.append((name, F[name]))
    for kind in ('complaint', 'measurement', 'vital', 'diagnosis'):
        feats.append((f'n_{kind}', F[f'n_kind_{kind}']))
    for kind in ('complaint', 'measurement', 'vital', 'diagnosis'):
        feats.append((f'share_{kind}', F[f'n_kind_{kind}'] / np.maximum(F['n_evidence'], 1)))
    feats += [('n_knowledge', F['n_kind_knowledge']), ('n_analyte', F['n_kind_analyte']), ('n_prior_dx', F['n_prior_dx'])]
    for rel in ('trajectory_of', 'recurrence_of', 'comorbid_with', 'co_complaint', 'baseline_of'):
        feats.append((f'rel_{rel}', F[f'n_rel_{rel}']))
    feats.append(('rel_medical_total', sum(F[f'n_rel_medical_{m}'] for m in MEDICAL)))
    for m in MEDICAL:
        feats.append((f'rel_medical_{m}', F[f'n_rel_medical_{m}']))
    return feats


def build_strata(F):
    """Strata columns: index 0 = all visits, then (feature, label) in order. Returns (M, meta, skipped)."""
    meta, cols, skipped = [('All visits', 'all')], [np.ones(len(F['visit_idx']), bool)], []
    for name, v in feature_list(F):
        bins = bins_of(np.asarray(v, float))
        if not bins:
            skipped.append(name)
        for label, mask in bins:
            meta.append((name, label))
            cols.append(mask)
    return np.stack(cols, axis=1).astype(float), meta, skipped


# --------------------------------------------------------------------------- comparisons
def comparisons():
    out = []
    for ver in VERSIONS:
        for src in ('IG', 'Grad', 'GNNExp'):
            out.append(dict(ver=ver, code=f'{src}/GSAT', family='GSAT', expl=src, cei=(ver, src), other=('gsat', src)))
            out.append(dict(ver=ver, code=f'{src}/GC', family='GraphCare', expl=src, cei=(ver, src),
                            other=('graphcare', src)))
        out.append(dict(ver=ver, code='bi/GSATbi', family='GSAT', expl='builtin', cei=(ver, 'builtin'),
                        other=('gsat', 'builtin')))
        out.append(dict(ver=ver, code='bi/GSATIG', family='GSAT', expl='builtin', cei=(ver, 'builtin'),
                        other=('gsat', 'IG')))
        out.append(dict(ver=ver, code='bi/GCGrad', family='GraphCare', expl='builtin', cei=(ver, 'builtin'),
                        other=('graphcare', 'Grad')))
    return out


LEGEND = ('Column codes: `X/GSAT` = CEI source X minus GSAT source X; `X/GC` = CEI source X minus GraphCare source X '
          '(X = IG, Grad, GNNExp); `bi/GSATbi` = CEI built-in minus GSAT built-in; `bi/GSATIG` = CEI built-in minus '
          'GSAT IG; `bi/GCGrad` = CEI built-in minus GraphCare Grad (GraphCare built-in is not usable). Positive = CEI '
          'is better (headline del = deletion_mean_change_vs_random).')


def conf_dummies(conf):
    s = np.sort(conf)
    edges = sorted({float(s[int(np.floor(q * (len(s) - 1)))]) for q in (0.2, 0.4, 0.6, 0.8)})
    cols = [(conf > e).astype(float) for e in edges]       # nested indicators: conf above each quintile edge
    return cols


def design(F, cmp):
    cols = [np.ones(len(F['visit_idx']))]
    cols += conf_dummies(F['conf_' + cmp['cei'][0]])
    cols += conf_dummies(F['conf_' + cmp['other'][0]])
    return np.stack(cols, axis=1)


def stratified(d, valid, X, M, Wvis):
    """Weighted stratum means of d (raw and confidence-adjusted) and their contrast with the rest, per weight row."""
    Wv = Wvis * valid
    d0 = np.where(valid, d, 0.0)
    den = Wv @ M
    p = X.shape[1]
    XX = (X[:, :, None] * X[:, None, :]).reshape(len(X), p * p)
    A = (Wv @ XX).reshape(-1, p, p) + 1e-9 * np.eye(p)
    beta = np.linalg.solve(A, ((Wv * d0) @ X)[:, :, None])[:, :, 0]
    fit = beta[:, 1:] @ X[:, 1:].T
    adj = d0 - fit + ((Wv * fit).sum(1) / Wv.sum(1))[:, None]
    res = {}
    with np.errstate(divide='ignore', invalid='ignore'):
        for tag, vals in (('raw', d0), ('adj', adj)):
            num = (Wv * vals) @ M
            mean = num / den
            rest = (num[:, [0]] - num) / (den[:, [0]] - den)
            contrast = mean - rest
            contrast[:, 0] = np.nan
            res[tag], res[tag + '_contrast'] = mean, contrast
    return res


def ci(draws):
    with np.errstate(all='ignore'):
        return np.nanpercentile(draws, 2.5, axis=0), np.nanpercentile(draws, 97.5, axis=0)


def run_comparison(F, V, cmp, M, patient_weights):
    d = V[cmp['cei']] - V[cmp['other']]
    valid = ~np.isnan(d)
    X = design(F, cmp)
    point = stratified(d, valid, X, M, np.ones((1, len(d))))
    boot = stratified(d, valid, X, M, patient_weights)
    out = {'n': (M * valid[:, None]).sum(0).astype(int), 'dropped': int((~valid).sum())}
    pidx = F['patient_idx'].astype(int)
    out['npat'] = np.array([np.unique(pidx[(M[:, s] > 0) & valid]).size for s in range(M.shape[1])])
    for tag in ('raw', 'adj', 'raw_contrast', 'adj_contrast'):
        lo, hi = ci(boot[tag])
        out[tag] = (point[tag][0], lo, hi)
    return out


# --------------------------------------------------------------------------- formatting
def sig(lo, hi):
    return bool(lo > 0 or hi < 0)


def cell(res, tag, s):
    n = res['n'][s]
    if n < MIN_PRINT:
        return 'n<10'
    m, lo, hi = (a[s] for a in res[tag])
    if np.isnan(m):
        return ''
    star = '*' if sig(lo, hi) else ''
    mark = '+' if n < MIN_INTERPRET else ''
    return f'{m:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}{mark}'


def small(n):
    return '<10' if 0 < n < 10 else str(n)


def monotone(vals):
    v = [x for x in vals if not np.isnan(x)]
    if len(v) < 3:
        return False
    diffs = np.diff(v)
    return bool((diffs > 0).all() or (diffs < 0).all())


# --------------------------------------------------------------------------- main
def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    F = load_features()
    n = len(F['visit_idx'])
    n_pat = int(F['patient_idx'].max()) + 1
    V, dropped = load_scores(n)
    M, meta, skipped = build_strata(F)
    say(f'visits {n}, patients {n_pat}, strata {M.shape[1]} (incl. all), constant features skipped: {skipped}')
    rng = np.random.default_rng(BOOT_SEED)
    counts = np.stack([np.bincount(rng.integers(0, n_pat, n_pat), minlength=n_pat) for _ in range(B)])
    patient_weights = counts[:, F['patient_idx'].astype(int)].astype(float)
    comps = comparisons()
    results = []
    for c in comps:
        results.append(run_comparison(F, V, c, M, patient_weights))
    say(f'bootstrap done: {len(comps)} comparisons x {B} draws, {time.time() - STARTED:.0f} s')

    features = list(dict.fromkeys(m[0] for m in meta[1:]))
    strata_of = {f: [s for s in range(1, len(meta)) if meta[s][0] == f] for f in features}

    # ---- CSV of every printable cell
    fields = ['feature', 'bin', 'version', 'comparison', 'n_visits', 'n_patients', 'raw_mean', 'raw_lo', 'raw_hi',
              'adj_mean', 'adj_lo', 'adj_hi', 'raw_contrast', 'raw_contrast_lo', 'raw_contrast_hi', 'adj_contrast',
              'adj_contrast_lo', 'adj_contrast_hi']
    with open(OUT / 'step2-strata.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for c, res in zip(comps, results):
            for s in range(len(meta)):
                if res['n'][s] < MIN_PRINT:
                    continue
                row = [meta[s][0], meta[s][1], c['ver'], c['code'], res['n'][s], res['npat'][s]]
                for tag in ('raw', 'adj', 'raw_contrast', 'adj_contrast'):
                    row += [f'{a[s]:.6f}' for a in res[tag]]
                writer.writerow(row)

    # ---- supported-stratum rule
    mono = {}
    for ci_, (c, res) in enumerate(zip(comps, results)):
        for f in features:
            for tag in ('raw', 'adj'):
                means = [res[tag][0][s] if res['n'][s] >= MIN_INTERPRET else np.nan for s in strata_of[f]]
                mono[(ci_, f, tag)] = monotone(means)
    supported = []                                   # (tag, feature, bin, family, sign, why)
    for tag in ('raw', 'adj'):
        for f in features:
            for s in strata_of[f]:
                for family in ('GSAT', 'GraphCare'):
                    for sign in (1, -1):
                        hits = []
                        for ci_, c in enumerate(comps):
                            if c['family'] != family or results[ci_]['n'][s] < MIN_INTERPRET:
                                continue
                            m, lo, hi = (a[s] for a in results[ci_][tag])
                            if sig(lo, hi) and np.sign(m) == sign:
                                hits.append(ci_)
                        if not hits:
                            continue
                        versions = {comps[h]['ver'] for h in hits}
                        expls = {comps[h]['expl'] for h in hits}
                        repeat = len(versions) == 2 and len(expls) >= 2
                        mono_hits = [comps[h]['code'] + '/' + comps[h]['ver'][4:] for h in hits if mono[(h, f, tag)]]
                        if repeat or mono_hits:
                            why = ('repeats(' + ','.join(sorted(expls)) + ')' if repeat else '') + \
                                  (' mono:' + ','.join(mono_hits) if mono_hits else '')
                            also = 0
                            for h in hits:
                                cm, clo, chi = (a[s] for a in results[h][tag + '_contrast'])
                                also += int(sig(clo, chi) and np.sign(cm) == sign)
                            supported.append((tag, f, meta[s][1], family, sign, why.strip(), len(hits), also))

    # ---- report text
    lines = []
    L = lines.append
    L('## 1. Setup')
    L('')
    L(f'Unit: the visit ({n} visits, {n_pat} patients, test fold, K=20, primary_dx_v4, fair_v10t). Measure: per visit, '
      'the mean over the seeds that are ok (a visit counts when ok in at least 2 of the 3 seeds) of '
      '`deletion_mean_change_vs_random` ("del"). Paired differences on the same visit; CEI = cei_final and, '
      'separately, cei_R1_ls0. CI: patient-cluster bootstrap (resample patients with replacement, '
      f'{B} draws, seed {BOOT_SEED}), 95% percentile. ' + LEGEND)
    L('')
    L('Cell format: `mean [95% CI]`; `*` = CI excludes 0; `+` = cell has fewer than 30 visits (printed, not interpreted); '
      '`n<10` = not printed. n visits and n patients are the cell sizes of the first column (GSAT IG comparison); the '
      'other columns differ only by the visits dropped below.')
    L('')
    L('**Raw** = plain paired difference. **Adjusted** = confidence-adjusted by regression: d is regressed on '
      'indicator bins of the CEI model\'s confidence and of the other model\'s confidence (quintile edges from all '
      'visits), refit in every bootstrap draw; the adjusted stratum mean is the mean of d minus the fitted confidence '
      'part (re-centred, so the all-visit mean is unchanged). It removes the additive part of the confidence '
      'difference, not any interaction.')
    L('')
    L('**Bin rule (fixed in the script before any difference was computed)**: features with at most 6 distinct values '
      'get one bin per value; otherwise bins are cut at the lower quartile values (q25, q50, q75 of the feature over '
      'the 1,345 visits, duplicates merged, so a feature with many zeros has a `0` bin). Features: n_nodes, '
      'n_evidence, n_edges, density, num_visits, counts and shares (of evidence nodes) of complaint / measurement / '
      'vital / diagnosis nodes, knowledge and analyte node counts, n_prior_dx, relation counts trajectory_of, '
      'recurrence_of, comorbid_with, co_complaint, baseline_of, and medical_* (total and each of the 7 relations). '
      + (f'Constant features skipped: {", ".join(skipped)}.' if skipped else 'No feature is constant.'))
    L('')
    L('**Visits dropped (ok in fewer than 2 of 3 seeds) per model and source**: ' +
      '; '.join(f'{k[0]} {k[1]}: {small(v)}' for k, v in sorted(dropped.items()) if v) + '.' if any(dropped.values())
      else '**Visits dropped (ok in fewer than 2 of 3 seeds)**: none for gsat, graphcare, cei_final, cei_R1_ls0.')
    L('')

    L('## 2. All visits (paired difference, CEI minus other)')
    L('')
    for ver in VERSIONS:
        L(f'**{ver}**')
        L('')
        L('| comparison | n visits | n patients | mean difference [95% CI] |')
        L('|---|---|---|---|')
        for ci_, c in enumerate(comps):
            if c['ver'] != ver:
                continue
            r = results[ci_]
            L(f'| {c["code"]} | {r["n"][0]} | {r["npat"][0]} | {cell(r, "raw", 0)} |')
        L('')

    # ---- overview
    L('## 3. Feature overview (all strata looked at)')
    L('')
    L('Per feature: number of bins; cells = (bin with at least 30 visits) x (18 comparisons: 9 per CEI version); '
      '"d excludes 0" counts cells whose CI of d excludes 0 (raw / adjusted; d is negative nearly everywhere because '
      'CEI is behind on average, so this counts mostly "CEI worse" cells); "d>0 sig." counts cells where CEI is '
      'significantly better; "contrast" counts cells where the stratum differs from the rest of the visits (CI of '
      'stratum mean minus rest mean excludes 0), raw / adjusted; "contrast >0" counts the subset where the stratum is '
      'better for CEI than the rest.')
    L('')
    L('| feature | bins | cells | d excludes 0 (raw/adj) | d>0 sig. (raw/adj) | contrast (raw/adj) | contrast >0 (raw/adj) |')
    L('|---|---|---|---|---|---|---|')
    overview = {}
    for f in features:
        cells = 0
        tallies = {k: [0, 0] for k in ('sig', 'pos', 'con', 'conpos')}
        for ci_, res in enumerate(results):
            for s in strata_of[f]:
                if res['n'][s] < MIN_INTERPRET:
                    continue
                cells += 1
                for j, tag in enumerate(('raw', 'adj')):
                    m, lo, hi = (a[s] for a in res[tag])
                    if sig(lo, hi):
                        tallies['sig'][j] += 1
                        tallies['pos'][j] += int(m > 0)
                    cm, clo, chi = (a[s] for a in res[tag + '_contrast'])
                    if sig(clo, chi):
                        tallies['con'][j] += 1
                        tallies['conpos'][j] += int(cm > 0)
        overview[f] = (cells, tallies)
        L(f'| {f} | {len(strata_of[f])} | {cells} | {tallies["sig"][0]}/{tallies["sig"][1]} | '
          f'{tallies["pos"][0]}/{tallies["pos"][1]} | {tallies["con"][0]}/{tallies["con"][1]} | '
          f'{tallies["conpos"][0]}/{tallies["conpos"][1]} |')
    L('')

    # ---- supported strata
    L('## 4. Supported strata (rule fixed before the analysis)')
    L('')
    L('A stratum effect is "supported" when its CI excludes 0 (cell of at least 30 visits) AND either the means of the '
      'feature\'s interpretable bins are strictly monotone across at least 3 bins for that comparison (`mono:`) OR the '
      'same-sign significant effect repeats for both CEI versions and at least two explainers within the same '
      'comparison model (`repeats(...)`). Every (feature, bin, comparison model, sign) that passes is listed; '
      'sign + means CEI better, - means CEI worse.')
    L('')
    for sign, title in ((1, 'CEI better than the other model (d > 0)'), (-1, 'CEI worse than the other model (d < 0)')):
        L(f'### 4.{1 if sign == 1 else 2} {title}')
        L('')
        rows = [x for x in supported if x[4] == sign]
        if not rows:
            L('None.')
            L('')
            continue
        L('| basis | feature | bin | vs | #sig comparisons | of which also differ from the rest | why |')
        L('|---|---|---|---|---|---|---|')
        for tag, f, b, family, _sg, why, k, also in rows:
            L(f'| {tag} | {f} | {b} | {family} | {k} | {also} | {why} |')
        L('')

    # ---- heterogeneity
    L('## 5. Strata that differ from the rest of the visits')
    L('')
    L('The contrast is stratum mean minus the mean of all other visits, bootstrapped like d. Listed: strata whose '
      'contrast CI excludes 0 in the same direction in both the raw and the adjusted analysis for at least 3 of the 18 '
      'comparisons (cells of at least 30 visits). This tests whether the stratum behaves differently from the average '
      'visit, which is the question behind "where does CEI-GNN explain better".')
    L('')
    L('| feature | bin | direction | raw & adj significant in (comparisons) |')
    L('|---|---|---|---|')
    hetero_n = 0
    for f in features:
        for s in strata_of[f]:
            for sign in (1, -1):
                hits = []
                for ci_, c in enumerate(comps):
                    res = results[ci_]
                    if res['n'][s] < MIN_INTERPRET:
                        continue
                    ok = True
                    for tag in ('raw_contrast', 'adj_contrast'):
                        m, lo, hi = (a[s] for a in res[tag])
                        ok = ok and sig(lo, hi) and np.sign(m) == sign
                    if ok:
                        hits.append(c['code'] + '/' + c['ver'][4:])
                if len(hits) >= 3:
                    hetero_n += 1
                    L(f'| {f} | {meta[s][1]} | {"CEI relatively better" if sign == 1 else "CEI relatively worse"} | '
                      f'{len(hits)}: {", ".join(hits)} |')
    if not hetero_n:
        L('| (none) | | | |')
    L('')

    # ---- per-feature tables
    L('## 6. Every stratum: paired difference per feature')
    L('')
    L('Rows are the bins of the feature; n and n patients refer to the visits in the bin (GSAT IG comparison). '
      'Sections 6.1-6.4: raw and confidence-adjusted, cei_final and cei_R1_ls0.')
    L('')
    sec = 0
    for ver in VERSIONS:
        for tag in ('raw', 'adj'):
            sec += 1
            L(f'### 6.{sec} {ver}, {"raw" if tag == "raw" else "confidence-adjusted"}')
            L('')
            idx = [i for i, c in enumerate(comps) if c['ver'] == ver]
            head = ['feature', 'bin', 'n visits', 'n patients'] + [comps[i]['code'] for i in idx]
            for f in features:
                L(f'**{f}**')
                L('')
                L('| ' + ' | '.join(head[1:]) + ' |')
                L('|' + '---|' * (len(head) - 1))
                for s in strata_of[f]:
                    r0 = results[idx[0]]
                    if r0['n'][s] < MIN_PRINT:
                        L(f'| {meta[s][1]} | n<10 | | ' + ' | '.join('' for _ in idx) + ' |')
                        continue
                    L(f'| {meta[s][1]} | {r0["n"][s]} | {r0["npat"][s]} | ' +
                      ' | '.join(cell(results[i], tag, s) for i in idx) + ' |')
                L('')

    L('## 7. Reproduce')
    L('')
    L('`env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 comparison/explain/graph_types/step2_structure.py` '
      '(runtime about ' + f'{time.time() - STARTED:.0f}' + ' s). Writes `out/step2-strata.csv` (every printable cell with '
      'mean, CI, contrast) and this report.')
    L('')

    body = '\n'.join(lines)
    (OUT / 'step2-report.md').write_text('# Step 2: graph-structure axis\n\n' + body + '\n', encoding='utf-8')

    # ---- stdout digest (aggregates only)
    say('')
    say('ALL VISITS')
    for ci_, c in enumerate(comps):
        r = results[ci_]
        say(f'  {c["ver"]:10s} {c["code"]:10s} n={r["n"][0]} pat={r["npat"][0]} raw {cell(r, "raw", 0)}')
    say('OVERVIEW (cells, sig raw/adj, pos raw/adj, contrast raw/adj, contrast>0 raw/adj)')
    for f, (cells, t) in overview.items():
        say(f'  {f:22s} bins {len(strata_of[f])} cells {cells:3d} sig {t["sig"]} pos {t["pos"]} con {t["con"]} conpos {t["conpos"]}')
    say(f'supported rows: +{sum(1 for x in supported if x[4] == 1)} / -{sum(1 for x in supported if x[4] == -1)}; '
        f'heterogeneity rows: {hetero_n}')
    for x in supported:
        if x[4] == 1:
            say(f'  SUPPORTED+ {x[0]} {x[1]} {x[2]} vs {x[3]} k={x[6]} {x[5]}')
    say(f'runtime {time.time() - STARTED:.0f} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
