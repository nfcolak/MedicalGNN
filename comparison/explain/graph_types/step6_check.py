"""Graph-type analysis, step 6: independent re-check of the numbers that steps 2-5 report and of the places where their
readings seem to disagree. Aggregates only (every printed cell holds >= 122 visits; no rows, ids or raw values).

Reads W/data/visit_features.csv and W/data/visit_scores.csv only inside this script. Writes W/out/step6-check.md.

Run from the repo root:
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 comparison/explain/graph_types/step6_check.py
Paths: _paths.py in this folder (W = working folder with data/ and out/; data/ holds per-visit tables with anonymous indices and is never committed).

Headline metric `del` = deletion_mean_change_vs_random, seed-averaged per visit (a visit counts for a key/source when
ok in >= 2 seeds). d = CEI - partner, paired on the visit. CIs: patient-cluster bootstrap, 2,000 draws, seed 20261025.
Test fold, descriptive: nothing is selected from these numbers.

Checks
  1. Overall d and label-smoothing cost against the values quoted in step 2 / step 5.
  2. Mean del per model and explainer (is GraphCare's GNNExplainer the weak one?).
  3. d at EQUAL confidence (regression intercept) and the confidence slope: how much of each gap is confidence.
  4. Contradiction A (steps 2, 3 and 5): size and history contrasts under three confidence adjustments (step 5 linear on the
     confidence difference, step 2 additive quintile bins of both confidences, joint model with size and history together).
  5. Strata scan over the step 5 strata (29): how many have CI > 0 / < 0 per comparison, raw and adjusted.
  6. Class groups: raw and adjusted d for IG, Grad, built-in.
  7. Contradiction B (step 2 vs step 5): built-in gap (CEI built-in vs GSAT built-in) by size tertile, raw and adjusted.
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
OUT = W / "out" / "step6-check.md"
SEED = 20261025
NB = 2000
METRIC = "deletion_mean_change_vs_random"
SOURCES = ["IG", "Grad", "GNNExp", "builtin"]
KEYS = ["gsat", "graphcare", "cei_final", "cei_R1_ls0"]
T0 = time.time()

# ------------------------------------------------------------------ features (bin rules fixed here, before any d)
frows = list(csv.DictReader(open(W / "data" / "visit_features.csv", encoding="utf-8")))
N = len(frows)


def fcol(name):
    return np.array([float(r[name]) for r in frows])


assert (fcol("visit_idx").astype(int) == np.arange(N)).all()
pat = fcol("patient_idx").astype(int)
P = int(pat.max()) + 1
n_nodes = fcol("n_nodes")
n_evid = fcol("n_evidence")
density = fcol("density")
num_visits = fcol("num_visits")
n_prior = fcol("n_prior_dx")
share_meas = fcol("n_kind_measurement") / n_evid
n_comp = fcol("n_kind_complaint")
n_diag = fcol("n_kind_diagnosis")
true_class = [r["true_class"] for r in frows]
CLASS_GROUPS = {
    "Cardiovascular": ["Heart failure", "Non-ST elevation (NSTEMI) myocardial infarction", "Unspecified atrial fibrillation"],
    "Infection and respiratory": ["Sepsis", "UTI or pyelonephritis", "Skin or soft-tissue infection",
                                  "Lower respiratory disease", "Acute pharyngitis"],
    "Metabolic renal blood": ["Acute kidney failure", "Diabetes mellitus", "Hypokalemia", "Anemia"],
    "Gastrointestinal": ["GI bleed", "Unsp intestnl obst"],
    "Injury and pain": ["Back or spine pain", "Limb injury or pain", "Head injury"],
    "Neuro psychiatric substance": ["Epilepsy", "Major depressive disorder", "Alcohol abuse with intoxication"],
}
cls2grp = {c: g for g, cs in CLASS_GROUPS.items() for c in cs}
assert all(c in cls2grp for c in true_class), "unmapped class"
group_of = np.array([cls2grp[c] for c in true_class])


def tertile_masks(name, v):
    e1, e2 = np.quantile(v, [1 / 3, 2 / 3])
    return [(f"{name} <= {e1:g}", v <= e1), (f"{name} mid", (v > e1) & (v <= e2)), (f"{name} > {e2:g}", v > e2)]


STRATA = (tertile_masks("n_nodes", n_nodes) + tertile_masks("n_evidence", n_evid) + tertile_masks("density", density)
          + [("num_visits = 1", num_visits == 1), ("num_visits = 2", num_visits == 2), ("num_visits >= 3", num_visits >= 3),
             ("n_prior_dx = 0", n_prior == 0), ("n_prior_dx 1..3", (n_prior >= 1) & (n_prior <= 3)),
             ("n_prior_dx >= 4", n_prior >= 4)]
          + tertile_masks("measurement_share", share_meas)
          + [("complaint nodes = 0", n_comp == 0), ("complaint nodes = 1", n_comp == 1), ("complaint nodes >= 2", n_comp >= 2),
             ("diagnosis nodes = 0", n_diag == 0), ("diagnosis nodes >= 1", n_diag >= 1)]
          + [(f"class: {g}", group_of == g) for g in CLASS_GROUPS])
STRATA = [(l, m) for l, m in STRATA if m.sum() > 0]
assert min(int(m.sum()) for _, m in STRATA) >= 30
M_ALL = np.array([np.ones(N, bool)] + [m for _, m in STRATA])
S_NAMES = ["all visits"] + [l for l, _ in STRATA]
CLASS_ROWS = [i for i, l in enumerate(S_NAMES) if l.startswith("class: ")]

# ------------------------------------------------------------------ scores
sums = {(k, s): np.zeros(N) for k in KEYS for s in SOURCES}
cnts = {(k, s): np.zeros(N, int) for k in KEYS for s in SOURCES}
with open(W / "data" / "visit_scores.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        k = r["key"]
        if k not in KEYS or r["ok"] not in ("1", "1.0", "True", "true") or r[METRIC] == "":
            continue
        i = int(r["visit_idx"])
        sums[(k, r["source"])][i] += float(r[METRIC])
        cnts[(k, r["source"])][i] += 1
V = {ks: np.where(cnts[ks] >= 2, sums[ks] / np.maximum(cnts[ks], 1), np.nan) for ks in sums}
CONF = {k: fcol(f"conf_{k}") for k in KEYS}
print("loaded in %.0fs" % (time.time() - T0))

rng = np.random.default_rng(SEED)
Wp = rng.multinomial(P, np.full(P, 1.0 / P), size=NB).astype(float)
Wv = Wp[:, pat]  # (NB, N)

CMPS = [("gsat_IG", "IG", "gsat", "IG"), ("gsat_Grad", "Grad", "gsat", "Grad"),
        ("gsat_GNNExp", "GNNExp", "gsat", "GNNExp"), ("gsat_builtin", "builtin", "gsat", "builtin"),
        ("gsat_builtin_vs_IG", "builtin", "gsat", "IG"),
        ("gc_IG", "IG", "graphcare", "IG"), ("gc_Grad", "Grad", "graphcare", "Grad"),
        ("gc_GNNExp", "GNNExp", "graphcare", "GNNExp"), ("gc_builtin_vs_Grad", "builtin", "graphcare", "Grad")]
CEI_KEYS = {"final": "cei_final", "R1": "cei_R1_ls0"}
TGT = {}  # (ver, cname) -> (d, conf_cei, conf_partner)
for cname, cs, pk, ps in CMPS:
    for ver, ck in CEI_KEYS.items():
        TGT[(ver, cname)] = (V[(ck, cs)] - V[(pk, ps)], CONF[ck], CONF[pk])


# ------------------------------------------------------------------ helpers
def pct(boot):
    return np.percentile(boot, [2.5, 97.5], axis=0)


def f(pt, lo, hi):
    star = "*" if (lo > 0 or hi < 0) else ""
    return f"{pt:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}"


def md(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def qdummies(x, q=5):
    e = np.quantile(x, np.linspace(0, 1, q + 1)[1:-1])
    b = np.searchsorted(e, x, side="right")
    return np.stack([(b == j).astype(float) for j in range(1, q)], 1)


def wls(X, y, w):
    A = X.T @ (w[:, None] * X) + 1e-10 * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ (w * y))


def reg(d, X, rowmask, idx):
    """Coefficients idx of d ~ X on rowmask rows (valid d), point + patient-cluster bootstrap draws."""
    ok = rowmask & ~np.isnan(d)
    Xv, yv = X[ok], d[ok]
    b0 = wls(Xv, yv, np.ones(len(yv)))
    wb = Wv[:, ok]
    boot = np.array([wls(Xv, yv, wb[b])[idx] for b in range(NB)])
    return b0[idx], boot, int(ok.sum()), len(np.unique(pat[ok]))


def mean_ci(x, mask=None):
    ok = ~np.isnan(x) if mask is None else (~np.isnan(x) & mask)
    pt = x[ok].mean()
    wb = Wv[:, ok]
    bm = (wb @ x[ok]) / wb.sum(1)
    lo, hi = pct(bm)
    return pt, lo, hi, int(ok.sum())


def strat_stats(d, cd):
    """Per stratum raw mean and confidence-adjusted mean (global slope of d on cd refit per draw, evaluated at the
    sample-average cd) with bootstrap draws; same rule as step 5."""
    valid = ~np.isnan(d) & ~np.isnan(cd)
    d0, c0, vf = np.where(valid, d, 0.0), np.where(valid, cd, 0.0), valid.astype(float)
    Mv = (M_ALL & valid[None, :]).astype(float)

    def run(Wm):
        wv = Wm * vf
        ws = wv.sum(1)
        dbar, cbar = (wv @ d0) / ws, (wv @ c0) / ws
        cc = c0 - cbar[:, None]
        b = (wv * (d0 - dbar[:, None]) * cc).sum(1) / np.maximum((wv * cc ** 2).sum(1), 1e-12)
        a = d0 - b[:, None] * cc
        den = wv @ Mv.T
        return (wv * d0) @ Mv.T / den, (wv * a) @ Mv.T / den

    r0, a0 = run(np.ones((1, N)))
    rb, ab = run(Wv)
    return r0[0], a0[0], rb, ab, Mv.sum(1).astype(int)


lines = []
add = lines.append
add("# Step 6 check tables (aggregates only)\n")
add("Produced by `comparison/explain/graph_types/step6_check.py`. Test fold, K=20, 1,345 visits / 1,000 patients, `del` = deletion_mean_change_vs_random, "
    "seed-averaged per visit, d = CEI minus partner, patient-cluster bootstrap 95% CI (2,000 draws, seed 20261025), `*` = CI excludes 0. "
    "Descriptive; nothing is selected from these numbers.\n")

# ------------------------------------------------------------------ 1. overall vs quoted
QUOTED = {("final", "gsat_IG"): -0.0506, ("final", "gsat_Grad"): -0.0473, ("final", "gsat_GNNExp"): -0.0296,
          ("final", "gsat_builtin"): -0.0163, ("final", "gsat_builtin_vs_IG"): -0.0339, ("final", "gc_IG"): -0.0483,
          ("final", "gc_Grad"): -0.0790, ("final", "gc_GNNExp"): 0.0645, ("final", "gc_builtin_vs_Grad"): -0.0575,
          ("R1", "gsat_IG"): -0.0158, ("R1", "gsat_Grad"): -0.0157, ("R1", "gsat_GNNExp"): -0.0152,
          ("R1", "gsat_builtin"): 0.0301, ("R1", "gsat_builtin_vs_IG"): 0.0125, ("R1", "gc_IG"): -0.0136,
          ("R1", "gc_Grad"): -0.0475, ("R1", "gc_GNNExp"): 0.0789, ("R1", "gc_builtin_vs_Grad"): -0.0111}
rows, worst = [], 0.0
for (ver, cname), (d, _, _) in TGT.items():
    pt, lo, hi, n = mean_ci(d)
    q = QUOTED[(ver, cname)]
    worst = max(worst, abs(pt - q))
    rows.append([ver, cname, n, f(pt, lo, hi), f"{q:+.4f}", f"{abs(pt - q):.5f}"])
add("## 1. Overall paired d, recomputed, against the value quoted in step 2 section 2 / step 3 / step 4 / step 5\n")
add(md(["CEI", "comparison", "n visits", "d [95% CI]", "quoted", "abs diff"], rows))
add(f"\nLargest absolute difference to the quoted values: {worst:.5f} (quoted values are rounded to 4 decimals; "
    f"all 18 within 0.0001: {worst < 1e-4}).\n")
print("overall max diff vs quoted: %.5f" % worst)

# label-smoothing cost
QLS = {"IG": -0.0347, "Grad": -0.0316, "GNNExp": -0.0144, "builtin": -0.0464}
rows = []
cdls = CONF["cei_final"] - CONF["cei_R1_ls0"]
pt, lo, hi, n = mean_ci(cdls)
add("## 1b. Label-smoothing cost (cei_final minus cei_R1_ls0)\n")
add(f"Mean confidence change from label smoothing: {f(pt, lo, hi)} (quoted -0.0894).\n")
for s in SOURCES:
    d = V[("cei_final", s)] - V[("cei_R1_ls0", s)]
    pt, lo, hi, n = mean_ci(d)
    X = np.stack([np.ones(N), cdls], 1)
    b0, bb, _, _ = reg(d, X, np.ones(N, bool), [0, 1])
    ci_int, ci_sl = pct(bb[:, 0]), pct(bb[:, 1])
    rows.append([s, n, f(pt, lo, hi), f"{QLS[s]:+.4f}", f(b0[0], *ci_int), f(b0[1], *ci_sl)])
add(md(["source", "n visits", "LS cost mean [CI]", "quoted step 5", "cost with no confidence change (intercept)",
        "slope on confidence change"], rows))
add("\nThe intercept is the part of the cost that does not move with the confidence drop (linear fit, descriptive).\n")
# non-parametric check of the intercept: cost by quintile of the confidence change (edges from all visits)
qe = np.quantile(cdls, [0.2, 0.4, 0.6, 0.8])
qb = np.searchsorted(qe, cdls, side="right")
rows = []
for j in range(5):
    m = qb == j
    row = [f"Q{j + 1} (conf change {cdls[m].min():+.3f} to {cdls[m].max():+.3f}; mean {cdls[m].mean():+.3f})", int(m.sum())]
    for s in SOURCES:
        d = V[("cei_final", s)] - V[("cei_R1_ls0", s)]
        pt, lo, hi, n = mean_ci(d, m)
        row.append(f(pt, lo, hi))
    rows.append(row)
add("LS cost by quintile of the confidence change (Q5 = smallest confidence drop):\n")
add(md(["quintile of confidence change", "n visits"] + SOURCES, rows))
add("\n")

# ------------------------------------------------------------------ 2. mean del per key/source
add("## 2. Mean `del` per model and explainer (all 1,345 visits; GraphCare GNNExplainer 1,344)\n")
rows = []
for k in KEYS:
    row = [k]
    for s in SOURCES:
        pt, lo, hi, n = mean_ci(V[(k, s)])
        row.append(f"{pt:.3f} [{lo:.3f}, {hi:.3f}] (n={n})")
    rows.append(row)
add(md(["model"] + SOURCES, rows))
add("\n")

# ------------------------------------------------------------------ 3. d at equal confidence
add("## 3. How much of each gap is confidence: d at equal confidence\n")
add("Regression d = a + b * (conf_CEI - conf_partner) over all visits. `a` is the paired gap at equal confidence; `b` the change per "
    "+1.0 of confidence difference; `mean dConf` the average confidence difference; `explained` = (mean d - a) / mean d (only "
    "meaningful when mean d is clearly away from 0).\n")
rows = []
for (ver, cname), (d, cc, cp) in TGT.items():
    cd = cc - cp
    X = np.stack([np.ones(N), cd], 1)
    b0, bb, n, npat = reg(d, X, np.ones(N, bool), [0, 1])
    a_ci, b_ci = pct(bb[:, 0]), pct(bb[:, 1])
    md_pt, mlo, mhi, _ = mean_ci(d)
    dcm, dlo, dhi, _ = mean_ci(cd)
    rows.append([ver, cname, n, f(md_pt, mlo, mhi), f(dcm, dlo, dhi), f(b0[0], *a_ci),
                 f"{b0[1]:+.3f} [{b_ci[0]:+.3f}, {b_ci[1]:+.3f}]",
                 f"{(md_pt - b0[0]) / md_pt:.0%}" if abs(md_pt) > 0.02 else "n/a"])
add(md(["CEI", "comparison", "n visits", "mean d", "mean dConf", "d at equal confidence (a)", "slope b", "explained by confidence"], rows))
add("\n")

# ------------------------------------------------------------------ 4. contradiction A: size and history
SIZE_HI = n_nodes > np.quantile(n_nodes, 2 / 3)
SIZE_LO = n_nodes <= np.quantile(n_nodes, 1 / 3)
MULTI = (num_visits >= 2).astype(float)
ZN = (n_nodes - n_nodes.mean()) / n_nodes.std()
ZD = (density - density.mean()) / density.std()
add("## 4. Contradiction A: do size and history effects survive confidence adjustment?\n")
add("Contrast = (high group) minus (low group) of d. Size: n_nodes top third (> %g) minus bottom third (<= %g). History: 2+ visits minus "
    "1 visit. Methods: **raw**; **lin** = regression on the linear confidence difference (step 5's adjustment); **bins** = additive "
    "quintile indicators of both models' confidences (step 2's adjustment). **joint** = one regression of d on the quintile bins "
    "of both confidences + (2+ visits) + standardized n_nodes (+ standardized density in joint+dens), all visits; cells show the "
    "coefficient of history / of n_nodes.\n" % (np.quantile(n_nodes, 2 / 3), np.quantile(n_nodes, 1 / 3)))
rows_s, rows_h, rows_j = [], [], []
for (ver, cname), (d, cc, cp) in TGT.items():
    cd = cc - cp
    B = np.hstack([qdummies(cc), qdummies(cp)])
    cells = {}
    for lab, hi_m, lo_m, ind in [("size", SIZE_HI, SIZE_LO, None), ("hist", num_visits >= 2, num_visits == 1, None)]:
        rm = hi_m | lo_m
        ind = hi_m.astype(float)
        res = []
        for meth, X in [("raw", np.stack([np.ones(N), ind], 1)),
                        ("lin", np.stack([np.ones(N), cd, ind], 1)),
                        ("bins", np.hstack([np.ones((N, 1)), B, ind[:, None]]))]:
            b0, bb, n, npat = reg(d, X, rm, [X.shape[1] - 1])
            res.append(f(b0[0], *pct(bb[:, 0])))
        cells[lab] = (res, n, npat)
    (rs, n_s, p_s), (rh, n_h, p_h) = cells["size"], cells["hist"]
    rows_s.append([ver, cname, n_s, p_s] + rs)
    rows_h.append([ver, cname, n_h, p_h] + rh)
    # joint models (all visits)
    Xj = np.hstack([np.ones((N, 1)), B, MULTI[:, None], ZN[:, None]])
    Xjd = np.hstack([Xj, ZD[:, None]])
    b1, bb1, n, npat = reg(d, Xj, np.ones(N, bool), [Xj.shape[1] - 2, Xj.shape[1] - 1])
    b2, bb2, _, _ = reg(d, Xjd, np.ones(N, bool), [Xjd.shape[1] - 3, Xjd.shape[1] - 2, Xjd.shape[1] - 1])
    rows_j.append([ver, cname, n, f(b1[0], *pct(bb1[:, 0])), f(b1[1], *pct(bb1[:, 1])),
                   f(b2[0], *pct(bb2[:, 0])), f(b2[1], *pct(bb2[:, 1])), f(b2[2], *pct(bb2[:, 2]))])
    print("contrast A done", ver, cname)
add("### 4a. Size contrast (top third of n_nodes minus bottom third)\n")
add(md(["CEI", "comparison", "n visits", "n patients", "raw", "lin", "bins"], rows_s))
add("\n### 4b. History contrast (2+ visits minus single-visit)\n")
add(md(["CEI", "comparison", "n visits", "n patients", "raw", "lin", "bins"], rows_h))
add("\n### 4c. Joint models (confidence bins + history + size [+ density]); coefficients per 1 SD for size and density\n")
add(md(["CEI", "comparison", "n visits", "history (joint)", "n_nodes per SD (joint)", "history (joint+dens)",
        "n_nodes per SD (joint+dens)", "density per SD (joint+dens)"], rows_j))
add("\n")

# ------------------------------------------------------------------ 5. strata scan
add("## 5. Strata scan: in how many of the %d strata (all >= 122 visits) is CEI better / worse than the partner?\n" % len(STRATA))
add("Per comparison, number of strata whose mean d is above 0 and whose 95% CI excludes 0 (`better`) / below 0 and excludes 0 (`worse`); "
    "raw and confidence-adjusted (step 5 rule, linear on the confidence difference, refit per draw). Strata overlap, so counts are not independent.\n")
rows = []
scan_pos = []
for (ver, cname), (d, cc, cp) in TGT.items():
    r0, a0, rb, ab, n = strat_stats(d, cc - cp)
    rlo, rhi = pct(rb)
    alo, ahi = pct(ab)
    sl = slice(1, None)
    better_r = int(((rlo > 0))[sl].sum())
    worse_r = int(((rhi < 0))[sl].sum())
    better_a = int(((alo > 0))[sl].sum())
    worse_a = int(((ahi < 0))[sl].sum())
    pos_mean = int((r0[sl] > 0).sum())
    rows.append([ver, cname, len(STRATA), pos_mean, better_r, worse_r, better_a, worse_a])
    for i in range(1, len(S_NAMES)):
        if alo[i] > 0:
            scan_pos.append((ver, cname, S_NAMES[i], n[i], r0[i], rlo[i], rhi[i], a0[i], alo[i], ahi[i]))
add(md(["CEI", "comparison", "strata", "strata with mean d > 0", "better raw (CI>0)", "worse raw (CI<0)",
        "better adj (CI>0)", "worse adj (CI<0)"], rows))
add("\nStrata where the confidence-adjusted CI is above 0 (CEI better at equal confidence), every one listed:\n")
rows = [[v, c, s, n, f(r, rl, rh), f(a, al, ah)] for v, c, s, n, r, rl, rh, a, al, ah in scan_pos]
add(md(["CEI", "comparison", "stratum", "n visits", "raw", "adjusted"], rows) if rows else "(none)")
add("\n")

# ------------------------------------------------------------------ 6. class groups
add("## 6. Class groups (six groups fixed in steps 3 and 5 before looking at differences): raw / adjusted d\n")
rows = []
for gi in CLASS_ROWS:
    for ver in ["final", "R1"]:
        row = [S_NAMES[gi].replace("class: ", ""), ver]
        for cname in ["gsat_IG", "gsat_Grad", "gsat_builtin", "gc_IG", "gc_Grad"]:
            d, cc, cp = TGT[(ver, cname)]
            r0, a0, rb, ab, n = strat_stats(d, cc - cp)
            rlo, rhi = pct(rb[:, gi:gi + 1])
            alo, ahi = pct(ab[:, gi:gi + 1])
            row.append(f"{f(r0[gi], rlo[0], rhi[0])} / {a0[gi]:+.3f}{'*' if (alo[0] > 0 or ahi[0] < 0) else ''}")
        row.insert(2, n[gi])
        rows.append(row)
add(md(["class group", "CEI", "n visits", "IG vs GSAT", "Grad vs GSAT", "built-in vs GSAT built-in", "IG vs GraphCare",
        "Grad vs GraphCare"], rows))
add("\nCell = raw d [CI] / adjusted d (star = adjusted CI excludes 0).\n")

# ------------------------------------------------------------------ 7. contradiction B: built-in by size
add("## 7. Contradiction B: built-in gap (CEI built-in minus GSAT built-in) by n_nodes tertile\n")
rows = []
tert = tertile_masks("n_nodes", n_nodes)
T_IDX = [S_NAMES.index(l) for l, _ in tert]
for ver in ["final", "R1"]:
    d, cc, cp = TGT[(ver, "gsat_builtin")]
    r0, a0, rb, ab, n = strat_stats(d, cc - cp)
    B = np.hstack([qdummies(cc), qdummies(cp)])
    for ti, (lab, m) in zip(T_IDX, tert):
        rlo, rhi = pct(rb[:, ti:ti + 1])
        alo, ahi = pct(ab[:, ti:ti + 1])
        # bins-adjusted stratum mean: residual of d after removing quintile-bin fit (re-centred), per step 2
        ok = ~np.isnan(d)
        Xb = np.hstack([np.ones((N, 1)), B])
        beta = wls(Xb[ok], d[ok], np.ones(ok.sum()))
        res = d - (Xb[:, 1:] @ beta[1:] - (Xb[ok][:, 1:] @ beta[1:]).mean())
        bl = []
        for b in range(NB):
            w = Wv[b] * ok
            be = wls(Xb[ok], d[ok], w[ok])
            fit = Xb[:, 1:] @ be[1:]
            adj = d - fit + (w[ok] @ fit[ok]) / w[ok].sum()
            bl.append((w[m & ok] @ adj[m & ok]) / w[m & ok].sum())
        bl = np.array(bl)
        blo, bhi = np.percentile(bl, [2.5, 97.5])
        rows.append([ver, lab, int(n[ti]), f(r0[ti], rlo[0], rhi[0]), f(a0[ti], alo[0], ahi[0]),
                     f(float(res[m & ok].mean()), blo, bhi)])
add(md(["CEI", "stratum", "n visits", "raw", "adjusted (lin)", "adjusted (bins)"], rows))
add("\nThe built-in contrast high-minus-low tertile is in section 4a (rows `gsat_builtin`).\n")

# ------------------------------------------------------------------ write
add("\n## Command\n")
add("`env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 comparison/explain/graph_types/step6_check.py` (repo root; writes this file)\n")
OUT.write_text("\n".join(lines), encoding="utf-8")
print("wrote %s in %.0fs" % (OUT, time.time() - T0))
