#!/usr/bin/env python3
"""Graph-type analysis, step 3: clinical context and confidence axis.

Paired per-visit differences d = CEI - comparator (comparator = GSAT, GraphCare or ProtGNN), CEI = cei_final and, separately,
cei_R1_ls0, stratified by
  true class (20) | clinically grouped classes (6) | correctness pattern | seed agreement |
  CEI confidence | comparator confidence | confidence difference | joint confidence grid (3 x 3, the matched comparison).

RULES WRITTEN BEFORE LOOKING AT ANY DIFFERENCE:
  * Unit = visit. A model/source value of a visit = mean over the seeds in which the source is ok; the visit counts only
    if ok in >= 2 of the 3 seeds. A paired difference needs both sides.
  * Class groups (by true_class_id), fixed here:
      cardiac                  = Heart failure, NSTEMI, Atrial fibrillation              (9, 14, 19)
      infection                = Sepsis, UTI/pyelonephritis, Skin/soft-tissue infection,
                                 Acute pharyngitis, Lower respiratory disease            (15, 17, 16, 1, 12)
      injury_pain              = Head injury, Limb injury/pain, Back/spine pain          (8, 11, 4)
      gastrointestinal         = GI bleed, Intestinal obstruction                        (7, 18)
      metabolic_renal_heme     = Acute kidney failure, Anemia, Diabetes, Hypokalemia     (0, 3, 5, 10)
      neuro_psych_substance    = Epilepsy, Major depressive disorder, Alcohol abuse      (6, 13, 2)
  * Correct = at least 2 of the 3 seeds predict the true class (correct_<key> > 0.5).
    Pattern = (CEI correct?, comparator correct?) -> both correct / CEI only / comparator only / both wrong.
  * Seed agreement = pred_agree_<key> == 1 (all 3 seeds predict the same class); joint 2 x 2 like the pattern.
  * Confidence = seed-mean predicted-class probability (conf_<key>). Quintile bins of each model's own confidence over
    all 1,345 visits (edges printed); quintile bins of conf_CEI - conf_comparator; joint grid = tertile of CEI conf x
    tertile of comparator conf (matched comparison).
  * Confidence-adjusted difference = regression adjustment: weighted least squares of d on a quadratic in
    (conf_CEI, conf_comparator) (intercept, a, b, a^2, b^2, a*b; a, b standardised), refitted inside every bootstrap
    draw; adjusted d_i = d_i - fitted_i + mean(fitted); the stratum mean of adjusted d is the stratum difference after
    removing what the two confidences alone predict (overall adjusted mean = overall raw mean by construction).
  * Uncertainty: patient-cluster bootstrap, resample patient_idx with replacement, 2,000 draws, seed 20261025, 95%
    percentile CI. n < 10 visits: nothing printed; 10 <= n < 30: printed, marked (dagger), not interpreted.
  * "Supported": CI excludes 0 (n >= 30) AND (repeats in BOTH CEI versions for >= 2 same-source explainers, or, for an
    ordered confidence axis, a monotone trend whose last-minus-first bin CI excludes 0 in both versions).
  * Contrast vs rest = stratum mean minus mean of all other visits (does the stratum differ from the rest).

Aggregates only; no row, identifier or raw value is printed or written. Inputs: data/visit_features.csv,
data/visit_scores.csv (read inside this script only). Outputs: out/step3-report.md, out/step3-tables.csv.
Run from the repo root:  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 comparison/explain/graph_types/step3_clinical.py
Paths: _paths.py in this folder (W = working folder with data/ and out/; data/ holds per-visit tables with anonymous indices and is never committed).
"""
import csv
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from comparison.explain.graph_types._paths import WORK  # noqa: E402
W = str(WORK)
DRAWS = 2000
BOOT_SEED = 20261025
MIN_PRINT = 10
MIN_INTERPRET = 30

CEI_KEYS = ["cei_final", "cei_R1_ls0"]
COMP_KEYS = {"gsat": "GSAT", "graphcare": "GraphCare", "protgnn": "ProtGNN"}
METRICS = {
    "del": "deletion_mean_change_vs_random",
    "ins": "insertion_mean_change_vs_random",
    "fid+": "fidelity_plus",
    "fid-": "fidelity_minus",
}
# (column name, CEI source, comparator source); same-source comparisons first.
COMPARISONS = {
    "gsat": [("IG", "IG", "IG"), ("Grad", "Grad", "Grad"), ("GNNExp", "GNNExp", "GNNExp"),
             ("built-in", "builtin", "builtin"), ("built-in vs GSAT IG", "builtin", "IG")],
    "graphcare": [("IG", "IG", "IG"), ("Grad", "Grad", "Grad"), ("GNNExp", "GNNExp", "GNNExp"),
                  ("built-in vs GC Grad", "builtin", "Grad")],
    # ProtGNN was added after steps 2-6 had been run for GSAT and GraphCare (the supervisor's comparison names GSAT and
    # ProtGNN); the rules of this script are applied to it unchanged. Its built-in subgraph has no eligible evidence in
    # part of the visits (nan there), and those visits drop out of the built-in comparisons.
    "protgnn": [("IG", "IG", "IG"), ("Grad", "Grad", "Grad"), ("GNNExp", "GNNExp", "GNNExp"),
                ("built-in", "builtin", "builtin"), ("built-in vs ProtGNN IG", "builtin", "IG")],
}
SAME_SOURCE = {"gsat": ["IG", "Grad", "GNNExp", "built-in"], "graphcare": ["IG", "Grad", "GNNExp"],
               "protgnn": ["IG", "Grad", "GNNExp", "built-in"]}
CONF_COL = "dConf"

CLASS_GROUPS = {
    "cardiac": [9, 14, 19],
    "infection": [15, 17, 16, 1, 12],
    "injury_pain": [8, 11, 4],
    "gastrointestinal": [7, 18],
    "metabolic_renal_heme": [0, 3, 5, 10],
    "neuro_psych_substance": [6, 13, 2],
}
ADJ_AXES = ["class", "group", "pattern", "agree"]
ORDERED_AXES = ["conf_cei", "conf_comp", "conf_diff"]
AXIS_TITLE = {
    "class": "True class (20 classes)",
    "group": "Clinically grouped classes",
    "pattern": "Correctness pattern (correct = at least 2 of 3 seeds predict the true class)",
    "agree": "Seed agreement (stable = all 3 seeds predict the same class)",
    "conf_cei": "CEI confidence (quintiles)",
    "conf_comp": "Comparator confidence (quintiles)",
    "conf_diff": "Confidence difference conf_CEI - conf_comparator (quintiles)",
    "grid": "Matched comparison: CEI confidence tertile x comparator confidence tertile",
}

warnings.filterwarnings("ignore")


def pct(a, q):
    return np.nanpercentile(a, q, axis=0)


# ----------------------------------------------------------------------------------------------- load
feat = pd.read_csv(os.path.join(W, "data", "visit_features.csv")).sort_values("visit_idx").reset_index(drop=True)
V = len(feat)
assert V == 1345 and (feat["visit_idx"].values == np.arange(V)).all()
patient = feat["patient_idx"].values.astype(int)
P = int(patient.max()) + 1
cls_id = feat["true_class_id"].values.astype(int)
cls_name = dict(zip(feat["true_class_id"], feat["true_class"]))
conf = {k: feat["conf_" + k].values.astype(float) for k in list(COMP_KEYS) + CEI_KEYS}
correct = {k: feat["correct_" + k].values.astype(float) > 0.5 for k in list(COMP_KEYS) + CEI_KEYS}
agree = {k: feat["pred_agree_" + k].values.astype(float) == 1 for k in list(COMP_KEYS) + CEI_KEYS}

need_keys = set(CEI_KEYS) | set(COMP_KEYS)
sc = pd.read_csv(os.path.join(W, "data", "visit_scores.csv"),
                 usecols=["visit_idx", "key", "seed", "source", "ok"] + list(METRICS.values()))
sc = sc[(sc["ok"] == 1) & sc["key"].isin(need_keys)]
val = {}
grp = sc.groupby(["key", "source", "visit_idx"])
for mname, col in METRICS.items():
    mean = grp[col].mean()
    cnt = grp[col].count()
    mean = mean[cnt >= 2]
    for (key, src), sub in mean.groupby(level=[0, 1]):
        arr = np.full(V, np.nan)
        arr[sub.index.get_level_values(2).values] = sub.values
        val[(key, src, mname)] = arr
del sc

# ------------------------------------------------------------------------------------------- bootstrap weights
rng = np.random.default_rng(BOOT_SEED)
C = rng.multinomial(P, np.full(P, 1.0 / P), size=DRAWS).astype(float)  # (DRAWS, P) patient resample counts
WV = np.vstack([np.ones((1, V)), C[:, patient]])  # row 0 = the original sample


def stats(Wm, Vv, Z):
    """Weighted stratum means and contrasts vs rest for every bootstrap row. Wm (R,V) weights*valid, Vv (R,V)."""
    num = (Wm * Vv) @ Z
    den = Wm @ Z
    mean = num / den
    rest = (num[:, :1] - num) / (den[:, :1] - den)
    return mean, mean - rest


def lohi(a):
    return pct(a[1:], 2.5), pct(a[1:], 97.5)


def quint_bins(x, n):
    qs = np.quantile(x, np.arange(1, n) / n)
    edges = np.unique(np.round(qs, 12))
    idx = np.searchsorted(edges, x, side="right")
    labels = []
    for i in range(len(edges) + 1):
        if i == 0:
            labels.append("Q1 (<%.3f)" % edges[0])
        elif i == len(edges):
            labels.append("Q%d (>=%.3f)" % (i + 1, edges[-1]))
        else:
            labels.append("Q%d [%.3f, %.3f)" % (i + 1, edges[i - 1], edges[i]))
    return idx, labels, edges


def build_axes(cei, comp):
    cn = COMP_KEYS[comp]
    axes = []
    # true class
    ids = sorted(cls_name)
    axes.append(dict(key="class", labels=["%d %s" % (i, cls_name[i]) for i in ids],
                     masks=[cls_id == i for i in ids]))
    gnames = list(CLASS_GROUPS)
    axes.append(dict(key="group", labels=gnames,
                     masks=[np.isin(cls_id, CLASS_GROUPS[g]) for g in gnames]))
    c1, c2 = correct[cei], correct[comp]
    axes.append(dict(key="pattern",
                     labels=["both correct", "CEI only correct", cn + " only correct", "both wrong"],
                     masks=[c1 & c2, c1 & ~c2, ~c1 & c2, ~c1 & ~c2]))
    a1, a2 = agree[cei], agree[comp]
    axes.append(dict(key="agree",
                     labels=["both seed-stable", "CEI stable only", cn + " stable only", "neither stable"],
                     masks=[a1 & a2, a1 & ~a2, ~a1 & a2, ~a1 & ~a2]))
    for key, x in (("conf_cei", conf[cei]), ("conf_comp", conf[comp]), ("conf_diff", conf[cei] - conf[comp])):
        idx, labels, edges = quint_bins(x, 5)
        axes.append(dict(key=key, labels=labels, masks=[idx == i for i in range(len(labels))], edges=edges))
    i1, l1, e1 = quint_bins(conf[cei], 3)
    i2, l2, e2 = quint_bins(conf[comp], 3)
    glabels, gmasks = [], []
    for a in range(len(l1)):
        for b in range(len(l2)):
            glabels.append("CEI T%d x %s T%d" % (a + 1, cn, b + 1))
            gmasks.append((i1 == a) & (i2 == b))
    axes.append(dict(key="grid", labels=glabels, masks=gmasks, edges=(e1, e2)))
    return axes


def design(cei, comp):
    a = conf[cei]
    b = conf[comp]
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    return np.column_stack([np.ones(V), a, b, a * a, b * b, a * b])


# ------------------------------------------------------------------------------------------------ analysis
results = {}   # (cei, comp, cname, metric, adj) -> dict
axes_of = {}   # (cei, comp) -> axes with column ranges
r2_of = {}
trend = {}     # (cei, comp, cname, metric, adj, axis) -> dict

for cei in CEI_KEYS:
    for comp in COMP_KEYS:
        axes = build_axes(cei, comp)
        cols = [np.ones(V, bool)]
        pos = 1
        for ax in axes:
            ax["lo"], ax["hi"] = pos, pos + len(ax["labels"])
            pos += len(ax["labels"])
            cols.extend(ax["masks"])
        Z = np.column_stack(cols).astype(float)
        axes_of[(cei, comp)] = (axes, Z)
        X = design(cei, comp)
        XX = (X[:, :, None] * X[:, None, :]).reshape(V, -1)
        k = X.shape[1]

        todo = []
        for cname, sc_c, sc_p in COMPARISONS[comp]:
            for m in METRICS:
                d = val[(cei, sc_c, m)] - val[(comp, sc_p, m)]
                todo.append((cname, m, d))
        todo.append((CONF_COL, "conf", conf[cei] - conf[comp]))

        for cname, m, d in todo:
            valid = ~np.isnan(d)
            dz = np.where(valid, d, 0.0)
            Wm = WV * valid[None, :]
            n_cell = Z[valid].sum(axis=0).astype(int)
            n_pat = np.array([len(np.unique(patient[valid & (Z[:, j] > 0)])) for j in range(Z.shape[1])])
            for adj in (False, True):
                if adj and m == "conf":
                    continue
                if not adj:
                    Vv = np.broadcast_to(dz, Wm.shape)
                else:
                    A = (Wm @ XX).reshape(-1, k, k) + 1e-9 * np.eye(k)
                    B = Wm @ (X * dz[:, None])
                    beta = np.linalg.solve(A, B[..., None])[..., 0]
                    f = beta @ X.T
                    fbar = (Wm * f).sum(1) / Wm.sum(1)
                    Vv = dz[None, :] - f
                    if m == "del":
                        sst = (Wm[0] * (dz - (Wm[0] * dz).sum() / Wm[0].sum()) ** 2).sum()
                        ssr = (Wm[0] * Vv[0] ** 2).sum()
                        r2_of[(cei, comp, cname)] = 1 - ssr / sst
                mean, contr = stats(Wm, Vv, Z)
                if adj:
                    mean = mean + fbar[:, None]
                lo, hi = lohi(mean)
                clo, chi = lohi(contr)
                results[(cei, comp, cname, m, adj)] = dict(
                    est=mean[0], lo=lo, hi=hi, cest=contr[0], clo=clo, chi=chi, n=n_cell, npat=n_pat)
                for ax in axes:
                    if ax["key"] not in ORDERED_AXES:
                        continue
                    idxs = [j for j in range(ax["lo"], ax["hi"]) if n_cell[j] >= MIN_INTERPRET]
                    if len(idxs) < 3:
                        continue
                    seq = mean[0, idxs]
                    df = np.diff(seq)
                    mono = bool((df >= 0).all() or (df <= 0).all())
                    tdiff = mean[:, idxs[-1]] - mean[:, idxs[0]]
                    tlo, thi = lohi(tdiff[:, None])
                    trend[(cei, comp, cname, m, adj, ax["key"])] = dict(
                        mono=mono, est=tdiff[0], lo=tlo[0], hi=thi[0], first=idxs[0] - ax["lo"],
                        last=idxs[-1] - ax["lo"])

# ------------------------------------------------------------------------------------------------ formatting


def fmt(r, j, with_contrast=False, nmin_print=MIN_PRINT):
    n = r["n"][j]
    if n < nmin_print:
        return "-"
    e, l, h = (r["cest"][j], r["clo"][j], r["chi"][j]) if with_contrast else (r["est"][j], r["lo"][j], r["hi"][j])
    if np.isnan(e):
        return "-"
    s = "%+.3f [%+.3f, %+.3f]" % (e, l, h)
    if n >= MIN_INTERPRET and (l > 0 or h < 0):
        s = "**" + s + "**"
    if n < MIN_INTERPRET:
        s += " †"
    return s


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def axis_table(cei, comp, ax_key, metric="del", adj=False, contrast=False, with_conf=True):
    axes, Z = axes_of[(cei, comp)]
    ax = [a for a in axes if a["key"] == ax_key][0]
    cnames = [c[0] for c in COMPARISONS[comp]]
    first = results[(cei, comp, cnames[0], metric, adj)]
    cols = cnames[:]
    header = ["stratum", "n visits", "n patients"] + (["dConf (CEI-comp)"] if with_conf else []) + cols
    rows = []
    idxs = [0] + list(range(ax["lo"], ax["hi"]))
    for j in idxs:
        if first["n"][j] < MIN_PRINT:
            continue
        label = "All visits" if j == 0 else ax["labels"][j - ax["lo"]]
        row = [label, int(first["n"][j]), int(first["npat"][j])]
        if with_conf:
            row.append(fmt(results[(cei, comp, CONF_COL, "conf", False)], j))
        for c in cnames:
            row.append(fmt(results[(cei, comp, c, metric, adj)], j, with_contrast=contrast and j > 0))
        rows.append(row)
    return md_table(header, rows)


def excl0(r, j, contrast=False):
    n = r["n"][j]
    if n < MIN_INTERPRET:
        return 0
    l, h = (r["clo"][j], r["chi"][j]) if contrast else (r["lo"][j], r["hi"][j])
    if l > 0:
        return 1
    if h < 0:
        return -1
    return 0


def support_lists():
    """Level and heterogeneity support over both CEI versions; trend support for ordered axes."""
    level, hetero, trends = [], [], []
    for comp in COMP_KEYS:
        same = SAME_SOURCE[comp]
        axes_f, _ = axes_of[("cei_final", comp)]
        axes_1, _ = axes_of[("cei_R1_ls0", comp)]
        for ai, ax in enumerate(axes_f):
            ax1 = axes_1[ai]
            for adj in (False, True):
                if adj and ax["key"] not in ADJ_AXES:
                    continue
                for li, label in enumerate(ax["labels"]):
                    jf, j1 = ax["lo"] + li, ax1["lo"] + li
                    for kind, store in (("level", level), ("contrast", hetero)):
                        hits = []
                        for c in same:
                            rf = results[("cei_final", comp, c, "del", adj)]
                            r1 = results[("cei_R1_ls0", comp, c, "del", adj)]
                            ef = excl0(rf, jf, kind == "contrast")
                            e1 = excl0(r1, j1, kind == "contrast")
                            if ef != 0 and ef == e1:
                                hits.append((c, ef))
                        if len(hits) >= 2 and len({h[1] for h in hits}) == 1:
                            store.append((COMP_KEYS[comp], AXIS_TITLE[ax["key"]].split(" (")[0], label,
                                          "adjusted" if adj else "raw", hits[0][1], ", ".join(h[0] for h in hits)))
            if ax["key"] in ORDERED_AXES:
                for adj in (False,):
                    hits = []
                    for c in same:
                        tf = trend.get(("cei_final", comp, c, "del", adj, ax["key"]))
                        t1 = trend.get(("cei_R1_ls0", comp, c, "del", adj, ax["key"]))
                        if not tf or not t1:
                            continue
                        sf = 1 if tf["lo"] > 0 else (-1 if tf["hi"] < 0 else 0)
                        s1 = 1 if t1["lo"] > 0 else (-1 if t1["hi"] < 0 else 0)
                        if tf["mono"] and t1["mono"] and sf != 0 and sf == s1:
                            hits.append((c, sf))
                    if len(hits) >= 2 and len({h[1] for h in hits}) == 1:
                        trends.append((COMP_KEYS[comp], AXIS_TITLE[ax["key"]], hits[0][1],
                                       ", ".join(h[0] for h in hits)))
    return level, hetero, trends


def not_different(kind):
    """Strata (class/group/pattern/agree, n >= 30) whose CI includes 0 in every same-source comparison, both versions."""
    out = []
    for comp in COMP_KEYS:
        axes_f, _ = axes_of[("cei_final", comp)]
        for ax in axes_f:
            if ax["key"] not in ADJ_AXES:
                continue
            for li, label in enumerate(ax["labels"]):
                j = ax["lo"] + li
                if results[("cei_final", comp, "IG", "del", False)]["n"][j] < MIN_INTERPRET:
                    continue
                allzero = True
                for cei in CEI_KEYS:
                    for c in SAME_SOURCE[comp]:
                        if excl0(results[(cei, comp, c, "del", False)], j, kind == "contrast") != 0:
                            allzero = False
                if allzero:
                    out.append((COMP_KEYS[comp], AXIS_TITLE[ax["key"]].split(" (")[0], label))
    return out


# --------------------------------------------------------------------------------------------------- report
L = []
L.append("# Step 3: clinical context and confidence axis\n")

L.append("## Method (rules fixed in the script before any difference was looked at)\n")
L.append("Test fold, K=20, primary_dx_v4, fair_v10t, 1,345 visits of 1,000 patients, 3 seeds per model. Descriptive and "
         "exploratory: nothing is selected or tuned from these numbers. d = CEI - comparator per visit (seed-averaged, "
         "visit counted only if the source is ok in >= 2 of 3 seeds); positive d = CEI scores higher. Headline score "
         "`del` = deletion_mean_change_vs_random (higher is better). For `fid-` (fidelity-) lower is better, so there "
         "a negative d favours CEI. CEI = cei_final (label smoothing 0.1) and, separately, cei_R1_ls0 (no label "
         "smoothing). Comparators: GSAT, GraphCare and ProtGNN (ProtGNN was added later, same rules; its built-in subgraph "
         "has no eligible evidence in part of the visits, which then drop out of the built-in comparisons). Columns IG / Grad / GNNExp compare the same explainer in both "
         "models; `built-in` is CEI built-in vs the comparator's built-in (GSAT, ProtGNN); `built-in vs GSAT IG`, `built-in vs GC Grad` and `built-in vs ProtGNN IG` compare "
         "the CEI built-in explanation with the comparator's best post-hoc explainer (GraphCare's built-in is not "
         "usable, so it is not compared). 95% CIs: patient-cluster bootstrap, 2,000 draws, seed 20261025. **Bold** = "
         "CI excludes 0 (n >= 30). † = 10 <= n < 30, shown but not interpreted; cells under 10 visits are not "
         "printed. `dConf` = mean confidence of CEI minus mean confidence of the comparator in the stratum (the "
         "confounder).\n")
L.append("Class groups (fixed up front): " + "; ".join(
    "%s = %s" % (g, ", ".join(cls_name[i] for i in ids)) for g, ids in CLASS_GROUPS.items()) + ".\n")
L.append("Correct = at least 2 of the 3 seeds predict the true class. Seed-stable = all 3 seeds predict the same class. "
         "Confidence = seed-mean predicted-class probability.\n")
L.append("Confidence-adjusted ('adj') = regression adjustment: weighted least squares of d on a quadratic in "
         "(CEI confidence, comparator confidence), refitted in every bootstrap draw; the adjusted stratum mean is "
         "what remains after removing what the two confidences alone predict (same overall mean as raw). The 3 x 3 "
         "matched grid (tertile of CEI confidence x tertile of comparator confidence) is the non-parametric "
         "confidence-matched comparison. Edges of the fixed confidence bins:\n")
bin_rows = []
for cei in CEI_KEYS:
    for comp in COMP_KEYS:
        axes, _ = axes_of[(cei, comp)]
        for ax in axes:
            if ax["key"] in ORDERED_AXES:
                bin_rows.append((cei, COMP_KEYS[comp], AXIS_TITLE[ax["key"]], "; ".join(ax["labels"])))
            if ax["key"] == "grid":
                bin_rows.append((cei, COMP_KEYS[comp], "grid tertile edges (CEI | comparator)",
                                 ", ".join("%.3f" % e for e in ax["edges"][0]) + " | " +
                                 ", ".join("%.3f" % e for e in ax["edges"][1])))
L.append(md_table(["CEI version", "comparator", "axis", "bins"], bin_rows) + "\n")

L.append("## Inclusion (visits and patients in each paired comparison, overall)\n")
inc = []
for cei in CEI_KEYS:
    for comp in COMP_KEYS:
        row = [cei, COMP_KEYS[comp]]
        for cname, _, _ in COMPARISONS[comp]:
            r = results[(cei, comp, cname, "del", False)]
            row.append("%s: %d visits / %d patients" % (cname, r["n"][0], r["npat"][0]))
        inc.append([row[0], row[1], "; ".join(row[2:])])
L.append(md_table(["CEI version", "comparator", "paired visits (all 1,345 minus visits missing a side)"], inc) + "\n")

L.append("## Overall paired differences and how much confidence explains\n")
rows = []
for cei in CEI_KEYS:
    for comp in COMP_KEYS:
        rc = results[(cei, comp, CONF_COL, "conf", False)]
        for cname, _, _ in COMPARISONS[comp]:
            r = results[(cei, comp, cname, "del", False)]
            rows.append([cei, COMP_KEYS[comp], cname, fmt(r, 0), fmt(rc, 0), "%.3f" % r2_of[(cei, comp, cname)]])
L.append(md_table(["CEI version", "comparator", "comparison", "mean d (del) [95% CI]",
                   "mean dConf [95% CI]", "R2 of d on confidence (quadratic)"], rows) + "\n")
L.append("R2 = share of the visit-level variance of d that the two confidences alone explain (point estimate, "
         "original sample).\n")

sec = 0
for comp in COMP_KEYS:
    for cei in CEI_KEYS:
        sec += 1
        L.append("## Tables: %s vs %s\n" % (cei, COMP_KEYS[comp]))
        for ax_key in ["group", "class", "pattern", "agree", "conf_cei", "conf_comp", "conf_diff", "grid"]:
            L.append("### %s - %s: raw difference in `del`\n" % (AXIS_TITLE[ax_key], cei + " vs " + COMP_KEYS[comp]))
            L.append(axis_table(cei, comp, ax_key) + "\n")
            if ax_key in ORDERED_AXES:
                rows = []
                axes, _ = axes_of[(cei, comp)]
                ax = [a for a in axes if a["key"] == ax_key][0]
                for cname, _, _ in COMPARISONS[comp]:
                    t = trend.get((cei, comp, cname, "del", False, ax_key))
                    if t:
                        rows.append([cname, "yes" if t["mono"] else "no",
                                     "%+.3f [%+.3f, %+.3f]" % (t["est"], t["lo"], t["hi"])])
                L.append("Trend over bins with n >= 30 (last bin minus first bin): \n")
                L.append(md_table(["comparison", "monotone", "last - first [95% CI]"], rows) + "\n")
            if ax_key in ADJ_AXES:
                L.append("### %s - %s: confidence-adjusted difference in `del`\n" % (
                    AXIS_TITLE[ax_key], cei + " vs " + COMP_KEYS[comp]))
                L.append(axis_table(cei, comp, ax_key, adj=True, with_conf=False) + "\n")
        L.append("### Class groups - %s vs %s: contrast with all other visits (`del`, raw)\n" % (
            cei, COMP_KEYS[comp]))
        L.append(axis_table(cei, comp, "group", contrast=True, with_conf=False) + "\n")
        for m in ("ins", "fid+", "fid-"):
            L.append("### Class groups - %s vs %s: raw difference in `%s`\n" % (cei, COMP_KEYS[comp], m))
            L.append(axis_table(cei, comp, "group", metric=m, with_conf=False) + "\n")

level, hetero, trends = support_lists()
L.append("## Supported findings (rule above; `del`; same-source explainers)\n")
L.append("### Level: CEI differs from the comparator inside the stratum (CI excludes 0 in both CEI versions for at "
         "least 2 same-source explainers, same sign, n >= 30)\n")
L.append(md_table(["comparator", "axis", "stratum (label of cei_final)", "scale", "sign (+ = CEI higher)",
                   "explainers"],
                  [[a, b, c, d, "+" if e > 0 else "-", f] for a, b, c, d, e, f in level]) if level
         else "None met the rule.\n")
L.append("")
L.append("### Heterogeneity: the stratum differs from all other visits (contrast CI excludes 0 in both CEI versions "
         "for at least 2 same-source explainers)\n")
L.append(md_table(["comparator", "axis", "stratum (label of cei_final)", "scale", "sign (+ = CEI more favourable "
                   "here than elsewhere)", "explainers"],
                  [[a, b, c, d, "+" if e > 0 else "-", f] for a, b, c, d, e, f in hetero]) if hetero
         else "None met the rule.\n")
L.append("")
L.append("### Monotone trend over confidence bins (monotone in both CEI versions and last-minus-first CI excludes 0 for "
         "at least 2 same-source explainers)\n")
L.append(md_table(["comparator", "axis", "sign of last - first", "explainers"],
                  [[a, b, "+" if c > 0 else "-", d] for a, b, c, d in trends]) if trends else "None met the rule.\n")
L.append("")
L.append("## Checked and not different\n")
nd = not_different("level")
L.append("Strata (n >= 30; class, group, pattern, agree axes) whose raw `del` CI includes 0 for every same-source "
         "explainer in both CEI versions (CEI neither better nor worse than the comparator there):\n")
L.append(md_table(["comparator", "axis", "stratum"], nd) + "\n" if nd else "None.\n")
nh = not_different("contrast")
L.append("Strata (n >= 30; same axes) whose contrast with the rest includes 0 for every same-source explainer in both "
         "CEI versions (the stratum behaves like the other visits):\n")
L.append(md_table(["comparator", "axis", "stratum"], nh) + "\n" if nh else "None.\n")

os.makedirs(os.path.join(W, "out"), exist_ok=True)
with open(os.path.join(W, "out", "step3-report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(L))

# ------------------------------------------------------------------------------------------------- csv
with open(os.path.join(W, "out", "step3-tables.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["cei", "comparator", "comparison", "metric", "adjusted", "axis", "stratum", "n_visits", "n_patients",
                "mean", "lo", "hi", "contrast_mean", "contrast_lo", "contrast_hi"])
    nrows = 0
    for (cei, comp, cname, m, adj), r in results.items():
        axes, _ = axes_of[(cei, comp)]
        for ax in axes:
            for li, label in enumerate(ax["labels"]):
                j = ax["lo"] + li
                if r["n"][j] < MIN_PRINT:
                    continue
                w.writerow([cei, comp, cname, m, int(adj), ax["key"], label, int(r["n"][j]), int(r["npat"][j]),
                            "%.6f" % r["est"][j], "%.6f" % r["lo"][j], "%.6f" % r["hi"][j],
                            "%.6f" % r["cest"][j], "%.6f" % r["clo"][j], "%.6f" % r["chi"][j]])
                nrows += 1

print("report lines:", sum(x.count("\n") + 1 for x in L))
print("csv rows:", nrows)
print("supported level / hetero / trend:", len(level), len(hetero), len(trends))
print("not-different level / contrast:", len(nd), len(nh))
