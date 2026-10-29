"""Graph-type analysis, step 5: label-smoothing cost by graph type, agreement of the CEI-vs-GSAT/GraphCare strata for cei_final and
cei_R1_ls0, a patient-cluster bootstrapped linear model and a depth-2 tree checked on a patient-level held-out split.

Reads only W/data/visit_features.csv and W/data/visit_scores.csv. Writes W/out/step5-tables.md. Prints progress and
aggregate checks only (no rows, ids or raw values; every printed cell holds >= 10 visits).

Run from the repo root:
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 comparison/explain/graph_types/step5_label_smoothing.py
Paths: _paths.py in this folder (W = working folder with data/ and out/; data/ holds per-visit tables with anonymous indices and is never committed).

Headline metric `del` = deletion_mean_change_vs_random, averaged over a visit's 3 seeds (visit counts for a key/source
when ok in >= 2 seeds). Test fold, descriptive: nothing is selected from these numbers.
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
OUT = W / "out" / "step5-tables.md"
SEED = 20261025
NB = 2000
MIN_PRINT = 10
MIN_INTERP = 30
SOURCES = ["IG", "Grad", "GNNExp", "builtin"]
METRIC = "deletion_mean_change_vs_random"
KEYS = ["gsat", "graphcare", "cei_final", "cei_R1_ls0"]
T0 = time.time()

# ----------------------------------------------------------------------------------------------------------------
# 1. Features and BIN RULES (fixed here, before any score or difference is loaded)
# ----------------------------------------------------------------------------------------------------------------
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
n_meas = fcol("n_kind_measurement")
n_comp = fcol("n_kind_complaint")
n_diag = fcol("n_kind_diagnosis")
n_know = fcol("n_kind_knowledge")
share_meas = n_meas / n_evid
true_class = [r["true_class"] for r in frows]

CLASS_GROUPS = {
    "Cardiovascular": ["Heart failure", "Non-ST elevation (NSTEMI) myocardial infarction",
                       "Unspecified atrial fibrillation"],
    "Infection and respiratory": ["Sepsis", "UTI or pyelonephritis", "Skin or soft-tissue infection",
                                  "Lower respiratory disease", "Acute pharyngitis"],
    "Metabolic renal blood": ["Acute kidney failure", "Diabetes mellitus", "Hypokalemia", "Anemia"],
    "Gastrointestinal": ["GI bleed", "Unsp intestnl obst"],
    "Injury and pain": ["Back or spine pain", "Limb injury or pain", "Head injury"],
    "Neuro psychiatric substance": ["Epilepsy", "Major depressive disorder", "Alcohol abuse with intoxication"],
}
REF_GROUP = "Infection and respiratory"
cls2grp = {c: g for g, cs in CLASS_GROUPS.items() for c in cs}
assert all(c in cls2grp for c in true_class), "unmapped class"
group_of = np.array([cls2grp[c] for c in true_class])


def tertiles(name, v, fmt):
    e1, e2 = np.quantile(v, [1 / 3, 2 / 3])
    items = [(f"{name} <= {fmt(e1)}", v <= e1),
             (f"{name} in ({fmt(e1)}, {fmt(e2)}]", (v > e1) & (v <= e2)),
             (f"{name} > {fmt(e2)}", v > e2)]
    return (name, True, [(l, m) for l, m in items if m.sum() > 0])


STRATA = [
    tertiles("n_nodes", n_nodes, lambda x: f"{x:g}"),
    tertiles("n_evidence", n_evid, lambda x: f"{x:g}"),
    tertiles("density", density, lambda x: f"{x:.4f}"),
    ("num_visits", True, [("num_visits = 1", num_visits == 1), ("num_visits = 2", num_visits == 2),
                          ("num_visits >= 3", num_visits >= 3)]),
    ("n_prior_dx", True, [("n_prior_dx = 0", n_prior == 0), ("n_prior_dx 1..3", (n_prior >= 1) & (n_prior <= 3)),
                          ("n_prior_dx >= 4", n_prior >= 4)]),
    tertiles("measurement_share", share_meas, lambda x: f"{x:.3f}"),
    ("n_complaint_nodes", True, [("complaint nodes = 0", n_comp == 0), ("complaint nodes = 1", n_comp == 1),
                                 ("complaint nodes >= 2", n_comp >= 2)]),
    ("n_diagnosis_nodes", True, [("diagnosis nodes = 0", n_diag == 0), ("diagnosis nodes >= 1", n_diag >= 1)]),
    ("class_group", False, [(f"class group: {g}", group_of == g) for g in CLASS_GROUPS]),
]
STRATA = [(f, o, [(l, m) for l, m in items if m.sum() >= 1]) for f, o, items in STRATA]
ROWS = [("all", "all visits", np.ones(N, bool))]
FEAT_ROWS = {}
for f, o, items in STRATA:
    FEAT_ROWS[f] = {"ordered": o, "idx": []}
    for l, m in items:
        FEAT_ROWS[f]["idx"].append(len(ROWS))
        ROWS.append((f, l, m))
M_ALL = np.array([m for _, _, m in ROWS])  # (S, N)
S_ROWS = len(ROWS)

# ----------------------------------------------------------------------------------------------------------------
# 2. Per-visit seed-averaged del per (key, source)
# ----------------------------------------------------------------------------------------------------------------
sums = {(k, s): np.zeros(N) for k in KEYS for s in SOURCES}
cnts = {(k, s): np.zeros(N, int) for k in KEYS for s in SOURCES}
with open(W / "data" / "visit_scores.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        k = r["key"]
        if k not in KEYS or r["ok"] not in ("1", "1.0", "True", "true"):
            continue
        v = r[METRIC]
        if v == "":
            continue
        i = int(r["visit_idx"])
        sums[(k, r["source"])][i] += float(v)
        cnts[(k, r["source"])][i] += 1
V = {ks: np.where(cnts[ks] >= 2, sums[ks] / np.maximum(cnts[ks], 1), np.nan) for ks in sums}
valid_counts = {ks: int((~np.isnan(V[ks])).sum()) for ks in V}
assert all(c > 0 for c in valid_counts.values()), valid_counts
CONF = {k: fcol(f"conf_{k}") for k in KEYS}
print("loaded in %.0fs; visits valid per key/source: %s" % (time.time() - T0, valid_counts))

# bootstrap weights: resample the 1000 patients with replacement
rng = np.random.default_rng(SEED)
Wp = rng.multinomial(P, np.full(P, 1.0 / P), size=NB).astype(float)  # (NB, P)
Wv = Wp[:, pat]  # (NB, N) visit weights

# targets
CMPS = [("gsat_IG", "IG", "gsat", "IG"), ("gsat_Grad", "Grad", "gsat", "Grad"),
        ("gsat_GNNExp", "GNNExp", "gsat", "GNNExp"), ("gsat_builtin", "builtin", "gsat", "builtin"),
        ("gsat_builtin_vs_IG", "builtin", "gsat", "IG"),
        ("gc_IG", "IG", "graphcare", "IG"), ("gc_Grad", "Grad", "graphcare", "Grad"),
        ("gc_GNNExp", "GNNExp", "graphcare", "GNNExp"), ("gc_builtin_vs_Grad", "builtin", "graphcare", "Grad")]
PRIMARY = {"gsat": ["gsat_IG", "gsat_Grad", "gsat_GNNExp", "gsat_builtin"],
           "gc": ["gc_IG", "gc_Grad", "gc_GNNExp", "gc_builtin_vs_Grad"]}
CMP_TITLE = {
    "gsat_IG": "CEI IG - GSAT IG", "gsat_Grad": "CEI Grad - GSAT Grad", "gsat_GNNExp": "CEI GNNExplainer - GSAT GNNExplainer",
    "gsat_builtin": "CEI built-in - GSAT built-in", "gsat_builtin_vs_IG": "CEI built-in - GSAT IG (GSAT's best post-hoc)",
    "gc_IG": "CEI IG - GraphCare IG", "gc_Grad": "CEI Grad - GraphCare Grad", "gc_GNNExp": "CEI GNNExplainer - GraphCare GNNExplainer",
    "gc_builtin_vs_Grad": "CEI built-in - GraphCare Grad (GraphCare's best post-hoc; its built-in is unusable)"}
CEI_KEYS = {"final": "cei_final", "R1": "cei_R1_ls0"}
TARGETS = {}  # name -> (d, cd)
for s in SOURCES:
    TARGETS[f"LS|{s}"] = (V[("cei_final", s)] - V[("cei_R1_ls0", s)], CONF["cei_final"] - CONF["cei_R1_ls0"])
for cname, cs, pk, ps in CMPS:
    for ver, ck in CEI_KEYS.items():
        TARGETS[f"{ver}|{cname}"] = (V[(ck, cs)] - V[(pk, ps)], CONF[ck] - CONF[pk])


# ----------------------------------------------------------------------------------------------------------------
# 3. Stratum statistics: raw and confidence-adjusted means with patient-cluster bootstrap
# ----------------------------------------------------------------------------------------------------------------
def strat_stats(d, cd):
    """Raw mean of d per stratum and confidence-adjusted mean (global slope of d on the confidence difference refit
    in every bootstrap draw; the stratum value is evaluated at the sample-average confidence difference)."""
    valid = ~np.isnan(d) & ~np.isnan(cd)
    d0 = np.where(valid, d, 0.0)
    c0 = np.where(valid, cd, 0.0)
    vf = valid.astype(float)
    Mv = (M_ALL & valid[None, :]).astype(float)
    n = Mv.sum(1).astype(int)
    npat = np.array([len(np.unique(pat[m])) for m in Mv.astype(bool)])

    def run(Wm):
        wv = Wm * vf
        ws = wv.sum(1)
        dbar = (wv @ d0) / ws
        cbar = (wv @ c0) / ws
        cc = c0 - cbar[:, None]
        b = (wv * (d0 - dbar[:, None]) * cc).sum(1) / np.maximum((wv * cc ** 2).sum(1), 1e-12)
        a = d0 - b[:, None] * cc
        den = wv @ Mv.T
        with np.errstate(divide="ignore", invalid="ignore"):
            return (wv * d0) @ Mv.T / den, (wv * a) @ Mv.T / den, b

    raw_pt, adj_pt, b_pt = run(np.ones((1, N)))
    raw_b, adj_b, b_b = run(Wv)
    return {"n": n, "npat": npat, "raw": raw_pt[0], "adj": adj_pt[0], "raw_b": raw_b, "adj_b": adj_b,
            "slope": b_pt[0], "slope_b": b_b}


def ci(boot, j=None):
    x = boot if j is None else boot[:, j]
    return np.percentile(x, [2.5, 97.5], axis=0)


def fmt(pt, lo, hi):
    return f"{pt:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def cell(st, key, j):
    lo, hi = ci(st[key + "_b"], j)
    return fmt(st[key][j], lo, hi)


def excl(st, key, j):
    lo, hi = ci(st[key + "_b"], j)
    return 1 if lo > 0 else (-1 if hi < 0 else 0)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return out


STATS = {}
for name, (d, cd) in TARGETS.items():
    STATS[name] = strat_stats(d, cd)
print("stratum statistics done in %.0fs (%d targets, %d strata rows)" % (time.time() - T0, len(TARGETS), S_ROWS))

LINES = []
add = LINES.append
add("# Step 5 tables (generated by comparison/explain/graph_types/step5_label_smoothing.py)")
add("")
add("Test fold, descriptive, nothing selected. `del` = deletion_mean_change_vs_random, per visit averaged over the 3 seeds "
    "(a visit counts for a key/source when ok in >= 2 seeds). Differences are paired on the same visit; 95% CIs are "
    f"patient-cluster bootstrap percentiles ({NB} draws, seed {SEED}, resampling the {P} patients). `raw` = plain mean "
    "difference; `adj` = confidence-adjusted: the global slope of the difference on the confidence difference "
    "(predicted-class probability of the two models, seed-averaged) is refit in every draw and the stratum value is "
    "evaluated at the sample-average confidence difference. Cells under 10 visits are not printed; cells under 30 "
    "visits are shown with n only (not interpreted).")
add("")
add("## Bin rules (fixed before any difference was looked at)")
add("")
for f, o, items in STRATA:
    add(f"- `{f}` ({'ordered' if o else 'unordered'}): " + "; ".join(f"{l} (n={int(m.sum())})" for l, m in items))
add("")
add("Class groups: " + "; ".join(f"{g} = {', '.join(cs)}" for g, cs in CLASS_GROUPS.items()) + ".")
add("")
add("Visits with a usable `del` per key/source: " + "; ".join(f"{k}/{s} {valid_counts[(k, s)]}" for k in KEYS for s in SOURCES
                                                              if valid_counts[(k, s)] != N) + (
    "" if any(valid_counts[(k, s)] != N for k in KEYS for s in SOURCES) else f"all {N}") + f" of {N}.")
add("")


def rows_for(names, with_contrast=True):
    """Row iterator: (label, row index or contrast spec)."""
    out = []
    for j, (f, l, m) in enumerate(ROWS):
        out.append((l, j))
    return out


def feature_contrasts(st_list, getter):
    """Ordered features: highest bin minus lowest bin from the same bootstrap draws (raw and adj)."""
    rows = []
    for f, info in FEAT_ROWS.items():
        if not info["ordered"] or len(info["idx"]) < 2:
            continue
        lo_j, hi_j = info["idx"][0], info["idx"][-1]
        rows.append((f, lo_j, hi_j))
    return rows


def contrast_cell(st, key, lo_j, hi_j):
    if min(st["n"][lo_j], st["n"][hi_j]) < MIN_INTERP:
        return "n<30, not interpreted"
    bt = st[key + "_b"][:, hi_j] - st[key + "_b"][:, lo_j]
    lo, hi = np.percentile(bt, [2.5, 97.5])
    return fmt(st[key][hi_j] - st[key][lo_j], lo, hi)


def monotone(st, key, idx):
    v = [st[key][j] for j in idx if st["n"][j] >= MIN_INTERP]
    if len(v) < 3:
        return "n/a"
    inc = all(v[i] <= v[i + 1] for i in range(len(v) - 1))
    dec = all(v[i] >= v[i + 1] for i in range(len(v) - 1))
    return "increasing" if inc else ("decreasing" if dec else "not monotone")


# ----------------------------------------------------------------------------------------------------------------
# A. Where the LS cost (cei_final - cei_R1_ls0) concentrates
# ----------------------------------------------------------------------------------------------------------------
add("## A. Label-smoothing cost: del(cei_final) - del(cei_R1_ls0), by graph type")
add("")
add("Negative = label smoothing lowers the score. `share of cost` = the stratum's summed cost divided by the summed "
    "cost over all visits (strata of one feature add to 100%; a share above the stratum's visit share means the cost "
    "concentrates there). Confidence change (cei_final - cei_R1_ls0 predicted-class probability), all visits: "
    f"{np.nanmean(CONF['cei_final'] - CONF['cei_R1_ls0']):+.4f}.")
add("")
LS_SUMMARY = {}
for s in SOURCES:
    st = STATS[f"LS|{s}"]
    d = TARGETS[f"LS|{s}"][0]
    tot = np.nansum(d)
    rows = []
    for j, (f, l, m) in enumerate(ROWS):
        if st["n"][j] < MIN_PRINT:
            continue
        vm = m & ~np.isnan(d)
        share = f"{100 * np.nansum(d[vm]) / tot:.0f}%"
        vshare = f"{100 * vm.sum() / (~np.isnan(d)).sum():.0f}%"
        if st["n"][j] < MIN_INTERP:
            rows.append([l, st["n"][j], st["npat"][j], vshare, "n<30, not interpreted", "", ""])
        else:
            rows.append([l, st["n"][j], st["npat"][j], vshare, cell(st, "raw", j), cell(st, "adj", j), share])
    add(f"### A.{SOURCES.index(s) + 1} {s}")
    add("")
    LINES.extend(md_table(["stratum", "n visits", "n patients", "visit share", "raw mean [95% CI]",
                           "adj mean [95% CI]", "share of cost"], rows))
    add("")
    crow = []
    for f, lo_j, hi_j in feature_contrasts(None, None):
        crow.append([f"{f}: highest bin - lowest bin", contrast_cell(st, "raw", lo_j, hi_j),
                     contrast_cell(st, "adj", lo_j, hi_j), monotone(st, "raw", FEAT_ROWS[f]["idx"])])
    LINES.extend(md_table(["ordered feature", "raw contrast [95% CI]", "adj contrast [95% CI]", "raw trend over bins"], crow))
    add("")
    add(f"Slope of the LS cost on the confidence change (per unit probability): {st['slope']:+.3f} "
        f"[{ci(st['slope_b'])[0]:+.3f}, {ci(st['slope_b'])[1]:+.3f}].")
    add("")

# ----------------------------------------------------------------------------------------------------------------
# B. Do the strata where CEI beats GSAT / GraphCare agree for cei_final and cei_R1_ls0?
# ----------------------------------------------------------------------------------------------------------------


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if len(a) < 4:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


add("## B. CEI minus partner by stratum, cei_final vs cei_R1_ls0")
add("")
add("Positive = CEI has the higher `del` (better). `agreement`: both+ / both- = same sign and both CIs exclude 0; same "
    "sign = same sign, at least one CI includes 0; differs = opposite signs.")
add("")
AGREE_SUM = []
SUP = {}  # (partner, row j) -> dict
for ci_, (cname, cs, pk, ps) in enumerate(CMPS):
    sf, sr = STATS[f"final|{cname}"], STATS[f"R1|{cname}"]
    rows = []
    xs, ys = [], []
    n_signs = n_agree = n_both = 0
    for j, (f, l, m) in enumerate(ROWS):
        if sf["n"][j] < MIN_PRINT:
            continue
        if sf["n"][j] < MIN_INTERP:
            rows.append([l, sf["n"][j], sf["npat"][j], "n<30, not interpreted", "", "", "", ""])
            continue
        ef, er = excl(sf, "raw", j), excl(sr, "raw", j)
        sgf, sgr = np.sign(sf["raw"][j]), np.sign(sr["raw"][j])
        if ef != 0 and ef == er:
            agree = "both+" if ef > 0 else "both-"
        elif sgf == sgr:
            agree = "same sign"
        else:
            agree = "differs"
        rows.append([l, sf["n"][j], sf["npat"][j], cell(sf, "raw", j), cell(sr, "raw", j), cell(sf, "adj", j),
                     cell(sr, "adj", j), agree])
        if j > 0:
            xs.append(sf["raw"][j])
            ys.append(sr["raw"][j])
            n_signs += 1
            n_agree += int(sgf == sgr)
            n_both += int(ef != 0 and ef == er)
    add(f"### B.{ci_ + 1} {CMP_TITLE[cname]}")
    add("")
    LINES.extend(md_table(["stratum", "n visits", "n patients", "cei_final raw", "cei_R1 raw", "cei_final adj",
                           "cei_R1 adj", "agreement (raw)"], rows))
    add("")
    crow = []
    for f, lo_j, hi_j in feature_contrasts(None, None):
        crow.append([f"{f}: highest - lowest bin", contrast_cell(sf, "raw", lo_j, hi_j), contrast_cell(sr, "raw", lo_j, hi_j),
                     contrast_cell(sf, "adj", lo_j, hi_j), contrast_cell(sr, "adj", lo_j, hi_j)])
    LINES.extend(md_table(["ordered feature", "cei_final raw", "cei_R1 raw", "cei_final adj", "cei_R1 adj"], crow))
    add("")
    AGREE_SUM.append([CMP_TITLE[cname], n_signs, n_agree, n_both, f"{spearman(np.array(xs), np.array(ys)):+.2f}"])
    for part, names in PRIMARY.items():
        if cname in names:
            for j in range(S_ROWS):
                if sf["n"][j] >= MIN_INTERP:
                    SUP.setdefault((part, j), {})[cname] = (excl(sf, "raw", j), excl(sr, "raw", j),
                                                           excl(sf, "adj", j), excl(sr, "adj", j))

add("### B.10 Agreement summary (strata of all features, excluding the all-visits row, n >= 30)")
add("")
LINES.extend(md_table(["comparison", "strata", "same sign cei_final / R1", "same sign and both CIs exclude 0",
                       "Spearman rho of stratum means"], AGREE_SUM))
add("")
add("### B.11 Strata that meet the 'supported' rule fixed before the analysis")
add("")
add("Rule (applied to the four explainers of each partner: IG, Grad, GNNExplainer, built-in; for GraphCare the built-in "
    "row is CEI built-in vs GraphCare Grad): the CI of the stratum difference excludes 0 for BOTH cei_final and cei_R1 "
    "with the same sign in at least two explainers. Columns count explainers with CI excluding 0 as (+ / -) for "
    "cei_final and cei_R1; `supported raw` uses the raw CIs, `supported adj` the confidence-adjusted CIs.")
add("")
for part, title in (("gsat", "CEI vs GSAT"), ("gc", "CEI vs GraphCare")):
    rows = []
    for j, (f, l, m) in enumerate(ROWS):
        if (part, j) not in SUP:
            continue
        rec = SUP[(part, j)]
        def count(idx, sign):
            return sum(1 for c in PRIMARY[part] if rec[c][idx] == sign)
        def sup(i_f, i_r):
            for sign in (1, -1):
                if count(i_f, sign) >= 2 and count(i_r, sign) >= 2:
                    return "supported CEI better" if sign > 0 else "supported CEI worse"
            return "no"
        rows.append([l, f"{count(0, 1)} / {count(0, -1)}", f"{count(1, 1)} / {count(1, -1)}", sup(0, 1), sup(2, 3)])
    add(f"**{title}**")
    add("")
    LINES.extend(md_table(["stratum", "cei_final (+ / -)", "cei_R1 (+ / -)", "supported raw", "supported adj"], rows))
    add("")

# ----------------------------------------------------------------------------------------------------------------
# C. Multivariable linear model (patient-cluster bootstrapped WLS)
# ----------------------------------------------------------------------------------------------------------------
assert (n_diag == n_prior).all()  # diagnosis nodes are exactly the prior-diagnosis nodes: keep one column (n_prior_dx)
BASE_NAMES = ["n_nodes", "density", "num_visits", "n_prior_dx", "measurement_share", "n_complaint_nodes",
              "n_knowledge_nodes"]
BASE_RAW = np.column_stack([n_nodes, density, num_visits, n_prior, share_meas, n_comp, n_know])
BASE_Z = (BASE_RAW - BASE_RAW.mean(0)) / BASE_RAW.std(0)
GROUP_NAMES = [g for g in CLASS_GROUPS if g != REF_GROUP]
GROUP_DUM = np.column_stack([(group_of == g).astype(float) for g in GROUP_NAMES])
COEF_NAMES = ["intercept"] + [f"z_{n}" for n in BASE_NAMES] + ["z_conf_diff"] + [f"class: {g} (vs {REF_GROUP})" for g in GROUP_NAMES]


def design(cd):
    valid_c = ~np.isnan(cd)
    z = np.where(valid_c, (cd - np.nanmean(cd)) / np.nanstd(cd), 0.0)
    return np.column_stack([np.ones(N), BASE_Z, z, GROUP_DUM])


def wls(X, y, w):
    XtW = X.T * w
    return np.linalg.solve(XtW @ X + 1e-8 * np.eye(X.shape[1]), XtW @ y)


def fit_boot(d, cd):
    valid = ~np.isnan(d) & ~np.isnan(cd)
    X = design(cd)
    y0 = np.where(valid, d, 0.0)
    vf = valid.astype(float)
    b_pt = wls(X, y0, vf)
    res = y0 - X @ b_pt
    r2 = 1 - (vf * res ** 2).sum() / (vf * (y0 - (vf * y0).sum() / vf.sum()) ** 2).sum()
    boot = np.empty((NB, X.shape[1]))
    for b in range(NB):
        boot[b] = wls(X, y0, Wv[b] * vf)
    return b_pt, boot, r2, int(valid.sum())


# variance inflation factors on the representative design (cei_final vs GSAT IG)
Xr = design(TARGETS["final|gsat_IG"][1])[:, 1:]
vif = []
for c in range(Xr.shape[1]):
    others = np.column_stack([np.ones(N), np.delete(Xr, c, axis=1)])
    beta = np.linalg.lstsq(others, Xr[:, c], rcond=None)[0]
    r2c = 1 - ((Xr[:, c] - others @ beta) ** 2).sum() / ((Xr[:, c] - Xr[:, c].mean()) ** 2).sum()
    vif.append(1 / max(1 - r2c, 1e-9))
add("## C. Multivariable linear model (weighted least squares, patient-cluster bootstrap)")
add("")
add("Target = the paired per-visit difference (CEI - partner, or cei_final - cei_R1 for the LS cost). Predictors: seven "
    "standardized graph features (n_nodes, density, num_visits, n_prior_dx, measurement share of evidence nodes, number "
    "of complaint nodes, number of knowledge nodes; the number of diagnosis nodes equals n_prior_dx in every visit "
    "and is not entered twice), the standardized confidence difference of "
    f"the two models, and five class-group indicators (reference: {REF_GROUP}; indicator coefficients are offsets in `del`, "
    "the others are per 1 SD). Coefficients are refit on each of the "
    f"{NB} patient-cluster draws; `*` marks a 95% CI that excludes 0. Variance inflation factors (representative design, cei_final vs "
    "GSAT IG): " + ", ".join(f"{n} {v:.1f}" for n, v in zip(COEF_NAMES[1:], vif)) + ". n_nodes, density, measurement share "
    "and n_knowledge_nodes move together; read their coefficients jointly, not one by one.")
add("")
COEF = {}
for name, (d, cd) in TARGETS.items():
    COEF[name] = fit_boot(d, cd)
print("linear models done in %.0fs" % (time.time() - T0))


def coef_cell(res, c):
    b_pt, boot, _, _ = res
    lo, hi = np.percentile(boot[:, c], [2.5, 97.5])
    star = "*" if (lo > 0 or hi < 0) else ""
    return fmt(b_pt[c], lo, hi) + star


rows = []
for c, nme in enumerate(COEF_NAMES):
    rows.append([nme] + [coef_cell(COEF[f"LS|{s}"], c) for s in SOURCES])
rows.append(["R^2 (in sample) / n visits"] + [f"{COEF[f'LS|{s}'][2]:.3f} / {COEF[f'LS|{s}'][3]}" for s in SOURCES])
add("### C.1 LS cost, del(cei_final) - del(cei_R1_ls0)")
add("")
LINES.extend(md_table(["predictor"] + SOURCES, rows))
add("")
for ci_, (cname, cs, pk, ps) in enumerate(CMPS):
    rows = []
    for c, nme in enumerate(COEF_NAMES):
        rows.append([nme, coef_cell(COEF[f"final|{cname}"], c), coef_cell(COEF[f"R1|{cname}"], c)])
    rows.append(["R^2 (in sample) / n visits", f"{COEF[f'final|{cname}'][2]:.3f} / {COEF[f'final|{cname}'][3]}",
                 f"{COEF[f'R1|{cname}'][2]:.3f} / {COEF[f'R1|{cname}'][3]}"])
    add(f"### C.{ci_ + 2} {CMP_TITLE[cname]}")
    add("")
    LINES.extend(md_table(["predictor", "cei_final", "cei_R1_ls0"], rows))
    add("")

# ----------------------------------------------------------------------------------------------------------------
# D. Depth-2 rule summary checked on a patient-level held-out split
# ----------------------------------------------------------------------------------------------------------------
TREE_NAMES = BASE_NAMES + ["conf_diff"] + [f"class_{g}" for g in CLASS_GROUPS]
MIN_LEAF = 50
N_SPLITS = 5


def tree_matrix(cd):
    return np.column_stack([BASE_RAW, cd] + [(group_of == g).astype(float) for g in CLASS_GROUPS])


def best_split(X, y, idx, cands):
    ys = y[idx]
    base = ((ys - ys.mean()) ** 2).sum()
    best = (0.0, None, None)
    for f in range(X.shape[1]):
        x = X[idx, f]
        for t in cands[f]:
            left = x <= t
            nl = int(left.sum())
            if nl < MIN_LEAF or len(x) - nl < MIN_LEAF:
                continue
            sse = ((ys[left] - ys[left].mean()) ** 2).sum() + ((ys[~left] - ys[~left].mean()) ** 2).sum()
            if base - sse > best[0] + 1e-12:
                best = (base - sse, f, t)
    return best


def grow(X, y, train_idx):
    cands = []
    for f in range(X.shape[1]):
        q = np.unique(np.quantile(X[train_idx, f], np.linspace(0.05, 0.95, 19)))
        cands.append(q)
    leaves = []  # list of condition lists [(f, t, "<=" or ">")]

    def rec(idx, conds, depth):
        if depth < 2:
            gain, f, t = best_split(X, y, idx, cands)
            if f is not None:
                left = X[idx, f] <= t
                rec(idx[left], conds + [(f, t, "<=")], depth + 1)
                rec(idx[~left], conds + [(f, t, ">")], depth + 1)
                return
        leaves.append(conds)

    rec(train_idx, [], 0)
    return leaves


def leaf_mask(X, conds):
    m = np.ones(len(X), bool)
    for f, t, side in conds:
        m &= (X[:, f] <= t) if side == "<=" else (X[:, f] > t)
    return m


def rule_text(conds):
    if not conds:
        return "all visits"
    return " and ".join(f"{TREE_NAMES[f]} {side} {t:.4g}" for f, t, side in conds)


def boot_leaf(y, m, pats, seed):
    up = np.unique(pats)
    local = np.searchsorted(up, pats)
    r = np.random.default_rng(seed)
    wb = r.multinomial(len(up), np.full(len(up), 1.0 / len(up)), size=NB).astype(float)
    s = np.bincount(local, weights=np.where(m, y, 0.0), minlength=len(up))
    c = np.bincount(local, weights=m.astype(float), minlength=len(up))
    den = wb @ c
    ok = den > 0
    bm = (wb[ok] @ s) / den[ok]
    return np.percentile(bm, [2.5, 97.5])


add("## D. Depth <= 2 rule summary, min leaf 50 visits, checked on a patient-level held-out split")
add("")
add("For each target the visits are split by patient into a training half and a held-out half (500 patients each; "
    f"{N_SPLITS} random splits, seeds {SEED}+0..{N_SPLITS - 1}). A greedy least-squares tree (candidate cuts at the 5%..95% "
    "quantiles of the training half) is grown on the training half; leaf means are then recomputed on the held-out half "
    "with patient-cluster CIs. Held-out R^2 = 1 - SSE(tree leaf means from the training half) / SSE(training-half overall "
    "mean), both on the held-out visits (0 = no better than a constant, negative = worse). Descriptive: rules are "
    "not used to select anything. Leaf table shown for split 0; `R^2` lists all splits.")
add("")
TREE_SUMMARY = []
for name, (d, cd) in TARGETS.items():
    valid = ~np.isnan(d) & ~np.isnan(cd)
    X = tree_matrix(np.where(np.isnan(cd), 0.0, cd))
    y = np.where(valid, d, 0.0)
    r2s = []
    leaf_rows = []
    nleaves0 = 0
    for sp in range(N_SPLITS):
        r = np.random.default_rng(SEED + sp)
        perm = r.permutation(P)
        train_p = np.zeros(P, bool)
        train_p[perm[: P // 2]] = True
        tr = np.where(valid & train_p[pat])[0]
        te = np.where(valid & ~train_p[pat])[0]
        leaves = grow(X, y, tr)
        sse_t = sse_c = 0.0
        cmean = y[tr].mean()
        for conds in leaves:
            mtr = leaf_mask(X[tr], conds)
            lm = y[tr][mtr].mean()
            mte = leaf_mask(X[te], conds)
            sse_t += ((y[te][mte] - lm) ** 2).sum()
        sse_c = ((y[te] - cmean) ** 2).sum()
        r2s.append(1 - sse_t / sse_c)
        if sp == 0:
            nleaves0 = len(leaves)
            for conds in leaves:
                mtr = leaf_mask(X[tr], conds)
                mte = leaf_mask(X[te], conds)
                ntr, nte = int(mtr.sum()), int(mte.sum())
                if nte < MIN_PRINT:
                    continue
                if nte < MIN_INTERP:
                    leaf_rows.append([rule_text(conds), ntr, f"{y[tr][mtr].mean():+.4f}", nte, "n<30, not interpreted", ""])
                    continue
                lo, hi = boot_leaf(y[te], mte, pat[te], SEED + 100)
                same = "yes" if np.sign(y[tr][mtr].mean()) == np.sign(y[te][mte].mean()) else "no"
                leaf_rows.append([rule_text(conds), ntr, f"{y[tr][mtr].mean():+.4f}", nte,
                                  fmt(y[te][mte].mean(), lo, hi), same])
    TREE_SUMMARY.append((name, nleaves0, r2s))
    title = "LS cost, " + name.split("|")[1] if name.startswith("LS|") else (
        ("cei_final" if name.startswith("final") else "cei_R1_ls0") + ": " + CMP_TITLE[name.split("|")[1]])
    add(f"### D. {title}")
    add("")
    add("R^2 held-out over the splits: " + ", ".join(f"{x:+.3f}" for x in r2s) + f" (splits with R^2 > 0: {sum(x > 0 for x in r2s)} of {N_SPLITS}).")
    add("")
    LINES.extend(md_table(["rule (split 0)", "n train", "train mean", "n held-out", "held-out mean [95% CI]",
                           "same sign as train"], leaf_rows))
    add("")
print("trees done in %.0fs" % (time.time() - T0))

add("### D.summary Held-out R^2 of the depth-2 trees")
add("")
LINES.extend(md_table(["target", "leaves (split 0)", "median R^2", "min R^2", "max R^2", "splits with R^2 > 0"],
                      [[n.replace("|", ": "), nl, f"{np.median(r):+.3f}", f"{min(r):+.3f}", f"{max(r):+.3f}",
                        f"{sum(x > 0 for x in r)} of {N_SPLITS}"] for n, nl, r in TREE_SUMMARY]))
add("")
add("## Command")
add("")
add("`env -u PYTHONPATH "
    "PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 comparison/explain/graph_types/step5_label_smoothing.py`")
OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
print("wrote %s (%d lines) in %.0fs" % (OUT, len(LINES), time.time() - T0))
