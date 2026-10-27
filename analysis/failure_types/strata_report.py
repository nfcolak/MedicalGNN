"""Failure-type analysis, step 2: how good are the explanations, by kind of visit.

    PYTHONPATH=. FT_K=20 FT_EXPLAIN_ROOT=... FT_SUMMARY_DIR=... FT_WORK=... python -m analysis.failure_types.strata_report

Reads the two per-visit tables of step 1 (`data/visit_features.csv`, `data/visit_scores.csv`) and writes
`out/strata-report.md` (aggregates only) and `out/strata-cells.csv` (every cell and metric).

This replaces steps 2 to 4 of Necati's graph-type analysis (`comparison/explain/graph_types`) for the final runs. Those
steps compare CEI-GNN with a comparator across strata and call a stratum "supported" only when the result repeats in a
second CEI model (the label-smoothing variant R1); the final runs have one CEI-GNN, so that rule does not exist. This step
keeps his rules where they apply and states the rest:

* Unit: the visit. A visit counts for a (method, source) when the source is ok in at least 2 of the 3 seeds; its value is
  the mean over its ok seeds. A paired difference needs both sides.
* Uncertainty: patient-cluster bootstrap (patients resampled with replacement, 2,000 draws, seed 20261025, one shared
  draw matrix), 95% percentile interval.
* Sample sizes: a cell with fewer than 10 visits is not printed; 10 to 29 is printed and marked (dagger), never
  interpreted; a paired difference is shown in bold only when n >= 30 and its interval excludes 0.
* Strata. Shared by all methods: diagnosis group and true class (by class NAME, so they hold at every K; classes outside
  the 20 grouped ones form one "other" stratum), number of visits, graph size (tertiles of the node count), edge-less
  graphs, labs present, prior diagnoses present. Per method: correct prediction (at least 2 of 3 seeds predict the true
  class), seed-stable prediction (all 3 seeds predict the same class), confidence quintiles. For CEI-GNN minus a
  comparator: the joint correctness pattern, quintiles of the confidence difference, and the 3 x 3 grid of both
  confidences (the confidence-matched comparison).
* Scores: `del` = deletion_mean_change_vs_random (headline, higher is better); the csv also holds insertion, fidelity+,
  fidelity- and sparsity. The raw differences are NOT adjusted for confidence (Necati's step 3 regression adjustment is
  not repeated); the confidence strata and the grid are the control, read them before any ranking.
* Descriptive and exploratory: nothing is selected or tuned from these numbers.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from analysis.failure_types._paths import FOLD, K, WORK
from comparison.explain.similar_patients import GROUP_OF_NAME

DRAWS, BOOT_SEED = 2000, 20261025
MIN_PRINT, MIN_INTERPRET = 10, 30
METHODS = {"cei_final": "CEI-GNN", "gsat": "GSAT", "graphcare": "GraphCare", "protgnn": "ProtGNN"}
COMPARATORS = ("gsat", "graphcare", "protgnn")
SOURCES = ("builtin", "IG", "Grad", "GNNExp")
TABLE_SOURCES = ("builtin", "IG")
METRICS = {"del": "deletion_mean_change_vs_random", "ins": "insertion_mean_change_vs_random",
           "fid+": "fidelity_plus", "fid-": "fidelity_minus", "sparsity": "mass90_sparsity"}
HEADLINE = "del"


# ---------------------------------------------------------------------------------------------- loading
def load_features(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda r: int(r["visit_idx"]))
    out = {"true_class": np.array([r["true_class"] for r in rows])}
    for name in rows[0]:
        if name != "true_class":
            out[name] = np.array([float(r[name]) for r in rows])
    out["visit_idx"] = out["visit_idx"].astype(int)
    if not (out["visit_idx"] == np.arange(len(rows))).all():
        raise SystemExit("visit_idx is not 0..V-1")
    return out


def load_values(path: Path, n_visits: int, keys, seeds_needed: int = 2) -> dict:
    """{(key, source, metric): per-visit mean over the ok seeds, nan when ok in fewer than `seeds_needed`}."""
    sums, counts = {}, {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["key"] not in keys or row["ok"] != "1":
                continue
            i = int(row["visit_idx"])
            for short, column in METRICS.items():
                if row.get(column, "") == "":
                    continue
                cell = (row["key"], row["source"], short)
                if cell not in sums:
                    sums[cell], counts[cell] = np.zeros(n_visits), np.zeros(n_visits)
                sums[cell][i] += float(row[column])
                counts[cell][i] += 1
    return {cell: np.where(counts[cell] >= seeds_needed, sums[cell] / np.maximum(counts[cell], 1), np.nan)
            for cell in sums}


# ---------------------------------------------------------------------------------------------- strata
def bins_by_quantile(x: np.ndarray, n: int, label: str):
    edges = np.unique(np.round(np.quantile(x, np.arange(1, n) / n), 12))
    index = np.searchsorted(edges, x, side="right")
    labels = []
    for i in range(len(edges) + 1):
        if i == 0:
            labels.append(f"{label} Q1 (<{edges[0]:.3g})" if len(edges) else f"{label} all")
        elif i == len(edges):
            labels.append(f"{label} Q{i + 1} (>={edges[-1]:.3g})")
        else:
            labels.append(f"{label} Q{i + 1} [{edges[i - 1]:.3g}, {edges[i]:.3g})")
    return index, labels


def axis(key: str, title: str, labels: list[str], masks: list[np.ndarray]) -> dict:
    return {"key": key, "title": title, "labels": labels, "masks": [np.asarray(m, bool) for m in masks]}


def shared_axes(feat: dict) -> list[dict]:
    names = feat["true_class"]
    groups = [GROUP_OF_NAME.get(n, "other (no group)") for n in names]
    group_labels = sorted(set(groups), key=lambda g: (g.startswith("other"), g))
    classes = sorted(set(names))
    visits = feat["num_visits"]
    size_index, size_labels = bins_by_quantile(feat["n_nodes"], 3, "nodes")
    return [
        axis("group", "Diagnosis group", group_labels, [np.array([g == lab for g in groups]) for lab in group_labels]),
        axis("class", "True class", classes, [names == c for c in classes]),
        axis("visits", "Number of ED visits in the record", ["1", "2", "3", "4 or more"],
             [visits == 1, visits == 2, visits == 3, visits >= 4]),
        axis("size", "Graph size (node count tertiles)", size_labels, [size_index == i for i in range(len(size_labels))]),
        axis("edges", "Edges", ["no edges", "has edges"], [feat["n_edges"] == 0, feat["n_edges"] > 0]),
        axis("labs", "Laboratory evidence", ["no lab measurement", "has lab measurements"],
             [feat["n_kind_measurement"] == 0, feat["n_kind_measurement"] > 0]),
        axis("prior_dx", "Prior diagnoses", ["none", "has prior diagnoses"], [feat["n_prior_dx"] == 0, feat["n_prior_dx"] > 0]),
    ]


def method_axes(feat: dict, key: str) -> list[dict]:
    correct = feat["correct_" + key] > 0.5
    stable = feat["pred_agree_" + key] == 1
    index, labels = bins_by_quantile(feat["conf_" + key], 5, "confidence")
    return [axis("correct", "Prediction correct (at least 2 of 3 seeds)", ["correct", "incorrect"], [correct, ~correct]),
            axis("stable", "Seed-stable prediction (all 3 seeds agree)", ["stable", "not stable"], [stable, ~stable]),
            axis("conf", "Confidence of the method (quintiles of its seed-mean predicted-class probability)", labels,
                 [index == i for i in range(len(labels))])]


def pair_axes(feat: dict, cei: str, other: str) -> list[dict]:
    c1, c2 = feat["correct_" + cei] > 0.5, feat["correct_" + other] > 0.5
    pattern = axis("pattern", "Correctness pattern", ["both correct", "CEI-GNN only", f"{METHODS[other]} only", "both wrong"],
                   [c1 & c2, c1 & ~c2, ~c1 & c2, ~c1 & ~c2])
    diff = feat["conf_" + cei] - feat["conf_" + other]
    d_index, d_labels = bins_by_quantile(diff, 5, "conf diff")
    i1, l1 = bins_by_quantile(feat["conf_" + cei], 3, "CEI-GNN conf")
    i2, l2 = bins_by_quantile(feat["conf_" + other], 3, f"{METHODS[other]} conf")
    grid_labels = [f"{a} x {b}" for a in l1 for b in l2]
    grid_masks = [(i1 == x) & (i2 == y) for x in range(len(l1)) for y in range(len(l2))]
    return [pattern,
            axis("conf_diff", "Confidence difference CEI-GNN minus comparator (quintiles)", d_labels,
                 [d_index == i for i in range(len(d_labels))]),
            axis("grid", "Confidence-matched grid (tertile x tertile)", grid_labels, grid_masks)]


# ---------------------------------------------------------------------------------------------- statistics
class Bootstrap:
    """Patient-cluster bootstrap with one shared draw matrix; row 0 of the weights is the original sample."""

    def __init__(self, patient: np.ndarray):
        self.patient = patient.astype(int)
        n_patients = int(self.patient.max()) + 1
        rng = np.random.default_rng(BOOT_SEED)
        draws = rng.multinomial(n_patients, np.full(n_patients, 1.0 / n_patients), size=DRAWS).astype(float)
        self.weights = np.vstack([np.ones((1, len(self.patient))), draws[:, self.patient]])

    def stratified(self, values: np.ndarray, axis_: dict) -> list[dict]:
        """Mean of `values` (nan = missing) in every stratum of the axis, with its interval and sizes."""
        valid = ~np.isnan(values)
        v = np.where(valid, values, 0.0)
        z = np.stack(axis_["masks"], axis=1).astype(float)
        w = self.weights * valid[None, :]
        with np.errstate(all="ignore"):
            num, den = (w * v[None, :]) @ z, w @ z
            mean = num / den
        cells = []
        for j in range(z.shape[1]):
            in_cell = valid & axis_["masks"][j]
            n = int(in_cell.sum())
            if n == 0 or np.isnan(mean[0, j]):
                cells.append({"n": n, "patients": 0, "est": None, "lo": None, "hi": None})
                continue
            cells.append({"n": n, "patients": int(len(np.unique(self.patient[in_cell]))), "est": float(mean[0, j]),
                          "lo": float(np.nanpercentile(mean[1:, j], 2.5)), "hi": float(np.nanpercentile(mean[1:, j], 97.5))})
        return cells


def fmt(cell: dict, *, bold_if_excluded: bool = False) -> str:
    if cell["n"] < MIN_PRINT or cell["est"] is None:
        return "-"
    text = f"{cell['est']:+.3f} [{cell['lo']:+.3f}, {cell['hi']:+.3f}]"
    if cell["n"] < MIN_INTERPRET:
        return text + " †"
    if bold_if_excluded and (cell["lo"] > 0 or cell["hi"] < 0):
        return f"**{text}**"
    return text


def table(header: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"] + \
           ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


# ---------------------------------------------------------------------------------------------- report
def build(feat: dict, values: dict):
    """(markdown lines, csv rows)."""
    patient = feat["patient_idx"]
    boot = Bootstrap(patient)
    n_visits, n_patients = len(patient), int(patient.max()) + 1
    csv_rows, lines = [], []

    def record(kind, ax, stratum, who, source, metric, cell):
        csv_rows.append({"table": kind, "axis": ax["key"], "stratum": stratum, "who": who, "source": source,
                         "metric": metric, "n_visits": cell["n"], "n_patients": cell["patients"],
                         "est": cell["est"], "lo": cell["lo"], "hi": cell["hi"]})

    lines += [f"# Failure-type analysis, K = {K}, {FOLD} fold", "",
              f"{n_visits} visits of {n_patients} patients, 4 methods x 3 seeds. Rules: see the module docstring of "
              "`analysis/failure_types/strata_report.py`. Cells under 10 visits are not printed; 10 to 29 are marked † and not "
              "interpreted; in difference tables bold = n >= 30 and the 95% patient-cluster interval excludes 0. Scores are "
              "`del` = deletion_mean_change_vs_random (higher is better); the other metrics are in `strata-cells.csv`. "
              "Differences are raw, not adjusted for confidence: read the confidence strata and the grid first.", ""]

    lines += ["## 1. Explanation score by kind of visit (all methods, same visits)", ""]
    for ax in shared_axes(feat):
        lines += [f"### {ax['title']}", ""]
        columns = [(m, s) for m in METHODS for s in TABLE_SOURCES]
        cells = {(m, s): boot.stratified(values.get((m, s, HEADLINE), np.full(n_visits, np.nan)), ax) for m, s in columns}
        rows = []
        for j, label in enumerate(ax["labels"]):
            first = next(iter(cells.values()))[j]
            vis = int((ax["masks"][j]).sum())
            if vis == 0:
                continue
            rows.append([label, vis, len(np.unique(patient[ax["masks"][j]]))] + [fmt(cells[c][j]) for c in columns])
        lines += table(["stratum", "visits", "patients"] + [f"{METHODS[m]} {s}" for m, s in columns], rows) + [""]
        for m in METHODS:
            for s in SOURCES:
                for metric in METRICS:
                    arr = values.get((m, s, metric))
                    if arr is None:
                        continue
                    for j, cell in enumerate(boot.stratified(arr, ax)):
                        record("score", ax, ax["labels"][j], METHODS[m], s, metric, cell)

    lines += ["## 2. Explanation score by the method's own prediction (correct, stable, confidence)", ""]
    for key in METHODS:
        for ax in method_axes(feat, key):
            lines += [f"### {METHODS[key]}: {ax['title']}", ""]
            cells = {s: boot.stratified(values.get((key, s, HEADLINE), np.full(n_visits, np.nan)), ax) for s in SOURCES}
            rows = []
            for j, label in enumerate(ax["labels"]):
                vis = int(ax["masks"][j].sum())
                if vis:
                    rows.append([label, vis, len(np.unique(patient[ax["masks"][j]]))] + [fmt(cells[s][j]) for s in SOURCES])
            lines += table(["stratum", "visits", "patients"] + list(SOURCES), rows) + [""]
            for s in SOURCES:
                for metric in METRICS:
                    arr = values.get((key, s, metric))
                    if arr is not None:
                        for j, cell in enumerate(boot.stratified(arr, ax)):
                            record("score_own", ax, ax["labels"][j], METHODS[key], s, metric, cell)

    lines += ["## 3. CEI-GNN minus each comparator, same source, same visits", "",
              "Positive = CEI-GNN scores higher. A visit needs both sides ok in at least 2 of 3 seeds.", ""]
    for other in COMPARATORS:
        for ax in shared_axes(feat) + pair_axes(feat, "cei_final", other):
            lines += [f"### CEI-GNN minus {METHODS[other]}: {ax['title']}", ""]
            cells = {}
            for s in SOURCES:
                a, b = values.get(("cei_final", s, HEADLINE)), values.get((other, s, HEADLINE))
                diff = np.full(n_visits, np.nan) if a is None or b is None else a - b
                cells[s] = boot.stratified(diff, ax)
                for metric in METRICS:
                    am, bm = values.get(("cei_final", s, metric)), values.get((other, s, metric))
                    if am is not None and bm is not None:
                        for j, cell in enumerate(boot.stratified(am - bm, ax)):
                            record("difference", ax, ax["labels"][j], f"CEI-GNN - {METHODS[other]}", s, metric, cell)
            rows = []
            for j, label in enumerate(ax["labels"]):
                vis = int(ax["masks"][j].sum())
                if vis:
                    rows.append([label, vis, len(np.unique(patient[ax["masks"][j]]))] +
                                [fmt(cells[s][j], bold_if_excluded=True) for s in SOURCES])
            lines += table(["stratum", "visits", "patients"] + list(SOURCES), rows) + [""]
    return lines, csv_rows


def main() -> int:
    data, out = WORK / "data", WORK / "out"
    out.mkdir(parents=True, exist_ok=True)
    feat = load_features(data / "visit_features.csv")
    values = load_values(data / "visit_scores.csv", len(feat["visit_idx"]), set(METHODS))
    lines, rows = build(feat, values)
    (out / "strata-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out / "strata-cells.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out / 'strata-report.md'} ({len(rows)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
