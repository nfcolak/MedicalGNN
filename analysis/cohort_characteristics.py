"""Aggregate cohort and graph statistics behind the report's cohort table, cohort-flow figure and graph description.

Reads the graph file (one pass), the targets file, the manifests of the two cohort folders and the bindings of
the trained runs, and writes aggregates only (counts, shares, means; counts below 25 are printed as "<25").
No identifiers, no per-visit rows.

Usage (repo root):
  python -m analysis.cohort_characteristics --graphs <graphs.jsonl> --targets <targets.csv> \
      --graph-manifest <manifest.json> --cohort-uncapped <dir> --cohort-capped <dir> \
      --runs <trained runs root> --out results/cohort
"""
import argparse
import collections
import csv
import json
import math
import os

try:
    import orjson
    _loads = orjson.loads
except ImportError:  # pragma: no cover
    _loads = json.loads

KS = (10, 20, 30)
FOLDS = ("train", "validation", "test")
SMALL = 25


def _cell(n):
    """Count with small-cell suppression."""
    return f"<{SMALL}" if n < SMALL else f"{n:,}"


def _pct(n, d):
    return f"{100.0 * n / d:.1f}" if d else "-"


def _quantile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    pos = q * (len(sorted_vals) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def scan_graphs(path):
    """One pass: per-visit age, gender, acuity and per-graph size statistics."""
    visit = {}
    size = collections.defaultdict(lambda: collections.defaultdict(float))
    with open(path, "rb") as fh:
        for line in fh:
            g = _loads(line)
            sid, split = g["sample_id"], g["split"]
            age = gender = acuity = None
            kinds = collections.Counter()
            for n in g["nodes"]:
                k = n.get("kind")
                kinds[k] += 1
                if k == "patient":
                    age, gender = n.get("age"), n.get("gender")
                if n.get("acuity") is not None:
                    acuity = n["acuity"]
            informative = sum(1 for e in g["edges"] if e.get("informative"))
            visit[sid] = (split, age, gender, acuity)
            s = size[split]
            s["graphs"] += 1
            s["nodes"] += len(g["nodes"])
            s["nodes_sq"] += len(g["nodes"]) ** 2
            s["edges"] += len(g["edges"])
            s["edges_sq"] += len(g["edges"]) ** 2
            s["informative_edges"] += informative
            for k, c in kinds.items():
                s[f"kind:{k}"] += c
    return visit, size


def summarize(rs, visit):
    """Visit, patient, age, sex, acuity and class-balance summary of a list of target rows."""
    ages = sorted(v[1] for r in rs if (v := visit.get(r["sample_id"])) and v[1] is not None)
    gend = [v[2] for r in rs if (v := visit.get(r["sample_id"]))]
    acu = [v[3] for r in rs if (v := visit.get(r["sample_id"]))]
    cls = collections.Counter(r["label"] for r in rs if r["label"])
    n = len(rs)
    mean = sum(ages) / len(ages)
    sd = math.sqrt(sum((x - mean) ** 2 for x in ages) / (len(ages) - 1))
    out = {
        "visits": n, "patients": len({r["subject_id"] for r in rs}),
        "age_mean": mean, "age_sd": sd, "age_median": _quantile(ages, 0.5), "age_q1": _quantile(ages, 0.25), "age_q3": _quantile(ages, 0.75),
        "age_missing": n - len(ages),
        "female": sum(1 for x in gend if x == "F"), "gender_missing": sum(1 for x in gend if x not in ("F", "M")),
        "acuity": {str(l): sum(1 for x in acu if x is not None and int(x) == l) for l in range(1, 6)},
        "acuity_missing": sum(1 for x in acu if x is None),
    }
    if cls:
        out.update({"majority_class_share": max(cls.values()) / n, "imbalance_largest_to_smallest": max(cls.values()) / max(1, min(cls.values())),
                    "smallest_class_visits": min(cls.values())})
    return out


def label_sets(runs_root):
    """Class names of each K task, from a trained run's binding."""
    out = {}
    for k in KS:
        b = json.load(open(os.path.join(runs_root, f"top{k}", "gsat_seed1234", "binding.json")))
        out[k] = list(b["label_order"])
    return out


def vocab_sizes(runs_root):
    out = {}
    for k in KS:
        d = os.path.join(runs_root, f"top{k}")
        g = json.load(open(os.path.join(d, "gsat_seed1234", "binding.json")))
        c = json.load(open(os.path.join(d, "cei_mp2_seed1234", "binding.json")))
        out[k] = {"token_vocabulary": g["vocabulary_size"], "token_min_count": g["token_min_count"],
                  "absence_items": c["method_config"]["universe_size"], "node_dim": g["node_dim"],
                  "edge_dim": g["edge_dim"], "num_relations": g["num_relations"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--graph-manifest", required=True)
    ap.add_argument("--cohort-uncapped", required=True)
    ap.add_argument("--cohort-capped", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    visit, size = scan_graphs(a.graphs)
    targets = list(csv.DictReader(open(a.targets)))
    labels = label_sets(a.runs)
    gm = json.load(open(a.graph_manifest))
    unc = json.load(open(os.path.join(a.cohort_uncapped, "manifest.json")))
    cap = json.load(open(os.path.join(a.cohort_capped, "manifest.json")))

    res = {"flow": {}, "graphs": {}, "vocabulary": vocab_sizes(a.runs), "tasks": {}, "pooled": {}}
    res["flow"] = {
        "all_visits_with_first_result": {"visits": unc["cohort_policy"]["index_visits"], "patients": unc["cohort_policy"]["patients"]},
        "after_max_6_visits_per_patient": {"visits": cap["cohort_policy"]["index_visits"], "patients": cap["cohort_policy"]["patients"],
                                           "raw_visits_in_cap_pool": cap["index_counts"]["visits"],
                                           "excluded_by_window_rules": cap["cohort_policy"]["visit_eligibility"]},
        "graphs": {"visits": gm["counts"]["graphs"], "patients": cap["cohort_policy"]["patients"]},
        "excluded_visits_total": unc["cohort_policy"]["index_visits"] - cap["cohort_policy"]["index_visits"],
        "excluded_patients_total": unc["cohort_policy"]["patients"] - cap["cohort_policy"]["patients"],
        "label_status": dict(collections.Counter(r["label_status"] for r in targets)),
        "label_reason_not_labelled": dict(collections.Counter(r["label_reason"] for r in targets if r["label_status"] != "labelled")),
        "by_fold": cap["cohort_policy"]["by_fold"],
    }
    tot = int(sum(s["graphs"] for s in size.values()))
    res["graphs"] = {
        "graphs": tot, "nodes": int(sum(s["nodes"] for s in size.values())), "edges": int(sum(s["edges"] for s in size.values())),
        "informative_edges": int(sum(s["informative_edges"] for s in size.values())),
        "manifest_counts": gm["counts"], "manifest_coverage_by_fold": gm["coverage"],
        "complaint_vocabulary": {k: gm["complaint_vocabulary"][k] for k in ("distinct_train_surface_forms", "dropped_surface_forms", "min_count")},
        "by_fold": {f: {"graphs": int(s["graphs"]), "mean_nodes": s["nodes"] / s["graphs"], "mean_edges": s["edges"] / s["graphs"],
                        "informative_share": s["informative_edges"] / s["edges"],
                        "mean_nodes_by_kind": {k[5:]: v / s["graphs"] for k, v in sorted(s.items()) if k.startswith("kind:")}}
                    for f, s in size.items()},
    }

    res["pooled"]["all_graphs"] = summarize(targets, visit)
    for k in KS:
        keep = set(labels[k])
        rows = [r for r in targets if r["label_status"] == "labelled" and r["label"] in keep]
        res["tasks"][k] = {"classes": len(keep), "folds": {}}
        for f in FOLDS:
            res["tasks"][k]["folds"][f] = summarize([r for r in rows if r["split"] == f], visit)
        res["pooled"][f"K{k}"] = summarize(rows, visit)

    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "cohort_characteristics.json"), "w"), indent=1, default=float)
    write_md(res, os.path.join(a.out, "cohort_characteristics.md"))


def write_md(res, path):
    L = ["# Cohort and graph characteristics", "",
         f"Aggregates computed by `analysis/cohort_characteristics.py` from the graph file, the targets file, the manifests of the two cohort folders and the bindings of the trained runs. Counts below {SMALL} are printed as \"<{SMALL}\". No identifiers or per-visit rows.", "",
         "## Cohort flow", ""]
    fl = res["flow"]
    a, b, g = fl["all_visits_with_first_result"], fl["after_max_6_visits_per_patient"], fl["graphs"]
    L += ["| Step | Visits | Patients |", "|---|---|---|",
          f"| All visits with a first laboratory result (no cap on visits per patient) | {a['visits']:,} | {a['patients']:,} |",
          f"| Cohort of the graphs (at most 6 visits per patient, one decision point per visit) | {b['visits']:,} | {b['patients']:,} |",
          f"| Excluded | {fl['excluded_visits_total']:,} | {fl['excluded_patients_total']:,} |",
          f"| Graphs | {g['visits']:,} | {g['patients']:,} |", ""]
    ve = b["excluded_by_window_rules"]
    L += [f"Cap pool of the 6-visit cohort: {b['raw_visits_in_cap_pool']:,} visits; of these, no laboratory charttime in the window: {ve['no_lab_charttime_in_window']:,}; no eligible recorded result in the window: {ve['no_eligible_recorded_result_in_window']:,}; end before start: {ve.get('end_before_start', 0):,} (counted outside the pool).", "",
          "Target labelling of the graphs:", "", "| Status / reason | Visits |", "|---|---|"]
    for k, v in sorted(fl["label_status"].items(), key=lambda x: -x[1]):
        L.append(f"| {k} | {v:,} |")
    for k, v in sorted(fl["label_reason_not_labelled"].items(), key=lambda x: -x[1]):
        L.append(f"| not labelled: {k} | {v:,} |")
    L += ["", "## Graph statistics", "", "| Fold | Graphs | Mean nodes | Mean edges | Informative edges (%) |", "|---|---|---|---|---|"]
    for f in FOLDS:
        s = res["graphs"]["by_fold"][f]
        L.append(f"| {f} | {s['graphs']:,} | {s['mean_nodes']:.1f} | {s['mean_edges']:.1f} | {100 * s['informative_share']:.1f} |")
    gr = res["graphs"]
    L += ["", f"All folds: {gr['graphs']:,} graphs, {gr['nodes']:,} nodes ({gr['nodes'] / gr['graphs']:.1f} per graph), {gr['edges']:,} edges ({gr['edges'] / gr['graphs']:.1f} per graph), {100 * gr['informative_edges'] / gr['edges']:.1f}% informative.", "",
          "Mean nodes per graph by kind (train fold):", "", "| Kind | Mean per graph |", "|---|---|"]
    for k, v in sorted(gr["by_fold"]["train"]["mean_nodes_by_kind"].items(), key=lambda x: -x[1]):
        L.append(f"| {k} | {v:.2f} |")
    cv = gr["manifest_coverage_by_fold"]["train"]
    n = cv["graphs"]
    L += ["", "Coverage of the training graphs (graph manifest):", "", "| With | Graphs | Share (%) |", "|---|---|---|"]
    for key, name in (("with_complaint", "at least one complaint"), ("with_vitals", "triage vital signs"), ("with_baseline", "a laboratory baseline from an earlier visit"), ("with_prior_diagnosis", "at least one prior diagnosis"), ("with_knowledge", "knowledge relations")):
        L.append(f"| {name} | {cv[key]:,} | {_pct(cv[key], n)} |")
    cvv = gr["complaint_vocabulary"]
    L += ["", f"Complaint vocabulary: {cvv['distinct_train_surface_forms'] - cvv['dropped_surface_forms']} surface forms kept of {cvv['distinct_train_surface_forms']:,} distinct in the training fold (minimum count {cvv['min_count']}).", "",
          "## Vocabulary and absence items per task", "", "| K | Token vocabulary | Token minimum count | Absence items | Node features | Edge features | Relations |", "|---|---|---|---|---|---|---|"]
    for k in KS:
        v = res["vocabulary"][k]
        L.append(f"| {k} | {v['token_vocabulary']} | {v['token_min_count']} | {v['absence_items']} | {v['node_dim']} | {v['edge_dim']} | {v['num_relations']} |")
    L += ["", "## Cohort characteristics, all folds pooled", "",
          "Age in years at the visit (mean (SD); median [Q1, Q3]); female and acuity shares in % of visits (acuity 4 and 5 merged, since level 5 has fewer than "
          f"{SMALL} visits); ESI triage level.", "",
          "| Group | Visits | Patients | Age mean (SD) | Age median [Q1, Q3] | Female (%) | Acuity 1 (%) | 2 | 3 | 4-5 | Missing |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, name in (("all_graphs", "All graphs (labelled or not)"), ("K20", "K = 20 task"), ("K10", "K = 10 task"), ("K30", "K = 30 task")):
        t = res["pooled"][key]
        n, ac = t["visits"], t["acuity"]
        L.append(f"| {name} | {n:,} | {t['patients']:,} | {t['age_mean']:.1f} ({t['age_sd']:.1f}) | {t['age_median']:.0f} [{t['age_q1']:.0f}, {t['age_q3']:.0f}] | {_pct(t['female'], n)} | "
                 f"{_pct(ac['1'], n)} | {_pct(ac['2'], n)} | {_pct(ac['3'], n)} | {_pct(ac['4'] + ac['5'], n)} | {_pct(t['acuity_missing'], n)} |")
    L += ["", "## Cohort characteristics per task and fold", "",
          "Labelled visits of the K most frequent diagnoses (ranked by training count). Age in years at the visit (mean ± SD; median [Q1, Q3]); acuity is the ESI triage level.", "",
          "| K | Fold | Visits | Patients | Age mean ± SD | Age median [Q1, Q3] | Age missing (%) | Female (%) | Sex missing (%) | Acuity 1 | 2 | 3 | 4 | 5 | Acuity missing (%) | Majority class (%) | Largest : smallest class |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in KS:
        for f in FOLDS:
            t = res["tasks"][k]["folds"][f]
            n = t["visits"]
            ac = t["acuity"]
            L.append(f"| {k} | {f} | {n:,} | {t['patients']:,} | {t['age_mean']:.1f} ± {t['age_sd']:.1f} | {t['age_median']:.0f} [{t['age_q1']:.0f}, {t['age_q3']:.0f}] | {_pct(t['age_missing'], n)} | {_pct(t['female'], n)} | {_pct(t['gender_missing'], n)} | "
                     + " | ".join(f"{_cell(ac[str(l)])} ({_pct(ac[str(l)], n)})" for l in range(1, 6))
                     + f" | {_pct(t['acuity_missing'], n)} | {100 * t['majority_class_share']:.1f} | {t['imbalance_largest_to_smallest']:.1f} : 1 |")
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
