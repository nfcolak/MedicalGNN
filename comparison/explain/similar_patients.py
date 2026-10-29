"""Are explanations consistent for clinically similar patients?

    python -m comparison.explain.similar_patients --explanations <dir or run dirs...> \
        --graphs <graphs.jsonl> [--runs <trained runs root>] [--k K] --out <new dir> [--bootstrap 2000] [--seed 1234]

Reads the record files of `core.explain.runner` (inference-free) and the graph file the runs were
built from. "Similar" is decided from the inputs alone, so the same visit pairs are used for every
method and explainer:

* an evidence item of a visit is an eligible node's token with its level and scope where it has
  them (`lab:50963|high|index`, `cc:dyspnea`); eligible kinds are those of the fidelity cut;
* two visits are similar when they are of different patients, their true diagnosis classes are in
  the same diagnosis group (`CLASS_GROUPS`, the grouping of `graph_types/step3_clinical.py`, kept by
  class name because class ids depend on K; classes outside its 20 are left out) and they share at least MIN_SHARED evidence items; the NEIGHBOURS most similar visits of a visit
  (Jaccard of the item sets) are its similar pairs;
* two baselines on the same visits, drawn with a seeded generator: BASELINE_PAIRS random visits of
  the same group that are not neighbours (`same_group`), and BASELINE_PAIRS of other groups
  (`other_group`), both with at least MIN_SHARED shared items.

Explanations of two different visits can only agree on what both contain, so agreement is computed
over the shared items: an item's importance is the largest |importance| among the visit's nodes
with that token; overlap of the top 20% (rule of `fidelity_v2.resolve_k`) and of the top 5, with
the chance level, and the Spearman rank correlation, are those of `seed_agreement.pair_agreement`.
A pair of a visit counts only when both visits were explained by the run.

The patient is the unit: a patient's number is the mean over its visits (and over the seeds of the
method), intervals are a patient-level bootstrap withheld below 10 patients, and a difference
between pair types is taken on the patients that have both. If explanations follow the patient and
not just the diagnosis, similar pairs should agree more than the baselines; agreement that is as
high for other-group pairs means the explainer highlights the same things whatever the patient.
This is consistency, not correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from comparison.explain.fair_report import MIN_PATIENTS_FOR_INTERVAL, interval, load_runs, paired_interval, sources_of
from comparison.explain.seed_agreement import fmt, group_runs, pair_agreement, source_vector

# The 20 classes of the K=20 runs in their label order. The class ids of a run depend on K (K=10 and K=30 number the
# same diagnoses differently), so the diagnosis groups below are kept by class NAME and a run's ids are translated
# through its own label order (`binding.json` of the trained run). Classes outside these 20 (K=30 adds ten) have no
# group and are left out of this analysis.
K20_LABELS = [
    "Acute kidney failure", "Acute pharyngitis", "Alcohol abuse with intoxication", "Anemia", "Back or spine pain",
    "Diabetes mellitus", "Epilepsy", "GI bleed", "Head injury", "Heart failure", "Hypokalemia", "Limb injury or pain",
    "Lower respiratory disease", "Major depressive disorder", "Non-ST elevation (NSTEMI) myocardial infarction", "Sepsis",
    "Skin or soft-tissue infection", "UTI or pyelonephritis", "Unsp intestnl obst", "Unspecified atrial fibrillation"]
# Same grouping as comparison/explain/graph_types/step3_clinical.py, whose class ids are the K=20 ids (checked by a test).
CLASS_GROUPS_K20_IDS = {
    "cardiac": [9, 14, 19],
    "infection": [15, 17, 16, 1, 12],
    "injury_pain": [8, 11, 4],
    "gastrointestinal": [7, 18],
    "metabolic_renal_heme": [0, 3, 5, 10],
    "neuro_psych_substance": [6, 13, 2],
}
CLASS_GROUPS = {group: [K20_LABELS[i] for i in ids] for group, ids in CLASS_GROUPS_K20_IDS.items()}
GROUP_OF_NAME = {name: group for group, names in CLASS_GROUPS.items() for name in names}
DEFAULT_RUNS = "comparison/standardized/clinical_runs_fair_v10t_20261011"
MIN_SHARED = 5
NEIGHBOURS = 5
BASELINE_PAIRS = 5
PAIR_TYPES = ("similar", "same_group", "other_group")
METRICS = ("overlap_cut", "chance_cut", "overlap_k5", "rank_correlation", "lift")


def read_nodes(graphs_path: Path, sample_ids) -> dict[str, list[tuple[str, str]]]:
    """{sample id: [(kind, item) per node position]} for the requested graphs, read from graphs.jsonl.

    The item is the node's token plus its level and scope where present; nodes without a token
    get an empty item."""
    wanted = set(sample_ids)
    found: dict[str, list[tuple[str, str]]] = {}
    pattern = re.compile(r'"sample_id": ?"([0-9a-f]+)"')
    with open(graphs_path) as handle:
        for line in handle:
            match = pattern.search(line)
            if match is None or match.group(1) not in wanted:
                continue
            graph = json.loads(line)
            found[graph["sample_id"]] = [
                (node["kind"], "|".join(str(node[f]) for f in ("token", "level", "scope") if node.get(f) is not None))
                for node in graph["nodes"]]
            if len(found) == len(wanted):
                break
    return found


def evidence_items(nodes, eligible_kinds) -> dict[int, str]:
    """{node position: item} of the eligible nodes that have an item."""
    return {i: item for i, (kind, item) in enumerate(nodes) if kind in eligible_kinds and item}


def jaccard(a: set, b: set) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _rng(sample_id: str, label: str, seed: int) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(f"{seed}:{label}:{sample_id}".encode()).hexdigest()[:8], 16))


def choose_pairs(visits: list[dict], *, seed: int = 1234) -> dict[str, list[tuple[str, str]]]:
    """Similar, same-group and other-group pairs (anchor, other) from the visits' inputs alone.

    Each visit is a dict with `sample_id`, `subject_id`, `group` and `items` (a set)."""
    order = sorted(visits, key=lambda v: v["sample_id"])
    pairs: dict[str, list[tuple[str, str]]] = {kind: [] for kind in PAIR_TYPES}
    for anchor in order:
        eligible = [(jaccard(anchor["items"], other["items"]), other) for other in order
                    if other["subject_id"] != anchor["subject_id"]
                    and len(anchor["items"] & other["items"]) >= MIN_SHARED]
        same = sorted((e for e in eligible if e[1]["group"] == anchor["group"]),
                      key=lambda e: (-e[0], e[1]["sample_id"]))
        neighbours = [o["sample_id"] for _, o in same[:NEIGHBOURS]]
        if not neighbours:
            continue
        pairs["similar"] += [(anchor["sample_id"], sid) for sid in neighbours]
        rest = [o["sample_id"] for _, o in same[NEIGHBOURS:]]
        other = [o["sample_id"] for _, o in eligible if o["group"] != anchor["group"]]
        for kind, pool in (("same_group", rest), ("other_group", other)):
            if pool:
                rng = _rng(anchor["sample_id"], kind, seed)
                picked = rng.choice(len(pool), size=min(BASELINE_PAIRS, len(pool)), replace=False)
                pairs[kind] += [(anchor["sample_id"], pool[i]) for i in sorted(picked)]
    return pairs


def item_vector(record: dict, source: str, items: dict[int, str]):
    """{item: largest |importance| of the nodes with that item} for one explained visit, or None."""
    vector = source_vector(record, source)
    if vector is None or vector.size < (max(items) + 1 if items else 0):
        return None
    best: dict[str, float] = {}
    for position, item in items.items():
        best[item] = max(best.get(item, 0.0), float(vector[position]))
    return best


def run_patient_values(records: list[dict], source: str, pairs, items_of) -> dict[str, dict[str, dict[str, float]]]:
    """{pair type: {metric: {patient: value}}} of one run and source."""
    by_id = {r["sample_id"]: r for r in records}
    vectors: dict[str, dict | None] = {}

    def vector_of(sample_id):
        if sample_id not in vectors:
            vectors[sample_id] = item_vector(by_id[sample_id], source, items_of[sample_id]) if sample_id in by_id else None
        return vectors[sample_id]

    result = {}
    for kind, kind_pairs in pairs.items():
        per_anchor = defaultdict(lambda: defaultdict(list))
        for anchor, other in kind_pairs:
            a, b = vector_of(anchor), vector_of(other)
            if a is None or b is None:
                continue
            shared = sorted(set(a) & set(b))
            if len(shared) < MIN_SHARED:
                continue
            agreement = pair_agreement(np.array([a[i] for i in shared]), np.array([b[i] for i in shared]),
                                       f"{anchor}:{other}")
            if agreement is None:
                continue
            if agreement["chance_cut"] < 1:
                agreement["lift"] = (agreement["overlap_cut"] - agreement["chance_cut"]) / (1 - agreement["chance_cut"])
            for metric in METRICS:
                if agreement.get(metric) is not None:
                    per_anchor[str(by_id[anchor]["subject_id"])][metric].append(agreement[metric])
        metrics = {m: {} for m in METRICS}
        for patient, values in per_anchor.items():
            for metric, numbers in values.items():
                metrics[metric][patient] = float(np.mean(numbers))
        result[kind] = metrics
    return result


def merge_seeds(per_seed: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    """Mean over the seeds of each patient's number."""
    merged = {kind: {m: defaultdict(list) for m in METRICS} for kind in PAIR_TYPES}
    for run in per_seed:
        for kind in PAIR_TYPES:
            for metric in METRICS:
                for patient, value in run[kind][metric].items():
                    merged[kind][metric][patient].append(value)
    return {kind: {m: {p: float(np.mean(v)) for p, v in merged[kind][m].items()} for m in METRICS} for kind in PAIR_TYPES}


def label_order(runs_root, run_names) -> list[str]:
    """The class names of the explained runs, by class id: `label_order` of a trained run's binding.json
    (every method of one K shares it). `run_names` look like `top<K>/<method>_seed<N>`."""
    for name in run_names:
        path = Path(runs_root) / name / "binding.json"
        if path.is_file():
            return json.loads(path.read_text())["label_order"]
    raise SystemExit(f"no trained run's binding.json under {runs_root} for {sorted(run_names)[:3]}: "
                     "pass --runs <root of the trained runs>")


def build_visits(records_by_run, nodes_of, labels) -> tuple[list[dict], dict[str, dict[int, str]]]:
    """Visits explained by every run, with their evidence items (inputs only)."""
    ids = set.intersection(*(set(r["sample_id"] for r in records) for records in records_by_run.values()))
    first = {r["sample_id"]: r for r in next(iter(records_by_run.values()))}
    visits, items_of = [], {}
    for sample_id in sorted(ids):
        record = first[sample_id]
        group = GROUP_OF_NAME.get(labels[record["true_class_id"]])
        if sample_id not in nodes_of or group is None:
            continue
        eligible = set(record.get("fidelity_v2_intervention", {}).get("eligible_node_kinds", ()))
        items = evidence_items(nodes_of[sample_id], eligible)
        items_of[sample_id] = items
        visits.append({"sample_id": sample_id, "subject_id": str(record["subject_id"]), "group": group,
                       "items": set(items.values())})
    return visits, items_of


def similar_report(runs, nodes_of, labels, *, bootstrap: int, seed: int):
    lines = ["## Agreement of explanations across similar patients", "",
             f"Similar = different patients, same diagnosis group, at least {MIN_SHARED} shared evidence items; each visit's "
             f"{NEIGHBOURS} most similar visits by Jaccard of their item sets. Baselines: up to {BASELINE_PAIRS} random "
             "same-group visits that are not neighbours, and up to "
             f"{BASELINE_PAIRS} random visits of other groups. Agreement is over the items both visits contain.", ""]
    data: dict = {}
    grouped = group_runs(runs)
    if not grouped:
        raise SystemExit("no run folder is named <method>_seed<N>")
    visits, items_of = build_visits({f"{m}_seed{s}": records for m, by in grouped.items() for s, records in by.items()},
                                    nodes_of, labels)
    pairs = choose_pairs(visits, seed=seed)
    by_id = {v["sample_id"]: v for v in visits}
    sizes = defaultdict(set)
    for v in visits:
        sizes[v["group"]].add(v["subject_id"])
    lines += ["Visits per diagnosis group (patients in brackets): " +
              "; ".join(f"{g} {sum(1 for v in visits if v['group'] == g)} ({len(sizes[g])})" for g in CLASS_GROUPS), ""]
    lines += ["| pair type | pairs | mean Jaccard of the item sets |", "|---|---|---|"]
    data["pairs"] = {}
    for kind in PAIR_TYPES:
        jac = [jaccard(by_id[a]["items"], by_id[b]["items"]) for a, b in pairs[kind]]
        data["pairs"][kind] = {"pairs": len(jac), "mean_jaccard": float(np.mean(jac)) if jac else None}
        lines.append(f"| {kind} | {len(jac)} | {np.mean(jac):.3f} |" if jac else f"| {kind} | 0 | - |")
    lines.append("")
    for method, by_seed in sorted(grouped.items()):
        lines += [f"### {method} (seeds {sorted(by_seed)})", "",
                  "| source | pair type | patients | overlap at cut | chance | lift | rank correlation |", "|---|---|---|---|---|---|---|"]
        diffs = []
        for source in sources_of([r for records in by_seed.values() for r in records]):
            merged = merge_seeds([run_patient_values(records, source, pairs, items_of) for records in by_seed.values()])
            entry = {kind: {m: interval(merged[kind][m], bootstrap=bootstrap, seed=seed) for m in METRICS}
                     for kind in PAIR_TYPES}
            for kind in PAIR_TYPES:
                e = entry[kind]
                lines.append(f"| {source} | {kind} | {e['lift']['n']} | {fmt(e['overlap_cut'])} | {fmt(e['chance_cut'])} | "
                             f"{fmt(e['lift'])} | {fmt(e['rank_correlation'])} |")
            cells = []
            for base in ("same_group", "other_group"):
                d = {m: paired_interval(merged["similar"][m], merged[base][m], bootstrap=bootstrap, seed=seed)
                     for m in ("lift", "rank_correlation")}
                entry[f"similar_minus_{base}"] = d
                cells += [fmt(d["lift"]), fmt(d["rank_correlation"])]
            diffs.append(f"| {source} | " + " | ".join(cells) + " |")
            data[f"{method}/{source}"] = entry
        lines += ["", "Similar minus baseline, on the patients that have both (positive: explanations agree more for similar patients):",
                  "", "| source | lift vs same-group | rank corr. vs same-group | lift vs other-group | rank corr. vs other-group |",
                  "|---|---|---|---|---|"] + diffs + [""]
    return lines, data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--explanations", nargs="+", required=True)
    parser.add_argument("--graphs", required=True, type=Path, help="graphs.jsonl the runs were built from")
    parser.add_argument("--runs", default=DEFAULT_RUNS, type=Path,
                        help="root of the trained runs (top<K>/<method>_seed<N>/binding.json): class names by id")
    parser.add_argument("--k", type=int, default=None, help="K of the explained runs when their folder names do not say")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"--out must be new or empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.explanations)
    wanted = {r["sample_id"] for records in runs.values() for r in records}
    nodes_of = read_nodes(args.graphs, wanted)
    if len(nodes_of) < len(wanted):
        print(f"warning: {len(wanted) - len(nodes_of)} of {len(wanted)} visits not found in {args.graphs}; they are left out")
    ks = {m.group(1) for name in runs if (m := re.search(r"top(\d+)/", name + "/"))} or (
        {str(args.k)} if args.k else set())
    if len(ks) != 1:
        raise SystemExit(f"cannot tell K from the run folders ({sorted(ks)}); pass --k with one K per call")
    labels = label_order(args.runs, [f"top{next(iter(ks))}/{n.split('/')[-1]}" for n in runs])
    lines, data = similar_report(runs, nodes_of, labels, bootstrap=args.bootstrap, seed=args.seed)
    document = ["# Explanation consistency across similar patients", "",
                f"Patient-level bootstrap, {args.bootstrap} resamples, 95% percentile intervals; intervals are withheld "
                f"below {MIN_PATIENTS_FOR_INTERVAL} patients.", ""] + lines
    (args.out / "similar_patients.md").write_text("\n".join(document) + "\n")
    (args.out / "similar_patients.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out / 'similar_patients.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
