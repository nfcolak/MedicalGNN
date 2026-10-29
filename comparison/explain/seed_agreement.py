"""Do different seeds point at the same evidence for the same visit?

    python -m comparison.explain.seed_agreement --explanations <dir or run dirs...> \
        --graphs <graphs.jsonl> --out <new dir> [--bootstrap 2000] [--seed 1234]

Reads the record files of `core.explain.runner` (inference-free) and, to know which node of a record
is which kind, the graph file the runs were built from. Runs are grouped by method and seed from
their folder name (`<method>_seed<N>`); a source is compared with itself across seeds, on the visits
that both seeds explained.

For every pair of seeds, visit and source, over the visit's eligible evidence nodes only (the
kinds in `fidelity_v2_intervention.eligible_node_kinds`, the same nodes the fidelity cut can delete):

* top-k overlap: the share of one seed's top-k nodes that are also in the other's top-k, at the
  main fidelity cut (top 20% of the eligible nodes, the rule of `fidelity_v2.resolve_k`) and at a
  fixed k = 5. Nodes are ranked by |importance|; ties at the cut are broken by seeded random
  orders (10 draws, one per vector) and the draws are averaged.
* chance: k / eligible nodes, what two random k-subsets share on average. The lift
  (overlap - chance) / (1 - chance) is 0 at chance and 1 for identical top-k sets.
* rank correlation: Spearman of the two importance vectors over the eligible nodes; undefined when
  one vector is constant.

A vector with no nonzero importance on an eligible node is left out (the fidelity code counts it as
"no eligible evidence"). The patient is the unit: a visit's value is the mean over the seed pairs it
has, a patient's the mean over its visits, intervals a patient-level bootstrap withheld below 10
patients. This measures agreement between separately trained models, not correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from comparison.explain.fair_report import MIN_PATIENTS_FOR_INTERVAL, interval, load_runs, sources_of

TOP_FRACTION = 0.2
FIXED_K = 5
TIE_DRAWS = 10
RUN_NAME = re.compile(r"^(?P<method>.+)_seed(?P<seed>\d+)$")


def node_kinds(graphs_path: Path, sample_ids) -> dict[str, list[str]]:
    """{sample id: node kind per position} for the requested graphs, read from graphs.jsonl."""
    wanted = set(sample_ids)
    found: dict[str, list[str]] = {}
    pattern = re.compile(r'"sample_id": ?"([0-9a-f]+)"')
    with open(graphs_path) as handle:
        for line in handle:
            match = pattern.search(line)
            if match is None or match.group(1) not in wanted:
                continue
            graph = json.loads(line)
            found[graph["sample_id"]] = [node["kind"] for node in graph["nodes"]]
            if len(found) == len(wanted):
                break
    return found


def source_vector(record: dict, source: str):
    """The visit's importance vector of one source, or None when the record has none."""
    if source == "builtin":
        vector = record.get("builtin_node_importance") if record.get("builtin_available") else None
    else:
        entry = record.get("graphxai", {}).get(source, {})
        vector = (entry.get("node_explanation") or {}).get("node_importance") if entry.get("status") == "success" else None
    return None if vector is None else np.abs(np.asarray(vector, dtype=float))


def top_k_set(importance: np.ndarray, k: int, rng: np.random.Generator) -> set[int]:
    """Indices of the k largest importances; ties at the cut broken by a random order."""
    order = np.lexsort((rng.random(importance.size), -importance))
    return set(order[:k].tolist())


def pair_agreement(a: np.ndarray, b: np.ndarray, sample_id: str) -> dict | None:
    """Overlap, chance and rank correlation of two importance vectors over the same eligible nodes."""
    m = a.size
    if m == 0 or not (a > 0).any() or not (b > 0).any():
        return None
    rng = np.random.default_rng(int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16))
    result = {}
    for label, k in (("cut", min(m, max(1, int(TOP_FRACTION * m)))), ("k5", min(FIXED_K, m))):
        overlap = np.mean([len(top_k_set(a, k, rng) & top_k_set(b, k, rng)) / k for _ in range(TIE_DRAWS)])
        result[f"overlap_{label}"] = float(overlap)
        result[f"chance_{label}"] = k / m
    constant = np.ptp(a) == 0 or np.ptp(b) == 0
    result["rank_correlation"] = None if constant else float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
    return result


def fmt(entry: dict, digits: int = 3) -> str:
    if entry["mean"] is None:
        return "-"
    text = f"{entry['mean']:.{digits}f}"
    if entry["ci"] is not None:
        text += f" [{entry['ci'][0]:.{digits}f}, {entry['ci'][1]:.{digits}f}]"
    return text + f" (n={entry['n']})"


def group_runs(runs: dict[str, list[dict]]) -> dict[str, dict[int, list[dict]]]:
    """{method: {seed: records}} from run names ending in `_seed<N>`; other runs are ignored."""
    grouped: dict[str, dict[int, list[dict]]] = defaultdict(dict)
    for name, records in runs.items():
        match = RUN_NAME.match(name.split("/")[-1])
        if match:
            grouped[match["method"]][int(match["seed"])] = records
    return grouped


def visit_values(by_seed: dict[int, list[dict]], source: str, kinds: dict[str, list[str]]) -> list[dict]:
    """Per visit, the mean over seed pairs of each agreement number, with the patient it belongs to."""
    indexed = {seed: {r["sample_id"]: r for r in records} for seed, records in by_seed.items()}
    visits = []
    for sample_id in sorted(set().union(*indexed.values())):
        per_pair, subject = [], None
        for s1, s2 in combinations(sorted(indexed), 2):
            r1, r2 = indexed[s1].get(sample_id), indexed[s2].get(sample_id)
            if r1 is None or r2 is None or sample_id not in kinds:
                continue
            a, b = source_vector(r1, source), source_vector(r2, source)
            if a is None or b is None or not (a.size == b.size == len(kinds[sample_id])):
                continue
            eligible = set(r1.get("fidelity_v2_intervention", {}).get("eligible_node_kinds", ()))
            mask = np.array([kind in eligible for kind in kinds[sample_id]])
            agreement = pair_agreement(a[mask], b[mask], sample_id)
            if agreement is not None:
                per_pair.append(agreement)
                subject = str(r1["subject_id"])
        if per_pair:
            row = {"subject_id": subject}
            for key in per_pair[0]:
                values = [p[key] for p in per_pair if p[key] is not None]
                row[key] = float(np.mean(values)) if values else None
            visits.append(row)
    return visits


def patient_means(visits: list[dict], key: str) -> dict[str, float]:
    grouped = defaultdict(list)
    for row in visits:
        if row.get(key) is not None:
            grouped[row["subject_id"]].append(row[key])
    return {patient: float(np.mean(v)) for patient, v in grouped.items()}


def agreement_report(runs, kinds, *, bootstrap: int, seed: int):
    lines = ["## Agreement between seeds", "",
             "Same source, same visits, pairs of seeds; eligible evidence nodes only. Overlap is the share of the top nodes two "
             "seeds share; chance is what random subsets of that size would share; lift is 0 at chance and 1 for identical "
             "top sets; rank correlation is Spearman over all eligible nodes.", ""]
    data = {}
    for method, by_seed in sorted(group_runs(runs).items()):
        if len(by_seed) < 2:
            lines += [f"### {method}", "", f"Only seed(s) {sorted(by_seed)}: no pair to compare.", ""]
            continue
        lines += [f"### {method} (seeds {sorted(by_seed)})", "",
                  "| source | patients | overlap at cut | chance | lift | overlap at k=5 | chance | rank correlation |",
                  "|---|---|---|---|---|---|---|---|"]
        every = [r for records in by_seed.values() for r in records]
        for source in sources_of(every):
            visits = visit_values(by_seed, source, kinds)
            cells = {key: interval(patient_means(visits, key), bootstrap=bootstrap, seed=seed)
                     for key in ("overlap_cut", "chance_cut", "overlap_k5", "chance_k5", "rank_correlation")}
            overlap, chance = patient_means(visits, "overlap_cut"), patient_means(visits, "chance_cut")
            lift = {p: (overlap[p] - chance[p]) / (1 - chance[p]) for p in overlap if chance[p] < 1}
            cells["lift"] = interval(lift, bootstrap=bootstrap, seed=seed)
            entry = dict(cells)
            data[f"{method}/{source}"] = entry
            lines.append(f"| {source} | {cells['overlap_cut']['n']} | {fmt(cells['overlap_cut'])} | "
                         f"{fmt(cells['chance_cut'])} | {fmt(cells['lift'])} | {fmt(cells['overlap_k5'])} | "
                         f"{fmt(cells['chance_k5'])} | {fmt(cells['rank_correlation'])} |")
        lines.append("")
    return lines, data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--explanations", nargs="+", required=True)
    parser.add_argument("--graphs", required=True, type=Path, help="graphs.jsonl the runs were built from")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"--out must be new or empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.explanations)
    wanted = {r["sample_id"] for records in runs.values() for r in records}
    kinds = node_kinds(args.graphs, wanted)
    missing = len(wanted) - len(kinds)
    if missing:
        print(f"warning: {missing} of {len(wanted)} visits not found in {args.graphs}; they are left out")
    lines, data = agreement_report(runs, kinds, bootstrap=args.bootstrap, seed=args.seed)
    document = ["# Seed agreement of the explanations", "",
                f"Patient-level bootstrap, {args.bootstrap} resamples, 95% percentile intervals; intervals are withheld "
                f"below {MIN_PATIENTS_FOR_INTERVAL} patients.", ""] + lines
    (args.out / "seed_agreement.md").write_text("\n".join(document) + "\n")
    (args.out / "seed_agreement.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out / 'seed_agreement.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
