"""Readable patient examples: which evidence did each method point at, in words, with its sign.

    python -m comparison.explain.decode_examples --explanations <root with top<K>/<method>_seed<N>> --k 20 \
        --graphs <graphs.jsonl> --runs <trained runs root> --out <new dir> [--seed 1234] [--per-group 1] \
        [--with-errors] [--classes "Heart failure,Sepsis"] [--top 8] [--sources builtin,IntegratedGradExplainer] \
        [--names <d_labitems.csv | token,name csv>] [--effects] [--fold test]

Reads the record files of `core.explain.runner` and the graph file the runs were built from. A record holds
importance values by node position only; the graph record's node order is that position, so each
position is decoded into its kind, token, value, unit, level, reference range and time (hours relative to the
prediction cutoff, negative = before).

Which visits (a rule fixed here, so the examples are not picked by eye): one diagnosis class per diagnosis
group of `similar_patients.CLASS_GROUPS` (its most frequent class among the explained visits), or the classes
named with `--classes`; within a class, the visit the reference model (CEI-GNN, else the first method) predicts
correctly with the probability closest to the median of its correct visits (ties by sample id); with
`--with-errors` also the misclassified visit closest to the median of its errors. The same visits are shown for
every method that explained them, so methods can be read side by side, each with its own prediction.

Per method and source the top nodes by |importance| over the shown kinds (complaint, measurement, vital,
diagnosis, knowledge) are listed. Two signs, kept apart:

* `signed contribution`: CEI-GNN's own evidence ledger (its built-in explanation) is additive and signed; a
  negative value lowers the explained class score. No other method stores a sign.
* `effect` (with `--effects`, needs `--runs`): the same deletion intervention the fidelity metrics use applied to
  this one node (and what depends on it): drop in the predicted-class probability when it is deleted. Positive =
  the model needs the node for its prediction; negative = the model would be more confident without it. Only
  for the eligible evidence kinds; knowledge and the other protected nodes show "-". It is an intervention on the
  model's input, not a statement about clinical causes.

Names: complaints, ICD-10 categories and knowledge tokens are already text. A lab shows as `lab:<id>` with its
unit, flag and range until `--names` gives names: a MIMIC-IV `d_labitems.csv` (columns itemid, label, optional
fluid) or any CSV with columns token, name.

Privacy: nothing identifying is written: no patient or stay id, age as a decade, sex, no race or other
demographics, times relative to the cutoff, a hashed patient key. The graph file is credentialed MIMIC data:
check what a thesis may show of it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from comparison.explain.fair_report import load_runs
from comparison.explain.seed_agreement import group_runs, source_vector
from comparison.explain.similar_patients import DEFAULT_RUNS, GROUP_OF_NAME, label_order

SHOWN_KINDS = ("complaint", "measurement", "vital", "diagnosis", "knowledge")
DEFAULT_SOURCES = ("builtin", "IntegratedGradExplainer")
REFERENCE_METHOD = "cei_mp2"


def read_graphs(graphs_path: Path, sample_ids) -> dict[str, dict]:
    """{sample id: graph record} for the requested graphs, read from graphs.jsonl."""
    wanted = set(sample_ids)
    found: dict[str, dict] = {}
    pattern = re.compile(r'"sample_id": ?"([0-9a-f]+)"')
    with open(graphs_path) as handle:
        for line in handle:
            match = pattern.search(line)
            if match is None or match.group(1) not in wanted:
                continue
            graph = json.loads(line)
            found[graph["sample_id"]] = graph
            if len(found) == len(wanted):
                break
    return found


def load_names(path) -> dict[str, str]:
    """{token: name} from d_labitems.csv (itemid, label[, fluid]) or a token,name CSV."""
    if path is None:
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    columns = set(rows[0])
    if {"itemid", "label"} <= columns:
        return {f"lab:{r['itemid']}": r["label"] + (f" ({r['fluid']})" if r.get("fluid") else "") for r in rows}
    if {"token", "name"} <= columns:
        return {r["token"]: r["name"] for r in rows}
    raise SystemExit("--names must have columns itemid,label[,fluid] or token,name")


def readable_token(token: str) -> str:
    """A token as words when it has no name: `cc:chest pain` -> complaint: chest pain."""
    if token.startswith("cc:"):
        return "complaint: " + token[3:]
    if token.startswith("dx:icd10:"):
        return "diagnosis ICD-10 " + token[9:]
    if token.startswith("knowledge:"):
        return "knowledge: " + token[10:].replace(":", " / ")
    if token.startswith("vital:"):
        return "vital: " + token[6:]
    if token.startswith("lab:"):
        return "lab " + token[4:]
    return token


def decode_node(node: dict, names: dict[str, str]) -> dict:
    """The words of one node: label, value with unit, level or flag, reference range, time."""
    token = node.get("token") or ""
    value = node.get("value")
    unit = node.get("unit") or ""
    lower, upper = node.get("ref_lower"), node.get("ref_upper")
    time = node.get("time_hours")
    return {"kind": node["kind"], "token": token,
            "label": names.get(token) or readable_token(token),
            "value": None if value is None else f"{value:g}" + (f" {unit}" if unit else ""),
            "level": node.get("level") or node.get("flag"),
            "range": f"{lower:g}-{upper:g}" if lower is not None and upper is not None else None,
            "time_hours": None if time is None else round(float(time), 1),
            "scope": node.get("scope")}


def patient_line(graph: dict, num_visits) -> str:
    """Sex, age as a decade and the number of ED visits; nothing else about the patient."""
    patient = next((n for n in graph["nodes"] if n["kind"] == "patient"), {})
    age = patient.get("age")
    band = "age unknown" if age is None else "age 90+" if age >= 90 else f"age {int(age // 10 * 10)}s"
    sex = {"F": "female", "M": "male"}.get(patient.get("gender"), "sex unknown")
    return f"{sex}, {band}, {num_visits} ED visit(s) in the record"


def probability(record: dict):
    return (record.get("edge_dependence") or {}).get("probability_unmasked")


def pick_nearest_median(candidates: list[tuple[str, float]]):
    """The (sample id, probability) closest to the median probability; ties by sample id."""
    if not candidates:
        return None
    median = float(np.median([p for _, p in candidates]))
    return min(candidates, key=lambda c: (abs(c[1] - median), c[0]))


def select_cases(by_method: dict[str, dict[str, dict]], labels: list[str], *, classes=None, with_errors=False):
    """[(case label, class name, sample id)] chosen by the rule in the module docstring."""
    reference = REFERENCE_METHOD if REFERENCE_METHOD in by_method else sorted(by_method)[0]
    common = set.intersection(*(set(records) for records in by_method.values()))
    visits = {sid: by_method[reference][sid] for sid in common
              if probability(by_method[reference][sid]) is not None}
    by_class = defaultdict(list)
    for sid in sorted(visits):
        by_class[labels[visits[sid]["true_class_id"]]].append(sid)
    if classes is None:
        best = {}
        for name, sids in by_class.items():
            group = GROUP_OF_NAME.get(name)
            if group is not None and (group not in best or (len(sids), name) > (len(by_class[best[group]]), best[group])):
                best[group] = name
        classes = [best[g] for g in sorted(best)]
    cases = []
    for name in classes:
        sids = by_class.get(name, [])
        for kind, keep in (("correct", lambda r: r["prediction_class_id"] == r["true_class_id"]),
                           ("error", lambda r: r["prediction_class_id"] != r["true_class_id"])):
            if kind == "error" and not with_errors:
                continue
            chosen = pick_nearest_median([(s, float(probability(visits[s]))) for s in sids if keep(visits[s])])
            if chosen:
                cases.append((kind, name, chosen[0]))
    return cases


def evidence_rows(record: dict, source: str, graph: dict, names: dict, *, top: int, kinds=SHOWN_KINDS,
                  effects: dict | None = None) -> list[dict]:
    """The top nodes of one source of one visit, decoded."""
    vector = source_vector(record, source)
    if vector is None or vector.size != len(graph["nodes"]):
        return []
    signed = None
    if source == "builtin":
        values = (record.get("builtin_detail") or {}).get("signed_node_contributions")
        if values is not None and len(values) == vector.size:
            signed = values
    shown = [i for i, node in enumerate(graph["nodes"]) if node["kind"] in kinds and vector[i] > 0]
    shown.sort(key=lambda i: (-vector[i], i))
    rows = []
    for rank, i in enumerate(shown[:top], start=1):
        row = {"rank": rank, "importance": float(vector[i]), **decode_node(graph["nodes"][i], names)}
        row["signed_contribution"] = None if signed is None else float(signed[i])
        row["effect"] = None if effects is None else effects.get(i)
        rows.append(row)
    return rows


def deletion_effects(run_dir, wanted: dict[str, tuple[int, set[int]]], *, fold: str = "test") -> dict[str, dict[int, float | None]]:
    """{sample id: {node position: drop in the predicted-class probability when that node is deleted}}.

    `wanted` maps a sample id to (predicted class, positions). The run is reloaded and must pass the explanation
    runner's replay check; a node of a protected kind gets None."""
    from core.explain import fidelity_v2 as fv2
    from core.explain import runner
    from core import NODE_KINDS

    bundle = runner.load_run_bundle(run_dir, fold)
    binding = bundle.binding
    splits = runner.rebuild_split(binding, fold)
    model, is_method_adapter = runner.build_model(binding, bundle.state_dict)
    runner.replay_check(model, splits, binding, is_method_adapter=is_method_adapter, fold=fold,
                        stored_npz=Path(run_dir) / f"{fold}.npz")
    data_of = {str(d.sample_id): d for d in splits[fold]}
    records = splits.visit_records(list(wanted))
    wrapper = runner.build_wrapper(binding, model)
    eligible = {NODE_KINDS.index(kind) for kind in fv2.EVIDENCE_NODE_KINDS}
    out: dict[str, dict[int, float | None]] = {}
    for sample_id, (predicted, positions) in wanted.items():
        scorer = records.scorer(wrapper.deletion_forward, data_of[sample_id])
        base = float(scorer.probs(()).view(-1)[predicted])
        out[sample_id] = {}
        for position in sorted(positions):
            if int(scorer.node_type[position]) not in eligible:
                out[sample_id][position] = None
                continue
            out[sample_id][position] = base - float(scorer.probs([position]).view(-1)[predicted])
    return out


def build_cases(by_method, chosen, graphs, labels, names, *, sources, top, effects_for=None):
    """The decoded cases for the chosen (kind, class name, sample id) of every method that explained them."""
    cases = []
    for number, (kind, class_name, sample_id) in enumerate(chosen, start=1):
        graph = graphs[sample_id]
        any_record = next(iter(by_method.values()))[sample_id]
        case = {"case": f"case-{number:02d}", "kind": kind, "true_class": class_name,
                "patient": patient_line(graph, any_record["num_visits"]),
                "patient_key": hashlib.sha256(f"decode:{any_record['subject_id']}".encode()).hexdigest()[:12],
                "methods": {}}
        for method, records in by_method.items():
            record = records[sample_id]
            entry = {"prediction": labels[record["prediction_class_id"]], "probability": probability(record), "sources": {}}
            for source in sources:
                shown_effects = None if effects_for is None else effects_for.get(method, {}).get(sample_id)
                entry["sources"][source] = evidence_rows(record, source, graph, names, top=top, effects=shown_effects)
            case["methods"][method] = entry
        cases.append(case)
    return cases


def render(cases) -> str:
    lines = ["# Patient examples: evidence per method", "",
             "Visits are chosen by a fixed rule (see the module docstring of `decode_examples`), not by eye. Times are hours "
             "relative to the prediction cutoff (negative = before). `signed` is CEI-GNN's ledger (negative lowers the explained "
             "class score); `effect` is the drop in the predicted-class probability when that node is deleted "
             "(positive = the model needs it). Neither is a clinical cause.", ""]
    for case in cases:
        lines += [f"## {case['case']}: {case['true_class']} ({case['kind']} prediction by the reference model)", "",
                  f"Patient: {case['patient']}.", ""]
        for method, entry in case["methods"].items():
            p = entry["probability"]
            lines += [f"### {method}: predicts {entry['prediction']}" + ("" if p is None else f" (p = {p:.2f})"), ""]
            for source, rows in entry["sources"].items():
                lines += [f"**{source}**", "", "| # | kind | evidence | value | level | time | importance | signed | effect |",
                          "|---|---|---|---|---|---|---|---|---|"]
                for r in rows:
                    value = r["value"] or "-"
                    if r["range"]:
                        value += f" (ref {r['range']})"
                    signed = "-" if r["signed_contribution"] is None else f"{r['signed_contribution']:+.3f}"
                    effect = "-" if r["effect"] is None else f"{r['effect']:+.3f}"
                    time = "-" if r["time_hours"] is None else f"{r['time_hours']:+.1f} h"
                    lines.append(f"| {r['rank']} | {r['kind']} | {r['label']} | {value} | {r['level'] or '-'} | {time} | "
                                 f"{r['importance']:.3f} | {signed} | {effect} |")
                lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--explanations", nargs="+", required=True)
    parser.add_argument("--graphs", required=True, type=Path)
    parser.add_argument("--runs", default=DEFAULT_RUNS, type=Path, help="root of the trained runs")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--classes", default=None, help="comma-separated class names (default: one per diagnosis group)")
    parser.add_argument("--with-errors", action="store_true")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--names", default=None, type=Path)
    parser.add_argument("--effects", action="store_true", help="add the deletion effect of each shown node (reloads the models)")
    parser.add_argument("--fold", default="test", choices=("test", "validation", "dev"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"--out must be new or empty: {args.out}")
    runs = load_runs(args.explanations)
    ks = {m.group(1) for name in runs if (m := re.search(r"top(\d+)/", name + "/"))} or ({str(args.k)} if args.k else set())
    if len(ks) != 1:
        raise SystemExit(f"cannot tell K from the run folders ({sorted(ks)}); pass --k with one K per call")
    k = int(next(iter(ks)))
    labels = label_order(args.runs, [f"top{k}/{n.split('/')[-1]}" for n in runs])
    sources = tuple(args.sources.split(","))
    classes = None if args.classes is None else [c.strip() for c in args.classes.split(",")]
    names = load_names(args.names)

    grouped = group_runs(runs)
    by_method = {m: {r["sample_id"]: r for r in by_seed[args.seed]} for m, by_seed in grouped.items() if args.seed in by_seed}
    if not by_method:
        raise SystemExit(f"no run of seed {args.seed} found")
    chosen = select_cases(by_method, labels, classes=classes, with_errors=args.with_errors)
    graphs = read_graphs(args.graphs, [sid for _, _, sid in chosen])
    missing = {sid for _, _, sid in chosen} - set(graphs)
    if missing:
        raise SystemExit(f"{len(missing)} selected visits are not in {args.graphs}")

    effects_for = None
    if args.effects:
        effects_for = {}
        for method, records in by_method.items():
            wanted = {}
            for _, _, sid in chosen:
                record = records[sid]
                positions = set()
                for source in sources:
                    positions |= {r_i for r_i, _ in _top_positions(record, source, graphs[sid], args.top)}
                wanted[sid] = (record["prediction_class_id"], positions)
            effects_for[method] = deletion_effects(args.runs / f"top{k}" / f"{method}_seed{args.seed}", wanted, fold=args.fold)
    cases = build_cases(by_method, chosen, graphs, labels, names, sources=sources, top=args.top, effects_for=effects_for)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "examples.md").write_text(render(cases))
    (args.out / "examples.json").write_text(json.dumps(cases, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out / 'examples.md'} ({len(cases)} cases)")
    return 0


def _top_positions(record, source, graph, top):
    """[(position, importance)] of the rows `evidence_rows` will show, to know which nodes need an effect."""
    vector = source_vector(record, source)
    if vector is None or vector.size != len(graph["nodes"]):
        return []
    shown = [i for i, node in enumerate(graph["nodes"]) if node["kind"] in SHOWN_KINDS and vector[i] > 0]
    shown.sort(key=lambda i: (-vector[i], i))
    return [(i, float(vector[i])) for i in shown[:top]]


if __name__ == "__main__":
    raise SystemExit(main())
