"""Report of the fairness checks on finished explanation outputs.

    python -m comparison.explain.fair_report --explanations <dir or run dirs...> --out <new dir> \
        [--bootstrap 2000] [--seed 1234]

Reads the record files of `core.explain.runner` (inference-free). By default the seeds of one method are
pooled (`--per-seed` reports each run alone): a patient's number is then the mean over its visits and seeds,
and comparisons are between methods, never between seeds. The patient is the unit: a
patient's number is the mean over its visits, intervals are percentile intervals of a patient-level
bootstrap, and a difference between two things is taken on the patients that have both.

Sections written to `fair_report.md` / `fair_report.json`:

1. Budgets. For every method and source: how many eligible evidence nodes the main fidelity cut
   (top 20%, capped at the explanation's own support) actually deleted, against the visit's eligible
   nodes. Different k means different questions are being asked.
2. Matched budgets. Fidelity+ and fidelity- of every source at the same fixed k, with the
   difference to random deletion of the same size on the same visits.
3. ProtGNN subgraph. Size, kinds and removable count of the subgraphs; what deleting (or keeping)
   the subgraph does to the explained prototype's similarity, the prototype's logit share and the
   class probability, against random nodes of the same number.
4. Explainers compared on the same patients: individual intervals and paired bootstrap differences
   between explainers of one method, and of one explainer across methods.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

MIN_PATIENTS_FOR_INTERVAL = 10


def load_runs(paths) -> dict[str, list[dict]]:
    """{run name: [records]} from output dirs (a dir holding subject_*.json, or a dir of such dirs)."""
    runs: dict[str, list[dict]] = {}
    for path in map(Path, paths):
        folders = [path] if list(path.glob("subject_*.json")) else sorted(
            p for p in list(path.glob("*")) + list(path.glob("*/*"))
            if p.is_dir() and list(p.glob("subject_*.json")))
        for folder in folders:
            name = f"{folder.parent.name}/{folder.name}" if folder != path else folder.name
            runs[name] = [json.loads(f.read_text()) for f in sorted(folder.glob("subject_*.json"))]
    if not runs:
        raise SystemExit("no explanation records found")
    return runs


def pool_seeds(runs: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """One entry per method and K: the records of all its seeds (`top20/cei_mp2_seed7` -> `top20/cei_mp2`).

    A patient's number is then the mean over its visits and over the seeds, so the interval is the patient-level
    sampling uncertainty of the pooled result; the spread between seeds is not in it (the summary table gives that)."""
    pooled: dict[str, list[dict]] = {}
    for name, records in runs.items():
        match = re.match(r"^(.*)_seed\d+$", name)
        pooled.setdefault(match.group(1) if match else name, []).extend(records)
    return pooled


def by_patient(records, value) -> dict[str, float]:
    """Mean over a patient's visits of value(record); records where it is None are skipped."""
    grouped = defaultdict(list)
    for record in records:
        number = value(record)
        if number is not None and np.isfinite(number):
            grouped[str(record["subject_id"])].append(float(number))
    return {patient: float(np.mean(v)) for patient, v in grouped.items()}


def interval(per_patient: dict[str, float], *, bootstrap: int, seed: int) -> dict:
    values = np.asarray(list(per_patient.values()), dtype=float)
    if values.size == 0:
        return {"n": 0, "mean": None, "ci": None}
    mean = float(values.mean())
    if values.size < MIN_PATIENTS_FOR_INTERVAL:
        return {"n": int(values.size), "mean": mean, "ci": None}
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(bootstrap, values.size))].mean(axis=1)
    return {"n": int(values.size), "mean": mean,
            "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


def paired_interval(a: dict[str, float], b: dict[str, float], *, bootstrap: int, seed: int) -> dict:
    common = sorted(set(a) & set(b))
    return interval({p: a[p] - b[p] for p in common}, bootstrap=bootstrap, seed=seed)


def fmt(entry: dict, digits: int = 3) -> str:
    if entry["mean"] is None:
        return "-"
    text = f"{entry['mean']:+.{digits}f}"
    if entry["ci"] is not None:
        text += f" [{entry['ci'][0]:+.{digits}f}, {entry['ci'][1]:+.{digits}f}]"
    return text + f" (n={entry['n']})"


def sources_of(records) -> list[str]:
    names = set()
    for record in records:
        names.update(name for name, m in record.get("fidelity_v2", {}).items() if "fidelity_plus" in m)
    return sorted(names, key=lambda n: (n != "builtin", n))


def section_budgets(runs, *, bootstrap, seed):
    lines = ["## 1. How many nodes the main fidelity cut removes", "",
             "Main cut = top 20% of the visit's eligible evidence nodes, capped at the explanation's own support. "
             "`k` is the number actually deleted; `eligible` the visit's eligible nodes; `capped` the share of visits "
             "where the support, not the 20%, set k.", ""]
    out = {}
    for run, records in runs.items():
        lines += [f"**{run}**", ""]
        for source in sources_of(records):
            def get(field):
                return lambda r: r.get("fidelity_v2", {}).get(source, {}).get("fidelity_plus", {}).get(field)
            k = by_patient(records, get("k"))
            eligible = by_patient(records, get("removable_node_count"))
            capped = by_patient(records, lambda r, s=source: (
                None if "fidelity_plus" not in r.get("fidelity_v2", {}).get(s, {}) else
                float(r["fidelity_v2"][s]["fidelity_plus"]["k"] < r["fidelity_v2"][s]["fidelity_plus"]["k_requested"])))
            lines.append(f"- {source}: k = {np.mean(list(k.values())):.1f}, eligible = {np.mean(list(eligible.values())):.1f}, "
                         f"capped by support in {100 * np.mean(list(capped.values())):.0f}% of patients "
                         f"(n={len(k)})")
            out.setdefault(run, {})[source] = {"k": float(np.mean(list(k.values()))),
                                               "eligible": float(np.mean(list(eligible.values()))),
                                               "capped_share": float(np.mean(list(capped.values()))), "n": len(k)}
        lines.append("")
    return lines, out


def section_matched(runs, *, bootstrap, seed):
    lines = ["## 2. Matched evidence budgets", "",
             "Every source scored at the same fixed k. Cells: probability drop with the top-k deleted (fidelity+, "
             "higher = more faithful) / with only the top-k kept (fidelity-, lower = more faithful); then the "
             "difference to random deletion of the same size on the same visits (+ column: higher is better; "
             "- column: lower is better). A source with fewer than k eligible nodes of nonzero importance is "
             "`beyond support` at k and is not padded.", ""]
    out = {}
    for run, records in runs.items():
        ks = sorted({int(k) for r in records for k in r.get("matched_budget", {}).get("ks", [])})
        if not ks:
            continue
        lines += [f"### {run}", ""]
        for source in sources_of(records):
            for k in ks:
                def pick(field, source=source, k=k):
                    def f(record):
                        row = record.get("matched_budget", {}).get("sources", {}).get(source, {}).get("by_k", {}).get(str(k), {})
                        return row.get(field)
                    return f
                reached = by_patient(records, lambda r, s=source, kk=k: (
                    None if "by_k" not in r.get("matched_budget", {}).get("sources", {}).get(s, {}) else
                    float(r["matched_budget"]["sources"][s]["by_k"].get(str(kk), {}).get("status") != "beyond_support")))
                if not reached:
                    continue
                share = 100 * np.mean(list(reached.values()))
                plus, minus = by_patient(records, pick("plus")), by_patient(records, pick("minus"))
                dplus = by_patient(records, pick("plus_minus_random"))
                dminus = by_patient(records, pick("minus_minus_random"))
                key = f"{run}|{source}|k{k}"
                out[key] = {"reached_share": share / 100,
                            "plus": interval(plus, bootstrap=bootstrap, seed=seed),
                            "minus": interval(minus, bootstrap=bootstrap, seed=seed),
                            "plus_minus_random": interval(dplus, bootstrap=bootstrap, seed=seed),
                            "minus_minus_random": interval(dminus, bootstrap=bootstrap, seed=seed)}
                if share < 100 and not plus:
                    lines.append(f"- {source}, k={k}: beyond support on every visit")
                    continue
                lines.append(
                    f"- {source}, k={k}: fidelity+ {fmt(out[key]['plus'])}, vs random {fmt(out[key]['plus_minus_random'])}; "
                    f"fidelity- {fmt(out[key]['minus'])}, vs random {fmt(out[key]['minus_minus_random'])}"
                    + ("" if share >= 99.5 else f"; reached by {share:.0f}% of patients"))
        lines.append("")
    return lines, out


def section_protgnn(runs, *, bootstrap, seed):
    lines = ["## 3. ProtGNN: does the subgraph carry the evidence the prediction used?", ""]
    out = {}
    for run, records in runs.items():
        checks = [(r, r["protgnn_prototype_check"]) for r in records if "protgnn_prototype_check" in r]
        if not checks:
            continue
        ok = [(r, c) for r, c in checks if c.get("status") == "ok"]
        lines += [f"### {run}", "",
                  f"visits with a check: {len(checks)}; with at least one removable node in the subgraph: {len(ok)}", ""]
        sizes = by_patient([r for r, _ in checks], lambda r: r["protgnn_prototype_check"]["subgraph_size"])
        removable = by_patient([r for r, _ in checks], lambda r: r["protgnn_prototype_check"]["subgraph_removable"])
        eligible = by_patient([r for r, _ in checks], lambda r: r["protgnn_prototype_check"]["removable_node_count"])
        kinds = defaultdict(int)
        for _, c in checks:
            for kind, n in c["subgraph_kinds"].items():
                kinds[kind] += n
        total = sum(kinds.values())
        lines += [f"- subgraph size: {np.mean(list(sizes.values())):.1f} nodes, of which removable "
                  f"{np.mean(list(removable.values())):.1f}; the visit has {np.mean(list(eligible.values())):.0f} eligible nodes",
                  "- node kinds in the subgraphs: " + ", ".join(f"{k} {100 * v / total:.0f}%" for k, v in
                                                              sorted(kinds.items(), key=lambda kv: -kv[1])), ""]
        out[run] = {"subgraph_size": float(np.mean(list(sizes.values()))),
                    "subgraph_removable": float(np.mean(list(removable.values()))),
                    "eligible": float(np.mean(list(eligible.values()))),
                    "kinds": dict(kinds)}
        if not ok:
            continue
        recs = [r for r, _ in ok]

        def drop(quantity, scenario, baseline="base"):
            def f(r):
                c = r["protgnn_prototype_check"]
                a = c[baseline][quantity]
                b = c[scenario][quantity]
                b = b["mean"] if isinstance(b, dict) else b
                return a - b
            return f

        def keep(quantity, scenario):
            def f(r):
                c = r["protgnn_prototype_check"]
                b = c[scenario][quantity]
                b = b["mean"] if isinstance(b, dict) else b
                return b
            return f
        lines += ["Change from the untouched visit (positive = the quantity went down). "
                  "Deleting the subgraph's eligible nodes vs deleting the same number of random eligible nodes; "
                  "the last column is the paired difference (subgraph - random).", "",
                  "| quantity | subgraph deleted | random deleted | difference |", "|---|---|---|---|"]
        for quantity, label in (("activation", "prototype similarity (activation)"),
                                ("contribution", "prototype's share of the class logit"),
                                ("probability", "class probability")):
            a = by_patient(recs, drop(quantity, "deleted"))
            b = by_patient(recs, drop(quantity, "random_deleted"))
            lines.append(f"| {label} | {fmt(interval(a, bootstrap=bootstrap, seed=seed))} | "
                         f"{fmt(interval(b, bootstrap=bootstrap, seed=seed))} | "
                         f"{fmt(paired_interval(a, b, bootstrap=bootstrap, seed=seed))} |")
            out[run][f"delete_{quantity}"] = {"subgraph": interval(a, bootstrap=bootstrap, seed=seed),
                                              "random": interval(b, bootstrap=bootstrap, seed=seed),
                                              "difference": paired_interval(a, b, bootstrap=bootstrap, seed=seed)}
        lines += ["", "Keeping only the subgraph's eligible nodes (all other eligible nodes deleted) vs keeping the same "
                  "number of random ones: values after the intervention, as a fraction of the untouched visit's value "
                  "(1 = fully preserved).", "",
                  "| quantity | subgraph kept | random kept | difference |", "|---|---|---|---|"]
        for quantity, label in (("activation", "prototype similarity (activation)"),
                                ("probability", "class probability")):
            def ratio(scenario, quantity=quantity):
                def f(r):
                    c = r["protgnn_prototype_check"]
                    v = c[scenario][quantity]
                    v = v["mean"] if isinstance(v, dict) else v
                    base = c["base"][quantity]
                    return v / base if base else None
                return f
            a = by_patient(recs, ratio("kept_only"))
            b = by_patient(recs, ratio("random_kept"))
            lines.append(f"| {label} | {fmt(interval(a, bootstrap=bootstrap, seed=seed))} | "
                         f"{fmt(interval(b, bootstrap=bootstrap, seed=seed))} | "
                         f"{fmt(paired_interval(a, b, bootstrap=bootstrap, seed=seed))} |")
            out[run][f"keep_{quantity}"] = {"subgraph": interval(a, bootstrap=bootstrap, seed=seed),
                                            "random": interval(b, bootstrap=bootstrap, seed=seed),
                                            "difference": paired_interval(a, b, bootstrap=bootstrap, seed=seed)}
        contribution = by_patient(recs, lambda r: r["protgnn_prototype_check"]["base"]["contribution"])
        logit = by_patient(recs, lambda r: r["protgnn_prototype_check"]["base"]["class_logit"])
        lines += ["", f"On the untouched visits the explained prototype contributes {np.mean(list(contribution.values())):.2f} "
                  f"to the class logit, whose value is {np.mean(list(logit.values())):.2f} (means over patients; the logit is "
                  "the sum over all prototypes of the classifier weight times the activation).", ""]
    return lines, out


METRICS = {
    "fidelity+": (lambda m: m["fidelity_plus"]["prob"], "higher = more faithful"),
    "fidelity-": (lambda m: m["fidelity_minus"]["prob"], "lower = more faithful"),
    "deletion vs random": (lambda m: m["vs_random"]["deletion_mean_change"]["difference"],
                           "width-matched difference to random deletion; higher is better"),
    "insertion vs random": (lambda m: m["vs_random"]["insertion_mean_change"]["difference"],
                            "width-matched difference to random insertion; lower is better"),
    "sparsity": (lambda m: m["sparsity"]["mass90_sparsity"], "share of eligible nodes not needed for 90% of the mass"),
}


def source_values(records, source, getter) -> dict[str, float]:
    def value(record):
        metrics = record.get("fidelity_v2", {}).get(source)
        if not metrics or "fidelity_plus" not in metrics:
            return None
        try:
            return getter(metrics)
        except (KeyError, TypeError):
            return None
    return by_patient(records, value)


def section_paired(runs, *, bootstrap, seed):
    lines = ["## 4. Explainers compared on the same patients", "",
             "Individual interval of each explainer, then every pair of explainers of one method as a paired "
             "difference (A - B, over the patients both have). Intervals are withheld below "
             f"{MIN_PATIENTS_FOR_INTERVAL} patients. With few patients none of this is a ranking.", ""]
    out = {}
    for run, records in runs.items():
        sources = sources_of(records)
        lines += [f"### {run}", ""]
        for metric, (getter, note) in METRICS.items():
            values = {s: source_values(records, s, getter) for s in sources}
            values = {s: v for s, v in values.items() if v}
            if len(values) < 1:
                continue
            lines += [f"**{metric}** ({note})", ""]
            out.setdefault(run, {})[metric] = {"individual": {}, "paired": {}}
            for s, v in values.items():
                entry = interval(v, bootstrap=bootstrap, seed=seed)
                out[run][metric]["individual"][s] = entry
                lines.append(f"- {s}: {fmt(entry)}")
            names = sorted(values)
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    entry = paired_interval(values[a], values[b], bootstrap=bootstrap, seed=seed)
                    out[run][metric]["paired"][f"{a} - {b}"] = entry
                    lines.append(f"- {a} - {b}: {fmt(entry)}")
            lines.append("")
    # the same source across methods, on the patients every method explained
    methods = {run: records for run, records in runs.items()}
    if len(methods) > 1:
        lines += ["### Same explainer across methods (paired over patients)", ""]
        common_sources = sorted(set.intersection(*[set(sources_of(r)) for r in methods.values()]))
        out["across_methods"] = {}
        for metric, (getter, note) in METRICS.items():
            for source in common_sources:
                values = {run: source_values(records, source, getter) for run, records in methods.items()}
                names = sorted(values)
                for i, a in enumerate(names):
                    for b in names[i + 1:]:
                        entry = paired_interval(values[a], values[b], bootstrap=bootstrap, seed=seed)
                        out["across_methods"][f"{metric}|{source}|{a} - {b}"] = entry
                        lines.append(f"- {metric}, {source}, {a} - {b}: {fmt(entry)}")
        lines.append("")
    return lines, out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--explanations", nargs="+", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--per-seed", action="store_true",
                        help="report every seed run on its own instead of one result per method pooled over its seeds")
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"--out must be new or empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.explanations)
    if not args.per_seed:
        runs = pool_seeds(runs)
    kwargs = dict(bootstrap=args.bootstrap, seed=args.seed)
    document = ["# Fairness checks on the explanation results", "",
                f"Patient-level bootstrap, {args.bootstrap} resamples, 95% percentile intervals; intervals are withheld "
                f"below {MIN_PATIENTS_FOR_INTERVAL} patients.", ""]
    result = {}
    for name, section in (("budgets", section_budgets), ("matched_budget", section_matched),
                          ("protgnn_prototype", section_protgnn), ("paired", section_paired)):
        lines, data = section(runs, **kwargs)
        document += lines
        result[name] = data
    (args.out / "fair_report.md").write_text("\n".join(document) + "\n")
    (args.out / "fair_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out / 'fair_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
