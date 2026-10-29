"""Do the explanations pass the two sanity controls?

    python -m comparison.explain.audit_report --explanations <dir or run dirs...> --out <new dir> \
        [--bootstrap 2000] [--seed 1234]

Reads the record files of `core.explain.runner` that carry an `audit` block (written when the runner
or the driver is given `--audit-subjects N`: the first N explained visits of a run). Per method and
source, over the visits audited:

* Weight randomisation (Adebayo et al., 2018). Every weight of the model is re-drawn and the visit is
  explained again. An explanation that really describes the model must change: Spearman correlation and
  top-k Jaccard (over the eligible evidence nodes, k from the fidelity cut) between the real and the
  randomised explanation should be low. High agreement means the explanation survives destroying the
  model and is not explaining it. Also given: the share of visits whose prediction changes.
* Stability. Small noise is added to the values of the measured nodes; draws that leave the prediction
  unchanged are explained again. Agreement between the original and the perturbed explanation should be
  high. Also given: the share of draws whose prediction changes (those draws are left out of the agreement).

A source whose explanation is constant has no Spearman (shown "-"). The patient is the unit (mean over its
audited visits, draws and seeds), intervals a patient-level bootstrap withheld below 10 patients. The audit
is a control on a subset of visits, not a measure of quality.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from comparison.explain.fair_report import MIN_PATIENTS_FOR_INTERVAL, interval, load_runs
from comparison.explain.seed_agreement import group_runs

METRICS = ("spearman", "top_k_jaccard")
# kept in the records for reference only; the fidelity tables skip it too (`runner.FIDELITY_SKIPPED_SOURCES`)
SKIPPED_SOURCES = ("IntegratedGradExplainerZeroBaseline",)


def fmt(entry: dict, digits: int = 3) -> str:
    if entry["mean"] is None:
        return "-"
    text = f"{entry['mean']:.{digits}f}"
    if entry["ci"] is not None:
        text += f" [{entry['ci'][0]:.{digits}f}, {entry['ci'][1]:.{digits}f}]"
    return text + f" (n={entry['n']})"


def visit_values(record: dict) -> dict:
    """One audited visit as numbers: {control: {source: {metric: value}}, plus the prediction-change shares}."""
    audit = record.get("audit") or {}
    out = {"weight": {}, "stability": {}, "weight_changed": None, "stability_changed": None}
    weight = audit.get("weight_randomisation") or {}
    if "sources" in weight:
        out["weight_changed"] = float(bool(weight.get("prediction_changed")))
        for source, entry in weight["sources"].items():
            if entry.get("status") != "failed":
                out["weight"][source] = {m: entry.get(m) for m in METRICS}
    draws = [d for d in (audit.get("stability") or {}).get("draws", []) if "prediction_changed" in d]
    if draws:
        out["stability_changed"] = float(np.mean([d["prediction_changed"] for d in draws]))
        kept = [d for d in draws if not d["prediction_changed"] and "sources" in d]
        for source in sorted({s for d in kept for s in d["sources"]}):
            entries = [d["sources"][source] for d in kept if source in d["sources"] and d["sources"][source].get("status") != "failed"]
            out["stability"][source] = {m: (float(np.mean([e[m] for e in entries if e.get(m) is not None]))
                                            if any(e.get(m) is not None for e in entries) else None) for m in METRICS}
    return out


def audited_records(by_seed: dict[int, list[dict]]) -> list[dict]:
    return [r for records in by_seed.values() for r in records if r.get("audit")]


def per_patient(records: list[dict], pick) -> dict[str, float]:
    grouped = defaultdict(list)
    for record in records:
        value = pick(visit_values(record))
        if value is not None and np.isfinite(value):
            grouped[str(record["subject_id"])].append(float(value))
    return {patient: float(np.mean(v)) for patient, v in grouped.items()}


def audit_report(runs, *, bootstrap: int, seed: int):
    lines = ["## Weight randomisation and stability of the explanations", "",
             "Weight randomisation: agreement between the real explanation and the one from a model with re-drawn weights "
             "(low is what a faithful explanation should show). Stability: agreement between the explanation and the one "
             "after small noise on the measured values, on draws where the prediction does not change (high is good).", ""]
    data = {}
    grouped = group_runs(runs)
    for method, by_seed in sorted(grouped.items()):
        records = audited_records(by_seed)
        if not records:
            lines += [f"### {method}", "", "No audited visits (the runs were made without `--audit-subjects`).", ""]
            continue
        sources = sorted({s for r in records for c in ("weight", "stability") for s in visit_values(r)[c]
                          if s not in SKIPPED_SOURCES}, key=lambda n: (n != "builtin", n))
        patients = len({str(r["subject_id"]) for r in records})
        changed_w = interval(per_patient(records, lambda v: v["weight_changed"]), bootstrap=bootstrap, seed=seed)
        changed_s = interval(per_patient(records, lambda v: v["stability_changed"]), bootstrap=bootstrap, seed=seed)
        lines += [f"### {method} (seeds {sorted(by_seed)}, {len(records)} audited visits, {patients} patients)", "",
                  f"Prediction changes under randomised weights: {fmt(changed_w)}; "
                  f"share of noise draws that change the prediction: {fmt(changed_s)}.", "",
                  "| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | "
                  "top-k Jaccard, original vs perturbed |", "|---|---|---|---|---|"]
        entry = {"prediction_changed_weight": changed_w, "prediction_changed_stability": changed_s}
        for source in sources:
            cells = {}
            for control in ("weight", "stability"):
                for metric in METRICS:
                    cells[(control, metric)] = interval(
                        per_patient(records, lambda v, c=control, s=source, m=metric: (v[c].get(s) or {}).get(m)),
                        bootstrap=bootstrap, seed=seed)
            entry[source] = {f"{c}_{m}": cells[(c, m)] for c, m in cells}
            lines.append(f"| {source} | " + " | ".join(fmt(cells[k]) for k in (
                ("weight", "spearman"), ("weight", "top_k_jaccard"), ("stability", "spearman"),
                ("stability", "top_k_jaccard"))) + " |")
        data[method] = entry
        lines.append("")
    return lines, data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--explanations", nargs="+", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"--out must be new or empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.explanations)
    lines, data = audit_report(runs, bootstrap=args.bootstrap, seed=args.seed)
    document = ["# Controls on the explanations", "",
                f"Patient-level bootstrap, {args.bootstrap} resamples, 95% percentile intervals; intervals are withheld "
                f"below {MIN_PATIENTS_FOR_INTERVAL} patients.", ""] + lines
    (args.out / "audit_report.md").write_text("\n".join(document) + "\n")
    (args.out / "audit_report.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out / 'audit_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
