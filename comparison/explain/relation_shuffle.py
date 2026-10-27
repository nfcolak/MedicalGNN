"""Do the trained models depend on which nodes their relations connect?

    python -m comparison.explain.relation_shuffle run    --runs <dir with top<K>/<method>_seed<N>> --out <new dir> \
        [--ks 10,20,30] [--methods cei_mp2,gsat,graphcare,protgnn] [--seeds 1234,2025,7] [--fold test] \
        [--sample-patients 500] [--draws 3] [--execute]
    python -m comparison.explain.relation_shuffle report --probes <out of run> --out <new dir> [--bootstrap 2000]

`run` takes each trained final run as it is (the checkpoint is reloaded and must pass the explanation
runner's replay check), re-encodes every explained visit from its graph record with the run's own
preprocessing, and scores it again after changing the relations, through the model's own forward and
the same re-encoding the deletion fidelity uses. Nothing is trained or fitted. Without `--execute` it
only prints the plan.

Arms (relation sets: `core.tensorize.INFORMATIVE_RELATIONS` / `STRUCTURAL_RELATIONS`):

* shuffle_informative / shuffle_structural / shuffle_all: the edges of those relations are swapped
  within the relation and the graph (`core.rewiring`: directed double-edge swaps, every node keeps its
  in- and out-degree, endpoints keep their node kind, no node is removed). `--draws` seeded shuffles
  per visit. An edge whose endpoints change gets its pair payload (lab delta, rate, interval,
  recurrence counts) recomputed from its new endpoints; nothing of the old pair is carried over.
* drop_informative / drop_all: the same relations removed; `drop_all` leaves no edge at all. These are
  the references: a shuffle that hurts less than dropping the relations means the model uses that the
  relations exist more than which nodes they connect.

Per visit and arm it records the predicted class after the change, the probability of the class the
unchanged model predicted, how many edges were replaced and how many remain. `report` gives, per
method and K: accuracy and macro-F1 before and after, the share of visits whose prediction flips, the
change of the predicted-class probability and the share of visits where at least one edge moved
(a shuffle can be a no-op on a small or complete graph). The patient is the unit, intervals are a
patient-level bootstrap withheld below 10 patients; macro-F1 is a mean and SD over seeds.

What this shows and does not show: the models were never trained on shuffled graphs, so a change is a
dependence of the trained model on the relations, not proof that it uses them in a clinically meaningful
way; the training-time version of this control (`train.py --rewire_relation`) exists for CEI-GNN only
and was not run. The test fold is the report fold; nothing is selected from these numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED_BASE = 1000
SAMPLE_PATIENTS = 500
DRAWS = 3
METHODS = ("cei_mp2", "gsat", "graphcare", "protgnn")
SEEDS = (1234, 2025, 7)
RUN_NAME = re.compile(r"^(?P<method>.+)_seed(?P<seed>\d+)$")
MIN_PATIENTS_FOR_INTERVAL = 10


def arms() -> dict[str, tuple[str, tuple[str, ...]]]:
    """{arm: (kind, relations)}; imported lazily so `report` needs no torch."""
    from core.tensorize import INFORMATIVE_RELATIONS, STRUCTURAL_RELATIONS

    informative, structural = tuple(INFORMATIVE_RELATIONS), tuple(STRUCTURAL_RELATIONS)
    return {"shuffle_informative": ("shuffle", informative), "shuffle_structural": ("shuffle", structural),
            "shuffle_all": ("shuffle", structural + informative),
            "drop_informative": ("drop", informative), "drop_all": ("drop", structural + informative)}


def patient_key(subject) -> str:
    return hashlib.sha256(f"relation_shuffle:{subject}".encode()).hexdigest()[:12]


def probe_run(run_dir, out_dir, *, fold: str = "test", sample_patients: int | None = SAMPLE_PATIENTS,
              sample_seed: int = 1234, draws: int = DRAWS, max_subjects: int | None = None) -> dict:
    """Probe one trained run; writes `visits.jsonl` and `relation_probe.json` into the new `out_dir`."""
    import torch

    from core.explain import runner
    from core.tensorize import encode_graph

    run_dir, out_dir = Path(run_dir), Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {out_dir}")
    started = time.perf_counter()
    bundle = runner.load_run_bundle(run_dir, fold)
    binding = bundle.binding
    if binding.get("rewired_relations"):
        raise ValueError("the run was trained on rewired edges; its graph records are not the real relations")
    splits = runner.rebuild_split(binding, fold)
    model, is_method_adapter = runner.build_model(binding, bundle.state_dict)
    replay = runner.replay_check(model, splits, binding, is_method_adapter=is_method_adapter, fold=fold,
                                 stored_npz=run_dir / f"{fold}.npz")
    subjects = runner.select_explained_subjects(splits[fold], max_subjects=max_subjects,
                                                sample_patients=sample_patients, sample_seed=sample_seed)
    records = splits.visit_records([d.sample_id for d in subjects])
    wrapper = runner.build_wrapper(binding, model)
    prep = splits.preprocessing
    dropped = tuple(binding.get("dropped_relations") or ())
    plan = arms()

    def encode(graph, membership, kind, relations, seed):
        data = encode_graph(graph, prep, membership, binding["edges"],
                            dropped + (relations if kind == "drop" else ()),
                            relations if kind == "shuffle" else (), seed,
                            edge_direction=binding["edge_direction"])
        return data

    def probabilities(data, y):
        data.y = torch.tensor([y], dtype=torch.long)
        with torch.no_grad():
            return torch.softmax(wrapper.deletion_forward(data), dim=-1).view(-1)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(out_dir / "visits.jsonl", "w") as handle:
        for data in subjects:
            sample_id = str(data.sample_id)
            scorer = records.scorer(wrapper.deletion_forward, data)      # refuses a record that is not the split tensor
            graph, membership = records.records[sample_id]
            y = int(torch.as_tensor(data.y).view(-1)[0])
            base = scorer.probs(()).view(-1)
            predicted = int(base.argmax())
            row = {"sample_id": sample_id, "patient": patient_key(data.subject), "y": y, "pred": predicted,
                   "base_prob": float(base[predicted]), "edges": int(scorer.data.edge_index.size(1)), "arms": {}}
            for arm, (kind, relations) in plan.items():
                results = []
                for draw in range(draws if kind == "shuffle" else 1):
                    changed = encode(graph, membership, kind, relations, SEED_BASE + draw)
                    probs = probabilities(changed, y)
                    results.append({"pred": int(probs.argmax()), "prob": float(probs[predicted]),
                                    "moved": int(getattr(changed, "rewired_edge_count", 0)),
                                    "edges": int(changed.edge_index.size(1))})
                row["arms"][arm] = results
            handle.write(json.dumps(row) + "\n")
            rows += 1
    header = {"schema": "medgnn.relation_probe", "schema_version": 1, "run": f"{run_dir.parent.name}/{run_dir.name}",
              "method": binding["method"], "conv": binding.get("conv"), "seed": binding["seed"],
              "k": binding.get("top_k_labels"), "fold": fold, "replay": replay.get("mode"),
              "sample_patients": sample_patients, "sample_seed": sample_seed, "draws": draws,
              "visits": rows, "patients": len({patient_key(d.subject) for d in subjects}),
              "arms": {name: {"kind": kind, "relations": list(relations)} for name, (kind, relations) in plan.items()},
              "code_sha256": hashlib.sha256(json.dumps(runner.explanation_source_hashes(), sort_keys=True)
                                            .encode()).hexdigest(),
              "seconds": round(time.perf_counter() - started, 1)}
    (out_dir / "relation_probe.json").write_text(json.dumps(header, indent=2, sort_keys=True) + "\n")
    return header


def plan_runs(root, ks, methods, seeds) -> list[tuple[int, str, int, Path]]:
    """[(K, method, seed, run dir)] of the trained runs that exist; a missing one stops the plan."""
    planned, missing = [], []
    for k in ks:
        for method in methods:
            for seed in seeds:
                run_dir = Path(root) / f"top{k}" / f"{method}_seed{seed}"
                (planned if (run_dir / "binding.json").is_file() else missing).append((k, method, seed, run_dir))
    if missing:
        raise SystemExit("trained runs missing: " + ", ".join(str(m[3]) for m in missing))
    return planned


def run_command(args) -> int:
    planned = plan_runs(args.runs, [int(k) for k in args.ks.split(",")], args.methods.split(","),
                        [int(s) for s in args.seeds.split(",")])
    sample = None if args.sample_patients == 0 else args.sample_patients
    for k, method, seed, run_dir in planned:
        print(f"top{k}/{method}_seed{seed}: fold {args.fold}, {sample or 'all'} patients, {args.draws} shuffles per arm")
    if not args.execute:
        print(f"{len(planned)} runs planned; nothing executed (pass --execute)")
        return 0
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"--out must be new or empty: {out}")
    for k, method, seed, run_dir in planned:
        header = probe_run(run_dir, out / f"top{k}" / f"{method}_seed{seed}", fold=args.fold, sample_patients=sample,
                           sample_seed=args.sample_seed, draws=args.draws, max_subjects=args.max_subjects)
        print(f"top{k}/{method}_seed{seed}: {header['visits']} visits, replay {header['replay']}, {header['seconds']} s",
              flush=True)
    return 0


# ----------------------------------------------------------------------------------------------- report
def macro_f1(y, pred) -> float:
    y, pred = np.asarray(y), np.asarray(pred)
    scores = []
    for label in sorted(set(y.tolist()) | set(pred.tolist())):
        tp = int(((y == label) & (pred == label)).sum())
        fp = int(((y != label) & (pred == label)).sum())
        fn = int(((y == label) & (pred != label)).sum())
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def interval(per_patient: dict[str, float], *, bootstrap: int, seed: int) -> dict:
    values = np.asarray(list(per_patient.values()), dtype=float)
    if values.size == 0:
        return {"n": 0, "mean": None, "ci": None}
    mean = float(values.mean())
    if values.size < MIN_PATIENTS_FOR_INTERVAL:
        return {"n": int(values.size), "mean": mean, "ci": None}
    draws = values[np.random.default_rng(seed).integers(0, values.size, size=(bootstrap, values.size))].mean(axis=1)
    return {"n": int(values.size), "mean": mean,
            "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


def fmt(entry: dict, scale: float = 1.0, digits: int = 2, signed: bool = True) -> str:
    if entry["mean"] is None:
        return "-"
    sign = "+" if signed else ""
    text = f"{entry['mean'] * scale:{sign}.{digits}f}"
    if entry["ci"] is not None:
        text += f" [{entry['ci'][0] * scale:{sign}.{digits}f}, {entry['ci'][1] * scale:{sign}.{digits}f}]"
    return text + f" (n={entry['n']})"


def load_probes(path) -> dict[tuple[int, str], dict[int, tuple[dict, list[dict]]]]:
    """{(K, method): {seed: (header, visit rows)}} from the folders `run` wrote."""
    found: dict = defaultdict(dict)
    for header_path in sorted(Path(path).glob("top*/*/relation_probe.json")):
        header = json.loads(header_path.read_text())
        match = RUN_NAME.match(header_path.parent.name)
        rows = [json.loads(line) for line in (header_path.parent / "visits.jsonl").read_text().splitlines()]
        found[(int(header["k"]), match["method"])][int(match["seed"])] = (header, rows)
    if not found:
        raise SystemExit("no relation_probe.json found")
    return found


def arm_values(by_seed, arm: str) -> dict[str, dict[str, dict[str, float]]]:
    """Per patient (mean over its visits, seeds and shuffles): accuracy change, flip, probability change,
    moved; plus per seed the macro-F1 and accuracy before/after."""
    per_patient = {m: defaultdict(list) for m in ("d_acc", "flip", "d_prob", "moved", "drop_edges")}
    per_seed = {}
    for seed, (header, rows) in by_seed.items():
        y = np.array([r["y"] for r in rows])
        base = np.array([r["pred"] for r in rows])
        draws = len(rows[0]["arms"][arm]) if rows else 0
        f1s, accs = [], []
        for d in range(draws):
            after = np.array([r["arms"][arm][d]["pred"] for r in rows])
            f1s.append(macro_f1(y, after) - macro_f1(y, base))
            accs.append(float((after == y).mean() - (base == y).mean()))
            for r, a in zip(rows, after):
                item = r["arms"][arm][d]
                per_patient["d_acc"][r["patient"]].append(float(a == r["y"]) - float(r["pred"] == r["y"]))
                per_patient["flip"][r["patient"]].append(float(a != r["pred"]))
                per_patient["d_prob"][r["patient"]].append(item["prob"] - r["base_prob"])
                per_patient["moved"][r["patient"]].append(float(item["moved"] > 0))
                per_patient["drop_edges"][r["patient"]].append(float(item["edges"] < r["edges"]))
        per_seed[seed] = {"d_f1": float(np.mean(f1s)) if f1s else None, "d_acc": float(np.mean(accs)) if accs else None,
                          "base_acc": float((base == y).mean()) if rows else None,
                          "base_f1": macro_f1(y, base) if rows else None}
    patients = {m: {p: float(np.mean(v)) for p, v in d.items()} for m, d in per_patient.items()}
    return patients, per_seed


def probe_report(found, *, bootstrap: int, seed: int):
    names = list(next(iter(next(iter(found.values())).values()))[0]["arms"])
    lines = ["## Effect of changing the relations of the trained models", "",
             "Accuracy and probability changes are after minus before, per patient (mean over visits, seeds and shuffles), "
             "patient-level 95% interval. Flip = the predicted class changes. Moved/removed = share of visits where at least "
             "one edge was replaced (shuffle) or the graph lost edges (drop). Macro-F1 change: mean ± SD over seeds.", ""]
    data = {}
    for k in sorted({key[0] for key in found}):
        lines += [f"### K = {k}", ""]
        for method in sorted(m for kk, m in found if kk == k):
            by_seed = found[(k, method)]
            headers = [h for h, _ in by_seed.values()]
            base_acc = [v["base_acc"] for v in arm_values(by_seed, names[0])[1].values()]
            lines += [f"**{method}** (seeds {sorted(by_seed)}, {headers[0]['visits']} visits, {headers[0]['patients']} "
                      f"patients, {headers[0]['draws']} shuffles per arm, accuracy before {np.mean(base_acc):.3f})", "",
                      "| arm | Δ accuracy (pp) | flip rate (%) | Δ probability of predicted class | moved / removed (%) | Δ macro-F1 |",
                      "|---|---|---|---|---|---|"]
            for arm in names:
                patients, per_seed = arm_values(by_seed, arm)
                entry = {m: interval(patients[m], bootstrap=bootstrap, seed=seed)
                         for m in ("d_acc", "flip", "d_prob", "moved", "drop_edges")}
                f1 = [v["d_f1"] for v in per_seed.values() if v["d_f1"] is not None]
                sd = float(np.std(f1, ddof=1)) if len(f1) > 1 else None
                kind = headers[0]["arms"][arm]["kind"]
                share = entry["moved" if kind == "shuffle" else "drop_edges"]
                lines.append(f"| {arm} | {fmt(entry['d_acc'], 100)} | {fmt(entry['flip'], 100, signed=False)} | "
                             f"{fmt(entry['d_prob'], 1, 3)} | {share['mean'] * 100:.1f} | "
                             f"{np.mean(f1):+.4f}" + (f" ± {sd:.4f}" if sd is not None else "") + " |")
                data[f"{k}/{method}/{arm}"] = {**entry, "d_macro_f1_per_seed": {str(s): v["d_f1"] for s, v in per_seed.items()}}
            lines.append("")
    return lines, data


def report_command(args) -> int:
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"--out must be new or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    found = load_probes(args.probes)
    lines, data = probe_report(found, bootstrap=args.bootstrap, seed=args.seed)
    document = ["# Relation shuffling of the trained final models", "",
                f"Patient-level bootstrap, {args.bootstrap} resamples, intervals withheld below {MIN_PATIENTS_FOR_INTERVAL} "
                "patients. The models were never trained on shuffled graphs: read a change as dependence on the relations, "
                "not as proof of meaningful use.", ""] + lines
    (out / "relation_shuffle.md").write_text("\n".join(document) + "\n")
    (out / "relation_shuffle.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out / 'relation_shuffle.md'}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--runs", required=True, type=Path)
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--ks", default="10,20,30")
    run.add_argument("--methods", default=",".join(METHODS))
    run.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    run.add_argument("--fold", default="test", choices=("test", "validation", "dev"))
    run.add_argument("--sample-patients", type=int, default=SAMPLE_PATIENTS, help="0 = every patient of the fold")
    run.add_argument("--sample-seed", type=int, default=1234)
    run.add_argument("--draws", type=int, default=DRAWS)
    run.add_argument("--max-subjects", type=int, default=None)
    run.add_argument("--execute", action="store_true")
    report = sub.add_parser("report")
    report.add_argument("--probes", required=True, type=Path)
    report.add_argument("--out", required=True, type=Path)
    report.add_argument("--bootstrap", type=int, default=2000)
    report.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)
    return run_command(args) if args.command == "run" else report_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
