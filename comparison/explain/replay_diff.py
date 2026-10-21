"""Replay difference tool: how closely does a reloaded run reproduce its stored predictions?

Three modes, none of which writes next to the run directory or records an identifier:

    python -m comparison.explain.replay_diff --run-dir <run> --fold test --out <new report.json> \\
        [--dump <new dump.json>] [--max-batches 3] [--threads N]
    python -m comparison.explain.replay_diff --compare <dump A.json> <dump B.json> [--out <new table.json>]

1. Agreement report (`--out`): reloads `best.pt`, re-runs the fold with `fold_probabilities` (the replay loop of the
   explanation runner) and compares every row with the run's stored `<fold>.npz`: row count, max and mean absolute
   difference, argmax agreement, rows over 1e-4, a histogram of the per-row max difference in decades from 1e-8 to
   1e-1, whether the byte hash of the predictions matches, and the environment (torch, python, platform, machine,
   thread count). `runner_rule` repeats the explanation runner's own replay rule on these numbers (the byte hash, or
   the stored predictions hashing to the recorded value, matching labels, the runner's tolerance, argmax flips only on
   stored ties) and says whether the runner would accept this replay; it reads `runner.REPLAY_TOLERANCE`, so the two
   cannot drift apart.
2. Dump (`--dump`): for the first `--max-batches` batches, fingerprints of the batch tensors before the forward, of the
   inputs and outputs of EVERY module (leaves and containers, in call order) through forward hooks, and of every tensor
   attribute named `last_*` on the model after the forward. A fingerprint is shape, dtype, sum, abs-sum, min, max and
   the dot product with a random vector seeded from the tensor's name; no raw tensors, no identifiers. Dumps made on
   two machines are compared with mode 3.
3. Compare (`--compare A B`, no model): the relative difference of every fingerprint = max abs difference over (sum,
   min, max, projection) divided by (abs-sum / numel + 1e-12), the full table in call order with the jump ratio to the
   previous row, and a mark on the module calls whose inputs agree within 10x the table's floor while their outputs
   jump by more than 100x. `START` = no inner call of that module is marked either, so the divergence starts in that
   module or in the code between its inner calls; `enclosing` = an inner call is marked too. The floor is the smallest
   positive relative difference of the table (at least 1e-12); there is no single fixed threshold.

Both writing modes refuse to overwrite an existing file or to write inside the run directory.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from core.explain import runner

BATCH_TENSORS = ("x", "token", "node_type", "edge_index", "edge_attr", "edge_relation", "edge_triple",
                 "visit_membership_index", "num_visits", "global_node_mask")
HISTOGRAM_EDGES = tuple(10.0 ** -e for e in range(8, 0, -1))   # 1e-8 .. 1e-1
OVER_THRESHOLD = 1e-4
JUMP_FACTOR = 100.0
AGREE_FACTOR = 10.0


def _environment() -> dict[str, Any]:
    info = runner.environment_info()
    try:
        import torch_geometric

        info["torch_geometric"] = torch_geometric.__version__
    except Exception:  # noqa: BLE001 - informational
        pass
    return info


def _refuse_target(path: Path | None, run_dir: Path | None) -> None:
    if path is None:
        return
    if path.exists():
        raise FileExistsError(f"refusing to overwrite an existing file: {path}")
    if run_dir is not None:
        try:
            path.resolve().relative_to(run_dir.resolve())
        except ValueError:
            return
        raise ValueError(f"refusing to write inside the run directory: {path}")


def _write_new(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite an existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def default_loader(run_dir, fold: str):
    """(model, is_method_adapter, splits, binding) of a finished run, hash-verified by the runner."""
    bundle = runner.load_run_bundle(run_dir, fold)
    splits = runner.rebuild_split(bundle.binding, fold)
    model, is_method_adapter = runner.build_model(bundle.binding, bundle.state_dict)
    return model, is_method_adapter, splits, bundle.binding


# ---- 1. agreement report -------------------------------------------------------------------------------------------

def replay_rule(stored, proba: np.ndarray, rows, binding: dict[str, Any], fold: str, exact: bool) -> dict[str, Any]:
    """The facts the runner's replay rule decides on, read the way the runner reads them.

    The runner accepts a replay when the byte hash of the replayed predictions equals the recorded one, or, when it
    does not reproduce on this machine, when the stored `<fold>.npz` is itself the recorded run (its predictions hash
    to the recorded value and its labels equal the labels of the rebuilt fold) and every row of the replay lies within
    `runner.REPLAY_TOLERANCE` of it with an argmax flip only on a row whose stored top-two gap is below twice that
    tolerance. `would_pass` applies exactly that; this tool changes nothing in the run directory."""
    tolerance = runner.REPLAY_TOLERANCE
    stored_proba = stored["proba"].astype(np.float64)
    stored_hash_ok = runner.proba_digest(stored["proba"]) == binding[runner.FOLD_BLOCK[fold]]["prediction_sha256"]
    labels = np.array([int(d.y.view(-1)[0]) for d in rows])
    labels_ok = "y" in stored.files and np.array_equal(np.asarray(stored["y"]).reshape(-1), labels)
    max_diff = float(np.abs(stored_proba - proba.astype(np.float64)).max()) if stored_proba.size else 0.0
    flipped = stored_proba.argmax(1) != proba.argmax(1) if len(stored_proba) else np.zeros(0, dtype=bool)
    top_two = np.sort(stored_proba, axis=1)[:, -2:] if stored_proba.shape[1:] and stored_proba.shape[1] > 1 \
        else np.zeros((len(stored_proba), 2))
    tie = (top_two[:, 1] - top_two[:, 0]) < 2 * tolerance
    flips_total, flips_tie = int(flipped.sum()), int((flipped & tie).sum())
    within = max_diff <= tolerance and flips_total == flips_tie
    return {"tolerance": tolerance, "stored_npz_hash_verified": bool(stored_hash_ok),
            "stored_labels_match": bool(labels_ok), "argmax_flips_total": flips_total,
            "argmax_flips_tie": flips_tie,
            "would_pass": bool(exact or (stored_hash_ok and labels_ok and within)),
            "mode": "exact_hash" if exact else ("stored_predictions_tolerance"
                                                 if stored_hash_ok and labels_ok and within else "refused")}


def agreement_report(model, splits, binding, run_dir: Path, fold: str, *, is_method_adapter: bool) -> dict[str, Any]:
    stored_path = Path(run_dir) / f"{fold}.npz"
    if not stored_path.is_file():
        raise FileNotFoundError(f"no stored {fold}.npz in the run directory")
    stored = np.load(stored_path, allow_pickle=False)
    ids = [str(d.sample_id) for d in splits[fold]]
    if ids != [str(s) for s in stored["sample_ids"]]:
        raise ValueError("stored sample ids differ from the rebuilt fold (order or content)")
    batch_size = binding.get("batch_size", 64)
    proba = runner.fold_probabilities(model, splits, fold, batch_size, is_method_adapter)
    stored_proba = stored["proba"].astype(np.float64)
    if stored_proba.shape != proba.shape:
        raise ValueError(f"stored proba shape {stored_proba.shape} differs from the replay {proba.shape}")
    diff = np.abs(stored_proba - proba.astype(np.float64))
    row_max = diff.max(axis=1) if diff.size else np.zeros(0)
    exact = runner.proba_digest(proba) == binding[runner.FOLD_BLOCK[fold]]["prediction_sha256"]
    rule = replay_rule(stored, proba, splits[fold], binding, fold, exact)
    edges = (0.0,) + HISTOGRAM_EDGES + (math.inf,)
    histogram = {}
    for low, high in zip(edges[:-1], edges[1:]):
        label = (f"lt_{high:.0e}" if low == 0.0 else f"ge_{low:.0e}" if math.isinf(high)
                 else f"{low:.0e}_{high:.0e}")
        histogram[label] = int(((row_max >= low) & (row_max < high)).sum())
    return {
        "environment": _environment(),
        "method": binding["method"] + (f"/{binding['conv']}" if binding.get("conv") else ""),
        "fold": fold,
        "rows": int(len(ids)),
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "argmax_agreement": float((stored_proba.argmax(1) == proba.argmax(1)).mean()) if len(ids) else 1.0,
        "rows_over_1e-4": int((row_max > OVER_THRESHOLD).sum()),
        "abs_diff_histogram": histogram,
        "exact_hash_match": exact,
        "runner_rule": rule,
    }


# ---- 2. fingerprint dump ---------------------------------------------------------------------------------------------

def fingerprint(tensor: torch.Tensor, name: str) -> dict[str, Any]:
    """Aggregates of one tensor in float64 plus its projection on a random vector seeded from `name`."""
    values = tensor.detach().to("cpu").double().reshape(-1)
    out: dict[str, Any] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype).replace("torch.", "")}
    if values.numel() == 0:
        return out | {"sum": 0.0, "abs_sum": 0.0, "min": None, "max": None, "projection": 0.0}
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:15], 16)
    generator = torch.Generator().manual_seed(seed)
    vector = torch.randn(values.numel(), generator=generator, dtype=torch.float64)
    return out | {"sum": float(values.sum()), "abs_sum": float(values.abs().sum()),
                  "min": float(values.min()), "max": float(values.max()),
                  "projection": float(torch.dot(values, vector))}


def _tensors_in(obj, path: str):
    if isinstance(obj, torch.Tensor):
        yield path, obj
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for field in dataclasses.fields(obj):
            yield from _tensors_in(getattr(obj, field.name), f"{path}.{field.name}")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from _tensors_in(value, f"{path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            yield from _tensors_in(value, f"{path}.{index}")


def _fingerprints(call: int, module_name: str, side: str, args, kwargs=None) -> dict[str, Any]:
    prints = {}
    pairs = list(_tensors_in(list(args), "a"))
    pairs += list(_tensors_in(dict(kwargs or {}), "k"))
    for path, tensor in pairs:
        prints[path] = fingerprint(tensor, f"{call}:{module_name}:{side}:{path}")
    return prints


def dump_fingerprints(model, splits, fold: str, batch_size: int, is_method_adapter: bool,
                      max_batches: int = 3, device="cpu") -> dict[str, Any]:
    """Fingerprints of the first `max_batches` batches of `fold` (see the module docstring)."""
    model = model.to(device).eval()
    loader = DataLoader(splits[fold], batch_size=batch_size)
    batches = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch = batch.to(device)
        entry: dict[str, Any] = {"batch": batch_index, "inputs": {}, "calls": [], "last": {}}
        for name in BATCH_TENSORS:
            value = getattr(batch, name, None)
            if isinstance(value, torch.Tensor):
                entry["inputs"][name] = fingerprint(value, f"batch:{name}")
        calls = entry["calls"]
        _names = {id(m): (n or "<root>") for n, m in model.named_modules()}
        open_calls: dict[int, list[int]] = {}      # id(module) -> stack of call indices
        running: list[int] = []                    # call indices in progress, innermost last
        handles = []

        def pre_hook(module, args, kwargs):
            index = len(calls)
            calls.append({"call": index, "module": _names[id(module)], "type": type(module).__name__,
                          "parent": running[-1] if running else None,
                          "inputs": _fingerprints(index, _names[id(module)], "in", args, kwargs),
                          "outputs": {}})
            open_calls.setdefault(id(module), []).append(index)
            running.append(index)

        def post_hook(module, args, kwargs, output):
            index = open_calls[id(module)].pop()
            running.remove(index)
            calls[index]["outputs"] = _fingerprints(index, _names[id(module)], "out", [output])

        try:
            for module in model.modules():
                handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
                handles.append(module.register_forward_hook(post_hook, with_kwargs=True))
            with torch.no_grad():
                if is_method_adapter:
                    model(batch, epoch=0)
                else:
                    model(batch)
        finally:
            for handle in handles:
                handle.remove()
        for name in sorted(vars(model)):
            value = vars(model)[name]
            if name.startswith("last_") and isinstance(value, torch.Tensor):
                entry["last"][name] = fingerprint(value, f"last:{name}")
        batches.append(entry)
    return {"environment": _environment(), "fold": fold, "batch_size": batch_size,
            "max_batches": max_batches, "batches": batches}


# ---- 3. comparison -----------------------------------------------------------------------------------------------------

def relative_difference(a: dict[str, Any], b: dict[str, Any]) -> float:
    """max abs difference over (sum, min, max, projection) / (abs-sum / numel + 1e-12); inf on shape or dtype mismatch."""
    if a["shape"] != b["shape"] or a["dtype"] != b["dtype"]:
        return math.inf
    numel = int(np.prod(a["shape"])) if a["shape"] else 1
    if numel == 0:
        return 0.0
    gaps = [abs(a[k] - b[k]) for k in ("sum", "min", "max", "projection")]
    return float(max(gaps) / (a["abs_sum"] / numel + 1e-12))


def _worst(left: dict[str, Any], right: dict[str, Any]):
    """Largest relative difference over the fingerprints both sides hold; (None, missing keys) if none."""
    keys = [k for k in left if k in right]
    missing = sorted(set(left) ^ set(right))
    values = [relative_difference(left[k], right[k]) for k in keys]
    return (max(values) if values else None), missing


def compare_dumps(dump_a: dict[str, Any], dump_b: dict[str, Any]) -> dict[str, Any]:
    if dump_a.get("fold") != dump_b.get("fold") or len(dump_a["batches"]) != len(dump_b["batches"]):
        raise ValueError("the two dumps are not of the same fold and number of batches")
    rows: list[dict[str, Any]] = []
    structure: list[str] = []
    for batch_a, batch_b in zip(dump_a["batches"], dump_b["batches"]):
        batch = batch_a["batch"]
        for name in sorted(set(batch_a["inputs"]) | set(batch_b["inputs"])):
            if name in batch_a["inputs"] and name in batch_b["inputs"]:
                rows.append({"batch": batch, "kind": "batch_input", "name": name,
                             "in_rel": None, "out_rel": relative_difference(batch_a["inputs"][name], batch_b["inputs"][name])})
            else:
                structure.append(f"batch {batch} input {name} only in one dump")
        if len(batch_a["calls"]) != len(batch_b["calls"]) or any(
                (x["module"], x["type"]) != (y["module"], y["type"])
                for x, y in zip(batch_a["calls"], batch_b["calls"])):
            raise ValueError(f"the module call sequence of batch {batch} differs between the dumps")
        for call_a, call_b in zip(batch_a["calls"], batch_b["calls"]):
            in_rel, in_missing = _worst(call_a["inputs"], call_b["inputs"])
            out_rel, out_missing = _worst(call_a["outputs"], call_b["outputs"])
            structure += [f"batch {batch} call {call_a['call']} {call_a['module']}: {k} only in one dump"
                          for k in in_missing + out_missing]
            rows.append({"batch": batch, "kind": "module", "name": call_a["module"], "type": call_a["type"],
                         "call": call_a["call"], "parent": call_a["parent"],
                         "in_rel": in_rel, "out_rel": out_rel})
        for name in sorted(set(batch_a["last"]) | set(batch_b["last"])):
            if name in batch_a["last"] and name in batch_b["last"]:
                rows.append({"batch": batch, "kind": "last", "name": name, "in_rel": None,
                             "out_rel": relative_difference(batch_a["last"][name], batch_b["last"][name])})
            else:
                structure.append(f"batch {batch} {name} only in one dump")
    positive = [r[key] for r in rows if r["kind"] != "batch_input" for key in ("in_rel", "out_rel")
                if r[key] and 0 < r[key] < math.inf]
    floor = max(min(positive), 1e-12) if positive else 1e-12
    previous = None
    for row in rows:
        out = row["out_rel"]
        row["jump_vs_previous"] = (None if out is None or previous is None
                                   else out / max(previous, floor))
        if out is not None:
            previous = out
        row["mark"] = ""
    marked: set[tuple[int, int]] = set()
    for row in rows:
        if row["kind"] != "module" or row["in_rel"] is None or row["out_rel"] is None:
            continue
        if row["in_rel"] <= AGREE_FACTOR * floor and row["out_rel"] > JUMP_FACTOR * max(row["in_rel"], floor):
            row["mark"] = "marked"
            marked.add((row["batch"], row["call"]))
    for row in rows:
        if row["mark"]:
            has_marked_child = any(r["kind"] == "module" and r["batch"] == row["batch"]
                                   and (r["batch"], r["call"]) in marked and r["parent"] == row["call"]
                                   for r in rows)
            row["mark"] = "enclosing" if has_marked_child else "START"
    return {"floor": floor, "agree_within": AGREE_FACTOR * floor, "jump_factor": JUMP_FACTOR,
            "environment_a": dump_a.get("environment"), "environment_b": dump_b.get("environment"),
            "structure_notes": structure, "rows": rows}


def format_table(result: dict[str, Any]) -> str:
    def cell(value):
        return "-" if value is None else ("inf" if value == math.inf else f"{value:.3g}")

    lines = [f"floor {result['floor']:.3g}; inputs agree within {result['agree_within']:.3g}; "
             f"outputs jump by more than {result['jump_factor']:g}x -> marked",
             f"{'batch':>5} {'kind':<11} {'module / tensor':<46} {'in_rel':>9} {'out_rel':>9} {'jump':>9}  mark"]
    for row in result["rows"]:
        lines.append(f"{row['batch']:>5} {row['kind']:<11} {row['name'][:46]:<46} {cell(row['in_rel']):>9} "
                     f"{cell(row['out_rel']):>9} {cell(row['jump_vs_previous']):>9}  {row['mark']}")
    lines += [f"note: {note}" for note in result["structure_notes"]]
    return "\n".join(lines)


# ---- CLI -----------------------------------------------------------------------------------------------------------------

def main(argv=None, *, loader=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--fold", choices=list(runner.FOLD_BLOCK), default="test")
    parser.add_argument("--out", type=Path, help="a new .json file (agreement report, or the compare table)")
    parser.add_argument("--dump", type=Path, help="optional new .json file for the activation fingerprints")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"))
    parser.add_argument("--max-batches", type=int, default=3)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args(argv)

    if args.compare:
        if args.run_dir or args.dump:
            parser.error("--compare takes no --run-dir or --dump")
        _refuse_target(args.out, None)
        result = compare_dumps(*(json.loads(path.read_text()) for path in args.compare))
        print(format_table(result))
        if args.out is not None:
            _write_new(args.out, result)
        return 0
    if args.run_dir is None or args.out is None:
        parser.error("--run-dir and --out are required unless --compare is given")
    if args.max_batches < 1:
        parser.error("--max-batches must be at least 1")
    if args.dump is not None and args.dump == args.out:
        parser.error("--dump and --out must be different files")
    _refuse_target(args.out, args.run_dir)
    _refuse_target(args.dump, args.run_dir)
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    model, is_method_adapter, splits, binding = (loader or default_loader)(args.run_dir, args.fold)
    report = agreement_report(model, splits, binding, args.run_dir, args.fold, is_method_adapter=is_method_adapter)
    _write_new(args.out, report)
    if args.dump is not None:
        dump = dump_fingerprints(model, splits, args.fold, binding.get("batch_size", 64), is_method_adapter,
                                 args.max_batches)
        dump["method"] = report["method"]
        _write_new(args.dump, dump)
    print(json.dumps({k: report[k] for k in ("method", "fold", "rows", "max_abs_diff", "argmax_agreement",
                                             "rows_over_1e-4", "exact_hash_match")}
                     | {"runner_rule_would_pass": report["runner_rule"]["would_pass"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
