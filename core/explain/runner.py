"""Explanation runner for clinical_graph_v2 runs.

Reloads a trained run (`binding.json` + `preprocessing.json` + `best.pt`, written by `core/train.py`),
rebuilds the exact model and the exact split of the chosen fold, and refuses to explain anything unless

- the targets file, the graph file and the visit-membership sidecar hash to the values the training run
  recorded, the kept classes are the recorded ones and the refitted preprocessing hashes to the recorded
  `preprocessing_sha256` (`rebuild_split`);
- the rebuilt model reports the recorded configuration (`build_model`);
- a replay of the fold reproduces the recorded prediction hash, or, when the byte hash cannot reproduce on
  this machine, the stored `<fold>.npz` is the training run's own (its probabilities hash to the recorded
  value and its labels are the fold's) and every replayed row is within REPLAY_TOLERANCE of it
  (`replay_check`).

Then, for every chosen visit of the fold, it runs GradExplainer, the vendored (zero-baseline) IG and
GNNExplainer (`core/explain/graphxai_standardized.explain_algorithms`), the clinical-baseline IG
(`core/explain/ig_clinical`, steps raised per visit until completeness holds or the largest candidate is
reached) and the model's own built-in explanation where it has one. Every explanation is scored with the
fidelity metrics of `fidelity_v2.py` (intervention contract version 6): evidence nodes are deleted from the
visit's graph RECORD, the record is re-encoded with the run's own preprocessing and the trained model scores
it with its own forward (`core/explain/node_deletion.DeletionScorer`, checked per visit against the split
tensor). Combo threshold states follow the combo rules of the threshold table the graph artifact was built
with (`artifact_combo_rules`, hash-checked against the artifact manifest). The report fold is
`test` (default); `validation` and `dev` are decision folds, labelled as such in the outputs (`fold_role`).

Cohort contract: `explanation_contract_v2.py` (COHORT_SCHEMA_VERSION = 2): the test, validation or dev fold
of ONE finished run, hash-pinned to that run's binding.json, written next to the records as cohort.json.

Method coverage: gsat, protgnn, graphcare (through `core/explain/graphxai_wrapper.py`), CEI-GNN v3 / mp2
(through `core/explain/cei_bridge.py`) and the black-box baselines (clinical_gnn with --conv
edge_conditioned | hgt, explained post hoc only). Any other registered method is refused with a clear
message; GCHM-PNA is not part of the explanation study.

Outputs: every run writes to a new, empty, caller-chosen directory: one record per explained visit (record
schema 3), cohort.json, `explanation_manifest.json` (settings, per-method disclosure, built-in explanation,
code hashes, replay rule, prototype projection) and `binding.json` (model run, cohort and settings, each
pinned by a hash). An occupied directory is refused, never overwritten. No absolute path is written.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import subprocess
import time
from argparse import Namespace
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from core import train
from core.contracts import VISIT_MEMBERSHIP_FILENAME, iter_graphs_with_membership, sample_ids_sha256
from core.explain import audit as audit_mod
from core.explain import fidelity_v2 as fv2
from core.explain import ig_clinical
from core.explain.explanation_contract_v2 import build_cohort_v2, validate_cohort_v2
from core.explain.graphxai_wrapper import (
    GRAPHCARE_VALUE_DEPENDENCE, GSAT_EDGE_ATTENTION_RULE, PROTGNN_EXPLANATION_STYLE,
    PROTGNN_OFFICIAL_SETTINGS, PROTGNN_SEARCH_ALGORITHM, PROTGNN_SUBGRAPH_SCORING,
    PROTGNN_SUBGRAPH_SIZE_NOTE, ClinicalGNNGraphXAIWrapper, ClinicalGraphXAIWrapper,
    GraphCareGraphXAIWrapper,
)
from core.explain.node_deletion import ComboRules, DeletionScorer, combo_source, load_combo_rules
from core.model import ClinicalGNN
from core.paths import CODE_ROOTS, REPO_ROOT
from core.registry import build_method
from core.schema import sha256
from core.tensorize import PAYLOAD_WIDTH, encode_graph, preprocessing_state

GRAPHXAI_METHODS = frozenset({"gsat", "protgnn", "graphcare", "cei_gnn_v3", "cei_gnn_mp2"})
CEI_METHODS = frozenset({"cei_gnn_v3", "cei_gnn_mp2"})
# The methods of the final explanation study; the manifest discloses each of them.
FINAL_METHODS = ("gsat", "protgnn", "graphcare", "cei_gnn_mp2")
RUNNER_SCHEMA = "medgnn.clinical_graph_v2_explanation_run"
RUNNER_SCHEMA_VERSION = 2
RECORD_SCHEMA = "medgnn.clinical_graph_v2_explanation_record"
RECORD_SCHEMA_VERSION = 3
BINDING_SCHEMA = "medgnn.clinical_graph_v2_explanation_binding"
BINDING_SCHEMA_VERSION = 2
# Black-box baselines: ClinicalGNN over genuine MessagePassing layers, explained
# post hoc with GNNExplainer through PyG's own mask hook (no model change).
BLACK_BOX_CONVS = ("edge_conditioned", "hgt")
# Kept in the records for comparison, but not scored with fidelity: it has no completeness check
# and the summary leaves it out of its tables.
FIDELITY_SKIPPED_SOURCES = ("IntegratedGradExplainerZeroBaseline",)
# The clinical IG start step count is chosen on this many decision-fold patients (one visit each).
IG_PILOT_PATIENTS = 30
VENDORED_GRAPHXAI = REPO_ROOT / "external/GraphXAI-main/graphxai"
GRAPHXAI_UPSTREAM_COMMIT = "a11e65f"
GNNEXPLAINER_SEED_RULE = ("per visit: the first 8 hex digits of sha256('<random_seed>:<sample_id>') as an "
                          "integer; the same seed drives the tie draws and the random reference")
IG_ESCALATION = ("per visit: from the start step count up the candidates until the relative completeness "
                 "error is within the tolerance, else the largest candidate (completeness_met false)")
PROJECTION_NOTE = ("projection runs at the start of an epoch, before that epoch's training steps; the selected "
                   "checkpoint's prototypes were trained for epochs_trained_after_last_projection epochs after "
                   "the last projection, so they are near, not equal to, training-subgraph embeddings")


def proba_digest(proba) -> str:
    """Identical hashing scheme to train.py's own (private) proba_digest, so a
    replayed prediction array hashes to the exact value binding.json recorded."""
    return hashlib.sha256(np.ascontiguousarray(proba).tobytes()).hexdigest()


def _canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


@dataclass(frozen=True)
class RunBundle:
    run_dir: Path
    binding: dict[str, Any]
    state_dict: dict[str, torch.Tensor]
    preprocessing: dict[str, Any] | None = None


FOLD_BLOCK = {"test": "selected_test", "validation": "selected_validation",
              "dev": "selected_dev"}
# test is the report fold; validation (and dev) only ever label a decision.
FOLD_ROLE = {"test": "report", "validation": "decision", "dev": "decision"}
# Fallback replay rule when the byte-exact hash cannot reproduce (another machine or
# thread count): every row within this of the stored <fold>.npz and 100% argmax agreement.
REPLAY_TOLERANCE = 1e-4
GCHM_REFUSAL = "GCHM-PNA is not part of the explanation study"


def _refuse_gchm(binding: dict[str, Any]) -> None:
    if binding.get("method") == "clinical_gnn" and str(binding.get("conv") or "").startswith("gchm"):
        raise ValueError(GCHM_REFUSAL)


def load_run_bundle(run_dir, fold: str = "test") -> RunBundle:
    """Load and hash-verify binding.json/preprocessing.json/best.pt. Refuses
    to proceed on any mismatch, before anything is explained."""
    run_dir = Path(run_dir)
    binding_path, prep_path, ckpt_path = (
        run_dir / "binding.json", run_dir / "preprocessing.json", run_dir / "best.pt"
    )
    for path, label in ((binding_path, "binding.json"), (prep_path, "preprocessing.json"),
                        (ckpt_path, "best.pt")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} missing from run directory: {run_dir}")
    binding = json.loads(binding_path.read_text())
    _refuse_gchm(binding)
    if sha256(prep_path) != binding.get("preprocessing_sha256"):
        raise ValueError(
            "preprocessing.json hash does not match binding.json's recorded "
            "preprocessing_sha256 -- this run directory is not internally "
            "consistent; refusing to explain a possibly-mismatched model/data pair."
        )
    if fold not in FOLD_BLOCK:
        raise ValueError(f"fold must be one of {sorted(FOLD_BLOCK)}, not {fold!r}")
    if binding.get(FOLD_BLOCK[fold]) is None:
        raise ValueError(
            f"binding.json has no {FOLD_BLOCK[fold]} block (this run never evaluated the "
            f"{fold} fold -- e.g. a --selection-fold dev run with --final-eval none keeps "
            "validation closed; try fold='dev'); nothing to replay or explain against."
        )
    state_dict = torch.load(ckpt_path, map_location="cpu")
    return RunBundle(run_dir=run_dir, binding=binding, state_dict=state_dict,
                     preprocessing=json.loads(prep_path.read_text()))


def encode_record(graph, membership, *, prep, edge_mode, drop_relations, rewire_seed, edge_direction):
    """`core.tensorize.encode_graph` with the run's fitted preprocessing and edge view, exactly as
    `train.build_dataset` encodes a visit (no rewiring)."""
    return encode_graph(graph, prep, membership, edge_mode, drop_relations, (), rewire_seed,
                        edge_direction=edge_direction)


def record_encoder(binding: dict[str, Any], prep):
    """The run's `encode_graph` partial: graph record + membership row -> encoded visit."""
    return functools.partial(
        encode_record, prep=prep, edge_mode=binding["edges"],
        drop_relations=tuple(binding.get("dropped_relations") or ()),
        rewire_seed=binding["seed"], edge_direction=binding["edge_direction"])


def artifact_combo_rules(binding: dict[str, Any]) -> ComboRules | None:
    """The combo rules of the threshold table the graph artifact was built with: the file the artifact
    manifest names in its `thresholds` block (`thresholds.csv` next to the graphs), refused when its
    sha256 differs from the one the manifest recorded (the file's own hash is kept when none is
    recorded), evaluated with the manifest's `level_source` (the builder's default `csv` when the
    manifest names none; a binding that recorded another level source is refused). None when the
    artifact has no threshold layer: its graphs then carry no threshold state, and the deletion
    records its fallback combo rule."""
    artifact = Path(binding["artifact"])
    manifest = json.loads((artifact / "manifest.json").read_text())
    block = manifest.get("thresholds")
    if block is None:
        return None
    level_source = manifest.get("level_source", "csv")
    recorded = binding.get("level_source")
    if recorded is not None and recorded != level_source:
        raise ValueError(f"binding.json records level_source {recorded!r}, the artifact manifest "
                         f"{level_source!r}; refusing to re-evaluate threshold states")
    return load_combo_rules(artifact / block.get("file", "thresholds.csv"), block.get("sha256"),
                            level_source=level_source)


def recorded_combo_source(combo_rules: ComboRules | None) -> dict[str, Any]:
    """Which combo rule the deletions of a run used (`node_deletion.combo_source`), with the table
    reduced to its file name."""
    return _without_local_paths(combo_source(combo_rules))


class VisitRecords:
    """The graph records (graphs.jsonl line, visit-membership sidecar row) of the explained visits and the
    run's record encoder: what each visit's `DeletionScorer` deletes evidence from, with the combo rules
    of the artifact's threshold table (None: the deletion's fallback combo rule)."""

    def __init__(self, records: dict[str, tuple[dict, dict]], encode,
                 combo_rules: ComboRules | None = None):
        self.records = dict(records)
        self.encode = encode
        self.combo_rules = combo_rules

    @classmethod
    def read(cls, artifact, sample_ids, encode, combo_rules: ComboRules | None = None) -> "VisitRecords":
        artifact = Path(artifact)
        wanted = {str(sample_id) for sample_id in sample_ids}
        found = {}
        for graph, membership in iter_graphs_with_membership(
                artifact / "graphs.jsonl", artifact / VISIT_MEMBERSHIP_FILENAME, wanted_ids=wanted):
            found[str(graph["sample_id"])] = (graph, membership)
        missing = wanted - set(found)
        if missing:
            raise ValueError(f"{len(missing)} explained visits have no graph record in the artifact")
        return cls(found, encode, combo_rules)

    def scorer(self, forward, data) -> DeletionScorer:
        """The visit's deletion scorer; refuses when the record's encoding is not the split tensor."""
        sample_id = str(data.sample_id)
        if sample_id not in self.records:
            raise ValueError("no graph record was read for this visit")
        graph, membership = self.records[sample_id]
        scorer = DeletionScorer(forward, graph, membership, self.encode,
                                y=int(torch.as_tensor(data.y).view(-1)[0]),
                                combo_rules=self.combo_rules)
        scorer.check_against(data)
        return scorer


class RebuiltSplits(dict):
    """{fold: [encoded visits]} exactly as `train.build_dataset` returned them, plus the fitted
    preprocessing (`preprocessing`), the run's record encoder (`encode`) and the combo rules of the
    artifact's threshold table (`combo_rules`, `artifact_combo_rules`)."""

    def __init__(self, splits, *, preprocessing, binding: dict[str, Any]):
        super().__init__(splits)
        self.preprocessing = preprocessing
        self.encode = record_encoder(binding, preprocessing)
        self.combo_rules = artifact_combo_rules(binding)
        self.combo_source = recorded_combo_source(self.combo_rules)
        self._artifact = Path(binding["artifact"])
        self._rewired = tuple(binding.get("rewired_relations") or ())

    def visit_records(self, sample_ids) -> VisitRecords:
        if self._rewired:
            raise ValueError("node deletion is not defined for a run trained on rewired edges "
                             f"({list(self._rewired)}): re-encoding a reduced record would rewire it anew")
        return VisitRecords.read(self._artifact, sample_ids, self.encode, self.combo_rules)


def _recorded(binding: dict[str, Any], key: str):
    if binding.get(key) is None:
        raise ValueError(f"binding.json records no {key}; the rebuilt split cannot be verified")
    return binding[key]


def rebuild_split(binding: dict[str, Any], fold: str = "test") -> RebuiltSplits:
    """Reconstruct the train/validation(/dev)(/test) splits exactly as train.py built them, from
    binding.json's own recorded (already-resolved) arguments, and verify every input against the
    training run: the targets file, the graph file, the visit-membership sidecar and the artifact
    manifest by their recorded sha256, the kept classes, the refitted preprocessing by
    the recorded `preprocessing_sha256` (the same bytes train.py wrote) and the fold's sample ids. The
    test rows are decoded only for `fold == "test"`; nothing is fitted or selected on them. Returns the
    splits with the fitted preprocessing and the record encoder."""
    targets_path = Path(binding["targets_path"])
    if sha256(targets_path) != _recorded(binding, "targets_sha256"):
        raise ValueError(
            "the targets file's sha256 differs from binding.json's targets_sha256 -- the labels "
            "have changed since training; refusing to explain against different labels")
    artifact = Path(binding["artifact"])
    for filename, key in (("graphs.jsonl", "artifact_graphs_sha256"),
                          (VISIT_MEMBERSHIP_FILENAME, "artifact_visit_membership_sha256")):
        if sha256(artifact / filename) != _recorded(binding, key):
            raise ValueError(
                f"the artifact's {filename} sha256 differs from binding.json's {key} -- the graph "
                "inputs have changed since training; refusing to explain different graphs")
    # the combo threshold table is read through this manifest, so its hash is required like the others
    if sha256(artifact / "manifest.json") != _recorded(binding, "graph_manifest_sha256"):
        raise ValueError("the artifact's manifest.json sha256 differs from binding.json's "
                         "graph_manifest_sha256; refusing to explain against a different artifact")
    targets = train.load_targets(targets_path)
    if binding.get("top_k_labels") is not None:
        targets, kept, _ = train.select_top_labels(targets, binding["top_k_labels"])
        recorded = _recorded(binding, "kept_label_indices")
        if [int(c) for c in kept] != [int(c) for c in recorded]:
            raise ValueError("the kept classes differ from binding.json's kept_label_indices; "
                             "refusing to explain against a different class order")
    splits, prep = train.build_dataset(
        binding["artifact"], targets, binding["edges"], binding["train_limit"],
        binding["token_min_count"], binding["seed"],
        tuple(binding.get("dropped_relations") or ()),
        tuple(binding.get("rewired_relations") or ()),
        binding.get("min_prior_visits", 0),
        edge_direction=binding["edge_direction"],
        dev_limit=binding.get("dev_limit"),
        sample_seed=binding.get("sample_seed"),
        load_test=(fold == "test"),
    )
    state = (json.dumps(preprocessing_state(prep), indent=2, sort_keys=True) + "\n").encode()
    if hashlib.sha256(state).hexdigest() != _recorded(binding, "preprocessing_sha256"):
        raise ValueError(
            "the refitted preprocessing differs from binding.json's preprocessing_sha256 -- the "
            "split would be encoded differently from training; refusing to explain it")
    if fold not in splits:
        raise ValueError(f"rebuilt splits have no {fold} fold")
    # The explained fold's hash must be recorded; every other rebuilt fold (the IG pilot comes from a
    # decision fold, the real-patient IG bank from train) is checked wherever training recorded it.
    recorded_hashes = binding.get("split_sample_ids_sha256") or {}
    for name in [fold] + sorted(set(splits) - {fold}):
        if name != fold and name not in recorded_hashes:
            continue
        actual = sample_ids_sha256(d.sample_id for d in splits[name])
        if recorded_hashes.get(name) != actual:
            raise ValueError(
                f"rebuilt {name} split's sample-ID hash does not match "
                f"binding.json's split_sample_ids_sha256[{name!r}] -- the "
                "artifact, targets file, or build arguments have drifted since "
                "training; refusing to explain against a different split than "
                "the model was actually trained and evaluated on."
            )
    return RebuiltSplits(splits, preprocessing=prep, binding=binding)


def rebuild_validation_split(binding: dict[str, Any], fold: str = "validation"):
    """Earlier name of `rebuild_split`, kept as an alias with its earlier default fold."""
    return rebuild_split(binding, fold)


def _infer_token_dim(state_dict: dict[str, torch.Tensor], *, is_method_adapter: bool) -> int:
    key = "token_embedding.weight" if is_method_adapter else "token.weight"
    if key not in state_dict:
        raise ValueError(f"checkpoint has no {key!r} tensor to infer token_dim from")
    return int(state_dict[key].shape[1])


# The common training profile train.py hands every adapter through its CLI namespace.
_SCHEDULE_KEYS = ("epochs", "lr", "weight_decay", "batch_size", "patience", "min_delta")


def _config_differences(recorded, rebuilt, prefix=""):
    """Dotted paths at which two run_config dicts differ."""
    if isinstance(recorded, dict) and isinstance(rebuilt, dict):
        paths = []
        for key in sorted(set(recorded) | set(rebuilt), key=str):
            paths += _config_differences(recorded.get(key), rebuilt.get(key), f"{prefix}{key}.")
        return paths
    return [] if recorded == rebuilt else [prefix.rstrip(".")]


def build_model(binding: dict[str, Any], state_dict: dict[str, torch.Tensor]):
    """Reconstruct the exact architecture and configuration binding.json/best.pt describe.

    A non-CEI adapter is built the way train.py builds it: the recorded dropout, the recorded common
    profile (epochs, lr, weight decay, batch size, patience, min delta), ProtGNN's
    select_after_projection (recorded as `selection_start_epoch_index`) and every recorded native
    setting. Its whole run_config() must then equal binding.json's method_config. GraphCare reports its
    recency decay as `recency_decay` but reads it as `graphcare_decay_rate` / `decay_rate`; the alias
    keeps the rebuilt model from falling back to the native default. CEI is rebuilt and checked by
    `cei_bridge.build_cei_model`.
    """
    method = binding["method"]
    conv = binding.get("conv")
    _refuse_gchm(binding)
    node_dim, edge_dim = binding["node_dim"], binding["edge_dim"]
    num_classes = binding["num_classes"]
    num_tokens = binding["vocabulary_size"]
    num_triples = binding["num_meta_relations"]
    hidden, layers = binding["hidden"], binding["layers"]

    if method == "clinical_gnn" and conv in BLACK_BOX_CONVS:
        model = ClinicalGNN(
            num_tokens=num_tokens, node_dim=node_dim, edge_dim=edge_dim,
            num_classes=num_classes, hidden=hidden, layers=layers,
            dropout=float(binding.get("dropout") or 0.0),
            token_dim=_infer_token_dim(state_dict, is_method_adapter=False),
            use_edge_payload=binding.get("edge_payload", True),
            use_message_passing=binding.get("message_passing", True),
            conv=conv, num_triples=num_triples, payload_dim=PAYLOAD_WIDTH,
            heads=binding.get("heads") or 4,
            modulation=binding.get("modulation") or "multiplicative",
        )
        model.load_state_dict(state_dict)
        return model, False

    if method not in GRAPHXAI_METHODS:
        raise NotImplementedError(
            f"method {method!r} is not supported by this runner (supported: "
            f"{sorted(GRAPHXAI_METHODS)} and clinical_gnn {list(BLACK_BOX_CONVS)}); its adapter "
            "does not accept `external_edge_mask`, which GNNExplainer requires")
    if method in CEI_METHODS:
        from core.explain.cei_bridge import build_cei_model

        return build_cei_model(binding, state_dict), True
    recorded_config = binding["method_config"]
    architecture = recorded_config.get("architecture", {})
    token_dim = architecture.get("token_dim")
    if token_dim is None:
        token_dim = _infer_token_dim(state_dict, is_method_adapter=True)
    effective_settings = dict(recorded_config.get("effective_settings", {}))
    if method == "graphcare" and "recency_decay" in effective_settings:
        effective_settings.setdefault("decay_rate", effective_settings["recency_decay"])
    namespace = {name: binding[name] for name in _SCHEDULE_KEYS if binding.get(name) is not None}
    namespace["select_after_projection"] = binding.get("selection_start_epoch_index") is not None
    namespace.update(effective_settings)
    model = build_method(
        method, num_tokens=num_tokens, node_dim=node_dim, edge_dim=edge_dim,
        num_classes=num_classes, hidden=hidden, layers=layers,
        dropout=float(binding.get("dropout") or 0.0), token_dim=int(token_dim),
        num_triples=num_triples,
        args=Namespace(edge_direction=binding["edge_direction"], **namespace),
    )
    # Fail closed: the rebuilt model must report the recorded configuration in full, so a setting
    # that is recorded under one name and read under another can never fall back to a default.
    rebuilt_config = json.loads(json.dumps(model.run_config(), default=str))
    differing = _config_differences(recorded_config, rebuilt_config)
    if differing:
        raise ValueError(f"rebuilt {method} model's run_config differs from binding.json's method_config "
                         f"at {differing}")
    model.load_state_dict(state_dict)
    return model, True


def rebuilt_method_config(model, binding: dict[str, Any]) -> dict[str, Any] | None:
    """The rebuilt model's own run_config(); the recorded one for the black-box ClinicalGNN, which has
    none (it is rebuilt from the recorded binding fields)."""
    run_config = getattr(model, "run_config", None)
    if callable(run_config):
        return json.loads(json.dumps(run_config(), default=str))
    return binding.get("method_config")


@torch.no_grad()
def fold_probabilities(model, splits, fold: str, batch_size: int, is_method_adapter: bool,
                       device="cpu") -> np.ndarray:
    """Softmax probabilities of every row of `fold`, batched the way training batched it
    (`batch_size`), in fold order. The one forward loop shared by `replay_check` and the
    replay difference tool."""
    model = model.to(device).eval()
    loader = DataLoader(splits[fold], batch_size=batch_size)
    logits = []
    for batch in loader:
        batch = batch.to(device)
        if is_method_adapter:
            logits.append(model(batch, epoch=0).logits.cpu())
        else:
            logits.append(model(batch).cpu())
    return torch.softmax(torch.cat(logits), dim=1).numpy()


def environment_info() -> dict[str, Any]:
    """The software and hardware the process runs on; the thread count is read now, i.e.
    the count the forward passes really used."""
    import platform

    return {"torch": torch.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "machine": platform.machine(),
            "torch_num_threads": int(torch.get_num_threads())}


def _fold_labels(rows) -> np.ndarray:
    return np.asarray([int(torch.as_tensor(d.y).view(-1)[0]) for d in rows], dtype=np.int64)


@torch.no_grad()
def replay_check(model, splits, binding: dict[str, Any], *, is_method_adapter: bool,
                 device="cpu", batch_size: int | None = None, fold: str = "test",
                 stored_npz: str | Path | None = None) -> dict[str, Any]:
    """Refuse to proceed unless a fresh pass over `fold` reproduces the exact recorded prediction
    hash. When that byte hash cannot reproduce on this machine, the stored `<fold>.npz` is used, but
    only after it is shown to be the training run's own: its probabilities must hash to the recorded
    `prediction_sha256` (train.py writes both from the same array) and its labels must be the rebuilt
    fold's. Then every row must be within REPLAY_TOLERANCE of it with the same argmax on every row
    except ties: a flip is accepted only on a row whose stored top-two probability gap is below
    twice the tolerance, since each of the two probabilities may move by up to the tolerance. The
    returned dict names the rule that passed and counts the flips. If anything upstream was wrong,
    this fails loudly, closed, before a single explanation is generated.

    batch_size defaults to binding["batch_size"], the training run's own value. The final models
    score each graph on its own (GSAT's InstanceNorm normalises per graph, its BatchNorm uses running
    statistics in eval); batch grouping changes only the float summation order. The byte hash depends
    on that order, so it can only reproduce with training's own batching.
    """
    if batch_size is None:
        batch_size = binding.get("batch_size", 64)
    proba = fold_probabilities(model, splits, fold, batch_size, is_method_adapter, device)
    digest = proba_digest(proba)
    expected = binding[FOLD_BLOCK[fold]]["prediction_sha256"]
    if digest == expected:
        return {"mode": "exact_hash", "prediction_sha256": digest, "stored_npz_hash_verified": None}
    # A byte-exact hash only reproduces on the training machine with the same thread
    # count and library build. On another machine (or another thread count) the float
    # sums differ in the last bits, so the fallback compares with the stored
    # predictions of the same fold, row by row (same rule as source masking).
    if stored_npz is not None and Path(stored_npz).exists():
        stored = np.load(stored_npz, allow_pickle=False)
        ids = [str(d.sample_id) for d in splits[fold]]
        if "proba" not in stored.files or proba_digest(stored["proba"]) != expected:
            detail = (f"the stored {fold}.npz is not the training run's: its probabilities do not hash "
                      f"to binding.json's {FOLD_BLOCK[fold]}.prediction_sha256")
        elif "y" not in stored.files or not np.array_equal(
                np.asarray(stored["y"]).astype(np.int64).reshape(-1), _fold_labels(splits[fold])):
            detail = f"the stored {fold}.npz labels differ from the rebuilt {fold} fold's labels"
        elif [str(s) for s in stored["sample_ids"]] != ids or stored["proba"].shape != proba.shape:
            detail = "stored sample ids or shape differ from the rebuilt fold"
        else:
            stored_proba = stored["proba"].astype(np.float64)
            diff = float(np.abs(stored_proba - proba).max())
            flipped = stored_proba.argmax(1) != proba.argmax(1)
            top_two = np.sort(stored_proba, axis=1)[:, -2:]
            tie = (top_two[:, 1] - top_two[:, 0]) < 2 * REPLAY_TOLERANCE
            flips_total, flips_tie = int(flipped.sum()), int((flipped & tie).sum())
            if diff <= REPLAY_TOLERANCE and flips_total == flips_tie:
                return {"mode": "stored_predictions_tolerance", "max_abs_diff": diff,
                        "tolerance": REPLAY_TOLERANCE,
                        "argmax_agreement": 1.0 - flips_total / max(1, len(ids)),
                        "argmax_flips_total": flips_total, "argmax_flips_tie": flips_tie,
                        "rows": len(ids), "prediction_sha256": digest,
                        "stored_npz_hash_verified": True, "stored_labels_verified": True}
            detail = (f"max abs diff {diff:.3g} (tolerance {REPLAY_TOLERANCE}), argmax flips "
                      f"{flips_total} of {len(ids)} rows, {flips_tie} of them on ties "
                      f"(stored top-two gap below {2 * REPLAY_TOLERANCE})")
    else:
        detail = f"no stored {fold}.npz to compare with"
    raise ValueError(
        f"REPLAY CHECK FAILED: reloading best.pt and re-running {fold} "
        f"produced prediction hash {digest}, not the recorded {expected}; {detail}. "
        "Refusing to generate explanations from a model that cannot be "
        "shown to be the one that was actually trained and evaluated."
    )


def build_wrapper(binding: dict[str, Any], model):
    """The wrapper of the explained method (core/explain/graphxai_wrapper.py, cei_bridge.py),
    dispatched by method/conv."""
    method = binding["method"]
    conv = binding.get("conv")
    kwargs = dict(
        node_dim=binding["node_dim"], edge_dim=binding["edge_dim"],
        num_tokens=binding["vocabulary_size"], num_triples=binding["num_meta_relations"],
        num_relations=binding["num_relations"],
    )
    if method == "graphcare":
        return GraphCareGraphXAIWrapper(model, **kwargs)
    if method in CEI_METHODS:
        from core.explain.cei_bridge import CEIGraphXAIWrapper

        return CEIGraphXAIWrapper(model, method=method)
    if method == "clinical_gnn" and conv in BLACK_BOX_CONVS:
        return ClinicalGNNGraphXAIWrapper(model, method=conv, **kwargs)
    if method == "clinical_gnn":
        return ClinicalGraphXAIWrapper(model, method=conv, is_method_adapter=False, **kwargs)
    return ClinicalGraphXAIWrapper(model, method=method, is_method_adapter=True, **kwargs)


def _single_graph_batch(data):
    """One PyG Data -> the (x, edge_index, batch_index) a wrapper's
    set_context/forward expect for a single-graph explanation."""
    from torch_geometric.data import Batch
    return Batch.from_data_list([data])


def _stable_seed(base: int, sample_id: str) -> int:
    """Deterministic per-subject seed, independent of Python's salted hash()."""
    digest = hashlib.sha256(f"{int(base)}:{sample_id}".encode()).hexdigest()
    return int(digest[:8], 16)


def select_explained_subjects(fold_rows, *, max_subjects: int | None = None,
                              sample_patients: int | None = None, sample_seed: int = 1234):
    """The visits to explain. `sample_patients=N` draws N PATIENTS (hash-ordered by
    seed, so the same patients are drawn for every method) and keeps every one of
    their visits of the fold in training order; the headline metrics are
    patient-equal, so the sampling unit is the patient. `max_subjects` then caps the
    list (a prefix), mostly for smoke runs."""
    subjects = list(fold_rows)
    if sample_patients is not None:
        patients = sorted({str(d.subject) for d in subjects},
                          key=lambda s: hashlib.sha256(f"{int(sample_seed)}:{s}".encode()).hexdigest())
        chosen = set(patients[:sample_patients])
        subjects = [d for d in subjects if str(d.subject) in chosen]
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
    return subjects


def _ig_path(wrapper):
    return wrapper.ig_path() if hasattr(wrapper, "ig_path") else nullcontext()


def _without_acc(metrics: dict[str, Any]) -> dict[str, Any]:
    """A true-class metrics entry: the probability change only (`acc` is a predicted-class notion)."""
    if not isinstance(metrics, dict) or "fidelity_plus" not in metrics:
        return metrics
    out = dict(metrics)
    for key in ("fidelity_plus", "fidelity_minus"):
        out[key] = {k: v for k, v in metrics[key].items() if k != "acc"}
    return out


def _source_metrics(scorer, importance, target_class, *, curve_points: int, tie_draws: int,
                    seed: int, reference=None) -> dict[str, Any]:
    """The fidelity metrics of one importance vector, all through the visit's deletion scorer:
    fidelity+/- at the shared top-20% of eligible nodes (capped at the explanation's own support),
    deletion/insertion curves, sparsity and tie summary over the eligible nodes and -- when a random
    reference was computed for this visit -- how far each number sits from random deletion. A vector
    with no nonzero importance on an eligible node is `no_eligible_evidence`."""
    importance = np.asarray(importance, dtype=float)
    node_type = scorer.node_type
    if fv2.eligible_support(node_type, importance) == 0:
        return {"status": "no_eligible_evidence", "support": 0}
    common = dict(tie_draws=tie_draws, seed=seed)
    plus = fv2.fidelity_plus_v2(scorer, importance, target_class, **common)
    minus = fv2.fidelity_minus_v2(scorer, importance, target_class, **common)
    deletion = fv2.deletion_curve(scorer, importance, target_class, num_points=curve_points, **common)
    insertion = fv2.insertion_curve(scorer, importance, target_class, num_points=curve_points, **common)
    result = {
        "fidelity_plus": plus, "fidelity_minus": minus,
        "deletion_curve": deletion, "insertion_curve": insertion,
        "sparsity": fv2.sparsity_over_eligible(node_type, importance),
        "tie_summary": fv2.tie_summary(node_type, importance),
    }
    if reference is not None:
        result["vs_random"] = reference.compare(
            fidelity_plus=plus, fidelity_minus=minus,
            deletion_curve=deletion, insertion_curve=insertion)
    return result


def _normalise_statuses(graphxai: dict[str, Any]) -> None:
    """Map the 'mask does not reach the output' verdicts onto 'unsupported'."""
    gnn = graphxai.get("GNNExplainer", {})
    if gnn.get("status") == "failed" and str(gnn.get("error", "")).startswith(
            "GNNExplainer edge mask gradient disconnected"):
        # d logit / d edge_mask is None: the mask never reaches the output at all
        # (e.g. a model with message passing switched off).
        gnn["status"] = "unsupported"
        gnn["unsupported_reason"] = (
            "d logit / d edge_mask is None: the mask does not reach this model's "
            "output, so GNNExplainer cannot explain it")
    for payload in graphxai.values():
        if payload.get("status") == "success" and payload.get("provenance", {}).get(
                "zero_predictive_edge_gradient"):
            payload["status"] = "unsupported"
            payload["unsupported_reason"] = (
                "d logit / d edge_mask is exactly zero (or the graph has no edges): "
                "the mask does not influence this prediction, so this is not an "
                "explanation of it and is not reported as a result")
            payload.pop("node_explanation", None)


def _clinical_ig_payloads(wrapper, x, batch, batch_index, target_class: int, ig: dict[str, Any]):
    """IG from the pre-registered clinical baseline (and, if requested, the real-patient
    reference variant), inside the wrapper's IG path, with the steps raised per visit until
    completeness holds; each payload records its completeness and the PLE term it ran on. The step
    count IG ran with is `steps_used` in both payloads; the run's start count is `start_steps`."""
    from core.explain.explanation_contract import build_node_explanation

    edge_index = batch.edge_index
    baseline = ig_clinical.not_recorded_baseline(x, batch.node_type, ig["layout"],
                                                 **ig.get("feature_spec", {}))
    with _ig_path(wrapper):
        ple_term = getattr(wrapper, "ple_term", None)
        attribution, completeness = ig_clinical.integrated_gradients_until_complete(
            wrapper, x, edge_index, baseline, target_class, start_steps=ig["steps"],
            candidates=ig["candidates"], tolerance=ig["tolerance"])
        refs = None
        if ig.get("bank") is not None and ig["reference_draws"] > 0:
            refs = ig_clinical.expected_gradients(
                wrapper, x, edge_index, batch.token, batch.node_type, ig["bank"], baseline,
                target_class, completeness["steps_used"], draws=ig["reference_draws"], seed=ig["seed"])
    importance = ig_clinical.node_importance(attribution).numpy()

    def payload(importance, provenance, standard):
        if not np.any(importance):
            return {"status": "unsupported", "provenance": provenance,
                    "unsupported_reason": "every attribution is exactly zero"}
        node_explanation = (build_node_explanation(wrapper, x, edge_index, batch=batch_index,
                                                   node_importance=importance)
                            if standard else {"node_importance": [float(v) for v in importance]})
        return {"status": "success", "provenance": provenance, "node_explanation": node_explanation}

    standard = ig["explain_predicted"]
    out = {"main": payload(importance, {
        "implementation": "core.explain.ig_clinical (clinical IG, not GraphXAI's IntegratedGradExplainer)",
        "baseline": "not_recorded",
        "feature_provenance": ig.get("feature_provenance") or {},
        **completeness,
        "start_steps": ig["steps"], "step_candidates": list(ig["candidates"]),
        "escalation": IG_ESCALATION, "step_selection": ig["step_selection"],
        "ple_term": ple_term, "node_reduction": ig_clinical.NODE_REDUCTION}, standard)}
    if refs is not None:
        averaged, summary = refs
        out["refs"] = payload(ig_clinical.node_importance(averaged).numpy(), {
            "implementation": "core.explain.ig_clinical", "baseline": "real_patient_refs",
            "steps_used": completeness["steps_used"], "ple_term": ple_term,
            "node_reduction": ig_clinical.NODE_REDUCTION, **summary}, standard)
    return out


def _collect_importances(wrapper, batch, x, *, method: str | None, steps: int, epochs: int, ig,
                         prediction: int, target_class: int | None = None,
                         seed: int | None = None) -> dict[str, Any]:
    """Every explanation source for one graph: the three GraphXAI explainers (the clinical IG
    added when `ig` is given, the vendored one kept as the zero-baseline reference) and the model's
    own built-in explanation, with its status: `available`, `unavailable` (with the model's reason)
    or `no_eligible_evidence` (no nonzero importance on an eligible evidence node)."""
    from core.explain.graphxai_standardized import explain_algorithms
    from core.tensorize import reversible_edge_pairs

    batch_index = torch.zeros(x.size(0), dtype=torch.long)
    inert = wrapper.inert_edge_mask() if hasattr(wrapper, "inert_edge_mask") else None
    graphxai = explain_algorithms(
        wrapper, x, batch.edge_index, batch=batch_index, steps=steps, epochs=epochs,
        reversible_edge_pairs=reversible_edge_pairs(batch.edge_relation),
        node_reduction="mean", target_class=target_class, seed=seed,
        inert_edge_mask=inert, method=method,
    )
    _normalise_statuses(graphxai)
    target = prediction if target_class is None else int(target_class)
    if ig is not None:
        ig_run = dict(ig, explain_predicted=target_class is None)
        payloads = _clinical_ig_payloads(wrapper, x, batch, batch_index, target, ig_run)
        vendored = graphxai.get("IntegratedGradExplainer")
        if ig.get("keep_zero_baseline") and vendored is not None:
            graphxai["IntegratedGradExplainerZeroBaseline"] = vendored
        graphxai["IntegratedGradExplainer"] = payloads["main"]
        if "refs" in payloads:
            graphxai["IntegratedGradExplainerRealPatientRefs"] = payloads["refs"]

    builtin, unavailable, detail, status, ties = None, None, None, "unavailable", None
    try:
        builtin = [float(v) for v in wrapper.builtin_node_importance(
            x, batch.edge_index, target_class=target_class)]
        support = fv2.eligible_support(batch.node_type, builtin)
    except (RuntimeError, ValueError) as exc:
        # The model itself produced no explanation for this graph (e.g. ProtGNN found no connected
        # subgraph, a CEI configuration without an additive ledger, a model without a built-in).
        # Recorded with its reason, never turned into a fake one.
        builtin, unavailable = None, str(exc) or type(exc).__name__
    else:
        status = "available" if support else "no_eligible_evidence"
        if support:
            ties = fv2.tie_summary(batch.node_type, builtin)
    if hasattr(wrapper, "builtin_detail"):
        try:
            detail = wrapper.builtin_detail(target_class)
        except (RuntimeError, ValueError):
            detail = None
    return {"graphxai": graphxai, "builtin": builtin, "builtin_status": status,
            "builtin_unavailable": unavailable, "builtin_detail": detail, "builtin_tie_summary": ties}


def _source_vectors(collected: dict[str, Any], *, skip=()) -> dict[str, np.ndarray]:
    vectors = {name: np.asarray(p["node_explanation"]["node_importance"], dtype=float)
               for name, p in collected["graphxai"].items()
               if name not in skip and p.get("status") == "success" and "node_explanation" in p}
    if collected["builtin"] is not None:
        vectors["builtin"] = np.asarray(collected["builtin"], dtype=float)
    return vectors


def _evaluate_sources(scorer, vectors: dict[str, np.ndarray], target: int, *, curve_points: int,
                      random_repeats: int, tie_draws: int, seed: int):
    """Fidelity metrics for every source, all calibrated against ONE random-deletion reference per
    visit through the same deletion scorer (so any out-of-distribution effect of deleting evidence is
    present on both sides of the comparison). One source's failure, or the reference's, never loses
    the others."""
    node_type = scorer.node_type
    removable_count = int(fv2.removable_node_mask(node_type).sum())
    ks: set[int] = set()
    for vector in vectors.values():
        try:
            support = fv2.eligible_support(node_type, vector)
        except ValueError:
            continue
        if support:
            ks.add(min(fv2.resolve_k(removable_count), support))
            ks.update(fv2.curve_ks(removable_count, support, curve_points))
    reference = None
    if random_repeats <= 0:
        reference_info = {"status": "not_requested", "repeats": 0, "seed": seed}
    elif not ks:
        reference_info = {"status": "no_eligible_evidence", "repeats": random_repeats, "seed": seed}
    else:
        try:
            reference = fv2.random_reference(scorer, target, sorted(ks), repeats=random_repeats, seed=seed)
            reference_info = {"status": "computed", "repeats": random_repeats, "seed": seed,
                              "ks": reference.ks}
        except Exception as exc:  # noqa: BLE001 - calibration must not block the metrics
            reference_info = {"status": "failed", "repeats": random_repeats, "seed": seed,
                              "error": str(exc)}
    results: dict[str, Any] = {}
    for name, vector in vectors.items():
        try:
            results[name] = _source_metrics(scorer, vector, target, curve_points=curve_points,
                                            tie_draws=tie_draws, seed=seed, reference=reference)
        except Exception as exc:  # noqa: BLE001 - one source's failure must not lose the rest
            results[name] = {"status": "failed", "error": str(exc)}
    return results, reference_info


def _audit_subject(wrapper, batch, x, base_vectors, prediction, *, method, steps, epochs, ig,
                   audit: dict[str, Any]) -> dict[str, Any]:
    """Weight-randomisation and stability controls for one graph."""
    out: dict[str, Any] = {}
    seed = audit["seed"]
    if audit.get("weight_randomisation", True):
        try:
            random_wrapper = audit["make_wrapper"](audit_mod.randomise_weights(audit["model"], seed))
            random_x = random_wrapper.set_context(batch)
            with torch.no_grad():
                random_prediction = int(random_wrapper(random_x, batch.edge_index).argmax(-1))
            collected = _collect_importances(random_wrapper, batch, random_x, method=method, steps=steps,
                                             epochs=epochs, ig=ig, prediction=random_prediction,
                                             seed=seed)
            out["weight_randomisation"] = {
                "prediction_changed": random_prediction != prediction,
                "sources": audit_mod.compare_sources(base_vectors, _source_vectors(collected),
                                                     batch.node_type),
            }
        except Exception as exc:  # noqa: BLE001 - a control must not sink the record
            out["weight_randomisation"] = {"status": "failed", "error": str(exc)}
    draws = []
    try:
        for draw in range(audit.get("stability_draws", 0)):
            x_perturbed, measured = audit_mod.perturb_values(
                batch.x.float(), audit["sigma"], seed + 1 + draw)
            perturbed = batch.clone()
            perturbed.x = x_perturbed
            perturbed_x = wrapper.set_context(perturbed)
            with torch.no_grad():
                perturbed_prediction = int(wrapper(perturbed_x, batch.edge_index).argmax(-1))
            if perturbed_prediction != prediction:
                draws.append({"prediction_changed": True, "measured_nodes_perturbed": measured})
                continue
            collected = _collect_importances(wrapper, perturbed, perturbed_x, method=method, steps=steps,
                                             epochs=epochs, ig=ig, prediction=prediction,
                                             seed=seed)
            draws.append({"prediction_changed": False, "measured_nodes_perturbed": measured,
                          "sources": audit_mod.compare_sources(
                              base_vectors, _source_vectors(collected), batch.node_type)})
    except Exception as exc:  # noqa: BLE001
        draws.append({"status": "failed", "error": str(exc)})
    finally:
        wrapper.set_context(batch)   # leave the wrapper on the original graph
    out["stability"] = {"sigma": audit.get("sigma"), "draws": draws}
    return out


def _single_graph_check(single_logits, stored_row) -> dict[str, Any] | None:
    """Informational only, never a refusal: how far the single-graph forward of an explained
    visit is from its stored `<fold>.npz` row (a single graph can differ from a batched row in the
    last float bits)."""
    if stored_row is None:
        return None
    try:
        stored = np.asarray(stored_row, dtype=np.float64).reshape(-1)
        proba = torch.softmax(single_logits.detach().reshape(1, -1).double(), dim=1)[0].numpy()
        if stored.shape != proba.shape:
            return None
        return {"max_abs_diff": float(np.abs(stored - proba).max()),
                "argmax_agrees": bool(stored.argmax() == proba.argmax())}
    except Exception:  # noqa: BLE001 - informational, must not block the record
        return None


COMBO_RULE_TEXT = ("the medical:indicates edges of one (state node, rule_id) of a combo rule form a group; "
                   "deleting any anchor deletes the group; when the latest index value of a combo token "
                   "is deleted the rule is evaluated again on the next-latest value with the graph "
                   "builder's rule check and the artifact's threshold table; an input missing from the "
                   "record is reported and the group is deleted; a rule that holds on the surviving values "
                   "gets its edges (and its state node when the record has none), so deleting a latest "
                   "value that failed a rule can add a state the record did not have")
EMPTY_VISIT_TEXT = "prior visits the deletion emptied are dropped; prior visits empty in the record are kept"
DEPARTURES_TEXT = ("the record with the nodes deleted, not the record the graph builder would produce "
                   "without the events: prior labs of an analyte whose index labs are all deleted stay "
                   "(with the analyte and its knowledge nodes), and a baseline_of edge is not re-anchored")


def intervention_record(combo: dict[str, Any] | None = None) -> dict[str, Any]:
    """The intervention every fidelity number of a record was computed with; `combo` is the combo
    source the deletions used (`recorded_combo_source`)."""
    return {"version": fv2.INTERVENTION_CONTRACT_VERSION,
            "definition": fv2.INTERVENTION_CONTRACT,
            "eligible_node_kinds": list(fv2.EVIDENCE_NODE_KINDS),
            "protected_node_kinds": list(fv2.STRUCTURAL_NODE_KINDS),
            "combos": COMBO_RULE_TEXT,
            "combo_source": combo,
            "cascade": ("analytes without a kept measurement, knowledge nodes without a kept anchor, "
                        "then nodes left without an edge"),
            "empty_visits": EMPTY_VISIT_TEXT,
            "chains": ("not relinked: a broken trajectory chain stays broken and a baseline_of edge is "
                       "not moved to the next index lab or the previous prior lab"),
            "departures_from_a_rebuild": DEPARTURES_TEXT}


def deletion_summary(scorer: DeletionScorer) -> dict[str, Any]:
    """What the visit's deletions did to combo states and visits, over every node set it scored:
    how many sets deleted a combo group, had a rule evaluated again or fired again, added a state
    node the record did not have, and every (rule, reason) a rule could not be evaluated again for."""
    infos = scorer.deletion_infos()
    untouched = infos[0]
    combos = [info.get("combos") or {} for info in infos]
    not_reevaluated = sorted({(case["rule_id"], case["reason"])
                              for combo in combos for case in combo.get("not_reevaluated", [])})
    return {"sets_scored": len(infos) - 1,
            "sets_with_combo_groups_deleted": sum(1 for c in combos if c.get("groups_deleted")),
            "sets_with_rules_reevaluated": sum(1 for c in combos if c.get("reevaluated")),
            "sets_with_rules_refired": sum(1 for c in combos if c.get("refired")),
            "sets_with_new_states": sum(1 for c in combos if c.get("new_state_nodes")),
            "not_reevaluated": [{"rule_id": rule_id, "reason": reason}
                                for rule_id, reason in not_reevaluated],
            "sets_with_visits_dropped": sum(1 for info in infos if info.get("visits_dropped")),
            "visits_originally_empty": untouched.get("visits_originally_empty")}


def explain_subject(wrapper, data, *, method: str, conv: str | None,
                    steps: int, epochs: int, num_relation_columns: int,
                    records: VisitRecords | None = None,
                    fidelity_v2_curve_points: int = fv2.CURVE_POINTS, ig: dict[str, Any] | None = None,
                    random_repeats: int = fv2.RANDOM_REPEATS, tie_draws: int = fv2.TIE_DRAWS,
                    random_seed: int = 1234, audit: dict[str, Any] | None = None,
                    true_class_explanations: bool = True,
                    stored_row=None) -> dict[str, Any]:
    """One record (schema 3) for one visit. `records` holds the visit's graph record and the run's
    encoder (`RebuiltSplits.visit_records`); every fidelity number deletes evidence from that record."""
    started = time.perf_counter()
    batch = _single_graph_batch(data)
    x = wrapper.set_context(batch)
    # Passing only x and edge_index is explaining a different model unless the wrapper provably
    # reproduces the real model (its own forward on the raw batch) on this very graph.
    faithful, wrapper_max_diff = wrapper.verify(x, batch.edge_index)
    if not faithful:
        raise ValueError(
            f"wrapper does not reproduce the model on sample {data.sample_id}: "
            f"max |logit difference| = {wrapper_max_diff:.3g}"
        )
    if records is None:
        raise ValueError("explain_subject needs the run's visit records (RebuiltSplits.visit_records) "
                         "to delete evidence from the visit's graph record")
    scorer = records.scorer(wrapper.deletion_forward, data)
    with torch.no_grad():
        single_logits = wrapper(x, batch.edge_index)
        prediction = int(single_logits.argmax(-1).item())
    single_graph_check = _single_graph_check(single_logits, stored_row)
    deletion_check = {
        "encoding_equals_split_tensor": True,
        "max_abs_prob_diff": float((torch.softmax(single_logits.detach(), dim=-1)
                                    - scorer.probs(())).abs().max()),
        "combo_source": _without_local_paths(scorer.combo_source),
    }
    true_class = int(data.y.view(-1)[0].item())
    seed = _stable_seed(random_seed, str(data.sample_id))
    explained_method = conv if method == "clinical_gnn" else method
    evaluate = functools.partial(_evaluate_sources, scorer, curve_points=fidelity_v2_curve_points,
                                 random_repeats=random_repeats, tie_draws=tie_draws, seed=seed)

    collected = _collect_importances(wrapper, batch, x, method=explained_method, steps=steps,
                                     epochs=epochs, ig=ig, prediction=prediction, seed=seed)
    fidelity_v2, random_info = evaluate(
        _source_vectors(collected, skip=FIDELITY_SKIPPED_SOURCES), prediction)

    try:
        if method in CEI_METHODS:
            edge_attr_contribution = {
                "status": "not_applicable",
                "reason": "CEI-GNN's graph metadata (edge_attr, relation, triple) is fixed inside "
                          "its bound wrapper, so the relation-vs-payload ablation cannot be applied"}
        elif conv == "hgt":
            edge_attr_contribution = {
                "status": "not_applicable",
                "reason": "hgt reads relation identity from edge_triple, not from edge_attr's "
                          "relation one-hot, so zeroing that block would under-report it"}
        else:
            edge_attr_contribution = fv2.relation_vs_payload_contribution(
                wrapper, x, batch.edge_index, batch.edge_attr, num_relation_columns, prediction,
            )
    except Exception as exc:  # noqa: BLE001 - must not block the rest of the record
        edge_attr_contribution = {"status": "failed", "error": str(exc)}
    try:
        dependence = audit_mod.edge_dependence(wrapper, x, batch.edge_index, prediction)
    except Exception as exc:  # noqa: BLE001
        dependence = {"status": "failed", "error": str(exc)}

    num_visits = getattr(data, "num_visits", None)
    record = {
        "schema": RECORD_SCHEMA,
        "schema_version": RECORD_SCHEMA_VERSION,
        "method": method,
        "conv": conv,
        "sample_id": str(data.sample_id),
        "subject_id": str(data.subject),
        "num_visits": None if num_visits is None else int(torch.as_tensor(num_visits).view(-1)[0]),
        "true_class_id": true_class,
        "prediction_class_id": prediction,
        "explained_target": "predicted_class",
        "wrapper_verification": {"max_abs_logit_diff": wrapper_max_diff, "tolerance": 1e-5},
        "single_graph_check": single_graph_check,
        "deletion_check": deletion_check,
        "builtin_node_importance": collected["builtin"],
        "builtin_available": collected["builtin"] is not None,
        "builtin_status": collected["builtin_status"],
        "builtin_class_specific": bool(getattr(wrapper, "builtin_class_specific", False)),
        "builtin_unavailable_reason": collected["builtin_unavailable"],
        "builtin_tie_summary": collected["builtin_tie_summary"],
        "builtin_detail": collected["builtin_detail"],
        "graphxai": collected["graphxai"],
        "fidelity_v2": fidelity_v2,
        "fidelity_v2_intervention": intervention_record(deletion_check["combo_source"]),
        "fidelity_v2_skipped_sources": [name for name in FIDELITY_SKIPPED_SOURCES
                                        if name in collected["graphxai"]],
        "random_reference": random_info,
        "edge_attr_contribution": edge_attr_contribution,
        "edge_dependence": dependence,
    }

    if true_class_explanations and true_class != prediction:
        # For a misclassified patient also explain the TRUE class, so a reader can see what the
        # model attended to versus what it should have.
        try:
            true_collected = _collect_importances(
                wrapper, batch, x, method=explained_method, steps=steps, epochs=epochs, ig=ig,
                prediction=prediction, target_class=true_class, seed=seed)
            true_change, true_random = evaluate(
                _source_vectors(true_collected, skip=FIDELITY_SKIPPED_SOURCES), true_class)
            record["true_class_explanations"] = {
                "target_class": true_class,
                "graphxai": {n: {k: v for k, v in p.items() if k != "node_explanation"} |
                             ({"node_importance": p["node_explanation"]["node_importance"]}
                              if "node_explanation" in p else {})
                             for n, p in true_collected["graphxai"].items()},
                "builtin_node_importance": true_collected["builtin"],
                "builtin_status": true_collected["builtin_status"],
                "builtin_unavailable_reason": true_collected["builtin_unavailable"],
                "builtin_detail": true_collected["builtin_detail"],
                "true_class_probability_change": {name: _without_acc(metrics)
                                                  for name, metrics in true_change.items()},
                "random_reference": true_random,
            }
        except Exception as exc:  # noqa: BLE001
            record["true_class_explanations"] = {"status": "failed", "error": str(exc)}

    if audit is not None:
        record["audit"] = _audit_subject(wrapper, batch, x, _source_vectors(collected), prediction,
                                         method=explained_method, steps=steps, epochs=epochs, ig=ig,
                                         audit=dict(audit, seed=seed))
    record["deletion_check"]["deletions"] = deletion_summary(scorer)
    record["seconds"] = round(time.perf_counter() - started, 3)
    return record


def _stored_rows(stored_npz: Path, prediction_sha256: str) -> dict[str, np.ndarray]:
    """sample id -> stored probability row of the fold's `<fold>.npz`; empty when it is absent or is
    not the training run's (its probabilities do not hash to the recorded `prediction_sha256`), which
    an exact-hash replay never opens the file to show."""
    if not stored_npz.exists():
        return {}
    stored = np.load(stored_npz, allow_pickle=False)
    if "proba" not in stored.files or proba_digest(stored["proba"]) != prediction_sha256:
        return {}
    return {str(i): row for i, row in zip(stored["sample_ids"], stored["proba"])}


def _ig_pilot_rows(splits, pilot: int, seed: int = 1234):
    """The IG start step count is a decision, so its pilot graphs come from a decision fold
    (validation, else dev), never from the report fold: `pilot` patients drawn hash-ordered as
    `select_explained_subjects` draws them, one visit (the first in fold order) per patient."""
    for pilot_fold in ("validation", "dev"):
        if splits.get(pilot_fold):
            drawn = select_explained_subjects(splits[pilot_fold], sample_patients=max(1, int(pilot)),
                                              sample_seed=seed)
            rows, seen = [], set()
            for data in drawn:
                if str(data.subject) not in seen:
                    seen.add(str(data.subject))
                    rows.append(data)
            return pilot_fold, rows
    raise ValueError("no validation or dev rows to choose the IG step count on")


def _prepare_clinical_ig(bundle, wrapper, splits, *, tolerance: float, candidates,
                         pilot_patients: int, reference_draws: int, keep_zero_baseline: bool,
                         seed: int, sample_seed: int) -> dict[str, Any]:
    """Pre-registered IG setup for one run: the baseline layout, and the start step count chosen
    from the completeness error on pilot patients of a decision fold (never a default, never the
    report fold). Per visit the steps are raised from there until completeness holds."""
    layout = (bundle.preprocessing or {}).get("node_feature_layout")
    if not layout:
        raise ValueError("preprocessing.json has no node_feature_layout; cannot build the IG baseline")
    candidates = tuple(sorted({int(steps) for steps in candidates}))
    spec = wrapper.ig_feature_spec() if hasattr(wrapper, "ig_feature_spec") else {}
    pilot_fold, pilot_rows = _ig_pilot_rows(splits, pilot_patients, seed=sample_seed)
    errors: dict[int, list[float]] = {}
    for steps in candidates:
        errors[steps] = []
        for data in pilot_rows:
            batch = _single_graph_batch(data)
            x = wrapper.set_context(batch)
            with torch.no_grad():
                target = int(wrapper(x, batch.edge_index).argmax(-1))
            baseline = ig_clinical.not_recorded_baseline(x, batch.node_type, layout, **spec)
            with _ig_path(wrapper):
                _, completeness = ig_clinical.integrated_gradients(
                    wrapper, x, batch.edge_index, baseline, target, steps)
            errors[steps].append(completeness["completeness_error_rel"])
        if max(errors[steps]) <= tolerance:
            break
    chosen = ig_clinical.choose_steps(errors, tolerance)
    bank = (ig_clinical.ReferenceBank.from_graphs(splits["train"][:2000])
            if reference_draws > 0 else None)
    chosen = dict(chosen, pilot_fold=pilot_fold, pilot_graphs=len(pilot_rows),
                  pilot_patients_requested=int(pilot_patients),
                  pilot_rule="patients hash-ordered by the sample seed, one visit each")
    return {"layout": layout, "steps": chosen["steps"], "tolerance": tolerance,
            "candidates": candidates, "feature_spec": spec,
            "feature_provenance": (wrapper.ig_feature_provenance()
                                   if hasattr(wrapper, "ig_feature_provenance") else {}),
            "step_selection": chosen, "bank": bank, "reference_draws": reference_draws,
            "keep_zero_baseline": keep_zero_baseline, "seed": seed}


def prototype_projection_status(run_dir, binding, fold: str | None = None) -> dict[str, Any] | None:
    """ProtGNN only: were the selected checkpoint's prototypes ever projected onto training
    subgraphs, and how long were they trained after the last projection? Projection runs at the
    start of epoch index `proj_epochs` (1-based epoch proj_epochs + 1) and every `proj_interval`
    epochs after, before that epoch's training steps; a checkpoint selected before that epoch
    carries learned prototype vectors that are not real training cases. The selected epoch
    (1-based) is read from result.json, else from binding.json's selected block of `fold`; None
    for other methods."""
    if binding.get("method") != "protgnn":
        return None
    settings = (binding.get("method_config") or {}).get("effective_settings") or {}
    if "proj_epochs" not in settings:
        return None
    selected, source = None, None
    try:
        selected = json.loads((Path(run_dir) / "result.json").read_text())["selected_epoch"]
        source = "result.json"
    except (OSError, KeyError, ValueError):
        block = binding.get(FOLD_BLOCK.get(fold or "", ""), None) if fold else None
        if isinstance(block, dict) and block.get("epoch") is not None:
            selected, source = block["epoch"], f"binding.json {FOLD_BLOCK[fold]}"
    if selected is None:
        return {"status": "selected_epoch_not_recorded"}
    selected = int(selected)
    first = int(settings["proj_epochs"]) + 1
    interval = int(settings.get("proj_interval", 0))
    projected = selected >= first
    last = None
    if projected:
        last = first if interval <= 0 else first + ((selected - first) // interval) * interval
    return {"selected_epoch": selected, "selected_epoch_source": source,
            "first_projection_epoch": first, "projection_interval": interval,
            "prototypes_projected": projected, "last_projection_epoch": last,
            "epochs_trained_after_last_projection": None if last is None else selected - last + 1,
            "note": PROJECTION_NOTE}


def explanation_source_hashes() -> dict[str, str]:
    """sha256 of every `.py` under the code roots (`core.paths.CODE_ROOTS`: the training code roots,
    which hold core/explain and comparison/explain), keyed `<root>/<relative path>`. `__pycache__` and
    names starting `._` (copy-tool leftovers) are skipped; the driver's `current_source_hashes`
    computes the same mapping."""
    hashes = {}
    for root in CODE_ROOTS:
        base = REPO_ROOT / root
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts or path.name.startswith("._"):
                continue
            hashes[f"{root}/{path.relative_to(base).as_posix()}"] = sha256(path)
    return hashes


def vendored_graphxai_tree_sha256() -> str:
    """One sha256 over the vendored GraphXAI package tree: every file (`__pycache__`, `.pyc` and
    `._` names skipped) keyed by its relative path."""
    files = sorted(path for path in VENDORED_GRAPHXAI.rglob("*")
                   if path.is_file() and "__pycache__" not in path.parts
                   and not path.name.startswith("._") and path.suffix != ".pyc")
    return _canonical_sha256({path.relative_to(VENDORED_GRAPHXAI).as_posix(): sha256(path)
                              for path in files})


def _git_state() -> dict[str, Any]:
    """HEAD and whether the working tree differs from it; `not_a_checkout` without a .git."""
    if not (REPO_ROOT / ".git").exists():
        return {"git_head": "not_a_checkout", "git_dirty": "not_a_checkout"}
    try:
        def git(*arguments):
            return subprocess.run(["git", *arguments], cwd=REPO_ROOT, capture_output=True, text=True,
                                  timeout=120, check=True).stdout
        head = git("rev-parse", "HEAD").strip()
        dirty = bool(git("status", "--porcelain").strip())
    except (OSError, subprocess.SubprocessError):
        return {"git_head": "unavailable", "git_dirty": "unavailable"}
    return {"git_head": head, "git_dirty": dirty}


def code_identity() -> dict[str, Any]:
    """The code an explanation run ran with: every code root's sources, the vendored GraphXAI tree
    and its upstream commit, and the checkout state."""
    return {"source_hashes": explanation_source_hashes(),
            "vendored_graphxai_tree_sha256": vendored_graphxai_tree_sha256(),
            "graphxai_upstream_commit": GRAPHXAI_UPSTREAM_COMMIT,
            **_git_state()}


def method_disclosure(method: str, *, epochs: int) -> dict[str, Any]:
    """What a reader needs to know to read one method's numbers: the intervention, the built-in
    explanation, the post-hoc explainers and the caveats."""
    from core.explain import graphxai_standardized as gx
    from core.explain.cei_bridge import EDGE_MASK_SCOPES, PLE_TERM_EXACT, PLE_TERM_IG

    intervention = (f"node deletion in the graph record (intervention contract version "
                    f"{fv2.INTERVENTION_CONTRACT_VERSION}): the record without the selected evidence "
                    "nodes and their derived concept nodes, re-encoded with the run's own preprocessing "
                    "and scored by the model's own forward; combo states: " + COMBO_RULE_TEXT + "; "
                    + EMPTY_VISIT_TEXT + "; trajectory chains are not relinked; this is "
                    + DEPARTURES_TEXT + "; nodes are ranked by |importance|, ties at a cut averaged "
                    "over seeded orders")
    explainers = {
        "gradient": ("GradExplainer and the clinical IG explain x = [numeric row | token vector]; node "
                     "type, edges and edge metadata stay fixed; node importance = sum of |attribution| "
                     "over every column of x"),
        "ig": ("clinical IG from the not-recorded baseline (token block zero, the pad embedding); "
               + IG_ESCALATION + "; the vendored zero-baseline IG is kept in the records only"),
        "gnnexplainer": (f"{epochs} epochs ({gx.GNNEXPLAINER_EPOCHS_NOTE}), seeded per visit, "
                         f"{gx.GNNEXPLAINER_COEFF_NOTE}; node importance = mean of the incident edge "
                         "masks; " + gx.SOFT_MASK_NOTE),
        "edge_mask_scope": gx.GNNEXPLAINER_DISCLOSURE.get(method),
    }
    caveats = ["a source whose importance is zero on every eligible evidence node is counted as "
               "no_eligible_evidence and left out of the means"]
    if method == "gsat":
        builtin = ("the node attention of a fresh eval pass of the model's own explain(); edge attention "
                   + GSAT_EDGE_ATTENTION_RULE + "; class-agnostic")
        caveats += ["GSAT attention can saturate at exactly 1.0, so equal importances (ties) are common"]
    elif method == "protgnn":
        builtin = (f"{PROTGNN_EXPLANATION_STYLE}; search: {PROTGNN_SEARCH_ALGORITHM}; class-specific")
        caveats += [f"search settings: the run's recorded settings (adaptation defaults unless "
                    f"overridden; {PROTGNN_OFFICIAL_SETTINGS})",
                    PROTGNN_SUBGRAPH_SCORING,
                    "a per-prediction subgraph (ProtGNN+ style), not the paper's whole-prototype view",
                    PROTGNN_SUBGRAPH_SIZE_NOTE,
                    PROJECTION_NOTE,
                    "runs selected before the first projection explain with learned prototype vectors "
                    "that are not training cases; they are flagged in prototype_projection"]
    elif method == "graphcare":
        builtin = ("the node attention of the last layer from a fresh unmasked eval forward: per node the "
                   "sum of last_visit_weight over its membership pairs; " + GRAPHCARE_VALUE_DEPENDENCE
                   + "; class-agnostic")
        caveats += ["on a single-visit graph the built-in is constant over the eligible nodes",
                    "edges with a global endpoint are never read by the model (inert_edges)"]
    elif method in CEI_METHODS:
        builtin = ("the exact additive accounting of the explained logit (node + half of each incident "
                   "edge/pair contribution), ranked by |signed contribution|; the shares of negative "
                   "eligible nodes and of mass on protected nodes are recorded; class-specific")
        explainers["ple_term"] = {"outside_ig": PLE_TERM_EXACT, "inside_ig": PLE_TERM_IG}
        caveats += [EDGE_MASK_SCOPES[method],
                    "a node that votes against the class is ranked by its magnitude"]
    else:
        builtin = None
        caveats += ["no built-in explanation (black-box baseline, explained post hoc only)"]
    return {"intervention": intervention, "builtin": builtin, "explainers": explainers,
            "caveats": caveats}


def builtin_explanation(binding: dict[str, Any], wrapper) -> dict[str, Any] | None:
    """Which built-in explanation the records carry, with its settings."""
    method = binding["method"]
    class_specific = bool(getattr(wrapper, "builtin_class_specific", False))
    if method == "protgnn":
        settings = wrapper._protgnn_settings()
        return {"method": method, "algorithm": PROTGNN_SEARCH_ALGORITHM, "settings": settings,
                "settings_source": wrapper._protgnn_settings_source(settings),
                "subgraph_scoring": PROTGNN_SUBGRAPH_SCORING, "class_specific": class_specific}
    if method == "gsat":
        return {"method": method, "algorithm": "node attention of a fresh eval pass (adapter.explain)",
                "settings": {"edge_attention_rule": GSAT_EDGE_ATTENTION_RULE},
                "class_specific": class_specific}
    if method == "graphcare":
        return {"method": method, "algorithm": "last-layer node attention (sum of last_visit_weight over "
                                               "membership pairs)",
                "settings": {"layer": "last", "value_depends_on": GRAPHCARE_VALUE_DEPENDENCE},
                "class_specific": class_specific}
    if method in CEI_METHODS:
        return {"method": method, "algorithm": "exact additive accounting of the logit (native_importance)",
                "settings": {"ranking": "|signed contribution|"}, "class_specific": class_specific}
    return None


def _relative_run_path(run_dir) -> str:
    """The run's path relative to the runs root (`top<K>/<method>_seed<seed>`): its last two parts."""
    return "/".join(Path(run_dir).resolve().parts[-2:])


def _without_local_paths(value):
    """A copy with every absolute path string replaced by its file name (outputs carry no local path;
    the files themselves are pinned by their recorded hashes)."""
    if isinstance(value, dict):
        return {key: _without_local_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_without_local_paths(item) for item in value]
    if isinstance(value, str) and os.path.isabs(value):
        return Path(value).name
    return value


def dated_output_dir(base, method: str, conv: str | None = None) -> Path:
    """A NEW, timestamped directory name under `base` (never reused)."""
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = f"explanations_{method}{'_' + conv if conv else ''}_{stamp}"
    return Path(base) / name


DEFAULT_EXPLAINED_PATIENTS = 500   # the same draw of 500 patients for every method


def run_explanations(run_dir, output_dir, *, fold: str = "test",
                     max_subjects: int | None = None,
                     sample_patients: int | None = None, sample_seed: int = 1234,
                     steps: int = 32, epochs: int = 50,
                     fidelity_v2_curve_points: int = fv2.CURVE_POINTS,
                     clinical_ig: bool = True, ig_tolerance: float = ig_clinical.DEFAULT_TOLERANCE,
                     ig_step_candidates=ig_clinical.DEFAULT_STEP_CANDIDATES,
                     ig_pilot_patients: int = IG_PILOT_PATIENTS, ig_reference_draws: int = 0,
                     keep_zero_baseline_ig: bool = True,
                     random_repeats: int = fv2.RANDOM_REPEATS, tie_draws: int = fv2.TIE_DRAWS,
                     random_seed: int = 1234,
                     audit_subjects: int = 0, stability_draws: int = 2,
                     stability_sigma: float = 0.05,
                     true_class_explanations: bool = True) -> dict[str, Any]:
    """The full pipeline: load -> rebuild and verify the split -> replay-check -> explain the chosen
    visits of `fold` ("test" = report fold; "validation" and "dev" = decision folds) -> write output.
    Raises before writing anything if a check fails. `steps` is the budget of the vendored
    (zero-baseline) IG; the clinical IG chooses its start step count from the completeness error on
    decision-fold pilot patients and raises it per visit until completeness holds or the largest
    candidate is reached."""
    run_started = time.perf_counter()
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"explanation output directory must be empty: {output_dir}")
    code = code_identity()

    bundle = load_run_bundle(run_dir, fold)
    binding = bundle.binding
    splits = rebuild_split(binding, fold)
    model, is_method_adapter = build_model(binding, bundle.state_dict)
    method_config = rebuilt_method_config(model, binding)
    replay = replay_check(model, splits, binding, is_method_adapter=is_method_adapter, fold=fold,
                          stored_npz=Path(run_dir) / f"{fold}.npz")

    subjects = select_explained_subjects(
        splits[fold], max_subjects=max_subjects,
        sample_patients=sample_patients, sample_seed=sample_seed)
    cohort = build_cohort_v2(
        binding, fold=fold,
        fold_sample_ids=[d.sample_id for d in splits[fold]],
        explained_sample_ids=[d.sample_id for d in subjects],
    )
    cohort["sampling"] = {"sample_patients": sample_patients, "sample_seed": sample_seed,
                          "max_subjects": max_subjects,
                          "patients_explained": len({str(d.subject) for d in subjects})}
    validate_cohort_v2(cohort, binding)
    records = splits.visit_records([d.sample_id for d in subjects])

    wrapper = build_wrapper(binding, model)
    method, conv = binding["method"], binding.get("conv")
    ig_config = None
    if clinical_ig and subjects:
        ig_config = _prepare_clinical_ig(
            bundle, wrapper, splits, tolerance=ig_tolerance, candidates=ig_step_candidates,
            pilot_patients=ig_pilot_patients, reference_draws=ig_reference_draws,
            keep_zero_baseline=keep_zero_baseline_ig, seed=random_seed, sample_seed=sample_seed)
    audit_config = {"model": model, "make_wrapper": lambda m: build_wrapper(binding, m),
                    "stability_draws": stability_draws, "sigma": stability_sigma,
                    "seed": random_seed}

    stored_rows = _stored_rows(Path(run_dir) / f"{fold}.npz", binding[FOLD_BLOCK[fold]]["prediction_sha256"])
    seconds_before_first_explanation = round(time.perf_counter() - run_started, 3)
    explained, failures = [], []
    for index, data in enumerate(subjects):
        try:
            explained.append(
                explain_subject(
                    wrapper, data, method=method, conv=conv, steps=steps, epochs=epochs,
                    num_relation_columns=binding["num_relations"], records=records,
                    fidelity_v2_curve_points=fidelity_v2_curve_points, ig=ig_config,
                    random_repeats=random_repeats, tie_draws=tie_draws, random_seed=random_seed,
                    audit=audit_config if index < audit_subjects else None,
                    true_class_explanations=true_class_explanations,
                    stored_row=stored_rows.get(str(data.sample_id)))
            )
        except Exception as exc:  # noqa: BLE001 - one subject's failure must not lose the rest
            failures.append({"sample_id": str(data.sample_id), "error": str(exc)})

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cohort.json").write_text(json.dumps(cohort, indent=2) + "\n")
    record_files = []
    for record in explained:
        filename = f"subject_{record['sample_id']}.json"
        (output_dir / filename).write_text(json.dumps(record, indent=2) + "\n")
        record_files.append(filename)
    ig_summary = None if ig_config is None else {
        "baseline": "not_recorded", "steps": ig_config["steps"], "start_steps": ig_config["steps"],
        "tolerance": ig_config["tolerance"], "step_candidates": list(ig_config["candidates"]),
        "escalation": IG_ESCALATION, "step_selection": ig_config["step_selection"],
        "real_patient_reference_draws": ig_config["reference_draws"],
        "kept_zero_baseline_reference": ig_config["keep_zero_baseline"]}
    settings = {
        "curve_points": fidelity_v2_curve_points,
        "random_repeats": random_repeats,
        "tie_draws": tie_draws,
        "random_seed": random_seed,
        "gnnexplainer_epochs": epochs,
        "gnnexplainer_seed_rule": GNNEXPLAINER_SEED_RULE,
        "gnnexplainer_feature_penalty": "mean",
        "gnnexplainer_node_reduction": "mean",
        "vendored_ig_steps": steps,
        "ig": ({"enabled": False} if ig_summary is None else {
            "enabled": True, "baseline": "not_recorded", "tolerance": ig_summary["tolerance"],
            "step_candidates": ig_summary["step_candidates"], "escalation": IG_ESCALATION,
            "pilot": f"{ig_pilot_patients} decision-fold patients, one visit each",
            "start_steps": ig_summary["start_steps"]}),
        "fidelity_skipped_sources": list(FIDELITY_SKIPPED_SOURCES),
        "true_class_explanations_for_misclassified": true_class_explanations,
        "audit": {"subjects": audit_subjects, "stability_draws": stability_draws,
                  "stability_sigma": stability_sigma},
    }
    config = {
        **settings,
        "steps_vendored_ig": steps, "node_reduction": "mean",
        "fidelity_v2_curve_points": fidelity_v2_curve_points,
        "intervention_contract_version": fv2.INTERVENTION_CONTRACT_VERSION,
        "eligible_node_kinds": list(fv2.EVIDENCE_NODE_KINDS),
        "protected_node_kinds": list(fv2.STRUCTURAL_NODE_KINDS),
        "combo_source": splits.combo_source,
        "clinical_ig": ig_summary,
        "random_reference": {"repeats": random_repeats, "seed": random_seed},
    }
    disclosure = {name: method_disclosure(name, epochs=epochs) for name in FINAL_METHODS}
    if method not in disclosure:
        disclosure[method] = method_disclosure(method if method != "clinical_gnn" else str(conv),
                                               epochs=epochs)
    run_name = _relative_run_path(run_dir)
    manifest = {
        "schema": RUNNER_SCHEMA,
        "schema_version": RUNNER_SCHEMA_VERSION,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "intervention_contract_version": fv2.INTERVENTION_CONTRACT_VERSION,
        "combo_source": splits.combo_source,
        "method": method,
        "conv": conv,
        "run_dir": run_name,
        "binding_preprocessing_sha256": binding["preprocessing_sha256"],
        "fold": fold,
        "fold_role": FOLD_ROLE[fold],
        "fold_sample_ids_sha256": binding["split_sample_ids_sha256"][fold],
        "replay_verified": True,
        "replay": replay,
        "cohort_schema_version": cohort["schema_version"],
        "cohort_file": "cohort.json",
        "cohort_is_full_fold": cohort["is_full_fold"],
        "binding_file": "binding.json",
        "requested_subject_count": len(subjects),
        "explained_count": len(explained),
        "failed_count": len(failures),
        "failures": failures,
        "steps": steps,
        "epochs": epochs,
        "clinical_ig": ig_summary,
        "settings": settings,
        "disclosure": disclosure,
        "builtin_explanation": builtin_explanation(binding, wrapper),
        "prototype_projection": prototype_projection_status(run_dir, binding, fold),
        "record_files": record_files,
        "environment": environment_info(),
        "seconds_before_first_explanation": seconds_before_first_explanation,
        "seconds_total": round(time.perf_counter() - run_started, 3),
        "code": code,
    }
    (output_dir / "explanation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # The run's own binding.json: what was explained, by which model and code, with which explainer
    # settings, each pinned by a hash.
    run_path = Path(run_dir)
    selected = binding.get(FOLD_BLOCK[fold]) or {}
    binding_out = {
        "schema": BINDING_SCHEMA,
        "schema_version": BINDING_SCHEMA_VERSION,
        "model_run": {
            "run_dir": run_name,
            "binding_json_sha256": sha256(run_path / "binding.json"),
            "checkpoint_sha256": sha256(run_path / "best.pt"),
            "preprocessing_sha256": binding["preprocessing_sha256"],
            "method": method, "conv": conv, "edge_direction": binding["edge_direction"],
            "protocol": binding.get("protocol"),
            "protocol_overrides": binding.get("protocol_overrides"),
            "top_k_labels": binding.get("top_k_labels"),
            "num_classes": binding.get("num_classes"),
            "seed": binding.get("seed"), "selection_fold": binding.get("selection_fold"),
            "final_eval": binding.get("final_eval"),
            "selected_epoch": selected.get("epoch"),
            "method_config_sha256": None if method_config is None else _canonical_sha256(method_config),
            "effective_settings": _without_local_paths(
                (method_config or {}).get("effective_settings") or {}),
        },
        "cohort": {
            "file": "cohort.json", "sha256": sha256(output_dir / "cohort.json"),
            "fold": cohort["fold"], "fold_role": FOLD_ROLE[fold],
            "fold_sample_ids_sha256": cohort["fold_sample_ids_sha256"],
            "explained_count": cohort["explained_count"], "is_full_fold": cohort["is_full_fold"],
            "sampling": cohort["sampling"],
        },
        "explainer_config": config,
        "records": {"count": len(explained), "manifest": "explanation_manifest.json",
                    "manifest_sha256": sha256(output_dir / "explanation_manifest.json")},
        "source_code": code["source_hashes"],
        "code": {key: value for key, value in code.items() if key != "source_hashes"},
    }
    (output_dir / "binding.json").write_text(json.dumps(binding_out, indent=2, sort_keys=True) + "\n")
    return manifest


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--output", type=Path, help="a new, empty directory")
    where.add_argument("--output-base", type=Path,
                       help="create a new dated directory under this path")
    parser.add_argument("--fold", choices=list(FOLD_BLOCK), default="test",
                        help="fold to explain: test (report fold, default), validation or dev "
                             "(decision folds)")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--sample-patients", type=int, default=DEFAULT_EXPLAINED_PATIENTS,
                        help="explain N patients (all their visits in the fold), the same draw for "
                             f"every method; default {DEFAULT_EXPLAINED_PATIENTS}; 0 = every patient")
    parser.add_argument("--sample-seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--fidelity-v2-curve-points", type=int, default=fv2.CURVE_POINTS)
    parser.add_argument("--no-clinical-ig", action="store_true",
                        help="keep only the vendored all-zero-baseline IG")
    parser.add_argument("--ig-tolerance", type=float, default=ig_clinical.DEFAULT_TOLERANCE)
    parser.add_argument("--ig-pilot-patients", type=int, default=IG_PILOT_PATIENTS,
                        help="decision-fold patients the IG start step count is chosen on")
    parser.add_argument("--ig-reference-draws", type=int, default=0,
                        help="real-patient-baseline IG robustness check (costs this many IG passes)")
    parser.add_argument("--random-repeats", type=int, default=fv2.RANDOM_REPEATS)
    parser.add_argument("--tie-draws", type=int, default=fv2.TIE_DRAWS)
    parser.add_argument("--audit-subjects", type=int, default=0,
                        help="run weight-randomisation + stability controls on the first N explained visits")
    args = parser.parse_args()
    output = args.output
    if output is None:
        run_binding = json.loads((args.run_dir / "binding.json").read_text())
        output = dated_output_dir(args.output_base, run_binding["method"], run_binding.get("conv"))
    manifest = run_explanations(
        args.run_dir, output, fold=args.fold, max_subjects=args.max_subjects,
        sample_patients=args.sample_patients or None, sample_seed=args.sample_seed,
        steps=args.steps, epochs=args.epochs,
        fidelity_v2_curve_points=args.fidelity_v2_curve_points,
        clinical_ig=not args.no_clinical_ig, ig_tolerance=args.ig_tolerance,
        ig_pilot_patients=args.ig_pilot_patients,
        ig_reference_draws=args.ig_reference_draws, random_repeats=args.random_repeats,
        tie_draws=args.tie_draws, audit_subjects=args.audit_subjects,
    )
    print(json.dumps({k: manifest[k] for k in
                      ("method", "conv", "explained_count", "failed_count")} | {"output": str(output)}))


if __name__ == "__main__":
    _cli()
