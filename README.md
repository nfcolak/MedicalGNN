# Self-Explainable GNN Diagnosis on MIMIC-IV ED Patient Graphs (final)

This repository holds the final benchmark of five self-explainable graph neural networks (CEI-GNN mp2, ProtGNN, GSAT,
GraphCare, GCHM-PNA) for primary-diagnosis prediction on multi-visit clinical graphs built from MIMIC-IV
emergency-department data. Protocol `fair_v10t`: the K = 10 / 20 / 30 most frequent diagnoses, train = the whole TRAIN
fold, checkpoint selection on validation, the held-out test fold scored once per run; 3 seeds per method. Decisions (checkpoint, hyper-parameters, baseline choice) use validation; every reported result is on test.
It contains only the code and data behind the final results. The exact commands are listed under "Final commands" below.

## Setup

```bash
conda env create -f environment.yml
conda activate protgnn-mimic
```

Run Python as `env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3` from the repo root.

## Reproduce the data from raw

The data is MIMIC-IV-ED and MIMIC-IV, available through PhysioNet credentialed access (https://physionet.org/content/mimic-iv-ed/ and https://physionet.org/content/mimiciv/). You must complete the PhysioNet credentialing and sign the data use agreements yourself. The raw CSVs and everything derived from them (cohorts, graphs, targets, predictions) are not included in this repository. With the raw files and this repository alone, the pipeline rebuilds every final input.

Download these 9 files and decompress them (`gunzip`) into ONE folder (`MIMIC_RAW_DIR`):

| File | Module | Read by the pipeline |
|---|---|---|
| `edstays.csv` | MIMIC-IV-ED 2.2 (`ed/`) | yes: event index, graph |
| `diagnosis.csv` | MIMIC-IV-ED 2.2 | yes: labels, graph |
| `triage.csv` | MIMIC-IV-ED 2.2 | yes: graph |
| `vitalsign.csv` | MIMIC-IV-ED 2.2 | no (listed for completeness) |
| `medrecon.csv` | MIMIC-IV-ED 2.2 | no (listed for completeness) |
| `pyxis.csv` | MIMIC-IV-ED 2.2 | no (listed for completeness) |
| `patients.csv` | MIMIC-IV 3.x (`hosp/`); 364,627 rows | yes: graph (age) |
| `omr.csv` | MIMIC-IV 3.x (`hosp/`) | no (listed for completeness) |
| `labevents.csv` | MIMIC-IV 3.x (`hosp/`), about 18 GB | yes: event index |

The two reference tables the label step needs are tracked in `data/reference/`: `icd9_to_icd10_mapping.csv` (the CMS General Equivalence Mappings ICD-9 -> ICD-10 table, not MIMIC data) and `label_reference_v1.json` (the frozen 30-class order and the hashes of the historical label sources, which the label step checks).

Then run all five stages (`events`, `stays`, `labels`, `graph`, `k4`):

```bash
MIMIC_RAW_DIR=/path/to/raw MEDGNN_DATA_ROOT=/path/to/data_root EXECUTE=1 bash comparison/final/pipeline.sh all
```

| Variable | Meaning |
|---|---|
| `MIMIC_RAW_DIR` | required: the one folder with the decompressed raw CSVs |
| `MEDGNN_DATA_ROOT` | root under which `comparison/standardized/event_inputs` is built and read (default: the repo root) |
| `EXECUTE` | `1` runs the commands; otherwise they are only printed |
| `PYTHON` | Python interpreter (default `python3`) |

Without `EXECUTE=1` the script only prints the commands, and a stage can be run alone (`events|stays|labels|graph|k4`). It refuses to write into an output folder that already exists. Optional: `MEDGNN_EVENT_INPUTS` (real older event manifests; the tests that need them are skipped when it is unset).

Measured durations of a full from-raw rebuild on one laptop (Apple silicon, SSD), about 65 minutes in total: the `events` stage takes about 48 minutes (four `labevents.csv` scans of 7-18 minutes each, legacy and v5 index, each uncapped and capped at 6 visits per subject, then the repairs and the +60 min index); `stays` and `labels` take a few seconds; the `graph` build takes about 12 minutes; `k4` takes about 4 minutes for the three states (K = 10, 20, 30).

Check the rebuild against these sha256 values (`sha256sum` or `shasum -a 256`; the paths are relative to `MEDGNN_DATA_ROOT/comparison/standardized/event_inputs/`):

| File | sha256 |
|---|---|
| `common_stays_withtest_v5_20261011/stays.txt` | `2d43991be35bb6ba88b867a847b64a97642766504546063ff1acaaa1c42c13d9` |
| `targets_primary_v4_v5plus60_v5_test_20261011/targets.csv` | `6b251176949593c38c0ed2c5fd0bd1d7d620a92f89db1a71de47966574a40f39` |
| `clinical_graph_L4_plus60_v5c_20261009/cohort.csv` | `2c531e7abbf4a7f4628c1494606cfc8f1d6f12d6eb1c044801f8dc7e36af68a9` |
| `clinical_graph_L4_plus60_v5c_20261009/visit_membership.jsonl` | `9e2399f39c57ef0fb8831eccd8f118e5315a3cb1fd781f1161685881300090ed` |
| `v3_state_L4_v5c_v10t_top10_20261011/K4.json` | `602cfcad3b7159239c2406291ef415c492617dca49ec742910bb89bbf30a8458` |
| `v3_state_L4_v5c_v10t_top20_20261011/K4.json` | `76d730398aeff69889933f04cdb0617fd66e9823d54d29822de4e0c5a157de17` |
| `v3_state_L4_v5c_v10t_top30_20261011/K4.json` | `e0354e283804dca9ad63d9654161df07081f4b8a928a577673d936a8a2f58afb` |
| `clinical_graph_L4_plus60_v5c_20261009/graphs.jsonl` | `3aa1dc658e7153faf045bb70f43ca065b222e362ea0c86bcb4a60a35666858dd` |

The graph hash is that of the `graphs.jsonl` file as produced by this repository. A full rebuild from the raw files reproduced all eight values.

Graph hash recorded by the final runs. The 45 final runs were trained on the graph built on 2026-10-10, and their `binding.json` files record different graph hashes than the table above. Before publication one provenance label was renamed in the graph builder: the `reviewed_by` value of the source-checked knowledge rows (`edges[].provenance.reviewed_by` in `graphs.jsonl`, the `reviewed_by` column of `thresholds.csv`) changed to `automated-source-check-not-clinical-review`. No other field changed: nodes, edges, features, labels and the visit membership are identical.

| File | Rebuilt by this repository | Trained file (recorded in the run bindings) |
|---|---|---|
| `graphs.jsonl` | `3aa1dc65…58dd`, 6,435,138,687 bytes | `53c92ea09159274528b2cb2ca454ad89ddb8064721007e29821a570f1ddf91ad`, 6,413,118,495 bytes |
| `thresholds.csv` | `3571a87d…4b12`, 79,192 bytes | `0113be57055febb5dc26b1b3b8b8fe9cf1d44e802fb3477f4485339b355d2b04`, 78,820 bytes |
| `visit_membership.jsonl` | `9e2399f3…90ed` | same |
| `manifest.json` | differs | `e5d8028ac68e31658cec6cc6588064ded23dec9aa41832597ef9cb067ffff34b` (records build-machine paths and code hashes; not reproducible) |

`comparison/final/graph_hash_check.py` hashes a rebuilt graph folder as it is and with the label swapped back in memory; the swapped hashes equal the trained files byte for byte (read-only, prints hashes only, about one minute):

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m comparison.final.graph_hash_check [--graph-dir DIR] [--binding <run>/binding.json]
```

### Regenerate the comparison tables

`comparison/final/compare_all.py` rebuilds the three result tables (K = 10, 20, 30) from the 45 stored final runs (5 methods x seeds 1234, 2025, 7) into a new folder. With `--check`, every produced file is compared with the stored tables.

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m comparison.final.compare_all \
  --runs comparison/standardized/clinical_runs_fair_v10t_20261011 \
  --out <new dir> \
  --check comparison/standardized/final_compare_fv10t_test_20261019
```

The stored tables in `final_compare_fv10t_test_20261019` list the test ranking first and keep the validation ranking as the labelled selection column. The folder `final_compare_fv10t_20261012` is the earlier layout (validation ranking first) and stays untouched. The check compares every produced file (`compare_top{10,20,30}.txt` and `compare_top{10,20,30}.json/comparison.json`) with the stored tables byte for byte. The default method list reproduces the published tables; `--methods` selects a subset.

### Hyper-parameter search

`comparison/final/hpo_all.py` lists the recorded search for the four tuned methods (CEI-GNN mp2, GSAT, GraphCare-style (adapted), ProtGNN): the `comparison.final.hpo` command (K = 17, 10k TRAIN sample, 20 epochs, 12 trials, TPE seed 2026, study name `fv10_k17_20261011`), the best trial from the stored `best.json`, and the matching `args()` line of `comparison/final/run.py`.

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m comparison.final.hpo_all
```

Result: 0 mismatches between the best trial values and the `run.py` flags. The default only prints. `--execute` reruns the search (training models, so it needs compute and approval).

### Split

`comparison/canonical_split.json` is frozen and is never regenerated in place. It was made in June 2026 from `merged_ed.csv`, the one-row-per-patient table written by the frozen raw preprocessing `data/s1_clean/merge_ed.py` (77,697 patients, sha256 `95b055d5…`). Rule: keep the 74,511 patients with a `disease_1` label and at least one code (medication class with prevalence of at least 1%, symptom or chief complaint); per `disease_1` class, shuffle with numpy `default_rng(1234)` and take round(0.8 n) for train, round(0.1 n) for validation and the rest for test. Each patient is in exactly one fold, so all visits of a patient share it.

`comparison/final/build_split.py` ports this rule and rebuilds the split from `merged_ed.csv` in memory (input path `comparison/standardized/event_inputs/split_input_merged_ed/merged_ed.csv`, git-ignored). It checks the input hash and compares the result with the frozen file. It writes only with `--out` to a new path and refuses the frozen file.

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m comparison.final.build_split
```

Result: the rebuilt split is byte-identical to the frozen file (sha256 `33f6cd71…`; 74,511 patients, 0 fold differences; train 59,607, validation 7,448, test 7,456; about 6 s).

`comparison/final/split_check.py` needs no input table. It checks the frozen file's structure, fold shares and agreement with the final targets, and never writes.

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m comparison.final.split_check
```

Result (origin not re-derivable, consistency checked): 74,511 subjects; folds train / validation / test 59,607 / 7,448 / 7,456 (0.800 / 0.100 / 0.100); no subject in two folds; all 51,961 subjects of the final targets have the same fold as in the targets file; per-class train share 0.778-0.844, within 0.017 of 0.8 for the 24 classes with at least 500 subjects. SHA-256 of the frozen file: `33f6cd71399ab2605948d66f80902030cc0b36ae1017bdde551f4504e4bf82b0`.

## Final commands

Protocol `fair_v10t`: K = 10 / 20 / 30 most frequent diagnoses; train = the whole TRAIN fold; checkpoint selected on validation; the test fold is scored once per run; 40 epochs; seeds 1234, 2025, 7.

Each of the five methods is trained with `python3 -m core.train <method args> --top-k-labels <K> --artifact <graph dir> --targets <targets dir> --output <new run dir> --seed <seed> --cpu-threads <n> --execute`, where `<method args>` are (as in `comparison/final/run.py`, function `args()`):

```
gsat       --method gsat --optimizer adamw_cosine --lr 0.00369 --weight-decay 6.17e-5 --dropout 0.137 --gsat-info-loss-coef 0.104
graphcare  --method graphcare --optimizer adamw_cosine --lr 0.00281 --weight-decay 1.05e-6 --dropout 0.286 --graphcare-decay-rate 0.0116
cei_mp2    --method cei_gnn_mp2 --method-option arm=C --method-option k=4 --method-option v3_state=<v3 state dir for K> --method-option mp_feed=dual --method-option mp_hub_max=1 --method-option mp_dedup=1 --method-option mp_ctx=table --method-option mp_root=bias --method-option mp_optimizer=adamw --method-option mp_lr_schedule=cosine --method-option mp_label_smoothing=0.1 --lr 0.00237 --weight-decay 1.03e-6 --dropout 0.493
protgnn    --method protgnn --optimizer adamw_cosine --lr 0.00261 --weight-decay 4.42e-6 --dropout 0.345 --protgnn-prototypes-per-class 5
gchm_pna   --method clinical_gnn --conv gchm
```

The `--weight-decay 1.03e-6` flag of the cei_mp2 command is ignored: CEI-GNN builds its own AdamW with decoupled weight decay 0.01 on weight matrices only (0 on biases, norms and embeddings; `mp_weight_decay`, `cei/options.py`), a cosine schedule, label smoothing 0.1 and DropEdge 0.2. The other methods use no label smoothing and no DropEdge: GSAT, GraphCare and ProtGNN use AdamW with their tuned weight decay (6.17e-5, 1.05e-6, 4.42e-6) on all parameters and the same cosine schedule; GCHM-PNA keeps its defaults (AdamW, weight decay 1e-5, constant learning rate).

`python3 -m comparison.final.run [--ks 10,20,30]` builds these commands for all 45 runs (`--plan` only checks them).

## Results

Test macro-F1, mean ± SD over 3 seeds (methods in test-ranking order of K = 10; the validation macro-F1 is the decision fold and is listed in the full tables). Full tables with accuracy, top-3 accuracy and paired bootstrap comparisons: [K = 10](results/test_top10.md), [K = 20](results/test_top20.md), [K = 30](results/test_top30.md).

| Method | K = 10 | K = 20 | K = 30 |
|---|---|---|---|
| GraphCare | 0.763 ± 0.003 | 0.671 ± 0.004 | 0.547 ± 0.002 |
| CEI-GNN mp2 | 0.761 ± 0.009 | 0.689 ± 0.001 | 0.578 ± 0.004 |
| GCHM-PNA | 0.756 ± 0.005 | 0.671 ± 0.010 | 0.556 ± 0.009 |
| ProtGNN | 0.751 ± 0.006 | 0.665 ± 0.002 | 0.557 ± 0.002 |
| GSAT | 0.751 ± 0.004 | 0.669 ± 0.004 | 0.557 ± 0.008 |

At K = 30, Hypothyroidism has 0 test rows (F1 0 for every method) and End stage renal disease has 5.

### Analysis packages and their results

Further analyses read the stored predictions of the final runs (no training). Each reports on test; validation stays as the labelled decision fold (checkpoint, best baseline, which arms enter an analysis).

| Package | What it analyses | Tables |
|---|---|---|
| `comparison/nasim` | section 2: CEI-GNN with the baselines' training recipe, GSAT grid, ProtGNN projection, GCHM-PNA cosine schedule (`analyze.py` arm vs its reference, `cross_method.py` arm vs every baseline, `edge_pilot.py` bidirectional-edge pilot on a 10k training sample) | `results/baseline_fidelity_k20/` |
| `comparison/s3_ledger` | section 3: evidence-ledger probes of CEI-GNN (block shares, ledger exactness, node-removal gap, pair additivity) | `results/s3_ledger/` |
| `comparison/ablation` | section 4: CEI-GNN component ablation at K = 20 | `results/ablation_k20/` |
| `comparison/splits` | section 4/6: replication of the ablation on two further patient splits | `results/split_replication_k20/` |
| `comparison/s6_stats` | section 6: per-seed results, patient-only and two-level seed and patient intervals, macro-F1 variants, per-class tables, the cut-off = ED discharge subset (`cutoff_subset.py`) | `results/s6_stats/` |
| `comparison/s7_why` | section 7: class-support analysis, same classes across K (`support.py`), inference-time source masking (`masking.py`) | `results/s7_why/support/`, `results/s7_why/masking/` |
| `comparison/s8_cost` | section 8: parameter counts, inference benchmark, inference memory, training cost | `results/s8_cost/` |
| `comparison/s9_repro` | section 9: graph scan for the reproduction check | `results/s9_repro/` |
| `comparison/explain` | explanations of the trained final models of GSAT, GraphCare, ProtGNN and CEI-GNN mp2 as they are: GradExplainer and GNNExplainer (vendored GraphXAI, `external/GraphXAI-main`, upstream a11e65f; GNNExplainer 50 epochs, seeded per visit, mean feature-mask penalty), Integrated Gradients from a not-recorded baseline (`core/explain/ig_clinical.py`; start step count chosen on validation, raised per visit until completeness holds or 256 steps are reached, the share of visits that met it is reported; the vendored zero-baseline IG stays in the records only) over the same input for every model (numeric row and token vector), and each model's own explanation (GSAT node attention, GraphCare last-layer visit attention, ProtGNN prototype subgraph from the model's own projection search with its recorded settings, CEI-GNN evidence ledger). Fidelity deletes the evidence from the graph record (with the concept nodes derived from it), re-encodes it with the run's preprocessing and scores the model's own forward (a combo threshold state goes with any of its required components and is evaluated again on the next-latest value with the artifact's threshold table, and only prior visits the deletion emptied are dropped); deletion/insertion curves and every number are compared with random deletion on the same graph, and the headline is the width-matched difference to random deletion. Not computed: the GraphXAI paper metrics that need ground truth (`run.py` drives `core/explain/runner.py` over the runs, `summarize.py` writes the aggregate tables, `replay_diff.py` shows how far a replay is from the stored predictions; `--checkpoint-dir <dir>` on the runner writes each explained visit's record there at once, so a long run that stopped is resumed by starting it again with the same directory) | not run yet |

Source masking (`results/s7_why/masking/`) scores the test fold with the stored final checkpoints (inference only; baselines chosen on validation; unmasked predictions match the stored test predictions in all 27 runs). `--fold validation` reproduces the earlier validation tables.

# Folder guide

## 1. Top level

```
MedicalGNN/
├── cei/            CEI-GNN (final version: mp2)
├── gsat/           GSAT
├── graphcare/      GraphCare
├── protgnn/        ProtGNN
├── gchm_pna/       GCHM-PNA (trained as --method clinical_gnn --conv gchm)
├── core/           training engine shared by all methods
├── data/           data pipeline, raw CSVs -> patient graphs (5 stages)
├── comparison/     final runner, comparison, split, inputs and results
├── tests/          unit tests for the kept code (run only in a testing phase)
├── results/        result tables (aggregates only): test_top{10,20,30}.md and one folder per analysis package (baseline_fidelity_k20, s3_ledger, ablation_k20, split_replication_k20, s6_stats, s7_why, s8_cost, s9_repro)
├── README.md       public front page
├── environment.yml / LICENSE / .gitignore
```

Rule of thumb: every method has its own top-level folder. Everything shared goes to `core/`. Everything that turns raw data into graphs is in `data/`. Everything that runs, compares or stores results is in `comparison/`.

## 2. Methods

| Folder | What is inside | Entry point |
|---|---|---|
| `cei/` | Final CEI-GNN = mp2: `options.py` (all mp2 options and their checks), `layers.py` (message-passing stack), `network.py` (the model), `plugin_cei_gnn_mp2.py` (adapter, method id `cei_gnn_mp2`), `pretrain.py` (optional pretraining, not used in the final runs). mp2 builds on the older CEI code, which stays: `cei_gnn.py`, `cei_gnn_v2.py`, `cei_gnn_v3.py`, `plugin_cei_gnn_v2.py`, `plugin_cei_gnn_v3.py`, and the `studies/` modules it imports (`cei_v3_study.py` fits the v3 state, `cei_v3_ple.py`, `cei_v3_absence.py`, `cei_pilot.py`, `cei_v3_ext/`). | `--method cei_gnn_mp2` |
| `gsat/` | `adapter.py` (GIN predictor + stochastic attention) | `--method gsat` |
| `graphcare/` | `adapter.py` (visit-conditioned attention) | `--method graphcare` |
| `protgnn/` | `adapter.py` (prototype-based GNN) | `--method protgnn` |
| `gchm_pna/` | `gchm_v2.py`, `gchm_v3.py` (hub-gated, relation-aware PNA layers used by `core/model.py`) | `--method clinical_gnn --conv gchm` |

Methods are found automatically: `core/registry.py` imports every `plugin_*.py` inside the folders listed in `METHOD_FOLDERS`.

## 3. `core/` (training engine)

| File | Role |
|---|---|
| `train.py` | The one training entry point (`python3 -m core.train`). Holds the protocol (`PROTOCOL_NAME = 'fair_v10t'`, `FAIR_PROTOCOL`, `ALLOWED_TOP_K_LABELS = (10, 20, 30)`), loads data, trains, selects the checkpoint on validation and scores the test fold once. Without `--execute` it only prints the plan. |
| `model.py` | Built-in GNN (`clinical_gnn`, incl. the `gchm` conv). |
| `tensorize.py` | Turns a graph (JSON) into PyTorch Geometric tensors. |
| `contracts.py` | Input checks and hashes written into each run's `binding.json`. |
| `registry.py` / `method_base.py` | Method discovery and the common adapter interface. |
| `paths.py` | Repo paths and the code folders hashed into each run binding. |
| `schema.py`, `rewiring.py` | Graph schema; optional relation rewiring controls. |

## 4. `data/` (pipeline, in stage order)

```
data/
├── reference/         tracked inputs of the label step (not MIMIC data)
│   ├── icd9_to_icd10_mapping.csv   ICD-9 -> ICD-10 table (CMS General Equivalence Mappings, GEMs)
│   └── label_reference_v1.json     frozen 30-class order + historical source hashes checked by the labels
├── s1_clean/          historical cleaning code (FROZEN: never edit, import or run)
├── s2_events/         event index + cohort
│   ├── first_lab.py           first-lab cohort scan (--lab-scope representable-v5; all-numeric-known-unit for the legacy cohort)
│   ├── event_store_v5.py      v5 event store
│   ├── repair_metadata.py     max-6-visit cap check, marks the index ready
│   ├── first_lab_delay.py     +60 min variant of the index (used by the final graph)
│   ├── knowledge_v3.csv, knowledge_thresholds_v2.csv   medical knowledge rules of the final graph
│   └── knowledge_seed.csv, enriched_spec.py, graph.py, ingest.py   (FROZEN)
├── s3_labels/         targets
│   ├── local_labels_v4.py     final labels (primary_dx_v4 = the visit's primary ED diagnosis)
│   ├── icd_mapping_v2.py + icd9_tiebreak_overrides_v1.csv   ICD-9 -> ICD-10 mapping
│   └── icd_mapping.py (FROZEN), labels.py, local_labels_v2/v3.py (older producers still imported)
├── s4_graph/          graph build
│   ├── build_delay.py         final L4 graph (+60 min, thresholds, row-range levels)
│   └── build.py, diagnosis.py, graph.py, store.py
└── s5_filter_split/
    └── selection.py           top-K labels, train/dev/validation id selection
```

Raw MIMIC CSVs are NOT in this repo (see "Reproduce the data from raw").

## 5. `comparison/` (run, compare, store)

```
comparison/
├── canonical_split.json      patient split 80/10/10 train/validation/test (FROZEN, never regenerate)
├── final/                    everything for the final results
│   ├── run.py                runs the 45 final trainings (5 methods x K 10/20/30 x 3 seeds); --plan = dry run
│   ├── pipeline.sh           prints (EXECUTE=1 runs) the steps that build the final inputs from the raw CSVs:
│   │                         events -> stays -> labels -> graph -> k4 (needs MIMIC_RAW_DIR)
│   ├── common_stays.py       stay list incl. test rows (legacy max6, v5 max6, v5 +60 cohorts), used to restrict the labels
│   ├── fit_k4.py             fits the CEI v3 state for one K
│   ├── compare_all.py        regenerates the three comparison tables (K = 10/20/30) from the 45 runs; --check = compare with the stored tables (default: final_compare_fv10t_test_20261019)
│   ├── hpo.py                the light hyper-parameter search (K=17, 10k sample, 20 epochs, dev only)
│   ├── hpo_all.py            recorded HPO of the four tuned methods: commands, best trials vs run.py flags (plan only by default)
│   ├── build_split.py        rebuilds canonical_split.json from merged_ed.csv and compares (never overwrites it)
│   ├── split_check.py        read-only consistency check of the frozen canonical_split.json (never regenerates it)
│   └── graph_hash_check.py   read-only: relates a rebuilt graph to the graph hashes recorded by the final runs
├── top3/
│   └── compare.py            compares runs of one K: ranking (test), validation ranking as the selection column, mean +- SD, paired bootstrap
├── nasim/ ablation/ splits/  analysis packages of sections 2 and 4 (see "Analysis packages and their results")
├── s3_ledger/                evidence-ledger probes (section 3)
├── s6_stats/ s7_why/         statistics (section 6), class support and source masking (section 7)
├── s8_cost/ s9_repro/        model cost (section 8), reproduction scan (section 9)
├── explain/                  explanation driver (run.py: --plan prints one runner call per final run, --execute runs them), summary tables (summarize.py), replay difference tool (replay_diff.py)
└── standardized/             DATA AND RESULTS (git-ignored; disk-free APFS clones of the originals)
    ├── event_inputs/
    │   ├── clinical_graph_L4_plus60_v5c_20261009/       the final graph (graphs.jsonl, manifest.json, ...)
    │   ├── targets_primary_v4_v5plus60_v5_test_20261011/ the final labels (targets.csv, test rows labelled)
    │   ├── common_stays_withtest_v5_20261011/           stay list used for the labels
    │   ├── first_recorded_lab_plus60_max6_v5_20261009/  +60 min event index (input of labels and graph)
    │   ├── v3_state_L4_v5c_v10t_top{10,20,30}_20261011/ CEI v3 state per K (K4.json; must match the run's K)
    │   └── v3_state_L4_v5c_top17_20261011/              v3 state of the HPO (K=17, 10k sample)
    ├── clinical_runs_fair_v10t_20261011/                the 45 final runs
    │   ├── top10/ top20/ top30/                          one folder per K
    │   │   └── <method>_seed<seed>/                     cei_mp2 | protgnn | gchm_pna | gsat | graphcare x 1234 | 2025 | 7
    │   │       ├── binding.json        exact settings, protocol, data and code hashes
    │   │       ├── result.json         selected epoch, validation and TEST metrics
    │   │       ├── history.json        per-epoch metrics
    │   │       ├── validation.npz      validation predictions (selected epoch)
    │   │       ├── test.npz            test predictions (scored once)
    │   │       ├── best.pt             selected checkpoint
    │   │       └── preprocessing.json  vocabulary / scaling fitted on train
    │   └── logs/top<K>_<method>_seed<s>.log              training logs
    ├── hpo_fv10_k17_20261011/                          HPO: per method study.db, trial logs, trials.json, best.json
    ├── final_compare_fv10t_test_20261019/              compare_top{10,20,30}.txt/.json (the result tables, test ranking first)
    └── final_compare_fv10t_20261012/                   the same tables in the earlier layout (validation ranking first), plan checks
```

## 6. Where do I find ...

| I want ... | Look here |
|---|---|
| the final result tables | `results/test_top<K>.md` (aggregates; the full comparison files are in `comparison/standardized/final_compare_fv10t_test_20261019/`) |
| the analysis tables (ablation, splits, statistics, class support, masking, cost) | `results/<package folder>/` (see "Analysis packages and their results") |
| the test score of one run | `comparison/standardized/clinical_runs_fair_v10t_20261011/top<K>/<method>_seed<s>/result.json` (`test` key) |
| the exact command / settings of a method | `comparison/final/run.py`, function `args()` (also under "Final commands" above) |
| how the settings were tuned | `comparison/final/hpo.py`; results in `comparison/standardized/hpo_fv10_k17_20261011/<method>/best.json` |
| the protocol (epochs, selection, test once) | `core/train.py`: `PROTOCOL_NAME`, `FAIR_PROTOCOL` |
| how labels are defined | `data/s3_labels/local_labels_v4.py` |
| how the graph is built | `data/s4_graph/build_delay.py` + `data/s2_events/knowledge_v3.csv`, `knowledge_thresholds_v2.csv` |
| which patients are train / validation / test | `comparison/canonical_split.json` |
| the mp2 model itself | `cei/network.py`, `cei/layers.py`, options in `cei/options.py` |

## 7. Common commands (repo root)

Prefix every command with `env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3`.

| Task | Command | Trains? |
|---|---|---|
| Check that all 15 final plans are valid | `-m comparison.final.run --plan` | no |
| Show the input-building steps | `MIMIC_RAW_DIR=<raw folder> bash comparison/final/pipeline.sh all` (no prefix) | no |
| Compare the runs of one K | `-m comparison.top3.compare --run <run dir> ... --out <new path>` | no |
| Regenerate and check the three comparison tables | `-m comparison.final.compare_all --check comparison/standardized/final_compare_fv10t_test_20261019 --out <new dir>` | no |
| Plan the recorded HPO (prints only) | `-m comparison.final.hpo_all` | no |
| Rebuild the split and compare with the frozen file | `-m comparison.final.build_split` | no |
| Check the frozen split (read only) | `-m comparison.final.split_check` | no |
| Show one training plan | `-m core.train <method args> --top-k-labels <K> --artifact ... --targets ... --output ... --seed 1234` | no |
| Re-run the final trainings | `-m comparison.final.run [--ks 10,20,30]` | yes, needs approval |
| Show the explanation runs (test fold, 500 patients per run) | `-m comparison.explain.run --out <new dir> --plan` | no |
| Explain the final runs (inference only, frozen checkpoints) | `-m comparison.explain.run --out <new dir> --execute`, then `-m comparison.explain.summarize --explanations <that dir> --out <new dir>` | yes, needs approval (long) |
| Explanation flags | driver: `--fold test\|validation`, `--sample-patients N` (0 = every patient), `--resume` (skip finished runs of the same plan and code), `--keep-going`, `--jobs N`, `--allow-mixed-code` (with `--resume`); summary: `--allow-incomplete`, `--allow-mixed-code` | no |

## 8. Rules that matter for the layout

- Frozen (byte-identical): `data/s1_clean/*`, `comparison/canonical_split.json`, `data/s3_labels/icd_mapping.py`, `data/s2_events/{enriched_spec,graph,ingest}.py`, `knowledge_seed.csv`.
- Nothing under `comparison/standardized/` is committed (patient-derived data). Never overwrite an output folder; new runs go to a new dated folder.
- New method: new top-level folder with `__init__.py` + `plugin_<name>.py` (or `adapter.py`), add it to `METHOD_FOLDERS` (`core/registry.py`) and `_METHOD_ROOTS` (`core/paths.py`).
- Decisions (checkpoint, HPO, baseline choice) use validation; every reported result is on test. The test fold is already opened: never tune, select or rerun because of a test score.
