# Self-Explainable GNN Diagnosis on MIMIC-IV ED Patient Graphs (final)

This repository holds the final benchmark of five self-explainable graph neural networks (CEI-GNN mp2, ProtGNN, GSAT,
GraphCare, GCHM-PNA) for primary-diagnosis prediction on multi-visit clinical graphs built from MIMIC-IV
emergency-department data. Protocol `fair_v10t`: the K = 10 / 20 / 30 most frequent diagnoses, train = the whole TRAIN
fold, checkpoint selection on validation, the held-out test fold scored once per run; 3 seeds per method.
It contains only the code and data behind the final results. The exact commands are listed under "Final commands" below.

## Setup

```bash
conda env create -f environment.yml
conda activate protgnn-mimic
```

Run Python as `env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3` from the repo root.

## Data access

The data is MIMIC-IV-ED, available through PhysioNet credentialed access (https://physionet.org/content/mimic-iv-ed/). You must complete the PhysioNet credentialing and sign the data use agreement yourself. The raw CSVs and everything derived from them (cohorts, graphs, targets, predictions) are not included in this repository.

Two environment variables point the scripts at your copy:

| Variable | Meaning |
|---|---|
| `MEDGNN_SOURCE_ROOT` | directory holding `data/raw/Original CSVs`, the old cohorts and the native class contract (read by `comparison/final/pipeline.sh`) |
| `MEDGNN_DATA_ROOT` | root under which `comparison/standardized/event_inputs` is built and read (default: the repo root) |

Optional: `MEDGNN_OLD_EVENT_INPUTS` and `MEDGNN_V5_EVENT_INPUTS` (event-input directories read by `comparison/final/common_stays.py`) and `MEDGNN_EVENT_INPUTS` (real older event manifests; the tests that need them are skipped when it is unset).

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

`python3 -m comparison.final.run [--ks 10,20,30]` builds these commands for all 45 runs (`--plan` only checks them).

## Results

Test macro-F1, mean ± SD over 3 seeds (methods in validation-ranking order of K = 10). Full tables with accuracy, top-3 accuracy and paired bootstrap comparisons: [K = 10](results/test_top10.md), [K = 20](results/test_top20.md), [K = 30](results/test_top30.md).

| Method | K = 10 | K = 20 | K = 30 |
|---|---|---|---|
| CEI-GNN mp2 | 0.761 ± 0.009 | 0.689 ± 0.001 | 0.578 ± 0.004 |
| ProtGNN | 0.751 ± 0.006 | 0.665 ± 0.002 | 0.557 ± 0.002 |
| GCHM-PNA | 0.756 ± 0.005 | 0.671 ± 0.010 | 0.556 ± 0.009 |
| GSAT | 0.751 ± 0.004 | 0.669 ± 0.004 | 0.557 ± 0.008 |
| GraphCare | 0.763 ± 0.003 | 0.671 ± 0.004 | 0.547 ± 0.002 |

At K = 30, Hypothyroidism has 0 test rows (F1 0 for every method) and End stage renal disease has 5.

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
├── results/        result tables (aggregates only): test_top10.md, test_top20.md, test_top30.md
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
├── s1_clean/          historical cleaning code (FROZEN: never edit, import or run)
├── s2_events/         event index + cohort
│   ├── first_lab.py           first-lab cohort scan (--lab-scope representable-v5)
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

Raw MIMIC-IV ED CSVs are NOT in this repo (see "Data access").

## 5. `comparison/` (run, compare, store)

```
comparison/
├── canonical_split.json      patient split 80/10/10 train/validation/test (FROZEN, never regenerate)
├── final/                    everything for the final results
│   ├── run.py                runs the 45 final trainings (5 methods x K 10/20/30 x 3 seeds); --plan = dry run
│   ├── pipeline.sh           prints (EXECUTE=1 runs) the steps that built the final inputs:
│   │                         events -> stays -> labels -> graph -> k4
│   ├── common_stays.py       stay list incl. test rows, used to restrict the labels
│   ├── fit_k4.py             fits the CEI v3 state for one K
│   └── hpo.py                the light hyper-parameter search (K=17, 10k sample, 20 epochs, dev only)
├── top3/
│   └── compare.py            compares runs of one K: ranking (validation), test mean +- SD, paired bootstrap
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
    └── final_compare_fv10t_20261012/                   compare_top{10,20,30}.txt/.json (the result tables), plan checks
```

## 6. Where do I find ...

| I want ... | Look here |
|---|---|
| the final result tables | `results/test_top<K>.md` (aggregates; the full comparison files are written to `comparison/standardized/final_compare_fv10t_20261012/`) |
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
| Show the input-building steps | `bash comparison/final/pipeline.sh all` (no prefix) | no |
| Compare the runs of one K | `-m comparison.top3.compare --run <run dir> ... --out <new path>` | no |
| Show one training plan | `-m core.train <method args> --top-k-labels <K> --artifact ... --targets ... --output ... --seed 1234` | no |
| Re-run the final trainings | `-m comparison.final.run [--ks 10,20,30]` | yes, needs approval |

## 8. Rules that matter for the layout

- Frozen (byte-identical): `data/s1_clean/*`, `comparison/canonical_split.json`, `data/s3_labels/icd_mapping.py`, `data/s2_events/{enriched_spec,graph,ingest}.py`, `knowledge_seed.csv`.
- Nothing under `comparison/standardized/` is committed (patient-derived data). Never overwrite an output folder; new runs go to a new dated folder.
- New method: new top-level folder with `__init__.py` + `plugin_<name>.py` (or `adapter.py`), add it to `METHOD_FOLDERS` (`core/registry.py`) and `_METHOD_ROOTS` (`core/paths.py`).
- The test fold is already opened: never tune or rerun because of a test score.
