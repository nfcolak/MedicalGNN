#!/bin/bash
# Per-split input pipeline: rebuilds the inputs that depend on the fold assignment for ONE new patient split r<seed>
# (same rule as the canonical split, another rng seed). Prints the commands; EXECUTE=1 runs them (needs approval).
# Minimal stages: the v5 branch of the event indexes, the labels,
# the L4 graph and the CEI v3 state for K=20. The legacy cohorts and the common-stays stage are skipped: the stay set does
# not depend on the fold, so the existing stays.txt is reused after a sha256 check.
# Inputs: MIMIC_RAW_DIR (required) = ONE folder with the decompressed raw CSVs (see README, "Reproduce the data from raw");
#         MEDGNN_DATA_ROOT (default: the repo root) = where comparison/standardized/event_inputs is written and read;
#         DATE (default 20261018) = date tag of every new folder; PYTHON (default python3).
# Stages (in order): split  -> comparison.final.build_split --seed N --out <split file>
#                    events -> v5 uncapped + max-6 event indexes with the new split as --canonical, repairs, +60 min index
#                    labels -> primary_dx_v4 targets with test labels (local_labels_v4 --restrict-stays, existing stays.txt)
#                    graph  -> L4 graph (+60 min, knowledge_v3, thresholds_v2, row-range levels)
#                    k4     -> CEI v3 state for K=20 only (fit_k4 on the new TRAIN ids)
# Usage: MIMIC_RAW_DIR=... [DATE=...] bash comparison/splits/pipeline.sh split|events|labels|graph|k4|all SEED
#        (EXECUTE=1 to run; outputs must not exist). Split id is r<SEED>; SEED 1234 is the canonical split (r0), refused.
#        Never overwrite an output directory.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
EXECUTE=${EXECUTE:-0}
ROOT=${MEDGNN_DATA_ROOT:-$REPO}
STAGE=${1:-}
SEED=${2:-}
case "$STAGE" in split|events|labels|graph|k4|all) ;;
  *) echo "usage: $0 split|events|labels|graph|k4|all SEED   (EXECUTE=1 to run; MIMIC_RAW_DIR=... for all but split)" >&2; exit 2;; esac
case "$SEED" in ''|*[!0-9]*) echo "SEED must be a non-negative integer (split id r<SEED>)" >&2; exit 2;; esac
[ "$SEED" = 1234 ] && { echo "refusing: seed 1234 is the canonical split (r0), never rebuilt here" >&2; exit 2; }
SID=r$SEED
TAG=${SID}_${DATE:-20261018}
EI=$ROOT/comparison/standardized/event_inputs
if [ "$STAGE" = split ]; then RAW=; else
  RAW=${MIMIC_RAW_DIR:?set MIMIC_RAW_DIR to the folder holding the decompressed raw MIMIC CSVs}
  RAW=$(cd "$RAW" && pwd)
fi
MERGED=$EI/split_input_merged_ed/merged_ed.csv
SPLIT=$EI/split_$TAG/split.json
KSEED=$REPO/data/s2_events/knowledge_seed.csv
KV3=$REPO/data/s2_events/knowledge_v3.csv
THR2=$REPO/data/s2_events/knowledge_thresholds_v2.csv
STAYS=$EI/common_stays_withtest_v5_20261011/stays.txt
STAYS_SHA256=2d43991be35bb6ba88b867a847b64a97642766504546063ff1acaaa1c42c13d9
EV_ALL=$EI/first_recorded_lab_all_visits_uncapped_v5_$TAG
EV_A=$EI/first_recorded_lab_all_visits_max6_v5_$TAG
EV_P=$EI/first_recorded_lab_plus60_max6_v5_$TAG
TARGETS=$EI/targets_primary_v4_v5plus60_v5_test_$TAG
GRAPH=$EI/clinical_graph_L4_plus60_v5c_$TAG
K4FIT=$ROOT/comparison/standardized/k4_fit/$TAG/top20
K4OUT=$EI/v3_state_L4_v5c_v10t_top20_$TAG
PY=(env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "${PYTHON:-python3}")

X() { if [ "$EXECUTE" = 1 ]; then echo "+ $*"; "$@"; else printf '+ '; printf '%q ' "$@"; echo; fi; }
need_raw() { for f in edstays diagnosis triage patients labevents; do [ -f "$RAW/$f.csv" ] || { echo "missing raw file: $RAW/$f.csv" >&2; exit 1; }; done; }
absent() { for p in "$@"; do [ -e "$p" ] && { echo "output exists, refusing: $p" >&2; [ "$EXECUTE" = 1 ] && exit 1; }; done; true; }
# an input produced by an earlier stage: in plan mode it may not exist yet (say so), with EXECUTE=1 it must
need_in() { for p in "$@"; do [ -e "$p" ] || { echo "input missing: $p" >&2; [ "$EXECUTE" = 1 ] && exit 1; }; done; true; }
check_stays() {
  [ -f "$STAYS" ] || { echo "missing $STAYS (copy of common_stays_withtest_v5_20261011)" >&2; exit 1; }
  [ "$(shasum -a 256 "$STAYS" | cut -d' ' -f1)" = "$STAYS_SHA256" ] || { echo "stays.txt sha256 differs from the README value, refusing: $STAYS" >&2; exit 1; }
}

echo "# split $SID (seed $SEED), tag $TAG"

stage_split() {
  [ -f "$MERGED" ] || { echo "missing split input: $MERGED" >&2; exit 1; }
  absent "$SPLIT" "$SPLIT.provenance.json"
  X "${PY[@]}" -m comparison.final.build_split --merged-ed "$MERGED" --seed "$SEED" --out "$SPLIT"
}

stage_events() {
  need_raw
  need_in "$SPLIT"
  absent "$EV_ALL" "$EV_A" "$EV_P"
  # v5 cohorts: uncapped index, then the max-6-visit index; both with the new split as --canonical (same hash, repair_metadata checks it)
  X "${PY[@]}" -m data.s2_events.first_lab --raw-root "$RAW" --canonical "$SPLIT" --knowledge "$KSEED" \
      --output "$EV_ALL" --lab-scope representable-v5 --execute
  X "${PY[@]}" -m data.s2_events.first_lab --raw-root "$RAW" --canonical "$SPLIT" --knowledge "$KSEED" \
      --output "$EV_A" --lab-scope representable-v5 --max-visits-per-subject 6 --execute
  # first_lab also writes graphs.jsonl; repair_metadata refuses a directory that has one (move aside, never delete)
  X bash -c 'for d in "$@"; do [ -e "$d/graphs.jsonl" ] && mv -n "$d/graphs.jsonl" "$d/graphs.event_v1.unused.jsonl"; done; true' _ "$EV_ALL" "$EV_A"
  X "${PY[@]}" -m data.s2_events.repair_metadata --artifact "$EV_ALL" --execute --offline
  X "${PY[@]}" -m data.s2_events.repair_metadata --artifact "$EV_A" --source-artifact "$EV_ALL" --max-visits-per-subject 6 --execute --offline
  X "${PY[@]}" -m data.s2_events.first_lab_delay --inherit-from "$EV_A" --delay-minutes 60 --output "$EV_P" --execute
}

stage_labels() {
  need_raw
  check_stays
  need_in "$EV_P"
  absent "$TARGETS"
  # the pinned Top-10 assertion of local_labels_v4 is unchanged: if it fires, stop and ask
  X "${PY[@]}" -m data.s3_labels.local_labels_v4 --output "$TARGETS" --raw-root "$RAW" --graph-root "$EV_P" \
      --restrict-stays "$STAYS"
}

stage_graph() {
  need_raw
  need_in "$SPLIT" "$EV_P"
  absent "$GRAPH"
  X "${PY[@]}" -m data.s4_graph.build_delay --inherit-from "$EV_P" --raw-root "$RAW" --canonical "$SPLIT" --knowledge "$KV3" \
      --output "$GRAPH" --with-diagnosis --diagnosis-policy v5 --age-policy visit --thresholds "$THR2" --level-source row-range --execute
}

stage_k4() {
  need_in "$GRAPH" "$TARGETS/targets.csv"
  absent "$K4FIT" "$K4OUT"
  X "${PY[@]}" -m comparison.final.fit_k4 --artifact "$GRAPH" --targets "$TARGETS/targets.csv" \
      --output-root "$K4FIT" --top-k-labels 20 --execute
  X mkdir "$K4OUT"
  X cp "$K4FIT/v3_state/K4.json" "$K4OUT/K4.json"
}

case "$STAGE" in
  split) stage_split;; events) stage_events;; labels) stage_labels;; graph) stage_graph;; k4) stage_k4;;
  all) stage_split; stage_events; stage_labels; stage_graph; stage_k4;;
esac
