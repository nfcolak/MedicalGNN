#!/bin/bash
# Final-input pipeline: raw MIMIC-IV-ED / MIMIC-IV hosp CSVs -> the inputs of the final trainings. Prints the commands;
# EXECUTE=1 runs them.
# Inputs: MIMIC_RAW_DIR (required) = ONE folder with the decompressed raw CSVs (see README, "Reproduce the data from raw");
#         MEDGNN_DATA_ROOT (default: the repo root) = where comparison/standardized/event_inputs is written and read.
# Stages (in order): events  -> legacy event indexes (all visits, max6) + v5 event indexes (all visits, max6, +60 min)
#                    stays   -> common stays of the legacy max6, v5 max6 and v5 +60 cohorts INCLUDING test rows
#                               (comparison/final/common_stays.py)
#                    labels  -> primary_dx_v4 targets with test labels (local_labels_v4 --restrict-stays)
#                    graph   -> L4 graph (+60 min, knowledge_v3, thresholds_v2, row-range levels)
#                    k4      -> CEI v3 state per K (fit_k4 on the fair_v10t train ids)
# Usage: MIMIC_RAW_DIR=... [MEDGNN_DATA_ROOT=...] bash comparison/final/pipeline.sh events|stays|labels|graph|k4|all
#        (EXECUTE=1 to run; outputs must not exist). Never overwrite an output directory.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
EXECUTE=${EXECUTE:-0}
ROOT=${MEDGNN_DATA_ROOT:-$REPO}
EI=$ROOT/comparison/standardized/event_inputs
RAW=${MIMIC_RAW_DIR:?set MIMIC_RAW_DIR to the folder holding the decompressed raw MIMIC CSVs}
RAW=$(cd "$RAW" && pwd)
CAN=$REPO/comparison/canonical_split.json
KSEED=$REPO/data/s2_events/knowledge_seed.csv
KV3=$REPO/data/s2_events/knowledge_v3.csv
THR2=$REPO/data/s2_events/knowledge_thresholds_v2.csv
SUF=v5_20261009
EV_LEGACY_ALL=$EI/first_recorded_lab_all_visits_v2
EV_LEGACY=$EI/first_recorded_lab_all_visits_v2_max6
EV_ALL=$EI/first_recorded_lab_all_visits_uncapped_$SUF
EV_A=$EI/first_recorded_lab_all_visits_max6_$SUF
EV_P=$EI/first_recorded_lab_plus60_max6_$SUF
STAYS=$EI/common_stays_withtest_v5_20261011
TARGETS=$EI/targets_primary_v4_v5plus60_v5_test_20261011
GRAPH=$EI/clinical_graph_L4_plus60_v5c_20261009
PY=(env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "${PYTHON:-python3}")

X() { if [ "$EXECUTE" = 1 ]; then echo "+ $*"; "$@"; else printf '+ '; printf '%q ' "$@"; echo; fi; }
need_raw() { for f in edstays diagnosis triage patients labevents; do [ -f "$RAW/$f.csv" ] || { echo "missing raw file: $RAW/$f.csv" >&2; exit 1; }; done; }
absent() { for p in "$@"; do [ -e "$p" ] && { echo "output exists, refusing: $p" >&2; [ "$EXECUTE" = 1 ] && exit 1; }; done; true; }

stage_events() {
  need_raw
  absent "$EV_LEGACY_ALL" "$EV_LEGACY" "$EV_ALL" "$EV_A" "$EV_P"
  # legacy cohort (all-numeric-known-unit labs): uncapped index, then the max-6-visit index
  X "${PY[@]}" -m data.s2_events.first_lab --raw-root "$RAW" --canonical "$CAN" --knowledge "$KSEED" \
      --output "$EV_LEGACY_ALL" --lab-scope all-numeric-known-unit --execute
  X "${PY[@]}" -m data.s2_events.first_lab --raw-root "$RAW" --canonical "$CAN" --knowledge "$KSEED" \
      --output "$EV_LEGACY" --lab-scope all-numeric-known-unit --max-visits-per-subject 6 --execute
  # v5 cohorts
  X "${PY[@]}" -m data.s2_events.first_lab --raw-root "$RAW" --canonical "$CAN" --knowledge "$KSEED" \
      --output "$EV_ALL" --lab-scope representable-v5 --execute
  X "${PY[@]}" -m data.s2_events.first_lab --raw-root "$RAW" --canonical "$CAN" --knowledge "$KSEED" \
      --output "$EV_A" --lab-scope representable-v5 --max-visits-per-subject 6 --execute
  # first_lab also writes graphs.jsonl; repair_metadata refuses a directory that has one (move aside, never delete)
  X bash -c 'for d in "$@"; do [ -e "$d/graphs.jsonl" ] && mv -n "$d/graphs.jsonl" "$d/graphs.event_v1.unused.jsonl"; done; true' _ "$EV_LEGACY_ALL" "$EV_LEGACY" "$EV_ALL" "$EV_A"
  X "${PY[@]}" -m data.s2_events.repair_metadata --artifact "$EV_LEGACY_ALL" --execute --offline
  X "${PY[@]}" -m data.s2_events.repair_metadata --artifact "$EV_LEGACY" --source-artifact "$EV_LEGACY_ALL" --max-visits-per-subject 6 --execute --offline
  X "${PY[@]}" -m data.s2_events.repair_metadata --artifact "$EV_ALL" --execute --offline
  X "${PY[@]}" -m data.s2_events.repair_metadata --artifact "$EV_A" --source-artifact "$EV_ALL" --max-visits-per-subject 6 --execute --offline
  X "${PY[@]}" -m data.s2_events.first_lab_delay --inherit-from "$EV_A" --delay-minutes 60 --output "$EV_P" --execute
}

stage_stays() {
  absent "$STAYS"
  X "${PY[@]}" -m comparison.final.common_stays "$STAYS" --legacy-max6 "$EV_LEGACY" --v5-max6 "$EV_A" --v5-plus60 "$EV_P"
}

stage_labels() {
  need_raw
  absent "$TARGETS"
  X "${PY[@]}" -m data.s3_labels.local_labels_v4 --output "$TARGETS" --raw-root "$RAW" --graph-root "$EV_P" \
      --restrict-stays "$STAYS/stays.txt"
}

stage_graph() {
  need_raw
  absent "$GRAPH"
  X "${PY[@]}" -m data.s4_graph.build_delay --inherit-from "$EV_P" --raw-root "$RAW" --canonical "$CAN" --knowledge "$KV3" \
      --output "$GRAPH" --with-diagnosis --diagnosis-policy v5 --age-policy visit --thresholds "$THR2" --level-source row-range --execute
}

stage_k4() {
  for K in 10 20 30; do
    out=$EI/v3_state_L4_v5c_v10t_top${K}_20261011
    absent "$out"
    X "${PY[@]}" -m comparison.final.fit_k4 --artifact "$GRAPH" --targets "$TARGETS/targets.csv" \
        --output-root "$ROOT/comparison/standardized/k4_fit/top$K" --top-k-labels "$K" --execute
    X mkdir "$out"
    X cp "$ROOT/comparison/standardized/k4_fit/top$K/v3_state/K4.json" "$out/K4.json"
  done
}

case "${1:-}" in
  events) stage_events;; stays) stage_stays;; labels) stage_labels;; graph) stage_graph;; k4) stage_k4;;
  all) stage_events; stage_stays; stage_labels; stage_graph; stage_k4;;
  *) echo "usage: $0 events|stays|labels|graph|k4|all   (EXECUTE=1 to run)" >&2; exit 2;;
esac
