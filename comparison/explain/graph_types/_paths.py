"""Paths of the graph-type analysis (steps 1-6), relative to the repository by default.

  MEDGNN_DATA_ROOT     folder that holds comparison/standardized (default: the repo root)
  MEDGNN_EXPLAIN_ROOT  folder that holds the explanation output folders read by step 1 (default:
                       <data root>/comparison/standardized/explain_k20)
  MEDGNN_GRAPH_TYPES   working folder of steps 1-6: data/ (per-visit tables, anonymous indices; never commit) and out/
                       (aggregate tables) (default: <data root>/comparison/standardized/graph_types_k20)
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STD = Path(os.environ.get('MEDGNN_DATA_ROOT', str(REPO))) / 'comparison/standardized'
EXPLAIN = Path(os.environ.get('MEDGNN_EXPLAIN_ROOT', str(STD / 'explain_k20')))
WORK = Path(os.environ.get('MEDGNN_GRAPH_TYPES', str(STD / 'graph_types_k20')))
