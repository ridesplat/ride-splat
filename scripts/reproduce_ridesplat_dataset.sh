#!/usr/bin/env bash
# RIDE-Splat on the three RIDE-Splat Dataset scenes, then a table next to the paper's numbers.
# About 1.5 GPU-hours per scene on an A100.
#
#   bash scripts/reproduce_ridesplat_dataset.sh [data root] [output root]
set -euo pipefail
DATA="${1:-data/ride-splat-dataset}"
OUT="${2:-output}"
for scene in rs1 rs2 rs3; do
  python run_ridesplat.py -s "$DATA/$scene" -m "$OUT/$scene" --gt "$DATA/$scene/eval"
done
python scripts/summarize.py "$OUT"
