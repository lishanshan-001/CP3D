#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "Usage: $0 PYTHON STEP3_TRAIN STEP3_TEST DATA_ROOT RECTIFICATION_ROOT OUTPUT_ROOT" >&2
  exit 2
fi

PYTHON=$1
STEP3_TRAIN=$2
STEP3_TEST=$3
DATA_ROOT=$4
RECT_ROOT=$5
OUTPUT_ROOT=$6

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SRC="$REPO/src"
CONFIG="$REPO/configs"
FVSA_OUTPUT="$OUTPUT_ROOT/step4_association"
HRTR_OUTPUT="$OUTPUT_ROOT/step5_reconstruction"

echo "[1/2] Running FVSA with frozen paper parameters"
"$PYTHON" "$SRC/step4_stereo_association_eval.py" \
  --step3-train-root "$STEP3_TRAIN" \
  --step3-test-root "$STEP3_TEST" \
  --data-root "$DATA_ROOT" \
  --rect-root "$RECT_ROOT" \
  --save-root "$FVSA_OUTPUT" \
  --methods Proposed \
  --fixed-params-json "$CONFIG/fvsa_fixed_parameters.json"

echo "[2/2] Running HRTR with frozen paper parameters"
"$PYTHON" "$SRC/step5_reliability_guided_3d_reconstruction.py" \
  --step4-root "$FVSA_OUTPUT" \
  --save-root "$HRTR_OUTPUT" \
  --fixed-step5-metadata "$CONFIG/hrtr_fixed_parameters.json"

echo "FVSA-HRTR outputs: $OUTPUT_ROOT"
