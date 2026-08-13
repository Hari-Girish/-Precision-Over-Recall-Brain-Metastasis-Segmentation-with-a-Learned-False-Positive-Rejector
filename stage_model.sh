#!/usr/bin/env bash
# Stage the five nnU-Net fold checkpoints into model/ before building the image.
# They are not tracked in git (238 MB each, 1.2 GB total -- over GitHub's limits).
#
# Usage: ./stage_model.sh /path/to/nnUNet_results
set -euo pipefail

SRC="${1:?usage: ./stage_model.sh <nnUNet_results_dir>}"
DATASET="Dataset501_BraTS_MET_2026_Training"
CONFIG="nnUNetTrainerDiceTopK10Loss_5000epochs__nnUNetPlans__3d_fullres"
DEST="model/${DATASET}/${CONFIG}"

mkdir -p "$DEST"
for f in 0 1 2 3 4; do
    mkdir -p "${DEST}/fold_${f}"
    cp "${SRC}/${DATASET}/${CONFIG}/fold_${f}/checkpoint_final.pth" "${DEST}/fold_${f}/checkpoint_final.pth"
done
# plans.json / dataset.json are tracked in model/; nnU-Net expects them beside the folds.
cp model/plans.json model/dataset.json "$DEST"/

echo "Staged 5 folds into ${DEST}"
find "$DEST" -name "checkpoint_final.pth" | wc -l | xargs -I{} echo "checkpoints present: {} (expected 5)"
