#!/bin/bash
##############################################################################
# Full 179-case scale/timing test of the BraTS-METS 2026 Task-1 submission
# container (M_base 5-fold nnU-Net + M_FPR ET false-positive rejector).
#
# Purpose: verify the container finishes the whole validation set inside the
# platform's 12-hour / 48GiB / 16GiB-shm budget, and that /output ends up FLAT
# (179 x <case>.nii.gz, no subdirectories -- subfolders invalidate the submission).
#
# NOTE: an earlier attempt was launched with `srun` from an interactive
# background shell; when that shell was torn down, srun forwarded the signal and
# killed the step at 50 min (fold 4, case 64/179) with 4 folds already complete.
# That was a launcher problem, not a container problem -- hence sbatch here, which
# detaches from any shell.
#
# /work is bind-mounted to persistent scratch and nnUNetv2_predict runs with
# --continue_prediction, so this run RESUMES: folds 0-3 (179/179 each) are skipped
# and only the remainder of fold 4 is computed before ensembling + rejection.
##############################################################################
#SBATCH --job-name=docker_task1_full
#SBATCH --account=PDE0022
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/fs/scratch/PDE0022/girishosu/docker_task1/full_test_%j.log
#SBATCH --error=/fs/scratch/PDE0022/girishosu/docker_task1/full_test_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=girish.17@buckeyemailosu.edu

set -euo pipefail

ROOT=/fs/scratch/PDE0022/girishosu/docker_task1

echo "=== node: $(hostname) | start: $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

# podman's image store is node-local (/tmp/containers), so the image must be
# loaded from the shared-scratch tarball on whichever node we land on.
echo "=== loading image ==="
podman load -i "$ROOT/brats2026-task1-dev.tar"

echo "=== FULL 179-case run (resuming; folds 0-3 already complete) ==="
# Flags mirror the challenge platform's documented run command exactly.
/usr/bin/time -v podman run --rm --network none --gpus=all \
  --memory=48g --shm-size=16g \
  --volume "$ROOT/input_full":/input:ro \
  --volume "$ROOT/output_full":/output:rw \
  --volume "$ROOT/work_full":/work:rw \
  brats2026-task1:dev

echo "=== output structure check ==="
N_FILES=$(find "$ROOT/output_full" -name "*.nii.gz" -type f | wc -l)
N_SUBDIRS=$(find "$ROOT/output_full" -mindepth 1 -type d | wc -l)
echo "flat .nii.gz files: $N_FILES (expected 179)"
echo "subdirectories:     $N_SUBDIRS (MUST be 0)"

if [ "$N_FILES" -ne 179 ] || [ "$N_SUBDIRS" -ne 0 ]; then
    echo "FAIL: output structure does not satisfy the submission spec."
    exit 1
fi
echo "PASS: 179 flat masks, no subdirectories."
echo "=== done: $(date) ==="
