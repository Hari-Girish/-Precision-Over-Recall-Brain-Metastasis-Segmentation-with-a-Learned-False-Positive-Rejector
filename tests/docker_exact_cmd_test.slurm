#!/bin/bash
##############################################################################
# Run the container with the EXACT command from the challenge's "Test Your Model
# Locally" instructions (Option B), i.e. WITHOUT the extra `--volume ...:/work`
# mount that our earlier 179-case test used.
#
# Why this matters: with no /work bind mount, every intermediate softmax .npz
# (~0.4 GB/case, ~86 GB over 179 cases) is written into the CONTAINER'S WRITABLE
# LAYER instead of a host filesystem. Some storage drivers cap that layer
# (devicemapper defaults to 10 GB), which would fail the run with ENOSPC. This
# test proves the no-/work invocation works at all, and reports how much the
# writable layer actually grew.
#
# Short walltime + 3 cases so it backfills quickly (deadline today 16:18).
##############################################################################
#SBATCH --job-name=docker_exact_cmd
#SBATCH --account=PDE0022
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:40:00
#SBATCH --output=/fs/scratch/PDE0022/girishosu/docker_task1/exact_cmd_%j.log
#SBATCH --error=/fs/scratch/PDE0022/girishosu/docker_task1/exact_cmd_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=girish.17@buckeyemailosu.edu

set -euo pipefail
ROOT=/fs/scratch/PDE0022/girishosu/docker_task1
IMG=docker.synapse.org/syn74793974/brats2026-task1:v1

echo "=== node: $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "=== load image + tag as the pushed Synapse reference ==="
podman load -i "$ROOT/brats2026-task1-dev.tar"
podman tag brats2026-task1:dev "$IMG"

rm -rf "$ROOT/output_exact"; mkdir -p "$ROOT/output_exact"

echo "=== container storage BEFORE ==="
podman system df 2>/dev/null | tail -5

echo "=== EXACT challenge command (Option B) — note: NO /work mount ==="
/usr/bin/time -v podman run \
  --rm \
  --network none \
  --gpus=all \
  --volume "$ROOT/input_exact":/input:ro \
  --volume "$ROOT/output_exact":/output:rw \
  --memory=48G --shm-size=16G \
  "$IMG"

echo "=== container storage AFTER (writable-layer growth) ==="
podman system df 2>/dev/null | tail -5

echo "=== output check ==="
N=$(find "$ROOT/output_exact" -name "*.nii.gz" -type f | wc -l)
D=$(find "$ROOT/output_exact" -mindepth 1 -type d | wc -l)
echo "flat .nii.gz: $N (expected 3)"
echo "subdirs:      $D (MUST be 0)"
ls -la "$ROOT/output_exact"
[ "$N" -eq 3 ] && [ "$D" -eq 0 ] && echo "PASS: exact challenge command works with no /work mount." || { echo "FAIL"; exit 1; }
echo "=== done $(date) ==="
