# Precision Over Recall — BraTS-METS 2026 Task 1 (Team Buckeye)

Submission code for the MICCAI **BraTS-METS 2026** challenge, Task 1 (Brain Metastases
Segmentation), from The Ohio State University.

The method is a two-stage pipeline:

- **Stage 1 — `M_base`:** an ensemble of five nnU-Net cross-validation fold models
  (`3d_fullres`, `PlainConvUNet`), trained for **5000 epochs** with a
  **Dice + TopK-10% cross-entropy** loss, combined by averaging softmax probabilities.
- **Stage 2 — `M_FPR`:** a learned **connected-component false-positive rejector**
  applied *only* to the enhancing-tumor (ET) channel. Each predicted ET component is
  described by 15 confidence / shape / intensity features and scored by a
  gradient-boosted classifier; components below the operating threshold are relabeled
  to background.

The rejector is trained **exclusively on out-of-fold predictions** of the 1296 training
studies, under grouped-by-case cross-validation. No challenge-validation annotations
were used at any point, and the operating threshold was fixed before submission.

## Results (179-case official validation set, lesion-wise DSC)

| Model | ET | TC | WT | RC |
|---|---|---|---|---|
| `M_base` (Stage 1 only) | 0.6964 | 0.7236 | 0.6827 | 0.5073 |
| **`M_FPR`** (Stage 1 + 2) | **0.7187** | **0.7429** | **0.6950** | 0.5073 |
| Δ | +0.0224 | +0.0193 | +0.0123 | ±0.0000 |

The rejector removes **110 of 1039** ET components (10.6%) across the 179 cases,
cutting small-ET false positives per case by 49% (1.79 → 0.91) at a cost of 0.21 true
positives per case.

RC is unchanged **by construction**: the rejector only edits ET, and ET ⊆ TC ⊆ WT, so
the resection cavity is never touched.

**Reported honestly:** the same operation *reduces* small-instance F1 (ET
0.4886 → 0.4717). Lesion-wise DSC and NSD improve while instance F1 declines, because
lesion-wise DSC penalizes an unmatched component through its zero-Dice contribution
whereas F1 weights missed and spurious instances comparably. Both directions are
reported in the paper rather than only the favorable one.

## Repository layout

```
Dockerfile              container definition (challenge submission image)
requirements.txt        exact pinned dependency versions
stage_model.sh          copies the 5 fold checkpoints into model/ (not in git)
src/
  run_inference.py      container entrypoint: /input -> /output
  apply_rejector.py     Stage 2, applies the rejector to ensembled masks
  build_components.py   component extraction + 15-feature construction
trainers/
  nnUNetTrainerDiceTopK10Loss_5000epochs.py   custom nnU-Net trainer
model/
  rejector_R37b_leakfree.joblib   trained Stage-2 classifier (390 KB)
  plans.json, dataset.json        nnU-Net configuration
training/
  train_all_folds.slurm   Stage 1: 5-fold training
  find_best.slurm         nnU-Net configuration selection
  predict.slurm           5-fold inference + ensembling
  train_rejector.py       Stage 2: fits the rejector on out-of-fold predictions
  build_components.py     shared feature construction
tests/
  docker_full_test.slurm       full 179-case scale/timing test
  docker_exact_cmd_test.slurm  the challenge's documented run command, verbatim
```

**Model weights are not tracked in git.** The five `checkpoint_final.pth` files are
238 MB each (1.2 GB total), well over GitHub's per-file limit. Stage them with
`./stage_model.sh <nnUNet_results_dir>` before building the image.

## Quick start

### Option A — pull the published image

```bash
docker pull docker.synapse.org/syn74793974/brats2026-task1:v1
```

### Option B — build locally

```bash
./stage_model.sh /path/to/nnUNet_results     # copies the 5 fold checkpoints
docker build -t brats2026-task1:dev .
```

### Run

```bash
docker run --rm --network none --gpus=all \
  --volume /PATH/TO/INPUT:/input:ro \
  --volume /PATH/TO/OUTPUT:/output:rw \
  --memory=48G --shm-size=16G \
  docker.synapse.org/syn74793974/brats2026-task1:v1
```

**Input contract:** `/input` contains one folder per case; each holds
`<case>-t1c.nii.gz`, `<case>-t1n.nii.gz`, `<case>-t2f.nii.gz`, `<case>-t2w.nii.gz`.

**Output contract:** `/output` receives a **flat** `<case>.nii.gz` per case, labels
`{0,1,2,3,4}`, `uint8`, matching the input geometry. The entrypoint asserts the output
count and the absence of subdirectories before exiting.

## Compute

Verified on the full 179-case validation set within the challenge's stated limits
(NVIDIA A10G 24 GB, 16 vCPU, 200 GB storage, ≤48 GiB memory, ≤16 GiB shm, 12-hour
total inference budget):

| Limit | Measured |
|---|---|
| 12-hour inference budget | ~2.8 h estimated cold-start (5 folds + ensemble + rejector) |
| ≤48 GiB memory | **36.2 GiB peak** |
| 200 GB storage | 86 GB (mostly transient per-fold softmax `.npz`) |
| Output structure | 179 flat masks, 0 subdirectories |

Testing was performed on NVIDIA A100 (40 GB) — no A10G partition was available on our
cluster — so the 24 GB VRAM fit was not directly measured. nnU-Net falls back to CPU
accumulation on GPU OOM, so the expected failure mode there is slower inference rather
than a crash.

Container output was compared voxel-by-voxel against the scored submission produced by
the original SLURM pipeline: **103/179 cases bit-identical**, with 239 differing voxels
out of 5,585,782,784 total (0.0000043%), attributable to GPU kernel non-determinism.
Geometry and affine match on all 179.

## Training reproduction

Stage 1 (5-fold nnU-Net) and Stage 2 (rejector) are trained separately:

```bash
sbatch training/train_all_folds.slurm    # Stage 1, 5 folds
sbatch training/find_best.slurm          # configuration selection
sbatch training/predict.slurm            # 5-fold inference + ensembling
python  training/train_rejector.py       # Stage 2, fits on out-of-fold predictions
```

The SLURM scripts target the Ohio Supercomputer Center and will need site-specific
paths and account settings adjusted.

Stage 2 details: `GradientBoostingClassifier` (300 estimators, learning rate 0.05, max
depth 3, subsample 0.8), grouped 5-fold CV AUC **0.86**, operating threshold **0.424**
selected by a lesion-wise Dice cost–benefit analysis rather than by classification
accuracy.

> The rejector must be loaded under the same scikit-learn version it was fitted with
> (1.7.2, pinned in `requirements.txt`); serialized estimators are not guaranteed to
> load correctly across versions.

## Citation

> Girish, H., Kong, M., Chan, S., Patil, N., Chan, J., Chakravarti, A., Zhu, S.
> *Precision Over Recall: Brain Metastasis Segmentation with a Learned False-Positive
> Rejector.* MICCAI BraTS-METS 2026 Challenge.

## License

MIT — see [LICENSE](LICENSE).
