#!/bin/bash
#SBATCH --job-name=R37b_5000ep_train
#SBATCH --account=PDE0022
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=1
#SBATCH --time=168:00:00
#SBATCH --array=0-4
#SBATCH --output=/fs/ess/PDE0022/hari/brats2026/pipelines/nnunet/logs/R37b_dicetopk_5000ep_real/training/train_fold%a_%j.log
#SBATCH --error=/fs/ess/PDE0022/hari/brats2026/pipelines/nnunet/logs/R37b_dicetopk_5000ep_real/training/train_fold%a_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=girish.17@buckeyemailosu.edu

##############################################################################
# R37b — Dice+TopK, 5000 epochs (REAL), 5-Fold Stratified CV
#
# RERUN of R37 with the num_epochs bug FIXED. The original R37 silently ran
# 1000ep (class-attr num_epochs shadowed by parent __init__). The fixed trainer
# overrides self.num_epochs=5000 AFTER super().__init__. This is the FIRST
# genuine 5000-epoch run — tests whether long training helps ET/RC at all, and
# whether it overfits.
#
# Trainer:  nnUNetTrainerDiceTopK10Loss_5000epochs (FIXED 2026-06-21)
# Planner:  nnUNetPlans (PlainConvUNet) | Splits: stratified (R5_vanilla)
# Epoch time ~43.5s (PlainConv) → 5000ep ≈ 60.4h/fold, fits 168h wall.
# --c kept as a safety net in case of preemption.
##############################################################################

echo "=============================================="
echo "R37b — Dice+TopK 5000ep REAL (5-fold stratified)"
echo "Started at: $(date) | Job: $SLURM_JOB_ID | Fold: $SLURM_ARRAY_TASK_ID | Node: $SLURM_NODELIST"
echo "=============================================="

module load python/3.10
module load cuda/12.4.1

cd /fs/ess/PDE0022/hari/brats2026/pipelines/nnunet
set +u
source .venv/bin/activate
set -u

export nnUNet_raw="/fs/ess/PDE0022/hari/brats2026/pipelines/nnunet/data/raw"
export nnUNet_preprocessed="/fs/ess/PDE0022/hari/brats2026/pipelines/nnunet/data/preprocessed/R5_vanilla"
export nnUNet_results="/fs/ess/PDE0022/hari/brats2026/pipelines/nnunet/data/results/R37b_dicetopk_5000ep_real"
export nnUNet_compile=false

DATASET_ID=501
CONFIGURATION="3d_fullres"
FOLD=$SLURM_ARRAY_TASK_ID
TRAINER="nnUNetTrainerDiceTopK10Loss_5000epochs"
PLANS="nnUNetPlans"

# ── Copy FIXED custom trainer into venv and verify num_epochs==5000 ───────────
VARIANTS_DIR=$(python -c "import nnunetv2, os; print(os.path.join(os.path.dirname(nnunetv2.__file__), 'training/nnUNetTrainer/variants/loss'))")
cp /fs/ess/PDE0022/hari/brats2026/pipelines/nnunet/custom_trainers/nnUNetTrainerDiceTopK10Loss_5000epochs.py "$VARIANTS_DIR/"
python -c "
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import nnunetv2.training.nnUNetTrainer.variants as v
cls = recursive_find_python_class(v.__path__[0], 'nnUNetTrainerDiceTopK10Loss_5000epochs', 'nnunetv2.training.nnUNetTrainer.variants')
assert cls is not None, 'ERROR: trainer not found'
# verify the bugfix: num_epochs must NOT be a stale class attr of 5000 that gets shadowed
import inspect
src = inspect.getsource(cls)
assert 'self.num_epochs = 5000' in src, 'ERROR: num_epochs not set in __init__ — bug not fixed!'
print(f'Trainer found + bugfix verified (self.num_epochs=5000 in __init__): {cls}')
"
[ $? -ne 0 ] && echo "Trainer verification FAILED — aborting" && exit 1

nvidia-smi
echo ""
echo "Trainer: $TRAINER | Plans: $PLANS | Fold: $FOLD | Epochs: 5000 (REAL)"
echo ""

nnUNetv2_train $DATASET_ID $CONFIGURATION $FOLD \
    -tr $TRAINER \
    -p $PLANS \
    --npz \
    --c

if [ $? -eq 0 ]; then
    echo "Fold $FOLD COMPLETED at $(date)"
else
    echo "ERROR: Fold $FOLD FAILED — resume with --c if wall time hit"
    exit 1
fi
