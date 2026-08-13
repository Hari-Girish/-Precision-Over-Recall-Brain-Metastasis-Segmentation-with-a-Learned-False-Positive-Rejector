#!/bin/bash
#SBATCH --job-name=R37b_predict
#SBATCH --account=PDE0022
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=/fs/ess/PDE0022/hari/brats2026/pipelines/nnunet/logs/R37b_dicetopk_5000ep_real/prediction/predict_%j.log
#SBATCH --error=/fs/ess/PDE0022/hari/brats2026/pipelines/nnunet/logs/R37b_dicetopk_5000ep_real/prediction/predict_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=girish.17@buckeyemailosu.edu

##############################################################################
# R37b_dicetopk_5000ep_real — PREDICTION (5-fold ensemble)
# Trainer: nnUNetTrainerDiceTopK10Loss_5000epochs | Plans: nnUNetPlans | Folds: 0-4
##############################################################################

echo "=============================================="
echo "R37b_dicetopk_5000ep_real Prediction (5-fold ensemble)"
echo "Started at: $(date) | Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
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

VARIANTS_DIR=$(python -c "import nnunetv2, os; print(os.path.join(os.path.dirname(nnunetv2.__file__), 'training/nnUNetTrainer/variants/loss'))")
cp /fs/ess/PDE0022/hari/brats2026/pipelines/nnunet/custom_trainers/nnUNetTrainerDiceTopK10Loss_5000epochs.py "$VARIANTS_DIR/"

INPUT="/fs/ess/PDE0022/hari/brats2026/pipelines/nnunet/data/raw/Dataset501_BraTS_MET_2026_Training/imagesTs"
OUT_ROOT="/fs/ess/PDE0022/hari/brats2026/pipelines/nnunet/data/results/predictions/R37b_dicetopk_5000ep_real"
FINAL="${OUT_ROOT}/ensemble"
mkdir -p "$FINAL"

FOLD_DIRS=()
for FOLD in 0 1 2 3 4; do
    FOLD_OUT="${OUT_ROOT}/fold_${FOLD}"
    FOLD_DIRS+=("$FOLD_OUT")
    mkdir -p "$FOLD_OUT"
    echo "--- Predicting fold $FOLD ---"
    nnUNetv2_predict \
        -i "$INPUT" -o "$FOLD_OUT" \
        -d 501 -c 3d_fullres \
        -tr nnUNetTrainerDiceTopK10Loss_5000epochs -p nnUNetPlans \
        -f "$FOLD" \
        --save_probabilities --continue_prediction \
        -step_size 0.5
    [ $? -ne 0 ] && echo "ERROR: fold $FOLD failed" && exit 1
    echo "Fold $FOLD: $(find $FOLD_OUT -name '*.npz' | wc -l) npz files"
done

echo "--- Ensembling 5 folds -> $FINAL ---"
nnUNetv2_ensemble -i "${FOLD_DIRS[@]}" -o "$FINAL" -np 2

if [ $? -eq 0 ]; then
    N=$(find "$FINAL" -name "*.nii.gz" | wc -l)
    echo "PREDICTION COMPLETED at $(date) — $N files in $FINAL"
else
    echo "ERROR: Ensembling failed!" && exit 1
fi
